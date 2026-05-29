#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// 3-elevator, 4-floor building as a discrete-event system — this one
// blends both halves: a SIMULATION of passenger arrivals layered with
// an MDP-based CONTROL policy that dispatches cars. Both halves run on
// the same tick clock.
//
// ARCHITECTURE: HOW DO ELEVATORS FIT THE STATIONARY/MOVING MODEL?
// ----------------------------------------------------------------
//
// The framework's central distinction is "stationary vs moving":
//   - STATIONARY entities live in the network. They have identity, hold
//     state across ticks, run a `runTimeStep(stepSize, t)`, and decide
//     how to interact with whatever is in their inbox / immediate
//     neighbourhood. Stations.
//   - MOVING entities are the things that flow. They carry timestamped
//     state across stations but the framework does not give them a
//     `runTimeStep` — the station they are currently *in* is responsible
//     for them while they are there.
//
// For a building, the natural temptation is to make elevator cars the
// "moving entities" and floors the "stations". But cars need to:
//   1. Hold passengers across ticks (state).
//   2. Make routing decisions every tick (logic).
//   3. Carry per-car continuous position (`currentFloor`).
//   4. Coordinate with other cars (read peers' state).
//
// Moving entities cannot do (2) or (4) in this framework — they have no
// `runTimeStep`. So the clean answer is the inverse: the cars themselves
// are STATIONARY entities (they are nodes in the station graph that hold
// state and run their own logic), and the only thing that flows between
// stations is the people. Floors are also stationary — each floor is one
// node that holds an up-queue and a down-queue and exposes `hasCall()` /
// `takeFromQueue()` to neighbouring elevators.
//
// So we have two stationary entity *kinds* that interact heavily:
//
//   Floor[f] -- Elevator[k] interaction at every tick where elevator k is
//   serving floor f. The interaction is:
//     - elevator boards from floor.upQueue / downQueue (mutates floor)
//     - elevator delivers to floor.exitedHere    (mutates floor)
//
// Cars and floors live together in a single "Building" station graph. The
// framework's per-tick loop calls `runTimeStep` on both kinds of stations,
// and the elevators read floor queues directly (they hold a reference to
// the floor array). This is allowed: two stationary entities are permitted
// to share state, the framework only forbids "global views". Each elevator
// can only see the floors it is asked to serve, not other elevators —
// except via the explicit `Coordinator` object that mediates dispatch.
//
// Final entity decomposition:
//
//   PersonSource ─▶ Floor[from] ─▶ Elevator[k] ─▶ Floor[to] ─▶ ExitSink
//   ───────────    ────────────    ────────────   ──────────    ────────
//   stationary      stationary       stationary    stationary    stationary
//
//   Person is the only moving entity in the system.
//
// Per-tick order (deterministic):
//   1. PersonSource emits any persons whose arrivalTime ≤ now.
//   2. Coordinator (if coordinated mode) snapshots active claims.
//   3. Floors and ExitSink run in shuffled order (no inter-floor coupling
//      to make ordering matter — the shuffle is purely cosmetic for them).
//   4. Elevators run in index order. Elevators MUST run in deterministic
//      order so coordinated picks are reproducible (lower-id wins ties).
//
// DISPATCH MODES (env DISPATCH={uncoordinated,coordinated,coordinated-pickup}):
//
//   * `uncoordinated`: every elevator runs SCAN/LOOK independently. Two
//     elevators may pick the same outstanding call; the first to arrive
//     boards everyone and the second arrives to an empty queue — wasted
//     stop, wasted travel time.
//
//   * `coordinated`: a `Coordinator` object holds a per-tick set of
//     "claimed" (floor, direction) pairs. Each elevator that needs to
//     pick a new target consults the coordinator and excludes already-
//     claimed pairs. Lower-id elevator claims first when two are
//     equidistant. Implements the user-stated rules:
//       (a) "if a car is already going to a floor, don't send a second"
//       (b) "skip a floor going up so other cars can stop at that floor"
//
//   * `coordinated-pickup`: `coordinated` plus fullness-aware opportunistic
//     mid-flight pickups. While MOVING in direction d, an elevator that
//     has spare capacity AND will pass an integer floor with a same-d
//     call AND that (floor, d) pair is unclaimed by any other elevator
//     stops there briefly to pick up. Implements the user-stated rule:
//       (c) "elevator knows if it's full and decides whether to pit-stop
//            at a passing floor for a pure pickup, or just continue to
//            its next destination"
//
// VALIDATION:
//   - SimPy continuous-time FEL reference: `external-references/elevator/`
//   - Per-tick invariant checker: `test/elevator-invariants-test.ts` runs
//     22 configurations, validates conservation of people, capacity bound,
//     state-machine consistency, position bounds, monotonic timestamps.
//   - Cross-mode comparison: `runners/compare-elevator-dispatch.ts`.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {fisherYatesShuffle} from './general/general';
import {mulberry32, withSeed} from './general/prng';
import {TimeSteppedStation as Station} from './general/time-stepped-station';

// -----------------------------------------------------------------------------
// Person — the only moving entity.
// -----------------------------------------------------------------------------

class Person {
  arrivalTime: number = -1;
  boardTime:   number = -1;
  exitTime:    number = -1;
  constructor(public id: number, public fromFloor: number, public toFloor: number) {}
  direction(): 'up' | 'down' { return this.toFloor > this.fromFloor ? 'up' : 'down'; }
}

// -----------------------------------------------------------------------------
// PersonSource — runs through a precomputed schedule.
// -----------------------------------------------------------------------------

interface ScheduledArrival {
  t: number;             // arrival time in seconds
  fromFloor: number;     // 1..floors
  toFloor: number;       // 1..floors, != fromFloor
}

class PersonSource extends Station {
  private idx = 0;
  private nextId = 0;
  constructor(id: string, public schedule: ReadonlyArray<ScheduledArrival>, public floors: Floor[]) {
    super(id);
  }
  runTimeStep(stepSize: number, t: number): void {
    const now = t * stepSize;
    while (this.idx < this.schedule.length && this.schedule[this.idx].t <= now) {
      const a = this.schedule[this.idx++];
      const p = new Person(this.nextId++, a.fromFloor, a.toFloor);
      p.arrivalTime = a.t;
      this.floors[a.fromFloor - 1].addPerson(p);
    }
  }
  isDone(): boolean { return this.idx >= this.schedule.length; }
}

// -----------------------------------------------------------------------------
// Floor — holds two FIFO queues of waiting people (up and down).
// -----------------------------------------------------------------------------

class Floor extends Station {
  upQueue:   Person[] = [];
  downQueue: Person[] = [];
  exitedHere: Person[] = [];      // deboarded this tick — drained to sink
  constructor(id: string, public floorNumber: number, public sink: ExitSink) {
    super(id);
  }
  addPerson(p: Person): void {
    if (p.toFloor > p.fromFloor) this.upQueue.push(p);
    else this.downQueue.push(p);
  }
  hasCall(direction: 'up' | 'down'): boolean {
    return direction === 'up' ? this.upQueue.length > 0 : this.downQueue.length > 0;
  }
  takeFromQueue(direction: 'up' | 'down', cap: number): Person[] {
    const q = direction === 'up' ? this.upQueue : this.downQueue;
    return q.splice(0, cap);
  }
  runTimeStep(stepSize: number, t: number): void {
    if (this.exitedHere.length > 0) {
      for (const p of this.exitedHere) this.sink.collect(p);
      this.exitedHere = [];
    }
  }
}

// -----------------------------------------------------------------------------
// Elevator — the most complex stationary entity.
// -----------------------------------------------------------------------------

type ElevatorState = 'IDLE' | 'MOVING' | 'SERVING';

/**
 * Per-tick claim registry. Maps (floor, direction) → elevator-idx that
 * claimed it. Allows callers to ask "is this claimed by anyone else?" so
 * an elevator considering an opportunistic pit-stop at a floor it itself
 * already owns is not blocked by its own claim. In `uncoordinated` mode
 * the coordinator is just `null` and every elevator picks targets
 * independently.
 */
class Coordinator {
  private claimedBy = new Map<string, number>();
  reset(): void { this.claimedBy.clear(); }
  static key(floor: number, dir: 'up' | 'down'): string { return `${floor}-${dir}`; }
  isClaimed(floor: number, dir: 'up' | 'down'): boolean {
    return this.claimedBy.has(Coordinator.key(floor, dir));
  }
  isClaimedByOther(floor: number, dir: 'up' | 'down', myIdx: number): boolean {
    const owner = this.claimedBy.get(Coordinator.key(floor, dir));
    return owner !== undefined && owner !== myIdx;
  }
  claim(floor: number, dir: 'up' | 'down', byIdx: number): void {
    const key = Coordinator.key(floor, dir);
    if (!this.claimedBy.has(key)) this.claimedBy.set(key, byIdx);
  }
  /**
   * Seed the coordinator with each elevator's current trajectory: a MOVING
   * or SERVING elevator's (targetFloor, direction) pair is already
   * committed and should not be re-picked by anyone else.
   */
  seedFromActive(elevators: Elevator[]): void {
    this.reset();
    for (const e of elevators) {
      if ((e.state === 'MOVING' || e.state === 'SERVING') &&
          (e.direction === 'up' || e.direction === 'down')) {
        this.claim(e.targetFloor, e.direction, e.idx);
      }
    }
  }
}

class Elevator extends Station {
  state: ElevatorState = 'IDLE';
  currentFloor: number;            // continuous position in [1, nFloors]
  targetFloor: number;
  direction: 'up' | 'down' | 'idle' = 'idle';
  passengers: Person[] = [];
  serviceRemaining = 0;            // seconds left in SERVING
  coordinator: Coordinator | null = null;
  opportunisticPickups = false;    // true in 'coordinated-pickup' mode
  constructor(
    id: string,
    public idx: number,            // numeric index for tie-break
    public floors: Floor[],
    public capacity: number,
    public floorTravelTime: number,// seconds per floor
    public serviceTime: number,    // seconds to open doors + (de)board
    startFloor = 1,
  ) {
    super(id);
    this.currentFloor = startFloor;
    this.targetFloor = startFloor;
  }
  /** Floors per second when moving. */
  private get speed(): number { return 1.0 / this.floorTravelTime; }

  /** Capacity awareness — exposed for the opportunistic-pickup dispatch mode. */
  isFull(): boolean { return this.passengers.length >= this.capacity; }
  spareCapacity(): number { return this.capacity - this.passengers.length; }
  loadFraction(): number { return this.passengers.length / this.capacity; }

  /**
   * Look for the closest integer floor strictly between currentFloor and
   * newFloor (inclusive of newFloor) that:
   *   - has a same-direction call queued, AND
   *   - is unclaimed by any other elevator (or claimed by *this* elevator,
   *     since we are still committed to it; but we'd be diverting), AND
   *   - we have spare capacity to pick up at least one person there.
   * Returns null if no such floor exists. Used by the opportunistic-pickup
   * mode in MOVING state to decide whether to pit-stop on the way to the
   * declared target.
   *
   * The check is "we will *cross or land on* this floor on this tick", not
   * "this floor is anywhere ahead", because a more-distant floor will be
   * checked again on a future tick and we don't want to pre-empt the
   * existing target until it's strictly necessary.
   */
  private opportunisticPitStop(newFloor: number): number | null {
    if (!this.opportunisticPickups) return null;
    if (this.isFull()) return null;
    if (this.direction !== 'up' && this.direction !== 'down') return null;
    const dir: 'up' | 'down' = this.direction;
    // The range we'll cross or reach this tick. We're at exactly
    // currentFloor right now, and we'll be at newFloor at the end of the
    // tick. We want pit-stops STRICTLY between current and new (in the
    // direction of travel), since we just left or are still at currentFloor.
    const eps = 1e-9;
    let best = -1;
    for (const f of this.floors) {
      const F = f.floorNumber;
      if (Math.abs(F - this.currentFloor) < eps) continue;   // we're on it
      // Must be in (currentFloor, newFloor] in the direction of travel.
      if (dir === 'up') {
        if (!(F > this.currentFloor + eps && F <= newFloor + eps)) continue;
      } else {
        if (!(F < this.currentFloor - eps && F >= newFloor - eps)) continue;
      }
      if (!f.hasCall(dir)) continue;
      // Don't pit-stop at the declared target — the normal MOVING→SERVING
      // transition will handle this same floor.
      if (F === this.targetFloor) continue;
      if (this.coordinator && this.coordinator.isClaimedByOther(F, dir, this.idx)) continue;
      // Closest in our direction: smaller F first when going up; larger F
      // first when going down.
      if (best < 0 ||
          (dir === 'up'   && F < best) ||
          (dir === 'down' && F > best)) {
        best = F;
      }
    }
    return best > 0 ? best : null;
  }

  runTimeStep(stepSize: number, t: number): void {
    const now = t * stepSize;

    if (this.state === 'IDLE') {
      const next = this.pickNextTarget();
      if (next !== null) {
        this.targetFloor = next.floor;
        this.direction   = next.dir;
        this.state       = 'MOVING';
        this.coordinator?.claim(next.floor, next.dir, this.idx);
      }
    }

    if (this.state === 'MOVING') {
      const sign = this.targetFloor > this.currentFloor ? 1 : -1;
      const remaining = this.targetFloor - this.currentFloor;
      const delta = this.speed * stepSize * sign;
      const newFloor = this.currentFloor + delta;
      const pit = this.opportunisticPitStop(newFloor);
      if (pit !== null) {
        // Pit-stop at an intermediate same-direction call.
        this.currentFloor     = pit;
        this.state            = 'SERVING';
        this.serviceRemaining = this.serviceTime;
        // Note: targetFloor is unchanged; we'll re-pick after this stop.
        // Add a claim so other elevators don't race here too. The claim
        // for the original target stays — it's still our intent.
        this.coordinator?.claim(pit, this.direction === 'idle' ? 'up' : this.direction, this.idx);
      } else if (Math.abs(delta) >= Math.abs(remaining) - 1e-12) {
        this.currentFloor    = this.targetFloor;
        this.state           = 'SERVING';
        this.serviceRemaining = this.serviceTime;
      } else {
        this.currentFloor = newFloor;
      }
    }

    if (this.state === 'SERVING') {
      // Door-open / door-close model (matches the SimPy reference):
      // deboard + board happens ONCE on the first SERVING tick, then we
      // dwell for serviceTime seconds with the doors closed regardless of
      // people arriving on this floor in the meantime.
      const f = this.floors[Math.round(this.currentFloor) - 1];
      if (this.serviceRemaining === this.serviceTime) {
        const deboard = this.passengers.filter(p => p.toFloor === f.floorNumber);
        this.passengers = this.passengers.filter(p => p.toFloor !== f.floorNumber);
        for (const p of deboard) {
          p.exitTime = now;
          f.exitedHere.push(p);
        }
        const dir = this.direction === 'idle' ? 'up' : this.direction;
        const slots = this.capacity - this.passengers.length;
        const boarding = f.takeFromQueue(dir, slots);
        for (const p of boarding) {
          p.boardTime = now;
          this.passengers.push(p);
        }
      }
      this.serviceRemaining -= stepSize;
      if (this.serviceRemaining <= 0) {
        const next = this.pickNextTarget();
        if (next === null) {
          this.state = 'IDLE';
          this.direction = 'idle';
        } else {
          this.targetFloor = next.floor;
          this.direction   = next.dir;
          this.state       = 'MOVING';
          this.coordinator?.claim(next.floor, next.dir, this.idx);
        }
      }
    }
  }

  /**
   * SCAN/LOOK dispatch. Picks the next floor to visit:
   *   1. If we have passengers, prefer the closest passenger destination in
   *      the current direction.
   *   2. Else if any same-direction floor call lies further in our direction,
   *      pick the closest one.
   *   3. Else reverse direction and try (1)+(2) again.
   *   4. Else (truly idle) scan all floors for any call and pick the closest.
   *
   * In coordinated mode, the candidate (floor, dir) is rejected if the
   * coordinator has already claimed it (some other elevator is committing
   * to serve that exact call). Passenger destinations are NEVER rejected
   * — passengers belong to *this* elevator and must be delivered.
   */
  private pickNextTarget(): {floor: number; dir: 'up' | 'down'} | null {
    const cur = this.currentFloor;
    const coord = this.coordinator;

    const tryDir = (dir: 'up' | 'down'): {floor: number; dir: 'up' | 'down'} | null => {
      const better = (a: number, b: number) =>
        dir === 'up' ? a < b : a > b;
      let best = -1;
      let fromPassenger = false;
      for (const p of this.passengers) {
        if ((dir === 'up' && p.toFloor > cur) || (dir === 'down' && p.toFloor < cur)) {
          if (best < 0 || better(p.toFloor, best)) { best = p.toFloor; fromPassenger = true; }
        }
      }
      for (const f of this.floors) {
        if ((dir === 'up' && f.floorNumber > cur) || (dir === 'down' && f.floorNumber < cur)) {
          if (f.hasCall(dir)) {
            // Coordinated mode: skip calls already claimed by another
            // elevator UNLESS we have a passenger going there anyway
            // (then the stop is unavoidable).
            if (coord && coord.isClaimedByOther(f.floorNumber, dir, this.idx) &&
                !fromPassengerHere(this.passengers, f.floorNumber)) {
              continue;
            }
            if (best < 0 || better(f.floorNumber, best)) { best = f.floorNumber; fromPassenger = false; }
          }
        }
      }
      return best > 0 ? {floor: best, dir} : null;
    };

    // Prefer current direction.
    if (this.direction === 'up') {
      const t = tryDir('up');     if (t) return t;
      const u = tryDir('down');   if (u) return u;
    } else if (this.direction === 'down') {
      const t = tryDir('down');   if (t) return t;
      const u = tryDir('up');     if (u) return u;
    }
    // IDLE: any call, pick closest. Skip claims belonging to other elevators.
    let bestDist = Infinity, bestFloor = -1, bestDir: 'up' | 'down' = 'up';
    for (const f of this.floors) {
      const d = Math.abs(f.floorNumber - cur);
      const upClaimed   = coord?.isClaimedByOther(f.floorNumber, 'up', this.idx);
      const downClaimed = coord?.isClaimedByOther(f.floorNumber, 'down', this.idx);
      if (f.hasCall('up')   && !upClaimed   && d < bestDist) {
        bestDist = d; bestFloor = f.floorNumber; bestDir = 'up';
      }
      if (f.hasCall('down') && !downClaimed && d < bestDist) {
        bestDist = d; bestFloor = f.floorNumber; bestDir = 'down';
      }
    }
    return bestFloor > 0 ? {floor: bestFloor, dir: bestDir} : null;
  }
}

function fromPassengerHere(passengers: Person[], floor: number): boolean {
  for (const p of passengers) if (p.toFloor === floor) return true;
  return false;
}

// -----------------------------------------------------------------------------
// ExitSink — collects completed people.
// -----------------------------------------------------------------------------

class ExitSink extends Station {
  collected: Person[] = [];
  constructor(id: string) { super(id); }
  collect(p: Person): void { this.collected.push(p); }
  runTimeStep(_stepSize: number, _t: number): void { /* nothing */ }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

export interface ElevatorConfig {
  nFloors: number;
  nElevators: number;
  capacity: number;
  floorTravelTime: number;   // seconds per floor
  serviceTime: number;       // seconds per stop (door open + load + close)
  arrivalRate: number;       // people per second
  simT: number;              // total simulated time, seconds
  stepSize: number;          // tick size, seconds
  seed: number;
  dispatchMode?: 'uncoordinated' | 'coordinated' | 'coordinated-pickup';
}

export interface ElevatorResult {
  config: ElevatorConfig;
  schedule: ScheduledArrival[];
  people: Array<Pick<Person, 'id' | 'fromFloor' | 'toFloor' | 'arrivalTime' | 'boardTime' | 'exitTime'>>;
  aggregates: {
    n: number;
    nServed: number;
    meanWait: number;
    meanTravel: number;
    meanTotal: number;
    p95Wait: number;
    p95Total: number;
  };
}

/**
 * Build a deterministic Poisson arrival schedule from a seeded PRNG. Inter-
 * arrival times are exponential with mean 1/lambda. fromFloor uniform 1..F,
 * toFloor uniform among the remaining F-1 floors.
 */
export function buildSchedule(cfg: ElevatorConfig): ScheduledArrival[] {
  return withSeed(cfg.seed, () => {
    const rng = mulberry32(cfg.seed);
    const out: ScheduledArrival[] = [];
    let t = 0;
    while (true) {
      // Inter-arrival ~ Exp(lambda): -ln(1-U)/lambda. Avoid log(0) at U=0.
      const u = 1 - rng();
      t += -Math.log(u) / cfg.arrivalRate;
      if (t > cfg.simT) break;
      const fromFloor = 1 + Math.floor(rng() * cfg.nFloors);
      let toFloor: number;
      do {
        toFloor = 1 + Math.floor(rng() * cfg.nFloors);
      } while (toFloor === fromFloor);
      out.push({t, fromFloor, toFloor});
    }
    return out;
  });
}

/**
 * The Building bundles together every stationary entity and exposes a
 * `tickOnce(t)` so external drivers (notably the invariants test suite)
 * can run the simulation tick by tick and inspect state in between.
 *
 * Public fields are intentionally accessible — the test driver reads them
 * to verify conservation, capacity, etc. Production code should use
 * `runElevator(cfg, schedule)`.
 */
export class Building {
  source: PersonSource;
  floors: Floor[];
  elevators: Elevator[];
  sink: ExitSink;
  coordinator: Coordinator | null;
  config: ElevatorConfig;
  constructor(cfg: ElevatorConfig, schedule: ReadonlyArray<ScheduledArrival>) {
    this.config = cfg;
    this.sink = new ExitSink('sink');
    this.floors = [];
    for (let i = 1; i <= cfg.nFloors; i++) {
      this.floors.push(new Floor(`F${i}`, i, this.sink));
    }
    this.source = new PersonSource('src', schedule, this.floors);
    this.elevators = [];
    for (let k = 0; k < cfg.nElevators; k++) {
      this.elevators.push(new Elevator(
        `E${k}`, k, this.floors, cfg.capacity,
        cfg.floorTravelTime, cfg.serviceTime,
        1 + (k % cfg.nFloors),
      ));
    }
    const dispatchMode = cfg.dispatchMode ?? 'uncoordinated';
    this.coordinator = dispatchMode !== 'uncoordinated' ? new Coordinator() : null;
    if (this.coordinator) {
      for (const e of this.elevators) {
        e.coordinator = this.coordinator;
        if (dispatchMode === 'coordinated-pickup') e.opportunisticPickups = true;
      }
    }
  }
  tickOnce(t: number): void {
    const dt = this.config.stepSize;
    // 1. Source first: delivers any newly-arrived persons to floors.
    this.source.runTimeStep(dt, t);
    // 2. Coordinator snapshot: lock in current active trajectories so new
    //    picks this tick can avoid stepping on them.
    if (this.coordinator) this.coordinator.seedFromActive(this.elevators);
    // 3. Passive stations (floors + sink) run in shuffled order. They have
    //    no inter-station coupling so the order is purely cosmetic — but
    //    the shuffle is the framework's way of certifying that.
    const passive: Station[] = [...this.floors, this.sink];
    for (const _ of fisherYatesShuffle(passive)) { /* drain generator */ }
    for (const s of passive) s.runTimeStep(dt, t);
    // 4. Elevators run in index order — required for deterministic
    //    coordinated picks.
    for (const e of this.elevators) e.runTimeStep(dt, t);
  }
  totalEmitted(): number   { return (this.source as any).nextId; }
  isComplete(scheduleLen: number): boolean { return this.sink.collected.length === scheduleLen; }
}

export function runElevator(
  cfg: ElevatorConfig,
  schedule: ReadonlyArray<ScheduledArrival>,
  onTick?: (b: Building, t: number) => void,
): ElevatorResult {
  const b = new Building(cfg, schedule);
  const N = Math.round(cfg.simT / cfg.stepSize);
  for (let t = 0; t < N; t++) {
    b.tickOnce(t);
    if (onTick) onTick(b, t);
  }
  // Drain anyone still in elevators or queues by extending the run a little.
  for (let extra = 0; extra < N && !b.isComplete(schedule.length); extra++) {
    b.tickOnce(N + extra);
    if (onTick) onTick(b, N + extra);
  }
  const sink = b.sink;

  const served = sink.collected.filter(p => p.exitTime > 0);
  const waits = served.map(p => p.boardTime - p.arrivalTime);
  const travels = served.map(p => p.exitTime - p.boardTime);
  const totals = served.map(p => p.exitTime - p.arrivalTime);
  const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
  const p95 = (xs: number[]) => {
    if (xs.length === 0) return 0;
    const sorted = [...xs].sort((a, b) => a - b);
    return sorted[Math.floor(0.95 * (sorted.length - 1))];
  };
  return {
    config: cfg,
    schedule: [...schedule],
    people: served.map(p => ({
      id: p.id, fromFloor: p.fromFloor, toFloor: p.toFloor,
      arrivalTime: p.arrivalTime, boardTime: p.boardTime, exitTime: p.exitTime,
    })),
    aggregates: {
      n: schedule.length,
      nServed: served.length,
      meanWait: mean(waits),
      meanTravel: mean(travels),
      meanTotal: mean(totals),
      p95Wait: p95(waits),
      p95Total: p95(totals),
    },
  };
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

async function main() {
  const baseCfg: Omit<ElevatorConfig, 'dispatchMode'> = {
    nFloors:        Number(process.env.FLOORS    ?? 4),
    nElevators:     Number(process.env.ELEVATORS ?? 3),
    capacity:       Number(process.env.CAPACITY  ?? 8),
    floorTravelTime: Number(process.env.TRAVEL_T ?? 4),
    serviceTime:    Number(process.env.SERVICE_T ?? 3),
    arrivalRate:    Number(process.env.LAMBDA    ?? 0.2),
    simT:           Number(process.env.SIM_T     ?? 1800),
    stepSize:       Number(process.env.STEPSIZE  ?? 0.5),
    seed:           Number(process.env.SEED      ?? 1),
  };
  console.log(`# Elevator simulation`);
  console.log(`#   ${baseCfg.nFloors} floors, ${baseCfg.nElevators} elevators, capacity ${baseCfg.capacity}`);
  console.log(`#   travel=${baseCfg.floorTravelTime}s/floor, service=${baseCfg.serviceTime}s, λ=${baseCfg.arrivalRate}/s`);
  console.log(`#   simT=${baseCfg.simT}s, dt=${baseCfg.stepSize}s, seed=${baseCfg.seed}`);

  const schedule = buildSchedule({...baseCfg, dispatchMode: 'uncoordinated'});
  console.log(`#   schedule has ${schedule.length} arrivals`);

  const modes: Array<ElevatorConfig['dispatchMode']> = ['uncoordinated', 'coordinated', 'coordinated-pickup'];
  const dispatchEnv = process.env.DISPATCH;
  const runs = dispatchEnv ? [dispatchEnv as ElevatorConfig['dispatchMode']] : modes;
  const animateMode = process.env.ANIMATE === '1'
    ? (process.env.ANIMATE_DISPATCH ?? 'coordinated-pickup') as ElevatorConfig['dispatchMode']
    : null;
  const animationSeries = {t: [] as number[], waiting: [] as number[],
                           inCar: [] as number[], served: [] as number[]};
  let animRecorder: any = null;
  let animBuilder: any = null;

  const results: ElevatorResult[] = [];
  for (const mode of runs) {
    const cfg: ElevatorConfig = {...baseCfg, dispatchMode: mode};
    const t0 = Date.now();
    const recordThis = animateMode !== null && mode === animateMode;
    let onTick: ((b: Building, t: number) => void) | undefined;
    if (recordThis) {
      const outDir = path.join(__dirname, '..', '..', 'out');
      const {FrameRecorder} = await import('./animation/frame-recorder');
      const {STAGE_W, STAGE_H, buildElevatorFrame} = await import('./animation/scenes/elevator-scene');
      animBuilder = buildElevatorFrame;
      animRecorder = new FrameRecorder({
        framesPath: path.join(outDir, 'elevator.frames.jsonl'),
        htmlPath:   path.join(outDir, 'elevator.html'),
        width: STAGE_W, height: STAGE_H, fps: 30,
        title: `Elevator simulation — ${mode}`,
        subtitle: `${baseCfg.nFloors} floors  ${baseCfg.nElevators} cars  cap=${baseCfg.capacity}  ` +
                  `λ=${baseCfg.arrivalRate}/s  travel=${baseCfg.floorTravelTime}s  service=${baseCfg.serviceTime}s  ` +
                  `dt=${baseCfg.stepSize}s`,
        liveTickLine: true,
        recordEveryTicks: Math.max(1, Math.floor(baseCfg.simT / baseCfg.stepSize / 600)),
      });
      onTick = (b, tk) => {
        const t = tk * baseCfg.stepSize;
        const wait = b.floors.reduce((s, f) => s + f.upQueue.length + f.downQueue.length, 0);
        const inCar = b.elevators.reduce((s, e) => s + e.passengers.length, 0);
        const served = b.sink.collected.filter(p => p.exitTime > 0).length;
        animationSeries.t.push(t);
        animationSeries.waiting.push(wait);
        animationSeries.inCar.push(inCar);
        animationSeries.served.push(served);
        animRecorder.frame(t, tk, () => animBuilder(t, tk, b));
      };
    }
    const result = runElevator(cfg, schedule, onTick);
    const ms = Date.now() - t0;
    results.push(result);
    const a = result.aggregates;
    console.log('');
    console.log(`# dispatchMode = ${mode!.padEnd(20)} (${ms} ms)`);
    console.log(`#   served ${a.nServed}/${a.n} people`);
    console.log(`#   meanWait   = ${a.meanWait.toFixed(2)} s     p95Wait  = ${a.p95Wait.toFixed(2)} s`);
    console.log(`#   meanTravel = ${a.meanTravel.toFixed(2)} s`);
    console.log(`#   meanTotal  = ${a.meanTotal.toFixed(2)} s     p95Total = ${a.p95Total.toFixed(2)} s`);
  }

  // Show pairwise improvements when multiple modes ran.
  if (results.length >= 2) {
    console.log('');
    console.log(`# pairwise improvements (vs uncoordinated baseline, results[0]):`);
    const baseline = results[0].aggregates;
    for (let i = 1; i < results.length; i++) {
      const a = results[i].aggregates;
      const m = results[i].config.dispatchMode;
      const dWait  = ((a.meanWait/baseline.meanWait - 1)*100).toFixed(1);
      const dP95   = ((a.p95Wait/baseline.p95Wait - 1)*100).toFixed(1);
      const dTotal = ((a.meanTotal/baseline.meanTotal - 1)*100).toFixed(1);
      console.log(`#   ${m!.padEnd(20)} meanWait ${dWait}% , p95Wait ${dP95}% , meanTotal ${dTotal}%`);
    }
  }

  // Write the FIRST result (uncoordinated by default, or whichever mode the
  // user selected via DISPATCH=) so the SimPy validator has a known input.
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'elevator-framework.json');
  fs.writeFileSync(outPath, JSON.stringify(results[0]));
  console.log(`# wrote ${outPath}`);

  // Also dump all modes side-by-side for downstream analysis.
  if (results.length >= 2) {
    const cmpPath = path.join(outDir, 'elevator-dispatch-comparison.json');
    fs.writeFileSync(cmpPath, JSON.stringify({
      schedule, runs: results.map(r => ({
        dispatchMode: r.config.dispatchMode,
        aggregates: r.aggregates,
        people: r.people,
      })),
    }));
    console.log(`# wrote ${cmpPath}`);
  }

  if (animRecorder) {
    const {buildElevatorChart} = await import('./animation/scenes/elevator-scene');
    animRecorder.setCharts([buildElevatorChart(animationSeries)]);
    await animRecorder.finish();
    console.log(`# wrote ${path.join(outDir, 'elevator.html')} (${animRecorder.getFrameCount()} frames)`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
