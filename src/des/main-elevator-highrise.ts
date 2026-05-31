#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_elevator_highrise.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-elevator-highrise.rs   (fn main)
// 1:1 file move. 50-floor, 6-shaft elevator model exploring dispatch policies;
// emits one HTML animation with a policy selector.
//
// Conversion notes (file-specific):
//   - HighrisePolicy string union -> enum.
//   - mulberry32 / withSeed PRNG -> SeededRandom (shared::capabilities).
//   - LARGE file: station classes -> struct + impl trait; MDP valueIteration ->
//     use crate::des::general::value_iteration.
//   - exported types make this both a model module and a bin; fs HTML write ->
//     std::fs; top-level run -> fn main.
// =============================================================================

// =============================================================================
// High-rise elevator model.
//
// This is intentionally separate from main-elevator.ts. The smaller model stays
// as a reference implementation and invariant target; this model explores
// dispatch-policy tradeoffs for a 50-floor, 6-shaft building and emits one HTML
// animation with a policy selector.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {Animation, ChartSpec, Frame, Shape} from './animation/types';
import {buildHTMLSet, AnimationVariant} from './animation/html-player';
import {mulberry32, withSeed} from './general/prng';
import {TimeSteppedStation} from './general/time-stepped-station';
import {SmartMovable} from './general/des-base/smart-movable';
import {MDPSpec, valueIteration} from './general/value-iteration';

export type HighrisePolicy =
  | 'fewest-stops'
  | 'lowest-total-time'
  | 'energy-efficient'
  | 'center-preposition'
  | 'zoned-service'
  | 'mdp-call-only'
  | 'mdp-tuned';

export type DecisionAuthority = 'central' | 'decentralized' | 'hybrid';
export type MDPObservability = 'call-only' | 'destination-dispatch';

export const HIGHRISE_POLICIES: HighrisePolicy[] = [
  'fewest-stops',
  'lowest-total-time',
  'energy-efficient',
  'center-preposition',
  'zoned-service',
  'mdp-call-only',
  'mdp-tuned',
];

export const DECISION_AUTHORITIES: DecisionAuthority[] = [
  'central',
  'decentralized',
  'hybrid',
];

export interface HighriseElevatorConfig {
  nFloors: number;          // floors are 0..nFloors-1
  nElevators: number;
  capacity: number;
  floorTravelTime: number;  // seconds per floor
  serviceTime: number;      // seconds per stop
  arrivalRate: number;      // people per second while source is active
  simT: number;             // source-active horizon
  drainT: number;           // extra seconds to clear queued passengers
  stepSize: number;
  seed: number;
  localSensorRadius: number;
  urgentWaitThreshold: number;
}

export interface HighrisePassengerSnapshot {
  id: number;
  fromFloor: number;
  toFloor: number;
  arrivalTime: number;
  boardTime: number;
  exitTime: number;
}

export interface HighriseElevatorResult {
  policy: HighrisePolicy;
  authority: DecisionAuthority;
  config: HighriseElevatorConfig;
  schedule: ScheduledArrival[];
  people: HighrisePassengerSnapshot[];
  aggregates: HighriseAggregates;
  mdpTuning?: MDPDispatchTuningSummary;
  mdpRun?: MDPRunDiagnostics;
  marginalVsLowestTime?: MarginalComparison;
}

export interface HighriseAggregates {
  n: number;
  nServed: number;
  meanWait: number;
  meanTravel: number;
  meanTotal: number;
  p95Wait: number;
  p95Total: number;
  totalStops: number;
  totalDistanceFloors: number;
  totalEnergy: number;
  timedOut: number;
}

interface ScheduledArrival {
  t: number;
  fromFloor: number;
  toFloor: number;
}

type Dir = -1 | 0 | 1;
type CarState = 'idle' | 'moving' | 'serving' | 'prepositioning';
type TargetReason = 'pickup' | 'dropoff' | 'home';
type DecisionSource = 'central' | 'local' | 'hybrid' | 'none';

interface DispatchScoreWeights {
  distance: number;
  trip: number;
  queue: number;
  wait: number;
  sameDirection: number;
  destinationGroup: number;
}

interface PickupFeatures {
  distance: number;
  oldestWait: number;
  queueLen: number;
  trip: number;
  sameSide: number;
  maxGroup: number;
}

interface MDPActionProfile {
  label: string;
  weights: DispatchScoreWeights;
}

interface MDPDispatchTuning {
  observability: MDPObservability;
  numStates: number;
  actions: MDPActionProfile[];
  policy: Int32Array;
  V: Float64Array;
  gamma: number;
  iterations: number;
  finalDelta: number;
  learnedWeights: DispatchScoreWeights;
  stateLabels: string[];
  actionLabels: string[];
}

export interface MDPDispatchTuningSummary {
  observability: MDPObservability;
  numStates: number;
  gamma: number;
  iterations: number;
  finalDelta: number;
  learnedWeights: DispatchScoreWeights;
  statePolicy: Array<{state: string; action: string}>;
}

export interface MDPRunDiagnostics {
  totalDecisions: number;
  actionCounts: Array<{action: string; count: number; share: number}>;
  topStates: Array<{state: string; action: string; count: number}>;
  marginals: Array<{
    variable: string;
    bins: Array<{bin: string; count: number; dominantAction: string; share: number}>;
  }>;
}

export interface MarginalComparison {
  baselinePolicy: HighrisePolicy;
  baselineAuthority: DecisionAuthority;
  meanWaitDelta: number;
  meanTotalDelta: number;
  stopsDelta: number;
  energyDelta: number;
}

interface MDPDecisionLogEntry {
  stateId: number;
  state: string;
  action: string;
  bins: Record<string, string>;
}

interface HighriseRunOptions {
  authority?: DecisionAuthority;
  recordEveryTicks?: number;
  mdpTuning?: MDPDispatchTuning;
}

class HighrisePassenger {
  boardTime = -1;
  exitTime = -1;

  constructor(
    readonly id: number,
    readonly fromFloor: number,
    readonly toFloor: number,
    readonly arrivalTime: number,
  ) {}

  direction(): Dir {
    return sign(this.toFloor - this.fromFloor);
  }
}

class ElevatorCar extends SmartMovable {
  currentFloor: number;
  targetFloor: number | null = null;
  targetReason: TargetReason | null = null;
  pickupDirection: Dir = 0;
  direction: Dir = 0;
  state: CarState = 'idle';
  passengers: HighrisePassenger[] = [];
  serviceRemaining = 0;
  decisionSource: DecisionSource = 'none';
  stops = 0;
  starts = 0;
  distanceFloors = 0;
  energy = 0;
  allowedFloors: Set<number> | null = null;

  constructor(readonly idx: number, startFloor: number, readonly capacity: number) {
    super(`E${idx}`);
    this.currentFloor = startFloor;
  }

  runTimeStep(): void {}

  spareCapacity(): number {
    return this.capacity - this.passengers.length;
  }

  isFull(): boolean {
    return this.spareCapacity() <= 0;
  }
}

interface FloorQueues {
  up: HighrisePassenger[];
  down: HighrisePassenger[];
}

export class HighriseBuilding extends TimeSteppedStation {
  readonly floors: FloorQueues[];
  readonly elevators: ElevatorCar[];
  readonly completed: HighrisePassenger[] = [];
  readonly people: HighrisePassenger[] = [];
  readonly mdpDecisionLog: MDPDecisionLogEntry[] = [];
  private nextArrivalIndex = 0;

  constructor(
    readonly config: HighriseElevatorConfig,
    readonly policy: HighrisePolicy,
    readonly schedule: readonly ScheduledArrival[],
    readonly authority: DecisionAuthority = 'central',
    readonly mdpTuning?: MDPDispatchTuning,
  ) {
    super(`highrise-${policy}-${authority}`);
    this.floors = Array.from({length: config.nFloors}, () => ({up: [], down: []}));
    this.elevators = Array.from({length: config.nElevators}, (_, i) => {
      const start = Math.round((config.nFloors - 1) * (i + 1) / (config.nElevators + 1));
      const car = new ElevatorCar(i, start, config.capacity);
      if (policy === 'zoned-service') car.allowedFloors = allowedFloorsFor(i, config.nFloors);
      return car;
    });
  }

  runTimeStep(_stepSize: number, tick: number): void {
    const now = tick * this.config.stepSize;
    this.emitArrivals(now);
    this.advanceCars(now);
    this.makeDispatchDecisions(now);
  }

  pendingPassengerCount(): number {
    return this.floors.reduce((s, f) => s + f.up.length + f.down.length, 0);
  }

  inCarCount(): number {
    return this.elevators.reduce((s, e) => s + e.passengers.length, 0);
  }

  allArrivalsEmitted(): boolean {
    return this.nextArrivalIndex >= this.schedule.length;
  }

  isDrained(): boolean {
    return this.allArrivalsEmitted() && this.pendingPassengerCount() === 0 && this.inCarCount() === 0;
  }

  totalEnergy(): number {
    return this.elevators.reduce((s, e) => s + e.energy, 0);
  }

  totalDistance(): number {
    return this.elevators.reduce((s, e) => s + e.distanceFloors, 0);
  }

  totalStops(): number {
    return this.elevators.reduce((s, e) => s + e.stops, 0);
  }

  private emitArrivals(now: number): void {
    while (this.nextArrivalIndex < this.schedule.length && this.schedule[this.nextArrivalIndex].t <= now) {
      const a = this.schedule[this.nextArrivalIndex++];
      const p = new HighrisePassenger(this.people.length, a.fromFloor, a.toFloor, a.t);
      this.people.push(p);
      const q = p.direction() > 0 ? this.floors[p.fromFloor].up : this.floors[p.fromFloor].down;
      q.push(p);
    }
  }

  private advanceCars(now: number): void {
    for (const car of this.elevators) {
      if (car.serviceRemaining > 0) {
        car.serviceRemaining = Math.max(0, car.serviceRemaining - this.config.stepSize);
        if (car.serviceRemaining > 0) continue;
        car.state = 'idle';
        car.targetFloor = null;
        car.targetReason = null;
        car.pickupDirection = 0;
      }

      if (car.targetFloor === null) continue;

      const delta = car.targetFloor - car.currentFloor;
      if (Math.abs(delta) < 1e-9) {
        this.serviceFloor(car, now);
        continue;
      }

      const dir = sign(delta);
      if (car.direction !== dir) {
        car.starts++;
        car.energy += 1.5;
      }
      car.direction = dir;
      car.state = car.targetReason === 'home' ? 'prepositioning' : 'moving';

      const stepFloors = this.config.stepSize / this.config.floorTravelTime;
      const move = Math.min(Math.abs(delta), stepFloors);
      car.currentFloor += dir * move;
      car.distanceFloors += move;
      car.energy += move * (1 + 0.055 * car.passengers.length);

      if (Math.abs(car.targetFloor - car.currentFloor) < 1e-9) {
        car.currentFloor = car.targetFloor;
        this.serviceFloor(car, now);
      }
    }
  }

  private serviceFloor(car: ElevatorCar, now: number): void {
    const floor = Math.round(car.currentFloor);
    let changed = false;

    const remaining: HighrisePassenger[] = [];
    for (const p of car.passengers) {
      if (p.toFloor === floor) {
        p.exitTime = now;
        this.completed.push(p);
        changed = true;
      } else {
        remaining.push(p);
      }
    }
    car.passengers = remaining;

    if (car.targetReason !== 'home') {
      const boarded = this.boardPassengers(car, floor, now);
      changed = changed || boarded > 0;
    }

    if (changed || car.targetReason === 'pickup' || car.targetReason === 'dropoff') {
      car.stops++;
      car.energy += 0.8;
      car.serviceRemaining = this.config.serviceTime;
      car.state = 'serving';
      car.direction = 0;
      return;
    }

    car.targetFloor = null;
    car.targetReason = null;
    car.pickupDirection = 0;
    car.state = 'idle';
    car.direction = 0;
  }

  private boardPassengers(car: ElevatorCar, floor: number, now: number): number {
    if (car.isFull()) return 0;
    const queues = this.floors[floor];
    const dir = this.boardingDirection(car, floor);
    if (dir === 0) return 0;
    const queue = dir > 0 ? queues.up : queues.down;
    if (queue.length === 0) return 0;

    const chosen: HighrisePassenger[] = [];
    const keep: HighrisePassenger[] = [];
    const directDest = this.policy === 'fewest-stops' ? dominantDestination(queue, car) : null;
    for (const p of queue) {
      if (car.spareCapacity() - chosen.length <= 0) {
        keep.push(p);
        continue;
      }
      if (!this.canCarServePassenger(car, p)) {
        keep.push(p);
        continue;
      }
      if (this.policy === 'fewest-stops' && directDest !== null && p.toFloor !== directDest) {
        keep.push(p);
        continue;
      }
      if (this.policy === 'energy-efficient' && car.passengers.length > 0 && wouldAddReverseStop(car, p)) {
        keep.push(p);
        continue;
      }
      chosen.push(p);
    }
    if (dir > 0) queues.up = keep;
    else queues.down = keep;

    for (const p of chosen) {
      p.boardTime = now;
      car.passengers.push(p);
    }
    car.energy += chosen.length * 0.08;
    return chosen.length;
  }

  private boardingDirection(car: ElevatorCar, floor: number): Dir {
    if (this.policy === 'fewest-stops' && car.passengers.length > 0) return 0;
    if (car.passengers.length > 0) {
      const next = chooseDropoff(car, this.policy);
      return next === null ? 0 : sign(next - floor);
    }
    if (car.pickupDirection !== 0) return car.pickupDirection;
    const up = this.floors[floor].up.length;
    const down = this.floors[floor].down.length;
    if (up === 0 && down === 0) return 0;
    if (up === 0) return -1;
    if (down === 0) return 1;
    const oldestUp = this.floors[floor].up[0]?.arrivalTime ?? Infinity;
    const oldestDown = this.floors[floor].down[0]?.arrivalTime ?? Infinity;
    return oldestUp <= oldestDown ? 1 : -1;
  }

  private makeDispatchDecisions(now: number): void {
    if (this.authority === 'central') {
      this.assignCentralCars(now);
    } else if (this.authority === 'decentralized') {
      this.assignAutonomousCars(now);
    } else {
      this.assignCentralCars(now, {urgentOnly: true, source: 'hybrid'});
      this.assignAutonomousCars(now, new Set<string>(), 'hybrid');
    }
  }

  private assignCentralCars(now: number, opts: {urgentOnly?: boolean; source?: DecisionSource} = {}): void {
    const claimed = new Set<string>();
    for (const car of this.elevators) {
      if (car.targetReason === 'pickup' && car.targetFloor !== null) {
        claimed.add(requestKey(car.targetFloor, car.pickupDirection));
      }
    }

    for (const car of this.elevators) {
      if (car.serviceRemaining > 0) continue;
      if (car.passengers.length > 0) {
        const dest = chooseDropoff(car, this.policy);
        if (dest !== null) this.setTarget(car, dest, 'dropoff', 0);
        car.decisionSource = opts.source ?? 'central';
        continue;
      }
      if (car.targetFloor !== null && car.targetReason !== 'home' &&
          Math.abs(car.currentFloor - car.targetFloor) > 1e-9) continue;

      const pickup = this.choosePickup(car, now, claimed, {urgentOnly: opts.urgentOnly});
      if (pickup) {
        claimed.add(requestKey(pickup.floor, pickup.dir));
        this.setTarget(car, pickup.floor, 'pickup', pickup.dir);
        car.decisionSource = opts.source ?? 'central';
        continue;
      }

      if (opts.urgentOnly) continue;
      const home = this.homeFloor(car);
      if (home !== null && Math.abs(home - car.currentFloor) > 0.1) {
        this.setTarget(car, home, 'home', 0);
        car.decisionSource = opts.source ?? 'central';
      } else {
        car.targetFloor = null;
        car.targetReason = null;
        car.pickupDirection = 0;
        car.state = 'idle';
        car.direction = 0;
        car.decisionSource = 'none';
      }
    }
  }

  private assignAutonomousCars(
    now: number,
    claimed: Set<string> = new Set<string>(),
    source: DecisionSource = 'local',
  ): void {
    for (const car of this.elevators) {
      if (car.serviceRemaining > 0) continue;
      if (car.passengers.length > 0) {
        const dest = chooseDropoff(car, this.policy);
        if (dest !== null) this.setTarget(car, dest, 'dropoff', 0);
        car.decisionSource = source;
        continue;
      }
      if (car.targetFloor !== null && car.targetReason !== 'home' &&
          Math.abs(car.currentFloor - car.targetFloor) > 1e-9) continue;

      const local = this.choosePickup(car, now, claimed, {localRadius: this.config.localSensorRadius});
      const pickup = local ?? this.choosePickup(car, now, claimed, {urgentOnly: true});
      if (pickup) {
        if (source !== 'local') claimed.add(requestKey(pickup.floor, pickup.dir));
        this.setTarget(car, pickup.floor, 'pickup', pickup.dir);
        car.decisionSource = source;
        continue;
      }

      const home = this.homeFloor(car);
      if (home !== null && Math.abs(home - car.currentFloor) > 0.1) {
        this.setTarget(car, home, 'home', 0);
        car.decisionSource = source;
      } else if (car.targetReason === 'home') {
        car.decisionSource = source;
      } else {
        car.targetFloor = null;
        car.targetReason = null;
        car.pickupDirection = 0;
        car.state = 'idle';
        car.direction = 0;
        car.decisionSource = 'none';
      }
    }
  }

  private choosePickup(
    car: ElevatorCar,
    now: number,
    claimed: Set<string>,
    opts: {localRadius?: number; urgentOnly?: boolean} = {},
  ): {floor: number; dir: Dir} | null {
    let best: {floor: number; dir: Dir; score: number; features: PickupFeatures} | null = null;
    for (let floor = 0; floor < this.floors.length; floor++) {
      for (const dir of [1, -1] as Dir[]) {
        const queue = dir > 0 ? this.floors[floor].up : this.floors[floor].down;
        const eligible = queue.filter(p => this.canCarServePassenger(car, p));
        if (eligible.length === 0) continue;
        const key = requestKey(floor, dir);
        if (claimed.has(key)) continue;
        const distance = Math.abs(car.currentFloor - floor);
        if (opts.localRadius !== undefined && distance > opts.localRadius) continue;
        const oldestWait = Math.max(0, now - eligible[0].arrivalTime);
        if (opts.urgentOnly && oldestWait < this.config.urgentWaitThreshold) continue;
        const queueLen = eligible.length;
        const trip = averageTripFloors(eligible);
        const sameSide = sign(floor - car.currentFloor) === dir ? 1 : 0;
        const maxGroup = largestDestinationGroup(eligible);
        const features = {distance, oldestWait, queueLen, trip, sameSide, maxGroup};
        const score = this.scorePickup(features);
        if (!best || score < best.score) best = {floor, dir, score, features};
      }
    }
    if (best) {
      this.recordMDPDecision(best.features);
      return {floor: best.floor, dir: best.dir};
    }
    return null;
  }

  private scorePickup(features: PickupFeatures): number {
    const w = this.weightsFor(features);
    return features.distance * w.distance
      + features.trip * w.trip
      - features.queueLen * w.queue
      - features.oldestWait * w.wait
      - features.sameSide * w.sameDirection
      - features.maxGroup * w.destinationGroup;
  }

  private weightsFor(features: PickupFeatures): DispatchScoreWeights {
    const decision = this.mdpDecisionFor(features);
    if (decision) return this.mdpTuning!.actions[decision.actionIdx]?.weights ?? this.mdpTuning!.learnedWeights;
    return POLICY_SCORE_WEIGHTS[this.policy] ?? POLICY_SCORE_WEIGHTS['lowest-total-time'];
  }

  private mdpDecisionFor(features: PickupFeatures): {stateId: number; actionIdx: number; action: string} | null {
    if (!isMDPPolicy(this.policy) || !this.mdpTuning) return null;
    const stateId = encodeMDPDispatchState(features, this.mdpTuning.observability);
    const actionIdx = Math.max(0, this.mdpTuning.policy[stateId]);
    return {
      stateId,
      actionIdx,
      action: this.mdpTuning.actionLabels[actionIdx] ?? `a${actionIdx}`,
    };
  }

  private recordMDPDecision(features: PickupFeatures): void {
    const decision = this.mdpDecisionFor(features);
    if (!decision || !this.mdpTuning) return;
    this.mdpDecisionLog.push({
      stateId: decision.stateId,
      state: this.mdpTuning.stateLabels[decision.stateId] ?? `s${decision.stateId}`,
      action: decision.action,
      bins: mdpBinLabels(decodeMDPDispatchState(decision.stateId, this.mdpTuning.observability)),
    });
  }

  private homeFloor(car: ElevatorCar): number | null {
    if (this.policy === 'center-preposition' || isMDPPolicy(this.policy)) {
      const center = (this.config.nFloors - 1) / 2;
      const spacing = 3;
      return clamp(Math.round(center + (car.idx - (this.config.nElevators - 1) / 2) * spacing), 0, this.config.nFloors - 1);
    }
    if (this.policy === 'zoned-service' && car.allowedFloors) {
      const floors = Array.from(car.allowedFloors).sort((a, b) => a - b);
      return floors[Math.floor(floors.length / 2)] ?? 0;
    }
    return null;
  }

  private setTarget(car: ElevatorCar, floor: number, reason: TargetReason, pickupDir: Dir): void {
    floor = clamp(Math.round(floor), 0, this.config.nFloors - 1);
    if (car.targetFloor !== floor && Math.abs(car.currentFloor - floor) > 1e-9) {
      car.starts++;
      car.energy += 1.5;
    }
    car.targetFloor = floor;
    car.targetReason = reason;
    car.pickupDirection = pickupDir;
  }

  private canCarServePassenger(car: ElevatorCar, p: HighrisePassenger): boolean {
    if (!car.allowedFloors) return true;
    return car.allowedFloors.has(p.fromFloor) && car.allowedFloors.has(p.toFloor);
  }
}

export function buildHighriseSchedule(cfg: HighriseElevatorConfig): ScheduledArrival[] {
  return withSeed(cfg.seed, () => {
    const rng = mulberry32(cfg.seed);
    const out: ScheduledArrival[] = [];
    let t = 0;
    while (true) {
      t += -Math.log(Math.max(1e-9, 1 - rng())) / cfg.arrivalRate;
      if (t > cfg.simT) break;
      const r = rng();
      let fromFloor: number;
      let toFloor: number;
      if (r < 0.55) {
        fromFloor = 0;
        toFloor = 1 + Math.floor(rng() * (cfg.nFloors - 1));
      } else if (r < 0.80) {
        fromFloor = 1 + Math.floor(rng() * (cfg.nFloors - 1));
        toFloor = 0;
      } else {
        fromFloor = 1 + Math.floor(rng() * (cfg.nFloors - 1));
        do {
          toFloor = 1 + Math.floor(rng() * (cfg.nFloors - 1));
        } while (toFloor === fromFloor);
      }
      out.push({t, fromFloor, toFloor});
    }
    return out;
  });
}

export function runHighriseElevators(
  cfg: HighriseElevatorConfig,
  policy: HighrisePolicy,
  schedule: readonly ScheduledArrival[],
  opts: HighriseRunOptions = {},
): {result: HighriseElevatorResult; animation: Animation} {
  const authority = opts.authority ?? 'central';
  const building = new HighriseBuilding(cfg, policy, schedule, authority, opts.mdpTuning);
  const recordEvery = opts.recordEveryTicks ?? Math.max(1, Math.round(1 / cfg.stepSize));
  const frames: Frame[] = [];
  const series = {
    t: [] as number[],
    waiting: [] as number[],
    inCar: [] as number[],
    served: [] as number[],
    energy: [] as number[],
  };

  const maxTicks = Math.round((cfg.simT + cfg.drainT) / cfg.stepSize);
  for (let tick = 0; tick <= maxTicks; tick++) {
    building.runTimeStep(cfg.stepSize, tick);
    const t = tick * cfg.stepSize;
    if (tick % recordEvery === 0) {
      frames.push({t, tick, ...buildHighriseFrame(t, tick, building)});
      series.t.push(t);
      series.waiting.push(building.pendingPassengerCount());
      series.inCar.push(building.inCarCount());
      series.served.push(building.completed.length);
      series.energy.push(building.totalEnergy());
    }
    if (tick * cfg.stepSize >= cfg.simT && building.isDrained()) break;
  }

  const result = makeResult(policy, authority, cfg, schedule, building, opts.mdpTuning);
  const animation: Animation = {
    width: STAGE_W,
    height: STAGE_H,
    fps: 18,
    title: 'High-rise elevator dispatch policies',
    subtitle: `${cfg.nFloors} floors, ${cfg.nElevators} shafts, cap=${cfg.capacity}, dt=${cfg.stepSize}s, ${schedule.length} arrivals`,
    background: '#ffffff',
    frames,
    charts: [buildHighriseChart(series)],
  };
  return {result, animation};
}

function makeResult(
  policy: HighrisePolicy,
  authority: DecisionAuthority,
  config: HighriseElevatorConfig,
  schedule: readonly ScheduledArrival[],
  building: HighriseBuilding,
  mdpTuning?: MDPDispatchTuning,
): HighriseElevatorResult {
  const served = building.people.filter(p => p.exitTime >= 0);
  const waits = served.map(p => p.boardTime - p.arrivalTime);
  const travels = served.map(p => p.exitTime - p.boardTime);
  const totals = served.map(p => p.exitTime - p.arrivalTime);
  return {
    policy,
    authority,
    config,
    schedule: schedule.slice(),
    people: building.people.map(p => ({
      id: p.id,
      fromFloor: p.fromFloor,
      toFloor: p.toFloor,
      arrivalTime: p.arrivalTime,
      boardTime: p.boardTime,
      exitTime: p.exitTime,
    })),
    aggregates: {
      n: building.people.length,
      nServed: served.length,
      meanWait: mean(waits),
      meanTravel: mean(travels),
      meanTotal: mean(totals),
      p95Wait: percentile(waits, 0.95),
      p95Total: percentile(totals, 0.95),
      totalStops: building.totalStops(),
      totalDistanceFloors: building.totalDistance(),
      totalEnergy: building.totalEnergy(),
      timedOut: building.people.length - served.length,
    },
    mdpTuning: mdpTuning ? summarizeMDPTuning(mdpTuning) : undefined,
    mdpRun: building.mdpDecisionLog.length > 0 ? summarizeMDPRun(building.mdpDecisionLog) : undefined,
  };
}

const STAGE_W = 1200;
const STAGE_H = 760;
const BUILD_X = 78;
const BUILD_Y = 44;
const BUILD_W = 760;
const BUILD_H = 560;
const METRIC_X = 870;
const METRIC_Y = 44;
const METRIC_W = 290;
const METRIC_H = 560;

const POLICY_LABELS: Record<HighrisePolicy, string> = {
  'fewest-stops': 'Fewest stops',
  'lowest-total-time': 'Lowest total per-person time',
  'energy-efficient': 'Energy efficient',
  'center-preposition': 'Center preposition',
  'zoned-service': 'Zoned / even-odd service',
  'mdp-call-only': 'MDP no queue info',
  'mdp-tuned': 'MDP destination dispatch',
};

const POLICY_SUMMARIES: Record<HighrisePolicy, string> = {
  'fewest-stops': 'Batches riders by destination and avoids intermediate pickups once occupied.',
  'lowest-total-time': 'Dispatches the best nearby car and accepts useful same-direction pickups.',
  'energy-efficient': 'Penalizes travel, starts, and stop churn; prefers existing motion and batching.',
  'center-preposition': 'Uses lowest-time dispatch, then parks idle cars near the building center.',
  'zoned-service': 'Constrains shafts to all, low, mid, high, even, and odd service patterns.',
  'mdp-call-only': 'Uses value iteration with only binary hall-call, age, direction, and car-distance observations.',
  'mdp-tuned': 'Uses value iteration with destination-dispatch counts and destination-group estimates.',
};

const AUTHORITY_LABELS: Record<DecisionAuthority, string> = {
  central: 'Central brain',
  decentralized: 'Smart movables',
  hybrid: 'Hybrid',
};

const AUTHORITY_SUMMARIES: Record<DecisionAuthority, string> = {
  central: 'One global controller claims requests and coordinates shafts.',
  decentralized: 'Each elevator chooses from its local sensor view; duplicate claims are allowed.',
  hybrid: 'The controller handles urgent calls while idle cars make local decisions.',
};

const POLICY_SCORE_WEIGHTS: Record<HighrisePolicy, DispatchScoreWeights> = {
  'fewest-stops': {
    distance: 2.0,
    trip: 0.15,
    queue: 0.15,
    wait: 0.025,
    sameDirection: 0.1,
    destinationGroup: 3.0,
  },
  'lowest-total-time': {
    distance: 1.25,
    trip: 0.2,
    queue: 1.1,
    wait: 0.08,
    sameDirection: 0.25,
    destinationGroup: 0.2,
  },
  'energy-efficient': {
    distance: 2.2,
    trip: 0.35,
    queue: 0.45,
    wait: 0.035,
    sameDirection: 0.8,
    destinationGroup: 0.6,
  },
  'center-preposition': {
    distance: 1.15,
    trip: 0.18,
    queue: 0.9,
    wait: 0.065,
    sameDirection: 0.3,
    destinationGroup: 0.2,
  },
  'zoned-service': {
    distance: 1.35,
    trip: 0.15,
    queue: 0.8,
    wait: 0.05,
    sameDirection: 0.4,
    destinationGroup: 0.25,
  },
  'mdp-call-only': {
    distance: 1.25,
    trip: 0.2,
    queue: 1.1,
    wait: 0.08,
    sameDirection: 0.25,
    destinationGroup: 0.2,
  },
  'mdp-tuned': {
    distance: 1.25,
    trip: 0.2,
    queue: 1.1,
    wait: 0.08,
    sameDirection: 0.25,
    destinationGroup: 0.2,
  },
};

function buildHighriseFrame(t: number, tick: number, b: HighriseBuilding): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const cfg = b.config;
  const floorH = BUILD_H / cfg.nFloors;
  const shaftW = BUILD_W / cfg.nElevators;
  const carW = Math.min(42, shaftW * 0.42);
  const carH = Math.max(7, floorH * 0.82);

  shapes.push({kind: 'rect', x: BUILD_X, y: BUILD_Y, w: BUILD_W, h: BUILD_H,
               fill: '#fff', stroke: '#c8c8c8', strokeWidth: 1, rx: 4});

  for (let f = 0; f < cfg.nFloors; f++) {
    const y = floorY(f, cfg);
    const major = f % 5 === 0 || f === cfg.nFloors - 1;
    if (major) {
      shapes.push({kind: 'line', x1: BUILD_X, y1: y, x2: BUILD_X + BUILD_W, y2: y,
                   stroke: f === 0 ? '#a3a3a3' : '#e3e3e3', strokeWidth: f === 0 ? 1.2 : 1});
      shapes.push({kind: 'text', x: BUILD_X - 10, y: y + 4, text: String(f),
                   fontSize: 10, fill: '#555', anchor: 'end'});
    }
    const queues = b.floors[f];
    const waiting = queues.up.length + queues.down.length;
    if (waiting > 0) {
      const yMid = y - floorH / 2;
      const upW = Math.min(80, queues.up.length * 4);
      const downW = Math.min(80, queues.down.length * 4);
      if (upW > 0) shapes.push({kind: 'rect', x: BUILD_X + 8, y: yMid - 4, w: upW, h: 4, fill: '#16a34a', rx: 1});
      if (downW > 0) shapes.push({kind: 'rect', x: BUILD_X + 8, y: yMid + 1, w: downW, h: 4, fill: '#2563eb', rx: 1});
      shapes.push({kind: 'text', x: BUILD_X + 94, y: yMid + 4, text: String(waiting),
                   fontSize: 8, fill: '#444', anchor: 'start'});
    }
  }

  for (let k = 0; k < cfg.nElevators; k++) {
    const sx = BUILD_X + k * shaftW + shaftW / 2;
    shapes.push({kind: 'line', x1: sx, y1: BUILD_Y, x2: sx, y2: BUILD_Y + BUILD_H,
                 stroke: '#ededed', strokeWidth: 1});
    shapes.push({kind: 'text', x: sx, y: BUILD_Y + BUILD_H + 18, text: `E${k}`,
                 fontSize: 10, fill: '#555', anchor: 'middle'});
  }

  for (const car of b.elevators) {
    const sx = BUILD_X + car.idx * shaftW + shaftW / 2;
    const y = floorY(car.currentFloor, cfg) - carH / 2;
    const fill = carColor(car);
    shapes.push({kind: 'rect', x: sx - carW / 2, y, w: carW, h: carH, fill,
                 stroke: '#222', strokeWidth: 0.7, rx: 2,
                 title: `${car.id} ${car.state} F${car.currentFloor.toFixed(1)} pax=${car.passengers.length}/${car.capacity}`});
    shapes.push({kind: 'text', x: sx, y: y + carH / 2 + 3, text: String(car.passengers.length),
                 fontSize: 9, fill: '#fff', anchor: 'middle', fontWeight: 'bold'});
    if (car.targetFloor !== null && car.targetReason !== 'home') {
      const ty = floorY(car.targetFloor, cfg);
      shapes.push({kind: 'line', x1: sx, y1: y + carH / 2, x2: sx, y2: ty,
                   stroke: '#777', strokeWidth: 0.7, dasharray: '2,3', opacity: 0.75});
      shapes.push({kind: 'circle', x: sx, y: ty, r: 2.4, fill: '#777'});
    }
  }

  drawMetrics(shapes, b, t, tick);
  return {
    shapes,
    caption: `policy=${POLICY_LABELS[b.policy]}  authority=${AUTHORITY_LABELS[b.authority]}  t=${t.toFixed(1)}s  waiting=${b.pendingPassengerCount()}  in-car=${b.inCarCount()}  served=${b.completed.length}`,
  };
}

function drawMetrics(shapes: Shape[], b: HighriseBuilding, t: number, tick: number): void {
  const a = makeResult(b.policy, b.authority, b.config, b.schedule, b, b.mdpTuning).aggregates;
  shapes.push({kind: 'rect', x: METRIC_X, y: METRIC_Y, w: METRIC_W, h: METRIC_H,
               fill: '#fbfbfb', stroke: '#ddd', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: METRIC_X + 14, y: METRIC_Y + 24,
               text: POLICY_LABELS[b.policy], fontSize: 15, fill: '#111', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: METRIC_X + 14, y: METRIC_Y + 44,
               text: `${AUTHORITY_LABELS[b.authority]}  tick ${tick}  t=${t.toFixed(1)}s`, fontSize: 11, fill: '#555'});

  const rows: Array<[string, string]> = [
    ['waiting', String(b.pendingPassengerCount())],
    ['in cars', String(b.inCarCount())],
    ['served', `${a.nServed}/${a.n}`],
    ['mean wait', `${a.meanWait.toFixed(1)}s`],
    ['mean total', `${a.meanTotal.toFixed(1)}s`],
    ['p95 total', `${a.p95Total.toFixed(1)}s`],
    ['stops', String(Math.round(a.totalStops))],
    ['distance', `${a.totalDistanceFloors.toFixed(1)} floors`],
    ['energy index', a.totalEnergy.toFixed(1)],
  ];
  for (let i = 0; i < rows.length; i++) {
    const y = METRIC_Y + 76 + i * 24;
    shapes.push({kind: 'text', x: METRIC_X + 14, y, text: rows[i][0], fontSize: 11, fill: '#666'});
    shapes.push({kind: 'text', x: METRIC_X + METRIC_W - 14, y, text: rows[i][1],
                 fontSize: 12, fill: '#222', anchor: 'end', fontWeight: 'bold'});
  }

  const y0 = METRIC_Y + 320;
  shapes.push({kind: 'text', x: METRIC_X + 14, y: y0, text: 'Shafts', fontSize: 12, fill: '#333', fontWeight: 'bold'});
  for (const car of b.elevators) {
    const y = y0 + 22 + car.idx * 28;
    shapes.push({kind: 'rect', x: METRIC_X + 14, y: y - 10, w: 12, h: 12, fill: carColor(car), rx: 2});
    shapes.push({kind: 'text', x: METRIC_X + 34, y, text: car.id, fontSize: 11, fill: '#222', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: METRIC_X + 66, y,
                 text: `F${car.currentFloor.toFixed(1)} ${car.passengers.length}/${car.capacity}`,
                 fontSize: 11, fill: '#444'});
    shapes.push({kind: 'text', x: METRIC_X + 150, y,
                 text: car.targetFloor === null ? `${car.state} ${car.decisionSource}` : `${car.state} ->F${car.targetFloor} ${car.decisionSource}`,
                 fontSize: 10, fill: '#666'});
  }
}

function buildHighriseChart(series: {t: number[]; waiting: number[]; inCar: number[]; served: number[]; energy: number[]}): ChartSpec {
  return {
    x: BUILD_X,
    y: 635,
    w: BUILD_W,
    h: 95,
    title: 'System trajectory',
    yMin: 0,
    series: [
      {label: 'waiting', color: '#dc2626', t: series.t, y: series.waiting},
      {label: 'in cars', color: '#2563eb', t: series.t, y: series.inCar},
      {label: 'served', color: '#16a34a', t: series.t, y: series.served},
      {label: 'energy/10', color: '#7c3aed', t: series.t, y: series.energy.map(v => v / 10)},
    ],
  };
}

const MDP_DISTANCE_BINS = [2, 8, 18, Infinity];
const MDP_QUEUE_BINS = [1, 4, Infinity];
const MDP_WAIT_BINS = [15, 45, Infinity];
const MDP_TRIP_BINS = [8, 22, Infinity];
const MDP_BATCH_BINS = [1, 3, Infinity];
const CALL_ONLY_EXPECTED_TRIP = 18;

const MDP_ACTION_PROFILES: MDPActionProfile[] = [
  {
    label: 'direct-batch',
    weights: {distance: 1.8, trip: 0.12, queue: 0.3, wait: 0.035, sameDirection: 0.2, destinationGroup: 2.8},
  },
  {
    label: 'latency',
    weights: {distance: 1.05, trip: 0.18, queue: 1.35, wait: 0.105, sameDirection: 0.25, destinationGroup: 0.3},
  },
  {
    label: 'energy',
    weights: {distance: 2.45, trip: 0.42, queue: 0.55, wait: 0.04, sameDirection: 1.1, destinationGroup: 0.65},
  },
  {
    label: 'balanced',
    weights: {distance: 1.35, trip: 0.2, queue: 0.95, wait: 0.07, sameDirection: 0.45, destinationGroup: 0.45},
  },
  {
    label: 'oldest-first',
    weights: {distance: 0.95, trip: 0.16, queue: 0.75, wait: 0.16, sameDirection: 0.2, destinationGroup: 0.15},
  },
];

interface MDPDispatchStateBins {
  distanceBin: number;
  waitBin: number;
  sameSide: number;
  queueBin?: number;
  tripBin?: number;
  batchBin?: number;
}

function isMDPPolicy(policy: HighrisePolicy): boolean {
  return policy === 'mdp-call-only' || policy === 'mdp-tuned';
}

function observabilityForPolicy(policy: HighrisePolicy): MDPObservability {
  return policy === 'mdp-call-only' ? 'call-only' : 'destination-dispatch';
}

function mdpNumStates(observability: MDPObservability): number {
  let n = MDP_DISTANCE_BINS.length * MDP_WAIT_BINS.length * 2;
  if (observability === 'destination-dispatch') {
    n *= MDP_QUEUE_BINS.length * MDP_TRIP_BINS.length * MDP_BATCH_BINS.length;
  }
  return n;
}

function optimizeHighriseDispatchMDP(observability: MDPObservability): MDPDispatchTuning {
  const {spec, stateLabels, actionLabels} = highriseDispatchMDPSpec(observability);
  const gamma = Number(process.env.MDP_GAMMA ?? 0.92);
  const vi = valueIteration(spec, {
    gamma,
    tol: Number(process.env.MDP_TOL ?? 1e-8),
    maxIter: Number(process.env.MDP_MAX_ITER ?? 10000),
    randomTieBreak: false,
  });
  const learnedWeights = averageMDPWeights(vi.policy, observability);
  return {
    observability,
    numStates: spec.numStates,
    actions: MDP_ACTION_PROFILES,
    policy: vi.policy,
    V: vi.V,
    gamma: vi.gamma,
    iterations: vi.iterations,
    finalDelta: vi.finalDelta,
    learnedWeights,
    stateLabels,
    actionLabels,
  };
}

function highriseDispatchMDPSpec(observability: MDPObservability): {spec: MDPSpec; stateLabels: string[]; actionLabels: string[]} {
  const numStates = mdpNumStates(observability);
  const stateLabels = Array.from({length: numStates}, (_, s) => labelMDPDispatchState(decodeMDPDispatchState(s, observability)));
  const actionLabels = MDP_ACTION_PROFILES.map(a => a.label);
  const spec: MDPSpec = {
    numStates,
    numActions: () => MDP_ACTION_PROFILES.length,
    outcomes: (s, a) => abstractDispatchOutcomes(s, a, observability),
    stateLabel: s => stateLabels[s],
    actionLabel: a => actionLabels[a],
  };
  return {spec, stateLabels, actionLabels};
}

function abstractDispatchOutcomes(
  s: number,
  actionIdx: number,
  observability: MDPObservability,
): Array<{prob: number; reward: number; nextState: number}> {
  const st = decodeMDPDispatchState(s, observability);
  const action = MDP_ACTION_PROFILES[actionIdx];
  const d = [1, 5, 13, 28][st.distanceBin];
  const q = observability === 'destination-dispatch' ? [1, 3, 7][st.queueBin ?? 0] : 1;
  const wait = [8, 30, 75][st.waitBin];
  const trip = observability === 'destination-dispatch' ? [4, 15, 31][st.tripBin ?? 0] : CALL_ONLY_EXPECTED_TRIP;
  const batch = observability === 'destination-dispatch' ? [1, 2, 5][st.batchBin ?? 0] : 1;
  const same = st.sameSide;

  const w = action.weights;
  const directDemand = observability === 'destination-dispatch' &&
    ((st.batchBin ?? 0) >= 1 || ((st.queueBin ?? 0) >= 1 && (st.tripBin ?? 0) >= 1)) ? 1 : 0;
  const urgency = st.waitBin / 2;
  const energyRisk = st.distanceBin / 3 + (same ? -0.25 : 0.25);
  const timeCost = d * 1.35 + trip * 0.65 + wait * 0.85 - q * 3.5 - batch * 1.5;
  const stopCost = Math.max(0.5, 2.2 - w.destinationGroup * directDemand - (st.batchBin ?? 0) * 0.35);
  const energyCost = d * (1.0 + 0.24 * w.distance) + trip * 0.07 - same * w.sameDirection;
  const hiddenQueuePenalty = observability === 'call-only'
    ? Math.max(0, urgency - w.wait * 7) + Math.max(0, 0.8 - w.distance * 0.18 - w.sameDirection * 0.25)
    : 0;
  const mismatch =
    Math.max(0, urgency - w.wait * 8)
    + Math.max(0, energyRisk - w.sameDirection * 0.45)
    + Math.max(0, directDemand + (st.batchBin ?? 0) * 0.35 - w.destinationGroup * 0.28)
    + hiddenQueuePenalty;
  const reward = -(timeCost + energyCost * 1.8 + stopCost * 6 + mismatch * 12);

  const fit = clamp(
    0.52
      + (w.wait > 0.09 && st.waitBin >= 1 ? 0.12 : 0)
      + (w.destinationGroup > 1.5 && directDemand ? 0.12 : 0)
      + (w.destinationGroup > 1.5 && (st.batchBin ?? 0) >= 1 ? 0.08 : 0)
      + (w.sameDirection > 0.7 && same ? 0.10 : 0)
      + (w.distance > 2.0 && st.distanceBin <= 1 ? 0.08 : 0)
      - (mismatch * 0.05),
    0.2,
    0.88,
  );

  const improved: MDPDispatchStateBins = {
    distanceBin: Math.max(0, st.distanceBin - (w.distance > 1.7 ? 1 : 0)),
    waitBin: Math.max(0, st.waitBin - (w.wait > 0.08 ? 1 : 0)),
    sameSide: same,
  };
  const degraded: MDPDispatchStateBins = {
    distanceBin: Math.min(MDP_DISTANCE_BINS.length - 1, st.distanceBin + (w.distance < 1.2 ? 1 : 0)),
    waitBin: Math.min(MDP_WAIT_BINS.length - 1, st.waitBin + (w.wait < 0.07 ? 1 : 0)),
    sameSide: same ? 1 : 0,
  };
  if (observability === 'destination-dispatch') {
    improved.queueBin = Math.max(0, (st.queueBin ?? 0) - (w.queue > 0.8 || w.destinationGroup > 1.5 ? 1 : 0));
    improved.tripBin = Math.max(0, (st.tripBin ?? 0) - (w.destinationGroup > 1.5 ? 1 : 0));
    improved.batchBin = Math.max(0, (st.batchBin ?? 0) - (w.destinationGroup > 1.5 ? 1 : 0));
    degraded.queueBin = Math.min(MDP_QUEUE_BINS.length - 1, (st.queueBin ?? 0) + (w.queue < 0.75 ? 1 : 0));
    degraded.tripBin = st.tripBin ?? 0;
    degraded.batchBin = st.batchBin ?? 0;
  }
  return [
    {prob: fit, reward, nextState: encodeMDPDispatchBins(improved, observability)},
    {prob: 1 - fit, reward: reward - 5 - wait * 0.05, nextState: encodeMDPDispatchBins(degraded, observability)},
  ];
}

function encodeMDPDispatchState(features: PickupFeatures, observability: MDPObservability): number {
  if (observability === 'call-only') {
    return encodeMDPDispatchBins({
      distanceBin: binIndex(features.distance, MDP_DISTANCE_BINS),
      waitBin: binIndex(features.oldestWait, MDP_WAIT_BINS),
      sameSide: features.sameSide > 0 ? 1 : 0,
    }, observability);
  }
  return encodeMDPDispatchBins({
    distanceBin: binIndex(features.distance, MDP_DISTANCE_BINS),
    queueBin: binIndex(features.queueLen, MDP_QUEUE_BINS),
    waitBin: binIndex(features.oldestWait, MDP_WAIT_BINS),
    tripBin: binIndex(features.trip, MDP_TRIP_BINS),
    batchBin: binIndex(features.maxGroup, MDP_BATCH_BINS),
    sameSide: features.sameSide > 0 ? 1 : 0,
  }, observability);
}

function encodeMDPDispatchBins(st: MDPDispatchStateBins, observability: MDPObservability): number {
  let idx = st.distanceBin;
  if (observability === 'destination-dispatch') idx = idx * MDP_QUEUE_BINS.length + (st.queueBin ?? 0);
  idx = idx * MDP_WAIT_BINS.length + st.waitBin;
  if (observability === 'destination-dispatch') {
    idx = idx * MDP_TRIP_BINS.length + (st.tripBin ?? 0);
    idx = idx * MDP_BATCH_BINS.length + (st.batchBin ?? 0);
  }
  idx = idx * 2 + st.sameSide;
  return idx;
}

function decodeMDPDispatchState(s: number, observability: MDPObservability): MDPDispatchStateBins {
  const sameSide = s % 2; s = Math.floor(s / 2);
  let batchBin: number | undefined;
  let tripBin: number | undefined;
  let queueBin: number | undefined;
  if (observability === 'destination-dispatch') {
    batchBin = s % MDP_BATCH_BINS.length; s = Math.floor(s / MDP_BATCH_BINS.length);
    tripBin = s % MDP_TRIP_BINS.length; s = Math.floor(s / MDP_TRIP_BINS.length);
  }
  const waitBin = s % MDP_WAIT_BINS.length; s = Math.floor(s / MDP_WAIT_BINS.length);
  if (observability === 'destination-dispatch') {
    queueBin = s % MDP_QUEUE_BINS.length; s = Math.floor(s / MDP_QUEUE_BINS.length);
  }
  const distanceBin = s;
  return {distanceBin, queueBin, waitBin, tripBin, batchBin, sameSide};
}

function labelMDPDispatchState(st: MDPDispatchStateBins): string {
  const parts = [`d${st.distanceBin}`];
  if (st.queueBin !== undefined) parts.push(`q${st.queueBin}`);
  parts.push(`w${st.waitBin}`);
  if (st.tripBin !== undefined) parts.push(`trip${st.tripBin}`);
  if (st.batchBin !== undefined) parts.push(`batch${st.batchBin}`);
  parts.push(st.sameSide ? 'same' : 'reverse');
  return parts.join('/');
}

function mdpBinLabels(st: MDPDispatchStateBins): Record<string, string> {
  const out: Record<string, string> = {
    distance: `d${st.distanceBin}`,
    wait: `w${st.waitBin}`,
    direction: st.sameSide ? 'same' : 'reverse',
  };
  if (st.queueBin !== undefined) out.queue = `q${st.queueBin}`;
  if (st.tripBin !== undefined) out.trip = `trip${st.tripBin}`;
  if (st.batchBin !== undefined) out.batch = `batch${st.batchBin}`;
  return out;
}

function binIndex(x: number, thresholds: number[]): number {
  for (let i = 0; i < thresholds.length; i++) if (x <= thresholds[i]) return i;
  return thresholds.length - 1;
}

function averageMDPWeights(policy: Int32Array, observability: MDPObservability): DispatchScoreWeights {
  const out: DispatchScoreWeights = {distance: 0, trip: 0, queue: 0, wait: 0, sameDirection: 0, destinationGroup: 0};
  let total = 0;
  for (let s = 0; s < policy.length; s++) {
    const st = decodeMDPDispatchState(s, observability);
    const importance = 1 + (st.queueBin ?? 0) + st.waitBin + (st.batchBin ?? 0) * 0.7 + (st.tripBin ?? 0) * 0.35;
    const weights = MDP_ACTION_PROFILES[Math.max(0, policy[s])].weights;
    out.distance += weights.distance * importance;
    out.trip += weights.trip * importance;
    out.queue += weights.queue * importance;
    out.wait += weights.wait * importance;
    out.sameDirection += weights.sameDirection * importance;
    out.destinationGroup += weights.destinationGroup * importance;
    total += importance;
  }
  for (const key of Object.keys(out) as Array<keyof DispatchScoreWeights>) out[key] /= total;
  return out;
}

function summarizeMDPTuning(tuning: MDPDispatchTuning): MDPDispatchTuningSummary {
  const interesting = [0, 1, 5, 17, 43, 87, 129, 173, tuning.numStates - 1]
    .filter((s, idx, arr) => s >= 0 && s < tuning.numStates && arr.indexOf(s) === idx);
  return {
    observability: tuning.observability,
    numStates: tuning.numStates,
    gamma: tuning.gamma,
    iterations: tuning.iterations,
    finalDelta: tuning.finalDelta,
    learnedWeights: tuning.learnedWeights,
    statePolicy: interesting.map(s => ({
      state: tuning.stateLabels[s],
      action: tuning.actionLabels[Math.max(0, tuning.policy[s])],
    })),
  };
}

function summarizeMDPRun(log: MDPDecisionLogEntry[]): MDPRunDiagnostics {
  const total = log.length;
  const actionCounts = new Map<string, number>();
  const stateCounts = new Map<string, {state: string; action: string; count: number}>();
  const marginal = new Map<string, Map<string, Map<string, number>>>();

  for (const row of log) {
    actionCounts.set(row.action, (actionCounts.get(row.action) ?? 0) + 1);
    const stateKey = `${row.state}|${row.action}`;
    const stateRow = stateCounts.get(stateKey) ?? {state: row.state, action: row.action, count: 0};
    stateRow.count += 1;
    stateCounts.set(stateKey, stateRow);

    for (const [variable, bin] of Object.entries(row.bins)) {
      let byBin = marginal.get(variable);
      if (!byBin) {
        byBin = new Map();
        marginal.set(variable, byBin);
      }
      let byAction = byBin.get(bin);
      if (!byAction) {
        byAction = new Map();
        byBin.set(bin, byAction);
      }
      byAction.set(row.action, (byAction.get(row.action) ?? 0) + 1);
    }
  }

  return {
    totalDecisions: total,
    actionCounts: Array.from(actionCounts.entries())
      .map(([action, count]) => ({action, count, share: total > 0 ? count / total : 0}))
      .sort((a, b) => b.count - a.count),
    topStates: Array.from(stateCounts.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, 8),
    marginals: Array.from(marginal.entries()).map(([variable, byBin]) => ({
      variable,
      bins: Array.from(byBin.entries()).map(([bin, byAction]) => {
        const count = Array.from(byAction.values()).reduce((s, x) => s + x, 0);
        let dominantAction = '';
        let dominantCount = -1;
        for (const [action, n] of byAction) {
          if (n > dominantCount) {
            dominantAction = action;
            dominantCount = n;
          }
        }
        return {bin, count, dominantAction, share: count > 0 ? dominantCount / count : 0};
      }).sort((a, b) => a.bin.localeCompare(b.bin)),
    })).sort((a, b) => a.variable.localeCompare(b.variable)),
  };
}

function compareToBaseline(result: HighriseElevatorResult, baseline: HighriseElevatorResult): MarginalComparison {
  return {
    baselinePolicy: baseline.policy,
    baselineAuthority: baseline.authority,
    meanWaitDelta: result.aggregates.meanWait - baseline.aggregates.meanWait,
    meanTotalDelta: result.aggregates.meanTotal - baseline.aggregates.meanTotal,
    stopsDelta: result.aggregates.totalStops - baseline.aggregates.totalStops,
    energyDelta: result.aggregates.totalEnergy - baseline.aggregates.totalEnergy,
  };
}

function variantSummary(result: HighriseElevatorResult): string {
  const a = result.aggregates;
  let out = `${POLICY_SUMMARIES[result.policy]} ${AUTHORITY_SUMMARIES[result.authority]} ` +
    `Mean total ${a.meanTotal.toFixed(1)}s, stops ${a.totalStops}, energy ${a.totalEnergy.toFixed(1)}.`;
  if (result.mdpTuning && result.mdpRun) {
    const w = result.mdpTuning.learnedWeights;
    const marginalName = result.mdpTuning.observability === 'destination-dispatch' ? 'batch' : 'wait';
    out += ` MDP is pre-solved by value iteration (${result.mdpTuning.numStates} states, ${result.mdpTuning.iterations} sweeps, observability=${result.mdpTuning.observability}), ` +
      `then this run exercised ${result.mdpRun.totalDecisions} learned pickup decisions. ` +
      `Observed actions: ${formatActionShares(result.mdpRun)}. ` +
      `${marginalName} marginal: ${formatMarginal(result.mdpRun, marginalName)}. ` +
      `Learned weights favor destination grouping=${w.destinationGroup.toFixed(2)}, distance=${w.distance.toFixed(2)}, wait=${w.wait.toFixed(2)}.`;
  }
  if (result.marginalVsLowestTime) {
    const m = result.marginalVsLowestTime;
    out += ` Marginal vs ${POLICY_LABELS[m.baselinePolicy]} / ${AUTHORITY_LABELS[m.baselineAuthority]}: ` +
      `mean total ${formatSigned(m.meanTotalDelta, 's')}, wait ${formatSigned(m.meanWaitDelta, 's')}, ` +
      `stops ${formatSigned(m.stopsDelta, '')}, energy ${formatSigned(m.energyDelta, '')}.`;
  }
  return out;
}

function formatActionShares(run: MDPRunDiagnostics, maxItems = 3): string {
  if (run.actionCounts.length === 0) return 'none';
  return run.actionCounts.slice(0, maxItems)
    .map(r => `${r.action} ${r.count}/${run.totalDecisions} (${(100 * r.share).toFixed(0)}%)`)
    .join(', ');
}

function formatMarginal(run: MDPRunDiagnostics, variable: string): string {
  const row = run.marginals.find(m => m.variable === variable);
  if (!row || row.bins.length === 0) return 'none';
  return row.bins
    .map(b => `${b.bin}->${b.dominantAction} ${(100 * b.share).toFixed(0)}%`)
    .join(', ');
}

function formatSigned(x: number, suffix: string): string {
  const sign = x > 0 ? '+' : '';
  return `${sign}${x.toFixed(1)}${suffix}`;
}

function chooseDropoff(car: ElevatorCar, policy: HighrisePolicy): number | null {
  if (car.passengers.length === 0) return null;
  if (policy === 'fewest-stops') return car.passengers[0].toFloor;
  const current = car.currentFloor;
  const dir = car.direction !== 0 ? car.direction : sign(car.passengers[0].toFloor - current);
  const ahead = car.passengers
    .map(p => p.toFloor)
    .filter(f => dir === 0 || sign(f - current) === dir || Math.abs(f - current) < 1e-9)
    .sort((a, b) => Math.abs(a - current) - Math.abs(b - current));
  if (ahead.length > 0) return ahead[0];
  return car.passengers
    .map(p => p.toFloor)
    .sort((a, b) => Math.abs(a - current) - Math.abs(b - current))[0];
}

function allowedFloorsFor(idx: number, nFloors: number): Set<number> {
  const floors = new Set<number>();
  const addRange = (a: number, b: number) => {
    for (let f = Math.max(0, a); f <= Math.min(nFloors - 1, b); f++) floors.add(f);
  };
  if (idx === 0) addRange(0, nFloors - 1);
  else if (idx === 1) addRange(0, 20);
  else if (idx === 2) { floors.add(0); addRange(15, 35); }
  else if (idx === 3) { floors.add(0); addRange(30, nFloors - 1); }
  else if (idx === 4) { for (let f = 0; f < nFloors; f += 2) floors.add(f); }
  else if (idx === 5) { floors.add(0); for (let f = 1; f < nFloors; f += 2) floors.add(f); }
  else addRange(0, nFloors - 1);
  return floors;
}

function dominantDestination(queue: HighrisePassenger[], car: ElevatorCar): number | null {
  const counts = new Map<number, number>();
  for (const p of queue) counts.set(p.toFloor, (counts.get(p.toFloor) ?? 0) + 1);
  let best: number | null = null;
  let bestCount = -1;
  for (const [floor, count] of counts) {
    if (count > bestCount || (count === bestCount && best !== null && Math.abs(floor - car.currentFloor) < Math.abs(best - car.currentFloor))) {
      best = floor;
      bestCount = count;
    }
  }
  return best;
}

function largestDestinationGroup(queue: HighrisePassenger[]): number {
  const counts = new Map<number, number>();
  let best = 0;
  for (const p of queue) {
    const n = (counts.get(p.toFloor) ?? 0) + 1;
    counts.set(p.toFloor, n);
    best = Math.max(best, n);
  }
  return best;
}

function averageTripFloors(queue: HighrisePassenger[]): number {
  if (queue.length === 0) return 0;
  return queue.reduce((s, p) => s + Math.abs(p.toFloor - p.fromFloor), 0) / queue.length;
}

function wouldAddReverseStop(car: ElevatorCar, p: HighrisePassenger): boolean {
  if (car.direction === 0) return false;
  return sign(p.toFloor - car.currentFloor) !== car.direction;
}

function floorY(floor: number, cfg: HighriseElevatorConfig): number {
  const span = Math.max(1, cfg.nFloors - 1);
  return BUILD_Y + BUILD_H - (floor / span) * BUILD_H;
}

function carColor(car: ElevatorCar): string {
  if (car.state === 'serving') return '#f59e0b';
  if (car.state === 'prepositioning') return '#7c3aed';
  if (car.direction > 0) return '#16a34a';
  if (car.direction < 0) return '#2563eb';
  return '#9ca3af';
}

function requestKey(floor: number, dir: Dir): string {
  return `${floor}:${dir}`;
}

function sign(x: number): Dir {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((s, x) => s + x, 0) / xs.length;
}

function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const sorted = xs.slice().sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * p)];
}

function defaultConfig(): HighriseElevatorConfig {
  return {
    nFloors: Number(process.env.FLOORS ?? 50),
    nElevators: Number(process.env.ELEVATORS ?? 6),
    capacity: Number(process.env.CAPACITY ?? 12),
    floorTravelTime: Number(process.env.TRAVEL_T ?? 1.35),
    serviceTime: Number(process.env.SERVICE_T ?? 3),
    arrivalRate: Number(process.env.LAMBDA ?? 0.22),
    simT: Number(process.env.SIM_T ?? 360),
    drainT: Number(process.env.DRAIN_T ?? 300),
    stepSize: Number(process.env.STEPSIZE ?? 0.1),
    seed: Number(process.env.SEED ?? 11),
    localSensorRadius: Number(process.env.LOCAL_SENSOR_RADIUS ?? 12),
    urgentWaitThreshold: Number(process.env.URGENT_WAIT ?? 45),
  };
}

async function main(): Promise<void> {
  const cfg = defaultConfig();
  const schedule = buildHighriseSchedule(cfg);
  const policies = (process.env.POLICIES
    ? process.env.POLICIES.split(',').map(s => s.trim()).filter(Boolean) as HighrisePolicy[]
    : HIGHRISE_POLICIES);
  const authorities = (process.env.AUTHORITIES
    ? process.env.AUTHORITIES.split(',').map(s => s.trim()).filter(Boolean) as DecisionAuthority[]
    : DECISION_AUTHORITIES);
  const recordEvery = Number(process.env.RECORD_EVERY ?? Math.max(1, Math.round(Number(process.env.ANIM_DT ?? 2) / cfg.stepSize)));
  const mdpTunings = new Map<HighrisePolicy, MDPDispatchTuning>();
  for (const policy of policies) {
    if (isMDPPolicy(policy) && !mdpTunings.has(policy)) {
      mdpTunings.set(policy, optimizeHighriseDispatchMDP(observabilityForPolicy(policy)));
    }
  }

  console.log('# High-rise elevator simulation');
  console.log(`#   ${cfg.nFloors} floors, ${cfg.nElevators} shafts, capacity ${cfg.capacity}`);
  console.log(`#   dt=${cfg.stepSize}s, recordEvery=${recordEvery} ticks, local sensor radius=${cfg.localSensorRadius} floors`);
  console.log(`#   ${schedule.length} scheduled arrivals, source active ${cfg.simT}s, drain ${cfg.drainT}s`);
  for (const [policy, mdpTuning] of mdpTunings) {
    const w = mdpTuning.learnedWeights;
    console.log(`#   ${POLICY_LABELS[policy]} VI: ${mdpTuning.numStates} abstract states, ${mdpTuning.actions.length} actions, ` +
      `${mdpTuning.iterations} sweeps, max|ΔV|=${mdpTuning.finalDelta.toExponential(3)}`);
    console.log(`#   MDP learned weights: distance=${w.distance.toFixed(3)}, trip=${w.trip.toFixed(3)}, ` +
      `queue=${w.queue.toFixed(3)}, wait=${w.wait.toFixed(3)}, sameDir=${w.sameDirection.toFixed(3)}, destGroup=${w.destinationGroup.toFixed(3)}`);
  }

  const variants: AnimationVariant[] = [];
  const results: HighriseElevatorResult[] = [];
  for (const authority of authorities) {
    for (const policy of policies) {
      const {result, animation} = runHighriseElevators(cfg, policy, schedule, {
        authority,
        recordEveryTicks: recordEvery,
        mdpTuning: mdpTunings.get(policy),
      });
      results.push(result);
      const baseline = results.find(r => r.authority === authority && r.policy === 'lowest-total-time');
      if (isMDPPolicy(policy) && baseline) {
        result.marginalVsLowestTime = compareToBaseline(result, baseline);
      }
      const a = result.aggregates;
      console.log('');
      console.log(`# ${POLICY_LABELS[policy]} / ${AUTHORITY_LABELS[authority]}`);
      console.log(`#   served ${a.nServed}/${a.n}, mean wait ${a.meanWait.toFixed(1)}s, mean total ${a.meanTotal.toFixed(1)}s, p95 total ${a.p95Total.toFixed(1)}s`);
      console.log(`#   stops ${a.totalStops}, distance ${a.totalDistanceFloors.toFixed(1)} floors, energy index ${a.totalEnergy.toFixed(1)}`);
      if (result.mdpRun) {
        const marginalName = result.mdpTuning?.observability === 'destination-dispatch' ? 'batch' : 'wait';
        console.log(`#   MDP observed actions: ${formatActionShares(result.mdpRun)}`);
        console.log(`#   MDP ${marginalName} marginal: ${formatMarginal(result.mdpRun, marginalName)}`);
      }
      variants.push({
        id: `${policy}-${authority}`,
        label: `${POLICY_LABELS[policy]} / ${AUTHORITY_LABELS[authority]}`,
        controls: {policy: POLICY_LABELS[policy], authority: AUTHORITY_LABELS[authority]},
        summary: variantSummary(result),
        animation,
      });
    }
  }

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const htmlPath = path.join(outDir, 'elevator-highrise.html');
  fs.writeFileSync(htmlPath, buildHTMLSet(variants, {
    title: 'High-rise elevator dispatch policies',
    subtitle: `${cfg.nFloors} floors, ${cfg.nElevators} shafts, dt=${cfg.stepSize}s, ${schedule.length} arrivals. Switch policy and decision authority.`,
  }));
  const jsonPath = path.join(outDir, 'elevator-highrise-results.json');
  fs.writeFileSync(jsonPath, JSON.stringify({
    schedule,
    mdpTunings: Object.fromEntries(Array.from(mdpTunings.entries()).map(([policy, tuning]) => [policy, summarizeMDPTuning(tuning)])),
    results,
  }, null, 2));
  console.log('');
  console.log(`# wrote ${htmlPath}`);
  console.log(`# wrote ${jsonPath}`);
}

if (require.main === module) {
  main().catch(err => {
    console.error(err instanceof Error ? err.stack ?? err.message : String(err));
    process.exit(1);
  });
}
