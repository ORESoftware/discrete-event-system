'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/traffic-flow.rs  (module des::general::traffic_flow)
// 1:1 file move. Small traffic DES: stationary grid/intersections/links + moving cars.
//
// Declarations → Rust:
//   interface TrafficNodeSpec/TrafficLinkSpec/TrafficSourceSpec/TrafficProblem/TrafficCarSnapshot/
//             TrafficLinkStats/TrafficTimeSample/TrafficSimulationResult -> structs (#[derive(Clone)])
//   type SignalAxis = 'EW'|'NS'|'ALL'                  -> enum
//   class TrafficCar extends BasicMovingEntity<Snapshot> implements Token -> struct + impl Token/entity trait
//   class IntersectionStation/RoadLinkStation/TrafficGridStation extends DESStation -> structs + impl trait
//   fn validateTrafficProblem/runTrafficSimulation/buildTrafficMaxFlowProblem/
//      buildDefaultTrafficProblem + private helpers (positiveModulo/hasDirectedPath) -> fns
//
// Conversion notes (file-specific):
//   - INJECT RNG: `mulberry32` seeds car arrivals/jitter -> `RandomSource` (SeededRandom).
//   - the grid owns many maps: `Map<number, ...>` (nodes/intersections/outgoing) ->
//     `HashMap<usize, _>`, `Map<string, ...>` (links/reservations/cache) -> `HashMap<String, _>`;
//     reachability uses `Set<number>` -> `HashSet<usize>`. Iteration order is N/A.
//   - links own continuous car positions/speeds -> `Vec<f64>` fields; prefer arena indices over
//     `Rc<RefCell>` for the grid↔link↔car graph.
//   - depends on max-flow.ts (MaxFlowProblem/solveMaxFlow) and entity-moving -> use crate::... paths.
//   - validate* throws -> `panic!` (invariant) or `Result`.
// =============================================================================
// general/traffic-flow.ts -- small continuous-time-ish traffic DES.
//
// The modelling choice mirrors the elevator lesson in this repository:
// the things with persistent state and local rules are stationary entities.
//
//   - TrafficGridStation: stationary coordinator for the road network.
//   - IntersectionStation: stationary signal controller.
//   - RoadLinkStation: stationary road segment holding continuous car
//     positions, speeds, exit credits, and occupancy statistics.
//   - TrafficCar: moving entity flowing Source -> Link -> Intersection ->
//     Link -> Sink.
//
// Cars interact without extending the base framework: road links sort their
// resident cars by position and enforce a continuous car-following gap. The
// grid knows every link, signal, and car, so it mediates downstream capacity
// and signal blocking. This is deliberately small (<300 cars by default) but
// captures the important DES-vs-traffic idea: interactions live at stations.
// =============================================================================

import {BasicMovingEntity} from '../entity-moving/moving';
import {DESStation, Token, assertNoValidationFailures, intrinsicCheck, runIterativeDES} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';
import {MaxFlowProblem, solveMaxFlow} from './max-flow';

export interface TrafficNodeSpec {
  id: number;
  name: string;
  x: number;
  y: number;
  signalOffsetSec?: number;
}

export interface TrafficLinkSpec {
  id: string;
  from: number;
  to: number;
  lengthM: number;
  speedLimitMps: number;
  capacity?: number;
  dischargePerMin?: number;
}

export interface TrafficSourceSpec {
  id: string;
  node: number;
  destNode: number;
  ratePerMin: number;
  maxGenerated?: number;
  startSec?: number;
  endSec?: number;
}

export interface TrafficProblem {
  nodes: TrafficNodeSpec[];
  links: TrafficLinkSpec[];
  sources: TrafficSourceSpec[];
  durationSec: number;
  dtSec: number;
  maxCars: number;
  minGapM: number;
  accelMps2: number;
  signalCycleSec: number;
  drainAfterSourcesSec?: number;
  seed?: number;
}

export interface TrafficCarSnapshot {
  carId: number;
  originNode: number;
  destNode: number;
  birthTimeSec: number;
  exitTimeSec?: number;
  currentLinkId?: string;
  positionM: number;
  speedMps: number;
}

export class TrafficCar extends BasicMovingEntity<TrafficCarSnapshot> implements Token {
  constructor(
    readonly carId: number,
    readonly originNode: number,
    readonly destNode: number,
    readonly birthTimeSec: number,
  ) {
    super();
  }

  exitTimeSec?: number;
  currentLinkId?: string;
  positionM = 0;
  speedMps = 0;

  snapshot(): TrafficCarSnapshot {
    return {
      carId: this.carId,
      originNode: this.originNode,
      destNode: this.destNode,
      birthTimeSec: this.birthTimeSec,
      exitTimeSec: this.exitTimeSec,
      currentLinkId: this.currentLinkId,
      positionM: this.positionM,
      speedMps: this.speedMps,
    };
  }

  override getValue(): {id: string; value: TrafficCarSnapshot} {
    return {id: this.id, value: this.snapshot()};
  }

  override runTimeStep(): void {
    // Cars are passive movables; RoadLinkStation advances their kinematics.
  }
}

export type SignalAxis = 'EW' | 'NS' | 'ALL';

export class IntersectionStation extends DESStation {
  private readonly controlled: boolean;
  private readonly cycleSec: number;
  private readonly offsetSec: number;

  constructor(readonly spec: TrafficNodeSpec, degree: number, cycleSec: number) {
    super(`intersection-${spec.id}`);
    this.controlled = degree > 2;
    this.cycleSec = cycleSec;
    this.offsetSec = spec.signalOffsetSec ?? 0;
  }

  axisAt(timeSec: number): SignalAxis {
    if (!this.controlled) return 'ALL';
    const phase = positiveModulo(timeSec + this.offsetSec, this.cycleSec);
    return phase < this.cycleSec / 2 ? 'EW' : 'NS';
  }

  allows(axis: SignalAxis, timeSec: number): boolean {
    const active = this.axisAt(timeSec);
    return active === 'ALL' || active === axis;
  }

  override hasWork(): boolean { return false; }
  runTimeStep(): void {}
}

interface PendingExit {
  car: TrafficCar;
  fromLinkId: string;
  atNode: number;
}

interface MoveContext {
  timeSec: number;
  dtSec: number;
  minGapM: number;
  accelMps2: number;
  canLeave: (link: RoadLinkStation, car: TrafficCar) => boolean;
  reserveExit: (link: RoadLinkStation, car: TrafficCar) => void;
}

export class RoadLinkStation extends DESStation {
  readonly capacity: number;
  readonly dischargePerMin: number;
  readonly cars: TrafficCar[] = [];
  private exitCredit = 0;
  private occupancyArea = 0;
  private maxOccupancy = 0;
  private entered = 0;
  private exited = 0;

  constructor(readonly spec: TrafficLinkSpec, minGapM: number) {
    super(`link-${spec.id}`);
    this.capacity = spec.capacity ?? Math.max(1, Math.floor(spec.lengthM / minGapM));
    this.dischargePerMin = spec.dischargePerMin ?? 30;
  }

  canAcceptEntry(minGapM: number, reservedIncoming = 0): boolean {
    if (this.cars.length + reservedIncoming >= this.capacity) return false;
    if (reservedIncoming > 0) return false;
    for (const car of this.cars) if (car.positionM < minGapM) return false;
    return true;
  }

  insertAtEntry(car: TrafficCar): void {
    car.currentLinkId = this.spec.id;
    car.positionM = 0;
    car.speedMps = Math.min(car.speedMps, this.spec.speedLimitMps);
    this.cars.push(car);
    this.entered++;
  }

  step(ctx: MoveContext): PendingExit[] {
    this.exitCredit += this.dischargePerMin * ctx.dtSec / 60;
    this.cars.sort((a, b) => b.positionM - a.positionM);
    const exits: PendingExit[] = [];
    const survivors: TrafficCar[] = [];

    for (let i = 0; i < this.cars.length; i++) {
      const car = this.cars[i];
      const leader = i > 0 ? this.cars[i - 1] : null;
      const desiredSpeed = Math.min(this.spec.speedLimitMps, car.speedMps + ctx.accelMps2 * ctx.dtSec);
      let maxMove = desiredSpeed * ctx.dtSec;

      if (leader && !exits.some(e => e.car === leader)) {
        maxMove = Math.min(maxMove, Math.max(0, leader.positionM - ctx.minGapM - car.positionM));
      } else {
        const canLeave = this.exitCredit >= 1 && ctx.canLeave(this, car);
        if (!canLeave) {
          maxMove = Math.min(maxMove, Math.max(0, this.spec.lengthM - ctx.minGapM - car.positionM));
        }
      }

      const move = Math.max(0, maxMove);
      car.positionM += move;
      car.speedMps = move / ctx.dtSec;

      if (car.positionM >= this.spec.lengthM - 1e-9 && this.exitCredit >= 1 && ctx.canLeave(this, car)) {
        this.exitCredit -= 1;
        ctx.reserveExit(this, car);
        this.exited++;
        exits.push({car, fromLinkId: this.spec.id, atNode: this.spec.to});
      } else {
        car.positionM = Math.min(car.positionM, this.spec.lengthM - 1e-6);
        survivors.push(car);
      }
    }

    this.cars.length = 0;
    this.cars.push(...survivors);
    this.occupancyArea += this.cars.length * ctx.dtSec;
    this.maxOccupancy = Math.max(this.maxOccupancy, this.cars.length);
    return exits;
  }

  stats(durationSec: number): TrafficLinkStats {
    return {
      id: this.spec.id,
      from: this.spec.from,
      to: this.spec.to,
      capacity: this.capacity,
      entered: this.entered,
      exited: this.exited,
      finalOccupancy: this.cars.length,
      maxOccupancy: this.maxOccupancy,
      avgOccupancy: this.occupancyArea / Math.max(1, durationSec),
    };
  }

  override hasWork(): boolean { return this.cars.length > 0; }
  runTimeStep(): void {}
}

interface SourceState {
  spec: TrafficSourceSpec;
  pending: number;
  generated: number;
  blockedAttempts: number;
}

export interface TrafficLinkStats {
  id: string;
  from: number;
  to: number;
  capacity: number;
  entered: number;
  exited: number;
  finalOccupancy: number;
  maxOccupancy: number;
  avgOccupancy: number;
}

export interface TrafficTimeSample {
  t: number;
  activeCars: number;
  completedCars: number;
  generatedCars: number;
}

export interface TrafficSimulationResult {
  generatedCars: number;
  completedCars: number;
  activeCars: number;
  maxActiveCars: number;
  blockedSourceAttempts: number;
  meanTravelTimeSec: number;
  p95TravelTimeSec: number;
  meanSpeedMps: number;
  throughputPerHour: number;
  maxFlowUpperBoundPerMin: number;
  throughputVsMaxFlow: number;
  totalSimulatedSec: number;
  linkStats: TrafficLinkStats[];
  timeSeries: TrafficTimeSample[];
  invariantViolations: string[];
}

export class TrafficGridStation extends DESStation {
  private readonly p: TrafficProblem;
  private readonly rng: () => number;
  private readonly nodesById = new Map<number, TrafficNodeSpec>();
  private readonly intersections = new Map<number, IntersectionStation>();
  private readonly linksById = new Map<string, RoadLinkStation>();
  private readonly outgoing = new Map<number, RoadLinkStation[]>();
  private readonly incomingReservations = new Map<string, number>();
  private readonly sources: SourceState[];
  private readonly completed: TrafficCar[] = [];
  private readonly timeSeries: TrafficTimeSample[] = [];
  private readonly invariantViolations: string[] = [];
  private readonly nextLinkCache = new Map<string, string | null>();
  private nextCarId = 1;
  private timeSec = 0;
  private maxActiveCars = 0;
  private speedIntegral = 0;
  private speedSamples = 0;

  constructor(p: TrafficProblem) {
    super('traffic-grid');
    validateTrafficProblem(p);
    this.p = p;
    this.rng = mulberry32(p.seed ?? 1);
    for (const n of p.nodes) this.nodesById.set(n.id, n);
    const degree = new Map<number, number>();
    for (const n of p.nodes) degree.set(n.id, 0);
    for (const l of p.links) {
      degree.set(l.from, (degree.get(l.from) ?? 0) + 1);
      degree.set(l.to, (degree.get(l.to) ?? 0) + 1);
      const st = new RoadLinkStation(l, p.minGapM);
      this.linksById.set(l.id, st);
      const arr = this.outgoing.get(l.from) ?? [];
      arr.push(st);
      this.outgoing.set(l.from, arr);
    }
    for (const n of p.nodes) {
      this.intersections.set(n.id, new IntersectionStation(n, degree.get(n.id) ?? 0, p.signalCycleSec));
    }
    this.sources = p.sources.map(spec => ({spec, pending: 0, generated: 0, blockedAttempts: 0}));

    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic.conservation',
      group: 'traffic-intrinsic',
      predicate: st => st.generatedCars() === st.completed.length + st.activeCars(),
      expected: 'generated = completed + active',
      observedFn: st => `generated=${st.generatedCars()}, completed=${st.completed.length}, active=${st.activeCars()}`,
    }));
    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic.car-cap',
      group: 'traffic-intrinsic',
      predicate: st => st.maxActiveCars <= st.p.maxCars && st.maxActiveCars < 300,
      expected: 'max active cars below configured cap and below 300',
      observedFn: st => `maxActive=${st.maxActiveCars}, cap=${st.p.maxCars}`,
    }));
  }

  override hasWork(): boolean {
    const drainUntil = this.p.durationSec + (this.p.drainAfterSourcesSec ?? 300);
    return this.timeSec < this.p.durationSec || (this.activeCars() > 0 && this.timeSec < drainUntil);
  }

  runTimeStep(): void {
    this.incomingReservations.clear();
    if (this.timeSec < this.p.durationSec) this.injectSources();

    const pendingExits: PendingExit[] = [];
    for (const link of this.linksById.values()) {
      const exits = link.step({
        timeSec: this.timeSec,
        dtSec: this.p.dtSec,
        minGapM: this.p.minGapM,
        accelMps2: this.p.accelMps2,
        canLeave: (l, car) => this.canLeave(l, car),
        reserveExit: (l, car) => this.reserveExit(l, car),
      });
      pendingExits.push(...exits);
    }
    for (const ex of pendingExits) this.applyExit(ex);
    this.recordInvariants();
    this.recordStats();
    this.timeSec += this.p.dtSec;
  }

  buildResult(): TrafficSimulationResult {
    const travelTimes = this.completed.map(c => (c.exitTimeSec ?? this.timeSec) - c.birthTimeSec).sort((a, b) => a - b);
    const meanTravel = travelTimes.reduce((s, x) => s + x, 0) / Math.max(1, travelTimes.length);
    const p95 = travelTimes.length > 0 ? travelTimes[Math.min(travelTimes.length - 1, Math.floor(0.95 * (travelTimes.length - 1)))] : NaN;
    const maxFlow = solveMaxFlow(buildTrafficMaxFlowProblem(this.p)).maxFlow;
    const simulatedMinutes = Math.max(1e-9, this.p.durationSec / 60);
    const throughputPerMin = this.completed.length / simulatedMinutes;
    return {
      generatedCars: this.generatedCars(),
      completedCars: this.completed.length,
      activeCars: this.activeCars(),
      maxActiveCars: this.maxActiveCars,
      blockedSourceAttempts: this.sources.reduce((s, x) => s + x.blockedAttempts, 0),
      meanTravelTimeSec: meanTravel,
      p95TravelTimeSec: p95,
      meanSpeedMps: this.speedIntegral / Math.max(1, this.speedSamples),
      throughputPerHour: throughputPerMin * 60,
      maxFlowUpperBoundPerMin: maxFlow,
      throughputVsMaxFlow: throughputPerMin / Math.max(1e-9, maxFlow),
      totalSimulatedSec: this.timeSec,
      linkStats: [...this.linksById.values()].map(l => l.stats(this.timeSec)),
      timeSeries: this.timeSeries.slice(),
      invariantViolations: this.invariantViolations.slice(),
    };
  }

  private injectSources(): void {
    for (const src of this.sources) {
      const s = src.spec;
      if (this.timeSec < (s.startSec ?? 0) || this.timeSec > (s.endSec ?? this.p.durationSec)) continue;
      src.pending += s.ratePerMin * this.p.dtSec / 60;
      while (src.pending >= 1 - 1e-12) {
        if (s.maxGenerated !== undefined && src.generated >= s.maxGenerated) {
          src.pending = 0;
          break;
        }
        if (this.activeCars() >= this.p.maxCars) {
          src.blockedAttempts++;
          break;
        }
        const nextLink = this.nextLinkFrom(s.node, s.destNode);
        if (!nextLink) {
          src.blockedAttempts++;
          src.pending -= 1;
          continue;
        }
        const link = this.linksById.get(nextLink)!;
        const reserved = this.incomingReservations.get(link.spec.id) ?? 0;
        if (!link.canAcceptEntry(this.p.minGapM, reserved)) {
          src.blockedAttempts++;
          break;
        }
        const car = new TrafficCar(this.nextCarId++, s.node, s.destNode, this.timeSec);
        link.insertAtEntry(car);
        src.generated++;
        src.pending -= 1;
      }
    }
  }

  private canLeave(link: RoadLinkStation, car: TrafficCar): boolean {
    const node = this.intersections.get(link.spec.to)!;
    const axis = this.axisOf(link.spec);
    if (!node.allows(axis, this.timeSec)) return false;
    if (link.spec.to === car.destNode) return true;
    const nextLinkId = this.nextLinkFrom(link.spec.to, car.destNode);
    if (!nextLinkId) return false;
    const next = this.linksById.get(nextLinkId)!;
    const reserved = this.incomingReservations.get(nextLinkId) ?? 0;
    return next.canAcceptEntry(this.p.minGapM, reserved);
  }

  private reserveExit(link: RoadLinkStation, car: TrafficCar): void {
    if (link.spec.to === car.destNode) return;
    const next = this.nextLinkFrom(link.spec.to, car.destNode);
    if (!next) return;
    this.incomingReservations.set(next, (this.incomingReservations.get(next) ?? 0) + 1);
  }

  private applyExit(ex: PendingExit): void {
    const car = ex.car;
    if (ex.atNode === car.destNode) {
      car.exitTimeSec = this.timeSec + this.p.dtSec;
      car.currentLinkId = undefined;
      car.doFinish();
      this.completed.push(car);
      return;
    }
    const nextLinkId = this.nextLinkFrom(ex.atNode, car.destNode);
    if (!nextLinkId) {
      this.invariantViolations.push(`car ${car.carId} has no route from ${ex.atNode} to ${car.destNode}`);
      return;
    }
    const next = this.linksById.get(nextLinkId)!;
    next.insertAtEntry(car);
  }

  private nextLinkFrom(node: number, dest: number): string | null {
    const key = `${node}->${dest}`;
    if (this.nextLinkCache.has(key)) return this.nextLinkCache.get(key)!;
    const result = this.shortestNextLink(node, dest);
    this.nextLinkCache.set(key, result);
    return result;
  }

  private shortestNextLink(node: number, dest: number): string | null {
    if (node === dest) return null;
    const dist = new Map<number, number>();
    const prevLink = new Map<number, string>();
    const prevNode = new Map<number, number>();
    const unsettled = new Set<number>(this.p.nodes.map(n => n.id));
    for (const n of unsettled) dist.set(n, Infinity);
    dist.set(node, 0);
    while (unsettled.size > 0) {
      let u: number | null = null;
      let best = Infinity;
      for (const n of unsettled) {
        const d = dist.get(n) ?? Infinity;
        if (d < best) { best = d; u = n; }
      }
      if (u === null || !Number.isFinite(best)) break;
      unsettled.delete(u);
      if (u === dest) break;
      for (const link of this.outgoing.get(u) ?? []) {
        const v = link.spec.to;
        if (!unsettled.has(v)) continue;
        const nd = best + link.spec.lengthM;
        if (nd < (dist.get(v) ?? Infinity)) {
          dist.set(v, nd);
          prevNode.set(v, u);
          prevLink.set(v, link.spec.id);
        }
      }
    }
    if (!prevLink.has(dest)) return null;
    let cur = dest;
    let firstLink = prevLink.get(cur)!;
    while ((prevNode.get(cur) ?? node) !== node) {
      cur = prevNode.get(cur)!;
      firstLink = prevLink.get(cur)!;
    }
    return firstLink;
  }

  private axisOf(link: TrafficLinkSpec): SignalAxis {
    const a = this.nodesById.get(link.from)!;
    const b = this.nodesById.get(link.to)!;
    return Math.abs(b.x - a.x) >= Math.abs(b.y - a.y) ? 'EW' : 'NS';
  }

  private activeCars(): number {
    let n = 0;
    for (const l of this.linksById.values()) n += l.cars.length;
    return n;
  }

  private generatedCars(): number {
    return this.sources.reduce((s, x) => s + x.generated, 0);
  }

  private recordStats(): void {
    const active = this.activeCars();
    this.maxActiveCars = Math.max(this.maxActiveCars, active);
    for (const link of this.linksById.values()) {
      for (const car of link.cars) {
        this.speedIntegral += car.speedMps;
        this.speedSamples++;
      }
    }
    if (Math.abs((this.timeSec / this.p.dtSec) % 10) < 1e-9) {
      this.timeSeries.push({
        t: this.timeSec,
        activeCars: active,
        completedCars: this.completed.length,
        generatedCars: this.generatedCars(),
      });
    }
  }

  private recordInvariants(): void {
    for (const link of this.linksById.values()) {
      const cars = link.cars.slice().sort((a, b) => a.positionM - b.positionM);
      if (cars.length > link.capacity) {
        this.invariantViolations.push(`${link.spec.id}: occupancy ${cars.length} exceeds cap ${link.capacity}`);
      }
      for (let i = 0; i < cars.length; i++) {
        const c = cars[i];
        if (c.positionM < -1e-6 || c.positionM > link.spec.lengthM + 1e-6) {
          this.invariantViolations.push(`${link.spec.id}: car ${c.carId} out of bounds at ${c.positionM}`);
        }
        if (i > 0 && cars[i].positionM - cars[i - 1].positionM < this.p.minGapM - 1e-6) {
          this.invariantViolations.push(`${link.spec.id}: car gap violation ${cars[i - 1].carId}/${cars[i].carId}`);
        }
      }
    }
  }
}

export function validateTrafficProblem(p: TrafficProblem): void {
  Preconditions.nonEmpty('traffic-flow', 'nodes', p.nodes);
  Preconditions.nonEmpty('traffic-flow', 'links', p.links);
  Preconditions.nonEmpty('traffic-flow', 'sources', p.sources);
  Preconditions.positive('traffic-flow', 'durationSec', p.durationSec);
  Preconditions.positive('traffic-flow', 'dtSec', p.dtSec);
  Preconditions.integerInRange('traffic-flow', 'maxCars', p.maxCars, 1, 299);
  Preconditions.positive('traffic-flow', 'minGapM', p.minGapM);
  Preconditions.positive('traffic-flow', 'accelMps2', p.accelMps2);
  Preconditions.positive('traffic-flow', 'signalCycleSec', p.signalCycleSec);
  const nodeIds = new Set<number>();
  for (const n of p.nodes) {
    Preconditions.check('traffic-flow', `node ${n.id}`, 'be unique', !nodeIds.has(n.id), n.id);
    nodeIds.add(n.id);
  }
  const linkIds = new Set<string>();
  for (const l of p.links) {
    Preconditions.check('traffic-flow', `link ${l.id}`, 'be unique', !linkIds.has(l.id), l.id);
    linkIds.add(l.id);
    Preconditions.check('traffic-flow', `${l.id}.from`, 'reference a node', nodeIds.has(l.from), l.from);
    Preconditions.check('traffic-flow', `${l.id}.to`, 'reference a node', nodeIds.has(l.to), l.to);
    Preconditions.positive('traffic-flow', `${l.id}.lengthM`, l.lengthM);
    Preconditions.positive('traffic-flow', `${l.id}.speedLimitMps`, l.speedLimitMps);
  if (l.capacity !== undefined) Preconditions.integerInRange('traffic-flow', `${l.id}.capacity`, l.capacity, 1, 299);
    if (l.dischargePerMin !== undefined) Preconditions.positive('traffic-flow', `${l.id}.dischargePerMin`, l.dischargePerMin);
  }
  const outgoing = new Map<number, number[]>();
  for (const l of p.links) outgoing.set(l.from, [...(outgoing.get(l.from) ?? []), l.to]);
  for (const s of p.sources) {
    Preconditions.check('traffic-flow', `${s.id}.node`, 'reference a node', nodeIds.has(s.node), s.node);
    Preconditions.check('traffic-flow', `${s.id}.destNode`, 'reference a node', nodeIds.has(s.destNode), s.destNode);
    Preconditions.check('traffic-flow', `${s.id}.node != destNode`, 'hold', s.node !== s.destNode, [s.node, s.destNode]);
    Preconditions.nonNegative('traffic-flow', `${s.id}.ratePerMin`, s.ratePerMin);
    if (s.maxGenerated !== undefined) Preconditions.integerInRange('traffic-flow', `${s.id}.maxGenerated`, s.maxGenerated, 0, 1e6);
    if (s.startSec !== undefined) Preconditions.nonNegative('traffic-flow', `${s.id}.startSec`, s.startSec);
    if (s.endSec !== undefined) Preconditions.nonNegative('traffic-flow', `${s.id}.endSec`, s.endSec);
    if (s.startSec !== undefined && s.endSec !== undefined) {
      Preconditions.check('traffic-flow', `${s.id}.startSec <= endSec`, 'hold', s.startSec <= s.endSec, [s.startSec, s.endSec]);
    }
    Preconditions.check('traffic-flow', `${s.id}.route`, 'exist in directed link graph', hasDirectedPath(s.node, s.destNode, outgoing), [s.node, s.destNode]);
  }
  if (p.drainAfterSourcesSec !== undefined) Preconditions.nonNegative('traffic-flow', 'drainAfterSourcesSec', p.drainAfterSourcesSec);
}

export function runTrafficSimulation(p: TrafficProblem): TrafficSimulationResult {
  const grid = new TrafficGridStation(p);
  const maxTicks = Math.ceil((p.durationSec + (p.drainAfterSourcesSec ?? 300)) / p.dtSec) + 5;
  const summary = runIterativeDES([grid], {shuffle: false, maxTicks});
  assertNoValidationFailures(summary, 'traffic');
  return grid.buildResult();
}

export function buildTrafficMaxFlowProblem(p: TrafficProblem): MaxFlowProblem {
  validateTrafficProblem(p);
  const superSource = p.nodes.length;
  const superSink = p.nodes.length + 1;
  const maxDemand = p.sources.reduce((s, x) => s + x.ratePerMin, 0);
  const sinkNodes = new Set(p.sources.map(s => s.destNode));
  return {
    numNodes: p.nodes.length + 2,
    source: superSource,
    sink: superSink,
    edges: [
      ...p.sources.map(s => ({from: superSource, to: s.node, capacity: s.ratePerMin, name: `src-${s.id}`})),
      ...p.links.map(l => ({from: l.from, to: l.to, capacity: l.dischargePerMin ?? 30, name: l.id})),
      ...[...sinkNodes].map(n => ({from: n, to: superSink, capacity: Math.max(1, maxDemand), name: `sink-${n}`})),
    ],
  };
}

export function buildDefaultTrafficProblem(): TrafficProblem {
  const nodes: TrafficNodeSpec[] = [
    {id: 0, name: 'W', x: 0, y: 1},
    {id: 1, name: 'C', x: 1, y: 1},
    {id: 2, name: 'E', x: 2, y: 1},
    {id: 3, name: 'N', x: 1, y: 2},
    {id: 4, name: 'S', x: 1, y: 0},
  ];
  const mk = (id: string, from: number, to: number): TrafficLinkSpec => ({
    id, from, to, lengthM: 180, speedLimitMps: 13.4, capacity: 24, dischargePerMin: 30,
  });
  return {
    nodes,
    links: [
      mk('W-C', 0, 1), mk('C-W', 1, 0),
      mk('C-E', 1, 2), mk('E-C', 2, 1),
      mk('N-C', 3, 1), mk('C-N', 1, 3),
      mk('S-C', 4, 1), mk('C-S', 1, 4),
    ],
    sources: [
      {id: 'west-to-east', node: 0, destNode: 2, ratePerMin: 12, maxGenerated: 90},
      {id: 'north-to-south', node: 3, destNode: 4, ratePerMin: 9, maxGenerated: 70},
      {id: 'south-to-east', node: 4, destNode: 2, ratePerMin: 6, maxGenerated: 50},
      {id: 'east-to-west', node: 2, destNode: 0, ratePerMin: 5, maxGenerated: 30},
    ],
    durationSec: 600,
    dtSec: 1,
    maxCars: 240,
    minGapM: 7.5,
    accelMps2: 2.0,
    signalCycleSec: 60,
    drainAfterSourcesSec: 420,
    seed: 7,
  };
}

function positiveModulo(x: number, m: number): number {
  return ((x % m) + m) % m;
}

function hasDirectedPath(source: number, sink: number, outgoing: Map<number, number[]>): boolean {
  const seen = new Set<number>([source]);
  const q = [source];
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi];
    if (u === sink) return true;
    for (const v of outgoing.get(u) ?? []) {
      if (seen.has(v)) continue;
      seen.add(v);
      q.push(v);
    }
  }
  return false;
}
