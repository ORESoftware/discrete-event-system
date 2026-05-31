'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/smart-traffic-flow.rs  (module des::general::smart_traffic_flow)
// 1:1 file move. Traffic flow where cars are self-stepping smart movables (faults/accidents).
//
// Declarations → Rust:
//   type SmartTrafficParams = TrafficParams & {...} -> struct composing/extending TrafficParams (no `&`)
//   type SmartTrafficFaultMode = 'accelerate-too-fast'|... -> enum
//   interface SmartTraffic* (CarSnapshot/TraceRow/Accident/CellStats/ExecutionStats/Result/...) -> structs
//   class SmartTrafficCellStation/SmartTrafficWorldStation extends DESStation -> structs + impl trait
//   class SmartTrafficCar extends SmartMovable      -> struct + impl SmartMovable trait
//   fn runSmartTrafficFlow + private validators/helpers -> fns
//
// Conversion notes (file-specific):
//   - INTERSECTION TYPE `TrafficParams & {...}`: Rust has no `&` on structs — embed TrafficParams
//     as a field or duplicate the fields in a new struct.
//   - INJECT RNG: `mulberry32` seeds car behaviour/fault injection -> `RandomSource` (SeededRandom).
//   - reuses network-flow.ts types (TrafficNetwork/Lane/Node/Signal/Params/...) -> use crate::...::network_flow.
//   - world owns a string-keyed grid (cells/lanes/cars) -> `HashMap<String, _>` + car ids `u64`;
//     cars propose moves committed at a tick barrier — model with a proposal buffer `Vec`.
//   - `logger?: OptimizationLogger` -> `Option<&dyn OptimizationLogger>`; validators throw -> panic/Result.
// =============================================================================
// Smart-movable traffic flow.
//
// This variant keeps cars as movables, not stations, but gives each car its own
// runTimeStep(). A fixed pool of SmartTrafficCar participants is passed to the
// iterative DES runner, so active cars are shuffled by the same Fisher-Yates loop
// as stations. The world station owns the shared grid and commits car proposals
// at the tick barrier.
// =============================================================================

import {
  DESStation,
  IterativeRunSummary,
  SmartMovable,
  intrinsicCheck,
  runIterativeDES,
  ValidationCheck,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';
import {
  buildFiveIntersectionTrafficNetwork,
  OptimizationLogger,
  TrafficLane,
  TrafficNetwork,
  TrafficNode,
  TrafficParams,
  TrafficScheduledTrip,
  TrafficSignal,
  TrafficSignalPhase,
  TrafficSource,
} from './network-flow';
import {discreteConvolveSelf, meanFromPMF, normalizePMF, sampleFromPMF, varianceFromPMF} from './random-variables';

const DRIVER_TRAIT_PMF = normalizePMF(discreteConvolveSelf([1, 1, 1, 1, 1, 1, 1], 4));
const DRIVER_TRAIT_MEAN = meanFromPMF(DRIVER_TRAIT_PMF);
const DRIVER_TRAIT_STD = Math.sqrt(varianceFromPMF(DRIVER_TRAIT_PMF));

export type SmartTrafficParams = TrafficParams & {
  /** Number of potential smart movable actors pre-registered with the runner. */
  smartCarPoolSize?: number;
  /** Seed used by the runner's Fisher-Yates actor/station shuffle. */
  actorShuffleSeed?: number;
  /** Dimensionless multiplier for behavior-derived fault hazard. */
  accidentRiskScale?: number;
  /** Backward-compatible alias for accidentRiskScale. */
  accidentProbability?: number;
  /** Extra acceleration applied during a control fault. */
  accidentAccelBoostMps2?: number;
  /** How long a driver/control fault persists once triggered. */
  accidentFaultDurationSec?: number;
  /** +/- uniform spread around each driver's following-distance preference multiplier of 1. */
  distancePreferenceSpread?: number;
  /** +/- uniform spread around each driver's startup hesitation/clearance preference multiplier of 1. */
  startPreferenceSpread?: number;
  /** How long accident flashes should remain visible in animation. */
  accidentFlashSeconds?: number;
};

export type SmartTrafficFaultMode = 'accelerate-too-fast' | 'brake-too-slow' | 'speeding';

export interface SmartTrafficCarSnapshot {
  id: number;
  actorId: string;
  slotIndex: number;
  laneId: string;
  positionM: number;
  speedMps: number;
  accelerationMps2: number;
  jerkMps3: number;
  targetAccelerationMps2: number;
  route: string[];
  routeIndex: number;
  destinationSinkId: string;
  createdAtSec: number;
  waitSec: number;
  distancePreference: number;
  startPreference: number;
  startReadySinceSec?: number;
  gridCellIds: string[];
  gridCellCount: number;
  leaderId?: number;
  leaderGapM?: number;
  runCount: number;
  accidentCount: number;
  faultMode?: SmartTrafficFaultMode;
  faultUntilSec?: number;
  lastRunTick?: number;
}

export interface SmartTrafficTraceRow {
  tick: number;
  timeSec: number;
  activeCars: number;
  scheduledSmartCars: number;
  smartMovableRuns: number;
  entered: number;
  exited: number;
  crashed: number;
  meanSpeedMps: number;
  meanTravelTimeSec: number;
  queueLength: number;
  laneOccupancy: Record<string, number>;
  activeGridCells: number;
  signalPhases: Record<string, string>;
  actorRunOrder: string[];
  accidents: SmartTrafficAccident[];
  cars: SmartTrafficCarSnapshot[];
}

export interface SmartTrafficAccident {
  tick: number;
  timeSec: number;
  laneId: string;
  positionM: number;
  cellId: string;
  carId: number;
  actorId: string;
  otherCarId: number;
  otherActorId: string;
  speedMps: number;
  faultMode: SmartTrafficFaultMode;
  riskScore: number;
  hazardPerSec: number;
  reason: 'body-contact-rear-end';
}

export interface SmartTrafficCellStats {
  cellSizeM: number;
  laneWidthM: number;
  carWidthM: number;
  activeCells: number;
  createdCellStations: number;
  accidentCellStations: number;
  accidentCellHits: number;
  maxCellOccupancy: number;
}

export interface SmartTrafficExecutionStats {
  participantCount: number;
  smartMovableCount: number;
  worldStationId: string;
  shuffledByRunner: boolean;
  actorShuffleSeed: number;
  totalSmartMovableRuns: number;
  maxSmartMovableRunsPerTick: number;
}

export interface SmartTrafficResult {
  params: SmartTrafficParams;
  network: TrafficNetwork;
  trace: SmartTrafficTraceRow[];
  finalCars: SmartTrafficCarSnapshot[];
  entered: number;
  exited: number;
  crashed: number;
  dropped: number;
  meanTravelTimeSec: number;
  meanSpeedMps: number;
  maxActiveCars: number;
  cellStats: SmartTrafficCellStats;
  execution: SmartTrafficExecutionStats;
  runSummary: IterativeRunSummary;
  accidents: SmartTrafficAccident[];
  validation: ValidationCheck[];
}

interface SmartTrafficSpatialIndex {
  byCell: Map<string, Set<number>>;
  activeCellCount: number;
}

interface SmartCarProposal {
  actor: SmartTrafficCar;
  speedMps: number;
  positionM: number;
  accelerationMps2: number;
  jerkMps3: number;
  targetAccelerationMps2: number;
  leaderId?: number;
  leaderGapM?: number;
  controlFault?: boolean;
  faultMode?: SmartTrafficFaultMode;
  faultUntilSec?: number;
  startReadySinceSec?: number;
  riskScore?: number;
  hazardPerSec?: number;
}

interface SmartTrafficCellBounds {
  laneId: string;
  longitudinalIndex: number;
  lateralIndex: number;
  x0M: number;
  x1M: number;
  y0M: number;
  y1M: number;
}

interface SmartTrafficKinematicSample {
  timeSec: number;
  laneId: string;
  positionM: number;
  speedMps: number;
  accelerationMps2: number;
}

export class SmartTrafficCellStation extends DESStation {
  readonly carIds = new Set<number>();
  readonly accidentIds: string[] = [];

  constructor(readonly bounds: SmartTrafficCellBounds) {
    super(`smart-traffic-cell-${bounds.laneId}-${bounds.longitudinalIndex}-${bounds.lateralIndex}`);
  }

  clearOccupancy(): void {
    this.carIds.clear();
  }

  occupy(carId: number): void {
    this.carIds.add(carId);
  }

  recordAccident(accident: SmartTrafficAccident): void {
    this.accidentIds.push(`${accident.tick}:${accident.carId}->${accident.otherCarId}`);
  }

  override hasWork(): boolean {
    return false;
  }

  runTimeStep(): void {}
}

export class SmartTrafficCar extends SmartMovable {
  carId = 0;
  laneId = '';
  positionM = 0;
  speedMps = 0;
  accelerationMps2 = 0;
  jerkMps3 = 0;
  targetAccelerationMps2 = 0;
  route: string[] = [];
  routeIndex = 0;
  destinationSinkId = '';
  createdAtSec = 0;
  waitSec = 0;
  distancePreference = 1;
  startPreference = 1;
  startReadySinceSec?: number;
  gridCellIds: string[] = [];
  gridCellCount = 0;
  leaderId?: number;
  leaderGapM?: number;
  history: SmartTrafficKinematicSample[] = [];
  accidents: SmartTrafficAccident[] = [];
  faultMode?: SmartTrafficFaultMode;
  faultUntilSec = 0;
  runCount = 0;
  lastRunTick?: number;

  constructor(readonly slotIndex: number, private readonly world: SmartTrafficWorldStation) {
    super(`smart-car-${slotIndex}`);
  }

  assign(opts: {
    carId: number;
    laneId: string;
    route: string[];
    destinationSinkId: string;
    createdAtSec: number;
    initialSpeedMps: number;
    distancePreference: number;
    startPreference: number;
  }): void {
    this.activate();
    this.carId = opts.carId;
    this.laneId = opts.laneId;
    this.positionM = 0;
    this.speedMps = opts.initialSpeedMps;
    this.accelerationMps2 = 0;
    this.jerkMps3 = 0;
    this.targetAccelerationMps2 = 0;
    this.route = opts.route.slice();
    this.routeIndex = 0;
    this.destinationSinkId = opts.destinationSinkId;
    this.createdAtSec = opts.createdAtSec;
    this.waitSec = 0;
    this.distancePreference = opts.distancePreference;
    this.startPreference = opts.startPreference;
    this.startReadySinceSec = undefined;
    this.gridCellIds = [];
    this.gridCellCount = 0;
    this.leaderId = undefined;
    this.leaderGapM = undefined;
    this.history = [];
    this.accidents = [];
    this.faultMode = undefined;
    this.faultUntilSec = 0;
    this.runCount = 0;
    this.lastRunTick = undefined;
    this.world.recordHistory(this, opts.createdAtSec);
  }

  recordAccident(accident: SmartTrafficAccident): void {
    this.accidents.push(accident);
  }

  retire(): void {
    this.deactivate();
    this.laneId = '';
    this.route = [];
    this.gridCellIds = [];
    this.gridCellCount = 0;
    this.distancePreference = 1;
    this.startPreference = 1;
    this.startReadySinceSec = undefined;
    this.leaderId = undefined;
    this.leaderGapM = undefined;
    this.faultMode = undefined;
    this.faultUntilSec = 0;
  }

  override hasWork(): boolean {
    return this.isActive() && this.world.acceptsSmartMovableRuns() && this.lastRunTick !== this.world.currentTick();
  }

  runTimeStep(): void {
    if (!this.hasWork()) return;
    this.lastRunTick = this.world.currentTick();
    this.runCount++;
    this.world.recordActorRun(this);
    this.world.proposeCarStep(this);
  }

  snapshot(): SmartTrafficCarSnapshot {
    return {
      id: this.carId,
      actorId: this.id,
      slotIndex: this.slotIndex,
      laneId: this.laneId,
      positionM: this.positionM,
      speedMps: this.speedMps,
      accelerationMps2: this.accelerationMps2,
      jerkMps3: this.jerkMps3,
      targetAccelerationMps2: this.targetAccelerationMps2,
      route: this.route.slice(),
      routeIndex: this.routeIndex,
      destinationSinkId: this.destinationSinkId,
      createdAtSec: this.createdAtSec,
      waitSec: this.waitSec,
      distancePreference: this.distancePreference,
      startPreference: this.startPreference,
      startReadySinceSec: this.startReadySinceSec,
      gridCellIds: this.gridCellIds.slice(),
      gridCellCount: this.gridCellCount,
      leaderId: this.leaderId,
      leaderGapM: this.leaderGapM,
      runCount: this.runCount,
      accidentCount: this.accidents.length,
      faultMode: this.faultMode,
      faultUntilSec: this.faultUntilSec > 0 ? this.faultUntilSec : undefined,
      lastRunTick: this.lastRunTick,
    };
  }
}

export class SmartTrafficWorldStation extends DESStation {
  readonly carActors: SmartTrafficCar[];
  private readonly network: TrafficNetwork;
  private readonly nodes = new Map<string, TrafficNode>();
  private readonly lanes = new Map<string, TrafficLane>();
  private readonly signalByNode = new Map<string, TrafficSignal>();
  private readonly routes = new Map<string, string[]>();
  private readonly sourceAccumulators = new Map<string, number>();
  private readonly scheduledTrips: TrafficScheduledTrip[];
  private readonly cellStations = new Map<string, SmartTrafficCellStation>();
  private readonly activeCellIds = new Set<string>();
  private readonly rng: () => number;
  private spatial: SmartTrafficSpatialIndex = {byCell: new Map(), activeCellCount: 0};
  private readonly proposals = new Map<number, SmartCarProposal>();
  private readonly currentActorRunOrder: string[] = [];
  private readonly completedTravelTimes: number[] = [];
  private readonly accidents: SmartTrafficAccident[] = [];
  private accidentsThisTick: SmartTrafficAccident[] = [];
  private nextScheduledTripIndex = 0;
  private nextCarId = 1;
  private tick = 0;
  private timeSec = 0;
  private entered = 0;
  private exited = 0;
  private crashed = 0;
  private dropped = 0;
  private speedSum = 0;
  private speedSamples = 0;
  private maxActiveCars = 0;
  private maxCellOccupancy = 0;
  private scheduledSmartCars = 0;
  private totalSmartMovableRuns = 0;
  private maxSmartMovableRunsPerTick = 0;
  readonly trace: SmartTrafficTraceRow[] = [];

  constructor(readonly params: SmartTrafficParams, private readonly options: {logger?: OptimizationLogger} = {}) {
    super('smart-traffic-world');
    this.network = params.network ?? buildFiveIntersectionTrafficNetwork();
    this.rng = mulberry32(params.seed);
    this.scheduledTrips = (params.scheduledTrips ?? []).slice().sort((a, b) => a.departSec - b.departSec);
    for (const node of this.network.nodes) this.nodes.set(node.id, node);
    for (const lane of this.network.lanes) this.lanes.set(lane.id, lane);
    for (const signal of this.network.signals ?? []) this.signalByNode.set(signal.nodeId, signal);
    for (const source of this.network.sources) this.sourceAccumulators.set(source.id, 0);
    const poolSize = params.smartCarPoolSize ?? params.maxCars;
    this.carActors = Array.from({length: poolSize}, (_, i) => new SmartTrafficCar(i, this));
    this.precomputeRoutes();
    this.addValidator(intrinsicCheck<SmartTrafficWorldStation>({
      name: 'smart-traffic-active-under-cap',
      predicate: s => s.maxActiveCars <= s.params.maxCars && s.maxActiveCars < 300,
      observedFn: s => `maxActive=${s.maxActiveCars} cap=${s.params.maxCars}`,
      expected: 'active smart cars never exceed maxCars or 299',
      group: 'smart-traffic-flow',
    }));
    this.addValidator(intrinsicCheck<SmartTrafficWorldStation>({
      name: 'smart-traffic-conservation',
      predicate: s => s.entered === s.exited + s.crashed + s.activeCars().length,
      observedFn: s => `entered=${s.entered} exited=${s.exited} crashed=${s.crashed} active=${s.activeCars().length}`,
      expected: 'entered = exited + crashed + active',
      group: 'smart-traffic-flow',
    }));
    this.addValidator(intrinsicCheck<SmartTrafficWorldStation>({
      name: 'smart-traffic-no-collisions',
      predicate: s => s.minimumBodyGap() >= -1e-7,
      observedFn: s => s.minimumBodyGap().toFixed(6),
      expected: 'same-lane smart cars do not physically overlap',
      group: 'smart-traffic-flow',
    }));
    this.addValidator(intrinsicCheck<SmartTrafficWorldStation>({
      name: 'smart-traffic-actor-run-coverage',
      predicate: s => s.trace.every(row => row.scheduledSmartCars === row.smartMovableRuns),
      observedFn: s => {
        const bad = s.trace.find(row => row.scheduledSmartCars !== row.smartMovableRuns);
        return bad ? `tick=${bad.tick} scheduled=${bad.scheduledSmartCars} ran=${bad.smartMovableRuns}` : 'all active smart cars ran';
      },
      expected: 'every active smart movable receives runTimeStep once per tick',
      group: 'smart-traffic-flow',
    }));
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.check('SmartTrafficWorldStation', 'network', 'be provided by builtin or network', p.builtin === 'five-intersection' || p.network !== undefined, p);
    Preconditions.positive('SmartTrafficWorldStation', 'durationSec', p.durationSec);
    Preconditions.positive('SmartTrafficWorldStation', 'dtSec', p.dtSec);
    Preconditions.check('SmartTrafficWorldStation', 'dtSec', 'be <= 5 seconds', p.dtSec <= 5, p.dtSec);
    Preconditions.integer('SmartTrafficWorldStation', 'seed', p.seed);
    Preconditions.integerInRange('SmartTrafficWorldStation', 'maxCars', p.maxCars, 1, 299);
    Preconditions.integerInRange('SmartTrafficWorldStation', 'smartCarPoolSize', p.smartCarPoolSize ?? p.maxCars, p.maxCars, 10000);
    if (p.actorShuffleSeed !== undefined) Preconditions.integer('SmartTrafficWorldStation', 'actorShuffleSeed', p.actorShuffleSeed);
    if (p.carLengthM !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'carLengthM', p.carLengthM);
    if (p.carWidthM !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'carWidthM', p.carWidthM);
    if (p.laneWidthM !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'laneWidthM', p.laneWidthM);
    if (p.minGapM !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'minGapM', p.minGapM);
    if (p.maxAccelMps2 !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'maxAccelMps2', p.maxAccelMps2);
    if (p.maxDecelMps2 !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'maxDecelMps2', p.maxDecelMps2);
    if (p.maxJerkMps3 !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'maxJerkMps3', p.maxJerkMps3);
    if (p.reactionTimeSec !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'reactionTimeSec', p.reactionTimeSec);
    if (p.timeHeadwaySec !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'timeHeadwaySec', p.timeHeadwaySec);
    if (p.gridCellSizeM !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'gridCellSizeM', p.gridCellSizeM);
    if (p.spawnRateMultiplier !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'spawnRateMultiplier', p.spawnRateMultiplier);
    if (p.accidentRiskScale !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'accidentRiskScale', p.accidentRiskScale);
    if (p.accidentProbability !== undefined) Preconditions.inRange('SmartTrafficWorldStation', 'accidentProbability', p.accidentProbability, 0, 1);
    if (p.accidentAccelBoostMps2 !== undefined) Preconditions.nonNegative('SmartTrafficWorldStation', 'accidentAccelBoostMps2', p.accidentAccelBoostMps2);
    if (p.accidentFaultDurationSec !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'accidentFaultDurationSec', p.accidentFaultDurationSec);
    if (p.distancePreferenceSpread !== undefined) Preconditions.inRange('SmartTrafficWorldStation', 'distancePreferenceSpread', p.distancePreferenceSpread, 0, 1.5);
    if (p.startPreferenceSpread !== undefined) Preconditions.inRange('SmartTrafficWorldStation', 'startPreferenceSpread', p.startPreferenceSpread, 0, 1.5);
    if (p.accidentFlashSeconds !== undefined) Preconditions.positive('SmartTrafficWorldStation', 'accidentFlashSeconds', p.accidentFlashSeconds);
    Preconditions.check('SmartTrafficWorldStation', 'carWidthM', 'fit within laneWidthM', this.carWidthM() <= this.laneWidthM(), {carWidthM: this.carWidthM(), laneWidthM: this.laneWidthM()});
    validateSmartTrafficNetwork(this.network);
    validateSmartTrafficScheduledTrips(this.network, p.scheduledTrips ?? [], p.durationSec);
  }

  beginRun(): void {
    this.prepareTick();
  }

  override hasWork(): boolean {
    return this.timeSec < this.params.durationSec - 1e-9;
  }

  runTimeStep(): void {
    // The world is deliberately a no-op participant in the shuffled order.
    // Tick-barrier work happens in finishTick(), after all smart movables ran.
  }

  currentTick(): number {
    return this.tick;
  }

  acceptsSmartMovableRuns(): boolean {
    return this.hasWork();
  }

  recordActorRun(car: SmartTrafficCar): void {
    this.currentActorRunOrder.push(car.id);
  }

  proposeCarStep(car: SmartTrafficCar): void {
    const lane = this.lane(car.laneId);
    const leader = this.findLeaderAheadFromGrid(car, lane) ?? this.sortedLeaderAhead(car);
    this.proposals.set(car.carId, {actor: car, ...this.nextKinematics(car, lane, leader)});
  }

  finishTick(): void {
    const dt = this.params.dtSec;
    const timeAfter = this.timeSec + dt;
    const proposals = Array.from(this.proposals.values()).sort((a, b) => a.actor.carId - b.actor.carId);
    for (const p of proposals) {
      const car = p.actor;
      if (!car.isActive()) continue;
      car.speedMps = p.speedMps;
      car.positionM = p.positionM;
      car.accelerationMps2 = p.accelerationMps2;
      car.jerkMps3 = p.jerkMps3;
      car.targetAccelerationMps2 = p.targetAccelerationMps2;
      car.leaderId = p.leaderId;
      car.leaderGapM = p.leaderGapM;
      car.startReadySinceSec = p.startReadySinceSec;
      if (p.controlFault && p.faultMode) {
        car.faultMode = p.faultMode;
        car.faultUntilSec = Math.max(car.faultUntilSec, p.faultUntilSec ?? timeAfter);
      }
      if (car.faultUntilSec <= timeAfter) {
        car.faultMode = undefined;
        car.faultUntilSec = 0;
      }
      if (car.speedMps < 0.5) {
        car.waitSec += dt;
      } else {
        car.startReadySinceSec = undefined;
      }
    }
    this.detectAccidents(timeAfter, proposals);
    for (const p of proposals) {
      if (!p.actor.isActive()) continue;
      this.handleLaneEnd(p.actor, timeAfter);
      if (p.actor.isActive()) this.recordHistory(p.actor, timeAfter);
    }
    this.spatial = this.rebuildSpatialIndex();
    for (const car of this.activeCars()) {
      this.speedSum += car.speedMps;
      this.speedSamples++;
    }
    this.maxActiveCars = Math.max(this.maxActiveCars, this.activeCars().length);
    this.totalSmartMovableRuns += this.currentActorRunOrder.length;
    this.maxSmartMovableRunsPerTick = Math.max(this.maxSmartMovableRunsPerTick, this.currentActorRunOrder.length);
    this.trace.push(this.snapshot(timeAfter));
    this.options.logger?.log({
      kind: 'smart-traffic-flow-tick',
      level: 'debug',
      tick: this.tick,
      timeSec: timeAfter,
      scheduledSmartCars: this.scheduledSmartCars,
      smartMovableRuns: this.currentActorRunOrder.length,
      activeCars: this.activeCars().length,
      entered: this.entered,
      exited: this.exited,
      crashed: this.crashed,
      dropped: this.dropped,
    });
    this.tick++;
    this.timeSec = timeAfter;
    if (this.hasWork()) this.prepareTick();
  }

  result(summary: IterativeRunSummary, validation: ValidationCheck[] = []): SmartTrafficResult {
    const travelTimes = this.completedTravelTimes.slice().sort((a, b) => a - b);
    const meanTravelTimeSec = travelTimes.length > 0 ? travelTimes.reduce((s, x) => s + x, 0) / travelTimes.length : 0;
    return {
      params: this.params,
      network: this.network,
      trace: this.trace.slice(),
      finalCars: this.snapCars(),
      entered: this.entered,
      exited: this.exited,
      crashed: this.crashed,
      dropped: this.dropped,
      meanTravelTimeSec,
      meanSpeedMps: this.speedSamples > 0 ? this.speedSum / this.speedSamples : 0,
      maxActiveCars: this.maxActiveCars,
      cellStats: {
        cellSizeM: this.gridCellSizeM(),
        laneWidthM: this.laneWidthM(),
        carWidthM: this.carWidthM(),
        activeCells: this.activeCellIds.size,
        createdCellStations: this.cellStations.size,
        accidentCellStations: Array.from(this.cellStations.values()).filter(cell => cell.accidentIds.length > 0).length,
        accidentCellHits: Array.from(this.cellStations.values()).reduce((sum, cell) => sum + cell.accidentIds.length, 0),
        maxCellOccupancy: this.maxCellOccupancy,
      },
      execution: {
        participantCount: this.carActors.length + 1,
        smartMovableCount: this.carActors.length,
        worldStationId: this.id,
        shuffledByRunner: true,
        actorShuffleSeed: this.params.actorShuffleSeed ?? this.params.seed + 1009,
        totalSmartMovableRuns: this.totalSmartMovableRuns,
        maxSmartMovableRunsPerTick: this.maxSmartMovableRunsPerTick,
      },
      runSummary: summary,
      accidents: this.accidents.slice(),
      validation,
    };
  }

  recordHistory(car: SmartTrafficCar, timeSec: number): void {
    car.history.push({
      timeSec,
      laneId: car.laneId,
      positionM: car.positionM,
      speedMps: car.speedMps,
      accelerationMps2: car.accelerationMps2,
    });
    const horizon = this.reactionTimeSec() + 2 * this.params.dtSec + 1;
    while (car.history.length > 2 && car.history[1].timeSec < timeSec - horizon) car.history.shift();
  }

  private prepareTick(): void {
    this.proposals.clear();
    this.currentActorRunOrder.length = 0;
    this.accidentsThisTick = [];
    if (this.timeSec < this.params.durationSec - 1e-9) this.spawnCars();
    this.spatial = this.rebuildSpatialIndex();
    this.scheduledSmartCars = this.activeCars().length;
  }

  private spawnCars(): void {
    if (this.scheduledTrips.length > 0) {
      this.spawnScheduledTrips();
      return;
    }
    const dt = this.params.dtSec;
    const mult = this.params.spawnRateMultiplier ?? 1;
    for (const source of this.network.sources) {
      const expected = source.ratePerMin * mult * dt / 60;
      let acc = (this.sourceAccumulators.get(source.id) ?? 0) + expected;
      const count = Math.floor(acc);
      acc -= count;
      this.sourceAccumulators.set(source.id, acc);
      for (let k = 0; k < count; k++) this.trySpawnFromSource(source);
    }
  }

  private spawnScheduledTrips(): void {
    while (this.nextScheduledTripIndex < this.scheduledTrips.length) {
      const trip = this.scheduledTrips[this.nextScheduledTripIndex];
      if (trip.departSec > this.timeSec + 1e-9) return;
      this.nextScheduledTripIndex++;
      const source = this.network.sources.find(s => s.id === trip.sourceId);
      if (!source) {
        this.dropped++;
        continue;
      }
      this.trySpawnFromSource(source, trip.destinationSinkId);
    }
  }

  private trySpawnFromSource(source: TrafficSource, destinationSinkId?: string): void {
    if (this.activeCars().length >= this.params.maxCars) {
      this.dropped++;
      return;
    }
    const actor = this.carActors.find(c => !c.isActive());
    if (!actor) {
      this.dropped++;
      return;
    }
    const sinkIds = source.destinationSinkIds ?? this.network.sinks.map(s => s.id);
    const sinkId = destinationSinkId ?? sinkIds[Math.floor(this.rng() * sinkIds.length)];
    if (!sinkIds.includes(sinkId)) {
      this.dropped++;
      return;
    }
    const route = this.routes.get(`${source.id}->${sinkId}`);
    if (!route || route.length === 0) {
      this.dropped++;
      return;
    }
    const lane = this.lane(route[0]);
    if (!this.canEnterLane(lane.id)) {
      this.dropped++;
      return;
    }
    actor.assign({
      carId: this.nextCarId++,
      laneId: lane.id,
      route,
      destinationSinkId: sinkId,
      createdAtSec: this.timeSec,
      initialSpeedMps: Math.min(2, lane.speedLimitMps),
      distancePreference: this.sampleDistancePreference(),
      startPreference: this.sampleStartPreference(),
    });
    this.entered++;
  }

  private precomputeRoutes(): void {
    for (const source of this.network.sources) {
      const sinks = source.destinationSinkIds ?? this.network.sinks.map(s => s.id);
      for (const sinkId of sinks) {
        const sink = this.network.sinks.find(s => s.id === sinkId);
        if (!sink) continue;
        const route = shortestLanePath(this.network, source.nodeId, sink.nodeId);
        if (route.length > 0) this.routes.set(`${source.id}->${sinkId}`, route);
      }
    }
  }

  private nextKinematics(car: SmartTrafficCar, lane: TrafficLane, leader: SmartTrafficCar | undefined): Omit<SmartCarProposal, 'actor'> {
    const dt = this.params.dtSec;
    const vehicleSpace = this.vehicleSpace();
    const carLength = this.carLengthM();
    const barrier = this.stopBarrierPosition(car, lane);
    const leaderPosition = leader?.positionM ?? Infinity;
    const currentLeaderGap = Number.isFinite(leaderPosition) ? leaderPosition - car.positionM - vehicleSpace : Infinity;
    const physicalLeaderGap = Number.isFinite(leaderPosition) ? leaderPosition - car.positionM - carLength : Infinity;
    const barrierGap = barrier === null ? Infinity : barrier - car.positionM - vehicleSpace;
    const useBarrier = barrierGap <= currentLeaderGap;
    const delayedLeader = leader ? this.perceivedSample(leader, this.timeSec - this.reactionTimeSec()) : null;
    const perceivedLeader = delayedLeader && delayedLeader.laneId === car.laneId ? delayedLeader : leader;
    const perceived = useBarrier
      ? {positionM: barrier ?? Infinity, speedMps: 0, id: undefined as number | undefined}
      : perceivedLeader
        ? {positionM: perceivedLeader.positionM, speedMps: perceivedLeader.speedMps, id: leader?.carId}
        : {positionM: Infinity, speedMps: lane.speedLimitMps, id: undefined as number | undefined};
    const perceivedGap = Math.max(0.05, perceived.positionM - car.positionM - vehicleSpace);
    const maxAccel = this.maxAccelMps2();
    const maxDecel = this.maxDecelMps2();
    const v = Math.max(0, car.speedMps);
    const v0 = lane.speedLimitMps;
    const distancePreference = car.distancePreference;
    const preferredVehicleSpace = this.carLengthM() + this.minGapM() * distancePreference;
    const timeHeadway = (this.timeHeadwaySec() + this.reactionTimeSec()) * distancePreference;
    const closingTerm = Math.max(0, v * (v - perceived.speedMps) / (2 * Math.sqrt(maxAccel * maxDecel)));
    const desiredGap = preferredVehicleSpace + v * timeHeadway + closingTerm;
    const freeRoad = 1 - Math.pow(Math.min(v / Math.max(1e-9, v0), 2), 2);
    const interaction = Number.isFinite(perceived.positionM) ? Math.pow(desiredGap / perceivedGap, 2) : 0;
    let targetAccelerationMps2 = clamp(maxAccel * (freeRoad - interaction), -maxDecel, maxAccel);
    let startReadySinceSec = car.startReadySinceSec;
    if (v < 0.5 && targetAccelerationMps2 > 0) {
      const hasStartupClearance = this.hasStartupClearance(car, perceived, physicalLeaderGap);
      if (hasStartupClearance) {
        startReadySinceSec = startReadySinceSec ?? this.timeSec;
      } else {
        startReadySinceSec = undefined;
      }
      const readyForSec = startReadySinceSec === undefined ? 0 : Math.max(0, this.timeSec - startReadySinceSec);
      if (!hasStartupClearance || readyForSec < this.startDelaySec(car.startPreference)) {
        targetAccelerationMps2 = Math.min(0, targetAccelerationMps2);
      }
    } else if (v >= 0.5 || targetAccelerationMps2 <= 0) {
      startReadySinceSec = undefined;
    }
    const maxJerkStep = this.maxJerkMps3() * dt;
    const closingSpeedMps = Number.isFinite(perceived.positionM) ? Math.max(0, v - perceived.speedMps) : 0;
    const timeToContactSec = closingSpeedMps > 1e-9 && Number.isFinite(physicalLeaderGap)
      ? physicalLeaderGap / closingSpeedMps
      : Infinity;
    const speedRisk = clamp((v / Math.max(1e-9, v0) - 1) / 0.35, 0, 1);
    const closeRisk = Number.isFinite(physicalLeaderGap)
      ? clamp((desiredGap / Math.max(0.05, physicalLeaderGap) - 1) / 4, 0, 1)
      : 0;
    const ttcRisk = Number.isFinite(timeToContactSec)
      ? clamp((3.0 - timeToContactSec) / 3.0, 0, 1)
      : 0;
    const brakingRisk = clamp((-targetAccelerationMps2 / Math.max(1e-9, maxDecel) - 0.35) / 0.65, 0, 1);
    const accelRisk = clamp((targetAccelerationMps2 / Math.max(1e-9, maxAccel) - 0.7) / 0.3, 0, 1);
    const riskScore = clamp(0.45 * ttcRisk + 0.25 * closeRisk + 0.2 * brakingRisk + 0.15 * speedRisk + 0.1 * accelRisk, 0, 1);
    const hazardPerSec = this.accidentRiskScale() * riskScore;
    const activeFault = car.faultUntilSec > this.timeSec;
    const startsFault = !activeFault && leader !== undefined && riskScore > 0 && this.rng() < 1 - Math.exp(-hazardPerSec * dt);
    const controlFault = activeFault || startsFault;
    const faultMode: SmartTrafficFaultMode | undefined = activeFault
      ? car.faultMode
      : startsFault
        ? this.faultModeForRisk(speedRisk, brakingRisk, accelRisk, ttcRisk)
        : undefined;
    const faultUntilSec = startsFault ? this.timeSec + this.accidentFaultDurationSec() : car.faultUntilSec;
    if (controlFault) {
      if (faultMode === 'brake-too-slow') {
        targetAccelerationMps2 = Math.max(targetAccelerationMps2, -maxDecel * 0.12);
      } else {
        targetAccelerationMps2 = maxAccel + this.accidentAccelBoostMps2() * Math.max(0.25, riskScore);
      }
    }
    let accelerationMps2 = clamp(
      car.accelerationMps2 + clamp(targetAccelerationMps2 - car.accelerationMps2, -maxJerkStep, maxJerkStep),
      -maxDecel,
      maxAccel,
    );
    if (controlFault && faultMode === 'brake-too-slow') {
      accelerationMps2 = Math.max(accelerationMps2, -maxDecel * 0.12);
    } else if (controlFault) {
      accelerationMps2 = Math.max(accelerationMps2, maxAccel + this.accidentAccelBoostMps2() * Math.max(0.25, riskScore));
    }
    let speedMps = controlFault
      ? clamp(v + accelerationMps2 * dt, 0, v0 * (faultMode === 'speeding' ? 1.6 : 1.3))
      : clamp(v + accelerationMps2 * dt, 0, v0);
    let positionM = car.positionM + v * dt + 0.5 * accelerationMps2 * dt * dt;
    const hardLimit = this.hardPositionLimit(car, lane, leader, barrier);
    if (!controlFault && positionM > hardLimit) {
      positionM = Math.max(car.positionM, hardLimit);
      speedMps = Math.max(0, Math.min(speedMps, (positionM - car.positionM) / dt));
    }
    accelerationMps2 = clamp((speedMps - v) / dt, -maxDecel, maxAccel);
    if (controlFault) accelerationMps2 = (speedMps - v) / dt;
    const jerkMps3 = controlFault
      ? (accelerationMps2 - car.accelerationMps2) / dt
      : clamp((accelerationMps2 - car.accelerationMps2) / dt, -this.maxJerkMps3(), this.maxJerkMps3());
    return {
      speedMps,
      positionM,
      accelerationMps2,
      jerkMps3,
      targetAccelerationMps2,
      leaderId: perceived.id,
      leaderGapM: Number.isFinite(perceived.positionM) ? Math.max(0, perceivedGap) : undefined,
      controlFault,
      faultMode,
      faultUntilSec: faultUntilSec > this.timeSec ? faultUntilSec : undefined,
      startReadySinceSec,
      riskScore,
      hazardPerSec,
    };
  }

  private detectAccidents(timeSec: number, proposals: SmartCarProposal[]): void {
    const proposedByCar = new Map<number, SmartCarProposal>();
    for (const p of proposals) proposedByCar.set(p.actor.carId, p);
    const crashedIds = new Set<number>();
    for (const lane of this.network.lanes) {
      const cars = this.activeCars()
        .filter(c => c.laneId === lane.id && !crashedIds.has(c.carId))
        .sort((a, b) => a.positionM - b.positionM);
      for (let i = 1; i < cars.length; i++) {
        const car = cars[i - 1];
        const leader = cars[i];
        if (!car.isActive() || !leader.isActive() || crashedIds.has(car.carId)) continue;
        const p = proposedByCar.get(car.carId);
        const contactPosition = leader.positionM - this.carLengthM();
        if (car.positionM < contactPosition) continue;
        const accident: SmartTrafficAccident = {
          tick: this.tick,
          timeSec,
          laneId: car.laneId,
          positionM: Math.max(0, contactPosition),
          cellId: this.accidentCellId(car.laneId, Math.max(0, contactPosition)),
          carId: car.carId,
          actorId: car.id,
          otherCarId: leader.carId,
          otherActorId: leader.id,
          speedMps: p?.speedMps ?? car.speedMps,
          faultMode: p?.faultMode ?? 'brake-too-slow',
          riskScore: p?.riskScore ?? 0,
          hazardPerSec: p?.hazardPerSec ?? 0,
          reason: 'body-contact-rear-end',
        };
        this.accidents.push(accident);
        this.accidentsThisTick.push(accident);
        this.crashed++;
        crashedIds.add(car.carId);
        car.recordAccident(accident);
        leader.recordAccident(accident);
        this.ensureCellStation(accident.cellId).recordAccident(accident);
        car.retire();
        this.options.logger?.log({kind: 'smart-traffic-accident', level: 'warn', ...accident});
      }
    }
  }

  private stopBarrierPosition(car: SmartTrafficCar, lane: TrafficLane): number | null {
    const nextLaneId = car.route[car.routeIndex + 1];
    if (nextLaneId === undefined) return null;
    if (this.signalAllows(lane.id) && this.canEnterLane(nextLaneId, car.carId)) return null;
    return lane.lengthM;
  }

  private hardPositionLimit(car: SmartTrafficCar, lane: TrafficLane, leader: SmartTrafficCar | undefined, barrier: number | null): number {
    let limit = Infinity;
    if (leader) limit = Math.min(limit, leader.positionM - this.vehicleSpace());
    if (barrier !== null) limit = Math.min(limit, barrier - this.vehicleSpace());
    if (!Number.isFinite(limit)) return lane.lengthM + Math.max(0, car.speedMps * this.params.dtSec);
    return Math.max(car.positionM, limit);
  }

  private handleLaneEnd(car: SmartTrafficCar, timeSec: number): void {
    let lane = this.lane(car.laneId);
    if (car.positionM < lane.lengthM - 1e-9) return;
    let overshoot = car.positionM - lane.lengthM;
    while (car.positionM >= lane.lengthM - 1e-9) {
      const nextLaneId = car.route[car.routeIndex + 1];
      if (nextLaneId === undefined) {
        this.exited++;
        this.completedTravelTimes.push(timeSec - car.createdAtSec);
        car.retire();
        return;
      }
      if (!this.signalAllows(lane.id) || !this.canEnterLane(nextLaneId, car.carId)) {
        car.positionM = this.blockedStopPosition(car);
        car.speedMps = 0;
        return;
      }
      car.routeIndex++;
      car.laneId = nextLaneId;
      lane = this.lane(car.laneId);
      car.positionM = Math.min(Math.max(0, overshoot), Math.max(0, lane.lengthM - this.vehicleSpace()));
      car.speedMps = Math.min(car.speedMps, lane.speedLimitMps);
      overshoot = Math.max(0, car.positionM - lane.lengthM);
      if (overshoot <= 1e-9) return;
    }
  }

  private blockedStopPosition(car: SmartTrafficCar): number {
    const lane = this.lane(car.laneId);
    const vehicleSpace = this.vehicleSpace();
    let safe = Math.max(0, lane.lengthM - vehicleSpace);
    const others = this.activeCars()
      .filter(c => c.carId !== car.carId && c.laneId === car.laneId)
      .sort((a, b) => b.positionM - a.positionM);
    for (const other of others) {
      if (other.positionM <= safe + vehicleSpace) safe = Math.min(safe, other.positionM - vehicleSpace);
    }
    return Math.max(0, safe);
  }

  private rebuildSpatialIndex(): SmartTrafficSpatialIndex {
    for (const cell of this.cellStations.values()) cell.clearOccupancy();
    this.activeCellIds.clear();
    const byCell = new Map<string, Set<number>>();
    for (const car of this.activeCars()) {
      const cellIds = this.occupiedCellIds(car);
      car.gridCellIds = cellIds;
      car.gridCellCount = cellIds.length;
      for (const cellId of cellIds) {
        const station = this.ensureCellStation(cellId);
        station.occupy(car.carId);
        this.maxCellOccupancy = Math.max(this.maxCellOccupancy, station.carIds.size);
        this.activeCellIds.add(cellId);
        const set = byCell.get(cellId) ?? new Set<number>();
        set.add(car.carId);
        byCell.set(cellId, set);
      }
    }
    return {byCell, activeCellCount: this.activeCellIds.size};
  }

  private findLeaderAheadFromGrid(car: SmartTrafficCar, lane: TrafficLane): SmartTrafficCar | undefined {
    const cellSize = this.gridCellSizeM();
    const lookAhead = Math.min(lane.lengthM - car.positionM, this.params.gridLookAheadM ?? Math.max(60, car.speedMps * (this.reactionTimeSec() + 4) + 3 * this.vehicleSpace()));
    const first = Math.max(0, Math.floor(car.positionM / cellSize));
    const last = Math.max(first, Math.floor((car.positionM + lookAhead) / cellSize));
    const lateral = this.occupiedLateralCellRange();
    let best: SmartTrafficCar | undefined;
    for (let x = first; x <= last; x++) {
      for (let y = lateral.start; y <= lateral.end; y++) {
        const ids = this.spatial.byCell.get(this.cellId(lane.id, x, y));
        if (!ids) continue;
        for (const id of ids) {
          if (id === car.carId) continue;
          const other = this.carById(id);
          if (!other || other.laneId !== car.laneId || other.positionM <= car.positionM) continue;
          if (!best || other.positionM < best.positionM) best = other;
        }
      }
      if (best && best.positionM <= (x + 1) * cellSize) break;
    }
    return best;
  }

  private sortedLeaderAhead(car: SmartTrafficCar): SmartTrafficCar | undefined {
    let best: SmartTrafficCar | undefined;
    for (const other of this.activeCars()) {
      if (other.carId === car.carId || other.laneId !== car.laneId || other.positionM <= car.positionM) continue;
      if (!best || other.positionM < best.positionM) best = other;
    }
    return best;
  }

  private snapshot(timeSec: number): SmartTrafficTraceRow {
    const laneOccupancy: Record<string, number> = {};
    for (const lane of this.network.lanes) laneOccupancy[lane.id] = 0;
    for (const car of this.activeCars()) laneOccupancy[car.laneId] = (laneOccupancy[car.laneId] ?? 0) + 1;
    const signalPhases: Record<string, string> = {};
    for (const signal of this.network.signals ?? []) signalPhases[signal.nodeId] = currentSignalPhase(signal, timeSec).name;
    const cars = this.snapCars();
    const meanSpeedMps = cars.length > 0 ? cars.reduce((s, c) => s + c.speedMps, 0) / cars.length : 0;
    const meanTravelTimeSec = this.completedTravelTimes.length > 0
      ? this.completedTravelTimes.reduce((s, x) => s + x, 0) / this.completedTravelTimes.length
      : 0;
    return {
      tick: this.tick,
      timeSec,
      activeCars: cars.length,
      scheduledSmartCars: this.scheduledSmartCars,
      smartMovableRuns: this.currentActorRunOrder.length,
      entered: this.entered,
      exited: this.exited,
      crashed: this.crashed,
      meanSpeedMps,
      meanTravelTimeSec,
      queueLength: cars.filter(c => c.speedMps < 0.5).length,
      laneOccupancy,
      activeGridCells: this.activeCellIds.size,
      signalPhases,
      actorRunOrder: this.currentActorRunOrder.slice(0, 24),
      accidents: this.accidentsThisTick.slice(),
      cars,
    };
  }

  private snapCars(): SmartTrafficCarSnapshot[] {
    return this.activeCars().sort((a, b) => a.carId - b.carId).map(c => c.snapshot());
  }

  private activeCars(): SmartTrafficCar[] {
    return this.carActors.filter(c => c.isActive());
  }

  private carById(carId: number): SmartTrafficCar | undefined {
    return this.carActors.find(c => c.isActive() && c.carId === carId);
  }

  private minimumHeadway(): number {
    let min = Infinity;
    for (const lane of this.network.lanes) {
      const cars = this.activeCars().filter(c => c.laneId === lane.id).sort((a, b) => a.positionM - b.positionM);
      for (let i = 1; i < cars.length; i++) min = Math.min(min, cars[i].positionM - cars[i - 1].positionM - this.vehicleSpace());
    }
    return Number.isFinite(min) ? min : 0;
  }

  private minimumBodyGap(): number {
    let min = Infinity;
    for (const lane of this.network.lanes) {
      const cars = this.activeCars().filter(c => c.laneId === lane.id).sort((a, b) => a.positionM - b.positionM);
      for (let i = 1; i < cars.length; i++) min = Math.min(min, cars[i].positionM - cars[i - 1].positionM - this.carLengthM());
    }
    return Number.isFinite(min) ? min : 0;
  }

  private signalAllows(incomingLaneId: string): boolean {
    const lane = this.lane(incomingLaneId);
    const node = this.nodes.get(lane.to);
    if (!node || node.kind !== 'intersection') return true;
    const signal = this.signalByNode.get(node.id);
    if (!signal) return true;
    return currentSignalPhase(signal, this.timeSec).greenLanes.includes(incomingLaneId);
  }

  private canEnterLane(laneId: string, ignoreCarId?: number): boolean {
    const lane = this.lane(laneId);
    const cars = this.activeCars().filter(c => c.laneId === laneId && c.carId !== ignoreCarId);
    const cap = lane.capacity ?? defaultLaneCapacity(lane, this.vehicleSpace());
    if (cars.length >= cap) return false;
    return cars.every(c => c.positionM >= this.vehicleSpace());
  }

  private lane(id: string): TrafficLane {
    const lane = this.lanes.get(id);
    if (!lane) throw new Error(`smart-traffic-flow: unknown lane "${id}"`);
    return lane;
  }

  private perceivedSample(car: SmartTrafficCar, targetTimeSec: number): SmartTrafficKinematicSample {
    for (let i = car.history.length - 1; i >= 0; i--) {
      if (car.history[i].timeSec <= targetTimeSec + 1e-12) return car.history[i];
    }
    return car.history[0] ?? {
      timeSec: targetTimeSec,
      laneId: car.laneId,
      positionM: car.positionM,
      speedMps: car.speedMps,
      accelerationMps2: car.accelerationMps2,
    };
  }

  private occupiedLateralCellRange(): {start: number; end: number} {
    const cellSize = this.gridCellSizeM();
    const laneWidth = this.laneWidthM();
    const carWidth = Math.min(this.carWidthM(), laneWidth);
    const left = Math.max(0, (laneWidth - carWidth) / 2);
    const right = Math.min(laneWidth, left + carWidth);
    return {
      start: Math.max(0, Math.floor(left / cellSize)),
      end: Math.max(0, Math.floor(Math.max(left, right - 1e-9) / cellSize)),
    };
  }

  private occupiedCellIds(car: SmartTrafficCar): string[] {
    const cellSize = this.gridCellSizeM();
    const rear = Math.max(0, car.positionM - this.carLengthM());
    const front = Math.max(rear, car.positionM);
    const x0 = Math.max(0, Math.floor(rear / cellSize));
    const x1 = Math.max(x0, Math.floor(front / cellSize));
    const y = this.occupiedLateralCellRange();
    const ids: string[] = [];
    for (let x = x0; x <= x1; x++) {
      for (let lat = y.start; lat <= y.end; lat++) ids.push(this.cellId(car.laneId, x, lat));
    }
    return ids;
  }

  private cellId(laneId: string, longitudinalIndex: number, lateralIndex: number): string {
    return `${laneId}#${longitudinalIndex}:${lateralIndex}`;
  }

  private accidentCellId(laneId: string, positionM: number): string {
    const x = Math.max(0, Math.floor(positionM / this.gridCellSizeM()));
    const y = this.occupiedLateralCellRange();
    return this.cellId(laneId, x, y.start);
  }

  private ensureCellStation(cellId: string): SmartTrafficCellStation {
    const existing = this.cellStations.get(cellId);
    if (existing) return existing;
    const parsed = this.parseCellId(cellId);
    const cellSize = this.gridCellSizeM();
    const station = new SmartTrafficCellStation({
      laneId: parsed.laneId,
      longitudinalIndex: parsed.longitudinalIndex,
      lateralIndex: parsed.lateralIndex,
      x0M: parsed.longitudinalIndex * cellSize,
      x1M: (parsed.longitudinalIndex + 1) * cellSize,
      y0M: parsed.lateralIndex * cellSize,
      y1M: (parsed.lateralIndex + 1) * cellSize,
    });
    this.cellStations.set(cellId, station);
    return station;
  }

  private parseCellId(cellId: string): {laneId: string; longitudinalIndex: number; lateralIndex: number} {
    const sep = cellId.lastIndexOf('#');
    const laneId = cellId.slice(0, sep);
    const [x, y] = cellId.slice(sep + 1).split(':').map(Number);
    return {laneId, longitudinalIndex: x, lateralIndex: y};
  }

  private vehicleSpace(): number { return (this.params.carLengthM ?? 4.8) + (this.params.minGapM ?? 2.5); }
  private carLengthM(): number { return this.params.carLengthM ?? 4.8; }
  private carWidthM(): number { return this.params.carWidthM ?? 1.8; }
  private laneWidthM(): number { return this.params.laneWidthM ?? 3.7; }
  private gridCellSizeM(): number { return this.params.gridCellSizeM ?? 0.3048; }
  private maxAccelMps2(): number { return this.params.maxAccelMps2 ?? 2.2; }
  private maxDecelMps2(): number { return this.params.maxDecelMps2 ?? 4.0; }
  private maxJerkMps3(): number { return this.params.maxJerkMps3 ?? 6.0; }
  private reactionTimeSec(): number { return this.params.reactionTimeSec ?? 0.8; }
  private timeHeadwaySec(): number { return this.params.timeHeadwaySec ?? 1.1; }
  private minGapM(): number { return this.params.minGapM ?? 2.5; }
  private accidentRiskScale(): number { return this.params.accidentRiskScale ?? this.params.accidentProbability ?? 0; }
  private accidentAccelBoostMps2(): number { return this.params.accidentAccelBoostMps2 ?? 10; }
  private accidentFaultDurationSec(): number { return this.params.accidentFaultDurationSec ?? 1.0; }
  private distancePreferenceSpread(): number { return this.params.distancePreferenceSpread ?? 0; }
  private startPreferenceSpread(): number { return this.params.startPreferenceSpread ?? 0; }

  private faultModeForRisk(speedRisk: number, brakingRisk: number, accelRisk: number, ttcRisk: number): SmartTrafficFaultMode {
    if (speedRisk >= brakingRisk && speedRisk >= ttcRisk) return 'speeding';
    return brakingRisk >= accelRisk ? 'brake-too-slow' : 'accelerate-too-fast';
  }

  private sampleDistancePreference(): number {
    return this.sampleDriverTrait(this.distancePreferenceSpread());
  }

  private sampleStartPreference(): number {
    return this.sampleDriverTrait(this.startPreferenceSpread());
  }

  private sampleDriverTrait(spread: number): number {
    if (spread <= 0) return 1;
    const k = sampleFromPMF(DRIVER_TRAIT_PMF, this.rng);
    const z = DRIVER_TRAIT_STD > 0 ? (k - DRIVER_TRAIT_MEAN) / DRIVER_TRAIT_STD : 0;
    return clamp(1 + (spread / Math.sqrt(3)) * z, 0.35, 2.25);
  }

  private hasStartupClearance(
    car: SmartTrafficCar,
    perceived: {positionM: number; speedMps: number; id?: number},
    physicalLeaderGap: number,
  ): boolean {
    if (!Number.isFinite(perceived.positionM) || perceived.id === undefined) return true;
    const leaderIsMovingAway = perceived.speedMps > 0.75;
    const requiredGapM = this.minGapM() * (0.45 + car.startPreference);
    return physicalLeaderGap >= requiredGapM || (leaderIsMovingAway && physicalLeaderGap >= this.minGapM() * 0.45);
  }

  private startDelaySec(startPreference: number): number {
    if (this.startPreferenceSpread() <= 0) return 0;
    return Math.max(0, (startPreference - 0.55) * this.reactionTimeSec() * 0.55);
  }
}

export function runSmartTrafficFlow(params: SmartTrafficParams, logger?: OptimizationLogger): SmartTrafficResult {
  const world = new SmartTrafficWorldStation(params, {logger});
  world.assertPreconditions();
  world.beginRun();
  const summary = runIterativeDES([world, ...world.carActors], {
    shuffle: true,
    rng: mulberry32(params.actorShuffleSeed ?? params.seed + 1009),
    maxTicks: Math.ceil(params.durationSec / params.dtSec) + 1,
    onTick: () => world.finishTick(),
  });
  return world.result(summary, summary.validation ?? []);
}

function validateSmartTrafficNetwork(network: TrafficNetwork): void {
  Preconditions.nonEmpty('SmartTrafficWorldStation', 'network.nodes', network.nodes);
  Preconditions.nonEmpty('SmartTrafficWorldStation', 'network.lanes', network.lanes);
  Preconditions.nonEmpty('SmartTrafficWorldStation', 'network.sources', network.sources);
  Preconditions.nonEmpty('SmartTrafficWorldStation', 'network.sinks', network.sinks);
  const nodeIds = new Set<string>();
  const nodeById = new Map<string, TrafficNode>();
  for (const node of network.nodes) {
    Preconditions.check('SmartTrafficWorldStation', `node.${node.id}`, 'have a non-empty id', typeof node.id === 'string' && node.id.length > 0, node.id);
    Preconditions.check('SmartTrafficWorldStation', `node.${node.id}.kind`, 'be source, intersection, or sink', ['source', 'intersection', 'sink'].includes(node.kind), node.kind);
    Preconditions.check('SmartTrafficWorldStation', `node.${node.id}`, 'be unique', !nodeIds.has(node.id), node.id);
    Preconditions.finite('SmartTrafficWorldStation', `node.${node.id}.x`, node.x);
    Preconditions.finite('SmartTrafficWorldStation', `node.${node.id}.y`, node.y);
    nodeIds.add(node.id);
    nodeById.set(node.id, node);
  }
  const laneIds = new Set<string>();
  for (const lane of network.lanes) {
    Preconditions.check('SmartTrafficWorldStation', `lane.${lane.id}`, 'have a non-empty id', typeof lane.id === 'string' && lane.id.length > 0, lane.id);
    Preconditions.check('SmartTrafficWorldStation', `lane.${lane.id}`, 'be unique', !laneIds.has(lane.id), lane.id);
    laneIds.add(lane.id);
    Preconditions.check('SmartTrafficWorldStation', `lane.${lane.id}.from`, 'reference a node', nodeIds.has(lane.from), lane.from);
    Preconditions.check('SmartTrafficWorldStation', `lane.${lane.id}.to`, 'reference a node', nodeIds.has(lane.to), lane.to);
    Preconditions.positive('SmartTrafficWorldStation', `lane.${lane.id}.lengthM`, lane.lengthM);
    Preconditions.positive('SmartTrafficWorldStation', `lane.${lane.id}.speedLimitMps`, lane.speedLimitMps);
    if (lane.capacity !== undefined) Preconditions.integerInRange('SmartTrafficWorldStation', `lane.${lane.id}.capacity`, lane.capacity, 1, 299);
  }
  for (const signal of network.signals ?? []) {
    Preconditions.check('SmartTrafficWorldStation', `signal.${signal.nodeId}`, 'reference a node', nodeIds.has(signal.nodeId), signal.nodeId);
    for (const phase of signal.phases) {
      Preconditions.positive('SmartTrafficWorldStation', `signal.${signal.nodeId}.${phase.name}.durationSec`, phase.durationSec);
      for (const laneId of phase.greenLanes) {
        Preconditions.check('SmartTrafficWorldStation', `signal.${signal.nodeId}.${phase.name}.greenLanes`, 'reference a lane', laneIds.has(laneId), laneId);
      }
    }
  }
  const sinkIds = new Set<string>();
  for (const sink of network.sinks) {
    Preconditions.check('SmartTrafficWorldStation', `sink.${sink.id}`, 'have a non-empty id', typeof sink.id === 'string' && sink.id.length > 0, sink.id);
    Preconditions.check('SmartTrafficWorldStation', `sink.${sink.id}`, 'be unique', !sinkIds.has(sink.id), sink.id);
    Preconditions.check('SmartTrafficWorldStation', `sink.${sink.id}.nodeId`, 'reference a node', nodeIds.has(sink.nodeId), sink.nodeId);
    Preconditions.check('SmartTrafficWorldStation', `sink.${sink.id}.nodeId`, 'reference a sink node', nodeById.get(sink.nodeId)?.kind === 'sink', sink.nodeId);
    sinkIds.add(sink.id);
  }
  const sourceIds = new Set<string>();
  for (const source of network.sources) {
    Preconditions.check('SmartTrafficWorldStation', `source.${source.id}`, 'have a non-empty id', typeof source.id === 'string' && source.id.length > 0, source.id);
    Preconditions.check('SmartTrafficWorldStation', `source.${source.id}`, 'be unique', !sourceIds.has(source.id), source.id);
    Preconditions.check('SmartTrafficWorldStation', `source.${source.id}.nodeId`, 'reference a node', nodeIds.has(source.nodeId), source.nodeId);
    Preconditions.check('SmartTrafficWorldStation', `source.${source.id}.nodeId`, 'reference a source node', nodeById.get(source.nodeId)?.kind === 'source', source.nodeId);
    Preconditions.nonNegative('SmartTrafficWorldStation', `source.${source.id}.ratePerMin`, source.ratePerMin);
    const destinationSinkIds = source.destinationSinkIds ?? Array.from(sinkIds);
    Preconditions.nonEmpty('SmartTrafficWorldStation', `source.${source.id}.destinationSinkIds`, destinationSinkIds);
    sourceIds.add(source.id);
    for (const sinkId of destinationSinkIds) {
      Preconditions.check('SmartTrafficWorldStation', `source.${source.id}.destinationSinkIds`, 'reference a sink id', sinkIds.has(sinkId), sinkId);
      const sink = network.sinks.find(s => s.id === sinkId);
      if (sink) {
        Preconditions.check(
          'SmartTrafficWorldStation',
          `route ${source.id}->${sinkId}`,
          'have at least one directed lane path',
          shortestLanePath(network, source.nodeId, sink.nodeId).length > 0,
          {source: source.nodeId, sink: sink.nodeId},
        );
      }
    }
  }
}

function validateSmartTrafficScheduledTrips(network: TrafficNetwork, trips: TrafficScheduledTrip[], durationSec: number): void {
  const sourceById = new Map(network.sources.map(s => [s.id, s]));
  const sinkById = new Map(network.sinks.map(s => [s.id, s]));
  for (const trip of trips) {
    Preconditions.nonNegative('SmartTrafficWorldStation', 'scheduledTrips.departSec', trip.departSec);
    Preconditions.check('SmartTrafficWorldStation', 'scheduledTrips.departSec', 'be within durationSec', trip.departSec <= durationSec + 1e-9, trip.departSec);
    const source = sourceById.get(trip.sourceId);
    const sink = sinkById.get(trip.destinationSinkId);
    Preconditions.check('SmartTrafficWorldStation', `scheduledTrips.${trip.sourceId}`, 'reference a source id', !!source, trip.sourceId);
    Preconditions.check('SmartTrafficWorldStation', `scheduledTrips.${trip.destinationSinkId}`, 'reference a sink id', !!sink, trip.destinationSinkId);
    if (!source || !sink) continue;
    const allowedSinkIds = source.destinationSinkIds ?? network.sinks.map(s => s.id);
    Preconditions.check('SmartTrafficWorldStation', `scheduledTrips.${trip.sourceId}->${trip.destinationSinkId}`, 'use a sink allowed by the source', allowedSinkIds.includes(trip.destinationSinkId), allowedSinkIds);
    Preconditions.check(
      'SmartTrafficWorldStation',
      `scheduledTrips.${trip.sourceId}->${trip.destinationSinkId}`,
      'have at least one directed lane path',
      shortestLanePath(network, source.nodeId, sink.nodeId).length > 0,
      {source: source.nodeId, sink: sink.nodeId},
    );
  }
}

function defaultLaneCapacity(lane: TrafficLane, vehicleSpace: number): number {
  return Math.max(1, Math.floor(lane.lengthM / vehicleSpace));
}

function currentSignalPhase(signal: TrafficSignal, timeSec: number): TrafficSignalPhase {
  const cycle = signal.phases.reduce((s, p) => s + p.durationSec, 0);
  let t = ((timeSec + (signal.offsetSec ?? 0)) % cycle + cycle) % cycle;
  for (const phase of signal.phases) {
    if (t < phase.durationSec) return phase;
    t -= phase.durationSec;
  }
  return signal.phases[signal.phases.length - 1];
}

function shortestLanePath(network: TrafficNetwork, sourceNodeId: string, sinkNodeId: string): string[] {
  const dist = new Map<string, number>();
  const prevLane = new Map<string, string>();
  const prevNode = new Map<string, string>();
  const nodes = network.nodes.map(n => n.id);
  for (const n of nodes) dist.set(n, Infinity);
  dist.set(sourceNodeId, 0);
  const pending = new Set(nodes);
  while (pending.size > 0) {
    let u = '';
    let best = Infinity;
    for (const n of pending) {
      const d = dist.get(n) ?? Infinity;
      if (d < best) { best = d; u = n; }
    }
    if (!u || !Number.isFinite(best)) break;
    pending.delete(u);
    if (u === sinkNodeId) break;
    for (const lane of network.lanes.filter(l => l.from === u)) {
      const alt = best + lane.lengthM;
      if (alt < (dist.get(lane.to) ?? Infinity)) {
        dist.set(lane.to, alt);
        prevLane.set(lane.to, lane.id);
        prevNode.set(lane.to, u);
      }
    }
  }
  if (!Number.isFinite(dist.get(sinkNodeId) ?? Infinity)) return [];
  const route: string[] = [];
  let cur = sinkNodeId;
  while (cur !== sourceNodeId) {
    const lane = prevLane.get(cur);
    const prev = prevNode.get(cur);
    if (!lane || !prev) return [];
    route.push(lane);
    cur = prev;
  }
  route.reverse();
  return route;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
