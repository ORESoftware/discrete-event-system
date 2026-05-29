'use strict';

// =============================================================================
// Network-flow and traffic-flow DES models.
//
// Max-flow is modeled as one augmenting path per DES tick. Traffic flow is a
// continuous-time fixed-step simulation where the stationary grid owns lanes,
// intersections, sources, and sinks; cars are moving tokens whose feasible
// motion is constrained by headway, downstream capacity, and signal phases.
// =============================================================================

import {
  DESStation,
  Token,
  intrinsicCheck,
  runIterativeDES,
  ValidationCheck,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

export interface OptimizationLogger {
  log(event: {kind: string; level?: 'trace' | 'debug' | 'info' | 'warn' | 'error'; [key: string]: unknown}): void;
}

// -----------------------------------------------------------------------------
// Max-flow optimization by augmenting-path DES ticks.
// -----------------------------------------------------------------------------

export interface FlowEdge {
  from: number;
  to: number;
  capacity: number;
  name?: string;
}

export interface MaxFlowParams {
  numNodes: number;
  source: number;
  sink: number;
  edges: FlowEdge[];
  maxAugmentations?: number;
  nodeCoordinates?: Array<[number, number]>;
  nodeNames?: string[];
}

export interface FlowEdgeResult extends FlowEdge {
  flow: number;
  residual: number;
}

export interface MaxFlowTraceRow {
  iter: number;
  pathNodes: number[];
  pathEdges: number[];
  bottleneck: number;
  value: number;
}

export interface MaxFlowMinCut {
  sourceSide: number[];
  sinkSide: number[];
  cutEdges: number[];
  capacity: number;
}

export interface MaxFlowResult {
  params: MaxFlowParams;
  maxFlow: number;
  edgeFlows: FlowEdgeResult[];
  minCut: MaxFlowMinCut;
  trace: MaxFlowTraceRow[];
  validation: ValidationCheck[];
}

interface ResidualStep {
  edge: number;
  dir: 1 | -1;
}

export class AugmentingPathToken implements Token {
  constructor(readonly row: MaxFlowTraceRow) {}
}

export class MaxFlowOptimizationStation extends DESStation {
  private readonly flow: number[];
  readonly trace: MaxFlowTraceRow[] = [];
  private done = false;
  private value = 0;

  constructor(readonly params: MaxFlowParams, private readonly logger?: OptimizationLogger) {
    super('max-flow');
    this.flow = new Array(params.edges.length).fill(0);
    this.addValidator(intrinsicCheck<MaxFlowOptimizationStation>({
      name: 'max-flow-capacity-feasible',
      predicate: s => s.edgeFlows().every(e => e.flow >= -1e-8 && e.flow <= e.capacity + 1e-8),
      expected: '0 <= flow <= capacity on every edge',
      group: 'max-flow',
    }));
    this.addValidator(intrinsicCheck<MaxFlowOptimizationStation>({
      name: 'max-flow-conservation',
      predicate: s => s.flowConservationOk(),
      expected: 'inflow equals outflow at every transshipment node',
      group: 'max-flow',
    }));
    this.addValidator(intrinsicCheck<MaxFlowOptimizationStation>({
      name: 'max-flow-min-cut-tight',
      predicate: s => Math.abs(s.currentValue() - s.minCut().capacity) <= 1e-7,
      observedFn: s => `flow=${s.currentValue().toFixed(6)} cut=${s.minCut().capacity.toFixed(6)}`,
      expected: 'max-flow value equals residual min-cut capacity',
      group: 'max-flow',
    }));
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.integerInRange('MaxFlowOptimizationStation', 'numNodes', p.numNodes, 2, 10000);
    Preconditions.integerInRange('MaxFlowOptimizationStation', 'source', p.source, 0, p.numNodes - 1);
    Preconditions.integerInRange('MaxFlowOptimizationStation', 'sink', p.sink, 0, p.numNodes - 1);
    Preconditions.check('MaxFlowOptimizationStation', 'sink', 'differ from source', p.sink !== p.source, p.sink);
    Preconditions.nonEmpty('MaxFlowOptimizationStation', 'edges', p.edges);
    for (let i = 0; i < p.edges.length; i++) {
      const e = p.edges[i];
      Preconditions.integerInRange('MaxFlowOptimizationStation', `edges[${i}].from`, e.from, 0, p.numNodes - 1);
      Preconditions.integerInRange('MaxFlowOptimizationStation', `edges[${i}].to`, e.to, 0, p.numNodes - 1);
      Preconditions.check('MaxFlowOptimizationStation', `edges[${i}]`, 'not be a self-loop', e.from !== e.to, e);
      Preconditions.nonNegative('MaxFlowOptimizationStation', `edges[${i}].capacity`, e.capacity);
    }
    if (p.maxAugmentations !== undefined) {
      Preconditions.integerInRange('MaxFlowOptimizationStation', 'maxAugmentations', p.maxAugmentations, 1, Number.MAX_SAFE_INTEGER);
    }
    if (p.nodeCoordinates !== undefined) Preconditions.lengthEq('MaxFlowOptimizationStation', 'nodeCoordinates', p.nodeCoordinates, p.numNodes);
    if (p.nodeNames !== undefined) Preconditions.lengthEq('MaxFlowOptimizationStation', 'nodeNames', p.nodeNames, p.numNodes);
  }

  override hasWork(): boolean {
    return !this.done;
  }

  runTimeStep(): void {
    if (this.done) return;
    if (this.trace.length >= (this.params.maxAugmentations ?? Number.MAX_SAFE_INTEGER)) {
      this.done = true;
      return;
    }
    const path = this.findAugmentingPath();
    if (!path) {
      this.done = true;
      return;
    }
    let bottleneck = Infinity;
    for (const step of path.steps) bottleneck = Math.min(bottleneck, this.residualCapacity(step));
    for (const step of path.steps) this.flow[step.edge] += step.dir * bottleneck;
    this.value += bottleneck;
    const row: MaxFlowTraceRow = {
      iter: this.trace.length + 1,
      pathNodes: path.nodes,
      pathEdges: path.steps.map(s => s.edge),
      bottleneck,
      value: this.value,
    };
    this.trace.push(row);
    this.logger?.log({kind: 'max-flow-augment', level: 'info', ...row});
    this.emit(new AugmentingPathToken(row));
  }

  currentValue(): number {
    return this.value;
  }

  edgeFlows(): FlowEdgeResult[] {
    return this.params.edges.map((e, i) => ({
      ...e,
      flow: this.flow[i],
      residual: e.capacity - this.flow[i],
    }));
  }

  result(validation: ValidationCheck[] = []): MaxFlowResult {
    return {
      params: this.params,
      maxFlow: this.value,
      edgeFlows: this.edgeFlows(),
      minCut: this.minCut(),
      trace: this.trace.slice(),
      validation,
    };
  }

  minCut(): MaxFlowMinCut {
    const seen = this.residualReachable();
    const sourceSide: number[] = [];
    const sinkSide: number[] = [];
    for (let v = 0; v < this.params.numNodes; v++) (seen[v] ? sourceSide : sinkSide).push(v);
    const cutEdges: number[] = [];
    let capacity = 0;
    for (let i = 0; i < this.params.edges.length; i++) {
      const e = this.params.edges[i];
      if (seen[e.from] && !seen[e.to]) {
        cutEdges.push(i);
        capacity += e.capacity;
      }
    }
    return {sourceSide, sinkSide, cutEdges, capacity};
  }

  private flowConservationOk(): boolean {
    const balance = new Array(this.params.numNodes).fill(0);
    for (let i = 0; i < this.params.edges.length; i++) {
      const e = this.params.edges[i];
      balance[e.from] -= this.flow[i];
      balance[e.to] += this.flow[i];
    }
    for (let v = 0; v < balance.length; v++) {
      if (v === this.params.source || v === this.params.sink) continue;
      if (Math.abs(balance[v]) > 1e-7) return false;
    }
    return Math.abs(balance[this.params.sink] - this.value) <= 1e-7 &&
           Math.abs(balance[this.params.source] + this.value) <= 1e-7;
  }

  private residualCapacity(step: ResidualStep): number {
    const e = this.params.edges[step.edge];
    return step.dir === 1 ? e.capacity - this.flow[step.edge] : this.flow[step.edge];
  }

  private neighbors(u: number): Array<{v: number; step: ResidualStep}> {
    const out: Array<{v: number; step: ResidualStep}> = [];
    for (let i = 0; i < this.params.edges.length; i++) {
      const e = this.params.edges[i];
      if (e.from === u && e.capacity - this.flow[i] > 1e-9) out.push({v: e.to, step: {edge: i, dir: 1}});
      if (e.to === u && this.flow[i] > 1e-9) out.push({v: e.from, step: {edge: i, dir: -1}});
    }
    return out;
  }

  private findAugmentingPath(): {nodes: number[]; steps: ResidualStep[]} | null {
    const parent: Array<{prev: number; step: ResidualStep} | null> = new Array(this.params.numNodes).fill(null);
    const q: number[] = [this.params.source];
    parent[this.params.source] = {prev: -1, step: {edge: -1, dir: 1}};
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      if (u === this.params.sink) break;
      for (const nb of this.neighbors(u)) {
        if (parent[nb.v] !== null) continue;
        parent[nb.v] = {prev: u, step: nb.step};
        q.push(nb.v);
      }
    }
    if (parent[this.params.sink] === null) return null;
    const nodes: number[] = [];
    const steps: ResidualStep[] = [];
    let cur = this.params.sink;
    while (cur !== this.params.source) {
      const p = parent[cur];
      if (!p) return null;
      nodes.push(cur);
      steps.push(p.step);
      cur = p.prev;
    }
    nodes.push(this.params.source);
    nodes.reverse();
    steps.reverse();
    return {nodes, steps};
  }

  private residualReachable(): boolean[] {
    const seen = new Array(this.params.numNodes).fill(false);
    const q = [this.params.source];
    seen[this.params.source] = true;
    for (let qi = 0; qi < q.length; qi++) {
      for (const nb of this.neighbors(q[qi])) {
        if (seen[nb.v]) continue;
        seen[nb.v] = true;
        q.push(nb.v);
      }
    }
    return seen;
  }
}

export function runMaxFlow(params: MaxFlowParams, logger?: OptimizationLogger): MaxFlowResult {
  const station = new MaxFlowOptimizationStation(params, logger);
  const summary = runIterativeDES([station], {shuffle: false, maxTicks: (params.maxAugmentations ?? params.edges.length * params.numNodes + 1) + 2});
  return station.result(summary.validation ?? []);
}

// -----------------------------------------------------------------------------
// Continuous-time traffic flow on a stationary grid.
// -----------------------------------------------------------------------------

export type TrafficNodeKind = 'source' | 'intersection' | 'sink';

export interface TrafficNode {
  id: string;
  kind: TrafficNodeKind;
  x: number;
  y: number;
}

export interface TrafficLane {
  id: string;
  from: string;
  to: string;
  lengthM: number;
  speedLimitMps: number;
  capacity?: number;
}

export interface TrafficSignalPhase {
  name: string;
  greenLanes: string[];
  durationSec: number;
}

export interface TrafficSignal {
  nodeId: string;
  phases: TrafficSignalPhase[];
  offsetSec?: number;
}

export interface TrafficSource {
  id: string;
  nodeId: string;
  ratePerMin: number;
  destinationSinkIds?: string[];
}

export interface TrafficSink {
  id: string;
  nodeId: string;
}

export interface TrafficNetwork {
  nodes: TrafficNode[];
  lanes: TrafficLane[];
  signals?: TrafficSignal[];
  sources: TrafficSource[];
  sinks: TrafficSink[];
}

export interface TrafficScheduledTrip {
  departSec: number;
  sourceId: string;
  destinationSinkId: string;
}

export interface TrafficParams {
  builtin?: 'five-intersection';
  network?: TrafficNetwork;
  durationSec: number;
  dtSec: number;
  seed: number;
  maxCars: number;
  carLengthM?: number;
  carWidthM?: number;
  laneWidthM?: number;
  minGapM?: number;
  maxAccelMps2?: number;
  maxDecelMps2?: number;
  maxJerkMps3?: number;
  reactionTimeSec?: number;
  timeHeadwaySec?: number;
  gridCellSizeM?: number;
  gridLookAheadM?: number;
  spawnRateMultiplier?: number;
  scheduledTrips?: TrafficScheduledTrip[];
}

export interface TrafficCarSnapshot {
  id: number;
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
  gridCellIds: string[];
  gridCellCount: number;
  leaderId?: number;
  leaderGapM?: number;
}

export interface TrafficTraceRow {
  tick: number;
  timeSec: number;
  activeCars: number;
  entered: number;
  exited: number;
  meanSpeedMps: number;
  meanTravelTimeSec: number;
  queueLength: number;
  laneOccupancy: Record<string, number>;
  activeGridCells: number;
  signalPhases: Record<string, string>;
  cars: TrafficCarSnapshot[];
}

export interface TrafficCellStats {
  cellSizeM: number;
  laneWidthM: number;
  carWidthM: number;
  activeCells: number;
  createdCellStations: number;
  maxCellOccupancy: number;
}

export interface TrafficResult {
  params: TrafficParams;
  network: TrafficNetwork;
  trace: TrafficTraceRow[];
  finalCars: TrafficCarSnapshot[];
  entered: number;
  exited: number;
  dropped: number;
  meanTravelTimeSec: number;
  meanSpeedMps: number;
  maxActiveCars: number;
  cellStats: TrafficCellStats;
  validation: ValidationCheck[];
}

interface TrafficCar {
  id: number;
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
  gridCellIds: string[];
  gridCellCount: number;
  leaderId?: number;
  leaderGapM?: number;
  history: TrafficKinematicSample[];
}

export class CarToken implements Token {
  constructor(readonly car: TrafficCarSnapshot) {}
}

interface TrafficKinematicSample {
  timeSec: number;
  laneId: string;
  positionM: number;
  speedMps: number;
  accelerationMps2: number;
}

interface TrafficCellBounds {
  laneId: string;
  longitudinalIndex: number;
  lateralIndex: number;
  x0M: number;
  x1M: number;
  y0M: number;
  y1M: number;
}

export class TrafficCellStation extends DESStation {
  readonly carIds = new Set<number>();

  constructor(readonly bounds: TrafficCellBounds) {
    super(`traffic-cell-${bounds.laneId}-${bounds.longitudinalIndex}-${bounds.lateralIndex}`);
  }

  clearOccupancy(): void {
    this.carIds.clear();
  }

  occupy(carId: number): void {
    this.carIds.add(carId);
  }

  override hasWork(): boolean { return false; }
  runTimeStep(): void {}
}

interface TrafficSpatialIndex {
  byCell: Map<string, Set<number>>;
  cellIdsByCar: Map<number, string[]>;
  activeCellCount: number;
}

interface TrafficOptions {
  logger?: OptimizationLogger;
}

export class TrafficGridStation extends DESStation {
  private readonly network: TrafficNetwork;
  private readonly nodes = new Map<string, TrafficNode>();
  private readonly lanes = new Map<string, TrafficLane>();
  private readonly signalByNode = new Map<string, TrafficSignal>();
  private readonly outgoing = new Map<string, string[]>();
  private readonly routes = new Map<string, string[]>();
  private readonly sourceAccumulators = new Map<string, number>();
  private readonly cars = new Map<number, TrafficCar>();
  private readonly cellStations = new Map<string, TrafficCellStation>();
  private readonly activeCellIds = new Set<string>();
  private readonly rng: () => number;
  private nextCarId = 1;
  private tick = 0;
  private entered = 0;
  private exited = 0;
  private dropped = 0;
  private travelTimeSum = 0;
  private speedSum = 0;
  private speedSamples = 0;
  private maxActiveCars = 0;
  private maxCellOccupancy = 0;
  readonly trace: TrafficTraceRow[] = [];

  constructor(readonly params: TrafficParams, private readonly options: TrafficOptions = {}) {
    super('traffic-flow-grid');
    this.network = params.network ?? buildFiveIntersectionTrafficNetwork();
    this.rng = mulberry32(params.seed);
    for (const node of this.network.nodes) this.nodes.set(node.id, node);
    for (const lane of this.network.lanes) {
      this.lanes.set(lane.id, lane);
      const arr = this.outgoing.get(lane.from) ?? [];
      arr.push(lane.id);
      this.outgoing.set(lane.from, arr);
    }
    for (const signal of this.network.signals ?? []) this.signalByNode.set(signal.nodeId, signal);
    for (const source of this.network.sources) this.sourceAccumulators.set(source.id, 0);
    this.assertPreconditions();
    this.precomputeRoutes();
    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic-active-under-cap',
      predicate: s => s.maxActiveCars <= s.params.maxCars,
      observedFn: s => String(s.maxActiveCars),
      expected: 'active car count never exceeds maxCars',
      group: 'traffic-flow',
    }));
    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic-conservation',
      predicate: s => s.entered === s.exited + s.cars.size,
      observedFn: s => `entered=${s.entered} exited=${s.exited} active=${s.cars.size}`,
      expected: 'entered = exited + active',
      group: 'traffic-flow',
    }));
    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic-no-collisions',
      predicate: s => s.minimumHeadway() >= -1e-7,
      observedFn: s => s.minimumHeadway().toFixed(6),
      expected: 'same-lane cars remain separated by carLength+minGap',
      group: 'traffic-flow',
    }));
    this.addValidator(intrinsicCheck<TrafficGridStation>({
      name: 'traffic-grid-cell-size',
      predicate: s => s.gridCellSizeM() <= 0.3048 + 1e-12,
      observedFn: s => `${s.gridCellSizeM().toFixed(4)} m`,
      expected: 'default/selected spatial grid cells are at most about one foot on a side',
      group: 'traffic-flow',
    }));
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.check('TrafficGridStation', 'network', 'be provided by builtin or network', p.builtin === 'five-intersection' || p.network !== undefined, p);
    Preconditions.positive('TrafficGridStation', 'durationSec', p.durationSec);
    Preconditions.positive('TrafficGridStation', 'dtSec', p.dtSec);
    Preconditions.check('TrafficGridStation', 'dtSec', 'be <= 5 seconds', p.dtSec <= 5, p.dtSec);
    Preconditions.integer('TrafficGridStation', 'seed', p.seed);
    Preconditions.integerInRange('TrafficGridStation', 'maxCars', p.maxCars, 1, 299);
    if (p.carLengthM !== undefined) Preconditions.positive('TrafficGridStation', 'carLengthM', p.carLengthM);
    if (p.carWidthM !== undefined) Preconditions.positive('TrafficGridStation', 'carWidthM', p.carWidthM);
    if (p.laneWidthM !== undefined) Preconditions.positive('TrafficGridStation', 'laneWidthM', p.laneWidthM);
    if (p.minGapM !== undefined) Preconditions.nonNegative('TrafficGridStation', 'minGapM', p.minGapM);
    if (p.maxAccelMps2 !== undefined) Preconditions.positive('TrafficGridStation', 'maxAccelMps2', p.maxAccelMps2);
    if (p.maxDecelMps2 !== undefined) Preconditions.positive('TrafficGridStation', 'maxDecelMps2', p.maxDecelMps2);
    if (p.maxJerkMps3 !== undefined) Preconditions.positive('TrafficGridStation', 'maxJerkMps3', p.maxJerkMps3);
    if (p.reactionTimeSec !== undefined) Preconditions.nonNegative('TrafficGridStation', 'reactionTimeSec', p.reactionTimeSec);
    if (p.timeHeadwaySec !== undefined) Preconditions.nonNegative('TrafficGridStation', 'timeHeadwaySec', p.timeHeadwaySec);
    if (p.gridCellSizeM !== undefined) Preconditions.positive('TrafficGridStation', 'gridCellSizeM', p.gridCellSizeM);
    if (p.gridLookAheadM !== undefined) Preconditions.positive('TrafficGridStation', 'gridLookAheadM', p.gridLookAheadM);
    if (p.spawnRateMultiplier !== undefined) Preconditions.nonNegative('TrafficGridStation', 'spawnRateMultiplier', p.spawnRateMultiplier);
    Preconditions.check('TrafficGridStation', 'carWidthM', 'fit within laneWidthM', this.carWidthM() <= this.laneWidthM(), {carWidthM: this.carWidthM(), laneWidthM: this.laneWidthM()});
    validateTrafficNetwork(this.network);
    const maxTicks = Math.ceil(p.durationSec / p.dtSec);
    Preconditions.integerInRange('TrafficGridStation', 'tick count', maxTicks, 1, 100000);
  }

  override hasWork(): boolean {
    return this.tick * this.params.dtSec < this.params.durationSec - 1e-9;
  }

  runTimeStep(): void {
    const timeSec = this.tick * this.params.dtSec;
    this.spawnCars(timeSec);
    this.advanceCars(timeSec);
    this.rebuildSpatialIndex();
    this.maxActiveCars = Math.max(this.maxActiveCars, this.cars.size);
    this.trace.push(this.snapshot(timeSec));
    this.options.logger?.log({
      kind: 'traffic-flow-tick',
      level: 'debug',
      tick: this.tick,
      timeSec,
      activeCars: this.cars.size,
      entered: this.entered,
      exited: this.exited,
      dropped: this.dropped,
    });
    this.tick++;
  }

  result(validation: ValidationCheck[] = []): TrafficResult {
    const finalCars = this.snapCars();
    const meanTravelTimeSec = this.exited > 0 ? this.travelTimeSum / this.exited : 0;
    const meanSpeedMps = this.speedSamples > 0 ? this.speedSum / this.speedSamples : 0;
    return {
      params: this.params,
      network: this.network,
      trace: this.trace.slice(),
      finalCars,
      entered: this.entered,
      exited: this.exited,
      dropped: this.dropped,
      meanTravelTimeSec,
      meanSpeedMps,
      maxActiveCars: this.maxActiveCars,
      cellStats: {
        cellSizeM: this.gridCellSizeM(),
        laneWidthM: this.laneWidthM(),
        carWidthM: this.carWidthM(),
        activeCells: this.activeCellIds.size,
        createdCellStations: this.cellStations.size,
        maxCellOccupancy: this.maxCellOccupancy,
      },
      validation,
    };
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

  private spawnCars(timeSec: number): void {
    const dt = this.params.dtSec;
    const mult = this.params.spawnRateMultiplier ?? 1;
    for (const source of this.network.sources) {
      const expected = source.ratePerMin * mult * dt / 60;
      let acc = (this.sourceAccumulators.get(source.id) ?? 0) + expected;
      const count = Math.floor(acc);
      acc -= count;
      this.sourceAccumulators.set(source.id, acc);
      for (let k = 0; k < count; k++) this.trySpawnFromSource(source, timeSec);
    }
  }

  private trySpawnFromSource(source: TrafficSource, timeSec: number): void {
    if (this.cars.size >= this.params.maxCars) {
      this.dropped++;
      return;
    }
    const sinkIds = source.destinationSinkIds ?? this.network.sinks.map(s => s.id);
    const sinkId = sinkIds[Math.floor(this.rng() * sinkIds.length)];
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
    const car: TrafficCar = {
      id: this.nextCarId++,
      laneId: lane.id,
      positionM: 0,
      speedMps: Math.min(2, lane.speedLimitMps),
      accelerationMps2: 0,
      jerkMps3: 0,
      targetAccelerationMps2: 0,
      route,
      routeIndex: 0,
      destinationSinkId: sinkId,
      createdAtSec: timeSec,
      waitSec: 0,
      gridCellIds: [],
      gridCellCount: 0,
      history: [],
    };
    this.recordHistory(car, timeSec);
    this.cars.set(car.id, car);
    this.entered++;
    this.emit(new CarToken(toCarSnapshot(car)));
  }

  private advanceCars(timeSec: number): void {
    const dt = this.params.dtSec;
    const spatial = this.rebuildSpatialIndex();
    const laneGroups = this.carsByLane();
    const updates: Array<{
      car: TrafficCar;
      speed: number;
      position: number;
      acceleration: number;
      jerk: number;
      targetAcceleration: number;
      leaderId?: number;
      leaderGapM?: number;
    }> = [];
    for (const [laneId, cars] of laneGroups.entries()) {
      const lane = this.lane(laneId);
      cars.sort((a, b) => b.positionM - a.positionM);
      for (let i = 0; i < cars.length; i++) {
        const car = cars[i];
        const sortedLeader = i === 0 ? undefined : cars[i - 1];
        const leader = this.findLeaderAheadFromGrid(car, lane, spatial) ?? sortedLeader;
        updates.push({car, ...this.nextKinematics(car, lane, leader, timeSec)});
      }
    }
    for (const u of updates) {
      if (!this.cars.has(u.car.id)) continue;
      u.car.jerkMps3 = u.jerk;
      u.car.accelerationMps2 = u.acceleration;
      u.car.targetAccelerationMps2 = u.targetAcceleration;
      u.car.leaderId = u.leaderId;
      u.car.leaderGapM = u.leaderGapM;
      u.car.speedMps = u.speed;
      u.car.positionM = u.position;
      if (u.speed < 0.5) u.car.waitSec += dt;
      this.speedSum += u.speed;
      this.speedSamples++;
      this.handleLaneEnd(u.car, timeSec + dt);
      this.recordHistory(u.car, timeSec + dt);
    }
  }

  private nextKinematics(car: TrafficCar, lane: TrafficLane, leader: TrafficCar | undefined, timeSec: number): {
    speed: number;
    position: number;
    acceleration: number;
    jerk: number;
    targetAcceleration: number;
    leaderId?: number;
    leaderGapM?: number;
  } {
    const dt = this.params.dtSec;
    const vehicleSpace = this.vehicleSpace();
    const barrier = this.stopBarrierPosition(car, lane, timeSec);
    const leaderPosition = leader?.positionM ?? Infinity;
    const currentLeaderGap = Number.isFinite(leaderPosition)
      ? leaderPosition - car.positionM - vehicleSpace
      : Infinity;
    const barrierGap = barrier === null ? Infinity : barrier - car.positionM - vehicleSpace;
    const useBarrier = barrierGap <= currentLeaderGap;
    const delayedLeader = leader ? this.perceivedSample(leader, timeSec - this.reactionTimeSec()) : null;
    const perceivedLeader = delayedLeader && delayedLeader.laneId === car.laneId
      ? delayedLeader
      : leader;
    const perceived = useBarrier
      ? {positionM: barrier ?? Infinity, speedMps: 0, id: undefined as number | undefined}
      : perceivedLeader
        ? {positionM: perceivedLeader.positionM, speedMps: perceivedLeader.speedMps, id: leader?.id}
        : {positionM: Infinity, speedMps: lane.speedLimitMps, id: undefined as number | undefined};
    const perceivedGap = Math.max(0.05, perceived.positionM - car.positionM - vehicleSpace);
    const maxAccel = this.maxAccelMps2();
    const maxDecel = this.maxDecelMps2();
    const v = Math.max(0, car.speedMps);
    const v0 = lane.speedLimitMps;
    const timeHeadway = this.timeHeadwaySec() + this.reactionTimeSec();
    const closingTerm = Math.max(0, v * (v - perceived.speedMps) / (2 * Math.sqrt(maxAccel * maxDecel)));
    const desiredGap = vehicleSpace + v * timeHeadway + closingTerm;
    const freeRoad = 1 - Math.pow(Math.min(v / Math.max(1e-9, v0), 2), 2);
    const interaction = Number.isFinite(perceived.positionM)
      ? Math.pow(desiredGap / perceivedGap, 2)
      : 0;
    const targetAcceleration = clamp(maxAccel * (freeRoad - interaction), -maxDecel, maxAccel);
    const maxJerkStep = this.maxJerkMps3() * dt;
    const acceleration = clamp(
      car.accelerationMps2 + clamp(targetAcceleration - car.accelerationMps2, -maxJerkStep, maxJerkStep),
      -maxDecel,
      maxAccel,
    );
    let speed = clamp(v + acceleration * dt, 0, v0);
    let position = car.positionM + v * dt + 0.5 * acceleration * dt * dt;

    const hardLimit = this.hardPositionLimit(car, lane, leader, barrier);
    if (position > hardLimit) {
      position = Math.max(car.positionM, hardLimit);
      speed = Math.max(0, Math.min(speed, (position - car.positionM) / dt));
    }
    const realizedAcceleration = clamp((speed - v) / dt, -maxDecel, maxAccel);
    const jerk = clamp((realizedAcceleration - car.accelerationMps2) / dt, -this.maxJerkMps3(), this.maxJerkMps3());
    return {
      speed,
      position,
      acceleration: realizedAcceleration,
      jerk,
      targetAcceleration,
      leaderId: perceived.id,
      leaderGapM: Number.isFinite(perceived.positionM) ? Math.max(0, perceivedGap) : undefined,
    };
  }

  private stopBarrierPosition(car: TrafficCar, lane: TrafficLane, timeSec: number): number | null {
    const nextLaneId = car.route[car.routeIndex + 1];
    if (nextLaneId === undefined) return null;
    if (this.signalAllows(lane.id, timeSec) && this.canEnterLane(nextLaneId, car.id)) return null;
    return lane.lengthM;
  }

  private hardPositionLimit(car: TrafficCar, lane: TrafficLane, leader: TrafficCar | undefined, barrier: number | null): number {
    let limit = Infinity;
    if (leader) limit = Math.min(limit, leader.positionM - this.vehicleSpace());
    if (barrier !== null) limit = Math.min(limit, barrier - this.vehicleSpace());
    if (!Number.isFinite(limit)) return lane.lengthM + Math.max(0, car.speedMps * this.params.dtSec);
    return Math.max(car.positionM, limit);
  }

  private handleLaneEnd(car: TrafficCar, timeSec: number): void {
    let lane = this.lane(car.laneId);
    if (car.positionM < lane.lengthM - 1e-9) return;
    let overshoot = car.positionM - lane.lengthM;
    while (car.positionM >= lane.lengthM - 1e-9) {
      const nextLaneId = car.route[car.routeIndex + 1];
      if (nextLaneId === undefined) {
        this.cars.delete(car.id);
        this.exited++;
        this.travelTimeSum += timeSec - car.createdAtSec;
        return;
      }
      if (!this.signalAllows(lane.id, timeSec) || !this.canEnterLane(nextLaneId, car.id)) {
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

  private blockedStopPosition(car: TrafficCar): number {
    const lane = this.lane(car.laneId);
    const vehicleSpace = this.vehicleSpace();
    let safe = Math.max(0, lane.lengthM - vehicleSpace);
    const others = Array.from(this.cars.values())
      .filter(c => c.id !== car.id && c.laneId === car.laneId)
      .sort((a, b) => b.positionM - a.positionM);
    for (const other of others) {
      if (other.positionM <= safe + vehicleSpace) safe = Math.min(safe, other.positionM - vehicleSpace);
    }
    return Math.max(0, safe);
  }

  private signalAllows(incomingLaneId: string, timeSec: number): boolean {
    const lane = this.lane(incomingLaneId);
    const node = this.nodes.get(lane.to);
    if (!node || node.kind !== 'intersection') return true;
    const signal = this.signalByNode.get(node.id);
    if (!signal) return true;
    return currentSignalPhase(signal, timeSec).greenLanes.includes(incomingLaneId);
  }

  private canEnterLane(laneId: string, ignoreCarId?: number): boolean {
    const lane = this.lane(laneId);
    const cars = Array.from(this.cars.values()).filter(c => c.laneId === laneId && c.id !== ignoreCarId);
    const cap = lane.capacity ?? defaultLaneCapacity(lane, this.vehicleSpace());
    if (cars.length >= cap) return false;
    const entryClear = cars.every(c => c.positionM >= this.vehicleSpace());
    return entryClear;
  }

  private rebuildSpatialIndex(): TrafficSpatialIndex {
    for (const cell of this.cellStations.values()) cell.clearOccupancy();
    this.activeCellIds.clear();
    const byCell = new Map<string, Set<number>>();
    const cellIdsByCar = new Map<number, string[]>();
    for (const car of this.cars.values()) {
      const cellIds = this.occupiedCellIds(car);
      car.gridCellIds = cellIds;
      car.gridCellCount = cellIds.length;
      cellIdsByCar.set(car.id, cellIds);
      for (const cellId of cellIds) {
        const station = this.ensureCellStation(cellId);
        station.occupy(car.id);
        this.maxCellOccupancy = Math.max(this.maxCellOccupancy, station.carIds.size);
        this.activeCellIds.add(cellId);
        const set = byCell.get(cellId) ?? new Set<number>();
        set.add(car.id);
        byCell.set(cellId, set);
      }
    }
    return {byCell, cellIdsByCar, activeCellCount: this.activeCellIds.size};
  }

  private findLeaderAheadFromGrid(car: TrafficCar, lane: TrafficLane, spatial: TrafficSpatialIndex): TrafficCar | undefined {
    const cellSize = this.gridCellSizeM();
    const lookAhead = Math.min(
      lane.lengthM - car.positionM,
      this.params.gridLookAheadM ?? Math.max(60, car.speedMps * (this.reactionTimeSec() + 4) + 3 * this.vehicleSpace()),
    );
    const first = Math.max(0, Math.floor(car.positionM / cellSize));
    const last = Math.max(first, Math.floor((car.positionM + lookAhead) / cellSize));
    const lateral = this.occupiedLateralCellRange();
    let best: TrafficCar | undefined;
    for (let x = first; x <= last; x++) {
      for (let y = lateral.start; y <= lateral.end; y++) {
        const ids = spatial.byCell.get(this.cellId(lane.id, x, y));
        if (!ids) continue;
        for (const id of ids) {
          if (id === car.id) continue;
          const other = this.cars.get(id);
          if (!other || other.laneId !== car.laneId || other.positionM <= car.positionM) continue;
          if (!best || other.positionM < best.positionM) best = other;
        }
      }
      if (best && best.positionM <= (x + 1) * cellSize) break;
    }
    return best;
  }

  private carsByLane(): Map<string, TrafficCar[]> {
    const groups = new Map<string, TrafficCar[]>();
    for (const car of this.cars.values()) {
      const arr = groups.get(car.laneId) ?? [];
      arr.push(car);
      groups.set(car.laneId, arr);
    }
    return groups;
  }

  private snapshot(timeSec: number): TrafficTraceRow {
    const laneOccupancy: Record<string, number> = {};
    for (const lane of this.network.lanes) laneOccupancy[lane.id] = 0;
    for (const car of this.cars.values()) laneOccupancy[car.laneId] = (laneOccupancy[car.laneId] ?? 0) + 1;
    const signalPhases: Record<string, string> = {};
    for (const signal of this.network.signals ?? []) signalPhases[signal.nodeId] = currentSignalPhase(signal, timeSec).name;
    const cars = this.snapCars();
    const meanSpeedMps = cars.length > 0 ? cars.reduce((s, c) => s + c.speedMps, 0) / cars.length : 0;
    const meanTravelTimeSec = this.exited > 0 ? this.travelTimeSum / this.exited : 0;
    const queueLength = cars.filter(c => c.speedMps < 0.5).length;
    return {
      tick: this.tick,
      timeSec,
      activeCars: cars.length,
      entered: this.entered,
      exited: this.exited,
      meanSpeedMps,
      meanTravelTimeSec,
      queueLength,
      laneOccupancy,
      activeGridCells: this.activeCellIds.size,
      signalPhases,
      cars,
    };
  }

  private snapCars(): TrafficCarSnapshot[] {
    return Array.from(this.cars.values()).sort((a, b) => a.id - b.id).map(toCarSnapshot);
  }

  private minimumHeadway(): number {
    let min = Infinity;
    const groups = this.carsByLane();
    for (const cars of groups.values()) {
      cars.sort((a, b) => a.positionM - b.positionM);
      for (let i = 1; i < cars.length; i++) {
        min = Math.min(min, cars[i].positionM - cars[i - 1].positionM - this.vehicleSpace());
      }
    }
    return Number.isFinite(min) ? min : 0;
  }

  private lane(id: string): TrafficLane {
    const lane = this.lanes.get(id);
    if (!lane) throw new Error(`traffic-flow: unknown lane "${id}"`);
    return lane;
  }

  private vehicleSpace(): number {
    return (this.params.carLengthM ?? 4.8) + (this.params.minGapM ?? 2.5);
  }

  private carLengthM(): number { return this.params.carLengthM ?? 4.8; }
  private carWidthM(): number { return this.params.carWidthM ?? 1.8; }
  private laneWidthM(): number { return this.params.laneWidthM ?? 3.7; }
  private gridCellSizeM(): number { return this.params.gridCellSizeM ?? 0.3048; }
  private maxAccelMps2(): number { return this.params.maxAccelMps2 ?? 2.2; }
  private maxDecelMps2(): number { return this.params.maxDecelMps2 ?? 4.0; }
  private maxJerkMps3(): number { return this.params.maxJerkMps3 ?? 6.0; }
  private reactionTimeSec(): number { return this.params.reactionTimeSec ?? 0.8; }
  private timeHeadwaySec(): number { return this.params.timeHeadwaySec ?? 1.1; }

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

  private occupiedCellIds(car: TrafficCar): string[] {
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

  private ensureCellStation(cellId: string): TrafficCellStation {
    const existing = this.cellStations.get(cellId);
    if (existing) return existing;
    const parsed = this.parseCellId(cellId);
    const cellSize = this.gridCellSizeM();
    const station = new TrafficCellStation({
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

  private perceivedSample(car: TrafficCar, targetTimeSec: number): TrafficKinematicSample {
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

  private recordHistory(car: TrafficCar, timeSec: number): void {
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
}

function toCarSnapshot(car: TrafficCar): TrafficCarSnapshot {
  return {
    id: car.id,
    laneId: car.laneId,
    positionM: car.positionM,
    speedMps: car.speedMps,
    accelerationMps2: car.accelerationMps2,
    jerkMps3: car.jerkMps3,
    targetAccelerationMps2: car.targetAccelerationMps2,
    route: car.route.slice(),
    routeIndex: car.routeIndex,
    destinationSinkId: car.destinationSinkId,
    createdAtSec: car.createdAtSec,
    waitSec: car.waitSec,
    gridCellIds: car.gridCellIds.slice(),
    gridCellCount: car.gridCellCount,
    leaderId: car.leaderId,
    leaderGapM: car.leaderGapM,
  };
}

function validateTrafficNetwork(network: TrafficNetwork): void {
  Preconditions.nonEmpty('TrafficGridStation', 'network.nodes', network.nodes);
  Preconditions.nonEmpty('TrafficGridStation', 'network.lanes', network.lanes);
  Preconditions.nonEmpty('TrafficGridStation', 'network.sources', network.sources);
  Preconditions.nonEmpty('TrafficGridStation', 'network.sinks', network.sinks);
  const nodeIds = new Set<string>();
  for (const node of network.nodes) {
    Preconditions.check('TrafficGridStation', 'node.id', 'be unique and non-empty', node.id.length > 0 && !nodeIds.has(node.id), node.id);
    nodeIds.add(node.id);
    Preconditions.finite('TrafficGridStation', `node.${node.id}.x`, node.x);
    Preconditions.finite('TrafficGridStation', `node.${node.id}.y`, node.y);
    Preconditions.check('TrafficGridStation', `node.${node.id}.kind`, 'be source, intersection, or sink', ['source', 'intersection', 'sink'].includes(node.kind), node.kind);
  }
  const laneIds = new Set<string>();
  for (const lane of network.lanes) {
    Preconditions.check('TrafficGridStation', 'lane.id', 'be unique and non-empty', lane.id.length > 0 && !laneIds.has(lane.id), lane.id);
    laneIds.add(lane.id);
    Preconditions.check('TrafficGridStation', `lane.${lane.id}.from`, 'reference a node', nodeIds.has(lane.from), lane.from);
    Preconditions.check('TrafficGridStation', `lane.${lane.id}.to`, 'reference a node', nodeIds.has(lane.to), lane.to);
    Preconditions.positive('TrafficGridStation', `lane.${lane.id}.lengthM`, lane.lengthM);
    Preconditions.positive('TrafficGridStation', `lane.${lane.id}.speedLimitMps`, lane.speedLimitMps);
    if (lane.capacity !== undefined) Preconditions.integerInRange('TrafficGridStation', `lane.${lane.id}.capacity`, lane.capacity, 1, 10000);
  }
  for (const signal of network.signals ?? []) {
    Preconditions.check('TrafficGridStation', `signal.${signal.nodeId}.nodeId`, 'reference a node', nodeIds.has(signal.nodeId), signal.nodeId);
    Preconditions.nonEmpty('TrafficGridStation', `signal.${signal.nodeId}.phases`, signal.phases);
    for (const phase of signal.phases) {
      Preconditions.positive('TrafficGridStation', `signal.${signal.nodeId}.phase.${phase.name}.durationSec`, phase.durationSec);
      for (const laneId of phase.greenLanes) Preconditions.check('TrafficGridStation', `signal.${signal.nodeId}.greenLanes`, 'reference a lane', laneIds.has(laneId), laneId);
    }
  }
  for (const source of network.sources) {
    Preconditions.check('TrafficGridStation', `source.${source.id}.nodeId`, 'reference a source node', nodeIds.has(source.nodeId), source.nodeId);
    Preconditions.nonNegative('TrafficGridStation', `source.${source.id}.ratePerMin`, source.ratePerMin);
  }
  const sinkIds = new Set(network.sinks.map(s => s.id));
  for (const sink of network.sinks) {
    Preconditions.check('TrafficGridStation', `sink.${sink.id}.nodeId`, 'reference a sink node', nodeIds.has(sink.nodeId), sink.nodeId);
  }
  for (const source of network.sources) {
    for (const sinkId of source.destinationSinkIds ?? Array.from(sinkIds)) {
      Preconditions.check('TrafficGridStation', `source.${source.id}.destinationSinkIds`, 'reference a sink id', sinkIds.has(sinkId), sinkId);
      const sink = network.sinks.find(s => s.id === sinkId);
      if (sink) {
        Preconditions.check(
          'TrafficGridStation',
          `route ${source.id}->${sinkId}`,
          'have at least one directed lane path',
          shortestLanePath(network, source.nodeId, sink.nodeId).length > 0,
          {source: source.nodeId, sink: sink.nodeId},
        );
      }
    }
  }
}

function defaultLaneCapacity(lane: TrafficLane, vehicleSpace: number): number {
  return Math.max(1, Math.floor(lane.lengthM / vehicleSpace));
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
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

export function buildFiveIntersectionTrafficNetwork(): TrafficNetwork {
  const nodes: TrafficNode[] = [
    {id: 'W', kind: 'source', x: 0, y: 1},
    {id: 'S0', kind: 'source', x: 1, y: 2},
    {id: 'N2', kind: 'source', x: 3, y: 0},
    {id: 'I0', kind: 'intersection', x: 1, y: 1},
    {id: 'I1', kind: 'intersection', x: 2, y: 1},
    {id: 'I2', kind: 'intersection', x: 3, y: 1},
    {id: 'I3', kind: 'intersection', x: 4, y: 1},
    {id: 'I4', kind: 'intersection', x: 5, y: 1},
    {id: 'E', kind: 'sink', x: 6, y: 1},
    {id: 'N1', kind: 'sink', x: 2, y: 0},
    {id: 'S4', kind: 'sink', x: 5, y: 2},
  ];
  const mk = (id: string, from: string, to: string, lengthM = 120): TrafficLane => ({id, from, to, lengthM, speedLimitMps: 13.5});
  const lanes: TrafficLane[] = [
    mk('W-I0', 'W', 'I0', 90),
    mk('S0-I0', 'S0', 'I0', 85),
    mk('I0-I1', 'I0', 'I1'),
    mk('I1-I2', 'I1', 'I2'),
    mk('N2-I2', 'N2', 'I2', 90),
    mk('I2-I3', 'I2', 'I3'),
    mk('I3-I4', 'I3', 'I4'),
    mk('I4-E', 'I4', 'E', 100),
    mk('I1-N1', 'I1', 'N1', 80),
    mk('I4-S4', 'I4', 'S4', 85),
  ];
  const phase = (name: string, greenLanes: string[], durationSec: number): TrafficSignalPhase => ({name, greenLanes, durationSec});
  const signals: TrafficSignal[] = [
    {nodeId: 'I0', phases: [phase('main', ['W-I0'], 28), phase('side', ['S0-I0'], 16)]},
    {nodeId: 'I1', phases: [phase('main', ['I0-I1'], 30)]},
    {nodeId: 'I2', phases: [phase('main', ['I1-I2'], 26), phase('side', ['N2-I2'], 18)], offsetSec: 5},
    {nodeId: 'I3', phases: [phase('main', ['I2-I3'], 30)]},
    {nodeId: 'I4', phases: [phase('main', ['I3-I4'], 26)]},
  ];
  const sources: TrafficSource[] = [
    {id: 'west', nodeId: 'W', ratePerMin: 18, destinationSinkIds: ['east', 'north1', 'south4']},
    {id: 'south0', nodeId: 'S0', ratePerMin: 7, destinationSinkIds: ['east', 'north1', 'south4']},
    {id: 'north2', nodeId: 'N2', ratePerMin: 8, destinationSinkIds: ['east', 'south4']},
  ];
  const sinks: TrafficSink[] = [
    {id: 'east', nodeId: 'E'},
    {id: 'north1', nodeId: 'N1'},
    {id: 'south4', nodeId: 'S4'},
  ];
  return {nodes, lanes, signals, sources, sinks};
}

export function runTrafficFlow(params: TrafficParams, logger?: OptimizationLogger): TrafficResult {
  const station = new TrafficGridStation(params, {logger});
  const summary = runIterativeDES([station], {
    shuffle: false,
    maxTicks: Math.ceil(params.durationSec / params.dtSec) + 1,
  });
  return station.result(summary.validation ?? []);
}
