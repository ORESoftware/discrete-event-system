'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/internal-solver-network.rs  (module des::general::internal_solver_network)
// 1:1 file move. Runnable DES station networks for SP/knapsack/TSP solvers with a wall-clock cap.
//
// Declarations → Rust:
//   const SOLUTION_CHANNEL/STOP_CHANNEL -> `const &str` (or ChannelId enum)
//   type InternalSolverKind = ... -> enum; interface Solver*/Snapshot*/*Params/*Result -> structs
//   class SolverSolutionToken/StopSignalToken (impl Token) -> structs `impl Token`
//   class WallClockCheckerStation/SolutionSinkStation/ShortestPathSolverStation/KnapsackDPStation/
//         KnapsackSAStation/ObservableTSP{SA,GA}Optimizer/TSPHeldKarpStation -> structs `impl` station traits
//   interface SnapshotProvider     -> trait SnapshotProvider
//   fn runInternalSolverNetwork    -> fn (or StatefulTransform)
//
// Conversion notes (file-specific):
//   - `WallClockCheckerStation` uses wall-clock time (Date.now) -> inject a `Clock` (shared/capabilities),
//     do NOT call the system clock directly; this makes the 3-min cap deterministic/testable.
//   - `mulberry32(seed)` in SA/GA solvers -> inject `RandomSource`.
//   - `extends X implements SnapshotProvider` (e.g. KnapsackSAStation) -> struct + multiple trait impls.
//   - `InternalSolverKind` union -> enum; tokens are nominal `impl Token`; channels -> typed queues.
// =============================================================================

// =============================================================================
// Internal solver networks.
//
// These are runnable DES station networks for common optimization/search
// problems. Solvers are stationary entities, incumbent/best solutions are
// moving tokens, and a wall-clock checker station provides the "cap most runs
// around 3 minutes" stop condition without relying on external solvers.
// =============================================================================

import {
  DESStation,
  Token,
  ValidationCheck,
  intrinsicCheck,
  runIterativeDES,
  IterativeRunSummary,
  SingleStateOptimizer,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';
import {
  Graph,
  buildRandomGraph,
  buildSmallChainGraph,
} from './shortest-path-des';
import {
  TSPInstance,
  Tour,
  buildPentagonTSP,
  buildRandomTSP,
  heldKarpExact,
  isPermutation,
  tourLength,
} from './genetic-tsp';
import {
  TSPGAOptimizer,
  TSPGAOptions,
} from './ga-des';
import {
  CoolingSchedule,
  TSPSAOptimizer,
  TSPSAOptions,
  temperatureAt,
} from './sa-des';

export const SOLUTION_CHANNEL = 'solution';
export const STOP_CHANNEL = 'stop';

export type InternalSolverKind =
  | 'shortest-path'
  | 'knapsack-dp'
  | 'knapsack-sa'
  | 'tsp-sa'
  | 'tsp-ga'
  | 'tsp-held-karp';

export interface SolverProgressPayload {
  solverId: string;
  solverKind: InternalSolverKind;
  tick: number;
  iteration: number;
  objective: number;
  bestState: unknown;
  feasible: boolean;
  done: boolean;
  metadata?: Record<string, unknown>;
}

export class SolverSolutionToken implements Token {
  readonly kind = 'solver-solution';
  constructor(readonly payload: SolverProgressPayload) {}
}

export class StopSignalToken implements Token {
  readonly kind = 'stop-signal';
  constructor(readonly payload: {checkerId: string; elapsedMs: number; budgetMs: number; tick: number}) {}
}

export interface SolverNetworkNode {
  id: string;
  kind: string;
  role: 'solver' | 'checker' | 'sink' | 'source';
}

export interface SolverNetworkEdge {
  from: string;
  to: string;
  movingEntity: string;
  channel: string;
}

export interface SolverNetworkDescription {
  stationaryEntities: SolverNetworkNode[];
  movingEntities: Array<{id: string; kind: string; tokenType: string}>;
  edges: SolverNetworkEdge[];
}

export interface InternalSolverRunResult {
  kind: InternalSolverKind;
  status: 'complete' | 'time-limit' | 'tick-limit';
  runSummary: IterativeRunSummary;
  best: SolverProgressPayload;
  trace: SolverProgressPayload[];
  stopSignals: StopSignalToken['payload'][];
  wallClock: {
    budgetMs: number;
    elapsedMs: number;
    checks: number;
    expired: boolean;
  };
  network: SolverNetworkDescription;
  validation: ValidationCheck[];
}

export interface InternalSolverRunParams {
  kind: InternalSolverKind;
  timeLimitMs?: number;
  maxTicks?: number;
  checkEveryTicks?: number;
  shortestPath?: ShortestPathSolverParams;
  knapsack?: KnapsackParams;
  tsp?: TSPSolverParams;
}

export interface ShortestPathSolverParams {
  algorithm: 'bellman-ford' | 'dijkstra';
  source: number;
  builtin?: 'small-chain';
  graph?: Graph;
  randomGraph?: {
    numNodes: number;
    edgeProb: number;
    wMin: number;
    wMax: number;
    seed: number;
  };
}

export interface KnapsackParams {
  values: number[];
  weights: number[];
  capacity: number;
  seed?: number;
  maxIterations?: number;
  cooling?: CoolingSchedule;
  stallLimit?: number;
  penalty?: number;
}

export interface TSPSolverParams {
  builtin?: 'pentagon' | 'random';
  n?: number;
  seed?: number;
  coordinates?: Array<[number, number]>;
  distance?: number[][];
  precedence?: Array<[number, number]>;
  sa?: Partial<TSPSAOptions>;
  ga?: Partial<TSPGAOptions>;
}

export interface SnapshotProvider {
  snapshot(done: boolean): SolverProgressPayload;
}

export class WallClockCheckerStation extends DESStation {
  private readonly startedAt: number;
  private tick = 0;
  private expiredFlag = false;
  private elapsed = 0;
  private checks = 0;

  constructor(
    id: string,
    readonly budgetMs: number = 180000,
    readonly checkEveryTicks: number = 1,
    private readonly now: () => number = () => Date.now(),
  ) {
    super(id);
    this.startedAt = this.now();
    this.addValidator(intrinsicCheck<WallClockCheckerStation>({
      name: `${id}.budget-nonnegative`,
      group: 'wall-clock-checker',
      predicate: s => s.budgetMs >= 0,
      expected: 'budgetMs >= 0',
      observedFn: s => String(s.budgetMs),
    }));
  }

  override assertPreconditions(): void {
    Preconditions.nonNegative('WallClockCheckerStation', 'budgetMs', this.budgetMs);
    Preconditions.integerInRange('WallClockCheckerStation', 'checkEveryTicks', this.checkEveryTicks, 1, Number.MAX_SAFE_INTEGER);
  }

  override hasWork(): boolean {
    return false;
  }

  runTimeStep(): void {
    if (this.expiredFlag) return;
    if (this.tick % this.checkEveryTicks === 0) {
      this.checks++;
      this.elapsed = Math.max(0, this.now() - this.startedAt);
      if (this.elapsed >= this.budgetMs) {
        this.expiredFlag = true;
        this.emit(new StopSignalToken({checkerId: this.id, elapsedMs: this.elapsed, budgetMs: this.budgetMs, tick: this.tick}), STOP_CHANNEL);
      }
    }
    this.tick++;
  }

  expired(): boolean {
    return this.expiredFlag;
  }

  elapsedMs(): number {
    return this.expiredFlag ? this.elapsed : Math.max(0, this.now() - this.startedAt);
  }

  numChecks(): number {
    return this.checks;
  }
}

export class SolutionSinkStation extends DESStation {
  readonly trace: SolverProgressPayload[] = [];
  readonly stops: StopSignalToken['payload'][] = [];

  constructor(id = 'solution-sink') {
    super(id);
  }

  override hasWork(): boolean {
    return false;
  }

  runTimeStep(): void {
    for (const token of this.drain<Token>(SOLUTION_CHANNEL)) {
      const t = token as SolverSolutionToken;
      if (t.kind !== 'solver-solution') throw new Error(`${this.id}: expected solver solution token`);
      this.trace.push(t.payload);
    }
    for (const token of this.drain<Token>(STOP_CHANNEL)) {
      const t = token as StopSignalToken;
      if (t.kind !== 'stop-signal') throw new Error(`${this.id}: expected stop signal token`);
      this.stops.push(t.payload);
    }
  }

  best(): SolverProgressPayload | undefined {
    let best: SolverProgressPayload | undefined;
    for (const row of this.trace) {
      if (!best || row.objective <= best.objective) best = row;
    }
    return best;
  }
}

export class ShortestPathSolverStation extends DESStation implements SnapshotProvider {
  private readonly graph: Graph;
  private readonly algorithm: 'bellman-ford' | 'dijkstra';
  private readonly source: number;
  private distance: number[];
  private predecessor: number[];
  private dirty: boolean[];
  private settled: boolean[];
  private pending: Array<{nodeId: number; distance: number}> = [];
  private iter = 0;
  private done = false;
  private waves = 0;
  private negativeCycle = false;

  constructor(id: string, params: ShortestPathSolverParams) {
    super(id);
    this.graph = graphFromParams(params);
    this.algorithm = params.algorithm;
    this.source = params.source;
    this.distance = new Array(this.graph.numNodes).fill(Infinity);
    this.predecessor = new Array(this.graph.numNodes).fill(-1);
    this.dirty = new Array(this.graph.numNodes).fill(false);
    this.settled = new Array(this.graph.numNodes).fill(false);
    this.distance[this.source] = 0;
    this.dirty[this.source] = true;
    this.pending.push({nodeId: this.source, distance: 0});
    this.addValidator(intrinsicCheck<ShortestPathSolverStation>({
      name: 'shortest-path-source-distance-zero',
      group: 'internal-solver-shortest-path',
      predicate: s => s.distance[s.source] === 0,
      expected: 'distance[source] = 0',
      observedFn: s => String(s.distance[s.source]),
    }));
  }

  override assertPreconditions(): void {
    validateGraph(this.graph);
    Preconditions.integerInRange('ShortestPathSolverStation', 'source', this.source, 0, this.graph.numNodes - 1);
    if (this.algorithm === 'dijkstra') {
      for (let u = 0; u < this.graph.numNodes; u++) {
        for (const e of this.graph.edges[u]) Preconditions.nonNegative('ShortestPathSolverStation', `edge ${u}->${e.to}`, e.weight);
      }
    }
  }

  override hasWork(): boolean {
    return !this.done;
  }

  runTimeStep(): void {
    if (this.done) return;
    if (this.algorithm === 'bellman-ford') this.stepBellmanFord();
    else this.stepDijkstra();
    this.emitSnapshot(this.done);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const reachable = this.distance.filter(Number.isFinite).length;
    const unresolvedPenalty = (this.graph.numNodes - reachable) * 1e12;
    const objective = unresolvedPenalty + this.distance.reduce((s, d) => s + (Number.isFinite(d) ? d : 0), 0);
    return {
      solverId: this.id,
      solverKind: 'shortest-path',
      tick: this.iter,
      iteration: this.iter,
      objective,
      bestState: {
        distance: this.distance.slice(),
        predecessor: this.predecessor.slice(),
        algorithm: this.algorithm,
        hasNegativeCycleFromSource: this.negativeCycle,
      },
      feasible: !this.negativeCycle,
      done,
      metadata: {reachable, wavesEmitted: this.waves},
    };
  }

  private stepBellmanFord(): void {
    this.iter++;
    const newDirty = new Array(this.graph.numNodes).fill(false);
    let anyChange = false;
    for (let u = 0; u < this.graph.numNodes; u++) {
      if (!this.dirty[u]) continue;
      const du = this.distance[u];
      for (const edge of this.graph.edges[u]) {
        this.waves++;
        const cand = du + edge.weight;
        if (cand < this.distance[edge.to] - 1e-12) {
          this.distance[edge.to] = cand;
          this.predecessor[edge.to] = u;
          newDirty[edge.to] = true;
          anyChange = true;
        }
      }
    }
    this.dirty = newDirty;
    if (!anyChange) this.done = true;
    if (this.iter >= this.graph.numNodes && anyChange) {
      this.negativeCycle = true;
      this.done = true;
    }
  }

  private stepDijkstra(): void {
    while (this.pending.length > 0) {
      this.pending.sort((a, b) => a.distance - b.distance);
      const top = this.pending.shift()!;
      if (this.settled[top.nodeId]) continue;
      this.settled[top.nodeId] = true;
      this.iter++;
      for (const edge of this.graph.edges[top.nodeId]) {
        this.waves++;
        const cand = top.distance + edge.weight;
        if (cand < this.distance[edge.to] - 1e-12) {
          this.distance[edge.to] = cand;
          this.predecessor[edge.to] = top.nodeId;
          this.pending.push({nodeId: edge.to, distance: cand});
        }
      }
      return;
    }
    this.done = true;
  }

  private emitSnapshot(done: boolean): void {
    this.emit(new SolverSolutionToken(this.snapshot(done)), SOLUTION_CHANNEL);
  }
}

export class KnapsackDPStation extends DESStation implements SnapshotProvider {
  private readonly values: number[];
  private readonly weights: number[];
  private readonly capacity: number;
  private readonly keep: boolean[][];
  private dp: number[];
  private item = 0;
  private done = false;

  constructor(id: string, params: KnapsackParams) {
    super(id);
    this.values = Array.isArray(params.values) ? params.values.slice() : [];
    this.weights = Array.isArray(params.weights) ? params.weights.slice() : [];
    this.capacity = Number.isFinite(params.capacity) ? params.capacity : -1;
    validateKnapsack({values: this.values, weights: this.weights, capacity: this.capacity});
    this.dp = new Array(this.capacity + 1).fill(0);
    this.keep = Array.from({length: this.values.length}, () => new Array(this.capacity + 1).fill(false));
    this.addValidator(intrinsicCheck<KnapsackDPStation>({
      name: 'knapsack-dp-capacity-feasible',
      group: 'internal-solver-knapsack',
      predicate: s => s.solution().weight <= s.capacity,
      expected: 'selected weight <= capacity',
      observedFn: s => `${s.solution().weight}/${s.capacity}`,
    }));
  }

  override assertPreconditions(): void {
    validateKnapsack({values: this.values, weights: this.weights, capacity: this.capacity});
    Preconditions.integerInRange('KnapsackDPStation', 'capacity', this.capacity, 0, 100000);
    Preconditions.check('KnapsackDPStation', 'state space', 'have at most 5,000,000 cells', this.values.length * (this.capacity + 1) <= 5000000, this.values.length * (this.capacity + 1));
  }

  override hasWork(): boolean {
    return !this.done;
  }

  runTimeStep(): void {
    if (this.done) return;
    if (this.item >= this.values.length) {
      this.done = true;
      this.emitSnapshot(true);
      return;
    }
    const i = this.item;
    const next = this.dp.slice();
    for (let c = 0; c <= this.capacity; c++) {
      const w = this.weights[i];
      if (w <= c) {
        const cand = this.dp[c - w] + this.values[i];
        if (cand > next[c]) {
          next[c] = cand;
          this.keep[i][c] = true;
        }
      }
    }
    this.dp = next;
    this.item++;
    if (this.item >= this.values.length) this.done = true;
    this.emitSnapshot(this.done);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const sol = this.solution();
    return {
      solverId: this.id,
      solverKind: 'knapsack-dp',
      tick: this.item,
      iteration: this.item,
      objective: -sol.value,
      bestState: sol,
      feasible: sol.weight <= this.capacity,
      done,
      metadata: {itemsProcessed: this.item, capacity: this.capacity},
    };
  }

  private emitSnapshot(done: boolean): void {
    this.emit(new SolverSolutionToken(this.snapshot(done)), SOLUTION_CHANNEL);
  }

  private solution(): {selected: number[]; value: number; weight: number; capacity: number} {
    const selected = new Array(this.values.length).fill(0);
    let c = this.capacity;
    for (let i = Math.min(this.item, this.values.length) - 1; i >= 0; i--) {
      if (this.keep[i][c]) {
        selected[i] = 1;
        c -= this.weights[i];
      }
    }
    let value = 0;
    let weight = 0;
    for (let i = 0; i < selected.length; i++) {
      value += selected[i] * this.values[i];
      weight += selected[i] * this.weights[i];
    }
    return {selected, value, weight, capacity: this.capacity};
  }
}

export class KnapsackSAStation extends SingleStateOptimizer<number[]> implements SnapshotProvider {
  private readonly values: number[];
  private readonly weights: number[];
  private readonly capacity: number;
  private readonly cooling: CoolingSchedule;
  private readonly maxIterations: number;
  private readonly stallLimit: number;
  private readonly penalty: number;
  private stall = 0;
  private prevBest = Infinity;

  constructor(id: string, params: KnapsackParams) {
    super(id, {rng: mulberry32(params.seed ?? 1)});
    this.values = Array.isArray(params.values) ? params.values.slice() : [];
    this.weights = Array.isArray(params.weights) ? params.weights.slice() : [];
    this.capacity = Number.isFinite(params.capacity) ? params.capacity : -1;
    this.cooling = params.cooling ?? {kind: 'geometric', T0: 50, alpha: 0.995, Tmin: 1e-6};
    this.maxIterations = params.maxIterations ?? 5000;
    this.stallLimit = params.stallLimit ?? 0;
    this.penalty = params.penalty ?? 1e6;
    validateKnapsack({values: this.values, weights: this.weights, capacity: this.capacity});
    this.bootstrap();
    this.prevBest = this.bestCost;
    this.addValidator(intrinsicCheck<KnapsackSAStation>({
      name: 'knapsack-sa-best-is-finite',
      group: 'internal-solver-knapsack',
      predicate: s => Number.isFinite(s.getBestCost()),
      expected: 'finite best cost',
      observedFn: s => String(s.getBestCost()),
    }));
  }

  protected initialState(_rng: () => number): number[] {
    const order = Array.from({length: this.values.length}, (_, i) => i)
      .sort((a, b) => (this.values[b] / this.weights[b]) - (this.values[a] / this.weights[a]));
    const x = new Array(this.values.length).fill(0);
    let w = 0;
    for (const i of order) {
      if (w + this.weights[i] <= this.capacity) {
        x[i] = 1;
        w += this.weights[i];
      }
    }
    return x;
  }

  protected cost(x: number[]): number {
    const s = knapsackScore(this.values, this.weights, this.capacity, x);
    return -s.value + this.penalty * Math.max(0, s.weight - this.capacity);
  }

  protected propose(x: number[], rng: () => number): number[] {
    const j = Math.floor(rng() * x.length);
    const next = x.slice();
    next[j] = 1 - next[j];
    return next;
  }

  protected accept(_current: number[], _candidate: number[], currentCost: number, candidateCost: number, iter: number, rng: () => number): boolean {
    const delta = candidateCost - currentCost;
    if (delta <= 0) return true;
    const T = temperatureAt(this.cooling, iter);
    return T > 0 && rng() < Math.exp(-delta / T);
  }

  protected clone(x: number[]): number[] {
    return x.slice();
  }

  protected shouldStop(iter: number): boolean {
    if (iter >= this.maxIterations) return true;
    if (this.stallLimit > 0 && this.stall >= this.stallLimit) return true;
    return false;
  }

  protected onAccept(_candidate: number[], _delta: number, iter: number): void {
    this.afterStep(iter);
  }

  protected onReject(_candidate: number[], _delta: number, iter: number): void {
    this.afterStep(iter);
  }

  protected onFinish(): void {
    this.emitSnapshot(true);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const score = knapsackScore(this.values, this.weights, this.capacity, this.getBest());
    return {
      solverId: this.id,
      solverKind: 'knapsack-sa',
      tick: this.getIteration(),
      iteration: this.getIteration(),
      objective: this.getBestCost(),
      bestState: score,
      feasible: score.weight <= this.capacity,
      done,
      metadata: {accepted: this.getAcceptedCount(), improvements: this.getImproveCount()},
    };
  }

  private afterStep(iter: number): void {
    if (this.bestCost < this.prevBest) {
      this.prevBest = this.bestCost;
      this.stall = 0;
    } else {
      this.stall++;
    }
    this.emitSnapshot(false);
  }

  private emitSnapshot(done: boolean): void {
    this.emit(new SolverSolutionToken(this.snapshot(done)), SOLUTION_CHANNEL);
  }
}

export class ObservableTSPSAOptimizer extends TSPSAOptimizer implements SnapshotProvider {
  constructor(id: string, private readonly instanceRef: TSPInstance, opts: TSPSAOptions) {
    super(id, instanceRef, opts);
  }

  protected override onAccept(candidate: Tour, delta: number, iter: number): void {
    super.onAccept(candidate, delta, iter);
    this.emitSnapshot(false);
  }

  protected override onReject(candidate: Tour, delta: number, iter: number): void {
    super.onReject(candidate, delta, iter);
    this.emitSnapshot(false);
  }

  protected override onFinish(): void {
    this.emitSnapshot(true);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const best = this.getBest();
    return {
      solverId: this.id,
      solverKind: 'tsp-sa',
      tick: this.getIteration(),
      iteration: this.getIteration(),
      objective: this.getBestCost(),
      bestState: {tour: best.slice(), length: tourLength(this.instanceRef, best)},
      feasible: isPermutation(best, this.instanceRef.n),
      done,
      metadata: {accepted: this.getAcceptedCount(), improvements: this.getImproveCount(), n: this.instanceRef.n},
    };
  }

  private emitSnapshot(done: boolean): void {
    this.emit(new SolverSolutionToken(this.snapshot(done)), SOLUTION_CHANNEL);
  }
}

export class ObservableTSPGAOptimizer extends TSPGAOptimizer implements SnapshotProvider {
  constructor(id: string, private readonly instanceRef: TSPInstance, opts: TSPGAOptions) {
    super(id, instanceRef, opts);
  }

  protected override onGeneration(gen: number): void {
    super.onGeneration(gen);
    this.emitSnapshot(false);
  }

  protected override onFinish(): void {
    this.emitSnapshot(true);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const best = this.getBestTour();
    return {
      solverId: this.id,
      solverKind: 'tsp-ga',
      tick: this.getGeneration(),
      iteration: this.getGeneration(),
      objective: this.getBestLength(),
      bestState: {tour: best.slice(), length: tourLength(this.instanceRef, best)},
      feasible: isPermutation(best, this.instanceRef.n),
      done,
      metadata: {n: this.instanceRef.n, population: this.getPopulation().length},
    };
  }

  private emitSnapshot(done: boolean): void {
    this.emit(new SolverSolutionToken(this.snapshot(done)), SOLUTION_CHANNEL);
  }
}

export class TSPHeldKarpStation extends DESStation implements SnapshotProvider {
  private done = false;
  private iter = 0;
  private best: {tour: Tour; length: number} | undefined;

  constructor(id: string, private readonly instance: TSPInstance) {
    super(id);
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange('TSPHeldKarpStation', 'n', this.instance.n, 3, 16);
  }

  override hasWork(): boolean {
    return !this.done;
  }

  runTimeStep(): void {
    if (this.done) return;
    this.best = heldKarpExact(this.instance);
    this.done = true;
    this.iter++;
    this.emit(new SolverSolutionToken(this.snapshot(true)), SOLUTION_CHANNEL);
  }

  snapshot(done: boolean): SolverProgressPayload {
    const best = this.best ?? {tour: [], length: Infinity};
    return {
      solverId: this.id,
      solverKind: 'tsp-held-karp',
      tick: this.iter,
      iteration: this.iter,
      objective: best.length,
      bestState: best,
      feasible: best.tour.length === 0 || isPermutation(best.tour, this.instance.n),
      done,
      metadata: {n: this.instance.n, exact: true},
    };
  }
}

export function runInternalSolverNetwork(params: InternalSolverRunParams): InternalSolverRunResult {
  const budgetMs = params.timeLimitMs ?? 180000;
  const checker = new WallClockCheckerStation('wall-clock-checker', budgetMs, params.checkEveryTicks ?? 1);
  const sink = new SolutionSinkStation();
  const solver = buildSolverStation(params);
  solver.pipe(sink, SOLUTION_CHANNEL, SOLUTION_CHANNEL);
  checker.pipe(sink, STOP_CHANNEL, STOP_CHANNEL);
  const maxTicks = params.maxTicks ?? defaultMaxTicks(params);
  const stations = [solver, checker, sink];
  const summary = runIterativeDES(stations, {
    shuffle: false,
    maxTicks,
    stopWhen: () => checker.expired(),
  });
  const fallback = solver.snapshot(summary.reason === 'done');
  const best = sink.best() ?? fallback;
  const status: InternalSolverRunResult['status'] = checker.expired()
    ? 'time-limit'
    : summary.reason === 'maxticks' ? 'tick-limit' : 'complete';
  return {
    kind: params.kind,
    status,
    runSummary: summary,
    best,
    trace: sink.trace.slice(),
    stopSignals: sink.stops.slice(),
    wallClock: {
      budgetMs,
      elapsedMs: checker.elapsedMs(),
      checks: checker.numChecks(),
      expired: checker.expired(),
    },
    network: describeNetwork(params.kind, solver.id),
    validation: summary.validation ?? [],
  };
}

function buildSolverStation(params: InternalSolverRunParams): DESStation & SnapshotProvider {
  switch (params.kind) {
    case 'shortest-path':
      return new ShortestPathSolverStation('shortest-path-solver', required(params.shortestPath, 'shortestPath'));
    case 'knapsack-dp':
      return new KnapsackDPStation('knapsack-dp-solver', required(params.knapsack, 'knapsack'));
    case 'knapsack-sa':
      return new KnapsackSAStation('knapsack-sa-solver', required(params.knapsack, 'knapsack'));
    case 'tsp-sa': {
      const tsp = required(params.tsp, 'tsp');
      const inst = tspInstance(tsp);
      return new ObservableTSPSAOptimizer('tsp-sa-solver', inst, {
        cooling: tsp.sa?.cooling ?? {kind: 'geometric', T0: 100, alpha: 0.995, Tmin: 1e-6},
        maxIterations: tsp.sa?.maxIterations ?? 5000,
        seed: tsp.sa?.seed ?? tsp.seed ?? 1,
        init: tsp.sa?.init ?? 'nearest-neighbor',
        moves: tsp.sa?.moves ?? 'mixed',
        penaltyPerViolation: tsp.sa?.penaltyPerViolation,
        traceStride: tsp.sa?.traceStride,
        stallLimit: tsp.sa?.stallLimit,
      });
    }
    case 'tsp-ga': {
      const tsp = required(params.tsp, 'tsp');
      const inst = tspInstance(tsp);
      return new ObservableTSPGAOptimizer('tsp-ga-solver', inst, {
        popSize: tsp.ga?.popSize ?? 60,
        numGenerations: tsp.ga?.numGenerations ?? 200,
        tournamentSize: tsp.ga?.tournamentSize,
        crossoverProb: tsp.ga?.crossoverProb,
        mutationProb: tsp.ga?.mutationProb,
        elitism: tsp.ga?.elitism,
        seed: tsp.ga?.seed ?? tsp.seed ?? 1,
        init: tsp.ga?.init ?? 'nearest-neighbor',
        penaltyPerViolation: tsp.ga?.penaltyPerViolation,
      });
    }
    case 'tsp-held-karp':
      return new TSPHeldKarpStation('tsp-held-karp-solver', tspInstance(required(params.tsp, 'tsp')));
  }
}

function describeNetwork(kind: InternalSolverKind, solverId: string): SolverNetworkDescription {
  return {
    stationaryEntities: [
      {id: 'initial-source', kind: `${kind}-initial-source`, role: 'source'},
      {id: solverId, kind, role: 'solver'},
      {id: 'wall-clock-checker', kind: 'wall-clock-checker', role: 'checker'},
      {id: 'solution-sink', kind: 'solution-sink', role: 'sink'},
    ],
    movingEntities: [
      {id: 'SolverSolutionToken', kind: 'incumbent-solution', tokenType: 'SolverSolutionToken'},
      {id: 'StopSignalToken', kind: 'stop-signal', tokenType: 'StopSignalToken'},
    ],
    edges: [
      {from: 'initial-source', to: solverId, movingEntity: 'initial-conditions', channel: 'constructor'},
      {from: solverId, to: 'solution-sink', movingEntity: 'SolverSolutionToken', channel: SOLUTION_CHANNEL},
      {from: 'wall-clock-checker', to: 'solution-sink', movingEntity: 'StopSignalToken', channel: STOP_CHANNEL},
    ],
  };
}

function graphFromParams(params: ShortestPathSolverParams): Graph {
  if (params.builtin === 'small-chain') return buildSmallChainGraph();
  if (params.randomGraph) {
    return buildRandomGraph(params.randomGraph.numNodes, params.randomGraph.edgeProb, params.randomGraph.wMin, params.randomGraph.wMax, params.randomGraph.seed);
  }
  if (params.graph) return params.graph;
  throw new Error('shortest-path solver requires builtin, graph, or randomGraph');
}

function tspInstance(params: TSPSolverParams): TSPInstance {
  if (params.builtin === 'pentagon') return buildPentagonTSP(params.n ?? 5);
  if (params.builtin === 'random') return buildRandomTSP(params.n ?? 20, params.seed ?? 1, {precedence: params.precedence});
  if (params.coordinates && params.distance) {
    return {n: params.coordinates.length, coordinates: params.coordinates, distance: params.distance, precedence: params.precedence};
  }
  return buildPentagonTSP(params.n ?? 5);
}

function validateGraph(graph: Graph): void {
  Preconditions.integerInRange('ShortestPathSolverStation', 'numNodes', graph.numNodes, 1, 100000);
  Preconditions.check('ShortestPathSolverStation', 'edges.length', 'equal numNodes', graph.edges.length === graph.numNodes, graph.edges.length);
  for (let u = 0; u < graph.numNodes; u++) {
    for (const edge of graph.edges[u]) {
      Preconditions.integerInRange('ShortestPathSolverStation', `edge ${u}.to`, edge.to, 0, graph.numNodes - 1);
      Preconditions.finite('ShortestPathSolverStation', `edge ${u}->${edge.to}.weight`, edge.weight);
    }
  }
}

function validateKnapsack(params: {values: number[]; weights: number[]; capacity: number}): void {
  Preconditions.nonEmpty('KnapsackSolver', 'values', params.values);
  Preconditions.lengthEq('KnapsackSolver', 'weights', params.weights, params.values.length);
  Preconditions.allFinite('KnapsackSolver', 'values', params.values);
  Preconditions.allFinite('KnapsackSolver', 'weights', params.weights);
  Preconditions.integerInRange('KnapsackSolver', 'capacity', params.capacity, 0, Number.MAX_SAFE_INTEGER);
  for (let i = 0; i < params.weights.length; i++) Preconditions.integerInRange('KnapsackSolver', `weights[${i}]`, params.weights[i], 0, Number.MAX_SAFE_INTEGER);
}

function knapsackScore(values: readonly number[], weights: readonly number[], capacity: number, selected: readonly number[]): {selected: number[]; value: number; weight: number; capacity: number} {
  let value = 0;
  let weight = 0;
  for (let i = 0; i < selected.length; i++) {
    value += selected[i] * values[i];
    weight += selected[i] * weights[i];
  }
  return {selected: selected.slice(), value, weight, capacity};
}

function defaultMaxTicks(params: InternalSolverRunParams): number {
  switch (params.kind) {
    case 'shortest-path': return 100000;
    case 'knapsack-dp': return (params.knapsack?.values.length ?? 1000) + 2;
    case 'knapsack-sa': return (params.knapsack?.maxIterations ?? 5000) + 2;
    case 'tsp-sa': return (params.tsp?.sa?.maxIterations ?? 5000) + 2;
    case 'tsp-ga': return (params.tsp?.ga?.numGenerations ?? 200) + 2;
    case 'tsp-held-karp': return 2;
  }
}

function required<T>(value: T | undefined, name: string): T {
  if (value === undefined) throw new Error(`internal-solver-network: ${name} parameters required`);
  return value;
}
