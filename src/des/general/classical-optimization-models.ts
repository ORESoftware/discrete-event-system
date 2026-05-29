'use strict';

// =============================================================================
// general/classical-optimization-models.ts
//
// Additional classic optimization routines as explicit DES station graphs:
//   - qp-projected-gradient
//   - qp-coordinate-descent
//   - hungarian-assignment
//   - auction-assignment
//   - vrp-savings
//   - vrp-nearest-neighbor
//   - job-shop-dispatch
//   - flow-shop-neh
// =============================================================================

import {
  ChannelName,
  DESStation,
  LatestTokenSinkStation,
  SingleTokenSourceStation,
  StationGraphSummary,
  Token,
  channelEdge,
  cloneMatrix,
  dot,
  emptyStationGraph,
  nonEmptyArray,
  norm2,
  Preconditions,
  runIterativeDES,
  runStateLoopPipeline,
  stateLoopTopology,
  stationGraph,
  zeros,
} from './des-base';

const CH_QP_STATE: ChannelName = 'qp-state';
const CH_QP_RESULT: ChannelName = 'qp-result';

export interface QPProjectedGradientParams {
  Q?: number[][];
  c?: number[];
  lower?: number[];
  upper?: number[];
  x0?: number[];
  stepSize?: number;
  maxIter?: number;
  tol?: number;
}

export interface QPProjectedGradientResult {
  x: number[];
  objective: number;
  iterations: number;
  gradientNorm: number;
  trace: Array<{iter: number; objective: number; gradientNorm: number; x: number[]}>;
  topology: StationGraphSummary;
}

class QPStateToken implements Token {
  constructor(readonly iter: number, readonly x: number[]) {}
}

class QPResultToken implements Token {
  constructor(readonly result: QPProjectedGradientResult) {}
}

class QPProjectedGradientStation extends DESStation {
  static readonly CH_STATE: ChannelName = CH_QP_STATE;
  static readonly CH_RESULT: ChannelName = CH_QP_RESULT;
  readonly trace: Array<{iter: number; objective: number; gradientNorm: number; x: number[]}> = [];

  constructor(
    id: string,
    private readonly Q: number[][],
    private readonly c: number[],
    private readonly lower: number[],
    private readonly upper: number[],
    private readonly stepSize: number,
    private readonly maxIter: number,
    private readonly tol: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(QPProjectedGradientStation.CH_STATE) > 0; }

  runTimeStep(): void {
    const states = this.drain<QPStateToken>(QPProjectedGradientStation.CH_STATE);
    for (const state of states) {
      const gradient = qpGradient(this.Q, this.c, state.x);
      const gradientNorm = norm2(gradient);
      const objective = qpObjective(this.Q, this.c, state.x);
      this.trace.push({iter: state.iter, objective, gradientNorm, x: state.x.slice()});
      if (state.iter >= this.maxIter || gradientNorm <= this.tol) {
        this.emit(new QPResultToken({
          x: state.x.slice(),
          objective,
          iterations: state.iter,
          gradientNorm,
          trace: this.trace.slice(),
          topology: emptyStationGraph(),
        }), QPProjectedGradientStation.CH_RESULT);
        continue;
      }
      const next = state.x.map((v, i) => Math.min(this.upper[i], Math.max(this.lower[i], v - this.stepSize * gradient[i])));
      this.emit(new QPStateToken(state.iter + 1, next), QPProjectedGradientStation.CH_STATE);
    }
  }
}

class QPCoordinateDescentStation extends DESStation {
  static readonly CH_STATE: ChannelName = CH_QP_STATE;
  static readonly CH_RESULT: ChannelName = CH_QP_RESULT;
  readonly trace: Array<{iter: number; objective: number; gradientNorm: number; x: number[]}> = [];

  constructor(
    id: string,
    private readonly Q: number[][],
    private readonly c: number[],
    private readonly lower: number[],
    private readonly upper: number[],
    private readonly maxIter: number,
    private readonly tol: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(QPCoordinateDescentStation.CH_STATE) > 0; }

  runTimeStep(): void {
    const states = this.drain<QPStateToken>(QPCoordinateDescentStation.CH_STATE);
    for (const state of states) {
      const gradient = qpGradient(this.Q, this.c, state.x);
      const gradientNorm = norm2(gradient);
      const objective = qpObjective(this.Q, this.c, state.x);
      this.trace.push({iter: state.iter, objective, gradientNorm, x: state.x.slice()});
      if (state.iter >= this.maxIter || gradientNorm <= this.tol) {
        this.emit(new QPResultToken({
          x: state.x.slice(),
          objective,
          iterations: state.iter,
          gradientNorm,
          trace: this.trace.slice(),
          topology: emptyStationGraph(),
        }), QPCoordinateDescentStation.CH_RESULT);
        continue;
      }
      const next = state.x.slice();
      for (let i = 0; i < next.length; i++) {
        const diag = this.Q[i][i];
        if (Math.abs(diag) <= 1e-12) continue;
        const g = dot(this.Q[i], next) + this.c[i];
        next[i] = Math.min(this.upper[i], Math.max(this.lower[i], next[i] - g / diag));
      }
      this.emit(new QPStateToken(state.iter + 1, next), QPCoordinateDescentStation.CH_STATE);
    }
  }
}

function validateQPInitialState(
  model: string,
  token: QPStateToken,
  n: number,
  lower: readonly number[],
  upper: readonly number[],
): void {
  Preconditions.integerInRange(model, 'iter', token.iter, 0, 1e9);
  Preconditions.lengthEq(model, 'x0', token.x, n);
  Preconditions.allFinite(model, 'x0', token.x);
  Preconditions.lengthEq(model, 'lower', lower, n);
  Preconditions.lengthEq(model, 'upper', upper, n);
  for (let i = 0; i < n; i++) {
    Preconditions.check(model, `lower[${i}] <= x0[${i}] <= upper[${i}]`, 'hold',
      lower[i] <= token.x[i] && token.x[i] <= upper[i], [lower[i], token.x[i], upper[i]]);
  }
}

export function runQPProjectedGradient(params: QPProjectedGradientParams): QPProjectedGradientResult {
  const Q = nonEmptyArray(params.Q, [[4, 1], [1, 2]]);
  const c = nonEmptyArray(params.c, [-8, -6]);
  const n = c.length;
  const lower = nonEmptyArray(params.lower, zeros(n));
  const upper = nonEmptyArray(params.upper, Array.from({length: n}, () => 10));
  const x0 = nonEmptyArray(params.x0, zeros(n));
  const source = new SingleTokenSourceStation<QPStateToken>(
    'qp-state-source',
    CH_QP_STATE,
    () => new QPStateToken(0, x0.slice()),
    token => validateQPInitialState('qp-projected-gradient-source', token, n, lower, upper),
  );
  const update = new QPProjectedGradientStation('projected-gradient-update', Q, c, lower, upper, params.stepSize ?? 0.12, params.maxIter ?? 200, params.tol ?? 1e-8);
  const sink = new LatestTokenSinkStation<QPResultToken>('qp-result-sink', CH_QP_RESULT);
  runStateLoopPipeline(source, update, sink, CH_QP_STATE, CH_QP_RESULT, {maxTicks: (params.maxIter ?? 200) + 10});
  if (!sink.latest) throw new Error('qp-projected-gradient did not produce a result');
  sink.latest.result.topology = stateLoopTopology(source, update, sink, CH_QP_STATE, CH_QP_RESULT, ['QPStateToken', 'QPResultToken']);
  return sink.latest.result;
}

export function runQPCoordinateDescent(params: QPProjectedGradientParams): QPProjectedGradientResult {
  const Q = nonEmptyArray(params.Q, [[4, 1], [1, 2]]);
  const c = nonEmptyArray(params.c, [-8, -6]);
  const n = c.length;
  const lower = nonEmptyArray(params.lower, zeros(n));
  const upper = nonEmptyArray(params.upper, Array.from({length: n}, () => 10));
  const x0 = nonEmptyArray(params.x0, zeros(n));
  const source = new SingleTokenSourceStation<QPStateToken>(
    'qp-coordinate-state-source',
    CH_QP_STATE,
    () => new QPStateToken(0, x0.slice()),
    token => validateQPInitialState('qp-coordinate-descent-source', token, n, lower, upper),
  );
  const update = new QPCoordinateDescentStation('coordinate-descent-update', Q, c, lower, upper, params.maxIter ?? 100, params.tol ?? 1e-8);
  const sink = new LatestTokenSinkStation<QPResultToken>('qp-result-sink', CH_QP_RESULT);
  runStateLoopPipeline(source, update, sink, CH_QP_STATE, CH_QP_RESULT, {maxTicks: (params.maxIter ?? 100) + 10});
  if (!sink.latest) throw new Error('qp-coordinate-descent did not produce a result');
  sink.latest.result.topology = stateLoopTopology(source, update, sink, CH_QP_STATE, CH_QP_RESULT, ['QPStateToken', 'QPResultToken']);
  return sink.latest.result;
}

export interface AssignmentParams {
  cost?: number[][];
}

export interface AssignmentResult {
  assignment: number[];
  objective: number;
  rowReductions: number[];
  colReductions: number[];
  topology: StationGraphSummary;
}

class AssignmentMatrixToken implements Token {
  constructor(
    readonly original: number[][],
    readonly reduced: number[][],
    readonly rowReductions: number[] = [],
    readonly colReductions: number[] = [],
  ) {}
}

class AssignmentResultToken implements Token {
  constructor(readonly result: AssignmentResult) {}
}

class AssignmentAuctionStateToken implements Token {
  constructor(
    readonly original: number[][],
    readonly prices: number[],
    readonly assignment: number[],
    readonly owner: number[],
    readonly iter: number,
  ) {}
}

class AssignmentSourceStation extends DESStation {
  static readonly CH_MATRIX: ChannelName = 'assignment-matrix';
  private emitted = false;
  constructor(id: string, private readonly cost: number[][]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(new AssignmentMatrixToken(cloneMatrix(this.cost), cloneMatrix(this.cost)), AssignmentSourceStation.CH_MATRIX);
    this.emitted = true;
  }
}

class RowReductionStation extends DESStation {
  static readonly CH_MATRIX: ChannelName = AssignmentSourceStation.CH_MATRIX;
  static readonly CH_REDUCED: ChannelName = 'row-reduced';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(RowReductionStation.CH_MATRIX) > 0; }
  runTimeStep(): void {
    for (const token of this.drain<AssignmentMatrixToken>(RowReductionStation.CH_MATRIX)) {
      const reduced = cloneMatrix(token.reduced);
      const rows = reduced.map(row => Math.min(...row));
      for (let i = 0; i < reduced.length; i++) for (let j = 0; j < reduced[i].length; j++) reduced[i][j] -= rows[i];
      this.emit(new AssignmentMatrixToken(token.original, reduced, rows, token.colReductions), RowReductionStation.CH_REDUCED);
    }
  }
}

class ColumnReductionStation extends DESStation {
  static readonly CH_REDUCED: ChannelName = RowReductionStation.CH_REDUCED;
  static readonly CH_REDUCED2: ChannelName = 'column-reduced';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(ColumnReductionStation.CH_REDUCED) > 0; }
  runTimeStep(): void {
    for (const token of this.drain<AssignmentMatrixToken>(ColumnReductionStation.CH_REDUCED)) {
      const reduced = cloneMatrix(token.reduced);
      const cols = zeros(reduced[0].length);
      for (let j = 0; j < cols.length; j++) cols[j] = Math.min(...reduced.map(row => row[j]));
      for (let i = 0; i < reduced.length; i++) for (let j = 0; j < reduced[i].length; j++) reduced[i][j] -= cols[j];
      this.emit(new AssignmentMatrixToken(token.original, reduced, token.rowReductions, cols), ColumnReductionStation.CH_REDUCED2);
    }
  }
}

class AssignmentSolverStation extends DESStation {
  static readonly CH_REDUCED: ChannelName = ColumnReductionStation.CH_REDUCED2;
  static readonly CH_RESULT: ChannelName = 'assignment-result';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(AssignmentSolverStation.CH_REDUCED) > 0; }
  runTimeStep(): void {
    for (const token of this.drain<AssignmentMatrixToken>(AssignmentSolverStation.CH_REDUCED)) {
      const solved = solveAssignmentDP(token.original);
      this.emit(new AssignmentResultToken({
        assignment: solved.assignment,
        objective: solved.objective,
        rowReductions: token.rowReductions,
        colReductions: token.colReductions,
        topology: emptyStationGraph(),
      }), AssignmentSolverStation.CH_RESULT);
    }
  }
}

class AssignmentSinkStation extends DESStation {
  static readonly CH_RESULT: ChannelName = AssignmentSolverStation.CH_RESULT;
  result: AssignmentResult | undefined;
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(AssignmentSinkStation.CH_RESULT) > 0; }
  runTimeStep(): void {
    const results = this.drain<AssignmentResultToken>(AssignmentSinkStation.CH_RESULT);
    if (results.length > 0) this.result = results[results.length - 1].result;
  }
}

class AssignmentAuctionSourceStation extends DESStation {
  static readonly CH_STATE: ChannelName = 'assignment-auction-state';
  private emitted = false;
  constructor(id: string, private readonly cost: number[][]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    const n = this.cost.length;
    this.emit(
      new AssignmentAuctionStateToken(cloneMatrix(this.cost), zeros(n), Array.from({length: n}, () => -1), Array.from({length: n}, () => -1), 0),
      AssignmentAuctionSourceStation.CH_STATE,
    );
    this.emitted = true;
  }
}

class AuctionAssignmentStation extends DESStation {
  static readonly CH_STATE: ChannelName = AssignmentAuctionSourceStation.CH_STATE;
  static readonly CH_RESULT: ChannelName = AssignmentSolverStation.CH_RESULT;

  constructor(id: string, private readonly epsilon: number, private readonly maxIter: number) { super(id); }

  override hasWork(): boolean { return this.inboxSize(AuctionAssignmentStation.CH_STATE) > 0; }

  runTimeStep(): void {
    for (const state of this.drain<AssignmentAuctionStateToken>(AuctionAssignmentStation.CH_STATE)) {
      const unassigned = state.assignment.findIndex(job => job < 0);
      if (unassigned < 0 || state.iter >= this.maxIter) {
        const objective = state.assignment.reduce((sum, job, worker) => sum + state.original[worker][job], 0);
        this.emit(new AssignmentResultToken({
          assignment: state.assignment.slice(),
          objective,
          rowReductions: [],
          colReductions: state.prices.slice(),
          topology: emptyStationGraph(),
        }), AuctionAssignmentStation.CH_RESULT);
        continue;
      }

      const nets = state.original[unassigned].map((cost, job) => ({job, value: -cost - state.prices[job]}));
      nets.sort((a, b) => b.value - a.value);
      const best = nets[0];
      const secondValue = nets[1]?.value ?? (best.value - this.epsilon);
      const prices = state.prices.slice();
      const assignment = state.assignment.slice();
      const owner = state.owner.slice();
      const previousOwner = owner[best.job];
      if (previousOwner >= 0) assignment[previousOwner] = -1;
      owner[best.job] = unassigned;
      assignment[unassigned] = best.job;
      prices[best.job] += best.value - secondValue + this.epsilon;
      this.emit(new AssignmentAuctionStateToken(state.original, prices, assignment, owner, state.iter + 1), AuctionAssignmentStation.CH_STATE);
    }
  }
}

export function runHungarianAssignment(params: AssignmentParams): AssignmentResult {
  const cost = cloneMatrix(nonEmptyArray(params.cost, [[9, 2, 7], [6, 4, 3], [5, 8, 1]]));
  const source = new AssignmentSourceStation('assignment-source', cost);
  const row = new RowReductionStation('row-reduction');
  const col = new ColumnReductionStation('column-reduction');
  const solver = new AssignmentSolverStation('assignment-builder');
  const sink = new AssignmentSinkStation('assignment-sink');
  source.pipe(row, AssignmentSourceStation.CH_MATRIX, RowReductionStation.CH_MATRIX);
  row.pipe(col, RowReductionStation.CH_REDUCED, ColumnReductionStation.CH_REDUCED);
  col.pipe(solver, ColumnReductionStation.CH_REDUCED2, AssignmentSolverStation.CH_REDUCED);
  solver.pipe(sink, AssignmentSolverStation.CH_RESULT, AssignmentSinkStation.CH_RESULT);
  runIterativeDES([source, row, col, solver, sink], {shuffle: false});
  if (!sink.result) throw new Error('hungarian-assignment did not produce a result');
  sink.result.topology = stationGraph([source, row, col, solver, sink], ['AssignmentMatrixToken', 'AssignmentResultToken'], [
    channelEdge(source, AssignmentSourceStation.CH_MATRIX, row, RowReductionStation.CH_MATRIX),
    channelEdge(row, RowReductionStation.CH_REDUCED, col, ColumnReductionStation.CH_REDUCED),
    channelEdge(col, ColumnReductionStation.CH_REDUCED2, solver, AssignmentSolverStation.CH_REDUCED),
    channelEdge(solver, AssignmentSolverStation.CH_RESULT, sink, AssignmentSinkStation.CH_RESULT),
  ]);
  return sink.result;
}

export function runAuctionAssignment(params: AssignmentParams & {epsilon?: number; maxIter?: number}): AssignmentResult {
  const cost = cloneMatrix(nonEmptyArray(params.cost, [[9, 2, 7], [6, 4, 3], [5, 8, 1]]));
  const source = new AssignmentAuctionSourceStation('auction-assignment-source', cost);
  const auction = new AuctionAssignmentStation('auction-bid-update', params.epsilon ?? 0.01, params.maxIter ?? Math.max(20, cost.length * cost.length * 20));
  const sink = new AssignmentSinkStation('assignment-sink');
  source.pipe(auction, AssignmentAuctionSourceStation.CH_STATE, AuctionAssignmentStation.CH_STATE);
  auction.pipe(auction, AuctionAssignmentStation.CH_STATE, AuctionAssignmentStation.CH_STATE);
  auction.pipe(sink, AuctionAssignmentStation.CH_RESULT, AssignmentSinkStation.CH_RESULT);
  runIterativeDES([source, auction, sink], {shuffle: false, maxTicks: (params.maxIter ?? Math.max(20, cost.length * cost.length * 20)) + 10});
  if (!sink.result) throw new Error('auction-assignment did not produce a result');
  sink.result.topology = stateLoopTopology(source, auction, sink, AssignmentAuctionSourceStation.CH_STATE, AuctionAssignmentStation.CH_RESULT, ['AssignmentAuctionStateToken', 'AssignmentResultToken']);
  return sink.result;
}

export interface VRPCustomer {
  id: string;
  x: number;
  y: number;
  demand: number;
}

export interface VRPSavingsParams {
  depot?: {x: number; y: number};
  customers?: VRPCustomer[];
  vehicleCapacity?: number;
}

export interface VRPRoute {
  customers: string[];
  load: number;
  distance: number;
}

export interface VRPSavingsResult {
  routes: VRPRoute[];
  totalDistance: number;
  savingsConsidered: number;
  topology: StationGraphSummary;
}

class VRPProblemToken implements Token {
  constructor(readonly depot: {x: number; y: number}, readonly customers: readonly VRPCustomer[], readonly capacity: number) {}
}

class VRPSavingsToken implements Token {
  constructor(readonly problem: VRPProblemToken, readonly savings: Array<{i: string; j: string; saving: number}>) {}
}

class VRPResultToken implements Token {
  constructor(readonly result: VRPSavingsResult) {}
}

class VRPSourceStation extends DESStation {
  static readonly CH_PROBLEM: ChannelName = 'vrp-problem';
  private emitted = false;
  constructor(id: string, private readonly problem: VRPProblemToken) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(this.problem, VRPSourceStation.CH_PROBLEM);
    this.emitted = true;
  }
}

class SavingsStation extends DESStation {
  static readonly CH_PROBLEM: ChannelName = VRPSourceStation.CH_PROBLEM;
  static readonly CH_SAVINGS: ChannelName = 'vrp-savings';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(SavingsStation.CH_PROBLEM) > 0; }
  runTimeStep(): void {
    for (const problem of this.drain<VRPProblemToken>(SavingsStation.CH_PROBLEM)) {
      const savings: Array<{i: string; j: string; saving: number}> = [];
      for (let a = 0; a < problem.customers.length; a++) {
        for (let b = a + 1; b < problem.customers.length; b++) {
          const i = problem.customers[a], j = problem.customers[b];
          savings.push({
            i: i.id,
            j: j.id,
            saving: dist(problem.depot, i) + dist(problem.depot, j) - dist(i, j),
          });
        }
      }
      savings.sort((a, b) => b.saving - a.saving);
      this.emit(new VRPSavingsToken(problem, savings), SavingsStation.CH_SAVINGS);
    }
  }
}

class RouteMergeStation extends DESStation {
  static readonly CH_SAVINGS: ChannelName = SavingsStation.CH_SAVINGS;
  static readonly CH_RESULT: ChannelName = 'vrp-result';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(RouteMergeStation.CH_SAVINGS) > 0; }
  runTimeStep(): void {
    for (const token of this.drain<VRPSavingsToken>(RouteMergeStation.CH_SAVINGS)) {
      const byId = new Map(token.problem.customers.map(c => [c.id, c]));
      let routes = token.problem.customers.map(c => [c.id]);
      for (const s of token.savings) {
        const ri = routes.find(r => r.includes(s.i));
        const rj = routes.find(r => r.includes(s.j));
        if (!ri || !rj || ri === rj) continue;
        const iAtEnd = ri[ri.length - 1] === s.i;
        const jAtStart = rj[0] === s.j;
        const jAtEnd = rj[rj.length - 1] === s.j;
        const iAtStart = ri[0] === s.i;
        let merged: string[] | undefined;
        if (iAtEnd && jAtStart) merged = [...ri, ...rj];
        else if (jAtEnd && iAtStart) merged = [...rj, ...ri];
        if (!merged) continue;
        const load = merged.reduce((sum, id) => sum + (byId.get(id)?.demand ?? 0), 0);
        if (load > token.problem.capacity) continue;
        routes = routes.filter(r => r !== ri && r !== rj);
        routes.push(merged);
      }
      const resultRoutes = routes.map(route => {
        const customers = route.map(id => byId.get(id)!);
        return {
          customers: route,
          load: customers.reduce((sum, c) => sum + c.demand, 0),
          distance: routeDistance(token.problem.depot, customers),
        };
      });
      this.emit(new VRPResultToken({
        routes: resultRoutes,
        totalDistance: resultRoutes.reduce((sum, r) => sum + r.distance, 0),
        savingsConsidered: token.savings.length,
        topology: emptyStationGraph(),
      }), RouteMergeStation.CH_RESULT);
    }
  }
}

class VRPSinkStation extends DESStation {
  static readonly CH_RESULT: ChannelName = RouteMergeStation.CH_RESULT;
  result: VRPSavingsResult | undefined;
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(VRPSinkStation.CH_RESULT) > 0; }
  runTimeStep(): void {
    const results = this.drain<VRPResultToken>(VRPSinkStation.CH_RESULT);
    if (results.length > 0) this.result = results[results.length - 1].result;
  }
}

class NearestNeighborRouteStation extends DESStation {
  static readonly CH_PROBLEM: ChannelName = VRPSourceStation.CH_PROBLEM;
  static readonly CH_RESULT: ChannelName = RouteMergeStation.CH_RESULT;
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(NearestNeighborRouteStation.CH_PROBLEM) > 0; }
  runTimeStep(): void {
    for (const problem of this.drain<VRPProblemToken>(NearestNeighborRouteStation.CH_PROBLEM)) {
      const byId = new Map(problem.customers.map(c => [c.id, c]));
      const unserved = new Set(problem.customers.map(c => c.id));
      const routes: VRPRoute[] = [];
      while (unserved.size > 0) {
        const route: string[] = [];
        let load = 0;
        let here: {x: number; y: number} = problem.depot;
        while (true) {
          const feasible = [...unserved]
            .map(id => byId.get(id)!)
            .filter(c => load + c.demand <= problem.capacity)
            .sort((a, b) => dist(here, a) - dist(here, b));
          if (feasible.length === 0) break;
          const next = feasible[0];
          route.push(next.id);
          load += next.demand;
          here = next;
          unserved.delete(next.id);
        }
        const customers = route.map(id => byId.get(id)!);
        routes.push({customers: route, load, distance: routeDistance(problem.depot, customers)});
      }
      this.emit(new VRPResultToken({
        routes,
        totalDistance: routes.reduce((sum, r) => sum + r.distance, 0),
        savingsConsidered: 0,
        topology: emptyStationGraph(),
      }), NearestNeighborRouteStation.CH_RESULT);
    }
  }
}

export function runVRPSavings(params: VRPSavingsParams): VRPSavingsResult {
  const customers = nonEmptyArray(params.customers, defaultCustomers());
  const problem = new VRPProblemToken(params.depot ?? {x: 0, y: 0}, customers, params.vehicleCapacity ?? 5);
  const source = new VRPSourceStation('vrp-source', problem);
  const savings = new SavingsStation('savings-calculator');
  const merge = new RouteMergeStation('route-merge');
  const sink = new VRPSinkStation('vrp-sink');
  source.pipe(savings, VRPSourceStation.CH_PROBLEM, SavingsStation.CH_PROBLEM);
  savings.pipe(merge, SavingsStation.CH_SAVINGS, RouteMergeStation.CH_SAVINGS);
  merge.pipe(sink, RouteMergeStation.CH_RESULT, VRPSinkStation.CH_RESULT);
  runIterativeDES([source, savings, merge, sink], {shuffle: false});
  if (!sink.result) throw new Error('vrp-savings did not produce a result');
  sink.result.topology = stationGraph([source, savings, merge, sink], ['VRPProblemToken', 'VRPSavingsToken', 'VRPResultToken'], [
    channelEdge(source, VRPSourceStation.CH_PROBLEM, savings, SavingsStation.CH_PROBLEM),
    channelEdge(savings, SavingsStation.CH_SAVINGS, merge, RouteMergeStation.CH_SAVINGS),
    channelEdge(merge, RouteMergeStation.CH_RESULT, sink, VRPSinkStation.CH_RESULT),
  ]);
  return sink.result;
}

export function runVRPNearestNeighbor(params: VRPSavingsParams): VRPSavingsResult {
  const customers = nonEmptyArray(params.customers, defaultCustomers());
  const problem = new VRPProblemToken(params.depot ?? {x: 0, y: 0}, customers, params.vehicleCapacity ?? 5);
  const source = new VRPSourceStation('vrp-source', problem);
  const route = new NearestNeighborRouteStation('nearest-neighbor-route');
  const sink = new VRPSinkStation('vrp-sink');
  source.pipe(route, VRPSourceStation.CH_PROBLEM, NearestNeighborRouteStation.CH_PROBLEM);
  route.pipe(sink, NearestNeighborRouteStation.CH_RESULT, VRPSinkStation.CH_RESULT);
  runIterativeDES([source, route, sink], {shuffle: false});
  if (!sink.result) throw new Error('vrp-nearest-neighbor did not produce a result');
  sink.result.topology = stationGraph([source, route, sink], ['VRPProblemToken', 'VRPResultToken'], [
    channelEdge(source, VRPSourceStation.CH_PROBLEM, route, NearestNeighborRouteStation.CH_PROBLEM),
    channelEdge(route, NearestNeighborRouteStation.CH_RESULT, sink, VRPSinkStation.CH_RESULT),
  ]);
  return sink.result;
}

export interface JobOperation {
  machine: string;
  duration: number;
}

export interface JobShopJob {
  id: string;
  due?: number;
  operations: JobOperation[];
}

export interface JobShopDispatchParams {
  jobs?: JobShopJob[];
  rule?: 'fifo' | 'spt' | 'edd';
}

export interface ScheduledOperation {
  jobId: string;
  opIndex: number;
  machine: string;
  start: number;
  finish: number;
}

export interface JobShopDispatchResult {
  schedule: ScheduledOperation[];
  makespan: number;
  totalFlowTime: number;
  topology: StationGraphSummary;
}

export interface FlowShopJob {
  id: string;
  processingTimes: number[];
  due?: number;
}

export interface FlowShopNEHParams {
  jobs?: FlowShopJob[];
}

export interface FlowShopNEHResult {
  sequence: string[];
  schedule: ScheduledOperation[];
  makespan: number;
  totalFlowTime: number;
  topology: StationGraphSummary;
}

class JobToken implements Token {
  constructor(readonly job: JobShopJob) {}
}

class ScheduleToken implements Token {
  constructor(readonly result: JobShopDispatchResult) {}
}

class JobSourceStation extends DESStation {
  static readonly CH_JOB: ChannelName = 'job';
  private emitted = false;
  constructor(id: string, private readonly jobs: readonly JobShopJob[]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    for (const job of this.jobs) this.emit(new JobToken(job), JobSourceStation.CH_JOB);
    this.emitted = true;
  }
}

class DispatchSchedulerStation extends DESStation {
  static readonly CH_JOB: ChannelName = JobSourceStation.CH_JOB;
  static readonly CH_SCHEDULE: ChannelName = 'schedule';
  private readonly jobs: JobShopJob[] = [];
  private scheduled = false;
  constructor(id: string, private readonly rule: 'fifo' | 'spt' | 'edd') { super(id); }
  override hasWork(): boolean {
    return this.inboxSize(DispatchSchedulerStation.CH_JOB) > 0 || (!this.scheduled && this.jobs.length > 0);
  }
  runTimeStep(): void {
    const incoming = this.drain<JobToken>(DispatchSchedulerStation.CH_JOB);
    this.jobs.push(...incoming.map(t => t.job));
    if (incoming.length > 0) return;
    if (this.scheduled || this.jobs.length === 0) return;
    const result = dispatchSchedule(this.jobs, this.rule);
    this.emit(new ScheduleToken(result), DispatchSchedulerStation.CH_SCHEDULE);
    this.scheduled = true;
  }
}

class ScheduleSinkStation extends DESStation {
  static readonly CH_SCHEDULE: ChannelName = DispatchSchedulerStation.CH_SCHEDULE;
  result: JobShopDispatchResult | undefined;
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(ScheduleSinkStation.CH_SCHEDULE) > 0; }
  runTimeStep(): void {
    const schedules = this.drain<ScheduleToken>(ScheduleSinkStation.CH_SCHEDULE);
    if (schedules.length > 0) this.result = schedules[schedules.length - 1].result;
  }
}

class FlowJobToken implements Token {
  constructor(readonly job: FlowShopJob) {}
}

class FlowSequenceToken implements Token {
  constructor(readonly jobs: FlowShopJob[]) {}
}

class FlowScheduleToken implements Token {
  constructor(readonly result: FlowShopNEHResult) {}
}

class FlowShopJobSourceStation extends DESStation {
  static readonly CH_JOB: ChannelName = 'flow-job';
  private emitted = false;
  constructor(id: string, private readonly jobs: readonly FlowShopJob[]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    for (const job of this.jobs) this.emit(new FlowJobToken(job), FlowShopJobSourceStation.CH_JOB);
    this.emitted = true;
  }
}

class NEHSequenceStation extends DESStation {
  static readonly CH_JOB: ChannelName = FlowShopJobSourceStation.CH_JOB;
  static readonly CH_SEQUENCE: ChannelName = 'flow-sequence';
  private readonly jobs: FlowShopJob[] = [];
  private sequenced = false;
  constructor(id: string) { super(id); }
  override hasWork(): boolean {
    return this.inboxSize(NEHSequenceStation.CH_JOB) > 0 || (!this.sequenced && this.jobs.length > 0);
  }
  runTimeStep(): void {
    const incoming = this.drain<FlowJobToken>(NEHSequenceStation.CH_JOB);
    this.jobs.push(...incoming.map(t => t.job));
    if (incoming.length > 0) return;
    if (this.sequenced || this.jobs.length === 0) return;
    this.emit(new FlowSequenceToken(nehSequence(this.jobs)), NEHSequenceStation.CH_SEQUENCE);
    this.sequenced = true;
  }
}

class FlowShopScheduleStation extends DESStation {
  static readonly CH_SEQUENCE: ChannelName = NEHSequenceStation.CH_SEQUENCE;
  static readonly CH_SCHEDULE: ChannelName = 'flow-schedule';
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(FlowShopScheduleStation.CH_SEQUENCE) > 0; }
  runTimeStep(): void {
    for (const token of this.drain<FlowSequenceToken>(FlowShopScheduleStation.CH_SEQUENCE)) {
      const schedule = buildFlowShopSchedule(token.jobs);
      const makespan = schedule.reduce((best, op) => Math.max(best, op.finish), 0);
      const totalFlowTime = token.jobs.reduce((sum, job) => {
        const finish = Math.max(...schedule.filter(op => op.jobId === job.id).map(op => op.finish));
        return sum + finish;
      }, 0);
      this.emit(new FlowScheduleToken({
        sequence: token.jobs.map(job => job.id),
        schedule,
        makespan,
        totalFlowTime,
        topology: emptyStationGraph(),
      }), FlowShopScheduleStation.CH_SCHEDULE);
    }
  }
}

class FlowShopSinkStation extends DESStation {
  static readonly CH_SCHEDULE: ChannelName = FlowShopScheduleStation.CH_SCHEDULE;
  result: FlowShopNEHResult | undefined;
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(FlowShopSinkStation.CH_SCHEDULE) > 0; }
  runTimeStep(): void {
    const schedules = this.drain<FlowScheduleToken>(FlowShopSinkStation.CH_SCHEDULE);
    if (schedules.length > 0) this.result = schedules[schedules.length - 1].result;
  }
}

export function runJobShopDispatch(params: JobShopDispatchParams): JobShopDispatchResult {
  const jobs = nonEmptyArray(params.jobs, defaultJobs());
  const source = new JobSourceStation('job-source', jobs);
  const scheduler = new DispatchSchedulerStation('dispatch-scheduler', params.rule ?? 'spt');
  const sink = new ScheduleSinkStation('schedule-sink');
  source.pipe(scheduler, JobSourceStation.CH_JOB, DispatchSchedulerStation.CH_JOB);
  scheduler.pipe(sink, DispatchSchedulerStation.CH_SCHEDULE, ScheduleSinkStation.CH_SCHEDULE);
  runIterativeDES([source, scheduler, sink], {shuffle: false});
  if (!sink.result) throw new Error('job-shop-dispatch did not produce a result');
  sink.result.topology = stationGraph([source, scheduler, sink], ['JobToken', 'ScheduleToken'], [
    channelEdge(source, JobSourceStation.CH_JOB, scheduler, DispatchSchedulerStation.CH_JOB),
    channelEdge(scheduler, DispatchSchedulerStation.CH_SCHEDULE, sink, ScheduleSinkStation.CH_SCHEDULE),
  ]);
  return sink.result;
}

export function runFlowShopNEH(params: FlowShopNEHParams): FlowShopNEHResult {
  const jobs = nonEmptyArray(params.jobs, defaultFlowShopJobs());
  const source = new FlowShopJobSourceStation('flow-shop-source', jobs);
  const neh = new NEHSequenceStation('neh-sequence-builder');
  const scheduler = new FlowShopScheduleStation('flow-shop-scheduler');
  const sink = new FlowShopSinkStation('flow-shop-sink');
  source.pipe(neh, FlowShopJobSourceStation.CH_JOB, NEHSequenceStation.CH_JOB);
  neh.pipe(scheduler, NEHSequenceStation.CH_SEQUENCE, FlowShopScheduleStation.CH_SEQUENCE);
  scheduler.pipe(sink, FlowShopScheduleStation.CH_SCHEDULE, FlowShopSinkStation.CH_SCHEDULE);
  runIterativeDES([source, neh, scheduler, sink], {shuffle: false});
  if (!sink.result) throw new Error('flow-shop-neh did not produce a result');
  sink.result.topology = stationGraph([source, neh, scheduler, sink], ['FlowJobToken', 'FlowSequenceToken', 'FlowScheduleToken'], [
    channelEdge(source, FlowShopJobSourceStation.CH_JOB, neh, NEHSequenceStation.CH_JOB),
    channelEdge(neh, NEHSequenceStation.CH_SEQUENCE, scheduler, FlowShopScheduleStation.CH_SEQUENCE),
    channelEdge(scheduler, FlowShopScheduleStation.CH_SCHEDULE, sink, FlowShopSinkStation.CH_SCHEDULE),
  ]);
  return sink.result;
}

function qpObjective(Q: number[][], c: readonly number[], x: readonly number[]): number {
  const qx = Q.map(row => dot(row, x));
  return 0.5 * dot(x, qx) + dot(c, x);
}

function qpGradient(Q: number[][], c: readonly number[], x: readonly number[]): number[] {
  return Q.map((row, i) => dot(row, x) + c[i]);
}

function solveAssignmentDP(cost: number[][]): {assignment: number[]; objective: number} {
  const n = cost.length;
  const memo = new Map<string, {objective: number; assignment: number[]}>();
  const solve = (row: number, usedMask: number): {objective: number; assignment: number[]} => {
    if (row === n) return {objective: 0, assignment: []};
    const key = `${row}:${usedMask}`;
    const hit = memo.get(key);
    if (hit) return hit;
    let best = {objective: Infinity, assignment: [] as number[]};
    for (let col = 0; col < cost[row].length; col++) {
      if ((usedMask & (1 << col)) !== 0) continue;
      const tail = solve(row + 1, usedMask | (1 << col));
      const objective = cost[row][col] + tail.objective;
      if (objective < best.objective) best = {objective, assignment: [col, ...tail.assignment]};
    }
    memo.set(key, best);
    return best;
  };
  return solve(0, 0);
}

function dist(a: {x: number; y: number}, b: {x: number; y: number}): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function routeDistance(depot: {x: number; y: number}, customers: readonly VRPCustomer[]): number {
  if (customers.length === 0) return 0;
  let d = dist(depot, customers[0]);
  for (let i = 1; i < customers.length; i++) d += dist(customers[i - 1], customers[i]);
  return d + dist(customers[customers.length - 1], depot);
}

function defaultCustomers(): VRPCustomer[] {
  return [
    {id: 'A', x: 1, y: 2, demand: 2},
    {id: 'B', x: 2, y: 1, demand: 2},
    {id: 'C', x: 4, y: 1, demand: 2},
    {id: 'D', x: 5, y: 2, demand: 1},
    {id: 'E', x: 3, y: 4, demand: 2},
  ];
}

function dispatchSchedule(jobs: readonly JobShopJob[], rule: 'fifo' | 'spt' | 'edd'): JobShopDispatchResult {
  const machineReady = new Map<string, number>();
  const jobReady = new Map<string, number>();
  const remaining = jobs.map(job => ({job, opIndex: 0}));
  const schedule: ScheduledOperation[] = [];
  while (remaining.some(r => r.opIndex < r.job.operations.length)) {
    const ready = remaining.filter(r => r.opIndex < r.job.operations.length);
    ready.sort((a, b) => {
      if (rule === 'edd') return (a.job.due ?? Infinity) - (b.job.due ?? Infinity);
      if (rule === 'spt') return a.job.operations[a.opIndex].duration - b.job.operations[b.opIndex].duration;
      return jobs.indexOf(a.job) - jobs.indexOf(b.job);
    });
    const next = ready[0];
    const op = next.job.operations[next.opIndex];
    const start = Math.max(machineReady.get(op.machine) ?? 0, jobReady.get(next.job.id) ?? 0);
    const finish = start + op.duration;
    schedule.push({jobId: next.job.id, opIndex: next.opIndex, machine: op.machine, start, finish});
    machineReady.set(op.machine, finish);
    jobReady.set(next.job.id, finish);
    next.opIndex += 1;
  }
  const makespan = Math.max(...schedule.map(op => op.finish));
  const totalFlowTime = jobs.reduce((sum, job) => {
    const finish = Math.max(...schedule.filter(op => op.jobId === job.id).map(op => op.finish));
    return sum + finish;
  }, 0);
  return {schedule, makespan, totalFlowTime, topology: emptyStationGraph()};
}

function defaultJobs(): JobShopJob[] {
  return [
    {id: 'J1', due: 10, operations: [{machine: 'M1', duration: 3}, {machine: 'M2', duration: 2}]},
    {id: 'J2', due: 8, operations: [{machine: 'M2', duration: 2}, {machine: 'M1', duration: 4}]},
    {id: 'J3', due: 12, operations: [{machine: 'M1', duration: 2}, {machine: 'M2', duration: 3}]},
  ];
}

function nehSequence(jobs: readonly FlowShopJob[]): FlowShopJob[] {
  const ordered = jobs.slice().sort((a, b) => totalProcessingTime(b) - totalProcessingTime(a));
  let sequence: FlowShopJob[] = [];
  for (const job of ordered) {
    let best = [job, ...sequence];
    let bestMakespan = flowShopMakespan(best);
    for (let pos = 1; pos <= sequence.length; pos++) {
      const candidate = sequence.slice();
      candidate.splice(pos, 0, job);
      const makespan = flowShopMakespan(candidate);
      if (makespan < bestMakespan) {
        best = candidate;
        bestMakespan = makespan;
      }
    }
    sequence = best;
  }
  return sequence;
}

function buildFlowShopSchedule(sequence: readonly FlowShopJob[]): ScheduledOperation[] {
  if (sequence.length === 0) return [];
  const machines = sequence[0].processingTimes.length;
  const machineReady = zeros(machines);
  const schedule: ScheduledOperation[] = [];
  for (const job of sequence) {
    let jobReady = 0;
    for (let m = 0; m < machines; m++) {
      const start = Math.max(machineReady[m], jobReady);
      const finish = start + job.processingTimes[m];
      schedule.push({jobId: job.id, opIndex: m, machine: `M${m + 1}`, start, finish});
      machineReady[m] = finish;
      jobReady = finish;
    }
  }
  return schedule;
}

function flowShopMakespan(sequence: readonly FlowShopJob[]): number {
  return buildFlowShopSchedule(sequence).reduce((best, op) => Math.max(best, op.finish), 0);
}

function totalProcessingTime(job: FlowShopJob): number {
  return job.processingTimes.reduce((sum, v) => sum + v, 0);
}

function defaultFlowShopJobs(): FlowShopJob[] {
  return [
    {id: 'F1', processingTimes: [2, 3, 2]},
    {id: 'F2', processingTimes: [4, 1, 3]},
    {id: 'F3', processingTimes: [3, 2, 4]},
    {id: 'F4', processingTimes: [2, 5, 1]},
  ];
}
