'use strict';

// =============================================================================
// general/multistage-stochastic.ts -- MULTI-STAGE STOCHASTIC PROGRAMMING
// via an SDDP-style discrete-event system.
//
// The model is a compact inventory / storage problem:
//
//   state s_t        inventory available at the start of stage t
//   noise D_t        random demand observed at stage t
//   decision         order q_t, sell y_t, stockout u_t, ending inventory s_{t+1}
//   balance          s_{t+1} = s_t + q_t - y_t
//   demand           y_t + u_t = D_t
//   bounds           0 <= q_t <= maxOrder_t, 0 <= s_{t+1} <= capacity
//
// Maximise expected profit:
//
//   price_t*y_t - orderCost_t*q_t - holdCost_t*s_{t+1}
//       - stockoutCost_t*u_t + terminalSalvage*s_T.
//
// SDDP representation:
//   - each stage owns an upper affine cut pool for the concave value V_t(s)
//   - one DES tick performs a forward sampled trajectory and a backward cut pass
//   - each backward stage LP uses the next stage's cut pool through a theta var
//   - a tiny exact extensive-form scenario tree is included for validation
//
// This is intentionally one-dimensional so the algorithm is readable in a
// course codebase while still demonstrating the real multi-stage recursion.
// =============================================================================

import {LPProblem, solveLPInternal} from './lp';
import {mulberry32} from './prng';
import {
  AffineCut, AffineCutPool, FixedPointIterationStation,
  runIterativeDES, intrinsicCheck,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

export interface DemandOutcome {
  demand: number;
  prob: number;
}

export interface MultiStageInventoryProblem {
  horizon: number;
  initialInventory: number;
  capacity: number;
  maxOrder: number[];
  price: number[];
  orderCost: number[];
  holdCost: number[];
  stockoutCost: number[];
  salvageValue: number;
  demands: DemandOutcome[][];
}

export interface StageDecision {
  status: 'optimal' | 'infeasible' | 'unbounded' | 'iter-limit' | 'numerical-error';
  value: number;
  immediateReward: number;
  order: number;
  sell: number;
  stockout: number;
  nextInventory: number;
  theta: number;
}

export interface SDDPIterationTrace {
  iter: number;
  sampledDemands: number[];
  states: number[];
  terminalInventory: number;
  cutsAdded: Array<{stage: number; alpha: number; beta: number; state: number}>;
  upperBound: number;
  policyValue?: number;
  gapToExact?: number;
}

export interface ExactTreeNodeResult {
  objective: number;
  nodeCount: number;
  lpVars: number;
  lpRows: number;
  status: string;
}

export interface SDDPOptions {
  maxIter?: number;
  tol?: number;
  seed?: number;
  exactObjective?: number;
  evaluatePolicyEvery?: number;
  finiteDiffStep?: number;
  cutGridSize?: number;
}

interface SDDPFilledOptions {
  maxIter: number;
  tol: number;
  seed: number;
  exactObjective?: number;
  evaluatePolicyEvery: number;
  finiteDiffStep: number;
  cutGridSize: number;
}

export interface SDDPResult {
  status: 'optimal' | 'iter-limit';
  iterations: number;
  upperBound: number;
  policyValue: number;
  exactObjective?: number;
  gapToExact?: number;
  cutsPerStage: number[];
  cuts: AffineCut[][];
  trace: SDDPIterationTrace[];
  samplePath: Array<{stage: number; demand: number; state: number; order: number; sell: number; stockout: number; nextInventory: number}>;
}

export interface MultiStageRunResult {
  exact: ExactTreeNodeResult;
  sddp: SDDPResult;
}

const MODEL = 'multi-stage-sddp';

// -----------------------------------------------------------------------------
// Public problem builders
// -----------------------------------------------------------------------------

export function buildDefaultMultiStageInventoryProblem(): MultiStageInventoryProblem {
  return {
    horizon: 4,
    initialInventory: 4,
    capacity: 10,
    maxOrder: [5, 5, 5, 5],
    price: [9, 9, 10, 10],
    orderCost: [3, 3, 4, 4],
    holdCost: [0.25, 0.25, 0.35, 0.35],
    stockoutCost: [7, 7, 8, 8],
    salvageValue: 1.5,
    demands: [
      [{demand: 2, prob: 0.45}, {demand: 6, prob: 0.55}],
      [{demand: 1, prob: 0.35}, {demand: 5, prob: 0.65}],
      [{demand: 3, prob: 0.50}, {demand: 7, prob: 0.50}],
      [{demand: 2, prob: 0.60}, {demand: 6, prob: 0.40}],
    ],
  };
}

export function validateMultiStageProblem(p: MultiStageInventoryProblem): void {
  Preconditions.integerInRange(MODEL, 'horizon', p.horizon, 1, 200);
  Preconditions.positive(MODEL, 'capacity', p.capacity);
  Preconditions.inRange(MODEL, 'initialInventory', p.initialInventory, 0, p.capacity);
  Preconditions.lengthEq(MODEL, 'maxOrder', p.maxOrder, p.horizon);
  Preconditions.lengthEq(MODEL, 'price', p.price, p.horizon);
  Preconditions.lengthEq(MODEL, 'orderCost', p.orderCost, p.horizon);
  Preconditions.lengthEq(MODEL, 'holdCost', p.holdCost, p.horizon);
  Preconditions.lengthEq(MODEL, 'stockoutCost', p.stockoutCost, p.horizon);
  Preconditions.lengthEq(MODEL, 'demands', p.demands, p.horizon);
  Preconditions.arrNonNegative(MODEL, 'maxOrder', p.maxOrder);
  Preconditions.arrNonNegative(MODEL, 'price', p.price);
  Preconditions.arrNonNegative(MODEL, 'orderCost', p.orderCost);
  Preconditions.arrNonNegative(MODEL, 'holdCost', p.holdCost);
  Preconditions.arrNonNegative(MODEL, 'stockoutCost', p.stockoutCost);
  Preconditions.nonNegative(MODEL, 'salvageValue', p.salvageValue);
  for (let t = 0; t < p.horizon; t++) {
    Preconditions.nonEmpty(MODEL, `demands[${t}]`, p.demands[t]);
    const probs = p.demands[t].map(d => d.prob);
    Preconditions.probabilityVector(MODEL, `demands[${t}].prob`, probs);
    for (let i = 0; i < p.demands[t].length; i++) {
      Preconditions.nonNegative(MODEL, `demands[${t}][${i}].demand`, p.demands[t][i].demand);
    }
  }
}

// -----------------------------------------------------------------------------
// Stage LP
// -----------------------------------------------------------------------------

export function solveStageDecision(
  p: MultiStageInventoryProblem,
  stage: number,
  state: number,
  demand: number,
  nextCuts: AffineCutPool,
): StageDecision {
  validateStageInputs(p, stage, state, demand, nextCuts);
  const c = [
    -p.orderCost[stage],
    p.price[stage],
    -p.stockoutCost[stage],
    -p.holdCost[stage],
    1,
  ];
  const A_ub: number[][] = [
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
  ];
  const b_ub: number[] = [p.maxOrder[stage], p.capacity];
  for (const cut of nextCuts.all()) {
    A_ub.push([0, 0, 0, -cut.beta[0], 1]);
    b_ub.push(cut.alpha);
  }
  const lp: LPProblem = {
    sense: 'max',
    c,
    A_ub,
    b_ub,
    A_eq: [
      [-1, 1, 0, 1, 0],
      [0, 1, 1, 0, 0],
    ],
    b_eq: [state, demand],
    lb: [0, 0, 0, 0, null],
    varNames: ['order', 'sell', 'stockout', 'nextInventory', 'theta'],
  };
  const sol = solveLPInternal(lp, {maxIter: 10000});
  if (sol.status !== 'optimal') {
    return {
      status: sol.status,
      value: NaN,
      immediateReward: NaN,
      order: NaN,
      sell: NaN,
      stockout: NaN,
      nextInventory: NaN,
      theta: NaN,
    };
  }
  const [order, sell, stockout, nextInventory, theta] = sol.x;
  const immediateReward =
    p.price[stage] * sell
    - p.orderCost[stage] * order
    - p.holdCost[stage] * nextInventory
    - p.stockoutCost[stage] * stockout;
  return {
    status: 'optimal',
    value: sol.objective,
    immediateReward,
    order,
    sell,
    stockout,
    nextInventory,
    theta,
  };
}

function validateStageInputs(
  p: MultiStageInventoryProblem,
  stage: number,
  state: number,
  demand: number,
  nextCuts: AffineCutPool,
): void {
  Preconditions.integerInRange(MODEL, 'stage', stage, 0, p.horizon - 1);
  Preconditions.inRange(MODEL, 'state', state, 0, p.capacity);
  Preconditions.nonNegative(MODEL, 'demand', demand);
  Preconditions.check(MODEL, 'nextCuts.dimension', 'equal 1', nextCuts.dimension === 1, nextCuts.dimension);
  Preconditions.check(MODEL, 'nextCuts.size()', 'be >= 1', nextCuts.size() >= 1, nextCuts.size());
}

export function expectedStageValue(
  p: MultiStageInventoryProblem,
  stage: number,
  state: number,
  nextCuts: AffineCutPool,
): number {
  let z = 0;
  for (const d of p.demands[stage]) {
    const dec = solveStageDecision(p, stage, state, d.demand, nextCuts);
    if (dec.status !== 'optimal') throw new Error(`${MODEL}: stage LP failed with status ${dec.status}`);
    z += d.prob * dec.value;
  }
  return z;
}

function generateValueCut(
  p: MultiStageInventoryProblem,
  stage: number,
  state: number,
  nextCuts: AffineCutPool,
  opts: Required<Pick<SDDPOptions, 'finiteDiffStep' | 'cutGridSize'>>,
  source: string,
): AffineCut {
  const value = expectedStageValue(p, stage, state, nextCuts);
  const h = Math.min(Math.max(opts.finiteDiffStep, 1e-7), p.capacity);
  let beta: number;
  if (state <= h) {
    const up = expectedStageValue(p, stage, Math.min(p.capacity, state + h), nextCuts);
    beta = (up - value) / Math.max(1e-12, Math.min(p.capacity, state + h) - state);
  } else if (p.capacity - state <= h) {
    const lo = expectedStageValue(p, stage, Math.max(0, state - h), nextCuts);
    beta = (value - lo) / Math.max(1e-12, state - Math.max(0, state - h));
  } else {
    const up = expectedStageValue(p, stage, state + h, nextCuts);
    const lo = expectedStageValue(p, stage, state - h, nextCuts);
    beta = (up - lo) / (2 * h);
  }
  let alpha = value - beta * state;

  // Finite differences produce a slope, not a formal LP dual. Lift the cut
  // over a small state grid so it remains a valid upper cut on the domain.
  const gridN = Math.max(2, opts.cutGridSize);
  let maxViolation = 0;
  for (let i = 0; i < gridN; i++) {
    const x = p.capacity * i / (gridN - 1);
    const vx = expectedStageValue(p, stage, x, nextCuts);
    const cutx = alpha + beta * x;
    if (vx > cutx + maxViolation) maxViolation = vx - cutx;
  }
  alpha += maxViolation + 1e-8;
  return {alpha, beta: [beta], source};
}

// -----------------------------------------------------------------------------
// SDDP DES station
// -----------------------------------------------------------------------------

interface SDDPState {
  iter: number;
  upperBound: number;
  policyValue?: number;
  gapToExact?: number;
}

export class SDDPStation extends FixedPointIterationStation<SDDPState> {
  readonly cutPools: AffineCutPool[] = [];
  readonly trace: SDDPIterationTrace[] = [];
  lastSamplePath: SDDPResult['samplePath'] = [];

  private readonly p: MultiStageInventoryProblem;
  private readonly rng: () => number;
  private readonly exactObjective?: number;
  private readonly evaluatePolicyEvery: number;
  private readonly finiteDiffStep: number;
  private readonly cutGridSize: number;
  private finalStatus: 'optimal' | 'iter-limit' = 'iter-limit';

  constructor(p: MultiStageInventoryProblem, opts: SDDPFilledOptions) {
    super(MODEL, {maxIter: opts.maxIter, tol: opts.tol, maxHistoryLen: Infinity});
    validateMultiStageProblem(p);
    this.p = p;
    this.rng = mulberry32(opts.seed);
    this.exactObjective = opts.exactObjective;
    this.evaluatePolicyEvery = opts.evaluatePolicyEvery;
    this.finiteDiffStep = opts.finiteDiffStep;
    this.cutGridSize = opts.cutGridSize;
    this.initialiseCutPools();
    this.bootstrap();

    this.addValidator(intrinsicCheck<SDDPStation>({
      name: 'sddp.cut-pools-nonempty',
      group: 'sddp-intrinsic',
      predicate: st => st.cutPools.every(pool => pool.size() >= 1),
      expected: 'every stage has at least one affine cut',
      observedFn: st => st.cutPools.map(pool => pool.size()).join(','),
    }));
    this.addValidator(intrinsicCheck<SDDPStation>({
      name: 'sddp.upper-bound-above-exact',
      group: 'sddp-intrinsic',
      predicate: st => st.exactObjective === undefined || st.getCurrent().upperBound + 1e-5 >= st.exactObjective,
      expected: 'SDDP upper approximation >= exact objective',
      observedFn: st => `upper=${st.getCurrent().upperBound}, exact=${st.exactObjective}`,
    }));
  }

  getStatus(): 'optimal' | 'iter-limit' { return this.finalStatus; }

  protected initialState(): SDDPState {
    return {iter: 0, upperBound: this.cutPools[0].evaluate([this.p.initialInventory])};
  }

  protected applyOperator(prev: SDDPState): SDDPState {
    const iter = prev.iter + 1;
    const states = new Array<number>(this.p.horizon + 1);
    const sampledDemands: number[] = [];
    const path: SDDPResult['samplePath'] = [];
    states[0] = this.p.initialInventory;

    for (let t = 0; t < this.p.horizon; t++) {
      const demand = sampleDemand(this.p.demands[t], this.rng);
      sampledDemands.push(demand);
      const dec = solveStageDecision(this.p, t, states[t], demand, this.cutPools[t + 1]);
      if (dec.status !== 'optimal') throw new Error(`${MODEL}: forward LP failed at stage ${t}: ${dec.status}`);
      path.push({
        stage: t,
        demand,
        state: states[t],
        order: dec.order,
        sell: dec.sell,
        stockout: dec.stockout,
        nextInventory: dec.nextInventory,
      });
      states[t + 1] = clamp(dec.nextInventory, 0, this.p.capacity);
    }
    this.lastSamplePath = path;

    const cutsAdded: SDDPIterationTrace['cutsAdded'] = [];
    for (let t = this.p.horizon - 1; t >= 0; t--) {
      const cut = generateValueCut(
        this.p,
        t,
        states[t],
        this.cutPools[t + 1],
        {finiteDiffStep: this.finiteDiffStep, cutGridSize: this.cutGridSize},
        `iter=${iter} stage=${t}`,
      );
      this.cutPools[t].add(cut);
      cutsAdded.push({stage: t, alpha: cut.alpha, beta: cut.beta[0], state: states[t]});
    }

    const upperBound = this.cutPools[0].evaluate([this.p.initialInventory]);
    let policyValue: number | undefined;
    let gapToExact: number | undefined;
    if (iter % this.evaluatePolicyEvery === 0 || iter >= this.maxIter || this.exactObjective !== undefined) {
      policyValue = evaluatePolicyExact(this.p, this.cutPools);
      gapToExact = this.exactObjective === undefined ? undefined : this.exactObjective - policyValue;
    }
    this.trace.push({
      iter,
      sampledDemands,
      states: states.slice(),
      terminalInventory: states[this.p.horizon],
      cutsAdded,
      upperBound,
      policyValue,
      gapToExact,
    });
    if (gapToExact !== undefined && Math.abs(gapToExact) <= this.tol) this.finalStatus = 'optimal';
    return {iter, upperBound, policyValue, gapToExact};
  }

  protected delta(prev: SDDPState, next: SDDPState): number {
    if (this.exactObjective !== undefined) return Math.abs(next.upperBound - this.exactObjective);
    return Math.abs(prev.upperBound - next.upperBound);
  }

  protected override shouldStop(iter: number, _lastDelta: number): boolean {
    if (this.finalStatus === 'optimal' && iter > 0) {
      this.convergenceReason = 'converged';
      return true;
    }
    if (iter >= this.maxIter) {
      this.finalStatus = this.finalStatus === 'optimal' ? 'optimal' : 'iter-limit';
      this.convergenceReason = this.finalStatus === 'optimal' ? 'converged' : 'maxiter';
      return true;
    }
    return false;
  }

  private initialiseCutPools(): void {
    const remainingRevenueUpper = new Array<number>(this.p.horizon + 1).fill(0);
    remainingRevenueUpper[this.p.horizon] = this.p.salvageValue * this.p.capacity;
    for (let t = this.p.horizon - 1; t >= 0; t--) {
      const maxDemand = Math.max(...this.p.demands[t].map(d => d.demand));
      remainingRevenueUpper[t] = remainingRevenueUpper[t + 1] + this.p.price[t] * maxDemand;
    }
    for (let t = 0; t <= this.p.horizon; t++) {
      const pool = new AffineCutPool(1, 'upper');
      if (t === this.p.horizon) {
        pool.add({alpha: 0, beta: [this.p.salvageValue], source: 'terminal-salvage'});
      } else {
        pool.add({alpha: remainingRevenueUpper[t], beta: [0], source: 'initial-constant-upper'});
      }
      this.cutPools.push(pool);
    }
  }
}

export function solveMultiStageSDDP(p: MultiStageInventoryProblem, opts: SDDPOptions = {}): SDDPResult {
  validateMultiStageProblem(p);
  const filled: SDDPFilledOptions = {
    maxIter: opts.maxIter ?? 80,
    tol: opts.tol ?? 1e-4,
    seed: opts.seed ?? 1,
    exactObjective: opts.exactObjective,
    evaluatePolicyEvery: opts.evaluatePolicyEvery ?? Number.MAX_SAFE_INTEGER,
    finiteDiffStep: opts.finiteDiffStep ?? Math.max(1e-4, p.capacity * 1e-5),
    cutGridSize: opts.cutGridSize ?? 21,
  };
  const station = new SDDPStation(p, filled);
  runIterativeDES([station]);
  const policyValue = evaluatePolicyExact(p, station.cutPools);
  const current = station.getCurrent();
  const exact = filled.exactObjective;
  return {
    status: station.getStatus(),
    iterations: station.getIteration(),
    upperBound: current.upperBound,
    policyValue,
    exactObjective: exact,
    gapToExact: exact === undefined ? undefined : exact - policyValue,
    cutsPerStage: station.cutPools.map(pool => pool.size()),
    cuts: station.cutPools.map(pool => pool.all()),
    trace: station.trace,
    samplePath: station.lastSamplePath,
  };
}

export function runMultiStageInventoryDemo(
  p: MultiStageInventoryProblem,
  opts: SDDPOptions = {},
): MultiStageRunResult {
  const exact = solveExactScenarioTree(p);
  const sddp = solveMultiStageSDDP(p, {...opts, exactObjective: exact.objective});
  return {exact, sddp};
}

// -----------------------------------------------------------------------------
// Exact extensive-form scenario tree LP
// -----------------------------------------------------------------------------

interface TreeNode {
  id: number;
  stage: number;
  demand: number;
  prob: number;
  parentId: number | null;
}

export function solveExactScenarioTree(p: MultiStageInventoryProblem): ExactTreeNodeResult {
  validateMultiStageProblem(p);
  const nodes = buildScenarioTree(p);
  const varCount = nodes.length * 4; // order, sell, stockout, nextInventory per node
  const idx = (nodeId: number, local: 0 | 1 | 2 | 3): number => nodeId * 4 + local;
  const c = new Array<number>(varCount).fill(0);
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  const A_eq: number[][] = [];
  const b_eq: number[] = [];

  for (const node of nodes) {
    const t = node.stage;
    c[idx(node.id, 0)] += node.prob * -p.orderCost[t];
    c[idx(node.id, 1)] += node.prob * p.price[t];
    c[idx(node.id, 2)] += node.prob * -p.stockoutCost[t];
    c[idx(node.id, 3)] += node.prob * -p.holdCost[t];
    if (t === p.horizon - 1) c[idx(node.id, 3)] += node.prob * p.salvageValue;

    const bal = new Array<number>(varCount).fill(0);
    bal[idx(node.id, 0)] = -1;
    bal[idx(node.id, 1)] = 1;
    bal[idx(node.id, 3)] = 1;
    if (node.parentId === null) {
      A_eq.push(bal); b_eq.push(p.initialInventory);
    } else {
      bal[idx(node.parentId, 3)] = -1;
      A_eq.push(bal); b_eq.push(0);
    }

    const demandRow = new Array<number>(varCount).fill(0);
    demandRow[idx(node.id, 1)] = 1;
    demandRow[idx(node.id, 2)] = 1;
    A_eq.push(demandRow); b_eq.push(node.demand);

    const orderBound = new Array<number>(varCount).fill(0);
    orderBound[idx(node.id, 0)] = 1;
    A_ub.push(orderBound); b_ub.push(p.maxOrder[t]);

    const invBound = new Array<number>(varCount).fill(0);
    invBound[idx(node.id, 3)] = 1;
    A_ub.push(invBound); b_ub.push(p.capacity);
  }

  const sol = solveLPInternal({
    sense: 'max',
    c,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    lb: new Array(varCount).fill(0),
  }, {maxIter: 100000});
  return {
    objective: sol.objective,
    nodeCount: nodes.length,
    lpVars: varCount,
    lpRows: A_ub.length + A_eq.length,
    status: sol.status,
  };
}

function buildScenarioTree(p: MultiStageInventoryProblem): TreeNode[] {
  const nodes: TreeNode[] = [];
  let frontier: Array<{parentId: number | null; prob: number}> = [{parentId: null, prob: 1}];
  for (let t = 0; t < p.horizon; t++) {
    const next: Array<{parentId: number | null; prob: number}> = [];
    for (const parent of frontier) {
      for (const d of p.demands[t]) {
        const id = nodes.length;
        nodes.push({id, stage: t, demand: d.demand, prob: parent.prob * d.prob, parentId: parent.parentId});
        next.push({parentId: id, prob: parent.prob * d.prob});
      }
    }
    frontier = next;
  }
  return nodes;
}

// -----------------------------------------------------------------------------
// Policy evaluation
// -----------------------------------------------------------------------------

export function evaluatePolicyExact(
  p: MultiStageInventoryProblem,
  cutPools: ReadonlyArray<AffineCutPool>,
): number {
  validateMultiStageProblem(p);
  Preconditions.lengthEq(MODEL, 'cutPools', cutPools, p.horizon + 1);
  const rec = (stage: number, state: number): number => {
    if (stage >= p.horizon) return p.salvageValue * state;
    let z = 0;
    for (const d of p.demands[stage]) {
      const dec = solveStageDecision(p, stage, state, d.demand, cutPools[stage + 1]);
      if (dec.status !== 'optimal') throw new Error(`${MODEL}: policy eval LP failed at stage ${stage}: ${dec.status}`);
      z += d.prob * (dec.immediateReward + rec(stage + 1, clamp(dec.nextInventory, 0, p.capacity)));
    }
    return z;
  };
  return rec(0, p.initialInventory);
}

function sampleDemand(outcomes: ReadonlyArray<DemandOutcome>, rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (const o of outcomes) {
    acc += o.prob;
    if (u <= acc) return o.demand;
  }
  return outcomes[outcomes.length - 1].demand;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
