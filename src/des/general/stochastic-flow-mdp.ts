'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/stochastic-flow-mdp.rs  (module des::general::stochastic_flow_mdp)
// 1:1 file move. Max-flow recast as a finite-horizon stochastic-control MDP (backward Bellman).
//
// Declarations → Rust:
//   interface StochasticFlowEdge/StochasticFlowMDPProblem/FlowMDPState/FlowMDPAction/
//             FlowMDPDecision/FlowMDPSimStep/StochasticFlowMDPResult -> structs (#[derive(Clone)])
//   interface IndexedAction extends FlowMDPAction          -> struct (compose, no inheritance)
//   class StochasticFlowMDPStation extends FiniteHorizonDPStation -> struct + impl DP-station trait
//   fn validateStochasticFlowMDPProblem/solveStochasticFlowMDP/simulateStochasticFlowPolicy/
//      buildDefaultStochasticFlowMDPProblem/stateKey/cloneState -> free/private fns
//
// Conversion notes (file-specific):
//   - `stateKey(s): string` builds a STRING key for the DP value/policy maps -> implement
//     `Hash + Eq` on `FlowMDPState` (node + remaining capacities) and key a `HashMap` directly.
//   - INJECT RNG: `simulateStochasticFlowPolicy` samples edge availability -> `RandomSource`.
//   - node/edge indices `usize`; capacities/flow/reward `f64`; DPOutcome from des-base.
//   - validate* throws -> `panic!` (invariant) or `Result`.
// =============================================================================
// general/stochastic-flow-mdp.ts -- max-flow as stochastic sequential control.
//
// Deterministic max-flow asks for a static feasible circulation certificate.
// If edge capacities/availability evolve stochastically, the same network
// becomes an MDP:
//
//   state  = (current packet node, remaining edge capacities)
//   action = choose an outgoing edge to try, or wait
//   noise  = selected edge is available/unavailable this tick
//   reward = delivered flow value minus routing/failure/waiting costs
//
// The Bellman recursion is the stochastic-flow-control counterpart of a
// max-flow augmenting path:
//
//   V_t(n, c) = max_a E[ r(n, c, a, W_t) + V_{t+1}(n', c') ]
//
// Each FiniteHorizonDPStation tick performs one backward Bellman stage.
// =============================================================================

import {
  assertNoValidationFailures,
  DPOutcome,
  FiniteHorizonDPStation,
  intrinsicCheck,
  runIterativeDES,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';
import {MaxFlowProblem, solveMaxFlow} from './max-flow';

export interface StochasticFlowEdge {
  from: number;
  to: number;
  capacity: number;
  /** Probability that one unit successfully traverses the edge when tried. */
  successProb: number;
  /** Optional per-attempt cost. Defaults to 0. */
  cost?: number;
  name?: string;
}

export interface StochasticFlowMDPProblem {
  numNodes: number;
  source: number;
  sink: number;
  edges: StochasticFlowEdge[];
  /** Number of sequential control ticks. */
  horizon: number;
  /** Reward for delivering one unit to the sink. Defaults to 1. */
  deliveredReward?: number;
  /** Penalty for choosing to wait. Defaults to 0. */
  waitPenalty?: number;
  /** Penalty when a chosen edge is unavailable. Defaults to 0. */
  failurePenalty?: number;
  /** Discount factor. Defaults to 1 for finite-horizon total reward. */
  discount?: number;
  /** Guardrail for exact state enumeration. Defaults to 20000. */
  maxStates?: number;
}

export interface FlowMDPState {
  node: number;
  capacities: number[];
}

export interface FlowMDPAction {
  kind: 'wait' | 'edge';
  edgeIndex?: number;
  label: string;
}

export interface FlowMDPDecision {
  stage: number;
  stateIndex: number;
  state: FlowMDPState;
  action: FlowMDPAction;
  value: number;
}

export interface FlowMDPSimStep {
  stage: number;
  nodeBefore: number;
  action: FlowMDPAction;
  success: boolean;
  nodeAfter: number;
  reward: number;
  deliveredSoFar: number;
  capacitiesAfter: number[];
}

export interface StochasticFlowMDPResult {
  status: 'optimal';
  horizon: number;
  numStates: number;
  initialStateIndex: number;
  expectedReward: number;
  deterministicMaxFlow: number;
  policy: FlowMDPDecision[];
  initialPolicy: FlowMDPDecision[];
  stageHistory: Array<{stage: number; maxV: number; minV: number}>;
  simulation: {
    seed: number;
    delivered: number;
    totalReward: number;
    finalState: FlowMDPState;
    steps: FlowMDPSimStep[];
  };
}

interface IndexedAction extends FlowMDPAction {
  edgeIndex?: number;
}

const MODEL = 'stochastic-flow-mdp';

export function validateStochasticFlowMDPProblem(p: StochasticFlowMDPProblem): void {
  Preconditions.integerInRange(MODEL, 'numNodes', p.numNodes, 2, 1000);
  Preconditions.integerInRange(MODEL, 'source', p.source, 0, p.numNodes - 1);
  Preconditions.integerInRange(MODEL, 'sink', p.sink, 0, p.numNodes - 1);
  Preconditions.check(MODEL, 'source != sink', 'hold', p.source !== p.sink, [p.source, p.sink]);
  Preconditions.nonEmpty(MODEL, 'edges', p.edges);
  Preconditions.integerInRange(MODEL, 'horizon', p.horizon, 1, 1000);
  Preconditions.positive(MODEL, 'deliveredReward', p.deliveredReward ?? 1);
  Preconditions.nonNegative(MODEL, 'waitPenalty', p.waitPenalty ?? 0);
  Preconditions.nonNegative(MODEL, 'failurePenalty', p.failurePenalty ?? 0);
  Preconditions.inRange(MODEL, 'discount', p.discount ?? 1, 0, 1);
  Preconditions.integerInRange(MODEL, 'maxStates', p.maxStates ?? 20000, 1, 1e7);
  for (let i = 0; i < p.edges.length; i++) {
    const e = p.edges[i];
    Preconditions.integerInRange(MODEL, `edges[${i}].from`, e.from, 0, p.numNodes - 1);
    Preconditions.integerInRange(MODEL, `edges[${i}].to`, e.to, 0, p.numNodes - 1);
    Preconditions.integerInRange(MODEL, `edges[${i}].capacity`, e.capacity, 0, 100);
    Preconditions.inRange(MODEL, `edges[${i}].successProb`, e.successProb, 0, 1);
    if (e.cost !== undefined) Preconditions.nonNegative(MODEL, `edges[${i}].cost`, e.cost);
  }
}

export class StochasticFlowMDPStation extends FiniteHorizonDPStation {
  readonly states: FlowMDPState[] = [];
  readonly initialStateIndex: number;

  private readonly p: StochasticFlowMDPProblem;
  private readonly keyToIndex = new Map<string, number>();
  private readonly outgoing = new Map<number, number[]>();
  private readonly actionCache = new Map<number, IndexedAction[]>();

  constructor(p: StochasticFlowMDPProblem) {
    super(MODEL);
    validateStochasticFlowMDPProblem(p);
    this.p = p;
    for (let i = 0; i < p.edges.length; i++) {
      const arr = this.outgoing.get(p.edges[i].from) ?? [];
      arr.push(i);
      this.outgoing.set(p.edges[i].from, arr);
    }
    this.enumerateStates();
    this.initialStateIndex = this.stateIndex({
      node: p.source,
      capacities: p.edges.map(e => e.capacity),
    });
    this.bootstrap();

    this.addValidator(intrinsicCheck<StochasticFlowMDPStation>({
      name: 'stochastic-flow-mdp.policy-actions-legal',
      group: 'stochastic-flow-mdp-intrinsic',
      predicate: st => st.policyActionsLegal(),
      expected: 'every policy action is legal for its state',
      observedFn: st => `states=${st.states.length}, horizon=${st.p.horizon}`,
    }));
    this.addValidator(intrinsicCheck<StochasticFlowMDPStation>({
      name: 'stochastic-flow-mdp.values-finite',
      group: 'stochastic-flow-mdp-intrinsic',
      predicate: st => st.V.every(row => row.every(Number.isFinite)),
      expected: 'all value-function entries finite',
      observedFn: st => `V0(initial)=${st.V[0]?.[st.initialStateIndex]}`,
    }));
  }

  protected horizon(): number { return this.p.horizon; }
  protected numStates(): number { return this.states.length; }
  protected stageDiscount(_stage: number): number { return this.p.discount ?? 1; }

  protected numActions(state: number, _stage: number): number {
    return this.legalActions(state).length;
  }

  protected transitions(stateIndex: number, actionIndex: number, _stage: number): DPOutcome[] {
    const state = this.states[stateIndex];
    const action = this.legalActions(stateIndex)[actionIndex];
    if (!action || action.kind === 'wait') {
      return [{prob: 1, reward: -(this.p.waitPenalty ?? 0), nextState: stateIndex}];
    }

    const edgeIndex = action.edgeIndex!;
    const edge = this.p.edges[edgeIndex];
    const attemptCost = edge.cost ?? 0;
    const failPenalty = this.p.failurePenalty ?? 0;
    const pSucc = edge.successProb;
    const nextCaps = state.capacities.slice();
    nextCaps[edgeIndex] -= 1;
    const delivered = edge.to === this.p.sink;
    const nextNode = delivered ? this.p.source : edge.to;
    const successState = this.stateIndex({node: nextNode, capacities: nextCaps});
    const successReward = (delivered ? (this.p.deliveredReward ?? 1) : 0) - attemptCost;
    const failureReward = -attemptCost - failPenalty;
    if (pSucc <= 0) return [{prob: 1, reward: failureReward, nextState: stateIndex}];
    if (pSucc >= 1) return [{prob: 1, reward: successReward, nextState: successState}];
    return [
      {prob: pSucc, reward: successReward, nextState: successState},
      {prob: 1 - pSucc, reward: failureReward, nextState: stateIndex},
    ];
  }

  legalActions(stateIndex: number): IndexedAction[] {
    const cached = this.actionCache.get(stateIndex);
    if (cached) return cached;
    const state = this.states[stateIndex];
    const out: IndexedAction[] = [{kind: 'wait', label: 'wait'}];
    if (state.node !== this.p.sink) {
      for (const edgeIndex of this.outgoing.get(state.node) ?? []) {
        if (state.capacities[edgeIndex] <= 0) continue;
        const e = this.p.edges[edgeIndex];
        out.push({
          kind: 'edge',
          edgeIndex,
          label: e.name ?? `${e.from}->${e.to}`,
        });
      }
    }
    this.actionCache.set(stateIndex, out);
    return out;
  }

  getActionDetail(stage: number, stateIndex: number): FlowMDPAction {
    const actionIndex = this.policy[stage][stateIndex];
    return this.legalActions(stateIndex)[actionIndex] ?? {kind: 'wait', label: 'wait'};
  }

  buildResult(seed = 1, maxPolicyRows = 24): StochasticFlowMDPResult {
    const initialPolicy: FlowMDPDecision[] = [];
    let s = this.initialStateIndex;
    for (let t = 0; t < this.p.horizon; t++) {
      const a = this.getActionDetail(t, s);
      initialPolicy.push({
        stage: t,
        stateIndex: s,
        state: cloneState(this.states[s]),
        action: {...a},
        value: this.V[t][s],
      });
      const outs = this.transitions(s, this.policy[t][s], t);
      const deliveredSuccess = outs.find(o => o.nextState !== s) ?? outs[0];
      s = deliveredSuccess.nextState;
    }
    return {
      status: 'optimal',
      horizon: this.p.horizon,
      numStates: this.states.length,
      initialStateIndex: this.initialStateIndex,
      expectedReward: this.V[0][this.initialStateIndex],
      deterministicMaxFlow: solveMaxFlow(this.asDeterministicMaxFlow()).maxFlow,
      policy: this.compactPolicy(maxPolicyRows),
      initialPolicy,
      stageHistory: this.stageHistory.slice(),
      simulation: simulateStochasticFlowPolicy(this.p, this, seed),
    };
  }

  asDeterministicMaxFlow(): MaxFlowProblem {
    return {
      numNodes: this.p.numNodes,
      source: this.p.source,
      sink: this.p.sink,
      edges: this.p.edges.map(e => ({from: e.from, to: e.to, capacity: e.capacity, name: e.name})),
    };
  }

  indexOfState(s: FlowMDPState): number {
    return this.stateIndex(s);
  }

  private compactPolicy(maxRows: number): FlowMDPDecision[] {
    const rows: FlowMDPDecision[] = [];
    for (let t = 0; t < this.p.horizon && rows.length < maxRows; t++) {
      for (let s = 0; s < this.states.length && rows.length < maxRows; s++) {
        const a = this.getActionDetail(t, s);
        if (a.kind === 'wait' && this.V[t][s] <= 1e-12) continue;
        rows.push({
          stage: t,
          stateIndex: s,
          state: cloneState(this.states[s]),
          action: {...a},
          value: this.V[t][s],
        });
      }
    }
    return rows;
  }

  private enumerateStates(): void {
    const caps = this.p.edges.map(e => e.capacity);
    const current = new Array<number>(caps.length).fill(0);
    const visitCaps = (idx: number) => {
      if (idx === caps.length) {
        for (let node = 0; node < this.p.numNodes; node++) {
          const st = {node, capacities: current.slice()};
          const key = stateKey(st);
          this.keyToIndex.set(key, this.states.length);
          this.states.push(st);
          if (this.states.length > (this.p.maxStates ?? 20000)) {
            throw new Error(`${MODEL}: exact state space exceeds maxStates=${this.p.maxStates ?? 20000}`);
          }
        }
        return;
      }
      for (let c = 0; c <= caps[idx]; c++) {
        current[idx] = c;
        visitCaps(idx + 1);
      }
    };
    visitCaps(0);
  }

  private stateIndex(s: FlowMDPState): number {
    const ix = this.keyToIndex.get(stateKey(s));
    if (ix === undefined) throw new Error(`${MODEL}: missing enumerated state ${stateKey(s)}`);
    return ix;
  }

  private policyActionsLegal(): boolean {
    for (let t = 0; t < this.p.horizon; t++) {
      for (let s = 0; s < this.states.length; s++) {
        const a = this.policy[t]?.[s];
        if (a === undefined) return false;
        if (a < 0 || a >= this.legalActions(s).length) return false;
      }
    }
    return true;
  }
}

export function solveStochasticFlowMDP(
  p: StochasticFlowMDPProblem,
  opts: {seed?: number; maxPolicyRows?: number} = {},
): StochasticFlowMDPResult {
  const station = new StochasticFlowMDPStation(p);
  const summary = runIterativeDES([station], {shuffle: false});
  assertNoValidationFailures(summary, MODEL);
  return station.buildResult(opts.seed ?? 1, opts.maxPolicyRows ?? 24);
}

export function simulateStochasticFlowPolicy(
  p: StochasticFlowMDPProblem,
  station: StochasticFlowMDPStation,
  seed: number,
): StochasticFlowMDPResult['simulation'] {
  const rng = mulberry32(seed);
  let stateIndex = station.initialStateIndex;
  let delivered = 0;
  let totalReward = 0;
  const steps: FlowMDPSimStep[] = [];
  for (let t = 0; t < p.horizon; t++) {
    const state = station.states[stateIndex];
    const action = station.getActionDetail(t, stateIndex);
    const before = state.node;
    let success = false;
    let reward = -(p.waitPenalty ?? 0);
    let nextStateIndex = stateIndex;
    if (action.kind === 'edge') {
      const edge = p.edges[action.edgeIndex!];
      success = rng() < edge.successProb;
      reward = -(edge.cost ?? 0);
      if (success) {
        const nextCaps = state.capacities.slice();
        nextCaps[action.edgeIndex!] -= 1;
        const reachedSink = edge.to === p.sink;
        if (reachedSink) {
          delivered++;
          reward += p.deliveredReward ?? 1;
        }
        nextStateIndex = station.indexOfState({
          node: reachedSink ? p.source : edge.to,
          capacities: nextCaps,
        });
      } else {
        reward -= p.failurePenalty ?? 0;
      }
    }
    totalReward += reward;
    const nextState = station.states[nextStateIndex];
    steps.push({
      stage: t,
      nodeBefore: before,
      action: {...action},
      success,
      nodeAfter: nextState.node,
      reward,
      deliveredSoFar: delivered,
      capacitiesAfter: nextState.capacities.slice(),
    });
    stateIndex = nextStateIndex;
  }
  return {
    seed,
    delivered,
    totalReward,
    finalState: cloneState(station.states[stateIndex]),
    steps,
  };
}

export function buildDefaultStochasticFlowMDPProblem(): StochasticFlowMDPProblem {
  return {
    numNodes: 5,
    source: 0,
    sink: 4,
    horizon: 8,
    deliveredReward: 1,
    waitPenalty: 0.01,
    failurePenalty: 0.03,
    discount: 1,
    edges: [
      {from: 0, to: 1, capacity: 2, successProb: 0.90, cost: 0.01, name: 's-a'},
      {from: 1, to: 4, capacity: 2, successProb: 0.80, cost: 0.01, name: 'a-t'},
      {from: 0, to: 2, capacity: 2, successProb: 0.65, cost: 0.01, name: 's-b'},
      {from: 2, to: 4, capacity: 2, successProb: 0.95, cost: 0.01, name: 'b-t'},
      {from: 1, to: 2, capacity: 1, successProb: 0.75, cost: 0.01, name: 'a-b'},
      {from: 2, to: 1, capacity: 1, successProb: 0.70, cost: 0.01, name: 'b-a'},
    ],
    maxStates: 10000,
  };
}

function stateKey(s: FlowMDPState): string {
  return `${s.node}|${s.capacities.join(',')}`;
}

function cloneState(s: FlowMDPState): FlowMDPState {
  return {node: s.node, capacities: s.capacities.slice()};
}
