'use strict';

// =============================================================================
// general/max-flow.ts -- maximum flow as a discrete-event optimisation.
//
// Nodes are stationary optimisation state; an augmenting path is the movable
// "event" that changes residual capacity. One Edmonds-Karp augmentation is one
// DES tick via FixedPointIterationStation. This keeps the max-flow model in the
// same iterative-algorithm family as Benders, SDDP, value iteration, and MILP
// branch-and-bound.
// =============================================================================

import {
  assertNoValidationFailures,
  FixedPointIterationStation,
  intrinsicCheck,
  runIterativeDES,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

export interface MaxFlowEdge {
  from: number;
  to: number;
  capacity: number;
  name?: string;
}

export interface MaxFlowProblem {
  numNodes: number;
  source: number;
  sink: number;
  edges: MaxFlowEdge[];
}

export interface MaxFlowTraceEntry {
  iter: number;
  path: number[];
  bottleneck: number;
  flowAfter: number;
}

export interface MaxFlowResult {
  status: 'optimal' | 'infeasible';
  maxFlow: number;
  source: number;
  sink: number;
  numNodes: number;
  edgeFlows: Array<MaxFlowEdge & {flow: number}>;
  minCut: {
    sourceSide: number[];
    sinkSide: number[];
    cutEdges: Array<MaxFlowEdge & {flow: number}>;
    capacity: number;
  };
  iterations: number;
  trace: MaxFlowTraceEntry[];
}

interface ResidualEdge {
  to: number;
  rev: number;
  cap: number;
  originalIndex: number;
}

interface ForwardRef {
  from: number;
  edgeIndex: number;
}

interface AugmentingPath {
  nodes: number[];
  bottleneck: number;
  parentNode: number[];
  parentEdge: number[];
}

interface MaxFlowState {
  iter: number;
  flow: number;
  done: boolean;
}

const MODEL = 'max-flow';

export function validateMaxFlowProblem(p: MaxFlowProblem): void {
  Preconditions.integerInRange(MODEL, 'numNodes', p.numNodes, 2, 1e7);
  Preconditions.integerInRange(MODEL, 'source', p.source, 0, p.numNodes - 1);
  Preconditions.integerInRange(MODEL, 'sink', p.sink, 0, p.numNodes - 1);
  Preconditions.check(MODEL, 'source != sink', 'hold', p.source !== p.sink, [p.source, p.sink]);
  Preconditions.nonEmpty(MODEL, 'edges', p.edges);
  for (let i = 0; i < p.edges.length; i++) {
    const e = p.edges[i];
    Preconditions.integerInRange(MODEL, `edges[${i}].from`, e.from, 0, p.numNodes - 1);
    Preconditions.integerInRange(MODEL, `edges[${i}].to`, e.to, 0, p.numNodes - 1);
    Preconditions.nonNegative(MODEL, `edges[${i}].capacity`, e.capacity);
  }
}

export class MaxFlowStation extends FixedPointIterationStation<MaxFlowState> {
  private readonly p: MaxFlowProblem;
  private readonly residual: ResidualEdge[][];
  private readonly forwardRefs: ForwardRef[] = [];
  readonly trace: MaxFlowTraceEntry[] = [];
  private finalFlow = 0;

  constructor(p: MaxFlowProblem) {
    super(MODEL, {tol: 0, maxIter: Math.max(1, p.numNodes * Math.max(1, p.edges.length) * Math.max(1, p.edges.length) + 1)});
    validateMaxFlowProblem(p);
    this.p = p;
    this.residual = Array.from({length: p.numNodes}, () => []);
    for (let i = 0; i < p.edges.length; i++) this.addResidualEdge(p.edges[i], i);
    this.bootstrap();

    this.addValidator(intrinsicCheck<MaxFlowStation>({
      name: 'max-flow.conservation',
      group: 'max-flow-intrinsic',
      predicate: st => st.conservationError() <= 1e-8,
      expected: 'flow conserved at every non-terminal node',
      observedFn: st => `max imbalance=${st.conservationError().toExponential(3)}`,
    }));
    this.addValidator(intrinsicCheck<MaxFlowStation>({
      name: 'max-flow.cut-equals-flow',
      group: 'max-flow-intrinsic',
      predicate: st => Math.abs(st.buildResult().minCut.capacity - st.finalFlow) <= 1e-8,
      expected: 'min-cut capacity equals max flow',
      observedFn: st => `cut=${st.buildResult().minCut.capacity}, flow=${st.finalFlow}`,
    }));
  }

  protected initialState(): MaxFlowState {
    return {iter: 0, flow: 0, done: false};
  }

  protected applyOperator(prev: MaxFlowState): MaxFlowState {
    const aug = this.findAugmentingPath();
    if (!aug) {
      this.finalFlow = prev.flow;
      return {iter: prev.iter + 1, flow: prev.flow, done: true};
    }
    for (let v = this.p.sink; v !== this.p.source; v = aug.parentNode[v]) {
      const u = aug.parentNode[v];
      const ei = aug.parentEdge[v];
      const e = this.residual[u][ei];
      e.cap -= aug.bottleneck;
      this.residual[e.to][e.rev].cap += aug.bottleneck;
    }
    const flow = prev.flow + aug.bottleneck;
    this.finalFlow = flow;
    this.trace.push({
      iter: prev.iter + 1,
      path: aug.nodes,
      bottleneck: aug.bottleneck,
      flowAfter: flow,
    });
    return {iter: prev.iter + 1, flow, done: false};
  }

  protected delta(_prev: MaxFlowState, next: MaxFlowState): number {
    return next.done ? 0 : Infinity;
  }

  protected override shouldStop(iter: number, _lastDelta: number): boolean {
    if (iter > 0 && this.current?.done) {
      this.convergenceReason = 'converged';
      return true;
    }
    return super.shouldStop(iter, _lastDelta);
  }

  getResidual(): ResidualEdge[][] {
    return this.residual;
  }

  buildResult(): MaxFlowResult {
    const edgeFlows = this.p.edges.map((e, i) => {
      const ref = this.forwardRefs[i];
      const residualEdge = this.residual[ref.from][ref.edgeIndex];
      return {...e, flow: e.capacity - residualEdge.cap};
    });
    const sourceSide = this.reachableFromSource();
    const sourceSet = new Set(sourceSide);
    const sinkSide: number[] = [];
    for (let i = 0; i < this.p.numNodes; i++) if (!sourceSet.has(i)) sinkSide.push(i);
    const cutEdges = edgeFlows.filter(e => sourceSet.has(e.from) && !sourceSet.has(e.to));
    const cutCapacity = cutEdges.reduce((s, e) => s + e.capacity, 0);
    return {
      status: 'optimal',
      maxFlow: this.finalFlow,
      source: this.p.source,
      sink: this.p.sink,
      numNodes: this.p.numNodes,
      edgeFlows,
      minCut: {sourceSide, sinkSide, cutEdges, capacity: cutCapacity},
      iterations: this.getIteration(),
      trace: this.trace.slice(),
    };
  }

  private addResidualEdge(e: MaxFlowEdge, originalIndex: number): void {
    const fwd: ResidualEdge = {to: e.to, rev: this.residual[e.to].length, cap: e.capacity, originalIndex};
    const rev: ResidualEdge = {to: e.from, rev: this.residual[e.from].length, cap: 0, originalIndex};
    this.residual[e.from].push(fwd);
    this.residual[e.to].push(rev);
    this.forwardRefs[originalIndex] = {from: e.from, edgeIndex: this.residual[e.from].length - 1};
  }

  private findAugmentingPath(): AugmentingPath | null {
    const n = this.p.numNodes;
    const parentNode = new Array<number>(n).fill(-1);
    const parentEdge = new Array<number>(n).fill(-1);
    const q: number[] = [this.p.source];
    parentNode[this.p.source] = this.p.source;
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      for (let ei = 0; ei < this.residual[u].length; ei++) {
        const e = this.residual[u][ei];
        if (e.cap <= 1e-12 || parentNode[e.to] !== -1) continue;
        parentNode[e.to] = u;
        parentEdge[e.to] = ei;
        if (e.to === this.p.sink) {
          const nodes: number[] = [];
          let bottleneck = Infinity;
          for (let v = this.p.sink; v !== this.p.source; v = parentNode[v]) {
            nodes.push(v);
            const pe = this.residual[parentNode[v]][parentEdge[v]];
            bottleneck = Math.min(bottleneck, pe.cap);
          }
          nodes.push(this.p.source);
          nodes.reverse();
          return {nodes, bottleneck, parentNode, parentEdge};
        }
        q.push(e.to);
      }
    }
    return null;
  }

  private reachableFromSource(): number[] {
    const seen = new Array<boolean>(this.p.numNodes).fill(false);
    const q = [this.p.source];
    seen[this.p.source] = true;
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi];
      for (const e of this.residual[u]) {
        if (e.cap > 1e-12 && !seen[e.to]) {
          seen[e.to] = true;
          q.push(e.to);
        }
      }
    }
    const out: number[] = [];
    for (let i = 0; i < seen.length; i++) if (seen[i]) out.push(i);
    return out;
  }

  private conservationError(): number {
    const flows = this.buildResult().edgeFlows;
    const balance = new Array<number>(this.p.numNodes).fill(0);
    for (const e of flows) {
      balance[e.from] -= e.flow;
      balance[e.to] += e.flow;
    }
    let err = 0;
    for (let i = 0; i < balance.length; i++) {
      if (i === this.p.source || i === this.p.sink) continue;
      err = Math.max(err, Math.abs(balance[i]));
    }
    return err;
  }
}

export function solveMaxFlow(p: MaxFlowProblem): MaxFlowResult {
  const st = new MaxFlowStation(p);
  const summary = runIterativeDES([st], {shuffle: false});
  assertNoValidationFailures(summary, 'max-flow');
  return st.buildResult();
}

export function buildTextbookMaxFlowProblem(): MaxFlowProblem {
  return {
    numNodes: 6,
    source: 0,
    sink: 5,
    edges: [
      {from: 0, to: 1, capacity: 16, name: 's-v1'},
      {from: 0, to: 2, capacity: 13, name: 's-v2'},
      {from: 1, to: 2, capacity: 10, name: 'v1-v2'},
      {from: 2, to: 1, capacity: 4, name: 'v2-v1'},
      {from: 1, to: 3, capacity: 12, name: 'v1-v3'},
      {from: 3, to: 2, capacity: 9, name: 'v3-v2'},
      {from: 2, to: 4, capacity: 14, name: 'v2-v4'},
      {from: 4, to: 3, capacity: 7, name: 'v4-v3'},
      {from: 3, to: 5, capacity: 20, name: 'v3-t'},
      {from: 4, to: 5, capacity: 4, name: 'v4-t'},
    ],
  };
}
