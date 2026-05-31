// RUST MIGRATION: target module src/des/general/ip_mip_des.rs.
// RUST MIGRATION: Relaxation algorithm unions, token state union, trace events, and branch/cut payloads become enums with serde tags.
// RUST MIGRATION: IPMIPProblem, options, solutions, performance stats, topology nodes, and constraints become serde structs with Vec<f64>/Vec<Vec<f64>> matrices.
// RUST MIGRATION: PayloadStatefulToken subclasses and DESStation/CompositeDESStation subclasses become structs implementing Token, StatefulToken, Station, and CompositeStation traits.
// RUST MIGRATION: solveIPMIPWithDES and buildIPMIPSolverTechniquePlan are graph-visible solver transforms; expose them as PureTransform entry structs returning Result.
// RUST MIGRATION: Partial<Record<...>> usage maps to HashMap<ConcreteLPRelaxationAlgorithm, usize>, and all validation/LP relaxation failures should flow through Result/status enums.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/ip-mip-des.rs  (module des::general::ip_mip_des)
// 1:1 file move. Integer/mixed-integer programming as a branch-and-cut DES station graph.
//
// Declarations → Rust:
//   type LPRelaxationAlgorithm / ConcreteLPRelaxationAlgorithm -> enums (string-literal unions)
//   type IPMIPTokenState = ... | ...  -> enum (discriminated union; `match` on kind)
//   interface IPMIP{Problem,SolveOptions,Solution,ProblemFeatures,SolverTechniquePlan,...} -> structs
//   class NodeToken/CompleteToken/RelaxationToken/CandidateToken/CutToken (extend PayloadStatefulToken)
//                                    -> structs `impl Token` carrying a typed payload
//   class *Station (extend DESStation) / BranchAndCutSolverStation (CompositeDESStation)
//                                    -> structs `impl` the station traits (bases -> traits)
//   fn solveIPMIPWithDES / analyzeIPMIPProblem / buildIPMIPSolverTechniquePlan / build* -> fns
//
// Conversion notes (file-specific):
//   - `IPMIPTokenState` and `LPRelaxationAlgorithm` are discriminated/string unions -> Rust enums.
//   - Pluggable LP backend (incremental/DES/internal/external) -> enum dispatch or `dyn LpBackend` trait.
//   - `as any` casts + `Partial<Record<algo, number>>` usage counters -> concrete types / `HashMap<Algo,u64>`.
//   - Stateful tokens flow through the graph mutating payloads -> `Rc<RefCell<..>>` or arena indices.
//   - `throw`/Preconditions on bad problems -> `Result`/`panic!` per recoverability.
// =============================================================================

// =============================================================================
// general/ip-mip-des.ts -- integer / mixed-integer programming as a DES graph.
//
// This module is deliberately one level more explicit than `milp-bnb.ts`.
// Instead of a single branch-and-bound station, it builds a graph of
// stationary solver roles and lets movable tokens carry subproblems,
// relaxation results, cuts, and integer candidates:
//
//   SearchController ──node──▶ LPRelaxation ──relaxation──▶ RoundingRepair
//          ▲                           │             │             │
//          │                           │             │             ▼
//       child nodes                    │             └────cut──▶ NodeDecision
//          │                           │                           │
//          └──────────────complete/children◀───────────────────────┘
//                                      │
//                                      ▼
//                                  Incumbent
//
// The LP relaxation station is intentionally pluggable: it can solve the
// same node with an incremental primal/dual simplex, DES simplex, internal
// two-phase simplex, or SciPy/HiGHS bridge. That gives uploaded JSON/MPS-like
// problem sets a stable route into the framework: parse the problem, build
// the station graph, choose the relaxation backend, and run the DES.
// =============================================================================

import {
  assertNoValidationFailures,
  CompositeDESStation,
  DESStation,
  PayloadStatefulToken,
  Token,
  TokenStateMode,
  StatefulTokenRegistry,
  StatefulTokenRegistryStats,
  StatefulToken,
  intrinsicCheck,
  runIterativeDES,
  transitionToken,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {IncrementalLP} from './incremental-lp';
import {LPProblem, LPSolution, solveLPExternal, solveLPInternal} from './lp';
import {solveLPViaDES} from './lp-des';

// -----------------------------------------------------------------------------
// Public problem / result types
// -----------------------------------------------------------------------------

export type ConcreteLPRelaxationAlgorithm =
  | 'incremental-primal-dual'
  | 'des-simplex-dantzig'
  | 'des-simplex-bland'
  | 'internal-simplex'
  | 'external-highs'
  | 'external-highs-ds'
  | 'external-highs-ipm';

export type LPRelaxationAlgorithm = ConcreteLPRelaxationAlgorithm | 'auto';

export interface IPMIPProblemFeatures {
  variableCount: number;
  constraintCount: number;
  integerCount: number;
  continuousCount: number;
  binaryCount: number;
  finiteUpperBounds: number;
  nonzeros: number;
  density: number;
  allInteger: boolean;
  allBinary: boolean;
  constraintVariableComponents: number;
}

export interface IPMIPSolverTechniquePlan {
  requestedLPAlgorithm: LPRelaxationAlgorithm;
  rootLPAlgorithm: ConcreteLPRelaxationAlgorithm;
  externalSolversAllowed: boolean;
  usesExternalSolvers: boolean;
  externalCandidate: boolean;
  primalDualDynamic: boolean;
  decompositionCandidate: boolean;
  decompositionReason?: string;
  rationale: string[];
  features: IPMIPProblemFeatures;
}

export interface IPMIPProblem {
  sense: 'max' | 'min';
  c: number[];
  A: number[][];
  b: number[];
  integerVars: boolean[];
  ub?: number[];
  varNames?: string[];
  conNames?: string[];
  /** Optional graph metadata for interpreting variables as movable entities. */
  variableNodes?: Array<{varIndex: number; nodeId: string; label?: string}>;
  /** Optional graph metadata for stationary constraint anchors. */
  constraintNodes?: Array<{rowIndex: number; nodeId: string; label?: string}>;
}

export interface IPMIPSolveOptions {
  maxNodes?: number;
  maxTicks?: number;
  /** Wall-clock cap for the DES solver loop. Returns the best incumbent found. */
  timeLimitMs?: number;
  lpMaxIters?: number;
  intTol?: number;
  branchRule?: 'most-fractional' | 'first-fractional';
  nodeSelection?: 'dfs' | 'best-bound';
  lpAlgorithm?: LPRelaxationAlgorithm;
  /** Defaults to false. External LP bridges are used only when explicitly enabled. */
  allowExternalSolvers?: boolean;
  maxCutRounds?: number;
  maxCutsPerNode?: number;
  heuristicPasses?: number;
  verbose?: boolean;
}

interface FilledIPMIPSolveOptions {
  maxNodes: number;
  maxTicks: number;
  timeLimitMs: number;
  lpMaxIters: number;
  intTol: number;
  branchRule: 'most-fractional' | 'first-fractional';
  nodeSelection: 'dfs' | 'best-bound';
  lpAlgorithm: LPRelaxationAlgorithm;
  allowExternalSolvers: boolean;
  maxCutRounds: number;
  maxCutsPerNode: number;
  heuristicPasses: number;
  verbose: boolean;
}

export interface IPMIPSolution {
  status: 'optimal' | 'infeasible' | 'unbounded' | 'maxnodes' | 'tick-limit' | 'time-limit';
  x: number[];
  z: number;
  bestBound: number;
  gap: number;
  nodesExplored: number;
  lpSolves: number;
  totalLPIterations: number;
  cutsAdded: number;
  candidatesTried: number;
  lpAlgorithm: LPRelaxationAlgorithm;
  lpAlgorithmUsage: Partial<Record<ConcreteLPRelaxationAlgorithm, number>>;
  techniquePlan: IPMIPSolverTechniquePlan;
  incumbentSource?: string;
  elapsedMs: number;
  inHouseOnly: boolean;
  usesExternalSolvers: boolean;
  performance: IPMIPPerformanceStats;
  solverKind: 'in-house-branch-and-cut';
  executionMode: 'single-threaded';
  compositeStationId: string;
  tokenStats: SolverTokenStats;
  trace: IPMIPTraceEvent[];
  topology: SolverTopologyNode[];
}

export interface IPMIPPerformanceStats {
  elapsedMs: number;
  ticks: number;
  nodesPerSecond: number;
  lpSolvesPerSecond: number;
  msPerNode: number;
  totalLPSolverMs: number;
  avgLPSolverMs: number;
  lpSolverTimeShare: number;
  avgLPIterationsPerSolve: number;
  cutsPerNode: number;
  candidatesPerNode: number;
  tokensCreated: number;
}

export interface IPMIPTraceEvent {
  nodeId: number;
  parentId: number | null;
  depth: number;
  lpStatus: LPSolution['status'];
  lpZ: number | null;
  solver: string;
  fractional: number[];
  action: 'branch' | 'cut' | 'prune' | 'incumbent' | 'unbounded';
  reason?: string;
  branchVar?: number;
  children?: number[];
  cutsAdded?: number;
  nodeTokenId?: string;
  lineageRoot?: string;
  tokenGeneration?: number;
  stateMode?: TokenStateMode;
}

export interface SolverTopologyNode {
  id: string;
  role: string;
  emits: string[];
  parentId?: string;
}

export type SolverTokenStats = StatefulTokenRegistryStats;

export interface BranchOrCutConstraint {
  coefs: number[];
  rhs: number;
  name: string;
  kind: 'branch' | 'cut';
}

interface IPNode {
  nodeId: number;
  parentId: number | null;
  depth: number;
  constraints: BranchOrCutConstraint[];
  cutRounds: number;
  branchVar: number | null;
  branchType: 'le' | 'ge' | null;
  branchValue: number | null;
  boundGuess?: number;
}

interface RelaxationPayload {
  node: IPNode;
  status: LPSolution['status'];
  x: number[];
  z: number;
  solver: string;
  selectedAlgorithm: ConcreteLPRelaxationAlgorithm;
  iters: number;
  fractional: number[];
}

interface CandidatePayload {
  nodeId: number;
  x: number[];
  z: number;
  source: string;
}

interface CutPayload {
  nodeId: number;
  cut: BranchOrCutConstraint;
}

type IPMIPTokenState =
  | 'queued'
  | 'relaxation-queued'
  | 'relaxed'
  | 'candidate'
  | 'cut'
  | 'complete';

class NodeToken extends PayloadStatefulToken<IPMIPTokenState, IPNode> {
  constructor(node: IPNode, opts: {
    tokenId: string;
    tick: number;
    stationId: string;
    parent?: StatefulToken<any>;
    event?: string;
    detail?: string;
  }) {
    super({kind: 'ip-node', payload: node, initialState: 'queued', ...opts});
  }
  get node(): IPNode { return this.payload; }
}

class CompleteToken extends PayloadStatefulToken<IPMIPTokenState, {nodeId: number}> {
  constructor(nodeId: number, opts: {
    tokenId: string;
    tick: number;
    stationId: string;
    parent?: StatefulToken<any>;
  }) {
    super({
      kind: 'ip-complete',
      payload: {nodeId},
      initialState: 'complete',
      event: 'node-complete',
      stateMode: 'stateless',
      ...opts,
    });
  }
  get nodeId(): number { return this.payload.nodeId; }
}

class RelaxationToken extends PayloadStatefulToken<IPMIPTokenState, RelaxationPayload> {
  constructor(payload: RelaxationPayload, parent: StatefulToken<any>, opts: {
    tokenId: string;
    tick: number;
    stationId: string;
  }) {
    super({kind: 'ip-relaxation', payload, initialState: 'relaxed', parent, event: 'lp-relaxed', ...opts});
  }
}

class CandidateToken extends PayloadStatefulToken<IPMIPTokenState, CandidatePayload> {
  constructor(payload: CandidatePayload, parent: StatefulToken<any>, opts: {
    tokenId: string;
    tick: number;
    stationId: string;
  }) {
    super({kind: 'ip-candidate', payload, initialState: 'candidate', parent, event: 'candidate-generated', ...opts});
  }
}

class CutToken extends PayloadStatefulToken<IPMIPTokenState, CutPayload> {
  constructor(payload: CutPayload, parent: StatefulToken<any>, opts: {
    tokenId: string;
    tick: number;
    stationId: string;
  }) {
    super({kind: 'ip-cut', payload, initialState: 'cut', parent, event: 'cut-generated', ...opts});
  }
}

const MODEL = 'ip-mip-des';
const EPS = 1e-9;

// -----------------------------------------------------------------------------
// Station graph
// -----------------------------------------------------------------------------

class SearchControllerStation extends DESStation {
  private readonly frontier: NodeToken[] = [];
  private inFlight = 0;
  private done = false;
  private maxNodesHit = false;
  nodesDispatched = 0;
  private nextNodeId = 1;
  private tick = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly opts: Required<Pick<IPMIPSolveOptions, 'maxNodes' | 'nodeSelection'>>,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-search-controller');
    const root: IPNode = {
      nodeId: 0, parentId: null, depth: 0, constraints: [], cutRounds: 0,
      branchVar: null, branchType: null, branchValue: null,
    };
    const tok = new NodeToken(root, {
      tokenId: 'ip-node-0',
      tick: this.tick,
      stationId: this.id,
      event: 'root-created',
    });
    this.tokenRegistry.track(tok);
    this.frontier.push(tok);
  }

  allocateNodeId(): number { return this.nextNodeId++; }
  hitNodeLimit(): boolean { return this.maxNodesHit; }
  frontierSize(): number { return this.frontier.length; }

  bestFrontierBound(): number | null {
    const finite = this.frontier.map(t => t.node.boundGuess).filter((x): x is number => Number.isFinite(x));
    if (finite.length === 0) return null;
    return this.p.sense === 'max' ? Math.max(...finite) : Math.min(...finite);
  }

  override hasWork(): boolean {
    return !this.done || super.hasWork();
  }

  runTimeStep(): void {
    for (const t of this.drain<NodeToken>('nodes')) this.pushNode(t);
    for (const t of this.drain<CompleteToken>('complete')) {
      this.tokenRegistry.track(t);
      this.inFlight = Math.max(0, this.inFlight - 1);
    }

    if (this.done) return;
    if (this.nodesDispatched >= this.opts.maxNodes) {
      if (this.frontier.length > 0) this.maxNodesHit = true;
      if (this.inFlight === 0) this.done = true;
      this.tick++;
      return;
    }
    const tok = this.popNode();
    if (!tok) {
      if (this.inFlight === 0) this.done = true;
      this.tick++;
      return;
    }
    this.nodesDispatched++;
    this.inFlight++;
    transitionToken(tok, 'relaxation-queued', {
      tick: this.tick,
      stationId: this.id,
      event: 'dispatch-to-relaxation',
    });
    this.tokenRegistry.track(tok);
    this.emit(tok, 'relax');
    this.tick++;
  }

  private pushNode(token: NodeToken): void {
    this.tokenRegistry.track(token);
    if (this.opts.nodeSelection === 'dfs') {
      this.frontier.push(token);
      return;
    }
    this.frontier.push(token);
    this.frontier.sort((a, b) => {
      const ba = a.node.boundGuess ?? (this.p.sense === 'max' ? -Infinity : Infinity);
      const bb = b.node.boundGuess ?? (this.p.sense === 'max' ? -Infinity : Infinity);
      return this.p.sense === 'max' ? ba - bb : bb - ba;
    });
  }

  private popNode(): NodeToken | null {
    return this.frontier.pop() ?? null;
  }
}

export class LPRelaxationStation extends DESStation {
  lpSolves = 0;
  totalIterations = 0;
  totalSolverElapsedMs = 0;
  readonly algorithmUsage: Partial<Record<ConcreteLPRelaxationAlgorithm, number>> = {};
  private tick = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly algorithm: LPRelaxationAlgorithm,
    private readonly techniquePlan: IPMIPSolverTechniquePlan,
    private readonly lpMaxIters: number,
    private readonly intTol: number,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-lp-relaxation');
  }

  runTimeStep(): void {
    const nodes = this.drain<NodeToken>('nodes');
    for (const tok of nodes) {
      this.tokenRegistry.track(tok);
      const selected = selectLPRelaxationAlgorithm(this.p, tok.node, this.algorithm, this.techniquePlan);
      let used = selected;
      let r = solveNodeRelaxation(this.p, tok.node, selected, this.lpMaxIters);
      if (this.algorithm === 'auto' && isExternalLPAlgorithm(selected) && r.status === 'numerical-error') {
        const fallbackMessage = r.message ?? 'external solver unavailable';
        used = hasNegativeRootRHS(this.p) ? 'internal-simplex' : 'incremental-primal-dual';
        r = solveNodeRelaxation(this.p, tok.node, used, this.lpMaxIters);
        r.message = `${r.message ?? ''}${r.message ? ' | ' : ''}auto fallback from ${selected}: ${fallbackMessage}`;
        r.solver = `${r.solver} (auto fallback from ${selected})`;
      }
      this.lpSolves++;
      this.totalIterations += r.iters ?? 0;
      this.totalSolverElapsedMs += r.elapsedMs;
      this.algorithmUsage[used] = (this.algorithmUsage[used] ?? 0) + 1;
      const fractional = r.status === 'optimal' ? listFractionals(r.x, this.p.integerVars, this.intTol) : [];
      const payload: RelaxationPayload = {
        node: tok.node,
        status: r.status,
        x: r.x,
        z: r.objective,
        solver: r.solver,
        selectedAlgorithm: used,
        iters: r.iters ?? 0,
        fractional,
      };
      transitionToken(tok, 'relaxed', {
        tick: this.tick,
        stationId: this.id,
        event: 'lp-relaxation-solved',
        detail: r.status,
      });
      const out = new RelaxationToken(payload, tok, {
        tokenId: `ip-relax-${tok.node.nodeId}-${this.lpSolves}`,
        tick: this.tick,
        stationId: this.id,
      });
      this.tokenRegistry.track(tok);
      this.tokenRegistry.track(out);
      this.emit(out, 'relaxed');
    }
    this.tick++;
  }
}

class RoundingRepairStation extends DESStation {
  candidatesTried = 0;
  private tick = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly intTol: number,
    private readonly heuristicPasses: number,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-rounding-repair');
  }

  runTimeStep(): void {
    for (const tok of this.drain<RelaxationToken>('relaxed')) {
      this.tokenRegistry.track(tok);
      const r = tok.payload;
      if (r.status !== 'optimal' || r.x.length === 0) continue;
      for (const cand of generateIntegerCandidates(this.p, r.x, this.intTol, this.heuristicPasses)) {
        this.candidatesTried++;
        const out = new CandidateToken({
          nodeId: r.node.nodeId,
          x: cand.x,
          z: objective(this.p, cand.x),
          source: cand.source,
        }, tok, {
          tokenId: `ip-candidate-${r.node.nodeId}-${this.candidatesTried}`,
          tick: this.tick,
          stationId: this.id,
        });
        this.tokenRegistry.track(out);
        this.emit(out, 'candidate');
      }
    }
    this.tick++;
  }
}

class IncumbentStation extends DESStation {
  bestX: number[] = [];
  bestZ: number;
  source?: string;
  updates = 0;
  candidatesSeen = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly intTol: number,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-incumbent');
    this.bestZ = p.sense === 'max' ? -Infinity : Infinity;
    this.addValidator(intrinsicCheck<IncumbentStation>({
      name: 'ip.incumbent-feasible',
      group: 'ip-mip-des-intrinsic',
      predicate: st => st.bestX.length === 0 || isIntegerFeasible(st.p, st.bestX, st.intTol),
      expected: 'incumbent satisfies Ax <= b, bounds, and integrality',
      observedFn: st => `z=${st.bestZ}, source=${st.source ?? 'none'}`,
    }));
  }

  hasIncumbent(): boolean { return this.bestX.length > 0; }

  isImprovement(z: number): boolean {
    return this.p.sense === 'max' ? z > this.bestZ + 1e-9 : z < this.bestZ - 1e-9;
  }

  runTimeStep(): void {
    for (const tok of this.drain<CandidateToken>('candidate')) {
      this.tokenRegistry.track(tok);
      const c = tok.payload;
      this.candidatesSeen++;
      if (!isIntegerFeasible(this.p, c.x, this.intTol)) continue;
      if (!this.isImprovement(c.z)) continue;
      this.bestX = c.x.slice();
      this.bestZ = c.z;
      this.source = `${c.source}@node${c.nodeId}`;
      this.updates++;
    }
  }
}

class CutGeneratorStation extends DESStation {
  cutsGenerated = 0;
  private tick = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly intTol: number,
    private readonly maxCutsPerNode: number,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-cut-generator');
  }

  runTimeStep(): void {
    for (const tok of this.drain<RelaxationToken>('relaxed')) {
      this.tokenRegistry.track(tok);
      const r = tok.payload;
      if (r.status !== 'optimal' || r.fractional.length === 0) continue;
      const cuts = generateBinaryCoverCuts(this.p, r.x, this.intTol, this.maxCutsPerNode, r.node);
      for (const cut of cuts) {
        this.cutsGenerated++;
        const out = new CutToken({nodeId: r.node.nodeId, cut}, tok, {
          tokenId: `ip-cut-${r.node.nodeId}-${this.cutsGenerated}`,
          tick: this.tick,
          stationId: this.id,
        });
        this.tokenRegistry.track(out);
        this.emit(out, 'cut');
      }
    }
    this.tick++;
  }
}

class NodeDecisionStation extends DESStation {
  readonly trace: IPMIPTraceEvent[] = [];
  private readonly cutsByNode = new Map<number, BranchOrCutConstraint[]>();
  sawUnbounded = false;
  private tick = 0;

  constructor(
    private readonly p: IPMIPProblem,
    private readonly controller: SearchControllerStation,
    private readonly incumbent: IncumbentStation,
    private readonly intTol: number,
    private readonly opts: Required<Pick<IPMIPSolveOptions, 'branchRule' | 'maxCutRounds' | 'verbose'>>,
    private readonly tokenRegistry: StatefulTokenRegistry,
  ) {
    super('ip-node-decision');
  }

  runTimeStep(): void {
    for (const tok of this.drain<CutToken>('cuts')) {
      this.tokenRegistry.track(tok);
      const arr = this.cutsByNode.get(tok.payload.nodeId) ?? [];
      arr.push(tok.payload.cut);
      this.cutsByNode.set(tok.payload.nodeId, arr);
    }

    for (const tok of this.drain<RelaxationToken>('relaxed')) {
      this.tokenRegistry.track(tok);
      this.decide(tok);
      const done = new CompleteToken(tok.payload.node.nodeId, {
        tokenId: `ip-complete-${tok.payload.node.nodeId}-${this.tick}`,
        tick: this.tick,
        stationId: this.id,
        parent: tok,
      });
      this.tokenRegistry.track(done);
      this.emit(done, 'complete');
    }
    this.tick++;
  }

  private decide(tok: RelaxationToken): void {
    const r = tok.payload;
    const node = r.node;
    if (r.status === 'infeasible') {
      this.record(tok, 'prune', 'LP infeasible');
      return;
    }
    if (r.status === 'unbounded') {
      this.sawUnbounded = true;
      this.record(tok, 'unbounded', 'LP relaxation unbounded');
      return;
    }
    if (r.status !== 'optimal') {
      this.record(tok, 'prune', r.status);
      return;
    }
    if (boundDominated(this.p, r.z, this.incumbent.bestZ, this.incumbent.hasIncumbent())) {
      this.record(tok, 'prune', 'bound dominated by incumbent');
      return;
    }
    if (r.fractional.length === 0 && isIntegerFeasible(this.p, r.x, this.intTol)) {
      const cand = new CandidateToken({nodeId: node.nodeId, x: r.x, z: r.z, source: 'lp-integer'}, tok, {
        tokenId: `ip-candidate-lp-${node.nodeId}-${this.tick}`,
        tick: this.tick,
        stationId: this.id,
      });
      this.tokenRegistry.track(cand);
      this.incumbent.take(cand, 'candidate');
      this.incumbent.runTimeStep();
      this.record(tok, 'incumbent', 'LP relaxation is integer-feasible');
      return;
    }

    const pendingCuts = this.cutsByNode.get(node.nodeId) ?? [];
    if (pendingCuts.length > 0 && node.cutRounds < this.opts.maxCutRounds) {
      const child: IPNode = {
        nodeId: this.controller.allocateNodeId(),
        parentId: node.nodeId,
        depth: node.depth,
        constraints: [...node.constraints, ...pendingCuts],
        cutRounds: node.cutRounds + 1,
        branchVar: node.branchVar,
        branchType: node.branchType,
        branchValue: node.branchValue,
        boundGuess: r.z,
      };
      const childTok = new NodeToken(child, {
        tokenId: `ip-node-${child.nodeId}`,
        tick: this.tick,
        stationId: this.id,
        parent: tok,
        event: 'cut-child-created',
        detail: `${pendingCuts.length} cuts`,
      });
      this.tokenRegistry.track(childTok);
      this.emit(childTok, 'nodes');
      this.record(tok, 'cut', `added ${pendingCuts.length} valid cut(s)`, undefined, [child.nodeId], pendingCuts.length);
      return;
    }

    const j = pickBranchVar(r.x, r.fractional, this.opts.branchRule);
    const xj = r.x[j];
    const lo = Math.floor(xj);
    const hi = Math.ceil(xj);
    const le = new Array(this.p.c.length).fill(0); le[j] = 1;
    const ge = new Array(this.p.c.length).fill(0); ge[j] = -1;
    const left: IPNode = {
      nodeId: this.controller.allocateNodeId(),
      parentId: node.nodeId,
      depth: node.depth + 1,
      constraints: [...node.constraints, {coefs: le, rhs: lo, name: `${varName(this.p, j)}<=${lo}`, kind: 'branch'}],
      cutRounds: 0,
      branchVar: j,
      branchType: 'le',
      branchValue: lo,
      boundGuess: r.z,
    };
    const right: IPNode = {
      nodeId: this.controller.allocateNodeId(),
      parentId: node.nodeId,
      depth: node.depth + 1,
      constraints: [...node.constraints, {coefs: ge, rhs: -hi, name: `${varName(this.p, j)}>=${hi}`, kind: 'branch'}],
      cutRounds: 0,
      branchVar: j,
      branchType: 'ge',
      branchValue: hi,
      boundGuess: r.z,
    };
    const leftTok = new NodeToken(left, {
      tokenId: `ip-node-${left.nodeId}`,
      tick: this.tick,
      stationId: this.id,
      parent: tok,
      event: 'branch-left-created',
      detail: `${varName(this.p, j)}<=${lo}`,
    });
    const rightTok = new NodeToken(right, {
      tokenId: `ip-node-${right.nodeId}`,
      tick: this.tick,
      stationId: this.id,
      parent: tok,
      event: 'branch-right-created',
      detail: `${varName(this.p, j)}>=${hi}`,
    });
    this.tokenRegistry.track(leftTok);
    this.tokenRegistry.track(rightTok);
    this.emit(leftTok, 'nodes');
    this.emit(rightTok, 'nodes');
    this.record(tok, 'branch', `branch on ${varName(this.p, j)}=${xj.toFixed(6)}`, j, [left.nodeId, right.nodeId]);
  }

  private record(
    tok: RelaxationToken,
    action: IPMIPTraceEvent['action'],
    reason?: string,
    branchVar?: number,
    children?: number[],
    cutsAdded?: number,
  ): void {
    const r = tok.payload;
    this.trace.push({
      nodeId: r.node.nodeId,
      parentId: r.node.parentId,
      depth: r.node.depth,
      lpStatus: r.status,
      lpZ: Number.isFinite(r.z) ? r.z : null,
      solver: r.solver,
      fractional: r.fractional.slice(0, 16),
      action,
      reason,
      branchVar,
      children,
      cutsAdded,
      nodeTokenId: tok.lineage.parentTokenId ?? tok.lineage.tokenId,
      lineageRoot: tok.lineage.rootTokenId,
      tokenGeneration: tok.lineage.generation,
      stateMode: tok.stateMode,
    });
    if (this.opts.verbose) {
      console.error(`node ${r.node.nodeId} d=${r.node.depth} z=${r.z} ${action}${reason ? ` (${reason})` : ''}`);
    }
  }
}

// -----------------------------------------------------------------------------
// Public solver
// -----------------------------------------------------------------------------

export class BranchAndCutSolverStation extends CompositeDESStation {
  readonly tokenRegistry = new StatefulTokenRegistry();
  readonly techniquePlan: IPMIPSolverTechniquePlan;
  readonly controller: SearchControllerStation;
  readonly lp: LPRelaxationStation;
  readonly heuristic: RoundingRepairStation;
  readonly incumbent: IncumbentStation;
  readonly cuts: CutGeneratorStation;
  readonly decision: NodeDecisionStation;

  constructor(
    id: string,
    private readonly p: IPMIPProblem,
    private readonly opts: FilledIPMIPSolveOptions,
  ) {
    super(id);
    this.techniquePlan = buildIPMIPSolverTechniquePlan(p, opts.lpAlgorithm, opts.allowExternalSolvers);
    this.controller = this.addSubstation(new SearchControllerStation(p, opts, this.tokenRegistry));
    this.lp = this.addSubstation(new LPRelaxationStation(
      p, opts.lpAlgorithm, this.techniquePlan, opts.lpMaxIters, opts.intTol, this.tokenRegistry));
    this.heuristic = this.addSubstation(new RoundingRepairStation(
      p, opts.intTol, opts.heuristicPasses, this.tokenRegistry));
    this.incumbent = this.addSubstation(new IncumbentStation(p, opts.intTol, this.tokenRegistry));
    this.cuts = this.addSubstation(new CutGeneratorStation(
      p, opts.intTol, opts.maxCutsPerNode, this.tokenRegistry));
    this.decision = this.addSubstation(new NodeDecisionStation(
      p, this.controller, this.incumbent, opts.intTol, opts, this.tokenRegistry));

    this.controller.pipe(this.lp, 'relax', 'nodes');
    this.lp.pipe(this.heuristic, 'relaxed', 'relaxed');
    this.lp.pipe(this.cuts, 'relaxed', 'relaxed');
    this.lp.pipe(this.decision, 'relaxed', 'relaxed');
    this.heuristic.pipe(this.incumbent, 'candidate', 'candidate');
    this.cuts.pipe(this.decision, 'cut', 'cuts');
    this.decision.pipe(this.controller, 'nodes', 'nodes');
    this.decision.pipe(this.controller, 'complete', 'complete');
  }

  bestBound(): number {
    return computeBestBound(this.p, this.incumbent, this.controller);
  }

  hasIncumbent(): boolean {
    return this.incumbent.hasIncumbent();
  }

  tokenStats(): SolverTokenStats {
    return this.tokenRegistry.snapshot();
  }

  topology(): SolverTopologyNode[] {
    return solverTopology(this.id);
  }
}

function fillIPMIPOptions(opts: IPMIPSolveOptions): FilledIPMIPSolveOptions {
  return {
    maxNodes: opts.maxNodes ?? 10000,
    maxTicks: opts.maxTicks ?? Math.max(100, (opts.maxNodes ?? 10000) * 8),
    timeLimitMs: opts.timeLimitMs ?? Infinity,
    lpMaxIters: opts.lpMaxIters ?? 2000,
    intTol: opts.intTol ?? 1e-6,
    branchRule: opts.branchRule ?? 'most-fractional',
    nodeSelection: opts.nodeSelection ?? 'dfs',
    lpAlgorithm: opts.lpAlgorithm ?? 'auto',
    allowExternalSolvers: opts.allowExternalSolvers ?? false,
    maxCutRounds: opts.maxCutRounds ?? 1,
    maxCutsPerNode: opts.maxCutsPerNode ?? 2,
    heuristicPasses: opts.heuristicPasses ?? 60,
    verbose: opts.verbose ?? false,
  };
}

export function solveIPMIPWithDES(p: IPMIPProblem, opts: IPMIPSolveOptions = {}): IPMIPSolution {
  validateIPMIPProblem(p);
  const filled = fillIPMIPOptions(opts);
  const t0 = Date.now();
  const solver = new BranchAndCutSolverStation('ip-branch-and-cut', p, filled);

  const summary = runIterativeDES(
    [solver],
    {
      shuffle: false,
      maxTicks: filled.maxTicks,
      stopWhen: () => Number.isFinite(filled.timeLimitMs) && Date.now() - t0 >= filled.timeLimitMs,
    },
  );
  assertNoValidationFailures(summary, MODEL);

  const hasInc = solver.hasIncumbent();
  const bestBound = solver.bestBound();
  const z = hasInc ? solver.incumbent.bestZ : (p.sense === 'max' ? -Infinity : Infinity);
  const optimal = summary.reason === 'done' && !solver.controller.hitNodeLimit() && hasInc;
  const status: IPMIPSolution['status'] =
    summary.reason === 'maxticks' ? 'tick-limit'
    : summary.reason === 'stop-when' ? 'time-limit'
    : solver.controller.hitNodeLimit() ? 'maxnodes'
    : optimal ? 'optimal'
    : !hasInc && solver.decision.sawUnbounded ? 'unbounded'
    : 'infeasible';
  const finalBound = optimal ? z : bestBound;
  const gap = !hasInc || !Number.isFinite(finalBound) ? Infinity
    : Math.abs(finalBound - z) / Math.max(1, Math.abs(z));
  const elapsedMs = Date.now() - t0;
  const tokenStats = solver.tokenStats();
  const usesExternalSolvers = didUseExternalLP(solver.lp.algorithmUsage);
  const performance = buildIPMIPPerformance({
    elapsedMs,
    ticks: summary.ticks,
    nodesExplored: solver.lp.lpSolves,
    lpSolves: solver.lp.lpSolves,
    totalLPIterations: solver.lp.totalIterations,
    totalLPSolverMs: solver.lp.totalSolverElapsedMs,
    cutsAdded: solver.cuts.cutsGenerated,
    candidatesTried: solver.heuristic.candidatesTried,
    tokensCreated: tokenStats.created,
  });

  return {
    status,
    x: hasInc ? solver.incumbent.bestX.slice() : [],
    z,
    bestBound: finalBound,
    gap: optimal ? 0 : gap,
    nodesExplored: solver.lp.lpSolves,
    lpSolves: solver.lp.lpSolves,
    totalLPIterations: solver.lp.totalIterations,
    cutsAdded: solver.cuts.cutsGenerated,
    candidatesTried: solver.heuristic.candidatesTried,
    lpAlgorithm: filled.lpAlgorithm,
    lpAlgorithmUsage: {...solver.lp.algorithmUsage},
    techniquePlan: solver.techniquePlan,
    incumbentSource: solver.incumbent.source,
    elapsedMs,
    inHouseOnly: !usesExternalSolvers,
    usesExternalSolvers,
    performance,
    solverKind: 'in-house-branch-and-cut',
    executionMode: 'single-threaded',
    compositeStationId: solver.id,
    tokenStats,
    trace: solver.decision.trace.slice(),
    topology: solver.topology(),
  };
}

// -----------------------------------------------------------------------------
// LP backend selection
// -----------------------------------------------------------------------------

function solveNodeRelaxation(
  p: IPMIPProblem,
  node: IPNode,
  algorithm: ConcreteLPRelaxationAlgorithm,
  lpMaxIters: number,
): LPSolution {
  if (algorithm === 'incremental-primal-dual') {
    return solveIncrementalRelaxation(p, node, lpMaxIters);
  }
  const lp = nodeToLPProblem(p, node);
  if (algorithm === 'internal-simplex') return solveLPInternal(lp, {maxIter: lpMaxIters});
  if (algorithm === 'des-simplex-dantzig') return solveLPViaDES(lp, {maxIter: lpMaxIters, pivotRule: 'dantzig'});
  if (algorithm === 'des-simplex-bland') return solveLPViaDES(lp, {maxIter: lpMaxIters, pivotRule: 'bland'});
  const method = algorithm === 'external-highs-ds' ? 'highs-ds'
    : algorithm === 'external-highs-ipm' ? 'highs-ipm'
    : 'highs';
  return solveLPExternal(lp, {method: method as any});
}

export function buildIPMIPSolverTechniquePlan(
  p: IPMIPProblem,
  requestedLPAlgorithm: LPRelaxationAlgorithm = 'auto',
  allowExternalSolvers: boolean = false,
): IPMIPSolverTechniquePlan {
  validateIPMIPProblem(p);
  const features = analyzeIPMIPProblem(p);
  const rationale: string[] = [];
  let rootLPAlgorithm: ConcreteLPRelaxationAlgorithm;
  const negativeRootRHS = hasNegativeRootRHS(p);

  if (requestedLPAlgorithm !== 'auto') {
    if (isExternalLPAlgorithm(requestedLPAlgorithm) && !allowExternalSolvers) {
      throw new Error(`${MODEL}: external LP backend "${requestedLPAlgorithm}" requested, but allowExternalSolvers is false`);
    }
    rootLPAlgorithm = requestedLPAlgorithm;
    rationale.push(`fixed LP relaxation backend requested: ${requestedLPAlgorithm}`);
    if (requestedLPAlgorithm === 'incremental-primal-dual' && negativeRootRHS) {
      rationale.push('warning: root has negative RHS rows; incremental LP requires a non-negative initial RHS');
    }
  } else if (negativeRootRHS) {
    rootLPAlgorithm = allowExternalSolvers && features.variableCount * features.constraintCount >= 2500
      ? 'external-highs'
      : 'internal-simplex';
    rationale.push(rootLPAlgorithm === 'external-highs'
      ? 'root relaxation has lower-bound rows with negative RHS, so auto uses an external Phase-1-capable LP backend'
      : 'root relaxation has lower-bound rows with negative RHS, so auto uses the in-house Phase-1 simplex backend');
  } else if (features.variableCount * features.constraintCount >= 2500) {
    rootLPAlgorithm = allowExternalSolvers
      ? features.density > 0.35 || features.variableCount > features.constraintCount * 3
      ? 'external-highs-ipm'
      : features.constraintCount > features.variableCount * 2
        ? 'external-highs-ds'
        : 'external-highs'
      : 'incremental-primal-dual';
    rationale.push(allowExternalSolvers
      ? `large relaxation (${features.variableCount} vars x ${features.constraintCount} rows) is an external-solver candidate`
      : `large relaxation (${features.variableCount} vars x ${features.constraintCount} rows) stays in-house because external solvers are disabled`);
  } else {
    rootLPAlgorithm = 'incremental-primal-dual';
    rationale.push('small/medium branch-cut relaxation uses in-engine incremental primal-dual simplex');
  }

  if (features.allBinary) {
    rationale.push('all integer variables are binary, so cover cuts and rounding/repair are active');
  } else if (features.continuousCount > 0) {
    rationale.push('mixed integer/continuous model keeps continuous variables in the LP relaxation');
  }

  const decompositionCandidate = features.constraintVariableComponents > 1;
  const decompositionReason = decompositionCandidate
    ? `constraint-variable graph has ${features.constraintVariableComponents} disconnected components`
    : undefined;
  if (decompositionCandidate) rationale.push(`${decompositionReason}; separable decomposition is structurally valid`);

  return {
    requestedLPAlgorithm,
    rootLPAlgorithm,
    externalSolversAllowed: allowExternalSolvers,
    usesExternalSolvers: isExternalLPAlgorithm(rootLPAlgorithm),
    externalCandidate: allowExternalSolvers && isExternalLPAlgorithm(rootLPAlgorithm),
    primalDualDynamic: rootLPAlgorithm === 'incremental-primal-dual' || requestedLPAlgorithm === 'auto',
    decompositionCandidate,
    decompositionReason,
    rationale,
    features,
  };
}

function selectLPRelaxationAlgorithm(
  p: IPMIPProblem,
  node: IPNode,
  requested: LPRelaxationAlgorithm,
  plan: IPMIPSolverTechniquePlan,
): ConcreteLPRelaxationAlgorithm {
  if (requested !== 'auto') return requested;
  if (hasNegativeRootRHS(p)) return plan.rootLPAlgorithm;
  if (!plan.externalSolversAllowed) return plan.rootLPAlgorithm;

  // Branching/cut nodes are related LPs with extra rows. The in-engine
  // IncrementalLP can restore feasibility via primal/dual simplex decisions.
  if (node.constraints.length > 0 || node.cutRounds > 0 || node.depth > 0) {
    return 'incremental-primal-dual';
  }

  const f = plan.features;
  if (f.variableCount * f.constraintCount >= 2500) return plan.rootLPAlgorithm;
  if (f.constraintCount > f.variableCount * 3 && f.constraintCount > 40) return 'external-highs-ds';
  if (f.variableCount > f.constraintCount * 4 && f.variableCount > 80) return 'external-highs-ipm';
  if (p.sense === 'min' && f.density > 0.5 && f.variableCount > 40) return 'external-highs';
  return plan.rootLPAlgorithm;
}

function isExternalLPAlgorithm(a: ConcreteLPRelaxationAlgorithm): boolean {
  return a === 'external-highs' || a === 'external-highs-ds' || a === 'external-highs-ipm';
}

function didUseExternalLP(usage: Partial<Record<ConcreteLPRelaxationAlgorithm, number>>): boolean {
  for (const key of Object.keys(usage) as ConcreteLPRelaxationAlgorithm[]) {
    if ((usage[key] ?? 0) > 0 && isExternalLPAlgorithm(key)) return true;
  }
  return false;
}

function buildIPMIPPerformance(opts: {
  elapsedMs: number;
  ticks: number;
  nodesExplored: number;
  lpSolves: number;
  totalLPIterations: number;
  totalLPSolverMs: number;
  cutsAdded: number;
  candidatesTried: number;
  tokensCreated: number;
}): IPMIPPerformanceStats {
  const seconds = Math.max(opts.elapsedMs / 1000, 1e-9);
  const nodeDenom = Math.max(opts.nodesExplored, 1);
  const lpDenom = Math.max(opts.lpSolves, 1);
  return {
    elapsedMs: opts.elapsedMs,
    ticks: opts.ticks,
    nodesPerSecond: opts.nodesExplored / seconds,
    lpSolvesPerSecond: opts.lpSolves / seconds,
    msPerNode: opts.elapsedMs / nodeDenom,
    totalLPSolverMs: opts.totalLPSolverMs,
    avgLPSolverMs: opts.totalLPSolverMs / lpDenom,
    lpSolverTimeShare: opts.elapsedMs > 0 ? opts.totalLPSolverMs / opts.elapsedMs : 0,
    avgLPIterationsPerSolve: opts.totalLPIterations / lpDenom,
    cutsPerNode: opts.cutsAdded / nodeDenom,
    candidatesPerNode: opts.candidatesTried / nodeDenom,
    tokensCreated: opts.tokensCreated,
  };
}

function hasNegativeRootRHS(p: IPMIPProblem): boolean {
  return p.b.some(v => v < -1e-9);
}

export function analyzeIPMIPProblem(p: IPMIPProblem): IPMIPProblemFeatures {
  const variableCount = p.c.length;
  const constraintCount = p.A.length;
  const integerCount = p.integerVars.filter(Boolean).length;
  const finiteUpperBounds = p.ub?.filter(v => Number.isFinite(v)).length ?? 0;
  let binaryCount = 0;
  for (let j = 0; j < variableCount; j++) {
    if (p.integerVars[j] && (p.ub?.[j] ?? Infinity) <= 1 + 1e-9) binaryCount++;
  }
  let nonzeros = 0;
  for (const row of p.A) for (const a of row) if (Math.abs(a) > 1e-12) nonzeros++;
  return {
    variableCount,
    constraintCount,
    integerCount,
    continuousCount: variableCount - integerCount,
    binaryCount,
    finiteUpperBounds,
    nonzeros,
    density: nonzeros / Math.max(1, variableCount * constraintCount),
    allInteger: integerCount === variableCount,
    allBinary: binaryCount === variableCount,
    constraintVariableComponents: countConstraintVariableComponents(p),
  };
}

function countConstraintVariableComponents(p: IPMIPProblem): number {
  const n = p.c.length;
  const m = p.A.length;
  const total = n + m;
  const adj: number[][] = Array.from({length: total}, () => []);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (Math.abs(p.A[i][j]) <= 1e-12) continue;
      const rowNode = n + i;
      adj[j].push(rowNode);
      adj[rowNode].push(j);
    }
  }
  const seen = new Array(total).fill(false);
  let components = 0;
  for (let k = 0; k < total; k++) {
    if (seen[k] || adj[k].length === 0) continue;
    components++;
    const stack = [k];
    seen[k] = true;
    while (stack.length > 0) {
      const u = stack.pop()!;
      for (const v of adj[u]) {
        if (seen[v]) continue;
        seen[v] = true;
        stack.push(v);
      }
    }
  }
  return components;
}

function solveIncrementalRelaxation(p: IPMIPProblem, node: IPNode, lpMaxIters: number): LPSolution {
  const t0 = Date.now();
  const root = rootIncrementalRows(p);
  const lp = new IncrementalLP({
    sense: p.sense,
    c: p.c,
    A: root.A,
    b: root.b,
    varNames: p.varNames,
    conNames: root.names,
  });
  for (const c of node.constraints) lp.applyAddConstraint(c.coefs, c.rhs, c.name);
  const trace = lp.solveToOptimum(lpMaxIters);
  const status = lp.status === 'optimal' ? 'optimal'
    : lp.status === 'infeasible' ? 'infeasible'
    : lp.status === 'unbounded' ? 'unbounded'
    : 'iter-limit';
  return {
    status,
    x: status === 'optimal' ? lp.getX() : [],
    objective: status === 'optimal' ? lp.getZ() : NaN,
    solver: 'incremental-primal-dual',
    elapsedMs: Date.now() - t0,
    iters: trace.filter(e => e.mode === 'primal' || e.mode === 'dual').length,
  };
}

function nodeToLPProblem(p: IPMIPProblem, node: IPNode): LPProblem {
  const A = p.A.map(r => r.slice());
  const b = p.b.slice();
  for (const c of node.constraints) {
    A.push(c.coefs.slice());
    b.push(c.rhs);
  }
  return {
    sense: p.sense,
    c: p.c.slice(),
    A_ub: A,
    b_ub: b,
    lb: new Array(p.c.length).fill(0),
    ub: p.ub ? p.ub.map(v => Number.isFinite(v) ? v : null) : undefined,
    varNames: p.varNames,
    conNames: p.conNames,
  };
}

function rootIncrementalRows(p: IPMIPProblem): {A: number[][]; b: number[]; names: string[]} {
  const A = p.A.map(r => r.slice());
  const b = p.b.slice();
  const names = p.conNames ? p.conNames.slice() : p.A.map((_, i) => `c${i}`);
  if (p.ub) {
    for (let j = 0; j < p.c.length; j++) {
      if (!Number.isFinite(p.ub[j])) continue;
      const row = new Array(p.c.length).fill(0); row[j] = 1;
      A.push(row); b.push(p.ub[j]); names.push(`ub_${varName(p, j)}`);
    }
  }
  return {A, b, names};
}

// -----------------------------------------------------------------------------
// Heuristics, cuts, and branching helpers
// -----------------------------------------------------------------------------

function generateIntegerCandidates(
  p: IPMIPProblem,
  xLP: readonly number[],
  tol: number,
  passes: number,
): Array<{x: number[]; source: string}> {
  const seeds: Array<{x: number[]; source: string}> = [];
  for (const mode of ['round', 'floor', 'ceil'] as const) {
    const x = xLP.slice();
    for (let j = 0; j < x.length; j++) {
      if (!p.integerVars[j]) continue;
      x[j] = mode === 'round' ? Math.round(x[j])
        : mode === 'floor' ? Math.floor(x[j])
        : Math.ceil(x[j]);
    }
    seeds.push({x: clampBounds(p, x), source: mode});
  }
  for (const j of listFractionals(xLP.slice(), p.integerVars, tol).slice(0, 4)) {
    for (const val of [Math.floor(xLP[j]), Math.ceil(xLP[j])]) {
      const x = xLP.slice();
      for (let k = 0; k < x.length; k++) if (p.integerVars[k]) x[k] = Math.round(x[k]);
      x[j] = val;
      seeds.push({x: clampBounds(p, x), source: `one-flip-${varName(p, j)}`});
    }
  }

  const out: Array<{x: number[]; source: string}> = [];
  const seen = new Set<string>();
  for (const s of seeds) {
    const repaired = repairAndImproveCandidate(p, s.x, tol, passes);
    if (!repaired) continue;
    const key = repaired.map(v => v.toFixed(9)).join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({x: repaired, source: `round-repair:${s.source}`});
  }
  return out;
}

function repairAndImproveCandidate(
  p: IPMIPProblem,
  x0: number[],
  tol: number,
  passes: number,
): number[] | null {
  let x = clampBounds(p, x0);
  for (let pass = 0; pass < passes && !satisfiesLinearRows(p, x, tol); pass++) {
    const baseViolation = totalViolation(p, x);
    let best: {j: number; dir: number; score: number} | null = null;
    for (let j = 0; j < x.length; j++) {
      if (!p.integerVars[j]) continue;
      for (const dir of [-1, 1]) {
        const y = x.slice();
        y[j] += dir;
        if (!boundsOk(p, y, tol)) continue;
        const newViolation = totalViolation(p, y);
        const reduction = baseViolation - newViolation;
        if (reduction <= 1e-12) continue;
        const objLoss = p.sense === 'max' ? -dir * p.c[j] : dir * p.c[j];
        const score = reduction / Math.max(1e-9, 1 + Math.max(0, objLoss));
        if (!best || score > best.score) best = {j, dir, score};
      }
    }
    if (!best) return null;
    x[best.j] += best.dir;
  }
  if (!isIntegerFeasible(p, x, tol)) return null;

  for (let pass = 0; pass < passes; pass++) {
    let improved = false;
    let bestX = x;
    let bestZ = objective(p, x);
    for (let j = 0; j < x.length; j++) {
      if (!p.integerVars[j]) continue;
      for (const dir of [-1, 1]) {
        const y = x.slice();
        y[j] += dir;
        if (!isIntegerFeasible(p, y, tol)) continue;
        const z = objective(p, y);
        if ((p.sense === 'max' && z > bestZ + 1e-9) || (p.sense === 'min' && z < bestZ - 1e-9)) {
          bestZ = z; bestX = y; improved = true;
        }
      }
    }
    x = bestX;
    if (!improved) break;
  }
  return x;
}

function generateBinaryCoverCuts(
  p: IPMIPProblem,
  x: readonly number[],
  tol: number,
  maxCuts: number,
  node: IPNode,
): BranchOrCutConstraint[] {
  const out: BranchOrCutConstraint[] = [];
  const existing = new Set(node.constraints.map(c => c.name));
  for (let r = 0; r < p.A.length && out.length < maxCuts; r++) {
    const row = p.A[r];
    if (row.some(a => a < -tol)) continue;
    const binary = row
      .map((a, j) => ({a, j, x: x[j]}))
      .filter(v => v.a > tol && p.integerVars[v.j] && (p.ub?.[v.j] ?? Infinity) <= 1 + tol)
      .sort((u, v) => v.x - u.x);
    if (binary.length < 2) continue;
    let sum = 0;
    const cover: number[] = [];
    for (const item of binary) {
      sum += item.a;
      cover.push(item.j);
      if (sum > p.b[r] + tol) break;
    }
    if (sum <= p.b[r] + tol || cover.length < 2) continue;
    const coefs = new Array(p.c.length).fill(0);
    for (const j of cover) coefs[j] = 1;
    const rhs = cover.length - 1;
    const lhs = cover.reduce((s, j) => s + x[j], 0);
    if (lhs <= rhs + 1e-7) continue;
    const name = `cover_r${r}_${cover.join('_')}`;
    if (existing.has(name)) continue;
    out.push({coefs, rhs, name, kind: 'cut'});
  }
  return out;
}

function listFractionals(x: readonly number[], integerVars: readonly boolean[], tol: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < x.length; j++) {
    if (!integerVars[j]) continue;
    const f = x[j] - Math.floor(x[j]);
    if (f > tol && f < 1 - tol) out.push(j);
  }
  return out;
}

function pickBranchVar(x: readonly number[], fractionals: readonly number[], rule: 'most-fractional' | 'first-fractional'): number {
  if (rule === 'first-fractional') return fractionals[0];
  let best = fractionals[0];
  let bestScore = -Infinity;
  for (const j of fractionals) {
    const f = x[j] - Math.floor(x[j]);
    const score = f * (1 - f);
    if (score > bestScore) { best = j; bestScore = score; }
  }
  return best;
}

// -----------------------------------------------------------------------------
// Validation and common math
// -----------------------------------------------------------------------------

export function validateIPMIPProblem(p: IPMIPProblem): void {
  Preconditions.check(MODEL, 'sense', 'be max or min', p.sense === 'max' || p.sense === 'min', p.sense);
  Preconditions.nonEmpty(MODEL, 'c', p.c);
  Preconditions.nonEmpty(MODEL, 'A', p.A);
  Preconditions.lengthEq(MODEL, 'b', p.b, p.A.length);
  Preconditions.lengthEq(MODEL, 'integerVars', p.integerVars, p.c.length);
  Preconditions.allFinite(MODEL, 'c', p.c);
  Preconditions.allFinite(MODEL, 'b', p.b);
  for (let i = 0; i < p.A.length; i++) {
    Preconditions.lengthEq(MODEL, `A[${i}]`, p.A[i], p.c.length);
    Preconditions.allFinite(MODEL, `A[${i}]`, p.A[i]);
  }
  if (p.ub) {
    Preconditions.lengthEq(MODEL, 'ub', p.ub, p.c.length);
    for (let j = 0; j < p.ub.length; j++) {
      if (!Number.isFinite(p.ub[j])) continue;
      Preconditions.nonNegative(MODEL, `ub[${j}]`, p.ub[j]);
    }
  }
  if (p.varNames) Preconditions.lengthEq(MODEL, 'varNames', p.varNames, p.c.length);
  if (p.conNames) Preconditions.lengthEq(MODEL, 'conNames', p.conNames, p.A.length);
}

function isIntegerFeasible(p: IPMIPProblem, x: readonly number[], tol: number): boolean {
  if (!boundsOk(p, x, tol) || !satisfiesLinearRows(p, x, tol)) return false;
  for (let j = 0; j < x.length; j++) {
    if (p.integerVars[j] && Math.abs(x[j] - Math.round(x[j])) > tol) return false;
  }
  return true;
}

function satisfiesLinearRows(p: IPMIPProblem, x: readonly number[], tol: number): boolean {
  for (let i = 0; i < p.A.length; i++) {
    let lhs = 0;
    for (let j = 0; j < x.length; j++) lhs += p.A[i][j] * x[j];
    if (lhs > p.b[i] + tol) return false;
  }
  return true;
}

function boundsOk(p: IPMIPProblem, x: readonly number[], tol: number): boolean {
  for (let j = 0; j < x.length; j++) {
    if (x[j] < -tol) return false;
    const ub = p.ub?.[j];
    if (ub !== undefined && Number.isFinite(ub) && x[j] > ub + tol) return false;
  }
  return true;
}

function totalViolation(p: IPMIPProblem, x: readonly number[]): number {
  let v = 0;
  for (let i = 0; i < p.A.length; i++) {
    let lhs = 0;
    for (let j = 0; j < x.length; j++) lhs += p.A[i][j] * x[j];
    v += Math.max(0, lhs - p.b[i]);
  }
  for (let j = 0; j < x.length; j++) {
    v += Math.max(0, -x[j]);
    const ub = p.ub?.[j];
    if (ub !== undefined && Number.isFinite(ub)) v += Math.max(0, x[j] - ub);
  }
  return v;
}

function clampBounds(p: IPMIPProblem, x: number[]): number[] {
  const y = x.slice();
  for (let j = 0; j < y.length; j++) {
    y[j] = Math.max(0, y[j]);
    const ub = p.ub?.[j];
    if (ub !== undefined && Number.isFinite(ub)) y[j] = Math.min(ub, y[j]);
  }
  return y;
}

function objective(p: IPMIPProblem, x: readonly number[]): number {
  let z = 0;
  for (let j = 0; j < p.c.length; j++) z += p.c[j] * x[j];
  return z;
}

function boundDominated(p: IPMIPProblem, bound: number, incumbent: number, hasIncumbent: boolean): boolean {
  if (!hasIncumbent) return false;
  return p.sense === 'max' ? bound <= incumbent + 1e-9 : bound >= incumbent - 1e-9;
}

function computeBestBound(p: IPMIPProblem, inc: IncumbentStation, ctrl: SearchControllerStation): number {
  const frontier = ctrl.bestFrontierBound();
  if (frontier !== null) return frontier;
  if (inc.hasIncumbent()) return inc.bestZ;
  return p.sense === 'max' ? Infinity : -Infinity;
}

function varName(p: IPMIPProblem, j: number): string {
  return p.varNames?.[j] ?? `x${j}`;
}

function solverTopology(parentId = 'ip-branch-and-cut'): SolverTopologyNode[] {
  return [
    {id: parentId, role: 'composite single-threaded in-house branch-and-cut solver', emits: []},
    {id: 'ip-search-controller', parentId, role: 'frontier of branch/cut subproblems', emits: ['node']},
    {id: 'ip-lp-relaxation', parentId, role: 'stationary LP solver block with selectable backend', emits: ['relaxation']},
    {id: 'ip-rounding-repair', parentId, role: 'movable-variable rounding, repair, and local search', emits: ['candidate']},
    {id: 'ip-incumbent', parentId, role: 'best feasible integer solution anchor', emits: []},
    {id: 'ip-cut-generator', parentId, role: 'valid-inequality station, currently binary cover cuts', emits: ['cut']},
    {id: 'ip-node-decision', parentId, role: 'prune, strengthen, or branch', emits: ['node', 'complete']},
  ];
}

// -----------------------------------------------------------------------------
// Convenience builders
// -----------------------------------------------------------------------------

export function buildBinaryKnapsackIP(values: number[], weights: number[], capacity: number): IPMIPProblem {
  Preconditions.lengthEq(MODEL, 'weights', weights, values.length);
  return {
    sense: 'max',
    c: values.slice(),
    A: [weights.slice()],
    b: [capacity],
    integerVars: new Array(values.length).fill(true),
    ub: new Array(values.length).fill(1),
    varNames: values.map((_, i) => `item_${i}`),
    conNames: ['capacity'],
    variableNodes: values.map((_, i) => ({varIndex: i, nodeId: `item_${i}`, label: `item ${i}`})),
    constraintNodes: [{rowIndex: 0, nodeId: 'capacity', label: 'capacity anchor'}],
  };
}

export function buildSmallMixedIP(): IPMIPProblem {
  return {
    sense: 'max',
    c: [1, 1, 1],
    A: [[1, 1, 0]],
    b: [3],
    integerVars: [true, true, false],
    ub: [10, 10, 10],
    varNames: ['x_int_a', 'x_int_b', 'y_cont'],
    conNames: ['integer_sum'],
  };
}
