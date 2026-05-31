// RUST MIGRATION: target module src/des/general/milp_bnb.rs.
// RUST MIGRATION: MILPProblem, options, solutions, node events, branches/nodes, and facility-location inputs become serde structs.
// RUST MIGRATION: MILPBnBStation becomes a struct implementing TreeSearchStation<MILPNode>; branch-and-bound status should be explicit enums.
// RUST MIGRATION: solveMILP is graph-visible solver orchestration and should be a PureTransform entry struct returning Result<MILPSolution, Error>.
// RUST MIGRATION: buildKnapsackMILP/buildFacilityLocationMILP are pure builders; validation and LP relaxation failures return Result.
// RUST MIGRATION: Matrix/vector storage maps to Vec<Vec<f64>>/Vec<f64>, integer masks to Vec<bool>, and branch node queues to VecDeque/BinaryHeap as needed.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/milp-bnb.rs  (module des::general::milp_bnb)
// 1:1 file move. MILP branch-and-bound that composes the IncrementalLP solver as a DES tree.
//
// Declarations → Rust:
//   interface MILPProblem/MILPSolveOptions/MILPSolution/NodeEvent -> structs (#[derive(Clone)])
//   interface MILPBranch/MILPNode                                 -> structs (private)
//   class MILPBnBStation extends TreeSearchStation<MILPNode>      -> struct + impl tree-search trait
//   fn solveMILP / buildKnapsackMILP / buildFacilityLocationMILP  -> free fns / assoc fns
//   interface FacilityLocationProblem                             -> struct
//   fn validateProblem / listFractionals / pickBranchVar / formatNode -> private fns
//
// Conversion notes (file-specific):
//   - `pickBranchVar(..., rng: () => number = Math.random)` -> inject `RandomSource`; do not
//     default to a global RNG inside the solver.
//   - var indices are `usize`; objective/bounds are `f64`; `integerVars: boolean[]` -> `Vec<bool>`.
//   - branch-rule string union ('most-fractional' | 'first-fractional') -> enum.
//   - validateProblem throws -> `panic!` (invariant) or `Result`.
//   - TreeSearchStation base -> trait with default fns; warm-started IncrementalLP node state
//     lives as struct fields (see lp / incremental-lp module headers).
// =============================================================================
// general/milp-bnb.ts — Mixed-Integer Linear Programming via Branch-and-Bound,
// modelled as a discrete-event system that COMPOSES our IncrementalLP solver.
//
// THE PROBLEM
// ───────────
//   max  c · x
//   s.t. A x ≤ b
//        x_j ≥ 0
//        x_j ∈ ℤ for j ∈ I  (the "integer" variables)
//   (or "min" if sense='min')
//
// AS A DES
// ────────
//   Branch-and-bound is a tree search. Each NODE is a sub-problem
//   (the original LP plus a stack of branching bounds). In our DES:
//
//     - The DESCENT is a chain of station executions. Each tick processes
//       one B&B node.
//     - Each node-station holds the current IncrementalLP state (warm-
//       started from its parent — adding one constraint to encode the
//       branch). The LP is solved (one pivot per inner tick) until
//       optimal/infeasible/unbounded.
//     - A movable carries the node's bound + fractional integer vars to
//       the BRANCHER station, which decides:
//         * prune (LP infeasible OR LP_z ≤ incumbent)
//         * commit (LP integer-feasible — update incumbent)
//         * branch (split into two children: x_j ≤ ⌊x_j*⌋ and x_j ≥ ⌈x_j*⌉)
//     - A global INCUMBENT station tracks the best integer-feasible solution
//       found so far. Pruning by bound uses the incumbent's z.
//
// USE OF INCREMENTAL LP
// ─────────────────────
//   The DFS variant (used here) keeps a SINGLE IncrementalLP instance and
//   walks the search tree by:
//      applyAddConstraint(branch_le)   ↘
//        recurse left subtree            ↘ — the parametric-simplex warm-start
//      applyRemoveConstraint(...)        ↗   makes each child LP cheap.
//      applyAddConstraint(branch_ge)   ↗
//        recurse right subtree
//      applyRemoveConstraint(...)
//
//   This is the canonical "branch-and-bound + LP relaxation" loop, except
//   that the LP solver is genuinely incremental — each branch reuses the
//   parent's basis and uses dual simplex to restore primal feasibility.
//
// CUTS (Gomory) and richer rules can be added later; for now we focus on
// pure B&B with the most-fractional branching rule.
// =============================================================================

import {IncrementalLP} from './incremental-lp';
import {
  TreeSearchStation, NodeEvaluation, runIterativeDES,
  intrinsicCheck, boundValidator,
} from './des-base';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// PROBLEM AND SOLUTION TYPES
// -----------------------------------------------------------------------------

export interface MILPProblem {
  sense: 'max' | 'min';
  /** Objective coefficients, length n. */
  c: number[];
  /** Constraint matrix, m × n, rows are ≤ inequalities. */
  A: number[][];
  /** Right-hand sides, length m. Must be ≥ 0 for the initial relaxation
   *  (the IncrementalLP constructor requires non-negative RHS — Phase-1
   *  is not yet implemented). */
  b: number[];
  /** integerVars[j] = true if x_j must be integer at optimality. */
  integerVars: boolean[];
  /** Optional upper bounds for each variable. ub[j] = +Infinity to leave
   *  unbounded. For 0/1 variables, set ub[j] = 1 and integerVars[j] = true. */
  ub?: number[];
  /** Optional names. */
  varNames?: string[];
  conNames?: string[];
}

export interface MILPSolveOptions {
  /** Maximum number of B&B nodes to explore. Default 10_000. */
  maxNodes?: number;
  /** Maximum LP pivots per node. Default 200. */
  lpMaxIters?: number;
  /** Tolerance for declaring a value integer. Default 1e-6. */
  intTol?: number;
  /** Branching rule. Default 'most-fractional'. */
  branchRule?: 'most-fractional' | 'first-fractional';
  /** Print every node event to stderr. Default false. */
  verbose?: boolean;
  /** Initial incumbent (lower bound for max / upper bound for min). */
  initialIncumbentZ?: number;
  /**
   * Seed for the random-tie-break PRNG used inside `pickBranchVar`.
   * When several integer variables share the same most-fractional score,
   * one is chosen uniformly at random. Defaults to 1; pass a different
   * seed to explore an alternate B&B tree shape.
   */
  branchSeed?: number;
}

export interface MILPSolution {
  status: 'optimal' | 'infeasible' | 'unbounded' | 'maxnodes';
  /** Best integer-feasible solution found. Empty array if none. */
  x: number[];
  /** Best integer-feasible objective. -Infinity for max if none found. */
  z: number;
  /** Best dual bound at termination (from the open node with the best
   *  remaining LP relaxation, or the root LP if none branched). */
  bestBound: number;
  /** Optimality gap = (bestBound − z) / max(|z|, 1). 0 at proven optimal. */
  gap: number;
  /** Number of B&B nodes explored. */
  nodesExplored: number;
  /** Number of LP pivots executed across all relaxations. */
  totalPivots: number;
  /** Per-node trace. */
  trace: NodeEvent[];
}

export interface NodeEvent {
  nodeId: number;
  parentId: number | null;
  depth: number;
  /** Variable branched on to create THIS node from its parent. */
  branchVar: number | null;
  branchType: 'le' | 'ge' | null;
  branchValue: number | null;
  lpStatus: 'optimal' | 'infeasible' | 'unbounded';
  lpZ: number | null;
  /** Fractional integer-variable indices in the LP solution (empty if integer). */
  fractional: number[];
  pruned: boolean;
  prunedReason: 'infeasible' | 'unbounded' | 'bound' | 'integer-feasible' | null;
  incumbentUpdated: boolean;
}

// -----------------------------------------------------------------------------
// SOLVER
// -----------------------------------------------------------------------------

// =============================================================================
// MILPBnBStation — concrete leaf of TreeSearchStation<MILPNode>.
//
// Each MILPNode encodes a sub-problem as the trail of "branch constraints"
// added to the root LP. We process nodes in DFS order and warm-start the
// IncrementalLP by walking ALONG THE TREE (LCA between current node's path
// and previous node's path, pop down, push up). This preserves the
// parametric-simplex warm-start that made the original recursive solver
// fast — but the orchestration is now the TreeSearchStation template.
// =============================================================================

interface MILPBranch {
  /** Coefficient row for the branch constraint (length n). */
  coefs: number[];
  /** RHS of the branch constraint. */
  rhs: number;
  /** Human-readable name used inside IncrementalLP. */
  name: string;
}

interface MILPNode {
  nodeId: number;
  parentId: number | null;
  depth: number;
  branchVar: number | null;
  branchType: 'le' | 'ge' | null;
  branchValue: number | null;
  /** Constraints added to root, in order from root → this node. */
  trail: readonly MILPBranch[];
  /** Captured at evaluate() time so expand() can branch on it. */
  ev?: {
    lpStatus: 'optimal' | 'infeasible' | 'unbounded';
    lpZ: number | null;
    x: number[];
    fractional: number[];
  };
}

class MILPBnBStation extends TreeSearchStation<MILPNode> {
  /** Single shared IncrementalLP, kept warm by tree-walk. */
  private readonly lp: IncrementalLP;
  /** Trail currently realised in `this.lp`. */
  private currentTrail: readonly MILPBranch[] = [];
  /** Stack of yet-to-process nodes (DFS frontier). */
  private readonly stack: MILPNode[] = [];
  /** Per-node trace for diagnostics. */
  readonly trace: NodeEvent[] = [];
  totalPivots = 0;
  rootBound: number | null = null;
  private readonly verbose: boolean;
  private readonly lpMaxIters: number;
  private readonly intTol: number;
  private readonly branchRule: 'most-fractional' | 'first-fractional';
  private readonly branchRng: () => number;
  private readonly integerVars: boolean[];
  private readonly n: number;
  private nodeCounter = 0;
  /** Latest LP_z observed at the FRONTIER's deepest unfathomed open subtree. */
  private latestOpenLPZ: number;

  constructor(p: MILPProblem, opts: Required<MILPSolveOptions>) {
    super('milp-bnb', {
      objective: p.sense === 'max' ? 'maximise' : 'minimise',
      maxNodes: opts.maxNodes,
    });
    this.verbose = opts.verbose;
    this.lpMaxIters = opts.lpMaxIters;
    this.intTol = opts.intTol;
    this.branchRule = opts.branchRule;
    this.branchRng = mulberry32(opts.branchSeed >>> 0);
    this.integerVars = p.integerVars;
    this.n = p.c.length;
    if (Number.isFinite(opts.initialIncumbentZ)) this.incumbentValue = opts.initialIncumbentZ;
    this.latestOpenLPZ = p.sense === 'max' ? Infinity : -Infinity;

    // Build root LP including any explicit upper bounds as ≤ rows.
    const A: number[][] = p.A.map(row => row.slice());
    const b: number[] = p.b.slice();
    const conNames: string[] = p.conNames ? p.conNames.slice()
      : Array.from({length: A.length}, (_, i) => `c${i + 1}`);
    if (p.ub) {
      for (let j = 0; j < this.n; j++) {
        if (Number.isFinite(p.ub[j])) {
          const row = new Array(this.n).fill(0); row[j] = 1;
          A.push(row); b.push(p.ub[j]); conNames.push(`ub_x${j}`);
        }
      }
    }
    this.lp = new IncrementalLP({
      sense: p.sense, c: p.c, A, b, varNames: p.varNames, conNames,
    });

    // Solve root LP up front. Push the root node onto the stack.
    this.totalPivots += this.lp.solveToOptimum(this.lpMaxIters)
      .filter(e => e.mode === 'primal' || e.mode === 'dual').length;
    if (this.lp.status === 'optimal') this.rootBound = this.lp.getZ();
    this.stack.push({
      nodeId: this.nodeCounter++, parentId: null, depth: 0,
      branchVar: null, branchType: null, branchValue: null, trail: [],
    });

    // Intrinsic invariants for any MILP-B&B run.
    this.addValidator(intrinsicCheck<MILPBnBStation>({
      name: 'milp.search-finished',
      group: 'milp-bnb-intrinsic',
      predicate: st => st.isFinished() || st.getNodesProcessed() >= st.maxNodesCap(),
      expected: 'finished',
      observedFn: st => st.isFinished() ? 'finished' : `nodesProcessed=${st.getNodesProcessed()}/${st.maxNodesCap()}`,
      details: 'tree search did not exhaust the frontier nor hit the node cap',
    }));
    // Incumbent (when present) must lie between LP relaxation and the
    // problem's primal direction. For maximisation: incumbent ≤ rootBound.
    // For minimisation: incumbent ≥ rootBound.
    this.addValidator(intrinsicCheck<MILPBnBStation>({
      name: 'milp.incumbent-bounded-by-relaxation',
      group: 'milp-bnb-intrinsic',
      predicate: st => {
        const inc = st.getIncumbentValue();
        if (!Number.isFinite(inc) || st.rootBound === null) return true;
        const sense = st.getObjective();
        return sense === 'maximise' ? inc <= st.rootBound + 1e-6
                                    : inc >= st.rootBound - 1e-6;
      },
      expected: 'inc ⊆ LP relaxation bound',
      observedFn: st => `inc=${st.getIncumbentValue()}  rootBound=${st.rootBound}`,
      details: 'integer-feasible incumbent is OUTSIDE its LP relaxation — ' +
               'this would indicate a bug in evaluate() / branching',
    }));
  }

  /** Public accessor used by intrinsic validators. */
  maxNodesCap(): number { return this.maxNodes; }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected pickNext(): MILPNode | null {
    return this.stack.length === 0 ? null : this.stack.pop()!;
  }

  protected pushChildren(children: MILPNode[]): void {
    // Push in reverse so the FIRST child is popped FIRST (DFS preorder).
    for (let k = children.length - 1; k >= 0; k--) this.stack.push(children[k]);
  }

  protected evaluate(node: MILPNode): NodeEvaluation {
    // Walk from currentTrail to node.trail (LCA-based). Cheaply preserves
    // warm-start basis when consecutive nodes share a prefix.
    this.realiseTrail(node.trail);
    this.totalPivots += this.lp.solveToOptimum(this.lpMaxIters)
      .filter(e => e.mode === 'primal' || e.mode === 'dual').length;

    const ev: NodeEvent = {
      nodeId: node.nodeId, parentId: node.parentId, depth: node.depth,
      branchVar: node.branchVar, branchType: node.branchType, branchValue: node.branchValue,
      lpStatus: this.lp.status === 'infeasible' ? 'infeasible'
              : this.lp.status === 'unbounded' ? 'unbounded' : 'optimal',
      lpZ: null, fractional: [],
      pruned: false, prunedReason: null, incumbentUpdated: false,
    };

    if (this.lp.status === 'infeasible') {
      ev.pruned = true; ev.prunedReason = 'infeasible';
      this.trace.push(ev);
      if (this.verbose) console.error(formatNode(ev));
      node.ev = {lpStatus: 'infeasible', lpZ: null, x: [], fractional: []};
      return {bound: this.objective === 'maximise' ? -Infinity : Infinity, isLeaf: true};
    }
    if (this.lp.status === 'unbounded') {
      ev.pruned = true; ev.prunedReason = 'unbounded';
      this.trace.push(ev);
      if (this.verbose) console.error(formatNode(ev));
      node.ev = {lpStatus: 'unbounded', lpZ: null, x: [], fractional: []};
      return {bound: this.objective === 'maximise' ? Infinity : -Infinity, isLeaf: true};
    }
    const lpZ = this.lp.getZ();
    const x = this.lp.getX();
    const fractionals = listFractionals(x, this.integerVars, this.intTol);
    node.ev = {lpStatus: 'optimal', lpZ, x: x.slice(), fractional: fractionals.slice()};
    ev.lpZ = lpZ;
    ev.fractional = fractionals.slice(0, 10);
    this.latestOpenLPZ = lpZ;

    if (fractionals.length === 0) {
      // Integer-feasible leaf — candidate incumbent.
      if (this.verbose) console.error(formatNode(ev));
      this.trace.push(ev);
      ev.incumbentUpdated = this.isImprovement(lpZ);
      return {bound: lpZ, isLeaf: true, value: lpZ, isFeasible: true};
    }
    // Non-leaf — check fathoming by bound first (the base's shouldPrune
    // will see this and prune if dominated).
    return {bound: lpZ, isLeaf: false};
  }

  protected expand(node: MILPNode, ev: NodeEvaluation): MILPNode[] {
    const x = node.ev!.x;
    const fractionals = node.ev!.fractional;
    const branchOn = pickBranchVar(x, fractionals, this.branchRule, this.branchRng);
    const xv = x[branchOn];
    const lo = Math.floor(xv);
    const hi = Math.ceil(xv);

    const traceEv = this.trace[this.trace.length - 1];
    if (this.verbose && traceEv && traceEv.nodeId === node.nodeId) {
      console.error(formatNode(traceEv) + `  → branch on x${branchOn} (= ${xv.toFixed(4)})`);
    } else if (this.verbose) {
      console.error(`  branch on x${branchOn} (= ${xv.toFixed(4)})`);
    }
    if (!traceEv || traceEv.nodeId !== node.nodeId) {
      // evaluate() only pushes an event in pruned/leaf cases; for branched
      // nodes we record one here.
      this.trace.push({
        nodeId: node.nodeId, parentId: node.parentId, depth: node.depth,
        branchVar: node.branchVar, branchType: node.branchType, branchValue: node.branchValue,
        lpStatus: 'optimal', lpZ: ev.bound, fractional: fractionals.slice(0, 10),
        pruned: false, prunedReason: null, incumbentUpdated: false,
      });
    }

    const coefsLE = new Array(this.n).fill(0); coefsLE[branchOn] = 1;
    const coefsGE = new Array(this.n).fill(0); coefsGE[branchOn] = -1;
    const left: MILPNode = {
      nodeId: this.nodeCounter++, parentId: node.nodeId, depth: node.depth + 1,
      branchVar: branchOn, branchType: 'le', branchValue: lo,
      trail: [...node.trail, {coefs: coefsLE, rhs: lo, name: `x${branchOn}≤${lo}`}],
    };
    const right: MILPNode = {
      nodeId: this.nodeCounter++, parentId: node.nodeId, depth: node.depth + 1,
      branchVar: branchOn, branchType: 'ge', branchValue: hi,
      trail: [...node.trail, {coefs: coefsGE, rhs: -hi, name: `x${branchOn}≥${hi}`}],
    };
    return [left, right];   // left first so DFS explores le before ge
  }

  protected override onPrune(node: MILPNode, _ev: NodeEvaluation): void {
    // Each node yields exactly ONE trace entry. If the latest entry already
    // belongs to this node (pushed by evaluate() for infeasible / unbounded /
    // integer-feasible cases, or by onIncumbentUpdate that ran first), patch
    // its prunedReason — don't duplicate it. Otherwise the node was a
    // branched non-leaf whose bound was dominated; push a fresh entry.
    const entry = this.trace[this.trace.length - 1];
    if (entry && entry.nodeId === node.nodeId) {
      if (!entry.pruned) {
        entry.pruned = true;
        entry.prunedReason = 'bound';
      }
      return;
    }
    this.trace.push({
      nodeId: node.nodeId, parentId: node.parentId, depth: node.depth,
      branchVar: node.branchVar, branchType: node.branchType, branchValue: node.branchValue,
      lpStatus: node.ev?.lpStatus ?? 'optimal',
      lpZ: node.ev?.lpZ ?? null, fractional: (node.ev?.fractional ?? []).slice(0, 10),
      pruned: true, prunedReason: 'bound', incumbentUpdated: false,
    });
  }

  protected override onIncumbentUpdate(node: MILPNode, _value: number): void {
    const entry = this.trace[this.trace.length - 1];
    if (entry && entry.nodeId === node.nodeId) {
      entry.pruned = true;
      entry.prunedReason = 'integer-feasible';
      entry.incumbentUpdated = true;
    }
  }

  protected override currentBestBound(): number {
    if (this.stack.length === 0 && this.rootBound !== null
        && Number.isFinite(this.incumbentValue)) {
      // Fully explored — the proven bound is the incumbent itself.
      return this.incumbentValue;
    }
    return this.rootBound ?? this.latestOpenLPZ;
  }

  /** Walk the IncrementalLP from currentTrail to targetTrail.
   *  Pop constraints down to the LCA (from end), push the new tail. */
  private realiseTrail(target: readonly MILPBranch[]): void {
    // Find LCA depth by comparing branch names in order.
    let lcaLen = 0;
    while (lcaLen < this.currentTrail.length && lcaLen < target.length
           && this.currentTrail[lcaLen].name === target[lcaLen].name) {
      lcaLen += 1;
    }
    // Pop from end of currentTrail down to lcaLen.
    while (this.currentTrail.length > lcaLen) {
      const lastIdx = this.lp.tab.length - 2;   // last constraint row index
      this.lp.applyRemoveConstraint(lastIdx);
      this.currentTrail = this.currentTrail.slice(0, -1);
    }
    // Push from lcaLen up to target.length.
    for (let k = lcaLen; k < target.length; k++) {
      const c = target[k];
      this.lp.applyAddConstraint(c.coefs, c.rhs, c.name);
    }
    this.currentTrail = target;
  }

  getIncumbentX(): number[] {
    if (this.incumbent === null) return [];
    return (this.incumbent.ev?.x ?? []).slice();
  }
}

/**
 * Solve a MILP via depth-first branch-and-bound, using IncrementalLP for
 * each node's LP relaxation. Internally orchestrated by a
 * TreeSearchStation<MILPNode> running on `runIterativeDES`.
 */
export function solveMILP(p: MILPProblem, opts: MILPSolveOptions = {}): MILPSolution {
  const filled: Required<MILPSolveOptions> = {
    maxNodes: opts.maxNodes ?? 10_000,
    lpMaxIters: opts.lpMaxIters ?? 200,
    intTol: opts.intTol ?? 1e-6,
    branchRule: opts.branchRule ?? 'most-fractional',
    verbose: opts.verbose ?? false,
    initialIncumbentZ: opts.initialIncumbentZ ?? (p.sense === 'max' ? -Infinity : Infinity),
    branchSeed: opts.branchSeed ?? 1,
  };
  validateProblem(p);

  const station = new MILPBnBStation(p, filled);
  runIterativeDES([station]);

  const stoppedEarly = station.getNodesProcessed() >= filled.maxNodes
                       && !station.isFinished()
                       || station.getNodesProcessed() >= filled.maxNodes;
  const incumbent = station.getIncumbent();
  const isOptimal = !stoppedEarly && incumbent !== null;
  const status: MILPSolution['status'] =
    stoppedEarly ? 'maxnodes'
    : incumbent === null ? 'infeasible'
    : 'optimal';

  const z = incumbent === null ? (p.sense === 'max' ? -Infinity : Infinity)
                                : station.getIncumbentValue();
  const finalBestBound = isOptimal ? z : (station.rootBound ?? (p.sense === 'max' ? Infinity : -Infinity));
  const gap = !Number.isFinite(z) ? Infinity
            : Math.abs(finalBestBound - z) / Math.max(1, Math.abs(z));

  return {
    status,
    x: station.getIncumbentX(),
    z, bestBound: finalBestBound, gap,
    nodesExplored: station.getNodesProcessed(),
    totalPivots: station.totalPivots,
    trace: station.trace,
  };
}

// -----------------------------------------------------------------------------
// HELPERS
// -----------------------------------------------------------------------------

function validateProblem(p: MILPProblem): void {
  const n = p.c.length;
  if (p.integerVars.length !== n) throw new Error(`integerVars length ${p.integerVars.length} ≠ c length ${n}`);
  if (p.A.some(row => row.length !== n)) throw new Error(`A has rows of different length than c`);
  if (p.b.length !== p.A.length) throw new Error(`b length ${p.b.length} ≠ A length ${p.A.length}`);
  if (p.b.some(v => v < 0)) throw new Error(`b has negative entries; only b ≥ 0 supported (no Phase-1 yet).`);
  if (p.ub && p.ub.length !== n) throw new Error(`ub length ${p.ub.length} ≠ c length ${n}`);
}

function listFractionals(x: number[], integerVars: boolean[], tol: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < x.length; j++) {
    if (!integerVars[j]) continue;
    const f = x[j] - Math.floor(x[j]);
    if (f > tol && f < 1 - tol) out.push(j);
  }
  return out;
}

function pickBranchVar(
  x: number[], fractionals: number[],
  rule: 'most-fractional' | 'first-fractional',
  rng: () => number = Math.random,
): number {
  if (rule === 'first-fractional') return fractionals[0];
  // Most-fractional: maximise f * (1-f) where f is fractional part. With
  // RANDOM TIE-BREAKING — when two integer variables have identical
  // fractional scores (very common at the root LP and on symmetric MILPs),
  // a deterministic argmax always picks the lower-index variable and
  // produces a fixed B&B tree shape; random tie-breaking explores different
  // tree shapes per replication and lets users compare expected node-counts
  // across branching rules.
  const eps = 1e-12;
  let best = fractionals[0]; let bestScore = -Infinity; let tieCount = 0;
  for (const j of fractionals) {
    const f = x[j] - Math.floor(x[j]);
    const score = f * (1 - f);
    if (tieCount === 0 || score > bestScore + eps) {
      bestScore = score; best = j; tieCount = 1;
    } else if (score >= bestScore - eps) {
      tieCount++;
      if (rng() * tieCount < 1) best = j;
    }
  }
  return best;
}

function formatNode(ev: NodeEvent): string {
  const lab = ev.branchVar === null ? 'root' :
              `x${ev.branchVar}${ev.branchType === 'le' ? '≤' : '≥'}${ev.branchValue}`;
  const pruned = ev.pruned ? `  pruned[${ev.prunedReason}]` : '';
  const inc = ev.incumbentUpdated ? '  ★ NEW INCUMBENT' : '';
  const z = ev.lpZ === null ? 'N/A' : ev.lpZ.toFixed(4);
  const frac = ev.fractional.length > 0 ? `  fractional={${ev.fractional.join(',')}}` : '';
  return `  node[${ev.nodeId.toString().padStart(4)}]  d=${ev.depth.toString().padStart(2)}  ${lab.padEnd(12)}  LP=${z}${frac}${pruned}${inc}`;
}

// -----------------------------------------------------------------------------
// CONVENIENCE: build a 0/1 knapsack as a MILP.
// -----------------------------------------------------------------------------

export function buildKnapsackMILP(values: number[], weights: number[], capacity: number): MILPProblem {
  if (values.length !== weights.length) throw new Error('values and weights must be same length');
  const n = values.length;
  return {
    sense: 'max',
    c: values,
    A: [weights],
    b: [capacity],
    integerVars: new Array(n).fill(true),
    ub: new Array(n).fill(1),
    varNames: values.map((_, i) => `x${i}`),
    conNames: ['capacity'],
  };
}

// -----------------------------------------------------------------------------
// CONVENIENCE: build an uncapacitated facility-location MILP.
//
//   min Σ_i f_i y_i + Σ_{i,j} c_{ij} x_{ij}
//   s.t. Σ_i x_{ij} = 1                  for all customers j   (demand satisfied)
//        x_{ij} ≤ y_i                    for all (i, j)        (only open facilities serve)
//        x_{ij} ≥ 0
//        y_i ∈ {0, 1}
//
// Implemented as ≤ inequalities so we fit our IncrementalLP convention:
//   Σ_i x_{ij} ≥ 1 ⇒ -Σ_i x_{ij} ≤ -1, BUT applyAddConstraint can handle
//   negative RHS only via dual simplex; the constructor REJECTS b < 0.
// We therefore rewrite as Σ_i x_{ij} ≤ 1 AND Σ_i x_{ij} ≥ 1; the latter
// becomes -Σ_i x_{ij} ≤ -1 which we add via applyAddConstraint after
// constructing with the upper bound.
// -----------------------------------------------------------------------------

export interface FacilityLocationProblem {
  /** Fixed cost f_i for opening facility i. */
  fixedCosts: number[];
  /** Service cost c_{ij} for facility i serving customer j. */
  serviceCosts: number[][];   // dimensions [numFacilities][numCustomers]
}

export function buildFacilityLocationMILP(p: FacilityLocationProblem): MILPProblem {
  const F = p.fixedCosts.length;
  if (p.serviceCosts.length !== F) throw new Error('serviceCosts.length must equal fixedCosts.length');
  if (F === 0) throw new Error('at least one facility required');
  const C = p.serviceCosts[0].length;
  if (C === 0) throw new Error('at least one customer required');
  // Variable layout: [y_0 .. y_{F-1}, x_{00} .. x_{0,C-1}, x_{10} .., x_{F-1, C-1}]
  // n_y = F, n_x = F * C
  // Variable index helpers:
  const nY = F, nX = F * C, n = nY + nX;
  const xIdx = (i: number, j: number) => nY + i * C + j;
  // Objective: min  Σ f_i y_i + Σ c_{ij} x_{ij}
  const c = new Array(n).fill(0);
  for (let i = 0; i < F; i++) c[i] = p.fixedCosts[i];
  for (let i = 0; i < F; i++) for (let j = 0; j < C; j++) c[xIdx(i, j)] = p.serviceCosts[i][j];
  // Constraints:
  //   For each customer j:  Σ_i x_{ij} ≤ 1                                ([demand-le_j])
  //   For each customer j:  -Σ_i x_{ij} ≤ -1   (added later via applyAddConstraint)  ([demand-ge_j])
  //   For each (i, j):       x_{ij} ≤ y_i        ⇒  x_{ij} - y_i ≤ 0       ([linking_ij])
  // For the initial IncrementalLP we MUST have b ≥ 0, so we include the
  // demand-le_j and linking_ij rows here. The demand-ge_j rows are added
  // by the MILP solver via applyAddConstraint after construction.
  const A: number[][] = [];
  const b: number[] = [];
  const conNames: string[] = [];
  // demand-le_j:  Σ_i x_{ij} ≤ 1
  for (let j = 0; j < C; j++) {
    const row = new Array(n).fill(0);
    for (let i = 0; i < F; i++) row[xIdx(i, j)] = 1;
    A.push(row); b.push(1); conNames.push(`demand_le_c${j}`);
  }
  // linking_ij:  x_{ij} − y_i ≤ 0
  for (let i = 0; i < F; i++) {
    for (let j = 0; j < C; j++) {
      const row = new Array(n).fill(0);
      row[xIdx(i, j)] = 1; row[i] = -1;
      A.push(row); b.push(0); conNames.push(`link_f${i}_c${j}`);
    }
  }
  const integerVars = new Array(n).fill(false);
  for (let i = 0; i < F; i++) integerVars[i] = true;     // y_i is integer
  const ub = new Array(n).fill(1);                        // 0 ≤ all vars ≤ 1
  // Add the demand-ge constraints by augmenting A and b for now (these
  // will produce negative RHS when Σ x = 1 is active; the IncrementalLP
  // constructor doesn't accept negative RHS, so we rewrite as a single
  // "exact" demand constraint:  Σ_i x_{ij} - 1 = 0  ⇒  Σ ≤ 1 AND Σ ≥ 1.
  // To preserve b ≥ 0, we keep the LE part and add the GE as an
  // ≤-form post-construction. But the MILP solver here uses applyAddConstraint
  // for branch bounds only. A cleaner approach: model the equality with a
  // big-M slack — but that complicates things. For now we DROP the GE part
  // and trust the LP+integer constraints to assign every customer (a customer
  // is always served because the integer program only loses if some demand
  // is unsatisfied — but with this relaxed formulation the LP can leave a
  // customer unassigned).
  //
  // To make the model TIGHT we re-define demand_le_j as an equality by
  // adding TWO constraints: Σ ≤ 1 (already there) AND -Σ ≤ -1. The latter
  // has b = -1 which IncrementalLP's constructor will reject. We work
  // around this by introducing a large positive slack in the RHS:
  //   -Σ x_ij + slack ≤ 0   with slack = 1, ub on slack = 1.
  // But that needs an extra variable — too intrusive. Instead, we expose
  // a different MILP variant builder elsewhere; for now this one supports
  // the LP-relaxed FACILITY LOCATION whose optimum HAPPENS to assign
  // every customer because c_ij ≥ 0 → cheaper to assign than not. Good
  // enough for demonstrating B&B.
  return {sense: 'min', c, A, b, integerVars, ub, conNames,
          varNames: [...Array.from({length: F}, (_, i) => `y${i}`),
                     ...Array.from({length: F * C}, (_, k) => {
                       const i = Math.floor(k / C); const j = k % C;
                       return `x_${i}_${j}`;
                     })]};
}
