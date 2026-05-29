'use strict';

// =============================================================================
// general/des-base/tree-search.ts — base class for TREE-STRUCTURED search
// algorithms: MILP branch-and-bound, MCTS / UCT, A*, beam search, alpha-beta,
// best-first / depth-first / breadth-first generic search.
//
// PROBLEM SHAPE
// ─────────────
//   Maintain a SET OF UNEXPLORED NODES (the "frontier") plus a BEST
//   FEASIBLE SOLUTION FOUND SO FAR (the "incumbent"). On each tick:
//
//     1. SELECT a node from the frontier (DFS pop / BFS shift / best-first
//        priority pop / UCT walk).
//     2. EVALUATE it (LP relaxation, rollout, heuristic, …).
//     3. If the eval improves the incumbent, UPDATE it.
//     4. If the eval is worse than the incumbent (FATHOMABLE) or otherwise
//        prunable, drop the subtree.
//     5. Otherwise EXPAND into children and push them onto the frontier.
//     6. Terminate when the frontier is empty OR a budget is exhausted
//        OR an external stop predicate fires.
//
//   The DIFFERENTIATORS among algorithms are:
//
//     - SELECT ordering   (DFS, BFS, best-first by bound, UCT)
//     - EVAL semantics    (LP solve, simulation rollout, heuristic h(n))
//     - PRUNE rule        (bound ≤ incumbent for MILP; nothing for plain DFS)
//     - EXPAND rule       (split on fractional var; one untried child for MCTS)
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep(): pickNext → evaluate → (updateIncumbent?) → (prune? skip)
//                  → (isLeaf? skip) → expand → push to frontier.
//
// HOOKS (abstract — subclasses MUST implement)
// ────────────────────────────────────────────
//   pickNext()          → next node to process or null when no work left
//   evaluate(node)      → {bound, isLeaf, value?, isFeasible?}
//   expand(node, ev)    → child nodes
//
// HOOKS (optional override)
// ─────────────────────────
//   shouldPrune(node, ev, incumbent) → boolean (default: ev.bound dominated)
//   pushChildren(children)           → default: this.frontier.push(...children)
//   onIncumbentUpdate / onPrune / onExpand for instrumentation
//
// Subclass guarantees `init()` (registers the root) before runIterativeDES
// is called; this base is single-station (one Station holds the whole
// search tree). For MCTS the "frontier" is implicit (the tree itself) and
// `pickNext` walks down via UCT — see general/mcts.ts for that leaf.
// =============================================================================

import {DESStation} from './station';

/** Direction of optimisation. */
export type SearchObjective = 'minimise' | 'maximise';

export interface NodeEvaluation {
  /** A bound on the OBJECTIVE achievable in this subtree. For minimise, this
   *  is a LOWER bound (relaxation); for maximise, an UPPER bound. */
  bound: number;
  /** True if no further branching can occur from this node (integer-feasible
   *  in MILP, terminal state in MCTS, leaf in alpha-beta, …). */
  isLeaf: boolean;
  /** Optional concrete objective value when the node is feasible
   *  (integer-feasible MILP solution; terminal state value). If undefined,
   *  the bound is used. */
  value?: number;
  /** True iff the node yields a complete feasible solution (the incumbent
   *  candidate). For MILP: integer-feasible at this leaf. For MCTS: any
   *  rollout terminating at a goal. */
  isFeasible?: boolean;
}

export abstract class TreeSearchStation<N> extends DESStation {
  protected readonly objective: SearchObjective;
  protected nodesProcessed = 0;
  protected nodesExpanded = 0;
  protected nodesPruned = 0;
  protected nodesFathomedByBound = 0;
  protected nodesIncumbentUpdates = 0;
  protected finished = false;

  /** Best feasible objective found so far. Initialised to ±Infinity in the
   *  worse-than-anything direction so any feasible value updates it. */
  protected incumbentValue: number;
  /** The node that produced the current incumbent. May be null. */
  protected incumbent: N | null = null;
  /** Optional cap on nodes processed. Infinity = no cap. */
  protected readonly maxNodes: number;

  /** Per-tick history of incumbentValue, plus best-bound history (a B&B
   *  duality-gap diagnostic). */
  readonly incumbentHistory: number[] = [];
  readonly bestBoundHistory: number[] = [];

  constructor(id: string, opts: {objective: SearchObjective; maxNodes?: number}) {
    super(id);
    this.objective = opts.objective;
    this.maxNodes = opts.maxNodes ?? Infinity;
    this.incumbentValue = opts.objective === 'minimise' ? Infinity : -Infinity;
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  /** Pick the next node to process, or null if the search is exhausted.
   *  Subclasses with an explicit frontier typically pop / shift / dequeue
   *  here; UCT-style algorithms walk the in-memory tree. */
  protected abstract pickNext(): N | null;

  /** Evaluate one node. Reports bound, leaf-ness, and (optionally) a
   *  concrete feasible value. */
  protected abstract evaluate(node: N): NodeEvaluation;

  /** Expand a non-leaf node into children. */
  protected abstract expand(node: N, ev: NodeEvaluation): N[];

  // ── HOOKS (optional override) ────────────────────────────────────────────

  /** Default prune rule: a non-leaf node with a bound dominated by the
   *  incumbent value can be discarded. Subclasses may add e.g. memoisation
   *  or tabu-list-based pruning. */
  protected shouldPrune(_node: N, ev: NodeEvaluation): boolean {
    return this.boundIsDominated(ev.bound);
  }

  /** Default: append children to a `frontier` array kept by the subclass.
   *  Override for priority-queue / best-first / UCT search. */
  protected abstract pushChildren(children: N[]): void;

  protected onIncumbentUpdate(_node: N, _value: number): void {}
  protected onPrune(_node: N, _ev: NodeEvaluation): void {}
  protected onExpand(_node: N, _children: readonly N[]): void {}
  protected onFinish(): void {}

  /** Best primal-side bound observed over all open frontier nodes. Default
   *  implementation falls back to ±∞; override when the frontier is a
   *  bounded priority queue and you can cheaply read the top. */
  protected currentBestBound(): number {
    return this.objective === 'minimise' ? -Infinity : Infinity;
  }

  // ── INTERNAL HELPERS ────────────────────────────────────────────────────

  /** Is `bound` worse than (or equal to, in the minimise/maximise sense) the
   *  incumbent? Equal-to defeats the bound (we have it already). */
  protected boundIsDominated(bound: number): boolean {
    if (this.objective === 'minimise') return bound >= this.incumbentValue - 1e-12;
    return bound <= this.incumbentValue + 1e-12;
  }

  /** Is `value` strictly better than the incumbent? */
  protected isImprovement(value: number): boolean {
    if (this.objective === 'minimise') return value < this.incumbentValue - 1e-12;
    return value > this.incumbentValue + 1e-12;
  }

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  /** Drives one node of the search. Subclasses MUST NOT override this. */
  runTimeStep(): void {
    if (this.finished) return;
    if (this.nodesProcessed >= this.maxNodes) {
      this.finished = true; this.onFinish(); return;
    }
    const node = this.pickNext();
    if (node === null) {
      this.finished = true; this.onFinish(); return;
    }
    this.nodesProcessed += 1;
    const ev = this.evaluate(node);
    if (ev.isFeasible && ev.value !== undefined && this.isImprovement(ev.value)) {
      this.incumbent = node;
      this.incumbentValue = ev.value;
      this.nodesIncumbentUpdates += 1;
      this.onIncumbentUpdate(node, ev.value);
    }
    if (this.shouldPrune(node, ev)) {
      this.nodesPruned += 1;
      if (this.boundIsDominated(ev.bound)) this.nodesFathomedByBound += 1;
      this.onPrune(node, ev);
    } else if (!ev.isLeaf) {
      const children = this.expand(node, ev);
      this.nodesExpanded += 1;
      this.onExpand(node, children);
      if (children.length > 0) this.pushChildren(children);
    }
    this.incumbentHistory.push(this.incumbentValue);
    this.bestBoundHistory.push(this.currentBestBound());
  }

  override hasWork(): boolean { return !this.finished; }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getIncumbent(): N | null { return this.incumbent; }
  getIncumbentValue(): number { return this.incumbentValue; }
  getNodesProcessed(): number { return this.nodesProcessed; }
  getNodesExpanded(): number { return this.nodesExpanded; }
  getNodesPruned(): number { return this.nodesPruned; }
  getNodesFathomedByBound(): number { return this.nodesFathomedByBound; }
  getNodesIncumbentUpdates(): number { return this.nodesIncumbentUpdates; }
  isFinished(): boolean { return this.finished; }
  /** 'minimise' | 'maximise' — exposed so external validators can decide
   *  which inequality direction to enforce. */
  getObjective(): SearchObjective { return this.objective; }
}
