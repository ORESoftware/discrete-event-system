// RUST MIGRATION: target module src/des/general/mcts.rs.
// RUST MIGRATION: MCTSEnv<S> is behavior and should become a trait with associated State/Action types or explicit generic bounds.
// RUST MIGRATION: MCTSOptions and private Node<S> become structs; MCTSStation<S> becomes a struct implementing TreeSearchStation<Node<S>> behavior through traits.
// RUST MIGRATION: mcts<S> is a generic solver helper and can remain a free function; wrap it in PureTransform only for registered DES graph use.
// RUST MIGRATION: Tree nodes need explicit ownership indexes or Rc/RefCell alternatives; prefer arena Vec<Node> plus node indexes for Rust friendliness.
// RUST MIGRATION: Rollouts must use injected rand::Rng and invalid terminal/action states should return Result or domain-specific status.
'use strict';

// =============================================================================
// Generic Monte Carlo Tree Search (UCT) with pluggable rollout policy.
//
// Designed to plug directly into a DES simulator: any DES whose state can
// be cloned and advanced one decision-epoch via `applyAction(state, a)` is
// a valid `MCTSEnv`. The DES becomes the search environment, and the
// rollout (default policy) is just any deterministic / randomized policy
// that completes a trajectory from a leaf.
//
// Useful when:
//   - the decision space is small (we enumerate actions per node)
//   - the horizon is long but the state space is too large to tabulate
//   - the DES is fast enough to run many rollouts per decision epoch
//
// Algorithm: UCT with one expansion per iteration, ε-decoupled rollout
// from the leaf. Each leaf rollout returns a scalar discounted return
// to the leaf. Backups update visit counts and average returns up the
// tree. Action selection at decision time: visit-count-weighted greedy
// (most-visited child = robust choice) by default, or value-greedy.
//
// References:
//   Kocsis & Szepesvári 2006 — Bandit-based Monte Carlo Planning (UCT)
//   Browne et al. 2012 — A Survey of Monte Carlo Tree Search Methods
//
// As a DES: orchestrated by MCTSStation (a leaf of TreeSearchStation<Node>).
// Each runTimeStep() = one UCT iteration (select → expand → simulate →
// backup). The "frontier" is implicit — pickNext walks down via UCT — so
// the base's default pushChildren is unused; we register children directly
// in the in-memory tree.
// =============================================================================

import {TreeSearchStation, NodeEvaluation, runIterativeDES, ARGMAX_EPS_DEFAULT} from './des-base';

export interface MCTSEnv<S> {
  /** Number of legal actions in state `s`. Constant action sets are simplest. */
  numActions: (s: S) => number;
  /** Apply action `a` in state `s`; returns the next state and the immediate
   *  reward received on that transition. The state object MUST be a fresh
   *  clone (or treated as immutable after this call) so the search tree
   *  doesn't share aliased mutable state across siblings. */
  applyAction: (s: S, a: number) => {next: S; reward: number; done: boolean};
  /** Optional: terminal predicate. Default: never terminal. */
  isTerminal?: (s: S) => boolean;
  /** Default rollout policy. If not provided, uniform random over legal
   *  actions. For DES-driven rollouts the natural choice is a fast
   *  heuristic (Shortest-Queue, Shortest-Expected-Completion, etc.). */
  rolloutPolicy?: (s: S) => number;
  /** Number of decision epochs in the rollout from a leaf before we cut off. */
  rolloutDepth?: number;
  /** Discount factor applied to rewards along the rollout. Default 1.0. */
  gamma?: number;
}

export interface MCTSOptions {
  /** Number of UCT iterations per decision call. Default 200. */
  iterations?: number;
  /** Exploration constant (Cp in the UCT formula). Default sqrt(2). */
  c?: number;
  /** Final action-selection rule. */
  selection?: 'visits' | 'value';
  /** PRNG returning [0, 1). Default Math.random. */
  rng?: () => number;
}

interface Node<S> {
  state: S;
  parent: Node<S> | null;
  /** Action that was taken in `parent.state` to reach this node. */
  fromAction: number;
  /** Reward received on the parent → this transition. */
  rewardIn: number;
  visits: number;
  /** Sum of returns observed below this node (averaged via /visits). */
  totalReturn: number;
  children: Map<number, Node<S>>;
  /** Untried action indices. Once empty, every action has a child. */
  untried: number[];
  done: boolean;
}

function makeNode<S>(state: S, parent: Node<S> | null, fromAction: number,
                     rewardIn: number, numActions: number, done: boolean): Node<S> {
  return {
    state, parent, fromAction, rewardIn,
    visits: 0, totalReturn: 0,
    children: new Map(),
    untried: Array.from({length: numActions}, (_, i) => i),
    done,
  };
}

/**
 * Run UCT for `iterations` steps starting from `rootState`, then return
 * the action recommended at the root.
 *
 * Each iteration consists of four phases:
 *   1. Selection: walk down the tree picking children that maximise
 *        UCT(s, a) = mean(s, a) + c · √(ln N(s) / n(s, a))
 *      until we find a node with untried actions or a terminal node.
 *   2. Expansion: instantiate one untried child (one action).
 *   3. Simulation: run the rollout policy from the new child until
 *      either a terminal state, the rollout depth cap, or the
 *      problem's natural horizon. Sum discounted rewards.
 *   4. Backup: walk back to the root, incrementing visit counts and
 *      adding the rollout return at each ancestor. Each ancestor's
 *      stored return is in the LEAF's frame, discounted by the depth
 *      to the leaf.
 *
 * The action returned at the root is either the most-visited child
 * (robust; default) or the highest-value child (greedy).
 */
/**
 * MCTSStation — concrete TreeSearchStation<Node<S>> leaf for UCT.
 *
 *  pickNext()  → walks down from root via UCT; returns the leaf chosen
 *                  for this iteration.
 *  evaluate()  → runs the rollout simulation from that leaf and reports
 *                  the discounted return as `bound`/`value`. `isLeaf` is
 *                  true for terminal nodes; otherwise we expand one
 *                  untried action immediately during pickNext.
 *  expand()    → no-op (the new child is created inline during selection,
 *                  so by the time evaluate finishes there's nothing left
 *                  to expand at the runner level).
 *  pushChildren → no-op (frontier is the in-memory tree, not a queue).
 */
export class MCTSStation<S> extends TreeSearchStation<Node<S>> {
  readonly root: Node<S>;
  private readonly env: MCTSEnv<S>;
  private readonly maxIters: number;
  private readonly c: number;
  private readonly rng: () => number;
  private readonly rolloutDepth: number;
  private readonly gamma: number;
  private readonly isTerminal: (s: S) => boolean;
  private readonly defaultPolicy: (s: S) => number;
  /** Last rollout's return, kept for backup() because TreeSearchStation
   *  separates evaluate() and expand()/onIncumbentUpdate(). */
  private lastG = 0;
  /** Path from root to the leaf evaluated this iteration; used to back up
   *  visits and totalReturn. */
  private lastPath: Node<S>[] = [];

  constructor(env: MCTSEnv<S>, rootState: S, opts: MCTSOptions = {}) {
    super('mcts', {objective: 'maximise', maxNodes: opts.iterations ?? 200});
    this.env = env;
    this.maxIters = opts.iterations ?? 200;
    this.c = opts.c ?? Math.sqrt(2);
    this.rng = opts.rng ?? Math.random;
    this.rolloutDepth = env.rolloutDepth ?? 50;
    this.gamma = env.gamma ?? 1.0;
    this.isTerminal = env.isTerminal ?? (() => false);
    this.defaultPolicy = env.rolloutPolicy
      ?? ((s: S) => Math.floor(this.rng() * this.env.numActions(s)));
    this.root = makeNode(rootState, null, -1, 0,
                          env.numActions(rootState), this.isTerminal(rootState));
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  /** Walk the in-memory tree via UCT down to a leaf, expand one untried
   *  action if available, and stash the path. Returns the chosen leaf. */
  protected pickNext(): Node<S> | null {
    if (this.nodesProcessed >= this.maxIters) return null;
    let node = this.root;
    const path: Node<S>[] = [node];
    while (node.untried.length === 0 && node.children.size > 0 && !node.done) {
      // UCT selection with UNIFORM RANDOM TIE-BREAKING. Critical at fresh
      // children where mean=0 and the sqrt term is identical: deterministic
      // argmax would always descend the lowest-action-id child, biasing the
      // search tree's shape.
      const eps = ARGMAX_EPS_DEFAULT;
      let bestUct = -Infinity;
      let bestChild: Node<S> | null = null;
      let tieCount = 0;
      for (const child of node.children.values()) {
        const mean = child.visits > 0 ? child.totalReturn / child.visits : 0;
        const uct = mean + this.c * Math.sqrt(Math.log(node.visits + 1) / (child.visits + 1e-12));
        if (bestChild === null || uct > bestUct + eps) {
          bestUct = uct; bestChild = child; tieCount = 1;
        } else if (uct >= bestUct - eps) {
          tieCount++;
          if (this.rng() * tieCount < 1) bestChild = child;
        }
      }
      node = bestChild!;
      path.push(node);
    }
    if (node.untried.length > 0 && !node.done) {
      const idx = Math.floor(this.rng() * node.untried.length);
      const action = node.untried.splice(idx, 1)[0];
      const {next, reward, done} = this.env.applyAction(node.state, action);
      const child = makeNode(next, node, action, reward,
                              this.env.numActions(next),
                              done || this.isTerminal(next));
      node.children.set(action, child);
      node = child;
      path.push(node);
    }
    this.lastPath = path;
    return node;
  }

  /** Run a rollout from the leaf, accumulating discounted reward into
   *  `lastG`. Returns the value as `bound` so the base records it (we
   *  don't actually use the incumbent semantics for MCTS — we use
   *  visit-count or value-greedy at the end). */
  protected evaluate(node: Node<S>): NodeEvaluation {
    let s = node.state;
    let G = node.rewardIn;
    let discount = this.gamma;
    if (!node.done) {
      for (let d = 0; d < this.rolloutDepth; d++) {
        if (this.isTerminal(s)) break;
        const a = this.defaultPolicy(s);
        const r = this.env.applyAction(s, a);
        G += discount * r.reward;
        discount *= this.gamma;
        s = r.next;
        if (r.done) break;
      }
    }
    this.lastG = G;
    // Backup happens here so the path's nodes accumulate visits and
    // returns BEFORE the next pickNext walks the tree again.
    let G_acc = G;
    for (let i = this.lastPath.length - 1; i >= 0; i--) {
      const cur = this.lastPath[i];
      cur.visits++;
      cur.totalReturn += G_acc;
      G_acc = cur.rewardIn + this.gamma * G_acc;
    }
    // Always treat as leaf so the runner doesn't try to expand again —
    // expansion was inlined in pickNext above.
    return {bound: G, isLeaf: true, value: G, isFeasible: false};
  }

  /** Never reached: every evaluate() returns isLeaf=true. */
  protected expand(_node: Node<S>, _ev: NodeEvaluation): Node<S>[] { return []; }
  /** Frontier is the in-memory tree itself — nothing to push. */
  protected pushChildren(_children: Node<S>[]): void {}

  // ── PUBLIC ROOT-CHILD ACCESSORS ──────────────────────────────────────────

  rootChildVisits(): Map<number, number> {
    const m = new Map<number, number>();
    for (const [a, child] of this.root.children) m.set(a, child.visits);
    return m;
  }
  rootChildValues(): Map<number, number> {
    const m = new Map<number, number>();
    for (const [a, child] of this.root.children) {
      m.set(a, child.visits > 0 ? child.totalReturn / child.visits : 0);
    }
    return m;
  }
}

export function mcts<S>(
  env: MCTSEnv<S>,
  rootState: S,
  opts: MCTSOptions = {},
): {action: number; visits: Map<number, number>; values: Map<number, number>} {
  const sel = opts.selection ?? 'visits';
  const station = new MCTSStation(env, rootState, opts);
  runIterativeDES([station]);

  const visits = station.rootChildVisits();
  const values = station.rootChildValues();
  // Final action choice with random tie-breaking. With low iteration budgets
  // multiple children often end up with identical visit counts (e.g. 1 each
  // after `iterations === numActions`), and deterministic argmax would
  // collapse to action 0. We use the station's own rng for reproducibility.
  const rng = (station as unknown as {rng: () => number}).rng ?? Math.random;
  const eps = ARGMAX_EPS_DEFAULT;
  const childKeys = [...station.root.children.keys()].sort((x, y) => x - y);
  let bestAction = childKeys[0] ?? 0;
  let bestScore = -Infinity;
  let tieCount = 0;
  for (const a of childKeys) {
    const child = station.root.children.get(a)!;
    const score = sel === 'visits' ? child.visits
                                   : (child.visits > 0 ? child.totalReturn / child.visits : -Infinity);
    if (tieCount === 0 || score > bestScore + eps) {
      bestScore = score; bestAction = a; tieCount = 1;
    } else if (score >= bestScore - eps) {
      tieCount++;
      if (rng() * tieCount < 1) bestAction = a;
    }
  }
  return {action: bestAction, visits, values};
}
