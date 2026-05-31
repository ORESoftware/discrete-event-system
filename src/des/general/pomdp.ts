// RUST MIGRATION: Target module `src/des/general/pomdp.rs`.
// RUST MIGRATION: Convert `POMDPSpec`, value-iteration/lookahead options, action values, and exact-result shapes to generic `serde` structs where type bounds permit serialization.
// RUST MIGRATION: Port solver classes (`QMDPSolver`, `BeliefLookaheadSolver`, `MostLikelyStateSolver`) as structs implementing solver traits with explicit generic bounds on state/action/observation keys.
// RUST MIGRATION: Use nominal enums/structs instead of structural tuples for leaves and alpha vectors; prefer `HashMap` indexes when generic states/actions need lookup.
// RUST MIGRATION: Make invalid probability, horizon, and missing-index cases return `Result`; keep pure solver helpers as free functions unless a DES-visible `PureTransform` wrapper is added.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/pomdp.rs  (module des::general::pomdp)
// 1:1 file move. POMDP solvers (most-likely-state, QMDP, exact α-vector finite horizon).
//
// Declarations → Rust:
//   interface POMDPSpec<S,A,O>/MDPVIOptions/MDPVIResult/BeliefLookaheadOptions/
//             BeliefActionValue/POMDPExactResult -> structs (generics carry over)
//   type BeliefLookaheadLeaf = 'zero' | 'qmdp'   -> enum
//   class QMDPSolver/BeliefLookaheadSolver/MostLikelyStateSolver<S,A,O> -> structs + impl
//   fn beliefUpdate/mdpValueIteration/expectedBeliefReward/observationDistribution/
//      pomdpExactFiniteHorizon/pruneAlphas       -> free fns / assoc fns
//
// Conversion notes (file-specific):
//   - POMDPSpec's T(s,a,s')/O(s',a,o)/R(s,a) are CALLBACK fields -> either `Fn(..)->f64`
//     trait-bound fields or precomputed `Vec<Vec<Vec<f64>>>` tables; pick one engine-wide.
//   - generic <S, A, O> default to indices; if kept generic they need `Copy + Eq + Hash`
//     for any map/lookahead keying.
//   - α-vectors and belief vectors are `number[]`/`number[][]` -> `Vec<f64>`/`Vec<Vec<f64>>`.
//   - α-vector pruning grows |A|^t·|Ω|^t — keep the |S|≤8, t≤6 caps as `assert!`/panic guards.
//   - deterministic (planning only); no RNG/clock here.
// =============================================================================
// POMDP machinery on top of the framework.
//
// We adopt the standard tuple ⟨S, A, Ω, T, O, R, γ⟩ where:
//   S = finite set of hidden states
//   A = finite set of actions
//   Ω = finite set of observations
//   T(s, a, s') = P(s' | s, a)
//   O(s', a, o)  = P(o  | s', a)
//   R(s, a)      = expected immediate reward
//   γ            = discount
//
// Three solvers are provided (in order of cost):
//
//   1. MOST-LIKELY-STATE: pretend the modal hidden state is the truth, then
//      run a 1-step greedy on R(s, a). Trivial baseline.
//
//   2. QMDP (Littman & Cassandra 1995): solve the underlying MDP via value
//      iteration, then act greedily under the belief —
//        a* = argmax_a Σ_s b(s) · Q*_MDP(s, a)
//      Optimal when the agent will receive perfect information at the next
//      step; in practice a strong heuristic.
//
//   3. FINITE-HORIZON BELIEF MDP via α-vectors (point-based). Computes the
//      true POMDP value function V*(b) as the upper envelope of |A|^t · |Ω|^t
//      α-vectors for horizon t. Tractable only for very small problems
//      (we cap |S| at 8 and t at 6); useful as a ground-truth oracle for
//      validating QMDP on toy POMDPs.
//
// The infrastructure does not assume anything specific to FactMachine; it
// is reused by main-factmachine.ts and the unit tests.
// =============================================================================

import {DiscreteBelief} from './belief';

export interface POMDPSpec<S = number, A = number, O = number> {
  states: ReadonlyArray<S>;
  actions: ReadonlyArray<A>;
  observations: ReadonlyArray<O>;
  /** P(s' | s, a) — return a vector of length |S| parallel to `states`. */
  transition: (sIdx: number, aIdx: number) => ReadonlyArray<number>;
  /** P(o | s', a) — return a vector of length |Ω| parallel to `observations`. */
  observation: (sNextIdx: number, aIdx: number) => ReadonlyArray<number>;
  /** Expected immediate reward R(s, a). */
  reward: (sIdx: number, aIdx: number) => number;
  discount: number;
  /** Optional initial belief b₀; defaults to uniform. */
  initialBelief?: ReadonlyArray<number>;
  /** Optional terminal flag — return true if the state is absorbing. */
  isTerminal?: (sIdx: number) => boolean;
}

// -----------------------------------------------------------------------------
// Belief update (Bayesian filter): b'(s') ∝ O(s', a, o) · Σ_s T(s, a, s') · b(s)
// -----------------------------------------------------------------------------
export function beliefUpdate<S, A, O>(
  spec: POMDPSpec<S, A, O>,
  b: DiscreteBelief<S>,
  aIdx: number,
  oIdx: number,
): DiscreteBelief<S> {
  const K = spec.states.length;
  // Predict: bp(s') = Σ_s T(s, a, s') · b(s).
  const bp = new Array<number>(K).fill(0);
  for (let i = 0; i < K; i++) {
    const tRow = spec.transition(i, aIdx);
    const w = b.weights[i];
    for (let j = 0; j < K; j++) bp[j] += w * tRow[j];
  }
  // Correct: weight by P(o | s', a).
  let total = 0;
  for (let j = 0; j < K; j++) {
    const oRow = spec.observation(j, aIdx);
    bp[j] *= oRow[oIdx];
    total += bp[j];
  }
  if (!Number.isFinite(total) || total <= 0) {
    // Impossible observation under the model. Fall back to uniform.
    return new DiscreteBelief(spec.states);
  }
  for (let j = 0; j < K; j++) bp[j] /= total;
  return new DiscreteBelief(spec.states, bp);
}

// -----------------------------------------------------------------------------
// MDP value iteration (used by QMDP).
// -----------------------------------------------------------------------------
export interface MDPVIOptions {
  tol?: number;
  maxIter?: number;
}
export interface MDPVIResult {
  V: number[];
  Q: number[][];        // Q[s][a]
  iterations: number;
  finalDelta: number;
  policy: number[];     // greedy action at each state
}

/** Value iteration on the underlying MDP (treating S as fully observable). */
export function mdpValueIteration<S, A, O>(
  spec: POMDPSpec<S, A, O>,
  opts: MDPVIOptions = {},
): MDPVIResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 5000;
  const K = spec.states.length;
  const numA = spec.actions.length;
  const γ = spec.discount;
  let V = new Array<number>(K).fill(0);
  let Q: number[][] = Array.from({length: K}, () => new Array<number>(numA).fill(0));
  let iter = 0;
  let delta = Infinity;
  while (iter < maxIter && delta > tol) {
    delta = 0;
    const Vnew = new Array<number>(K).fill(0);
    for (let s = 0; s < K; s++) {
      if (spec.isTerminal && spec.isTerminal(s)) {
        Vnew[s] = 0;
        for (let a = 0; a < numA; a++) Q[s][a] = 0;
        continue;
      }
      let best = -Infinity;
      for (let a = 0; a < numA; a++) {
        let q = spec.reward(s, a);
        const tRow = spec.transition(s, a);
        for (let sp = 0; sp < K; sp++) q += γ * tRow[sp] * V[sp];
        Q[s][a] = q;
        if (q > best) best = q;
      }
      Vnew[s] = best;
      const d = Math.abs(Vnew[s] - V[s]);
      if (d > delta) delta = d;
    }
    V = Vnew;
    iter++;
  }
  const policy = V.map((_, s) => {
    let bi = 0; let bv = -Infinity;
    for (let a = 0; a < numA; a++) {
      if (Q[s][a] > bv) { bv = Q[s][a]; bi = a; }
    }
    return bi;
  });
  return {V, Q, iterations: iter, finalDelta: delta, policy};
}

// -----------------------------------------------------------------------------
// QMDP heuristic (Littman & Cassandra 1995).
// -----------------------------------------------------------------------------
export class QMDPSolver<S, A, O> {
  readonly Q: number[][];
  constructor(public spec: POMDPSpec<S, A, O>, opts: MDPVIOptions = {}) {
    const r = mdpValueIteration(spec, opts);
    this.Q = r.Q;
  }

  /** a* = argmax_a Σ_s b(s) Q(s, a) with optional ε-greedy exploration. */
  act(b: DiscreteBelief<S>, rng?: () => number, epsilon = 0): number {
    if (rng && epsilon > 0 && rng() < epsilon) {
      return Math.floor((rng()) * this.spec.actions.length);
    }
    const numA = this.spec.actions.length;
    let bi = 0; let bv = -Infinity;
    for (let a = 0; a < numA; a++) {
      let q = 0;
      for (let s = 0; s < this.Q.length; s++) q += b.weights[s] * this.Q[s][a];
      if (q > bv) { bv = q; bi = a; }
    }
    return bi;
  }

  /** Expected QMDP value E_b[Q(s, a)] — useful for ranking actions. */
  qBelief(b: DiscreteBelief<S>, aIdx: number): number {
    let q = 0;
    for (let s = 0; s < this.Q.length; s++) q += b.weights[s] * this.Q[s][aIdx];
    return q;
  }
}

// -----------------------------------------------------------------------------
// Generic finite-horizon belief lookahead.
//
// This is a reusable POMDP planning primitive between QMDP and exact alpha-vector
// dynamic programming. It searches the belief tree for a configurable depth:
//
//   Q_d(b, a) = R(b, a) + gamma * sum_o P(o | b, a) V_{d-1}(tau(b, a, o))
//
// The leaf value can be zero or the QMDP value function. With horizon=1 and a
// QMDP leaf this generalizes the Tiger "one-step-lookahead" policy; with larger
// horizons it reasons over multiple future information-gathering observations.
// -----------------------------------------------------------------------------

export type BeliefLookaheadLeaf = 'zero' | 'qmdp';

export interface BeliefLookaheadOptions {
  horizon?: number;
  leaf?: BeliefLookaheadLeaf;
  memoize?: boolean;
  beliefPrecision?: number;
  maxNodes?: number;
}

export interface BeliefActionValue {
  action: number;
  q: number;
}

export function expectedBeliefReward<S, A, O>(
  spec: POMDPSpec<S, A, O>,
  b: DiscreteBelief<S>,
  aIdx: number,
): number {
  let r = 0;
  for (let s = 0; s < spec.states.length; s++) r += b.weights[s] * spec.reward(s, aIdx);
  return r;
}

export function observationDistribution<S, A, O>(
  spec: POMDPSpec<S, A, O>,
  b: DiscreteBelief<S>,
  aIdx: number,
): number[] {
  const numO = spec.observations.length;
  const out = new Array<number>(numO).fill(0);
  for (let s = 0; s < spec.states.length; s++) {
    const tRow = spec.transition(s, aIdx);
    for (let sp = 0; sp < spec.states.length; sp++) {
      const pNext = b.weights[s] * tRow[sp];
      if (pNext === 0) continue;
      const oRow = spec.observation(sp, aIdx);
      for (let o = 0; o < numO; o++) out[o] += pNext * oRow[o];
    }
  }
  return out;
}

export class BeliefLookaheadSolver<S, A, O> {
  private readonly horizon: number;
  private readonly leaf: BeliefLookaheadLeaf;
  private readonly memoize: boolean;
  private readonly precision: number;
  private readonly maxNodes: number;
  private readonly qmdp: QMDPSolver<S, A, O>;
  private readonly cache = new Map<string, number>();
  private nodesVisited = 0;

  constructor(readonly spec: POMDPSpec<S, A, O>, opts: BeliefLookaheadOptions = {}) {
    this.horizon = opts.horizon ?? 2;
    this.leaf = opts.leaf ?? 'qmdp';
    this.memoize = opts.memoize ?? true;
    this.precision = opts.beliefPrecision ?? 1e-6;
    this.maxNodes = opts.maxNodes ?? 250000;
    if (!Number.isInteger(this.horizon) || this.horizon < 0) {
      throw new Error(`BeliefLookaheadSolver: horizon must be a nonnegative integer; got ${this.horizon}`);
    }
    if (this.precision <= 0 || !Number.isFinite(this.precision)) {
      throw new Error(`BeliefLookaheadSolver: beliefPrecision must be positive; got ${this.precision}`);
    }
    this.qmdp = new QMDPSolver(spec);
  }

  act(b: DiscreteBelief<S>, rng?: () => number, epsilon = 0): number {
    if (rng && epsilon > 0 && rng() < epsilon) {
      return Math.floor(rng() * this.spec.actions.length);
    }
    const values = this.actionValues(b, this.horizon);
    let best = values[0];
    for (const v of values.slice(1)) if (v.q > best.q) best = v;
    return best.action;
  }

  actionValues(b: DiscreteBelief<S>, depth = this.horizon): BeliefActionValue[] {
    this.nodesVisited = 0;
    return this.actionValuesInner(b, depth);
  }

  value(b: DiscreteBelief<S>, depth = this.horizon): number {
    this.nodesVisited = 0;
    return this.valueInner(b, depth);
  }

  private actionValuesInner(b: DiscreteBelief<S>, depth: number): BeliefActionValue[] {
    const out: BeliefActionValue[] = [];
    for (let a = 0; a < this.spec.actions.length; a++) {
      let q = expectedBeliefReward(this.spec, b, a);
      if (depth > 0) {
        const obs = observationDistribution(this.spec, b, a);
        let future = 0;
        for (let o = 0; o < obs.length; o++) {
          if (obs[o] <= 0) continue;
          const bp = beliefUpdate(this.spec, b, a, o);
          future += obs[o] * this.valueInner(bp, depth - 1);
        }
        q += this.spec.discount * future;
      }
      out.push({action: a, q});
    }
    return out.sort((x, y) => y.q - x.q || x.action - y.action);
  }

  private valueInner(b: DiscreteBelief<S>, depth: number): number {
    this.nodesVisited++;
    if (this.nodesVisited > this.maxNodes) {
      throw new Error(`BeliefLookaheadSolver: exceeded maxNodes=${this.maxNodes}; reduce horizon or increase maxNodes`);
    }
    if (depth <= 0) return this.leafValue(b);
    const key = this.memoize ? this.cacheKey(b, depth) : '';
    if (key) {
      const cached = this.cache.get(key);
      if (cached !== undefined) return cached;
    }
    const best = this.actionValuesInner(b, depth)[0]?.q ?? 0;
    if (key) this.cache.set(key, best);
    return best;
  }

  private leafValue(b: DiscreteBelief<S>): number {
    if (this.leaf === 'zero') return 0;
    let best = -Infinity;
    for (let a = 0; a < this.spec.actions.length; a++) best = Math.max(best, this.qmdp.qBelief(b, a));
    return best;
  }

  private cacheKey(b: DiscreteBelief<S>, depth: number): string {
    return `${depth}|${b.weights.map(w => Math.round(w / this.precision)).join(',')}`;
  }
}

// -----------------------------------------------------------------------------
// Most-likely-state heuristic: act as if the modal hidden state is the truth.
// -----------------------------------------------------------------------------
export class MostLikelyStateSolver<S, A, O> {
  constructor(public spec: POMDPSpec<S, A, O>, public mdpResult = mdpValueIteration(spec)) {}
  act(b: DiscreteBelief<S>): number {
    return this.mdpResult.policy[b.modeIndex()];
  }
}

// -----------------------------------------------------------------------------
// Finite-horizon point-based value iteration, EXACT for small POMDPs.
//
// Represent V_t as a set of α-vectors {α_i}, V_t(b) = max_i ⟨α_i, b⟩.
// Backup operator (Sondik):
//   for each a, for each combination of {α_o}_{o ∈ Ω}:
//     α(s) = R(s, a) + γ Σ_{s'} T(s, a, s') Σ_o O(s', a, o) · α_{o}(s')
//   prune dominated α-vectors.
//
// Combinatorial cost ~ |A| · |α|^|Ω|. Tractable for |Ω| ≤ 4 and small horizons.
// -----------------------------------------------------------------------------
export interface POMDPExactResult {
  alphaVectors: Array<{vec: number[]; action: number}>;
  V: (b: ReadonlyArray<number>) => number;
  act: (b: DiscreteBelief<unknown>) => number;
}

export function pomdpExactFiniteHorizon<S, A, O>(
  spec: POMDPSpec<S, A, O>,
  horizon: number,
): POMDPExactResult {
  const K = spec.states.length;
  const NA = spec.actions.length;
  const NO = spec.observations.length;
  // Initialise V_0 to immediate reward (each action gives one α-vector).
  let alphas: Array<{vec: number[]; action: number}> = [];
  for (let a = 0; a < NA; a++) {
    const v = new Array<number>(K);
    for (let s = 0; s < K; s++) v[s] = spec.reward(s, a);
    alphas.push({vec: v, action: a});
  }
  alphas = pruneAlphas(alphas, K);

  for (let t = 1; t < horizon; t++) {
    const next: Array<{vec: number[]; action: number}> = [];
    for (let a = 0; a < NA; a++) {
      // For each observation, the future α-vectors backprop through O × T.
      // Index over all combinations of |alphas|^|O|.
      const total = Math.pow(alphas.length, NO);
      if (total > 200000) {
        throw new Error(`pomdpExactFiniteHorizon: combinatorial blowup (|alphas|=${alphas.length}, |Ω|=${NO}, total=${total}). Reduce horizon or use QMDP.`);
      }
      const idxs = new Array<number>(NO).fill(0);
      for (let combo = 0; combo < total; combo++) {
        const v = new Array<number>(K);
        for (let s = 0; s < K; s++) {
          let val = spec.reward(s, a);
          const tRow = spec.transition(s, a);
          for (let sp = 0; sp < K; sp++) {
            const oRow = spec.observation(sp, a);
            let inner = 0;
            for (let o = 0; o < NO; o++) {
              inner += oRow[o] * alphas[idxs[o]].vec[sp];
            }
            val += spec.discount * tRow[sp] * inner;
          }
          v[s] = val;
        }
        next.push({vec: v, action: a});
        // Increment idxs in base alphas.length.
        for (let k = NO - 1; k >= 0; k--) {
          idxs[k]++;
          if (idxs[k] < alphas.length) break;
          idxs[k] = 0;
        }
      }
    }
    alphas = pruneAlphas(next, K);
  }

  const V = (b: ReadonlyArray<number>): number => {
    let best = -Infinity;
    for (const {vec} of alphas) {
      let v = 0;
      for (let i = 0; i < K; i++) v += vec[i] * b[i];
      if (v > best) best = v;
    }
    return best;
  };
  const act = (b: DiscreteBelief<unknown>): number => {
    let best = -Infinity; let bestA = 0;
    for (const {vec, action} of alphas) {
      let v = 0;
      for (let i = 0; i < K; i++) v += vec[i] * b.weights[i];
      if (v > best) { best = v; bestA = action; }
    }
    return bestA;
  };
  return {alphaVectors: alphas, V, act};
}

function pruneAlphas(
  alphas: Array<{vec: number[]; action: number}>, K: number,
): Array<{vec: number[]; action: number}> {
  // Cheap pruning: remove vectors that are pointwise dominated.
  const keep: Array<{vec: number[]; action: number}> = [];
  for (let i = 0; i < alphas.length; i++) {
    let dominated = false;
    for (let j = 0; j < alphas.length; j++) {
      if (i === j) continue;
      let allLeq = true; let strict = false;
      for (let k = 0; k < K; k++) {
        if (alphas[j].vec[k] < alphas[i].vec[k] - 1e-12) { allLeq = false; break; }
        if (alphas[j].vec[k] > alphas[i].vec[k] + 1e-12) strict = true;
      }
      if (allLeq && strict) { dominated = true; break; }
    }
    if (!dominated) {
      // Also dedupe near-identical vectors.
      let dup = false;
      for (const k of keep) {
        let same = true;
        for (let q = 0; q < K; q++) {
          if (Math.abs(k.vec[q] - alphas[i].vec[q]) > 1e-9) { same = false; break; }
        }
        if (same) { dup = true; break; }
      }
      if (!dup) keep.push(alphas[i]);
    }
  }
  return keep.length > 0 ? keep : [alphas[0]];
}
