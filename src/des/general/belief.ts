// RUST MIGRATION: target module src/des/general/belief.rs.
// RUST MIGRATION: DiscreteBelief<S> becomes a generic struct with explicit HashMap<S, f64> or Vec<(S, f64)> storage depending on ordering needs.
// RUST MIGRATION: Instance methods map to impl DiscreteBelief<S>; require S: Eq + Hash + Clone if using HashMap, and return Result for invalid probabilities.
// RUST MIGRATION: brierScore and klDivergence are pure numeric helpers and can remain free functions.
'use strict';

// =============================================================================
// DiscreteBelief — a probability distribution over a finite set of hidden
// states, with Bayesian update.
//
// This is the workhorse of every POMDP in the framework:
//
//   * The agent maintains b(s) — a vector of weights, one per possible
//     hidden state.
//   * After observing o, b'(s) ∝ P(o | s) · b(s); we renormalise.
//   * After taking action a (with stochastic transitions T(s' | s, a)),
//     b'(s') = Σ_s T(s' | s, a) · b(s).
//
// We keep things in plain `number[]` and avoid mathjs to stay free of the
// arbitrary-precision biases noted in the project README. The numerical
// pitfalls we DO care about (catastrophic cancellation in renormalisation,
// log-sum-exp for log-likelihoods) are handled here.
// =============================================================================

export class DiscreteBelief<S = number> {
  readonly states: ReadonlyArray<S>;
  readonly weights: number[];

  constructor(states: ReadonlyArray<S>, prior?: ReadonlyArray<number>) {
    if (states.length === 0) throw new Error('DiscreteBelief: empty state set');
    this.states = states;
    if (prior) {
      if (prior.length !== states.length) {
        throw new Error(`prior length ${prior.length} ≠ states length ${states.length}`);
      }
      const total = prior.reduce((s, w) => s + w, 0);
      if (total <= 0 || !Number.isFinite(total)) {
        throw new Error(`prior is degenerate (sum=${total})`);
      }
      this.weights = prior.map(w => w / total);
    } else {
      this.weights = states.map(() => 1 / states.length);
    }
  }

  /** Bayesian update b'(s) ∝ likelihood(s) · b(s). Returns the normalising
   *  constant Σ_s likelihood(s) · b(s) (the marginal likelihood of the
   *  observation), useful for model evidence / online learning. */
  update(likelihood: (state: S, index: number) => number): number {
    let total = 0;
    const next = new Array<number>(this.weights.length);
    for (let i = 0; i < this.weights.length; i++) {
      const l = likelihood(this.states[i], i);
      if (l < 0) throw new Error(`likelihood(${i}) returned negative value ${l}`);
      next[i] = this.weights[i] * l;
      total += next[i];
    }
    if (!Number.isFinite(total) || total <= 0) {
      // Belief collapse: caller should consider it invalid (impossible
      // observation under their model). We fall back to a uniform belief
      // and let the caller decide if that's acceptable.
      const u = 1 / this.weights.length;
      for (let i = 0; i < this.weights.length; i++) this.weights[i] = u;
      return total;
    }
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = next[i] / total;
    return total;
  }

  /** Predictive update for the hidden state evolving via T(s' | s).
   *  `transition(prevState, prevIndex)` should return an array of weights
   *  parallel to `this.states`, representing P(s'_i | prevState).
   *  The new belief is b'(s') = Σ_s T(s' | s) · b(s). */
  propagate(transition: (prevState: S, prevIndex: number) => ReadonlyArray<number>): void {
    const next = new Array<number>(this.weights.length).fill(0);
    for (let i = 0; i < this.weights.length; i++) {
      const tRow = transition(this.states[i], i);
      if (tRow.length !== this.weights.length) {
        throw new Error(`transition row length ${tRow.length} ≠ K = ${this.weights.length}`);
      }
      const w = this.weights[i];
      for (let j = 0; j < tRow.length; j++) next[j] += w * tRow[j];
    }
    let total = 0;
    for (const v of next) total += v;
    if (!Number.isFinite(total) || total <= 0) {
      throw new Error(`belief.propagate: degenerate distribution (sum=${total})`);
    }
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = next[i] / total;
  }

  /** E[f(s)] under the current belief. */
  expectation(f: (state: S, index: number) => number): number {
    let m = 0;
    for (let i = 0; i < this.weights.length; i++) m += f(this.states[i], i) * this.weights[i];
    return m;
  }

  /** E[s] when S is numeric (cast through `Number`). */
  mean(): number {
    let m = 0;
    for (let i = 0; i < this.weights.length; i++) {
      m += Number(this.states[i] as unknown as number) * this.weights[i];
    }
    return m;
  }

  /** Var[s] when S is numeric. */
  variance(): number {
    const mu = this.mean();
    let v = 0;
    for (let i = 0; i < this.weights.length; i++) {
      const x = Number(this.states[i] as unknown as number);
      v += (x - mu) * (x - mu) * this.weights[i];
    }
    return v;
  }

  /** Shannon entropy in nats. H = -Σ b log b. */
  entropy(): number {
    let h = 0;
    for (const w of this.weights) if (w > 0) h -= w * Math.log(w);
    return h;
  }

  /** Argmax (mode) of the belief. */
  modeIndex(): number {
    let bi = 0;
    for (let i = 1; i < this.weights.length; i++) {
      if (this.weights[i] > this.weights[bi]) bi = i;
    }
    return bi;
  }

  mode(): S {
    return this.states[this.modeIndex()];
  }

  /** Sample one hidden state from the belief. */
  sample(rng: () => number): S {
    const u = rng();
    let acc = 0;
    for (let i = 0; i < this.weights.length; i++) {
      acc += this.weights[i];
      if (u <= acc) return this.states[i];
    }
    return this.states[this.weights.length - 1];
  }

  /** A copy of the weight vector (defensive). */
  asArray(): number[] {
    return this.weights.slice();
  }

  clone(): DiscreteBelief<S> {
    return new DiscreteBelief(this.states, this.weights);
  }
}

// -----------------------------------------------------------------------------
// Standalone helpers for cross-checking calibration in validators.
// -----------------------------------------------------------------------------

/** Brier score for a probabilistic prediction p ∈ [0,1] against a binary
 *  outcome y ∈ {0,1}. Smaller is better. Always in [0, 1]. */
export function brierScore(p: number, y: 0 | 1): number {
  return (p - y) * (p - y);
}

/** KL divergence KL(p || q) for two discrete distributions of the same length. */
export function klDivergence(p: ReadonlyArray<number>, q: ReadonlyArray<number>): number {
  if (p.length !== q.length) throw new Error('klDivergence: length mismatch');
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] === 0) continue;
    if (q[i] <= 0) return Infinity;
    kl += p[i] * Math.log(p[i] / q[i]);
  }
  return kl;
}
