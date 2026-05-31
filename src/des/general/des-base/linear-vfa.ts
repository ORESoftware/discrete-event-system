'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/linear_vfa.rs
// - Keep file-for-file. LinearVFAOptions becomes a config struct and
//   LinearVFAStation becomes a trait plus shared RL-agent state struct.
// - Feature extraction and value approximation hooks should become trait
//   methods; weights map to Vec<Vec<f64>> or a flat Vec<f64> with indexing.
// - Pure feature functions can stay associated/private helpers, or become
//   PureTransform/PureTransformEntity if represented as DES graph nodes.
// - Convert feature dimension, action count, and feature-shape throws to Result.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/linear_vfa.rs  (module des::general::des_base::linear_vfa)
// 1:1 file move. Approximate DP with linear function approximation (semi-gradient
// TD / linear Sarsa-Q); extends RLAgentStation.
//
// Declarations → Rust:
//   interface LinearVFAOptions      -> struct (#[derive(Default)] except required fields)
//   abstract class LinearVFAStation<S> -> trait/struct: RLAgentStation<S, usize>
//
// Conversion notes (file-specific):
//   - INHERITANCE: extends RLAgentStation and IMPLEMENTS its pickAction/update/
//     endOfEpisode hooks -> in Rust this struct provides those required trait fns;
//     it ADDS its own `features` hook (-> required trait fn) and `legalActions`
//     (-> provided default returning None).
//   - `theta: Float64Array` flat (A×d) matrix -> `Vec<f64>` (or `ndarray`) indexed
//     `a*d + i`; `.fill()` -> `vec![init; A*d]`.
//   - `rng: () => number` -> inject `RandomSource` (shared/capabilities); greedy
//     tie-break should reuse argmax.rs helpers.
//   - non-ASCII `φ`, `δ` identifiers -> `phi`, `delta`.
//   - `legalActions(): readonly number[] | null` -> `Option<Vec<usize>>`.
//   - `throw new Error` on bad dims -> `Result`/`panic!`.
// =============================================================================

// =============================================================================
// general/des-base/linear-vfa.ts — base class for APPROXIMATE DYNAMIC
// PROGRAMMING with LINEAR FUNCTION APPROXIMATION (linear value-function
// approximation, "linear VFA"). Underlies: linear semi-gradient TD(0) for
// state-value; linear Sarsa / Q-learning for action-value; LSPI; LSTD;
// linear actor-critic (value head); etc.
//
// CORE IDEA
// ─────────
//   When |S| is too large for a tabular V[s], approximate
//
//       V_θ(s) = θ · φ(s),         (or Q_θ(s, a) = θ_a · φ(s))
//
//   with a feature vector φ(s) ∈ ℝ^d and learn θ via the semi-gradient
//   TD update:
//
//       δ_t  = r_t + γ V_θ(s') − V_θ(s)
//       θ ← θ + α δ_t φ(s)
//
//   For action-value approximation we keep one θ_a per discrete action
//   ("disjoint" linear function approximation — the simplest variant
//   that still scales to large state spaces).
//
// AS A DES STATION
// ────────────────
//   LinearVFAStation extends RLAgentStation: it consumes Transition
//   tokens and updates θ by linear semi-gradient. The pickAction hook
//   is ε-greedy over Q_θ.
//
// HOOKS (abstract)
// ────────────────
//   features(state) → φ(s) ∈ ℝ^d
//   numActions      → A
//
// HOOKS (optional override)
// ─────────────────────────
//   featureDim()    → d (default: features(0).length)
//
// USAGE
// ─────
//   The numerical state used by `pickAction` and `update` is whatever
//   the environment emits — typically an integer index, but the user is
//   free to encode richer states (e.g. tuples) and have `features`
//   decode them.
// =============================================================================

import {RLAgentStation} from './rl-agent';
import {ARGMAX_EPS_DEFAULT} from './argmax';

export interface LinearVFAOptions {
  rng: () => number;
  /** Feature dimension d. Required (constructor cannot probe φ). */
  featureDim: number;
  numActions: number;
  /** Step size α. Default 0.1. */
  alpha?: number;
  /** Discount factor γ. Default 0.95. */
  gamma?: number;
  /** ε-greedy exploration probability. Default 0.1. */
  epsilon?: number;
  /** ε decay multiplier per episode. Default 1 (no decay). */
  epsilonDecay?: number;
  /** Floor for ε. Default 0.01. */
  epsilonMin?: number;
  /** Initial θ value (broadcast to all entries). Default 0. */
  initTheta?: number;
}

export abstract class LinearVFAStation<S = number> extends RLAgentStation<S, number> {
  /** θ_a ∈ ℝ^d for each action a. Stored as a flat (A × d) matrix. */
  protected readonly theta: Float64Array;
  protected readonly d: number;
  protected readonly A: number;
  protected readonly alpha: number;
  protected readonly gamma: number;
  protected epsilon: number;
  protected readonly epsilonDecay: number;
  protected readonly epsilonMin: number;
  /** Per-episode TD-error history (mean |δ| over episode). */
  readonly tdErrorHistory: number[] = [];
  private episodeAbsTd = 0;
  private episodeUpdates = 0;

  constructor(id: string, opts: LinearVFAOptions) {
    super(id, {rng: opts.rng});
    if (opts.featureDim < 1) throw new Error('featureDim must be ≥ 1');
    if (opts.numActions < 1) throw new Error('numActions must be ≥ 1');
    this.d = opts.featureDim;
    this.A = opts.numActions;
    this.alpha = opts.alpha ?? 0.1;
    this.gamma = opts.gamma ?? 0.95;
    this.epsilon = opts.epsilon ?? 0.1;
    this.epsilonDecay = opts.epsilonDecay ?? 1;
    this.epsilonMin = opts.epsilonMin ?? 0.01;
    this.theta = new Float64Array(this.A * this.d);
    if (opts.initTheta && opts.initTheta !== 0) this.theta.fill(opts.initTheta);
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected abstract features(state: S): readonly number[];

  /** Optional override: action mask. Default: all actions legal. */
  protected legalActions(_state: S): readonly number[] | null { return null; }

  // ── Q AND POLICY ─────────────────────────────────────────────────────────

  /** Q_θ(s, a) = θ_a · φ(s). */
  Q(state: S, action: number): number {
    const φ = this.features(state);
    if (φ.length !== this.d) throw new Error(`features() returned dim ${φ.length}, expected ${this.d}`);
    const off = action * this.d;
    let q = 0;
    for (let i = 0; i < this.d; i++) q += this.theta[off + i] * φ[i];
    return q;
  }

  /**
   * Argmax over actions of Q_θ(s, ·) with UNIFORM RANDOM TIE-BREAKING.
   *
   * Critical for linear VFA: when θ is initialized to 0, every Q-value is
   * identically 0, so deterministic argmax would always return action 0,
   * blocking exploration under ε=0 greedy and even biasing ε-greedy
   * (the greedy fallback path) for the early learning phase. Random
   * tie-breaking via reservoir-sampling over the tied set fixes this.
   */
  greedyAction(state: S): number {
    const legal = this.legalActions(state);
    const eps = ARGMAX_EPS_DEFAULT;
    let bestA = -1;
    let bestQ = -Infinity;
    let tieCount = 0;
    const scan = (a: number) => {
      const q = this.Q(state, a);
      if (bestA < 0 || q > bestQ + eps) {
        bestQ = q; bestA = a; tieCount = 1;
      } else if (q >= bestQ - eps) {
        tieCount++;
        if (this.rng() * tieCount < 1) bestA = a;
      }
    };
    if (legal) for (const a of legal) scan(a);
    else for (let a = 0; a < this.A; a++) scan(a);
    return bestA < 0 ? 0 : bestA;
  }

  /** ε-greedy. */
  protected pickAction(state: S, rng: () => number): number {
    const legal = this.legalActions(state);
    if (rng() < this.epsilon) {
      if (legal && legal.length > 0) return legal[Math.floor(rng() * legal.length)];
      return Math.floor(rng() * this.A);
    }
    return this.greedyAction(state);
  }

  /** Semi-gradient TD(0) update with Q-learning (max over next actions). */
  protected update(state: S, action: number, reward: number, nextState: S, done: boolean): void {
    const φ = this.features(state);
    const qSa = this.Q(state, action);
    let bootstrap = 0;
    if (!done) {
      // max_a' Q(s', a')
      let maxQ = -Infinity;
      const legalNext = this.legalActions(nextState);
      if (legalNext) {
        for (const a of legalNext) { const q = this.Q(nextState, a); if (q > maxQ) maxQ = q; }
      } else {
        for (let a = 0; a < this.A; a++) { const q = this.Q(nextState, a); if (q > maxQ) maxQ = q; }
      }
      bootstrap = this.gamma * maxQ;
    }
    const δ = reward + bootstrap - qSa;
    const off = action * this.d;
    for (let i = 0; i < this.d; i++) this.theta[off + i] += this.alpha * δ * φ[i];
    this.episodeAbsTd += Math.abs(δ);
    this.episodeUpdates += 1;
  }

  protected override endOfEpisode(_episodeId: number): void {
    if (this.episodeUpdates > 0) {
      this.tdErrorHistory.push(this.episodeAbsTd / this.episodeUpdates);
    }
    this.episodeAbsTd = 0; this.episodeUpdates = 0;
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getTheta(): Float64Array { return this.theta; }
  getEpsilon(): number { return this.epsilon; }
  getFeatureDim(): number { return this.d; }
  getNumActions(): number { return this.A; }
}
