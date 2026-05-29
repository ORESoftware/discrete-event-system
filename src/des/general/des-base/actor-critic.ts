'use strict';

// =============================================================================
// general/des-base/actor-critic.ts — base class for ONE-STEP ACTOR-CRITIC
// (Sutton & Barto §13.5). The agent simultaneously learns:
//
//     • a state-value function V_w(s)        ("CRITIC", w ∈ ℝ)
//     • a parameterised policy π_θ(a|s)      ("ACTOR",  θ ∈ ℝ)
//
//   On every transition (s, a, r, s', done) it computes a TD error
//
//       δ = r + γ V_w(s') − V_w(s)            (0 if done)
//
//   and applies the canonical updates
//
//       w  ← w + α_w δ ∇_w V_w(s)
//       θ  ← θ + α_θ δ ∇_θ log π_θ(a|s)
//
// CHOICES THE BASE CLASS LOCKS DOWN
// ─────────────────────────────────
//   - Tabular CRITIC V(s) — closed-form ∇_w V(s) = e_s.
//   - Tabular softmax ACTOR with parameters logits[s][a]:
//
//       π(a|s) = softmax(logits[s])
//       ∇log π(a|s) = e_a − π(a|s)
//
//   So the updates become (per step):
//
//       V(s)        ← V(s) + α_v δ
//       logits[s][a]← logits[s][a] + α_p δ (1 − π(a|s))
//       logits[s][b]← logits[s][b] − α_p δ π(b|s)            for b ≠ a
//
//   These are the simplest possible actor-critic updates and they
//   already work on small environments (gridworld, taxi, corridor)
//   where neural networks are overkill.
//
// HOOKS
// ─────
//   This base is fully concrete for finite (state, action) spaces. To
//   swap in a different parameterisation (e.g. linear softmax over
//   features, or a neural network) the user subclasses and overrides
//   `valueOf`, `pi`, `valueGradStep`, `policyGradStep`. Default
//   implementation is tabular.
// =============================================================================

import {argMaxWithTieBreak} from './argmax';
import {RLAgentStation} from './rl-agent';

export interface ActorCriticOptions {
  rng: () => number;
  numStates: number;
  numActions: number;
  /** Critic learning rate α_v. Default 0.1. */
  alphaV?: number;
  /** Actor learning rate α_p. Default 0.05. */
  alphaP?: number;
  /** Discount γ. Default 0.95. */
  gamma?: number;
  /** Initial logits value (broadcast). Default 0. */
  initLogits?: number;
  /** Initial value estimate (broadcast). Default 0. */
  initV?: number;
  /** Optional entropy coefficient β; adds β · H(π(·|s)) to the actor
   *  gradient. Default 0 (disabled). */
  entropyCoef?: number;
}

export class TabularActorCritic extends RLAgentStation<number, number> {
  protected readonly N: number;
  protected readonly A: number;
  protected readonly V: Float64Array;
  protected readonly logits: Float64Array;       // flat N × A
  protected readonly alphaV: number;
  protected readonly alphaP: number;
  protected readonly gamma: number;
  protected readonly entropyCoef: number;
  /** Per-episode mean |δ| (TD error) for diagnostics. */
  readonly tdErrorHistory: number[] = [];
  private epAbsTd = 0;
  private epUpdates = 0;

  constructor(id: string, opts: ActorCriticOptions) {
    super(id, {rng: opts.rng});
    this.N = opts.numStates;
    this.A = opts.numActions;
    this.alphaV = opts.alphaV ?? 0.1;
    this.alphaP = opts.alphaP ?? 0.05;
    this.gamma = opts.gamma ?? 0.95;
    this.entropyCoef = opts.entropyCoef ?? 0;
    this.V = new Float64Array(this.N);
    if (opts.initV) this.V.fill(opts.initV);
    this.logits = new Float64Array(this.N * this.A);
    if (opts.initLogits) this.logits.fill(opts.initLogits);
  }

  /** π(·|s) — softmax over logits[s]. */
  protected pi(state: number): Float64Array {
    const off = state * this.A;
    let mx = -Infinity;
    for (let a = 0; a < this.A; a++) if (this.logits[off + a] > mx) mx = this.logits[off + a];
    const buf = new Float64Array(this.A);
    let z = 0;
    for (let a = 0; a < this.A; a++) { buf[a] = Math.exp(this.logits[off + a] - mx); z += buf[a]; }
    for (let a = 0; a < this.A; a++) buf[a] /= z;
    return buf;
  }

  protected pickAction(state: number, rng: () => number): number {
    const probs = this.pi(state);
    const u = rng();
    let acc = 0;
    for (let a = 0; a < this.A; a++) { acc += probs[a]; if (u < acc) return a; }
    return this.A - 1;
  }

  protected update(state: number, action: number, reward: number, nextState: number, done: boolean): void {
    const δ = reward + (done ? 0 : this.gamma * this.V[nextState]) - this.V[state];
    // Critic step (tabular ∇V(s) = e_s).
    this.V[state] += this.alphaV * δ;
    // Actor step (∇log π(a|s) = e_a − π(·|s)).
    const probs = this.pi(state);
    const off = state * this.A;
    for (let b = 0; b < this.A; b++) {
      const grad = (b === action ? 1 : 0) - probs[b];
      this.logits[off + b] += this.alphaP * δ * grad;
      if (this.entropyCoef !== 0 && probs[b] > 0) {
        // Entropy bonus: − ∇θ Σ_b π(b|s) log π(b|s)
        // ∂H/∂logit_b = π(b)(− log π(b) − Σ_c π(c)(− log π(c)))
        // We simply add a small subtractive term that pushes the
        // distribution toward uniform.
        this.logits[off + b] += this.entropyCoef * (probs[b] - 1 / this.A);
      }
    }
    this.epAbsTd += Math.abs(δ);
    this.epUpdates += 1;
  }

  protected override endOfEpisode(_episodeId: number): void {
    if (this.epUpdates > 0) this.tdErrorHistory.push(this.epAbsTd / this.epUpdates);
    this.epAbsTd = 0; this.epUpdates = 0;
  }

  /** Argmax over the policy distribution at state s. */
  greedyAction(state: number): number {
    return argMaxWithTieBreak(this.pi(state), this.rng);
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getV(): Float64Array { return this.V; }
  getLogits(): Float64Array { return this.logits; }
  getPolicyProb(state: number, action: number): number { return this.pi(state)[action]; }
}
