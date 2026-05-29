'use strict';

// =============================================================================
// general/des-base/monte-carlo-rl.ts — base class for ON-POLICY MONTE CARLO
// CONTROL (Sutton & Barto §5.4): every-visit and first-visit Monte Carlo
// estimation of Q*, ε-greedy policy improvement.
//
// PROBLEM SHAPE
// ─────────────
//   Episodic environment. Agent rolls out an entire episode under its
//   ε-soft policy, collects {(s_t, a_t, r_{t+1})}_{t=0..T-1}, then
//   computes returns
//
//       G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}
//
//   and updates Q[s][a] toward the empirical return either every visit
//   or only on the first visit per (s, a) pair within the episode.
//   Updates use INCREMENTAL averaging:
//
//       N(s, a) ← N(s, a) + 1
//       Q(s, a) ← Q(s, a) + (G - Q(s, a)) / N(s, a)
//
//   Policy improvement is implicit through ε-greedy action selection
//   over Q.
//
// CONTRAST WITH TD METHODS
// ────────────────────────
//   - No bootstrapping: targets are full returns G_t, NOT r + γ V(s').
//   - Updates land at end of episode, NOT online.
//   - Unbiased but high-variance.
//   - Very natural for episodic problems where the model is unknown
//     (e.g. card games, episodic robotics).
//
// CHANNEL FLOW (inherits from RLAgentStation)
// ───────────────────────────────────────────
//   Same as Q-learning:  state → pickAction → env step → transition.
//   The DIFFERENCE is that this base accumulates the episode internally
//   and does NOT update on every transition. Q is updated when `done`
//   arrives.
// =============================================================================

import {RLAgentStation} from './rl-agent';

export interface MonteCarloOptions {
  rng: () => number;
  numStates: number;
  numActions: number;
  /** First-visit (true) vs every-visit (false). Default true. */
  firstVisit?: boolean;
  /** Discount γ. Default 1.0 (canonical for episodic MC). */
  gamma?: number;
  /** Exploration ε. Default 0.1. */
  epsilon?: number;
  /** ε-decay multiplier per episode. Default 1. */
  epsilonDecay?: number;
  /** ε-floor. Default 0.01. */
  epsilonMin?: number;
  /** Initial Q value (broadcast). Default 0. */
  initQ?: number;
}

export class MonteCarloAgent extends RLAgentStation<number, number> {
  protected readonly N: number;
  protected readonly A: number;
  protected readonly Q: Float64Array;            // flat N × A
  protected readonly visitCount: Int32Array;     // flat N × A
  protected readonly firstVisit: boolean;
  protected readonly gamma: number;
  protected epsilon: number;
  protected readonly epsilonDecay: number;
  protected readonly epsilonMin: number;
  /** Per-episode trajectory: parallel arrays of states, actions, rewards. */
  private trajS: number[] = [];
  private trajA: number[] = [];
  private trajR: number[] = [];

  constructor(id: string, opts: MonteCarloOptions) {
    super(id, {rng: opts.rng});
    this.N = opts.numStates;
    this.A = opts.numActions;
    this.firstVisit = opts.firstVisit ?? true;
    this.gamma = opts.gamma ?? 1.0;
    this.epsilon = opts.epsilon ?? 0.1;
    this.epsilonDecay = opts.epsilonDecay ?? 1;
    this.epsilonMin = opts.epsilonMin ?? 0.01;
    this.Q = new Float64Array(this.N * this.A);
    if (opts.initQ) this.Q.fill(opts.initQ);
    this.visitCount = new Int32Array(this.N * this.A);
  }

  protected pickAction(state: number, rng: () => number): number {
    if (rng() < this.epsilon) return Math.floor(rng() * this.A);
    const off = state * this.A;
    let bestA = 0; let bestQ = -Infinity;
    for (let a = 0; a < this.A; a++) {
      const q = this.Q[off + a];
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    return bestA;
  }

  /** Each transition just APPENDS to the in-progress trajectory; no Q
   *  update happens until `done`. */
  protected update(state: number, action: number, reward: number, _nextState: number, done: boolean): void {
    this.trajS.push(state);
    this.trajA.push(action);
    this.trajR.push(reward);
    if (done) this.applyEpisode();
  }

  /** Apply Monte Carlo first-visit / every-visit updates over the
   *  trajectory just collected, then reset the buffer. */
  protected applyEpisode(): void {
    const T = this.trajS.length;
    const seen: Set<number> = new Set();
    // Compute returns from the back.
    let G = 0;
    for (let t = T - 1; t >= 0; t--) {
      G = this.gamma * G + this.trajR[t];
      const s = this.trajS[t]; const a = this.trajA[t];
      const key = s * this.A + a;
      if (this.firstVisit && seen.has(key)) continue;
      seen.add(key);
      this.visitCount[key] += 1;
      this.Q[key] += (G - this.Q[key]) / this.visitCount[key];
    }
    this.trajS = []; this.trajA = []; this.trajR = [];
  }

  protected override endOfEpisode(_episodeId: number): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  /** Argmax over Q. */
  greedyAction(state: number): number {
    const off = state * this.A;
    let bestA = 0; let bestQ = -Infinity;
    for (let a = 0; a < this.A; a++) {
      const q = this.Q[off + a];
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    return bestA;
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getQ(): Float64Array { return this.Q; }
  getVisitCounts(): Int32Array { return this.visitCount; }
  getEpsilon(): number { return this.epsilon; }
}
