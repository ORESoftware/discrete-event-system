'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/rl-environments.rs  (module des::general::rl_environments)
// 1:1 file move. Small pure RL environments (GridWorld, Corridor) shared by the RL agents.
//
// Declarations → Rust:
//   interface Environment              -> trait Environment (numStates/numActions/reset/step/render?)
//   class GridWorld / Corridor implements Environment -> structs + impl Environment trait
//   fn evalPolicy                      -> free fn / assoc fn
//
// Conversion notes (file-specific):
//   - `step` returns an inline `{nextState, reward, done}` -> a named struct (e.g. `StepOutcome`)
//     with `next_state: usize, reward: f64, done: bool`.
//   - `render?(state): string` optional method -> `fn render(&self, s: usize) -> Option<String>`
//     (or a separate optional trait); states/actions are `usize`.
//   - environments are PURE/deterministic (no RNG); the DES wrapping lives in the agent files.
// =============================================================================
// general/rl-environments.ts — small reinforcement-learning environments
// shared by qlearning-des.ts and ppo-des.ts.
//
// Both environments implement the same minimal interface:
//
//   reset() → starting state
//   step(s, a) → {nextState, reward, done}
//   numStates / numActions
//
// Environments are PURE (no DES); the DES wrapping happens in the
// per-algorithm files via an EnvironmentStation that routes Action
// tokens through `step` and emits Transition tokens.
// =============================================================================

export interface Environment {
  numStates: number;
  numActions: number;
  reset(): number;
  step(state: number, action: number): {nextState: number; reward: number; done: boolean};
  /** Optional: render an ASCII view of a state (used for debug output). */
  render?(state: number): string;
}

// -----------------------------------------------------------------------------
// 4×4 GRIDWORLD
// -----------------------------------------------------------------------------
//
//   . . . .         start at top-left, goal at bottom-right.
//   . X . .         X = pit (terminal, large negative reward).
//   . . X .
//   . . . G         actions: 0=up, 1=right, 2=down, 3=left.
//
// Reward: -1 per step (encourages short paths); -10 in a pit; +10 at goal.
// -----------------------------------------------------------------------------

export class GridWorld implements Environment {
  readonly width: number;
  readonly height: number;
  readonly numStates: number;
  readonly numActions = 4;
  readonly start: number;
  readonly goal: number;
  readonly pits: Set<number>;
  /** Action displacements: 0=up, 1=right, 2=down, 3=left. */
  static readonly DR = [-1, 0, 1, 0];
  static readonly DC = [0, 1, 0, -1];

  constructor(opts: {width?: number; height?: number; start?: number; goal?: number; pits?: number[]} = {}) {
    this.width = opts.width ?? 4;
    this.height = opts.height ?? 4;
    this.numStates = this.width * this.height;
    this.start = opts.start ?? 0;
    this.goal = opts.goal ?? this.numStates - 1;
    this.pits = new Set(opts.pits ?? [5, 10]);
  }

  reset(): number { return this.start; }

  step(state: number, action: number): {nextState: number; reward: number; done: boolean} {
    if (state === this.goal || this.pits.has(state)) {
      // Terminal absorbing — bug guard: reset on next call.
      return {nextState: state, reward: 0, done: true};
    }
    const r = Math.floor(state / this.width);
    const c = state % this.width;
    const dr = GridWorld.DR[action];
    const dc = GridWorld.DC[action];
    let nr = r + dr, nc = c + dc;
    // Clamp at walls (no-op move).
    if (nr < 0 || nr >= this.height || nc < 0 || nc >= this.width) { nr = r; nc = c; }
    const ns = nr * this.width + nc;
    if (ns === this.goal) return {nextState: ns, reward: 10, done: true};
    if (this.pits.has(ns))  return {nextState: ns, reward: -10, done: true};
    return {nextState: ns, reward: -1, done: false};
  }

  render(state: number): string {
    const lines: string[] = [];
    for (let r = 0; r < this.height; r++) {
      const row: string[] = [];
      for (let c = 0; c < this.width; c++) {
        const idx = r * this.width + c;
        if (idx === state) row.push('A');
        else if (idx === this.goal) row.push('G');
        else if (this.pits.has(idx)) row.push('X');
        else row.push('.');
      }
      lines.push(row.join(' '));
    }
    return lines.join('\n');
  }

  /** Iterate the gridworld's true Bellman optimal V* via value iteration.
   *  Used for validation. */
  optimalV(gamma = 0.95, tol = 1e-9, maxIters = 5000): {V: number[]; pi: number[]} {
    const V = new Array(this.numStates).fill(0);
    const pi = new Array(this.numStates).fill(0);
    for (let it = 0; it < maxIters; it++) {
      let maxDelta = 0;
      for (let s = 0; s < this.numStates; s++) {
        if (s === this.goal || this.pits.has(s)) continue;
        let bestQ = -Infinity, bestA = 0;
        for (let a = 0; a < this.numActions; a++) {
          const {nextState, reward, done} = this.step(s, a);
          const q = reward + (done ? 0 : gamma * V[nextState]);
          if (q > bestQ) { bestQ = q; bestA = a; }
        }
        const delta = Math.abs(bestQ - V[s]);
        if (delta > maxDelta) maxDelta = delta;
        V[s] = bestQ; pi[s] = bestA;
      }
      if (maxDelta < tol) break;
    }
    return {V, pi};
  }
}

// -----------------------------------------------------------------------------
// 1-D CORRIDOR (length N)
// -----------------------------------------------------------------------------
//
//   o─o─o─o─o─o─G        start at 0, goal at N-1, only two actions: left/right.
//
// Reward: -1 per step, +10 at goal. Optimal value V*(s) = 10·γ^(N-1-s) - sum.
// Useful for PPO since the action-value gap is monotone — easy to learn.
// -----------------------------------------------------------------------------

export class Corridor implements Environment {
  readonly numStates: number;
  readonly numActions = 2;        // 0 = left, 1 = right
  readonly start: number;
  readonly goal: number;

  constructor(length = 8, start = 0) {
    this.numStates = length;
    this.start = start;
    this.goal = length - 1;
  }

  reset(): number { return this.start; }

  step(state: number, action: number): {nextState: number; reward: number; done: boolean} {
    if (state === this.goal) return {nextState: state, reward: 0, done: true};
    let ns = action === 0 ? state - 1 : state + 1;
    if (ns < 0) ns = 0;
    if (ns >= this.numStates) ns = this.numStates - 1;
    if (ns === this.goal) return {nextState: ns, reward: 10, done: true};
    return {nextState: ns, reward: -1, done: false};
  }

  render(state: number): string {
    const cells: string[] = [];
    for (let i = 0; i < this.numStates; i++) {
      cells.push(i === state ? 'A' : i === this.goal ? 'G' : 'o');
    }
    return cells.join('─');
  }

  optimalV(gamma = 0.95, tol = 1e-9, maxIters = 5000): {V: number[]; pi: number[]} {
    const V = new Array(this.numStates).fill(0);
    const pi = new Array(this.numStates).fill(0);
    for (let it = 0; it < maxIters; it++) {
      let maxDelta = 0;
      for (let s = 0; s < this.numStates; s++) {
        if (s === this.goal) continue;
        let bestQ = -Infinity, bestA = 0;
        for (let a = 0; a < this.numActions; a++) {
          const {nextState, reward, done} = this.step(s, a);
          const q = reward + (done ? 0 : gamma * V[nextState]);
          if (q > bestQ) { bestQ = q; bestA = a; }
        }
        const delta = Math.abs(bestQ - V[s]);
        if (delta > maxDelta) maxDelta = delta;
        V[s] = bestQ; pi[s] = bestA;
      }
      if (maxDelta < tol) break;
    }
    return {V, pi};
  }
}

// -----------------------------------------------------------------------------
// EVALUATE A POLICY by Monte-Carlo rollouts.
// -----------------------------------------------------------------------------
export function evalPolicy(
  env: Environment,
  pickAction: (state: number, rng: () => number) => number,
  opts: {numEpisodes?: number; maxStepsPerEpisode?: number; rng?: () => number; gamma?: number} = {},
): {meanReturn: number; meanLength: number; successRate: number} {
  const N = opts.numEpisodes ?? 100;
  const maxSteps = opts.maxStepsPerEpisode ?? 200;
  const rng = opts.rng ?? Math.random;
  const gamma = opts.gamma ?? 1.0;
  let totalReturn = 0, totalLen = 0, successes = 0;
  for (let ep = 0; ep < N; ep++) {
    let s = env.reset();
    let G = 0; let len = 0; let g = 1;
    let done = false;
    while (!done && len < maxSteps) {
      const a = pickAction(s, rng);
      const r = env.step(s, a);
      G += g * r.reward; g *= gamma;
      s = r.nextState; done = r.done; len++;
    }
    totalReturn += G; totalLen += len;
    if (done && len < maxSteps) successes++;
  }
  return {meanReturn: totalReturn / N, meanLength: totalLen / N, successRate: successes / N};
}
