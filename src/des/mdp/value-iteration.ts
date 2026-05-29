'use strict';

// =============================================================================
// Generic value iteration for finite-state, finite-action MDPs.
//
// Bellman optimality:
//   V*(s) = max_a [Σ_s' T(s'|s,a) * (r(s,a,s') + γ V*(s'))]
//   π*(s) = argmax_a [Σ_s' T(s'|s,a) * (r(s,a,s') + γ V*(s'))]
//
// We solve via Gauss–Seidel synchronous updates (in-place writes during
// the same sweep are NOT used — V_next is built into a fresh array, then
// swapped). Sum-coalescing in usacc-mdp.outcomes() means each (s,a) has
// at most a handful of outcomes, so the inner loop is fast.
//
// Convergence: stop when max_s |V_next(s) - V(s)| < tol.
// =============================================================================

import {ACCEPTED, CLOSED, EXHAUSTED, isTerminal, N_ACTIONS, N_STATES, Outcome, outcomes, terminalReward} from './usacc-mdp';

export interface VIOptions {
  gamma?: number;        // discount factor (default 0.95)
  tol?:   number;        // convergence tolerance on max |ΔV|
  maxIter?: number;      // hard cap on iterations
}

export interface VIResult {
  V: Float64Array;       // optimal value function, one entry per state id
  policy: Int32Array;    // action index per state id (-1 for absorbing terminals)
  iterations: number;
  finalDelta: number;
  gamma: number;
}

/**
 * Pre-build a transition table so value iteration doesn't recompute
 * outcomes() on every sweep. This is a substantial speedup (~50x) for
 * even moderate state spaces.
 */
export function buildTransitionTable(): Outcome[][][] {
  const table: Outcome[][][] = [];
  for (let s = 0; s < N_STATES; s++) {
    const perAction: Outcome[][] = [];
    for (let a = 0; a < N_ACTIONS; a++) {
      perAction.push(outcomes(s, a));
    }
    table.push(perAction);
  }
  return table;
}

export function valueIteration(opts: VIOptions = {}): VIResult {
  const gamma = opts.gamma ?? 0.95;
  const tol = opts.tol ?? 1e-9;
  const maxIter = opts.maxIter ?? 5000;

  const T = buildTransitionTable();
  let V = new Float64Array(N_STATES);
  // Initialize terminal states to their terminal reward; non-terminal to 0.
  V[ACCEPTED]  = terminalReward(ACCEPTED);
  V[CLOSED]    = terminalReward(CLOSED);
  V[EXHAUSTED] = terminalReward(EXHAUSTED);

  let iterations = 0;
  let finalDelta = Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    const Vn = new Float64Array(N_STATES);
    Vn[ACCEPTED]  = V[ACCEPTED];
    Vn[CLOSED]    = V[CLOSED];
    Vn[EXHAUSTED] = V[EXHAUSTED];
    let delta = 0;
    for (let s = 0; s < N_STATES; s++) {
      if (isTerminal(s)) continue;
      let best = -Infinity;
      for (let a = 0; a < N_ACTIONS; a++) {
        const ol = T[s][a];
        let q = 0;
        for (let i = 0; i < ol.length; i++) {
          const o = ol[i];
          q += o.prob * (o.reward + gamma * V[o.nextState]);
        }
        if (q > best) best = q;
      }
      Vn[s] = best;
      const d = Math.abs(Vn[s] - V[s]);
      if (d > delta) delta = d;
    }
    V = Vn;
    iterations = iter + 1;
    finalDelta = delta;
    if (delta < tol) break;
  }

  // Extract greedy policy from V.
  const policy = new Int32Array(N_STATES);
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(s)) { policy[s] = -1; continue; }
    let bestA = 0; let bestQ = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) {
      const ol = T[s][a];
      let q = 0;
      for (let i = 0; i < ol.length; i++) {
        const o = ol[i];
        q += o.prob * (o.reward + gamma * V[o.nextState]);
      }
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    policy[s] = bestA;
  }
  return {V, policy, iterations, finalDelta, gamma};
}
