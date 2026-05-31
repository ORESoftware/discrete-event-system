// RUST MIGRATION: target module src/des/general/des_lp_bridge.rs.
// RUST MIGRATION: MDPLPSolution and RollingHorizonStep become serde structs; generic S/M state-machine types should use explicit trait bounds.
// RUST MIGRATION: solveLPThenSimulate, buildMDPLP, solveMDPAsLP, and lpRollingHorizon are bridge helpers and can remain free functions unless registered as graph transforms.
// RUST MIGRATION: LP tables and policy maps should become Vec<Vec<f64>> or HashMap keyed by state/action; convert infeasible/invalid cases to Result.
'use strict';

// =============================================================================
// DES ↔ LP integration patterns.
//
// In its SIMULATION mode (the original framing of DES), the engine isn't
// itself an exact LP solver — simplex / interior-point methods exploit
// polyhedral geometry that the simulation half doesn't expose. But DES
// + LP is one of the most commonly-deployed combinations in operations
// research, because real systems violate every assumption of an
// idealised LP (stochastic delays, finite buffers, machine downtime,
// random arrivals, queueing). The LP gives the "nominal optimum"; DES
// (in its simulation half) gives "operational realism".
//
// In its ALGORITHMIC half (DES as a "discrete event SYSTEM"), the same
// substrate DOES solve LPs natively — see `general/incremental-lp.ts`
// for the warm-startable simplex-as-DES, and `general/stochastic-lp.ts`
// for Benders/L-shaped on top of it. The integration patterns below
// remain useful for the simulation-optimisation paradigm; the
// incremental/stochastic LP modules are what to reach for when the LP
// is solved repeatedly.
//
// Three patterns are exposed here:
//
// (A) `solveLPThenSimulate(lp, simulator, evaluator)`
//        Solve the LP once for the deterministic plan; pass the LP
//        solution as a fixed policy to the DES simulator; collect
//        realised metrics. The pattern surfaces the gap between
//        nominal LP optimum and stochastic realisation.
//
// (B) `solveMDPAsLP(mdp, gamma)`
//        Convert a finite-state MDP (`MDPSpec` from `value-iteration.ts`)
//        into its primal-LP formulation:
//
//          min Σ_s μ_s · V(s)
//          s.t. V(s) ≥ Σ_{s'} T(s'|s,a) · [r(s,a,s') + γ V(s')]    ∀ s, a
//
//        Solve via the external simplex / interior-point. The optimal
//        V* matches value-iteration to ~1e-9 (verified in
//        runners/validate-lp.ts). This is the "DES → MDP → LP" pipeline
//        the user described.
//
// (C) `lpRollingHorizon(lp, simulator, replanEvery, totalTicks)`
//        Standard MPC-style loop: solve LP, simulate `replanEvery`
//        ticks, observe the realised state, build a fresh LP from that
//        state, repeat. Used for production planning under uncertainty.
// =============================================================================

import {LPProblem, LPSolution, solveLP, ExternalSolverOptions, InternalSimplexOptions} from './lp';
import {MDPSpec, VIOptions} from './value-iteration';

// -----------------------------------------------------------------------------
// (A) Plan-then-simulate.
// -----------------------------------------------------------------------------

/**
 * Run the LP, then pass the optimal x to a DES simulator. The simulator
 * returns realised metrics (e.g. throughput, makespan). Surfaces the gap
 * between nominal LP optimum and stochastic realisation.
 *
 * @param lp        the LP to solve (e.g. ideal production quantities).
 * @param simulate  callback that runs DES given the LP plan; returns a
 *                  metrics object whose `realisedObjective` field is the
 *                  same scalar as the LP's `c^T x`, but evaluated under
 *                  realistic dynamics.
 */
export function solveLPThenSimulate<M>(
  lp: LPProblem,
  simulate: (lpPlan: LPSolution) => M,
  opts: ExternalSolverOptions & InternalSimplexOptions = {},
): {plan: LPSolution; realised: M} {
  const plan = solveLP(lp, opts);
  if (plan.status !== 'optimal') {
    throw new Error(`LP failed with status=${plan.status}: ${plan.message ?? ''}`);
  }
  const realised = simulate(plan);
  return {plan, realised};
}

// -----------------------------------------------------------------------------
// (B) MDP-as-LP.
//
// The Bellman optimality equation
//
//   V*(s) = max_a Σ_{s'} T(s'|s,a) [r(s,a,s') + γ V*(s')]
//
// is equivalent to the LP
//
//   min Σ_s μ_s V(s)
//   s.t.  V(s) − γ Σ_{s'} T(s'|s,a) V(s') ≥ Σ_{s'} T(s'|s,a) r(s,a,s')   ∀ s, a
//
// where μ ≻ 0 is any strictly-positive state-distribution (we use the
// uniform). The optimal V* is unique and matches value iteration; the
// optimal policy π*(s) is the action whose constraint is binding.
//
// (Equivalently the dual LP is the long-run discounted occupancy LP whose
// dual variables d(s,a) yield the policy directly. We use the primal here
// because it fits scipy.linprog more directly with no extra bookkeeping.)
// -----------------------------------------------------------------------------

export interface MDPLPSolution {
  V: number[];          // value function, length numStates
  policy: number[];     // greedy policy, length numStates
  lp: LPSolution;       // raw LP result (for diagnostics)
}

export function buildMDPLP(mdp: MDPSpec, gamma: number,
                            stateDist?: number[]): LPProblem {
  const N = mdp.numStates;
  if (gamma < 0 || gamma >= 1) throw new Error('MDP-as-LP requires 0 ≤ γ < 1');
  const mu = stateDist ?? new Array(N).fill(1 / N);
  if (mu.length !== N) throw new Error('stateDist length mismatch');
  // V(s) is unbounded below for MDPs with negative rewards; allow free V_s.
  const lb: (number | null)[] = new Array(N).fill(null);
  const ub: (number | null)[] = new Array(N).fill(null);

  // For each (s, a) build the inequality:
  //   V(s) − γ Σ_{s'} T(s'|s,a) V(s')  ≥  Σ_{s'} T(s'|s,a) r(s,a,s')
  // which we encode in canonical-≤ form as:
  //   −V(s) + γ Σ_{s'} T(s'|s,a) V(s')  ≤  − Σ_{s'} T(s'|s,a) r(s,a,s')
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  for (let s = 0; s < N; s++) {
    if (mdp.isTerminal && mdp.isTerminal(s)) {
      // Pin V(s) = terminalReward(s).
      const tr = mdp.terminalReward ? mdp.terminalReward(s) : 0;
      // Two ≤ inequalities encode the equality:
      //   V(s) ≤ tr  and  −V(s) ≤ −tr
      const row1 = new Array(N).fill(0); row1[s] = 1; A_ub.push(row1); b_ub.push(tr);
      const row2 = new Array(N).fill(0); row2[s] = -1; A_ub.push(row2); b_ub.push(-tr);
      continue;
    }
    const A = mdp.numActions(s);
    for (let a = 0; a < A; a++) {
      const outcomes = mdp.outcomes(s, a);
      if (outcomes.length === 0) continue;
      const row = new Array(N).fill(0);
      let rhs = 0;
      row[s] = -1;
      for (const o of outcomes) {
        row[o.nextState] += gamma * o.prob;
        rhs -= o.prob * o.reward;
      }
      A_ub.push(row);
      b_ub.push(rhs);
    }
  }

  return {
    sense: 'min',
    c: mu.slice(),
    A_ub, b_ub,
    lb, ub,
    varNames: Array.from({length: N}, (_, s) =>
      mdp.stateLabel ? `V(${mdp.stateLabel(s)})` : `V(s${s})`),
  };
}

/** Solve a finite MDP via its LP formulation. */
export function solveMDPAsLP(
  mdp: MDPSpec,
  gamma: number,
  opts: ExternalSolverOptions & InternalSimplexOptions & {stateDist?: number[]} = {},
): MDPLPSolution {
  const lp = buildMDPLP(mdp, gamma, opts.stateDist);
  const sol = solveLP(lp, opts);
  if (sol.status !== 'optimal') {
    throw new Error(`MDP-LP failed with status=${sol.status}: ${sol.message ?? ''}`);
  }
  const V = sol.x.slice();
  // Greedy policy from V: π*(s) = argmax_a Σ T(s'|s,a)[r + γV(s')].
  const N = mdp.numStates;
  const policy: number[] = new Array(N).fill(-1);
  for (let s = 0; s < N; s++) {
    if (mdp.isTerminal && mdp.isTerminal(s)) continue;
    const A = mdp.numActions(s);
    let bestQ = -Infinity;
    let bestA = -1;
    for (let a = 0; a < A; a++) {
      const outcomes = mdp.outcomes(s, a);
      if (outcomes.length === 0) continue;
      let q = 0;
      for (const o of outcomes) q += o.prob * (o.reward + gamma * V[o.nextState]);
      if (q > bestQ + 1e-12) { bestQ = q; bestA = a; }
    }
    policy[s] = bestA;
  }
  return {V, policy, lp: sol};
}

// -----------------------------------------------------------------------------
// (C) LP-assisted rolling-horizon (MPC-style).
//
// Each `replanEvery` ticks, the simulator hands us its current state; we
// build a fresh LP from that state, solve it, and the simulator uses the
// new plan for the next chunk of ticks. This is the standard pattern in
// production scheduling, supply-chain planning, and energy dispatch.
// -----------------------------------------------------------------------------

export interface RollingHorizonStep<S, M> {
  tickStart: number;
  state: S;
  plan: LPSolution;
  metrics: M;
}

export function lpRollingHorizon<S, M>(
  initState: S,
  buildLP: (state: S, ticksLeft: number) => LPProblem,
  step: (state: S, plan: LPSolution, ticksToRun: number) => {nextState: S; metrics: M},
  totalTicks: number,
  replanEvery: number,
  opts: ExternalSolverOptions & InternalSimplexOptions = {},
): RollingHorizonStep<S, M>[] {
  const log: RollingHorizonStep<S, M>[] = [];
  let state = initState;
  let t = 0;
  while (t < totalTicks) {
    const ticksLeft = totalTicks - t;
    const lp = buildLP(state, ticksLeft);
    const plan = solveLP(lp, opts);
    if (plan.status !== 'optimal') {
      throw new Error(`rolling LP failed at t=${t}: ${plan.status} ${plan.message ?? ''}`);
    }
    const ticksToRun = Math.min(replanEvery, ticksLeft);
    const result = step(state, plan, ticksToRun);
    log.push({tickStart: t, state, plan, metrics: result.metrics});
    state = result.nextState;
    t += ticksToRun;
  }
  return log;
}
