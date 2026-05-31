// RUST MIGRATION: target src/bin/main_mdp_lp.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// main-mdp-lp.ts — solve an MDP via its LP formulation.
//
// THE DEEP CONNECTION
// ───────────────────
// A finite-state, finite-action, discounted MDP can be solved EXACTLY as
// a linear program. The Bellman optimality equation
//
//   V*(s) = max_a Σ_{s'} T(s'|s,a) · [ r(s,a,s') + γ · V*(s') ]
//
// has the LP characterisation
//
//   min  Σ_s μ_s · V(s)         (any μ ≻ 0; we use uniform)
//   s.t. V(s) ≥ Σ_{s'} T(s'|s,a) · [ r(s,a,s') + γ · V(s') ]    ∀ s, a
//
// At the optimum, exactly one constraint per non-terminal state is
// binding, and the binding-action gives the greedy policy π*. The
// value function V* matches value-iteration to machine precision.
//
// This demo solves a small inventory-control MDP three ways:
//   (1) generic value iteration                  →  V_VI,  π_VI
//   (2) MDP-as-LP via in-process simplex         →  V_LP,  π_LP
//   (3) MDP-as-LP via scipy:highs (interior-point or simplex via env)
//
// All three should produce the SAME V* and π*. The chain
//
//     real system  →  DES simulator  →  MDP abstraction  →  LP formulation  →  simplex / interior-point
//
// is one of the deepest connections between OR, control theory, and RL.
//
// USAGE
// ─────
//   node dist/des/main-mdp-lp.js
//   LP_SOLVER=scipy:highs-ipm node dist/des/main-mdp-lp.js
//   LP_SOLVER=internal node dist/des/main-mdp-lp.js
//   PROBLEM=inventory node dist/des/main-mdp-lp.js
//   PROBLEM=chain node dist/des/main-mdp-lp.js
//   PROBLEM=gridworld node dist/des/main-mdp-lp.js
// =============================================================================

import {MDPSpec, valueIteration} from './general/value-iteration';
import {solveMDPAsLP} from './general/des-lp-bridge';

// -----------------------------------------------------------------------------
// Problem 1: inventory control.
//
// State: 0..S_max units in stock.
// Action: order 0..K extra (delivered before next-day demand).
// Demand: Poisson(λ), capped at 10. Sales = min(demand, stock+order).
// Cost: order cost = c·k + holding cost = h·stock_after_order
//        revenue   = p·sold − stockout cost = q·max(0, demand−(stock+order))
//
// Reward (we MAXIMISE) = revenue − cost − stockout penalty.
// -----------------------------------------------------------------------------
function buildInventoryMDP(): MDPSpec {
  const S_max = 10;
  const K_max = 6;
  const λ = 4;
  const p = 6;     // selling price
  const c = 2;     // order cost per unit
  const h = 1;     // holding cost per unit per period
  const q = 5;     // stockout penalty per missed unit

  // Truncated Poisson PMF on demand 0..D_cap.
  const D_cap = 10;
  const pmf: number[] = [];
  let factorial = 1;
  for (let k = 0; k <= D_cap; k++) {
    if (k > 0) factorial *= k;
    pmf.push(Math.pow(λ, k) * Math.exp(-λ) / factorial);
  }
  const norm = pmf.reduce((s, v) => s + v, 0);
  for (let k = 0; k <= D_cap; k++) pmf[k] /= norm;

  const numStates = S_max + 1;
  return {
    numStates,
    numActions: () => K_max + 1,
    outcomes: (s: number, a: number) => {
      const stockAfterOrder = Math.min(S_max, s + a);
      const orderCost = c * a;
      const holdCost = h * stockAfterOrder;
      const out = [];
      for (let d = 0; d <= D_cap; d++) {
        if (pmf[d] === 0) continue;
        const sold = Math.min(stockAfterOrder, d);
        const stockout = d - sold;
        const revenue = p * sold;
        const stockoutPenalty = q * stockout;
        const reward = revenue - orderCost - holdCost - stockoutPenalty;
        const nextState = stockAfterOrder - sold;
        out.push({prob: pmf[d], reward, nextState});
      }
      return out;
    },
    stateLabel: (s) => `inv=${s}`,
    actionLabel: (a) => `order=${a}`,
  };
}

// -----------------------------------------------------------------------------
// Problem 2: chain MDP — 5 states in a line, action = left or right,
//            reward only at right endpoint.
// -----------------------------------------------------------------------------
function buildChainMDP(): MDPSpec {
  const N = 5;
  return {
    numStates: N,
    numActions: () => 2,
    outcomes: (s, a) => {
      if (s === N - 1) return [{prob: 1, reward: 0, nextState: s}];
      const target = a === 1 ? Math.min(N - 1, s + 1) : Math.max(0, s - 1);
      const reward = target === N - 1 ? 1 : 0;
      return [{prob: 1, reward, nextState: target}];
    },
    isTerminal: (s) => s === N - 1,
    terminalReward: () => 0,
    stateLabel: (s) => `s${s}`,
    actionLabel: (a) => a === 0 ? 'left' : 'right',
  };
}

// -----------------------------------------------------------------------------
// Problem 3: 4×4 grid-world with stochastic transitions.
// -----------------------------------------------------------------------------
function buildGridMDP(): MDPSpec {
  const W = 4, H = 4;
  const N = W * H;
  const idx = (x: number, y: number) => y * W + x;
  const move = (s: number, a: number): number => {
    const x = s % W, y = Math.floor(s / W);
    if (a === 0) return idx(x, Math.max(0, y - 1));        // up
    if (a === 1) return idx(x, Math.min(H - 1, y + 1));     // down
    if (a === 2) return idx(Math.max(0, x - 1), y);         // left
    return idx(Math.min(W - 1, x + 1), y);                  // right
  };
  const slip: Record<number, [number, number]> = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]};
  const goal = idx(3, 3);
  const pit = idx(1, 2);
  return {
    numStates: N,
    numActions: () => 4,
    outcomes: (s, a) => {
      if (s === goal || s === pit) return [{prob: 1, reward: 0, nextState: s}];
      const intended = move(s, a);
      const [s1, s2] = slip[a];
      const sl1 = move(s, s1), sl2 = move(s, s2);
      const r = (sp: number) => sp === goal ? 1 : (sp === pit ? -1 : -0.04);
      return [
        {prob: 0.8, reward: r(intended), nextState: intended},
        {prob: 0.1, reward: r(sl1),      nextState: sl1},
        {prob: 0.1, reward: r(sl2),      nextState: sl2},
      ];
    },
    isTerminal: (s) => s === goal || s === pit,
    terminalReward: () => 0,
    stateLabel: (s) => `(${s % W},${Math.floor(s / W)})`,
    actionLabel: (a) => ['up', 'down', 'left', 'right'][a],
  };
}

// -----------------------------------------------------------------------------
// Helpers.
// -----------------------------------------------------------------------------
function maxAbs(u: ArrayLike<number>, v: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < u.length; i++) m = Math.max(m, Math.abs(u[i] - v[i]));
  return m;
}

// -----------------------------------------------------------------------------
async function main(): Promise<void> {
  const which = process.env.PROBLEM ?? 'inventory';
  const gamma = Number(process.env.GAMMA ?? 0.95);
  const solverEnv = process.env.LP_SOLVER ?? 'scipy:highs';

  const builder: Record<string, () => MDPSpec> = {
    inventory: buildInventoryMDP,
    chain:     buildChainMDP,
    gridworld: buildGridMDP,
  };
  if (!builder[which]) {
    console.error(`unknown PROBLEM='${which}'; expected one of: ${Object.keys(builder).join(', ')}`);
    process.exit(2);
  }

  const mdp = builder[which]();
  console.log(`# MDP-as-LP: solving '${which}' MDP with ${mdp.numStates} states, γ=${gamma}`);
  console.log(`#   LP_SOLVER=${solverEnv}`);
  console.log('');

  // ---- 1. Value iteration (reference) ----
  const t0 = Date.now();
  const vi = valueIteration(mdp, {gamma, tol: 1e-12, maxIter: 100000});
  const tVI = Date.now() - t0;
  console.log(`# Value iteration (reference):`);
  console.log(`#   iterations  = ${vi.iterations}`);
  console.log(`#   final delta = ${vi.finalDelta.toExponential(3)}`);
  console.log(`#   wall time   = ${tVI}ms`);
  console.log('');

  // ---- 2. MDP-as-LP ----
  const t1 = Date.now();
  const lp = solveMDPAsLP(mdp, gamma);
  const tLP = Date.now() - t1;
  console.log(`# MDP-as-LP:`);
  console.log(`#   solver      = ${lp.lp.solver}`);
  console.log(`#   iterations  = ${lp.lp.iters}`);
  console.log(`#   wall time   = ${tLP}ms (incl. Python startup if external)`);
  console.log('');

  // ---- 3. Compare ----
  const dV = maxAbs(vi.V, lp.V);
  console.log(`# Comparison:`);
  console.log(`#   max|V_LP − V_VI|  = ${dV.toExponential(3)}`);
  let polMatch = true;
  for (let s = 0; s < mdp.numStates; s++) {
    if (mdp.isTerminal && mdp.isTerminal(s)) continue;
    if (vi.policy[s] !== lp.policy[s]) polMatch = false;
  }
  console.log(`#   π_LP ≡ π_VI?      = ${polMatch}`);
  console.log('');

  // ---- 4. Pretty-print V* and π* ----
  const N = mdp.numStates;
  console.log(`# Optimal V* and π*:`);
  console.log(`#   ${'state'.padEnd(14)} ${'V*'.padStart(12)}  ${'π*'.padEnd(12)}`);
  for (let s = 0; s < N; s++) {
    const lbl = (mdp.stateLabel ? mdp.stateLabel(s) : `s${s}`).padEnd(14);
    const v = lp.V[s].toFixed(6).padStart(12);
    const a = lp.policy[s];
    const al = a < 0 ? '(terminal)' : (mdp.actionLabel ? mdp.actionLabel(a) : `a${a}`);
    console.log(`#   ${lbl} ${v}  ${al}`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
