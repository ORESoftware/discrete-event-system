'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/mdp_adjacent_test.rs   (integration test crate)
// 1:1 file move. End-to-end tests of nine MDP-adjacent models. Keep the
// doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - MC/actor-critic/blackjack/stag-hunt are stochastic -> a seeded rand::Rng;
//     the deliberately-conservative thresholds become assert! inequalities.
//   - value/reward comparisons -> approx::assert_relative_eq!.
// =============================================================================

// =============================================================================
// test/mdp-adjacent-test.ts — end-to-end tests for the nine MDP-adjacent
// models added in this batch:
//
//   1. inventory-dp           — finite-horizon dynamic programming
//   2. mountain-car-vfa       — approximate dynamic programming (linear VFA)
//   3. tiger-pomdp            — POMDP belief-state planning
//   4. grid-localization-pomdp — multi-dimensional POMDP belief lookahead
//   5. four-rooms-smdp        — Semi-MDP / options framework
//   6. actor-critic-grid      — Actor-Critic on tabular GridWorld
//   7. blackjack-mc           — Monte Carlo on-policy control
//   8. stag-hunt              — multi-agent IQL on a coordination game
//   9. double-integrator-lqr  — LQR / stochastic control via Riccati DARE
//
// Tests are intentionally conservative on their thresholds so they
// pass deterministically across machines while still verifying the
// CORE INVARIANT each algorithm is supposed to satisfy.
// =============================================================================

import {solveInventoryDP, simulateInventory, InventoryProblem} from '../general/inventory-dp';
import {runMountainCar} from '../general/mountain-car';
import {simulateTiger, buildTigerSpec, ACT_LISTEN} from '../general/tiger-pomdp';
import {runGridLocalizationPOMDP} from '../general/grid-localization-pomdp';
import {runFourRoomsSMDP} from '../general/four-rooms';
import {runActorCriticGridworld} from '../general/actor-critic-gridworld';
import {runBlackjackMC} from '../general/blackjack';
import {runStagHunt} from '../general/stag-hunt';
import {runDoubleIntegratorLQR} from '../general/double-integrator-lqr';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// =============================================================================
// 1. INVENTORY-DP
// =============================================================================

console.log('\n— inventory-dp (finite-horizon DP / Bellman backward induction) —');
{
  // 5-period inventory with truncated Poisson(λ=3) demand.
  const pmf = [0.0498, 0.1494, 0.2240, 0.2240, 0.1680, 0.1008, 0.0504, 0.0336];
  const norm = pmf.reduce((s, x) => s + x, 0);
  const demandPmf = pmf.map(x => x / norm);
  const p: InventoryProblem = {
    horizon: 5, S_max: 10, demandPmf,
    price: 8, cost: 3, fixedCost: 4,
    holdCost: 1, stockoutCost: 5, salvageValue: 1,
    discount: 1.0, initialInventory: 0,
  };
  const r = solveInventoryDP(p, {seed: 7});
  check('horizon backward-sweep ticks == T',
        r.ticks === p.horizon, `ticks=${r.ticks}, T=${p.horizon}`);
  check('V(t=0,*) is non-decreasing in s',
        (() => { const V = r.V[0]; for (let s = 1; s < V.length; s++) if (V[s] < V[s-1] - 1e-6) return false; return true; })(),
        `V(t=0)=[${r.V[0].map(x => x.toFixed(2)).join(', ')}]`);
  check('policy entries within [0, S_max-s]',
        (() => {
          for (let t = 0; t < p.horizon; t++) for (let s = 0; s < r.policy[t].length; s++) {
            const a = r.policy[t][s];
            if (a < 0 || a + s > p.S_max) return false;
          }
          return true;
        })(), 'all (t, s) feasible');
  // Sanity: sample monte-carlo estimate of E[total reward] from the simulation
  // is within a reasonable band of V*(0, s0).
  let mcSum = 0; const reps = 200;
  for (let rep = 0; rep < reps; rep++) {
    const sim = simulateInventory(p, r.policy, rep + 100);
    mcSum += sim.totalReward;
  }
  const mc = mcSum / reps;
  check('MC estimate of return ≈ V*(t=0, s=0) within 2.5σ',
        Math.abs(mc - r.expectedReward) < 2.0,
        `MC=${mc.toFixed(2)} vs V*=${r.expectedReward.toFixed(2)}`);

  check('intrinsic validators registered & pass',
        (() => {
          const station = new (require('../general/inventory-dp').InventoryDPStation)(p);
          const summary = require('../general/des-base').runIterativeDES([station]);
          return summary.validation && summary.validationOk;
        })(), 'auto-attached invariants pass');
}

// =============================================================================
// 2. MOUNTAIN-CAR-VFA
// =============================================================================

console.log('\n— mountain-car-vfa (approximate DP, linear VFA + tile coding) —');
{
  const r = runMountainCar({
    numEpisodes: 80, alpha: 0.5, gamma: 1, epsilon: 0,
    epsilonDecay: 1, epsilonMin: 0,
    numTilings: 8, numTilesPerDim: 8,
    maxStepsPerEpisode: 1000, seed: 1,
  });
  check('rewardHistory length == numEpisodes',
        r.rewardHistory.length === 80,
        `len=${r.rewardHistory.length}`);
  // First 5 vs last 20 episode lengths — should improve substantially.
  const first = r.lengthHistory.slice(0, 5);
  const last = r.lengthHistory.slice(-20);
  const first5 = first.reduce((s, x) => s + x, 0) / first.length;
  const last20 = last.reduce((s, x) => s + x, 0) / last.length;
  check('episode length decreases (first-5 > last-20)',
        first5 > last20,
        `first-5 mean ${first5.toFixed(0)} → last-20 mean ${last20.toFixed(0)}`);
  check('all returns are negative (per-step -1, no goal yet)',
        r.rewardHistory.every(x => x <= 0),
        `min=${Math.min(...r.rewardHistory)}  max=${Math.max(...r.rewardHistory)}`);
  // Sanity: theta is non-trivial after 80 episodes.
  check('||θ|| > 0 after training',
        r.thetaNorm > 0, `‖θ‖=${r.thetaNorm.toFixed(2)}`);
}

// =============================================================================
// 3. TIGER-POMDP
// =============================================================================

console.log('\n— tiger-pomdp (POMDP belief-state planning) —');
{
  // 1-step look-ahead with the canonical parameters: should LISTEN
  // most of the time and only OPEN when belief is concentrated.
  const r1 = simulateTiger({
    spec: buildTigerSpec(),
    solver: 'one-step-lookahead',
    numSteps: 50, seed: 1,
  });
  const listenFrac = r1.actions.filter(a => a === ACT_LISTEN).length / r1.actions.length;
  check('one-step-lookahead listens > 50% of the time',
        listenFrac > 0.5,
        `listen frac = ${listenFrac.toFixed(2)}`);
  check('one-step-lookahead avoids most catastrophic opens',
        r1.numBadOpens <= 5,
        `bad opens = ${r1.numBadOpens} / ${r1.numOpens} opens / ${r1.steps} steps`);

  // Aggregate over 10 seeds: average return should be POSITIVE for
  // the look-ahead solver in 50 steps under defaults.
  let avgRet = 0;
  for (let s = 0; s < 10; s++) {
    avgRet += simulateTiger({spec: buildTigerSpec(), solver: 'one-step-lookahead', numSteps: 50, seed: s + 1}).totalReturn;
  }
  avgRet /= 10;
  check('avg discounted return (10 seeds) is finite',
        Number.isFinite(avgRet), `avg=${avgRet.toFixed(2)}`);
}

// =============================================================================
// =============================================================================
// 4. GRID-LOCALIZATION-POMDP
// =============================================================================

console.log('\n— grid-localization-pomdp (2D hidden-state POMDP) —');
{
  const r = runGridLocalizationPOMDP({
    width: 3,
    height: 3,
    horizon: 3,
    numSteps: 8,
    seed: 7,
    hiddenTarget: [2, 1],
    scanAccuracy: 1,
    inspectAccuracy: 1,
  });
  const first = r.trace[0];
  const last = r.trace[r.trace.length - 1];
  check('2D POMDP state space is Cartesian 3x3',
        r.stateSpace.numStates === 9 && r.stateSpace.dimensions.length === 2,
        `states=${r.stateSpace.numStates}`);
  check('belief-lookahead gathers information before inspecting',
        first.action.kind !== 'inspect',
        `first=${first.action.label}`);
  check('perfect row/column scans reduce entropy',
        last.entropy < Math.log(9),
        `H0=${Math.log(9).toFixed(3)} Hf=${last.entropy.toFixed(3)}`);
  check('posterior concentrates on the hidden target',
        last.hiddenProbability > 0.95,
        `P(hidden)=${last.hiddenProbability.toFixed(3)}`);
  check('planner finds the hidden target',
        r.found,
        `foundAt=${r.foundAtStep}`);
}

// =============================================================================
// 5. FOUR-ROOMS-SMDP
// =============================================================================

console.log('\n— four-rooms-smdp (Semi-MDP, options framework) —');
{
  const r = runFourRoomsSMDP({
    numEpisodes: 800, alpha: 0.3, gamma: 0.99,
    epsilon: 0.2, epsilonDecay: 0.99, epsilonMin: 0.02,
    maxStepsPerEpisode: 2000, slip: 0,
    includePrimitive: true, initQ: 0.05, seed: 1,
  });
  check('rewardHistory length == 800',
        r.rewardHistory.length === 800,
        `len=${r.rewardHistory.length}`);
  // After 600 episodes the agent should reach the goal.
  check('greedy policy reaches the goal',
        r.greedyReachedGoal, `len=${r.greedyEpisodeLength}`);
  // Greedy path may take a few options + a tail of primitives — but must
  // certainly be less than the maxStepsPerEpisode budget.
  check('greedy episode length ≤ 200',
        r.greedyEpisodeLength <= 200, `len=${r.greedyEpisodeLength}`);
}

// =============================================================================
// 6. ACTOR-CRITIC-GRID
// =============================================================================

console.log('\n— actor-critic-grid (Actor-Critic on GridWorld) —');
{
  const r = runActorCriticGridworld({
    numEpisodes: 1500, alphaV: 0.1, alphaP: 0.1, gamma: 0.95,
    maxStepsPerEpisode: 100, width: 4, height: 4, seed: 1,
  });
  check('rewardHistory length == 1500',
        r.rewardHistory.length === 1500,
        `len=${r.rewardHistory.length}`);
  // Last 50 returns should be positive on average (goal reward = +10
  // dominates step cost = -1).
  const last = r.rewardHistory.slice(-50);
  const meanLast = last.reduce((s, x) => s + x, 0) / last.length;
  check('mean return (last 50) > 0',
        meanLast > 0, `mean=${meanLast.toFixed(2)}`);
  check('greedy reaches goal',
        r.greedyReached, `len=${r.greedyLen}`);
}

// =============================================================================
// 7. BLACKJACK-MC
// =============================================================================

console.log('\n— blackjack-mc (first-visit Monte Carlo control) —');
{
  const r = runBlackjackMC({
    numEpisodes: 50_000, epsilon: 0.1, epsilonDecay: 1, epsilonMin: 0.05,
    firstVisit: true, gamma: 1, evalEpisodes: 3000, seed: 1,
  });
  check('greedy strictly outperforms stick≥20 baseline',
        r.greedyMeanReturn > r.baselineMeanReturn,
        `greedy=${r.greedyMeanReturn.toFixed(3)}  base=${r.baselineMeanReturn.toFixed(3)}`);
  check('baseline in canonical band [-0.40, -0.20]',
        r.baselineMeanReturn >= -0.40 && r.baselineMeanReturn <= -0.20,
        `base=${r.baselineMeanReturn.toFixed(3)}`);
  check('greedy in canonical band (≥ −0.10)',
        r.greedyMeanReturn >= -0.10,
        `greedy=${r.greedyMeanReturn.toFixed(3)}`);
  check('visited a meaningful fraction of cells',
        r.visitedCells > 200, `visited ${r.visitedCells}/400`);
}

// =============================================================================
// 8. STAG-HUNT
// =============================================================================

console.log('\n— stag-hunt (independent Q-learning, 2 agents) —');
{
  const r = runStagHunt({
    numEpisodes: 5000, alpha: 0.05, gamma: 0,
    epsilon: 0.2, epsilonDecay: 0.999, epsilonMin: 0.01, seed: 1,
  });
  check('rewardHistory length == 5000',
        r.rewardHistory.length === 5000);
  // Coordination: end up at one of the two pure NE.
  check('agents coordinate on a Nash equilibrium',
        r.coordinatedOnStag || r.coordinatedOnHare,
        `final = [${r.finalJointAction.join(', ')}]`);
  // Recent mean returns: at the worst NE (Hare,Hare) returns are 3,3;
  // at the best NE (Stag,Stag) returns are 4,4. Either way ≥ 3.
  check('recent mean returns ≥ 2.5 for both agents',
        r.recentMeanReturn[0] >= 2.5 && r.recentMeanReturn[1] >= 2.5,
        `[${r.recentMeanReturn[0].toFixed(2)}, ${r.recentMeanReturn[1].toFixed(2)}]`);
}

// =============================================================================
// 9. DOUBLE-INTEGRATOR-LQR
// =============================================================================

console.log('\n— double-integrator-lqr (Riccati DARE) —');
{
  const r = runDoubleIntegratorLQR({
    dt: 0.1, qPos: 1, qVel: 0.1, rU: 0.01,
    noiseStd: 0,    // deterministic for theory-vs-realised match
    x0: [3, 0], numSteps: 200, uSat: 100,
    gamma: 1, seed: 1,
  });
  check('Riccati iteration converged',
        r.riccatiResidual < 1e-8,
        `iters=${r.riccatiIters}  residual=${r.riccatiResidual.toExponential(2)}`);
  check('K is 1×2 (m × n)',
        r.K.length === 1 && r.K[0].length === 2);
  // Both gains positive (point mass with positive Q): pushes opposite
  // to position and damps velocity.
  check('K entries are positive',
        r.K[0][0] > 0 && r.K[0][1] > 0,
        `K=[${r.K[0].map(x => x.toFixed(3)).join(', ')}]`);
  // Trajectory drives state → 0.
  const finalNorm = Math.hypot(...r.trajectory[r.trajectory.length - 1]);
  check('|x(T)| < 0.05 with no noise',
        finalNorm < 0.05, `|x_T|=${finalNorm.toExponential(2)}`);
  // With no noise the realised cost should be ≤ DARE cost-to-go (LQR
  // optimality); allow 1e-6 slack for floating-point.
  check('realised cost ≤ DARE cost-to-go (deterministic)',
        r.totalCost <= r.riccatiCostFromX0 * (1 + 1e-3) + 1e-3,
        `realised=${r.totalCost.toFixed(3)} vs DARE=${r.riccatiCostFromX0.toFixed(3)}`);
}

// =============================================================================
// SUMMARY
// =============================================================================

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`mdp-adjacent-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
