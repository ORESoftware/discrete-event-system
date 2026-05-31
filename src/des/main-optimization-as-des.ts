'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-optimization-as-des.rs   (fn main)
// 1:1 file move. Runs SA / hill-climb / GA / Q-learning / PPO as DES on shared
// problems and prints one comparison table.
//
// Conversion notes (file-specific):
//   - imports many general/* algorithm-as-DES modules -> use crate::des::
//     general::{sa_des, ga_des, qlearning_des, ppo_des, genetic_tsp,
//     rl_environments}.
//   - all algorithms randomise -> route through SeededRandom for reproducibility.
//   - top-level studies -> fn main.
// =============================================================================

// =============================================================================
// main-optimization-as-des.ts — runs all four "algorithm-as-DES"
// implementations on small, comparable problems and reports a single
// comparison table.
//
// The four algorithms:
//
//   • Simulated Annealing       (SingleStateOptimizer)
//   • Hill Climber              (SingleStateOptimizer, override accept)
//   • Genetic Algorithm         (PopulationOptimizer)
//   • Q-learning                (RLAgentStation)
//   • PPO (clipped)             (PolicyGradientAgent + PolicyUpdateStation)
//
// All five share the SAME runIterativeDES runner and the SAME DESStation
// channel mechanics. Only the algorithmic hooks differ.
// =============================================================================

import {runTSPSADES, runTSPHillClimberDES} from './general/sa-des';
import {runTSPGADES} from './general/ga-des';
import {runQLearningDES} from './general/qlearning-des';
import {runPPODES} from './general/ppo-des';
import {
  buildPentagonTSP, buildRandomTSP, tourLength, isPermutation, heldKarpExact,
} from './general/genetic-tsp';
import {GridWorld, Corridor, evalPolicy} from './general/rl-environments';

function fmt(x: number, n = 4): string { return Number(x).toFixed(n); }
function pct(x: number): string { return (100 * x).toFixed(2) + '%'; }

// -----------------------------------------------------------------------------
// STUDY 1 — TSP n=5 pentagon (exact optimum known)
// -----------------------------------------------------------------------------

function tspPentagonStudy(): void {
  console.log('\n=== STUDY 1 ─ Pentagon TSP (n=5, exact = perimeter) ─ algorithm comparison');
  const inst = buildPentagonTSP(5, 50);
  const opt = tourLength(inst, [0, 1, 2, 3, 4]);

  const sa = runTSPSADES(inst, {
    cooling: {kind: 'geometric', T0: 50, alpha: 0.998}, maxIterations: 3000, seed: 1,
  });
  const hc = runTSPHillClimberDES(inst, {
    cooling: {kind: 'geometric', T0: 50, alpha: 0.998}, maxIterations: 3000, seed: 1,
  });
  const ga = runTSPGADES(inst, {popSize: 30, numGenerations: 80, seed: 1});

  console.log('  algo                length        gap          ticks  hooks invoked');
  console.log(`  SA                  ${fmt(sa.bestCost)}  ${pct(sa.bestCost / opt - 1)}     ${sa.ticks.toString().padStart(5)}  ${sa.iterations} iter, ${sa.acceptedCount} acc, ${sa.improveCount} impr`);
  console.log(`  HC                  ${fmt(hc.bestCost)}  ${pct(hc.bestCost / opt - 1)}     ${hc.ticks.toString().padStart(5)}  ${hc.iterations} iter, ${hc.acceptedCount} acc`);
  console.log(`  GA                  ${fmt(ga.bestLength)}  ${pct(ga.bestLength / opt - 1)}     ${ga.ticks.toString().padStart(5)}  gens=${ga.generations}`);
  console.log(`  optimum             ${fmt(opt)}`);
  if (!isPermutation(sa.bestTour, inst.n)) throw new Error('SA tour invalid');
  if (!isPermutation(ga.bestTour, inst.n)) throw new Error('GA tour invalid');
}

// -----------------------------------------------------------------------------
// STUDY 2 — TSP n=12 random vs Held-Karp
// -----------------------------------------------------------------------------

function tspRandom12Study(): void {
  console.log('\n=== STUDY 2 ─ Random TSP (n=12) ─ SA / HC / GA vs Held-Karp');
  const inst = buildRandomTSP(12, 17);
  const exact = heldKarpExact(inst);

  const sa = runTSPSADES(inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.998}, maxIterations: 5000, seed: 1,
  });
  const hc = runTSPHillClimberDES(inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.998}, maxIterations: 5000, seed: 1,
  });
  const ga = runTSPGADES(inst, {popSize: 50, numGenerations: 200, seed: 1, init: 'nearest-neighbor'});

  console.log('  algo                length        gap          ticks');
  console.log(`  SA                  ${fmt(sa.bestCost)}  ${pct(sa.bestCost / exact.length - 1)}     ${sa.ticks}`);
  console.log(`  HC                  ${fmt(hc.bestCost)}  ${pct(hc.bestCost / exact.length - 1)}     ${hc.ticks}`);
  console.log(`  GA                  ${fmt(ga.bestLength)}  ${pct(ga.bestLength / exact.length - 1)}     ${ga.ticks}`);
  console.log(`  Held-Karp (exact)   ${fmt(exact.length)}`);
}

// -----------------------------------------------------------------------------
// STUDY 3 — GridWorld via Q-learning
// -----------------------------------------------------------------------------

function gridWorldStudy(): void {
  console.log('\n=== STUDY 3 ─ 4x4 GridWorld ─ Q-learning vs Bellman-optimal V*');
  const env = new GridWorld();
  const opt = env.optimalV(0.95);
  const ql = runQLearningDES(env, {
    numEpisodes: 600, alpha: 0.3, gamma: 0.95,
    epsilon: 0.8, epsilonDecay: 0.995, epsilonMin: 0.05,
    maxStepsPerEpisode: 50, seed: 1,
  });
  const evalQ = evalPolicy(env, (s) => ql.policy[s], {numEpisodes: 200, maxStepsPerEpisode: 100, gamma: 0.95});
  console.log('  state    optimal V*    learned max_a Q[s,a]    optimal a*    learned a*');
  for (let s = 0; s < env.numStates; s++) {
    const v = Math.max(...ql.Q[s]);
    console.log(`  ${s.toString().padStart(5)}    ${fmt(opt.V[s], 3).padStart(10)}    ${fmt(v, 3).padStart(20)}    ${opt.pi[s].toString().padStart(10)}    ${ql.policy[s].toString().padStart(10)}`);
  }
  console.log(`  episodes=${ql.totalEpisodes}  steps=${ql.totalSteps}  ticks=${ql.totalTicks}`);
  console.log(`  greedy success=${pct(evalQ.successRate)}  meanReturn=${fmt(evalQ.meanReturn, 3)}  optimalReturn=${fmt(opt.V[0], 3)}`);
}

// -----------------------------------------------------------------------------
// STUDY 4 — Corridor via PPO
// -----------------------------------------------------------------------------

function corridorStudy(): void {
  console.log('\n=== STUDY 4 ─ Corridor(8) ─ PPO vs Bellman-optimal V*');
  const env = new Corridor(8);
  const opt = env.optimalV(0.95);
  const ppo = runPPODES(env, {
    totalSteps: 10_000, rolloutLen: 64,
    numEpochs: 6, miniBatchSize: 16,
    policyLr: 0.05, valueLr: 0.1,
    gamma: 0.95, lambda: 0.95, clipEps: 0.2,
    entropyCoef: 0.01, maxStepsPerEpisode: 30, seed: 1,
  });
  const evalP = evalPolicy(env, (s) => ppo.policy[s], {numEpisodes: 200, maxStepsPerEpisode: 30, gamma: 0.95});
  console.log('  state    optimal V*    PPO V_φ(s)    optimal a*    PPO a*');
  for (let s = 0; s < env.numStates; s++) {
    console.log(`  ${s.toString().padStart(5)}    ${fmt(opt.V[s], 3).padStart(10)}    ${fmt(ppo.V[s], 3).padStart(10)}    ${opt.pi[s].toString().padStart(10)}    ${ppo.policy[s].toString().padStart(7)}`);
  }
  console.log(`  episodes=${ppo.totalEpisodes}  steps=${ppo.totalSteps}  updates=${ppo.totalUpdates}  ticks=${ppo.totalTicks}`);
  console.log(`  greedy success=${pct(evalP.successRate)}  meanReturn=${fmt(evalP.meanReturn, 3)}  optimalReturn=${fmt(opt.V[0], 3)}`);
}

function main(): void {
  console.log('=== optimization-as-DES — SA, HC, GA, Q-learning, PPO ─ all on the same engine');
  tspPentagonStudy();
  tspRandom12Study();
  gridWorldStudy();
  corridorStudy();
  console.log('\nAll five algorithms are concrete LEAVES of the four algorithm-family');
  console.log('base classes (SingleStateOptimizer, PopulationOptimizer, RLAgentStation,');
  console.log('PolicyGradientAgent) and share the same runIterativeDES runner.');
}

main();
