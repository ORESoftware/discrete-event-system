// RUST MIGRATION:
// - Target: src/bin/validate_optimization_as_des.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert CheckRow and scenario fixtures to nominal structs, and keep each optimization family behind migrated module APIs.
// - Pure assertion helpers stay private; DES-wrapped algorithm calls may later become trait implementations over a common optimizer interface.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-optimization-as-des.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. End-to-end validation that the four base-class hierarchies
// (SA / GA / Q-learning / PPO as DES) behave correctly on ground-truthed problems.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - `CheckRow` accumulator -> a `struct CheckRow { name, passed, detail }`.
//   - solver RNG -> inject `SeededRandom` (shared::capabilities).
//   - `console.log` PASS/FAIL + `process.exit` -> `println!` / `std::process::exit`.
// =============================================================================

// =============================================================================
// runners/validate-optimization-as-des.ts — end-to-end validation that
// the four base-class hierarchies produce algorithmically correct
// behaviour on small, ground-truthed problems.
// =============================================================================

import {runTSPSADES, runTSPHillClimberDES} from '../general/sa-des';
import {runTSPGADES} from '../general/ga-des';
import {runQLearningDES} from '../general/qlearning-des';
import {runPPODES} from '../general/ppo-des';
import {
  buildPentagonTSP, buildRandomTSP, tourLength, isPermutation, heldKarpExact,
} from '../general/genetic-tsp';
import {GridWorld, Corridor, evalPolicy} from '../general/rl-environments';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// =============================================================================
// SA validation
// =============================================================================

console.log('\n=== SA (SingleStateOptimizer leaf) ===');
{
  // Pentagon → exact reachable.
  for (const seed of [1, 2, 3, 4, 5]) {
    const inst = buildPentagonTSP(5, 50);
    const opt = tourLength(inst, [0, 1, 2, 3, 4]);
    const sa = runTSPSADES(inst, {
      cooling: {kind: 'geometric', T0: 50, alpha: 0.998}, maxIterations: 3000, seed,
    });
    check(`SA seed=${seed} pentagon optimum`, Math.abs(sa.bestCost - opt) < 1e-9,
      `cost=${sa.bestCost.toFixed(4)} opt=${opt.toFixed(4)}`);
    check(`SA seed=${seed} valid tour`, isPermutation(sa.bestTour, inst.n));
  }
}
{
  // n=10 random vs Held-Karp.
  const inst = buildRandomTSP(10, 23);
  const exact = heldKarpExact(inst);
  const sa = runTSPSADES(inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.999}, maxIterations: 8000, seed: 7,
  });
  check('SA n=10 within 5% of Held-Karp',
    sa.bestCost <= exact.length * 1.05,
    `cost=${sa.bestCost.toFixed(4)} HK=${exact.length.toFixed(4)} gap=${((sa.bestCost / exact.length - 1) * 100).toFixed(2)}%`);
}
{
  // bestHistory monotone.
  const inst = buildRandomTSP(10, 23);
  const sa = runTSPSADES(inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.999}, maxIterations: 4000, seed: 1,
  });
  let monotone = true;
  for (let i = 1; i < sa.bestHistory.length; i++) {
    if (sa.bestHistory[i] > sa.bestHistory[i - 1] + 1e-12) { monotone = false; break; }
  }
  check('SA bestHistory monotone non-increasing', monotone);
}
{
  // Reproducibility from seed.
  const inst = buildRandomTSP(8, 5);
  const a = runTSPSADES(inst, {cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, maxIterations: 2000, seed: 42});
  const b = runTSPSADES(inst, {cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, maxIterations: 2000, seed: 42});
  check('SA seed reproducibility (cost)', a.bestCost === b.bestCost);
  check('SA seed reproducibility (tour)', a.bestTour.join(',') === b.bestTour.join(','));
}

// =============================================================================
// HC validation
// =============================================================================

console.log('\n=== HC (SingleStateOptimizer override) ===');
{
  // Hill climbing: every accepted move is an improvement.
  const inst = buildRandomTSP(15, 11);
  const hc = runTSPHillClimberDES(inst, {
    cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, maxIterations: 5000, seed: 1,
  });
  check('HC accepted == improvements', hc.acceptedCount === hc.improveCount,
    `accepted=${hc.acceptedCount} improvements=${hc.improveCount}`);
  // HC should never INCREASE current cost.
  let nonIncreasing = true;
  for (let i = 1; i < hc.currentHistory.length; i++) {
    if (hc.currentHistory[i] > hc.currentHistory[i - 1] + 1e-12) { nonIncreasing = false; break; }
  }
  check('HC currentHistory monotone non-increasing', nonIncreasing);
}

// =============================================================================
// GA validation
// =============================================================================

console.log('\n=== GA (PopulationOptimizer leaf) ===');
{
  for (const seed of [1, 2, 3]) {
    const inst = buildPentagonTSP(5, 50);
    const opt = tourLength(inst, [0, 1, 2, 3, 4]);
    const ga = runTSPGADES(inst, {popSize: 30, numGenerations: 60, seed});
    check(`GA seed=${seed} pentagon optimum`, Math.abs(ga.bestLength - opt) < 1e-9,
      `len=${ga.bestLength.toFixed(4)} opt=${opt.toFixed(4)}`);
  }
  const inst = buildRandomTSP(10, 23);
  const exact = heldKarpExact(inst);
  const ga = runTSPGADES(inst, {popSize: 60, numGenerations: 200, seed: 1, init: 'nearest-neighbor'});
  check('GA n=10 within 5% of Held-Karp',
    ga.bestLength <= exact.length * 1.05,
    `len=${ga.bestLength.toFixed(4)} HK=${exact.length.toFixed(4)}`);
  let monotone = true;
  for (let i = 1; i < ga.bestHistory.length; i++) {
    if (ga.bestHistory[i] > ga.bestHistory[i - 1] + 1e-12) { monotone = false; break; }
  }
  check('GA bestHistory monotone (elitism)', monotone);
  // mean ≥ best at every generation.
  let meanCheck = true;
  for (let i = 0; i < ga.bestHistory.length; i++) {
    if (ga.meanHistory[i] < ga.bestHistory[i] - 1e-9) { meanCheck = false; break; }
  }
  check('GA meanHistory ≥ bestHistory pointwise', meanCheck);
}
{
  const inst = buildRandomTSP(8, 5);
  const a = runTSPGADES(inst, {popSize: 30, numGenerations: 50, seed: 42});
  const b = runTSPGADES(inst, {popSize: 30, numGenerations: 50, seed: 42});
  check('GA seed reproducibility', a.bestLength === b.bestLength
    && a.bestTour.join(',') === b.bestTour.join(','));
}

// =============================================================================
// Q-learning validation
// =============================================================================

console.log('\n=== Q-learning (RLAgentStation leaf) ===');
{
  const env = new GridWorld();
  const opt = env.optimalV(0.95);
  for (const seed of [1, 2, 3]) {
    const ql = runQLearningDES(env, {
      numEpisodes: 600, alpha: 0.3, gamma: 0.95,
      epsilon: 0.8, epsilonDecay: 0.995, epsilonMin: 0.05,
      maxStepsPerEpisode: 50, seed,
    });
    const v0 = Math.max(...ql.Q[0]);
    check(`Q-learning seed=${seed} V(0) close to optimal`,
      Math.abs(v0 - opt.V[0]) < 0.05,
      `learned=${v0.toFixed(3)} opt=${opt.V[0].toFixed(3)}`);
    const evalQ = evalPolicy(env, (s) => ql.policy[s], {numEpisodes: 100, maxStepsPerEpisode: 50, gamma: 0.95});
    check(`Q-learning seed=${seed} greedy 100% success`, evalQ.successRate === 1);
    check(`Q-learning seed=${seed} mean return matches V*(0)`,
      Math.abs(evalQ.meanReturn - opt.V[0]) < 0.01);
  }
}

// =============================================================================
// PPO validation
// =============================================================================

console.log('\n=== PPO (PolicyGradientAgent + PolicyUpdateStation leaf) ===');
{
  const cor = new Corridor(8);
  const opt = cor.optimalV(0.95);
  for (const seed of [1, 2, 3]) {
    const ppo = runPPODES(cor, {
      totalSteps: 10_000, rolloutLen: 64,
      numEpochs: 6, miniBatchSize: 16,
      policyLr: 0.05, valueLr: 0.1,
      gamma: 0.95, lambda: 0.95, clipEps: 0.2,
      entropyCoef: 0.01, maxStepsPerEpisode: 30, seed,
    });
    check(`PPO seed=${seed} V(0) close to optimal`,
      Math.abs(ppo.V[0] - opt.V[0]) < 0.1,
      `learned=${ppo.V[0].toFixed(3)} opt=${opt.V[0].toFixed(3)}`);
    check(`PPO seed=${seed} action(0) is right (=1)`, ppo.policy[0] === 1);
    const evalP = evalPolicy(cor, (s) => ppo.policy[s], {numEpisodes: 50, maxStepsPerEpisode: 30, gamma: 0.95});
    check(`PPO seed=${seed} greedy 100% success`, evalP.successRate === 1);
    check(`PPO seed=${seed} mean return matches V*(0)`,
      Math.abs(evalP.meanReturn - opt.V[0]) < 0.05);
  }
}
{
  // PPO should run ~ totalSteps/rolloutLen updates.
  const cor = new Corridor(8);
  const ppo = runPPODES(cor, {
    totalSteps: 5_000, rolloutLen: 100,
    numEpochs: 4, miniBatchSize: 16,
    policyLr: 0.05, valueLr: 0.1,
    gamma: 0.95, lambda: 0.95, clipEps: 0.2,
    entropyCoef: 0.01, maxStepsPerEpisode: 30, seed: 1,
  });
  check('PPO updates ≈ steps / rolloutLen',
    Math.abs(ppo.totalUpdates - 5000 / 100) <= 2,
    `updates=${ppo.totalUpdates}`);
}

// =============================================================================
// SUMMARY
// =============================================================================

const passed = checks.filter(c => c.passed).length;
const failed = checks.length - passed;
console.log(`\n=== validate-optimization-as-DES summary: ${passed}/${checks.length} passed, ${failed} failed`);
if (failed > 0) {
  console.log('Failures:');
  for (const c of checks) if (!c.passed) console.log('  - ' + c.name + (c.detail ? ': ' + c.detail : ''));
  process.exit(1);
}
