// RUST MIGRATION:
// - Target: src/bin/validate_soccer.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert sample problem/check data to nominal structs and keep solver result comparisons explicit.
// - Route LP/DES solver variants through migrated traits or enum-backed adapters, leaving tolerance helpers private.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-soccer.rs  (a `fn main` binary; an
//                    `examples/…rs` also works)
// 1:1 file move. Validates the 7v7 rotation problem (LP relaxation, MDP-VI,
// Hungarian bounds, cross-solver agreement, stochastic match).
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - the stochastic-match Welch-t study draws random matches -> inject
//     `SeededRandom`.
//   - `console.log` PASS/FAIL + `process.exit` -> `println!` / `std::process::exit`.
// =============================================================================

// =============================================================================
// runners/validate-soccer.ts — validate the 7v7 rotation problem.
//
// Five studies:
//   1. LP relaxation upper bound = MDP-VI exact optimum (LP is tight)
//   2. State augmentation: memoryless MDP violates fairness, augmented MDP
//      respects it, and the affinity gap = the cost of the constraint
//   3. Cross-solver agreement on the LP relaxation: internal simplex ≡
//      DES-engine simplex ≡ scipy:HiGHS-DS ≡ scipy:HiGHS-IPM
//   4. Hungarian per-period upper bound: greedy-Hungarian ≤ MDP-VI ≤ LP
//   5. Stochastic match: MDP-VI vs random Welch-t test on goal differential
// =============================================================================

import {
  buildSampleSoccerProblem,
  evaluateSchedule, validateScheduleStructure,
  buildSoccerLP,
  policyRandomSchedule, policyGreedyHungarian,
  policyMDPVI, policyMDPVIMemoryless, policyLPRelaxed,
  runManyMatches, welchT,
} from '../general/soccer-rotation';
import {solveLPInternal, solveLPExternal} from '../general/lp';
import {solveLPViaDES} from '../general/lp-des';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}

const problem = buildSampleSoccerProblem({seed: 4242});

// =============================================================================
console.log('\nStudy 1 — LP relaxation upper bound = MDP-VI exact optimum');
// =============================================================================
{
  const lp = policyLPRelaxed(problem);
  const mdp = policyMDPVI(problem);
  const mdpEval = evaluateSchedule(problem, mdp);
  console.log(`    LP upper bound       = ${lp.lpValue.toFixed(6)}`);
  console.log(`    MDP-VI optimum       = ${mdp.optimalValue.toFixed(6)}`);
  console.log(`    LP-rounded affinity  = ${evaluateSchedule(problem, lp.schedule).affinitySum.toFixed(6)}`);
  console.log(`    MDP-VI affinity      = ${mdpEval.affinitySum.toFixed(6)}`);
  check('LP upper bound ≥ MDP optimum (relaxation property)',
        lp.lpValue + 1e-9 >= mdp.optimalValue,
        `Δ = ${(lp.lpValue - mdp.optimalValue).toFixed(6)}`);
  check('MDP optimum = LP upper bound to 1e-6 (LP is tight ⇒ no integrality gap)',
        Math.abs(lp.lpValue - mdp.optimalValue) < 1e-6,
        `|Δ| = ${Math.abs(lp.lpValue - mdp.optimalValue).toExponential(2)}`);
  check('MDP-VI affinity = MDP optimum (consistent)',
        Math.abs(mdpEval.affinitySum - mdp.optimalValue) < 1e-9);
  check('MDP-VI is structurally valid',
        validateScheduleStructure(problem, mdp) === null);
  check('MDP-VI satisfies fairness constraint',
        mdpEval.fairnessOk, `${mdpEval.fairnessViolations.length} violations`);
}

// =============================================================================
console.log('\nStudy 2 — State augmentation: memoryless MDP violates, augmented MDP enforces');
// =============================================================================
{
  const memoryless = policyMDPVIMemoryless(problem);
  const memEval = evaluateSchedule(problem, memoryless.schedule);
  const augmented = policyMDPVI(problem);
  const augEval = evaluateSchedule(problem, augmented);
  console.log(`    state = (t,)              affinity = ${memEval.affinitySum.toFixed(4)}, fairness = ${memEval.fairnessOk}`);
  console.log(`    state = (t, prev_bench)   affinity = ${augEval.affinitySum.toFixed(4)}, fairness = ${augEval.fairnessOk}`);
  console.log(`    cost of fairness          = ${(memEval.affinitySum - augEval.affinitySum).toFixed(4)}`);
  check('memoryless MDP affinity > augmented MDP affinity (relaxation is unconstrained)',
        memEval.affinitySum > augEval.affinitySum,
        `Δ = ${(memEval.affinitySum - augEval.affinitySum).toFixed(4)}`);
  check('memoryless MDP VIOLATES fairness (no history in state ⇒ constraint inexpressible)',
        !memEval.fairnessOk,
        `${memEval.fairnessViolations.length} violations`);
  check('augmented MDP RESPECTS fairness (history in state ⇒ constraint enforced)',
        augEval.fairnessOk,
        `0 violations expected, got ${augEval.fairnessViolations.length}`);
  // The textbook lifting theorem: augmented = memoryless + a constraint
  // that the action set is restricted to {b : b ∩ prev_bench = ∅}.
  // So memoryless ≥ augmented as upper bound, and equality iff the
  // memoryless solver happens to land in the feasible set by accident.
  check('augmented MDP optimum within the LP upper bound',
        augEval.affinitySum <= policyLPRelaxed(problem).lpValue + 1e-9);
}

// =============================================================================
console.log('\nStudy 3 — Cross-solver consistency on the soccer LP');
// =============================================================================
{
  const lp = buildSoccerLP(problem);
  console.log(`    LP shape: ${lp.c.length} variables, `
              + `${lp.A_eq?.length ?? 0} equality + ${lp.A_ub?.length ?? 0} inequality rows`);
  const sInt = solveLPInternal(lp);
  const sDES = solveLPViaDES(lp);
  const sDS  = solveLPExternal(lp, {method: 'highs-ds'});
  const sIPM = solveLPExternal(lp, {method: 'highs-ipm'});
  console.log(`    internal simplex     status=${sInt.status} obj=${sInt.objective.toFixed(8)}`);
  console.log(`    DES-engine simplex   status=${sDES.status} obj=${sDES.objective.toFixed(8)}`);
  console.log(`    scipy:highs-ds       status=${sDS.status} obj=${sDS.objective.toFixed(8)}`);
  console.log(`    scipy:highs-ipm      status=${sIPM.status} obj=${sIPM.objective.toFixed(8)}`);
  const objs = [sInt, sDES, sDS, sIPM]
    .filter(s => s.status === 'optimal').map(s => s.objective);
  if (objs.length < 2) {
    check('At least two LP solvers available', false, 'cannot cross-validate');
  } else {
    const refObj = objs[0];
    const maxDiff = Math.max(...objs.map(o => Math.abs(o - refObj)));
    check('all available solvers agree on the LP objective to 1e-5',
          maxDiff < 1e-5,
          `max |Δ| = ${maxDiff.toExponential(2)}`);
  }
}

// =============================================================================
console.log('\nStudy 4 — Per-period Hungarian dominates random');
// =============================================================================
{
  const random = evaluateSchedule(problem, policyRandomSchedule(problem, 7));
  const greedy = evaluateSchedule(problem, policyGreedyHungarian(problem, {fairnessAware: true}));
  const mdp = evaluateSchedule(problem, policyMDPVI(problem));
  console.log(`    random           affinity = ${random.affinitySum.toFixed(3)}`);
  console.log(`    greedy-Hungarian affinity = ${greedy.affinitySum.toFixed(3)}`);
  console.log(`    MDP-VI exact     affinity = ${mdp.affinitySum.toFixed(3)}`);
  check('greedy-Hungarian > random (per-period assignment beats permutation)',
        greedy.affinitySum > random.affinitySum + 1.0,
        `Δ = ${(greedy.affinitySum - random.affinitySum).toFixed(3)}`);
  check('MDP-VI ≥ greedy-Hungarian (multi-period optimum ≥ greedy)',
        mdp.affinitySum + 1e-9 >= greedy.affinitySum,
        `Δ = ${(mdp.affinitySum - greedy.affinitySum).toFixed(3)}`);
}

// =============================================================================
console.log('\nStudy 5 — DES match: MDP-VI vs random goal differential (Welch t-test)');
// =============================================================================
{
  const numMatches = 50;
  const random = runManyMatches(problem, policyRandomSchedule(problem, 11),
    'random', numMatches, 999);
  const mdp = runManyMatches(problem, policyMDPVI(problem),
    'mdp', numMatches, 999);
  const greedy = runManyMatches(problem, policyGreedyHungarian(problem, {fairnessAware: true}),
    'greedy', numMatches, 999);
  console.log(`    random:  goal diff = ${random.meanGoalDiff.toFixed(2)} ± ${random.sdGoalDiff.toFixed(2)}`);
  console.log(`    greedy:  goal diff = ${greedy.meanGoalDiff.toFixed(2)} ± ${greedy.sdGoalDiff.toFixed(2)}`);
  console.log(`    MDP-VI:  goal diff = ${mdp.meanGoalDiff.toFixed(2)} ± ${mdp.sdGoalDiff.toFixed(2)}`);
  const tMdpVsRandom = welchT(mdp.rawGoalDiffs, random.rawGoalDiffs);
  const tGreedyVsRandom = welchT(greedy.rawGoalDiffs, random.rawGoalDiffs);
  console.log(`    Welch t  random→MDP    = ${tMdpVsRandom.toFixed(2)}`);
  console.log(`    Welch t  random→greedy = ${tGreedyVsRandom.toFixed(2)}`);
  check('MDP-VI > random in goal differential (Welch-t > 3)',
        tMdpVsRandom > 3,
        `t = ${tMdpVsRandom.toFixed(2)}`);
  check('greedy > random in goal differential (Welch-t > 3)',
        tGreedyVsRandom > 3,
        `t = ${tGreedyVsRandom.toFixed(2)}`);
  check('MDP-VI fairness OK in DES match',
        mdp.fairnessOk);
  check('greedy-Hungarian fairness OK in DES match',
        greedy.fairnessOk);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
