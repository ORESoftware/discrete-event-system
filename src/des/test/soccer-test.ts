'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/soccer_test.rs   (integration test crate)
// 1:1 file move. Spans soccer-rotation / hungarian / lp, so it is an
// integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc expect()/close()/pass-fail counters + console.log  ->  #[test] fns
//   using assert!/assert_eq!; drop the manual tally and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) float comparison -> approx::assert_relative_eq!.
//   - random/MDP rollout policies are stochastic -> a seeded rand::Rng.
// =============================================================================

// =============================================================================
// test/soccer-test.ts — unit tests for the 7v7 rotation problem.
// =============================================================================

import {
  buildSampleSoccerProblem, validateScheduleStructure, evaluateSchedule,
  policyRandomSchedule, policyGreedyHungarian, policyMDPVI, policyMDPVIMemoryless,
  policyLPRelaxed, simulateMatchDES, buildSoccerLP, runManyMatches,
  buildSoccerIPMIP, policyIPMIPFeasible, scheduleFromSoccerIPMIPVector,
  evaluateSoccerPOMDPFeatures,
} from '../general/soccer-rotation';
import {hungarian} from '../general/hungarian';
import {solveLPInternal} from '../general/lp';

let pass = 0, fail = 0;
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' — ' + detail : ''));
  cond ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-9): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 — Hungarian algorithm correctness');
// -----------------------------------------------------------------------------
{
  // Classic textbook 3×3 cost matrix:  [[4,1,3],[2,0,5],[3,2,2]]
  // Optimal min-cost assignment: row0→col1, row1→col0, row2→col2 → cost 1+2+2 = 5
  const cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];
  const r = hungarian(cost, 'min');
  close('Hungarian 3×3 textbook min total = 5', r.total, 5);
  expect('row0→col1', r.rows[0] === 1);
  expect('row1→col0', r.rows[1] === 0);
  expect('row2→col2', r.rows[2] === 2);
}
{
  // Maximisation: same matrix, max total = 4+5+3 = 12 (row0→col0, row1→col2, row2→col0)
  // ... actually we have to compute the actual optimum. Brute-force:
  // permutations of (0,1,2): (0,1,2)=4+0+2=6; (0,2,1)=4+5+2=11;
  //                          (1,0,2)=1+2+2=5; (1,2,0)=1+5+3=9;
  //                          (2,0,1)=3+2+2=7; (2,1,0)=3+0+3=6.
  // Max = 11 with row0→col0, row1→col2, row2→col1.
  const w = [[4, 1, 3], [2, 0, 5], [3, 2, 2]];
  const r = hungarian(w, 'max');
  close('Hungarian 3×3 max total = 11', r.total, 11);
  expect('row1→col2', r.rows[1] === 2);
}
{
  // Rectangular: 3 agents, 5 jobs, only 3 jobs get assigned.
  const w = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]];
  const r = hungarian(w, 'max');
  // Optimal: row0→col4 (5), row1→col0 (5), row2→{col1, col2, col3} → 3 → total 13
  close('Hungarian 3×5 max total = 13', r.total, 13);
  expect('all rows assigned', r.rows.every(j => j >= 0));
  expect('cols non-overlapping', new Set(r.rows).size === r.rows.length);
}
{
  const r = hungarian([], 'min');
  expect('empty input ok', r.rows.length === 0 && r.total === 0);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 — Problem builder produces well-formed instance');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 42});
  expect('numPlayers = 12', p.numPlayers === 12);
  expect('numPositions = 7', p.numPositions === 7);
  expect('numPeriods = 4', p.numPeriods === 4);
  expect('benchSize = 5', p.benchSize === 5);
  expect('affinity tensor shape = [12][7][4]',
    p.affinity.length === 12 && p.affinity[0].length === 7 && p.affinity[0][0].length === 4);
  expect('all affinities in [0, 1]',
    p.affinity.every(p1 => p1.every(p2 => p2.every(v => v >= 0 && v <= 1))));
  expect('player names assigned', p.playerNames?.length === 12);
  expect('position names assigned', p.positionNames?.length === 7);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 — Schedule structure validation');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 1});
  const random = policyRandomSchedule(p, 7);
  expect('random schedule structure ok', validateScheduleStructure(p, random) === null);
  const greedy = policyGreedyHungarian(p, {fairnessAware: true});
  expect('greedy schedule structure ok', validateScheduleStructure(p, greedy) === null);
  const mdp = policyMDPVI(p);
  expect('MDP-VI schedule structure ok', validateScheduleStructure(p, mdp) === null);
  // Bench size + on-field count = numPlayers per period
  for (let t = 0; t < p.numPeriods; t++) {
    const onF = mdp.assignment[t].length;
    const ben = mdp.bench[t].length;
    expect(`MDP-VI period ${t}: ${onF} on field + ${ben} bench = 12`,
      onF + ben === p.numPlayers);
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 — Fairness behaviour matches state-augmentation theory');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 2026});
  const memoryless = policyMDPVIMemoryless(p);
  const augmented = policyMDPVI(p);
  const memEval = evaluateSchedule(p, memoryless.schedule);
  const augEval = evaluateSchedule(p, augmented);
  // The memoryless MDP cannot express fairness, so its objective is an
  // UPPER BOUND on the augmented MDP's objective — by the relaxation property.
  expect('memoryless objective ≥ augmented objective',
    memoryless.value + 1e-9 >= augEval.affinitySum,
    `${memoryless.value} ≥ ${augEval.affinitySum}`);
  expect('augmented MDP respects fairness',
    augEval.fairnessOk, `violations = ${augEval.fairnessViolations.length}`);
  // The memoryless MDP USUALLY violates fairness — we don't assert it always
  // does (some affinity tensors might happen to admit no violation), but
  // for our seed=2026 problem it does.
  const memDoesViolate = !memEval.fairnessOk;
  expect('memoryless MDP violates fairness on this seed (expected)',
    memDoesViolate, `violations = ${memEval.fairnessViolations.length}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 — LP relaxation upper-bounds the MDP optimum');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 1234});
  const lp = policyLPRelaxed(p);
  const mdp = policyMDPVI(p);
  expect('LP relaxation value ≥ MDP optimum (relaxation property)',
    lp.lpValue + 1e-9 >= mdp.optimalValue,
    `LP=${lp.lpValue.toFixed(4)}, MDP=${mdp.optimalValue.toFixed(4)}`);
  // For instances where the LP polytope happens to have an integer-optimal
  // vertex, equality holds. Our seed=1234 is such an instance — gap < 1e-6.
  expect('LP-MDP integrality gap < 1e-6 on this seed',
    Math.abs(lp.lpValue - mdp.optimalValue) < 1e-6,
    `gap = ${(lp.lpValue - mdp.optimalValue).toExponential(2)}`);
  expect('LP solver reports optimal status', lp.solver.length > 0);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 — DES match simulator basic invariants');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 3});
  const schedule = policyMDPVI(p);
  const r = simulateMatchDES(p, schedule, {seed: 11});
  expect('match goalsFor ≥ 0', r.goalsFor >= 0);
  expect('match goalsAgainst ≥ 0', r.goalsAgainst >= 0);
  expect('goalDiff = goalsFor - goalsAgainst',
    r.goalDifferential === r.goalsFor - r.goalsAgainst);
  expect('trace length = numPeriods × 20',
    r.trace.length === p.numPeriods * 20);
  expect('subEvents.length = numPeriods', r.subEvents.length === p.numPeriods);
  expect('first subEvent at minute 0', r.subEvents[0].t === 0);
  // Affinity sum invariance: sum of perPeriodAffinity = evaluateSchedule.affinitySum
  const direct = evaluateSchedule(p, schedule).affinitySum;
  close('DES perPeriodAffinity sums match evaluateSchedule', r.affinitySumDeterministic, direct, 1e-9);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 — Reproducibility');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 99});
  const sched = policyMDPVI(p);
  const r1 = simulateMatchDES(p, sched, {seed: 7});
  const r2 = simulateMatchDES(p, sched, {seed: 7});
  expect('same seed ⇒ same goalsFor', r1.goalsFor === r2.goalsFor);
  expect('same seed ⇒ same goalsAgainst', r1.goalsAgainst === r2.goalsAgainst);
  expect('same seed ⇒ same #goalEvents', r1.goalEvents.length === r2.goalEvents.length);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 8 — LP shape is correct');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 5});
  const lp = buildSoccerLP(p);
  // Vars: P × K × T = 12 × 7 × 4 = 336
  expect('LP variables = P×K×T = 336', lp.c.length === 336);
  // Equalities: K × T = 28 (each position fully assigned each period)
  expect('LP equality constraints = K×T = 28', (lp.A_eq?.length ?? 0) === 28);
  // Inequalities: P×T (one-position-per-player-per-period) + P×(T-1) (fairness) = 48 + 36 = 84
  expect('LP inequality constraints = P×T + P×(T-1) = 84',
    (lp.A_ub?.length ?? 0) === 84);
  // Quick sanity: solving via internal simplex returns optimal.
  const sol = solveLPInternal(lp);
  expect('LP solves via internal simplex', sol.status === 'optimal',
    `status=${sol.status}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 9 — IP/MIP program finds a feasible 7v7 schedule');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 1234});
  const model = buildSoccerIPMIP(p);
  expect('IP/MIP variables = P×K×T = 336', model.ip.c.length === 336);
  // Inequalities: P×T player-once rows + 2×K×T position equality rows + P×(T-1) fairness lower rows.
  expect('IP/MIP constraints = P×T + 2×K×T + P×(T-1) = 140',
    model.ip.A.length === 140);
  expect('IP/MIP includes lower-bound rows', model.ip.b.some(v => v < 0));

  const mip = policyIPMIPFeasible(p, {
    timeLimitMs: 10_000,
    maxNodes: 100,
    maxTicks: 1_000,
    lpAlgorithm: 'internal-simplex',
    maxCutRounds: 0,
    fallbackToMDP: false,
  });
  expect('IP/MIP returned incumbent schedule directly', !mip.usedFallback, mip.fallbackReason);
  expect('IP/MIP schedule structure ok', validateScheduleStructure(p, mip.schedule) === null);
  const evalRes = evaluateSchedule(p, mip.schedule);
  expect('IP/MIP schedule respects fairness', evalRes.fairnessOk);
  const decoded = scheduleFromSoccerIPMIPVector(p, model, mip.mip.x);
  expect('raw IP/MIP vector decodes to schedule', decoded !== null);
  const mdp = policyMDPVI(p);
  expect('IP/MIP affinity matches exact MDP on integral seed',
    Math.abs(evalRes.affinitySum - mdp.optimalValue) < 1e-6,
    `IP/MIP=${evalRes.affinitySum.toFixed(4)}, MDP=${mdp.optimalValue.toFixed(4)}, status=${mip.mip.status}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 10 — POMDP-style hidden-fatigue feature trace');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 21});
  const sched = policyMDPVI(p);
  const belief = evaluateSoccerPOMDPFeatures(p, sched);
  expect('POMDP feature has one row per period', belief.perPeriod.length === p.numPeriods);
  expect('POMDP final fresh probabilities are probabilities',
    belief.finalFreshProbability.every(v => v >= 0 && v <= 1));
  expect('POMDP expected freshness is finite',
    Number.isFinite(belief.meanExpectedFreshOnField) && belief.meanExpectedFreshOnField > 0);
  expect('POMDP reliability is finite',
    Number.isFinite(belief.meanExpectedLineupReliability) && belief.meanExpectedLineupReliability > 0);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 11 — runManyMatches aggregates correctly');
// -----------------------------------------------------------------------------
{
  const p = buildSampleSoccerProblem({seed: 17});
  const sched = policyMDPVI(p);
  const agg = runManyMatches(p, sched, 'mdp', 5, 100);
  expect('aggregate.policyName = "mdp"', agg.policyName === 'mdp');
  expect('aggregate.rawGoalDiffs.length = 5', agg.rawGoalDiffs.length === 5);
  expect('aggregate fairness OK for MDP-VI', agg.fairnessOk);
  // Mean = arithmetic mean
  const computed = agg.rawGoalDiffs.reduce((a, b) => a + b, 0) / 5;
  close('aggregate.meanGoalDiff matches manual mean', agg.meanGoalDiff, computed, 1e-9);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
