// RUST MIGRATION:
// - Target: src/bin/validate_genetic_tsp.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert TSP instances/results/checks to nominal structs and keep precedence validation as private pure helpers.
// - Optimization algorithm calls should remain calls into migrated ga_des modules rather than embedding runner-specific logic.
'use strict';

// =============================================================================
// runners/validate-genetic-tsp.ts — verify the GA-TSP solver against
// known-optimal instances and the constraint-handling policies.
// =============================================================================

import {
  buildPentagonTSP, buildRandomTSP, runGeneticTSP,
  tourLength, checkPrecedence, isPermutation,
  heldKarpExact, oneTreeLowerBound,
} from '../general/genetic-tsp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}

// =============================================================================
console.log('\nStudy 1 — Pentagon: GA reaches the analytical optimum');
// =============================================================================
{
  // n cities equally spaced on a circle of radius R: optimal tour visits
  // them in cyclic order with length n × side, where side = 2R sin(π/n).
  const n = 5, R = 50;
  const inst = buildPentagonTSP(n, R);
  const optimal = n * 2 * R * Math.sin(Math.PI / n);
  console.log(`    analytical optimum = ${optimal.toFixed(6)}`);
  const r = runGeneticTSP(inst, {populationSize: 60, numGenerations: 100, seed: 1});
  console.log(`    GA best length     = ${r.bestLength.toFixed(6)}`);
  check('GA finds an optimal pentagon tour (within 1e-9)',
        Math.abs(r.bestLength - optimal) < 1e-9,
        `Δ = ${(r.bestLength - optimal).toExponential(2)}`);
  check('best tour is a valid permutation', isPermutation(r.bestTour, n));
}

// =============================================================================
console.log('\nStudy 2 — Small random instance: GA matches Held–Karp exact');
// =============================================================================
{
  for (const seedTSP of [3, 17, 99]) {
    const inst = buildRandomTSP(10, seedTSP);
    const exact = heldKarpExact(inst);
    const r = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: seedTSP + 1});
    console.log(`    seed ${seedTSP}: HK = ${exact.length.toFixed(3)}, GA = ${r.bestLength.toFixed(3)}`);
    check(`seed=${seedTSP}: GA within 0.5% of Held–Karp optimum`,
          r.bestLength <= exact.length * 1.005,
          `gap = ${((r.bestLength - exact.length) / exact.length * 100).toFixed(3)}%`);
  }
}

// =============================================================================
console.log('\nStudy 3 — 1-tree lower bound is a valid bound');
// =============================================================================
{
  for (const n of [8, 12, 15]) {
    const inst = buildRandomTSP(n, n);
    const lb = oneTreeLowerBound(inst);
    const r = runGeneticTSP(inst, {populationSize: 60, numGenerations: 100, seed: n + 100});
    console.log(`    n=${n}: 1-tree lb = ${lb.toFixed(2)}, GA best = ${r.bestLength.toFixed(2)}`);
    check(`n=${n}: 1-tree lower bound ≤ GA best`,
          lb <= r.bestLength + 1e-9,
          `lb=${lb.toFixed(3)}, ga=${r.bestLength.toFixed(3)}`);
  }
}

// =============================================================================
console.log('\nStudy 4 — Precedence constraints: all branches respected');
// =============================================================================
{
  const inst = buildRandomTSP(15, 42);
  inst.precedence = [[0, 14], [1, 13], [2, 12], [3, 11]];
  const r = runGeneticTSP(inst, {
    populationSize: 80, numGenerations: 150, seed: 42,
    feasibility: 'cut', retryLimit: 12,
  });
  check('best tour is a valid permutation', isPermutation(r.bestTour, 15));
  check('best tour respects all 4 precedence pairs (no violations remain)',
        checkPrecedence(inst, r.bestTour) === null,
        `violation: ${JSON.stringify(checkPrecedence(inst, r.bestTour))}`);
  console.log(`    feasible kids evaluated = ${r.totalFeasibleEvaluated}, infeasible kids cut = ${r.totalInfeasibleCut}`);
  check('at least some children were infeasible (i.e. branch-cutting active)',
        r.totalInfeasibleCut > 0,
        `cut count = ${r.totalInfeasibleCut}`);
}

// =============================================================================
console.log('\nStudy 5 — Constraint policies converge differently');
// =============================================================================
{
  // For an instance with several precedence pairs, "cut" should produce
  // a feasible best tour with no penalty, while "penalize" might leave a
  // slightly fractional / infeasible best (or much higher reported length).
  const inst = buildRandomTSP(16, 11);
  inst.precedence = [[0, 15], [2, 13], [4, 11], [6, 9]];
  const cut = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: 11,
    feasibility: 'cut', retryLimit: 12});
  const pen = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: 11,
    feasibility: 'penalize', penaltyPerViolation: 1e6});
  const repair = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: 11,
    feasibility: 'repair'});
  console.log(`    cut     best length = ${cut.bestLength.toFixed(3)}, infeasible cut = ${cut.totalInfeasibleCut}`);
  console.log(`    penalize best length = ${pen.bestLength.toFixed(3)} (might include +∞ if infeasible)`);
  console.log(`    repair  best length = ${repair.bestLength.toFixed(3)}`);
  check('cut policy: best tour is feasible',
        checkPrecedence(inst, cut.bestTour) === null);
  check('repair policy: best tour is feasible',
        checkPrecedence(inst, repair.bestTour) === null);
  check('cut policy actually cut some offspring',
        cut.totalInfeasibleCut > 0);
}

// =============================================================================
console.log('\nStudy 6 — Convergence: best tour length is monotone non-increasing');
// =============================================================================
{
  const inst = buildRandomTSP(20, 7);
  const r = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: 7, elitism: 4});
  let monotone = true;
  for (let g = 1; g < r.perGenerationBest.length; g++) {
    if (r.perGenerationBest[g] > r.perGenerationBest[g - 1] + 1e-9) { monotone = false; break; }
  }
  check('elitism guarantees best-so-far is monotone non-increasing',
        monotone,
        `first ${Math.min(15, r.perGenerationBest.length)} = ${r.perGenerationBest.slice(0, 15).map(v => v.toFixed(2)).join(' ')}`);
  check('GA improves over the initial population',
        r.perGenerationBest[r.perGenerationBest.length - 1] < r.perGenerationBest[0],
        `${r.perGenerationBest[0].toFixed(2)} → ${r.perGenerationBest[r.perGenerationBest.length - 1].toFixed(2)}`);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
