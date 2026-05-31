// RUST MIGRATION: Prefer moving these focused checks into `src/des/general/genetic_tsp.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate TSP/GA assertions into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, approximate-float helpers, and deterministic PRNG seeds.

'use strict';

// =============================================================================
// test/genetic-tsp-test.ts — unit tests for the GA-TSP module.
// =============================================================================

import {
  buildPentagonTSP, buildRandomTSP, runGeneticTSP,
  tourLength, checkPrecedence, isPermutation,
  orderCrossover, inversionMutate, swapMutate, tournamentSelect,
  heldKarpExact, oneTreeLowerBound, repairPrecedence, twoOptImprove,
} from '../general/genetic-tsp';
import {mulberry32} from '../general/prng';

let pass = 0, fail = 0;
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' — ' + detail : ''));
  cond ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-9): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 — Tour length and feasibility');
// -----------------------------------------------------------------------------
{
  const inst = buildPentagonTSP(4, 50);
  const tour = [0, 1, 2, 3];
  const len = tourLength(inst, tour);
  // Square: side = 2*50*sin(45°) = 50√2 ≈ 70.71, perimeter = 4 * 70.71 ≈ 282.84
  close('square perimeter ≈ 4·50√2', len, 4 * 50 * Math.sqrt(2), 1e-6);
  expect('valid permutation', isPermutation(tour, 4));
  expect('repeated city is not a permutation', !isPermutation([0, 1, 1, 3], 4));
  expect('out-of-range is not a permutation', !isPermutation([0, 1, 4, 3], 4));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 — Order-Crossover preserves permutations');
// -----------------------------------------------------------------------------
{
  const rng = mulberry32(11);
  for (let trial = 0; trial < 50; trial++) {
    const n = 8;
    const p1 = Array.from({length: n}, (_, i) => i);
    const p2 = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1)); [p2[i], p2[j]] = [p2[j], p2[i]];
    }
    const child = orderCrossover(p1, p2, rng);
    if (!isPermutation(child, n)) {
      expect(`OX trial ${trial} produces a permutation`, false, JSON.stringify(child));
    }
  }
  expect('all 50 OX trials produced valid permutations', true);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 — Mutations preserve permutations');
// -----------------------------------------------------------------------------
{
  const rng = mulberry32(13);
  const tour = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let trial = 0; trial < 30; trial++) {
    const m = trial % 2 === 0 ? inversionMutate(tour, rng) : swapMutate(tour, rng);
    if (!isPermutation(m, 8)) { expect('mutation valid', false); break; }
  }
  expect('30 mutations produced valid permutations', true);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 — Tournament selection picks lower-cost chromosome');
// -----------------------------------------------------------------------------
{
  const lengths = [10, 5, 100, 200, 1, 50];
  const rng = mulberry32(1);
  // Tournament of size 6 must always pick the global min (index 4, value 1).
  let foundOne = false;
  for (let i = 0; i < 50; i++) {
    const idx = tournamentSelect(lengths, 6, rng);
    if (idx === 4) foundOne = true;
  }
  expect('size-N tournament eventually picks global best', foundOne);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 — Held–Karp on 4-city L-shape gives the right answer');
// -----------------------------------------------------------------------------
{
  // Cities at (0,0), (10,0), (10,10), (0,10) — square, optimal cycle
  // perimeter 40.
  const inst = {
    n: 4,
    coordinates: [[0, 0], [10, 0], [10, 10], [0, 10]] as Array<[number, number]>,
    distance: [
      [0, 10, Math.sqrt(200), 10],
      [10, 0, 10, Math.sqrt(200)],
      [Math.sqrt(200), 10, 0, 10],
      [10, Math.sqrt(200), 10, 0],
    ],
  };
  const r = heldKarpExact(inst);
  close('square perimeter = 40', r.length, 40, 1e-9);
  expect('exact tour is a permutation', isPermutation(r.tour, 4));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 — 1-tree lower bound respects optimum');
// -----------------------------------------------------------------------------
{
  for (const seed of [1, 5, 11]) {
    const inst = buildRandomTSP(8, seed);
    const lb = oneTreeLowerBound(inst);
    const exact = heldKarpExact(inst);
    expect(`seed=${seed}: 1-tree lb (${lb.toFixed(2)}) ≤ optimum (${exact.length.toFixed(2)})`,
      lb <= exact.length + 1e-9);
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 — GA solves a tiny pentagon to optimum');
// -----------------------------------------------------------------------------
{
  const inst = buildPentagonTSP(5, 40);
  const optimal = 5 * 2 * 40 * Math.sin(Math.PI / 5);
  const r = runGeneticTSP(inst, {populationSize: 50, numGenerations: 60, seed: 42});
  close('GA finds the pentagon optimum', r.bestLength, optimal, 1e-9);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 8 — Reproducibility');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(10, 7);
  const r1 = runGeneticTSP(inst, {populationSize: 40, numGenerations: 30, seed: 99});
  const r2 = runGeneticTSP(inst, {populationSize: 40, numGenerations: 30, seed: 99});
  close('same seed ⇒ same best length', r1.bestLength, r2.bestLength, 1e-12);
  expect('same seed ⇒ same best tour', JSON.stringify(r1.bestTour) === JSON.stringify(r2.bestTour));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 9 — Precedence: cut policy yields feasible tours');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(12, 21);
  inst.precedence = [[0, 11], [1, 10], [2, 9]];
  const r = runGeneticTSP(inst, {populationSize: 50, numGenerations: 80, seed: 21,
    feasibility: 'cut', retryLimit: 12});
  expect('best tour respects all precedence pairs',
    checkPrecedence(inst, r.bestTour) === null);
  expect('best tour is a valid permutation', isPermutation(r.bestTour, 12));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 10 — repairPrecedence does what it says');
// -----------------------------------------------------------------------------
{
  const inst = {
    n: 4,
    coordinates: [[0, 0], [1, 0], [2, 0], [3, 0]] as Array<[number, number]>,
    distance: [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]],
    precedence: [[0, 1], [2, 3]] as Array<[number, number]>,
  };
  // Tour [3, 2, 1, 0] violates both pairs.
  const r = repairPrecedence(inst, [3, 2, 1, 0]);
  expect('repaired tour is feasible', r.feasible,
    `final = ${JSON.stringify(r.tour)}, violation = ${JSON.stringify(checkPrecedence(inst, r.tour))}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 11 — Optional 2-opt local search never worsens a tour');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(14, 55);
  const badTour = [0, 5, 2, 9, 3, 10, 4, 11, 1, 12, 6, 13, 7, 8];
  const improved = twoOptImprove(inst, badTour, 12);
  expect('2-opt preserves permutation', isPermutation(improved, inst.n));
  expect('2-opt does not increase tour length',
    tourLength(inst, improved) <= tourLength(inst, badTour) + 1e-9,
    `${tourLength(inst, badTour).toFixed(3)} -> ${tourLength(inst, improved).toFixed(3)}`);

  const plain = runGeneticTSP(inst, {populationSize: 40, numGenerations: 40, seed: 5});
  const memetic = runGeneticTSP(inst, {
    populationSize: 40,
    numGenerations: 40,
    seed: 5,
    localSearch: 'two-opt',
    localSearchPasses: 2,
  });
  expect('memetic GA applied local search', memetic.localSearchApplications > 0,
    `applications=${memetic.localSearchApplications}`);
  expect('memetic GA remains permutation-feasible', isPermutation(memetic.bestTour, inst.n));
  expect('memetic GA is no worse than same-seed plain GA',
    memetic.bestLength <= plain.bestLength + 1e-9,
    `${plain.bestLength.toFixed(3)} -> ${memetic.bestLength.toFixed(3)}`);
  expect('GA performance telemetry is populated',
    memetic.performance.elapsedMs >= 0
      && memetic.performance.estimatedEvaluations > 0
      && memetic.performance.finalBest === memetic.bestLength,
    JSON.stringify(memetic.performance));
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
