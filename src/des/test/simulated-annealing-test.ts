// RUST MIGRATION: Prefer moving these focused checks into `src/des/general/simulated_annealing.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate annealing/TSP assertions into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, approximate helpers, and deterministic seeds.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: #[cfg(test)] mod tests in src/des/general/simulated_annealing.rs
// 1:1 file move. Unit tests one module (simulated-annealing), so prefer that
// module's `#[cfg(test)] mod tests` over a separate `tests/` file.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) relative float comparison -> approx::assert_relative_eq!.
//   - SA acceptance is stochastic -> drive runs from a seeded rand::Rng so
//     the asserted outcomes are reproducible.
// =============================================================================

// =============================================================================
// Unit tests for general/simulated-annealing.ts.
// Run with: node dist/des/test/simulated-annealing-test.js
// =============================================================================

import {
  runSimulatedAnnealing, buildTSPSAProblem, buildKnapsackSAProblem,
  temperatureAt, SAProblem,
} from '../general/simulated-annealing';
import {buildPentagonTSP, buildRandomTSP, tourLength, isPermutation} from '../general/genetic-tsp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

// -----------------------------------------------------------------------------
console.log('\n[1] temperatureAt');
// -----------------------------------------------------------------------------
{
  check('1.1 geometric T(0) = T0', temperatureAt({kind: 'geometric', T0: 100, alpha: 0.9}, 0) === 100);
  check('1.2 geometric T(1) = T0·α', close(temperatureAt({kind: 'geometric', T0: 100, alpha: 0.9}, 1), 90));
  check('1.3 geometric T(10) = T0·α^10', close(temperatureAt({kind: 'geometric', T0: 100, alpha: 0.9}, 10), 100 * Math.pow(0.9, 10)));
  check('1.4 logarithmic T(0) = T0/log(2)', close(temperatureAt({kind: 'logarithmic', T0: 100}, 0), 100 / Math.log(2)));
  check('1.5 linear T(k) = T0 - rate·k', close(temperatureAt({kind: 'linear', T0: 100, rate: 1}, 50), 50));
  check('1.6 Tmin floor', temperatureAt({kind: 'geometric', T0: 100, alpha: 0.5, Tmin: 5}, 1000) === 5);
  check('1.7 exp-restart cycles', close(
    temperatureAt({kind: 'exp-restart', T0: 100, alpha: 0.9, period: 10}, 5),
    temperatureAt({kind: 'exp-restart', T0: 100, alpha: 0.9, period: 10}, 15)));
}

// -----------------------------------------------------------------------------
console.log('\n[2] TSP adapter — initial state is a valid permutation');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(15, 42);
  const p = buildTSPSAProblem(inst, {init: 'random'});
  let s = 0;
  const rng = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 0x100000000; };
  const init = p.initial(rng);
  check('2.1 random init is permutation', isPermutation(init, 15));
  const p2 = buildTSPSAProblem(inst, {init: 'nearest-neighbor'});
  const init2 = p2.initial(rng);
  check('2.2 nearest-neighbor init is permutation', isPermutation(init2, 15));
  // Nearest-neighbor should be ≤ random in expectation.
  const len1 = tourLength(inst, init);
  const len2 = tourLength(inst, init2);
  check('2.3 nearest-neighbor length is finite', Number.isFinite(len2));
}

// -----------------------------------------------------------------------------
console.log('\n[3] TSP adapter — neighbour preserves permutation');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(15, 42);
  const p = buildTSPSAProblem(inst);
  let s = 1;
  const rng = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 0x100000000; };
  const init = p.initial(rng);
  let allPerm = true;
  for (let i = 0; i < 200; i++) {
    const nb = p.neighbour(init, rng);
    if (!isPermutation(nb, 15)) { allPerm = false; break; }
  }
  check('3.1 200 random neighbours are all permutations', allPerm);

  // 2-opt only mode: reverses a segment. Length must change at most by edges.
  const p2 = buildTSPSAProblem(inst, {moves: '2-opt'});
  let allPerm2 = true;
  for (let i = 0; i < 200; i++) {
    const nb = p2.neighbour(init, rng);
    if (!isPermutation(nb, 15)) { allPerm2 = false; break; }
  }
  check('3.2 2-opt-only neighbours are permutations', allPerm2);

  // or-opt only.
  const p3 = buildTSPSAProblem(inst, {moves: 'or-opt'});
  let allPerm3 = true;
  for (let i = 0; i < 200; i++) {
    const nb = p3.neighbour(init, rng);
    if (!isPermutation(nb, 15)) { allPerm3 = false; break; }
  }
  check('3.3 or-opt-only neighbours are permutations', allPerm3);
}

// -----------------------------------------------------------------------------
console.log('\n[4] Pentagon TSP — SA finds exact optimum');
// -----------------------------------------------------------------------------
{
  const inst = buildPentagonTSP(5, 50);
  const opt = tourLength(inst, [0, 1, 2, 3, 4]);
  const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 3000,
    cooling: {kind: 'geometric', T0: 50, alpha: 0.999},
    seed: 1,
  });
  check('4.1 best matches optimum', close(r.bestCost, opt, 1e-4), `SA=${r.bestCost.toFixed(4)}, opt=${opt.toFixed(4)}`);
  check('4.2 best state is valid permutation', isPermutation(r.bestState, 5));
}

// -----------------------------------------------------------------------------
console.log('\n[5] Knapsack SA');
// -----------------------------------------------------------------------------
{
  const inst = {values: [60, 100, 120], weights: [10, 20, 30], capacity: 50};
  const p = buildKnapsackSAProblem(inst);
  const r = runSimulatedAnnealing(p, {
    maxIterations: 3000, cooling: {kind: 'geometric', T0: 30, alpha: 0.999}, seed: 1,
  });
  // Optimal: items 1 and 2 (100 + 120 = 220, w=50). Or items 0,2 = 180, w=40.
  // 220 is optimal.
  const value = -r.bestCost;
  check('5.1 SA achieves optimal value 220', close(value, 220, 1e-3), `value=${value}`);
  check('5.2 best state is binary', r.bestState.every(v => v === 0 || v === 1));
  check('5.3 weight constraint satisfied',
    inst.weights.reduce((s, w, i) => s + w * r.bestState[i], 0) <= inst.capacity);
}

// -----------------------------------------------------------------------------
console.log('\n[6] Reproducibility');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(10, 1);
  const r1 = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 1000, cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, seed: 42,
  });
  const r2 = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 1000, cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, seed: 42,
  });
  check('6.1 same seed → same best cost', close(r1.bestCost, r2.bestCost, 1e-12));
  check('6.2 same seed → same iterations', r1.iterations === r2.iterations);
  check('6.3 same seed → same accepted', r1.acceptedCount === r2.acceptedCount);
  check('6.4 same seed → same final state', JSON.stringify(r1.finalState) === JSON.stringify(r2.finalState));
}

// -----------------------------------------------------------------------------
console.log('\n[7] Best history is monotonic');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(15, 4);
  const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 5000, cooling: {kind: 'geometric', T0: 50, alpha: 0.999}, seed: 1,
  });
  let mono = true;
  for (let i = 1; i < r.bestHistory.length; i++) {
    if (r.bestHistory[i] > r.bestHistory[i - 1] + 1e-12) { mono = false; break; }
  }
  check('7.1 bestHistory monotonically non-increasing', mono);
  check('7.2 bestHistory length matches iterations / stride', r.bestHistory.length === Math.ceil(r.iterations / 1));
}

// -----------------------------------------------------------------------------
console.log('\n[8] Generic adapter — quadratic minimisation');
// -----------------------------------------------------------------------------
{
  // Minimise (x − 3)² over integer x ∈ [-100, 100].
  // Optimum: x = 3, cost = 0.
  const p: SAProblem<{x: number}> = {
    cost: s => (s.x - 3) ** 2,
    neighbour: (s, rng) => ({x: Math.max(-100, Math.min(100, s.x + (rng() < 0.5 ? -1 : 1)))}),
    initial: () => ({x: 50}),
    clone: s => ({x: s.x}),
  };
  const r = runSimulatedAnnealing(p, {
    maxIterations: 5000, cooling: {kind: 'geometric', T0: 100, alpha: 0.99}, seed: 1,
  });
  check('8.1 reaches near-optimal x = 3', Math.abs(r.bestState.x - 3) <= 1, `x = ${r.bestState.x}`);
  check('8.2 cost ≤ 1', r.bestCost <= 1);
}

// -----------------------------------------------------------------------------
console.log('\n[9] Trace recording');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(8, 2);
  const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 100, cooling: {kind: 'geometric', T0: 50, alpha: 0.99},
    seed: 1, recordTrace: true,
  });
  check('9.1 trace defined when recordTrace=true', r.trace !== undefined);
  check('9.2 trace.length matches iterations', r.trace!.length === r.iterations);
  check('9.3 trace[0].k = 0', r.trace![0].k === 0);
  check('9.4 acceptedCount matches trace.accept count',
    r.trace!.filter(e => e.accept).length === r.acceptedCount);
}

// -----------------------------------------------------------------------------
console.log('\n[10] Stall-limit terminates early');
// -----------------------------------------------------------------------------
{
  const inst = buildRandomTSP(8, 1);
  const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 100000,
    cooling: {kind: 'geometric', T0: 1e-12, alpha: 1.0},
    seed: 1, stallLimit: 30,
  });
  check('10.1 stall-limit triggers early exit', r.iterations < 100000, `iters=${r.iterations}`);
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
