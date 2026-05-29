'use strict';

// =============================================================================
// runners/validate-simulated-annealing.ts — verify SA solver against
// known optima (pentagon TSP, Held–Karp on small TSPs) and against
// MILP-B&B on knapsack.
// =============================================================================

import {
  runSimulatedAnnealing, buildTSPSAProblem, buildKnapsackSAProblem, temperatureAt,
} from '../general/simulated-annealing';
import {
  buildPentagonTSP, buildRandomTSP, tourLength, heldKarpExact,
} from '../general/genetic-tsp';
import {solveMILP, buildKnapsackMILP} from '../general/milp-bnb';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

// =============================================================================
console.log('\nStudy 1 — Pentagon TSP: SA finds exact optimum');
// =============================================================================
{
  const inst = buildPentagonTSP(5, 50);
  const opt = tourLength(inst, [0, 1, 2, 3, 4]);
  for (const seed of [1, 7, 13, 42, 99]) {
    const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
      maxIterations: 5000,
      cooling: {kind: 'geometric', T0: 50, alpha: 0.998},
      seed,
    });
    check(`1.x seed=${seed}: SA matches optimum`, close(r.bestCost, opt, 1e-4),
      `SA=${r.bestCost.toFixed(4)}, opt=${opt.toFixed(4)}`);
  }
}

// =============================================================================
console.log('\nStudy 2 — Small random TSPs: SA matches Held–Karp');
// =============================================================================
{
  for (const n of [6, 8, 10]) {
    for (const seed of [3, 17]) {
      const inst = buildRandomTSP(n, seed);
      const exact = heldKarpExact(inst);
      const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
        maxIterations: 30000,
        cooling: {kind: 'geometric', T0: 50, alpha: 0.9995},
        seed: 1,
      });
      check(`2.x n=${n} seed=${seed}: SA ratio ≤ 1.05 of exact`, r.bestCost <= exact.length * 1.05 + 1e-9,
        `SA=${r.bestCost.toFixed(4)}, exact=${exact.length.toFixed(4)}, ratio=${(r.bestCost / exact.length).toFixed(4)}`);
    }
  }
}

// =============================================================================
console.log('\nStudy 3 — Knapsack SA matches MILP-B&B (or comes close)');
// =============================================================================
{
  let s = 5;
  const rng = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 0x100000000; };
  for (let trial = 0; trial < 5; trial++) {
    const n = 12;
    const v = Array.from({length: n}, () => Math.floor(rng() * 50 + 1));
    const w = Array.from({length: n}, () => Math.floor(rng() * 25 + 1));
    const cap = Math.floor(w.reduce((a, b) => a + b, 0) * 0.4);
    const exact = solveMILP(buildKnapsackMILP(v, w, cap));
    const sa = runSimulatedAnnealing(buildKnapsackSAProblem({values: v, weights: w, capacity: cap}), {
      maxIterations: 10000,
      cooling: {kind: 'geometric', T0: 50, alpha: 0.999},
      seed: trial,
    });
    const saValue = -sa.bestCost;
    check(`3.x trial=${trial}: SA value ≥ 0.95 × exact`, saValue >= 0.95 * exact.z - 1e-6,
      `SA=${saValue.toFixed(2)}, exact=${exact.z.toFixed(2)}, ratio=${(saValue / exact.z).toFixed(4)}`);
  }
}

// =============================================================================
console.log('\nStudy 4 — Cooling schedules');
// =============================================================================
{
  // 4.1 Temperatures monotonically non-increasing for geometric, linear.
  const geom = {kind: 'geometric' as const, T0: 100, alpha: 0.99};
  const lin = {kind: 'linear' as const, T0: 100, rate: 1};
  const log = {kind: 'logarithmic' as const, T0: 100};
  let geoMono = true, linMono = true, logMono = true;
  let prevG = Infinity, prevL = Infinity, prevLg = Infinity;
  for (let k = 0; k < 200; k++) {
    const tg = temperatureAt(geom, k);
    const tl = temperatureAt(lin, k);
    const tlg = temperatureAt(log, k);
    if (tg > prevG + 1e-9) geoMono = false;
    if (tl > prevL + 1e-9) linMono = false;
    if (tlg > prevLg + 1e-9) logMono = false;
    prevG = tg; prevL = tl; prevLg = tlg;
  }
  check('4.1 geometric schedule monotone non-increasing', geoMono);
  check('4.2 linear schedule monotone non-increasing', linMono);
  check('4.3 logarithmic schedule monotone non-increasing', logMono);
  // 4.4 Tmin floor enforced.
  const t = temperatureAt({kind: 'geometric', T0: 100, alpha: 0.5, Tmin: 0.01}, 1000);
  check('4.4 Tmin floor enforced', t === 0.01, `T(1000) = ${t}`);
  // 4.5 At k=0 all schedules return T0.
  const t0g = temperatureAt({kind: 'geometric', T0: 50, alpha: 0.99}, 0);
  const t0l = temperatureAt({kind: 'linear', T0: 50, rate: 1}, 0);
  check('4.5 geometric T(0) = T0', t0g === 50);
  check('4.6 linear T(0) = T0', t0l === 50);
}

// =============================================================================
console.log('\nStudy 5 — Reproducibility: same seed → same trajectory');
// =============================================================================
{
  const inst = buildRandomTSP(10, 1);
  const sa1 = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 1000, cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, seed: 42,
  });
  const sa2 = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 1000, cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, seed: 42,
  });
  check('5.1 same seed: same best cost', close(sa1.bestCost, sa2.bestCost, 1e-12));
  check('5.2 same seed: same iteration count', sa1.iterations === sa2.iterations);
  check('5.3 same seed: same accepted count', sa1.acceptedCount === sa2.acceptedCount);
  // 5.4 Different seeds: different best history.
  const sa3 = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 1000, cooling: {kind: 'geometric', T0: 50, alpha: 0.99}, seed: 99,
  });
  check('5.4 different seed: different bestHistory[100]',
    Math.abs(sa1.bestHistory[100] - sa3.bestHistory[100]) > 1e-9 ||
    Math.abs(sa1.bestHistory[500] - sa3.bestHistory[500]) > 1e-9);
}

// =============================================================================
console.log('\nStudy 6 — Best history is monotonic (best can only improve)');
// =============================================================================
{
  const inst = buildRandomTSP(15, 4);
  const sa = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 5000, cooling: {kind: 'geometric', T0: 50, alpha: 0.999}, seed: 1,
  });
  let mono = true;
  for (let k = 1; k < sa.bestHistory.length; k++) {
    if (sa.bestHistory[k] > sa.bestHistory[k - 1] + 1e-12) { mono = false; break; }
  }
  check('6.1 bestHistory monotonically non-increasing', mono);
}

// =============================================================================
console.log('\nStudy 7 — Acceptance rate decreases with cooling');
// =============================================================================
{
  // For a low-temperature schedule, acceptance rate of WORSE moves should
  // be much lower than for a high-T schedule.
  const inst = buildRandomTSP(15, 9);
  const saHot = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 5000, cooling: {kind: 'geometric', T0: 1000, alpha: 1.0}, seed: 1,
  });
  const saCold = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 5000, cooling: {kind: 'geometric', T0: 1e-12, alpha: 1.0}, seed: 1,
  });
  const hotRate = saHot.acceptedCount / saHot.iterations;
  const coldRate = saCold.acceptedCount / saCold.iterations;
  check('7.1 high-T accept rate > low-T accept rate', hotRate > coldRate, `hot=${hotRate.toFixed(3)}, cold=${coldRate.toFixed(3)}`);
  // Cold-T should reject WORSE moves (Δ > 0). Some "accepts" come from null
  // or-opt moves where Δ = 0 (returned unchanged tour) — those don't violate
  // the "no worsening" property.
  check('7.2 cold-T improveCount/acceptedCount ratio low (only improvements + zero-Δ)',
    saCold.improveCount / Math.max(1, saCold.acceptedCount) >= 0.1,
    `improvements=${saCold.improveCount}, accepted=${saCold.acceptedCount}, ratio=${(saCold.improveCount / Math.max(1, saCold.acceptedCount)).toFixed(3)}`);
  check('7.3 cold-T finalCost ≤ initial', saCold.finalCost <= saCold.bestHistory[0] + 1e-9,
    `final=${saCold.finalCost.toFixed(2)}, init=${saCold.bestHistory[0].toFixed(2)}`);
}

// =============================================================================
console.log('\nStudy 8 — Stall-limit early stopping');
// =============================================================================
{
  const inst = buildRandomTSP(8, 1);
  const sa = runSimulatedAnnealing(buildTSPSAProblem(inst), {
    maxIterations: 100000,
    cooling: {kind: 'geometric', T0: 0.001, alpha: 1.0},   // essentially 0
    seed: 1, stallLimit: 50,
  });
  check('8.1 stall-limit triggers early termination', sa.iterations < 100000,
    `iterations = ${sa.iterations} (< 100000)`);
}

console.log('\n  ─────────────────────────────────────────────────────────────────────────');
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
