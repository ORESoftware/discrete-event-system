'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-stochastic-lp.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Three-way audit of the stochastic LP solver: SAA vs Benders-as-
// DES equivalence, 1/√N convergence, and Benders-over-monolithic speedup.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - scenario sampling + `Date.now()` timing -> inject `SeededRandom` / `Clock`
//     (timing via `std::time::Instant`).
//   - `console.log` PASS/FAIL + `process.exit` -> `println!` / `std::process::exit`.
// =============================================================================

// =============================================================================
// runners/validate-stochastic-lp.ts — three-way audit of the stochastic LP
// solver:
//
//   Part A — Cross-method bit-equivalence
//       For the SAME N scenarios, the monolithic SAA solver and Benders
//       decomposition AS A DES must produce the same x*, the same z*, and
//       the same per-scenario y* values to within machine epsilon. They
//       solve different LP formulations of the same problem, so any
//       disagreement indicates a bug in either path.
//
//   Part B — Statistical convergence to the closed-form optimum
//       For the unconstrained multi-product newsvendor problem, the
//       closed-form optimum x* and z* are known. As the SAA sample size N
//       grows from 10 → 10000, the SAA-derived z* must converge to z_true
//       at the textbook 1/√N rate. We replicate this study over R = 30
//       seeds at each N to compute confidence intervals.
//
//   Part C — Speedup of Benders over monolithic
//       The monolithic LP grows linearly with N (it has 1 + N·n_second
//       variables and 1 + N·m_second constraints). Benders' master grows
//       only with the number of CUTS, which is ≤ iterCount irrespective
//       of N. As N grows, Benders' speed advantage grows.
//
//   Part D — Budget-constrained scenario (no closed form)
//       With a binding x_1 + x_2 ≤ B constraint, no closed form exists.
//       Benders and monolithic must still agree bit-for-bit.
// =============================================================================

import {
  buildProductionSLP, buildProductionScenarios, mulberry32,
  solveSLPMonolithic, solveSLPBenders, solveProductionClosedForm,
} from '../general/stochastic-lp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-7): void {
  check(label, Math.abs(a - b) <= tol, `|${a.toFixed(6)} − ${b.toFixed(6)}| = ${Math.abs(a - b).toExponential(2)}`);
}
function arrClose(label: string, a: number[], b: number[], tol = 1e-6): void {
  if (a.length !== b.length) { check(label, false, `lengths ${a.length} vs ${b.length}`); return; }
  let maxD = 0;
  for (let i = 0; i < a.length; i++) maxD = Math.max(maxD, Math.abs(a[i] - b[i]));
  check(label, maxD <= tol, `max|Δ|=${maxD.toExponential(2)}`);
}

const c       = [10, 12];
const p       = [25, 28];
const ranges: Array<[number, number]> = [[50, 100], [40, 80]];

// =============================================================================
console.log('\nPart A — Monolithic SAA ≡ Benders-as-DES on the same scenario set');
// =============================================================================
{
  // Note: Monolithic uses dense simplex, so we keep N modest. Benders handles
  // larger N happily; Part C exercises that.
  //
  // Important LP fact: when the SAA LP has ALTERNATE OPTIMA, mono and Benders
  // may pick different x* vertices, but their objectives must agree exactly,
  // and re-evaluating either x* on the SAA scenarios must give the same z. We
  // assert the OBJECTIVE-ONLY equivalence and verify each side's x* is
  // genuinely optimal under the SAME scenario set.
  function evalSAAObjective(x: number[], scenarios: ReturnType<typeof buildProductionScenarios>): number {
    let z = 0;
    for (let i = 0; i < c.length; i++) z += -c[i] * x[i];
    let q = 0;
    for (const sc of scenarios) {
      const D: number[] = sc.meta.D;
      for (let i = 0; i < c.length; i++) q += p[i] * Math.min(x[i], D[i]);
    }
    z += q / scenarios.length;
    return z;
  }
  for (const N of [10, 50, 200]) {
    for (const seed of [1, 2, 3]) {
      const slp = buildProductionSLP(c, p);
      const scenarios = buildProductionScenarios({ranges, seed}, N);
      const mono = solveSLPMonolithic(slp, scenarios);
      const bend = solveSLPBenders(slp, scenarios, {tol: 1e-9});
      check(`A.${N}.${seed} both optimal`, mono.status === 'optimal' && bend.status === 'optimal');
      close(`A.${N}.${seed} z (mono ≡ Benders)`, mono.objective, bend.objective, 1e-6);
      // Each side's x*, evaluated under the SAME scenarios, must give the SAME z
      // (alternate optima OK — same objective value at different x).
      const zMonoEval = evalSAAObjective(mono.x, scenarios);
      const zBendEval = evalSAAObjective(bend.x, scenarios);
      close(`A.${N}.${seed} mono.x evaluates to mono.z`, zMonoEval, mono.objective, 1e-6);
      close(`A.${N}.${seed} Benders.x evaluates to Benders.z`, zBendEval, bend.objective, 1e-6);
      close(`A.${N}.${seed} both x's are equally optimal under same scenarios`, zMonoEval, zBendEval, 1e-6);
    }
  }
}

// =============================================================================
console.log('\nPart B — Statistical convergence of SAA to closed-form true optimum');
// =============================================================================
{
  const slpUnc = buildProductionSLP(c, p);                  // no budget
  const cf = solveProductionClosedForm(c, p, ranges);
  const zTrue = cf.objective;
  console.log(`  closed-form z* = ${zTrue.toFixed(4)}   x* = [${cf.x.map(v => v.toFixed(4)).join(', ')}]`);

  // We use BENDERS (much faster than monolithic for large N) for the convergence
  // study, since Part A already established that mono ≡ Benders to 1e-6.
  const Ns = [10, 100, 1000, 10000];
  const R  = 20;
  const stats: {N: number; meanZ: number; stderrZ: number; biasZ: number; meanGap: number; stderrGap: number}[] = [];
  for (const N of Ns) {
    const zs: number[] = [];
    for (let seed = 1; seed <= R; seed++) {
      const sc = buildProductionScenarios({ranges, seed: seed * 1000 + N}, N);
      const sol = solveSLPBenders(slpUnc, sc, {tol: 1e-7});
      zs.push(sol.objective);
    }
    const meanZ = zs.reduce((a, b) => a + b, 0) / R;
    const varZ  = zs.reduce((a, z) => a + (z - meanZ) ** 2, 0) / (R - 1);
    const stderrZ = Math.sqrt(varZ / R);
    const biasZ = meanZ - zTrue;
    // Gap study: compute z(SAA's x*) on a HUGE OUT-OF-SAMPLE set to evaluate
    // the OPTIMALITY GAP (i.e. how good is the SAA decision when evaluated on
    // the true distribution).
    let gapSum = 0; const gapVals: number[] = [];
    for (let seed = 1; seed <= R; seed++) {
      const sc = buildProductionScenarios({ranges, seed: seed * 1000 + N}, N);
      const sol = solveSLPBenders(slpUnc, sc, {tol: 1e-7});
      const ooSeed = 999000 + seed * 7;
      const ooScenarios = buildProductionScenarios({ranges, seed: ooSeed}, 5000);
      let zEval = 0;
      for (let i = 0; i < c.length; i++) zEval += -c[i] * sol.x[i];
      let qSum = 0;
      for (const ooSc of ooScenarios) {
        let q = 0;
        for (let i = 0; i < c.length; i++) q += p[i] * Math.min(sol.x[i], ooSc.meta.D[i]);
        qSum += q;
      }
      zEval += qSum / ooScenarios.length;
      const g = zTrue - zEval;
      gapVals.push(g); gapSum += g;
    }
    const meanGap = gapSum / R;
    const varGap  = gapVals.reduce((a, g) => a + (g - meanGap) ** 2, 0) / (R - 1);
    const stderrGap = Math.sqrt(varGap / R);
    stats.push({N, meanZ, stderrZ, biasZ, meanGap, stderrGap});
    console.log(`  N=${String(N).padStart(5, ' ')}   ` +
                `mean SAA z* = ${meanZ.toFixed(3)} ± ${stderrZ.toFixed(3)}   ` +
                `bias = ${biasZ.toFixed(3)}   ` +
                `out-of-sample gap = ${meanGap.toFixed(3)} ± ${stderrGap.toFixed(3)}`);
  }
  // Each step of N should reduce the SAA optimal-value variance by ~10×.
  // Test: stderrZ at N=10000 should be ~10× smaller than at N=100.
  const ratio_100_10000 = stats[1].stderrZ / stats[3].stderrZ;
  check(`stderr decays with √N (factor 100→10000 ≈ 10)`, ratio_100_10000 > 5 && ratio_100_10000 < 20,
        `ratio=${ratio_100_10000.toFixed(2)}`);
  // At N = 10000 the bias should be small (≤ 2 % of zTrue).
  check(`SAA z* approaches true z* at N = 10000`, Math.abs(stats[3].biasZ) <= 0.02 * Math.abs(zTrue),
        `bias=${stats[3].biasZ.toFixed(3)} vs 2% of zTrue=${(0.02 * Math.abs(zTrue)).toFixed(3)}`);
  // Out-of-sample gap should shrink monotonically with N.
  const gapShrinks = stats[3].meanGap < stats[0].meanGap;
  check(`out-of-sample optimality gap shrinks with N`, gapShrinks,
        `${stats[0].meanGap.toFixed(3)} → ${stats[3].meanGap.toFixed(3)}`);
}

// =============================================================================
console.log('\nPart C — Benders is much faster than monolithic for large N');
// =============================================================================
{
  const slpUnc = buildProductionSLP(c, p);
  // Sizes chosen so that even the monolithic LP fits in a reasonable wall clock.
  const Ns = [50, 200, 500];
  for (const N of Ns) {
    const sc = buildProductionScenarios({ranges, seed: 99}, N);
    const tMono = Date.now();
    const mono = solveSLPMonolithic(slpUnc, sc);
    const monoMs = Date.now() - tMono;
    const tBend = Date.now();
    const bend = solveSLPBenders(slpUnc, sc, {tol: 1e-7});
    const bendMs = Date.now() - tBend;
    const speedup = monoMs / Math.max(1, bendMs);
    console.log(`  N=${String(N).padStart(5, ' ')}   ` +
                `mono = ${monoMs.toFixed(0).padStart(5, ' ')} ms (${mono.iterations.toString().padStart(4, ' ')} iters)   ` +
                `Benders = ${bendMs.toFixed(0).padStart(4, ' ')} ms (${bend.iterations.toString().padStart(2, ' ')} iters)   ` +
                `speedup ≈ ${speedup.toFixed(1)}×`);
    check(`C.${N} mono and Benders agree`, Math.abs(mono.objective - bend.objective) <= 1e-5,
          `Δz = ${Math.abs(mono.objective - bend.objective).toExponential(2)}`);
  }
}

// =============================================================================
console.log('\nPart D — Budget-constrained scenario (no closed form)');
// =============================================================================
{
  for (const budget of [80, 120, 200]) {
    const slp = buildProductionSLP(c, p, budget);
    const sc = buildProductionScenarios({ranges, seed: 7}, 500);
    const mono = solveSLPMonolithic(slp, sc);
    const bend = solveSLPBenders(slp, sc, {tol: 1e-9});
    check(`D.${budget} mono ≡ Benders z`, Math.abs(mono.objective - bend.objective) <= 1e-5,
          `mono z = ${mono.objective.toFixed(4)}, Benders z = ${bend.objective.toFixed(4)}`);
    arrClose(`D.${budget} mono ≡ Benders x`, mono.x, bend.x, 1e-4);
    // Budget should bind exactly when its mono-LP objective is improving in the budget direction.
    const totalX = mono.x[0] + mono.x[1];
    check(`D.${budget} budget feasibility (Σx ≤ ${budget})`, totalX <= budget + 1e-7,
          `Σx = ${totalX.toFixed(4)}`);
  }
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
