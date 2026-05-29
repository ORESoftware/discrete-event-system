'use strict';

// =============================================================================
// main-stochastic-lp.ts — Two-stage stochastic LP via DES + incremental LP.
//
// PROBLEM: 2-product capacity-planning under demand uncertainty.
//
//   First stage:   choose capacity x ∈ R²  (cost c · x, optional budget Σx ≤ B)
//   Random data:   demand D ∈ R² ~ Uniform([a_i, b_i])
//   Second stage:  produce y ∈ R²  s.t.  y ≤ x  AND  y ≤ D,  revenue p · y
//
//   maximise      −c · x  +  E[ p · min(x, D) ]
//
// THREE WAYS, COMPARED:
//   1. Closed-form newsvendor optimum (no budget — analytical reference).
//   2. Sample Average Approximation (monolithic LP with 1 + N·n_second
//      variables, solved from scratch by `solveLPInternal`).
//   3. Benders / L-shaped decomposition expressed as a DES, with the master
//      LP being our `IncrementalLP` instance growing one cut per iteration.
//
// USAGE
//   node dist/des/main-stochastic-lp.js                       # default scenario
//   N=500 BUDGET=80 SEED=7 node dist/des/main-stochastic-lp.js
//   VERBOSE=1 node dist/des/main-stochastic-lp.js             # print Benders trace
// =============================================================================

import {
  buildProductionSLP, buildProductionScenarios,
  solveSLPMonolithic, solveSLPBenders, solveProductionClosedForm,
} from './general/stochastic-lp';

function pad(s: string, n: number): string { return s + ' '.repeat(Math.max(0, n - s.length)); }
function fmt(v: number, w: number = 8): string { return v.toFixed(2).padStart(w, ' '); }

async function main(): Promise<void> {
  const N        = Number(process.env.N      ?? 200);
  const seed     = Number(process.env.SEED   ?? 42);
  const budget   = process.env.BUDGET ? Number(process.env.BUDGET) : undefined;
  const verbose  = process.env.VERBOSE === '1';

  const c = [10, 12];
  const p = [25, 28];
  const ranges: Array<[number, number]> = [[50, 100], [40, 80]];

  console.log('# Two-stage stochastic LP — capacity planning under demand uncertainty');
  console.log(`#   first-stage cost c    = [${c.join(', ')}]`);
  console.log(`#   second-stage revenue p = [${p.join(', ')}]`);
  console.log(`#   demand D_i ~ Uniform${ranges.map(r => `[${r[0]}, ${r[1]}]`).join(' × ')}`);
  console.log(`#   budget = ${budget ?? '∞'},   N = ${N},   seed = ${seed}`);
  console.log('');

  const slp        = buildProductionSLP(c, p, budget);
  const scenarios  = buildProductionScenarios({ranges, seed}, N);

  // ── 1. Closed-form (only valid when there's no budget) ───────────────
  let cf = null as null | ReturnType<typeof solveProductionClosedForm>;
  if (budget === undefined) {
    cf = solveProductionClosedForm(c, p, ranges);
    console.log('## Method 1 — analytical closed form (newsvendor critical fractile)');
    console.log(`     x*       = [${cf.x.map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`     z*_true  = ${cf.objective.toFixed(4)}`);
    console.log(`     elapsed  = ${cf.elapsedMs} ms`);
    console.log('');
  }

  // ── 2. Monolithic SAA (single big LP) ────────────────────────────────
  console.log('## Method 2 — monolithic SAA (one giant LP via solveLPInternal)');
  const t0 = Date.now();
  const mono = solveSLPMonolithic(slp, scenarios);
  console.log(`     status    = ${mono.status}`);
  console.log(`     x*        = [${mono.x.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`     z*        = ${mono.objective.toFixed(4)}`);
  console.log(`     simplex iters = ${mono.iterations},  elapsed = ${mono.elapsedMs} ms`);
  if (cf) console.log(`     vs closed-form Δ = ${(mono.objective - cf.objective).toFixed(4)}  (Monte Carlo error)`);
  console.log('');

  // ── 3. Benders L-shaped as a DES ─────────────────────────────────────
  console.log('## Method 3 — Benders decomposition AS A DES (master = IncrementalLP, cuts arrive as movables)');
  const bend = solveSLPBenders(slp, scenarios, {tol: 1e-7, maxIter: 200, verbose});
  const trace = bend.bendersTrace ?? [];
  console.log(`     status    = ${bend.status}`);
  console.log(`     x*        = [${bend.x.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`     z*        = ${bend.objective.toFixed(4)}`);
  console.log(`     iters     = ${bend.iterations}  (one tick per master+subproblem round)`);
  console.log(`     cuts      = ${trace.filter(t => t.cutAdded).length}`);
  console.log(`     elapsed   = ${bend.elapsedMs} ms`);
  console.log(`     vs monolithic: |Δz| = ${Math.abs(bend.objective - mono.objective).toExponential(2)},  speedup ≈ ${(mono.elapsedMs / Math.max(1, bend.elapsedMs)).toFixed(1)}×`);
  if (cf) console.log(`     vs closed-form Δ = ${(bend.objective - cf.objective).toFixed(4)}`);
  console.log('');

  // ── Benders convergence table ────────────────────────────────────────
  console.log('## Benders convergence trace (UB = master objective, LB = feasible value at this x*, gap = UB − LB)');
  console.log('     ' + pad('iter', 5) + pad('x_master', 28) + pad('θ_master', 12) + pad('E[Q]', 12) + pad('UB', 12) + pad('LB', 12) + pad('gap', 12));
  let bestLB = -Infinity;
  for (const it of trace) {
    bestLB = Math.max(bestLB, it.lowerBound);
    console.log('     ' +
      pad(String(it.iter), 5) +
      pad('[' + it.xMaster.map(v => v.toFixed(2)).join(', ') + ']', 28) +
      pad(it.thetaMaster.toFixed(3), 12) +
      pad(it.expectedQ.toFixed(3), 12) +
      pad(it.upperBound.toFixed(3), 12) +
      pad(it.lowerBound.toFixed(3), 12) +
      pad(it.gap.toExponential(2), 12) +
      (it.stopReason ? `  ${it.stopReason}` : '')
    );
    if (it.cutAdded && verbose) {
      console.log('         cut  ' + it.cutAdded.coefs.map((v, i) => `${v.toFixed(3)}·${i < it.cutAdded!.coefs.length - 1 ? `x${i+1}` : 'θ'}`).join(' + ') + ` ≤ ${it.cutAdded.rhs.toFixed(3)}`);
    }
  }
  console.log('');

  // ── 4. Out-of-sample evaluation ──────────────────────────────────────
  // The OBJECTIVE we report from monolithic / Benders is the SAA OBJECTIVE,
  // an unbiased estimator that's noisy at small N. To assess how well the
  // decision generalises, we evaluate x* on a HUGE fresh sample.
  const ooN = 50000;
  const ooScenarios = buildProductionScenarios({ranges, seed: 99999}, ooN);
  function evalOutOfSample(x: number[]): number {
    let z = 0; for (let i = 0; i < c.length; i++) z += -c[i] * x[i];
    let qSum = 0;
    for (const sc of ooScenarios) {
      for (let i = 0; i < c.length; i++) qSum += p[i] * Math.min(x[i], sc.meta.D[i]);
    }
    return z + qSum / ooN;
  }
  console.log('## Out-of-sample policy evaluation (N_oos = 50000 fresh scenarios)');
  console.log(`     monolithic x*: z_oos = ${evalOutOfSample(mono.x).toFixed(4)}`);
  console.log(`     Benders   x*: z_oos = ${evalOutOfSample(bend.x).toFixed(4)}`);
  if (cf) console.log(`     closed-form x*: z_oos = ${evalOutOfSample(cf.x).toFixed(4)}  (≈ true z*)`);
  console.log('');
  console.log('# Done.');
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
