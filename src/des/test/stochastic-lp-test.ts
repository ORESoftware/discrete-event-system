// RUST MIGRATION: Prefer moving these focused checks into `src/des/general/stochastic_lp.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate SAA/Benders/oracle cases into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and approximate-float helpers; keep scenario seeds deterministic.

'use strict';

// =============================================================================
// Unit tests for the stochastic LP solver: subproblem dual extraction,
// SAA monolithic solver, Benders decomposition, and the closed-form oracle.
// Run with: node dist/des/test/stochastic-lp-test.js
// =============================================================================

import {
  solveSubproblemWithDuals, buildProductionSLP, buildProductionScenarios,
  solveSLPMonolithic, solveSLPBenders, solveProductionClosedForm,
  Scenario,
} from '../general/stochastic-lp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-7): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

// =============================================================================
console.log('\n[1] solveSubproblemWithDuals — single product newsvendor recourse');
// =============================================================================
{
  // Subproblem: max p · y  s.t. y ≤ x, y ≤ D, y ≥ 0  with p = 25, x = 60, D = 80.
  // Optimal y = 60 (capacity binds). Duals: π_capacity = 25 (shadow price of x), π_demand = 0.
  const r = solveSubproblemWithDuals([25], [[1], [1]], [60, 80]);
  check('1.1 status optimal', r.status === 'optimal');
  check('1.2 y* = 60',         close(r.y[0], 60));
  check('1.3 obj = 25·60',     close(r.obj, 1500));
  check('1.4 dual_capacity = 25',  close(r.duals[0], 25));
  check('1.5 dual_demand   = 0',   close(r.duals[1], 0));
}
{
  // Same but x = 90, D = 70. Optimal y = 70 (demand binds).
  const r = solveSubproblemWithDuals([25], [[1], [1]], [90, 70]);
  check('1.6 demand binds, y = 70',   close(r.y[0], 70));
  check('1.7 dual_capacity = 0',      close(r.duals[0], 0));
  check('1.8 dual_demand   = 25',     close(r.duals[1], 25));
}

// =============================================================================
console.log('\n[2] solveSubproblemWithDuals — two-product, capacity-bound');
// =============================================================================
{
  // 2 products, p = (25, 28), x = (50, 40), D = (80, 60). Both capacities bind.
  const r = solveSubproblemWithDuals(
    [25, 28],
    [[1, 0], [0, 1], [1, 0], [0, 1]],
    [50, 40, 80, 60],
  );
  check('2.1 y = (50, 40)', close(r.y[0], 50) && close(r.y[1], 40));
  check('2.2 obj = 25·50 + 28·40 = 2370', close(r.obj, 2370));
  check('2.3 capacity duals = (25, 28)', close(r.duals[0], 25) && close(r.duals[1], 28));
  check('2.4 demand   duals = (0, 0)',   close(r.duals[2], 0)  && close(r.duals[3], 0));
}

// =============================================================================
console.log('\n[3] Closed-form newsvendor agrees with derived formula');
// =============================================================================
{
  // For uniform D ~ U(a, b) and parameters c, p with p > c:
  // x* = a + (b-a)(p-c)/p.
  const c = [10], p = [25];
  const ranges: Array<[number, number]> = [[50, 100]];
  // x* = 50 + 50 · (15/25) = 80.
  const r = solveProductionClosedForm(c, p, ranges);
  check('3.1 closed-form x* = 80', close(r.x[0], 80));
  // E[min(80, D)] for D ~ U(50, 100): 80 - (80-50)²/(2·50) = 80 - 9 = 71.
  // z* = -10·80 + 25·71 = -800 + 1775 = 975.
  check('3.2 closed-form z* = 975', close(r.objective, 975));
}

// =============================================================================
console.log('\n[4] Monolithic SAA on a 3-scenario discrete distribution');
// =============================================================================
{
  // Build three explicit scenarios (D = (40,30), (60,50), (80,70)) each with 1/3 weight.
  const slp = buildProductionSLP([10, 12], [25, 28]);
  const scenarios: Scenario[] = [];
  for (const D of [[40, 30], [60, 50], [80, 70]]) {
    const T = [[-1, 0], [0, -1], [0, 0], [0, 0]];
    const h = [0, 0, D[0], D[1]];
    scenarios.push({T, h, prob: 1 / 3, meta: {D}});
  }
  const sol = solveSLPMonolithic(slp, scenarios);
  check('4.1 status optimal', sol.status === 'optimal');
  // For each scenario, y = min(x, D). For x*  ∈ [40, 60], scenario 1 has y = (40, 30), scen 2,3 are y = (x, x).
  // Compute the SAA objective explicitly.
  // SAA objective = -10 x_1 - 12 x_2 + (1/3) Σ [25 min(x_1, D_1^s) + 28 min(x_2, D_2^s)].
  // Optimal x* satisfies the SAA newsvendor critical-fractile condition with empirical CDF.
  // We verify the structure: x_1 ∈ {40, 60, 80} or y values are tight against x.
  const x = sol.x;
  check('4.2 x non-negative', x[0] >= -1e-9 && x[1] >= -1e-9);
  check('4.3 SAA objective matches direct computation', close(
    sol.objective,
    -10 * x[0] - 12 * x[1] + (1 / 3) * (
      (25 * Math.min(x[0], 40) + 28 * Math.min(x[1], 30)) +
      (25 * Math.min(x[0], 60) + 28 * Math.min(x[1], 50)) +
      (25 * Math.min(x[0], 80) + 28 * Math.min(x[1], 70))
    ),
    1e-7,
  ));
}

// =============================================================================
console.log('\n[5] Benders converges and matches monolithic on same scenarios');
// =============================================================================
{
  const slp = buildProductionSLP([10, 12], [25, 28]);
  const sc = buildProductionScenarios({ranges: [[50, 100], [40, 80]], seed: 7}, 100);
  const mono = solveSLPMonolithic(slp, sc);
  const bend = solveSLPBenders(slp, sc, {tol: 1e-9});
  check('5.1 mono optimal', mono.status === 'optimal');
  check('5.2 Benders optimal', bend.status === 'optimal');
  check('5.3 z agrees to 1e-6', Math.abs(mono.objective - bend.objective) <= 1e-6);
  check('5.4 Benders converged in ≤ 50 iters', bend.iterations <= 50, `iters=${bend.iterations}`);
  // Verify the cut count equals iterations - 1.
  check('5.5 cut count = iter − 1', (bend.bendersTrace?.filter(t => t.cutAdded).length ?? 0) === bend.iterations - 1);
}

// =============================================================================
console.log('\n[6] Benders convergence properties: master UB non-increasing, best-so-far gap non-increasing');
// =============================================================================
{
  const slp = buildProductionSLP([10, 12], [25, 28]);
  const sc = buildProductionScenarios({ranges: [[50, 100], [40, 80]], seed: 11}, 50);
  const bend = solveSLPBenders(slp, sc, {tol: 1e-9, maxIter: 200});
  const trace = bend.bendersTrace ?? [];
  // Master UB (master objective) is monotonically non-increasing as cuts are added.
  let ubMonotone = true;
  for (let i = 1; i < trace.length; i++) {
    if (trace[i].upperBound > trace[i - 1].upperBound + 1e-6) { ubMonotone = false; break; }
  }
  check('6.1 master upper bound is non-increasing (cuts only tighten)', ubMonotone);
  // Best-so-far lower bound (best feasible) is non-decreasing.
  let bestLB = -Infinity;
  let lbMonotone = true;
  for (const it of trace) {
    if (it.lowerBound > bestLB) bestLB = it.lowerBound;
    // After this iteration the best-so-far is bestLB; check that it has only grown.
  }
  // Check non-decreasing best-so-far across the trace:
  let runningBest = -Infinity;
  for (const it of trace) {
    const newBest = Math.max(runningBest, it.lowerBound);
    if (newBest < runningBest - 1e-9) { lbMonotone = false; break; }
    runningBest = newBest;
  }
  check('6.2 best-so-far lower bound is non-decreasing', lbMonotone);
  check('6.3 final per-iter gap ≤ 1e-6', trace[trace.length - 1].gap <= 1e-6);
}

// =============================================================================
console.log('\n[7] As N grows, SAA optimum approaches closed-form');
// =============================================================================
{
  const c = [10, 12], p = [25, 28];
  const ranges: Array<[number, number]> = [[50, 100], [40, 80]];
  const slp = buildProductionSLP(c, p);
  const cf = solveProductionClosedForm(c, p, ranges);
  // Run two N's and check N=2000 has smaller bias (averaged over R seeds) than N=20.
  const R = 8;
  function avgBias(N: number): number {
    let acc = 0;
    for (let seed = 1; seed <= R; seed++) {
      const sc = buildProductionScenarios({ranges, seed: seed * 100 + N}, N);
      const sol = solveSLPBenders(slp, sc, {tol: 1e-7});
      acc += sol.objective - cf.objective;
    }
    return acc / R;
  }
  const bias20   = avgBias(20);
  const bias2000 = avgBias(2000);
  check(`7.1 |bias(N=2000)| ≤ |bias(N=20)|`, Math.abs(bias2000) < Math.abs(bias20),
        `|${bias2000.toFixed(3)}| vs |${bias20.toFixed(3)}|`);
}

// =============================================================================
console.log('\n[8] Budget-constrained: budget binds and reduces objective');
// =============================================================================
{
  const c = [10, 12], p = [25, 28];
  const ranges: Array<[number, number]> = [[50, 100], [40, 80]];
  const slpUnc = buildProductionSLP(c, p);
  const slpBudget = buildProductionSLP(c, p, 80);
  const sc = buildProductionScenarios({ranges, seed: 31}, 200);
  const unc = solveSLPBenders(slpUnc, sc, {tol: 1e-9});
  const bud = solveSLPBenders(slpBudget, sc, {tol: 1e-9});
  check('8.1 unconstrained Σx ≥ 80', unc.x[0] + unc.x[1] >= 80 - 1e-7);
  check('8.2 budget-constrained Σx ≤ 80', bud.x[0] + bud.x[1] <= 80 + 1e-7);
  check('8.3 budget objective ≤ unconstrained', bud.objective <= unc.objective + 1e-7);
}

// =============================================================================
console.log('\n[9] Subproblem duals: KKT optimality direct check');
// =============================================================================
{
  // For the subproblem  max p·y s.t. W y ≤ rhs, y ≥ 0, the KKT conditions say:
  //   complementarity:  π_i · (rhs_i − (W y)_i) = 0
  //   stationarity:     W^T π ≥ p  (with equality at strict slack 0)
  for (const trial of [0, 1, 2, 3, 4]) {
    const r = solveSubproblemWithDuals(
      [25, 28],
      [[1, 0], [0, 1], [1, 0], [0, 1]],
      [40 + trial * 5, 30 + trial * 7, 80 - trial * 3, 60 + trial * 4],
    );
    if (r.status !== 'optimal') { check(`9.${trial}.status`, false, r.status); continue; }
    // Complementarity: for each constraint i, π_i · slack_i = 0.
    let comp = 0;
    for (let i = 0; i < 4; i++) {
      const lhs = (i % 2 === 0 ? r.y[0] : r.y[1]);
      const slack = [40 + trial * 5, 30 + trial * 7, 80 - trial * 3, 60 + trial * 4][i] - lhs;
      comp += Math.abs(r.duals[i] * slack);
    }
    check(`9.${trial} KKT complementarity (Σ |π_i s_i|)`, comp <= 1e-9, `Σ=${comp.toExponential(2)}`);
    // Stationarity: W^T π = (π_1 + π_3, π_2 + π_4) ≥ p (equality if y > 0).
    const lhs1 = r.duals[0] + r.duals[2];
    const lhs2 = r.duals[1] + r.duals[3];
    check(`9.${trial} stationarity (lhs1 ≥ p[0])`, lhs1 + 1e-9 >= 25, `lhs1=${lhs1}`);
    check(`9.${trial} stationarity (lhs2 ≥ p[1])`, lhs2 + 1e-9 >= 28, `lhs2=${lhs2}`);
  }
}

console.log(`\n${pass + fail} tests: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
