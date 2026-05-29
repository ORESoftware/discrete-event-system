'use strict';

// =============================================================================
// Unit tests for general/milp-bnb.ts.
// Run with: node dist/des/test/milp-bnb-test.js
// =============================================================================

import {
  solveMILP, buildKnapsackMILP, MILPProblem,
} from '../general/milp-bnb';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

// -----------------------------------------------------------------------------
console.log('\n[1] Knapsack builder');
// -----------------------------------------------------------------------------
{
  const k = buildKnapsackMILP([10, 20], [1, 2], 3);
  check('1.1 sense=max', k.sense === 'max');
  check('1.2 c = values', k.c[0] === 10 && k.c[1] === 20);
  check('1.3 A = [weights]', k.A.length === 1 && k.A[0][0] === 1 && k.A[0][1] === 2);
  check('1.4 b = [capacity]', k.b[0] === 3);
  check('1.5 all ints', k.integerVars.every(b => b));
  check('1.6 all 0/1 ub', k.ub!.every(b => b === 1));
}

// -----------------------------------------------------------------------------
console.log('\n[2] Trivial knapsack — exact solution');
// -----------------------------------------------------------------------------
{
  // Two items: take both if capacity allows.
  const r1 = solveMILP(buildKnapsackMILP([5, 3], [1, 1], 2));
  check('2.1 status optimal', r1.status === 'optimal');
  check('2.2 z = 8 (both items)', close(r1.z, 8));
  check('2.3 x = [1, 1]', r1.x[0] === 1 && r1.x[1] === 1);

  // Capacity 1 only fits one of the two; pick higher value.
  const r2 = solveMILP(buildKnapsackMILP([5, 3], [1, 1], 1));
  check('2.4 z = 5 (just first item)', close(r2.z, 5));
  check('2.5 x = [1, 0]', r2.x[0] === 1 && r2.x[1] === 0);

  // Capacity 0: take nothing.
  const r3 = solveMILP(buildKnapsackMILP([5, 3], [1, 1], 0));
  check('2.6 z = 0 (nothing fits)', close(r3.z, 0));
  check('2.7 x = [0, 0]', r3.x[0] === 0 && r3.x[1] === 0);
}

// -----------------------------------------------------------------------------
console.log('\n[3] Pure LP (no integer constraints) reduces to root');
// -----------------------------------------------------------------------------
{
  const lp: MILPProblem = {
    sense: 'max', c: [3, 5], A: [[1, 0], [0, 2], [3, 2]], b: [4, 12, 18],
    integerVars: [false, false],
  };
  const r = solveMILP(lp);
  check('3.1 z = 36', close(r.z, 36));
  check('3.2 x = (2, 6)', close(r.x[0], 2) && close(r.x[1], 6));
  check('3.3 only root explored', r.nodesExplored === 1);
  check('3.4 gap = 0', r.gap < 1e-9);
}

// -----------------------------------------------------------------------------
console.log('\n[4] Mixed integer/continuous');
// -----------------------------------------------------------------------------
{
  const milp: MILPProblem = {
    sense: 'max', c: [1, 1, 1], A: [[1, 1, 0]], b: [3], integerVars: [true, true, false],
    ub: [10, 10, 10],
  };
  const r = solveMILP(milp);
  check('4.1 status optimal', r.status === 'optimal');
  check('4.2 x_0 + x_1 ≤ 3', r.x[0] + r.x[1] <= 3 + 1e-6);
  check('4.3 integer vars are integer', Math.abs(r.x[0] - Math.round(r.x[0])) < 1e-6 && Math.abs(r.x[1] - Math.round(r.x[1])) < 1e-6);
  // Optimum: x_0=3 or x_1=3, x_2=10 → z = 13 (max c is x_2 unbounded if no ub; with ub=10, x_2=10).
  check('4.4 z = 13', close(r.z, 13));
}

// -----------------------------------------------------------------------------
console.log('\n[5] Bounding properties');
// -----------------------------------------------------------------------------
{
  // The LP relaxation is always an upper bound for max sense.
  const milp: MILPProblem = {
    sense: 'max', c: [10, 6, 4], A: [[1, 1, 1], [10, 4, 5], [2, 2, 6]],
    b: [100, 600, 300], integerVars: [true, true, true], ub: [Infinity, Infinity, Infinity],
  };
  const r = solveMILP(milp, {maxNodes: 5000});
  check('5.1 status optimal', r.status === 'optimal');
  check('5.2 bestBound ≥ z (max)', r.bestBound >= r.z - 1e-6, `bestBound=${r.bestBound}, z=${r.z}`);
  check('5.3 gap ≥ 0', r.gap >= 0);
}

// -----------------------------------------------------------------------------
console.log('\n[6] Feasibility constraints satisfied');
// -----------------------------------------------------------------------------
{
  const milp = buildKnapsackMILP([8, 12, 15, 22, 7], [3, 5, 6, 9, 4], 12);
  const r = solveMILP(milp);
  check('6.1 status optimal', r.status === 'optimal');
  // x ≥ 0
  check('6.2 x ≥ 0', r.x.every(v => v >= -1e-9));
  // x ≤ ub
  check('6.3 x ≤ 1 (binary)', r.x.every(v => v <= 1 + 1e-9));
  // Σ w_i x_i ≤ capacity
  const w = [3, 5, 6, 9, 4];
  let used = 0; for (let i = 0; i < r.x.length; i++) used += w[i] * r.x[i];
  check('6.4 Σ w_i x_i ≤ 12', used <= 12 + 1e-9, `used = ${used.toFixed(3)}`);
  // x integer
  check('6.5 x integer', r.x.every(v => Math.abs(v - Math.round(v)) < 1e-6));
}

// -----------------------------------------------------------------------------
console.log('\n[7] Branch-rule choice changes node count but not optimum');
// -----------------------------------------------------------------------------
{
  const milp = buildKnapsackMILP([15, 17, 8, 9, 12, 5, 30, 25], [10, 12, 5, 7, 8, 3, 15, 13], 35);
  const rMost = solveMILP(milp, {branchRule: 'most-fractional'});
  const rFirst = solveMILP(milp, {branchRule: 'first-fractional'});
  check('7.1 same optimum z', close(rMost.z, rFirst.z));
  check('7.2 same optimum value of x (or alt-optimum same z)', close(rMost.z, rFirst.z));
}

// -----------------------------------------------------------------------------
console.log('\n[8] Trace recording');
// -----------------------------------------------------------------------------
{
  const r = solveMILP(buildKnapsackMILP([10, 40, 30, 50], [5, 4, 6, 3], 10));
  check('8.1 trace.length = nodesExplored', r.trace.length === r.nodesExplored);
  check('8.2 trace[0] is root', r.trace[0].nodeId === 0 && r.trace[0].depth === 0);
  check('8.3 trace[0] has no branch', r.trace[0].branchVar === null);
  // At least one node should have integer-feasible solution and update incumbent.
  check('8.4 ≥ 1 node updated incumbent', r.trace.some(e => e.incumbentUpdated));
  // All branched nodes have fractional info.
  const branched = r.trace.filter(e => !e.pruned);
  check('8.5 branched nodes have fractional integers', branched.every(e => e.fractional.length > 0));
}

// -----------------------------------------------------------------------------
console.log('\n[9] maxNodes early termination');
// -----------------------------------------------------------------------------
{
  const milp = buildKnapsackMILP(
    Array.from({length: 30}, (_, i) => 1 + i),
    Array.from({length: 30}, (_, i) => 1 + i),
    100,
  );
  const r = solveMILP(milp, {maxNodes: 5});
  check('9.1 stops at maxNodes', r.nodesExplored <= 5);
  check('9.2 status is "maxnodes" if not yet optimal', r.status === 'optimal' || r.status === 'maxnodes');
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
