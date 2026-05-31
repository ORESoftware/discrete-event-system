// RUST MIGRATION:
// - Target: src/bin/validate_incremental_lp.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert State and LP scenario data to nominal structs, with arrays mapped to Vec<f64> or fixed-size arrays where stable.
// - Keep close/arrayClose/check as private validation helpers and route solver calls through migrated LP traits/modules.
'use strict';

// =============================================================================
// runners/validate-incremental-lp.ts — validate the warm-startable
// incremental LP solver against the static `solveLPInternal` solver
// after every modification step.
//
// THE CONTRACT
// ────────────
// Apply a sequence of modifications (add/remove constraints, change c,
// add/remove variables) to an IncrementalLP instance, solve to optimum
// after each modification, and compare the obtained (x, z) to what a
// from-scratch solveLPInternal call yields on the equivalent static LP.
// They must agree to 1e-7 on every check.
// =============================================================================

import {IncrementalLP, LPEvent} from '../general/incremental-lp';
import {solveLPInternal, LPProblem} from '../general/lp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-7): void {
  check(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}
function arrayClose(label: string, a: number[], b: number[], tol = 1e-7): void {
  if (a.length !== b.length) { check(label, false, `lengths ${a.length} vs ${b.length}`); return; }
  let maxD = 0;
  for (let i = 0; i < a.length; i++) maxD = Math.max(maxD, Math.abs(a[i] - b[i]));
  check(label, maxD <= tol, `max|Δ|=${maxD.toExponential(2)}`);
}

interface State {
  sense: 'max' | 'min';
  c: number[];
  A: number[][];
  b: number[];
}

function solveStatic(s: State): {x: number[]; z: number; status: string} {
  const lp: LPProblem = {sense: s.sense, c: s.c.slice(), A_ub: s.A.map(r => r.slice()), b_ub: s.b.slice()};
  const sol = solveLPInternal(lp);
  return {x: sol.x, z: sol.objective, status: sol.status};
}

// =============================================================================
console.log('\nStudy 1 — Baseline 2D LP, no modifications');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  const stat = solveStatic(init);
  arrayClose('baseline x matches static',  inc.getX(), stat.x);
  close('baseline z matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 2 — Add constraint after solving (dual simplex restart)');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // Add x1 ≤ 30. Forces dual simplex.
  inc.applyEvent({tick: 0, kind: 'add-constraint', coefs: [1, 0], rhs: 30});
  inc.solveToOptimum();
  const stat = solveStatic({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3], [1, 0]], b: [100, 90, 30]});
  arrayClose('post-add-constraint x  matches static', inc.getX(), stat.x);
  close('post-add-constraint z  matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 3 — Remove a binding constraint');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // The labor constraint (row 0) is presumably binding; remove it.
  inc.applyEvent({tick: 0, kind: 'remove-constraint', index: 0});
  inc.solveToOptimum();
  // Static equivalent: only the second constraint x1 + 3x2 ≤ 90, x1, x2 ≥ 0.
  const stat = solveStatic({sense: 'max', c: [3, 5], A: [[1, 3]], b: [90]});
  arrayClose('post-remove-constraint x matches static', inc.getX(), stat.x);
  close('post-remove-constraint z matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 4 — Change objective (primal simplex restart)');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // Now widgets are more valuable: maximise 5x1 + 3x2.
  inc.applyEvent({tick: 0, kind: 'change-objective', newC: [5, 3]});
  inc.solveToOptimum();
  const stat = solveStatic({sense: 'max', c: [5, 3], A: [[2, 1], [1, 3]], b: [100, 90]});
  arrayClose('post-change-objective x matches static', inc.getX(), stat.x);
  close('post-change-objective z matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 5 — Add a variable mid-run');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // Add a new product x3 with column [1, 1] (uses 1 unit of each resource)
  // and a strong objective 7. Should be selected into the basis.
  inc.applyEvent({tick: 0, kind: 'add-variable', column: [1, 1], cNew: 7});
  inc.solveToOptimum();
  const stat = solveStatic({sense: 'max', c: [3, 5, 7], A: [[2, 1, 1], [1, 3, 1]], b: [100, 90]});
  arrayClose('post-add-variable x matches static', inc.getX(), stat.x);
  close('post-add-variable z matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 6 — Remove a variable mid-run');
// =============================================================================
{
  const init = {sense: 'max' as const, c: [3, 5, 7], A: [[2, 1, 1], [1, 3, 1]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // Drop x3 (the product whose machine just broke).
  inc.applyEvent({tick: 0, kind: 'remove-variable', structIndex: 2});
  inc.solveToOptimum();
  const stat = solveStatic({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  arrayClose('post-remove-variable x matches static', inc.getX(), stat.x);
  close('post-remove-variable z matches static', inc.getZ(), stat.z);
}

// =============================================================================
console.log('\nStudy 7 — Sequence of all 5 modifications, validating each step');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  const base = {sense: 'max' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  inc.solveToOptimum();
  arrayClose('S7.0 initial x', inc.getX(), solveStatic(base).x);

  // (a) Add x1 ≤ 30
  inc.applyEvent({tick: 0, kind: 'add-constraint', coefs: [1, 0], rhs: 30});
  inc.solveToOptimum();
  let st = solveStatic({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3], [1, 0]], b: [100, 90, 30]});
  arrayClose('S7.a after add-constraint x', inc.getX(), st.x);
  close('S7.a z', inc.getZ(), st.z);

  // (b) Change objective
  inc.applyEvent({tick: 0, kind: 'change-objective', newC: [5, 3]});
  inc.solveToOptimum();
  st = solveStatic({sense: 'max', c: [5, 3], A: [[2, 1], [1, 3], [1, 0]], b: [100, 90, 30]});
  arrayClose('S7.b after change-objective x', inc.getX(), st.x);
  close('S7.b z', inc.getZ(), st.z);

  // (c) Remove the labor constraint (row 0).
  inc.applyEvent({tick: 0, kind: 'remove-constraint', index: 0});
  inc.solveToOptimum();
  st = solveStatic({sense: 'max', c: [5, 3], A: [[1, 3], [1, 0]], b: [90, 30]});
  arrayClose('S7.c after remove-constraint x', inc.getX(), st.x);
  close('S7.c z', inc.getZ(), st.z);

  // (d) Add a new variable.
  inc.applyEvent({tick: 0, kind: 'add-variable', column: [1, 0], cNew: 4});
  inc.solveToOptimum();
  st = solveStatic({sense: 'max', c: [5, 3, 4], A: [[1, 3, 1], [1, 0, 0]], b: [90, 30]});
  arrayClose('S7.d after add-variable x', inc.getX(), st.x);
  close('S7.d z', inc.getZ(), st.z);

  // (e) Remove x2.
  inc.applyEvent({tick: 0, kind: 'remove-variable', structIndex: 1});
  inc.solveToOptimum();
  st = solveStatic({sense: 'max', c: [5, 4], A: [[1, 1], [1, 0]], b: [90, 30]});
  arrayClose('S7.e after remove-variable x', inc.getX(), st.x);
  close('S7.e z', inc.getZ(), st.z);
}

// =============================================================================
console.log('\nStudy 8 — Random 3-variable LP, randomised modification stream');
// =============================================================================
{
  // Build a random feasible LP and apply 8 random modifications, checking
  // agreement with the static solver after every one.
  function rng(seed: number): () => number {
    let s = seed >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const r = rng(1234);
  const baseN = 3, baseM = 3;
  const c = Array.from({length: baseN}, () => 1 + Math.floor(r() * 9));
  const A = Array.from({length: baseM}, () => Array.from({length: baseN}, () => 1 + Math.floor(r() * 5)));
  const b = Array.from({length: baseM}, () => 30 + Math.floor(r() * 50));
  const state: State = {sense: 'max', c, A, b};
  const inc = new IncrementalLP({sense: 'max', c: c.slice(), A: A.map(row => row.slice()), b: b.slice()});
  inc.solveToOptimum();
  let st = solveStatic(state);
  arrayClose('S8.0 initial x matches static', inc.getX(), st.x);

  // Modification 1: add a constraint x1 + x2 + x3 ≤ 50.
  state.A.push([1, 1, 1]); state.b.push(50);
  inc.applyEvent({tick: 0, kind: 'add-constraint', coefs: [1, 1, 1], rhs: 50});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.1 after add x1+x2+x3≤50',  inc.getX(), st.x);
  close      ('S8.1 z',                     inc.getZ(), st.z);

  // 2: change objective.
  state.c = [10, 7, 4];
  inc.applyEvent({tick: 0, kind: 'change-objective', newC: state.c.slice()});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.2 after change obj', inc.getX(), st.x);
  close('S8.2 z', inc.getZ(), st.z);

  // 3: remove constraint 1 (the second one, originally A[1]).
  state.A.splice(1, 1); state.b.splice(1, 1);
  inc.applyEvent({tick: 0, kind: 'remove-constraint', index: 1});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.3 after remove constraint 1', inc.getX(), st.x);
  close('S8.3 z', inc.getZ(), st.z);

  // 4: add another constraint.
  state.A.push([2, 0, 1]); state.b.push(40);
  inc.applyEvent({tick: 0, kind: 'add-constraint', coefs: [2, 0, 1], rhs: 40});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.4 after add constraint', inc.getX(), st.x);
  close('S8.4 z', inc.getZ(), st.z);

  // 5: add a 4th variable with column [1, 1, 1] and c = 6.
  for (const row of state.A) row.push(1);
  state.c.push(6);
  inc.applyEvent({tick: 0, kind: 'add-variable', column: state.A.map(r => r[r.length - 1]), cNew: 6});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.5 after add variable',  inc.getX(), st.x);
  close      ('S8.5 z',                  inc.getZ(), st.z);

  // 6: change obj again.
  state.c = [3, 12, 5, 8];
  inc.applyEvent({tick: 0, kind: 'change-objective', newC: state.c.slice()});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.6 after change obj #2', inc.getX(), st.x);
  close('S8.6 z', inc.getZ(), st.z);

  // 7: remove variable 0.
  for (const row of state.A) row.splice(0, 1);
  state.c.splice(0, 1);
  inc.applyEvent({tick: 0, kind: 'remove-variable', structIndex: 0});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.7 after remove variable 0', inc.getX(), st.x);
  close('S8.7 z', inc.getZ(), st.z);

  // 8: change obj final.
  state.c = state.c.map((_, i) => i + 1);
  inc.applyEvent({tick: 0, kind: 'change-objective', newC: state.c.slice()});
  inc.solveToOptimum();
  st = solveStatic(state);
  arrayClose('S8.8 final state x',  inc.getX(), st.x);
  close      ('S8.8 final state z', inc.getZ(), st.z);
}

// =============================================================================
console.log('\nStudy 9 — min-LP (sense flip)');
// =============================================================================
{
  const init = {sense: 'min' as const, c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]};
  const inc = new IncrementalLP(init);
  inc.solveToOptimum();
  // For min with c ≥ 0 and rhs ≥ 0, x = 0 is optimal with z = 0.
  close('min-LP at origin: z = 0', inc.getZ(), 0);
  arrayClose('min-LP at origin: x = 0', inc.getX(), [0, 0]);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
