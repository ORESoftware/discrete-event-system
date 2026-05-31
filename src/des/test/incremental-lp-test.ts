// RUST MIGRATION: Prefer moving these focused LP checks into `src/des/general/incremental_lp.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate warm-start solver cases into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and approximate-float helpers; keep fixtures deterministic.

'use strict';

// =============================================================================
// Unit tests for the incremental/warm-startable LP solver.
// Run with: node dist/des/test/incremental-lp-test.js
// =============================================================================

import {IncrementalLP} from '../general/incremental-lp';
import {solveLPInternal} from '../general/lp';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
function arrClose(a: number[], b: number[], tol = 1e-7): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!close(a[i], b[i], tol)) return false;
  return true;
}

// =============================================================================
console.log('\n[1] Constructor + initial pivot');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  check('1.1 numStruct', inc.numStruct === 2);
  check('1.2 initial basis = slacks', inc.basis.length === 2 && inc.basis[0] === 2 && inc.basis[1] === 3);
  check('1.3 initial status primal', inc.status === 'primal');
  check('1.4 initial x = 0', arrClose(inc.getX(), [0, 0]));
  inc.solveToOptimum();
  check('1.5 after solve status optimal', inc.status === 'optimal');
  check('1.6 x ≈ [42, 16]', arrClose(inc.getX(), [42, 16]));
  check('1.7 z = 206',     close(inc.getZ(), 206));
}

// =============================================================================
console.log('\n[2] Add constraint, dual simplex restart');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  inc.applyAddConstraint([1, 0], 30);
  check('2.1 status after add (x1 ≤ 30 was non-binding)', inc.status === 'optimal' || inc.status === 'dual');
  inc.solveToOptimum();
  check('2.2 still optimal after solve', inc.status === 'optimal');
  // x1=42 violates x1 ≤ 30, so dual simplex must repair.
  check('2.3 x1 ≤ 30 satisfied', inc.getX()[0] <= 30 + 1e-9);
  // tighten further
  inc.applyAddConstraint([0, 1], 10);
  inc.solveToOptimum();
  check('2.4 x2 ≤ 10 satisfied', inc.getX()[1] <= 10 + 1e-9);
}

// =============================================================================
console.log('\n[3] Remove constraint');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3], [1, 0]], b: [100, 90, 30]});
  inc.solveToOptimum();
  // Drop x1 ≤ 30 (index 2 — the third constraint).
  inc.applyRemoveConstraint(2);
  inc.solveToOptimum();
  const stat = solveLPInternal({sense: 'max', c: [3, 5], A_ub: [[2, 1], [1, 3]], b_ub: [100, 90]});
  check('3.1 z matches static after remove', close(inc.getZ(), stat.objective));
  check('3.2 x matches static after remove', arrClose(inc.getX(), stat.x));
  check('3.3 numConstraints reduced',         inc.tab.length - 1 === 2);
}

// =============================================================================
console.log('\n[4] Change objective');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [1, 1], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  const z1 = inc.getZ();
  inc.applyChangeObjective([10, 1]);
  inc.solveToOptimum();
  check('4.1 z increased after favouring x1',  inc.getZ() > z1);
  // Static cross-check
  const stat = solveLPInternal({sense: 'max', c: [10, 1], A_ub: [[2, 1], [1, 3]], b_ub: [100, 90]});
  check('4.2 z matches static',  close(inc.getZ(), stat.objective));
  check('4.3 x matches static',  arrClose(inc.getX(), stat.x));
}

// =============================================================================
console.log('\n[5] Add variable');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  // New variable with column [1, 1]^T and very high obj coefficient — should enter.
  inc.applyAddVariable([1, 1], 100);
  inc.solveToOptimum();
  check('5.1 numStruct = 3',           inc.numStruct === 3);
  check('5.2 new x3 in basis',         inc.basis.includes(2));
  // Static check
  const stat = solveLPInternal({sense: 'max', c: [3, 5, 100], A_ub: [[2, 1, 1], [1, 3, 1]], b_ub: [100, 90]});
  check('5.3 x matches static',  arrClose(inc.getX(), stat.x));
  check('5.4 z matches static',  close(inc.getZ(), stat.objective));
}

// =============================================================================
console.log('\n[6] Remove variable (non-basic)');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5, -10], A: [[2, 1, 1], [1, 3, 1]], b: [100, 90]});
  inc.solveToOptimum();
  // x3 has negative obj, will be non-basic. Remove it.
  check('6.1 x3 non-basic', !inc.basis.includes(2));
  const zBefore = inc.getZ();
  inc.applyRemoveVariable(2);
  check('6.2 numStruct = 2',  inc.numStruct === 2);
  check('6.3 z preserved',    close(inc.getZ(), zBefore));
  inc.solveToOptimum();
  const stat = solveLPInternal({sense: 'max', c: [3, 5], A_ub: [[2, 1], [1, 3]], b_ub: [100, 90]});
  check('6.4 z matches static', close(inc.getZ(), stat.objective));
}

// =============================================================================
console.log('\n[7] Remove variable (basic)');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  check('7.1 x1 is basic before removal', inc.basis.includes(0));
  inc.applyRemoveVariable(0);             // remove x1
  check('7.2 numStruct = 1',               inc.numStruct === 1);
  inc.solveToOptimum();
  const stat = solveLPInternal({sense: 'max', c: [5], A_ub: [[1], [3]], b_ub: [100, 90]});
  check('7.3 z matches static',  close(inc.getZ(), stat.objective));
  check('7.4 x matches static',  arrClose(inc.getX(), stat.x));
}

// =============================================================================
console.log('\n[8] Idempotence: same modification twice should not destabilise');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  inc.applyChangeObjective([3, 5]);
  inc.solveToOptimum();
  inc.applyChangeObjective([3, 5]);
  inc.solveToOptimum();
  check('8.1 z stable under no-op obj change', close(inc.getZ(), 206));
}

// =============================================================================
console.log('\n[9] Detect unboundedness after constraint removal');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [1, 1], A: [[1, 1]], b: [10]});
  inc.solveToOptimum();
  check('9.1 bounded with single constraint', inc.status === 'optimal');
  inc.applyRemoveConstraint(0);
  inc.solveToOptimum();
  check('9.2 unbounded after dropping the only constraint', inc.status === 'unbounded');
}

// =============================================================================
console.log('\n[10] Snapshot integrity');
// =============================================================================
{
  const inc = new IncrementalLP({sense: 'max', c: [3, 5], A: [[2, 1], [1, 3]], b: [100, 90]});
  inc.solveToOptimum();
  const snap = inc.snapshot();
  check('10.1 snapshot has correct numStruct',     snap.numStruct === 2);
  check('10.2 snapshot has correct numConstraints', snap.numConstraints === 2);
  check('10.3 snapshot z matches getZ',             close(snap.z, inc.getZ()));
  check('10.4 snapshot reduced costs all ≥ 0',      snap.reducedCosts.every(r => r >= -1e-9));
  check('10.5 snapshot rhs all ≥ 0',                snap.rhs.every(v => v >= -1e-9));
  check('10.6 snapshot isOptimal flag',             snap.isOptimal === true);
  check('10.7 snapshot mode = optimal',             snap.mode === 'optimal');
}

// =============================================================================
console.log('\n[11] Many modifications agree with static solver throughout');
// =============================================================================
{
  function rng(seed: number): () => number {
    let s = seed >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const r = rng(424242);
  let okCount = 0, total = 0;
  for (let trial = 0; trial < 12; trial++) {
    const n = 2 + Math.floor(r() * 3);
    const m = 2 + Math.floor(r() * 3);
    const c = Array.from({length: n}, () => 1 + Math.floor(r() * 9));
    const A = Array.from({length: m}, () => Array.from({length: n}, () => 1 + Math.floor(r() * 5)));
    const b = Array.from({length: m}, () => 30 + Math.floor(r() * 40));
    const incLP = new IncrementalLP({sense: 'max', c: c.slice(), A: A.map(r => r.slice()), b: b.slice()});
    incLP.solveToOptimum();
    const stat = solveLPInternal({sense: 'max', c, A_ub: A, b_ub: b});
    total++;
    if (close(incLP.getZ(), stat.objective, 1e-7)) okCount++;
  }
  check(`11.1 random LPs agree (${okCount}/${total})`, okCount === total, `${okCount}/${total}`);
}

console.log(`\n${pass + fail} tests: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
