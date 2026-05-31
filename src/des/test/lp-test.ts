// RUST MIGRATION: Port file-for-file to `tests/lp_test.rs` as integration coverage for internal simplex, DES LP solving, and MDP-as-LP transforms.
// Test-port notes: translate solver cases into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, and approximate-float helpers; keep LP fixtures deterministic.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/lp_test.rs   (integration test crate)
// 1:1 file move. Spans lp / lp-des / des-lp-bridge / value-iteration, so it is
// an integration test under `tests/`, not a single module's `#[cfg(test)] mod`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - approx(a,b,tol) and maxAbs() float comparisons -> the `approx` crate:
//     assert_relative_eq!(a, b, max_relative = 1e-7) (and abs_diff for vectors).
// =============================================================================

// =============================================================================
// Unit tests for the LP infrastructure: in-process simplex, DES-engine
// simplex, and MDP-as-LP transformation.
//
// Run with: node dist/des/test/lp-test.js
// =============================================================================

import {LPProblem, solveLPInternal, solveLPExternal, lpToString} from '../general/lp';
import {solveLPViaDES} from '../general/lp-des';
import {buildMDPLP, solveMDPAsLP} from '../general/des-lp-bridge';
import {MDPSpec, valueIteration} from '../general/value-iteration';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function approx(a: number, b: number, tol = 1e-7) { return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)); }
function maxAbs(u: ArrayLike<number>, v: ArrayLike<number>) {
  let m = 0;
  for (let i = 0; i < u.length; i++) m = Math.max(m, Math.abs(u[i] - v[i]));
  return m;
}

// =============================================================================
console.log('=== Unit: in-process simplex, hand-checkable LPs ===');
{
  // 1. Trivial: max x  s.t. x ≤ 5  → x* = 5
  {
    const lp: LPProblem = {sense: 'max', c: [1], A_ub: [[1]], b_ub: [5]};
    const r = solveLPInternal(lp);
    check('max x s.t. x ≤ 5  → x*=5', r.status === 'optimal' && approx(r.x[0], 5),
          `x=${r.x[0]} obj=${r.objective}`);
  }
  // 2. Multi-vertex 2-D: max 3x+2y s.t. x+y≤4, x+3y≤6, x,y≥0  → (4, 0), obj 12
  {
    const lp: LPProblem = {sense: 'max', c: [3, 2], A_ub: [[1, 1], [1, 3]], b_ub: [4, 6]};
    const r = solveLPInternal(lp);
    check('2-var: optimum at (4,0), obj=12',
          r.status === 'optimal' && approx(r.x[0], 4) && approx(r.x[1], 0) && approx(r.objective, 12));
  }
  // 3. Equality constraint: x + y = 1, max x → x*=1, y*=0
  {
    const lp: LPProblem = {sense: 'max', c: [1, 0], A_eq: [[1, 1]], b_eq: [1]};
    const r = solveLPInternal(lp);
    check('equality: x+y=1, max x  → x*=1', r.status === 'optimal' && approx(r.x[0], 1) && approx(r.x[1], 0));
  }
  // 4. Infeasibility: x ≥ 5 AND x ≤ 3  (encoded as -x ≤ -5 and x ≤ 3)
  {
    const lp: LPProblem = {sense: 'max', c: [1], A_ub: [[-1], [1]], b_ub: [-5, 3]};
    const r = solveLPInternal(lp);
    check('infeasibility detected', r.status === 'infeasible');
  }
  // 5. Unboundedness: max x s.t. -x ≤ 1  (x ≥ -1, no upper bound)
  {
    const lp: LPProblem = {sense: 'max', c: [1], A_ub: [[-1]], b_ub: [1]};
    const r = solveLPInternal(lp);
    check('unbounded direction detected', r.status === 'unbounded');
  }
  // 6. Min-form: min x  s.t. x ≥ 2 (encoded as -x ≤ -2)
  {
    const lp: LPProblem = {sense: 'min', c: [1], A_ub: [[-1]], b_ub: [-2]};
    const r = solveLPInternal(lp);
    check('min x s.t. x≥2 → x*=2', r.status === 'optimal' && approx(r.x[0], 2));
  }
}

// =============================================================================
console.log('\n=== Unit: lpToString pretty-printer ===');
{
  const lp: LPProblem = {
    sense: 'max', c: [3, 2],
    A_ub: [[1, 1], [1, 3]], b_ub: [4, 6],
    varNames: ['x', 'y'],
  };
  const s = lpToString(lp);
  check('pretty-printer contains "max"', s.includes('max'));
  check('pretty-printer contains "x + y ≤ 4"', s.includes('x + y ≤ 4'));
  check('pretty-printer contains "x + 3y ≤ 6"', s.includes('x + 3y ≤ 6'));
}

// =============================================================================
console.log('\n=== Unit: DES-engine simplex bit-equivalence with in-process simplex ===');
{
  const cases: {name: string; lp: LPProblem}[] = [
    {name: 'simple max', lp: {sense: 'max', c: [3, 2], A_ub: [[1, 1], [1, 3]], b_ub: [4, 6]}},
    {name: 'redundant constraints', lp: {sense: 'max', c: [1, 1], A_ub: [[1, 1], [2, 2]], b_ub: [3, 6]}},
    {name: 'min with phase-1', lp: {sense: 'min', c: [1, 1], A_ub: [[-1, -1]], b_ub: [-1]}},
    {name: 'equality constraints', lp: {sense: 'max', c: [1, 1], A_eq: [[1, 1]], b_eq: [3]}},
  ];
  for (const {name, lp} of cases) {
    const internal = solveLPInternal(lp);
    const desD = solveLPViaDES(lp, {pivotRule: 'dantzig'});
    const desB = solveLPViaDES(lp, {pivotRule: 'bland'});
    check(`'${name}': all three solvers report ${internal.status}`,
          desD.status === internal.status && desB.status === internal.status);
    if (internal.status === 'optimal') {
      check(`'${name}': DES-Dantzig obj ≡ internal obj  (|Δ| ≤ 1e-9)`,
            approx(desD.objective, internal.objective, 1e-9),
            `Δ=${Math.abs(desD.objective - internal.objective).toExponential(2)}`);
      check(`'${name}': DES-Bland   obj ≡ internal obj  (|Δ| ≤ 1e-9)`,
            approx(desB.objective, internal.objective, 1e-9),
            `Δ=${Math.abs(desB.objective - internal.objective).toExponential(2)}`);
    }
  }
}

// =============================================================================
console.log('\n=== Unit: DES simplex pivot trace is monotone in obj (in phase 2) ===');
{
  // For a feasible bounded max LP, the objective along the pivot path
  // is non-decreasing in phase 2 (each pivot is an improving step).
  const lp: LPProblem = {sense: 'max', c: [3, 2], A_ub: [[1, 1], [1, 3], [2, 1]], b_ub: [4, 6, 7]};
  const r = solveLPViaDES(lp, {pivotRule: 'dantzig'});
  check('pivot trace exists', r.trace.pivotHistory.length > 0);
  let monotone = true;
  let prev = -Infinity;
  for (const p of r.trace.pivotHistory) {
    if (p.phase !== 2) continue;
    if (p.obj < prev - 1e-9) monotone = false;
    prev = p.obj;
  }
  check('phase-2 objective trace is non-decreasing across pivots', monotone);
  check('final obj equals optimum',
        approx(r.objective, prev),
        `final=${r.objective}  last-trace=${prev}`);
}

// =============================================================================
console.log('\n=== Unit: MDP-as-LP buildMDPLP shape check ===');
{
  // 3-state, 2-action MDP; expect 2 ≤ inequalities per non-terminal state.
  const mdp: MDPSpec = {
    numStates: 3,
    numActions: () => 2,
    outcomes: (s, a) => {
      if (s === 2) return [{prob: 1, reward: 0, nextState: 2}];
      const target = a === 1 ? Math.min(2, s + 1) : Math.max(0, s - 1);
      const reward = target === 2 ? 1 : 0;
      return [{prob: 1, reward, nextState: target}];
    },
    isTerminal: (s) => s === 2,
    terminalReward: () => 0,
  };
  const lp = buildMDPLP(mdp, 0.9);
  check('LP has 3 variables (one per state)', lp.c.length === 3);
  check('LP minimises uniform sum of V', lp.sense === 'min');
  check('μ_s = 1/N for uniform stationary measure', lp.c.every(v => approx(v, 1/3)));
  // Two ≤ rows per non-terminal state + 2 rows for the terminal pin = 6 rows.
  const numConstraints = (lp.A_ub?.length ?? 0);
  check('# inequality rows = 2*(non-terminal states × actions) + 2*(terminal pin)',
        numConstraints === 2 * 2 + 2 * 1, `got ${numConstraints}`);
}

// =============================================================================
console.log('\n=== Unit: solveMDPAsLP ≡ valueIteration on a tiny MDP ===');
{
  // 4-state line MDP, both directions stochastic 80/20.
  const mdp: MDPSpec = {
    numStates: 4,
    numActions: () => 2,
    outcomes: (s, a) => {
      if (s === 3) return [{prob: 1, reward: 0, nextState: 3}];
      const intended = a === 1 ? Math.min(3, s + 1) : Math.max(0, s - 1);
      const slip = a === 1 ? Math.max(0, s - 1) : Math.min(3, s + 1);
      const r = (sp: number) => sp === 3 ? 1 : 0;
      return [
        {prob: 0.8, reward: r(intended), nextState: intended},
        {prob: 0.2, reward: r(slip),     nextState: slip},
      ];
    },
    isTerminal: (s) => s === 3,
    terminalReward: () => 0,
  };
  const vi = valueIteration(mdp, {gamma: 0.9, tol: 1e-12, maxIter: 100000});
  const lp = solveMDPAsLP(mdp, 0.9);
  check('V*_LP ≡ V*_VI on stochastic 4-state line MDP (max|Δ| ≤ 1e-7)',
        maxAbs(lp.V, vi.V) < 1e-7,
        `max|Δ|=${maxAbs(lp.V, vi.V).toExponential(2)}`);
  let polMatch = true;
  for (let s = 0; s < 4; s++) if (s !== 3 && lp.policy[s] !== vi.policy[s]) polMatch = false;
  check('π*_LP ≡ π*_VI on every non-terminal state', polMatch);
}

// =============================================================================
console.log(`\nsummary: ${pass} pass, ${fail} fail`);
process.exit(fail === 0 ? 0 : 1);
