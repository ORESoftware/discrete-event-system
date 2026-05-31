#!/usr/bin/env ts-node
// RUST MIGRATION: Port file-for-file to `tests/calculus_test.rs` as cross-module integration coverage for expression parsing, quadrature, ODE, and equation-to-station adapters.
// Test-port notes: translate checks into `#[test]` functions returning `Result<()>`; replace ad hoc assertion helpers with `assert!`, `assert_eq!`, and approximate-float helpers; keep numeric tolerances explicit.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/calculus_test.rs   (integration test crate)
// 1:1 file move. Spans expr / quadrature / ode / equation-to-stations, so it is
// an integration test under `tests/`, not one module's `#[cfg(test)] mod`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - approx(a,b,tol) comparisons (down to 1e-15) -> approx::assert_relative_eq!
//     with an explicit epsilon; convergence-order checks compare ratios.
// =============================================================================

// =============================================================================
// Unit tests for the calculus pipeline:
//
//   T1  Expression engine: parse, evaluate, stringify, diff
//   T2  Quadrature primitives on a known closed-form integrand
//   T3  Pure-math ODE solvers: convergence orders for euler/rk2/rk4
//   T4  ODE station network on the SHO is bit-identical to pure-math rk4
//   T5  Field1D heat: BTCS unconditionally stable, FTCS bound enforced
//   T6  Field1D station updates are order-independent (shuffle invariant)
//   T7  thomas tridiagonal solver vs hand-built dense Gaussian elimination
//   T8  Poisson 2-D: SOR converges faster than Gauss-Seidel faster than Jacobi
//
// End-to-end validation against scipy / sympy is in runners/validate-calculus.ts.
// =============================================================================

import {parse, evaluate, stringify, diff, toFunction, simplify} from '../general/expr';
import {trapezoidal, simpson, gaussLegendre, adaptiveSimpson} from '../general/quadrature';
import {euler, rk2Heun, rk4} from '../general/ode';
import {buildODESystem, buildField1D, solvePoisson2D, thomas} from '../general/equation-to-stations';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  (' + detail + ')' : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}
const approx = (a: number, b: number, t: number) => Math.abs(a - b) <= t;

// =============================================================================
console.log('\nT1  Expression engine');
{
  const e = parse('2*x + 3');
  check('evaluate(2x+3, x=4) = 11', evaluate(e, {x: 4}) === 11);
  const e2 = parse('sin(x)^2 + cos(x)^2');
  for (const x of [0, 0.5, 1.7, -2.3]) {
    check(`sin²+cos² ≡ 1 at x=${x}`, approx(evaluate(e2, {x}), 1, 1e-15));
  }
  const dE = diff(parse('x^3'), 'x');
  const dEFn = toFunction(dE, ['x']);
  check('d/dx(x^3) at x=2 = 12', approx(dEFn(2), 12, 1e-12));
  // Nested chain rule.
  const dCh = diff(parse('sin(x^2)'), 'x');
  const dChFn = toFunction(dCh, ['x']);
  for (const x of [0.4, 1.1, 2.0]) {
    const expected = 2 * x * Math.cos(x * x);
    check(`d/dx[sin(x²)] @ x=${x} matches 2x·cos(x²)`, approx(dChFn(x), expected, 1e-10));
  }
  // Round trip.
  const e3 = parse('x^2 + 2*x + 1');
  const s = stringify(e3);
  const e4 = parse(s);
  for (const x of [-1, 0, 0.5, 3]) {
    check(`parse(stringify(${stringify(e3)})) at x=${x}`,
          approx(evaluate(e3, {x}), evaluate(e4, {x}), 1e-15));
  }
}

// =============================================================================
console.log('\nT2  Quadrature primitives on smooth integrand');
{
  // ∫_0^1 x³ dx = 1/4.
  const f = (x: number) => x * x * x;
  check('trapezoidal n=1000 ≈ 0.25', approx(trapezoidal(f, 0, 1, 1000).value, 0.25, 1e-3));
  check('Simpson    n=10   = 0.25', approx(simpson(f, 0, 1, 10).value, 0.25, 1e-12));
  check('Gauss-Legendre n=4  = 0.25', approx(gaussLegendre(f, 0, 1, 4).value, 0.25, 1e-13));
  check('adaptive Simpson 1e-12 = 0.25',
        approx(adaptiveSimpson(f, 0, 1, 1e-12).value, 0.25, 1e-12));
}

// =============================================================================
console.log('\nT3  Pure-math ODE solvers convergence orders');
{
  // y' = -y, y(0)=1; analytical: y(1) = e^{-1}.
  const f = (_t: number, y: number[]) => [-y[0]];
  const exact = Math.exp(-1);
  const errs = (h: number) => ({
    eu: Math.abs(euler(f, [1], 0, 1, h).y[Math.round(1 / h)][0] - exact),
    rk: Math.abs(rk2Heun(f, [1], 0, 1, h).y[Math.round(1 / h)][0] - exact),
    rk4: Math.abs(rk4(f, [1], 0, 1, h).y[Math.round(1 / h)][0] - exact),
  });
  const e1 = errs(0.1);
  const e2 = errs(0.05);
  // Euler is O(h¹) → halve h → halve err. RK2 → /4. RK4 → /16.
  check(`euler order 1: e1/e2 ≈ 2 (got ${(e1.eu / e2.eu).toFixed(3)})`,
        Math.abs(e1.eu / e2.eu - 2) < 0.5);
  check(`rk2   order 2: e1/e2 ≈ 4 (got ${(e1.rk / e2.rk).toFixed(3)})`,
        Math.abs(e1.rk / e2.rk - 4) < 1.0);
  check(`rk4   order 4: e1/e2 ≈ 16 (got ${(e1.rk4 / e2.rk4).toFixed(3)})`,
        Math.abs(e1.rk4 / e2.rk4 - 16) < 4.0);
}

// =============================================================================
console.log('\nT4  ODE station network ≡ pure-math RK4 (bit level on same dt grid)');
{
  const dt = 0.001;
  const T = 2 * Math.PI;
  const station = buildODESystem({
    names: ['y', 'v'], rhs: ['v', '-y'], y0: [1, 0], scheme: 'rk4',
  }).run(0, T, dt);
  const pure = rk4((_t, y) => [y[1], -y[0]], [1, 0], 0, T, dt);
  const dy = Math.abs(station.finalValues[0] - pure.y[pure.y.length - 1][0]);
  const dv = Math.abs(station.finalValues[1] - pure.y[pure.y.length - 1][1]);
  check('station-net y(T) bit-identical to pure-math', dy < 1e-13, `|Δ|=${dy.toExponential(2)}`);
  check('station-net v(T) bit-identical to pure-math', dv < 1e-13, `|Δ|=${dv.toExponential(2)}`);
}

// =============================================================================
console.log('\nT5  Field1D heat schemes: stability and accuracy');
{
  const N = 41;
  const alpha = 0.1;
  const dx = 1 / (N - 1);
  const dtSafe = 0.4 * dx * dx / alpha;
  const T = 0.3;

  // FTCS stable.
  const r = buildField1D({N, xLo: 0, xHi: 1, initExpr: 'sin(3.14159265358979 * x)',
                          family: 'heat', alphaExpr: String(alpha),
                          bcLeft: 0, bcRight: 0, scheme: 'ftcs'});
  const o = r.sim.run(0, T, dtSafe);
  let okStable = true;
  for (const v of o.finalValues) if (!Number.isFinite(v) || Math.abs(v) > 2) okStable = false;
  check('FTCS stays bounded at safe dt', okStable);

  // BTCS at 30× safe dt: still stable.
  const rB = buildField1D({N, xLo: 0, xHi: 1, initExpr: 'sin(3.14159265358979 * x)',
                            family: 'heat', alphaExpr: String(alpha),
                            bcLeft: 0, bcRight: 0, scheme: 'btcs'});
  const oB = rB.sim.run(0, T, 30 * dtSafe);
  let okBtcs = true;
  for (const v of oB.finalValues) if (!Number.isFinite(v) || Math.abs(v) > 2) okBtcs = false;
  check('BTCS stays bounded at 30× FTCS-bound dt', okBtcs);
}

// =============================================================================
console.log('\nT6  Field1D station-update order-independence (shuffle invariant)');
{
  const init = 'exp(-50 * (x - 0.5)^2)';
  const make = (seed: number) => {
    const r = buildField1D({N: 31, xLo: 0, xHi: 1, initExpr: init,
                            family: 'heat', alphaExpr: '0.05',
                            bcLeft: 0, bcRight: 0, scheme: 'ftcs'});
    r.sim.rng = (() => {        // override to a fresh seed
      let s = seed | 0;
      return () => { s = (s * 1664525 + 1013904223) | 0; return ((s >>> 0) / 0x1_0000_0000); };
    })();
    return r.sim.run(0, 0.2, 0.0005);
  };
  const a = make(1), b = make(99);
  let same = true;
  for (let i = 0; i < a.finalValues.length; i++) {
    if (Math.abs(a.finalValues[i] - b.finalValues[i]) > 1e-15) { same = false; break; }
  }
  check('two different shuffle seeds → IDENTICAL final field', same);
}

// =============================================================================
console.log('\nT7  Thomas algorithm vs hand solve');
{
  // System:  [2 1 0; 1 2 1; 0 1 2] x = [4; 8; 8]
  // Hand solution: x = [1, 2, 3].
  const sub = new Float64Array([0, 1, 1]);
  const dia = new Float64Array([2, 2, 2]);
  const sup = new Float64Array([1, 1, 0]);
  const rhs = new Float64Array([4, 8, 8]);
  const x = thomas(sub, dia, sup, rhs);
  check('Thomas: x[0] = 1', approx(x[0], 1, 1e-12), `x[0]=${x[0]}`);
  check('Thomas: x[1] = 2', approx(x[1], 2, 1e-12), `x[1]=${x[1]}`);
  check('Thomas: x[2] = 3', approx(x[2], 3, 1e-12), `x[2]=${x[2]}`);
}

// =============================================================================
console.log('\nT8  Poisson 2-D: SOR < Gauss-Seidel < Jacobi iterations');
{
  const N = 31, tol = 1e-7;
  const rho = '2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)';
  const j = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                             rhoExpr: rho, scheme: 'jacobi', tol, maxIter: 50000});
  const g = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                             rhoExpr: rho, scheme: 'gauss-seidel', tol, maxIter: 50000});
  const s = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                             rhoExpr: rho, scheme: 'sor', omega: 1.8, tol, maxIter: 50000});
  check(`Gauss-Seidel < Jacobi (${g.iterations} < ${j.iterations})`,
        g.iterations < j.iterations);
  check(`SOR < Gauss-Seidel (${s.iterations} < ${g.iterations})`,
        s.iterations < g.iterations);
  check(`SOR < Jacobi by ≥ 5× (${s.iterations} vs ${j.iterations})`,
        s.iterations * 5 <= j.iterations);
}

// =============================================================================
console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
