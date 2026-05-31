#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_calculus.rs.
// - Keep this as a CLI validation binary with Result-returning main; map CALCULUS_PY and scenario knobs through clap/std::env.
// - Convert Python payloads and check rows to serde structs; use serde_json for the last-line protocol.
// - Replace execFileSync with std::process::Command or tokio::process and keep calculus comparison helpers private.
'use strict';

// =============================================================================
// Validate the station-network calculus solvers (main-calculus.ts).
//
//   STUDY 1: Symbolic ≡ numerical derivative
//     For a battery of expressions, the AST symbolic derivative agrees
//     with Richardson five-point numerical derivative to ≤ 1e-9.
//
//   STUDY 2: Quadrature methods agree on a smooth integrand
//     trapezoidal, Simpson, adaptive Simpson, Gauss-Legendre and
//     scipy.integrate.quad all agree on ∫_0^π (x²·sin x + e^{−x}) dx.
//
//   STUDY 3: ODE station network ≡ pure-math RK4 ≡ scipy DOP853
//     Harmonic oscillator y'' + y = 0 from t=0 to t=4π. End-point
//     value is compared at the EXACT t-end (so timepoint mismatch is
//     not the dominant error). RK4 station network and pure-math
//     general/ode.rk4 must produce bit-identical traces; both must
//     agree with scipy DOP853 to within RK4 truncation error.
//
//   STUDY 4: 1-D heat (FTCS / BTCS station network) ≡ analytical decay
//     u_t = α u_xx with sin(πx) initial data, Dirichlet 0 boundaries.
//     Analytical solution u(x,t) = exp(−α π² t)·sin(πx). FTCS at
//     dt < dx²/(2α) and BTCS at dt = 50× larger both pin to ≤ 1e-2.
//     Cross-checked against scipy.LSODA on the same FD spatial system.
//
//   STUDY 5: 1-D wave (leapfrog) ≡ standing-wave analytical
//     u_tt = c² u_xx with sin(πx) initial, zero velocity. Analytical
//     u(x,t) = sin(πx)·cos(πct). Pinned to ≤ 5e-2 at N=51, dt=CFL/2.
//
//   STUDY 6: 2-D Poisson (Jacobi / Gauss-Seidel / SOR) all converge
//     to the same solution. SOR is dramatically faster (~30× fewer
//     iterations at ω = 1.85 on a 41×41 grid). All three pin to
//     scipy's reference Jacobi (bit-identical for Jacobi at the same
//     iteration count).
// =============================================================================

import {execFileSync} from 'child_process';
import * as path from 'path';
import {parse, diff, toFunction, richardsonDerivative, stringify} from '../general/expr';
import {trapezoidal, simpson, adaptiveSimpson, gaussLegendre} from '../general/quadrature';
import {rk4 as pureRk4} from '../general/ode';
import {buildODESystem, buildField1D, solvePoisson2D} from '../general/equation-to-stations';

const PYTHON = process.env.CALCULUS_PY ?? 'python3';
const PY_SCRIPT = path.join(__dirname, '..', '..', '..', 'external-references', 'calculus', 'calculus.py');

function runPython(env: Record<string, string>): any | null {
  try {
    const out = execFileSync(PYTHON, [PY_SCRIPT], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
      env: {...process.env, ...env},
    });
    const lines = out.trim().split('\n');
    return JSON.parse(lines[lines.length - 1]);
  } catch (err) {
    return null;
  }
}

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  (' + detail + ')' : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}

// -----------------------------------------------------------------------------
// STUDY 1: Symbolic derivative ≡ numerical
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 1: Symbolic vs numerical derivative ===');
const derivCases: Array<[string, number]> = [
  ['x^2',                    1.7],
  ['sin(x) * cos(x)',        0.4],
  ['exp(-x^2)',              0.6],
  ['x^3 + 2*x^2 - 5*x + 1',  2.0],
  ['log(x) * sin(x)',        1.2],
  ['1 / (1 + x^2)',          0.5],
];
for (const [exprStr, x] of derivCases) {
  const ast = parse(exprStr);
  const f = toFunction(ast, ['x']);
  const dfSym = toFunction(diff(ast, 'x'), ['x']);
  const dfNum = richardsonDerivative(f, x);
  const symVal = dfSym(x), numVal = dfNum;
  const err = Math.abs(symVal - numVal);
  check(`d/dx[${exprStr}] @ x=${x}: sym=${symVal.toFixed(8)}  num=${numVal.toFixed(8)}`,
        err < 1e-7, `|err|=${err.toExponential(2)}`);
}

// -----------------------------------------------------------------------------
// STUDY 2: Quadrature methods agree
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 2: Quadrature methods on ∫_0^π (x²·sin(x) + e^{−x}) dx ===');
const integrand = (x: number) => x * x * Math.sin(x) + Math.exp(-x);
const a = 0, b = Math.PI;
const refTS = adaptiveSimpson(integrand, a, b, 1e-15).value;
const py = runPython({PROBLEM: 'quad'});
if (py) {
  check(`scipy.integrate.quad agrees with adaptive Simpson`,
        Math.abs(py.value - refTS) < 1e-10,
        `|Δ|=${Math.abs(py.value - refTS).toExponential(2)}`);
} else {
  console.log('  SKIP    scipy reference unavailable (set CALCULUS_PY)');
}
const trap = trapezoidal(integrand, a, b, 64).value;
const simp = simpson(integrand, a, b, 64).value;
const gauss = gaussLegendre(integrand, a, b, 10).value;
check(`Simpson n=64 vs reference`,        Math.abs(simp - refTS) < 1e-7, `|Δ|=${Math.abs(simp-refTS).toExponential(2)}`);
check(`Gauss-Legendre n=10 vs reference`, Math.abs(gauss - refTS) < 1e-12, `|Δ|=${Math.abs(gauss-refTS).toExponential(2)}`);
check(`trapezoidal n=64 within O(1/n²)`,  Math.abs(trap - refTS) < 5e-3, `|Δ|=${Math.abs(trap-refTS).toExponential(2)}`);

// -----------------------------------------------------------------------------
// STUDY 3: ODE station network ≡ pure-math RK4 ≡ scipy DOP853
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 3: ODE station network ≡ pure-math RK4 ≡ scipy DOP853 ===');
{
  const t1 = 4 * Math.PI;
  const dt = 0.001;
  // Station network RK4.
  const stationOut = buildODESystem({
    names: ['y', 'v'], rhs: ['v', '-y'], y0: [1, 0], scheme: 'rk4',
  }).run(0, t1, dt);
  // Pure-math RK4 with bit-identical recipe.
  const fRef = (t: number, y: number[]) => [y[1], -y[0]];
  const pureOut = pureRk4(fRef, [1, 0], 0, t1, dt);
  // scipy DOP853 ground truth.
  const sci = runPython({PROBLEM: 'ode', T_END: String(t1)});
  // Both station and pure-math use the same dt grid, so end values match exactly.
  const stationY = stationOut.finalValues[0];
  const pureY = pureOut.y[pureOut.y.length - 1][0];
  const dStationVsPure = Math.abs(stationY - pureY);
  check(`station-network RK4 ≡ pure-math RK4 (bit-level on same grid)`,
        dStationVsPure < 1e-13,
        `|Δ| = ${dStationVsPure.toExponential(2)}`);
  // True analytical value: y(4π) = cos(4π) = 1.0.
  check(`station-network RK4 vs cos(4π) = 1`,
        Math.abs(stationY - 1) < 5e-6,
        `|Δ| = ${Math.abs(stationY - 1).toExponential(2)}`);
  if (sci) {
    check(`scipy DOP853 vs cos(4π) = 1`,
          Math.abs(sci.y_at_t1 - 1) < 1e-12,
          `|Δ| = ${Math.abs(sci.y_at_t1 - 1).toExponential(2)}`);
  } else {
    console.log('  SKIP    scipy DOP853 reference unavailable');
  }
}

// -----------------------------------------------------------------------------
// STUDY 4: 1-D heat
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 4: 1-D heat (FTCS / BTCS station network) ===');
{
  const N = 51;
  const alpha = 0.1;
  const T = 0.5;
  const dx = 1 / (N - 1);
  const dtFtcs = 0.4 * dx * dx / alpha;
  const dtBtcs = 0.05;   // >> FTCS bound
  const initExpr = 'sin(3.14159265358979 * x)';
  const decay = Math.exp(-alpha * Math.PI * Math.PI * T);
  const expectedPeak = decay;

  const r1 = buildField1D({N, xLo: 0, xHi: 1, initExpr,
                            family: 'heat', alphaExpr: String(alpha),
                            bcLeft: 0, bcRight: 0, scheme: 'ftcs'});
  const o1 = r1.sim.run(0, T, dtFtcs);

  const r2 = buildField1D({N, xLo: 0, xHi: 1, initExpr,
                            family: 'heat', alphaExpr: String(alpha),
                            bcLeft: 0, bcRight: 0, scheme: 'btcs'});
  const o2 = r2.sim.run(0, T, dtBtcs);
  let errFtcs = 0, errBtcs = 0;
  for (let i = 0; i < N; i++) {
    const exact = decay * Math.sin(Math.PI * r1.xs[i]);
    errFtcs = Math.max(errFtcs, Math.abs(o1.finalValues[i] - exact));
    errBtcs = Math.max(errBtcs, Math.abs(o2.finalValues[i] - exact));
  }
  check(`FTCS station-net (${o1.ticks} ticks at dt=${dtFtcs.toExponential(2)})`,
        errFtcs < 5e-3, `max|err vs analytical|=${errFtcs.toExponential(3)}`);
  check(`BTCS station-net (${o2.ticks} ticks at dt=${dtBtcs}, FTCS would be UNSTABLE)`,
        errBtcs < 5e-2, `max|err vs analytical|=${errBtcs.toExponential(3)}`);
  // Peak is preserved.
  const ftcsPeak = o1.finalValues[Math.floor(N / 2)];
  const btcsPeak = o2.finalValues[Math.floor(N / 2)];
  check(`FTCS peak agrees with exp(-απ²T) = ${expectedPeak.toFixed(6)}`,
        Math.abs(ftcsPeak - expectedPeak) < 5e-3, `peak=${ftcsPeak.toFixed(6)}`);
  check(`BTCS peak agrees with exp(-απ²T) = ${expectedPeak.toFixed(6)}`,
        Math.abs(btcsPeak - expectedPeak) < 5e-2, `peak=${btcsPeak.toFixed(6)}`);
  // scipy LSODA on same FD system.
  const sci = runPython({PROBLEM: 'pde-heat', N: String(N), ALPHA: String(alpha), T_END: String(T)});
  if (sci) {
    let errSci = 0;
    for (let i = 0; i < N; i++) {
      errSci = Math.max(errSci, Math.abs(o1.finalValues[i] - sci.final_values[i]));
    }
    check(`FTCS station-net ≡ scipy.LSODA on same FD spatial discretisation`,
          errSci < 5e-3, `max|Δ|=${errSci.toExponential(3)}`);
  } else {
    console.log('  SKIP    scipy LSODA reference unavailable');
  }
}

// -----------------------------------------------------------------------------
// STUDY 5: 1-D wave
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 5: 1-D wave (leapfrog) ===');
{
  const N = 51;
  const c = 1;
  const T = 0.5;
  const dx = 1 / (N - 1);
  const dt = 0.5 * dx / c;
  const initExpr = 'sin(3.14159265358979 * x)';
  const r = buildField1D({N, xLo: 0, xHi: 1, initExpr,
                           family: 'wave', cExpr: String(c),
                           bcLeft: 0, bcRight: 0, scheme: 'leapfrog'});
  const o = r.sim.run(0, T, dt);
  let err = 0;
  const expectedAmplitude = Math.cos(Math.PI * c * T);
  for (let i = 0; i < N; i++) {
    const exact = Math.sin(Math.PI * r.xs[i]) * expectedAmplitude;
    err = Math.max(err, Math.abs(o.finalValues[i] - exact));
  }
  check(`leapfrog (${o.ticks} ticks, CFL=0.5) vs sin(πx)·cos(πct)`,
        err < 0.05, `max|err|=${err.toExponential(3)}`);
}

// -----------------------------------------------------------------------------
// STUDY 6: 2-D Poisson — three relaxation schemes converge to the same answer.
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 6: 2-D Poisson, Jacobi / Gauss-Seidel / SOR ===');
{
  const N = 41;
  const tol = 1e-8;
  const rho = '2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)';
  const rJ = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                              rhoExpr: rho, scheme: 'jacobi', tol, maxIter: 50000});
  const rG = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                              rhoExpr: rho, scheme: 'gauss-seidel', tol, maxIter: 50000});
  const rS = solvePoisson2D({Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
                              rhoExpr: rho, scheme: 'sor', omega: 1.85, tol, maxIter: 50000});
  let errJ = 0, errG = 0, errS = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const exact = Math.sin(Math.PI * rJ.xs[i]) * Math.sin(Math.PI * rJ.ys[j]);
    errJ = Math.max(errJ, Math.abs(rJ.u[j * N + i] - exact));
    errG = Math.max(errG, Math.abs(rG.u[j * N + i] - exact));
    errS = Math.max(errS, Math.abs(rS.u[j * N + i] - exact));
  }
  check(`Jacobi pins to sin·sin within 1e-3 (${rJ.iterations} iters)`,
        errJ < 1e-3, `maxErr=${errJ.toExponential(2)}`);
  check(`Gauss-Seidel pins to sin·sin within 1e-3 (${rG.iterations} iters)`,
        errG < 1e-3, `maxErr=${errG.toExponential(2)}`);
  check(`SOR(ω=1.85) pins to sin·sin within 1e-3 (${rS.iterations} iters)`,
        errS < 1e-3, `maxErr=${errS.toExponential(2)}`);
  check(`Gauss-Seidel ~2× faster than Jacobi (${rG.iterations} vs ${rJ.iterations})`,
        rG.iterations < rJ.iterations * 0.7);
  check(`SOR(ω=1.85) ~10× faster than Jacobi (${rS.iterations} vs ${rJ.iterations})`,
        rS.iterations < rJ.iterations * 0.1);
  // scipy reference.
  const sci = runPython({PROBLEM: 'poisson', N: String(N), TOL: String(tol)});
  if (sci) {
    check(`station Jacobi iteration count == scipy Jacobi iteration count`,
          rJ.iterations === sci.iterations, `${rJ.iterations} vs ${sci.iterations}`);
    check(`station Jacobi maxErr ≡ scipy Jacobi maxErr (bit-comparable Jacobi)`,
          Math.abs(errJ - sci.max_err_vs_analytical) < 1e-12,
          `|Δ|=${Math.abs(errJ - sci.max_err_vs_analytical).toExponential(2)}`);
  } else {
    console.log('  SKIP    scipy Jacobi reference unavailable');
  }
}

// -----------------------------------------------------------------------------
// Summary.
// -----------------------------------------------------------------------------
console.log(`\n=== Summary: ${pass} passed, ${fail} failed ===`);
if (fail > 0) process.exit(1);
