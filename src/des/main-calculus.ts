#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_calculus.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-calculus.rs   (fn main)
// 1:1 file move. CLI dispatching expr / ode / pde / poisson problems onto the
// station network and comparing against reference solvers.
//
// Conversion notes (file-specific):
//   - process.env (PROBLEM, RHS_i, y0, ...) -> std::env::var; dispatch ->
//     match on PROBLEM.
//   - expr parse -> use crate::des::general::expr (Expr enum, match on kind).
//   - quadrature/ode helpers -> use crate::des::general::{quadrature, ode}.
//   - fs/path output -> std::fs; top-level run -> fn main.
// =============================================================================

// =============================================================================
// MAIN-CALCULUS: parse a math expression / equation, build a station network,
// solve, and compare with reference solvers.
//
// Three problem classes, all dispatched from the same CLI:
//
//   PROBLEM=expr      Inspect a scalar expression. Reports value at a point,
//                     symbolic derivative, integral via 5 quadrature methods,
//                     and (where possible) a closed-form answer.
//
//   PROBLEM=ode       ODE system y'_i = f_i(t, y_1, …, y_n). Reads RHS_i and
//                     y0 from env vars. Solves via three schemes (Euler /
//                     RK2 / RK4) on the station network AND via the
//                     pure-math RK45 reference; reports max |Δ|.
//
//   PROBLEM=pde       1-D PDE via the Field1D station network. Heat / wave /
//                     advection. Solves at multiple resolutions; reports
//                     decay or wave-front correctness against analytical
//                     where available.
//
//   PROBLEM=poisson   2-D Poisson equation via the Field2D iterative
//                     relaxation. Compares Jacobi / Gauss-Seidel / SOR.
//
// All paths use the framework's station + census substrate; nothing about
// "ODE" or "PDE" is special-cased in the simulation engine — only the
// per-station updater changes.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {parse, stringify, diff, evaluate, toFunction, richardsonDerivative} from './general/expr';
import {trapezoidal, simpson, adaptiveSimpson, gaussLegendre, monteCarlo} from './general/quadrature';
import {rk45} from './general/ode';
import {buildODESystem, buildField1D, solvePoisson2D} from './general/equation-to-stations';
import {FrameRecorder} from './animation/frame-recorder';

async function main() {
  const problem = process.env.PROBLEM ?? 'expr';

  if (problem === 'expr') {
    const exprStr = process.env.EXPR ?? 'x^2 * sin(x) + exp(-x)';
    const xVal = Number(process.env.X ?? 1.0);
    const a = Number(process.env.A ?? 0);
    const b = Number(process.env.B ?? Math.PI);
    console.log(`# Expression: f(x) = ${exprStr}`);
    const f = parse(exprStr);
    const fJs = toFunction(f, ['x']);
    const dfdx = diff(f, 'x');
    const dfJs = toFunction(dfdx, ['x']);
    console.log(`# f(${xVal}) = ${fJs(xVal).toFixed(10)}`);
    console.log(`# Symbolic derivative: f'(x) = ${stringify(dfdx)}`);
    console.log(`# f'(${xVal}) symbolic   = ${dfJs(xVal).toFixed(10)}`);
    console.log(`# f'(${xVal}) Richardson = ${richardsonDerivative(fJs, xVal).toFixed(10)}`);
    console.log(`\n# Quadrature ∫_${a}^${b} f(x) dx, comparing 5 methods:`);
    const refTrue = adaptiveSimpson(fJs, a, b, 1e-15);
    const ref = refTrue.value;
    console.log(`#   reference (adaptive Simpson at tol 1e-15) = ${ref.toFixed(12)}  (${refTrue.evaluations} evals)`);
    const methods: Array<[string, () => {value: number; evaluations: number; stderr?: number}]> = [
      ['trapezoidal n=64',        () => trapezoidal(fJs, a, b, 64)],
      ['Simpson    n=64',         () => simpson(fJs, a, b, 64)],
      ['adaptive Simpson tol=1e-9', () => adaptiveSimpson(fJs, a, b, 1e-9)],
      ['Gauss-Legendre n=10',     () => gaussLegendre(fJs, a, b, 10)],
      ['Monte Carlo n=100k',      () => monteCarlo(fJs, a, b, 100_000)],
    ];
    for (const [name, fn] of methods) {
      const r = fn();
      const err = Math.abs(r.value - ref);
      console.log(`#   ${name.padEnd(28)} = ${r.value.toFixed(8)}  err=${err.toExponential(2)}  evals=${String(r.evaluations).padStart(6)}` +
                  (r.stderr !== undefined ? `  stderr=${r.stderr.toExponential(2)}` : ''));
    }
    return;
  }

  if (problem === 'ode') {
    // Default: SHO  y'' + ω² y = 0 → state [y, v], y' = v, v' = -ω² y.
    const omega = Number(process.env.OMEGA ?? 1.0);
    const names = (process.env.NAMES ?? 'y,v').split(',');
    const rhs = (process.env.RHS ?? `v;-${omega}*${omega}*y`).split(';');
    const y0 = (process.env.Y0 ?? '1,0').split(',').map(Number);
    const t1 = Number(process.env.T_END ?? 2 * Math.PI);
    const dt = Number(process.env.DT ?? 0.01);
    console.log(`# ODE system:  d/dt [${names.join(', ')}] = [${rhs.join(', ')}]`);
    console.log(`#   y(0) = [${y0.join(', ')}],  t ∈ [0, ${t1.toFixed(4)}],  dt = ${dt}`);

    // Pure-math reference: RK45 (adaptive).
    const rhsExprs = rhs.map(parse);
    const fns = rhsExprs.map(e => toFunction(e, ['t', ...names]));
    const fRef = (t: number, y: number[]) => fns.map(fi => fi(t, ...y));
    const ref = rk45(fRef, y0, 0, t1, {rtol: 1e-12, atol: 1e-14});

    console.log(`\n# Reference (RK45 adaptive, rtol=1e-12): ${ref.t.length} accepted steps`);
    const refFinal = ref.y[ref.y.length - 1];
    for (let i = 0; i < names.length; i++) console.log(`#   ${names[i]}(${t1.toFixed(4)}) = ${refFinal[i].toFixed(10)}`);

    console.log(`\n# Station-network solvers (one station per state variable, dt = ${dt}):`);
    for (const scheme of ['euler', 'rk2', 'rk4'] as const) {
      const sim = buildODESystem({names, rhs, y0, scheme}).run(0, t1, dt);
      let maxErr = 0;
      for (let i = 0; i < names.length; i++) {
        const e = Math.abs(sim.finalValues[i] - refFinal[i]);
        if (e > maxErr) maxErr = e;
      }
      const finalsStr = names.map((n, i) => `${n}=${sim.finalValues[i].toFixed(8)}`).join('  ');
      console.log(`#   ${scheme.padEnd(6)} ${finalsStr}  max|Δ vs RK45 ref| = ${maxErr.toExponential(3)}`);
    }

    const outDir = path.join(__dirname, '..', '..', 'out');
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
    fs.writeFileSync(path.join(outDir, 'calculus-ode.json'), JSON.stringify({
      names, rhs, y0, t1, dt,
      reference: {t: ref.t, y: ref.y},
    }));
    return;
  }

  if (problem === 'pde') {
    const family = (process.env.FAMILY ?? 'heat') as 'heat' | 'wave' | 'advection';
    const N = Number(process.env.N ?? 51);
    const T = Number(process.env.T_END ?? 0.5);
    if (family === 'heat') {
      const alpha = Number(process.env.ALPHA ?? 0.1);
      const initExpr = process.env.INIT ?? 'sin(3.14159265358979 * x)';
      // FTCS stability bound dx²/(2α). BTCS unconditionally stable; we run it past that.
      const dx = 1 / (N - 1);
      const dtSafe = 0.4 * dx * dx / alpha;
      const dtBig = 0.05;
      console.log(`# PDE: heat 1D  u_t = ${alpha} · u_xx,  init=${initExpr},  N=${N},  T=${T}`);
      console.log(`#   FTCS stability bound:  dt ≤ ${(dx*dx/(2*alpha)).toFixed(6)};  using dt=${dtSafe.toFixed(6)}`);
      console.log(`#   BTCS unconditionally stable;             using dt=${dtBig}`);
      // Build both, compare against analytical decay (works only for sin(πx) initial here).
      const rFtcs = buildField1D({
        N, xLo: 0, xHi: 1, initExpr, family: 'heat', alphaExpr: String(alpha),
        bcLeft: 0, bcRight: 0, scheme: 'ftcs',
      });
      const rBtcs = buildField1D({
        N, xLo: 0, xHi: 1, initExpr, family: 'heat', alphaExpr: String(alpha),
        bcLeft: 0, bcRight: 0, scheme: 'btcs',
      });
      const outFtcs = rFtcs.sim.run(0, T, dtSafe);
      const outBtcs = rBtcs.sim.run(0, T, dtBig);

      // Optional animation: ANIMATE=1 to record FTCS evolution.
      if (process.env.ANIMATE === '1') {
        const {STAGE_W, STAGE_H, buildField1DFrame, buildField1DChart} =
          await import('./animation/scenes/calculus-scene');
        const outDir = path.join(__dirname, '..', '..', 'out');
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
        const rec = new FrameRecorder({
          framesPath: path.join(outDir, 'calculus-heat1d.frames.jsonl'),
          htmlPath:   path.join(outDir, 'calculus-heat1d.html'),
          width: STAGE_W, height: STAGE_H, fps: 30,
          title: `1-D heat:  u_t = ${alpha}·u_xx,  N=${N},  scheme=FTCS`,
          subtitle: `Each station = one spatial cell;  Census snapshot drives FTCS update`,
        });
        const maxFrames = Number(process.env.ANIM_FRAMES ?? 200);
        const stride = Math.max(1, Math.floor(outFtcs.trace.values.length / maxFrames));
        let vMax = 0;
        for (const arr of outFtcs.trace.values) for (let i = 0; i < arr.length; i++) {
          if (Math.abs(arr[i]) > vMax) vMax = Math.abs(arr[i]);
        }
        for (let k = 0; k < outFtcs.trace.values.length; k += stride) {
          rec.frame(outFtcs.trace.t[k], k, () =>
            buildField1DFrame(outFtcs.trace.t[k], k, outFtcs.trace.values[k],
                              rFtcs.xs, vMax, 'FTCS', 'heat'));
        }
        rec.setCharts([buildField1DChart(outFtcs.trace)]);
        await rec.finish();
        console.log(`#   animation written to ${path.relative(process.cwd(), path.join(outDir, 'calculus-heat1d.html'))}`);
      }

      const decay = Math.exp(-alpha * Math.PI * Math.PI * T);
      let errFtcs = 0, errBtcs = 0;
      for (let i = 0; i < N; i++) {
        const exact = decay * Math.sin(Math.PI * rFtcs.xs[i]);
        errFtcs = Math.max(errFtcs, Math.abs(outFtcs.finalValues[i] - exact));
        errBtcs = Math.max(errBtcs, Math.abs(outBtcs.finalValues[i] - exact));
      }
      console.log(`\n# Final t = ${T},  analytical peak = exp(-απ²T) = ${decay.toFixed(6)}`);
      console.log(`#   FTCS (${outFtcs.ticks} ticks):  max|err| = ${errFtcs.toExponential(3)}`);
      console.log(`#   BTCS (${outBtcs.ticks} ticks):  max|err| = ${errBtcs.toExponential(3)}`);
      const outDir = path.join(__dirname, '..', '..', 'out');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
      fs.writeFileSync(path.join(outDir, 'calculus-heat1d.json'), JSON.stringify({
        N, T, alpha, dtSafe, dtBig,
        xs: rFtcs.xs,
        finalFtcs: Array.from(outFtcs.finalValues),
        finalBtcs: Array.from(outBtcs.finalValues),
        analytical: rFtcs.xs.map(x => decay * Math.sin(Math.PI * x)),
      }));
    } else if (family === 'wave') {
      const c = Number(process.env.C ?? 1);
      const initExpr = process.env.INIT ?? 'sin(3.14159265358979 * x)';
      const dx = 1 / (N - 1);
      const dt = 0.5 * dx / c;
      console.log(`# PDE: wave 1D  u_tt = ${c}² · u_xx,  init=${initExpr},  v(x,0)=0,  N=${N},  T=${T}`);
      console.log(`#   CFL bound:  c·dt/dx ≤ 1;  using dt=${dt.toFixed(6)} (c·dt/dx = 0.5)`);
      const r = buildField1D({
        N, xLo: 0, xHi: 1, initExpr, family: 'wave', cExpr: String(c),
        bcLeft: 0, bcRight: 0, scheme: 'leapfrog',
      });
      const out = r.sim.run(0, T, dt);
      // Standing wave: u(x,t) = sin(πx)·cos(πct).
      let err = 0;
      for (let i = 0; i < N; i++) {
        const exact = Math.sin(Math.PI * r.xs[i]) * Math.cos(Math.PI * c * T);
        err = Math.max(err, Math.abs(out.finalValues[i] - exact));
      }
      console.log(`#   leapfrog (${out.ticks} ticks):  max|err vs cos(πct)·sin(πx)| = ${err.toExponential(3)}`);
    }
    return;
  }

  if (problem === 'poisson') {
    const N = Number(process.env.N ?? 41);
    const rhoExpr = process.env.RHO ?? '2 * 3.14159265358979^2 * sin(3.14159265358979*x) * sin(3.14159265358979*y)';
    const tol = Number(process.env.TOL ?? 1e-8);
    console.log(`# 2-D Poisson: ∇²u = −ρ(x, y),  ρ = ${rhoExpr}`);
    console.log(`#   grid ${N}×${N}, [0,1]², u=0 on boundary, tol=${tol}`);
    const schemes = ['jacobi', 'gauss-seidel', 'sor'] as const;
    const omega = Number(process.env.OMEGA ?? 1.85);
    for (const scheme of schemes) {
      const r = solvePoisson2D({
        Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
        rhoExpr, scheme, omega, tol, maxIter: 50000,
      });
      let maxErr = 0;
      for (let j = 0; j < r.Ny; j++) for (let i = 0; i < r.Nx; i++) {
        const exact = Math.sin(Math.PI * r.xs[i]) * Math.sin(Math.PI * r.ys[j]);
        const got = r.u[j * r.Nx + i];
        if (Math.abs(exact - got) > maxErr) maxErr = Math.abs(exact - got);
      }
      console.log(`#   ${scheme.padEnd(13)}  iters=${String(r.iterations).padStart(6)}  finalΔ=${r.finalDelta.toExponential(2)}  maxErr vs sin·sin = ${maxErr.toExponential(2)}`);
    }
    if (process.env.ANIMATE === '1') {
      const r = solvePoisson2D({
        Nx: N, Ny: N, xLo: 0, xHi: 1, yLo: 0, yHi: 1,
        rhoExpr, scheme: 'sor', omega, tol, maxIter: 50000,
      });
      const {POISSON_W, POISSON_H, buildPoissonFrame} =
        await import('./animation/scenes/calculus-scene');
      const outDir = path.join(__dirname, '..', '..', 'out');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
      const rec = new FrameRecorder({
        framesPath: path.join(outDir, 'calculus-poisson2d.frames.jsonl'),
        htmlPath:   path.join(outDir, 'calculus-poisson2d.html'),
        width: POISSON_W, height: POISSON_H, fps: 1,
        title: `2-D Poisson  ∇²u = -ρ   ${N}×${N} grid (SOR)`,
        subtitle: `Each cell = one station; SOR converged in ${r.iterations} iters`,
      });
      let vMax = 0;
      for (let i = 0; i < r.u.length; i++) if (Math.abs(r.u[i]) > vMax) vMax = Math.abs(r.u[i]);
      rec.frame(0, 0, () => buildPoissonFrame(r.u, N, N, vMax));
      await rec.finish();
      console.log(`#   animation written to ${path.relative(process.cwd(), path.join(outDir, 'calculus-poisson2d.html'))}`);
    }
    return;
  }

  console.error(`Unknown PROBLEM='${problem}'. Try expr | ode | pde | poisson.`);
  process.exit(1);
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
