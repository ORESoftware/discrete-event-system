'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/ode.rs  (module des::general::ode)
// 1:1 file move. Fixed-step + adaptive ODE integrators for first-order systems.
//
// Declarations → Rust:
//   type RHS = (t, y[]) => y[]   -> trait bound `Fn(f64, &[f64]) -> Vec<f64>` (or `dyn`)
//   type Jac = (t, y[]) => y[][] -> trait bound `Fn(f64, &[f64]) -> Vec<Vec<f64>>`
//   interface ODETrace / RK45Options -> structs (ODETrace #[derive(Clone)], RK45Options Default)
//   fn euler/rk2Heun/rk4/rk45/backwardEuler/secondOrderToFirstOrder -> free fns or PureTransform
//                                                                       classes (per §3.1)
//   fn solveLinear (private)         -> private fn (or reuse shared/linalg)
//
// Conversion notes (file-specific):
//   - the solvers are currently FREE FUNCTIONS; the §3.1 target is PureTransform<Problem,ODETrace>
//     classes (config in ctor, the IVP as the single input). Mirror whichever optim.ts settled on.
//   - `f`/`J` are closures -> generic `F: Fn(..)`/`J: Fn(..)` params; watch borrow rules if they
//     capture mutable state (`FnMut`).
//   - state vectors `number[]` -> `Vec<f64>`; Jacobian `number[][]` -> `Vec<Vec<f64>>`.
//   - deterministic (no RNG/clock); backwardEuler's Newton inner-iter throws on singular -> Result/panic.
// =============================================================================
// ODE solvers for first-order systems  y'(t) = f(t, y),  y(t₀) = y₀.
//
// All solvers operate on vector-valued y of any dimension. Higher-order
// ODEs y'' = … are reduced to first-order by stacking [y, y'] in the
// state vector — `harmonicOscillator()` example below shows the recipe.
//
// Methods (cost / order in parentheses):
//   euler(f, y0, t0, t1, dt)              forward Euler        (1 fevals/step, O(dt))
//   rk2Heun(f, y0, t0, t1, dt)            improved Euler       (2 fevals/step, O(dt²))
//   rk4(f, y0, t0, t1, dt)                classical RK4        (4 fevals/step, O(dt⁴))
//   rk45(f, y0, t0, t1, opts)             Dormand-Prince adaptive (6 fevals/step, O(dt⁵))
//   backwardEuler(f, J, y0, t0, t1, dt)   implicit Euler       (Newton inner-iter, A-stable)
//
// Adaptive RK45 is the workhorse for non-stiff problems. Backward Euler
// is for stiff problems where explicit methods would need impossibly
// small dt (e.g. Robertson chemistry, fast-relaxing components).
// =============================================================================

import {PureTransform} from '../shared/transform';

export type RHS = (t: number, y: number[]) => number[];
export type Jac = (t: number, y: number[]) => number[][];   // ∂f/∂y, used by backwardEuler

export interface ODETrace {
  t: number[];
  y: number[][];      // y[i] is the state at time t[i]
}

/** An initial-value problem y'(t)=f(t,y), y(t₀)=y₀, integrated over [t₀, t₁].
 *  Bundles the positional (f, y0, t0, t1) arguments into one named input so the
 *  integrators keep the `Transform<I, O>` shape. */
export interface IVP {
  f: RHS;
  y0: number[];
  t0: number;
  t1: number;
}

/** A stiff IVP that additionally carries the Jacobian J=∂f/∂y (or `null` to fall
 *  back to fixed-point iteration), consumed by `BackwardEulerIntegrator`. */
export interface StiffIVP extends IVP {
  J: Jac | null;
}

const vplus  = (a: number[], b: number[], s = 1) => a.map((v, i) => v + s * b[i]);
const vscale = (a: number[], s: number) => a.map(v => v * s);
const vmax   = (a: number[]) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

// -----------------------------------------------------------------------------
// Forward Euler.  y_{n+1} = y_n + dt · f(t_n, y_n)
// -----------------------------------------------------------------------------

/** Forward Euler integrator. Fixed step `dt` is config; the IVP is the input. */
export class EulerIntegrator extends PureTransform<IVP, ODETrace> {
  constructor(private readonly dt: number) {
    super();
  }

  transform(problem: IVP): ODETrace {
    const {f, y0, t0, t1} = problem;
    const dt = this.dt;
    const t: number[] = [t0]; const y: number[][] = [y0.slice()];
    let tn = t0; let yn = y0.slice();
    while (tn + 0.5 * dt < t1) {
      const fn = f(tn, yn);
      yn = vplus(yn, fn, dt);
      tn += dt;
      t.push(tn); y.push(yn.slice());
    }
    return {t, y};
  }
}

/** @deprecated Use `new EulerIntegrator(dt).transform({f, y0, t0, t1})`. */
export function euler(f: RHS, y0: number[], t0: number, t1: number, dt: number): ODETrace {
  return new EulerIntegrator(dt).transform({f, y0, t0, t1});
}

// -----------------------------------------------------------------------------
// Heun's method (RK2 / improved Euler). Predictor + corrector.
//   k1 = f(t,y);  k2 = f(t+dt, y+dt·k1)
//   y_{n+1} = y_n + dt/2 · (k1 + k2)
// -----------------------------------------------------------------------------

/** Heun's method (RK2 / improved Euler). Fixed step `dt` is config. */
export class HeunIntegrator extends PureTransform<IVP, ODETrace> {
  constructor(private readonly dt: number) {
    super();
  }

  transform(problem: IVP): ODETrace {
    const {f, y0, t0, t1} = problem;
    const dt = this.dt;
    const t: number[] = [t0]; const y: number[][] = [y0.slice()];
    let tn = t0; let yn = y0.slice();
    while (tn + 0.5 * dt < t1) {
      const k1 = f(tn, yn);
      const k2 = f(tn + dt, vplus(yn, k1, dt));
      yn = vplus(yn, vplus(k1, k2), dt / 2);
      tn += dt;
      t.push(tn); y.push(yn.slice());
    }
    return {t, y};
  }
}

/** @deprecated Use `new HeunIntegrator(dt).transform({f, y0, t0, t1})`. */
export function rk2Heun(f: RHS, y0: number[], t0: number, t1: number, dt: number): ODETrace {
  return new HeunIntegrator(dt).transform({f, y0, t0, t1});
}

// -----------------------------------------------------------------------------
// Classical RK4. The textbook fourth-order Runge–Kutta scheme.
// -----------------------------------------------------------------------------

/** Classical fourth-order Runge–Kutta (RK4). Fixed step `dt` is config. */
export class RK4Integrator extends PureTransform<IVP, ODETrace> {
  constructor(private readonly dt: number) {
    super();
  }

  transform(problem: IVP): ODETrace {
    const {f, y0, t0, t1} = problem;
    const dt = this.dt;
    const t: number[] = [t0]; const y: number[][] = [y0.slice()];
    let tn = t0; let yn = y0.slice();
    while (tn + 0.5 * dt < t1) {
      const k1 = f(tn,           yn);
      const k2 = f(tn + dt / 2,  vplus(yn, k1, dt / 2));
      const k3 = f(tn + dt / 2,  vplus(yn, k2, dt / 2));
      const k4 = f(tn + dt,      vplus(yn, k3, dt));
      const incr = vplus(vplus(k1, k4), vplus(vscale(k2, 2), vscale(k3, 2)));
      yn = vplus(yn, incr, dt / 6);
      tn += dt;
      t.push(tn); y.push(yn.slice());
    }
    return {t, y};
  }
}

/** @deprecated Use `new RK4Integrator(dt).transform({f, y0, t0, t1})`. */
export function rk4(f: RHS, y0: number[], t0: number, t1: number, dt: number): ODETrace {
  return new RK4Integrator(dt).transform({f, y0, t0, t1});
}

// -----------------------------------------------------------------------------
// Dormand-Prince RK45 with adaptive step size.
// scipy.integrate.solve_ivp(method='RK45') uses the same Butcher tableau.
// -----------------------------------------------------------------------------

export interface RK45Options {
  rtol?: number;          // relative tolerance (default 1e-6)
  atol?: number;          // absolute tolerance (default 1e-9)
  hInit?: number;         // initial step size (default (t1-t0)/100)
  hMin?: number;          // minimum step size (default 1e-12)
  hMax?: number;          // maximum step size (default t1-t0)
  maxSteps?: number;      // hard cap (default 1_000_000)
}

const C2 = 1/5,    C3 = 3/10,  C4 = 4/5,  C5 = 8/9, C6 = 1, C7 = 1;
const A21 = 1/5;
const A31 = 3/40,        A32 = 9/40;
const A41 = 44/45,       A42 = -56/15,        A43 = 32/9;
const A51 = 19372/6561,  A52 = -25360/2187,   A53 = 64448/6561,  A54 = -212/729;
const A61 = 9017/3168,   A62 = -355/33,       A63 = 46732/5247,  A64 = 49/176, A65 = -5103/18656;
const A71 = 35/384,      A72 = 0,             A73 = 500/1113,    A74 = 125/192, A75 = -2187/6784, A76 = 11/84;
// 5th-order solution coefficients (= A7*).
const E1 = 71/57600, E3 = -71/16695, E4 = 71/1920, E5 = -17253/339200, E6 = 22/525, E7 = -1/40;
// E_i = b_i − b̂_i, used to estimate the error.

/** Dormand-Prince RK45 with adaptive step size. CONFIG (tolerances, step-size
 *  bounds, step cap) lives on the constructor; the IVP is the `transform` input. */
export class RK45Integrator extends PureTransform<IVP, ODETrace> {
  constructor(private readonly opts: RK45Options = {}) {
    super();
  }

  transform(problem: IVP): ODETrace {
    const {f, y0, t0, t1} = problem;
    const opts = this.opts;
    const rtol = opts.rtol ?? 1e-6;
    const atol = opts.atol ?? 1e-9;
    const hInit = opts.hInit ?? (t1 - t0) / 100;
    const hMin = opts.hMin ?? 1e-12;
    const hMax = opts.hMax ?? (t1 - t0);
    const maxSteps = opts.maxSteps ?? 1_000_000;

    const t: number[] = [t0]; const y: number[][] = [y0.slice()];
    let tn = t0; let yn = y0.slice(); let h = Math.min(hMax, Math.max(hMin, hInit));
    const n = y0.length;
    let step = 0;
    while (tn < t1 - 1e-15) {
      if (step++ > maxSteps) {
        console.warn(`[ode.rk45] exceeded maxSteps=${maxSteps} at t=${tn} (target t1=${t1}, current h=${h}); integration aborted.`);
        throw new Error(`rk45: exceeded ${maxSteps} steps`);
      }
      if (tn + h > t1) h = t1 - tn;
      const k1 = f(tn,             yn);
      const k2 = f(tn + C2 * h,    vplus(yn, k1, h * A21));
      const yk3in = yn.map((v, i) => v + h * (A31 * k1[i] + A32 * k2[i]));
      const k3 = f(tn + C3 * h,    yk3in);
      const yk4in = yn.map((v, i) => v + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]));
      const k4 = f(tn + C4 * h,    yk4in);
      const yk5in = yn.map((v, i) => v + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]));
      const k5 = f(tn + C5 * h,    yk5in);
      const yk6in = yn.map((v, i) => v + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]));
      const k6 = f(tn + C6 * h,    yk6in);
      // 5th-order solution at t+h.
      const y5 = yn.map((v, i) => v + h * (A71 * k1[i] + A72 * k2[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]));
      const k7 = f(tn + h, y5);
      // Error estimate (5th − 4th order).
      let errNorm = 0;
      for (let i = 0; i < n; i++) {
        const sci = atol + rtol * Math.max(Math.abs(yn[i]), Math.abs(y5[i]));
        const ei = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        errNorm += (ei / sci) * (ei / sci);
      }
      errNorm = Math.sqrt(errNorm / n);
      if (errNorm <= 1) {
        tn = tn + h;
        yn = y5;
        t.push(tn); y.push(yn.slice());
        // Step expansion factor (PI controller would be tighter; this is a simple I controller).
        const factor = errNorm === 0 ? 5 : Math.min(5, 0.9 * Math.pow(errNorm, -1 / 5));
        h = Math.min(hMax, Math.max(hMin, h * factor));
      } else {
        const factor = Math.max(0.1, 0.9 * Math.pow(errNorm, -1 / 5));
        h = Math.max(hMin, h * factor);
        if (h <= hMin) {
          console.warn(`[ode.rk45] step size underflow at t=${tn}: h=${h} ≤ hMin=${hMin} with errNorm=${errNorm}; problem may be stiff (try backwardEuler).`);
          throw new Error(`rk45: step underflow at t=${tn}`);
        }
      }
    }
    return {t, y};
  }
}

/** @deprecated Use `new RK45Integrator(opts).transform({f, y0, t0, t1})`. */
export function rk45(f: RHS, y0: number[], t0: number, t1: number, opts: RK45Options = {}): ODETrace {
  return new RK45Integrator(opts).transform({f, y0, t0, t1});
}

// -----------------------------------------------------------------------------
// Backward (implicit) Euler.  y_{n+1} = y_n + dt · f(t_{n+1}, y_{n+1}).
// Solves the implicit equation by Newton iteration using the Jacobian J
// (= ∂f/∂y). Use for stiff problems. Falls back to fixed-point if no J.
// -----------------------------------------------------------------------------

/**
 * Backward (implicit) Euler for stiff systems. CONFIG (fixed step `dt`, Newton
 * tolerance and iteration cap) lives on the constructor; the stiff IVP — which
 * carries the Jacobian `J` (or `null` for fixed-point fallback) — is the input.
 */
export class BackwardEulerIntegrator extends PureTransform<StiffIVP, ODETrace> {
  constructor(
    private readonly dt: number,
    private readonly newtonTol = 1e-10,
    private readonly newtonMaxIter = 50,
  ) {
    super();
  }

  transform(problem: StiffIVP): ODETrace {
    const {f, J, y0, t0, t1} = problem;
    const dt = this.dt;
    const newtonTol = this.newtonTol;
    const newtonMaxIter = this.newtonMaxIter;
    const t: number[] = [t0]; const y: number[][] = [y0.slice()];
    let tn = t0; let yn = y0.slice();
    while (tn + 0.5 * dt < t1) {
      const tNext = tn + dt;
      let yNext = yn.slice();
      let success = false;
      for (let iter = 0; iter < newtonMaxIter; iter++) {
        const fNext = f(tNext, yNext);
        const G = yNext.map((v, i) => v - yn[i] - dt * fNext[i]);
        const gNorm = vmax(G);
        if (gNorm < newtonTol) { success = true; break; }
        if (J) {
          const Jmat = J(tNext, yNext);
          const n = yNext.length;
          const M: number[][] = [];
          for (let i = 0; i < n; i++) {
            const row = new Array<number>(n);
            for (let j = 0; j < n; j++) row[j] = (i === j ? 1 : 0) - dt * Jmat[i][j];
            M.push(row);
          }
          const dy = solveLinear(M, G);
          for (let i = 0; i < n; i++) yNext[i] -= dy[i];
        } else {
          // Fixed-point: y^{k+1} = y_n + dt · f(t_{n+1}, y^k). Often diverges for stiff.
          for (let i = 0; i < yNext.length; i++) yNext[i] = yn[i] + dt * fNext[i];
        }
      }
      if (!success && J) {
        console.warn(`[ode.backwardEuler] Newton iteration failed to converge (tol=${newtonTol}, maxIter=${newtonMaxIter}) at t=${tn}; Jacobian may be wrong or step dt=${dt} too large.`);
        throw new Error(`backwardEuler: Newton failed at t=${tn}`);
      }
      yn = yNext; tn = tNext;
      t.push(tn); y.push(yn.slice());
    }
    return {t, y};
  }
}

/** @deprecated Use `new BackwardEulerIntegrator(dt, newtonTol, newtonMaxIter).transform({f, J, y0, t0, t1})`. */
export function backwardEuler(
  f: RHS, J: Jac | null, y0: number[], t0: number, t1: number, dt: number,
  newtonTol = 1e-10, newtonMaxIter = 50,
): ODETrace {
  return new BackwardEulerIntegrator(dt, newtonTol, newtonMaxIter).transform({f, J, y0, t0, t1});
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map(r => r.slice());
  const x = b.slice();
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
    if (Math.abs(M[p][i]) < 1e-15) {
      console.warn(`[ode.backwardEuler] singular Newton matrix (pivot ${M[p][i]} at column ${i}/${n}); cannot solve the implicit step.`);
      throw new Error('singular matrix in backwardEuler');
    }
    if (p !== i) { [M[i], M[p]] = [M[p], M[i]]; [x[i], x[p]] = [x[p], x[i]]; }
    for (let k = i + 1; k < n; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j < n; j++) M[k][j] -= f * M[i][j];
      x[k] -= f * x[i];
    }
  }
  const y = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * y[j];
    y[i] = s / M[i][i];
  }
  return y;
}

// -----------------------------------------------------------------------------
// Helper: build the stacked first-order system for a 2nd-order ODE
//   y'' + p(t)·y' + q(t)·y = r(t)
// by setting state = [y, y']. Caller supplies p, q, r as JS functions
// or expression-derived functions.
// -----------------------------------------------------------------------------

export function secondOrderToFirstOrder(
  p: (t: number) => number,
  q: (t: number) => number,
  r: (t: number) => number,
): RHS {
  return (t: number, y: number[]) => {
    // y[0] = y, y[1] = y'
    return [y[1], r(t) - p(t) * y[1] - q(t) * y[0]];
  };
}
