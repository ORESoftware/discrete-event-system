'use strict';

// =============================================================================
// Multivariable optimization.
//
// Solvers:
//   gradientDescent(f, grad, x0, opts)           first-order with backtracking
//   newtonOptim(f, grad, hess, x0, opts)         second-order (Newton)
//   bfgs(f, grad, x0, opts)                      quasi-Newton (BFGS)
//
// All minimize f. For maximization, negate.
// =============================================================================

import {numericalGradient} from './expr';

export interface OptimOptions {
  tol?: number;            // gradient norm tolerance for stopping (default 1e-8)
  maxIter?: number;        // hard cap (default 500)
  initialStep?: number;    // initial step size (default 1)
  c1?: number;             // Armijo (default 1e-4)
  rho?: number;            // backtrack ratio (default 0.5)
}

export interface OptimResult {
  x: number[];
  fx: number;
  iterations: number;
  converged: boolean;
  finalGradNorm: number;
  history: Array<{iter: number; fx: number; gradNorm: number}>;
}

const norm2 = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
const dot   = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

/**
 * Gradient descent with Armijo backtracking line search.
 *   x_{k+1} = x_k − α_k · ∇f(x_k)
 * α is initialized to `initialStep` and halved until f decreases enough.
 * Convergence: linear for strongly-convex f.
 */
export function gradientDescent(
  f: (x: number[]) => number,
  grad: (x: number[]) => number[],
  x0: number[], opts: OptimOptions = {},
): OptimResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 500;
  const c1 = opts.c1 ?? 1e-4;
  const rho = opts.rho ?? 0.5;
  const initialStep = opts.initialStep ?? 1;

  const x = x0.slice();
  let fx = f(x);
  const history: OptimResult['history'] = [];
  for (let iter = 0; iter < maxIter; iter++) {
    const g = grad(x);
    const gn = norm2(g);
    history.push({iter, fx, gradNorm: gn});
    if (gn < tol) return {x, fx, iterations: iter, converged: true, finalGradNorm: gn, history};
    let alpha = initialStep;
    let xNew = x.map((v, i) => v - alpha * g[i]);
    let fNew = f(xNew);
    let bt = 0;
    while (fNew > fx - c1 * alpha * gn * gn && bt < 50) {
      alpha *= rho; bt++;
      xNew = x.map((v, i) => v - alpha * g[i]);
      fNew = f(xNew);
    }
    for (let i = 0; i < x.length; i++) x[i] = xNew[i];
    fx = fNew;
  }
  const g = grad(x); const gn = norm2(g);
  return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
}

/**
 * Newton's method for optimization with damping.
 *   x_{k+1} = x_k − α_k · H(x_k)^{-1} · ∇f(x_k)
 * H is the Hessian. We solve H · p = ∇f via a small linear-algebra
 * helper (LU on n×n matrices); damped via Armijo backtracking. Falls
 * back to gradient direction if H is singular.
 */
export function newtonOptim(
  f: (x: number[]) => number,
  grad: (x: number[]) => number[],
  hess: (x: number[]) => number[][],
  x0: number[], opts: OptimOptions = {},
): OptimResult {
  const tol = opts.tol ?? 1e-10;
  const maxIter = opts.maxIter ?? 100;
  const c1 = opts.c1 ?? 1e-4;
  const rho = opts.rho ?? 0.5;

  const x = x0.slice();
  let fx = f(x);
  const history: OptimResult['history'] = [];
  for (let iter = 0; iter < maxIter; iter++) {
    const g = grad(x);
    const gn = norm2(g);
    history.push({iter, fx, gradNorm: gn});
    if (gn < tol) return {x, fx, iterations: iter, converged: true, finalGradNorm: gn, history};
    const H = hess(x);
    let p: number[];
    try { p = solveLinear(H, g); }
    catch { p = g.slice(); }                   // fall back to gradient direction
    // Direction: −p (since p satisfies H p = g, the Newton step is −H^{-1}g = −p).
    let alpha = 1;
    let xNew = x.map((v, i) => v - alpha * p[i]);
    let fNew = f(xNew);
    const directional = dot(p, g);
    let bt = 0;
    while (fNew > fx - c1 * alpha * directional && bt < 50 && directional > 0) {
      alpha *= rho; bt++;
      xNew = x.map((v, i) => v - alpha * p[i]);
      fNew = f(xNew);
    }
    for (let i = 0; i < x.length; i++) x[i] = xNew[i];
    fx = fNew;
  }
  const g = grad(x); const gn = norm2(g);
  return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
}

/**
 * BFGS quasi-Newton. Maintains an approximation H^{-1} updated by the
 * standard rank-2 BFGS formula. Combined with Armijo line search.
 * Superlinear convergence on smooth problems without exact Hessian.
 */
export function bfgs(
  f: (x: number[]) => number,
  grad: (x: number[]) => number[],
  x0: number[], opts: OptimOptions = {},
): OptimResult {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 200;
  const c1 = opts.c1 ?? 1e-4;
  const rho = opts.rho ?? 0.5;

  const n = x0.length;
  const x = x0.slice();
  let fx = f(x);
  let g = grad(x);
  let H = identity(n);    // initial inverse Hessian
  const history: OptimResult['history'] = [];
  for (let iter = 0; iter < maxIter; iter++) {
    const gn = norm2(g);
    history.push({iter, fx, gradNorm: gn});
    if (gn < tol) return {x, fx, iterations: iter, converged: true, finalGradNorm: gn, history};
    const p = matVec(H, g).map(v => -v);  // search direction = −H g
    const directional = dot(g, p);
    let alpha = 1;
    let xNew = x.map((v, i) => v + alpha * p[i]);
    let fNew = f(xNew);
    let bt = 0;
    while (fNew > fx + c1 * alpha * directional && bt < 50) {
      alpha *= rho; bt++;
      xNew = x.map((v, i) => v + alpha * p[i]);
      fNew = f(xNew);
    }
    const s = xNew.map((v, i) => v - x[i]);
    const gNew = grad(xNew);
    const y = gNew.map((v, i) => v - g[i]);
    const sy = dot(s, y);
    if (sy > 1e-12) {
      const Hy = matVec(H, y);
      const yHy = dot(y, Hy);
      const rho2 = 1 / sy;
      // BFGS inverse-Hessian update.
      // H_{k+1} = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
      // Implemented by direct formula.
      const Hnew = identity(n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          Hnew[i][j] = H[i][j]
            - rho2 * (s[i] * Hy[j] + Hy[i] * s[j])
            + rho2 * rho2 * yHy * s[i] * s[j]
            + rho2 * s[i] * s[j];
        }
      }
      H = Hnew;
    }
    for (let i = 0; i < n; i++) x[i] = xNew[i];
    fx = fNew; g = gNew;
  }
  const gn = norm2(g);
  return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
}

// -----------------------------------------------------------------------------
// Linear-algebra helpers (n is small; allocation cost dominates anyway).
// -----------------------------------------------------------------------------

function identity(n: number): number[][] {
  const M: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n).fill(0);
    row[i] = 1;
    M.push(row);
  }
  return M;
}
function matVec(M: number[][], v: number[]): number[] {
  return M.map(row => dot(row, v));
}
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map(row => row.slice());
  const x = b.slice();
  // Gaussian elimination with partial pivoting.
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    if (Math.abs(M[pivot][i]) < 1e-15) throw new Error('singular matrix');
    if (pivot !== i) { [M[i], M[pivot]] = [M[pivot], M[i]]; [x[i], x[pivot]] = [x[pivot], x[i]]; }
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

/** Convenience: wrap a function in a numerical gradient if no analytical grad available. */
export function autoGradient(f: (x: number[]) => number, h = 1e-6): (x: number[]) => number[] {
  return (x: number[]) => numericalGradient(f, x, h);
}
