'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/optim.rs  (module des::general::optim)
// 1:1 file move. Solvers are PureTransform classes (see notes below):
//   class GradientDescent/NewtonOptim/Bfgs/AutoGradient -> struct + impl Transform
//   interface OptimOptions/OptimResult/FirstOrderProblem/SecondOrderProblem -> structs
//   @deprecated free-fn shims -> drop in Rust (call the struct directly).
//   Closures `(x: number[]) => number` -> `impl Fn(&[f64]) -> f64`.
// =============================================================================

// =============================================================================
// Multivariable optimization.
//
// Solvers (each a PureTransform whose CONFIG lives in the constructor and whose
// PROBLEM is the `transform` input):
//   GradientDescent   first-order with Armijo backtracking
//   NewtonOptim       second-order (damped Newton, LU solve of H·p = g)
//   Bfgs              quasi-Newton (BFGS inverse-Hessian update)
//
// All minimize f. For maximization, negate.
//
// MIGRATION NOTE
//   These were previously free functions `gradientDescent(...)`, etc. They are
//   now classes implementing `Transform<Problem, OptimResult>` so the unit of
//   behaviour maps to a Rust `struct + impl Transform`. Thin function wrappers
//   are kept at the bottom (marked @deprecated) only to avoid churn; new code
//   should instantiate the class. Shared vector/matrix math comes from
//   `shared/linalg` rather than file-local helpers.
// =============================================================================

import {numericalGradient} from './expr';
import {PureTransform} from '../shared/transform';
import {LinAlg, LinearSystem, Mat, Vec, VecOps} from '../shared/linalg';

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

/** Objective with a starting point. First-order solvers also need `grad`;
 *  Newton additionally needs `hess`. Bundling them as a single named input
 *  keeps the `Transform<I, O>` shape and avoids positional-argument drift. */
export interface FirstOrderProblem {
  f: (x: Vec) => number;
  grad: (x: Vec) => Vec;
  x0: Vec;
}

export interface SecondOrderProblem extends FirstOrderProblem {
  hess: (x: Vec) => Mat;
}

/**
 * Gradient descent with Armijo backtracking line search.
 *   x_{k+1} = x_k − α_k · ∇f(x_k)
 * α is initialized to `initialStep` and halved until f decreases enough.
 * Convergence: linear for strongly-convex f.
 */
export class GradientDescent extends PureTransform<FirstOrderProblem, OptimResult> {
  constructor(private readonly opts: OptimOptions = {}) {
    super();
  }

  transform(problem: FirstOrderProblem): OptimResult {
    const {f, grad, x0} = problem;
    const tol = this.opts.tol ?? 1e-8;
    const maxIter = this.opts.maxIter ?? 500;
    const c1 = this.opts.c1 ?? 1e-4;
    const rho = this.opts.rho ?? 0.5;
    const initialStep = this.opts.initialStep ?? 1;

    const x = x0.slice();
    let fx = f(x);
    const history: OptimResult['history'] = [];
    for (let iter = 0; iter < maxIter; iter++) {
      const g = grad(x);
      const gn = VecOps.norm2(g);
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
    const g = grad(x);
    const gn = VecOps.norm2(g);
    if (gn >= tol) {
      console.warn(`[optim.GradientDescent] hit maxIter=${maxIter} without converging; final |grad|=${gn} (tol=${tol}), fx=${fx}.`);
    }
    return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
  }
}

/**
 * Newton's method for optimization with damping.
 *   x_{k+1} = x_k − α_k · H(x_k)^{-1} · ∇f(x_k)
 * H is the Hessian. We solve H · p = ∇f via `LinearSystem` (Gaussian
 * elimination with partial pivoting), damped via Armijo backtracking. Falls
 * back to the gradient direction if H is singular.
 */
export class NewtonOptim extends PureTransform<SecondOrderProblem, OptimResult> {
  constructor(private readonly opts: OptimOptions = {}) {
    super();
  }

  transform(problem: SecondOrderProblem): OptimResult {
    const {f, grad, hess, x0} = problem;
    const tol = this.opts.tol ?? 1e-10;
    const maxIter = this.opts.maxIter ?? 100;
    const c1 = this.opts.c1 ?? 1e-4;
    const rho = this.opts.rho ?? 0.5;

    const x = x0.slice();
    let fx = f(x);
    const history: OptimResult['history'] = [];
    for (let iter = 0; iter < maxIter; iter++) {
      const g = grad(x);
      const gn = VecOps.norm2(g);
      history.push({iter, fx, gradNorm: gn});
      if (gn < tol) return {x, fx, iterations: iter, converged: true, finalGradNorm: gn, history};
      const H = hess(x);
      let p: Vec;
      try { p = new LinearSystem(H, g).solve(); }
      catch {
        console.warn(`[optim.NewtonOptim] Hessian singular at iter ${iter} (x=[${x.map(v => v.toFixed(3)).join(', ')}]); falling back to gradient direction.`);
        p = g.slice();
      }
      // Direction: −p (since p satisfies H p = g, the Newton step is −H^{-1}g = −p).
      let alpha = 1;
      let xNew = x.map((v, i) => v - alpha * p[i]);
      let fNew = f(xNew);
      const directional = VecOps.dot(p, g);
      let bt = 0;
      while (fNew > fx - c1 * alpha * directional && bt < 50 && directional > 0) {
        alpha *= rho; bt++;
        xNew = x.map((v, i) => v - alpha * p[i]);
        fNew = f(xNew);
      }
      for (let i = 0; i < x.length; i++) x[i] = xNew[i];
      fx = fNew;
    }
    const g = grad(x);
    const gn = VecOps.norm2(g);
    if (gn >= tol) {
      console.warn(`[optim.NewtonOptim] hit maxIter=${maxIter} without converging; final |grad|=${gn} (tol=${tol}), fx=${fx}.`);
    }
    return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
  }
}

/**
 * BFGS quasi-Newton. Maintains an approximation H^{-1} updated by the
 * standard rank-2 BFGS formula. Combined with Armijo line search.
 * Superlinear convergence on smooth problems without exact Hessian.
 */
export class Bfgs extends PureTransform<FirstOrderProblem, OptimResult> {
  constructor(private readonly opts: OptimOptions = {}) {
    super();
  }

  transform(problem: FirstOrderProblem): OptimResult {
    const {f, grad, x0} = problem;
    const tol = this.opts.tol ?? 1e-8;
    const maxIter = this.opts.maxIter ?? 200;
    const c1 = this.opts.c1 ?? 1e-4;
    const rho = this.opts.rho ?? 0.5;

    const n = x0.length;
    const x = x0.slice();
    let fx = f(x);
    let g = grad(x);
    let H = LinAlg.identity(n);    // initial inverse Hessian
    const history: OptimResult['history'] = [];
    for (let iter = 0; iter < maxIter; iter++) {
      const gn = VecOps.norm2(g);
      history.push({iter, fx, gradNorm: gn});
      if (gn < tol) return {x, fx, iterations: iter, converged: true, finalGradNorm: gn, history};
      const p = LinAlg.matVec(H, g).map(v => -v);  // search direction = −H g
      const directional = VecOps.dot(g, p);
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
      const sy = VecOps.dot(s, y);
      if (sy <= 1e-12) {
        console.debug(`[optim.Bfgs] curvature condition sᵀy=${sy} ≤ 1e-12 at iter ${iter}; skipping inverse-Hessian update to keep H positive-definite.`);
      }
      if (sy > 1e-12) {
        const Hy = LinAlg.matVec(H, y);
        const yHy = VecOps.dot(y, Hy);
        const rho2 = 1 / sy;
        // BFGS inverse-Hessian update.
        // H_{k+1} = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
        const Hnew = LinAlg.identity(n);
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
    const gn = VecOps.norm2(g);
    if (gn >= tol) {
      console.warn(`[optim.Bfgs] hit maxIter=${maxIter} without converging; final |grad|=${gn} (tol=${tol}), fx=${fx}.`);
    }
    return {x, fx, iterations: maxIter, converged: gn < tol, finalGradNorm: gn, history};
  }
}

/** Wrap a function in a numerical gradient when no analytical grad is available.
 *  A `PureTransform` from objective to gradient-evaluator. */
export class AutoGradient extends PureTransform<(x: Vec) => number, (x: Vec) => Vec> {
  constructor(private readonly h = 1e-6) {
    super();
  }

  transform(f: (x: Vec) => number): (x: Vec) => Vec {
    return (x: Vec) => numericalGradient(f, x, this.h);
  }
}

// -----------------------------------------------------------------------------
// Backward-compatible function shims (kept thin; prefer the classes above).
// -----------------------------------------------------------------------------

/** @deprecated Use `new GradientDescent(opts).transform({f, grad, x0})`. */
export function gradientDescent(
  f: (x: Vec) => number, grad: (x: Vec) => Vec, x0: Vec, opts: OptimOptions = {},
): OptimResult {
  return new GradientDescent(opts).transform({f, grad, x0});
}

/** @deprecated Use `new NewtonOptim(opts).transform({f, grad, hess, x0})`. */
export function newtonOptim(
  f: (x: Vec) => number, grad: (x: Vec) => Vec, hess: (x: Vec) => Mat, x0: Vec, opts: OptimOptions = {},
): OptimResult {
  return new NewtonOptim(opts).transform({f, grad, hess, x0});
}

/** @deprecated Use `new Bfgs(opts).transform({f, grad, x0})`. */
export function bfgs(
  f: (x: Vec) => number, grad: (x: Vec) => Vec, x0: Vec, opts: OptimOptions = {},
): OptimResult {
  return new Bfgs(opts).transform({f, grad, x0});
}

/** @deprecated Use `new AutoGradient(h).transform(f)`. */
export function autoGradient(f: (x: Vec) => number, h = 1e-6): (x: Vec) => Vec {
  return new AutoGradient(h).transform(f);
}
