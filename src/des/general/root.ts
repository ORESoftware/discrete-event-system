'use strict';

// =============================================================================
// Root finding for f: R → R.
//
// Methods:
//   bisection(f, a, b, tol, maxIter)             requires sign change
//   newton(f, df, x0, tol, maxIter)              quadratic convergence; needs derivative
//   secant(f, x0, x1, tol, maxIter)              superlinear; no derivative needed
//
// All return {root, iterations, converged}.
// =============================================================================

export interface RootResult {
  root: number;
  iterations: number;
  converged: boolean;
  finalResidual: number;
}

/**
 * Bisection on [a, b] with sign change. Linear convergence; one bit per
 * iteration. Robust: always converges if f is continuous and f(a)·f(b) < 0.
 */
export function bisection(
  f: (x: number) => number,
  a: number, b: number,
  tol: number = 1e-12, maxIter: number = 200,
): RootResult {
  let fa = f(a); let fb = f(b);
  if (fa * fb > 0) throw new Error(`bisection: no sign change on [${a}, ${b}]: f(a)=${fa}, f(b)=${fb}`);
  let iter = 0;
  while (iter < maxIter) {
    const m = 0.5 * (a + b);
    const fm = f(m);
    if (Math.abs(fm) < tol || (b - a) / 2 < tol) {
      return {root: m, iterations: iter + 1, converged: true, finalResidual: Math.abs(fm)};
    }
    if (fa * fm < 0) { b = m; fb = fm; } else { a = m; fa = fm; }
    iter++;
  }
  const m = 0.5 * (a + b);
  return {root: m, iterations: iter, converged: false, finalResidual: Math.abs(f(m))};
}

/**
 * Newton's method using analytical derivative df. Quadratic convergence
 * near a simple root; can diverge if df(x_k) ≈ 0 or x_0 is far. Uses
 * Armijo-style damping if a step would increase |f|.
 */
export function newton(
  f: (x: number) => number,
  df: (x: number) => number,
  x0: number,
  tol: number = 1e-12, maxIter: number = 100,
): RootResult {
  let x = x0;
  let fx = f(x);
  for (let i = 0; i < maxIter; i++) {
    if (Math.abs(fx) < tol) {
      return {root: x, iterations: i, converged: true, finalResidual: Math.abs(fx)};
    }
    const dfx = df(x);
    if (dfx === 0 || !isFinite(dfx)) break;
    let step = fx / dfx;
    let alpha = 1;
    let xNext = x - alpha * step;
    let fNext = f(xNext);
    let damp = 0;
    while (Math.abs(fNext) > Math.abs(fx) && damp < 20) {
      alpha *= 0.5;
      xNext = x - alpha * step;
      fNext = f(xNext);
      damp++;
    }
    x = xNext; fx = fNext;
  }
  return {root: x, iterations: maxIter, converged: Math.abs(fx) < tol, finalResidual: Math.abs(fx)};
}

/**
 * Secant method: derivative-free Newton, using finite-difference slope
 * between the last two iterates. Superlinear (golden ratio) convergence.
 */
export function secant(
  f: (x: number) => number,
  x0: number, x1: number,
  tol: number = 1e-12, maxIter: number = 100,
): RootResult {
  let f0 = f(x0); let f1 = f(x1);
  for (let i = 0; i < maxIter; i++) {
    if (Math.abs(f1) < tol) {
      return {root: x1, iterations: i, converged: true, finalResidual: Math.abs(f1)};
    }
    if (f0 === f1) break;
    const x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0 = x1; f0 = f1;
    x1 = x2; f1 = f(x1);
  }
  return {root: x1, iterations: maxIter, converged: Math.abs(f1) < tol, finalResidual: Math.abs(f1)};
}
