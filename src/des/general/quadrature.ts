// RUST MIGRATION: Target module `src/des/general/quadrature.rs`.
// RUST MIGRATION: Convert `QuadResult` to a `serde` struct; represent integrands as generic `Fn(f64) -> f64` bounds or boxed callbacks.
// RUST MIGRATION: Keep quadrature routines as free numerical functions; create `PureTransform` wrappers only for DES graph-visible integration blocks.
// RUST MIGRATION: Convert the Gauss-Legendre node table to `const` arrays or a `match` on supported orders.
// RUST MIGRATION: Inject RNG for Monte Carlo routines and return `Result<QuadResult, QuadError>` for invalid intervals, sample counts, dimensions, or unsupported orders.
'use strict';

// =============================================================================
// Numerical integration (quadrature) — multiple methods.
//
// All functions take f: (x) → number, integration limits [a, b], and a
// method-specific accuracy parameter. Each returns {value, evaluations}
// so callers can compare cost across methods.
//
// Methods:
//   trapezoidal(f, a, b, n)            composite trapezoid, n subintervals
//   simpson(f, a, b, n)                composite Simpson 1/3, n even
//   adaptiveSimpson(f, a, b, tol)      recursive Simpson with error gauge
//   gaussLegendre(f, a, b, n)          n-point Gauss-Legendre (n ∈ {2,3,4,5,7,10})
//   monteCarlo(f, a, b, n, rng)        n random samples; "the simulation path"
//
// Errors (analytical): trapezoid O(h²), Simpson O(h⁴), Gauss-Legendre
// O((b−a)^{2n+1}/(2n+1)) for smooth integrands, Monte Carlo
// stochastic O(1/√n).
// =============================================================================

import {mulberry32} from './prng';

export interface QuadResult {
  value: number;
  evaluations: number;
  /** Optional uncertainty estimate (Monte Carlo only). */
  stderr?: number;
}

// -----------------------------------------------------------------------------
// Composite trapezoid.
//   ∫_a^b f(x) dx ≈ h · (f(a)/2 + f(a+h) + … + f(b−h) + f(b)/2)
// -----------------------------------------------------------------------------

export function trapezoidal(f: (x: number) => number, a: number, b: number, n: number): QuadResult {
  if (n < 1) throw new Error(`trapezoidal: n must be ≥ 1, got ${n}`);
  const h = (b - a) / n;
  let s = 0.5 * (f(a) + f(b));
  for (let i = 1; i < n; i++) s += f(a + i * h);
  return {value: s * h, evaluations: n + 1};
}

// -----------------------------------------------------------------------------
// Composite Simpson 1/3.
//   ∫_a^b f(x) dx ≈ (h/3) · (f₀ + 4·Σ f_odd + 2·Σ f_even + f_n)
//   n must be even.
// -----------------------------------------------------------------------------

export function simpson(f: (x: number) => number, a: number, b: number, n: number): QuadResult {
  if (n < 2 || n % 2 !== 0) throw new Error(`simpson: n must be even and ≥ 2, got ${n}`);
  const h = (b - a) / n;
  let s = f(a) + f(b);
  for (let i = 1; i < n; i++) {
    const x = a + i * h;
    s += (i % 2 === 0 ? 2 : 4) * f(x);
  }
  return {value: s * h / 3, evaluations: n + 1};
}

// -----------------------------------------------------------------------------
// Adaptive Simpson via recursive bisection. Splits interval where the
// estimated error exceeds tol; uses the standard 15·(S − S_left − S_right)
// error gauge.
// -----------------------------------------------------------------------------

export function adaptiveSimpson(
  f: (x: number) => number, a: number, b: number, tol: number = 1e-9, maxDepth = 40,
): QuadResult {
  let evals = 0;
  function S(a: number, fa: number, fb: number, fm: number, b: number): number {
    return (b - a) * (fa + 4 * fm + fb) / 6;
  }
  function recurse(a: number, fa: number, fb: number, fm: number, b: number,
                   whole: number, tol: number, depth: number): number {
    const m = (a + b) / 2;
    const lm = (a + m) / 2; const rm = (m + b) / 2;
    const flm = f(lm); const frm = f(rm); evals += 2;
    const left  = S(a, fa, fm, flm, m);
    const right = S(m, fm, fb, frm, b);
    const err = (left + right - whole) / 15;
    if (depth >= maxDepth && Math.abs(err) > tol) {
      console.warn(`[quadrature.adaptiveSimpson] max recursion depth ${maxDepth} reached on [${a}, ${b}] with error gauge ${Math.abs(err).toExponential(2)} > tol ${tol}; result may be inaccurate.`);
    }
    if (Math.abs(err) <= tol || depth >= maxDepth) return left + right + err;
    return recurse(a, fa, fm, flm, m, left,  tol / 2, depth + 1) +
           recurse(m, fm, fb, frm, b, right, tol / 2, depth + 1);
  }
  const m = (a + b) / 2;
  const fa = f(a); const fb = f(b); const fm = f(m); evals += 3;
  const whole = S(a, fa, fb, fm, b);
  const value = recurse(a, fa, fb, fm, b, whole, tol, 0);
  return {value, evaluations: evals};
}

// -----------------------------------------------------------------------------
// Gauss-Legendre quadrature. Nodes & weights for [−1, 1]; we transform
// to [a, b] via x = (b−a)/2·t + (b+a)/2,  dx = (b−a)/2 dt.
// -----------------------------------------------------------------------------

const GL_NODES: Record<number, {x: number[]; w: number[]}> = {
  2:  {x: [-0.5773502691896257, 0.5773502691896257],
       w: [1.0, 1.0]},
  3:  {x: [-0.7745966692414834, 0.0, 0.7745966692414834],
       w: [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]},
  4:  {x: [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
       w: [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]},
  5:  {x: [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
       w: [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]},
  7:  {x: [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
            0.4058451513773972,  0.7415311855993945,  0.9491079123427585],
       w: [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
           0.3818300505051189, 0.2797053914892766, 0.1294849661688697]},
  10: {x: [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
           -0.1488743389816312,  0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
            0.8650633666889845,  0.9739065285171717],
       w: [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
           0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
           0.1494513491505806, 0.0666713443086881]},
};

export function gaussLegendre(f: (x: number) => number, a: number, b: number, n: number = 5): QuadResult {
  const nodes = GL_NODES[n];
  if (!nodes) throw new Error(`gaussLegendre: only n ∈ {2,3,4,5,7,10} supported (got ${n})`);
  const half = (b - a) / 2;
  const mid = (a + b) / 2;
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += nodes.w[i] * f(half * nodes.x[i] + mid);
  }
  return {value: s * half, evaluations: n};
}

// -----------------------------------------------------------------------------
// Monte Carlo integration. Returns an unbiased estimate plus stderr.
// Uses a seeded PRNG by default for reproducibility.
// -----------------------------------------------------------------------------

export function monteCarlo(
  f: (x: number) => number, a: number, b: number, n: number = 10000,
  rng: () => number = mulberry32(1),
): QuadResult {
  let s = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const x = a + (b - a) * rng();
    const y = f(x);
    s += y; ss += y * y;
  }
  const mean = s / n;
  const variance = (ss / n) - mean * mean;
  const value = mean * (b - a);
  const stderr = Math.sqrt(Math.max(0, variance) / n) * (b - a);
  return {value, evaluations: n, stderr};
}

// -----------------------------------------------------------------------------
// Multidimensional Monte Carlo over a box.
// -----------------------------------------------------------------------------

export function monteCarloND(
  f: (x: number[]) => number, lo: number[], hi: number[], n: number = 10000,
  rng: () => number = mulberry32(1),
): QuadResult {
  const d = lo.length;
  let volume = 1;
  for (let k = 0; k < d; k++) volume *= (hi[k] - lo[k]);
  let s = 0, ss = 0;
  const x = new Array<number>(d);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < d; k++) x[k] = lo[k] + (hi[k] - lo[k]) * rng();
    const y = f(x);
    s += y; ss += y * y;
  }
  const mean = s / n;
  const variance = (ss / n) - mean * mean;
  const value = mean * volume;
  const stderr = Math.sqrt(Math.max(0, variance) / n) * volume;
  return {value, evaluations: n, stderr};
}
