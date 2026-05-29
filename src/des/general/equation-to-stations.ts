'use strict';

// =============================================================================
// equation-to-stations: take an ODE/PDE specification (expression strings)
// and build a FieldSimulation network that solves it.
//
// Each builder returns a FieldSimulation already wired and ready to run.
// The caller picks an integration scheme by passing it as a string:
//
//   ODE:        'euler' | 'rk2' | 'rk4'
//   1-D PDE:    'ftcs'  | 'btcs' (heat); 'leapfrog' (wave); 'upwind' (advection)
//   2-D PDE:    'jacobi' | 'gauss-seidel' | 'sor' (Poisson, iterative relaxation)
//
// The integration recipe is encoded in each station's `updater`. The
// FieldSimulation engine just walks tick by tick; nothing scheme-specific
// lives there.
// =============================================================================

import {Expr, evaluate, parse, toFunction} from './expr';
import {Census, FieldSimulation, FieldStation, FieldUpdater} from './field-station';

// -----------------------------------------------------------------------------
// ODE system as a station network.
//
// User provides:
//   names: ['y1', 'y2', …]
//   rhs:   ['expression for y1\'', 'expression for y2\'', …]
//   y0:    [initial values]
//
// Variables in scope inside each RHS expression: 't' and every name.
//
// Scheme semantics:
//   'euler':  y_{n+1} = y_n + dt · f(t_n, y_n)
//   'rk2'  :  Heun's predictor-corrector
//   'rk4'  :  classical Runge–Kutta
//
// All schemes use the census snapshot as f's "y" argument: each station
// reads ONLY the snapshot, never another station's mid-tick value, so
// the multi-station evaluation is order-independent and produces
// bit-identical results to running the ODE on a single dense vector.
// -----------------------------------------------------------------------------

export type ODEScheme = 'euler' | 'rk2' | 'rk4';

export interface ODESystemSpec {
  names: string[];
  rhs: string[];           // expression strings, one per name; vars [t, names]
  y0: number[];
  scheme: ODEScheme;
  /** Optional: precomputed RHS expressions (skips parse). */
  rhsExprs?: Expr[];
}

export function buildODESystem(spec: ODESystemSpec): FieldSimulation {
  const n = spec.names.length;
  if (spec.rhs.length !== n || spec.y0.length !== n) {
    throw new Error(`buildODESystem: names/rhs/y0 lengths must match (got ${n}, ${spec.rhs.length}, ${spec.y0.length})`);
  }
  const exprs = spec.rhsExprs ?? spec.rhs.map(parse);
  // Compile each RHS to a JS function over (t, y_1, …, y_n).
  const args = ['t', ...spec.names];
  const fns = exprs.map(e => toFunction(e, args));
  const fAt = (t: number, y: Float64Array | number[]): number[] => {
    const vals = new Array<number>(n);
    for (let i = 0; i < n; i++) vals[i] = fns[i](t, ...Array.from(y));
    return vals;
  };

  const stations: FieldStation[] = [];
  for (let i = 0; i < n; i++) {
    const updater: FieldUpdater = (() => {
      const idx = i;
      switch (spec.scheme) {
        case 'euler': {
          return (_prev, cur, _self, dt, t) => {
            return cur[idx] + dt * fAt(t, cur)[idx];
          };
        }
        case 'rk2': {
          return (_prev, cur, _self, dt, t) => {
            const k1 = fAt(t, cur);
            const yMid: number[] = new Array(n);
            for (let j = 0; j < n; j++) yMid[j] = cur[j] + dt * k1[j];
            const k2 = fAt(t + dt, yMid);
            return cur[idx] + dt / 2 * (k1[idx] + k2[idx]);
          };
        }
        case 'rk4': {
          return (_prev, cur, _self, dt, t) => {
            const k1 = fAt(t, cur);
            const yk2: number[] = new Array(n);
            for (let j = 0; j < n; j++) yk2[j] = cur[j] + dt / 2 * k1[j];
            const k2 = fAt(t + dt / 2, yk2);
            const yk3: number[] = new Array(n);
            for (let j = 0; j < n; j++) yk3[j] = cur[j] + dt / 2 * k2[j];
            const k3 = fAt(t + dt / 2, yk3);
            const yk4: number[] = new Array(n);
            for (let j = 0; j < n; j++) yk4[j] = cur[j] + dt * k3[j];
            const k4 = fAt(t + dt, yk4);
            return cur[idx] + dt / 6 * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]);
          };
        }
      }
    })();
    const fs = new FieldStation(spec.names[i], spec.y0[i], updater, null as any);
    stations.push(fs);
  }
  return new FieldSimulation(stations);
}

// -----------------------------------------------------------------------------
// 1-D PDE on a uniform grid x_0 < x_1 < … < x_{N-1}.
// We support three families:
//
//   HEAT       u_t = α(x) · u_xx + s(t, x)
//     'ftcs'    explicit:  u_i^{n+1} = u_i^n + α·dt/dx² · (u_{i+1} − 2u_i + u_{i-1}) + dt·s
//     'btcs'    implicit:  (I − α·dt/dx²·L) u^{n+1} = u^n + dt·s     (tridiagonal solve)
//
//   WAVE       u_tt = c(x)² · u_xx
//     'leapfrog'  u_i^{n+1} = 2 u_i^n − u_i^{n-1} + (c·dt/dx)² · (u_{i+1} − 2u_i + u_{i-1})
//
//   ADVECTION  u_t + a(x) · u_x = 0
//     'upwind'    u_i^{n+1} = u_i^n − a·dt/dx · (u_i − u_{i-1})  for a > 0
//
// Boundaries are Dirichlet by default; pass `bcLeft`/`bcRight` constants.
// For Neumann (zero-flux), pass `'neumann'`.
// -----------------------------------------------------------------------------

export type Field1DScheme = 'ftcs' | 'btcs' | 'leapfrog' | 'upwind';
export type BC = number | 'neumann';

export interface Field1DSpec {
  /** Number of spatial cells. */
  N: number;
  /** Domain extent. */
  xLo: number;
  xHi: number;
  /** Initial condition u(x, t=0) as expression string in 'x'. */
  initExpr: string;
  /** Equation type. */
  family: 'heat' | 'wave' | 'advection';
  /** Required for 'heat': diffusion coefficient α(x) as expression in 'x' (or just a number). */
  alphaExpr?: string;
  /** Optional source term s(t, x) for heat (defaults to 0). */
  sourceExpr?: string;
  /** Required for 'wave': wave speed c(x) expression in 'x'. */
  cExpr?: string;
  /** Required for 'advection': velocity a(x) expression in 'x'. */
  aExpr?: string;
  /** Dirichlet boundary values (or 'neumann' for zero-flux). */
  bcLeft: BC;
  bcRight: BC;
  /** Discretisation scheme. */
  scheme: Field1DScheme;
}

export interface Field1DBuild {
  sim: FieldSimulation;
  xs: number[];
  dx: number;
}

export function buildField1D(spec: Field1DSpec): Field1DBuild {
  const N = spec.N;
  const dx = (spec.xHi - spec.xLo) / (N - 1);
  const xs = Array.from({length: N}, (_, i) => spec.xLo + i * dx);
  const initFn = toFunction(parse(spec.initExpr), ['x']);
  const u0 = xs.map(x => initFn(x));

  const alphaFn = spec.alphaExpr ? toFunction(parse(spec.alphaExpr), ['x']) : null;
  const sourceFn = spec.sourceExpr ? toFunction(parse(spec.sourceExpr), ['t', 'x']) : null;
  const cFn = spec.cExpr ? toFunction(parse(spec.cExpr), ['x']) : null;
  const aFn = spec.aExpr ? toFunction(parse(spec.aExpr), ['x']) : null;

  const stations: FieldStation[] = [];
  for (let i = 0; i < N; i++) {
    const xi = xs[i];
    const isLeft = i === 0;
    const isRight = i === N - 1;
    let updater: FieldUpdater;

    // Helper closures for boundary read.
    const readLeft = (cur: Float64Array): number => {
      if (i === 0) {
        if (spec.bcLeft === 'neumann') return cur[0];     // ghost-cell mirror
        return spec.bcLeft as number;
      }
      return cur[i - 1];
    };
    const readRight = (cur: Float64Array): number => {
      if (i === N - 1) {
        if (spec.bcRight === 'neumann') return cur[N - 1];
        return spec.bcRight as number;
      }
      return cur[i + 1];
    };

    if (spec.family === 'heat' && spec.scheme === 'ftcs') {
      const a = alphaFn ? alphaFn(xi) : 0;
      updater = (_prev, cur, self, dt, t) => {
        if ((isLeft && typeof spec.bcLeft === 'number') || (isRight && typeof spec.bcRight === 'number')) {
          // Dirichlet boundary: pin the value.
          return isLeft ? (spec.bcLeft as number) : (spec.bcRight as number);
        }
        const lap = (readRight(cur) - 2 * cur[self] + readLeft(cur)) / (dx * dx);
        const s = sourceFn ? sourceFn(t, xi) : 0;
        return cur[self] + dt * (a * lap + s);
      };
    } else if (spec.family === 'wave' && spec.scheme === 'leapfrog') {
      const c = cFn ? cFn(xi) : 1;
      updater = (prev, cur, self, dt, _t) => {
        if ((isLeft && typeof spec.bcLeft === 'number') || (isRight && typeof spec.bcRight === 'number')) {
          return isLeft ? (spec.bcLeft as number) : (spec.bcRight as number);
        }
        const lap = (readRight(cur) - 2 * cur[self] + readLeft(cur)) / (dx * dx);
        // Leapfrog: u^{n+1} = 2 u^n − u^{n-1} + (c·dt)² · u_xx
        return 2 * cur[self] - prev[self] + (c * c) * (dt * dt) * lap;
      };
    } else if (spec.family === 'advection' && spec.scheme === 'upwind') {
      const a = aFn ? aFn(xi) : 1;
      updater = (_prev, cur, self, dt, _t) => {
        if (isLeft && typeof spec.bcLeft === 'number') return spec.bcLeft as number;
        if (isRight && typeof spec.bcRight === 'number') return spec.bcRight as number;
        // First-order upwind. For a > 0: backward difference.
        if (a >= 0) {
          return cur[self] - a * dt / dx * (cur[self] - readLeft(cur));
        } else {
          return cur[self] - a * dt / dx * (readRight(cur) - cur[self]);
        }
      };
    } else if (spec.family === 'heat' && spec.scheme === 'btcs') {
      // BTCS / implicit: handled OUT-OF-LINE. We mark each station with
      // a no-op updater and run the implicit solve in a wrapper. This is
      // because the BTCS update couples ALL stations through a tridiagonal
      // system and cannot be expressed as N independent local updates.
      updater = (_prev, cur, self) => cur[self];
    } else {
      throw new Error(`Field1D: scheme '${spec.scheme}' not supported for family '${spec.family}'`);
    }
    const fs = new FieldStation(`x_${i}`, u0[i], updater, null as any);
    fs.position = xi;
    stations.push(fs);
  }
  const sim = new FieldSimulation(stations);

  // For BTCS we override the simulation's `run` method to do the
  // tridiagonal solve each tick (still via the Census/snap pattern, just
  // with one global solve replacing the per-station local updates).
  if (spec.family === 'heat' && spec.scheme === 'btcs') {
    const a = alphaFn ? alphaFn(0) : 1;   // homogeneous α only for BTCS demo
    sim.run = function (t0, t1, dt) {
      const N = this.fields.length;
      const r = a * dt / (dx * dx);
      const t: number[] = [t0];
      const values: Float64Array[] = [new Float64Array(this.census.snap)];
      let tn = t0;
      let tick = 0;
      // Tridiagonal A · u^{n+1} = u^n + dt·s; A = I − r·L (scaled Laplacian).
      // Coefficients per row i: a_sub = −r, a_diag = 1 + 2r, a_sup = −r.
      const aSub = -r, aDiag = 1 + 2 * r, aSup = -r;
      while (tn + 0.5 * dt < t1) {
        this.census.runTimeStep(dt, tn);
        // Build the RHS vector with boundary corrections.
        const rhs = new Float64Array(N);
        for (let i = 0; i < N; i++) {
          const s = sourceFn ? sourceFn(tn + dt, xs[i]) : 0;
          rhs[i] = this.census.snap[i] + dt * s;
        }
        // Apply Dirichlet boundary values.
        if (typeof spec.bcLeft === 'number') {
          rhs[0] = spec.bcLeft;
        } else {
          rhs[0] += -aSub * 0;   // Neumann ghost-cell modification (simplified)
        }
        if (typeof spec.bcRight === 'number') {
          rhs[N - 1] = spec.bcRight;
        }
        // Build per-row tridiagonal coefficients with BC overrides.
        const sub = new Float64Array(N);
        const dg = new Float64Array(N);
        const sup = new Float64Array(N);
        for (let i = 0; i < N; i++) { sub[i] = aSub; dg[i] = aDiag; sup[i] = aSup; }
        if (typeof spec.bcLeft === 'number') { sub[0] = 0; dg[0] = 1; sup[0] = 0; }
        if (typeof spec.bcRight === 'number') { sub[N - 1] = 0; dg[N - 1] = 1; sup[N - 1] = 0; }
        const u = thomas(sub, dg, sup, rhs);
        for (let i = 0; i < N; i++) this.fields[i].value = u[i];
        tn += dt;
        tick++;
        t.push(tn);
        values.push(new Float64Array(u));
      }
      return {
        trace: {t, values},
        finalValues: new Float64Array(this.fields.map(f => f.value)),
        ticks: tick,
      };
    };
  }
  return {sim, xs, dx};
}

/**
 * Thomas algorithm for tridiagonal systems. Solves A · x = d where A
 * has subdiagonal `a`, diagonal `b`, superdiagonal `c`. O(N).
 * a[0] and c[N-1] are unused.
 */
export function thomas(a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>, d: ArrayLike<number>): Float64Array {
  const n = d.length;
  const cp = new Float64Array(n);
  const dp = new Float64Array(n);
  cp[0] = c[0] / b[0];
  dp[0] = d[0] / b[0];
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1];
    cp[i] = c[i] / m;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
  }
  const x = new Float64Array(n);
  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
  return x;
}

// -----------------------------------------------------------------------------
// 2-D Poisson / Laplace on an Nx × Ny grid.  ∇²u = −ρ(x, y).
//
// Iterative relaxation schemes:
//   'jacobi'         u_{i,j}^{k+1} = (1/4) · (sum of 4 snap neighbours) + h²/4 · ρ
//   'gauss-seidel'   same but reads the LIVE u of already-updated neighbours
//                    (faster convergence; in our framework expressed by reading
//                    `census.snap` for not-yet-updated cells and live `value`
//                    for already-updated cells, which means we DON'T shuffle
//                    the order)
//   'sor'            ω-relaxed Gauss-Seidel: u ← (1−ω)·u + ω·gauss-seidel-update
//
// We treat each iteration as one "tick" with dt = 1; the trace records
// the residual norm history. Stop when max|Δu| < tol.
// -----------------------------------------------------------------------------

export type Field2DScheme = 'jacobi' | 'gauss-seidel' | 'sor';

export interface Poisson2DSpec {
  Nx: number; Ny: number;
  xLo: number; xHi: number;
  yLo: number; yHi: number;
  /** ρ(x, y) as expression in 'x' and 'y'. */
  rhoExpr: string;
  /** u(x, y, t=0) initial guess (defaults to zero). */
  initExpr?: string;
  /** Dirichlet boundary u_b(x, y) on all four edges, expression in 'x' and 'y'. Default 0. */
  bcExpr?: string;
  scheme: Field2DScheme;
  omega?: number;       // SOR relaxation parameter (default 1.5)
  maxIter?: number;     // default 5000
  tol?: number;         // default 1e-8
}

export interface Poisson2DResult {
  u: Float64Array;       // length Nx·Ny, row-major (j*Nx + i)
  iterations: number;
  finalDelta: number;
  residualHistory: number[];
  Nx: number; Ny: number;
  dx: number; dy: number;
  xs: number[]; ys: number[];
}

/** Solves the 2-D Poisson equation with a station network of Nx·Ny cells. */
export function solvePoisson2D(spec: Poisson2DSpec): Poisson2DResult {
  const {Nx, Ny} = spec;
  const dx = (spec.xHi - spec.xLo) / (Nx - 1);
  const dy = (spec.yHi - spec.yLo) / (Ny - 1);
  const xs = Array.from({length: Nx}, (_, i) => spec.xLo + i * dx);
  const ys = Array.from({length: Ny}, (_, j) => spec.yLo + j * dy);
  const rhoFn = toFunction(parse(spec.rhoExpr), ['x', 'y']);
  const bcFn = toFunction(parse(spec.bcExpr ?? '0'), ['x', 'y']);
  const initFn = toFunction(parse(spec.initExpr ?? '0'), ['x', 'y']);

  const idx = (i: number, j: number) => j * Nx + i;
  const u = new Float64Array(Nx * Ny);
  const uOld = new Float64Array(Nx * Ny);
  for (let j = 0; j < Ny; j++) {
    for (let i = 0; i < Nx; i++) {
      const onBoundary = (i === 0 || i === Nx - 1 || j === 0 || j === Ny - 1);
      u[idx(i, j)] = onBoundary ? bcFn(xs[i], ys[j]) : initFn(xs[i], ys[j]);
    }
  }

  const omega = spec.omega ?? 1.5;
  const maxIter = spec.maxIter ?? 5000;
  const tol = spec.tol ?? 1e-8;
  // Five-point stencil coefficient.
  const dx2 = dx * dx, dy2 = dy * dy;
  const denom = 2 * (dx2 + dy2);
  const residualHistory: number[] = [];
  let iter = 0;
  let finalDelta = Infinity;
  while (iter < maxIter) {
    uOld.set(u);
    let maxDelta = 0;
    for (let j = 1; j < Ny - 1; j++) {
      for (let i = 1; i < Nx - 1; i++) {
        const k = idx(i, j);
        const rho = rhoFn(xs[i], ys[j]);
        // The reference (Jacobi) uses uOld for ALL neighbours.
        // Gauss-Seidel and SOR use the live u for already-updated cells (i-1, j-1).
        const uE = (spec.scheme === 'jacobi') ? uOld[idx(i + 1, j)] : u[idx(i + 1, j)];
        const uW = (spec.scheme === 'jacobi') ? uOld[idx(i - 1, j)] : u[idx(i - 1, j)];
        const uN = (spec.scheme === 'jacobi') ? uOld[idx(i, j + 1)] : u[idx(i, j + 1)];
        const uS = (spec.scheme === 'jacobi') ? uOld[idx(i, j - 1)] : u[idx(i, j - 1)];
        const gs = (dy2 * (uE + uW) + dx2 * (uN + uS) + dx2 * dy2 * rho) / denom;
        const next = (spec.scheme === 'sor') ? (1 - omega) * u[k] + omega * gs : gs;
        const delta = Math.abs(next - u[k]);
        if (delta > maxDelta) maxDelta = delta;
        u[k] = next;
      }
    }
    iter++;
    finalDelta = maxDelta;
    residualHistory.push(maxDelta);
    if (maxDelta < tol) break;
  }
  return {u, iterations: iter, finalDelta, residualHistory,
          Nx, Ny, dx, dy, xs, ys};
}

// Re-export the simulation type so callers can inspect / extend.
export {FieldSimulation, FieldStation, Census} from './field-station';
