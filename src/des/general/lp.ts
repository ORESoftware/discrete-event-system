'use strict';

// =============================================================================
// Linear-programming bridge for the DES framework.
//
// HISTORICAL NOTE on the framing here: this module was written when DES
// in this codebase was used purely as a SIMULATION engine, and treated
// LP as something best delegated to an external solver. That stance is
// correct for one-shot solves: simplex / interior-point methods exploit
// polyhedral geometry the simulation engine doesn't expose. But for
// REPEATED LP solves — warm-starting after modifications, Benders cuts,
// column generation, MILP branch-and-bound, MPC — the right substrate
// is exactly a long-lived station that holds the basis and accepts
// movables that mutate it. See `general/incremental-lp.ts` and
// `general/stochastic-lp.ts` for those realisations.
//
// What this module provides is the foundational LP layer (one-shot
// solves and the external-solver bridge) that the incremental and
// stochastic modules build on. Plus the older simulation-optimisation
// integration patterns:
//
//   1.  An LPProblem JSON-serialisable type that any consumer can build.
//   2.  A small in-process two-phase revised-simplex solver (educational
//       fallback, used when no external solver is available; intentionally
//       NOT designed for big problems).
//   3.  An external-solver dispatcher that shells out to scipy.optimize
//       .linprog (HiGHS by default, or the older simplex / interior-point
//       methods, all selectable). The external solver is the canonical
//       path for any non-trivial LP.
//
// Patterns built on top of this (see `general/des-lp-bridge.ts`):
//
//   * `solveLPThenSimulate`  — LP gives the deterministic plan, DES
//     simulates the plan against stochastic delays / breakdowns / finite
//     buffers, returning the realised metrics.
//   * `solveMDPAsLP`         — convert a finite-state MDP into its LP
//     formulation (the Bellman-optimality LP), solve via external
//     simplex or interior-point, extract V* and π*. Bit-equivalent to
//     the in-process value-iteration solver.
//   * `lpAssistedRollingHorizon` — DES runs k ticks, LP re-plans on the
//     observed state, simulate k more, repeat. Standard MPC-style loop.
//
// Solver selection is via env var:
//   LP_SOLVER=internal           use the in-process simplex
//   LP_SOLVER=scipy:highs        scipy.linprog method=highs (default)
//   LP_SOLVER=scipy:highs-ds     scipy.linprog method=highs-ds (dual simplex)
//   LP_SOLVER=scipy:highs-ipm    scipy.linprog method=highs-ipm (interior-point)
//   LP_SOLVER=scipy:simplex      scipy.linprog method=simplex (legacy)
//   LP_SOLVER=scipy:interior-point  scipy.linprog method=interior-point (legacy)
//
// The Python wrapper script lives at
//   external-references/lp/lp_solve.py
// and is invoked via `python3` (configurable with PYTHON env var).
// =============================================================================

import {spawnSync} from 'child_process';
import * as path from 'path';

// -----------------------------------------------------------------------------
// LP problem types.
// -----------------------------------------------------------------------------

/**
 * A linear program in canonical form:
 *
 *     max  c^T x
 *     s.t. A_ub  · x ≤ b_ub      (inequality constraints)
 *          A_eq  · x  =  b_eq    (equality constraints)
 *          lb ≤ x ≤ ub           (per-variable bounds; default [0, +∞))
 *
 * For minimisation, set `sense: 'min'`.
 */
export interface LPProblem {
  sense: 'max' | 'min';
  /** Objective coefficient vector, length n. */
  c: number[];
  /** Inequality LHS, shape m × n (each row a constraint). May be empty. */
  A_ub?: number[][];
  /** Inequality RHS, length m. */
  b_ub?: number[];
  /** Equality LHS, shape p × n. May be empty. */
  A_eq?: number[][];
  /** Equality RHS, length p. */
  b_eq?: number[];
  /** Lower bounds, length n. `null` ⇒ −∞. Default 0. */
  lb?: (number | null)[];
  /** Upper bounds, length n. `null` ⇒ +∞. Default +∞. */
  ub?: (number | null)[];
  /** Optional human-readable variable names, length n. */
  varNames?: string[];
  /** Optional human-readable constraint names. */
  conNames?: string[];
}

export type LPStatus =
  | 'optimal'
  | 'infeasible'
  | 'unbounded'
  | 'iter-limit'
  | 'numerical-error';

export interface LPSolution {
  status: LPStatus;
  /** Optimal x (length n). Empty if status ≠ 'optimal'. */
  x: number[];
  /** Objective value c^T x. NaN if status ≠ 'optimal'. */
  objective: number;
  /** Dual variables for A_ub rows (shadow prices), length m. May be empty if not provided. */
  dualUB?: number[];
  /** Dual variables for A_eq rows, length p. May be empty if not provided. */
  dualEQ?: number[];
  /** Reduced costs (for primal variable bounds), length n. May be empty. */
  reducedCosts?: number[];
  /** Iteration count if reported by the solver. */
  iters?: number;
  /** Solver name (e.g. 'internal', 'scipy:highs', 'scipy:simplex'). */
  solver: string;
  /** Wall-clock time in milliseconds. */
  elapsedMs: number;
  /** Free-form human-readable message. */
  message?: string;
}

// -----------------------------------------------------------------------------
// In-process two-phase revised simplex.
//
// This is intentionally simple: a textbook implementation that handles the
// canonical-form LPs we generate from MDPs and small factory problems. It
// uses Bland's rule to guarantee finite termination on small problems and
// is meant as (a) a fallback when no external solver is on PATH and (b)
// a baseline to validate the external solver against.
//
// For any problem larger than a few hundred variables / constraints, use
// scipy:highs or another industrial solver via the external bridge.
// -----------------------------------------------------------------------------

export interface InternalSimplexOptions {
  /** Maximum total simplex iterations across both phases. Default 5000. */
  maxIter?: number;
  /** Pivot tolerance. Default 1e-9. */
  tol?: number;
}

/** Solve via the in-process simplex. Always returns a fully-populated LPSolution. */
export function solveLPInternal(p: LPProblem, opts: InternalSimplexOptions = {}): LPSolution {
  const t0 = Date.now();
  const tol = opts.tol ?? 1e-9;
  const maxIter = opts.maxIter ?? 5000;
  const n = p.c.length;
  const A_ub = p.A_ub ?? [];
  const b_ub = p.b_ub ?? [];
  const A_eq = p.A_eq ?? [];
  const b_eq = p.b_eq ?? [];
  if (A_ub.length !== b_ub.length) throw new Error('A_ub / b_ub length mismatch');
  if (A_eq.length !== b_eq.length) throw new Error('A_eq / b_eq length mismatch');
  const lb = p.lb ?? new Array(n).fill(0);
  const ub = p.ub ?? new Array(n).fill(null);

  // ---- Standardise: make all variables non-negative by shifting and free-variable splitting ----
  // For each x_i with finite lower bound l_i, substitute x_i = y_i + l_i (so y_i ≥ 0).
  // For x_i with l_i = null (free), substitute x_i = y_i^+ − y_i^- with both ≥ 0.
  // Upper bounds are added as new ≤ inequality constraints.
  const shifts: number[] = new Array(n).fill(0);
  const freeNeg: number[] = []; // indices of original x's that need a "-" twin
  const yIndexOfPos: number[] = []; // y-index of the "+" part of each original x
  let yCount = 0;
  for (let i = 0; i < n; i++) {
    const l = lb[i];
    if (l === null) {
      yIndexOfPos.push(yCount++);
      freeNeg.push(yCount++);
      shifts[i] = 0;
    } else {
      yIndexOfPos.push(yCount++);
      freeNeg.push(-1);
      shifts[i] = l;
    }
  }

  // Convert objective from x-space to y-space: c^T x = c^T (y_pos − y_neg + shift)
  const ny = yCount;
  const cY: number[] = new Array(ny).fill(0);
  let constShift = 0;
  for (let i = 0; i < n; i++) {
    const sign = p.sense === 'max' ? 1 : -1;
    cY[yIndexOfPos[i]] += sign * p.c[i];
    if (freeNeg[i] >= 0) cY[freeNeg[i]] += -sign * p.c[i];
    constShift += sign * p.c[i] * shifts[i];
  }

  // Build the standard inequality system Ay ≤ b in y-space.
  const Ay: number[][] = [];
  const by: number[] = [];
  // 1. A_ub · x ≤ b_ub  ⇒  A_ub · (y_pos − y_neg + shift) ≤ b_ub
  for (let r = 0; r < A_ub.length; r++) {
    const row = new Array(ny).fill(0);
    let rhs = b_ub[r];
    for (let i = 0; i < n; i++) {
      row[yIndexOfPos[i]] += A_ub[r][i];
      if (freeNeg[i] >= 0) row[freeNeg[i]] += -A_ub[r][i];
      rhs -= A_ub[r][i] * shifts[i];
    }
    Ay.push(row);
    by.push(rhs);
  }
  // 2. A_eq · x = b_eq  ⇒ encode as two ≤ inequalities.
  for (let r = 0; r < A_eq.length; r++) {
    const row = new Array(ny).fill(0);
    let rhs = b_eq[r];
    for (let i = 0; i < n; i++) {
      row[yIndexOfPos[i]] += A_eq[r][i];
      if (freeNeg[i] >= 0) row[freeNeg[i]] += -A_eq[r][i];
      rhs -= A_eq[r][i] * shifts[i];
    }
    // Ay_row · y ≤ rhs
    Ay.push(row.slice());
    by.push(rhs);
    // -Ay_row · y ≤ -rhs
    Ay.push(row.map(v => -v));
    by.push(-rhs);
  }
  // 3. Upper bounds on x.
  for (let i = 0; i < n; i++) {
    if (ub[i] !== null) {
      const row = new Array(ny).fill(0);
      row[yIndexOfPos[i]] = 1;
      if (freeNeg[i] >= 0) row[freeNeg[i]] = -1;
      Ay.push(row);
      by.push((ub[i] as number) - shifts[i]);
    }
  }

  // ---- Solve  max cY^T y   s.t.  Ay · y ≤ by, y ≥ 0  via Big-M two-phase simplex ----
  const m = Ay.length;
  // Form initial tableau. Convert ≤ constraints to equalities by adding a slack variable per row.
  // If by[r] < 0, multiply that row by -1, which converts a ≤ to a ≥ ; we then add a surplus
  // and an artificial variable to provide an initial BFS.
  const Acopy = Ay.map(r => r.slice());
  const bcopy = by.slice();
  const slackCols: number[] = [];     // column index in tableau of the slack for each row
  const artificialCols: number[] = []; // column index of artificial (or -1 if none)
  // Tableau columns: y (ny) | slacks (m) | artificials (k) | RHS
  const totalCols = ny + m;            // +artificials below, dynamically
  const slackStart = ny;
  // Step 1: ensure b ≥ 0 row by row. For rows with b < 0, multiply by -1.
  for (let r = 0; r < m; r++) {
    if (bcopy[r] < 0) {
      for (let j = 0; j < ny; j++) Acopy[r][j] = -Acopy[r][j];
      bcopy[r] = -bcopy[r];
      // After flip, the original ≤ becomes ≥; we'll add -slack + artificial.
      slackCols.push(slackStart + r);
      artificialCols.push(-2); // marker: needs artificial
    } else {
      slackCols.push(slackStart + r);
      artificialCols.push(-1); // no artificial needed
    }
  }
  // Allocate artificials.
  let artCount = 0;
  for (let r = 0; r < m; r++) if (artificialCols[r] === -2) artificialCols[r] = totalCols + artCount++;
  const fullCols = totalCols + artCount;

  // Build tableau: m rows × (fullCols + 1)
  const T: number[][] = [];
  for (let r = 0; r < m; r++) {
    const row = new Array(fullCols + 1).fill(0);
    for (let j = 0; j < ny; j++) row[j] = Acopy[r][j];
    // slack: +1 if no flip on row, -1 if flipped (we added artificial in that case)
    if (artificialCols[r] === -1) row[slackStart + r] = 1;
    else                         row[slackStart + r] = -1;
    if (artificialCols[r] >= 0) row[artificialCols[r]] = 1;
    row[fullCols] = bcopy[r];
    T.push(row);
  }

  const basis: number[] = [];
  for (let r = 0; r < m; r++) basis.push(artificialCols[r] >= 0 ? artificialCols[r] : slackStart + r);

  // ---- Phase 1: minimise sum of artificials (via max −sum) ----
  const phase1Cost = new Array(fullCols).fill(0);
  for (let r = 0; r < m; r++) if (artificialCols[r] >= 0) phase1Cost[artificialCols[r]] = -1;

  let iters = 0;
  if (artCount > 0) {
    const phase1Result = simplexCore(T, basis, phase1Cost, tol, maxIter - iters);
    iters += phase1Result.iters;
    if (phase1Result.status !== 'optimal') {
      return {status: phase1Result.status, x: [], objective: NaN,
              solver: 'internal', elapsedMs: Date.now() - t0, iters,
              message: 'phase 1 ' + phase1Result.status};
    }
    // If any artificial remains positive, infeasible.
    let phase1Obj = 0;
    for (let r = 0; r < m; r++) {
      if (artificialCols[r] >= 0 && Math.abs(T[r][fullCols]) > 1e-7
          && basis[r] === artificialCols[r]) {
        phase1Obj -= T[r][fullCols];
      }
    }
    if (phase1Obj < -1e-7) {
      return {status: 'infeasible', x: [], objective: NaN,
              solver: 'internal', elapsedMs: Date.now() - t0, iters,
              message: 'phase 1 residual sum of artificials = ' + (-phase1Obj).toExponential(3)};
    }
    // Drive any artificials still in the basis (at value 0) out via degenerate pivots.
    for (let r = 0; r < m; r++) {
      if (basis[r] >= ny + m) {
        for (let j = 0; j < ny + m; j++) {
          if (Math.abs(T[r][j]) > tol) { pivot(T, basis, r, j, tol); break; }
        }
      }
    }
  }

  // ---- Phase 2: maximise cY^T y. Build phase-2 cost over full column set (artificials excluded). ----
  const phase2Cost = new Array(fullCols).fill(0);
  for (let j = 0; j < ny; j++) phase2Cost[j] = cY[j];
  // Forbid artificials by giving them −∞ so they never re-enter the basis.
  for (let j = ny + m; j < fullCols; j++) phase2Cost[j] = -1e15;
  const phase2Result = simplexCore(T, basis, phase2Cost, tol, maxIter - iters);
  iters += phase2Result.iters;
  if (phase2Result.status === 'unbounded') {
    return {status: 'unbounded', x: [], objective: p.sense === 'max' ? +Infinity : -Infinity,
            solver: 'internal', elapsedMs: Date.now() - t0, iters};
  }
  if (phase2Result.status !== 'optimal') {
    return {status: phase2Result.status, x: [], objective: NaN,
            solver: 'internal', elapsedMs: Date.now() - t0, iters};
  }

  // Extract y from the tableau; reconstruct x.
  const yVals = new Array(ny).fill(0);
  for (let r = 0; r < m; r++) if (basis[r] < ny) yVals[basis[r]] = T[r][fullCols];
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const yp = yVals[yIndexOfPos[i]];
    const yn = freeNeg[i] >= 0 ? yVals[freeNeg[i]] : 0;
    x[i] = yp - yn + shifts[i];
  }
  // Objective in original sense.
  let obj = 0;
  for (let i = 0; i < n; i++) obj += p.c[i] * x[i];

  return {status: 'optimal', x, objective: obj, solver: 'internal',
          elapsedMs: Date.now() - t0, iters,
          message: `internal simplex: phase1+phase2, ${iters} iters`};
}

// Pivoting machinery for the simplex tableau.
// Bland's rule for entering and leaving variables to guarantee termination.
type SimplexResult = {status: LPStatus, iters: number};

function simplexCore(T: number[][], basis: number[], cost: number[],
                     tol: number, maxIter: number): SimplexResult {
  const m = T.length;
  if (m === 0) return {status: 'optimal', iters: 0};
  const ncols = T[0].length - 1;
  let iters = 0;
  while (iters < maxIter) {
    iters++;
    // Reduced costs: c_j - c_B · (B^{-1} A)_j  =  cost[j] - Σ_r cost[basis[r]] · T[r][j]
    let entering = -1;
    let bestRC = tol;
    for (let j = 0; j < ncols; j++) {
      let rc = cost[j];
      for (let r = 0; r < m; r++) rc -= cost[basis[r]] * T[r][j];
      if (rc > bestRC) { bestRC = rc; entering = j; break; } // Bland: first improving (using > rather than ≥)
    }
    if (entering < 0) return {status: 'optimal', iters};
    // Min-ratio test.
    let leaving = -1;
    let bestRatio = +Infinity;
    for (let r = 0; r < m; r++) {
      if (T[r][entering] > tol) {
        const ratio = T[r][ncols] / T[r][entering];
        if (ratio < bestRatio - tol || (Math.abs(ratio - bestRatio) <= tol && (leaving < 0 || basis[r] < basis[leaving]))) {
          bestRatio = ratio;
          leaving = r;
        }
      }
    }
    if (leaving < 0) return {status: 'unbounded', iters};
    pivot(T, basis, leaving, entering, tol);
  }
  return {status: 'iter-limit', iters};
}

function pivot(T: number[][], basis: number[], pivotRow: number, pivotCol: number, _tol: number): void {
  const ncols = T[0].length;
  const pv = T[pivotRow][pivotCol];
  for (let j = 0; j < ncols; j++) T[pivotRow][j] /= pv;
  for (let r = 0; r < T.length; r++) {
    if (r === pivotRow) continue;
    const factor = T[r][pivotCol];
    if (factor === 0) continue;
    for (let j = 0; j < ncols; j++) T[r][j] -= factor * T[pivotRow][j];
  }
  basis[pivotRow] = pivotCol;
}

// -----------------------------------------------------------------------------
// External-solver dispatcher.
// -----------------------------------------------------------------------------

export interface ExternalSolverOptions {
  /** scipy linprog method: 'highs', 'highs-ds', 'highs-ipm', 'simplex', 'interior-point'. */
  method?: 'highs' | 'highs-ds' | 'highs-ipm' | 'simplex' | 'interior-point';
  /** Override the python executable. Defaults to PYTHON env var or 'python3'. */
  python?: string;
  /** Override the script path. Defaults to <repo>/external-references/lp/lp_solve.py. */
  script?: string;
  /** Maximum stdout/stderr buffer in bytes. Default 32 MB. */
  maxBuffer?: number;
}

/**
 * Solve via an external scipy.optimize.linprog process. Throws if scipy
 * (or python) is unavailable. Use `solveLP` (below) for graceful fallback
 * to the internal solver.
 */
export function solveLPExternal(p: LPProblem, opts: ExternalSolverOptions = {}): LPSolution {
  const t0 = Date.now();
  const method = opts.method ?? 'highs';
  const python = opts.python ?? process.env.PYTHON ?? 'python3';
  const script = opts.script ?? path.join(__dirname, '..', '..', '..',
                                          'external-references', 'lp', 'lp_solve.py');
  const maxBuffer = opts.maxBuffer ?? 32 * 1024 * 1024;
  const payload = JSON.stringify({lp: p, method});
  const res = spawnSync(python, [script, '--method', method], {
    input: payload, encoding: 'utf8', maxBuffer,
  });
  if (res.status !== 0) {
    return {status: 'numerical-error', x: [], objective: NaN,
            solver: `scipy:${method}`, elapsedMs: Date.now() - t0,
            message: `external solver exited with ${res.status}: ${res.stderr ?? '(no stderr)'}`};
  }
  let out: any;
  try { out = JSON.parse(res.stdout); }
  catch (e) {
    return {status: 'numerical-error', x: [], objective: NaN,
            solver: `scipy:${method}`, elapsedMs: Date.now() - t0,
            message: `failed to parse external solver stdout as JSON: ${(e as Error).message}`};
  }
  return {
    status: out.status as LPStatus,
    x: out.x ?? [],
    objective: typeof out.objective === 'number' ? out.objective : NaN,
    dualUB: out.dualUB,
    dualEQ: out.dualEQ,
    reducedCosts: out.reducedCosts,
    iters: out.iters,
    solver: `scipy:${method}`,
    elapsedMs: Date.now() - t0,
    message: out.message,
  };
}

/**
 * Solve an LP using the solver selected by env var `LP_SOLVER`, falling
 * back to the internal simplex if no external solver is available.
 *
 *   LP_SOLVER=internal           in-process two-phase simplex
 *   LP_SOLVER=scipy:highs        scipy linprog method=highs (DEFAULT)
 *   LP_SOLVER=scipy:highs-ipm    scipy interior-point HiGHS
 *   LP_SOLVER=scipy:highs-ds     scipy dual simplex HiGHS
 *   LP_SOLVER=scipy:simplex      legacy scipy simplex
 *   LP_SOLVER=scipy:interior-point  legacy scipy interior-point
 */
export function solveLP(p: LPProblem, opts: ExternalSolverOptions & InternalSimplexOptions = {}): LPSolution {
  const choice = (process.env.LP_SOLVER ?? 'scipy:highs').trim();
  if (choice === 'internal') return solveLPInternal(p, opts);
  if (choice.startsWith('scipy:')) {
    const method = choice.slice('scipy:'.length) as ExternalSolverOptions['method'];
    const ext = solveLPExternal(p, {...opts, method});
    if (ext.status !== 'numerical-error') return ext;
    // Fall back to internal if the external bridge failed (no scipy, no python, etc).
    const fallback = solveLPInternal(p, opts);
    fallback.message = (fallback.message ? fallback.message + ' | ' : '')
      + 'external solver unavailable, fell back to internal: ' + (ext.message ?? '');
    return fallback;
  }
  throw new Error(`unknown LP_SOLVER value: ${choice}`);
}

// -----------------------------------------------------------------------------
// Convenience pretty-printer.
// -----------------------------------------------------------------------------

export function lpToString(p: LPProblem): string {
  const lines: string[] = [];
  const names = p.varNames ?? p.c.map((_, i) => `x${i}`);
  const term = (a: number, name: string) => {
    if (a === 0) return '';
    const sign = a >= 0 ? ' + ' : ' − ';
    const mag = Math.abs(a);
    return `${sign}${mag === 1 ? '' : mag.toString()}${name}`;
  };
  const objLine = p.c.map((a, i) => term(a, names[i])).join('').replace(/^ \+ /, '');
  lines.push(`${p.sense}  ${objLine}`);
  if (p.A_ub && p.A_ub.length) {
    lines.push('s.t.');
    for (let r = 0; r < p.A_ub.length; r++) {
      const lhs = p.A_ub[r].map((a, i) => term(a, names[i])).join('').replace(/^ \+ /, '');
      lines.push(`     ${lhs} ≤ ${p.b_ub![r]}`);
    }
  }
  if (p.A_eq && p.A_eq.length) {
    for (let r = 0; r < p.A_eq.length; r++) {
      const lhs = p.A_eq[r].map((a, i) => term(a, names[i])).join('').replace(/^ \+ /, '');
      lines.push(`     ${lhs} = ${p.b_eq![r]}`);
    }
  }
  if (p.lb || p.ub) {
    const n = p.c.length;
    for (let i = 0; i < n; i++) {
      const l = p.lb ? p.lb[i] : 0;
      const u = p.ub ? p.ub[i] : null;
      if (l === 0 && u === null) continue;
      lines.push(`     ${l === null ? '−∞' : l} ≤ ${names[i]} ≤ ${u === null ? '+∞' : u}`);
    }
  }
  return lines.join('\n');
}
