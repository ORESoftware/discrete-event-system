'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/preconditions.rs
// - Keep file-for-file. PreconditionError should become a typed error enum or
//   struct implementing std::error::Error.
// - The Preconditions namespace maps cleanly to module functions or an empty
//   Preconditions struct with associated functions; prefer Result<(), Error>
//   return values over throwing.
// - Matrix/vector checks should accept slices such as &[f64] and &[Vec<f64>];
//   keep pure checks private/module-level unless exported through mod.rs.
// - If validation is represented as a DES graph node later, wrap the check in a
//   PureTransform/PureTransformEntity implementation.

// =============================================================================
// general/des-base/preconditions.ts — uniform PRE-RUN guards for every
// model's initial conditions and parameters.
//
// MOTIVATION
// ──────────
//   A discrete-event/iterative model is only well-defined if its
//   parameters land in the right corner of input space. Examples of
//   silently-wrong inputs that the engine couldn't catch before:
//
//     • stochastic-LP problem with a probability vector that sums to 0.99
//     • LQR with R = 0          → divide-by-zero in (R + B'PB)^{-1}
//     • DP with discount = 1.5  → unbounded value function
//     • MRAC with sign(b) wrong → unstable (no ε small enough saves you)
//     • SMC with η ≤ D          → reaching condition violated, no convergence
//     • dt = 0                  → infinite Euler steps
//
//   These should fail FAST and CLEARLY at construction, with a message
//   that names the offending parameter and shows the violated condition.
//
// API
// ───
//   `Preconditions` is a namespace of guard functions. Each guard either
//   returns silently or throws a `PreconditionError` whose message has
//   the form
//
//     <ModelName>: <param-name> must <condition>; got <value>
//
//   The runner protocol is:
//
//     • `DESStation.assertPreconditions()` — virtual hook called ONCE by
//       `runIterativeDES` before any tick. Default no-op. Subclasses
//       override to call the relevant `Preconditions.*` guards.
//
//     • `PlantBlock.assertPreconditions()` /
//       `ControllerBlock.assertPreconditions()` /
//       `EstimatorBlock.assertPreconditions()` — same hook on the
//       entity-style block hierarchy. Default no-op. Called once by
//       `runClosedLoop` before any tick.
//
//   Guards are also safe to call at construction time (typically the
//   place where the user's intent is closest to the call). The
//   convention in this codebase is:
//
//     • cheap, parameter-only checks → call from the constructor (fail
//       fast on obviously bad inputs)
//     • checks that require the FULL state graph → call from
//       `assertPreconditions()` (so wiring is in place)
//
//   That distinction is documented per-class.
// =============================================================================

// -----------------------------------------------------------------------------
// PRECONDITION ERROR
// -----------------------------------------------------------------------------

/** Thrown by `Preconditions.*` when a parameter check fails. */
export class PreconditionError extends Error {
  /** The model / class that performed the check. */
  readonly model: string;
  /** The parameter name (or expression) that failed. */
  readonly param: string;
  /** Human-readable description of the violated condition. */
  readonly condition: string;
  /** Offending value (if printable). */
  readonly observed?: unknown;
  constructor(model: string, param: string, condition: string, observed?: unknown) {
    const obsStr = observed === undefined ? '' :
                   typeof observed === 'number' ? `; got ${observed}` :
                   `; got ${JSON.stringify(observed)}`;
    super(`${model}: ${param} must ${condition}${obsStr}`);
    console.warn(`[precondition] ${model}: "${param}" must ${condition}${obsStr} — failing fast before the run starts.`);
    this.name = 'PreconditionError';
    this.model = model;
    this.param = param;
    this.condition = condition;
    this.observed = observed;
  }
}

// -----------------------------------------------------------------------------
// COMMON GUARDS
// -----------------------------------------------------------------------------

export namespace Preconditions {
  /** Throw if `x` is not a finite number (NaN / ±Infinity rejected). */
  export function finite(model: string, param: string, x: number): void {
    if (typeof x !== 'number' || !Number.isFinite(x)) {
      throw new PreconditionError(model, param, 'be a finite number', x);
    }
  }

  /** Throw if `x` is not strictly > 0 (so cannot serve as a denominator
   *  or step size). Calling guard for divide-by-zero scenarios. */
  export function positive(model: string, param: string, x: number): void {
    finite(model, param, x);
    if (x <= 0) throw new PreconditionError(model, param, 'be > 0 (positive, not zero)', x);
  }

  /** Throw if `x` is not >= 0. */
  export function nonNegative(model: string, param: string, x: number): void {
    finite(model, param, x);
    if (x < 0) throw new PreconditionError(model, param, 'be >= 0', x);
  }

  /** Throw if `x` < `lo` or `x` > `hi`. */
  export function inRange(model: string, param: string, x: number, lo: number, hi: number): void {
    finite(model, param, x);
    if (x < lo || x > hi) {
      throw new PreconditionError(model, param, `be in [${lo}, ${hi}]`, x);
    }
  }

  /** Throw unless `x` is an integer. */
  export function integer(model: string, param: string, x: number): void {
    finite(model, param, x);
    if (!Number.isInteger(x)) throw new PreconditionError(model, param, 'be an integer', x);
  }

  /** Throw if x is not an integer in [lo, hi]. */
  export function integerInRange(model: string, param: string, x: number, lo: number, hi: number): void {
    integer(model, param, x);
    if (x < lo || x > hi) throw new PreconditionError(model, param, `be an integer in [${lo}, ${hi}]`, x);
  }

  /** Throw if any element of `arr` is NaN / ±∞. */
  export function allFinite(model: string, param: string, arr: readonly number[]): void {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        throw new PreconditionError(model, param + `[${i}]`, 'be a finite number', arr[i]);
      }
    }
  }

  /** Throw unless `arr` has at least one element. */
  export function nonEmpty<T>(model: string, param: string, arr: readonly T[]): void {
    if (!arr || arr.length === 0) throw new PreconditionError(model, param, 'be non-empty', arr?.length);
  }

  /** Throw unless `arr.length === expected`. */
  export function lengthEq<T>(model: string, param: string, arr: readonly T[], expected: number): void {
    if (!arr || arr.length !== expected) {
      throw new PreconditionError(model, param + '.length', `equal ${expected}`, arr?.length);
    }
  }

  /** Throw unless every element is >= 0. */
  export function arrNonNegative(model: string, param: string, arr: readonly number[]): void {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i]) || arr[i] < 0) {
        throw new PreconditionError(model, `${param}[${i}]`, 'be >= 0', arr[i]);
      }
    }
  }

  /** Throw unless `arr` is a probability mass function: every entry in
   *  [0, 1] AND total mass within `tol` of 1. */
  export function probabilityVector(model: string, param: string, arr: readonly number[], tol = 1e-6): void {
    nonEmpty(model, param, arr);
    let s = 0;
    for (let i = 0; i < arr.length; i++) {
      const p = arr[i];
      if (!Number.isFinite(p) || p < 0 || p > 1 + tol) {
        throw new PreconditionError(model, `${param}[${i}]`, 'be in [0, 1]', p);
      }
      s += p;
    }
    if (Math.abs(s - 1) > tol) {
      throw new PreconditionError(model, param, `sum to 1 (within ${tol})`, s);
    }
  }

  /** Throw unless `M` is rectangular (each row same length) AND every
   *  cell is finite. */
  export function rectangularMatrix(model: string, param: string, M: ReadonlyArray<readonly number[]>): void {
    if (!M || M.length === 0) throw new PreconditionError(model, param, 'be a non-empty matrix', M);
    const cols = M[0].length;
    for (let i = 0; i < M.length; i++) {
      if (M[i].length !== cols) {
        throw new PreconditionError(model, `${param}[${i}].length`, `equal ${cols}`, M[i].length);
      }
      for (let j = 0; j < cols; j++) {
        if (!Number.isFinite(M[i][j])) {
          throw new PreconditionError(model, `${param}[${i}][${j}]`, 'be finite', M[i][j]);
        }
      }
    }
  }

  /** Throw unless M is square. */
  export function squareMatrix(model: string, param: string, M: ReadonlyArray<readonly number[]>): void {
    rectangularMatrix(model, param, M);
    if (M.length !== M[0].length) {
      throw new PreconditionError(model, param, `be a square matrix`, [M.length, M[0].length]);
    }
  }

  /** Symmetric square matrix to within tol. */
  export function symmetricMatrix(model: string, param: string, M: ReadonlyArray<readonly number[]>, tol = 1e-9): void {
    squareMatrix(model, param, M);
    const n = M.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(M[i][j] - M[j][i]) > tol) {
          throw new PreconditionError(model, param, `be symmetric (M[${i}][${j}] vs M[${j}][${i}])`,
                                       [M[i][j], M[j][i]]);
        }
      }
    }
  }

  /** Throw unless `M` is positive-semidefinite-LIKE: square, symmetric,
   *  and all diagonal entries >= 0 (necessary, not sufficient — but
   *  catches the obvious wrong sign / wrong diagonal user errors). For
   *  a stricter check use `positiveDefiniteCholesky`. */
  export function positiveSemidefiniteDiag(
    model: string, param: string, M: ReadonlyArray<readonly number[]>, tol = 1e-9): void {
    symmetricMatrix(model, param, M, tol);
    const n = M.length;
    for (let i = 0; i < n; i++) {
      if (M[i][i] < -tol) {
        throw new PreconditionError(model, `${param}[${i}][${i}]`, 'be >= 0 (PSD diagonal)', M[i][i]);
      }
    }
  }

  /** Throw unless `M` admits a Cholesky factorisation (i.e. M is PD).
   *  This is a cheap O(n³) test that catches non-PSD user error;
   *  callers usually need PD (not just PSD) for a covariance prior or
   *  cost weight. */
  export function positiveDefiniteCholesky(
    model: string, param: string, M: ReadonlyArray<readonly number[]>): void {
    squareMatrix(model, param, M);
    const n = M.length;
    const L: number[][] = Array.from({length: n}, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = M[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 1e-12) throw new PreconditionError(model, param, 'be positive-definite (Cholesky failed)', s);
          L[i][j] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
  }

  /** Throw if denom is "small enough to be effectively zero". */
  export function notDivByZero(model: string, param: string, denom: number, tol = 1e-12): void {
    finite(model, param, denom);
    if (Math.abs(denom) < tol) {
      throw new PreconditionError(model, param, `be non-zero (>${tol} in magnitude) — would divide by zero`, denom);
    }
  }

  /** Generic predicate guard: throw with `condition` text if predicate is false. */
  export function check(model: string, param: string, condition: string, ok: boolean,
                        observed?: unknown): void {
    if (!ok) throw new PreconditionError(model, param, condition, observed);
  }

  /** Same as `check` but for "x must equal y" style messages. */
  export function equal(model: string, param: string, x: unknown, expected: unknown): void {
    if (x !== expected) {
      throw new PreconditionError(model, param, `equal ${JSON.stringify(expected)}`, x);
    }
  }

  /** Throw unless `m`'s magnitude is at most `bound`. */
  export function magnitudeLeq(model: string, param: string, x: number, bound: number): void {
    finite(model, param, x);
    if (Math.abs(x) > bound) {
      throw new PreconditionError(model, param, `have magnitude <= ${bound}`, x);
    }
  }
}
