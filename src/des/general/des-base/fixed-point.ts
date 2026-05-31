'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/fixed_point.rs
// - Keep file-for-file. FixedPointOptions becomes a config struct and
//   FixedPointIterationStation becomes a trait plus shared state struct.
// - Preserve residual/history fields as Vec<f64>; abstract iteration hooks map
//   to trait methods with default stopping/instrumentation impls.
// - Pure contraction/update operators can stay private/associated functions, or
//   become PureTransform/PureTransformEntity if lifted into a DES graph.
// - Convert invalid tolerance/iteration settings and convergence failures to
//   Result-returning constructors or methods.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/fixed_point.rs  (module des::general::des_base::fixed_point)
// 1:1 file move. Template-method base for fixed-point iteration (value/policy
// iteration, Jacobi/Gauss-Seidel, belief backups, Benders) over a state `S`.
//
// Declarations → Rust:
//   interface FixedPointOptions     -> struct (#[derive(Default)])
//   abstract class FixedPointIterationStation<S> -> trait FixedPointIterationStation<S>: DESStation
//
// Conversion notes (file-specific):
//   - TEMPLATE METHOD: `runTimeStep` is final; required hooks
//     initialState/applyOperator/delta -> required trait fns; shouldStop/
//     onIteration/onConverged/onMaxIter -> provided defaults.
//   - `convergenceReason: 'converged'|'maxiter'|'running'` -> enum ConvergenceReason.
//   - `current!: S` definite-assignment -> `Option<S>` / two-phase init.
//   - `applyOperator` must return a NEW state (no mutate) -> takes `&S` returns `S`
//     (`S: Clone` if reuse needed).
//   - No RNG/clock; `maxIter`/`maxHistoryLen` default Infinity -> `usize::MAX` or `Option`.
// =============================================================================

// =============================================================================
// general/des-base/fixed-point.ts — base class for FIXED-POINT iteration:
// value iteration (Bellman backups), policy iteration, Jacobi / Gauss-Seidel
// for linear systems, alpha-vector / belief-state backups in POMDPs, Benders
// cutting-plane convergence, equilibrium computation, …
//
// PROBLEM SHAPE
// ─────────────
//   We have a state x_k, an update operator T, and we iterate
//   x_{k+1} = T(x_k) until ‖x_{k+1} − x_k‖ < ε  OR k ≥ K_max.
//
//   The DIFFERENTIATOR among algorithms is which operator T is. Examples:
//
//     - Value iteration:    V_{k+1}(s) = max_a Σ p(s'|s,a) [r + γ V_k(s')]
//     - Policy evaluation:  V_{k+1}(s) = Σ p(s'|s,π(s)) [r + γ V_k(s')]
//     - Jacobi:             x_{k+1}_i = (b_i − Σ_{j≠i} A_ij x_k_j) / A_ii
//     - Belief update:      b_{k+1} = η · O · M · b_k
//     - Benders:            x_{k+1} adds one optimality cut + re-solves master
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep():
//     if shouldStop(k, lastDelta) → finalise
//     newState = applyOperator(currentState)
//     delta    = norm(newState − currentState)
//     currentState = newState
//     k += 1
//     record delta history
//
// HOOKS (abstract — subclasses MUST implement)
// ────────────────────────────────────────────
//   initialState() → x_0
//   applyOperator(x_k) → x_{k+1}
//   delta(x_k, x_{k+1}) → number       — convergence metric
//
// HOOKS (optional override)
// ─────────────────────────
//   shouldStop(iter, lastDelta) → default: lastDelta < tol or iter ≥ maxIter
//   onIteration / onConverged / onMaxIter
// =============================================================================

import {DESStation} from './station';

export interface FixedPointOptions {
  /** Convergence tolerance on `delta(x_k, x_{k+1})`. Default 1e-9. */
  tol?: number;
  /** Hard cap on iterations. Default 5000. */
  maxIter?: number;
  /** Optional cap on history length. Default Infinity. */
  maxHistoryLen?: number;
}

export abstract class FixedPointIterationStation<S> extends DESStation {
  /** Current iterate. Subclass populates via initialState() called from
   *  bootstrap(). */
  protected current!: S;
  protected iteration = 0;
  protected lastDelta = Infinity;
  /** True after shouldStop() returns true. */
  protected finished = false;
  protected convergenceReason: 'converged' | 'maxiter' | 'running' = 'running';

  /** Per-iteration delta history (always recorded). */
  readonly deltaHistory: number[] = [];

  protected readonly tol: number;
  protected readonly maxIter: number;
  protected readonly maxHistoryLen: number;

  constructor(id: string, opts: FixedPointOptions = {}) {
    super(id);
    this.tol = opts.tol ?? 1e-9;
    this.maxIter = opts.maxIter ?? 5000;
    this.maxHistoryLen = opts.maxHistoryLen ?? Infinity;
    // Subclasses MUST call bootstrap() at end of own ctor.
  }

  protected bootstrap(): void {
    this.current = this.initialState();
  }

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  /** Build x_0. */
  protected abstract initialState(): S;
  /** Apply the operator T(x_k). MUST return a NEW state — do not mutate `prev`. */
  protected abstract applyOperator(prev: S): S;
  /** Convergence metric — typically max-norm or L2-norm of (next − prev). */
  protected abstract delta(prev: S, next: S): number;

  // ── HOOKS (optional override) ────────────────────────────────────────────

  protected shouldStop(iter: number, lastDelta: number): boolean {
    if (iter >= this.maxIter) {
      this.convergenceReason = 'maxiter';
      return true;
    }
    if (iter > 0 && lastDelta < this.tol) {
      this.convergenceReason = 'converged';
      return true;
    }
    return false;
  }

  protected onIteration(_iter: number, _delta: number): void {}
  protected onConverged(_iter: number, _delta: number): void {}
  protected onMaxIter(_iter: number, _delta: number): void {}

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  runTimeStep(): void {
    if (this.finished) return;
    if (this.shouldStop(this.iteration, this.lastDelta)) {
      this.finished = true;
      if (this.convergenceReason === 'converged') this.onConverged(this.iteration, this.lastDelta);
      if (this.convergenceReason === 'maxiter')  this.onMaxIter(this.iteration, this.lastDelta);
      return;
    }
    const next = this.applyOperator(this.current);
    this.lastDelta = this.delta(this.current, next);
    this.current = next;
    this.iteration += 1;
    if (this.deltaHistory.length < this.maxHistoryLen) this.deltaHistory.push(this.lastDelta);
    this.onIteration(this.iteration, this.lastDelta);
  }

  override hasWork(): boolean { return !this.finished; }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getCurrent(): S { return this.current; }
  getIteration(): number { return this.iteration; }
  getLastDelta(): number { return this.lastDelta; }
  isFinished(): boolean { return this.finished; }
  getReason(): 'converged' | 'maxiter' | 'running' { return this.convergenceReason; }
}
