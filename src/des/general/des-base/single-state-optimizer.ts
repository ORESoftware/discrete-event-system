'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/single_state_optimizer.rs
// - Keep file-for-file. Channel constants become pub consts; initial/result
//   token classes and SingleStateResultSnapshot become token/data structs.
// - SingleStateSourceStation and SingleStateSinkStation become concrete
//   DESStation structs; SingleStateOptimizer becomes a trait plus shared
//   optimizer-state struct for current/best/history fields.
// - Proposal, cost, accept, clone, and stop hooks map to trait methods; pure
//   cost/proposal graph adapters should implement PureTransform/PureTransformEntity.
// - Convert duplicate seed, uninitialized optimizer, and non-finite cost throws
//   to Result.

// =============================================================================
// general/des-base/single-state-optimizer.ts — base class for SINGLE-WALKER
// iterative optimisation: simulated annealing, hill climbing, tabu search,
// gradient descent (in state-space), Newton, threshold accepting, …
//
// PROBLEM SHAPE (what this captures)
// ──────────────────────────────────
//   minimise f(s) over s ∈ S
//   by repeatedly proposing a candidate s' ∈ N(s) and conditionally
//   accepting it. The DIFFERENTIATOR among algorithms in this family is
//   the ACCEPTANCE rule:
//
//     - SA:           accept if Δ ≤ 0 OR rng() < exp(−Δ/T_iter)
//     - Hill climb:   accept iff Δ < 0
//     - Tabu:         accept best non-tabu candidate (uses memory)
//     - Threshold:    accept if Δ ≤ τ_iter
//
// TEMPLATE METHOD (final, do NOT override)
// ────────────────────────────────────────
//   runTimeStep():
//     if shouldStop()   → finalise
//     candidate = propose(current, rng)
//     candCost  = cost(candidate)
//     if accept(current, candidate, currentCost, candCost, iter) → adopt
//     update best, history, iter
//
// HOOKS (subclasses MUST implement)
// ─────────────────────────────────
//   initialState, cost, propose, accept, clone, shouldStop
//
// HOOKS (optional override)
// ────────────────────────
//   onAccept, onReject, onFinish
// =============================================================================

import {ChannelName, DESStation, Token} from './station';

export const SINGLE_STATE_INITIAL_CHANNEL: ChannelName = 'single-state-initial';
export const SINGLE_STATE_RESULT_CHANNEL: ChannelName = 'single-state-result';

export class SingleStateInitialToken<S> implements Token {
  constructor(readonly state: S) {}
}

export interface SingleStateResultSnapshot<S> {
  best: S;
  bestCost: number;
  current: S;
  currentCost: number;
  iteration: number;
  acceptedCount: number;
  improveCount: number;
}

export class SingleStateResultToken<S> implements Token {
  constructor(readonly snapshot: SingleStateResultSnapshot<S>) {}
}

export class SingleStateSourceStation<S> extends DESStation {
  static readonly CH_INITIAL_STATE = SINGLE_STATE_INITIAL_CHANNEL;
  private emitted = false;

  constructor(
    id: string,
    private readonly initialState: () => S,
    private readonly validateInitialState: (state: S) => void = () => {},
  ) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    const state = this.initialState();
    this.validateInitialState(state);
    this.emit(new SingleStateInitialToken(state), SingleStateSourceStation.CH_INITIAL_STATE);
    this.emitted = true;
  }
}

export class SingleStateSinkStation<S> extends DESStation {
  static readonly CH_RESULT = SINGLE_STATE_RESULT_CHANNEL;
  latest: SingleStateResultToken<S> | undefined;

  constructor(id: string) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(SingleStateSinkStation.CH_RESULT) > 0; }

  runTimeStep(): void {
    const tokens = this.drain<SingleStateResultToken<S>>(SingleStateSinkStation.CH_RESULT);
    if (tokens.length > 0) this.latest = tokens[tokens.length - 1];
  }
}

export abstract class SingleStateOptimizer<S> extends DESStation {
  static readonly CH_INITIAL_STATE = SINGLE_STATE_INITIAL_CHANNEL;
  static readonly CH_RESULT = SINGLE_STATE_RESULT_CHANNEL;

  /** Walker's current position. */
  protected current!: S;
  /** Cost at `current`. */
  protected currentCost!: number;
  /** Best position ever seen. */
  protected best!: S;
  /** Cost at `best` (lower is better). */
  protected bestCost!: number;
  /** Iteration counter (one increment per runTimeStep). */
  protected iteration = 0;
  /** Number of times accept() returned true. */
  protected acceptedCount = 0;
  /** Number of strict improvements (Δ<0) accepted. */
  protected improveCount = 0;
  /** True after the runner terminates this station's loop. */
  protected finished = false;
  protected initialized = false;
  private resultEmitted = false;

  /** Best-cost history, downsampled by `traceStride` (default 1). */
  readonly bestHistory: number[] = [];
  /** Current-cost history, downsampled. */
  readonly currentHistory: number[] = [];
  protected readonly traceStride: number;

  /** RNG handed to subclass hooks. */
  protected readonly rng: () => number;

  constructor(id: string, opts: {rng: () => number; traceStride?: number}) {
    super(id);
    this.rng = opts.rng;
    this.traceStride = Math.max(1, opts.traceStride ?? 1);
    // Note: subclasses MUST call this.bootstrap() at the end of their
    // own constructor — we can't call abstract methods here in the base.
  }

  /** Subclass calls this once after constructing to seed `current`/`best`. */
  protected bootstrap(): void {
    this.bootstrapFromState(this.initialState(this.rng));
  }

  /** Source-driven bootstrap used by runners that model initial conditions
   *  as a movable token emitted from an explicit source station. */
  protected bootstrapFromState(initialState: S): void {
    if (this.initialized) throw new Error(`${this.id}: initial state already supplied`);
    this.current = this.clone(initialState);
    this.currentCost = this.cost(this.current);
    if (!Number.isFinite(this.currentCost)) {
      throw new Error(`${this.id}: initial state cost must be finite; got ${this.currentCost}`);
    }
    this.best = this.clone(this.current);
    this.bestCost = this.currentCost;
    this.bestHistory.push(this.bestCost);
    this.currentHistory.push(this.currentCost);
    this.initialized = true;
    this.onBootstrap();
  }

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  /** Build the initial walker state. */
  protected abstract initialState(rng: () => number): S;
  /** Compute scalar cost of a state (lower = better). */
  protected abstract cost(state: S): number;
  /** Propose a neighbour s' ∈ N(s). MUST NOT mutate `s`. */
  protected abstract propose(state: S, rng: () => number): S;
  /** Decide whether to move from `current` to `candidate`. */
  protected abstract accept(
    current: S, candidate: S,
    currentCost: number, candidateCost: number,
    iter: number, rng: () => number,
  ): boolean;
  /** Deep copy a state (for stashing in `best`). */
  protected abstract clone(state: S): S;
  /** Return true to terminate the optimiser. */
  protected abstract shouldStop(iter: number): boolean;

  // ── HOOKS (optional override) ────────────────────────────────────────────

  protected onAccept(_candidate: S, _delta: number, _iter: number): void {}
  protected onReject(_candidate: S, _delta: number, _iter: number): void {}
  protected onBootstrap(): void {}
  protected onFinish(): void {}

  // ── TEMPLATE METHOD (DO NOT override in concrete subclasses) ─────────────

  /** Drives one iteration. Marked here as the template method — leaf
   *  classes should implement only the hooks above. */
  runTimeStep(): void {
    if (this.finished) return;
    if (!this.initialized) {
      const seeds = this.drain<SingleStateInitialToken<S>>(SingleStateOptimizer.CH_INITIAL_STATE);
      if (seeds.length === 0) return;
      if (seeds.length > 1) throw new Error(`${this.id}: expected exactly one initial-state token, got ${seeds.length}`);
      this.bootstrapFromState(seeds[0].state);
      return;
    }
    if (this.inboxSize(SingleStateOptimizer.CH_INITIAL_STATE) > 0) {
      throw new Error(`${this.id}: received an initial-state token after initialization`);
    }
    if (this.shouldStop(this.iteration)) {
      this.finished = true; this.onFinish(); this.emitResult(); return;
    }
    const candidate = this.propose(this.current, this.rng);
    const candCost = this.cost(candidate);
    const delta = candCost - this.currentCost;
    const ok = this.accept(this.current, candidate, this.currentCost, candCost, this.iteration, this.rng);
    if (ok) {
      this.current = candidate;
      this.currentCost = candCost;
      if (candCost < this.bestCost) {
        this.bestCost = candCost;
        this.best = this.clone(candidate);
      }
      this.acceptedCount++;
      if (delta < 0) this.improveCount++;
      this.onAccept(candidate, delta, this.iteration);
    } else {
      this.onReject(candidate, delta, this.iteration);
    }
    if (this.iteration % this.traceStride === 0) {
      this.bestHistory.push(this.bestCost);
      this.currentHistory.push(this.currentCost);
    }
    this.iteration++;
  }

  /** While this station is not finished, the runner should keep ticking. */
  override hasWork(): boolean {
    return this.inboxSize(SingleStateOptimizer.CH_INITIAL_STATE) > 0 ||
      (this.initialized && !this.finished);
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getBest(): S { this.assertInitializedForRead(); return this.best; }
  getBestCost(): number { this.assertInitializedForRead(); return this.bestCost; }
  getCurrent(): S { this.assertInitializedForRead(); return this.current; }
  getCurrentCost(): number { this.assertInitializedForRead(); return this.currentCost; }
  getIteration(): number { return this.iteration; }
  getAcceptedCount(): number { return this.acceptedCount; }
  getImproveCount(): number { return this.improveCount; }
  isFinished(): boolean { return this.finished; }
  isInitialized(): boolean { return this.initialized; }

  private emitResult(): void {
    if (this.resultEmitted) return;
    this.emit(new SingleStateResultToken({
      best: this.clone(this.best),
      bestCost: this.bestCost,
      current: this.clone(this.current),
      currentCost: this.currentCost,
      iteration: this.iteration,
      acceptedCount: this.acceptedCount,
      improveCount: this.improveCount,
    }), SingleStateOptimizer.CH_RESULT);
    this.resultEmitted = true;
  }

  private assertInitializedForRead(): void {
    if (!this.initialized) throw new Error(`${this.id}: optimizer has not received an initial state`);
  }
}
