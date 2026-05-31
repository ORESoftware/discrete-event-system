'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/control_blocks.rs  (module des::general::des_base::control_blocks)
// 1:1 file move. Block-diagram control on the HEAVYWEIGHT StationaryEntity /
// AbstractMovingEntity framework (plant / controller / estimator + a closed-loop driver).
//
// Declarations → Rust:
//   class VectorSignal              -> struct VectorSignal: SignalValue (moving entity)
//   abstract class PlantBlock       -> trait PlantBlock: MultiDirectionalSignalEntity
//                                       (required dynamics; provided observe/runTimeStep/...)
//   abstract class ControllerBlock  -> trait ControllerBlock (required controlLaw; provided rest)
//   abstract class EstimatorBlock   -> trait EstimatorBlock (required update/getEstimate)
//   interface ClosedLoopOpts / ClosedLoopResult -> structs
//   fn runClosedLoop                -> fn run_closed_loop(...) -> ClosedLoopResult
//   fn ensureConnected (private)    -> fn ensure_connected(...)
//
// Conversion notes (file-specific):
//   - DIFFERENT BASE than the rest of des-base: these extend the abstract/ queueing
//     entity tree (StationaryEntity + AbstractMovingEntity), NOT DESStation. Port
//     those base traits (see abstract/*.rs headers) first.
//   - `mathjs.BigNumber` step size (`bgn(dt)`, `runTimeStep(stepSize: BigNumber)`) ->
//     pick ONE engine-wide numeric type (decimal crate or `f64`); `dt` is plain f64 here.
//   - `(this as any).id = id` writes a base field via a cast -> in Rust the id lives
//     on the embedded entity-core struct; set it normally, no cast.
//   - `item as unknown as VectorSignal` downcasts of queue items -> `dyn Any`
//     downcast or a typed signal enum; `sig.kind` string tag -> an enum.
//   - `connectionsOut`/`addOutConnection`/`addInConnection` graph edges ->
//     `Rc<RefCell<..>>` / arena (see abstract/interfaces.rs notes).
//   - `uMin/uMax: number[] | null` -> `Option<Vec<f64>>`.
//   - History `number[][]` -> `Vec<Vec<f64>>`; `.slice()` copies -> `.clone()`.
//   - `throw new Error` (dt<=0, shape) + Preconditions.* -> `Result`/`panic!`.
//   - non-ASCII `x̂`, `λ` in docs only; identifiers stay ASCII (`xhat`).
// =============================================================================

// =============================================================================
// general/des-base/control-blocks.ts — control-systems block diagrams that
// USE the queueing-style StationaryEntity + AbstractMovingEntity framework
// (the "heavyweight" branch of the codebase) instead of the lightweight
// DESStation pattern.
//
// MOTIVATION
// ──────────
//   The DESStation lightweight pattern (used by RL agents, optimisers, …)
//   is great for shuffled per-tick execution. But for CLASSICAL CONTROL
//   SYSTEMS the canonical mental model is the BLOCK DIAGRAM:
//
//       ┌──────────┐ y(t) ┌──────────┐ u(t) ┌──────────┐
//       │  PLANT   │─────▶│ CONTROL  │─────▶│  PLANT   │
//       │ (state)  │      │  LAW     │      │ (state)  │  …(closed loop)
//       └──────────┘      └──────────┘      └──────────┘
//
//   Blocks are STATIONARY ENTITIES; the SIGNALS that flow between them
//   (measurement y, control u, state estimate x̂, error e) are MOVING
//   ENTITIES carrying numeric vectors. This file provides the
//   tower of base classes for that view.
//
// FRAMEWORK CHOICES
// ─────────────────
//   We extend `MultiDirectionalSignalEntity` (the bidirectional signal
//   block from `signals/multi-directional-signal-entity.ts`) — it
//   ALREADY implements the relevant `HasManyInputConnections` /
//   `HasManyOutputConnections` plumbing and a per-block input queue.
//   Concrete blocks override `runTimeStep` to drain their inbox, run
//   the appropriate algorithm (dynamics step / control law / filter
//   update), and emit the next signal.
//
// PROVIDED BLOCKS
// ───────────────
//   • `VectorSignal` — moving entity carrying a `number[]`. Used as
//     measurement, control, state estimate, error.
//   • `PlantBlock<X>` — owns continuous state x ∈ ℝⁿ, advances via
//     a user-supplied `dynamics(x, u, dt) → x'` hook, emits y = h(x).
//   • `ControllerBlock<Y, U>` — receives y, computes u via `controlLaw`,
//     emits u.
//   • `EstimatorBlock<Y, X̂>` — receives y, emits x̂ via `update`.
//   • `runClosedLoop(plant, controller, opts)` — drives the loop for
//     `numSteps`, returns trajectory + control history.
//
// STEP CONVENTION
// ───────────────
//   We use a synchronous lock-step driver (not the signed BigNumber
//   queue's stepSize) — `dt` is plain `number` in seconds. The
//   block-diagram tick is: PLANT.runTimeStep → CONTROLLER.runTimeStep
//   → PLANT.runTimeStep … repeated `numSteps` times. The user-facing
//   API exposed here therefore reads as ordinary control-systems
//   pseudocode without `mathjs.BigNumber` boilerplate.
//
//   For users who want the FULL StationaryEntity stepping (BigNumber
//   step sizes, queue-driven timing) the same blocks ALSO work in that
//   mode because they inherit the standard `runTimeStep(stepSize, opts)`
//   signature — just call `plant.doTimeStep(bgn(0.1))` etc.
// =============================================================================

import * as math from 'mathjs';
import {bgn} from '../general';
import {MultiDirectionalSignalEntity} from '../../signals/multi-directional-signal-entity';
import {AbstractMovingEntity} from '../../entity-moving/moving';
import {SignalValue} from '../../signals/signal-value';
import {Preconditions} from './preconditions';

// -----------------------------------------------------------------------------
// VECTOR SIGNAL (moving entity)
// -----------------------------------------------------------------------------

/** Moving entity carrying an arbitrary numeric vector (`number[]`).
 *  This is what flows along block-diagram connections — measurements,
 *  control commands, state estimates, errors. */
export class VectorSignal extends SignalValue<unknown, number> {
  /** Numeric payload. */
  readonly vec: number[];
  /** Optional kind tag (e.g. "y", "u", "xhat", "e") — used only for
   *  diagnostics. */
  readonly kind: string;
  /** Discrete tick at which this signal was generated. */
  readonly tick: number;

  constructor(vec: readonly number[], kind: string, tick: number) {
    super({val: 0});
    this.vec = vec.slice();
    this.kind = kind;
    this.tick = tick;
  }

  override getValue(): number { return this.vec[0]; }
}

// -----------------------------------------------------------------------------
// PLANT BLOCK
// -----------------------------------------------------------------------------

/** A controllable plant. Owns continuous state x ∈ ℝⁿ. On every tick:
 *
 *    1. drain the latest control u from the inbox (or use 0)
 *    2. advance state x ← f(x, u, dt)
 *    3. emit measurement y = h(x) on every out connection
 *
 *  The default observation function is the identity — y = x. Subclasses
 *  override `observe` for partial / noisy observations. */
export abstract class PlantBlock extends MultiDirectionalSignalEntity<unknown, number> {
  /** Continuous state vector. */
  state: number[];
  /** Sample period dt (seconds). */
  readonly dt: number;
  /** Per-tick history of (x, u, y) for analysis / plotting. */
  readonly stateHistory: number[][] = [];
  readonly inputHistory: number[][] = [];
  readonly outputHistory: number[][] = [];
  /** Discrete tick counter. */
  protected tick = 0;
  /** Most recent control received. */
  protected lastU: number[];

  constructor(id: string, x0: readonly number[], dt: number, mDim: number) {
    super(id);
    (this as any).id = id;
    if (dt <= 0) throw new Error(`PlantBlock ${id}: dt must be positive`);
    this.state = x0.slice();
    this.dt = dt;
    this.lastU = new Array(mDim).fill(0);
    this.stateHistory.push(x0.slice());
  }

  // ── abstract hooks ──────────────────────────────────────────────────────

  /** Plant dynamics x' = f(x, u). Return the new state vector. The
   *  user is responsible for the discretisation choice (Euler / RK4 /
   *  exact). */
  protected abstract dynamics(x: readonly number[], u: readonly number[], dt: number): number[];

  /** Measurement equation y = h(x). Default identity. */
  protected observe(x: readonly number[]): number[] { return x.slice(); }

  /** Pre-run guard. Called ONCE by `runClosedLoop` before any tick.
   *  The default checks the universal invariants (dt > 0, x0 finite,
   *  state non-empty, mDim > 0). Subclasses override to add
   *  algorithm-specific checks but should call `super.assertPreconditions()`
   *  first to inherit the universal ones. */
  assertPreconditions(): void {
    Preconditions.positive('PlantBlock', 'dt', this.dt);
    Preconditions.nonEmpty('PlantBlock', 'state', this.state);
    Preconditions.allFinite('PlantBlock', 'state', this.state);
    Preconditions.allFinite('PlantBlock', 'lastU', this.lastU);
  }

  // ── StationaryEntity API ────────────────────────────────────────────────

  doValidationBeforeRun(): boolean { return true; }
  doValidation(): void {}
  acceptItem(_m: AbstractMovingEntity<any>): boolean { return true; }
  takeItem(m: AbstractMovingEntity<any>): void { this.queue.enqueue(m); }
  runFinish(): void {}
  getValue(): number { return this.state[0]; }

  /** Drain incoming controls, advance dynamics, emit measurement. */
  runTimeStep(_stepSize?: math.BigNumber): void {
    // 1. Drain ALL incoming controls — keep the most recent (last write wins).
    for (const [item] of this.queue.dequeueIterator()) {
      const sig = item as unknown as VectorSignal;
      if (sig.kind === 'u' || sig.kind === 'control') this.lastU = sig.vec.slice();
    }
    // 2. Advance state.
    const xNew = this.dynamics(this.state, this.lastU, this.dt);
    this.inputHistory.push(this.lastU.slice());
    this.state = xNew;
    this.stateHistory.push(xNew.slice());
    this.tick += 1;
    // 3. Emit measurement.
    const y = this.observe(this.state);
    this.outputHistory.push(y.slice());
    const yMsg = new VectorSignal(y, 'y', this.tick);
    for (const conn of this.connectionsOut) {
      if (conn.target.acceptItem(yMsg)) conn.target.takeItem(yMsg);
    }
  }
}

// -----------------------------------------------------------------------------
// CONTROLLER BLOCK
// -----------------------------------------------------------------------------

/** A feedback controller. Drains incoming measurements y, computes u,
 *  emits u. Subclasses override `controlLaw`. */
export abstract class ControllerBlock extends MultiDirectionalSignalEntity<unknown, number> {
  /** Number of control inputs m. */
  readonly mDim: number;
  /** Tick counter. */
  protected tick = 0;
  /** Per-tick history of (y, u). */
  readonly inputHistory: number[][] = [];
  readonly outputHistory: number[][] = [];
  /** Optional saturation bounds. */
  protected uMin: number[] | null = null;
  protected uMax: number[] | null = null;
  /** Last received measurement (kept so `runTimeStep` can run on
   *  empty inboxes — useful when sample rate < tick rate). */
  protected lastY: number[] | null = null;

  constructor(id: string, mDim: number) {
    super(id);
    (this as any).id = id;
    this.mDim = mDim;
  }

  protected abstract controlLaw(y: readonly number[], tick: number, t: number): number[];

  protected setSaturation(uMin: number[] | null, uMax: number[] | null): void {
    this.uMin = uMin; this.uMax = uMax;
  }

  /** Pre-run guard. Default checks `mDim > 0` and that any saturation
   *  bounds are coherent. Subclasses override to add algorithm-specific
   *  checks (gains positive, λ > 0, etc.) — call `super.assertPreconditions()`
   *  first to inherit the universal checks. */
  assertPreconditions(): void {
    Preconditions.integer('ControllerBlock', 'mDim', this.mDim);
    Preconditions.check('ControllerBlock', 'mDim', 'be >= 1', this.mDim >= 1, this.mDim);
    if (this.uMin && this.uMax) {
      Preconditions.lengthEq('ControllerBlock', 'uMin', this.uMin, this.mDim);
      Preconditions.lengthEq('ControllerBlock', 'uMax', this.uMax, this.mDim);
      for (let i = 0; i < this.mDim; i++) {
        Preconditions.check('ControllerBlock', `uMin[${i}] <= uMax[${i}]`, 'satisfy uMin <= uMax',
                            this.uMin[i] <= this.uMax[i], [this.uMin[i], this.uMax[i]]);
      }
    }
  }

  doValidationBeforeRun(): boolean { return true; }
  doValidation(): void {}
  acceptItem(_m: AbstractMovingEntity<any>): boolean { return true; }
  takeItem(m: AbstractMovingEntity<any>): void { this.queue.enqueue(m); }
  runFinish(): void {}
  getValue(): number {
    return this.outputHistory.length > 0 ? this.outputHistory[this.outputHistory.length - 1][0] : 0;
  }

  runTimeStep(_stepSize?: math.BigNumber): void {
    let y: number[] | null = this.lastY ? this.lastY.slice() : null;
    for (const [item] of this.queue.dequeueIterator()) {
      const sig = item as unknown as VectorSignal;
      if (sig.kind === 'y' || sig.kind === 'meas') y = sig.vec.slice();
    }
    if (!y) return;
    this.lastY = y;
    this.tick += 1;
    let u = this.controlLaw(y, this.tick, this.tick * this.getDt());
    if (this.uMin || this.uMax) u = this.saturate(u);
    this.inputHistory.push(y.slice());
    this.outputHistory.push(u.slice());
    const uMsg = new VectorSignal(u, 'u', this.tick);
    for (const conn of this.connectionsOut) {
      if (conn.target.acceptItem(uMsg)) conn.target.takeItem(uMsg);
    }
  }

  /** Override if your controller needs to know dt for time-derivative
   *  terms (most do). Default: return 1 (assume "tick = 1 unit"). */
  protected getDt(): number { return 1; }

  protected saturate(u: number[]): number[] {
    const out = u.slice();
    if (this.uMin) for (let i = 0; i < out.length; i++) if (out[i] < this.uMin[i]) out[i] = this.uMin[i];
    if (this.uMax) for (let i = 0; i < out.length; i++) if (out[i] > this.uMax[i]) out[i] = this.uMax[i];
    return out;
  }
}

// -----------------------------------------------------------------------------
// ESTIMATOR BLOCK (e.g. Kalman filter, observer)
// -----------------------------------------------------------------------------

/** A state estimator. Drains incoming measurements y AND the latest
 *  control u (so it can run a process model), emits state estimate x̂.
 *  Subclasses override `update`. */
export abstract class EstimatorBlock extends MultiDirectionalSignalEntity<unknown, number> {
  protected tick = 0;
  /** History of state estimates. */
  readonly estimateHistory: number[][] = [];
  /** History of measurements actually consumed. */
  readonly measurementHistory: number[][] = [];

  protected lastU: number[] | null = null;

  constructor(id: string) {
    super(id);
    (this as any).id = id;
  }

  /** One filter step: take (y, u) → emit new state estimate x̂. */
  protected abstract update(y: readonly number[], u: readonly number[] | null): number[];

  /** Public access to the current estimate. */
  abstract getEstimate(): number[];

  /** Pre-run guard. Default checks the current estimate is finite.
   *  Subclasses override to assert filter-specific conditions (covariance
   *  PSD, observation matrix dimensions, …). */
  assertPreconditions(): void {
    const e = this.getEstimate();
    Preconditions.allFinite('EstimatorBlock', 'estimate', e);
  }

  doValidationBeforeRun(): boolean { return true; }
  doValidation(): void {}
  acceptItem(_m: AbstractMovingEntity<any>): boolean { return true; }
  takeItem(m: AbstractMovingEntity<any>): void { this.queue.enqueue(m); }
  runFinish(): void {}
  getValue(): number {
    const e = this.getEstimate(); return e.length > 0 ? e[0] : 0;
  }

  runTimeStep(_stepSize?: math.BigNumber): void {
    let y: number[] | null = null;
    for (const [item] of this.queue.dequeueIterator()) {
      const sig = item as unknown as VectorSignal;
      if (sig.kind === 'y' || sig.kind === 'meas') y = sig.vec.slice();
      else if (sig.kind === 'u' || sig.kind === 'control') this.lastU = sig.vec.slice();
    }
    if (!y) return;
    this.tick += 1;
    this.measurementHistory.push(y.slice());
    const xhat = this.update(y, this.lastU);
    this.estimateHistory.push(xhat.slice());
    const msg = new VectorSignal(xhat, 'xhat', this.tick);
    for (const conn of this.connectionsOut) {
      if (conn.target.acceptItem(msg)) conn.target.takeItem(msg);
    }
  }
}

// -----------------------------------------------------------------------------
// CLOSED-LOOP DRIVER
// -----------------------------------------------------------------------------

export interface ClosedLoopOpts {
  numSteps: number;
  /** Seed control u0 — fed to the plant before any controller fires.
   *  Default = zero vector. */
  u0?: number[];
  /** Optional in-loop estimator (between plant and controller). */
  estimator?: EstimatorBlock;
}

export interface ClosedLoopResult {
  trajectory: number[][];
  controls: number[][];
  measurements: number[][];
  estimates?: number[][];
  numSteps: number;
}

/** Drive a plant + controller (and optional estimator) in lockstep for
 *  `numSteps` ticks. Wires up the connections if not already wired.
 *  Calls each block's `assertPreconditions()` BEFORE any tick. */
export function runClosedLoop(plant: PlantBlock, controller: ControllerBlock,
                              opts: ClosedLoopOpts): ClosedLoopResult {
  Preconditions.integer('runClosedLoop', 'numSteps', opts.numSteps);
  Preconditions.check('runClosedLoop', 'numSteps', 'be >= 1', opts.numSteps >= 1, opts.numSteps);
  // Auto-wire if the user hasn't already.
  ensureConnected(plant, opts.estimator ?? controller);
  if (opts.estimator) ensureConnected(opts.estimator, controller);
  ensureConnected(controller, plant);
  // Estimator usually also needs the control signal, so loop u into estimator.
  if (opts.estimator) ensureConnected(controller, opts.estimator);

  // Pre-run guards on every block.
  plant.assertPreconditions();
  controller.assertPreconditions();
  if (opts.estimator) opts.estimator.assertPreconditions();

  // Seed control.
  const u0 = (opts.u0 ?? new Array(controller.mDim).fill(0));
  Preconditions.lengthEq('runClosedLoop', 'u0', u0, controller.mDim);
  Preconditions.allFinite('runClosedLoop', 'u0', u0);
  plant.takeItem(new VectorSignal(u0, 'u', 0));

  // Step loop.
  const stepBN = bgn(plant.dt);
  for (let k = 0; k < opts.numSteps; k++) {
    plant.runTimeStep(stepBN);
    if (opts.estimator) opts.estimator.runTimeStep(stepBN);
    controller.runTimeStep(stepBN);
  }
  return {
    trajectory: plant.stateHistory.map(x => x.slice()),
    controls: controller.outputHistory.map(u => u.slice()),
    measurements: plant.outputHistory.map(y => y.slice()),
    estimates: opts.estimator ? opts.estimator.estimateHistory.map(x => x.slice()) : undefined,
    numSteps: opts.numSteps,
  };
}

function ensureConnected(src: MultiDirectionalSignalEntity<unknown, number>,
                         tgt: MultiDirectionalSignalEntity<unknown, number>): void {
  for (const c of src.connectionsOut) if (c.target === tgt) return;
  src.addOutConnection(tgt as any);
  tgt.addInConnection(src as any);
}
