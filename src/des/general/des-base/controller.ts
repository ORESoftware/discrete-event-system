'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/controller.rs
// - Keep file-for-file. ObservationToken and ControlToken become token structs
//   parameterized over observation/control payload types.
// - ControllerStation becomes a trait plus shared station-state struct; preserve
//   observe/control/reset hooks as trait methods implemented by concrete
//   controllers.
// - Keep channel routing as explicit enums/consts where possible; any pure
//   controller law used as a graph node should become a PureTransform or
//   PureTransformEntity implementation.
// - Convert validation failures to Result instead of throwing.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/controller.rs  (module des::general::des_base::controller)
// 1:1 file move. Template-method base for feedback controllers (bang-bang / PID /
// fuzzy / MPC / sliding-mode / LQR-LQG) over generic obs `O` and control `U`.
//
// Declarations → Rust:
//   class ObservationToken<O> / ControlToken<U> -> struct + impl Token
//   abstract class ControllerStation<O,U>        -> trait ControllerStation<O, U>: DESStation
//
// Conversion notes (file-specific):
//   - TEMPLATE METHOD: `runTimeStep` (and `step`) are final; required hook
//     controlLaw -> required trait fn; uMin/uMax/onTick/reset/clamp -> provided defaults.
//   - `clamp` uses `typeof u !== 'number'` then `as unknown as U` — a runtime
//     type test that DOESN'T translate. In Rust, make saturation part of a
//     `Saturate` bound on `U`, or specialise `ControllerStation<f64, f64>`;
//     non-scalar `U` simply skips clamping.
//   - `uMin()/uMax(): U | null` -> `Option<U>`.
//   - `ControlToken.observation: unknown` -> a concrete `O` generic or `Box<dyn Any>`
//     (prefer generic).
//   - History `U[]`/`O[]` -> `Vec<U>`/`Vec<O>`; `.length = 0` reset -> `.clear()`.
// =============================================================================

// =============================================================================
// general/des-base/controller.ts — base class for FEEDBACK CONTROL stations:
// bang-bang, PID, fuzzy, model-predictive control (MPC), receding-horizon
// DP, sliding-mode, LQR / LQG, …
//
// PROBLEM SHAPE
// ─────────────
//   On each tick the controller receives a measurement / observation
//   `o_k`, optionally a forecast / model, and produces a control action
//   `u_k` to apply to the plant. It may carry persistent state across
//   ticks (PID integrator, MPC plan, fuzzy memory, …).
//
//   The DIFFERENTIATOR is the control law itself.
//
// TOPOLOGY
// ────────
//
//   in:  CH_OBSERVATION  (ObservationToken)   ← from sensor / plant
//   out: CH_CONTROL      (ControlToken)       → to actuator / plant
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep():
//     for each ObservationToken in the inbox:
//       u  = controlLaw(o, internalState)        ← the abstract hook
//       u' = clamp(u, uMin, uMax)                ← optional saturation
//       record(o, u, u')                          ← per-tick history
//       emit ControlToken(u') on CH_CONTROL
//
// HOOKS (abstract)
// ────────────────
//   controlLaw(o) → u                   — the actual control rule
//
// HOOKS (optional override)
// ─────────────────────────
//   uMin / uMax (default ±∞)            — saturation
//   onTick(o, u, uClamped)              — instrumentation
//   reset()                             — wipe internal state
// =============================================================================

import {DESStation, ChannelName, Token} from './station';

/** A single sensor reading delivered to the controller. */
export class ObservationToken<O = number> implements Token {
  constructor(public observation: O, public tick: number, public time: number) {}
}

/** A single control action sent from the controller. */
export class ControlToken<U = number> implements Token {
  constructor(public control: U, public observation: unknown, public tick: number, public time: number) {}
}

export abstract class ControllerStation<O = number, U = number> extends DESStation {
  static readonly CH_OBSERVATION: ChannelName = 'observation';
  static readonly CH_CONTROL: ChannelName = 'control';

  /** Tick counter: incremented every time a control law fires. */
  protected ticksProcessed = 0;
  /** Per-tick history (uClamped). Subclass adds richer trace if needed. */
  readonly controlHistory: U[] = [];
  readonly observationHistory: O[] = [];

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  /** The CONTROL LAW — receive an observation, produce a control output.
   *  Subclasses are free to read/mutate their own persistent state here
   *  (PID integrator, fuzzy-PI accumulator, MPC plan cache, …). */
  protected abstract controlLaw(observation: O, tick: number, time: number): U;

  // ── HOOKS (optional override) ────────────────────────────────────────────

  /** Control-input lower saturation bound. Subclasses overriding both this
   *  and `uMax` get hard saturation for free. Default: -Infinity. */
  protected uMin(): U | null { return null; }
  /** Control-input upper saturation bound. Default: +Infinity. */
  protected uMax(): U | null { return null; }
  /** Per-tick instrumentation hook. */
  protected onTick(_observation: O, _u: U, _uClamped: U): void {}
  /** Reset internal state (e.g. start of a new run). */
  reset(): void {
    this.ticksProcessed = 0;
    this.controlHistory.length = 0;
    this.observationHistory.length = 0;
  }

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  runTimeStep(): void {
    const observations = this.drain<ObservationToken<O>>(ControllerStation.CH_OBSERVATION);
    for (const obs of observations) {
      const u = this.controlLaw(obs.observation, obs.tick, obs.time);
      const uClamped = this.clamp(u);
      this.observationHistory.push(obs.observation);
      this.controlHistory.push(uClamped);
      this.onTick(obs.observation, u, uClamped);
      this.emit(new ControlToken<U>(uClamped, obs.observation, obs.tick, obs.time),
                ControllerStation.CH_CONTROL);
      this.ticksProcessed += 1;
    }
  }

  override hasWork(): boolean {
    return this.inboxSize(ControllerStation.CH_OBSERVATION) > 0;
  }

  /** Synchronous one-shot helper — runs the controlLaw on a single
   *  observation and returns the clamped control. Useful when a tight
   *  simulation loop wants to embed the controller without spinning up
   *  a full DES topology, while still benefiting from saturation,
   *  history bookkeeping, and the onTick hook. */
  step(observation: O, tick: number, time: number): U {
    const u = this.controlLaw(observation, tick, time);
    const uClamped = this.clamp(u);
    this.observationHistory.push(observation);
    this.controlHistory.push(uClamped);
    this.onTick(observation, u, uClamped);
    this.ticksProcessed += 1;
    return uClamped;
  }

  // ── INTERNAL ─────────────────────────────────────────────────────────────

  /** Saturation. Default works for `U = number`; subclasses with vector U
   *  can override. */
  protected clamp(u: U): U {
    if (typeof u !== 'number') return u;
    const lo = this.uMin();
    const hi = this.uMax();
    let v = u;
    if (typeof lo === 'number' && v < lo) v = lo;
    if (typeof hi === 'number' && v > hi) v = hi;
    return v as unknown as U;
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getTicksProcessed(): number { return this.ticksProcessed; }
}
