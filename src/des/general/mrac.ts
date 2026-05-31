'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/mrac.rs  (module des::general::mrac)
// 1:1 file move. Model-reference adaptive control (Lyapunov/MIT-rule) as a DES control loop.
//
// Declarations → Rust:
//   interface MRACOpts                          -> struct (Default-derivable)
//   interface MRACResult extends ClosedLoopResult -> struct embedding/flattening ClosedLoopResult
//   class UnknownGainPlant extends PlantBlock     -> struct + impl Plant trait (private)
//   class ReferenceModel                          -> struct + impl (private)
//   class MRACController extends ControllerBlock   -> struct + impl Controller trait (private)
//   fn runMRAC                                    -> free fn / assoc fn
//
// Conversion notes (file-specific):
//   - PlantBlock/ControllerBlock template-method bases -> traits with default fns; adaptive
//     gains θ_x, θ_r are mutable controller struct fields advanced via `&mut self` each tick.
//   - `extends ClosedLoopResult`: compose by embedding the base struct (no interface inheritance).
//   - all numerics are `f64`; deterministic (no RNG/clock/Map).
// =============================================================================
// general/mrac.ts — MODEL REFERENCE ADAPTIVE CONTROL (Whitaker 1958, MIT
// rule; Narendra & Annaswamy 1989, Lyapunov-based MRAC).
//
// PROBLEM
// ───────
//   First-order plant with UNKNOWN gain b > 0 and unknown coefficient a:
//
//       ẋ = a x + b u
//
//   Reference model (the desired closed-loop response):
//
//       ẋ_m = a_m x_m + b_m r,         a_m < 0 (stable)
//
//   We want the plant output x to track x_m for any reference r(t), in
//   spite of the unknown a, b. Choose
//
//       u = θ_x x + θ_r r
//
//   The Lyapunov-based MIT-rule update law is
//
//       θ̇_x = −γ · e · x · sign(b)
//       θ̇_r = −γ · e · r · sign(b)
//
//   where e = x − x_m (tracking error) and γ > 0 is the adaptation gain.
//   This drives e → 0 as t → ∞ for any bounded r without ever knowing
//   a, b (only sign(b) is needed).
//
// AS A DES BLOCK
// ──────────────
//   `MRACController extends ControllerBlock` keeps θ_x, θ_r as internal
//   state, advances them with the gradient law each tick. The plant
//   `UnknownGainPlant` runs ẋ = a x + b u with hidden a, b.
// =============================================================================

import {
  PlantBlock, ControllerBlock, runClosedLoop, ClosedLoopResult,
} from './des-base/control-blocks';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// PLANT (TRUE PARAMETERS HIDDEN FROM CONTROLLER)
// -----------------------------------------------------------------------------

class UnknownGainPlant extends PlantBlock {
  private readonly a: number;
  private readonly b: number;
  constructor(x0: number, a: number, b: number, dt: number) {
    super('unknown-gain-plant', [x0], dt, 1);
    this.a = a; this.b = b;
  }
  protected dynamics(x: readonly number[], u: readonly number[], dt: number): number[] {
    // Forward Euler.
    return [x[0] + dt * (this.a * x[0] + this.b * u[0])];
  }
}

// -----------------------------------------------------------------------------
// REFERENCE MODEL (PLAYS A SECOND PLANT BLOCK BUT WITH KNOWN PARAMS)
// -----------------------------------------------------------------------------

class ReferenceModel {
  private xm: number;
  private readonly am: number;
  private readonly bm: number;
  private readonly dt: number;
  readonly history: number[] = [];

  constructor(xm0: number, am: number, bm: number, dt: number) {
    this.xm = xm0; this.am = am; this.bm = bm; this.dt = dt;
    this.history.push(xm0);
  }
  step(r: number): number {
    this.xm = this.xm + this.dt * (this.am * this.xm + this.bm * r);
    this.history.push(this.xm);
    return this.xm;
  }
  current(): number { return this.xm; }
}

// -----------------------------------------------------------------------------
// MRAC CONTROLLER
// -----------------------------------------------------------------------------

class MRACController extends ControllerBlock {
  private theta_x = 0;
  private theta_r = 0;
  private readonly gamma: number;
  private readonly signB: number;
  private readonly dtCache: number;
  private readonly refModel: ReferenceModel;
  private readonly r: (t: number) => number;
  /** Tracking-error history. */
  readonly trackingError: number[] = [];
  readonly thetaXHistory: number[] = [];
  readonly thetaRHistory: number[] = [];

  constructor(opts: {gamma: number; signB: number; dt: number;
                     refModel: ReferenceModel; r: (t: number) => number;
                     uBound?: number}) {
    super('mrac', 1);
    this.gamma = opts.gamma; this.signB = opts.signB; this.dtCache = opts.dt;
    this.refModel = opts.refModel; this.r = opts.r;
    if (opts.uBound !== undefined)
      this.setSaturation([-opts.uBound], [opts.uBound]);
  }
  protected getDt(): number { return this.dtCache; }
  protected controlLaw(y: readonly number[], _tick: number, t: number): number[] {
    const x = y[0];
    const r = this.r(t);
    // 1. Step the reference model first to compute x_m(t).
    const xm = this.refModel.step(r);
    // 2. Tracking error e = x - x_m.
    const e = x - xm;
    // 3. Gradient update on θ.
    this.theta_x += -this.gamma * e * x * this.signB * this.dtCache;
    this.theta_r += -this.gamma * e * r * this.signB * this.dtCache;
    // 4. Control u = θ_x x + θ_r r.
    const u = this.theta_x * x + this.theta_r * r;
    this.trackingError.push(e);
    this.thetaXHistory.push(this.theta_x);
    this.thetaRHistory.push(this.theta_r);
    return [u];
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface MRACOpts {
  /** True (hidden) plant parameter a. Default 1 (unstable plant). */
  a?: number;
  /** True (hidden) plant parameter b > 0. Default 2. */
  b?: number;
  /** Reference model param a_m (must be negative). Default −2. */
  am?: number;
  /** Reference model param b_m. Default 2. */
  bm?: number;
  x0?: number;
  xm0?: number;
  /** Adaptation gain γ. Default 5. */
  gamma?: number;
  /** Reference signal r(t). Default: square wave amplitude 1, period 4s. */
  reference?: (t: number) => number;
  dt?: number;
  numSteps?: number;
  uBound?: number;
}

export interface MRACResult extends ClosedLoopResult {
  trackingError: number[];
  thetaXHistory: number[];
  thetaRHistory: number[];
  /** Reference-model trajectory (x_m). */
  referenceTrajectory: number[];
  /** Reference signal r(t) at each tick. */
  rHistory: number[];
  /** RMS tracking error over the LAST half of the run (steady-state). */
  rmsErrorSteadyState: number;
  /** Final θ_x, θ_r (should approach the ideal θ*_x = (a_m − a)/b,
   *  θ*_r = b_m/b). */
  finalTheta: [number, number];
  /** Ideal θ values (closed-form). */
  idealTheta: [number, number];
}

export function runMRAC(opts: MRACOpts = {}): MRACResult {
  const a = opts.a ?? 1;
  const b = opts.b ?? 2;
  const am = opts.am ?? -2;
  const bm = opts.bm ?? 2;
  const x0 = opts.x0 ?? 0;
  const xm0 = opts.xm0 ?? 0;
  const gamma = opts.gamma ?? 5;
  const dt = opts.dt ?? 0.01;
  const numSteps = opts.numSteps ?? 4000;
  const refSignal = opts.reference ?? ((t: number) => Math.floor(t / 2) % 2 === 0 ? 1 : -1);

  // Pre-run guards. The two key invariants for MRAC convergence are
  //   (i)  b > 0  (sign-known assumption — change the law for b < 0)
  //   (ii) a_m < 0  (reference model must be Hurwitz-stable).
  const cls = 'runMRAC';
  Preconditions.finite(cls, 'a', a);
  Preconditions.positive(cls, 'b', b);
  Preconditions.check(cls, 'am', 'be < 0 (reference model must be stable)', am < 0, am);
  Preconditions.finite(cls, 'bm', bm);
  Preconditions.finite(cls, 'x0', x0);
  Preconditions.finite(cls, 'xm0', xm0);
  Preconditions.positive(cls, 'gamma', gamma);
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.integerInRange(cls, 'numSteps', numSteps, 1, 1e9);
  // Stability margin guard: the MIT-rule update is conservatively
  // safe when γ·dt is moderate. A huge γ·dt blows up the gradient.
  Preconditions.check(cls, 'gamma*dt', 'be <= 1 for numerical stability of the MIT-rule',
    gamma * dt <= 1 + 1e-9, gamma * dt);

  const plant = new UnknownGainPlant(x0, a, b, dt);
  const refModel = new ReferenceModel(xm0, am, bm, dt);
  const ctrl = new MRACController({
    gamma, signB: Math.sign(b), dt,
    refModel, r: refSignal, uBound: opts.uBound,
  });
  const closed = runClosedLoop(plant, ctrl, {numSteps});

  // RMS tracking error in steady state.
  const half = Math.floor(ctrl.trackingError.length / 2);
  const tail = ctrl.trackingError.slice(half);
  const rms = Math.sqrt(tail.reduce((s, x) => s + x * x, 0) / Math.max(1, tail.length));

  const idealThetaX = (am - a) / b;
  const idealThetaR = bm / b;

  // Reconstruct r history.
  const rHist: number[] = [];
  for (let k = 0; k < numSteps; k++) rHist.push(refSignal(k * dt));

  return {
    ...closed,
    trackingError: ctrl.trackingError.slice(),
    thetaXHistory: ctrl.thetaXHistory.slice(),
    thetaRHistory: ctrl.thetaRHistory.slice(),
    referenceTrajectory: refModel.history.slice(),
    rHistory: rHist,
    rmsErrorSteadyState: rms,
    finalTheta: [ctrl.thetaXHistory[ctrl.thetaXHistory.length - 1],
                 ctrl.thetaRHistory[ctrl.thetaRHistory.length - 1]],
    idealTheta: [idealThetaX, idealThetaR],
  };
}
