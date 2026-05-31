// RUST MIGRATION: target module src/des/general/feedback_linearization.rs.
// RUST MIGRATION: PendulumParams, FeedbackLinearizationOpts, and FeedbackLinearizationResult become serde structs; ClosedLoopResult inheritance becomes composition/flattening.
// RUST MIGRATION: PendulumPlant and FeedbackLinearizationController become structs implementing Plant/Controller traits instead of subclassing blocks.
// RUST MIGRATION: runFeedbackLinearization is a DES/control transform returning Result; rk4 is a private free function over slices/arrays.
// RUST MIGRATION: Keep vector math ownership explicit with [f64; 2] where possible to make borrow checking simple.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/feedback-linearization.rs  (module des::general::feedback_linearization)
// 1:1 file move. Feedback-linearization (computed-torque) control of an inverted pendulum, as DES blocks.
//
// Declarations → Rust:
//   interface PendulumParams / FeedbackLinearizationOpts -> structs (Default; optionals -> Option<T>)
//   interface FeedbackLinearizationResult extends ClosedLoopResult -> struct (compose/flatten base fields)
//   class PendulumPlant / FeedbackLinearizationController -> structs `impl` PlantBlock/ControllerBlock traits
//   fn runFeedbackLinearization   -> free fn (or PureTransform<Opts, Result>)
//   fn rk4                        -> assoc fn taking `f: impl Fn(&[f64]) -> Vec<f64>`
//
// Conversion notes (file-specific):
//   - `rk4` takes a derivative CLOSURE `f` -> generic `Fn` bound (or `&dyn Fn`).
//   - `interface ... extends ClosedLoopResult` -> Rust has no interface inheritance; flatten or compose.
//   - Greek idents in docs (θ, ℓ) are comments only; deterministic (no RNG/clock).
// =============================================================================

// =============================================================================
// general/feedback-linearization.ts — FEEDBACK LINEARIZATION (also called
// "computed-torque control" in robotics; Khalil 2002 ch.13) on the
// canonical INVERTED PENDULUM swing-down/track problem.
//
// PROBLEM
// ───────
//   Simple (mass-on-a-rod) pendulum:
//
//       θ̈ = −(g/ℓ) sin θ − (c/(mℓ²)) θ̇ + (1/(mℓ²)) τ
//
//   where θ is the angle from vertical-DOWN equilibrium, ℓ length,
//   g gravity, c viscous damping, m mass at the tip, τ the applied
//   torque. With the abbreviations
//
//       a(θ, θ̇) = −(g/ℓ) sin θ − (c/(mℓ²)) θ̇
//       b       = 1/(mℓ²)
//
//   we have θ̈ = a(θ, θ̇) + b τ. FEEDBACK LINEARIZATION cancels the
//   nonlinearity by setting
//
//       τ = (1/b)·(v − a(θ, θ̇))
//
//   so that the closed-loop dynamics become the linear system
//
//       θ̈ = v
//
//   We then choose v as a standard PD law to track a desired angle θ_d:
//
//       v = θ̈_d + k_v (θ̇_d − θ̇) + k_p (θ_d − θ)
//
//   This drives the error (θ − θ_d, θ̇ − θ̇_d) → 0 with poles
//   determined by k_p, k_v.
//
// AS A DES BLOCK
// ──────────────
//   `PendulumPlant extends PlantBlock`,
//   `FeedbackLinearizationController extends ControllerBlock`. The
//   controller assumes EXACT knowledge of m, ℓ, g, c (the canonical
//   feedback-lin assumption) — the demonstration here verifies that
//   under that assumption the trajectory tracks any smooth reference
//   θ_d(t) to within numerical roundoff.
//
//   For ROBUSTNESS to model mismatch the standard practice is to add a
//   sliding-mode outer loop on top of v — left as exercise (we already
//   have that controller in `sliding-mode-control.ts`).
// =============================================================================

import {
  PlantBlock, ControllerBlock, runClosedLoop, ClosedLoopResult,
} from './des-base/control-blocks';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// PLANT: SIMPLE PENDULUM
// -----------------------------------------------------------------------------

export interface PendulumParams {
  m: number;       // mass at tip [kg]
  l: number;       // length [m]
  g: number;       // gravity [m/s²]
  c: number;       // viscous damping [N·m·s/rad]
}

class PendulumPlant extends PlantBlock {
  readonly params: PendulumParams;
  constructor(theta0: number, thetaDot0: number, params: PendulumParams, dt: number) {
    super('pendulum-plant', [theta0, thetaDot0], dt, 1);
    this.params = params;
  }
  protected dynamics(x: readonly number[], u: readonly number[], dt: number): number[] {
    // RK4 for accuracy (the Euler version visibly drifts on a stiff
    // swing-up). The state is [θ, θ̇].
    const f = (xx: readonly number[]): readonly number[] => {
      const [θ, θd] = xx;
      const {m, l, g, c} = this.params;
      const θdd = -(g / l) * Math.sin(θ) - (c / (m * l * l)) * θd + (1 / (m * l * l)) * u[0];
      return [θd, θdd];
    };
    return rk4(x, f, dt);
  }
}

// -----------------------------------------------------------------------------
// FEEDBACK-LINEARIZATION CONTROLLER (PD ON LINEARISED LOOP)
// -----------------------------------------------------------------------------

class FeedbackLinearizationController extends ControllerBlock {
  private readonly params: PendulumParams;
  private readonly kp: number;
  private readonly kv: number;
  private readonly ref: (t: number) => {
    theta: number; thetaDot: number; thetaDDot: number;
  };
  private readonly dtCache: number;
  /** Running history of reference (θ_d, θ̇_d) for diagnostics. */
  readonly thetaDHistory: number[] = [];
  readonly thetaDDotHistory: number[] = [];
  /** Tracking error history. */
  readonly errorHistory: number[] = [];

  constructor(opts: {params: PendulumParams; kp: number; kv: number;
                     ref: FeedbackLinearizationController['ref']; dt: number;
                     uBound?: number}) {
    super('feedback-lin', 1);
    this.params = opts.params;
    this.kp = opts.kp; this.kv = opts.kv;
    this.ref = opts.ref;
    this.dtCache = opts.dt;
    if (opts.uBound !== undefined)
      this.setSaturation([-opts.uBound], [opts.uBound]);
  }
  protected getDt(): number { return this.dtCache; }
  protected controlLaw(y: readonly number[], _tick: number, t: number): number[] {
    const θ = y[0]; const θd = y[1];
    const {theta: θd_ref, thetaDot: θdd_ref, thetaDDot: θddd_ref} = this.ref(t);
    const e = θ - θd_ref;
    const ed = θd - θdd_ref;
    const v = θddd_ref - this.kv * ed - this.kp * e;
    const {m, l, g, c} = this.params;
    const a = -(g / l) * Math.sin(θ) - (c / (m * l * l)) * θd;
    const b = 1 / (m * l * l);
    const tau = (1 / b) * (v - a);
    this.thetaDHistory.push(θd_ref);
    this.thetaDDotHistory.push(θdd_ref);
    this.errorHistory.push(e);
    return [tau];
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface FeedbackLinearizationOpts {
  params?: Partial<PendulumParams>;
  /** Initial angle θ₀ (rad). Default π (downward = stable equilibrium); use
   *  0 to track to the unstable upright. */
  theta0?: number;
  /** Initial angular velocity θ̇₀ (rad/s). Default 0. */
  thetaDot0?: number;
  /** Reference function returning (θ_d(t), θ̇_d(t), θ̈_d(t)). Default:
   *  track sinusoidal motion of amplitude 1 rad at 0.5 Hz. */
  reference?: (t: number) => {theta: number; thetaDot: number; thetaDDot: number};
  /** PD gains. Default kp = 25, kv = 10 (critically damped, ω_n = 5). */
  kp?: number;
  kv?: number;
  /** Saturation on torque |τ|. Default 100. */
  uBound?: number;
  dt?: number;
  numSteps?: number;
}

export interface FeedbackLinearizationResult extends ClosedLoopResult {
  /** RMS tracking error in steady state. */
  rmsErrorSteadyState: number;
  /** Reference history for plotting. */
  thetaDHistory: number[];
}

export function runFeedbackLinearization(opts: FeedbackLinearizationOpts = {}):
  FeedbackLinearizationResult {
  const params: PendulumParams = {
    m: opts.params?.m ?? 1,
    l: opts.params?.l ?? 1,
    g: opts.params?.g ?? 9.81,
    c: opts.params?.c ?? 0.1,
  };
  const cls = 'runFeedbackLinearization';
  // Mass and length must be positive — they appear as 1/(m l²) in
  // the dynamics, divide-by-zero hazards otherwise.
  Preconditions.positive(cls, 'params.m', params.m);
  Preconditions.positive(cls, 'params.l', params.l);
  Preconditions.nonNegative(cls, 'params.g', params.g);
  Preconditions.nonNegative(cls, 'params.c', params.c);
  // PD gains must be positive for closed-loop stability of the linearised
  // system theta_ddot = v with v = -kp·e - kv·e_dot.
  Preconditions.positive(cls, 'kp', opts.kp ?? 25);
  Preconditions.positive(cls, 'kv', opts.kv ?? 10);
  if (opts.uBound !== undefined) Preconditions.positive(cls, 'uBound', opts.uBound);
  Preconditions.positive(cls, 'dt', opts.dt ?? 0.01);
  Preconditions.integerInRange(cls, 'numSteps', opts.numSteps ?? 1000, 1, 1e9);
  if (opts.theta0 !== undefined) Preconditions.finite(cls, 'theta0', opts.theta0);
  if (opts.thetaDot0 !== undefined) Preconditions.finite(cls, 'thetaDot0', opts.thetaDot0);
  const θ0 = opts.theta0 ?? Math.PI;       // start hanging down
  const θd0 = opts.thetaDot0 ?? 0;
  const ref = opts.reference ?? ((t: number) => ({
    theta: Math.sin(2 * Math.PI * 0.5 * t),
    thetaDot: 2 * Math.PI * 0.5 * Math.cos(2 * Math.PI * 0.5 * t),
    thetaDDot: -((2 * Math.PI * 0.5) ** 2) * Math.sin(2 * Math.PI * 0.5 * t),
  }));
  const kp = opts.kp ?? 25;
  const kv = opts.kv ?? 10;
  const dt = opts.dt ?? 0.01;
  const numSteps = opts.numSteps ?? 1000;
  const plant = new PendulumPlant(θ0, θd0, params, dt);
  const ctrl = new FeedbackLinearizationController({
    params, kp, kv, ref, dt, uBound: opts.uBound ?? 100,
  });
  const closed = runClosedLoop(plant, ctrl, {numSteps});

  const half = Math.floor(ctrl.errorHistory.length / 2);
  const tail = ctrl.errorHistory.slice(half);
  const rms = Math.sqrt(tail.reduce((s, x) => s + x * x, 0) / Math.max(1, tail.length));
  return {
    ...closed,
    rmsErrorSteadyState: rms,
    thetaDHistory: ctrl.thetaDHistory.slice(),
  };
}

// -----------------------------------------------------------------------------
// RK4 STEP (used by the pendulum plant for accuracy)
// -----------------------------------------------------------------------------

function rk4(x: readonly number[], f: (x: readonly number[]) => readonly number[], dt: number): number[] {
  const k1 = f(x);
  const x2 = x.map((xi, i) => xi + 0.5 * dt * k1[i]);
  const k2 = f(x2);
  const x3 = x.map((xi, i) => xi + 0.5 * dt * k2[i]);
  const k3 = f(x3);
  const x4 = x.map((xi, i) => xi + dt * k3[i]);
  const k4 = f(x4);
  return x.map((xi, i) => xi + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}
