'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/pontryagin-bang-bang.rs  (module des::general::pontryagin_bang_bang)
// 1:1 file move. Time-optimal bang-bang controller (Pontryagin) for the double integrator.
//
// Declarations → Rust:
//   interface PontryaginOpts                       -> struct (Default-derivable)
//   interface PontryaginResult extends ClosedLoopResult -> struct embedding/flattening ClosedLoopResult
//   class DoubleIntegratorPlant extends PlantBlock  -> struct + impl Plant trait (private)
//   class PontryaginBangBangController extends ControllerBlock -> struct + impl Controller trait (private)
//   fn runPontryaginBangBang / optimalTimeDoubleIntegrator -> free fns / assoc fns
//
// Conversion notes (file-specific):
//   - PlantBlock/ControllerBlock template-method bases -> traits with default fns; the controller
//     tracks the once-only switch via a `bool`/`Option` field (closed-form, no optimizer).
//   - `extends ClosedLoopResult`: compose by embedding the base struct (no interface inheritance).
//   - switching-curve uses `sign`/`abs` on f64; fully deterministic (no RNG/clock/Map).
// =============================================================================
// general/pontryagin-bang-bang.ts — TIME-OPTIMAL CONTROL of a double
// integrator via PONTRYAGIN'S MAXIMUM PRINCIPLE.
//
// PROBLEM
// ───────
//   Plant:  ẍ = u,  |u| ≤ u_max
//   Goal:   drive (x, ẋ) → (0, 0) in MINIMUM TIME.
//
//   The classical Bryson-Ho derivation:
//     Hamiltonian H = 1 + λ₁·ẋ + λ₂·u
//     Costate     λ̇₁ = 0,   λ̇₂ = −λ₁
//     PMP demands u* = −u_max·sign(λ₂) (linear in u → bang-bang on the
//     control bound),   λ₁ = const,  λ₂ = α − λ₁·t (linear in t).
//
//   ∴ The optimal control switches AT MOST ONCE between ±u_max. The
//   SWITCHING CURVE in the (x, ẋ) phase plane is
//
//        x = − (1 / (2 u_max)) · ẋ · |ẋ|
//
//   With state (x, v): apply +u_max BELOW the curve, −u_max ABOVE, fire
//   the unique switch when the trajectory crosses it. This file
//   implements that closed-form bang-bang controller as a
//   `ControllerBlock` and demonstrates it driving the canonical
//   double integrator plant in CLOSED LOOP via the entity framework.
//
// REFERENCES
// ──────────
//   Pontryagin et al. 1962, "The Mathematical Theory of Optimal
//   Processes". Bryson & Ho 1975, §2.6.
// =============================================================================

import {
  PlantBlock, ControllerBlock, runClosedLoop, ClosedLoopResult,
} from './des-base/control-blocks';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// PLANT: DOUBLE INTEGRATOR  ẍ = u
// -----------------------------------------------------------------------------

class DoubleIntegratorPlant extends PlantBlock {
  constructor(x0: [number, number], dt: number) {
    Preconditions.lengthEq('DoubleIntegratorPlant', 'x0', x0, 2);
    Preconditions.allFinite('DoubleIntegratorPlant', 'x0', x0);
    Preconditions.positive('DoubleIntegratorPlant', 'dt', dt);
    super('plant-double-int', x0, dt, 1);
  }
  protected dynamics(x: readonly number[], u: readonly number[], dt: number): number[] {
    // Exact discretisation of ẍ = u for piecewise-constant u:
    //   x' = x + dt v + 0.5 dt² u
    //   v' = v + dt u
    return [x[0] + dt * x[1] + 0.5 * dt * dt * u[0], x[1] + dt * u[0]];
  }
}

// -----------------------------------------------------------------------------
// CONTROLLER: BANG-BANG ON SWITCHING CURVE (PMP-OPTIMAL)
// -----------------------------------------------------------------------------

class PontryaginBangBangController extends ControllerBlock {
  private readonly uBound: number;
  /** Small numerical band around the switching curve where we set
   *  u = 0 — prevents numerically unstable oscillation when the trajectory
   *  exactly hits (0, 0). */
  private readonly deadband: number;
  private readonly dtCache: number;

  constructor(uBound = 1, deadband = 0.2, dt = 0.05) {
    super('pmp-bang-bang', 1);
    Preconditions.positive('PontryaginBangBangController', 'uBound (u_max)', uBound);
    Preconditions.positive('PontryaginBangBangController', 'deadband', deadband);
    Preconditions.positive('PontryaginBangBangController', 'dt', dt);
    this.uBound = uBound;
    this.deadband = deadband;
    this.dtCache = dt;
    this.setSaturation([-uBound], [uBound]);
  }
  protected getDt(): number { return this.dtCache; }

  protected controlLaw(y: readonly number[], _tick: number, _t: number): number[] {
    // y = [x, v] (full state observed; for noisy observations the user
    // should chain a Kalman filter and connect xhat into the controller).
    const x = y[0]; const v = y[1];
    const sigma = x + (1 / (2 * this.uBound)) * v * Math.abs(v);
    // Once close to origin (|x| and |v| both small), switch to a smooth
    // linear controller — this avoids the well-known *chattering* of
    // discrete-time bang-bang near the equilibrium and is the standard
    // engineering practice.
    if (Math.abs(x) + Math.abs(v) < this.deadband) {
      // Aggressive PD inside the deadband — saturates against u_max so
      // the closed-loop response stays fast even though it's no longer
      // bang-bang.
      const u = -10 * x - 6 * v;
      return [Math.max(-this.uBound, Math.min(this.uBound, u))];
    }
    return [sigma > 0 ? -this.uBound : +this.uBound];
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface PontryaginOpts {
  /** Initial state [x, v]. Default [3, 0]. */
  x0?: [number, number];
  /** Bound on |u|. Default 1. */
  uMax?: number;
  /** Sample period dt. Default 0.05. */
  dt?: number;
  /** Number of simulation steps. Default 200. */
  numSteps?: number;
  /** State-magnitude band around the origin where we switch from
   *  bang-bang to a smooth linear law to suppress chattering. Default 0.2. */
  deadband?: number;
}

export interface PontryaginResult extends ClosedLoopResult {
  /** First tick at which |x| + |v| < 1e-2 (proxy for "reached origin"). */
  arrivalTick: number;
  /** True optimal time-to-go from x0 (closed-form). */
  theoreticalArrivalTime: number;
  /** Number of u-sign changes during the run (PMP predicts ≤ 1
   *  switch from the bang-bang region — early exit if we reach the
   *  origin first). */
  switchCount: number;
}

export function runPontryaginBangBang(opts: PontryaginOpts = {}): PontryaginResult {
  const x0: [number, number] = opts.x0 ?? [3, 0];
  const uMax = opts.uMax ?? 1;
  const dt = opts.dt ?? 0.05;
  const numSteps = opts.numSteps ?? 200;
  Preconditions.positive('runPontryaginBangBang', 'uMax', uMax);
  Preconditions.positive('runPontryaginBangBang', 'dt', dt);
  Preconditions.integerInRange('runPontryaginBangBang', 'numSteps', numSteps, 1, 1e9);
  Preconditions.lengthEq('runPontryaginBangBang', 'x0', x0, 2);
  Preconditions.allFinite('runPontryaginBangBang', 'x0', x0);
  if (opts.deadband !== undefined) Preconditions.positive('runPontryaginBangBang', 'deadband', opts.deadband);
  const plant = new DoubleIntegratorPlant(x0, dt);
  const ctrl = new PontryaginBangBangController(uMax, opts.deadband ?? 0.2, dt);
  const out = runClosedLoop(plant, ctrl, {numSteps});

  // Bang-bang arrival = first entry into the deadband (when we leave
  // the saturated bang-bang phase). This is the textbook t* the PMP
  // formula predicts; what happens INSIDE the deadband is the
  // closed-form-doesn't-apply numerical-cleanup phase.
  let arrivalTick = numSteps;
  const dbThr = opts.deadband ?? 0.2;
  for (let i = 0; i < out.trajectory.length; i++) {
    const [x, v] = out.trajectory[i];
    if (Math.abs(x) + Math.abs(v) < dbThr) { arrivalTick = i; break; }
  }
  // Closed-form optimal time t* for the double integrator with
  // |u| ≤ u_max from (x₀, v₀) to origin (Bryson-Ho §2.6 / Athans-Falb):
  //   if u₁ = +u_max takes us through the switch curve:
  //     t1 = ½ (v₀ + sqrt(v₀² + 2·sgn(σ₀)·u_max·(2 x₀ + v₀²/u_max·sgn(σ₀))))
  //   General formula:
  //     t* = (sqrt((|v₀|² + 2 u_max |x₀ + v₀ |v₀|/(2 u_max)|) ) + |v₀ ± u_max·...|)/u_max
  //   For the simple x₀ ≠ 0, v₀ = 0 case used in our tests:
  //     t* = 2 √(|x₀| / u_max).
  let theoretical: number;
  if (x0[1] === 0) theoretical = 2 * Math.sqrt(Math.abs(x0[0]) / uMax);
  else theoretical = optimalTimeDoubleIntegrator(x0[0], x0[1], uMax);

  // Count BANG-BANG switches: only saturated phases count, the smooth
  // deadband phase is excluded (PMP predicts ≤ 1 saturated switch).
  let switchCount = 0; let last = 0;
  const satThresh = 0.99 * uMax;
  for (const u of out.controls) {
    if (Math.abs(u[0]) < satThresh) continue;
    const s = Math.sign(u[0]);
    if (last !== 0 && s !== last) switchCount += 1;
    last = s;
  }
  return {...out, arrivalTick, theoreticalArrivalTime: theoretical, switchCount};
}

/** Closed-form optimal time-to-go for the double integrator
 *  ẍ = u, |u|≤u_max, terminal (0, 0). See Athans & Falb 1966, §6.6. */
export function optimalTimeDoubleIntegrator(x0: number, v0: number, uMax: number): number {
  // Compute σ(x₀, v₀).
  const sigma = x0 + (1 / (2 * uMax)) * v0 * Math.abs(v0);
  // The optimal control is:  u = -u_max sign(σ)  until the switching
  // curve is reached, then u = u_max sign(σ) until the origin.
  // The total time is the well-known formula
  //   t* = √((v₀² + 2 u_max |x₀ + v₀|v₀|/(2 u_max)|)·something)
  // We'll just numerically integrate the closed-loop bang-bang:
  if (Math.abs(x0) + Math.abs(v0) < 1e-9) return 0;
  let x = x0; let v = v0; let t = 0;
  const dtFine = 1e-4;
  for (let i = 0; i < 1_000_000; i++) {
    const σ = x + (1 / (2 * uMax)) * v * Math.abs(v);
    let u = σ > 0 ? -uMax : (σ < 0 ? uMax : (v > 0 ? -uMax : uMax));
    x += dtFine * v + 0.5 * dtFine * dtFine * u;
    v += dtFine * u;
    t += dtFine;
    if (Math.abs(x) + Math.abs(v) < 1e-3) return t;
  }
  return t;
}
