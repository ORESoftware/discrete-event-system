'use strict';

// =============================================================================
// control-systems/wind-mppt.ts — Maximum Power Point Tracking (MPPT) for a
// variable-speed wind-energy conversion system (WECS) with a permanent-magnet
// synchronous generator (PMSG).
//
// REFERENCE
// ─────────
//   K. K. Pandey & A. N. Tiwari, "Maximum Power Point Tracking of Wind Energy
//   Conversion System with Permanent Magnet Synchronous Generator", IJERT
//   Vol. 1 Issue 5, July 2012 (IJERTV1IS5198).
//
// PHYSICS
// ───────
//   Wind power through swept area A = πR²:
//       P_wind = ½·ρ·A·V³
//   Captured mechanical power via the power coefficient C_p(λ, β):
//       P_mech = ½·ρ·A·C_p(λ, β)·V³
//   Tip-speed ratio and (Heier) coefficient model:
//       λ  = ω_r·R / V
//       1/λ_i = 1/(λ + 0.08β) − 0.035/(β³ + 1)
//       C_p = 0.5176·(116/λ_i − 0.4β − 5)·exp(−21/λ_i) + 0.0068·λ
//   Rotor mechanical ODE (single state ω_r):
//       J·dω_r/dt = T_aero − T_gen − B·ω_r,   T_aero = P_mech / ω_r
//
// MPPT CONTROL (paper §3)
// ───────────────────────
//   The captured power is a cubic function of generator speed at the optimal
//   tip-speed ratio λ*, so MPPT tracks the optimal speed/torque curve. Two
//   controllers are provided:
//     • OptimalTorqueMpptController  — T_gen = K_opt·ω_r²  (drives λ → λ*)
//         K_opt = ½·ρ·π·R⁵·C_p,max / λ*³
//     • SpeedPiMpptController        — the paper's loop: derive a reference
//         speed ω* from the measured power via the power–speed curve, then a
//         PI regulator on (ω* − ω_r) produces the reference torque.
//
// DES STRUCTURE
// ─────────────
//   Lightweight `DESStation` graph driven by `runIterativeDES`:
//     WindTurbinePlantStation (self-clocking ODE plant, runTimeStep + RK4)
//          ── state ──▶  <controller>  ── torque ──▶ back to plant
//          ── state ──▶  WindMpptSinkStation
//   The controllers are zero-backlog/queue-backed transforms
//   (PureTransformEntity / MemoryTransformEntity); no free functions are used.
// =============================================================================

import {ChannelName, DESStation, Token} from '../des-base/station';
import {MemoryTransformEntity, PureTransformEntity, TransformContext} from '../des-base/transform-entity';
import {Preconditions} from '../des-base/preconditions';
import {OdeSystem, RungeKutta4Integrator} from './numerical-solvers';

// -----------------------------------------------------------------------------
// CHANNELS
// -----------------------------------------------------------------------------

export class WindMpptChannels {
  static readonly STATE: ChannelName = 'turbine-state';
  static readonly TORQUE: ChannelName = 'gen-torque';
}

// -----------------------------------------------------------------------------
// TOKENS
// -----------------------------------------------------------------------------

/** Snapshot of the turbine emitted once per discrete tick. */
export class TurbineStateToken implements Token {
  constructor(
    readonly tick: number,
    readonly time: number,
    /** rotor speed ω_r [rad/s] */
    readonly omega: number,
    /** wind speed V [m/s] used this step */
    readonly windSpeed: number,
    /** tip-speed ratio λ */
    readonly lambda: number,
    /** power coefficient C_p */
    readonly cp: number,
    /** captured mechanical power P_mech [W] */
    readonly mechPower: number,
    /** generator (load) torque applied this step [N·m] */
    readonly genTorque: number,
  ) {}
}

/** Generator electromagnetic torque command produced by the MPPT controller. */
export class GenTorqueToken implements Token {
  constructor(readonly tick: number, readonly torque: number) {}
}

// -----------------------------------------------------------------------------
// AERODYNAMICS
// -----------------------------------------------------------------------------

export interface WindTurbineAeroOpts {
  /** air density ρ [kg/m³]. Default 1.225. */
  airDensity?: number;
  /** blade radius R [m]. */
  bladeRadius: number;
  /** blade pitch angle β [deg]. Default 0. */
  pitchDeg?: number;
}

/** Aerodynamic model: C_p(λ, β), captured power, aero torque, and the optimal
 *  operating point used by the MPPT controllers. */
export class WindTurbineAerodynamics {
  readonly airDensity: number;
  readonly bladeRadius: number;
  readonly pitchDeg: number;
  private optLambda: number | null = null;
  private optCp: number | null = null;

  constructor(opts: WindTurbineAeroOpts) {
    this.airDensity = opts.airDensity ?? 1.225;
    this.bladeRadius = opts.bladeRadius;
    this.pitchDeg = opts.pitchDeg ?? 0;
    Preconditions.positive('WindTurbineAerodynamics', 'airDensity', this.airDensity);
    Preconditions.positive('WindTurbineAerodynamics', 'bladeRadius', this.bladeRadius);
    Preconditions.nonNegative('WindTurbineAerodynamics', 'pitchDeg', this.pitchDeg);
  }

  /** Swept area A = πR². */
  sweptArea(): number {
    return Math.PI * this.bladeRadius * this.bladeRadius;
  }

  /** Tip-speed ratio λ = ωR/V (guards V≈0). */
  tipSpeedRatio(omega: number, windSpeed: number): number {
    const v = Math.max(windSpeed, 1e-6);
    return (omega * this.bladeRadius) / v;
  }

  /** Heier C_p(λ, β) model. Clamped at 0 (no negative capture). */
  powerCoefficient(lambda: number, pitchDeg = this.pitchDeg): number {
    if (lambda <= 0) return 0;
    const beta = pitchDeg;
    const invLi = 1 / (lambda + 0.08 * beta) - 0.035 / (beta * beta * beta + 1);
    const li = 1 / invLi;
    const cp = 0.5176 * (116 * invLi - 0.4 * beta - 5) * Math.exp(-21 * invLi) + 0.0068 * lambda;
    return cp > 0 ? cp : 0;
  }

  /** Captured mechanical power P_mech = ½ρA·C_p·V³. */
  mechanicalPower(windSpeed: number, omega: number): number {
    const lambda = this.tipSpeedRatio(omega, windSpeed);
    const cp = this.powerCoefficient(lambda);
    return 0.5 * this.airDensity * this.sweptArea() * cp * windSpeed * windSpeed * windSpeed;
  }

  /** Aerodynamic torque T_aero = P_mech / ω (guards ω≈0 with the C_p/λ form). */
  aeroTorque(windSpeed: number, omega: number): number {
    const lambda = this.tipSpeedRatio(omega, windSpeed);
    const cp = this.powerCoefficient(lambda);
    const power = 0.5 * this.airDensity * this.sweptArea() * cp * windSpeed * windSpeed * windSpeed;
    if (omega > 1e-3) return power / omega;
    // ω → 0 limit: T = ½ρA·R·(C_p/λ)·V²  (finite startup torque).
    if (lambda <= 1e-9) return 0;
    return 0.5 * this.airDensity * this.sweptArea() * this.bladeRadius * (cp / lambda) * windSpeed * windSpeed;
  }

  /** Optimal tip-speed ratio λ* maximising C_p (scanned + cached). */
  optimalTipSpeedRatio(): number {
    this.computeOptimum();
    return this.optLambda as number;
  }

  /** Maximum power coefficient C_p,max (cached). */
  maxPowerCoefficient(): number {
    this.computeOptimum();
    return this.optCp as number;
  }

  /** Optimal-torque gain K_opt with T_opt = K_opt·ω².
   *  K_opt = ½·ρ·π·R⁵·C_p,max / λ*³. */
  optimalTorqueGain(): number {
    const lambdaStar = this.optimalTipSpeedRatio();
    const cpMax = this.maxPowerCoefficient();
    const r5 = Math.pow(this.bladeRadius, 5);
    return (0.5 * this.airDensity * Math.PI * r5 * cpMax) / Math.pow(lambdaStar, 3);
  }

  /** Power gain K_p with P_opt = K_p·ω³ at λ* (used to invert the power–speed
   *  curve, P → ω*). */
  optimalPowerGain(): number {
    const lambdaStar = this.optimalTipSpeedRatio();
    const cpMax = this.maxPowerCoefficient();
    const r5 = Math.pow(this.bladeRadius, 5);
    return (0.5 * this.airDensity * Math.PI * r5 * cpMax) / Math.pow(lambdaStar, 3);
  }

  private computeOptimum(): void {
    if (this.optLambda !== null) return;
    let bestLambda = 0;
    let bestCp = -Infinity;
    for (let lambda = 0.1; lambda <= 20; lambda += 0.001) {
      const cp = this.powerCoefficient(lambda);
      if (cp > bestCp) {
        bestCp = cp;
        bestLambda = lambda;
      }
    }
    this.optLambda = bestLambda;
    this.optCp = bestCp;
  }
}

// -----------------------------------------------------------------------------
// WIND PROFILE + ROTOR DYNAMICS (ODE)
// -----------------------------------------------------------------------------

export interface WindProfileSegment {
  /** start time of this segment [s] */
  fromTime: number;
  /** wind speed during this segment [m/s] */
  speed: number;
}

/** Piecewise-constant wind speed schedule, V(t). */
export class WindProfile {
  private readonly segments: WindProfileSegment[];

  constructor(segments: readonly WindProfileSegment[]) {
    if (segments.length === 0) throw new Error('WindProfile: at least one segment required');
    this.segments = segments.slice().sort((a, b) => a.fromTime - b.fromTime);
    for (const s of this.segments) Preconditions.nonNegative('WindProfile', 'speed', s.speed);
  }

  /** Wind speed at time t. */
  speedAt(time: number): number {
    let speed = this.segments[0].speed;
    for (const s of this.segments) {
      if (time + 1e-12 >= s.fromTime) speed = s.speed;
      else break;
    }
    return speed;
  }
}

/** Single-state rotor ODE  J·dω/dt = T_aero(V, ω) − T_gen − B·ω.
 *  The wind speed and generator torque are MUTABLE conditions set by the
 *  plant station before each numerical step. */
export class RotorDynamics implements OdeSystem {
  private windSpeed = 0;
  private genTorque = 0;

  constructor(
    readonly aero: WindTurbineAerodynamics,
    /** rotor inertia J [kg·m²] */
    readonly inertia: number,
    /** viscous friction B [N·m·s] */
    readonly friction: number,
  ) {
    Preconditions.positive('RotorDynamics', 'inertia', inertia);
    Preconditions.nonNegative('RotorDynamics', 'friction', friction);
  }

  dimension(): number {
    return 1;
  }

  /** Set the operating conditions for the upcoming numerical step. */
  setConditions(windSpeed: number, genTorque: number): void {
    this.windSpeed = windSpeed;
    this.genTorque = genTorque;
  }

  derivative(_t: number, state: readonly number[]): number[] {
    const omega = Math.max(state[0], 0);
    const tAero = this.aero.aeroTorque(this.windSpeed, omega);
    const domega = (tAero - this.genTorque - this.friction * omega) / this.inertia;
    return [domega];
  }
}

// -----------------------------------------------------------------------------
// PLANT STATION (self-clocking ODE integrator)
// -----------------------------------------------------------------------------

export interface WindTurbinePlantOpts {
  aero: WindTurbineAerodynamics;
  windProfile: WindProfile;
  /** rotor inertia J [kg·m²] */
  inertia: number;
  /** viscous friction B [N·m·s] */
  friction: number;
  /** integration / sample step dt [s] */
  dt: number;
  /** number of discrete ticks to simulate */
  steps: number;
  /** initial rotor speed ω₀ [rad/s] */
  initialOmega: number;
}

/** The turbine PLANT. Self-clocks for `steps` ticks. Each tick it drains the
 *  latest generator-torque command, advances the rotor ODE one RK4 step, and
 *  emits a `TurbineStateToken`. */
export class WindTurbinePlantStation extends DESStation {
  private readonly dynamics: RotorDynamics;
  private readonly integrator = new RungeKutta4Integrator();
  private readonly windProfile: WindProfile;
  private readonly aero: WindTurbineAerodynamics;
  private readonly dt: number;
  private readonly steps: number;
  private omega: number;
  private tick = 0;
  private lastGenTorque = 0;
  readonly trace: TurbineStateToken[] = [];

  constructor(id: string, opts: WindTurbinePlantOpts) {
    super(id);
    Preconditions.positive('WindTurbinePlantStation', 'dt', opts.dt);
    Preconditions.integerInRange('WindTurbinePlantStation', 'steps', opts.steps, 1, 10_000_000);
    Preconditions.nonNegative('WindTurbinePlantStation', 'initialOmega', opts.initialOmega);
    this.aero = opts.aero;
    this.windProfile = opts.windProfile;
    this.dynamics = new RotorDynamics(opts.aero, opts.inertia, opts.friction);
    this.dt = opts.dt;
    this.steps = opts.steps;
    this.omega = opts.initialOmega;
  }

  override hasWork(): boolean {
    return this.tick < this.steps;
  }

  override assertPreconditions(): void {
    Preconditions.positive('WindTurbinePlantStation', 'dt', this.dt);
    Preconditions.finite('WindTurbinePlantStation', 'initialOmega', this.omega);
  }

  runTimeStep(): void {
    if (this.tick >= this.steps) return;
    // 1. Drain torque commands — keep the most recent (last write wins).
    for (const cmd of this.drain<GenTorqueToken>(WindMpptChannels.TORQUE)) {
      this.lastGenTorque = cmd.torque;
    }
    // 2. Advance the rotor ODE one RK4 step under the current conditions.
    const time = this.tick * this.dt;
    const windSpeed = this.windProfile.speedAt(time);
    this.dynamics.setConditions(windSpeed, this.lastGenTorque);
    const next = this.integrator.step(this.dynamics, time, [this.omega], this.dt);
    this.omega = Math.max(next[0], 0);
    // 3. Emit the measured turbine state.
    const lambda = this.aero.tipSpeedRatio(this.omega, windSpeed);
    const cp = this.aero.powerCoefficient(lambda);
    const mechPower = this.aero.mechanicalPower(windSpeed, this.omega);
    const token = new TurbineStateToken(
      this.tick, (this.tick + 1) * this.dt, this.omega, windSpeed, lambda, cp, mechPower, this.lastGenTorque,
    );
    this.trace.push(token);
    this.emit(token, WindMpptChannels.STATE);
    this.tick += 1;
  }

  getOmega(): number {
    return this.omega;
  }

  getTrace(): readonly TurbineStateToken[] {
    return this.trace;
  }
}

// -----------------------------------------------------------------------------
// MPPT CONTROLLERS
// -----------------------------------------------------------------------------

/** Optimal-torque MPPT: T_gen = K_opt·ω². A memoryless control law, so it is a
 *  zero-backlog `PureTransformEntity` from turbine state → torque command. */
export class OptimalTorqueMpptController extends PureTransformEntity<TurbineStateToken, GenTorqueToken> {
  private readonly kOpt: number;

  constructor(id: string, aero: WindTurbineAerodynamics) {
    super(id, {inputChannels: WindMpptChannels.STATE, outputChannel: WindMpptChannels.TORQUE});
    this.kOpt = aero.optimalTorqueGain();
  }

  transform(token: TurbineStateToken): GenTorqueToken {
    const torque = this.kOpt * token.omega * token.omega;
    return new GenTorqueToken(token.tick, torque);
  }

  getOptimalTorqueGain(): number {
    return this.kOpt;
  }
}

export interface SpeedPiMpptOpts {
  /** proportional gain on speed error [N·m·s/rad] */
  kp: number;
  /** integral gain on speed error [N·m/rad] */
  ki: number;
  /** sample step dt [s] (for the integrator accumulation) */
  dt: number;
  /** torque saturation [N·m]. Default no clamp. */
  maxTorque?: number;
}

/** Speed-loop MPPT (paper §3): track the optimal speed reference
 *  ω* = λ*·V / R that places the turbine on the maximum-C_p ridge, and let a
 *  PI regulator on the speed error e = ω − ω* set the generator braking
 *  torque. When ω < ω* the (negative) command is clamped to zero so the aero
 *  torque accelerates the rotor; when ω > ω* the generator brakes it. The PI
 *  integral removes the steady-state offset the friction term would otherwise
 *  leave. Queue-backed `MemoryTransformEntity` whose `previous` carries the
 *  integral accumulator. */
export class SpeedPiMpptController extends MemoryTransformEntity<TurbineStateToken, GenTorqueToken, number> {
  private readonly kp: number;
  private readonly ki: number;
  private readonly dt: number;
  private readonly maxTorque: number;
  private readonly lambdaStar: number;
  private readonly bladeRadius: number;

  constructor(id: string, aero: WindTurbineAerodynamics, opts: SpeedPiMpptOpts) {
    super(id, 0, {inputChannels: WindMpptChannels.STATE, outputChannel: WindMpptChannels.TORQUE});
    Preconditions.nonNegative('SpeedPiMpptController', 'kp', opts.kp);
    Preconditions.nonNegative('SpeedPiMpptController', 'ki', opts.ki);
    Preconditions.positive('SpeedPiMpptController', 'dt', opts.dt);
    this.kp = opts.kp;
    this.ki = opts.ki;
    this.dt = opts.dt;
    this.maxTorque = opts.maxTorque ?? Infinity;
    this.lambdaStar = aero.optimalTipSpeedRatio();
    this.bladeRadius = aero.bladeRadius;
  }

  protected transformQueued(token: TurbineStateToken, _ctx: TransformContext<TurbineStateToken, GenTorqueToken>): GenTorqueToken {
    const referenceSpeed = (this.lambdaStar * token.windSpeed) / this.bladeRadius;
    const error = token.omega - referenceSpeed;
    const candidateIntegral = this.previous + error * this.dt;
    let torque = this.kp * error + this.ki * candidateIntegral;
    if (torque < 0) {
      torque = 0;
    } else if (torque > this.maxTorque) {
      torque = this.maxTorque;
    } else {
      // Anti-windup: only accumulate the integral while unsaturated.
      this.previous = candidateIntegral;
    }
    return new GenTorqueToken(token.tick, torque);
  }
}

// -----------------------------------------------------------------------------
// SINK
// -----------------------------------------------------------------------------

/** Collects the turbine-state trajectory for analysis / validation. */
export class WindMpptSinkStation extends DESStation {
  readonly samples: TurbineStateToken[] = [];

  constructor(id: string) {
    super(id);
  }

  override hasWork(): boolean {
    return this.inboxSize(WindMpptChannels.STATE) > 0;
  }

  runTimeStep(): void {
    this.samples.push(...this.drain<TurbineStateToken>(WindMpptChannels.STATE));
  }

  /** Final captured power [W]. */
  finalPower(): number {
    return this.samples.length ? this.samples[this.samples.length - 1].mechPower : 0;
  }

  /** Final tip-speed ratio λ. */
  finalLambda(): number {
    return this.samples.length ? this.samples[this.samples.length - 1].lambda : 0;
  }

  /** Final power coefficient C_p. */
  finalCp(): number {
    return this.samples.length ? this.samples[this.samples.length - 1].cp : 0;
  }
}
