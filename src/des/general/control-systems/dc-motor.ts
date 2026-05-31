// RUST MIGRATION: Target module `src/des/general/control_systems/dc_motor.rs`.
// RUST MIGRATION: Convert motor dynamics, plant stations, load/reference profiles, and PI controllers into structs with `OdeSystem`/controller traits.
// RUST MIGRATION: Use `f64` vectors/matrices for state/current/speed math, inject plant/controller config explicitly, and return `Result` for invalid physical params.
// RUST MIGRATION: Graph-visible controller/evaluator logic should become PureTransform-style structs with a `transform` method.
'use strict';

// =============================================================================
// control-systems/dc-motor.ts — separately-excited / permanent-magnet DC motor
// modelled as a two-state ODE system with explicit BACK-EMF coupling, driven
// and regulated inside the lightweight DES graph.
//
// MODEL
// ─────
//   State x = [i, ω]  (armature current [A], rotor speed [rad/s]).
//   Electrical:   L·di/dt = V − R·i − E,        E = K_e·ω   ← BACK-EMF
//   Mechanical:   J·dω/dt = K_t·i − B·ω − T_L
//
//   The back-EMF E = K_e·ω is the velocity-dependent voltage the spinning
//   rotor induces back into the armature circuit. It is the term that couples
//   the mechanical state ω into the electrical equation, and it is what makes
//   an un-driven motor settle (the faster it spins, the more it opposes the
//   applied voltage, limiting current and hence torque).
//
//   In state-space form with input u = V and output y = ω:
//       A = [[−R/L, −K_e/L],
//            [ K_t/J, −B/J ]]
//       B = [[1/L], [0]]
//       C = [[0, 1]],   D = [[0]]
//   (`stateSpace()` exposes this so the same plant can be fed straight into
//    the observability / controllability evaluator.)
//
// DES STRUCTURE
// ─────────────
//   DcMotorPlantStation (self-clocking ODE plant, runTimeStep + RK4)
//        ── state ──▶  SpeedPiVoltageController  ── voltage ──▶ back to plant
//        ── state ──▶  DcMotorSinkStation
//   The controller is a queue-backed `MemoryTransformEntity` (PI integrator in
//   `previous`); the plant carries the back-EMF in every emitted state token.
// =============================================================================

import {ChannelName, DESStation, Token} from '../des-base/station';
import {MemoryTransformEntity, TransformContext} from '../des-base/transform-entity';
import {Preconditions} from '../des-base/preconditions';
import {Mat} from './linear-algebra';
import {OdeSystem, RungeKutta4Integrator} from './numerical-solvers';

// -----------------------------------------------------------------------------
// CHANNELS
// -----------------------------------------------------------------------------

export class DcMotorChannels {
  static readonly STATE: ChannelName = 'motor-state';
  static readonly VOLTAGE: ChannelName = 'armature-voltage';
}

// -----------------------------------------------------------------------------
// TOKENS
// -----------------------------------------------------------------------------

/** Measured motor state emitted once per discrete tick. */
export class MotorStateToken implements Token {
  constructor(
    readonly tick: number,
    readonly time: number,
    /** armature current i [A] */
    readonly current: number,
    /** rotor speed ω [rad/s] */
    readonly omega: number,
    /** back-EMF E = K_e·ω [V] */
    readonly backEmf: number,
    /** electromagnetic torque K_t·i [N·m] */
    readonly torque: number,
    /** applied armature voltage this step [V] */
    readonly voltage: number,
    /** load torque this step [N·m] */
    readonly loadTorque: number,
  ) {}
}

/** Armature voltage command produced by the controller. */
export class VoltageToken implements Token {
  constructor(readonly tick: number, readonly voltage: number) {}
}

// -----------------------------------------------------------------------------
// MOTOR PARAMETERS + DYNAMICS (ODE)
// -----------------------------------------------------------------------------

export interface DcMotorParams {
  /** armature resistance R [Ω] */
  resistance: number;
  /** armature inductance L [H] */
  inductance: number;
  /** back-EMF constant K_e [V·s/rad] */
  backEmfConstant: number;
  /** torque constant K_t [N·m/A] */
  torqueConstant: number;
  /** rotor inertia J [kg·m²] */
  inertia: number;
  /** viscous friction B [N·m·s] */
  friction: number;
}

/** Two-state DC-motor ODE. The applied voltage and load torque are MUTABLE
 *  conditions set by the plant station before each numerical step. */
export class DcMotorDynamics implements OdeSystem {
  private voltage = 0;
  private loadTorque = 0;

  constructor(readonly params: DcMotorParams) {
    Preconditions.positive('DcMotorDynamics', 'resistance', params.resistance);
    Preconditions.positive('DcMotorDynamics', 'inductance', params.inductance);
    Preconditions.positive('DcMotorDynamics', 'backEmfConstant', params.backEmfConstant);
    Preconditions.positive('DcMotorDynamics', 'torqueConstant', params.torqueConstant);
    Preconditions.positive('DcMotorDynamics', 'inertia', params.inertia);
    Preconditions.nonNegative('DcMotorDynamics', 'friction', params.friction);
  }

  dimension(): number {
    return 2;
  }

  /** Set the inputs for the upcoming numerical step. */
  setInputs(voltage: number, loadTorque: number): void {
    this.voltage = voltage;
    this.loadTorque = loadTorque;
  }

  /** Back-EMF E = K_e·ω. */
  backEmf(omega: number): number {
    return this.params.backEmfConstant * omega;
  }

  /** Electromagnetic torque T_e = K_t·i. */
  electromagneticTorque(current: number): number {
    return this.params.torqueConstant * current;
  }

  derivative(_t: number, state: readonly number[]): number[] {
    const [i, omega] = state;
    const {resistance, inductance, torqueConstant, inertia, friction} = this.params;
    const e = this.backEmf(omega);
    const di = (this.voltage - resistance * i - e) / inductance;
    const domega = (torqueConstant * i - friction * omega - this.loadTorque) / inertia;
    return [di, domega];
  }

  /** Continuous-time state-space matrices (input u = V, output y = ω). */
  stateSpace(): {A: Mat; B: Mat; C: Mat; D: Mat} {
    const {resistance, inductance, backEmfConstant, torqueConstant, inertia, friction} = this.params;
    return {
      A: [
        [-resistance / inductance, -backEmfConstant / inductance],
        [torqueConstant / inertia, -friction / inertia],
      ],
      B: [[1 / inductance], [0]],
      C: [[0, 1]],
      D: [[0]],
    };
  }
}

// -----------------------------------------------------------------------------
// LOAD-TORQUE PROFILE
// -----------------------------------------------------------------------------

export interface LoadSegment {
  fromTime: number;
  torque: number;
}

/** Piecewise-constant load-torque schedule T_L(t). */
export class LoadProfile {
  private readonly segments: LoadSegment[];

  constructor(segments: readonly LoadSegment[]) {
    this.segments = (segments.length ? segments.slice() : [{fromTime: 0, torque: 0}]).sort((a, b) => a.fromTime - b.fromTime);
  }

  torqueAt(time: number): number {
    let t = this.segments[0].torque;
    for (const s of this.segments) {
      if (time + 1e-12 >= s.fromTime) t = s.torque;
      else break;
    }
    return t;
  }
}

// -----------------------------------------------------------------------------
// PLANT STATION (self-clocking ODE integrator)
// -----------------------------------------------------------------------------

export interface DcMotorPlantOpts {
  params: DcMotorParams;
  /** integration / sample step dt [s] */
  dt: number;
  /** number of discrete ticks to simulate */
  steps: number;
  /** initial state [i₀, ω₀] */
  initialState?: readonly number[];
  /** load-torque schedule. Default zero load. */
  load?: LoadProfile;
}

/** The DC-motor PLANT. Self-clocks for `steps` ticks; each tick it drains the
 *  latest armature voltage, advances the 2-state ODE one RK4 step, and emits a
 *  `MotorStateToken` carrying the back-EMF. */
export class DcMotorPlantStation extends DESStation {
  private readonly dynamics: DcMotorDynamics;
  private readonly integrator = new RungeKutta4Integrator();
  private readonly dt: number;
  private readonly steps: number;
  private readonly load: LoadProfile;
  private state: number[];
  private tick = 0;
  private lastVoltage = 0;
  readonly trace: MotorStateToken[] = [];

  constructor(id: string, opts: DcMotorPlantOpts) {
    super(id);
    Preconditions.positive('DcMotorPlantStation', 'dt', opts.dt);
    Preconditions.integerInRange('DcMotorPlantStation', 'steps', opts.steps, 1, 10_000_000);
    this.dynamics = new DcMotorDynamics(opts.params);
    this.dt = opts.dt;
    this.steps = opts.steps;
    this.load = opts.load ?? new LoadProfile([{fromTime: 0, torque: 0}]);
    this.state = (opts.initialState ?? [0, 0]).slice();
    Preconditions.lengthEq('DcMotorPlantStation', 'initialState', this.state, 2);
    Preconditions.allFinite('DcMotorPlantStation', 'initialState', this.state);
  }

  override hasWork(): boolean {
    return this.tick < this.steps;
  }

  override assertPreconditions(): void {
    Preconditions.positive('DcMotorPlantStation', 'dt', this.dt);
    Preconditions.allFinite('DcMotorPlantStation', 'state', this.state);
  }

  /** Override the constant open-loop drive voltage (used when no controller is
   *  wired — the motor is driven by a fixed armature voltage). */
  setOpenLoopVoltage(voltage: number): void {
    this.lastVoltage = voltage;
  }

  runTimeStep(): void {
    if (this.tick >= this.steps) return;
    // 1. Drain voltage commands — last write wins.
    for (const cmd of this.drain<VoltageToken>(DcMotorChannels.VOLTAGE)) {
      this.lastVoltage = cmd.voltage;
    }
    // 2. Advance the 2-state ODE one RK4 step.
    const time = this.tick * this.dt;
    const loadTorque = this.load.torqueAt(time);
    this.dynamics.setInputs(this.lastVoltage, loadTorque);
    this.state = this.integrator.step(this.dynamics, time, this.state, this.dt);
    // 3. Emit the measured state (back-EMF included).
    const [current, omega] = this.state;
    const token = new MotorStateToken(
      this.tick, (this.tick + 1) * this.dt, current, omega,
      this.dynamics.backEmf(omega), this.dynamics.electromagneticTorque(current),
      this.lastVoltage, loadTorque,
    );
    this.trace.push(token);
    this.emit(token, DcMotorChannels.STATE);
    this.tick += 1;
  }

  getState(): readonly number[] {
    return this.state.slice();
  }

  getDynamics(): DcMotorDynamics {
    return this.dynamics;
  }

  getTrace(): readonly MotorStateToken[] {
    return this.trace;
  }
}

// -----------------------------------------------------------------------------
// SPEED PI CONTROLLER
// -----------------------------------------------------------------------------

export interface SpeedReferenceSegment {
  fromTime: number;
  speed: number;
}

export interface SpeedPiVoltageOpts {
  /** proportional gain [V·s/rad] */
  kp: number;
  /** integral gain [V/rad] */
  ki: number;
  /** sample step dt [s] */
  dt: number;
  /** reference-speed schedule ω*(t) [rad/s] */
  reference: readonly SpeedReferenceSegment[];
  /** armature-voltage saturation magnitude [V]. Default ±Infinity. */
  maxVoltage?: number;
}

/** PI speed controller: V = K_p·e + K_i·∫e with e = ω* − ω. The reference
 *  schedule lives inside the controller so it only needs the measured-state
 *  channel; the integral accumulator is the `MemoryTransformEntity.previous`
 *  field (with anti-windup clamping). */
export class SpeedPiVoltageController extends MemoryTransformEntity<MotorStateToken, VoltageToken, number> {
  private readonly kp: number;
  private readonly ki: number;
  private readonly dt: number;
  private readonly maxVoltage: number;
  private readonly reference: SpeedReferenceSegment[];

  constructor(id: string, opts: SpeedPiVoltageOpts) {
    super(id, 0, {inputChannels: DcMotorChannels.STATE, outputChannel: DcMotorChannels.VOLTAGE});
    Preconditions.nonNegative('SpeedPiVoltageController', 'kp', opts.kp);
    Preconditions.nonNegative('SpeedPiVoltageController', 'ki', opts.ki);
    Preconditions.positive('SpeedPiVoltageController', 'dt', opts.dt);
    Preconditions.nonEmpty('SpeedPiVoltageController', 'reference', opts.reference);
    this.kp = opts.kp;
    this.ki = opts.ki;
    this.dt = opts.dt;
    this.maxVoltage = opts.maxVoltage ?? Infinity;
    this.reference = opts.reference.slice().sort((a, b) => a.fromTime - b.fromTime);
  }

  /** Reference speed ω*(t) from the schedule. */
  referenceAt(time: number): number {
    let r = this.reference[0].speed;
    for (const s of this.reference) {
      if (time + 1e-12 >= s.fromTime) r = s.speed;
      else break;
    }
    return r;
  }

  protected transformQueued(token: MotorStateToken, _ctx: TransformContext<MotorStateToken, VoltageToken>): VoltageToken {
    const reference = this.referenceAt(token.time);
    const error = reference - token.omega;
    const candidateIntegral = this.previous + error * this.dt;
    let voltage = this.kp * error + this.ki * candidateIntegral;
    if (voltage > this.maxVoltage) {
      voltage = this.maxVoltage;
    } else if (voltage < -this.maxVoltage) {
      voltage = -this.maxVoltage;
    } else {
      this.previous = candidateIntegral;
    }
    return new VoltageToken(token.tick, voltage);
  }
}

// -----------------------------------------------------------------------------
// SINK
// -----------------------------------------------------------------------------

/** Collects the motor-state trajectory for analysis / validation. */
export class DcMotorSinkStation extends DESStation {
  readonly samples: MotorStateToken[] = [];

  constructor(id: string) {
    super(id);
  }

  override hasWork(): boolean {
    return this.inboxSize(DcMotorChannels.STATE) > 0;
  }

  runTimeStep(): void {
    this.samples.push(...this.drain<MotorStateToken>(DcMotorChannels.STATE));
  }

  finalState(): MotorStateToken | undefined {
    return this.samples[this.samples.length - 1];
  }

  finalOmega(): number {
    return this.finalState()?.omega ?? 0;
  }

  finalBackEmf(): number {
    return this.finalState()?.backEmf ?? 0;
  }
}
