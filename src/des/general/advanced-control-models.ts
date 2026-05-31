// RUST MIGRATION: target module src/des/general/advanced_control_models.rs.
// RUST MIGRATION: HInfinityRobustControlParams/Result and PursuitEvasionGameParams/Result become serde structs; advancedControlChannels can be associated consts.
// RUST MIGRATION: ScalarRobustPlant, PursuitEvasionPlant, controllers, and disturbance policies become structs implementing closed-loop station/policy traits instead of TS inheritance.
// RUST MIGRATION: runHInfinityRobustControl and runPursuitEvasionGame should be PureTransform entry structs because they assemble DES-visible closed-loop graphs; return Result for validation.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/advanced-control-models.rs  (module des::general::advanced_control_models)
// 1:1 file move. H-infinity robust control + pursuit-evasion game models as station graphs.
//
// Declarations → Rust:
//   interface *Params / *Result       -> structs (Default; optionals -> Option<T>)
//   class ScalarRobustPlant / LinearRobustController / WorstCaseScalarDisturbance
//   class PursuitEvasionPlant / PurePursuit/EvasionController
//                                      -> structs `impl` the adversarial-control station traits
//                                         (ClosedLoopPlantStation/Feedback/DisturbancePolicyStation are bases -> traits)
//   fn runHInfinityRobustControl / runPursuitEvasionGame -> free fns (or PureTransform per param->result)
//   const advancedControlChannels      -> assoc consts / a channels struct of channel ids
//
// Conversion notes (file-specific):
//   - Free helpers `clamp`, `norm2` are vanilla numeric algorithms -> assoc fns or
//     `VecOps` in `shared/linalg.rs`; not stations.
//   - Deterministic (no Math.random/Date.now); state vectors are small `[f64; N]`/`Vec<f64>`.
//   - `.slice()` clones on trace rows -> `.clone()` of `Vec`s when copying history.
// =============================================================================

// =============================================================================
// Additional decision/control models built as station graphs.
// =============================================================================

import {
  CH_CONTROL,
  CH_DISTURBANCE,
  CH_OBSERVATION,
  ClosedLoopGameTraceRow,
  ClosedLoopPlantStation,
  DisturbancePolicyStation,
  FeedbackPolicyStation,
  StateObservationToken,
  runClosedLoopGame,
} from './des-base/adversarial-control';
import {Preconditions} from './des-base/preconditions';
import {StationGraphTopology, stationGraphTopology} from './des-base/model-topology';

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function norm2(x: readonly number[]): number {
  return Math.sqrt(x.reduce((acc, v) => acc + v * v, 0));
}

// -----------------------------------------------------------------------------
// H-infinity-style bounded-disturbance robust control
// -----------------------------------------------------------------------------

export interface HInfinityRobustControlParams {
  x0?: number;
  a?: number;
  b?: number;
  gain?: number;
  disturbanceMax?: number;
  controlMax?: number;
  gamma?: number;
  dt?: number;
  numSteps?: number;
}

export interface HInfinityRobustControlResult {
  trace: ClosedLoopGameTraceRow[];
  finalState: number;
  peakAbsState: number;
  l2GainEstimate: number;
  gamma: number;
  boundedByGamma: boolean;
  topology: StationGraphTopology;
}

class ScalarRobustPlant extends ClosedLoopPlantStation {
  constructor(
    private readonly a: number,
    private readonly b: number,
    params: Required<HInfinityRobustControlParams>,
  ) {
    super('hinfinity-plant', {
      x0: [params.x0],
      dt: params.dt,
      numSteps: params.numSteps,
      controlDim: 1,
      disturbanceDim: 1,
    });
  }

  protected dynamics(state: readonly number[], control: readonly number[], disturbance: readonly number[], dt: number): number[] {
    const xdot = this.a * state[0] + this.b * control[0] + disturbance[0];
    return [state[0] + dt * xdot];
  }

  protected stageCost(
    _state: readonly number[],
    control: readonly number[],
    disturbance: readonly number[],
    nextState: readonly number[],
  ): number {
    return nextState[0] * nextState[0] + 0.02 * control[0] * control[0] - 0.02 * disturbance[0] * disturbance[0];
  }
}

class LinearRobustController extends FeedbackPolicyStation {
  constructor(private readonly gain: number, private readonly controlMax: number) {
    super('hinfinity-state-feedback-controller', 1);
  }

  protected policy(observation: StateObservationToken): number[] {
    return [clamp(-this.gain * observation.state[0], -this.controlMax, this.controlMax)];
  }
}

class WorstCaseScalarDisturbance extends DisturbancePolicyStation {
  constructor(private readonly disturbanceMax: number) {
    super('worst-case-disturbance-station', 1);
  }

  protected policy(observation: StateObservationToken): number[] {
    const x = observation.state[0];
    return [x >= 0 ? this.disturbanceMax : -this.disturbanceMax];
  }
}

export function runHInfinityRobustControl(params: HInfinityRobustControlParams = {}): HInfinityRobustControlResult {
  const defaults: Required<HInfinityRobustControlParams> = {
    x0: params.x0 ?? 2,
    a: params.a ?? 0.25,
    b: params.b ?? 1,
    gain: params.gain ?? 3.2,
    disturbanceMax: params.disturbanceMax ?? 0.45,
    controlMax: params.controlMax ?? 5,
    gamma: params.gamma ?? 2.5,
    dt: params.dt ?? 0.03,
    numSteps: params.numSteps ?? 260,
  };
  Preconditions.finite('runHInfinityRobustControl', 'x0', defaults.x0);
  Preconditions.finite('runHInfinityRobustControl', 'a', defaults.a);
  Preconditions.finite('runHInfinityRobustControl', 'b', defaults.b);
  Preconditions.positive('runHInfinityRobustControl', 'gain', defaults.gain);
  Preconditions.nonNegative('runHInfinityRobustControl', 'disturbanceMax', defaults.disturbanceMax);
  Preconditions.positive('runHInfinityRobustControl', 'controlMax', defaults.controlMax);
  Preconditions.positive('runHInfinityRobustControl', 'gamma', defaults.gamma);

  const plant = new ScalarRobustPlant(defaults.a, defaults.b, defaults);
  const controller = new LinearRobustController(defaults.gain, defaults.controlMax);
  const adversary = new WorstCaseScalarDisturbance(defaults.disturbanceMax);
  runClosedLoopGame(plant, controller, adversary);

  const stateEnergy = plant.trace.reduce((acc, row) => acc + row.state[0] * row.state[0], 0);
  const disturbanceEnergy = plant.trace.reduce((acc, row) => acc + row.disturbance[0] * row.disturbance[0], 0);
  const l2GainEstimate = Math.sqrt(stateEnergy / Math.max(1e-12, disturbanceEnergy));
  const peakAbsState = plant.stateHistory.reduce((acc, state) => Math.max(acc, Math.abs(state[0])), 0);
  const finalState = plant.getState()[0];
  return {
    trace: plant.trace.map(row => ({...row, state: row.state.slice(), control: row.control.slice(), disturbance: row.disturbance.slice()})),
    finalState,
    peakAbsState,
    l2GainEstimate,
    gamma: defaults.gamma,
    boundedByGamma: l2GainEstimate <= defaults.gamma,
    topology: stationGraphTopology(
      ['hinfinity-plant', 'hinfinity-state-feedback-controller', 'worst-case-disturbance-station'],
      ['StateObservationToken', 'ControlMoveToken', 'DisturbanceMoveToken'],
    ),
  };
}

// -----------------------------------------------------------------------------
// Differential game: pursuit/evasion with two competing controllers
// -----------------------------------------------------------------------------

export interface PursuitEvasionGameParams {
  pursuer?: [number, number];
  evader?: [number, number];
  pursuerSpeed?: number;
  evaderSpeed?: number;
  captureRadius?: number;
  dt?: number;
  numSteps?: number;
}

export interface PursuitEvasionGameResult {
  trace: ClosedLoopGameTraceRow[];
  distanceHistory: number[];
  captureTick: number | null;
  finalDistance: number;
  topology: StationGraphTopology;
}

class PursuitEvasionPlant extends ClosedLoopPlantStation {
  private captureTick: number | null = null;

  constructor(private readonly captureRadius: number, params: Required<PursuitEvasionGameParams>) {
    super('pursuit-evasion-plant', {
      x0: [params.pursuer[0], params.pursuer[1], params.evader[0], params.evader[1]],
      dt: params.dt,
      numSteps: params.numSteps,
      controlDim: 2,
      disturbanceDim: 2,
    });
  }

  protected dynamics(state: readonly number[], control: readonly number[], disturbance: readonly number[], dt: number): number[] {
    return [
      state[0] + dt * control[0],
      state[1] + dt * control[1],
      state[2] + dt * disturbance[0],
      state[3] + dt * disturbance[1],
    ];
  }

  protected stageCost(
    _state: readonly number[],
    _control: readonly number[],
    _disturbance: readonly number[],
    nextState: readonly number[],
  ): number {
    return this.distance(nextState);
  }

  protected terminal(state: readonly number[], tick: number): boolean {
    if (this.distance(state) <= this.captureRadius) {
      if (this.captureTick === null) this.captureTick = tick;
      return true;
    }
    return false;
  }

  getCaptureTick(): number | null { return this.captureTick; }

  private distance(state: readonly number[]): number {
    return Math.hypot(state[2] - state[0], state[3] - state[1]);
  }
}

class PurePursuitController extends FeedbackPolicyStation {
  constructor(private readonly speed: number) {
    super('pure-pursuit-controller', 2);
  }

  protected policy(observation: StateObservationToken): number[] {
    const dx = observation.state[2] - observation.state[0];
    const dy = observation.state[3] - observation.state[1];
    const n = Math.max(1e-12, Math.hypot(dx, dy));
    return [this.speed * dx / n, this.speed * dy / n];
  }
}

class PureEvasionController extends DisturbancePolicyStation {
  constructor(private readonly speed: number) {
    super('pure-evasion-controller', 2);
  }

  protected policy(observation: StateObservationToken): number[] {
    const dx = observation.state[2] - observation.state[0];
    const dy = observation.state[3] - observation.state[1];
    const n = Math.max(1e-12, Math.hypot(dx, dy));
    return [this.speed * dx / n, this.speed * dy / n];
  }
}

export function runPursuitEvasionGame(params: PursuitEvasionGameParams = {}): PursuitEvasionGameResult {
  const defaults: Required<PursuitEvasionGameParams> = {
    pursuer: params.pursuer ?? [0, 0],
    evader: params.evader ?? [6, 2],
    pursuerSpeed: params.pursuerSpeed ?? 1.25,
    evaderSpeed: params.evaderSpeed ?? 0.6,
    captureRadius: params.captureRadius ?? 0.25,
    dt: params.dt ?? 0.1,
    numSteps: params.numSteps ?? 120,
  };
  Preconditions.lengthEq('runPursuitEvasionGame', 'pursuer', defaults.pursuer, 2);
  Preconditions.lengthEq('runPursuitEvasionGame', 'evader', defaults.evader, 2);
  Preconditions.allFinite('runPursuitEvasionGame', 'pursuer', defaults.pursuer);
  Preconditions.allFinite('runPursuitEvasionGame', 'evader', defaults.evader);
  Preconditions.positive('runPursuitEvasionGame', 'pursuerSpeed', defaults.pursuerSpeed);
  Preconditions.nonNegative('runPursuitEvasionGame', 'evaderSpeed', defaults.evaderSpeed);
  Preconditions.positive('runPursuitEvasionGame', 'captureRadius', defaults.captureRadius);

  const plant = new PursuitEvasionPlant(defaults.captureRadius, defaults);
  const pursuer = new PurePursuitController(defaults.pursuerSpeed);
  const evader = new PureEvasionController(defaults.evaderSpeed);
  runClosedLoopGame(plant, pursuer, evader);

  const distanceHistory = plant.stateHistory.map(state => norm2([state[2] - state[0], state[3] - state[1]]));
  const finalDistance = distanceHistory[distanceHistory.length - 1];
  return {
    trace: plant.trace.map(row => ({...row, state: row.state.slice(), control: row.control.slice(), disturbance: row.disturbance.slice()})),
    distanceHistory,
    captureTick: plant.getCaptureTick(),
    finalDistance,
    topology: stationGraphTopology(
      ['pursuit-evasion-plant', 'pure-pursuit-controller', 'pure-evasion-controller'],
      ['StateObservationToken', 'ControlMoveToken', 'DisturbanceMoveToken'],
    ),
  };
}

export const advancedControlChannels = {
  observation: CH_OBSERVATION,
  control: CH_CONTROL,
  disturbance: CH_DISTURBANCE,
};
