// RUST MIGRATION: target module src/des/general/iterative_learning_control.rs.
// RUST MIGRATION: ILCReferenceKind becomes an enum; params, trial summaries, and results become serde structs.
// RUST MIGRATION: ILC token classes and DESStation subclasses become Token/Station trait impl structs with explicit owned Vec<f64> control/reference programs.
// RUST MIGRATION: runIterativeLearningControl is DES-visible orchestration and should be a PureTransform entry struct; helper reference/RMS functions stay private free functions.
// RUST MIGRATION: Validation and clamp/bounds errors should return Result instead of throwing.
'use strict';

// =============================================================================
// general/iterative-learning-control.ts
//
// Iterative Learning Control (ILC) as an explicit DES station graph.
//
// A repeated-trial controller learns a feedforward control sequence for a
// plant that must track the same reference trajectory on every trial:
//
//   u_{j+1}[k] = sat(u_j[k] + L * e_j[k + 1])
//
// where j is the trial index, k is the time index inside the trial, and
// e_j is the tracking error from the previous trial. The model is deliberately
// expressed as source/station/sink/movable pieces:
//
//   ILCTrialSourceStation -> ILCControllerProgramStation -> ILCPlantTrialStation
//        ^                                                     |
//        |                                                     v
//   ILCLearningUpdateStation <-------------------------- ILCTrialResultToken
//                                          |
//                                          v
//                                  ILCResultSinkStation
//
// The learner is a stationary entity with memory; trial plans, controller
// programs, and trial results are movables flowing through typed channels.
// =============================================================================

import {
  ChannelName,
  DESStation,
  StationGraphSummary,
  Token,
  channelEdge,
  runIterativeDES,
  stationGraph,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

export type ILCReferenceKind = 'sine' | 'step' | 'ramp';

export interface IterativeLearningControlParams {
  trials?: number;
  horizon?: number;
  dt?: number;
  plantRate?: number;
  plantGain?: number;
  learningGain?: number;
  feedbackGain?: number;
  controlMax?: number;
  referenceKind?: ILCReferenceKind;
  referenceAmplitude?: number;
  initialOutput?: number;
}

export interface ILCTrialSummary {
  trial: number;
  rmsError: number;
  maxAbsError: number;
  maxAbsControl: number;
  finalOutput: number;
  finalReference: number;
}

export interface IterativeLearningControlResult {
  referenceTrajectory: number[];
  trialSummaries: ILCTrialSummary[];
  initialRmsError: number;
  finalRmsError: number;
  improvementRatio: number;
  finalOutputTrajectory: number[];
  finalControlSequence: number[];
  finalFeedforwardSequence: number[];
  topology: StationGraphSummary;
}

class ILCTrialPlanToken implements Token {
  constructor(
    readonly trial: number,
    readonly reference: number[],
    readonly feedforward: number[],
  ) {}
}

class ILCControlProgramToken implements Token {
  constructor(
    readonly trial: number,
    readonly reference: number[],
    readonly feedforward: number[],
    readonly feedbackGain: number,
    readonly controlMax: number,
  ) {}
}

class ILCTrialResultToken implements Token {
  constructor(
    readonly trial: number,
    readonly reference: number[],
    readonly feedforward: number[],
    readonly controls: number[],
    readonly output: number[],
    readonly errors: number[],
    readonly rmsError: number,
    readonly maxAbsError: number,
    readonly maxAbsControl: number,
  ) {}
}

class ILCTrialSourceStation extends DESStation {
  static readonly CH_TRIAL: ChannelName = 'trial-plan';
  private emitted = false;

  constructor(
    id: string,
    private readonly reference: number[],
    private readonly horizon: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(
      new ILCTrialPlanToken(0, this.reference.slice(), Array.from({length: this.horizon}, () => 0)),
      ILCTrialSourceStation.CH_TRIAL,
    );
    this.emitted = true;
  }
}

class ILCControllerProgramStation extends DESStation {
  static readonly CH_TRIAL: ChannelName = ILCTrialSourceStation.CH_TRIAL;
  static readonly CH_PROGRAM: ChannelName = 'control-program';

  constructor(
    id: string,
    private readonly feedbackGain: number,
    private readonly controlMax: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(ILCControllerProgramStation.CH_TRIAL) > 0; }

  runTimeStep(): void {
    const trials = this.drain<ILCTrialPlanToken>(ILCControllerProgramStation.CH_TRIAL);
    for (const trial of trials) {
      this.emit(
        new ILCControlProgramToken(
          trial.trial,
          trial.reference.slice(),
          trial.feedforward.slice(),
          this.feedbackGain,
          this.controlMax,
        ),
        ILCControllerProgramStation.CH_PROGRAM,
      );
    }
  }
}

class ILCPlantTrialStation extends DESStation {
  static readonly CH_PROGRAM: ChannelName = ILCControllerProgramStation.CH_PROGRAM;
  static readonly CH_RESULT: ChannelName = 'trial-result';

  constructor(
    id: string,
    private readonly plantRate: number,
    private readonly plantGain: number,
    private readonly dt: number,
    private readonly initialOutput: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(ILCPlantTrialStation.CH_PROGRAM) > 0; }

  runTimeStep(): void {
    const programs = this.drain<ILCControlProgramToken>(ILCPlantTrialStation.CH_PROGRAM);
    for (const program of programs) this.emit(this.runTrial(program), ILCPlantTrialStation.CH_RESULT);
  }

  private runTrial(program: ILCControlProgramToken): ILCTrialResultToken {
    const horizon = program.feedforward.length;
    let y = this.initialOutput;
    const output = [y];
    const controls: number[] = [];
    const errors: number[] = [];

    for (let k = 0; k < horizon; k++) {
      const error = program.reference[k] - y;
      const u = clamp(program.feedforward[k] + program.feedbackGain * error,
                      -program.controlMax, program.controlMax);
      controls.push(u);
      errors.push(error);
      y = y + this.dt * (-this.plantRate * y + this.plantGain * u);
      output.push(y);
    }

    const rmsError = rms(errors);
    const maxAbsError = errors.reduce((acc, e) => Math.max(acc, Math.abs(e)), 0);
    const maxAbsControl = controls.reduce((acc, u) => Math.max(acc, Math.abs(u)), 0);
    return new ILCTrialResultToken(
      program.trial,
      program.reference.slice(),
      program.feedforward.slice(),
      controls,
      output,
      errors,
      rmsError,
      maxAbsError,
      maxAbsControl,
    );
  }
}

class ILCLearningUpdateStation extends DESStation {
  static readonly CH_RESULT: ChannelName = ILCPlantTrialStation.CH_RESULT;
  static readonly CH_TRIAL: ChannelName = ILCTrialSourceStation.CH_TRIAL;

  constructor(
    id: string,
    private readonly maxTrials: number,
    private readonly learningGain: number,
    private readonly controlMax: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(ILCLearningUpdateStation.CH_RESULT) > 0; }

  runTimeStep(): void {
    const results = this.drain<ILCTrialResultToken>(ILCLearningUpdateStation.CH_RESULT);
    for (const result of results) {
      const nextTrial = result.trial + 1;
      if (nextTrial >= this.maxTrials) continue;
      const nextFeedforward = result.feedforward.map((u, k) => {
        const nextError = result.reference[k + 1] - result.output[k + 1];
        return clamp(u + this.learningGain * nextError, -this.controlMax, this.controlMax);
      });
      this.emit(
        new ILCTrialPlanToken(nextTrial, result.reference.slice(), nextFeedforward),
        ILCLearningUpdateStation.CH_TRIAL,
      );
    }
  }
}

class ILCResultSinkStation extends DESStation {
  static readonly CH_RESULT: ChannelName = ILCPlantTrialStation.CH_RESULT;
  readonly results: ILCTrialResultToken[] = [];

  constructor(id: string) { super(id); }

  override hasWork(): boolean { return this.inboxSize(ILCResultSinkStation.CH_RESULT) > 0; }

  runTimeStep(): void {
    this.results.push(...this.drain<ILCTrialResultToken>(ILCResultSinkStation.CH_RESULT));
  }
}

export function runIterativeLearningControl(
  params: IterativeLearningControlParams = {},
): IterativeLearningControlResult {
  const trials = params.trials ?? 30;
  const horizon = params.horizon ?? 80;
  const dt = params.dt ?? 0.1;
  const plantRate = params.plantRate ?? 1.2;
  const plantGain = params.plantGain ?? 1;
  const learningGain = params.learningGain ?? 0.8;
  const feedbackGain = params.feedbackGain ?? 0.8;
  const controlMax = params.controlMax ?? 5;
  const referenceKind = params.referenceKind ?? 'sine';
  const referenceAmplitude = params.referenceAmplitude ?? 1;
  const initialOutput = params.initialOutput ?? 0;

  const cls = 'runIterativeLearningControl';
  Preconditions.integerInRange(cls, 'trials', trials, 1, 1e6);
  Preconditions.integerInRange(cls, 'horizon', horizon, 2, 1e6);
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.positive(cls, 'plantRate', plantRate);
  Preconditions.positive(cls, 'plantGain', plantGain);
  Preconditions.inRange(cls, 'learningGain', learningGain, 0, 2);
  Preconditions.nonNegative(cls, 'feedbackGain', feedbackGain);
  Preconditions.positive(cls, 'controlMax', controlMax);
  Preconditions.nonNegative(cls, 'referenceAmplitude', referenceAmplitude);
  Preconditions.finite(cls, 'initialOutput', initialOutput);
  Preconditions.check(
    cls,
    'referenceKind',
    'be one of sine, step, ramp',
    referenceKind === 'sine' || referenceKind === 'step' || referenceKind === 'ramp',
    referenceKind,
  );

  const reference = buildReference(referenceKind, horizon, referenceAmplitude);
  const source = new ILCTrialSourceStation('ilc-trial-source', reference, horizon);
  const controller = new ILCControllerProgramStation('ilc-controller-program-station', feedbackGain, controlMax);
  const plant = new ILCPlantTrialStation('ilc-plant-trial-station', plantRate, plantGain, dt, initialOutput);
  const learner = new ILCLearningUpdateStation('ilc-learning-update-station', trials, learningGain, controlMax);
  const sink = new ILCResultSinkStation('ilc-result-sink');

  source.pipe(controller, ILCTrialSourceStation.CH_TRIAL, ILCControllerProgramStation.CH_TRIAL);
  controller.pipe(plant, ILCControllerProgramStation.CH_PROGRAM, ILCPlantTrialStation.CH_PROGRAM);
  plant.pipe(learner, ILCPlantTrialStation.CH_RESULT, ILCLearningUpdateStation.CH_RESULT);
  plant.pipe(sink, ILCPlantTrialStation.CH_RESULT, ILCResultSinkStation.CH_RESULT);
  learner.pipe(controller, ILCLearningUpdateStation.CH_TRIAL, ILCControllerProgramStation.CH_TRIAL);

  runIterativeDES([source, controller, plant, learner, sink], {
    shuffle: false,
    maxTicks: trials + 5,
    runValidators: false,
  });

  if (sink.results.length !== trials) {
    throw new Error(`iterative-learning-control produced ${sink.results.length} trials, expected ${trials}`);
  }

  const first = sink.results[0];
  const last = sink.results[sink.results.length - 1];
  return {
    referenceTrajectory: reference.slice(),
    trialSummaries: sink.results.map(toSummary),
    initialRmsError: first.rmsError,
    finalRmsError: last.rmsError,
    improvementRatio: last.rmsError / Math.max(first.rmsError, 1e-12),
    finalOutputTrajectory: last.output.slice(),
    finalControlSequence: last.controls.slice(),
    finalFeedforwardSequence: last.feedforward.slice(),
    topology: stationGraph([source, controller, plant, learner, sink], [
      'ILCTrialPlanToken',
      'ILCControlProgramToken',
      'ILCTrialResultToken',
    ], [
      channelEdge(source, ILCTrialSourceStation.CH_TRIAL, controller, ILCControllerProgramStation.CH_TRIAL),
      channelEdge(controller, ILCControllerProgramStation.CH_PROGRAM, plant, ILCPlantTrialStation.CH_PROGRAM),
      channelEdge(plant, ILCPlantTrialStation.CH_RESULT, learner, ILCLearningUpdateStation.CH_RESULT),
      channelEdge(plant, ILCPlantTrialStation.CH_RESULT, sink, ILCResultSinkStation.CH_RESULT),
      channelEdge(learner, ILCLearningUpdateStation.CH_TRIAL, controller, ILCControllerProgramStation.CH_TRIAL),
    ]),
  };
}

function toSummary(result: ILCTrialResultToken): ILCTrialSummary {
  return {
    trial: result.trial,
    rmsError: result.rmsError,
    maxAbsError: result.maxAbsError,
    maxAbsControl: result.maxAbsControl,
    finalOutput: result.output[result.output.length - 1],
    finalReference: result.reference[result.reference.length - 1],
  };
}

function buildReference(kind: ILCReferenceKind, horizon: number, amplitude: number): number[] {
  const denom = Math.max(1, horizon - 1);
  return Array.from({length: horizon + 1}, (_, k) => {
    if (kind === 'step') return k < Math.floor(0.15 * horizon) ? 0 : amplitude;
    if (kind === 'ramp') return amplitude * k / horizon;
    const phase = 2 * Math.PI * k / denom;
    return amplitude * (Math.sin(phase) + 0.4 * Math.sin(2 * phase));
  });
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function rms(xs: readonly number[]): number {
  return Math.sqrt(xs.reduce((acc, x) => acc + x * x, 0) / Math.max(1, xs.length));
}
