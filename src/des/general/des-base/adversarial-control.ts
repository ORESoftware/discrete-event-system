'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/adversarial_control.rs
// - Keep file-for-file. Channel constants become pub const values; observation,
//   control, and disturbance token classes become token structs.
// - ClosedLoopGameTraceRow and ClosedLoopPlantOptions become data structs, while
//   ClosedLoopPlantStation, FeedbackPolicyStation, and DisturbancePolicyStation
//   become traits plus shared station-state structs.
// - wireClosedLoopGame and runClosedLoopGame can remain module functions for
//   graph assembly/running; if individual policies become pure graph adapters,
//   model them as PureTransform/PureTransformEntity implementors.
// - Convert validation and runner failures from thrown errors to Result.

// =============================================================================
// Shared station/token bases for closed-loop adversarial and stochastic control.
//
// Topology:
//   Plant station --StateObservationToken--> controller station
//   Plant station --StateObservationToken--> disturbance/adversary station
//   Controller station --ControlMoveToken--> plant station
//   Adversary station --DisturbanceMoveToken--> plant station
//
// The plant owns the continuous state. Policies and adversaries are stationary
// entities that emit movable command tokens. This is intentionally visual-editor
// friendly: every edge has a concrete token type.
// =============================================================================

import {DESStation, ChannelName, Token} from './station';
import {Preconditions} from './preconditions';
import {IterativeRunSummary, runIterativeDES} from './runner';

export const CH_OBSERVATION: ChannelName = 'observation';
export const CH_CONTROL: ChannelName = 'control';
export const CH_DISTURBANCE: ChannelName = 'disturbance';

export class StateObservationToken implements Token {
  constructor(
    readonly state: number[],
    readonly tick: number,
    readonly time: number,
  ) {}
}

export class ControlMoveToken implements Token {
  constructor(
    readonly control: number[],
    readonly tick: number,
    readonly time: number,
  ) {}
}

export class DisturbanceMoveToken implements Token {
  constructor(
    readonly disturbance: number[],
    readonly tick: number,
    readonly time: number,
  ) {}
}

export interface ClosedLoopGameTraceRow {
  tick: number;
  time: number;
  state: number[];
  control: number[];
  disturbance: number[];
  cost: number;
}

export interface ClosedLoopPlantOptions {
  x0: number[];
  dt: number;
  numSteps: number;
  controlDim: number;
  disturbanceDim: number;
}

export abstract class ClosedLoopPlantStation extends DESStation {
  protected state: number[];
  protected readonly dt: number;
  protected readonly numSteps: number;
  protected readonly controlDim: number;
  protected readonly disturbanceDim: number;
  protected control: number[];
  protected disturbance: number[];
  protected tick = 0;
  protected emittedInitialObservation = false;
  protected finished = false;

  readonly trace: ClosedLoopGameTraceRow[] = [];
  readonly stateHistory: number[][] = [];

  constructor(id: string, opts: ClosedLoopPlantOptions) {
    super(id);
    Preconditions.nonEmpty(id, 'x0', opts.x0);
    Preconditions.allFinite(id, 'x0', opts.x0);
    Preconditions.positive(id, 'dt', opts.dt);
    Preconditions.integerInRange(id, 'numSteps', opts.numSteps, 1, 1e9);
    Preconditions.integerInRange(id, 'controlDim', opts.controlDim, 1, 1e6);
    Preconditions.integerInRange(id, 'disturbanceDim', opts.disturbanceDim, 1, 1e6);
    this.state = opts.x0.slice();
    this.dt = opts.dt;
    this.numSteps = opts.numSteps;
    this.controlDim = opts.controlDim;
    this.disturbanceDim = opts.disturbanceDim;
    this.control = new Array(this.controlDim).fill(0);
    this.disturbance = new Array(this.disturbanceDim).fill(0);
    this.stateHistory.push(this.state.slice());
  }

  override assertPreconditions(): void {
    const cls = this.constructor.name;
    Preconditions.nonEmpty(cls, 'x0', this.state);
    Preconditions.allFinite(cls, 'x0', this.state);
    Preconditions.positive(cls, 'dt', this.dt);
    Preconditions.integerInRange(cls, 'numSteps', this.numSteps, 1, 1e9);
    Preconditions.integerInRange(cls, 'controlDim', this.controlDim, 1, 1e6);
    Preconditions.integerInRange(cls, 'disturbanceDim', this.disturbanceDim, 1, 1e6);
  }

  protected abstract dynamics(
    state: readonly number[],
    control: readonly number[],
    disturbance: readonly number[],
    dt: number,
  ): number[];

  protected stageCost(
    state: readonly number[],
    control: readonly number[],
    disturbance: readonly number[],
    nextState: readonly number[],
  ): number {
    const stateCost = nextState.reduce((acc, x) => acc + x * x, 0);
    const controlCost = control.reduce((acc, u) => acc + u * u, 0);
    const disturbanceCost = disturbance.reduce((acc, w) => acc + w * w, 0);
    return stateCost + 0.01 * controlCost + 0.01 * disturbanceCost;
  }

  protected terminal(_state: readonly number[], _tick: number): boolean { return false; }

  runTimeStep(): void {
    if (this.finished) return;
    if (!this.emittedInitialObservation) {
      this.emittedInitialObservation = true;
      this.emitObservation();
      return;
    }

    for (const t of this.drain<ControlMoveToken>(CH_CONTROL)) this.control = t.control.slice();
    for (const t of this.drain<DisturbanceMoveToken>(CH_DISTURBANCE)) this.disturbance = t.disturbance.slice();
    Preconditions.lengthEq(this.id, 'control', this.control, this.controlDim);
    Preconditions.allFinite(this.id, 'control', this.control);
    Preconditions.lengthEq(this.id, 'disturbance', this.disturbance, this.disturbanceDim);
    Preconditions.allFinite(this.id, 'disturbance', this.disturbance);
    if (this.tick >= this.numSteps || this.terminal(this.state, this.tick)) {
      this.finished = true;
      return;
    }

    const prev = this.state.slice();
    const next = this.dynamics(prev, this.control, this.disturbance, this.dt);
    Preconditions.lengthEq(this.id, 'next state', next, prev.length);
    Preconditions.allFinite(this.id, 'next state', next);
    const cost = this.stageCost(prev, this.control, this.disturbance, next);
    Preconditions.finite(this.id, 'stage cost', cost);
    this.state = next.slice();
    this.tick += 1;
    this.stateHistory.push(this.state.slice());
    this.trace.push({
      tick: this.tick,
      time: this.tick * this.dt,
      state: this.state.slice(),
      control: this.control.slice(),
      disturbance: this.disturbance.slice(),
      cost,
    });
    this.emitObservation();
  }

  override hasWork(): boolean { return !this.finished; }

  getState(): number[] { return this.state.slice(); }
  getTick(): number { return this.tick; }
  getDt(): number { return this.dt; }
  getNumSteps(): number { return this.numSteps; }

  private emitObservation(): void {
    this.emit(new StateObservationToken(this.state.slice(), this.tick, this.tick * this.dt), CH_OBSERVATION);
  }
}

export abstract class FeedbackPolicyStation extends DESStation {
  readonly controlHistory: number[][] = [];

  constructor(id: string, protected readonly controlDim: number) {
    super(id);
    Preconditions.integerInRange(id, 'controlDim', controlDim, 1, 1e6);
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange(this.constructor.name, 'controlDim', this.controlDim, 1, 1e6);
  }

  protected abstract policy(observation: StateObservationToken): number[];

  runTimeStep(): void {
    for (const obs of this.drain<StateObservationToken>(CH_OBSERVATION)) {
      const u = this.policy(obs);
      Preconditions.lengthEq(this.id, 'control', u, this.controlDim);
      Preconditions.allFinite(this.id, 'control', u);
      this.controlHistory.push(u.slice());
      this.emit(new ControlMoveToken(u.slice(), obs.tick, obs.time), CH_CONTROL);
    }
  }

  override hasWork(): boolean { return this.inboxSize(CH_OBSERVATION) > 0; }
}

export abstract class DisturbancePolicyStation extends DESStation {
  readonly disturbanceHistory: number[][] = [];

  constructor(id: string, protected readonly disturbanceDim: number) {
    super(id);
    Preconditions.integerInRange(id, 'disturbanceDim', disturbanceDim, 1, 1e6);
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange(this.constructor.name, 'disturbanceDim', this.disturbanceDim, 1, 1e6);
  }

  protected abstract policy(observation: StateObservationToken): number[];

  runTimeStep(): void {
    for (const obs of this.drain<StateObservationToken>(CH_OBSERVATION)) {
      const w = this.policy(obs);
      Preconditions.lengthEq(this.id, 'disturbance', w, this.disturbanceDim);
      Preconditions.allFinite(this.id, 'disturbance', w);
      this.disturbanceHistory.push(w.slice());
      this.emit(new DisturbanceMoveToken(w.slice(), obs.tick, obs.time), CH_DISTURBANCE);
    }
  }

  override hasWork(): boolean { return this.inboxSize(CH_OBSERVATION) > 0; }
}

export function wireClosedLoopGame(
  plant: ClosedLoopPlantStation,
  controller: FeedbackPolicyStation,
  adversary: DisturbancePolicyStation,
): void {
  plant.pipe(controller, CH_OBSERVATION, CH_OBSERVATION);
  plant.pipe(adversary, CH_OBSERVATION, CH_OBSERVATION);
  controller.pipe(plant, CH_CONTROL, CH_CONTROL);
  adversary.pipe(plant, CH_DISTURBANCE, CH_DISTURBANCE);
}

export interface ClosedLoopGameRunOptions {
  maxTicks?: number;
  runValidators?: boolean;
}

export function runClosedLoopGame(
  plant: ClosedLoopPlantStation,
  controller: FeedbackPolicyStation,
  adversary: DisturbancePolicyStation,
  opts: ClosedLoopGameRunOptions = {},
): IterativeRunSummary {
  wireClosedLoopGame(plant, controller, adversary);
  return runIterativeDES([plant, controller, adversary], {
    shuffle: false,
    maxTicks: opts.maxTicks ?? plant.getNumSteps() + 3,
    runValidators: opts.runValidators ?? false,
  });
}
