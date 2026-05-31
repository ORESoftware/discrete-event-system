// RUST MIGRATION: Target module `src/des/general/nonlinear_optimization_models.rs`.
// RUST MIGRATION: Convert opt/NLS params, points, and results to `serde` structs; keep the topology alias as the Rust graph-summary type alias.
// RUST MIGRATION: Replace abstract update-station inheritance with traits for unconstrained and least-squares update behavior, implemented by Newton/BFGS/Gauss-Newton/LM structs.
// RUST MIGRATION: Solver runners can remain free functions because they build and execute DES graphs; graph-visible station leaves should be trait objects or enums where dispatch is finite.
// RUST MIGRATION: Turn validation, backtracking failure, and linear-solve singularity into `Result` errors, with vectors/matrices represented as `Vec<f64>`/`Vec<Vec<f64>>`.
'use strict';

// =============================================================================
// general/nonlinear-optimization-models.ts
//
// Newton/quasi-Newton and nonlinear least-squares routines as DES state-token
// loops. Each model is a flat graph: source -> update station (self-loop) ->
// sink, with movable state/result tokens.
// =============================================================================

import {
  ChannelName,
  DESStation,
  LatestTokenSinkStation,
  SingleTokenSourceStation,
  StationGraphSummary,
  Token,
  dot,
  emptyStationGraph,
  nonEmptyArray,
  norm2,
  Preconditions,
  runStateLoopPipeline,
  stateLoopTopology,
} from './des-base';

export type NonlinearTopology = StationGraphSummary;

const CH_OPT_STATE: ChannelName = 'opt-state';
const CH_OPT_RESULT: ChannelName = 'opt-result';
const CH_NLS_STATE: ChannelName = 'nls-state';
const CH_NLS_RESULT: ChannelName = 'nls-result';

export interface UnconstrainedOptParams {
  x0?: number[];
  maxIter?: number;
  tol?: number;
}

export interface UnconstrainedOptResult {
  x: number[];
  objective: number;
  gradientNorm: number;
  iterations: number;
  trace: Array<{iter: number; objective: number; gradientNorm: number; x: number[]}>;
  topology: NonlinearTopology;
}

class OptStateToken implements Token {
  constructor(readonly iter: number, readonly x: number[], readonly H?: number[][]) {}
}

class OptResultToken implements Token {
  constructor(readonly result: UnconstrainedOptResult) {}
}

abstract class UnconstrainedUpdateStation extends DESStation {
  static readonly CH_STATE: ChannelName = CH_OPT_STATE;
  static readonly CH_RESULT: ChannelName = CH_OPT_RESULT;
  readonly trace: Array<{iter: number; objective: number; gradientNorm: number; x: number[]}> = [];

  constructor(id: string, private readonly maxIter: number, private readonly tol: number) { super(id); }

  override hasWork(): boolean { return this.inboxSize(UnconstrainedUpdateStation.CH_STATE) > 0; }

  protected abstract objective(x: readonly number[]): number;
  protected abstract gradient(x: readonly number[]): number[];
  protected abstract nextState(state: OptStateToken, gradient: readonly number[]): OptStateToken;

  runTimeStep(): void {
    const states = this.drain<OptStateToken>(UnconstrainedUpdateStation.CH_STATE);
    for (const state of states) {
      const gradient = this.gradient(state.x);
      const gradientNorm = norm2(gradient);
      const objective = this.objective(state.x);
      this.trace.push({iter: state.iter, objective, gradientNorm, x: state.x.slice()});
      if (state.iter >= this.maxIter || gradientNorm <= this.tol) {
        this.emit(new OptResultToken({
          x: state.x.slice(),
          objective,
          gradientNorm,
          iterations: state.iter,
          trace: this.trace.slice(),
          topology: emptyStationGraph(),
        }), UnconstrainedUpdateStation.CH_RESULT);
        continue;
      }
      this.emit(this.nextState(state, gradient), UnconstrainedUpdateStation.CH_STATE);
    }
  }
}

class NewtonRosenbrockStation extends UnconstrainedUpdateStation {
  protected objective(x: readonly number[]): number { return rosenbrock(x); }
  protected gradient(x: readonly number[]): number[] { return rosenbrockGrad(x); }
  protected nextState(state: OptStateToken, gradient: readonly number[]): OptStateToken {
    const h = rosenbrockHessian(state.x);
    const step = solve2(h, [-gradient[0], -gradient[1]]);
    const alpha = backtracking(state.x, step, rosenbrock, gradient);
    return new OptStateToken(state.iter + 1, [state.x[0] + alpha * step[0], state.x[1] + alpha * step[1]]);
  }
}

class BFGSRosenbrockStation extends UnconstrainedUpdateStation {
  protected objective(x: readonly number[]): number { return rosenbrock(x); }
  protected gradient(x: readonly number[]): number[] { return rosenbrockGrad(x); }
  protected nextState(state: OptStateToken, gradient: readonly number[]): OptStateToken {
    const H = state.H ?? [[1, 0], [0, 1]];
    const step = [-dot(H[0], gradient), -dot(H[1], gradient)];
    const alpha = backtracking(state.x, step, rosenbrock, gradient);
    const xNext = [state.x[0] + alpha * step[0], state.x[1] + alpha * step[1]];
    const gNext = rosenbrockGrad(xNext);
    const s = [xNext[0] - state.x[0], xNext[1] - state.x[1]];
    const y = [gNext[0] - gradient[0], gNext[1] - gradient[1]];
    const Hnext = bfgsInverseUpdate(H, s, y);
    return new OptStateToken(state.iter + 1, xNext, Hnext);
  }
}

export function runNewtonRosenbrock(params: UnconstrainedOptParams): UnconstrainedOptResult {
  const x0 = nonEmptyArray(params.x0, [-1.2, 1]);
  return runUnconstrained('newton-state-source', new NewtonRosenbrockStation('newton-update', params.maxIter ?? 50, params.tol ?? 1e-8), x0);
}

export function runBFGSRosenbrock(params: UnconstrainedOptParams): UnconstrainedOptResult {
  const x0 = nonEmptyArray(params.x0, [-1.2, 1]);
  return runUnconstrained('bfgs-state-source', new BFGSRosenbrockStation('bfgs-update', params.maxIter ?? 100, params.tol ?? 1e-6), x0);
}

function runUnconstrained(sourceId: string, update: UnconstrainedUpdateStation, x0: number[]): UnconstrainedOptResult {
  const source = new SingleTokenSourceStation<OptStateToken>(
    sourceId,
    CH_OPT_STATE,
    () => new OptStateToken(0, x0.slice()),
    token => validateOptInitialState(sourceId, token),
  );
  const sink = new LatestTokenSinkStation<OptResultToken>('opt-result-sink', CH_OPT_RESULT);
  runStateLoopPipeline(source, update, sink, CH_OPT_STATE, CH_OPT_RESULT, {maxTicks: 500});
  if (!sink.latest) throw new Error(`${update.id} did not produce a result`);
  sink.latest.result.topology = stateLoopTopology(source, update, sink, CH_OPT_STATE, CH_OPT_RESULT, ['OptStateToken', 'OptResultToken']);
  return sink.latest.result;
}

export interface CurveFitPoint {
  x: number;
  y: number;
}

export interface NonlinearLeastSquaresParams {
  points?: CurveFitPoint[];
  initial?: number[];
  maxIter?: number;
  tol?: number;
  lambda?: number;
}

export interface NonlinearLeastSquaresResult {
  params: number[];
  sse: number;
  gradientNorm: number;
  iterations: number;
  trace: Array<{iter: number; sse: number; gradientNorm: number; params: number[]}>;
  topology: NonlinearTopology;
}

class NLStateToken implements Token {
  constructor(readonly iter: number, readonly params: number[], readonly lambda: number) {}
}

class NLResultToken implements Token {
  constructor(readonly result: NonlinearLeastSquaresResult) {}
}

abstract class NonlinearLeastSquaresStation extends DESStation {
  static readonly CH_STATE: ChannelName = CH_NLS_STATE;
  static readonly CH_RESULT: ChannelName = CH_NLS_RESULT;
  readonly trace: Array<{iter: number; sse: number; gradientNorm: number; params: number[]}> = [];

  constructor(id: string, protected readonly points: readonly CurveFitPoint[], private readonly maxIter: number, private readonly tol: number) { super(id); }
  override hasWork(): boolean { return this.inboxSize(NonlinearLeastSquaresStation.CH_STATE) > 0; }
  protected abstract damping(state: NLStateToken): number;

  runTimeStep(): void {
    for (const state of this.drain<NLStateToken>(NonlinearLeastSquaresStation.CH_STATE)) {
      const system = normalEquations(state.params, this.points, this.damping(state));
      const gradientNorm = norm2(system.gradient);
      const sse = expSSE(state.params, this.points);
      this.trace.push({iter: state.iter, sse, gradientNorm, params: state.params.slice()});
      if (state.iter >= this.maxIter || gradientNorm <= this.tol) {
        this.emit(new NLResultToken({
          params: state.params.slice(),
          sse,
          gradientNorm,
          iterations: state.iter,
          trace: this.trace.slice(),
          topology: emptyStationGraph(),
        }), NonlinearLeastSquaresStation.CH_RESULT);
        continue;
      }
      const step = solveLinear(system.A, system.b);
      const next = state.params.map((v, i) => v + step[i]);
      this.emit(new NLStateToken(state.iter + 1, next, state.lambda), NonlinearLeastSquaresStation.CH_STATE);
    }
  }
}

class GaussNewtonStation extends NonlinearLeastSquaresStation {
  protected damping(_state: NLStateToken): number { return 0; }
}

class LevenbergMarquardtStation extends NonlinearLeastSquaresStation {
  protected damping(state: NLStateToken): number { return state.lambda; }
}

export function runGaussNewtonCurveFit(params: NonlinearLeastSquaresParams): NonlinearLeastSquaresResult {
  const points = nonEmptyArray(params.points, defaultFitPoints());
  const initial = nonEmptyArray(params.initial, [1, -0.2]);
  return runNLS('gauss-newton-source', new GaussNewtonStation('gauss-newton-update', points, params.maxIter ?? 20, params.tol ?? 1e-8), initial, 0);
}

export function runLevenbergMarquardtCurveFit(params: NonlinearLeastSquaresParams): NonlinearLeastSquaresResult {
  const points = nonEmptyArray(params.points, defaultFitPoints());
  const initial = nonEmptyArray(params.initial, [1, -0.2]);
  return runNLS('lm-source', new LevenbergMarquardtStation('levenberg-marquardt-update', points, params.maxIter ?? 30, params.tol ?? 1e-8), initial, params.lambda ?? 0.1);
}

function runNLS(sourceId: string, update: NonlinearLeastSquaresStation, initial: number[], lambda: number): NonlinearLeastSquaresResult {
  const source = new SingleTokenSourceStation<NLStateToken>(
    sourceId,
    CH_NLS_STATE,
    () => new NLStateToken(0, initial.slice(), lambda),
    token => validateNLSInitialState(sourceId, token),
  );
  const sink = new LatestTokenSinkStation<NLResultToken>('nls-result-sink', CH_NLS_RESULT);
  runStateLoopPipeline(source, update, sink, CH_NLS_STATE, CH_NLS_RESULT, {maxTicks: 200});
  if (!sink.latest) throw new Error(`${update.id} did not produce a result`);
  sink.latest.result.topology = stateLoopTopology(source, update, sink, CH_NLS_STATE, CH_NLS_RESULT, ['NLStateToken', 'NLResultToken']);
  return sink.latest.result;
}

function validateOptInitialState(model: string, token: OptStateToken): void {
  Preconditions.integerInRange(model, 'iter', token.iter, 0, 1e9);
  Preconditions.lengthEq(model, 'x0', token.x, 2);
  Preconditions.allFinite(model, 'x0', token.x);
  if (token.H !== undefined) {
    Preconditions.lengthEq(model, 'H', token.H, 2);
    Preconditions.lengthEq(model, 'H[0]', token.H[0], 2);
    Preconditions.lengthEq(model, 'H[1]', token.H[1], 2);
    Preconditions.allFinite(model, 'H[0]', token.H[0]);
    Preconditions.allFinite(model, 'H[1]', token.H[1]);
  }
}

function validateNLSInitialState(model: string, token: NLStateToken): void {
  Preconditions.integerInRange(model, 'iter', token.iter, 0, 1e9);
  Preconditions.lengthEq(model, 'initial', token.params, 2);
  Preconditions.allFinite(model, 'initial', token.params);
  Preconditions.nonNegative(model, 'lambda', token.lambda);
}

function rosenbrock(x: readonly number[]): number {
  return Math.pow(1 - x[0], 2) + 100 * Math.pow(x[1] - x[0] * x[0], 2);
}

function rosenbrockGrad(x: readonly number[]): number[] {
  return [
    -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] * x[0]),
    200 * (x[1] - x[0] * x[0]),
  ];
}

function rosenbrockHessian(x: readonly number[]): number[][] {
  return [
    [2 - 400 * x[1] + 1200 * x[0] * x[0], -400 * x[0]],
    [-400 * x[0], 200],
  ];
}

function solve2(A: number[][], b: readonly number[]): number[] {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (Math.abs(det) < 1e-12) return b.slice();
  return [
    (b[0] * A[1][1] - A[0][1] * b[1]) / det,
    (A[0][0] * b[1] - b[0] * A[1][0]) / det,
  ];
}

function backtracking(x: readonly number[], p: readonly number[], f: (x: readonly number[]) => number, g: readonly number[]): number {
  let alpha = 1;
  const f0 = f(x);
  const slope = dot(g, p);
  while (alpha > 1e-8) {
    const next = x.map((v, i) => v + alpha * p[i]);
    if (f(next) <= f0 + 1e-4 * alpha * slope) return alpha;
    alpha *= 0.5;
  }
  return alpha;
}

function bfgsInverseUpdate(H: number[][], s: readonly number[], y: readonly number[]): number[][] {
  const ys = dot(y, s);
  if (ys <= 1e-12) return H.map(row => row.slice());
  const rho = 1 / ys;
  const Hy = [dot(H[0], y), dot(H[1], y)];
  const yHy = dot(y, Hy);
  const out = H.map(row => row.slice());
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      out[i][j] += (1 + yHy * rho) * rho * s[i] * s[j] - rho * (s[i] * Hy[j] + Hy[i] * s[j]);
    }
  }
  return out;
}

function expResiduals(params: readonly number[], points: readonly CurveFitPoint[]): number[] {
  const [a, b] = params;
  return points.map(p => a * Math.exp(b * p.x) - p.y);
}

function expJacobian(params: readonly number[], points: readonly CurveFitPoint[]): number[][] {
  const [a, b] = params;
  return points.map(p => {
    const e = Math.exp(b * p.x);
    return [e, a * p.x * e];
  });
}

function expSSE(params: readonly number[], points: readonly CurveFitPoint[]): number {
  return expResiduals(params, points).reduce((s, r) => s + r * r, 0);
}

function normalEquations(params: readonly number[], points: readonly CurveFitPoint[], lambda: number): {A: number[][]; b: number[]; gradient: number[]} {
  const r = expResiduals(params, points);
  const J = expJacobian(params, points);
  const A = [[lambda, 0], [0, lambda]];
  const b = [0, 0];
  const gradient = [0, 0];
  for (let k = 0; k < points.length; k++) {
    for (let i = 0; i < 2; i++) {
      b[i] -= J[k][i] * r[k];
      gradient[i] += 2 * J[k][i] * r[k];
      for (let j = 0; j < 2; j++) A[i][j] += J[k][i] * J[k][j];
    }
  }
  return {A, b, gradient};
}

function solveLinear(A: number[][], b: readonly number[]): number[] {
  return solve2(A, b);
}

function defaultFitPoints(): CurveFitPoint[] {
  return [
    {x: 0, y: 2.00},
    {x: 1, y: 1.22},
    {x: 2, y: 0.74},
    {x: 3, y: 0.45},
    {x: 4, y: 0.27},
  ];
}
