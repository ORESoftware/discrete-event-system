'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/learning_optimization.rs
// - Keep file-for-file. Token classes become token structs; StationGraphSummary,
//   GradientEvaluation, and optimizer options become data/config structs.
// - Source/sink/evaluator/optimizer station classes become structs implementing
//   DESStation; GradientOptimizerStation and CandidateEvaluatorStation should be
//   traits plus shared state structs for their template-method hooks.
// - Pure numeric helpers such as dot, norm2, sigmoid, softmax, zeros, and
//   cloneMatrix can stay module functions; if a helper is used as a graph node,
//   make it a PureTransform/PureTransformEntity implementation.
// - Convert batch-size, learning-rate, and gradient-shape throws to Result.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/learning_optimization.rs  (module des::general::des_base::learning_optimization)
// 1:1 file move. Stations/tokens for supervised learning + candidate optimization
// pipelines (samples → batches → gradient steps; candidates → evals → incumbent),
// plus a small math/topology helper kit.
//
// Declarations → Rust:
//   interface StationGraphSummary / GradientEvaluation / GradientOptimizerOptions -> structs
//   class Vector*/Gradient*/Candidate*/Evaluated*/Incumbent* Token -> struct + impl Token
//   class VectorSampleSourceStation / MiniBatchStation / GradientTraceSinkStation /
//         CandidateSourceStation<S> / IncumbentSinkStation<S> /
//         SingleTokenSourceStation<T> / LatestTokenSinkStation<T> -> struct + impl DESStation
//   abstract class GradientOptimizerStation -> trait/struct: DESStation (required
//                                              evaluateBatch hook; SGD/Adam template)
//   abstract class CandidateEvaluatorStation<S> -> trait/struct: DESStation (required
//                                              evaluateCandidate hook)
//   fn stationGraph/emptyStationGraph/channelEdge/stateLoopTopology/runStateLoopPipeline/
//      nonEmptyArray/cloneMatrix/zeros/dot/norm2/sigmoid/softmax -> free fns
//
// Conversion notes (file-specific):
//   - `optimizer?: 'sgd' | 'adam'` -> enum Optimizer { Sgd, Adam }; Adam keeps m/v
//     moment buffers (`Vec<f64>`).
//   - `meta: Record<string, unknown>` -> `HashMap<String, serde_json::Value>`.
//   - `SingleTokenSourceStation<T>` takes a `tokenFactory: () => T` + validateToken
//     closure -> boxed `Fn`/`FnMut`; `T extends Token` -> `T: Token`.
//   - GENERIC math helpers (zeros/dot/norm2/softmax/sigmoid) DUPLICATE shared/linalg —
//     prefer `crate::des::shared::linalg`/`VecOps` in Rust; drop the local copies.
//   - `stations: (DESStation | string)[]` union -> enum or `&dyn HasId`; `channelEdge`
//     formats ids -> `format!`.
//   - `samples[0]?.meta.epoch ?? 0` optional-chain coercion via `Number(..)` ->
//     explicit `Option` + parse; no implicit coercion.
//   - `throw new Error` (bad batchSize/lr/gradient len) -> `Result`/`panic!`.
//   - `.slice()` defensive copies throughout -> `.clone()`.
// =============================================================================

// =============================================================================
// general/des-base/learning-optimization.ts
//
// Shared stationary stations and movable tokens for supervised learning and
// vector/candidate optimization models. These helpers keep the DES topology
// explicit: samples, batches, candidates, evaluations, gradients, parameters,
// and incumbents move through named station channels instead of being hidden
// inside a monolithic solver call.
// =============================================================================

import {ChannelName, DEFAULT_CHANNEL, DESStation, Token} from './station';
import {NumericVector} from './neural-network';
import {IterativeRunOptions, IterativeRunSummary, runIterativeDES} from './runner';

export interface StationGraphSummary {
  stations: string[];
  movables: string[];
  edges: string[];
}

export class VectorSampleToken implements Token {
  constructor(
    public readonly id: string,
    public readonly input: NumericVector,
    public readonly target: NumericVector,
    public readonly weight = 1,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class VectorBatchToken implements Token {
  constructor(
    public readonly id: string,
    public readonly samples: readonly VectorSampleToken[],
    public readonly epoch: number,
    public readonly batchIndex: number,
  ) {}
}

export class GradientStepToken implements Token {
  constructor(
    public readonly step: number,
    public readonly loss: number,
    public readonly gradientNorm: number,
    public readonly parameters: NumericVector,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class CandidateToken<S> implements Token {
  constructor(
    public readonly id: string,
    public readonly candidate: S,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class EvaluatedCandidateToken<S> implements Token {
  constructor(
    public readonly id: string,
    public readonly candidate: S,
    public readonly objective: number,
    public readonly feasible = true,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class IncumbentToken<S> implements Token {
  constructor(
    public readonly id: string,
    public readonly candidate: S,
    public readonly objective: number,
    public readonly evaluationCount: number,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class VectorSampleSourceStation extends DESStation {
  static readonly CH_SAMPLE: ChannelName = 'sample';

  private epoch = 0;

  constructor(
    id: string,
    private readonly samples: readonly VectorSampleToken[],
    private readonly epochs = 1,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.epoch < this.epochs; }

  runTimeStep(): void {
    if (this.epoch >= this.epochs) return;
    for (const sample of this.samples) {
      this.emit(
        new VectorSampleToken(
          `${sample.id}:e${this.epoch}`,
          sample.input.slice(),
          sample.target.slice(),
          sample.weight,
          {...sample.meta, epoch: this.epoch},
        ),
        VectorSampleSourceStation.CH_SAMPLE,
      );
    }
    this.epoch += 1;
  }

  getEpoch(): number { return this.epoch; }
}

export class MiniBatchStation extends DESStation {
  static readonly CH_SAMPLE: ChannelName = VectorSampleSourceStation.CH_SAMPLE;
  static readonly CH_BATCH: ChannelName = 'batch';

  private readonly buffer: VectorSampleToken[] = [];
  private batchIndex = 0;

  constructor(
    id: string,
    private readonly batchSize: number,
    private readonly flushPartial = true,
  ) {
    super(id);
    if (!Number.isInteger(batchSize) || batchSize <= 0) throw new Error('batchSize must be a positive integer');
  }

  override hasWork(): boolean {
    return this.inboxSize(MiniBatchStation.CH_SAMPLE) > 0 || this.buffer.length > 0;
  }

  runTimeStep(): void {
    const incoming = this.drain<VectorSampleToken>(MiniBatchStation.CH_SAMPLE);
    this.buffer.push(...incoming);
    while (this.buffer.length >= this.batchSize) this.emitNextBatch(this.batchSize);
    if (incoming.length === 0 && this.flushPartial && this.buffer.length > 0) {
      this.emitNextBatch(this.buffer.length);
    }
  }

  private emitNextBatch(size: number): void {
    const samples = this.buffer.splice(0, size);
    const epoch = Number(samples[0]?.meta.epoch ?? 0);
    this.emit(
      new VectorBatchToken(`batch-${this.batchIndex}`, samples, epoch, this.batchIndex),
      MiniBatchStation.CH_BATCH,
    );
    this.batchIndex += 1;
  }

  getBatchCount(): number { return this.batchIndex; }
}

export interface GradientEvaluation {
  loss: number;
  gradient: NumericVector;
  meta?: Record<string, unknown>;
}

export interface GradientOptimizerOptions {
  initialParameters: NumericVector;
  learningRate: number;
  optimizer?: 'sgd' | 'adam';
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export abstract class GradientOptimizerStation extends DESStation {
  static readonly CH_BATCH: ChannelName = MiniBatchStation.CH_BATCH;
  static readonly CH_STEP: ChannelName = 'gradient-step';

  protected parameters: NumericVector;
  protected step = 0;
  protected readonly lossHistory: number[] = [];
  protected readonly gradientNormHistory: number[] = [];
  private readonly learningRate: number;
  private readonly optimizer: 'sgd' | 'adam';
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;
  private readonly m: NumericVector;
  private readonly v: NumericVector;

  constructor(id: string, opts: GradientOptimizerOptions) {
    super(id);
    this.parameters = opts.initialParameters.slice();
    this.learningRate = opts.learningRate;
    this.optimizer = opts.optimizer ?? 'sgd';
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.epsilon = opts.epsilon ?? 1e-8;
    this.m = zeros(this.parameters.length);
    this.v = zeros(this.parameters.length);
    if (this.learningRate <= 0) throw new Error('learningRate must be positive');
  }

  override hasWork(): boolean {
    return this.inboxSize(GradientOptimizerStation.CH_BATCH) > 0;
  }

  protected abstract evaluateBatch(batch: VectorBatchToken, parameters: readonly number[]): GradientEvaluation;

  runTimeStep(): void {
    const batches = this.drain<VectorBatchToken>(GradientOptimizerStation.CH_BATCH);
    for (const batch of batches) {
      const evaluation = this.evaluateBatch(batch, this.parameters);
      if (evaluation.gradient.length !== this.parameters.length) {
        throw new Error(`gradient length ${evaluation.gradient.length} != parameter length ${this.parameters.length}`);
      }
      this.step += 1;
      this.applyGradient(evaluation.gradient);
      const gradNorm = norm2(evaluation.gradient);
      this.lossHistory.push(evaluation.loss);
      this.gradientNormHistory.push(gradNorm);
      this.emit(
        new GradientStepToken(
          this.step,
          evaluation.loss,
          gradNorm,
          this.parameters.slice(),
          evaluation.meta ?? {},
        ),
        GradientOptimizerStation.CH_STEP,
      );
    }
  }

  private applyGradient(gradient: readonly number[]): void {
    if (this.optimizer === 'sgd') {
      for (let i = 0; i < this.parameters.length; i++) {
        this.parameters[i] -= this.learningRate * gradient[i];
      }
      return;
    }
    for (let i = 0; i < this.parameters.length; i++) {
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * gradient[i];
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * gradient[i] * gradient[i];
      const mHat = this.m[i] / (1 - Math.pow(this.beta1, this.step));
      const vHat = this.v[i] / (1 - Math.pow(this.beta2, this.step));
      this.parameters[i] -= this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
  }

  getParameters(): NumericVector { return this.parameters.slice(); }
  getStep(): number { return this.step; }
  getLossHistory(): NumericVector { return this.lossHistory.slice(); }
  getGradientNormHistory(): NumericVector { return this.gradientNormHistory.slice(); }
}

export class GradientTraceSinkStation extends DESStation {
  static readonly CH_STEP: ChannelName = GradientOptimizerStation.CH_STEP;
  readonly trace: GradientStepToken[] = [];

  constructor(id: string) { super(id); }

  override hasWork(): boolean { return this.inboxSize(GradientTraceSinkStation.CH_STEP) > 0; }

  runTimeStep(): void {
    this.trace.push(...this.drain<GradientStepToken>(GradientTraceSinkStation.CH_STEP));
  }
}

export class CandidateSourceStation<S> extends DESStation {
  static readonly CH_CANDIDATE: ChannelName = 'candidate';
  private emitted = false;

  constructor(id: string, private readonly candidates: readonly CandidateToken<S>[]) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    for (const candidate of this.candidates) this.emit(candidate, CandidateSourceStation.CH_CANDIDATE);
    this.emitted = true;
  }
}

export abstract class CandidateEvaluatorStation<S> extends DESStation {
  static readonly CH_CANDIDATE: ChannelName = CandidateSourceStation.CH_CANDIDATE;
  static readonly CH_EVALUATED: ChannelName = 'evaluated';

  override hasWork(): boolean {
    return this.inboxSize(CandidateEvaluatorStation.CH_CANDIDATE) > 0;
  }

  protected abstract evaluateCandidate(token: CandidateToken<S>): EvaluatedCandidateToken<S>;

  runTimeStep(): void {
    const candidates = this.drain<CandidateToken<S>>(CandidateEvaluatorStation.CH_CANDIDATE);
    for (const candidate of candidates) {
      this.emit(this.evaluateCandidate(candidate), CandidateEvaluatorStation.CH_EVALUATED);
    }
  }
}

export class IncumbentSinkStation<S> extends DESStation {
  static readonly CH_EVALUATED: ChannelName = CandidateEvaluatorStation.CH_EVALUATED;

  readonly evaluations: EvaluatedCandidateToken<S>[] = [];
  private incumbent: EvaluatedCandidateToken<S> | undefined;

  constructor(id: string) { super(id); }

  override hasWork(): boolean {
    return this.inboxSize(IncumbentSinkStation.CH_EVALUATED) > 0;
  }

  runTimeStep(): void {
    const evaluated = this.drain<EvaluatedCandidateToken<S>>(IncumbentSinkStation.CH_EVALUATED);
    for (const item of evaluated) {
      this.evaluations.push(item);
      if (!item.feasible) continue;
      if (!this.incumbent || item.objective < this.incumbent.objective) this.incumbent = item;
    }
    if (this.incumbent) {
      this.emit(
        new IncumbentToken(
          this.incumbent.id,
          this.incumbent.candidate,
          this.incumbent.objective,
          this.evaluations.length,
          this.incumbent.meta,
        ),
        DEFAULT_CHANNEL,
      );
    }
  }

  getIncumbent(): EvaluatedCandidateToken<S> | undefined { return this.incumbent; }
}

export class SingleTokenSourceStation<T extends Token> extends DESStation {
  private emitted = false;
  private token: T | undefined;

  constructor(
    id: string,
    readonly outputChannel: ChannelName,
    private readonly tokenFactory: () => T,
    private readonly validateToken: (token: T) => void = () => {},
  ) {
    super(id);
  }

  override assertPreconditions(): void {
    this.initialToken();
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(this.initialToken(), this.outputChannel);
    this.emitted = true;
  }

  private initialToken(): T {
    if (!this.token) {
      this.token = this.tokenFactory();
      this.validateToken(this.token);
    }
    return this.token;
  }
}

export class LatestTokenSinkStation<T extends Token> extends DESStation {
  latest: T | undefined;

  constructor(id: string, readonly inputChannel: ChannelName) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(this.inputChannel) > 0; }

  runTimeStep(): void {
    const tokens = this.drain<T>(this.inputChannel);
    if (tokens.length > 0) this.latest = tokens[tokens.length - 1];
  }
}

export function stationGraph(
  stations: readonly (DESStation | string)[],
  movables: readonly string[],
  edges: readonly string[],
): StationGraphSummary {
  return {
    stations: stations.map(s => typeof s === 'string' ? s : s.id),
    movables: movables.slice(),
    edges: edges.slice(),
  };
}

export function emptyStationGraph(): StationGraphSummary {
  return stationGraph([], [], []);
}

export function channelEdge(
  source: DESStation | string,
  sourceChannel: ChannelName,
  target: DESStation | string,
  targetChannel: ChannelName = sourceChannel,
): string {
  const sourceId = typeof source === 'string' ? source : source.id;
  const targetId = typeof target === 'string' ? target : target.id;
  return `${sourceId}:${sourceChannel} -> ${targetId}:${targetChannel}`;
}

export function stateLoopTopology(
  source: DESStation,
  update: DESStation,
  sink: DESStation,
  stateChannel: ChannelName,
  resultChannel: ChannelName,
  movables: readonly string[],
): StationGraphSummary {
  return stationGraph([source, update, sink], movables, [
    channelEdge(source, stateChannel, update, stateChannel),
    channelEdge(update, stateChannel, update, stateChannel),
    channelEdge(update, resultChannel, sink, resultChannel),
  ]);
}

export function runStateLoopPipeline(
  source: DESStation,
  update: DESStation,
  sink: DESStation,
  stateChannel: ChannelName,
  resultChannel: ChannelName,
  opts: IterativeRunOptions = {},
): IterativeRunSummary {
  source.pipe(update, stateChannel, stateChannel);
  update.pipe(update, stateChannel, stateChannel);
  update.pipe(sink, resultChannel, resultChannel);
  return runIterativeDES([source, update, sink], {shuffle: false, ...opts});
}

export function nonEmptyArray<T>(value: readonly T[] | undefined, fallback: readonly T[]): T[] {
  return value && value.length > 0 ? value.slice() : fallback.slice();
}

export function cloneMatrix<T>(matrix: readonly (readonly T[])[]): T[][] {
  return matrix.map(row => row.slice());
}

export function zeros(n: number): NumericVector {
  return Array.from({length: n}, () => 0);
}

export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm2(v: readonly number[]): number {
  return Math.sqrt(dot(v, v));
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function softmax(logits: readonly number[]): NumericVector {
  const m = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - m));
  const z = exps.reduce((s, v) => s + v, 0);
  return exps.map(v => v / z);
}
