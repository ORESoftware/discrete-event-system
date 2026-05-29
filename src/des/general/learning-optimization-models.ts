'use strict';

// =============================================================================
// general/learning-optimization-models.ts
//
// Station-graph implementations for supervised optimization models:
//   - linear-regression-ls
//   - ridge-regression-ls
//   - logistic-regression-sgd
//   - backprop-mlp-classifier
//
// Every runner builds stationary source/batch/evaluator/update/sink stations
// and moves typed tokens between them. Numerical routines live inside stations,
// not as hidden one-shot logic in adapters.
// =============================================================================

import {
  ChannelName,
  DESStation,
  GradientEvaluation,
  GradientOptimizerStation,
  GradientTraceSinkStation,
  MiniBatchStation,
  StationGraphSummary,
  Token,
  VectorBatchToken,
  VectorSampleSourceStation,
  VectorSampleToken,
  channelEdge,
  dot,
  nonEmptyArray,
  norm2,
  runIterativeDES,
  sigmoid,
  softmax,
  stationGraph,
  zeros,
} from './des-base';
import {mulberry32} from './prng';

export interface SupervisedSample {
  x: number[];
  y: number | number[];
}

export interface LinearRegressionParams {
  samples?: SupervisedSample[];
  fitIntercept?: boolean;
  ridge?: number;
}

export interface LinearRegressionResult {
  coefficients: number[];
  intercept: number;
  mse: number;
  predictions: number[];
  residuals: number[];
  sampleCount: number;
  topology: StationGraphSummary;
}

class RegressionFitToken implements Token {
  constructor(
    readonly coefficients: number[],
    readonly intercept: number,
    readonly sampleCount: number,
  ) {}
}

class NormalEquationStation extends DESStation {
  static readonly CH_SAMPLE: ChannelName = VectorSampleSourceStation.CH_SAMPLE;
  static readonly CH_FIT: ChannelName = 'fit';

  private readonly xtx: number[][];
  private readonly xty: number[];
  private sampleCount = 0;

  constructor(
    id: string,
    private readonly inputDim: number,
    private readonly fitIntercept: boolean,
    private readonly ridge: number,
  ) {
    super(id);
    const d = inputDim + (fitIntercept ? 1 : 0);
    this.xtx = Array.from({length: d}, () => zeros(d));
    this.xty = zeros(d);
  }

  override hasWork(): boolean { return this.inboxSize(NormalEquationStation.CH_SAMPLE) > 0; }

  runTimeStep(): void {
    const samples = this.drain<VectorSampleToken>(NormalEquationStation.CH_SAMPLE);
    for (const sample of samples) {
      const row = this.designRow(sample.input);
      const y = sample.target[0];
      for (let i = 0; i < row.length; i++) {
        this.xty[i] += sample.weight * row[i] * y;
        for (let j = 0; j < row.length; j++) this.xtx[i][j] += sample.weight * row[i] * row[j];
      }
      this.sampleCount += 1;
    }
    if (this.sampleCount > 0) {
      const mat = this.xtx.map((row, i) => row.map((v, j) => v + (i === j ? this.ridge : 0)));
      const beta = solveLinearSystem(mat, this.xty);
      const intercept = this.fitIntercept ? beta[beta.length - 1] : 0;
      const coefficients = this.fitIntercept ? beta.slice(0, -1) : beta;
      this.emit(new RegressionFitToken(coefficients, intercept, this.sampleCount), NormalEquationStation.CH_FIT);
    }
  }

  private designRow(x: readonly number[]): number[] {
    if (x.length !== this.inputDim) throw new Error(`expected input dimension ${this.inputDim}, got ${x.length}`);
    return this.fitIntercept ? [...x, 1] : x.slice();
  }
}

class RegressionFitSinkStation extends DESStation {
  static readonly CH_FIT: ChannelName = NormalEquationStation.CH_FIT;
  fit: RegressionFitToken | undefined;

  constructor(id: string) { super(id); }

  override hasWork(): boolean { return this.inboxSize(RegressionFitSinkStation.CH_FIT) > 0; }

  runTimeStep(): void {
    const fits = this.drain<RegressionFitToken>(RegressionFitSinkStation.CH_FIT);
    if (fits.length > 0) this.fit = fits[fits.length - 1];
  }
}

export function runLinearRegressionLS(params: LinearRegressionParams): LinearRegressionResult {
  const rawSamples = nonEmptyArray(params.samples, defaultRegressionSamples());
  const samples = toVectorSamples(rawSamples);
  const inputDim = samples[0]?.input.length ?? 0;
  const source = new VectorSampleSourceStation('sample-source', samples);
  const normal = new NormalEquationStation('normal-equation-accumulator', inputDim, params.fitIntercept ?? true, params.ridge ?? 0);
  const sink = new RegressionFitSinkStation('regression-fit-sink');
  source.pipe(normal, VectorSampleSourceStation.CH_SAMPLE, NormalEquationStation.CH_SAMPLE);
  normal.pipe(sink, NormalEquationStation.CH_FIT, RegressionFitSinkStation.CH_FIT);
  runIterativeDES([source, normal, sink], {shuffle: false});
  if (!sink.fit) throw new Error('linear-regression-ls did not produce a fit');
  const predictions = samples.map(s => dot(s.input, sink.fit!.coefficients) + sink.fit!.intercept);
  const residuals = samples.map((s, i) => predictions[i] - s.target[0]);
  const mse = residuals.reduce((acc, r) => acc + r * r, 0) / Math.max(1, residuals.length);
  return {
    coefficients: sink.fit.coefficients,
    intercept: sink.fit.intercept,
    mse,
    predictions,
    residuals,
    sampleCount: sink.fit.sampleCount,
    topology: stationGraph([source, normal, sink], ['VectorSampleToken', 'RegressionFitToken'], [
      channelEdge(source, VectorSampleSourceStation.CH_SAMPLE, normal, NormalEquationStation.CH_SAMPLE),
      channelEdge(normal, NormalEquationStation.CH_FIT, sink, RegressionFitSinkStation.CH_FIT),
    ]),
  };
}

export interface RidgeRegressionParams extends LinearRegressionParams {
  ridge?: number;
}

export function runRidgeRegressionLS(params: RidgeRegressionParams): LinearRegressionResult {
  return runLinearRegressionLS({...params, ridge: params.ridge ?? 0.1});
}

export interface LogisticRegressionSGDParams {
  samples?: SupervisedSample[];
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  optimizer?: 'sgd' | 'adam';
  l2?: number;
}

export interface GradientTrainingResult {
  parameters: number[];
  weights: number[];
  bias: number;
  lossHistory: number[];
  gradientNormHistory: number[];
  finalLoss: number;
  accuracy: number;
  predictions: number[];
  topology: StationGraphSummary;
}

class LogisticRegressionStation extends GradientOptimizerStation {
  constructor(
    id: string,
    inputDim: number,
    learningRate: number,
    optimizer: 'sgd' | 'adam',
    private readonly l2: number,
  ) {
    super(id, {initialParameters: zeros(inputDim + 1), learningRate, optimizer});
  }

  protected evaluateBatch(batch: VectorBatchToken, parameters: readonly number[]): GradientEvaluation {
    const gradient = zeros(parameters.length);
    let loss = 0;
    for (const sample of batch.samples) {
      const z = dot(parameters.slice(0, -1), sample.input) + parameters[parameters.length - 1];
      const p = sigmoid(z);
      const y = sample.target[0];
      loss += -sample.weight * (y * Math.log(Math.max(p, 1e-12)) + (1 - y) * Math.log(Math.max(1 - p, 1e-12)));
      const err = sample.weight * (p - y);
      for (let i = 0; i < sample.input.length; i++) gradient[i] += err * sample.input[i];
      gradient[gradient.length - 1] += err;
    }
    for (let i = 0; i < parameters.length - 1; i++) {
      loss += 0.5 * this.l2 * parameters[i] * parameters[i];
      gradient[i] += this.l2 * parameters[i];
    }
    const denom = Math.max(1, batch.samples.length);
    return {loss: loss / denom, gradient: gradient.map(g => g / denom), meta: {batch: batch.id}};
  }
}

export function runLogisticRegressionSGD(params: LogisticRegressionSGDParams): GradientTrainingResult {
  const rawSamples = nonEmptyArray(params.samples, defaultLogisticSamples());
  const samples = toVectorSamples(rawSamples);
  const inputDim = samples[0]?.input.length ?? 0;
  const source = new VectorSampleSourceStation('sample-source', samples, params.epochs ?? 120);
  const batcher = new MiniBatchStation('mini-batch', params.batchSize ?? 4);
  const learner = new LogisticRegressionStation('logistic-gradient-update', inputDim, params.learningRate ?? 0.2, params.optimizer ?? 'sgd', params.l2 ?? 0);
  const trace = new GradientTraceSinkStation('gradient-trace-sink');
  source.pipe(batcher, VectorSampleSourceStation.CH_SAMPLE, MiniBatchStation.CH_SAMPLE);
  batcher.pipe(learner, MiniBatchStation.CH_BATCH, GradientOptimizerStation.CH_BATCH);
  learner.pipe(trace, GradientOptimizerStation.CH_STEP, GradientTraceSinkStation.CH_STEP);
  runIterativeDES([source, batcher, learner, trace], {shuffle: false});
  return gradientTrainingResult(samples, learner, trace, stationGraph([source, batcher, learner, trace], ['VectorSampleToken', 'VectorBatchToken', 'GradientStepToken'], [
    channelEdge(source, VectorSampleSourceStation.CH_SAMPLE, batcher, MiniBatchStation.CH_SAMPLE),
    channelEdge(batcher, MiniBatchStation.CH_BATCH, learner, GradientOptimizerStation.CH_BATCH),
    channelEdge(learner, GradientOptimizerStation.CH_STEP, trace, GradientTraceSinkStation.CH_STEP),
  ]), logisticPredict);
}

export interface BackpropMLPParams {
  samples?: SupervisedSample[];
  hiddenUnits?: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  optimizer?: 'sgd' | 'adam';
  seed?: number;
}

class BackpropMLPStation extends GradientOptimizerStation {
  private readonly inputDim: number;
  private readonly hiddenUnits: number;

  constructor(id: string, inputDim: number, hiddenUnits: number, learningRate: number, optimizer: 'sgd' | 'adam', seed: number) {
    const rng = mulberry32(seed);
    const parameterCount = hiddenUnits * inputDim + hiddenUnits + hiddenUnits + 1;
    const initialParameters = Array.from({length: parameterCount}, () => (rng() - 0.5) * 0.6);
    super(id, {initialParameters, learningRate, optimizer});
    this.inputDim = inputDim;
    this.hiddenUnits = hiddenUnits;
  }

  protected evaluateBatch(batch: VectorBatchToken, parameters: readonly number[]): GradientEvaluation {
    const gradient = zeros(parameters.length);
    let loss = 0;
    for (const sample of batch.samples) {
      const f = this.forward(parameters, sample.input);
      const y = sample.target[0];
      loss += -sample.weight * (y * Math.log(Math.max(f.output, 1e-12)) + (1 - y) * Math.log(Math.max(1 - f.output, 1e-12)));
      const dOut = sample.weight * (f.output - y);
      const w2Offset = this.hiddenUnits * this.inputDim + this.hiddenUnits;
      for (let h = 0; h < this.hiddenUnits; h++) gradient[w2Offset + h] += dOut * f.hidden[h];
      gradient[w2Offset + this.hiddenUnits] += dOut;
      for (let h = 0; h < this.hiddenUnits; h++) {
        const w2 = parameters[w2Offset + h];
        const dHidden = dOut * w2 * f.hidden[h] * (1 - f.hidden[h]);
        for (let i = 0; i < this.inputDim; i++) gradient[h * this.inputDim + i] += dHidden * sample.input[i];
        gradient[this.hiddenUnits * this.inputDim + h] += dHidden;
      }
    }
    const denom = Math.max(1, batch.samples.length);
    return {loss: loss / denom, gradient: gradient.map(g => g / denom), meta: {batch: batch.id}};
  }

  predict(input: readonly number[]): number {
    return this.forward(this.getParameters(), input).output;
  }

  private forward(parameters: readonly number[], input: readonly number[]): {hidden: number[]; output: number} {
    const hidden = zeros(this.hiddenUnits);
    const b1Offset = this.hiddenUnits * this.inputDim;
    const w2Offset = b1Offset + this.hiddenUnits;
    for (let h = 0; h < this.hiddenUnits; h++) {
      let z = parameters[b1Offset + h];
      for (let i = 0; i < this.inputDim; i++) z += parameters[h * this.inputDim + i] * input[i];
      hidden[h] = sigmoid(z);
    }
    const output = sigmoid(dot(parameters.slice(w2Offset, w2Offset + this.hiddenUnits), hidden) + parameters[w2Offset + this.hiddenUnits]);
    return {hidden, output};
  }
}

export function runBackpropMLPClassifier(params: BackpropMLPParams): GradientTrainingResult {
  const rawSamples = nonEmptyArray(params.samples, defaultXorSamples());
  const samples = toVectorSamples(rawSamples);
  const inputDim = samples[0]?.input.length ?? 0;
  const source = new VectorSampleSourceStation('sample-source', samples, params.epochs ?? 800);
  const batcher = new MiniBatchStation('mini-batch', params.batchSize ?? samples.length);
  const learner = new BackpropMLPStation('backprop-gradient-update', inputDim, params.hiddenUnits ?? 4, params.learningRate ?? 0.08, params.optimizer ?? 'adam', params.seed ?? 7);
  const trace = new GradientTraceSinkStation('gradient-trace-sink');
  source.pipe(batcher, VectorSampleSourceStation.CH_SAMPLE, MiniBatchStation.CH_SAMPLE);
  batcher.pipe(learner, MiniBatchStation.CH_BATCH, GradientOptimizerStation.CH_BATCH);
  learner.pipe(trace, GradientOptimizerStation.CH_STEP, GradientTraceSinkStation.CH_STEP);
  runIterativeDES([source, batcher, learner, trace], {shuffle: false, maxTicks: (params.epochs ?? 800) * 5 + 20});
  return gradientTrainingResult(samples, learner, trace, stationGraph([source, batcher, learner, trace], ['VectorSampleToken', 'VectorBatchToken', 'GradientStepToken'], [
    channelEdge(source, VectorSampleSourceStation.CH_SAMPLE, batcher, MiniBatchStation.CH_SAMPLE),
    channelEdge(batcher, MiniBatchStation.CH_BATCH, learner, GradientOptimizerStation.CH_BATCH),
    channelEdge(learner, GradientOptimizerStation.CH_STEP, trace, GradientTraceSinkStation.CH_STEP),
  ]), (parameters, input) => {
    void parameters;
    return learner.predict(input);
  });
}

function gradientTrainingResult(
  samples: readonly VectorSampleToken[],
  learner: GradientOptimizerStation,
  trace: GradientTraceSinkStation,
  topology: StationGraphSummary,
  predict: (parameters: readonly number[], input: readonly number[]) => number,
): GradientTrainingResult {
  const parameters = learner.getParameters();
  const weights = parameters.slice(0, -1);
  const bias = parameters[parameters.length - 1];
  const predictions = samples.map(s => predict(parameters, s.input));
  const accuracy = predictions.filter((p, i) => (p >= 0.5 ? 1 : 0) === samples[i].target[0]).length / Math.max(1, samples.length);
  const lossHistory = learner.getLossHistory();
  return {
    parameters,
    weights,
    bias,
    lossHistory,
    gradientNormHistory: learner.getGradientNormHistory(),
    finalLoss: lossHistory[lossHistory.length - 1] ?? NaN,
    accuracy,
    predictions,
    topology,
  };
}

function logisticPredict(parameters: readonly number[], input: readonly number[]): number {
  return sigmoid(dot(parameters.slice(0, -1), input) + parameters[parameters.length - 1]);
}

function toVectorSamples(samples: readonly SupervisedSample[]): VectorSampleToken[] {
  return samples.map((s, i) => {
    const target = Array.isArray(s.y) ? s.y.slice() : [s.y];
    return new VectorSampleToken(`sample-${i}`, s.x.slice(), target);
  });
}

function solveLinearSystem(a: number[][], b: readonly number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    if (Math.abs(m[pivot][col]) < 1e-12) throw new Error('normal equations are singular; add ridge regularization');
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const div = m[col][col];
    for (let c = col; c <= n; c++) m[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r][col];
      for (let c = col; c <= n; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return m.map(row => row[n]);
}

function defaultRegressionSamples(): SupervisedSample[] {
  return [
    {x: [0], y: 1},
    {x: [1], y: 3},
    {x: [2], y: 5},
    {x: [3], y: 7},
    {x: [4], y: 9},
  ];
}

function defaultLogisticSamples(): SupervisedSample[] {
  return [
    {x: [-2, -1], y: 0},
    {x: [-1, -1], y: 0},
    {x: [-1, 0], y: 0},
    {x: [0, 1], y: 1},
    {x: [1, 1], y: 1},
    {x: [2, 1], y: 1},
  ];
}

function defaultXorSamples(): SupervisedSample[] {
  return [
    {x: [0, 0], y: 0},
    {x: [0, 1], y: 1},
    {x: [1, 0], y: 1},
    {x: [1, 1], y: 0},
  ];
}

export function multiclassAccuracy(logits: readonly number[][], labels: readonly number[]): number {
  let ok = 0;
  for (let i = 0; i < logits.length; i++) {
    const p = softmax(logits[i]);
    let best = 0;
    for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
    if (best === labels[i]) ok += 1;
  }
  return ok / Math.max(1, logits.length);
}
