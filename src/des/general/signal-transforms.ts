'use strict';

// =============================================================================
// Z, Laplace, and Fourier transforms as DES station graphs.
//
// Transform runs are intentionally not monolithic numerical helpers. They are
// built from the entity vocabulary used across the project:
//
//   TransformSampleSourceStation -> TransformKernelStation
//     -> TransformAccumulatorStation -> TransformResultSinkStation
//
// Samples, per-point contributions, and final result envelopes are movable
// Token instances. The stationary entities own only local state and communicate
// through named channels.
// =============================================================================

import {ChannelName, DESStation, Token} from './des-base/station';
import {IterativeRunSummary, runIterativeDES} from './des-base/runner';
import {StationGraphSummary, channelEdge, stationGraph} from './des-base/learning-optimization';
import {ValidationCheck, intrinsicCheck} from './des-base/validation';
import {Preconditions} from './des-base/preconditions';
import {evaluate, parse} from './expr';

export type TransformKind = 'z' | 'laplace' | 'fourier';
export type QuadratureRule = 'rectangular' | 'trapezoid';

export interface ComplexValue {
  re: number;
  im: number;
}

export interface ComplexPointInput {
  label?: string;
  re: number;
  im?: number;
}

export interface ComplexPoint extends ComplexValue {
  label: string;
}

export interface TransformSampleRecord {
  sampleIndex: number;
  abscissaName: 'n' | 't';
  abscissa: number;
  value: number;
  weight: number;
}

export interface TransformContributionRecord {
  sampleIndex: number;
  abscissa: number;
  pointIndex: number;
  pointLabel: string;
  contribution: ComplexValue;
  cumulative: ComplexValue;
}

export interface TransformOutputPoint {
  pointIndex: number;
  label: string;
  point: ComplexValue;
  value: ComplexValue;
  magnitude: number;
  phase: number;
  samplesUsed: number;
  directReference: ComplexValue;
  absoluteError: number;
}

export interface TransformEntityFrameworkSummary {
  sources: string[];
  stations: string[];
  sinks: string[];
  movableEntities: string[];
  edges: string[];
}

export interface TransformRunResult {
  kind: TransformKind;
  convention: string;
  samples: TransformSampleRecord[];
  outputs: TransformOutputPoint[];
  trace: TransformContributionRecord[];
  topology: StationGraphSummary;
  entityFramework: TransformEntityFrameworkSummary;
  runSummary: IterativeRunSummary;
  validation: ValidationCheck[];
}

export interface ZTransformParams {
  sequence?: number[];
  expression?: string;
  constants?: Record<string, number>;
  terms?: number;
  startIndex?: number;
  zValues?: ComplexPointInput[];
  tolerance?: number;
}

export interface LaplaceTransformParams {
  samples?: number[];
  expression?: string;
  constants?: Record<string, number>;
  t0?: number;
  t1?: number;
  dt?: number;
  quadrature?: QuadratureRule;
  sValues?: ComplexPointInput[];
  tolerance?: number;
}

export interface FourierTransformParams {
  samples?: number[];
  expression?: string;
  constants?: Record<string, number>;
  t0?: number;
  t1?: number;
  dt?: number;
  quadrature?: QuadratureRule;
  omegaValues?: number[];
  tolerance?: number;
}

const SAMPLE_CHANNEL: ChannelName = 'transform-sample';
const CONTRIBUTION_CHANNEL: ChannelName = 'transform-contribution';
const RESULT_CHANNEL: ChannelName = 'transform-result';

function complex(re: number, im = 0): ComplexValue {
  return {re, im};
}

function complexAdd(a: ComplexValue, b: ComplexValue): ComplexValue {
  return {re: a.re + b.re, im: a.im + b.im};
}

function complexScale(a: ComplexValue, k: number): ComplexValue {
  return {re: a.re * k, im: a.im * k};
}

function complexExp(re: number, im: number): ComplexValue {
  const mag = Math.exp(re);
  return {re: mag * Math.cos(im), im: mag * Math.sin(im)};
}

function complexMagnitude(a: ComplexValue): number {
  return Math.hypot(a.re, a.im);
}

function complexAbsDiff(a: ComplexValue, b: ComplexValue): number {
  return Math.hypot(a.re - b.re, a.im - b.im);
}

function complexPowInteger(base: ComplexValue, exponent: number): ComplexValue {
  Preconditions.integer('signal-transform', 'integer power exponent', exponent);
  if (exponent === 0) return complex(1, 0);
  const r = complexMagnitude(base);
  if (r === 0) {
    if (exponent < 0) throw new Error('z-transform is undefined at z=0 for positive sequence indices');
    return complex(0, 0);
  }
  const theta = Math.atan2(base.im, base.re);
  const mag = Math.pow(r, exponent);
  return {re: mag * Math.cos(exponent * theta), im: mag * Math.sin(exponent * theta)};
}

export function formatComplex(z: ComplexValue, digits = 6): string {
  const re = z.re.toPrecision(digits);
  const imAbs = Math.abs(z.im).toPrecision(digits);
  const sign = z.im < 0 ? '-' : '+';
  return `${re} ${sign} ${imAbs}i`;
}

function finiteComplex(model: string, param: string, z: ComplexValue): void {
  Preconditions.finite(model, `${param}.re`, z.re);
  Preconditions.finite(model, `${param}.im`, z.im);
}

function finiteConstants(constants: Record<string, number> | undefined): Record<string, number> {
  const out: Record<string, number> = {pi: Math.PI, e: Math.E};
  if (!constants) return out;
  for (const [key, value] of Object.entries(constants)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      throw new Error(`constant name must be an identifier: ${key}`);
    }
    Preconditions.finite('signal-transform', `constants.${key}`, value);
    out[key] = value;
  }
  return out;
}

function normalizeComplexPoints(
  values: readonly ComplexPointInput[] | undefined,
  fallback: readonly ComplexPointInput[],
  pointName: string,
): ComplexPoint[] {
  const raw = values && values.length > 0 ? values : fallback;
  Preconditions.nonEmpty('signal-transform', pointName, raw);
  return raw.map((p, i) => {
    const point = {re: p.re, im: p.im ?? 0, label: p.label ?? `${pointName}[${i}]`};
    finiteComplex('signal-transform', `${pointName}[${i}]`, point);
    return point;
  });
}

function normalizeOmegaPoints(values: readonly number[] | undefined): ComplexPoint[] {
  const raw = values && values.length > 0 ? values : [0];
  Preconditions.nonEmpty('fourier-transform', 'omegaValues', raw);
  return raw.map((omega, i) => {
    Preconditions.finite('fourier-transform', `omegaValues[${i}]`, omega);
    return {re: omega, im: 0, label: `omega=${omega}`};
  });
}

class TransformSampleToken implements Token {
  readonly kind = 'transform-sample';
  constructor(readonly sample: TransformSampleRecord) {}
}

class TransformContributionToken implements Token {
  readonly kind = 'transform-contribution';
  constructor(
    readonly sample: TransformSampleRecord,
    readonly pointIndex: number,
    readonly point: ComplexPoint,
    readonly contribution: ComplexValue,
  ) {}
}

class TransformTotalsToken implements Token {
  readonly kind = 'transform-totals';
  constructor(
    readonly outputs: Omit<TransformOutputPoint, 'directReference' | 'absoluteError'>[],
    readonly trace: TransformContributionRecord[],
  ) {}
}

class TransformSampleSourceStation extends DESStation {
  static readonly CH_SAMPLE = SAMPLE_CHANNEL;
  private index = 0;

  constructor(id: string, private readonly samples: readonly TransformSampleRecord[]) {
    super(id);
    this.addValidator(intrinsicCheck<TransformSampleSourceStation>({
      name: `${id}.emitted-all-samples`,
      group: 'signal-transform',
      predicate: s => s.index === s.samples.length,
      observedFn: s => `${s.index}/${s.samples.length}`,
      expected: 'every input sample emitted exactly once',
    }));
  }

  override assertPreconditions(): void {
    Preconditions.nonEmpty('TransformSampleSourceStation', `${this.id}.samples`, this.samples);
    for (const sample of this.samples) {
      Preconditions.integerInRange('TransformSampleSourceStation', 'sampleIndex', sample.sampleIndex, 0, 1e9);
      Preconditions.finite('TransformSampleSourceStation', 'abscissa', sample.abscissa);
      Preconditions.finite('TransformSampleSourceStation', 'value', sample.value);
      Preconditions.finite('TransformSampleSourceStation', 'weight', sample.weight);
    }
  }

  override hasWork(): boolean {
    return this.index < this.samples.length;
  }

  runTimeStep(): void {
    if (this.index >= this.samples.length) return;
    this.emit(new TransformSampleToken(this.samples[this.index]), TransformSampleSourceStation.CH_SAMPLE);
    this.index += 1;
  }
}

class TransformKernelStation extends DESStation {
  static readonly CH_SAMPLE = SAMPLE_CHANNEL;
  static readonly CH_CONTRIBUTION = CONTRIBUTION_CHANNEL;

  constructor(
    id: string,
    private readonly points: readonly ComplexPoint[],
    private readonly kernel: (sample: TransformSampleRecord, point: ComplexPoint) => ComplexValue,
  ) {
    super(id);
  }

  override hasWork(): boolean {
    return this.inboxSize(TransformKernelStation.CH_SAMPLE) > 0;
  }

  runTimeStep(): void {
    for (const token of this.drain<TransformSampleToken>(TransformKernelStation.CH_SAMPLE)) {
      for (let pointIndex = 0; pointIndex < this.points.length; pointIndex++) {
        const point = this.points[pointIndex];
        const contribution = this.kernel(token.sample, point);
        finiteComplex('TransformKernelStation', `${this.id}.contribution`, contribution);
        this.emit(
          new TransformContributionToken(token.sample, pointIndex, point, contribution),
          TransformKernelStation.CH_CONTRIBUTION,
        );
      }
    }
  }
}

class TransformAccumulatorStation extends DESStation {
  static readonly CH_CONTRIBUTION = CONTRIBUTION_CHANNEL;
  static readonly CH_RESULT = RESULT_CHANNEL;
  private readonly sums: ComplexValue[];
  private readonly counts: number[];
  private readonly trace: TransformContributionRecord[] = [];
  private totalContributions = 0;
  private emitted = false;

  constructor(id: string, private readonly points: readonly ComplexPoint[], private readonly expectedSamples: number) {
    super(id);
    this.sums = points.map(() => complex(0, 0));
    this.counts = points.map(() => 0);
    this.addValidator(intrinsicCheck<TransformAccumulatorStation>({
      name: `${id}.complete-contribution-count`,
      group: 'signal-transform',
      predicate: s => s.counts.every(count => count === s.expectedSamples),
      observedFn: s => s.counts.join(','),
      expected: 'one contribution per sample per evaluation point',
    }));
    this.addValidator(intrinsicCheck<TransformAccumulatorStation>({
      name: `${id}.finite-sums`,
      group: 'signal-transform',
      predicate: s => s.sums.every(sum => Number.isFinite(sum.re) && Number.isFinite(sum.im)),
      expected: 'all accumulated complex sums finite',
    }));
  }

  override hasWork(): boolean {
    return this.inboxSize(TransformAccumulatorStation.CH_CONTRIBUTION) > 0;
  }

  runTimeStep(): void {
    for (const token of this.drain<TransformContributionToken>(TransformAccumulatorStation.CH_CONTRIBUTION)) {
      const pointIndex = token.pointIndex;
      this.sums[pointIndex] = complexAdd(this.sums[pointIndex], token.contribution);
      this.counts[pointIndex] += 1;
      this.totalContributions += 1;
      this.trace.push({
        sampleIndex: token.sample.sampleIndex,
        abscissa: token.sample.abscissa,
        pointIndex,
        pointLabel: token.point.label,
        contribution: token.contribution,
        cumulative: {...this.sums[pointIndex]},
      });
    }
    if (!this.emitted && this.totalContributions === this.expectedSamples * this.points.length) {
      this.emit(new TransformTotalsToken(this.outputs(), this.trace.slice()), TransformAccumulatorStation.CH_RESULT);
      this.emitted = true;
    }
  }

  private outputs(): Omit<TransformOutputPoint, 'directReference' | 'absoluteError'>[] {
    return this.points.map((point, pointIndex) => {
      const value = this.sums[pointIndex];
      return {
        pointIndex,
        label: point.label,
        point: {re: point.re, im: point.im},
        value: {re: value.re, im: value.im},
        magnitude: complexMagnitude(value),
        phase: Math.atan2(value.im, value.re),
        samplesUsed: this.counts[pointIndex],
      };
    });
  }
}

class TransformResultSinkStation extends DESStation {
  static readonly CH_RESULT = RESULT_CHANNEL;
  latest: TransformTotalsToken | undefined;

  constructor(id: string) {
    super(id);
    this.addValidator(intrinsicCheck<TransformResultSinkStation>({
      name: `${id}.received-result`,
      group: 'signal-transform',
      predicate: s => s.latest !== undefined,
      expected: 'one transform result token reaches the sink',
    }));
  }

  override hasWork(): boolean {
    return this.inboxSize(TransformResultSinkStation.CH_RESULT) > 0;
  }

  runTimeStep(): void {
    const tokens = this.drain<TransformTotalsToken>(TransformResultSinkStation.CH_RESULT);
    if (tokens.length > 0) this.latest = tokens[tokens.length - 1];
  }
}

function buildEntityFramework(
  source: DESStation,
  kernel: DESStation,
  accumulator: DESStation,
  sink: DESStation,
): TransformEntityFrameworkSummary {
  const edges = [
    channelEdge(source, SAMPLE_CHANNEL, kernel, SAMPLE_CHANNEL),
    channelEdge(kernel, CONTRIBUTION_CHANNEL, accumulator, CONTRIBUTION_CHANNEL),
    channelEdge(accumulator, RESULT_CHANNEL, sink, RESULT_CHANNEL),
  ];
  return {
    sources: [source.id],
    stations: [kernel.id, accumulator.id],
    sinks: [sink.id],
    movableEntities: ['TransformSampleToken', 'TransformContributionToken', 'TransformTotalsToken'],
    edges,
  };
}

function directTransform(
  samples: readonly TransformSampleRecord[],
  points: readonly ComplexPoint[],
  kernel: (sample: TransformSampleRecord, point: ComplexPoint) => ComplexValue,
): ComplexValue[] {
  const sums = points.map(() => complex(0, 0));
  for (const sample of samples) {
    for (let i = 0; i < points.length; i++) {
      sums[i] = complexAdd(sums[i], kernel(sample, points[i]));
    }
  }
  return sums;
}

function referenceChecks(
  outputs: readonly TransformOutputPoint[],
  tolerance: number,
  kind: TransformKind,
): ValidationCheck[] {
  return outputs.map(output => ({
    name: `${kind}-transform.reference.${output.label}`,
    group: 'signal-transform-reference',
    passed: output.absoluteError <= tolerance,
    observed: formatComplex(output.value),
    expected: formatComplex(output.directReference),
    details: output.absoluteError <= tolerance ? undefined :
      `abs-error=${output.absoluteError.toExponential(3)} > tolerance=${tolerance}`,
  }));
}

function runTransformPipeline(args: {
  kind: TransformKind;
  convention: string;
  samples: readonly TransformSampleRecord[];
  points: readonly ComplexPoint[];
  kernel: (sample: TransformSampleRecord, point: ComplexPoint) => ComplexValue;
  tolerance: number;
}): TransformRunResult {
  const source = new TransformSampleSourceStation(`${args.kind}-sample-source`, args.samples);
  const kernelStation = new TransformKernelStation(`${args.kind}-kernel-station`, args.points, args.kernel);
  const accumulator = new TransformAccumulatorStation(`${args.kind}-accumulator-station`, args.points, args.samples.length);
  const sink = new TransformResultSinkStation(`${args.kind}-result-sink`);

  source.pipe(kernelStation, SAMPLE_CHANNEL, SAMPLE_CHANNEL);
  kernelStation.pipe(accumulator, CONTRIBUTION_CHANNEL, CONTRIBUTION_CHANNEL);
  accumulator.pipe(sink, RESULT_CHANNEL, RESULT_CHANNEL);

  const runSummary = runIterativeDES([source, kernelStation, accumulator, sink], {
    maxTicks: args.samples.length + 10,
    shuffle: false,
  });
  if (!sink.latest) throw new Error(`${args.kind}-transform did not produce a result`);

  const direct = directTransform(args.samples, args.points, args.kernel);
  const outputs: TransformOutputPoint[] = sink.latest.outputs.map(output => {
    const directReference = direct[output.pointIndex];
    const absoluteError = complexAbsDiff(output.value, directReference);
    return {...output, directReference, absoluteError};
  });
  const validation = [...(runSummary.validation ?? []), ...referenceChecks(outputs, args.tolerance, args.kind)];
  const entityFramework = buildEntityFramework(source, kernelStation, accumulator, sink);
  const topology = stationGraph(
    [source, kernelStation, accumulator, sink],
    entityFramework.movableEntities,
    entityFramework.edges,
  );

  return {
    kind: args.kind,
    convention: args.convention,
    samples: args.samples.map(s => ({...s})),
    outputs,
    trace: sink.latest.trace,
    topology,
    entityFramework,
    runSummary,
    validation,
  };
}

function buildZSamples(params: ZTransformParams): TransformSampleRecord[] {
  const startIndex = params.startIndex ?? 0;
  Preconditions.integer('z-transform', 'startIndex', startIndex);
  const sequence = params.sequence && params.sequence.length > 0
    ? params.sequence.slice()
    : buildExpressionSequence(params);
  Preconditions.nonEmpty('z-transform', 'sequence', sequence);
  Preconditions.allFinite('z-transform', 'sequence', sequence);
  return sequence.map((value, sampleIndex) => ({
    sampleIndex,
    abscissaName: 'n' as const,
    abscissa: startIndex + sampleIndex,
    value,
    weight: 1,
  }));
}

function buildExpressionSequence(params: ZTransformParams): number[] {
  if (!params.expression) {
    throw new Error('z-transform requires either a finite sequence or a sequence expression');
  }
  const terms = params.terms ?? 8;
  const startIndex = params.startIndex ?? 0;
  Preconditions.integerInRange('z-transform', 'terms', terms, 1, 1000000);
  const ast = parse(params.expression);
  const constants = finiteConstants(params.constants);
  const values: number[] = [];
  for (let i = 0; i < terms; i++) {
    const n = startIndex + i;
    const value = evaluate(ast, {...constants, n, index: i, tick: i});
    Preconditions.finite('z-transform', `expression[${i}]`, value);
    values.push(value);
  }
  return values;
}

function buildContinuousSamples(
  model: 'laplace-transform' | 'fourier-transform',
  params: LaplaceTransformParams | FourierTransformParams,
): TransformSampleRecord[] {
  const t0 = params.t0 ?? 0;
  const dt = params.dt ?? 0.01;
  const quadrature = params.quadrature ?? 'trapezoid';
  Preconditions.finite(model, 't0', t0);
  Preconditions.positive(model, 'dt', dt);
  if (quadrature !== 'rectangular' && quadrature !== 'trapezoid') {
    throw new Error(`${model}: quadrature must be rectangular or trapezoid`);
  }

  const values = params.samples && params.samples.length > 0
    ? params.samples.slice()
    : buildExpressionSamples(model, params, t0, dt, quadrature);
  Preconditions.nonEmpty(model, 'samples', values);
  Preconditions.allFinite(model, 'samples', values);
  if (quadrature === 'trapezoid') Preconditions.check(model, 'samples.length', 'be at least 2 for trapezoid quadrature', values.length >= 2, values.length);

  return values.map((value, sampleIndex) => ({
    sampleIndex,
    abscissaName: 't' as const,
    abscissa: t0 + sampleIndex * dt,
    value,
    weight: quadratureWeight(sampleIndex, values.length, dt, quadrature),
  }));
}

function buildExpressionSamples(
  model: 'laplace-transform' | 'fourier-transform',
  params: LaplaceTransformParams | FourierTransformParams,
  t0: number,
  dt: number,
  quadrature: QuadratureRule,
): number[] {
  if (!params.expression) {
    throw new Error(`${model} requires either samples or an expression`);
  }
  const t1 = params.t1 ?? 1;
  Preconditions.finite(model, 't1', t1);
  Preconditions.check(model, 't1', 'be greater than t0', t1 > t0, {t0, t1});
  const exactSteps = (t1 - t0) / dt;
  const steps = Math.round(exactSteps);
  Preconditions.check(model, '(t1 - t0) / dt', 'be an integer number of steps', Math.abs(exactSteps - steps) <= 1e-9 * Math.max(1, Math.abs(exactSteps)), exactSteps);
  Preconditions.integerInRange(model, 'steps', steps, 1, 1000000);
  const sampleCount = quadrature === 'trapezoid' ? steps + 1 : steps;
  const ast = parse(params.expression);
  const constants = finiteConstants(params.constants);
  const values: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    const t = t0 + i * dt;
    const value = evaluate(ast, {...constants, t, time: t, tick: i});
    Preconditions.finite(model, `expression[${i}]`, value);
    values.push(value);
  }
  return values;
}

function quadratureWeight(i: number, n: number, dt: number, rule: QuadratureRule): number {
  if (rule === 'rectangular') return dt;
  return i === 0 || i === n - 1 ? 0.5 * dt : dt;
}

function zKernel(sample: TransformSampleRecord, point: ComplexPoint): ComplexValue {
  const zToMinusN = complexPowInteger(point, -sample.abscissa);
  return complexScale(zToMinusN, sample.value);
}

function laplaceKernel(sample: TransformSampleRecord, point: ComplexPoint): ComplexValue {
  return complexScale(
    complexExp(-point.re * sample.abscissa, -point.im * sample.abscissa),
    sample.value * sample.weight,
  );
}

function fourierKernel(sample: TransformSampleRecord, point: ComplexPoint): ComplexValue {
  return complexScale(
    complexExp(0, -point.re * sample.abscissa),
    sample.value * sample.weight,
  );
}

function validateZPoints(samples: readonly TransformSampleRecord[], points: readonly ComplexPoint[]): void {
  const hasPositiveIndex = samples.some(sample => sample.abscissa > 0);
  if (!hasPositiveIndex) return;
  for (const point of points) {
    Preconditions.check('z-transform', `z=${point.label}`, 'be nonzero when any n > 0', complexMagnitude(point) > 0, point);
  }
}

export function runZTransform(params: ZTransformParams): TransformRunResult {
  const samples = buildZSamples(params);
  const points = normalizeComplexPoints(params.zValues, [{label: 'z=1', re: 1, im: 0}], 'zValues');
  validateZPoints(samples, points);
  return runTransformPipeline({
    kind: 'z',
    convention: 'X(z) = sum_n x[n] z^(-n), evaluated over the supplied finite sequence.',
    samples,
    points,
    kernel: zKernel,
    tolerance: params.tolerance ?? 1e-9,
  });
}

export function runLaplaceTransform(params: LaplaceTransformParams): TransformRunResult {
  const samples = buildContinuousSamples('laplace-transform', params);
  const points = normalizeComplexPoints(params.sValues, [{label: 's=1', re: 1, im: 0}], 'sValues');
  return runTransformPipeline({
    kind: 'laplace',
    convention: 'F(s) = integral f(t) exp(-s t) dt, evaluated by weighted sample tokens.',
    samples,
    points,
    kernel: laplaceKernel,
    tolerance: params.tolerance ?? 1e-9,
  });
}

export function runFourierTransform(params: FourierTransformParams): TransformRunResult {
  const samples = buildContinuousSamples('fourier-transform', params);
  const points = normalizeOmegaPoints(params.omegaValues);
  return runTransformPipeline({
    kind: 'fourier',
    convention: 'F(omega) = integral f(t) exp(-i omega t) dt, evaluated by weighted sample tokens.',
    samples,
    points,
    kernel: fourierKernel,
    tolerance: params.tolerance ?? 1e-9,
  });
}
