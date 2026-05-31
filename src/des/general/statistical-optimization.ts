// RUST MIGRATION: Target module `src/des/general/statistical_optimization.rs`.
// RUST MIGRATION: Convert distribution, demand, risk-capacity, SDDP, adaptive-simulation params/results/traces to `serde` structs; family/method unions become enums.
// RUST MIGRATION: Port iteration stations as structs implementing the fixed-point iteration trait, embedding shared station state instead of TS inheritance.
// RUST MIGRATION: Use `HashMap`/`HashSet` for scenario/cut/candidate indexes and typed structs for cuts and adaptive alternatives.
// RUST MIGRATION: Inject RNG for all sampling paths, keep pure fit/scenario/profit helpers as free functions, and return `Result` for invalid distributions, grids, and linear solves.
'use strict';

// =============================================================================
// Statistical + stochastic optimisation extensions.
//
// This module adds the missing layer above the existing two-stage SLP:
//   - distribution fitting (MLE vs method of moments),
//   - multi-stage SLP via an SDDP-style cut station,
//   - CVaR / chance-constrained / DRO-lite scenario optimisation,
//   - adaptive simulation optimisation for policy/capacity candidates.
//
// The implementations are deliberately small and inspectable. They use the
// existing DES base classes, attach validators, expose traces for observability
// and animation, and keep exact grid-DP or brute-force oracles for tests.
// =============================================================================

import {
  FixedPointIterationStation,
  intrinsicCheck,
  monotonicityValidator,
  runResultStation,
  ValidationCheck,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {sampleGamma, samplePoisson} from './random-variables';
import {mulberry32} from './prng';

export interface OptimizationLogger {
  log(event: {kind: string; level?: 'trace' | 'debug' | 'info' | 'warn' | 'error'; [key: string]: unknown}): void;
}

function mean(xs: readonly number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function varianceN(xs: readonly number[]): number {
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) * (x - m), 0) / xs.length;
}

function varianceUnbiased(xs: readonly number[]): number {
  if (xs.length <= 1) return 0;
  const m = mean(xs);
  return xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1);
}

function stddev(xs: readonly number[]): number {
  return Math.sqrt(Math.max(0, varianceUnbiased(xs)));
}

const MAX_GRID_CANDIDATES = 200000;
const MAX_SDDP_GRID_POINTS = 2000;

function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = Math.max(0, Math.min(sorted.length - 1, q * (sorted.length - 1)));
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const w = pos - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function cloneCuts(cuts: readonly Cut[][]): Cut[][] {
  return cuts.map(stage => stage.map(c => ({...c})));
}

function sampleNormal(mu: number, sigma: number, rng: () => number): number {
  const u1 = 1 - rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

function sampleExponential(rate: number, rng: () => number): number {
  const u = 1 - rng();
  return -Math.log(u) / rate;
}

function digamma(x0: number): number {
  let x = x0;
  let result = 0;
  while (x < 7) { result -= 1 / x; x += 1; }
  const inv = 1 / x;
  const inv2 = inv * inv;
  return result + Math.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
}

function trigamma(x0: number): number {
  let x = x0;
  let result = 0;
  while (x < 7) { result += 1 / (x * x); x += 1; }
  const inv = 1 / x;
  const inv2 = inv * inv;
  return result + inv + inv2 / 2 + inv2 * inv / 6 - inv2 * inv2 * inv / 30;
}

// -----------------------------------------------------------------------------
// Distribution fitting
// -----------------------------------------------------------------------------

export type DistributionFamily = 'normal' | 'lognormal' | 'exponential' | 'gamma' | 'poisson' | 'empirical';
export type FitMethod = 'mle' | 'moments';

export interface EmpiricalPoint {
  value: number;
  prob: number;
}

export interface FittedDistribution {
  family: DistributionFamily;
  method: FitMethod;
  params: Record<string, number>;
  logLikelihood: number;
  aic: number;
  mean: number;
  variance: number;
  support: 'real' | 'positive' | 'nonnegative-integer' | 'empirical';
  empirical?: EmpiricalPoint[];
}

export interface DistributionFitParams {
  samples: number[];
  families?: DistributionFamily[];
  methods?: FitMethod[];
}

export interface DistributionFitResult {
  samples: number[];
  sampleMean: number;
  sampleVariance: number;
  fits: FittedDistribution[];
  bestByAIC: FittedDistribution;
  validation: ValidationCheck[];
}

export function fitDistribution(samples: readonly number[], family: DistributionFamily, method: FitMethod): FittedDistribution {
  const cls = 'fitDistribution';
  Preconditions.nonEmpty(cls, 'samples', samples);
  Preconditions.allFinite(cls, 'samples', samples);
  const n = samples.length;
  const m = mean(samples);
  const vN = Math.max(1e-12, varianceN(samples));
  const vU = Math.max(1e-12, varianceUnbiased(samples));

  if (family === 'normal') {
    const sigma2 = method === 'mle' ? vN : vU;
    const sigma = Math.sqrt(sigma2);
    const ll = samples.reduce((s, x) => s - 0.5 * Math.log(2 * Math.PI * sigma2) - (x - m) * (x - m) / (2 * sigma2), 0);
    return {family, method, params: {mu: m, sigma}, logLikelihood: ll, aic: 2 * 2 - 2 * ll, mean: m, variance: sigma2, support: 'real'};
  }

  if (family === 'lognormal') {
    const positive = samples.filter(x => x > 0);
    if (positive.length !== n) throw new Error('lognormal fit requires all samples > 0');
    if (method === 'mle') {
      const logs = samples.map(Math.log);
      const mu = mean(logs);
      const sigma2 = Math.max(1e-12, varianceN(logs));
      const sigma = Math.sqrt(sigma2);
      const ll = samples.reduce((s, x) => s - Math.log(x) - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) - (Math.log(x) - mu) * (Math.log(x) - mu) / (2 * sigma2), 0);
      const mn = Math.exp(mu + sigma2 / 2);
      const vv = (Math.exp(sigma2) - 1) * Math.exp(2 * mu + sigma2);
      return {family, method, params: {muLog: mu, sigmaLog: sigma}, logLikelihood: ll, aic: 2 * 2 - 2 * ll, mean: mn, variance: vv, support: 'positive'};
    }
    const sigma2 = Math.log(1 + vU / Math.max(1e-12, m * m));
    const mu = Math.log(Math.max(1e-12, m)) - sigma2 / 2;
    const sigma = Math.sqrt(Math.max(1e-12, sigma2));
    const ll = samples.reduce((s, x) => s - Math.log(x) - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) - (Math.log(x) - mu) * (Math.log(x) - mu) / (2 * sigma2), 0);
    return {family, method, params: {muLog: mu, sigmaLog: sigma}, logLikelihood: ll, aic: 2 * 2 - 2 * ll, mean: m, variance: vU, support: 'positive'};
  }

  if (family === 'exponential') {
    if (samples.some(x => x < 0)) throw new Error('exponential fit requires samples >= 0');
    const rate = 1 / Math.max(1e-12, m);
    const ll = samples.reduce((s, x) => s + Math.log(rate) - rate * x, 0);
    return {family, method, params: {rate}, logLikelihood: ll, aic: 2 - 2 * ll, mean: 1 / rate, variance: 1 / (rate * rate), support: 'positive'};
  }

  if (family === 'gamma') {
    if (samples.some(x => x <= 0)) throw new Error('gamma fit requires samples > 0');
    let shape = Math.max(1e-6, m * m / (method === 'mle' ? vN : vU));
    if (method === 'mle') {
      const s = Math.log(m) - mean(samples.map(Math.log));
      for (let i = 0; i < 25; i++) {
        const f = Math.log(shape) - digamma(shape) - s;
        const fp = 1 / shape - trigamma(shape);
        const next = shape - f / fp;
        if (!Number.isFinite(next) || next <= 0) break;
        if (Math.abs(next - shape) < 1e-10) { shape = next; break; }
        shape = next;
      }
    }
    const scale = m / shape;
    const ll = samples.reduce((acc, x) => acc + (shape - 1) * Math.log(x) - x / scale - shape * Math.log(scale) - logGamma(shape), 0);
    return {family, method, params: {shape, scale}, logLikelihood: ll, aic: 2 * 2 - 2 * ll, mean: shape * scale, variance: shape * scale * scale, support: 'positive'};
  }

  if (family === 'poisson') {
    if (samples.some(x => x < 0 || !Number.isInteger(x))) throw new Error('poisson fit requires non-negative integer samples');
    const lambda = Math.max(1e-12, m);
    const ll = samples.reduce((s, x) => s + x * Math.log(lambda) - lambda - logFactorial(x), 0);
    return {family, method, params: {lambda}, logLikelihood: ll, aic: 2 - 2 * ll, mean: lambda, variance: lambda, support: 'nonnegative-integer'};
  }

  const counts = new Map<number, number>();
  for (const x of samples) counts.set(x, (counts.get(x) ?? 0) + 1);
  const empirical = [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([value, count]) => ({value, prob: count / n}));
  const ll = samples.reduce((s, x) => s + Math.log(Math.max(1e-12, empirical.find(p => p.value === x)?.prob ?? 0)), 0);
  return {family, method, params: {}, logLikelihood: ll, aic: 2 * empirical.length - 2 * ll, mean: m, variance: vN, support: 'empirical', empirical};
}

function logFactorial(n: number): number {
  if (n < 2) return 0;
  return logGamma(n + 1);
}

function logGamma(z: number): number {
  const p = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.6150291621406, 12.507343278686905, -0.13857109526572012,
    9.984369578019572e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  let x = 0.99999999999980993;
  const zz = z - 1;
  for (let i = 0; i < p.length; i++) x += p[i] / (zz + i + 1);
  const t = zz + p.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (zz + 0.5) * Math.log(t) - t + Math.log(x);
}

function sampleFittedDistributionUnchecked(fit: FittedDistribution, rng: () => number): number {
  switch (fit.family) {
    case 'normal': return sampleNormal(fit.params.mu, fit.params.sigma, rng);
    case 'lognormal': return Math.exp(sampleNormal(fit.params.muLog, fit.params.sigmaLog, rng));
    case 'exponential': return sampleExponential(fit.params.rate, rng);
    case 'gamma': return sampleGamma(fit.params.shape, fit.params.scale, rng);
    case 'poisson': return samplePoisson(fit.params.lambda, rng);
    case 'empirical': {
      const points = fit.empirical ?? [];
      let u = rng();
      for (const p of points) {
        u -= p.prob;
        if (u <= 0) return p.value;
      }
      return points[points.length - 1]?.value ?? 0;
    }
  }
}

export function sampleFittedDistribution(fit: FittedDistribution, rng: () => number): number {
  validateFittedDistribution('sampleFittedDistribution', 'fit', fit);
  return sampleFittedDistributionUnchecked(fit, rng);
}

export class DistributionFitStation extends FixedPointIterationStation<{idx: number; fits: FittedDistribution[]}> {
  private readonly families: DistributionFamily[];
  private readonly methods: FitMethod[];
  readonly errors: string[] = [];

  constructor(readonly params: DistributionFitParams) {
    super('distribution-fit', {tol: 0, maxIter: (params.families?.length ?? 6) * (params.methods?.length ?? 2) + 1});
    this.families = params.families ?? ['normal', 'lognormal', 'exponential', 'gamma', 'poisson', 'empirical'];
    this.methods = params.methods ?? ['mle', 'moments'];
    this.addValidator(intrinsicCheck<DistributionFitStation>({
      name: 'distribution-fit-has-at-least-one-fit',
      predicate: s => s.getCurrent().fits.length > 0,
      expected: 'at least one admissible family/method',
      group: 'distribution-fit',
    }));
    this.addValidator(intrinsicCheck<DistributionFitStation>({
      name: 'distribution-fit-aic-finite',
      predicate: s => s.getCurrent().fits.every(f => Number.isFinite(f.aic)),
      expected: 'finite AIC for all fits',
      group: 'distribution-fit',
    }));
    this.assertPreconditions();
    this.bootstrap();
  }

  override assertPreconditions(): void {
    Preconditions.nonEmpty('DistributionFitStation', 'samples', this.params.samples);
    Preconditions.check('DistributionFitStation', 'samples.length', 'be at least 2', this.params.samples.length >= 2, this.params.samples.length);
    Preconditions.allFinite('DistributionFitStation', 'samples', this.params.samples);
    Preconditions.nonEmpty('DistributionFitStation', 'families', this.families);
    Preconditions.nonEmpty('DistributionFitStation', 'methods', this.methods);
    for (const family of this.families) {
      Preconditions.check(
        'DistributionFitStation',
        'families',
        'contain only supported families',
        ['normal', 'lognormal', 'exponential', 'gamma', 'poisson', 'empirical'].includes(String(family)),
        family,
      );
    }
    for (const method of this.methods) {
      Preconditions.check(
        'DistributionFitStation',
        'methods',
        'contain only supported methods',
        ['mle', 'moments'].includes(String(method)),
        method,
      );
    }
  }

  protected initialState(): {idx: number; fits: FittedDistribution[]} {
    return {idx: 0, fits: []};
  }

  protected applyOperator(prev: {idx: number; fits: FittedDistribution[]}): {idx: number; fits: FittedDistribution[]} {
    const pairs: Array<[DistributionFamily, FitMethod]> = [];
    for (const f of this.families) for (const m of this.methods) pairs.push([f, m]);
    if (prev.idx >= pairs.length) return prev;
    const [family, method] = pairs[prev.idx];
    const fits = prev.fits.slice();
    try {
      fits.push(fitDistribution(this.params.samples, family, method));
    } catch (e) {
      this.errors.push(`${family}/${method}: ${e instanceof Error ? e.message : String(e)}`);
    }
    return {idx: prev.idx + 1, fits};
  }

  protected delta(prev: {idx: number}, next: {idx: number}): number {
    return next.idx === prev.idx ? 0 : 1;
  }

  protected override shouldStop(iter: number): boolean {
    const total = this.families.length * this.methods.length;
    if (this.current && this.current.idx >= total) {
      this.convergenceReason = 'converged';
      return true;
    }
    return super.shouldStop(iter, this.lastDelta);
  }

  result(validation: ValidationCheck[] = []): DistributionFitResult {
    const fits = this.getCurrent().fits.sort((a, b) => a.aic - b.aic);
    if (fits.length === 0) throw new Error(`no distribution fit succeeded: ${this.errors.join('; ')}`);
    return {
      samples: this.params.samples.slice(),
      sampleMean: mean(this.params.samples),
      sampleVariance: varianceUnbiased(this.params.samples),
      fits,
      bestByAIC: fits[0],
      validation,
    };
  }
}

export function runDistributionFit(params: DistributionFitParams): DistributionFitResult {
  const station = new DistributionFitStation(params);
  return runResultStation(station, {shuffle: false});
}

// -----------------------------------------------------------------------------
// Shared capacity-planning scenario utilities.
// -----------------------------------------------------------------------------

export interface DemandRange {
  low: number;
  high: number;
}

export interface DemandSpec {
  kind: 'uniform' | 'fitted' | 'empirical';
  ranges?: DemandRange[];
  fitted?: FittedDistribution[];
  empirical?: EmpiricalPoint[][];
}

export interface DemandScenario {
  demand: number[];
  prob: number;
}

function validateFittedDistribution(model: string, param: string, fit: FittedDistribution): void {
  Preconditions.check(model, `${param}.family`, 'be provided', fit.family !== undefined, fit.family);
  switch (fit.family) {
    case 'normal':
      Preconditions.finite(model, `${param}.params.mu`, fit.params.mu);
      Preconditions.positive(model, `${param}.params.sigma`, fit.params.sigma);
      break;
    case 'lognormal':
      Preconditions.finite(model, `${param}.params.muLog`, fit.params.muLog);
      Preconditions.positive(model, `${param}.params.sigmaLog`, fit.params.sigmaLog);
      break;
    case 'exponential':
      Preconditions.positive(model, `${param}.params.rate`, fit.params.rate);
      break;
    case 'gamma':
      Preconditions.positive(model, `${param}.params.shape`, fit.params.shape);
      Preconditions.positive(model, `${param}.params.scale`, fit.params.scale);
      break;
    case 'poisson':
      Preconditions.nonNegative(model, `${param}.params.lambda`, fit.params.lambda);
      break;
    case 'empirical': {
      const points = fit.empirical ?? [];
      Preconditions.nonEmpty(model, `${param}.empirical`, points);
      Preconditions.allFinite(model, `${param}.empirical.values`, points.map(p => p.value));
      Preconditions.probabilityVector(model, `${param}.empirical.prob`, points.map(p => p.prob));
      break;
    }
  }
}

function validateDemandSpec(model: string, param: string, spec: DemandSpec, nProducts: number): void {
  Preconditions.check(model, param, 'be provided', spec !== undefined && spec !== null, spec);
  Preconditions.integerInRange(model, 'nProducts', nProducts, 1, Number.MAX_SAFE_INTEGER);
  Preconditions.check(model, `${param}.kind`, 'be one of uniform, fitted, empirical', spec.kind === 'uniform' || spec.kind === 'fitted' || spec.kind === 'empirical', spec.kind);
  if (spec.kind === 'uniform') {
    const ranges = spec.ranges ?? [];
    Preconditions.lengthEq(model, `${param}.ranges`, ranges, nProducts);
    for (let i = 0; i < ranges.length; i++) {
      Preconditions.nonNegative(model, `${param}.ranges[${i}].low`, ranges[i].low);
      Preconditions.nonNegative(model, `${param}.ranges[${i}].high`, ranges[i].high);
      Preconditions.check(model, `${param}.ranges[${i}].high`, 'be >= low', ranges[i].high >= ranges[i].low, ranges[i]);
    }
    return;
  }
  if (spec.kind === 'fitted') {
    const fitted = spec.fitted ?? [];
    Preconditions.lengthEq(model, `${param}.fitted`, fitted, nProducts);
    for (let i = 0; i < fitted.length; i++) validateFittedDistribution(model, `${param}.fitted[${i}]`, fitted[i]);
    return;
  }
  const empirical = spec.empirical ?? [];
  Preconditions.lengthEq(model, `${param}.empirical`, empirical, nProducts);
  for (let i = 0; i < empirical.length; i++) {
    Preconditions.nonEmpty(model, `${param}.empirical[${i}]`, empirical[i]);
    Preconditions.allFinite(model, `${param}.empirical[${i}].values`, empirical[i].map(p => p.value));
    Preconditions.probabilityVector(model, `${param}.empirical[${i}].prob`, empirical[i].map(p => p.prob));
  }
}

function sampleDemandVectorUnchecked(spec: DemandSpec, nProducts: number, rng: () => number): number[] {
  if (spec.kind === 'uniform') {
    const ranges = spec.ranges ?? [];
    return ranges.map(r => r.low + rng() * (r.high - r.low));
  }
  if (spec.kind === 'fitted') {
    const fitted = spec.fitted ?? [];
    return fitted.map(f => Math.max(0, sampleFittedDistributionUnchecked(f, rng)));
  }
  const empirical = spec.empirical ?? [];
  return empirical.map(points => {
    let u = rng();
    for (const p of points) {
      u -= p.prob;
      if (u <= 0) return p.value;
    }
    return points[points.length - 1]?.value ?? 0;
  });
}

export function sampleDemandVector(spec: DemandSpec, nProducts: number, rng: () => number): number[] {
  validateDemandSpec('sampleDemandVector', 'spec', spec, nProducts);
  return sampleDemandVectorUnchecked(spec, nProducts, rng);
}

export function buildDemandScenarios(spec: DemandSpec, nProducts: number, N: number, seed: number): DemandScenario[] {
  Preconditions.integerInRange('buildDemandScenarios', 'N', N, 1, Number.MAX_SAFE_INTEGER);
  validateDemandSpec('buildDemandScenarios', 'spec', spec, nProducts);
  const rng = mulberry32(seed);
  const out: DemandScenario[] = [];
  for (let i = 0; i < N; i++) out.push({demand: sampleDemandVectorUnchecked(spec, nProducts, rng), prob: 1 / N});
  return out;
}

export function capacityProfit(x: readonly number[], demand: readonly number[], cost: readonly number[], price: readonly number[]): number {
  let z = 0;
  for (let i = 0; i < x.length; i++) z += -cost[i] * x[i] + price[i] * Math.min(x[i], demand[i]);
  return z;
}

function totalShortfall(x: readonly number[], demand: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.max(0, demand[i] - x[i]);
  return s;
}

function enumerateGrid(n: number, xMax: number, step: number): number[][] {
  const levels: number[] = [];
  for (let x = 0; x <= xMax + 1e-9; x += step) levels.push(Number(x.toFixed(10)));
  const out: number[][] = [];
  const cur = new Array(n).fill(0);
  const rec = (i: number): void => {
    if (i === n) { out.push(cur.slice()); return; }
    for (const v of levels) { cur[i] = v; rec(i + 1); }
  };
  rec(0);
  return out;
}

function gridPointCount(n: number, xMax: number, step: number): number {
  return Math.pow(Math.floor(xMax / step) + 1, n);
}

function riskGridSize(params: RiskCapacityParams): number {
  Preconditions.nonEmpty('RiskCapacityStation', 'cost', params.cost);
  Preconditions.positive('RiskCapacityStation', 'xMax', params.xMax);
  Preconditions.positive('RiskCapacityStation', 'step', params.step);
  const count = gridPointCount(params.cost.length, params.xMax, params.step);
  Preconditions.integerInRange('RiskCapacityStation', 'grid candidate count', count, 1, MAX_GRID_CANDIDATES);
  return count;
}

function sddpGridSize(params: SDDPParams): number {
  Preconditions.positive('CapacityExpansionSDDPStation', 'xMax', params.xMax);
  Preconditions.positive('CapacityExpansionSDDPStation', 'step', params.step);
  const count = Math.floor(params.xMax / params.step) + 1;
  Preconditions.integerInRange('CapacityExpansionSDDPStation', 'grid point count', count, 1, MAX_SDDP_GRID_POINTS);
  return count;
}

function adaptiveMaxIter(params: AdaptiveSimOptParams): number {
  Preconditions.nonEmpty('AdaptiveSimulationOptimizerStation', 'alternatives', params.alternatives);
  Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'batchSize', params.batchSize, 1, Number.MAX_SAFE_INTEGER);
  Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'budget', params.budget, 1, Number.MAX_SAFE_INTEGER);
  return Math.ceil(params.budget / params.batchSize) + params.alternatives.length + 2;
}

// -----------------------------------------------------------------------------
// CVaR / chance / DRO-lite scenario optimisation.
// -----------------------------------------------------------------------------

export interface RiskCapacityParams {
  cost: number[];
  price: number[];
  demand: DemandSpec;
  numScenarios: number;
  seed: number;
  xMax: number;
  step: number;
  risk: {
    kind: 'expectation' | 'cvar' | 'chance' | 'dro';
    alpha?: number;
    lambda?: number;
    minServiceLevel?: number;
    shortfallLimit?: number;
    radius?: number;
  };
}

export interface RiskCandidateResult {
  x: number[];
  meanProfit: number;
  sdProfit: number;
  cvarLoss: number;
  serviceLevel: number;
  robustObjective: number;
  feasible: boolean;
}

export interface RiskCapacityResult {
  params: RiskCapacityParams;
  scenarios: DemandScenario[];
  candidates: RiskCandidateResult[];
  best: RiskCandidateResult;
  validation: ValidationCheck[];
}

export class RiskCapacityStation extends FixedPointIterationStation<{idx: number; candidates: RiskCandidateResult[]}> {
  readonly scenarios: DemandScenario[];
  private readonly grid: number[][];

  constructor(readonly params: RiskCapacityParams) {
    super('risk-capacity', {tol: 0, maxIter: riskGridSize(params) + 1});
    this.assertPreconditions();
    this.scenarios = buildDemandScenarios(params.demand, params.cost.length, params.numScenarios, params.seed);
    this.grid = enumerateGrid(params.cost.length, params.xMax, params.step);
    this.addValidator(intrinsicCheck<RiskCapacityStation>({
      name: 'risk-capacity-evaluated-entire-grid',
      predicate: s => s.getCurrent().candidates.length === s.grid.length,
      observedFn: s => `${s.getCurrent().candidates.length}/${s.grid.length}`,
      expected: 'all grid candidates evaluated',
      group: 'risk-capacity',
    }));
    this.addValidator(intrinsicCheck<RiskCapacityStation>({
      name: 'risk-capacity-best-feasible-if-feasible-exists',
      predicate: s => {
        const cs = s.getCurrent().candidates;
        return !cs.some(c => c.feasible) || s.best().feasible;
      },
      expected: 'best candidate is feasible when any feasible candidate exists',
      group: 'risk-capacity',
    }));
    this.bootstrap();
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.nonEmpty('RiskCapacityStation', 'cost', p.cost);
    Preconditions.lengthEq('RiskCapacityStation', 'price', p.price, p.cost.length);
    Preconditions.allFinite('RiskCapacityStation', 'cost', p.cost);
    Preconditions.allFinite('RiskCapacityStation', 'price', p.price);
    Preconditions.arrNonNegative('RiskCapacityStation', 'cost', p.cost);
    Preconditions.arrNonNegative('RiskCapacityStation', 'price', p.price);
    Preconditions.check('RiskCapacityStation', 'risk', 'be provided', p.risk !== undefined && p.risk !== null, p.risk);
    Preconditions.check('RiskCapacityStation', 'risk.kind', 'be one of expectation, cvar, chance, dro', ['expectation', 'cvar', 'chance', 'dro'].includes(String(p.risk.kind)), p.risk.kind);
    Preconditions.integerInRange('RiskCapacityStation', 'numScenarios', p.numScenarios, 1, Number.MAX_SAFE_INTEGER);
    Preconditions.integer('RiskCapacityStation', 'seed', p.seed);
    Preconditions.positive('RiskCapacityStation', 'xMax', p.xMax);
    Preconditions.positive('RiskCapacityStation', 'step', p.step);
    Preconditions.check('RiskCapacityStation', 'step', 'be <= xMax', p.step <= p.xMax, p.step);
    Preconditions.integerInRange('RiskCapacityStation', 'grid candidate count', this.grid?.length ?? riskGridSize(p), 1, MAX_GRID_CANDIDATES);
    validateDemandSpec('RiskCapacityStation', 'demand', p.demand, p.cost.length);
    if (p.risk.alpha !== undefined) Preconditions.inRange('RiskCapacityStation', 'risk.alpha', p.risk.alpha, 0.5, 0.999);
    if (p.risk.minServiceLevel !== undefined) Preconditions.inRange('RiskCapacityStation', 'risk.minServiceLevel', p.risk.minServiceLevel, 0, 1);
    if (p.risk.shortfallLimit !== undefined) Preconditions.nonNegative('RiskCapacityStation', 'risk.shortfallLimit', p.risk.shortfallLimit);
    if (p.risk.lambda !== undefined) Preconditions.nonNegative('RiskCapacityStation', 'risk.lambda', p.risk.lambda);
    if (p.risk.radius !== undefined) Preconditions.nonNegative('RiskCapacityStation', 'risk.radius', p.risk.radius);
  }

  protected initialState(): {idx: number; candidates: RiskCandidateResult[]} {
    return {idx: 0, candidates: []};
  }

  protected applyOperator(prev: {idx: number; candidates: RiskCandidateResult[]}): {idx: number; candidates: RiskCandidateResult[]} {
    if (prev.idx >= this.grid.length) return prev;
    const candidates = prev.candidates.slice();
    candidates.push(this.evaluate(this.grid[prev.idx]));
    return {idx: prev.idx + 1, candidates};
  }

  protected delta(prev: {idx: number}, next: {idx: number}): number {
    return next.idx === prev.idx ? 0 : 1;
  }

  protected override shouldStop(iter: number): boolean {
    if (this.current && this.current.idx >= this.grid.length) {
      this.convergenceReason = 'converged';
      return true;
    }
    return super.shouldStop(iter, this.lastDelta);
  }

  best(): RiskCandidateResult {
    const cs = this.getCurrent().candidates;
    const feasible = cs.filter(c => c.feasible);
    const pool = feasible.length > 0 ? feasible : cs;
    return pool.reduce((a, b) => b.robustObjective > a.robustObjective ? b : a, pool[0]);
  }

  result(validation: ValidationCheck[] = []): RiskCapacityResult {
    return {
      params: this.params,
      scenarios: this.scenarios,
      candidates: this.getCurrent().candidates,
      best: this.best(),
      validation,
    };
  }

  private evaluate(x: number[]): RiskCandidateResult {
    const profits = this.scenarios.map(s => capacityProfit(x, s.demand, this.params.cost, this.params.price));
    const shortfalls = this.scenarios.map(s => totalShortfall(x, s.demand));
    const losses = shortfalls.slice().sort((a, b) => a - b);
    const alpha = this.params.risk.alpha ?? 0.9;
    const varLoss = quantileSorted(losses, alpha);
    const tail = losses.filter(l => l >= varLoss - 1e-12);
    const cvarLoss = tail.length === 0 ? varLoss : mean(tail);
    const meanProfit = mean(profits);
    const sdProfit = stddev(profits);
    const shortfallLimit = this.params.risk.shortfallLimit ?? 0;
    const serviceLevel = shortfalls.filter(s => s <= shortfallLimit + 1e-12).length / shortfalls.length;
    const minSL = this.params.risk.minServiceLevel ?? 0;
    const feasible = this.params.risk.kind !== 'chance' || serviceLevel >= minSL - 1e-12;
    let robustObjective = meanProfit;
    if (this.params.risk.kind === 'cvar') robustObjective = meanProfit - (this.params.risk.lambda ?? 1) * cvarLoss;
    if (this.params.risk.kind === 'dro') robustObjective = meanProfit - (this.params.risk.radius ?? 1) * sdProfit;
    if (this.params.risk.kind === 'chance' && !feasible) robustObjective = meanProfit - 1e6 * (minSL - serviceLevel);
    return {x, meanProfit, sdProfit, cvarLoss, serviceLevel, robustObjective, feasible};
  }
}

export function runRiskCapacity(params: RiskCapacityParams): RiskCapacityResult {
  const st = new RiskCapacityStation(params);
  return runResultStation(st, {shuffle: false});
}

// -----------------------------------------------------------------------------
// Multi-stage SDDP-style capacity expansion.
// -----------------------------------------------------------------------------

interface Cut {
  slope: number;
  intercept: number;
  stage: number;
  at: number;
  value: number;
}

export interface SDDPParams {
  horizon: number;
  demand: DemandRange[];
  price: number[];
  expansionCost: number[];
  initialCapacity: number;
  xMax: number;
  step: number;
  samplesPerStage: number;
  seed: number;
  maxIter?: number;
  tol?: number;
}

export interface SDDPIteration {
  iter: number;
  upperBound: number;
  lowerBound: number;
  exactObjective: number;
  gap: number;
  cutCounts: number[];
  forwardStates: number[];
  forwardReturn: number;
}

export interface SDDPResult {
  params: SDDPParams;
  exactObjective: number;
  exactPolicy: number[][];
  finalUpperBound: number;
  finalLowerBound: number;
  gap: number;
  cuts: Cut[][];
  trace: SDDPIteration[];
  validation: ValidationCheck[];
}

interface SDDPState {
  iter: number;
  cuts: Cut[][];
  upperBound: number;
  lowerBound: number;
  forwardStates: number[];
  forwardReturn: number;
}

export class CapacityExpansionSDDPStation extends FixedPointIterationStation<SDDPState> {
  private readonly grid: number[];
  private readonly scenarios: number[][];
  private readonly exact: {objective: number; policy: number[][]};
  readonly trace: SDDPIteration[] = [];
  readonly upperHistory: number[] = [];

  constructor(readonly params: SDDPParams, private readonly logger?: OptimizationLogger) {
    super('sddp-capacity', {tol: params.tol ?? 1e-4, maxIter: params.maxIter ?? 60});
    this.assertPreconditions();
    this.grid = [];
    for (let x = 0; x <= params.xMax + 1e-9; x += params.step) this.grid.push(Number(x.toFixed(10)));
    this.scenarios = this.buildStageScenarios();
    this.exact = this.solveExactDP();
    this.addValidator(intrinsicCheck<CapacityExpansionSDDPStation>({
      name: 'sddp-upper-bound-dominates-exact',
      predicate: s => s.getCurrent().upperBound + 1e-6 >= s.exact.objective,
      observedFn: s => `${s.getCurrent().upperBound.toFixed(6)} vs exact ${s.exact.objective.toFixed(6)}`,
      expected: 'upper bound >= exact sampled-grid objective',
      group: 'sddp',
    }));
    this.addValidator(intrinsicCheck<CapacityExpansionSDDPStation>({
      name: 'sddp-lower-bound-no-better-than-exact',
      predicate: s => s.getCurrent().lowerBound <= s.exact.objective + 1e-6,
      observedFn: s => `${s.getCurrent().lowerBound.toFixed(6)} vs exact ${s.exact.objective.toFixed(6)}`,
      expected: 'policy lower bound <= exact objective',
      group: 'sddp',
    }));
    this.addValidator(monotonicityValidator<CapacityExpansionSDDPStation>({
      name: 'sddp-upper-history-non-increasing',
      extract: s => s.upperHistory,
      direction: 'non-increasing',
      tol: 1e-8,
      group: 'sddp',
    }));
    this.bootstrap();
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.integerInRange('CapacityExpansionSDDPStation', 'horizon', p.horizon, 1, 200);
    Preconditions.lengthEq('CapacityExpansionSDDPStation', 'demand', p.demand, p.horizon);
    Preconditions.lengthEq('CapacityExpansionSDDPStation', 'price', p.price, p.horizon);
    Preconditions.lengthEq('CapacityExpansionSDDPStation', 'expansionCost', p.expansionCost, p.horizon);
    Preconditions.nonNegative('CapacityExpansionSDDPStation', 'initialCapacity', p.initialCapacity);
    Preconditions.positive('CapacityExpansionSDDPStation', 'xMax', p.xMax);
    Preconditions.positive('CapacityExpansionSDDPStation', 'step', p.step);
    sddpGridSize(p);
    Preconditions.integerInRange('CapacityExpansionSDDPStation', 'samplesPerStage', p.samplesPerStage, 1, 1000000);
    Preconditions.integer('CapacityExpansionSDDPStation', 'seed', p.seed);
    if (p.maxIter !== undefined) Preconditions.integerInRange('CapacityExpansionSDDPStation', 'maxIter', p.maxIter, 1, Number.MAX_SAFE_INTEGER);
    if (p.tol !== undefined) Preconditions.nonNegative('CapacityExpansionSDDPStation', 'tol', p.tol);
    Preconditions.check('CapacityExpansionSDDPStation', 'initialCapacity', 'be <= xMax', p.initialCapacity <= p.xMax, p.initialCapacity);
    for (let t = 0; t < p.horizon; t++) {
      Preconditions.nonNegative('CapacityExpansionSDDPStation', `demand[${t}].low`, p.demand[t].low);
      Preconditions.nonNegative('CapacityExpansionSDDPStation', `demand[${t}].high`, p.demand[t].high);
      Preconditions.check('CapacityExpansionSDDPStation', `demand[${t}].high`, 'be >= low', p.demand[t].high >= p.demand[t].low, p.demand[t]);
      Preconditions.nonNegative('CapacityExpansionSDDPStation', `price[${t}]`, p.price[t]);
      Preconditions.nonNegative('CapacityExpansionSDDPStation', `expansionCost[${t}]`, p.expansionCost[t]);
    }
  }

  protected initialState(): SDDPState {
    const cuts: Cut[][] = [];
    for (let t = 0; t < this.params.horizon; t++) {
      const upper = this.remainingRevenueUpper(t);
      cuts.push([{slope: 0, intercept: upper, stage: t, at: 0, value: upper}]);
    }
    const upperBound = this.vhat(cuts, 0, this.params.initialCapacity);
    const lowerBound = this.evaluateGreedyPolicy(cuts);
    return {iter: 0, cuts, upperBound, lowerBound, forwardStates: [this.params.initialCapacity], forwardReturn: 0};
  }

  protected applyOperator(prev: SDDPState): SDDPState {
    const cuts = cloneCuts(prev.cuts);
    const forward = this.forwardPass(cuts, prev.iter);
    for (let t = this.params.horizon - 1; t >= 0; t--) {
      const x = forward.states[t];
      const cut = this.makeCut(cuts, t, x);
      cuts[t].push(cut);
      if (cuts[t].length > 80) cuts[t].splice(1, cuts[t].length - 80);
      this.logger?.log({kind: 'sddp-cut', level: 'debug', iter: prev.iter + 1, stage: t, at: x, slope: cut.slope, intercept: cut.intercept});
    }
    const upperBound = this.vhat(cuts, 0, this.params.initialCapacity);
    const lowerBound = this.evaluateGreedyPolicy(cuts);
    const next: SDDPState = {
      iter: prev.iter + 1,
      cuts,
      upperBound,
      lowerBound,
      forwardStates: forward.states,
      forwardReturn: forward.total,
    };
    const traceRow: SDDPIteration = {
      iter: next.iter,
      upperBound,
      lowerBound,
      exactObjective: this.exact.objective,
      gap: upperBound - lowerBound,
      cutCounts: cuts.map(c => c.length),
      forwardStates: forward.states,
      forwardReturn: forward.total,
    };
    this.trace.push(traceRow);
    this.upperHistory.push(upperBound);
    this.logger?.log({kind: 'sddp-iteration', level: 'info', ...traceRow});
    return next;
  }

  protected delta(prev: SDDPState, next: SDDPState): number {
    return Math.abs(prev.upperBound - next.upperBound);
  }

  protected override shouldStop(iter: number, lastDelta: number): boolean {
    if (this.current && iter > 0 && this.current.upperBound - this.current.lowerBound <= this.tol) {
      this.convergenceReason = 'converged';
      return true;
    }
    return super.shouldStop(iter, lastDelta);
  }

  result(validation: ValidationCheck[] = []): SDDPResult {
    const cur = this.getCurrent();
    return {
      params: this.params,
      exactObjective: this.exact.objective,
      exactPolicy: this.exact.policy,
      finalUpperBound: cur.upperBound,
      finalLowerBound: cur.lowerBound,
      gap: cur.upperBound - cur.lowerBound,
      cuts: cur.cuts,
      trace: this.trace,
      validation,
    };
  }

  private buildStageScenarios(): number[][] {
    const rng = mulberry32(this.params.seed);
    return this.params.demand.map(r => {
      const xs: number[] = [];
      for (let i = 0; i < this.params.samplesPerStage; i++) xs.push(r.low + rng() * (r.high - r.low));
      return xs;
    });
  }

  private remainingRevenueUpper(t0: number): number {
    let s = 0;
    for (let t = t0; t < this.params.horizon; t++) s += this.params.price[t] * this.params.demand[t].high;
    return s;
  }

  private idx(x: number): number {
    const k = Math.round(x / this.params.step);
    return Math.max(0, Math.min(this.grid.length - 1, k));
  }

  private vhat(cuts: readonly Cut[][], stage: number, x: number): number {
    if (stage >= this.params.horizon) return 0;
    return Math.min(...cuts[stage].map(c => c.slope * x + c.intercept));
  }

  private bellmanApprox(cuts: readonly Cut[][], stage: number, x: number): number {
    const p = this.params;
    let best = -Infinity;
    for (const xNext of this.grid) {
      if (xNext + 1e-9 < x) continue;
      let q = -p.expansionCost[stage] * (xNext - x);
      let rev = 0;
      for (const d of this.scenarios[stage]) rev += p.price[stage] * Math.min(xNext, d);
      q += rev / this.scenarios[stage].length + this.vhat(cuts, stage + 1, xNext);
      if (q > best) best = q;
    }
    return best;
  }

  private makeCut(cuts: readonly Cut[][], stage: number, x: number): Cut {
    const h = this.params.step;
    const xLo = Math.max(0, x - h);
    const xHi = Math.min(this.params.xMax, x + h);
    const val = this.bellmanApprox(cuts, stage, x);
    const lo = this.bellmanApprox(cuts, stage, xLo);
    const hi = this.bellmanApprox(cuts, stage, xHi);
    const slope = xHi === xLo ? 0 : (hi - lo) / (xHi - xLo);
    return {slope, intercept: val - slope * x, stage, at: x, value: val};
  }

  private chooseNext(cuts: readonly Cut[][], stage: number, x: number, demand: number | null): number {
    const p = this.params;
    let bestX = x;
    let best = -Infinity;
    for (const xNext of this.grid) {
      if (xNext + 1e-9 < x) continue;
      let revenue: number;
      if (demand === null) {
        revenue = this.scenarios[stage].reduce((s, d) => s + p.price[stage] * Math.min(xNext, d), 0) / this.scenarios[stage].length;
      } else {
        revenue = p.price[stage] * Math.min(xNext, demand);
      }
      const q = -p.expansionCost[stage] * (xNext - x) + revenue + this.vhat(cuts, stage + 1, xNext);
      if (q > best + 1e-12) { best = q; bestX = xNext; }
    }
    return bestX;
  }

  private forwardPass(cuts: readonly Cut[][], iter: number): {states: number[]; total: number} {
    const rng = mulberry32(this.params.seed + 1000003 * (iter + 1));
    const states = [this.params.initialCapacity];
    let x = this.params.initialCapacity;
    let total = 0;
    for (let t = 0; t < this.params.horizon; t++) {
      const r = this.params.demand[t];
      const d = r.low + rng() * (r.high - r.low);
      const xNext = this.chooseNext(cuts, t, x, d);
      total += -this.params.expansionCost[t] * (xNext - x) + this.params.price[t] * Math.min(xNext, d);
      x = xNext;
      states.push(x);
    }
    return {states, total};
  }

  private evaluateGreedyPolicy(cuts: readonly Cut[][]): number {
    const T = this.params.horizon;
    let next = new Array(this.grid.length).fill(0);
    for (let t = T - 1; t >= 0; t--) {
      const cur = new Array(this.grid.length).fill(0);
      for (let i = 0; i < this.grid.length; i++) {
        const x = this.grid[i];
        const xNext = this.chooseNext(cuts, t, x, null);
        const j = this.idx(xNext);
        const revenue = this.scenarios[t].reduce((s, d) => s + this.params.price[t] * Math.min(xNext, d), 0) / this.scenarios[t].length;
        cur[i] = -this.params.expansionCost[t] * (xNext - x) + revenue + next[j];
      }
      next = cur;
    }
    return next[this.idx(this.params.initialCapacity)];
  }

  private solveExactDP(): {objective: number; policy: number[][]} {
    const T = this.params.horizon;
    let next = new Array(this.grid.length).fill(0);
    const policy: number[][] = Array.from({length: T}, () => new Array(this.grid.length).fill(0));
    for (let t = T - 1; t >= 0; t--) {
      const cur = new Array(this.grid.length).fill(0);
      for (let i = 0; i < this.grid.length; i++) {
        const x = this.grid[i];
        let best = -Infinity, bestX = x;
        for (const xNext of this.grid) {
          if (xNext + 1e-9 < x) continue;
          const revenue = this.scenarios[t].reduce((s, d) => s + this.params.price[t] * Math.min(xNext, d), 0) / this.scenarios[t].length;
          const q = -this.params.expansionCost[t] * (xNext - x) + revenue + next[this.idx(xNext)];
          if (q > best) { best = q; bestX = xNext; }
        }
        cur[i] = best;
        policy[t][i] = bestX;
      }
      next = cur;
    }
    return {objective: next[this.idx(this.params.initialCapacity)], policy};
  }
}

export function runCapacityExpansionSDDP(params: SDDPParams, logger?: OptimizationLogger): SDDPResult {
  const st = new CapacityExpansionSDDPStation(params, logger);
  return runResultStation(st, {shuffle: false});
}

// -----------------------------------------------------------------------------
// Adaptive simulation optimisation.
// -----------------------------------------------------------------------------

export interface AdaptiveAlternative {
  name: string;
  x: number[];
}

export interface AdaptiveSimOptParams {
  cost: number[];
  price: number[];
  demand: DemandSpec;
  alternatives: AdaptiveAlternative[];
  seed: number;
  initialSamples: number;
  budget: number;
  batchSize: number;
  exploration: number;
}

export interface AlternativeStats {
  name: string;
  x: number[];
  n: number;
  mean: number;
  m2: number;
  sd: number;
  stderr: number;
  ucb: number;
}

export interface AdaptiveTraceRow {
  iter: number;
  sampled: string;
  totalSamples: number;
  bestName: string;
  bestMean: number;
  maxStderr: number;
}

export interface AdaptiveSimOptResult {
  params: AdaptiveSimOptParams;
  stats: AlternativeStats[];
  best: AlternativeStats;
  trace: AdaptiveTraceRow[];
  validation: ValidationCheck[];
}

interface AdaptiveState {
  iter: number;
  stats: AlternativeStats[];
  totalSamples: number;
  trace: AdaptiveTraceRow[];
}

export class AdaptiveSimulationOptimizerStation extends FixedPointIterationStation<AdaptiveState> {
  private rng: () => number;

  constructor(readonly params: AdaptiveSimOptParams, private readonly logger?: OptimizationLogger) {
    super('adaptive-simopt', {tol: 0, maxIter: adaptiveMaxIter(params)});
    this.assertPreconditions();
    this.rng = mulberry32(params.seed);
    this.addValidator(intrinsicCheck<AdaptiveSimulationOptimizerStation>({
      name: 'adaptive-budget-respected',
      predicate: s => s.getCurrent().totalSamples >= s.params.budget,
      observedFn: s => String(s.getCurrent().totalSamples),
      expected: 'totalSamples >= budget',
      group: 'adaptive-simopt',
    }));
    this.addValidator(intrinsicCheck<AdaptiveSimulationOptimizerStation>({
      name: 'adaptive-all-alternatives-sampled',
      predicate: s => s.getCurrent().stats.every(a => a.n >= s.params.initialSamples),
      expected: 'each alternative has initialSamples',
      group: 'adaptive-simopt',
    }));
    this.bootstrap();
  }

  override assertPreconditions(): void {
    const p = this.params;
    Preconditions.nonEmpty('AdaptiveSimulationOptimizerStation', 'alternatives', p.alternatives);
    Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'alternatives.length', p.alternatives.length, 2, Number.MAX_SAFE_INTEGER);
    Preconditions.nonEmpty('AdaptiveSimulationOptimizerStation', 'cost', p.cost);
    Preconditions.lengthEq('AdaptiveSimulationOptimizerStation', 'price', p.price, p.cost.length);
    Preconditions.arrNonNegative('AdaptiveSimulationOptimizerStation', 'cost', p.cost);
    Preconditions.arrNonNegative('AdaptiveSimulationOptimizerStation', 'price', p.price);
    validateDemandSpec('AdaptiveSimulationOptimizerStation', 'demand', p.demand, p.cost.length);
    Preconditions.integer('AdaptiveSimulationOptimizerStation', 'seed', p.seed);
    Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'initialSamples', p.initialSamples, 1, Number.MAX_SAFE_INTEGER);
    Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'budget', p.budget, p.alternatives.length * p.initialSamples, Number.MAX_SAFE_INTEGER);
    Preconditions.integerInRange('AdaptiveSimulationOptimizerStation', 'batchSize', p.batchSize, 1, Number.MAX_SAFE_INTEGER);
    Preconditions.nonNegative('AdaptiveSimulationOptimizerStation', 'exploration', p.exploration);
    const names = new Set<string>();
    for (let i = 0; i < p.alternatives.length; i++) {
      const a = p.alternatives[i];
      Preconditions.check('AdaptiveSimulationOptimizerStation', `alternatives[${i}].name`, 'be a non-empty string', typeof a.name === 'string' && a.name.trim().length > 0, a.name);
      Preconditions.check('AdaptiveSimulationOptimizerStation', `alternatives[${i}].name`, 'be unique', !names.has(a.name), a.name);
      names.add(a.name);
      Preconditions.lengthEq('AdaptiveSimulationOptimizerStation', `alternative.${a.name}.x`, a.x, p.cost.length);
      Preconditions.arrNonNegative('AdaptiveSimulationOptimizerStation', `alternative.${a.name}.x`, a.x);
    }
  }

  protected initialState(): AdaptiveState {
    const stats = this.params.alternatives.map(a => ({name: a.name, x: a.x.slice(), n: 0, mean: 0, m2: 0, sd: 0, stderr: Infinity, ucb: Infinity}));
    let totalSamples = 0;
    for (const st of stats) {
      for (let i = 0; i < this.params.initialSamples; i++) {
        this.sampleInto(st);
        totalSamples++;
      }
    }
    return {iter: 0, stats, totalSamples, trace: []};
  }

  protected applyOperator(prev: AdaptiveState): AdaptiveState {
    const stats = prev.stats.map(s => ({...s, x: s.x.slice()}));
    const chosen = stats.reduce((a, b) => b.ucb > a.ucb ? b : a, stats[0]);
    let totalSamples = prev.totalSamples;
    const reps = Math.min(this.params.batchSize, Math.max(0, this.params.budget - totalSamples));
    for (let i = 0; i < reps; i++) {
      this.sampleInto(chosen);
      totalSamples++;
    }
    const best = stats.reduce((a, b) => b.mean > a.mean ? b : a, stats[0]);
    const row: AdaptiveTraceRow = {
      iter: prev.iter + 1,
      sampled: chosen.name,
      totalSamples,
      bestName: best.name,
      bestMean: best.mean,
      maxStderr: Math.max(...stats.map(s => Number.isFinite(s.stderr) ? s.stderr : 0)),
    };
    const trace = prev.trace.concat(row);
    this.logger?.log({kind: 'adaptive-simopt-iteration', level: 'info', ...row});
    return {iter: prev.iter + 1, stats, totalSamples, trace};
  }

  protected delta(prev: AdaptiveState, next: AdaptiveState): number {
    return Math.abs(prev.totalSamples - next.totalSamples);
  }

  protected override shouldStop(iter: number, lastDelta: number): boolean {
    if (this.current && this.current.totalSamples >= this.params.budget) {
      this.convergenceReason = 'converged';
      return true;
    }
    return super.shouldStop(iter, lastDelta);
  }

  result(validation: ValidationCheck[] = []): AdaptiveSimOptResult {
    const stats = this.getCurrent().stats;
    const best = stats.reduce((a, b) => b.mean > a.mean ? b : a, stats[0]);
    return {params: this.params, stats, best, trace: this.getCurrent().trace, validation};
  }

  private sampleInto(st: AlternativeStats): void {
    const d = sampleDemandVectorUnchecked(this.params.demand, this.params.cost.length, this.rng);
    const z = capacityProfit(st.x, d, this.params.cost, this.params.price);
    st.n += 1;
    const delta = z - st.mean;
    st.mean += delta / st.n;
    st.m2 += delta * (z - st.mean);
    st.sd = st.n > 1 ? Math.sqrt(Math.max(0, st.m2 / (st.n - 1))) : 0;
    st.stderr = st.n > 1 ? st.sd / Math.sqrt(st.n) : Infinity;
    st.ucb = st.mean + this.params.exploration * (Number.isFinite(st.stderr) ? st.stderr : 1e9);
  }
}

export function runAdaptiveSimOpt(params: AdaptiveSimOptParams, logger?: OptimizationLogger): AdaptiveSimOptResult {
  const st = new AdaptiveSimulationOptimizerStation(params, logger);
  return runResultStation(st, {shuffle: false});
}
