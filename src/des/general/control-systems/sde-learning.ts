// RUST MIGRATION: Target module `src/des/general/control_systems/sde_learning.rs`.
// RUST MIGRATION: Convert SDE families, MLE, EnKF, MLP, diffusion model, and estimator station into structs plus traits for estimators/controllers.
// RUST MIGRATION: Use `f64` vectors/matrices for trajectories, ensembles, weights, and gradients; inject RNG/config explicitly.
// RUST MIGRATION: Graph-visible pure estimators should follow PureTransform-style `transform` methods and return `Result` for fit/filter failures.
'use strict';

// =============================================================================
// control-systems/sde-learning.ts — THREE machine-learning algorithms for
// stochastic differential equations, one per classic ML paradigm:
//
//   ML-1  SYSTEM IDENTIFICATION  — `SdeMaximumLikelihoodEstimator`
//         Learn the drift/diffusion parameters of an SDE from one observed
//         sample path by gradient (Adam) ascent on the Euler–Maruyama
//         transition log-likelihood. (supervised / MLE)
//
//   ML-2  FILTERING / INFERENCE  — `EnsembleKalmanFilter` (+ DES station)
//         Track the posterior over the hidden state online from noisy
//         observations using a Monte-Carlo ensemble and the Kalman analysis
//         update. (sequential Bayesian estimation)
//
//   ML-3  GENERATIVE MODELING    — `DenoisingDiffusionModel` (+ tiny `Mlp`)
//         Learn the score of a data distribution by denoising score matching
//         and draw new samples by integrating the REVERSE-TIME SDE (the DDPM
//         discretisation of the variance-preserving SDE). (generative)
//
// Class-only: LinAlg / MatrixInverse for algebra, Mulberry32 for randomness.
// =============================================================================

import {ChannelName} from '../des-base/station';
import {MemoryTransformEntity, TransformContext} from '../des-base/transform-entity';
import {Preconditions} from '../des-base/preconditions';
import {LinAlg, Mat, MatrixInverse, Vec} from './linear-algebra';
import {Mulberry32} from './empirical-control';
import {
  EulerMaruyamaIntegrator,
  GeometricBrownianMotion,
  OrnsteinUhlenbeck,
  SdeChannels,
  SdeEstimateToken,
  SdeObservationToken,
  SdeSystem,
} from './stochastic-sde';

// =============================================================================
// ML-1. SDE PARAMETER ESTIMATION (maximum likelihood, Adam gradient ascent)
// =============================================================================

/** A parametric SDE family: maps an UNCONSTRAINED parameter vector θ to a
 *  concrete `SdeSystem`, so the optimiser can search ℝ^k freely (positivity is
 *  enforced through exp() reparametrisations inside `instantiate`). */
export abstract class ParametricSdeFamily {
  abstract name(): string;
  abstract paramDim(): number;
  abstract initialGuess(): Vec;
  abstract instantiate(theta: Vec): SdeSystem;
  /** Human-readable named parameters for reporting. */
  abstract describe(theta: Vec): Record<string, number>;
}

/** GBM family θ = [μ, log σ]. */
export class GbmFamily extends ParametricSdeFamily {
  name(): string { return 'GBM'; }
  paramDim(): number { return 2; }
  initialGuess(): Vec { return [0, Math.log(0.1)]; }
  instantiate(theta: Vec): SdeSystem { return new GeometricBrownianMotion(theta[0], Math.exp(theta[1])); }
  describe(theta: Vec): Record<string, number> { return {mu: theta[0], sigma: Math.exp(theta[1])}; }
}

/** OU family θ = [log θ, μ, log σ]. */
export class OuFamily extends ParametricSdeFamily {
  name(): string { return 'OU'; }
  paramDim(): number { return 3; }
  initialGuess(): Vec { return [Math.log(0.5), 0, Math.log(0.5)]; }
  instantiate(theta: Vec): SdeSystem { return new OrnsteinUhlenbeck(Math.exp(theta[0]), theta[1], Math.exp(theta[2])); }
  describe(theta: Vec): Record<string, number> { return {theta: Math.exp(theta[0]), mu: theta[1], sigma: Math.exp(theta[2])}; }
}

export interface MleFitResult {
  theta: Vec;
  params: Record<string, number>;
  system: SdeSystem;
  finalNegLogLik: number;
  iterations: number;
}

/** Maximum-likelihood estimator for a parametric SDE family from one path. */
export class SdeMaximumLikelihoodEstimator {
  constructor(private readonly opts: {iterations?: number; learningRate?: number; fdEps?: number} = {}) {}

  /** Euler–Maruyama transition negative log-likelihood of `path` under θ. */
  negLogLikelihood(family: ParametricSdeFamily, theta: Vec, times: readonly number[], path: readonly Vec[]): number {
    const sys = family.instantiate(theta);
    const n = sys.dimension();
    let nll = 0;
    for (let k = 0; k < path.length - 1; k++) {
      const dt = times[k + 1] - times[k];
      const f = sys.drift(times[k], path[k]);
      const g = sys.diffusion(times[k], path[k]);
      const cov = LinAlg.scale(LinAlg.matMul(g, LinAlg.transpose(g)), dt);   // n×n
      // regularise to stay invertible.
      for (let i = 0; i < n; i++) cov[i][i] += 1e-12;
      const r = new Array<number>(n);
      for (let i = 0; i < n; i++) r[i] = path[k + 1][i] - (path[k][i] + f[i] * dt);
      const inv = new MatrixInverse(cov).inverse();
      const cinvr = LinAlg.matVec(inv, r);
      let quad = 0;
      for (let i = 0; i < n; i++) quad += r[i] * cinvr[i];
      nll += 0.5 * (n * Math.log(2 * Math.PI) + Math.log(Math.max(this.determinant(cov), 1e-300)) + quad);
    }
    return nll;
  }

  /** Fit θ by Adam descent on the transition NLL (central-difference grad). */
  fit(family: ParametricSdeFamily, times: readonly number[], path: readonly Vec[]): MleFitResult {
    Preconditions.nonEmpty('SdeMaximumLikelihoodEstimator', 'path', path);
    const iters = this.opts.iterations ?? 1500;
    const lr = this.opts.learningRate ?? 0.05;
    const eps = this.opts.fdEps ?? 1e-4;
    const d = family.paramDim();
    let theta = family.initialGuess();
    const m = new Array<number>(d).fill(0);
    const v = new Array<number>(d).fill(0);
    const b1 = 0.9, b2 = 0.999;
    let last = 0;
    for (let it = 1; it <= iters; it++) {
      const grad = new Array<number>(d).fill(0);
      for (let j = 0; j < d; j++) {
        const tp = theta.slice(); tp[j] += eps;
        const tm = theta.slice(); tm[j] -= eps;
        grad[j] = (this.negLogLikelihood(family, tp, times, path) - this.negLogLikelihood(family, tm, times, path)) / (2 * eps);
      }
      for (let j = 0; j < d; j++) {
        m[j] = b1 * m[j] + (1 - b1) * grad[j];
        v[j] = b2 * v[j] + (1 - b2) * grad[j] * grad[j];
        const mhat = m[j] / (1 - Math.pow(b1, it));
        const vhat = v[j] / (1 - Math.pow(b2, it));
        theta[j] -= lr * mhat / (Math.sqrt(vhat) + 1e-8);
      }
      if (it === iters) last = this.negLogLikelihood(family, theta, times, path);
    }
    return {
      theta, params: family.describe(theta), system: family.instantiate(theta),
      finalNegLogLik: last, iterations: iters,
    };
  }

  private determinant(M: Mat): number {
    const n = M.length;
    const a = M.map(r => r.slice());
    let det = 1;
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let i = col + 1; i < n; i++) if (Math.abs(a[i][col]) > Math.abs(a[piv][col])) piv = i;
      if (Math.abs(a[piv][col]) < 1e-300) return 0;
      if (piv !== col) { const t = a[piv]; a[piv] = a[col]; a[col] = t; det = -det; }
      det *= a[col][col];
      for (let i = col + 1; i < n; i++) {
        const f = a[i][col] / a[col][col];
        for (let j = col; j < n; j++) a[i][j] -= f * a[col][j];
      }
    }
    return det;
  }
}

// =============================================================================
// ML-2. ENSEMBLE KALMAN FILTER (sequential Bayesian state estimation)
// =============================================================================

export interface EnkfOptions {
  ensembleSize?: number;
  observationMatrix: Mat;           // H (p×n)
  observationNoiseVar: Vec;         // diag R (length p)
  initialMean: Vec;                 // x̂₀ (length n)
  initialStd: Vec;                  // sqrt diag P₀ (length n)
  seed?: number;
}

/** Stochastic (perturbed-observation) Ensemble Kalman Filter. The forecast
 *  step pushes every ensemble member through the SDE's Euler–Maruyama
 *  transition; the analysis step nudges them toward the observation using the
 *  ensemble-estimated covariance. */
export class EnsembleKalmanFilter {
  private readonly em = new EulerMaruyamaIntegrator();
  private readonly rng: Mulberry32;
  private ensemble: Vec[];
  private t = 0;
  private readonly N: number;
  private readonly H: Mat;
  private readonly R: Vec;

  constructor(private readonly sys: SdeSystem, private readonly dt: number, opts: EnkfOptions) {
    Preconditions.positive('EnsembleKalmanFilter', 'dt', dt);
    this.N = opts.ensembleSize ?? 100;
    this.H = LinAlg.copy(opts.observationMatrix);
    this.R = opts.observationNoiseVar.slice();
    this.rng = new Mulberry32(opts.seed ?? 4242);
    const n = sys.dimension();
    this.ensemble = Array.from({length: this.N}, () =>
      Array.from({length: n}, (_, i) => opts.initialMean[i] + this.rng.normal() * opts.initialStd[i]));
  }

  /** Forecast: advance each member one SDE step with independent noise. */
  predict(): void {
    const m = this.sys.noiseDimension();
    this.ensemble = this.ensemble.map(x => {
      const dW = this.em.brownianIncrement(m, this.dt, this.rng);
      return this.em.step(this.sys, this.t, x, this.dt, dW);
    });
    this.t += this.dt;
  }

  /** Analysis: perturbed-observation Kalman update with the ensemble covariance. */
  update(obs: Vec): void {
    const n = this.sys.dimension();
    const p = LinAlg.rows(this.H);
    const xbar = this.mean();
    // Anomalies A (n×N).
    const A: Mat = LinAlg.zeros(n, this.N);
    for (let j = 0; j < this.N; j++) for (let i = 0; i < n; i++) A[i][j] = this.ensemble[j][i] - xbar[i];
    const Pf = LinAlg.scale(LinAlg.matMul(A, LinAlg.transpose(A)), 1 / (this.N - 1));   // n×n
    const Ht = LinAlg.transpose(this.H);
    const PfHt = LinAlg.matMul(Pf, Ht);                                                 // n×p
    let S = LinAlg.matMul(this.H, PfHt);                                                // p×p
    for (let i = 0; i < p; i++) S[i][i] += this.R[i];
    const K = LinAlg.matMul(PfHt, new MatrixInverse(S).inverse());                      // n×p
    this.ensemble = this.ensemble.map(x => {
      const dPerturbed = obs.map((o, i) => o + this.rng.normal() * Math.sqrt(this.R[i]));
      const innov = LinAlg.matVec(this.H, x).map((hx, i) => dPerturbed[i] - hx);
      const corr = LinAlg.matVec(K, innov);
      return x.map((xi, i) => xi + corr[i]);
    });
  }

  mean(): Vec {
    const n = this.sys.dimension();
    const out = new Array<number>(n).fill(0);
    for (const x of this.ensemble) for (let i = 0; i < n; i++) out[i] += x[i] / this.N;
    return out;
  }

  /** Per-dimension posterior variance (diagonal of ensemble covariance). */
  variance(): Vec {
    const n = this.sys.dimension();
    const xbar = this.mean();
    const out = new Array<number>(n).fill(0);
    for (const x of this.ensemble) for (let i = 0; i < n; i++) out[i] += (x[i] - xbar[i]) ** 2 / (this.N - 1);
    return out;
  }

  /** One filtering step: forecast then assimilate the observation. */
  step(obs: Vec): {mean: Vec; variance: Vec} {
    this.predict();
    this.update(obs);
    return {mean: this.mean(), variance: this.variance()};
  }
}

/** Streaming EnKF as a DES station: consumes observation tokens, emits state
 *  estimate tokens. One observation per tick → one forecast+analysis. */
export class EnsembleKalmanFilterStation extends MemoryTransformEntity<SdeObservationToken, SdeEstimateToken, EnsembleKalmanFilter> {
  constructor(id: string, filter: EnsembleKalmanFilter,
              inputChannel: ChannelName = SdeChannels.OBSERVATION,
              outputChannel: ChannelName = SdeChannels.ESTIMATE) {
    super(id, filter, {inputChannels: inputChannel, outputChannel});
  }
  protected transformQueued(token: SdeObservationToken, _ctx: TransformContext<SdeObservationToken, SdeEstimateToken>): SdeEstimateToken {
    const {mean, variance} = this.previous.step(token.obs);
    return new SdeEstimateToken(token.time, token.step, mean, variance);
  }
}

// =============================================================================
// ML-3. SCORE-BASED DIFFUSION MODEL (learned reverse-time SDE)
// =============================================================================

/** A minimal one-hidden-layer tanh MLP with manual backprop (scalar output),
 *  trained by SGD on a squared-error target. Used as the noise predictor
 *  ε_θ(x, t) of the diffusion model. */
export class Mlp {
  private readonly W1: Mat;
  private readonly b1: Vec;
  private readonly W2: Vec;
  private b2 = 0;

  constructor(private readonly inputDim: number, private readonly hidden: number, rng: Mulberry32) {
    const s = 1 / Math.sqrt(inputDim);
    this.W1 = Array.from({length: hidden}, () => Array.from({length: inputDim}, () => rng.normal() * s));
    this.b1 = new Array<number>(hidden).fill(0);
    this.W2 = Array.from({length: hidden}, () => rng.normal() / Math.sqrt(hidden));
  }

  predict(x: Vec): number {
    let out = this.b2;
    for (let h = 0; h < this.hidden; h++) {
      let z = this.b1[h];
      for (let i = 0; i < this.inputDim; i++) z += this.W1[h][i] * x[i];
      out += this.W2[h] * Math.tanh(z);
    }
    return out;
  }

  /** Forward + backprop one example for loss ½(out − target)²; SGD update. */
  trainExample(x: Vec, target: number, lr: number): number {
    const a1 = new Array<number>(this.hidden);
    let out = this.b2;
    for (let h = 0; h < this.hidden; h++) {
      let z = this.b1[h];
      for (let i = 0; i < this.inputDim; i++) z += this.W1[h][i] * x[i];
      a1[h] = Math.tanh(z);
      out += this.W2[h] * a1[h];
    }
    const dOut = out - target;
    for (let h = 0; h < this.hidden; h++) {
      const dz = dOut * this.W2[h] * (1 - a1[h] * a1[h]);
      this.W2[h] -= lr * dOut * a1[h];
      for (let i = 0; i < this.inputDim; i++) this.W1[h][i] -= lr * dz * x[i];
      this.b1[h] -= lr * dz;
    }
    this.b2 -= lr * dOut;
    return 0.5 * dOut * dOut;
  }
}

export interface DiffusionOptions {
  steps?: number;          // T discretisation steps of the VP-SDE
  betaMin?: number;
  betaMax?: number;
  hidden?: number;
  seed?: number;
}

/** Denoising Diffusion Probabilistic Model on 1-D data — the discrete-time
 *  variance-preserving SDE  dx = −½β(t)x dt + √β(t) dW  whose reverse-time
 *  process is integrated using a learned noise predictor. Data is standardised
 *  internally so the unit-variance prior matches. */
export class DenoisingDiffusionModel {
  private readonly T: number;
  private readonly beta: Vec;
  private readonly alpha: Vec;
  private readonly alphaBar: Vec;
  private readonly net: Mlp;
  private readonly rng: Mulberry32;
  private dataMean = 0;
  private dataStd = 1;

  constructor(opts: DiffusionOptions = {}) {
    this.T = opts.steps ?? 100;
    const bMin = opts.betaMin ?? 1e-4;
    // βmax chosen so ᾱ_T → ~0 at this (small) T, i.e. the forward process
    // actually reaches the N(0,1) prior the sampler starts from.
    const bMax = opts.betaMax ?? 0.2;
    this.rng = new Mulberry32(opts.seed ?? 7);
    this.net = new Mlp(2, opts.hidden ?? 64, this.rng);
    this.beta = new Array<number>(this.T);
    this.alpha = new Array<number>(this.T);
    this.alphaBar = new Array<number>(this.T);
    let abar = 1;
    for (let t = 0; t < this.T; t++) {
      this.beta[t] = bMin + (bMax - bMin) * (t / (this.T - 1));
      this.alpha[t] = 1 - this.beta[t];
      abar *= this.alpha[t];
      this.alphaBar[t] = abar;
    }
  }

  /** Train ε_θ(x_t, t/T) to predict the injected noise (denoising score matching). */
  train(data: readonly number[], opts: {iterations?: number; learningRate?: number} = {}): number {
    Preconditions.nonEmpty('DenoisingDiffusionModel', 'data', data);
    this.dataMean = data.reduce((a, v) => a + v, 0) / data.length;
    const varD = data.reduce((a, v) => a + (v - this.dataMean) ** 2, 0) / data.length;
    this.dataStd = Math.sqrt(varD) || 1;
    const std = data.map(v => (v - this.dataMean) / this.dataStd);
    const iters = opts.iterations ?? 20000;
    const lr = opts.learningRate ?? 0.01;
    let lastLoss = 0;
    for (let it = 0; it < iters; it++) {
      const x0 = std[Math.floor(this.rng.next() * std.length)];
      const t = Math.floor(this.rng.next() * this.T);
      const z = this.rng.normal();
      const ab = this.alphaBar[t];
      const xt = Math.sqrt(ab) * x0 + Math.sqrt(1 - ab) * z;
      lastLoss = this.net.trainExample([xt, (t + 1) / this.T], z, lr);
    }
    return lastLoss;
  }

  /** Draw `count` samples by ancestral reverse-time sampling, then de-standardise. */
  sample(count: number): number[] {
    const out: number[] = [];
    for (let s = 0; s < count; s++) {
      let x = this.rng.normal();
      for (let t = this.T - 1; t >= 0; t--) {
        const eps = this.net.predict([x, (t + 1) / this.T]);
        const ab = this.alphaBar[t];
        const mean = (1 / Math.sqrt(this.alpha[t])) * (x - (this.beta[t] / Math.sqrt(1 - ab)) * eps);
        x = t > 0 ? mean + Math.sqrt(this.beta[t]) * this.rng.normal() : mean;
      }
      out.push(x * this.dataStd + this.dataMean);
    }
    return out;
  }

  /** Fraction of the original signal still present at the final forward step,
   *  √ᾱ_T. Near 0 means the forward SDE has reached the N(0,1) prior the
   *  reverse sampler starts from. */
  terminalSignalRetention(): number {
    return Math.sqrt(this.alphaBar[this.T - 1]);
  }

  /** Number of diffusion (discretised reverse-SDE) steps. */
  numSteps(): number { return this.T; }

  /** Sample mean / std for quick validation against the data distribution. */
  static summarise(samples: readonly number[]): {mean: number; std: number} {
    const mean = samples.reduce((a, v) => a + v, 0) / samples.length;
    const std = Math.sqrt(samples.reduce((a, v) => a + (v - mean) ** 2, 0) / samples.length);
    return {mean, std};
  }
}
