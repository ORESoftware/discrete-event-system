// RUST MIGRATION: Target module `src/des/general/control_systems/stochastic_sde.rs`.
// RUST MIGRATION: Convert `SdeSystem`, Euler-Maruyama, GBM/OU/stochastic motor plants, and stations into structs plus plant/integrator traits.
// RUST MIGRATION: Use `f64` state/noise vectors and matrices, inject RNG/config explicitly, and return `Result` for dimension/noise validation.
// RUST MIGRATION: Any graph-visible pure SDE propagation/evaluation should become a PureTransform-style struct with a `transform` method.
'use strict';

// =============================================================================
// control-systems/stochastic-sde.ts — stochastic differential equations.
//
//   dX_t = f(X_t, t) dt + g(X_t, t) dW_t          (Itô SDE)
//
// The solution X_t is a RANDOM PROCESS, not a deterministic function. This file
// provides:
//   • `SdeSystem`            — the drift f / diffusion g contract (methods only)
//   • `EulerMaruyamaIntegrator` — the canonical fixed-step SDE solver
//   • `GeometricBrownianMotion` (Black–Scholes), `OrnsteinUhlenbeck`
//     (mean-reverting), `StochasticDcMotor` (the motor ODE + process noise)
//   • a self-clocking `SdePlantStation` that streams truth + noisy observation
//     tokens, so the ML estimators in `sde-learning.ts` can consume them online.
//
// Everything is CLASSES with METHODS (LinAlg for the algebra, the seedable
// Mulberry32 RNG from empirical-control for the Brownian increments).
// =============================================================================

import {ChannelName, DESStation, Token} from '../des-base/station';
import {Preconditions} from '../des-base/preconditions';
import {LinAlg, Mat, Vec} from './linear-algebra';
import {Mulberry32} from './empirical-control';

// =============================================================================
// SDE contract + solver
// =============================================================================

/** An Itô SDE  dX = f(X,t) dt + g(X,t) dW  with state dim n and noise dim m. */
export interface SdeSystem {
  /** State dimension n. */
  dimension(): number;
  /** Brownian-motion dimension m. */
  noiseDimension(): number;
  /** Drift f(t, x) — length n. */
  drift(t: number, x: Vec): Vec;
  /** Diffusion g(t, x) — n×m matrix multiplying dW. */
  diffusion(t: number, x: Vec): Mat;
}

/** Fixed-step Euler–Maruyama:  x_{k+1} = x_k + f Δt + g √Δt ξ,  ξ ~ N(0, I_m). */
export class EulerMaruyamaIntegrator {
  /** One step given a pre-drawn Brownian increment dW (length m). */
  step(sys: SdeSystem, t: number, x: Vec, dt: number, dW: Vec): Vec {
    const f = sys.drift(t, x);
    const g = sys.diffusion(t, x);
    const gdW = LinAlg.matVec(g, dW);
    const out = new Array<number>(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] + f[i] * dt + gdW[i];
    return out;
  }

  /** Draw a Brownian increment dW = √Δt · ξ. */
  brownianIncrement(m: number, dt: number, rng: Mulberry32): Vec {
    const s = Math.sqrt(dt);
    return Array.from({length: m}, () => rng.normal() * s);
  }

  /** Simulate one sample path; returns times[0..steps] and path[0..steps]. */
  simulate(sys: SdeSystem, x0: Vec, dt: number, steps: number, rng: Mulberry32):
  {times: number[]; path: Vec[]} {
    Preconditions.positive('EulerMaruyamaIntegrator', 'dt', dt);
    const times = [0];
    const path: Vec[] = [x0.slice()];
    let x = x0.slice();
    for (let k = 0; k < steps; k++) {
      const dW = this.brownianIncrement(sys.noiseDimension(), dt, rng);
      x = this.step(sys, k * dt, x, dt, dW);
      times.push((k + 1) * dt);
      path.push(x);
    }
    return {times, path};
  }
}

// =============================================================================
// Concrete systems
// =============================================================================

/** dX = μ X dt + σ X dW — geometric Brownian motion (Black–Scholes asset). */
export class GeometricBrownianMotion implements SdeSystem {
  constructor(readonly mu: number, readonly sigma: number) {
    Preconditions.finite('GeometricBrownianMotion', 'mu', mu);
    Preconditions.nonNegative('GeometricBrownianMotion', 'sigma', sigma);
  }
  dimension(): number { return 1; }
  noiseDimension(): number { return 1; }
  drift(_t: number, x: Vec): Vec { return [this.mu * x[0]]; }
  diffusion(_t: number, x: Vec): Mat { return [[this.sigma * x[0]]]; }

  /** Closed-form solution X_t = X_0 exp((μ−σ²/2)t + σ W_t). */
  exact(x0: number, t: number, Wt: number): number {
    return x0 * Math.exp((this.mu - 0.5 * this.sigma * this.sigma) * t + this.sigma * Wt);
  }
  /** E[X_t] = X_0 e^{μt}. */
  meanAt(x0: number, t: number): number { return x0 * Math.exp(this.mu * t); }
  /** Var[X_t] = X_0² e^{2μt}(e^{σ²t} − 1). */
  varAt(x0: number, t: number): number {
    return x0 * x0 * Math.exp(2 * this.mu * t) * (Math.exp(this.sigma * this.sigma * t) - 1);
  }
}

/** dX = θ(μ − X) dt + σ dW — Ornstein–Uhlenbeck (mean-reverting). */
export class OrnsteinUhlenbeck implements SdeSystem {
  constructor(readonly theta: number, readonly mu: number, readonly sigma: number) {
    Preconditions.positive('OrnsteinUhlenbeck', 'theta', theta);
    Preconditions.nonNegative('OrnsteinUhlenbeck', 'sigma', sigma);
  }
  dimension(): number { return 1; }
  noiseDimension(): number { return 1; }
  drift(_t: number, x: Vec): Vec { return [this.theta * (this.mu - x[0])]; }
  diffusion(_t: number, _x: Vec): Mat { return [[this.sigma]]; }

  /** Stationary distribution N(μ, σ²/(2θ)). */
  stationaryMean(): number { return this.mu; }
  stationaryVariance(): number { return (this.sigma * this.sigma) / (2 * this.theta); }
}

export interface StochasticDcMotorSpec {
  resistance: number; inductance: number; backEmfConstant: number;
  torqueConstant: number; inertia: number; friction: number;
  voltage: number; loadTorque?: number;
  currentNoise: number; speedNoise: number;
}

/** Stochastic DC motor: the deterministic [i, ω] ODE plus additive process
 *  noise on each state — di = (V−Ri−K_eω)/L dt + σ_i dW₁,
 *  dω = (K_t i−Bω−T_L)/J dt + σ_ω dW₂. */
export class StochasticDcMotor implements SdeSystem {
  constructor(private readonly p: StochasticDcMotorSpec) {
    const cls = 'StochasticDcMotor';
    Preconditions.positive(cls, 'inductance', p.inductance);
    Preconditions.positive(cls, 'inertia', p.inertia);
  }
  dimension(): number { return 2; }
  noiseDimension(): number { return 2; }
  drift(_t: number, x: Vec): Vec {
    const [i, w] = x;
    const p = this.p;
    return [
      (p.voltage - p.resistance * i - p.backEmfConstant * w) / p.inductance,
      (p.torqueConstant * i - p.friction * w - (p.loadTorque ?? 0)) / p.inertia,
    ];
  }
  diffusion(_t: number, _x: Vec): Mat {
    return [[this.p.currentNoise, 0], [0, this.p.speedNoise]];
  }
}

// =============================================================================
// DES pipeline — a streaming SDE plant.
// =============================================================================

export class SdeChannels {
  static readonly STATE: ChannelName = 'sde-state';
  static readonly OBSERVATION: ChannelName = 'sde-observation';
  static readonly ESTIMATE: ChannelName = 'sde-estimate';
}

export class SdeStateToken implements Token {
  constructor(readonly time: number, readonly step: number, readonly state: Vec) {}
}

export class SdeObservationToken implements Token {
  constructor(readonly time: number, readonly step: number, readonly obs: Vec) {}
}

export class SdeEstimateToken implements Token {
  constructor(readonly time: number, readonly step: number, readonly mean: Vec, readonly variance: Vec) {}
}

export interface SdePlantOptions {
  system: SdeSystem;
  x0: Vec;
  dt: number;
  steps: number;
  /** observation matrix H (p×n); default = identity (observe full state). */
  observationMatrix?: Mat;
  /** per-observation-channel measurement-noise std; default 0. */
  observationNoiseStd?: Vec;
  seed?: number;
}

/** Self-clocking plant: each tick advances the SDE one Euler–Maruyama step and
 *  emits the true state plus a noisy observation y = H x + v. */
export class SdePlantStation extends DESStation {
  private readonly em = new EulerMaruyamaIntegrator();
  private readonly rng: Mulberry32;
  private readonly H: Mat;
  private readonly obsNoise: Vec;
  private x: Vec;
  private k = 0;

  readonly trueStates: SdeStateToken[] = [];
  readonly observations: SdeObservationToken[] = [];

  constructor(id: string, private readonly opts: SdePlantOptions) {
    super(id);
    Preconditions.positive('SdePlantStation', 'dt', opts.dt);
    Preconditions.integerInRange('SdePlantStation', 'steps', opts.steps, 1, 10_000_000);
    this.rng = new Mulberry32(opts.seed ?? 20260529);
    const n = opts.system.dimension();
    this.H = opts.observationMatrix ?? LinAlg.identity(n);
    this.obsNoise = opts.observationNoiseStd ?? new Array<number>(LinAlg.rows(this.H)).fill(0);
    this.x = opts.x0.slice();
  }

  override hasWork(): boolean { return this.k < this.opts.steps; }

  runTimeStep(): void {
    if (this.k >= this.opts.steps) return;
    const t = this.k * this.opts.dt;
    const dW = this.em.brownianIncrement(this.opts.system.noiseDimension(), this.opts.dt, this.rng);
    this.x = this.em.step(this.opts.system, t, this.x, this.opts.dt, dW);
    this.k++;
    const tNow = this.k * this.opts.dt;
    const stateTok = new SdeStateToken(tNow, this.k, this.x.slice());
    const yClean = LinAlg.matVec(this.H, this.x);
    const y = yClean.map((v, i) => v + this.rng.normal() * this.obsNoise[i]);
    const obsTok = new SdeObservationToken(tNow, this.k, y);
    this.trueStates.push(stateTok);
    this.observations.push(obsTok);
    this.emit(stateTok, SdeChannels.STATE);
    this.emit(obsTok, SdeChannels.OBSERVATION);
  }
}

/** Collects truth + estimate tokens and reports filtering accuracy. */
export class SdeEstimateSinkStation extends DESStation {
  readonly estimates: SdeEstimateToken[] = [];
  readonly truth: SdeStateToken[] = [];

  constructor(id: string) { super(id); }

  override hasWork(): boolean {
    return this.inboxSize(SdeChannels.ESTIMATE) > 0 || this.inboxSize(SdeChannels.STATE) > 0;
  }

  runTimeStep(): void {
    this.truth.push(...this.drain<SdeStateToken>(SdeChannels.STATE));
    this.estimates.push(...this.drain<SdeEstimateToken>(SdeChannels.ESTIMATE));
  }

  /** Per-state-dimension RMSE between estimate.mean and the aligned truth. */
  rmseByDimension(): Vec {
    const byStep = new Map<number, SdeStateToken>();
    for (const t of this.truth) byStep.set(t.step, t);
    const n = this.truth.length ? this.truth[0].state.length : 0;
    const sse = new Array<number>(n).fill(0);
    let count = 0;
    for (const e of this.estimates) {
      const t = byStep.get(e.step);
      if (!t) continue;
      count++;
      for (let i = 0; i < n; i++) { const d = e.mean[i] - t.state[i]; sse[i] += d * d; }
    }
    return sse.map(s => (count > 0 ? Math.sqrt(s / count) : NaN));
  }
}
