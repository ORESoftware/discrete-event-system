'use strict';

// =============================================================================
// general/kalman-filter.ts — LINEAR KALMAN FILTER (Kalman 1960) on the
// canonical RADAR / GPS TRACKING problem.
//
// PROBLEM
// ───────
//   A 1-D point mass moves with random acceleration noise:
//
//       x_{k+1} = A x_k + w_k,       w_k ∼ N(0, Q)
//       y_k     = H x_k + v_k,       v_k ∼ N(0, R)
//
//   with state x = [position, velocity]ᵀ, A = [[1, dt], [0, 1]],
//   H = [1, 0] (we only observe position), Q = noise cov from the
//   integral of acceleration noise, R = sensor variance.
//
// KALMAN UPDATE (textbook, Welch & Bishop 1995)
// ─────────────────────────────────────────────
//   PREDICT  x̂⁻ = A x̂                P⁻ = A P Aᵀ + Q
//   UPDATE   K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
//            x̂  = x̂⁻ + K (y − H x̂⁻)   P  = (I − K H) P⁻
//
// AS A DES BLOCK
// ──────────────
//   `KalmanFilterBlock extends EstimatorBlock` — drains incoming
//   measurements y on every tick, runs ONE predict-update step, emits
//   the posterior estimate x̂ on its out connections. The plant block
//   is `RadarPlant` (linear motion + i.i.d. position noise).
//
//   `runRadarTracking(opts)` wires plant → KF → (passive sink) and
//   returns the trajectory + estimates + measurement RMSE diagnostics.
// =============================================================================

import {
  PlantBlock, ControllerBlock, EstimatorBlock, runClosedLoop,
} from './des-base/control-blocks';
import {matMul, matT, matAdd, matSub, matInv, matMV, Mat, Vec}
  from './des-base/lqr-controller';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// PLANT: noisy 1-D constant-velocity model with position-only sensor
// -----------------------------------------------------------------------------

class RadarPlant extends PlantBlock {
  private readonly procNoiseStd: number;
  private readonly measNoiseStd: number;
  private readonly rng: () => number;

  constructor(x0: [number, number], dt: number, procNoiseStd: number,
              measNoiseStd: number, rng: () => number) {
    super('radar-plant', x0, dt, 1);
    this.procNoiseStd = procNoiseStd;
    this.measNoiseStd = measNoiseStd;
    this.rng = rng;
  }

  protected dynamics(x: readonly number[], _u: readonly number[], dt: number): number[] {
    // Constant velocity + Gaussian acceleration noise.
    const accNoise = this.procNoiseStd * gaussian(this.rng);
    return [x[0] + dt * x[1] + 0.5 * dt * dt * accNoise,
            x[1] + dt * accNoise];
  }

  protected observe(x: readonly number[]): number[] {
    return [x[0] + this.measNoiseStd * gaussian(this.rng)];
  }
}

// -----------------------------------------------------------------------------
// KALMAN FILTER BLOCK
// -----------------------------------------------------------------------------

export class KalmanFilterBlock extends EstimatorBlock {
  /** State estimate. */
  private xhat: Vec;
  /** Posterior covariance. */
  private P: Mat;
  private readonly A: Mat;
  private readonly H: Mat;
  private readonly Q: Mat;
  private readonly R: Mat;

  constructor(spec: {x0: Vec; P0: Mat; A: Mat; H: Mat; Q: Mat; R: Mat}) {
    super('kalman-filter');
    const cls = 'KalmanFilterBlock';
    const n = spec.x0.length;
    Preconditions.check(cls, 'x0.length', 'be >= 1', n >= 1, n);
    Preconditions.allFinite(cls, 'x0', spec.x0);
    Preconditions.lengthEq(cls, 'A', spec.A, n);
    Preconditions.rectangularMatrix(cls, 'A', spec.A);
    Preconditions.lengthEq(cls, 'A[0]', spec.A[0], n);
    Preconditions.symmetricMatrix(cls, 'P0', spec.P0);
    Preconditions.lengthEq(cls, 'P0', spec.P0, n);
    Preconditions.positiveSemidefiniteDiag(cls, 'P0', spec.P0);
    Preconditions.symmetricMatrix(cls, 'Q', spec.Q);
    Preconditions.lengthEq(cls, 'Q', spec.Q, n);
    Preconditions.positiveSemidefiniteDiag(cls, 'Q', spec.Q);
    Preconditions.rectangularMatrix(cls, 'H', spec.H);
    Preconditions.lengthEq(cls, 'H[0]', spec.H[0], n);
    const m = spec.H.length;
    Preconditions.check(cls, 'H.length (output dim m)', 'be >= 1', m >= 1, m);
    Preconditions.symmetricMatrix(cls, 'R', spec.R);
    Preconditions.lengthEq(cls, 'R', spec.R, m);
    // R MUST be PD (we invert HPH^T + R and need a strictly positive
    // total innovation covariance — measurement noise zero ⇒ degenerate).
    Preconditions.positiveDefiniteCholesky(cls, 'R', spec.R);
    this.xhat = spec.x0.slice();
    this.P = spec.P0.map(r => r.slice());
    this.A = spec.A; this.H = spec.H; this.Q = spec.Q; this.R = spec.R;
  }

  protected update(y: readonly number[], _u: readonly number[] | null): number[] {
    // PREDICT.
    const xhatPred = matMV(this.A, this.xhat);
    const APAt = matMul(matMul(this.A, this.P), matT(this.A));
    const Ppred = matAdd(APAt, this.Q);
    // UPDATE: K = P H^T (H P H^T + R)^{-1}
    const HP = matMul(this.H, Ppred);
    const HPHt = matMul(HP, matT(this.H));
    const S = matAdd(HPHt, this.R);
    const Sinv = matInv(S);
    const K = matMul(matMul(Ppred, matT(this.H)), Sinv);
    // Innovation y - H xhatPred.
    const Hxhat = matMV(this.H, xhatPred);
    const innov = y.map((yi, i) => yi - Hxhat[i]);
    // x̂ = x̂⁻ + K innov
    const Kinnov = matMV(K, innov);
    this.xhat = xhatPred.map((v, i) => v + Kinnov[i]);
    // P = (I - K H) P⁻
    const KH = matMul(K, this.H);
    const I = identity(KH.length);
    this.P = matMul(matSub(I, KH), Ppred);
    return this.xhat.slice();
  }

  getEstimate(): number[] { return this.xhat.slice(); }
  getCovariance(): Mat { return this.P.map(r => r.slice()); }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface RadarTrackingOpts {
  x0?: [number, number];
  dt?: number;
  numSteps?: number;
  /** Process-noise std σ_w (acceleration). Default 0.1. */
  procNoiseStd?: number;
  /** Sensor std σ_v. Default 1.0. */
  measNoiseStd?: number;
  /** Initial covariance scale (P0 = sigma·I). Default 10. */
  P0Scale?: number;
  seed?: number;
}

export interface RadarTrackingResult {
  trueTrajectory: number[][];
  measurements: number[][];
  estimates: number[][];
  /** RMSE between true position and KF estimate. */
  rmsePos: number;
  /** RMSE between true position and raw measurement (baseline). */
  rmseMeasPos: number;
  /** Final position covariance trace. */
  finalCovTrace: number;
  numSteps: number;
}

export function runRadarTracking(opts: RadarTrackingOpts = {}): RadarTrackingResult {
  const x0: [number, number] = opts.x0 ?? [0, 1];
  const dt = opts.dt ?? 0.1;
  const numSteps = opts.numSteps ?? 200;
  const procNoiseStd = opts.procNoiseStd ?? 0.1;
  const measNoiseStd = opts.measNoiseStd ?? 1.0;
  const P0Scale = opts.P0Scale ?? 10;
  const cls = 'runRadarTracking';
  Preconditions.lengthEq(cls, 'x0', x0, 2);
  Preconditions.allFinite(cls, 'x0', x0);
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.integerInRange(cls, 'numSteps', numSteps, 1, 1e9);
  Preconditions.nonNegative(cls, 'procNoiseStd', procNoiseStd);
  // Sensor noise must be strictly positive — KF inverts H P H' + R, so
  // R = 0 collapses to a perfect-sensor degenerate filter (use a
  // dedicated observer instead).
  Preconditions.positive(cls, 'measNoiseStd', measNoiseStd);
  Preconditions.positive(cls, 'P0Scale', P0Scale);
  const rng = mulberry32(opts.seed ?? 1);

  const plant = new RadarPlant(x0, dt, procNoiseStd, measNoiseStd, rng);

  // Build KF model matrices for constant-velocity dynamics.
  const A: Mat = [[1, dt], [0, 1]];
  const H: Mat = [[1, 0]];
  // Process noise from continuous-time acceleration variance σ_w²:
  //   Q = σ_w² · [[dt^4/4, dt^3/2], [dt^3/2, dt²]]
  const sw2 = procNoiseStd * procNoiseStd;
  const Q: Mat = [
    [sw2 * dt * dt * dt * dt / 4, sw2 * dt * dt * dt / 2],
    [sw2 * dt * dt * dt / 2,      sw2 * dt * dt],
  ];
  const R: Mat = [[measNoiseStd * measNoiseStd]];
  // Initial state estimate: take the (eventual) first measurement at zero
  // velocity. We also make P0 large to express uncertainty.
  const kf = new KalmanFilterBlock({
    x0: [x0[0], 0],
    P0: [[P0Scale, 0], [0, P0Scale]],
    A, H, Q, R,
  });

  // Use a "no-op" controller — no closed-loop feedback for a pure
  // tracking demonstration; the plant evolves under noise alone.
  const passive = new NullController();
  // Emit a no-op u so the plant gets one tick of input zero each round.
  runClosedLoop(plant, passive, {numSteps, estimator: kf});

  // Compute diagnostics.
  let rmseEstSum = 0; let rmseMeasSum = 0; let n = 0;
  // The estimator starts producing output from tick 1 onward. The plant's
  // measurement at tick t corresponds to state at tick t.
  const trueTraj = plant.stateHistory; // length numSteps+1
  const meas = plant.outputHistory;    // length numSteps
  const est = kf.estimateHistory;      // length numSteps
  for (let i = 0; i < est.length; i++) {
    const truePos = trueTraj[i + 1][0];
    const ePos = est[i][0];
    const mPos = meas[i][0];
    rmseEstSum += (truePos - ePos) * (truePos - ePos);
    rmseMeasSum += (truePos - mPos) * (truePos - mPos);
    n += 1;
  }
  const rmsePos = Math.sqrt(rmseEstSum / Math.max(1, n));
  const rmseMeasPos = Math.sqrt(rmseMeasSum / Math.max(1, n));
  const Pfinal = kf.getCovariance();
  const finalCovTrace = Pfinal[0][0] + Pfinal[1][1];

  return {
    trueTrajectory: trueTraj.map(x => x.slice()),
    measurements: meas.map(y => y.slice()),
    estimates: est.map(x => x.slice()),
    rmsePos, rmseMeasPos, finalCovTrace, numSteps,
  };
}

// -----------------------------------------------------------------------------
// HELPERS
// -----------------------------------------------------------------------------

class NullController extends ControllerBlock {
  constructor() { super('null-controller', 1); }
  protected controlLaw(_y: readonly number[]): number[] { return [0]; }
}

function gaussian(rng: () => number): number {
  let u = rng(); let v = rng();
  if (u < 1e-12) u = 1e-12;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function identity(n: number): Mat {
  const I: Mat = Array.from({length: n}, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}
