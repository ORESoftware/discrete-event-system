// RUST MIGRATION: target module src/des/general/double_integrator_lqr.rs.
// RUST MIGRATION: DoubleIntegratorOpts and DoubleIntegratorResult become serde structs; matrix/vector traces should use Vec<[f64; 2]> or Vec<Vec<f64>> consistently.
// RUST MIGRATION: runDoubleIntegratorLQR is a control simulation entrypoint and should be a PureTransform-style struct returning Result.
// RUST MIGRATION: gaussian noise must use an injected rand::Rng plus a normal sampler crate; validation/linear algebra failures should return Result.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/double-integrator-lqr.rs  (module des::general::double_integrator_lqr)
// 1:1 file move. Double-integrator plant controlled by an LQR from the discrete-time Riccati eqn.
//
// Declarations → Rust:
//   fn gaussian                 -> assoc fn taking `&mut impl Rng` (Box-Muller)
//   interface DoubleIntegratorOpts / DoubleIntegratorResult -> structs (Default; optionals -> Option<T>)
//   fn runDoubleIntegratorLQR   -> free fn (or PureTransform<Opts, Result>)
//
// Conversion notes (file-specific):
//   - `mulberry32(seed)` closure RNG -> seeded `RandomSource`; `gaussian` consumes it.
//   - Greek local `γ` -> `gamma` (non-ASCII idents won't carry over).
//   - tuples `[number, number]` -> `(f64, f64)` or `[f64; 2]`; `K: number[][]` -> matrix type.
//   - LQRController is a stateful station base -> struct + impl; `.slice()` copies -> `.clone()`.
// =============================================================================

// =============================================================================
// general/double-integrator-lqr.ts — the canonical DOUBLE INTEGRATOR
// (point mass under direct force) controlled by an LQR derived from
// the discrete-time algebraic Riccati equation.
//
// CONTINUOUS-TIME PLANT
// ─────────────────────
//   ẍ = u           (m = 1, no damping)
//   x_state = [position; velocity]
//
// DISCRETE-TIME (sample period τ)
// ───────────────────────────────
//   A = [[1, τ], [0, 1]]
//   B = [[τ²/2], [τ]]
//
// COSTS
// ─────
//   Q = diag(q_pos, q_vel),  R = [[r_u]]
//
// The infinite-horizon optimal LQR minimises Σ_k x^T Q x + u^T R u.
// We add a small Gaussian process noise w ∼ N(0, σ_w² I_2) and observe
// the closed-loop trajectory in the DES framework.
//
// USAGE
// ─────
//   `runDoubleIntegratorLQR(opts)` builds the LQR controller, simulates
//   the closed-loop system from a random initial state, and returns
//   the trajectory and key diagnostics (final state, cumulative cost).
// =============================================================================

import {LQRController, LQRSpec} from './des-base/lqr-controller';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

/** Box-Muller standard normal sample using a `[0, 1)` RNG. */
function gaussian(rng: () => number): number {
  let u = rng(); let v = rng();
  if (u < 1e-12) u = 1e-12;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export interface DoubleIntegratorOpts {
  /** Sample period τ. Default 0.1. */
  dt?: number;
  /** Position weight in Q. Default 1. */
  qPos?: number;
  /** Velocity weight in Q. Default 0.1. */
  qVel?: number;
  /** Control weight in R. Default 0.01. */
  rU?: number;
  /** Process-noise stddev. Default 0.05. */
  noiseStd?: number;
  /** Initial state [pos, vel]. Default [3, 0]. */
  x0?: [number, number];
  /** Number of simulation steps. Default 100. */
  numSteps?: number;
  /** Saturation on |u|. Default Infinity. */
  uSat?: number;
  /** Discount γ. Default 1. */
  gamma?: number;
  seed?: number;
}

export interface DoubleIntegratorResult {
  /** Closed-loop trajectory [pos(t), vel(t)]. Length numSteps+1. */
  trajectory: Array<[number, number]>;
  /** Control input at each step. */
  controls: number[];
  /** Per-step running cost x^T Q x + u^T R u. */
  stageCosts: number[];
  /** Σ stage costs. */
  totalCost: number;
  /** Optimal cost-to-go from x0 according to the Riccati P (theory). */
  riccatiCostFromX0: number;
  /** Riccati iteration count to convergence. */
  riccatiIters: number;
  /** Final residual ||P_new − P||_∞ at convergence. */
  riccatiResidual: number;
  /** Optimal feedback gain K. */
  K: number[][];
}

export function runDoubleIntegratorLQR(opts: DoubleIntegratorOpts = {}): DoubleIntegratorResult {
  const dt = opts.dt ?? 0.1;
  const qPos = opts.qPos ?? 1;
  const qVel = opts.qVel ?? 0.1;
  const rU = opts.rU ?? 0.01;
  const noiseStd = opts.noiseStd ?? 0.05;
  const x0: [number, number] = opts.x0 ?? [3, 0];
  const N = opts.numSteps ?? 100;
  const uSat = opts.uSat ?? Infinity;
  const γ = opts.gamma ?? 1;
  // Pre-run guards.
  const cls = 'runDoubleIntegratorLQR';
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.nonNegative(cls, 'qPos', qPos);
  Preconditions.nonNegative(cls, 'qVel', qVel);
  // R = rU > 0 mandatory: DARE requires (R + B'PB) invertible.
  Preconditions.positive(cls, 'rU', rU);
  Preconditions.nonNegative(cls, 'noiseStd', noiseStd);
  Preconditions.lengthEq(cls, 'x0', x0, 2);
  Preconditions.allFinite(cls, 'x0', x0);
  Preconditions.integerInRange(cls, 'numSteps', N, 1, 1e9);
  if (Number.isFinite(uSat)) Preconditions.positive(cls, 'uSat', uSat);
  Preconditions.inRange(cls, 'gamma', γ, 1e-9, 1);
  const rng = mulberry32(opts.seed ?? 1);

  const spec: LQRSpec = {
    n: 2, m: 1,
    A: [[1, dt], [0, 1]],
    B: [[dt * dt / 2], [dt]],
    Q: [[qPos, 0], [0, qVel]],
    R: [[rU]],
    gamma: γ,
    uMinVec: isFinite(uSat) ? [-uSat] : undefined,
    uMaxVec: isFinite(uSat) ? [uSat]  : undefined,
  };
  const ctrl = new LQRController('double-int-lqr', spec);

  const traj: Array<[number, number]> = [[x0[0], x0[1]]];
  const ctrls: number[] = [];
  const stageCosts: number[] = [];
  let total = 0;
  let x: [number, number] = [x0[0], x0[1]];
  for (let k = 0; k < N; k++) {
    const u = ctrl.step([x[0], x[1]], k, k * dt);
    const uVal = u[0];
    ctrls.push(uVal);
    // Stage cost x^T Q x + u^T R u.
    const sc = qPos * x[0] * x[0] + qVel * x[1] * x[1] + rU * uVal * uVal;
    stageCosts.push(sc);
    total += sc;
    // Dynamics step: x_{k+1} = A x_k + B u_k + w_k.
    const w0 = noiseStd > 0 ? noiseStd * gaussian(rng) : 0;
    const w1 = noiseStd > 0 ? noiseStd * gaussian(rng) : 0;
    const xNext: [number, number] = [
      x[0] + dt * x[1] + (dt * dt / 2) * uVal + w0,
      x[1] + dt * uVal + w1,
    ];
    traj.push(xNext);
    x = xNext;
  }
  return {
    trajectory: traj, controls: ctrls, stageCosts, totalCost: total,
    riccatiCostFromX0: ctrl.optimalCostFromInitialState([x0[0], x0[1]]),
    riccatiIters: ctrl.riccatiIters,
    riccatiResidual: ctrl.riccatiResidual,
    K: ctrl.getGain().map(r => r.slice()),
  };
}
