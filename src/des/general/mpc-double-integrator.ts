// RUST MIGRATION: target module src/des/general/mpc_double_integrator.rs.
// RUST MIGRATION: MPCDoubleIntOpts and MPCDoubleIntResult become serde structs; ClosedLoopResult extension should be flattened or composed.
// RUST MIGRATION: DoubleIntegratorPlant and MPCDoubleIntegratorController become Plant/Controller trait impl structs instead of block subclasses.
// RUST MIGRATION: runMPCDoubleIntegrator is a DES/control PureTransform returning Result; projected-gradient QP helper logic can remain private free functions.
// RUST MIGRATION: Use fixed [f64; 2] state/control vectors where possible and return Result for infeasible horizon/bounds settings.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/mpc-double-integrator.rs  (module des::general::mpc_double_integrator)
// 1:1 file move. Constrained MPC for the double integrator (projected-gradient QP per tick).
//
// Declarations → Rust:
//   interface MPCDoubleIntOpts                          -> struct (Default-derivable)
//   interface MPCDoubleIntResult extends ClosedLoopResult -> struct embedding/flattening ClosedLoopResult
//   class DoubleIntegratorPlant extends PlantBlock       -> struct + impl Plant trait (private)
//   class MPCDoubleIntegratorController extends ControllerBlock -> struct + impl Controller trait (private)
//   fn runMPCDoubleIntegrator                            -> free fn / assoc fn
//
// Conversion notes (file-specific):
//   - PlantBlock/ControllerBlock are template-method bases -> traits with default fns; the
//     controller's QP state (horizon buffers) are struct fields, solved with f64 vectors.
//   - `extends ClosedLoopResult`: Rust has no interface inheritance — compose by embedding the
//     base struct or copy its fields explicitly.
//   - all numerics are `f64`; no RNG/clock/Map here (deterministic projected-gradient inner loop).
// =============================================================================
// general/mpc-double-integrator.ts — MODEL PREDICTIVE CONTROL (Mayne et
// al. 2000) with input box constraints, applied to the double
// integrator. Solves a small QP every tick by projected gradient (no
// external QP library needed) and applies the FIRST control of the
// receding horizon.
//
// PROBLEM
// ───────
//   Plant:  ẍ = u,         |u| ≤ u_max
//
//   At each tick k, the controller solves:
//
//       min_{u_0 .. u_{N-1}} ∑_{i=0..N-1} xᵢᵀQxᵢ + uᵢ²·R   +   x_NᵀQ_f x_N
//       s.t.  x_{i+1} = A x_i + B u_i,   |u_i| ≤ u_max
//
//   over the horizon N, then applies u_0. This rolls forward to k+1 and
//   re-solves (receding horizon).
//
//   This is the INDUSTRIAL workhorse for any LINEAR plant with hard
//   actuator limits — chemical plants, autonomous vehicles, building
//   HVAC, microgrid dispatch, all use a variant of this. The version
//   here uses NO external solver: the QP is dense but tiny (N ≤ 20
//   inputs), so a few hundred steps of projected gradient are plenty.
//
// AS A DES BLOCK
// ──────────────
//   `MPCDoubleIntegratorController extends ControllerBlock` — drains
//   incoming state y, builds the QP, solves it, emits u_0 and discards
//   the rest. The plant is the standard `DoubleIntegratorPlant` from
//   `des-base/control-blocks` style — defined locally for clarity.
// =============================================================================

import {
  PlantBlock, ControllerBlock, runClosedLoop, ClosedLoopResult,
} from './des-base/control-blocks';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// PLANT
// -----------------------------------------------------------------------------

class DoubleIntegratorPlant extends PlantBlock {
  constructor(x0: [number, number], dt: number) {
    super('mpc-double-int-plant', x0, dt, 1);
  }
  protected dynamics(x: readonly number[], u: readonly number[], dt: number): number[] {
    return [x[0] + dt * x[1] + 0.5 * dt * dt * u[0], x[1] + dt * u[0]];
  }
}

// -----------------------------------------------------------------------------
// MPC CONTROLLER
// -----------------------------------------------------------------------------

class MPCDoubleIntegratorController extends ControllerBlock {
  /** Horizon length N. */
  private readonly N: number;
  /** State weight Q (2×2 diagonal). */
  private readonly Q: [number, number];
  /** Terminal weight Q_f. */
  private readonly Qf: [number, number];
  /** Input weight R (scalar). */
  private readonly R: number;
  /** Saturation bound on u. */
  private readonly uMaxScalar: number;
  /** Sample period dt. */
  private readonly dtCache: number;
  /** A and B matrices for the discrete double integrator. */
  private readonly A: [[number, number], [number, number]];
  private readonly B: [number, number];
  /** Warm-start sequence (length N). */
  private readonly warmStart: number[];

  constructor(opts: {N?: number; Q?: [number, number]; Qf?: [number, number];
                     R?: number; uMax?: number; dt: number}) {
    super('mpc-double-int', 1);
    const cls = 'MPCDoubleIntegratorController';
    Preconditions.integerInRange(cls, 'N (horizon)', opts.N ?? 15, 1, 1000);
    if (opts.Q) {
      Preconditions.lengthEq(cls, 'Q', opts.Q, 2);
      Preconditions.arrNonNegative(cls, 'Q', opts.Q);
    }
    if (opts.Qf) {
      Preconditions.lengthEq(cls, 'Qf', opts.Qf, 2);
      Preconditions.arrNonNegative(cls, 'Qf', opts.Qf);
    }
    // R MUST be > 0 — appears as 2 R u in the gradient, R = 0 makes
    // the problem ill-posed (control unbounded along nullspace).
    Preconditions.positive(cls, 'R', opts.R ?? 0.1);
    Preconditions.positive(cls, 'uMax', opts.uMax ?? 1);
    Preconditions.positive(cls, 'dt', opts.dt);
    this.N = opts.N ?? 15;
    this.Q = opts.Q ?? [10, 1];
    this.Qf = opts.Qf ?? [50, 5];
    this.R = opts.R ?? 0.1;
    this.uMaxScalar = opts.uMax ?? 1;
    this.dtCache = opts.dt;
    const dt = opts.dt;
    this.A = [[1, dt], [0, 1]];
    this.B = [0.5 * dt * dt, dt];
    this.warmStart = new Array(this.N).fill(0);
    this.setSaturation([-this.uMaxScalar], [this.uMaxScalar]);
  }
  protected getDt(): number { return this.dtCache; }
  protected controlLaw(y: readonly number[]): number[] {
    const x0: [number, number] = [y[0], y[1]];
    const useq = this.solveQP(x0);
    // Warm-start: shift solution left, append zero.
    for (let i = 0; i < this.N - 1; i++) this.warmStart[i] = useq[i + 1];
    this.warmStart[this.N - 1] = 0;
    return [useq[0]];
  }

  /** Project u onto [-uMax, uMax]. */
  private clip(v: number): number {
    return v < -this.uMaxScalar ? -this.uMaxScalar : (v > this.uMaxScalar ? this.uMaxScalar : v);
  }

  /** Compute J(useq) and ∇_u J via reverse rollout. */
  private costAndGrad(x0: [number, number], useq: number[]): {J: number; grad: number[]} {
    const N = this.N;
    // Forward rollout to get x_0..x_N.
    const xs: [number, number][] = new Array(N + 1) as [number, number][];
    xs[0] = [x0[0], x0[1]];
    let J = 0;
    for (let i = 0; i < N; i++) {
      const x = xs[i]; const u = useq[i];
      // stage cost x^T Q x + R u²
      J += this.Q[0] * x[0] * x[0] + this.Q[1] * x[1] * x[1] + this.R * u * u;
      const xn: [number, number] = [
        this.A[0][0] * x[0] + this.A[0][1] * x[1] + this.B[0] * u,
        this.A[1][0] * x[0] + this.A[1][1] * x[1] + this.B[1] * u,
      ];
      xs[i + 1] = xn;
    }
    // terminal cost
    J += this.Qf[0] * xs[N][0] * xs[N][0] + this.Qf[1] * xs[N][1] * xs[N][1];

    // Backward sweep for gradient.
    // λ_N = 2 [Qf_0 x_N0; Qf_1 x_N1]
    let lambda: [number, number] = [
      2 * this.Qf[0] * xs[N][0],
      2 * this.Qf[1] * xs[N][1],
    ];
    const grad = new Array(N).fill(0);
    for (let i = N - 1; i >= 0; i--) {
      // grad_i = 2 R u_i + B^T λ_{i+1}
      grad[i] = 2 * this.R * useq[i] + this.B[0] * lambda[0] + this.B[1] * lambda[1];
      // λ_i = 2 Q x_i + A^T λ_{i+1}
      const lambdaPrev: [number, number] = [
        2 * this.Q[0] * xs[i][0] + this.A[0][0] * lambda[0] + this.A[1][0] * lambda[1],
        2 * this.Q[1] * xs[i][1] + this.A[0][1] * lambda[0] + this.A[1][1] * lambda[1],
      ];
      lambda = lambdaPrev;
    }
    return {J, grad};
  }

  /** Projected gradient descent on the box-constrained QP. */
  private solveQP(x0: [number, number]): number[] {
    let useq = this.warmStart.slice();
    let alpha = 0.05;          // step size
    let lastJ = Infinity;
    for (let it = 0; it < 200; it++) {
      const {J, grad} = this.costAndGrad(x0, useq);
      // Simple line search / step-size adaptation.
      if (J > lastJ - 1e-10) alpha *= 0.5;     // shrink if no progress
      else                   alpha = Math.min(alpha * 1.05, 0.1);
      lastJ = J;
      // Step.
      for (let i = 0; i < this.N; i++) useq[i] = this.clip(useq[i] - alpha * grad[i]);
      // Termination on gradient norm (ignoring components fixed at the bound).
      let gnorm = 0;
      for (let i = 0; i < this.N; i++) {
        const ui = useq[i]; const gi = grad[i];
        const atUpper = ui >= this.uMaxScalar - 1e-9 && gi < 0;
        const atLower = ui <= -this.uMaxScalar + 1e-9 && gi > 0;
        if (!atUpper && !atLower) gnorm += gi * gi;
      }
      if (Math.sqrt(gnorm) < 1e-3) break;
    }
    return useq;
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface MPCDoubleIntOpts {
  x0?: [number, number];
  uMax?: number;
  /** Horizon length. Default 15. */
  N?: number;
  /** Stage state weights diag(Q). Default [10, 1]. */
  Q?: [number, number];
  /** Terminal weights diag(Qf). Default [50, 5]. */
  Qf?: [number, number];
  /** Input weight R. Default 0.1. */
  R?: number;
  dt?: number;
  numSteps?: number;
}

export interface MPCDoubleIntResult extends ClosedLoopResult {
  /** First tick at which |x| + |v| < 1e-2. */
  arrivalTick: number;
  /** Maximum |u| over the run — should be ≤ uMax. */
  maxAbsU: number;
}

export function runMPCDoubleIntegrator(opts: MPCDoubleIntOpts = {}): MPCDoubleIntResult {
  const x0: [number, number] = opts.x0 ?? [3, 0];
  const dt = opts.dt ?? 0.1;
  const numSteps = opts.numSteps ?? 100;
  const cls = 'runMPCDoubleIntegrator';
  Preconditions.lengthEq(cls, 'x0', x0, 2);
  Preconditions.allFinite(cls, 'x0', x0);
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.integerInRange(cls, 'numSteps', numSteps, 1, 1e9);
  const plant = new DoubleIntegratorPlant(x0, dt);
  const ctrl = new MPCDoubleIntegratorController({
    N: opts.N, Q: opts.Q, Qf: opts.Qf, R: opts.R, uMax: opts.uMax, dt,
  });
  const out = runClosedLoop(plant, ctrl, {numSteps});

  let arrivalTick = numSteps;
  for (let i = 0; i < out.trajectory.length; i++) {
    const [x, v] = out.trajectory[i];
    if (Math.abs(x) + Math.abs(v) < 1e-2) { arrivalTick = i; break; }
  }
  let maxAbsU = 0;
  for (const u of out.controls) if (Math.abs(u[0]) > maxAbsU) maxAbsU = Math.abs(u[0]);

  return {...out, arrivalTick, maxAbsU};
}
