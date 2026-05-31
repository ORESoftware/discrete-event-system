'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/lqr_controller.rs  (module des::general::des_base::lqr_controller)
// 1:1 file move. Infinite-horizon LQR: solves the DARE by fixed-point iteration,
// then runs u = -Kx as a ControllerStation control law.
//
// Declarations → Rust:
//   type Vec = number[] / Mat = number[][] -> type Vec = Vec<f64>; type Mat = Vec<Vec<f64>>;
//                                             (or `nalgebra`/`ndarray` types)
//   interface LQRSpec               -> struct LQRSpec (#[derive(Clone)]; Option fields)
//   class LQRController             -> struct: ControllerStation<Vec, Vec>
//   fn matCopy/matT/matMul/matAdd/matSub/matScale/matMV/matInv -> reuse shared/linalg.rs
//                                       (LinAlg/VecOps/MatrixInverse) — do NOT re-port these
//
// Conversion notes (file-specific):
//   - LOCAL matrix helpers duplicate shared/linalg.ts — in Rust delete them and
//     use `crate::des::shared::linalg::*`; the `export {matMul, ...}` re-export goes away.
//   - Implements ControllerStation's controlLaw + overrides clamp -> required/over-
//     ridden trait fns; the Riccati solve happens in the constructor.
//   - `matInv` throws on singular -> return `Result` / `panic!`; DARE precondition
//     (R positive-definite) already guarded via preconditions.rs Cholesky.
//   - non-ASCII `γ` -> `gamma`.
//   - All-`number` matrices -> `f64`.
//   - `uMinVec/uMaxVec?: Vec` -> `Option<Vec<f64>>`.
// =============================================================================

// =============================================================================
// general/des-base/lqr-controller.ts — base class for the LINEAR QUADRATIC
// REGULATOR, the canonical "stochastic control = MDP" example.
//
// CONTROL PROBLEM
// ───────────────
//   Plant:  x_{k+1} = A x_k + B u_k + w_k     (w_k ∼ N(0, Σ_w), optional)
//   Cost:   J = E[ Σ_k x_k^T Q x_k + u_k^T R u_k ]
//
//   Optimal control law (infinite-horizon, discounted γ ∈ (0, 1]):
//     u_k = −K x_k     where    K = (B^T P B + R)^{−1} B^T P A
//   and P solves the discrete-time algebraic Riccati equation (DARE):
//     P = Q + γ A^T P A − γ A^T P B (B^T P B + R)^{−1} B^T P A
//
//   We solve the DARE iteratively (the standard fixed-point iteration
//   converges for stabilisable (A, B) and detectable (Q^{1/2}, A)) and
//   then run the resulting affine state-feedback law as the
//   `controlLaw` hook of `ControllerStation`.
//
// AS A DES STATION
// ────────────────
//   `LQRController extends ControllerStation<Vector, Vector>`. The
//   `step()` API and the runTimeStep machinery are inherited; only
//   the control law is provided. Saturation can be added by overriding
//   `uMin / uMax` per component.
//
// VECTOR ARITHMETIC
// ─────────────────
//   We use plain `number[]` for vectors and `number[][]` for matrices
//   (row-major), zero deps. Tiny helper utilities (matMul, matInv,
//   matAdd, matMV) live at the bottom.
// =============================================================================

import {ControllerStation} from './controller';
import {Preconditions} from './preconditions';

export type Vec = number[];
export type Mat = number[][];

export interface LQRSpec {
  /** State dimension n. */
  n: number;
  /** Control dimension m. */
  m: number;
  /** A (n × n). */
  A: Mat;
  /** B (n × m). */
  B: Mat;
  /** State cost Q (n × n, symmetric PSD). */
  Q: Mat;
  /** Control cost R (m × m, symmetric PD). */
  R: Mat;
  /** Discount factor γ ∈ (0, 1]. Default 1. */
  gamma?: number;
  /** Per-component saturation [u_min, u_max]^m. Default no clamp. */
  uMinVec?: Vec;
  uMaxVec?: Vec;
  /** Riccati iteration tolerance. Default 1e-10. */
  riccatiTol?: number;
  /** Riccati iteration max iters. Default 5000. */
  riccatiMaxIter?: number;
}

export class LQRController extends ControllerStation<Vec, Vec> {
  readonly spec: LQRSpec;
  /** Optimal feedback gain K (m × n). */
  readonly K: Mat;
  /** Riccati solution P (n × n). */
  readonly P: Mat;
  readonly riccatiIters: number;
  readonly riccatiResidual: number;

  constructor(id: string, spec: LQRSpec) {
    super(id);
    this.spec = spec;
    // Pre-construction guards — DARE math is only valid if these hold.
    const cls = 'LQRController';
    Preconditions.integerInRange(cls, 'spec.n', spec.n, 1, 10_000);
    Preconditions.integerInRange(cls, 'spec.m', spec.m, 1, 10_000);
    Preconditions.lengthEq(cls, 'spec.A', spec.A, spec.n);
    Preconditions.rectangularMatrix(cls, 'spec.A', spec.A);
    Preconditions.lengthEq(cls, 'spec.A[0]', spec.A[0], spec.n);
    Preconditions.lengthEq(cls, 'spec.B', spec.B, spec.n);
    Preconditions.rectangularMatrix(cls, 'spec.B', spec.B);
    Preconditions.lengthEq(cls, 'spec.B[0]', spec.B[0], spec.m);
    Preconditions.symmetricMatrix(cls, 'spec.Q', spec.Q);
    Preconditions.lengthEq(cls, 'spec.Q', spec.Q, spec.n);
    Preconditions.positiveSemidefiniteDiag(cls, 'spec.Q', spec.Q);
    Preconditions.symmetricMatrix(cls, 'spec.R', spec.R);
    Preconditions.lengthEq(cls, 'spec.R', spec.R, spec.m);
    // R MUST be positive-definite (we take its inverse). Cholesky test
    // catches user errors like R = 0.
    Preconditions.positiveDefiniteCholesky(cls, 'spec.R', spec.R);
    if (spec.gamma !== undefined) {
      Preconditions.inRange(cls, 'spec.gamma', spec.gamma, 1e-9, 1);
    }
    if (spec.uMinVec) {
      Preconditions.lengthEq(cls, 'spec.uMinVec', spec.uMinVec, spec.m);
      Preconditions.allFinite(cls, 'spec.uMinVec', spec.uMinVec);
    }
    if (spec.uMaxVec) {
      Preconditions.lengthEq(cls, 'spec.uMaxVec', spec.uMaxVec, spec.m);
      Preconditions.allFinite(cls, 'spec.uMaxVec', spec.uMaxVec);
    }
    if (spec.uMinVec && spec.uMaxVec) {
      for (let i = 0; i < spec.m; i++) {
        Preconditions.check(cls, `uMin[${i}] <= uMax[${i}]`, 'satisfy uMin <= uMax',
          spec.uMinVec[i] <= spec.uMaxVec[i], [spec.uMinVec[i], spec.uMaxVec[i]]);
      }
    }
    const γ = spec.gamma ?? 1;
    const tol = spec.riccatiTol ?? 1e-10;
    const maxIter = spec.riccatiMaxIter ?? 5000;
    Preconditions.positive(cls, 'riccatiTol', tol);
    Preconditions.integerInRange(cls, 'riccatiMaxIter', maxIter, 1, 10_000_000);
    let P = matCopy(spec.Q);
    let iter = 0; let res = Infinity;
    for (; iter < maxIter; iter++) {
      // R_eff = B^T P B + R
      const BtP = matMul(matT(spec.B), P);
      const BtPB = matMul(BtP, spec.B);
      const Reff = matAdd(BtPB, spec.R);
      const ReffInv = matInv(Reff);
      // K = R_eff^{-1} B^T P A
      const BtPA = matMul(BtP, spec.A);
      const K = matMul(ReffInv, BtPA);
      // P_new = Q + γ A^T P A − γ A^T P B K
      const AtP = matMul(matT(spec.A), P);
      const AtPA = matMul(AtP, spec.A);
      const AtPB = matMul(AtP, spec.B);
      const AtPBK = matMul(AtPB, K);
      const Pnew = matAdd(spec.Q, matScale(matSub(AtPA, AtPBK), γ));
      // Residual = ||P_new − P||_∞
      let r = 0;
      for (let i = 0; i < spec.n; i++) for (let j = 0; j < spec.n; j++) {
        const d = Math.abs(Pnew[i][j] - P[i][j]);
        if (d > r) r = d;
      }
      P = Pnew;
      res = r;
      if (r < tol) { iter += 1; break; }
    }
    this.P = P;
    this.riccatiIters = iter;
    this.riccatiResidual = res;
    // Final K from final P.
    const BtP = matMul(matT(spec.B), P);
    const BtPB = matMul(BtP, spec.B);
    const Reff = matAdd(BtPB, spec.R);
    const ReffInv = matInv(Reff);
    const BtPA = matMul(BtP, spec.A);
    this.K = matMul(ReffInv, BtPA);
  }

  /** u = −K x, then per-component saturation. */
  protected controlLaw(observation: Vec, _tick: number, _time: number): Vec {
    const Kx = matMV(this.K, observation);
    const u = new Array<number>(this.spec.m);
    for (let i = 0; i < this.spec.m; i++) u[i] = -Kx[i];
    return u;
  }

  protected override clamp(u: Vec): Vec {
    const lo = this.spec.uMinVec; const hi = this.spec.uMaxVec;
    if (!lo && !hi) return u;
    const v = u.slice();
    for (let i = 0; i < v.length; i++) {
      if (lo && v[i] < lo[i]) v[i] = lo[i];
      if (hi && v[i] > hi[i]) v[i] = hi[i];
    }
    return v;
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getGain(): Mat { return this.K; }
  getRiccatiP(): Mat { return this.P; }

  /** J(x_0) under the optimal policy in the DARE solution: x_0^T P x_0. */
  optimalCostFromInitialState(x0: Vec): number {
    const Px = matMV(this.P, x0);
    let v = 0;
    for (let i = 0; i < x0.length; i++) v += x0[i] * Px[i];
    return v;
  }
}

// -----------------------------------------------------------------------------
// MATRIX HELPERS (zero-dep, small-matrix friendly)
// -----------------------------------------------------------------------------

function matCopy(M: Mat): Mat { return M.map(r => r.slice()); }
function matT(M: Mat): Mat {
  const r = M.length; const c = M[0].length;
  const out: Mat = Array.from({length: c}, () => new Array(r).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = M[i][j];
  return out;
}
function matMul(A: Mat, B: Mat): Mat {
  const ra = A.length; const ca = A[0].length; const cb = B[0].length;
  if (B.length !== ca) throw new Error(`matMul shape mismatch: ${ra}×${ca} × ${B.length}×${cb}`);
  const out: Mat = Array.from({length: ra}, () => new Array(cb).fill(0));
  for (let i = 0; i < ra; i++) for (let k = 0; k < ca; k++) {
    const a = A[i][k]; if (a === 0) continue;
    for (let j = 0; j < cb; j++) out[i][j] += a * B[k][j];
  }
  return out;
}
function matAdd(A: Mat, B: Mat): Mat {
  const r = A.length; const c = A[0].length;
  const out: Mat = Array.from({length: r}, () => new Array(c).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] + B[i][j];
  return out;
}
function matSub(A: Mat, B: Mat): Mat {
  const r = A.length; const c = A[0].length;
  const out: Mat = Array.from({length: r}, () => new Array(c).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] - B[i][j];
  return out;
}
function matScale(A: Mat, s: number): Mat {
  return A.map(r => r.map(x => x * s));
}
function matMV(M: Mat, v: Vec): Vec {
  const r = M.length;
  const out = new Array<number>(r).fill(0);
  for (let i = 0; i < r; i++) {
    let acc = 0;
    for (let j = 0; j < v.length; j++) acc += M[i][j] * v[j];
    out[i] = acc;
  }
  return out;
}
/** Gauss-Jordan inverse for small dense matrices. Throws on singular. */
function matInv(M: Mat): Mat {
  const n = M.length;
  const a: Mat = M.map(r => r.slice());
  const inv: Mat = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => i === j ? 1 : 0));
  for (let col = 0; col < n; col++) {
    // Pivot.
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error(`matInv: singular matrix at col ${col}`);
    if (pivot !== col) { [a[col], a[pivot]] = [a[pivot], a[col]]; [inv[col], inv[pivot]] = [inv[pivot], inv[col]]; }
    const p = a[col][col];
    for (let j = 0; j < n; j++) { a[col][j] /= p; inv[col][j] /= p; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r][col];
      if (f === 0) continue;
      for (let j = 0; j < n; j++) { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
    }
  }
  return inv;
}

export {matMul, matT, matAdd, matSub, matScale, matInv, matMV, matCopy};
