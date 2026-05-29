'use strict';

// =============================================================================
// general/sliding-mode-control.ts — SLIDING MODE CONTROL (Utkin 1977,
// Edwards & Spurgeon 1998), the canonical *robust* control method:
// guaranteed convergence even under matched disturbances of known bound.
//
// PROBLEM
// ───────
//   Plant (uncertain second-order):
//
//       ẍ = u + d(t)        |d(t)| ≤ D     unknown bounded disturbance
//
//   Goal: drive (x, ẋ) → (0, 0).
//
//   SMC trick: define a SLIDING SURFACE
//
//       s(x, ẋ) = ẋ + λ x
//
//   so that on the surface (s = 0) we have ẋ = −λ x → x → 0
//   exponentially with rate λ. Choose u so that
//
//       s · ṡ ≤ −η |s|           (η > D, the "reaching" condition)
//
//   then by the Lyapunov function V = ½ s² the trajectory reaches s = 0
//   in FINITE TIME, regardless of the disturbance magnitude (so long as
//   |d| ≤ D < η).
//
//   Concrete law:
//
//       u = −λ ẋ − η · sign(s)
//
//   We replace `sign(s)` with a saturation `tanh(s / boundary)` (a
//   common smoothing) to suppress the textbook chattering, the famous
//   downside of vanilla SMC.
//
// AS A DES BLOCK
// ──────────────
//   `SlidingModeController extends ControllerBlock`. The plant
//   `UncertainDoubleIntegratorPlant` adds an unknown bounded
//   disturbance d(t) to ẍ on every tick. The closed-loop demo verifies
//   that despite the disturbance, the state converges to a small
//   neighbourhood of the origin (proportional to boundary layer width).
// =============================================================================

import {
  PlantBlock, ControllerBlock, runClosedLoop, ClosedLoopResult,
} from './des-base/control-blocks';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// PLANT WITH UNKNOWN BOUNDED DISTURBANCE
// -----------------------------------------------------------------------------

class UncertainDoubleIntegratorPlant extends PlantBlock {
  /** Disturbance generator d(t) : ℝ → ℝ, |d(t)| ≤ D. */
  private readonly disturbance: (t: number) => number;
  /** Internal time. */
  private t = 0;

  constructor(x0: [number, number], dt: number, disturbance: (t: number) => number) {
    super('smc-plant', x0, dt, 1);
    this.disturbance = disturbance;
  }
  protected dynamics(x: readonly number[], u: readonly number[], dt: number): number[] {
    const d = this.disturbance(this.t);
    this.t += dt;
    const ueff = u[0] + d;
    return [x[0] + dt * x[1] + 0.5 * dt * dt * ueff,
            x[1] + dt * ueff];
  }
}

// -----------------------------------------------------------------------------
// SLIDING-MODE CONTROLLER
// -----------------------------------------------------------------------------

class SlidingModeController extends ControllerBlock {
  private readonly lambda: number;
  private readonly eta: number;
  private readonly boundary: number;
  private readonly uBound: number;

  constructor(opts: {lambda: number; eta: number; boundary: number; uBound: number}) {
    super('sliding-mode', 1);
    const cls = 'SlidingModeController';
    Preconditions.positive(cls, 'lambda', opts.lambda);
    Preconditions.positive(cls, 'eta', opts.eta);
    Preconditions.positive(cls, 'boundary', opts.boundary);
    Preconditions.positive(cls, 'uBound', opts.uBound);
    this.lambda = opts.lambda;
    this.eta = opts.eta;
    this.boundary = opts.boundary;
    this.uBound = opts.uBound;
    this.setSaturation([-opts.uBound], [opts.uBound]);
  }
  protected controlLaw(y: readonly number[]): number[] {
    const x = y[0]; const v = y[1];
    const s = v + this.lambda * x;
    // Smoothed sign: tanh(s / boundary) instead of sign(s) to suppress chatter.
    const sat = Math.tanh(s / this.boundary);
    const u = -this.lambda * v - this.eta * sat;
    return [u];
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface SlidingModeOpts {
  x0?: [number, number];
  dt?: number;
  numSteps?: number;
  /** Sliding-surface gain λ. Default 2. */
  lambda?: number;
  /** Reaching gain η. Must exceed disturbance bound D. Default 3. */
  eta?: number;
  /** Boundary-layer width (smoothing). Default 0.05. */
  boundary?: number;
  /** Bound on |u|. Default 5. */
  uBound?: number;
  /** Disturbance amplitude D. Default 1. */
  disturbanceAmp?: number;
  /** Disturbance type: 'sin', 'square', 'random'. Default 'sin'. */
  disturbanceType?: 'sin' | 'square' | 'random';
  seed?: number;
}

export interface SlidingModeResult extends ClosedLoopResult {
  /** Final |x| + |v|. */
  finalDistanceFromOrigin: number;
  /** First tick at which |s(x, v)| < boundary (sliding surface reached). */
  reachingTick: number;
  /** True iff state stays in a neighbourhood of size ≤ 0.1 after t=numSteps/2. */
  stayedNearOrigin: boolean;
}

export function runSlidingMode(opts: SlidingModeOpts = {}): SlidingModeResult {
  const x0: [number, number] = opts.x0 ?? [3, 0];
  const dt = opts.dt ?? 0.05;
  const numSteps = opts.numSteps ?? 400;
  const lambda = opts.lambda ?? 2;
  const eta = opts.eta ?? 3;
  const boundary = opts.boundary ?? 0.05;
  const uBound = opts.uBound ?? 5;
  const D = opts.disturbanceAmp ?? 1;
  const dType = opts.disturbanceType ?? 'sin';
  // Pre-run guards.
  const cls = 'runSlidingMode';
  Preconditions.lengthEq(cls, 'x0', x0, 2);
  Preconditions.allFinite(cls, 'x0', x0);
  Preconditions.positive(cls, 'dt', dt);
  Preconditions.integerInRange(cls, 'numSteps', numSteps, 1, 1e9);
  Preconditions.positive(cls, 'lambda', lambda);
  Preconditions.positive(cls, 'eta', eta);
  Preconditions.positive(cls, 'boundary', boundary);
  Preconditions.positive(cls, 'uBound', uBound);
  Preconditions.nonNegative(cls, 'disturbanceAmp', D);
  // CORE SMC reaching condition: η must strictly exceed disturbance bound.
  Preconditions.check(cls, 'eta > disturbanceAmp',
    'satisfy the SMC reaching condition (eta strictly > D)',
    eta > D, {eta, D});
  Preconditions.check(cls, 'disturbanceType', "be one of 'sin'/'square'/'random'",
    dType === 'sin' || dType === 'square' || dType === 'random', dType);
  const rng = mulberry32(opts.seed ?? 1);

  const dist = (t: number) => {
    if (dType === 'sin')   return D * Math.sin(2 * Math.PI * 0.5 * t);
    if (dType === 'square') return (Math.floor(t * 2) % 2 === 0 ? D : -D);
    return D * (2 * rng() - 1);
  };
  const plant = new UncertainDoubleIntegratorPlant(x0, dt, dist);
  const ctrl = new SlidingModeController({lambda, eta, boundary, uBound});
  const out = runClosedLoop(plant, ctrl, {numSteps});

  let reachingTick = -1;
  for (let i = 0; i < out.trajectory.length; i++) {
    const [x, v] = out.trajectory[i];
    const s = v + lambda * x;
    if (Math.abs(s) < boundary) { reachingTick = i; break; }
  }
  const halfPoint = Math.floor(numSteps / 2);
  let stayedNearOrigin = true;
  for (let i = halfPoint; i < out.trajectory.length; i++) {
    const [x, v] = out.trajectory[i];
    if (Math.abs(x) + Math.abs(v) > 0.5) { stayedNearOrigin = false; break; }
  }
  const final = out.trajectory[out.trajectory.length - 1];
  return {
    ...out,
    finalDistanceFromOrigin: Math.abs(final[0]) + Math.abs(final[1]),
    reachingTick: reachingTick < 0 ? numSteps : reachingTick,
    stayedNearOrigin,
  };
}
