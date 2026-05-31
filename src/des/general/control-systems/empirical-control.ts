'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/control-systems/empirical-control.rs
//   (module des::general::control_systems::empirical_control)
// 1:1 file move. Quantitative controllability/observability via Gramians + trials.
//
// Declarations → Rust:
//   class Mulberry32                     -> struct Mulberry32 + impl (seedable RNG;
//                                           consider implementing rand::RngCore)
//   class DiscreteLinearSystem           -> struct + impl (+ assoc fn fromContinuous)
//   class GramianDegree                  -> struct + impl (base for the two Gramians)
//   class ControllabilityGramian / ObservabilityGramian extends GramianDegree
//                                        -> struct composing GramianDegree (NO inherit)
//   class MinEnergyController / MonteCarloControllability / MonteCarloObservability /
//         MdpControllabilityDegree / BeliefTracker / MonteCarloDistinguishability
//                                        -> struct + impl
//   interface MonteCarloControllabilityResult / MonteCarloObservabilityResult /
//             PomdpObservabilityResult   -> struct
//   type DegreeKind = 'lti-degree'|...   -> enum DegreeKind (#[serde(rename_all="kebab-case")])
//   class *Token (implements Token)      -> struct + impl Token
//   class *SourceStation / *EvaluatorStation / DegreeReportSinkStation
//         extends DESStation / PureTransformEntity -> struct + impl trait
//
// Conversion notes (file-specific):
//   - Mulberry32 relies on uint32 wraparound (`>>> 0`, Math.imul) -> use u32
//     wrapping_add / wrapping_mul to match the bitstream exactly.
//   - GotChA: `(tracker as unknown as {belief:number[]}).belief = b` reaches into
//     a private field; in Rust add a proper `set_belief(&mut self, ..)` method —
//     do NOT mirror the cast.
//   - extends GramianDegree (ControllabilityGramian/ObservabilityGramian) is
//     config-in-ctor inheritance -> compose a GramianDegree field; the subclass
//     ctor builds the matrix then `super(W)`.
//   - Mat/Vec are shared::linalg aliases; SymmetricEigen/MatrixInverse from there.
//   - default opts via `?? n` -> Option::unwrap_or; seeds kept for reproducibility.
// =============================================================================

// =============================================================================
// control-systems/empirical-control.ts — QUANTITATIVE / EMPIRICAL estimation of
// controllability and observability.
//
// MOTIVATION
// ──────────
//   `observability-controllability.ts` answers the BINARY structural question
//   analytically (Kalman rank = n? reachable? distinguishable?). But a real
//   system is rarely "yes/no": some directions are easy to drive/see and some
//   are nearly impossible. This module measures the DEGREE — the MIN and MAX
//   controllability/observability — two complementary ways:
//
//   A. GRAMIANS (numerical, closed-form-ish):
//        W_c = Σ_{k<H} A^k B Bᵀ (Aᵀ)^k     (controllability Gramian)
//        W_o = Σ_{k<H} (Aᵀ)^k Cᵀ C A^k     (observability Gramian)
//      Their eigenvalues are the squared singular values of the reachability /
//      observability maps. λ_min = hardest direction (min controllability /
//      observability), λ_max = easiest. This is the textbook quantitative
//      measure and the ground truth the trial methods should recover.
//
//   B. TRIAL / SIMULATION (reverse-engineered, no rank algebra):
//        • drive the system with MANY random control sequences and look at
//          where the state actually lands → the empirical reachable cloud's
//          covariance recovers W_c's principal axes (Cov = σ_u²·W_c).
//        • feed MANY random initial states through the output map (with sensor
//          noise) and reconstruct them by least squares → the reconstruction
//          error per direction recovers W_o's weak axes.
//        • for an MDP, run random-policy rollouts and a value-iteration planner
//          to measure how reachable each target is (controllability degree).
//        • for a POMDP, run Monte-Carlo trajectories with Bayesian belief
//          tracking and measure residual entropy / hit-probability about the
//          true state (observability degree).
//
//   Everything is CLASSES with METHODS (LinAlg / SymmetricEigen / MatrixInverse
//   do the numerics; a seedable Mulberry32 RNG drives the trials).
// =============================================================================

import {ChannelName, DESStation, Token} from '../des-base/station';
import {PureTransformEntity} from '../des-base/transform-entity';
import {Preconditions} from '../des-base/preconditions';
import {LinAlg, Mat, MatrixInverse, SymmetricEigen, Vec} from './linear-algebra';
import {MarkovDecisionProcess, PartiallyObservableProcess, StateSpaceModel} from './observability-controllability';

// =============================================================================
// Deterministic RNG (class, seedable) — Mulberry32.
// =============================================================================

export class Mulberry32 {
  private state: number;

  constructor(seed = 0x9e3779b9) {
    this.state = seed >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [-a, a]. */
  uniform(a: number): number {
    return (this.next() * 2 - 1) * a;
  }

  /** Standard normal via Box–Muller. */
  normal(): number {
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Sample an index from a pmf. */
  categorical(pmf: readonly number[]): number {
    const r = this.next();
    let acc = 0;
    for (let i = 0; i < pmf.length; i++) {
      acc += pmf[i];
      if (r <= acc) return i;
    }
    return pmf.length - 1;
  }
}

// =============================================================================
// Discrete linear system  x_{k+1} = Ad x_k + Bd u_k,  y_k = C x_k.
// =============================================================================

export class DiscreteLinearSystem {
  readonly Ad: Mat;
  readonly Bd: Mat;
  readonly C: Mat;

  constructor(Ad: Mat, Bd: Mat, C: Mat) {
    const cls = 'DiscreteLinearSystem';
    Preconditions.squareMatrix(cls, 'Ad', Ad);
    Preconditions.lengthEq(cls, 'Bd', Bd, Ad.length);
    Preconditions.lengthEq(cls, 'C[0]', C[0], Ad.length);
    this.Ad = LinAlg.copy(Ad);
    this.Bd = LinAlg.copy(Bd);
    this.C = LinAlg.copy(C);
  }

  /** Forward-Euler discretisation of a continuous StateSpaceModel. */
  static fromContinuous(model: StateSpaceModel, dt: number): DiscreteLinearSystem {
    Preconditions.positive('DiscreteLinearSystem', 'dt', dt);
    const n = model.stateDim();
    const Ad = LinAlg.add(LinAlg.identity(n), LinAlg.scale(model.A, dt));
    const Bd = LinAlg.scale(model.B, dt);
    return new DiscreteLinearSystem(Ad, Bd, model.C);
  }

  stateDim(): number { return this.Ad.length; }
  inputDim(): number { return LinAlg.cols(this.Bd); }
  outputDim(): number { return this.C.length; }

  step(x: Vec, u: Vec): Vec {
    return LinAlg.add([LinAlg.matVec(this.Ad, x)], [LinAlg.matVec(this.Bd, u)])[0];
  }

  /** Roll the state forward under an input sequence; returns terminal state. */
  rollout(x0: Vec, inputs: readonly Vec[]): Vec {
    let x = x0.slice();
    for (const u of inputs) x = this.step(x, u);
    return x;
  }

  /** Output sequence of length H from x0 under an input sequence (default zero
   *  inputs), i.e. [y_0, …, y_{H-1}]. */
  outputs(x0: Vec, H: number, inputs?: readonly Vec[]): Vec[] {
    const ys: Vec[] = [];
    let x = x0.slice();
    for (let k = 0; k < H; k++) {
      ys.push(LinAlg.matVec(this.C, x));
      const u = inputs ? inputs[k] : new Array<number>(this.inputDim()).fill(0);
      x = this.step(x, u);
    }
    return ys;
  }

  /** Reachability map R = [ A^{H-1}B | … | AB | B ] so x_H = R · stack(u). */
  reachabilityMap(H: number): Mat {
    const blocks: Mat[] = [];
    for (let k = H - 1; k >= 0; k--) blocks.push(LinAlg.matMul(LinAlg.power(this.Ad, k), this.Bd));
    return LinAlg.hstack(blocks);
  }

  /** Observability map O = [ C; CA; …; CA^{H-1} ] so stack(y) = O · x0. */
  observabilityMap(H: number): Mat {
    const blocks: Mat[] = [];
    let cap = this.C;
    blocks.push(cap);
    for (let k = 1; k < H; k++) { cap = LinAlg.matMul(cap, this.Ad); blocks.push(cap); }
    return LinAlg.vstack(blocks);
  }
}

// =============================================================================
// A. GRAMIANS — quantitative controllability / observability degree.
// =============================================================================

/** Shared eigen-summary of a symmetric PSD Gramian. */
export class GramianDegree {
  private readonly eig: SymmetricEigen;
  constructor(readonly gramian: Mat) {
    this.eig = new SymmetricEigen(gramian);
  }
  matrix(): Mat { return this.gramian; }
  eigenvalues(): Vec { return this.eig.values(); }
  /** Min degree (hardest / weakest direction). */
  min(): number { return this.eig.minEigenvalue(); }
  /** Max degree (easiest / strongest direction). */
  max(): number { return this.eig.maxEigenvalue(); }
  /** Direction (unit eigenvector) that is hardest to drive / see. */
  weakestDirection(): Vec { return this.eig.minEigenvector(); }
  /** Direction that is easiest to drive / see. */
  strongestDirection(): Vec { return this.eig.maxEigenvector(); }
  /** λ_max/λ_min — anisotropy; ∞ when a direction is uncontrollable/unobservable. */
  conditionNumber(): number { return this.eig.conditionNumber(); }
}

/** W_c = Σ_{k=0}^{H-1} Ad^k Bd Bdᵀ (Adᵀ)^k. */
export class ControllabilityGramian extends GramianDegree {
  constructor(sys: DiscreteLinearSystem, horizon: number) {
    Preconditions.integerInRange('ControllabilityGramian', 'horizon', horizon, 1, 100000);
    const n = sys.stateDim();
    let W = LinAlg.zeros(n, n);
    let aPow = LinAlg.identity(n);
    const BBt = LinAlg.matMul(sys.Bd, LinAlg.transpose(sys.Bd));
    for (let k = 0; k < horizon; k++) {
      const term = LinAlg.matMul(LinAlg.matMul(aPow, BBt), LinAlg.transpose(aPow));
      W = LinAlg.add(W, term);
      aPow = LinAlg.matMul(aPow, sys.Ad);
    }
    super(W);
  }

  /** Minimum control energy to reach unit-norm state x* in the horizon:
   *  E = x*ᵀ W_c⁻¹ x*. Small λ ⇒ huge energy ⇒ weakly controllable. */
  minEnergyToReach(target: Vec): number {
    const Winv = new MatrixInverse(this.regularised()).inverse();
    const w = LinAlg.matVec(Winv, target);
    let e = 0;
    for (let i = 0; i < target.length; i++) e += target[i] * w[i];
    return e;
  }

  private regularised(): Mat {
    const n = this.matrix().length;
    return LinAlg.add(this.matrix(), LinAlg.scale(LinAlg.identity(n), 1e-12));
  }
}

/** W_o = Σ_{k=0}^{H-1} (Adᵀ)^k Cᵀ C Ad^k. */
export class ObservabilityGramian extends GramianDegree {
  constructor(sys: DiscreteLinearSystem, horizon: number) {
    Preconditions.integerInRange('ObservabilityGramian', 'horizon', horizon, 1, 100000);
    const n = sys.stateDim();
    let W = LinAlg.zeros(n, n);
    let aPow = LinAlg.identity(n);
    const CtC = LinAlg.matMul(LinAlg.transpose(sys.C), sys.C);
    for (let k = 0; k < horizon; k++) {
      const term = LinAlg.matMul(LinAlg.matMul(LinAlg.transpose(aPow), CtC), aPow);
      W = LinAlg.add(W, term);
      aPow = LinAlg.matMul(aPow, sys.Ad);
    }
    super(W);
  }
}

// =============================================================================
// B1. TRIAL-BASED CONTROLLABILITY — random shooting + least-squares targeting.
// =============================================================================

/** Least-squares (minimum-energy) open-loop controller: pick the input stack
 *  u that drives x0=0 → target in H steps. Solved by the right pseudo-inverse
 *  of the reachability map. This is the "reverse-engineer the control" step. */
export class MinEnergyController {
  private readonly R: Mat;          // reachability map (n × H·m)
  private readonly RRtInv: Mat;     // (R Rᵀ + εI)⁻¹  (n × n)

  constructor(private readonly sys: DiscreteLinearSystem, private readonly horizon: number, ridge = 1e-9) {
    this.R = sys.reachabilityMap(horizon);
    const n = sys.stateDim();
    const RRt = LinAlg.add(LinAlg.matMul(this.R, LinAlg.transpose(this.R)), LinAlg.scale(LinAlg.identity(n), ridge));
    this.RRtInv = new MatrixInverse(RRt).inverse();
  }

  /** Stacked input u* = Rᵀ (RRᵀ)⁻¹ target. */
  inputFor(target: Vec): Vec {
    return LinAlg.matVec(LinAlg.transpose(this.R), LinAlg.matVec(this.RRtInv, target));
  }

  /** The state actually reached by u* (= projection of target onto reachable subspace). */
  reachedState(target: Vec): Vec {
    return LinAlg.matVec(this.R, this.inputFor(target));
  }

  /** ‖target − reached‖ — zero iff target lies in the controllable subspace. */
  reachError(target: Vec): number {
    const reached = this.reachedState(target);
    let s = 0;
    for (let i = 0; i < target.length; i++) { const d = target[i] - reached[i]; s += d * d; }
    return Math.sqrt(s);
  }
}

export interface MonteCarloControllabilityResult {
  trials: number;
  /** principal-axis variances of the reached cloud (ascending). */
  spreadEigenvalues: Vec;
  /** fraction of random unit targets reached within tolerance. */
  targetSuccessRate: number;
  /** max ‖x_H‖ observed across random-input rollouts. */
  reachRadius: number;
}

/** Drives the system with MANY random input sequences and analyses where the
 *  state lands, plus how often a least-squares controller hits random targets.
 *  Recovers controllability degree WITHOUT any rank computation. */
export class MonteCarloControllability {
  constructor(
    private readonly sys: DiscreteLinearSystem,
    private readonly horizon: number,
    private readonly opts: {trials?: number; inputBound?: number; targetRadius?: number; tol?: number; seed?: number} = {},
  ) {}

  run(): MonteCarloControllabilityResult {
    const trials = this.opts.trials ?? 2000;
    const uBound = this.opts.inputBound ?? 1;
    const targetRadius = this.opts.targetRadius ?? 1;
    const tol = this.opts.tol ?? 0.05;
    const rng = new Mulberry32(this.opts.seed ?? 12345);
    const n = this.sys.stateDim();
    const m = this.sys.inputDim();

    // 1. Random-input rollouts → reached-state cloud.
    const cloud: Vec[] = [];
    let reachRadius = 0;
    for (let t = 0; t < trials; t++) {
      const inputs: Vec[] = [];
      for (let k = 0; k < this.horizon; k++) {
        inputs.push(Array.from({length: m}, () => rng.uniform(uBound)));
      }
      const xT = this.sys.rollout(new Array<number>(n).fill(0), inputs);
      cloud.push(xT);
      const r = Math.sqrt(xT.reduce((a, v) => a + v * v, 0));
      if (r > reachRadius) reachRadius = r;
    }
    const spread = new SymmetricEigen(this.covariance(cloud)).values();

    // 2. Least-squares targeting of random unit-direction targets.
    const controller = new MinEnergyController(this.sys, this.horizon);
    let hits = 0;
    const probes = Math.min(500, trials);
    for (let t = 0; t < probes; t++) {
      const dir = Array.from({length: n}, () => rng.normal());
      const norm = Math.sqrt(dir.reduce((a, v) => a + v * v, 0)) || 1;
      const target = dir.map(v => (v / norm) * targetRadius);
      if (controller.reachError(target) <= tol * targetRadius) hits++;
    }

    return {trials, spreadEigenvalues: spread, targetSuccessRate: hits / probes, reachRadius};
  }

  private covariance(cloud: readonly Vec[]): Mat {
    const n = cloud[0].length;
    const mean = new Array<number>(n).fill(0);
    for (const x of cloud) for (let i = 0; i < n; i++) mean[i] += x[i] / cloud.length;
    const cov = LinAlg.zeros(n, n);
    for (const x of cloud) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]) / cloud.length;
      }
    }
    return cov;
  }
}

// =============================================================================
// B2. TRIAL-BASED OBSERVABILITY — random states + noisy least-squares recovery.
// =============================================================================

export interface MonteCarloObservabilityResult {
  trials: number;
  /** mean ‖x0 − x̂0‖ across trials at the given sensor-noise level. */
  meanReconstructionError: number;
  /** worst per-trial reconstruction error. */
  worstReconstructionError: number;
  /** observability-Gramian eigenvalues (ascending) for reference. */
  gramianEigenvalues: Vec;
}

/** Feeds MANY random initial states through the (noisy) output map and
 *  reconstructs them by least squares. High error ⇒ weak observability —
 *  recovered by simulation rather than by inspecting rank. */
export class MonteCarloObservability {
  constructor(
    private readonly sys: DiscreteLinearSystem,
    private readonly horizon: number,
    private readonly opts: {trials?: number; noiseStd?: number; stateScale?: number; seed?: number} = {},
  ) {}

  run(): MonteCarloObservabilityResult {
    const trials = this.opts.trials ?? 1000;
    const noiseStd = this.opts.noiseStd ?? 0.01;
    const scale = this.opts.stateScale ?? 1;
    const rng = new Mulberry32(this.opts.seed ?? 2024);
    const n = this.sys.stateDim();
    const O = this.sys.observabilityMap(this.horizon);
    // Least-squares reconstruction operator: (OᵀO + εI)⁻¹ Oᵀ.
    const OtO = LinAlg.add(LinAlg.matMul(LinAlg.transpose(O), O), LinAlg.scale(LinAlg.identity(n), 1e-9));
    const recon = LinAlg.matMul(new MatrixInverse(OtO).inverse(), LinAlg.transpose(O));

    let sum = 0, worst = 0;
    for (let t = 0; t < trials; t++) {
      const x0 = Array.from({length: n}, () => rng.normal() * scale);
      const ys = this.sys.outputs(x0, this.horizon);
      const stacked: Vec = [];
      for (const y of ys) for (const yi of y) stacked.push(yi + rng.normal() * noiseStd);
      const xhat = LinAlg.matVec(recon, stacked);
      let e = 0;
      for (let i = 0; i < n; i++) { const d = x0[i] - xhat[i]; e += d * d; }
      e = Math.sqrt(e);
      sum += e;
      if (e > worst) worst = e;
    }
    return {
      trials,
      meanReconstructionError: sum / trials,
      worstReconstructionError: worst,
      gramianEigenvalues: new ObservabilityGramian(this.sys, this.horizon).eigenvalues(),
    };
  }
}

// =============================================================================
// B3. MDP CONTROLLABILITY DEGREE — value iteration + random-policy rollouts.
// =============================================================================

export class MdpControllabilityDegree {
  constructor(private readonly mdp: MarkovDecisionProcess) {}

  /** Min expected number of steps to reach `target` from each state under the
   *  best action (Bellman value iteration). Unreachable ⇒ +Infinity. */
  expectedHittingTimes(target: number, iters = 1000, tol = 1e-9): Vec {
    const n = this.mdp.numStates;
    let V = new Array<number>(n).fill(Infinity);
    V[target] = 0;
    for (let it = 0; it < iters; it++) {
      let delta = 0;
      const next = V.slice();
      for (let s = 0; s < n; s++) {
        if (s === target) continue;
        let best = Infinity;
        for (let a = 0; a < this.mdp.numActions; a++) {
          let exp = 1;
          let finite = true;
          for (let t = 0; t < n; t++) {
            const p = this.mdp.transition[a][s][t];
            if (p <= 0) continue;
            if (!Number.isFinite(V[t])) { finite = false; break; }
            exp += p * V[t];
          }
          if (finite && exp < best) best = exp;
        }
        next[s] = best;
        if (Number.isFinite(best)) delta = Math.max(delta, Math.abs(best - V[s]));
      }
      V = next;
      if (delta < tol) break;
    }
    return V;
  }

  /** Empirical reach frequency to `target` within `horizon` steps under a
   *  uniform-random policy — the trial-based controllability estimate. */
  randomPolicyReachRate(target: number, opts: {episodes?: number; horizon?: number; seed?: number} = {}): Vec {
    const episodes = opts.episodes ?? 400;
    const horizon = opts.horizon ?? this.mdp.numStates * 4;
    const rng = new Mulberry32(opts.seed ?? 7);
    const n = this.mdp.numStates;
    const rate = new Array<number>(n).fill(0);
    for (let s0 = 0; s0 < n; s0++) {
      let hits = 0;
      for (let e = 0; e < episodes; e++) {
        let s = s0;
        for (let k = 0; k < horizon; k++) {
          if (s === target) { hits++; break; }
          const a = Math.floor(rng.next() * this.mdp.numActions);
          s = rng.categorical(this.mdp.transition[a][s]);
        }
        if (s === target) hits++;     // counts terminal landing too
      }
      rate[s0] = Math.min(1, hits / episodes);
    }
    return rate;
  }

  /** Controllability degree per target = mean reach rate from all sources.
   *  Returns the per-target degrees; min/max are the worst/best targets. */
  perTargetDegree(opts: {episodes?: number; horizon?: number; seed?: number} = {}): Vec {
    const n = this.mdp.numStates;
    const deg = new Array<number>(n).fill(0);
    for (let t = 0; t < n; t++) {
      const rates = this.randomPolicyReachRate(t, opts);
      deg[t] = rates.reduce((a, v) => a + v, 0) / n;
    }
    return deg;
  }
}

// =============================================================================
// B4. POMDP OBSERVABILITY DEGREE — Bayesian belief tracking + Monte-Carlo.
// =============================================================================

/** Maintains a belief (pmf over states) and updates it by Bayes' rule on
 *  (action, observation) pairs. */
export class BeliefTracker {
  private belief: number[];

  constructor(private readonly pomdp: PartiallyObservableProcess, prior?: number[]) {
    const n = pomdp.mdp.numStates;
    this.belief = prior ? prior.slice() : new Array<number>(n).fill(1 / n);
  }

  current(): number[] { return this.belief.slice(); }

  /** Predict with action a, then correct with observation o. */
  update(action: number, observation: number): void {
    const n = this.pomdp.mdp.numStates;
    const predicted = new Array<number>(n).fill(0);
    for (let s = 0; s < n; s++) {
      if (this.belief[s] === 0) continue;
      for (let t = 0; t < n; t++) predicted[t] += this.belief[s] * this.pomdp.mdp.transition[action][s][t];
    }
    let z = 0;
    for (let t = 0; t < n; t++) { predicted[t] *= this.pomdp.observation[t][observation]; z += predicted[t]; }
    if (z > 0) for (let t = 0; t < n; t++) predicted[t] /= z;
    this.belief = predicted;
  }

  /** Shannon entropy of the current belief (bits). */
  entropy(): number {
    let h = 0;
    for (const p of this.belief) if (p > 0) h -= p * Math.log2(p);
    return h;
  }
}

export interface PomdpObservabilityResult {
  /** per-true-state probability mass assigned to the true state after H steps. */
  hitProbability: Vec;
  /** per-true-state mean residual belief entropy (bits) after H steps. */
  residualEntropy: Vec;
  /** overall observability degree in [0,1] (min over states of hit prob). */
  minDegree: number;
  maxDegree: number;
}

/** Runs MANY simulated trajectories from each true state, tracks the belief,
 *  and measures how concentrated the belief becomes on the true state. The
 *  numerical/simulation analog of "can the outputs reveal the state?". */
export class MonteCarloDistinguishability {
  constructor(private readonly pomdp: PartiallyObservableProcess) {}

  run(opts: {episodes?: number; horizon?: number; seed?: number} = {}): PomdpObservabilityResult {
    const episodes = opts.episodes ?? 400;
    const horizon = opts.horizon ?? this.pomdp.mdp.numStates * 4;
    const rng = new Mulberry32(opts.seed ?? 99);
    const n = this.pomdp.mdp.numStates;
    const hit = new Array<number>(n).fill(0);
    const ent = new Array<number>(n).fill(0);

    for (let s0 = 0; s0 < n; s0++) {
      for (let e = 0; e < episodes; e++) {
        const tracker = new BeliefTracker(this.pomdp);
        let s = s0;
        // First observation (no action yet) to seed the belief.
        let o = rng.categorical(this.pomdp.observation[s]);
        this.bayesObserve(tracker, o);
        for (let k = 0; k < horizon; k++) {
          const a = Math.floor(rng.next() * this.pomdp.mdp.numActions);
          s = rng.categorical(this.pomdp.mdp.transition[a][s]);
          o = rng.categorical(this.pomdp.observation[s]);
          tracker.update(a, o);
        }
        hit[s0] += tracker.current()[s] / episodes;
        ent[s0] += tracker.entropy() / episodes;
      }
    }
    return {
      hitProbability: hit,
      residualEntropy: ent,
      minDegree: Math.min(...hit),
      maxDegree: Math.max(...hit),
    };
  }

  private bayesObserve(tracker: BeliefTracker, observation: number): void {
    const b = tracker.current();
    const n = this.pomdp.mdp.numStates;
    let z = 0;
    for (let s = 0; s < n; s++) { b[s] *= this.pomdp.observation[s][observation]; z += b[s]; }
    if (z > 0) for (let s = 0; s < n; s++) b[s] /= z;
    // Re-seed the tracker with the corrected prior.
    (tracker as unknown as {belief: number[]}).belief = b;
  }
}

// =============================================================================
// DES PIPELINE — empirical evaluators as stations.
// =============================================================================

export class EmpiricalChannels {
  static readonly SYSTEM: ChannelName = 'empirical-system';
  static readonly MDP: ChannelName = 'empirical-mdp';
  static readonly POMDP: ChannelName = 'empirical-pomdp';
  static readonly REPORT: ChannelName = 'empirical-report';
}

export class DiscreteSystemToken implements Token {
  constructor(readonly label: string, readonly sys: DiscreteLinearSystem, readonly horizon: number) {}
}

export class MdpDegreeToken implements Token {
  constructor(readonly label: string, readonly mdp: MarkovDecisionProcess) {}
}

export class PomdpDegreeToken implements Token {
  constructor(readonly label: string, readonly pomdp: PartiallyObservableProcess) {}
}

export type DegreeKind = 'lti-degree' | 'mdp-degree' | 'pomdp-degree';

/** A quantitative min/max degree report flowing on the report channel. */
export class DegreeReportToken implements Token {
  constructor(
    readonly label: string,
    readonly kind: DegreeKind,
    readonly minControllability: number,
    readonly maxControllability: number,
    readonly minObservability: number,
    readonly maxObservability: number,
    readonly detail: string,
  ) {}
}

export class DiscreteSystemSourceStation extends DESStation {
  private emitted = false;
  constructor(id: string, private readonly items: readonly DiscreteSystemToken[]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    for (const it of this.items) this.emit(it, EmpiricalChannels.SYSTEM);
    this.emitted = true;
  }
}

export class MdpDegreeSourceStation extends DESStation {
  private emitted = false;
  constructor(id: string, private readonly items: readonly MdpDegreeToken[]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    for (const it of this.items) this.emit(it, EmpiricalChannels.MDP);
    this.emitted = true;
  }
}

export class PomdpDegreeSourceStation extends DESStation {
  private emitted = false;
  constructor(id: string, private readonly items: readonly PomdpDegreeToken[]) { super(id); }
  override hasWork(): boolean { return !this.emitted; }
  runTimeStep(): void {
    if (this.emitted) return;
    for (const it of this.items) this.emit(it, EmpiricalChannels.POMDP);
    this.emitted = true;
  }
}

/** Gramian-based min/max controllability & observability degree for an LTI. */
export class LtiDegreeEvaluatorStation extends PureTransformEntity<DiscreteSystemToken, DegreeReportToken> {
  constructor(id: string) {
    super(id, {inputChannels: EmpiricalChannels.SYSTEM, outputChannel: EmpiricalChannels.REPORT});
  }
  transform(token: DiscreteSystemToken): DegreeReportToken {
    const wc = new ControllabilityGramian(token.sys, token.horizon);
    const wo = new ObservabilityGramian(token.sys, token.horizon);
    return new DegreeReportToken(
      token.label, 'lti-degree', wc.min(), wc.max(), wo.min(), wo.max(),
      `W_c λ∈[${wc.min().toExponential(2)}, ${wc.max().toExponential(2)}] (cond ${this.fmtCond(wc.conditionNumber())}); ` +
      `W_o λ∈[${wo.min().toExponential(2)}, ${wo.max().toExponential(2)}] (cond ${this.fmtCond(wo.conditionNumber())})`,
    );
  }
  private fmtCond(c: number): string { return Number.isFinite(c) ? c.toExponential(1) : '∞'; }
}

/** Random-policy reach degree (controllability) for an MDP. */
export class MdpDegreeEvaluatorStation extends PureTransformEntity<MdpDegreeToken, DegreeReportToken> {
  constructor(id: string) {
    super(id, {inputChannels: EmpiricalChannels.MDP, outputChannel: EmpiricalChannels.REPORT});
  }
  transform(token: MdpDegreeToken): DegreeReportToken {
    const deg = new MdpControllabilityDegree(token.mdp).perTargetDegree();
    return new DegreeReportToken(
      token.label, 'mdp-degree', Math.min(...deg), Math.max(...deg), NaN, NaN,
      `random-policy reach degree per target: [${deg.map(d => d.toFixed(2)).join(', ')}]`,
    );
  }
}

/** Belief-tracking distinguishability degree (observability) for a POMDP. */
export class PomdpDegreeEvaluatorStation extends PureTransformEntity<PomdpDegreeToken, DegreeReportToken> {
  constructor(id: string) {
    super(id, {inputChannels: EmpiricalChannels.POMDP, outputChannel: EmpiricalChannels.REPORT});
  }
  transform(token: PomdpDegreeToken): DegreeReportToken {
    const r = new MonteCarloDistinguishability(token.pomdp).run();
    return new DegreeReportToken(
      token.label, 'pomdp-degree', NaN, NaN, r.minDegree, r.maxDegree,
      `belief hit-prob per state: [${r.hitProbability.map(d => d.toFixed(2)).join(', ')}]; ` +
      `residual entropy: [${r.residualEntropy.map(d => d.toFixed(2)).join(', ')}] bits`,
    );
  }
}

export class DegreeReportSinkStation extends DESStation {
  readonly reports: DegreeReportToken[] = [];
  constructor(id: string) { super(id); }
  override hasWork(): boolean { return this.inboxSize(EmpiricalChannels.REPORT) > 0; }
  runTimeStep(): void { this.reports.push(...this.drain<DegreeReportToken>(EmpiricalChannels.REPORT)); }
}
