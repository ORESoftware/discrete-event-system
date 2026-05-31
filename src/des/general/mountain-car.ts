// RUST MIGRATION: target module src/des/general/mountain_car.rs.
// RUST MIGRATION: MountainCarOpts, MountainCarTrainOpts, and MountainCarResult become serde structs; episode traces should be Vec rows with explicit numeric fields.
// RUST MIGRATION: MountainCarEnv implements Environment and MountainCarLinearVFA implements LinearVFAStation behavior through Rust traits.
// RUST MIGRATION: runMountainCar is a training PureTransform returning Result; tile-coding/value-function tables should use Vec<f64> or HashMap feature indexes.
// RUST MIGRATION: All exploration and environment randomness must use injected rand::Rng.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/mountain-car.rs  (module des::general::mountain_car)
// 1:1 file move. Mountain Car control via tile-coded linear value-function approximation.
//
// Declarations → Rust:
//   interface MountainCarOpts/MountainCarTrainOpts/MountainCarResult -> structs (Default where sensible)
//   class MountainCarEnv implements Environment        -> struct + impl Environment trait
//   class MountainCarLinearVFA extends LinearVFAStation<number> -> struct + impl VFA-station trait
//   fn runMountainCar                                  -> free fn / assoc fn
//
// Conversion notes (file-specific):
//   - INJECT RNG: `Math.random()` for the random start state and `rng: () => number` for the
//     agent -> `RandomSource` (shared/capabilities); thread the SAME source for determinism.
//   - `states: Map<number, [number, number]>` -> `HashMap<usize, (f64, f64)>` (episode-id keyed).
//   - REMOVE `as any`: `(greedyEnv as any).nextId++` / `.states.set(...)` reach private fields
//     across env instances — in Rust expose typed accessors or restructure the greedy-eval clone
//     so there is no cast.
//   - tile-coded feature vector is binary over numTilings×posBins×velBins -> `Vec<f64>`/index set.
// =============================================================================
// general/mountain-car.ts — the classic MOUNTAIN CAR control problem
// (Moore 1990 → Sutton & Barto 1998 §10.1) solved with linear function
// approximation over a TILE-CODED feature set.
//
// PROBLEM
// ───────
//   Continuous state x = (position, velocity). Underpowered car at the
//   bottom of a valley must "rock" back and forth to build momentum and
//   reach the goal at position ≥ 0.5. Reward −1 per step until goal,
//   episode terminates at goal. Three discrete actions: −1 (reverse),
//   0 (coast), +1 (forward).
//
//   Dynamics (textbook):
//
//     v_{t+1} = clamp(v_t + 0.001 · a_t − 0.0025 · cos(3 · x_t),
//                     −0.07, 0.07)
//     x_{t+1} = clamp(x_t + v_{t+1}, −1.2, 0.5)
//
//   If x_{t+1} hits the LEFT wall, v is zeroed (the wall absorbs).
//
// TILE CODING (Sutton-Albus CMAC)
// ───────────────────────────────
//   Discretise (position, velocity) into a coarse grid; use multiple
//   offset tilings to give a smooth basis. Each tiling contributes one
//   active feature per (state, action) pair → totalFeatures =
//   numTilings × posBins × velBins. The feature vector for one state is
//   binary (1 in the active tile of each tiling, 0 elsewhere) which makes
//   linear VFA particularly well-conditioned.
//
// AS A DES SYSTEM
// ───────────────
//   `MountainCarLinearVFA` extends `LinearVFAStation` and is wrapped by
//   an `EnvironmentStation` in `runMountainCar`. The training loop is
//   driven by `runIterativeDES`.
// =============================================================================

import {
  LinearVFAStation, EnvironmentStation, runIterativeDES,
} from './des-base';
import {Environment} from './rl-environments';
import {mulberry32} from './prng';
import {Preconditions} from './des-base/preconditions';
import {RandomSource, DEFAULT_RANDOM} from '../shared/capabilities';

export interface MountainCarOpts {
  /** Number of tilings stacked over (position, velocity). Default 8. */
  numTilings?: number;
  /** Tiles per tiling (per dimension). Default 8. */
  numTilesPerDim?: number;
  /** Position range. Default [−1.2, 0.5]. */
  posRange?: [number, number];
  /** Velocity range. Default [−0.07, 0.07]. */
  velRange?: [number, number];
  /** Max steps per episode. Default 1000. */
  maxStepsPerEpisode?: number;
  /** Goal position. Default 0.5. */
  goalPos?: number;
}

/** Continuous state encoded as a single integer index for the
 *  EnvironmentStation interface. We store (position, velocity) in a
 *  side-table keyed by integer id. */
class MountainCarEnv implements Environment {
  readonly numStates = 1;  // not used (we re-key states via the side table)
  readonly numActions = 3;
  private readonly opts: Required<MountainCarOpts>;
  /** Continuous state for each integer id we've handed out. */
  private readonly states: Map<number, [number, number]> = new Map();
  private nextId = 0;

  constructor(opts: MountainCarOpts = {}, private readonly rng: RandomSource = DEFAULT_RANDOM) {
    this.opts = {
      numTilings: opts.numTilings ?? 8,
      numTilesPerDim: opts.numTilesPerDim ?? 8,
      posRange: opts.posRange ?? [-1.2, 0.5],
      velRange: opts.velRange ?? [-0.07, 0.07],
      maxStepsPerEpisode: opts.maxStepsPerEpisode ?? 1000,
      goalPos: opts.goalPos ?? 0.5,
    };
  }

  reset(): number {
    const id = this.nextId++;
    // Standard MC starting position: uniform on [-0.6, -0.4] with v=0.
    const x = -0.6 + 0.2 * this.rng.nextFloat();
    this.states.set(id, [x, 0]);
    return id;
  }

  /** Reset using an injected RNG (for reproducibility). */
  resetWithRng(rng: () => number): number {
    const id = this.nextId++;
    const x = -0.6 + 0.2 * rng();
    this.states.set(id, [x, 0]);
    return id;
  }

  step(stateId: number, action: number): {nextState: number; reward: number; done: boolean} {
    const cur = this.states.get(stateId);
    if (!cur) throw new Error(`unknown stateId ${stateId}`);
    const [x, v] = cur;
    const a = action - 1;  // 0,1,2 → -1,0,+1
    let vNew = v + 0.001 * a - 0.0025 * Math.cos(3 * x);
    vNew = Math.max(this.opts.velRange[0], Math.min(this.opts.velRange[1], vNew));
    let xNew = x + vNew;
    if (xNew < this.opts.posRange[0]) { xNew = this.opts.posRange[0]; vNew = 0; }
    if (xNew > this.opts.posRange[1]) xNew = this.opts.posRange[1];
    const done = xNew >= this.opts.goalPos;
    const id = this.nextId++;
    this.states.set(id, [xNew, vNew]);
    return {nextState: id, reward: -1, done};
  }

  getContinuousState(stateId: number): [number, number] {
    const s = this.states.get(stateId);
    if (!s) throw new Error(`unknown stateId ${stateId}`);
    return s;
  }

  getOpts(): Required<MountainCarOpts> { return this.opts; }
}

// -----------------------------------------------------------------------------
// LINEAR VFA AGENT WITH TILE CODING
// -----------------------------------------------------------------------------

class MountainCarLinearVFA extends LinearVFAStation<number> {
  private readonly env: MountainCarEnv;
  private readonly numTilings: number;
  private readonly numTilesPerDim: number;
  private readonly posLow: number;
  private readonly posSpan: number;
  private readonly velLow: number;
  private readonly velSpan: number;
  /** Pre-allocated feature scratch buffer (binary indicators). */
  private readonly featureBuf: Float64Array;

  constructor(env: MountainCarEnv, opts: {
    rng: () => number; alpha: number; gamma: number; epsilon: number;
    epsilonDecay: number; epsilonMin: number;
  }) {
    const o = env.getOpts();
    const featureDim = o.numTilings * o.numTilesPerDim * o.numTilesPerDim;
    super('mc-vfa', {
      rng: opts.rng, featureDim, numActions: 3,
      alpha: opts.alpha / o.numTilings, // canonical: divide by numTilings
      gamma: opts.gamma, epsilon: opts.epsilon,
      epsilonDecay: opts.epsilonDecay, epsilonMin: opts.epsilonMin,
    });
    this.env = env;
    this.numTilings = o.numTilings;
    this.numTilesPerDim = o.numTilesPerDim;
    this.posLow = o.posRange[0];
    this.posSpan = o.posRange[1] - o.posRange[0];
    this.velLow = o.velRange[0];
    this.velSpan = o.velRange[1] - o.velRange[0];
    this.featureBuf = new Float64Array(featureDim);
  }

  /** Sutton-Albus tile coding: each tiling is offset by k/numTilings of a
   *  single tile width. Active tile for tiling k at (x, v) is
   *  ((p_idx + k_off) mod n) * n + ((v_idx + k_off) mod n). */
  protected features(stateId: number): readonly number[] {
    const buf = this.featureBuf;
    buf.fill(0);
    const [x, v] = this.env.getContinuousState(stateId);
    const px = (x - this.posLow) / this.posSpan;
    const vy = (v - this.velLow) / this.velSpan;
    const n = this.numTilesPerDim;
    for (let k = 0; k < this.numTilings; k++) {
      const offset = k / this.numTilings;
      let pIdx = Math.floor((px + offset) * n);
      let vIdx = Math.floor((vy + offset) * n);
      pIdx = Math.max(0, Math.min(n - 1, pIdx));
      vIdx = Math.max(0, Math.min(n - 1, vIdx));
      const tileIdx = pIdx * n + vIdx;
      const featIdx = k * n * n + tileIdx;
      buf[featIdx] = 1;
    }
    // Convert Float64Array → number[] for the readonly contract.
    return Array.from(buf);
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface MountainCarTrainOpts {
  numEpisodes: number;
  maxStepsPerEpisode?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  seed?: number;
  numTilings?: number;
  numTilesPerDim?: number;
}

export interface MountainCarResult {
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  tdErrorHistory: readonly number[];
  /** True iff the GREEDY policy reaches the goal from a quiet start
   *  (x = -0.5, v = 0) within maxStepsPerEpisode. */
  greedySolves: boolean;
  greedyEpisodeLength: number;
  finalEpsilon: number;
  thetaNorm: number;
  ticks: number;
}

export function runMountainCar(opts: MountainCarTrainOpts): MountainCarResult {
  // Pre-run guards.
  const cls = 'runMountainCar';
  Preconditions.integerInRange(cls, 'numEpisodes', opts.numEpisodes, 1, 1e9);
  if (opts.maxStepsPerEpisode !== undefined)
    Preconditions.integerInRange(cls, 'maxStepsPerEpisode', opts.maxStepsPerEpisode, 1, 1e9);
  if (opts.alpha !== undefined) Preconditions.positive(cls, 'alpha', opts.alpha);
  if (opts.gamma !== undefined) Preconditions.inRange(cls, 'gamma', opts.gamma, 0, 1);
  if (opts.epsilon !== undefined) Preconditions.inRange(cls, 'epsilon', opts.epsilon, 0, 1);
  if (opts.epsilonDecay !== undefined) Preconditions.inRange(cls, 'epsilonDecay', opts.epsilonDecay, 0, 1);
  if (opts.epsilonMin !== undefined) Preconditions.inRange(cls, 'epsilonMin', opts.epsilonMin, 0, 1);
  if (opts.numTilings !== undefined) Preconditions.integerInRange(cls, 'numTilings', opts.numTilings, 1, 1e6);
  if (opts.numTilesPerDim !== undefined) Preconditions.integerInRange(cls, 'numTilesPerDim', opts.numTilesPerDim, 1, 1e6);
  const rng = mulberry32(opts.seed ?? 1);
  const env = new MountainCarEnv({
    numTilings: opts.numTilings, numTilesPerDim: opts.numTilesPerDim,
    maxStepsPerEpisode: opts.maxStepsPerEpisode,
  });
  const agent = new MountainCarLinearVFA(env, {
    rng, alpha: opts.alpha ?? 0.5, gamma: opts.gamma ?? 1.0,
    epsilon: opts.epsilon ?? 0.0,  // canonical MC: greedy is enough with optimistic init via tile coding
    epsilonDecay: opts.epsilonDecay ?? 1, epsilonMin: opts.epsilonMin ?? 0,
  });
  const pureEnv = {
    numStates: 1, numActions: 3,
    reset: () => env.resetWithRng(rng),
    step: (s: number, a: number) => env.step(s, a),
  };
  const envStation = new EnvironmentStation<number, number>('env', pureEnv, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: opts.maxStepsPerEpisode ?? 1000,
  });
  envStation.pipe(agent, EnvironmentStation.CH_STATE, MountainCarLinearVFA.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, MountainCarLinearVFA.CH_TRANSITION);
  agent.pipe(envStation, MountainCarLinearVFA.CH_ACTION, EnvironmentStation.CH_ACTION);

  const summary = runIterativeDES([envStation, agent], {rng});

  // Greedy rollout from quiet start to evaluate the learned policy.
  const greedyEnv = new MountainCarEnv({
    numTilings: opts.numTilings, numTilesPerDim: opts.numTilesPerDim,
    maxStepsPerEpisode: opts.maxStepsPerEpisode,
  });
  const startId = (greedyEnv as any).nextId++;
  (greedyEnv as any).states.set(startId, [-0.5, 0]);
  let s = startId; let solves = false; let len = 0;
  const max = opts.maxStepsPerEpisode ?? 1000;
  for (let t = 0; t < max; t++) {
    // Inject this state into our agent's env so features() can read it.
    (env as any).states.set(s, (greedyEnv as any).states.get(s));
    const a = agent.greedyAction(s);
    const stp = greedyEnv.step(s, a);
    len += 1;
    if (stp.done) { solves = true; break; }
    s = stp.nextState;
  }

  let thetaNorm = 0;
  const θ = agent.getTheta();
  for (let i = 0; i < θ.length; i++) thetaNorm += θ[i] * θ[i];
  thetaNorm = Math.sqrt(thetaNorm);

  return {
    rewardHistory: agent.rewardHistory.slice(),
    lengthHistory: agent.lengthHistory.slice(),
    tdErrorHistory: agent.tdErrorHistory.slice(),
    greedySolves: solves, greedyEpisodeLength: len,
    finalEpsilon: agent.getEpsilon(), thetaNorm,
    ticks: summary.ticks,
  };
}
