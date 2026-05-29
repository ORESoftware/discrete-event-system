'use strict';

// =============================================================================
// general/stag-hunt.ts — the canonical STAG HUNT coordination game with
// INDEPENDENT Q-LEARNING (Tan 1993) on top of the multi-agent base.
//
// MATRIX (per Rousseau 1755 / Skyrms 2004)
// ────────────────────────────────────────
//                          Player 2
//                       Stag      Hare
//   Player 1   Stag    (4, 4)    (0, 3)
//              Hare    (3, 0)    (3, 3)
//
//   Two pure-strategy Nash equilibria:
//     - (Stag, Stag) — payoff-dominant, requires coordination
//     - (Hare, Hare) — risk-dominant, safe
//
//   Independent Q-learners can converge to EITHER equilibrium depending
//   on initialisation, ε, and α. The agents that "find" the (Stag, Stag)
//   equilibrium reach the social optimum (8 total reward); failing
//   coordination drops to the risk-dominant (Hare, Hare) at 6 total.
//
// AS A MULTI-AGENT DES SYSTEM
// ───────────────────────────
//   Single-state stateless game (we use state=0 always). Each agent is
//   a Q-learner over 1 state × 2 actions. The JointEnvStation samples
//   the matrix payoff; the agents update independently. After ~5k
//   episodes the agents almost always coordinate on the same action.
// =============================================================================

import {
  JointEnvironment, JointEnvStation, MultiAgentSystem, RLAgentStation, argMaxWithTieBreak,
} from './des-base';
import {ActionToken, TransitionToken, StateToken} from './des-base/rl-tokens';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// MATRIX-GAME ENVIRONMENT
// -----------------------------------------------------------------------------

const STAG = 0;
const HARE = 1;
const PAYOFF = [
  // [a1=Stag,a1=Hare] for each a2
  // payoff[a1][a2] = (r1, r2)
  [[4, 4], [0, 3]],   // a1 = Stag
  [[3, 0], [3, 3]],   // a1 = Hare
];

class StagHuntEnv implements JointEnvironment<number, number> {
  readonly numAgents = 2;
  reset(): number[] { return [0, 0]; }
  step(_states: readonly number[], actions: readonly number[]):
    {nextStates: number[]; rewards: number[]; dones: boolean[]} {
    const a1 = actions[0]; const a2 = actions[1];
    const [r1, r2] = PAYOFF[a1][a2];
    return {nextStates: [0, 0], rewards: [r1, r2], dones: [true, true]};
  }
}

// -----------------------------------------------------------------------------
// SIMPLE TABULAR Q-LEARNING AGENT (1 state × 2 actions)
// -----------------------------------------------------------------------------

class StagHuntQLearner extends RLAgentStation<number, number> {
  protected readonly Q: Float64Array;
  protected readonly A: number;
  protected readonly alpha: number;
  protected readonly gamma: number;
  protected epsilon: number;
  protected readonly epsilonDecay: number;
  protected readonly epsilonMin: number;

  constructor(id: string, opts: {rng: () => number; alpha: number; gamma: number;
                                  epsilon: number; epsilonDecay: number; epsilonMin: number}) {
    super(id, {rng: opts.rng});
    this.A = 2;
    this.Q = new Float64Array(2);     // single state
    this.alpha = opts.alpha;
    this.gamma = opts.gamma;
    this.epsilon = opts.epsilon;
    this.epsilonDecay = opts.epsilonDecay;
    this.epsilonMin = opts.epsilonMin;
  }
  protected pickAction(_state: number, rng: () => number): number {
    if (rng() < this.epsilon) return Math.floor(rng() * this.A);
    return argMaxWithTieBreak(this.Q, rng);
  }
  protected update(_s: number, action: number, reward: number, _ns: number, _done: boolean): void {
    // No bootstrap: stateless single-step game.
    this.Q[action] += this.alpha * (reward - this.Q[action]);
  }
  protected override endOfEpisode(_id: number): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }
  getQ(): Float64Array { return this.Q; }
  greedyAction(): number { return argMaxWithTieBreak(this.Q, this.rng); }
  getEpsilon(): number { return this.epsilon; }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface StagHuntOpts {
  numEpisodes: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  seed?: number;
}

export interface StagHuntResult {
  /** Per-episode joint reward [r1, r2]. */
  rewardHistory: number[][];
  /** Final greedy joint action [a1, a2]. */
  finalJointAction: [number, number];
  /** Per-agent total return over the last 100 episodes. */
  recentMeanReturn: [number, number];
  /** True iff both agents converged to STAG (the payoff-dominant
   *  equilibrium). */
  coordinatedOnStag: boolean;
  /** True iff both agents converged to HARE (the risk-dominant
   *  equilibrium). */
  coordinatedOnHare: boolean;
  ticks: number;
}

export function runStagHunt(opts: StagHuntOpts): StagHuntResult {
  const cls = 'runStagHunt';
  Preconditions.integerInRange(cls, 'numEpisodes', opts.numEpisodes, 1, 1e9);
  if (opts.alpha !== undefined) Preconditions.positive(cls, 'alpha', opts.alpha);
  if (opts.gamma !== undefined) Preconditions.inRange(cls, 'gamma', opts.gamma, 0, 1);
  if (opts.epsilon !== undefined) Preconditions.inRange(cls, 'epsilon', opts.epsilon, 0, 1);
  if (opts.epsilonDecay !== undefined) Preconditions.inRange(cls, 'epsilonDecay', opts.epsilonDecay, 0, 1);
  if (opts.epsilonMin !== undefined) Preconditions.inRange(cls, 'epsilonMin', opts.epsilonMin, 0, 1);
  const rng = mulberry32(opts.seed ?? 1);
  const env = new StagHuntEnv();
  const envStation = new JointEnvStation('stag-hunt-env', env, {
    numEpisodes: opts.numEpisodes, maxStepsPerEpisode: 1,
  });
  const a1 = new StagHuntQLearner('agent-0', {
    rng,
    alpha: opts.alpha ?? 0.05, gamma: opts.gamma ?? 0,
    epsilon: opts.epsilon ?? 0.2,
    epsilonDecay: opts.epsilonDecay ?? 0.999, epsilonMin: opts.epsilonMin ?? 0.01,
  });
  const a2 = new StagHuntQLearner('agent-1', {
    rng,
    alpha: opts.alpha ?? 0.05, gamma: opts.gamma ?? 0,
    epsilon: opts.epsilon ?? 0.2,
    epsilonDecay: opts.epsilonDecay ?? 0.999, epsilonMin: opts.epsilonMin ?? 0.01,
  });
  const sys = new MultiAgentSystem(envStation, [a1, a2]);
  const summary = sys.run({rng});

  const greedy = [a1.greedyAction(), a2.greedyAction()] as [number, number];
  // Recent returns.
  const last = summary.rewardHistory.slice(-Math.min(100, summary.rewardHistory.length));
  const r1 = last.reduce((s, e) => s + e[0], 0) / Math.max(1, last.length);
  const r2 = last.reduce((s, e) => s + e[1], 0) / Math.max(1, last.length);
  return {
    rewardHistory: summary.rewardHistory,
    finalJointAction: greedy,
    recentMeanReturn: [r1, r2],
    coordinatedOnStag: greedy[0] === STAG && greedy[1] === STAG,
    coordinatedOnHare: greedy[0] === HARE && greedy[1] === HARE,
    ticks: summary.ticks,
  };
}
