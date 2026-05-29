'use strict';

// =============================================================================
// general/actor-critic-gridworld.ts — ONE-STEP TABULAR ACTOR-CRITIC
// (Sutton & Barto §13.5) on a small GridWorld.
//
// The TabularActorCritic base class lives in `des-base/actor-critic.ts`.
// This module is the runnable driver: wire the agent to an
// EnvironmentStation, run the training loop, and report the learned
// V(s) and π(·|s) at the start state.
//
// CLASSIC USE CASE
// ────────────────
//   4×4 GridWorld with two pits and a goal in the bottom-right corner.
//   Reward shaping: -1 per step, +10 at goal, -10 at pit. Discount
//   γ = 0.95. Actor-Critic learns a softmax policy over the 4
//   directional actions (N, E, S, W) at each cell. After 1k episodes
//   the greedy policy reaches the goal in ≤ 8 steps from the start
//   (optimal is 6).
// =============================================================================

import {
  TabularActorCritic, EnvironmentStation, runIterativeDES,
} from './des-base';
import {GridWorld} from './rl-environments';
import {mulberry32} from './prng';
import {Preconditions} from './des-base/preconditions';

export interface ActorCriticTrainOpts {
  numEpisodes: number;
  maxStepsPerEpisode?: number;
  alphaV?: number;
  alphaP?: number;
  gamma?: number;
  entropyCoef?: number;
  seed?: number;
  /** GridWorld dimensions. Default 4×4. */
  width?: number;
  height?: number;
}

export interface ActorCriticResult {
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  tdErrorHistory: readonly number[];
  /** V_θ at the start state — proxy for "how good is the start". */
  Vstart: number;
  /** Whether the GREEDY policy reaches the goal from the start. */
  greedyReached: boolean;
  greedyLen: number;
  ticks: number;
}

export function runActorCriticGridworld(opts: ActorCriticTrainOpts): ActorCriticResult {
  const cls = 'runActorCriticGridworld';
  Preconditions.integerInRange(cls, 'numEpisodes', opts.numEpisodes, 1, 1e9);
  if (opts.maxStepsPerEpisode !== undefined)
    Preconditions.integerInRange(cls, 'maxStepsPerEpisode', opts.maxStepsPerEpisode, 1, 1e9);
  if (opts.alphaV !== undefined) Preconditions.positive(cls, 'alphaV', opts.alphaV);
  if (opts.alphaP !== undefined) Preconditions.positive(cls, 'alphaP', opts.alphaP);
  if (opts.gamma !== undefined) Preconditions.inRange(cls, 'gamma', opts.gamma, 0, 1);
  if (opts.entropyCoef !== undefined) Preconditions.nonNegative(cls, 'entropyCoef', opts.entropyCoef);
  if (opts.width !== undefined) Preconditions.integerInRange(cls, 'width', opts.width, 1, 10000);
  if (opts.height !== undefined) Preconditions.integerInRange(cls, 'height', opts.height, 1, 10000);
  const rng = mulberry32(opts.seed ?? 1);
  const env = new GridWorld({width: opts.width ?? 4, height: opts.height ?? 4});
  const agent = new TabularActorCritic('ac-grid', {
    rng,
    numStates: env.numStates,
    numActions: env.numActions,
    alphaV: opts.alphaV ?? 0.1,
    alphaP: opts.alphaP ?? 0.05,
    gamma: opts.gamma ?? 0.95,
    entropyCoef: opts.entropyCoef ?? 0,
  });
  const envStation = new EnvironmentStation<number, number>('env', env, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: opts.maxStepsPerEpisode ?? 100,
  });
  envStation.pipe(agent, EnvironmentStation.CH_STATE, TabularActorCritic.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, TabularActorCritic.CH_TRANSITION);
  agent.pipe(envStation, TabularActorCritic.CH_ACTION, EnvironmentStation.CH_ACTION);
  const summary = runIterativeDES([envStation, agent], {rng});

  // Greedy rollout.
  const evalEnv = new GridWorld({width: opts.width ?? 4, height: opts.height ?? 4});
  let s = evalEnv.reset();
  let len = 0; let reached = false;
  const max = opts.maxStepsPerEpisode ?? 100;
  for (let t = 0; t < max; t++) {
    const a = agent.greedyAction(s);
    const r = evalEnv.step(s, a);
    len += 1;
    if (r.done) { reached = r.reward > 0; break; }
    s = r.nextState;
  }
  return {
    rewardHistory: agent.rewardHistory.slice(),
    lengthHistory: agent.lengthHistory.slice(),
    tdErrorHistory: agent.tdErrorHistory.slice(),
    Vstart: agent.getV()[env.start],
    greedyReached: reached,
    greedyLen: len,
    ticks: summary.ticks,
  };
}
