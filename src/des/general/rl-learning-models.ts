// RUST MIGRATION: Target module `src/des/general/rl_learning_models.rs`.
// RUST MIGRATION: Convert policy-gradient and Expected-SARSA params/results to `serde` structs; keep `RLTopology` as the Rust graph-summary alias.
// RUST MIGRATION: Port agent/update classes as structs implementing `PolicyGradientAgent`, `PolicyUpdateStation`, and `RLAgentStation` traits.
// RUST MIGRATION: Replace inheritance overrides with trait impls and embed shared base state explicitly.
// RUST MIGRATION: Inject RNG for softmax/action choice and return `Result` for invalid alpha/gamma/episode options or malformed environment dimensions.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/rl-learning-models.rs  (module des::general::rl_learning_models)
// 1:1 file move. REINFORCE policy-gradient corridor + expected-SARSA gridworld DES models.
//
// Declarations → Rust:
//   type RLTopology = StationGraphSummary  -> type alias
//   interface PolicyGradientCorridorParams/Result, ExpectedSarsaGridParams/Result -> structs
//   class SoftmaxPolicyGradientAgent extends PolicyGradientAgent<number,number> -> struct + impl
//   class ReinforceUpdateStation extends PolicyUpdateStation -> struct + impl
//   class ExpectedSarsaAgent extends RLAgentStation<number,number> -> struct + impl
//   fn runPolicyGradientCorridor / runExpectedSarsaGridworld -> free fns / assoc fns
//
// Conversion notes (file-specific):
//   - INJECT RNG: `mulberry32` + sampling/ε-greedy -> take a `RandomSource` (SeededRandom).
//   - PolicyGradientAgent/PolicyUpdateStation/RLAgentStation are template-method bases ->
//     traits with default fns; tabular θ/Q are `Vec<Vec<f64>>` struct fields (`&mut self`).
//   - depends on rl-environments.ts (Corridor/GridWorld/evalPolicy) -> use crate::...::rl_environments.
//   - `softmax` helper from des-base -> shared fn over `Vec<f64>`.
// =============================================================================
// general/rl-learning-models.ts
//
// Additional RL station-graph models built from shared DES base classes:
//   - policy-gradient-corridor  (PolicyGradientAgent + PolicyUpdateStation)
//   - expected-sarsa-gridworld  (RLAgentStation + EnvironmentStation)
// =============================================================================

import {
  ActionToken,
  EnvironmentStation,
  PolicyGradientAgent,
  PolicyUpdateStation,
  RLAgentStation,
  ResumeToken,
  RolloutEntry,
  StationGraphSummary,
  StateToken,
  TransitionToken,
  argMaxWithTieBreak,
  channelEdge,
  runIterativeDES,
  softmax,
  stationGraph,
} from './des-base';
import {Corridor, GridWorld, evalPolicy} from './rl-environments';
import {mulberry32} from './prng';

export type RLTopology = StationGraphSummary;

export interface PolicyGradientCorridorParams {
  numEpisodes?: number;
  maxStepsPerEpisode?: number;
  rolloutLen?: number;
  alpha?: number;
  gamma?: number;
  seed?: number;
  length?: number;
}

export interface PolicyGradientCorridorResult {
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  greedySuccessRate: number;
  greedyMeanLength: number;
  policy: number[];
  updates: number;
  topology: RLTopology;
}

class SoftmaxPolicyGradientAgent extends PolicyGradientAgent<number, number> {
  readonly theta: number[][];

  constructor(id: string, readonly numStates: number, readonly numActions: number, opts: {rolloutLen: number; rng: () => number}) {
    super(id, opts);
    this.theta = Array.from({length: numStates}, () => Array.from({length: numActions}, () => 0));
  }

  protected samplePolicyAndValue(state: number, rng: () => number): {action: number; logProb: number; value: number} {
    const probs = softmax(this.theta[state]);
    let u = rng();
    for (let a = 0; a < probs.length; a++) {
      u -= probs[a];
      if (u <= 0) return {action: a, logProb: Math.log(Math.max(probs[a], 1e-12)), value: 0};
    }
    const action = probs.length - 1;
    return {action, logProb: Math.log(Math.max(probs[action], 1e-12)), value: 0};
  }

  applyPolicyGradient(state: number, action: number, advantage: number, alpha: number): void {
    const probs = softmax(this.theta[state]);
    for (let a = 0; a < this.numActions; a++) {
      this.theta[state][a] += alpha * advantage * ((a === action ? 1 : 0) - probs[a]);
    }
  }

  greedyAction(state: number): number {
    return argMaxWithTieBreak(this.theta[state], this.rng);
  }
}

class ReinforceUpdateStation extends PolicyUpdateStation {
  readonly updateReturns: number[] = [];

  constructor(
    id: string,
    private readonly agent: SoftmaxPolicyGradientAgent,
    private readonly alpha: number,
    private readonly gamma: number,
  ) {
    super(id);
  }

  protected runUpdate(): void {
    const buffer = this.agent.getBuffer();
    let g = 0;
    const returns = new Array(buffer.length).fill(0);
    for (let i = buffer.length - 1; i >= 0; i--) {
      const r = buffer[i].r ?? 0;
      g = r + this.gamma * g;
      returns[i] = g;
      if (buffer[i].done) g = 0;
    }
    const baseline = returns.reduce((s, x) => s + x, 0) / Math.max(1, returns.length);
    for (let i = 0; i < buffer.length; i++) {
      const e = buffer[i] as RolloutEntry<number, number>;
      if (e.r === undefined) continue;
      this.agent.applyPolicyGradient(e.s, e.a, returns[i] - baseline, this.alpha);
    }
    this.updateReturns.push(...returns);
    this.agent.clearBuffer();
  }
}

export function runPolicyGradientCorridor(params: PolicyGradientCorridorParams): PolicyGradientCorridorResult {
  const env = new Corridor(params.length ?? 7);
  const rng = mulberry32(params.seed ?? 1);
  const envStation = new EnvironmentStation<number, number>('corridor-env', env, {
    numEpisodes: params.numEpisodes ?? 300,
    maxStepsPerEpisode: params.maxStepsPerEpisode ?? 40,
  });
  const agent = new SoftmaxPolicyGradientAgent('softmax-policy-agent', env.numStates, env.numActions, {
    rolloutLen: params.rolloutLen ?? 12,
    rng,
  });
  const updater = new ReinforceUpdateStation('reinforce-update', agent, params.alpha ?? 0.04, params.gamma ?? 0.95);
  envStation.pipe(agent, EnvironmentStation.CH_STATE, PolicyGradientAgent.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, PolicyGradientAgent.CH_TRANSITION);
  agent.pipe(envStation, PolicyGradientAgent.CH_ACTION, EnvironmentStation.CH_ACTION);
  agent.pipe(updater, PolicyGradientAgent.CH_TRAIN, PolicyUpdateStation.CH_TRAIN);
  updater.pipe(agent, PolicyUpdateStation.CH_RESUME, PolicyGradientAgent.CH_RESUME);
  runIterativeDES([envStation, agent, updater], {rng, maxTicks: (params.numEpisodes ?? 300) * (params.maxStepsPerEpisode ?? 40) * 4});
  const evaluation = evalPolicy(env, s => agent.greedyAction(s), {
    numEpisodes: 50,
    maxStepsPerEpisode: params.maxStepsPerEpisode ?? 40,
    rng: mulberry32(99),
  });
  return {
    rewardHistory: envStation.rewardHistory.slice(),
    lengthHistory: envStation.lengthHistory.slice(),
    greedySuccessRate: evaluation.successRate,
    greedyMeanLength: evaluation.meanLength,
    policy: Array.from({length: env.numStates}, (_, s) => agent.greedyAction(s)),
    updates: updater.numUpdates,
    topology: stationGraph([envStation, agent, updater], ['StateToken', 'ActionToken', 'TransitionToken', 'TrainTriggerToken', 'ResumeToken'], [
      channelEdge(envStation, EnvironmentStation.CH_STATE, agent, PolicyGradientAgent.CH_STATE),
      channelEdge(agent, PolicyGradientAgent.CH_ACTION, envStation, EnvironmentStation.CH_ACTION),
      channelEdge(envStation, EnvironmentStation.CH_TRANSITION, agent, PolicyGradientAgent.CH_TRANSITION),
      channelEdge(agent, PolicyGradientAgent.CH_TRAIN, updater, PolicyUpdateStation.CH_TRAIN),
      channelEdge(updater, PolicyUpdateStation.CH_RESUME, agent, PolicyGradientAgent.CH_RESUME),
    ]),
  };
}

export interface ExpectedSarsaGridParams {
  numEpisodes?: number;
  maxStepsPerEpisode?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  seed?: number;
}

export interface ExpectedSarsaGridResult {
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  greedyReached: boolean;
  greedyLen: number;
  qStart: number[];
  policy: number[];
  topology: RLTopology;
}

class ExpectedSarsaAgent extends RLAgentStation<number, number> {
  private readonly q: number[][];
  private epsilon: number;

  constructor(
    id: string,
    private readonly numStates: number,
    private readonly numActions: number,
    opts: {rng: () => number; alpha: number; gamma: number; epsilon: number; epsilonDecay: number; epsilonMin: number},
  ) {
    super(id, {rng: opts.rng});
    this.q = Array.from({length: numStates}, () => Array.from({length: numActions}, () => 0));
    this.alpha = opts.alpha;
    this.gamma = opts.gamma;
    this.epsilon = opts.epsilon;
    this.epsilonDecay = opts.epsilonDecay;
    this.epsilonMin = opts.epsilonMin;
  }

  private readonly alpha: number;
  private readonly gamma: number;
  private readonly epsilonDecay: number;
  private readonly epsilonMin: number;

  protected pickAction(state: number, rng: () => number): number {
    if (rng() < this.epsilon) return Math.floor(rng() * this.numActions);
    return this.greedyAction(state);
  }

  protected update(state: number, action: number, reward: number, nextState: number, done: boolean): void {
    const expected = done ? 0 : this.expectedValue(nextState);
    const target = reward + this.gamma * expected;
    this.q[state][action] += this.alpha * (target - this.q[state][action]);
  }

  protected endOfEpisode(_episodeId: number): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  expectedValue(state: number): number {
    const greedy = this.greedyAction(state);
    let v = 0;
    for (let a = 0; a < this.numActions; a++) {
      const p = this.epsilon / this.numActions + (a === greedy ? 1 - this.epsilon : 0);
      v += p * this.q[state][a];
    }
    return v;
  }

  greedyAction(state: number): number {
    return argMaxWithTieBreak(this.q[state], this.rng);
  }

  qValues(state: number): number[] { return this.q[state].slice(); }
}

export function runExpectedSarsaGridworld(params: ExpectedSarsaGridParams): ExpectedSarsaGridResult {
  const rng = mulberry32(params.seed ?? 1);
  const env = new GridWorld({width: 4, height: 4});
  const envStation = new EnvironmentStation<number, number>('grid-env', env, {
    numEpisodes: params.numEpisodes ?? 900,
    maxStepsPerEpisode: params.maxStepsPerEpisode ?? 80,
  });
  const agent = new ExpectedSarsaAgent('expected-sarsa-agent', env.numStates, env.numActions, {
    rng,
    alpha: params.alpha ?? 0.2,
    gamma: params.gamma ?? 0.95,
    epsilon: params.epsilon ?? 0.35,
    epsilonDecay: params.epsilonDecay ?? 0.995,
    epsilonMin: params.epsilonMin ?? 0.02,
  });
  envStation.pipe(agent, EnvironmentStation.CH_STATE, RLAgentStation.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, RLAgentStation.CH_TRANSITION);
  agent.pipe(envStation, RLAgentStation.CH_ACTION, EnvironmentStation.CH_ACTION);
  runIterativeDES([envStation, agent], {rng, maxTicks: (params.numEpisodes ?? 900) * (params.maxStepsPerEpisode ?? 80) * 3});

  let s = env.reset();
  let reached = false;
  let len = 0;
  for (let t = 0; t < (params.maxStepsPerEpisode ?? 80); t++) {
    const step = env.step(s, agent.greedyAction(s));
    len += 1;
    if (step.done) { reached = step.reward > 0; break; }
    s = step.nextState;
  }

  return {
    rewardHistory: agent.rewardHistory.slice(),
    lengthHistory: agent.lengthHistory.slice(),
    greedyReached: reached,
    greedyLen: len,
    qStart: agent.qValues(env.start),
    policy: Array.from({length: env.numStates}, (_, state) => agent.greedyAction(state)),
    topology: stationGraph([envStation, agent], ['StateToken', 'ActionToken', 'TransitionToken'], [
      channelEdge(envStation, EnvironmentStation.CH_STATE, agent, RLAgentStation.CH_STATE),
      channelEdge(agent, RLAgentStation.CH_ACTION, envStation, EnvironmentStation.CH_ACTION),
      channelEdge(envStation, EnvironmentStation.CH_TRANSITION, agent, RLAgentStation.CH_TRANSITION),
    ]),
  };
}
