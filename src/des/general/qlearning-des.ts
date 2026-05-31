// RUST MIGRATION: Target module `src/des/general/qlearning_des.rs`.
// RUST MIGRATION: Convert Q-learning options and result interfaces to `serde` structs with explicit numeric state/action newtypes or `usize`.
// RUST MIGRATION: Port `QLearningAgent` as a struct implementing the `RLAgentStation` trait; inherited hooks become trait methods.
// RUST MIGRATION: Keep `runQLearningDES` as a graph runner free function, or wrap it in `PureTransform` only if it is surfaced as a graph-visible transform.
// RUST MIGRATION: Replace epsilon-greedy randomness with an injected RNG trait/closure and return `Result` for invalid learning rates, discounts, or episode counts.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/qlearning-des.rs  (module des::general::qlearning_des)
// 1:1 file move. Tabular off-policy Q-learning agent driven as a DES.
//
// Declarations → Rust:
//   interface QLearningOptions / QLearningResult     -> structs (Default where sensible)
//   class QLearningAgent extends RLAgentStation<number, number> -> struct + impl agent trait
//   fn runQLearningDES                               -> free fn / assoc fn
//
// Conversion notes (file-specific):
//   - INJECT RNG: `mulberry32` + ε-greedy tie-breaking -> take a `RandomSource`
//     (SeededRandom == mulberry32); do not import the closure or hit a global.
//   - RLAgentStation is a template-method base -> trait with default fns; the Q-table is a
//     `Vec<Vec<f64>>` struct field updated via `&mut self`.
//   - state/action are numeric indices (`usize`); rewards/values `f64`.
// =============================================================================
// general/qlearning-des.ts — Q-learning as a DES.
//
// Concrete leaf class extending RLAgentStation: implements ε-greedy
// `pickAction` and the textbook off-policy TD(0) `update` rule:
//
//      Q[s,a] ← Q[s,a] + α · ( r + γ · max_a' Q[s',a'] − Q[s,a] )
//
// Topology:
//
//      ┌──────────────────────┐      action      ┌───────────────────┐
//      │   QLearningAgent     │ ───────────────▶ │ EnvironmentStation │
//      │   (RLAgentStation)   │                  └───────────────────┘
//      │                      │ ◀─────────── state / transition
//      └──────────────────────┘
// =============================================================================

import {
  RLAgentStation, EnvironmentStation, PureEnvironment,
  runIterativeDES, IterativeRunOptions,
  argMaxWithTieBreak,
} from './des-base';
import {mulberry32} from './prng';

export interface QLearningOptions {
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonMin?: number;
  /** Per-EPISODE multiplicative decay of ε. */
  epsilonDecay?: number;
  numStates: number;
  numActions: number;
  /** Optional Q-init function (default: zeros). */
  qInit?: (s: number, a: number, rng: () => number) => number;
}

export class QLearningAgent extends RLAgentStation<number, number> {
  readonly Q: number[][];
  readonly numStates: number;
  readonly numActions: number;
  readonly opts: QLearningOptions;
  protected currentEpsilon: number;

  constructor(id: string, opts: QLearningOptions & {rng: () => number}) {
    super(id, {rng: opts.rng});
    this.opts = opts;
    this.numStates = opts.numStates;
    this.numActions = opts.numActions;
    this.currentEpsilon = opts.epsilon;
    const init = opts.qInit ?? (() => 0);
    this.Q = Array.from({length: opts.numStates}, (_, s) =>
      Array.from({length: opts.numActions}, (_, a) => init(s, a, opts.rng)));
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected pickAction(state: number, rng: () => number): number {
    if (rng() < this.currentEpsilon) {
      return Math.floor(rng() * this.numActions);
    }
    // Random tie-breaking on argmax: with zero-init Q-tables all actions
    // tie initially; deterministic `>` would always pick action 0 and
    // never break out of that bias under ε=0 greedy evaluation.
    return argMaxWithTieBreak(this.Q[state], rng);
  }

  protected update(s: number, a: number, r: number, sNext: number, done: boolean): void {
    const Qsa = this.Q[s][a];
    const target = r + (done ? 0 : this.opts.gamma * Math.max(...this.Q[sNext]));
    this.Q[s][a] = Qsa + this.opts.alpha * (target - Qsa);
  }

  protected override endOfEpisode(_id: number): void {
    if (this.opts.epsilonDecay !== undefined) {
      this.currentEpsilon = Math.max(
        this.opts.epsilonMin ?? 0,
        this.currentEpsilon * this.opts.epsilonDecay,
      );
    }
  }

  // ── HELPERS ──────────────────────────────────────────────────────────────

  greedyPolicy(): number[] {
    return this.Q.map(row => argMaxWithTieBreak(row, this.rng));
  }

  getEpsilon(): number { return this.currentEpsilon; }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

export interface QLearningResult {
  Q: number[][];
  policy: number[];
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  totalEpisodes: number;
  totalSteps: number;
  totalTicks: number;
}

export function runQLearningDES(env: PureEnvironment<number, number>, opts: {
  numEpisodes: number;
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonMin?: number;
  epsilonDecay?: number;
  maxStepsPerEpisode?: number;
  seed?: number;
  desOptions?: IterativeRunOptions;
}): QLearningResult {
  const rng = mulberry32(opts.seed ?? 1);
  const agent = new QLearningAgent('q-agent', {
    alpha: opts.alpha, gamma: opts.gamma, epsilon: opts.epsilon,
    epsilonMin: opts.epsilonMin, epsilonDecay: opts.epsilonDecay,
    numStates: env.numStates, numActions: env.numActions,
    rng,
  });
  const envSt = new EnvironmentStation<number, number>('env', env, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: opts.maxStepsPerEpisode,
  });
  // Wire channels: env → agent (state, transition), agent → env (action).
  envSt.pipe(agent, EnvironmentStation.CH_STATE, RLAgentStation.CH_STATE);
  envSt.pipe(agent, EnvironmentStation.CH_TRANSITION, RLAgentStation.CH_TRANSITION);
  agent.pipe(envSt, RLAgentStation.CH_ACTION, EnvironmentStation.CH_ACTION);

  const summary = runIterativeDES([envSt, agent], {rng, ...opts.desOptions});

  return {
    Q: agent.Q,
    policy: agent.greedyPolicy(),
    rewardHistory: agent.rewardHistory,
    lengthHistory: agent.lengthHistory,
    totalEpisodes: agent.rewardHistory.length,
    totalSteps: agent.totalSteps,
    totalTicks: summary.ticks,
  };
}
