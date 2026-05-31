'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/multi_agent.rs
// - Keep file-for-file. JointEnvironment becomes a behavior trait and
//   MultiAgentSystemOpts becomes a config struct.
// - JointEnvStation and MultiAgentSystem become state-owning structs; per-agent
//   action buffers map from Map to HashMap/BTreeMap keyed by agent id.
// - Joint step/policy helpers can stay associated methods; if a coordination
//   rule is lifted into a graph node, use PureTransform/PureTransformEntity.
// - Convert agent-count and missing-action errors to Result.

// =============================================================================
// general/des-base/multi-agent.ts — base class for SIMULTANEOUS-MOVE
// MULTI-AGENT REINFORCEMENT LEARNING ON A SHARED ENVIRONMENT.
//
// PROBLEM SHAPE
// ─────────────
//   N agents share an environment. At each tick:
//     1. Every agent observes its own state s_i.
//     2. Every agent picks an action a_i in parallel (no peeking).
//     3. The environment applies the joint action a = (a_1, …, a_N),
//        emits joint next state s' = (s'_1, …, s'_N) and a per-agent
//        reward vector r = (r_1, …, r_N).
//     4. Each agent updates its own Q-table (or whatever) on its own
//        transition. NO cross-agent updates here — that's Independent
//        Q-Learning (IQL).
//
//   IQL is famously simple but suffers in non-stationary regimes (each
//   agent is non-stationary from the others' POV). For coordination
//   games it nonetheless converges to ONE Nash equilibrium reasonably
//   well, especially with small ε.
//
// THIS BASE PROVIDES
// ──────────────────
//   • `JointEnvironment<S, A>` interface — N-tuple of states/actions per
//     step.
//   • `MultiAgentSystem<S, A>` — orchestrator that owns N RLAgentStations
//     plus a JointEnvironment. Wraps everything as DESStations and
//     drives the train loop via runIterativeDES.
//   • Per-agent reward histories AND group-level histories for diagnostics.
//
// USAGE
// ─────
//   This base is INTENTIONALLY MINIMAL — it lets the user pass arbitrary
//   pre-built RLAgentStations and a JointEnvironment, and runs them
//   together. Concrete games (Stag Hunt, Iterated Prisoner's Dilemma,
//   Pursuit-Evasion) plug into this pattern by supplying the environment.
// =============================================================================

import {DESStation, ChannelName} from './station';
import {RLAgentStation} from './rl-agent';
import {StateToken, ActionToken, TransitionToken} from './rl-tokens';
import {runIterativeDES} from './runner';
import {VectorEpisodeAccounting} from './episode-accounting';

export interface JointEnvironment<S = number, A = number> {
  readonly numAgents: number;
  /** Reset returns the joint state s = [s_1, …, s_N]. */
  reset(): S[];
  /** step receives the joint action a = [a_1, …, a_N] and returns
   *  joint next state, per-agent reward, per-agent done flag (all of
   *  length numAgents). The episode ENDS when all agents are done OR
   *  some env-wide termination condition fires (in which case all
   *  done flags should be true). */
  step(states: readonly S[], actions: readonly A[]): {
    nextStates: S[];
    rewards: number[];
    dones: boolean[];
  };
}

/** A single tick of the joint env: deal joint state to every agent,
 *  collect every agent's action, apply env.step, dispatch transitions
 *  + new states. */
export class JointEnvStation<S = number, A = number> extends DESStation {
  static readonly CH_AGENT_ACTION_PREFIX = 'agent-action-';
  static readonly CH_AGENT_STATE_PREFIX = 'agent-state-';
  static readonly CH_AGENT_TRANSITION_PREFIX = 'agent-transition-';

  protected readonly env: JointEnvironment<S, A>;
  protected readonly numEpisodes: number;
  protected readonly maxStepsPerEpisode: number;

  protected curStates: S[];
  protected episodeId = 0;
  protected stepInEpisode = 0;
  protected emittedStart = false;
  protected pendingActions: Map<number, A> = new Map();

  /** Per-episode return per agent, recorded at episode end. */
  protected readonly episodeAccounting: VectorEpisodeAccounting;
  readonly rewardHistory: number[][];
  readonly lengthHistory: number[];
  get totalSteps(): number { return this.episodeAccounting.totalSteps; }
  set totalSteps(value: number) { this.episodeAccounting.totalSteps = value; }
  protected get curReturn(): number[] { return this.episodeAccounting.currentRewards; }
  protected set curReturn(value: number[]) { this.episodeAccounting.setCurrentRewards(value); }

  constructor(id: string, env: JointEnvironment<S, A>, opts: {
    numEpisodes?: number; maxStepsPerEpisode?: number;
  } = {}) {
    super(id);
    this.env = env;
    this.numEpisodes = opts.numEpisodes ?? Infinity;
    this.maxStepsPerEpisode = opts.maxStepsPerEpisode ?? 1_000_000;
    this.curStates = env.reset();
    this.episodeAccounting = new VectorEpisodeAccounting(env.numAgents);
    this.rewardHistory = this.episodeAccounting.rewardHistory;
    this.lengthHistory = this.episodeAccounting.lengthHistory;
  }

  override hasWork(): boolean {
    if (!this.emittedStart) return true;
    if (this.episodeId >= this.numEpisodes) return false;
    // We have work if our action channels have tokens we haven't drained yet,
    // OR if we already drained enough into pendingActions on a previous tick.
    if (this.pendingActions.size === this.env.numAgents) return true;
    for (let i = 0; i < this.env.numAgents; i++) {
      if (this.inboxSize(JointEnvStation.CH_AGENT_ACTION_PREFIX + i) > 0) return true;
    }
    return false;
  }

  /** Each agent calls this with its action for the current joint state. */
  takeAgentAction(agentIdx: number, action: A): void {
    this.pendingActions.set(agentIdx, action);
  }

  runTimeStep(): void {
    if (!this.emittedStart) {
      this.emittedStart = true;
      this.emitStateForAllAgents();
      return;
    }
    if (this.episodeId >= this.numEpisodes) return;
    if (this.pendingActions.size < this.env.numAgents) {
      // Drain any pending action tokens.
      for (let i = 0; i < this.env.numAgents; i++) {
        const ch = JointEnvStation.CH_AGENT_ACTION_PREFIX + i;
        for (const tok of this.drain<ActionToken<S, A>>(ch)) {
          this.pendingActions.set(i, tok.action);
        }
      }
      if (this.pendingActions.size < this.env.numAgents) return;
    }
    const actions: A[] = [];
    for (let i = 0; i < this.env.numAgents; i++) actions.push(this.pendingActions.get(i)!);
    this.pendingActions.clear();
    const r = this.env.step(this.curStates, actions);
    this.episodeAccounting.recordStep(r.rewards);
    this.stepInEpisode += 1;
    const truncated = this.stepInEpisode >= this.maxStepsPerEpisode;
    const allDone = r.dones.every(d => d) || truncated;
    for (let i = 0; i < this.env.numAgents; i++) {
      this.emit(
        new TransitionToken<S, A>(
          this.curStates[i], actions[i], r.rewards[i], r.nextStates[i], allDone, this.episodeId,
        ),
        JointEnvStation.CH_AGENT_TRANSITION_PREFIX + i,
      );
    }
    this.curStates = r.nextStates;
    if (allDone) {
      this.episodeAccounting.finishEpisode(this.stepInEpisode);
      this.stepInEpisode = 0;
      this.episodeId += 1;
      if (this.episodeId < this.numEpisodes) {
        this.curStates = this.env.reset();
        this.emitStateForAllAgents();
      }
    }
  }

  protected emitStateForAllAgents(): void {
    for (let i = 0; i < this.env.numAgents; i++) {
      this.emit(new StateToken<S>(this.curStates[i], this.episodeId),
                JointEnvStation.CH_AGENT_STATE_PREFIX + i);
    }
  }

  /** Action channel name for agent i. */
  static actionChannel(i: number): ChannelName { return JointEnvStation.CH_AGENT_ACTION_PREFIX + i; }
  static stateChannel(i: number): ChannelName  { return JointEnvStation.CH_AGENT_STATE_PREFIX + i; }
  static transitionChannel(i: number): ChannelName { return JointEnvStation.CH_AGENT_TRANSITION_PREFIX + i; }
}

// -----------------------------------------------------------------------------
// MULTI-AGENT SYSTEM ORCHESTRATOR
// -----------------------------------------------------------------------------

export interface MultiAgentSystemOpts {
  rng?: () => number;
}

export class MultiAgentSystem<S = number, A = number> {
  readonly env: JointEnvStation<S, A>;
  readonly agents: RLAgentStation<S, A>[];

  constructor(env: JointEnvStation<S, A>, agents: RLAgentStation<S, A>[]) {
    if (agents.length !== env['env'].numAgents) {
      throw new Error(`expected ${env['env'].numAgents} agents, got ${agents.length}`);
    }
    this.env = env;
    this.agents = agents;
    // Wire the channels.
    for (let i = 0; i < agents.length; i++) {
      env.pipe(agents[i], JointEnvStation.stateChannel(i), RLAgentStation.CH_STATE);
      env.pipe(agents[i], JointEnvStation.transitionChannel(i), RLAgentStation.CH_TRANSITION);
      agents[i].pipe(env, RLAgentStation.CH_ACTION, JointEnvStation.actionChannel(i));
    }
  }

  /** Run the full multi-agent training loop. */
  run(opts: MultiAgentSystemOpts = {}): {
    ticks: number;
    rewardHistory: number[][];
    perAgentRewardHistory: number[][];
  } {
    const summary = runIterativeDES([this.env, ...this.agents], {rng: opts.rng});
    const perAgent: number[][] = this.agents.map(_ => []);
    for (const ep of this.env.rewardHistory) for (let i = 0; i < ep.length; i++) perAgent[i].push(ep[i]);
    return {
      ticks: summary.ticks,
      rewardHistory: this.env.rewardHistory.map(r => r.slice()),
      perAgentRewardHistory: perAgent,
    };
  }
}
