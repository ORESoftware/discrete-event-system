'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/rl_agent.rs
// - Keep file-for-file. RLAgentStation becomes the core RL-agent trait plus a
//   shared state struct for RNG, episode accounting, and channel behavior.
// - pickAction/update/endOfEpisode hooks should become trait methods, with
//   run_time_step implemented once over the shared state.
// - State/action/transition tokens come from rl_tokens.rs as structs; channel
//   strings can become pub consts or small enums.
// - Pure policy/value functions used as graph nodes should implement
//   PureTransform/PureTransformEntity. Convert invalid transitions to Result.

// =============================================================================
// general/des-base/rl-agent.ts — base class for ONLINE TEMPORAL-DIFFERENCE
// agents: Q-learning, SARSA, expected SARSA, Double-Q, Q(λ), …
//
// PROBLEM SHAPE
// ─────────────
//   An agent that:
//     • receives a `StateToken` at the start of each episode
//     • receives a `TransitionToken (s, a, r, s', done)` after every step
//     • emits an `ActionToken (s, a)` whenever it needs to act
//     • applies an UPDATE rule on each transition
//
//   The DIFFERENTIATOR among agents is the UPDATE rule:
//
//     - Q-learning:    Q[s,a] ← Q[s,a] + α(r + γ max_a' Q[s',a'] − Q[s,a])
//     - SARSA:         Q[s,a] ← Q[s,a] + α(r + γ Q[s',a'] − Q[s,a])
//     - Expected SARSA: Q[s,a] ← Q[s,a] + α(r + γ Σ_a' π(a'|s') Q[s',a'] − Q[s,a])
//     - REINFORCE:     θ ← θ + α (Σ_t γ^t r_t) ∇log π(a|s)
//
// CHANNEL CONVENTIONS
// ───────────────────
//   in:  CH_STATE       — StateToken at episode start
//   in:  CH_TRANSITION  — TransitionToken after each env step
//   out: CH_ACTION      — ActionToken to send to the env
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep():
//     for each transition: update + bookkeeping; if !done emit next ActionToken
//     for each new-episode state: emit ActionToken
//
// HOOKS (abstract)
// ────────────────
//   pickAction(state, rng) → action
//   update(s, a, r, sNext, done)
//
// HOOKS (optional override)
// ─────────────────────────
//   endOfEpisode(episodeId)
// =============================================================================

import {DESStation, ChannelName} from './station';
import {StateToken, ActionToken, TransitionToken} from './rl-tokens';
import {EpisodeAccounting} from './episode-accounting';

export abstract class RLAgentStation<S = number, A = number> extends DESStation {
  static readonly CH_STATE: ChannelName = 'state';
  static readonly CH_TRANSITION: ChannelName = 'transition';
  static readonly CH_ACTION: ChannelName = 'action';

  protected readonly rng: () => number;

  protected readonly episodeAccounting = new EpisodeAccounting();
  readonly rewardHistory: number[] = this.episodeAccounting.rewardHistory;
  readonly lengthHistory: number[] = this.episodeAccounting.lengthHistory;
  get totalSteps(): number { return this.episodeAccounting.totalSteps; }
  set totalSteps(value: number) { this.episodeAccounting.totalSteps = value; }
  protected get episodeReward(): number { return this.episodeAccounting.currentReward; }
  protected set episodeReward(value: number) { this.episodeAccounting.currentReward = value; }
  protected get episodeLength(): number { return this.episodeAccounting.currentLength; }
  protected set episodeLength(value: number) { this.episodeAccounting.currentLength = value; }

  constructor(id: string, opts: {rng: () => number}) {
    super(id);
    this.rng = opts.rng;
  }

  override hasWork(): boolean {
    return this.inboxSize(RLAgentStation.CH_TRANSITION) > 0
        || this.inboxSize(RLAgentStation.CH_STATE) > 0;
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  /** Pick an action in `state`. Typically ε-greedy; subclasses encode policy. */
  protected abstract pickAction(state: S, rng: () => number): A;
  /** Apply a TD update from a transition. */
  protected abstract update(state: S, action: A, reward: number, nextState: S, done: boolean): void;
  /** Optional: hook called when an episode ends (with the just-finished
   *  episode's id). Decay ε here, log, etc. */
  protected endOfEpisode(_episodeId: number): void {}

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  runTimeStep(): void {
    // 1. Apply transitions (Q-update, then act on s' if not done).
    const transitions = this.drain<TransitionToken<S, A>>(RLAgentStation.CH_TRANSITION);
    for (const t of transitions) {
      this.update(t.state, t.action, t.reward, t.nextState, t.done);
      this.episodeAccounting.recordStep(t.reward);
      if (t.done) {
        this.episodeAccounting.finishEpisode();
        this.endOfEpisode(t.episodeId);
        // Note: do NOT emit on done — wait for the env's next StateToken
        // (it will arrive on the same tick or the next).
      } else {
        const a = this.pickAction(t.nextState, this.rng);
        this.emit(new ActionToken<S, A>(t.nextState, a, t.episodeId), RLAgentStation.CH_ACTION);
      }
    }
    // 2. Process StateTokens (start of an episode).
    const states = this.drain<StateToken<S>>(RLAgentStation.CH_STATE);
    for (const s of states) {
      const a = this.pickAction(s.state, this.rng);
      this.emit(new ActionToken<S, A>(s.state, a, s.episodeId), RLAgentStation.CH_ACTION);
    }
  }
}
