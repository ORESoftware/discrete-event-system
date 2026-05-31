'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/policy_gradient_agent.rs
// - Keep file-for-file. RolloutEntry becomes a data struct.
// - PolicyGradientAgent and PolicyUpdateStation become traits plus shared
//   station-state structs for rollout buffering, reward accounting, and update
//   emission.
// - Policy/value/update hooks map to trait methods; any pure advantage or loss
//   computation lifted into the graph should use PureTransform/PureTransformEntity.
// - Convert invalid rollout/update state and emitted-token failures to Result.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/policy_gradient_agent.rs  (module des::general::des_base::policy_gradient_agent)
// 1:1 file move. Template-method bases for policy-gradient methods (REINFORCE /
// A2C / PPO / TRPO) plus the paired PolicyUpdateStation.
//
// Declarations → Rust:
//   interface RolloutEntry<S,A>     -> struct RolloutEntry<S, A> (Option fields r/done/sNext)
//   abstract class PolicyGradientAgent<S,A> -> trait/struct: DESStation (template-method
//                                              runTimeStep + samplePolicyAndValue hook)
//   abstract class PolicyUpdateStation -> trait/struct: DESStation (runUpdate hook)
//
// Conversion notes (file-specific):
//   - TEMPLATE METHOD: agent's `runTimeStep` is final; required hook
//     samplePolicyAndValue -> required trait fn; PolicyUpdateStation's runUpdate
//     -> required trait fn. The UPDATE mutates `theta`/`V` in place.
//   - `rng: () => number` -> inject `RandomSource` (shared/capabilities).
//   - `pendingActionState: {state; episodeId} | null` -> `Option<PendingAction<S>>`.
//   - Buffer search matches `e.s === t.state && e.a === t.action` by value
//     equality -> require `S: PartialEq, A: PartialEq` (no JS reference identity).
//   - getter/setter pairs -> plain methods; `rewardHistory` alias -> accessor
//     borrowing the inner EpisodeAccounting (no owned alias in Rust).
//   - The agent/update station communicate via tokens AND a shared param handle —
//     model with `Rc<RefCell<Params>>` so the update station can mutate the agent.
// =============================================================================

// =============================================================================
// general/des-base/policy-gradient-agent.ts — base class for POLICY-GRADIENT
// methods: REINFORCE, A2C, PPO (clipped & adaptive-KL), TRPO, IMPALA, …
//
// PROBLEM SHAPE
// ─────────────
//   The agent samples actions from π_θ(a|s), records
//   (s, a, log π_θ_old(a|s), V_φ(s)) in a ROLLOUT BUFFER, and after T
//   environment steps it computes advantages and applies a parameter
//   update. The DIFFERENTIATOR is the update rule:
//
//     - REINFORCE:   θ ← θ + α A_t ∇log π_θ(a_t|s_t)
//     - A2C:         actor + critic SGD on advantages
//     - PPO clip:    L = E[ min(r·A, clip(r,1−ε,1+ε)·A) ]
//     - TRPO:        natural-gradient with KL trust region
//
// LIFECYCLE OF ONE ROLLOUT
// ────────────────────────
//   1. Env emits StateToken (start of episode) → agent samples and emits
//      Action; agent appends buffer entry {s, a, logp, v}.
//   2. Env emits TransitionToken → agent FILLS the latest entry's
//      (r, sNext, done). If !done → sample next action; if done → wait
//      for env's next StateToken.
//   3. When buffer is full (>= rolloutLen) the agent PAUSES, emits
//      TrainTriggerToken on CH_TRAIN, and waits for ResumeToken.
//   4. (External) PolicyUpdateStation consumes the trigger, runs the
//      update on the agent's parameters, clears the buffer, sends
//      ResumeToken.
//   5. Agent unpauses; rollout resumes.
//
// CHANNELS
// ────────
//   in:  CH_STATE        StateToken
//   in:  CH_TRANSITION   TransitionToken
//   in:  CH_RESUME       ResumeToken
//   out: CH_ACTION       ActionToken (to env)
//   out: CH_TRAIN        TrainTriggerToken (to update station)
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep(): see implementation below.
//
// HOOKS (abstract)
// ────────────────
//   samplePolicyAndValue(state, rng) → {action, logProb, value}
//   The actual UPDATE is performed by a separate PolicyUpdateStation
//   subclass (e.g. PPOClipUpdateStation), which mutates `this.theta`
//   and `this.V` in place and signals ResumeToken.
// =============================================================================

import {DESStation, ChannelName} from './station';
import {StateToken, ActionToken, TransitionToken, TrainTriggerToken, ResumeToken} from './rl-tokens';
import {EpisodeAccounting} from './episode-accounting';

/** A single (s, a, logp_old, v, r, done, sNext) tuple in the buffer. */
export interface RolloutEntry<S, A> {
  s: S;
  a: A;
  logProbOld: number;
  v: number;
  r?: number;
  done?: boolean;
  sNext?: S;
}

export abstract class PolicyGradientAgent<S = number, A = number> extends DESStation {
  static readonly CH_STATE: ChannelName = 'state';
  static readonly CH_TRANSITION: ChannelName = 'transition';
  static readonly CH_ACTION: ChannelName = 'action';
  static readonly CH_TRAIN: ChannelName = 'train';
  static readonly CH_RESUME: ChannelName = 'resume';

  protected readonly rng: () => number;
  protected readonly rolloutLen: number;
  /** True while waiting for ResumeToken after triggering an update. */
  protected paused = false;

  protected buffer: RolloutEntry<S, A>[] = [];
  /** Stash for the state we owe an action on after the next ResumeToken.
   *  Set when the buffer fills mid-rollout on a NON-terminal transition;
   *  consumed in the resume-handling branch of the next runTimeStep. */
  private pendingActionState: {state: S; episodeId: number} | null = null;

  numUpdates = 0;
  protected readonly episodeAccounting = new EpisodeAccounting();
  readonly rewardHistory: number[] = this.episodeAccounting.rewardHistory;
  readonly lengthHistory: number[] = this.episodeAccounting.lengthHistory;
  get totalSteps(): number { return this.episodeAccounting.totalSteps; }
  set totalSteps(value: number) { this.episodeAccounting.totalSteps = value; }
  protected get episodeReward(): number { return this.episodeAccounting.currentReward; }
  protected set episodeReward(value: number) { this.episodeAccounting.currentReward = value; }

  constructor(id: string, opts: {rolloutLen: number; rng: () => number}) {
    super(id);
    this.rolloutLen = opts.rolloutLen;
    this.rng = opts.rng;
  }

  override hasWork(): boolean {
    if (this.paused) return this.inboxSize(PolicyGradientAgent.CH_RESUME) > 0;
    return this.inboxSize(PolicyGradientAgent.CH_STATE) > 0
        || this.inboxSize(PolicyGradientAgent.CH_TRANSITION) > 0;
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  /** Sample a' ~ π_θ(·|s) and report (a, log π(a|s), V_φ(s)). */
  protected abstract samplePolicyAndValue(state: S, rng: () => number): {
    action: A; logProb: number; value: number;
  };

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  runTimeStep(): void {
    if (this.paused) {
      const resumes = this.drain<ResumeToken>(PolicyGradientAgent.CH_RESUME);
      if (resumes.length === 0) return;
      this.paused = false;
      // Resume the rollout: act on the state we owe.
      if (this.pendingActionState !== null) {
        const ps = this.pendingActionState;
        this.pendingActionState = null;
        const {action, logProb, value} = this.samplePolicyAndValue(ps.state, this.rng);
        this.buffer.push({s: ps.state, a: action, logProbOld: logProb, v: value});
        this.emit(new ActionToken<S, A>(ps.state, action, ps.episodeId), PolicyGradientAgent.CH_ACTION);
      }
    }
    // 1. Process transitions: fill the buffer entry for (s, a) just emitted.
    const transitions = this.drain<TransitionToken<S, A>>(PolicyGradientAgent.CH_TRANSITION);
    for (const t of transitions) {
      // Find the most-recent un-completed entry; this should be the last.
      for (let i = this.buffer.length - 1; i >= 0; i--) {
        const e = this.buffer[i];
        if (e.r === undefined && e.s === t.state && e.a === t.action) {
          e.r = t.reward; e.done = t.done; e.sNext = t.nextState;
          break;
        }
      }
      this.episodeAccounting.recordStep(t.reward);
      if (t.done) {
        this.episodeAccounting.finishEpisode();
      }
      // If buffer is full → trigger train phase. Pause; remaining
      // transitions in this drain are processed AFTER resume (won't
      // happen — env is also paused-on-no-actions, and on next tick
      // the env will receive no actions because we don't emit until
      // we resume).
      if (this.buffer.length >= this.rolloutLen) {
        this.paused = true;
        // If the just-completed transition was NON-terminal we owe an
        // action on s' once the update station finishes — stash it so
        // the resume branch can fire it. If t.done, the env will reset
        // and a fresh StateToken will arrive after we resume.
        if (!t.done) {
          this.pendingActionState = {state: t.nextState, episodeId: t.episodeId};
        }
        this.emit(new TrainTriggerToken(), PolicyGradientAgent.CH_TRAIN);
        return;
      }
      // If !done, sample next action on s'. If done, env will emit a
      // new StateToken — handled below in this same tick.
      if (!t.done) {
        const {action, logProb, value} = this.samplePolicyAndValue(t.nextState, this.rng);
        this.buffer.push({s: t.nextState, a: action, logProbOld: logProb, v: value});
        this.emit(new ActionToken<S, A>(t.nextState, action, t.episodeId), PolicyGradientAgent.CH_ACTION);
      }
    }
    // 2. Process new-episode states.
    const states = this.drain<StateToken<S>>(PolicyGradientAgent.CH_STATE);
    for (const s of states) {
      const {action, logProb, value} = this.samplePolicyAndValue(s.state, this.rng);
      this.buffer.push({s: s.state, a: action, logProbOld: logProb, v: value});
      this.emit(new ActionToken<S, A>(s.state, action, s.episodeId), PolicyGradientAgent.CH_ACTION);
    }
  }

  // ── INTROSPECTION HELPERS USED BY UPDATE STATIONS ────────────────────────

  getBuffer(): readonly RolloutEntry<S, A>[] { return this.buffer; }
  clearBuffer(): void { this.buffer = []; }
  isPaused(): boolean { return this.paused; }
  numQueuedTrain(): number { return this.inboxSize(PolicyGradientAgent.CH_RESUME); }
}

// -----------------------------------------------------------------------------
// PolicyUpdateStation — counterpart to PolicyGradientAgent. Listens for
// CH_TRAIN, runs an update on the agent's parameters, emits CH_RESUME.
//
// The actual update rule (PPO-clip / vanilla / TRPO) is in subclasses.
// -----------------------------------------------------------------------------

export abstract class PolicyUpdateStation extends DESStation {
  static readonly CH_TRAIN: ChannelName = 'train';
  static readonly CH_RESUME: ChannelName = 'resume';

  numUpdates = 0;

  constructor(id: string) { super(id); }

  override hasWork(): boolean {
    return this.inboxSize(PolicyUpdateStation.CH_TRAIN) > 0;
  }

  /** Subclass: actually mutate params using the rollout buffer of the
   *  PolicyGradientAgent attached at construction time. */
  protected abstract runUpdate(): void;

  runTimeStep(): void {
    const triggers = this.drain<TrainTriggerToken>(PolicyUpdateStation.CH_TRAIN);
    for (const _ of triggers) {
      this.runUpdate();
      this.numUpdates += 1;
      this.emit(new ResumeToken(), PolicyUpdateStation.CH_RESUME);
    }
  }
}
