'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/environment.rs  (module des::general::des_base::environment)
// 1:1 file move. Wraps a pure (non-DES) Environment in an RL station emitting
// StateToken / TransitionToken and consuming ActionToken.
//
// Declarations → Rust:
//   interface PureEnvironment<S,A>      -> trait PureEnvironment<S, A>
//                                          (step returns struct StepResult { next_state, reward, done })
//   interface EnvironmentStationOptions -> struct (#[derive(Default)])
//   class EnvironmentStation<S,A>       -> struct + impl DESStation (holds a `dyn PureEnvironment`)
//
// Conversion notes (file-specific):
//   - `step` returns an inline object `{nextState, reward, done}` -> a named
//     `StepResult<S>` struct (no anonymous structs in Rust).
//   - `render?(state)` optional method -> provided default returning `String`.
//   - `env: PureEnvironment` is held by composition -> `Box<dyn PureEnvironment<S,A>>`.
//   - getter/setter pairs + `rewardHistory` alias of episodeAccounting -> plain
//     accessor methods borrowing the inner EpisodeAccounting (no owned alias).
//   - `Required<EnvironmentStationOptions>` (all-filled opts) -> a resolved
//     config struct with concrete fields; `Infinity` default -> `f64::INFINITY`
//     or `Option<u64>`/`u64::MAX` for episode caps.
//   - `done` is a public externally-set flag -> `pub done: bool`.
// =============================================================================

// =============================================================================
// general/des-base/environment.ts — generic Environment Station for RL.
//
// Wraps a pure (non-DES) Environment in a station that:
//   • emits a StateToken on its 'state' channel at the start of every
//     episode (and once at the very beginning to seed the loop)
//   • accepts ActionToken on its 'action' channel; calls env.step;
//     emits a TransitionToken on its 'transition' channel
//   • optionally truncates episodes that exceed maxStepsPerEpisode
//   • stops emitting once `numEpisodes` is reached OR an external
//     observer flips `done = true` (used by PPO when the global step
//     budget is reached)
//
// Channel summary
// ───────────────
//   in:   'action'      (ActionToken)
//   out:  'state'       (StateToken)         — only at episode start
//   out:  'transition'  (TransitionToken)    — after every env.step
// =============================================================================

import {DESStation, ChannelName} from './station';
import {StateToken, ActionToken, TransitionToken} from './rl-tokens';
import {EpisodeAccounting} from './episode-accounting';

export interface PureEnvironment<S = number, A = number> {
  numStates: number;
  numActions: number;
  reset(): S;
  step(state: S, action: A): {nextState: S; reward: number; done: boolean};
  render?(state: S): string;
}

export interface EnvironmentStationOptions {
  /** Maximum episodes to run. Use Infinity for "until external stop". */
  numEpisodes?: number;
  /** Truncate episodes longer than this many steps. */
  maxStepsPerEpisode?: number;
}

export class EnvironmentStation<S = number, A = number> extends DESStation {
  protected readonly env: PureEnvironment<S, A>;
  protected readonly opts: Required<EnvironmentStationOptions>;

  protected curState: S;
  protected episodeId = 0;
  protected stepInEpisode = 0;
  protected emittedStart = false;
  /** Externally settable termination flag — used by step-budget drivers. */
  done = false;

  /** Per-episode return + length, populated when an episode ends. */
  protected readonly episodeAccounting = new EpisodeAccounting();
  readonly rewardHistory: number[] = this.episodeAccounting.rewardHistory;
  readonly lengthHistory: number[] = this.episodeAccounting.lengthHistory;
  get totalSteps(): number { return this.episodeAccounting.totalSteps; }
  set totalSteps(value: number) { this.episodeAccounting.totalSteps = value; }
  protected get curReturn(): number { return this.episodeAccounting.currentReward; }
  protected set curReturn(value: number) { this.episodeAccounting.currentReward = value; }
  protected get curLength(): number { return this.episodeAccounting.currentLength; }
  protected set curLength(value: number) { this.episodeAccounting.currentLength = value; }

  static readonly CH_ACTION: ChannelName = 'action';
  static readonly CH_STATE: ChannelName = 'state';
  static readonly CH_TRANSITION: ChannelName = 'transition';

  constructor(id: string, env: PureEnvironment<S, A>, opts: EnvironmentStationOptions = {}) {
    super(id);
    this.env = env;
    this.opts = {
      numEpisodes: opts.numEpisodes ?? Infinity,
      maxStepsPerEpisode: opts.maxStepsPerEpisode ?? 1_000_000,
    };
    this.curState = env.reset();
  }

  override hasWork(): boolean {
    if (this.done) return false;
    if (!this.emittedStart) return true;
    if (this.episodeId >= this.opts.numEpisodes) return false;
    return this.inboxSize(EnvironmentStation.CH_ACTION) > 0;
  }

  runTimeStep(): void {
    if (this.done) return;
    if (!this.emittedStart) {
      this.emittedStart = true;
      this.emit(new StateToken<S>(this.curState, this.episodeId), EnvironmentStation.CH_STATE);
      return;
    }
    if (this.episodeId >= this.opts.numEpisodes) return;
    const actions = this.drain<ActionToken<S, A>>(EnvironmentStation.CH_ACTION);
    for (const a of actions) {
      if (a.episodeId !== this.episodeId) continue;
      const r = this.env.step(a.state, a.action);
      this.episodeAccounting.recordStep(r.reward);
      this.stepInEpisode += 1;
      const truncated = this.stepInEpisode >= this.opts.maxStepsPerEpisode;
      const isDone = r.done || truncated;
      this.emit(
        new TransitionToken<S, A>(a.state, a.action, r.reward, r.nextState, isDone, this.episodeId),
        EnvironmentStation.CH_TRANSITION,
      );
      if (isDone) {
        this.episodeAccounting.finishEpisode();
        this.stepInEpisode = 0;
        this.episodeId += 1;
        if (this.episodeId < this.opts.numEpisodes) {
          this.curState = this.env.reset();
          this.emit(new StateToken<S>(this.curState, this.episodeId), EnvironmentStation.CH_STATE);
        }
      }
    }
  }
}
