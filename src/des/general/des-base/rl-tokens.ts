'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/rl_tokens.rs  (module des::general::des_base::rl_tokens)
// 1:1 file move. Common token payloads exchanged by RL stations.
//
// Declarations → Rust:
//   class StateToken<S>       -> struct StateToken<S> { state, episode_id } + impl Token
//   class ActionToken<S,A>    -> struct ActionToken<S, A> + impl Token
//   class TransitionToken<S,A> -> struct TransitionToken<S, A> + impl Token
//   class TrainTriggerToken   -> unit struct + impl Token
//   class ResumeToken         -> unit struct + impl Token
//
// Conversion notes (file-specific):
//   - Plain data carriers; generic defaults `S = number` / `A = number` -> pick
//     `S = f64` / `A = usize` (action index) or keep the type params.
//   - All implement the `Token` marker trait from station.rs.
// =============================================================================

// =============================================================================
// general/des-base/rl-tokens.ts — common token types for RL stations.
// =============================================================================

import {Token} from './station';

/** Sent by the environment to the agent at the start of each episode. */
export class StateToken<S = number> implements Token {
  constructor(public state: S, public episodeId: number) {}
}

/** Sent by the agent to the environment to apply an action. */
export class ActionToken<S = number, A = number> implements Token {
  constructor(public state: S, public action: A, public episodeId: number) {}
}

/** Sent by the environment after each step. */
export class TransitionToken<S = number, A = number> implements Token {
  constructor(
    public state: S,
    public action: A,
    public reward: number,
    public nextState: S,
    public done: boolean,
    public episodeId: number,
  ) {}
}

/** Sent by an agent to a "policy update" station when its rollout is full. */
export class TrainTriggerToken implements Token {}

/** Sent back to an agent that paused awaiting fresh parameters. */
export class ResumeToken implements Token {}
