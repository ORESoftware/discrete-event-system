'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/rl_tokens.rs
// - Keep file-for-file. StateToken, ActionToken, TransitionToken,
//   TrainTriggerToken, and ResumeToken become token structs.
// - Generic state/action payloads should become type parameters where the owning
//   agent/environment is generic, or concrete enums when crossing dynamic
//   boundaries.
// - Empty marker token classes map to unit structs implementing the Token marker
//   trait from station.rs.
// - No throws here; constructors should stay infallible unless future validation
//   requires Result.

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
