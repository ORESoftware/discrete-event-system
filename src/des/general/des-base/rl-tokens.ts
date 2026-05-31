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
