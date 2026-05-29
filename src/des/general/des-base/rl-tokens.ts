'use strict';

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
