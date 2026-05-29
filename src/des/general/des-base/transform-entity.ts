'use strict';

// =============================================================================
// TransformEntity
//
// A zero-backlog stationary entity for modeling ordinary program functions.
//
// In the DES vocabulary:
//   - free/pure "function functions" are stationary transform entities;
//   - movable/smart-movable methods are still allowed to mutate their owner.
//
// Unlike DESStation's default inbox behavior, TransformEntity does not queue.
// take(token, channel) immediately applies the transform and emits any returned
// token(s). It is therefore useful for lightweight adapters, parsers, validators,
// projections, feature extractors, and cost/reward functions that should be
// visible in the station graph without adding artificial waiting time.
// =============================================================================

import {ChannelName, DEFAULT_CHANNEL, DESStation, Token} from './station';
import {Preconditions} from './preconditions';

export interface TransformContext<I extends Token, O extends Token> {
  readonly station: TransformEntity<I, O>;
  readonly channel: ChannelName;
  readonly sequence: number;
  emit(token: O, channel?: ChannelName): void;
}

export type TransformResult<O extends Token> = O | readonly O[] | null | undefined;
export type TransformFunction<I extends Token, O extends Token> =
  (token: I, context: TransformContext<I, O>) => TransformResult<O>;

export interface TransformEntityOptions<I extends Token, O extends Token> {
  /** Accepted input channel(s). Defaults to the normal DES default channel. */
  inputChannels?: ChannelName | readonly ChannelName[];
  /** Destination channel for returned token(s). Defaults to the input channel. */
  outputChannel?: ChannelName | ((inputChannel: ChannelName, token: O) => ChannelName);
  /** Optional fail-fast guard for incoming tokens. */
  validateInput?: (token: I, channel: ChannelName) => void;
  /** Optional fail-fast guard for outgoing tokens. */
  validateOutput?: (token: O, channel: ChannelName) => void;
}

export abstract class TransformEntity<I extends Token = Token, O extends Token = Token> extends DESStation {
  private readonly inputChannels: Set<ChannelName>;
  private readonly outputChannel?: ChannelName | ((inputChannel: ChannelName, token: O) => ChannelName);
  private readonly validateInput: (token: I, channel: ChannelName) => void;
  private readonly validateOutput: (token: O, channel: ChannelName) => void;

  processedCount = 0;
  emittedCount = 0;
  droppedCount = 0;

  constructor(
    id: string,
    opts: TransformEntityOptions<I, O> = {},
  ) {
    super(id);
    const channels = opts.inputChannels ?? DEFAULT_CHANNEL;
    this.inputChannels = new Set(Array.isArray(channels) ? channels : [channels]);
    this.outputChannel = opts.outputChannel;
    this.validateInput = opts.validateInput ?? (() => {});
    this.validateOutput = opts.validateOutput ?? (() => {});
  }

  override assertPreconditions(): void {
    Preconditions.nonEmpty('TransformEntity', `${this.id}.inputChannels`, Array.from(this.inputChannels));
  }

  override take(token: Token, channel: ChannelName = DEFAULT_CHANNEL): void {
    const input = this.validateTransformInput(token, channel);
    super.take(input, channel);
  }

  protected validateTransformInput(token: Token, channel: ChannelName): I {
    if (!this.inputChannels.has(channel)) {
      throw new Error(`TransformEntity(${this.id}): unexpected input channel "${channel}"`);
    }
    const input = token as I;
    this.validateInput(input, channel);
    return input;
  }

  protected inputChannelNames(): ChannelName[] {
    return Array.from(this.inputChannels);
  }

  protected hasQueuedInput(): boolean {
    return this.inputChannelNames().some(channel => this.inboxSize(channel) > 0);
  }

  protected processTransformResult(
    input: I,
    channel: ChannelName,
    transform: TransformFunction<I, O>,
  ): void {
    const sequence = this.processedCount;
    this.processedCount++;
    const ctx: TransformContext<I, O> = {
      station: this,
      channel,
      sequence,
      emit: (out, outChannel = this.resolveOutputChannel(channel, out)) => this.emitOutput(out, outChannel),
    };
    const emittedBefore = this.emittedCount;
    const result = transform(input, ctx);
    if (result === undefined || result === null) {
      if (this.emittedCount === emittedBefore) this.droppedCount++;
      return;
    }
    const outputs = Array.isArray(result) ? result : [result];
    if (outputs.length === 0) {
      if (this.emittedCount === emittedBefore) this.droppedCount++;
      return;
    }
    for (const out of outputs) this.emitOutput(out, this.resolveOutputChannel(channel, out));
  }

  private emitOutput(token: O, channel: ChannelName): void {
    this.validateOutput(token, channel);
    this.emit(token, channel);
    this.emittedCount++;
  }

  private resolveOutputChannel(inputChannel: ChannelName, token: O): ChannelName {
    if (typeof this.outputChannel === 'function') return this.outputChannel(inputChannel, token);
    return this.outputChannel ?? inputChannel;
  }
}

/**
 * Base class for ordinary pure functions modeled as DES graph nodes.
 *
 * Subclasses supply only transform(), while TransformEntity keeps the
 * connection, channel, validation, fan-out, and zero-backlog accounting logic.
 */
export abstract class PureTransformEntity<I extends Token = Token, O extends Token = Token>
  extends TransformEntity<I, O> {

  constructor(id: string, opts: TransformEntityOptions<I, O> = {}) {
    super(id, opts);
  }

  override hasWork(): boolean {
    return false;
  }

  override take(token: Token, channel: ChannelName = DEFAULT_CHANNEL): void {
    const input = this.validateTransformInput(token, channel);
    this.processTransformResult(input, channel, (x, context) => this.transform(x, context));
  }

  runTimeStep(): void {}

  abstract transform(token: I, context?: TransformContext<I, O>): TransformResult<O>;
}

/**
 * Queue-backed transform for functions that need local memory or operate over
 * the current DES time slice. Inputs wait in named inboxes until runTimeStep().
 */
export abstract class MemoryTransformEntity<
  I extends Token = Token,
  O extends Token = Token,
  P = unknown,
> extends TransformEntity<I, O> {
  constructor(id: string, public previous: P, opts: TransformEntityOptions<I, O> = {}) {
    super(id, opts);
  }

  override hasWork(): boolean {
    return this.hasQueuedInput();
  }

  runTimeStep(): void {
    for (const channel of this.inputChannelNames()) {
      for (const input of this.drain<I>(channel)) {
        this.processTransformResult(input, channel, (token, context) => this.transformQueued(token, context));
      }
    }
  }

  protected abstract transformQueued(token: I, context: TransformContext<I, O>): TransformResult<O>;
}

/** Concrete pure function station for users who prefer function-call syntax. */
export class FunctionEntity<I extends Token = Token, O extends Token = Token> extends PureTransformEntity<I, O> {
  constructor(
    id: string,
    private readonly fn: TransformFunction<I, O>,
    opts: TransformEntityOptions<I, O> = {},
  ) {
    super(id, opts);
  }

  transform(token: I, context: TransformContext<I, O>): TransformResult<O> {
    return this.fn(token, context);
  }
}
