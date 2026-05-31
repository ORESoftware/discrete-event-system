'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/station.rs
// - Keep file-for-file. Token and DESRunLoopEntity become traits; ChannelName
//   can be a String newtype or enum-backed identifier.
// - DESStation becomes the foundational shared state struct plus station trait
//   impls for inboxes, outgoing edges, validators, and lifecycle hooks.
// - Map fields map to HashMap/BTreeMap of channel names to token queues/edges;
//   structural typing must become explicit trait bounds and boxed trait objects.
// - Pure station-graph adapters should derive from PureTransform/PureTransformEntity
//   in transform_entity.rs. Convert thrown validation/connection errors to Result.

// =============================================================================
// general/des-base/station.ts — DESStation, the foundation for the
// "iterative algorithm as DES" hierarchy in des-base/.
//
// This is a DELIBERATELY thinner alternative to the StationaryEntity /
// AbstractMovingEntity tree in `abstract/`, which is built for queueing
// networks (BigNumber time accrual, queue-of-queues, processor stations).
// For algorithm-as-DES we want exactly:
//
//   1. A station that can hold internal mutable state (Q-table, weights,
//      population, walker, etc.)
//   2. Multiple TYPED inboxes addressed by channel name
//   3. An explicit edge graph (this.pipe(other, channel)) — no FEL
//   4. A `runTimeStep()` hook driven by the iterative runner
//
// Algorithm-family base classes (SingleStateOptimizer, PopulationOptimizer,
// RLAgentStation, PolicyGradientAgent, EnvironmentStation, …) extend
// DESStation and IMPLEMENT runTimeStep as a TEMPLATE METHOD that calls
// abstract hooks. Concrete algorithms (SA, GA, Q-learning, PPO) implement
// only the hooks. The template-method's runTimeStep should NOT be
// overridden by leaf classes — that's the structural enforcement.
// =============================================================================

/** Marker for a payload that flows on an edge. */
export interface Token {}

/** Shared contract for anything that can be advanced by the DES tick loop.
 *  Stations implement this, and smart movables can implement it too. */
export interface HasRunTimeStep {
  runTimeStep(): void;
}

/** Channel names are simple strings; algorithms typically expose a named
 *  enum-like constant set. The default channel is `'default'`. */
export type ChannelName = string;
export const DEFAULT_CHANNEL: ChannelName = 'default';

import {Validator, ValidationCheck, runValidators, formatValidationReport} from './validation';

export interface DESRunLoopEntity extends HasRunTimeStep {
  readonly id?: string;
  assertPreconditions?(): void;
  hasWork?(): boolean;
  onFinalize?(): void;
  numValidators?(): number;
  runValidation?(): ValidationCheck[];
}

interface OutEdge {
  target: DESStation;
  /** Channel name on the TARGET station. */
  targetChannel: ChannelName;
}

export abstract class DESStation implements DESRunLoopEntity {
  readonly id: string;

  /** Inbound queues, one per channel. Lazily created on first take(). */
  protected readonly inboxes: Map<ChannelName, Token[]> = new Map();

  /** Outbound edges, one bucket per channel name on THIS station.
   *  emit(t, ch) routes to all (target, targetChannel) tuples in
   *  `outs.get(ch)`. */
  protected readonly outs: Map<ChannelName, OutEdge[]> = new Map();

  /** Pluggable validators registered on this station. The runner calls
   *  `runValidation()` after `onFinalize()` once the algorithm has
   *  terminated, so leaf classes can ship intrinsic invariants AND
   *  external-reference comparisons attached to the model itself. */
  protected readonly validators: Validator<this>[] = [];

  constructor(id: string) {
    this.id = id;
  }

  // ── PIPELINE WIRING ──────────────────────────────────────────────────────

  /** Connect THIS station's `srcChannel` output to `target`'s
   *  `tgtChannel` input. If `tgtChannel` is omitted it equals `srcChannel`. */
  pipe(target: DESStation, srcChannel: ChannelName = DEFAULT_CHANNEL,
       tgtChannel: ChannelName = srcChannel): this {
    let arr = this.outs.get(srcChannel);
    if (!arr) { arr = []; this.outs.set(srcChannel, arr); }
    arr.push({target, targetChannel: tgtChannel});
    return this;
  }

  // ── TOKEN I/O ────────────────────────────────────────────────────────────

  /** Place a token on the inbox of the given channel. Generally called
   *  by another station's emit() — algorithms shouldn't call this on
   *  themselves. */
  take(t: Token, channel: ChannelName = DEFAULT_CHANNEL): void {
    let arr = this.inboxes.get(channel);
    if (!arr) { arr = []; this.inboxes.set(channel, arr); }
    arr.push(t);
  }

  /** Drain (and clear) the inbox for a channel; returns the tokens. */
  protected drain<T extends Token = Token>(channel: ChannelName = DEFAULT_CHANNEL): T[] {
    const arr = this.inboxes.get(channel);
    if (!arr || arr.length === 0) return [];
    const out = arr as T[];
    this.inboxes.set(channel, []);
    return out;
  }

  /** Peek without consuming. */
  protected peek<T extends Token = Token>(channel: ChannelName = DEFAULT_CHANNEL): readonly T[] {
    return (this.inboxes.get(channel) ?? []) as T[];
  }

  /** Number of pending tokens on a channel. */
  protected inboxSize(channel: ChannelName = DEFAULT_CHANNEL): number {
    return this.inboxes.get(channel)?.length ?? 0;
  }

  /** Emit a token on a channel to every connected target. */
  protected emit(t: Token, channel: ChannelName = DEFAULT_CHANNEL): void {
    const arr = this.outs.get(channel);
    if (!arr) return;
    for (const edge of arr) edge.target.take(t, edge.targetChannel);
  }

  // ── RUNNER PROTOCOL ──────────────────────────────────────────────────────

  /** Pre-run guard. Called ONCE by `runIterativeDES` before any tick.
   *  Subclasses override to fail-fast on invalid initial conditions /
   *  parameters using `Preconditions.*` guards from
   *  `des-base/preconditions.ts`. The default implementation is a no-op
   *  so models that don't (yet) override it still work, but every
   *  algorithm SHOULD override this and assert its core invariants
   *  (probabilities sum to 1, dt > 0, gains non-negative, etc.). */
  assertPreconditions(): void {}

  /** Default: any non-empty inbox = work. Subclasses with internal
   *  scheduling (e.g. paused state, "I have an episode running") can
   *  override but must still respect inbox tokens. */
  hasWork(): boolean {
    for (const arr of this.inboxes.values()) if (arr.length > 0) return true;
    return false;
  }

  /** Single iteration of this station's behaviour. Algorithm-family
   *  bases (SingleStateOptimizer, PopulationOptimizer, RLAgentStation,
   *  PolicyGradientAgent) IMPLEMENT this as a template method calling
   *  abstract hooks; concrete algorithms (SA, GA, Q-learning, PPO)
   *  IMPLEMENT THE HOOKS, not runTimeStep. */
  abstract runTimeStep(): void;

  // ── FINALIZATION + VALIDATION ────────────────────────────────────────────

  /** Called once by `runIterativeDES` after the loop terminates and BEFORE
   *  validators run. Override for end-of-run cleanup, summary computation,
   *  or to programmatically attach validators based on the final state.
   *  Default: no-op. */
  onFinalize(): void {}

  /** Register a validator on this station. Validators run after the
   *  algorithm terminates and produce ValidationCheck entries that the
   *  runner aggregates into its summary. */
  addValidator(v: Validator<this>): this {
    this.validators.push(v);
    return this;
  }

  /** Number of validators currently registered. */
  numValidators(): number { return this.validators.length; }

  /** Run all registered validators and return a flat list of checks.
   *  Validators that throw are captured as failed checks so one buggy
   *  validator never blocks the rest. */
  runValidation(): ValidationCheck[] {
    return runValidators(this, this.validators);
  }

  /** Pretty-printed multi-line validation report. */
  validationReport(): string {
    return formatValidationReport(this.runValidation());
  }

  // ── INTROSPECTION ────────────────────────────────────────────────────────

  /** Snapshot of inbox sizes by channel — useful for tests + logs. */
  inboxSizes(): Record<ChannelName, number> {
    const out: Record<ChannelName, number> = {};
    for (const [ch, arr] of this.inboxes) out[ch] = arr.length;
    return out;
  }
}
