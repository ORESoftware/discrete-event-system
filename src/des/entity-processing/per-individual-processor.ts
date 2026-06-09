'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_processing/per_individual_processor.rs
// - PerIndividualProcessorOpts and QueuedItem become structs; the processor
//   itself owns `Vec<QueuedItem>` plus connection state and implements the same
//   processor/queue/entity traits as EntityProcessor.
// - `drawDuration: () => number` should be an injected RNG/service-time trait or
//   generic closure parameter; keep this as a PureTransform boundary for
//   per-entity residence-time draws.
// - The `(m as any)` compatibility shims, `reg.registerProcessor(this as any)`,
//   and no-downstream retry loop need typed trait bounds and Result/logging
//   decisions in Rust.

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-processing/per-individual-processor.rs  (module des::entity_processing::per_individual_processor)
// 1:1 file move. FEL-style per-individual service-time processor (M/M/inf-ish).
//
// Declarations → Rust:
//   interface PerIndividualProcessorOpts -> struct { draw_duration, rv?, output_routing? }
//   interface QueuedItem                  -> struct QueuedItem { entity, remaining_time: f64 }
//   class PerIndividualProcessor<S,T>     -> struct + impl (+ impl AbstractBidirectionalEntity,
//                                            HasComputedProperties, HasInternalQueue)
//
// Conversion notes (file-specific):
//   - `drawDuration: () => number` is a CLOSURE that draws a residence time ->
//     `Box<dyn FnMut() -> f64>` (or a RandomSource-backed sampler trait); use
//     `move` and watch the borrow checker. Prefer injecting RandomSource over a raw closure.
//   - `['isProcessor'] = true` brand -> a trait / enum tag (no symbol property).
//   - `m instanceof ProcessableMovingEntity || (m as any).startNewStation` is a
//     runtime type test + dynamic method probe -> model entities as a trait/enum;
//     Rust has no `instanceof`. The many `(x as any)?.field` accesses need concrete types.
//   - `reg.registerProcessor(this as any)` cast -> a typed registry entry.
//   - `items: QueuedItem[]` is the real store; `queue: LinkedQueue` is only a
//     HasInternalQueue facade -> back items with `Vec<QueuedItem>`; `items.unshift`
//     retry -> `VecDeque::push_front`.
//   - `Number(stepSize)` BigNumber->f64 conversion happens up front (`dt`).
//   - `getSerializableData(): Partial<this>` spread+null -> serde DTO with skips.
// =============================================================================

// =============================================================================
// PerIndividualProcessor: a station type that mirrors the FEL kernel's
// semantics inside the framework's run-loop.
//
// Differences from EntityProcessor:
//   - Single queue of {entity, remainingTime} entries (no input/processing/out
//     three-stage dance).
//   - When takeItem(m) is called, an inter-completion duration is drawn from
//     the configured uniform distribution and stored alongside the entity.
//   - On each runTimeStep(stepSize), every entity's remainingTime is
//     decremented by stepSize. Entities whose remainingTime reaches <=0 are
//     immediately routed downstream via the configured output-routing policy.
//
// This is exactly the M/M/inf-with-fixed-time-step-resolution semantic, which
// matches the classical FEL kernel's per-station service-time semantics
// (which our reference implementation uses). With small stepSize the agreement
// should be tight; with stepSize = 0 (impossible) it would be exact.
// =============================================================================

import * as math from 'mathjs';
import {AbstractBidirectionalEntity, EntityConnection, TimeStepOpts} from '../abstract/abstract';
import {AbstractMovingEntity, BasicMovingEntity, ProcessableMovingEntity} from '../entity-moving/moving';
import {HasComputedProperties} from '../general/general';
import {HasInternalQueue} from '../abstract/interfaces';
import {RandomVariable} from '../random-variables/rv';
import {LinkedQueue} from '@oresoftware/linked-queue';
import {reg} from '../general/entity-registration';
import {OutputConnectionRouter, OutputRoutingPolicy} from '../entity-routing/output-routing-policy';
import {debugLog} from '../shared/debug-log';

export interface PerIndividualProcessorOpts {
  /**
   * Per-individual residence time draw (days). Returns a fresh independent
   * sample each call. Wired with a uniform [a,b] in the runner.
   */
  drawDuration: () => number;

  /** Optional RV shim for compatibility with the entity registry. */
  rv?: RandomVariable;

  /**
   * random      - Fisher-Yates per completed entity; removes connection-order bias.
   * round-robin - declared connection order, rotating after each successful route.
   * ordered     - declared connection order every time; intentional priority.
   */
  outputRouting?: OutputRoutingPolicy;
}

interface QueuedItem {
  entity: AbstractMovingEntity<any>;
  remainingTime: number;
}

export class PerIndividualProcessor<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasComputedProperties<PerIndividualProcessor<S, T>>,
             HasInternalQueue<AbstractMovingEntity<any>> {

  ['isProcessor'] = true;

  /**
   * Internal storage as a parallel array - we want O(1) iteration and we
   * don't need fast removal-by-key (the LinkedQueue's selling point).
   */
  private items: QueuedItem[] = [];

  /** Public read-only view to satisfy HasInternalQueue. */
  queue = new LinkedQueue<AbstractMovingEntity<any>>();
  maxQueueSize: number = -1;

  opts: PerIndividualProcessorOpts;
  private readonly outputRouter: OutputConnectionRouter<EntityConnection<any, any>>;

  constructor(id: string, opts: PerIndividualProcessorOpts) {
    super(id);
    this.opts = opts;
    this.outputRouter = new OutputConnectionRouter(opts.outputRouting ?? 'random');
    // Register so doAudit + reg.getAllProcessors() still works against this type.
    reg.registerProcessor(this as any);
  }

  doValidation(): void { /* nothing to do */ }
  doValidationBeforeRun(): boolean { return true; }
  doSetupAfterInputConn(): boolean { return true; }
  doSetupAfterOutputConn(): boolean { return true; }

  isEmpty(): boolean { return this.items.length === 0; }
  isFull(): boolean  { return false; }

  takeItem(m: AbstractMovingEntity<any>): void {
    if (m instanceof ProcessableMovingEntity || (m as any).startNewStation) {
      (m as any).stationsVisitedCount = ((m as any).stationsVisitedCount ?? 0) + 1;
      (m as any).startNewStation?.(this.id);
      (m as any).addVisitedStation?.(this.id);
    }
    this.items.push({entity: m, remainingTime: this.opts.drawDuration()});
  }

  acceptItem(_m: AbstractMovingEntity<any>): boolean { return true; }

  getOutConnections() { return this.connectionsOut; }

  doAudit() {
    return {totalSize: this.items.length};
  }

  getWithComputedProperties(): this {
    return this;
  }

  getSerializableData(): Partial<this> {
    return {
      ...this,
      subscribers: undefined,
      subscribersByEvent: undefined,
    } as any;
  }

  getGraphData() {
    return {processedCount: this.items.length};
  }

  doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void {
    return this.runTimeStep(stepSize, opts);
  }

  runTimeStep(stepSize: math.BigNumber, _opts?: TimeStepOpts): void {
    this.timeStepCount++;
    const dt = Number(stepSize);

    // Decrement remaining times in place. Build a list of "ready to leave".
    const ready: QueuedItem[] = [];
    const stillWaiting: QueuedItem[] = [];
    for (const item of this.items) {
      item.remainingTime -= dt;
      if (item.remainingTime <= 0) ready.push(item);
      else stillWaiting.push(item);
    }
    this.items = stillWaiting;

    // Route everyone who's done. Pick the first accepting downstream target
    // according to the configured output-routing policy.
    for (const item of ready) {
      let routed = false;
      const connections = Array.from(this.connectionsOut) as EntityConnection<any, any>[];
      if (connections.length < 1) {
        console.warn(`[per-individual:${this.id}] entity ${(item.entity as any)?.id} finished service but station has no out-connections; it will loop back indefinitely.`);
      }
      for (const conn of this.outputRouter.order(connections)) {
        const target = (conn as EntityConnection<any, any>).getTarget();
        if (!target) continue;
        if (target.acceptItem(item.entity)) {
          this.outputRouter.markAccepted(connections, conn);
          target.takeItem(item.entity);
          routed = true;
          break;
        }
      }
      if (!routed) {
        // Nobody accepted - stick the item back at the front of the queue
        // with a tiny remaining time so it'll be retried next step.
        debugLog(() => `[per-individual:${this.id}] no downstream accepted finished entity ${(item.entity as any)?.id}; retrying next step (queue size ${this.items.length + 1}).`);
        item.remainingTime = 0;
        this.items.unshift(item);
      }
    }
  }
}
