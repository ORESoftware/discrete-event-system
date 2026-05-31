'use strict'

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-routing/entity-splitter.rs  (module des::entity_routing::entity_splitter)
// 1:1 file move. A node that BROADCASTS each queued item to every out-connection.
//
// Declarations → Rust:
//   interface DecisionEntityGraph (empty)  -> marker struct / `()`
//   class EntitySplitter<S,T>              -> struct + impl (+ impl AbstractBidirectionalEntity,
//                                             HasComputedProperties, HasInternalQueue)
//
// Conversion notes (file-specific):
//   - `queue: LinkedQueue` drained via `dequeueIterator()` yielding `[k,v]` ->
//     `VecDeque<_>` with `drain(..)`; no void sentinel.
//   - Broadcast semantics: EVERY out-connection must accept; a refusal does
//     `throw makeError(..)` -> `Result`/`panic!` (invariant). `(x as any)?.id` -> typed.
//   - `opts.replayItemsIfNotFirstAccepted?: false` is typed as the literal `false`
//     (quirk) -> a plain `bool` field in Rust.
//   - `getGraphData()` returns a hardcoded `{processedCount: 3}` stub -> placeholder.
//   - `math.BigNumber` stepSize -> decimal/f64; `getSerializableData(): Partial<this>` -> serde DTO.
//   - `getOutConnections(): Set<EntityConnection>` -> Vec/HashSet of `Rc<RefCell<..>>` edges.
// =============================================================================

import {AbstractBidirectionalEntity, TimeStepOpts} from "../abstract/abstract";
import { HasInternalQueue} from "../abstract/interfaces";
import {HasComputedProperties, makeError} from "../general/general";
import * as math from "mathjs";
import {TimeDelayEntityGraphData} from "../entity-travel/time-delay";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {RandomVariable} from "../random-variables/rv";
import {AbstractMovingEntity} from "../entity-moving/moving";


export interface DecisionEntityGraph {

}

export class EntitySplitter<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasComputedProperties<EntitySplitter<S, T>>, HasInternalQueue<AbstractMovingEntity<any>> {

  queue = new LinkedQueue<AbstractMovingEntity<any>>();

  opts: {
    xx?: boolean,
    replayItemsIfNotFirstAccepted?: false
  }

  maxQueueSize: number = -1;

  constructor(id: string, v: EntitySplitter<S, T>['opts']) {
    super(id);
    this.opts = Object.assign({}, v, {
      // defaults
    });
  }

  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m);
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }

  doValidationBeforeRun(): boolean {
    return true;
  }

  getWithComputedProperties(): EntitySplitter<S, T> {
    return this;
  }

  isEmpty(): boolean {
    return this.queue.length < 1;
  }

  isFull(): boolean {
    return false;
  }

  runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {

    this.timeStepCount++;

    if (this.getOutConnections().size < 1) {
      console.warn(`[splitter:${this.id}] has no out-connections; queued items cannot be broadcast downstream.`);
    }
    for(const [k,v] of this.queue.dequeueIterator()){
      // send each queue item to each out connection
      for(const conn of this.getOutConnections()){
        if(conn.target.acceptItem(k)){
          conn.target.takeItem(k);
        } else {
          console.warn(`[splitter:${this.id}] downstream "${(conn.target as any)?.id}" refused item ${(k as any)?.id}; splitter requires all targets to accept (broadcast semantics).`);
          throw makeError('must accept item:', k);
        }
      }
    }

  }

  getSerializableData(): Partial<this> {
    return this;
  }

  doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {
    return this.runTimeStep(stepSize, opts);
  }

  getGraphData(): TimeDelayEntityGraphData {
    // get final graph data vs. get data snapshot
    // only iterate though non-moving entities
    // add traveltime/timedelay module
    return {
      processedCount: 3
    };
  }

  doValidation(): void {
  }


}
