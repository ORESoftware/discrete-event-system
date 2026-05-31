'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-queue/queue.rs  (module des::entity_queue::queue)
// 1:1 file move. The base buffering queue entity (parent of processors).
//
// Declarations → Rust:
//   interface QueueEntityGraphData -> struct QueueEntityGraphData { processed_count }
//                                     (composes EntityGraphData)
//   class QueueEntity<S,T>         -> struct + impl (+ impl AbstractBidirectionalEntity,
//                                     HasInternalQueue); this is a base others `extend`.
//
// Conversion notes (file-specific):
//   - INHERITANCE: EntityProcessor / EntityNumericProcessor `extend` QueueEntity ->
//     model as a `QueueEntity` trait (default fns) + shared field-bag struct that
//     processors compose; not `extends`.
//   - `queue: LinkedQueue<AbstractMovingEntity>` -> `VecDeque<_>` (entities behind
//     `Rc<RefCell>`/index).
//   - `getSerializableData()` logs then `throw makeError(..)` BEFORE its return
//     (dead unreachable return) -> `panic!`/`unimplemented!`; the spread DTO is unreachable.
//   - `opts: { xx?: boolean }` is a near-empty marker -> trim or a small config struct.
//   - `math.BigNumber` stepSize -> decimal/f64.
//   - `getGraphData()` returns hardcoded `processedCount: 3` stub -> placeholder.
// =============================================================================

import {number} from "mathjs";
import * as math from "mathjs";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {AbstractBidirectionalEntity} from "../abstract/abstract";
import {EntityGraphData, HasInternalQueue} from "../abstract/interfaces";
import {LinkedQueue} from "@oresoftware/linked-queue";

export interface QueueEntityGraphData extends EntityGraphData {
  processedCount: number;
}


export class QueueEntity<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasInternalQueue<AbstractMovingEntity<any>> {

  queue = new LinkedQueue<AbstractMovingEntity<any>>();

  opts: {
    xx?: boolean
  }


  maxQueueSize: number = -1;

  constructor(id: string, v: QueueEntity<S, T>['opts']) {
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

  isEmpty(): boolean {
    return this.queue.length < 1;
  }

  isFull(): boolean {
    return false;
  }

  runTimeStep(stepSize: math.BigNumber) {
    this.timeStepCount++;
    return;
  }

  getSerializableData(): Partial<this> {
    return {
      maxQueueSize: this.maxQueueSize,
      opts: undefined
    } as Partial<this>;
  }

  doTimeStep(stepSize: math.BigNumber) {
    return this.runTimeStep(stepSize);
  }

  getWithComputedProperties() {
    return Object.assign({}, {
      'queue.size': this.queue.size
    });
  }

  getGraphData(): QueueEntityGraphData {
    // get final graph data vs. get data snapshot
    // only iterate though non-moving entities
    // add traveltime/timedelay module
    return Object.assign(
      this.getWithComputedProperties(), {
        processedCount: 3
      });
  }

  doValidationBeforeRun(): boolean {
    return false;
  }

  doValidation() {
  }


}
