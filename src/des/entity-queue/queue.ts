'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_queue/queue.rs
// - QueueEntityGraphData becomes a serde-friendly struct; QueueEntity<S,T>
//   becomes a queue-station struct composing BidirectionalEntityState and
//   VecDeque/queue storage.
// - `HasInternalQueue<AbstractMovingEntity<any>>` should become a nominal trait
//   with an associated Item type, avoiding broad trait objects unless mixed
//   moving-entity queues are required.
// - Base getSerializableData now returns a minimal snapshot; Rust should derive
//   Serialize on a QueueEntitySnapshot rather than serializing graph pointers.

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
