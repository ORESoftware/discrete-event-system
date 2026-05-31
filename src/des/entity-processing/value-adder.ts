'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_processing/value_adder.rs
// - EntityNumericProcessor<S,T> becomes a small station struct with queue state
//   and ProcessorLike/QueueLike impls; DoesFanOut can stay as a composed helper
//   or PureTransform-style routing strategy.
// - Basic numeric reduction (`k.value + p.value`) should move into a typed
//   transform over BasicQuantityMovingEntity values instead of `any` casts.
// - Replace LinkedQueue, Symbol marker fields, util.inspect customization, and
//   thrown queue invariant errors with VecDeque/typed markers/Debug impls/Result.

import * as math from "mathjs";
import {QueueEntity, QueueEntityGraphData} from "../entity-queue/queue";
import {EntityGraphData, HasEntityValidation} from "../abstract/interfaces";
import {AbstractMovingEntity, BasicMovingEntity, BasicQuantityMovingEntity} from "../entity-moving/moving";
import {IterableInt} from "iterable.int";

import {
  bgn, DESMap,
  fisherYatesShuffle,
  getSortedHistogram,
  HasComputedProperties,
  makeError
} from "../general/general";
import {IsVoid, LinkedQueue} from '@oresoftware/linked-queue';
import {RandomVariable} from "../random-variables/rv";
import * as util from "util";
import {number, to} from "mathjs";
import {DoesFanOut} from "../abstract/composers";

const processorSymbol = Symbol('is-processor');

export const isProcessor = (v: any) => {
  return v && v[processorSymbol] === true;
}

export interface GraphData extends QueueEntityGraphData {

}


export class EntityNumericProcessor<S, T>
  extends QueueEntity<S, T>
  implements HasEntityValidation {

  [processorSymbol] = true;
  ['isProcessor'] = true;

  processedCount = 0;
  timeStepCount = 0;

  doesFanOut = new DoesFanOut<any>({entity:this});

  constructor(id: string) {
    super(id, {
      xx: true
    });


  }

  doValidation() {
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }


  getCleanVersion() {
    return {};
  }

  getWithComputedProperties(): any { // TODO: return specific type
    return Object.assign(this.getCleanVersion(), {
      queue: this.queue.getComputedProperties(),
    });
  }

  isEmpty(): boolean {
    return this.queue.size < 1;
  }

  isFull(): boolean {
    return false;
  }

  validate() {
    return true;
  }

  [util.inspect.custom](depth: number, options: any) {
    return this.getCleanVersion();
  }

  getGraphData(): any {
    return this;
    // return Object.assign(
    //   this.getWithComputedProperties(), {
    //     id: this.id,
    //     timeStepCount: this.timeStepCount,
    //     processedCount: this.processedCount
    //   });
  }

  getQueueSizes() {
    return {
      queue: this.queue.size,
    }
  }

  doAudit() {
    const queueSizes = this.getQueueSizes();
    return {
      queues: queueSizes,
      totalSize: Object.values(queueSizes).reduce((a, b) => a + b, 0)
    }
  }

  getSerializableData(): Partial<any> {
    return {
      ...this,
      subscribers: undefined,
      subscribersByEvent: undefined,
    }
  }

  toJSON() {
    return this.getSerializableData();
  }


  runTimeStep(stepSize: math.BigNumber) {

    this.timeStepCount++;
    this.sendUpdateToSubs('GRAPH_DATA:PROCESSING', this.getGraphData());

    if (this.queue.length < 2) {
      return;
    }

    if (this.queue.length > 4) {
      throw new Error('Queue length should never be')
    }

    // console.log('queue:',this.queue['head'], this.queue['tail']);

    const [k, v] = this.queue.dequeue();

    if (IsVoid.check(k)) {
      throw new Error('void item in front of queue.')
    }

    const [p] = this.queue.peek();

    if (IsVoid.check(p)) {
      throw makeError('queue item should not be void:', this.queue);
    }

    // console.log('k/p',k.value, p.value);
    const sum = k.value + p.value;
    // console.log({sum});
    const ame = new BasicQuantityMovingEntity(sum).init();

    // TODO: round-robin instead of fisher-yates shuffle?

    const {accepted} = this.doesFanOut.doFanOut(ame);

    if (!accepted) {
      throw makeError('moving entity was not accepted but it must be accepted:', ame);
    }


  }

  acceptItem(m: AbstractMovingEntity<any>) {
    return true;
  }

  takeItem(m: AbstractMovingEntity<any>) {
    m.stationsVisitedCount++;
    m.addVisitedStation(this.id);
    this.queue.enqueue(m);
  }

}

