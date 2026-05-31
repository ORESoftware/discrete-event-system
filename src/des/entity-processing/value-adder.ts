'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-processing/value-adder.rs  (module des::entity_processing::value_adder)
// 1:1 file move. A processor that pops two queued items and emits their numeric sum.
//
// Declarations → Rust:
//   const isProcessor (fn)                 -> free fn / trait-tag check
//   interface GraphData                    -> struct (composes QueueEntityGraphData)
//   class EntityNumericProcessor<S,T>      -> struct + impl (+ impl QueueEntity, HasEntityValidation)
//
// Conversion notes (file-specific):
//   - `[processorSymbol] = true` + `['isProcessor'] = true` brand + `isProcessor(v)`
//     -> a real trait/enum tag. NOTE: processorSymbol/isProcessor are ALSO defined
//     in processing.ts — unify to a single marker in Rust (don't define twice).
//   - `doesFanOut = new DoesFanOut({entity: this})` -> a composed field holding the
//     fan-out helper (see abstract/composers.rs); `this`-as-entity needs `Rc<RefCell>`/index.
//   - `LinkedQueue` `dequeue()`/`peek()` returning `[k,v]`/`[p]` + `IsVoid.check` ->
//     `VecDeque<_>` with `Option` returns; no void-sentinel.
//   - `k.value + p.value` is plain `number` addition here (NOT BigNumber) -> `f64`.
//   - `throw new Error(..)` / `throw makeError(..)` invariants -> `panic!`
//     (recoverable cases -> `Result`).
//   - `[util.inspect.custom]` -> `impl fmt::Debug`; `getSerializableData` spread -> serde DTO.
// =============================================================================

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


