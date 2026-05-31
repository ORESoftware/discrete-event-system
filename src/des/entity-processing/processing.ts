'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-processing/processing.rs  (module des::entity_processing::processing)
// 1:1 file move. The multi-server queueing processor (input/processing/out stages).
//
// Declarations → Rust:
//   const isProcessor (fn)                    -> free fn or `dyn`-downcast helper
//   interface ProcessorEntityGraphData        -> struct (composes QueueEntityGraphData)
//   class EntityProcessor<S,T>                -> struct + impl (+ impl QueueEntity, HasEntityValidation)
//
// Conversion notes (file-specific):
//   - BRANDING: `[processorSymbol] = true` + `['isProcessor'] = true` and the
//     `isProcessor(v)` duck-type check -> use a real trait / enum variant tag,
//     NOT a symbol property. (processorSymbol is duplicated in value-adder.ts;
//     unify to one marker in Rust.)
//   - `math.BigNumber` accumulators (busy/idle time, histograms) -> decimal/f64;
//     `math.add/divide(...) as math.BigNumber` casts -> plain ops.
//   - `DESMap<number, math.BigNumber>` histograms -> `HashMap<usize, Decimal>`
//     (number key -> usize/i64); ordering for getSortedHistogram is N/A in Rust.
//   - `LinkedQueue` with keyed `remove(k)` and `[k,v]` iterators -> a keyed
//     structure (`IndexMap`/`HashMap` + `VecDeque`); std VecDeque has no remove-by-key.
//   - `[util.inspect.custom]` -> `impl fmt::Debug`.
//   - `getSerializableData(): Partial<any>` spreads `...this` then nulls fields ->
//     a `#[derive(Serialize)]` DTO with `#[serde(skip)]` for subscribers.
//   - `throw 'wtf'` (bare-string throw) and `throw makeError/new Error(..)` in the
//     hot loop -> `panic!` for invariant violations; `Result` where recoverable.
//   - `rv.getNextEventQuantity(stepSize)` -> RandomSource injection (no Math.random).
//   - `getWithComputedProperties(): any` -> a concrete computed-stats DTO.
//   - `outputRouter.order/markAccepted` -> see entity-routing/output-routing-policy.
//   - `bumpHistogram_old` is legacy/dead (Map<number,Set<string>>) — likely drop.
//   - console.warn/debug -> `tracing` macros.
// =============================================================================

import * as math from "mathjs";
import {QueueEntity, QueueEntityGraphData} from "../entity-queue/queue";
import {EntityGraphData, HasEntityValidation} from "../abstract/interfaces";
import {AbstractMovingEntity, BasicMovingEntity, ProcessableMovingEntity} from "../entity-moving/moving";
import {IterableInt} from "iterable.int";
import {
  bgn, DESMap,
  getSortedHistogram,
  getSortedTimeHistogram,
  HasComputedProperties,
  makeError
} from "../general/general";
import {IsVoid, LinkedQueue} from '@oresoftware/linked-queue';
import {RandomVariable} from "../random-variables/rv";
import * as util from "util";
import {number, to} from "mathjs";
import * as des from '../general/time-accrued'
import {reg} from "../general/entity-registration";
import {OutputConnectionRouter, OutputRoutingPolicy} from "../entity-routing/output-routing-policy";

const processorSymbol = Symbol('is-processor');

export const isProcessor = (v: any) => {
  return v && v[processorSymbol] === true;
}

export interface ProcessorEntityGraphData extends QueueEntityGraphData {

}


export class EntityProcessor<S, T>
  extends QueueEntity<S, T>
  implements HasEntityValidation {

  [processorSymbol] = true;
  ['isProcessor'] = true;

  concurrency = 5;
  inputQueue = this.queue;
  processingQueue = new LinkedQueue<ProcessableMovingEntity<any>>();
  outQueue = new LinkedQueue<ProcessableMovingEntity<any>>();
  totalServerBusyTime = bgn(0);
  totalServerIdleTime = bgn(0);
  processedCount = 0;
  timeStepCount = 0;
  inputQueueHistogram = new DESMap<number, math.BigNumber>();
  processingQueueHistogram = new DESMap<number, math.BigNumber>();
  outputQueueHistogram = new DESMap<number, math.BigNumber>();

  // TODO: these should be a binary tree where the max depth is ~ 12 so 2^12 total items in tree
  // https://stackoverflow.com/questions/2942517/how-do-i-iterate-over-binary-tree
  // inputQueueTimes = new LinkedQueue();
  // processingQueueTimes = new LinkedQueue();
  // outputQueueTimes = new LinkedQueue();

  inputQueueTimeHistogram = new DESMap<number, math.BigNumber>();
  processingQueueTimeHistogram = new DESMap<number, math.BigNumber>();
  outputQueueTimeHistogram = new DESMap<number, math.BigNumber>();

  // inputQueueTimeHistogram = new Map<number, Set<string>>();
  // processingQueueTimeHistogram = new Map<number, Set<string>>();
  // outputQueueTimeHistogram = new Map<number, Set<string>>();


  opts: {
    xx?: true,
    rv: RandomVariable,
    /**
     * How this processor chooses among downstream acceptors when an item can
     * route to exactly one out-connection.
     *
     * random      - Fisher-Yates per item; removes insertion-order bias.
     * round-robin - declared connection order, rotating after each success.
     * ordered     - declared connection order every time; intentional priority.
     */
    outputRouting?: OutputRoutingPolicy
  }

  rv: RandomVariable;
  private readonly outputRouter: OutputConnectionRouter<any>;

  constructor(id: string, opts: EntityProcessor<S, T>['opts']) {
    super(id, {
      xx: true
    });
    this.rv = opts.rv;
    this.opts = Object.assign({}, opts);
    this.outputRouter = new OutputConnectionRouter(opts.outputRouting ?? 'random');
    reg.registerProcessor(this);
  }

  doValidation() {
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }

  getServerUtilization(): math.BigNumber {
    return math.divide(
      this.totalServerBusyTime,
      math.add(this.totalServerBusyTime, this.totalServerIdleTime)
    ) as math.BigNumber
  }

  addToInputHistogram(timeStep: math.BigNumber) {
    const size = this.inputQueue.size;
    if (!this.inputQueueHistogram.has(size)) {
      this.inputQueueHistogram.set(size, bgn(0));
    }
    const val = this.inputQueueHistogram.get(size) as math.BigNumber;
    this.inputQueueHistogram.set(size, math.add(val, timeStep) as math.BigNumber);
  }

  addToProcessingHistogram(timeStep: math.BigNumber) {
    const size = this.processingQueue.size;
    if (!this.processingQueueHistogram.has(size)) {
      this.processingQueueHistogram.set(size, bgn(0));
    }
    const val = this.processingQueueHistogram.get(size) as math.BigNumber;
    this.processingQueueHistogram.set(size, math.add(val, timeStep) as math.BigNumber);
  }

  addToOutputHistogram(timeStep: math.BigNumber) {
    const size = this.outQueue.size;
    if (!this.outputQueueHistogram.has(size)) {
      this.outputQueueHistogram.set(size, bgn(0));
    }
    const val = this.outputQueueHistogram.get(size) as math.BigNumber;
    this.outputQueueHistogram.set(size, math.add(val, timeStep) as math.BigNumber);
  }


  getWithComputedProperties(): any { // TODO: return specific type
    return Object.assign(this.getCleanVersion(), {
      serverUtilization: Number(this.getServerUtilization().toFixed(6)),
      queue: this.queue.getComputedProperties(),
      inputQueue: this.inputQueue.getComputedProperties(),
      processingQueue: this.processingQueue.getComputedProperties(),
      outQueue: this.outQueue.getComputedProperties(),
      inputQueueHistogram: getSortedHistogram(this.inputQueueHistogram),
      processingQueueHistogram: getSortedHistogram(this.processingQueueHistogram),
      outputQueueHistogram: getSortedHistogram(this.outputQueueHistogram),
      //

      inputQueueTimeHistogram: getSortedTimeHistogram(this.inputQueueTimeHistogram),
      processingQueueTimeHistogram: getSortedTimeHistogram(this.processingQueueTimeHistogram),
      outputQueueTimeHistogram: getSortedTimeHistogram(this.outputQueueTimeHistogram),
    });
  }

  bumpTotalServerBusyTime(stepSize: math.BigNumber) {
    this.totalServerBusyTime = math.add(this.totalServerBusyTime, stepSize);
  }

  bumpTotalServerIdleTime(stepSize: math.BigNumber) {
    this.totalServerIdleTime = math.add(this.totalServerIdleTime, stepSize);
  }


  isEmpty(): boolean {
    return false;
  }

  isFull(): boolean {
    return false;
  }

  validate() {
    return true;
  }

  getCleanVersion() {
    return Object.assign({}, this, {
      subscribers: {
        size: this.subscribers.size
      },
      connectionsOut: {
        size: this.connectionsOut.size
      },
      connectionsIn: {
        size: this.connectionsIn.size
      },
      inputQueue: {
        size: this.inputQueue.length
      },
      outQueue: {
        size: this.outQueue.length
      },
      processingQueue: {
        size: this.processingQueue.length
      },
      queue: {
        size: this.queue.length
      },
      inputQueueHistogram: getSortedHistogram(this.inputQueueHistogram)
    });
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
      inputQueue: this.inputQueue.size,
      outQueue: this.outQueue.size,
      processingQueue: this.processingQueue.size
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

  bumpHistogram(key: number, m: DESMap<number, math.BigNumber>) {

    if (!m.has(key)) {
      m.set(key, bgn(1));
      return;
    }

    m.set(
      key,
      math.add(bgn(1), m.get(key) as math.BigNumber)
    );

  }

  bumpHistogram_old(key: number, x: { id: string }, m: Map<number, Set<string>>) {

    const id = x.id;

    if(!x.id){
      throw new Error('boof');
    }

    // for(const v of (x as any).processingTimeByStation){
    //    console.log(v);
    // }

    if (!m.has(key)) {
      m.set(key, new Set([id]));
      return;
    }

    const s = <Set<string>>m.get(key);
    if(s.has(id)){
      throw 'wtf'
    }
    s.add(id);

  }

  getKeyForHistogram(t: math.BigNumber) {
    // t is time (t is chunks of time step sizes), aka, some multiple of the time step size
    return Math.floor(Number(math.divide(math.add(t, bgn(1)), des.getStepSize())));
  }


  runTimeStep(stepSize: math.BigNumber) {

    this.timeStepCount++;
    this.addToInputHistogram(stepSize);
    this.addToProcessingHistogram(stepSize);
    this.addToOutputHistogram(stepSize);

    this.sendUpdateToSubs('GRAPH_DATA:PROCESSING', this.getGraphData());

    const s = this.concurrency - this.processingQueue.size;

    for (const z of new IterableInt(0, s)) {
      // represents idle servers
      this.bumpTotalServerIdleTime(stepSize);
    }

    for (const z of new IterableInt(0, this.processingQueue.size)) {
      this.bumpTotalServerBusyTime(stepSize);
    }

    for (let [k, v] of this.outQueue.iterator()) {

      // we need to try to flush the outQueue

      v.bumpTotalWaitTime(stepSize);
      v.bumpTimeInSystem(stepSize);
      v.bumpOutQueueWaitTime(stepSize);

      const connections = Array.from(this.getOutConnections());
      if (connections.length < 1) {
        console.warn(`[processor:${this.id}] outQueue flush: entity has no out-connections; item will remain stuck in outQueue (size=${this.outQueue.size}).`);
      }
      for (const conn of this.outputRouter.order(connections)) {
        const target = conn.getTarget();
        if (!target) {
          console.warn(`[processor:${this.id}] outQueue flush: out-connection has no resolvable target; skipping connection.`);
          continue;
        }

        if (target.acceptItem(v)) {
          this.outputRouter.markAccepted(connections, conn);
          const size = this.outQueue.size;
          this.outQueue.remove(k);
          if (this.outQueue.size !== (size - 1)) {
            console.warn(`[processor:${this.id}] outQueue.remove did not shrink queue by 1 (before=${size}, after=${this.outQueue.size}) — possible duplicate/missing key.`);
            throw makeError('queue size should be one smaller.');
          }

          {
            const t = v.setTimeInOutputQueue(this.id);
            const key = this.getKeyForHistogram(t);
            // console.log('k/v',k, v);
            this.bumpHistogram(key,  this.outputQueueTimeHistogram);
          }

          target.takeItem(v as AbstractMovingEntity<any>);
          break;
        }
      }

    }

    for (let [v] of this.processingQueue.iterator()) {
      v.bumpTimeInSystem(stepSize);
      v.bumpTotalProcessingTime(stepSize);
    }

    const evq = this.rv.getNextEventQuantity(stepSize);
    console.debug(`[processor:${this.id}] event quantity (service completions) this step: ${evq}; processingQueue.size=${this.processingQueue.size}`);

    for (let i = 0; i < evq; i++) {

      const [next,v] = this.processingQueue.dequeue();

      if (IsVoid.check(next)) {
        if (this.processingQueue.size > 0) {
          console.warn(`[processor:${this.id}] dequeue returned void but processingQueue.size=${this.processingQueue.size} (>0) — queue invariant violated at completion ${i}/${evq}.`);
          throw makeError('warning, non-empty queue, but zeroth item (peek) was falsy.');
        }
        console.debug(`[processor:${this.id}] no more items to service this step (drew ${evq} completions, processed ${i}).`);
        break;
      }

      {
        const t = next.setTimeInProcessingQueue(this.id);
        const key = this.getKeyForHistogram(t);
        // console.log('k/v',next, v);
        this.bumpHistogram(key,this.processingQueueTimeHistogram);
      }

      this.processedCount++;

      const connections = Array.from(this.getOutConnections());
      if (connections.length < 1) {
        console.warn(`[processor:${this.id}] processed item but has zero out-connections — item ${(next as any)?.id} will be dropped into outQueue with nowhere to go.`);
      }
      for (const conn of this.outputRouter.order(connections)) {
        const target = conn.getTarget();
        if (!target) {
          console.warn(`[processor:${this.id}] processed item: out-connection has no resolvable target; skipping connection.`);
          continue;
        }
        if (target.acceptItem(<AbstractMovingEntity<any>>next)) {
          this.outputRouter.markAccepted(connections, conn);
          target.takeItem(<AbstractMovingEntity<any>>next);
          break;
        }

        // if not takers, we add to out queue
        console.debug(`[processor:${this.id}] downstream "${(target as any)?.id}" rejected item; buffering in outQueue (size now ${this.outQueue.size + 1}).`);
        next.setStartTimeInOutputQueue(this.id);
        this.outQueue.enqueue(next);
        break;
      }
    }

    for (let [z,v] of this.inputQueue) {
      z.bumpTimeInSystem(stepSize);
      z.bumpTotalWaitTime(stepSize);
      if (this.processingQueue.size < this.concurrency) {

        {
          const t = z.setTimeInInputQueue(this.id);
          const key = this.getKeyForHistogram(t);
          // console.log('k/v',z, v);
          this.bumpHistogram(key, this.inputQueueTimeHistogram);
        }

        this.inputQueue.remove(z);
        this.processingQueue.enqueue(z);
        z.setStartTimeInProcessQueue(this.id);
      }
    }

  }

  acceptItem(m: AbstractMovingEntity<any>) {
    return true;
  }

  takeItem(m: ProcessableMovingEntity<any>) {
    m.stationsVisitedCount++;
    m.startNewStation(this.id);
    m.addVisitedStation(this.id);

    // Option A: skip the inputQueue entirely when there is processing
    // capacity. The 1-step inputQueue lag was the dominant timing-granularity
    // artifact: items arriving via takeItem used to wait for the end-of-
    // runTimeStep input->processing transfer, then for the next step's evq.
    // Eager promotion lets evq see the new item in the SAME step (when this
    // station's runTimeStep runs after the upstream station's), and removes
    // an unnecessary one-step delay otherwise.
    //
    // Invariants preserved:
    //   - concurrency limit is respected; we fall through to inputQueue when
    //     processingQueue is full.
    //   - histogram counters get a 0-sample for inputQueue time, keeping the
    //     "every visit recorded" invariant intact.
    //   - moving-entity timestamp asserts (-1 sentinel) hold because
    //     startNewStation() above just reset them.
    //   - mass conservation is unchanged: an item moves from one bucket to
    //     another, never duplicated or dropped.
    if (this.processingQueue.size < this.concurrency) {
      const t = m.setTimeInInputQueue(this.id);
      const key = this.getKeyForHistogram(t);
      this.bumpHistogram(key, this.inputQueueTimeHistogram);
      this.processingQueue.enqueue(m);
      m.setStartTimeInProcessQueue(this.id);
      return;
    }

    // Overflow path: all servers busy. End-of-runTimeStep input->processing
    // loop will drain the inputQueue when capacity opens up - same behavior
    // as before Option A.
    console.debug(`[processor:${this.id}] all ${this.concurrency} servers busy; item ${(m as any)?.id} overflows to inputQueue (size now ${this.queue.size + 1}).`);
    this.queue.enqueue(m);
  }

}
