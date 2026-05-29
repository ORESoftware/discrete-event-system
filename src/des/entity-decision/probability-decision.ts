'use strict'

import {AbstractBidirectionalEntity, EntityConnection, TimeStepOpts} from "../abstract/abstract";
import {HasInternalQueue} from "../abstract/interfaces";
import {bgn, HasComputedProperties, makeError} from "../general/general";
import * as math from "mathjs";
import {TimeDelayEntityGraphData} from "../entity-travel/time-delay";
import {IsVoid, LinkedQueue} from "@oresoftware/linked-queue";
import {RandomVariable} from "../random-variables/rv";
import {AbstractMovingEntity, BasicMovingEntity} from "../entity-moving/moving";
import {reg} from "../general/entity-registration";


export interface DecisionEntityGraph {

}

export class ProbabilityDecisionEntity<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasComputedProperties<ProbabilityDecisionEntity<S, T>>, HasInternalQueue<AbstractMovingEntity<any>> {


  queue = new LinkedQueue<AbstractMovingEntity<any>>();

  connectionsOutByIndex = new Map<number, EntityConnection<any, any>>();
  connectionsInByIndex= new Map<number, EntityConnection<any, any>>();
  rv: RandomVariable;

  opts: {
    rv: RandomVariable,
    probabilities: Array<{
      index: number,
      prob: math.BigNumber
    }>
  }

  maxQueueSize: number = -1;

  constructor(id: string, v: ProbabilityDecisionEntity<S, T>['opts']) {
    super(id);
    this.opts = Object.assign({}, v, {
      // defaults
    });

    reg.registerDecision(this);

    this.rv = v.rv;

    let checkSum = bgn(0);

    for (const v of this.opts.probabilities) {
      checkSum = math.add(checkSum, v.prob);
    }

    if (math.larger(checkSum, bgn(1.00001))) {
      throw new Error('probability sum too high')
    }

    if (math.smaller(checkSum, bgn(0.9999))) {
      throw new Error('probability sum too high')
    }

  }

  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m);
  }

  getWithComputedProperties(): ProbabilityDecisionEntity<S, T> {
    return Object.assign({}, this);
  }

  isEmpty(): boolean {
    return this.queue.length < 1;
  }

  isFull(): boolean {
    return false;
  }

  doValidation(): void {
    throw new Error("Method not implemented.");
  }

  doValidationBeforeRun() {
    if (this.connectionsOut.size !== this.opts.probabilities.length) {
      throw makeError('connections out size must be the same size as probabilities.')
    }
    return true;
  }

  doSetupAfterInputConn(): boolean {
    let index = -1;
    for (const v of this.connectionsOut) {
      index++;
      this.connectionsOutByIndex.set(index, v);
    }
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    let index = -1;
    for (const v of this.connectionsIn) {
      index++;
      this.connectionsInByIndex.set(index, v);
    }
    return true;
  }

  doAudit(){
    return {
      totalSize: this.queue.size
    }
  }

  runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {

    const rejected: Array<BasicMovingEntity> = [];

    for (const [k, v] of this.queue.dequeueIterator()) {

      if(IsVoid.check(k)){
        console.error('void value from queue:', {k,v}, (this.queue as any).head, (this.queue as any).tail);
        process.exit(0);
        break;
      }

      const r = bgn(Math.random());

      let sum = bgn(0);
      let index = -1;
      for (const v of this.opts.probabilities) {
        index++;
        sum = math.sum(bgn(v.prob), sum);
        if (math.smaller(r, sum)) {
          break;
        }
      }

      const outConn = this.connectionsOutByIndex.get(index);

      if (!outConn) {
        console.log(this.opts);
        console.log(this, this.connectionsOutByIndex);
        throw makeError(`missing connection with index:`, index);
      }

      if (outConn.target.acceptItem(v)) {
        outConn.target.takeItem(v);
      } else {
        rejected.push(v);
      }
    }

    for (const v of rejected) {
      this.queue.enqueue(v);
    }

    return;
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


}
