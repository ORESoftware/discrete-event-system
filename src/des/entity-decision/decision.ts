'use strict'

// RUST MIGRATION:
// - Target: src/des/entity_decision/decision.rs
// - DecisionEntityGraph maps to a graph-data struct; DecisionEntity<S,T> maps to
//   a reusable decision-station struct plus trait impls for entity lifecycle,
//   bidirectional connections, queue ownership, and computed properties.
// - Replace `LinkedQueue<AbstractMovingEntity<any>>` and TS structural
//   interfaces with VecDeque plus nominal MovingEntity/Endpoint traits.
// - Unimplemented methods should become explicit `todo!()` during the first
//   port, then Result-returning trait methods once call sites are audited.

import {AbstractBidirectionalEntity, TimeStepOpts} from "../abstract/abstract";
import {HasInternalQueue} from "../abstract/interfaces";
import {HasComputedProperties} from "../general/general";
import * as math from "mathjs";
import {TimeDelayEntityGraphData} from "../entity-travel/time-delay";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {RandomVariable} from "../random-variables/rv";
import {AbstractMovingEntity} from "../entity-moving/moving";


export interface DecisionEntityGraph {

}

export class DecisionEntity<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasComputedProperties<DecisionEntity<S, T>>, HasInternalQueue<AbstractMovingEntity<any>> {

  queue = new LinkedQueue<AbstractMovingEntity<any>>();

  opts: {
    xx: boolean
  }

  maxQueueSize: number = -1;

  constructor(id: string, v: DecisionEntity<S, T>['opts']) {
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

  getWithComputedProperties(): DecisionEntity<S, T> {
    return Object.assign({}, this);
  }

  isEmpty(): boolean {
    return this.queue.length < 1;
  }

  isFull(): boolean {
    return false;
  }

  runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {
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

    doValidation(): void {
    }


}
