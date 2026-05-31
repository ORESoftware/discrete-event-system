'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_travel/time_delay.rs
// - TimeDelayEntityGraphData and DelayTimeStepOpts become structs;
//   TimeDelayOrTravelEntity<S,T> becomes a delay/travel station struct with
//   queue state and bidirectional endpoint trait impls.
// - Queue storage should be VecDeque with an associated MovingEntity item type;
//   RandomVariable should be a trait object or generic parameter depending on
//   how many concrete delay distributions are needed at runtime.
// - The current unimplemented validation/takeItem throws should become
//   explicit Result<_, TimeDelayError> paths before real behavior is filled in.

import {number} from "mathjs";
import * as math from "mathjs";
import {AbstractBidirectionalEntity, TimeStepOpts} from "../abstract/abstract";
import {HasComputedProperties} from "../general/general";
import {EntityGraphData, HasInternalQueue} from "../abstract/interfaces";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {RandomVariable} from "../random-variables/rv";
import { AbstractMovingEntity } from "../entity-moving/moving";


export interface TimeDelayEntityGraphData extends EntityGraphData {
    processedCount: number;
}

export interface DelayTimeStepOpts extends TimeStepOpts {

}


export class TimeDelayOrTravelEntity<S, T>
  extends AbstractBidirectionalEntity<S, T>
  implements HasComputedProperties<TimeDelayOrTravelEntity<S, T>>, HasInternalQueue<AbstractMovingEntity<any>> {


    doValidation(): void {
        return;
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
    rv: RandomVariable;
    queue = new LinkedQueue<AbstractMovingEntity<any>, AbstractMovingEntity<any>>();

    opts: {
        xx: boolean,
        rv: RandomVariable
    }

    maxQueueSize: number = -1;

    constructor(id: string, v: TimeDelayOrTravelEntity<S, T>['opts']) {
        super(id);
        this.opts = Object.assign({}, v, {
            // defaults
        });
        this.rv = v.rv;
    }

    doValidationBeforeRun(): boolean {
       return true;
    }

    getWithComputedProperties(): this {
        return Object.assign({}, this);
    }

    isEmpty(): boolean {
        return this.queue.length < 1;
    }

    isFull(): boolean {
        return false;
    }

    runTimeStep(stepSize: math.BigNumber, opts?: DelayTimeStepOpts) {
        return;
    }

    getSerializableData(): Partial<this> {
        return this;
    }

    doTimeStep(stepSize: math.BigNumber, opts?: DelayTimeStepOpts) {
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
