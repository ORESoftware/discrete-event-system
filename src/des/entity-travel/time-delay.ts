'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-travel/time-delay.rs  (module des::entity_travel::time_delay)
// 1:1 file move. A travel/time-delay node (currently a mostly-unimplemented stub).
//
// Declarations → Rust:
//   interface TimeDelayEntityGraphData -> struct { processed_count } (composes EntityGraphData)
//   interface DelayTimeStepOpts        -> struct (extends TimeStepOpts; currently empty)
//   class TimeDelayOrTravelEntity<S,T> -> struct + impl (+ impl AbstractBidirectionalEntity,
//                                         HasComputedProperties, HasInternalQueue)
//
// Conversion notes (file-specific):
//   - `doValidation()`/`takeItem()` are `throw new Error("Method not implemented.")`
//     stubs -> `unimplemented!()`/`todo!()` in Rust.
//   - `queue: LinkedQueue<E, E>` (keyed by the entity itself) -> `VecDeque<_>`;
//     entities behind `Rc<RefCell>`/index.
//   - `rv: RandomVariable` must carry an injected RandomSource (no Math.random).
//   - `getWithComputedProperties(): this` returns `Object.assign({}, this)` (a shallow
//     clone typed as `this`) -> return a real computed DTO, not `Self`.
//   - `math.BigNumber` stepSize -> decimal/f64; `getGraphData()` is a hardcoded stub.
//   - `(m as any)?.id` dynamic access -> concrete entity type.
// =============================================================================

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
