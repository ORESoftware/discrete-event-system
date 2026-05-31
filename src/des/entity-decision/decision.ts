'use strict'

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-decision/decision.rs  (module des::entity_decision::decision)
// 1:1 file move. Base decision/branching node (mostly a stub here).
//
// Declarations → Rust:
//   interface DecisionEntityGraph (empty) -> marker struct / `()`
//   class DecisionEntity<S,T>             -> struct + impl (+ impl AbstractBidirectionalEntity,
//                                            HasComputedProperties, HasInternalQueue)
//
// Conversion notes (file-specific):
//   - `DecisionEntityGraph` is an empty interface duplicated in
//     probability-decision.ts and binary-decision.ts — define ONCE in Rust.
//   - `getWithComputedProperties()` is `throw new Error("Method not implemented.")`
//     -> `unimplemented!()`.
//   - `queue: LinkedQueue<AbstractMovingEntity>` -> `VecDeque<_>` (entities behind Rc<RefCell>/index).
//   - `math.BigNumber` stepSize -> decimal/f64; `getGraphData()` is a hardcoded stub.
//   - `opts: { xx: boolean }` is a placeholder -> trim / small config struct.
// =============================================================================

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
    throw new Error("Method not implemented.");
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
