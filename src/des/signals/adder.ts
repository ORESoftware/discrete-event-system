'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/adder.rs  (module des::signals::adder)
// 1:1 file move. A signal node that sums incoming values and broadcasts the running total.
//
// Declarations → Rust:
//   interface IntegratorTimeStepOpts -> struct (NOTE: misnamed copy from integral.ts)
//   const marker (Symbol)            -> n/a (unused leftover)
//   class Adder<E,V>                 -> struct + impl (+ impl MultiDirectionalSignalEntity)
//
// Conversion notes (file-specific):
//   - `runningTotal: BigNumber` accumulator -> decimal/f64; `math.add` -> ops.
//   - `queue.dequeueIterator()` yielding `[d]` -> `VecDeque::drain`; reads `d.getValue()`.
//   - DUPLICATE LOGIC: this is identical to `Integrator` (integral.ts) — share one impl
//     in Rust. `IntegratorTimeStepOpts` + `marker` here are dead copy-paste -> drop.
//   - `getValue()/doValidation()/runFinish()` `throw` -> `unimplemented!()`.
//   - emits `new SignalValue({val: runningTotal})` then broadcasts to connectionsOut
//     (accept/take) -> borrow targets via `Rc<RefCell>` and release between calls.
// =============================================================================

import {SignalEntity} from "./abstract";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import * as math from 'mathjs';
import {bgn} from "../general/general";
import {SignalValue} from "./signal-value";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {MultiDirectionalSignalEntity} from "./multi-directional-signal-entity";
import * as uuid from 'uuid';

export interface IntegratorTimeStepOpts extends TimeStepOpts {

}

const marker = Symbol('integrator.marker');

export class Adder<E,V>
  extends MultiDirectionalSignalEntity<E,V>
  implements HasManyInputConnections<any, any>, HasManyOutputConnections<any, any> {


  runningTotal = bgn(0);

  doValidationBeforeRun(): boolean {
    return true;
  }

  getValue(): V {
    throw new Error("Method not implemented.");
  }
  doValidation(): void {
    throw new Error("Method not implemented.");
  }


  runTimeStep(stepSize: math.BigNumber, opts?: IntegratorTimeStepOpts): void {

    for (const [d] of this.queue.dequeueIterator()) {
      this.runningTotal = math.add(this.runningTotal, d.getValue());
    }

    const sv = new SignalValue({val: this.runningTotal});

    for (const v of this.connectionsOut) {
      if (v.target.acceptItem(sv)) {
        v.target.takeItem(sv);
      }
    }

  }

  acceptItem(m: AbstractMovingEntity<any>): boolean {
    // TODO: should reject items if full?
    return true;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m);
  }

  runFinish(): void {
    throw new Error('not yet implemented.');
  }


}