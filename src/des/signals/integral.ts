'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/integral.rs  (module des::signals::integral)
// 1:1 file move. A signal integrator node (accumulates incoming values).
//
// Declarations → Rust:
//   interface IntegratorTimeStepOpts -> struct (currently empty)
//   const marker (Symbol)            -> n/a (unused leftover)
//   class Integrator<E,V>            -> struct + impl (+ impl MultiDirectionalSignalEntity)
//
// Conversion notes (file-specific):
//   - DUPLICATE of `Adder` (adder.rs): same dequeue-and-sum body -> share one impl in Rust.
//   - It is a RUNNING SUM, NOT a true integral: `stepSize` is unused (a real integral
//     would multiply by dt). Preserve current behaviour; flag if dt-scaling is wanted.
//   - `runningTotal: BigNumber` -> decimal/f64; `math.add` -> ops.
//   - `queue.dequeueIterator()` -> `VecDeque::drain`; `getValue()` returns
//     `<unknown>undefined as any` -> `Option`/`None`.
//   - `runFinish()` `throw` -> `unimplemented!()`.
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

export class Integrator<E,V>
  extends MultiDirectionalSignalEntity<E,V>
  implements HasManyInputConnections<any, any>, HasManyOutputConnections<any, any> {

  getValue() {
     return <unknown>undefined as any;
  }

  runningTotal = bgn(0);

  doValidationBeforeRun(): boolean {
    return true;
  }

  doValidation() {
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