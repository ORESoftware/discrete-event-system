'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/differential.rs  (module des::signals::differential)
// 1:1 file move. A signal differentiator node (emits successive-sample differences).
//
// Declarations → Rust:
//   interface DifferentialTimeStepOpts -> struct (currently empty)
//   const marker (Symbol)              -> n/a
//   class Differentiator<E,V>          -> struct + impl (+ impl MultiDirectionalSignalEntity)
//
// Conversion notes (file-specific):
//   - `previousValue = <unknown>null as { [marker]: SignalValue }` uses a SYMBOL-KEYED
//     wrapper object to stash the previous sample -> just an `Option<SignalValue>` field;
//     the symbol-key wrapper is a TS hack with no Rust analogue.
//   - `(v as any).getValue()` dynamic access -> concrete `SignalValue` type.
//   - `queue.dequeue()` `[k,v]` + `IsVoid.check` + `console.error` -> `VecDeque` + `Option`
//     (no void sentinel); tracing instead of console.
//   - `stepSize` is unused (a true derivative divides the diff by dt) -> preserve current
//     behaviour and flag.
//   - `acceptItem(m: SignalValue)` here narrows the param vs the trait's
//     `acceptItem(m: AbstractMovingEntity)` -> reconcile the signature in the Rust trait.
//   - `math.subtract` -> ops; `runFinish()` `throw` -> `unimplemented!()`.
// =============================================================================

import {SignalEntity} from "./abstract";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import * as math from 'mathjs';
import {bgn} from "../general/general";
import {SignalValue} from "./signal-value";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import {IsVoid, LinkedQueue, LinkedQueueValue} from "@oresoftware/linked-queue";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {MultiDirectionalSignalEntity} from "./multi-directional-signal-entity";

export interface DifferentialTimeStepOpts extends TimeStepOpts {

}

const marker = Symbol('differential.marker');

export class Differentiator<E,V>
  extends MultiDirectionalSignalEntity<E,V>
  implements HasManyInputConnections<any, any>, HasManyOutputConnections<any, any> {


  runningTotal = bgn(0);
  previousValue = <unknown>null as { [marker]: SignalValue<E,V> };
  queue = new LinkedQueue<SignalValue<E,V>>();

  doValidationBeforeRun(): boolean {
    return true;
  }

  getValue(): V {
    return this.runningTotal as unknown as V;
  }

  runTimeStep(stepSize: math.BigNumber, opts?: DifferentialTimeStepOpts): void {

    while (this.queue.size > 0) {

      const [k, v] = this.queue.dequeue();

      if (IsVoid.check(k)) {
        console.error('void element in linked-queue:', {k, v});
        continue;
      }

      if (!this.previousValue) {
        this.previousValue = {[marker]: k};
        break;
      }

      const diff = math.subtract(
        (v as any).getValue(),
        this.previousValue[marker].getValue()
      );

      const newSignalVal = new SignalValue({val: diff});

      for (const c of this.connectionsOut) {
        if (c.target.acceptItem(newSignalVal)) {
          c.target.takeItem(newSignalVal);
        }
      }

    }

  }

  acceptItem(m: SignalValue<E,V>): boolean {
    // TODO: should reject items if full?
    return true;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m);
  }

  runFinish(): void {
    return;
  }

  doValidation() {
  }


}
