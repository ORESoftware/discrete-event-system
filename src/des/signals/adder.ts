'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/adder.rs
// - IntegratorTimeStepOpts should be renamed or split from integral.rs if this
//   remains an adder-specific option struct; Adder<E,V> becomes a signal
//   transform node with composed multidirectional state.
// - The accumulation loop is a PureTransform over queued signal values and
//   running total; use a numeric trait/Decimal alias instead of mathjs dynamic
//   arithmetic.
// - Replace `any` moving-entity accepts, LinkedQueue storage, Symbol marker, and
//   thrown unimplemented methods with typed signal items and Result/todo! stubs.

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
    return this.runningTotal as unknown as V;
  }
  doValidation(): void {
    return;
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
    return;
  }


}
