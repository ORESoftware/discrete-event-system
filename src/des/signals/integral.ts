'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/integral.rs
// - IntegratorTimeStepOpts becomes a struct; Integrator<E,V> becomes a signal
//   transform node with MultiDirectionalSignalState and explicit numeric bounds.
// - The integration step is a PureTransform-style accumulation from queued
//   SignalValue inputs to a new SignalValue output.
// - Replace LinkedQueue, Symbol marker, broad AbstractMovingEntity<any> inputs,
//   and dynamic mathjs values with VecDeque, Option/Result, and typed signal
//   item/numeric traits.

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

  getValue(): V {
    return this.runningTotal as unknown as V;
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
    return;
  }


}
