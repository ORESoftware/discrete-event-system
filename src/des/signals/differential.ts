'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/differential.rs
// - DifferentialTimeStepOpts becomes a struct; Differentiator<E,V> becomes a
//   signal transform node struct with MultiDirectionalSignalState composed in.
// - The difference calculation is a PureTransform candidate:
//   (previous_value, next_value) -> SignalValue(diff), with Result for missing
//   or nonnumeric values.
// - Replace Symbol marker state, LinkedQueue, `any` casts, and console-error
//   void handling with Option<SignalValue>, VecDeque, typed numeric traits, and
//   Result/diagnostic events.

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
