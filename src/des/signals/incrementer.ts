'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/incrementer.rs
// - IncrementorTimeStepOpts becomes a struct; SignalIncrementor<E,V> becomes a
//   concrete signal transform node with composed multidirectional signal state.
// - Constructor currently passes `null as any` to the parent id; Rust should use
//   an explicit id parameter or generated id constructor.
// - Current runTimeStep is intentionally inert until increment semantics are
//   specified; port it as a typed Result/todo! method if behavior is still open.
// - Queue intake and runningTotal mirror the other signal transforms; in Rust
//   use VecDeque<SignalValue<E,V>> plus a numeric Decimal/trait bound.

import {SignalEntity} from "./abstract";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import * as math from 'mathjs';
import {bgn} from "../general/general";
import {SignalValue} from "./signal-value";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import { LinkedQueue } from "@oresoftware/linked-queue";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {MultiDirectionalSignalEntity} from "./multi-directional-signal-entity";

export interface IncrementorTimeStepOpts extends TimeStepOpts {

}

export class SignalIncrementor<E,V>
  extends MultiDirectionalSignalEntity<E,V>
  implements HasManyInputConnections<any,any>,HasManyOutputConnections<any, any>{

  constructor() {
    super(null as any);
  }

  getValue(): V {
    return this.runningTotal as unknown as V;
  }

  runningTotal = bgn(0);
  queue = new LinkedQueue<SignalValue<E,V>>();

  doValidationBeforeRun(): boolean {
    return true;
  }

  doValidation() {
  }

  runTimeStep(stepSize: math.BigNumber, opts?: IncrementorTimeStepOpts): void {

  }

  acceptItem(m: AbstractMovingEntity<any>): boolean {
    return true;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m as unknown as SignalValue<E,V>);
  }

  runFinish(): void {
    return;
  }



}
