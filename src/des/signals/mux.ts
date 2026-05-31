'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/mux.rs
// - MultiplexerTimeStepOpts becomes a struct; Multiplexer<E,V> becomes a signal
//   routing/transform node with composed multidirectional signal state.
// - Multiplexing should be modeled as a PureTransform once selection semantics
//   are known: queued inputs -> selected SignalValue output(s).
// - Current selection/runTimeStep behavior is intentionally open; port the
//   method as a typed Result/todo! if mux semantics are still unspecified.
// - Queue intake mirrors other signal transforms; Rust should use
//   VecDeque<SignalValue<E,V>> and explicit event/notification traits.

import {SignalEntity} from "./abstract";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import * as math from 'mathjs';
import {bgn} from "../general/general";
import {SignalValue} from "./signal-value";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {MultiDirectionalSignalEntity} from "./multi-directional-signal-entity";

export interface MultiplexerTimeStepOpts extends TimeStepOpts {

}

export class Multiplexer<E,V>
  extends MultiDirectionalSignalEntity<E,V>
  implements HasManyInputConnections<any, any>, HasManyOutputConnections<any, any> {

  runningTotal = bgn(0);
  queue = new LinkedQueue<SignalValue<E,V>>();


  doValidation() {
  }

  getValue(): V {
    return <unknown>undefined as V;
  }

  notifySources(): void {
    return;
  }

  notifyTargets(): void {
    return;
  }

  doValidationBeforeRun(): boolean {
    return true;
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }

  runTimeStep(stepSize: math.BigNumber, opts?: MultiplexerTimeStepOpts): void {


  }

  acceptItem(m: AbstractMovingEntity<any>): boolean {
    // TODO: should reject items if full?
    return true;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
    this.queue.enqueue(m as unknown as SignalValue<E,V>);
  }

  runFinish(): void {
    return;
  }


}
