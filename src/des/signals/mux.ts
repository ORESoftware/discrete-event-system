'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/mux.rs  (module des::signals::mux)
// 1:1 file move. A signal multiplexer node (currently an unimplemented stub).
//
// Declarations → Rust:
//   interface MultiplexerTimeStepOpts -> struct (currently empty)
//   class Multiplexer<E,V>            -> struct + impl (+ impl MultiDirectionalSignalEntity)
//
// Conversion notes (file-specific):
//   - `runTimeStep` is empty, `acceptItem` returns false, `takeItem` is a no-op ->
//     a not-yet-implemented node; port the shape and leave `todo!()` bodies.
//   - `getValue()` returns `<unknown>undefined as V` -> `Option<V>::None`.
//   - `notifySources()/notifyTargets()/runFinish()` `throw` -> `unimplemented!()`.
//   - `runningTotal: BigNumber` is unused here -> decimal/f64 if kept.
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
    throw new Error("Method not implemented.");
  }

  notifyTargets(): void {
    throw new Error("Method not implemented.");
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
    return false;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
  }

  runFinish(): void {
    throw new Error('not implemented.');
  }


}