// RUST MIGRATION:
// - Target: src/des/signals/abstract.rs
// - SignalMarker becomes a zero-sized marker type or enum variant;
//   SignalTimeStepOpts and SignalEntityGraphData become typed structs rather
//   than symbol-indexed structural interfaces.
// - SignalEntity<E,V> should be a SignalEntity trait layered on MovingEntity,
//   with shared signal state composed into concrete structs.
// - Null/undefined placeholders (`return null as any`) need typed Option/Result
//   return contracts before the Rust port.

import {Entity, TimeStepOpts} from "../abstract/abstract";
import {EntityGraphData} from "../abstract/interfaces";
import * as math from 'mathjs';
import {AbstractMovingEntity} from "../entity-moving/moving";

export const SignalMarker = Symbol('signal')

export interface SignalTimeStepOpts extends TimeStepOpts{
  [SignalMarker]: true
}

export interface SignalEntityGraphData extends EntityGraphData {
  [SignalMarker]: true
}

export abstract class SignalEntity<E,V> extends AbstractMovingEntity<E,V> {

  doTimeStep(stepSize: math.BigNumber, opts?: SignalTimeStepOpts): void {
    return this.runTimeStep(stepSize, opts);
  }

  getGraphData(): SignalEntityGraphData {
    return null as any ;
  }

  getWithComputedProperties(): any {
    return null as any
  }

  abstract runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void;

  getSerializableData(): any {
    return undefined;
  }


}
