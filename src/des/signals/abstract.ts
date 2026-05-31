// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/abstract.rs  (module des::signals::abstract_)
// 1:1 file move. Base of the signal-processing entity family (signals flow as moving-entities).
// (`abstract` is a Rust keyword — name the module `abstract_` or `base`.)
//
// Declarations → Rust:
//   const SignalMarker (Symbol)                 -> n/a; use a marker trait / enum tag
//   interface SignalTimeStepOpts                -> struct (the `[SignalMarker]: true` brand
//                                                  becomes a trait impl, not a field)
//   interface SignalEntityGraphData             -> struct (same brand treatment)
//   abstract class SignalEntity<E,V>            -> trait SignalEntity (extends the MovingEntity
//                                                  trait) + base struct
//
// Conversion notes (file-specific):
//   - `SignalMarker = Symbol('signal')` is used as a COMPUTED PROPERTY KEY
//     (`[SignalMarker]: true`) to brand signal types -> Rust has no symbol keys;
//     model the brand as a marker trait or enum variant.
//   - `getGraphData()/getWithComputedProperties()` return `null as any` and
//     `getSerializableData()` returns `undefined` -> `Option<_>`/`None` or `unimplemented!()`.
//   - `extends AbstractMovingEntity` (inheritance) -> compose the moving-entity field-bag
//     + `impl` the MovingEntity trait; don't mirror `extends`.
//   - `math.BigNumber` stepSize -> decimal/f64.
// =============================================================================

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