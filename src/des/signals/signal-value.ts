'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/signal-value.rs  (module des::signals::signal_value)
// 1:1 file move. The signal "sample" carried through the signal graph, plus 0/1 constants.
//
// Declarations → Rust:
//   abstract class AbstractSignalValue<E,V> -> trait + base (extends SignalEntity)
//   class SignalValue<E,V>                   -> struct SignalValue { value: Option<V> } + impl
//   class SignalValueUnity<E>                -> struct (value = 1) — specializes V = BigNumber
//   class SignalValueZero<E>                 -> struct (value = 0) — specializes V = BigNumber
//
// Conversion notes (file-specific):
//   - INHERITANCE chain SignalEntity -> AbstractSignalValue -> SignalValue ->
//     {Unity, Zero}. Unity/Zero specialize the generic `V` to `BigNumber` -> in Rust
//     monomorphize (drop the generic, or provide concrete constructors).
//   - `value = <unknown>null as V` placeholder + `math.isNumber(val)` guard ->
//     `Option<V>`; only set when numeric.
//   - `getMatrixValue(): Array<Array<BigNumber>>` -> `Vec<Vec<f64>>` (decimal/f64).
//   - `getShortUUID()` default id -> `uuid` crate.
//   - `runFinish()/doValidation()` `throw new Error('not implemented')` -> `unimplemented!()`.
// =============================================================================

import {SignalEntity, SignalTimeStepOpts} from "./abstract";
import {BigNumber} from "mathjs";
import {bgn, getShortUUID} from "../general/general";
import * as math from 'mathjs';
import {TimeStepOpts} from "../abstract/abstract";


export abstract class AbstractSignalValue<E,V> extends SignalEntity<E,V> {

  protected constructor(id: string) {
    super(id);
  }

  // abstract getMatrixValue(): Array<Array<BigNumber>>
  abstract getValue(): V

  runTimeStep(stepSize: math.BigNumber, opts?: SignalTimeStepOpts) {
    return this.doTimeStep(stepSize, opts);
  }

}

export class SignalValue<E,V> extends AbstractSignalValue<E,V> {

  value = <unknown>null as V;

  constructor({id, val}:{id?: string, val?: V}) {
    super(id || getShortUUID());
    if (math.isNumber(val)) {
      this.value = val;
    }
  }

  doValidation() {
  }

  doValidationBeforeRun(): boolean {
    return true;
  }

  getMatrixValue(): Array<Array<BigNumber>> {
    return [];
  }

  getValue(): V {
    return this.value;
  }

  setValue(m: V): void {
    this.value = m;
  }

  runFinish(): void {
    return;
  }

}


export class SignalValueUnity<E> extends SignalValue<E,math.BigNumber> {

  value = bgn(1);

  getValue(): BigNumber {
    return this.value;
  }

  setValue(m: math.BigNumber): void {
    this.value = m;
  }

  runFinish(): void {
    return;
  }

  doValidation(): void {
    return;
  }

}

export class SignalValueZero<E> extends SignalValue<E, math.BigNumber> {

  value = bgn(0);

  runFinish(): void {
    return;
  }

  doValidation(): void {
    return;
  }

}
