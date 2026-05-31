'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/signal_value.rs
// - AbstractSignalValue, SignalValue, SignalValueUnity, and SignalValueZero map
//   to a SignalValueLike trait plus concrete structs/constructors for generic,
//   unity, and zero values.
// - `math.BigNumber` signal values need the shared Decimal alias; generic V
//   should have trait bounds for Clone/Debug/Serialize only where required.
// - Replace TS constructor object bags and nullable casts with builder structs,
//   typed defaults, and Result-returning validation where construction can fail.

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
