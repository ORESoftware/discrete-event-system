'use strict';

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
      throw new Error("Method not implemented.");
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
    return false;
  }


  takeItem(m: AbstractMovingEntity<any>): void {
  }

  runFinish(): void {
    throw new Error('not implemented.');
  }



}