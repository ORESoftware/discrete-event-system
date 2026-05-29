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