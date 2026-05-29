'use strict';

import * as safe from "@oresoftware/safe-stringify";
import * as math from "mathjs";
import {AbstractMovingEntity} from "../entity-moving/moving";
import {
  EntityGraphData, HasInput,
  HasManyInputConnections,
  HasManyOutputConnections,
  HasOutput,
  IsObservable
} from "./interfaces";
import {DESSet, HasComputedProperties, makeError} from "../general/general";
import * as uuid from 'uuid';
import {RandomVariable} from "../random-variables/rv";

export abstract class EntityObserver<T extends Entity<any>> {
  abstract doUpdate(type: string, m: T): void
}

export interface IsSerializable<T> {

  getSerializableData(): T

  toJSON(): T

  serialize(): string

  serializePretty(): string
}

export abstract class Serializable<T> implements IsSerializable<any> {

  abstract getSerializableData(): T

  toJSON() {
    return this.getSerializableData();
  }

  serialize(): string {
    return JSON.stringify(this.getSerializableData());
    // return safes.stringify(this);
  }

  serializePretty(): string {
    return JSON.stringify(this.getSerializableData(), null, 2);
  }

  // serialize(): string {
  //     return safe.stringify(this.getSerializableData());
  // }
  //
  // serializePretty(): string {
  //     return JSON.stringify(
  //       JSON.parse(
  //         safe.stringify(this.getSerializableData())
  //       ),
  //       null,
  //       2
  //     );
  // }
}

export interface TimeStepOpts {
  isFinalTimeStep?: boolean
}

export abstract class Entity<E, V= any>
  extends Serializable<any>
  implements IsObservable, HasComputedProperties {

  subscribersByEvent = new Map()
  subscribers = new Set<EntityObserver<any>>();
  _shortUuid: string | null = null;

  id: string;
  timeStepCount = 0;

  constructor(id: string) {
    super();
    this.id = id;
  }

  get shortUuid(): string | null {
    return this._shortUuid;
  }

  abstract doValidation(): void;

  abstract doValidationBeforeRun() : boolean;

  getInitialGraphData() {
    return {
      initialGraphData: true
    }
  }

  setShortUuid(value: string) {
    if (this._shortUuid) {
      throw makeError('Should not be re-setting uuid on entity')
    }
    this._shortUuid = value;
  }

  abstract getWithComputedProperties(): any

  abstract getSerializableData(): Partial<this>;

  abstract doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void

  abstract runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void;

  abstract getGraphData(): EntityGraphData;

  subscribe(o: EntityObserver<any>): this {
    this.subscribers.add(o);
    return this;
  }

  subscribeTo(name: string, o: EntityObserver<any>): this {
    const s = this.subscribersByEvent.get(name);
    if (s) {
      s.add(o);
      return this;
    }

    this.subscribersByEvent.set(name, new Set([o]));
    return this;
  }

  unsubscribe(o: EntityObserver<any>): boolean {
    if (!this.subscribers.has(o)) {
      return false;
    }
    this.subscribers.delete(o);
    return true;
  }

  sendUpdateToSubs<T>(type: string, v: T): void {
    // TODO: subscribe by key/type, not for all keys/types
    for (const s of this.subscribers) {
      s.doUpdate(type, v);
    }
  }

  subscribeWithFrequency(count: number, o: EntityObserver<any>): this {
    return this;
  };

}




interface EntityConnectionGraphData extends EntityGraphData {

}

export class EntityConnection<S, T>
  extends Entity<any>
  implements HasComputedProperties<EntityConnection<S, T>> {  // S = Source, T = target
  //
  source: HasOutput<any, any> = <unknown>null as HasOutput<any, any>;
  target: HasInput<any, any> = <unknown>null as HasInput<any, any>;
  opts: {}

  constructor(source: HasOutput<any, any>, target: HasInput<any, any>, opts?: EntityConnection<S, T>['opts']) {
    super(uuid.v4().slice(-10));
    this.target = target;
    this.source = source;
    this.opts = Object.assign({}, opts, {
      // defaults
    });
  }

  getSerializableData(): Partial<this>{
    return {
      ...this,
      source: this.source.id,
      target: this.target.id,
      opts: undefined
    }
  }

  getWithComputedProperties() {
    return Object.assign({}, this);
  }

  getTarget() {
    return this.target;
  }

  doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {
    return this.runTimeStep(stepSize, opts);
  }

  getGraphData(): EntityConnectionGraphData {
    return {};
  }

  runTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts): void {
    // noop
    // console.log('running time step for EntityConnection');
  }

  subscribeWithFrequency(count: number, o: EntityObserver<any>): this {
    return this;
  }

  doValidationBeforeRun(): boolean {
    return false;
  }

  doValidation(): void {
  }

}


export interface HasNumericValue {
  value: number
}

export abstract class StationaryEntity<E> extends Entity<E> {
  // stationaryId: string

  abstract doSetupAfterInputConn(): boolean;

  abstract doSetupAfterOutputConn(): boolean;

}

export abstract class AbstractBidirectionalEntity<S, T>
  extends StationaryEntity<any>
  implements HasManyInputConnections<S, T>, HasManyOutputConnections<S, T> {

  connectionsIn = new DESSet<EntityConnection<S, T>>();
  connectionsOut = new DESSet<EntityConnection<S, T>>();

  protected constructor(id: string) {
    super(id);
  }

  addInConnection<S extends HasManyOutputConnections<any, this>>(source: S): EntityConnection<S, this> {
    const conn = new EntityConnection<S, this>(source, this);
    this.connectionsIn.add(conn);
    return conn;
  }

  addOutConnection<T>(target: HasManyInputConnections<this, any>): EntityConnection<this, T> {
    const conn = new EntityConnection<this, T>(this, target);
    this.connectionsOut.add(conn);
    return conn;
  }

  doTimeStep(stepSize: math.BigNumber, opts?: TimeStepOpts) {
    return this.runTimeStep(stepSize, opts);
  }

  getOutConnections() {
    return this.connectionsOut;
  }

  getInConnections() {
    return this.connectionsIn;
  }

  acceptItem(m: AbstractMovingEntity<any>) {
    // TODO: should reject items if full?
    return true;
  }

  abstract takeItem(m: AbstractMovingEntity<any>): void;

  abstract doSetupAfterInputConn(): boolean;

  abstract doSetupAfterOutputConn(): boolean

  notifySources(): void {
  }

  notifyTargets(): void {
  }
}