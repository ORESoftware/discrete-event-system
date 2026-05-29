import {SignalEntity} from "./abstract";
import {
  HasManyInputConnections,
  HasInternalQueue,
  HasManyOutputConnections,
  HasSingleInputConnection, HasSingleOutputConnection, HasOutput, HasInput
} from "../abstract/interfaces";
import * as math from "mathjs";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import { AbstractMovingEntity } from "../entity-moving/moving";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {SignalValue} from "./signal-value";


export abstract class SingleInManyOutSignalEntity<E,V>
  extends SignalEntity<E,V>
  implements HasInternalQueue<any>, HasSingleInputConnection<any, any>, HasManyOutputConnections<any, any> {

  maxQueueSize = <unknown>null as number;
  queue = new LinkedQueue<SignalValue<E,V>>();
  connectionIn = <unknown>null as EntityConnection<any,any>;
  connectionsOut = new Set<EntityConnection<any, any>>();

  abstract acceptItem(m: AbstractMovingEntity<any>): boolean;
  abstract takeItem(m: AbstractMovingEntity<any>): void;

  addInConnection<S extends HasOutput<any, this>>(source: S): EntityConnection<S, this> {
    const conn = new EntityConnection<S, this>(source, this);
    return this.connectionIn = conn;
  }

  addOutConnection<T>(target: HasInput<this, any>): EntityConnection<this, T> {
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

  getInConnection() {
    return this.connectionIn;
  }

  isEmpty(): boolean {
    return this.queue.size < 1;
  }

  isFull(): boolean {
    return this.queue.size >= this.maxQueueSize;
  }

  doSetupAfterInputConn(): boolean {
    return false;
  }

  doSetupAfterOutputConn(): boolean {
    return false;
  }

  notifySources(): void {
    //TODO: fill-in
  }

  notifyTargets(): void {
  }


}




