'use strict';

// RUST MIGRATION:
// - Target: src/des/signals/multi_directional_signal_entity.rs
// - MultiDirectionalSignalEntity<E,V> becomes reusable signal connection state
//   plus trait impls for SignalEntity, HasManyInput, HasManyOutput, and
//   HasInternalQueue with associated SignalValue item type.
// - `maxQueueSize = null as number` should become Option<usize> or a bounded
//   queue policy; LinkedQueue maps to VecDeque unless key removal is required.
// - Abstract accept/take hooks should return Result or explicit acceptance
//   enums when Rust callers need backpressure details.

import {SignalEntity} from "./abstract";
import {HasManyInputConnections, HasInternalQueue, HasManyOutputConnections} from "../abstract/interfaces";
import * as math from "mathjs";
import {EntityConnection, TimeStepOpts} from "../abstract/abstract";
import { AbstractMovingEntity } from "../entity-moving/moving";
import {LinkedQueue} from "@oresoftware/linked-queue";
import {SignalValue} from "./signal-value";


export abstract class MultiDirectionalSignalEntity<E,V>
  extends SignalEntity<E,V>
  implements HasInternalQueue<any>, HasManyInputConnections<any, any>, HasManyOutputConnections<any, any> {

  maxQueueSize = <unknown>null as number;
  queue = new LinkedQueue<SignalValue<E,V>>();
  connectionsIn = new Set<EntityConnection<any, any>>();
  connectionsOut = new Set<EntityConnection<any, any>>();

  abstract acceptItem(m: AbstractMovingEntity<any>): boolean;
  abstract takeItem(m: AbstractMovingEntity<any>): void;

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
    //TODO: fill-in
  }


}


