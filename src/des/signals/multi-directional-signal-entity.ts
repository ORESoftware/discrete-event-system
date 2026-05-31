'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/signals/multi-directional-signal-entity.rs  (module des::signals::multi_directional_signal_entity)
// 1:1 file move. Signal node base: MANY inputs, MANY outputs (parent of Adder/Mux/…).
//
// Declarations → Rust:
//   abstract class MultiDirectionalSignalEntity<E,V> -> trait + base struct
//                 (+ impl HasInternalQueue, HasManyInputConnections, HasManyOutputConnections)
//
// Conversion notes (file-specific):
//   - This is the BASE that Adder / Multiplexer / Integrator / Differentiator /
//     SignalIncrementor `extend` -> a trait with defaults + composed field-bag struct.
//   - `maxQueueSize = <unknown>null as number` placeholder-null -> `Option<usize>`.
//   - `connectionsIn/Out: Set<EntityConnection>` -> `Vec`/`HashSet` of `Rc<RefCell<..>>` edges;
//     `queue: LinkedQueue<SignalValue>` -> `VecDeque<_>`.
//   - structural interfaces -> explicit trait `impl`s on each concrete signal node.
// =============================================================================

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



