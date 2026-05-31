'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/abstract/interfaces.rs  (module des::abstract::interfaces)
// 1:1 file move. Defines the capability traits + graph-data shapes of the
// queueing-network entity model.
//
// Declarations → Rust:
//   interface IsObservable / HasOutput / HasInput / HasManyInputConnections /
//             HasManyOutputConnections / HasSingleInput/OutputConnection /
//             HasEntityValidation / HasInternalQueue / HasId
//                                  -> traits (these are BEHAVIORAL contracts)
//   interface EntityGraphData      -> struct EntityGraphData (empty marker; `()` or unit struct)
//   enum EventNames                -> enum EventNames
//
// Conversion notes (file-specific):
//   - These interfaces are heavily structural; in Rust every entity struct must
//     carry an EXPLICIT `impl HasOutput for X { .. }` etc. Group related traits
//     so a blanket impl is possible where behaviour is identical.
//   - `HasInput`/`HasOutput` reference EntityConnection<S,T> and
//     AbstractMovingEntity — model the graph edges with `Rc<RefCell<..>>` or an
//     arena/index (Vec + indices) to satisfy the borrow checker; raw mutual
//     references won't compile.
//   - `LinkedQueue<T>` (@oresoftware/linked-queue) -> std `VecDeque<T>`.
//   - Generic params <S, T> carry over directly as trait/struct generics.
//   - Methods returning `this` (builder style) -> return `&mut self` or `Self`.
// =============================================================================

import {AbstractMovingEntity} from "../entity-moving/moving";
import {number, string} from "mathjs";
import {EntityConnection, EntityObserver} from "./abstract";
import {LinkedQueue} from "@oresoftware/linked-queue";


export interface EntityGraphData {

}

type EmptyObject = Record<string, never>;


export interface HasEntityValidation {
  validate(): boolean;
}


export enum EventNames {
  FOO,
}


export interface IsObservable {
  subscribersByEvent: Map<string, Set<EntityObserver<any>>>
  subscribers: Set<EntityObserver<any>>;

  subscribeTo(name: string, o: EntityObserver<any>): this;

  subscribe(o: EntityObserver<any>): this;

  // TODO: observers only get updates every 5 timesteps, etc
  subscribeWithFrequency(count: number, o: EntityObserver<any>): this;

  unsubscribe(o: EntityObserver<any>): boolean; // returns true the Subject did have the Observer
  sendUpdateToSubs<T>(type: string, v: T): void;
}

export interface HasOutput<S, T> {
  id: string,
  addOutConnection(target: HasInput<S, T>): EntityConnection<S, T> | null;
  doSetupAfterInputConn(): boolean;

  notifyTargets(): void;
  doSetupAfterOutputConn(): boolean;
}

export interface HasSingleOutputConnection<S, T> extends HasOutput<S, T> {
  getOutConnection(): EntityConnection<S, T>;
}

export interface HasManyOutputConnections<S, T> extends HasOutput<S, T> {
  getOutConnections(): Set<EntityConnection<S, T>>;
}

export interface HasInternalQueue<T> {

  maxQueueSize: number;
  queue: LinkedQueue<T>
  isFull: () => boolean;
  isEmpty: () => boolean;

}

export interface HasId {
  id: string
}


export interface HasInput<S, T> {
  id: string,
  acceptItem(m: AbstractMovingEntity<any>): boolean

  takeItem(m: AbstractMovingEntity<any>): void;

  doSetupAfterInputConn(): boolean;

  notifySources(): void;

  doSetupAfterOutputConn(): boolean;

  addInConnection(source: HasManyOutputConnections<S, T>): EntityConnection<S, T> | null;

}

export interface HasSingleInputConnection<S, T> extends HasInput<S, T> {
  getInConnection(): EntityConnection<S, T>;
}

export interface HasManyInputConnections<S, T> extends HasInput<S, T> {
  getInConnections(): Set<EntityConnection<S, T>>;
}