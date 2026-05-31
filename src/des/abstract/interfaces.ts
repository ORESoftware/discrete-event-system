'use strict';

// RUST MIGRATION:
// - Target: src/des/abstract/interfaces.rs
// - Keep this as the trait-contract companion to abstract.rs. TypeScript
//   structural interfaces must become nominal Rust traits or concrete data
//   structs; name the trait after behavior, not after incidental fields.
// - EntityGraphData should become a serializable struct or enum once graph
//   variants are known. EmptyObject is a TS workaround and should disappear.
// - IsObservable maps to an Observable trait backed by explicit subscriber
//   storage in EntityState. Avoid unconstrained generic observers at public
//   boundaries; use associated event payload types where practical.
// - HasInput/HasOutput and single/many variants map to endpoint traits. In Rust,
//   connection cardinality should be encoded in the owning struct fields rather
//   than by structural interface extension.
// - HasInternalQueue<T> should wrap a VecDeque<T> or custom queue abstraction.
//   Keep capacity checks explicit with Result-returning enqueue methods.
// - Replace `AbstractMovingEntity<any>` with a MovingEntity trait object only
//   where heterogeneity is required; prefer generic `T: MovingEntity` elsewhere.

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
