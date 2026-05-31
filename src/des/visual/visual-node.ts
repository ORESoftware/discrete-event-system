'use strict';

// RUST MIGRATION:
// - Target: src/des/visual/visual_node.rs
// - VisualNodeObserver, VisualConnection, VisualNodeEvents, VisualNode, and the
//   arity marker subclasses become observer/graph-view structs and enums; the
//   empty subclasses likely become enum variants or type aliases.
// - IsObservable should be a nominal Observable trait with typed event payloads;
//   `Map<VisualNode, VisualConnection>` needs stable node ids or arena handles
//   because Rust HashMap keys cannot be mutable object identities.
// - Replace anonymous inner EntityObserver classes, `any` subscribers, and icon
//   URL strings with concrete observer structs, typed graph events, and
//   Result-returning APIs.

import {Entity, EntityObserver, StationaryEntity} from "../abstract/abstract";
import {IsObservable} from "../abstract/interfaces";


export class VisualNodeObserver extends EntityObserver<any> {

  entity: StationaryEntity<any>;
  fn: ((type: string, m: any) => void | null) = null as any

  constructor(e: StationaryEntity<any>, fn: (type: string, m: any) => void) {
    super();
    e.subscribe(this);
    this.entity = e;
    this.fn = fn;
  }

  doUpdate<T>(type: string, m: T): void {
    this.fn(type, m)
  }


}

export class VisualConnection {
  source: VisualNode;
  target: VisualNode;

  constructor(v: { source: VisualNode, target: VisualNode }) {
    this.source = v.source;
    this.target = v.target
  }

}

export enum VisualNodeEvents {
  FOO = 'FOO'
}

export class VisualNode<T = any> implements IsObservable {

  entity: StationaryEntity<any>
  label: string
  iconUrl: string; // url
  subscription: EntityObserver<any>;
  subscribersByEvent = new Map<string, Set<EntityObserver<any>>>();
  subscribers = new Set<EntityObserver<any>>();
  connectionsOut = new Map<VisualNode, VisualConnection>();
  connectionsIn = new Map<VisualNode, VisualConnection>();

  constructor(v: { label: string, iconUrl: string, entity: StationaryEntity<T> }) {
    this.entity = v.entity;
    this.label = v.label;
    this.iconUrl = v.iconUrl;

    this.subscription = new VisualNodeObserver(v.entity, (type, m) => {
      this.sendUpdateToSubs(type, m);
      // wss.connections.forEach(c => {
      //   c.send({
      //     data: v.entity.getGraphData()
      //   });
      // });
    });
  }

  doValidationBeforeRun(): boolean{
    return true;
  }

  subscribeTo(name: VisualNodeEvents, o: EntityObserver<any>): this {
    const subscribers = this.subscribersByEvent.get(name);
    if (subscribers) {
      subscribers.add(o);
      return this;
    }
    this.subscribersByEvent.set(name, new Set([o]));
    return this;
  }

  sub(fn: (type: string, m: Entity<any>) => void): this {
    return this.subscribe(new class extends EntityObserver<any> {
      doUpdate<T extends Entity<any>>(type: string, m: T): void {
        return fn(type, m);
      }
    });
  }

  subscribe(o: EntityObserver<any>): this {
    this.subscribers.add(o);
    return this;
  }

  unsubscribe(o: EntityObserver<any>): boolean {
    if (!this.subscribers.has(o)) {
      return false;
    }
    this.subscribers.delete(o);
    return true;
  }

  subscribeWithFrequency(count: number, o: EntityObserver<any>): this {
    void count;
    return this.subscribe(o);
  }

  sendUpdateToSubs<T>(type: string, v: T): void {
    const alreadyNotified = new Set<EntityObserver<any>>();
    const eventSubscribers = this.subscribersByEvent.get(type);

    if (eventSubscribers) {
      eventSubscribers.forEach(s => {
        alreadyNotified.add(s);
        s.doUpdate(type, v);
      });
    }

    this.subscribers.forEach(s => {
      if (!alreadyNotified.has(s)) {
        s.doUpdate(type, v);
      }
    });
  }

  addVisualConnectionOut(target: VisualNode) {
    this.connectionsOut.set(target, new VisualConnection({source: this, target: target}))
  }

  addVisualConnectionIn(target: VisualNode) {
    this.connectionsIn.set(target, new VisualConnection({source: target, target: this}));
  }

}


export class OneInOneOut extends VisualNode {

}

export class OneInManyOut extends VisualNode {

}


export class ZeroInManyOut extends VisualNode {

}


export class ZeroOutManyIn extends VisualNode {

}

export class ManyInManyOut {


}
