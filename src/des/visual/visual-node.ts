'use strict';

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
  subscribersByEvent = new Map();
  subscribers = new Set<EntityObserver<any>>();
  connectionsOut = new Map<VisualNode, VisualConnection>();
  connectionsIn = new Map<VisualNode, VisualConnection>();

  constructor(v: { label: string, iconUrl: string, entity: StationaryEntity<T> }) {
    this.entity = v.entity;
    this.label = v.label;
    this.iconUrl = v.iconUrl;

    this.subscription = new VisualNodeObserver(v.entity, (type, m) => {
      this.subscribers.forEach(s => {
        s.doUpdate(type,m);
      });
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
        throw new Error("Method not implemented.");
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
    throw new Error("Method not implemented.");
  }

  sendUpdateToSubs<T>(type: string, v: T): void {
    throw new Error("Method not implemented.");
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