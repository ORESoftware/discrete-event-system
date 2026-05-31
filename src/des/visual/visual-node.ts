'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/visual/visual-node.rs  (module des::visual::visual_node)
// 1:1 file move. Visualization wrapper nodes mirroring the entity graph for UI.
//
// Declarations → Rust:
//   class VisualNodeObserver extends EntityObserver -> struct + impl EntityObserver
//   class VisualConnection                          -> struct { source, target }
//   enum VisualNodeEvents { FOO = 'FOO' }           -> enum (#[serde(rename_all=...)] string-valued)
//   class VisualNode<T> implements IsObservable     -> struct + impl IsObservable
//   class OneInOneOut/OneInManyOut/ZeroInManyOut/ZeroOutManyIn extends VisualNode
//                                                   -> marker subtypes (newtype wrappers
//                                                      or an arity enum; no `extends`)
//   class ManyInManyOut                             -> struct
//
// Conversion notes (file-specific):
//   - `VisualNodeEvents` is a STRING enum -> Rust enum with `#[serde(rename_all)]`.
//   - The ctor wires `subscription` to a CLOSURE that captures `this` then iterates
//     `this.subscribers` -> self-referential observer; use `Rc<RefCell<..>>` +
//     `FnMut`/`move` and mind the borrow checker (don't hold a borrow across `doUpdate`).
//   - `fn: ((type,m)=>void|null) = null as any` placeholder -> `Option<Box<dyn FnMut(..)>>`.
//   - `connectionsOut/In: Map<VisualNode, VisualConnection>` keyed by NODE IDENTITY
//     -> key by id/index (`HashMap<NodeId, VisualConnection>`); a whole struct as a
//     map key needs `Hash + Eq` and is the wrong model.
//   - `sub()` builds an anonymous `new class extends EntityObserver {..}` -> a
//     closure-backed adapter struct implementing the trait.
//   - Many `throw new Error("Method not implemented.")` (subscribeTo,
//     subscribeWithFrequency, sendUpdateToSubs) -> `unimplemented!()`.
//   - `subscribers: Set<EntityObserver>` / `subscribersByEvent: Map<..>` ->
//     `Vec<Rc<RefCell<dyn EntityObserver>>>` / `HashMap<String, Vec<..>>`.
//   - The empty `OneInOneOut`… subclasses are arity markers — likely collapse to
//     one struct + an arity field/enum rather than four empty types.
// =============================================================================

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