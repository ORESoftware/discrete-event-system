'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-source/source.rs  (module des::entity_source::source)
// 1:1 file move. Source entities that inject new moving-entities into the network.
//
// Declarations → Rust:
//   abstract class AbstractSourceEntity<S,T> -> trait SourceEntity (defaults) +
//                                               base struct { connections_out }
//   type EntitySourceGraphData               -> struct EntitySourceGraphData
//   class EntitySource<S,T>                   -> struct + impl (Poisson-ish via RV)
//   class DefiniteFiniteSource<V,S,T>         -> struct + impl (drains initialValues)
//
// Conversion notes (file-specific):
//   - `addOutConnection<T>(...)` redeclares a generic `T` that SHADOWS the class
//     `T`; in Rust rename the method generic to avoid the shadow.
//   - `(global as any).turnOffSources` is an ambient global flag -> pass shared
//     run-state/config in, not a process global.
//   - `rv.getNextEventQuantity(stepSize)` pulls randomness -> the RandomVariable
//     must carry an injected `RandomSource` (shared/capabilities), not Math.random.
//   - `[util.inspect.custom](depth, options)` debug hook -> `impl fmt::Debug`.
//   - `Object.assign({turnOffAfterCount:-1}, opts)` -> struct fields w/ defaults.
//   - `math.BigNumber` stepSize -> decimal crate / `f64`.
//   - `LinkedQueue<V>` (queue/outQueue) -> `VecDeque<V>`; dequeue `[k,v]` tuple
//     cast `as [any,any]` -> typed `Option<(K,V)>`; `getNextValue(): [V,V]` typed.
//   - `getSerializableData(): Partial<this>` returning `this` -> a serde DTO.
//   - `connectionsOut: Set<EntityConnection>` -> `Vec`/`HashSet` of `Rc<RefCell<..>>`
//     trait objects (graph edges); `acceptItem`/`takeItem` borrow target then release.
//   - `throw new Error('needs initial values')` -> `Result` or `panic!` on construct.
//   - console.warn/debug -> `tracing` macros.
// =============================================================================

import * as math from "mathjs";
import {number} from "mathjs";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import {Entity, EntityConnection, EntityObserver, HasNumericValue, StationaryEntity} from "../abstract/abstract";
import {BasicMovingEntity, BasicQuantityMovingEntity, ProcessableMovingEntity} from "../entity-moving/moving";
import {getShortUUID, HasComputedProperties, makeError} from "../general/general";
import {RandomVariable} from "../random-variables/rv";
import * as util from "util";
import {IsVoid, LinkedQueue, LinkedQueueValue} from "@oresoftware/linked-queue";
import * as uuid from "uuid";
import {reg} from "../general/entity-registration";
import {debugLog} from "../shared/debug-log";


export abstract class AbstractSourceEntity<S, T>
  extends Entity<AbstractSourceEntity<S, T>>
  implements HasComputedProperties<AbstractSourceEntity<S, T>>, HasManyOutputConnections<S, T> {

  connectionsOut = new Set<EntityConnection<S, T>>();

  abstract getWithComputedProperties(): this;

  addOutConnection<T>(target: HasManyInputConnections<this, T>): EntityConnection<this, T> {
    const conn = new EntityConnection<this, T>(this, target);
    this.connectionsOut.add(conn);
    // target.addInConnection(this);
    return conn;
  }

  doTimeStep(stepSize: math.BigNumber) {
    return this.runTimeStep(stepSize);
  }

  getOutConnections(): Set<EntityConnection<S, T>> {
    return this.connectionsOut;
  }

  doSetupAfterInputConn(): boolean {
    return false;
  }

  doSetupAfterOutputConn(): boolean {
    return false;
  }

  notifyTargets(): void {
  }

}


type EntitySourceGraphData = {
  createdCount: number
}

export class EntitySource<S, T>  // S = Source, T= Target
  extends AbstractSourceEntity<S, T>
  implements StationaryEntity<EntitySource<S, T>>, HasManyOutputConnections<S, T> {

  createdCount = 0;
  rv: RandomVariable;
  opts: { turnOffAfterCount: number, rv: RandomVariable };
  // TODO: this should be a linkedqueue!
  queue: Array<BasicMovingEntity> = [];

  constructor(id: string, opts: EntitySource<S, T>['opts']) {
    super(id);
    this.rv = opts.rv;
    reg.registerSource(this);
    this.opts = Object.assign({
      turnOffAfterCount: -1
    }, opts, {});
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }

  getCleanVersion() {
    return Object.assign({}, this, {
      subscribers: {
        size: this.subscribers.size
      },
      connectionsOut: {
        size: this.connectionsOut.size
      },
      queue: {
        size: this.queue.length
      }
    });
  }

  [util.inspect.custom](depth: number, options: any) {
    return this.getCleanVersion();
  }

  getWithComputedProperties(): this {
    return Object.assign(this.getCleanVersion(), {});
  }

  addOutConnection<T>(target: HasManyInputConnections<this, T>): EntityConnection<this, T> {
    const conn = new EntityConnection(this, target);
    this.connectionsOut.add(conn);
    return conn;
  }

  doTimeStep(stepSize: math.BigNumber): void {
    return this.runTimeStep(stepSize)
  }

  getSerializableData(): Partial<this> {
    return this;
  }

  checkIfSourcesOff(): boolean {

    if ((global as any).turnOffSources) {
      return true;
    }

    if (this.opts.turnOffAfterCount > 0) {
      if (this.createdCount > this.opts.turnOffAfterCount) {
        return true;
      }
    }
    return false;
  }

  runTimeStep(stepSize: math.BigNumber): void {

    this.timeStepCount++;

    if (this.checkIfSourcesOff()) {
      return;
    }

    // process previously untaken items

    // while(true){
    //     const next = this.queue[0];
    //     for(const target of this.getOutConnections()){
    //         if(target.acceptItem(next)){
    //             target.takeItem(this.queue.shift());
    //             break;
    //         }
    //     }
    // }


    const numEvents = this.rv.getNextEventQuantity(stepSize);

    for (let i = 0; i < numEvents; i++) {

      if (this.checkIfSourcesOff()) {
        break;
      }

      const next = new ProcessableMovingEntity().init();
      this.sendUpdateToSubs('NEW_BASIC_MOVING_ENTITY', next);
      this.createdCount++;

      let accepted = false;
      for (const conn of this.getOutConnections()) {
        const target = conn.getTarget();
        if (!target) {
          console.warn(`[source:${this.id}] out-connection has no resolvable target; cannot route newly-created entity.`);
          continue;
        }
        if ((accepted = (target.acceptItem(next)))) {
          target.takeItem(next);
          break;
        }
      }

      if (!accepted) {
        debugLog(() => `[source:${this.id}] no downstream accepted new entity (backpressure); buffering in source queue (size now ${this.queue.length + 1}).`);
        this.queue.push(next);
      }

    }

    this.sendUpdateToSubs('GRAPH_DATA:SOURCE', this.getGraphData());
  }

  getGraphData(): any {
    return {
      id: this.id,
      createdCount: this.createdCount
    }
  }

  doValidationBeforeRun(): boolean {
    return false;
  }

  notifyTargets(): void {
    //TODO: we would notify targets of whatever relevant updates happen in this source
  }

  doValidation(): void {
  }

}


export class DefiniteFiniteSource<V extends HasNumericValue, S, T>  // S = Source, T= Target
  extends AbstractSourceEntity<S, T>
  implements StationaryEntity<DefiniteFiniteSource<V, S, T>>, HasManyOutputConnections<S, T> {

  createdCount = 0;
  opts: { turnOffAfterCount: number, initialValues: V[] };
  // TODO: this should be a linked-queue!
  queue = new LinkedQueue<V>();
  outQueue = new LinkedQueue<V>();

  constructor(id: string, opts: DefiniteFiniteSource<V, S, T>['opts']) {
    super(id);
    this.opts = Object.assign({
      turnOffAfterCount: -1
    }, opts);


    if(this.opts.initialValues.length < 1){
      throw new Error('needs initial values.')
    }

    while (this.opts.initialValues.length > 0) {
      // pop from back, but add to front to keep order, pop is faster than shift
      const val = this.opts.initialValues.pop();
      this.queue.addToFront(val);
    }
  }

  isDoneEmitting(){
    //TODO: if done emitting, we can remove from main simulation iteration
    return this.queue.size < 1;
  }

  doSetupAfterInputConn(): boolean {
    return true;
  }

  doSetupAfterOutputConn(): boolean {
    return true;
  }

  getNextValue(): [V,V] {
    return this.queue.dequeue() as [V,V];
  }

  getCleanVersion() {
    return Object.assign({}, this, {
      subscribers: {
        size: this.subscribers.size
      },
      connectionsOut: {
        size: this.connectionsOut.size
      },
      queue: {
        size: this.queue.length
      }
    });
  }

  [util.inspect.custom](depth: number, options: any) {
    return this.getCleanVersion();
  }

  getWithComputedProperties(): this {
    return Object.assign(this.getCleanVersion(), {});
  }

  addOutConnection<T>(target: HasManyInputConnections<this, T>): EntityConnection<this, T> {
    const conn = new EntityConnection(this, target);
    this.connectionsOut.add(conn);
    return conn;
  }

  doTimeStep(stepSize: math.BigNumber): void {
    return this.runTimeStep(stepSize)
  }

  getSerializableData(): Partial<this> {
    return this;
  }

  checkIfSourcesOff(): boolean {

    if ((global as any).turnOffSources) {
      return true;
    }

    if (this.opts.turnOffAfterCount > 0) {
      if (this.createdCount > this.opts.turnOffAfterCount) {
        return true;
      }
    }
    return false;
  }

  runTimeStep(stepSize: math.BigNumber): void {

    //TODO: return a symbol from runTimeStep, which gives us information about
    // 1. should this element be removed from the loop, like a source with no more emissions?

    this.timeStepCount++;

    if (this.checkIfSourcesOff()) {
      return;
    }

    // process previously untaken items
    let count = this.outQueue.size;

    // if(count < 1){
    //   // throw makeError('finite source has been exhausted')
    //   console.error('no items in queue.')
    //   return;
    // }

    while (count-- >= 0) {
      const [k, v] = this.outQueue.dequeue() as [any, any];
      if (IsVoid.check(k)) {
        continue;
      }
      for (const conn of this.getOutConnections()) {
        if (conn.target.acceptItem(k)) {
          conn.target.takeItem(k);
        } else {
          // TODO: we put it on back of queue instead of front
          this.outQueue.enqueue([k, v]);
        }
        break;
      }
    }

    if(this.queue.size < 1){
      //   // throw makeError('finite source has been exhausted')
      //   console.error('no items in queue:', this.timeStepCount)
        return;
    }

    const [k,v] = this.getNextValue();

    if(IsVoid.check(k)){
      // the initialValues are probably drained
      console.warn(`[finite-source:${this.id}] getNextValue() returned void at step ${this.timeStepCount}; initial values appear drained (queue.size=${this.queue.size}).`);
      return;
    }

    const next = new BasicQuantityMovingEntity(k.value).init();
    this.sendUpdateToSubs('NEW_BASIC_MOVING_ENTITY', next);
    this.createdCount++;

    let accepted = false;
    for (const conn of this.getOutConnections()) {
      const target = conn.getTarget();
      if (!target) {
        console.warn(`[finite-source:${this.id}] out-connection has no resolvable target; cannot route emitted value.`);
        continue;
      }
      if ((accepted = (target.acceptItem(next)))) {
        target.takeItem(next);
        break;
      }
    }

    if (!accepted) {
      debugLog(() => `[finite-source:${this.id}] no downstream accepted emitted value (backpressure); buffering in outQueue (size now ${this.outQueue.size + 1}).`);
      this.outQueue.enqueue(next);
    }

    this.sendUpdateToSubs('GRAPH_DATA:SOURCE', this.getGraphData());
  }

  getGraphData(): any {
    return {
      id: this.id,
      createdCount: this.createdCount
    }
  }

  doValidationBeforeRun(): boolean {
    return false;
  }

  notifyTargets(): void {
    //TODO: we would notify targets of whatever relevant updates happen in this source
  }

  doValidation(): void {
  }

}
