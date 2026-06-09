'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-sink/sink.rs  (module des::entity_sink::sink)
// 1:1 file move. Terminal entities that absorb (destroy) moving-entities.
//
// Declarations → Rust:
//   abstract class AbstractSinkEntity<S,T> -> trait SinkEntity (defaults) +
//                                             base struct { connections_in }
//   type EntitySinkGraphData               -> struct { destroyed_count }
//   const entityType (Symbol)              -> n/a (use an enum/trait tag)
//   class EntitySink<S,T>                  -> struct + impl (+ impl StationaryEntity)
//
// Conversion notes (file-specific):
//   - `[entityType] = 'Sink'` + `['entity.type'] = 'Sink'` brand -> an enum
//     variant / trait tag, not a symbol property. (entityType + EntitySinkGraphData
//     are ALSO declared in generic-sink.ts — unify to one definition in Rust.)
//   - `abstract acceptItem(): boolean` takes NO args here but the HasInput trait's
//     `acceptItem(m)` takes one — reconcile the signature in the Rust trait.
//   - ctor takes `rv: RandomVariable` but never uses it -> drop the parameter.
//   - `opts: {}` empty -> `()` / drop.
//   - `connectionsIn: Set<EntityConnection>` -> `Vec`/`HashSet` of `Rc<RefCell<..>>` edges.
//   - `math.BigNumber` stepSize -> decimal/f64.
//   - `getSerializableData(): any` / `getWithComputedProperties(): any` -> serde DTO.
//   - `[util.inspect.custom]` -> `impl fmt::Debug`; `m.doFinish()` is the absorb side effect.
//   - console.debug -> `tracing`.
// =============================================================================

import * as math from "mathjs";
import {number} from "mathjs";
import {Entity, EntityConnection, StationaryEntity} from "../abstract/abstract";
import {HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";
import {AbstractSourceEntity} from "../entity-source/source";
import {BasicMovingEntity} from "../entity-moving/moving";
import {HasComputedProperties, makeError} from "../general/general";
import {RandomVariable} from "../random-variables/rv";
import * as util from "util";
import {reg} from "../general/entity-registration";
import {debugLog} from "../shared/debug-log";

export abstract class AbstractSinkEntity<S, T>
  extends Entity<AbstractSinkEntity<S, T>>
  implements HasComputedProperties<AbstractSinkEntity<S, T>>, HasManyInputConnections<S, T> {

  connectionsIn = new Set<EntityConnection<S, T>>();

  abstract getWithComputedProperties(): this;

  addInConnection(target: HasManyOutputConnections<any, any>): EntityConnection<S, T> | null {
    const conn = new EntityConnection<AbstractSourceEntity<S, T>, StationaryEntity<AbstractSinkEntity<S, T>>>(target, this)
    this.connectionsIn.add(conn);
    return conn;
  }

  doTimeStep(stepSize: math.BigNumber) {
    return this.runTimeStep(stepSize);
  }

  getInConnections(): Set<EntityConnection<S, T>> {
    return this.connectionsIn;
  }

  abstract acceptItem(): boolean;

  abstract takeItem(m: BasicMovingEntity): void;

  doSetupAfterInputConn(): boolean {
    return false;
  }

  doSetupAfterOutputConn(): boolean {
    return false;
  }

  notifySources(): void {
  }
}


type EntitySinkGraphData = {
  destroyedCount: number
}

const entityType = Symbol('entity.type');

export class EntitySink<S, T>
  extends AbstractSinkEntity<S, T>
  implements StationaryEntity<EntitySink<S, T>>, HasManyInputConnections<S, T> {

  [entityType] = 'Sink';
  ['entity.type'] = 'Sink';

  opts: {}
  destroyedCount = 0;

  constructor(id: string, rv: RandomVariable, opts: EntitySink<S, T>['opts']) {
    super(id);
    this.opts = Object.assign({}, opts);
    reg.registerSink(this);
  }

  addInConnection(source: HasManyOutputConnections<any, any>): EntityConnection<S, T> {
    const conn = new EntityConnection(source, this);
    this.connectionsIn.add(conn);
    return conn;
  }

  getSerializableData(): any {
    return Object.assign(
      this.getWithComputedProperties(), {
        destroyedCount: this.destroyedCount
      });
  }

  runTimeStep(stepSize: math.BigNumber): void {
    // console.log('running time step for EntitySink');
    this.timeStepCount++;

    this.sendUpdateToSubs('SINK', this.getGraphData());
  }

  getGraphData(): EntitySinkGraphData {
    return Object.assign(
      this.getWithComputedProperties(), {
        destroyedCount: this.destroyedCount,
        timeStepCount: this.timeStepCount
      });
  }

  getWithComputedProperties(): any {
    return {
      id: this.id,
      destroyedCount: this.destroyedCount,
    }
  }

  acceptItem() {
    // TODO: should reject items if full?
    return true;
  }

  takeItem(m: BasicMovingEntity) {
    this.destroyedCount++;
    debugLog(() => `[sink:${this.id}] absorbed entity ${(m as any)?.id}; destroyedCount=${this.destroyedCount}.`);
    m.doFinish();
  }

  getCleanVersion() {
    return Object.assign({}, this, {
      subscribers: {
        size: this.subscribers.size
      },
      connectionsIn: {
        size: this.connectionsIn.size
      },
    });
  }

  [util.inspect.custom](depth: number, options: any) {
    return this.getCleanVersion();
  }

  doValidationBeforeRun(): boolean {
    return false;
  }

  doAudit() {
    return {
      totalSize: this.destroyedCount
    }

  }

  doValidation() {
  }

  doSetupAfterInputConn(): boolean {
    return false;
  }

  doSetupAfterOutputConn(): boolean {
    return false;
  }

  notifySources(): void {
  }

}
