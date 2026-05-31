'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-sink/generic-sink.rs  (module des::entity_sink::generic_sink)
// 1:1 file move. A sink that logs each absorbed entity's value before destroying it.
//
// Declarations → Rust:
//   type EntitySinkGraphData          -> struct { destroyed_count }
//   const entityType (Symbol)         -> n/a (enum/trait tag)
//   class GenericEntitySink<S,T>      -> struct + impl (+ impl AbstractSinkEntity, StationaryEntity)
//
// Conversion notes (file-specific):
//   - NOTE: `EntitySinkGraphData` and the `entityType` Symbol are duplicated from
//     sink.ts — define them ONCE in Rust (shared sink module) and reuse.
//   - `[entityType]='Sink'` + `['entity.type']='Sink'` brand -> enum/trait tag.
//   - `opts: {}` empty -> `()` / drop; `Object.assign({}, opts)` is a no-op clone.
//   - `console.log('generic sink value:', m.getValue())` is the only behavioural
//     difference from EntitySink -> `tracing`/`println!`.
//   - `math.BigNumber` stepSize -> decimal/f64; `getSerializableData/getWithComputedProperties: any` -> serde DTO.
//   - `[util.inspect.custom]` -> `impl fmt::Debug`; `connectionsIn: Set<..>` -> Vec/HashSet of edges.
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
import {AbstractSinkEntity} from "./sink";


type EntitySinkGraphData = {
  destroyedCount: number
};

const entityType = Symbol('entity.type');

export class GenericEntitySink<S, T>
  extends AbstractSinkEntity<S, T>
  implements StationaryEntity<GenericEntitySink<S, T>>, HasManyInputConnections<S, T> {

  [entityType] = 'Sink';
  ['entity.type'] = 'Sink';

  opts: {}
  destroyedCount = 0;

  constructor(id: string, opts: GenericEntitySink<S, T>['opts']) {
    super(id);
    this.opts = Object.assign({}, opts);
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
    console.log('generic sink value:',m.getValue());
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
