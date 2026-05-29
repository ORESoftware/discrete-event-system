'use strict';

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
