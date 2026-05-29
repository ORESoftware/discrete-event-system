'use strict';

import {EntityObserver} from "../abstract/abstract";
import {AbstractMovingEntity} from "../entity-moving/moving";


export class ProgramObserver extends EntityObserver<any> {

  movingEntities = new Set<AbstractMovingEntity<any>>();

  doUpdate<T>(type: string, m: T): void {

    if (type === 'NEW_BASIC_MOVING_ENTITY') {
      this.movingEntities.add(m as any);
    }

  }

  addMovingEntityRef(m: any): void {
    this.movingEntities.add(m);
  }

  getStatus() {
    for (const v of this.movingEntities) {
      console.log(v.getGraphData())
    }
  }

}
