'use strict';

// RUST MIGRATION:
// - Target: src/des/observers/program_observer.rs
// - ProgramObserver becomes a concrete Observer trait implementation that owns a
//   collection of moving-entity handles or ids.
// - `Set<AbstractMovingEntity<any>>` needs an ownership decision: Rc<RefCell<_>>,
//   Arc<Mutex<_>>, ids into an arena, or Box<dyn MovingEntity> depending on
//   simulator threading and graph ownership.
// - doUpdate should take a typed event enum/payload instead of string +
//   generic/any, and getStatus should return graph data rather than printing.

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
