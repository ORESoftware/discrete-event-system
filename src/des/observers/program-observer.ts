'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/observers/program-observer.rs  (module des::observers::program_observer)
// 1:1 file move. An observer that tracks all created moving-entities.
//
// Declarations → Rust:
//   class ProgramObserver extends EntityObserver<any> -> struct + impl EntityObserver
//
// Conversion notes (file-specific):
//   - `doUpdate` dispatches on `type === 'NEW_BASIC_MOVING_ENTITY'` (a string) ->
//     match on an event enum, not a raw `String`.
//   - `movingEntities: Set<AbstractMovingEntity>` -> `Vec`/`HashSet` of
//     `Rc<RefCell<dyn MovingEntity>>` (trait objects; need Hash+Eq for a set).
//   - `m as any` payload casts -> a typed event payload (enum) instead of `any`.
//   - `getStatus()` console.log loop -> `tracing`.
// =============================================================================

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
