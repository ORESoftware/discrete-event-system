// =============================================================================
// RUST MIGRATION  —  target: src/des/abstract/composers.rs  (module des::abstract_::composers)
// 1:1 file move. Reusable behaviour "mixins" composed into stationary entities.
// (`abstract` is a Rust keyword — name the parent module `abstract_` or `core`.)
//
// Declarations → Rust:
//   class DoesFanOut<V>  -> struct DoesFanOut { entity } + impl
//                           (the fan-out routing helper a node delegates to)
//
// Conversion notes (file-specific):
//   - `entity: HasManyOutputConnections<any,any>` -> hold a trait object
//     `Rc<RefCell<dyn HasManyOutputConnections<S,T>>>`; replace the `<any,any>`
//     with concrete generics on DoesFanOut.
//   - The `opts` field + `Object.assign({}, {entity})` is just constructor
//     plumbing; in Rust take `entity` directly as a struct field (no opts twin).
//   - The generic `<V>` is unused in the body — drop it unless a real type is
//     threaded through during migration.
//   - `doFanOut` loops `getOutConnections()` and calls `acceptItem`/`takeItem`
//     on the resolved target; with `Rc<RefCell<..>>` borrow the target inside
//     the loop and release before re-borrowing to satisfy the borrow checker.
//   - `console.error(...)` -> `eprintln!`/`tracing::warn!`.
// =============================================================================

import {AbstractMovingEntity, BasicMovingEntity} from "../entity-moving/moving";
import {StationaryEntity} from "./abstract";
import {HasManyOutputConnections} from "./interfaces";


export class DoesFanOut<V> {

  entity: HasManyOutputConnections<any,any>

  opts: {
    entity: HasManyOutputConnections<any,any>
  }

  constructor({entity}: DoesFanOut<V>['opts']) {

    this.opts = Object.assign({}, {
      entity: entity
    });

    this.entity = entity;
  }

  doFanOut(ame: AbstractMovingEntity<any>) : {accepted: boolean} {

    let accepted = false;
    for (const conn of this.entity.getOutConnections()) {
      const target = conn.getTarget();
      if (!target) {
        console.error('warning: could not find target.')
        continue;
      }

      if ((accepted = (target.acceptItem(ame)))) {
        // const size = this.queue.size;
        // console.log('queue size before remove:',this.queue);
        // this.queue.remove(k);
        // if (this.queue.size !== (size - 1)) {
        //   throw makeError('queue size should be one smaller.');
        // }
        target.takeItem(ame);
        break;
      }
    }

    return {accepted};
  }

}
