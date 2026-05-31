// RUST MIGRATION:
// - Target: src/des/abstract/composers.rs
// - DoesFanOut<V> becomes a small struct holding a boxed/generic output-endpoint
//   trait object; the fan-out behavior can implement a PureTransform-like trait
//   with `transform(entity) -> FanOutResult`.
// - Replace `HasManyOutputConnections<any, any>` and `AbstractMovingEntity<any>`
//   with explicit endpoint and moving-entity traits. Rust should make the
//   accepted/rejected branch a Result or enum instead of console/error side
//   effects.

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
