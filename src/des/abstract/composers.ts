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