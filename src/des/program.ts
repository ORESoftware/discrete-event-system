#!/usr/bin/env ts-node
// RUST MIGRATION: target src/des/program.rs.
// RUST MIGRATION: Keep this as a library-ish orchestration module; if runnable behavior remains, add a thin src/bin/program.rs wrapper.
// RUST MIGRATION: Map classes/interfaces to structs/traits and keep visual/runtime effects behind explicit ports returning Result.
// RUST MIGRATION: Use clap/std::env/PathBuf only at wrapper boundaries and keep JSON examples/config as serde-deserialized structs.
'use strict';



import {VisualNode} from "./visual/visual-node";
import {EntitySource} from "./entity-source/source";
import {ExponentialRandomVariable, PoissonRandomVariable} from "./random-variables/rv";
import {bgn} from "./general/general";
import * as math from "mathjs";
import {EntityProcessor} from "./entity-processing/processing";
import {EntitySink} from "./entity-sink/sink";

export const getEntities = (stepSize: math.BigNumber) => {

  return new Map<string, VisualNode<any>>([

    [
      'A',

      new VisualNode({
        label: 'A',
        iconUrl: 'https://xyz.com',
        entity: new EntitySource(
          'A',
          {turnOffAfterCount: -1, rv:  new ExponentialRandomVariable({
              lambda: bgn(math.divide(5, 100)),
              timeStep: stepSize
            })}
        )
      })
    ],

    [
      'B',
      new VisualNode({
        label: 'B',
        iconUrl: 'https://xyz.com',
        entity: new EntityProcessor(
          'B',
          {rv:new ExponentialRandomVariable({
            lambda: bgn(math.divide(1, 10)),
            timeStep: stepSize
          })}
        )
      })
    ],

    [
      'C',
      new VisualNode({
        label: 'C',
        iconUrl: 'https://xyz.com',
        entity: new EntityProcessor(
          'C',
          {rv:new ExponentialRandomVariable({
            lambda: bgn(math.divide(1, 10)),
            timeStep: stepSize
          })}
        )
      })
    ],


    [
      'D',
      new VisualNode({
        label: 'D',
        iconUrl: 'https://xyz.com',
        entity: new EntityProcessor(
          'D',
          {rv:new ExponentialRandomVariable({
            lambda: bgn(math.divide(1, 10)),
            timeStep: stepSize
          })}
        )
      })
    ],


    [
      'E',
      new VisualNode({
        label: 'E',
        iconUrl: 'https://xyz.com',
        entity: new EntitySink(
          'E',
          new PoissonRandomVariable(),
          {}
        )
      })
    ],

  ]);
}
