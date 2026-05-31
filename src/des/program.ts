#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/program.rs   (module des::program)
// 1:1 file move. Library: builds the default entity graph (getEntities).
//
// Conversion notes (file-specific):
//   - Despite the shebang this is a LIBRARY (exports getEntities, no
//     require.main guard) -> plain module, NOT a bin.
//   - getEntities(stepSize) returns Map<string, VisualNode> -> fn returning
//     HashMap<String, VisualNode<...>>.
//   - mathjs BigNumber stepSize -> f64 / decimal (one engine-wide choice).
//   - constructs entities from many des modules -> use crate::des::...
// =============================================================================



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