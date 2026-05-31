#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-fibonacci-recursion.rs   (fn main)
// 1:1 file move. Wires a small entity graph (source -> value-adder/splitter ->
// sink) that computes Fibonacci by recurrent feedback.
//
// Conversion notes (file-specific):
//   - const run = () => {...} closure + top-level invocation -> fn main().
//   - imports des entity modules (entity-source, value-adder, entity-splitter,
//     generic-sink, observers) -> use crate::des::...
//   - mathjs bgn(500) stepSize -> f64 / decimal.
// =============================================================================

import {DefiniteFiniteSource, EntitySource} from "./entity-source/source";
import {Entity, EntityObserver} from "./abstract/abstract";
import {bgn, fisherYatesShuffle} from "./general/general";
import {ProgramObserver} from "./observers/program-observer";
import {EntityNumericProcessor} from "./entity-processing/value-adder";
import {EntitySplitter} from "./entity-routing/entity-splitter";
import {GenericEntitySink} from "./entity-sink/generic-sink";

// export interface MovingEntity {
//     nextTarget: StationaryEntity, // null if in processing
//     timeInSystem: math.BigNumber;
//     runTimeStep(stepSize: math.BigNumber): void;
// }



// TODO:
// using start conditions [0,1], use recursion until condition is met

// input(0,1) ------> recurse ----> condition -----> done
//                      ^                ⌄
//                       \               |
//                        \____________ /


// perhaps we have two separate previous values, and we sum them


const run = () => {

  const stepSizeMillis = bgn(500);
  const obs = new ProgramObserver();

  const programEntities = new Map<string, Entity<any>>([

    [
      'A',
      new DefiniteFiniteSource(
        'A',
        {turnOffAfterCount: -1, initialValues: [{value: 0}, {value: 1}]}
      ).subscribe(obs)
    ],

    [
      'B', new EntityNumericProcessor('B').subscribe(obs)
    ],

    [
      'C', new EntitySplitter('C', {}).subscribe(obs)
    ],

    [
      'D', new GenericEntitySink('D', {}).subscribe(obs)
    ],

  ]);


  for (const [sourceId, targetId] of [['A', 'B'], ['B', 'C'], ['C', 'D'], ['C', 'B']]) {

    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId);

    source.addOutConnection(target as any);

  }


  const programList = Array.from(programEntities);


  for (let i = 0; i < 100; i++) { // 1000 time steps

    for (const [k, v] of programList) {
      v.doTimeStep(stepSizeMillis);
    }

  }

  // for (const b of obs.movingEntities) {
  //   console.log('foo:',b);
  // }

  // let i = 0;
  // for (const e of programEntities.values()) {
  //   i++;
  //   console.log(i, i, i, i, i, i, i, i, i, '**************************************')
  //   console.log(e.getWithComputedProperties());
  // }

}

run();

