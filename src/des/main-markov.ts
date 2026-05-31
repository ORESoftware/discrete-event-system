#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-markov.rs   (fn main)
// 1:1 file move. Wires a Markov-chain-style entity network of processors and
// observes its steady-state behaviour.
//
// Conversion notes (file-specific):
//   - imports many des entity modules -> use crate::des::...
//   - Set<EntityProcessor> (allProcessors) -> HashSet (needs Hash+Eq) or Vec.
//   - uuid.v4 -> uuid::Uuid::new_v4; safe-stringify -> serde_json; mathjs bgn
//     -> f64 / decimal.
//   - `any` generics -> concrete entity types; top-level run -> fn main.
// =============================================================================


import * as safe from '@oresoftware/safe-stringify';
import * as math from 'mathjs';
import * as uuid from 'uuid';

import {number} from "mathjs";
import {EntitySource} from "./entity-source/source";
import {EntityProcessor} from "./entity-processing/processing";
import {EntitySink} from "./entity-sink/sink";
import {Entity, EntityObserver} from "./abstract/abstract";
import {bgn, fisherYatesShuffle, makeError} from "./general/general";
import {ExponentialRandomVariable, PoissonRandomVariable, UniformRandomVariable} from "./random-variables/rv";
import {ProgramObserver} from "./observers/program-observer";

// export interface MovingEntity {
//     nextTarget: StationaryEntity, // null if in processing
//     timeInSystem: math.BigNumber;
//     runTimeStep(stepSize: math.BigNumber): void;
// }

const allProcessors = new Set<EntityProcessor<any, any>>();

const addToSet = (e: EntityProcessor<any, any>) => {
  allProcessors.add(e);
  return e;
}

const auditSizes = (s: Set<EntityProcessor<any, any>>) => {

  let first = true;
  let previousTotal = 0;

  return () => {

    let total = 0;

    for (const v of s) {
      // console.log(v.doAudit());
      total += v.doAudit().totalSize;
    }

    if (!first) {
      if (previousTotal !== total) {
        throw makeError('totals are not equal')
      }
    }

    previousTotal = total;
    first = false;

    // console.log({total});
  }
}


const doAudit = auditSizes(allProcessors);


const run = () => {

  const stepSizeMillis = bgn(500);
  const obs = new ProgramObserver();

  // const programEntities = new Map<string, Entity<any>>([
  //
  //   [
  //     'A-source',
  //     new EntitySource(
  //       'A-source',
  //       new ExponentialRandomVariable({lambda: bgn(math.divide(5, 100)), timeStep: stepSizeMillis}),
  //       {turnOffAfterCount: 100}
  //     ).subscribe(obs)
  //   ],
  //
  //   [
  //     'A', new EntityProcessor(
  //     'A',
  //     new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
  //   ).subscribe(obs)
  //   ],
  //
  //
  //   [
  //     'B-source', new EntitySource(
  //     'B-source',
  //     new ExponentialRandomVariable({lambda: bgn(math.divide(5, 100)), timeStep: stepSizeMillis}),
  //     {turnOffAfterCount: 100}
  //   ).subscribe(obs)
  //   ],
  //
  //   [
  //     'B', new EntityProcessor(
  //     'B',
  //     new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
  //   ).subscribe(obs)
  //   ],
  //
  //   [
  //     'C', new EntityProcessor(
  //     'C',
  //     new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
  //   ).subscribe(obs)
  //   ],
  //
  //   [
  //     'C-source', new EntitySource(
  //     'C-source',
  //     new ExponentialRandomVariable({lambda: bgn(math.divide(5, 100)), timeStep: stepSizeMillis}),
  //     {turnOffAfterCount: 100}
  //   ).subscribe(obs)
  //   ],
  //
  //
  //   [
  //     'D',
  //
  //     new EntityProcessor(
  //       'D',
  //       new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis}),
  //     ).subscribe(obs)
  //   ],
  //
  //   [
  //     'D-source',
  //
  //     new EntitySource(
  //       'D-source',
  //       new ExponentialRandomVariable({lambda: bgn(math.divide(5, 100)), timeStep: stepSizeMillis}),
  //       {turnOffAfterCount: 100}
  //     ).subscribe(obs)
  //   ],
  //
  //
  // ]);

  const programEntities = new Map<string, Entity<any>>([

    [
      'A-source',
      new EntitySource(
        'A-source',
        {turnOffAfterCount: 300, rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),}
      ).subscribe(obs)
    ],

    [
      'A', addToSet(
      new EntityProcessor(
        'A',
        {rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})}
      )).subscribe(obs)
    ],


    [
      'B-source', new EntitySource(
      'B-source',
      {turnOffAfterCount: 300, rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),}
    ).subscribe(obs)
    ],

    [
      'B', addToSet(
      new EntityProcessor(
        'B',
        {rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      )
    ).subscribe(obs)
    ],

    [
      'C', addToSet(
      new EntityProcessor(
        'C',
        {rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})}
      )
    ).subscribe(obs)
    ],

    [
      'C-source',
      new EntitySource(
        'C-source',
        {turnOffAfterCount: 300, rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),}
      ).subscribe(obs)
    ],


    [
      'D',

      addToSet(
        new EntityProcessor(
          'D',
          {rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})}
        )
      ).subscribe(obs)
    ],

    [
      'E',

      addToSet(
        new EntityProcessor(
          'E',
          {rv:new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
        )
      ).subscribe(obs)
    ],

    [
      'F',

      addToSet(
        new EntityProcessor(
          'F',
          {rv:new UniformRandomVariable({aVal: bgn(5), bVal: bgn(10)})}
        )
      ).subscribe(obs)
    ],

    [
      'D-source',
      new EntitySource(
        'D-source',
        {turnOffAfterCount: 300, rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),}
      ).subscribe(obs)
    ],

  ]);

  for (const [sourceId, targetId] of [
    ['A-source', 'A'],
    ['A', 'F'],
    ['A', 'B'],
    ['B', 'A'],
    ['B-source', 'B'],
    ['B', 'C'],
    ['C', 'B'],
    ['C-source', 'C'],
    ['C', 'D'],
    ['D', 'C'],
    ['D-source', 'D'],
    ['D', 'E'],
    ['E', 'D'],
    ['E', 'F'],
    ['F', 'E'],
    ['F', 'A'],

  ]) {

    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId) as any;

    console.log({source});

    source.addOutConnection(target as any);
    target.addInConnection(source);

  }


  // for (const [sourceId, targetId] of [
  //   ['A-source', 'A'],
  //   ['A', 'B'],
  //   ['B', 'A'],
  //   ['B-source', 'B'],
  //   ['B', 'C'],
  //   ['C', 'B'],
  //   ['C-source', 'C'],
  //   ['C', 'D'],
  //   ['D', 'C'],
  //   ['D-source', 'D'],
  //   ['D', 'E'],
  //   ['E', 'D'],
  //   ['E', 'F'],
  //   ['F', 'E'],
  //
  // ]) {
  //
  //   const source = programEntities.get(sourceId) as any;
  //   const target = programEntities.get(targetId) as any;
  //
  //   console.log({source});
  //
  //   source.addOutConnection(target as any);
  //   target.addInConnection(source);
  //
  // }


  console.log(
    programEntities.get('A')
  )

  const programList = Array.from(programEntities);

  for (let i = 0; i < 1000; i++) { // 1000 time steps

    console.log('doing first iteration:', i);

    for (const [k, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSizeMillis);
    }

  }

  (global as any).turnOffSources = true;


  for (let i = 0; i < 500; i++) { // 1000 time steps

    console.log('doing second iteration:', i);

    doAudit();

    for (const [k, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(bgn(500));
    }

  }


  for (const b of obs.movingEntities) {
    console.log(b);
  }

  let i = 0;
  for (const e of programEntities.values()) {
    i++;
    console.log(i, i, i, i, i, i, i, i, i, '**************************************')
    console.log(i, i, i, i, i, i, i, i, i, '**************************************')
    console.log(e.getWithComputedProperties());
  }

}

run();

