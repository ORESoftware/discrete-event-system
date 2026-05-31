#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_epidemic.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';


import * as safe from '@oresoftware/safe-stringify';
import * as math from 'mathjs';
import * as uuid from 'uuid';

import {number} from "mathjs";
import {EntitySource} from "./entity-source/source";
import {EntityProcessor} from "./entity-processing/processing";
import {EntitySink} from "./entity-sink/sink";
import {Entity, EntityObserver, StationaryEntity} from "./abstract/abstract";
import {AbstractMovingEntity} from "./entity-moving/moving";
import {EntityGraphData, HasManyInputConnections, HasManyOutputConnections} from "./abstract/interfaces";
import {bgn, fisherYatesShuffle, makeError} from "./general/general";
import {
  ExponentialRandomVariable as ExponentialRandomVariable3,
  // ExponentialRandomVariable3,
  PoissonRandomVariable,
  UniformRandomVariable
} from "./random-variables/rv";
import {getWebsocketServer} from "./ws-server/ws-server";
import {VisualNode} from "./visual/visual-node";
import {ProgramObserver} from "./observers/program-observer";
import {ProbabilityDecisionEntity} from "./entity-decision/probability-decision";
import * as entityReg from './general/entity-registration';
import {reg} from "./general/entity-registration";
import {doAudit} from "./general/do-audit";

// export interface MovingEntity {
//     nextTarget: StationaryEntity, // null if in processing
//     timeInSystem: math.BigNumber;
//     runTimeStep(stepSize: math.BigNumber): void;
// }


const run = () => {

  const stepSizeMillis = bgn(21);
  const obs = new ProgramObserver();

  const lambda = math.divide(bgn(1), bgn(400)) as math.BigNumber;

  const programEntities = new Map<string, StationaryEntity<any>>([

    [
      'main-source',
      new EntitySource(
        'main-source',
        {turnOffAfterCount: 300, rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})}
      ).subscribe(obs)
    ],

    [
      'S',
      new EntityProcessor(
        'S',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'E',
      new EntityProcessor(
        'E',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'I-P',
      new EntityProcessor(
        'I-P',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'I-P-Decision',
      new ProbabilityDecisionEntity(
        'I-P-Decision',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)}),
        {
          rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),
          probabilities: [
            {index: 0, prob: bgn(0.4)},
            {index: 1, prob: bgn(0.6)}
          ]
        }
      ).subscribe(obs)
    ],

    [
      'I-S',
      new EntityProcessor(
        'I-S',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'I-S-Decision',
      new ProbabilityDecisionEntity(
        'I-S-Decision',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)}),
        {
          probabilities: [
            {index: 0, prob: bgn(0.4)},
            {index: 1, prob: bgn(0.6)}
          ],
          rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),
        }
      ).subscribe(obs)
    ],

    [
      'I-A',
      new EntityProcessor(
        'I-A',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'I-H',
      new EntityProcessor(
        'I-H',
        // new ExponentialRandomVariable3({lambda, timeStep: bgn(stepSizeMillis)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'I-H-Decision',
      new ProbabilityDecisionEntity(
        'I-H-Decision',
        {
          rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)}),

          probabilities: [
            {index: 0, prob: bgn(0.4)},
            {index: 1, prob: bgn(0.6)}
          ]
        }
      ).subscribe(obs)
    ],

    [
      'R',

      new EntityProcessor(
        'R',
        // new ExponentialRandomVariable3({lambda: bgn(400), timeStep: bgn(500)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'D',


      new EntityProcessor(
        'D',
        // new ExponentialRandomVariable3({lambda: bgn(400), timeStep: bgn(500)})
        {rv: new UniformRandomVariable({aVal: bgn(10), bVal: bgn(20)})},
      ).subscribe(obs)
    ],

    [
      'main-sink',
      new EntitySink(
        'main-sink',
        new PoissonRandomVariable(),
        {}
      ).subscribe(obs)
    ],

  ]);


  //   susceptible (S),
  //   exposed (E),
  //   presymptomatic (IP),
  //   asymptomatic (IA),
  //   symptomatic (IS),
  //   hospitalized (IH),
  //   recovered (R),
  //   or dead (D).
  //

  for (const [sourceId, targetId] of [

    ['main-source', 'S'],
    ['S', 'E'],
    ['E', 'I-P'],
    ['I-P', 'I-A'],
    ['I-P', 'I-S'],
    ['I-P', 'I-P-Decision'],
    ['I-P-Decision', 'I-A'],
    ['I-P-Decision', 'I-S'],
    ['I-A', 'R'],
    ['I-S', 'I-S-Decision'],
    ['I-S-Decision', 'R'],
    ['I-S-Decision', 'I-H'],
    ['I-H', 'I-H-Decision'],
    ['I-H-Decision', 'R'],
    ['I-H-Decision', 'D'],
    ['D', 'main-sink'],
    ['R', 'S'],

  ]) {

    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId) as any;

    console.log({source});

    source.addOutConnection(target as any);
    target.addInConnection(source);

  }

  for (const [k, v] of programEntities) {
    v.doSetupAfterOutputConn();
    v.doSetupAfterInputConn();

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

  const now = Date.now();
  for (let i = 0; i < 1000; i++) { // 1000 time steps

    console.log('doing first iteration:', i);

    // doAudit();

    for (const [k, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSizeMillis);
    }

  }

  console.log(Date.now() - now);

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
