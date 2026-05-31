#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main.rs   (fn main)
// 1:1 file move. Top-level wiring of the queueing-network DES (sources,
// processors, sinks, ws server) — the primary runnable entry point.
//
// Conversion notes (file-specific):
//   - Top-level executable code (stepSize + model build + run) -> fn main().
//   - Imports many des modules -> use crate::des::{entity_source, entity_processing,
//     entity_sink, abstract_, observers, random_variables, visual, ws_server, ...}.
//   - mathjs BigNumber (bgn / des.getStepSize) -> pick ONE engine-wide numeric
//     (f64 or a decimal crate).
//   - uuid.v4() -> uuid::Uuid::new_v4(); safe-stringify -> serde_json.
//   - ws server (`ws`) -> tokio-tungstenite (needs an async runtime).
//   - pervasive `any` / `<any>` generics -> concrete entity enums/traits.
// =============================================================================


import * as safe from '@oresoftware/safe-stringify';
import * as math from 'mathjs';
import * as uuid from 'uuid';

import {number} from "mathjs";
import {EntitySource} from "./entity-source/source";
import {EntityProcessor} from "./entity-processing/processing";
import {EntitySink} from "./entity-sink/sink";
import {Entity, EntityObserver} from "./abstract/abstract";
import {AbstractMovingEntity} from "./entity-moving/moving";
import {EntityGraphData, HasManyInputConnections, HasManyOutputConnections} from "./abstract/interfaces";
import {bgn, fisherYatesShuffle, makeError} from "./general/general";
import {ExponentialRandomVariable, PoissonRandomVariable, UniformRandomVariable} from "./random-variables/rv";
import {getWebsocketServer} from "./ws-server/ws-server";
import {VisualNode} from "./visual/visual-node";
import {ProgramObserver} from "./observers/program-observer";
import * as des from "./general/time-accrued";
import {doAudit} from "./general/do-audit";
const stepSizeMillis = des.getStepSize();

// export interface MovingEntity {
//     nextTarget: StationaryEntity, // null if in processing
//     timeInSystem: math.BigNumber;
//     runTimeStep(stepSize: math.BigNumber): void;
// }

// TODO: PID controller with simulink
// TODO: continuous simulator - https://www.mathworks.com/help/simulink/ug/modeling-a-continuous-system.html
// TODO: traffic simulator - create grid, each node is intersection,
//  almost all nodes bidirectional (probably misnomer, maybe multidirectional), each car is "born" and gets a random destination
// TODO: For every-time-step, get a separate handle on all moving elements, count time-in-system and compare it with that
// TODO: which was determine by servers (stationary modules)
// TODO: deterministic server (just make it uniform with very small diff between b-a)
// TODO: simulate a Markov chain with X nodes
// TODO: filo queue instead of fifo
// TODO: do uniform distribution vs lognormal
// TODO: have arrivals come according to a schedule (not just a steady state rate)
// TODO: have servers come according to a schedule
// TODO: wait time vs processing time
// TODO: put icons and labels on nodes
// TODO: total throughput by entity type
// TODO: priority queuing, starvation
// TODO: shortest job first
// TODO: balking, jockeying, reneging
// TODO: tell components which time-step is the last one, so they flush all updates/send remainders to subscribers


const run = () => {

  const obs = new ProgramObserver();

  const programEntities = new Map<string, Entity<any>>([

    [
      'A',
      new EntitySource(
        'A',
        {
          turnOffAfterCount: -1,
          rv: new UniformRandomVariable({aVal: bgn(200), bVal: bgn(500)})
          // rv: new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
        }
      ).subscribe(obs)
    ],

    [
      'B', new EntityProcessor(
      'B',
      {
        rv: new UniformRandomVariable({aVal: bgn(300), bVal: bgn(500)})
        // rv: new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
      }
    ).subscribe(obs)
    ],

    [
      'C', new EntityProcessor(
      'C',
      {
        rv: new UniformRandomVariable({aVal: bgn(300), bVal: bgn(500)})
        // rv: new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
      }
    ).subscribe(obs)
    ],

    [
      'D',

      new EntityProcessor(
        'D',
        {
          rv: new UniformRandomVariable({aVal: bgn(100), bVal: bgn(500)})
          // rv: new ExponentialRandomVariable({lambda: bgn(math.divide(1, 10)), timeStep: stepSizeMillis})
        }
      ).subscribe(obs)
    ],


    [
      'E', new EntitySink(
      'E',
      new PoissonRandomVariable(),
      {}
    ).subscribe(obs)
    ],

  ]);


  for (const [sourceId, targetId] of [['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']]) {

    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId);

    source.addOutConnection(target as any);

  }


  console.log(
    programEntities.get('A')
  )

  const programList = Array.from(programEntities);

  const finalize = () => {

    // for (const b of obs.movingEntities) {
    //   // console.log(b);
    // }

    let i = 0;
    for (const e of programEntities.values()) {
      i++;
      console.log(i, i, i, i, i, i, i, i, i, '**************************************')
      console.log(e.getWithComputedProperties());
    }

    console.log('obs.movingEntities.size:', obs.movingEntities.size)
  };


  // TODO: every 100 or so iterations, do a process.nextTick or setImmediate, to read from I/O in case
  // user wants to speed up or slow down simulation

  const runAll = (i: number) => {

    des.bumpTimeAccruedByTimeStep(stepSizeMillis);

    for (const [k, v] of fisherYatesShuffle(programList)) {
      console.log(i);
      v.doTimeStep(stepSizeMillis);
    }

    if (i > 20000) {

      (global as any).turnOffSources = true;
      runAfterSourcesOff(102);

    } else {

      i % 100 !== 0 ? runAll(i + 1): setImmediate(() => {
        runAll(i + 1);
      });

    }

  };

  const runAfterSourcesOff = (i: number) => {

    des.bumpTimeAccruedByTimeStep(stepSizeMillis);
    console.log('doing the audit:')
    doAudit();

    for (const [k, v] of fisherYatesShuffle(programList)) {

      // TODO: remove v from the list if it's, for example, a source, that's done emitting
      console.log(i);
      v.doTimeStep(stepSizeMillis);
    }

    const nextiVal = i+1;

    if (i === 20000) {
      finalize();
    } else {
      i % 100 !== 0 ? runAfterSourcesOff(nextiVal): setImmediate(() => {
        runAfterSourcesOff(nextiVal);
      });
    }
  };

  runAll(102);

}

run();
