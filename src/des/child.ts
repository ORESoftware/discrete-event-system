'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/child.rs   (fn main)
// 1:1 file move. Forked worker process: runs the simulation and streams
// batched updates back to the parent over IPC.
//
// Conversion notes (file-specific):
//   - Spawned via cp.fork -> compile as a separate binary; process.send / IPC
//     -> an ipc channel (e.g. ipc-channel) or stdout pipe to the parent.
//   - Top-level `program` state + setTimeout batching loop -> fn main() owning
//     the state with a timer/loop.
//   - process.env.step_size -> std::env::var.
//   - `<any[]>` batch payloads -> a concrete message enum; safe-stringify ->
//     serde_json.
//   - ws (`ws`) -> tokio-tungstenite.
// =============================================================================

import {getWebsocketServer, wss} from "./ws-server/ws-server";
import {bgn, deJSON, fisherYatesShuffle} from "./general/general";
import {VisualNode} from "./visual/visual-node";
import * as WebSocket from "ws";
import {getEntities} from "./program";
import * as safe from '@oresoftware/safe-stringify';

const makeTimeout = () => {
  return setTimeout(() => {
    if (program.batches.length > 0) {
      sendMessageToParent(program.batches);
    }
  }, 2000);
}

const program = {
  stepSize: bgn(parseInt(process.env.step_size || '500')),
  batches: <any[]>[],
  to: makeTimeout(),
  stop: false,
  turnOffSources: false,
  running: false
};

const sendMessageToParent = (m: any) => {
  program.batches.push(m);
  if (program.batches.length > 10) {
    const v = program.batches;
    program.batches = [];
    sendBatchMessageToParent(v);
  }
};

const sendBatchMessageToParent = (m: any[]) => {
  clearTimeout(program.to);
  program.to = makeTimeout();
  if(!process.send){
    throw new Error('missing process.send method.')
  }
  process.send(safe.stringify(m));
};

// process.on('SIGSTOP', () => {
//   console.log('process paused.');
// });

process.on('SIGCONT', () => {
  console.log('process re-started.');
});

process.on('message', (m: any) => {

  console.log('child received this message:', m);

  if (!m) {
    console.error('message was falsy:', m);
    return;
  }

  if (Number.isInteger(m.stepSize)) {
    program.stepSize = m.stepSize;
  }

  if (m.turnOffSources === true) {
    (global as any).turnOffSources = true;
  }

  if (m.runRun === true) {
    if (!program.running) {
      program.running = true;
      run();
    }

  }

});


const run = () => {

  const programEntities = getEntities(program.stepSize);

  for (const [_, v] of programEntities) {
    v.sub((type, m) => {
      // console.log('sending message:', type,m);
      // sendMessageToParent({type, value: JSON.stringify(m)});
      console.log((m as any).constructor);
      try{
        if(m.toJSON){
          console.log(m.toJSON.toString())
        } else {
          console.log('missing toJSON:', m);
        }

      }catch (err){
        console.error(err);
      }

      sendMessageToParent({type, m});
    });
  }

  for (const [sourceId, targetId] of [['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']]) {

    {
      const source = programEntities.get(sourceId) as VisualNode;
      const target = programEntities.get(targetId) as VisualNode;

      (source.entity as any).addOutConnection(target.entity);
    }


    {
      const source = programEntities.get(sourceId) as VisualNode;
      const target = programEntities.get(targetId) as VisualNode;

      source.addVisualConnectionOut(target);
    }

  }

  const programList = Array.from(programEntities);

  for(const [k,v] of programList){
    v.doValidationBeforeRun();
  }

  const doLoop = () => {

    for (let i = 0; i < 10; i++) { // 1000 time steps

      if (program.stop) {
        console.warn('stop flag was flipped, breaking out of loop.')
        break;
      }

      for (const [k, v] of fisherYatesShuffle(programList)) {

        if (program.stop) {
          console.warn('stop flag was flipped, breaking out of loop.')
          break;
        }

        v.entity.doTimeStep(program.stepSize);
      }

    }


    (global as any).turnOffSources = true;

    for (let i = 0; i < 100; i++) { // 1000 time steps

      if (program.stop) {
        console.warn('stop flag was flipped, breaking out of loop.')
        break;
      }

      for (const [k, v] of fisherYatesShuffle(programList)) {

        if (program.stop) {
          console.warn('stop flag was flipped, breaking out of loop.')
          break;
        }

        v.entity.doTimeStep(program.stepSize);
      }

    }


    let i = 0;
    for (const e of programEntities.values()) {
      i++;
      console.log(i, i, i, i, i, i, i, i, i, '**************************************')
      console.log(e.entity.getWithComputedProperties());
    }

    // process.nextTick(doLoop);

  };

  console.log('running loop.');
  doLoop();


}

run();

