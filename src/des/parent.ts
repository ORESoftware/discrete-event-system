#!/usr/bin/env ts-node
'use strict';


import {getWebsocketServer, wss} from "./ws-server/ws-server";
import {bgn, deJSON, fisherYatesShuffle, sendRaw} from "./general/general";
import * as WebSocket from "ws";
import {HasManyOutputConnections} from "./abstract/interfaces";
import * as safe from '@oresoftware/safe-stringify';
import {getHTTPServer} from "./http-server";
import * as cp from 'child_process';
import * as path from "path";
const childPath = path.resolve(__dirname + '/child.js')
// import log from 'bunion';

const program = {
  stepSize: bgn(500)
};

const httpServer = getHTTPServer(program);

let requestCount = 0;

httpServer.on('request', (a,b) => {
  console.info('server received request:', ++requestCount, a.method, a.url);
});

const wsServer = getWebsocketServer();
let started = false;

wsServer.on('connection', c => {

  c.on('message', deJSON((v: any) => {

    console.log('got a message:', v);

    if (!started && v.start === true) {
      started = true;
      run(c);
    }

  }));
});



const run = (c: WebSocket) => {

  const k = <any>cp.fork(childPath, [], {
     // stdio: ['pipe','pipe', 'pipe']
    stdio: 'pipe'
  });

  k.stdout.pipe(process.stdout);
  k.stderr.pipe(process.stderr);

  k.on('message', (m: any) => {

    console.log('got a message:', m);
    // c.send(JSON.stringify(m));
    c.send(m);

  });

}

