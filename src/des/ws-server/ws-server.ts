'use strict';

import * as  WebSocket from 'ws';
import {WebSocketServer} from "ws";
import * as safe from '@oresoftware/safe-stringify';

let server: WebSocket.Server | null = null;

export const wss  ={
  connections: new Set<WebSocket.WebSocket>()
}

export const getWebsocketServer = (): WebSocketServer => {

  if (!server) {
    server = new WebSocket.Server({host: '0.0.0.0', port: 6969});

    server.on('connection', c => {
      console.log('new connection.')

      c.send(safe.stringify({received:true}));

       wss.connections.add(c);

      c.once('close', () => {
        wss.connections.delete(c);
      });

    });

    server.on('close', () => {
       console.log('websocket server closed.')
    });
  }

  return server;

};
