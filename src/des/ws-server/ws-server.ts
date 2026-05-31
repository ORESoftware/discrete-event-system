'use strict';

// RUST MIGRATION:
// - Target: src/des/ws_server/ws_server.rs
// - Replace ws.WebSocketServer with axum websocket routes or tokio-tungstenite on tokio.
// - wss.connections should become shared connection state such as Arc<Mutex<HashSet<ConnectionId>>> plus sender handles, not raw socket objects.
// - safe-stringify maps to serde_json::to_string with explicit serializable structs; send/close handlers should return Result.
// - getWebsocketServer should become an async startup function that binds once and exposes typed broadcast/connection APIs.

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
