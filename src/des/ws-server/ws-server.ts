'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/ws-server/ws-server.rs   (module des::ws_server::ws_server)
// 1:1 file move. A WebSocket server that tracks live connections for broadcast.
//
// Declarations → Rust:
//   let server: WebSocket.Server | null   -> shared singleton (OnceLock/Arc), not a mutable global
//   const wss = { connections: Set<..> }   -> shared connection registry behind Arc<Mutex<..>>
//   const getWebsocketServer = () =>       -> `fn get_websocket_server() -> ...`
//
// Conversion notes (file-specific):
//   - `ws` -> `tokio-tungstenite`: this is async and REQUIRES a `tokio` runtime;
//     `new WebSocket.Server({host, port})` -> bind a `TcpListener` + accept loop.
//   - `wss.connections: Set<WebSocket>` -> a shared set of per-connection sink
//     handles (`Arc<Mutex<HashMap<ConnId, Sender>>>` / `HashSet`); raw socket
//     objects aren't `Hash`, so key by an id rather than the socket itself.
//   - Event callbacks (`server.on('connection')`, `c.once('close')`) -> spawn an
//     async task per accepted socket and drive its stream in a loop (`FnMut`/`async move`).
//   - module-level `let server` cache -> `OnceLock` / shared state, not reassignable global.
//   - `safe.stringify({received:true})` -> `serde_json::to_string` over a DTO.
// =============================================================================

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

      // ws emits 'error' on a socket; an unhandled 'error' event throws and
      // would take down the whole process. Log it and drop the connection.
      c.on('error', (err) => {
        console.warn(`[ws-server] connection error: ${(err as Error)?.message}`);
        wss.connections.delete(c);
      });

    });

    server.on('error', (err) => {
      console.error(`[ws-server] server error: ${(err as Error)?.message}`);
    });

    server.on('close', () => {
       console.log('websocket server closed.')
       // Drop stale socket references and allow a clean re-create on next call.
       wss.connections.clear();
       server = null;
    });
  }

  return server;

};
