// =============================================================================
// RUST MIGRATION  —  target: src/des/http-server/mod.rs   (module des::http_server)
// 1:1 file move. `index.ts` barrel -> `mod.rs`. Serves the WS demo HTML with the
// current entity formation injected as JSON.
//
// Declarations → Rust:
//   const index (HTML read from disk)   -> `include_str!(...)` or a lazily-read static
//   let server: http.Server             -> shared singleton, NOT a mutable global (see below)
//   const getHTTPServer = (program) =>  -> `fn get_http_server(program: &Program) -> ...`
//
// Conversion notes (file-specific):
//   - Node `http` -> `axum` (or `hyper`); the server is async, so this needs a
//     `tokio` runtime; `server.listen(5000, '0.0.0.0', cb)` -> bind + `axum::serve`.
//   - The mutable module-level `let server` cache -> `OnceLock<...>` / shared app
//     state behind `Arc`, not a reassignable global.
//   - `fs.readFileSync(__dirname + '/.../index-ws.html')` at import time ->
//     `include_str!` (compile-time) or a `OnceLock` lazy read.
//   - `program: any` -> a concrete `Program` type (or generic).
//   - `JSON.stringify({...})` -> `serde_json::to_string` over a `#[derive(Serialize)]` DTO.
//   - `Array.from(programEntities)` (a Map) / `connectionsOut.keys()` -> iterate a
//     `HashMap`; key/value types need `Hash + Eq`; iteration order is unspecified.
//   - `index.replace('{{∆∆∆}}', json)` template fill -> `str::replace` (keep token).
// =============================================================================

import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import {sendRaw} from "../general/general";
import {getEntities} from "../program";

const index = String(fs.readFileSync(path.join(__dirname, '/../../../app/index-ws.html')));
let server: http.Server = <any>null;


export const getHTTPServer = (program: any) => {

  if (server) {
    return server;
  }

  server = http.createServer((req, res) => {

    const programEntities = getEntities(program.stepSize);
    const programList = Array.from(programEntities);

    const json = JSON.stringify({
      formation: programList.map(([k, v]) => {
        return {
          name: k,
          id: k,
          entity: v.entity.getInitialGraphData(),
          iconUrl: v.iconUrl,
          label: v.label,
          connectionsOut: Array.from(v.connectionsOut.keys()).map(v => v.label)
        }
      })
    });

    res.end(
      index.replace(
        '{{∆∆∆}}', json
      )
    );

  });

  server.listen(5000, '0.0.0.0', () => {
    console.log('http server listening on port:', 5000);
  });

  return server;

}
