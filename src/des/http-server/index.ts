// RUST MIGRATION:
// - Target: src/des/http_server/mod.rs
// - Replace Node http singleton with an axum Router plus a tokio listener; handlers should return Result<impl IntoResponse, ServerError>.
// - Replace program:any with a typed ProgramState/ProgramHandle, likely Arc-owned if shared across async handlers.
// - Template loading should use include_str! for static HTML or PathBuf/std::fs at startup, not per-request loose fs reads.
// - JSON formation payload should become serde structs; send_raw/get_entities boundaries need explicit Rust traits or typed functions.

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
