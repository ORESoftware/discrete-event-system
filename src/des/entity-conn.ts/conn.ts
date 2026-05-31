'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-conn.ts/conn.rs  (module des::entity_conn::conn)
// 1:1 file move. Connection configuration options for graph edges.
//
// Declarations → Rust:
//   interface ConnectionOpts -> struct ConnectionOpts { travel_time: f64, is_bidirectional: bool }
//                               (#[derive(Clone, Default)])
//
// Conversion notes (file-specific):
//   - The parent directory name `entity-conn.ts` contains a DOT, which is not a
//     valid Rust module path — rename it to `entity-conn`/`entity_conn` (module
//     `des::entity_conn`) when moving.
//   - `isBidirectional: false` is typed as the literal `false` -> a plain `bool`.
//   - `travelTime: number` -> `f64`.
// =============================================================================

import {Entity} from "../abstract/abstract";
import * as math from "mathjs";
import {EntityGraphData, HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";

export interface ConnectionOpts {
    travelTime: number;
    isBidirectional: false;
}
