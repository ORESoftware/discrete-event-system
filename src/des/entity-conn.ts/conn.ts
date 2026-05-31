'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_conn_ts/conn.rs
// - ConnectionOpts becomes a plain struct with typed fields. The odd
//   `entity-conn.ts` directory name should be normalized to a Rust-safe module
//   name during the file-for-file move.
// - Imported Entity/graph endpoint types are unused here today; decide whether
//   this module owns connection configuration only or should fold into
//   abstract/entity_connection.rs before adding Rust APIs.

import {Entity} from "../abstract/abstract";
import * as math from "mathjs";
import {EntityGraphData, HasManyInputConnections, HasManyOutputConnections} from "../abstract/interfaces";

export interface ConnectionOpts {
    travelTime: number;
    isBidirectional: false;
}
