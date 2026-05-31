'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/model_topology.rs  (module des::general::des_base::model_topology)
// 1:1 file move. Minimal station-graph topology metadata for the visual/animation layer.
//
// Declarations → Rust:
//   interface StationGraphTopology  -> struct StationGraphTopology { stations: Vec<String>, movables: Vec<String> }
//                                      (#[derive(Clone, Serialize)])
//   fn stationGraphTopology         -> fn station_graph_topology(&[String], &[String]) -> StationGraphTopology
//
// Conversion notes (file-specific):
//   - Trivial DTO; `.slice()` copies -> `.clone()`/`.to_vec()`. No DESStation, no logic.
// =============================================================================

// Shared result metadata for station-graph models. The visual editor and
// animation layer can consume this without knowing model-specific result types.

export interface StationGraphTopology {
  stations: string[];
  movables: string[];
}

export function stationGraphTopology(
  stations: readonly string[],
  movables: readonly string[],
): StationGraphTopology {
  return {stations: stations.slice(), movables: movables.slice()};
}
