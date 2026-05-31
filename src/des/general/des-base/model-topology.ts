'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/model_topology.rs
// - Keep file-for-file. StationGraphTopology becomes a data struct with station
//   and movable vectors.
// - stationGraphTopology can remain a pure module function over ids; if it is
//   represented as a runnable graph projection, wrap it in
//   PureTransform/PureTransformEntity with transform().
// - Prefer explicit string newtypes/enums for ids if this grows beyond
//   diagnostic topology output.
// - Return Result only if future validation is added; today this stays
//   infallible.

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
