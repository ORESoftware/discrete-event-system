'use strict';

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
