#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// main-traffic.ts -- small traffic-flow simulation using DES station entities.
//
// The active traffic model builds TrafficCellStation objects at roughly
// one-foot resolution. Traffic cars carry position, velocity, acceleration,
// jerk, reaction delay, and current grid-cell IDs.
// =============================================================================

import {
  buildFiveIntersectionTrafficNetwork,
  runTrafficFlow,
} from './general/network-flow';

function fmt(x: number, digits = 2): string {
  return Number.isFinite(x) ? x.toFixed(digits) : 'n/a';
}

function main(): void {
  const network = buildFiveIntersectionTrafficNetwork();
  const result = runTrafficFlow({
    builtin: 'five-intersection',
    network,
    durationSec: 180,
    dtSec: 0.25,
    seed: 19,
    maxCars: 250,
    spawnRateMultiplier: 1,
    gridCellSizeM: 0.3048,
    reactionTimeSec: 1.0,
    timeHeadwaySec: 1.2,
    maxJerkMps3: 2.5,
  });
  const meanAbsJerkMps3 = meanAbsJerk(result);
  const minLeaderGapM = minLeaderGap(result);

  console.log('# Traffic-flow DES');
  console.log('# TrafficCellStation grid + moving car snapshots; kinematics stepped at dt');
  console.log(`# nodes=${network.nodes.length}, lanes=${network.lanes.length}, sources=${network.sources.length}, cells=${result.cellStats.createdCellStations}`);
  console.log(`# dt=${result.params.dtSec}s, cell=${fmt(result.cellStats.cellSizeM, 4)}m, configured cap=${result.params.maxCars}, max active=${result.maxActiveCars}`);
  console.log('');

  console.log('## Demand and throughput');
  console.log(`  entered cars:         ${result.entered}`);
  console.log(`  exited cars:          ${result.exited}`);
  console.log(`  active at stop:       ${result.finalCars.length}`);
  console.log(`  dropped attempts:     ${result.dropped}`);
  console.log('');

  console.log('## Kinematics');
  console.log(`  mean travel:       ${fmt(result.meanTravelTimeSec, 1)} sec`);
  console.log(`  mean speed:        ${fmt(result.meanSpeedMps, 2)} m/s`);
  console.log(`  mean |jerk|:       ${fmt(meanAbsJerkMps3, 2)} m/s^3`);
  console.log(`  min leader gap:    ${fmt(minLeaderGapM, 3)} m`);
  console.log(`  max cell occup.:   ${result.cellStats.maxCellOccupancy}`);
  console.log('');

  console.log('## Final sample');
  for (const car of result.finalCars.slice(0, 10)) {
    console.log(`  car=${String(car.id).padStart(3)} lane=${car.laneId.padEnd(6)} x=${fmt(car.positionM, 2)}m v=${fmt(car.speedMps, 2)}m/s a=${fmt(car.accelerationMps2, 2)}m/s^2 cells=${car.gridCellIds.slice(0, 3).join('|')}`);
  }
}

function meanAbsJerk(result: ReturnType<typeof runTrafficFlow>): number {
  const jerks = result.trace.flatMap(row => row.cars.map(car => Math.abs(car.jerkMps3)));
  return jerks.length > 0 ? jerks.reduce((sum, x) => sum + x, 0) / jerks.length : 0;
}

function minLeaderGap(result: ReturnType<typeof runTrafficFlow>): number {
  const gaps = result.trace.flatMap(row => row.cars.map(car => car.leaderGapM).filter((x): x is number => x !== undefined));
  return gaps.length > 0 ? Math.min(...gaps) : 0;
}

if (require.main === module) {
  main();
}
