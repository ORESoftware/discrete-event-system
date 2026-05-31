'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-network-mutex.rs   (fn main)
// 1:1 file move. Thin runner: distributed-mutex DES (source/worker/lock) with
// invariant checks, prints summary + completion order.
//
// Conversion notes (file-specific):
//   - top-level main() -> fn main(); process.env params (ITEMS, INTERARRIVAL,
//     PROCESSING_TICKS, GRANT_DELAY_TICKS) -> std::env::var.
//   - delegates to general/network-mutex -> use crate::des::general::network_mutex.
// =============================================================================

import {runNetworkMutexSimulation} from './general/network-mutex';

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toFixed(3) : 'n/a';
}

function main(): void {
  const result = runNetworkMutexSimulation({
    source: {count: Number(process.env.ITEMS ?? 10), interarrivalTicks: Number(process.env.INTERARRIVAL ?? 1)},
    worker: {processingTicks: Number(process.env.PROCESSING_TICKS ?? 4)},
    lock: {grantDelayTicks: Number(process.env.GRANT_DELAY_TICKS ?? 2)},
  });

  console.log('Network mutex DES');
  console.log('=================');
  console.log(`generated:              ${result.generated}`);
  console.log(`completed:              ${result.completed}`);
  console.log(`total ticks:            ${result.totalTicks}`);
  console.log(`worker max queue:       ${result.worker.maxQueue}`);
  console.log(`mean queue wait:        ${fmt(result.worker.meanQueueWaitTicks)} ticks`);
  console.log(`mean lock wait:         ${fmt(result.worker.meanLockWaitTicks)} ticks`);
  console.log(`mean time in system:    ${fmt(result.worker.meanTimeInSystemTicks)} ticks`);
  console.log(`child lock requests:    ${result.worker.childRequestsSpawned}`);
  console.log(`child lock releases:    ${result.worker.childReleasesSpawned}`);
  console.log(`lock grants/releases:   ${result.lock.grantCount}/${result.lock.releaseCount}`);
  console.log(`lock max wait queue:    ${result.lock.maxWaitQueue}`);
  console.log(`lock utilization:       ${fmt(result.lock.utilization)}`);
  console.log(`invariants:             ${result.invariantViolations.length === 0 ? 'ok' : result.invariantViolations.join('; ')}`);

  console.log('');
  console.log('Completion order:');
  console.log(result.completedItems.map(x => x.itemId).join(' -> '));

  console.log('');
  console.log('First trace events:');
  for (const e of result.trace.slice(0, 24)) {
    const item = e.itemId ? ` ${e.itemId}` : '';
    const child = e.childTokenId ? ` child=${e.childTokenId}` : '';
    const detail = e.detail ? ` (${e.detail})` : '';
    console.log(`  t=${e.tick.toString().padStart(3)} ${e.stationId.padEnd(14)} ${e.event}${item}${child}${detail}`);
  }
}

main();
