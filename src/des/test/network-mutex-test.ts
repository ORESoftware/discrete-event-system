// RUST MIGRATION: Port file-for-file to `tests/network_mutex_test.rs` as integration coverage for network mutex stations and stateful token transitions.
// Test-port notes: translate concurrency/state scenarios into `#[test]` functions returning `Result<()>`; replace helper checks with `assert!`, `assert_eq!`, approximate-float helpers, and deterministic token fixtures.

'use strict';

import {
  buildNetworkMutexStations,
  MUTEX_DONE_CHANNEL,
  MUTEX_GRANT_CHANNEL,
  MUTEX_RELEASE_CHANNEL,
  MUTEX_REQUEST_CHANNEL,
  MUTEX_WORK_CHANNEL,
  NetworkMutexLockServiceStation,
  runNetworkMutexSimulation,
} from '../general/network-mutex';
import {makeStatefulToken, runIterativeDES, spawnStatefulChildToken, transitionToken} from '../general/des-base';

let pass = 0;
let fail = 0;

function check(label: string, ok: boolean, detail?: string): void {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ' -- ' + detail : ''}`);
  ok ? pass++ : fail++;
}

function close(label: string, actual: number, expected: number, tol = 1e-12): void {
  check(label, Math.abs(actual - expected) <= tol,
    `actual=${actual}, expected=${expected}, diff=${Math.abs(actual - expected).toExponential(3)}`);
}

console.log('\nGroup 1 -- Station A obtains lock from Station B before releasing work to Station C');
{
  const r = runNetworkMutexSimulation({
    source: {count: 5, interarrivalTicks: 1},
    worker: {processingTicks: 3},
    lock: {grantDelayTicks: 2},
  });
  check('all generated items complete', r.generated === 5 && r.completed === 5,
    `generated=${r.generated}, completed=${r.completed}`);
  check('completion order remains FIFO',
    r.completedItems.map(x => x.itemId).join(',') === 'item-1,item-2,item-3,item-4,item-5',
    r.completedItems.map(x => x.itemId).join(','));
  check('every item has lock timing metadata',
    r.completedItems.every(x => x.lock?.requestedTick !== undefined && x.lock.grantedTick !== undefined && x.lock.releasedTick !== undefined));
  check('request child token spawned per item', r.worker.childRequestsSpawned === 5);
  check('release child token spawned per item', r.worker.childReleasesSpawned === 5);
  check('lock service grants/releases match completed work',
    r.lock.grantCount === 5 && r.lock.releaseCount === 5,
    `grants=${r.lock.grantCount}, releases=${r.lock.releaseCount}`);
  check('no invariant violations', r.invariantViolations.length === 0, r.invariantViolations.join('; '));
}

console.log('\nGroup 2 -- Contention builds queue and lock wait');
{
  const r = runNetworkMutexSimulation({
    source: {count: 8, interarrivalTicks: 1},
    worker: {processingTicks: 4},
    lock: {grantDelayTicks: 2},
  });
  check('worker queue builds under single mutex contention', r.worker.maxQueue > 1, `maxQueue=${r.worker.maxQueue}`);
  check('mean lock wait includes network grant delay', r.worker.meanLockWaitTicks >= 2,
    `meanLockWait=${r.worker.meanLockWaitTicks}`);
  check('mean time in system exceeds processing time under contention', r.worker.meanTimeInSystemTicks > 4,
    `meanTIS=${r.worker.meanTimeInSystemTicks}`);
  check('lock utilization is nonzero and bounded', r.lock.utilization > 0 && r.lock.utilization <= 1,
    `util=${r.lock.utilization}`);
}

console.log('\nGroup 3 -- Child-token wiring is explicit');
{
  const {source, worker, lock, sink, events} = buildNetworkMutexStations({
    source: {count: 2, interarrivalTicks: 1},
    worker: {processingTicks: 2},
    lock: {grantDelayTicks: 1},
  });
  runIterativeDES([source, worker, lock, sink], {shuffle: false, maxTicks: 100});
  const seq = events.map(e => e.event);
  check('trace includes child request spawn', seq.includes('request-spawned'));
  check('trace includes lock grant receipt', seq.includes('grant-received'));
  check('trace includes child release spawn', seq.includes('release-spawned'));
  check('sink receives completed work', sink.completed.length === 2);
  check('Station A is a composite with queue and processor substations',
    worker.childStations().some(s => s.id.endsWith(':queue')) &&
    worker.childStations().some(s => s.id.endsWith(':processor')),
    worker.childStations().map(s => s.id).join(','));
  const states = sink.completed[0].stateHistory?.map(s => s.to).join('>') ?? '';
  check('parent movable records state transitions through Station A substations',
    states.includes('created>queued>waiting-lock>lock-granted>processing>releasing>completed'),
    states);
  check('worker/lock channels are named protocol channels',
    [MUTEX_WORK_CHANNEL, MUTEX_REQUEST_CHANNEL, MUTEX_GRANT_CHANNEL, MUTEX_RELEASE_CHANNEL, MUTEX_DONE_CHANNEL]
      .join('|') === 'work|lock-request|lock-grant|lock-release|done');
}

console.log('\nGroup 4 -- Lock service rejects bad release tokens');
{
  const events: any[] = [];
  const lock = new NetworkMutexLockServiceStation('lock', {grantDelayTicks: 0}, events);
  lock.take({
    kind: 'lock-release',
    parentKind: 'mutex-work',
    parentItemId: 'orphan',
    tokenId: 'bad-token',
    ownerId: 'station-A',
    createdTick: 0,
    releasedTick: 0,
  }, MUTEX_RELEASE_CHANNEL);
  runIterativeDES([lock], {shuffle: false, maxTicks: 10});
  const stats = lock.stats(1);
  check('invalid release is counted', stats.invalidReleaseCount === 1, `invalid=${stats.invalidReleaseCount}`);
}

console.log('\nGroup 5 -- Faster arrivals have worse queueing than slower arrivals');
{
  const fast = runNetworkMutexSimulation({
    source: {count: 8, interarrivalTicks: 1},
    worker: {processingTicks: 4},
    lock: {grantDelayTicks: 2},
  });
  const slow = runNetworkMutexSimulation({
    source: {count: 8, interarrivalTicks: 8},
    worker: {processingTicks: 4},
    lock: {grantDelayTicks: 2},
  });
  check('fast arrivals produce higher max queue', fast.worker.maxQueue > slow.worker.maxQueue,
    `fast=${fast.worker.maxQueue}, slow=${slow.worker.maxQueue}`);
  check('fast arrivals produce higher mean queue wait', fast.worker.meanQueueWaitTicks > slow.worker.meanQueueWaitTicks,
    `fast=${fast.worker.meanQueueWaitTicks}, slow=${slow.worker.meanQueueWaitTicks}`);
  close('slow run still completes all work', slow.completed, slow.generated, 0);
}

console.log('\nGroup 6 -- Generic smart movable lineage helpers');
{
  const parent = makeStatefulToken<'new' | 'waiting'>({
    kind: 'parent',
    tokenId: 'p1',
    initialState: 'new',
    tick: 0,
    stationId: 'src',
  });
  transitionToken(parent, 'waiting', {tick: 1, stationId: 'A', event: 'queued'});
  const child = spawnStatefulChildToken(parent, {
    kind: 'child-request',
    tokenId: 'c1',
    initialState: 'spawned',
    tick: 2,
    stationId: 'A',
  });
  check('child token points back to parent token',
    child.lineage.parentTokenId === parent.lineage.tokenId,
    `parent=${child.lineage.parentTokenId}`);
  check('child token preserves root causality',
    child.lineage.rootTokenId === parent.lineage.rootTokenId && child.lineage.generation === 1,
    `root=${child.lineage.rootTokenId}, gen=${child.lineage.generation}`);
  check('state transition history records previous state',
    parent.stateHistory?.[1]?.from === 'new' && parent.stateHistory?.[1]?.to === 'waiting',
    JSON.stringify(parent.stateHistory));
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
