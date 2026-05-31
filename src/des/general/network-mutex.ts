'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/network-mutex.rs  (module des::general::network_mutex)
// 1:1 file move. A distributed-mutex DES: workers spawn lock request/release child tokens.
//
// Declarations → Rust:
//   type MutexWorkState / MutexChildState (string-literal unions) -> enums (match on variant)
//   interface MutexWorkItem/MutexChildToken/LockRequest/Grant/ReleaseToken extends StatefulToken<S>
//                                       -> structs carrying a state enum + impl Token (no inheritance)
//   interface NetworkMutex*Opts/*Stats/*Result/TraceEvent/MutexSourceSpec -> structs
//   class MutexWorkSourceStation/NetworkMutexLockServiceStation/MutexCompletionSinkStation
//         extends DESStation                            -> structs + impl DESStation trait
//   class NetworkMutexWorkerStation extends CompositeDESStation -> struct + impl composite trait
//   class MutexQueueSubstation/MutexProcessorSubstation (private) -> structs + impl
//   fn buildNetworkMutexStations / runNetworkMutexSimulation + helpers -> fns
//
// Conversion notes (file-specific):
//   - StatefulToken<S> + transitionToken/makeStatefulToken/spawnStatefulChildToken model an
//     explicit state machine -> in Rust the token holds a state enum and transitions are
//     `self.state = NewState` (validate with match); the string `*_CHANNEL` consts -> &'static str or enum.
//   - the FIFO of held work + pending grants/queue -> `VecDeque<_>` struct fields.
//   - `new Set(...).size === len` dedupe check in a validator -> `HashSet` cardinality compare.
//   - CompositeDESStation (sub-stations) -> a struct owning child station structs + impl trait.
// =============================================================================
// general/network-mutex.ts -- a network mutex as DES stations + child tokens.
//
// Model:
//   Source -> Station A (lock-aware worker) -> Station C (sink)
//                 | request/release child tokens
//                 v
//             Station B (lock service)
//
// The real item stays in Station A's internal FIFO until a child
// LockRequestToken is granted by Station B. A then processes the item while the
// lock is held, sends a LockReleaseToken, and only then emits the item onward.
// This gives us "request spawning request" semantics without needing hidden
// global events or a special scheduler.
// =============================================================================

import {
  CompositeDESStation,
  DESStation,
  StatefulToken,
  Token,
  failedValidationChecks,
  intrinsicCheck,
  makeStatefulToken,
  runIterativeDES,
  spawnStatefulChildToken,
  transitionToken,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

export const MUTEX_WORK_CHANNEL = 'work';
export const MUTEX_DONE_CHANNEL = 'done';
export const MUTEX_REQUEST_CHANNEL = 'lock-request';
export const MUTEX_GRANT_CHANNEL = 'lock-grant';
export const MUTEX_RELEASE_CHANNEL = 'lock-release';
const MUTEX_LOCKED_WORK_CHANNEL = 'locked-work';

export type MutexWorkState =
  | 'created'
  | 'queued'
  | 'waiting-lock'
  | 'lock-granted'
  | 'processing'
  | 'releasing'
  | 'completed';

export type MutexChildState = 'spawned' | 'queued' | 'granted' | 'released' | 'accepted' | 'invalid';

export interface MutexWorkItem extends StatefulToken<MutexWorkState> {
  kind: 'mutex-work';
  itemId: string;
  createdTick: number;
  payload?: unknown;
  lock?: {
    requestId: string;
    requestedTick: number;
    grantedTick?: number;
    processingStartedTick?: number;
    processingFinishedTick?: number;
    releasedTick?: number;
  };
}

export interface MutexChildToken extends StatefulToken<MutexChildState> {
  parentKind: 'mutex-work';
  parentItemId: string;
  tokenId: string;
  ownerId: string;
  createdTick: number;
}

export interface LockRequestToken extends MutexChildToken {
  kind: 'lock-request';
}

export interface LockGrantToken extends MutexChildToken {
  kind: 'lock-grant';
  grantedTick: number;
  serviceRequestQueuedTick: number;
}

export interface LockReleaseToken extends MutexChildToken {
  kind: 'lock-release';
  releasedTick: number;
}

export interface NetworkMutexTraceEvent {
  tick: number;
  stationId: string;
  event:
    | 'work-arrived'
    | 'request-spawned'
    | 'request-queued'
    | 'grant-scheduled'
    | 'grant-received'
    | 'processing-started'
    | 'processing-finished'
    | 'release-spawned'
    | 'release-accepted'
    | 'work-completed'
    | 'invalid-release';
  itemId?: string;
  childTokenId?: string;
  detail?: string;
}

export interface MutexSourceSpec {
  count: number;
  interarrivalTicks: number;
  firstArrivalTick?: number;
}

export interface NetworkMutexLockServiceOpts {
  /** Ticks from lock-service decision to grant token arrival at Station A. */
  grantDelayTicks?: number;
}

export interface NetworkMutexWorkerOpts {
  processingTicks: number;
}

export interface NetworkMutexSimulationOpts {
  source?: MutexSourceSpec;
  worker?: NetworkMutexWorkerOpts;
  lock?: NetworkMutexLockServiceOpts;
  maxTicks?: number;
}

export interface NetworkMutexLockStats {
  grantCount: number;
  releaseCount: number;
  invalidReleaseCount: number;
  finalHolderItemId?: string;
  waitingRequests: number;
  maxWaitQueue: number;
  meanServiceQueueWaitTicks: number;
  lockHeldTicks: number;
  utilization: number;
}

export interface NetworkMutexWorkerStats {
  arrived: number;
  completed: number;
  finalQueue: number;
  maxQueue: number;
  meanQueueWaitTicks: number;
  meanLockWaitTicks: number;
  meanTimeInSystemTicks: number;
  childRequestsSpawned: number;
  childReleasesSpawned: number;
}

export interface NetworkMutexSimulationResult {
  generated: number;
  completed: number;
  totalTicks: number;
  worker: NetworkMutexWorkerStats;
  lock: NetworkMutexLockStats;
  completedItems: MutexWorkItem[];
  trace: NetworkMutexTraceEvent[];
  invariantViolations: string[];
}

interface QueuedWork {
  item: MutexWorkItem;
  enqueuedTick: number;
}

interface ActiveWork {
  item: MutexWorkItem;
  grant: LockGrantToken;
  remainingTicks: number;
}

interface LockedWorkToken extends Token {
  item: MutexWorkItem;
  grant: LockGrantToken;
}

interface PendingGrant {
  grant: LockGrantToken;
  deliverAtTick: number;
}

interface CurrentHolder {
  request: LockRequestToken;
  acquiredTick: number;
}

function trace(
  log: NetworkMutexTraceEvent[],
  tick: number,
  stationId: string,
  event: NetworkMutexTraceEvent['event'],
  itemId?: string,
  childTokenId?: string,
  detail?: string,
): void {
  log.push({tick, stationId, event, itemId, childTokenId, detail});
}

export class MutexWorkSourceStation extends DESStation {
  private readonly spec: Required<MutexSourceSpec>;
  private tick = 0;
  private emitted = 0;

  constructor(id: string, spec: MutexSourceSpec) {
    super(id);
    this.spec = {
      firstArrivalTick: spec.firstArrivalTick ?? 0,
      count: spec.count,
      interarrivalTicks: spec.interarrivalTicks,
    };
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange('network-mutex', `${this.id}.count`, this.spec.count, 0, 1e9);
    Preconditions.integerInRange('network-mutex', `${this.id}.interarrivalTicks`, this.spec.interarrivalTicks, 1, 1e9);
    Preconditions.integerInRange('network-mutex', `${this.id}.firstArrivalTick`, this.spec.firstArrivalTick, 0, 1e9);
  }

  override hasWork(): boolean {
    return this.emitted < this.spec.count;
  }

  runTimeStep(): void {
    const nextArrival = this.spec.firstArrivalTick + this.emitted * this.spec.interarrivalTicks;
    if (this.emitted < this.spec.count && this.tick >= nextArrival) {
      const itemId = `item-${this.emitted + 1}`;
      const base = makeStatefulToken<MutexWorkState>({
        kind: 'mutex-work',
        tokenId: `work:${itemId}`,
        initialState: 'created',
        tick: this.tick,
        stationId: this.id,
      });
      const item: MutexWorkItem = {
        ...base,
        kind: 'mutex-work',
        itemId,
        createdTick: this.tick,
      };
      this.emitted++;
      this.emit(item, MUTEX_WORK_CHANNEL);
    }
    this.tick++;
  }

  emittedCount(): number {
    return this.emitted;
  }
}

export class NetworkMutexLockServiceStation extends DESStation {
  private readonly grantDelayTicks: number;
  private readonly waitQueue: LockRequestToken[] = [];
  private readonly pendingGrants: PendingGrant[] = [];
  private holder?: CurrentHolder;
  private tick = 0;
  private maxWaitQueue = 0;
  private totalServiceQueueWaitTicks = 0;
  private grants = 0;
  private releases = 0;
  private invalidReleases = 0;
  private lockHeldTicks = 0;

  constructor(id: string, opts: NetworkMutexLockServiceOpts = {}, private readonly events: NetworkMutexTraceEvent[] = []) {
    super(id);
    this.grantDelayTicks = opts.grantDelayTicks ?? 2;
    this.addValidator(intrinsicCheck<NetworkMutexLockServiceStation>({
      name: 'network-mutex.lock.single-holder',
      group: 'network-mutex',
      predicate: s => s.pendingGrantCount() <= 1 || s.currentHolder() !== undefined,
      expected: 'at most one outstanding grant for the current holder',
      observedFn: s => `holder=${s.currentHolder() ?? 'none'}, pendingGrants=${s.pendingGrantCount()}`,
    }));
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange('network-mutex', `${this.id}.grantDelayTicks`, this.grantDelayTicks, 0, 1e9);
  }

  override hasWork(): boolean {
    return super.hasWork() || this.waitQueue.length > 0 || this.pendingGrants.length > 0;
  }

  runTimeStep(): void {
    for (const release of this.drain<LockReleaseToken>(MUTEX_RELEASE_CHANNEL)) {
      if (this.holder && release.tokenId === this.holder.request.tokenId) {
        transitionToken(release, 'accepted',
          {tick: this.tick, stationId: this.id, event: 'release-accepted'});
        this.lockHeldTicks += Math.max(0, this.tick - this.holder.acquiredTick);
        this.holder = undefined;
        this.releases++;
        trace(this.events, this.tick, this.id, 'release-accepted', release.parentItemId, release.tokenId);
      } else {
        transitionToken(release, 'invalid',
          {tick: this.tick, stationId: this.id, event: 'invalid-release'});
        this.invalidReleases++;
        trace(this.events, this.tick, this.id, 'invalid-release', release.parentItemId, release.tokenId);
      }
    }

    for (const req of this.drain<LockRequestToken>(MUTEX_REQUEST_CHANNEL)) {
      transitionToken(req, 'queued',
        {tick: this.tick, stationId: this.id, event: 'request-queued'});
      this.waitQueue.push(req);
      this.maxWaitQueue = Math.max(this.maxWaitQueue, this.waitQueue.length);
      trace(this.events, this.tick, this.id, 'request-queued', req.parentItemId, req.tokenId);
    }

    if (!this.holder && this.waitQueue.length > 0) {
      const req = this.waitQueue.shift()!;
      this.holder = {request: req, acquiredTick: this.tick};
      this.totalServiceQueueWaitTicks += Math.max(0, this.tick - req.createdTick);
      const grantBase = spawnStatefulChildToken<MutexChildState>(req, {
        kind: 'lock-grant',
        tokenId: `${req.tokenId}:grant`,
        initialState: 'granted',
        tick: this.tick,
        stationId: this.id,
        event: 'grant-scheduled',
      });
      const grant: LockGrantToken = {
        ...grantBase,
        kind: 'lock-grant',
        parentKind: 'mutex-work',
        parentItemId: req.parentItemId,
        tokenId: req.tokenId,
        ownerId: req.ownerId,
        createdTick: this.tick,
        grantedTick: this.tick,
        serviceRequestQueuedTick: req.createdTick,
      };
      this.pendingGrants.push({grant, deliverAtTick: this.tick + this.grantDelayTicks});
      this.grants++;
      trace(this.events, this.tick, this.id, 'grant-scheduled', req.parentItemId, req.tokenId,
        `deliverAt=${this.tick + this.grantDelayTicks}`);
    }

    const due: PendingGrant[] = [];
    const keep: PendingGrant[] = [];
    for (const pg of this.pendingGrants) {
      if (pg.deliverAtTick <= this.tick) due.push(pg);
      else keep.push(pg);
    }
    this.pendingGrants.length = 0;
    this.pendingGrants.push(...keep);
    for (const pg of due) this.emit(pg.grant, MUTEX_GRANT_CHANNEL);
    this.tick++;
  }

  stats(totalTicks: number): NetworkMutexLockStats {
    return {
      grantCount: this.grants,
      releaseCount: this.releases,
      invalidReleaseCount: this.invalidReleases,
      finalHolderItemId: this.holder?.request.parentItemId,
      waitingRequests: this.waitQueue.length,
      maxWaitQueue: this.maxWaitQueue,
      meanServiceQueueWaitTicks: this.totalServiceQueueWaitTicks / Math.max(1, this.grants),
      lockHeldTicks: this.lockHeldTicks,
      utilization: this.lockHeldTicks / Math.max(1, totalTicks),
    };
  }

  currentHolder(): string | undefined {
    return this.holder?.request.parentItemId;
  }

  pendingGrantCount(): number {
    return this.pendingGrants.length;
  }
}

interface MutexQueueSubstationStats {
  arrived: number;
  finalQueue: number;
  maxQueue: number;
  meanQueueWaitTicks: number;
  meanLockWaitTicks: number;
  childRequestsSpawned: number;
}

class MutexQueueSubstation extends DESStation {
  private readonly queue: QueuedWork[] = [];
  private outstanding?: LockRequestToken;
  private tick = 0;
  private maxQueue = 0;
  private queueArea = 0;
  private arrived = 0;
  private childRequests = 0;
  private totalQueueWaitTicks = 0;
  private totalLockWaitTicks = 0;

  constructor(id: string, private readonly ownerId: string, private readonly events: NetworkMutexTraceEvent[] = []) {
    super(id);
  }

  override hasWork(): boolean {
    return super.hasWork() || this.queue.length > 0 || !!this.outstanding;
  }

  runTimeStep(): void {
    for (const item of this.drain<MutexWorkItem>(MUTEX_WORK_CHANNEL)) {
      transitionToken(item, 'queued', {tick: this.tick, stationId: this.id, event: 'work-arrived'});
      this.queue.push({item, enqueuedTick: this.tick});
      this.arrived++;
      this.maxQueue = Math.max(this.maxQueue, this.queue.length);
      trace(this.events, this.tick, this.id, 'work-arrived', item.itemId);
    }

    for (const grant of this.drain<LockGrantToken>(MUTEX_GRANT_CHANNEL)) {
      if (!this.outstanding || grant.tokenId !== this.outstanding.tokenId || this.queue.length === 0) continue;
      const queued = this.queue.shift()!;
      transitionToken(queued.item, 'lock-granted', {tick: this.tick, stationId: this.id, event: 'grant-received'});
      queued.item.lock = {
        ...(queued.item.lock ?? {requestId: grant.tokenId, requestedTick: this.outstanding.createdTick}),
        grantedTick: this.tick,
      };
      this.totalQueueWaitTicks += Math.max(0, this.tick - queued.enqueuedTick);
      this.totalLockWaitTicks += Math.max(0, this.tick - this.outstanding.createdTick);
      this.outstanding = undefined;
      this.emit({item: queued.item, grant}, MUTEX_LOCKED_WORK_CHANNEL);
      trace(this.events, this.tick, this.id, 'grant-received', queued.item.itemId, grant.tokenId);
    }

    if (!this.outstanding && this.queue.length > 0) {
      const head = this.queue[0].item;
      const tokenId = `${this.ownerId}:${head.itemId}:lock`;
      const reqBase = spawnStatefulChildToken<MutexChildState>(head, {
        kind: 'lock-request',
        tokenId,
        initialState: 'spawned',
        tick: this.tick,
        stationId: this.id,
        event: 'request-spawned',
      });
      const req: LockRequestToken = {
        ...reqBase,
        kind: 'lock-request',
        parentKind: 'mutex-work',
        parentItemId: head.itemId,
        tokenId,
        ownerId: this.ownerId,
        createdTick: this.tick,
      };
      transitionToken(head, 'waiting-lock', {tick: this.tick, stationId: this.id, event: 'request-spawned'});
      head.lock = {requestId: req.tokenId, requestedTick: req.createdTick};
      this.outstanding = req;
      this.childRequests++;
      this.emit(req, MUTEX_REQUEST_CHANNEL);
      trace(this.events, this.tick, this.id, 'request-spawned', head.itemId, req.tokenId);
    }

    this.queueArea += this.queue.length;
    this.maxQueue = Math.max(this.maxQueue, this.queue.length);
    this.tick++;
  }

  stats(completed: number): MutexQueueSubstationStats {
    return {
      arrived: this.arrived,
      finalQueue: this.queue.length + (this.outstanding ? 1 : 0),
      maxQueue: this.maxQueue,
      meanQueueWaitTicks: this.totalQueueWaitTicks / Math.max(1, completed),
      meanLockWaitTicks: this.totalLockWaitTicks / Math.max(1, completed),
      childRequestsSpawned: this.childRequests,
    };
  }
}

class MutexProcessorSubstation extends DESStation {
  private readonly ready: LockedWorkToken[] = [];
  private active?: ActiveWork;
  private tick = 0;
  private completed = 0;
  private childReleases = 0;
  private totalTimeInSystemTicks = 0;
  private readonly completedItems: MutexWorkItem[] = [];

  constructor(
    id: string,
    private readonly ownerId: string,
    private readonly processingTicks: number,
    private readonly events: NetworkMutexTraceEvent[] = [],
  ) {
    super(id);
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange('network-mutex', `${this.id}.processingTicks`, this.processingTicks, 1, 1e9);
  }

  override hasWork(): boolean {
    return super.hasWork() || this.ready.length > 0 || !!this.active;
  }

  runTimeStep(): void {
    this.ready.push(...this.drain<LockedWorkToken>(MUTEX_LOCKED_WORK_CHANNEL));

    if (!this.active && this.ready.length > 0) {
      const next = this.ready.shift()!;
      transitionToken(next.item, 'processing', {tick: this.tick, stationId: this.id, event: 'processing-started'});
      next.item.lock = {
        ...(next.item.lock ?? {requestId: next.grant.tokenId, requestedTick: next.grant.serviceRequestQueuedTick}),
        processingStartedTick: this.tick,
      };
      this.active = {item: next.item, grant: next.grant, remainingTicks: this.processingTicks};
      trace(this.events, this.tick, this.id, 'processing-started', next.item.itemId, next.grant.tokenId);
    }

    if (this.active) {
      this.active.remainingTicks -= 1;
      if (this.active.remainingTicks <= 0) {
        const item = this.active.item;
        const tokenId = this.active.grant.tokenId;
        transitionToken(item, 'releasing', {tick: this.tick, stationId: this.id, event: 'processing-finished'});
        item.lock = {
          ...(item.lock ?? {requestId: tokenId, requestedTick: this.active.grant.serviceRequestQueuedTick}),
          processingFinishedTick: this.tick,
          releasedTick: this.tick,
        };
        const releaseBase = spawnStatefulChildToken<MutexChildState>(item, {
          kind: 'lock-release',
          tokenId: `${tokenId}:release`,
          initialState: 'released',
          tick: this.tick,
          stationId: this.id,
          event: 'release-spawned',
        });
        const release: LockReleaseToken = {
          ...releaseBase,
          kind: 'lock-release',
          parentKind: 'mutex-work',
          parentItemId: item.itemId,
          tokenId,
          ownerId: this.ownerId,
          createdTick: this.tick,
          releasedTick: this.tick,
        };
        this.childReleases++;
        this.emit(release, MUTEX_RELEASE_CHANNEL);
        transitionToken(item, 'completed', {tick: this.tick, stationId: this.id, event: 'work-completed'});
        this.emit(item, MUTEX_DONE_CHANNEL);
        this.completed++;
        this.completedItems.push(item);
        this.totalTimeInSystemTicks += Math.max(0, this.tick - item.createdTick + 1);
        trace(this.events, this.tick, this.id, 'processing-finished', item.itemId, tokenId);
        trace(this.events, this.tick, this.id, 'release-spawned', item.itemId, tokenId);
        trace(this.events, this.tick, this.id, 'work-completed', item.itemId, tokenId);
        this.active = undefined;
      }
    }
    this.tick++;
  }

  stats(): Pick<NetworkMutexWorkerStats, 'completed' | 'meanTimeInSystemTicks' | 'childReleasesSpawned' | 'finalQueue'> {
    return {
      completed: this.completed,
      finalQueue: this.ready.length + (this.active ? 1 : 0),
      meanTimeInSystemTicks: this.totalTimeInSystemTicks / Math.max(1, this.completed),
      childReleasesSpawned: this.childReleases,
    };
  }

  get completedItemsView(): readonly MutexWorkItem[] {
    return this.completedItems;
  }
}

export class NetworkMutexWorkerStation extends CompositeDESStation {
  private readonly queueStation: MutexQueueSubstation;
  private readonly processorStation: MutexProcessorSubstation;

  constructor(id: string, opts: NetworkMutexWorkerOpts, events: NetworkMutexTraceEvent[] = []) {
    super(id);
    this.queueStation = this.addSubstation(new MutexQueueSubstation(`${id}:queue`, id, events));
    this.processorStation = this.addSubstation(new MutexProcessorSubstation(`${id}:processor`, id, opts.processingTicks, events));
    this.exposeInput(MUTEX_WORK_CHANNEL, this.queueStation);
    this.exposeInput(MUTEX_GRANT_CHANNEL, this.queueStation);
    this.queueStation.pipe(this.processorStation, MUTEX_LOCKED_WORK_CHANNEL);
    this.exposeOutput(this.queueStation, MUTEX_REQUEST_CHANNEL);
    this.exposeOutput(this.processorStation, MUTEX_RELEASE_CHANNEL);
    this.exposeOutput(this.processorStation, MUTEX_DONE_CHANNEL);
    this.addValidator(intrinsicCheck<NetworkMutexWorkerStation>({
      name: 'network-mutex.worker.no-duplicate-completions',
      group: 'network-mutex',
      predicate: s => new Set(s.completedItemsView.map(x => x.itemId)).size === s.completedItemsView.length,
      expected: 'completed item ids unique',
      observedFn: s => `completed=${s.completedItemsView.map(x => x.itemId).join(',')}`,
    }));
  }

  stats(): NetworkMutexWorkerStats {
    const processor = this.processorStation.stats();
    const queue = this.queueStation.stats(processor.completed);
    return {
      arrived: queue.arrived,
      completed: processor.completed,
      finalQueue: queue.finalQueue + processor.finalQueue,
      maxQueue: queue.maxQueue,
      meanQueueWaitTicks: queue.meanQueueWaitTicks,
      meanLockWaitTicks: queue.meanLockWaitTicks,
      meanTimeInSystemTicks: processor.meanTimeInSystemTicks,
      childRequestsSpawned: queue.childRequestsSpawned,
      childReleasesSpawned: processor.childReleasesSpawned,
    };
  }

  get completedItemsView(): readonly MutexWorkItem[] {
    return this.processorStation.completedItemsView;
  }
}

export class MutexCompletionSinkStation extends DESStation {
  readonly completed: MutexWorkItem[] = [];

  runTimeStep(): void {
    this.completed.push(...this.drain<MutexWorkItem>(MUTEX_DONE_CHANNEL));
  }
}

export function buildNetworkMutexStations(opts: NetworkMutexSimulationOpts = {}): {
  source: MutexWorkSourceStation;
  worker: NetworkMutexWorkerStation;
  lock: NetworkMutexLockServiceStation;
  sink: MutexCompletionSinkStation;
  events: NetworkMutexTraceEvent[];
} {
  const events: NetworkMutexTraceEvent[] = [];
  const source = new MutexWorkSourceStation('source', opts.source ?? {count: 8, interarrivalTicks: 1});
  const worker = new NetworkMutexWorkerStation('station-A', opts.worker ?? {processingTicks: 4}, events);
  const lock = new NetworkMutexLockServiceStation('station-B-lock', opts.lock ?? {grantDelayTicks: 2}, events);
  const sink = new MutexCompletionSinkStation('station-C');

  source.pipe(worker, MUTEX_WORK_CHANNEL);
  worker.pipe(lock, MUTEX_REQUEST_CHANNEL);
  worker.pipe(lock, MUTEX_RELEASE_CHANNEL);
  lock.pipe(worker, MUTEX_GRANT_CHANNEL);
  worker.pipe(sink, MUTEX_DONE_CHANNEL);

  return {source, worker, lock, sink, events};
}

export function runNetworkMutexSimulation(opts: NetworkMutexSimulationOpts = {}): NetworkMutexSimulationResult {
  const {source, worker, lock, sink, events} = buildNetworkMutexStations(opts);
  const maxTicks = opts.maxTicks ?? 10_000;
  const summary = runIterativeDES([source, worker, lock, sink], {shuffle: false, maxTicks});
  const invariantViolations: string[] = [];
  if (summary.reason === 'maxticks') invariantViolations.push(`network mutex reached maxTicks=${maxTicks}`);
  if (sink.completed.length !== source.emittedCount()) {
    invariantViolations.push(`completed ${sink.completed.length} != generated ${source.emittedCount()}`);
  }
  if (lock.stats(summary.ticks).invalidReleaseCount > 0) {
    invariantViolations.push(`invalid releases: ${lock.stats(summary.ticks).invalidReleaseCount}`);
  }
  invariantViolations.push(...failedValidationChecks(summary).map(c => `${c.name}: ${c.details ?? c.observed ?? 'failed'}`));
  return {
    generated: source.emittedCount(),
    completed: sink.completed.length,
    totalTicks: summary.ticks,
    worker: worker.stats(),
    lock: lock.stats(summary.ticks),
    completedItems: sink.completed.slice(),
    trace: events.slice(),
    invariantViolations,
  };
}
