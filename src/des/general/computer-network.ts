// RUST MIGRATION: target module src/des/general/computer_network.rs.
// RUST MIGRATION: Network specs, snapshots, metrics, and result interfaces become serde structs; node/link/protocol/routing string unions become enums.
// RUST MIGRATION: NetworkPacket, NetworkStation, node/delay/link stations, and ComputerNetworkStation become structs implementing Token, Entity, and Station traits rather than inheritance.
// RUST MIGRATION: Map/Set adjacency, flow, queue, and routing tables map to HashMap/HashSet/VecDeque; preserve deterministic ordering explicitly where reports depend on it.
// RUST MIGRATION: runComputerNetworkSimulation is graph-visible and should be a PureTransform entry struct; validation and normalization return Result.
'use strict';

// =============================================================================
// general/computer-network.ts -- packet-switched computer networking as DES.
//
// This is the network-engineering counterpart to the queueing and flow models:
//
//   - NetworkHostStation / NetworkRouterStation / NetworkSwitchStation are
//     stationary entities. They hold queues, routing state, counters, and local
//     forwarding rules.
//   - NetworkLinkStation is a stationary directed link. It models bandwidth
//     serialization, propagation latency, queue limits, utilization, and cost.
//   - NetworkPacket is the movable entity. It carries per-flow timing, route,
//     hop, cost, and drop/delivery state through the topology.
//
// A JSON topology can therefore be run directly as a DES without changing the
// base framework.
// =============================================================================

import {BasicMovingEntity} from '../entity-moving/moving';
import {
  assertNoValidationFailures,
  DESStation,
  Token,
  intrinsicCheck,
  runIterativeDES,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

const MODEL = 'computer-network';

export type NetworkNodeKind = 'host' | 'router' | 'switch';
export type NetworkRoutingMetric = 'latency' | 'cost' | 'hop';
export type NetworkProtocol = 'raw' | 'tcp' | 'udp' | 'http';
export type PacketDropReason =
  | 'node-queue-overflow'
  | 'link-queue-overflow'
  | 'no-route'
  | 'ttl-expired'
  | 'max-packets-in-system';

export interface NetworkNodeSpec {
  id: string;
  kind: NetworkNodeKind;
  /** Packets per second that the node can forward/consume. Defaults by kind. */
  forwardingRatePps?: number;
  /** FIFO input buffer limit. Defaults by kind. */
  queueLimitPackets?: number;
}

export interface NetworkLinkSpec {
  id: string;
  from: string;
  to: string;
  bandwidthMbps: number;
  latencyMs: number;
  /** Monetary or abstract cost charged per megabyte traversing the link. */
  costPerMb?: number;
  /** Max packets scheduled on this link at once. Defaults to 64. */
  queueLimitPackets?: number;
  /** If true, creates a reverse directed link with identical parameters. */
  bidirectional?: boolean;
}

export interface NetworkFlowSpec {
  id: string;
  source: string;
  destination: string;
  /** Application/transport model used to derive headers and startup delay. Defaults to raw. */
  protocol?: NetworkProtocol;
  /** Payload bytes before protocol overhead. */
  ratePps: number;
  packetSizeBytes: number;
  startMs?: number;
  endMs?: number;
  maxPackets?: number;
  ttlHops?: number;
}

export interface ComputerNetworkProblem {
  nodes: NetworkNodeSpec[];
  links: NetworkLinkSpec[];
  flows: NetworkFlowSpec[];
  durationMs: number;
  dtMs: number;
  routingMetric?: NetworkRoutingMetric;
  drainAfterSourcesMs?: number;
  maxPacketsInSystem?: number;
  sampleEveryMs?: number;
}

export interface NetworkPacketSnapshot {
  packetId: number;
  flowId: string;
  protocol: NetworkProtocol;
  source: string;
  destination: string;
  payloadBytes: number;
  sizeBytes: number;
  createdAtMs: number;
  deliveredAtMs?: number;
  droppedAtMs?: number;
  dropReason?: PacketDropReason;
  currentNodeId?: string;
  currentLinkId?: string;
  hops: string[];
  cost: number;
}

export interface NetworkFlowStats {
  id: string;
  protocol: NetworkProtocol;
  source: string;
  destination: string;
  generatedPackets: number;
  deliveredPackets: number;
  droppedPackets: number;
  deliveryRatio: number;
  generatedBytes: number;
  deliveredBytes: number;
  offeredLoadMbps: number;
  throughputMbps: number;
  goodputMbps: number;
  meanLatencyMs: number;
  p95LatencyMs: number;
  meanTimeInSystemMs: number;
  p95TimeInSystemMs: number;
  totalCost: number;
  meanCostPerDeliveredPacket: number;
}

export interface NetworkNodeStats {
  id: string;
  kind: NetworkNodeKind;
  forwardingRatePps: number;
  queueLimitPackets: number;
  receivedPackets: number;
  forwardedPackets: number;
  deliveredPackets: number;
  droppedPackets: number;
  finalQueue: number;
  maxQueue: number;
  avgQueue: number;
  meanQueueDelayMs: number;
  maxQueueDelayMs: number;
}

export interface NetworkLinkStats {
  id: string;
  from: string;
  to: string;
  bandwidthMbps: number;
  latencyMs: number;
  costPerMb: number;
  queueLimitPackets: number;
  enqueuedPackets: number;
  deliveredPackets: number;
  droppedPackets: number;
  transmittedBytes: number;
  throughputMbps: number;
  utilization: number;
  finalInFlight: number;
  maxInFlight: number;
  avgInFlight: number;
  meanQueueDelayMs: number;
  maxQueueDelayMs: number;
  meanTimeOnLinkMs: number;
  maxTimeOnLinkMs: number;
  totalCost: number;
}

export interface NetworkTimeSample {
  tMs: number;
  generatedPackets: number;
  deliveredPackets: number;
  droppedPackets: number;
  activePackets: number;
  nodeQueues: Record<string, number>;
  linkInFlight: Record<string, number>;
  linkUtilization: Record<string, number>;
}

export interface NetworkBottleneckReport {
  id: string;
  kind: 'node' | 'link';
  score: number;
  reason: string;
  utilization?: number;
  avgQueue: number;
  maxQueue: number;
  droppedPackets: number;
  meanQueueDelayMs: number;
}

export interface ComputerNetworkResult {
  generatedPackets: number;
  deliveredPackets: number;
  droppedPackets: number;
  activePackets: number;
  maxActivePackets: number;
  deliveryRatio: number;
  offeredLoadMbps: number;
  throughputMbps: number;
  goodputMbps: number;
  meanLatencyMs: number;
  p95LatencyMs: number;
  totalCost: number;
  totalSimulatedMs: number;
  routingMetric: NetworkRoutingMetric;
  flowStats: NetworkFlowStats[];
  nodeStats: NetworkNodeStats[];
  linkStats: NetworkLinkStats[];
  bottlenecks: NetworkBottleneckReport[];
  timeSeries: NetworkTimeSample[];
  deliveredPacketsTrace: NetworkPacketSnapshot[];
  droppedPacketsTrace: NetworkPacketSnapshot[];
  invariantViolations: string[];
}

interface FlowRuntimeState {
  spec: NetworkFlowSpec;
  pending: number;
  generated: number;
  droppedAtSource: number;
}

interface ScheduledPacket {
  packet: NetworkPacket;
  deliverAtMs: number;
}

interface QueuedNodePacket {
  packet: NetworkPacket;
  enqueuedAtMs: number;
}

export class NetworkPacket extends BasicMovingEntity<NetworkPacketSnapshot> implements Token {
  deliveredAtMs?: number;
  droppedAtMs?: number;
  dropReason?: PacketDropReason;
  currentNodeId?: string;
  currentLinkId?: string;
  readonly hops: string[];
  cost = 0;

  constructor(
    readonly packetId: number,
    readonly flowId: string,
    readonly protocol: NetworkProtocol,
    readonly source: string,
    readonly destination: string,
    readonly payloadBytes: number,
    readonly sizeBytes: number,
    readonly createdAtMs: number,
    readonly ttlHops: number,
  ) {
    super();
    this.hops = [source];
    this.currentNodeId = source;
  }

  snapshot(): NetworkPacketSnapshot {
    return {
      packetId: this.packetId,
      flowId: this.flowId,
      protocol: this.protocol,
      source: this.source,
      destination: this.destination,
      payloadBytes: this.payloadBytes,
      sizeBytes: this.sizeBytes,
      createdAtMs: this.createdAtMs,
      deliveredAtMs: this.deliveredAtMs,
      droppedAtMs: this.droppedAtMs,
      dropReason: this.dropReason,
      currentNodeId: this.currentNodeId,
      currentLinkId: this.currentLinkId,
      hops: this.hops.slice(),
      cost: this.cost,
    };
  }

  override getValue(): {id: string; value: NetworkPacketSnapshot} {
    return {id: this.id, value: this.snapshot()};
  }

  override runTimeStep(): void {
    // Packets are passive movables; nodes and links advance them.
  }
}

export abstract class NetworkStation extends DESStation {
  protected constructor(id: string) {
    super(id);
  }
}

export abstract class NetworkNodeStation extends NetworkStation {
  readonly queueLimitPackets: number;
  readonly forwardingRatePps: number;
  protected readonly inputQueue: QueuedNodePacket[] = [];
  private forwardingCredit = 0;
  private queueArea = 0;
  private maxQueue = 0;
  private totalQueueDelayMs = 0;
  private maxQueueDelayMs = 0;
  private processedFromQueue = 0;
  receivedPackets = 0;
  forwardedPackets = 0;
  deliveredPackets = 0;
  droppedPackets = 0;

  protected constructor(readonly spec: NetworkNodeSpec) {
    super(`node-${spec.id}`);
    this.queueLimitPackets = spec.queueLimitPackets ?? defaultNodeQueueLimit(spec.kind);
    this.forwardingRatePps = spec.forwardingRatePps ?? defaultForwardingRate(spec.kind);
  }

  get nodeId(): string {
    return this.spec.id;
  }

  get kind(): NetworkNodeKind {
    return this.spec.kind;
  }

  canAcceptPacket(reservedIncoming = 0): boolean {
    return this.inputQueue.length + reservedIncoming < this.queueLimitPackets;
  }

  receivePacket(packet: NetworkPacket, timeMs = 0): boolean {
    if (!this.canAcceptPacket()) return false;
    packet.currentNodeId = this.nodeId;
    packet.currentLinkId = undefined;
    this.inputQueue.push({packet, enqueuedAtMs: timeMs});
    this.receivedPackets++;
    this.maxQueue = Math.max(this.maxQueue, this.inputQueue.length);
    return true;
  }

  step(
    timeMs: number,
    dtMs: number,
    route: (nodeId: string, destination: string) => NetworkLinkStation | null,
    deliver: (packet: NetworkPacket, node: NetworkNodeStation) => void,
    drop: (packet: NetworkPacket, reason: PacketDropReason, atStationId: string) => void,
  ): void {
    this.forwardingCredit += this.forwardingRatePps * dtMs / 1000;
    let budget = Math.floor(this.forwardingCredit + 1e-12);
    while (budget > 0 && this.inputQueue.length > 0) {
      const queued = this.inputQueue.shift()!;
      const packet = queued.packet;
      const queueDelayMs = Math.max(0, timeMs - queued.enqueuedAtMs);
      this.totalQueueDelayMs += queueDelayMs;
      this.maxQueueDelayMs = Math.max(this.maxQueueDelayMs, queueDelayMs);
      this.processedFromQueue++;
      this.forwardingCredit -= 1;
      budget--;
      if (packet.destination === this.nodeId) {
        this.deliveredPackets++;
        deliver(packet, this);
        continue;
      }
      if (packet.hops.length - 1 >= packet.ttlHops) {
        this.droppedPackets++;
        drop(packet, 'ttl-expired', this.nodeId);
        continue;
      }
      const link = route(this.nodeId, packet.destination);
      if (!link) {
        this.droppedPackets++;
        drop(packet, 'no-route', this.nodeId);
        continue;
      }
      if (!link.canAcceptPacket()) {
        this.droppedPackets++;
        link.droppedPackets++;
        drop(packet, 'link-queue-overflow', link.linkId);
        continue;
      }
      this.forwardedPackets++;
      link.enqueuePacket(packet, timeMs);
    }
    this.recordQueue(dtMs);
  }

  stats(totalMs: number): NetworkNodeStats {
    return {
      id: this.nodeId,
      kind: this.kind,
      forwardingRatePps: this.forwardingRatePps,
      queueLimitPackets: this.queueLimitPackets,
      receivedPackets: this.receivedPackets,
      forwardedPackets: this.forwardedPackets,
      deliveredPackets: this.deliveredPackets,
      droppedPackets: this.droppedPackets,
      finalQueue: this.inputQueue.length,
      maxQueue: this.maxQueue,
      avgQueue: this.queueArea / Math.max(1, totalMs),
      meanQueueDelayMs: this.totalQueueDelayMs / Math.max(1, this.processedFromQueue),
      maxQueueDelayMs: this.maxQueueDelayMs,
    };
  }

  queuedPackets(): number {
    return this.inputQueue.length;
  }

  override hasWork(): boolean {
    return this.inputQueue.length > 0;
  }

  runTimeStep(): void {}

  private recordQueue(dtMs: number): void {
    this.queueArea += this.inputQueue.length * dtMs;
    this.maxQueue = Math.max(this.maxQueue, this.inputQueue.length);
  }
}

export class NetworkHostStation extends NetworkNodeStation {
  constructor(spec: Omit<NetworkNodeSpec, 'kind'> & {kind?: 'host'}) {
    super({...spec, kind: 'host'});
  }
}

export class NetworkRouterStation extends NetworkNodeStation {
  constructor(spec: Omit<NetworkNodeSpec, 'kind'> & {kind?: 'router'}) {
    super({...spec, kind: 'router'});
  }
}

export class NetworkSwitchStation extends NetworkNodeStation {
  constructor(spec: Omit<NetworkNodeSpec, 'kind'> & {kind?: 'switch'}) {
    super({...spec, kind: 'switch'});
  }
}

export abstract class NetworkDelayStation extends NetworkStation {
  readonly isDelayBlock = true;
  abstract scheduledCount(): number;
}

export class NetworkLinkStation extends NetworkDelayStation {
  readonly queueLimitPackets: number;
  readonly costPerMb: number;
  private readonly scheduled: ScheduledPacket[] = [];
  private availableAtMs = 0;
  private occupancyArea = 0;
  private maxInFlight = 0;
  enqueuedPackets = 0;
  deliveredPackets = 0;
  droppedPackets = 0;
  transmittedBytes = 0;
  totalSerializationMs = 0;
  private totalQueueDelayMs = 0;
  private maxQueueDelayMs = 0;
  private totalTimeOnLinkMs = 0;
  private maxTimeOnLinkMs = 0;
  totalCost = 0;

  constructor(readonly spec: NetworkLinkSpec) {
    super(`link-${spec.id}`);
    this.queueLimitPackets = spec.queueLimitPackets ?? 64;
    this.costPerMb = spec.costPerMb ?? 0;
  }

  get linkId(): string {
    return this.spec.id;
  }

  canAcceptPacket(): boolean {
    return this.scheduled.length < this.queueLimitPackets;
  }

  enqueuePacket(packet: NetworkPacket, timeMs: number): void {
    const serializationMs = this.serializationMs(packet);
    const startAtMs = Math.max(timeMs, this.availableAtMs);
    const queueDelayMs = Math.max(0, startAtMs - timeMs);
    const deliverAtMs = startAtMs + serializationMs + this.spec.latencyMs;
    const timeOnLinkMs = queueDelayMs + serializationMs + this.spec.latencyMs;
    this.availableAtMs = startAtMs + serializationMs;
    const packetCost = mb(packet.sizeBytes) * this.costPerMb;
    packet.cost += packetCost;
    packet.currentNodeId = undefined;
    packet.currentLinkId = this.linkId;
    packet.hops.push(this.spec.to);
    this.enqueuedPackets++;
    this.transmittedBytes += packet.sizeBytes;
    this.totalSerializationMs += serializationMs;
    this.totalQueueDelayMs += queueDelayMs;
    this.maxQueueDelayMs = Math.max(this.maxQueueDelayMs, queueDelayMs);
    this.totalTimeOnLinkMs += timeOnLinkMs;
    this.maxTimeOnLinkMs = Math.max(this.maxTimeOnLinkMs, timeOnLinkMs);
    this.totalCost += packetCost;
    this.scheduled.push({packet, deliverAtMs});
    this.maxInFlight = Math.max(this.maxInFlight, this.scheduled.length);
  }

  releaseArrivals(timeMs: number): NetworkPacket[] {
    const ready: NetworkPacket[] = [];
    let keep = 0;
    for (let i = 0; i < this.scheduled.length; i++) {
      const item = this.scheduled[i];
      if (item.deliverAtMs <= timeMs + 1e-9) {
        ready.push(item.packet);
        this.deliveredPackets++;
      } else {
        this.scheduled[keep++] = item;
      }
    }
    this.scheduled.length = keep;
    return ready;
  }

  stepOccupancy(dtMs: number): void {
    this.occupancyArea += this.scheduled.length * dtMs;
    this.maxInFlight = Math.max(this.maxInFlight, this.scheduled.length);
  }

  stats(totalMs: number): NetworkLinkStats {
    const simulatedSec = Math.max(1e-9, totalMs / 1000);
    return {
      id: this.linkId,
      from: this.spec.from,
      to: this.spec.to,
      bandwidthMbps: this.spec.bandwidthMbps,
      latencyMs: this.spec.latencyMs,
      costPerMb: this.costPerMb,
      queueLimitPackets: this.queueLimitPackets,
      enqueuedPackets: this.enqueuedPackets,
      deliveredPackets: this.deliveredPackets,
      droppedPackets: this.droppedPackets,
      transmittedBytes: this.transmittedBytes,
      throughputMbps: this.transmittedBytes * 8 / simulatedSec / 1e6,
      utilization: Math.min(1, this.totalSerializationMs / Math.max(1e-9, totalMs)),
      finalInFlight: this.scheduled.length,
      maxInFlight: this.maxInFlight,
      avgInFlight: this.occupancyArea / Math.max(1, totalMs),
      meanQueueDelayMs: this.totalQueueDelayMs / Math.max(1, this.enqueuedPackets),
      maxQueueDelayMs: this.maxQueueDelayMs,
      meanTimeOnLinkMs: this.totalTimeOnLinkMs / Math.max(1, this.enqueuedPackets),
      maxTimeOnLinkMs: this.maxTimeOnLinkMs,
      totalCost: this.totalCost,
    };
  }

  scheduledCount(): number {
    return this.scheduled.length;
  }

  override hasWork(): boolean {
    return this.scheduled.length > 0;
  }

  runTimeStep(): void {}

  private serializationMs(packet: NetworkPacket): number {
    return packet.sizeBytes * 8 / (this.spec.bandwidthMbps * 1e6) * 1000;
  }
}

export class ComputerNetworkStation extends NetworkStation {
  private readonly p: ComputerNetworkProblem;
  private readonly nodes = new Map<string, NetworkNodeStation>();
  private readonly links = new Map<string, NetworkLinkStation>();
  private readonly outgoing = new Map<string, NetworkLinkStation[]>();
  private readonly routeCache = new Map<string, NetworkLinkStation | null>();
  private readonly flows: FlowRuntimeState[];
  private readonly delivered: NetworkPacket[] = [];
  private readonly dropped: NetworkPacket[] = [];
  private readonly timeSeries: NetworkTimeSample[] = [];
  private readonly invariantViolations: string[] = [];
  private nextPacketId = 1;
  private timeMs = 0;
  private maxActivePackets = 0;
  private nextSampleAtMs = 0;

  constructor(p: ComputerNetworkProblem) {
    super('computer-network');
    validateComputerNetworkProblem(p);
    this.p = normalizeComputerNetworkProblem(p);
    for (const n of this.p.nodes) {
      const station = makeNodeStation(n);
      this.nodes.set(n.id, station);
    }
    for (const l of this.p.links) {
      const station = new NetworkLinkStation(l);
      this.links.set(l.id, station);
      const arr = this.outgoing.get(l.from) ?? [];
      arr.push(station);
      this.outgoing.set(l.from, arr);
    }
    this.flows = this.p.flows.map(spec => ({spec, pending: 0, generated: 0, droppedAtSource: 0}));

    this.addValidator(intrinsicCheck<ComputerNetworkStation>({
      name: 'computer-network.conservation',
      group: 'computer-network-intrinsic',
      predicate: st => st.generatedPackets() === st.delivered.length + st.dropped.length + st.activePackets(),
      expected: 'generated = delivered + dropped + active',
      observedFn: st => `generated=${st.generatedPackets()}, delivered=${st.delivered.length}, dropped=${st.dropped.length}, active=${st.activePackets()}`,
    }));
    this.addValidator(intrinsicCheck<ComputerNetworkStation>({
      name: 'computer-network.queues-within-capacity',
      group: 'computer-network-intrinsic',
      predicate: st => st.allQueuesWithinCapacity(),
      expected: 'node and link queues within configured packet limits',
      observedFn: st => `violations=${st.invariantViolations.length}`,
    }));
  }

  override hasWork(): boolean {
    const drainUntil = this.p.durationMs + (this.p.drainAfterSourcesMs ?? 1000);
    return this.timeMs < this.p.durationMs || (this.activePackets() > 0 && this.timeMs < drainUntil);
  }

  runTimeStep(): void {
    this.releaseLinkArrivals();
    if (this.timeMs < this.p.durationMs) this.generateFlowPackets();
    for (const node of this.nodes.values()) {
      node.step(
        this.timeMs,
        this.p.dtMs,
        (nodeId, destination) => this.nextLink(nodeId, destination),
        (packet, atNode) => this.deliver(packet, atNode),
        (packet, reason, atStationId) => this.drop(packet, reason, atStationId),
      );
    }
    for (const link of this.links.values()) link.stepOccupancy(this.p.dtMs);
    this.recordInvariants();
    this.recordStats();
    this.timeMs += this.p.dtMs;
  }

  buildResult(): ComputerNetworkResult {
    const latencies = this.delivered
      .map(p => (p.deliveredAtMs ?? this.timeMs) - p.createdAtMs)
      .sort((a, b) => a - b);
    const totalDeliveredBytes = this.delivered.reduce((s, p) => s + p.sizeBytes, 0);
    const totalDeliveredPayloadBytes = this.delivered.reduce((s, p) => s + p.payloadBytes, 0);
    const totalGeneratedBytes = this.flows.reduce((s, f) => s + f.generated * effectivePacketSizeBytes(f.spec), 0);
    const totalCost = this.delivered.reduce((s, p) => s + p.cost, 0) +
      this.dropped.reduce((s, p) => s + p.cost, 0);
    const simulatedSec = Math.max(1e-9, this.p.durationMs / 1000);
    const nodeStats = [...this.nodes.values()].map(n => n.stats(this.timeMs));
    const linkStats = [...this.links.values()].map(l => l.stats(this.timeMs));
    const flowStats = this.buildFlowStats();
    return {
      generatedPackets: this.generatedPackets(),
      deliveredPackets: this.delivered.length,
      droppedPackets: this.dropped.length,
      activePackets: this.activePackets(),
      maxActivePackets: this.maxActivePackets,
      deliveryRatio: this.delivered.length / Math.max(1, this.generatedPackets()),
      offeredLoadMbps: totalGeneratedBytes * 8 / simulatedSec / 1e6,
      throughputMbps: totalDeliveredBytes * 8 / simulatedSec / 1e6,
      goodputMbps: totalDeliveredPayloadBytes * 8 / simulatedSec / 1e6,
      meanLatencyMs: mean(latencies),
      p95LatencyMs: percentile(latencies, 0.95),
      totalCost,
      totalSimulatedMs: this.timeMs,
      routingMetric: this.p.routingMetric ?? 'latency',
      flowStats,
      nodeStats,
      linkStats,
      bottlenecks: identifyBottlenecks(nodeStats, linkStats),
      timeSeries: this.timeSeries.slice(),
      deliveredPacketsTrace: this.delivered.slice(0, 200).map(p => p.snapshot()),
      droppedPacketsTrace: this.dropped.slice(0, 200).map(p => p.snapshot()),
      invariantViolations: this.invariantViolations.slice(),
    };
  }

  private releaseLinkArrivals(): void {
    for (const link of this.links.values()) {
      const arrivals = link.releaseArrivals(this.timeMs);
      for (const packet of arrivals) {
        const node = this.nodes.get(link.spec.to);
        if (!node) {
          this.drop(packet, 'no-route', link.spec.to);
          continue;
        }
        if (!node.receivePacket(packet, this.timeMs)) {
          node.droppedPackets++;
          this.drop(packet, 'node-queue-overflow', node.nodeId);
        }
      }
    }
  }

  private generateFlowPackets(): void {
    for (const flow of this.flows) {
      const spec = flow.spec;
      const profile = protocolProfile(spec.protocol);
      const flowStartMs = (spec.startMs ?? 0) + profile.startupDelayMs;
      if (this.timeMs < flowStartMs || this.timeMs > (spec.endMs ?? this.p.durationMs)) continue;
      flow.pending += spec.ratePps * this.p.dtMs / 1000;
      while (flow.pending >= 1 - 1e-12) {
        if (spec.maxPackets !== undefined && flow.generated >= spec.maxPackets) {
          flow.pending = 0;
          break;
        }
        const packet = new NetworkPacket(
          this.nextPacketId++,
          spec.id,
          profile.protocol,
          spec.source,
          spec.destination,
          spec.packetSizeBytes,
          effectivePacketSizeBytes(spec),
          this.timeMs,
          spec.ttlHops ?? Math.max(8, this.p.nodes.length * 4),
        );
        flow.generated++;
        flow.pending -= 1;
        const source = this.nodes.get(spec.source)!;
        if (this.activePackets() >= (this.p.maxPacketsInSystem ?? Infinity)) {
          flow.droppedAtSource++;
          source.droppedPackets++;
          this.drop(packet, 'max-packets-in-system', source.nodeId);
          continue;
        }
        if (!source.receivePacket(packet, this.timeMs)) {
          flow.droppedAtSource++;
          source.droppedPackets++;
          this.drop(packet, 'node-queue-overflow', source.nodeId);
        }
      }
    }
  }

  private deliver(packet: NetworkPacket, atNode: NetworkNodeStation): void {
    packet.deliveredAtMs = this.timeMs;
    packet.currentNodeId = atNode.nodeId;
    packet.currentLinkId = undefined;
    packet.doFinish();
    this.delivered.push(packet);
  }

  private drop(packet: NetworkPacket, reason: PacketDropReason, atStationId: string): void {
    packet.droppedAtMs = this.timeMs;
    packet.dropReason = reason;
    packet.currentNodeId = atStationId;
    packet.currentLinkId = undefined;
    packet.doFinish();
    this.dropped.push(packet);
  }

  private nextLink(nodeId: string, destination: string): NetworkLinkStation | null {
    const key = `${nodeId}->${destination}`;
    if (this.routeCache.has(key)) return this.routeCache.get(key)!;
    const link = this.shortestNextLink(nodeId, destination);
    this.routeCache.set(key, link);
    return link;
  }

  private shortestNextLink(source: string, destination: string): NetworkLinkStation | null {
    if (source === destination) return null;
    const dist = new Map<string, number>();
    const prevNode = new Map<string, string>();
    const prevLink = new Map<string, NetworkLinkStation>();
    const unsettled = new Set<string>(this.nodes.keys());
    for (const id of unsettled) dist.set(id, Infinity);
    dist.set(source, 0);
    while (unsettled.size > 0) {
      let u: string | null = null;
      let best = Infinity;
      for (const id of unsettled) {
        const d = dist.get(id) ?? Infinity;
        if (d < best) { best = d; u = id; }
      }
      if (u === null || !Number.isFinite(best)) break;
      unsettled.delete(u);
      if (u === destination) break;
      for (const link of this.outgoing.get(u) ?? []) {
        const v = link.spec.to;
        if (!unsettled.has(v)) continue;
        const nd = best + linkWeight(link.spec, this.p.routingMetric ?? 'latency');
        if (nd < (dist.get(v) ?? Infinity)) {
          dist.set(v, nd);
          prevNode.set(v, u);
          prevLink.set(v, link);
        }
      }
    }
    if (!prevLink.has(destination)) return null;
    let cur = destination;
    let first = prevLink.get(cur)!;
    while ((prevNode.get(cur) ?? source) !== source) {
      cur = prevNode.get(cur)!;
      first = prevLink.get(cur)!;
    }
    return first;
  }

  private buildFlowStats(): NetworkFlowStats[] {
    return this.flows.map(flow => {
      const spec = flow.spec;
      const protocol = protocolProfile(spec.protocol).protocol;
      const delivered = this.delivered.filter(p => p.flowId === spec.id);
      const dropped = this.dropped.filter(p => p.flowId === spec.id);
      const latencies = delivered.map(p => (p.deliveredAtMs ?? this.timeMs) - p.createdAtMs).sort((a, b) => a - b);
      const deliveredBytes = delivered.reduce((s, p) => s + p.sizeBytes, 0);
      const deliveredPayloadBytes = delivered.reduce((s, p) => s + p.payloadBytes, 0);
      const totalCost = delivered.reduce((s, p) => s + p.cost, 0) + dropped.reduce((s, p) => s + p.cost, 0);
      const simulatedSec = Math.max(1e-9, this.p.durationMs / 1000);
      return {
        id: spec.id,
        protocol,
        source: spec.source,
        destination: spec.destination,
        generatedPackets: flow.generated,
        deliveredPackets: delivered.length,
        droppedPackets: dropped.length,
        deliveryRatio: delivered.length / Math.max(1, flow.generated),
        generatedBytes: flow.generated * effectivePacketSizeBytes(spec),
        deliveredBytes,
        offeredLoadMbps: flow.generated * effectivePacketSizeBytes(spec) * 8 / simulatedSec / 1e6,
        throughputMbps: deliveredBytes * 8 / simulatedSec / 1e6,
        goodputMbps: deliveredPayloadBytes * 8 / simulatedSec / 1e6,
        meanLatencyMs: mean(latencies),
        p95LatencyMs: percentile(latencies, 0.95),
        meanTimeInSystemMs: mean(latencies),
        p95TimeInSystemMs: percentile(latencies, 0.95),
        totalCost,
        meanCostPerDeliveredPacket: totalCost / Math.max(1, delivered.length),
      };
    });
  }

  private activePackets(): number {
    let n = 0;
    for (const node of this.nodes.values()) n += node.queuedPackets();
    for (const link of this.links.values()) n += link.scheduledCount();
    return n;
  }

  private generatedPackets(): number {
    return this.flows.reduce((s, f) => s + f.generated, 0);
  }

  private allQueuesWithinCapacity(): boolean {
    for (const node of this.nodes.values()) {
      if (node.queuedPackets() > node.queueLimitPackets) return false;
    }
    for (const link of this.links.values()) {
      if (link.scheduledCount() > link.queueLimitPackets) return false;
    }
    return true;
  }

  private recordStats(): void {
    const active = this.activePackets();
    this.maxActivePackets = Math.max(this.maxActivePackets, active);
    const sampleEveryMs = this.p.sampleEveryMs ?? Math.max(this.p.dtMs, 100);
    if (this.timeMs + 1e-9 < this.nextSampleAtMs) return;

    const nodeQueues: Record<string, number> = {};
    for (const node of this.nodes.values()) nodeQueues[node.nodeId] = node.queuedPackets();

    const linkInFlight: Record<string, number> = {};
    const linkUtilization: Record<string, number> = {};
    const elapsedMs = Math.max(1, this.timeMs + this.p.dtMs);
    for (const link of this.links.values()) {
      linkInFlight[link.linkId] = link.scheduledCount();
      linkUtilization[link.linkId] = link.stats(elapsedMs).utilization;
    }

    this.timeSeries.push({
      tMs: this.timeMs,
      generatedPackets: this.generatedPackets(),
      deliveredPackets: this.delivered.length,
      droppedPackets: this.dropped.length,
      activePackets: active,
      nodeQueues,
      linkInFlight,
      linkUtilization,
    });
    this.nextSampleAtMs += sampleEveryMs;
  }

  private recordInvariants(): void {
    for (const node of this.nodes.values()) {
      if (node.queuedPackets() > node.queueLimitPackets) {
        this.invariantViolations.push(`${node.nodeId}: node queue ${node.queuedPackets()} > ${node.queueLimitPackets}`);
      }
    }
    for (const link of this.links.values()) {
      if (link.scheduledCount() > link.queueLimitPackets) {
        this.invariantViolations.push(`${link.linkId}: link queue ${link.scheduledCount()} > ${link.queueLimitPackets}`);
      }
    }
  }
}

export function validateComputerNetworkProblem(p: ComputerNetworkProblem): void {
  Preconditions.nonEmpty(MODEL, 'nodes', p.nodes);
  Preconditions.nonEmpty(MODEL, 'links', p.links);
  Preconditions.nonEmpty(MODEL, 'flows', p.flows);
  Preconditions.positive(MODEL, 'durationMs', p.durationMs);
  Preconditions.positive(MODEL, 'dtMs', p.dtMs);
  if (p.routingMetric !== undefined) {
    Preconditions.check(MODEL, 'routingMetric', 'be latency, cost, or hop',
      ['latency', 'cost', 'hop'].includes(p.routingMetric), p.routingMetric);
  }
  if (p.drainAfterSourcesMs !== undefined) Preconditions.nonNegative(MODEL, 'drainAfterSourcesMs', p.drainAfterSourcesMs);
  if (p.maxPacketsInSystem !== undefined) Preconditions.integerInRange(MODEL, 'maxPacketsInSystem', p.maxPacketsInSystem, 1, 1e7);
  if (p.sampleEveryMs !== undefined) Preconditions.positive(MODEL, 'sampleEveryMs', p.sampleEveryMs);

  const nodeIds = new Set<string>();
  const nodeById = new Map<string, NetworkNodeSpec>();
  for (const n of p.nodes) {
    Preconditions.check(MODEL, `node ${n.id}`, 'have a non-empty id', typeof n.id === 'string' && n.id.length > 0, n.id);
    Preconditions.check(MODEL, `${n.id}.kind`, 'be host, router, or switch', ['host', 'router', 'switch'].includes(n.kind), n.kind);
    Preconditions.check(MODEL, `node ${n.id}`, 'be unique', !nodeIds.has(n.id), n.id);
    nodeIds.add(n.id);
    nodeById.set(n.id, n);
    if (n.forwardingRatePps !== undefined) Preconditions.positive(MODEL, `${n.id}.forwardingRatePps`, n.forwardingRatePps);
    if (n.queueLimitPackets !== undefined) Preconditions.integerInRange(MODEL, `${n.id}.queueLimitPackets`, n.queueLimitPackets, 1, 1e7);
  }

  const linkIds = new Set<string>();
  for (const l of p.links) {
    Preconditions.check(MODEL, `link ${l.id}`, 'have a non-empty id', typeof l.id === 'string' && l.id.length > 0, l.id);
    Preconditions.check(MODEL, `link ${l.id}`, 'be unique', !linkIds.has(l.id), l.id);
    linkIds.add(l.id);
    Preconditions.check(MODEL, `${l.id}.from`, 'reference a node', nodeIds.has(l.from), l.from);
    Preconditions.check(MODEL, `${l.id}.to`, 'reference a node', nodeIds.has(l.to), l.to);
    Preconditions.check(MODEL, `${l.id}.from != to`, 'hold', l.from !== l.to, [l.from, l.to]);
    Preconditions.positive(MODEL, `${l.id}.bandwidthMbps`, l.bandwidthMbps);
    Preconditions.nonNegative(MODEL, `${l.id}.latencyMs`, l.latencyMs);
    if (l.costPerMb !== undefined) Preconditions.nonNegative(MODEL, `${l.id}.costPerMb`, l.costPerMb);
    if (l.queueLimitPackets !== undefined) Preconditions.integerInRange(MODEL, `${l.id}.queueLimitPackets`, l.queueLimitPackets, 1, 1e7);
  }

  const normalized = normalizeComputerNetworkProblem(p);
  const outgoing = new Map<string, string[]>();
  for (const l of normalized.links) outgoing.set(l.from, [...(outgoing.get(l.from) ?? []), l.to]);
  const flowIds = new Set<string>();
  for (const f of p.flows) {
    Preconditions.check(MODEL, `flow ${f.id}`, 'have a non-empty id', typeof f.id === 'string' && f.id.length > 0, f.id);
    Preconditions.check(MODEL, `flow ${f.id}`, 'be unique', !flowIds.has(f.id), f.id);
    flowIds.add(f.id);
    Preconditions.check(MODEL, `${f.id}.source`, 'reference a node', nodeIds.has(f.source), f.source);
    Preconditions.check(MODEL, `${f.id}.destination`, 'reference a node', nodeIds.has(f.destination), f.destination);
    Preconditions.check(MODEL, `${f.id}.source`, 'reference a host source entity', nodeById.get(f.source)?.kind === 'host', f.source);
    Preconditions.check(MODEL, `${f.id}.destination`, 'reference a host sink entity', nodeById.get(f.destination)?.kind === 'host', f.destination);
    Preconditions.check(MODEL, `${f.id}.source != destination`, 'hold', f.source !== f.destination, [f.source, f.destination]);
    if (f.protocol !== undefined) {
      Preconditions.check(MODEL, `${f.id}.protocol`, 'be raw, tcp, udp, or http',
        ['raw', 'tcp', 'udp', 'http'].includes(f.protocol), f.protocol);
    }
    Preconditions.nonNegative(MODEL, `${f.id}.ratePps`, f.ratePps);
    Preconditions.integerInRange(MODEL, `${f.id}.packetSizeBytes`, f.packetSizeBytes, 1, 1e9);
    if (f.startMs !== undefined) Preconditions.nonNegative(MODEL, `${f.id}.startMs`, f.startMs);
    if (f.startMs !== undefined) Preconditions.check(MODEL, `${f.id}.startMs`, 'fall within durationMs', f.startMs <= p.durationMs, [f.startMs, p.durationMs]);
    if (f.endMs !== undefined) Preconditions.nonNegative(MODEL, `${f.id}.endMs`, f.endMs);
    if (f.startMs !== undefined && f.endMs !== undefined) {
      Preconditions.check(MODEL, `${f.id}.startMs <= endMs`, 'hold', f.startMs <= f.endMs, [f.startMs, f.endMs]);
    }
    if (f.maxPackets !== undefined) Preconditions.integerInRange(MODEL, `${f.id}.maxPackets`, f.maxPackets, 0, 1e9);
    if (f.ttlHops !== undefined) Preconditions.integerInRange(MODEL, `${f.id}.ttlHops`, f.ttlHops, 1, 1e6);
    Preconditions.check(MODEL, `${f.id}.route`, 'exist in directed link graph', hasDirectedPath(f.source, f.destination, outgoing), [f.source, f.destination]);
  }
}

export function runComputerNetworkSimulation(p: ComputerNetworkProblem): ComputerNetworkResult {
  const network = new ComputerNetworkStation(p);
  const problem = normalizeComputerNetworkProblem(p);
  const maxTicks = Math.ceil((problem.durationMs + (problem.drainAfterSourcesMs ?? 1000)) / problem.dtMs) + 5;
  const summary = runIterativeDES([network], {shuffle: false, maxTicks});
  assertNoValidationFailures(summary, MODEL);
  return network.buildResult();
}

export function buildDefaultComputerNetworkProblem(): ComputerNetworkProblem {
  return {
    nodes: [
      {id: 'client-a', kind: 'host', forwardingRatePps: 2000, queueLimitPackets: 256},
      {id: 'client-b', kind: 'host', forwardingRatePps: 2000, queueLimitPackets: 256},
      {id: 'edge-1', kind: 'router', forwardingRatePps: 6000, queueLimitPackets: 512},
      {id: 'core-1', kind: 'router', forwardingRatePps: 8000, queueLimitPackets: 512},
      {id: 'server', kind: 'host', forwardingRatePps: 4000, queueLimitPackets: 512},
    ],
    links: [
      {id: 'client-a-edge', from: 'client-a', to: 'edge-1', bandwidthMbps: 100, latencyMs: 1, costPerMb: 0.001, queueLimitPackets: 128, bidirectional: true},
      {id: 'client-b-edge', from: 'client-b', to: 'edge-1', bandwidthMbps: 50, latencyMs: 2, costPerMb: 0.001, queueLimitPackets: 128, bidirectional: true},
      {id: 'edge-core', from: 'edge-1', to: 'core-1', bandwidthMbps: 25, latencyMs: 8, costPerMb: 0.004, queueLimitPackets: 96, bidirectional: true},
      {id: 'core-server', from: 'core-1', to: 'server', bandwidthMbps: 100, latencyMs: 3, costPerMb: 0.002, queueLimitPackets: 128, bidirectional: true},
    ],
    flows: [
      {id: 'a-to-server', source: 'client-a', destination: 'server', protocol: 'http', ratePps: 650, packetSizeBytes: 1200, maxPackets: 650},
      {id: 'b-to-server', source: 'client-b', destination: 'server', protocol: 'tcp', ratePps: 300, packetSizeBytes: 1000, maxPackets: 300},
    ],
    durationMs: 1000,
    dtMs: 1,
    routingMetric: 'latency',
    drainAfterSourcesMs: 1500,
    maxPacketsInSystem: 5000,
    sampleEveryMs: 100,
  };
}

export function buildBottleneckComputerNetworkProblem(): ComputerNetworkProblem {
  return {
    nodes: [
      {id: 'web-client', kind: 'host', forwardingRatePps: 6000, queueLimitPackets: 512},
      {id: 'telemetry-client', kind: 'host', forwardingRatePps: 6000, queueLimitPackets: 512},
      {id: 'edge', kind: 'switch', forwardingRatePps: 12000, queueLimitPackets: 1024},
      {id: 'wan-router', kind: 'router', forwardingRatePps: 9000, queueLimitPackets: 1024},
      {id: 'api-server', kind: 'host', forwardingRatePps: 9000, queueLimitPackets: 1024},
    ],
    links: [
      {id: 'web-edge', from: 'web-client', to: 'edge', bandwidthMbps: 100, latencyMs: 1, costPerMb: 0.001, queueLimitPackets: 256, bidirectional: true},
      {id: 'telemetry-edge', from: 'telemetry-client', to: 'edge', bandwidthMbps: 100, latencyMs: 1, costPerMb: 0.001, queueLimitPackets: 256, bidirectional: true},
      {id: 'edge-wan', from: 'edge', to: 'wan-router', bandwidthMbps: 5, latencyMs: 25, costPerMb: 0.010, queueLimitPackets: 96, bidirectional: true},
      {id: 'wan-api', from: 'wan-router', to: 'api-server', bandwidthMbps: 100, latencyMs: 4, costPerMb: 0.002, queueLimitPackets: 256, bidirectional: true},
    ],
    flows: [
      {id: 'http-api', source: 'web-client', destination: 'api-server', protocol: 'http', ratePps: 900, packetSizeBytes: 1100, maxPackets: 1800},
      {id: 'udp-telemetry', source: 'telemetry-client', destination: 'api-server', protocol: 'udp', ratePps: 700, packetSizeBytes: 900, maxPackets: 1400},
      {id: 'tcp-bulk', source: 'web-client', destination: 'api-server', protocol: 'tcp', ratePps: 350, packetSizeBytes: 1400, maxPackets: 700},
    ],
    durationMs: 2000,
    dtMs: 1,
    routingMetric: 'latency',
    drainAfterSourcesMs: 4000,
    maxPacketsInSystem: 10000,
    sampleEveryMs: 100,
  };
}

function normalizeComputerNetworkProblem(p: ComputerNetworkProblem): ComputerNetworkProblem {
  const links: NetworkLinkSpec[] = [];
  const ids = new Set<string>();
  for (const link of p.links) {
    links.push({...link, bidirectional: false});
    ids.add(link.id);
    if (link.bidirectional) {
      let reverseId = `${link.id}:rev`;
      let i = 2;
      while (ids.has(reverseId)) reverseId = `${link.id}:rev${i++}`;
      ids.add(reverseId);
      links.push({
        ...link,
        id: reverseId,
        from: link.to,
        to: link.from,
        bidirectional: false,
      });
    }
  }
  return {
    ...p,
    links,
    routingMetric: p.routingMetric ?? 'latency',
  };
}

function makeNodeStation(spec: NetworkNodeSpec): NetworkNodeStation {
  switch (spec.kind) {
    case 'host': return new NetworkHostStation(spec as Omit<NetworkNodeSpec, 'kind'> & {kind: 'host'});
    case 'router': return new NetworkRouterStation(spec as Omit<NetworkNodeSpec, 'kind'> & {kind: 'router'});
    case 'switch': return new NetworkSwitchStation(spec as Omit<NetworkNodeSpec, 'kind'> & {kind: 'switch'});
  }
}

function protocolProfile(protocol: NetworkProtocol | undefined): {
  protocol: NetworkProtocol;
  overheadBytes: number;
  startupDelayMs: number;
} {
  switch (protocol ?? 'raw') {
    case 'http':
      return {protocol: 'http', overheadBytes: 640, startupDelayMs: 40};
    case 'tcp':
      return {protocol: 'tcp', overheadBytes: 40, startupDelayMs: 20};
    case 'udp':
      return {protocol: 'udp', overheadBytes: 28, startupDelayMs: 0};
    case 'raw':
    default:
      return {protocol: 'raw', overheadBytes: 0, startupDelayMs: 0};
  }
}

function effectivePacketSizeBytes(spec: NetworkFlowSpec): number {
  return spec.packetSizeBytes + protocolProfile(spec.protocol).overheadBytes;
}

function identifyBottlenecks(
  nodeStats: readonly NetworkNodeStats[],
  linkStats: readonly NetworkLinkStats[],
): NetworkBottleneckReport[] {
  const reports: NetworkBottleneckReport[] = [];

  for (const l of linkStats) {
    const queuePressure = l.avgInFlight / Math.max(1, l.queueLimitPackets);
    const delayPressure = Math.min(1, l.meanQueueDelayMs / 1000);
    const dropPressure = Math.min(1, l.droppedPackets / Math.max(1, l.enqueuedPackets + l.droppedPackets));
    const score = l.utilization + queuePressure + delayPressure + dropPressure;
    reports.push({
      id: l.id,
      kind: 'link',
      score,
      reason: bottleneckReason(l.utilization, l.avgInFlight, l.maxInFlight, l.droppedPackets, l.meanQueueDelayMs),
      utilization: l.utilization,
      avgQueue: l.avgInFlight,
      maxQueue: l.maxInFlight,
      droppedPackets: l.droppedPackets,
      meanQueueDelayMs: l.meanQueueDelayMs,
    });
  }

  for (const n of nodeStats) {
    const queuePressure = n.avgQueue / Math.max(1, n.queueLimitPackets);
    const delayPressure = Math.min(1, n.meanQueueDelayMs / 1000);
    const dropPressure = Math.min(1, n.droppedPackets / Math.max(1, n.receivedPackets + n.droppedPackets));
    const servicePressure = n.forwardedPackets > 0 && n.avgQueue > 0 ? 0.25 : 0;
    const score = queuePressure + delayPressure + dropPressure + servicePressure;
    reports.push({
      id: n.id,
      kind: 'node',
      score,
      reason: bottleneckReason(undefined, n.avgQueue, n.maxQueue, n.droppedPackets, n.meanQueueDelayMs),
      avgQueue: n.avgQueue,
      maxQueue: n.maxQueue,
      droppedPackets: n.droppedPackets,
      meanQueueDelayMs: n.meanQueueDelayMs,
    });
  }

  return reports
    .filter(r => r.score > 0 || r.droppedPackets > 0 || r.maxQueue > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);
}

function bottleneckReason(
  utilization: number | undefined,
  avgQueue: number,
  maxQueue: number,
  droppedPackets: number,
  meanQueueDelayMs: number,
): string {
  if (droppedPackets > 0) return 'drops observed';
  if (utilization !== undefined && utilization >= 0.95) return 'saturated link';
  if (meanQueueDelayMs >= 10) return 'queueing delay';
  if (avgQueue >= 1 || maxQueue >= 10) return 'queue buildup';
  if (utilization !== undefined && utilization >= 0.75) return 'high utilization';
  return 'capacity pressure';
}

function defaultForwardingRate(kind: NetworkNodeKind): number {
  switch (kind) {
    case 'host': return 1000;
    case 'switch': return 10000;
    case 'router': return 5000;
  }
}

function defaultNodeQueueLimit(kind: NetworkNodeKind): number {
  switch (kind) {
    case 'host': return 128;
    case 'switch': return 512;
    case 'router': return 256;
  }
}

function linkWeight(link: NetworkLinkSpec, metric: NetworkRoutingMetric): number {
  switch (metric) {
    case 'hop': return 1;
    case 'cost': return link.costPerMb ?? 0;
    case 'latency':
    default:
      return link.latencyMs + (1500 * 8 / (link.bandwidthMbps * 1e6) * 1000);
  }
}

function hasDirectedPath(source: string, sink: string, outgoing: Map<string, string[]>): boolean {
  const seen = new Set<string>([source]);
  const q = [source];
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi];
    if (u === sink) return true;
    for (const v of outgoing.get(u) ?? []) {
      if (seen.has(v)) continue;
      seen.add(v);
      q.push(v);
    }
  }
  return false;
}

function mean(xs: readonly number[]): number {
  if (xs.length === 0) return NaN;
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)));
  return sorted[idx];
}

function mb(bytes: number): number {
  return bytes / 1_000_000;
}
