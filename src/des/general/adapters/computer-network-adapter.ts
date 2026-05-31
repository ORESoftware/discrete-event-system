// RUST MIGRATION: Target module `src/des/general/adapters/computer_network_adapter.rs`.
// RUST MIGRATION: Convert the computer-network adapter and built-in problem selection into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map network nodes, links, flows, params, and results to `serde` config/result structs; file/runtime paths become `PathBuf`.
// RUST MIGRATION: Express schema validation and builtin lookup failures as `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// JSON adapter for computer-network packet DES.
// =============================================================================

import {DESModelRegistration, DESRuntimeConfig, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {csvRow, defaultFramesPath, writeCsvLines} from './adapter-utils';
import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  ComputerNetworkProblem,
  ComputerNetworkResult,
  runComputerNetworkSimulation,
} from '../computer-network';

interface ComputerNetworkParams {
  builtin?: 'small-enterprise' | 'bottleneck-lab';
  problem?: ComputerNetworkProblem;
}

const networkNodeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    kind: {kind: 'string', enum: ['host', 'router', 'switch']},
    forwardingRatePps: {kind: 'number', min: 1e-9},
    queueLimitPackets: {kind: 'number', integer: true, min: 1},
  },
  required: ['id', 'kind'],
};

const networkLinkSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    from: {kind: 'string'},
    to: {kind: 'string'},
    bandwidthMbps: {kind: 'number', min: 1e-9},
    latencyMs: {kind: 'number', min: 0},
    costPerMb: {kind: 'number', min: 0},
    queueLimitPackets: {kind: 'number', integer: true, min: 1},
    bidirectional: {kind: 'boolean', default: false},
  },
  required: ['id', 'from', 'to', 'bandwidthMbps', 'latencyMs'],
};

const networkFlowSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    source: {kind: 'string'},
    destination: {kind: 'string'},
    protocol: {kind: 'string', enum: ['raw', 'tcp', 'udp', 'http'], default: 'raw'},
    ratePps: {kind: 'number', min: 0},
    packetSizeBytes: {kind: 'number', integer: true, min: 1},
    startMs: {kind: 'number', min: 0},
    endMs: {kind: 'number', min: 0},
    maxPackets: {kind: 'number', integer: true, min: 0},
    ttlHops: {kind: 'number', integer: true, min: 1},
  },
  required: ['id', 'source', 'destination', 'ratePps', 'packetSizeBytes'],
};

const computerNetworkProblemSchema: ParamSchema = {
  kind: 'object',
  description: 'Packet-switched topology with hosts/routers/switches, links, and traffic flows.',
  fields: {
    nodes: {kind: 'array', items: networkNodeSchema, minLength: 2},
    links: {kind: 'array', items: networkLinkSchema, minLength: 1},
    flows: {kind: 'array', items: networkFlowSchema, minLength: 1},
    durationMs: {kind: 'number', min: 1e-9},
    dtMs: {kind: 'number', min: 1e-9},
    routingMetric: {kind: 'string', enum: ['latency', 'cost', 'hop'], default: 'latency'},
    drainAfterSourcesMs: {kind: 'number', min: 0},
    maxPacketsInSystem: {kind: 'number', integer: true, min: 1},
    sampleEveryMs: {kind: 'number', min: 1e-9},
  },
  required: ['nodes', 'links', 'flows', 'durationMs', 'dtMs'],
};

const computerNetworkSchema: ParamSchema = {
  kind: 'object',
  description: 'Computer-network DES with stationary host/router/switch/link entities and moving packet entities.',
  fields: {
    builtin: {kind: 'string', enum: ['small-enterprise', 'bottleneck-lab'], default: 'small-enterprise'},
    problem: computerNetworkProblemSchema,
  },
  required: [],
};

const adapter: DESModelRegistration<ComputerNetworkParams, ComputerNetworkResult> = {
  id: 'computer-network',
  description: 'Packet-switched computer-network DES from JSON topology, with latency, throughput, drops, and cost stats.',
  schema: computerNetworkSchema,
  run(params, _runtime: DESRuntimeConfig) {
    return runComputerNetworkSimulation(params.problem ?? problemFromBuiltin(params.builtin));
  },
  summarize(result) {
    const top = result.bottlenecks[0];
    return [
      'COMPUTER-NETWORK DES',
      '--------------------',
      `  Routing metric:   ${result.routingMetric}`,
      `  Generated:        ${result.generatedPackets}`,
      `  Delivered:        ${result.deliveredPackets}`,
      `  Dropped:          ${result.droppedPackets}`,
      `  Active at stop:   ${result.activePackets}`,
      `  Max active:       ${result.maxActivePackets}`,
      `  Delivery ratio:   ${result.deliveryRatio.toFixed(4)}`,
      `  Offered load:     ${result.offeredLoadMbps.toFixed(4)} Mbps`,
      `  Wire throughput:  ${result.throughputMbps.toFixed(4)} Mbps`,
      `  Goodput:          ${result.goodputMbps.toFixed(4)} Mbps`,
      `  Mean latency:     ${fmt(result.meanLatencyMs)} ms`,
      `  P95 latency:      ${fmt(result.p95LatencyMs)} ms`,
      `  Top bottleneck:   ${top ? `${top.kind}:${top.id} (${top.reason})` : 'none'}`,
      `  Total cost:       ${result.totalCost.toFixed(6)}`,
      `  Invariants:       ${result.invariantViolations.length === 0 ? 'ok' : `${result.invariantViolations.length} violations`}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['kind,id,from,to,generated,delivered,dropped,throughput_mbps,goodput_mbps,mean_latency_ms,p95_latency_ms,total_cost,utilization,avg_queue,max_queue,mean_queue_delay_ms'];
    for (const f of result.flowStats) {
      lines.push(csvRow([
        'flow',
        f.id,
        f.source,
        f.destination,
        f.generatedPackets,
        f.deliveredPackets,
        f.droppedPackets,
        f.throughputMbps,
        f.goodputMbps,
        f.meanLatencyMs,
        f.p95LatencyMs,
        f.totalCost,
        '',
        '',
        '',
        '',
      ]));
    }
    for (const l of result.linkStats) {
      lines.push(csvRow([
        'link',
        l.id,
        l.from,
        l.to,
        l.enqueuedPackets,
        l.deliveredPackets,
        l.droppedPackets,
        l.throughputMbps,
        '',
        '',
        '',
        l.totalCost,
        l.utilization,
        l.avgInFlight,
        l.maxInFlight,
        l.meanQueueDelayMs,
      ]));
    }
    for (const n of result.nodeStats) {
      lines.push(csvRow([
        'node',
        n.id,
        '',
        '',
        n.receivedPackets,
        n.forwardedPackets + n.deliveredPackets,
        n.droppedPackets,
        '',
        '',
        '',
        '',
        '',
        '',
        n.avgQueue,
        n.maxQueue,
        n.meanQueueDelayMs,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(result, params, runtime) {
    const out = runtime.outputs ?? {};
    if (!out.html) return;
    const {FrameRecorder} = await import('../../animation/frame-recorder');
    const {
      buildComputerNetworkAnimation,
      COMPUTER_NETWORK_STAGE_W,
      COMPUTER_NETWORK_STAGE_H,
    } = await import('../../animation/scenes/computer-network-scene');
    const problem = params.problem ?? problemFromBuiltin(params.builtin);
    const framesPath = out.frames ?? defaultFramesPath(out.html);
    const {frames, charts} = buildComputerNetworkAnimation(problem, result);
    const recorder = new FrameRecorder({
      framesPath,
      htmlPath: out.html,
      width: COMPUTER_NETWORK_STAGE_W,
      height: COMPUTER_NETWORK_STAGE_H,
      fps: 8,
      title: 'Computer Network DES',
      subtitle: 'Packet motion, queue buildup, bottlenecks, and fan-out routing policy semantics.',
      background: '#f8fafc',
      recordEveryTicks: 1,
    });
    for (const f of frames) recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
    recorder.setCharts(charts);
    await recorder.finish();
  },
  examples: [
    {
      name: 'small-enterprise',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'computer-network',
        description: 'Small enterprise packet network with two clients, edge/core routers, and a server.',
        parameters: {builtin: 'small-enterprise'},
      },
    },
    {
      name: 'bottleneck-lab',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'computer-network',
        description: 'HTTP, UDP, and TCP flows over a narrow WAN link to expose traffic buildup and bottlenecks.',
        parameters: {builtin: 'bottleneck-lab'},
      },
    },
  ],
};

registerModel(adapter);

function problemFromBuiltin(builtin: ComputerNetworkParams['builtin']): ComputerNetworkProblem {
  switch (builtin ?? 'small-enterprise') {
    case 'bottleneck-lab': return buildBottleneckComputerNetworkProblem();
    case 'small-enterprise':
    default: return buildDefaultComputerNetworkProblem();
  }
}

function fmt(x: number): string {
  return Number.isFinite(x) ? x.toFixed(3) : 'n/a';
}
