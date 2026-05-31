// RUST MIGRATION: Target module `src/des/general/adapters/network_flow_adapter.rs`.
// RUST MIGRATION: Convert max-flow, traffic, and smart-traffic adapter registrations into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Encode graph edges, traffic networks, signals, traces, and animation data as `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Use `Result<_, ValidationError>` for graph normalization, capacity, lane, and signal validation.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/network-flow-adapter.rs
//   (module des::general::adapters::network_flow_adapter)
// 1:1 file move. Registers max-flow / traffic-flow / smart-traffic JSON adapters
// (with SVG-frame animation builders).
//
// Declarations → Rust:
//   type MaxFlowAdapterParams = MaxFlowParams & {…}  -> struct composing the fields
//             (intersection types: flatten with #[serde(flatten)] or duplicate)
//   const *Schema: ParamSchema                       -> serde + validator metadata
//   fn networkForTraffic / nodeLayout / trafficPoint / trafficVector /
//      trafficPolyline / trafficCarColor / fmtMetric / buildTeachingMaxFlowParams /
//      hasDirectMaxFlowParams / normalizeMaxFlowParams -> plain `fn` helpers
//   registerModel<P,R>({ id, run, summarize, writeCsv, animate }) -> struct impl of a
//             ModelAdapter trait; `animate` is an async method building Shape frames
//
// Conversion notes (file-specific):
//   - GotChA: `(trafficSchema as Extract<ParamSchema, {kind:'object'}>).fields`
//     narrows a union by cast -> in Rust match on the schema enum variant instead.
//   - Shapes are pushed into `Shape[]` arrays (animation/types) -> Vec<Shape>; the
//     Shape union -> an enum.
//   - `network.lanes.find(...)`, `.reduce(...)` over arrays -> iterator find/fold.
//   - `row.laneOccupancy[lane.id] ?? 0`, `car.leaderGapM !== undefined` -> HashMap
//     get + Option; lane/node id maps are string-keyed -> HashMap<String, _>.
//   - `throw new Error` in normalizeMaxFlowParams (missing fields) -> Result/validation.
//   - async run/animate use withLogger + FrameRecorder -> async fns; many `?? default`
//     -> Option::unwrap_or.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {registerModel} from '../des-registry';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {csvRow, framesPath, validationLine, withLogger, writeCsvLines} from './adapter-utils';
import {
  buildFiveIntersectionTrafficNetwork,
  MaxFlowParams,
  MaxFlowResult,
  runMaxFlow,
  runTrafficFlow,
  TrafficNetwork,
  TrafficParams,
  TrafficResult,
} from '../network-flow';
import {runSmartTrafficFlow, SmartTrafficParams, SmartTrafficResult} from '../smart-traffic-flow';

type MaxFlowAdapterParams = MaxFlowParams & {
  builtin?: 'textbook';
  problem?: MaxFlowParams;
};

const flowEdgeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    from: {kind: 'number', integer: true, min: 0},
    to: {kind: 'number', integer: true, min: 0},
    capacity: {kind: 'number', min: 0},
    name: {kind: 'string'},
  },
  required: ['from', 'to', 'capacity'],
};

const maxFlowProblemFields: Record<string, ParamSchema> = {
  numNodes: {kind: 'number', integer: true, min: 2},
  source: {kind: 'number', integer: true, min: 0},
  sink: {kind: 'number', integer: true, min: 0},
  edges: {kind: 'array', items: flowEdgeSchema, minLength: 1},
  maxAugmentations: {kind: 'number', integer: true, min: 1},
  nodeCoordinates: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2}},
  nodeNames: {kind: 'array', items: {kind: 'string'}},
};

const maxFlowProblemSchema: ParamSchema = {
  kind: 'object',
  fields: maxFlowProblemFields,
  required: ['numNodes', 'source', 'sink', 'edges'],
};

const maxFlowSchema: ParamSchema = {
  kind: 'object',
  fields: {
    builtin: {kind: 'string', enum: ['textbook'], default: 'textbook'},
    problem: maxFlowProblemSchema,
    ...maxFlowProblemFields,
  },
  required: [],
};

const trafficNodeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    kind: {kind: 'string', enum: ['source', 'intersection', 'sink']},
    x: {kind: 'number'},
    y: {kind: 'number'},
  },
  required: ['id', 'kind', 'x', 'y'],
};

const trafficLaneSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    from: {kind: 'string'},
    to: {kind: 'string'},
    lengthM: {kind: 'number', min: 0},
    speedLimitMps: {kind: 'number', min: 0},
    capacity: {kind: 'number', integer: true, min: 1},
  },
  required: ['id', 'from', 'to', 'lengthM', 'speedLimitMps'],
};

const trafficSignalPhaseSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    greenLanes: {kind: 'array', items: {kind: 'string'}, minLength: 1},
    durationSec: {kind: 'number', min: 0},
  },
  required: ['name', 'greenLanes', 'durationSec'],
};

const trafficSignalSchema: ParamSchema = {
  kind: 'object',
  fields: {
    nodeId: {kind: 'string'},
    phases: {kind: 'array', items: trafficSignalPhaseSchema, minLength: 1},
    offsetSec: {kind: 'number', default: 0},
  },
  required: ['nodeId', 'phases'],
};

const trafficSourceSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    nodeId: {kind: 'string'},
    ratePerMin: {kind: 'number', min: 0},
    destinationSinkIds: {kind: 'array', items: {kind: 'string'}, minLength: 1},
  },
  required: ['id', 'nodeId', 'ratePerMin'],
};

const trafficSinkSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    nodeId: {kind: 'string'},
  },
  required: ['id', 'nodeId'],
};

const trafficNetworkSchema: ParamSchema = {
  kind: 'object',
  fields: {
    nodes: {kind: 'array', items: trafficNodeSchema, minLength: 1},
    lanes: {kind: 'array', items: trafficLaneSchema, minLength: 1},
    signals: {kind: 'array', items: trafficSignalSchema},
    sources: {kind: 'array', items: trafficSourceSchema, minLength: 1},
    sinks: {kind: 'array', items: trafficSinkSchema, minLength: 1},
  },
  required: ['nodes', 'lanes', 'sources', 'sinks'],
};

const trafficSchema: ParamSchema = {
  kind: 'object',
  fields: {
    builtin: {kind: 'string', enum: ['five-intersection'], default: 'five-intersection'},
    network: trafficNetworkSchema,
    durationSec: {kind: 'number', min: 1, default: 180},
    dtSec: {kind: 'number', min: 0.01, default: 1},
    seed: {kind: 'number', integer: true, default: 19},
    maxCars: {kind: 'number', integer: true, min: 1, max: 299, default: 250},
    carLengthM: {kind: 'number', min: 0.1, default: 4.8},
    carWidthM: {kind: 'number', min: 0.1, default: 1.8},
    laneWidthM: {kind: 'number', min: 0.1, default: 3.7},
    minGapM: {kind: 'number', min: 0, default: 2.5},
    maxAccelMps2: {kind: 'number', min: 0.1, default: 2.2},
    maxDecelMps2: {kind: 'number', min: 0.1, default: 4},
    maxJerkMps3: {kind: 'number', min: 0.1, default: 6},
    reactionTimeSec: {kind: 'number', min: 0, default: 0.8},
    timeHeadwaySec: {kind: 'number', min: 0, default: 1.1},
    gridCellSizeM: {kind: 'number', min: 0.01, default: 0.3048},
    gridLookAheadM: {kind: 'number', min: 0.1},
    spawnRateMultiplier: {kind: 'number', min: 0, default: 1},
  },
  required: ['durationSec', 'dtSec', 'seed', 'maxCars'],
};

const smartTrafficSchema: ParamSchema = {
  kind: 'object',
  fields: {
    ...(trafficSchema as Extract<ParamSchema, {kind: 'object'}>).fields,
    dtSec: {kind: 'number', min: 0.01, default: 0.1},
    smartCarPoolSize: {kind: 'number', integer: true, min: 1, max: 10000, default: 250},
    actorShuffleSeed: {kind: 'number', integer: true},
    accidentRiskScale: {kind: 'number', min: 0, default: 0},
    accidentProbability: {kind: 'number', min: 0, max: 1, default: 0},
    accidentAccelBoostMps2: {kind: 'number', min: 0, default: 10},
    accidentFaultDurationSec: {kind: 'number', min: 0.1, default: 1},
    distancePreferenceSpread: {kind: 'number', min: 0, max: 1.5, default: 0},
    startPreferenceSpread: {kind: 'number', min: 0, max: 1.5, default: 0},
    accidentFlashSeconds: {kind: 'number', min: 0.1, default: 2},
  },
  required: ['durationSec', 'dtSec', 'seed', 'maxCars'],
};

function networkForTraffic(params: TrafficParams): TrafficNetwork {
  return params.network ?? buildFiveIntersectionTrafficNetwork();
}

function nodeLayout(params: MaxFlowParams): Array<[number, number]> {
  if (params.nodeCoordinates) return params.nodeCoordinates;
  const out: Array<[number, number]> = [];
  const n = params.numNodes;
  for (let i = 0; i < n; i++) {
    const a = 2 * Math.PI * i / Math.max(1, n);
    out.push([450 + 260 * Math.cos(a), 250 + 170 * Math.sin(a)]);
  }
  return out;
}

function trafficPoint(network: TrafficNetwork, laneId: string, positionM: number): [number, number] {
  const lane = network.lanes.find(l => l.id === laneId);
  if (!lane) return [0, 0];
  const a = network.nodes.find(n => n.id === lane.from);
  const b = network.nodes.find(n => n.id === lane.to);
  if (!a || !b) return [0, 0];
  const margin = 70;
  const sx = (x: number) => margin + x * 115;
  const sy = (y: number) => margin + y * 150;
  const q = Math.max(0, Math.min(1, positionM / lane.lengthM));
  return [sx(a.x) + q * (sx(b.x) - sx(a.x)), sy(a.y) + q * (sy(b.y) - sy(a.y))];
}

function trafficVector(network: TrafficNetwork, laneId: string): {ux: number; uy: number; nx: number; ny: number} {
  const lane = network.lanes.find(l => l.id === laneId);
  if (!lane) return {ux: 1, uy: 0, nx: 0, ny: 1};
  const a = network.nodes.find(n => n.id === lane.from);
  const b = network.nodes.find(n => n.id === lane.to);
  if (!a || !b) return {ux: 1, uy: 0, nx: 0, ny: 1};
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.max(1e-9, Math.hypot(dx, dy));
  const ux = dx / len;
  const uy = dy / len;
  return {ux, uy, nx: -uy, ny: ux};
}

function trafficPolyline(network: TrafficNetwork, laneId: string, fromM: number, toM: number, offsetPx = 0): {x1: number; y1: number; x2: number; y2: number} {
  const a = trafficPoint(network, laneId, fromM);
  const b = trafficPoint(network, laneId, toM);
  const v = trafficVector(network, laneId);
  return {
    x1: a[0] + v.nx * offsetPx,
    y1: a[1] + v.ny * offsetPx,
    x2: b[0] + v.nx * offsetPx,
    y2: b[1] + v.ny * offsetPx,
  };
}

function trafficCarColor(accelerationMps2: number, speedMps: number): string {
  if (speedMps < 0.35) return '#991b1b';
  if (accelerationMps2 < -1.25) return '#dc2626';
  if (accelerationMps2 < -0.25) return '#f97316';
  if (accelerationMps2 > 0.75) return '#16a34a';
  return '#2563eb';
}

function fmtMetric(x: number | undefined, digits = 1): string {
  return x === undefined || !Number.isFinite(x) ? 'n/a' : x.toFixed(digits);
}

function buildTeachingMaxFlowParams(): MaxFlowParams {
  return {
    numNodes: 6,
    source: 0,
    sink: 5,
    nodeNames: ['s', 'a', 'b', 'c', 'd', 't'],
    nodeCoordinates: [[90, 260], [260, 160], [260, 360], [520, 160], [520, 360], [760, 260]],
    edges: [
      {from: 0, to: 1, capacity: 16, name: 's-a'},
      {from: 0, to: 2, capacity: 13, name: 's-b'},
      {from: 1, to: 2, capacity: 10, name: 'a-b'},
      {from: 2, to: 1, capacity: 4, name: 'b-a'},
      {from: 1, to: 3, capacity: 12, name: 'a-c'},
      {from: 3, to: 2, capacity: 9, name: 'c-b'},
      {from: 2, to: 4, capacity: 14, name: 'b-d'},
      {from: 4, to: 3, capacity: 7, name: 'd-c'},
      {from: 3, to: 5, capacity: 20, name: 'c-t'},
      {from: 4, to: 5, capacity: 4, name: 'd-t'},
    ],
  };
}

function hasDirectMaxFlowParams(params: MaxFlowAdapterParams): boolean {
  return params.numNodes !== undefined ||
    params.source !== undefined ||
    params.sink !== undefined ||
    (Array.isArray(params.edges) && params.edges.length > 0);
}

function normalizeMaxFlowParams(params: MaxFlowAdapterParams): MaxFlowParams {
  if (params.problem !== undefined) return params.problem;
  if (hasDirectMaxFlowParams(params)) {
    if (
      params.numNodes === undefined ||
      params.source === undefined ||
      params.sink === undefined ||
      params.edges === undefined
    ) {
      throw new Error('max-flow: direct parameters require numNodes, source, sink, and edges');
    }
    return params;
  }
  return buildTeachingMaxFlowParams();
}

registerModel<MaxFlowAdapterParams, MaxFlowResult>({
  id: 'max-flow',
  description: 'Maximum s-t flow via augmenting-path DES ticks with min-cut validation.',
  schema: maxFlowSchema,
  async run(params, runtime) {
    const actual = normalizeMaxFlowParams(params);
    return withLogger(runtime, logger => {
      logger?.log({kind: 'max-flow-start', level: 'info', nodes: actual.numNodes, edges: actual.edges.length});
      const result = runMaxFlow(actual, logger);
      logger?.log({kind: 'max-flow-finish', level: 'info', maxFlow: result.maxFlow, augmentations: result.trace.length});
      return result;
    });
  },
  summarize(r) {
    return [
      'MAX FLOW',
      '------------------------',
      `  nodes=${r.params.numNodes} edges=${r.params.edges.length}`,
      `  max flow=${r.maxFlow.toFixed(4)} augmentations=${r.trace.length}`,
      `  min cut capacity=${r.minCut.capacity.toFixed(4)} cut edges=[${r.minCut.cutEdges.join(', ')}]`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['edge,from,to,capacity,flow,residual'];
    r.edgeFlows.forEach((e, i) => lines.push(csvRow([e.name ?? i, e.from, e.to, e.capacity, e.flow, e.residual])));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const params = r.params;
    const {htmlPath, frames} = framesPath(runtime, 'max-flow');
    const coords = nodeLayout(params);
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 900, height: 560, fps: 4, title: 'Max-flow optimization', subtitle: 'One augmenting path per DES tick'});
    const frameRows = r.trace.length > 0 ? r.trace : [{iter: 0, pathNodes: [], pathEdges: [], bottleneck: 0, value: 0}];
    for (const row of frameRows) {
      rec.frame(row.iter, row.iter, () => {
        const active = new Set(row.pathEdges);
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 900, h: 560, fill: '#f8fafc'},
          {kind: 'text', x: 450, y: 34, text: `Flow value ${row.value.toFixed(2)}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
        ];
        params.edges.forEach((e, i) => {
          const [x1, y1] = coords[e.from];
          const [x2, y2] = coords[e.to];
          shapes.push({kind: 'line', x1, y1, x2, y2, stroke: active.has(i) ? '#f97316' : '#64748b', strokeWidth: active.has(i) ? 4 : 2, opacity: 0.8});
          shapes.push({kind: 'text', x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 6, text: `${r.edgeFlows[i].flow.toFixed(0)}/${e.capacity.toFixed(0)}`, fontSize: 10, anchor: 'middle', fill: '#334155'});
        });
        coords.forEach(([x, y], i) => {
          const fill = i === params.source ? '#22c55e' : i === params.sink ? '#ef4444' : '#dbeafe';
          shapes.push({kind: 'circle', x, y, r: 20, fill, stroke: '#0f172a'});
          shapes.push({kind: 'text', x, y: y + 4, text: params.nodeNames?.[i] ?? String(i), fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
        });
        return {shapes, caption: row.iter > 0 ? `augment ${row.iter}: bottleneck=${row.bottleneck.toFixed(2)} path=${row.pathNodes.join(' -> ')}` : 'no augmenting path'};
      });
    }
    rec.setCharts([{x: 80, y: 390, w: 740, h: 120, title: 'Flow value by augmentation', series: [{label: 'max-flow', color: '#2563eb', t: r.trace.map(t => t.iter), y: r.trace.map(t => t.value)}]}]);
    await rec.finish();
  },
  examples: [{name: 'six-node teaching network', spec: {$schema: 'des/model-spec/v1', model: 'max-flow', parameters: {
    numNodes: 6,
    source: 0,
    sink: 5,
    nodeNames: ['s', 'a', 'b', 'c', 'd', 't'],
    nodeCoordinates: [[90, 260], [260, 160], [260, 360], [520, 160], [520, 360], [760, 260]],
    edges: [
      {from: 0, to: 1, capacity: 16}, {from: 0, to: 2, capacity: 13},
      {from: 1, to: 2, capacity: 10}, {from: 2, to: 1, capacity: 4},
      {from: 1, to: 3, capacity: 12}, {from: 3, to: 2, capacity: 9},
      {from: 2, to: 4, capacity: 14}, {from: 4, to: 3, capacity: 7},
      {from: 3, to: 5, capacity: 20}, {from: 4, to: 5, capacity: 4},
    ],
  }, runtime: {animate: true}}}],
});

registerModel<TrafficParams, TrafficResult>({
  id: 'traffic-flow',
  description: 'Continuous-time traffic flow on a stationary grid with moving cars, signals, sources, and sinks.',
  schema: trafficSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'traffic-flow-start', level: 'info', durationSec: params.durationSec, maxCars: params.maxCars});
      const result = runTrafficFlow(params, logger);
      logger?.log({kind: 'traffic-flow-finish', level: 'info', entered: result.entered, exited: result.exited, maxActiveCars: result.maxActiveCars});
      return result;
    });
  },
  summarize(r) {
    return [
      'TRAFFIC FLOW',
      '------------------------',
      `  network nodes=${r.network.nodes.length} lanes=${r.network.lanes.length} intersections=${r.network.nodes.filter(n => n.kind === 'intersection').length}`,
      `  entered=${r.entered} exited=${r.exited} active=${r.finalCars.length} dropped=${r.dropped}`,
      `  max active cars=${r.maxActiveCars} mean speed=${r.meanSpeedMps.toFixed(2)} m/s mean travel=${r.meanTravelTimeSec.toFixed(1)} s`,
      `  grid cell=${r.cellStats.cellSizeM.toFixed(4)} m active cells=${r.cellStats.activeCells} created stations=${r.cellStats.createdCellStations}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['tick,time_sec,active_cars,entered,exited,mean_speed_mps,mean_travel_time_sec,queue_length'];
    for (const t of r.trace) lines.push(csvRow([t.tick, t.timeSec, t.activeCars, t.entered, t.exited, t.meanSpeedMps, t.meanTravelTimeSec, t.queueLength]));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'traffic-flow');
    const network = networkForTraffic(params);
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 1040,
      height: 720,
      fps: 12,
      title: 'Traffic micro-simulation',
      subtitle: 'One-foot sparse cells, signal control, reaction delay, jerk-limited car following',
      background: '#f8fafc',
      recordEveryTicks: Math.max(1, Math.floor(1 / params.dtSec)),
    });
    const activeMax = Math.max(1, ...r.trace.map(t => t.activeCars));
    const gridMax = Math.max(1, ...r.trace.map(t => t.activeGridCells));
    const speedMax = Math.max(1, ...r.trace.map(t => t.meanSpeedMps));
    const queueMax = Math.max(1, ...r.trace.map(t => t.queueLength));
    for (const row of r.trace) {
      rec.frame(row.timeSec, row.tick, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 1040, h: 720, fill: '#f8fafc'},
          {kind: 'rect', x: 26, y: 24, w: 812, h: 442, fill: '#ffffff', stroke: '#d1d5db', rx: 6},
          {kind: 'rect', x: 858, y: 24, w: 156, h: 442, fill: '#ffffff', stroke: '#d1d5db', rx: 6},
          {kind: 'text', x: 872, y: 50, text: 'Live State', fontSize: 14, fontWeight: 'bold', fill: '#0f172a'},
          {kind: 'text', x: 872, y: 78, text: `time ${row.timeSec.toFixed(1)} s`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 100, text: `active ${row.activeCars}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 122, text: `exited ${row.exited}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 144, text: `queue ${row.queueLength}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 166, text: `cells ${row.activeGridCells}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 188, text: `mean v ${row.meanSpeedMps.toFixed(2)}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 232, text: 'Car Color', fontSize: 13, fontWeight: 'bold', fill: '#0f172a'},
          {kind: 'rect', x: 872, y: 248, w: 14, h: 10, fill: '#16a34a', rx: 2},
          {kind: 'text', x: 892, y: 258, text: 'accelerating', fontSize: 11, fill: '#334155'},
          {kind: 'rect', x: 872, y: 270, w: 14, h: 10, fill: '#2563eb', rx: 2},
          {kind: 'text', x: 892, y: 280, text: 'cruising', fontSize: 11, fill: '#334155'},
          {kind: 'rect', x: 872, y: 292, w: 14, h: 10, fill: '#f97316', rx: 2},
          {kind: 'text', x: 892, y: 302, text: 'braking', fontSize: 11, fill: '#334155'},
          {kind: 'rect', x: 872, y: 314, w: 14, h: 10, fill: '#991b1b', rx: 2},
          {kind: 'text', x: 892, y: 324, text: 'stopped', fontSize: 11, fill: '#334155'},
        ];
        const hotLane = row.cars.length > 0
          ? row.cars.reduce((best, car) => (row.laneOccupancy[car.laneId] ?? 0) > (row.laneOccupancy[best] ?? 0) ? car.laneId : best, row.cars[0].laneId)
          : undefined;
        for (const lane of network.lanes) {
          const base = trafficPolyline(network, lane.id, 0, lane.lengthM);
          const occ = row.laneOccupancy[lane.id] ?? 0;
          const opacity = lane.id === hotLane ? 0.9 : 0.56;
          shapes.push({kind: 'line', ...base, stroke: '#334155', strokeWidth: 12, opacity});
          shapes.push({kind: 'line', ...base, stroke: '#e5e7eb', strokeWidth: 2, dasharray: '7,7', opacity: 0.95});
          if (occ > 0) {
            const v = trafficVector(network, lane.id);
            const labelPoint = trafficPoint(network, lane.id, lane.lengthM * 0.5);
            shapes.push({kind: 'text', x: labelPoint[0] + v.nx * 17, y: labelPoint[1] + v.ny * 17 + 4, text: String(occ), fontSize: 10, anchor: 'middle', fill: '#0f172a', fontWeight: 'bold'});
          }
          if ((row.laneOccupancy[lane.id] ?? 0) > 0) {
            const approxCells = Math.max(1, Math.min(18, Math.round((row.laneOccupancy[lane.id] ?? 0) * 0.8)));
            for (let k = 0; k < approxCells; k++) {
              const from = lane.lengthM * k / approxCells;
              const to = lane.lengthM * (k + 0.55) / approxCells;
              const cell = trafficPolyline(network, lane.id, from, to, -8);
              shapes.push({kind: 'line', ...cell, stroke: '#93c5fd', strokeWidth: 2, opacity: 0.32});
            }
          }
        }
        for (const node of network.nodes) {
          const x = 70 + node.x * 115, y = 70 + node.y * 150;
          const phase = row.signalPhases[node.id];
          const fill = node.kind === 'intersection' ? (phase === 'side' ? '#f59e0b' : '#22c55e') : node.kind === 'source' ? '#bfdbfe' : '#bbf7d0';
          shapes.push({kind: 'rect', x: x - 15, y: y - 15, w: 30, h: 30, fill, stroke: '#0f172a', strokeWidth: 1.2, rx: 5, title: phase ? `${node.id} phase=${phase}` : node.id});
          shapes.push({kind: 'text', x, y: y - 19, text: node.id, fontSize: 10, anchor: 'middle', fill: '#334155'});
        }
        for (const car of row.cars) {
          const [x, y] = trafficPoint(network, car.laneId, car.positionM);
          const v = trafficVector(network, car.laneId);
          const color = trafficCarColor(car.accelerationMps2, car.speedMps);
          const body = {
            x: x - 5 + v.ux * 3,
            y: y - 3 + v.uy * 3,
            w: 10,
            h: 6,
          };
          shapes.push({
            kind: 'rect',
            ...body,
            fill: color,
            stroke: '#ffffff',
            strokeWidth: 1,
            rx: 2,
            title: `car ${car.id} lane=${car.laneId} v=${car.speedMps.toFixed(2)} m/s a=${car.accelerationMps2.toFixed(2)} m/s^2 jerk=${car.jerkMps3.toFixed(2)} m/s^3 gap=${fmtMetric(car.leaderGapM)} m cells=${car.gridCellCount}`,
          });
          if (car.leaderGapM !== undefined && car.leaderGapM < 16) {
            const front = trafficPoint(network, car.laneId, car.positionM + Math.min(car.leaderGapM, 14));
            shapes.push({kind: 'line', x1: x, y1: y, x2: front[0], y2: front[1], stroke: '#ef4444', strokeWidth: 1, opacity: 0.45});
          }
        }
        shapes.push({kind: 'text', x: 432, y: 54, text: `t=${row.timeSec.toFixed(1)}s   active=${row.activeCars}   exited=${row.exited}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
        return {
          shapes,
          caption: `mean speed=${row.meanSpeedMps.toFixed(2)} m/s | queue=${row.queueLength} | entered=${row.entered} | dropped=${r.dropped} | active 1ft cells=${row.activeGridCells}`,
        };
      });
    }
    rec.setCharts([
      {x: 42, y: 496, w: 220, h: 150, title: 'Active cars', yMin: 0, yMax: activeMax, series: [{label: 'active', color: '#2563eb', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.activeCars)}]},
      {x: 286, y: 496, w: 220, h: 150, title: 'Mean speed', yMin: 0, yMax: Math.ceil(speedMax), series: [{label: 'm/s', color: '#16a34a', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.meanSpeedMps)}]},
      {x: 530, y: 496, w: 220, h: 150, title: 'Queue length', yMin: 0, yMax: queueMax, series: [{label: 'stopped', color: '#dc2626', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.queueLength)}]},
      {x: 774, y: 496, w: 220, h: 150, title: 'Active 1ft cells', yMin: 0, yMax: gridMax, series: [{label: 'cells', color: '#7c3aed', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.activeGridCells)}]},
    ]);
    await rec.finish();
  },
  examples: [{name: 'five intersections under signal control', spec: {$schema: 'des/model-spec/v1', model: 'traffic-flow', parameters: {builtin: 'five-intersection', durationSec: 180, dtSec: 1, seed: 19, maxCars: 250, spawnRateMultiplier: 1}, runtime: {animate: true}}}],
});

registerModel<SmartTrafficParams, SmartTrafficResult>({
  id: 'smart-traffic-flow',
  description: 'Traffic flow where each car is a smart movable participant with its own shuffled runTimeStep.',
  schema: smartTrafficSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({
        kind: 'smart-traffic-flow-start',
        level: 'info',
        durationSec: params.durationSec,
        maxCars: params.maxCars,
        smartCarPoolSize: params.smartCarPoolSize ?? params.maxCars,
      });
      const result = runSmartTrafficFlow(params, logger);
      logger?.log({
        kind: 'smart-traffic-flow-finish',
        level: 'info',
        entered: result.entered,
        exited: result.exited,
        maxActiveCars: result.maxActiveCars,
        totalSmartMovableRuns: result.execution.totalSmartMovableRuns,
      });
      return result;
    });
  },
  summarize(r) {
    return [
      'SMART TRAFFIC FLOW',
      '------------------------',
      `  network nodes=${r.network.nodes.length} lanes=${r.network.lanes.length} intersections=${r.network.nodes.filter(n => n.kind === 'intersection').length}`,
      `  participants=${r.execution.participantCount} smart movables=${r.execution.smartMovableCount} shuffled=${r.execution.shuffledByRunner}`,
      `  entered=${r.entered} exited=${r.exited} crashed=${r.crashed} active=${r.finalCars.length} dropped=${r.dropped}`,
      `  max active cars=${r.maxActiveCars} mean speed=${r.meanSpeedMps.toFixed(2)} m/s mean travel=${r.meanTravelTimeSec.toFixed(1)} s`,
      `  accidents=${r.accidents.length} accident risk scale=${(r.params.accidentRiskScale ?? r.params.accidentProbability ?? 0).toFixed(2)}`,
      `  distance preference spread=${(r.params.distancePreferenceSpread ?? 0).toFixed(2)} start preference spread=${(r.params.startPreferenceSpread ?? 0).toFixed(2)}`,
      `  smart movable runs=${r.execution.totalSmartMovableRuns} max per tick=${r.execution.maxSmartMovableRunsPerTick}`,
      `  grid cell=${r.cellStats.cellSizeM.toFixed(4)} m active cells=${r.cellStats.activeCells} created stations=${r.cellStats.createdCellStations}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['tick,time_sec,active_cars,scheduled_smart_cars,smart_movable_runs,entered,exited,crashed,accidents_this_tick,mean_speed_mps,mean_travel_time_sec,queue_length'];
    for (const t of r.trace) {
      lines.push(csvRow([t.tick, t.timeSec, t.activeCars, t.scheduledSmartCars, t.smartMovableRuns, t.entered, t.exited, t.crashed, t.accidents.length, t.meanSpeedMps, t.meanTravelTimeSec, t.queueLength]));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(r, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'smart-traffic-flow');
    const network = networkForTraffic(params);
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 1040,
      height: 720,
      fps: 12,
      title: 'Smart-movable traffic',
      subtitle: 'Each active car runs its own shuffled runTimeStep; the world station commits proposals afterward',
      background: '#f8fafc',
      recordEveryTicks: 1,
    });
    const activeMax = Math.max(1, ...r.trace.map(t => t.activeCars));
    const speedMax = Math.max(1, ...r.trace.map(t => t.meanSpeedMps));
    const runMax = Math.max(1, ...r.trace.map(t => t.smartMovableRuns));
    const crashMax = Math.max(1, ...r.trace.map(t => t.crashed));
    const allAccidents = r.trace.flatMap(t => t.accidents);
    const flashSeconds = params.accidentFlashSeconds ?? 2;
    for (const row of r.trace) {
      rec.frame(row.timeSec, row.tick, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 1040, h: 720, fill: '#f8fafc'},
          {kind: 'rect', x: 26, y: 24, w: 812, h: 442, fill: '#ffffff', stroke: '#d1d5db', rx: 6},
          {kind: 'rect', x: 858, y: 24, w: 156, h: 442, fill: '#ffffff', stroke: '#d1d5db', rx: 6},
          {kind: 'text', x: 872, y: 50, text: 'Runner', fontSize: 14, fontWeight: 'bold', fill: '#0f172a'},
          {kind: 'text', x: 872, y: 78, text: `participants ${r.execution.participantCount}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 100, text: `pool ${r.execution.smartMovableCount}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 122, text: `ran ${row.smartMovableRuns}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 144, text: `scheduled ${row.scheduledSmartCars}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 166, text: `crashed ${row.crashed}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 188, text: `cells ${row.activeGridCells}`, fontSize: 12, fill: '#334155'},
          {kind: 'text', x: 872, y: 208, text: 'Run Order', fontSize: 13, fontWeight: 'bold', fill: '#0f172a'},
        ];
        const orderText = row.actorRunOrder.length > 0 ? row.actorRunOrder.slice(0, 8).join(' ') : 'none';
        shapes.push({kind: 'text', x: 872, y: 232, text: orderText, fontSize: 10, fill: '#334155'});
        for (const lane of network.lanes) {
          const base = trafficPolyline(network, lane.id, 0, lane.lengthM);
          const occ = row.laneOccupancy[lane.id] ?? 0;
          shapes.push({kind: 'line', ...base, stroke: '#334155', strokeWidth: 12, opacity: occ > 0 ? 0.78 : 0.5});
          shapes.push({kind: 'line', ...base, stroke: '#e5e7eb', strokeWidth: 2, dasharray: '7,7', opacity: 0.95});
          if (occ > 0) {
            const v = trafficVector(network, lane.id);
            const p = trafficPoint(network, lane.id, lane.lengthM * 0.5);
            shapes.push({kind: 'text', x: p[0] + v.nx * 17, y: p[1] + v.ny * 17 + 4, text: String(occ), fontSize: 10, anchor: 'middle', fill: '#0f172a', fontWeight: 'bold'});
          }
        }
        for (const node of network.nodes) {
          const x = 70 + node.x * 115, y = 70 + node.y * 150;
          const phase = row.signalPhases[node.id];
          const fill = node.kind === 'intersection' ? (phase === 'side' ? '#f59e0b' : '#22c55e') : node.kind === 'source' ? '#bfdbfe' : '#bbf7d0';
          shapes.push({kind: 'rect', x: x - 15, y: y - 15, w: 30, h: 30, fill, stroke: '#0f172a', strokeWidth: 1.2, rx: 5, title: phase ? `${node.id} phase=${phase}` : node.id});
          shapes.push({kind: 'text', x, y: y - 19, text: node.id, fontSize: 10, anchor: 'middle', fill: '#334155'});
        }
        for (const car of row.cars) {
          const [x, y] = trafficPoint(network, car.laneId, car.positionM);
          const v = trafficVector(network, car.laneId);
          const color = trafficCarColor(car.accelerationMps2, car.speedMps);
          shapes.push({
            kind: 'rect',
            x: x - 5 + v.ux * 3,
            y: y - 3 + v.uy * 3,
            w: 10,
            h: 6,
            fill: color,
            stroke: '#ffffff',
            strokeWidth: 1,
            rx: 2,
            title: `actor ${car.actorId} car ${car.id} v=${car.speedMps.toFixed(2)} a=${car.accelerationMps2.toFixed(2)} follow=${car.distancePreference.toFixed(2)} start=${car.startPreference.toFixed(2)} runCount=${car.runCount} gap=${fmtMetric(car.leaderGapM)} m`,
          });
        }
        const recentAccidents = allAccidents.filter(a => a.timeSec <= row.timeSec && row.timeSec - a.timeSec <= flashSeconds);
        for (const accident of recentAccidents) {
          const [x, y] = trafficPoint(network, accident.laneId, accident.positionM);
          shapes.push({kind: 'circle', x, y, r: 16, fill: '#fde68a', stroke: '#f97316', strokeWidth: 2, opacity: 0.82});
          shapes.push({kind: 'text', x, y: y + 7, text: '⚡', fontSize: 22, anchor: 'middle', fontWeight: 'bold', fill: '#b45309'});
        }
        shapes.push({kind: 'text', x: 432, y: 54, text: `t=${row.timeSec.toFixed(1)}s   active=${row.activeCars}   car runTimeStep calls=${row.smartMovableRuns}`, fontSize: 19, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
        return {
          shapes,
          caption: `mean speed=${row.meanSpeedMps.toFixed(2)} m/s | queue=${row.queueLength} | actor runs=${row.smartMovableRuns}/${row.scheduledSmartCars} | crashes=${row.crashed} | entered=${row.entered} | dropped=${r.dropped}`,
        };
      });
    }
    rec.setCharts([
      {x: 42, y: 496, w: 220, h: 150, title: 'Active cars', yMin: 0, yMax: activeMax, series: [{label: 'active', color: '#2563eb', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.activeCars)}]},
      {x: 286, y: 496, w: 220, h: 150, title: 'Mean speed', yMin: 0, yMax: Math.ceil(speedMax), series: [{label: 'm/s', color: '#16a34a', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.meanSpeedMps)}]},
      {x: 530, y: 496, w: 220, h: 150, title: 'Smart runs', yMin: 0, yMax: runMax, series: [{label: 'calls', color: '#7c3aed', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.smartMovableRuns)}]},
      {x: 774, y: 496, w: 220, h: 150, title: 'Crashes', yMin: 0, yMax: crashMax, series: [{label: 'crashes', color: '#dc2626', t: r.trace.map(t => t.timeSec), y: r.trace.map(t => t.crashed)}]},
    ]);
    await rec.finish();
  },
  examples: [{name: 'smart movable cars on five intersections', spec: {$schema: 'des/model-spec/v1', model: 'smart-traffic-flow', parameters: {
    builtin: 'five-intersection',
    durationSec: 180,
    dtSec: 0.1,
    seed: 19,
    actorShuffleSeed: 2026,
    maxCars: 250,
    smartCarPoolSize: 400,
    spawnRateMultiplier: 3,
    accidentRiskScale: 16,
    accidentAccelBoostMps2: 12,
    accidentFaultDurationSec: 1,
    distancePreferenceSpread: 0.54,
    startPreferenceSpread: 0.65,
    accidentFlashSeconds: 2.5,
  }, runtime: {animate: true}}}],
});
