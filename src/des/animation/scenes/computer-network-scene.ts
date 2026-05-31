'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/computer-network-scene.rs   (module des::animation::scenes::computer_network_scene)
// 1:1 file move. Builds frames + charts for the computer-network DES animation.
//
// Declarations → Rust:
//   const *_STAGE_W/H, NET_*/PANEL_* consts  -> `pub const`/`const` (f64)
//   const PROTOCOL_COLOR: Record<NetworkProtocol,string> -> `match` on enum NetworkProtocol -> &str
//   interface Point                          -> struct Point { x: f64, y: f64 }
//   function buildComputerNetworkAnimation   -> pub fn -> { frames, charts } (a small struct)
//   function build*/draw*/layoutNodes/...     -> free fns
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `draw*(shapes: Shape[], ..)` push into a shared array -> `fn(shapes: &mut Vec<Shape>, ..)`.
//   - `layoutNodes` returns `Map<string, Point>` -> `HashMap<String, Point>`.
//   - `PROTOCOL_COLOR` keyed by the `NetworkProtocol` literal union -> `match` on the enum.
//   - interpolated `rgb(..)` color strings -> `format!`.
//   - imports problem/result types from ../../general/computer-network -> `use crate::des::general::computer_network::*`.
// =============================================================================

// =============================================================================
// Computer-network scene: packet motion, queue buildup, bottlenecks, and
// fan-out policy semantics.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {
  ComputerNetworkProblem,
  ComputerNetworkResult,
  NetworkLinkSpec,
  NetworkPacketSnapshot,
  NetworkProtocol,
  NetworkTimeSample,
} from '../../general/computer-network';

export const COMPUTER_NETWORK_STAGE_W = 1200;
export const COMPUTER_NETWORK_STAGE_H = 760;

const NET_X = 40;
const NET_Y = 50;
const NET_W = 820;
const NET_H = 440;
const PANEL_X = 890;
const PANEL_Y = 50;
const PANEL_W = 270;
const PANEL_H = 440;

const PROTOCOL_COLOR: Record<NetworkProtocol, string> = {
  raw: '#64748b',
  tcp: '#2563eb',
  udp: '#16a34a',
  http: '#dc2626',
};

interface Point {
  x: number;
  y: number;
}

export function buildComputerNetworkAnimation(
  problem: ComputerNetworkProblem,
  result: ComputerNetworkResult,
): {
  frames: Array<{t: number; tick: number} & Omit<Frame, 't' | 'tick'>>;
  charts: ChartSpec[];
} {
  const normalized = normalizeLinks(problem.links);
  const coords = layoutNodes(problem);
  const packetTraces = result.deliveredPacketsTrace.concat(result.droppedPacketsTrace);
  const frames: Array<{t: number; tick: number} & Omit<Frame, 't' | 'tick'>> = [];
  const samples = result.timeSeries.length > 0 ? result.timeSeries : [{
    tMs: 0,
    generatedPackets: result.generatedPackets,
    deliveredPackets: result.deliveredPackets,
    droppedPackets: result.droppedPackets,
    activePackets: result.activePackets,
    nodeQueues: {},
    linkInFlight: {},
    linkUtilization: {},
  }];

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const frame = buildComputerNetworkFrame(sample, i, {
      problem,
      links: normalized,
      coords,
      result,
      packetTraces,
    });
    frames.push({t: sample.tMs, tick: i, ...frame});
  }

  return {frames, charts: buildComputerNetworkCharts(result)};
}

function buildComputerNetworkFrame(
  sample: NetworkTimeSample,
  tick: number,
  args: {
    problem: ComputerNetworkProblem;
    links: NetworkLinkSpec[];
    coords: Map<string, Point>;
    result: ComputerNetworkResult;
    packetTraces: NetworkPacketSnapshot[];
  },
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const {problem, links, coords, result, packetTraces} = args;
  const linkStats = new Map(result.linkStats.map(l => [l.id, l]));
  const nodeStats = new Map(result.nodeStats.map(n => [n.id, n]));
  const topBottleneckId = result.bottlenecks[0]?.id;

  shapes.push({kind: 'rect', x: 0, y: 0, w: COMPUTER_NETWORK_STAGE_W, h: COMPUTER_NETWORK_STAGE_H, fill: '#f8fafc'});
  shapes.push({kind: 'text', x: 40, y: 28, text: 'Network topology', fontSize: 18, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: 220, y: 28, text: `t=${sample.tMs.toFixed(0)}ms  active=${sample.activePackets}  delivered=${sample.deliveredPackets}  dropped=${sample.droppedPackets}`, fontSize: 12, fill: '#475569'});

  shapes.push({kind: 'rect', x: NET_X, y: NET_Y, w: NET_W, h: NET_H, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 6});

  for (const link of links) drawLink(shapes, link, coords, linkStats, sample, topBottleneckId);
  for (const pkt of packetTraces) drawPacketIfActive(shapes, pkt, sample.tMs, coords);
  for (const node of problem.nodes) drawNode(shapes, node.id, node.kind, coords, sample, nodeStats, topBottleneckId);

  drawLegend(shapes);
  drawMetricsPanel(shapes, result, sample);
  drawFanoutPolicyPanel(shapes, tick);

  const top = result.bottlenecks[0];
  return {
    shapes,
    caption: top
      ? `top bottleneck ${top.kind}:${top.id} (${top.reason}); active=${sample.activePackets}, dropped=${sample.droppedPackets}`
      : `active=${sample.activePackets}, dropped=${sample.droppedPackets}`,
  };
}

function drawLink(
  shapes: Shape[],
  link: NetworkLinkSpec,
  coords: Map<string, Point>,
  linkStats: Map<string, ComputerNetworkResult['linkStats'][number]>,
  sample: NetworkTimeSample,
  topBottleneckId: string | undefined,
): void {
  const a = coords.get(link.from);
  const b = coords.get(link.to);
  if (!a || !b) return;
  const isReverse = link.id.includes(':rev');
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.max(1, Math.sqrt(dx * dx + dy * dy));
  const offset = isReverse ? -7 : 7;
  const n = {x: -dy / len * offset, y: dx / len * offset};
  const aa = {x: a.x + n.x, y: a.y + n.y};
  const bb = {x: b.x + n.x, y: b.y + n.y};
  const st = linkStats.get(link.id);
  const util = sample.linkUtilization[link.id] ?? st?.utilization ?? 0;
  const drops = st?.droppedPackets ?? 0;
  const isTop = link.id === topBottleneckId;
  const stroke = drops > 0 ? '#dc2626' : util > 0.9 ? '#f97316' : util > 0.5 ? '#eab308' : '#64748b';
  const width = isTop ? 5 : 1.4 + 4 * Math.min(1, util);
  shapes.push({kind: 'line', x1: aa.x, y1: aa.y, x2: bb.x, y2: bb.y, stroke, strokeWidth: width, opacity: isReverse ? 0.45 : 0.85, dasharray: isReverse ? '4,4' : undefined});
  drawArrow(shapes, aa, bb, stroke);
  const mid = {x: (aa.x + bb.x) / 2, y: (aa.y + bb.y) / 2};
  const inflight = sample.linkInFlight[link.id] ?? st?.finalInFlight ?? 0;
  const qW = Math.min(74, 8 + inflight * 0.7);
  shapes.push({kind: 'rect', x: mid.x - qW / 2, y: mid.y - 27, w: qW, h: 12, fill: queueColor(inflight, st?.queueLimitPackets ?? 1), stroke: '#ffffff', strokeWidth: 1, rx: 3,
    title: `${link.id}: in flight ${inflight}, util ${(util * 100).toFixed(1)}%`});
  if (!isReverse) {
    shapes.push({kind: 'text', x: mid.x, y: mid.y - 33, text: `${link.id}`, fontSize: 9, fill: '#334155', anchor: 'middle'});
  }
}

function drawNode(
  shapes: Shape[],
  id: string,
  kind: string,
  coords: Map<string, Point>,
  sample: NetworkTimeSample,
  nodeStats: Map<string, ComputerNetworkResult['nodeStats'][number]>,
  topBottleneckId: string | undefined,
): void {
  const p = coords.get(id);
  if (!p) return;
  const st = nodeStats.get(id);
  const q = sample.nodeQueues[id] ?? st?.finalQueue ?? 0;
  const isTop = id === topBottleneckId;
  const fill = kind === 'host' ? '#0ea5e9' : kind === 'switch' ? '#7c3aed' : '#0f766e';
  shapes.push({kind: 'circle', x: p.x, y: p.y, r: isTop ? 28 : 24, fill, stroke: isTop ? '#dc2626' : '#0f172a', strokeWidth: isTop ? 4 : 2,
    title: `${id} (${kind}) queue=${q}, dropped=${st?.droppedPackets ?? 0}`});
  shapes.push({kind: 'text', x: p.x, y: p.y + 4, text: shortNodeLabel(id), fontSize: 10, fill: '#ffffff', anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: p.x, y: p.y + 42, text: id, fontSize: 10, fill: '#0f172a', anchor: 'middle'});
  const qH = Math.min(54, q * 2);
  shapes.push({kind: 'rect', x: p.x + 30, y: p.y + 24 - qH, w: 10, h: qH, fill: queueColor(q, st?.queueLimitPackets ?? 1), stroke: '#334155', strokeWidth: 1, rx: 2});
}

function drawPacketIfActive(
  shapes: Shape[],
  pkt: NetworkPacketSnapshot,
  tMs: number,
  coords: Map<string, Point>,
): void {
  const end = pkt.deliveredAtMs ?? pkt.droppedAtMs;
  if (end === undefined || tMs < pkt.createdAtMs || tMs > end) return;
  if (pkt.hops.length < 2) return;
  const progress = Math.max(0, Math.min(0.999, (tMs - pkt.createdAtMs) / Math.max(1, end - pkt.createdAtMs)));
  const segFloat = progress * (pkt.hops.length - 1);
  const seg = Math.min(pkt.hops.length - 2, Math.floor(segFloat));
  const local = segFloat - seg;
  const a = coords.get(pkt.hops[seg]);
  const b = coords.get(pkt.hops[seg + 1]);
  if (!a || !b) return;
  const x = a.x + (b.x - a.x) * local;
  const y = a.y + (b.y - a.y) * local;
  shapes.push({kind: 'circle', x, y, r: pkt.protocol === 'http' ? 4.5 : 3.7, fill: PROTOCOL_COLOR[pkt.protocol], stroke: '#ffffff', strokeWidth: 1, opacity: pkt.droppedAtMs ? 0.55 : 0.9,
    title: `packet ${pkt.packetId} ${pkt.protocol} ${pkt.flowId}`});
}

function drawMetricsPanel(shapes: Shape[], result: ComputerNetworkResult, sample: NetworkTimeSample): void {
  shapes.push({kind: 'rect', x: PANEL_X, y: PANEL_Y, w: PANEL_W, h: 145, fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: PANEL_X + 14, y: PANEL_Y + 24, text: 'Metrics', fontSize: 16, fill: '#f8fafc', fontWeight: 'bold'});
  const rows = [
    `offered ${result.offeredLoadMbps.toFixed(2)} Mbps`,
    `wire ${result.throughputMbps.toFixed(2)} Mbps`,
    `goodput ${result.goodputMbps.toFixed(2)} Mbps`,
    `delivery ${(result.deliveryRatio * 100).toFixed(1)}%`,
    `active now ${sample.activePackets} / max ${result.maxActivePackets}`,
  ];
  for (let i = 0; i < rows.length; i++) {
    shapes.push({kind: 'text', x: PANEL_X + 16, y: PANEL_Y + 52 + i * 18, text: rows[i], fontSize: 12, fill: '#cbd5e1'});
  }

  shapes.push({kind: 'rect', x: PANEL_X, y: PANEL_Y + 160, w: PANEL_W, h: 100, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: PANEL_X + 14, y: PANEL_Y + 183, text: 'Top bottlenecks', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  for (let i = 0; i < Math.min(3, result.bottlenecks.length); i++) {
    const b = result.bottlenecks[i];
    shapes.push({kind: 'text', x: PANEL_X + 16, y: PANEL_Y + 207 + i * 18, text: `${i + 1}. ${b.kind}:${b.id} ${b.reason}`, fontSize: 11, fill: i === 0 ? '#dc2626' : '#334155'});
  }
}

function drawFanoutPolicyPanel(shapes: Shape[], tick: number): void {
  const x = PANEL_X;
  const y = PANEL_Y + 280;
  shapes.push({kind: 'rect', x, y, w: PANEL_W, h: 210, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: x + 14, y: y + 24, text: 'Fan-out order / bias', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: x + 14, y: y + 43, text: 'Competitive out-connections; queues stay FIFO.', fontSize: 10, fill: '#64748b'});

  const rows = [
    {policy: 'random', desc: 'shuffle each entity', pick: ['B', 'C', 'A', 'B', 'A', 'C'][tick % 6], color: '#2563eb'},
    {policy: 'round-robin', desc: 'rotate declared order', pick: ['A', 'B', 'C'][tick % 3], color: '#16a34a'},
    {policy: 'ordered', desc: 'priority / bias', pick: 'A', color: '#dc2626'},
  ];
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    const yy = y + 72 + r * 44;
    shapes.push({kind: 'text', x: x + 14, y: yy, text: row.policy, fontSize: 12, fill: '#0f172a', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: x + 14, y: yy + 15, text: row.desc, fontSize: 9, fill: '#64748b'});
    const labels = ['A', 'B', 'C'];
    for (let i = 0; i < labels.length; i++) {
      const lx = x + 140 + i * 36;
      const active = row.pick === labels[i];
      shapes.push({kind: 'circle', x: lx, y: yy - 4, r: active ? 13 : 10, fill: active ? row.color : '#e2e8f0', stroke: active ? '#0f172a' : '#94a3b8', strokeWidth: 1.2});
      shapes.push({kind: 'text', x: lx, y: yy, text: labels[i], fontSize: 10, fill: active ? '#ffffff' : '#334155', anchor: 'middle', fontWeight: 'bold'});
    }
  }
}

function drawLegend(shapes: Shape[]): void {
  const x = NET_X + 16;
  const y = NET_Y + NET_H - 68;
  shapes.push({kind: 'rect', x, y, w: 250, h: 50, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 5});
  const entries: Array<[NetworkProtocol, string]> = [['http', 'HTTP'], ['tcp', 'TCP'], ['udp', 'UDP'], ['raw', 'raw']];
  for (let i = 0; i < entries.length; i++) {
    const [p, label] = entries[i];
    const xx = x + 18 + i * 58;
    shapes.push({kind: 'circle', x: xx, y: y + 22, r: 5, fill: PROTOCOL_COLOR[p]});
    shapes.push({kind: 'text', x: xx + 10, y: y + 26, text: label, fontSize: 10, fill: '#334155'});
  }
}

function buildComputerNetworkCharts(result: ComputerNetworkResult): ChartSpec[] {
  const t = result.timeSeries.map(s => s.tMs);
  return [
    {
      x: 40, y: 520, w: 360, h: 200,
      title: 'Traffic buildup',
      yMin: 0,
      series: [
        {label: 'active', color: '#2563eb', t, y: result.timeSeries.map(s => s.activePackets)},
        {label: 'dropped', color: '#dc2626', t, y: result.timeSeries.map(s => s.droppedPackets)},
        {label: 'delivered', color: '#16a34a', t, y: result.timeSeries.map(s => s.deliveredPackets)},
      ],
    },
    {
      x: 430, y: 520, w: 360, h: 200,
      title: 'Top link utilization',
      yMin: 0,
      yMax: 1,
      series: topUtilizationSeries(result),
    },
    {
      x: 820, y: 520, w: 330, h: 200,
      title: 'Flow goodput (Mbps)',
      yMin: 0,
      series: result.flowStats.map((f, i) => ({
        label: f.id,
        color: [PROTOCOL_COLOR.http, PROTOCOL_COLOR.tcp, PROTOCOL_COLOR.udp, '#7c3aed'][i % 4],
        t: [0, result.totalSimulatedMs],
        y: [f.goodputMbps, f.goodputMbps],
      })),
    },
  ];
}

function topUtilizationSeries(result: ComputerNetworkResult): ChartSpec['series'] {
  const topLinks = result.linkStats
    .filter(l => l.deliveredPackets > 0 || l.droppedPackets > 0)
    .sort((a, b) => b.utilization - a.utilization)
    .slice(0, 3);
  const colors = ['#dc2626', '#f97316', '#2563eb'];
  return topLinks.map((l, i) => ({
    label: l.id,
    color: colors[i],
    t: result.timeSeries.map(s => s.tMs),
    y: result.timeSeries.map(s => s.linkUtilization[l.id] ?? l.utilization),
  }));
}

function layoutNodes(problem: ComputerNetworkProblem): Map<string, Point> {
  const ids = problem.nodes.map(n => n.id);
  const preset: Record<string, Point> = {
    'web-client': {x: 145, y: 165},
    'telemetry-client': {x: 145, y: 355},
    'edge': {x: 370, y: 260},
    'wan-router': {x: 590, y: 260},
    'api-server': {x: 780, y: 260},
    'client-a': {x: 145, y: 165},
    'client-b': {x: 145, y: 355},
    'edge-1': {x: 370, y: 260},
    'core-1': {x: 590, y: 260},
    'server': {x: 780, y: 260},
  };
  const out = new Map<string, Point>();
  if (ids.every(id => preset[id])) {
    for (const id of ids) out.set(id, preset[id]);
    return out;
  }

  const cx = NET_X + NET_W / 2;
  const cy = NET_Y + NET_H / 2;
  const r = Math.min(NET_W, NET_H) * 0.36;
  for (let i = 0; i < ids.length; i++) {
    const a = -Math.PI / 2 + 2 * Math.PI * i / Math.max(1, ids.length);
    out.set(ids[i], {x: cx + r * Math.cos(a), y: cy + r * Math.sin(a)});
  }
  return out;
}

function normalizeLinks(links: NetworkLinkSpec[]): NetworkLinkSpec[] {
  const out: NetworkLinkSpec[] = [];
  const ids = new Set<string>();
  for (const link of links) {
    out.push({...link, bidirectional: false});
    ids.add(link.id);
    if (!link.bidirectional) continue;
    let reverseId = `${link.id}:rev`;
    let i = 2;
    while (ids.has(reverseId)) reverseId = `${link.id}:rev${i++}`;
    ids.add(reverseId);
    out.push({...link, id: reverseId, from: link.to, to: link.from, bidirectional: false});
  }
  return out;
}

function drawArrow(shapes: Shape[], a: Point, b: Point, color: string): void {
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  const tip = {x: b.x - 28 * Math.cos(ang), y: b.y - 28 * Math.sin(ang)};
  const left = {x: tip.x - 9 * Math.cos(ang - Math.PI / 6), y: tip.y - 9 * Math.sin(ang - Math.PI / 6)};
  const right = {x: tip.x - 9 * Math.cos(ang + Math.PI / 6), y: tip.y - 9 * Math.sin(ang + Math.PI / 6)};
  shapes.push({kind: 'path', d: `M ${tip.x} ${tip.y} L ${left.x} ${left.y} L ${right.x} ${right.y} Z`, fill: color, stroke: color, opacity: 0.9});
}

function queueColor(q: number, cap: number): string {
  const p = q / Math.max(1, cap);
  if (p > 0.85) return '#dc2626';
  if (p > 0.45) return '#f97316';
  if (q > 0) return '#eab308';
  return '#e2e8f0';
}

function shortNodeLabel(id: string): string {
  return id.split(/[-_]/).map(s => s[0]?.toUpperCase() ?? '').join('').slice(0, 3) || id.slice(0, 2).toUpperCase();
}
