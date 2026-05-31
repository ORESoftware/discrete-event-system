'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/soccer-ipmip-solver-scene.rs   (module des::animation::scenes::soccer_ipmip_solver_scene)
// 1:1 file move. Builds frames + charts for the soccer IP/MIP solver-entity DES animation.
//
// Declarations → Rust:
//   const SOCCER_IPMIP_SOLVER_W/H, SOLVER_FRAMES_PER_EVENT  -> `pub const`/`const`
//   interface NodeBox / Edge                                -> struct
//   const NODES / EDGES                                     -> `const`/`static` slices
//   const NODE_BY_ID = new Map(NODES.map(..))               -> `HashMap<&str, &NodeBox>` (key by id)
//   helper fns center/interpolate/fmt/pct/trunc/traceEvent/eventPath/activeSegment/
//             edgeKey/nodeFill/drawArrow/drawNode/drawMetric/drawSidePanel -> fns
//   function soccerIPMIPSolverFrameCount / buildSoccerIPMIPSolverFrame /
//            buildSoccerIPMIPSolverCharts                   -> pub fns
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `NODE_BY_ID` (a `Map` built from NODES) -> `HashMap` keyed by node id (`&str`/`String`).
//   - `edgeKey(a, b)` builds a string key -> use a `(String, String)` tuple key instead.
//   - `draw*(shapes: Shape[], ..)` -> `&mut Vec<Shape>`.
//   - `trunc(s: string | undefined, n)` -> `Option<&str>`.
//   - imports IPMIPTraceEvent/IPMIPSolution from ../../general/ip-mip-des ->
//     `use crate::des::general::ip_mip_des::*`.
// =============================================================================

// =============================================================================
// Soccer IP/MIP solver-entity animation.
//
// This scene shows the optimization side of the 7v7 rotation example as a DES
// network: a model source emits branch/cut subproblem tokens, stationary solver
// entities process LP relaxations and candidates, and the schedule sink receives
// the best feasible incumbent. It is intentionally paired with the pitch
// animation so external users can review both the plan and the solver path that
// produced it.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {IPMIPTraceEvent, IPMIPSolution} from '../../general/ip-mip-des';

export const SOCCER_IPMIP_SOLVER_W = 1100;
export const SOCCER_IPMIP_SOLVER_H = 680;
export const SOLVER_FRAMES_PER_EVENT = 7;

interface NodeBox {
  id: string;
  label: string;
  role: string;
  x: number;
  y: number;
  w: number;
  h: number;
  kind: 'source' | 'station' | 'sink';
}

interface Edge {
  from: string;
  to: string;
  label: string;
}

const NODES: NodeBox[] = [
  {id: 'source', label: 'Model Source', role: 'emits soccer IP/MIP root problem', x: 36, y: 105, w: 135, h: 58, kind: 'source'},
  {id: 'ip-search-controller', label: 'Search Controller', role: 'frontier of branch/cut subproblems', x: 220, y: 105, w: 150, h: 58, kind: 'station'},
  {id: 'ip-lp-relaxation', label: 'LP Relaxation', role: 'internal simplex relaxation station', x: 430, y: 105, w: 150, h: 58, kind: 'station'},
  {id: 'ip-rounding-repair', label: 'Rounding Repair', role: 'candidate movable repair', x: 655, y: 52, w: 155, h: 58, kind: 'station'},
  {id: 'ip-cut-generator', label: 'Cut Generator', role: 'valid inequality station', x: 655, y: 158, w: 155, h: 58, kind: 'station'},
  {id: 'ip-incumbent', label: 'Incumbent', role: 'best feasible solution anchor', x: 882, y: 52, w: 150, h: 58, kind: 'sink'},
  {id: 'ip-node-decision', label: 'Node Decision', role: 'prune, cut, branch, or accept', x: 882, y: 158, w: 150, h: 58, kind: 'station'},
  {id: 'sink', label: 'Schedule Sink', role: 'feasible lineup schedule', x: 882, y: 296, w: 150, h: 58, kind: 'sink'},
];

const EDGES: Edge[] = [
  {from: 'source', to: 'ip-search-controller', label: 'root'},
  {from: 'ip-search-controller', to: 'ip-lp-relaxation', label: 'node'},
  {from: 'ip-lp-relaxation', to: 'ip-rounding-repair', label: 'relax'},
  {from: 'ip-rounding-repair', to: 'ip-incumbent', label: 'candidate'},
  {from: 'ip-lp-relaxation', to: 'ip-cut-generator', label: 'relax'},
  {from: 'ip-cut-generator', to: 'ip-node-decision', label: 'cut'},
  {from: 'ip-lp-relaxation', to: 'ip-node-decision', label: 'relax'},
  {from: 'ip-node-decision', to: 'ip-search-controller', label: 'children'},
  {from: 'ip-node-decision', to: 'sink', label: 'complete'},
  {from: 'ip-incumbent', to: 'sink', label: 'incumbent'},
];

const NODE_BY_ID = new Map(NODES.map(n => [n.id, n]));

function center(n: NodeBox): {x: number; y: number} {
  return {x: n.x + n.w / 2, y: n.y + n.h / 2};
}

function interpolate(a: {x: number; y: number}, b: {x: number; y: number}, u: number): {x: number; y: number} {
  return {x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u};
}

function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : String(x);
}

function pct(x: number): string {
  return Number.isFinite(x) ? `${(100 * x).toFixed(1)}%` : 'n/a';
}

function trunc(s: string | undefined, n: number): string {
  if (!s) return '';
  return s.length <= n ? s : s.slice(0, Math.max(0, n - 1)) + '...';
}

function traceEvent(solution: IPMIPSolution, eventIndex: number): IPMIPTraceEvent | null {
  if (solution.trace.length === 0) return null;
  return solution.trace[Math.max(0, Math.min(solution.trace.length - 1, eventIndex))];
}

function eventPath(ev: IPMIPTraceEvent | null): string[] {
  if (!ev) return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-rounding-repair', 'ip-incumbent', 'sink'];
  if (ev.action === 'cut') {
    return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-cut-generator', 'ip-node-decision', 'ip-search-controller'];
  }
  if (ev.action === 'branch') {
    return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-node-decision', 'ip-search-controller'];
  }
  if (ev.action === 'incumbent') {
    return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-node-decision', 'ip-incumbent', 'sink'];
  }
  if (ev.action === 'prune' && ev.reason?.includes('incumbent')) {
    return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-rounding-repair', 'ip-incumbent', 'ip-node-decision', 'sink'];
  }
  return ['source', 'ip-search-controller', 'ip-lp-relaxation', 'ip-node-decision', 'sink'];
}

function activeSegment(path: string[], subframe: number): {from: string; to: string; u: number} {
  const segments = Math.max(1, path.length - 1);
  const progress = Math.min(0.999, Math.max(0, subframe / Math.max(1, SOLVER_FRAMES_PER_EVENT - 1)));
  const raw = progress * segments;
  const idx = Math.min(segments - 1, Math.floor(raw));
  return {from: path[idx], to: path[idx + 1], u: raw - idx};
}

function edgeKey(a: string, b: string): string {
  return `${a}->${b}`;
}

function nodeFill(n: NodeBox, active: boolean): string {
  if (active) return '#fde68a';
  if (n.kind === 'source') return '#dbeafe';
  if (n.kind === 'sink') return '#dcfce7';
  return '#f8fafc';
}

function drawArrow(shapes: Shape[], a: {x: number; y: number}, b: {x: number; y: number}, color: string, width: number, opacity = 1): void {
  shapes.push({kind: 'line', x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: color, strokeWidth: width, opacity});
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const size = 8 + width;
  const p1 = {x: b.x - Math.cos(angle - 0.45) * size, y: b.y - Math.sin(angle - 0.45) * size};
  const p2 = {x: b.x - Math.cos(angle + 0.45) * size, y: b.y - Math.sin(angle + 0.45) * size};
  shapes.push({kind: 'path', d: `M ${b.x} ${b.y} L ${p1.x} ${p1.y} L ${p2.x} ${p2.y} Z`, fill: color, opacity});
}

function drawNode(shapes: Shape[], n: NodeBox, active: boolean, used: boolean): void {
  const stroke = active ? '#92400e' : used ? '#2563eb' : '#64748b';
  const width = active ? 3 : used ? 2 : 1.2;
  shapes.push({
    kind: 'rect', x: n.x, y: n.y, w: n.w, h: n.h, rx: 8,
    fill: nodeFill(n, active), stroke, strokeWidth: width,
    title: n.role,
  });
  shapes.push({kind: 'text', x: n.x + n.w / 2, y: n.y + 24, text: n.label, fontSize: 13, fill: '#0f172a', anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: n.x + n.w / 2, y: n.y + 43, text: n.kind, fontSize: 10, fill: '#475569', anchor: 'middle'});
}

function drawMetric(shapes: Shape[], x: number, y: number, label: string, value: string, color = '#0f172a'): void {
  shapes.push({kind: 'text', x, y, text: label, fontSize: 11, fill: '#64748b'});
  shapes.push({kind: 'text', x: x + 158, y, text: value, fontSize: 12, fill: color, anchor: 'end', fontWeight: 'bold'});
}

function drawSidePanel(shapes: Shape[], solution: IPMIPSolution, ev: IPMIPTraceEvent | null, frameIndex: number): void {
  const x = 36;
  const y = 390;
  const w = 996;
  shapes.push({kind: 'rect', x, y, w, h: 106, rx: 8, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1});
  shapes.push({kind: 'text', x: x + 18, y: y + 24, text: 'Solver status', fontSize: 15, fill: '#0f172a', fontWeight: 'bold'});
  drawMetric(shapes, x + 18, y + 48, 'status', solution.status, solution.status === 'optimal' ? '#15803d' : '#b45309');
  drawMetric(shapes, x + 18, y + 70, 'objective', fmt(solution.z, 4));
  drawMetric(shapes, x + 18, y + 92, 'gap', pct(solution.gap));
  drawMetric(shapes, x + 238, y + 48, 'LP backend', solution.lpAlgorithm);
  drawMetric(shapes, x + 238, y + 70, 'LP solves', String(solution.lpSolves));
  drawMetric(shapes, x + 238, y + 92, 'elapsed', `${solution.elapsedMs}ms`);
  drawMetric(shapes, x + 458, y + 48, 'nodes', String(solution.nodesExplored));
  drawMetric(shapes, x + 458, y + 70, 'cuts', String(solution.cutsAdded));
  drawMetric(shapes, x + 458, y + 92, 'candidates', String(solution.candidatesTried));

  const ex = x + 690;
  shapes.push({kind: 'text', x: ex, y: y + 24, text: `Event ${frameIndex + 1} / ${Math.max(1, solution.trace.length)}`, fontSize: 15, fill: '#0f172a', fontWeight: 'bold'});
  if (ev) {
    shapes.push({kind: 'text', x: ex, y: y + 48, text: `node ${ev.nodeId}, depth ${ev.depth}, action ${ev.action}`, fontSize: 12, fill: '#334155'});
    shapes.push({kind: 'text', x: ex, y: y + 70, text: `LP z ${ev.lpZ === null ? 'n/a' : fmt(ev.lpZ, 4)}, fractional vars ${ev.fractional.length}`, fontSize: 12, fill: '#334155'});
    shapes.push({kind: 'text', x: ex, y: y + 92, text: trunc(ev.reason, 44), fontSize: 12, fill: '#64748b'});
  } else {
    shapes.push({kind: 'text', x: ex, y: y + 55, text: 'root model solved without branch trace events', fontSize: 12, fill: '#334155'});
  }
}

export function soccerIPMIPSolverFrameCount(solution: IPMIPSolution): number {
  return Math.max(1, solution.trace.length) * SOLVER_FRAMES_PER_EVENT;
}

export function buildSoccerIPMIPSolverFrame(
  solution: IPMIPSolution,
  frameIndex: number,
): Omit<Frame, 't' | 'tick'> {
  const eventCount = Math.max(1, solution.trace.length);
  const eventIndex = Math.min(eventCount - 1, Math.floor(frameIndex / SOLVER_FRAMES_PER_EVENT));
  const subframe = frameIndex % SOLVER_FRAMES_PER_EVENT;
  const ev = traceEvent(solution, eventIndex);
  const path = eventPath(ev);
  const seg = activeSegment(path, subframe);
  const activeEdge = edgeKey(seg.from, seg.to);
  const usedNodes = new Set(path);
  const shapes: Shape[] = [];

  shapes.push({kind: 'rect', x: 0, y: 0, w: SOCCER_IPMIP_SOLVER_W, h: SOCCER_IPMIP_SOLVER_H, fill: '#f8fafc'});
  shapes.push({kind: 'text', x: 36, y: 36, text: '7v7 IP/MIP solver entities', fontSize: 22, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: 36, y: 60, text: 'Internal LP relaxation station + DES branch/cut token flow', fontSize: 13, fill: '#475569'});

  shapes.push({kind: 'rect', x: 24, y: 78, w: 1020, h: 292, rx: 8, fill: '#eef2ff', stroke: '#c7d2fe', strokeWidth: 1});
  shapes.push({kind: 'text', x: 42, y: 96, text: 'station graph', fontSize: 11, fill: '#4338ca', fontWeight: 'bold'});

  for (const e of EDGES) {
    const a = NODE_BY_ID.get(e.from)!;
    const b = NODE_BY_ID.get(e.to)!;
    const ca = center(a);
    const cb = center(b);
    const isActive = edgeKey(e.from, e.to) === activeEdge;
    drawArrow(shapes, ca, cb, isActive ? '#2563eb' : '#94a3b8', isActive ? 4 : 1.6, isActive ? 0.95 : 0.55);
    const mid = interpolate(ca, cb, 0.52);
    shapes.push({kind: 'text', x: mid.x, y: mid.y - 6, text: e.label, fontSize: 9, fill: isActive ? '#1d4ed8' : '#64748b', anchor: 'middle'});
  }

  for (const n of NODES) drawNode(shapes, n, n.id === seg.from || n.id === seg.to, usedNodes.has(n.id));

  const a = NODE_BY_ID.get(seg.from)!;
  const b = NODE_BY_ID.get(seg.to)!;
  const p = interpolate(center(a), center(b), seg.u);
  shapes.push({kind: 'circle', x: p.x, y: p.y, r: 13, fill: '#f97316', stroke: '#fff7ed', strokeWidth: 3, title: 'movable solver token'});
  shapes.push({kind: 'text', x: p.x, y: p.y + 4, text: 'x', fontSize: 11, fill: '#ffffff', anchor: 'middle', fontWeight: 'bold'});

  drawSidePanel(shapes, solution, ev, eventIndex);

  const usage = Object.entries(solution.lpAlgorithmUsage).map(([k, v]) => `${k}=${v}`).join(', ') || 'none';
  shapes.push({kind: 'text', x: 36, y: 526, text: `LP usage: ${usage}`, fontSize: 12, fill: '#334155'});
  shapes.push({kind: 'text', x: 36, y: 546, text: `Incumbent source: ${solution.incumbentSource ?? 'none'}`, fontSize: 12, fill: '#334155'});

  return {
    shapes,
    caption: ev
      ? `event=${eventIndex + 1}/${solution.trace.length} node=${ev.nodeId} action=${ev.action} status=${solution.status}`
      : `status=${solution.status} objective=${fmt(solution.z, 4)}`,
  };
}

export function buildSoccerIPMIPSolverCharts(solution: IPMIPSolution): ChartSpec[] {
  const events = solution.trace.length ? solution.trace : [null];
  const t = events.map((_, i) => i);
  const lpBound = events.map(ev => ev && ev.lpZ !== null ? ev.lpZ : solution.bestBound);
  const fractional = events.map(ev => ev ? ev.fractional.length : 0);
  const incumbent = events.map((_, i) => i === events.length - 1 ? solution.z : solution.z);
  return [
    {
      x: 650, y: 520, w: 180, h: 110,
      title: 'LP bound / incumbent',
      series: [
        {label: 'lp bound', color: '#2563eb', t, y: lpBound},
        {label: 'incumbent', color: '#16a34a', t, y: incumbent},
      ],
    },
    {
      x: 852, y: 520, w: 180, h: 110,
      title: 'fractional vars',
      yMin: 0,
      series: [{label: 'fractional', color: '#f97316', t, y: fractional}],
    },
  ];
}
