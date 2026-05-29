'use strict';

// =============================================================================
// Shortest-path-DES scene: graph nodes, directed edges, propagating
// "wave" of distance updates per tick. Each node is colored by its
// current distance estimate (cool/blue = far, hot/yellow = close to source);
// nodes that received a NEW best distance this tick are highlighted; the
// edge over which the relaxation flowed is drawn thicker.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {Graph, SPResult} from '../../general/shortest-path-des';

export const STAGE_W = 1000;
export const STAGE_H = 640;
const VIEW_X = 60, VIEW_Y = 40, VIEW_W = 600, VIEW_H = 560;
const META_X = 700, META_Y = 40, META_W = 260, META_H = 560;

function distanceColor(d: number, maxFinite: number): string {
  if (!isFinite(d)) return '#475569';
  const t = maxFinite > 0 ? d / maxFinite : 0;
  // Cool to warm gradient (low distance = bright yellow, high = blue).
  const r = Math.round(250 - 230 * t);
  const g = Math.round(220 - 80 * t);
  const b = Math.round(60 + 180 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function buildShortestPathFrame(
  t: number, tick: number, args: {
    graph: Graph;
    distanceNow: number[];
    waveEvents: Array<{from: number; to: number; newDistance: number; improved: boolean}>;
    source: number;
    iteration: number;
    algorithm: 'bellman-ford-des' | 'dijkstra-des';
  },
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const {graph} = args;
  const coords = graph.coordinates ?? [];
  // Figure out the projection.
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < graph.numNodes; i++) {
    const c = coords[i] ?? [50, 50];
    if (c[0] < xMin) xMin = c[0]; if (c[0] > xMax) xMax = c[0];
    if (c[1] < yMin) yMin = c[1]; if (c[1] > yMax) yMax = c[1];
  }
  const pad = 30;
  const sx = (VIEW_W - 2 * pad) / Math.max(1e-9, xMax - xMin);
  const sy = (VIEW_H - 2 * pad) / Math.max(1e-9, yMax - yMin);
  const project = (i: number): [number, number] => {
    const c = coords[i] ?? [50, 50];
    return [VIEW_X + pad + (c[0] - xMin) * sx, VIEW_Y + pad + (c[1] - yMin) * sy];
  };

  // Find max finite distance for the colour scale.
  let maxFinite = 0;
  for (const d of args.distanceNow) if (isFinite(d) && d > maxFinite) maxFinite = d;

  // Frame.
  shapes.push({kind: 'rect', x: VIEW_X, y: VIEW_Y, w: VIEW_W, h: VIEW_H,
               fill: '#0b1220', stroke: '#334155', strokeWidth: 1, rx: 6});

  // Edges: thin grey by default, thick yellow if a wave fired this tick.
  const firedSet = new Set(args.waveEvents.map(e => `${e.from}->${e.to}`));
  const improvedSet = new Set(args.waveEvents.filter(e => e.improved).map(e => `${e.from}->${e.to}`));
  for (let u = 0; u < graph.numNodes; u++) {
    for (const e of graph.edges[u]) {
      const [x1, y1] = project(u);
      const [x2, y2] = project(e.to);
      const key = `${u}->${e.to}`;
      const isImproved = improvedSet.has(key);
      const isFired = firedSet.has(key);
      shapes.push({kind: 'line', x1, y1, x2, y2,
                   stroke: isImproved ? '#fde68a' : (isFired ? '#facc15' : '#475569'),
                   strokeWidth: isImproved ? 3 : (isFired ? 2 : 1),
                   opacity: isFired ? 1 : 0.5});
      // Edge weight label at midpoint.
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      shapes.push({kind: 'text', x: mx, y: my - 4,
                   text: e.weight.toFixed(1), fontSize: 9, fill: '#94a3b8',
                   anchor: 'middle'});
    }
  }

  // Nodes.
  for (let v = 0; v < graph.numNodes; v++) {
    const [x, y] = project(v);
    const color = v === args.source ? '#22c55e' : distanceColor(args.distanceNow[v], maxFinite);
    shapes.push({kind: 'circle', x, y, r: 14, fill: color,
                 stroke: '#0b1220', strokeWidth: 2,
                 title: `${graph.nodeNames?.[v] ?? v}: distance = ${isFinite(args.distanceNow[v]) ? args.distanceNow[v].toFixed(2) : '∞'}`});
    shapes.push({kind: 'text', x, y: y + 4,
                 text: graph.nodeNames?.[v] ?? v.toString(),
                 fontSize: 11, fill: '#0b1220', anchor: 'middle', fontWeight: 'bold'});
    // Distance label below.
    const dStr = isFinite(args.distanceNow[v]) ? args.distanceNow[v].toFixed(1) : '∞';
    shapes.push({kind: 'text', x, y: y + 26, text: `d=${dStr}`,
                 fontSize: 10, fill: '#94a3b8', anchor: 'middle'});
  }

  // Sidebar.
  shapes.push({kind: 'rect', x: META_X, y: META_Y, w: META_W, h: META_H,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 32,
               text: 'Shortest Path DES', fontSize: 22, fill: '#f1f5f9',
               anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 56,
               text: args.algorithm.toUpperCase(), fontSize: 12, fill: '#94a3b8',
               anchor: 'middle'});
  let y0 = META_Y + 100;
  const line = (text: string, color: string) => {
    shapes.push({kind: 'text', x: META_X + 20, y: y0, text, fontSize: 12, fill: color});
    y0 += 22;
  };
  line(`Iteration ${args.iteration}`, '#cbd5e1');
  line(`Source = ${graph.nodeNames?.[args.source] ?? args.source}`, '#22c55e');
  line(`Waves this tick = ${args.waveEvents.length}`, '#facc15');
  line(`Improved this tick = ${args.waveEvents.filter(e => e.improved).length}`, '#fde68a');
  y0 += 8;
  line(`Settled / known finite distances:`, '#f1f5f9');
  for (let v = 0; v < graph.numNodes; v++) {
    const dStr = isFinite(args.distanceNow[v]) ? args.distanceNow[v].toFixed(2) : '∞';
    line(`  ${graph.nodeNames?.[v] ?? v}: ${dStr}`, distanceColor(args.distanceNow[v], maxFinite));
  }

  return {
    shapes,
    caption: `iter ${args.iteration}  waves=${args.waveEvents.length}  improved=${args.waveEvents.filter(e => e.improved).length}`,
  };
}

export function buildShortestPathCharts(
  ticks: number[], minFiniteDistance: number[], maxFiniteDistance: number[],
): ChartSpec[] {
  return [
    {x: META_X, y: META_Y + META_H + 10, w: META_W, h: 100,
     title: 'min / max finite distance per tick',
     series: [
       {label: 'min', color: '#22d3ee', t: ticks, y: minFiniteDistance},
       {label: 'max', color: '#fde68a', t: ticks, y: maxFiniteDistance},
     ]},
  ];
}
