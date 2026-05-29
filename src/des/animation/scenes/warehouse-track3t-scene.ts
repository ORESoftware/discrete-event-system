'use strict';

import {ChartSpec, Shape} from '../types';
import {
  StationDefinition,
  WarehouseLayout,
  WarehouseComparisonResult,
  WarehouseScenarioResult,
  WarehouseStepTrace,
} from '../../general/factory-floor-track3t';

export const WAREHOUSE_TRACK3T_STAGE_W = 1200;
export const WAREHOUSE_TRACK3T_STAGE_H = 720;

interface PanelGeom {
  x: number;
  y: number;
  w: number;
  h: number;
  plotX: number;
  plotY: number;
  plotW: number;
  plotH: number;
  maxX: number;
  maxY: number;
}

interface MotionFrame {
  row?: WarehouseStepTrace;
  rowIndex: number;
  phase: number;
}

export function buildWarehouseComparisonFrame(
  result: WarehouseComparisonResult,
  frameIndex: number,
  framesPerTraceStep = 4,
): {shapes: Shape[]; caption: string} {
  const shapes: Shape[] = [
    {kind: 'rect', x: 0, y: 0, w: WAREHOUSE_TRACK3T_STAGE_W, h: WAREHOUSE_TRACK3T_STAGE_H, fill: '#f8fafc'},
    {kind: 'text', x: 34, y: 34, text: 'Warehouse floor: smart-movable forklifts and pallet flow', fontSize: 21, fill: '#111827', fontWeight: 'bold'},
    {kind: 'text', x: 34, y: 58, text: 'Forklifts and pallets move in 2D space; rings show the POMDP belief over pallet location.', fontSize: 13, fill: '#4b5563'},
  ];

  const basePanel: PanelGeom = makePanel(34, 86, result.layout.stations);
  const trackPanel: PanelGeom = makePanel(626, 86, result.layout.stations);
  const baseMotion = motionAt(result.baseline.trace, frameIndex, framesPerTraceStep);
  const trackMotion = motionAt(result.track3t.trace, frameIndex, framesPerTraceStep);
  drawPanel(shapes, result.baseline, baseMotion, basePanel, '#b91c1c');
  drawPanel(shapes, result.track3t, trackMotion, trackPanel, '#047857');
  drawDeltaSummary(shapes, result);

  const caption = [
    `frame ${frameIndex + 1}`,
    `baseline t=${interpolatedTime(baseMotion).toFixed(1)} min`,
    `track3t t=${interpolatedTime(trackMotion).toFixed(1)} min`,
  ].join(' | ');
  return {shapes, caption};
}

export function warehouseComparisonFrameCount(
  result: WarehouseComparisonResult,
  framesPerTraceStep = 4,
): number {
  return Math.max(result.baseline.trace.length, result.track3t.trace.length) * Math.max(1, framesPerTraceStep);
}

export function warehouseComparisonFrameTime(
  result: WarehouseComparisonResult,
  frameIndex: number,
  framesPerTraceStep = 4,
): number {
  const base = interpolatedTime(motionAt(result.baseline.trace, frameIndex, framesPerTraceStep));
  const track = interpolatedTime(motionAt(result.track3t.trace, frameIndex, framesPerTraceStep));
  return Math.max(base, track);
}

export function buildWarehouseComparisonCharts(result: WarehouseComparisonResult): ChartSpec[] {
  return [
    {
      x: 34, y: 552, w: 544, h: 130,
      title: 'Cumulative delivered jobs',
      yLabel: 'jobs',
      yMin: 0,
      yMax: Math.max(result.baseline.metrics.completedJobs, result.track3t.metrics.completedJobs),
      series: [
        cumulativeSeries(result.baseline, 'baseline', '#b91c1c', r => r.cumulativeDelivered),
        cumulativeSeries(result.track3t, 'track3t', '#047857', r => r.cumulativeDelivered),
      ],
    },
    {
      x: 626, y: 552, w: 544, h: 130,
      title: 'Cumulative search misses and delivery errors',
      yLabel: 'count',
      yMin: 0,
      yMax: Math.max(
        1,
        ...result.baseline.trace.map(r => r.cumulativeSearchMisses + r.cumulativeErrors),
        ...result.track3t.trace.map(r => r.cumulativeSearchMisses + r.cumulativeErrors),
      ),
      series: [
        cumulativeSeries(result.baseline, 'baseline', '#b91c1c', r => r.cumulativeSearchMisses + r.cumulativeErrors),
        cumulativeSeries(result.track3t, 'track3t', '#047857', r => r.cumulativeSearchMisses + r.cumulativeErrors),
      ],
    },
  ];
}

function drawPanel(
  shapes: Shape[],
  scenarioResult: WarehouseScenarioResult,
  motion: MotionFrame,
  g: PanelGeom,
  accent: string,
): void {
  const row = motion.row;
  shapes.push(
    {kind: 'rect', x: g.x, y: g.y, w: g.w, h: g.h, fill: '#ffffff', stroke: '#d1d5db', strokeWidth: 1, rx: 6},
    {kind: 'text', x: g.x + 18, y: g.y + 28, text: scenarioResult.scenario.label, fontSize: 17, fill: '#111827', fontWeight: 'bold'},
    {kind: 'text', x: g.x + 18, y: g.y + 50, text: metricLine(scenarioResult), fontSize: 12, fill: '#4b5563'},
  );

  drawRoutes(shapes, scenarioResult.layout, g);
  const compact = scenarioResult.layout.stations.length > 12;
  for (let i = 0; i < scenarioResult.layout.stations.length; i++) {
    const station = scenarioResult.layout.stations[i];
    const p = stationPoint(station, g);
    const belief = row?.beliefByStation[i] ?? 0;
    if (belief > 0.01) {
      shapes.push({
        kind: 'circle',
        x: p.x,
        y: p.y,
        r: (compact ? 9 : 14) + (compact ? 22 : 30) * belief,
        fill: accent,
        opacity: 0.13 + 0.28 * Math.min(1, belief),
        title: `${station.label}: belief ${belief.toFixed(2)}`,
      });
    }
  }

  const destination = row?.destination;
  for (const station of scenarioResult.layout.stations) {
    const p = stationPoint(station, g);
    const isDestination = station.id === destination;
    const boxW = compact ? (station.kind === 'storage' ? 34 : 52) : 68;
    const boxH = compact ? (station.kind === 'storage' ? 24 : 30) : 34;
    shapes.push({
      kind: 'rect',
      x: p.x - boxW / 2,
      y: p.y - boxH / 2,
      w: boxW,
      h: boxH,
      rx: 5,
      fill: stationFill(station),
      stroke: isDestination ? accent : '#374151',
      strokeWidth: isDestination ? 3 : 1,
      title: `${station.label} (${station.kind})`,
    });
    shapes.push({
      kind: 'text',
      x: p.x,
      y: p.y + 4,
      text: station.label,
      fontSize: compact ? 9 : 10,
      fill: '#111827',
      anchor: 'middle',
    });
  }

  if (row) {
    const route = routePathPoints(scenarioResult.layout, row.forkliftBefore, row.forkliftAfter, g);
    drawMotionPath(shapes, route, accent);
    const forklift = pointOnPolyline(route, motion.phase);
    const pallet = palletPoint(scenarioResult.layout, row, motion.phase, forklift, g);
    drawPallet(shapes, pallet.x, pallet.y, row.carryingBefore || row.carryingAfter || row.event === 'pickup' || row.event === 'delivered');
    drawForklift(shapes, forklift.x, forklift.y, accent, row.carryingBefore || row.carryingAfter, directionAngle(route, motion.phase));
    shapes.push(
      {kind: 'text', x: g.x + 18, y: g.y + g.h - 58, text: `${row.jobId}: ${row.event} via ${row.actionTarget}`, fontSize: 12, fill: '#111827', fontWeight: 'bold'},
      {kind: 'text', x: g.x + 18, y: g.y + g.h - 38, text: `obs: ${row.observation}`, fontSize: 11, fill: '#4b5563'},
      {kind: 'text', x: g.x + 18, y: g.y + g.h - 18, text: `cycle ${row.cycleTimeSoFar.toFixed(1)} min | entropy ${row.beliefEntropy.toFixed(2)} | errors ${row.cumulativeErrors}`, fontSize: 11, fill: '#4b5563'},
    );
  }
}

function drawMotionPath(shapes: Shape[], points: Array<{x: number; y: number}>, accent: string): void {
  if (points.length < 2) return;
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  shapes.push({kind: 'path', d, stroke: accent, strokeWidth: 2, fill: 'none', opacity: 0.3});
}

function drawForklift(
  shapes: Shape[],
  x: number,
  y: number,
  accent: string,
  loaded: boolean,
  angle: number,
): void {
  const bodyW = 28;
  const bodyH = 16;
  shapes.push(
    {kind: 'rect', x: x - bodyW / 2, y: y - bodyH / 2 - 20, w: bodyW, h: bodyH, rx: 4, fill: accent, stroke: '#111827', strokeWidth: 1.3, title: 'smart-movable forklift'},
    {kind: 'rect', x: x - 4, y: y - bodyH / 2 - 29, w: 13, h: 10, rx: 3, fill: '#e0f2fe', stroke: '#0f172a', strokeWidth: 1},
    {kind: 'circle', x: x - 9, y: y - 10, r: 3, fill: '#111827'},
    {kind: 'circle', x: x + 9, y: y - 10, r: 3, fill: '#111827'},
  );
  const forkStartX = x + Math.cos(angle) * 12;
  const forkStartY = y - 20 + Math.sin(angle) * 12;
  shapes.push(
    {kind: 'line', x1: forkStartX, y1: forkStartY, x2: forkStartX + Math.cos(angle) * 18, y2: forkStartY + Math.sin(angle) * 18, stroke: '#111827', strokeWidth: 2},
    {kind: 'line', x1: forkStartX, y1: forkStartY + 5, x2: forkStartX + Math.cos(angle) * 18, y2: forkStartY + 5 + Math.sin(angle) * 18, stroke: '#111827', strokeWidth: 2},
    {kind: 'text', x: x, y: y - 17, text: 'F', fontSize: 10, fill: '#ffffff', anchor: 'middle', fontWeight: 'bold'},
  );
  if (loaded) {
    shapes.push({kind: 'text', x: x + 22, y: y - 24, text: 'loaded', fontSize: 10, fill: accent, fontWeight: 'bold'});
  }
}

function drawPallet(shapes: Shape[], x: number, y: number, active: boolean): void {
  shapes.push({
    kind: 'rect',
    x: x - 10,
    y: y - 7,
    w: 20,
    h: 14,
    rx: 2,
    fill: active ? '#f59e0b' : '#fbbf24',
    stroke: '#92400e',
    strokeWidth: 1,
    label: 'P',
    title: 'movable pallet',
  });
}

function palletPoint(
  layout: WarehouseLayout,
  row: WarehouseStepTrace,
  phase: number,
  forklift: {x: number; y: number},
  g: PanelGeom,
): {x: number; y: number} {
  const before = stationById(layout, row.palletBefore);
  const after = stationById(layout, row.palletAfter);
  const beforeP = before ? stationPoint(before, g) : forklift;
  const afterP = after ? stationPoint(after, g) : forklift;
  if (row.carryingBefore && row.carryingAfter) return {x: forklift.x + 18, y: forklift.y - 22};
  if (!row.carryingBefore && row.carryingAfter) {
    return phase < 0.78 ? {x: beforeP.x, y: beforeP.y + 24} : {x: forklift.x + 18, y: forklift.y - 22};
  }
  if (row.carryingBefore && !row.carryingAfter) {
    return phase < 0.86 ? {x: forklift.x + 18, y: forklift.y - 22} : {x: afterP.x, y: afterP.y + 24};
  }
  return {x: beforeP.x + (afterP.x - beforeP.x) * phase, y: beforeP.y + 24 + (afterP.y - beforeP.y) * phase};
}

function drawRoutes(shapes: Shape[], layout: WarehouseLayout, g: PanelGeom): void {
  const byId = new Map(layout.stations.map(s => [s.id, s]));
  const routePairs = layout.routeEdges ?? [];
  for (const [aId, bId] of routePairs) {
    const a = byId.get(aId);
    const b = byId.get(bId);
    if (!a || !b) continue;
    const pa = stationPoint(a, g);
    const pb = stationPoint(b, g);
    shapes.push({kind: 'line', x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: '#cbd5e1', strokeWidth: 4, opacity: 0.8});
  }
}

function drawDeltaSummary(shapes: Shape[], result: WarehouseComparisonResult): void {
  const x = 34;
  const y = 504;
  shapes.push(
    {kind: 'rect', x, y, w: 1136, h: 32, fill: '#111827', rx: 5},
    {kind: 'text', x: x + 16, y: y + 21, text: `Track3t lift: cycle ${result.deltas.meanCycleTimeReductionPct.toFixed(1)}% faster, throughput ${result.deltas.throughputLiftPct.toFixed(1)}% higher, search misses ${result.deltas.searchMissReductionPct.toFixed(1)}% lower, errors ${result.deltas.errorReductionPct.toFixed(1)}% lower`, fontSize: 13, fill: '#f9fafb', fontWeight: 'bold'},
  );
}

function makePanel(x: number, y: number, stations: StationDefinition[]): PanelGeom {
  const maxX = Math.max(...stations.map(s => s.x));
  const maxY = Math.max(...stations.map(s => s.y));
  return {x, y, w: 540, h: 398, plotX: x + 58, plotY: y + 82, plotW: 420, plotH: 230, maxX, maxY};
}

function stationPoint(station: StationDefinition, g: PanelGeom): {x: number; y: number} {
  const x = g.plotX + station.x / Math.max(1, g.maxX) * g.plotW;
  const y = g.plotY + station.y / Math.max(1, g.maxY) * g.plotH;
  return {x, y};
}

function stationFill(station: StationDefinition): string {
  if (station.kind === 'source') return '#dbeafe';
  if (station.kind === 'storage') return '#fef3c7';
  if (station.kind === 'sink') return '#dcfce7';
  return '#e5e7eb';
}

function motionAt(trace: WarehouseStepTrace[], frameIndex: number, framesPerTraceStep: number): MotionFrame {
  if (trace.length === 0) return {rowIndex: -1, phase: 1};
  const subframes = Math.max(1, Math.floor(framesPerTraceStep));
  const rowIndex = Math.min(trace.length - 1, Math.floor(frameIndex / subframes));
  const subIndex = Math.min(subframes - 1, frameIndex % subframes);
  const phase = subframes <= 1 ? 1 : subIndex / (subframes - 1);
  return {row: trace[rowIndex], rowIndex, phase};
}

function interpolatedTime(motion: MotionFrame): number {
  if (!motion.row) return 0;
  return motion.row.timeStart + (motion.row.timeEnd - motion.row.timeStart) * motion.phase;
}

function stationById(layout: WarehouseLayout, id: string): StationDefinition | undefined {
  return layout.stations.find(s => s.id === id);
}

function routePathPoints(
  layout: WarehouseLayout,
  fromId: string,
  toId: string,
  g: PanelGeom,
): Array<{x: number; y: number}> {
  const from = stationById(layout, fromId);
  const to = stationById(layout, toId);
  if (!from || !to) return [];
  if (fromId === toId) {
    const p = stationPoint(from, g);
    return [p, p];
  }
  const corridor = rowCorridorPath(layout, from, to, g);
  if (corridor) return corridor;
  const byId = new Map(layout.stations.map(s => [s.id, s]));
  const adjacency = new Map<string, string[]>();
  const add = (a: string, b: string) => {
    if (!adjacency.has(a)) adjacency.set(a, []);
    adjacency.get(a)!.push(b);
  };
  for (const [a, b] of layout.routeEdges ?? []) {
    add(a, b);
    add(b, a);
  }
  const prev = new Map<string, string | undefined>();
  const q = [fromId];
  prev.set(fromId, undefined);
  for (let i = 0; i < q.length; i++) {
    const cur = q[i];
    if (cur === toId) break;
    for (const next of adjacency.get(cur) ?? []) {
      if (prev.has(next)) continue;
      prev.set(next, cur);
      q.push(next);
    }
  }
  if (!prev.has(toId)) return [stationPoint(from, g), stationPoint(to, g)];
  const ids: string[] = [];
  for (let cur: string | undefined = toId; cur !== undefined; cur = prev.get(cur)) ids.push(cur);
  ids.reverse();
  return ids
    .map(id => byId.get(id))
    .filter((s): s is StationDefinition => !!s)
    .map(s => stationPoint(s, g));
}

function rowCorridorPath(
  layout: WarehouseLayout,
  from: StationDefinition,
  to: StationDefinition,
  g: PanelGeom,
): Array<{x: number; y: number}> | undefined {
  const rows = reserveRows(layout);
  const main = stationById(layout, 'aisle-main');
  const staging = stationById(layout, 'staging');
  const receiving = stationById(layout, 'receiving');
  if (!main || !staging || rows.length === 0) return undefined;

  const fromReserve = reserveRowKey(from.id);
  const toReserve = reserveRowKey(to.id);
  const usesRightSide = from.kind === 'sink' || to.kind === 'sink' || from.id === main.id || to.id === main.id;
  const usesReserve = fromReserve !== undefined || toReserve !== undefined;
  if (!usesRightSide && !usesReserve) return undefined;

  const row = chooseCorridorRow(rows, from, to);
  const points: StationDefinition[] = [];
  const push = (s: StationDefinition | undefined) => {
    if (!s) return;
    if (points.length === 0 || points[points.length - 1].id !== s.id) points.push(s);
  };

  if (from.kind === 'sink') {
    push(from);
    push(main);
    if (to.kind === 'sink') {
      push(to);
    } else if (to.id === main.id) {
      push(to);
    } else {
      appendRowFromMain(points, row.stations, to);
      if (to.id === staging.id) push(staging);
      else if (to.id === receiving?.id) { push(staging); push(receiving); }
      else if (toReserve === undefined && to.id !== staging.id && to.id !== receiving?.id) push(to);
    }
    return points.map(s => stationPoint(s, g));
  }

  push(from);
  if (from.id === receiving?.id) push(staging);
  if (to.kind === 'sink' || to.id === main.id) {
    appendRowToMain(points, row.stations, from);
    push(main);
    if (to.kind === 'sink') push(to);
    return points.map(s => stationPoint(s, g));
  }

  if (from.id === main.id) {
    appendRowFromMain(points, row.stations, to);
    if (to.id === receiving?.id) { push(staging); push(receiving); }
    else if (to.id === staging.id) push(staging);
    return points.map(s => stationPoint(s, g));
  }

  if (toReserve !== undefined) {
    const sameRow = fromReserve === toReserve;
    if (sameRow) push(to);
    else {
      appendRowToMain(points, row.stations, from);
      push(main);
      const toRow = rows.find(r => r.key === toReserve) ?? row;
      appendRowFromMain(points, toRow.stations, to);
    }
    return points.map(s => stationPoint(s, g));
  }

  return undefined;
}

interface ReserveRow {
  key: string;
  y: number;
  stations: StationDefinition[];
}

function reserveRows(layout: WarehouseLayout): ReserveRow[] {
  const buckets = new Map<string, StationDefinition[]>();
  for (const station of layout.stations) {
    const key = reserveRowKey(station.id);
    if (!key) continue;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(station);
  }
  return Array.from(buckets.entries()).map(([key, stations]) => {
    stations.sort((a, b) => a.x - b.x);
    return {key, stations, y: stations.reduce((sum, s) => sum + s.y, 0) / stations.length};
  }).sort((a, b) => a.y - b.y);
}

function reserveRowKey(id: string): string | undefined {
  const m = id.match(/^reserve-([a-z])\d+$/);
  return m ? m[1] : undefined;
}

function chooseCorridorRow(rows: ReserveRow[], from: StationDefinition, to: StationDefinition): ReserveRow {
  const fromKey = reserveRowKey(from.id);
  const toKey = reserveRowKey(to.id);
  const exact = fromKey
    ? rows.find(r => r.key === fromKey)
    : toKey
      ? rows.find(r => r.key === toKey)
      : undefined;
  if (exact) return exact;
  const y = from.kind === 'sink' ? from.y
    : to.kind === 'sink' ? to.y
      : (from.y + to.y) / 2;
  return rows.reduce((best, row) => Math.abs(row.y - y) < Math.abs(best.y - y) ? row : best, rows[0]);
}

function appendRowToMain(points: StationDefinition[], row: StationDefinition[], from: StationDefinition): void {
  const fromIdx = row.findIndex(s => s.id === from.id);
  const start = fromIdx >= 0 ? fromIdx + 1 : 0;
  for (let i = start; i < row.length; i++) pushPoint(points, row[i]);
}

function appendRowFromMain(points: StationDefinition[], row: StationDefinition[], to: StationDefinition): void {
  const toIdx = row.findIndex(s => s.id === to.id);
  const end = toIdx >= 0 ? toIdx : 0;
  for (let i = row.length - 1; i >= end; i--) pushPoint(points, row[i]);
}

function pushPoint(points: StationDefinition[], station: StationDefinition): void {
  if (points.length === 0 || points[points.length - 1].id !== station.id) points.push(station);
}

function pointOnPolyline(points: Array<{x: number; y: number}>, phase: number): {x: number; y: number} {
  if (points.length === 0) return {x: 0, y: 0};
  if (points.length === 1) return points[0];
  const lengths: number[] = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const len = Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    lengths.push(len);
    total += len;
  }
  if (total <= 0) return points[points.length - 1];
  let target = Math.max(0, Math.min(1, phase)) * total;
  for (let i = 1; i < points.length; i++) {
    const len = lengths[i - 1];
    if (target <= len || i === points.length - 1) {
      const local = len <= 0 ? 1 : target / len;
      return {
        x: points[i - 1].x + (points[i].x - points[i - 1].x) * local,
        y: points[i - 1].y + (points[i].y - points[i - 1].y) * local,
      };
    }
    target -= len;
  }
  return points[points.length - 1];
}

function directionAngle(points: Array<{x: number; y: number}>, phase: number): number {
  if (points.length < 2) return 0;
  const p = pointOnPolyline(points, Math.max(0, phase - 0.03));
  const q = pointOnPolyline(points, Math.min(1, phase + 0.03));
  return Math.atan2(q.y - p.y, q.x - p.x);
}

function metricLine(r: WarehouseScenarioResult): string {
  const m = r.metrics;
  return [
    `${m.completedJobs}/${m.jobsCreated} done`,
    `${m.meanCycleTime.toFixed(1)} min/job`,
    `${m.throughputPerHour.toFixed(1)} jobs/hr`,
    `${(m.shippingErrorRate * 100).toFixed(1)}% err`,
  ].join(' | ');
}

function cumulativeSeries(
  result: WarehouseScenarioResult,
  label: string,
  color: string,
  value: (row: WarehouseStepTrace) => number,
): {label: string; color: string; t: number[]; y: number[]} {
  const t: number[] = [];
  const y: number[] = [];
  for (const row of result.trace) {
    t.push(row.timeEnd);
    y.push(value(row));
  }
  return {label, color, t, y};
}
