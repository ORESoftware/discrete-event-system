'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/elevator_scene.rs
// - Keep buildElevatorFrame/buildElevatorChart as module helpers over a typed Building struct imported from main_elevator.
// - Direction/state string unions should become enums; color selection becomes match expressions.
// - Shape/ChartSpec outputs are serde structs/enums and private drawing helpers should push into Vec<Shape>.
// - If the elevator scene participates in the DES graph, wrap Building -> Frame generation as a PureTransform implementor.

// =============================================================================
// Elevator scene builder.
//
// Layout (900×640):
//   ┌──────────────────────────────────────────────────────────┐
//   │   F1  F2  F3  F4   ↕                     metrics │       │
//   │  ┌── ──────── ──────── ──────── ───────┐ ──────  │       │
//   │  │ E0:◼◼◼◼  E1:◼◻◻◻  E2:◻◻◻◻         │ served  │       │
//   │  │  ↑                                  │ waiting │       │
//   │  │ F4 [⬣⬣]                              │ moving  │       │
//   │  │ F3 [⬣]   E0(2) E1(0)                │ ...     │       │
//   │  │ F2                                   │         │       │
//   │  │ F1 [⬣⬣⬣]                             │         │       │
//   │  └──────────────────────────────────────┘         │       │
//   └──────────────────────────────────────────────────────────┘
//
// Coordinates: floors are horizontal lanes. Elevators are tall thin
// rectangles whose vertical position represents currentFloor (which is
// continuous). Passenger count is shown in the rect; passengers' target
// floors are listed underneath. Up- and down-queue people are drawn as
// small circles beside each floor.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {Building} from '../../main-elevator';

export const STAGE_W = 1000;
export const STAGE_H = 640;

const VIEW_X = 60;
const VIEW_Y = 40;
const VIEW_W = 660;
const VIEW_H = 460;
const METRIC_X = 740;
const METRIC_Y = 40;
const METRIC_W = 220;
const METRIC_H = 460;

const COLOR_IDLE  = '#9ca3af';
const COLOR_UP    = '#16a34a';
const COLOR_DOWN  = '#2563eb';
const COLOR_SERVE = '#f59e0b';

function dirColor(dir: 'idle' | 'up' | 'down', state: 'IDLE' | 'MOVING' | 'SERVING'): string {
  if (state === 'SERVING') return COLOR_SERVE;
  if (dir === 'up')   return COLOR_UP;
  if (dir === 'down') return COLOR_DOWN;
  return COLOR_IDLE;
}

export function buildElevatorFrame(t: number, tick: number, b: Building): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const cfg = b.config;
  const nF = cfg.nFloors;
  const nE = b.elevators.length;

  // Frame border.
  shapes.push({kind: 'rect', x: VIEW_X, y: VIEW_Y, w: VIEW_W, h: VIEW_H,
               fill: '#fff', stroke: '#bbb', strokeWidth: 1, rx: 4});

  // Floor lanes. Floor 1 at the bottom, floor nF at the top.
  const laneH = VIEW_H / nF;
  for (let f = 1; f <= nF; f++) {
    const y = VIEW_Y + (nF - f) * laneH;
    shapes.push({kind: 'line', x1: VIEW_X, y1: y, x2: VIEW_X + VIEW_W, y2: y,
                 stroke: '#e5e7eb', strokeWidth: 1});
    shapes.push({kind: 'text', x: VIEW_X + 8, y: y + laneH / 2 + 4,
                 text: `F${f}`, fontSize: 12, fill: '#666', anchor: 'start'});
  }
  // Top frame line.
  shapes.push({kind: 'line', x1: VIEW_X, y1: VIEW_Y, x2: VIEW_X + VIEW_W, y2: VIEW_Y,
               stroke: '#e5e7eb', strokeWidth: 1});

  // Floor up/down queues. Drawn as horizontal rows of small circles next to
  // floor labels, just to the right of the F label.
  const queueX = VIEW_X + 50;
  const queueDot = 5;
  for (let f = 1; f <= nF; f++) {
    const y = VIEW_Y + (nF - f) * laneH + laneH / 2;
    const floor = b.floors[f - 1];
    let dx = 0;
    for (let i = 0; i < floor.upQueue.length; i++) {
      shapes.push({kind: 'circle', x: queueX + dx, y: y - 6, r: queueDot,
                   fill: COLOR_UP, stroke: '#0a3d22', strokeWidth: 0.5,
                   title: `up-bound to F${floor.upQueue[i].toFloor}`});
      dx += queueDot * 2 + 1;
      if (dx > 160) { dx = 0; break; }
    }
    dx = 0;
    for (let i = 0; i < floor.downQueue.length; i++) {
      shapes.push({kind: 'circle', x: queueX + dx, y: y + 6, r: queueDot,
                   fill: COLOR_DOWN, stroke: '#162e60', strokeWidth: 0.5,
                   title: `down-bound to F${floor.downQueue[i].toFloor}`});
      dx += queueDot * 2 + 1;
      if (dx > 160) { dx = 0; break; }
    }
  }

  // Elevators. Slot them across the right portion of the building view.
  const carX0 = VIEW_X + 250;
  const carSlotW = (VIEW_W - 250 - 16) / nE;
  const carW = Math.min(60, carSlotW * 0.8);
  const carH = laneH * 0.85;
  for (let k = 0; k < nE; k++) {
    const e = b.elevators[k];
    // Continuous vertical position from currentFloor [1..nF].
    const yCenter = VIEW_Y + VIEW_H - (e.currentFloor - 0.5) * laneH;
    const x = carX0 + k * carSlotW + (carSlotW - carW) / 2;
    const y = yCenter - carH / 2;
    const fill = dirColor((e as any).direction, (e as any).state);
    shapes.push({kind: 'rect', x, y, w: carW, h: carH, fill, rx: 3,
                 stroke: '#222', strokeWidth: 0.8,
                 title: `E${k}: state=${(e as any).state} dir=${(e as any).direction} ` +
                        `floor=${e.currentFloor.toFixed(2)} pax=${e.passengers.length}/${e.capacity}` +
                        ((e as any).targetFloor !== null ? ` target=F${(e as any).targetFloor}` : '')});
    // Passenger count (big number in the middle).
    shapes.push({kind: 'text', x: x + carW / 2, y: y + carH / 2 + 5,
                 text: String(e.passengers.length), fontSize: 14, fill: '#fff',
                 anchor: 'middle', fontWeight: 'bold'});
    // Capacity label below.
    shapes.push({kind: 'text', x: x + carW / 2, y: y + carH + 12,
                 text: `E${k}`, fontSize: 10, fill: '#444', anchor: 'middle'});
    // Target floor indicator: dashed line from car to target lane.
    if ((e as any).targetFloor !== null && (e as any).state === 'MOVING') {
      const tgtY = VIEW_Y + VIEW_H - ((e as any).targetFloor - 0.5) * laneH;
      shapes.push({kind: 'line', x1: x + carW / 2, y1: yCenter,
                   x2: x + carW / 2, y2: tgtY,
                   stroke: '#999', strokeWidth: 1, dasharray: '2,2'});
      shapes.push({kind: 'circle', x: x + carW / 2, y: tgtY, r: 3, fill: '#999'});
    }
    // Render passenger destinations as small ticks in the car.
    const dests = e.passengers.map(p => p.toFloor);
    for (let p = 0; p < dests.length; p++) {
      const dx = (p % 4) * 6 + 4;
      const dy = Math.floor(p / 4) * 4 + 4;
      shapes.push({kind: 'circle', x: x + dx, y: y + dy, r: 1.5, fill: '#fff'});
    }
  }

  // Metrics panel.
  shapes.push({kind: 'rect', x: METRIC_X, y: METRIC_Y, w: METRIC_W, h: METRIC_H,
               fill: '#fafafa', stroke: '#ddd', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: METRIC_X + 12, y: METRIC_Y + 22,
               text: `Tick ${tick}   t=${t.toFixed(2)}`, fontSize: 13, fill: '#222', fontWeight: 'bold'});

  const totalWaiting = b.floors.reduce((s, f) => s + f.upQueue.length + f.downQueue.length, 0);
  const totalInCar   = b.elevators.reduce((s, e) => s + e.passengers.length, 0);
  const totalServed  = b.sink.collected.filter(p => p.exitTime > 0).length;
  const lines: Array<[string, string]> = [
    ['waiting',     String(totalWaiting)],
    ['in elevator', String(totalInCar)],
    ['served',      String(totalServed)],
    ['mode',        b.config.dispatchMode ?? 'uncoordinated'],
    ['nFloors',     String(b.config.nFloors)],
    ['nElevators',  String(b.elevators.length)],
    ['capacity',    String(b.config.capacity)],
  ];
  for (let i = 0; i < lines.length; i++) {
    const y = METRIC_Y + 50 + i * 22;
    shapes.push({kind: 'text', x: METRIC_X + 12, y, text: lines[i][0],
                 fontSize: 11, fill: '#666'});
    shapes.push({kind: 'text', x: METRIC_X + METRIC_W - 12, y, text: lines[i][1],
                 fontSize: 12, fill: '#222', anchor: 'end', fontWeight: 'bold'});
  }
  // Per-elevator status block.
  const baseY = METRIC_Y + 50 + lines.length * 22 + 10;
  shapes.push({kind: 'text', x: METRIC_X + 12, y: baseY,
               text: 'Per elevator', fontSize: 12, fill: '#444', fontWeight: 'bold'});
  for (let k = 0; k < nE; k++) {
    const e = b.elevators[k];
    const y = baseY + 18 + k * 30;
    const sw = 12;
    const fill = dirColor((e as any).direction, (e as any).state);
    shapes.push({kind: 'rect', x: METRIC_X + 12, y: y - 9, w: sw, h: sw, fill, rx: 2});
    shapes.push({kind: 'text', x: METRIC_X + 32, y, text: `E${k}`,
                 fontSize: 11, fill: '#222', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: METRIC_X + 60, y,
                 text: `F${e.currentFloor.toFixed(1)}  ${e.passengers.length}/${e.capacity}`,
                 fontSize: 11, fill: '#444'});
    shapes.push({kind: 'text', x: METRIC_X + 12, y: y + 12,
                 text: `${(e as any).state.toLowerCase()} ${(e as any).direction}` +
                       ((e as any).targetFloor !== null ? ` →F${(e as any).targetFloor}` : ''),
                 fontSize: 10, fill: '#666'});
  }

  // Legend at bottom.
  const legY = VIEW_Y + VIEW_H + 28;
  const legendItems: Array<[string, string]> = [
    ['idle',          COLOR_IDLE],
    ['moving up',     COLOR_UP],
    ['moving down',   COLOR_DOWN],
    ['serving',       COLOR_SERVE],
  ];
  let lx = VIEW_X;
  for (const [label, color] of legendItems) {
    shapes.push({kind: 'rect', x: lx, y: legY - 10, w: 14, h: 14, fill: color, rx: 2});
    shapes.push({kind: 'text', x: lx + 20, y: legY, text: label,
                 fontSize: 11, fill: '#555'});
    lx += 110;
  }

  const caption =
    `tick=${tick}  t=${t.toFixed(2)}s  waiting=${totalWaiting}  ` +
    `in-car=${totalInCar}  served=${totalServed}`;
  return {shapes, caption};
}

/**
 * Build a global time-series chart of waiting count vs in-car count vs
 * served count. The recorder feeds this in via setCharts() at finish().
 */
export function buildElevatorChart(
  series: {t: number[]; waiting: number[]; inCar: number[]; served: number[]},
  panelY = 510, panelH = 110,
): ChartSpec {
  return {
    x: VIEW_X, y: panelY, w: VIEW_W, h: panelH,
    title: 'System occupancy over time',
    yMin: 0,
    series: [
      {label: 'waiting',     color: '#dc2626', t: series.t, y: series.waiting},
      {label: 'in elevator', color: '#2563eb', t: series.t, y: series.inCar},
      {label: 'served',      color: '#16a34a', t: series.t, y: series.served},
    ],
  };
}
