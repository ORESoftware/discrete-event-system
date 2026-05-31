'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/newsvendor-scene.rs   (module des::animation::scenes::newsvendor_scene)
// 1:1 file move. Builds frames + charts for the newsvendor inventory animation.
//
// Declarations → Rust:
//   const STAGE_W/H              -> `pub const`
//   const COLORS (palette object) -> a struct/module of `&str` consts
//   interface NewsvendorFrameData -> struct
//   function buildNewsvendorFrame / buildNewsvendorChart -> pub fns
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `COLORS` object literal -> `const &str`s.
//   - all coords/values are `number` -> `f64`.
// =============================================================================

// =============================================================================
// Newsvendor / inventory animation scene.
//
// Layout (1100×680):
//   ┌─────────────────────────────────────┬────────────────────┐
//   │ left:  inventory + demand bar pair  │ right: metrics     │
//   │                                      │                    │
//   ├─────────────────────────────────────┴────────────────────┤
//   │ chart: profit / inventory over time                       │
//   └───────────────────────────────────────────────────────────┘
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';

export const STAGE_W = 1100;
export const STAGE_H = 680;

const COLORS = {
  inventory: '#3b82f6',
  ordered:   '#10b981',
  sold:      '#22c55e',
  leftover:  '#f59e0b',
  lost:      '#ef4444',
  demand:    '#a855f7',
  profitPos: '#16a34a',
  profitNeg: '#dc2626',
};

export interface NewsvendorFrameData {
  day: number;
  /** Starting inventory before order. */
  startInv: number;
  /** Quantity ordered this period. */
  ordered: number;
  /** Realised demand. */
  demand: number;
  /** Units sold (= min(start+order, demand)). */
  sold: number;
  /** Leftover at end of period. */
  leftover: number;
  /** Unmet demand. */
  lost: number;
  /** Profit / reward this period. */
  profit: number;
  /** Cumulative profit. */
  cumProfit: number;
  /** Maximum quantity for axis scaling. */
  qScale: number;
  /** Optional policy label (e.g., "(s,S) = (14, 47)"). */
  policy?: string;
}

export function buildNewsvendorFrame(d: NewsvendorFrameData): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const left = 40, top = 40, panelW = 700, panelH = 360;
  const margin = 40;
  const innerW = panelW - margin * 2;

  // Left panel: stacked bars for the day's inventory flow.
  shapes.push({kind: 'rect', x: left, y: top, w: panelW, h: panelH,
               fill: '#fafafa', stroke: '#ccc', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: left + 16, y: top + 24, fontSize: 16,
               fill: '#222', fontWeight: 'bold', text: `Day ${d.day}`});
  if (d.policy) {
    shapes.push({kind: 'text', x: left + panelW - 16, y: top + 24, anchor: 'end',
                 fontSize: 12, fill: '#666', text: d.policy});
  }

  // Bar geometry: four stacked bars at heights proportional to qScale.
  const barTop = top + 60;
  const barH = panelH - 120;
  const barW = (innerW - 60) / 4;
  const xCenters = [0, 1, 2, 3].map(i => left + margin + barW / 2 + i * (barW + 20));
  const labels = ['start inv', 'ordered', 'sold', 'leftover'];
  const values = [d.startInv, d.ordered, d.sold, d.leftover];
  const colors = [COLORS.inventory, COLORS.ordered, COLORS.sold, COLORS.leftover];

  for (let i = 0; i < 4; i++) {
    const v = values[i];
    const h = (v / Math.max(1, d.qScale)) * barH;
    shapes.push({kind: 'rect', x: xCenters[i] - barW / 2, y: barTop + barH - h,
                 w: barW, h, fill: colors[i], stroke: '#333', strokeWidth: 0.6, rx: 2});
    shapes.push({kind: 'text', x: xCenters[i], y: barTop + barH - h - 6,
                 anchor: 'middle', fontSize: 12, fontWeight: 'bold',
                 fill: '#222', text: String(v)});
    shapes.push({kind: 'text', x: xCenters[i], y: barTop + barH + 18,
                 anchor: 'middle', fontSize: 11, fill: '#555', text: labels[i]});
  }
  shapes.push({kind: 'line', x1: left + margin - 5, y1: barTop + barH,
               x2: left + panelW - margin + 5, y2: barTop + barH,
               stroke: '#999', strokeWidth: 1});

  // Demand line over the whole bar area.
  const demH = (d.demand / Math.max(1, d.qScale)) * barH;
  shapes.push({kind: 'line', x1: left + margin - 5, y1: barTop + barH - demH,
               x2: left + panelW - margin + 5, y2: barTop + barH - demH,
               stroke: COLORS.demand, strokeWidth: 1.5});
  shapes.push({kind: 'text', x: left + panelW - margin + 8, y: barTop + barH - demH + 4,
               fontSize: 11, fill: COLORS.demand, text: `D=${d.demand}`});

  // Right panel: metrics.
  const rx = left + panelW + 24;
  const ry = top;
  const rw = STAGE_W - rx - 40;
  const rh = panelH;
  shapes.push({kind: 'rect', x: rx, y: ry, w: rw, h: rh,
               fill: '#fff', stroke: '#ccc', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: rx + 14, y: ry + 24, fontSize: 14,
               fill: '#222', fontWeight: 'bold', text: 'Metrics'});

  type Row = {label: string; val: string; color?: string} | {separator: true};
  const rows: Row[] = [
    {label: 'demand',   val: String(d.demand),   color: COLORS.demand},
    {label: 'sold',     val: String(d.sold),     color: COLORS.sold},
    {label: 'leftover', val: String(d.leftover), color: COLORS.leftover},
    {label: 'lost',     val: String(d.lost),     color: COLORS.lost},
    {separator: true},
    {label: 'profit',     val: d.profit.toFixed(2),    color: d.profit    >= 0 ? COLORS.profitPos : COLORS.profitNeg},
    {label: 'cumulative', val: d.cumProfit.toFixed(2), color: d.cumProfit >= 0 ? COLORS.profitPos : COLORS.profitNeg},
  ];
  let yy = ry + 56;
  for (const row of rows) {
    if ('separator' in row) { yy += 12; continue; }
    shapes.push({kind: 'text', x: rx + 14, y: yy, fontSize: 12, fill: '#555', text: row.label});
    shapes.push({kind: 'text', x: rx + rw - 14, y: yy, anchor: 'end',
                 fontSize: 13, fill: row.color ?? '#222', fontWeight: 'bold', text: row.val});
    yy += 24;
  }

  return {shapes, caption: `day=${d.day}  start=${d.startInv}  order=${d.ordered}  ` +
                           `D=${d.demand}  sold=${d.sold}  leftover=${d.leftover}  lost=${d.lost}  ` +
                           `profit=${d.profit.toFixed(2)}  cum=${d.cumProfit.toFixed(2)}`};
}

export function buildNewsvendorChart(
  trace: {t: number[]; inv: number[]; profit: number[]; cumProfit: number[]},
): ChartSpec[] {
  const yMaxA = Math.max(1, Math.max(...trace.inv));
  const yMaxB = Math.max(1, Math.max(...trace.cumProfit));
  const yMinB = Math.min(0, Math.min(...trace.cumProfit));
  return [
    {x: 40, y: 420, w: 510, h: 230,
     title: 'inventory & per-period profit',
     yMin: -yMaxA, yMax: yMaxA,
     series: [
       {label: 'inv (start)', color: COLORS.inventory, t: trace.t, y: trace.inv},
       {label: 'profit',      color: COLORS.profitPos, t: trace.t, y: trace.profit},
     ],
    },
    {x: 570, y: 420, w: 510, h: 230,
     title: 'cumulative profit',
     yMin: yMinB, yMax: yMaxB,
     series: [
       {label: 'cumulative', color: '#2563eb', t: trace.t, y: trace.cumProfit},
     ],
    },
  ];
}
