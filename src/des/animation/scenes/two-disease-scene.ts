'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/two_disease_scene.rs
// - CompartmentCounts should become a Rust struct or enum-indexed map; COLORS can be a match over compartment enum variants.
// - buildBars/buildFrame/buildCompartmentChart stay module helpers returning Vec<Shape>, Frame fragments, and ChartSpec.
// - Animation/Frame/Shape/ChartSpec are serde data from animation::types; optional captions become Option<String>.
// - If this renderer is wired into the DES graph, wrap snapshot-to-frame as a PureTransform implementor.

// =============================================================================
// Two-disease scene builder.
//
// Layout (900×640):
//   ┌───────────────────────────────────────────────────────────────┐
//   │ S    A    B    AB    R    D                                   │  6 vertical bars (top half)
//   │ █    █    █    █     █    █                                   │
//   │ █    █    █    █     █    █                                   │
//   │ ─────────────────────────────                                 │
//   │ Compartments over time (line chart, animated cursor)         │  chart (bottom half)
//   └───────────────────────────────────────────────────────────────┘
// =============================================================================

import {Animation, ChartSpec, Frame, Shape} from '../types';

export interface CompartmentCounts {
  S: number; A: number; B: number; AB: number; R: number; D: number;
}

const ORDER: Array<keyof CompartmentCounts> = ['S', 'A', 'B', 'AB', 'R', 'D'];
export const COLORS: Record<keyof CompartmentCounts, string> = {
  S:  '#3b82f6',  // blue   = susceptible
  A:  '#f59e0b',  // amber  = disease A
  B:  '#10b981',  // emerald = disease B
  AB: '#8b5cf6',  // violet = co-infected
  R:  '#6b7280',  // gray   = recovered
  D:  '#ef4444',  // red    = dead
};

export const STAGE_W = 900;
export const STAGE_H = 640;
const BAR_AREA_X = 60;
const BAR_AREA_Y = 60;
const BAR_AREA_W = STAGE_W - 120;
const BAR_AREA_H = 220;
const CHART_X = 60;
const CHART_Y = 320;
const CHART_W = STAGE_W - 120;
const CHART_H = 280;

export function buildBars(counts: CompartmentCounts, N: number): Shape[] {
  const shapes: Shape[] = [];
  shapes.push({kind: 'text', x: BAR_AREA_X, y: BAR_AREA_Y - 14,
               text: 'Population by compartment', fontSize: 13, fill: '#333', fontWeight: 'bold'});

  const bw = BAR_AREA_W / 6 * 0.7;
  const gap = (BAR_AREA_W - bw * 6) / 7;

  for (let i = 0; i < ORDER.length; i++) {
    const k = ORDER[i];
    const v = counts[k];
    const h = (v / N) * BAR_AREA_H;
    const x = BAR_AREA_X + gap + i * (bw + gap);
    const y = BAR_AREA_Y + BAR_AREA_H - h;
    shapes.push({kind: 'rect', x, y, w: bw, h, fill: COLORS[k], rx: 3,
                 title: `${k} = ${v}`});
    shapes.push({kind: 'rect', x, y: BAR_AREA_Y, w: bw, h: BAR_AREA_H,
                 fill: 'none', stroke: '#ddd', strokeWidth: 1});
    shapes.push({kind: 'text', x: x + bw / 2, y: BAR_AREA_Y + BAR_AREA_H + 16,
                 text: k, fontSize: 14, fill: '#333', anchor: 'middle', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: x + bw / 2, y: y - 6,
                 text: String(v), fontSize: 11, fill: '#222', anchor: 'middle'});
  }
  return shapes;
}

/**
 * Build the entire frame (bars + axis labels). The line chart is global
 * to the animation and is drawn separately by the player.
 */
export function buildFrame(t: number, tick: number, counts: CompartmentCounts, N: number): Omit<Frame, 't' | 'tick'> {
  const shapes = buildBars(counts, N);
  const live = counts.S + counts.A + counts.B + counts.AB + counts.R;
  const dead = counts.D;
  const caption =
    `t=${t.toFixed(2)}   alive=${live}   dead=${dead}   ` +
    ORDER.map(k => `${k}=${counts[k]}`).join('  ');
  return {shapes, caption};
}

/**
 * Build the global compartment-over-time chart from a trace produced by
 * `runTwoDisease`. Used as the chart panel beneath the bars.
 */
export function buildCompartmentChart(
  trace: {t: number[]; S: number[]; A: number[]; B: number[]; AB: number[]; R: number[]; D: number[]},
  N: number,
): ChartSpec {
  const series = ORDER.map(k => ({
    label: k,
    color: COLORS[k],
    t:     trace.t,
    y:     (trace as any)[k] as number[],
  }));
  return {
    x: CHART_X, y: CHART_Y, w: CHART_W, h: CHART_H,
    title: 'Compartment populations over time',
    yMin: 0, yMax: N,
    series,
  };
}
