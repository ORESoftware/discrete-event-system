'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/calculus_scene.rs
// - Keep buildField1DFrame/buildField1DChart/buildPoissonFrame as module helpers returning Frame/ChartSpec serde data.
// - Numeric arrays should become Vec<f64> or typed matrix/grid structs; choose a matrix crate only if later callers need it.
// - valueToColor and projection helpers remain private pure functions.
// - If a PDE field renderer becomes DES graph-visible, wrap it in a PureTransform struct with transform(field_state) -> Frame fragment.

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/calculus-scene.rs   (module des::animation::scenes::calculus_scene)
// 1:1 file move. Builds frames + charts for the calculus (1-D field / Poisson) animation.
//
// Declarations → Rust:
//   const STAGE_W/H, STRIP_*/METRIC_*/CHART_* consts, POISSON_W/H -> `pub const`/`const` (f64)
//   function valueToColor                  -> fn -> String
//   function buildField1DFrame / buildField1DChart / buildPoissonFrame -> pub fns
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `valueToColor(v, vMax)` builds `rgb(..)` -> `format!`.
//   - all coords/values are `number` -> `f64`.
// =============================================================================

// =============================================================================
// Field-evolution scene: render a 1-D PDE field u(x, t) as a coloured
// strip (each cell = one station, colour encodes value, vertical bars
// show value height) plus a time-series chart of the peak/center value.
// Also a 2-D Poisson scene that draws the converged solution as a
// false-colour image.
//
// Layout 1-D (1000×680):
//   ┌─────────────────────────────────────────────┬──────────────────┐
//   │  Field strip: N stations as vertical bars,  │ summary metrics  │
//   │  one per spatial cell. Height = u_i. Colour │  scheme, t, tick │
//   │  = signed magnitude (red ↔ blue).           │  peak, mean      │
//   └─────────────────────────────────────────────┴──────────────────┘
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ Charts: peak(t), mean energy(t), L2 norm                        │
//   └─────────────────────────────────────────────────────────────────┘
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';

export const STAGE_W = 1000;
export const STAGE_H = 720;
const STRIP_X = 30;
const STRIP_Y = 60;
const STRIP_W = 720;
const STRIP_H = 360;
const METRIC_X = 770;
const METRIC_Y = 60;
const METRIC_W = 200;
const METRIC_H = 360;
const CHART_X = 30;
const CHART_Y = 460;
const CHART_W = 940;
const CHART_H = 240;

/** Map a value v ∈ [-vMax, vMax] to a hex colour (blue → white → red). */
function valueToColor(v: number, vMax: number): string {
  const t = Math.max(-1, Math.min(1, v / Math.max(1e-12, vMax)));
  if (t >= 0) {
    const r = Math.round(255), g = Math.round(255 * (1 - t)), b = Math.round(255 * (1 - t));
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  } else {
    const r = Math.round(255 * (1 + t)), g = Math.round(255 * (1 + t)), b = 255;
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  }
}

export function buildField1DFrame(
  t: number, tick: number,
  values: ArrayLike<number>,
  xs: ArrayLike<number>,
  vMax: number,
  scheme: string,
  family: string,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const N = values.length;

  // Border + title.
  shapes.push({kind: 'rect', x: STRIP_X - 6, y: STRIP_Y - 6,
               w: STRIP_W + 12, h: STRIP_H + 12,
               fill: '#fff', stroke: '#bbb', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: STRIP_X, y: STRIP_Y - 14,
               text: `Field u(x, t),  family=${family},  scheme=${scheme},  N=${N} stations`,
               fontSize: 13, fill: '#333', fontWeight: 'bold'});

  // Zero line.
  const yMid = STRIP_Y + STRIP_H / 2;
  shapes.push({kind: 'line', x1: STRIP_X, y1: yMid, x2: STRIP_X + STRIP_W, y2: yMid,
               stroke: '#bbb', strokeWidth: 0.6});

  // Bars: each station is a vertical bar; colour by signed magnitude.
  const cellW = STRIP_W / N;
  let peak = 0, sumSq = 0;
  for (let i = 0; i < N; i++) {
    const v = values[i];
    if (Math.abs(v) > Math.abs(peak)) peak = v;
    sumSq += v * v;
    const h = (Math.abs(v) / vMax) * (STRIP_H / 2 - 4);
    const x = STRIP_X + i * cellW + cellW * 0.05;
    const w = cellW * 0.9;
    const yTop = v >= 0 ? yMid - h : yMid;
    shapes.push({kind: 'rect', x, y: yTop, w, h: Math.max(0.1, h),
                 fill: valueToColor(v, vMax)});
  }

  // x-axis ticks (5).
  for (let k = 0; k <= 4; k++) {
    const xx = STRIP_X + (STRIP_W * k) / 4;
    const xv = xs[Math.min(N - 1, Math.round((N - 1) * k / 4))];
    shapes.push({kind: 'text', x: xx, y: STRIP_Y + STRIP_H + 14,
                 text: xv.toFixed(2), fontSize: 10, fill: '#666', anchor: 'middle'});
  }

  // Metrics panel.
  shapes.push({kind: 'rect', x: METRIC_X, y: METRIC_Y, w: METRIC_W, h: METRIC_H,
               fill: '#fafafa', stroke: '#ddd', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: METRIC_X + 12, y: METRIC_Y + 22,
               text: `t = ${t.toFixed(4)}`, fontSize: 14, fill: '#222', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: METRIC_X + 12, y: METRIC_Y + 40,
               text: `tick ${tick}`, fontSize: 11, fill: '#666'});
  const stats: Array<[string, string]> = [
    ['peak |u|',     Math.abs(peak).toFixed(4)],
    ['L2 norm',      Math.sqrt(sumSq / N).toFixed(4)],
    ['scheme',       scheme],
    ['family',       family],
  ];
  for (let i = 0; i < stats.length; i++) {
    const y = METRIC_Y + 80 + i * 22;
    shapes.push({kind: 'text', x: METRIC_X + 12, y, text: stats[i][0],
                 fontSize: 11, fill: '#666'});
    shapes.push({kind: 'text', x: METRIC_X + METRIC_W - 12, y, text: stats[i][1],
                 fontSize: 11, fill: '#222', anchor: 'end', fontWeight: 'bold'});
  }
  return {shapes};
}

export function buildField1DChart(
  trace: {t: number[]; values: Float64Array[]},
): ChartSpec {
  // Track peak amplitude over time as the headline metric.
  const series: number[] = [];
  for (const v of trace.values) {
    let m = 0;
    for (let i = 0; i < v.length; i++) if (Math.abs(v[i]) > m) m = Math.abs(v[i]);
    series.push(m);
  }
  return {
    x: CHART_X, y: CHART_Y, w: CHART_W, h: CHART_H,
    title: 'peak |u(x, t)| over time',
    yLabel: 'peak |u|',
    series: [
      {label: 'peak', color: '#ef4444', t: trace.t.slice(), y: series},
    ],
  };
}

// -----------------------------------------------------------------------------
// 2-D Poisson scene: a single static frame showing the converged u(x, y).
// -----------------------------------------------------------------------------
export const POISSON_W = 720;
export const POISSON_H = 720;

export function buildPoissonFrame(
  u: ArrayLike<number>, Nx: number, Ny: number, vMax: number,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const PAD = 40;
  const W = POISSON_W - 2 * PAD;
  const H = POISSON_H - 2 * PAD;
  const cellW = W / Nx;
  const cellH = H / Ny;
  shapes.push({kind: 'rect', x: PAD - 4, y: PAD - 4, w: W + 8, h: H + 8,
               fill: '#fff', stroke: '#bbb', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: PAD, y: PAD - 12,
               text: `2-D Poisson  ∇²u = -ρ   grid ${Nx}×${Ny}`,
               fontSize: 13, fill: '#333', fontWeight: 'bold'});
  for (let j = 0; j < Ny; j++) {
    for (let i = 0; i < Nx; i++) {
      const v = u[j * Nx + i];
      shapes.push({kind: 'rect',
                   x: PAD + i * cellW, y: PAD + (Ny - 1 - j) * cellH,
                   w: cellW + 0.5, h: cellH + 0.5,
                   fill: valueToColor(v, vMax)});
    }
  }
  return {shapes};
}
