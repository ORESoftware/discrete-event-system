'use strict';

// =============================================================================
// Contact-SEIR scene builder. Each person is a small dot in a 2-D
// phyllotaxis grid (golden-angle spiral); state is encoded in fill
// color. Optionally, transmission events from the most recent tick
// can be overlaid as faint lines from infector → infectee.
//
// Layout (1000×680):
//   ┌─────────────────────────────────────────┬──────────────────┐
//   │ 60×40 grid of N people (or sqrt-ish)    │ summary metrics  │
//   │ S blue, E orange, I red, R gray         │  S, E, I, R      │
//   │                                          │  attack, R0      │
//   └─────────────────────────────────────────┴──────────────────┘
//   ┌─────────────────────────────────────────────────────────────┐
//   │ Compartment time-series chart (animated cursor)             │
//   └─────────────────────────────────────────────────────────────┘
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';

export const STAGE_W = 1000;
export const STAGE_H = 720;
const GRID_X = 30;
const GRID_Y = 40;
const GRID_W = 720;
const GRID_H = 420;
const METRIC_X = 770;
const METRIC_Y = 40;
const METRIC_W = 200;
const METRIC_H = 420;
const CHART_X = 30;
const CHART_Y = 480;
const CHART_W = 940;
const CHART_H = 220;

const COLORS = {
  S: '#3b82f6',  // blue
  E: '#f59e0b',  // amber
  I: '#ef4444',  // red
  R: '#9ca3af',  // gray
};

export interface PersonView {
  id: number;
  state: 'S' | 'E' | 'I' | 'R';
  c: number;        // contact rate (controls dot radius)
}

/**
 * Compute fixed (x, y) positions for N people on a regular grid in the
 * given rectangle. Returns parallel arrays. Idempotent — call once and
 * reuse for every frame.
 */
export function layoutGrid(N: number): {x: number[]; y: number[]} {
  const cols = Math.ceil(Math.sqrt(N * GRID_W / GRID_H));
  const rows = Math.ceil(N / cols);
  const cellW = GRID_W / cols;
  const cellH = GRID_H / rows;
  const x = new Array<number>(N);
  const y = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    const c = i % cols;
    const r = Math.floor(i / cols);
    x[i] = GRID_X + (c + 0.5) * cellW;
    y[i] = GRID_Y + (r + 0.5) * cellH;
  }
  return {x, y};
}

export function buildContactFrame(
  t: number, tick: number,
  people: ReadonlyArray<PersonView>,
  pos: {x: number[]; y: number[]},
  meanC: number,
  totalContacts: number,
  totalTransmissions: number,
  kernel: string,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const N = people.length;

  // Grid border + title.
  shapes.push({kind: 'rect', x: GRID_X - 6, y: GRID_Y - 6,
               w: GRID_W + 12, h: GRID_H + 12,
               fill: '#fff', stroke: '#bbb', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: GRID_X, y: GRID_Y - 14,
               text: `Population (${N} people, kernel=${kernel})`,
               fontSize: 13, fill: '#333', fontWeight: 'bold'});

  // Dots: radius proportional to per-person contact rate, capped.
  const baseR = Math.min(GRID_W, GRID_H) / Math.sqrt(N) / 4;
  let nS = 0, nE = 0, nI = 0, nR = 0;
  for (let i = 0; i < N; i++) {
    const p = people[i];
    if      (p.state === 'S') nS++;
    else if (p.state === 'E') nE++;
    else if (p.state === 'I') nI++;
    else                       nR++;
    const r = baseR * Math.min(2.5, Math.max(0.5, Math.sqrt(p.c / Math.max(1e-9, meanC))));
    // No per-dot title — it would balloon the frames file by ~10×.
    shapes.push({kind: 'circle',
                 x: pos.x[i], y: pos.y[i], r,
                 fill: COLORS[p.state],
                 stroke: '#fff', strokeWidth: 0.3});
  }

  // Metrics panel.
  shapes.push({kind: 'rect', x: METRIC_X, y: METRIC_Y, w: METRIC_W, h: METRIC_H,
               fill: '#fafafa', stroke: '#ddd', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: METRIC_X + 12, y: METRIC_Y + 22,
               text: `t = ${t.toFixed(2)}`, fontSize: 14, fill: '#222', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: METRIC_X + 12, y: METRIC_Y + 40,
               text: `tick ${tick}`, fontSize: 11, fill: '#666'});

  const lines: Array<[string, string, string]> = [
    ['S', String(nS), COLORS.S],
    ['E', String(nE), COLORS.E],
    ['I', String(nI), COLORS.I],
    ['R', String(nR), COLORS.R],
  ];
  for (let i = 0; i < lines.length; i++) {
    const y = METRIC_Y + 70 + i * 26;
    shapes.push({kind: 'rect', x: METRIC_X + 12, y: y - 11, w: 14, h: 14,
                 fill: lines[i][2], rx: 3});
    shapes.push({kind: 'text', x: METRIC_X + 32, y, text: lines[i][0],
                 fontSize: 12, fill: '#333', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: METRIC_X + METRIC_W - 12, y, text: lines[i][1],
                 fontSize: 12, fill: '#222', anchor: 'end', fontWeight: 'bold'});
  }
  const yEnd = METRIC_Y + 70 + lines.length * 26 + 6;
  shapes.push({kind: 'line', x1: METRIC_X + 12, y1: yEnd,
               x2: METRIC_X + METRIC_W - 12, y2: yEnd,
               stroke: '#ddd', strokeWidth: 1});

  const stats: Array<[string, string]> = [
    ['attack rate', `${((1 - nS / N) * 100).toFixed(1)}%`],
    ['contacts',    String(totalContacts)],
    ['transmissions', String(totalTransmissions)],
    ['kernel',       kernel],
  ];
  for (let i = 0; i < stats.length; i++) {
    const y = yEnd + 18 + i * 22;
    shapes.push({kind: 'text', x: METRIC_X + 12, y, text: stats[i][0],
                 fontSize: 11, fill: '#666'});
    shapes.push({kind: 'text', x: METRIC_X + METRIC_W - 12, y, text: stats[i][1],
                 fontSize: 12, fill: '#222', anchor: 'end', fontWeight: 'bold'});
  }

  // Legend.
  const legY = GRID_Y + GRID_H + 28;
  let lx = GRID_X;
  for (const [state, color] of [['S = susceptible', COLORS.S], ['E = exposed', COLORS.E], ['I = infectious', COLORS.I], ['R = recovered', COLORS.R]] as const) {
    shapes.push({kind: 'rect', x: lx, y: legY - 10, w: 14, h: 14, fill: color, rx: 2});
    shapes.push({kind: 'text', x: lx + 20, y: legY, text: state, fontSize: 11, fill: '#555'});
    lx += 150;
  }
  shapes.push({kind: 'text', x: GRID_X + 700, y: legY,
               text: 'dot radius ∝ √(contact rate)', fontSize: 11, fill: '#888', anchor: 'end'});

  const caption = `tick=${tick}  t=${t.toFixed(2)}  S=${nS} E=${nE} I=${nI} R=${nR}` +
                  `   attack=${((1 - nS / N) * 100).toFixed(1)}%  ` +
                  `transmissions=${totalTransmissions}`;
  return {shapes, caption};
}

export function buildContactChart(
  trace: {t: number[]; S: number[]; E: number[]; I: number[]; R: number[]},
  N: number,
): ChartSpec {
  return {
    x: CHART_X, y: CHART_Y, w: CHART_W, h: CHART_H,
    title: 'Compartment populations over time',
    yMin: 0, yMax: N,
    series: [
      {label: 'S', color: COLORS.S, t: trace.t, y: trace.S},
      {label: 'E', color: COLORS.E, t: trace.t, y: trace.E},
      {label: 'I', color: COLORS.I, t: trace.t, y: trace.I},
      {label: 'R', color: COLORS.R, t: trace.t, y: trace.R},
    ],
  };
}
