'use strict';

// =============================================================================
// Soccer pitch + bench animation scene.
//
// Layout (1000 × 640):
//   ┌──────────────────────────────────────────────────────────────────┐
//   │  scoreboard / period   │  pitch ⚽                       │ bench │
//   │  Us 2 - 1 Them   P3/4  │   ──────────────────────         │  P3   │
//   │                        │   |  A         G   |             │  P7   │
//   │  affinity history line │   |    B   F       |             │  P9   │
//   │                        │   |     E          |             │  P11  │
//   │                        │   |   D       C    |             │  P12  │
//   │                        │   |________________|             │       │
//   │                        │   on-field affinity bar          │       │
//   └──────────────────────────────────────────────────────────────────┘
//
// Player ⛹️ is rendered as a circle with their id as label. On a sub
// event, the bench/field labels in the next frame change, so the
// player-position remap is naturally animated by the frame-to-frame
// position changes (no extra interpolation logic needed — players
// teleport to their new locations between frames).
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {SoccerProblem} from '../../general/soccer-rotation';

export const STAGE_W = 1100;
export const STAGE_H = 640;

const PITCH_X = 320;
const PITCH_Y = 60;
const PITCH_W = 580;
const PITCH_H = 460;

const BENCH_X = 920;
const BENCH_Y = 60;
const BENCH_W = 160;
const BENCH_ROW_H = 56;

const META_X = 30;
const META_Y = 60;
const META_W = 270;
const META_H = 460;

// 7 position layout on the pitch (4-2-1 / 2-3-1 youth diamond-ish).
// Coordinates relative to PITCH origin in [0, 1].
const POSITION_RELATIVE: Array<{x: number; y: number}> = [
  {x: 0.5,  y: 0.95},   // A: GK (back)
  {x: 0.18, y: 0.72},   // B: LB
  {x: 0.82, y: 0.72},   // C: RB
  {x: 0.5,  y: 0.55},   // D: CB / sweeper
  {x: 0.27, y: 0.30},   // E: LM
  {x: 0.73, y: 0.30},   // F: RM
  {x: 0.5,  y: 0.10},   // G: ST
];

const COLOR_PITCH = '#16a34a';       // grass
const COLOR_LINE = '#ffffff';
const COLOR_PLAYER_FIELD = '#1d4ed8';
const COLOR_PLAYER_BENCH = '#94a3b8';
const COLOR_GOAL_US = '#facc15';
const COLOR_GOAL_THEM = '#dc2626';

export interface SoccerFrameInput {
  /** Game minute (integer). */
  t: number;
  /** Period index (0-based). */
  period: number;
  /** Length numPositions; positions[pos] = playerId. */
  positions: number[];
  /** Length benchSize; bench[i] = playerId. */
  bench: number[];
  goalsFor: number;
  goalsAgainst: number;
  affinityNow: number;          // average affinity in [0, 1]
  /** Optional flash for "this minute had a goal event". */
  goalThisTick?: 'us' | 'them' | null;
  problem: SoccerProblem;
}

export function buildSoccerFrame(t: number, tick: number, input: SoccerFrameInput): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const playerNames = input.problem.playerNames ?? [];
  const positionNames = input.problem.positionNames ?? [];

  // ── Side panel: scoreboard, period, affinity bar ─────────────────────
  shapes.push({kind: 'rect', x: META_X, y: META_Y, w: META_W, h: META_H,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 36,
               text: '7v7 Match', fontSize: 22, fill: '#f1f5f9',
               anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 70,
               text: `Period ${input.period + 1} / ${input.problem.numPeriods}`,
               fontSize: 16, fill: '#cbd5e1', anchor: 'middle'});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 110,
               text: `Minute ${input.t} / ${input.problem.numPeriods * 20}`,
               fontSize: 14, fill: '#94a3b8', anchor: 'middle'});
  // Score
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 170,
               text: `Us  ${input.goalsFor}  —  ${input.goalsAgainst}  Them`,
               fontSize: 26, fill: '#fde68a', anchor: 'middle', fontWeight: 'bold'});
  // Goal flash (last-tick event)
  if (input.goalThisTick === 'us') {
    shapes.push({kind: 'rect', x: META_X + 30, y: META_Y + 200, w: META_W - 60, h: 30,
                 fill: COLOR_GOAL_US, opacity: 0.9, rx: 6});
    shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 222,
                 text: 'GOAL!', fontSize: 18, fill: '#000', anchor: 'middle', fontWeight: 'bold'});
  } else if (input.goalThisTick === 'them') {
    shapes.push({kind: 'rect', x: META_X + 30, y: META_Y + 200, w: META_W - 60, h: 30,
                 fill: COLOR_GOAL_THEM, opacity: 0.9, rx: 6});
    shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 222,
                 text: 'CONCEDED', fontSize: 16, fill: '#fff', anchor: 'middle', fontWeight: 'bold'});
  }
  // Affinity bar
  shapes.push({kind: 'text', x: META_X + 16, y: META_Y + 280,
               text: 'On-field avg affinity', fontSize: 12, fill: '#94a3b8'});
  shapes.push({kind: 'rect', x: META_X + 16, y: META_Y + 290, w: META_W - 32, h: 14,
               fill: '#334155', stroke: '#475569', strokeWidth: 1, rx: 3});
  shapes.push({kind: 'rect', x: META_X + 16, y: META_Y + 290,
               w: (META_W - 32) * Math.max(0, Math.min(1, input.affinityNow)), h: 14,
               fill: '#22d3ee', rx: 3});
  shapes.push({kind: 'text', x: META_X + META_W - 16, y: META_Y + 320,
               text: `${(input.affinityNow * 100).toFixed(0)}%`, fontSize: 11,
               fill: '#94a3b8', anchor: 'end'});

  // Period boundary watermark.
  if (input.t % 20 === 0 && input.t > 0) {
    shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 380,
                 text: '⟳ SUB WINDOW', fontSize: 18, fill: '#fbbf24',
                 anchor: 'middle', fontWeight: 'bold'});
  }

  // ── Pitch ────────────────────────────────────────────────────────────
  shapes.push({kind: 'rect', x: PITCH_X, y: PITCH_Y, w: PITCH_W, h: PITCH_H,
               fill: COLOR_PITCH, stroke: COLOR_LINE, strokeWidth: 2});
  // Halfway line
  shapes.push({kind: 'line', x1: PITCH_X, y1: PITCH_Y + PITCH_H / 2,
               x2: PITCH_X + PITCH_W, y2: PITCH_Y + PITCH_H / 2,
               stroke: COLOR_LINE, strokeWidth: 2});
  // Centre circle
  shapes.push({kind: 'circle', x: PITCH_X + PITCH_W / 2, y: PITCH_Y + PITCH_H / 2,
               r: 50, fill: 'none', stroke: COLOR_LINE, strokeWidth: 2});
  // Penalty boxes (top + bottom)
  shapes.push({kind: 'rect', x: PITCH_X + PITCH_W / 2 - 90, y: PITCH_Y,
               w: 180, h: 70, fill: 'none', stroke: COLOR_LINE, strokeWidth: 2});
  shapes.push({kind: 'rect', x: PITCH_X + PITCH_W / 2 - 90, y: PITCH_Y + PITCH_H - 70,
               w: 180, h: 70, fill: 'none', stroke: COLOR_LINE, strokeWidth: 2});
  // Goals
  shapes.push({kind: 'rect', x: PITCH_X + PITCH_W / 2 - 30, y: PITCH_Y - 8,
               w: 60, h: 8, fill: COLOR_LINE});
  shapes.push({kind: 'rect', x: PITCH_X + PITCH_W / 2 - 30, y: PITCH_Y + PITCH_H,
               w: 60, h: 8, fill: COLOR_LINE});

  // Players on the pitch
  for (let pos = 0; pos < input.problem.numPositions; pos++) {
    const slot = POSITION_RELATIVE[pos] ?? POSITION_RELATIVE[0];
    const cx = PITCH_X + slot.x * PITCH_W;
    const cy = PITCH_Y + slot.y * PITCH_H;
    const playerId = input.positions[pos];
    const name = playerNames[playerId] ?? `P${playerId}`;
    shapes.push({kind: 'circle', x: cx, y: cy, r: 22, fill: COLOR_PLAYER_FIELD,
                 stroke: '#fff', strokeWidth: 2,
                 title: `${name}  @  position ${positionNames[pos] ?? pos}`});
    shapes.push({kind: 'text', x: cx, y: cy + 5, text: name, fontSize: 12,
                 fill: '#fff', anchor: 'middle', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: cx, y: cy - 28, text: positionNames[pos] ?? '?',
                 fontSize: 11, fill: '#fde68a', anchor: 'middle', fontWeight: 'bold'});
  }

  // ── Bench ────────────────────────────────────────────────────────────
  shapes.push({kind: 'rect', x: BENCH_X, y: BENCH_Y, w: BENCH_W,
               h: BENCH_ROW_H * input.bench.length + 40,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: BENCH_X + BENCH_W / 2, y: BENCH_Y + 26,
               text: 'BENCH', fontSize: 14, fill: '#fde68a', anchor: 'middle',
               fontWeight: 'bold'});
  for (let i = 0; i < input.bench.length; i++) {
    const playerId = input.bench[i];
    const name = playerNames[playerId] ?? `P${playerId}`;
    const y = BENCH_Y + 40 + i * BENCH_ROW_H;
    shapes.push({kind: 'circle', x: BENCH_X + BENCH_W / 2 - 30, y: y + BENCH_ROW_H / 2,
                 r: 18, fill: COLOR_PLAYER_BENCH, stroke: '#94a3b8'});
    shapes.push({kind: 'text', x: BENCH_X + BENCH_W / 2 - 30, y: y + BENCH_ROW_H / 2 + 4,
                 text: name, fontSize: 11, fill: '#0f172a',
                 anchor: 'middle', fontWeight: 'bold'});
  }

  return {
    shapes,
    caption: `t=${input.t}min  P${input.period + 1}  Us ${input.goalsFor}-${input.goalsAgainst} Them  affinity ${(input.affinityNow * 100).toFixed(0)}%`,
  };
}

/** Build companion charts: per-tick affinity and cumulative goal differential. */
export function buildSoccerCharts(
  ts: number[],
  affinity: number[],
  goalsFor: number[],
  goalsAgainst: number[],
): ChartSpec[] {
  return [
    {
      x: META_X, y: META_Y + META_H + 20, w: META_W, h: 100,
      title: 'On-field avg affinity',
      yMin: 0, yMax: 1,
      series: [{label: 'affinity', color: '#22d3ee', t: ts, y: affinity}],
    },
    {
      x: PITCH_X, y: PITCH_Y + PITCH_H + 20, w: PITCH_W, h: 100,
      title: 'Cumulative goals',
      series: [
        {label: 'us',   color: COLOR_GOAL_US,   t: ts, y: goalsFor},
        {label: 'them', color: COLOR_GOAL_THEM, t: ts, y: goalsAgainst},
      ],
    },
  ];
}
