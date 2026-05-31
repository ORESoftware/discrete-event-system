'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/temp_control_scene.rs
// - SceneData should become a Rust struct and buildTempControlFrame/buildTempControlAnimation remain module helpers.
// - RunResult/TickRecord stay typed imports from general::temp_control; animation output is serde Animation/Frame/Shape data.
// - Private drawing/chart helpers should take &mut Vec<Shape> and return Result only if rendering can fail.
// - If temperature-control frames become graph-visible, use a TempControlSceneTransform implementing PureTransform::transform.

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/temp-control-scene.rs   (module des::animation::scenes::temp_control_scene)
// 1:1 file move. Builds frames + charts for the temperature-control DES animation.
//
// Declarations → Rust:
//   const STAGE_W/H, COL_* consts   -> `pub const`/`const`
//   interface SceneData             -> struct SceneData
//   function buildTempControlFrame  -> pub fn -> FrameParts
//   function buildTempControlAnimation -> pub fn
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - multi-line station labels via `s.label.split('\n')` -> `str::split('\n')`.
//   - all coords/temperatures are `number` -> `f64`.
//   - imports RunResult/TickRecord from ../../general/temp-control -> `use crate::des::general::temp_control::*`.
// =============================================================================

// =============================================================================
// Temperature-control scene builder.
//
// Layout (1000×640):
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │  (top row: station diagram with movables flowing in animated dots)   │
//   │                                                                      │
//   │  (middle row: house thermometer + heater dial)                       │
//   │                                                                      │
//   │  Charts: T_in & T_out & target band over time                        │
//   │          Heater command Q over time                                  │
//   └──────────────────────────────────────────────────────────────────────┘
// =============================================================================

import {Frame, Shape} from '../types';
import {RunResult, TickRecord} from '../../general/temp-control';

export const STAGE_W = 1000;
export const STAGE_H = 700;

const COL_TARGET = '#16a34a';
const COL_BAND   = '#bbf7d0';
const COL_T_IN   = '#dc2626';
const COL_T_OUT  = '#1d4ed8';
const COL_HEAT   = '#f97316';
const COL_BG     = '#f9fafb';

interface SceneData {
  tick: TickRecord;
  T_target: number;
  band: number;
  Q_max: number;
  controllerName: string;
  energy: number;
  comfortPct: number;
}

/** Build one frame's static layout (thermometer, heater, controller diagram). */
export function buildTempControlFrame(t: number, tick: number, d: SceneData): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];

  // Background
  shapes.push({kind: 'rect', x: 0, y: 0, w: STAGE_W, h: STAGE_H, fill: COL_BG});

  // ── Top row: station-flow diagram ───────────────────────────────────────
  const stations = [
    {x: 60,  label: 'Outdoor\nSource'},
    {x: 220, label: 'Forecast\nStation'},
    {x: 380, label: 'Sensor +\nComparator'},
    {x: 540, label: d.controllerName},
    {x: 700, label: 'Heater'},
    {x: 860, label: 'House\n(Physics)'},
  ];
  const stRowY = 70;
  for (const s of stations) {
    shapes.push({kind: 'rect', x: s.x - 60, y: stRowY, w: 120, h: 60,
      fill: '#fff', stroke: '#888', strokeWidth: 1.2, rx: 6});
    const lines = s.label.split('\n');
    for (let i = 0; i < lines.length; i++) {
      shapes.push({kind: 'text', x: s.x, y: stRowY + 25 + i * 16, text: lines[i],
        fontSize: 12, fill: '#222', anchor: 'middle', fontWeight: i === 0 ? 'bold' : 'normal'});
    }
  }
  // Arrows between stations.
  for (let i = 0; i < stations.length - 1; i++) {
    const x1 = stations[i].x + 60, x2 = stations[i + 1].x - 60;
    shapes.push({kind: 'line', x1, y1: stRowY + 30, x2: x2 - 6, y2: stRowY + 30,
      stroke: '#888', strokeWidth: 1.5});
    shapes.push({kind: 'path', d: `M ${x2 - 8},${stRowY + 26} L ${x2},${stRowY + 30} L ${x2 - 8},${stRowY + 34}`,
      fill: '#888', stroke: '#888'});
  }
  // Feedback arrow from House → Sensor.
  shapes.push({kind: 'path',
    d: `M ${stations[5].x},${stRowY + 60} V ${stRowY + 100} H ${stations[2].x} V ${stRowY + 60}`,
    stroke: '#888', strokeWidth: 1.5, fill: 'transparent'});
  shapes.push({kind: 'text', x: 620, y: stRowY + 115, text: 'feedback',
    fontSize: 11, fill: '#666', anchor: 'middle'});
  // Pulse circle on the line representing the current movable.
  const phase = (tick % 60) / 60;
  const idx = Math.floor(phase * (stations.length - 1));
  const seg = phase * (stations.length - 1) - idx;
  const cx = stations[idx].x + 60 + seg * (stations[idx + 1].x - 60 - (stations[idx].x + 60));
  shapes.push({kind: 'circle', x: cx, y: stRowY + 30, r: 4, fill: '#ec4899', stroke: '#831843'});

  // ── Middle row: thermometer + heater dial + numeric readouts ────────────
  const midY = 200;
  // Thermometer: vertical bar from 50°F to 90°F, with target band shaded.
  const thermX = 120, thermW = 40;
  const thermYTop = midY, thermYBot = midY + 280;
  const tMin = 50, tMax = 90;
  const tToY = (T: number) => thermYBot - ((T - tMin) / (tMax - tMin)) * (thermYBot - thermYTop);
  // Frame
  shapes.push({kind: 'rect', x: thermX, y: thermYTop, w: thermW, h: thermYBot - thermYTop,
    fill: '#fff', stroke: '#444', strokeWidth: 1.5, rx: 4});
  // Target band shading
  const yBandHi = tToY(d.T_target + d.band), yBandLo = tToY(d.T_target - d.band);
  shapes.push({kind: 'rect', x: thermX + 1, y: yBandHi, w: thermW - 2, h: yBandLo - yBandHi,
    fill: COL_BAND, opacity: 0.7});
  // Indoor temperature column (red)
  const yIn = tToY(d.tick.T_in_true);
  shapes.push({kind: 'rect', x: thermX + 12, y: yIn, w: thermW - 24,
    h: thermYBot - yIn, fill: COL_T_IN, opacity: 0.85});
  // Tick marks every 5°F
  for (let T = tMin; T <= tMax; T += 5) {
    const y = tToY(T);
    shapes.push({kind: 'line', x1: thermX + thermW, y1: y, x2: thermX + thermW + 6, y2: y,
      stroke: '#444', strokeWidth: 1});
    shapes.push({kind: 'text', x: thermX + thermW + 10, y: y + 4, text: `${T}`,
      fontSize: 10, fill: '#444', anchor: 'start'});
  }
  // Target line
  shapes.push({kind: 'line', x1: thermX - 5, y1: tToY(d.T_target), x2: thermX + thermW + 5, y2: tToY(d.T_target),
    stroke: COL_TARGET, strokeWidth: 2, dasharray: '4,3'});
  // Bulb at the bottom
  shapes.push({kind: 'circle', x: thermX + thermW / 2, y: thermYBot + 18, r: 22,
    fill: COL_T_IN, stroke: '#444', strokeWidth: 1.5});
  // T_in numeric readout
  shapes.push({kind: 'text', x: thermX + thermW / 2, y: thermYTop - 12,
    text: `T_in = ${d.tick.T_in_true.toFixed(2)}°F`, fontSize: 14, fill: '#222', anchor: 'middle', fontWeight: 'bold'});

  // Heater dial — circle gauge, 0..Q_max
  const dialX = 320, dialY = midY + 130, dialR = 90;
  shapes.push({kind: 'circle', x: dialX, y: dialY, r: dialR + 5,
    fill: '#fff', stroke: '#444', strokeWidth: 1.5});
  // Arc for current Q level
  const Qfrac = Math.max(0, Math.min(1, d.tick.Q / d.Q_max));
  // Pie wedge from -π/2 around clockwise; angle = π * (1 + Qfrac) for 180° max, but use 270° gauge.
  const startAngle = Math.PI * (5 / 6);          // 150° (lower-left)
  const endAngle = Math.PI * (5 / 6) + Qfrac * (Math.PI * 4 / 3);   // 240° span
  const arcEndX = dialX + dialR * Math.cos(endAngle);
  const arcEndY = dialY + dialR * Math.sin(endAngle);
  const arcStartX = dialX + dialR * Math.cos(startAngle);
  const arcStartY = dialY + dialR * Math.sin(startAngle);
  const largeArc = (endAngle - startAngle) > Math.PI ? 1 : 0;
  shapes.push({kind: 'path',
    d: `M ${dialX} ${dialY} L ${arcStartX} ${arcStartY} A ${dialR} ${dialR} 0 ${largeArc} 1 ${arcEndX} ${arcEndY} Z`,
    fill: COL_HEAT, opacity: 0.85, stroke: '#7c2d12', strokeWidth: 1});
  shapes.push({kind: 'text', x: dialX, y: dialY - 5, text: `Q = ${d.tick.Q.toFixed(2)} kW`,
    fontSize: 14, fill: '#222', anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: dialX, y: dialY + 18, text: `${(Qfrac * 100).toFixed(0)}% of max`,
    fontSize: 11, fill: '#666', anchor: 'middle'});

  // Outdoor temperature mini-display
  const outX = 480, outY = midY;
  shapes.push({kind: 'rect', x: outX, y: outY, w: 200, h: 80, fill: '#fff', stroke: '#444', rx: 6});
  shapes.push({kind: 'text', x: outX + 100, y: outY + 18, text: 'OUTSIDE',
    fontSize: 11, fill: '#666', anchor: 'middle'});
  shapes.push({kind: 'text', x: outX + 100, y: outY + 50,
    text: `${d.tick.T_out_true.toFixed(1)}°F`, fontSize: 28, fill: COL_T_OUT, anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: outX + 100, y: outY + 70,
    text: `t = ${(d.tick.t_h).toFixed(2)} h`, fontSize: 11, fill: '#666', anchor: 'middle'});

  // Comfort + energy box
  const cmfX = 480, cmfY = midY + 100;
  shapes.push({kind: 'rect', x: cmfX, y: cmfY, w: 200, h: 70, fill: '#fff', stroke: '#444', rx: 6});
  shapes.push({kind: 'text', x: cmfX + 8, y: cmfY + 18, text: 'COMFORT', fontSize: 11, fill: '#666', anchor: 'start'});
  shapes.push({kind: 'text', x: cmfX + 8, y: cmfY + 38, text: `${(d.comfortPct * 100).toFixed(1)}% in band`,
    fontSize: 14, fill: '#222', anchor: 'start', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: cmfX + 8, y: cmfY + 56, text: `Energy: ${d.energy.toFixed(2)} kWh`,
    fontSize: 12, fill: '#444', anchor: 'start'});
  // Out-of-band indicator
  if (!d.tick.in_band) {
    shapes.push({kind: 'rect', x: cmfX + 168, y: cmfY + 8, w: 24, h: 24,
      fill: '#dc2626', stroke: '#7f1d1d', rx: 4});
    shapes.push({kind: 'text', x: cmfX + 180, y: cmfY + 25, text: '!',
      fontSize: 18, fill: '#fff', anchor: 'middle', fontWeight: 'bold'});
  }

  // Title
  shapes.push({kind: 'text', x: STAGE_W / 2, y: 30, text: `Temperature Control — ${d.controllerName}`,
    fontSize: 18, fill: '#111', anchor: 'middle', fontWeight: 'bold'});

  return {
    shapes,
    caption: `t = ${d.tick.t_h.toFixed(2)}h   T_in = ${d.tick.T_in_true.toFixed(2)}°F   T_out = ${d.tick.T_out_true.toFixed(1)}°F   Q = ${d.tick.Q.toFixed(2)} kW   in_band=${d.tick.in_band}`,
  };
}

/** Convenience: build all frames + charts for a given RunResult. */
export function buildTempControlAnimation(run: RunResult, controllerName: string, recordEvery: number = 5):
{frames: Array<{t: number; tick: number} & Omit<Frame, 't' | 'tick'>>; charts: any[]} {
  const T_target = run.cfg.T_target;
  const band = run.cfg.band ?? 2;
  const Q_max = 5;
  // Build per-tick scene data
  const frames: Array<{t: number; tick: number} & Omit<Frame, 't' | 'tick'>> = [];
  let inBandCount = 0;
  for (let k = 0; k < run.trace.length; k++) {
    if (run.trace[k].in_band) inBandCount++;
    if (k % recordEvery !== 0 && k !== run.trace.length - 1) continue;
    const tk = run.trace[k];
    const data: SceneData = {
      tick: tk, T_target, band, Q_max, controllerName,
      energy: tk.energy_cum_kWh,
      comfortPct: inBandCount / (k + 1),
    };
    const f = buildTempControlFrame(tk.t_h, tk.tick, data);
    frames.push({t: tk.t_h, tick: tk.tick, ...f});
  }

  // Charts
  const t_h_arr = run.trace.map(r => r.t_h);
  const charts = [
    {
      x: 40, y: 510, w: 600, h: 170,
      title: 'Temperatures (°F)',
      yMin: Math.min(...run.T_out, ...run.T_in) - 2,
      yMax: Math.max(...run.T_out, ...run.T_in) + 2,
      yLabel: '°F',
      series: [
        {label: 'T_in', color: COL_T_IN, t: t_h_arr, y: run.T_in},
        {label: 'T_out', color: COL_T_OUT, t: t_h_arr, y: run.T_out},
        {label: 'target', color: COL_TARGET, t: [t_h_arr[0], t_h_arr[t_h_arr.length - 1]], y: [T_target, T_target]},
      ],
    },
    {
      x: 660, y: 510, w: 320, h: 170,
      title: 'Heater Q (kW)',
      yMin: -0.2, yMax: Q_max + 0.2,
      yLabel: 'kW',
      series: [
        {label: 'Q', color: COL_HEAT, t: t_h_arr, y: run.Q},
      ],
    },
  ];
  return {frames, charts};
}
