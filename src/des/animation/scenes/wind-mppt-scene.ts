'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/wind-mppt-scene.rs   (module des::animation::scenes::wind_mppt_scene)
// 1:1 file move. Class-based builder for the wind-turbine MPPT animation.
//
// Declarations → Rust:
//   const WIND_STAGE_W/H, COL_* consts  -> `pub const`/`const`
//   interface WindSceneOpts             -> struct WindSceneOpts
//   class WindMpptScene                 -> struct WindMpptScene { fields } + impl
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `samples: readonly TurbineStateToken[]` -> `&[TurbineStateToken]` (or owned `Vec`).
//   - class methods that build frames -> `&self` methods on the struct.
//   - all coords/physics values are `number` -> `f64`.
//   - imports TurbineStateToken from ../../general/control-systems/wind-mppt ->
//     `use crate::des::general::control_systems::wind_mppt::TurbineStateToken`.
// =============================================================================

// =============================================================================
// Wind-MPPT scene builder (class-based).
//
// Renders a spinning variable-speed wind turbine, live wind arrows, λ / C_p
// gauges against their optimal targets, and time-series charts of λ, C_p, ω
// and captured power. Frames are replayed from the recorded TurbineStateToken
// trajectory.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {TurbineStateToken} from '../../general/control-systems/wind-mppt';

export const WIND_STAGE_W = 1000;
export const WIND_STAGE_H = 760;

export interface WindSceneOpts {
  samples: readonly TurbineStateToken[];
  dt: number;
  lambdaStar: number;
  cpMax: number;
  kOpt: number;
  controllerName: string;
}

const COL_BG = '#0b1021';
const COL_PANEL = '#161d33';
const COL_BLADE = '#e2e8f0';
const COL_HUB = '#94a3b8';
const COL_WIND = '#38bdf8';
const COL_LAMBDA = '#f59e0b';
const COL_CP = '#34d399';
const COL_OMEGA = '#a78bfa';
const COL_POWER = '#fb7185';
const COL_TARGET = '#22c55e';

export class WindMpptScene {
  private readonly opts: WindSceneOpts;
  private readonly spinAngle: number[] = [];
  private readonly times: number[];
  private readonly maxPower: number;
  private readonly maxOmega: number;

  constructor(opts: WindSceneOpts) {
    this.opts = opts;
    // Pre-integrate a (visually scaled) blade rotation angle from ω.
    let angle = 0;
    for (const s of opts.samples) {
      angle += s.omega * opts.dt * 0.25;
      this.spinAngle.push(angle);
    }
    this.times = opts.samples.map(s => s.time);
    this.maxPower = Math.max(1, ...opts.samples.map(s => s.mechPower));
    this.maxOmega = Math.max(1, ...opts.samples.map(s => s.omega));
  }

  frameCount(): number {
    return this.opts.samples.length;
  }

  /** Simulation time at sample index `i`. */
  timeAt(i: number): number {
    return this.times[i];
  }

  /** Build the scene at sample index `i`. */
  frameAt(i: number): Omit<Frame, 't' | 'tick'> {
    const s = this.opts.samples[i];
    const shapes: Shape[] = [];
    shapes.push({kind: 'rect', x: 0, y: 0, w: WIND_STAGE_W, h: WIND_STAGE_H, fill: COL_BG});
    shapes.push({kind: 'text', x: WIND_STAGE_W / 2, y: 34, anchor: 'middle', fontSize: 22, fontWeight: 'bold',
      fill: '#f8fafc', text: `Wind MPPT — PMSG WECS  ·  ${this.opts.controllerName}`});

    this.drawWind(shapes, s.windSpeed);
    this.drawTurbine(shapes, 360, 300, this.spinAngle[i]);
    this.drawGauges(shapes, s);
    this.drawReadouts(shapes, s);

    const capturePct = (s.cp / this.opts.cpMax) * 100;
    return {
      shapes,
      caption: `t=${s.time.toFixed(2)}s   V=${s.windSpeed.toFixed(2)} m/s   ω=${s.omega.toFixed(2)} rad/s   ` +
        `λ=${s.lambda.toFixed(2)} (λ*=${this.opts.lambdaStar.toFixed(2)})   C_p=${s.cp.toFixed(3)} (${capturePct.toFixed(1)}% of max)   ` +
        `P=${(s.mechPower / 1000).toFixed(2)} kW`,
    };
  }

  charts(): ChartSpec[] {
    const t = this.times;
    const samples = this.opts.samples;
    const end = t[t.length - 1];
    return [
      {
        x: 40, y: 540, w: 460, h: 200, title: 'Tip-speed ratio λ', yLabel: 'λ',
        yMin: 0, yMax: Math.max(this.opts.lambdaStar * 1.4, ...samples.map(s => s.lambda)) + 1,
        series: [
          {label: 'λ', color: COL_LAMBDA, t, y: samples.map(s => s.lambda)},
          {label: 'λ*', color: COL_TARGET, t: [t[0], end], y: [this.opts.lambdaStar, this.opts.lambdaStar]},
        ],
      },
      {
        x: 520, y: 540, w: 440, h: 200, title: 'Power coefficient C_p', yLabel: 'C_p',
        yMin: 0, yMax: this.opts.cpMax * 1.25,
        series: [
          {label: 'C_p', color: COL_CP, t, y: samples.map(s => s.cp)},
          {label: 'C_p,max', color: COL_TARGET, t: [t[0], end], y: [this.opts.cpMax, this.opts.cpMax]},
        ],
      },
      {
        x: 40, y: 300, w: 240, h: 150, title: 'Rotor speed ω (rad/s)', yLabel: 'ω',
        yMin: 0, yMax: this.maxOmega * 1.15,
        series: [{label: 'ω', color: COL_OMEGA, t, y: samples.map(s => s.omega)}],
      },
      {
        x: 40, y: 460, w: 240, h: 70, title: 'Captured power (kW)', yLabel: 'kW',
        yMin: 0, yMax: (this.maxPower / 1000) * 1.15,
        series: [{label: 'P', color: COL_POWER, t, y: samples.map(s => s.mechPower / 1000)}],
      },
    ];
  }

  private drawWind(shapes: Shape[], windSpeed: number): void {
    const arrowCount = Math.max(2, Math.round(windSpeed / 2));
    const len = 40 + windSpeed * 6;
    for (let k = 0; k < arrowCount; k++) {
      const y = 140 + k * 36;
      shapes.push({kind: 'line', x1: 60, y1: y, x2: 60 + len, y2: y, stroke: COL_WIND, strokeWidth: 2, opacity: 0.8});
      shapes.push({kind: 'path', d: `M ${60 + len - 10},${y - 5} L ${60 + len},${y} L ${60 + len - 10},${y + 5}`,
        stroke: COL_WIND, fill: COL_WIND});
    }
    shapes.push({kind: 'text', x: 60, y: 120, anchor: 'start', fontSize: 14, fill: COL_WIND, fontWeight: 'bold',
      text: `wind ${windSpeed.toFixed(1)} m/s →`});
  }

  private drawTurbine(shapes: Shape[], cx: number, cy: number, angle: number): void {
    // Tower.
    shapes.push({kind: 'path', d: `M ${cx - 14},${cy + 260} L ${cx - 5},${cy} L ${cx + 5},${cy} L ${cx + 14},${cy + 260} Z`,
      fill: '#475569', stroke: '#1e293b'});
    // Nacelle.
    shapes.push({kind: 'rect', x: cx - 18, y: cy - 14, w: 50, h: 28, rx: 6, fill: '#64748b', stroke: '#1e293b'});
    // Three blades.
    for (let b = 0; b < 3; b++) {
      const a = angle + (b * 2 * Math.PI) / 3;
      const tipX = cx + 170 * Math.cos(a);
      const tipY = cy + 170 * Math.sin(a);
      const perpX = 12 * Math.cos(a + Math.PI / 2);
      const perpY = 12 * Math.sin(a + Math.PI / 2);
      shapes.push({kind: 'path',
        d: `M ${cx + perpX},${cy + perpY} L ${tipX},${tipY} L ${cx - perpX},${cy - perpY} Z`,
        fill: COL_BLADE, stroke: '#94a3b8', opacity: 0.95});
    }
    // Hub.
    shapes.push({kind: 'circle', x: cx, y: cy, r: 14, fill: COL_HUB, stroke: '#1e293b', strokeWidth: 2});
  }

  private drawGauges(shapes: Shape[], s: TurbineStateToken): void {
    this.drawBar(shapes, 720, 110, 'λ / λ*', s.lambda, this.opts.lambdaStar, this.opts.lambdaStar * 1.4, COL_LAMBDA);
    this.drawBar(shapes, 850, 110, 'C_p / C_p,max', s.cp, this.opts.cpMax, this.opts.cpMax * 1.25, COL_CP);
  }

  private drawBar(shapes: Shape[], x: number, y: number, label: string, value: number, target: number, max: number, color: string): void {
    const h = 300, w = 46;
    shapes.push({kind: 'rect', x, y, w, h, rx: 6, fill: COL_PANEL, stroke: '#334155'});
    const frac = Math.max(0, Math.min(1, value / max));
    const fillH = frac * (h - 4);
    shapes.push({kind: 'rect', x: x + 2, y: y + h - 2 - fillH, w: w - 4, h: fillH, rx: 4, fill: color, opacity: 0.9});
    const tgtY = y + h - 2 - Math.min(1, target / max) * (h - 4);
    shapes.push({kind: 'line', x1: x - 6, y1: tgtY, x2: x + w + 6, y2: tgtY, stroke: COL_TARGET, strokeWidth: 2, dasharray: '5,3'});
    shapes.push({kind: 'text', x: x + w / 2, y: y - 10, anchor: 'middle', fontSize: 12, fill: '#cbd5e1', fontWeight: 'bold', text: label});
    shapes.push({kind: 'text', x: x + w / 2, y: y + h + 18, anchor: 'middle', fontSize: 13, fill: color, fontWeight: 'bold',
      text: value.toFixed(2)});
  }

  private drawReadouts(shapes: Shape[], s: TurbineStateToken): void {
    const x = 700, y = 440, w = 260, h = 80;
    shapes.push({kind: 'rect', x, y, w, h, rx: 8, fill: COL_PANEL, stroke: '#334155'});
    shapes.push({kind: 'text', x: x + 14, y: y + 24, anchor: 'start', fontSize: 13, fill: '#94a3b8', text: 'Captured power'});
    shapes.push({kind: 'text', x: x + 14, y: y + 56, anchor: 'start', fontSize: 26, fill: COL_POWER, fontWeight: 'bold',
      text: `${(s.mechPower / 1000).toFixed(2)} kW`});
    shapes.push({kind: 'text', x: x + w - 14, y: y + 24, anchor: 'end', fontSize: 12, fill: '#94a3b8',
      text: `T_gen = ${s.genTorque.toFixed(2)} N·m`});
    shapes.push({kind: 'text', x: x + w - 14, y: y + 56, anchor: 'end', fontSize: 12, fill: '#94a3b8',
      text: `K_opt = ${this.opts.kOpt.toExponential(2)}`});
  }
}
