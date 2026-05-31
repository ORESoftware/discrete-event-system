'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/dc_motor_scene.rs
// - DcMotorSceneOpts becomes a Rust config struct and DcMotorScene becomes a state/lightweight builder struct with inherent methods.
// - MotorStateToken and DcMotorParams should stay typed imports from general::control_systems::dc_motor.
// - Returned Frame/ChartSpec/Shape values should be serde structs/enums; optional controls become Option<T>.
// - If the class is inserted into the DES graph, implement PureTransform for DcMotorScene with transform(state) -> Frame fragment.

// =============================================================================
// DC-motor scene builder (class-based).
//
// Renders the armature circuit (supply V, resistance R, inductance L, and the
// back-EMF source E = K_eω), a spinning rotor, live gauges, and time-series
// charts of speed (with reference), back-EMF, current and applied voltage.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {DcMotorParams, MotorStateToken} from '../../general/control-systems/dc-motor';

export const MOTOR_STAGE_W = 1000;
export const MOTOR_STAGE_H = 760;

export interface DcMotorSceneOpts {
  samples: readonly MotorStateToken[];
  dt: number;
  params: DcMotorParams;
  modeName: string;
  /** reference speed per sample (closed loop), or null for open loop */
  reference?: readonly number[] | null;
}

const COL_BG = '#0b1021';
const COL_PANEL = '#161d33';
const COL_WIRE = '#64748b';
const COL_V = '#fbbf24';
const COL_EMF = '#f472b6';
const COL_I = '#38bdf8';
const COL_OMEGA = '#a78bfa';
const COL_REF = '#22c55e';

export class DcMotorScene {
  private readonly opts: DcMotorSceneOpts;
  private readonly spinAngle: number[] = [];
  private readonly times: number[];
  private readonly maxAbsI: number;
  private readonly maxV: number;
  private readonly maxOmega: number;

  constructor(opts: DcMotorSceneOpts) {
    this.opts = opts;
    let angle = 0;
    for (const s of opts.samples) {
      angle += s.omega * opts.dt * 0.15;
      this.spinAngle.push(angle);
    }
    this.times = opts.samples.map(s => s.time);
    this.maxAbsI = Math.max(1, ...opts.samples.map(s => Math.abs(s.current)));
    this.maxV = Math.max(1, ...opts.samples.map(s => Math.abs(s.voltage)));
    this.maxOmega = Math.max(1, ...opts.samples.map(s => s.omega), ...(opts.reference ?? []));
  }

  frameCount(): number {
    return this.opts.samples.length;
  }

  timeAt(i: number): number {
    return this.times[i];
  }

  frameAt(i: number): Omit<Frame, 't' | 'tick'> {
    const s = this.opts.samples[i];
    const ref = this.opts.reference ? this.opts.reference[i] : null;
    const shapes: Shape[] = [];
    shapes.push({kind: 'rect', x: 0, y: 0, w: MOTOR_STAGE_W, h: MOTOR_STAGE_H, fill: COL_BG});
    shapes.push({kind: 'text', x: MOTOR_STAGE_W / 2, y: 34, anchor: 'middle', fontSize: 22, fontWeight: 'bold',
      fill: '#f8fafc', text: `DC Motor (back-EMF ODE) — ${this.opts.modeName}`});

    this.drawCircuit(shapes, s);
    this.drawRotor(shapes, 700, 250, this.spinAngle[i], s.omega);
    this.drawGauges(shapes, s, ref);

    const refStr = ref === null ? '' : `   ω*=${ref.toFixed(1)}`;
    return {
      shapes,
      caption: `t=${s.time.toFixed(3)}s   V=${s.voltage.toFixed(2)} V   i=${s.current.toFixed(3)} A   ` +
        `ω=${s.omega.toFixed(2)} rad/s${refStr}   E=K_eω=${s.backEmf.toFixed(2)} V   T_L=${s.loadTorque.toFixed(2)} N·m`,
    };
  }

  charts(): ChartSpec[] {
    const t = this.times;
    const s = this.opts.samples;
    const series = [{label: 'ω', color: COL_OMEGA, t, y: s.map(x => x.omega)}];
    if (this.opts.reference) series.push({label: 'ω*', color: COL_REF, t, y: this.opts.reference.slice()});
    return [
      {
        x: 40, y: 540, w: 460, h: 200, title: 'Rotor speed ω (rad/s)', yLabel: 'rad/s',
        yMin: 0, yMax: this.maxOmega * 1.15, series,
      },
      {
        x: 520, y: 540, w: 440, h: 200, title: 'Back-EMF  E = K_e·ω (V)', yLabel: 'V',
        yMin: 0, yMax: Math.max(1, ...s.map(x => x.backEmf)) * 1.2,
        series: [{label: 'E', color: COL_EMF, t, y: s.map(x => x.backEmf)}],
      },
      {
        x: 40, y: 470, w: 460, h: 60, title: 'Armature current i (A)', yLabel: 'A',
        yMin: -this.maxAbsI * 1.1, yMax: this.maxAbsI * 1.1,
        series: [{label: 'i', color: COL_I, t, y: s.map(x => x.current)}],
      },
      {
        x: 520, y: 470, w: 440, h: 60, title: 'Applied voltage V (V)', yLabel: 'V',
        yMin: Math.min(0, ...s.map(x => x.voltage)) * 1.1 - 0.1, yMax: this.maxV * 1.1,
        series: [{label: 'V', color: COL_V, t, y: s.map(x => x.voltage)}],
      },
    ];
  }

  private drawCircuit(shapes: Shape[], s: MotorStateToken): void {
    // Loop rectangle: supply (left vertical), top wire to motor, motor (right), bottom wire.
    const L = 90, R = 560, T = 110, B = 360;
    shapes.push({kind: 'rect', x: L - 30, y: T - 30, w: R - L + 120, h: B - T + 70, rx: 10, fill: COL_PANEL, stroke: '#334155'});
    // Wires.
    shapes.push({kind: 'line', x1: L, y1: T, x2: R, y2: T, stroke: COL_WIRE, strokeWidth: 3});
    shapes.push({kind: 'line', x1: L, y1: B, x2: R, y2: B, stroke: COL_WIRE, strokeWidth: 3});
    shapes.push({kind: 'line', x1: L, y1: T, x2: L, y2: B, stroke: COL_WIRE, strokeWidth: 3});
    shapes.push({kind: 'line', x1: R, y1: T, x2: R, y2: B, stroke: COL_WIRE, strokeWidth: 3});

    // Supply source (left).
    shapes.push({kind: 'circle', x: L, y: (T + B) / 2, r: 26, fill: '#1e293b', stroke: COL_V, strokeWidth: 2});
    shapes.push({kind: 'text', x: L, y: (T + B) / 2 - 2, anchor: 'middle', fontSize: 13, fill: COL_V, fontWeight: 'bold', text: 'V'});
    shapes.push({kind: 'text', x: L, y: (T + B) / 2 + 16, anchor: 'middle', fontSize: 11, fill: COL_V, text: `${s.voltage.toFixed(1)}`});

    // Resistor R (top wire, box).
    shapes.push({kind: 'rect', x: 220, y: T - 12, w: 70, h: 24, rx: 3, fill: '#1e293b', stroke: COL_WIRE});
    shapes.push({kind: 'text', x: 255, y: T + 5, anchor: 'middle', fontSize: 12, fill: '#cbd5e1', text: `R=${this.opts.params.resistance}Ω`});
    // Inductor L (top wire, coil hint).
    shapes.push({kind: 'rect', x: 340, y: T - 12, w: 70, h: 24, rx: 12, fill: '#1e293b', stroke: COL_WIRE});
    shapes.push({kind: 'text', x: 375, y: T + 5, anchor: 'middle', fontSize: 12, fill: '#cbd5e1', text: `L=${this.opts.params.inductance}H`});

    // Back-EMF source (right vertical, inside loop just left of motor).
    const emfX = R, emfY = (T + B) / 2;
    shapes.push({kind: 'circle', x: emfX, y: emfY, r: 26, fill: '#1e293b', stroke: COL_EMF, strokeWidth: 2});
    shapes.push({kind: 'text', x: emfX, y: emfY - 2, anchor: 'middle', fontSize: 12, fill: COL_EMF, fontWeight: 'bold', text: 'E'});
    shapes.push({kind: 'text', x: emfX, y: emfY + 16, anchor: 'middle', fontSize: 11, fill: COL_EMF, text: `${s.backEmf.toFixed(1)}`});
    shapes.push({kind: 'text', x: emfX, y: emfY + 44, anchor: 'middle', fontSize: 10, fill: '#94a3b8', text: 'K_e·ω'});

    // Current-flow markers around the loop (direction = sign of i; speed ∝ |i|).
    this.drawCurrentMarkers(shapes, L, R, T, B, s.current);
    shapes.push({kind: 'text', x: 255, y: B + 26, anchor: 'middle', fontSize: 13, fill: COL_I, fontWeight: 'bold',
      text: `i = ${s.current.toFixed(3)} A`});
  }

  private drawCurrentMarkers(shapes: Shape[], L: number, R: number, T: number, B: number, current: number): void {
    const perim = 2 * (R - L) + 2 * (B - T);
    const count = 8;
    // Phase advances with cumulative |i|; sign sets direction.
    const phase = ((current >= 0 ? 1 : -1) * (Math.abs(current) * 7) % perim + perim) % perim;
    for (let k = 0; k < count; k++) {
      let d = (phase + (k / count) * perim) % perim;
      const p = this.perimeterPoint(d, L, R, T, B);
      const opacity = 0.35 + 0.5 * Math.min(1, Math.abs(current) / Math.max(1, this.maxAbsI));
      shapes.push({kind: 'circle', x: p.x, y: p.y, r: 4, fill: COL_I, opacity});
    }
  }

  private perimeterPoint(d: number, L: number, R: number, T: number, B: number): {x: number; y: number} {
    const top = R - L, right = B - T, bottom = R - L, left = B - T;
    if (d < top) return {x: L + d, y: T};
    d -= top;
    if (d < right) return {x: R, y: T + d};
    d -= right;
    if (d < bottom) return {x: R - d, y: B};
    d -= bottom;
    return {x: L, y: B - d};
  }

  private drawRotor(shapes: Shape[], cx: number, cy: number, angle: number, omega: number): void {
    shapes.push({kind: 'circle', x: cx, y: cy, r: 64, fill: '#1e293b', stroke: COL_WIRE, strokeWidth: 3});
    shapes.push({kind: 'circle', x: cx, y: cy, r: 58, fill: 'transparent', stroke: '#334155', strokeWidth: 1});
    // Rotor spokes (4) to show rotation.
    for (let k = 0; k < 4; k++) {
      const a = angle + (k * Math.PI) / 2;
      shapes.push({kind: 'line', x1: cx, y1: cy, x2: cx + 54 * Math.cos(a), y2: cy + 54 * Math.sin(a),
        stroke: COL_OMEGA, strokeWidth: 3});
    }
    shapes.push({kind: 'circle', x: cx, y: cy, r: 8, fill: COL_OMEGA});
    shapes.push({kind: 'text', x: cx, y: cy + 92, anchor: 'middle', fontSize: 13, fill: COL_OMEGA, fontWeight: 'bold',
      text: `ω = ${omega.toFixed(1)} rad/s`});
    shapes.push({kind: 'text', x: cx, y: cy - 84, anchor: 'middle', fontSize: 12, fill: '#94a3b8', text: 'ROTOR'});
  }

  private drawGauges(shapes: Shape[], s: MotorStateToken, ref: number | null): void {
    const x = 820, y = 110, w = 150, rowH = 64;
    const rows: Array<[string, string, string]> = [
      ['speed ω', `${s.omega.toFixed(1)} rad/s`, COL_OMEGA],
      ['back-EMF E', `${s.backEmf.toFixed(2)} V`, COL_EMF],
      ['current i', `${s.current.toFixed(3)} A`, COL_I],
      ['voltage V', `${s.voltage.toFixed(2)} V`, COL_V],
    ];
    if (ref !== null) rows.unshift(['reference ω*', `${ref.toFixed(1)} rad/s`, COL_REF]);
    shapes.push({kind: 'rect', x, y, w, h: rows.length * rowH + 12, rx: 8, fill: COL_PANEL, stroke: '#334155'});
    rows.forEach((r, k) => {
      const ry = y + 12 + k * rowH;
      shapes.push({kind: 'text', x: x + 12, y: ry + 18, anchor: 'start', fontSize: 11, fill: '#94a3b8', text: r[0]});
      shapes.push({kind: 'text', x: x + 12, y: ry + 42, anchor: 'start', fontSize: 18, fill: r[2], fontWeight: 'bold', text: r[1]});
    });
  }
}
