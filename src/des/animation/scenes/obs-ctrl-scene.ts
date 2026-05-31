'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/obs_ctrl_scene.rs
// - StoryStep aliases Frame-without-time in TS; make it a small Rust struct or reuse a FrameBuilder data type.
// - ObsCtrlScene becomes a storyboard struct with inherent methods returning Vec<Shape>/Frame fragments, not a superclass hierarchy.
// - Matrix imports should stay as typed linear_algebra module values; avoid serde_json::Value except at external boundaries.
// - If this storyboard becomes graph-visible, expose a PureTransform implementation that maps control-system state to StoryStep.

// =============================================================================
// Observability / Controllability scene builder (class-based storyboard).
//
// Unlike the dynamical-system scenes, this is a narrated sequence of static
// frames that walk through the structural tests:
//   • Kalman controllability matrix 𝒞 = [B AB … Aⁿ⁻¹B] built column-by-column
//   • Kalman observability matrix 𝒪 = [C; CA; …] built row-by-row
//   • MDP reachability flood (controllability analog)
//   • POMDP observation classes (observability analog)
// `steps()` returns the logical frames; the runner holds each for several
// display frames so it reads like a slideshow.
// =============================================================================

import {Frame, Shape} from '../types';
import {LinAlg, Mat} from '../../general/control-systems/linear-algebra';
import {
  MarkovDecisionProcess,
  PartiallyObservableProcess,
  StateSpaceModel,
} from '../../general/control-systems/observability-controllability';

export const OC_STAGE_W = 1000;
export const OC_STAGE_H = 640;

const COL_BG = '#0b1021';
const COL_PANEL = '#161d33';
const COL_TEXT = '#e2e8f0';
const COL_DIM = '#94a3b8';
const COL_B = '#38bdf8';
const COL_C = '#f59e0b';
const COL_OK = '#22c55e';
const COL_BAD = '#ef4444';
const COL_NODE = '#334155';
const COL_REACH = '#22c55e';

export type StoryStep = Omit<Frame, 't' | 'tick'>;

export class ObsCtrlScene {
  private readonly steps_: StoryStep[] = [];

  constructor() {
    this.buildTitle();
    this.buildLti(
      'Worked example: double integrator',
      new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}),
    );
    this.buildLti(
      'Decoupled modes (input/output touch one mode)',
      new StateSpaceModel({A: [[1, 0], [0, 2]], B: [[1], [0]], C: [[1, 0]]}),
    );
    this.buildMdp();
    this.buildPomdp();
    this.buildRecap();
  }

  steps(): readonly StoryStep[] {
    return this.steps_;
  }

  // ── shared chrome ─────────────────────────────────────────────────────────

  private base(title: string): Shape[] {
    const shapes: Shape[] = [];
    shapes.push({kind: 'rect', x: 0, y: 0, w: OC_STAGE_W, h: OC_STAGE_H, fill: COL_BG});
    shapes.push({kind: 'text', x: OC_STAGE_W / 2, y: 34, anchor: 'middle', fontSize: 21, fontWeight: 'bold',
      fill: COL_TEXT, text: 'Controllability & Observability — structural evaluator'});
    shapes.push({kind: 'text', x: OC_STAGE_W / 2, y: 62, anchor: 'middle', fontSize: 15, fill: COL_DIM, text: title});
    return shapes;
  }

  private badge(shapes: Shape[], x: number, y: number, label: string, ok: boolean): void {
    const w = 230, h = 40;
    shapes.push({kind: 'rect', x, y, w, h, rx: 8, fill: ok ? '#052e16' : '#450a0a', stroke: ok ? COL_OK : COL_BAD, strokeWidth: 2});
    shapes.push({kind: 'text', x: x + w / 2, y: y + 26, anchor: 'middle', fontSize: 15, fontWeight: 'bold',
      fill: ok ? COL_OK : COL_BAD, text: `${label}: ${ok ? 'YES' : 'NO'}`});
  }

  /** Render a labelled matrix grid; optionally tint specific column blocks. */
  private matrix(shapes: Shape[], x: number, y: number, label: string, M: Mat,
                 colColors?: Map<number, string>): {w: number; h: number} {
    const rows = LinAlg.rows(M), cols = LinAlg.cols(M);
    const cw = 46, ch = 30;
    shapes.push({kind: 'text', x, y: y - 8, anchor: 'start', fontSize: 13, fill: COL_DIM, fontWeight: 'bold', text: label});
    const w = cols * cw, h = rows * ch;
    // Brackets.
    shapes.push({kind: 'line', x1: x - 4, y1: y, x2: x - 4, y2: y + h, stroke: COL_TEXT, strokeWidth: 2});
    shapes.push({kind: 'line', x1: x - 4, y1: y, x2: x + 4, y2: y, stroke: COL_TEXT, strokeWidth: 2});
    shapes.push({kind: 'line', x1: x - 4, y1: y + h, x2: x + 4, y2: y + h, stroke: COL_TEXT, strokeWidth: 2});
    shapes.push({kind: 'line', x1: x + w + 4, y1: y, x2: x + w + 4, y2: y + h, stroke: COL_TEXT, strokeWidth: 2});
    shapes.push({kind: 'line', x1: x + w + 4, y1: y, x2: x + w - 4, y2: y, stroke: COL_TEXT, strokeWidth: 2});
    shapes.push({kind: 'line', x1: x + w + 4, y1: y + h, x2: x + w - 4, y2: y + h, stroke: COL_TEXT, strokeWidth: 2});
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const tint = colColors?.get(c);
        if (tint) shapes.push({kind: 'rect', x: x + c * cw + 2, y: y + r * ch + 2, w: cw - 4, h: ch - 4, rx: 3, fill: tint, opacity: 0.18});
        const v = M[r][c];
        shapes.push({kind: 'text', x: x + c * cw + cw / 2, y: y + r * ch + ch / 2 + 5, anchor: 'middle',
          fontSize: 14, fill: tint ?? COL_TEXT, text: this.fmt(v)});
      }
    }
    return {w, h};
  }

  private fmt(v: number): string {
    if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
    return v.toFixed(2);
  }

  // ── title ───────────────────────────────────────────────────────────────

  private buildTitle(): void {
    const shapes = this.base('three lenses on one idea');
    const lines = [
      ['Controllability', 'Can an input drive the state anywhere?', COL_B],
      ['Observability', 'Can the output reveal the full internal state?', COL_C],
    ];
    lines.forEach(([h, d, col], i) => {
      const y = 160 + i * 120;
      shapes.push({kind: 'rect', x: 120, y, w: 760, h: 92, rx: 10, fill: COL_PANEL, stroke: '#334155'});
      shapes.push({kind: 'text', x: 150, y: y + 38, anchor: 'start', fontSize: 22, fontWeight: 'bold', fill: col as string, text: h as string});
      shapes.push({kind: 'text', x: 150, y: y + 68, anchor: 'start', fontSize: 16, fill: COL_TEXT, text: d as string});
    });
    shapes.push({kind: 'text', x: OC_STAGE_W / 2, y: 470, anchor: 'middle', fontSize: 14, fill: COL_DIM,
      text: 'Linear state-space  ·  MDP (reachability)  ·  POMDP (distinguishability)'});
    this.steps_.push({shapes, caption: 'Two fundamental structural properties of dynamical systems.'});
  }

  // ── LTI ───────────────────────────────────────────────────────────────────

  private buildLti(title: string, m: StateSpaceModel): void {
    const n = m.stateDim();
    // Step 1: show A, B, C.
    {
      const shapes = this.base(title);
      this.matrix(shapes, 120, 140, 'A', m.A);
      this.matrix(shapes, 320, 140, 'B', m.B, new Map([[0, COL_B]]));
      this.matrix(shapes, 470, 140, 'C', m.C, undefined);
      shapes.push({kind: 'text', x: 120, y: 320, anchor: 'start', fontSize: 14, fill: COL_DIM,
        text: 'ẋ = A x + B u,    y = C x'});
      this.steps_.push({shapes, caption: `${title}: state matrices (n = ${n}).`});
    }
    // Step 2: controllability matrix, column by column.
    const ctrl = m.controllabilityMatrix();
    {
      const shapes = this.base(title);
      const colColors = new Map<number, string>();
      for (let c = 0; c < LinAlg.cols(ctrl); c++) colColors.set(c, COL_B);
      this.matrix(shapes, 120, 150, '𝒞 = [ B  AB  …  Aⁿ⁻¹B ]', ctrl, colColors);
      const rank = m.controllabilityRank();
      const ok = m.isControllable();
      shapes.push({kind: 'text', x: 120, y: 300, anchor: 'start', fontSize: 15, fill: COL_TEXT,
        text: `rank 𝒞 = ${rank},   n = ${n}`});
      this.badge(shapes, 120, 330, 'Controllable', ok);
      this.steps_.push({shapes, caption: `Kalman controllability: rank 𝒞 = ${rank} / ${n} → ${ok ? 'controllable' : 'NOT controllable'}.`});
    }
    // Step 3: observability matrix.
    const obs = m.observabilityMatrix();
    {
      const shapes = this.base(title);
      this.matrix(shapes, 120, 150, '𝒪 = [ C ; CA ; … ; CAⁿ⁻¹ ]', obs);
      const rank = m.observabilityRank();
      const ok = m.isObservable();
      shapes.push({kind: 'text', x: 120, y: 290, anchor: 'start', fontSize: 15, fill: COL_TEXT,
        text: `rank 𝒪 = ${rank},   n = ${n}`});
      this.badge(shapes, 120, 320, 'Observable', ok);
      this.badge(shapes, 380, 320, 'Controllable', m.isControllable());
      this.steps_.push({shapes, caption: `Kalman observability: rank 𝒪 = ${rank} / ${n} → ${ok ? 'observable' : 'NOT observable'}.`});
    }
  }

  // ── MDP ─────────────────────────────────────────────────────────────────

  private buildMdp(): void {
    const ring = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]});
    const trap = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]]});
    this.mdpStep('MDP controllability ≈ reachability — ring (s→s+1)', ring);
    this.mdpStep('MDP controllability ≈ reachability — trap (state 2 absorbing)', trap);
  }

  private mdpStep(title: string, mdp: MarkovDecisionProcess): void {
    const shapes = this.base(title);
    const cx = 320, cy = 320, r = 150;
    const adj = mdp.oneStepAdjacency();
    const reach = mdp.reachabilityClosure();
    const pos = [0, 1, 2].map(i => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / 3;
      return {x: cx + r * Math.cos(a), y: cy + r * Math.sin(a)};
    });
    // Edges with arrowheads.
    for (let s = 0; s < 3; s++) for (let t = 0; t < 3; t++) {
      if (s === t || !adj[s][t]) continue;
      const a = pos[s], b = pos[t];
      const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
      const ux = dx / len, uy = dy / len;
      const sx = a.x + ux * 34, sy = a.y + uy * 34, ex = b.x - ux * 34, ey = b.y - uy * 34;
      shapes.push({kind: 'line', x1: sx, y1: sy, x2: ex, y2: ey, stroke: COL_DIM, strokeWidth: 2});
      const ax = ex - ux * 10, ay = ey - uy * 10, px = -uy * 6, py = ux * 6;
      shapes.push({kind: 'path', d: `M ${ax + px},${ay + py} L ${ex},${ey} L ${ax - px},${ay - py}`, stroke: COL_DIM, fill: COL_DIM});
    }
    // Nodes (reachable-from-0 highlighted green).
    for (let i = 0; i < 3; i++) {
      const reachableFrom0 = reach[0][i];
      shapes.push({kind: 'circle', x: pos[i].x, y: pos[i].y, r: 30,
        fill: reachableFrom0 ? '#052e16' : COL_NODE, stroke: reachableFrom0 ? COL_REACH : '#475569', strokeWidth: 2});
      shapes.push({kind: 'text', x: pos[i].x, y: pos[i].y + 6, anchor: 'middle', fontSize: 18, fontWeight: 'bold',
        fill: COL_TEXT, text: `s${i}`});
    }
    shapes.push({kind: 'text', x: cx, y: 130, anchor: 'middle', fontSize: 13, fill: COL_REACH,
      text: 'green = reachable from s0'});
    const ok = mdp.isStructurallyControllable();
    shapes.push({kind: 'text', x: 640, y: 220, anchor: 'start', fontSize: 15, fill: COL_TEXT,
      text: `reachable ordered pairs = ${mdp.reachablePairCount()} / S² = 9`});
    this.badge(shapes, 640, 250, 'Controllable', ok);
    this.steps_.push({shapes, caption: `${title}: ${ok ? 'strongly connected → controllable' : 'cannot leave the trap → NOT controllable'}.`});
  }

  // ── POMDP ─────────────────────────────────────────────────────────────────

  private buildPomdp(): void {
    const distinct = new PartiallyObservableProcess({
      numStates: 2, numActions: 1, transition: [[[0.5, 0.5], [0.5, 0.5]]],
      numObservations: 2, observation: [[1, 0], [0, 1]],
    });
    const aliased = new PartiallyObservableProcess({
      numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]],
      numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]],
    });
    this.pomdpStep('POMDP observability ≈ distinguishability — distinct sensors', distinct);
    this.pomdpStep('POMDP observability ≈ distinguishability — aliased sensors', aliased);
  }

  private pomdpStep(title: string, pomdp: PartiallyObservableProcess): void {
    const shapes = this.base(title);
    const labels = pomdp.distinguishabilityClasses();
    const palette = ['#38bdf8', '#f59e0b', '#a78bfa', '#34d399'];
    const n = pomdp.mdp.numStates;
    for (let s = 0; s < n; s++) {
      const x = 220 + s * 280, y = 240;
      const col = palette[labels[s] % palette.length];
      shapes.push({kind: 'rect', x, y, w: 180, h: 120, rx: 12, fill: COL_PANEL, stroke: col, strokeWidth: 3});
      shapes.push({kind: 'text', x: x + 90, y: y + 34, anchor: 'middle', fontSize: 20, fontWeight: 'bold', fill: COL_TEXT, text: `state s${s}`});
      shapes.push({kind: 'text', x: x + 90, y: y + 64, anchor: 'middle', fontSize: 13, fill: COL_DIM,
        text: `obs P = [${pomdp.observation[s].map(p => p.toFixed(2)).join(', ')}]`});
      shapes.push({kind: 'text', x: x + 90, y: y + 92, anchor: 'middle', fontSize: 13, fill: col, fontWeight: 'bold',
        text: `class ${labels[s]}`});
    }
    const ok = pomdp.isStructurallyObservable();
    shapes.push({kind: 'text', x: OC_STAGE_W / 2, y: 420, anchor: 'middle', fontSize: 15, fill: COL_TEXT,
      text: `distinguishability classes = ${pomdp.classCount()} / S = ${n}`});
    this.badge(shapes, OC_STAGE_W / 2 - 115, 445, 'Observable', ok);
    const aliasing = pomdp.indistinguishablePairs();
    if (aliasing.length) {
      shapes.push({kind: 'text', x: OC_STAGE_W / 2, y: 520, anchor: 'middle', fontSize: 13, fill: COL_BAD,
        text: `aliased (indistinguishable) state pairs: ${aliasing.map(p => `(s${p[0]},s${p[1]})`).join(' ')}`});
    }
    this.steps_.push({shapes, caption: `${title}: ${ok ? 'every state has its own class → observable' : 'states collapse to one class → NOT observable'}.`});
  }

  // ── recap ──────────────────────────────────────────────────────────────────

  private buildRecap(): void {
    const shapes = this.base('summary');
    const rows: Array<[string, boolean, boolean]> = [
      ['double integrator (LTI)', true, true],
      ['decoupled modes (LTI)', false, false],
      ['ring MDP', true, true],
      ['trap MDP', false, true],
      ['distinct-sensor POMDP', true, true],
      ['aliased-sensor POMDP', true, false],
    ];
    shapes.push({kind: 'text', x: 200, y: 120, anchor: 'start', fontSize: 14, fill: COL_DIM, text: 'system'});
    shapes.push({kind: 'text', x: 620, y: 120, anchor: 'middle', fontSize: 14, fill: COL_B, text: 'controllable'});
    shapes.push({kind: 'text', x: 820, y: 120, anchor: 'middle', fontSize: 14, fill: COL_C, text: 'observable'});
    rows.forEach((r, i) => {
      const y = 150 + i * 60;
      shapes.push({kind: 'rect', x: 160, y, w: 720, h: 48, rx: 8, fill: COL_PANEL, stroke: '#334155'});
      shapes.push({kind: 'text', x: 200, y: y + 30, anchor: 'start', fontSize: 15, fill: COL_TEXT, text: r[0]});
      shapes.push({kind: 'text', x: 620, y: y + 31, anchor: 'middle', fontSize: 16, fontWeight: 'bold', fill: r[1] ? COL_OK : COL_BAD, text: r[1] ? '✓' : '✗'});
      shapes.push({kind: 'text', x: 820, y: y + 31, anchor: 'middle', fontSize: 16, fontWeight: 'bold', fill: r[2] ? COL_OK : COL_BAD, text: r[2] ? '✓' : '✗'});
    });
    this.steps_.push({shapes, caption: 'Controllability = can I move the states?   Observability = can I infer the states?'});
  }
}
