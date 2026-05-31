'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/incremental_lp_scene.rs
// - Keep buildIncrementalLPFrame/buildIncrementalLPCharts as module helpers over typed LPSnapshot/LPEvent structs.
// - Numeric matrices/vectors should become Vec<Vec<f64>> initially, with a later nalgebra-style type only if shared math needs it.
// - Project/polytope helper closures can become small private structs or functions; thrown/invalid geometry paths should be Result.
// - If LP rendering becomes DES graph-visible, expose a PureTransform from snapshot/event stream to Frame.

// =============================================================================
// Incremental-LP scene: 2D polytope + objective gradient + simplex path,
// plus a tableau readout panel. The polytope re-shapes as constraints are
// added/removed; the objective arrow rotates when the obj vector changes;
// the optimum dot slides between vertices as primal/dual pivots fire.
//
// The scene supports up to 2 structural variables — beyond that the
// polytope view collapses to "tableau-only" mode.
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {LPSnapshot, LPEvent} from '../../general/incremental-lp';

export const STAGE_W = 1180;
export const STAGE_H = 700;
const POLY_X = 30,  POLY_Y = 40,  POLY_W = 600, POLY_H = 600;
const TAB_X  = 660, TAB_Y  = 40,  TAB_W  = 490, TAB_H  = 600;

// Normalisation of the polytope view: we rebuild the bounding box from the
// active constraints each frame so that as constraints are added or removed
// the visible polytope auto-fits.
const VIEW_PAD = 30;

function projectFn(xMin: number, xMax: number, yMin: number, yMax: number) {
  const sx = (POLY_W - 2 * VIEW_PAD) / Math.max(1e-9, xMax - xMin);
  const sy = (POLY_H - 2 * VIEW_PAD) / Math.max(1e-9, yMax - yMin);
  return (x: number, y: number): [number, number] => [
    POLY_X + VIEW_PAD + (x - xMin) * sx,
    POLY_Y + POLY_H - VIEW_PAD - (y - yMin) * sy,
  ];
}

/** Compute polytope vertices as the set of pairwise constraint intersections
 *  filtered down to those satisfying every constraint (and x ≥ 0). For 2D
 *  this is enough to draw the feasible region as a convex hull. */
function computePolytopeVertices(A: number[][], b: number[]): number[][] {
  const all: number[][] = [];
  // Treat x_1 ≥ 0 and x_2 ≥ 0 as additional half-planes for intersection.
  const Ax: number[][] = [...A.map(r => r.slice()), [-1, 0], [0, -1]];
  const bx: number[] = [...b, 0, 0];
  // Pairwise line intersections.
  for (let i = 0; i < Ax.length; i++) {
    for (let j = i + 1; j < Ax.length; j++) {
      const [a1, c1] = [Ax[i][0], Ax[i][1]];
      const [a2, c2] = [Ax[j][0], Ax[j][1]];
      const det = a1 * c2 - a2 * c1;
      if (Math.abs(det) < 1e-9) continue;
      const x = (bx[i] * c2 - bx[j] * c1) / det;
      const y = (a1 * bx[j] - a2 * bx[i]) / det;
      // Feasibility check.
      let ok = true;
      for (let k = 0; k < Ax.length; k++) {
        if (Ax[k][0] * x + Ax[k][1] * y > bx[k] + 1e-7) { ok = false; break; }
      }
      if (ok) all.push([x, y]);
    }
  }
  // Sort by polar angle around centroid for a clean polygon.
  if (all.length === 0) return [];
  const cx = all.reduce((s, p) => s + p[0], 0) / all.length;
  const cy = all.reduce((s, p) => s + p[1], 0) / all.length;
  all.sort((u, v) => Math.atan2(u[1] - cy, u[0] - cx) - Math.atan2(v[1] - cy, v[0] - cx));
  // Deduplicate.
  const out: number[][] = [];
  for (const p of all) {
    if (out.length === 0 || Math.hypot(p[0] - out[out.length - 1][0], p[1] - out[out.length - 1][1]) > 1e-6) out.push(p);
  }
  if (out.length > 1 && Math.hypot(out[0][0] - out[out.length - 1][0], out[0][1] - out[out.length - 1][1]) < 1e-6) out.pop();
  return out;
}

export function buildIncrementalLPFrame(
  t: number, tick: number, args: {
    snap: LPSnapshot;
    A: number[][];        // current full-form constraint matrix (m × n_struct)
    b: number[];          // current full-form rhs (length m)
    c: number[];          // current objective (length n_struct)
    sense: 'max' | 'min';
    history: number[][];  // list of past x* visited (for the simplex trail)
    eventLabel?: string;  // human-readable description of current event firing this tick
    eventFlash?: number;  // 0..1 flash intensity
    pivotLabel?: string;  // current pivot description
  },
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const {snap, A, b, c, sense, history, eventLabel, eventFlash, pivotLabel} = args;

  // ---------------- Left panel: polytope ----------------
  shapes.push({kind: 'rect', x: POLY_X, y: POLY_Y, w: POLY_W, h: POLY_H,
               fill: '#0b1220', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: POLY_X + POLY_W / 2, y: POLY_Y + 26,
               text: 'Polytope (feasible region) + simplex trajectory',
               fontSize: 14, fill: '#f1f5f9', anchor: 'middle', fontWeight: 'bold'});

  if (snap.numStruct === 2) {
    // Compute the feasible polytope.
    const verts = computePolytopeVertices(A, b);
    // Auto-bbox: include verts, the origin, and an "ideal" direction along c.
    let xMin = 0, xMax = 1, yMin = 0, yMax = 1;
    for (const v of verts) {
      if (v[0] < xMin) xMin = v[0]; if (v[0] > xMax) xMax = v[0];
      if (v[1] < yMin) yMin = v[1]; if (v[1] > yMax) yMax = v[1];
    }
    for (const x of history) {
      if (x[0] < xMin) xMin = x[0]; if (x[0] > xMax) xMax = x[0];
      if (x[1] < yMin) yMin = x[1]; if (x[1] > yMax) yMax = x[1];
    }
    if (snap.x[0] > xMax) xMax = snap.x[0]; if (snap.x[1] > yMax) yMax = snap.x[1];
    xMax = Math.max(xMax, 1); yMax = Math.max(yMax, 1);
    const padFactor = 0.08;
    xMax = xMax + (xMax - xMin) * padFactor;
    yMax = yMax + (yMax - yMin) * padFactor;
    const project = projectFn(xMin, xMax, yMin, yMax);
    // Axes.
    const [ox, oy] = project(0, 0);
    const [axEnd] = project(xMax, 0);
    const [, ayEnd] = [project(0, yMax)[0], project(0, yMax)[1]];
    shapes.push({kind: 'line', x1: ox, y1: oy, x2: axEnd, y2: oy, stroke: '#334155', strokeWidth: 1});
    shapes.push({kind: 'line', x1: ox, y1: oy, x2: ox, y2: project(0, yMax)[1], stroke: '#334155', strokeWidth: 1});
    shapes.push({kind: 'text', x: axEnd - 8, y: oy + 16, text: snap.varNames[0] ?? 'x1', fontSize: 11, fill: '#94a3b8', anchor: 'end'});
    shapes.push({kind: 'text', x: ox - 8,    y: project(0, yMax)[1] + 12, text: snap.varNames[1] ?? 'x2', fontSize: 11, fill: '#94a3b8', anchor: 'end'});
    // Polytope (filled polygon path).
    if (verts.length >= 2) {
      let d = '';
      for (let k = 0; k < verts.length; k++) {
        const [px, py] = project(verts[k][0], verts[k][1]);
        d += (k === 0 ? 'M' : 'L') + px + ',' + py + ' ';
      }
      d += 'Z';
      shapes.push({kind: 'path', d,
                   fill: '#1e293b', stroke: '#38bdf8', strokeWidth: 2, opacity: 0.8});
    }
    // Each constraint line.
    for (let i = 0; i < A.length; i++) {
      const a1 = A[i][0], a2 = A[i][1], rhs = b[i];
      // Two endpoints of the constraint line within the bbox.
      let pts: [number, number][] = [];
      if (Math.abs(a2) > 1e-9) {
        for (const xC of [xMin, xMax]) {
          const yC = (rhs - a1 * xC) / a2;
          if (yC >= yMin - 1e-3 && yC <= yMax + 1e-3) pts.push([xC, yC]);
        }
      }
      if (Math.abs(a1) > 1e-9) {
        for (const yC of [yMin, yMax]) {
          const xC = (rhs - a2 * yC) / a1;
          if (xC >= xMin - 1e-3 && xC <= xMax + 1e-3) pts.push([xC, yC]);
        }
      }
      if (pts.length >= 2) {
        const [pa, pb] = pts;
        const [x1, y1] = project(pa[0], pa[1]);
        const [x2, y2] = project(pb[0], pb[1]);
        shapes.push({kind: 'line', x1, y1, x2, y2,
                     stroke: '#38bdf8', strokeWidth: 1.5, dasharray: '4,3', opacity: 0.7});
        // Label near midpoint, slightly offset.
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        shapes.push({kind: 'text', x: mx, y: my - 4,
                     text: snap.conNames[i] ?? `c${i + 1}`,
                     fontSize: 10, fill: '#7dd3fc', anchor: 'middle'});
      }
    }
    // Objective gradient arrow at the centroid of the polytope.
    if (verts.length >= 1) {
      const cx = verts.reduce((s, p) => s + p[0], 0) / verts.length;
      const cy = verts.reduce((s, p) => s + p[1], 0) / verts.length;
      const norm = Math.hypot(c[0], c[1]) || 1;
      const dirX = c[0] / norm * (xMax - xMin) * 0.18;
      const dirY = c[1] / norm * (yMax - yMin) * 0.18;
      const [ax, ay] = project(cx, cy);
      const [bx, by] = project(cx + dirX, cy + dirY);
      shapes.push({kind: 'line', x1: ax, y1: ay, x2: bx, y2: by,
                   stroke: '#f59e0b', strokeWidth: 3});
      // Arrowhead with two short strokes.
      const dx = bx - ax, dy = by - ay; const aLen = Math.hypot(dx, dy) || 1;
      const ux = dx / aLen, uy = dy / aLen; const px = -uy, py = ux;
      const headSize = 8;
      shapes.push({kind: 'line', x1: bx, y1: by, x2: bx - ux * headSize + px * headSize / 2,
                   y2: by - uy * headSize + py * headSize / 2,
                   stroke: '#f59e0b', strokeWidth: 3});
      shapes.push({kind: 'line', x1: bx, y1: by, x2: bx - ux * headSize - px * headSize / 2,
                   y2: by - uy * headSize - py * headSize / 2,
                   stroke: '#f59e0b', strokeWidth: 3});
      shapes.push({kind: 'text', x: bx + 8, y: by - 4,
                   text: `${sense === 'max' ? 'max' : 'min'} ∇c = (${c[0]}, ${c[1]})`,
                   fontSize: 11, fill: '#fbbf24'});
    }
    // Trail of past x*'s.
    for (let k = 1; k < history.length; k++) {
      const [x1, y1] = project(history[k - 1][0], history[k - 1][1]);
      const [x2, y2] = project(history[k][0], history[k][1]);
      shapes.push({kind: 'line', x1, y1, x2, y2,
                   stroke: '#f97316', strokeWidth: 2, opacity: 0.6});
    }
    // Visited vertices.
    for (const x of history) {
      const [px, py] = project(x[0], x[1]);
      shapes.push({kind: 'circle', x: px, y: py, r: 4, fill: '#fb923c', opacity: 0.7});
    }
    // Current x*.
    const [cx, cy] = project(snap.x[0], snap.x[1]);
    shapes.push({kind: 'circle', x: cx, y: cy, r: 9,
                 fill: snap.isOptimal ? '#22c55e' : '#fbbf24',
                 stroke: '#0b1220', strokeWidth: 2});
    shapes.push({kind: 'text', x: cx + 14, y: cy + 4,
                 text: `x* = (${snap.x[0].toFixed(2)}, ${snap.x[1].toFixed(2)})  z = ${snap.z.toFixed(2)}`,
                 fontSize: 11, fill: '#f1f5f9'});
  } else {
    // 3+ structural variables: just print a notice.
    shapes.push({kind: 'text', x: POLY_X + POLY_W / 2, y: POLY_Y + POLY_H / 2,
                 text: `${snap.numStruct} structural variables — polytope view limited to 2D`,
                 fontSize: 16, fill: '#94a3b8', anchor: 'middle'});
    shapes.push({kind: 'text', x: POLY_X + POLY_W / 2, y: POLY_Y + POLY_H / 2 + 24,
                 text: 'See tableau panel for the full state', fontSize: 12, fill: '#64748b', anchor: 'middle'});
  }

  // Event flash — when a modification fires this tick, paint a coloured banner.
  if (eventLabel && (eventFlash ?? 0) > 0) {
    const alpha = Math.max(0, Math.min(1, eventFlash ?? 0));
    shapes.push({kind: 'rect', x: POLY_X + 30, y: POLY_Y + 50, w: POLY_W - 60, h: 36,
                 fill: '#dc2626', opacity: 0.55 * alpha, stroke: '#fca5a5', strokeWidth: 1, rx: 4});
    shapes.push({kind: 'text', x: POLY_X + POLY_W / 2, y: POLY_Y + 73,
                 text: `EVENT: ${eventLabel}`, fontSize: 14, fill: '#fff',
                 fontWeight: 'bold', anchor: 'middle'});
  }

  // ---------------- Right panel: tableau + status ----------------
  shapes.push({kind: 'rect', x: TAB_X, y: TAB_Y, w: TAB_W, h: TAB_H,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: TAB_X + TAB_W / 2, y: TAB_Y + 26,
               text: 'Simplex tableau', fontSize: 14, fill: '#f1f5f9',
               anchor: 'middle', fontWeight: 'bold'});

  // Status line.
  const modeColor =
    snap.mode === 'optimal'    ? '#22c55e' :
    snap.mode === 'primal'     ? '#fb923c' :
    snap.mode === 'dual'       ? '#a78bfa' :
    snap.mode === 'unbounded'  ? '#ef4444' :
    snap.mode === 'infeasible' ? '#ef4444' : '#94a3b8';
  shapes.push({kind: 'text', x: TAB_X + 14, y: TAB_Y + 56,
               text: `mode: ${snap.mode.toUpperCase()}`, fontSize: 12, fill: modeColor, fontWeight: 'bold'});
  shapes.push({kind: 'text', x: TAB_X + 14, y: TAB_Y + 76,
               text: `tick ${snap.tick}  •  z = ${snap.z.toFixed(3)}`, fontSize: 12, fill: '#cbd5e1'});
  shapes.push({kind: 'text', x: TAB_X + 14, y: TAB_Y + 94,
               text: `n=${snap.numStruct}  m=${snap.numConstraints}  basis=[${snap.basis.map(b => b < snap.numStruct ? snap.varNames[b] : (snap.conNames[b - snap.numStruct] ?? `s${b - snap.numStruct + 1}`) + '_s').join(', ')}]`,
               fontSize: 11, fill: '#94a3b8'});

  if (pivotLabel) {
    shapes.push({kind: 'text', x: TAB_X + 14, y: TAB_Y + 112,
                 text: `pivot: ${pivotLabel}`, fontSize: 11, fill: '#7dd3fc'});
  }

  // Header row.
  let yRow = TAB_Y + 142;
  shapes.push({kind: 'text', x: TAB_X + 14, y: yRow, text: 'basic', fontSize: 10, fill: '#64748b'});
  // Column headers: structural variables, then slacks, then b.
  const colNames: string[] = [];
  for (let j = 0; j < snap.numStruct; j++) colNames.push(snap.varNames[j] ?? `x${j + 1}`);
  for (let j = 0; j < snap.numConstraints; j++) colNames.push((snap.conNames[j] ?? `c${j + 1}`) + '_s');
  colNames.push('rhs');
  const colW = (TAB_W - 80) / colNames.length;
  for (let j = 0; j < colNames.length; j++) {
    shapes.push({kind: 'text', x: TAB_X + 80 + j * colW + colW / 2, y: yRow,
                 text: colNames[j], fontSize: 9, fill: '#94a3b8', anchor: 'middle'});
  }
  yRow += 14;
  // Z-row.
  shapes.push({kind: 'text', x: TAB_X + 14, y: yRow, text: 'z', fontSize: 10, fill: '#fbbf24', fontWeight: 'bold'});
  for (let j = 0; j < snap.numStruct + snap.numConstraints; j++) {
    const val = snap.reducedCosts[j] ?? 0;
    const color = val < -1e-7 ? '#fb923c' : (Math.abs(val) < 1e-9 ? '#475569' : '#cbd5e1');
    shapes.push({kind: 'text', x: TAB_X + 80 + j * colW + colW / 2, y: yRow,
                 text: val.toFixed(2), fontSize: 9, fill: color, anchor: 'middle'});
  }
  shapes.push({kind: 'text', x: TAB_X + 80 + (snap.numStruct + snap.numConstraints) * colW + colW / 2, y: yRow,
               text: snap.z.toFixed(2), fontSize: 9, fill: '#fbbf24', anchor: 'middle', fontWeight: 'bold'});
  yRow += 14;
  // Constraint rows.
  for (let i = 0; i < snap.numConstraints; i++) {
    const baseCol = snap.basis[i];
    const baseName = baseCol < snap.numStruct ?
        (snap.varNames[baseCol] ?? `x${baseCol + 1}`) :
        (snap.conNames[baseCol - snap.numStruct] ?? `c${baseCol - snap.numStruct + 1}`) + '_s';
    shapes.push({kind: 'text', x: TAB_X + 14, y: yRow,
                 text: baseName, fontSize: 10, fill: '#22d3ee', fontWeight: 'bold'});
    // Show rhs (basic value); other coefficients aren't included in snap to keep it cheap, so just show rhs.
    const rhsCol = snap.numStruct + snap.numConstraints;
    shapes.push({kind: 'text', x: TAB_X + 80 + rhsCol * colW + colW / 2, y: yRow,
                 text: snap.rhs[i].toFixed(2), fontSize: 9,
                 fill: snap.rhs[i] < -1e-7 ? '#ef4444' : '#cbd5e1', anchor: 'middle'});
    yRow += 13;
    if (yRow > TAB_Y + TAB_H - 80) break;
  }

  // Footer: feasibility flags.
  const footY = TAB_Y + TAB_H - 60;
  shapes.push({kind: 'text', x: TAB_X + 14, y: footY,
               text: `primal-feasible:  ${snap.primalFeasible ? 'YES' : 'NO'}`,
               fontSize: 11, fill: snap.primalFeasible ? '#22c55e' : '#ef4444'});
  shapes.push({kind: 'text', x: TAB_X + 14, y: footY + 18,
               text: `dual-feasible:    ${snap.dualFeasible ? 'YES' : 'NO'}`,
               fontSize: 11, fill: snap.dualFeasible ? '#22c55e' : '#ef4444'});
  shapes.push({kind: 'text', x: TAB_X + 14, y: footY + 36,
               text: `optimal:          ${snap.isOptimal ? 'YES' : 'NO'}`,
               fontSize: 11, fill: snap.isOptimal ? '#22c55e' : '#94a3b8'});

  // Caption: short summary.
  let caption = `tick ${snap.tick}  •  z = ${snap.z.toFixed(3)}  •  mode = ${snap.mode}`;
  if (eventLabel) caption += `  •  event: ${eventLabel}`;
  if (pivotLabel) caption += `  •  pivot: ${pivotLabel}`;

  return {shapes, caption};
}

/** Telemetry charts to plot underneath the main panels. */
export function buildIncrementalLPCharts(
  ticks: number[], zValues: number[], xSeries: number[][],
): ChartSpec[] {
  const series = [{label: 'z', color: '#fbbf24', t: ticks, y: zValues}];
  const xColors = ['#22d3ee', '#a78bfa', '#fb923c', '#34d399', '#f472b6', '#facc15'];
  const xCharts: ChartSpec[] = [];
  if (xSeries.length > 0) {
    const xs = xSeries[0].map((_, j) => xSeries.map(x => x[j] ?? 0));
    const xSeriesOut = xs.map((y, j) => ({label: `x${j + 1}`, color: xColors[j % xColors.length], t: ticks, y}));
    xCharts.push({x: 30, y: 660, w: 600, h: 30, title: 'x* (per structural variable)', series: xSeriesOut});
  }
  return [
    {x: 660, y: 660, w: 490, h: 30, title: 'objective z over time', series},
    ...xCharts,
  ];
}
