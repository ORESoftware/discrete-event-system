'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/genetic_tsp_scene.rs
// - Keep exported STAGE constants plus buildGeneticTSPFrame/buildGeneticTSPCharts as module helpers returning Frame/ChartSpec serde structs.
// - ArchitectureFrameArgs becomes a Rust struct; TSPInstance/Tour imports should stay typed domain structs from general::genetic_tsp.
// - If these scene builders become DES graph-visible, wrap them in a PureTransform struct with transform(...) rather than leaving a bare function.
// - Local draw helpers remain private functions that push into Vec<Shape>.

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/scenes/genetic-tsp-scene.rs   (module des::animation::scenes::genetic_tsp_scene)
// 1:1 file move. Builds frames + charts for the genetic-algorithm TSP animation.
//
// Declarations → Rust:
//   const STAGE_W/H, ARCH_*/VIEW_*/META_* consts        -> `pub const`/`const`
//   const STATION_NAMES / STATION_* color consts        -> `&[&str]` / `&str`
//   function drawStation                                -> fn(&mut Vec<Shape>, ..)
//   interface ArchitectureFrameArgs                     -> struct
//   function buildGeneticTSPFrame / buildGeneticTSPCharts -> pub fns
//
// Conversion notes (file-specific):
//   - Frame data (Shape[]/Frame/ChartSpec) is serialized for JSON -> serde structs (see types.rs).
//   - `drawStation(shapes: Shape[], ..)` pushes into a shared array -> `&mut Vec<Shape>`.
//   - `STATION_NAMES` fixed table -> `const`/`static` slice.
//   - all coords are `number` -> `f64`.
// =============================================================================

// =============================================================================
// Genetic-TSP scene — REWRITTEN to put the DES architecture (the GA
// station chain) on the LEFT and analytics on the RIGHT.
//
// LEFT:  STATION GRAPH — six stations laid out as a horizontal pipeline
//          Selection ─→ Crossover ─→ Mutation ─→ Feasibility ─→ Fitness ─→ Replacement
//        Movables (chromosomes) flow along the chain. Each generation ↔
//        one DES tick. Within a tick we emit 6 sub-frames, one per phase,
//        animating chromosomes in flight between stations.
//
// RIGHT: cities + best-tour-this-generation polygon, plus a small per-
//        generation telemetry panel (best, mean, # cut by feasibility).
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {TSPInstance, Tour} from '../../general/genetic-tsp';

export const STAGE_W = 1200;
export const STAGE_H = 720;

// Architecture pipeline panel.
const ARCH_X = 20, ARCH_Y = 40, ARCH_W = 700, ARCH_H = 200;
// Cities + tour view (right panel, below the architecture).
const VIEW_X = 20, VIEW_Y = 260, VIEW_W = 700, VIEW_H = 440;
// Sidebar.
const META_X = 740, META_Y = 40, META_W = 440, META_H = 660;

const STATION_NAMES = ['Selection', 'Crossover', 'Mutation', 'Feasibility', 'Fitness', 'Replacement'];
const STATION_COLOR = '#1e293b';
const STATION_ACTIVE_FILL = '#fef3c7';
const STATION_ACTIVE_STROKE = '#f59e0b';

function drawStation(shapes: Shape[], cx: number, cy: number, w: number, h: number,
                     title: string, sub: string, active: boolean): void {
  shapes.push({kind: 'rect', x: cx - w / 2, y: cy - h / 2, w, h,
               fill: active ? STATION_ACTIVE_FILL : STATION_COLOR,
               stroke: active ? STATION_ACTIVE_STROKE : '#475569',
               strokeWidth: active ? 3 : 1.5, rx: 6});
  shapes.push({kind: 'text', x: cx, y: cy - 6, text: title,
               fontSize: 12, fill: active ? '#92400e' : '#fde68a',
               fontWeight: 'bold', anchor: 'middle'});
  if (sub) shapes.push({kind: 'text', x: cx, y: cy + 12, text: sub,
                 fontSize: 10, fill: active ? '#1f2937' : '#cbd5e1', anchor: 'middle'});
}

export interface ArchitectureFrameArgs {
  /** Generation index. */
  generation: number;
  /** Phase 0..5 — which station is currently active. */
  phase: 0 | 1 | 2 | 3 | 4 | 5;
  /** Number of feasibility-cut events this generation. */
  cutThisGen: number;
  /** Number of accepted offspring this generation. */
  acceptThisGen: number;
}

export function buildGeneticTSPFrame(t: number, tick: number, args: {
  instance: TSPInstance;
  eliteTour: Tour;
  best: number;
  mean: number;
  worst: number;
  generation: number;
  numFeasibleChildren: number;
  numInfeasibleChildren: number;
  precedenceCount: number;
  arch?: Partial<ArchitectureFrameArgs>;
}): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];
  const {instance} = args;

  // ============ ARCHITECTURE (top panel) ============
  shapes.push({kind: 'rect', x: ARCH_X, y: ARCH_Y, w: ARCH_W, h: ARCH_H,
               fill: '#0b1220', stroke: '#334155', strokeWidth: 1.5, rx: 6});
  const phase = args.arch?.phase ?? 0;
  shapes.push({kind: 'text', x: ARCH_X + ARCH_W / 2, y: ARCH_Y + 22,
    text: `GA station chain — generation ${args.generation} — phase: ${STATION_NAMES[phase]}`,
    fontSize: 13, fill: '#f1f5f9', fontWeight: 'bold', anchor: 'middle'});

  const N = STATION_NAMES.length;
  const stationW = 95, stationH = 60;
  const padding = 12;
  const totalW = N * stationW + (N - 1) * padding * 2;
  const startX = ARCH_X + (ARCH_W - totalW) / 2 + stationW / 2;
  const stationY = ARCH_Y + 100;
  const stationXs: number[] = [];
  for (let i = 0; i < N; i++) {
    const x = startX + i * (stationW + padding * 2);
    stationXs.push(x);
    const subText = i === 0 ? 'tournament' : i === 1 ? 'OX' : i === 2 ? 'inv/swap'
                  : i === 3 ? `cut ${args.numInfeasibleChildren}` : i === 4 ? 'tour len' : 'μ+λ';
    drawStation(shapes, x, stationY, stationW, stationH, STATION_NAMES[i], subText, phase === i);
  }
  // Edges with chromosomes in flight.
  for (let i = 0; i < N - 1; i++) {
    const x1 = stationXs[i] + stationW / 2;
    const x2 = stationXs[i + 1] - stationW / 2;
    const y = stationY;
    shapes.push({kind: 'line', x1, y1: y, x2, y2: y, stroke: '#64748b', strokeWidth: 1.5, opacity: 0.7});
    // Arrow tip.
    shapes.push({kind: 'line', x1: x2 - 6, y1: y - 3, x2, y2: y, stroke: '#64748b', strokeWidth: 1.5});
    shapes.push({kind: 'line', x1: x2 - 6, y1: y + 3, x2, y2: y, stroke: '#64748b', strokeWidth: 1.5});
    // Movables (chromosome dots) in transit between active phase and the next.
    const inFlight = phase === i;
    if (inFlight) {
      const numDots = i === 3 ? 3 : 5;     // feasibility lets fewer through
      const isCutEdge = i === 3;
      for (let k = 0; k < numDots; k++) {
        const tt = 0.2 + (k / numDots) * 0.6;
        const dx = x1 + (x2 - x1) * tt;
        // If this is the feasibility-cut edge and this dot is in the "cut zone",
        // render it as a red X instead of flowing through.
        const cutThis = isCutEdge && k === 0 && args.numInfeasibleChildren > 0;
        if (cutThis) {
          shapes.push({kind: 'text', x: dx, y: y + 4, text: '✗', fontSize: 14, fill: '#ef4444',
                       anchor: 'middle', fontWeight: 'bold'});
        } else {
          shapes.push({kind: 'circle', x: dx, y, r: 3, fill: '#22d3ee',
                       stroke: '#0b1220', strokeWidth: 0.5});
        }
      }
    }
  }
  // Side annotation: "cut" branch from feasibility station drops down.
  if (args.numInfeasibleChildren > 0 && phase === 3) {
    const fX = stationXs[3];
    shapes.push({kind: 'line', x1: fX, y1: stationY + stationH / 2, x2: fX, y2: stationY + stationH / 2 + 28,
                 stroke: '#ef4444', strokeWidth: 1.5, dasharray: '3 3'});
    shapes.push({kind: 'text', x: fX, y: stationY + stationH / 2 + 42,
                 text: `${args.numInfeasibleChildren} cut`,
                 fontSize: 10, fill: '#ef4444', anchor: 'middle', fontWeight: 'bold'});
  }

  // ============ TOUR / CITIES (left bottom) ============
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const [x, y] of instance.coordinates) {
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }
  const pad = 30;
  const sx = (VIEW_W - 2 * pad) / Math.max(1e-9, xMax - xMin);
  const sy = (VIEW_H - 2 * pad) / Math.max(1e-9, yMax - yMin);
  const project = (p: [number, number]): [number, number] =>
    [VIEW_X + pad + (p[0] - xMin) * sx, VIEW_Y + pad + (p[1] - yMin) * sy];

  shapes.push({kind: 'rect', x: VIEW_X, y: VIEW_Y, w: VIEW_W, h: VIEW_H,
               fill: '#0b1220', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: VIEW_X + VIEW_W / 2, y: VIEW_Y + 18,
               text: `Elite tour for generation ${args.generation} (length = ${args.best.toFixed(2)})`,
               fontSize: 13, fill: '#f1f5f9', fontWeight: 'bold', anchor: 'middle'});

  // Tour polygon.
  for (let i = 0; i < instance.n; i++) {
    const [x1, y1] = project(instance.coordinates[args.eliteTour[i]]);
    const [x2, y2] = project(instance.coordinates[args.eliteTour[(i + 1) % instance.n]]);
    shapes.push({kind: 'line', x1, y1, x2, y2, stroke: '#22d3ee', strokeWidth: 2, opacity: 0.9});
  }
  // Cities.
  for (let i = 0; i < instance.n; i++) {
    const [x, y] = project(instance.coordinates[i]);
    shapes.push({kind: 'circle', x, y, r: 6, fill: '#fde68a', stroke: '#f59e0b', strokeWidth: 1});
    shapes.push({kind: 'text', x: x + 9, y: y + 4, text: i.toString(),
                 fontSize: 10, fill: '#cbd5e1'});
  }
  // Precedence arcs.
  if (instance.precedence) {
    for (const [a, b] of instance.precedence) {
      const [x1, y1] = project(instance.coordinates[a]);
      const [x2, y2] = project(instance.coordinates[b]);
      shapes.push({kind: 'line', x1, y1, x2, y2, stroke: '#ef4444',
                   strokeWidth: 1, dasharray: '3,3', opacity: 0.5});
    }
  }

  // ============ SIDEBAR ============
  shapes.push({kind: 'rect', x: META_X, y: META_Y, w: META_W, h: META_H,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 6});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 32,
               text: 'Genetic-TSP', fontSize: 22, fill: '#f1f5f9',
               anchor: 'middle', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 60,
               text: `Generation ${args.generation}`, fontSize: 14,
               fill: '#cbd5e1', anchor: 'middle'});
  shapes.push({kind: 'text', x: META_X + META_W / 2, y: META_Y + 78,
               text: `Phase: ${STATION_NAMES[phase]}`, fontSize: 12,
               fill: '#fde68a', anchor: 'middle'});
  let y = META_Y + 110;
  const line = (text: string, color: string) => {
    shapes.push({kind: 'text', x: META_X + 20, y, text, fontSize: 12, fill: color});
    y += 22;
  };
  line(`best  tour length = ${args.best.toFixed(2)}`, '#22d3ee');
  line(`mean  tour length = ${args.mean.toFixed(2)}`, '#94a3b8');
  line(`worst tour length = ${args.worst.toFixed(2)}`, '#ef4444');
  y += 8;
  line(`# feasible kids   = ${args.numFeasibleChildren}`, '#22c55e');
  line(`# cut (infeasible)= ${args.numInfeasibleChildren}`, '#facc15');
  y += 8;
  line(`# cities            = ${instance.n}`, '#cbd5e1');
  line(`# precedence pairs  = ${args.precedenceCount}`, '#cbd5e1');
  y += 8;
  // Architecture legend.
  line(`Architecture legend:`, '#f1f5f9');
  shapes.push({kind: 'circle', x: META_X + 30, y: y - 4, r: 4, fill: '#22d3ee'});
  shapes.push({kind: 'text', x: META_X + 42, y, text: 'chromosome (movable)', fontSize: 11, fill: '#cbd5e1'});
  y += 18;
  shapes.push({kind: 'text', x: META_X + 30, y: y - 1, text: '✗', fontSize: 12, fill: '#ef4444', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: META_X + 42, y, text: 'cut by feasibility station', fontSize: 11, fill: '#cbd5e1'});

  return {
    shapes,
    caption: `gen=${args.generation} phase=${STATION_NAMES[phase]} best=${args.best.toFixed(2)}`,
  };
}

export function buildGeneticTSPCharts(
  generations: number[], best: number[], mean: number[],
): ChartSpec[] {
  return [
    {x: META_X, y: META_Y + 380, w: META_W, h: 280,
     title: 'Best & mean tour length per generation',
     series: [
       {label: 'best', color: '#22d3ee', t: generations, y: best},
       {label: 'mean', color: '#94a3b8', t: generations, y: mean},
     ]},
  ];
}
