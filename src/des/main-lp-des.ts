'use strict';

// =============================================================================
// main-lp-des.ts — solve an LP using the DES engine.
//
// THE PATTERN
// ───────────
// Simplex is naturally a discrete-event process: each PIVOT is an event
// where the algorithm walks one edge of the feasible polytope to the
// next vertex. The DES engine drives the walk; LP-specific roles are
// four station classes:
//
//     EnteringStation  → "find the steepest improving direction"
//                        (Dantzig: most-negative reduced cost; or
//                         Bland: first negative reduced cost)
//     LeavingStation   → "min-ratio test: how far along that direction
//                         can we travel before some basic var hits 0?"
//     PivotStation     → "elementary row operations: update tableau"
//     ObserverStation  → "snapshot the new vertex + objective into trace"
//
// One pivot = one tick. Termination when EnteringStation finds no
// improving direction (optimality), no positive ratio is available
// (unbounded), or we hit the iteration cap. Two phases of simplex
// (phase-1 for feasibility, phase-2 for optimality) share the same
// DES loop with a different cost row.
//
// IS THIS EFFICIENT?
// ──────────────────
// No — a 60-line direct simplex would run 5–10× faster because the
// per-tick scaffolding (status checks, history captures, phase
// transitions) is overhead. We're doing this to:
//
//   (1) prove the DES engine is computationally general — events,
//       queues, stations, and movables suffice for vertex-walking
//       optimisation, even when LP geometry would normally invite a
//       specialised algorithm;
//   (2) get the trace + animation infrastructure for free — the same
//       FrameRecorder / HtmlPlayer used for SEIR + FactMachine + heat
//       PDE works directly on the per-pivot vertex history;
//   (3) demonstrate that DES + simplex can be HOSTED in the same
//       runtime, so the bridge to LP-as-MDP-as-DES (see
//       main-mdp-lp.ts) flows through ONE engine.
//
// USAGE
// ─────
//   node dist/des/main-lp-des.js                              # default 2-var demo
//   node dist/des/main-lp-des.js                              # show pivot trace
//   PIVOT_RULE=bland   node dist/des/main-lp-des.js
//   PROBLEM=diet       node dist/des/main-lp-des.js
//   PROBLEM=transport  node dist/des/main-lp-des.js
//   PROBLEM=2var ANIMATE=1  node dist/des/main-lp-des.js      # post-hoc animated polytope walk
//
// Validation: see runners/validate-lp.ts (Studies 8, 9). DES simplex
// agrees with scipy:HiGHS to machine epsilon on canonical LPs and on
// 50 random feasible LPs.
// =============================================================================

import * as path from 'path';
import * as fs from 'fs';
import {LPProblem, lpToString, solveLPExternal, solveLPInternal} from './general/lp';
import {solveLPViaDES, DESSimplexSolution} from './general/lp-des';

// -----------------------------------------------------------------------------
// Library of canonical example LPs.
// -----------------------------------------------------------------------------
const PROBLEMS: Record<string, LPProblem> = {
  '2var': {
    sense: 'max', c: [3, 2],
    A_ub: [[1, 1], [1, 3]], b_ub: [4, 6],
    varNames: ['x', 'y'],
  },
  '2var-diamond': {
    // Slightly more interesting 2-D polytope with 5 vertices to walk.
    sense: 'max', c: [2, 3],
    A_ub: [[1, 0], [0, 1], [1, 1], [1, 2]],
    b_ub: [4, 5, 6, 9],
    varNames: ['x', 'y'],
  },
  'diet': {
    sense: 'min', c: [0.5, 0.3, 0.7, 0.2],
    A_ub: [[-2, -3, -1, -4], [-1, -2, -3, -1], [-3, -1, -2, 0]],
    b_ub: [-12, -6, -4],
    varNames: ['bread', 'cheese', 'meat', 'rice'],
    conNames: ['protein', 'vit-A', 'vit-C'],
  },
  'transport': {
    sense: 'min',
    c: [4, 6, 8, 3, 5, 7, 9, 2, 1],
    A_eq: [
      [1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 1],
      [1, 0, 0, 1, 0, 0, 1, 0, 0],
      [0, 1, 0, 0, 1, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1],
    ],
    b_eq: [20, 30, 25, 25, 25, 25],
    varNames: ['x11', 'x12', 'x13', 'x21', 'x22', 'x23', 'x31', 'x32', 'x33'],
  },
};

async function main(): Promise<void> {
  const which = (process.env.PROBLEM ?? '2var').trim();
  if (!PROBLEMS[which]) {
    console.error(`unknown PROBLEM='${which}'; expected one of: ${Object.keys(PROBLEMS).join(', ')}`);
    process.exit(2);
  }
  const lp = PROBLEMS[which];
  const pivotRule = (process.env.PIVOT_RULE ?? 'dantzig') as 'dantzig' | 'bland';

  console.log(`# DES-driven simplex on '${which}' problem  (pivotRule = ${pivotRule})`);
  console.log('#');
  console.log('# LP:');
  for (const line of lpToString(lp).split('\n')) console.log('#   ' + line);
  console.log('');

  // Solve three ways: DES simplex, in-process simplex, scipy:HiGHS.
  const des = solveLPViaDES(lp, {pivotRule, maxIter: 1000});
  const internal = solveLPInternal(lp);
  const external = solveLPExternal(lp, {method: 'highs'});

  console.log(`# DES simplex (this engine):`);
  console.log(`#   status     = ${des.status}`);
  console.log(`#   pivots     = ${des.trace.pivotHistory.length}`);
  console.log(`#   x*         = [ ${des.x.map(v => v.toFixed(6)).join(', ')} ]`);
  console.log(`#   objective  = ${des.objective.toFixed(8)}`);
  console.log(`#   wall time  = ${des.elapsedMs}ms`);
  console.log('');

  console.log(`# In-process simplex (textbook two-phase, NOT through DES):`);
  console.log(`#   status     = ${internal.status}    obj = ${internal.objective?.toFixed(8) ?? '-'}    iters = ${internal.iters}    Δ = ${Math.abs(des.objective - internal.objective).toExponential(3)}`);

  console.log(`# scipy:HiGHS (external simplex):`);
  console.log(`#   status     = ${external.status}    obj = ${external.objective?.toFixed(8) ?? '-'}    iters = ${external.iters}    Δ = ${Math.abs(des.objective - external.objective).toExponential(3)}`);
  if (external.dualUB && external.dualUB.length) {
    console.log(`#   shadow prices on each ≤ constraint = [ ${external.dualUB.map(v => v.toFixed(4)).join(', ')} ]`);
  }
  console.log('');

  console.log(`# Pivot trajectory (each row = one tick / one vertex visit):`);
  console.log(`#   tick  phase   enter   leave    pivot         vertex (x*)`);
  // Show initial vertex (no pivot yet) + each pivot's resulting vertex.
  console.log(`#   ${'init'.padEnd(6)} ${''.padEnd(7)} ${''.padEnd(7)} ${''.padEnd(7)} ${''.padEnd(13)} [ ${des.trace.vertexHistory[0].map(v => v.toFixed(3)).join(', ')} ]    obj = ${des.trace.objHistory[0].toFixed(4)}`);
  for (let i = 0; i < des.trace.pivotHistory.length; i++) {
    const p = des.trace.pivotHistory[i];
    const v = des.trace.vertexHistory[i + 1];
    const objStr = isNaN(p.obj) ? '(phase-1)' : p.obj.toFixed(4);
    console.log(`#   ${p.tick.toString().padEnd(6)} ${p.phase.toString().padEnd(7)} ${('col=' + p.enter).padEnd(7)} ${('row=' + p.leave).padEnd(7)} ${p.pivotElt.toExponential(3).padEnd(13)} [ ${v.map(v => v.toFixed(3)).join(', ')} ]    obj = ${objStr}`);
  }
  console.log('');

  // -- Post-hoc animation for 2-D LPs --
  if (process.env.ANIMATE === '1' && lp.c.length === 2) {
    await renderPolytopeAnimation(lp, des);
  } else if (process.env.ANIMATE === '1') {
    console.log(`# (animation skipped: only 2-D LPs render visibly; ${lp.c.length}-D LP just shows objective time-series)`);
    await renderObjectiveAnimation(lp, des);
  }
}

// -----------------------------------------------------------------------------
// 2-D polytope walk animation.
// -----------------------------------------------------------------------------
async function renderPolytopeAnimation(lp: LPProblem, des: DESSimplexSolution): Promise<void> {
  const {FrameRecorder} = await import('./animation/frame-recorder');
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const W = 800, H = 600;
  const margin = 60;
  const A_ub = lp.A_ub ?? [];
  const b_ub = lp.b_ub ?? [];
  // Bounding box: at most 1.4 × the largest vertex coordinate.
  const allX = des.trace.vertexHistory.flatMap(v => v);
  const maxCoord = Math.max(1.0, ...allX, ...b_ub);
  const xMax = maxCoord * 1.2, yMax = maxCoord * 1.2;
  const sx = (x: number) => margin + (x / xMax) * (W - 2 * margin);
  const sy = (y: number) => H - margin - (y / yMax) * (H - 2 * margin);

  // Sample feasible region by checking grid points (cheap visualisation).
  const feasiblePts: Array<{x: number; y: number}> = [];
  const Nsamp = 40;
  for (let ii = 0; ii <= Nsamp; ii++) for (let jj = 0; jj <= Nsamp; jj++) {
    const xv = (ii / Nsamp) * xMax;
    const yv = (jj / Nsamp) * yMax;
    let ok = true;
    for (let r = 0; r < A_ub.length; r++) {
      if (A_ub[r][0] * xv + A_ub[r][1] * yv > b_ub[r] + 1e-9) { ok = false; break; }
    }
    if (ok) feasiblePts.push({x: xv, y: yv});
  }

  const html = path.join(outDir, 'lp-des.html');
  const rec = new (FrameRecorder as any)({
    framesPath: path.join(outDir, 'lp-des.frames.jsonl'),
    htmlPath: html,
    width: W, height: H, fps: 1,
    title: `LP-as-DES — pivot walk on the feasible polytope`,
    subtitle: `Each tick = one simplex pivot. EnteringStation picks the steepest improving edge; LeavingStation picks how far to travel; PivotStation updates the tableau.`,
  });

  for (let i = 0; i < des.trace.vertexHistory.length; i++) {
    const v = des.trace.vertexHistory[i];
    const obj = des.trace.objHistory[i];
    const objStr = isNaN(obj) ? '(phase-1)' : obj.toFixed(4);
    const t = i;
    const visited = des.trace.vertexHistory.slice(0, i + 1);
    rec.frame(t, t, () => {
      const shapes: any[] = [];
      // Axes.
      shapes.push({type: 'line', x1: margin, y1: H - margin, x2: W - margin, y2: H - margin, stroke: '#888', sw: 2});
      shapes.push({type: 'line', x1: margin, y1: H - margin, x2: margin, y2: margin, stroke: '#888', sw: 2});
      shapes.push({type: 'text', x: W / 2, y: H - 20, fill: '#444', fontSize: 14, text: lp.varNames?.[0] ?? 'x', anchor: 'middle'});
      shapes.push({type: 'text', x: 20, y: H / 2, fill: '#444', fontSize: 14, text: lp.varNames?.[1] ?? 'y', anchor: 'middle'});
      // Feasible region (grid samples).
      for (const p of feasiblePts) {
        shapes.push({type: 'circle', cx: sx(p.x), cy: sy(p.y), r: 2, fill: '#cfe2ff'});
      }
      // Path so far.
      for (let j = 1; j < visited.length; j++) {
        shapes.push({type: 'line',
                     x1: sx(visited[j - 1][0]), y1: sy(visited[j - 1][1]),
                     x2: sx(visited[j][0]), y2: sy(visited[j][1]),
                     stroke: '#0a58ca', sw: 3});
      }
      // Visited vertices.
      for (let j = 0; j < visited.length - 1; j++) {
        shapes.push({type: 'circle', cx: sx(visited[j][0]), cy: sy(visited[j][1]), r: 5, fill: '#0a58ca'});
      }
      // Current vertex.
      shapes.push({type: 'circle', cx: sx(v[0]), cy: sy(v[1]), r: 9, fill: '#dc3545', stroke: '#222', sw: 2});
      shapes.push({type: 'text', x: sx(v[0]) + 12, y: sy(v[1]) - 6, fill: '#222', fontSize: 13,
                   text: `(${v[0].toFixed(2)}, ${v[1].toFixed(2)})  obj=${objStr}`});
      // Tick label.
      shapes.push({type: 'text', x: W - margin, y: margin, fill: '#333', fontSize: 14,
                   text: `pivot ${t}/${des.trace.vertexHistory.length - 1}`, anchor: 'end'});
      return shapes;
    });
  }
  rec.setCharts([{
    title: 'objective vs pivot count',
    series: [{
      label: 'obj',
      data: des.trace.objHistory.map((v, i) => ({x: i, y: isNaN(v) ? 0 : v})),
      stroke: '#0a58ca',
    }],
  }]);
  await rec.finish();
  console.log(`# animation written to out/lp-des.html`);
}

// -----------------------------------------------------------------------------
// Higher-D fallback: just the objective time series.
// -----------------------------------------------------------------------------
async function renderObjectiveAnimation(_lp: LPProblem, des: DESSimplexSolution): Promise<void> {
  const {FrameRecorder} = await import('./animation/frame-recorder');
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const W = 700, H = 360;
  const html = path.join(outDir, 'lp-des.html');
  const rec = new (FrameRecorder as any)({
    framesPath: path.join(outDir, 'lp-des.frames.jsonl'),
    htmlPath: html, width: W, height: H, fps: 1,
    title: 'LP-as-DES — pivot trajectory',
    subtitle: 'Each tick = one simplex pivot.',
  });
  for (let i = 0; i < des.trace.objHistory.length; i++) {
    const t = i;
    rec.frame(t, t, () => [
      {type: 'text', x: W / 2, y: H / 2, anchor: 'middle', fill: '#222', fontSize: 22,
       text: `pivot ${i}  obj=${isNaN(des.trace.objHistory[i]) ? '(phase-1)' : des.trace.objHistory[i].toFixed(6)}`},
    ]);
  }
  rec.setCharts([{
    title: 'objective vs pivot count',
    series: [{
      label: 'obj',
      data: des.trace.objHistory.map((v, i) => ({x: i, y: isNaN(v) ? 0 : v})),
      stroke: '#0a58ca',
    }],
  }]);
  await rec.finish();
  console.log(`# animation written to out/lp-des.html`);
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
