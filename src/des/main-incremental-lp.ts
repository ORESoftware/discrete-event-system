'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-incremental-lp.rs   (fn main)
// 1:1 file move. Incremental LP as a DES: each model edit is a movable, each
// pivot a tick; optionally animates the polytope/optimum.
//
// Conversion notes (file-specific):
//   - LPEvent / PivotEvent unions -> enum (match on kind).
//   - process.env.ANIMATE -> std::env::var.
//   - imports general/incremental-lp + animation scene -> use crate::des::...
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-incremental-lp.ts — Incremental Linear Programming as a DES.
//
// Story: a 2D production-planning LP that evolves over time as the
// business changes. Every modification is a movable arriving at the
// LPTableauStation; every pivot is a tick. The animation shows:
//   • the polytope reshape when a constraint is added/removed,
//   • the objective gradient rotate when c changes,
//   • the optimum dot slide between vertices as the simplex pivots,
//   • the tableau (basis, reduced costs, rhs) update in real time.
//
// USAGE
//   node dist/des/main-incremental-lp.js                 # default scenario, prints trace
//   ANIMATE=1 node dist/des/main-incremental-lp.js       # also writes out/incremental-lp.html
// =============================================================================

import * as path from 'path';
import {IncrementalLP, LPEvent, PivotEvent, LPSnapshot} from './general/incremental-lp';
import {FrameRecorder} from './animation/frame-recorder';
import {STAGE_W, STAGE_H, buildIncrementalLPFrame, buildIncrementalLPCharts} from './animation/scenes/incremental-lp-scene';

interface ScenarioStep {
  /** Wall-clock tick at which this event fires. */
  tick: number;
  event: LPEvent;
  description: string;
}

/** Default 2D LP scenario exercising all 5 modification types.
 *
 * Initial: max 3 x_widget + 5 x_gadget
 *          s.t.  2 x_widget + x_gadget ≤ 100   (labor)
 *                  x_widget + 3 x_gadget ≤ 90  (material)
 *                x_widget, x_gadget ≥ 0
 *
 * Modifications happen on a clock that lets the simplex settle between events.
 *
 *   tick 4:  add capacity constraint  x_widget ≤ 30                 (dual restart)
 *   tick 8:  change obj to (5, 3)  — widgets become more valuable    (primal restart)
 *   tick 12: remove the labor constraint                              (no work needed)
 *   tick 16: add a new product x_thingamajig with column [1, 1] and c=7
 *                                                                      (primal pivot to enter)
 *   tick 22: remove the material constraint                            (loosens further)
 *   tick 26: change obj to (1, 1, 8) — thingamajig dominates           (primal restart)
 */
function buildDefaultScenario(): {init: ConstructorParameters<typeof IncrementalLP>[0]; steps: ScenarioStep[]} {
  return {
    init: {
      sense: 'max',
      c: [3, 5],
      A: [[2, 1], [1, 3]],
      b: [100, 90],
      varNames: ['widget', 'gadget'],
      conNames: ['labor', 'material'],
    },
    steps: [
      {tick:  4, event: {tick:  4, kind: 'add-constraint',    coefs: [1, 0], rhs: 30, name: 'cap_widget'},
                                                              description: 'add x_widget ≤ 30'},
      {tick:  8, event: {tick:  8, kind: 'change-objective',  newC: [5, 3]},
                                                              description: 'change c → (5, 3)'},
      {tick: 12, event: {tick: 12, kind: 'remove-constraint', index: 0},
                                                              description: 'remove labor constraint'},
      {tick: 16, event: {tick: 16, kind: 'add-variable',      column: [1, 1], cNew: 7, name: 'thingamajig'},
                                                              description: 'add new product x_thingamajig (c=7, col [1,1])'},
      {tick: 22, event: {tick: 22, kind: 'remove-constraint', index: 0},
                                                              description: 'remove material constraint (LP becomes unbounded)'},
      {tick: 26, event: {tick: 26, kind: 'add-constraint',    coefs: [1, 1, 1], rhs: 50, name: 'budget'},
                                                              description: 'add budget: w+g+t ≤ 50  (re-bounds the LP)'},
      {tick: 32, event: {tick: 32, kind: 'change-objective',  newC: [1, 1, 8]},
                                                              description: 'change c → (1, 1, 8) — favour thingamajig'},
      {tick: 36, event: {tick: 36, kind: 'remove-variable',   structIndex: 1},
                                                              description: 'remove gadget (line discontinued)'},
    ],
  };
}

/** Snapshot the LP's current standard-form (A, b, c) by reading the
 *  IncrementalLP's "untransformed" data. We track these alongside
 *  modifications so the animation can rebuild the polytope each frame.
 *  We DON'T inspect the transformed tableau here — we maintain a parallel
 *  shadow record because the tableau has been pivoted. */
class StandardFormShadow {
  c: number[];
  A: number[][];
  b: number[];
  sense: 'max' | 'min';
  constructor(init: ConstructorParameters<typeof IncrementalLP>[0]) {
    this.c = init.c.slice();
    this.A = init.A.map(r => r.slice());
    this.b = init.b.slice();
    this.sense = init.sense;
  }
  apply(e: LPEvent): void {
    switch (e.kind) {
      case 'add-constraint':
        this.A.push(e.coefs.slice());
        this.b.push(e.rhs);
        break;
      case 'remove-constraint':
        this.A.splice(e.index, 1);
        this.b.splice(e.index, 1);
        break;
      case 'change-objective':
        this.c = e.newC.slice();
        break;
      case 'add-variable':
        for (let i = 0; i < this.A.length; i++) this.A[i].push(e.column[i]);
        this.c.push(e.cNew);
        break;
      case 'remove-variable':
        for (let i = 0; i < this.A.length; i++) this.A[i].splice(e.structIndex, 1);
        this.c.splice(e.structIndex, 1);
        break;
    }
  }
}

async function main(): Promise<void> {
  const scenario = buildDefaultScenario();
  const inc = new IncrementalLP(scenario.init);
  const shadow = new StandardFormShadow(scenario.init);

  console.log('# Incremental LP solver as DES — adaptive to add/remove/change events');
  console.log('# Each pivot = one tick. Each modification = one movable arriving at the tableau.');
  console.log('# Initial: max 3·widget + 5·gadget   s.t.  2w+g ≤ 100,  w+3g ≤ 90');
  console.log('');

  // Per-tick records.
  const ticks: number[] = [];
  const zValues: number[] = [];
  const xSeries: number[][] = [];
  const Asnap: number[][][] = [];
  const bsnap: number[][] = [];
  const csnap: number[][] = [];
  const sensSnap: ('max' | 'min')[] = [];
  const eventLabels: (string | undefined)[] = [];
  const eventFlashes: number[] = [];
  const pivotLabels: (string | undefined)[] = [];
  const snapshots: LPSnapshot[] = [];

  const totalTicks = Math.max(...scenario.steps.map(s => s.tick)) + 8;   // run a bit past the last event
  let history: number[][] = [];
  let lastEventTick = -10;

  // Record tick 0 (initial state, no pivot yet).
  ticks.push(0); zValues.push(inc.getZ()); xSeries.push(inc.getX().slice());
  Asnap.push(shadow.A.map(r => r.slice())); bsnap.push(shadow.b.slice());
  csnap.push(shadow.c.slice()); sensSnap.push(shadow.sense);
  eventLabels.push('initial state'); eventFlashes.push(1.0); pivotLabels.push(undefined);
  snapshots.push(inc.snapshot());
  history.push(inc.getX().slice());

  for (let tick = 1; tick <= totalTicks; tick++) {
    // 1. Apply any events scheduled for THIS tick.
    let appliedEvent: LPEvent | undefined;
    let appliedDesc: string | undefined;
    for (const s of scenario.steps) {
      if (s.tick === tick) {
        inc.applyEvent(s.event);
        shadow.apply(s.event);
        appliedEvent = s.event; appliedDesc = s.description;
        lastEventTick = tick;
        // Reset history when dimensionality changes — past trail no longer maps to the new polytope.
        if (s.event.kind === 'add-variable' || s.event.kind === 'remove-variable' ||
            s.event.kind === 'add-constraint' || s.event.kind === 'remove-constraint') {
          history = [inc.getX().slice()];
        }
      }
    }
    // 2. One pivot per tick.
    const ev = inc.step();
    let pivotLabel: string | undefined;
    if (ev.mode === 'primal' || ev.mode === 'dual') {
      pivotLabel = `${ev.mode}: ${ev.enteringName ?? '?'} enters, ${ev.leavingName ?? '?'} leaves`;
    } else if (ev.mode === 'optimal') {
      pivotLabel = 'optimal';
    } else if (ev.mode === 'idle') {
      pivotLabel = undefined;
    } else {
      pivotLabel = ev.mode;
    }
    history.push(inc.getX().slice());
    if (history.length > 12) history = history.slice(-12);
    // 3. Record.
    ticks.push(tick); zValues.push(inc.getZ()); xSeries.push(inc.getX().slice());
    Asnap.push(shadow.A.map(r => r.slice())); bsnap.push(shadow.b.slice());
    csnap.push(shadow.c.slice()); sensSnap.push(shadow.sense);
    eventLabels.push(appliedDesc);
    // Flash decays linearly over 4 ticks after an event fires.
    const ticksSinceEvent = tick - lastEventTick;
    eventFlashes.push(Math.max(0, 1 - ticksSinceEvent / 4));
    pivotLabels.push(pivotLabel);
    snapshots.push(inc.snapshot(appliedEvent, ev));
    // Console trace.
    const eventStr = appliedDesc ? `[${appliedDesc}]  ` : '';
    const pivotStr = pivotLabel ?? '';
    console.log(`tick ${String(tick).padStart(2, ' ')}  z=${inc.getZ().toFixed(3).padStart(8, ' ')}  ` +
                `x=[${inc.getX().map(v => v.toFixed(2)).join(', ')}]  ${eventStr}${pivotStr}`);
  }
  console.log('');
  console.log(`# Final: z = ${inc.getZ().toFixed(4)}, x = [${inc.getX().map(v => v.toFixed(4)).join(', ')}], status = ${inc.status}`);

  // Animation.
  if (process.env.ANIMATE === '1') {
    const outDir = path.join(__dirname, '..', '..', 'out');
    const framesPath = path.join(outDir, 'incremental-lp.frames.jsonl');
    const htmlPath   = path.join(outDir, 'incremental-lp.html');
    const rec = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H, fps: 2,
      title: 'Incremental LP solver as DES',
      subtitle: 'Each pivot is a tick; constraints/objectives/variables change live',
      background: '#020617',
    });
    for (let i = 0; i < ticks.length; i++) {
      const t = ticks[i];
      rec.frame(t, t, () => buildIncrementalLPFrame(t, t, {
        snap:        snapshots[i],
        A:           Asnap[i],
        b:           bsnap[i],
        c:           csnap[i],
        sense:       sensSnap[i],
        history:     history.slice(),
        eventLabel:  eventLabels[i],
        eventFlash:  eventFlashes[i],
        pivotLabel:  pivotLabels[i],
      }));
    }
    rec.setCharts(buildIncrementalLPCharts(ticks, zValues, xSeries));
    rec.finish();
    console.log(`# Animation written to ${htmlPath}`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
