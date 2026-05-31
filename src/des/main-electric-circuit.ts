#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-electric-circuit.rs   (fn main)
// 1:1 file move. Series RLC step-response as a DES (the tick clock is the
// numerical integrator) via VoltageSource/Inductor/Capacitor stations.
//
// Conversion notes (file-specific):
//   - station classes -> struct + impl trait.
//   - the `pending`-emission map flushed after all stations run -> a HashMap
//     then swap (two-phase update preserves the frozen-inbox semantics).
//   - integrator scheme (Euler/RK4) -> enum; top-level run -> fn main.
// =============================================================================

// =============================================================================
// Series RLC step-response as a discrete-event system. The circuit is a
// genuine continuous-time physical system; the DES tick clock is the
// numerical integrator (Euler / RK4 etc.) — illustrating that the same
// "stations + movables" substrate can simulate continuous physics.
//
//   V_step                R                 L                C
//     ┌──[VoltageSource]──[Resistor]──┬──[Inductor]──┬──[Capacitor]──┐
//     │                               │              │               │
//     └───────────────────────────────┴──────────────┴───────────────┘
//
// Three stationary entities communicate via synchronous data flow:
//
//   * VoltageSource emits V_in (a Heaviside step at t=0).
//   * Inductor holds state I (current).        dI/dt  = (V_in - I*R - V_C) / L
//   * Capacitor holds state V_C (voltage).     dV_C/dt = I / C
//
// Each tick:
//   1. Every station reads its inbox (FROZEN at start of tick — all values
//      are from the previous tick's emissions).
//   2. Every station runs runTimeStep, computing its new state via forward
//      Euler and writing emissions to a `pending` map.
//   3. After all stations have run, `pending` is flushed into target inboxes
//      to be visible NEXT tick.
//
// This synchronous data-flow semantic makes the simulation order-independent
// (Fisher–Yates of station order doesn't change a single value), and turns
// the chain into textbook forward Euler with one tick of latency on every
// edge.
//
// Validation:
//   * Analytical underdamped step response (closed-form).
//   * scipy.integrate.solve_ivp with LSODA (adaptive 1e-10 tolerance) for a
//     reference numerical solution.
//
// Forward Euler will agree with both at small dt and diverge at large dt.
// `external-references/electric-circuit/circuit.py` produces the analytical
// + scipy traces; `validate-electric-circuit.ts` reports max-abs-error for a
// sweep of dt values to demonstrate convergence.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {fisherYatesShuffle} from './general/general';
import {SynchronousDataflowStation as Station} from './general/time-stepped-station';

class VoltageSource extends Station {
  constructor(id: string, public Vstep: number, public t0: number = 0) { super(id); }
  runTimeStep(_stepSize: number, t: number): void {
    const V = t >= this.t0 ? this.Vstep : 0;
    this.emit('V_in', V);
  }
}

class Inductor extends Station {
  I = 0;
  constructor(id: string, public L: number, public R: number) { super(id); }
  runTimeStep(stepSize: number, _t: number): void {
    const V_in = this.inbox.get('V_in') ?? 0;
    const V_C  = this.inbox.get('V_C')  ?? 0;
    // dI/dt = (V_in - I*R - V_C) / L     (Kirchhoff's voltage law).
    this.I = this.I + stepSize * (V_in - this.I * this.R - V_C) / this.L;
    this.emit('I', this.I);
  }
}

class Capacitor extends Station {
  V_C = 0;
  constructor(id: string, public C: number) { super(id); }
  runTimeStep(stepSize: number, _t: number): void {
    const I = this.inbox.get('I') ?? 0;
    // dV_C/dt = I / C
    this.V_C = this.V_C + stepSize * I / this.C;
    this.emit('V_C', this.V_C);
  }
}

class Recorder extends Station {
  trace: Array<{t: number; I: number; V_C: number; V_in: number}> = [];
  constructor(id: string, public inductor: Inductor, public capacitor: Capacitor) { super(id); }
  runTimeStep(stepSize: number, t: number): void {
    const V_in = this.inbox.get('V_in') ?? 0;
    this.trace.push({t: t * stepSize, I: this.inductor.I, V_C: this.capacitor.V_C, V_in});
  }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

export interface RLCConfig {
  R: number;     // resistance, ohms
  L: number;     // inductance, henrys
  C: number;     // capacitance, farads
  Vstep: number; // step amplitude, volts
  T: number;     // total simulated time, seconds
  dt: number;    // forward-Euler timestep, seconds
}

export interface RLCResult {
  config: RLCConfig;
  ticks: number;
  trace: Array<{t: number; I: number; V_C: number; V_in: number}>;
}

export function runRLC(cfg: RLCConfig): RLCResult {
  const src = new VoltageSource('src', cfg.Vstep);
  const ind = new Inductor('L', cfg.L, cfg.R);
  const cap = new Capacitor('C', cfg.C);
  const rec = new Recorder('rec', ind, cap);

  src.addOut('V_in', ind);
  src.addOut('V_in', rec);
  ind.addOut('I', cap);
  cap.addOut('V_C', ind);

  const stations: Station[] = [src, ind, cap, rec];

  const N = Math.round(cfg.T / cfg.dt);
  for (let t = 0; t < N; t++) {
    // Phase 1: shuffle and run runTimeStep on every station.
    const order = [...stations];
    for (const _ of fisherYatesShuffle(order)) { /* generator side-effect */ }
    for (const s of order) s.runTimeStep(cfg.dt, t);
    // Phase 2: deliver pending emissions to inboxes for NEXT tick.
    for (const s of stations) s.commit();
  }

  return {config: cfg, ticks: N, trace: rec.trace};
}

// -----------------------------------------------------------------------------
// CLI: write a sweep of dt values for validation.
// -----------------------------------------------------------------------------

function defaultConfig(dt: number): RLCConfig {
  // Underdamped: ω0 = 1/√(LC), α = R/(2L). Pick R, L, C so ω0 ≈ 1 rad/s,
  // α ≈ 0.1 (mild damping). Then natural ringing has period ~2π and decays
  // over ~10 periods.
  return {
    R: 0.2,         // ohms
    L: 1.0,         // henrys
    C: 1.0,         // farads
    Vstep: 1.0,     // volts
    T: 30.0,        // seconds
    dt,
  };
}

function main() {
  const dts = (process.env.DTS ?? '0.5,0.1,0.05,0.01,0.005,0.001').split(',').map(Number);
  const T = Number(process.env.T ?? 30);

  console.log('# Series RLC step response sweep');
  console.log(`#   R=0.2 ohm, L=1 H, C=1 F, V_step=1 V`);
  console.log(`#   ω0 = 1 rad/s, α = R/(2L) = 0.1, period = 2π s`);
  console.log(`#   T = ${T} s`);

  const out: any = {sweep: []};
  for (const dt of dts) {
    const cfg = defaultConfig(dt);
    cfg.T = T;
    const result = runRLC(cfg);
    const last = result.trace[result.trace.length - 1];
    console.log(`  dt=${dt.toString().padEnd(8)}  ticks=${result.ticks.toString().padStart(6)}  V_C(${last.t.toFixed(3)})=${last.V_C.toFixed(6)}  I=${last.I.toExponential(2)}`);
    out.sweep.push({
      dt,
      ticks: result.ticks,
      trace: result.trace,
      final_V_C: last.V_C,
      final_I: last.I,
    });
  }
  out.config = {R: 0.2, L: 1.0, C: 1.0, Vstep: 1.0, T};

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'electric-circuit-framework.json');
  fs.writeFileSync(outPath, JSON.stringify(out));
  console.log(`# wrote ${outPath}`);
}

if (require.main === module) main();
