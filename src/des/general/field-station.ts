'use strict';

// =============================================================================
// FieldStation — the framework substrate for ODE / PDE problems.
//
// CORE IDEA
// ---------
// Each ODE/PDE problem is mapped to a set of stationary entities ("field
// stations"), each holding a scalar value. A central CENSUS station
// snapshots every value at the start of every tick. Each field station
// then runs `runTimeStep`, reading only the snapshot — never another
// station's mid-tick state. This is the same synchronous-data-flow
// pattern used by main-two-disease.ts, generalised to:
//
//   * 0-D scalar systems (single variable y, e.g. y' = −y)
//   * 1-D systems of state variables (ODE system y_1', …, y_n')
//   * 1-D PDE on a spatial grid u(x, t) (heat, wave, advection)
//   * 2-D PDE on a NxN grid u(x, y, t) (Poisson, Laplace, 2-D heat)
//
// The "moving entity" interpretation is loose for ODEs (values don't
// physically move; the equation's coupling determines who reads whom).
// For diffusion-type PDEs it can be made literal — a particle / Walk-
// on-Spheres simulator is a separate path that DOES use movables.
//
// RELATIONSHIP TO THE PURE-MATH SOLVERS
// -------------------------------------
// `general/ode.ts` and (later) `general/pde.ts` provide the same
// numerical methods as plain functions of (f, y0, t0, t1, …). The
// station-based solver IS the same finite-difference/Runge-Kutta
// recipe, just structured as a network of stations. The validators
// pin the station output ≡ pure-math output to f64 precision.
// =============================================================================

import {mulberry32} from './prng';
import {TimeSteppedStation} from './time-stepped-station';

function shuffleInPlace<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// -----------------------------------------------------------------------------
// Base classes.
// -----------------------------------------------------------------------------

export abstract class Station extends TimeSteppedStation {}

/**
 * Census station: snapshots every field station's `value` at the start
 * of every tick. All field stations read from `snap` (a frozen array)
 * during their own `runTimeStep`, so the order in which field stations
 * run within a single tick does not matter — exactly the property that
 * makes finite-difference schemes order-independent.
 */
export class Census extends Station {
  snap: Float64Array;
  prevSnap: Float64Array;   // value snapshot from the PREVIOUS tick (for leapfrog/wave)
  fields: FieldStation[];
  constructor(id: string, fields: FieldStation[]) {
    super(id);
    this.fields = fields;
    this.snap = new Float64Array(fields.length);
    this.prevSnap = new Float64Array(fields.length);
  }
  runTimeStep(_dt: number, _t: number): void {
    // Promote current snapshot to prev (used by leapfrog/wave equation).
    this.prevSnap.set(this.snap);
    for (let i = 0; i < this.fields.length; i++) this.snap[i] = this.fields[i].value;
  }
}

/**
 * Generic field station: holds a scalar, exposes `value`, and applies
 * an `updater` function each tick that reads the census snapshot and
 * returns the new value.
 *
 * The updater receives:
 *   prev  — Float64Array of values one tick ago (for leapfrog)
 *   cur   — Float64Array of values at THIS tick start
 *   self  — index into cur of this station
 *   dt, t — time step and elapsed time
 */
export type FieldUpdater = (
  prev: Float64Array, cur: Float64Array, self: number, dt: number, t: number,
) => number;

export class FieldStation extends Station {
  /** Index into the census's snap array. Set by FieldSimulation. */
  index = -1;
  /** Optional spatial position (used by 1-D / 2-D PDE schemes). */
  position?: number | [number, number];
  constructor(id: string, public value: number, public updater: FieldUpdater,
              public census: Census) {
    super(id);
  }
  runTimeStep(dt: number, t: number): void {
    this.value = this.updater(this.census.prevSnap, this.census.snap, this.index, dt, t);
  }
}

// -----------------------------------------------------------------------------
// FieldSimulation — drives the tick loop.
// -----------------------------------------------------------------------------

export interface FieldSimulationOptions {
  /** Random seed for the per-tick processing-order shuffle. Default 1. */
  seed?: number;
  /**
   * If false, do NOT shuffle the field stations' processing order each tick.
   * Field stations should be order-independent by construction (because
   * they read only `census.snap`), so shuffling is not strictly required —
   * but the test of "still gives the same answer under any order" is a
   * good sanity check, so we ship the shuffle by default.
   */
  shuffle?: boolean;
  /** If true, record the full snapshot at each tick into trace. Default true. */
  recordTrace?: boolean;
}

export interface FieldSimulationResult {
  trace: {t: number[]; values: Float64Array[]};
  finalValues: Float64Array;
  ticks: number;
}

export class FieldSimulation {
  fields: FieldStation[];
  census: Census;
  rng: () => number;
  shuffle: boolean;
  recordTrace: boolean;
  constructor(fields: FieldStation[], opts: FieldSimulationOptions = {}) {
    this.fields = fields;
    this.census = new Census('census', fields);
    for (let i = 0; i < fields.length; i++) {
      fields[i].index = i;
      fields[i].census = this.census;
    }
    // Initialise both snap and prevSnap to the starting values so leapfrog
    // schemes have a sane "u(t = −dt)" reading on tick 0.
    for (let i = 0; i < fields.length; i++) {
      this.census.snap[i] = fields[i].value;
      this.census.prevSnap[i] = fields[i].value;
    }
    this.rng = mulberry32(opts.seed ?? 1);
    this.shuffle = opts.shuffle ?? true;
    this.recordTrace = opts.recordTrace ?? true;
  }
  run(t0: number, t1: number, dt: number): FieldSimulationResult {
    const t: number[] = [];
    const values: Float64Array[] = [];
    if (this.recordTrace) {
      t.push(t0);
      values.push(new Float64Array(this.census.snap));
    }
    let tn = t0;
    let tick = 0;
    while (tn + 0.5 * dt < t1) {
      this.census.runTimeStep(dt, tn);
      const order = this.fields.slice();
      if (this.shuffle) shuffleInPlace(order, this.rng);
      for (const f of order) f.runTimeStep(dt, tn);
      tn += dt;
      tick++;
      if (this.recordTrace) {
        t.push(tn);
        values.push(new Float64Array(this.fields.length));
        for (let i = 0; i < this.fields.length; i++) values[values.length - 1][i] = this.fields[i].value;
      }
    }
    return {
      trace: {t, values},
      finalValues: new Float64Array(this.fields.map(f => f.value)),
      ticks: tick,
    };
  }
}
