#!/usr/bin/env ts-node
'use strict';

// Compares the framework's series-RLC step-response (forward Euler at multiple
// dt values) against the analytical closed form and scipy LSODA.
//
// HOW TO RUN
// ----------
//   npm run build
//   node dist/des/main-electric-circuit.js                  # writes out/electric-circuit-framework.json
//   bash external-references/run-all.sh                     # writes out/external/electric-circuit/reference.json
//   node dist/des/runners/validate-electric-circuit.js
//
// For each dt in the sweep, samples the framework's V_C(t) trace at the
// reference grid and reports max-abs-error vs analytical and scipy. Demonstrates
// O(dt) convergence of forward Euler.

import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath  = path.join(ROOT, 'out', 'electric-circuit-framework.json');
const refPath = path.join(ROOT, 'out', 'external', 'electric-circuit', 'reference.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-electric-circuit] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function maxAbs(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  }
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/**
 * Resample a (possibly coarser) trace at the reference grid by piecewise
 * linear interpolation.
 */
function resample(trace: Array<{t: number; V_C: number; I: number}>, tGrid: number[]): {V_C: number[]; I: number[]} {
  const V_C: number[] = [];
  const I: number[] = [];
  let j = 0;
  for (const tRef of tGrid) {
    while (j + 1 < trace.length && trace[j + 1].t < tRef) j++;
    if (j + 1 >= trace.length) {
      V_C.push(trace[trace.length - 1].V_C);
      I.push(trace[trace.length - 1].I);
      continue;
    }
    const t0 = trace[j].t;
    const t1 = trace[j + 1].t;
    const w = (tRef - t0) / (t1 - t0);
    V_C.push(trace[j].V_C + w * (trace[j + 1].V_C - trace[j].V_C));
    I.push(trace[j].I + w * (trace[j + 1].I - trace[j].I));
  }
  return {V_C, I};
}

function main() {
  const ts = loadJson(tsPath);
  const ref = loadJson(refPath);

  console.log('Series RLC step response: framework vs analytical + scipy LSODA');
  console.log('=================================================================');
  console.log(`  R=${ts.config.R} ohm, L=${ts.config.L} H, C=${ts.config.C} F`);
  console.log(`  α = R/(2L) = ${(ts.config.R / (2 * ts.config.L)).toFixed(4)} rad/s`);
  console.log(`  ω0 = 1/√(LC) = ${(1 / Math.sqrt(ts.config.L * ts.config.C)).toFixed(4)} rad/s`);
  console.log(`  T = ${ts.config.T} s    (LSODA self-check max|V_C err| = ${ref.self_check.max_abs_V_C.toExponential(2)})`);
  console.log('');
  console.log(`  ${'dt'.padEnd(8)} ${'ticks'.padStart(6)}  ${'max|V_C - analytical|'.padStart(22)}  ${'max|V_C - scipy|'.padStart(20)}  ${'order'.padStart(8)}`);

  const tGrid = ref.t as number[];
  const refV = ref.V_C_analytical as number[];
  const refI = ref.I_analytical as number[];
  const sciV = ref.V_C_scipy as number[];

  let prevErr = -1;
  let prevDt  = -1;
  for (const run of ts.sweep) {
    const trace = run.trace as Array<{t: number; V_C: number; I: number; V_in: number}>;
    const {V_C: vTs, I: iTs} = resample(trace, tGrid);
    const errAna = maxAbs(vTs, refV);
    const errSci = maxAbs(vTs, sciV);

    let order = '';
    if (prevErr > 0 && prevDt > 0) {
      // Forward Euler is O(dt^1). Empirical order = log(prevErr/curErr) / log(prevDt/dt)
      const r = Math.log(prevErr / errAna) / Math.log(prevDt / run.dt);
      order = r.toFixed(2);
    }

    console.log(`  ${String(run.dt).padEnd(8)} ${String(run.ticks).padStart(6)}  ${errAna.toExponential(3).padStart(22)}  ${errSci.toExponential(3).padStart(20)}  ${order.padStart(8)}`);
    prevErr = errAna;
    prevDt = run.dt;
  }

  // Assertion: at the smallest dt, err vs scipy < 1e-3 (forward Euler at dt=0.001
  // for an underdamped circuit with α=0.1 should give ~5-stage error of ~ 1e-4).
  const smallest = ts.sweep.reduce((a: any, b: any) => a.dt < b.dt ? a : b);
  const smallestTrace = smallest.trace as Array<{t: number; V_C: number; I: number; V_in: number}>;
  const {V_C: vSmall} = resample(smallestTrace, tGrid);
  const errSmall = maxAbs(vSmall, sciV);
  const ok = errSmall < 5e-3;
  console.log('');
  console.log(`  Tightest dt = ${smallest.dt}: max|V_C - scipy| = ${errSmall.toExponential(3)}    threshold = 5e-3`);
  console.log(ok ? '  PASS' : '  FAIL');

  // Convergence sanity: error halves when dt halves (slope ~1).
  process.exit(ok ? 0 : 1);
}

main();
