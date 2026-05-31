#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_two_disease.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert framework/Python JSON fixtures and comparison statistics to serde structs.
// - Keep Welch/integration/error helpers private module functions and read golden external adapter output via std::fs/std::path.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-two-disease.rs  (a `fn main` binary;
//                    an `examples/…rs` also works)
// 1:1 file move. Compares the framework two-disease ensemble mean against the
// scipy LSODA ODE and the Python Gillespie SSA ensemble.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env (`N`/`REPS`/`SIM_T`/`STEPSIZE`) -> `std::env::var`.
//   - `fs`/`path` reading the external JSON -> `std::fs` + serde structs.
//   - `as any` on parsed JSON -> concrete typed structs.
//   - ensemble RNG -> inject `SeededRandom`.
//   - `process.exit(code)` -> `std::process::exit(code)`.
// =============================================================================

// Compares the framework's two-disease ensemble mean against
//   (1) the deterministic mean-field ODE (scipy LSODA), and
//   (2) the Python Gillespie SSA ensemble mean,
// both produced by `external-references/two-disease/two_disease.py`.
//
// HOW TO RUN
// ----------
//   npm run build
//   N=1000 REPS=30 SIM_T=200 STEPSIZE=0.1 node dist/des/main-two-disease.js
//   bash external-references/run-all.sh    # runs two_disease.py too
//   node dist/des/runners/validate-two-disease.js
//
// REPORTS
// -------
//   max relative error over the trajectory between framework-mean and ODE
//   max relative error between framework-mean and SSA-mean
//   Welch t-test on final-D between framework reps and SSA reps
//   PASS if all three checks pass.
//
// Why both ODE and SSA: the ODE is a deterministic mean-field reference
// (no stochastic noise; converges to it as N → ∞). The SSA gives an
// independent stochastic implementation of the same per-person rates,
// in a different language, so it tests whether the framework's per-tick
// stochastic dynamics agree with continuous-time event-driven dynamics.

import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath = path.join(ROOT, 'out', 'two-disease-framework.json');
const pyPath = path.join(ROOT, 'out', 'external', 'two-disease', 'python.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-two-disease] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function maxRelDiff(a: ReadonlyArray<number>, b: ReadonlyArray<number>, floor = 5): {max: number; meanAbs: number} {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let mx = 0, sum = 0, n = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(floor, Math.abs(a[i]) + Math.abs(b[i]));
    const r = Math.abs(a[i] - b[i]) / denom;
    if (r > mx) mx = r;
    sum += Math.abs(a[i] - b[i]); n++;
  }
  return {max: mx, meanAbs: sum / Math.max(1, n)};
}

/**
 * Time-integrated population: trapezoid rule on (t, x). For nonlinear
 * stochastic systems, ensemble means of trajectories don't always equal
 * the deterministic ODE trajectory at every instant (peak-time variance
 * spreads the ensemble peak), but the time-INTEGRATED quantities are
 * much more stable and are the right thing to compare.
 */
function timeIntegrate(t: ReadonlyArray<number>, x: ReadonlyArray<number>): number {
  let sum = 0;
  for (let i = 1; i < t.length; i++) sum += 0.5 * (x[i] + x[i-1]) * (t[i] - t[i-1]);
  return sum;
}

function welchT(xs: ReadonlyArray<number>, ys: ReadonlyArray<number>): {t: number; df: number; p: number} {
  const m = (a: ReadonlyArray<number>) => a.reduce((s,v)=>s+v,0) / a.length;
  const v = (a: ReadonlyArray<number>) => {
    const mu = m(a);
    return a.reduce((s,vv)=>s+(vv-mu)*(vv-mu), 0) / Math.max(1, a.length - 1);
  };
  const mx = m(xs), my = m(ys);
  const vx = v(xs), vy = v(ys);
  const nx = xs.length, ny = ys.length;
  const se = Math.sqrt(vx/nx + vy/ny);
  const t = se === 0 ? 0 : (mx - my) / se;
  const num = (vx/nx + vy/ny) ** 2;
  const den = ((vx/nx) ** 2) / Math.max(1, nx - 1) + ((vy/ny) ** 2) / Math.max(1, ny - 1);
  const df = den === 0 ? Infinity : num / den;
  // Two-sided p-value via normal approximation (df is usually large here).
  const z = Math.abs(t);
  const phi = 0.5 * (1 + erf(z / Math.SQRT2));
  return {t, df, p: 2 * (1 - phi)};
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  const a1 =  0.254829592, a2 = -0.284496736, a3 =  1.421413741;
  const a4 = -1.453152027, a5 =  1.061405429, p  =  0.3275911;
  const tt = 1 / (1 + p * x);
  const y = 1 - (((((a5*tt + a4)*tt) + a3)*tt + a2)*tt + a1)*tt * Math.exp(-x*x);
  return sign * y;
}

function main() {
  const ts = loadJson(tsPath);
  const py = loadJson(pyPath);

  const meanTs = ts.meanTrace;
  const ode = py.ode;
  const ssa = py.ssa_mean;

  console.log('Two-disease framework vs Python (LSODA + Gillespie SSA)');
  console.log('==========================================================================');
  console.log(`  N=${ts.params.N}  reps=${ts.reps}  simT=${ts.params.simT}  dt=${ts.params.stepSize}`);
  console.log('');

  const compartments: Array<keyof typeof meanTs> = ['S', 'A', 'B', 'AB', 'R', 'D'];
  // Note the framework records at the END of each tick (t = (i+1)·dt) while
  // ODE/SSA were evaluated on that same grid. So the arrays line up.
  //
  // We compare three independent metrics, in increasing order of rigor:
  //   (a) per-tick max-relative-error (sensitive to peak-time variance —
  //       expected to be high for explosive epidemics with finite N);
  //   (b) time-integrated population ∫ X(t) dt (stable, rep-averaged);
  //   (c) Welch t-test on final D (most rigorous of all).
  console.log(`  Trajectory (ensemble mean) — framework (${ts.reps} reps) vs LSODA / SSA-mean`);
  console.log('  Compartment  |  max-rel  |  ∫framework  |   ∫LSODA   |   ∫SSA   | rel-err vs LSODA | rel-err vs SSA');
  console.log('  ─────────────┼───────────┼──────────────┼────────────┼──────────┼──────────────────┼────────────────');
  // Track per-compartment integrated errors so we can apply different
  // tolerances to large (R, D, S) vs small (A, B, AB) compartments.
  const intErrOde: Record<string, number> = {};
  const intErrSsa: Record<string, number> = {};
  let worstPeakOde = 0;
  for (const k of compartments) {
    if (k === 't') continue;
    const f: number[] = (meanTs as any)[k];
    const o: number[] = (ode  as any)[k];
    const s: number[] = (ssa  as any)[k];
    const tArr: number[] = meanTs.t;
    const oR = maxRelDiff(f, o).max;
    if (oR > worstPeakOde) worstPeakOde = oR;
    const intF = timeIntegrate(tArr, f);
    const intO = timeIntegrate(tArr, o);
    const intS = timeIntegrate(tArr, s);
    const intROde = Math.abs(intF - intO) / Math.max(1, intO);
    const intRSsa = Math.abs(intF - intS) / Math.max(1, intS);
    intErrOde[String(k)] = intROde;
    intErrSsa[String(k)] = intRSsa;
    console.log(`  ${String(k).padEnd(11)}  |  ${(oR*100).toFixed(1).padStart(6)} %  |  ${intF.toFixed(0).padStart(8)}    |  ${intO.toFixed(0).padStart(8)}  |  ${intS.toFixed(0).padStart(6)}  |  ${(intROde*100).toFixed(2).padStart(13)} %  |  ${(intRSsa*100).toFixed(2).padStart(11)} %`);
  }

  // Final-state Welch test on D.
  const tsFinalD: number[] = ts.finalDeaths;
  const pySsaMeanD: number = py.ssa_final_D_mean;
  const pySsaStdD:  number = py.ssa_final_D_std;
  const pySsaReps:  number = py.ssa_reps;
  // Reconstruct synthetic SSA samples from mean+std for Welch (a coarser
  // check; better would be to ship the SSA samples themselves).
  const tsMeanD = tsFinalD.reduce((s,v)=>s+v,0) / tsFinalD.length;
  const tsStdD  = Math.sqrt(tsFinalD.reduce((s,v)=>s+(v-tsMeanD)*(v-tsMeanD), 0) / Math.max(1, tsFinalD.length - 1));
  const seGap = Math.sqrt(tsStdD*tsStdD/tsFinalD.length + pySsaStdD*pySsaStdD/pySsaReps);
  const tStat = seGap === 0 ? 0 : (tsMeanD - pySsaMeanD) / seGap;
  const z = Math.abs(tStat);
  const p = 2 * (1 - 0.5 * (1 + erf(z / Math.SQRT2)));
  console.log('');
  console.log('  Welch test on final-D (framework reps vs SSA reps):');
  console.log(`    framework: mean=${tsMeanD.toFixed(2)}  std=${tsStdD.toFixed(2)}  n=${tsFinalD.length}`);
  console.log(`    Python SSA: mean=${pySsaMeanD.toFixed(2)}  std=${pySsaStdD.toFixed(2)}  n=${pySsaReps}`);
  console.log(`    t = ${tStat.toFixed(3)}    p ≈ ${p.toFixed(3)}`);

  // ODE vs SSA ensemble-mean (sanity check that python ref is internally consistent)
  const odeFinalD = ode.D[ode.D.length - 1];
  console.log(`    LSODA mean-field final D = ${odeFinalD.toFixed(2)} (compare to SSA mean ${pySsaMeanD.toFixed(2)})`);

  // Bounds. Two regimes:
  //   - MONOTONIC accumulator compartments (R, D): only grow over the
  //     epidemic, so the integrated population is dominated by the
  //     deterministic limit and ensemble variance is small. Tolerance:
  //     5% vs LSODA.
  //   - TRANSIENT compartments (S, A, B, AB): rise then fall, peak-time
  //     variance dominates; ensemble-mean integrals deviate from the
  //     deterministic ODE because the ensemble peak is broader than each
  //     rep's spike. Tolerance: 20% vs LSODA, 10% vs SSA (both stochastic).
  //   - Per-tick max-rel-err vs LSODA: tolerated up to 50% (peak-time
  //     variance — see Anderson & May 1992 ch.2).
  //   - Welch p > 0.01 on final D: most rigorous, must pass.
  const tolIntOdeMon       = 0.05;
  const tolIntOdeTransient = 0.20;
  const tolIntSsa          = 0.10;
  const tolPeak            = 0.50;
  const monOk       = ['R','D'].every(k => intErrOde[k] < tolIntOdeMon);
  const transientOk = ['S','A','B','AB'].every(k => intErrOde[k] < tolIntOdeTransient);
  const largeOk = monOk;
  const smallOk = transientOk;
  const ssaOk   = compartments.filter(k => k !== 't').every(k => intErrSsa[String(k)] < tolIntSsa);
  const okPeak  = worstPeakOde < tolPeak;
  const okWelch = p > 0.01;
  console.log('');
  console.log(`  ∫-rel-err vs LSODA, monotonic (R,D)        < ${(tolIntOdeMon*100).toFixed(0)}%: ${monOk ? 'yes' : 'NO'}`);
  console.log(`  ∫-rel-err vs LSODA, transient (S,A,B,AB)   < ${(tolIntOdeTransient*100).toFixed(0)}%: ${transientOk ? 'yes' : 'NO'}`);
  console.log(`  ∫-rel-err vs SSA-mean (all)       < ${(tolIntSsa*100).toFixed(0)}%: ${ssaOk ? 'yes' : 'NO'}`);
  console.log(`  max peak-rel-err vs LSODA         < ${(tolPeak*100).toFixed(0)}%: ${okPeak ? 'yes' : 'NO'}  (got ${(worstPeakOde*100).toFixed(2)}%)`);
  console.log(`  Welch p > 0.01 (final D)              : ${okWelch ? 'yes' : 'NO'}  (got p=${p.toFixed(3)})`);

  const ok = largeOk && smallOk && ssaOk && okPeak && okWelch;
  console.log(ok ? '  PASS' : '  FAIL');
  process.exit(ok ? 0 : 1);
}

main();
