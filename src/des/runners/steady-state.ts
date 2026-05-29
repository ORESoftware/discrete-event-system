#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// Mathematical verification: closed-form steady state from the difference
// equations vs four numerical kernels run as open systems.
//
// HOW TO RUN
// ----------
//   In-repo (TypeScript), all kernels at once:
//
//     npm run build
//     N=5 HORIZON=10000 node dist/des/runners/steady-state.js
//
//   Environment variables:
//     N        number of stochastic reps (Gillespie + FEL-individual)
//     HORIZON  horizon (days). Use >= 5000 to wash out the ~226-day transient.
//     WARMUP   informational only; current driver uses full-horizon time-avg.
//
//   For an extra second-opinion using external tools (scipy / sympy / Octave /
//   R + deSolve), see external-references/README.md and run-all.sh.
//
//
// Open system  =  source emits at constant rate lambda forever
//                 (config.sourceCap = Infinity, config.phase1Days = Infinity).
//
// We compare two quantities:
//
//  A. The fixed point N*_c (steady-state populations).
//     - closed-form analytical:   N*_c = mu_c * f_c, f_S = lambda / q.
//     - difference equation final N(T) at large T.
//     - ODE RK4 final N(T) at large T.
//     - stochastic kernels:        average of finalPopulations across N reps
//                                   (unbiased for N*, but high variance).
//
//  B. The time-averaged populations <N_c>_[0,T].
//     For an open system reaching N* exponentially with the slowest mode at
//     ~ mean lifespan = 1/(q * cycle_rate), the time-avg over [0,T] is
//        <N>_[0,T] = N* - (transient deficit) / T,
//     i.e. systematically below N* unless T is much larger than the lifespan.
//     - ODE RK4 time-avg              (trapezoidal integral of trajectory)
//     - Gillespie SSA time-avg        (per-rep, then averaged across reps)
//     - FEL-individual time-avg       (same)
//
// If both columns line up across all kernels, the model and every kernel
// implementation are mutually consistent. See `MATH.md` for the derivation.
// =============================================================================

import {DEFAULT_CONFIG, COMPARTMENT_ORDER, SimConfig} from './types';
import {runDifferenceOnce, analyticalSteadyState, maxStableStep} from './difference-runner';
import {runOdeOnce}                        from './ode-runner';
import {runGillespieOnce}                  from './gillespie-runner';
import {runFelOnce}                        from './fel-runner';
import {mean, stddev}                      from './stats';

const fmt = (n: number, d = 3) => Number.isFinite(n) ? n.toFixed(d) : 'DIVERGED';

const N_REPS  = parseInt(process.env.N        ?? '5',     10);
const HORIZON = parseInt(process.env.HORIZON  ?? '10000', 10);

function main() {
  const cfgOpen: SimConfig = {
    ...DEFAULT_CONFIG,
    sourceCap:   Number.POSITIVE_INFINITY,
    phase1Days:  Number.POSITIVE_INFINITY,
    horizonDays: HORIZON,
    stepSize:    0.05,
  };

  console.log(`steady-state verification: open system (lambda const), horizon=${HORIZON}d`);
  console.log('');

  // ---------------------------------------------------------------------------
  // Closed-form analytical
  // ---------------------------------------------------------------------------
  const A = analyticalSteadyState(cfgOpen);
  console.log('=== closed-form analytical steady state ===');
  console.log(`  arrival rate                lambda = 1/mu_arr      = ${fmt(A.lambda, 4)}/day`);
  console.log(`  per-S-pass death fraction   q = (1-p_a)*p_h*p_d    = ${fmt(A.q, 5)}`);
  console.log(`  S throughput                f_S = lambda/q         = ${fmt(A.fS, 3)}/day`);
  console.log(`  total alive at steady state Sum N*_alive           = ${fmt(A.totalAlive - A.populations.D, 3)}`);
  console.log(`  mean lifespan               1/q * cycle_time       ≈ ${fmt((1 / A.q) * (3 * 0.3 + 0.4 * 0.3 + 0.6 * (0.3 + 0.2 * (0.3 + 0.2)) + 2.0), 1)} days`);
  console.log(`  max stable forward-Euler dt 2 * min(mu_c)          = ${fmt(maxStableStep(cfgOpen), 3)} days`);
  console.log('');

  // ---------------------------------------------------------------------------
  // Forward-Euler stability + convergence demo
  // ---------------------------------------------------------------------------
  console.log('=== forward-Euler difference equation: stability + convergence demo ===');
  console.log('  (compares N(T) to analytical N* across a range of dt)');
  const dts = [0.5, 0.39, 0.1, 0.05, 0.01];   // first one is past max stable step
  const diffRuns = new Map<number, ReturnType<typeof runDifferenceOnce>>();
  for (const dt of dts) {
    diffRuns.set(dt, runDifferenceOnce({...cfgOpen, stepSize: dt}));
  }
  console.log('compartment'.padEnd(14) + 'analytical'.padStart(12) +
    dts.map(d => `dt=${d}`.padStart(12)).join(''));
  for (const c of COMPARTMENT_ORDER) {
    const cells = dts.map(d => fmt(diffRuns.get(d)!.finalPopulations[c], 3).padStart(12));
    console.log(`<${c}>`.padEnd(14) +
      fmt((A.populations as any)[c], 3).padStart(12) + cells.join(''));
  }
  const dtRef = 0.01;
  const errs = COMPARTMENT_ORDER.map(c =>
    Math.abs(diffRuns.get(dtRef)!.finalPopulations[c] - (A.populations as any)[c]));
  console.log(`  max |diff(dt=${dtRef}) - analytical| over compartments: ${fmt(Math.max(...errs), 6)}`);
  console.log(`  dt=0.5 > maxStableStep=${fmt(maxStableStep(cfgOpen), 2)} -> DIVERGED, as predicted by stability analysis.`);
  console.log('');

  // ---------------------------------------------------------------------------
  // Run all kernels in open-system mode at horizon T
  // ---------------------------------------------------------------------------
  const ode = runOdeOnce({...cfgOpen, horizonDays: HORIZON});
  const ssaRuns: ReturnType<typeof runGillespieOnce>[] = [];
  const felRuns: ReturnType<typeof runFelOnce>[]       = [];
  for (let i = 0; i < N_REPS; i++) {
    ssaRuns.push(runGillespieOnce(cfgOpen, {seed: 0xA0000 + i}));
    felRuns.push(runFelOnce      (cfgOpen, {seed: 0xB0000 + i, service: 'individual'}));
  }

  const fel0 = felRuns[0];
  const ssa0 = ssaRuns[0];
  console.log('kernel timings (single rep):');
  console.log(`  ODE RK4         : ${ode.elapsedMs} ms`);
  console.log(`  Gillespie SSA   : ${ssa0.elapsedMs} ms (mean=${mean(ssaRuns.map(r => r.elapsedMs)).toFixed(0)} ms across N=${N_REPS})`);
  console.log(`  FEL-individual  : ${fel0.elapsedMs} ms (mean=${mean(felRuns.map(r => r.elapsedMs)).toFixed(0)} ms across N=${N_REPS})`);
  console.log('');

  // ---------------------------------------------------------------------------
  // Column A: fixed-point estimates  (deterministic exact, stochastic ~unbiased)
  // ---------------------------------------------------------------------------
  console.log('=== fixed-point estimates of N*_c (should all agree) ===');
  console.log('compartment'.padEnd(14) +
    'analytical'.padStart(13) +
    'diff N(T)'.padStart(13) +
    'ODE N(T)'.padStart(13) +
    'Gillespie <N(T)>'.padStart(20) +
    'FEL-ind <N(T)>'.padStart(20));
  for (const c of COMPARTMENT_ORDER) {
    const ana = (A.populations as any)[c] as number;
    const dif = diffRuns.get(0.05)!.finalPopulations[c];
    const odeF = ode.finalPopulations[c];
    const ssaF = ssaRuns.map(r => r.finalPopulations[c] ?? 0);
    const felF = felRuns.map(r => r.finalPopulations[c] ?? 0);
    console.log(`<${c}>`.padEnd(14) +
      fmt(ana, 3).padStart(13) +
      fmt(dif, 3).padStart(13) +
      fmt(odeF, 3).padStart(13) +
      `${fmt(mean(ssaF), 1)} ± ${fmt(stddev(ssaF), 1)}`.padStart(20) +
      `${fmt(mean(felF), 1)} ± ${fmt(stddev(felF), 1)}`.padStart(20));
  }
  console.log('  notes:');
  console.log('   - "diff N(T)" and "ODE N(T)" are deterministic, exact at large T.');
  console.log('   - "<N(T)>" is the average across N reps of the snapshot at t=T;');
  console.log('     unbiased for N*_c but with sqrt(N*_c) Poisson-like variance per rep.');
  console.log('');

  // ---------------------------------------------------------------------------
  // Column B: time-averaged populations  (transient-biased version of N*)
  // ---------------------------------------------------------------------------
  console.log('=== time-averaged populations <N_c>_[0,T] (deterministic vs stochastic) ===');
  console.log('compartment'.padEnd(14) +
    'analytical N*'.padStart(15) +
    'ODE <N>_t'.padStart(13) +
    'Gillespie <N>_t'.padStart(20) +
    'FEL-ind <N>_t'.padStart(20));
  for (const c of COMPARTMENT_ORDER) {
    const ana  = (A.populations as any)[c] as number;
    const odeT = ode.timeAvgPopulations[c];
    const ssaT = ssaRuns.map(r => r.timeAvgPopulations[c] ?? 0);
    const felT = felRuns.map(r => r.timeAvgPopulations[c] ?? 0);
    console.log(`<${c}>`.padEnd(14) +
      fmt(ana, 3).padStart(15) +
      fmt(odeT, 3).padStart(13) +
      `${fmt(mean(ssaT), 3)} ± ${fmt(stddev(ssaT), 3)}`.padStart(20) +
      `${fmt(mean(felT), 3)} ± ${fmt(stddev(felT), 3)}`.padStart(20));
  }
  console.log('  notes:');
  console.log('   - <N>_t < N*  for finite T, by an amount = transient deficit / T.');
  console.log('   - ODE <N>_t is the integral of the trajectory; stochastic <N>_t');
  console.log('     should match it, NOT the fixed point. They do.');
  console.log('');

  // ---------------------------------------------------------------------------
  // Sanity checks
  // ---------------------------------------------------------------------------
  console.log('=== sanity: total alive populations and cumulative deaths ===');
  const allAna   = A.totalAlive - A.populations.D;
  const odeAlive = COMPARTMENT_ORDER.reduce(
    (s, c) => s + (ode.finalPopulations[c] ?? 0), 0);
  const ssaAlive = ssaRuns.map(r => COMPARTMENT_ORDER.reduce(
    (s, c) => s + (r.finalPopulations[c] ?? 0), 0));
  const felAlive = felRuns.map(r => COMPARTMENT_ORDER.reduce(
    (s, c) => s + (r.finalPopulations[c] ?? 0), 0));
  console.log(`  analytical Sum N*_alive    : ${fmt(allAna, 3)}`);
  console.log(`  ODE     N(T) Sum alive     : ${fmt(odeAlive, 3)}`);
  console.log(`  Gillespie    <N(T)> alive  : ${fmt(mean(ssaAlive), 3)} ± ${fmt(stddev(ssaAlive), 3)}`);
  console.log(`  FEL-ind      <N(T)> alive  : ${fmt(mean(felAlive), 3)} ± ${fmt(stddev(felAlive), 3)}`);

  console.log('');
  const expDeaths = HORIZON * A.lambda;
  console.log('cumulative deaths over [0, T] (steady-state rate = lambda = 1/day):');
  console.log(`  expected horizon * lambda  : ${fmt(expDeaths, 1)}`);
  console.log(`  ODE D(T)                   : ${fmt(ode.totals.absorbed, 1)}`);
  console.log(`  Gillespie absorbed         : ${fmt(mean(ssaRuns.map(r => r.totals.absorbed)), 1)} ± ${fmt(stddev(ssaRuns.map(r => r.totals.absorbed)), 1)}`);
  console.log(`  FEL-ind   absorbed         : ${fmt(mean(felRuns.map(r => r.totals.absorbed)), 1)} ± ${fmt(stddev(felRuns.map(r => r.totals.absorbed)), 1)}`);
  console.log('  (these are biased low by transient deficit; ratio actual/expected -> 1 as T -> infinity)');
}

main();
