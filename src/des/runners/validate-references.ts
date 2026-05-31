#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_references.rs.
// - Keep this as a CLI validation binary with Result-returning main; map N/STEPSIZE to clap/std::env.
// - Convert per-kernel summaries and Welch outputs into golden comparison structs reused by report rendering.
// - Keep kernel calls delegated to migrated runner modules and stats math delegated to runners::stats.
'use strict';

// =============================================================================
// Compare four independent reference kernels against the framework's new
// PerIndividualProcessor. The four references are:
//
//   FEL-individual : per-entity exit clocks pulled by a global event list,
//                    uniform U(a,b) residence (matches PI exactly in the limit
//                    as stepSize -> 0).
//   Gillespie SSA  : compartment-level direct method, exponential rates.
//                    Splits agree exactly; mean populations agree because mean
//                    residence equals (a+b)/2 in both.
//   ODE RK4        : deterministic mean-field of the same compartmental model.
//                    Splits are analytical; populations are the deterministic
//                    expectation that all stochastic kernels should agree
//                    with on average.
//   PerIndividual  : framework run-loop with per-entity exit clocks, uniform
//                    residence. Our subject-under-test.
//
// We run N reps of each stochastic kernel (Welch t-test for splits and
// populations) and one deterministic ODE run as the analytical reference.
// =============================================================================

import {DEFAULT_CONFIG, RunResult, COMPARTMENT_ORDER} from './types';
import {runFelOnce}            from './fel-runner';
import {runPerIndividualOnce}  from './per-individual-runner';
import {runGillespieOnce}      from './gillespie-runner';
import {runOdeOnce}            from './ode-runner';
import {mean, stddev, welch}   from './stats';

const N = parseInt(process.env.N ?? '20', 10);
const PI_STEPSIZE = parseFloat(process.env.STEPSIZE ?? '0.05');

const fmt = (n: number, d = 4) => Number.isFinite(n) ? n.toFixed(d) : String(n);
const collectSplit = (rs: RunResult[], from: string, to: string) =>
  rs.map(r => r.splitProbs[from]?.[to] ?? 0);
const collectPop = (rs: RunResult[], c: string) =>
  rs.map(r => r.timeAvgPopulations[c] ?? 0);

function kernelStats(rs: RunResult[], extractor: (r: RunResult) => number) {
  const xs = rs.map(extractor);
  return `${fmt(mean(xs), 4)} ± ${fmt(stddev(xs), 4)}`.padStart(20);
}

function main() {
  const cfg = {...DEFAULT_CONFIG, stepSize: PI_STEPSIZE};
  console.log(`validate-references.ts: N=${N} reps per stochastic kernel; PI stepSize=${PI_STEPSIZE}d`);
  console.log(`  horizon=${cfg.horizonDays}d   phase1=${cfg.phase1Days}d   sourceCap=${cfg.sourceCap}`);
  console.log('');

  const piRuns:    RunResult[] = [];
  const felRuns:   RunResult[] = [];
  const ssaRuns:   RunResult[] = [];

  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    piRuns.push (runPerIndividualOnce(cfg,            {seed: 0xC0000 + i}));
    felRuns.push(runFelOnce          (DEFAULT_CONFIG, {seed: 0xD0000 + i, service: 'individual'}));
    ssaRuns.push(runGillespieOnce    (DEFAULT_CONFIG, {seed: 0xE0000 + i}));
  }
  const ode = runOdeOnce(DEFAULT_CONFIG);

  const elapsed = Date.now() - t0;
  console.log(`total wall: ${elapsed} ms`);
  console.log(`  per-individual mean wall:  ${fmt(mean(piRuns.map(r => r.elapsedMs)), 1)} ms / rep`);
  console.log(`  fel-individual mean wall:  ${fmt(mean(felRuns.map(r => r.elapsedMs)), 1)} ms / rep`);
  console.log(`  gillespie SSA  mean wall:  ${fmt(mean(ssaRuns.map(r => r.elapsedMs)), 1)} ms / rep`);
  console.log(`  ODE RK4  wall:             ${ode.elapsedMs} ms (deterministic)`);

  console.log('');
  console.log('=== empirical branching probabilities ===');
  console.log('              ' + ['expected', 'PerIndividual', 'FEL-individual', 'Gillespie SSA', 'ODE'].map(s => s.padStart(20)).join(''));
  const splits: Array<[string, string, number]> = [
    ['I-P', 'I-A', cfg.probabilities.asymptomaticShare],
    ['I-P', 'I-S', 1 - cfg.probabilities.asymptomaticShare],
    ['I-S', 'R',   1 - cfg.probabilities.hospitalizationGivenSymptom],
    ['I-S', 'I-H', cfg.probabilities.hospitalizationGivenSymptom],
    ['I-H', 'R',   1 - cfg.probabilities.caseFatalityGivenHospital],
    ['I-H', 'D',   cfg.probabilities.caseFatalityGivenHospital],
  ];
  for (const [from, to, expected] of splits) {
    const fel = collectSplit(felRuns, from, to);
    const pi  = collectSplit(piRuns,  from, to);
    const ssa = collectSplit(ssaRuns, from, to);
    const odeVal = ode.splitProbs[from]?.[to] ?? 0;
    console.log(
      (from + ' -> ' + to).padEnd(14) +
      fmt(expected, 4).padStart(20) +
      `${fmt(mean(pi), 4)} ± ${fmt(stddev(pi), 4)}`.padStart(20) +
      `${fmt(mean(fel), 4)} ± ${fmt(stddev(fel), 4)}`.padStart(20) +
      `${fmt(mean(ssa), 4)} ± ${fmt(stddev(ssa), 4)}`.padStart(20) +
      fmt(odeVal, 4).padStart(20),
    );
  }

  console.log('');
  console.log('=== time-averaged compartment populations ===');
  console.log('              ' + ['PerIndividual', 'FEL-individual', 'Gillespie SSA', 'ODE'].map(s => s.padStart(20)).join(''));
  for (const c of COMPARTMENT_ORDER) {
    const pi  = collectPop(piRuns,  c);
    const fel = collectPop(felRuns, c);
    const ssa = collectPop(ssaRuns, c);
    console.log(
      `<${c}>`.padEnd(14) +
      `${fmt(mean(pi), 3)} ± ${fmt(stddev(pi), 3)}`.padStart(20) +
      `${fmt(mean(fel), 3)} ± ${fmt(stddev(fel), 3)}`.padStart(20) +
      `${fmt(mean(ssa), 3)} ± ${fmt(stddev(ssa), 3)}`.padStart(20) +
      fmt(ode.timeAvgPopulations[c], 3).padStart(20),
    );
  }

  console.log('');
  console.log('=== pairwise Welch t-tests on time-averaged populations ===');
  const pairs: Array<[string, RunResult[], RunResult[]]> = [
    ['PI vs FEL-ind ', piRuns,  felRuns],
    ['PI vs Gillesp ', piRuns,  ssaRuns],
    ['FEL vs Gilles ', felRuns, ssaRuns],
  ];
  console.log('compartment    ' + pairs.map(p => `${p[0]}  t (p)`.padStart(30)).join(''));
  for (const c of COMPARTMENT_ORDER) {
    const cells = pairs.map(([_, a, b]) => {
      const w = welch(collectPop(a, c), collectPop(b, c));
      const verdict = w.reject99 ? '  NO99 ' : w.reject95 ? '  no95 ' : '  yes  ';
      return `${fmt(w.t, 2).padStart(7)} (p=${fmt(w.pValueTwoSided, 3)}) ${verdict}`.padStart(30);
    });
    console.log(`<${c}>`.padEnd(14) + cells.join(''));
  }

  console.log('');
  console.log('=== totals ===');
  console.log(
    'created '.padEnd(14) +
    kernelStats(piRuns,  r => r.totals.created) +
    kernelStats(felRuns, r => r.totals.created) +
    kernelStats(ssaRuns, r => r.totals.created) +
    fmt(ode.totals.created, 1).padStart(20),
  );
  console.log(
    'absorbed (D)'.padEnd(14) +
    kernelStats(piRuns,  r => r.totals.absorbed) +
    kernelStats(felRuns, r => r.totals.absorbed) +
    kernelStats(ssaRuns, r => r.totals.absorbed) +
    fmt(ode.totals.absorbed, 1).padStart(20),
  );
}

main();
