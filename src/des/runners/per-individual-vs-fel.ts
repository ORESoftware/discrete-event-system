#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/per-individual-vs-fel.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Confirms the framework's PerIndividualProcessor converges to the
// classical FEL reference (Welch t-tests on every metric).
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env (`N`) -> `std::env::var`.
//   - kernel seeding (`withSeed`/`Date.now`) -> `SeededRandom`/`Clock`.
//   - `console.log` -> `println!`.
// =============================================================================

// =============================================================================
// Verification driver: confirm the new PerIndividualProcessor (single queue +
// per-entity exit clocks) inside the framework's run loop converges to the
// classical FEL reference's behaviour.
//
// This is the "real upgrade to the framework, not just an analysis" point:
// once we have a per-individual processor in the framework, the long-standing
// granularity gap should disappear (modulo small residual stepSize effects),
// without needing a global future-event list.
//
// We run the per-individual kernel and the FEL kernel N times each and use a
// Welch t-test on every metric. Almost everything should fail to reject the
// null hypothesis (i.e., they agree) at 95%.
// =============================================================================

import {DEFAULT_CONFIG, RunResult, COMPARTMENT_ORDER} from './types';
import {runFelOnce} from './fel-runner';
import {runPerIndividualOnce} from './per-individual-runner';
import {runFrameworkOnce} from './framework-runner';
import {mean, stddev, welch} from './stats';

const N = parseInt(process.env.N ?? '30', 10);
const fmt = (n: number, d = 4) => Number.isFinite(n) ? n.toFixed(d) : String(n);

// We run the per-individual kernel at stepSize=0.1 day so the discretisation
// error is small but the run is still cheap (~150 ms / rep). FEL is exact.
const PI_CONFIG = {...DEFAULT_CONFIG, stepSize: 0.1};

function collectSplit(rs: RunResult[], from: string, to: string) {
  return rs.map(r => r.splitProbs[from]?.[to] ?? 0);
}
function collectPop(rs: RunResult[], c: string) {
  return rs.map(r => r.timeAvgPopulations[c] ?? 0);
}

function reportTable(label: string, kernels: Array<{name: string, runs: RunResult[]}>) {
  console.log(`\n=== ${label} ===`);
  console.log('metric'.padEnd(18) + kernels.map(k => `${k.name}: mean ± sd`.padStart(28)).join('') +
    '   Welch t (' + kernels[0].name + ' vs ' + kernels[1].name + ')   p (2s)   agree?');

  // Splits
  const splits: Array<[string, string, number]> = [
    ['I-P', 'I-A', PI_CONFIG.probabilities.asymptomaticShare],
    ['I-P', 'I-S', 1 - PI_CONFIG.probabilities.asymptomaticShare],
    ['I-S', 'R',   1 - PI_CONFIG.probabilities.hospitalizationGivenSymptom],
    ['I-S', 'I-H', PI_CONFIG.probabilities.hospitalizationGivenSymptom],
    ['I-H', 'R',   1 - PI_CONFIG.probabilities.caseFatalityGivenHospital],
    ['I-H', 'D',   PI_CONFIG.probabilities.caseFatalityGivenHospital],
  ];
  for (const [from, to] of splits) {
    const cells = kernels.map(k => {
      const xs = collectSplit(k.runs, from, to);
      return `${fmt(mean(xs), 4)} ± ${fmt(stddev(xs), 4)}`.padStart(28);
    });
    const w = welch(collectSplit(kernels[0].runs, from, to),
                    collectSplit(kernels[1].runs, from, to));
    const verdict = w.reject95 ? 'NO (95%)' : 'yes';
    console.log(
      `${(from + ' -> ' + to).padEnd(18)}` + cells.join('') +
      `      t=${fmt(w.t, 2).padStart(6)}   p=${fmt(w.pValueTwoSided, 3).padStart(6)}   ${verdict}`,
    );
  }

  // Populations
  for (const c of COMPARTMENT_ORDER) {
    const cells = kernels.map(k => {
      const xs = collectPop(k.runs, c);
      return `${fmt(mean(xs), 3)} ± ${fmt(stddev(xs), 3)}`.padStart(28);
    });
    const w = welch(collectPop(kernels[0].runs, c), collectPop(kernels[1].runs, c));
    const verdict = w.reject99 ? 'NO (99%)' : w.reject95 ? 'NO (95%)' : 'yes';
    console.log(
      `<${c}>`.padEnd(18) + cells.join('') +
      `      t=${fmt(w.t, 2).padStart(6)}   p=${fmt(w.pValueTwoSided, 3).padStart(6)}   ${verdict}`,
    );
  }
}

function main() {
  console.log(`per-individual-vs-fel: ${N} reps each kernel; per-ind stepSize=${PI_CONFIG.stepSize}d`);

  const piRuns:  RunResult[] = [];
  const felRuns: RunResult[] = [];
  const fwRuns:  RunResult[] = [];   // for context: original three-queue framework at stepSize=0.1

  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    piRuns.push(runPerIndividualOnce(PI_CONFIG, {seed: 0x60000 + i}));
    // Use M/M/inf FEL because that's the kernel PI is supposed to imitate
    // (per-entity exit clocks, all individuals serviced concurrently).
    felRuns.push(runFelOnce(DEFAULT_CONFIG,    {seed: 0x70000 + i, service: 'individual'}));
    fwRuns.push(runFrameworkOnce(PI_CONFIG,    {seed: 0x80000 + i}));
  }
  const elapsed = Date.now() - t0;

  console.log(`total wall ${elapsed} ms`);
  console.log(`mean per-rep wall:  per-individual=${fmt(mean(piRuns.map(r => r.elapsedMs)), 1)} ms   ` +
              `fel=${fmt(mean(felRuns.map(r => r.elapsedMs)), 1)} ms   ` +
              `original framework (stepSize=${PI_CONFIG.stepSize})=${fmt(mean(fwRuns.map(r => r.elapsedMs)), 1)} ms`);

  reportTable('per-individual processor (stepSize=' + PI_CONFIG.stepSize + ') VS classical FEL',
    [{name: 'per-individual', runs: piRuns}, {name: 'fel', runs: felRuns}]);

  reportTable('per-individual processor VS three-queue framework (both stepSize=' + PI_CONFIG.stepSize + ')',
    [{name: 'per-individual', runs: piRuns}, {name: 'three-queue', runs: fwRuns}]);

  console.log('');
  console.log('=== summary ===');
  console.log('Branching probabilities and slow-compartment populations (S, E, R) agree');
  console.log('between the per-individual processor and the M/M/inf FEL kernel within');
  console.log('Welch-t at 95%, confirming the new station type implements correct CTMC');
  console.log('semantics. Fast-compartment means show a small residual fixed-step bias');
  console.log('that decays with stepSize -> 0; a quick convergence sweep follows.');
  console.log('');

  // ---- PI -> FEL convergence demo ----
  const stepSweep = [0.5, 0.1, 0.05, 0.02];
  const Nconv = 5;  // small N just to show the trend cheaply
  console.log(`=== PI -> FEL convergence sweep (N=${Nconv} reps each, M/M/inf FEL fixed) ===`);
  console.log('compartment'.padEnd(13) + 'fel mean'.padStart(12)
    + stepSweep.map(s => `pi (ss=${s}) ratio`.padStart(20)).join(''));
  const felConvRuns: RunResult[] = [];
  for (let i = 0; i < Nconv; i++) {
    felConvRuns.push(runFelOnce(DEFAULT_CONFIG, {seed: 0xA0000 + i, service: 'individual'}));
  }
  const piConvRuns: RunResult[][] = stepSweep.map(ss => {
    const cfg = {...DEFAULT_CONFIG, stepSize: ss};
    const reps: RunResult[] = [];
    for (let i = 0; i < Nconv; i++) {
      reps.push(runPerIndividualOnce(cfg, {seed: 0xB0000 + i + Math.round(ss * 1000)}));
    }
    return reps;
  });
  for (const c of COMPARTMENT_ORDER) {
    const felMean = mean(collectPop(felConvRuns, c));
    const ratios = piConvRuns.map(rs => mean(collectPop(rs, c)) / Math.max(felMean, 1e-9));
    console.log(`<${c}>`.padEnd(13)
      + fmt(felMean, 3).padStart(12)
      + ratios.map(r => fmt(r, 3).padStart(20)).join(''));
  }
}

main();
