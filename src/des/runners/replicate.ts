#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/replicate.rs  (a `fn main` binary; an
//                    `examples/replicate.rs` also works)
// 1:1 file move. Runs N independent replications of {framework, FEL} kernels and
// Welch t-tests every empirical metric.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env (`N`) -> `std::env::var`.
//   - `fs`/`path` + any `JSON` dump -> `std::fs` + `serde_json`.
//   - fixed seeds (`0x10000`/`0x20000`) + `Date.now()` -> `SeededRandom`/`Clock`.
//   - `as any` / `?? 0` on RunResult maps -> typed struct + `HashMap` lookups.
//   - `console.log` -> `println!`.
// =============================================================================

// =============================================================================
// N=30 independent replications of {framework, FEL} kernels on the same model.
// Welch's t-test on every empirical metric (split probabilities, time-averaged
// populations, totals) tells us where the two kernels really agree and where
// they don't, beyond single-run noise.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {DEFAULT_CONFIG, RunResult, COMPARTMENT_ORDER} from './types';
import {runFrameworkOnce} from './framework-runner';
import {runFelOnce} from './fel-runner';
import {mean, stddev, welch} from './stats';

const N = parseInt(process.env.N ?? '30', 10);
const BASE_SEED_FW  = 0x10000;
const BASE_SEED_FEL = 0x20000;

function fmt(n: number, d = 4) { return Number.isFinite(n) ? n.toFixed(d) : String(n); }

function collectSplit(results: RunResult[], from: string, to: string): number[] {
  return results.map(r => r.splitProbs[from]?.[to] ?? 0);
}
function collectPop(results: RunResult[], compartment: string): number[] {
  return results.map(r => r.timeAvgPopulations[compartment] ?? 0);
}
function collect(results: RunResult[], extractor: (r: RunResult) => number): number[] {
  return results.map(extractor);
}

function main() {
  const config = DEFAULT_CONFIG;
  console.log(`replicate.ts: ${N} replications per kernel`);
  console.log(`  config:   stepSize=${config.stepSize}d horizon=${config.horizonDays}d cap=${config.sourceCap}`);

  const fwResults: RunResult[]  = [];
  const felResults: RunResult[] = [];

  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    fwResults.push(runFrameworkOnce(config, {seed: BASE_SEED_FW + i}));
    felResults.push(runFelOnce(config,       {seed: BASE_SEED_FEL + i}));
  }
  const elapsed = Date.now() - t0;
  console.log(`  total:    ${elapsed} ms`);
  console.log(`  framework mean wall: ${fmt(mean(fwResults.map(r => r.elapsedMs)), 1)} ms`);
  console.log(`  fel       mean wall: ${fmt(mean(felResults.map(r => r.elapsedMs)), 1)} ms`);
  console.log('');

  const splits: Array<[string, string, number]> = [
    ['I-P', 'I-A', config.probabilities.asymptomaticShare],
    ['I-P', 'I-S', 1 - config.probabilities.asymptomaticShare],
    ['I-S', 'R',   1 - config.probabilities.hospitalizationGivenSymptom],
    ['I-S', 'I-H', config.probabilities.hospitalizationGivenSymptom],
    ['I-H', 'R',   1 - config.probabilities.caseFatalityGivenHospital],
    ['I-H', 'D',   config.probabilities.caseFatalityGivenHospital],
  ];

  console.log('=== empirical branching probabilities (N=' + N + ' reps each) ===');
  console.log(
    'transition'.padEnd(15) +
    'expected'.padStart(10) +
    '  framework: mean ± sd'.padStart(28) +
    '  fel: mean ± sd'.padStart(22) +
    '  Welch t'.padStart(12) +
    '  p (2-sided)'.padStart(15) +
    '  agree?'.padStart(10),
  );
  for (const [from, to, expected] of splits) {
    const fw = collectSplit(fwResults, from, to);
    const fl = collectSplit(felResults, from, to);
    const w = welch(fw, fl);
    const verdict = w.reject95 ? 'NO (95%)' : 'yes';
    console.log(
      `${(from + ' -> ' + to).padEnd(15)}` +
      `${fmt(expected, 4).padStart(10)}` +
      `      ${fmt(w.meanA, 4)} ± ${fmt(stddev(fw), 4)}` +
      `    ${fmt(w.meanB, 4)} ± ${fmt(stddev(fl), 4)}` +
      `   t=${fmt(w.t, 2).padStart(6)}` +
      `   p=${fmt(w.pValueTwoSided, 4).padStart(8)}` +
      `      ${verdict}`,
    );
  }

  console.log('');
  console.log('=== time-averaged compartment populations (N=' + N + ' reps each) ===');
  console.log(
    'compartment'.padEnd(15) +
    '  framework: mean ± sd'.padStart(28) +
    '  fel: mean ± sd'.padStart(22) +
    '  Welch t'.padStart(12) +
    '  p (2-sided)'.padStart(15) +
    '  agree?'.padStart(10),
  );
  for (const c of COMPARTMENT_ORDER) {
    const fw = collectPop(fwResults, c);
    const fl = collectPop(felResults, c);
    const w = welch(fw, fl);
    const verdict = w.reject99 ? 'NO (99%)' : w.reject95 ? 'NO (95%)' : 'yes';
    console.log(
      `<${c}>`.padEnd(15) +
      `      ${fmt(w.meanA, 3)} ± ${fmt(stddev(fw), 3)}` +
      `    ${fmt(w.meanB, 3)} ± ${fmt(stddev(fl), 3)}` +
      `   t=${fmt(w.t, 2).padStart(6)}` +
      `   p=${fmt(w.pValueTwoSided, 4).padStart(8)}` +
      `      ${verdict}`,
    );
  }

  console.log('');
  console.log('=== totals (created, absorbed-deaths) ===');
  for (const [label, extract] of [
    ['created  ', (r: RunResult) => r.totals.created],
    ['absorbed ', (r: RunResult) => r.totals.absorbed],
  ] as const) {
    const fw = collect(fwResults, extract as any);
    const fl = collect(felResults, extract as any);
    const w = welch(fw, fl);
    const verdict = w.reject95 ? 'NO (95%)' : 'yes';
    console.log(
      `${label}` +
      `   framework=${fmt(w.meanA, 1)} ± ${fmt(stddev(fw), 1)}` +
      `   fel=${fmt(w.meanB, 1)} ± ${fmt(stddev(fl), 1)}` +
      `   t=${fmt(w.t, 2)}   p=${fmt(w.pValueTwoSided, 4)}   ${verdict}`,
    );
  }

  // Persist replicate-level data for downstream tools.
  const outDir = path.resolve(__dirname, '..', '..', '..', 'out');
  fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'replicate-results.json');
  fs.writeFileSync(outPath, JSON.stringify({
    n: N, config, framework: fwResults, fel: felResults,
  }, null, 2));
  console.log(`\nartifacts written:\n  ${outPath}`);
}

main();
