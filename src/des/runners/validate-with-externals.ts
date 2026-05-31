#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_with_externals.rs.
// - Keep this as a CLI validation binary with Result-returning main; map N/STEPSIZE/output dirs to clap/std::env.
// - Convert ExternalRun, per-kernel summaries, and Welch outputs into serde structs for golden comparison reports.
// - Keep external fixture loading at the std::fs/std::path boundary and reuse migrated runner/stat modules internally.
'use strict';

// =============================================================================
// validate-with-externals: side-by-side comparison of every kernel we have,
// including JSON results dropped by external Python tools (SimPy, Ciw, ...).
//
// Reads every <tool>/<seed>.json under out/external/, treats them as extra
// kernel columns, and shows splits + time-averaged populations + pairwise
// Welch t-tests against PerIndividual / FEL-individual / Gillespie / ODE.
//
// Run external tools first:
//
//   bash external-references/run-all.sh
//
// then this driver (which only looks at JSON files - it does not invoke the
// external Python interpreter itself).
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {DEFAULT_CONFIG, RunResult, COMPARTMENT_ORDER} from './types';
import {runFelOnce}            from './fel-runner';
import {runPerIndividualOnce}  from './per-individual-runner';
import {runGillespieOnce}      from './gillespie-runner';
import {runOdeOnce}            from './ode-runner';
import {mean, stddev, welch}   from './stats';

const N = parseInt(process.env.N ?? '20', 10);
const PI_STEPSIZE = parseFloat(process.env.STEPSIZE ?? '0.05');
const EXTERNAL_DIR = path.resolve(__dirname, '..', '..', '..', 'out', 'external');

const fmt = (n: number, d = 4) => Number.isFinite(n) ? n.toFixed(d) : String(n);
// Defensive: external runs of non-SEIR shape (e.g. backpropagation,
// court-mdp, electric-circuit, two-disease, …) lack splitProbs and
// timeAvgPopulations. Treat missing values as 0 so the table still renders.
const collectSplit = (rs: RunResult[], from: string, to: string) =>
  rs.map(r => r.splitProbs?.[from]?.[to] ?? 0);
const collectPop = (rs: RunResult[], c: string) =>
  rs.map(r => r.timeAvgPopulations?.[c] ?? 0);

interface ExternalRun {
  kernel: string;
  seed: number;
  totals: {created: number; absorbed: number};
  finalPopulations: Record<string, number>;
  transitionCounts: Record<string, Record<string, number>>;
  splitProbs: Record<string, Record<string, number>>;
  timeAvgPopulations: Record<string, number>;
  peakPopulations: Record<string, number>;
  elapsedMs: number;
}

function loadExternal(toolDir: string): ExternalRun[] {
  if (!fs.existsSync(toolDir)) return [];
  return fs.readdirSync(toolDir)
    .filter(f => f.endsWith('.json'))
    .map(f => JSON.parse(fs.readFileSync(path.join(toolDir, f), 'utf8')))
    // Only keep SEIR-shaped JSONs. Other models live under out/external/
    // (backpropagation, court-mdp, electric-circuit, two-disease, …) and
    // have their own dedicated validators.
    .filter(j => j && j.splitProbs && j.timeAvgPopulations && j.finalPopulations);
}

function toRunResult(e: ExternalRun): RunResult {
  return {
    kernel: e.kernel as any,
    config: DEFAULT_CONFIG,
    seed: e.seed,
    totals: e.totals,
    finalPopulations: e.finalPopulations,
    transitionCounts: e.transitionCounts,
    splitProbs: e.splitProbs,
    timeAvgPopulations: e.timeAvgPopulations,
    peakPopulations: e.peakPopulations,
    elapsedMs: e.elapsedMs,
  };
}

function main() {
  const cfg = {...DEFAULT_CONFIG, stepSize: PI_STEPSIZE};
  console.log(`validate-with-externals: PI stepSize=${PI_STEPSIZE}d   N=${N} reps (in-repo) | external runs read from ${EXTERNAL_DIR}`);
  console.log('');

  // Run in-repo kernels with N reps.
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
  const inRepoMs = Date.now() - t0;

  // Discover external tool runs.
  const externalDirs = fs.existsSync(EXTERNAL_DIR)
    ? fs.readdirSync(EXTERNAL_DIR).filter(d =>
        fs.statSync(path.join(EXTERNAL_DIR, d)).isDirectory())
    : [];
  const externals: Array<{name: string, runs: RunResult[]}> = [];
  for (const tool of externalDirs) {
    const runs = loadExternal(path.join(EXTERNAL_DIR, tool)).map(toRunResult);
    if (runs.length > 0) externals.push({name: tool, runs});
  }
  if (externals.length === 0) {
    console.log('NOTE: no external JSONs found under', EXTERNAL_DIR);
    console.log('      run `bash external-references/run-all.sh` first to populate them.');
    console.log('');
  }

  // Build the column list: in-repo first, then externals.
  const columns: Array<{name: string, runs?: RunResult[], single?: RunResult}> = [
    {name: 'PerIndividual',  runs: piRuns},
    {name: 'FEL-individual', runs: felRuns},
    {name: 'Gillespie SSA',  runs: ssaRuns},
    {name: 'ODE (det)',      single: ode},
  ];
  for (const ext of externals) columns.push({name: ext.name, runs: ext.runs});

  console.log(`in-repo wall: ${inRepoMs} ms`);
  for (const col of columns) {
    if (col.runs) {
      console.log(`  ${col.name.padEnd(18)} N=${col.runs.length.toString().padStart(3)}  ` +
                  `mean wall=${fmt(mean(col.runs.map(r => r.elapsedMs)), 1)} ms / rep`);
    } else if (col.single) {
      console.log(`  ${col.name.padEnd(18)} (deterministic)  wall=${col.single.elapsedMs} ms`);
    }
  }
  console.log('');

  const splits: Array<[string, string, number]> = [
    ['I-P', 'I-A', cfg.probabilities.asymptomaticShare],
    ['I-P', 'I-S', 1 - cfg.probabilities.asymptomaticShare],
    ['I-S', 'R',   1 - cfg.probabilities.hospitalizationGivenSymptom],
    ['I-S', 'I-H', cfg.probabilities.hospitalizationGivenSymptom],
    ['I-H', 'R',   1 - cfg.probabilities.caseFatalityGivenHospital],
    ['I-H', 'D',   cfg.probabilities.caseFatalityGivenHospital],
  ];

  const colWidth = 22;
  console.log('=== empirical branching probabilities ===');
  console.log('transition'.padEnd(14) + 'expected'.padStart(10) +
    columns.map(c => c.name.padStart(colWidth)).join(''));
  for (const [from, to, expected] of splits) {
    const cells = columns.map(col => {
      if (col.runs) {
        const xs = collectSplit(col.runs, from, to);
        return `${fmt(mean(xs), 4)} ± ${fmt(stddev(xs), 4)}`.padStart(colWidth);
      } else if (col.single) {
        return fmt(col.single.splitProbs[from]?.[to] ?? 0, 4).padStart(colWidth);
      }
      return ''.padStart(colWidth);
    });
    console.log((from + ' -> ' + to).padEnd(14) +
                fmt(expected, 4).padStart(10) + cells.join(''));
  }

  console.log('');
  console.log('=== time-averaged compartment populations ===');
  console.log('compartment'.padEnd(14) +
    columns.map(c => c.name.padStart(colWidth)).join(''));
  for (const c of COMPARTMENT_ORDER) {
    const cells = columns.map(col => {
      if (col.runs) {
        const xs = collectPop(col.runs, c);
        return `${fmt(mean(xs), 3)} ± ${fmt(stddev(xs), 3)}`.padStart(colWidth);
      } else if (col.single) {
        return fmt(col.single.timeAvgPopulations[c] ?? 0, 3).padStart(colWidth);
      }
      return ''.padStart(colWidth);
    });
    console.log(`<${c}>`.padEnd(14) + cells.join(''));
  }

  // Pairwise Welch t-tests vs FEL-individual (the canonical in-repo M/M/inf
  // reference). External tools should agree.
  if (externals.length > 0) {
    console.log('');
    console.log('=== pairwise Welch t-tests vs FEL-individual (populations) ===');
    const refRuns = felRuns;
    const others: Array<[string, RunResult[]]> = [
      ['PerIndividual ', piRuns],
      ['Gillespie SSA ', ssaRuns],
      ...externals.map(e => [e.name.padEnd(14), e.runs] as [string, RunResult[]]),
    ];
    console.log('compartment    ' + others.map(o => `${o[0]}  t (p)`.padStart(28)).join(''));
    for (const c of COMPARTMENT_ORDER) {
      const cells = others.map(([_, rs]) => {
        const w = welch(collectPop(rs, c), collectPop(refRuns, c));
        const verdict = w.reject99 ? 'NO99 ' : w.reject95 ? 'no95 ' : ' yes ';
        return `${fmt(w.t, 2).padStart(6)} (p=${fmt(w.pValueTwoSided, 3)}) ${verdict}`.padStart(28);
      });
      console.log(`<${c}>`.padEnd(14) + cells.join(''));
    }
  }

  console.log('');
  console.log('=== totals ===');
  console.log('metric'.padEnd(14) +
    columns.map(c => c.name.padStart(colWidth)).join(''));
  for (const [label, extract] of [
    ['created   ', (r: RunResult) => r.totals.created],
    ['absorbed D', (r: RunResult) => r.totals.absorbed],
  ] as const) {
    const cells = columns.map(col => {
      if (col.runs) {
        const xs: number[] = col.runs.map(r => (extract as (r: RunResult) => number)(r));
        return `${fmt(mean(xs), 1)} ± ${fmt(stddev(xs), 1)}`.padStart(colWidth);
      } else if (col.single) {
        return fmt((extract as (r: RunResult) => number)(col.single), 1).padStart(colWidth);
      }
      return ''.padStart(colWidth);
    });
    console.log(label.padEnd(14) + cells.join(''));
  }
}

main();
