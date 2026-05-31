#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/reference/compare-epidemic.rs  (module des::reference::compare_epidemic)
// 1:1 file move. CLI BINARY: diff framework vs. FEL-reference event logs.
//
// Declarations → Rust:
//   const fmt (fn)            -> free fn
//   interface RunSummary      -> struct RunSummary { label, events, start, end, ... }
//   function summarize        -> free fn `fn summarize(label, file) -> RunSummary`
//   function compareRow       -> free fn (prints a comparison line)
//   function main()           -> `fn main()` of a `[[bin]]`/examples target
//
// Conversion notes (file-specific):
//   - ENTRY SCRIPT (shebang + `main()` at EOF + `process.argv`) -> a Rust binary;
//     `process.argv[2/3]` -> `std::env::args`.
//   - Event records are `Record<string, any>` -> typed `#[derive(Deserialize)]`
//     structs deserialized from the JSONL (avoid `any`). Same event shapes as logger.rs.
//   - `events.find(..)!` non-null assertions -> `Option` + `expect`.
//   - nested `Record<string, Record<string, number>>` -> `HashMap<String, HashMap<String,u64>>`.
//   - `Number.POSITIVE_INFINITY` -> `f64::INFINITY`; `padEnd/padStart/toFixed` -> `format!`.
//   - `Math.sqrt`/`Math.abs` -> `f64` methods; `Object.entries/values` -> map iteration.
// =============================================================================

// =============================================================================
// Side-by-side comparison: framework (no-FEL, station-driven) vs. classical
// FEL reference. Reads both JSONL event logs and computes metrics side-by-side
// with absolute and relative differences plus rough Monte-Carlo tolerance.
//
// Comparing two single runs is noisy. To get statistically defensible numbers
// you'd want N independent replications of each kernel and a t-test / KS-test
// on the result distributions. For the purposes of "did our framework get the
// model right?", a single run plus a tolerance band based on Poisson sqrt(N)
// is sufficient.
// =============================================================================

import * as path from 'path';
import {readEvents} from '../observability/logger';

const fmt = (n: number, digits = 3) =>
  Number.isFinite(n) ? n.toFixed(digits) : String(n);

interface RunSummary {
  label: string;
  events: Array<Record<string, any>>;
  start: Record<string, any>;
  end: Record<string, any>;
  transitions: Array<Record<string, any>>;
  ticks: Array<Record<string, any>>;
  totalsByDestination: Record<string, number>;
  splitsByFrom: Record<string, Record<string, number>>;
  timeAvgPopulations: Record<string, number>;
  peakPopulations: Record<string, number>;
}

function summarize(label: string, file: string): RunSummary {
  const events = readEvents(file);
  const start = events.find(e => e.kind === 'sim_start')!;
  const end   = events.find(e => e.kind === 'sim_end')!;
  const transitions = events.filter(e => e.kind === 'transition');
  const ticks       = events.filter(e => e.kind === 'tick');

  const totalsByDestination: Record<string, number> = {};
  const splitsByFrom: Record<string, Record<string, number>> = {};
  for (const t of transitions) {
    totalsByDestination[t.to] = (totalsByDestination[t.to] ?? 0) + 1;
    const row = splitsByFrom[t.from] ?? (splitsByFrom[t.from] = {});
    row[t.to] = (row[t.to] ?? 0) + 1;
  }

  // Time-averaged compartment populations (mean over all ticks).
  const compartments = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R'];
  const timeAvg: Record<string, number> = {};
  const peak: Record<string, number> = {};
  for (const c of compartments) {
    let sum = 0;
    let pk = 0;
    for (const tk of ticks) {
      const v = (tk.populations as Record<string, number>)[c] ?? 0;
      sum += v;
      if (v > pk) pk = v;
    }
    timeAvg[c] = ticks.length > 0 ? sum / ticks.length : 0;
    peak[c]    = pk;
  }

  return {
    label, events, start, end, transitions, ticks,
    totalsByDestination, splitsByFrom,
    timeAvgPopulations: timeAvg, peakPopulations: peak,
  };
}

function compareRow(label: string, a: number, b: number, digits = 3) {
  const diff = b - a;
  const rel = a === 0 ? (b === 0 ? 0 : Number.POSITIVE_INFINITY) : diff / a;
  const sigma = Math.sqrt(Math.max(a, 1)); // Poisson tolerance for counts
  const within = Math.abs(diff) <= 2.5 * sigma;
  const verdict = within ? 'OK' : 'WIDE';
  console.log(
    `  ${label.padEnd(38)} framework=${fmt(a, digits).padStart(10)}  ` +
    `fel=${fmt(b, digits).padStart(10)}  diff=${fmt(diff, digits).padStart(8)}  ` +
    `rel=${fmt(rel, 3).padStart(7)}  ${verdict}`,
  );
}

function main() {
  const fwFile  = process.argv[2] ?? path.resolve(__dirname, '..', '..', '..', 'out', 'epidemic-events.jsonl');
  const felFile = process.argv[3] ?? path.resolve(__dirname, '..', '..', '..', 'out', 'epidemic-events-fel.jsonl');

  console.log('================================================================');
  console.log('framework (no-FEL) vs. classical FEL reference');
  console.log(`  framework log: ${fwFile}`);
  console.log(`  fel log:       ${felFile}`);
  console.log('================================================================');

  const fw  = summarize('framework', fwFile);
  const fel = summarize('fel-ref',   felFile);

  // ------ totals ----------------------------------------------------------
  console.log('');
  console.log('--- totals (count-based, Poisson tolerance ~2.5 sigma) ---');
  compareRow('entities created (source -> S)',
    fw.totalsByDestination['S'] ?? 0,
    fel.totalsByDestination['S'] ?? 0, 0);
  compareRow('S-visits (anything -> S)',
    fw.transitions.filter(t => t.to === 'S').length,
    fel.transitions.filter(t => t.to === 'S').length, 0);
  compareRow('cumulative deaths (sink absorbs)',
    fw.end.totals.absorbed,
    fel.end.totals.absorbed, 0);
  compareRow('total transitions logged',
    fw.transitions.length,
    fel.transitions.length, 0);

  // ------ branching probabilities -----------------------------------------
  console.log('');
  console.log('--- empirical branching probabilities ---');
  const expectedSplits: Record<string, Record<string, number>> = {
    'I-P': {'I-A': fw.start.config.probabilities.asymptomaticShare,
            'I-S': 1 - fw.start.config.probabilities.asymptomaticShare},
    'I-S': {'R':   1 - fw.start.config.probabilities.hospitalizationGivenSymptom,
            'I-H': fw.start.config.probabilities.hospitalizationGivenSymptom},
    'I-H': {'R':   1 - fw.start.config.probabilities.caseFatalityGivenHospital,
            'D':   fw.start.config.probabilities.caseFatalityGivenHospital},
  };
  for (const [from, exp] of Object.entries(expectedSplits)) {
    for (const [to, p] of Object.entries(exp)) {
      const fwTot = Object.values(fw.splitsByFrom[from] ?? {}).reduce((a, b) => a + b, 0);
      const flTot = Object.values(fel.splitsByFrom[from] ?? {}).reduce((a, b) => a + b, 0);
      const fwHat = fwTot ? (fw.splitsByFrom[from]?.[to] ?? 0) / fwTot : 0;
      const flHat = flTot ? (fel.splitsByFrom[from]?.[to] ?? 0) / flTot : 0;
      console.log(
        `  ${(from + ' -> ' + to).padEnd(15)}  expected=${fmt(p, 3)}  ` +
        `framework=${fmt(fwHat, 3)} (n=${String(fwTot).padStart(4)})  ` +
        `fel=${fmt(flHat, 3)} (n=${String(flTot).padStart(4)})`,
      );
    }
  }

  // ------ time-averaged populations ---------------------------------------
  console.log('');
  console.log('--- time-averaged compartment populations (over all ticks) ---');
  for (const c of ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R']) {
    compareRow(`<${c}>(t)`, fw.timeAvgPopulations[c], fel.timeAvgPopulations[c], 2);
  }

  // ------ peak populations ------------------------------------------------
  console.log('');
  console.log('--- peak compartment populations ---');
  for (const c of ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R']) {
    compareRow(`max ${c}(t)`, fw.peakPopulations[c], fel.peakPopulations[c], 0);
  }

  console.log('');
  console.log('================================================================');
  console.log('verdicts: "OK" = within Poisson 2.5 sigma; "WIDE" = larger gap');
  console.log('(single-replicate comparison; for tighter bounds, run N reps)');
  console.log('================================================================');
}

main();
