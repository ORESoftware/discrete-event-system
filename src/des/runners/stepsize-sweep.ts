#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/stepsize_sweep.rs.
// - Keep this as a CLI sweep runner with Result-returning main; map N/STEPSIZES to clap/std::env parsers.
// - Convert sweep rows to serde/csv-serializable structs and keep SVG rendering as a private report helper.
// - Reuse migrated kernel runner modules and stats helpers; file output should use std::fs/std::path.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/stepsize-sweep.rs  (a `fn main` binary; an
//                    `examples/stepsize-sweep.rs` also works)
// 1:1 file move. Sweeps the framework kernel's stepSize vs the (stepSize-free)
// FEL reference; emits a markdown table, ASCII chart, and CSV/SVG artifacts.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env (`N`, `STEPSIZES` comma list) -> `std::env::var` + split/parse.
//   - `fs`/`path` writing CSV/SVG -> `std::fs` / `std::path`; any `JSON` -> serde.
//   - kernel seeding (`withSeed`/`Date.now`) -> `SeededRandom`/`Clock`.
//   - `console.log` -> `println!`.
// =============================================================================

// =============================================================================
// stepSize sweep: run the framework kernel at a range of step sizes (1.0, 0.5,
// 0.1, 0.05 days), compare its time-averaged compartment populations against
// the FEL reference (which is stepSize-independent). Visualises how the
// granularity-induced gap shrinks as stepSize -> 0, both as a markdown table
// and as an ASCII bar chart, plus persists the data as CSV / SVG.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {DEFAULT_CONFIG, COMPARTMENT_ORDER, RunResult} from './types';
import {runFrameworkOnce} from './framework-runner';
import {runFelOnce} from './fel-runner';
import {mean, stddev} from './stats';

const N = parseInt(process.env.N ?? '8', 10);
const STEP_SIZES = (process.env.STEPSIZES ?? '1.0,0.5,0.1,0.05')
  .split(',').map(s => parseFloat(s));

const fmt = (n: number, d = 3) => Number.isFinite(n) ? n.toFixed(d) : String(n);

interface SweepPoint {
  stepSize: number;
  // mean of run-level time-averaged populations, plus stddev across reps
  fwMean: Record<string, number>;
  fwSd:   Record<string, number>;
  felMean: Record<string, number>;
  felSd:   Record<string, number>;
  /** ratio = fw / fel for each compartment; 1.0 == perfect agreement */
  ratio: Record<string, number>;
  fwWallMs: number;
  felWallMs: number;
}

function asciiBars(label: string, values: number[], maxLen = 40): string {
  const max = Math.max(...values, 1);
  return values.map((v, i) =>
    `  ss=${fmt(STEP_SIZES[i], 2).padStart(5)}  ${label}=${fmt(v, 3).padStart(7)}  |${
      '#'.repeat(Math.round((v / max) * maxLen))
    }`,
  ).join('\n');
}

function main() {
  console.log(`stepsize-sweep.ts: ${N} reps per stepSize, sweeping ${JSON.stringify(STEP_SIZES)}`);

  // FEL is stepSize-independent; run it once with N reps for comparison.
  const felRunsPerStep: RunResult[] = [];
  const t0 = Date.now();
  for (let i = 0; i < N; i++) felRunsPerStep.push(runFelOnce(DEFAULT_CONFIG, {seed: 0x40000 + i}));
  const felWall = Date.now() - t0;

  const felMean: Record<string, number> = {};
  const felSd:   Record<string, number> = {};
  for (const c of COMPARTMENT_ORDER) {
    const xs = felRunsPerStep.map(r => r.timeAvgPopulations[c]);
    felMean[c] = mean(xs); felSd[c] = stddev(xs);
  }
  console.log(`fel reference: ${N} reps, total wall ${felWall} ms`);

  const sweep: SweepPoint[] = [];

  for (const ss of STEP_SIZES) {
    const cfg = {...DEFAULT_CONFIG, stepSize: ss};
    const tStart = Date.now();
    const reps: RunResult[] = [];
    for (let i = 0; i < N; i++) reps.push(runFrameworkOnce(cfg, {seed: 0x50000 + i}));
    const fwWall = Date.now() - tStart;

    const fwMean: Record<string, number> = {};
    const fwSd:   Record<string, number> = {};
    const ratio:  Record<string, number> = {};
    for (const c of COMPARTMENT_ORDER) {
      const xs = reps.map(r => r.timeAvgPopulations[c]);
      fwMean[c] = mean(xs);
      fwSd[c]   = stddev(xs);
      ratio[c]  = felMean[c] > 0 ? fwMean[c] / felMean[c] : NaN;
    }

    sweep.push({stepSize: ss, fwMean, fwSd, felMean, felSd, ratio,
                fwWallMs: fwWall, felWallMs: felWall});
    console.log(`  stepSize=${ss.toString().padStart(5)}  framework wall ${fwWall} ms ` +
                `(mean per rep ${fmt(fwWall / N, 1)} ms)`);
  }

  // ---- Markdown table ----------------------------------------------------
  console.log('');
  console.log('=== framework / fel time-averaged population ratios ===');
  console.log('(1.000 = perfect agreement; > 1 means framework over-estimates)');
  console.log('');
  const headerCells = ['stepSize'].concat(COMPARTMENT_ORDER);
  console.log(headerCells.map(h => h.padEnd(10)).join('  '));
  for (const sp of sweep) {
    const cells = [fmt(sp.stepSize, 3)].concat(
      COMPARTMENT_ORDER.map(c => fmt(sp.ratio[c], 3)));
    console.log(cells.map(s => s.padEnd(10)).join('  '));
  }

  // ---- Per-compartment <S>, <E>, <I-P> bar charts ------------------------
  console.log('');
  console.log('=== ASCII bar chart: framework <S>(t) vs stepSize ===');
  console.log(asciiBars('<S>',   sweep.map(s => s.fwMean['S'])));
  console.log(`  fel <S>=${fmt(felMean['S'], 3)}     <-- this is the target`);

  console.log('');
  console.log('=== ASCII bar chart: framework <E>(t) vs stepSize ===');
  console.log(asciiBars('<E>',   sweep.map(s => s.fwMean['E'])));
  console.log(`  fel <E>=${fmt(felMean['E'], 3)}     <-- this is the target`);

  console.log('');
  console.log('=== ASCII bar chart: framework <I-P>(t) vs stepSize ===');
  console.log(asciiBars('<I-P>', sweep.map(s => s.fwMean['I-P'])));
  console.log(`  fel <I-P>=${fmt(felMean['I-P'], 3)}     <-- this is the target`);

  console.log('');
  console.log('=== ASCII bar chart: framework / fel ratio for <I-P> ===');
  console.log(asciiBars('ratio<I-P>', sweep.map(s => s.ratio['I-P'])));

  // ---- Persist as CSV ----------------------------------------------------
  const outDir = path.resolve(__dirname, '..', '..', '..', 'out');
  fs.mkdirSync(outDir, {recursive: true});
  const csvPath = path.join(outDir, 'stepsize-sweep.csv');
  const cols = ['stepSize',
    ...COMPARTMENT_ORDER.map(c => `fw_${c}_mean`),
    ...COMPARTMENT_ORDER.map(c => `fw_${c}_sd`),
    ...COMPARTMENT_ORDER.map(c => `fel_${c}_mean`),
    ...COMPARTMENT_ORDER.map(c => `fel_${c}_sd`),
    ...COMPARTMENT_ORDER.map(c => `ratio_${c}`),
  ];
  const rows = sweep.map(sp => [
    sp.stepSize,
    ...COMPARTMENT_ORDER.map(c => sp.fwMean[c].toFixed(6)),
    ...COMPARTMENT_ORDER.map(c => sp.fwSd[c].toFixed(6)),
    ...COMPARTMENT_ORDER.map(c => sp.felMean[c].toFixed(6)),
    ...COMPARTMENT_ORDER.map(c => sp.felSd[c].toFixed(6)),
    ...COMPARTMENT_ORDER.map(c => sp.ratio[c].toFixed(6)),
  ]);
  fs.writeFileSync(csvPath, [cols.join(','), ...rows.map(r => r.join(','))].join('\n') + '\n');

  // ---- Persist as SVG ----------------------------------------------------
  const svg = renderSvg(sweep, felMean);
  const svgPath = path.join(outDir, 'stepsize-sweep.svg');
  fs.writeFileSync(svgPath, svg);

  console.log(`\nartifacts written:`);
  console.log(`  ${csvPath}`);
  console.log(`  ${svgPath}`);
}

function renderSvg(sweep: SweepPoint[], felMean: Record<string, number>): string {
  // Plot framework <S>, <E>, <I-P> mean populations vs stepSize on a log-x
  // axis, with FEL reference values shown as horizontal dashed lines.
  const W = 760, H = 420, PAD = 60;
  const compartments = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H'];
  const colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b'];

  const xs = sweep.map(s => Math.log10(s.stepSize));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const yvals: number[] = [];
  for (const c of compartments) {
    for (const sp of sweep) yvals.push(sp.fwMean[c]);
    yvals.push(felMean[c]);
  }
  const ymax = Math.max(...yvals, 0.001);

  const xToPx = (x: number) => PAD + (xmax === xmin ? W / 2 :
    ((x - xmin) / (xmax - xmin)) * (W - 2 * PAD));
  const yToPx = (y: number) => H - PAD - (y / ymax) * (H - 2 * PAD);

  let svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 ${W} ${H}' font-family='monospace' font-size='11'>`;
  svg += `<rect width='100%' height='100%' fill='white'/>`;
  // Axes
  svg += `<line x1='${PAD}' y1='${H - PAD}' x2='${W - PAD}' y2='${H - PAD}' stroke='black'/>`;
  svg += `<line x1='${PAD}' y1='${PAD}' x2='${PAD}' y2='${H - PAD}' stroke='black'/>`;
  // Title and axis labels
  svg += `<text x='${W / 2}' y='${PAD - 25}' text-anchor='middle' font-size='14' font-weight='bold'>Framework time-averaged compartment populations vs stepSize (FEL reference dashed)</text>`;
  svg += `<text x='${W / 2}' y='${H - 15}' text-anchor='middle'>log10(stepSize, days)</text>`;
  svg += `<text x='15' y='${H / 2}' text-anchor='middle' transform='rotate(-90 15 ${H / 2})'>mean population</text>`;
  // X-tick labels
  for (const sp of sweep) {
    const px = xToPx(Math.log10(sp.stepSize));
    svg += `<line x1='${px}' y1='${H - PAD}' x2='${px}' y2='${H - PAD + 5}' stroke='black'/>`;
    svg += `<text x='${px}' y='${H - PAD + 18}' text-anchor='middle'>${sp.stepSize}</text>`;
  }
  // Y-tick labels (5 evenly spaced)
  for (let i = 0; i <= 5; i++) {
    const yv = (ymax * i) / 5;
    const py = yToPx(yv);
    svg += `<line x1='${PAD - 5}' y1='${py}' x2='${PAD}' y2='${py}' stroke='black'/>`;
    svg += `<text x='${PAD - 8}' y='${py + 4}' text-anchor='end'>${yv.toFixed(2)}</text>`;
  }
  // Series
  compartments.forEach((c, i) => {
    const color = colors[i];
    const points = sweep.map(sp => `${xToPx(Math.log10(sp.stepSize))},${yToPx(sp.fwMean[c])}`);
    svg += `<polyline points='${points.join(' ')}' fill='none' stroke='${color}' stroke-width='2'/>`;
    for (const sp of sweep) {
      const cx = xToPx(Math.log10(sp.stepSize));
      const cy = yToPx(sp.fwMean[c]);
      svg += `<circle cx='${cx}' cy='${cy}' r='4' fill='${color}'/>`;
    }
    // Dashed FEL reference line
    const py = yToPx(felMean[c]);
    svg += `<line x1='${PAD}' y1='${py}' x2='${W - PAD}' y2='${py}' stroke='${color}' stroke-width='1' stroke-dasharray='5,3' opacity='0.6'/>`;
  });
  // Legend
  let lx = W - PAD - 110;
  let ly = PAD + 10;
  svg += `<rect x='${lx - 10}' y='${ly - 14}' width='120' height='${compartments.length * 16 + 8}' fill='white' stroke='#888'/>`;
  compartments.forEach((c, i) => {
    svg += `<line x1='${lx}' y1='${ly + i * 16 - 4}' x2='${lx + 18}' y2='${ly + i * 16 - 4}' stroke='${colors[i]}' stroke-width='2'/>`;
    svg += `<text x='${lx + 24}' y='${ly + i * 16}'><${c}>(t)  fel=${felMean[c].toFixed(3)}</text>`;
  });

  svg += `</svg>`;
  return svg;
}

main();
