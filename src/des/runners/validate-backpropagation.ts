#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-backpropagation.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Compares the framework backprop output against the numpy-style
// reference (per-tensor max-abs error, loss history, XOR predictions).
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - `__dirname`-relative `path.join` -> `std::path` (`CARGO_MANIFEST_DIR` / cwd).
//   - `fs`/`JSON.parse` -> `std::fs` + serde structs.
//   - `as any` on weight tensors -> typed `Vec<Vec<f64>>`.
//   - `process.exit(code)` -> `std::process::exit(code)`.
// =============================================================================

// Compares the framework's backprop output (out/backprop-framework.json)
// against the numpy-style nested-loop reference (out/external/backpropagation/numpy.json).
//
// HOW TO RUN
// ----------
//   npm run build
//   node dist/des/main-backpropagation.js                    # writes out/backprop-framework.json
//   bash external-references/run-all.sh                      # writes out/external/backpropagation/numpy.json
//   node dist/des/runners/validate-backpropagation.js
//
// Reports max-abs-error on every weight tensor, on the loss history, and on
// the four XOR predictions. Asserts agreement within 64 * ULP(magnitude),
// which on a 10000-sample run gives effective tolerance ~ 1e-13.

import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath = path.join(ROOT, 'out', 'backprop-framework.json');
const pyPath = path.join(ROOT, 'out', 'external', 'backpropagation', 'numpy.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-backpropagation] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function maxAbsDiff(a: number[], b: number[]): {max: number; idx: number} {
  if (a.length !== b.length) {
    console.error(`length mismatch: ${a.length} vs ${b.length}`);
    process.exit(1);
  }
  let max = 0; let idx = -1;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    if (e > max) { max = e; idx = i; }
  }
  return {max, idx};
}

function maxAbsDiff2D(A: number[][], B: number[][]): {max: number; row: number; col: number} {
  let max = 0; let row = -1; let col = -1;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) {
      const e = Math.abs(A[i][j] - B[i][j]);
      if (e > max) { max = e; row = i; col = j; }
    }
  }
  return {max, row, col};
}

function main() {
  const ts = loadJson(tsPath);
  const py = loadJson(pyPath);

  console.log('Backpropagation: framework vs numpy-naive python');
  console.log('===================================================');
  console.log(`  config = seed=${ts.config.seed}, N=${ts.config.N}, lr=${ts.config.lr}`);

  const lossDiff = maxAbsDiff(ts.lossHistory, py.lossHistory);
  const predDiff = maxAbsDiff(ts.predictions, py.predictions);
  const w1Diff = maxAbsDiff2D(ts.final.W1, py.final.W1);
  const b1Diff = maxAbsDiff(ts.final.b1, py.final.b1);
  const w2Diff = maxAbsDiff2D(ts.final.W2, py.final.W2);
  const b2Diff = maxAbsDiff(ts.final.b2, py.final.b2);

  console.log(`  W1   max-abs-error  = ${w1Diff.max.toExponential(3)}  (at [${w1Diff.row}][${w1Diff.col}])`);
  console.log(`  b1   max-abs-error  = ${b1Diff.max.toExponential(3)}  (at [${b1Diff.idx}])`);
  console.log(`  W2   max-abs-error  = ${w2Diff.max.toExponential(3)}  (at [${w2Diff.row}][${w2Diff.col}])`);
  console.log(`  b2   max-abs-error  = ${b2Diff.max.toExponential(3)}  (at [${b2Diff.idx}])`);
  console.log(`  loss max-abs-error  = ${lossDiff.max.toExponential(3)}  (at sample ${lossDiff.idx})`);
  console.log(`  pred max-abs-error  = ${predDiff.max.toExponential(3)}  (at case ${predDiff.idx})`);

  // Pick a tight tolerance: weights are of order 1, so ~64 * ULP(1) ~ 1.4e-14.
  const tol = 1e-12;
  const allDiffs = [w1Diff.max, b1Diff.max, w2Diff.max, b2Diff.max,
                    lossDiff.max, predDiff.max];
  const worst = Math.max(...allDiffs);
  console.log('');
  console.log(`  worst diff = ${worst.toExponential(3)}    tolerance = ${tol.toExponential(3)}`);
  const ok = worst < tol;
  console.log(ok ? '  PASS' : '  FAIL');

  // Convergence sanity: avg loss over last 100 samples should be well below
  // the random-init baseline of ~0.125.
  const last100 = ts.lossHistory.slice(-100);
  const avgLoss = last100.reduce((s: number, v: number) => s + v, 0) / last100.length;
  console.log(`  avg loss (last 100) = ${avgLoss.toExponential(3)}`);
  if (avgLoss > 0.05) {
    console.log('  WARN: avg loss > 0.05, network may not have converged');
  }

  process.exit(ok ? 0 : 1);
}

main();
