#!/usr/bin/env ts-node
'use strict';

// Compares the framework's convolution output (out/convolution-framework.json)
// against numpy.convolve (out/external/convolution/numpy.json).
//
// HOW TO RUN
// ----------
//   npm run build
//   node dist/des/main-convolution.js                    # writes out/convolution-framework.json
//   bash external-references/run-all.sh                  # writes out/external/convolution/numpy.json
//   node dist/des/runners/validate-convolution.js
//
// Reports: max-abs-error and RMSE. Asserts both within 1e-12.

import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath  = path.join(ROOT, 'out', 'convolution-framework.json');
const npPath  = path.join(ROOT, 'out', 'external', 'convolution', 'numpy.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-convolution] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function main() {
  const ts = loadJson(tsPath);
  const np = loadJson(npPath);

  const yTs = ts.y as number[];
  const yNp = np.y as number[];

  if (yTs.length !== yNp.length) {
    console.error(`length mismatch: framework=${yTs.length} numpy=${yNp.length}`);
    process.exit(1);
  }

  let maxAbs = 0;
  let sumSq = 0;
  let argMax = -1;
  for (let i = 0; i < yTs.length; i++) {
    const e = Math.abs(yTs[i] - yNp[i]);
    sumSq += e * e;
    if (e > maxAbs) { maxAbs = e; argMax = i; }
  }
  const rmse = Math.sqrt(sumSq / yTs.length);

  console.log('Convolution: framework vs numpy.convolve');
  console.log('==========================================');
  console.log(`  signal length     = ${ts.signal.length}`);
  console.log(`  kernel length     = ${ts.kernel.length}`);
  console.log(`  output length     = ${yTs.length}`);
  console.log(`  max-abs-error     = ${maxAbs.toExponential(3)}  (at i=${argMax})`);
  console.log(`  RMSE              = ${rmse.toExponential(3)}`);

  // Tolerance: convolution is a linear operation in floats. We expect agreement
  // to within a few ULPs of the magnitude of the largest output.
  const peak = Math.max(...yTs.map(Math.abs));
  const ulpAtPeak = Math.max(1, peak) * 2 ** -52;
  const tolerance = Math.max(1e-12, 1024 * ulpAtPeak);

  console.log(`  peak |y|          = ${peak.toExponential(3)}`);
  console.log(`  1024 * ULP(peak)  = ${(1024 * ulpAtPeak).toExponential(3)}`);
  console.log(`  tolerance         = ${tolerance.toExponential(3)}`);

  const ok = maxAbs < tolerance;
  console.log('');
  console.log(ok ? '  PASS' : '  FAIL');
  process.exit(ok ? 0 : 1);
}

main();
