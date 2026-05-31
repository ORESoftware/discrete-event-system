#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-signal-processing.rs   (fn main)
// 1:1 file move. Small CLI demo of Z / Laplace / Fourier transform models.
//
// Conversion notes (file-specific):
//   - top-level main() -> fn main().
//   - complex outputs (formatComplex) -> num_complex::Complex<f64>.
//   - TransformRunResult shape -> struct (#[derive(Clone)]).
//   - use crate::des::general::signal_transforms.
// =============================================================================

// Small CLI demo for transform models. For JSON-driven runs use:
//   npm run from-json -- examples/z-transform-geometric.json

import {
  TransformRunResult,
  formatComplex,
  runFourierTransform,
  runLaplaceTransform,
  runZTransform,
} from './general/signal-transforms';

function printResult(result: TransformRunResult): void {
  console.log(`\n${result.kind.toUpperCase()} TRANSFORM`);
  console.log(`  ${result.convention}`);
  console.log(`  samples=${result.samples.length} points=${result.outputs.length}`);
  console.log(`  source=${result.entityFramework.sources.join(', ')} stations=${result.entityFramework.stations.join(' -> ')} sink=${result.entityFramework.sinks.join(', ')}`);
  for (const output of result.outputs) {
    console.log(`  ${output.label.padEnd(12)} ${formatComplex(output.value)}  |.|=${output.magnitude.toPrecision(6)}`);
  }
}

function main(): void {
  printResult(runZTransform({
    sequence: [1, 0.5, 0.25, 0.125, 0.0625],
    zValues: [
      {label: 'z=2', re: 2},
      {label: 'z=1', re: 1},
    ],
  }));

  printResult(runLaplaceTransform({
    expression: 'exp(-a*t)',
    constants: {a: 2},
    t0: 0,
    t1: 8,
    dt: 0.01,
    sValues: [
      {label: 's=1', re: 1},
      {label: 's=0.5+i', re: 0.5, im: 1},
    ],
  }));

  printResult(runFourierTransform({
    expression: 'sin(omega0*t)',
    constants: {omega0: 2},
    t0: 0,
    t1: 2 * Math.PI,
    dt: 2 * Math.PI / 2000,
    omegaValues: [0, 2, -2],
  }));
}

if (require.main === module) main();
