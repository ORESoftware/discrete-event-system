'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/signal_transforms_test.rs   (integration test crate)
// 1:1 file move. Tests Z / Laplace / Fourier transform station graphs, so it is
// an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close()/complexClose({re,im}) comparisons -> approx::assert_relative_eq!;
//     {re,im} -> num_complex::Complex<f64>.
//   - async checkThrows()/await -> #[test] asserting Result::Err / #[should_panic].
// =============================================================================

// =============================================================================
// Tests for Z, Laplace, and Fourier transform DES station graphs.
// =============================================================================

import * as fs from 'fs';
import {getModel, runFromSpec} from '../general/des-registry';
import {
  TransformRunResult,
  runFourierTransform,
  runLaplaceTransform,
  runZTransform,
} from '../general/signal-transforms';

let pass = 0, fail = 0;

function check(label: string, ok: boolean, detail = ''): void {
  if (ok) {
    pass++;
    console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`);
  } else {
    fail++;
    console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`);
  }
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function complexClose(z: {re: number; im: number}, w: {re: number; im: number}, tol = 1e-8): boolean {
  return close(z.re, w.re, tol) && close(z.im, w.im, tol);
}

async function checkThrows(label: string, fn: () => unknown | Promise<unknown>, contains?: string): Promise<void> {
  let threw = false;
  let message = '';
  try {
    await fn();
  } catch (e) {
    threw = true;
    message = e instanceof Error ? e.message : String(e);
  }
  check(label, threw && (contains === undefined || message.includes(contains)), threw ? `message=${JSON.stringify(message)}` : 'did not throw');
}

async function main(): Promise<void> {
  console.log('\n[1] Z-transform station graph');
  {
    const z = runZTransform({
      sequence: [1, 2, 3],
      zValues: [
        {label: 'z=2', re: 2},
        {label: 'z=-1', re: -1},
      ],
    });
    check('1.1 validators pass', z.validation.every(c => c.passed));
    check('1.2 finite sequence X(2) equals direct Z-transform', complexClose(z.outputs[0].value, {re: 2.75, im: 0}, 1e-12), `observed=${z.outputs[0].value.re}`);
    check('1.3 finite sequence X(-1) equals direct Z-transform', complexClose(z.outputs[1].value, {re: 2, im: 0}, 1e-12), `observed=${z.outputs[1].value.re}`);
    check('1.4 topology exposes source/stations/sink', z.entityFramework.sources.length === 1 && z.entityFramework.stations.length === 2 && z.entityFramework.sinks.length === 1);
    check('1.5 movable entities are explicit', z.entityFramework.movableEntities.includes('TransformSampleToken') && z.entityFramework.movableEntities.includes('TransformContributionToken'));

    const geometric = runZTransform({
      expression: 'a^n',
      constants: {a: 0.5},
      terms: 4,
      zValues: [{label: 'z=2', re: 2}],
    });
    check('1.6 expression-generated geometric sequence is supported', close(geometric.outputs[0].value.re, 1.328125, 1e-12));
  }

  console.log('\n[2] Laplace transform station graph');
  {
    const laplace = runLaplaceTransform({
      expression: 'exp(-a*t)',
      constants: {a: 2},
      t0: 0,
      t1: 8,
      dt: 0.002,
      quadrature: 'trapezoid',
      sValues: [{label: 's=1', re: 1}],
    });
    const exactFiniteWindow = (1 - Math.exp(-24)) / 3;
    check('2.1 validators pass', laplace.validation.every(c => c.passed));
    check('2.2 Laplace transform approximates finite-window integral', close(laplace.outputs[0].value.re, exactFiniteWindow, 1e-6), `observed=${laplace.outputs[0].value.re}`);
    check('2.3 Laplace result is real for real exponential and real s', close(laplace.outputs[0].value.im, 0, 1e-9));
  }

  console.log('\n[3] Fourier transform station graph');
  {
    const dt = 2 * Math.PI / 2000;
    const fourier = runFourierTransform({
      expression: 'sin(omega0*t)',
      constants: {omega0: 2},
      t0: 0,
      t1: 2 * Math.PI,
      dt,
      quadrature: 'trapezoid',
      omegaValues: [0, 2, -2],
    });
    check('3.1 validators pass', fourier.validation.every(c => c.passed));
    check('3.2 DC component of one-period sine is near zero', complexClose(fourier.outputs[0].value, {re: 0, im: 0}, 1e-9), `observed=${JSON.stringify(fourier.outputs[0].value)}`);
    check('3.3 positive-frequency coefficient is -i*pi', complexClose(fourier.outputs[1].value, {re: 0, im: -Math.PI}, 1e-6), `observed=${JSON.stringify(fourier.outputs[1].value)}`);
    check('3.4 negative-frequency coefficient is +i*pi', complexClose(fourier.outputs[2].value, {re: 0, im: Math.PI}, 1e-6), `observed=${JSON.stringify(fourier.outputs[2].value)}`);
  }

console.log('\n[4] JSON registry integration');
{
    check('4.1 registry has z-transform', getModel('z-transform').id === 'z-transform');
    check('4.2 registry has laplace-transform', getModel('laplace-transform').id === 'laplace-transform');
    check('4.3 registry has fourier-transform', getModel('fourier-transform').id === 'fourier-transform');

    const csvPath = 'out/signal-transforms-test-z.csv';
    const htmlPath = 'out/signal-transforms-test-z.html';
    const framesPath = 'out/signal-transforms-test-z.frames.jsonl';
    const summary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'z-transform',
      parameters: {
        sequence: [1, 2, 3],
        zValues: [{label: 'z=2', re: 2}],
      },
      runtime: {verbose: false, outputs: {csv: csvPath, html: htmlPath, frames: framesPath}},
    }, {verbose: false});
    const result = summary.result as TransformRunResult;
    check('4.4 JSON run returns transform result', result.kind === 'z' && close(result.outputs[0].value.re, 2.75, 1e-12));
    check('4.5 JSON run writes CSV output', fs.readFileSync(csvPath, 'utf8').includes('z=2'));
    check('4.6 JSON run writes animation HTML', summary.outputs.some(o => o.kind === 'html' && o.path === htmlPath) && fs.readFileSync(htmlPath, 'utf8').includes('Z transform'));
    check('4.7 JSON run writes animation frames', summary.outputs.some(o => o.kind === 'frames' && o.path === framesPath) && fs.readFileSync(framesPath, 'utf8').includes('"kind":"animation-frame"'));

    const laplaceAnim = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'laplace-transform',
      parameters: {
        expression: 'exp(-t)',
        t0: 0,
        t1: 1,
        dt: 0.05,
        sValues: [{label: 's=1', re: 1}],
      },
      runtime: {verbose: false, outputs: {html: 'out/signal-transforms-test-laplace.html'}},
    }, {verbose: false});
    check('4.8 Laplace animation writes default frames output', laplaceAnim.outputs.some(o => o.kind === 'frames' && o.path === 'out/signal-transforms-test-laplace.frames.jsonl'));

    const fourierAnim = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'fourier-transform',
      parameters: {
        expression: 'sin(t)',
        t0: 0,
        t1: 6.283185307179586,
        dt: 0.06283185307179587,
        omegaValues: [1],
      },
      runtime: {verbose: false, outputs: {html: 'out/signal-transforms-test-fourier.html'}},
    }, {verbose: false});
    check('4.9 Fourier animation writes default frames output', fourierAnim.outputs.some(o => o.kind === 'frames' && o.path === 'out/signal-transforms-test-fourier.frames.jsonl'));
  }

  console.log('\n[5] Input validation');
  {
    await checkThrows('5.1 z-transform requires sequence or expression', () => runZTransform({
      zValues: [{re: 1}],
    }), 'requires either');
    await checkThrows('5.2 Laplace transform requires samples or expression', () => runLaplaceTransform({
      sValues: [{re: 1}],
    }), 'requires either');
    await checkThrows('5.3 Fourier transform requires samples or expression', () => runFourierTransform({
      omegaValues: [1],
    }), 'requires either');
    await checkThrows('5.4 JSON Zod schema rejects malformed complex points', () => runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'z-transform',
      parameters: {
        sequence: [1],
        zValues: [{label: 'bad', re: 'not-a-number'}],
      } as any,
      runtime: {verbose: false, animate: false},
    }, {verbose: false}), '$.zValues.0.re');
    await checkThrows('5.5 JSON Zod schema rejects unknown transform params', () => runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'fourier-transform',
      parameters: {
        expression: 'sin(t)',
        omegaValues: [1],
        unexpectedField: true,
      } as any,
      runtime: {verbose: false, animate: false},
    }, {verbose: false}), 'Unrecognized key');
  }

  console.log(`\nSignal transform tests complete: ${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
