'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/nonlinear_optimization_test.rs   (integration test crate)
// 1:1 file move. Tests Newton / quasi-Newton / nonlinear-least-squares models
// via the des-registry, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) relative float comparison -> approx::assert_relative_eq!.
//   - throws(fn) -> assert on Result::Err / #[should_panic].
//   - async main()/await -> a plain sync #[test].
// =============================================================================

// =============================================================================
// Tests for Newton/quasi-Newton and nonlinear least-squares DES models.
// =============================================================================

import {getModel, runFromSpec} from '../general/des-registry';
import {
  runBFGSRosenbrock,
  runGaussNewtonCurveFit,
  runLevenbergMarquardtCurveFit,
  runNewtonRosenbrock,
} from '../general/nonlinear-optimization-models';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function close(a: number, b: number, tol = 1e-4): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

async function main(): Promise<void> {
console.log('\n-- newton-rosenbrock --');
{
  const r = runNewtonRosenbrock({});
  check('Newton reaches Rosenbrock x*=1', close(r.x[0], 1), `x0=${r.x[0]}`);
  check('Newton reaches Rosenbrock y*=1', close(r.x[1], 1), `x1=${r.x[1]}`);
  check('Newton objective is tiny', r.objective < 1e-12, `f=${r.objective}`);
  check('Newton uses optimization state tokens', r.topology.movables.includes('OptStateToken'));
  check('Newton source rejects invalid initial state before update station',
    throws(() => runNewtonRosenbrock({x0: [Number.NaN, 1]})));
}

console.log('\n-- bfgs-rosenbrock --');
{
  const r = runBFGSRosenbrock({});
  check('BFGS reaches Rosenbrock x*=1', close(r.x[0], 1), `x0=${r.x[0]}`);
  check('BFGS reaches Rosenbrock y*=1', close(r.x[1], 1), `x1=${r.x[1]}`);
  check('BFGS objective is tiny', r.objective < 1e-10, `f=${r.objective}`);
  check('BFGS uses optimization state tokens', r.topology.movables.includes('OptStateToken'));
}

console.log('\n-- nonlinear least squares --');
{
  const gn = runGaussNewtonCurveFit({});
  const lm = runLevenbergMarquardtCurveFit({});
  check('Gauss-Newton fits a near 2', close(gn.params[0], 2, 2e-3), `a=${gn.params[0]}`);
  check('Gauss-Newton fits b near -0.5', close(gn.params[1], -0.5, 6e-3), `b=${gn.params[1]}`);
  check('LM fits a near 2', close(lm.params[0], 2, 2e-3), `a=${lm.params[0]}`);
  check('LM fits b near -0.5', close(lm.params[1], -0.5, 6e-3), `b=${lm.params[1]}`);
  check('Gauss-Newton uses NLS state tokens', gn.topology.movables.includes('NLStateToken'));
  check('LM uses NLS state tokens', lm.topology.movables.includes('NLStateToken'));
}

console.log('\n-- registry smoke --');
{
  for (const id of ['newton-rosenbrock', 'bfgs-rosenbrock', 'gauss-newton-curve-fit', 'levenberg-marquardt-curve-fit']) {
    const reg = getModel(id);
    check(`registry has ${id}`, reg.id === id);
  }
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'levenberg-marquardt-curve-fit',
    parameters: {},
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('runFromSpec executes LM model', summary.modelId === 'levenberg-marquardt-curve-fit');
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`nonlinear-optimization-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
