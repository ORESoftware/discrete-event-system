// RUST MIGRATION: Port file-for-file to `tests/learning_optimization_test.rs` as integration coverage for station-graph learning and optimization models.
// Test-port notes: translate learning scenarios into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, approximate-float helpers, and deterministic PRNG seeds.

'use strict';

// =============================================================================
// Tests for station-graph learning and optimization models.
// =============================================================================

import {getModel, runFromSpec} from '../general/des-registry';
import {
  runBackpropMLPClassifier,
  runLinearRegressionLS,
  runLogisticRegressionSGD,
  runRidgeRegressionLS,
} from '../general/learning-optimization-models';
import {runExpectedSarsaGridworld, runPolicyGradientCorridor} from '../general/rl-learning-models';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

async function main(): Promise<void> {
console.log('\n-- linear-regression-ls --');
{
  const r = runLinearRegressionLS({});
  check('fits y = 2x + 1 slope', close(r.coefficients[0], 2), `slope=${r.coefficients[0]}`);
  check('fits y = 2x + 1 intercept', close(r.intercept, 1), `intercept=${r.intercept}`);
  check('near-zero MSE on exact line', r.mse < 1e-20, `mse=${r.mse}`);
  check('uses stationary stations', r.topology.stations.includes('normal-equation-accumulator'));
  check('uses movable fit token', r.topology.movables.includes('RegressionFitToken'));
}

console.log('\n-- ridge-regression-ls --');
{
  const r = runRidgeRegressionLS({ridge: 0.01});
  check('ridge produces finite coefficients', r.coefficients.every(Number.isFinite) && Number.isFinite(r.intercept));
  check('ridge keeps default line fit close', close(r.coefficients[0], 2, 0.02), `slope=${r.coefficients[0]}`);
  check('ridge uses shared fit token', r.topology.movables.includes('RegressionFitToken'));
}

console.log('\n-- logistic-regression-sgd --');
{
  const r = runLogisticRegressionSGD({epochs: 160, batchSize: 3, learningRate: 0.2});
  check('records gradient steps', r.lossHistory.length > 0, `steps=${r.lossHistory.length}`);
  check('classifies separable defaults', r.accuracy === 1, `accuracy=${r.accuracy}`);
  check('loss is finite and small', Number.isFinite(r.finalLoss) && r.finalLoss < 0.2, `loss=${r.finalLoss}`);
  check('uses batch tokens', r.topology.movables.includes('VectorBatchToken'));
  check('uses gradient step tokens', r.topology.movables.includes('GradientStepToken'));
}

console.log('\n-- backprop-mlp-classifier --');
{
  const r = runBackpropMLPClassifier({hiddenUnits: 4, epochs: 800, batchSize: 4, learningRate: 0.08, optimizer: 'adam', seed: 7});
  check('records backprop gradient steps', r.lossHistory.length >= 800, `steps=${r.lossHistory.length}`);
  check('solves XOR defaults', r.accuracy === 1, `accuracy=${r.accuracy}, y=[${r.predictions.map(v => v.toFixed(3)).join(', ')}]`);
  check('final loss is finite', Number.isFinite(r.finalLoss), `loss=${r.finalLoss}`);
  check('backprop graph has explicit updater station', r.topology.stations.includes('backprop-gradient-update'));
  check('backprop graph uses sample/batch/gradient movables',
        ['VectorSampleToken', 'VectorBatchToken', 'GradientStepToken'].every(t => r.topology.movables.includes(t)));
}

console.log('\n-- policy-gradient-corridor --');
{
  const r = runPolicyGradientCorridor({numEpisodes: 300, rolloutLen: 12, alpha: 0.04, gamma: 0.95, seed: 1, length: 7});
  check('policy-gradient finishes requested episodes', r.rewardHistory.length === 300, `episodes=${r.rewardHistory.length}`);
  check('policy-gradient greedy policy solves corridor', r.greedySuccessRate === 1, `success=${r.greedySuccessRate}`);
  check('policy-gradient update station ran', r.updates > 0, `updates=${r.updates}`);
  check('policy-gradient graph uses train/resume movables',
        ['TrainTriggerToken', 'ResumeToken'].every(t => r.topology.movables.includes(t)));
}

console.log('\n-- expected-sarsa-gridworld --');
{
  const r = runExpectedSarsaGridworld({numEpisodes: 900, alpha: 0.2, gamma: 0.95, epsilon: 0.35, epsilonDecay: 0.995, seed: 1});
  check('expected SARSA finishes requested episodes', r.rewardHistory.length === 900, `episodes=${r.rewardHistory.length}`);
  check('expected SARSA greedy policy reaches goal', r.greedyReached, `len=${r.greedyLen}`);
  check('expected SARSA graph uses RL movables',
        ['StateToken', 'ActionToken', 'TransitionToken'].every(t => r.topology.movables.includes(t)));
}

console.log('\n-- registry smoke --');
{
  for (const id of [
    'linear-regression-ls',
    'ridge-regression-ls',
    'logistic-regression-sgd',
    'backprop-mlp-classifier',
    'policy-gradient-corridor',
    'expected-sarsa-gridworld',
  ]) {
    const reg = getModel(id);
    check(`registry has ${id}`, reg.id === id);
  }
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'backprop-mlp-classifier',
    parameters: {hiddenUnits: 4, epochs: 120, batchSize: 4, learningRate: 0.08, optimizer: 'adam', seed: 7},
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('runFromSpec executes backprop model', summary.modelId === 'backprop-mlp-classifier');
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`learning-optimization-test: ${passed}/${checks.length} checks passed.`);
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
