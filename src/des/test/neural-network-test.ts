'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/neural_network_test.rs   (integration test crate)
// 1:1 file move. Tests neural-net support across the hybrid boundary (supervised
// DES training, neural Q-learning, neural ODE). Keep the doc-block below.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - net weight init + training are stochastic -> a seeded rand::Rng so the
//     XOR / Q-learning / ODE results are reproducible.
//   - mean()/loss comparisons -> approx::assert_relative_eq!.
// =============================================================================

// =============================================================================
// test/neural-network-test.ts — neural-net support across the hybrid boundary:
// supervised DES training, neural Q-learning for MDPs, and neural ODE solves.
// =============================================================================

import {
  FeedForwardNetwork,
  NeuralODESolverStation,
  NeuralODESolveToken,
  NeuralODESolutionToken,
  NeuralPredictionSink,
  runNeuralQLearningDES,
  runXorNeuralNetDES,
  solveNeuralODE,
} from '../general/neural-network';
import {
  NeuralInferenceToken,
  NeuralNetworkStation,
  runIterativeDES,
} from '../general/des-base';
import {Corridor, evalPolicy} from '../general/rl-environments';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function mean(xs: readonly number[]): number {
  return xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
}

console.log('\n-- feed-forward network + supervised DES training --');
{
  const r = runXorNeuralNetDES({
    seed: 7,
    epochs: 8000,
    learningRate: 0.3,
    hiddenLayers: [4],
  });
  const first = mean(r.lossHistory.slice(0, 100));
  const last = mean(r.lossHistory.slice(-100));
  const y = r.predictions.map(v => v[0]);
  const classified = y[0] < 0.2 && y[1] > 0.8 && y[2] > 0.8 && y[3] < 0.2;
  check('XOR loss decreases by at least 10x', last < first / 10, `first=${first.toFixed(4)} last=${last.toFixed(4)}`);
  check('XOR predictions classify all four cases', classified, `[${y.map(v => v.toFixed(3)).join(', ')}]`);
  check('network reports positive parameter count', r.network.parameterCount() > 0, `${r.network.parameterCount()}`);
}

console.log('\n-- inference station queue semantics --');
{
  const net = new FeedForwardNetwork([
    {weights: [[2, -1]], biases: [0.5], activation: 'linear'},
  ]);
  const station = new NeuralNetworkStation('infer', net);
  const sink = new NeuralPredictionSink('sink');
  station.pipe(sink, NeuralNetworkStation.CH_PREDICTION, NeuralNetworkStation.CH_PREDICTION);
  station.take(new NeuralInferenceToken('a', [3, 4]), NeuralNetworkStation.CH_INFER);
  runIterativeDES([station, sink], {shuffle: false, maxTicks: 5});
  check('inference token emits one prediction token', sink.predictions.length === 1);
  check('prediction is numerically correct', Math.abs(sink.predictions[0].output[0] - 2.5) < 1e-12,
        `y=${sink.predictions[0]?.output[0]}`);
}

console.log('\n-- neural Q-learning over an MDP --');
{
  const env = new Corridor(6);
  const r = runNeuralQLearningDES(env, {
    numEpisodes: 600,
    maxStepsPerEpisode: 40,
    alpha: 0.25,
    gamma: 0.95,
    epsilon: 0.8,
    epsilonDecay: 0.99,
    epsilonMin: 0.02,
    seed: 1,
  });
  const e = evalPolicy(env, s => r.policy[s], {numEpisodes: 50, maxStepsPerEpisode: 40});
  check('neural Q-learning trained requested episodes', r.totalEpisodes === 600, `episodes=${r.totalEpisodes}`);
  check('greedy policy reaches goal on all eval episodes', e.successRate === 1, `success=${e.successRate}`);
  check('policy moves right at the start', r.policy[0] === 1, `policy=[${r.policy.join(', ')}]`);
  check('TD losses recorded', r.lossHistory.length === r.totalSteps, `losses=${r.lossHistory.length}, steps=${r.totalSteps}`);
}

console.log('\n-- neural ODE vector field --');
{
  const rate = 0.5;
  const net = new FeedForwardNetwork([
    {weights: [[-rate]], biases: [0], activation: 'linear'},
  ]);
  const trace = solveNeuralODE(net, {y0: [1], t0: 0, t1: 2, dt: 0.05, solver: 'rk4'});
  const final = trace.y[trace.y.length - 1][0];
  const exact = Math.exp(-rate * 2);
  check('neural ODE RK4 tracks exponential decay', Math.abs(final - exact) < 1e-7,
        `final=${final.toFixed(8)} exact=${exact.toFixed(8)}`);
}

console.log('\n-- neural ODE station queue semantics --');
{
  class SolutionSink extends NeuralPredictionSink {
    readonly solutions: NeuralODESolutionToken[] = [];
    override hasWork(): boolean { return this.inboxSizes().solution > 0; }
    override runTimeStep(): void {
      this.solutions.push(...(this as any).drain('solution') as NeuralODESolutionToken[]);
    }
  }

  const net = new FeedForwardNetwork([
    {weights: [[-1]], biases: [0], activation: 'linear'},
  ]);
  const solver = new NeuralODESolverStation('ode', net);
  const sink = new SolutionSink('solutions');
  solver.pipe(sink, NeuralODESolverStation.CH_SOLUTION, NeuralODESolverStation.CH_SOLUTION);
  solver.take(new NeuralODESolveToken('decay', {y0: [1], t0: 0, t1: 1, dt: 0.1}), NeuralODESolverStation.CH_SOLVE);
  runIterativeDES([solver, sink], {shuffle: false, maxTicks: 5});
  check('ODE solve token emits one solution', sink.solutions.length === 1);
  check('solution trace includes start and end', sink.solutions[0].trace.t.length > 2);
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`neural-network-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
