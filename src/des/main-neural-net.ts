#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-neural-net.rs   (fn main)
// 1:1 file move. Thin runner: XOR net, neural Q-learning, and neural-ODE demos.
//
// Conversion notes (file-specific):
//   - top-level main() -> fn main(); process.env (SEED, XOR_EPOCHS, XOR_LR)
//     -> std::env::var; seed -> SeededRandom.
//   - delegates to general/neural-network + rl-environments -> use crate::des::
//     general::{neural_network, rl_environments}.
// =============================================================================

import {
  FeedForwardNetwork,
  runNeuralQLearningDES,
  runXorNeuralNetDES,
  solveNeuralODE,
} from './general/neural-network';
import {Corridor, evalPolicy} from './general/rl-environments';

function meanLast(xs: readonly number[], n: number): number {
  const tail = xs.slice(-Math.min(n, xs.length));
  return tail.reduce((s, x) => s + x, 0) / Math.max(1, tail.length);
}

function main(): void {
  const seed = Number(process.env.SEED ?? 7);

  console.log('# Neural-net DES demo');
  console.log(`# seed = ${seed}`);

  const xor = runXorNeuralNetDES({
    seed,
    epochs: Number(process.env.XOR_EPOCHS ?? 8000),
    learningRate: Number(process.env.XOR_LR ?? 0.3),
    hiddenLayers: [4],
  });
  console.log('');
  console.log('## Supervised XOR');
  console.log(`samples trained = ${xor.lossHistory.length}`);
  console.log(`ticks = ${xor.ticks} (${xor.reason})`);
  console.log(`avg loss last 100 = ${meanLast(xor.lossHistory, 100).toExponential(3)}`);
  console.log(`predictions [00, 01, 10, 11] = [${xor.predictions.map(v => v[0].toFixed(4)).join(', ')}]`);

  const env = new Corridor(6);
  const q = runNeuralQLearningDES(env, {
    numEpisodes: Number(process.env.Q_EPISODES ?? 600),
    maxStepsPerEpisode: 40,
    alpha: Number(process.env.Q_ALPHA ?? 0.25),
    gamma: 0.95,
    epsilon: 0.8,
    epsilonDecay: 0.99,
    epsilonMin: 0.02,
    seed,
  });
  const evalQ = evalPolicy(env, s => q.policy[s], {numEpisodes: 50, maxStepsPerEpisode: 40});
  console.log('');
  console.log('## Neural Q-learning on Corridor MDP');
  console.log(`episodes = ${q.totalEpisodes}, steps = ${q.totalSteps}, ticks = ${q.totalTicks}`);
  console.log(`greedy policy = [${q.policy.join(', ')}]`);
  console.log(`eval success = ${(100 * evalQ.successRate).toFixed(1)}%, mean length = ${evalQ.meanLength.toFixed(1)}`);
  console.log(`avg TD loss last 100 = ${meanLast(q.lossHistory, 100).toExponential(3)}`);

  const rate = 0.5;
  const odeNet = new FeedForwardNetwork([
    {weights: [[-rate]], biases: [0], activation: 'linear'},
  ]);
  const trace = solveNeuralODE(odeNet, {y0: [1], t0: 0, t1: 2, dt: 0.05, solver: 'rk4'});
  const final = trace.y[trace.y.length - 1][0];
  const exact = Math.exp(-rate * 2);
  console.log('');
  console.log('## Neural ODE');
  console.log(`dy/dt = -${rate} y, final = ${final.toFixed(6)}, exact = ${exact.toFixed(6)}, abs error = ${Math.abs(final - exact).toExponential(3)}`);
}

if (require.main === module) main();
