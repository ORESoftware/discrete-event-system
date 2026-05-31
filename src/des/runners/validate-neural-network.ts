#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_neural_network.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert reference network payloads and CheckRow values into serde structs read with serde_json.
// - Treat NEURAL_NETWORK_REFERENCE_ID as an external adapter behind external_program using std::process or tokio::process.
'use strict';

// =============================================================================
// validate-neural-network.ts
//
// Runs the dependency-free Python neural-network reference through the sanctioned
// external-program helper, then cross-checks the framework's:
//   - XOR supervised training result
//   - neural Q-learning corridor policy against external value iteration
//   - neural ODE RK4 decay result
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {runExternalModule, repoRootFromRunner} from './external-program';
import {NEURAL_NETWORK_REFERENCE_ID} from './external-modules';
import {
  FeedForwardNetwork,
  runNeuralQLearningDES,
  runXorNeuralNetDES,
  solveNeuralODE,
} from '../general/neural-network';
import {Corridor, evalPolicy} from '../general/rl-environments';

const ROOT = repoRootFromRunner();
const OUT_PATH = path.join(ROOT, 'out', 'external', 'neural-network', 'reference.json');

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function main(): void {
  const ext = runExternalModule(NEURAL_NETWORK_REFERENCE_ID, {out: OUT_PATH});

  console.log('Neural-network: framework vs external Python reference');
  console.log('======================================================');
  console.log(`  external command: ${ext.command} ${ext.args.map(a => JSON.stringify(a)).join(' ')}`);
  if (ext.stdout.trim()) console.log(ext.stdout.trim());
  if (ext.stderr.trim()) console.error(ext.stderr.trim());
  if (ext.status !== 0) {
    console.error(`external reference exited with status ${ext.status}`);
    process.exit(1);
  }

  const ref = JSON.parse(fs.readFileSync(OUT_PATH, 'utf8'));

  console.log('');
  console.log('-- XOR supervised network --');
  const xor = runXorNeuralNetDES({
    seed: 7,
    epochs: 8000,
    learningRate: 0.3,
    hiddenLayers: [4],
  });
  const xorPred = xor.predictions.map(v => v[0]);
  const predDiff = maxAbsDiff(xorPred, ref.xor.predictions);
  const lossDiff = maxAbsDiff(
    xor.lossHistory.slice(-100),
    (ref.xor.lossHistory as number[]).slice(-100),
  );
  check('XOR predictions match external reference', predDiff < 1e-12,
        `max abs diff=${predDiff.toExponential(3)}`);
  check('XOR trailing losses match external reference', lossDiff < 1e-12,
        `max abs diff=${lossDiff.toExponential(3)}`);

  console.log('');
  console.log('-- Neural Q-learning corridor --');
  const env = new Corridor(6);
  const q = runNeuralQLearningDES(env, {
    numEpisodes: 600,
    maxStepsPerEpisode: 40,
    alpha: 0.25,
    gamma: 0.95,
    epsilon: 0.8,
    epsilonDecay: 0.99,
    epsilonMin: 0.02,
    seed: 1,
  });
  const evalQ = evalPolicy(env, s => q.policy[s], {numEpisodes: 50, maxStepsPerEpisode: 40});
  check('learned policy matches external optimal policy on nonterminal states',
        JSON.stringify(q.policy.slice(0, 5)) === JSON.stringify((ref.corridor.policy as number[]).slice(0, 5)),
        `learned=[${q.policy.join(', ')}], optimal=[${ref.corridor.policy.join(', ')}]`);
  check('learned greedy policy succeeds in evaluation', evalQ.successRate === 1,
        `success=${evalQ.successRate}`);

  console.log('');
  console.log('-- Neural ODE decay --');
  const net = new FeedForwardNetwork([
    {weights: [[-0.5]], biases: [0], activation: 'linear'},
  ]);
  const trace = solveNeuralODE(net, {y0: [1], t0: 0, t1: 2, dt: 0.05, solver: 'rk4'});
  const frameworkFinal = trace.y[trace.y.length - 1][0];
  const refFinal = ref.neuralOdeDecay.final as number;
  const finalDiff = Math.abs(frameworkFinal - refFinal);
  check('neural ODE final state matches external RK4', finalDiff < 1e-12,
        `diff=${finalDiff.toExponential(3)}`);
  check('neural ODE agrees with analytical decay', Math.abs(frameworkFinal - Math.exp(-1)) < 1e-7,
        `error=${Math.abs(frameworkFinal - Math.exp(-1)).toExponential(3)}`);

  console.log('');
  console.log('========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`validate-neural-network: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main();
