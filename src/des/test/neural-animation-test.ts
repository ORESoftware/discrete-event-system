'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/neural_animation_test.rs   (integration test crate)
// 1:1 file move. Smoke-tests neural animation scenes built from neural-network
// runs, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - fs/os/path temp dirs -> the `tempfile` crate; JSONL roundtrip -> serde_json.
//   - async writeAnim/await -> plain sync #[test] unless the recorder is async.
//   - neural net weight init is stochastic -> a seeded rand::Rng for reproducible
//     scenes.
// =============================================================================

// =============================================================================
// test/neural-animation-test.ts — smoke tests for neural animation scenes.
// =============================================================================

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {FrameRecorder, readAnimation} from '../animation/frame-recorder';
import {
  buildNeuralOdeAnimation,
  buildNeuralQCorridorAnimation,
  buildNeuralXorAnimation,
  NEURAL_STAGE_H,
  NEURAL_STAGE_W,
} from '../animation/scenes/neural-network-scene';
import {
  FeedForwardNetwork,
  runNeuralQLearningDES,
  runXorNeuralNetDES,
  solveNeuralODE,
} from '../general/neural-network';
import {Corridor} from '../general/rl-environments';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

async function writeAnim(
  name: string,
  built: ReturnType<typeof buildNeuralXorAnimation>,
): Promise<{framesPath: string; htmlPath: string}> {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), `des-${name}-`));
  const framesPath = path.join(tmpDir, `${name}.frames.jsonl`);
  const htmlPath = path.join(tmpDir, `${name}.html`);
  const rec = new FrameRecorder({
    framesPath,
    htmlPath,
    width: NEURAL_STAGE_W,
    height: NEURAL_STAGE_H,
    fps: 12,
    title: name,
  });
  for (const f of built.frames) rec.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
  rec.setCharts(built.charts);
  await rec.finish();
  return {framesPath, htmlPath};
}

async function main(): Promise<void> {
  console.log('\n-- neural XOR animation --');
  const xor = runXorNeuralNetDES({seed: 7, epochs: 200, learningRate: 0.3, hiddenLayers: [4]});
  const xorAnim = buildNeuralXorAnimation(xor);
  check('XOR animation has frames', xorAnim.frames.length >= 20, `frames=${xorAnim.frames.length}`);
  check('XOR animation has charts', xorAnim.charts.length === 2);
  const xorOut = await writeAnim('neural-xor-test', xorAnim);
  const xorRead = readAnimation(xorOut.framesPath);
  check('XOR frames round-trip', xorRead.frames.length === xorAnim.frames.length);
  check('XOR HTML exists', fs.existsSync(xorOut.htmlPath));

  console.log('\n-- neural Q corridor animation --');
  const env = new Corridor(6);
  const q = runNeuralQLearningDES(env, {
    numEpisodes: 120,
    maxStepsPerEpisode: 40,
    alpha: 0.25,
    gamma: 0.95,
    epsilon: 0.8,
    epsilonDecay: 0.99,
    epsilonMin: 0.02,
    seed: 1,
  });
  const qAnim = buildNeuralQCorridorAnimation(q, 6);
  check('Q animation has rollout frames', qAnim.frames.length >= 2);
  check('Q animation has charts', qAnim.charts.length === 2);
  const qOut = await writeAnim('neural-q-test', qAnim);
  check('Q HTML contains Animation JSON', fs.readFileSync(qOut.htmlPath, 'utf8').includes('anim-data'));

  console.log('\n-- neural ODE animation --');
  const rate = 0.5;
  const net = new FeedForwardNetwork([{weights: [[-rate]], biases: [0], activation: 'linear'}]);
  const trace = solveNeuralODE(net, {y0: [1], t0: 0, t1: 2, dt: 0.05, solver: 'rk4'});
  const exact = Math.exp(-1);
  const odeAnim = buildNeuralOdeAnimation(trace, rate, exact, Math.abs(trace.y[trace.y.length - 1][0] - exact));
  check('ODE animation has one frame per trace point', odeAnim.frames.length === trace.t.length);
  check('ODE animation has chart', odeAnim.charts.length === 1);
  const odeOut = await writeAnim('neural-ode-test', odeAnim);
  const odeRead = readAnimation(odeOut.framesPath);
  check('ODE chart round-trip', !!odeRead.charts && odeRead.charts.length === 1);

  console.log('\n========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`neural-animation-test: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main();
