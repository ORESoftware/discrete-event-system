// RUST MIGRATION: Target module `src/des/general/adapters/neural_network_adapters.rs`.
// RUST MIGRATION: Convert XOR, neural-Q, and neural-ODE adapter registrations into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Represent network configs, samples, traces, learned weights, and run results as `serde` config/result structs; output paths become `PathBuf`.
// RUST MIGRATION: Use `Result<_, ValidationError>` for layer-shape, sample, and hyperparameter validation.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/neural-network-adapters.rs
//   (module des::general::adapters::neural_network_adapters)
// 1:1 file move. Registers 3 neural-network JSON adapters: neural-xor,
// neural-qlearning-corridor, neural-ode-decay (each lazily imports its scene).
//
// Declarations → Rust:
//   interface NeuralXorParams / NeuralQCorridorParams / NeuralODEDecayParams /
//             NeuralODEDecayResult                 -> struct (#[derive(Deserialize)];
//             all-optional params -> Option fields + serde default)
//   type NeuralQCorridorResult = ReturnType<typeof runNeuralQLearningDES> & {eval}
//                                        -> a NAMED struct (base result + `eval` field)
//   const hiddenLayersSchema/neuralXorSchema/neuralQSchema/neuralOdeDecaySchema
//                                        -> serde + validator metadata
//   registerModel(...) x3                       -> one struct + impl ModelAdapter trait each
//   fn meanLast                          -> plain `fn` (slice tail mean)
//
// Conversion notes (file-specific):
//   - GotChA: `NeuralQCorridorResult` is `ReturnType<...> & {eval: {...}}` — name the
//     struct explicitly in Rust; `run` builds it via `{...r, eval: ...}` spread ->
//     construct the struct from the base run result plus an evalPolicy() call.
//   - `s => r.policy[s]` (greedy-policy closure passed to evalPolicy) -> a closure /
//     fn capturing the learned policy Vec.
//   - `solver: 'euler'|'heun'|'rk4'|'rk45'` literal union -> enum (NeuralODESolverName).
//   - All `?? default` chains -> Option::unwrap_or; `hiddenLayers && len>0 ? .. : [4]`
//     -> Option filter on non-empty then unwrap_or(vec![4]).
//   - GotChA: every `animate` uses dynamic `await import(...)` of FrameRecorder + the
//     neural-network-scene -> ordinary `use` imports (no lazy import in Rust).
//   - `network.parameterCount()` and FeedForwardNetwork construction are ported types.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {defaultFramesPath, writeCsvLines} from './adapter-utils';
import {
  FeedForwardNetwork,
  NeuralODESolverName,
  SupervisedNeuralNetDESResult,
  runNeuralQLearningDES,
  runXorNeuralNetDES,
  solveNeuralODE,
} from '../neural-network';
import {Corridor, evalPolicy} from '../rl-environments';
import {ODETrace} from '../ode';

// -----------------------------------------------------------------------------
// 1. neural-xor
// -----------------------------------------------------------------------------

interface NeuralXorParams {
  epochs?: number;
  learningRate?: number;
  seed?: number;
  hiddenLayers?: number[];
  samplesPerTick?: number;
  shuffleEachEpoch?: boolean;
}

const hiddenLayersSchema: ParamSchema = {
  kind: 'array',
  items: {kind: 'number', integer: true, min: 1},
  description: 'Hidden layer widths. Missing/empty uses [4].',
};

const neuralXorSchema: ParamSchema = {
  kind: 'object',
  description: 'XOR learned by a feed-forward neural network running as DES training stations.',
  fields: {
    epochs: {kind: 'number', integer: true, min: 1, default: 8000},
    learningRate: {kind: 'number', min: 0, default: 0.3},
    seed: {kind: 'number', integer: true, default: 7},
    hiddenLayers: hiddenLayersSchema,
    samplesPerTick: {kind: 'number', integer: true, min: 1, default: 1},
    shuffleEachEpoch: {kind: 'boolean', default: false},
  },
  required: [],
};

registerModel<NeuralXorParams, SupervisedNeuralNetDESResult<FeedForwardNetwork>>({
  id: 'neural-xor',
  description: 'Feed-forward neural net trained on XOR with queued DES sample tokens.',
  schema: neuralXorSchema,
  run(p) {
    return runXorNeuralNetDES({
      epochs: p.epochs,
      learningRate: p.learningRate,
      seed: p.seed,
      hiddenLayers: p.hiddenLayers && p.hiddenLayers.length > 0 ? p.hiddenLayers : [4],
      samplesPerTick: p.samplesPerTick,
      shuffleEachEpoch: p.shuffleEachEpoch,
    });
  },
  summarize(r, p) {
    const avg = meanLast(r.lossHistory, 100);
    const preds = r.predictions.map(v => v[0]);
    return [
      'NEURAL XOR (supervised DES training)',
      '────────────────────────────────────',
      `  Epochs:                 ${p.epochs ?? 8000}`,
      `  Hidden layers:          ${(p.hiddenLayers && p.hiddenLayers.length > 0 ? p.hiddenLayers : [4]).join(', ')}`,
      `  Samples trained:        ${r.lossHistory.length}`,
      `  Ticks:                  ${r.ticks} (${r.reason})`,
      `  Avg loss (last 100):    ${avg.toExponential(3)}`,
      `  XOR predictions:        [${preds.map(x => x.toFixed(4)).join(', ')}]`,
      `  Parameter count:        ${r.network.parameterCount()}`,
    ].join('\n');
  },
  async animate(r, _p, runtime) {
    const out = runtime.outputs ?? {};
    if (!out.html) return;
    const {FrameRecorder} = await import('../../animation/frame-recorder');
    const {NEURAL_STAGE_W, NEURAL_STAGE_H, buildNeuralXorAnimation} =
      await import('../../animation/scenes/neural-network-scene');
    const {frames, charts} = buildNeuralXorAnimation(r);
    const recorder = new FrameRecorder({
      framesPath: out.frames ?? defaultFramesPath(out.html),
      htmlPath: out.html,
      width: NEURAL_STAGE_W,
      height: NEURAL_STAGE_H,
      fps: 18,
      title: 'Neural XOR',
      subtitle: 'Feed-forward network trained by queued DES sample tokens',
      background: '#f8fafc',
    });
    for (const f of frames) recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
    recorder.setCharts(charts);
    await recorder.finish();
  },
});

// -----------------------------------------------------------------------------
// 2. neural-qlearning-corridor
// -----------------------------------------------------------------------------

interface NeuralQCorridorParams {
  length?: number;
  numEpisodes?: number;
  maxStepsPerEpisode?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  seed?: number;
  hiddenLayers?: number[];
}

const neuralQSchema: ParamSchema = {
  kind: 'object',
  description: 'Neural Q-learning on a small corridor MDP.',
  fields: {
    length: {kind: 'number', integer: true, min: 2, default: 6},
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 600},
    maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 40},
    alpha: {kind: 'number', min: 0, default: 0.25},
    gamma: {kind: 'number', min: 0, max: 1, default: 0.95},
    epsilon: {kind: 'number', min: 0, max: 1, default: 0.8},
    epsilonDecay: {kind: 'number', min: 0, max: 1, default: 0.99},
    epsilonMin: {kind: 'number', min: 0, max: 1, default: 0.02},
    seed: {kind: 'number', integer: true, default: 1},
    hiddenLayers: hiddenLayersSchema,
  },
  required: [],
};

type NeuralQCorridorResult = ReturnType<typeof runNeuralQLearningDES> & {
  eval: {meanReturn: number; meanLength: number; successRate: number};
};

registerModel<NeuralQCorridorParams, NeuralQCorridorResult>({
  id: 'neural-qlearning-corridor',
  description: 'Neural Q-learning agent learning a corridor MDP through DES environment tokens.',
  schema: neuralQSchema,
  run(p) {
    const env = new Corridor(p.length ?? 6);
    const r = runNeuralQLearningDES(env, {
      numEpisodes: p.numEpisodes ?? 600,
      alpha: p.alpha ?? 0.25,
      gamma: p.gamma ?? 0.95,
      epsilon: p.epsilon ?? 0.8,
      epsilonDecay: p.epsilonDecay ?? 0.99,
      epsilonMin: p.epsilonMin ?? 0.02,
      maxStepsPerEpisode: p.maxStepsPerEpisode ?? 40,
      seed: p.seed ?? 1,
      hiddenLayers: p.hiddenLayers && p.hiddenLayers.length > 0 ? p.hiddenLayers : [],
    });
    return {
      ...r,
      eval: evalPolicy(env, s => r.policy[s], {
        numEpisodes: 50,
        maxStepsPerEpisode: p.maxStepsPerEpisode ?? 40,
        gamma: p.gamma ?? 0.95,
      }),
    };
  },
  summarize(r, p) {
    return [
      'NEURAL Q-LEARNING (Corridor MDP)',
      '────────────────────────────────',
      `  Corridor length:         ${p.length ?? 6}`,
      `  Episodes:                ${r.totalEpisodes}`,
      `  Steps:                   ${r.totalSteps}`,
      `  Ticks:                   ${r.totalTicks}`,
      `  Greedy policy:           [${r.policy.join(', ')}]`,
      `  Eval success rate:       ${(100 * r.eval.successRate).toFixed(1)}%`,
      `  Eval mean length:        ${r.eval.meanLength.toFixed(1)}`,
      `  Avg TD loss (last 100):  ${meanLast(r.lossHistory, 100).toExponential(3)}`,
    ].join('\n');
  },
  async animate(r, p, runtime) {
    const out = runtime.outputs ?? {};
    if (!out.html) return;
    const {FrameRecorder} = await import('../../animation/frame-recorder');
    const {NEURAL_STAGE_W, NEURAL_STAGE_H, buildNeuralQCorridorAnimation} =
      await import('../../animation/scenes/neural-network-scene');
    const {frames, charts} = buildNeuralQCorridorAnimation(r, p.length ?? 6);
    const recorder = new FrameRecorder({
      framesPath: out.frames ?? defaultFramesPath(out.html),
      htmlPath: out.html,
      width: NEURAL_STAGE_W,
      height: NEURAL_STAGE_H,
      fps: 4,
      title: 'Neural Q-learning Corridor',
      subtitle: `Episodes=${r.totalEpisodes}, steps=${r.totalSteps}, success=${(100 * r.eval.successRate).toFixed(1)}%`,
      background: '#f8fafc',
    });
    for (const f of frames) recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
    recorder.setCharts(charts);
    await recorder.finish();
  },
});

// -----------------------------------------------------------------------------
// 3. neural-ode-decay
// -----------------------------------------------------------------------------

interface NeuralODEDecayParams {
  rate?: number;
  y0?: number;
  t1?: number;
  dt?: number;
  solver?: NeuralODESolverName;
}

interface NeuralODEDecayResult {
  trace: ODETrace;
  exactFinal: number;
  error: number;
}

const neuralOdeDecaySchema: ParamSchema = {
  kind: 'object',
  description: 'Solve y\' = -rate*y where the vector field is represented by a one-layer neural net.',
  fields: {
    rate: {kind: 'number', min: 0, default: 0.5},
    y0: {kind: 'number', default: 1},
    t1: {kind: 'number', min: 0, default: 2},
    dt: {kind: 'number', min: 1e-9, default: 0.05},
    solver: {kind: 'string', enum: ['euler', 'heun', 'rk4', 'rk45'], default: 'rk4'},
  },
  required: [],
};

registerModel<NeuralODEDecayParams, NeuralODEDecayResult>({
  id: 'neural-ode-decay',
  description: 'Neural ODE demo: a network supplies dy/dt and the existing ODE solver integrates it.',
  schema: neuralOdeDecaySchema,
  run(p) {
    const rate = p.rate ?? 0.5;
    const y0 = p.y0 ?? 1;
    const t1 = p.t1 ?? 2;
    const network = new FeedForwardNetwork([
      {weights: [[-rate]], biases: [0], activation: 'linear'},
    ]);
    const trace = solveNeuralODE(network, {
      y0: [y0],
      t0: 0,
      t1,
      dt: p.dt ?? 0.05,
      solver: p.solver ?? 'rk4',
    });
    const final = trace.y[trace.y.length - 1][0];
    const exactFinal = y0 * Math.exp(-rate * t1);
    return {trace, exactFinal, error: Math.abs(final - exactFinal)};
  },
  summarize(r, p) {
    const final = r.trace.y[r.trace.y.length - 1][0];
    return [
      'NEURAL ODE DECAY',
      '────────────────────────────────',
      `  Equation:                y' = -${p.rate ?? 0.5} y`,
      `  Solver:                  ${p.solver ?? 'rk4'}`,
      `  Steps recorded:          ${r.trace.t.length}`,
      `  Final y:                 ${final.toFixed(6)}`,
      `  Exact y:                 ${r.exactFinal.toFixed(6)}`,
      `  Abs error:               ${r.error.toExponential(3)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,y'];
    for (let i = 0; i < r.trace.t.length; i++) {
      lines.push(`${r.trace.t[i].toFixed(8)},${r.trace.y[i][0].toFixed(12)}`);
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(r, p, runtime) {
    const out = runtime.outputs ?? {};
    if (!out.html) return;
    const {FrameRecorder} = await import('../../animation/frame-recorder');
    const {NEURAL_STAGE_W, NEURAL_STAGE_H, buildNeuralOdeAnimation} =
      await import('../../animation/scenes/neural-network-scene');
    const {frames, charts} = buildNeuralOdeAnimation(r.trace, p.rate ?? 0.5, r.exactFinal, r.error);
    const recorder = new FrameRecorder({
      framesPath: out.frames ?? defaultFramesPath(out.html),
      htmlPath: out.html,
      width: NEURAL_STAGE_W,
      height: NEURAL_STAGE_H,
      fps: 12,
      title: 'Neural ODE Decay',
      subtitle: `Network vector field, solver=${p.solver ?? 'rk4'}, abs error=${r.error.toExponential(3)}`,
      background: '#f8fafc',
    });
    for (const f of frames) recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
    recorder.setCharts(charts);
    await recorder.finish();
  },
});

function meanLast(xs: readonly number[], n: number): number {
  const tail = xs.slice(-Math.min(n, xs.length));
  return tail.reduce((s, x) => s + x, 0) / Math.max(1, tail.length);
}
