// RUST MIGRATION: Target module `src/des/general/neural_network.rs`.
// RUST MIGRATION: Convert activation/solver-name unions to enums and configs, samples, options, traces, and results to `serde` structs.
// RUST MIGRATION: Port `FeedForwardNetwork`, supervised stations, Q-learning agent, ODE tokens/station, and prediction sink as structs implementing neural/RL/DES traits.
// RUST MIGRATION: Closure typedefs such as `StateEncoder` should become trait bounds or boxed `Fn` ports; exported runners can stay free functions unless graph-visible wrappers are needed.
// RUST MIGRATION: Replace `Math.random` call sites with an injected RNG trait/closure, and surface training/ODE validation failures as `Result`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/neural-network.rs  (module des::general::neural_network)
// 1:1 file move. Feed-forward MLP + supervised/Q-learning/neural-ODE DES stations.
//
// Declarations → Rust:
//   type ActivationName / NeuralODESolverName (string unions) -> enums
//   type StateEncoder<S> = (s: S) => NumericVector            -> Fn(S) -> Vec<f64> bound / boxed closure
//   interface DenseLayerConfig/SupervisedSample/SupervisedNeuralNetDESResult/XorNeuralNetOptions/
//             NeuralQLearningOptions/NeuralQLearningResult/NeuralODEOptions/ForwardTrace -> structs
//   class FeedForwardNetwork implements TrainableNeuralNetwork -> struct + impl trait
//   class SupervisedDatasetSource/NeuralODESolverStation/NeuralPredictionSink extends DESStation
//                                                              -> structs + impl DESStation trait
//   class NeuralQLearningAgent<S> extends RLAgentStation<S, number> -> struct + impl agent trait
//   class NeuralODESolveToken/NeuralODESolutionToken implements Token -> structs + impl Token
//   fn run*/solveNeuralODE/oneHotEncoder/argmax + const XOR_DATASET -> fns / const
//
// Conversion notes (file-specific):
//   - INJECT RNG: weight init / ε-greedy use `opts.rng ?? Math.random` -> take a `RandomSource`
//     (shared/capabilities); never default to the global inside the net.
//   - imports the ODE solvers (euler/rk2Heun/rk4/rk45) from ode.ts -> use crate::des::general::ode::*.
//   - weights/activations are `number[]`/`number[][]` -> `Vec<f64>`/`Vec<Vec<f64>>` (or ndarray).
//   - generic agent param <S> carries over; the encoder closure is the boundary to a typed state.
// =============================================================================
// general/neural-network.ts
//
// Feed-forward neural networks as DES components:
//   1. A small trainable MLP implementation for numerical forward/backward
//      passes.
//   2. Supervised-learning stations that receive samples as queued DES tokens.
//   3. A neural Q-learning agent for MDP/RL environments.
//   4. A neural ODE solver station that treats a network as dy/dt = f(t, y).
// =============================================================================

import {
  ChannelName,
  DESStation,
  EnvironmentStation,
  IterativeRunOptions,
  IterativeRunSummary,
  NeuralNetworkLike,
  NeuralNetworkStation,
  NeuralPredictionToken,
  NumericVector,
  PureEnvironment,
  RLAgentStation,
  SupervisedNeuralNetworkStation,
  SupervisedSampleToken,
  Token,
  TrainableNeuralNetwork,
  argMaxWithTieBreak,
  runIterativeDES,
} from './des-base';
import {euler, rk2Heun, rk4, rk45, ODETrace} from './ode';
import {mulberry32} from './prng';

export type ActivationName = 'linear' | 'sigmoid' | 'tanh' | 'relu';

export interface DenseLayerConfig {
  weights: number[][];   // [outDim][inDim]
  biases: number[];
  activation: ActivationName;
}

interface ForwardTrace {
  z: number[][];
  activations: number[][]; // activations[0] is the input
}

export class FeedForwardNetwork implements TrainableNeuralNetwork {
  readonly layers: DenseLayerConfig[];
  readonly inputDim: number;
  readonly outputDim: number;

  constructor(layers: DenseLayerConfig[]) {
    if (layers.length === 0) throw new Error('FeedForwardNetwork requires at least one layer');
    this.layers = layers.map(layer => ({
      weights: layer.weights.map(row => row.slice()),
      biases: layer.biases.slice(),
      activation: layer.activation,
    }));
    this.validateShape();
    this.inputDim = this.layers[0].weights[0].length;
    this.outputDim = this.layers[this.layers.length - 1].biases.length;
  }

  static random(opts: {
    inputDim: number;
    hiddenLayers?: number[];
    outputDim: number;
    hiddenActivation?: ActivationName;
    outputActivation?: ActivationName;
    rng?: () => number;
    weightScale?: number;
  }): FeedForwardNetwork {
    const rng = opts.rng ?? Math.random;
    const dims = [opts.inputDim, ...(opts.hiddenLayers ?? []), opts.outputDim];
    const layers: DenseLayerConfig[] = [];
    for (let k = 0; k < dims.length - 1; k++) {
      const fanIn = dims[k];
      const fanOut = dims[k + 1];
      const limit = opts.weightScale ?? Math.sqrt(6 / (fanIn + fanOut));
      layers.push({
        weights: Array.from({length: fanOut}, () =>
          Array.from({length: fanIn}, () => (2 * rng() - 1) * limit)),
        biases: new Array(fanOut).fill(0),
        activation: k === dims.length - 2
          ? (opts.outputActivation ?? 'linear')
          : (opts.hiddenActivation ?? 'tanh'),
      });
    }
    return new FeedForwardNetwork(layers);
  }

  clone(): FeedForwardNetwork {
    return new FeedForwardNetwork(this.toLayerConfigs());
  }

  toLayerConfigs(): DenseLayerConfig[] {
    return this.layers.map(layer => ({
      weights: layer.weights.map(row => row.slice()),
      biases: layer.biases.slice(),
      activation: layer.activation,
    }));
  }

  predict(input: NumericVector): NumericVector {
    return this.forward(input).activations[this.layers.length].slice();
  }

  trainSample(input: NumericVector, target: NumericVector, learningRate: number): {
    loss: number;
    prediction: NumericVector;
  } {
    if (learningRate < 0) throw new Error(`learningRate must be non-negative, got ${learningRate}`);
    this.assertVector(input, this.inputDim, 'input');
    this.assertVector(target, this.outputDim, 'target');

    const trace = this.forward(input);
    const prediction = trace.activations[this.layers.length].slice();
    let loss = 0;
    let dA = new Array<number>(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) {
      const e = prediction[i] - target[i];
      loss += 0.5 * e * e;
      dA[i] = e;
    }

    for (let k = this.layers.length - 1; k >= 0; k--) {
      const layer = this.layers[k];
      const prevA = trace.activations[k];
      const curA = trace.activations[k + 1];
      const curZ = trace.z[k];
      const delta = curA.map((a, i) =>
        dA[i] * activationPrimeFromOutput(layer.activation, a, curZ[i]));

      const dPrev = new Array<number>(prevA.length).fill(0);
      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          dPrev[j] += layer.weights[i][j] * delta[i];
        }
      }

      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] -= learningRate * delta[i] * prevA[j];
        }
        layer.biases[i] -= learningRate * delta[i];
      }
      dA = dPrev;
    }

    return {loss, prediction};
  }

  trainBatch(
    samples: ReadonlyArray<{input: NumericVector; target: NumericVector}>,
    learningRate: number,
  ): {meanLoss: number} {
    let total = 0;
    for (const s of samples) total += this.trainSample(s.input, s.target, learningRate).loss;
    return {meanLoss: total / Math.max(1, samples.length)};
  }

  parameterCount(): number {
    let n = 0;
    for (const layer of this.layers) {
      n += layer.biases.length;
      for (const row of layer.weights) n += row.length;
    }
    return n;
  }

  l2Norm(): number {
    let ss = 0;
    for (const layer of this.layers) {
      for (const b of layer.biases) ss += b * b;
      for (const row of layer.weights) for (const w of row) ss += w * w;
    }
    return Math.sqrt(ss);
  }

  private forward(input: NumericVector): ForwardTrace {
    this.assertVector(input, this.inputDim, 'input');
    const activations: number[][] = [input.slice()];
    const z: number[][] = [];
    let a = input.slice();
    for (const layer of this.layers) {
      const zk = new Array<number>(layer.biases.length);
      const ak = new Array<number>(layer.biases.length);
      for (let i = 0; i < layer.biases.length; i++) {
        let zi = layer.biases[i];
        for (let j = 0; j < layer.weights[i].length; j++) zi += layer.weights[i][j] * a[j];
        zk[i] = zi;
        ak[i] = activate(layer.activation, zi);
      }
      z.push(zk);
      activations.push(ak);
      a = ak;
    }
    return {z, activations};
  }

  private validateShape(): void {
    let prevOut: number | null = null;
    for (let k = 0; k < this.layers.length; k++) {
      const layer = this.layers[k];
      if (layer.biases.length === 0) throw new Error(`layer ${k}: biases cannot be empty`);
      if (layer.weights.length !== layer.biases.length) {
        throw new Error(`layer ${k}: weights rows must equal biases length`);
      }
      const width = layer.weights[0]?.length;
      if (!width || width < 1) throw new Error(`layer ${k}: weights rows cannot be empty`);
      for (const row of layer.weights) {
        if (row.length !== width) throw new Error(`layer ${k}: ragged weight matrix`);
      }
      if (prevOut !== null && width !== prevOut) {
        throw new Error(`layer ${k}: input dim ${width} does not match previous output dim ${prevOut}`);
      }
      prevOut = layer.biases.length;
    }
  }

  private assertVector(v: NumericVector, dim: number, name: string): void {
    if (v.length !== dim) throw new Error(`${name} dim ${v.length} does not match expected ${dim}`);
    for (const x of v) if (!Number.isFinite(x)) throw new Error(`${name} contains non-finite value ${x}`);
  }
}

function activate(name: ActivationName, z: number): number {
  switch (name) {
    case 'linear': return z;
    case 'sigmoid': return 1 / (1 + Math.exp(-z));
    case 'tanh': return Math.tanh(z);
    case 'relu': return z > 0 ? z : 0;
  }
}

function activationPrimeFromOutput(name: ActivationName, a: number, z: number): number {
  switch (name) {
    case 'linear': return 1;
    case 'sigmoid': return a * (1 - a);
    case 'tanh': return 1 - a * a;
    case 'relu': return z > 0 ? 1 : 0;
  }
}

// -----------------------------------------------------------------------------
// Supervised learning over DES queues.
// -----------------------------------------------------------------------------

export interface SupervisedSample {
  input: NumericVector;
  target: NumericVector;
}

export class SupervisedDatasetSource extends DESStation {
  static readonly CH_TRAIN: ChannelName = 'train';

  private epoch = 0;
  private cursor = 0;
  private emitted = 0;
  private order: number[];

  constructor(
    id: string,
    private readonly dataset: SupervisedSample[],
    private readonly opts: {
      epochs: number;
      samplesPerTick?: number;
      shuffleEachEpoch?: boolean;
      rng?: () => number;
    },
  ) {
    super(id);
    if (dataset.length === 0) throw new Error('SupervisedDatasetSource requires at least one sample');
    this.order = Array.from({length: dataset.length}, (_, i) => i);
    if (opts.shuffleEachEpoch) this.shuffleOrder();
  }

  override hasWork(): boolean {
    return this.epoch < this.opts.epochs;
  }

  runTimeStep(): void {
    const n = this.opts.samplesPerTick ?? 1;
    for (let i = 0; i < n; i++) {
      const next = this.nextSample();
      if (!next) return;
      this.emit(
        new SupervisedSampleToken(
          `sample-${this.emitted}`,
          next.sample.input.slice(),
          next.sample.target.slice(),
          {epoch: next.epoch, index: next.index},
        ),
        SupervisedDatasetSource.CH_TRAIN,
      );
      this.emitted += 1;
    }
  }

  getEmittedCount(): number { return this.emitted; }

  private nextSample(): {sample: SupervisedSample; index: number; epoch: number} | null {
    if (this.epoch >= this.opts.epochs) return null;
    if (this.cursor >= this.dataset.length) {
      this.epoch += 1;
      this.cursor = 0;
      if (this.epoch >= this.opts.epochs) return null;
      if (this.opts.shuffleEachEpoch) this.shuffleOrder();
    }
    const index = this.order[this.cursor++];
    return {sample: this.dataset[index], index, epoch: this.epoch};
  }

  private shuffleOrder(): void {
    const rng = this.opts.rng ?? Math.random;
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
  }
}

export interface SupervisedNeuralNetDESResult<N extends TrainableNeuralNetwork = TrainableNeuralNetwork> {
  network: N;
  lossHistory: readonly number[];
  predictions: NumericVector[];
  ticks: number;
  reason: IterativeRunSummary['reason'];
}

export function runSupervisedNeuralNetDES<N extends TrainableNeuralNetwork>(opts: {
  network: N;
  dataset: SupervisedSample[];
  epochs: number;
  learningRate: number;
  seed?: number;
  samplesPerTick?: number;
  shuffleEachEpoch?: boolean;
  snapshotEvery?: number;
  desOptions?: IterativeRunOptions;
}): SupervisedNeuralNetDESResult<N> {
  const rng = mulberry32(opts.seed ?? 1);
  const source = new SupervisedDatasetSource('dataset', opts.dataset, {
    epochs: opts.epochs,
    samplesPerTick: opts.samplesPerTick ?? 1,
    shuffleEachEpoch: opts.shuffleEachEpoch ?? false,
    rng,
  });
  const trainer = new SupervisedNeuralNetworkStation<N>('nn', opts.network, {
    learningRate: opts.learningRate,
    snapshotEvery: opts.snapshotEvery,
  });
  source.pipe(trainer, SupervisedDatasetSource.CH_TRAIN, SupervisedNeuralNetworkStation.CH_TRAIN);

  const maxTicks = opts.epochs * Math.ceil(opts.dataset.length / (opts.samplesPerTick ?? 1)) + 1000;
  const summary = runIterativeDES([source, trainer], {
    rng,
    shuffle: false,
    maxTicks,
    ...opts.desOptions,
  });

  return {
    network: opts.network,
    lossHistory: trainer.lossHistory,
    predictions: opts.dataset.map(s => opts.network.predict(s.input)),
    ticks: summary.ticks,
    reason: summary.reason,
  };
}

export const XOR_DATASET: SupervisedSample[] = [
  {input: [0, 0], target: [0]},
  {input: [0, 1], target: [1]},
  {input: [1, 0], target: [1]},
  {input: [1, 1], target: [0]},
];

export interface XorNeuralNetOptions {
  epochs?: number;
  learningRate?: number;
  seed?: number;
  hiddenLayers?: number[];
  samplesPerTick?: number;
  shuffleEachEpoch?: boolean;
}

export function runXorNeuralNetDES(opts: XorNeuralNetOptions = {}):
  SupervisedNeuralNetDESResult<FeedForwardNetwork> {
  const seed = opts.seed ?? 7;
  const rng = mulberry32(seed);
  const network = FeedForwardNetwork.random({
    inputDim: 2,
    hiddenLayers: opts.hiddenLayers ?? [4],
    outputDim: 1,
    hiddenActivation: 'tanh',
    outputActivation: 'sigmoid',
    rng,
  });
  return runSupervisedNeuralNetDES({
    network,
    dataset: XOR_DATASET,
    epochs: opts.epochs ?? 8000,
    learningRate: opts.learningRate ?? 0.3,
    seed,
    samplesPerTick: opts.samplesPerTick ?? 1,
    shuffleEachEpoch: opts.shuffleEachEpoch ?? false,
  });
}

// -----------------------------------------------------------------------------
// Neural Q-learning: RLAgentStation with a neural function approximator.
// -----------------------------------------------------------------------------

export type StateEncoder<S = number> = (state: S) => NumericVector;

export interface NeuralQLearningOptions<S = number> {
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonMin?: number;
  epsilonDecay?: number;
  numActions: number;
  stateEncoder: StateEncoder<S>;
  rng: () => number;
}

export class NeuralQLearningAgent<S = number> extends RLAgentStation<S, number> {
  readonly lossHistory: number[] = [];
  readonly tdErrorHistory: number[] = [];
  private currentEpsilon: number;

  constructor(
    id: string,
    private readonly network: TrainableNeuralNetwork,
    private readonly opts: NeuralQLearningOptions<S>,
  ) {
    super(id, {rng: opts.rng});
    this.currentEpsilon = opts.epsilon;
  }

  override assertPreconditions(): void {
    if (this.network.outputDim !== this.opts.numActions) {
      throw new Error(`network outputDim ${this.network.outputDim} must equal numActions ${this.opts.numActions}`);
    }
  }

  protected pickAction(state: S, rng: () => number): number {
    if (rng() < this.currentEpsilon) return Math.floor(rng() * this.opts.numActions);
    return argmax(this.network.predict(this.opts.stateEncoder(state)), rng);
  }

  protected update(state: S, action: number, reward: number, nextState: S, done: boolean): void {
    const x = this.opts.stateEncoder(state);
    const q = this.network.predict(x);
    const oldQ = q[action];
    const qNext = done ? [] : this.network.predict(this.opts.stateEncoder(nextState));
    const target = q.slice();
    target[action] = reward + (done ? 0 : this.opts.gamma * Math.max(...qNext));
    const r = this.network.trainSample(x, target, this.opts.alpha);
    this.lossHistory.push(r.loss);
    this.tdErrorHistory.push(target[action] - oldQ);
  }

  protected override endOfEpisode(_id: number): void {
    if (this.opts.epsilonDecay !== undefined) {
      this.currentEpsilon = Math.max(
        this.opts.epsilonMin ?? 0,
        this.currentEpsilon * this.opts.epsilonDecay,
      );
    }
  }

  predictQ(state: S): NumericVector {
    return this.network.predict(this.opts.stateEncoder(state));
  }

  getEpsilon(): number { return this.currentEpsilon; }

  getNetwork(): TrainableNeuralNetwork { return this.network; }
}

export interface NeuralQLearningResult {
  network: TrainableNeuralNetwork;
  policy: number[];
  qValues: number[][];
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  lossHistory: readonly number[];
  tdErrorHistory: readonly number[];
  totalEpisodes: number;
  totalSteps: number;
  totalTicks: number;
}

export function oneHotEncoder(numStates: number): StateEncoder<number> {
  return (state: number) => {
    const x = new Array<number>(numStates).fill(0);
    if (state < 0 || state >= numStates || !Number.isInteger(state)) {
      throw new Error(`state ${state} is outside [0, ${numStates})`);
    }
    x[state] = 1;
    return x;
  };
}

export function runNeuralQLearningDES(env: PureEnvironment<number, number>, opts: {
  numEpisodes: number;
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonMin?: number;
  epsilonDecay?: number;
  maxStepsPerEpisode?: number;
  seed?: number;
  network?: TrainableNeuralNetwork;
  hiddenLayers?: number[];
  hiddenActivation?: ActivationName;
  stateEncoder?: StateEncoder<number>;
  desOptions?: IterativeRunOptions;
}): NeuralQLearningResult {
  const rng = mulberry32(opts.seed ?? 1);
  const encoder = opts.stateEncoder ?? oneHotEncoder(env.numStates);
  const network = opts.network ?? FeedForwardNetwork.random({
    inputDim: env.numStates,
    hiddenLayers: opts.hiddenLayers ?? [],
    outputDim: env.numActions,
    hiddenActivation: opts.hiddenActivation ?? 'tanh',
    outputActivation: 'linear',
    rng,
    weightScale: 0.01,
  });
  const agent = new NeuralQLearningAgent<number>('neural-q-agent', network, {
    alpha: opts.alpha,
    gamma: opts.gamma,
    epsilon: opts.epsilon,
    epsilonMin: opts.epsilonMin,
    epsilonDecay: opts.epsilonDecay,
    numActions: env.numActions,
    stateEncoder: encoder,
    rng,
  });
  const envSt = new EnvironmentStation<number, number>('env', env, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: opts.maxStepsPerEpisode,
  });
  envSt.pipe(agent, EnvironmentStation.CH_STATE, RLAgentStation.CH_STATE);
  envSt.pipe(agent, EnvironmentStation.CH_TRANSITION, RLAgentStation.CH_TRANSITION);
  agent.pipe(envSt, RLAgentStation.CH_ACTION, EnvironmentStation.CH_ACTION);

  const summary = runIterativeDES([envSt, agent], {rng, ...opts.desOptions});
  const qValues = Array.from({length: env.numStates}, (_, s) => agent.predictQ(s));
  const policy = qValues.map(row => argmax(row, rng));
  return {
    network,
    policy,
    qValues,
    rewardHistory: agent.rewardHistory,
    lengthHistory: agent.lengthHistory,
    lossHistory: agent.lossHistory,
    tdErrorHistory: agent.tdErrorHistory,
    totalEpisodes: agent.rewardHistory.length,
    totalSteps: agent.totalSteps,
    totalTicks: summary.ticks,
  };
}

function argmax(row: readonly number[], rng: () => number): number {
  return argMaxWithTieBreak(row, rng);
}

// -----------------------------------------------------------------------------
// Neural ODE support: use a network as the vector field dy/dt = f(t, y).
// -----------------------------------------------------------------------------

export type NeuralODESolverName = 'euler' | 'heun' | 'rk4' | 'rk45';

export interface NeuralODEOptions {
  y0: NumericVector;
  t0: number;
  t1: number;
  dt: number;
  solver?: NeuralODESolverName;
  includeTime?: boolean;
  rk45?: {
    rtol?: number;
    atol?: number;
    hInit?: number;
    hMin?: number;
    hMax?: number;
    maxSteps?: number;
  };
}

export function solveNeuralODE(network: NeuralNetworkLike, opts: NeuralODEOptions): ODETrace {
  const inputDim = opts.includeTime ? opts.y0.length + 1 : opts.y0.length;
  if (network.inputDim !== inputDim) {
    throw new Error(`neural ODE network inputDim ${network.inputDim} must equal ${inputDim}`);
  }
  if (network.outputDim !== opts.y0.length) {
    throw new Error(`neural ODE network outputDim ${network.outputDim} must equal state dim ${opts.y0.length}`);
  }
  const rhs = (t: number, y: number[]) => {
    const input = opts.includeTime ? [t, ...y] : y;
    return network.predict(input);
  };
  switch (opts.solver ?? 'rk4') {
    case 'euler': return euler(rhs, opts.y0, opts.t0, opts.t1, opts.dt);
    case 'heun': return rk2Heun(rhs, opts.y0, opts.t0, opts.t1, opts.dt);
    case 'rk4': return rk4(rhs, opts.y0, opts.t0, opts.t1, opts.dt);
    case 'rk45': return rk45(rhs, opts.y0, opts.t0, opts.t1, {
      hInit: opts.dt,
      hMax: opts.dt,
      ...(opts.rk45 ?? {}),
    });
  }
}

export class NeuralODESolveToken implements Token {
  constructor(
    public readonly id: string,
    public readonly options: NeuralODEOptions,
  ) {}
}

export class NeuralODESolutionToken implements Token {
  constructor(
    public readonly id: string,
    public readonly trace: ODETrace,
  ) {}
}

export class NeuralODESolverStation extends DESStation {
  static readonly CH_SOLVE: ChannelName = 'solve';
  static readonly CH_SOLUTION: ChannelName = 'solution';

  constructor(id: string, private readonly network: NeuralNetworkLike) {
    super(id);
  }

  override hasWork(): boolean {
    return this.inboxSize(NeuralODESolverStation.CH_SOLVE) > 0;
  }

  runTimeStep(): void {
    const requests = this.drain<NeuralODESolveToken>(NeuralODESolverStation.CH_SOLVE);
    for (const req of requests) {
      this.emit(
        new NeuralODESolutionToken(req.id, solveNeuralODE(this.network, req.options)),
        NeuralODESolverStation.CH_SOLUTION,
      );
    }
  }
}

export class NeuralPredictionSink extends DESStation {
  readonly predictions: NeuralPredictionToken[] = [];

  override hasWork(): boolean {
    return this.inboxSize(NeuralNetworkStation.CH_PREDICTION) > 0;
  }

  runTimeStep(): void {
    this.predictions.push(...this.drain<NeuralPredictionToken>(NeuralNetworkStation.CH_PREDICTION));
  }
}
