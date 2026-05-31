'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/neural_network.rs
// - Keep file-for-file. NumericVector aliases map to Vec<f64>; neural network
//   interfaces become behavior traits for predict/train/snapshot.
// - Neural token classes become token structs. NeuralNetworkStation and
//   SupervisedNeuralNetworkStation become structs implementing DESStation over a
//   generic network trait object or type parameter.
// - Pure inference/training adapters can stay methods; graph-level inference
//   should be represented as PureTransform/PureTransformEntity.
// - Convert invalid sample shapes and backend failures to Result.

// =============================================================================
// general/des-base/neural-network.ts — DES base classes for neural networks.
//
// Neural nets are the hybrid case in this codebase:
//   • their forward/backward passes are numerical computations;
//   • their training data, inference requests, policy actions, and ODE solve
//     requests move through the DES station graph as queued tokens.
//
// These classes keep that boundary explicit. A neural model is just a
// `NeuralNetworkLike`; stations provide queueing semantics and typed channels.
// =============================================================================

import {DESStation, ChannelName, Token} from './station';

export type NumericVector = number[];

export interface NeuralNetworkLike {
  readonly inputDim: number;
  readonly outputDim: number;
  predict(input: NumericVector): NumericVector;
  parameterCount?(): number;
  clone?(): NeuralNetworkLike;
}

export interface TrainableNeuralNetwork extends NeuralNetworkLike {
  trainSample(input: NumericVector, target: NumericVector, learningRate: number): {
    loss: number;
    prediction: NumericVector;
  };
}

export class NeuralInferenceToken implements Token {
  constructor(
    public readonly id: string,
    public readonly input: NumericVector,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class SupervisedSampleToken implements Token {
  constructor(
    public readonly id: string,
    public readonly input: NumericVector,
    public readonly target: NumericVector,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class NeuralPredictionToken implements Token {
  constructor(
    public readonly id: string,
    public readonly input: NumericVector,
    public readonly output: NumericVector,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class NeuralTrainingResultToken implements Token {
  constructor(
    public readonly sampleId: string,
    public readonly loss: number,
    public readonly prediction: NumericVector,
    public readonly target: NumericVector,
    public readonly step: number,
    public readonly meta: Record<string, unknown> = {},
  ) {}
}

export class NeuralSnapshotToken implements Token {
  constructor(
    public readonly trainingStep: number,
    public readonly loss: number | null,
    public readonly parameterCount: number | null,
  ) {}
}

export class NeuralNetworkStation<N extends NeuralNetworkLike = NeuralNetworkLike>
  extends DESStation {
  static readonly CH_INFER: ChannelName = 'infer';
  static readonly CH_PREDICTION: ChannelName = 'prediction';
  static readonly CH_SNAPSHOT: ChannelName = 'snapshot';

  constructor(id: string, protected readonly network: N) {
    super(id);
  }

  getNetwork(): N { return this.network; }

  override hasWork(): boolean {
    return this.inboxSize(NeuralNetworkStation.CH_INFER) > 0;
  }

  protected processInferenceQueue(): void {
    const requests = this.drain<NeuralInferenceToken>(NeuralNetworkStation.CH_INFER);
    for (const req of requests) {
      const output = this.network.predict(req.input);
      this.emit(
        new NeuralPredictionToken(req.id, req.input.slice(), output, req.meta),
        NeuralNetworkStation.CH_PREDICTION,
      );
    }
  }

  runTimeStep(): void {
    this.processInferenceQueue();
  }
}

export interface SupervisedNeuralNetworkStationOptions {
  learningRate: number;
  /** Emit a snapshot after every N samples. Default 0 disables snapshots. */
  snapshotEvery?: number;
}

export class SupervisedNeuralNetworkStation<N extends TrainableNeuralNetwork = TrainableNeuralNetwork>
  extends NeuralNetworkStation<N> {
  static readonly CH_TRAIN: ChannelName = 'train';
  static readonly CH_TRAINING_RESULT: ChannelName = 'training-result';

  readonly lossHistory: number[] = [];
  private trainingStep = 0;
  private readonly learningRate: number;
  private readonly snapshotEvery: number;

  constructor(id: string, network: N, opts: SupervisedNeuralNetworkStationOptions) {
    super(id, network);
    this.learningRate = opts.learningRate;
    this.snapshotEvery = opts.snapshotEvery ?? 0;
  }

  override hasWork(): boolean {
    return super.hasWork() || this.inboxSize(SupervisedNeuralNetworkStation.CH_TRAIN) > 0;
  }

  override runTimeStep(): void {
    const samples = this.drain<SupervisedSampleToken>(SupervisedNeuralNetworkStation.CH_TRAIN);
    for (const sample of samples) {
      const r = this.network.trainSample(sample.input, sample.target, this.learningRate);
      this.trainingStep += 1;
      this.lossHistory.push(r.loss);
      this.emit(
        new NeuralTrainingResultToken(
          sample.id, r.loss, r.prediction, sample.target.slice(), this.trainingStep, sample.meta,
        ),
        SupervisedNeuralNetworkStation.CH_TRAINING_RESULT,
      );
      if (this.snapshotEvery > 0 && this.trainingStep % this.snapshotEvery === 0) {
        this.emit(
          new NeuralSnapshotToken(
            this.trainingStep,
            r.loss,
            this.network.parameterCount ? this.network.parameterCount() : null,
          ),
          NeuralNetworkStation.CH_SNAPSHOT,
        );
      }
    }
    this.processInferenceQueue();
  }

  getTrainingStep(): number { return this.trainingStep; }
}
