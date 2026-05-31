#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-backpropagation.rs   (fn main)
// 1:1 file move. Backprop through a 2-3-1 sigmoid net expressed as a DES
// (layers are stations; activations/gradients flow forward and backward).
//
// Conversion notes (file-specific):
//   - mulberry32 / withSeed -> inject SeededRandom (shared::capabilities);
//     withSeed's global-seed wrapper -> pass the Rng explicitly.
//   - LayerStation classes (extends BidirectionalTimeSteppedStation) -> struct
//     + impl trait.
//   - fs/path artifact writes -> std::fs; top-level run -> fn main.
// =============================================================================

// =============================================================================
// Backpropagation through a 2-3-1 fully-connected sigmoid network as a
// discrete-event system. Each tick is one mini-batch step; layers are
// stations; activations and gradients are movables flowing forward and
// backward over the same edge graph.
//
// Architecture (no global FEL):
//
//                forward                 forward                 forward
//   XorSource ───────────▶  Layer1 (2→3) ─────────▶  Layer2 (3→1) ─────────▶  LossStation
//        ▲                    ▲                       ▲                          │
//        │                    │                       │                          │
//        │backward            │backward               │backward                  │
//        └─────────────────── └────────────────────── └──────────────────────────┘
//
// Each LayerStation holds W, b, and per-sample input/activation cache. On the
// forward pass it computes a = σ(Wx + b); on the backward pass it computes
// gradients, updates W and b in-place, and emits the upstream gradient. The
// XorSource waits for a backward done-signal before emitting the next sample
// (sequential SGD) so the per-sample weight updates apply in the same order
// regardless of station-execution order.
//
// "Recursive": the training cycle is the source emitting until N samples are
// done; each sample's backward must close the loop before the next one starts.
//
// Validation: numpy reference (external-references/backpropagation/bp.py)
// uses identical initial weights and naive nested-loop matrix-vector products,
// so per-step weights should agree with the framework to within a few ULPs.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {fisherYatesShuffle} from './general/general';
import {mulberry32, withSeed} from './general/prng';
import {BidirectionalTimeSteppedStation} from './general/time-stepped-station';

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
// d/dz σ(z) = σ(z)(1-σ(z)) -- expressed via the activation a = σ(z).
const sigmoidPrimeFromA = (a: number) => a * (1 - a);

interface SampleCache {
  input: number[];
  activation: number[];
}

class ForwardToken {
  constructor(
    public sampleId: number,
    public payload: number[],
    public target: number[],
  ) {}
}

class BackwardToken {
  constructor(
    public sampleId: number,
    public grad: number[],
  ) {}
}

abstract class BackpropStation extends BidirectionalTimeSteppedStation<ForwardToken, BackwardToken> {}

/**
 * z = W·x + b is computed with a naive nested loop (as opposed to a fused-
 * multiply-add or BLAS dot product) so the float-summation order matches the
 * Python reference exactly.
 */
class LayerStation extends BackpropStation {
  cache = new Map<number, SampleCache>();
  constructor(
    id: string,
    public W: number[][],     // [outDim][inDim]
    public b: number[],       // [outDim]
    public lr: number,
  ) { super(id); }
  runTimeStep(): void {
    while (this.forwardInbox.length > 0) {
      const t = this.forwardInbox.shift()!;
      const input = t.payload;
      const a = new Array(this.W.length);
      for (let i = 0; i < this.W.length; i++) {
        let zi = this.b[i];
        for (let j = 0; j < this.W[i].length; j++) zi += this.W[i][j] * input[j];
        a[i] = sigmoid(zi);
      }
      this.cache.set(t.sampleId, {input, activation: a});
      this.emitForward(new ForwardToken(t.sampleId, a, t.target));
    }
    while (this.backwardInbox.length > 0) {
      const t = this.backwardInbox.shift()!;
      const c = this.cache.get(t.sampleId)!;
      this.cache.delete(t.sampleId);
      const a = c.activation;
      const inDim = c.input.length;
      const outDim = a.length;
      // dL/dz = grad_a ∘ σ'(z),   σ'(z) = a(1-a)
      const dz = new Array(outDim);
      for (let i = 0; i < outDim; i++) dz[i] = t.grad[i] * sigmoidPrimeFromA(a[i]);
      // grad_input[j] = Σ_i W[i][j] * dz[i]   (compute BEFORE we mutate W).
      const gradInput = new Array(inDim).fill(0);
      for (let i = 0; i < outDim; i++) {
        for (let j = 0; j < inDim; j++) gradInput[j] += this.W[i][j] * dz[i];
      }
      for (let i = 0; i < outDim; i++) {
        for (let j = 0; j < inDim; j++) this.W[i][j] -= this.lr * dz[i] * c.input[j];
        this.b[i] -= this.lr * dz[i];
      }
      this.emitBackward(new BackwardToken(t.sampleId, gradInput));
    }
  }
}

/**
 * MSE loss: L = ½ Σ (a - y)². The initial gradient pushed back is (a - y),
 * which the upstream layer multiplies by σ'(z) to produce dL/dz on its output.
 */
class LossStation extends BackpropStation {
  losses: number[] = [];
  constructor() { super('loss'); }
  runTimeStep(): void {
    while (this.forwardInbox.length > 0) {
      const t = this.forwardInbox.shift()!;
      const a = t.payload;
      const y = t.target;
      let loss = 0;
      const grad = new Array(a.length);
      for (let i = 0; i < a.length; i++) {
        const e = a[i] - y[i];
        loss += 0.5 * e * e;
        grad[i] = e;
      }
      this.losses.push(loss);
      this.emitBackward(new BackwardToken(t.sampleId, grad));
    }
  }
}

/**
 * Cycles through XOR samples. Sequential SGD: only emits the next forward
 * token after the previous sample's backward chain has reached the source.
 * That guarantees one sample is in flight at any time, so per-sample weight
 * updates apply in the same order regardless of station execution order.
 */
class XorSource extends BackpropStation {
  private static SAMPLES: ReadonlyArray<{x: number[]; y: number[]}> = [
    {x: [0, 0], y: [0]},
    {x: [0, 1], y: [1]},
    {x: [1, 0], y: [1]},
    {x: [1, 1], y: [0]},
  ];
  private idx = 0;
  private inFlight = 0;
  constructor(id: string, public total: number) { super(id); }
  runTimeStep(): void {
    // Drain done-signal backward tokens.
    while (this.backwardInbox.length > 0) {
      this.backwardInbox.shift();
      this.inFlight--;
    }
    if (this.inFlight === 0 && this.idx < this.total) {
      const s = XorSource.SAMPLES[this.idx % 4];
      this.emitForward(new ForwardToken(this.idx, [...s.x], [...s.y]));
      this.idx++;
      this.inFlight = 1;
    }
  }
  isDone(): boolean { return this.idx >= this.total && this.inFlight === 0; }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

export interface InitialWeights {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
}

export interface BackpropResult {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  lossHistory: number[];      // one entry per training sample
  ticks: number;              // wall-clock simulation ticks consumed
  predictions: number[];      // final predictions on the 4 XOR cases
}

/**
 * Initial weights: each entry drawn from U(-1, 1) using mulberry32 with the
 * given seed. Dumped to JSON so the Python reference can pick up the SAME
 * initialisation and produce bit-comparable training trajectories.
 */
export function initWeights(seed: number, hidden = 3): InitialWeights {
  return withSeed(seed, () => {
    const rng = mulberry32(seed);
    const W1 = Array.from({length: hidden}, () =>
      Array.from({length: 2}, () => 2 * rng() - 1));
    const b1 = Array.from({length: hidden}, () => 2 * rng() - 1);
    const W2 = [Array.from({length: hidden}, () => 2 * rng() - 1)];
    const b2 = [2 * rng() - 1];
    return {W1, b1, W2, b2};
  });
}

export function runBackprop(
  init: InitialWeights,
  totalSamples: number,
  lr: number,
): BackpropResult {
  const src = new XorSource('src', totalSamples);
  const l1 = new LayerStation('L1', deepCopy(init.W1), [...init.b1], lr);
  const l2 = new LayerStation('L2', deepCopy(init.W2), [...init.b2], lr);
  const loss = new LossStation();
  src.addForwardOut(l1);
  l1.addForwardOut(l2);
  l2.addForwardOut(loss);
  loss.addBackwardOut(l2);
  l2.addBackwardOut(l1);
  l1.addBackwardOut(src);

  const stations: BackpropStation[] = [src, l1, l2, loss];

  let ticks = 0;
  while (!src.isDone() ||
         l1.forwardInbox.length > 0 || l1.backwardInbox.length > 0 ||
         l2.forwardInbox.length > 0 || l2.backwardInbox.length > 0 ||
         loss.forwardInbox.length > 0) {
    const order = [...stations];
    for (const _ of fisherYatesShuffle(order)) { /* generator side-effect */ }
    for (const s of order) s.runTimeStep(1.0, ticks);
    ticks++;
    if (ticks > totalSamples * 100) {
      throw new Error('runaway: training did not converge ticks');
    }
  }

  // Final predictions on the 4 XOR cases (forward-only, no weight update).
  const predictions: number[] = [];
  for (const s of [[0,0],[0,1],[1,0],[1,1]]) {
    const a1 = new Array(l1.W.length);
    for (let i = 0; i < l1.W.length; i++) {
      let z = l1.b[i];
      for (let j = 0; j < l1.W[i].length; j++) z += l1.W[i][j] * s[j];
      a1[i] = sigmoid(z);
    }
    const a2 = new Array(l2.W.length);
    for (let i = 0; i < l2.W.length; i++) {
      let z = l2.b[i];
      for (let j = 0; j < l2.W[i].length; j++) z += l2.W[i][j] * a1[j];
      a2[i] = sigmoid(z);
    }
    predictions.push(a2[0]);
  }

  return {
    W1: l1.W, b1: l1.b,
    W2: l2.W, b2: l2.b,
    lossHistory: loss.losses,
    ticks,
    predictions,
  };
}

function deepCopy<T>(x: T): T { return JSON.parse(JSON.stringify(x)); }

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

function main() {
  const seed = Number(process.env.SEED ?? 7);
  const N = Number(process.env.N ?? 10000);
  const lr = Number(process.env.LR ?? 0.5);

  const init = initWeights(seed, 3);

  console.log(`# Backpropagation simulation (2-3-1, sigmoid + MSE)`);
  console.log(`#   seed = ${seed}   samples = ${N}   lr = ${lr}`);

  const t0 = Date.now();
  const result = runBackprop(init, N, lr);
  const dtMs = Date.now() - t0;

  console.log(`# wall-clock ms = ${dtMs}    ticks = ${result.ticks}`);
  const last100 = result.lossHistory.slice(-100);
  const avgLoss = last100.reduce((s, v) => s + v, 0) / last100.length;
  console.log(`# avg loss over last 100 samples = ${avgLoss.toExponential(3)}`);
  console.log(`# XOR predictions:`);
  console.log(`    0 XOR 0  →  ${result.predictions[0].toFixed(4)}    (target 0)`);
  console.log(`    0 XOR 1  →  ${result.predictions[1].toFixed(4)}    (target 1)`);
  console.log(`    1 XOR 0  →  ${result.predictions[2].toFixed(4)}    (target 1)`);
  console.log(`    1 XOR 1  →  ${result.predictions[3].toFixed(4)}    (target 0)`);

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'backprop-framework.json');
  fs.writeFileSync(outPath, JSON.stringify({
    config: {seed, N, lr},
    init,
    final: {W1: result.W1, b1: result.b1, W2: result.W2, b2: result.b2},
    predictions: result.predictions,
    lossHistory: result.lossHistory,
    ticks: result.ticks,
  }));
  console.log(`# wrote ${outPath}`);
}

if (require.main === module) main();
