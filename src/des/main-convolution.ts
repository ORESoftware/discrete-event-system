#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-convolution.rs   (fn main)
// 1:1 file move. 1-D convolution as a streaming DES pipeline (SignalSource ->
// ConvolutionStation -> CollectorSink).
//
// Conversion notes (file-specific):
//   - mulberry32 / withSeed -> SeededRandom (shared::capabilities); the
//     fisherYatesShuffle tick ordering must use the injected Rng.
//   - classes (Sample, SignalSource extends RoutedTimeSteppedStation) -> struct
//     + impl trait.
//   - fs/path output -> std::fs; top-level run -> fn main.
// =============================================================================

// =============================================================================
// 1-D convolution as a discrete-event system (a streaming arithmetic
// pipeline rather than a probabilistic simulation, but the same DES
// substrate — stations and movables on a tick clock).
//
// `y[n] = (x * h)[n] = Σ_k h[k] * x[n-k]`
//
// The simulation is a pipeline of three stationary entities, no global FEL:
//
//     SignalSource  ── x[n] ─▶  ConvolutionStation  ── y[n] ─▶  CollectorSink
//
//   * SignalSource emits one Sample per step, carrying the next value from a
//     fixed input array.
//   * ConvolutionStation buffers the last K = len(kernel) samples and emits
//     y[n] each step.
//   * CollectorSink accumulates outputs.
//
// Order of `runTimeStep` calls within a step is randomized by
// fisherYatesShuffle so the result is order-independent.
//
// Validation: numpy.convolve with mode='full' (or scipy.signal.convolve).
// Run `bash external-references/run-all.sh` and then
// `node dist/des/runners/validate-convolution.js`.
// =============================================================================

import {fisherYatesShuffle} from './general/general';
import {mulberry32, withSeed} from './general/prng';
import {RoutedTimeSteppedStation} from './general/time-stepped-station';
import * as fs from 'fs';
import * as path from 'path';

class Sample {
  constructor(public n: number, public value: number) {}
}

class SignalSource extends RoutedTimeSteppedStation<Sample> {
  private idx = 0;
  constructor(id: string, public signal: ReadonlyArray<number>) { super(id); }
  runTimeStep(_stepSize: number, _t: number): void {
    if (this.idx < this.signal.length) {
      this.emit(new Sample(this.idx, this.signal[this.idx]));
      this.idx++;
    }
  }
  isDone(): boolean { return this.idx >= this.signal.length; }
}

class ConvolutionStation extends RoutedTimeSteppedStation<Sample> {
  // Circular ring buffer of the K most recent x values.
  private buffer: number[];
  private head = 0;
  private warmup = 0;       // how many samples have we ingested
  private outIdx = 0;
  private flushedAfter = 0; // how many tail flushes after EOF, for 'full' mode
  private wantFullMode: boolean;
  constructor(id: string, public kernel: ReadonlyArray<number>, opts: {fullMode?: boolean} = {}) {
    super(id);
    this.buffer = new Array(kernel.length).fill(0);
    this.wantFullMode = opts.fullMode ?? true;
  }
  /**
   * The DSP-textbook convolution sum:
   *
   *   y[n] = Σ_{k=0}^{K-1} h[k] * x[n - k]
   *
   * With `kernel = h`, store the K most recent x values in `buffer`. After we
   * place the newest at position (K-1), the index that holds x[n-k] is
   * (K-1-k). We use a circular ring to avoid Array.shift() (O(K) per step).
   */
  private dot(): number {
    const K = this.kernel.length;
    let y = 0;
    for (let k = 0; k < K; k++) {
      const idx = (this.head - 1 - k + K * 2) % K;
      y += this.kernel[k] * this.buffer[idx];
    }
    return y;
  }
  /**
   * Called by the program loop AFTER source.isDone() to flush the K-1 trailing
   * outputs (mode='full'). Each call advances by one zero sample.
   */
  flushOnce(): void {
    this.buffer[this.head] = 0;
    this.head = (this.head + 1) % this.kernel.length;
    this.flushedAfter++;
    this.emit(new Sample(this.outIdx++, this.dot()));
  }
  needsFlush(): boolean {
    return this.wantFullMode && this.flushedAfter < this.kernel.length - 1;
  }
  runTimeStep(_stepSize: number, _t: number): void {
    // Drain inbox: one sample per tick is the typical case, but support
    // multiple in case the source emits faster than the convolver runs.
    while (this.inbox.length > 0) {
      const s = this.inbox.shift()!;
      this.buffer[this.head] = s.value;
      this.head = (this.head + 1) % this.kernel.length;
      this.warmup++;
      this.emit(new Sample(this.outIdx++, this.dot()));
    }
  }
}

class CollectorSink extends RoutedTimeSteppedStation<Sample> {
  results: Sample[] = [];
  constructor(id: string) { super(id); }
  runTimeStep(_stepSize: number, _t: number): void {
    while (this.inbox.length > 0) this.results.push(this.inbox.shift()!);
  }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

export interface ConvolutionResult {
  y: number[];
  ticks: number;
  meta: {
    signalLen: number;
    kernelLen: number;
    mode: 'full';
  };
}

export function runConvolution(
  signal: ReadonlyArray<number>,
  kernel: ReadonlyArray<number>,
  opts: {seed?: number} = {},
): ConvolutionResult {
  return withSeed(opts.seed ?? 1, () => {
    const src = new SignalSource('src', signal);
    const conv = new ConvolutionStation('conv', kernel, {fullMode: true});
    const sink = new CollectorSink('sink');
    src.addOutConnection(conv);
    conv.addOutConnection(sink);
    const stations: RoutedTimeSteppedStation<Sample>[] = [src, conv, sink];

    // Run until the source is exhausted AND the convolver has flushed all
    // trailing zeros (mode='full'). Each tick: shuffle, then runTimeStep.
    let ticks = 0;
    while (true) {
      const order = [...stations];
      for (const _ of fisherYatesShuffle(order)) { /* generator side-effect */ }
      for (const s of order) s.runTimeStep(1.0, ticks);
      ticks++;
      if (src.isDone() && conv.needsFlush()) {
        conv.flushOnce();
      } else if (src.isDone() && !conv.needsFlush()) {
        // Drain anything left in sink.inbox that the random order may have
        // skipped this tick.
        sink.runTimeStep(1.0, ticks);
        break;
      }
    }

    return {
      y: sink.results.map(s => s.value),
      ticks,
      meta: {signalLen: signal.length, kernelLen: kernel.length, mode: 'full'},
    };
  });
}

// -----------------------------------------------------------------------------
// CLI: run a small demo and dump JSON for the validation driver to consume.
// -----------------------------------------------------------------------------

function makeTriangleKernel(K: number): number[] {
  // Symmetric triangular FIR (exact peak in the middle), normalized to sum=1.
  const h = new Array(K).fill(0);
  const peak = (K - 1) / 2;
  let s = 0;
  for (let i = 0; i < K; i++) {
    h[i] = 1 - Math.abs(i - peak) / (peak + 1);
    s += h[i];
  }
  return h.map(v => v / s);
}

function makeTestSignal(N: number, seed: number): number[] {
  const rng = mulberry32(seed);
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    // Multiscale signal: 0.1 Hz sine + 0.4 Hz cosine + small white noise.
    out[i] = Math.sin(2 * Math.PI * 0.1 * i) +
             0.5 * Math.cos(2 * Math.PI * 0.4 * i) +
             0.1 * (rng() - 0.5);
  }
  return out;
}

function main() {
  const N = Number(process.env.N ?? 64);
  const K = Number(process.env.K ?? 7);
  const seed = Number(process.env.SEED ?? 42);

  const signal = makeTestSignal(N, seed);
  const kernel = makeTriangleKernel(K);

  console.log(`# Convolution simulation`);
  console.log(`#   signal length = ${signal.length}`);
  console.log(`#   kernel length = ${kernel.length}  (triangular, normalized)`);
  console.log(`#   seed          = ${seed}`);

  const result = runConvolution(signal, kernel, {seed});

  console.log(`# output length    = ${result.y.length}`);
  console.log(`# wall-clock ticks = ${result.ticks}`);
  console.log('# first 12 outputs:');
  for (let i = 0; i < Math.min(12, result.y.length); i++) {
    console.log(`  y[${i}] = ${result.y[i].toFixed(6)}`);
  }
  console.log(`# ...`);
  console.log(`# last 4 outputs:`);
  for (let i = Math.max(0, result.y.length - 4); i < result.y.length; i++) {
    console.log(`  y[${i}] = ${result.y[i].toFixed(6)}`);
  }

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'convolution-framework.json');
  fs.writeFileSync(outPath, JSON.stringify({
    signal,
    kernel,
    y: result.y,
    meta: result.meta,
    ticks: result.ticks,
    seed,
  }, null, 2));
  console.log(`# wrote ${outPath}`);
}

if (require.main === module) main();
