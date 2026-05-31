// RUST MIGRATION: target module src/des/general/math_blocks.rs.
// RUST MIGRATION: BlockModelLogger should become a logging trait, MathSignal/MathSample/options/results/node/edge/trace structs become serde structs, and operator unions become enums.
// RUST MIGRATION: MathBlock and each concrete block become structs implementing VisualBlock/MathBlock traits; inheritance chains such as SubtractBlock extends SumBlock become shared helper traits/composition.
// RUST MIGRATION: runMathBlockDiagram, runODEBlockSystem, and runHeat1DBlockGrid are graph-visible transforms and should be PureTransform entry structs returning Result.
// RUST MIGRATION: Record<string, number> signal maps become HashMap<String, f64>, expression callbacks need a parser/trait port, and validation returns Result/ValidationCheck vectors.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/math-blocks.rs  (module des::general::math_blocks)
// 1:1 file move. Calculus/control block diagrams (sources/sums/gains/integrators/...) as DES VisualBlocks.
//
// Declarations → Rust:
//   const MATH_IN/MATH_OUT        -> `const &str`
//   type IntegratorMethod/ComparatorOp/LogicOp = '...'|'...' -> enums
//   interface MathSignal extends Token / BlockModelLogger / MathSample / MathBlock*/ODE*/Heat1D*/BlockGraph* -> structs/traits
//   abstract class MathBlock extends VisualBlock -> trait MathBlock (defaults) + structs that compose a VisualBlock core
//   class Constant/Function/Expression Source / Sink / Sum / Subtract / Product / Gain / Saturation /
//         Integrator / Derivative / FirstOrderFilter / Comparator / Logic / Expression / Laplacian1D Block
//                                    -> structs `impl` the MathBlock trait (NO `extends`)
//   fn runMathBlockDiagram / runODEBlockSystem / runHeat1DBlockGrid -> fns
//
// Conversion notes (file-specific):
//   - DEEP inheritance (MathBlock <- VisualBlock; SubtractBlock <- SumBlock) -> trait + composition, not extends.
//   - `FunctionSourceBlock` wraps a CLOSURE `(t) => number` -> `Box<dyn Fn(f64) -> f64>` field.
//   - `IntegratorMethod`/`ComparatorOp`/`LogicOp` string unions -> enums matched with `match`.
//   - `MathSignal extends Token` + `metadata?: Record<string, unknown>` -> struct `impl Token` + `serde_json::Value`.
//   - `BlockModelLogger.log(event)` callback -> `&dyn Logger` trait; uses expr.rs (parse/evaluate).
// =============================================================================

// =============================================================================
// Math block diagrams as DES stations.
//
// Blocks are stationary VisualBlock instances. Numeric MathSignal tokens move
// between them over named channels. This gives calculus/control style block
// diagrams (sources, sums, gains, integrators, differentiators, filters, logic)
// a first-class home without changing the DES base classes.
// =============================================================================

import {Token} from './des-base/station';
import {IterativeRunSummary, runIterativeDES} from './des-base/runner';
import {ValidationCheck, intrinsicCheck} from './des-base/validation';
import {VisualBlock, VisualBlockSpec, visualBlockSpecs} from './des-base/visual-block';
import {Preconditions} from './des-base/preconditions';
import {evaluate, parse, Expr} from './expr';

export const MATH_IN = 'in';
export const MATH_OUT = 'out';

export interface BlockModelLogger {
  log(event: {kind: string; level?: 'trace' | 'debug' | 'info' | 'warn' | 'error'; [key: string]: unknown}): void;
}

export interface MathSignal extends Token {
  kind: 'math-signal';
  sourceId: string;
  channel: string;
  tick: number;
  time: number;
  value: number;
  metadata?: Record<string, unknown>;
}

export interface MathSample {
  blockId: string;
  channel: string;
  tick: number;
  time: number;
  value: number;
}

export interface MathBlockOptions {
  dt: number;
  /** Number of runTimeStep executions. A trace over N integration steps uses N + 1 ticks. */
  ticks: number;
  t0?: number;
}

export interface MathBlockRunResult {
  summary: IterativeRunSummary;
  validation: ValidationCheck[];
  outputs: MathSample[];
  visualBlocks: VisualBlockSpec[];
}

function isMathSignal(t: Token): t is MathSignal {
  const maybe = t as Partial<MathSignal>;
  return maybe.kind === 'math-signal' &&
    typeof maybe.sourceId === 'string' &&
    typeof maybe.channel === 'string' &&
    typeof maybe.tick === 'number' &&
    typeof maybe.time === 'number' &&
    typeof maybe.value === 'number';
}

function assertName(model: string, param: string, name: string): void {
  Preconditions.check(model, param, 'match /^[A-Za-z_][A-Za-z0-9_]*$/', /^[A-Za-z_][A-Za-z0-9_]*$/.test(name), name);
}

function assertUnique(model: string, param: string, values: readonly string[]): void {
  const seen = new Set<string>();
  for (const value of values) {
    Preconditions.check(model, param, 'contain unique names', !seen.has(value), value);
    seen.add(value);
  }
}

function last<T>(xs: readonly T[]): T | undefined {
  return xs.length === 0 ? undefined : xs[xs.length - 1];
}

function latestUnusedAtOrBefore(signals: readonly MathSignal[], tick: number, consumedThroughTick: number): MathSignal | undefined {
  let best: MathSignal | undefined;
  for (const s of signals) {
    if (s.tick > tick || s.tick <= consumedThroughTick) continue;
    if (!best || s.tick >= best.tick) best = s;
  }
  return best;
}

function durationSteps(model: string, t0: number, t1: number, dt: number): number {
  Preconditions.finite(model, 't0', t0);
  Preconditions.finite(model, 't1', t1);
  Preconditions.positive(model, 'dt', dt);
  Preconditions.check(model, 't1', 'be greater than t0', t1 > t0, {t0, t1});
  const exact = (t1 - t0) / dt;
  const steps = Math.round(exact);
  Preconditions.check(model, 'duration/dt', 'be an integer number of steps', Math.abs(exact - steps) <= 1e-9 * Math.max(1, Math.abs(exact)), exact);
  Preconditions.integerInRange(model, 'steps', steps, 1, 1000000);
  return steps;
}

function finiteRecord(model: string, param: string, r: Record<string, number> | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  if (!r) return out;
  for (const [k, v] of Object.entries(r)) {
    assertName(model, `${param}.${k}`, k);
    Preconditions.finite(model, `${param}.${k}`, v);
    out[k] = v;
  }
  return out;
}

export abstract class MathBlock extends VisualBlock {
  protected currentTick = 0;
  protected readonly latestByChannel = new Map<string, MathSignal>();
  readonly outputHistory: MathSample[] = [];
  readonly dt: number;
  readonly ticks: number;
  readonly t0: number;

  protected constructor(id: string, opts: MathBlockOptions) {
    super(id, {
      kind: 'math-block',
      ports: {
        inputs: [{id: MATH_IN, kind: 'MathSignal'}],
        outputs: [{id: MATH_OUT, kind: 'MathSignal'}],
      },
      style: {fill: '#f8fafc', stroke: '#2563eb', text: '#0f172a'},
    });
    this.dt = opts.dt;
    this.ticks = opts.ticks;
    this.t0 = opts.t0 ?? 0;
    this.addValidator(intrinsicCheck<MathBlock>({
      name: `math-block-finite-output/${id}`,
      predicate: s => s.outputHistory.every(x => Number.isFinite(x.value) && Number.isFinite(x.time)),
      expected: 'all emitted math samples are finite',
      group: 'math-blocks',
    }));
  }

  override assertPreconditions(): void {
    Preconditions.positive('MathBlock', `${this.id}.dt`, this.dt);
    Preconditions.integerInRange('MathBlock', `${this.id}.ticks`, this.ticks, 1, 1000000);
    Preconditions.finite('MathBlock', `${this.id}.t0`, this.t0);
  }

  override hasWork(): boolean {
    return this.currentTick < this.ticks;
  }

  runTimeStep(): void {
    if (!this.hasWork()) return;
    const tick = this.currentTick;
    this.step(tick, this.t0 + tick * this.dt, this.dt);
    this.currentTick++;
  }

  protected abstract step(tick: number, time: number, dt: number): void;

  protected drainMath(channel: string): MathSignal[] {
    const tokens = this.drain<Token>(channel);
    const signals: MathSignal[] = [];
    for (const token of tokens) {
      if (!isMathSignal(token)) {
        throw new Error(`${this.id}: expected MathSignal on channel "${channel}"`);
      }
      Preconditions.finite(this.id, `signal.${channel}.value`, token.value);
      Preconditions.finite(this.id, `signal.${channel}.time`, token.time);
      this.latestByChannel.set(channel, token);
      signals.push(token);
    }
    return signals;
  }

  protected latestInput(channel: string): MathSignal | undefined {
    return this.latestByChannel.get(channel);
  }

  protected latestValue(channel: string): number | undefined {
    return this.latestByChannel.get(channel)?.value;
  }

  protected inputValue(channel: string, holdLast: boolean): number | undefined {
    const fresh = this.drainMath(channel);
    const signal = last(fresh) ?? (holdLast ? this.latestInput(channel) : undefined);
    return signal?.value;
  }

  protected inputValues(channels: readonly string[], holdLast: boolean): number[] | undefined {
    const values: number[] = [];
    for (const channel of channels) {
      const value = this.inputValue(channel, holdLast);
      if (value === undefined) return undefined;
      values.push(value);
    }
    return values;
  }

  protected emitValue(value: number, tick: number, time: number, channel: string = MATH_OUT, metadata?: Record<string, unknown>): void {
    Preconditions.finite(this.id, `output.${channel}`, value);
    const signal: MathSignal = {
      kind: 'math-signal',
      sourceId: this.id,
      channel,
      tick,
      time,
      value,
      metadata,
    };
    this.outputHistory.push({blockId: this.id, channel, tick, time, value});
    this.emit(signal, channel);
  }
}

export class ConstantSourceBlock extends MathBlock {
  constructor(id: string, readonly value: number, opts: MathBlockOptions, readonly outputChannel: string = MATH_OUT) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.finite('ConstantSourceBlock', `${this.id}.value`, this.value);
  }

  protected step(tick: number, time: number): void {
    this.emitValue(this.value, tick, time, this.outputChannel);
  }
}

export class FunctionSourceBlock extends MathBlock {
  constructor(
    id: string,
    readonly fn: (time: number, tick: number) => number,
    opts: MathBlockOptions,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  protected step(tick: number, time: number): void {
    this.emitValue(this.fn(time, tick), tick, time, this.outputChannel);
  }
}

export class ExpressionSourceBlock extends MathBlock {
  private readonly ast: Expr;
  private readonly constants: Record<string, number>;

  constructor(
    id: string,
    readonly expression: string,
    opts: MathBlockOptions,
    constants?: Record<string, number>,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
    this.ast = parse(expression);
    this.constants = finiteRecord('ExpressionSourceBlock', `${id}.constants`, constants);
  }

  protected step(tick: number, time: number): void {
    const env = {...this.constants, t: time, tick};
    this.emitValue(evaluate(this.ast, env), tick, time, this.outputChannel);
  }
}

export class SinkBlock extends MathBlock {
  readonly received: MathSignal[] = [];

  constructor(id: string, opts: MathBlockOptions, readonly inputChannels: string[] = [MATH_IN]) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.nonEmpty('SinkBlock', `${this.id}.inputChannels`, this.inputChannels);
  }

  protected step(): void {
    for (const channel of this.inputChannels) {
      this.received.push(...this.drainMath(channel));
    }
  }

  series(sourceId?: string): MathSample[] {
    return this.received
      .filter(s => sourceId === undefined || s.sourceId === sourceId)
      .map(s => ({blockId: s.sourceId, channel: s.channel, tick: s.tick, time: s.time, value: s.value}));
  }
}

export class SumBlock extends MathBlock {
  private readonly weights: number[];

  constructor(
    id: string,
    readonly inputChannels: string[],
    opts: MathBlockOptions,
    weights?: readonly number[],
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
    this.weights = weights ? weights.slice() : inputChannels.map(() => 1);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.nonEmpty('SumBlock', `${this.id}.inputChannels`, this.inputChannels);
    Preconditions.lengthEq('SumBlock', `${this.id}.weights`, this.weights, this.inputChannels.length);
    assertUnique('SumBlock', `${this.id}.inputChannels`, this.inputChannels);
    Preconditions.allFinite('SumBlock', `${this.id}.weights`, this.weights);
  }

  protected step(tick: number, time: number): void {
    const xs = this.inputValues(this.inputChannels, this.holdLast);
    if (!xs) return;
    const y = xs.reduce((sum, x, i) => sum + this.weights[i] * x, 0);
    this.emitValue(y, tick, time, this.outputChannel);
  }
}

export class SubtractBlock extends SumBlock {
  constructor(id: string, positiveInput: string, negativeInput: string, opts: MathBlockOptions, holdLast = true, outputChannel: string = MATH_OUT) {
    super(id, [positiveInput, negativeInput], opts, [1, -1], holdLast, outputChannel);
  }
}

export class ProductBlock extends MathBlock {
  constructor(
    id: string,
    readonly inputChannels: string[],
    opts: MathBlockOptions,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.nonEmpty('ProductBlock', `${this.id}.inputChannels`, this.inputChannels);
    assertUnique('ProductBlock', `${this.id}.inputChannels`, this.inputChannels);
  }

  protected step(tick: number, time: number): void {
    const xs = this.inputValues(this.inputChannels, this.holdLast);
    if (!xs) return;
    const y = xs.reduce((product, x) => product * x, 1);
    this.emitValue(y, tick, time, this.outputChannel);
  }
}

export class GainBlock extends MathBlock {
  constructor(
    id: string,
    readonly gain: number,
    opts: MathBlockOptions,
    readonly inputChannel: string = MATH_IN,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.finite('GainBlock', `${this.id}.gain`, this.gain);
  }

  protected step(tick: number, time: number): void {
    const x = this.inputValue(this.inputChannel, this.holdLast);
    if (x === undefined) return;
    this.emitValue(this.gain * x, tick, time, this.outputChannel);
  }
}

export class SaturationBlock extends MathBlock {
  constructor(
    id: string,
    readonly min: number,
    readonly max: number,
    opts: MathBlockOptions,
    readonly inputChannel: string = MATH_IN,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.finite('SaturationBlock', `${this.id}.min`, this.min);
    Preconditions.finite('SaturationBlock', `${this.id}.max`, this.max);
    Preconditions.check('SaturationBlock', `${this.id}.bounds`, 'satisfy min <= max', this.min <= this.max, {min: this.min, max: this.max});
  }

  protected step(tick: number, time: number): void {
    const x = this.inputValue(this.inputChannel, this.holdLast);
    if (x === undefined) return;
    this.emitValue(Math.max(this.min, Math.min(this.max, x)), tick, time, this.outputChannel);
  }
}

export type IntegratorMethod = 'euler' | 'trapezoid';

export class IntegratorBlock extends MathBlock {
  private state: number;
  private stateTick = 0;
  private lastInput: MathSignal | undefined;
  private consumedThroughTick = -Infinity;

  constructor(
    id: string,
    initialState: number,
    opts: MathBlockOptions,
    readonly method: IntegratorMethod = 'euler',
    readonly inputChannel: string = MATH_IN,
    readonly holdLast = false,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
    this.state = initialState;
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.finite('IntegratorBlock', `${this.id}.initialState`, this.state);
    Preconditions.check('IntegratorBlock', `${this.id}.method`, 'be euler or trapezoid', this.method === 'euler' || this.method === 'trapezoid', this.method);
  }

  currentState(): number {
    return this.state;
  }

  protected step(tick: number, time: number, dt: number): void {
    const incoming = this.drainMath(this.inputChannel);
    if (this.stateTick < tick) this.advanceToward(tick, incoming, dt);
    this.emitValue(this.state, tick, time, this.outputChannel);
    if (this.stateTick === tick && tick < this.ticks - 1) this.advanceToward(tick + 1, incoming, dt);
  }

  private advanceToward(targetTick: number, incoming: readonly MathSignal[], dt: number): void {
    while (this.stateTick < targetTick) {
      const fresh = latestUnusedAtOrBefore(incoming, this.stateTick, this.consumedThroughTick);
      const sig = fresh ?? (this.holdLast ? this.lastInput : undefined);
      if (!sig) return;
      const slope = this.method === 'trapezoid' && this.lastInput
        ? 0.5 * (this.lastInput.value + sig.value)
        : sig.value;
      this.state += dt * slope;
      Preconditions.finite('IntegratorBlock', `${this.id}.state`, this.state);
      this.lastInput = sig;
      if (fresh) this.consumedThroughTick = fresh.tick;
      this.stateTick++;
    }
  }
}

export class DerivativeBlock extends MathBlock {
  private previous: MathSignal | undefined;

  constructor(
    id: string,
    opts: MathBlockOptions,
    readonly inputChannel: string = MATH_IN,
    readonly holdLast = true,
    readonly initialOutput = 0,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  protected step(tick: number, time: number, dt: number): void {
    const fresh = this.drainMath(this.inputChannel);
    const sig = last(fresh) ?? (this.holdLast ? this.latestInput(this.inputChannel) : undefined);
    if (!sig) return;
    if (!this.previous) {
      this.previous = sig;
      this.emitValue(this.initialOutput, tick, time, this.outputChannel);
      return;
    }
    const denom = Math.abs(sig.time - this.previous.time) > 1e-12 ? sig.time - this.previous.time : dt;
    Preconditions.notDivByZero('DerivativeBlock', `${this.id}.dt`, denom);
    const value = (sig.value - this.previous.value) / denom;
    this.previous = sig;
    this.emitValue(value, tick, time, this.outputChannel);
  }
}

export class FirstOrderFilterBlock extends MathBlock {
  private y: number;

  constructor(
    id: string,
    readonly tau: number,
    initial: number,
    opts: MathBlockOptions,
    readonly inputChannel: string = MATH_IN,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
    this.y = initial;
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.positive('FirstOrderFilterBlock', `${this.id}.tau`, this.tau);
    Preconditions.finite('FirstOrderFilterBlock', `${this.id}.initial`, this.y);
  }

  protected step(tick: number, time: number, dt: number): void {
    const x = this.inputValue(this.inputChannel, this.holdLast);
    if (x !== undefined) {
      const alpha = dt / (this.tau + dt);
      this.y += alpha * (x - this.y);
    }
    this.emitValue(this.y, tick, time, this.outputChannel);
  }
}

export type ComparatorOp = 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq';

export class ComparatorBlock extends MathBlock {
  constructor(
    id: string,
    readonly op: ComparatorOp,
    opts: MathBlockOptions,
    readonly leftChannel: string = 'left',
    readonly rightChannel?: string,
    readonly threshold?: number,
    readonly tolerance = 1e-9,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.check('ComparatorBlock', `${this.id}.op`, 'be a supported comparison', ['gt', 'gte', 'lt', 'lte', 'eq', 'neq'].includes(this.op), this.op);
    Preconditions.nonNegative('ComparatorBlock', `${this.id}.tolerance`, this.tolerance);
    if (this.rightChannel === undefined) Preconditions.finite('ComparatorBlock', `${this.id}.threshold`, this.threshold as number);
  }

  protected step(tick: number, time: number): void {
    const left = this.inputValue(this.leftChannel, this.holdLast);
    if (left === undefined) return;
    const right = this.rightChannel === undefined ? this.threshold : this.inputValue(this.rightChannel, this.holdLast);
    if (right === undefined) return;
    this.emitValue(this.compare(left, right) ? 1 : 0, tick, time, this.outputChannel);
  }

  private compare(a: number, b: number): boolean {
    switch (this.op) {
      case 'gt': return a > b;
      case 'gte': return a >= b;
      case 'lt': return a < b;
      case 'lte': return a <= b;
      case 'eq': return Math.abs(a - b) <= this.tolerance;
      case 'neq': return Math.abs(a - b) > this.tolerance;
    }
  }
}

export type LogicOp = 'and' | 'or' | 'not' | 'xor';

export class LogicBlock extends MathBlock {
  constructor(
    id: string,
    readonly op: LogicOp,
    readonly inputChannels: string[],
    opts: MathBlockOptions,
    readonly threshold = 0.5,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.check('LogicBlock', `${this.id}.op`, 'be and, or, not, or xor', ['and', 'or', 'not', 'xor'].includes(this.op), this.op);
    Preconditions.nonEmpty('LogicBlock', `${this.id}.inputChannels`, this.inputChannels);
    if (this.op === 'not') Preconditions.lengthEq('LogicBlock', `${this.id}.inputChannels`, this.inputChannels, 1);
    assertUnique('LogicBlock', `${this.id}.inputChannels`, this.inputChannels);
  }

  protected step(tick: number, time: number): void {
    const xs = this.inputValues(this.inputChannels, this.holdLast);
    if (!xs) return;
    const bits = xs.map(x => x > this.threshold);
    let result: boolean;
    switch (this.op) {
      case 'and': result = bits.every(Boolean); break;
      case 'or': result = bits.some(Boolean); break;
      case 'not': result = !bits[0]; break;
      case 'xor': result = bits.filter(Boolean).length % 2 === 1; break;
    }
    this.emitValue(result ? 1 : 0, tick, time, this.outputChannel);
  }
}

export class ExpressionBlock extends MathBlock {
  private readonly ast: Expr;
  private readonly constants: Record<string, number>;

  constructor(
    id: string,
    readonly expression: string,
    readonly variableChannels: Record<string, string>,
    opts: MathBlockOptions,
    constants?: Record<string, number>,
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
    this.ast = parse(expression);
    this.constants = finiteRecord('ExpressionBlock', `${id}.constants`, constants);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    const names = Object.keys(this.variableChannels);
    Preconditions.nonEmpty('ExpressionBlock', `${this.id}.variables`, names);
    for (const name of names) assertName('ExpressionBlock', `${this.id}.variable.${name}`, name);
    assertUnique('ExpressionBlock', `${this.id}.variables`, names);
    for (const key of Object.keys(this.constants)) {
      Preconditions.check('ExpressionBlock', `${this.id}.constants`, 'not use reserved names t or tick', key !== 't' && key !== 'tick', key);
    }
  }

  protected step(tick: number, time: number): void {
    const env: Record<string, number> = {...this.constants, t: time, tick};
    const entries = Object.entries(this.variableChannels);
    const values = this.inputValues(entries.map(([, channel]) => channel), this.holdLast);
    if (!values) return;
    entries.forEach(([name], i) => { env[name] = values[i]; });
    this.emitValue(evaluate(this.ast, env), tick, time, this.outputChannel, {expression: this.expression});
  }
}

export class Laplacian1DBlock extends MathBlock {
  constructor(
    id: string,
    readonly coefficient: number,
    opts: MathBlockOptions,
    readonly leftChannel = 'left',
    readonly centerChannel = 'center',
    readonly rightChannel = 'right',
    readonly holdLast = true,
    readonly outputChannel: string = MATH_OUT,
  ) {
    super(id, opts);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.finite('Laplacian1DBlock', `${this.id}.coefficient`, this.coefficient);
  }

  protected step(tick: number, time: number): void {
    const left = this.inputValue(this.leftChannel, this.holdLast);
    const center = this.inputValue(this.centerChannel, this.holdLast);
    const right = this.inputValue(this.rightChannel, this.holdLast);
    if (left === undefined || center === undefined || right === undefined) return;
    this.emitValue(this.coefficient * (left - 2 * center + right), tick, time, this.outputChannel);
  }
}

export function runMathBlockDiagram(blocks: MathBlock[], opts: {maxTicks?: number; logger?: BlockModelLogger} = {}): MathBlockRunResult {
  Preconditions.nonEmpty('runMathBlockDiagram', 'blocks', blocks);
  assertUnique('runMathBlockDiagram', 'block ids', blocks.map(b => b.id));
  const maxTicks = opts.maxTicks ?? Math.max(...blocks.map(b => b.ticks)) + 1;
  opts.logger?.log({kind: 'math-block-run-start', level: 'info', blocks: blocks.length, maxTicks});
  const summary = runIterativeDES(blocks, {shuffle: false, maxTicks});
  const outputs = blocks.flatMap(b => b.outputHistory);
  opts.logger?.log({kind: 'math-block-run-finish', level: 'info', ticks: summary.ticks, reason: summary.reason, outputs: outputs.length});
  return {summary, validation: summary.validation ?? [], outputs, visualBlocks: visualBlockSpecs(blocks)};
}

export interface ODEStateSpec {
  name: string;
  initial: number;
  derivative: string;
}

export interface ODEBlockSystemParams {
  states: ODEStateSpec[];
  t0?: number;
  t1: number;
  dt: number;
  method?: IntegratorMethod;
  constants?: Record<string, number>;
}

export interface ODETraceRow {
  tick: number;
  time: number;
  state: Record<string, number>;
  derivatives: Record<string, number>;
}

export interface BlockGraphNode {
  id: string;
  kind: string;
  expression?: string;
  inputs?: Record<string, string> | string[];
  output?: string;
}

export interface BlockGraphEdge {
  from: string;
  to: string;
  fromChannel: string;
  toChannel: string;
  signal: 'MathSignal';
}

export interface ODEBlockSystemResult {
  params: ODEBlockSystemParams;
  steps: number;
  trace: ODETraceRow[];
  finalState: Record<string, number>;
  blockGraph: BlockGraphNode[];
  blockGraphEdges: BlockGraphEdge[];
  visualBlocks: VisualBlockSpec[];
  validation: ValidationCheck[];
  runSummary: IterativeRunSummary;
}

export function runODEBlockSystem(params: ODEBlockSystemParams, logger?: BlockModelLogger): ODEBlockSystemResult {
  validateODEParams(params);
  const t0 = params.t0 ?? 0;
  const steps = durationSteps('ODEBlockSystem', t0, params.t1, params.dt);
  const ticks = steps + 1;
  const opts: MathBlockOptions = {dt: params.dt, ticks, t0};
  const constants = finiteRecord('ODEBlockSystem', 'constants', params.constants);
  const method = params.method ?? 'euler';
  const names = params.states.map(s => s.name);
  const blockGraph: BlockGraphNode[] = [
    ...params.states.map(s => ({id: `integrator:${s.name}`, kind: 'integrator', inputs: [MATH_IN], output: MATH_OUT})),
    ...params.states.map(s => ({id: `rhs:${s.name}`, kind: 'expression', expression: s.derivative, inputs: Object.fromEntries(names.map(n => [n, n])), output: MATH_OUT})),
  ];
  const blockGraphEdges: BlockGraphEdge[] = [];

  const integrators = params.states.map(s => new IntegratorBlock(`integrator:${s.name}`, s.initial, opts, method));
  const rhsBlocks = params.states.map(s => {
    const variableChannels: Record<string, string> = {};
    for (const name of names) variableChannels[name] = name;
    return new ExpressionBlock(`rhs:${s.name}`, s.derivative, variableChannels, opts, constants);
  });

  for (const integ of integrators) {
    const stateName = integ.id.slice('integrator:'.length);
    for (const rhs of rhsBlocks) {
      integ.pipe(rhs, MATH_OUT, stateName);
      blockGraphEdges.push({from: integ.id, to: rhs.id, fromChannel: MATH_OUT, toChannel: stateName, signal: 'MathSignal'});
    }
  }
  for (let i = 0; i < rhsBlocks.length; i++) {
    rhsBlocks[i].pipe(integrators[i], MATH_OUT, MATH_IN);
    blockGraphEdges.push({from: rhsBlocks[i].id, to: integrators[i].id, fromChannel: MATH_OUT, toChannel: MATH_IN, signal: 'MathSignal'});
  }

  logger?.log({kind: 'math-ode-start', level: 'info', states: names, steps, dt: params.dt});
  const run = runMathBlockDiagram([...integrators, ...rhsBlocks], {logger});
  const trace: ODETraceRow[] = [];
  for (let tick = 0; tick < ticks; tick++) {
    const state: Record<string, number> = {};
    const derivatives: Record<string, number> = {};
    for (let i = 0; i < params.states.length; i++) {
      const name = params.states[i].name;
      state[name] = integrators[i].outputHistory[tick]?.value ?? NaN;
      derivatives[name] = rhsBlocks[i].outputHistory[tick]?.value ?? NaN;
    }
    const row = {tick, time: t0 + tick * params.dt, state, derivatives};
    trace.push(row);
    logger?.log({kind: 'math-ode-tick', level: 'debug', ...row});
  }
  const finalState = {...trace[trace.length - 1].state};
  const validation = run.validation.concat(validateODETrace(trace, ticks, params.t1));
  return {
    params,
    steps,
    trace,
    finalState,
    blockGraph,
    blockGraphEdges,
    visualBlocks: run.visualBlocks,
    validation,
    runSummary: run.summary,
  };
}

function validateODEParams(params: ODEBlockSystemParams): void {
  Preconditions.nonEmpty('ODEBlockSystem', 'states', params.states);
  Preconditions.integerInRange('ODEBlockSystem', 'states.length', params.states.length, 1, 100);
  const names = params.states.map(s => s.name);
  for (const name of names) assertName('ODEBlockSystem', 'state.name', name);
  assertUnique('ODEBlockSystem', 'state.name', names);
  for (const s of params.states) {
    Preconditions.finite('ODEBlockSystem', `${s.name}.initial`, s.initial);
    Preconditions.check('ODEBlockSystem', `${s.name}.derivative`, 'be non-empty', s.derivative.trim().length > 0, s.derivative);
    parse(s.derivative);
  }
  finiteRecord('ODEBlockSystem', 'constants', params.constants);
  Preconditions.check('ODEBlockSystem', 'method', 'be euler or trapezoid', params.method === undefined || params.method === 'euler' || params.method === 'trapezoid', params.method);
}

function validateODETrace(trace: readonly ODETraceRow[], ticks: number, t1: number): ValidationCheck[] {
  const finite = trace.every(row =>
    Number.isFinite(row.time) &&
    Object.values(row.state).every(Number.isFinite) &&
    Object.values(row.derivatives).every(Number.isFinite));
  return [
    {name: 'ode-trace-length', passed: trace.length === ticks, observed: String(trace.length), expected: String(ticks), group: 'math-ode'},
    {name: 'ode-trace-finite', passed: finite, expected: 'all state and derivative values finite', group: 'math-ode'},
    {
      name: 'ode-final-time',
      passed: Math.abs(trace[trace.length - 1].time - t1) <= 1e-9 * Math.max(1, Math.abs(t1)),
      observed: trace[trace.length - 1].time.toPrecision(12),
      expected: t1.toPrecision(12),
      group: 'math-ode',
    },
  ];
}

export interface Heat1DBlockParams {
  cells: number;
  length: number;
  alpha: number;
  t0?: number;
  t1: number;
  dt: number;
  initialExpression?: string;
  initialValues?: number[];
  leftBoundary?: number;
  rightBoundary?: number;
  constants?: Record<string, number>;
}

export interface Heat1DTraceRow {
  tick: number;
  time: number;
  values: number[];
  min: number;
  max: number;
  mean: number;
}

export interface Heat1DBlockResult {
  params: Heat1DBlockParams;
  dx: number;
  cfl: number;
  steps: number;
  x: number[];
  trace: Heat1DTraceRow[];
  finalValues: number[];
  blockGraph: BlockGraphNode[];
  blockGraphEdges: BlockGraphEdge[];
  visualBlocks: VisualBlockSpec[];
  validation: ValidationCheck[];
  runSummary: IterativeRunSummary;
}

export function runHeat1DBlockGrid(params: Heat1DBlockParams, logger?: BlockModelLogger): Heat1DBlockResult {
  validateHeatParams(params);
  const t0 = params.t0 ?? 0;
  const steps = durationSteps('Heat1DBlockGrid', t0, params.t1, params.dt);
  const ticks = steps + 1;
  const dx = params.length / (params.cells - 1);
  const coefficient = params.alpha / (dx * dx);
  const cfl = coefficient * params.dt;
  Preconditions.check('Heat1DBlockGrid', 'alpha*dt/dx^2', 'be <= 0.5 for explicit block-grid stability', cfl <= 0.5 + 1e-12, cfl);
  const x = Array.from({length: params.cells}, (_, i) => i * dx);
  const initial = buildHeatInitialValues(params, x);
  const leftBoundary = params.leftBoundary ?? initial[0];
  const rightBoundary = params.rightBoundary ?? initial[initial.length - 1];
  initial[0] = leftBoundary;
  initial[initial.length - 1] = rightBoundary;

  const opts: MathBlockOptions = {dt: params.dt, ticks, t0};
  const cells: MathBlock[] = [];
  const blockGraphEdges: BlockGraphEdge[] = [];
  for (let i = 0; i < params.cells; i++) {
    if (i === 0) cells.push(new ConstantSourceBlock(`cell:${i}`, leftBoundary, opts));
    else if (i === params.cells - 1) cells.push(new ConstantSourceBlock(`cell:${i}`, rightBoundary, opts));
    else cells.push(new IntegratorBlock(`cell:${i}`, initial[i], opts, 'euler'));
  }
  const laplacians: Laplacian1DBlock[] = [];
  for (let i = 1; i < params.cells - 1; i++) {
    const lap = new Laplacian1DBlock(`laplacian:${i}`, coefficient, opts);
    cells[i - 1].pipe(lap, MATH_OUT, 'left');
    cells[i].pipe(lap, MATH_OUT, 'center');
    cells[i + 1].pipe(lap, MATH_OUT, 'right');
    lap.pipe(cells[i], MATH_OUT, MATH_IN);
    blockGraphEdges.push({from: cells[i - 1].id, to: lap.id, fromChannel: MATH_OUT, toChannel: 'left', signal: 'MathSignal'});
    blockGraphEdges.push({from: cells[i].id, to: lap.id, fromChannel: MATH_OUT, toChannel: 'center', signal: 'MathSignal'});
    blockGraphEdges.push({from: cells[i + 1].id, to: lap.id, fromChannel: MATH_OUT, toChannel: 'right', signal: 'MathSignal'});
    blockGraphEdges.push({from: lap.id, to: cells[i].id, fromChannel: MATH_OUT, toChannel: MATH_IN, signal: 'MathSignal'});
    laplacians.push(lap);
  }

  logger?.log({kind: 'math-heat1d-start', level: 'info', cells: params.cells, steps, dx, cfl});
  const run = runMathBlockDiagram([...cells, ...laplacians], {logger});
  const trace: Heat1DTraceRow[] = [];
  for (let tick = 0; tick < ticks; tick++) {
    const values = cells.map(cell => cell.outputHistory[tick]?.value ?? NaN);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const row = {tick, time: t0 + tick * params.dt, values, min, max, mean};
    trace.push(row);
    logger?.log({kind: 'math-heat1d-tick', level: 'debug', tick, time: row.time, min, max, mean});
  }
  const validation = run.validation.concat(validateHeatTrace(trace, ticks, initial, leftBoundary, rightBoundary));
  return {
    params,
    dx,
    cfl,
    steps,
    x,
    trace,
    finalValues: trace[trace.length - 1].values.slice(),
    blockGraph: [
      ...cells.map((_, i) => ({id: `cell:${i}`, kind: i === 0 || i === params.cells - 1 ? 'constant-boundary' : 'integrator', output: MATH_OUT})),
      ...laplacians.map((_, i) => ({id: `laplacian:${i + 1}`, kind: 'laplacian-1d', inputs: ['left', 'center', 'right'], output: MATH_OUT})),
    ],
    blockGraphEdges,
    visualBlocks: run.visualBlocks,
    validation,
    runSummary: run.summary,
  };
}

function validateHeatParams(params: Heat1DBlockParams): void {
  Preconditions.integerInRange('Heat1DBlockGrid', 'cells', params.cells, 3, 1000);
  Preconditions.positive('Heat1DBlockGrid', 'length', params.length);
  Preconditions.nonNegative('Heat1DBlockGrid', 'alpha', params.alpha);
  Preconditions.positive('Heat1DBlockGrid', 'dt', params.dt);
  Preconditions.finite('Heat1DBlockGrid', 't1', params.t1);
  if (params.t0 !== undefined) Preconditions.finite('Heat1DBlockGrid', 't0', params.t0);
  if (params.initialValues !== undefined) {
    Preconditions.lengthEq('Heat1DBlockGrid', 'initialValues', params.initialValues, params.cells);
    Preconditions.allFinite('Heat1DBlockGrid', 'initialValues', params.initialValues);
  }
  if (params.initialExpression !== undefined) parse(params.initialExpression);
  if (params.leftBoundary !== undefined) Preconditions.finite('Heat1DBlockGrid', 'leftBoundary', params.leftBoundary);
  if (params.rightBoundary !== undefined) Preconditions.finite('Heat1DBlockGrid', 'rightBoundary', params.rightBoundary);
  finiteRecord('Heat1DBlockGrid', 'constants', params.constants);
}

function buildHeatInitialValues(params: Heat1DBlockParams, x: readonly number[]): number[] {
  if (params.initialValues !== undefined) return params.initialValues.slice();
  const expression = params.initialExpression ?? 'sin(pi*x/length)';
  const ast = parse(expression);
  const constants = {pi: Math.PI, e: Math.E, length: params.length, ...(params.constants ?? {})};
  return x.map(xi => {
    const value = evaluate(ast, {...constants, x: xi});
    Preconditions.finite('Heat1DBlockGrid', 'initialExpression', value);
    return value;
  });
}

function validateHeatTrace(
  trace: readonly Heat1DTraceRow[],
  ticks: number,
  initial: readonly number[],
  leftBoundary: number,
  rightBoundary: number,
): ValidationCheck[] {
  const finite = trace.every(row => Number.isFinite(row.time) && row.values.every(Number.isFinite));
  const lo = Math.min(...initial, leftBoundary, rightBoundary) - 1e-9;
  const hi = Math.max(...initial, leftBoundary, rightBoundary) + 1e-9;
  const maxPrinciple = trace.every(row => row.values.every(v => v >= lo && v <= hi));
  return [
    {name: 'heat-trace-length', passed: trace.length === ticks, observed: String(trace.length), expected: String(ticks), group: 'math-heat1d'},
    {name: 'heat-trace-finite', passed: finite, expected: 'all grid values finite', group: 'math-heat1d'},
    {name: 'heat-maximum-principle', passed: maxPrinciple, observed: `[${Math.min(...trace.map(r => r.min)).toPrecision(8)}, ${Math.max(...trace.map(r => r.max)).toPrecision(8)}]`, expected: `[${lo.toPrecision(8)}, ${hi.toPrecision(8)}]`, group: 'math-heat1d'},
  ];
}
