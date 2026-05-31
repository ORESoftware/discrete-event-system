// RUST MIGRATION: Prefer moving these focused checks into `src/des/general/des_base/transform_entity.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate entity-transform scenarios into `#[test]` functions returning `Result<()>`; replace helper checks with `assert!`, `assert_eq!`, and approximate-float helpers; keep token fixtures deterministic.

'use strict';

import {ChannelName, DESStation, Token} from '../general/des-base/station';
import {runIterativeDES} from '../general/des-base/runner';
import {SingleTokenSourceStation} from '../general/des-base/learning-optimization';
import {FunctionEntity, MemoryTransformEntity, PureTransformEntity} from '../general/des-base/transform-entity';

class NumberToken implements Token {
  constructor(readonly value: number) {}
}

class CollectSink<T extends Token> extends DESStation {
  readonly received: T[] = [];

  override hasWork(): boolean {
    return Object.values(this.inboxSizes()).some(n => n > 0);
  }

  runTimeStep(): void {
    for (const channel of Object.keys(this.inboxSizes())) {
      this.received.push(...this.drain<T>(channel));
    }
  }
}

class SquareNumberTransform extends PureTransformEntity<NumberToken, NumberToken> {
  transform(token: NumberToken): NumberToken {
    return new NumberToken(token.value * token.value);
  }
}

class RunningSumTransform extends MemoryTransformEntity<NumberToken, NumberToken, number> {
  protected transformQueued(token: NumberToken): NumberToken {
    this.previous += token.value;
    return new NumberToken(this.previous);
  }
}

let pass = 0;
let fail = 0;

function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' - ' + detail : ''));
  cond ? pass++ : fail++;
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

console.log('\ntransform-entity - zero-backlog function-as-station primitive');

{
  const source = new SingleTokenSourceStation('source', 'in', () => new NumberToken(4));
  const square = new FunctionEntity<NumberToken, NumberToken>(
    'square-fn',
    token => new NumberToken(token.value * token.value),
    {inputChannels: 'in', outputChannel: 'out'},
  );
  const sink = new CollectSink<NumberToken>('sink');
  source.pipe(square, 'in', 'in');
  square.pipe(sink, 'out', 'out');

  const summary = runIterativeDES([source, square, sink], {shuffle: false, maxTicks: 5});
  expect('pipeline completes in one tick plus quiescence',
    summary.reason === 'done' && summary.ticks === 1,
    `reason=${summary.reason}, ticks=${summary.ticks}`);
  expect('transform emits squared value immediately',
    sink.received.length === 1 && sink.received[0].value === 16,
    `received=[${sink.received.map(t => t.value).join(',')}]`);
  expect('transform has no internal backlog',
    !square.hasWork() && Object.keys(square.inboxSizes()).length === 0);
  expect('processed/emitted counters update',
    square.processedCount === 1 && square.emittedCount === 1 && square.droppedCount === 0,
    `processed=${square.processedCount}, emitted=${square.emittedCount}, dropped=${square.droppedCount}`);
}

{
  const fanout = new FunctionEntity<NumberToken, NumberToken>(
    'fanout',
    token => [new NumberToken(token.value), new NumberToken(token.value + 1)],
  );
  const sink = new CollectSink<NumberToken>('fanout-sink');
  fanout.pipe(sink);
  fanout.take(new NumberToken(10));
  runIterativeDES([fanout, sink], {shuffle: false, maxTicks: 2});
  expect('returning an array emits multiple tokens',
    JSON.stringify(sink.received.map(t => t.value)) === '[10,11]',
    `received=[${sink.received.map(t => t.value).join(',')}]`);
}

{
  const square = new SquareNumberTransform('pure-square');
  const sink = new CollectSink<NumberToken>('pure-transform-sink');
  square.pipe(sink);
  square.take(new NumberToken(5));
  runIterativeDES([square, sink], {shuffle: false, maxTicks: 2});
  expect('PureTransformEntity subclasses model ordinary functions',
    sink.received.length === 1 && sink.received[0].value === 25,
    `received=[${sink.received.map(t => t.value).join(',')}]`);
}

{
  const running = new RunningSumTransform('running-sum', 0);
  const sink = new CollectSink<NumberToken>('memory-transform-sink');
  running.pipe(sink);
  running.take(new NumberToken(2));
  running.take(new NumberToken(3));
  runIterativeDES([running, sink], {shuffle: false, maxTicks: 2});
  expect('MemoryTransformEntity keeps explicit local memory behind runTimeStep queueing',
    JSON.stringify(sink.received.map(t => t.value)) === '[2,5]' &&
    running.previous === 5 &&
    !running.hasWork(),
    `received=[${sink.received.map(t => t.value).join(',')}], previous=${running.previous}`);
}

{
  const even = new CollectSink<NumberToken>('even');
  const odd = new CollectSink<NumberToken>('odd');
  const router = new FunctionEntity<NumberToken, NumberToken>(
    'route-number',
    (token, ctx) => {
      ctx.emit(token, token.value % 2 === 0 ? 'even' : 'odd');
      return undefined;
    },
  );
  router.pipe(even, 'even', 'in');
  router.pipe(odd, 'odd', 'in');
  router.take(new NumberToken(2));
  router.take(new NumberToken(3));
  runIterativeDES([router, even, odd], {shuffle: false, maxTicks: 2});
  expect('context.emit can route to named output channels',
    even.received.length === 1 && even.received[0].value === 2 &&
    odd.received.length === 1 && odd.received[0].value === 3);
  expect('manual ctx.emit increments emitted count',
    router.emittedCount === 2 && router.droppedCount === 0,
    `emitted=${router.emittedCount}, dropped=${router.droppedCount}`);
}

{
  const strict = new FunctionEntity<NumberToken, NumberToken>(
    'strict-channel',
    token => token,
    {inputChannels: ['allowed'] as ChannelName[]},
  );
  expect('unexpected input channel fails loudly',
    throws(() => strict.take(new NumberToken(1), 'wrong')));
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
