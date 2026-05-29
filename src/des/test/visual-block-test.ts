'use strict';

import * as fs from 'fs';
import {DESStation, Token} from '../general/des-base/station';
import {runIterativeDES} from '../general/des-base/runner';
import {FunctionEntity} from '../general/des-base/transform-entity';
import {VisualBlock, renderVisualBlocks, visualBlockSpecs} from '../general/des-base/visual-block';
import {FrameRecorder} from '../animation/frame-recorder';

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

let pass = 0;
let fail = 0;

function check(label: string, ok: boolean, detail = ''): void {
  if (ok) {
    pass++;
    console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`);
  } else {
    fail++;
    console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`);
  }
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

async function main(): Promise<void> {
  console.log('\nvisual-block - always-rendered block wrapper');

  const block = new VisualBlock('pipeline-block', {
    label: 'Pipeline block',
    kind: 'program-block',
    layout: {x: 32, y: 28, w: 210, h: 72},
    ports: {inputs: ['in'], outputs: ['out']},
  });
  const doubler = block.addSubstation(new FunctionEntity<NumberToken, NumberToken>(
    'double-transform',
    token => new NumberToken(token.value * 2),
    {inputChannels: 'in', outputChannel: 'out'},
  ));
  block.addVisualMember({id: 'legacy-stationary-entity'});
  block.exposeInput('in', doubler, 'in');
  block.exposeOutput(doubler, 'out', 'out');

  const sink = new CollectSink<NumberToken>('sink');
  block.pipe(sink, 'out', 'out');
  block.take(new NumberToken(7), 'in');
  const summary = runIterativeDES([block, sink], {shuffle: false, maxTicks: 4});

  check('1.1 VisualBlock can contain and run FunctionEntity substations',
    summary.reason === 'done' && sink.received.length === 1 && sink.received[0].value === 14,
    `reason=${summary.reason} received=${sink.received.map(t => t.value).join(',')}`);

  const specs = visualBlockSpecs([block]);
  check('1.2 VisualBlock spec is marked always-rendered',
    specs.length === 1 && specs[0].alwaysRenderInHtml === true && specs[0].id === 'pipeline-block');
  check('1.3 VisualBlock records contained DES and non-DES members',
    specs[0].contains.some(c => c.id === 'double-transform') &&
    specs[0].contains.some(c => c.id === 'legacy-stationary-entity'),
    `contains=${specs[0].contains.map(c => c.id ?? c.kind).join(',')}`);

  const shapes = renderVisualBlocks(specs);
  check('1.4 VisualBlock renders to HTML/SVG shape primitives',
    shapes.some(s => s.kind === 'rect' && s.visualBlockId === 'pipeline-block') &&
    shapes.some(s => s.kind === 'text' && s.visualBlockId === 'pipeline-block'));

  const framesPath = 'out/visual-block-test.frames.jsonl';
  const htmlPath = 'out/visual-block-test.html';
  const rec = new FrameRecorder({
    framesPath,
    htmlPath,
    width: 360,
    height: 180,
    fps: 1,
    title: 'VisualBlock test',
    visualBlocks: [block],
  });
  rec.frame(0, 0, () => ({shapes: [], caption: 'VisualBlock should still render'}));
  await rec.finish();
  const frames = fs.readFileSync(framesPath, 'utf8');
  const html = fs.readFileSync(htmlPath, 'utf8');
  check('1.5 FrameRecorder appends VisualBlock shapes to every HTML frame',
    frames.includes('"visualBlockId":"pipeline-block"') && html.includes('Pipeline block'));

  const source = VisualBlock.source('visual-source', [{id: 'value', kind: 'NumberToken'}]);
  const transform = new VisualBlock('visual-transform', {
    role: 'transform',
    ports: {
      inputs: [{id: 'in', kind: 'NumberToken'}],
      outputs: [{id: 'out', kind: 'NumberToken'}],
    },
  });
  const visualSink = VisualBlock.sink('visual-sink', [{id: 'value', kind: 'NumberToken'}]);
  source.connectTo(transform, {fromPort: 'value', toPort: 'in', wireDES: false});
  transform.connectTo(visualSink, {fromPort: 'out', toPort: 'value', wireDES: false});
  const sourceSpec = source.visualBlockSpec();
  const transformSpec = transform.visualBlockSpec();
  const sinkSpec = visualSink.visualBlockSpec();
  check('1.6 source, transform, and sink ports expose direction and kind',
    sourceSpec.role === 'source' &&
    sourceSpec.ports.inputs.length === 0 &&
    sourceSpec.ports.outputs[0].kind === 'NumberToken' &&
    transformSpec.ports.inputs[0].direction === 'in' &&
    transformSpec.ports.outputs[0].direction === 'out' &&
    sinkSpec.role === 'sink' &&
    sinkSpec.ports.outputs.length === 0);
  check('1.7 VisualBlock connection records link typed output and input ports',
    sourceSpec.connectionsOut[0].kind === 'NumberToken' &&
    sourceSpec.connectionsOut[0].from.portId === 'value' &&
    sourceSpec.connectionsOut[0].to.blockId === 'visual-transform' &&
    sinkSpec.connectionsIn[0].from.blockId === 'visual-transform');

  const wrongKindSink = VisualBlock.sink('wrong-kind-sink', [{id: 'value', kind: 'StringToken'}]);
  check('1.8 incompatible port kinds fail loudly',
    throws(() => source.connectTo(wrongKindSink, {fromPort: 'value', toPort: 'value', wireDES: false})));
  check('1.9 source and sink roles enforce one-sided ports',
    throws(() => new VisualBlock('bad-source', {role: 'source', ports: {inputs: ['in'], outputs: ['out']}})) &&
    throws(() => new VisualBlock('bad-sink', {role: 'sink', ports: {inputs: ['in'], outputs: ['out']}})));

  console.log(`\nvisual-block-test summary: ${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
