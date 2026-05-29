#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// Smoke test for the animation plugin.
//
// Verifies:
//   - FrameRecorder writes a parseable header+frames+charts JSONL
//   - readAnimation reads it back without loss
//   - buildHTML produces a non-empty, syntactically reasonable HTML file
//   - HTML contains the embedded Animation JSON
//   - HTML contains references to renderShape, renderChart, requestAnimationFrame
//
// Run with:
//   npm run build
//   node dist/des/test/animation-test.js
// =============================================================================

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {FrameRecorder, readAnimation} from '../animation/frame-recorder';
import {buildHTML} from '../animation/html-player';
import {Animation, Frame, Shape} from '../animation/types';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (ok) { pass++; console.log(`  PASS    ${label}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'des-anim-test-'));
const framesPath = path.join(tmpDir, 'sample.frames.jsonl');
const htmlPath   = path.join(tmpDir, 'sample.html');

console.log('\nT1  Write + read roundtrip');

async function run() {
  const rec = new FrameRecorder({
    framesPath, htmlPath,
    width: 400, height: 300, fps: 30,
    title: 'Test',
    subtitle: 'sub',
  });
  const N = 20;
  for (let tick = 0; tick < N; tick++) {
    const t = tick * 0.1;
    rec.frame(t, tick, () => {
      const shapes: Shape[] = [
        {kind: 'circle', x: 50 + tick * 10, y: 150, r: 10, fill: '#f00'},
        {kind: 'rect',   x: 0, y: 280, w: 400, h: 20, fill: '#eee', stroke: '#ccc'},
        {kind: 'text',   x: 200, y: 30, text: `tick ${tick}`, fontSize: 14, anchor: 'middle'},
      ];
      return {shapes, caption: `frame at t=${t.toFixed(2)}`};
    });
  }
  rec.setCharts([{
    x: 0, y: 200, w: 400, h: 80,
    title: 'sample',
    series: [{
      label: 'sin',
      color: '#08f',
      t: Array.from({length: 20}, (_, i) => i * 0.1),
      y: Array.from({length: 20}, (_, i) => Math.sin(i * 0.3)),
    }],
  }]);
  const anim = await rec.finish();

  check('framesPath exists',   fs.existsSync(framesPath));
  check('htmlPath   exists',   fs.existsSync(htmlPath));
  check('rec.getFrameCount()', rec.getFrameCount() === N, `${rec.getFrameCount()}`);
  check('anim.frames.length',  anim.frames.length === N);
  check('anim.charts present', !!anim.charts && anim.charts.length === 1);
  check('chart series ok',     anim.charts![0].series.length === 1);
  check('chart series len',    anim.charts![0].series[0].t.length === 20);

  console.log('\nT2  Read back from disk');
  const reread = readAnimation(framesPath);
  check('round-trip frames eq', reread.frames.length === N);
  check('round-trip width',     reread.width === 400);
  check('round-trip title',     reread.title === 'Test');
  check('round-trip subtitle',  reread.subtitle === 'sub');
  check('round-trip charts',    !!reread.charts && reread.charts.length === 1);

  console.log('\nT3  HTML output structure');
  const html = fs.readFileSync(htmlPath, 'utf8');
  check('HTML starts with <!DOCTYPE>', html.startsWith('<!DOCTYPE html>'));
  check('HTML contains anim-data',     html.includes('id="anim-data"'));
  check('HTML contains stage svg',     html.includes('id="stage"'));
  check('HTML contains play button',   html.includes('id="play"'));
  check('HTML contains scrubber',      html.includes('id="scrub"'));
  check('HTML contains JSON title',    html.includes('"title":"Test"'));
  check('HTML contains renderShape',   html.includes('function renderShape'));
  check('HTML contains renderChart',   html.includes('function renderChart'));
  check('HTML contains rAF call',      html.includes('requestAnimationFrame'));

  console.log('\nT4  buildHTML escapes title');
  const evilAnim: Animation = {
    width: 100, height: 100, fps: 30,
    title: 'Hello <script>alert(1)</script>',
    frames: [],
  };
  const evilHtml = buildHTML(evilAnim);
  check('XSS in title is escaped',
        !evilHtml.includes('<script>alert(1)</script>'));
  check('escaped title appears',
        evilHtml.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));

  console.log('\nT5  recordEveryTicks filter');
  const f2 = path.join(tmpDir, 'f2.frames.jsonl');
  const rec2 = new FrameRecorder({
    framesPath: f2, width: 100, height: 100, fps: 30,
    recordEveryTicks: 5,
  });
  for (let i = 0; i < 50; i++) rec2.frame(i * 0.1, i, () => ({shapes: []}));
  await rec2.finish();
  const a2 = readAnimation(f2);
  check('recordEveryTicks=5 yields 10 frames', a2.frames.length === 10,
        `${a2.frames.length}`);

  console.log('\nsummary: ' + pass + ' pass, ' + fail + ' fail');
  process.exit(fail === 0 ? 0 : 1);
}

run();
