#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/des/animation/render.rs
// - Keep file-for-file as the post-hoc renderer module; a thin src/bin wrapper can call main_result() if this becomes a CLI binary.
// - Convert process.argv/process.exit flow to a Result-returning main_result(args: impl Iterator<Item=String>) and map errors to exit codes at the boundary.
// - Filesystem paths should use PathBuf/std::fs; preserve the .frames.jsonl -> .html suffix rule with typed path helpers.
// - readAnimation/buildHTML become frame_recorder::read_animation and html_player::build_html returning Results.
'use strict';

// Post-hoc renderer: read a `.frames.jsonl` file produced by FrameRecorder
// and emit a self-contained HTML animation. Useful when you ran the
// simulation earlier with ANIMATE=1 (or piped frames out of a different
// process) and want to re-render to HTML without re-running.
//
// Usage:
//   node dist/des/animation/render.js <input.frames.jsonl> [output.html]
//
// If output is omitted, the HTML is written next to the input with a
// `.html` suffix in place of `.frames.jsonl`.

import * as fs from 'fs';
import {readAnimation} from './frame-recorder';
import {buildHTML} from './html-player';

function main() {
  const args = process.argv.slice(2);
  if (args.length < 1 || args.length > 2) {
    console.error('usage: render <input.frames.jsonl> [output.html]');
    process.exit(2);
  }
  const inputPath = args[0];
  const outputPath = args[1] ?? inputPath.replace(/\.frames\.jsonl$/, '.html').replace(/\.jsonl$/, '.html');

  if (!fs.existsSync(inputPath)) {
    console.error(`render: input not found: ${inputPath}`);
    process.exit(1);
  }
  const anim = readAnimation(inputPath);
  fs.writeFileSync(outputPath, buildHTML(anim));
  console.log(`render: ${inputPath} → ${outputPath}  (${anim.frames.length} frames, ${anim.width}×${anim.height})`);
}

if (require.main === module) main();
