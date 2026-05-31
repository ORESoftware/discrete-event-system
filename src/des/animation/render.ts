#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/render.rs   (a `fn main()` binary)
// 1:1 file move. CLI entry point: read a `.frames.jsonl` file and emit standalone HTML.
//
// Declarations → Rust:
//   function main()  -> `fn main()` (this file is a binary; consider src/bin/render.rs)
//
// Conversion notes (file-specific):
//   - `#!/usr/bin/env ts-node` shebang + `require.main === module` guard -> just `fn main()`.
//   - `process.argv.slice(2)` -> `std::env::args().skip(1)`; `process.exit(n)` -> `std::process::exit(n)`.
//   - `fs.existsSync` / `fs.writeFileSync` -> `std::path::Path::exists` / `std::fs::write`.
//   - `inputPath.replace(/\.frames\.jsonl$/, ...)` regex path munging -> use
//     `std::path` (`with_extension`/`file_stem`) or the `regex` crate.
//   - calls `readAnimation` / `buildHTML` from sibling modules -> `use` from
//     `crate::des::animation::{frame_recorder, html_player}`.
// =============================================================================

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
