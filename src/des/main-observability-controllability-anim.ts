'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-observability-controllability-anim.rs   (fn main)
// 1:1 file move. Generates an HTML slideshow walking through the
// controllability/observability tests for LTI, MDP, and POMDP systems.
//
// Conversion notes (file-specific):
//   - class ObsCtrlAnimator -> struct + impl; async run() -> fn main.
//   - FrameRecorder + ObsCtrlScene -> use crate::des::animation::...
// =============================================================================

// =============================================================================
// main-observability-controllability-anim.ts — generate an HTML slideshow
// animation walking through the controllability / observability tests for
// linear, MDP, and POMDP systems.
//
//   npm run obs-ctrl-anim
// =============================================================================

import * as path from 'path';
import {FrameRecorder} from './animation/frame-recorder';
import {OC_STAGE_H, OC_STAGE_W, ObsCtrlScene} from './animation/scenes/obs-ctrl-scene';

class ObsCtrlAnimator {
  private readonly fps = 24;
  private readonly holdFrames = 30;   // ~1.25s per storyboard step

  async run(): Promise<void> {
    const scene = new ObsCtrlScene();
    const steps = scene.steps();
    const out = path.join('out', 'obs-ctrl', 'animation.html');
    const recorder = new FrameRecorder({
      framesPath: out.replace(/\.html$/, '.frames.jsonl'), htmlPath: out,
      width: OC_STAGE_W, height: OC_STAGE_H, fps: this.fps,
      title: 'Controllability & Observability — structural evaluator',
      subtitle: 'Kalman rank tests · MDP reachability · POMDP distinguishability',
      background: '#0b1021',
    });
    let tick = 0;
    for (const step of steps) {
      for (let h = 0; h < this.holdFrames; h++) {
        recorder.frame(tick / this.fps, tick, () => step);
        tick++;
      }
    }
    await recorder.finish();
    console.log(`Obs/Ctrl animation: ${path.resolve(out)}  (${steps.length} steps, ${tick} frames)`);
  }
}

new ObsCtrlAnimator().run().catch(e => { console.error(e); process.exit(1); });
