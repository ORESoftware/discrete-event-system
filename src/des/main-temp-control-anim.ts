'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-temp-control-anim.rs   (fn main)
// 1:1 file move. Generates an HTML animation of the temperature-control DES
// for a chosen controller.
//
// Conversion notes (file-specific):
//   - process.argv flag parsing (--controller/--out) -> std::env::args / clap;
//     process.exit -> std::process::exit.
//   - controller union 'bang-bang'|'pid'|'fuzzy'|'mdp-mpc' -> enum.
//   - imports general/temp-control + animation scene -> use crate::des::...
//   - async main -> fn main.
// =============================================================================

// =============================================================================
// main-temp-control-anim.ts — Generate an HTML animation of the
// temperature-control DES with a chosen controller.
// =============================================================================

import * as path from 'path';
import {runTempControl, ControllerSpec, SimConfig} from './general/temp-control';
import {FrameRecorder} from './animation/frame-recorder';
import {STAGE_W, STAGE_H, buildTempControlAnimation} from './animation/scenes/temp-control-scene';

interface Args {
  controller: 'bang-bang' | 'pid' | 'fuzzy' | 'mdp-mpc';
  out: string;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  let controller: Args['controller'] = 'pid';
  let out = path.join('out', 'temp-control', 'animation.html');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--controller' && i + 1 < argv.length) {
      const v = argv[++i] as Args['controller'];
      if (['bang-bang', 'pid', 'fuzzy', 'mdp-mpc'].includes(v)) controller = v;
      else { console.error(`unknown controller "${v}"`); process.exit(1); }
    } else if (argv[i] === '--out' && i + 1 < argv.length) {
      out = argv[++i];
    } else if (argv[i] === '-h' || argv[i] === '--help') {
      console.log('Usage: node main-temp-control-anim.js [--controller bang-bang|pid|fuzzy|mdp-mpc] [--out path]');
      process.exit(0);
    }
  }
  return {controller, out};
}

async function main(): Promise<void> {
  const args = parseArgs();
  const ctrl: Record<Args['controller'], ControllerSpec> = {
    'bang-bang': {kind: 'bang-bang'},
    'pid':       {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
    'fuzzy':     {kind: 'fuzzy'},
    'mdp-mpc':   {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 1.0},
  };
  const cfg: SimConfig = {
    T_target: 70, band: 2, duration_h: 24, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 6,
    seed: 42,
    controller: ctrl[args.controller],
  };
  const t0 = Date.now();
  const r = runTempControl(cfg);
  const elapsed = Date.now() - t0;
  console.log(`Simulated ${cfg.duration_h}h with ${args.controller} in ${elapsed}ms`);
  console.log(`  energy = ${r.energy_kWh.toFixed(2)} kWh`);
  console.log(`  comfort = ${(100 * r.comfort_pct).toFixed(1)}%`);
  console.log(`  cost = $${r.cost_$.toFixed(2)}`);

  // Build the animation
  const ctlNames: Record<Args['controller'], string> = {
    'bang-bang': 'Bang-bang',
    'pid': 'PID (filtered-D)',
    'fuzzy': 'Fuzzy-PI (Mamdani)',
    'mdp-mpc': 'MDP-MPC (H=6h)',
  };
  const recordEvery = 5;        // 5-min frames → 24h × 12fps
  const {frames, charts} = buildTempControlAnimation(r, ctlNames[args.controller], recordEvery);

  const framesPath = args.out.replace(/\.html$/, '.frames.jsonl');
  const recorder = new FrameRecorder({
    framesPath, htmlPath: args.out,
    width: STAGE_W, height: STAGE_H, fps: 12,
    title: `Temperature Control — ${ctlNames[args.controller]}`,
    subtitle: `24-hour winter scenario, target = ${cfg.T_target}°F ± ${cfg.band}°F   |   energy = ${r.energy_kWh.toFixed(2)} kWh, comfort = ${(100 * r.comfort_pct).toFixed(1)}%, cost = $${r.cost_$.toFixed(2)}`,
    background: '#f9fafb',
  });
  for (const f of frames) {
    recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
  }
  recorder.setCharts(charts);
  await recorder.finish();
  console.log(`\nFrames: ${framesPath}`);
  console.log(`HTML:   ${args.out}`);
  console.log(`Open in browser: file://${path.resolve(args.out)}`);
}

main().catch(e => { console.error(e); process.exit(1); });
