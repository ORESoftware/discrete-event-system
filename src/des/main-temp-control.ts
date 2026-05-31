// RUST MIGRATION: target src/bin/main_temp_control.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-temp-control.rs   (fn main)
// 1:1 file move. CLI driver comparing bang-bang / PID / fuzzy-PI / MDP-MPC
// controllers on one scenario, plus an MDP-MPC sensitivity sweep.
//
// Conversion notes (file-specific):
//   - controller specs (bang-bang/pid/fuzzy/mdp-mpc) -> enum ControllerSpec.
//   - noisy outdoor-temperature sampling -> inject RandomSource/SeededRandom.
//   - use crate::des::general::temp_control; fs write -> std::fs.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-temp-control.ts — CLI driver for the temperature-control DES.
//
// Runs four interchangeable controllers — bang-bang, PID, fuzzy-PI, MDP-MPC —
// on the SAME 24-hour winter-day scenario (cold morning, mild afternoon,
// noisy diurnal outdoor temperature) and prints a side-by-side energy /
// comfort / cost comparison. Then runs an MDP-MPC sensitivity sweep over
// (forecast horizon × tracking weight) showing how forecast-aware planning
// trades energy against tracking quality.
//
// The DES topology is the same in every run; only the ControllerStation
// changes. That's the whole point of the model patterns in this repo —
// stations are functionally pluggable as long as their I/O contract holds.
// =============================================================================

import {runTempControl, RunResult, ControllerSpec, SimConfig} from './general/temp-control';
import * as fs from 'fs';
import * as path from 'path';

interface RunRow {
  name: string;
  result: RunResult;
}

function fmt(r: RunResult, name: string): string {
  const minT = Math.min(...r.T_in);
  const maxT = Math.max(...r.T_in);
  return [
    name.padEnd(22),
    'energy=' + r.energy_kWh.toFixed(2).padStart(7) + ' kWh',
    'comfort=' + (100 * r.comfort_pct).toFixed(1).padStart(5) + '%',
    'violation=' + r.violation_Fh.toFixed(2).padStart(6) + ' °F·h',
    'cost=$' + r.cost_$.toFixed(2).padStart(6),
    'T_in=[' + minT.toFixed(2) + ', ' + maxT.toFixed(2) + ']',
  ].join('   ');
}

function header(s: string): void {
  console.log();
  console.log('═'.repeat(120));
  console.log('  ' + s);
  console.log('═'.repeat(120));
}

function main(): void {
  const SEED = 42;
  const cfg: SimConfig = {
    T_target: 70, band: 2, duration_h: 24, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 6,
    seed: SEED,
    controller: {kind: 'bang-bang'},
  };

  header('TEMPERATURE CONTROL — 24-hour winter day, T_target = 70°F ± 2°F');
  console.log('  Outdoor: cold winter day. Mean 25°F, ±15°F diurnal swing, 1.5°F noise.');
  console.log('           Coldest at 3 AM (≈ 10°F), warmest at 3 PM (≈ 40°F).');
  console.log('  House:   τ = 12 h thermal time constant, heater max 5 kW.');
  console.log('  Sensors: indoor sensor noise σ = 0.2°F, forecast noise σ = 1.5°F.');
  console.log('  Cost:    $0.15/kWh energy + $0.50/(°F)²/h comfort violation outside band.');

  header('CONTROLLER COMPARISON (same scenario, same seed)');
  const runs: RunRow[] = [];
  for (const ctrl of [
    {name: 'bang-bang',         spec: {kind: 'bang-bang'} as ControllerSpec},
    {name: 'PID (filtered-D)',  spec: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5} as ControllerSpec},
    {name: 'Fuzzy-PI (Mamdani)', spec: {kind: 'fuzzy'} as ControllerSpec},
    {name: 'MDP-MPC (H=1h)',    spec: {kind: 'mdp-mpc', horizon_h: 1, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 1.0} as ControllerSpec},
    {name: 'MDP-MPC (H=6h)',    spec: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 1.0} as ControllerSpec},
    {name: 'MDP-MPC (loose,w=0.05)', spec: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 0.05} as ControllerSpec},
  ]) {
    const t0 = Date.now();
    const r = runTempControl({...cfg, controller: ctrl.spec});
    const dt = Date.now() - t0;
    console.log('  ' + fmt(r, ctrl.name) + '   wall=' + dt + 'ms');
    runs.push({name: ctrl.name, result: r});
  }

  header('MDP-MPC SENSITIVITY: forecast horizon × tracking weight');
  console.log('  Demonstrates the energy/comfort frontier the MDP discovers when given more lookahead');
  console.log('  or different relative penalties. With weak tracking (w=0.05) the controller saves');
  console.log('  energy by riding closer to the band edges; with strong tracking (w=1.0) it stays');
  console.log('  near the centre at slightly higher cost.');
  console.log();
  console.log('  ' + 'horizon_h'.padEnd(11) + 'trackWeight'.padEnd(13) + '  ' +
              'energy_kWh'.padStart(11) + '  ' + 'comfort%'.padStart(9) + '  ' + 'violation'.padStart(11) + '  ' + 'cost_$'.padStart(8));
  for (const H of [1, 2, 4, 6]) {
    for (const w of [0.05, 0.5, 1.0, 2.0]) {
      const r = runTempControl({...cfg, controller: {
        kind: 'mdp-mpc', horizon_h: H, nLevels: 6,
        comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: w,
      }});
      console.log('  ' + H.toString().padEnd(11) + w.toFixed(2).padEnd(13) + '  ' +
        r.energy_kWh.toFixed(2).padStart(11) + '  ' + (100 * r.comfort_pct).toFixed(1).padStart(9) +
        '  ' + r.violation_Fh.toFixed(3).padStart(11) + '  ' + r.cost_$.toFixed(2).padStart(8));
    }
  }

  header('STRESS TEST: tight ±1°F band, harder weather');
  const stress: SimConfig = {
    ...cfg,
    band: 1,
    comfort_penalty: 2.0,
    outdoor: {mean: 15, amp: 20, phase: 9, noiseStd: 2.5},
  };
  for (const ctrl of [
    {name: 'bang-bang',          spec: {kind: 'bang-bang'} as ControllerSpec},
    {name: 'PID',                 spec: {kind: 'pid', Kp: 5, Ki: 1, Kd: 1} as ControllerSpec},
    {name: 'Fuzzy-PI',            spec: {kind: 'fuzzy'} as ControllerSpec},
    {name: 'MDP-MPC (H=6h, w=1)', spec: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 2.0, cost_per_kWh: 0.15, trackWeight: 1.0} as ControllerSpec},
  ]) {
    const r = runTempControl({...stress, controller: ctrl.spec});
    console.log('  ' + fmt(r, ctrl.name));
  }

  header('SAVE TIME-SERIES TRACES TO out/temp-control/');
  const outDir = path.join(__dirname, '..', '..', 'out', 'temp-control');
  fs.mkdirSync(outDir, { recursive: true });
  for (const {name, result} of runs) {
    const safeName = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    const lines = ['tick,t_h,T_out,T_in,Q,energy_cum_kWh,error,in_band,violation_Fh'];
    for (const r of result.trace) {
      lines.push([r.tick, r.t_h.toFixed(4), r.T_out_true.toFixed(3), r.T_in_true.toFixed(3),
        r.Q.toFixed(3), r.energy_cum_kWh.toFixed(3), r.error.toFixed(3),
        r.in_band ? 1 : 0, r.violation_Fh.toFixed(4)].join(','));
    }
    const f = path.join(outDir, safeName + '.csv');
    fs.writeFileSync(f, lines.join('\n'));
    console.log('  ' + name.padEnd(22) + ' → ' + path.relative(process.cwd(), f));
  }
  console.log();
}

main();
