'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/temp-control-adapter.rs
//   (module des::general::adapters::temp_control_adapter)
// 1:1 file move. JSON adapter for the indoor temperature-control DES (bang-bang /
// PID / fuzzy / MDP-MPC controllers), with a lazily-imported animation.
//
// Declarations → Rust:
//   interface TempControlParams                -> struct (#[derive(Deserialize)];
//             house?/outdoor? -> Option<struct>)
//   const controllerSchema/tempControlSchema: ParamSchema -> serde + validator metadata
//   const adapter: DESModelRegistration<P,R>   -> struct + impl ModelAdapter trait;
//             registerModel(adapter) -> explicit registration
//   fn describeController                        -> match on the controller enum
//
// Conversion notes (file-specific):
//   - ControllerSpec is a discriminated union (bang-bang | pid | fuzzy | mdp-mpc),
//     encoded as a `oneOf` schema -> enum ControllerSpec matched on `kind`
//     (#[serde(tag="kind", rename_all="kebab-case")]); describeController is the match.
//   - GotChA: `animate` uses dynamic `await import(...)` of the temp-control scene +
//     FrameRecorder -> ordinary `use` imports in Rust (no lazy import).
//   - `run` copies params into a SimConfig and injects `seed: runtime.seed` ->
//     thread a seeded RNG through; `params.band ?? 2`, `dt_min ?? 1` -> unwrap_or.
//   - `path` import is unused-ish here; CSV is plain string rows.
// =============================================================================

import * as path from 'path';
import {
  runTempControl, RunResult, ControllerSpec, SimConfig,
} from '../temp-control';
import {
  DESModelRegistration, ParamSchema, DESRuntimeConfig, DESModelSpec,
} from '../des-spec';
import {registerModel} from '../des-registry';
import {csvRow, defaultFramesPath, writeCsvLines} from './adapter-utils';

// -----------------------------------------------------------------------------
// Parameter type and schema
// -----------------------------------------------------------------------------

interface TempControlParams {
  T_target: number;
  band: number;
  duration_h: number;
  dt_min: number;
  cost_per_kWh: number;
  comfort_penalty: number;
  controller: ControllerSpec;
  house?: {tau?: number; G?: number; Q_max?: number; T_init?: number};
  outdoor?: {mean: number; amp: number; phase: number; noiseStd: number};
  sensorNoiseStd?: number;
  forecastNoiseStd?: number;
  forecastHorizon_h?: number;
}

const controllerSchema: ParamSchema = {
  kind: 'oneOf',
  description: 'Controller type and its hyperparameters.',
  variants: [
    {tag: 'bang-bang', schema: {kind: 'object', fields: {kind: {kind: 'string', enum: ['bang-bang']}}, required: ['kind']}},
    {tag: 'pid', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['pid']},
      Kp: {kind: 'number', min: 0, description: 'Proportional gain (kW/°F)'},
      Ki: {kind: 'number', min: 0, description: 'Integral gain (kW/°F·h)'},
      Kd: {kind: 'number', min: 0, description: 'Derivative gain (kW·h/°F)'},
    }, required: ['kind', 'Kp', 'Ki', 'Kd']}},
    {tag: 'fuzzy', schema: {kind: 'object', fields: {kind: {kind: 'string', enum: ['fuzzy']}}, required: ['kind']}},
    {tag: 'mdp-mpc', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['mdp-mpc']},
      horizon_h: {kind: 'number', min: 0.1, description: 'Lookahead horizon in hours'},
      nLevels: {kind: 'number', integer: true, min: 2, max: 20, description: 'Number of discrete heater levels (2-20)'},
      comfort_penalty: {kind: 'number', min: 0},
      cost_per_kWh: {kind: 'number', min: 0},
      trackWeight: {kind: 'number', min: 0, default: 1.0, description: 'Soft tracking weight inside band'},
    }, required: ['kind', 'horizon_h', 'nLevels', 'comfort_penalty', 'cost_per_kWh']}},
  ],
};

const tempControlSchema: ParamSchema = {
  kind: 'object',
  description: 'Temperature-control simulation parameters.',
  fields: {
    T_target: {kind: 'number', description: 'Target indoor temperature (°F)'},
    band: {kind: 'number', min: 0, default: 2, description: '±band defining comfort interval (°F)'},
    duration_h: {kind: 'number', min: 0, description: 'Simulated duration (hours)'},
    dt_min: {kind: 'number', min: 0, default: 1, description: 'Tick length (minutes)'},
    cost_per_kWh: {kind: 'number', min: 0, default: 0.15, description: 'Energy price ($/kWh)'},
    comfort_penalty: {kind: 'number', min: 0, default: 0.5, description: 'Comfort violation penalty ($/(°F)²/h)'},
    controller: controllerSchema,
    house: {kind: 'object', fields: {
      tau: {kind: 'number', min: 0.01, default: 12, description: 'Thermal time constant (h)'},
      G:   {kind: 'number', min: 0, default: 1.0, description: 'Heater gain (°F per kW per hour)'},
      Q_max: {kind: 'number', min: 0, default: 5.0, description: 'Max heater power (kW)'},
      T_init: {kind: 'number', default: 70, description: 'Initial indoor temperature (°F)'},
    }, required: []},
    outdoor: {kind: 'object', fields: {
      mean: {kind: 'number', default: 25},
      amp: {kind: 'number', default: 15, min: 0},
      phase: {kind: 'number', default: 9},
      noiseStd: {kind: 'number', default: 1.5, min: 0},
    }, required: []},
    sensorNoiseStd: {kind: 'number', min: 0, default: 0.0},
    forecastNoiseStd: {kind: 'number', min: 0, default: 0.0},
    forecastHorizon_h: {kind: 'number', min: 0.1, default: 6.0},
  },
  required: ['T_target', 'duration_h', 'controller'],
};

// -----------------------------------------------------------------------------
// Adapter
// -----------------------------------------------------------------------------

const adapter: DESModelRegistration<TempControlParams, RunResult> = {
  id: 'temp-control',
  description: 'Indoor temperature-control DES with bang-bang / PID / Fuzzy / MDP-MPC controllers.',
  schema: tempControlSchema,

  run(params, runtime) {
    const cfg: SimConfig = {
      T_target: params.T_target,
      band: params.band,
      duration_h: params.duration_h,
      dt_min: params.dt_min,
      cost_per_kWh: params.cost_per_kWh,
      comfort_penalty: params.comfort_penalty,
      controller: params.controller,
      house: params.house,
      outdoor: params.outdoor,
      sensorNoiseStd: params.sensorNoiseStd,
      forecastNoiseStd: params.forecastNoiseStd,
      forecastHorizon_h: params.forecastHorizon_h,
      seed: runtime.seed,
    };
    return runTempControl(cfg);
  },

  summarize(result, params) {
    const minT = Math.min(...result.T_in);
    const maxT = Math.max(...result.T_in);
    const lines = [
      'TEMPERATURE-CONTROL RUN SUMMARY',
      '──────────────────────────────────',
      `  Controller:      ${describeController(params.controller)}`,
      `  Target:          ${params.T_target.toFixed(2)}°F ± ${(params.band ?? 2).toFixed(2)}°F`,
      `  Duration:        ${params.duration_h.toFixed(2)} h  (${result.trace.length} ticks of ${(params.dt_min ?? 1)} min)`,
      `  Indoor range:    [${minT.toFixed(2)}, ${maxT.toFixed(2)}] °F`,
      `  Energy used:     ${result.energy_kWh.toFixed(2)} kWh`,
      `  Comfort:         ${(100 * result.comfort_pct).toFixed(2)}% in band`,
      `  Violation:       ${result.violation_Fh.toFixed(3)} °F·h outside band`,
      `  Total cost:      $${result.cost_$.toFixed(2)}`,
    ];
    return lines.join('\n');
  },

  writeCsv(result, csvPath) {
    const lines = ['tick,t_h,T_out,T_in,Q,energy_cum_kWh,error,in_band,violation_Fh'];
    for (const r of result.trace) {
      lines.push(csvRow([
        r.tick, r.t_h.toFixed(4), r.T_out_true.toFixed(3), r.T_in_true.toFixed(3),
        r.Q.toFixed(3), r.energy_cum_kWh.toFixed(3), r.error.toFixed(3),
        r.in_band ? 1 : 0, r.violation_Fh.toFixed(4),
      ]));
    }
    writeCsvLines(csvPath, lines);
  },

  async animate(result, params, runtime) {
    // Lazy import: avoids loading the animation graph if no animation
    // is requested.
    const {STAGE_W, STAGE_H, buildTempControlAnimation} = await import('../../animation/scenes/temp-control-scene');
    const {FrameRecorder} = await import('../../animation/frame-recorder');
    const out = runtime.outputs ?? {};
    const htmlPath = out.html;
    if (!htmlPath) return;
    const framesPath = out.frames ?? defaultFramesPath(htmlPath);
    const ctlName = describeController(params.controller);
    const recordEvery = Math.max(1, Math.round((params.dt_min ?? 1) >= 1 ? 5 : 1));
    const {frames, charts} = buildTempControlAnimation(result, ctlName, recordEvery);
    const recorder = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H, fps: 12,
      title: `Temperature Control — ${ctlName}`,
      subtitle: `T_target = ${params.T_target}°F ± ${(params.band ?? 2)}°F  |  energy=${result.energy_kWh.toFixed(2)} kWh, comfort=${(100 * result.comfort_pct).toFixed(1)}%, cost=$${result.cost_$.toFixed(2)}`,
      background: '#f9fafb',
    });
    for (const f of frames) recorder.frame(f.t, f.tick, () => ({shapes: f.shapes, caption: f.caption}));
    recorder.setCharts(charts);
    await recorder.finish();
  },

  examples: [
    {
      name: 'PID winter day',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'temp-control',
        description: '24-hour winter day, PID controller',
        parameters: {
          T_target: 70, band: 2, duration_h: 24, dt_min: 1,
          cost_per_kWh: 0.15, comfort_penalty: 0.5,
          controller: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
          sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 6,
        },
        runtime: {seed: 42},
      },
    },
  ],
};

function describeController(c: ControllerSpec): string {
  switch (c.kind) {
    case 'bang-bang': return 'Bang-bang';
    case 'pid':       return `PID (Kp=${c.Kp}, Ki=${c.Ki}, Kd=${c.Kd})`;
    case 'fuzzy':     return 'Fuzzy-PI (Mamdani)';
    case 'mdp-mpc':   return `MDP-MPC (H=${c.horizon_h}h, levels=${c.nLevels}, w=${c.trackWeight ?? 1.0})`;
  }
}

registerModel(adapter);
