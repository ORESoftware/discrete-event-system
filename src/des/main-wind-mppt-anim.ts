// RUST MIGRATION: target src/bin/main_wind_mppt_anim.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-wind-mppt-anim.rs   (fn main)
// 1:1 file move. Generates an HTML animation of the wind-MPPT DES.
//
// Conversion notes (file-specific):
//   - class WindMpptAnimator -> struct + impl; async run() -> fn main.
//   - process.env.CONTROLLER -> std::env + enum.
//   - FrameRecorder + WindMpptScene + general/control-systems/wind-mppt ->
//     use crate::des::...
// =============================================================================

// =============================================================================
// main-wind-mppt-anim.ts — generate an HTML animation of the wind-MPPT DES.
//
//   npm run wind-mppt-anim                 # optimal-torque controller
//   CONTROLLER=pi npm run wind-mppt-anim   # PI speed-loop controller
// =============================================================================

import * as path from 'path';
import {runIterativeDES} from './general/des-base/runner';
import {DESStation} from './general/des-base/station';
import {FrameRecorder} from './animation/frame-recorder';
import {WIND_STAGE_H, WIND_STAGE_W, WindMpptScene} from './animation/scenes/wind-mppt-scene';
import {
  OptimalTorqueMpptController,
  SpeedPiMpptController,
  WindMpptChannels,
  WindMpptSinkStation,
  WindProfile,
  WindTurbineAerodynamics,
  WindTurbinePlantStation,
} from './general/control-systems/wind-mppt';

class WindMpptAnimator {
  private readonly aero = new WindTurbineAerodynamics({bladeRadius: 2.5, pitchDeg: 0});
  private readonly dt = 0.05;
  private readonly steps = 1200;

  async run(kind: 'optimal-torque' | 'pi'): Promise<void> {
    const windProfile = new WindProfile([
      {fromTime: 0, speed: 8},
      {fromTime: 20, speed: 11},
      {fromTime: 40, speed: 9},
    ]);
    const plant = new WindTurbinePlantStation('turbine', {
      aero: this.aero, windProfile, inertia: 6, friction: 0.02, dt: this.dt, steps: this.steps, initialOmega: 2,
    });
    const controller: DESStation = kind === 'pi'
      ? new SpeedPiMpptController('mppt-pi', this.aero, {kp: 8, ki: 4, dt: this.dt})
      : new OptimalTorqueMpptController('mppt-opt-torque', this.aero);
    const sink = new WindMpptSinkStation('sink');
    plant.pipe(controller, WindMpptChannels.STATE, WindMpptChannels.STATE);
    plant.pipe(sink, WindMpptChannels.STATE, WindMpptChannels.STATE);
    controller.pipe(plant, WindMpptChannels.TORQUE, WindMpptChannels.TORQUE);
    runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: this.steps + 5});

    const controllerName = kind === 'pi' ? 'PI speed loop' : 'optimal torque';
    const scene = new WindMpptScene({
      samples: sink.samples, dt: this.dt,
      lambdaStar: this.aero.optimalTipSpeedRatio(),
      cpMax: this.aero.maxPowerCoefficient(),
      kOpt: this.aero.optimalTorqueGain(),
      controllerName,
    });

    const out = path.join('out', 'wind-mppt', `animation-${kind}.html`);
    const recorder = new FrameRecorder({
      framesPath: out.replace(/\.html$/, '.frames.jsonl'), htmlPath: out,
      width: WIND_STAGE_W, height: WIND_STAGE_H, fps: 30,
      title: `Wind MPPT — ${controllerName}`,
      subtitle: `PMSG WECS · λ* = ${this.aero.optimalTipSpeedRatio().toFixed(2)}, C_p,max = ${this.aero.maxPowerCoefficient().toFixed(3)}`,
      background: '#0b1021',
    });
    const stride = 3;   // 1200 samples → ~400 frames @ 30 fps
    for (let i = 0; i < scene.frameCount(); i += stride) {
      recorder.frame(scene.timeAt(i), i, () => scene.frameAt(i));
    }
    recorder.setCharts(scene.charts());
    await recorder.finish();
    console.log(`Wind-MPPT animation (${controllerName}): ${path.resolve(out)}`);
  }
}

const kind = (process.env.CONTROLLER ?? '').toLowerCase() === 'pi' ? 'pi' : 'optimal-torque';
new WindMpptAnimator().run(kind).catch(e => { console.error(e); process.exit(1); });
