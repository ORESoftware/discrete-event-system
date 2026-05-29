'use strict';

// =============================================================================
// main-dc-motor-anim.ts — generate an HTML animation of the back-EMF DC-motor.
//
//   npm run dc-motor-anim              # closed-loop PI speed control
//   MODE=open npm run dc-motor-anim    # open-loop step voltage (back-EMF rise)
// =============================================================================

import * as path from 'path';
import {runIterativeDES} from './general/des-base/runner';
import {FrameRecorder} from './animation/frame-recorder';
import {DcMotorScene, MOTOR_STAGE_H, MOTOR_STAGE_W} from './animation/scenes/dc-motor-scene';
import {
  DcMotorChannels,
  DcMotorParams,
  DcMotorPlantStation,
  DcMotorSinkStation,
  LoadProfile,
  SpeedPiVoltageController,
} from './general/control-systems/dc-motor';

class DcMotorAnimator {
  private readonly params: DcMotorParams = {
    resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002,
  };
  private readonly dt = 0.005;

  async run(mode: 'open' | 'closed'): Promise<void> {
    if (mode === 'open') await this.runOpen();
    else await this.runClosed();
  }

  private async runOpen(): Promise<void> {
    const steps = 3000;
    const plant = new DcMotorPlantStation('motor', {params: this.params, dt: this.dt, steps});
    plant.setOpenLoopVoltage(12);
    const sink = new DcMotorSinkStation('sink');
    plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
    runIterativeDES([plant, sink], {shuffle: false, maxTicks: steps + 5});

    const scene = new DcMotorScene({samples: sink.samples, dt: this.dt, params: this.params, modeName: 'open loop (12 V step)'});
    await this.record(scene, 'open', 8, 'Back-EMF rises as ω climbs, throttling armature current');
  }

  private async runClosed(): Promise<void> {
    const steps = 6000;
    const load = new LoadProfile([{fromTime: 0, torque: 0}, {fromTime: 18, torque: 0.3}]);
    const plant = new DcMotorPlantStation('motor', {params: this.params, dt: this.dt, steps, load});
    const controller = new SpeedPiVoltageController('speed-pi', {
      kp: 1.5, ki: 1, dt: this.dt, maxVoltage: 48,
      reference: [{fromTime: 0, speed: 60}, {fromTime: 10, speed: 100}],
    });
    const sink = new DcMotorSinkStation('sink');
    plant.pipe(controller, DcMotorChannels.STATE, DcMotorChannels.STATE);
    plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
    controller.pipe(plant, DcMotorChannels.VOLTAGE, DcMotorChannels.VOLTAGE);
    runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: steps + 5});

    const reference = sink.samples.map(s => controller.referenceAt(s.time));
    const scene = new DcMotorScene({
      samples: sink.samples, dt: this.dt, params: this.params, reference,
      modeName: 'closed-loop PI speed control',
    });
    await this.record(scene, 'closed', 15, 'PI tracks 60 → 100 rad/s and rejects a 0.3 N·m load step');
  }

  private async record(scene: DcMotorScene, tag: string, stride: number, subtitle: string): Promise<void> {
    const out = path.join('out', 'dc-motor', `animation-${tag}.html`);
    const recorder = new FrameRecorder({
      framesPath: out.replace(/\.html$/, '.frames.jsonl'), htmlPath: out,
      width: MOTOR_STAGE_W, height: MOTOR_STAGE_H, fps: 30,
      title: `DC motor — ${tag === 'open' ? 'open loop' : 'closed-loop PI'}`,
      subtitle, background: '#0b1021',
    });
    for (let i = 0; i < scene.frameCount(); i += stride) {
      recorder.frame(scene.timeAt(i), i, () => scene.frameAt(i));
    }
    recorder.setCharts(scene.charts());
    await recorder.finish();
    console.log(`DC-motor animation (${tag}): ${path.resolve(out)}`);
  }
}

const mode = (process.env.MODE ?? '').toLowerCase() === 'open' ? 'open' : 'closed';
new DcMotorAnimator().run(mode).catch(e => { console.error(e); process.exit(1); });
