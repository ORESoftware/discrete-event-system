'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-dc-motor.rs   (fn main)
// 1:1 file move. Runnable demo of the back-EMF DC-motor ODE system (open-loop
// step vs closed-loop PI speed control).
//
// Conversion notes (file-specific):
//   - class DcMotorDemo -> struct + impl; process.env.MODE -> std::env + enum.
//   - imports general/control-systems/dc-motor -> use crate::des::general::
//     control_systems::dc_motor.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-dc-motor.ts — runnable demo of the back-EMF DC-motor ODE system.
//
//   npm run dc-motor             # closed-loop PI speed control (default)
//   MODE=open npm run dc-motor   # open-loop step voltage (back-EMF rise)
//
// The open-loop run shows the back-EMF E = K_e·ω climbing as the rotor spins
// up, throttling the armature current. The closed-loop run shows the PI
// controller tracking a stepped speed reference despite a load-torque step.
// =============================================================================

import {runIterativeDES} from './general/des-base/runner';
import {
  DcMotorChannels,
  DcMotorParams,
  DcMotorPlantStation,
  DcMotorSinkStation,
  LoadProfile,
  MotorStateToken,
  SpeedPiVoltageController,
} from './general/control-systems/dc-motor';

class DcMotorDemo {
  private readonly params: DcMotorParams = {
    resistance: 2,
    inductance: 0.5,
    backEmfConstant: 0.1,
    torqueConstant: 0.1,
    inertia: 0.02,
    friction: 0.002,
  };
  private readonly dt = 0.005;
  private readonly steps = 3000;

  run(mode: 'open' | 'closed'): void {
    if (mode === 'open') this.runOpenLoop();
    else this.runClosedLoop();
  }

  private runOpenLoop(): void {
    const plant = new DcMotorPlantStation('motor', {params: this.params, dt: this.dt, steps: this.steps});
    plant.setOpenLoopVoltage(12);
    const sink = new DcMotorSinkStation('sink');
    plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
    runIterativeDES([plant, sink], {shuffle: false, maxTicks: this.steps + 5});

    const wssDenominator = (this.params.resistance * this.params.friction) / this.params.torqueConstant + this.params.backEmfConstant;
    const omegaSs = 12 / wssDenominator;
    console.log('\n============================================================');
    console.log(' DC motor — OPEN LOOP (12 V step), back-EMF rise');
    console.log('============================================================');
    this.printParams(this.steps);
    console.log(`  analytic ω_ss           : ${omegaSs.toFixed(3)} rad/s`);
    console.log(`  analytic back-EMF_ss    : ${(this.params.backEmfConstant * omegaSs).toFixed(3)} V`);
    this.printTable(sink.samples);
    const f = sink.finalState() as MotorStateToken;
    console.log(`  final ω                 : ${f.omega.toFixed(3)} rad/s`);
    console.log(`  final back-EMF          : ${f.backEmf.toFixed(3)} V`);
    console.log(`  final current           : ${f.current.toFixed(4)} A`);
    console.log('============================================================\n');
  }

  private runClosedLoop(): void {
    const closedLoopSteps = 6000;
    const load = new LoadProfile([{fromTime: 0, torque: 0}, {fromTime: 18, torque: 0.3}]);
    const plant = new DcMotorPlantStation('motor', {params: this.params, dt: this.dt, steps: closedLoopSteps, load});
    const controller = new SpeedPiVoltageController('speed-pi', {
      kp: 1.5, ki: 1, dt: this.dt, maxVoltage: 48,
      reference: [{fromTime: 0, speed: 60}, {fromTime: 10, speed: 100}],
    });
    const sink = new DcMotorSinkStation('sink');
    plant.pipe(controller, DcMotorChannels.STATE, DcMotorChannels.STATE);
    plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
    controller.pipe(plant, DcMotorChannels.VOLTAGE, DcMotorChannels.VOLTAGE);
    runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: closedLoopSteps + 5});

    console.log('\n============================================================');
    console.log(' DC motor — CLOSED LOOP PI speed control');
    console.log('============================================================');
    this.printParams(closedLoopSteps);
    console.log('  reference: 60 rad/s → 100 rad/s @ t=10s;  load step 0.3 N·m @ t=18s');
    this.printTable(sink.samples);
    const f = sink.finalState() as MotorStateToken;
    console.log(`  final ω (ref 100)       : ${f.omega.toFixed(3)} rad/s`);
    console.log(`  final tracking error    : ${(100 - f.omega).toFixed(4)} rad/s`);
    console.log(`  final back-EMF          : ${f.backEmf.toFixed(3)} V`);
    console.log(`  final armature voltage  : ${f.voltage.toFixed(3)} V`);
    console.log('============================================================\n');
  }

  private printParams(steps: number): void {
    const p = this.params;
    console.log(`  R=${p.resistance}Ω  L=${p.inductance}H  K_e=${p.backEmfConstant}  K_t=${p.torqueConstant}  J=${p.inertia}  B=${p.friction}`);
    console.log(`  dt=${this.dt}s  steps=${steps}`);
  }

  private printTable(samples: readonly MotorStateToken[]): void {
    console.log('  ----------------------------------------------------------');
    console.log('    t[s]     V[V]    i[A]    ω[rad/s]   E=K_eω[V]   T_L[N·m]');
    const n = samples.length;
    for (const i of [0, Math.floor(n * 0.1), Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1]) {
      const s = samples[i];
      console.log(
        `   ${s.time.toFixed(3).padStart(6)}  ${s.voltage.toFixed(2).padStart(6)}  ${s.current.toFixed(4).padStart(7)}` +
        `  ${s.omega.toFixed(3).padStart(8)}  ${s.backEmf.toFixed(3).padStart(8)}  ${s.loadTorque.toFixed(3).padStart(8)}`,
      );
    }
    console.log('  ----------------------------------------------------------');
  }
}

const mode = (process.env.MODE ?? '').toLowerCase() === 'open' ? 'open' : 'closed';
new DcMotorDemo().run(mode);
