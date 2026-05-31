'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-wind-mppt.rs   (fn main)
// 1:1 file move. Runnable MPPT demo for a PMSG wind-energy system (optimal-
// torque vs PI speed-loop controller).
//
// Conversion notes (file-specific):
//   - class WindMpptDemo -> struct + impl; top-level run -> fn main.
//   - process.env.CONTROLLER -> std::env + enum.
//   - imports general/control-systems/wind-mppt -> use crate::des::general::
//     control_systems::wind_mppt.
// =============================================================================

// =============================================================================
// main-wind-mppt.ts — runnable demo of MPPT for a PMSG wind-energy conversion
// system. Wires the self-clocking turbine plant to an MPPT controller and a
// trajectory sink, runs the lightweight DES loop, and prints convergence to
// the optimal tip-speed ratio / power coefficient.
//
//   npm run wind-mppt              # optimal-torque MPPT (default)
//   CONTROLLER=pi npm run wind-mppt  # PI speed-loop MPPT
// =============================================================================

import {runIterativeDES} from './general/des-base/runner';
import {DESStation} from './general/des-base/station';
import {
  OptimalTorqueMpptController,
  SpeedPiMpptController,
  WindMpptChannels,
  WindMpptSinkStation,
  WindProfile,
  WindTurbineAerodynamics,
  WindTurbinePlantStation,
} from './general/control-systems/wind-mppt';

/** Builds, runs, and reports a single wind-MPPT simulation. */
class WindMpptDemo {
  private readonly aero = new WindTurbineAerodynamics({bladeRadius: 2.5, pitchDeg: 0});
  private readonly dt = 0.05;
  private readonly steps = 1200;

  run(controllerKind: 'optimal-torque' | 'pi'): void {
    const windProfile = new WindProfile([
      {fromTime: 0, speed: 8},
      {fromTime: 20, speed: 11},
      {fromTime: 40, speed: 9},
    ]);

    const plant = new WindTurbinePlantStation('turbine', {
      aero: this.aero,
      windProfile,
      inertia: 6,
      friction: 0.02,
      dt: this.dt,
      steps: this.steps,
      initialOmega: 2,
    });

    const controller: DESStation = controllerKind === 'pi'
      ? new SpeedPiMpptController('mppt-pi', this.aero, {kp: 8, ki: 4, dt: this.dt})
      : new OptimalTorqueMpptController('mppt-opt-torque', this.aero);

    const sink = new WindMpptSinkStation('sink');

    plant.pipe(controller, WindMpptChannels.STATE, WindMpptChannels.STATE);
    plant.pipe(sink, WindMpptChannels.STATE, WindMpptChannels.STATE);
    controller.pipe(plant, WindMpptChannels.TORQUE, WindMpptChannels.TORQUE);

    const summary = runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: this.steps + 5});

    this.report(controllerKind, plant, sink, summary.ticks);
  }

  private report(kind: string, plant: WindTurbinePlantStation, sink: WindMpptSinkStation, ticks: number): void {
    const lambdaStar = this.aero.optimalTipSpeedRatio();
    const cpMax = this.aero.maxPowerCoefficient();
    console.log('\n============================================================');
    console.log(` Wind MPPT — PMSG WECS   (controller: ${kind})`);
    console.log('============================================================');
    console.log(`  blade radius R          : ${this.aero.bladeRadius} m`);
    console.log(`  swept area A            : ${this.aero.sweptArea().toFixed(3)} m²`);
    console.log(`  optimal λ*              : ${lambdaStar.toFixed(4)}`);
    console.log(`  C_p,max                 : ${cpMax.toFixed(4)}`);
    console.log(`  K_opt (½ρπR⁵C_p/λ*³)    : ${this.aero.optimalTorqueGain().toExponential(4)}`);
    console.log(`  ticks run               : ${ticks}  (dt=${this.dt}s)`);
    console.log('  ----------------------------------------------------------');
    console.log('   step      V[m/s]   ω[rad/s]     λ       C_p     P[kW]');
    const n = sink.samples.length;
    const idxs = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1];
    for (const i of idxs) {
      const s = sink.samples[i];
      console.log(
        `   ${String(s.tick).padStart(5)}   ${s.windSpeed.toFixed(2).padStart(7)}   ${s.omega.toFixed(3).padStart(8)}` +
        `   ${s.lambda.toFixed(3).padStart(6)}   ${s.cp.toFixed(4).padStart(6)}   ${(s.mechPower / 1000).toFixed(3).padStart(6)}`,
      );
    }
    console.log('  ----------------------------------------------------------');
    const lambdaErr = Math.abs(sink.finalLambda() - lambdaStar);
    console.log(`  final λ                 : ${sink.finalLambda().toFixed(4)}   (|λ−λ*| = ${lambdaErr.toFixed(4)})`);
    console.log(`  final C_p / C_p,max     : ${(sink.finalCp() / cpMax * 100).toFixed(2)}%`);
    console.log(`  final captured power    : ${(sink.finalPower() / 1000).toFixed(3)} kW`);
    console.log('============================================================\n');
  }
}

const kind = (process.env.CONTROLLER ?? '').toLowerCase() === 'pi' ? 'pi' : 'optimal-torque';
new WindMpptDemo().run(kind);
