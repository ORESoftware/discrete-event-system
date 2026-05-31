// RUST MIGRATION: Prefer moving these unit checks into `src/des/general/control_systems/dc_motor.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate ODE/back-EMF checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and approximate-float helpers; keep integrator tolerances explicit.

'use strict';

// =============================================================================
// Unit tests for general/control-systems/dc-motor.ts (back-EMF ODE system).
// Run with: ./node_modules/.bin/ts-node src/des/test/dc-motor-test.ts
// =============================================================================

import {runIterativeDES} from '../general/des-base/runner';
import {
  DcMotorChannels,
  DcMotorDynamics,
  DcMotorParams,
  DcMotorPlantStation,
  DcMotorSinkStation,
  LoadProfile,
  SpeedPiVoltageController,
} from '../general/control-systems/dc-motor';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

const params: DcMotorParams = {
  resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002,
};
/** Analytic open-loop steady-state speed for a constant armature voltage V. */
function omegaSteadyState(V: number): number {
  return V / ((params.resistance * params.friction) / params.torqueConstant + params.backEmfConstant);
}

// -----------------------------------------------------------------------------
console.log('\n[1] Back-EMF & dynamics algebra');
// -----------------------------------------------------------------------------
{
  const dyn = new DcMotorDynamics(params);
  check('1.1 back-EMF E = K_e·ω', close(dyn.backEmf(50), 0.1 * 50, 1e-12));
  check('1.2 electromagnetic torque T = K_t·i', close(dyn.electromagneticTorque(3), 0.1 * 3, 1e-12));
  // Electrical equation: L·di/dt = V − R·i − E.
  dyn.setInputs(12, 0);
  const d = dyn.derivative(0, [2, 30]);
  const diExpected = (12 - params.resistance * 2 - params.backEmfConstant * 30) / params.inductance;
  const dwExpected = (params.torqueConstant * 2 - params.friction * 30 - 0) / params.inertia;
  check('1.3 di/dt = (V−Ri−K_eω)/L', close(d[0], diExpected, 1e-12));
  check('1.4 dω/dt = (K_t·i−Bω−T_L)/J', close(d[1], dwExpected, 1e-12));
  // State-space matrices match the ODE.
  const ss = dyn.stateSpace();
  check('1.5 A matrix matches dynamics',
    close(ss.A[0][0], -params.resistance / params.inductance) &&
    close(ss.A[0][1], -params.backEmfConstant / params.inductance) &&
    close(ss.A[1][0], params.torqueConstant / params.inertia) &&
    close(ss.A[1][1], -params.friction / params.inertia));
  check('1.6 B matrix = [1/L; 0]', close(ss.B[0][0], 1 / params.inductance) && close(ss.B[1][0], 0));
  check('1.7 C measures speed [0 1]', ss.C[0][0] === 0 && ss.C[0][1] === 1);
}

// -----------------------------------------------------------------------------
console.log('\n[2] LoadProfile schedule');
// -----------------------------------------------------------------------------
{
  const lp = new LoadProfile([{fromTime: 0, torque: 0}, {fromTime: 5, torque: 0.4}]);
  check('2.1 before step → 0', close(lp.torqueAt(2), 0));
  check('2.2 at/after step → 0.4', close(lp.torqueAt(5), 0.4) && close(lp.torqueAt(100), 0.4));
}

// -----------------------------------------------------------------------------
console.log('\n[3] Open-loop step — back-EMF rises, current limited, ω → analytic');
// -----------------------------------------------------------------------------
{
  const dt = 0.005, steps = 4000;
  const plant = new DcMotorPlantStation('motor', {params, dt, steps});
  plant.setOpenLoopVoltage(12);
  const sink = new DcMotorSinkStation('sink');
  plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
  runIterativeDES([plant, sink], {shuffle: false, maxTicks: steps + 5});

  const s = sink.samples;
  check('3.1 one sample per tick', s.length === steps, `got ${s.length}`);
  check('3.2 starts near rest after one step', Math.abs(s[0].omega) < 0.05 && Math.abs(s[0].current) < 0.5,
        `ω₀=${s[0].omega.toExponential(2)}, i₀=${s[0].current.toFixed(4)}`);
  // Back-EMF is monotonically non-decreasing while spinning up.
  let monoEmf = true;
  for (let k = 1; k < s.length; k++) if (s[k].backEmf < s[k - 1].backEmf - 1e-6) { monoEmf = false; break; }
  check('3.3 back-EMF rises monotonically', monoEmf);
  check('3.4 ω → analytic steady state', close(sink.finalOmega(), omegaSteadyState(12), 1e-3),
        `ω=${sink.finalOmega().toFixed(3)} vs ${omegaSteadyState(12).toFixed(3)}`);
  check('3.5 final back-EMF = K_e·ω_ss', close(sink.finalBackEmf(), 0.1 * omegaSteadyState(12), 1e-3),
        `E=${sink.finalBackEmf().toFixed(4)}`);
  // Back-EMF opposes the supply: steady-state E < applied 12 V.
  check('3.6 back-EMF < applied voltage', sink.finalBackEmf() < 12);
  // Steady-state armature current i_ss = (V − E)/R.
  const iSs = (12 - sink.finalBackEmf()) / params.resistance;
  check('3.7 i_ss = (V−E)/R', close(sink.finalState()!.current, iSs, 1e-2), `i=${sink.finalState()!.current.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[4] Closed-loop PI — tracks reference, rejects load disturbance');
// -----------------------------------------------------------------------------
{
  const dt = 0.005, steps = 6000;
  const load = new LoadProfile([{fromTime: 0, torque: 0}, {fromTime: 18, torque: 0.3}]);
  const plant = new DcMotorPlantStation('motor', {params, dt, steps, load});
  const controller = new SpeedPiVoltageController('speed-pi', {
    kp: 1.5, ki: 1, dt, maxVoltage: 48,
    reference: [{fromTime: 0, speed: 60}, {fromTime: 10, speed: 100}],
  });
  const sink = new DcMotorSinkStation('sink');
  plant.pipe(controller, DcMotorChannels.STATE, DcMotorChannels.STATE);
  plant.pipe(sink, DcMotorChannels.STATE, DcMotorChannels.STATE);
  controller.pipe(plant, DcMotorChannels.VOLTAGE, DcMotorChannels.VOLTAGE);
  runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: steps + 5});

  const s = sink.samples;
  const at = (t: number) => s[Math.min(s.length - 1, Math.round(t / dt) - 1)];
  check('4.1 tracks first reference (60) before the step', Math.abs(at(9.5).omega - 60) < 0.5,
        `ω@9.5s=${at(9.5).omega.toFixed(3)}`);
  check('4.2 tracks second reference (100) before load', Math.abs(at(17.5).omega - 100) < 0.5,
        `ω@17.5s=${at(17.5).omega.toFixed(3)}`);
  check('4.3 rejects load step, final ω → 100', Math.abs(sink.finalOmega() - 100) < 0.1,
        `ω=${sink.finalOmega().toFixed(4)}`);
  // Power balance at steady state under load: K_t·i = B·ω + T_L.
  const f = sink.finalState()!;
  check('4.4 steady-state torque balance K_t·i = B·ω + T_L',
        close(params.torqueConstant * f.current, params.friction * f.omega + f.loadTorque, 1e-2),
        `K_t·i=${(params.torqueConstant * f.current).toFixed(4)}`);
  // Steady-state voltage balance: V = R·i + K_e·ω.
  check('4.5 steady-state voltage balance V = R·i + K_e·ω',
        close(f.voltage, params.resistance * f.current + f.backEmf, 1e-2),
        `V=${f.voltage.toFixed(3)}`);
  check('4.6 overshoot on first step is moderate (<20%)',
        (() => { let p = 0; for (const x of s) if (x.time <= 10) p = Math.max(p, x.omega); return p < 72; })());
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
