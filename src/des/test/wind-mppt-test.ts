'use strict';

// =============================================================================
// Unit tests for general/control-systems/wind-mppt.ts and the shared
// numerical solvers. Run with:
//   ./node_modules/.bin/ts-node src/des/test/wind-mppt-test.ts
// =============================================================================

import {runIterativeDES} from '../general/des-base/runner';
import {DESStation} from '../general/des-base/station';
import {ForwardEulerIntegrator, OdeSystem, RungeKutta4Integrator} from '../general/control-systems/numerical-solvers';
import {
  OptimalTorqueMpptController,
  SpeedPiMpptController,
  WindMpptChannels,
  WindMpptSinkStation,
  WindProfile,
  WindTurbineAerodynamics,
  WindTurbinePlantStation,
} from '../general/control-systems/wind-mppt';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

/** dx/dt = -k·x  (analytic: x(t) = x0·e^{-k t}). */
class ExponentialDecay implements OdeSystem {
  constructor(private readonly k: number) {}
  dimension(): number { return 1; }
  derivative(_t: number, state: readonly number[]): number[] { return [-this.k * state[0]]; }
}

// -----------------------------------------------------------------------------
console.log('\n[1] Numerical solvers — RK4 vs forward Euler accuracy');
// -----------------------------------------------------------------------------
{
  const sys = new ExponentialDecay(1);
  const exact = Math.exp(-1);
  const rk4 = new RungeKutta4Integrator().integrate(sys, 0, [1], 0.1, 10);
  const euler = new ForwardEulerIntegrator().integrate(sys, 0, [1], 0.1, 10);
  const rk4Err = Math.abs(rk4.states[rk4.states.length - 1][0] - exact);
  const eulerErr = Math.abs(euler.states[euler.states.length - 1][0] - exact);
  check('1.1 RK4 reaches t=1 in 10 steps', rk4.times.length === 11 && close(rk4.times[10], 1, 1e-12));
  check('1.2 RK4 hits e^{-1} tightly', rk4Err < 1e-6, `err=${rk4Err.toExponential(2)}`);
  check('1.3 RK4 far more accurate than Euler', rk4Err < eulerErr / 100,
        `rk4=${rk4Err.toExponential(2)} euler=${eulerErr.toExponential(2)}`);
  check('1.4 integrate rejects dt<=0', (() => { try { new RungeKutta4Integrator().integrate(sys, 0, [1], 0, 1); return false; } catch { return true; } })());
}

// -----------------------------------------------------------------------------
console.log('\n[2] Aerodynamics — C_p model & optimal operating point');
// -----------------------------------------------------------------------------
{
  const aero = new WindTurbineAerodynamics({bladeRadius: 2.5, pitchDeg: 0});
  check('2.1 swept area = πR²', close(aero.sweptArea(), Math.PI * 6.25, 1e-9));
  check('2.2 tip-speed ratio λ = ωR/V', close(aero.tipSpeedRatio(16, 5), 8, 1e-9));
  const lambdaStar = aero.optimalTipSpeedRatio();
  check('2.3 optimal λ* near 8.1', lambdaStar > 7.5 && lambdaStar < 8.5, `λ*=${lambdaStar.toFixed(3)}`);
  const cpMax = aero.maxPowerCoefficient();
  check('2.4 C_p,max near 0.48', cpMax > 0.45 && cpMax < 0.50, `C_p,max=${cpMax.toFixed(4)}`);
  // C_p(λ*) is the maximum: no other λ beats it.
  let cpMaxIsMax = true;
  for (let l = 0.5; l <= 18; l += 0.5) if (aero.powerCoefficient(l) > cpMax + 1e-9) cpMaxIsMax = false;
  check('2.5 C_p(λ*) is the global max on the grid', cpMaxIsMax);
  check('2.6 K_opt > 0', aero.optimalTorqueGain() > 0);
  // Captured power equals ½ρA·C_p·V³.
  const V = 9, omega = 20;
  const lambda = aero.tipSpeedRatio(omega, V);
  const expected = 0.5 * 1.225 * aero.sweptArea() * aero.powerCoefficient(lambda) * V ** 3;
  check('2.7 P_mech = ½ρA·C_p·V³', close(aero.mechanicalPower(V, omega), expected, 1e-9));
  // Aero torque = P/ω for ω well away from 0.
  check('2.8 T_aero = P_mech/ω', close(aero.aeroTorque(V, omega), expected / omega, 1e-9));
}

// -----------------------------------------------------------------------------
console.log('\n[3] WindProfile — piecewise-constant schedule');
// -----------------------------------------------------------------------------
{
  const wp = new WindProfile([{fromTime: 0, speed: 8}, {fromTime: 20, speed: 11}, {fromTime: 40, speed: 9}]);
  check('3.1 t<20 → 8 m/s', close(wp.speedAt(5), 8));
  check('3.2 t=20 → 11 m/s', close(wp.speedAt(20), 11));
  check('3.3 t≥40 → 9 m/s', close(wp.speedAt(100), 9));
}

// -----------------------------------------------------------------------------
console.log('\n[4] Optimal-torque MPPT — converges to the λ* ridge');
// -----------------------------------------------------------------------------
{
  const aero = new WindTurbineAerodynamics({bladeRadius: 2.5, pitchDeg: 0});
  const dt = 0.05, steps = 800;
  const plant = new WindTurbinePlantStation('turbine', {
    aero, windProfile: new WindProfile([{fromTime: 0, speed: 9}]),
    inertia: 6, friction: 0.02, dt, steps, initialOmega: 2,
  });
  const controller = new OptimalTorqueMpptController('opt-torque', aero);
  const sink = new WindMpptSinkStation('sink');
  plant.pipe(controller, WindMpptChannels.STATE, WindMpptChannels.STATE);
  plant.pipe(sink, WindMpptChannels.STATE, WindMpptChannels.STATE);
  controller.pipe(plant, WindMpptChannels.TORQUE, WindMpptChannels.TORQUE);
  runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: steps + 5});

  check('4.1 emits one sample per tick', sink.samples.length === steps, `got ${sink.samples.length}`);
  check('4.2 λ converges near λ* (no wind sensor)', Math.abs(sink.finalLambda() - aero.optimalTipSpeedRatio()) < 0.05,
        `λ=${sink.finalLambda().toFixed(4)}`);
  check('4.3 final C_p ≥ 99.5% of C_p,max', sink.finalCp() >= 0.995 * aero.maxPowerCoefficient(),
        `C_p=${sink.finalCp().toFixed(4)}`);
  check('4.4 T_gen = K_opt·ω² law', close(controller.getOptimalTorqueGain(), aero.optimalTorqueGain(), 1e-12));
  check('4.5 power strictly positive at steady state', sink.finalPower() > 0);
}

// -----------------------------------------------------------------------------
console.log('\n[5] PI speed-loop MPPT — converges with integral action');
// -----------------------------------------------------------------------------
{
  const aero = new WindTurbineAerodynamics({bladeRadius: 2.5, pitchDeg: 0});
  const dt = 0.05, steps = 1000;
  const plant = new WindTurbinePlantStation('turbine', {
    aero, windProfile: new WindProfile([{fromTime: 0, speed: 10}]),
    inertia: 6, friction: 0.02, dt, steps, initialOmega: 2,
  });
  const controller = new SpeedPiMpptController('pi', aero, {kp: 8, ki: 4, dt});
  const sink = new WindMpptSinkStation('sink');
  plant.pipe(controller, WindMpptChannels.STATE, WindMpptChannels.STATE);
  plant.pipe(sink, WindMpptChannels.STATE, WindMpptChannels.STATE);
  controller.pipe(plant, WindMpptChannels.TORQUE, WindMpptChannels.TORQUE);
  runIterativeDES([plant, controller, sink], {shuffle: false, maxTicks: steps + 5});

  check('5.1 PI drives λ → λ* (steady-state)', Math.abs(sink.finalLambda() - aero.optimalTipSpeedRatio()) < 0.02,
        `λ=${sink.finalLambda().toFixed(4)}`);
  check('5.2 final C_p ≥ 99.9% of C_p,max', sink.finalCp() >= 0.999 * aero.maxPowerCoefficient(),
        `C_p=${sink.finalCp().toFixed(4)}`);
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
