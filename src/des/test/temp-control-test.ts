'use strict';

// =============================================================================
// Unit tests for general/temp-control.ts.
// Run with: node dist/des/test/temp-control-test.js
// =============================================================================

import {
  houseStep, trueOutdoorTemp, mulberry32,
  fuzzyDeltaController, mdpMPCController, controllerStep, runTempControl,
  DEFAULT_HOUSE, DEFAULT_OUTDOOR,
  ControllerSpec, ControllerState,
} from '../general/temp-control';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

// -----------------------------------------------------------------------------
console.log('\n[1] House physics — forward-Euler step');
// -----------------------------------------------------------------------------
{
  // 1.1 Identity: dt = 0 → T unchanged.
  check('1.1 dt=0 → T unchanged', close(houseStep(70, 30, 5, 0, DEFAULT_HOUSE), 70));
  // 1.2 No heater + isothermal: T unchanged.
  check('1.2 T_out = T_in, Q=0 → T unchanged', close(houseStep(60, 60, 0, 1, DEFAULT_HOUSE), 60));
  // 1.3 Linearity in Q: T(Q1+Q2)−T(Q1) = T(Q2)−T(0) for fixed (T_in, T_out, Δt).
  const a = houseStep(70, 30, 0, 0.5, DEFAULT_HOUSE);
  const b = houseStep(70, 30, 3, 0.5, DEFAULT_HOUSE);
  const c = houseStep(70, 30, 5, 0.5, DEFAULT_HOUSE);
  check('1.3 linearity in Q', close((b - a), (3 / 5) * (c - a), 1e-9));
  // 1.4 Steady-state Q matches closed-form.
  const Q_ss = (70 - 30) / (DEFAULT_HOUSE.tau * DEFAULT_HOUSE.G);
  check('1.4 steady-state Q gives no change', close(houseStep(70, 30, Q_ss, 1, DEFAULT_HOUSE), 70));
  // 1.5 Heater-only (insulated wall) raises T by Q·G·Δt.
  const ins = {...DEFAULT_HOUSE, tau: 1e9};
  check('1.5 insulated, Q=5, dt=1 → ΔT=5°F', close(houseStep(70, 30, 5, 1, ins) - 70, 5, 1e-3));
}

// -----------------------------------------------------------------------------
console.log('\n[2] Outdoor temperature pattern');
// -----------------------------------------------------------------------------
{
  // 2.1 No noise, mean = mean.
  const p = {mean: 30, amp: 0, phase: 0, noiseStd: 0};
  for (const t of [0, 6, 12, 18, 24]) {
    check(`2.1 amp=0: T_out(t=${t}) = mean`, close(trueOutdoorTemp(t, p), 30));
  }
  // 2.2 Sinusoidal: peaks at 1 amp above mean.
  const q = {mean: 25, amp: 15, phase: 9, noiseStd: 0};
  // Peak when sin(2π(t−phase)/24) = 1 → t − 9 = 6 → t = 15.
  check('2.2 peak at t=15 hits mean+amp', close(trueOutdoorTemp(15, q), 40, 1e-9));
  check('2.2 trough at t=3 hits mean−amp', close(trueOutdoorTemp(3, q), 10, 1e-9));
  // 2.3 24-hour periodicity.
  check('2.3 24h periodic', close(trueOutdoorTemp(7, q), trueOutdoorTemp(31, q), 1e-9));
}

// -----------------------------------------------------------------------------
console.log('\n[3] PRNG reproducibility');
// -----------------------------------------------------------------------------
{
  const r1 = mulberry32(123); const r2 = mulberry32(123);
  let same = true;
  for (let i = 0; i < 100; i++) if (!close(r1(), r2())) { same = false; break; }
  check('3.1 mulberry32 deterministic', same);
}

// -----------------------------------------------------------------------------
console.log('\n[4] Fuzzy-PI controller — Δ-Q rule output');
// -----------------------------------------------------------------------------
{
  // 4.1 Symmetry: dq(−e, −de) = −dq(e, de).
  const e = 3, de = 2;
  const a = fuzzyDeltaController(e, de);
  const b = fuzzyDeltaController(-e, -de);
  check('4.1 odd symmetry', close(a, -b, 1e-9), `dq(${e},${de})=${a.toFixed(3)}, dq(−,−)=${b.toFixed(3)}`);
  // 4.2 Saturation at large positive error/rate ≈ +1.
  check('4.2 saturated positive Δ-Q', fuzzyDeltaController(20, 20) > 0.99);
  check('4.3 saturated negative Δ-Q', fuzzyDeltaController(-20, -20) < -0.99);
  // 4.4 Output is bounded by [-1, 1].
  let outOfRange = false;
  for (const e of [-10, -5, -1, 0, 1, 5, 10]) {
    for (const de of [-10, -5, -1, 0, 1, 5, 10]) {
      const v = fuzzyDeltaController(e, de);
      if (v < -1 - 1e-12 || v > 1 + 1e-12) outOfRange = true;
    }
  }
  check('4.4 Δ-Q ∈ [−1, +1] over grid', !outOfRange);
}

// -----------------------------------------------------------------------------
console.log('\n[5] PID controller — anti-windup + steady state');
// -----------------------------------------------------------------------------
{
  // 5.1 Constant outdoor → near-zero steady-state error after settling.
  // Need ≥ 5τ = 60 hours of settle time for the integral to converge.
  const r = runTempControl({
    T_target: 70, band: 2, duration_h: 80, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0, forecastNoiseStd: 0, forecastHorizon_h: 1, seed: 1,
    outdoor: {mean: 25, amp: 0, phase: 0, noiseStd: 0},
    controller: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
  });
  const last = r.T_in.slice(-60);
  const mean = last.reduce((a, b) => a + b, 0) / last.length;
  check('5.1 PID steady-state |err| < 0.1°F', Math.abs(mean - 70) < 0.1, `mean = ${mean.toFixed(3)}°F`);
  // 5.2 PID Q output respects [0, Q_max] bounds.
  const allInRange = r.Q.every(q => q >= 0 && q <= DEFAULT_HOUSE.Q_max + 1e-9);
  check('5.2 Q ∈ [0, Q_max]', allInRange);
}

// -----------------------------------------------------------------------------
console.log('\n[6] MDP-MPC — basic correctness');
// -----------------------------------------------------------------------------
{
  const fc = new Array(360).fill(20);
  // 6.1 Cold room → strong heating.
  const Q_cold = mdpMPCController(60, fc, 6, 6, 70, 1/60, 5, DEFAULT_HOUSE, 0.5, 0.15, 1.0);
  check('6.1 cold room → max heating', Q_cold >= 4, `Q = ${Q_cold}`);
  // 6.2 Hot room → no heating.
  const Q_hot = mdpMPCController(75, fc, 6, 6, 70, 1/60, 5, DEFAULT_HOUSE, 0.5, 0.15, 1.0);
  check('6.2 hot room → no heating', Q_hot === 0, `Q = ${Q_hot}`);
  // 6.3 At target → moderate heating to maintain.
  const Q_at = mdpMPCController(70, fc, 6, 6, 70, 1/60, 5, DEFAULT_HOUSE, 0.5, 0.15, 1.0);
  check('6.3 at target → some heating', Q_at > 0 && Q_at <= 5, `Q = ${Q_at}`);
  // 6.4 Action grid is a valid subset of [0, Q_max].
  const fcMild = new Array(60).fill(50);
  const Q5 = mdpMPCController(70, fcMild, 1, 5, 70, 1/60, 5, DEFAULT_HOUSE, 0.5, 0.15, 1.0);
  // 5 levels means values are {0, 1.25, 2.5, 3.75, 5}.
  const valid = [0, 1.25, 2.5, 3.75, 5].some(v => Math.abs(v - Q5) < 1e-9);
  check('6.4 Q is on the action grid', valid, `Q = ${Q5}`);
  // 6.5 Empty forecast (horizon=0) returns 0 (no decisions to make).
  // (We avoid this — interface requires ≥ 1 forecast point — but verify.)
}

// -----------------------------------------------------------------------------
console.log('\n[7] Bang-bang controller');
// -----------------------------------------------------------------------------
{
  const ctx = {T_target: 70, T_in_meas: 0, forecast: [30], dt_h: 1/60, Q_max: 5, house: DEFAULT_HOUSE};
  const st: ControllerState = {};
  // 7.1 Below target → MAX.
  ctx.T_in_meas = 65;
  check('7.1 T < target → Q = Q_max', controllerStep({kind: 'bang-bang'}, st, ctx) === 5);
  // 7.2 Above target → 0.
  ctx.T_in_meas = 75;
  check('7.2 T > target → Q = 0', controllerStep({kind: 'bang-bang'}, st, ctx) === 0);
  // 7.3 Exactly at target → 0 (e = 0, not > 0).
  ctx.T_in_meas = 70;
  check('7.3 T = target → Q = 0', controllerStep({kind: 'bang-bang'}, st, ctx) === 0);
}

// -----------------------------------------------------------------------------
console.log('\n[8] Full-run invariants');
// -----------------------------------------------------------------------------
{
  const cfg = {
    T_target: 70, band: 2, duration_h: 4, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0, forecastNoiseStd: 0, forecastHorizon_h: 1, seed: 1,
    controller: {kind: 'fuzzy'} as ControllerSpec,
  };
  const r = runTempControl(cfg);
  // 8.1 Monotonic energy.
  let monotone = true;
  for (let k = 1; k < r.energy.length; k++) {
    if (r.energy[k] < r.energy[k - 1] - 1e-9) { monotone = false; break; }
  }
  check('8.1 energy is monotonic', monotone);
  // 8.2 trace.length = duration / dt.
  const expected = Math.round(cfg.duration_h / (cfg.dt_min / 60));
  check('8.2 trace length matches duration', r.trace.length === expected, `got ${r.trace.length}, expected ${expected}`);
  // 8.3 violation_Fh is also monotonic.
  let monoViol = true;
  for (let k = 1; k < r.trace.length; k++) {
    if (r.trace[k].violation_Fh < r.trace[k - 1].violation_Fh - 1e-9) { monoViol = false; break; }
  }
  check('8.3 violation_Fh is monotonic', monoViol);
  // 8.4 cost = energy*price + violation*penalty.
  const cost_check = cfg.cost_per_kWh * r.energy_kWh + cfg.comfort_penalty * r.violation_Fh;
  check('8.4 cost = energy_$+ violation_$', close(r.cost_$, cost_check, 1e-9), `${r.cost_$.toFixed(6)} vs ${cost_check.toFixed(6)}`);
  // 8.5 Final cumulative energy in trace == returned energy_kWh.
  check('8.5 trace energy_cum_kWh consistency',
    close(r.trace[r.trace.length - 1].energy_cum_kWh, r.energy_kWh, 1e-9));
}

// -----------------------------------------------------------------------------
console.log('\n[9] Different controllers, same scenario, all stay in band');
// -----------------------------------------------------------------------------
{
  const base = {
    T_target: 70, band: 2, duration_h: 12, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.1, forecastNoiseStd: 1.0, forecastHorizon_h: 4, seed: 42,
    controller: {kind: 'bang-bang'} as ControllerSpec,
  };
  const specs: ControllerSpec[] = [
    {kind: 'bang-bang'},
    {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
    {kind: 'fuzzy'},
    {kind: 'mdp-mpc', horizon_h: 4, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 1.0},
  ];
  for (const spec of specs) {
    const r = runTempControl({...base, controller: spec});
    check(`9.x ${spec.kind} 100% comfort`, r.comfort_pct >= 0.99, `${(100*r.comfort_pct).toFixed(1)}%`);
  }
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
