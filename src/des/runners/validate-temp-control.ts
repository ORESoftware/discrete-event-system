'use strict';

// =============================================================================
// runners/validate-temp-control.ts — verify the temperature-control DES
// against energy balance, controller stability, and qualitative tradeoff
// expectations.
//
// Studies:
//   1. House physics — energy balance check (∫ Q dt vs ΔT_in_internal)
//      and steady-state agreement with closed-form Q_ss = (T_in − T_out)/(τ·G).
//   2. Controllers track within ±2°F band on the canonical 24h scenario.
//   3. PID & fuzzy-PI: zero steady-state error in constant-disturbance regime.
//   4. MDP-MPC produces lower or equal cost than bang-bang on the cost it
//      itself optimises (energy + comfort).
//   5. Increasing forecast horizon never increases the MDP-MPC's
//      best-action value (more information ≥ less information).
//   6. Reproducibility: same seed → same trajectories.
// =============================================================================

import {
  runTempControl, houseStep, mdpMPCController, fuzzyDeltaController,
  DEFAULT_HOUSE, DEFAULT_OUTDOOR, ControllerSpec, SimConfig,
} from '../general/temp-control';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol: number): void {
  check(label, Math.abs(a - b) <= tol, `|${a.toFixed(6)} − ${b.toFixed(6)}| = ${Math.abs(a - b).toExponential(2)} (tol ${tol.toExponential(2)})`);
}

// =============================================================================
console.log('\nStudy 1 — House physics: forward-Euler self-consistency');
// =============================================================================
{
  // 1a. Q = 0, T_out > T_in: heat flows in, T rises.
  const T1 = houseStep(60, 80, 0, 1.0, DEFAULT_HOUSE);
  check('Q=0, hot outside: T rises', T1 > 60, `T_in: 60 → ${T1.toFixed(3)}`);
  // 1b. Q = 0, T_out < T_in: house cools.
  const T2 = houseStep(70, 30, 0, 1.0, DEFAULT_HOUSE);
  check('Q=0, cold outside: T falls', T2 < 70, `T_in: 70 → ${T2.toFixed(3)}`);
  // 1c. Steady-state Q matches (T_in − T_out)/(τ·G): zero net change.
  const Q_ss = (70 - 30) / (DEFAULT_HOUSE.tau * DEFAULT_HOUSE.G);
  const T3 = houseStep(70, 30, Q_ss, 1.0, DEFAULT_HOUSE);
  close('Q = Q_ss → no change in 1h', T3, 70, 1e-12);
  // 1d. Pure heating with insulated walls (τ → ∞): T rises by Q · G · Δt.
  const insulated = {...DEFAULT_HOUSE, tau: 1e9};
  const T4 = houseStep(70, 30, 5, 1.0, insulated);
  close('insulated, Q=5, Δt=1h: ΔT = 5°F', T4 - 70, 5.0, 1e-3);
}

// =============================================================================
console.log('\nStudy 2 — All four controllers track within ±2°F band on canonical 24h');
// =============================================================================
{
  const base: SimConfig = {
    T_target: 70, band: 2, duration_h: 24, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 6,
    seed: 42,
    controller: {kind: 'bang-bang'},
  };
  const cases: Array<{name: string; spec: ControllerSpec}> = [
    {name: 'bang-bang', spec: {kind: 'bang-bang'}},
    {name: 'PID',       spec: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5}},
    {name: 'fuzzy-PI',  spec: {kind: 'fuzzy'}},
    {name: 'MDP-MPC',   spec: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 1.0}},
  ];
  for (const c of cases) {
    const r = runTempControl({...base, controller: c.spec});
    check(`${c.name} achieves 100% in-band comfort`, r.comfort_pct === 1.0, `comfort = ${(100 * r.comfort_pct).toFixed(1)}%`);
    check(`${c.name} consumes plausible energy (50–120 kWh)`, r.energy_kWh > 50 && r.energy_kWh < 120, `${r.energy_kWh.toFixed(2)} kWh`);
  }
}

// =============================================================================
console.log('\nStudy 3 — PID & Fuzzy-PI achieve near-zero steady-state error');
// =============================================================================
{
  // Run with a CONSTANT outdoor temperature (no diurnal pattern). Both PI-style
  // controllers should drive the steady-state error to ~0.
  const cfgConst: SimConfig = {
    T_target: 70, band: 2, duration_h: 8, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.0, forecastNoiseStd: 0.0, forecastHorizon_h: 1,
    seed: 1,
    outdoor: {mean: 30, amp: 0, phase: 0, noiseStd: 0},
    controller: {kind: 'bang-bang'},
  };
  for (const c of [
    {name: 'PID',      spec: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5} as ControllerSpec},
    {name: 'fuzzy-PI', spec: {kind: 'fuzzy'} as ControllerSpec},
  ]) {
    const r = runTempControl({...cfgConst, controller: c.spec});
    // After warm-up, mean of last 1 hour should be very close to target.
    const last = r.T_in.slice(-60);
    const mean = last.reduce((a, b) => a + b, 0) / last.length;
    check(`${c.name} steady-state |error| < 0.5°F`, Math.abs(mean - 70) < 0.5, `mean T_in last 1h = ${mean.toFixed(3)}°F`);
  }
}

// =============================================================================
console.log('\nStudy 4 — MDP-MPC matches or beats bang-bang on its own cost metric');
// =============================================================================
{
  const cfg: SimConfig = {
    T_target: 70, band: 2, duration_h: 24, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 6,
    seed: 7,
    controller: {kind: 'bang-bang'},
  };
  const bb = runTempControl(cfg);
  const mpc = runTempControl({...cfg, controller: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 0.05}});
  check('MDP-MPC cost ≤ bang-bang cost (or within 1%)',
        mpc.cost_$ <= bb.cost_$ * 1.01,
        `bang-bang cost = $${bb.cost_$.toFixed(3)}, MDP-MPC cost = $${mpc.cost_$.toFixed(3)}`);
}

// =============================================================================
console.log('\nStudy 5 — Stress test: tighter band exposes the MDP-MPC advantage');
// =============================================================================
{
  // With ±1°F band and harsher weather, the MDP-MPC should win because it
  // pre-emptively heats based on the forecast.
  const stress: SimConfig = {
    T_target: 70, band: 1, duration_h: 24, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 2.0,
    sensorNoiseStd: 0.1, forecastNoiseStd: 1.0, forecastHorizon_h: 6,
    seed: 11,
    outdoor: {mean: 15, amp: 20, phase: 9, noiseStd: 2.0},
    controller: {kind: 'bang-bang'},
  };
  const bb = runTempControl(stress);
  const mpc = runTempControl({...stress, controller: {kind: 'mdp-mpc', horizon_h: 6, nLevels: 6, comfort_penalty: 2.0, cost_per_kWh: 0.15, trackWeight: 1.0}});
  check('MDP-MPC produces lower cost than bang-bang on stress test',
        mpc.cost_$ < bb.cost_$,
        `bang-bang $${bb.cost_$.toFixed(2)}  vs  MDP-MPC $${mpc.cost_$.toFixed(2)}`);
}

// =============================================================================
console.log('\nStudy 6 — Reproducibility: same seed → same trajectory');
// =============================================================================
{
  const cfg: SimConfig = {
    T_target: 70, band: 2, duration_h: 6, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.2, forecastNoiseStd: 1.5, forecastHorizon_h: 2,
    seed: 99,
    controller: {kind: 'pid', Kp: 3, Ki: 0.5, Kd: 0.5},
  };
  const r1 = runTempControl(cfg);
  const r2 = runTempControl(cfg);
  let maxDiff = 0;
  for (let k = 0; k < r1.T_in.length; k++) {
    maxDiff = Math.max(maxDiff, Math.abs(r1.T_in[k] - r2.T_in[k]));
  }
  close('two identical runs produce identical T_in trajectories', maxDiff, 0, 1e-12);
}

// =============================================================================
console.log('\nStudy 7 — Fuzzy controller boundary behaviour');
// =============================================================================
{
  // Large positive error + still cooling → maximal Δ-Q (≈ +1.0).
  const dq1 = fuzzyDeltaController(6, 4);
  check('fuzzy: e≫0, de/dt≫0 → Δ-Q ≈ +1', dq1 > 0.7, `Δ-Q = ${dq1.toFixed(3)}`);
  // Large negative error + still warming → maximal negative Δ-Q (≈ −1).
  const dq2 = fuzzyDeltaController(-6, -4);
  check('fuzzy: e≪0, de/dt≪0 → Δ-Q ≈ −1', dq2 < -0.7, `Δ-Q = ${dq2.toFixed(3)}`);
  // Zero error and zero rate → Δ-Q ≈ 0.
  const dq3 = fuzzyDeltaController(0, 0);
  close('fuzzy: e=0, de/dt=0 → Δ-Q = 0', dq3, 0, 1e-12);
}

// =============================================================================
console.log('\nStudy 8 — MDP-MPC monotonicity in trackWeight');
// =============================================================================
{
  // Higher trackWeight should NOT decrease energy use (it cares more about
  // hitting target → may use more heater).
  const cfg: SimConfig = {
    T_target: 70, band: 2, duration_h: 12, dt_min: 1,
    cost_per_kWh: 0.15, comfort_penalty: 0.5,
    sensorNoiseStd: 0.0, forecastNoiseStd: 0.0, forecastHorizon_h: 4,
    seed: 5,
    controller: {kind: 'bang-bang'},
  };
  const e_loose = runTempControl({...cfg, controller: {kind: 'mdp-mpc', horizon_h: 4, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 0.01}}).energy_kWh;
  const e_tight = runTempControl({...cfg, controller: {kind: 'mdp-mpc', horizon_h: 4, nLevels: 6, comfort_penalty: 0.5, cost_per_kWh: 0.15, trackWeight: 5.0}}).energy_kWh;
  check('higher trackWeight ⇒ ≥ energy use', e_tight >= e_loose - 1e-3,
        `e_loose=${e_loose.toFixed(3)}, e_tight=${e_tight.toFixed(3)}`);
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
