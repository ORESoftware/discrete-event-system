// RUST MIGRATION: Target module `src/des/general/temp_control.rs`.
// RUST MIGRATION: Convert house/outdoor/controller/simulation/tick/result interfaces to `serde` structs; `ControllerSpec`, fuzzy terms, and output levels should become enums.
// RUST MIGRATION: Replace `TempControllerBase` inheritance with a controller trait plus shared embedded state; concrete controllers become structs implementing the trait.
// RUST MIGRATION: Keep physical model and controller helper functions as free functions, or wrap controller steps as `PureTransform` when graph-visible.
// RUST MIGRATION: Inject RNG for outdoor noise, represent BigNumber-free values as `f64`, and return `Result` for invalid time steps, gains, or controller specs.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/temp-control.rs  (module des::general::temp_control)
// 1:1 file move. Indoor temperature control comparing bang-bang/PID/fuzzy/MDP-MPC controllers.
//
// Declarations → Rust:
//   interface HouseParams/OutdoorPattern/ControllerState/TempObs/SimConfig/TickRecord/RunResult -> structs
//   type ControllerSpec = {...}  -> enum (matched in `makeTempController` factory)
//   type Term/OutLevel (fuzzy-set string unions) -> enums
//   abstract class TempControllerBase extends ControllerStation<TempObs, number> -> trait w/ default fns
//   class BangBang/PID/Fuzzy/MdpMpcController extends TempControllerBase -> structs + impl
//   fn trueOutdoorTemp/houseStep/controllerStep/fuzzyDeltaController/mdpMPCController/
//      makeTempController/runTempControl + private membership fns -> fns / assoc fns
//
// Conversion notes (file-specific):
//   - ControllerSpec is a DISCRIMINATED UNION -> `enum`; `makeTempController` becomes a `match`
//     returning a boxed `dyn TempController` (or an enum of controllers).
//   - INJECT RNG/CLOCK: this file RE-DECLARES `mulberry32`; outdoor temp uses Gaussian noise +
//     a diurnal time signal -> take a `RandomSource` and a `Clock` (shared/capabilities); use the
//     single SeededRandom, not a second copy.
//   - fuzzy membership returns named structs `{NL,NS,Z,PS,PL}` -> a small struct or `[f64; 5]`.
//   - TempControllerBase is a template-method base -> trait; controller state (PID integral, MPC
//     buffers) are `&mut self` struct fields; all numerics `f64`.
// =============================================================================
// general/temp-control.ts — INDOOR TEMPERATURE CONTROL as a discrete-event
// system, with four interchangeable controllers compared on the same physical
// house and the same 24-hour outdoor temperature trajectory.
//
// THE PROBLEM
// ───────────
//   Keep indoor temperature within ±2°F of a target (default 70°F) while
//   minimising heating energy use. The outside temperature follows a known
//   diurnal pattern (cold at night, warmer mid-day) plus noise. The
//   controller may consult a NOISY FORECAST of the next H hours of
//   outside temperature — that's where the partial observability lives.
//
// THE DES
// ───────
//   Each TICK is one minute of simulated time (Δt = 1/60 h).
//   Stations (one job each — the cardinal rule):
//
//     OutdoorSource     emits the TRUE outside temperature for this tick
//                        (a diurnal sinusoid + Gaussian noise).
//     ForecastStation   emits a NOISY forecast of T_out over the next
//                        H ticks (the controller can plan against this).
//     ThermostatSensor  emits the indoor temperature (with optional
//                        sensor noise).
//     Comparator        emits the error = T_target − T_in_measured.
//     ControllerStation emits the heater command Q (kW). Pluggable —
//                        bang-bang | PID | fuzzy-logic | MDP-MPC.
//     HeaterActuator    applies Q to the house and emits an energy event.
//     House (physics)   integrates the 1st-order thermal ODE
//                          dT_in/dt = (T_out − T_in) / τ + Q · G.
//     EnergyMeter       accumulates Σ Q · Δt.
//     ComfortMonitor    tallies time spent outside the ±2°F band and the
//                        integrated absolute violation.
//
//   Movables (carried between stations on each tick):
//     T_out reading, forecast vector, T_in reading, error scalar,
//     command Q, energy event, comfort event.
//
//   The whole thing is RECURSIVE — every tick the previous T_in feeds the
//   next physics step, the previous error feeds the PID integral and
//   derivative terms, the previous belief feeds the MPC plan. Feedback is
//   the entire point.
//
// CONTROLLERS
// ───────────
//   1. bang-bang  — heater FULL ON if T_in < T_target, FULL OFF otherwise.
//                   Baseline: stupid but simple. Oscillates around target.
//   2. PID        — classical proportional-integral-derivative feedback.
//                   Q = clip(K_p · e + K_i · ∫e dt + K_d · de/dt, 0, Q_max).
//   3. fuzzy      — Mamdani-style fuzzy controller with 5 linguistic
//                   levels each on (error, error-rate); a 5×5 rule base
//                   maps to 5 power levels; defuzzification by
//                   centre-of-gravity.
//   4. mdp-mpc    — receding-horizon dynamic programming. At each tick
//                   the controller solves a finite-horizon MDP using the
//                   forecast as the model of T_out, picks the optimal
//                   first action, executes it, advances one tick, and
//                   re-plans. This is "MDP for control with a noisy model"
//                   — i.e. the practical face of POMDP-via-certainty-
//                   equivalence. We compare against a pure no-look-ahead
//                   PID for the value of forecast-aware planning.
//
// COMPARISON METRICS
// ──────────────────
//   energy_kWh     total heater energy used over the run
//   comfort_pct    fraction of ticks within ±2°F of target
//   violation_°Fh  ∫ max(0, |T_in − T_target| − 2) dt   (severity of misses)
//   total_cost     cost_per_kWh · energy_kWh + comfort_penalty · violation_°Fh
//
// =============================================================================
//
// AS A DES (base-class hierarchy):
//
//   The four controllers are concrete leaves of ControllerStation<TempObs,
//   number> in `general/des-base/`. Each leaf implements ONLY its
//   controlLaw hook; the base provides the observation/control channels,
//   saturation clamping (uMin = 0, uMax = Q_max), and the per-tick
//   history. See `makeTempController(spec, Q_max)`.
//
// =============================================================================

import {
  ControllerStation, ObservationToken, ControlToken, runIterativeDES,
  intrinsicCheck,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// PHYSICAL HOUSE MODEL
// -----------------------------------------------------------------------------

export interface HouseParams {
  /** Thermal time constant τ (hours). dT/dt has a (T_out − T_in)/τ term. */
  tau: number;
  /** Heater gain G (°F per kW per hour). dT/dt has a Q · G term. */
  G: number;
  /** Maximum heater power Q_max (kW). */
  Q_max: number;
  /** Initial indoor temperature (°F). */
  T_init: number;
}

export const DEFAULT_HOUSE: HouseParams = {tau: 12, G: 1.0, Q_max: 5.0, T_init: 70};

export interface OutdoorPattern {
  /** Mean outdoor temperature over a day (°F). */
  mean: number;
  /** Diurnal swing amplitude (°F). */
  amp: number;
  /** Phase shift in hours (peak temperature occurs at hour `phase + 6`). */
  phase: number;
  /** Standard deviation of additive noise (°F). */
  noiseStd: number;
}

// Default: cold winter day. Peak at 3 PM (phase = 9 ⇒ sin = 1 at t = 15),
// minimum at 3 AM. Mean 25°F, swing ±15°F → coldest morning ~10°F, peak ~40°F.
export const DEFAULT_OUTDOOR: OutdoorPattern = {mean: 25, amp: 15, phase: 9, noiseStd: 1.5};

/** True outside temperature at simulation time `t` hours, with optional rng noise. */
export function trueOutdoorTemp(t_hours: number, pattern: OutdoorPattern, rng?: () => number): number {
  const periodic = pattern.mean + pattern.amp * Math.sin(2 * Math.PI * (t_hours - pattern.phase) / 24);
  if (!rng || pattern.noiseStd === 0) return periodic;
  // Approx. Gaussian via sum-of-uniforms.
  const u = rng() + rng() + rng() + rng() - 2;     // mean 0, std ≈ 0.577
  return periodic + pattern.noiseStd * (u / 0.577);
}

/** Forward-Euler step of the 1st-order thermal ODE.
 *  dT_in/dt = (T_out − T_in) / τ + Q · G */
export function houseStep(T_in: number, T_out: number, Q: number, dt_h: number, h: HouseParams): number {
  const dT = (T_out - T_in) / h.tau + Q * h.G;
  return T_in + dT * dt_h;
}

// -----------------------------------------------------------------------------
// PRNG (mulberry32) — reproducible noise
// -----------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// -----------------------------------------------------------------------------
// CONTROLLERS  (each is one ControllerStation implementation)
// -----------------------------------------------------------------------------

export type ControllerSpec =
  | {kind: 'bang-bang'}
  | {kind: 'pid';      Kp: number; Ki: number; Kd: number}
  | {kind: 'fuzzy'}
  | {kind: 'mdp-mpc';  horizon_h: number; nLevels: number; comfort_penalty: number; cost_per_kWh: number;
                       /** Soft tracking weight inside the band (default 1.0). */
                       trackWeight?: number};

/** Persistent state owned by a controller across ticks. */
export interface ControllerState {
  /** PID integral accumulator and fuzzy-PI integrated output. */
  integral?: number;
  /** Previous error (PID derivative term + Fuzzy error-rate). */
  prevError?: number;
  /** Low-pass-filtered error derivative (PID, °F/h). */
  dErrFilt?: number;
  /** Fuzzy-PI integrated Q command (kW), held across ticks. */
  fuzzyQ?: number;
}

/** Compute heater command Q ∈ [0, Q_max] given the current observation context. */
export function controllerStep(
  spec: ControllerSpec,
  state: ControllerState,
  ctx: {
    T_target: number;
    T_in_meas: number;
    forecast: number[];          // forecasted T_out for the next H ticks (length ≥ horizon)
    dt_h: number;
    Q_max: number;
    house: HouseParams;
  },
): number {
  const e = ctx.T_target - ctx.T_in_meas;
  switch (spec.kind) {
    case 'bang-bang': {
      return e > 0 ? ctx.Q_max : 0;
    }
    case 'pid': {
      // Conditional integration anti-windup + 1st-order low-pass filter on
      // the derivative term (a standard practical addition: raw discrete
      // de/dt is far too noisy and produces "derivative kick" oscillations).
      // Filter time constant τ_d = 5 ticks (5 minutes). Typical industry
      // recommendation is τ_d ≈ Δt · 5–10.
      const Iprev = state.integral ?? 0;
      const de_raw = (e - (state.prevError ?? e)) / ctx.dt_h;
      const alpha = 1 / 6;
      const dErr = (1 - alpha) * (state.dErrFilt ?? 0) + alpha * de_raw;
      state.dErrFilt = dErr;
      const u_pre = spec.Kp * e + spec.Ki * Iprev + spec.Kd * dErr;
      const sat_high = u_pre >= ctx.Q_max && e > 0;
      const sat_low  = u_pre <= 0         && e < 0;
      state.integral = (sat_high || sat_low) ? Iprev : Iprev + e * ctx.dt_h;
      state.prevError = e;
      const u = spec.Kp * e + spec.Ki * state.integral + spec.Kd * dErr;
      return Math.max(0, Math.min(ctx.Q_max, u));
    }
    case 'fuzzy': {
      // Fuzzy-PI: the rule base outputs Δ-Q normalised to [-1,+1] (in
      // units of Q_max per hour); we integrate it over time to produce
      // an offset-free command. This eliminates the steady-state error
      // a pure rule-based fuzzy controller would otherwise carry.
      const de_dt = (e - (state.prevError ?? e)) / ctx.dt_h;
      state.prevError = e;
      const dQ_norm = fuzzyDeltaController(e, de_dt);
      const dQ = dQ_norm * ctx.Q_max * ctx.dt_h * 6;        // 6/h gain factor — empirical
      const Qprev = state.fuzzyQ ?? 0;
      const Q = Math.max(0, Math.min(ctx.Q_max, Qprev + dQ));
      state.fuzzyQ = Q;
      return Q;
    }
    case 'mdp-mpc': {
      return mdpMPCController(
        ctx.T_in_meas, ctx.forecast, spec.horizon_h, spec.nLevels,
        ctx.T_target, ctx.dt_h, ctx.Q_max, ctx.house,
        spec.comfort_penalty, spec.cost_per_kWh,
        spec.trackWeight ?? 1.0,
      );
    }
  }
}

// ── Fuzzy logic controller (Fuzzy-PI form) ───────────────────────────────────
//
// Inputs:
//   e    = T_target − T_in_meas        (°F; positive = room is cold)
//   de/dt = derivative of e            (°F/h; positive = still cooling)
//
// Output: Δ-Q ∈ [-1, +1] (normalised). Integrated externally to produce
//         the actual heater command. This is a Mamdani fuzzy-PI controller —
//         a textbook design (Lee, 1990; Driankov et al., 1996) that yields
//         offset-free tracking like a PI controller but with smooth,
//         interpretable rule-based behaviour.
//
// Linguistic terms on each input: NL, NS, Z, PS, PL (negative-large …
// positive-large). Triangular membership functions evenly spaced over
// the input range; centre-of-gravity defuzzification.
//
// Rule base — sign of Δ-Q follows the diagonal: large positive error AND
// rising error => increase Q rapidly; negative error AND falling error
// => decrease Q rapidly. Mid-cells choose more conservative Δ-Q.
//
//      e\de    NL     NS    Z     PS    PL
//      NL      ND     ND    NS    NS    Z
//      NS      ND     NS    NS    Z     PS
//      Z       NS     NS    Z     PS    PS
//      PS      NS     Z     PS    PS    PD
//      PL      Z      PS    PS    PD    PD
//   ND/PD = ±1.0  (drive)
//   NS/PS = ±0.5  (small)
//   Z     =  0.0

function tri(x: number, a: number, b: number, c: number): number {
  if (x <= a || x >= c) return 0;
  if (x === b) return 1;
  return x < b ? (x - a) / (b - a) : (c - x) / (c - b);
}
const E_RANGE = 6.0;         // °F — saturate above this
const DE_RANGE = 4.0;        // °F/h — saturate above this
function muE(x: number): {NL: number; NS: number; Z: number; PS: number; PL: number} {
  const E = Math.max(-E_RANGE, Math.min(E_RANGE, x));
  return {
    NL: tri(E, -E_RANGE * 1.5, -E_RANGE, -E_RANGE / 2),
    NS: tri(E, -E_RANGE, -E_RANGE / 2, 0),
    Z:  tri(E, -E_RANGE / 2, 0, E_RANGE / 2),
    PS: tri(E, 0, E_RANGE / 2, E_RANGE),
    PL: tri(E, E_RANGE / 2, E_RANGE, E_RANGE * 1.5),
  };
}
function muDE(x: number): {NL: number; NS: number; Z: number; PS: number; PL: number} {
  const X = Math.max(-DE_RANGE, Math.min(DE_RANGE, x));
  return {
    NL: tri(X, -DE_RANGE * 1.5, -DE_RANGE, -DE_RANGE / 2),
    NS: tri(X, -DE_RANGE, -DE_RANGE / 2, 0),
    Z:  tri(X, -DE_RANGE / 2, 0, DE_RANGE / 2),
    PS: tri(X, 0, DE_RANGE / 2, DE_RANGE),
    PL: tri(X, DE_RANGE / 2, DE_RANGE, DE_RANGE * 1.5),
  };
}
type Term = 'NL' | 'NS' | 'Z' | 'PS' | 'PL';
const TERMS: Term[] = ['NL', 'NS', 'Z', 'PS', 'PL'];
type OutLevel = 'ND' | 'NS' | 'Z' | 'PS' | 'PD';
const OUT_VAL: Record<OutLevel, number> = {ND: -1.0, NS: -0.5, Z: 0.0, PS: 0.5, PD: 1.0};
const RULES: Record<Term, Record<Term, OutLevel>> = {
  NL: {NL: 'ND', NS: 'ND', Z: 'NS', PS: 'NS', PL: 'Z'},
  NS: {NL: 'ND', NS: 'NS', Z: 'NS', PS: 'Z',  PL: 'PS'},
  Z:  {NL: 'NS', NS: 'NS', Z: 'Z',  PS: 'PS', PL: 'PS'},
  PS: {NL: 'NS', NS: 'Z',  Z: 'PS', PS: 'PS', PL: 'PD'},
  PL: {NL: 'Z',  NS: 'PS', Z: 'PS', PS: 'PD', PL: 'PD'},
};
/** Mamdani fuzzy controller: returns Δ-Q normalised to [-1, +1]. */
export function fuzzyDeltaController(e: number, de_dt: number): number {
  const ME = muE(e), MD = muDE(de_dt);
  let num = 0, den = 0;
  for (const i of TERMS) {
    for (const j of TERMS) {
      const w = Math.min(ME[i], MD[j]);
      if (w === 0) continue;
      const out = OUT_VAL[RULES[i][j]];
      num += w * out;
      den += w;
    }
  }
  return den > 0 ? num / den : 0;
}

// ── MDP-MPC controller (receding-horizon DP) ─────────────────────────────────
//
// At each tick the controller looks at forecasts for the next H ticks, builds a
// finite-horizon discrete MDP on a grid of (indoor temperature × time), runs
// backward induction to compute the optimal action sequence, and executes the
// FIRST action. Then the world advances one tick, the forecast is updated, and
// we re-plan. This is the ENGINEERING practice equivalent of an MPC: classical
// in process control and a direct application of value iteration.
//
// State:   T_in discretised into bins around the band [T_target − 8, T_target + 8] °F
// Action:  Q ∈ {Q_max · k / (n−1) : k = 0…n-1}    (n discrete power levels)
// Reward:  −cost_per_kWh · Q · Δt − comfort_penalty · max(0, |T_in − T_target| − 2)²

export function mdpMPCController(
  T_in_now: number,
  forecast: number[],
  horizon_h: number,
  nLevels: number,
  T_target: number,
  dt_h: number,
  Q_max: number,
  house: HouseParams,
  comfort_penalty: number,
  cost_per_kWh: number,
  trackWeight: number = 1.0,
): number {
  const H = Math.min(forecast.length, Math.round(horizon_h / dt_h));
  // T_in grid covering [T_target − 10, T_target + 10] in fine 0.1°F steps.
  // The grid must be FINE enough that a single tick of heating moves to a
  // different cell — otherwise the DP cannot tell the difference between
  // actions. Per tick at Q_max=5kW the temperature changes by only
  // 5 · G · Δt = 0.083 °F, so we need T_step ≤ that. We use linear
  // interpolation of V[k+1] at the continuous T_next to remove all
  // grid-quantisation effects.
  const T_lo = T_target - 10, T_hi = T_target + 10, T_step = 0.1;
  const nT = Math.round((T_hi - T_lo) / T_step) + 1;
  const tVal = (i: number) => T_lo + i * T_step;
  // Linear-interpolate V[k+1] at continuous T_next, clamped to the grid.
  const interpV = (Vrow: number[], T: number): number => {
    const x = (T - T_lo) / T_step;
    if (x <= 0) return Vrow[0];
    if (x >= nT - 1) return Vrow[nT - 1];
    const i0 = Math.floor(x), i1 = i0 + 1;
    const w = x - i0;
    return (1 - w) * Vrow[i0] + w * Vrow[i1];
  };
  // Action grid.
  const actions = new Array(nLevels).fill(0).map((_, k) => (Q_max * k) / (nLevels - 1));
  // Backward induction.
  // V[k][i] = optimal cost-to-go starting at tick k with T_in in bin i.
  const V: number[][] = new Array(H + 1).fill(0).map(() => new Array(nT).fill(0));
  const Pi: number[][] = new Array(H).fill(0).map(() => new Array(nT).fill(0));
  // Penalty design: soft quadratic tracking penalty PLUS the larger
  // band-violation penalty. The tracking term is necessary because the
  // band-only penalty is identically zero inside the band, so the
  // cheapest action (Q = 0) wins until T_in finally drifts outside —
  // by which time recovery is expensive. trackWeight defaults to 1.0
  // ($/(°F)²/hour); raise it for tighter tracking, lower it for cheaper
  // operation that just clings to the band boundary.
  for (let k = H - 1; k >= 0; k--) {
    const T_out_k = forecast[k];
    for (let i = 0; i < nT; i++) {
      let bestQ = 0, bestVal = Infinity;
      for (let a = 0; a < actions.length; a++) {
        const Q = actions[a];
        const T_now = tVal(i);
        const T_next = T_now + ((T_out_k - T_now) / house.tau + Q * house.G) * dt_h;
        const futureV = interpV(V[k + 1], T_next);
        const energyCost = cost_per_kWh * Q * dt_h;
        const trackErr = T_now - T_target;
        const dev = Math.max(0, Math.abs(trackErr) - 2);
        const comfortCost = (trackWeight * trackErr * trackErr + comfort_penalty * dev * dev) * dt_h;
        const total = energyCost + comfortCost + futureV;
        if (total < bestVal) { bestVal = total; bestQ = Q; }
      }
      V[k][i] = bestVal;
      Pi[k][i] = bestQ;
    }
  }
  // Look up the policy at the continuous initial T_in via linear
  // interpolation of the action — fall back to the snapped bin if T is
  // outside the grid (in which case behaviour will be conservative).
  const x = (T_in_now - T_lo) / T_step;
  if (x <= 0) return Pi[0][0];
  if (x >= nT - 1) return Pi[0][nT - 1];
  const i0 = Math.floor(x), i1 = i0 + 1;
  // Use the DOMINANT cell — interpolating actions can pick a non-grid
  // power that the simulator wouldn't quantise the same way. The bin is
  // only 0.1°F wide so the difference is negligible.
  const w = x - i0;
  return w < 0.5 ? Pi[0][i0] : Pi[0][i1];
}

// -----------------------------------------------------------------------------
// CONTROLLER STATIONS — concrete leaves of ControllerStation<TempObs, number>
// -----------------------------------------------------------------------------

/** What every controller observes on each tick. */
export interface TempObs {
  T_target: number;
  T_in_meas: number;
  forecast: number[];
  dt_h: number;
  Q_max: number;
  house: HouseParams;
}

/** Common base for the four temperature controllers — owns the
 *  ControllerState across ticks and clamps the heater command to
 *  [0, Q_max]. Subclasses implement only `controlLaw`. */
export abstract class TempControllerBase extends ControllerStation<TempObs, number> {
  protected ctrlState: ControllerState = {};
  protected qMaxCached: number;

  constructor(id: string, qMax: number) {
    super(id);
    this.qMaxCached = qMax;
    // Intrinsic invariant: every emitted control must lie in [0, Q_max].
    // The base ControllerStation.clamp() already enforces this; this
    // validator gives end-of-run confirmation for free.
    this.addValidator(intrinsicCheck<TempControllerBase>({
      name: 'temp-control.u-in-saturation',
      group: 'temp-control-intrinsic',
      predicate: st => {
        const lo = 0;
        const hi = st.qMaxCached;
        for (const u of st.controlHistory) {
          if (u < lo - 1e-9 || u > hi + 1e-9) return false;
        }
        return true;
      },
      expected: '0 ≤ u ≤ Q_max',
      observedFn: st => `n=${st.controlHistory.length}  Q_max=${st.qMaxCached}`,
      details: 'controller emitted a u outside its saturation band',
    }));
  }
  protected override uMin(): number { return 0; }
  protected override uMax(): number { return this.qMaxCached; }
  override reset(): void { super.reset(); this.ctrlState = {}; }
}

export class BangBangController extends TempControllerBase {
  protected controlLaw(o: TempObs): number {
    return controllerStep({kind: 'bang-bang'}, this.ctrlState, o);
  }
}

export class PIDController extends TempControllerBase {
  constructor(id: string, qMax: number,
              private readonly gains: {Kp: number; Ki: number; Kd: number}) {
    super(id, qMax);
  }
  protected controlLaw(o: TempObs): number {
    return controllerStep({kind: 'pid', ...this.gains}, this.ctrlState, o);
  }
}

export class FuzzyController extends TempControllerBase {
  protected controlLaw(o: TempObs): number {
    return controllerStep({kind: 'fuzzy'}, this.ctrlState, o);
  }
}

export class MdpMpcController extends TempControllerBase {
  constructor(id: string, qMax: number,
              private readonly mpcSpec: Extract<ControllerSpec, {kind: 'mdp-mpc'}>) {
    super(id, qMax);
  }
  protected controlLaw(o: TempObs): number {
    return controllerStep(this.mpcSpec, this.ctrlState, o);
  }
}

/** Factory: build the right ControllerStation leaf for a spec. */
export function makeTempController(spec: ControllerSpec, qMax: number, id = 'ctrl'): TempControllerBase {
  switch (spec.kind) {
    case 'bang-bang': return new BangBangController(id, qMax);
    case 'pid':       return new PIDController(id, qMax, {Kp: spec.Kp, Ki: spec.Ki, Kd: spec.Kd});
    case 'fuzzy':     return new FuzzyController(id, qMax);
    case 'mdp-mpc':   return new MdpMpcController(id, qMax, spec);
  }
}

// -----------------------------------------------------------------------------
// SIMULATION RUNNER — orchestrates the stations on each tick.
// -----------------------------------------------------------------------------

export interface SimConfig {
  /** Target indoor temperature (°F). */
  T_target: number;
  /** Comfort band (±band in °F). Default 2. */
  band: number;
  /** Total simulated time (hours). */
  duration_h: number;
  /** Tick length (minutes). */
  dt_min: number;
  /** Controller specification. */
  controller: ControllerSpec;
  /** House parameters. */
  house?: Partial<HouseParams>;
  /** Outdoor temperature pattern. */
  outdoor?: Partial<OutdoorPattern>;
  /** Cost per kWh ($). */
  cost_per_kWh: number;
  /** Comfort penalty ($ per (°F)² per hour outside the band). */
  comfort_penalty: number;
  /** Sensor noise std (°F). */
  sensorNoiseStd?: number;
  /** Forecast noise std (°F). */
  forecastNoiseStd?: number;
  /** Forecast horizon (hours) used by mpc-style controllers. */
  forecastHorizon_h?: number;
  /** PRNG seed. */
  seed?: number;
}

export interface TickRecord {
  tick: number;
  t_h: number;
  T_out_true: number;
  T_out_meas: number;
  T_in_true: number;
  T_in_meas: number;
  error: number;
  Q: number;
  energy_cum_kWh: number;
  in_band: boolean;
  violation_Fh: number;
}

export interface RunResult {
  cfg: SimConfig;
  trace: TickRecord[];
  energy_kWh: number;
  comfort_pct: number;
  violation_Fh: number;
  cost_$: number;
  // Convenience time series for plotting / animation
  ticks: number[];
  T_in: number[];
  T_out: number[];
  Q: number[];
  energy: number[];
}

/** Run a single controller through a full episode. */
export function runTempControl(cfg: SimConfig): RunResult {
  const cls = 'runTempControl';
  Preconditions.positive(cls, 'cfg.dt_min', cfg.dt_min);
  Preconditions.positive(cls, 'cfg.duration_h', cfg.duration_h);
  Preconditions.finite(cls, 'cfg.T_target', cfg.T_target);
  if (cfg.band !== undefined) Preconditions.positive(cls, 'cfg.band', cfg.band);
  if (cfg.sensorNoiseStd !== undefined) Preconditions.nonNegative(cls, 'cfg.sensorNoiseStd', cfg.sensorNoiseStd);
  if (cfg.forecastNoiseStd !== undefined) Preconditions.nonNegative(cls, 'cfg.forecastNoiseStd', cfg.forecastNoiseStd);
  if (cfg.forecastHorizon_h !== undefined) Preconditions.positive(cls, 'cfg.forecastHorizon_h', cfg.forecastHorizon_h);
  Preconditions.nonNegative(cls, 'cfg.cost_per_kWh', cfg.cost_per_kWh);
  Preconditions.nonNegative(cls, 'cfg.comfort_penalty', cfg.comfort_penalty);
  if (cfg.house) {
    if (cfg.house.Q_max !== undefined) Preconditions.positive(cls, 'cfg.house.Q_max', cfg.house.Q_max);
    if (cfg.house.tau !== undefined) Preconditions.positive(cls, 'cfg.house.tau', cfg.house.tau);
  }
  const house = {...DEFAULT_HOUSE, ...(cfg.house ?? {})};
  const outdoor = {...DEFAULT_OUTDOOR, ...(cfg.outdoor ?? {})};
  const dt_h = cfg.dt_min / 60;
  const N = Math.round(cfg.duration_h / dt_h);
  const T_target = cfg.T_target;
  const band = cfg.band ?? 2;
  const sensorStd = cfg.sensorNoiseStd ?? 0;
  const forecastStd = cfg.forecastNoiseStd ?? 0;
  const horizon_h = cfg.forecastHorizon_h ?? 6;
  const horizonTicks = Math.round(horizon_h / dt_h);
  const cost_per_kWh = cfg.cost_per_kWh;
  const comfort_penalty = cfg.comfort_penalty;
  const rng = mulberry32(cfg.seed ?? 12345);
  const fcRng = mulberry32((cfg.seed ?? 12345) ^ 0xa5a5a5a5);
  const sensorRng = mulberry32((cfg.seed ?? 12345) ^ 0x5a5a5a5a);
  let T_in = house.T_init;
  let energy = 0;
  let violation = 0;
  let inBand = 0;
  const trace: TickRecord[] = [];
  const ticks: number[] = [];
  const T_inTrace: number[] = [];
  const T_outTrace: number[] = [];
  const QTrace: number[] = [];
  const energyTrace: number[] = [];
  // Controller is now a ControllerStation<TempObs, number> leaf — calling
  // ctrl.step(o, k, t_h) on each tick uses the station's controlLaw hook
  // (one of bang-bang / PID / fuzzy / mdp-mpc), automatic saturation
  // clamping to [0, Q_max], and the per-tick history.
  const ctrl = makeTempController(cfg.controller, house.Q_max, 'tempctrl');
  for (let k = 0; k < N; k++) {
    const t_h = k * dt_h;
    const T_out_true = trueOutdoorTemp(t_h, outdoor, rng);
    const T_in_meas = T_in + (sensorStd > 0 ? sensorStd * (sensorRng() + sensorRng() + sensorRng() + sensorRng() - 2) / 0.577 : 0);
    // Forecast: peek ahead at the noiseless mean trajectory + Gaussian noise.
    const forecast: number[] = new Array(horizonTicks);
    for (let i = 0; i < horizonTicks; i++) {
      const t_future = t_h + i * dt_h;
      const periodic = outdoor.mean + outdoor.amp * Math.sin(2 * Math.PI * (t_future - outdoor.phase) / 24);
      const fcNoise = forecastStd > 0 ? forecastStd * (fcRng() + fcRng() + fcRng() + fcRng() - 2) / 0.577 : 0;
      forecast[i] = periodic + fcNoise;
    }
    const Q = ctrl.step(
      {T_target, T_in_meas, forecast, dt_h, Q_max: house.Q_max, house},
      k, t_h,
    );
    // House physics + energy + comfort.
    const T_next = houseStep(T_in, T_out_true, Q, dt_h, house);
    energy += Q * dt_h;
    const dev = Math.abs(T_in - T_target);
    const inB = dev <= band;
    if (inB) inBand++;
    violation += Math.max(0, dev - band) * dt_h;
    trace.push({
      tick: k, t_h,
      T_out_true,
      T_out_meas: T_out_true,
      T_in_true: T_in,
      T_in_meas,
      error: T_target - T_in_meas,
      Q,
      energy_cum_kWh: energy,
      in_band: inB,
      violation_Fh: violation,
    });
    ticks.push(k); T_inTrace.push(T_in); T_outTrace.push(T_out_true); QTrace.push(Q); energyTrace.push(energy);
    T_in = T_next;
  }
  const comfort_pct = inBand / N;
  const cost = cost_per_kWh * energy + comfort_penalty * violation;
  return {
    cfg, trace,
    energy_kWh: energy, comfort_pct, violation_Fh: violation, cost_$: cost,
    ticks, T_in: T_inTrace, T_out: T_outTrace, Q: QTrace, energy: energyTrace,
  };
}
