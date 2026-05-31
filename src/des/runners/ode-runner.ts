// RUST MIGRATION:
// - Target: src/des/runners/ode_runner.rs.
// - Keep file-for-file as a library runner exposing run_ode_once; State becomes a fixed-field struct or indexed compartment vector.
// - Keep RK/linear-combination helpers private pure functions unless lifted into a numerical Integrator trait.
// - Convert logging/output construction to Result-capable boundaries while preserving RunResult compatibility with other kernels.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/runners/ode_runner.rs
//                    (module des::runners::ode_runner — hyphen → underscore)
// 1:1 file move. Deterministic mean-field SEIR ODE solver (RK4) reference kernel.
//
// Conversion notes (file-specific):
//   - Helper module imported by the binaries, not a `fn main()`.
//   - Fully deterministic; any `Date.now()`/`withSeed` is edge-only -> `Clock`.
//   - `as any` on the assembled `RunResult` -> the concrete typed `RunResult`
//     struct from types.rs (no dynamic field access).
// =============================================================================

// =============================================================================
// Deterministic mean-field SEIR ODE solver (RK4) as a fourth independent
// reference. No randomness, no entities, no events - just the continuous-flow
// limit of the same compartmental model.
//
// dS/dt   = R/mu_R + lambda_src(t) - S/mu_S
// dE/dt   = S/mu_S - E/mu_E
// dI-P/dt = E/mu_E - I-P/mu_IP
// dI-A/dt = I-P*p_a/mu_IP - I-A/mu_IA
// dI-S/dt = I-P*(1-p_a)/mu_IP - I-S/mu_IS
// dI-H/dt = I-S*p_h/mu_IS - I-H/mu_IH
// dR/dt   = I-A/mu_IA + I-S*(1-p_h)/mu_IS + I-H*(1-p_d)/mu_IH - R/mu_R
// dD/dt   = I-H*p_d/mu_IH
// dC/dt   = lambda_src(t)        (cumulative source emissions, used to gate)
//
// lambda_src(t) is 1/mu_arrival while C(t) < cap and t < phase1, else 0.
//
// This is the limit you'd get from a Markov chain with exponential service
// rates 1/mu and infinite-server semantics. In steady state the expected
// populations match Gillespie / FEL-individual / PerIndividualProcessor
// kernels at the same MEAN residence times.
// =============================================================================

import {SimConfig, RunOpts, RunResult, COMPARTMENT_ORDER, Kernel} from './types';
import {JsonlLogger} from '../observability/logger';
import {
  analyticalTransitionTables,
  averageRecord,
  meanResidence,
  updatePeaks,
  zeroCompartmentRecord,
} from './shared';

interface State {
  S: number; E: number; 'I-P': number; 'I-A': number;
  'I-S': number; 'I-H': number; R: number; D: number; C: number;
}

const zeros = (): State => ({S:0, E:0, 'I-P':0, 'I-A':0, 'I-S':0, 'I-H':0, R:0, D:0, C:0});
const lin   = (a: State, b: State, k: number): State => ({
  S:    a.S    + k * b.S,
  E:    a.E    + k * b.E,
  'I-P': a['I-P'] + k * b['I-P'],
  'I-A': a['I-A'] + k * b['I-A'],
  'I-S': a['I-S'] + k * b['I-S'],
  'I-H': a['I-H'] + k * b['I-H'],
  R:    a.R    + k * b.R,
  D:    a.D    + k * b.D,
  C:    a.C    + k * b.C,
});

export function runOdeOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const sampleEvery = opts.sampleEveryDays ?? 1;
  const logger = opts.logEvents && opts.logPath
    ? new JsonlLogger(opts.logPath, 'info') : null;

  const mu = {
    arrival: (config.arrivalsInterarrival[0] + config.arrivalsInterarrival[1]) / 2,
    S:    meanResidence(config, 'S'),
    E:    meanResidence(config, 'E'),
    'I-P': meanResidence(config, 'I-P'),
    'I-A': meanResidence(config, 'I-A'),
    'I-S': meanResidence(config, 'I-S'),
    'I-H': meanResidence(config, 'I-H'),
    R:    meanResidence(config, 'R'),
  };
  const p = config.probabilities;

  const lambda_src = (t: number, C: number) =>
    (C < config.sourceCap && t < config.phase1Days) ? 1/mu.arrival : 0;

  const deriv = (t: number, x: State): State => {
    const src = lambda_src(t, x.C);
    return {
      S:    x.R / mu.R + src - x.S / mu.S,
      E:    x.S / mu.S - x.E / mu.E,
      'I-P': x.E / mu.E - x['I-P'] / mu['I-P'],
      'I-A': x['I-P'] * p.asymptomaticShare / mu['I-P'] - x['I-A'] / mu['I-A'],
      'I-S': x['I-P'] * (1 - p.asymptomaticShare) / mu['I-P'] - x['I-S'] / mu['I-S'],
      'I-H': x['I-S'] * p.hospitalizationGivenSymptom / mu['I-S']
              - x['I-H'] / mu['I-H'],
      R:    x['I-A'] / mu['I-A']
              + x['I-S'] * (1 - p.hospitalizationGivenSymptom) / mu['I-S']
              + x['I-H'] * (1 - p.caseFatalityGivenHospital) / mu['I-H']
              - x.R / mu.R,
      D:    x['I-H'] * p.caseFatalityGivenHospital / mu['I-H'],
      C:    src,
    };
  };

  const dt = 0.05;  // ODE integration timestep, fine enough for RK4
  const stepsPerSample = Math.max(1, Math.round(sampleEvery / dt));
  const totalSteps = Math.round(config.horizonDays / dt);

  let x: State = zeros();
  let t = 0;
  const popSums = zeroCompartmentRecord();
  const peak = zeroCompartmentRecord();
  let samples = 0;

  if (logger) {
    logger.log({kind: 'sim_start', config: {
      kernel: 'ode-rk4', seed: opts.seed ?? 'deterministic',
      dt, tPhase1: config.phase1Days, tMax: config.horizonDays,
      sourceCap: config.sourceCap, meanResidence: mu, probabilities: config.probabilities,
    }});
  }

  const startedAt = Date.now();

  for (let i = 0; i < totalSteps; i++) {
    // Trapezoidal integration for time-averaged populations:
    //   integrate N_c(t) dt = sum over steps of N_c * dt (left-Riemann is fine
    //   for smooth ODE solutions with small dt).
    for (const c of COMPARTMENT_ORDER) popSums[c] += (x as any)[c] * dt;

    // RK4 step
    const k1 = deriv(t, x);
    const k2 = deriv(t + dt/2, lin(x, k1, dt/2));
    const k3 = deriv(t + dt/2, lin(x, k2, dt/2));
    const k4 = deriv(t + dt,   lin(x, k3, dt));
    x = {
      S:    x.S    + dt * (k1.S + 2*k2.S + 2*k3.S + k4.S) / 6,
      E:    x.E    + dt * (k1.E + 2*k2.E + 2*k3.E + k4.E) / 6,
      'I-P': x['I-P'] + dt * (k1['I-P'] + 2*k2['I-P'] + 2*k3['I-P'] + k4['I-P']) / 6,
      'I-A': x['I-A'] + dt * (k1['I-A'] + 2*k2['I-A'] + 2*k3['I-A'] + k4['I-A']) / 6,
      'I-S': x['I-S'] + dt * (k1['I-S'] + 2*k2['I-S'] + 2*k3['I-S'] + k4['I-S']) / 6,
      'I-H': x['I-H'] + dt * (k1['I-H'] + 2*k2['I-H'] + 2*k3['I-H'] + k4['I-H']) / 6,
      R:    x.R    + dt * (k1.R + 2*k2.R + 2*k3.R + k4.R) / 6,
      D:    x.D    + dt * (k1.D + 2*k2.D + 2*k3.D + k4.D) / 6,
      C:    x.C    + dt * (k1.C + 2*k2.C + 2*k3.C + k4.C) / 6,
    };
    t += dt;

    updatePeaks(peak, x as any);

    if ((i + 1) % stepsPerSample === 0) {
      samples++;
      if (logger) {
        const populations: Record<string, number> = {};
        for (const c of COMPARTMENT_ORDER) populations[c] = (x as any)[c];
        logger.log({
          kind: 'tick', t, populations,
          cumD: x.D, alive: COMPARTMENT_ORDER.reduce((a, c) => a + (x as any)[c], 0),
          sourcesActive: t < config.phase1Days && x.C < config.sourceCap,
        });
      }
    }
  }

  const elapsed = Date.now() - startedAt;
  const finalPopulations: Record<string, number> = {};
  for (const c of COMPARTMENT_ORDER) finalPopulations[c] = (x as any)[c];

  // Splits in the ODE are exact, given by the branching probabilities. We
  // compute synthetic counts by integrating each transition's flux over time.
  // (For comparison with stochastic kernels we just emit the analytical splits.)
  const {counts, splits} = analyticalTransitionTables(p);

  const timeAvg = averageRecord(popSums, config.horizonDays);

  if (logger) {
    logger.log({
      kind: 'sim_end', t: config.horizonDays, elapsedMs: elapsed,
      totals: {created: x.C, absorbed: x.D, finalPopulations},
    });
    void logger.close();
  }

  return {
    kernel: 'ode' as Kernel,
    config,
    seed: opts.seed ?? 0,
    totals: {created: x.C, absorbed: x.D},
    finalPopulations,
    transitionCounts: counts,
    splitProbs: splits,
    timeAvgPopulations: timeAvg,
    peakPopulations: peak,
    elapsedMs: elapsed,
  };
}
