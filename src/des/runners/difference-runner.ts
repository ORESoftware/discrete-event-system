// RUST MIGRATION:
// - Target: src/des/runners/difference_runner.rs.
// - Keep file-for-file as a library runner; export run_difference_once, analytical_steady_state, max_stable_step, and SteadyState.
// - Convert State and MeanResidences to private structs, preserving COMPARTMENT_ORDER indexing with typed keys or small enums.
// - Pure numerical helpers stay private module functions unless promoted into a PureTransform-style trait implementation.
'use strict';

// =============================================================================
// Discrete-time difference-equation kernel for the SEIR-with-hospitalization
// model, plus the closed-form steady-state solution.
//
// HOW TO RUN
// ----------
//   1. In-repo (TypeScript), executes this kernel via the steady-state driver:
//
//        npm run build
//        N=5 HORIZON=10000 node dist/des/runners/steady-state.js
//
//      The driver runs analyticalSteadyState(), runDifferenceOnce() at a
//      sweep of dt values, the ODE RK4 integrator, and the stochastic
//      kernels for cross-comparison. See ./MATH.md for the derivation.
//
//   2. External references (Python / Octave / R) reimplement the same
//      difference equation and ODE in different tools as second-opinion
//      verification:
//
//        bash ../../../external-references/run-all.sh
//        node dist/des/runners/validate-with-externals.js
//
//      run-all.sh discovers interpreters via $PATH using these env vars:
//        PYTHON_BIN=python3   (override with PYTHON_BIN=/opt/py311/bin/python)
//        OCTAVE_BIN=octave    (override with OCTAVE_BIN=octave-cli)
//        RSCRIPT_BIN=Rscript  (override with RSCRIPT_BIN=Rscript-4.3)
//      Tools whose interpreter is missing are skipped with a message.
//      See external-references/README.md for details.
//
// Continuous-time mean-field ODE (this is what `ode-runner.ts` integrates):
//
//   dN_S /dt  = lambda(t) + N_R / mu_R - N_S / mu_S
//   dN_E /dt  = N_S / mu_S - N_E / mu_E
//   dN_IP/dt  = N_E / mu_E - N_IP / mu_IP
//   dN_IA/dt  =       p_a  * N_IP / mu_IP - N_IA / mu_IA
//   dN_IS/dt  = (1 - p_a)  * N_IP / mu_IP - N_IS / mu_IS
//   dN_IH/dt  =       p_h  * N_IS / mu_IS - N_IH / mu_IH
//   dN_R /dt  = N_IA / mu_IA + (1 - p_h) * N_IS / mu_IS
//                            + (1 - p_d) * N_IH / mu_IH - N_R / mu_R
//   dN_D /dt  =       p_d  * N_IH / mu_IH       (D drains instantly to sink)
//
// Forward-Euler difference equation (this kernel):
//
//   N(t + dt) = N(t) + dt * f(N(t), t)        where f = right-hand side above
//
//   Equivalently as a linear iteration:
//   N(t + dt) = (I + dt * A) * N(t) + dt * b(t)
//
//   with A = transition-rate matrix and b(t) = [lambda(t), 0, 0, 0, 0, 0, 0]^T.
//
// Stability: forward Euler requires |1 - dt / mu_c| < 1 for every compartment,
// i.e. dt < 2 * min(mu_c). For our defaults min(mu_c) = 0.20 (D), so dt < 0.4.
//
// Steady state (open system, lambda(t) = lambda constant):
//   Set N(t + dt) = N(t)  =>  A * N* = -b  =>  see analyticalSteadyState().
// The derivation lives in src/des/runners/MATH.md.
// =============================================================================

import {SimConfig, RunOpts, RunResult, COMPARTMENT_ORDER, Kernel} from './types';
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

const zeros = (): State => ({
  S: 0, E: 0, 'I-P': 0, 'I-A': 0, 'I-S': 0, 'I-H': 0, R: 0, D: 0, C: 0,
});

interface MeanResidences {
  arrival: number; S: number; E: number; 'I-P': number; 'I-A': number;
  'I-S': number; 'I-H': number; R: number; D: number;
}

function mus(config: SimConfig): MeanResidences {
  return {
    arrival: (config.arrivalsInterarrival[0] + config.arrivalsInterarrival[1]) / 2,
    S:    meanResidence(config, 'S'),
    E:    meanResidence(config, 'E'),
    'I-P': meanResidence(config, 'I-P'),
    'I-A': meanResidence(config, 'I-A'),
    'I-S': meanResidence(config, 'I-S'),
    'I-H': meanResidence(config, 'I-H'),
    R:    meanResidence(config, 'R'),
    D:    meanResidence(config, 'D'),
  };
}

export function runDifferenceOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const dt = config.stepSize;
  const mu = mus(config);
  const p  = config.probabilities;

  const lambdaSrc = (t: number, C: number) =>
    (C < config.sourceCap && t < config.phase1Days) ? 1 / mu.arrival : 0;

  let x: State = zeros();
  let t = 0;
  const totalSteps = Math.max(1, Math.round(config.horizonDays / dt));
  const popSums = zeroCompartmentRecord();
  const peak = zeroCompartmentRecord();
  const startedAt = Date.now();

  for (let i = 0; i < totalSteps; i++) {
    for (const c of COMPARTMENT_ORDER) popSums[c] += (x as any)[c] * dt;

    const src = lambdaSrc(t, x.C);
    const dS  = src + x.R / mu.R - x.S / mu.S;
    const dE  = x.S / mu.S - x.E / mu.E;
    const dIP = x.E / mu.E - x['I-P'] / mu['I-P'];
    const dIA = x['I-P'] * p.asymptomaticShare         / mu['I-P']
              - x['I-A'] / mu['I-A'];
    const dIS = x['I-P'] * (1 - p.asymptomaticShare)   / mu['I-P']
              - x['I-S'] / mu['I-S'];
    const dIH = x['I-S'] * p.hospitalizationGivenSymptom / mu['I-S']
              - x['I-H'] / mu['I-H'];
    const dR  = x['I-A'] / mu['I-A']
              + x['I-S'] * (1 - p.hospitalizationGivenSymptom) / mu['I-S']
              + x['I-H'] * (1 - p.caseFatalityGivenHospital)   / mu['I-H']
              - x.R / mu.R;
    const dD  = x['I-H'] * p.caseFatalityGivenHospital / mu['I-H'];

    x = {
      S:    x.S    + dt * dS,
      E:    x.E    + dt * dE,
      'I-P': x['I-P'] + dt * dIP,
      'I-A': x['I-A'] + dt * dIA,
      'I-S': x['I-S'] + dt * dIS,
      'I-H': x['I-H'] + dt * dIH,
      R:    x.R    + dt * dR,
      D:    x.D    + dt * dD,
      C:    x.C    + dt * src,
    };
    t += dt;

    updatePeaks(peak, x as any);
  }

  const elapsed = Date.now() - startedAt;
  const finalPopulations: Record<string, number> = {};
  for (const c of COMPARTMENT_ORDER) finalPopulations[c] = (x as any)[c];

  const {counts, splits} = analyticalTransitionTables(p);

  const timeAvg = averageRecord(popSums, config.horizonDays);

  return {
    kernel: 'difference' as Kernel,
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

// -----------------------------------------------------------------------------
// Closed-form analytical steady state for the open system (lambda const).
//
// Let q = (1 - p_a) * p_h * p_d         (the per-S-pass death fraction).
// Let lambda = 1 / mu_arr               (mean source rate).
// At fixed point, flow conservation gives:
//
//   f_S  = lambda / q
//   f_E  = f_S
//   f_IP = f_S
//   f_IA = p_a * f_S
//   f_IS = (1 - p_a) * f_S
//   f_IH = p_h * (1 - p_a) * f_S
//   f_D  = p_d * p_h * (1 - p_a) * f_S = q * f_S = lambda
//   f_R  = (1 - q) * f_S
//
// Then by Little's law for M/M/inf service:  N*_c = mu_c * f_c.
// -----------------------------------------------------------------------------

export interface SteadyState {
  /** Mean residence times (days) used in the derivation. */
  mu: MeanResidences;
  /** Source emission rate (entities/day). */
  lambda: number;
  /** Per-S-pass death fraction (1 - p_a) * p_h * p_d. */
  q: number;
  /** Throughput at S (entities/day). All other f_c are simple multiples. */
  fS: number;
  /** Throughput rates f_c into each compartment (entities/day). */
  flows: Record<string, number>;
  /** Fixed-point populations N*_c = mu_c * f_c. */
  populations: Record<string, number>;
  /** Sum N*_S + ... + N*_R + N*_D. */
  totalAlive: number;
}

export function analyticalSteadyState(config: SimConfig): SteadyState {
  const mu = mus(config);
  const p  = config.probabilities;

  const lambda = 1 / mu.arrival;
  const q      = (1 - p.asymptomaticShare)
               * p.hospitalizationGivenSymptom
               * p.caseFatalityGivenHospital;
  const fS     = lambda / q;

  const flows = {
    S:    fS,
    E:    fS,
    'I-P': fS,
    'I-A': p.asymptomaticShare         * fS,
    'I-S': (1 - p.asymptomaticShare)   * fS,
    'I-H': p.hospitalizationGivenSymptom * (1 - p.asymptomaticShare) * fS,
    'D':   q * fS,           // == lambda by construction
    R:    (1 - q) * fS,
  };

  const populations = {
    S:    mu.S    * flows.S,
    E:    mu.E    * flows.E,
    'I-P': mu['I-P'] * flows['I-P'],
    'I-A': mu['I-A'] * flows['I-A'],
    'I-S': mu['I-S'] * flows['I-S'],
    'I-H': mu['I-H'] * flows['I-H'],
    R:    mu.R    * flows.R,
    'D':   mu.D    * flows['D'],
  };

  const totalAlive = COMPARTMENT_ORDER.reduce(
    (s, c) => s + (populations as any)[c], 0) + populations['D'];

  return {mu, lambda, q, fS, flows, populations, totalAlive};
}

// -----------------------------------------------------------------------------
// Largest forward-Euler step that is provably stable for this model.
// |1 - dt / mu_c| < 1  =>  dt < 2 * min(mu_c).
// -----------------------------------------------------------------------------
export function maxStableStep(config: SimConfig): number {
  const mu = mus(config);
  return 2 * Math.min(mu.S, mu.E, mu['I-P'], mu['I-A'],
                      mu['I-S'], mu['I-H'], mu.R, mu.D);
}
