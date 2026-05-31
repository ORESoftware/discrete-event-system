'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/runners/gillespie_runner.rs
//                    (module des::runners::gillespie_runner — hyphen → underscore)
// 1:1 file move. Gillespie SSA (direct method) compartment-level kernel.
//
// Conversion notes (file-specific):
//   - Helper module imported by the binaries, not a `fn main()`.
//   - `Math.random()` (exponential dt draw + reaction selection) + `withSeed`
//     -> inject `RandomSource` / `SeededRandom`; no ambient RNG.
//   - `interface Reaction { propensity: () => number; fire: () => void }` closures
//     -> a `struct Reaction` holding `Box<dyn Fn() -> f64>` / `Box<dyn FnMut()>`,
//     or (cleaner) an enum of reactions + a `match`; mind the borrow checker on
//     the shared population state captured by `fire`.
// =============================================================================

// =============================================================================
// Gillespie Stochastic Simulation Algorithm (direct method) as a third
// independent validator. Operates ONLY at the compartment level - no entity
// objects, no event list, just N_c counts and per-reaction propensities.
//
// Each transition becomes a "reaction" whose propensity is N_from / mean(res)
// times the branching probability where applicable. Each iteration:
//   1. Compute total propensity lambda = sum of all reaction propensities.
//   2. Draw exponential dt = -ln(U) / lambda.
//   3. Pick reaction proportional to its propensity, fire it.
//
// Because the rate of leaving a compartment is N_c / mean(res), each
// individual implicitly has its own exponential exit clock. This is M/M/inf
// with EXPONENTIAL service. By the same Little's-law argument that powers our
// FEL-individual / PerIndividualProcessor kernels, the steady-state mean
// populations agree with the uniform-residence kernels at the same MEAN
// residence; only higher moments (variance, peaks) differ.
// =============================================================================

import {SimConfig, RunOpts, RunResult, COMPARTMENT_ORDER, Kernel} from './types';
import {withSeed} from '../general/prng';
import {RandomSource, DEFAULT_RANDOM} from '../shared/capabilities';
import {JsonlLogger} from '../observability/logger';
import {
  averageRecord,
  meanResidence,
  TransitionCounter,
  updatePeaks,
  zeroCompartmentRecord,
} from './shared';

interface Reaction {
  id: string;
  from: string;
  to: string;
  propensity: () => number;
  fire: () => void;
}

export function runGillespieOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const seed = opts.seed ?? Date.now();
  return withSeed(seed, () => runGillespieInner(config, {...opts, seed}));
}

function runGillespieInner(config: SimConfig, opts: RunOpts, rng: RandomSource = DEFAULT_RANDOM): RunResult {
  const sampleEvery = opts.sampleEveryDays ?? 1;
  const logger = opts.logEvents && opts.logPath
    ? new JsonlLogger(opts.logPath, 'info') : null;

  const N: Record<string, number> = {
    S: 0, E: 0, 'I-P': 0, 'I-A': 0, 'I-S': 0, 'I-H': 0, R: 0,
  };

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

  const transitions = new TransitionCounter();

  let absorbed = 0;
  let sourceCreated = 0;
  let phase2 = false;
  let t = 0;

  const reactions: Reaction[] = [
    {id: 'src',     from: '__source__', to: 'S',
     propensity: () => (sourceCreated < config.sourceCap && !phase2) ? 1/mu.arrival : 0,
     fire: () => { N.S++; sourceCreated++; transitions.record('__source__', 'S'); }},
    {id: 'S->E',    from: 'S',   to: 'E',
     propensity: () => N.S / mu.S,
     fire: () => { N.S--; N.E++; transitions.record('S', 'E'); }},
    {id: 'E->I-P',  from: 'E',   to: 'I-P',
     propensity: () => N.E / mu.E,
     fire: () => { N.E--; N['I-P']++; transitions.record('E', 'I-P'); }},
    {id: 'I-P->I-A', from: 'I-P', to: 'I-A',
     propensity: () => N['I-P'] * p.asymptomaticShare / mu['I-P'],
     fire: () => { N['I-P']--; N['I-A']++; transitions.record('I-P', 'I-A'); }},
    {id: 'I-P->I-S', from: 'I-P', to: 'I-S',
     propensity: () => N['I-P'] * (1 - p.asymptomaticShare) / mu['I-P'],
     fire: () => { N['I-P']--; N['I-S']++; transitions.record('I-P', 'I-S'); }},
    {id: 'I-A->R', from: 'I-A', to: 'R',
     propensity: () => N['I-A'] / mu['I-A'],
     fire: () => { N['I-A']--; N.R++; transitions.record('I-A', 'R'); }},
    {id: 'I-S->R', from: 'I-S', to: 'R',
     propensity: () => N['I-S'] * (1 - p.hospitalizationGivenSymptom) / mu['I-S'],
     fire: () => { N['I-S']--; N.R++; transitions.record('I-S', 'R'); }},
    {id: 'I-S->I-H', from: 'I-S', to: 'I-H',
     propensity: () => N['I-S'] * p.hospitalizationGivenSymptom / mu['I-S'],
     fire: () => { N['I-S']--; N['I-H']++; transitions.record('I-S', 'I-H'); }},
    {id: 'I-H->R', from: 'I-H', to: 'R',
     propensity: () => N['I-H'] * (1 - p.caseFatalityGivenHospital) / mu['I-H'],
     fire: () => { N['I-H']--; N.R++; transitions.record('I-H', 'R'); }},
    {id: 'I-H->D', from: 'I-H', to: 'D',
     propensity: () => N['I-H'] * p.caseFatalityGivenHospital / mu['I-H'],
     fire: () => {
       N['I-H']--;
       absorbed++;
       // Mirror the FEL/framework topology: I-H -> D, then D -> main-sink.
       transitions.record('I-H', 'D');
       transitions.record('D', 'main-sink');
     }},
    {id: 'R->S', from: 'R', to: 'S',
     propensity: () => N.R / mu.R,
     fire: () => { N.R--; N.S++; transitions.record('R', 'S'); }},
  ];

  const popSums = zeroCompartmentRecord();
  const peak = zeroCompartmentRecord();
  let nextSampleAt = sampleEvery;
  let samples = 0;

  const sampleAt = (tNow: number) => {
    let totalAlive = 0;
    const pops: Record<string, number> = {};
    for (const c of COMPARTMENT_ORDER) {
      pops[c] = N[c];
      totalAlive += N[c];
    }
    updatePeaks(peak, pops);
    samples++;
    logger?.log({kind: 'tick', t: tNow, populations: pops, cumD: absorbed,
                 alive: totalAlive, sourcesActive: !phase2});
  };

  if (logger) {
    logger.log({kind: 'sim_start', config: {
      kernel: 'gillespie-ssa', seed: opts.seed,
      tPhase1: config.phase1Days, tMax: config.horizonDays,
      sourceCap: config.sourceCap, meanResidence: mu, probabilities: config.probabilities,
    }});
  }

  const startedAt = Date.now();

  while (t < config.horizonDays) {
    if (!phase2 && t >= config.phase1Days) {
      phase2 = true;
      logger?.log({kind: 'phase_change', t: Math.floor(t), phase: 'drain'});
    }

    const props = reactions.map(r => r.propensity());
    const total = props.reduce((a, b) => a + b, 0);

    if (total <= 0) {
      // No reactions enabled. Skip ahead.
      const dt = config.horizonDays - t;
      for (const c of COMPARTMENT_ORDER) popSums[c] += N[c] * dt;
      while (nextSampleAt <= config.horizonDays) {
        sampleAt(nextSampleAt);
        nextSampleAt += sampleEvery;
      }
      t = config.horizonDays;
      break;
    }

    const dt = -Math.log(rng.nextFloat()) / total;
    for (const c of COMPARTMENT_ORDER) popSums[c] += N[c] * dt;
    while (nextSampleAt <= t + dt && nextSampleAt <= config.horizonDays) {
      sampleAt(nextSampleAt);
      nextSampleAt += sampleEvery;
    }
    t += dt;
    if (t > config.horizonDays) break;

    const u = rng.nextFloat() * total;
    let cum = 0, fired = reactions.length - 1;
    for (let i = 0; i < reactions.length; i++) {
      cum += props[i];
      if (u < cum) { fired = i; break; }
    }
    reactions[fired].fire();
    logger?.log({kind: 'transition', t, from: reactions[fired].from, to: reactions[fired].to});
  }

  const elapsed = Date.now() - startedAt;
  const finalPopulations: Record<string, number> = {};
  for (const c of COMPARTMENT_ORDER) finalPopulations[c] = N[c];

  const {counts, splits} = transitions.tables();

  const timeAvg = averageRecord(popSums, config.horizonDays);

  if (logger) {
    logger.log({
      kind: 'sim_end', t: config.horizonDays, elapsedMs: elapsed,
      totals: {created: sourceCreated, absorbed, finalPopulations, transitionCounts: counts},
    });
    void logger.close();
  }

  return {
    kernel: 'gillespie' as Kernel,
    config,
    seed: opts.seed!,
    totals: {created: sourceCreated, absorbed},
    finalPopulations,
    transitionCounts: counts,
    splitProbs: splits,
    timeAvgPopulations: timeAvg,
    peakPopulations: peak,
    elapsedMs: elapsed,
  };
}
