// RUST MIGRATION:
// - Target: src/des/runners/framework_runner.rs.
// - Keep file-for-file as a library runner exposing run_framework_once over migrated DES entity structs/traits.
// - Replace framework construction helpers with explicit builder structs; DES graph callbacks should become PureTransform-style trait impls.
// - Isolate mathjs/logging/seed behavior behind numeric, JsonlLogger, and RNG traits so the runner ports cleanly.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/runners/framework_runner.rs
//                    (module des::runners::framework_runner — hyphen → underscore)
// 1:1 file move. The framework SEIR-with-hospitalization kernel as a callable fn.
//
// Declarations → Rust:
//   fn runFrameworkOnce(config, opts) -> RunResult
//        -> fn run_framework_once(config: &SimConfig, opts: &RunOpts) -> RunResult
//
// Conversion notes (file-specific):
//   - Helper module imported by the binaries (replicate/stepsize-sweep/...), not
//     itself a `fn main()`.
//   - `withSeed(seed, fn)` + `opts.seed ?? Date.now()` -> take an injected
//     `SeededRandom`/`Clock` (shared::capabilities); `opts.seed.unwrap_or_else(||
//     clock.now())`. Do NOT call a global RNG/clock inside.
//   - Builds the EntitySource/Processor/Sink graph -> use those structs' impls;
//     mutual entity references need `Rc<RefCell<..>>` or an arena (borrow checker).
// =============================================================================

// =============================================================================
// Framework kernel as a callable function. Wires up the same SEIR-with-
// hospitalization graph the live demo uses, but every parameter (stepSize,
// horizon, residence intervals, branching probabilities) is supplied via
// SimConfig so we can run replications and stepSize sweeps.
// =============================================================================

import * as math from 'mathjs';
import {EntitySource} from '../entity-source/source';
import {EntityProcessor} from '../entity-processing/processing';
import {EntitySink} from '../entity-sink/sink';
import {StationaryEntity} from '../abstract/abstract';
import {AbstractMovingEntity} from '../entity-moving/moving';
import {bgn, fisherYatesShuffle} from '../general/general';
import {PoissonRandomVariable, UniformRandomVariable} from '../random-variables/rv';
import {ProgramObserver} from '../observers/program-observer';
import {ProbabilityDecisionEntity} from '../entity-decision/probability-decision';
import {JsonlLogger} from '../observability/logger';
import {withSeed} from '../general/prng';
import {
  SimConfig, RunOpts, RunResult,
  COMPARTMENT_ORDER, EDGES,
} from './types';
import {
  averageRecord,
  compartmentPopulations,
  TransitionCounter,
  updatePeaks,
  zeroCompartmentRecord,
} from './shared';

export function runFrameworkOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const seed = opts.seed ?? Date.now();
  return withSeed(seed, () => runFrameworkOnceInner(config, {...opts, seed}));
}

function runFrameworkOnceInner(config: SimConfig, opts: RunOpts): RunResult {
  const sampleEvery = opts.sampleEveryDays ?? 1;
  const logger = opts.logEvents && opts.logPath
    ? new JsonlLogger(opts.logPath, 'info') : null;

  // Reset the global "turn off sources" flag in case a previous run left it set.
  (global as any).turnOffSources = false;

  const stepSize = bgn(config.stepSize);
  const phase1Steps = Math.round(config.phase1Days / config.stepSize);
  const phase2Steps = Math.round((config.horizonDays - config.phase1Days) / config.stepSize);
  const stepsPerSample = Math.max(1, Math.round(sampleEvery / config.stepSize));

  if (!Number.isFinite(phase1Steps) || !Number.isFinite(phase2Steps)) {
    console.warn(`[framework-runner] non-finite phase step counts (phase1=${phase1Steps}, phase2=${phase2Steps}) from stepSize=${config.stepSize}, phase1Days=${config.phase1Days}, horizon=${config.horizonDays} — an open-system config will not terminate.`);
  }
  if (config.stepSize <= 0) {
    console.warn(`[framework-runner] stepSize=${config.stepSize} is not positive; the simulation loop will not advance time correctly.`);
  }

  const obs = new ProgramObserver();

  const uni = (id: string) => {
    const [a, b] = config.residence[id];
    return new UniformRandomVariable({aVal: bgn(a), bVal: bgn(b)});
  };

  const programEntities = new Map<string, StationaryEntity<any>>([
    ['main-source',
      new EntitySource('main-source', {
        turnOffAfterCount: config.sourceCap,
        rv: new UniformRandomVariable({
          aVal: bgn(config.arrivalsInterarrival[0]),
          bVal: bgn(config.arrivalsInterarrival[1]),
        }),
      }).subscribe(obs)],
    ['S',   new EntityProcessor('S',   {rv: uni('S')}).subscribe(obs)],
    ['E',   new EntityProcessor('E',   {rv: uni('E')}).subscribe(obs)],
    ['I-P', new EntityProcessor('I-P', {rv: uni('I-P')}).subscribe(obs)],
    ['I-P-Decision',
      new ProbabilityDecisionEntity('I-P-Decision', {
        rv: uni('I-P-Decision'),
        probabilities: [
          {index: 0, prob: bgn(config.probabilities.asymptomaticShare)},
          {index: 1, prob: math.subtract(bgn(1), bgn(config.probabilities.asymptomaticShare)) as math.BigNumber},
        ],
      }).subscribe(obs)],
    ['I-A', new EntityProcessor('I-A', {rv: uni('I-A')}).subscribe(obs)],
    ['I-S', new EntityProcessor('I-S', {rv: uni('I-S')}).subscribe(obs)],
    ['I-S-Decision',
      new ProbabilityDecisionEntity('I-S-Decision', {
        rv: uni('I-S-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), bgn(config.probabilities.hospitalizationGivenSymptom)) as math.BigNumber},
          {index: 1, prob: bgn(config.probabilities.hospitalizationGivenSymptom)},
        ],
      }).subscribe(obs)],
    ['I-H', new EntityProcessor('I-H', {rv: uni('I-H')}).subscribe(obs)],
    ['I-H-Decision',
      new ProbabilityDecisionEntity('I-H-Decision', {
        rv: uni('I-H-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), bgn(config.probabilities.caseFatalityGivenHospital)) as math.BigNumber},
          {index: 1, prob: bgn(config.probabilities.caseFatalityGivenHospital)},
        ],
      }).subscribe(obs)],
    ['R', new EntityProcessor('R', {rv: uni('R')}).subscribe(obs)],
    ['D', new EntityProcessor('D', {rv: uni('D')}).subscribe(obs)],
    ['main-sink',
      new EntitySink('main-sink', new PoissonRandomVariable(), {}).subscribe(obs)],
  ]);

  for (const [sourceId, targetId] of EDGES) {
    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId) as any;
    source.addOutConnection(target);
    target.addInConnection(source);
  }
  for (const [, e] of programEntities) {
    e.doSetupAfterOutputConn();
    e.doSetupAfterInputConn();
  }

  // Population sampler (folds decision nodes into upstream compartment).
  const stationPopulation = (id: string): number => {
    const e = programEntities.get(id) as any;
    if (!e) {
      console.warn(`[framework-runner] stationPopulation: no entity registered for id "${id}"; counting it as 0 (compartment totals may be wrong).`);
      return 0;
    }
    let n = 0;
    if (e.inputQueue)      n += e.inputQueue.length ?? e.inputQueue.size ?? 0;
    if (e.processingQueue) n += e.processingQueue.length ?? e.processingQueue.size ?? 0;
    if (e.outQueue)        n += e.outQueue.length ?? e.outQueue.size ?? 0;
    if (e.queue && !e.inputQueue) n += e.queue.length ?? e.queue.size ?? 0;
    return n;
  };

  if (logger) {
    logger.log({
      kind: 'sim_start',
      config: {
        kernel: 'framework',
        seed: opts.seed,
        stepSize: config.stepSize,
        phase1Steps,
        phase2Steps,
        sourceCap: config.sourceCap,
        arrivalsInterarrival: config.arrivalsInterarrival,
        residence: config.residence,
        probabilities: config.probabilities,
        edges: EDGES,
      },
    });
  }

  // Transition tracking + JSONL log instrumentation via takeItem wrapper.
  const transitions = new TransitionCounter();
  const lastProcessor = new WeakMap<AbstractMovingEntity<any>, string>();

  let currentDay = 0;
  const PROCESSOR_IDS = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D'];
  for (const id of PROCESSOR_IDS) {
    const proc = programEntities.get(id) as any;
    const orig = proc.takeItem.bind(proc);
    proc.takeItem = (m: AbstractMovingEntity<any>) => {
      const prev = lastProcessor.get(m) ?? '__source__';
      transitions.record(prev, id);
      lastProcessor.set(m, id);
      logger?.log({kind: 'transition', t: currentDay, entity: m.id, from: prev, to: id});
      return orig(m);
    };
  }
  {
    const sink = programEntities.get('main-sink') as any;
    const origTake = sink.takeItem.bind(sink);
    sink.takeItem = (m: AbstractMovingEntity<any>) => {
      const prev = lastProcessor.get(m) ?? '__source__';
      transitions.record(prev, 'main-sink');
      logger?.log({kind: 'transition', t: currentDay, entity: m.id, from: prev, to: 'main-sink'});
      return origTake(m);
    };
  }

  // Trajectory accumulators.
  const popSums = zeroCompartmentRecord();
  const peak = zeroCompartmentRecord();
  let samples = 0;

  const sampleNow = (t: number) => {
    const populations = compartmentPopulations(stationPopulation);
    let totalAlive = 0;
    for (const c of COMPARTMENT_ORDER) {
      popSums[c] += populations[c];
      totalAlive += populations[c];
    }
    updatePeaks(peak, populations);
    const cumD = (programEntities.get('main-sink') as any).destroyedCount as number;
    samples++;
    logger?.log({
      kind: 'tick', t, populations, cumD, alive: totalAlive,
      sourcesActive: !(global as any).turnOffSources,
    });
  };

  const programList = Array.from(programEntities);

  // Quiet the processor's noisy hardcoded per-step logging during the loop
  // (console.log historically, console.debug after the per-step logs were
  // moved to debug level). Restored after the loop.
  const origConsoleLog = console.log;
  const origConsoleDebug = console.debug;
  console.log = () => {};
  console.debug = () => {};

  const startedAt = Date.now();

  for (let i = 0; i < phase1Steps; i++) {
    currentDay = (i + 1) * config.stepSize;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    if ((i + 1) % stepsPerSample === 0) sampleNow(currentDay);
  }
  (global as any).turnOffSources = true;
  origConsoleDebug(`[framework-runner] phase change to 'drain' at day ${currentDay}: sources turned off, running ${phase2Steps} drain steps.`);
  logger?.log({kind: 'phase_change', t: currentDay, phase: 'drain'});
  for (let i = 0; i < phase2Steps; i++) {
    currentDay = (phase1Steps + i + 1) * config.stepSize;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    if ((phase1Steps + i + 1) % stepsPerSample === 0) sampleNow(currentDay);
  }

  console.log = origConsoleLog;
  console.debug = origConsoleDebug;
  const elapsed = Date.now() - startedAt;

  const created = (programEntities.get('main-source') as any).createdCount as number;
  const absorbed = (programEntities.get('main-sink') as any).destroyedCount as number;
  const finalPopulations = compartmentPopulations(stationPopulation);

  const {counts, splits} = transitions.tables();
  const timeAvg = averageRecord(popSums, samples);

  if (logger) {
    logger.log({
      kind: 'sim_end', t: currentDay, elapsedMs: elapsed,
      totals: {created, absorbed, finalPopulations, transitionCounts: counts},
    });
    void logger.close();
  }

  return {
    kernel: 'framework',
    config,
    seed: opts.seed!,
    totals: {created, absorbed},
    finalPopulations,
    transitionCounts: counts,
    splitProbs: splits,
    timeAvgPopulations: timeAvg,
    peakPopulations: peak,
    elapsedMs: elapsed,
  };
}
