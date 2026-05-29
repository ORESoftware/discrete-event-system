'use strict';

// =============================================================================
// runPerIndividualOnce: same SEIR-with-hospitalization graph as the framework
// runner, but every processor station is a PerIndividualProcessor instead of
// the three-queue EntityProcessor. This kernel still uses the framework's
// fixed-step run loop (so it's not classical FEL), but each entity gets an
// independent residence-time draw at takeItem time, which is how a CTMC
// kernel actually behaves. With small stepSize this should converge to the
// FEL reference's behaviour.
// =============================================================================

import * as math from 'mathjs';
import {EntitySource} from '../entity-source/source';
import {EntitySink} from '../entity-sink/sink';
import {StationaryEntity} from '../abstract/abstract';
import {AbstractMovingEntity} from '../entity-moving/moving';
import {bgn, fisherYatesShuffle} from '../general/general';
import {PoissonRandomVariable, UniformRandomVariable} from '../random-variables/rv';
import {ProgramObserver} from '../observers/program-observer';
import {ProbabilityDecisionEntity} from '../entity-decision/probability-decision';
import {PerIndividualProcessor} from '../entity-processing/per-individual-processor';
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

export function runPerIndividualOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const seed = opts.seed ?? Date.now();
  return withSeed(seed, () => runInner(config, {...opts, seed}));
}

function runInner(config: SimConfig, opts: RunOpts): RunResult {
  const sampleEvery = opts.sampleEveryDays ?? 1;
  const logger = opts.logEvents && opts.logPath
    ? new JsonlLogger(opts.logPath, 'info') : null;

  (global as any).turnOffSources = false;

  const stepSize = bgn(config.stepSize);
  const phase1Steps = Math.round(config.phase1Days / config.stepSize);
  const phase2Steps = Math.round((config.horizonDays - config.phase1Days) / config.stepSize);
  const stepsPerSample = Math.max(1, Math.round(sampleEvery / config.stepSize));

  const obs = new ProgramObserver();

  const drawFn = (id: string) => {
    const [a, b] = config.residence[id];
    return () => a + Math.random() * (b - a);
  };

  // Decision nodes still need a RandomVariable (framework contract).
  const uniRv = (id: string) => {
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
    ['S',   new PerIndividualProcessor('S',   {drawDuration: drawFn('S')}).subscribe(obs)],
    ['E',   new PerIndividualProcessor('E',   {drawDuration: drawFn('E')}).subscribe(obs)],
    ['I-P', new PerIndividualProcessor('I-P', {drawDuration: drawFn('I-P')}).subscribe(obs)],
    ['I-P-Decision',
      new ProbabilityDecisionEntity('I-P-Decision', {
        rv: uniRv('I-P-Decision'),
        probabilities: [
          {index: 0, prob: bgn(config.probabilities.asymptomaticShare)},
          {index: 1, prob: math.subtract(bgn(1), bgn(config.probabilities.asymptomaticShare)) as math.BigNumber},
        ],
      }).subscribe(obs)],
    ['I-A', new PerIndividualProcessor('I-A', {drawDuration: drawFn('I-A')}).subscribe(obs)],
    ['I-S', new PerIndividualProcessor('I-S', {drawDuration: drawFn('I-S')}).subscribe(obs)],
    ['I-S-Decision',
      new ProbabilityDecisionEntity('I-S-Decision', {
        rv: uniRv('I-S-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), bgn(config.probabilities.hospitalizationGivenSymptom)) as math.BigNumber},
          {index: 1, prob: bgn(config.probabilities.hospitalizationGivenSymptom)},
        ],
      }).subscribe(obs)],
    ['I-H', new PerIndividualProcessor('I-H', {drawDuration: drawFn('I-H')}).subscribe(obs)],
    ['I-H-Decision',
      new ProbabilityDecisionEntity('I-H-Decision', {
        rv: uniRv('I-H-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), bgn(config.probabilities.caseFatalityGivenHospital)) as math.BigNumber},
          {index: 1, prob: bgn(config.probabilities.caseFatalityGivenHospital)},
        ],
      }).subscribe(obs)],
    ['R', new PerIndividualProcessor('R', {drawDuration: drawFn('R')}).subscribe(obs)],
    ['D', new PerIndividualProcessor('D', {drawDuration: drawFn('D')}).subscribe(obs)],
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

  // Population: PerIndividualProcessor exposes its internal items as
  // `(this as any).items.length`; decisions still have `.queue`.
  const stationPopulation = (id: string): number => {
    const e = programEntities.get(id) as any;
    if (!e) return 0;
    if (Array.isArray(e.items)) return e.items.length;
    if (e.queue) return e.queue.length ?? e.queue.size ?? 0;
    return 0;
  };

  if (logger) {
    logger.log({
      kind: 'sim_start',
      config: {
        kernel: 'per-individual',
        seed: opts.seed,
        stepSize: config.stepSize,
        phase1Steps,
        phase2Steps,
        sourceCap: config.sourceCap,
        residence: config.residence,
        probabilities: config.probabilities,
        edges: EDGES,
      },
    });
  }

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

  const origConsoleLog = console.log;
  console.log = () => {};

  const startedAt = Date.now();
  for (let i = 0; i < phase1Steps; i++) {
    currentDay = (i + 1) * config.stepSize;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    if ((i + 1) % stepsPerSample === 0) sampleNow(currentDay);
  }
  (global as any).turnOffSources = true;
  logger?.log({kind: 'phase_change', t: currentDay, phase: 'drain'});
  for (let i = 0; i < phase2Steps; i++) {
    currentDay = (phase1Steps + i + 1) * config.stepSize;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    if ((phase1Steps + i + 1) % stepsPerSample === 0) sampleNow(currentDay);
  }
  console.log = origConsoleLog;
  const elapsed = Date.now() - startedAt;

  const created  = (programEntities.get('main-source') as any).createdCount as number;
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
    kernel: 'per-individual',
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
