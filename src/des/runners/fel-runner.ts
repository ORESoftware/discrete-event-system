// RUST MIGRATION:
// - Target: src/des/runners/fel_runner.rs.
// - Keep file-for-file as a library runner exposing run_fel_once; FelEvent becomes an ordered event struct.
// - Replace Math.random fallbacks with an injected RNG trait and keep with_seed behavior as a deterministic RNG adapter.
// - Convert throw/implicit failure points to Result only at construction/logging boundaries; event-loop helpers can stay private.
'use strict';

// =============================================================================
// FEL (Future-Event-List) reference kernel as a callable function. Supports
// two service disciplines, switched via opts.service:
//
//   'fifo'        Single per-station service clock, FIFO queue (M/M/1).
//                 Matches the framework's three-queue EntityProcessor, which
//                 also has one rv-driven completion clock per station.
//   'individual'  Per-entity exit clock drawn at arrival (M/M/inf). Matches
//                 the new framework PerIndividualProcessor and is the
//                 physically correct semantic for an SEIR model where many
//                 individuals can be sick concurrently.
//
// Both modes share the rest of the kernel (event loop, sampling, logging,
// transition counting) so we can A/B them under a single comparison rig.
// =============================================================================

import {JsonlLogger} from '../observability/logger';
import {withSeed} from '../general/prng';
import {
  SimConfig, RunOpts, RunResult,
  COMPARTMENT_ORDER, EDGES, buildSuccessors,
} from './types';
import {
  averageRecord,
  compartmentPopulations,
  TransitionCounter,
  updatePeaks,
  zeroCompartmentRecord,
} from './shared';

const drawUniform = (a: number, b: number) => a + Math.random() * (b - a);

const drawSuccessor = (
  successors: Record<string, Array<{prob: number, to: string}>>,
  from: string,
): string => {
  const succs = successors[from];
  if (succs.length === 1) return succs[0].to;
  const r = Math.random();
  let cum = 0;
  for (const s of succs) {
    cum += s.prob;
    if (r < cum) return s.to;
  }
  if (Math.abs(cum - 1) > 1e-6) {
    console.warn(`[fel-runner] successor probabilities from "${from}" sum to ${cum} (≠ 1); draw r=${r} fell through, defaulting to last successor "${succs[succs.length - 1].to}".`);
  }
  return succs[succs.length - 1].to;
};

interface FelEvent {
  time: number;
  kind: 'source' | 'service' | 'exit';
  /** kind=service or kind=exit */
  station?: string;
  /** kind=exit: the specific entity */
  entity?: string;
}

const insertEvent = (fel: FelEvent[], e: FelEvent) => {
  let i = 0;
  while (i < fel.length && fel[i].time <= e.time) i++;
  fel.splice(i, 0, e);
};

export function runFelOnce(config: SimConfig, opts: RunOpts = {}): RunResult {
  const seed = opts.seed ?? Date.now();
  return withSeed(seed, () => runFelOnceInner(config, {...opts, seed}));
}

function runFelOnceInner(config: SimConfig, opts: RunOpts): RunResult {
  const sampleEvery = opts.sampleEveryDays ?? 1;
  const service = opts.service ?? 'fifo';
  const logger = opts.logEvents && opts.logPath
    ? new JsonlLogger(opts.logPath, 'info') : null;

  const successors = buildSuccessors(config.probabilities);

  // For 'fifo' we maintain real FIFO queues; for 'individual' we just track
  // headcount per station because exits are scheduled by per-entity events.
  const queues:     Record<string, string[]> = {};
  const population: Record<string, number>   = {};
  for (const s of Object.keys(successors)) {
    queues[s] = [];
    population[s] = 0;
  }
  population['main-sink'] = 0;

  if (logger) {
    logger.log({kind: 'sim_start', config: {
      kernel: `fel-${service}`,
      seed: opts.seed,
      service,
      tPhase1: config.phase1Days,
      tMax: config.horizonDays,
      sourceCap: config.sourceCap,
      arrivalsInterarrival: config.arrivalsInterarrival,
      residence: config.residence,
      probabilities: config.probabilities,
      edges: EDGES,
    }});
  }

  const transitions = new TransitionCounter();

  let absorbed = 0;
  let sourceCreated = 0;
  let nextEntityId = 0;
  let phase2 = false;
  const fel: FelEvent[] = [];
  const lastStation = new Map<string, string>();

  const arrive = (t: number, entityId: string, fromStation: string, toStation: string) => {
    if (toStation === 'main-sink') {
      transitions.record(fromStation, 'main-sink');
      logger?.log({kind: 'transition', t, entity: entityId, from: fromStation, to: 'main-sink'});
      absorbed++;
      return;
    }
    if (toStation.endsWith('-Decision')) {
      // Decisions are instantaneous: pick a successor immediately and recurse.
      const dest = drawSuccessor(successors, toStation);
      arrive(t, entityId, fromStation, dest);
      return;
    }
    transitions.record(fromStation, toStation);
    logger?.log({kind: 'transition', t, entity: entityId, from: fromStation, to: toStation});
    population[toStation]++;
    if (service === 'fifo') {
      queues[toStation].push(entityId);
      lastStation.set(entityId, toStation);
    } else {
      if (!config.residence[toStation]) {
        console.warn(`[fel-runner] no residence interval configured for station "${toStation}"; per-individual exit event cannot be scheduled.`);
      }
      const [a, b] = config.residence[toStation];
      insertEvent(fel, {
        time: t + drawUniform(a, b),
        kind: 'exit',
        station: toStation,
        entity: entityId,
      });
    }
  };

  // Trajectory sampling.
  const popSums = zeroCompartmentRecord();
  const peak = zeroCompartmentRecord();
  let nextSampleAt = sampleEvery;
  let samples = 0;

  const sampleAt = (t: number) => {
    const populations = compartmentPopulations(sid => population[sid] ?? 0);
    let totalAlive = 0;
    for (const c of COMPARTMENT_ORDER) {
      totalAlive += populations[c];
      popSums[c] += populations[c];
    }
    updatePeaks(peak, populations);
    samples++;
    logger?.log({
      kind: 'tick', t, populations, cumD: absorbed, alive: totalAlive,
      sourcesActive: !phase2,
    });
  };

  // Initial events.
  insertEvent(fel, {
    time: drawUniform(config.arrivalsInterarrival[0], config.arrivalsInterarrival[1]),
    kind: 'source',
  });
  if (service === 'fifo') {
    for (const s of Object.keys(successors)) {
      if (s === 'main-source') continue;
      const [a, b] = config.residence[s];
      insertEvent(fel, {time: drawUniform(a, b), kind: 'service', station: s});
    }
  }

  const startedAt = Date.now();

  while (true) {
    const e = fel.shift();
    if (!e) break;
    if (e.time > config.horizonDays) break;

    while (nextSampleAt <= Math.floor(e.time) && nextSampleAt <= config.horizonDays) {
      sampleAt(nextSampleAt);
      nextSampleAt += sampleEvery;
    }

    if (!phase2 && e.time >= config.phase1Days) {
      phase2 = true;
      console.debug(`[fel-runner] phase change to 'drain' at t=${Math.floor(e.time)}: arrivals stop (created ${sourceCreated}/${config.sourceCap} so far).`);
      logger?.log({kind: 'phase_change', t: Math.floor(e.time), phase: 'drain'});
    }

    if (e.kind === 'source') {
      if (sourceCreated < config.sourceCap && !phase2) {
        const id = `f${nextEntityId++}`;
        arrive(e.time, id, '__source__', 'S');
        sourceCreated++;
      }
      const [a, b] = config.arrivalsInterarrival;
      insertEvent(fel, {time: e.time + drawUniform(a, b), kind: 'source'});
    } else if (e.kind === 'service') {
      // FIFO single-server tick: pop head if any, route it, schedule next.
      const station = e.station!;
      const head = queues[station].shift();
      if (head !== undefined) {
        population[station]--;
        const dest = drawSuccessor(successors, station);
        arrive(e.time, head, station, dest);
      }
      const [a, b] = config.residence[station];
      insertEvent(fel, {time: e.time + drawUniform(a, b), kind: 'service', station});
    } else {
      // Per-individual exit event.
      population[e.station!]--;
      const dest = drawSuccessor(successors, e.station!);
      arrive(e.time, e.entity!, e.station!, dest);
    }
  }
  while (nextSampleAt <= config.horizonDays) {
    sampleAt(nextSampleAt);
    nextSampleAt += sampleEvery;
  }

  const elapsed = Date.now() - startedAt;
  const finalPopulations = compartmentPopulations(sid => population[sid] ?? 0);

  const {counts, splits} = transitions.tables();

  const timeAvg = averageRecord(popSums, samples);

  if (logger) {
    logger.log({
      kind: 'sim_end', t: config.horizonDays, elapsedMs: elapsed,
      totals: {created: sourceCreated, absorbed, finalPopulations, transitionCounts: counts},
    });
    void logger.close();
  }

  return {
    kernel: 'fel',
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
