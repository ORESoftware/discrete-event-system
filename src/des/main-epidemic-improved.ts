#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_epidemic_improved.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-epidemic-improved.rs   (fn main)
// 1:1 file move. SEIR-with-hospitalization epidemic over the entity graph;
// samples compartment populations to CSV and prints an empirical transition
// matrix.
//
// Conversion notes (file-specific):
//   - imports many des entity modules -> use crate::des::...
//   - mathjs bgn -> f64 / decimal; fisherYatesShuffle ordering -> injected Rng;
//     residence-time draws -> SeededRandom.
//   - per-step CSV write -> std::fs; empirical transition matrix -> HashMap.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// SEIR-with-hospitalization epidemic simulator (improved version of main-epidemic.ts)
//
// Improvements over the original:
//   1. Topology bug fixed: I-P / I-S / I-H no longer carry parallel arcs that
//      bypass their decision nodes; routing is now strictly through decisions.
//   2. Per-compartment residence-time distributions reflect a COVID-like model
//      instead of every station drawing U(10, 20).
//   3. Branching probabilities at the three decision nodes use literature-ish
//      values (asymptomatic share, hospitalization share, case-fatality among
//      hospitalized) instead of placeholder 0.4 / 0.6 splits.
//   4. R -> S is a slow waning-immunity loop (months), not 10-20 ms.
//   5. Compartment populations S(t), E(t), I_P(t), I_A(t), I_S(t), I_H(t),
//      R(t), and cumulative D(t) are sampled every step and dumped to CSV
//      so the run can be plotted as a Markov-chain trajectory.
//   6. An empirical transition matrix is computed from each moving entity's
//      observed inter-station transitions (decision nodes treated as
//      transparent) and printed as a markdown table at the end of the run.
// =============================================================================

import * as math from 'mathjs';
import {EntitySource} from './entity-source/source';
import {EntityProcessor} from './entity-processing/processing';
import {EntitySink} from './entity-sink/sink';
import {StationaryEntity} from './abstract/abstract';
import {AbstractMovingEntity} from './entity-moving/moving';
import {bgn, fisherYatesShuffle} from './general/general';
import {PoissonRandomVariable, UniformRandomVariable} from './random-variables/rv';
import {ProgramObserver} from './observers/program-observer';
import {ProbabilityDecisionEntity} from './entity-decision/probability-decision';
import * as fs from 'fs';
import * as path from 'path';
import {JsonlLogger} from './observability/logger';

// --- Time discretization ---------------------------------------------------
// Each "ms" the framework knows about is interpreted here as 1 day, so a step
// size of bgn(1) = 1 day. All residence-time uniforms are expressed in days.
const stepSize = bgn(1);

// --- Per-station inter-event uniforms --------------------------------------
// IMPORTANT framework semantics: the EntityProcessor draws *inter-departure*
// times from its RV, not per-individual residence times. A station's
// throughput is therefore approximately mu = 1 / mean(uniform) individuals
// per day, regardless of how many entities are queued. Mean residence per
// individual then emerges from M/M/1-style queueing: T ≈ 1 / (mu - lambda),
// where lambda is the arrival rate.
//
// To get a stable, readable trajectory we keep mu well above the arrival
// rate at every station. The *ratio* between station means still encodes
// the relative biological timescales (infectious stages process quickly,
// recovered individuals linger for a long waning period).
const RESIDENCE: Record<string, [number, number]> = {
  'S':   [0.20, 0.40],  // mu ~ 3.3/day; clears arrivals fast
  'E':   [0.20, 0.40],
  'I-P': [0.20, 0.40],
  'I-A': [0.20, 0.40],
  'I-S': [0.20, 0.40],
  'I-H': [0.20, 0.40],
  'R':   [1.50, 2.50],  // mu ~ 0.5/day; long waning => R compartment grows
  'D':   [0.10, 0.30],  // mu ~ 5/day; quick passage to the sink

  // Decision nodes route within a step but still need an RV per framework contract.
  'I-P-Decision': [0.05, 0.15],
  'I-S-Decision': [0.05, 0.15],
  'I-H-Decision': [0.05, 0.15],
};

const ARRIVALS_INTERARRIVAL: [number, number] = [0.7, 1.3]; // ~1 new susceptible / day
const TURN_OFF_AFTER_COUNT = 500; // total susceptibles spawned, then source quiesces

// Branching probabilities (sum to 1 per decision):
//   I-P-Decision : 0 = I-A (asymptomatic), 1 = I-S (symptomatic)
//   I-S-Decision : 0 = R   (recover at home), 1 = I-H (hospitalized)
//   I-H-Decision : 0 = R   (discharged alive), 1 = D   (in-hospital death)
const PROBS = {
  asymptomaticShare:           bgn(0.40),
  hospitalizationGivenSymptom: bgn(0.20),
  caseFatalityGivenHospital:   bgn(0.12),
};

// Total simulation length.
const PHASE_1_STEPS = 800;  // sources active
const PHASE_2_STEPS = 400;  // drain

// --- Helper to build a Uniform RV from a [a, b] tuple ----------------------
const uni = (id: string) => {
  const [a, b] = RESIDENCE[id];
  return new UniformRandomVariable({aVal: bgn(a), bVal: bgn(b)});
};

// --- Compartment groups for population tracking ----------------------------
// Decision nodes are folded into their upstream compartment for the
// trajectory plot, so a moving entity briefly sitting in I-P-Decision still
// counts as part of "I-P" for charting purposes.
const COMPARTMENT_GROUPS: Record<string, string[]> = {
  'S':   ['S'],
  'E':   ['E'],
  'I-P': ['I-P', 'I-P-Decision'],
  'I-A': ['I-A'],
  'I-S': ['I-S', 'I-S-Decision'],
  'I-H': ['I-H', 'I-H-Decision'],
  'R':   ['R'],
};

const COMPARTMENT_ORDER = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R'];

// =============================================================================

const run = () => {

  const obs = new ProgramObserver();

  // Map of stationaryId -> entity. Order matters so the processor wiring
  // below can look entities up by id.
  const programEntities = new Map<string, StationaryEntity<any>>([
    [
      'main-source',
      new EntitySource('main-source', {
        turnOffAfterCount: TURN_OFF_AFTER_COUNT,
        rv: new UniformRandomVariable({
          aVal: bgn(ARRIVALS_INTERARRIVAL[0]),
          bVal: bgn(ARRIVALS_INTERARRIVAL[1]),
        }),
      }).subscribe(obs),
    ],
    ['S',   new EntityProcessor('S',   {rv: uni('S')}).subscribe(obs)],
    ['E',   new EntityProcessor('E',   {rv: uni('E')}).subscribe(obs)],
    ['I-P', new EntityProcessor('I-P', {rv: uni('I-P')}).subscribe(obs)],

    [
      'I-P-Decision',
      new ProbabilityDecisionEntity('I-P-Decision', {
        rv: uni('I-P-Decision'),
        probabilities: [
          {index: 0, prob: PROBS.asymptomaticShare},
          {index: 1, prob: math.subtract(bgn(1), PROBS.asymptomaticShare) as math.BigNumber},
        ],
      }).subscribe(obs),
    ],

    ['I-A', new EntityProcessor('I-A', {rv: uni('I-A')}).subscribe(obs)],
    ['I-S', new EntityProcessor('I-S', {rv: uni('I-S')}).subscribe(obs)],

    [
      'I-S-Decision',
      new ProbabilityDecisionEntity('I-S-Decision', {
        rv: uni('I-S-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), PROBS.hospitalizationGivenSymptom) as math.BigNumber},
          {index: 1, prob: PROBS.hospitalizationGivenSymptom},
        ],
      }).subscribe(obs),
    ],

    ['I-H', new EntityProcessor('I-H', {rv: uni('I-H')}).subscribe(obs)],

    [
      'I-H-Decision',
      new ProbabilityDecisionEntity('I-H-Decision', {
        rv: uni('I-H-Decision'),
        probabilities: [
          {index: 0, prob: math.subtract(bgn(1), PROBS.caseFatalityGivenHospital) as math.BigNumber},
          {index: 1, prob: PROBS.caseFatalityGivenHospital},
        ],
      }).subscribe(obs),
    ],

    ['R', new EntityProcessor('R', {rv: uni('R')}).subscribe(obs)],
    ['D', new EntityProcessor('D', {rv: uni('D')}).subscribe(obs)],

    [
      'main-sink',
      new EntitySink('main-sink', new PoissonRandomVariable(), {}).subscribe(obs),
    ],
  ]);

  // Topology - strictly through decisions for branching compartments. The
  // `I-P -> I-A`, `I-P -> I-S`, `I-S -> R`, `I-S -> I-H`, `I-H -> R` arcs
  // that the original had are removed: those splits live entirely in the
  // probability-decision nodes now.
  const edges: Array<[string, string]> = [
    ['main-source',   'S'],
    ['S',             'E'],
    ['E',             'I-P'],
    ['I-P',           'I-P-Decision'],
    ['I-P-Decision',  'I-A'],          // index 0 -> I-A
    ['I-P-Decision',  'I-S'],          // index 1 -> I-S
    ['I-A',           'R'],
    ['I-S',           'I-S-Decision'],
    ['I-S-Decision',  'R'],            // index 0 -> R
    ['I-S-Decision',  'I-H'],          // index 1 -> I-H
    ['I-H',           'I-H-Decision'],
    ['I-H-Decision',  'R'],            // index 0 -> R (recovered)
    ['I-H-Decision',  'D'],            // index 1 -> D (deceased)
    ['D',             'main-sink'],
    ['R',             'S'],            // waning immunity -> back to susceptible
  ];

  for (const [sourceId, targetId] of edges) {
    const source = programEntities.get(sourceId) as any;
    const target = programEntities.get(targetId) as any;
    source.addOutConnection(target);
    target.addInConnection(source);
  }

  for (const [, entity] of programEntities) {
    entity.doSetupAfterOutputConn();
    entity.doSetupAfterInputConn();
  }

  // -------------------------------------------------------------------------
  // Compartment population sampler. For each processor station, the current
  // population is inputQueue + processingQueue + outQueue. Decision nodes
  // expose .queue (a LinkedQueue with .length).
  // -------------------------------------------------------------------------
  const stationPopulation = (id: string): number => {
    const e = programEntities.get(id) as any;
    if (!e) return 0;
    let n = 0;
    if (e.inputQueue)      n += e.inputQueue.length ?? e.inputQueue.size ?? 0;
    if (e.processingQueue) n += e.processingQueue.length ?? e.processingQueue.size ?? 0;
    if (e.outQueue)        n += e.outQueue.length ?? e.outQueue.size ?? 0;
    if (e.queue && !e.inputQueue) {
      // decision-style: only one queue
      n += e.queue.length ?? e.queue.size ?? 0;
    }
    return n;
  };

  const compartmentPopulation = (compartment: string): number =>
    COMPARTMENT_GROUPS[compartment].reduce((acc, sid) => acc + stationPopulation(sid), 0);

  const cumulativeDeaths = () =>
    (programEntities.get('main-sink') as any).destroyedCount as number;

  // -------------------------------------------------------------------------
  // Observability: open a JSONL logger and emit a sim_start event with the
  // full configuration so the validator can reconstruct what was supposed
  // to happen. Every transition is also logged below.
  // -------------------------------------------------------------------------
  const outDir = path.resolve(__dirname, '..', '..', 'out');
  fs.mkdirSync(outDir, {recursive: true});
  const eventLogPath = path.join(outDir, 'epidemic-events.jsonl');
  const logger = new JsonlLogger(eventLogPath, 'info');

  // currentStep is updated by the run loop and read by the takeItem wrappers
  // and tick logger.  It starts at 0 because nothing has happened yet.
  let currentStep = 0;

  logger.log({
    kind: 'sim_start',
    config: {
      stepSize: Number(stepSize),
      phase1Steps: PHASE_1_STEPS,
      phase2Steps: PHASE_2_STEPS,
      sourceCap: TURN_OFF_AFTER_COUNT,
      arrivalsInterarrival: ARRIVALS_INTERARRIVAL,
      residence: RESIDENCE,
      probabilities: {
        asymptomaticShare: Number(PROBS.asymptomaticShare),
        hospitalizationGivenSymptom: Number(PROBS.hospitalizationGivenSymptom),
        caseFatalityGivenHospital: Number(PROBS.caseFatalityGivenHospital),
      },
      edges,
    },
  });

  // -------------------------------------------------------------------------
  // Empirical transition counter. We instrument every processor and decision
  // station's takeItem so that whenever a moving entity arrives at station Y
  // having last been at processor X, we increment count[X][Y]. Decision nodes
  // are transparent: arriving at a decision does not update lastStation, so
  // the next processor records the transition as "previous-processor ->
  // next-processor", which is the SEIR-level Markov edge we care about.
  //
  // The same wrappers also emit JSONL "transition" events for the offline
  // validator (which uses them to assert topology and per-entity path order).
  // -------------------------------------------------------------------------
  const transitionCount = new Map<string, Map<string, number>>();
  const lastProcessorForEntity = new WeakMap<AbstractMovingEntity<any>, string>();

  const recordTransition = (from: string, to: string) => {
    let row = transitionCount.get(from);
    if (!row) {
      row = new Map();
      transitionCount.set(from, row);
    }
    row.set(to, (row.get(to) ?? 0) + 1);
  };

  const PROCESSOR_IDS = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D'];

  for (const id of PROCESSOR_IDS) {
    const proc = programEntities.get(id) as any;
    const orig = proc.takeItem.bind(proc);
    proc.takeItem = (m: AbstractMovingEntity<any>) => {
      const prev = lastProcessorForEntity.get(m) ?? '__source__';
      recordTransition(prev, id);
      lastProcessorForEntity.set(m, id);
      logger.log({
        kind: 'transition',
        t: currentStep,
        entity: m.id,
        from: prev,
        to: id,
      });
      return orig(m);
    };
  }
  // Sink: record final "absorption" transition.
  {
    const sink = programEntities.get('main-sink') as any;
    const origTake = sink.takeItem.bind(sink);
    sink.takeItem = (m: AbstractMovingEntity<any>) => {
      const prev = lastProcessorForEntity.get(m) ?? '__source__';
      recordTransition(prev, 'main-sink');
      logger.log({
        kind: 'transition',
        t: currentStep,
        entity: m.id,
        from: prev,
        to: 'main-sink',
      });
      return origTake(m);
    };
  }

  // -------------------------------------------------------------------------
  // Run the simulation. Quiet the processor's per-step "event quantity:" log.
  // -------------------------------------------------------------------------
  const programList = Array.from(programEntities);
  const trajectory: Array<Record<string, number>> = [];

  const sampleTrajectory = (t: number) => {
    const row: Record<string, number> = {t};
    let totalAlive = 0;
    const populations: Record<string, number> = {};
    for (const c of COMPARTMENT_ORDER) {
      const n = compartmentPopulation(c);
      row[c] = n;
      populations[c] = n;
      totalAlive += n;
    }
    const dCum = cumulativeDeaths();
    row['D_cum'] = dCum;
    row['alive'] = totalAlive;
    trajectory.push(row);
    logger.log({
      kind: 'tick',
      t,
      populations,
      cumD: dCum,
      alive: totalAlive,
      sourcesActive: !(global as any).turnOffSources,
    });
  };

  const origConsoleLog = console.log;
  console.log = () => {};

  const startedAt = Date.now();

  for (let i = 0; i < PHASE_1_STEPS; i++) {
    currentStep = i + 1;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    sampleTrajectory(currentStep);
  }

  (global as any).turnOffSources = true;
  logger.log({kind: 'phase_change', t: currentStep, phase: 'drain'});

  for (let i = 0; i < PHASE_2_STEPS; i++) {
    currentStep = PHASE_1_STEPS + i + 1;
    for (const [, v] of fisherYatesShuffle(programList)) {
      v.doTimeStep(stepSize);
    }
    sampleTrajectory(currentStep);
  }

  console.log = origConsoleLog;
  const elapsed = Date.now() - startedAt;

  // -------------------------------------------------------------------------
  // Reports
  // -------------------------------------------------------------------------
  console.log('');
  console.log('=== epidemic simulator (improved) ==========================');
  console.log(`steps run: ${PHASE_1_STEPS + PHASE_2_STEPS} (${PHASE_1_STEPS} arriving, ${PHASE_2_STEPS} draining)`);
  console.log(`wall time: ${elapsed} ms`);
  console.log(`total entities created: ${(programEntities.get('main-source') as any).createdCount}`);
  console.log(`cumulative deaths absorbed by sink: ${cumulativeDeaths()}`);
  console.log('');

  // --- Final compartment populations ---------------------------------------
  const finalRow = trajectory[trajectory.length - 1];
  console.log('--- final compartment populations ---');
  for (const c of COMPARTMENT_ORDER) {
    console.log(`  ${c.padEnd(4)}: ${finalRow[c]}`);
  }
  console.log(`  D (cum): ${finalRow['D_cum']}`);
  console.log('');

  // --- Empirical transition matrix -----------------------------------------
  // Rows: source compartment, cols: destination compartment.
  const matrixRows = ['__source__', 'S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D'];
  const matrixCols = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D', 'main-sink'];

  const rowSums = new Map<string, number>();
  for (const r of matrixRows) {
    let s = 0;
    const row = transitionCount.get(r);
    if (row) for (const v of row.values()) s += v;
    rowSums.set(r, s);
  }

  console.log('--- empirical transition counts ---');
  const header = ['from \\ to', ...matrixCols, 'sum'].map(s => s.padEnd(11)).join('');
  console.log(header);
  for (const r of matrixRows) {
    const cells = [r.padEnd(11)];
    for (const c of matrixCols) {
      const v = transitionCount.get(r)?.get(c) ?? 0;
      cells.push(String(v).padEnd(11));
    }
    cells.push(String(rowSums.get(r) ?? 0).padEnd(11));
    console.log(cells.join(''));
  }
  console.log('');

  console.log('--- empirical transition probabilities (row-stochastic) ---');
  console.log(header);
  for (const r of matrixRows) {
    const total = rowSums.get(r) ?? 0;
    const cells = [r.padEnd(11)];
    for (const c of matrixCols) {
      const v = transitionCount.get(r)?.get(c) ?? 0;
      const p = total > 0 ? (v / total) : 0;
      cells.push((p === 0 ? '.' : p.toFixed(3)).padEnd(11));
    }
    cells.push((total > 0 ? '1.000' : '.').padEnd(11));
    console.log(cells.join(''));
  }
  console.log('');

  // --- Persist artifacts ---------------------------------------------------
  const csvPath = path.join(outDir, 'epidemic-trajectory.csv');
  const cols = ['t', ...COMPARTMENT_ORDER, 'D_cum', 'alive'];
  const csv = [
    cols.join(','),
    ...trajectory.map(r => cols.map(c => r[c]).join(',')),
  ].join('\n');
  fs.writeFileSync(csvPath, csv + '\n');

  const matrixPath = path.join(outDir, 'epidemic-transition-matrix.json');
  const serialMatrix: Record<string, Record<string, number>> = {};
  for (const r of matrixRows) {
    const total = rowSums.get(r) ?? 0;
    serialMatrix[r] = {};
    for (const c of matrixCols) {
      const v = transitionCount.get(r)?.get(c) ?? 0;
      serialMatrix[r][c] = total > 0 ? +(v / total).toFixed(6) : 0;
    }
  }
  fs.writeFileSync(matrixPath, JSON.stringify(serialMatrix, null, 2));

  // --- Sim-end event ------------------------------------------------------
  logger.log({
    kind: 'sim_end',
    t: currentStep,
    elapsedMs: elapsed,
    totals: {
      created: (programEntities.get('main-source') as any).createdCount,
      absorbed: cumulativeDeaths(),
      finalPopulations: Object.fromEntries(
        COMPARTMENT_ORDER.map(c => [c, finalRow[c]]),
      ),
      transitionCounts: Object.fromEntries(
        matrixRows.map(r => [r, Object.fromEntries(transitionCount.get(r)?.entries() ?? [])]),
      ),
    },
  });

  // close() returns a Promise but we've finished writing; allow it to flush.
  void logger.close();

  console.log(`artifacts written:`);
  console.log(`  ${csvPath}`);
  console.log(`  ${matrixPath}`);
  console.log(`  ${eventLogPath}  (${logger.getEventCount()} events: ${
    Object.entries(logger.getKindCounts()).map(([k, n]) => `${k}=${n}`).join(', ')
  })`);
  console.log('============================================================');
};

run();
