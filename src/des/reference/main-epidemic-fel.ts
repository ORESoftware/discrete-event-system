#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/des/reference/main_epidemic_fel.rs.
// - Keep file-for-file as the golden FEL reference; FelEvent and emitted summaries become serde comparison structs.
// - Replace Math.random with an injected RNG trait and keep event-log/CSV/matrix output behind std::fs plus serde_json/csv writers.
// - Treat this module as an external adapter-compatible reference implementation, with Result-returning run/main boundaries.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/reference/main-epidemic-fel.rs  (module des::reference::main_epidemic_fel)
// 1:1 file move. CLI BINARY: classical Future-Event-List reference SEIR simulation.
//
// Declarations → Rust:
//   const RESIDENCE / ARRIVALS_INTERARRIVAL / PROBS / T_PHASE_1 / T_MAX /
//         COMPARTMENT_GROUPS / COMPARTMENT_ORDER / SUCCESSORS  -> module consts/statics
//   type FelEvent                                              -> struct FelEvent { time: f64, station }
//   const fel + insertEvent/popEvent                           -> a `BinaryHeap`/owned Vec PQ
//   const drawUniform / drawSuccessor (fns)                    -> free fns
//   const run (fn)                                             -> `fn main()` (writes artifacts)
//
// Conversion notes (file-specific):
//   - ENTRY SCRIPT (`run()` at EOF; writes CSV/JSON/JSONL files) -> a Rust binary.
//   - DETERMINISM: `Math.random()` in drawUniform/drawSuccessor -> injected
//     `RandomSource` (shared/capabilities) so it matches the framework's seeded runs.
//   - The FEL is a sorted-array PQ maintained by `splice` -> `BinaryHeap<Reverse<FelEvent>>`
//     (min-heap by `time`); `FelEvent` needs `Ord`/`PartialOrd` on `time`.
//   - MODULE-LEVEL MUTABLE `fel` + closures that mutate it -> own the PQ inside the
//     run fn/struct; Rust has no mutable module globals (no `let fel = []` at top level).
//   - `Record<string,[number,number]>` (RESIDENCE) / `Record<string, Array<{prob,to}>>`
//     (SUCCESSORS) -> `HashMap<&str,(f64,f64)>` / `HashMap<&str, Vec<Branch>>` or `match`.
//   - `Map<string, Map<string, number>>` transitionCount -> nested `HashMap`.
//   - `Date.now()` wall timing -> `std::time::Instant`/`Clock`.
//   - `fs.mkdirSync`/`writeFileSync` + `JSON.stringify`/`logger.log({...})` ->
//     `std::fs` + `serde_json` (typed events or `serde_json::Value`).
//   - `Object.fromEntries/entries/keys` + `+(x).toFixed(6)` -> map iteration + f64 rounding.
//   - `succs.length===1` fast path and cumulative-prob sampling -> straight port.
// =============================================================================

// =============================================================================
// FEL (Future-Event-List) reference implementation of the same epidemic model.
//
// Why this exists
// ---------------
// The framework in src/des/ runs every station's local logic on every fixed
// time step and never maintains a global schedule. A *classical* DES kernel
// (SimPy, Arena, SSJ) instead maintains a priority queue of future events,
// pops the next-soonest one, advances the simulation clock to that time, and
// runs only the affected handler. Different scheduling algorithm, same model.
//
// If our framework is correctly implementing the model, a classical FEL
// implementation should produce the same statistics modulo Monte-Carlo
// noise. This file is the reference both as a sanity check and as something
// the validator can compare against.
//
// Modeled semantics
// -----------------
// Same SEIR-with-hospitalization model used by main-epidemic-improved.ts:
//   - source emits at U(0.7, 1.3) day inter-arrivals, capped at 500
//   - every station has its own service-completion clock with U(a,b) inter-event
//   - branching probabilities live in I-P-Decision / I-S-Decision / I-H-Decision
//   - R waning loops back to S
//   - D drains to a sink
//
// The framework's semantic in particular: a station's RV controls a global
// service clock for that station, NOT a per-individual residence draw.
// That is exactly what we replicate here: each station has one clock; when
// it fires, the head of that station's queue (if any) is routed downstream
// and the clock is rescheduled.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {JsonlLogger} from '../observability/logger';
import {RandomSource, DEFAULT_RANDOM} from '../shared/capabilities';

// --- Config (intentionally identical to main-epidemic-improved.ts) ----------
const RESIDENCE: Record<string, [number, number]> = {
  'S':   [0.20, 0.40], 'E':   [0.20, 0.40], 'I-P': [0.20, 0.40],
  'I-A': [0.20, 0.40], 'I-S': [0.20, 0.40], 'I-H': [0.20, 0.40],
  'R':   [1.50, 2.50], 'D':   [0.10, 0.30],
  'I-P-Decision': [0.05, 0.15],
  'I-S-Decision': [0.05, 0.15],
  'I-H-Decision': [0.05, 0.15],
};
const ARRIVALS_INTERARRIVAL: [number, number] = [0.7, 1.3];
const TURN_OFF_AFTER_COUNT = 500;

const PROBS = {
  asymptomaticShare:           0.40,
  hospitalizationGivenSymptom: 0.20,
  caseFatalityGivenHospital:   0.12,
};

const T_PHASE_1 = 800;   // sources active
const T_MAX     = 1200;  // total horizon

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

// Graph as a successor map. Probabilities sum to 1 per row.
const SUCCESSORS: Record<string, Array<{prob: number, to: string}>> = {
  'main-source': [{prob: 1, to: 'S'}],
  'S':           [{prob: 1, to: 'E'}],
  'E':           [{prob: 1, to: 'I-P'}],
  'I-P':         [{prob: 1, to: 'I-P-Decision'}],
  'I-P-Decision': [
    {prob:     PROBS.asymptomaticShare, to: 'I-A'},
    {prob: 1 - PROBS.asymptomaticShare, to: 'I-S'},
  ],
  'I-A':         [{prob: 1, to: 'R'}],
  'I-S':         [{prob: 1, to: 'I-S-Decision'}],
  'I-S-Decision': [
    {prob: 1 - PROBS.hospitalizationGivenSymptom, to: 'R'},
    {prob:     PROBS.hospitalizationGivenSymptom, to: 'I-H'},
  ],
  'I-H':         [{prob: 1, to: 'I-H-Decision'}],
  'I-H-Decision': [
    {prob: 1 - PROBS.caseFatalityGivenHospital, to: 'R'},
    {prob:     PROBS.caseFatalityGivenHospital, to: 'D'},
  ],
  'R':           [{prob: 1, to: 'S'}],
  'D':           [{prob: 1, to: 'main-sink'}],
};

// --- Random helpers ---------------------------------------------------------
const drawUniform = (a: number, b: number, rng: RandomSource = DEFAULT_RANDOM) => a + rng.nextFloat() * (b - a);

const drawSuccessor = (from: string, rng: RandomSource = DEFAULT_RANDOM): string => {
  const succs = SUCCESSORS[from];
  if (succs.length === 1) return succs[0].to;
  const r = rng.nextFloat();
  let cum = 0;
  for (const s of succs) {
    cum += s.prob;
    if (r < cum) return s.to;
  }
  return succs[succs.length - 1].to;
};

// --- Future event list (sorted-array PQ; fine for ~12 stations) ------------
type FelEvent = {time: number, station: string};
const fel: FelEvent[] = [];

const insertEvent = (e: FelEvent) => {
  // Linear scan is O(n) but n <= number-of-stations (==12) here so no need
  // for a heap. Keeping the implementation textbook-readable matters more.
  let i = 0;
  while (i < fel.length && fel[i].time <= e.time) i++;
  fel.splice(i, 0, e);
};
const popEvent = (): FelEvent | undefined => fel.shift();

// --- Run --------------------------------------------------------------------
const run = () => {
  const outDir = path.resolve(__dirname, '..', '..', '..', 'out');
  fs.mkdirSync(outDir, {recursive: true});
  const eventLogPath = path.join(outDir, 'epidemic-events-fel.jsonl');
  const logger = new JsonlLogger(eventLogPath, 'info');

  logger.log({
    kind: 'sim_start',
    config: {
      kernel: 'fel-reference',
      tPhase1: T_PHASE_1,
      tMax: T_MAX,
      sourceCap: TURN_OFF_AFTER_COUNT,
      arrivalsInterarrival: ARRIVALS_INTERARRIVAL,
      residence: RESIDENCE,
      probabilities: PROBS,
      // Provide the same edges shape so the validator can be pointed at this log too.
      edges: [
        ['main-source', 'S'], ['S', 'E'], ['E', 'I-P'],
        ['I-P', 'I-P-Decision'],
        ['I-P-Decision', 'I-A'], ['I-P-Decision', 'I-S'],
        ['I-A', 'R'],
        ['I-S', 'I-S-Decision'],
        ['I-S-Decision', 'R'], ['I-S-Decision', 'I-H'],
        ['I-H', 'I-H-Decision'],
        ['I-H-Decision', 'R'], ['I-H-Decision', 'D'],
        ['D', 'main-sink'],
        ['R', 'S'],
      ],
    },
  });

  // Per-station FIFO queue of entity IDs.
  const queues: Record<string, string[]> = {};
  for (const s of Object.keys(SUCCESSORS)) queues[s] = [];
  queues['main-sink'] = []; // not used but keeps lookups uniform

  // Last non-decision station an entity was at, for transition logging that
  // mirrors the live simulator's "decisions are transparent" convention.
  const lastStation = new Map<string, string>();

  let sourceCreated = 0;
  let absorbed = 0;
  let nextEntityId = 0;
  let phase2 = false;

  // Trajectory snapshots are taken at integer times t = 1..T_MAX, just like
  // the framework's per-step sampling.
  const trajectory: Array<Record<string, number>> = [];
  let nextSampleAt = 1;

  const sampleAt = (t: number) => {
    const row: Record<string, number> = {t};
    let totalAlive = 0;
    const populations: Record<string, number> = {};
    for (const c of COMPARTMENT_ORDER) {
      const n = COMPARTMENT_GROUPS[c]
        .reduce((acc, sid) => acc + (queues[sid]?.length ?? 0), 0);
      row[c] = n;
      populations[c] = n;
      totalAlive += n;
    }
    row['D_cum'] = absorbed;
    row['alive'] = totalAlive;
    trajectory.push(row);
    logger.log({
      kind: 'tick',
      t,
      populations,
      cumD: absorbed,
      alive: totalAlive,
      sourcesActive: !phase2,
    });
  };

  const transitionCount = new Map<string, Map<string, number>>();
  const recordTransition = (from: string, to: string) => {
    let row = transitionCount.get(from);
    if (!row) { row = new Map(); transitionCount.set(from, row); }
    row.set(to, (row.get(to) ?? 0) + 1);
  };

  const arrive = (t: number, entityId: string, station: string) => {
    if (station === 'main-sink') {
      const prev = lastStation.get(entityId) ?? '__source__';
      recordTransition(prev, 'main-sink');
      logger.log({kind: 'transition', t, entity: entityId, from: prev, to: 'main-sink'});
      absorbed++;
      return;
    }
    if (!station.endsWith('-Decision')) {
      const prev = lastStation.get(entityId) ?? '__source__';
      recordTransition(prev, station);
      logger.log({kind: 'transition', t, entity: entityId, from: prev, to: station});
      lastStation.set(entityId, station);
    }
    queues[station].push(entityId);
  };

  // Schedule one initial service event per station (and the source). Each
  // station gets its first inter-event draw at simulation time = draw.
  for (const s of Object.keys(SUCCESSORS)) {
    const [a, b] = s === 'main-source' ? ARRIVALS_INTERARRIVAL : RESIDENCE[s];
    insertEvent({time: drawUniform(a, b), station: s});
  }

  const startedAt = Date.now();

  while (true) {
    const e = popEvent();
    if (!e) break;
    if (e.time > T_MAX) break;

    // Sample any integer ticks we passed since the last event.
    while (nextSampleAt <= Math.floor(e.time) && nextSampleAt <= T_MAX) {
      sampleAt(nextSampleAt);
      nextSampleAt++;
    }

    // Phase change: turn off source past T_PHASE_1.
    if (!phase2 && e.time >= T_PHASE_1) {
      phase2 = true;
      logger.log({kind: 'phase_change', t: Math.floor(e.time), phase: 'drain'});
    }

    if (e.station === 'main-source') {
      if (sourceCreated < TURN_OFF_AFTER_COUNT && !phase2) {
        const id = `f${nextEntityId++}`;
        arrive(e.time, id, 'S');
        sourceCreated++;
      }
      const [a, b] = ARRIVALS_INTERARRIVAL;
      insertEvent({time: e.time + drawUniform(a, b), station: 'main-source'});
    } else {
      // Service one entity at this station's queue head, if any.
      const head = queues[e.station].shift();
      if (head !== undefined) {
        const dest = drawSuccessor(e.station);
        arrive(e.time, head, dest);
      }
      const [a, b] = RESIDENCE[e.station];
      insertEvent({time: e.time + drawUniform(a, b), station: e.station});
    }
  }

  // Flush any remaining ticks up to T_MAX.
  while (nextSampleAt <= T_MAX) {
    sampleAt(nextSampleAt);
    nextSampleAt++;
  }

  const elapsed = Date.now() - startedAt;

  // ---- Reports -----------------------------------------------------------
  console.log('');
  console.log('=== epidemic simulator (FEL reference) =====================');
  console.log(`horizon: ${T_MAX} (${T_PHASE_1} arriving, ${T_MAX - T_PHASE_1} draining)`);
  console.log(`wall time: ${elapsed} ms`);
  console.log(`total entities created: ${sourceCreated}`);
  console.log(`cumulative deaths absorbed by sink: ${absorbed}`);
  console.log('');

  const finalRow = trajectory[trajectory.length - 1];
  console.log('--- final compartment populations ---');
  for (const c of COMPARTMENT_ORDER) {
    console.log(`  ${c.padEnd(4)}: ${finalRow[c]}`);
  }
  console.log(`  D (cum): ${finalRow['D_cum']}`);
  console.log('');

  // ---- Persist artifacts --------------------------------------------------
  const csvPath = path.join(outDir, 'epidemic-trajectory-fel.csv');
  const cols = ['t', ...COMPARTMENT_ORDER, 'D_cum', 'alive'];
  fs.writeFileSync(csvPath, [
    cols.join(','),
    ...trajectory.map(r => cols.map(c => r[c]).join(',')),
  ].join('\n') + '\n');

  const matrixPath = path.join(outDir, 'epidemic-transition-matrix-fel.json');
  const matrixRows = ['__source__', 'S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D'];
  const matrixCols = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D', 'main-sink'];
  const serialMatrix: Record<string, Record<string, number>> = {};
  for (const r of matrixRows) {
    const row = transitionCount.get(r);
    let total = 0;
    if (row) for (const v of row.values()) total += v;
    serialMatrix[r] = {};
    for (const c of matrixCols) {
      const v = row?.get(c) ?? 0;
      serialMatrix[r][c] = total > 0 ? +(v / total).toFixed(6) : 0;
    }
  }
  fs.writeFileSync(matrixPath, JSON.stringify(serialMatrix, null, 2));

  logger.log({
    kind: 'sim_end',
    t: T_MAX,
    elapsedMs: elapsed,
    totals: {
      created: sourceCreated,
      absorbed,
      finalPopulations: Object.fromEntries(
        COMPARTMENT_ORDER.map(c => [c, finalRow[c]]),
      ),
      transitionCounts: Object.fromEntries(
        matrixRows.map(r => [r, Object.fromEntries(transitionCount.get(r)?.entries() ?? [])]),
      ),
    },
  });
  void logger.close();

  console.log(`artifacts written:`);
  console.log(`  ${csvPath}`);
  console.log(`  ${matrixPath}`);
  console.log(`  ${eventLogPath}`);
  console.log('============================================================');
};

run();
