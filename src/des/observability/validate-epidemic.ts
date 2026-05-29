#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// Offline validator for the epidemic simulation.
//
// Reads the JSONL event stream produced by main-epidemic-improved.ts and
// asserts a battery of invariants. Prints a pass/fail report. Exits non-zero
// if any invariant fails so this can be wired into CI later.
//
// Invariants checked
//   I1. Topology adherence:  every observed transition (from -> to) must be a
//                            valid graph edge after flattening decision nodes
//                            (I-P-Decision is collapsed: I-P -> I-A and I-P
//                            -> I-S become first-class edges, etc.).
//   I2. Per-entity continuity: for each entity, the n-th transition's `from`
//                            equals the (n-1)-th transition's `to` (or
//                            __source__ for the first transition).
//   I3. Branching probability: for each decision-flattened branch point
//                            (I-P, I-S, I-H), observed split is within a
//                            binomial 99% confidence interval of the configured
//                            probability.
//   I4. Mass conservation:   total transitions out of __source__ equals
//                            createdCount; transitions into main-sink equals
//                            destroyedCount; and at any tick, sum of populations
//                            + cumD equals number of entities ever created
//                            minus those still in transit (== 0 in our model).
//   I5. Per-cycle death rate: P(D | one S-visit) close to the closed-form
//                            P(I-S|I-P) * P(I-H|I-S) * P(D|I-H).
//   I6. Tick monotonicity:   tick events are emitted in non-decreasing t and
//                            transitions happen at t consistent with surrounding
//                            ticks.
// =============================================================================

import * as path from 'path';
import {readEvents} from './logger';

interface Failure {
  invariant: string;
  detail: string;
  context?: any;
}

const failures: Failure[] = [];
const fail = (invariant: string, detail: string, context?: any) =>
  failures.push({invariant, detail, context});

const fmt = (n: number, digits = 4) =>
  Number.isFinite(n) ? n.toFixed(digits) : String(n);

// ---------------------------------------------------------------------------
// Inverse normal CDF for 99% two-sided CI: z_{0.995} = 2.5758
// ---------------------------------------------------------------------------
const Z_99 = 2.5758;

function binomialCI99(p_hat: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const margin = Z_99 * Math.sqrt(p_hat * (1 - p_hat) / n);
  return [Math.max(0, p_hat - margin), Math.min(1, p_hat + margin)];
}

// ---------------------------------------------------------------------------
function main() {
  const eventLogPath = process.argv[2] ?? path.resolve(__dirname, '..', '..', '..', 'out', 'epidemic-events.jsonl');
  const events = readEvents(eventLogPath);

  const start = events.find(e => e.kind === 'sim_start');
  const end   = events.find(e => e.kind === 'sim_end');
  if (!start) throw new Error('no sim_start event found');
  if (!end)   throw new Error('no sim_end event found');

  const transitions = events.filter(e => e.kind === 'transition');
  const ticks       = events.filter(e => e.kind === 'tick');

  console.log('================================================================');
  console.log(`epidemic event log validator`);
  console.log(`  file:        ${eventLogPath}`);
  console.log(`  events:      ${events.length}  (transitions=${transitions.length}, ticks=${ticks.length})`);
  console.log(`  sim wall ms: ${end.elapsedMs}`);
  console.log('================================================================');
  console.log('');

  // ----- I1: topology adherence -------------------------------------------
  // Build the decision-flattened edge set.  In the live simulator, decision
  // nodes are transparent in the transition log: the wrapper records
  // (last-processor -> next-processor), skipping the decision.  So the
  // valid transitions are: every direct (non-decision -> non-decision) edge,
  // plus, for every (X -> decision) edge, all (X -> Y) where (decision -> Y)
  // is in the graph.
  const edges: Array<[string, string]> = start.config.edges;
  const isDecision = (s: string) => /Decision$/.test(s);

  const decisionTargets = new Map<string, string[]>();
  for (const [a, b] of edges) {
    if (isDecision(a)) {
      const list = decisionTargets.get(a) ?? [];
      list.push(b);
      decisionTargets.set(a, list);
    }
  }
  const flat = new Set<string>();
  flat.add('__source__->S');  // main-source -> S becomes __source__ -> S in the log
  for (const [a, b] of edges) {
    if (isDecision(a)) continue; // handled via the predecessor below
    if (a === 'main-source') continue; // already added as __source__->S
    if (isDecision(b)) {
      for (const tgt of decisionTargets.get(b) ?? []) {
        flat.add(`${a}->${tgt}`);
      }
    } else {
      flat.add(`${a}->${b}`);
    }
  }

  let i1Bad = 0;
  for (const t of transitions) {
    if (!flat.has(`${t.from}->${t.to}`)) {
      i1Bad++;
      if (i1Bad <= 3) {
        fail('I1 topology', `unexpected transition ${t.from} -> ${t.to}`, {
          t: t.t, entity: t.entity,
        });
      }
    }
  }
  if (i1Bad > 3) {
    fail('I1 topology', `... and ${i1Bad - 3} more invalid transitions`);
  }
  console.log(`I1 topology adherence:        ${i1Bad === 0 ? 'PASS' : 'FAIL'}  (${i1Bad}/${transitions.length} bad)`);

  // ----- I2: per-entity continuity ----------------------------------------
  const lastSeen = new Map<string, string>();
  let i2Bad = 0;
  for (const t of transitions) {
    const expected = lastSeen.get(t.entity) ?? '__source__';
    if (t.from !== expected) {
      i2Bad++;
      if (i2Bad <= 3) {
        fail('I2 continuity', `entity ${t.entity} jumped from ${expected} to ${t.from} at t=${t.t}`);
      }
    }
    lastSeen.set(t.entity, t.to);
  }
  console.log(`I2 per-entity continuity:     ${i2Bad === 0 ? 'PASS' : 'FAIL'}  (${i2Bad} jump(s))`);

  // ----- I3: branching probability ----------------------------------------
  // For each "from" station that has multiple distinct "to" stations,
  // check the empirical split is in the 99% binomial CI of the configured
  // probability.  This indirectly tests the decision nodes.
  const transitionsByFrom = new Map<string, Map<string, number>>();
  for (const t of transitions) {
    let row = transitionsByFrom.get(t.from);
    if (!row) {
      row = new Map();
      transitionsByFrom.set(t.from, row);
    }
    row.set(t.to, (row.get(t.to) ?? 0) + 1);
  }

  const expectedSplits: Record<string, Record<string, number>> = {
    'I-P': {
      'I-A': start.config.probabilities.asymptomaticShare,
      'I-S': 1 - start.config.probabilities.asymptomaticShare,
    },
    'I-S': {
      'R':   1 - start.config.probabilities.hospitalizationGivenSymptom,
      'I-H': start.config.probabilities.hospitalizationGivenSymptom,
    },
    'I-H': {
      'R': 1 - start.config.probabilities.caseFatalityGivenHospital,
      'D': start.config.probabilities.caseFatalityGivenHospital,
    },
  };

  let i3Bad = 0;
  console.log('I3 branching probabilities:');
  for (const [from, exp] of Object.entries(expectedSplits)) {
    const row = transitionsByFrom.get(from);
    const total = row ? [...row.values()].reduce((a, b) => a + b, 0) : 0;
    for (const [to, p_expected] of Object.entries(exp)) {
      const observed = row?.get(to) ?? 0;
      const p_hat = total > 0 ? observed / total : 0;
      const [lo, hi] = binomialCI99(p_hat, total);
      const within = p_expected >= lo && p_expected <= hi;
      if (!within) i3Bad++;
      console.log(
        `  ${from.padEnd(3)} -> ${to.padEnd(3)}  expected=${fmt(p_expected, 3)}  observed=${fmt(p_hat, 3)}  ` +
        `99%CI=[${fmt(lo, 3)}, ${fmt(hi, 3)}]  n=${total}  ${within ? 'PASS' : 'FAIL'}`,
      );
      if (!within) {
        fail('I3 branching', `${from} -> ${to} expected ${p_expected.toFixed(3)} not in 99% CI [${lo.toFixed(3)}, ${hi.toFixed(3)}]`);
      }
    }
  }

  // ----- I4: mass conservation --------------------------------------------
  const sourceOut = transitions.filter(t => t.from === '__source__').length;
  const sinkIn    = transitions.filter(t => t.to === 'main-sink').length;

  const created  = end.totals.created;
  const absorbed = end.totals.absorbed;
  const finalPop = end.totals.finalPopulations;
  const totalAlive = Object.values(finalPop as Record<string, number>).reduce((a, b) => a + b, 0);

  const i4SourceOK = sourceOut === created;
  const i4SinkOK   = sinkIn   === absorbed;
  const i4MassOK   = created === absorbed + totalAlive;

  console.log(`I4 mass conservation:`);
  console.log(`  source emissions in log: ${sourceOut}     createdCount: ${created}     ${i4SourceOK ? 'PASS' : 'FAIL'}`);
  console.log(`  sink absorptions in log: ${sinkIn}        destroyedCount: ${absorbed}  ${i4SinkOK ? 'PASS' : 'FAIL'}`);
  console.log(`  created == absorbed + alive: ${created} == ${absorbed} + ${totalAlive}  ${i4MassOK ? 'PASS' : 'FAIL'}`);
  if (!i4SourceOK) fail('I4 mass', `source emissions ${sourceOut} != createdCount ${created}`);
  if (!i4SinkOK)   fail('I4 mass', `sink absorptions ${sinkIn} != destroyedCount ${absorbed}`);
  if (!i4MassOK)   fail('I4 mass', `created ${created} != absorbed ${absorbed} + alive ${totalAlive}`);

  // ----- I5: per-cycle death rate -----------------------------------------
  // Each "S-visit" is one (X -> S) transition (sourced or via R). The
  // probability of dying during *that* infection cycle (before getting
  // back to S via R) is q = p(IS|IP) * p(IH|IS) * p(D|IH).
  // Empirically, q == #deaths-since-most-recent-S-visit / #S-visits.
  const sVisits = transitions.filter(t => t.to === 'S').length;
  const deaths  = transitions.filter(t => t.to === 'D').length;
  const q_theoretical =
    (1 - start.config.probabilities.asymptomaticShare) *
    start.config.probabilities.hospitalizationGivenSymptom *
    start.config.probabilities.caseFatalityGivenHospital;
  const q_observed = sVisits > 0 ? deaths / sVisits : 0;
  const [qLo, qHi] = binomialCI99(q_observed, sVisits);
  const i5OK = q_theoretical >= qLo && q_theoretical <= qHi;
  console.log(`I5 per-cycle death rate:`);
  console.log(`  q_theoretical = ${fmt(q_theoretical, 4)}`);
  console.log(`  q_observed    = ${fmt(q_observed, 4)}  99%CI=[${fmt(qLo, 4)}, ${fmt(qHi, 4)}]  S-visits=${sVisits}  deaths=${deaths}  ${i5OK ? 'PASS' : 'FAIL'}`);
  if (!i5OK) fail('I5 death rate', `theoretical ${q_theoretical.toFixed(4)} not in 99% CI`);

  // ----- I6: tick monotonicity --------------------------------------------
  let i6Bad = 0;
  let prevT = -1;
  for (const e of ticks) {
    if (e.t <= prevT) i6Bad++;
    prevT = e.t;
  }
  // also: every transition's t must lie within [first-tick.t-1, last-tick.t+1]
  const tFirst = ticks[0]?.t ?? 0;
  const tLast  = ticks[ticks.length - 1]?.t ?? 0;
  let oob = 0;
  for (const t of transitions) {
    if (t.t < tFirst - 1 || t.t > tLast + 1) oob++;
  }
  console.log(`I6 tick monotonicity:         ${i6Bad === 0 ? 'PASS' : 'FAIL'}  (${i6Bad} non-monotonic, ${oob} out-of-band transitions)`);
  if (i6Bad > 0) fail('I6 tick monotonicity', `${i6Bad} non-monotonic ticks`);
  if (oob > 0)    fail('I6 tick monotonicity', `${oob} out-of-band transitions`);

  // ----- Summary -----------------------------------------------------------
  console.log('');
  console.log('================================================================');
  if (failures.length === 0) {
    console.log('All invariants PASSED.');
  } else {
    console.log(`${failures.length} invariant failure(s):`);
    for (const f of failures) {
      console.log(`  - [${f.invariant}] ${f.detail}`);
    }
  }
  console.log('================================================================');

  process.exit(failures.length === 0 ? 0 : 1);
}

main();
