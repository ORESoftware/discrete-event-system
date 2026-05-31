// RUST MIGRATION:
// - Target: src/des/runners/shared.rs.
// - Keep this as the shared runner utility module; TransitionCounter becomes a struct with impl methods.
// - Map TransitionCountMap to HashMap/BTreeMap depending on deterministic iteration needs, and serialize TransitionTables with serde.
// - Keep table builders and population aggregators as private/public pure functions unless lifted into PureTransform-style traits.
'use strict';

import {COMPARTMENT_GROUPS, COMPARTMENT_ORDER, SimConfig} from './types';

export const TRANSITION_MATRIX_ROWS = ['__source__', 'S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D'];
export const TRANSITION_MATRIX_COLS = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R', 'D', 'main-sink'];

export type TransitionCountMap = Map<string, Map<string, number>>;

export interface TransitionTables {
  counts: Record<string, Record<string, number>>;
  splits: Record<string, Record<string, number>>;
}

export class TransitionCounter {
  private readonly countsByFrom: TransitionCountMap = new Map();

  record(from: string, to: string): void {
    let row = this.countsByFrom.get(from);
    if (!row) {
      row = new Map();
      this.countsByFrom.set(from, row);
    }
    row.set(to, (row.get(to) ?? 0) + 1);
  }

  tables(
    rows: readonly string[] = TRANSITION_MATRIX_ROWS,
    cols: readonly string[] = TRANSITION_MATRIX_COLS,
  ): TransitionTables {
    return buildTransitionTables(this.countsByFrom, rows, cols);
  }
}

export function buildTransitionTables(
  transitionCount: TransitionCountMap,
  rows: readonly string[] = TRANSITION_MATRIX_ROWS,
  cols: readonly string[] = TRANSITION_MATRIX_COLS,
): TransitionTables {
  const counts: Record<string, Record<string, number>> = {};
  const splits: Record<string, Record<string, number>> = {};
  for (const r of rows) {
    counts[r] = {};
    splits[r] = {};
    const row = transitionCount.get(r);
    let total = 0;
    for (const c of cols) {
      const v = row?.get(c) ?? 0;
      counts[r][c] = v;
      total += v;
    }
    for (const c of cols) {
      splits[r][c] = total > 0 ? counts[r][c] / total : 0;
    }
  }
  return {counts, splits};
}

export function analyticalTransitionTables(
  probabilities: SimConfig['probabilities'],
): TransitionTables {
  const sparse: Record<string, Record<string, number>> = {
    '__source__': {S: 1},
    S: {E: 1},
    E: {'I-P': 1},
    'I-P': {
      'I-A': probabilities.asymptomaticShare,
      'I-S': 1 - probabilities.asymptomaticShare,
    },
    'I-A': {R: 1},
    'I-S': {
      R: 1 - probabilities.hospitalizationGivenSymptom,
      'I-H': probabilities.hospitalizationGivenSymptom,
    },
    'I-H': {
      R: 1 - probabilities.caseFatalityGivenHospital,
      D: probabilities.caseFatalityGivenHospital,
    },
    R: {S: 1},
    D: {'main-sink': 1},
  };

  const map: TransitionCountMap = new Map();
  for (const [from, row] of Object.entries(sparse)) {
    map.set(from, new Map(Object.entries(row)));
  }
  return buildTransitionTables(map);
}

export function zeroCompartmentRecord(): Record<string, number> {
  return Object.fromEntries(COMPARTMENT_ORDER.map(c => [c, 0]));
}

export function compartmentPopulations(
  populationOfStation: (stationId: string) => number,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of COMPARTMENT_ORDER) {
    out[c] = COMPARTMENT_GROUPS[c].reduce((a, sid) => a + populationOfStation(sid), 0);
  }
  return out;
}

export function averageRecord(
  sums: Record<string, number>,
  denominator: number,
): Record<string, number> {
  const out: Record<string, number> = {};
  const safeDenominator = Math.max(1, denominator);
  for (const c of COMPARTMENT_ORDER) out[c] = sums[c] / safeDenominator;
  return out;
}

export function updatePeaks(
  peak: Record<string, number>,
  values: Record<string, number>,
): void {
  for (const c of COMPARTMENT_ORDER) if (values[c] > peak[c]) peak[c] = values[c];
}

export function meanResidence(config: SimConfig, id: string): number {
  const [a, b] = config.residence[id];
  return (a + b) / 2;
}
