// RUST MIGRATION: Port file-for-file to `tests/factory_floor_track3t_test.rs` as integration coverage for warehouse scenarios and comparison animations.
// Test-port notes: translate scenario checks into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, and approximate-float helpers; keep animation fixtures deterministic.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/factory_floor_track3t_test.rs   (integration test crate)
// 1:1 file move. Tests the warehouse MDP/POMDP comparison + its animation
// scene, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc expect()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - state-space cardinality identities (n*n*2 + 1) -> assert_eq! on usize.
//   - observation-accuracy comparisons are float inequalities -> assert!; any
//     stochastic scenario sampling -> a seeded rand::Rng.
// =============================================================================

import {
  BASELINE_WAREHOUSE_SCENARIO,
  TRACK3T_WAREHOUSE_SCENARIO,
  buildWarehousePOMDP,
  defaultWarehouseLayout,
  runWarehouseComparison,
} from '../general/factory-floor-track3t';
import {buildWarehouseComparisonCharts, buildWarehouseComparisonFrame} from '../animation/scenes/warehouse-track3t-scene';

let pass = 0;
let fail = 0;

function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' - ' + detail : ''));
  cond ? pass++ : fail++;
}

console.log('\nfactory-floor-track3t - MDP/POMDP warehouse comparison');

const layout = defaultWarehouseLayout();
const destinationIndex = layout.stations.findIndex(s => s.id === 'shipping');
const baseModel = buildWarehousePOMDP(layout, BASELINE_WAREHOUSE_SCENARIO, destinationIndex);
const trackModel = buildWarehousePOMDP(layout, TRACK3T_WAREHOUSE_SCENARIO, destinationIndex);

expect('POMDP state space includes terminal state',
  baseModel.states.length === layout.stations.length * layout.stations.length * 2 + 1,
  `states=${baseModel.states.length}`);
expect('action set drives to every stationary entity',
  baseModel.actions.length === layout.stations.length,
  `actions=${baseModel.actions.length}, stations=${layout.stations.length}`);
expect('Track3t observation model is sharper than baseline',
  TRACK3T_WAREHOUSE_SCENARIO.locationAccuracy > BASELINE_WAREHOUSE_SCENARIO.locationAccuracy,
  `${TRACK3T_WAREHOUSE_SCENARIO.locationAccuracy} > ${BASELINE_WAREHOUSE_SCENARIO.locationAccuracy}`);
expect('both models use same hidden-state cardinality',
  trackModel.states.length === baseModel.states.length,
  `track=${trackModel.states.length}, base=${baseModel.states.length}`);

const result = runWarehouseComparison({jobs: 120, seed: 7, recordTrace: true});
const b = result.baseline.metrics;
const t = result.track3t.metrics;

expect('Track3t improves mean cycle time',
  t.meanCycleTime < b.meanCycleTime,
  `baseline=${b.meanCycleTime.toFixed(2)}, track3t=${t.meanCycleTime.toFixed(2)}`);
expect('Track3t improves throughput',
  t.throughputPerHour > b.throughputPerHour,
  `baseline=${b.throughputPerHour.toFixed(2)}, track3t=${t.throughputPerHour.toFixed(2)}`);
expect('Track3t reduces search misses',
  t.meanSearchMissesPerJob < b.meanSearchMissesPerJob,
  `baseline=${b.meanSearchMissesPerJob.toFixed(2)}, track3t=${t.meanSearchMissesPerJob.toFixed(2)}`);
expect('Track3t reduces shipping error rate',
  t.shippingErrorRate < b.shippingErrorRate,
  `baseline=${(b.shippingErrorRate * 100).toFixed(1)}%, track3t=${(t.shippingErrorRate * 100).toFixed(1)}%`);
expect('larger sample keeps nonzero residual Track3t errors',
  t.shippingErrors > 0 && result.deltas.errorReductionPct < 100,
  `track3t errors=${t.shippingErrors}, reduction=${result.deltas.errorReductionPct.toFixed(1)}%`);
expect('Track3t keeps a lower belief entropy',
  t.meanBeliefEntropy < b.meanBeliefEntropy,
  `baseline=${b.meanBeliefEntropy.toFixed(3)}, track3t=${t.meanBeliefEntropy.toFixed(3)}`);
expect('comparison trace records both scenarios',
  result.baseline.trace.length > 0 && result.track3t.trace.length > 0,
  `frames=${result.baseline.trace.length}/${result.track3t.trace.length}`);

const frame = buildWarehouseComparisonFrame(result, 0);
const charts = buildWarehouseComparisonCharts(result);
expect('animation frame has drawable shapes',
  frame.shapes.length > layout.stations.length * 2,
  `shapes=${frame.shapes.length}`);
expect('animation exposes comparison charts',
  charts.length === 2 && charts.every(c => c.series.length === 2),
  `charts=${charts.length}`);

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
