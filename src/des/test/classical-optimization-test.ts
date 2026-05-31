'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/classical_optimization_test.rs   (integration test crate)
// 1:1 file move. Tests classic optimization station-graph models (assignment,
// VRP, flow/job shop, QP) via the des-registry, so it is an integration test.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) relative float comparison -> approx::assert_relative_eq!.
//   - throws(fn) -> assert on Result::Err / #[should_panic].
// =============================================================================

// =============================================================================
// Tests for classic optimization station-graph models.
// =============================================================================

import {getModel, runFromSpec} from '../general/des-registry';
import {
  runAuctionAssignment,
  runFlowShopNEH,
  runHungarianAssignment,
  runJobShopDispatch,
  runQPCoordinateDescent,
  runQPProjectedGradient,
  runVRPNearestNeighbor,
  runVRPSavings,
} from '../general/classical-optimization-models';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function close(a: number, b: number, tol = 1e-5): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

async function main(): Promise<void> {
console.log('\n-- qp-projected-gradient --');
{
  const r = runQPProjectedGradient({});
  check('QP x[0] near analytic optimum', close(r.x[0], 10 / 7), `x0=${r.x[0]}`);
  check('QP x[1] near analytic optimum', close(r.x[1], 16 / 7), `x1=${r.x[1]}`);
  check('QP gradient norm is small', r.gradientNorm < 1e-6, `norm=${r.gradientNorm}`);
  check('QP uses movable state tokens', r.topology.movables.includes('QPStateToken'));
  check('QP source rejects invalid initial state before update station',
    throws(() => runQPProjectedGradient({x0: [11, 0]})));
}

console.log('\n-- qp-coordinate-descent --');
{
  const r = runQPCoordinateDescent({});
  check('coordinate QP x[0] near analytic optimum', close(r.x[0], 10 / 7), `x0=${r.x[0]}`);
  check('coordinate QP x[1] near analytic optimum', close(r.x[1], 16 / 7), `x1=${r.x[1]}`);
  check('coordinate QP gradient norm is small', r.gradientNorm < 1e-6, `norm=${r.gradientNorm}`);
  check('coordinate QP uses movable state tokens', r.topology.movables.includes('QPStateToken'));
}

console.log('\n-- hungarian-assignment --');
{
  const r = runHungarianAssignment({});
  check('assignment objective is optimal for default matrix', r.objective === 9, `z=${r.objective}`);
  check('assignment covers every row', r.assignment.length === 3);
  check('assignment columns are unique', new Set(r.assignment).size === r.assignment.length);
  check('assignment uses matrix tokens', r.topology.movables.includes('AssignmentMatrixToken'));
}

console.log('\n-- auction-assignment --');
{
  const r = runAuctionAssignment({});
  check('auction objective is optimal for default matrix', r.objective === 9, `z=${r.objective}`);
  check('auction assignment covers every row', r.assignment.length === 3);
  check('auction columns are unique', new Set(r.assignment).size === r.assignment.length);
  check('auction uses movable state tokens', r.topology.movables.includes('AssignmentAuctionStateToken'));
}

console.log('\n-- vrp-savings --');
{
  const r = runVRPSavings({});
  check('VRP produces two default routes', r.routes.length === 2, `routes=${r.routes.length}`);
  check('VRP route loads respect capacity', r.routes.every(route => route.load <= 5), r.routes.map(route => String(route.load)).join(','));
  check('VRP has positive distance', r.totalDistance > 0, `distance=${r.totalDistance}`);
  check('VRP uses savings tokens', r.topology.movables.includes('VRPSavingsToken'));
}

console.log('\n-- vrp-nearest-neighbor --');
{
  const r = runVRPNearestNeighbor({});
  check('nearest-neighbor VRP produces two default routes', r.routes.length === 2, `routes=${r.routes.length}`);
  check('nearest-neighbor route loads respect capacity', r.routes.every(route => route.load <= 5), r.routes.map(route => String(route.load)).join(','));
  check('nearest-neighbor VRP has positive distance', r.totalDistance > 0, `distance=${r.totalDistance}`);
  check('nearest-neighbor VRP uses problem/result tokens', ['VRPProblemToken', 'VRPResultToken'].every(t => r.topology.movables.includes(t)));
}

console.log('\n-- job-shop-dispatch --');
{
  const r = runJobShopDispatch({rule: 'spt'});
  check('job shop schedules all six operations', r.schedule.length === 6, `ops=${r.schedule.length}`);
  check('job shop makespan matches deterministic fixture', r.makespan === 10, `makespan=${r.makespan}`);
  check('job shop respects job precedence', respectsPrecedence(r.schedule));
  check('job shop has no machine overlap', noMachineOverlap(r.schedule));
  check('job shop uses job tokens', r.topology.movables.includes('JobToken'));
}

console.log('\n-- flow-shop-neh --');
{
  const r = runFlowShopNEH({});
  check('flow shop schedules all default operations', r.schedule.length === 12, `ops=${r.schedule.length}`);
  check('flow shop NEH deterministic makespan', r.makespan === 16, `makespan=${r.makespan}`);
  check('flow shop respects job precedence', respectsPrecedence(r.schedule));
  check('flow shop has no machine overlap', noMachineOverlap(r.schedule));
  check('flow shop uses sequence tokens', r.topology.movables.includes('FlowSequenceToken'));
}

console.log('\n-- registry smoke --');
{
  for (const id of [
    'qp-projected-gradient',
    'qp-coordinate-descent',
    'hungarian-assignment',
    'auction-assignment',
    'vrp-savings',
    'vrp-nearest-neighbor',
    'job-shop-dispatch',
    'flow-shop-neh',
  ]) {
    const reg = getModel(id);
    check(`registry has ${id}`, reg.id === id);
  }
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'hungarian-assignment',
    parameters: {},
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('runFromSpec executes assignment model', summary.modelId === 'hungarian-assignment');
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`classical-optimization-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
}

function respectsPrecedence(schedule: readonly {jobId: string; opIndex: number; start: number; finish: number}[]): boolean {
  for (const op of schedule) {
    if (op.opIndex === 0) continue;
    const prev = schedule.find(other => other.jobId === op.jobId && other.opIndex === op.opIndex - 1);
    if (!prev || prev.finish > op.start) return false;
  }
  return true;
}

function noMachineOverlap(schedule: readonly {machine: string; start: number; finish: number}[]): boolean {
  const machines = new Set(schedule.map(op => op.machine));
  for (const machine of machines) {
    const ops = schedule.filter(op => op.machine === machine).sort((a, b) => a.start - b.start);
    for (let i = 1; i < ops.length; i++) if (ops[i - 1].finish > ops[i].start) return false;
  }
  return true;
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
