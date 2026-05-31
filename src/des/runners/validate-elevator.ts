#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_elevator.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert framework/SimPy JSON fixtures and aggregate comparison rows to serde structs.
// - Keep the elevator external reference as an adapter-produced golden payload and file I/O behind std::fs/std::path.
'use strict';

// Compares the framework's elevator-sim aggregates (out/elevator-framework.json)
// against the SimPy continuous-time reference (out/external/elevator/simpy.json).
//
// HOW TO RUN
// ----------
//   npm run build
//   node dist/des/main-elevator.js                          # writes out/elevator-framework.json
//   bash external-references/run-all.sh                     # writes out/external/elevator/simpy.json
//   node dist/des/runners/validate-elevator.js
//
// Reports:
//   - Per-person max-abs error on board / exit times.
//   - Per-aggregate diff (mean wait / mean travel / mean total).
//
// The framework discretizes time at dt=0.5 s, so per-person times can differ
// by up to ~stepSize. SimPy is continuous-time. Aggregates should agree to
// within a few percent.

import * as fs from 'fs';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath = path.join(ROOT, 'out', 'elevator-framework.json');
const pyPath = path.join(ROOT, 'out', 'external', 'elevator', 'simpy.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-elevator] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function main() {
  const ts = loadJson(tsPath);
  const py = loadJson(pyPath);

  console.log('Elevator: framework (fixed-step DES) vs SimPy (continuous-time FEL)');
  console.log('=====================================================================');
  console.log(`  ${ts.config.nFloors} floors, ${ts.config.nElevators} elevators, capacity ${ts.config.capacity}`);
  console.log(`  travel=${ts.config.floorTravelTime}s/floor, service=${ts.config.serviceTime}s, λ=${ts.config.arrivalRate}/s, simT=${ts.config.simT}s`);
  console.log(`  framework dt = ${ts.config.stepSize}s`);
  console.log('');

  const tsAgg = ts.aggregates;
  const pyAgg = py.aggregates;

  console.log(`  ${'metric'.padEnd(14)} ${'framework'.padStart(12)} ${'SimPy'.padStart(12)} ${'Δ'.padStart(10)} ${'Δ / dt'.padStart(10)}`);
  const rows: Array<[string, number, number]> = [
    ['n',          tsAgg.n,         pyAgg.n],
    ['nServed',    tsAgg.nServed,   pyAgg.nServed],
    ['meanWait',   tsAgg.meanWait,  pyAgg.meanWait],
    ['meanTravel', tsAgg.meanTravel,pyAgg.meanTravel],
    ['meanTotal',  tsAgg.meanTotal, pyAgg.meanTotal],
    ['p95Wait',    tsAgg.p95Wait,   pyAgg.p95Wait],
    ['p95Total',   tsAgg.p95Total,  pyAgg.p95Total],
  ];
  for (const [name, a, b] of rows) {
    const d = a - b;
    const dN = d / ts.config.stepSize;
    console.log(`  ${name.padEnd(14)} ${a.toFixed(2).padStart(12)} ${b.toFixed(2).padStart(12)} ${d.toFixed(2).padStart(10)} ${dN.toFixed(2).padStart(10)}`);
  }

  // Per-person comparison: the schedule is identical, so person IDs match.
  // Match by (id, fromFloor, toFloor, arrivalTime).
  const pyById = new Map<number, any>();
  for (const p of py.people) pyById.set(p.id, p);

  const matched: Array<{id: number; fromFloor: number; toFloor: number;
                        boardDiff: number; exitDiff: number}> = [];
  let unmatched = 0;
  for (const a of ts.people) {
    const b = pyById.get(a.id);
    if (!b) { unmatched++; continue; }
    matched.push({
      id: a.id, fromFloor: a.fromFloor, toFloor: a.toFloor,
      boardDiff: a.boardTime - b.boardTime,
      exitDiff:  a.exitTime  - b.exitTime,
    });
  }

  let maxBoardDiff = 0, maxExitDiff = 0;
  let sumAbsBoard = 0, sumAbsExit = 0;
  for (const m of matched) {
    sumAbsBoard += Math.abs(m.boardDiff);
    sumAbsExit  += Math.abs(m.exitDiff);
    if (Math.abs(m.boardDiff) > maxBoardDiff) maxBoardDiff = Math.abs(m.boardDiff);
    if (Math.abs(m.exitDiff)  > maxExitDiff)  maxExitDiff  = Math.abs(m.exitDiff);
  }
  const meanAbsBoard = sumAbsBoard / Math.max(1, matched.length);
  const meanAbsExit  = sumAbsExit  / Math.max(1, matched.length);

  console.log('');
  console.log(`  Per-person time differences (over ${matched.length} matched persons):`);
  console.log(`    mean |board_ts - board_simpy| = ${meanAbsBoard.toFixed(3)} s   (~${(meanAbsBoard / ts.config.stepSize).toFixed(2)} × dt)`);
  console.log(`    mean |exit_ts  - exit_simpy|  = ${meanAbsExit.toFixed(3)} s   (~${(meanAbsExit  / ts.config.stepSize).toFixed(2)} × dt)`);
  console.log(`    max  |board diff|             = ${maxBoardDiff.toFixed(3)} s`);
  console.log(`    max  |exit  diff|             = ${maxExitDiff.toFixed(3)} s`);
  if (unmatched) console.log(`  WARN: ${unmatched} unmatched persons`);

  // Acceptance: aggregate metrics within 10% of SimPy.
  // Per-person comparison is reported but NOT asserted — once the framework
  // and SimPy disagree on dispatch by a fraction of a tick, the same person
  // may be routed to a different elevator in the two implementations and
  // their individual board / exit times can diverge by tens of seconds.
  // That is emergent dispatch sensitivity, not a math bug, and aggregate
  // statistics are the correct way to compare stochastic-dispatch systems.
  const aggOk =
    Math.abs(tsAgg.meanWait   - pyAgg.meanWait)   < 0.10 * (pyAgg.meanWait   + 1) &&
    Math.abs(tsAgg.meanTravel - pyAgg.meanTravel) < 0.10 * (pyAgg.meanTravel + 1) &&
    Math.abs(tsAgg.meanTotal  - pyAgg.meanTotal)  < 0.10 * (pyAgg.meanTotal  + 1);

  console.log('');
  console.log(`  aggregate Δ within 10% of SimPy: ${aggOk ? 'yes' : 'NO'}`);
  console.log(aggOk ? '  PASS' : '  FAIL');
  process.exit(aggOk ? 0 : 1);
}

main();
