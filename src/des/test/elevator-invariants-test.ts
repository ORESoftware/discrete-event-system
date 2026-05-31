#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/elevator_invariants_test.rs   (integration test crate)
// 1:1 file move. Runs the elevator/floor/person model and checks invariants
// after every tick. Keep the rich doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc per-tick invariant checks + console.log  ->  #[test] fns using
//   assert!/assert_eq! (assert the invariant inside the tick loop).
//
// Conversion notes (file-specific):
//   - continuous currentFloor bounds (I3) and timestamp monotonicity (I5) are
//     float comparisons -> approx::assert_relative_eq!.
//   - conservation-of-people (I1) counts across floor queues / elevators / sink
//     -> assert_eq! on summed counts; floor queues mutated from inside
//     elevator.runTimeStep need shared ownership (Rc<RefCell<..>>) in Rust.
// =============================================================================

// =============================================================================
// Elevator system invariants test.
//
// Validates that the elevator + floor + person model is correct, by running
// many configurations under all three dispatch modes and checking — *after
// every single tick* — that a battery of invariants hold.
//
// Why this matters: the elevator is the most architecturally tricky
// simulation we have. Two stationary entity kinds (elevators and floors)
// share state (people boarding/deboarding mutate floor queues from inside
// elevator.runTimeStep). With three dispatch modes, multi-elevator
// coordination, and continuous `currentFloor` positions, there are lots of
// places a bug could go undetected if we only checked aggregate metrics.
//
// The seven invariants (each gets one assertion family):
//
//   I1  CONSERVATION OF PEOPLE.
//       Every person ever emitted by the source is in EXACTLY one of:
//         - a Floor.upQueue or Floor.downQueue (waiting)
//         - a Floor.exitedHere (just deboarded this tick)
//         - an Elevator.passengers (in transit)
//         - the Sink.collected (done)
//       No person is ever lost or duplicated.
//
//   I2  ELEVATOR CAPACITY.
//       For every elevator, |passengers| ≤ capacity.
//
//   I3  ELEVATOR POSITION BOUNDS.
//       For every elevator, 1 ≤ currentFloor ≤ nFloors. The position is
//       continuous, so we test the floats.
//
//   I4  STATE-MACHINE CONSISTENCY.
//         IDLE     → direction === 'idle'  AND  no passengers
//         MOVING   → direction ∈ {'up','down'}  AND  targetFloor !== currentFloor (mostly)
//         SERVING  → serviceRemaining ≥ 0
//
//   I5  TIMESTAMP MONOTONICITY (people in the sink).
//         boardTime ≥ arrivalTime  AND  exitTime ≥ boardTime
//
//   I6  FLOOR-QUEUE-DIRECTION CONSISTENCY.
//       Every person in floor.upQueue has toFloor > fromFloor,
//       every person in floor.downQueue has toFloor < fromFloor.
//
//   I7  COORDINATOR EXCLUSIVITY (coordinated modes only).
//       Two elevators MAY both be MOVING toward the same (target,
//       direction) ONLY when at least one of them has a passenger going
//       to that floor. The coordinator only excludes already-claimed
//       *floor calls* — passenger destinations are always honoured
//       because the passenger has already chosen their elevator. So
//       the invariant is: if E_a and E_b are both MOVING toward (F, d)
//       with F not a passenger destination of either, that's a bug.
//
// All invariants are checked after every tick. If any fails, the test
// reports the first failing tick and configuration and exits non-zero.
// =============================================================================

import {Building, ElevatorConfig, buildSchedule} from '../main-elevator';

interface InvariantError {
  tick: number;
  invariant: string;
  detail: string;
}

function checkInvariants(b: Building, t: number): InvariantError[] {
  const errs: InvariantError[] = [];
  const cfg = b.config;
  const dt = cfg.stepSize;
  const totalEmitted = b.totalEmitted();

  // Snapshot person locations.
  const idsInFloor = new Map<number, string>();
  for (const f of b.floors) {
    for (const p of f.upQueue)    idsInFloor.set(p.id, `Floor${f.floorNumber}.upQueue`);
    for (const p of f.downQueue)  idsInFloor.set(p.id, `Floor${f.floorNumber}.downQueue`);
    for (const p of f.exitedHere) idsInFloor.set(p.id, `Floor${f.floorNumber}.exitedHere`);
  }
  const idsInElevator = new Map<number, string>();
  for (const e of b.elevators) {
    for (const p of e.passengers) idsInElevator.set(p.id, e.id);
  }
  const idsInSink = new Set(b.sink.collected.map(p => p.id));

  // I1  conservation
  const allLocations = new Map<number, string[]>();
  const note = (id: number, loc: string) => {
    if (!allLocations.has(id)) allLocations.set(id, []);
    allLocations.get(id)!.push(loc);
  };
  for (const [id, loc] of idsInFloor)    note(id, loc);
  for (const [id, loc] of idsInElevator) note(id, loc);
  for (const id of idsInSink) note(id, 'sink');
  for (const [id, locs] of allLocations) {
    if (locs.length > 1) {
      errs.push({tick: t, invariant: 'I1-no-duplication',
        detail: `person ${id} in [${locs.join(', ')}]`});
    }
  }
  if (allLocations.size !== totalEmitted) {
    errs.push({tick: t, invariant: 'I1-conservation',
      detail: `${totalEmitted} emitted, ${allLocations.size} accounted for`});
  }

  // I2  capacity
  for (const e of b.elevators) {
    if (e.passengers.length > e.capacity) {
      errs.push({tick: t, invariant: 'I2-capacity',
        detail: `${e.id}: ${e.passengers.length} passengers > capacity ${e.capacity}`});
    }
  }

  // I3  position
  for (const e of b.elevators) {
    if (!(e.currentFloor >= 1 - 1e-9 && e.currentFloor <= cfg.nFloors + 1e-9)) {
      errs.push({tick: t, invariant: 'I3-position',
        detail: `${e.id}: currentFloor=${e.currentFloor} out of [1, ${cfg.nFloors}]`});
    }
    if (!Number.isFinite(e.currentFloor)) {
      errs.push({tick: t, invariant: 'I3-position-finite',
        detail: `${e.id}: currentFloor=${e.currentFloor}`});
    }
  }

  // I4  state-machine consistency
  for (const e of b.elevators) {
    if (e.state === 'IDLE') {
      if (e.direction !== 'idle') {
        errs.push({tick: t, invariant: 'I4-idle-direction',
          detail: `${e.id}: IDLE but direction=${e.direction}`});
      }
      if (e.passengers.length !== 0) {
        errs.push({tick: t, invariant: 'I4-idle-passengers',
          detail: `${e.id}: IDLE but has ${e.passengers.length} passengers`});
      }
    } else if (e.state === 'MOVING') {
      if (e.direction !== 'up' && e.direction !== 'down') {
        errs.push({tick: t, invariant: 'I4-moving-direction',
          detail: `${e.id}: MOVING but direction=${e.direction}`});
      }
    } else if (e.state === 'SERVING') {
      if (e.serviceRemaining < -dt - 1e-9) {  // allow current-tick epsilon
        errs.push({tick: t, invariant: 'I4-serving-time',
          detail: `${e.id}: SERVING but serviceRemaining=${e.serviceRemaining}`});
      }
    }
  }

  // I5  timestamp monotonicity
  for (const p of b.sink.collected) {
    if (p.boardTime < p.arrivalTime - 1e-9) {
      errs.push({tick: t, invariant: 'I5-board-after-arrival',
        detail: `person ${p.id}: arrival=${p.arrivalTime} > board=${p.boardTime}`});
    }
    if (p.exitTime < p.boardTime - 1e-9) {
      errs.push({tick: t, invariant: 'I5-exit-after-board',
        detail: `person ${p.id}: board=${p.boardTime} > exit=${p.exitTime}`});
    }
  }

  // I6  floor-queue direction consistency
  for (const f of b.floors) {
    for (const p of f.upQueue) {
      if (!(p.toFloor > p.fromFloor)) {
        errs.push({tick: t, invariant: 'I6-up-queue-direction',
          detail: `${f.id}: person ${p.id} (${p.fromFloor}→${p.toFloor}) in upQueue`});
      }
    }
    for (const p of f.downQueue) {
      if (!(p.toFloor < p.fromFloor)) {
        errs.push({tick: t, invariant: 'I6-down-queue-direction',
          detail: `${f.id}: person ${p.id} (${p.fromFloor}→${p.toFloor}) in downQueue`});
      }
    }
  }

  // I7  coordinator exclusivity (only meaningful in coordinated modes)
  // For each (targetFloor, direction), collect MOVING elevators committed to
  // it. The constraint is: if more than one elevator is committed, then at
  // least all-but-one must have a passenger destination at that floor (i.e.
  // their visit is justified by passenger delivery, not by the call).
  if (b.coordinator) {
    const targets = new Map<string, Array<{e: string; hasDestPassenger: boolean}>>();
    for (const e of b.elevators) {
      if (e.state === 'MOVING' && (e.direction === 'up' || e.direction === 'down')) {
        const key = `${e.targetFloor}-${e.direction}`;
        if (!targets.has(key)) targets.set(key, []);
        const hasDest = e.passengers.some(p => p.toFloor === e.targetFloor);
        targets.get(key)!.push({e: e.id, hasDestPassenger: hasDest});
      }
    }
    for (const [key, list] of targets) {
      if (list.length < 2) continue;
      // It's OK if N-1 of them have a destPassenger reason — only one
      // (at most) is allowed to be there for the call.
      const forCall = list.filter(x => !x.hasDestPassenger);
      if (forCall.length > 1) {
        errs.push({tick: t, invariant: 'I7-target-exclusivity',
          detail: `${list.length} elevators MOVING to ${key}, ${forCall.length} of them for the call (no passenger destination): ${forCall.map(x => x.e).join(', ')}`});
      }
    }
  }

  return errs;
}

function configsToTest(): ElevatorConfig[] {
  const out: ElevatorConfig[] = [];
  const dispatchModes: Array<ElevatorConfig['dispatchMode']> = ['uncoordinated', 'coordinated', 'coordinated-pickup'];
  // Vary nFloors, nElevators, capacity, lambda. Keep simT short so the test
  // finishes in reasonable time with per-tick checks.
  const sizes: Array<{nFloors: number; nElevators: number; capacity: number}> = [
    {nFloors: 4,  nElevators: 1, capacity: 4},   // single elevator, smallest building
    {nFloors: 4,  nElevators: 3, capacity: 8},   // canonical baseline
    {nFloors: 4,  nElevators: 4, capacity: 8},   // more elevators than floors-1
    {nFloors: 8,  nElevators: 2, capacity: 6},   // medium building
    {nFloors: 12, nElevators: 4, capacity: 10},  // tall + busy
  ];
  const lambdas = [0.05, 0.2, 0.4];
  let seed = 7;
  for (const sz of sizes) {
    for (const lambda of lambdas) {
      for (const mode of dispatchModes) {
        out.push({
          ...sz,
          arrivalRate: lambda,
          floorTravelTime: 4, serviceTime: 3,
          simT: 600, stepSize: 0.5, seed: seed++,
          dispatchMode: mode,
        });
      }
    }
  }
  return out;
}

function main() {
  const configs = configsToTest();
  console.log(`Elevator/Floor invariants — ${configs.length} configurations × per-tick checks`);
  console.log('===================================================================================');
  let totalTicks = 0;
  let totalConfigs = 0;
  let failedConfigs = 0;
  const allErrors: Array<{cfg: ElevatorConfig; err: InvariantError}> = [];
  for (const cfg of configs) {
    const schedule = buildSchedule(cfg);
    let configTicks = 0;
    let configErrors: InvariantError[] = [];
    const b = new Building(cfg, schedule);
    const N = Math.round(cfg.simT / cfg.stepSize);
    for (let t = 0; t < N; t++) {
      b.tickOnce(t);
      configTicks++;
      const errs = checkInvariants(b, t);
      if (errs.length > 0) {
        configErrors = errs;
        break;
      }
    }
    if (configErrors.length === 0) {
      // Drain remaining people too, with checks.
      for (let extra = 0; extra < N && !b.isComplete(schedule.length); extra++) {
        b.tickOnce(N + extra);
        configTicks++;
        const errs = checkInvariants(b, N + extra);
        if (errs.length > 0) { configErrors = errs; break; }
      }
    }
    totalTicks += configTicks;
    totalConfigs++;
    if (configErrors.length > 0) {
      failedConfigs++;
      for (const e of configErrors) allErrors.push({cfg, err: e});
      const mode = cfg.dispatchMode!.padEnd(20);
      console.log(`  FAIL  ${mode} F=${cfg.nFloors} E=${cfg.nElevators} C=${cfg.capacity} λ=${cfg.arrivalRate} seed=${cfg.seed}`);
      for (const e of configErrors.slice(0, 3)) {
        console.log(`        tick ${e.tick}: ${e.invariant}: ${e.detail}`);
      }
    }
  }
  console.log('');
  console.log(`  Total configurations: ${totalConfigs}`);
  console.log(`  Total ticks checked : ${totalTicks}`);
  console.log(`  Failed configurations: ${failedConfigs}`);
  console.log(`  Invariant violations : ${allErrors.length}`);
  console.log(failedConfigs === 0 ? '  PASS' : '  FAIL');
  process.exit(failedConfigs === 0 ? 0 : 1);
}

main();
