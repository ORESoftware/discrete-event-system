'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/output_routing_policy_test.rs   (integration test crate)
// 1:1 file move. Exercises OutputConnectionRouter against moving/processing
// entities, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc expect()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - bgn()/BigNumber values -> a decimal crate or f64 (pick ONE engine-wide).
//   - inline sink object literals -> small test structs implementing the sink
//     trait; `received: ProcessableMovingEntity[]` -> Vec<...>.
// =============================================================================

// =============================================================================
// test/output-routing-policy-test.ts -- competitive out-connection policies.
// =============================================================================

import {ProcessableMovingEntity} from '../entity-moving/moving';
import {PerIndividualProcessor} from '../entity-processing/per-individual-processor';
import {OutputConnectionRouter} from '../entity-routing/output-routing-policy';
import {bgn} from '../general/general';

let pass = 0, fail = 0;
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' -- ' + detail : ''));
  cond ? pass++ : fail++;
}

function sink(id: string, cap = Infinity) {
  const received: ProcessableMovingEntity<any>[] = [];
  return {
    id,
    received,
    acceptItem(): boolean { return received.length < cap; },
    takeItem(m: ProcessableMovingEntity<any>): void { received.push(m); },
  };
}

function makeEntity(): ProcessableMovingEntity<any> {
  return new ProcessableMovingEntity().init();
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 -- OutputConnectionRouter order policies');
// -----------------------------------------------------------------------------
{
  const rr = new OutputConnectionRouter<string>('round-robin');
  const conns = ['A', 'B', 'C'];
  const picks: string[] = [];
  for (let i = 0; i < 7; i++) {
    const ordered = rr.order(conns);
    const accepted = ordered[0];
    picks.push(accepted);
    rr.markAccepted(conns, accepted);
  }
  expect('round-robin rotates through declared order', picks.join('') === 'ABCABCA', picks.join(''));

  const ordered = new OutputConnectionRouter<string>('ordered');
  expect('ordered keeps declared order', ordered.order(conns).join('') === 'ABC');
  ordered.markAccepted(conns, conns[0]);
  expect('ordered does not rotate after accept', ordered.order(conns).join('') === 'ABC');
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 -- PerIndividualProcessor round-robin routing');
// -----------------------------------------------------------------------------
{
  const p = new PerIndividualProcessor<any, any>('P-rr', {
    drawDuration: () => 0,
    outputRouting: 'round-robin',
  });
  const a = sink('A');
  const b = sink('B');
  const c = sink('C');
  p.addOutConnection(a as any);
  p.addOutConnection(b as any);
  p.addOutConnection(c as any);
  for (let i = 0; i < 6; i++) p.takeItem(makeEntity());
  p.runTimeStep(bgn(1));
  expect('round-robin sends two to each of three declared sinks',
    a.received.length === 2 && b.received.length === 2 && c.received.length === 2,
    `A=${a.received.length}, B=${b.received.length}, C=${c.received.length}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 -- PerIndividualProcessor ordered priority routing');
// -----------------------------------------------------------------------------
{
  const p = new PerIndividualProcessor<any, any>('P-ordered', {
    drawDuration: () => 0,
    outputRouting: 'ordered',
  });
  const a = sink('A');
  const b = sink('B');
  const c = sink('C');
  p.addOutConnection(a as any);
  p.addOutConnection(b as any);
  p.addOutConnection(c as any);
  for (let i = 0; i < 6; i++) p.takeItem(makeEntity());
  p.runTimeStep(bgn(1));
  expect('ordered keeps first declared sink as priority when it accepts',
    a.received.length === 6 && b.received.length === 0 && c.received.length === 0,
    `A=${a.received.length}, B=${b.received.length}, C=${c.received.length}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 -- Round-robin skips full acceptors');
// -----------------------------------------------------------------------------
{
  const p = new PerIndividualProcessor<any, any>('P-cap', {
    drawDuration: () => 0,
    outputRouting: 'round-robin',
  });
  const a = sink('A', 1);
  const b = sink('B');
  p.addOutConnection(a as any);
  p.addOutConnection(b as any);
  for (let i = 0; i < 4; i++) p.takeItem(makeEntity());
  p.runTimeStep(bgn(1));
  expect('round-robin tries declared order but uses next accepting sink',
    a.received.length === 1 && b.received.length === 3,
    `A=${a.received.length}, B=${b.received.length}`);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
