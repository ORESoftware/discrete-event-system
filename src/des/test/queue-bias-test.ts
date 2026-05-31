#!/usr/bin/env ts-node
// RUST MIGRATION: Port file-for-file to `tests/queue_bias_test.rs` if the linked-queue behavior remains an external compatibility target.
// Test-port notes: translate bias/correctness checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, approximate helpers where needed, and deterministic seeds.

'use strict';

// =============================================================================
// Bias / correctness tests for @oresoftware/linked-queue.
//
// The framework's EntityProcessor and PerIndividualProcessor both use this
// queue (or a wrapper around it) to hold pending entities. If the queue had
// any FIFO bias, every result we computed would inherit it. This test
// verifies the queue has none of the failure modes that could affect
// simulation results:
//
//   T1  Pure FIFO:           N enqueues then N dequeues yield items in order.
//   T2  Mixed FIFO:          Random interleaved enqueue/dequeue still yields
//                            items in original enqueue order.
//   T3  Remove preserves:    remove(k) does not perturb the relative order
//                            of any other items.
//   T4  Random key uniform:  getRandomKey() is uniform over the keys
//                            (chi-square test, alpha = 0.001).
//   T5  No leaks:            after equal enqueue+dequeue, internal Map is
//                            empty, head/tail null, size = 0.
//   T6  Head/tail invariants:after every operation, head.before == null,
//                            tail.after == null, size matches lookup.size,
//                            head/tail are mutually reachable through `after`.
//   T7  Iterator order:      for-of yields items in insertion order; after
//                            dequeue/remove, only surviving items appear.
//   T8  addToFront LIFO:     addToFront then dequeue yields LIFO front.
//
// All tests are deterministic given a seeded PRNG so failures are
// reproducible.
// =============================================================================

import {LinkedQueue} from '@oresoftware/linked-queue';
import {mulberry32}  from '../general/prng';

let pass = 0;
let fail = 0;
const failures: string[] = [];

function check(label: string, ok: boolean, detail = '') {
  if (ok) {
    pass++;
    console.log(`  PASS  ${label}`);
  } else {
    fail++;
    failures.push(`${label}${detail ? '\n        ' + detail : ''}`);
    console.log(`  FAIL  ${label}${detail ? '\n        ' + detail : ''}`);
  }
}

// -----------------------------------------------------------------------------
// Helper: walk the doubly-linked list and verify all the invariants the
// queue *itself* claims to hold.
// -----------------------------------------------------------------------------
function structuralInvariants(q: any): {ok: boolean, why: string} {
  const head = q.head;
  const tail = q.tail;
  const size = q.lookup.size;

  if (size === 0) {
    if (head !== null) return {ok: false, why: 'size=0 but head!=null'};
    if (tail !== null) return {ok: false, why: 'size=0 but tail!=null'};
    return {ok: true, why: ''};
  }

  if (!head) return {ok: false, why: `size=${size} but head=null`};
  if (!tail) return {ok: false, why: `size=${size} but tail=null`};
  if (head.before !== null && head.before !== undefined) {
    return {ok: false, why: 'head.before is not null/undefined'};
  }
  if (tail.after !== null && tail.after !== undefined) {
    return {ok: false, why: 'tail.after is not null/undefined'};
  }

  // walk forward
  let count = 0;
  let v = head;
  while (v) {
    count++;
    if (count > size + 5) {
      return {ok: false, why: 'forward walk longer than size + 5'};
    }
    if (!q.lookup.get(v.key)) {
      return {ok: false, why: `node key ${String(v.key)} missing from lookup`};
    }
    if (v.after && v.after.before !== v) {
      return {ok: false, why: `bad backlink at key ${String(v.key)}`};
    }
    v = v.after || null;
  }
  if (count !== size) {
    return {ok: false, why: `forward walk count=${count} != size=${size}`};
  }

  // walk backward
  let bcount = 0;
  v = tail;
  while (v) {
    bcount++;
    if (bcount > size + 5) {
      return {ok: false, why: 'reverse walk longer than size + 5'};
    }
    v = v.before || null;
  }
  if (bcount !== size) {
    return {ok: false, why: `reverse walk count=${bcount} != size=${size}`};
  }

  return {ok: true, why: ''};
}

// -----------------------------------------------------------------------------
// T1: pure FIFO with N enqueue then N dequeue
// -----------------------------------------------------------------------------
function t1_pureFifo(N: number) {
  console.log(`T1 pure FIFO (N=${N})`);
  const q: any = new LinkedQueue();
  for (let i = 0; i < N; i++) q.enqueue(`k${i}`, i);

  const inv = structuralInvariants(q);
  check('  invariants after N enqueues', inv.ok, inv.why);
  check(`  size == ${N}`, q.size === N, `got ${q.size}`);

  let firstBad = -1;
  for (let i = 0; i < N; i++) {
    const [, val] = q.dequeue();
    if (val !== i) { firstBad = i; break; }
  }
  check('  dequeue order is FIFO',
        firstBad === -1, firstBad >= 0 ? `first divergence at i=${firstBad}` : '');

  const inv2 = structuralInvariants(q);
  check('  empty queue invariants', inv2.ok && q.size === 0,
        `size=${q.size} ${inv2.why}`);
}

// -----------------------------------------------------------------------------
// T2: mixed traffic still preserves FIFO for whichever items were enqueued
// before any other still-queued item.
// -----------------------------------------------------------------------------
function t2_mixedFifo(steps: number, seed: number) {
  console.log(`T2 mixed FIFO (${steps} ops, seed=${seed})`);
  const rng = mulberry32(seed);
  const q: any = new LinkedQueue();
  let nextId = 0;
  const dequeued: number[] = [];

  for (let i = 0; i < steps; i++) {
    const r = rng();
    if (r < 0.6 || q.size === 0) {
      q.enqueue(`k${nextId}`, nextId);
      nextId++;
    } else {
      const [, val] = q.dequeue();
      dequeued.push(val as number);
    }
  }
  while (q.size > 0) {
    const [, val] = q.dequeue();
    dequeued.push(val as number);
  }

  let firstBad = -1;
  for (let i = 0; i < dequeued.length; i++) {
    if (dequeued[i] !== i) { firstBad = i; break; }
  }
  check(`  ${dequeued.length} items dequeued in enqueue order`,
        firstBad === -1,
        firstBad >= 0 ? `first divergence at i=${firstBad}: got ${dequeued[firstBad]}` : '');

  const inv = structuralInvariants(q);
  check('  invariants hold at end', inv.ok && q.size === 0,
        `size=${q.size} ${inv.why}`);
}

// -----------------------------------------------------------------------------
// T3: remove(k) preserves relative order of survivors
// -----------------------------------------------------------------------------
function t3_removePreservesOrder(N: number, removeFrac: number, seed: number) {
  console.log(`T3 remove preserves order (N=${N}, remove ${removeFrac * 100}%, seed=${seed})`);
  const rng = mulberry32(seed);
  const q: any = new LinkedQueue();
  for (let i = 0; i < N; i++) q.enqueue(`k${i}`, i);

  const removed = new Set<number>();
  const targetRemovals = Math.floor(N * removeFrac);
  while (removed.size < targetRemovals) {
    const i = Math.floor(rng() * N);
    if (!removed.has(i)) {
      const [, val] = q.remove(`k${i}`);
      if (val !== i) {
        check(`  remove returned wrong value at i=${i}`, false, `got ${val}`);
        return;
      }
      removed.add(i);
    }
  }

  const inv = structuralInvariants(q);
  check(`  invariants after ${removed.size} random removes`, inv.ok, inv.why);
  check(`  size == ${N - removed.size}`, q.size === N - removed.size, `got ${q.size}`);

  const survivors: number[] = [];
  while (q.size > 0) {
    const [, val] = q.dequeue();
    survivors.push(val as number);
  }
  let bad = -1;
  for (let i = 1; i < survivors.length; i++) {
    if (survivors[i] <= survivors[i - 1]) { bad = i; break; }
  }
  check('  survivor order preserved (strictly increasing original indices)',
        bad === -1, bad >= 0 ? `inversion at i=${bad}` : '');
}

// -----------------------------------------------------------------------------
// T4: getRandomKey is uniform.
// chi-square statistic = sum((O - E)^2 / E) with df = N - 1.
// For N=100, df=99 and the 99.9-percentile critical value is ~ 148.2.
// -----------------------------------------------------------------------------
function t4_randomKeyUniform(N: number, draws: number, seed: number) {
  console.log(`T4 getRandomKey uniform (N=${N}, draws=${draws}, seed=${seed})`);
  const rng = mulberry32(seed);
  const origRandom = Math.random;
  Math.random = rng;          // make the queue's getRandomKey deterministic
  try {
    const q: any = new LinkedQueue();
    for (let i = 0; i < N; i++) q.enqueue(`k${i}`, i);

    const counts: number[] = new Array(N).fill(0);
    for (let i = 0; i < draws; i++) {
      const k = q.getRandomKey() as string;
      const idx = parseInt(k.slice(1), 10);
      counts[idx]++;
    }

    const expected = draws / N;
    let chi2 = 0;
    for (let i = 0; i < N; i++) {
      const d = counts[i] - expected;
      chi2 += (d * d) / expected;
    }
    // df = N - 1 = 99. Critical values:
    //   95%   -> 123.2
    //   99%   -> 134.6
    //   99.9% -> 148.2
    const crit_999 = 148.2;
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);
    check(`  chi-square = ${chi2.toFixed(2)} < ${crit_999} (alpha=0.001, df=${N - 1})`,
          chi2 < crit_999,
          `min=${minCount}, max=${maxCount}, expected=${expected}`);
  } finally {
    Math.random = origRandom;
  }
}

// -----------------------------------------------------------------------------
// T5: no leaks after equal enqueue+dequeue (sanity for the side Map)
// -----------------------------------------------------------------------------
function t5_noLeaks(N: number) {
  console.log(`T5 no-leak after enqueue+dequeue (N=${N})`);
  const q: any = new LinkedQueue();
  for (let i = 0; i < N; i++) q.enqueue(`k${i}`, i);
  for (let i = 0; i < N; i++) q.dequeue();
  check('  size == 0',          q.size === 0,           `got ${q.size}`);
  check('  lookup.size == 0',   q.lookup.size === 0,    `got ${q.lookup.size}`);
  check('  head == null',       q.head === null,        `got ${q.head}`);
  check('  tail == null',       q.tail === null,        `got ${q.tail}`);
}

// -----------------------------------------------------------------------------
// T6: head/tail/size invariants after a long random workload
// -----------------------------------------------------------------------------
function t6_invariantsUnderLoad(steps: number, seed: number) {
  console.log(`T6 invariants under random workload (${steps} ops, seed=${seed})`);
  const rng = mulberry32(seed);
  const q: any = new LinkedQueue();
  const live = new Set<string>();
  let nextId = 0;
  let firstBadOp = -1;
  let firstBadWhy = '';

  for (let i = 0; i < steps; i++) {
    const r = rng();
    if (r < 0.5 || live.size === 0) {
      const k = `k${nextId++}`;
      q.enqueue(k, k);
      live.add(k);
    } else if (r < 0.8) {
      const [k] = q.dequeue();
      if (typeof k === 'string') live.delete(k);
    } else {
      const arr = Array.from(live);
      const k = arr[Math.floor(rng() * arr.length)];
      q.remove(k);
      live.delete(k);
    }
    if (i % 1000 === 0 || i === steps - 1) {
      const inv = structuralInvariants(q);
      if (!inv.ok || q.size !== live.size) {
        firstBadOp = i;
        firstBadWhy = inv.ok
          ? `size=${q.size} live=${live.size}`
          : inv.why;
        break;
      }
    }
  }
  check('  all invariants held throughout',
        firstBadOp === -1,
        firstBadOp >= 0 ? `first failure at op ${firstBadOp}: ${firstBadWhy}` : '');
}

// -----------------------------------------------------------------------------
// T7: iterator order matches insertion order
// -----------------------------------------------------------------------------
function t7_iteratorOrder(N: number) {
  console.log(`T7 iterator order (N=${N})`);
  const q: any = new LinkedQueue();
  for (let i = 0; i < N; i++) q.enqueue(`k${i}`, i);

  let firstBad = -1;
  let i = 0;
  for (const [, v] of q.iterator()) {
    if (v !== i) { firstBad = i; break; }
    i++;
  }
  check('  for-of yields insertion order', firstBad === -1,
        firstBad >= 0 ? `first divergence at i=${firstBad}` : '');

  const rev: number[] = [];
  for (const [, v] of q.reverseIterator()) rev.push(v as number);
  let firstBadR = -1;
  for (let j = 0; j < N; j++) {
    if (rev[j] !== N - 1 - j) { firstBadR = j; break; }
  }
  check('  reverseIterator yields reverse insertion order',
        firstBadR === -1,
        firstBadR >= 0 ? `first divergence at j=${firstBadR}: got ${rev[firstBadR]}` : '');
}

// -----------------------------------------------------------------------------
// T8: addToFront pushes to the front (LIFO front, FIFO back)
// -----------------------------------------------------------------------------
function t8_addToFront(N: number) {
  console.log(`T8 addToFront LIFO (N=${N})`);
  const q: any = new LinkedQueue();
  for (let i = 0; i < N; i++) q.addToFront(`k${i}`, i);

  let firstBad = -1;
  for (let i = 0; i < N; i++) {
    const [, val] = q.dequeue();
    if (val !== N - 1 - i) { firstBad = i; break; }
  }
  check('  dequeue yields LIFO order after addToFront',
        firstBad === -1,
        firstBad >= 0 ? `first divergence at i=${firstBad}` : '');
}

// -----------------------------------------------------------------------------
// Run it
// -----------------------------------------------------------------------------
function main() {
  console.log('@oresoftware/linked-queue bias / correctness tests');
  console.log('===================================================');
  console.log('');

  t1_pureFifo(100_000);
  t2_mixedFifo(200_000, 0xC0FFEE);
  t3_removePreservesOrder(10_000, 0.30, 0xDEADBEEF);
  t3_removePreservesOrder(10_000, 0.70, 0xFEEDFACE);
  t4_randomKeyUniform(100, 1_000_000, 0x1337C0DE);
  t5_noLeaks(50_000);
  t6_invariantsUnderLoad(200_000, 0xBADF00D);
  t7_iteratorOrder(10_000);
  t8_addToFront(10_000);

  console.log('');
  console.log(`summary: ${pass} pass, ${fail} fail`);
  if (fail > 0) {
    console.log('');
    console.log('failures:');
    for (const f of failures) console.log('  - ' + f);
    process.exit(1);
  }
}

main();
