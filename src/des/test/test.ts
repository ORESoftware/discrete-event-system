'use strict';

// =============================================================================
// RUST MIGRATION  —  target: examples/rv_sampling_bench.rs   (example binary)
// 1:1 file move. Ad-hoc throughput/distribution probe for the exponential RVs;
// it PRINTS buckets + timings and has no assertions, so it fits an `examples/`
// binary (a fn main) rather than a `#[test]` — add asserts if it becomes a test.
//
// Test harness → Rust:
//   no PASS/FAIL harness here — console.log bucket dumps -> if promoted to a
//   #[test], turn the histogram expectations into assert!/assert_eq!.
//
// Conversion notes (file-specific):
//   - Date.now() timing -> std::time::Instant (a bench, not an assertion).
//   - mathjs bgn()/BigNumber -> a decimal crate or f64 (pick ONE engine-wide).
//   - Map<number,number> buckets -> HashMap<i64, i64>; `... as any` casts drop.
//   - process.exit(0) mid-file -> the dead tail code below it is unreachable;
//     port only the reachable part (or delete the post-exit block).
// =============================================================================

import {IterableInt} from 'iterable.int';
import {bgn, makeError} from "../general/general";
import * as math from "mathjs";
import {ExponentialRandomVariable, ExponentialRandomVariable2, UniformRandomVariable} from "../random-variables/rv";


const ts = bgn(600);

const erv = new ExponentialRandomVariable({lambda: math.divide(bgn(1), bgn(500)) as math.BigNumber, timeStep: ts});

const date1 = Date.now();
const bucket1 = new Map<number, number>([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]);
for (const v of new IterableInt(100000)) {
  const val = erv.getNextEventQuantity(ts);
  let g = bucket1.get(val) as any;
  bucket1.set(val, ++g);
  console.log('erv:', erv.getNextEventQuantity(ts));
}
const date1done = Date.now() - date1;

const bucket2 = new Map<number, number>([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]);
const erv2 = new ExponentialRandomVariable2({lambda: math.divide(bgn(1), bgn(500)) as math.BigNumber, timeStep: ts});

const date2 = Date.now();
for (const v of new IterableInt(100000)) {
  const val = erv2.getNextEventQuantity(ts);
  let g = bucket2.get(val) as any;
  bucket2.set(val, ++g);
  console.log('erv2:', val);
}

console.log('1:', date1done);
console.log('2:', Date.now() - date2);

console.log({
  bucket1,
  bucket2
});


console.log('===================================');
console.log('===================================');
console.log('===================================');
console.log('===================================');

process.exit(0) as any;

const bucket3 = new Map()

const urv = new UniformRandomVariable({aVal: bgn(0), bVal: bgn(2)});

for (const v of new IterableInt(10000)) {

  const val = urv.getNextEventQuantity(ts);
  let g = bucket3.get(val) || 0;
  bucket3.set(val, ++g);
  console.log('urv:', val);
}

console.log({
  bucket3: new Map([...bucket3.entries()].sort())
})

// for(const v of new IterableInt(500)){
//     console.log(erv.getNextEventQuantity(bgn(1)))
// }

// console.log(erv.getNextEventCountTest());
//
//
// throw makeError('foo', 5, true, {ugly:'duckling'});


// let i = 0;
// while(true){
//     console.log(process.pid, i++);
// }

