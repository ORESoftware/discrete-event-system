// RUST MIGRATION: Port file-for-file to `tests/test.rs` only if this remains a smoke suite for general helpers and random-variable behavior.
// Test-port notes: convert console/manual checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, approximate-float helpers, and deterministic seeds.

'use strict';

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
