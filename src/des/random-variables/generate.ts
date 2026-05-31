#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/random-variables/generate.rs  (module des::random_variables::generate)
// 1:1 file move. Demo/CLI script: sample uniform/exponential draws and print moments.
//
// Declarations → Rust:
//   const runUniform (fn)     -> free fn
//   const runExponential (fn) -> free fn
//
// Conversion notes (file-specific):
//   - DEMO SCRIPT: the `if (require.main === module) runExponential()` guard +
//     `console.log` output -> a `[[bin]]`/examples main, not library code.
//   - `const math = require('mathjs')` (CommonJS) mixed with `import {bignumber}`
//     -> a single `use`; the require disappears.
//   - `math.random()` / `Math.random()` -> injected RandomSource (no ambient rng).
//   - moment estimators on `bignumber`/`number` -> decimal/f64; `math.pow/log/sum` -> ops.
// =============================================================================

import {bignumber} from "mathjs";

const math = require('mathjs');

export const runUniform = () => {

  const a = 5;
  const b = 8;
  const count = 100000;

  const values = [];

  for (let i = 0; i < count; i++) {
    values.push(
      a + math.random() * (b - a)
    );
  }

  for (const v of values) {
    console.log(v);
  }


};


export const runExponential = () => {

  const lambda = 1;
  const count = bignumber(1000);
  const values = [];

  for (let i = 0; i < Number(count); i++) {
    values.push(
      math.multiply(math.divide(-1, lambda), math.log(1 - math.random()))
    );
  }

  const mapToSquare = ((v: number) => math.multiply(v, v));
  const sumReducer = (a: number, b: number) => math.sum(a, b);

  const squared = values.map(mapToSquare);
  const cubed = squared.map(mapToSquare);
  const quad = cubed.map(mapToSquare);
  const quint = quad.map(mapToSquare);

  const sum1 = values.reduce(sumReducer, 0);
  const sum2 = squared.reduce(sumReducer, 0);
  const sum3 = cubed.reduce(sumReducer, 0);
  const sum4 = quad.reduce(sumReducer, 0);
  const sum5 = quint.reduce(sumReducer, 0);

  const firstOrder = (1) / (math.pow(1, 1));
  const secondOrder = (2) / (math.pow(1, 2));
  const thirdOrder = (3 * 2) / (math.pow(1, 3));
  const fourthOrder = (4 * 3 * 2) / (math.pow(1, 4));
  const fifthOrder = (5 * 4 * 3 * 2) / (math.pow(1, 5));

  console.log(
    firstOrder,
    secondOrder,
    thirdOrder,
    fourthOrder,
    fifthOrder
  );

  console.log(
    math.divide(bignumber(sum1), bignumber(count)),
    math.divide(bignumber(sum2), bignumber(count)),
    math.divide(bignumber(sum3), bignumber(count)),
    math.divide(bignumber(sum4), bignumber(count)),
    math.divide(bignumber(sum5), bignumber(count)),
  );

};


if (require.main === module) {
  runExponential();
}