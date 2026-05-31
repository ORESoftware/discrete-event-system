#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/random-variables/half.rs  (module des::random_variables::half)
// 1:1 file move. Throwaway CLI script that down-samples an array by pairwise averaging.
//
// Declarations → Rust:
//   const goFrom_131072_to_1024 (arrow fn) -> free fn
//
// Conversion notes (file-specific):
//   - This is a SCRIPT (top-level `console.log`) -> a `[[bin]]`/examples main, not library.
//   - `number[]` -> `Vec<f64>`; `reduceByHalf` uses `any[]` + a `[array, number|null]`
//     tuple accumulator -> a typed fold (e.g. fold into `(Vec<f64>, Option<f64>)`).
//   - LATENT BUG to preserve verbatim: the `while` loop reduces the ORIGINAL `v`
//     (not `ret`), so it never converges on the intended input — translate as-is
//     and flag, don't silently "fix".
// =============================================================================

const goFrom_131072_to_1024 = (v: number[]) => {

  let ret = [...v].sort();

  const reduceByHalf = (v: any[]) => {
    return v.reduce((a,b,currentIndex) => {

      if(currentIndex % 2 !== 0){
         return [a[0],b]
      }

      a[0].push((a[1] + b)/2);
      return [a[0], null];

    },[[], null]);
  };

  while(ret.length > 1024){
    console.log(ret.length);
    ret = reduceByHalf(v)[0];
  }

  return ret;
}

console.log(
  goFrom_131072_to_1024(
    new Array(131072).fill(null).map((v,i) => i)
  )
);