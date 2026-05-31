#!/usr/bin/env ts-node
'use strict';

// RUST MIGRATION:
// - Target: src/des/random_variables/half.rs
// - This helper should become a named PureTransform struct, e.g.
//   DownsampleByHalving { target_len }, with `transform(Vec<f64>) -> Vec<f64>`.
// - Fix the implicit JS sort before porting: Rust needs an explicit numeric
//   comparator/ordering, especially for floats.
// - Replace `any[]`, tuple accumulator tricks, and console-driven execution
//   with typed Vec operations and a separate bin/test harness.

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
