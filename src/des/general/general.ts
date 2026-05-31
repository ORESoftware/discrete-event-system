'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/general.rs  (module des::general::general)
// 1:1 file move. Grab-bag utilities: ws send, JSON decode, shuffle, UUID, BigNumber
// histograms, DESSet/DESMap collection wrappers, error builder.
//
// Declarations → Rust:
//   interface HasComputedProperties  -> trait HasComputedProperties { fn get_with_computed_properties(&self) -> T }
//   class DESSet<V> extends Set<V>   -> newtype `struct DESSet<V>(HashSet<V>)` (V: Hash+Eq, HasId) + Serialize
//   class DESMap<K,V> extends Map    -> newtype `struct DESMap<K,V>(HashMap<K,V>)` + Serialize
//   const sendRaw/deJSON/fisherYatesShuffle/getShortUUID/getReasonableU(/Native)/
//     getSortedHistogram*/getSortedTimeHistogram/bgn/makeError -> free fns / assoc fns
//
// Conversion notes (file-specific):
//   - HEAVY `mathjs.BigNumber` usage (bgn/getReasonable*/histograms) -> pick ONE decimal
//     crate (e.g. rust_decimal) or `f64` engine-wide; `math.add/multiply/round` -> ops.
//   - `Math.random()` (fisherYatesShuffle, getReasonableU/Native) -> inject `RandomSource`.
//   - `fisherYatesShuffle` is a `function*` generator -> return an `Iterator`/`impl Iterator`.
//   - `ws` WebSocket `sendRaw` -> tokio-tungstenite; `safe.stringify`/`JSON.parse`/`toJSON` -> serde_json.
//   - extends `Set`/`Map` (subclassing builtins) has NO Rust analogue -> wrap, don't inherit.
//   - PROTOTYPE MONKEY-PATCH `(math.bignumber(69) as any).__proto__.toJSON = ...` and the
//     per-value `toJSON` patch in `bgn` -> NOT portable; implement `Serialize`/rounding explicitly.
//   - Pervasive `any`/`as any` -> choose concrete types; `makeError` (cli-color + util.inspect) -> a Display error.
// =============================================================================

import * as math from "mathjs";
import * as util from "util";

const clc = require('cli-color');
import * as uuid from 'uuid';
import * as safe from "@oresoftware/safe-stringify";
import {WebSocket} from "ws";
import {HasId} from "../abstract/interfaces";
import {RandomSource, DEFAULT_RANDOM} from "../shared/capabilities";

export const sendRaw = (c: WebSocket, data: any, options?: any, cb?: () => void) => {
  return c.send(safe.stringify(data), options, cb);
};

export const deJSON = (fn: (v: any) => void) => {
  return (z: any) => {

    // if(typeof z !== 'string'){
    //   return fn(z);
    // }

    try {
      var val = JSON.parse(z);
    } catch (err) {
      console.error('Could not parse:', z);
      console.error('Could not parse:', String(z));
      console.error(err);
      return fn(z);
    }

    return fn(val);

  };
}

export const fisherYatesShuffle = function* <T>(deck: Array<T>, rng: RandomSource = DEFAULT_RANDOM) {
  // TODO: store previous array
  for (let i = deck.length - 1; i >= 0; i--) {
    const swapIndex = Math.floor(rng.nextFloat() * (i + 1));
    [deck[i], deck[swapIndex]] = [deck[swapIndex], deck[i]];
    yield deck[i];
  }
};

export interface HasComputedProperties<T = any> {
  getWithComputedProperties(): T;
}

export const getShortUUID = (): string => {
  return uuid.v4().slice(-10);
};


export class DESSet<V extends HasId> extends Set<V> {


  // toJSON(){
  //   return Array.from(Object.fromEntries(this)).map((v: any) => {
  //     return [v[0], v[1].value];
  //   });
  // }

  toJSON() {

    return {
      size: this.size,
      values: Array.from(this.values())
        .map(v => v.id)
    }

  }

  // toJSON(){
  //   return Object.fromEntries(this);
  // }

}



export class DESMap<K extends Number, V extends math.BigNumber> extends Map<K, V> {

  // constructor() {
  //   super();
  // }

  // toJSON(){
  //   return Array.from(Object.fromEntries(this)).map((v: any) => {
  //     return [v[0], v[1].value];
  //   });
  // }

  toJSON() {
    const ret = {} as any;
    for (const [k, v] of this.entries()) {
      ret[k] = Number((v as any));
    }
    return ret;
  }

  // toJSON(){
  //   return Object.fromEntries(this);
  // }

}

export const getReasonableU = (rng: RandomSource = DEFAULT_RANDOM): math.BigNumber => {
  // smallest u is 0.001, biggest is 0.999
  const u = math.max(0.001, rng.nextFloat());
  return bgn(math.min(0.999, u));
}

export const getReasonableUNative = (rng: RandomSource = DEFAULT_RANDOM): number => {
  // smallest u is 0.001, biggest is 0.999
  const u = Math.max(0.001, rng.nextFloat());
  return Math.min(0.999, u);
}


export const getSortedHistogram_NEW = (h: Map<number, math.BigNumber>): Array<math.BigNumber> => {
  // ...
  let total = bgn(0);

  const newMap: Array<math.BigNumber> = [];
  for (const v of h.values()) {
    total = math.add(total, v);
  }

  for (const [k, v] of h.entries()) {

    const product = math.multiply(
      100,
      math.divide(v, total) as math.BigNumber
    );

    const roundTo = math.larger(product, bgn(10)) ? 4 : 3;
    newMap[k] = math.round(product as math.BigNumber, roundTo) as math.BigNumber;
  }

  return newMap;
}

export const getSortedTimeHistogram = (h: Map<number, math.BigNumber>) => {
  return new Map(Array.from(h.entries()).sort((a, b) => a[0] - b[0]))
};

export const getSortedHistogram = (h: Map<number, math.BigNumber>) => {
  // ...
  let total = bgn(0);
  const newMap = new Map();
  for (const v of h.values()) {
    total = math.add(total, v);
  }
  for (const [k, v] of h.entries()) {

    const product = math.multiply(
      100,
      math.divide(v, total) as math.BigNumber
    );

    const roundTo = math.larger(product, bgn(10)) ? 3 : 4;

    newMap.set(k, math.round(
      product as math.BigNumber,
      roundTo) as math.BigNumber
    );
  }

  return new Map([...newMap].sort((a, b) => a[0] - b[0]));
}

(math.bignumber(69) as any).__proto__.toJSON = function () {
  const val = this as any;
  return Number((math.larger(100, val) ? math.round(val, 2) : math.round(val, 4) as any))
};


export const bgn = (v: number | math.BigNumber) => {
  const z = math.bignumber(v) as math.BigNumber;
  (z as any).toJSON = () => {
    return Number((math.larger(100, z) ? math.round(z, 2) : math.round(z, 4) as any))
  }
  return z;
}

const isPrimitive = (v: any) => {
  return !(v && typeof v === 'object');
};

export const makeError = (...msg: Array<any>) => {
  return new Error(msg.map(v => {
    return clc.bold(isPrimitive(v) ? v : util.inspect(v, {depth: 20}));
  }).join(' '));
}

