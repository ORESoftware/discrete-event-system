// RUST MIGRATION: Target module `src/des/general/prng.rs`.
// RUST MIGRATION: Port `mulberry32` as a small seeded RNG struct implementing a local RNG trait or `rand_core::RngCore`.
// RUST MIGRATION: Do not reproduce `withSeed` as a global `Math.random` swap; pass RNG handles explicitly through simulator/model APIs.
// RUST MIGRATION: Keep the helper API free-function shaped for compatibility, but make Rust callers prefer dependency-injected RNG ports for deterministic tests.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/prng.rs  (module des::general::prng)
// 1:1 file move. Seedable mulberry32 PRNG used for reproducible Monte-Carlo replications.
//
// Declarations → Rust:
//   fn mulberry32(seed) -> () => number  -> this IS the `SeededRandom` capability
//                                           (shared/capabilities); return a struct impl RandomSource.
//   fn withSeed<T>(seed, fn)             -> NO direct Rust analogue (see note); express as
//                                           "run this closure with an injected SeededRandom".
//
// Conversion notes (file-specific):
//   - `withSeed` MONKEY-PATCHES the global via `(Math as any).random = prng` and restores it.
//     Rust has no global mutable RNG to swap: the whole RNG-capability port (shared/capabilities,
//     prng.ts ↔ random-variables.ts) exists to replace this. Migrate call sites to take a
//     `RandomSource` parameter instead of relying on a swapped global.
//   - mulberry32 uses u32 wrapping ops (`>>> 0`, `Math.imul`) -> `u32` `wrapping_*` / `^` in Rust;
//     output is `(t >>> 0) / 4294967296` in [0,1) -> `as f64 / u32::MAX as f64`-style mapping.
//   - the `as any` casts exist ONLY to mutate `Math.random`; they disappear under injection.
// =============================================================================
// Seedable PRNG so we can run reproducible Monte-Carlo replications.
//
// `withSeed(seed, fn)` swaps Math.random with a Mulberry32 PRNG keyed on the
// given seed for the duration of fn(), then restores the original random.
// This works for the entire simulator (UniformRandomVariable, fisher-yates
// shuffles, ProbabilityDecisionEntity coin flips, etc.) because every code
// path eventually goes through Math.random.
// =============================================================================

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ORIG_RANDOM: () => number = Math.random;

export function withSeed<T>(seed: number, fn: () => T): T {
  const prng = mulberry32(seed);
  (Math as any).random = prng;
  try {
    return fn();
  } finally {
    (Math as any).random = ORIG_RANDOM;
  }
}
