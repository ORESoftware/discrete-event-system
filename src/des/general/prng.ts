'use strict';

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
