'use strict';

// =============================================================================
// RUST MIGRATION — target: src/des/shared/capabilities.rs   (module des::shared::capabilities)
// 1:1 file move. See the module doc below for the full Rust mapping.
// =============================================================================

// =============================================================================
// shared/capabilities.ts — capability ports for non-deterministic / impure
// dependencies (randomness, wall-clock time).
//
// WHY
// ----
// Rust trait impls don't reach out to ambient globals; dependencies are passed
// in. Direct `Math.random()` / `Date.now()` calls are (a) non-deterministic, so
// simulations aren't reproducible, and (b) awkward to translate because the
// Rust side will inject an `Rng` / `Clock`. Routing these through small traits
// makes both problems disappear: production code uses the default adapters,
// tests/simulations use a seeded RNG, and the Rust migration swaps in the
// `rand` crate behind the same trait.
//
// Rust mapping:
//   interface RandomSource  ->  trait RandomSource  (cf. rand::Rng)
//   interface Clock         ->  trait Clock
// =============================================================================

/** Source of pseudo-randomness. `nextFloat()` returns [0, 1). */
export interface RandomSource {
  /** Uniform float in [0, 1). */
  nextFloat(): number;
  /** Uniform integer in [min, max). */
  nextInt(min: number, max: number): number;
  /** Standard-normal sample (mean 0, variance 1). */
  nextGaussian(): number;
}

/** Source of the current time, in milliseconds since the Unix epoch. */
export interface Clock {
  nowMs(): number;
}

// -----------------------------------------------------------------------------
// Default adapters — wrap the platform globals. These are the ONLY places in
// the engine that should touch Math.random() / Date.now() directly.
// -----------------------------------------------------------------------------

export class SystemRandom implements RandomSource {
  nextFloat(): number {
    return Math.random();
  }

  nextInt(min: number, max: number): number {
    return min + Math.floor(Math.random() * (max - min));
  }

  nextGaussian(): number {
    return gaussianFrom(() => Math.random());
  }
}

export class SystemClock implements Clock {
  nowMs(): number {
    return Date.now();
  }
}

// -----------------------------------------------------------------------------
// Seeded, reproducible RNG (mulberry32). Deterministic given a seed, so
// simulations and tests are repeatable. Maps cleanly to a small Rust struct.
// -----------------------------------------------------------------------------

export class SeededRandom implements RandomSource {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextFloat(): number {
    // mulberry32
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextInt(min: number, max: number): number {
    return min + Math.floor(this.nextFloat() * (max - min));
  }

  nextGaussian(): number {
    return gaussianFrom(() => this.nextFloat());
  }
}

/** Box–Muller transform shared by all RandomSource implementations. */
function gaussianFrom(uniform: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = uniform();
  while (v === 0) v = uniform();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Process-wide default capabilities for code paths that have not yet been
 *  threaded with explicit ports. Prefer injecting a RandomSource/Clock. */
export const DEFAULT_RANDOM: RandomSource = new SystemRandom();
export const DEFAULT_CLOCK: Clock = new SystemClock();
