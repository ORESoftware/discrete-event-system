// RUST MIGRATION: Target module `src/des/general/random_variables.rs`.
// RUST MIGRATION: Keep PMF math and sampling helpers as free functions over slices/`Vec<f64>`; add `PureTransform` wrappers only when a DES graph consumes them directly.
// RUST MIGRATION: Replace callback RNGs with a generic RNG trait parameter, and make every sampler accept the RNG explicitly.
// RUST MIGRATION: Convert invalid probabilities, rates, shapes, and non-normalized mass checks into `Result` errors instead of throwing.
// RUST MIGRATION: Use iterator-based convolution and distribution builders, but preserve deterministic output ordering for tests and future serde snapshots.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/random-variables.rs  (module des::general::random_variables)
// 1:1 file move. PMF algebra (convolution, Poisson-binomial, competing risks) + samplers.
//
// Declarations → Rust:
//   fn pmf*/normalizePMF/mean/varianceFromPMF/discreteConvolve*/bernoulli/binomial/
//      poissonBinomialPMF/competingRisks/discretisePDF -> free fns or PureTransform classes (§3.1)
//   fn sampleCategorical/sampleFromPMF/samplePoisson/sampleExponential/sampleGamma
//      -> StatefulTransform / fns taking a RandomSource (NOT pure)
//
// Conversion notes (file-specific):
//   - INJECT RNG: every `sample*` takes `rng: () => number` -> take a `RandomSource`
//     (shared/capabilities). This module is the consumer side of the prng.ts ↔ capability
//     port: pure PMF math stays deterministic; only the samplers need the RNG.
//   - PMFs are `ReadonlyArray<number>` -> `&[f64]` / `Vec<f64>`; return fresh `Vec<f64>`.
//   - `discretisePDF` takes a `(x)=>number` closure -> generic `F: Fn(f64) -> f64`.
//   - validation throws on bad mass/inputs -> `panic!` (invariant) or `Result`.
// =============================================================================
// Random-variable toolkit for the framework.
//
// Why this module exists
// ----------------------
// Several places in the simulation need to compute or sample from sums of
// independent random variables. The math fact is well-known: the
// distribution of X + Y, when X and Y are independent, is the convolution
// of their distributions:
//
//                f_{X+Y}(z) = (f_X * f_Y)(z) = ∫ f_X(t) f_Y(z - t) dt
//
// (Discrete: P(X+Y = k) = Σ_i P(X=i) · P(Y = k - i).)
//
// In the framework we encounter sums of independent RVs in many places:
//
//   - Per-tick competing-risk transitions in the two-disease model.
//     A person in compartment A faces three independent rates (acquire B,
//     recover, die) and we need P(none happen) and P(specific one happens
//     | something happened). The exact answer is NOT "draw uniform and
//     partition by lambda·dt"; that's a first-order approximation that is
//     fine for small dt but biased for large dt. The exact formula uses
//     1 − exp(−Λ·dt) for "any event", multiplied by lambda_j/Λ for the
//     specific event, and is independent of dt.
//
//   - Number-of-deaths-this-tick in any compartment.
//     N people each have an independent Bernoulli(p_death) draw. The total
//     number of deaths is Poisson-binomial-distributed (or exactly
//     binomial if p is the same for all). Convolving N Bernoulli PMFs
//     gives the exact PMF, which we may need for analytic invariants or
//     cross-checks against simulation.
//
//   - Any "sum of K Erlangs / gammas / exponentials" reasoning needed by
//     the queueing analyses (M/M/1 service-time variance, etc.).
//
// What this module provides
// -------------------------
// All functions are pure, deterministic given their inputs (samplers are
// driven by an injected RNG callback so they remain reproducible). All
// PMFs are represented as plain `number[]`, where `pmf[k] = P(X = k)`.
// For continuous distributions, we use a discretised representation:
// "PMF on a regular grid".
//
//   discreteConvolve(p, q)         O(|p|·|q|) convolution of two PMFs
//   discreteConvolveMany(pmfs)     iterative left-fold
//   binomialPMF(n, p)              closed-form
//   poissonBinomialPMF(probs)      exact Poisson-binomial PMF via convolution
//   competingRisks(rates, dt)      exact discrete-time first-event probs
//   sampleCategorical(probs, rng)  categorical sampling
//   sampleFromPMF(pmf, rng)        sample from a discrete RV
//   meanFromPMF(pmf)               Σ k · p_k
//   varianceFromPMF(pmf)           E[X^2] − (E[X])^2
//   normalizePMF(pmf)              divide by total mass; tolerates tiny drift
//   pmfTotalMass(pmf)              Σ p_k (for sanity checks)
//
// All functions are documented inline. Tests in `test/random-variables-test.ts`
// pin every claim with either an analytic identity, a Monte Carlo
// cross-check (≥ 10^5 samples), or both.
// =============================================================================

import {PureTransform} from '../shared/transform';

// -----------------------------------------------------------------------------
// PMF utilities
// -----------------------------------------------------------------------------

export function pmfTotalMass(pmf: ReadonlyArray<number>): number {
  let s = 0;
  for (let i = 0; i < pmf.length; i++) s += pmf[i];
  return s;
}

export function normalizePMF(pmf: ReadonlyArray<number>): number[] {
  const m = pmfTotalMass(pmf);
  if (m <= 0) throw new Error('cannot normalise zero-mass PMF');
  const out = new Array<number>(pmf.length);
  for (let i = 0; i < pmf.length; i++) out[i] = pmf[i] / m;
  return out;
}

export function meanFromPMF(pmf: ReadonlyArray<number>): number {
  let s = 0;
  for (let k = 0; k < pmf.length; k++) s += k * pmf[k];
  return s;
}

export function varianceFromPMF(pmf: ReadonlyArray<number>): number {
  let m = 0, m2 = 0;
  for (let k = 0; k < pmf.length; k++) { m += k * pmf[k]; m2 += k * k * pmf[k]; }
  return m2 - m * m;
}

// -----------------------------------------------------------------------------
// Convolution
// -----------------------------------------------------------------------------

/**
 * Discrete linear convolution of two PMFs. If P(X = i) = p[i] and
 * P(Y = j) = q[j] with X ⊥ Y, then the returned array is P(X + Y = k)
 * for k = 0 .. (|p| + |q| − 2).
 *
 * Cost: O(|p| · |q|). For very large PMFs an FFT-based variant is
 * faster, but is not provided here — typical PMFs in this codebase are
 * small (≤ a few thousand entries).
 */
/** The two PMFs to convolve. */
export interface ConvolvePair {
  p: ReadonlyArray<number>;
  q: ReadonlyArray<number>;
}

/** Discrete linear convolution of two PMFs. See {@link discreteConvolve}. */
export class DiscreteConvolve extends PureTransform<ConvolvePair, number[]> {
  transform({p, q}: ConvolvePair): number[] {
    if (p.length === 0 || q.length === 0) return [];
    const out = new Array<number>(p.length + q.length - 1).fill(0);
    for (let i = 0; i < p.length; i++) {
      if (p[i] === 0) continue;
      for (let j = 0; j < q.length; j++) {
        out[i + j] += p[i] * q[j];
      }
    }
    return out;
  }
}

/** @deprecated Use `new DiscreteConvolve().transform({p, q})`. */
export function discreteConvolve(p: ReadonlyArray<number>, q: ReadonlyArray<number>): number[] {
  return new DiscreteConvolve().transform({p, q});
}

/** Iterative convolution: P(Σ X_k = n) given each X_k's PMF. */
export class DiscreteConvolveMany extends PureTransform<ReadonlyArray<ReadonlyArray<number>>, number[]> {
  transform(pmfs: ReadonlyArray<ReadonlyArray<number>>): number[] {
    if (pmfs.length === 0) return [1];
    let acc: number[] = [...pmfs[0]];
    for (let i = 1; i < pmfs.length; i++) acc = discreteConvolve(acc, pmfs[i]);
    return acc;
  }
}

/** @deprecated Use `new DiscreteConvolveMany().transform(pmfs)`. */
export function discreteConvolveMany(pmfs: ReadonlyArray<ReadonlyArray<number>>): number[] {
  return new DiscreteConvolveMany().transform(pmfs);
}

/**
 * n-fold self-convolution: PMF of X_1 + … + X_n where each X_k has the
 * same distribution `pmf`. Uses repeated squaring (log_2 n convolutions
 * instead of n).
 */
export class DiscreteConvolveSelf extends PureTransform<ReadonlyArray<number>, number[]> {
  constructor(private readonly n: number) {
    super();
  }

  transform(pmf: ReadonlyArray<number>): number[] {
    const n = this.n;
    if (n < 0 || !Number.isInteger(n)) throw new Error(`bad n ${n}`);
    if (n === 0) return [1];
    let result: number[] | null = null;
    let base: number[] = [...pmf];
    let m = n;
    while (m > 0) {
      if (m & 1) result = result === null ? [...base] : discreteConvolve(result, base);
      m >>>= 1;
      if (m > 0) base = discreteConvolve(base, base);
    }
    return result!;
  }
}

/** @deprecated Use `new DiscreteConvolveSelf(n).transform(pmf)`. */
export function discreteConvolveSelf(pmf: ReadonlyArray<number>, n: number): number[] {
  return new DiscreteConvolveSelf(n).transform(pmf);
}

// -----------------------------------------------------------------------------
// Common PMFs
// -----------------------------------------------------------------------------

/** Bernoulli(p) PMF: [1−p, p]. */
export function bernoulliPMF(p: number): number[] {
  if (!(p >= 0 && p <= 1)) throw new Error(`bad p ${p}`);
  return [1 - p, p];
}

/** Binomial(n, p) closed-form PMF. Stable for n ≤ ~1500 in float64. The
 *  success probability `p` is the configuration; `n` (trial count) is the input. */
export class BinomialPMF extends PureTransform<number, number[]> {
  constructor(private readonly p: number) {
    super();
  }

  transform(n: number): number[] {
    const p = this.p;
    if (!(p >= 0 && p <= 1)) throw new Error(`bad p ${p}`);
    if (!(n >= 0 && Number.isInteger(n))) throw new Error(`bad n ${n}`);
    if (n === 0) return [1];
    // Recursive formula: P(k+1) = P(k) · (n-k)/(k+1) · p/(1-p)
    const out = new Array<number>(n + 1).fill(0);
    if (p === 0)         { out[0] = 1; return out; }
    if (p === 1)         { out[n] = 1; return out; }
    out[0] = Math.pow(1 - p, n);
    const r = p / (1 - p);
    for (let k = 0; k < n; k++) {
      out[k + 1] = out[k] * (n - k) * r / (k + 1);
    }
    return out;
  }
}

/** @deprecated Use `new BinomialPMF(p).transform(n)`. */
export function binomialPMF(n: number, p: number): number[] {
  return new BinomialPMF(p).transform(n);
}

/**
 * Poisson-binomial PMF: P(Σ B_i = k) where B_i ~ Bernoulli(probs[i])
 * are independent with possibly different probabilities. Computed
 * exactly by convolving the per-item Bernoulli PMFs.
 *
 * If all probs are equal, this is binomial; we delegate to the closed
 * form for stability. Otherwise we accumulate left-to-right; cost is
 * O(N²) in the number of items (N convolutions of length 2 each).
 */
export class PoissonBinomialPMF extends PureTransform<ReadonlyArray<number>, number[]> {
  transform(probs: ReadonlyArray<number>): number[] {
    if (probs.length === 0) return [1];
    // Detect uniform p — closed-form binomial is more numerically stable.
    let allEqual = true;
    for (let i = 1; i < probs.length; i++) {
      if (Math.abs(probs[i] - probs[0]) > 1e-15) { allEqual = false; break; }
    }
    if (allEqual) return binomialPMF(probs.length, probs[0]);
    // Iterative convolution of Bernoulli(p_i) PMFs.
    let pmf: number[] = [1];
    for (let i = 0; i < probs.length; i++) {
      const p = probs[i];
      if (!(p >= 0 && p <= 1)) throw new Error(`bad p[${i}] ${p}`);
      const next = new Array<number>(pmf.length + 1).fill(0);
      for (let k = 0; k < pmf.length; k++) {
        next[k]     += pmf[k] * (1 - p);
        next[k + 1] += pmf[k] * p;
      }
      pmf = next;
    }
    return pmf;
  }
}

/** @deprecated Use `new PoissonBinomialPMF().transform(probs)`. */
export function poissonBinomialPMF(probs: ReadonlyArray<number>): number[] {
  return new PoissonBinomialPMF().transform(probs);
}

// -----------------------------------------------------------------------------
// Competing-risks discrete-time transition probabilities
// -----------------------------------------------------------------------------

/**
 * Given K independent continuous rates λ_1, …, λ_K and a time step dt,
 * compute the exact discrete-time transition probabilities:
 *
 *   p_no  = exp(−Λ·dt)              where Λ = Σ λ_k
 *   p_k   = (λ_k / Λ) · (1 − p_no)
 *
 * Returns `[p_no, p_1, p_2, …, p_K]`. The probabilities sum to 1 modulo
 * float drift. This is the EXACT first-event distribution and is
 * preferred over the linear approximation `λ_k · dt` whenever Λ·dt is
 * not very small.
 *
 * Linear approximation comparison:
 *   p_linear = λ·dt              (first-order Taylor)
 *   p_exact  = (λ/Λ)·(1 − e^{−Λdt})
 *   relative error ≈ Λ·dt / 2 + O((Λ·dt)²)
 *
 * So at Λ·dt = 0.1 the linear bias is ~5%, at Λ·dt = 0.5 it's ~25%. For
 * robustness, use the exact form everywhere.
 */
export class CompetingRisks extends PureTransform<ReadonlyArray<number>, number[]> {
  constructor(private readonly dt: number) {
    super();
  }

  transform(rates: ReadonlyArray<number>): number[] {
    const dt = this.dt;
    if (dt < 0) throw new Error(`bad dt ${dt}`);
    let total = 0;
    for (let i = 0; i < rates.length; i++) {
      if (rates[i] < 0) throw new Error(`bad rate[${i}] ${rates[i]}`);
      total += rates[i];
    }
    if (total === 0) {
      const out = new Array<number>(rates.length + 1).fill(0);
      out[0] = 1;
      return out;
    }
    const pNo = Math.exp(-total * dt);
    const pAny = 1 - pNo;
    const out = new Array<number>(rates.length + 1);
    out[0] = pNo;
    for (let i = 0; i < rates.length; i++) {
      out[i + 1] = (rates[i] / total) * pAny;
    }
    return out;
  }
}

/** @deprecated Use `new CompetingRisks(dt).transform(rates)`. */
export function competingRisks(rates: ReadonlyArray<number>, dt: number): number[] {
  return new CompetingRisks(dt).transform(rates);
}

// -----------------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------------

/**
 * Categorical sampler: given probs that sum to 1 (within float drift),
 * draw an index in 0..probs.length-1 with the given probabilities.
 * Uses linear search (O(K)) — fine for small K. For large K, replace
 * with the alias method.
 */
export class CategoricalSampler extends PureTransform<() => number, number> {
  constructor(private readonly probs: ReadonlyArray<number>) {
    super();
  }

  transform(rng: () => number): number {
    const probs = this.probs;
    const r = rng();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r <= cum) return i;
    }
    return probs.length - 1;
  }
}

/** @deprecated Use `new CategoricalSampler(probs).transform(rng)`. */
export function sampleCategorical(probs: ReadonlyArray<number>, rng: () => number): number {
  return new CategoricalSampler(probs).transform(rng);
}

/**
 * Sample a single integer outcome from a PMF over {0, 1, …, |pmf|-1}.
 * Equivalent to sampleCategorical but the function name is more
 * informative when the array represents a numerical RV's PMF.
 */
export function sampleFromPMF(pmf: ReadonlyArray<number>, rng: () => number): number {
  return sampleCategorical(pmf, rng);
}

// -----------------------------------------------------------------------------
// Continuous-distribution samplers (used by contact-based simulations).
// -----------------------------------------------------------------------------

/**
 * Draw a Poisson-distributed integer with mean λ. For small λ uses
 * Knuth's algorithm (multiply uniforms until the cumulative product
 * drops below e^{-λ}); for λ ≥ 30 uses a normal approximation with
 * continuity correction. Both are unbiased asymptotically; the normal
 * approximation has < 0.1% relative error in the body of the
 * distribution at λ ≥ 30.
 *
 * Why not use a single algorithm: Knuth gets numerically unstable for
 * λ ≳ 700 (e^{-λ} underflows). The normal switch keeps things in
 * float64 range.
 */
export class PoissonSampler extends PureTransform<() => number, number> {
  constructor(private readonly lambda: number) {
    super();
  }

  transform(rng: () => number): number {
    const lambda = this.lambda;
    if (lambda < 0) throw new Error(`bad lambda ${lambda}`);
    if (lambda === 0) return 0;
    if (lambda < 30) {
      const L = Math.exp(-lambda);
      let k = 0; let p = 1;
      while (true) {
        k++;
        p *= rng();
        if (p <= L) return k - 1;
      }
    }
    // Normal approximation with continuity correction. mean = lambda,
    // variance = lambda. We use Box-Muller for the normal draw.
    const u1 = 1 - rng();
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const x = lambda + Math.sqrt(lambda) * z + 0.5;
    return Math.max(0, Math.floor(x));
  }
}

/** @deprecated Use `new PoissonSampler(lambda).transform(rng)`. */
export function samplePoisson(lambda: number, rng: () => number): number {
  return new PoissonSampler(lambda).transform(rng);
}

/**
 * Draw an Exponential(rate) sample. rate must be > 0.
 *   X = −ln(U) / rate     where U ~ Uniform(0, 1].
 */
export class ExponentialSampler extends PureTransform<() => number, number> {
  constructor(private readonly rate: number) {
    super();
  }

  transform(rng: () => number): number {
    const rate = this.rate;
    if (!(rate > 0)) throw new Error(`bad rate ${rate}`);
    const u = 1 - rng();   // Uniform(0, 1] (avoid log(0))
    return -Math.log(u) / rate;
  }
}

/** @deprecated Use `new ExponentialSampler(rate).transform(rng)`. */
export function sampleExponential(rate: number, rng: () => number): number {
  return new ExponentialSampler(rate).transform(rng);
}

/**
 * Draw a Gamma(shape, scale) sample using Marsaglia & Tsang's method
 * (2000), which is exact and very fast. Mean = shape · scale,
 * variance = shape · scale².
 *
 * Used to draw heterogeneous per-person contact rates with a
 * specified mean and coefficient of variation:
 *   shape = 1 / cv²,  scale = mean · cv²
 *
 * cv = 0 returns a degenerate "always equal to mean" (special-case).
 */
export class GammaSampler extends PureTransform<() => number, number> {
  constructor(private readonly shape: number, private readonly scale: number) {
    super();
  }

  transform(rng: () => number): number {
    const shape = this.shape;
    const scale = this.scale;
    if (!(shape > 0) || !(scale > 0)) throw new Error(`bad shape/scale ${shape}/${scale}`);
    if (shape < 1) {
      // Boost: sample Gamma(shape+1, scale) and scale by U^{1/shape}.
      const g = sampleGamma(shape + 1, scale, rng);
      const u = 1 - rng();
      return g * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x: number, v: number;
      do {
        const u1 = 1 - rng();
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        x = z;
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rng();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
    }
  }
}

/** @deprecated Use `new GammaSampler(shape, scale).transform(rng)`. */
export function sampleGamma(shape: number, scale: number, rng: () => number): number {
  return new GammaSampler(shape, scale).transform(rng);
}

// -----------------------------------------------------------------------------
// Continuous: simple grid-based convolution for completeness.
// -----------------------------------------------------------------------------

/**
 * Discretise a continuous PDF f on the regular grid x = x0 + i·h for
 * i = 0..n−1, returning a PMF of approximate point masses
 *   pmf[i] ≈ f(x0 + i·h) · h.
 * Caller is responsible for normalisation and for choosing a grid wide
 * enough to capture the support.
 */
export class DiscretisePDF extends PureTransform<(x: number) => number, number[]> {
  constructor(
    private readonly x0: number,
    private readonly h: number,
    private readonly n: number,
  ) {
    super();
  }

  transform(f: (x: number) => number): number[] {
    const {x0, h, n} = this;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = f(x0 + i * h) * h;
    return out;
  }
}

/** @deprecated Use `new DiscretisePDF(x0, h, n).transform(f)`. */
export function discretisePDF(
  f: (x: number) => number, x0: number, h: number, n: number,
): number[] {
  return new DiscretisePDF(x0, h, n).transform(f);
}
