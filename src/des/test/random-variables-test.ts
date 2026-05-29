#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// Tests for `general/random-variables.ts`. Each claim made by the module is
// pinned by either an analytic identity or a Monte Carlo cross-check, often
// both. Run with:
//   npm run build
//   node dist/des/test/random-variables-test.js
// =============================================================================

import {
  bernoulliPMF, binomialPMF, competingRisks,
  discreteConvolve, discreteConvolveMany, discreteConvolveSelf,
  meanFromPMF, normalizePMF, pmfTotalMass, poissonBinomialPMF,
  sampleExponential, sampleFromPMF, sampleCategorical, sampleGamma,
  samplePoisson, varianceFromPMF,
} from '../general/random-variables';
import {mulberry32} from '../general/prng';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (ok) { pass++; console.log(`  PASS    ${label}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}
function approx(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}
function arrApprox(a: ReadonlyArray<number>, b: ReadonlyArray<number>, tol: number): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > tol) return false;
  return true;
}

// =============================================================================
// T1  Convolution identities and total mass
// =============================================================================
console.log('\nT1  Convolution preserves total mass and sums of independent RVs');

{
  // Simple two-PMF sanity.
  const p = [0.2, 0.5, 0.3];
  const q = [0.1, 0.4, 0.5];
  const c = discreteConvolve(p, q);
  check('discreteConvolve length is |p| + |q| − 1', c.length === 5);
  check('discreteConvolve total mass = 1', approx(pmfTotalMass(c), 1, 1e-12));
}

{
  // Convolving two Bernoulli(0.3) gives Binomial(2, 0.3).
  const b1 = bernoulliPMF(0.3);
  const expected = binomialPMF(2, 0.3);
  const got = discreteConvolve(b1, b1);
  check('Bernoulli(p) ⊕ Bernoulli(p) = Binomial(2, p)', arrApprox(got, expected, 1e-15));
}

{
  // n-fold self-convolution of Bernoulli(p) = Binomial(n, p).
  const ns = [1, 5, 17, 32, 100];
  for (const n of ns) {
    const expected = binomialPMF(n, 0.37);
    const got = discreteConvolveSelf(bernoulliPMF(0.37), n);
    const same = arrApprox(got, expected, 1e-12);
    check(`Bernoulli(0.37)^{*${n}} = Binomial(${n}, 0.37)`, same);
  }
}

{
  // Convolution is associative: (p*q)*r = p*(q*r).
  const p = [0.1, 0.4, 0.3, 0.2];
  const q = [0.5, 0.3, 0.2];
  const r = [0.25, 0.25, 0.25, 0.25];
  const left  = discreteConvolve(discreteConvolve(p, q), r);
  const right = discreteConvolve(p, discreteConvolve(q, r));
  check('discreteConvolve is associative (8-pt PMF)', arrApprox(left, right, 1e-14));
}

{
  // Mean and variance add for independent sums.
  const p = binomialPMF(10, 0.4);  // mean 4, var 2.4
  const q = binomialPMF(10, 0.4);  // mean 4, var 2.4
  const c = discreteConvolve(p, q); // mean 8, var 4.8 (== Binomial(20, 0.4))
  check('mean(p ⊕ q) = mean(p) + mean(q)',
        approx(meanFromPMF(c), meanFromPMF(p) + meanFromPMF(q), 1e-12));
  check('var(p ⊕ q) = var(p) + var(q) for independent X, Y',
        approx(varianceFromPMF(c), varianceFromPMF(p) + varianceFromPMF(q), 1e-10));
}

// =============================================================================
// T2  Poisson-binomial PMF
// =============================================================================
console.log('\nT2  Poisson-binomial PMF');

{
  // Equal probabilities → matches binomial closed form.
  const probs = new Array(20).fill(0.42);
  const pb = poissonBinomialPMF(probs);
  const bin = binomialPMF(20, 0.42);
  check('PoissonBinomial(uniform p) = Binomial(n, p)', arrApprox(pb, bin, 1e-13));
}

{
  // Non-uniform probabilities. Mean of Poisson-binomial = Σ p_i.
  const probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  const expectedMean = probs.reduce((s, p) => s + p, 0);
  const expectedVar  = probs.reduce((s, p) => s + p * (1 - p), 0);
  const pb = poissonBinomialPMF(probs);
  check('PoissonBinomial mean = Σ p_i',     approx(meanFromPMF(pb),     expectedMean, 1e-13));
  check('PoissonBinomial var  = Σ p_i(1-p_i)', approx(varianceFromPMF(pb), expectedVar, 1e-12));
  check('PoissonBinomial total mass = 1',     approx(pmfTotalMass(pb), 1, 1e-13));
}

{
  // Monte Carlo cross-check.
  const probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8];
  const pb = poissonBinomialPMF(probs);
  const N = 200_000;
  const rng = mulberry32(0xBEEF);
  const empirical = new Array<number>(probs.length + 1).fill(0);
  for (let s = 0; s < N; s++) {
    let k = 0;
    for (const p of probs) if (rng() < p) k++;
    empirical[k]++;
  }
  for (let i = 0; i < empirical.length; i++) empirical[i] /= N;
  let maxAbsDiff = 0;
  for (let i = 0; i < pb.length; i++) {
    const d = Math.abs(pb[i] - empirical[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
  }
  check(`PoissonBinomial PMF matches Monte Carlo (N=${N})`,
        maxAbsDiff < 0.005, `max |Δ| = ${maxAbsDiff.toExponential(2)}`);
}

// =============================================================================
// T3  Competing risks formula
// =============================================================================
console.log('\nT3  Competing risks: exact discrete-time first-event probabilities');

{
  // K=1: collapses to {exp(−λdt), 1−exp(−λdt)}.
  const lambda = 0.7;
  const dt = 0.5;
  const out = competingRisks([lambda], dt);
  check('K=1: p_no = exp(−λdt)', approx(out[0], Math.exp(-lambda * dt), 1e-15));
  check('K=1: p_1 = 1 − exp(−λdt)', approx(out[1], 1 - Math.exp(-lambda * dt), 1e-15));
  check('K=1: probs sum to 1', approx(out[0] + out[1], 1, 1e-15));
}

{
  // Two equal rates: each event has prob (1/2)(1 − exp(−2λdt)).
  const lambda = 0.4;
  const dt = 1.0;
  const out = competingRisks([lambda, lambda], dt);
  const Lambda = 2 * lambda;
  const expectedAny = 1 - Math.exp(-Lambda * dt);
  check('K=2 equal: p_1 = p_2 = (1/2)(1 − exp(−Λdt))',
        approx(out[1], expectedAny / 2, 1e-15) && approx(out[2], expectedAny / 2, 1e-15));
  check('K=2 equal: probs sum to 1', approx(out.reduce((s, v) => s + v, 0), 1, 1e-15));
}

{
  // Linear approximation comparison: at small Λ·dt, exact ≈ λ·dt.
  const lambdas = [0.05, 0.05, 0.05];
  const dt = 0.1;  // Λ·dt = 0.015 — linear approximation should match.
  const exact = competingRisks(lambdas, dt);
  for (let i = 0; i < lambdas.length; i++) {
    check(`small Λ·dt: exact ≈ linear for λ_${i+1}`,
      approx(exact[i + 1], lambdas[i] * dt, 0.001));
  }
}

{
  // Monte Carlo: simulate competing exponentials.
  const lambdas = [0.5, 1.0, 0.3];
  const dt = 0.4;
  const exact = competingRisks(lambdas, dt);
  const N = 100_000;
  const rng = mulberry32(0xFEED);
  const cnt = new Array<number>(lambdas.length + 1).fill(0);
  for (let s = 0; s < N; s++) {
    // First-event simulation: exponential time of each, take the min.
    let minT = Infinity, who = -1;
    for (let i = 0; i < lambdas.length; i++) {
      const u = 1 - rng();
      const time = -Math.log(u) / lambdas[i];
      if (time < minT) { minT = time; who = i; }
    }
    if (minT > dt) cnt[0]++;
    else            cnt[who + 1]++;
  }
  let maxDiff = 0;
  for (let i = 0; i < cnt.length; i++) {
    const e = cnt[i] / N;
    const d = Math.abs(e - exact[i]);
    if (d > maxDiff) maxDiff = d;
  }
  check(`competingRisks matches first-event Monte Carlo (N=${N})`,
        maxDiff < 0.005, `max |Δ| = ${maxDiff.toExponential(2)}`);
}

// =============================================================================
// T4  PMF utilities
// =============================================================================
console.log('\nT4  PMF utilities');

{
  const skewed = [0.5, 1.5, 2, 1, 0];  // sum = 5
  const norm = normalizePMF(skewed);
  check('normalizePMF total mass = 1', approx(pmfTotalMass(norm), 1, 1e-15));
  check('normalizePMF preserves ratios',
    approx(norm[1] / norm[0], skewed[1] / skewed[0], 1e-15));
}

{
  // sampleCategorical histogram matches input probs.
  const probs = [0.1, 0.2, 0.3, 0.25, 0.15];
  const N = 50_000;
  const rng = mulberry32(0xC0DE);
  const cnt = new Array<number>(probs.length).fill(0);
  for (let s = 0; s < N; s++) cnt[sampleCategorical(probs, rng)]++;
  let maxDiff = 0;
  for (let i = 0; i < probs.length; i++) {
    const d = Math.abs(cnt[i] / N - probs[i]);
    if (d > maxDiff) maxDiff = d;
  }
  check(`sampleCategorical matches input distribution (N=${N})`,
    maxDiff < 0.01, `max |Δ| = ${maxDiff.toFixed(4)}`);
}

// =============================================================================
// T5  Convolveself with repeated squaring matches naive iterative convolution
// =============================================================================
console.log('\nT5  Self-convolution via repeated squaring matches iterative convolution');

{
  const base = [0.2, 0.5, 0.3];
  for (const n of [0, 1, 2, 5, 13, 64]) {
    const expectedArr: ReadonlyArray<number>[] = [];
    for (let i = 0; i < n; i++) expectedArr.push(base);
    const expected = discreteConvolveMany(expectedArr);
    const got = discreteConvolveSelf(base, n);
    if (n === 0) {
      check(`self-conv n=0 = [1] (identity)`, arrApprox(got, [1], 1e-15));
    } else {
      check(`self-conv n=${n} matches naive iterative`,
        arrApprox(got, expected, 1e-13));
    }
  }
}

// =============================================================================
// T6  Continuous samplers: Poisson, Exponential, Gamma
// =============================================================================
console.log('\nT6  Continuous distribution samplers');

{
  // Poisson(λ): mean = λ, variance = λ.
  const rng = mulberry32(0xCAFE);
  for (const lambda of [0.7, 5, 30, 100]) {
    const N = 50_000;
    let s = 0, s2 = 0;
    for (let i = 0; i < N; i++) {
      const x = samplePoisson(lambda, rng);
      s += x; s2 += x * x;
    }
    const m = s / N;
    const v = s2 / N - m * m;
    const tolMean = lambda < 30 ? 0.05 : 0.5;  // larger tol for normal approx body
    const tolVar  = lambda * 0.10;
    check(`Poisson(${lambda}) mean ≈ λ`,     approx(m, lambda, tolMean), `got ${m.toFixed(3)}`);
    check(`Poisson(${lambda}) variance ≈ λ`, approx(v, lambda, tolVar),  `got ${v.toFixed(3)}`);
  }
}

{
  // Exponential(rate): mean = 1/rate, variance = 1/rate².
  const rng = mulberry32(0xC001);
  const rate = 2.5;
  const N = 50_000;
  let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) {
    const x = sampleExponential(rate, rng);
    s += x; s2 += x * x;
  }
  const m = s / N;
  const v = s2 / N - m * m;
  check('Exponential mean ≈ 1/rate',     approx(m, 1 / rate, 0.01),       `got ${m.toFixed(4)}`);
  check('Exponential var  ≈ 1/rate²',    approx(v, 1 / (rate * rate), 0.05), `got ${v.toFixed(4)}`);
}

{
  // Gamma(shape, scale): mean = shape·scale, variance = shape·scale².
  const rng = mulberry32(0xB055);
  for (const [shape, scale] of [[2, 1.5], [0.5, 2], [10, 0.3]]) {
    const N = 50_000;
    let s = 0, s2 = 0;
    for (let i = 0; i < N; i++) {
      const x = sampleGamma(shape, scale, rng);
      s += x; s2 += x * x;
    }
    const expectedMean = shape * scale;
    const expectedVar  = shape * scale * scale;
    const m = s / N;
    const v = s2 / N - m * m;
    check(`Gamma(${shape}, ${scale}) mean`, approx(m, expectedMean, expectedMean * 0.03), `got ${m.toFixed(3)}`);
    check(`Gamma(${shape}, ${scale}) var`,  approx(v, expectedVar,  expectedVar  * 0.10), `got ${v.toFixed(3)}`);
  }
}

console.log('\nsummary: ' + pass + ' pass, ' + fail + ' fail');
process.exit(fail === 0 ? 0 : 1);
