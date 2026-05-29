'use strict';

// Tiny stats helpers: mean, sample variance, Welch's t-test, and a normal-
// approximation two-sided p-value good enough for n=30.

export function mean(xs: number[]): number {
  if (xs.length === 0) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

export function sampleVariance(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) {
    const d = x - m;
    s += d * d;
  }
  return s / (xs.length - 1);
}

export function stddev(xs: number[]): number {
  return Math.sqrt(sampleVariance(xs));
}

/**
 * Welch's t-test. Returns t-statistic and degrees of freedom.
 * The two-sided p-value is approximated using a normal CDF via erfc; for
 * df > 30 this is within a couple of percent of the true Student-t value.
 */
export interface WelchResult {
  meanA: number;
  meanB: number;
  varA: number;
  varB: number;
  nA: number;
  nB: number;
  t: number;
  df: number;
  pValueTwoSided: number;
  reject95: boolean;  // |t| > 1.96 (normal approx)
  reject99: boolean;  // |t| > 2.58
}

export function welch(a: number[], b: number[]): WelchResult {
  const mA = mean(a);
  const mB = mean(b);
  const vA = sampleVariance(a);
  const vB = sampleVariance(b);
  const nA = a.length;
  const nB = b.length;
  const seSq = vA / nA + vB / nB;
  const t = seSq > 0 ? (mA - mB) / Math.sqrt(seSq) : 0;
  const df = seSq > 0
    ? (seSq * seSq) /
      (((vA / nA) ** 2) / Math.max(1, nA - 1) + ((vB / nB) ** 2) / Math.max(1, nB - 1))
    : 1;
  const pValueTwoSided = seSq > 0 ? 2 * (1 - normalCdf(Math.abs(t))) : 1;
  return {
    meanA: mA, meanB: mB, varA: vA, varB: vB, nA, nB,
    t, df, pValueTwoSided,
    reject95: Math.abs(t) > 1.96,
    reject99: Math.abs(t) > 2.58,
  };
}

// Standard-normal CDF using Abramowitz-Stegun erfc approximation.
function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26, max abs error 1.5e-7.
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}
