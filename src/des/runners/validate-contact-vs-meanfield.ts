#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/validate_contact_vs_meanfield.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace process.exit with ExitCode.
// - Convert scenario/check/result records to nominal structs and keep numerical tolerances as named constants.
// - Pure comparison/math helpers remain private module functions unless a reusable mean-field transform trait emerges.
'use strict';

// =============================================================================
// Validate the contact-SEIR kernels against each other and against the
// mass-action mean-field. Three studies in one file:
//
//   STUDY 1: Convergence as N → ∞.
//     Run mass-action and pairwise on the same parameters (homogeneous
//     contacts) at increasing N. The two should agree on:
//       (a) the deterministic mean attack rate (Welch t-test, p > 0.05),
//       (b) the deterministic mean R₀ from index cases.
//     The variance gap closes with N.
//
//   STUDY 2: Heterogeneity changes R₀ — pairwise kernel only.
//     Theoretical: R₀_het = R₀_hom · (1 + CV²) where CV = stdev(c) / mean(c).
//     Sweep CV ∈ {0, 0.5, 1, 2}. Compare measured index-case R₀ to the
//     theoretical scaled value.
//
//   STUDY 3: Triplet has a sharp threshold — pairwise vs triplet.
//     Sweep initial-I seeds at fixed parameters. Pairwise spreads at
//     I₀ ≪ N, triplet needs I₀/N above some critical threshold to
//     ignite. Show the attack-rate curve has a step shape for triplet.
//
// Reads no external files. Run with:
//   npm run build
//   node dist/des/runners/validate-contact-vs-meanfield.js
// =============================================================================

import {ContactSEIRParams, runContactSEIR} from '../main-contact-seir';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (ok) { pass++; console.log(`  PASS    ${label}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}

function mean(xs: number[]): number {
  return xs.reduce((s, v) => s + v, 0) / xs.length;
}
function variance(xs: number[]): number {
  const m = mean(xs);
  return xs.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, xs.length - 1);
}
function welch(xs: number[], ys: number[]): {t: number; p: number} {
  const mx = mean(xs), my = mean(ys);
  const vx = variance(xs), vy = variance(ys);
  const se = Math.sqrt(vx / xs.length + vy / ys.length);
  const t = se === 0 ? 0 : (mx - my) / se;
  // Two-sided p via normal approximation.
  const z = Math.abs(t);
  const phi = 0.5 * (1 + erf(z / Math.SQRT2));
  return {t, p: 2 * (1 - phi)};
}
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const tt = 1 / (1 + p * x);
  const y = 1 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * Math.exp(-x * x);
  return sign * y;
}

const baseline: Omit<ContactSEIRParams, 'N' | 'kernel' | 'seed'> = {
  initialI: 10,
  contactRate: 6,
  contactRateCV: 0,
  pTransmit: 0.05,
  sigma: 1 / 5.2,
  gamma: 1 / 7.0,
  simT: 120,
  stepSize: 0.1,
};

// =============================================================================
console.log('\nStudy 1  Convergence: mass-action ≡ pairwise as N → ∞');
console.log('==========================================================================');

{
  const REPS = 12;
  for (const N of [500, 2000, 5000]) {
    const massAttack: number[] = [];
    const pairAttack: number[] = [];
    const massR0: number[] = [];
    const pairR0: number[] = [];
    for (let r = 0; r < REPS; r++) {
      const seed = 1 + r;
      const massA = runContactSEIR({...baseline, N, kernel: 'mass-action', seed});
      const pairA = runContactSEIR({...baseline, N, kernel: 'pairwise',    seed});
      massAttack.push(massA.finalAttackRate);
      pairAttack.push(pairA.finalAttackRate);
      massR0.push(massA.R0_indexOnly);
      pairR0.push(pairA.R0_indexOnly);
    }
    const tAttack = welch(massAttack, pairAttack);
    const tR0     = welch(massR0,     pairR0);
    const aMass = mean(massAttack), aPair = mean(pairAttack);
    const r0Mass = mean(massR0),    r0Pair = mean(pairR0);
    console.log(`  N=${N.toString().padStart(5)}  ` +
      `attack: mass=${(aMass*100).toFixed(1)}% pair=${(aPair*100).toFixed(1)}%  Welch p=${tAttack.p.toFixed(3)}    ` +
      `R₀(idx): mass=${r0Mass.toFixed(2)} pair=${r0Pair.toFixed(2)}  Welch p=${tR0.p.toFixed(3)}`);
    // Tighter tolerance at larger N (variance shrinks like 1/N).
    const tolP = N >= 2000 ? 0.05 : 0.01;
    check(`N=${N}: attack-rate Welch p > ${tolP}`, tAttack.p > tolP, `p=${tAttack.p.toFixed(3)}`);
    check(`N=${N}: R₀(index) Welch p > ${tolP}`,    tR0.p     > tolP, `p=${tR0.p.toFixed(3)}`);
  }
}

// =============================================================================
console.log('\nStudy 2  Heterogeneity: super-spreader effect (Gini coefficient of offspring)');
console.log('==========================================================================');
console.log('  Theoretical: with heterogeneous contact rates, a small fraction of cases');
console.log('  produces a large fraction of secondary infections — the 20/80 rule.');
console.log('  Mean-field (mass-action) cannot reproduce this because infectors are');
console.log('  selected uniformly at random; the offspring distribution stays Poisson');
console.log('  regardless of CV. Symmetric pairwise reproduces it because high-c');
console.log('  individuals BOTH initiate more contacts AND are partner more often,');
console.log('  multiplicatively increasing their per-individual offspring count.');
console.log('  We measure: Gini coefficient of offspring distribution, and "share of');
console.log('  secondaries from top 20% of cases".');

function gini(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  const n = sorted.length;
  const s = sorted.reduce((acc, v) => acc + v, 0);
  if (s === 0) return 0;
  let cum = 0;
  for (let i = 0; i < n; i++) cum += (i + 1) * sorted[i];
  return (2 * cum) / (n * s) - (n + 1) / n;
}
function shareTopK(xs: number[], k = 0.2): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => b - a);
  const cutoff = Math.floor(xs.length * k);
  const total = sorted.reduce((s, v) => s + v, 0);
  if (total === 0) return 0;
  const top = sorted.slice(0, cutoff).reduce((s, v) => s + v, 0);
  return top / total;
}

{
  const N = 5000;
  const REPS = 20;
  console.log('  CV    pairwise Gini   pairwise top-20% share    mass-action Gini   mass-action top-20% share');
  console.log('  ────  ─────────────   ──────────────────────    ────────────────   ─────────────────────────');
  let pairwiseHighCvGini = 0;
  let massActionHighCvGini = 0;
  for (const cv of [0, 0.5, 1.0, 2.0]) {
    const ginisP: number[] = [], sharesP: number[] = [];
    const ginisM: number[] = [], sharesM: number[] = [];
    for (let r = 0; r < REPS; r++) {
      const pair = runContactSEIR({...baseline, N, initialI: 5, contactRateCV: cv, kernel: 'pairwise',    seed: 1 + r});
      const mass = runContactSEIR({...baseline, N, initialI: 5, contactRateCV: cv, kernel: 'mass-action', seed: 1 + r});
      const oP = pair.perPerson.filter(p => p.ever).map(p => p.offspring);
      const oM = mass.perPerson.filter(p => p.ever).map(p => p.offspring);
      ginisP.push(gini(oP));    sharesP.push(shareTopK(oP, 0.2));
      ginisM.push(gini(oM));    sharesM.push(shareTopK(oM, 0.2));
    }
    const gP = mean(ginisP), gM = mean(ginisM);
    const sP = mean(sharesP), sM = mean(sharesM);
    if (cv === 2.0) { pairwiseHighCvGini = gP; massActionHighCvGini = gM; }
    console.log(`  ${cv.toFixed(1).padStart(3)}   ${gP.toFixed(3)}           ` +
                `${(sP*100).toFixed(1)}%`.padStart(14) + `              ` +
                `${gM.toFixed(3)}              ` +
                `${(sM*100).toFixed(1)}%`.padStart(14));
    if (cv === 0) {
      // Homogeneous: pairwise and mass-action should produce similar Ginis.
      check(`CV=0: pairwise Gini ≈ mass-action Gini`,
            Math.abs(gP - gM) < 0.10,
            `pair=${gP.toFixed(3)} mass=${gM.toFixed(3)}`);
    } else {
      // Heterogeneous: pairwise Gini should be HIGHER than mass-action.
      check(`CV=${cv}: pairwise Gini > mass-action Gini`,
            gP > gM,
            `pair=${gP.toFixed(3)} mass=${gM.toFixed(3)}`);
    }
  }
  // At CV = 2, pairwise should show clear over-dispersion vs mass-action.
  check('CV=2: pairwise Gini > 0.6 (heavy super-spreader regime)',
        pairwiseHighCvGini > 0.6, `Gini=${pairwiseHighCvGini.toFixed(3)}`);
  check('CV=2: mass-action Gini < pairwise Gini by > 0.05',
        pairwiseHighCvGini - massActionHighCvGini > 0.05,
        `pair=${pairwiseHighCvGini.toFixed(3)} mass=${massActionHighCvGini.toFixed(3)}`);
}

// =============================================================================
console.log('\nStudy 3  Triplet has a sharp threshold; pairwise does not');
console.log('==========================================================================');
console.log('  Sweep I₀ from 5 → 1000 (in N=5000) and measure final attack rate.');
console.log('  Pairwise: epidemic ignites at any I₀ ≥ 1 (linear-in-I₀ early growth).');
console.log('  Triplet: epidemic needs I₀ above a critical density (quadratic-in-I₀).');

{
  const N = 5000;
  const REPS = 6;
  // Bump contact rate × p_transmit so triplet has a chance once it ignites.
  const tripletParams = {...baseline, contactRate: 30, pTransmit: 0.05};
  const I0s = [5, 50, 200, 500, 1000];
  console.log('   I₀     I₀/N     pairwise-attack    triplet-attack');
  console.log('  ─────  ───────   ──────────────     ──────────────');
  let pairwiseAlwaysHigh = true;
  let tripletStartsLow = false;
  let tripletEndsHigh = false;
  for (const initialI of I0s) {
    const pa: number[] = [], ta: number[] = [];
    for (let r = 0; r < REPS; r++) {
      pa.push(runContactSEIR({...baseline, N, initialI, kernel: 'pairwise', seed: 1 + r}).finalAttackRate);
      ta.push(runContactSEIR({...tripletParams, N, initialI, kernel: 'triplet',  seed: 1 + r}).finalAttackRate);
    }
    const pAvg = mean(pa) * 100, tAvg = mean(ta) * 100;
    console.log(`  ${initialI.toString().padStart(5)}  ${(initialI/N).toFixed(4)}    ` +
                `${pAvg.toFixed(1)}%`.padStart(14) + `      ` +
                `${tAvg.toFixed(1)}%`.padStart(14));
    if (pAvg < 30) pairwiseAlwaysHigh = false;
    if (initialI === I0s[0]            && tAvg < 5)  tripletStartsLow = true;
    if (initialI === I0s[I0s.length-1] && tAvg > 50) tripletEndsHigh  = true;
  }
  check('pairwise attack rate > 30% for all I₀',     pairwiseAlwaysHigh);
  check('triplet attack rate < 5% at smallest I₀',   tripletStartsLow);
  check('triplet attack rate > 50% at largest I₀',   tripletEndsHigh);
}

// =============================================================================
console.log('\nsummary: ' + pass + ' pass, ' + fail + ' fail');
process.exit(fail === 0 ? 0 : 1);
