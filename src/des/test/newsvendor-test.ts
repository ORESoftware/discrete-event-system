#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// Unit tests for the newsvendor and inventory MDP modules.
//
// We test the building blocks (PMF construction, profit / E[profit],
// optimal-q methods) against analytic identities and cross-method
// agreement. Validation against external (Python) and end-to-end
// behaviour lives in `runners/validate-newsvendor.ts`.
// =============================================================================

import {
  analyticalOptimalQ, bruteSearchOptimalQ, cdfFromPMF, demandPoissonPMF,
  demandUniformPMF, expectedProfit, mdpOptimalQ, NewsvendorParams,
  meanFromDemand, profit,
} from '../main-newsvendor';
import {detectPolicyStructure, inventoryMDPSpec, InventoryParams} from '../main-inventory-mdp';
import {valueIteration} from '../general/value-iteration';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (ok) { pass++; console.log(`  PASS    ${label}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}
const approx = (a: number, b: number, t: number) => Math.abs(a - b) <= t;

// =============================================================================
console.log('\nT1  Demand PMFs: total mass and mean');
{
  const lam = 25;
  const d = demandPoissonPMF(lam, 100);
  const total = d.pmf.reduce((s, v) => s + v, 0);
  check('Poisson PMF sums to 1', approx(total, 1, 1e-12), `total=${total}`);
  check('Poisson PMF mean ≈ λ', approx(meanFromDemand(d), lam, 1e-3), `mean=${meanFromDemand(d).toFixed(4)}`);
}
{
  const d = demandUniformPMF(5, 15, 30);
  const total = d.pmf.reduce((s, v) => s + v, 0);
  check('Uniform PMF sums to 1', approx(total, 1, 1e-12));
  check('Uniform PMF mean = (lo+hi)/2', approx(meanFromDemand(d), 10, 1e-12));
}
{
  const d = demandPoissonPMF(20, 60);
  const cdf = cdfFromPMF(d);
  check('CDF is monotonic non-decreasing', cdf.every((v, i) => i === 0 || v >= cdf[i - 1]));
  check('CDF ends at ≈1', approx(cdf[cdf.length - 1], 1, 1e-12), `last=${cdf[cdf.length - 1]}`);
}

// =============================================================================
console.log('\nT2  profit() identities');
{
  const p: NewsvendorParams = {
    unitCost: 0.5, unitPrice: 1.0, unitSalvage: 0.1,
    demand: demandPoissonPMF(10, 30), qMax: 30,
  };
  // q = 0 ⇒ no revenue, no salvage, no cost; profit = 0.
  check('profit(0, d) = 0 for any d', [0, 5, 10, 30].every(d => profit(0, d, p) === 0));
  // d = 0, q > 0 ⇒ -(c·q) + s·q = (s-c)·q  (worst case, all salvaged at loss).
  for (const q of [1, 5, 20]) {
    const expected = (p.unitSalvage - p.unitCost) * q;
    check(`profit(${q}, 0) = (s−c)·q = ${expected.toFixed(2)}`,
          approx(profit(q, 0, p), expected, 1e-12),
          `got ${profit(q, 0, p)}`);
  }
  // d ≥ q ⇒ profit = (p − c)·q.
  for (const q of [1, 5, 20]) {
    const expected = (p.unitPrice - p.unitCost) * q;
    check(`profit(${q}, d≥q) = (p−c)·q = ${expected.toFixed(2)}`,
          approx(profit(q, q + 5, p), expected, 1e-12));
  }
}

// =============================================================================
console.log('\nT3  Critical-fractile identity at common ratios');
{
  // CR = 0.5 (equal underage and overage costs) ⇒ q* = median of D.
  const lam = 30;
  const params: NewsvendorParams = {
    unitCost: 0.5,
    unitPrice: 1.0,                 // c_u = 0.5
    unitSalvage: 0.0,               // c_o = 0.5  →  CR = 0.5
    demand: demandPoissonPMF(lam, 100),
    qMax: 100,
  };
  const a = analyticalOptimalQ(params);
  const cdf = cdfFromPMF(params.demand);
  // Median = inf{k : CDF(k) ≥ 0.5}.
  let median = 0;
  for (let k = 0; k < cdf.length; k++) {
    if (cdf[k] >= 0.5) { median = k; break; }
  }
  check('CR = 0.5 ⇒ q* = median of demand',
        a.qStar === median,
        `q*=${a.qStar} median=${median}`);
}
{
  // CR → 1 (huge margin, no salvage cost) ⇒ q* well above the mean demand.
  // For Poisson(λ=10) the tail at q ≈ λ + 3·√λ ≈ 19 is already very thin,
  // so q* should be ≥ λ + 2·√λ ≈ 16. We can't expect q* ≥ qMax − 1 because
  // CDF(qMax) ≈ 1 already; q* is bounded by the distribution's support.
  const params: NewsvendorParams = {
    unitCost: 0.01, unitPrice: 100.0, unitSalvage: 0.0,
    demand: demandPoissonPMF(10, 30), qMax: 30,
  };
  const a = analyticalOptimalQ(params);
  const lam = 10;
  const lower = Math.ceil(lam + 2 * Math.sqrt(lam));
  check(`high-margin q* > λ + 2√λ = ${lower}`,
        a.qStar >= lower, `q*=${a.qStar}, threshold=${lower}`);
}
{
  // CR → 0 ⇒ q* = 0 (don't stock at all).
  const params: NewsvendorParams = {
    unitCost: 1.0, unitPrice: 1.05, unitSalvage: -10.0,  // huge overage cost
    demand: demandPoissonPMF(10, 30), qMax: 30,
  };
  const a = analyticalOptimalQ(params);
  check('huge overage cost ⇒ q* small', a.qStar <= 4, `q*=${a.qStar}`);
}

// =============================================================================
console.log('\nT4  Three solution methods agree');
const SCEN: NewsvendorParams[] = [
  {unitCost: 0.4, unitPrice: 1.0, unitSalvage: 0.1, demand: demandPoissonPMF(15, 50),  qMax: 50},
  {unitCost: 0.7, unitPrice: 1.5, unitSalvage: 0.3, demand: demandPoissonPMF(40, 120), qMax: 120},
  {unitCost: 0.2, unitPrice: 0.8, unitSalvage: 0.0, demand: demandUniformPMF(0, 10, 20), qMax: 20},
];
for (let i = 0; i < SCEN.length; i++) {
  const params = SCEN[i];
  const a = analyticalOptimalQ(params);
  const b = bruteSearchOptimalQ(params);
  const m = mdpOptimalQ(params);
  check(`scenario ${i + 1}: analytical ≡ brute`,    a.qStar === b.qStar, `${a.qStar} vs ${b.qStar}`);
  check(`scenario ${i + 1}: analytical ≡ MDP`,      a.qStar === m.qStar, `${a.qStar} vs ${m.qStar}`);
  check(`scenario ${i + 1}: E[profit] all agree`,
        approx(expectedProfit(a.qStar, params), b.profileEP[b.qStar], 1e-12) &&
        approx(expectedProfit(a.qStar, params), m.V0, 1e-12));
}

// =============================================================================
console.log('\nT5  Inventory MDP: structural sanity');
{
  // No fixed cost ⇒ base-stock; structure detector identifies it.
  const params: InventoryParams = {
    xMax: 30, aMax: 30,
    demand: demandPoissonPMF(10, 30),
    unitCost: 1, fixedCost: 0, unitPrice: 2,
    holdCost: 0.1, lostCost: 0.5, gamma: 0.9,
  };
  const r = valueIteration(inventoryMDPSpec(params), {gamma: params.gamma});
  const policy = Array.from(r.policy).map(v => Math.max(0, v));
  const struct = detectPolicyStructure(policy);
  check('no fixed cost ⇒ base-stock detected', struct.kind === 'base-stock');
  check('S* in (0, xMax]', struct.S > 0 && struct.S <= params.xMax);
}
{
  // Positive fixed cost ⇒ (s, S); sufficient discount and demand variability.
  const params: InventoryParams = {
    xMax: 50, aMax: 50,
    demand: demandPoissonPMF(15, 40),
    unitCost: 1, fixedCost: 20, unitPrice: 2,
    holdCost: 0.1, lostCost: 0.5, gamma: 0.95,
  };
  const r = valueIteration(inventoryMDPSpec(params), {gamma: params.gamma});
  const policy = Array.from(r.policy).map(v => Math.max(0, v));
  const struct = detectPolicyStructure(policy);
  check('fixed cost ⇒ (s, S) detected', struct.kind === 's-S');
  check('S − s ≥ 2 with significant K', struct.S - struct.reorderPoint >= 2,
        `S=${struct.S} s=${struct.reorderPoint}`);
}

// =============================================================================
console.log('\nT6  Value-iteration determinism and convergence');
{
  // Same MDP spec → identical V and policy.
  const params: InventoryParams = {
    xMax: 20, aMax: 20,
    demand: demandPoissonPMF(8, 20),
    unitCost: 1, fixedCost: 5, unitPrice: 2,
    holdCost: 0.1, lostCost: 0.5, gamma: 0.9,
  };
  const r1 = valueIteration(inventoryMDPSpec(params), {gamma: params.gamma, tol: 1e-9});
  const r2 = valueIteration(inventoryMDPSpec(params), {gamma: params.gamma, tol: 1e-9});
  let same = true;
  for (let s = 0; s < r1.V.length; s++) {
    if (Math.abs(r1.V[s] - r2.V[s]) > 1e-15) { same = false; break; }
  }
  check('value iteration deterministic across calls', same);
  check('converges within tolerance',                 r1.finalDelta < 1e-9);
  check('iterations < maxIter (no truncation)',       r1.iterations < 5000);
}

// =============================================================================
console.log('\nsummary: ' + pass + ' pass, ' + fail + ' fail');
process.exit(fail === 0 ? 0 : 1);
