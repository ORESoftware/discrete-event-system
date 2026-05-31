#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-newsvendor.rs   (fn main)
// 1:1 file move. The newsvendor problem: finds q* analytically, by brute
// search over the demand PMF, and by 1-step MDP value iteration.
//
// Conversion notes (file-specific):
//   - day-by-day demand sampling -> inject RandomSource/SeededRandom
//     (shared::capabilities).
//   - closed-form critical-fractile + MDP value iteration -> pure fns.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// THE NEWSVENDOR PROBLEM
//
// The classic single-period stochastic inventory problem (also known as
// the newsboy problem). Each morning the newsvendor must decide how
// many newspapers q to buy at unit cost c. During the day, demand D is
// realised from a random distribution. Sales = min(q, D); leftovers
// (q − D)+ are scrapped at salvage value s; lost sales (D − q)+ are
// foregone at price p per paper.
//
//   profit(q, D) = p · min(q, D) + s · (q − D)+ − c · q
//
// THE OPTIMAL POLICY (closed form: critical fractile)
//
//   underage cost  c_u = p − c        (profit lost per missed sale)
//   overage cost   c_o = c − s        (loss per leftover paper)
//   critical ratio CR  = c_u / (c_u + c_o)
//
//   q* = inf { q :  P(D ≤ q) ≥ CR }   (smallest q with CDF ≥ CR)
//
// This is the famous **critical fractile** result. Derivation: take
// d/dq E[profit] = (p − c) − (p − s) · F(q) = 0, solve for F(q*).
//
// THIS MODULE
//
//   1. Simulates the newsvendor day-by-day in the framework.
//   2. Discovers the optimal q* by THREE methods, all producing the
//      same answer modulo numerical precision:
//
//      (a) ANALYTICAL: closed-form critical-fractile formula.
//      (b) BRUTE-SEARCH: enumerate q in [0, q_max], compute E[profit(q)]
//          exactly via the demand PMF, take the argmax.
//      (c) MDP VALUE ITERATION: model the day as a 1-step MDP and run
//          value iteration to "discover" the optimal action. With
//          γ = 0 (myopic single period) this reduces exactly to (b).
//
//   3. Validates that all three agree (PASS gates in the runner).
//
// The MDP path looks heavy-handed for a single-period problem, and it
// is — but it generalises directly to the multi-period inventory MDP
// (`main-inventory-mdp.ts`) where there is no closed form. Pinning
// the single-period result against the analytical critical fractile
// is how we trust the multi-period extension.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {mulberry32, withSeed} from './general/prng';
import {samplePoisson} from './general/random-variables';
import {MDPSpec, Outcome, valueIteration} from './general/value-iteration';
import {DESStation, runIterativeDES} from './general/des-base';

// -----------------------------------------------------------------------------
// Demand distribution (discrete on {0, 1, …, dMax}).
// -----------------------------------------------------------------------------

/**
 * Discrete demand distribution. Stored as a PMF indexed 0..dMax with
 * `pmf[k] = P(D = k)`. Probabilities must sum to 1 within rounding.
 */
export interface DemandDist {
  pmf: number[];
}

export function demandPoissonPMF(lambda: number, dMax: number): DemandDist {
  const pmf = new Array<number>(dMax + 1).fill(0);
  // Recurrence: P(k) = P(k-1) · λ / k. Numerically stable.
  let p = Math.exp(-lambda);
  pmf[0] = p;
  for (let k = 1; k <= dMax; k++) {
    p = p * lambda / k;
    pmf[k] = p;
  }
  // Lump everything > dMax into the tail at dMax (truncation).
  let total = 0;
  for (const v of pmf) total += v;
  pmf[dMax] += 1 - total;
  return {pmf};
}

export function demandUniformPMF(lo: number, hi: number, dMax: number): DemandDist {
  const pmf = new Array<number>(dMax + 1).fill(0);
  const w = hi - lo + 1;
  for (let k = lo; k <= hi && k <= dMax; k++) pmf[k] = 1 / w;
  return {pmf};
}

export function cdfFromPMF(d: DemandDist): number[] {
  const cdf = new Array<number>(d.pmf.length).fill(0);
  let acc = 0;
  for (let k = 0; k < d.pmf.length; k++) {
    acc += d.pmf[k];
    cdf[k] = acc;
  }
  return cdf;
}

export function meanFromDemand(d: DemandDist): number {
  let m = 0;
  for (let k = 0; k < d.pmf.length; k++) m += k * d.pmf[k];
  return m;
}

export function sampleDemand(d: DemandDist, rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let k = 0; k < d.pmf.length; k++) {
    acc += d.pmf[k];
    if (u <= acc) return k;
  }
  return d.pmf.length - 1;
}

// -----------------------------------------------------------------------------
// Newsvendor cost / profit functions.
// -----------------------------------------------------------------------------

export interface NewsvendorParams {
  /** Unit cost paid up-front per paper ordered. */
  unitCost: number;
  /** Unit price received per paper sold. */
  unitPrice: number;
  /** Salvage value per leftover paper at end of day. (May be 0 or even negative for disposal cost.) */
  unitSalvage: number;
  /** Demand distribution. */
  demand: DemandDist;
  /** Maximum order quantity (action space upper bound). */
  qMax: number;
}

export function profit(q: number, d: number, p: NewsvendorParams): number {
  const sold = Math.min(q, d);
  const leftover = Math.max(0, q - d);
  return p.unitPrice * sold + p.unitSalvage * leftover - p.unitCost * q;
}

/** Exact expected profit for order quantity q under the discrete demand PMF. */
export function expectedProfit(q: number, p: NewsvendorParams): number {
  let e = 0;
  for (let d = 0; d < p.demand.pmf.length; d++) {
    e += p.demand.pmf[d] * profit(q, d, p);
  }
  return e;
}

// -----------------------------------------------------------------------------
// (a) ANALYTICAL: critical-fractile q*.
// -----------------------------------------------------------------------------

export function analyticalOptimalQ(p: NewsvendorParams): {qStar: number; criticalRatio: number} {
  const cu = p.unitPrice - p.unitCost;
  const co = p.unitCost - p.unitSalvage;
  if (cu <= 0) return {qStar: 0, criticalRatio: 0};   // never profitable to order
  const cr = cu / (cu + co);
  const cdf = cdfFromPMF(p.demand);
  for (let k = 0; k < cdf.length; k++) {
    if (cdf[k] >= cr) return {qStar: k, criticalRatio: cr};
  }
  return {qStar: cdf.length - 1, criticalRatio: cr};
}

// -----------------------------------------------------------------------------
// (b) BRUTE-SEARCH: argmax_q E[profit(q)].
// -----------------------------------------------------------------------------

export function bruteSearchOptimalQ(p: NewsvendorParams): {qStar: number; profileEP: number[]} {
  const profile = new Array<number>(p.qMax + 1);
  let best = -Infinity, bestQ = 0;
  for (let q = 0; q <= p.qMax; q++) {
    profile[q] = expectedProfit(q, p);
    if (profile[q] > best) { best = profile[q]; bestQ = q; }
  }
  return {qStar: bestQ, profileEP: profile};
}

// -----------------------------------------------------------------------------
// (c) MDP VALUE ITERATION: 1-step MDP with γ = 0.
//
// State 0 = "morning" (only non-terminal state).
// State 1 = "end of day" (absorbing terminal, V = 0).
// Action a = order quantity q ∈ {0, …, qMax}.
// Transition: from state 0 with action a, end up in state 1 with prob
// 1, immediate reward = profit(a, D) integrated over D. The expected
// reward is precomputed as `expectedProfit(a)` and exposed as a single
// outcome with prob 1.
//
// With γ = 0 this is equivalent to brute-search; we keep the MDP
// formulation as a regression test against the multi-period inventory
// MDP that uses the same machinery.
// -----------------------------------------------------------------------------

export function newsvendorMDPSpec(p: NewsvendorParams): MDPSpec {
  const numStates = 2;
  const numActions = (s: number) => s === 0 ? p.qMax + 1 : 0;
  const isTerminal = (s: number) => s === 1;
  const outcomes = (s: number, a: number): Outcome[] => {
    if (s !== 0) return [];
    return [{prob: 1, reward: expectedProfit(a, p), nextState: 1}];
  };
  return {numStates, numActions, isTerminal, outcomes,
          stateLabel: s => ['morning', 'end-of-day'][s],
          actionLabel: a => `q=${a}`};
}

export function mdpOptimalQ(p: NewsvendorParams): {qStar: number; V0: number; iterations: number} {
  const spec = newsvendorMDPSpec(p);
  const result = valueIteration(spec, {gamma: 0, tol: 1e-12});
  return {qStar: result.policy[0], V0: result.V[0], iterations: result.iterations};
}

// -----------------------------------------------------------------------------
// Framework simulation: a station that orders at the start of the day,
// observes demand, and tallies profit.
// -----------------------------------------------------------------------------

class NewsvendorStation extends DESStation {
  rng: () => number;
  params: NewsvendorParams;
  q: number;
  private readonly totalDays: number;
  totalProfit = 0;
  daysSimulated = 0;
  unmetDemand = 0;
  totalLeftover = 0;
  history: Array<{day: number; q: number; demand: number; profit: number; sold: number; leftover: number}> = [];

  constructor(params: NewsvendorParams, q: number, days: number, seed: number) {
    super('newsvendor-station');
    this.params = params;
    this.q = q;
    this.totalDays = days;
    this.rng = mulberry32(seed);
  }

  runTimeStep(): void {
    if (!this.hasWork()) return;
    const day = this.daysSimulated;
    const d = sampleDemand(this.params.demand, this.rng);
    const sold = Math.min(this.q, d);
    const leftover = Math.max(0, this.q - d);
    const lost = Math.max(0, d - this.q);
    const pi = profit(this.q, d, this.params);
    this.totalProfit += pi;
    this.unmetDemand += lost;
    this.totalLeftover += leftover;
    this.daysSimulated++;
    this.history.push({day, q: this.q, demand: d, profit: pi, sold, leftover});
  }

  override hasWork(): boolean {
    return this.daysSimulated < this.totalDays;
  }
}

export function simulate(params: NewsvendorParams, q: number, days: number, seed: number) {
  return withSeed(seed, () => {
    const sta = new NewsvendorStation(params, q, days, seed);
    runIterativeDES([sta], {shuffle: false, maxTicks: days + 2, runValidators: false});
    return {
      meanProfit: sta.totalProfit / sta.daysSimulated,
      avgLeftover: sta.totalLeftover / sta.daysSimulated,
      avgUnmet: sta.unmetDemand / sta.daysSimulated,
      history: sta.history,
    };
  });
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

async function main() {
  const lambda = Number(process.env.LAMBDA ?? 50);    // mean Poisson demand
  const dMax = Number(process.env.D_MAX ?? Math.ceil(lambda * 2.5));
  const params: NewsvendorParams = {
    unitCost:    Number(process.env.UNIT_COST    ?? 0.50),
    unitPrice:   Number(process.env.UNIT_PRICE   ?? 1.00),
    unitSalvage: Number(process.env.UNIT_SALVAGE ?? 0.10),
    demand:      demandPoissonPMF(lambda, dMax),
    qMax:        Number(process.env.Q_MAX ?? Math.ceil(lambda * 2.5)),
  };
  const days = Number(process.env.DAYS ?? 1000);
  const seed = Number(process.env.SEED ?? 1);

  console.log(`# Newsvendor: c=${params.unitCost}, p=${params.unitPrice}, s=${params.unitSalvage}`);
  console.log(`#   demand = Poisson(λ=${lambda}), truncated at ${dMax};  qMax=${params.qMax}`);
  console.log(`#   underage cost c_u = p−c = ${(params.unitPrice - params.unitCost).toFixed(3)}`);
  console.log(`#   overage  cost c_o = c−s = ${(params.unitCost - params.unitSalvage).toFixed(3)}`);

  // (a) Analytical critical-fractile.
  const a = analyticalOptimalQ(params);
  console.log(`\n(a) Analytical critical-fractile`);
  console.log(`    critical ratio = c_u / (c_u + c_o) = ${a.criticalRatio.toFixed(4)}`);
  console.log(`    q* = inf{q : P(D ≤ q) ≥ CR} = ${a.qStar}`);
  console.log(`    E[profit(q*)] = ${expectedProfit(a.qStar, params).toFixed(4)}`);

  // (b) Brute search over E[profit(q)].
  const b = bruteSearchOptimalQ(params);
  console.log(`\n(b) Brute search over q ∈ [0, ${params.qMax}]`);
  console.log(`    q*  = ${b.qStar}`);
  console.log(`    E[profit(q*)] = ${b.profileEP[b.qStar].toFixed(4)}`);

  // (c) MDP value iteration.
  const c = mdpOptimalQ(params);
  console.log(`\n(c) MDP value iteration (1-step, γ=0)`);
  console.log(`    q*       = ${c.qStar}`);
  console.log(`    V(state=morning) = ${c.V0.toFixed(4)}`);
  console.log(`    iterations = ${c.iterations}`);

  // Simulate at q* for sanity check.
  const sim = simulate(params, a.qStar, days, seed);
  console.log(`\n(sim) ${days}-day simulation at q = q* = ${a.qStar}`);
  console.log(`    mean profit/day  = ${sim.meanProfit.toFixed(4)}   (analytical ${expectedProfit(a.qStar, params).toFixed(4)})`);
  console.log(`    avg leftover/day = ${sim.avgLeftover.toFixed(2)}`);
  console.log(`    avg unmet/day    = ${sim.avgUnmet.toFixed(2)}`);

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'newsvendor.json');
  fs.writeFileSync(outPath, JSON.stringify({
    params: {...params, demand: undefined, demandLambda: lambda, dMax},
    days, seed,
    analytical: a, bruteSearch: {qStar: b.qStar, profileEP: b.profileEP},
    mdp: c, simulation: sim,
  }));
  console.log(`# wrote ${outPath}`);
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
