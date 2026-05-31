#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_factmachine.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// MAIN-FACTMACHINE: a POMDP model of factmachine.com.
//
// FactMachine is an *opinion market*: bettors trade YES/NO shares on the
// outcome of a future vote, and at resolution a panel of N voters casts
// votes on the same question. The bettor's payout is determined by the
// majority vote — so the hidden state is the true voter split θ ∈ [0, 1],
// not "the truth" of the underlying question. This makes it a POMDP, not
// an MDP: the bettor cannot directly observe θ, only signals correlated
// with it (price, volume, order flow, time).
//
// Reference: factmachine-mdp.pdf (in the chat thread). Key claims from that
// document we model and validate:
//   1. "Belief states converge as the market progresses"  → Brier curves
//   2. "QMDP/myopic policy outperforms uninformed policies"  → P/L ranking
//   3. "Late-stage entropy can SPIKE if late voters deviate from the
//       early-trader signal" — the non-obvious insight on page 8 of the PDF.
//   4. Oracle (knows θ) ≥ rational POMDP agent ≥ random/hold (value of info).
//
// ARCHITECTURE — fits the framework's station + movable model:
//
//   MarketStation       LMSR market maker. Holds q_yes, q_no, current price.
//   NoiseTraderStation  Each tick produces K orders; each is YES with
//                       probability  θ·informedness + 0.5·(1-informedness).
//                       So order flow is a noisy observation of θ.
//   BettorStation       Maintains b(θ) over a 21-bin discretisation. Each
//                       tick: observe order-flow ratio, Bayesian update,
//                       choose action, place trade.
//   VoterStation        At t = T, draws N votes ~ Bernoulli(θ); outcome =
//                       majority. Pays out shares accordingly.
//
//   Movables: each "order" is conceptually a movable that flows from
//   trader stations (noise traders, bettor) into the market station. We
//   model orders as direct messages (the LMSR is purely state-aggregating)
//   to keep the simulation simple.
//
// POLICIES (5):
//   'random'    pick uniformly from {hold, buy_yes, buy_no}.
//   'hold'      never trade. Best when fees > 0 and you have no edge.
//   'myopic'    1-step Bayesian: a* = argmax_a E_b[θ-p, p-θ, 0] − fee.
//   'qmdp'      QMDP via mdpValueIteration over the bettor's MDP given θ.
//   'oracle'    knows θ; trades aggressively when |θ-p| > fee.
//
// Run:
//   npm run factmachine
//   POLICY=qmdp TRUE_THETA=0.7 SEED=42 npm run factmachine
//   POLICY=myopic ANIMATE=1 npm run factmachine
//   N_REPS=200 npm run factmachine     # multi-rep summary
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {DiscreteBelief, brierScore} from './general/belief';
import {mulberry32} from './general/prng';
import {samplePoisson} from './general/random-variables';
import {Station} from './general/field-station';
import {
  bFromLiquidity as fmBFromLiquidity,
  optionPrices as fmOptionPrices,
  lmsrCost as fmLmsrCost,
  buyExecution as fmBuyExecution,
  sellExecution as fmSellExecution,
  recapitalization as fmRecapitalization,
} from './general/factmachine-math';
import type {FrameRecorder} from './animation/frame-recorder';

// -----------------------------------------------------------------------------
// Parameters.
// -----------------------------------------------------------------------------
export interface FactMachineParams {
  T: number;               // number of trading periods
  N_voters: number;        // voters at resolution
  K_noise: number;         // expected noise traders per period
  informedness: number;    // 0..1: fraction of noise traders that are informed
  fee: number;             // additive fee on each trade (in cost units)
  liquidity: number;       // LMSR liquidity parameter b
  thetaBins: number;       // discretisation count for the bettor's belief
  trueTheta: number;       // hidden truth (unknown to bettor)
  seed: number;
  policy: 'random' | 'hold' | 'myopic' | 'qmdp' | 'oracle';
  /**
   * 'bernoulli'  — terminal outcome ~ Bernoulli(θ). Cleanest POMDP demo:
   *                bettor's E[θ] is exactly the win probability they trade on.
   * 'majority'   — N voters draw Bernoulli(θ); outcome = majority. More
   *                realistic to factmachine.com but the law-of-large-numbers
   *                concentration of N voters makes win probability nearly
   *                {0, 1} once θ is far from 0.5; bettors rationally bet
   *                much more aggressively in this mode.
   */
  resolutionMode: 'bernoulli' | 'majority';
  /**
   * 'binary' — YES/NO market. Two-outcome LMSR, two-action policy.
   * 'scalar' — discretised range market with K bins covering the realised
   *            vote-fraction X = (#YES votes) / N_voters. Each bin pays $1
   *            per share if X ∈ [bin_lo, bin_hi). Forces resolutionMode =
   *            'majority' (Bernoulli outcome is degenerate with K bins).
   *            K = thetaBins for symmetry between belief and market.
   */
  marketType: 'binary' | 'scalar';
  /** Force a "coordinated late-voter surge" at t = T-2: 10× more noise
   *  traders flood in for one tick with the FLIPPED signal q' = 1 − q.
   *  This is the PDF's "late-stage entropy spike" scenario; without the
   *  surge the flip is too weak to be visible against accumulated evidence. */
  lateFlip?: boolean;
  /** Multiplier on K_noise during the late-flip tick. Default 10. */
  lateFlipMultiplier?: number;
}

export function defaultParams(overrides: Partial<FactMachineParams> = {}): FactMachineParams {
  return {
    T: 24, N_voters: 51,
    K_noise: 20, informedness: 0.6, fee: 0.01,
    liquidity: 50, thetaBins: 21,
    trueTheta: 0.65,     seed: 1,
    policy: 'qmdp',
    resolutionMode: 'bernoulli',
    marketType: 'binary',
    lateFlip: false,
    lateFlipMultiplier: 10,
    ...overrides,
  };
}

/** Probability that YES wins, given the realised θ and the resolution mode. */
function pYesWins(theta: number, params: FactMachineParams): number {
  if (params.resolutionMode === 'bernoulli') return theta;
  // Majority of N Bernoulli(θ) draws. We compute exactly via the binomial CDF.
  const N = params.N_voters;
  const half = Math.floor(N / 2);
  // P(X > half) where X ~ Binomial(N, θ).
  let p = 0;
  let logP = N * Math.log(Math.max(1e-300, 1 - theta));
  let lcoef = 0;
  for (let k = 0; k <= N; k++) {
    if (k > half) p += Math.exp(lcoef + logP);
    if (k < N) {
      lcoef += Math.log(N - k) - Math.log(k + 1);
      logP += Math.log(Math.max(1e-300, theta)) - Math.log(Math.max(1e-300, 1 - theta));
    }
  }
  return Math.max(0, Math.min(1, p));
}

// -----------------------------------------------------------------------------
// LMSR market maker (Hanson 2003), generalised to N ≥ 2 outcomes.
//
// Cost      C(q) = b · log( Σ_i exp(q_i / b) ).
// Price     p_i  = exp(q_i / b) / Σ_j exp(q_j / b).
// Bounded loss   b · log(N).
//
// PRODUCTION CONVENTION (factmachine-monorepo packages/math/src/trading/lmsr.ts)
// ─────────────────────────────────────────────────────────────────────────────
//   The user-facing parameter is "initial liquidity" L (in USDC), and the
//   cost-function `b` is derived as b = L / ln(N). For N=2 this is L/ln(2).
//   The LMSR class respects this: pass `{liquidity: L}` to follow production
//   semantics. The legacy quirk of treating `liquidity` directly as `b` is
//   still available via the `liquidityIsB: true` opt-in for backwards
//   compatibility with the original DES experiments.
// -----------------------------------------------------------------------------
export interface LMSROptions {
  /** When true, `liquidity` is interpreted as `b` directly (legacy DES
   *  semantics). When false (default), `liquidity` is the user-facing L
   *  and the internal b is derived as L / ln(N), matching production. */
  liquidityIsB?: boolean;
}

export class LMSR {
  q: number[];
  /** Internal cost-function parameter b. Always derived; never equal to
   *  the user-facing liquidity unless `liquidityIsB` is set. */
  readonly b: number;
  constructor(public liquidity: number, public N: number = 2, options: LMSROptions = {}) {
    if (N < 2) throw new Error('LMSR: need N ≥ 2 outcomes');
    if (!Number.isFinite(liquidity) || liquidity <= 0) throw new Error('LMSR: liquidity must be a finite positive number');
    this.q = new Array(N).fill(0);
    this.b = options.liquidityIsB ? liquidity : liquidity / Math.log(N);
  }
  prices(): number[] {
    const m = Math.max(...this.q) / this.b;
    const exps = this.q.map(qi => Math.exp(qi / this.b - m));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / s);
  }
  /** Convenience for the binary case: P(outcome 0) — the first column. */
  priceYes(): number { return this.prices()[0]; }
  /** Production convenience: returns {optionOne, optionTwo}. Throws unless N=2. */
  binaryPrices(): {optionOne: number; optionTwo: number} {
    if (this.N !== 2) throw new Error('LMSR.binaryPrices() requires N = 2');
    return fmOptionPrices(this.q[0], this.q[1], this.b);
  }
  cost(dq: ReadonlyArray<number>): number {
    if (dq.length !== this.N) throw new Error(`LMSR.cost: dq length ${dq.length} ≠ N=${this.N}`);
    if (this.N === 2) {
      // Production-equivalent log-sum-exp form.
      return fmLmsrCost(this.q[0] + dq[0], this.q[1] + dq[1], this.b)
           - fmLmsrCost(this.q[0],         this.q[1],         this.b);
    }
    const m0 = Math.max(...this.q) / this.b;
    const m1 = Math.max(...this.q.map((qi, i) => qi + dq[i])) / this.b;
    let s0 = 0, s1 = 0;
    for (let i = 0; i < this.N; i++) {
      s0 += Math.exp(this.q[i] / this.b - m0);
      s1 += Math.exp((this.q[i] + dq[i]) / this.b - m1);
    }
    return this.b * ((m1 + Math.log(s1)) - (m0 + Math.log(s0)));
  }
  trade(dq: ReadonlyArray<number>): number {
    const c = this.cost(dq);
    for (let i = 0; i < this.N; i++) this.q[i] += dq[i];
    return c;
  }

  // ── Production-equivalent execution helpers (binary only) ──
  /** Buy `amount` USDC of `isOptionOne ? option-1 : option-2` shares,
   *  matching the production formula `shares = b · ln(1 + (exp(amount/b) − 1)/price)`.
   *  Updates internal q. */
  buy(amount: number, isOptionOne: boolean = true, feeBps: number = 0) {
    if (this.N !== 2) throw new Error('LMSR.buy() requires N = 2');
    const exec = fmBuyExecution({
      amount, optionOneShares: this.q[0], optionTwoShares: this.q[1],
      b: this.b, feeBps, isOptionOne,
    });
    if (isOptionOne) this.q[0] += exec.shares; else this.q[1] += exec.shares;
    return exec;
  }
  /** Sell `sharesOut` shares of `isOptionOne ? option-1 : option-2`. Updates
   *  internal q. */
  sell(sharesOut: number, isOptionOne: boolean = true, feeBps: number = 0) {
    if (this.N !== 2) throw new Error('LMSR.sell() requires N = 2');
    const exec = fmSellExecution({
      sharesOut, optionOneShares: this.q[0], optionTwoShares: this.q[1],
      b: this.b, feeBps, isOptionOne,
    });
    if (isOptionOne) this.q[0] -= sharesOut; else this.q[1] -= sharesOut;
    return exec;
  }
  /** Recapitalise (add or remove liquidity); preserves prices by construction. */
  recap(newLiquidity: number, options: LMSROptions = {}): {capitalDelta: number} {
    if (this.N !== 2) throw new Error('LMSR.recap() requires N = 2');
    const newB = options.liquidityIsB ? newLiquidity : newLiquidity / Math.log(this.N);
    const r = fmRecapitalization({
      optionOneShares: this.q[0], optionTwoShares: this.q[1],
      currentB: this.b, newB,
    });
    this.q[0] = r.newOptionOneShares;
    this.q[1] = r.newOptionTwoShares;
    (this as any).b = r.newB;
    return {capitalDelta: r.capitalDelta};
  }
}

/** Convenience: convert a USDC initial-liquidity amount L to LMSR's b
 *  parameter for an N-outcome market. b = L / ln(N). */
export function liquidityToB(L: number, N: number = 2): number {
  if (N === 2) return fmBFromLiquidity(L);
  return L / Math.log(N);
}

// -----------------------------------------------------------------------------
// Bettor agent. `shares[i]` is shares held in market outcome i.
//   * binary  market: i ∈ {0=YES, 1=NO}.
//   * scalar  market: i ∈ {0, 1, …, K-1} = bins of vote-fraction X.
// -----------------------------------------------------------------------------
export interface Bettor {
  belief: DiscreteBelief<number>;
  shares: number[];
  cash: number;          // cumulative cost paid for shares (signed; negative)
  fees_paid: number;
}

function newBettor(params: FactMachineParams): Bettor {
  const states: number[] = [];
  for (let i = 0; i < params.thetaBins; i++) states.push(i / (params.thetaBins - 1));
  const N = params.marketType === 'binary' ? 2 : params.thetaBins;
  return {
    belief: new DiscreteBelief(states),
    shares: new Array(N).fill(0),
    cash: 0, fees_paid: 0,
  };
}

// -----------------------------------------------------------------------------
// Outcome-distribution helpers.
//   binary, bernoulli:  P(YES) = θ.
//   binary, majority:   P(YES) = pYesWins(θ, params).
//   scalar (always majority):  P(bin_j) = Σ_{k : k/N_voters ∈ bin_j} Binomial(N, θ).pmf(k).
// We precompute once per (params) for efficiency. Returns a (thetaBins × N_outcomes)
// matrix where row i column j = P(outcome = j | θ = θ_i).
// -----------------------------------------------------------------------------
export function outcomeMatrix(params: FactMachineParams): number[][] {
  const K = params.thetaBins;
  const thetas: number[] = [];
  for (let i = 0; i < K; i++) thetas.push(i / (K - 1));
  if (params.marketType === 'binary') {
    return thetas.map(θ => {
      const pYes = pYesWins(θ, params);
      return [pYes, 1 - pYes];        // [YES, NO]
    });
  }
  // Scalar: K bins of vote fraction X = (#YES)/N over [0, 1].
  const N = params.N_voters;
  const matrix: number[][] = [];
  // Pre-compute binomial PMFs once per θ.
  for (const θ of thetas) {
    const pmf = binomialPMFInternal(N, θ);
    const row = new Array<number>(K).fill(0);
    for (let k = 0; k <= N; k++) {
      const x = k / N;                 // fraction in [0, 1]
      const j = Math.min(K - 1, Math.floor(x * K));
      row[j] += pmf[k];
    }
    matrix.push(row);
  }
  return matrix;
}

/** Stable binomial PMF for use in the scalar bin-probability matrix. */
function binomialPMFInternal(n: number, p: number): number[] {
  const out = new Array<number>(n + 1).fill(0);
  if (p <= 0)      { out[0] = 1; return out; }
  if (p >= 1)      { out[n] = 1; return out; }
  // log-space for stability.
  const logP = Math.log(p), logQ = Math.log(1 - p);
  let logCoef = 0;
  out[0] = Math.exp(n * logQ);
  for (let k = 1; k <= n; k++) {
    logCoef += Math.log(n - k + 1) - Math.log(k);
    out[k] = Math.exp(logCoef + k * logP + (n - k) * logQ);
  }
  return out;
}

// -----------------------------------------------------------------------------
// Likelihood of observing `yesOrders` out of `total` given θ and informedness.
// Each order is YES with probability  q(θ) = θ·informedness + 0.5·(1 - informedness).
// So  P(yes | θ) = Binomial(total, q(θ))[yes].
// We factor 1/total out (constant in θ) to keep the Bayes update unbiased.
// -----------------------------------------------------------------------------
function orderProb(theta: number, informedness: number): number {
  return theta * informedness + 0.5 * (1 - informedness);
}

function obsLikelihood(theta: number, yesOrders: number, total: number, informedness: number): number {
  const p = orderProb(theta, informedness);
  // Stable log-binomial. We don't include the binomial coefficient (constant in θ).
  const logL = yesOrders * Math.log(Math.max(1e-300, p))
             + (total - yesOrders) * Math.log(Math.max(1e-300, 1 - p));
  return Math.exp(logL);
}

// -----------------------------------------------------------------------------
// Action: -1 means hold; otherwise it's the index of the outcome to buy
// 1 share of. Generic over binary (N=2) and scalar (N=K) markets.
// -----------------------------------------------------------------------------

function pickAction(
  params: FactMachineParams,
  bettor: Bettor,
  market: LMSR,
  rng: () => number,
  /** time remaining (T - t). */
  tau: number,
  /** Pre-computed outcomeMatrix(params): row i col j = P(outcome=j | θ=θ_i). */
  outcomes: ReadonlyArray<ReadonlyArray<number>>,
): number {
  const N = market.N;
  const prices = market.prices();
  const fee = params.fee;
  switch (params.policy) {
    case 'random': {
      const u = Math.floor(rng() * (N + 1));   // last bucket is "hold"
      return u < N ? u : -1;
    }
    case 'hold': return -1;
    case 'oracle': {
      // Knows θ exactly; P(outcome=j | θ_true) is one row of the matrix.
      const trueIdx = Math.round(params.trueTheta * (params.thetaBins - 1));
      const row = outcomes[trueIdx];
      let bestJ = -1, bestEv = fee;   // require EV > fee to act
      for (let j = 0; j < N; j++) {
        const ev = row[j] - prices[j];
        if (ev > bestEv) { bestEv = ev; bestJ = j; }
      }
      return bestJ;
    }
    case 'myopic': {
      // Compute expected probability of each outcome under the belief:
      //   E_b[P(outcome = j)] = Σ_i b_i · outcomes[i][j].
      let bestJ = -1, bestEv = fee;
      for (let j = 0; j < N; j++) {
        let pj = 0;
        for (let i = 0; i < bettor.belief.weights.length; i++) {
          pj += bettor.belief.weights[i] * outcomes[i][j];
        }
        const ev = pj - prices[j];
        if (ev > bestEv) { bestEv = ev; bestJ = j; }
      }
      return bestJ;
    }
    case 'qmdp': {
      // Approximate QMDP: for each θ_i and outcome j, the per-share immediate
      // EV is outcomes[i][j] − prices[j] − fee. Future EV ≈ tau · max over j
      // (you'd buy the best bin every remaining period if profitable).
      // Then a* = argmax_j Σ_i b_i [max(0, EV_ij) + tau · max_j' max(0, EV_ij')].
      const Q = new Array<number>(N + 1).fill(0);
      for (let i = 0; i < bettor.belief.weights.length; i++) {
        const w = bettor.belief.weights[i];
        let bestEvHere = 0;
        for (let j = 0; j < N; j++) {
          const ev = outcomes[i][j] - prices[j] - fee;
          if (ev > bestEvHere) bestEvHere = ev;
        }
        const future = bestEvHere * tau;
        for (let j = 0; j < N; j++) {
          const ev = outcomes[i][j] - prices[j] - fee;
          Q[j] += w * (Math.max(0, ev) + future);
        }
        Q[N] += w * future;     // hold
      }
      let bestJ = -1, bestQ = Q[N] + 1e-12;
      for (let j = 0; j < N; j++) {
        if (Q[j] > bestQ) { bestQ = Q[j]; bestJ = j; }
      }
      return bestJ;
    }
  }
}

// -----------------------------------------------------------------------------
// One full market lifecycle.
// -----------------------------------------------------------------------------
// =============================================================================
// Stationary entities (extend `general/field-station.Station`).
//
// The simulation graph:
//
//     [NoiseTraderStation] ──orders──▶ [MarketStation] ──prices──▶ [BettorStation]
//                                            │                            │
//                                            └────────orders ◀────────────┘
//                                            │
//                                            ▼
//                                     [WorldCensus]                ▶ [FrameRecorder]
//                                            │
//                                            ▼
//                                     [ResolutionStation]   (fires once at t=T)
//
// Movables flow as `Order` objects from trader stations into the market's
// pending queue. The market drains its queue in two phases per tick (noise
// first, then bettor) so the bettor can observe noise-trader aggregate
// flow before deciding.
// =============================================================================

/** Movable: a single buy of one share of a market outcome. */
interface Order {
  kind: 'noise' | 'bettor';
  side: number;       // outcome index in [0, N_outcomes)
  /** For binary noise orders: was the underlying signal "YES"? Used to
   *  compute the yes/total observation that the bettor will see. */
  isYes?: boolean;
}

/** Movable: one voter's ballot at resolution time. */
interface Vote { value: 0 | 1; }

class MarketStation extends Station {
  lmsr: LMSR;
  noiseQueue: Order[] = [];
  bettorQueue: Order[] = [];
  /** Aggregates from the most recently settled noise batch. */
  lastNoiseYes = 0;
  lastNoiseTotal = 0;
  /** Cost (in cash) the bettor paid in the most recent settlement. */
  lastBettorCost = 0;
  lastBettorSide = -1;
  constructor(id: string, public N_outcomes: number, public liquidity: number) {
    super(id);
    this.lmsr = new LMSR(liquidity, N_outcomes);
  }
  enqueueNoise(side: number, isYes: boolean): void {
    this.noiseQueue.push({kind: 'noise', side, isYes});
  }
  enqueueBettor(side: number): void {
    this.bettorQueue.push({kind: 'bettor', side});
  }
  /** Phase 1 each tick: drain the noise batch. Called by orchestrator after
   *  NoiseTraderStation.runTimeStep. */
  settleNoise(): void {
    let yes = 0;
    const dq = new Array(this.N_outcomes).fill(0);
    for (const o of this.noiseQueue) {
      if (o.isYes) yes++;
      dq[o.side]++;
    }
    if (this.noiseQueue.length > 0) this.lmsr.trade(dq);
    this.lastNoiseYes = yes;
    this.lastNoiseTotal = this.noiseQueue.length;
    this.noiseQueue = [];
  }
  /** Phase 2 each tick: drain the bettor's order. Called by orchestrator
   *  after BettorStation.runTimeStep. */
  settleBettor(): void {
    if (this.bettorQueue.length === 0) {
      this.lastBettorCost = 0;
      this.lastBettorSide = -1;
      return;
    }
    const dq = new Array(this.N_outcomes).fill(0);
    let side = -1;
    for (const o of this.bettorQueue) { dq[o.side]++; side = o.side; }
    this.lastBettorCost = this.lmsr.trade(dq);
    this.lastBettorSide = side;
    this.bettorQueue = [];
  }
  /** Required by Station base; phasing happens via settleNoise/settleBettor. */
  runTimeStep(_dt: number, _t: number): void {}
}

class NoiseTraderStation extends Station {
  constructor(id: string,
              public market: MarketStation,
              public params: FactMachineParams,
              public rng: () => number) { super(id); }
  runTimeStep(_dt: number, t: number): void {
    let K = this.params.K_noise;
    let qSignal = orderProb(this.params.trueTheta, this.params.informedness);
    if (this.params.lateFlip && t === this.params.T - 2) {
      K = this.params.K_noise * (this.params.lateFlipMultiplier ?? 10);
      qSignal = orderProb(1 - this.params.trueTheta, this.params.informedness);
    }
    const total = Math.max(1, samplePoisson(K, this.rng));
    const N_out = this.market.N_outcomes;
    const half = Math.floor(N_out / 2);
    let yesCount = 0, noCount = 0;
    for (let k = 0; k < total; k++) {
      const isYes = this.rng() < qSignal;
      if (isYes) yesCount++; else noCount++;
      let side: number;
      if (this.params.marketType === 'binary') {
        side = isYes ? 0 : 1;       // 0 = YES, 1 = NO
      } else {
        // For scalar: yes orders concentrate in the upper half of bins;
        // no orders concentrate in the lower half. Same partitioning as
        // the original imperative model.
        if (isYes) side = half + (yesCount - 1) % Math.max(1, N_out - half);
        else       side = (noCount - 1) % Math.max(1, half);
      }
      this.market.enqueueNoise(side, isYes);
    }
  }
}

class BettorStation extends Station {
  belief: DiscreteBelief<number>;
  shares: number[];
  cash = 0;
  fees_paid = 0;
  /** Time-series the recorder will read each tick. */
  beliefMean: number[] = [];
  beliefVar: number[] = [];
  beliefEntropy: number[] = [];
  constructor(id: string,
              public market: MarketStation,
              public params: FactMachineParams,
              public rng: () => number,
              public outcomes: ReadonlyArray<ReadonlyArray<number>>) {
    super(id);
    const states: number[] = [];
    for (let i = 0; i < params.thetaBins; i++) states.push(i / (params.thetaBins - 1));
    this.belief = new DiscreteBelief(states);
    this.shares = new Array(market.N_outcomes).fill(0);
    this.beliefMean.push(this.belief.mean());
    this.beliefVar.push(this.belief.variance());
    this.beliefEntropy.push(this.belief.entropy());
  }
  runTimeStep(_dt: number, t: number): void {
    // The bettor reads the noise-tick aggregate directly from the market
    // (the orchestrator has already called market.settleNoise() this tick).
    const yes = this.market.lastNoiseYes;
    const total = this.market.lastNoiseTotal;
    if (total > 0) {
      this.belief.update(θ => obsLikelihood(θ, yes, total, this.params.informedness));
    }
    const tau = this.params.T - t;
    const action = pickAction(this.params, this as unknown as Bettor, this.market.lmsr,
                               this.rng, tau, this.outcomes);
    if (action >= 0) this.market.enqueueBettor(action);
  }
  /** Apply settlement (called by orchestrator after market.settleBettor()). */
  applySettlement(): void {
    if (this.market.lastBettorSide >= 0) {
      this.shares[this.market.lastBettorSide] += 1;
      this.cash -= this.market.lastBettorCost;
      this.fees_paid += this.params.fee;
    }
    this.beliefMean.push(this.belief.mean());
    this.beliefVar.push(this.belief.variance());
    this.beliefEntropy.push(this.belief.entropy());
  }
}

class WorldCensus extends Station {
  /** Cached market price vector (frozen during this tick). */
  prices: number[] = [];
  /** Cached belief weight vector (frozen during this tick). */
  beliefWeights: number[] = [];
  noiseYes = 0;
  noiseTotal = 0;
  constructor(id: string, public market: MarketStation, public bettor: BettorStation) { super(id); }
  runTimeStep(_dt: number, _t: number): void {
    this.prices = this.market.lmsr.prices();
    this.beliefWeights = this.bettor.belief.asArray();
    this.noiseYes = this.market.lastNoiseYes;
    this.noiseTotal = this.market.lastNoiseTotal;
  }
}

class ResolutionStation extends Station {
  outcomeIdx = 0;
  voteFraction = 0;
  payout = 0;
  fired = false;
  votes: Vote[] = [];
  constructor(id: string,
              public market: MarketStation,
              public bettor: BettorStation,
              public params: FactMachineParams,
              public rng: () => number) { super(id); }
  runTimeStep(_dt: number, t: number): void {
    if (t < this.params.T || this.fired) return;
    this.fired = true;
    // Generate voter movables.
    let yesVotes = 0;
    for (let i = 0; i < this.params.N_voters; i++) {
      const v: Vote = {value: this.rng() < this.params.trueTheta ? 1 : 0};
      this.votes.push(v);
      if (v.value === 1) yesVotes++;
    }
    this.voteFraction = yesVotes / this.params.N_voters;
    // Determine winning outcome.
    if (this.params.marketType === 'binary') {
      if (this.params.resolutionMode === 'majority') {
        this.outcomeIdx = yesVotes > this.params.N_voters / 2 ? 0 : 1;
      } else {
        this.outcomeIdx = this.rng() < this.params.trueTheta ? 0 : 1;
      }
    } else {
      this.outcomeIdx = Math.min(this.market.N_outcomes - 1,
                                  Math.floor(this.voteFraction * this.market.N_outcomes));
    }
    this.payout = this.bettor.shares[this.outcomeIdx];
  }
}

export interface FactMachineResult {
  params: FactMachineParams;
  finalOutcomeIdx: number;       // 0=YES, 1=NO for binary; 0..K-1 for scalar
  finalOutcome: 0 | 1;            // legacy: 1 if YES (majority/Bernoulli), 0 if NO
  finalTheta: number;
  finalVoteFraction: number;      // realised fraction of YES votes (majority mode)
  shares: number[];               // per-outcome shares held
  shares_yes: number;             // legacy convenience (binary mode = shares[0])
  shares_no: number;              // legacy convenience (binary mode = shares[1])
  trade_cost: number;
  fees_paid: number;
  payout: number;
  pnl: number;
  beliefMean: number[];           // per period (t = 0..T)
  beliefVar: number[];
  beliefEntropy: number[];
  /** Per-tick belief weight vector (length T+1, each weight vector has thetaBins entries).
   *  Captured during simulation so the post-hoc animation never has to re-run the model. */
  beliefSnapshots: number[][];
  priceHistory: number[][];       // per period: full price vector (length N_outcomes)
  yesOrdersHistory: number[];
  totalOrdersHistory: number[];
  /** Per-tick bettor action (−1 = hold, else outcome index bought). */
  actionHistory: number[];
}

// -----------------------------------------------------------------------------
// runFactMachine — DES orchestration over five stationary entities.
//
// Each tick the orchestrator drives the stations in this fixed order:
//
//   1.  NoiseTraderStation.runTimeStep
//          generates K orders (movables) and pushes them into market.noiseQueue
//
//   2.  market.settleNoise()
//          drains noiseQueue as a single LMSR transaction; updates last-tick
//          aggregates (yesOrders, total) for the bettor to observe
//
//   3.  CensusStation.runTimeStep   ← snapshots current market prices + bettor belief
//          read by the trace recorder; never reads or writes simulation state
//
//   4.  BettorStation.runTimeStep
//          reads market.lastNoiseYes / lastNoiseTotal, updates b(θ) via
//          Bayesian filter, picks an action, pushes its order into market.bettorQueue
//
//   5.  market.settleBettor()
//          drains bettorQueue against the LMSR; computes the bettor's cost
//
//   6.  bettor.applySettlement()
//          updates shares / cash / fees, pushes belief stats into the trace
//
// At t = T:
//
//   7.  ResolutionStation.runTimeStep
//          generates N_voters Vote movables, decides the winning outcome,
//          settles the bettor's payout
//
// The animation is generated AFTER the simulation finishes from
// `result.beliefSnapshots`, `result.priceHistory`, etc. The simulation
// never builds any frame data inline, so the hot loop stays fast and the
// player can later replay at any speed.
// -----------------------------------------------------------------------------
export function runFactMachine(params: FactMachineParams): FactMachineResult {
  if (params.marketType === 'scalar' && params.resolutionMode !== 'majority') {
    // Scalar markets need a multi-outcome resolution; Bernoulli is degenerate.
    params = {...params, resolutionMode: 'majority'};
  }
  const rng = mulberry32(params.seed);
  const N_outcomes = params.marketType === 'binary' ? 2 : params.thetaBins;
  const outcomes = outcomeMatrix(params);

  // --- Build the station graph -------------------------------------------
  const market   = new MarketStation('market', N_outcomes, params.liquidity);
  const noise    = new NoiseTraderStation('noise', market, params, rng);
  const bettor   = new BettorStation('bettor', market, params, rng, outcomes);
  const census   = new WorldCensus('census', market, bettor);
  const resolver = new ResolutionStation('resolution', market, bettor, params, rng);

  // --- Trace -------------------------------------------------------------
  const trace: FactMachineResult = {
    params,
    finalOutcomeIdx: 0, finalOutcome: 0,
    finalTheta: params.trueTheta, finalVoteFraction: 0,
    shares: new Array(N_outcomes).fill(0),
    shares_yes: 0, shares_no: 0,
    trade_cost: 0, fees_paid: 0, payout: 0, pnl: 0,
    beliefMean: bettor.beliefMean,
    beliefVar:  bettor.beliefVar,
    beliefEntropy: bettor.beliefEntropy,
    beliefSnapshots: [bettor.belief.asArray()],
    priceHistory: [market.lmsr.prices()],
    yesOrdersHistory: [],
    totalOrdersHistory: [],
    actionHistory: [],
  };

  // --- Tick loop: 6-phase per-tick orchestration -------------------------
  for (let t = 0; t < params.T; t++) {
    noise.runTimeStep(1, t);          // phase 1: noise → market.noiseQueue
    market.settleNoise();             // phase 2: drain noise batch
    census.runTimeStep(1, t);         // phase 3: snapshot for trace
    bettor.runTimeStep(1, t);         // phase 4: belief update + action
    market.settleBettor();            // phase 5: drain bettor's order
    bettor.applySettlement();         // phase 6: shares, cash, belief stats

    trace.beliefSnapshots.push(bettor.belief.asArray());
    trace.priceHistory.push(census.prices.slice());
    trace.yesOrdersHistory.push(market.lastNoiseYes);
    trace.totalOrdersHistory.push(market.lastNoiseTotal);
    trace.actionHistory.push(market.lastBettorSide);
    trace.fees_paid = bettor.fees_paid;
  }

  // --- Resolution --------------------------------------------------------
  resolver.runTimeStep(1, params.T);
  trace.finalOutcomeIdx   = resolver.outcomeIdx;
  trace.finalOutcome      = (resolver.outcomeIdx === 0) ? 1 : 0;
  trace.finalVoteFraction = resolver.voteFraction;
  trace.shares            = bettor.shares.slice();
  trace.shares_yes        = bettor.shares[0];
  trace.shares_no         = N_outcomes >= 2 ? bettor.shares[1] : 0;
  trace.trade_cost        = -bettor.cash;
  trace.payout            = resolver.payout;
  trace.pnl               = resolver.payout + bettor.cash - bettor.fees_paid;
  return trace;
}

// -----------------------------------------------------------------------------
// CLI.
// -----------------------------------------------------------------------------
async function main(): Promise<void> {
  const baseParams = defaultParams({
    T: Number(process.env.T ?? 24),
    trueTheta: Number(process.env.TRUE_THETA ?? 0.65),
    informedness: Number(process.env.INFORMEDNESS ?? 0.6),
    K_noise: Number(process.env.K_NOISE ?? 20),
    fee: Number(process.env.FEE ?? 0.01),
    liquidity: Number(process.env.LIQUIDITY ?? 50),
    thetaBins: Number(process.env.THETA_BINS ?? 21),
    seed: Number(process.env.SEED ?? 1),
    policy: (process.env.POLICY ?? 'qmdp') as FactMachineParams['policy'],
    resolutionMode: ((process.env.RESOLUTION ?? 'bernoulli') as FactMachineParams['resolutionMode']),
    marketType: ((process.env.MARKET ?? 'binary') as FactMachineParams['marketType']),
    lateFlip: process.env.LATE_FLIP === '1',
    lateFlipMultiplier: Number(process.env.LATE_FLIP_MUL ?? 10),
  });
  // Scalar markets need majority resolution to be non-degenerate.
  if (baseParams.marketType === 'scalar' && baseParams.resolutionMode !== 'majority') {
    baseParams.resolutionMode = 'majority';
  }

  const reps = Number(process.env.N_REPS ?? 1);
  if (reps === 1) {
    const r = runFactMachine(baseParams);
    const N_outcomes = r.priceHistory[0].length;
    console.log(`# FactMachine POMDP single run`);
    console.log(`#   marketType=${baseParams.marketType} (${N_outcomes} outcomes)  resolution=${baseParams.resolutionMode}  policy=${baseParams.policy}`);
    console.log(`#   T=${baseParams.T}  trueθ=${baseParams.trueTheta}  informedness=${baseParams.informedness}  K_noise=${baseParams.K_noise}  fee=${baseParams.fee}  liq=${baseParams.liquidity}`);
    console.log(`#`);
    if (baseParams.marketType === 'binary') {
      console.log(`# t      P(YES)      E[θ]       Var[θ]    H(b)     yes/total`);
      for (let t = 0; t <= baseParams.T; t++) {
        const yo = t > 0 ? r.yesOrdersHistory[t - 1] : 0;
        const to = t > 0 ? r.totalOrdersHistory[t - 1] : 0;
        console.log(`# ${String(t).padStart(2)}  ${r.priceHistory[t][0].toFixed(4)}  ${r.beliefMean[t].toFixed(4)}  ${r.beliefVar[t].toFixed(5)}  ${r.beliefEntropy[t].toFixed(3)}  ${yo}/${to}`);
      }
    } else {
      console.log(`# t   E[θ]  H(b)  market mode bin   |   peak market price`);
      for (let t = 0; t <= baseParams.T; t++) {
        const ph = r.priceHistory[t];
        let bestJ = 0; for (let j = 1; j < ph.length; j++) if (ph[j] > ph[bestJ]) bestJ = j;
        const binCenter = (bestJ + 0.5) / ph.length;
        console.log(`# ${String(t).padStart(2)}  ${r.beliefMean[t].toFixed(3)}  ${r.beliefEntropy[t].toFixed(3)}    bin ${String(bestJ).padStart(2)} ≈ ${binCenter.toFixed(2)}     ${ph[bestJ].toFixed(3)}`);
      }
    }
    console.log(`#`);
    if (baseParams.marketType === 'binary') {
      console.log(`# RESOLUTION: vote fraction = ${r.finalVoteFraction.toFixed(3)}  outcome=${r.finalOutcome === 1 ? 'YES' : 'NO'}  shares: yes=${r.shares_yes}, no=${r.shares_no}`);
    } else {
      const winBin = r.finalOutcomeIdx;
      const winShares = r.shares[winBin];
      const totalShares = r.shares.reduce((a, b) => a + b, 0);
      console.log(`# RESOLUTION: vote fraction = ${r.finalVoteFraction.toFixed(3)}  → bin ${winBin} of ${N_outcomes}`);
      console.log(`#   shares = [${r.shares.map(s => String(s)).join(', ')}]   (winning bin holds ${winShares} of ${totalShares} total)`);
    }
    console.log(`#   trade cost  = ${r.trade_cost.toFixed(4)}`);
    console.log(`#   fees paid   = ${r.fees_paid.toFixed(4)}`);
    console.log(`#   payout      = ${r.payout.toFixed(4)}`);
    console.log(`#   PnL         = ${r.pnl.toFixed(4)}`);
    console.log(`#   final E[θ]  = ${r.beliefMean[r.beliefMean.length-1].toFixed(4)}  (true ${baseParams.trueTheta})`);

    if (process.env.ANIMATE === '1') {
      // Animation is rendered POST-HOC from the result trace. The
      // simulation itself does no frame work — keeps the hot loop fast and
      // lets the user replay at any speed in the HTML player.
      await renderAnimation(r, baseParams);
      console.log(`#   animation written to out/factmachine.html`);
    }
    return;
  }

  // Multi-rep summary across policies.
  console.log(`# FactMachine POMDP — N_REPS=${reps} per policy  (market=${baseParams.marketType}, resolution=${baseParams.resolutionMode})`);
  console.log(`#   trueθ=${baseParams.trueTheta}  T=${baseParams.T}  informedness=${baseParams.informedness}  fee=${baseParams.fee}\n`);
  const policies: FactMachineParams['policy'][] = ['hold', 'random', 'myopic', 'qmdp', 'oracle'];
  console.log(`# policy     mean PnL    sd PnL    win-rate   final-Brier   total shares   trades`);
  for (const policy of policies) {
    let sum = 0, sumSq = 0, wins = 0, brier = 0, totalShares = 0, trades = 0;
    for (let r = 0; r < reps; r++) {
      const params = {...baseParams, seed: 1000 + r, policy};
      const out = runFactMachine(params);
      sum += out.pnl; sumSq += out.pnl * out.pnl;
      if (out.pnl > 0) wins++;
      brier += brierScore(out.beliefMean[out.beliefMean.length - 1],
                          out.finalOutcome as 0 | 1);
      const ts = out.shares.reduce((a, b) => a + b, 0);
      totalShares += ts;
      trades += ts;     // 1 trade = 1 share in our model
    }
    const mean = sum / reps;
    const variance = Math.max(0, sumSq / reps - mean * mean);
    const sd = Math.sqrt(variance);
    console.log(`# ${policy.padEnd(8)}  ${mean.toFixed(4).padStart(9)}  ${sd.toFixed(4).padStart(8)}  ${(wins / reps).toFixed(3).padStart(8)}    ${(brier / reps).toFixed(4).padStart(8)}      ${(totalShares/reps).toFixed(2).padStart(7)}    ${(trades/reps).toFixed(2)}`);
  }
}

/**
 * Render the animation POST-HOC from a completed simulation trace.
 *
 * The simulation itself never builds frame data inline; it just records
 * `beliefSnapshots`, `priceHistory`, `actionHistory`, etc. into the result.
 * Here we walk that trace in O(T) time, ask the scene builder for SVG
 * shapes per tick, and let the FrameRecorder write a self-contained HTML
 * player. The player's own UI controls playback speed; we set a default
 * fps but the user can scrub / pause / replay at any rate.
 *
 * Optional env vars:
 *   ANIM_FPS     default playback rate. Default 4 (one tick per 250 ms).
 *   ANIM_FRAMES  cap total frames; ticks are subsampled if T > cap.
 */
export async function renderAnimation(
  r: FactMachineResult,
  params: FactMachineParams,
  outPath?: string,
): Promise<string> {
  const {STAGE_W, STAGE_H, buildFactMachineFrame, buildFactMachineCharts} =
    await import('./animation/scenes/factmachine-scene');
  const {FrameRecorder} = await import('./animation/frame-recorder');
  const outDir = outPath ?? path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  // Five sub-frames per simulation tick, one per phase:
  //   0 noise→market   1 noise settles + census   2 bettor reads + sends
  //   3 bettor settles 4 (last tick only) resolution fires
  const fps = Number(process.env.ANIM_FPS ?? 8);   // 8 fps × 5 phases ≈ 1.6 ticks/sec
  const html = path.join(outDir, 'factmachine.html');
  const rec = new (FrameRecorder as any)({
    framesPath: path.join(outDir, 'factmachine.frames.jsonl'),
    htmlPath:   html,
    width: STAGE_W, height: STAGE_H, fps,
    title:    `FactMachine POMDP — market=${params.marketType}, policy=${params.policy}, true θ=${params.trueTheta}`,
    subtitle: `Each tick has 5 phases: noise→market, settle, bettor reads+trades, bettor settles, (resolution at t=T). Movables are the coloured dots.`,
  });
  // For each integer simulation tick, emit 5 sub-frames showing the
  // phasing of the DES orchestrator. The animation player advances one
  // sub-frame at a time so the user can see movables in flight.
  let subTick = 0;
  for (let t = 0; t <= params.T; t++) {
    const beliefAtT = r.beliefSnapshots[t];
    const pricesAtT = r.priceHistory[t];
    const noiseTotalThisTick = t < params.T ? (r.totalOrdersHistory[t] ?? 0) : 0;
    const noiseYesThisTick   = t < params.T ? (r.yesOrdersHistory[t]  ?? 0) : 0;
    const bettorActionThisTick = t < params.T ? (r.actionHistory[t]   ?? -1) : -1;
    const phaseCount = (t === params.T) ? 5 : 4;     // resolution phase only at t=T
    for (let phase = 0 as 0|1|2|3|4; phase < phaseCount; phase = (phase + 1) as 0|1|2|3|4) {
      const tCapture = t + phase / 5;
      const sub = subTick++;
      const arch = {
        tick: t, phase,
        noiseOrderCount: noiseTotalThisTick,
        noiseYes: noiseYesThisTick,
        noiseTotal: noiseTotalThisTick,
        bettorAction: bettorActionThisTick,
        voterCount: params.N_voters,
        resolutionOutcome: t === params.T ? r.finalOutcomeIdx : undefined,
        voteFraction: t === params.T ? r.finalVoteFraction : undefined,
        beliefWeights: beliefAtT,
        prices: pricesAtT,
      };
      rec.frame(tCapture, sub, () => buildFactMachineFrame(t, beliefAtT, r, params, arch));
    }
  }
  rec.setCharts(buildFactMachineCharts(r));
  await rec.finish();
  return html;
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
