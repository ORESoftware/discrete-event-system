// RUST MIGRATION: target module src/des/general/factmachine_math.rs.
// RUST MIGRATION: OptionPrices, BuyExecution, SellExecution, RecapResult, OptionAggregates, ReplayOrder, and ReplayResult become serde structs.
// RUST MIGRATION: All LMSR/PnL helpers are pure vanilla functions; keep them as free functions or wrap public API calls in small PureTransform structs only if graph-visible.
// RUST MIGRATION: Replace thrown numeric/domain checks with Result, and decide up front whether f64 is sufficient or a decimal crate is needed for production parity.
// RUST MIGRATION: ReadonlyArray inputs become slices, replayOrders consumes &[ReplayOrder], and aggregate state becomes an explicit mutable struct.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/factmachine-math.rs  (module des::general::factmachine_math)
// 1:1 file move. float64 mirror of the production LMSR/PnL trading math (binary-market).
//
// Declarations → Rust:
//   const LN2 / BPS_BASE          -> `const` (LN2 = std::f64::consts::LN_2)
//   interface OptionPrices/BuyExecution/SellExecution/RecapResult/OptionAggregates/
//             ReplayOrder/ReplayResult -> structs (#[derive(Clone, Copy/Debug)])
//   fn bFromLiquidity/optionOnePrice/lmsrCost/optionPrices/sharesFromBudget/buyExecution/
//      sellExecution/maxPriceWithSlippage/minPriceWithSlippage/recapitalization/avgCostBasis/
//      netPosition/unrealizedPnl/finalPnl/replayOrders/buyThenSellRoundTrip -> assoc fns
//                                    (or PureTransform<Input, Output> — they are vanilla, deterministic)
//
// Conversion notes (file-specific):
//   - Functions taking an `args: { ... }` object -> a named `Input` struct each (no positional sprawl).
//   - `ReplayResult extends OptionAggregates` -> compose/flatten the base fields (no interface inheritance).
//   - `ReplayOrder.action: 'BUY'|'SELL'` -> enum; `time?: number | null` -> `Option<f64>`.
//   - Production uses Decimal.js (BigNumber); this is the f64 mirror — keep f64 unless exact decimal needed.
//   - `throw` on invalid params (b<=0, etc.) -> `Result`/`panic!`; `expm1`/`log1p` -> f64::exp_m1/ln_1p.
//   - nulls-first stable sort comparator -> custom sort key with `Option` ordering.
// =============================================================================

// =============================================================================
// general/factmachine-math.ts — pure float64 mirror of the production
// `@factmachine/math/trading` API as shipped in
//   factmachine-monorepo/packages/math/src/trading/{lmsr,pnl,decimal}.ts
//
// WHY THIS MODULE EXISTS
// ──────────────────────
// The DES POMDP simulation in `main-factmachine.ts` historically used a
// generalised log-sum-exp LMSR class that treated its `liquidity` argument
// directly as the cost-function parameter `b`. Production, however, uses
// the BINARY-MARKET specialisation with a different convention:
//
//   • The user-facing parameter is "initial liquidity" L (USDC), and the
//     LMSR cost-function parameter is derived as  b = L / ln(2).
//     This is the textbook relation `L_max = b · ln(N)`  with N=2.
//
//   • Trades are BUDGET-driven, not quantity-driven.
//
//   • Fees are PROPORTIONAL (basis points), not additive.
//
//   • All math runs in arbitrary-precision Decimal.js.
//
// This file exposes the same algebra in float64 so the DES simulation
// can step millions of ticks per second while staying numerically equivalent
// to production within the regime that production itself tests
// (|q2-q1|/b ≤ ~14 with arbitrary q's and L ≥ 500). The cross-validation
// runner `runners/validate-factmachine-math.ts` confirms bit-level
// agreement on a property-based test grid.
//
// PRODUCTION FILE → THIS FILE MAPPING
// ────────────────────────────────────
//   computeOptionOnePrice            → optionOnePrice
//   computeLmsrCost                  → lmsrCost
//   computeBuyExecution              → buyExecution
//   computeSellExecution             → sellExecution
//   computeOptionPrices              → optionPrices
//   computeMaxPrice                  → maxPriceWithSlippage
//   computeMinPrice                  → minPriceWithSlippage
//   computeBFromInitialLiquidity     → bFromLiquidity   (L → b = L/ln(2))
//   computeRecapitalization          → recapitalization
//   computeAvgCostBasis              → avgCostBasis
//   computeNetPosition               → netPosition
//   computeUnrealizedPnl             → unrealizedPnl
//   computeFinalPnl                  → finalPnl
//   getOrderAggregates               → replayOrders
// =============================================================================

import {PureTransform} from '../shared/transform';

const LN2 = Math.log(2);
const BPS_BASE = 10_000;

/** Canonical: convert a USDC liquidity amount into the LMSR `b` parameter.
 *  Throws if `L` is not finite or non-positive. */
export function bFromLiquidity(L: number): number {
  if (!Number.isFinite(L) || L <= 0) throw new Error('initialLiquidity must be a finite positive number');
  return L / LN2;
}

// -----------------------------------------------------------------------------
// LMSR PRIMITIVES (binary market specialisation)
// -----------------------------------------------------------------------------

/** Bundled args for the LMSR price/cost primitives (qOne, qTwo, liquidity b). */
export interface LmsrPriceInput { qOne: number; qTwo: number; b: number; }

/** Production formula: `price_1 = 1 / (1 + exp((q2 − q1) / b))`. Always in (0, 1). */
export class OptionOnePrice extends PureTransform<LmsrPriceInput, number> {
  transform(input: LmsrPriceInput): number {
    const {qOne, qTwo, b} = input;
    if (b <= 0) throw new Error('b must be > 0');
    // Guard the exponent to prevent Infinity for huge |q2 − q1|/b: production
    // ranges keep |q2-q1|/b ≤ ~14, so we clamp at ±700 (where Math.exp would
    // overflow IEEE 754 at ±710).
    const exponent = (qTwo - qOne) / b;
    if (exponent >  700) return 0;       // overwhelmingly P(option2) = 1
    if (exponent < -700) return 1;       // overwhelmingly P(option1) = 1
    return 1 / (1 + Math.exp(exponent));
  }
}

/** @deprecated Use `new OptionOnePrice().transform({qOne, qTwo, b})`. */
export function optionOnePrice(qOne: number, qTwo: number, b: number): number {
  return new OptionOnePrice().transform({qOne, qTwo, b});
}

/** Production formula: `C(q) = b · ln(exp(q1/b) + exp(q2/b))`, log-sum-exp form. */
export class LmsrCost extends PureTransform<LmsrPriceInput, number> {
  transform(input: LmsrPriceInput): number {
    const {qOne, qTwo, b} = input;
    if (b <= 0) throw new Error('b must be > 0');
    const max = qOne >= qTwo ? qOne : qTwo;
    const min = qOne >= qTwo ? qTwo : qOne;
    const expTerm = Math.exp((min - max) / b);
    return max + b * Math.log1p(expTerm);
  }
}

/** @deprecated Use `new LmsrCost().transform({qOne, qTwo, b})`. */
export function lmsrCost(qOne: number, qTwo: number, b: number): number {
  return new LmsrCost().transform({qOne, qTwo, b});
}

export interface OptionPrices { optionOne: number; optionTwo: number; }
export function optionPrices(qOne: number, qTwo: number, b: number): OptionPrices {
  const p1 = optionOnePrice(qOne, qTwo, b);
  return {optionOne: p1, optionTwo: 1 - p1};
}

// -----------------------------------------------------------------------------
// EXECUTION FORMULAS (budget-based buys, share-based sells)
// -----------------------------------------------------------------------------

/** Production formula `shares = b · ln(1 + (exp(budget/b) − 1) / currentPrice)`.
 *  This is the inversion of the LMSR cost function: buying `budget` USDC of
 *  shares at the current marginal price moves `q_buy` by exactly `shares`
 *  amount. */
export interface SharesFromBudgetInput { budget: number; currentPrice: number; b: number; }

export class SharesFromBudget extends PureTransform<SharesFromBudgetInput, number> {
  transform(input: SharesFromBudgetInput): number {
    const {budget, currentPrice, b} = input;
    if (b <= 0) throw new Error('b must be > 0');
    if (currentPrice <= 0) throw new Error('currentPrice must be > 0');
    if (budget <= 0) return 0;
    // Math.expm1(x) = exp(x) - 1, accurate near 0; avoids cancellation when
    // budget << b (small trades).
    const expm1 = Math.expm1(budget / b);
    return b * Math.log1p(expm1 / currentPrice);
  }
}

/** @deprecated Use `new SharesFromBudget().transform({budget, currentPrice, b})`. */
export function sharesFromBudget(budget: number, currentPrice: number, b: number): number {
  return new SharesFromBudget().transform({budget, currentPrice, b});
}

export interface BuyExecution {
  shares: number;
  buyAmount: number;        // amount net of fees
  averagePrice: number;     // buyAmount / shares
  feeAmount: number;
  reward: number;           // mirrors prod: equals shares for a buy
}
export interface BuyExecutionInput {
  amount: number;
  optionOneShares: number;
  optionTwoShares: number;
  b: number;
  feeBps?: number;
  isOptionOne?: boolean;
}

export class BuyExecutor extends PureTransform<BuyExecutionInput, BuyExecution> {
  transform(args: BuyExecutionInput): BuyExecution {
    const feeBps = args.feeBps ?? 0;
    if (feeBps >= BPS_BASE) throw new Error('feeBps must be less than 10000 (100%)');
    const isOptionOne = args.isOptionOne ?? true;
    const price = new OptionOnePrice().transform({qOne: args.optionOneShares, qTwo: args.optionTwoShares, b: args.b});
    const feeAmount = (args.amount * feeBps) / BPS_BASE;
    const buyAmount = args.amount - feeAmount;
    const sideCurrentPrice = isOptionOne ? price : 1 - price;
    const shares = new SharesFromBudget().transform({budget: buyAmount, currentPrice: sideCurrentPrice, b: args.b});
    const averagePrice = shares === 0 ? 0 : buyAmount / shares;
    return {shares, buyAmount, averagePrice, feeAmount, reward: shares};
  }
}

/** @deprecated Use `new BuyExecutor().transform(args)`. */
export function buyExecution(args: BuyExecutionInput): BuyExecution {
  return new BuyExecutor().transform(args);
}

export interface SellExecution {
  usdcOut: number;
  sellAmount: number;       // gross USDC before fees
  averagePrice: number;     // sellAmount / sharesOut
  feeAmount: number;
  reward: number;           // mirrors prod: equals usdcOut for a sell
}
export interface SellExecutionInput {
  sharesOut: number;
  optionOneShares: number;
  optionTwoShares: number;
  b: number;
  feeBps?: number;
  isOptionOne?: boolean;
}

export class SellExecutor extends PureTransform<SellExecutionInput, SellExecution> {
  transform(args: SellExecutionInput): SellExecution {
    const feeBps = args.feeBps ?? 0;
    if (feeBps >= BPS_BASE) throw new Error('feeBps must be less than 10000 (100%)');
    const isOptionOne = args.isOptionOne ?? true;
    const lmsr = new LmsrCost();
    const costBefore = lmsr.transform({qOne: args.optionOneShares, qTwo: args.optionTwoShares, b: args.b});
    const newQ1 = isOptionOne ? args.optionOneShares - args.sharesOut : args.optionOneShares;
    const newQ2 = isOptionOne ? args.optionTwoShares                  : args.optionTwoShares - args.sharesOut;
    const costAfter = lmsr.transform({qOne: newQ1, qTwo: newQ2, b: args.b});
    const sellAmount = costBefore - costAfter;
    const feeAmount = (sellAmount * feeBps) / BPS_BASE;
    const usdcOut = sellAmount - feeAmount;
    const averagePrice = args.sharesOut === 0 ? 0 : sellAmount / args.sharesOut;
    return {usdcOut, sellAmount, averagePrice, feeAmount, reward: usdcOut};
  }
}

/** @deprecated Use `new SellExecutor().transform(args)`. */
export function sellExecution(args: SellExecutionInput): SellExecution {
  return new SellExecutor().transform(args);
}

// -----------------------------------------------------------------------------
// SLIPPAGE (production: clamp to [0, 1])
// -----------------------------------------------------------------------------

export function maxPriceWithSlippage(price: number, slippageBps: number): number {
  const factor = 1 + slippageBps / BPS_BASE;
  return Math.max(0, Math.min(1, price * factor));
}
export function minPriceWithSlippage(price: number, slippageBps: number): number {
  const factor = 1 - slippageBps / BPS_BASE;
  return Math.max(0, Math.min(1, price * factor));
}

// -----------------------------------------------------------------------------
// RECAPITALIZATION (add or remove liquidity)
// -----------------------------------------------------------------------------

export interface RecapResult {
  newOptionOneShares: number;
  newOptionTwoShares: number;
  newB: number;
  capitalDelta: number;      // |C(new) - C(old)|; absolute USDC moved in/out
}
/** Production: `newQ = oldQ · newB / oldB` preserves prices because
 *  prices depend only on (q2 − q1)/b, and (q2-q1) and b scale together. */
export interface RecapitalizationInput {
  optionOneShares: number;
  optionTwoShares: number;
  currentB: number;
  newB: number;
}

export class Recapitalization extends PureTransform<RecapitalizationInput, RecapResult> {
  transform(args: RecapitalizationInput): RecapResult {
    if (args.currentB <= 0) throw new Error('currentB must be > 0');
    if (args.newB     <= 0) throw new Error('newB must be > 0');
    if (args.currentB === args.newB) throw new Error('newB must differ from currentB');
    const ratio = args.newB / args.currentB;
    const newOptionOneShares = args.optionOneShares * ratio;
    const newOptionTwoShares = args.optionTwoShares * ratio;
    const lmsr = new LmsrCost();
    const costOld = lmsr.transform({qOne: args.optionOneShares, qTwo: args.optionTwoShares, b: args.currentB});
    const costNew = lmsr.transform({qOne: newOptionOneShares, qTwo: newOptionTwoShares, b: args.newB});
    return {
      newOptionOneShares, newOptionTwoShares, newB: args.newB,
      capitalDelta: Math.abs(costNew - costOld),
    };
  }
}

/** @deprecated Use `new Recapitalization().transform(args)`. */
export function recapitalization(args: RecapitalizationInput): RecapResult {
  return new Recapitalization().transform(args);
}

// -----------------------------------------------------------------------------
// PnL ACCOUNTING (weighted-average cost basis, mirrors production pnl.ts)
// -----------------------------------------------------------------------------

export interface OptionAggregates {
  totalSharesBought: number;
  totalSharesSold: number;
  totalBuyAmount: number;
  totalSellProceeds: number;
  realizedPnl: number;
}

/** `avgCostBasis = (totalBuyAmount − totalSellProceeds + realizedPnl) / netPosition`.
 *  Returns 0 when netPosition ≤ 0. */
export function avgCostBasis(s: OptionAggregates): number {
  const net = s.totalSharesBought - s.totalSharesSold;
  if (net <= 0) return 0;
  return (s.totalBuyAmount - s.totalSellProceeds + s.realizedPnl) / net;
}

export function netPosition(s: Pick<OptionAggregates, 'totalSharesBought' | 'totalSharesSold'>): number {
  return Math.max(0, s.totalSharesBought - s.totalSharesSold);
}

export function unrealizedPnl(args: {
  netPosition: number; currentPrice: number; avgCostBasis: number;
}): number {
  if (args.netPosition <= 0) return 0;
  return args.netPosition * (args.currentPrice - args.avgCostBasis);
}

/** Final PnL after market resolution: `proceeds + (resolutionPrice · netPos) − totalBuy`. */
export function finalPnl(args: {
  totalSellProceeds: number; totalBuyAmount: number;
  netPosition: number; resolutionPrice: number;
}): number {
  return args.totalSellProceeds + args.resolutionPrice * args.netPosition - args.totalBuyAmount;
}

export interface ReplayOrder {
  action: 'BUY' | 'SELL';
  shares: number;
  usdc: number;
  time?: number | null;
}

export interface ReplayResult extends OptionAggregates {
  netPosition: number;
  avgCostBasis: number;
  totalOrders: number;
  totalVolume: number;
  lastTime: number | null;
}

/** Replay a sequence of orders using weighted-average cost accounting. */
export class ReplayOrders extends PureTransform<ReadonlyArray<ReplayOrder>, ReplayResult> {
  transform(orders: ReadonlyArray<ReplayOrder>): ReplayResult {
    let totalSharesBought = 0, totalSharesSold = 0;
    let totalBuyAmount = 0, totalSellProceeds = 0;
    let realizedPnl = 0, totalVolume = 0;
    let lastTime: number | null = null;
    // Stable sort by time, with nulls first (to mirror production behaviour).
    const sorted = [...orders].sort((a, b) => {
      if (a.time == null && b.time == null) return 0;
      if (a.time == null) return -1;
      if (b.time == null) return 1;
      return a.time! - b.time!;
    });
    for (const o of sorted) {
      if (o.time != null) lastTime = o.time;
      if (o.action === 'BUY') {
        totalSharesBought += o.shares;
        totalBuyAmount    += o.usdc;
        totalVolume       += Math.abs(o.usdc);
      } else {
        const acb = avgCostBasis({totalSharesBought, totalSharesSold,
                                  totalBuyAmount, totalSellProceeds, realizedPnl});
        realizedPnl       += o.usdc - o.shares * acb;
        totalSharesSold   += o.shares;
        totalSellProceeds += o.usdc;
        totalVolume       += Math.abs(o.usdc);
      }
    }
    return {
      totalSharesBought, totalSharesSold, totalBuyAmount,
      totalSellProceeds, realizedPnl,
      netPosition: Math.max(0, totalSharesBought - totalSharesSold),
      avgCostBasis: avgCostBasis({totalSharesBought, totalSharesSold,
                                  totalBuyAmount, totalSellProceeds, realizedPnl}),
      totalOrders: orders.length,
      totalVolume, lastTime,
    };
  }
}

/** @deprecated Use `new ReplayOrders().transform(orders)`. */
export function replayOrders(orders: ReadonlyArray<ReplayOrder>): ReplayResult {
  return new ReplayOrders().transform(orders);
}

// -----------------------------------------------------------------------------
// HIGHER-LEVEL HELPERS / INVARIANT CHECKS
// -----------------------------------------------------------------------------

/** Buy-then-sell round-trip: returns the proceeds you get back if you
 *  immediately sell all shares you just bought. Production tests that this
 *  is ≤ original buy amount (no-arb). */
export interface BuyThenSellRoundTripInput {
  amount: number; optionOneShares: number; optionTwoShares: number;
  b: number; isOptionOne?: boolean; feeBps?: number;
}

export interface RoundTripResult { buy: BuyExecution; sell: SellExecution; net: number; }

export class BuyThenSellRoundTrip extends PureTransform<BuyThenSellRoundTripInput, RoundTripResult> {
  transform(args: BuyThenSellRoundTripInput): RoundTripResult {
    const buy = new BuyExecutor().transform(args);
    const newQ1 = (args.isOptionOne ?? true) ? args.optionOneShares + buy.shares : args.optionOneShares;
    const newQ2 = (args.isOptionOne ?? true) ? args.optionTwoShares : args.optionTwoShares + buy.shares;
    const sell = new SellExecutor().transform({
      sharesOut: buy.shares,
      optionOneShares: newQ1, optionTwoShares: newQ2,
      b: args.b, isOptionOne: args.isOptionOne, feeBps: args.feeBps,
    });
    return {buy, sell, net: sell.usdcOut - args.amount};
  }
}

/** @deprecated Use `new BuyThenSellRoundTrip().transform(args)`. */
export function buyThenSellRoundTrip(args: BuyThenSellRoundTripInput): RoundTripResult {
  return new BuyThenSellRoundTrip().transform(args);
}
