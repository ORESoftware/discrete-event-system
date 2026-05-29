'use strict';

// =============================================================================
// runners/validate-factmachine-math.ts — audit + harden the FactMachine
// math layer used inside the DES POMDP simulation.
//
// CHECKS RUN BY THIS RUNNER
// ─────────────────────────
//   PART A — invariants (always run)
//     A1. Prices sum to 1
//     A2. Each price strictly in (0, 1)
//     A3. Equal shares ⇒ price = 0.5
//     A4. Symmetry under share swap: p₁(q1, q2) = p₂(q2, q1)
//     A5. Monotonicity in shares
//     A6. Buy: fee + buyAmount = total
//     A7. Buy: positive amount → strictly positive shares
//     A8. Buy: averagePrice = buyAmount / shares
//     A9. Buy: averagePrice < 1
//    A10. Buy: monotone in spending
//    A11. Sell: usdcOut + fee = sellAmount
//    A12. Sell: monotone in shares sold
//    A13. Round-trip: sell-after-buy ≤ original buy amount (no-arb)
//    A14. Recapitalisation preserves prices
//    A15. Slippage maxPrice ≥ price ≥ minPrice, both in [0, 1]
//
//   PART B — cross-validation against the actual production package
//     `@factmachine/math/trading` loaded directly from
//     factmachine-monorepo/packages/math/dist/trading.js.
//     Compares each of the 7 production functions output-by-output for
//     numerical agreement on a 100-point random grid.
//
//   PART C — buy-then-sell PnL invariant + position bookkeeping
//     C1. replayOrders mass balance: totalBuy − totalSell + position·price = 0
//          when no realised PnL has actually occurred (closing trade)
//     C2. avgCostBasis non-negative
// =============================================================================

import {
  bFromLiquidity,
  optionOnePrice, optionPrices, lmsrCost,
  buyExecution, sellExecution,
  maxPriceWithSlippage, minPriceWithSlippage,
  recapitalization, sharesFromBudget,
  buyThenSellRoundTrip,
  avgCostBasis, netPosition, replayOrders,
} from '../general/factmachine-math';
import {LMSR} from '../main-factmachine';
import {mulberry32} from '../general/prng';

const PROD_TRADING_PATH =
  process.env.FACTMACHINE_TRADING_PATH ??
  '/Users/maca5/codes/factmachine/factmachine-monorepo/packages/math/dist/trading.js';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-10): void {
  const d = Math.abs(a - b);
  check(label, d <= tol, `|${a} − ${b}| = ${d.toExponential(2)}`);
}

// -----------------------------------------------------------------------------
// PART A — invariants (DOES NOT REQUIRE THE PRODUCTION PACKAGE)
// -----------------------------------------------------------------------------
console.log('\n=== PART A — invariants on float64 math layer ===\n');

const rng = mulberry32(424242);

// Generate a property-style grid. Production tests bound |q2-q1|/b ≤ ~14
// with shares up to 10_000 and L ≥ 500.
const TRIALS = 200;

(function partA() {
  for (let t = 0; t < TRIALS; t++) {
    const q1  = rng() * 10_000;
    const q2  = rng() * 10_000;
    const liq = 500 + rng() * 100_000;
    const b   = bFromLiquidity(liq);
    const p   = optionPrices(q1, q2, b);
    if (Math.abs(p.optionOne + p.optionTwo - 1) > 1e-12) {
      check('A1 prices sum to 1 (sample failure)', false, `t=${t}, q1=${q1}, q2=${q2}, sum=${p.optionOne + p.optionTwo}`); return;
    }
    if (!(p.optionOne > 0 && p.optionOne < 1 && p.optionTwo > 0 && p.optionTwo < 1)) {
      check('A2 prices strictly in (0, 1)', false, `t=${t}, p1=${p.optionOne}, p2=${p.optionTwo}`); return;
    }
  }
  check(`A1 prices sum to 1 across ${TRIALS} random trials`, true);
  check(`A2 prices strictly in (0, 1) across ${TRIALS} random trials`, true);
})();

(function () {
  for (let t = 0; t < 50; t++) {
    const q   = rng() * 5_000;
    const liq = 500 + rng() * 50_000;
    const b   = bFromLiquidity(liq);
    const p   = optionPrices(q, q, b);
    if (Math.abs(p.optionOne - 0.5) > 1e-12) {
      check('A3 equal-shares ⇒ price = 0.5', false, `t=${t}, q=${q}, p=${p.optionOne}`); return;
    }
  }
  check('A3 equal-shares ⇒ price = 0.5 across 50 trials', true);
})();

(function () {
  for (let t = 0; t < 50; t++) {
    const q1  = rng() * 5_000, q2 = rng() * 5_000;
    const b   = bFromLiquidity(500 + rng() * 50_000);
    const p12 = optionPrices(q1, q2, b);
    const p21 = optionPrices(q2, q1, b);
    if (Math.abs(p12.optionOne - p21.optionTwo) > 1e-12) {
      check('A4 symmetry under share swap', false, `t=${t}`); return;
    }
  }
  check('A4 p₁(q1, q2) = p₂(q2, q1) across 50 trials', true);
})();

(function () {
  for (let t = 0; t < 50; t++) {
    const q1 = rng() * 1_000, q2 = rng() * 1_000;
    const delta = 1 + rng() * 100;
    const b = bFromLiquidity(500 + rng() * 50_000);
    if (optionPrices(q1 + delta, q2, b).optionOne < optionPrices(q1, q2, b).optionOne - 1e-12) {
      check('A5 more shares on option-1 raises price-1', false, `t=${t}`); return;
    }
  }
  check('A5 more shares on option-1 raises price-1 across 50 trials', true);
})();

(function () {
  for (let t = 0; t < 50; t++) {
    const amount = 0.01 + rng() * 100;
    const q1 = rng() * 1_000, q2 = rng() * 1_000;
    const b  = bFromLiquidity(500 + rng() * 50_000);
    const fee = Math.floor(rng() * 1000);
    const r = buyExecution({amount, optionOneShares: q1, optionTwoShares: q2, b, feeBps: fee, isOptionOne: rng() < 0.5});
    if (Math.abs(r.buyAmount + r.feeAmount - amount) > 1e-9) {
      check('A6 buy: fee + buyAmount = total', false, `t=${t}`); return;
    }
    if (!(r.shares > 0)) {
      check('A7 buy: amount>0 ⇒ shares>0', false, `t=${t}`); return;
    }
    if (Math.abs(r.averagePrice - r.buyAmount / r.shares) > 1e-12) {
      check('A8 buy: averagePrice = buyAmount / shares', false, `t=${t}`); return;
    }
    if (!(r.averagePrice < 1)) {
      check('A9 buy: averagePrice < 1', false, `t=${t}, ap=${r.averagePrice}`); return;
    }
  }
  check('A6 buy: fee + buyAmount = total (50 trials)', true);
  check('A7 buy: amount > 0 ⇒ shares > 0 (50 trials)', true);
  check('A8 buy: averagePrice = buyAmount / shares (50 trials)', true);
  check('A9 buy: averagePrice strictly < 1 (50 trials)', true);
})();

(function () {
  for (let t = 0; t < 30; t++) {
    const a = 0.01 + rng() * 50, bAmt = 0.01 + rng() * 50;
    const q1 = rng() * 100, q2 = rng() * 100;
    const b = bFromLiquidity(500 + rng() * 50_000);
    const small = Math.min(a, bAmt), large = Math.max(a, bAmt);
    const rs = buyExecution({amount: small, optionOneShares: q1, optionTwoShares: q2, b});
    const rl = buyExecution({amount: large, optionOneShares: q1, optionTwoShares: q2, b});
    if (rl.shares < rs.shares - 1e-12) {
      check('A10 buy: monotone in spending', false, `t=${t}, small=${small}→${rs.shares}, large=${large}→${rl.shares}`); return;
    }
  }
  check('A10 buy: monotone in spending (30 trials)', true);
})();

(function () {
  for (let t = 0; t < 30; t++) {
    const sharesOut = 0.01 + rng() * 30;
    const q1 = sharesOut + 50 + rng() * 100, q2 = rng() * 100;
    const b = bFromLiquidity(500 + rng() * 50_000);
    const fee = Math.floor(rng() * 1000);
    const r = sellExecution({sharesOut, optionOneShares: q1, optionTwoShares: q2, b, feeBps: fee});
    if (Math.abs(r.usdcOut + r.feeAmount - r.sellAmount) > 1e-9) {
      check('A11 sell: usdcOut + fee = sellAmount', false, `t=${t}`); return;
    }
  }
  check('A11 sell: usdcOut + fee = sellAmount (30 trials)', true);
})();

(function () {
  for (let t = 0; t < 30; t++) {
    const a = 0.01 + rng() * 30, bAmt = 0.01 + rng() * 30;
    const q1 = a + bAmt + 50, q2 = rng() * 100;
    const b = bFromLiquidity(500 + rng() * 50_000);
    const small = Math.min(a, bAmt), large = Math.max(a, bAmt);
    const rs = sellExecution({sharesOut: small, optionOneShares: q1, optionTwoShares: q2, b});
    const rl = sellExecution({sharesOut: large, optionOneShares: q1, optionTwoShares: q2, b});
    if (rl.usdcOut < rs.usdcOut - 1e-12) {
      check('A12 sell: monotone in shares', false, `t=${t}`); return;
    }
  }
  check('A12 sell: monotone in shares (30 trials)', true);
})();

(function () {
  let maxLoss = 0;
  for (let t = 0; t < 100; t++) {
    const amount = 1 + rng() * 100;
    const q1 = rng() * 200, q2 = rng() * 200;
    const b = bFromLiquidity(500 + rng() * 50_000);
    const isOne = rng() < 0.5;
    const rt = buyThenSellRoundTrip({amount, optionOneShares: q1, optionTwoShares: q2, b, isOptionOne: isOne});
    if (rt.net > 1e-8) {
      check('A13 round-trip: sell-after-buy ≤ amount (no-arb)', false, `t=${t}, net=${rt.net.toExponential(2)}`); return;
    }
    if (-rt.net > maxLoss) maxLoss = -rt.net;
  }
  check(`A13 round-trip ≤ buy amount across 100 trials  (max market-maker spread = ${maxLoss.toExponential(2)})`, true);
})();

(function () {
  for (let t = 0; t < 30; t++) {
    const q1 = rng() * 1_000, q2 = rng() * 1_000;
    const liqOld = 500 + rng() * 50_000;
    const liqNew = liqOld * (1.1 + rng() * 2);
    const bOld = bFromLiquidity(liqOld), bNew = bFromLiquidity(liqNew);
    const before = optionPrices(q1, q2, bOld);
    const r = recapitalization({optionOneShares: q1, optionTwoShares: q2, currentB: bOld, newB: bNew});
    const after = optionPrices(r.newOptionOneShares, r.newOptionTwoShares, r.newB);
    if (Math.abs(after.optionOne - before.optionOne) > 1e-10) {
      check('A14 recapitalisation preserves prices', false, `t=${t}, |Δp|=${Math.abs(after.optionOne - before.optionOne).toExponential(2)}`); return;
    }
  }
  check('A14 recapitalisation preserves prices (30 trials)', true);
})();

(function () {
  for (let t = 0; t < 30; t++) {
    const price = 0.01 + rng() * 0.98;
    const slip = Math.floor(rng() * 2000);
    const max = maxPriceWithSlippage(price, slip);
    const min = minPriceWithSlippage(price, slip);
    if (!(max >= 0 && max <= 1 && min >= 0 && min <= 1)) {
      check('A15 slippage clamps to [0, 1]', false, `t=${t}, max=${max}, min=${min}`); return;
    }
    if (!(max >= price - 1e-15 && min <= price + 1e-15)) {
      check('A15 maxPrice ≥ price ≥ minPrice', false, `t=${t}, price=${price}, max=${max}, min=${min}`); return;
    }
  }
  check('A15 slippage clamps to [0, 1] and maxPrice ≥ price ≥ minPrice (30 trials)', true);
})();

// -----------------------------------------------------------------------------
// PART B — cross-validate against the production @factmachine/math package
// -----------------------------------------------------------------------------
console.log('\n=== PART B — cross-validation vs production @factmachine/math ===\n');

let prod: any = null;
try { prod = require(PROD_TRADING_PATH); }
catch (e) {
  console.log('  SKIP (production package not loadable at ' + PROD_TRADING_PATH + ')');
  console.log('  Set FACTMACHINE_TRADING_PATH to override; or `pnpm -C ../.. build` in the monorepo.');
}

if (prod) {
  const Decimal = require('decimal.js');
  const D = (x: number) => new Decimal(x);

  // B1 — bFromLiquidity ≡ computeBFromInitialLiquidity
  let maxBDiff = 0;
  for (let t = 0; t < 30; t++) {
    const L = 500 + rng() * 50_000;
    const ours  = bFromLiquidity(L);
    const theirs = prod.computeBFromInitialLiquidity(D(L)).toNumber();
    maxBDiff = Math.max(maxBDiff, Math.abs(ours - theirs));
  }
  check(`B1  bFromLiquidity   ≡  computeBFromInitialLiquidity   (30 trials)`,
        maxBDiff < 1e-10, `max|Δ|=${maxBDiff.toExponential(2)}`);

  // B2 — optionOnePrice
  let maxPDiff = 0;
  for (let t = 0; t < 50; t++) {
    const q1 = rng() * 5_000, q2 = rng() * 5_000;
    const L  = 500 + rng() * 50_000;
    const b  = bFromLiquidity(L);
    const ours  = optionOnePrice(q1, q2, b);
    const theirs = prod.computeOptionOnePrice({qOne: D(q1), qTwo: D(q2), b: D(b)}).toNumber();
    maxPDiff = Math.max(maxPDiff, Math.abs(ours - theirs));
  }
  check(`B2  optionOnePrice  ≡  computeOptionOnePrice            (50 trials)`,
        maxPDiff < 1e-12, `max|Δ|=${maxPDiff.toExponential(2)}`);

  // B3 — lmsrCost
  let maxCDiff = 0;
  for (let t = 0; t < 50; t++) {
    const q1 = rng() * 5_000, q2 = rng() * 5_000;
    const b  = bFromLiquidity(500 + rng() * 50_000);
    const ours  = lmsrCost(q1, q2, b);
    const theirs = prod.computeLmsrCost({optionOneShares: D(q1), optionTwoShares: D(q2), b: D(b)}).toNumber();
    const rel = Math.abs(ours - theirs) / Math.max(1, Math.abs(theirs));
    maxCDiff = Math.max(maxCDiff, rel);
  }
  check(`B3  lmsrCost        ≡  computeLmsrCost                 (50 trials, rel)`,
        maxCDiff < 1e-12, `max rel|Δ|=${maxCDiff.toExponential(2)}`);

  // B4 — buyExecution
  let maxBuyShares = 0, maxBuyAmount = 0;
  for (let t = 0; t < 50; t++) {
    const amount = 0.01 + rng() * 100;
    const q1 = rng() * 1_000, q2 = rng() * 1_000;
    const b  = bFromLiquidity(500 + rng() * 50_000);
    const fee = Math.floor(rng() * 1000);
    const isOne = rng() < 0.5;
    const ours  = buyExecution({amount, optionOneShares: q1, optionTwoShares: q2, b, feeBps: fee, isOptionOne: isOne});
    const theirs = prod.computeBuyExecution({
      amount: D(amount), optionOneShares: D(q1), optionTwoShares: D(q2),
      b: D(b), feeBps: D(fee), isOptionOne: isOne,
    });
    maxBuyShares  = Math.max(maxBuyShares,  Math.abs(ours.shares - theirs.shares.toNumber()));
    maxBuyAmount  = Math.max(maxBuyAmount,  Math.abs(ours.buyAmount - theirs.buyAmount.toNumber()));
  }
  check(`B4a buyExecution.shares       ≡  prod                       (50 trials)`,
        maxBuyShares < 1e-10, `max|Δ|=${maxBuyShares.toExponential(2)}`);
  check(`B4b buyExecution.buyAmount    ≡  prod                       (50 trials)`,
        maxBuyAmount < 1e-10, `max|Δ|=${maxBuyAmount.toExponential(2)}`);

  // B5 — sellExecution
  let maxSellOut = 0, maxSellGross = 0;
  for (let t = 0; t < 50; t++) {
    const sharesOut = 0.01 + rng() * 50;
    const q1 = sharesOut + 100 + rng() * 500, q2 = rng() * 500;
    const b  = bFromLiquidity(500 + rng() * 50_000);
    const fee = Math.floor(rng() * 1000);
    const ours  = sellExecution({sharesOut, optionOneShares: q1, optionTwoShares: q2, b, feeBps: fee});
    const theirs = prod.computeSellExecution({
      sharesOut: D(sharesOut), optionOneShares: D(q1), optionTwoShares: D(q2),
      b: D(b), feeBps: D(fee), isOptionOne: true,
    });
    maxSellOut   = Math.max(maxSellOut,   Math.abs(ours.usdcOut    - theirs.usdcOut.toNumber()));
    maxSellGross = Math.max(maxSellGross, Math.abs(ours.sellAmount - theirs.sellAmount.toNumber()));
  }
  check(`B5a sellExecution.usdcOut     ≡  prod                       (50 trials)`,
        maxSellOut < 1e-10, `max|Δ|=${maxSellOut.toExponential(2)}`);
  check(`B5b sellExecution.sellAmount  ≡  prod                       (50 trials)`,
        maxSellGross < 1e-10, `max|Δ|=${maxSellGross.toExponential(2)}`);

  // B6 — slippage
  let maxSlip = 0;
  for (let t = 0; t < 30; t++) {
    const price = 0.01 + rng() * 0.98;
    const slip  = Math.floor(rng() * 2000);
    const oursMax  = maxPriceWithSlippage(price, slip);
    const theirsMax = prod.computeMaxPrice({price: D(price), slippageBps: D(slip)}).toNumber();
    const oursMin  = minPriceWithSlippage(price, slip);
    const theirsMin = prod.computeMinPrice({price: D(price), slippageBps: D(slip)}).toNumber();
    maxSlip = Math.max(maxSlip, Math.abs(oursMax - theirsMax), Math.abs(oursMin - theirsMin));
  }
  check(`B6  slippage         ≡  computeMax/MinPrice              (30 trials)`,
        maxSlip < 1e-12, `max|Δ|=${maxSlip.toExponential(2)}`);

  // B7 — recapitalisation
  let maxRecapShares = 0, maxRecapDelta = 0;
  for (let t = 0; t < 30; t++) {
    const q1 = rng() * 500, q2 = rng() * 500;
    const liqOld = 500 + rng() * 50_000;
    const liqNew = liqOld * (1.1 + rng() * 2);
    const bOld = bFromLiquidity(liqOld), bNew = bFromLiquidity(liqNew);
    const ours  = recapitalization({optionOneShares: q1, optionTwoShares: q2, currentB: bOld, newB: bNew});
    const theirs = prod.computeRecapitalization({
      optionOneShares: D(q1), optionTwoShares: D(q2),
      currentB: D(bOld), newB: D(bNew),
    });
    maxRecapShares = Math.max(maxRecapShares, Math.abs(ours.newOptionOneShares - theirs.newOptionOneShares.toNumber()));
    maxRecapDelta  = Math.max(maxRecapDelta,  Math.abs(ours.capitalDelta       - theirs.capitalDelta.toNumber()));
  }
  check(`B7a recapitalisation shares   ≡  prod                     (30 trials)`,
        maxRecapShares < 1e-10, `max|Δ|=${maxRecapShares.toExponential(2)}`);
  check(`B7b recapitalisation Δcapital ≡  prod                     (30 trials)`,
        maxRecapDelta  < 1e-9,  `max|Δ|=${maxRecapDelta.toExponential(2)}`);

  // B8 — PnL aggregator (replayOrders ↔ getOrderAggregates)
  const orders = [
    {action: 'BUY' as const,  shares: 5,  usdc: 1.50, time: 1},
    {action: 'BUY' as const,  shares: 7,  usdc: 2.20, time: 2},
    {action: 'SELL' as const, shares: 4,  usdc: 1.30, time: 3},
    {action: 'BUY' as const,  shares: 3,  usdc: 1.10, time: 4},
    {action: 'SELL' as const, shares: 2,  usdc: 0.80, time: 5},
  ];
  const ours = replayOrders(orders);
  const theirs = prod.getOrderAggregates(orders);
  close('B8a replay netPosition       ≡  prod', ours.netPosition,  theirs.netPosition.toNumber(),  1e-12);
  close('B8b replay realizedPnl       ≡  prod', ours.realizedPnl,  theirs.realizedPnl.toNumber(),  1e-12);
  close('B8c replay avgCostBasis      ≡  prod', ours.avgCostBasis, theirs.avgCostBasis.toNumber(), 1e-12);
  close('B8d replay totalVolume       ≡  prod', ours.totalVolume,  theirs.totalVolume.toNumber(),  1e-12);
}

// -----------------------------------------------------------------------------
// PART C — POMDP-side: LMSR class adopts production conventions
// -----------------------------------------------------------------------------
console.log('\n=== PART C — POMDP LMSR class uses production conventions ===\n');

(function () {
  // Default (production) semantics: liquidity L = 50, expected b = 50/ln(2).
  const m = new LMSR(50, 2);
  close('C1 default LMSR(50, 2) gives b = 50/ln(2)', m.b, 50 / Math.log(2), 1e-12);
  // Legacy semantics: liquidityIsB explicit opt-in.
  const mLegacy = new LMSR(50, 2, {liquidityIsB: true});
  close('C2 LMSR(50, 2, {liquidityIsB:true}) gives b = 50', mLegacy.b, 50, 1e-12);
  // Equal q ⇒ price 0.5
  const p = m.binaryPrices();
  close('C3 LMSR.binaryPrices() at q=0 returns 0.5/0.5', p.optionOne, 0.5, 1e-12);
  // Buy: production-equivalent execution.
  const exec = m.buy(10, true, 0);
  check('C4 LMSR.buy(10, YES) returns positive shares', exec.shares > 0, `shares=${exec.shares.toFixed(6)}`);
  // After the buy, P(YES) must have risen.
  const pAfter = m.binaryPrices();
  check('C5 P(YES) rose after buying YES', pAfter.optionOne > 0.5,
        `P(YES) before=0.5, after=${pAfter.optionOne.toFixed(6)}`);
  // Round-trip: sell back the same shares; usdcOut ≤ amount.
  const sellBack = m.sell(exec.shares, true, 0);
  check('C6 LMSR sell-back ≤ original amount (no-arb)', sellBack.usdcOut <= 10 + 1e-9,
        `sellBack=${sellBack.usdcOut.toFixed(6)} ≤ 10`);
})();

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
