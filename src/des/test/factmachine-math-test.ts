// RUST MIGRATION: Port file-for-file to `tests/factmachine_math_test.rs` unless the float64 math layer lands with local `#[cfg(test)]` tests.
// Test-port notes: translate LMSR/math checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and approximate-float helpers; pin tolerances explicitly.

'use strict';

// =============================================================================
// test/factmachine-math-test.ts — focused unit tests for the float64
// FactMachine math layer (mirrors `factmachine-monorepo/.../lmsr.spec.ts`).
// =============================================================================

import {
  bFromLiquidity, optionOnePrice, optionPrices, lmsrCost,
  buyExecution, sellExecution, sharesFromBudget,
  maxPriceWithSlippage, minPriceWithSlippage,
  recapitalization, buyThenSellRoundTrip,
  avgCostBasis, netPosition, replayOrders, finalPnl, unrealizedPnl,
} from '../general/factmachine-math';

let pass = 0, fail = 0;
function expect(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-12): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 — bFromLiquidity');
// -----------------------------------------------------------------------------
{
  close('L=ln(2) → b=1', bFromLiquidity(Math.log(2)), 1);
  close('L=100 → b=100/ln(2)', bFromLiquidity(100), 100 / Math.log(2));
  let threw = false; try { bFromLiquidity(0); } catch (e) { threw = true; }
  expect('L=0 throws', threw);
  threw = false; try { bFromLiquidity(-1); } catch (e) { threw = true; }
  expect('L<0 throws', threw);
  threw = false; try { bFromLiquidity(NaN); } catch (e) { threw = true; }
  expect('L=NaN throws', threw);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 — optionOnePrice / optionPrices');
// -----------------------------------------------------------------------------
{
  close('q1=q2 → 0.5', optionOnePrice(50, 50, 100), 0.5);
  close('q1>>q2 → near 1', optionOnePrice(1000, 0, 100), 1 / (1 + Math.exp(-10)));
  close('q1<<q2 → near 0', optionOnePrice(0, 1000, 100), 1 / (1 + Math.exp(10)));
  let threw = false; try { optionOnePrice(0, 0, 0); } catch (e) { threw = true; }
  expect('b=0 throws', threw);
  // Symmetry
  for (let t = 0; t < 10; t++) {
    const q1 = Math.random() * 100, q2 = Math.random() * 100, b = 50;
    const p12 = optionPrices(q1, q2, b);
    const p21 = optionPrices(q2, q1, b);
    if (Math.abs(p12.optionOne - p21.optionTwo) > 1e-12) {
      expect('symmetry holds', false); break;
    }
  }
  expect('p₁(q1,q2) = p₂(q2,q1) over 10 trials', true);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 — lmsrCost monotonicity');
// -----------------------------------------------------------------------------
{
  // C(q + e_i) > C(q) for any e_i > 0 (shifting a single q up costs USDC).
  const before = lmsrCost(0, 0, 50);
  const afterY = lmsrCost(10, 0, 50);
  const afterN = lmsrCost(0, 10, 50);
  expect('cost increased after buying YES', afterY > before, `${before} → ${afterY}`);
  expect('cost increased after buying NO',  afterN > before, `${before} → ${afterN}`);
  // C(q1, q2) = C(q2, q1) by symmetry of log-sum-exp.
  for (let t = 0; t < 10; t++) {
    const q1 = Math.random() * 100, q2 = Math.random() * 100;
    if (Math.abs(lmsrCost(q1, q2, 50) - lmsrCost(q2, q1, 50)) > 1e-12) {
      expect('cost symmetric', false); break;
    }
  }
  expect('cost C(q1,q2) = C(q2,q1) over 10 trials', true);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 — buyExecution');
// -----------------------------------------------------------------------------
{
  const r = buyExecution({amount: 10, optionOneShares: 0, optionTwoShares: 0, b: 100});
  expect('positive amount yields positive shares', r.shares > 0, `shares = ${r.shares}`);
  close('zero fee → buyAmount = amount', r.buyAmount, 10);
  close('zero fee → feeAmount = 0', r.feeAmount, 0);
  close('reward = shares', r.reward, r.shares);
  close('averagePrice = buyAmount / shares', r.averagePrice, r.buyAmount / r.shares);

  const r2 = buyExecution({amount: 10, optionOneShares: 0, optionTwoShares: 0, b: 100, feeBps: 100});
  close('feeBps=100 → fee = 1% of amount', r2.feeAmount, 0.10);
  close('feeBps=100 → buyAmount = 99% of amount', r2.buyAmount, 9.90);

  let threw = false;
  try { buyExecution({amount: 10, optionOneShares: 0, optionTwoShares: 0, b: 100, feeBps: 10000}); }
  catch (e) { threw = true; }
  expect('feeBps ≥ 10000 throws', threw);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 — sellExecution');
// -----------------------------------------------------------------------------
{
  // Build a market with shares first, then sell some back.
  const r = sellExecution({sharesOut: 5, optionOneShares: 50, optionTwoShares: 30, b: 100});
  expect('sellAmount ≥ 0', r.sellAmount >= 0);
  close('zero fee → usdcOut = sellAmount', r.usdcOut, r.sellAmount);
  close('reward = usdcOut', r.reward, r.usdcOut);
  close('averagePrice = sellAmount / sharesOut', r.averagePrice, r.sellAmount / 5);

  const r2 = sellExecution({sharesOut: 5, optionOneShares: 50, optionTwoShares: 30, b: 100, feeBps: 50});
  close('feeBps=50 → usdcOut + fee = sellAmount', r2.usdcOut + r2.feeAmount, r2.sellAmount);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 — buy-then-sell round-trip is non-positive');
// -----------------------------------------------------------------------------
{
  let maxLoss = 0;
  for (let t = 0; t < 30; t++) {
    const amount = 1 + Math.random() * 50;
    const q1 = Math.random() * 200, q2 = Math.random() * 200;
    const b = 1000;
    const rt = buyThenSellRoundTrip({amount, optionOneShares: q1, optionTwoShares: q2, b});
    if (rt.net > 1e-9) { expect('round-trip never profitable', false, `t=${t}, net=${rt.net}`); break; }
    if (-rt.net > maxLoss) maxLoss = -rt.net;
  }
  expect(`round-trip never profitable; max market-maker spread = ${maxLoss.toExponential(2)}`, true);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 — recapitalization');
// -----------------------------------------------------------------------------
{
  const before = optionPrices(50, 30, 100);
  const r = recapitalization({optionOneShares: 50, optionTwoShares: 30, currentB: 100, newB: 200});
  const after = optionPrices(r.newOptionOneShares, r.newOptionTwoShares, r.newB);
  close('prices preserved (option1)', after.optionOne, before.optionOne, 1e-12);
  close('prices preserved (option2)', after.optionTwo, before.optionTwo, 1e-12);
  expect('capital delta is positive', r.capitalDelta > 0, `Δ = ${r.capitalDelta}`);
  close('shares scaled by ratio (option1)', r.newOptionOneShares, 50 * 200 / 100, 1e-12);

  let threw = false;
  try { recapitalization({optionOneShares: 0, optionTwoShares: 0, currentB: 100, newB: 100}); }
  catch (e) { threw = true; }
  expect('equal currentB == newB throws', threw);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 8 — slippage helpers');
// -----------------------------------------------------------------------------
{
  close('zero slippage maxPrice = price', maxPriceWithSlippage(0.5, 0), 0.5);
  close('zero slippage minPrice = price', minPriceWithSlippage(0.5, 0), 0.5);
  expect('maxPrice ≥ price', maxPriceWithSlippage(0.5, 100) >= 0.5);
  expect('minPrice ≤ price', minPriceWithSlippage(0.5, 100) <= 0.5);
  close('clamp to [0, 1] (high price + slippage)', maxPriceWithSlippage(0.99, 5000), 1);
  close('clamp to [0, 1] (low price + slippage)', minPriceWithSlippage(0.01, 5000), 0.005);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 9 — PnL helpers');
// -----------------------------------------------------------------------------
{
  // Bought 10 shares for $4. Average cost basis = 0.4.
  const acb = avgCostBasis({totalSharesBought: 10, totalSharesSold: 0,
                            totalBuyAmount: 4, totalSellProceeds: 0, realizedPnl: 0});
  close('avgCostBasis = $4 / 10 = 0.4', acb, 0.4);

  close('netPosition: 10 bought, 3 sold = 7',
        netPosition({totalSharesBought: 10, totalSharesSold: 3}), 7);
  close('netPosition cannot be negative',
        netPosition({totalSharesBought: 3, totalSharesSold: 10}), 0);

  // Resolution: 7 winning shares at $1 each, $4 spent, $0 sell proceeds → +$3 PnL
  close('finalPnl: bought 10 for $4, hold 7 winners → +3',
        finalPnl({totalBuyAmount: 4, totalSellProceeds: 0, netPosition: 7, resolutionPrice: 1}), 3);

  // Position 7 at avg cost 0.4, current price 0.6 → unrealized $1.4
  close('unrealizedPnl: 7 × (0.6 − 0.4) = 1.4',
        unrealizedPnl({netPosition: 7, currentPrice: 0.6, avgCostBasis: 0.4}), 1.4);
  close('unrealizedPnl = 0 when no position',
        unrealizedPnl({netPosition: 0, currentPrice: 0.6, avgCostBasis: 0.4}), 0);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 10 — replayOrders');
// -----------------------------------------------------------------------------
{
  const orders = [
    {action: 'BUY' as const,  shares: 10, usdc: 4, time: 1},
    {action: 'SELL' as const, shares: 4,  usdc: 2, time: 2},
  ];
  const r = replayOrders(orders);
  close('totalSharesBought = 10', r.totalSharesBought, 10);
  close('totalSharesSold = 4', r.totalSharesSold, 4);
  close('totalBuyAmount = 4', r.totalBuyAmount, 4);
  close('totalSellProceeds = 2', r.totalSellProceeds, 2);
  close('netPosition = 6', r.netPosition, 6);
  close('totalVolume = 6', r.totalVolume, 6);
  expect('lastTime = 2', r.lastTime === 2);
  // Realised PnL on the sell: sold 4 shares @ avgCost 0.4 → cost basis 1.6,
  // proceeds 2 → realized PnL = 2 − 1.6 = +0.4.
  close('realizedPnl = +0.4', r.realizedPnl, 0.4, 1e-12);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 11 — sharesFromBudget edge cases');
// -----------------------------------------------------------------------------
{
  close('budget=0 → shares=0', sharesFromBudget(0, 0.5, 100), 0);
  close('budget<0 → shares=0', sharesFromBudget(-5, 0.5, 100), 0);
  let threw = false; try { sharesFromBudget(10, 0, 100); } catch (e) { threw = true; }
  expect('currentPrice=0 throws', threw);
  threw = false; try { sharesFromBudget(10, 0.5, 0); } catch (e) { threw = true; }
  expect('b=0 throws', threw);
  // Tiny budget: shares ≈ budget / currentPrice (linearisation).
  const small = sharesFromBudget(1e-6, 0.5, 100);
  close('small budget linearises: shares ≈ budget / price', small, 1e-6 / 0.5, 1e-9);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
