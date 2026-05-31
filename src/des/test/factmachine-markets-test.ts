#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/factmachine_markets_test.rs   (integration test crate)
// 1:1 file move. Focused checks for the FactMachine multi-market MDP/POMDP
// simulator, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - the simulator is seeded (seed: 777, buildDailyMarketCaps(..,777)) ->
//     a seeded rand::Rng so portfolio runs are reproducible.
//   - MarketKind union + Partial<PortfolioConfig> overrides -> a Rust enum and
//     a config struct with Default + explicit field overrides.
// =============================================================================

// =============================================================================
// Focused checks for the FactMachine multi-market MDP/POMDP simulator.
// =============================================================================

import {
  buildDailyMarketCaps,
  buildOperatorMDP,
  dailyMarketCapForDay,
  defaultConfig,
  MarketKind,
  PortfolioConfig,
  runPortfolio,
} from '../main-factmachine-markets';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  (' + detail + ')' : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}

function testConfig(overrides: Partial<PortfolioConfig> = {}): PortfolioConfig {
  const horizonDays = 14;
  return {
    ...defaultConfig(),
    scenarioLabel: 'test-1k',
    horizonH: 24 * horizonDays,
    stepH: 1,
    maxConcurrent: 10,
    minDailyMarkets: 2,
    maxDailyMarkets: 10,
    dailyMarketCaps: buildDailyMarketCaps(horizonDays, 2, 10, 777),
    seed: 777,
    minMarketParticipants: 80,
    liquidity: 200,
    scalarBins: 7,
    ...overrides,
  };
}

console.log('\nT1  Daily launch-cap schedule');
{
  const caps = buildDailyMarketCaps(50, 2, 10, 42);
  check('50 daily caps generated', caps.length === 50, `length=${caps.length}`);
  check('daily caps stay inside [2, 10]', caps.every(x => x >= 2 && x <= 10), caps.join(','));
  check('schedule includes a 2-market day', caps.includes(2), caps.join(','));
  check('schedule includes a 10-market day', caps.includes(10), caps.join(','));
  const cfg = testConfig({dailyMarketCaps: caps, horizonH: 24 * 50});
  check('dailyMarketCapForDay clamps after horizon', dailyMarketCapForDay(999, cfg) === caps[caps.length - 1]);
}

console.log('\nT2  MDP/POMDP portfolio run captures contract kinds and daily summaries');
{
  const mdp = buildOperatorMDP();
  const cfg = testConfig();
  const run = runPortfolio('greedy-buzz', cfg, mdp);
  const kindTotals = new Map<MarketKind, number>();
  for (const row of run.kindBreakdown) kindTotals.set(row.kind, row.markets);
  const dailyOpenedOk = run.daily
    .filter(d => d.day < Math.ceil(cfg.horizonH / 24))
    .every(d => d.opened <= d.marketCap);
  const totalByKind = run.kindBreakdown.reduce((s, row) => s + row.markets, 0);
  const totalByDay = run.daily.reduce((s, row) => s + row.closed, 0);
  check('opened markets obey each day cap', dailyOpenedOk);
  check('run has binary markets', (kindTotals.get('binary') ?? 0) > 0, JSON.stringify(run.kindBreakdown));
  check('run has scalar markets', (kindTotals.get('scalar') ?? 0) > 0, JSON.stringify(run.kindBreakdown));
  check('kind breakdown sums to closed markets', totalByKind === run.closedMarkets.length,
        `${totalByKind} vs ${run.closedMarkets.length}`);
  check('daily summaries sum to closed markets', totalByDay === run.closedMarkets.length,
        `${totalByDay} vs ${run.closedMarkets.length}`);
  check('timeline captures bettors and trades', run.timeline.some(x => x.bettors > 0 && x.trades > 0));
  check('run captures opinion sampling error',
        run.closedMarkets.every(m => Number.isFinite(m.opinionSamplingError) && m.opinionSamplingError >= 0 && m.opinionSamplingError <= 1));
  check('run captures prediction-market Brier score',
        run.closedMarkets.every(m => Number.isFinite(m.predictionBrierScore) && m.predictionBrierScore >= 0 && m.predictionBrierScore <= 1));
  check('aggregate separates opinion and prediction accuracy',
        Number.isFinite(run.aggregate.avgOpinionSamplingError) && Number.isFinite(run.aggregate.avgPredictionBrierScore),
        `opinion=${run.aggregate.avgOpinionSamplingError} brier=${run.aggregate.avgPredictionBrierScore}`);
  check('no market opens after the 14-day acceptance horizon',
        run.closedMarkets.every(m => m.openAt < cfg.horizonH),
        `max openAt=${Math.max(...run.closedMarkets.map(m => m.openAt)).toFixed(2)}`);
}

console.log('\nT3  1k vs 10k participant-scale comparison stays visible');
{
  const mdp = buildOperatorMDP();
  const low = runPortfolio('pomdp-belief', testConfig({
    scenarioLabel: '1,000 participants',
    minMarketParticipants: 1000,
  }), mdp);
  const high = runPortfolio('pomdp-belief', testConfig({
    scenarioLabel: '10,000 participants',
    minMarketParticipants: 10000,
  }), mdp);
  check('10k run records at least as many votes as 1k run',
        high.aggregate.votes >= low.aggregate.votes,
        `1k=${low.aggregate.votes} 10k=${high.aggregate.votes}`);
  check('10k run records at least as many bettors as 1k run',
        high.aggregate.bettors >= low.aggregate.bettors,
        `1k=${low.aggregate.bettors} 10k=${high.aggregate.bettors}`);
  check('both scales carry daily summaries',
        low.daily.length >= 14 && high.daily.length >= 14,
        `1k=${low.daily.length} 10k=${high.daily.length}`);
}

console.log(`\nsummary: ${pass} pass, ${fail} fail`);
if (fail > 0) process.exit(1);
