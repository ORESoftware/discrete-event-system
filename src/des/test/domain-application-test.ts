'use strict';

// Tests for the applied domain model pack.

import {getModel, runFromSpec} from '../general/des-registry';
import {
  DomainModelResult,
  runActiveLearningAcquisition,
  runAdaptiveFuzzyControl,
  runBottleneckProductionControl,
  runBuyerAwareDynamicPricing,
  runDynamicPricingRevenue,
  runEnergyStorageDispatch,
  runLogisticsRoutingHeuristics,
  runPortfolioDrawdownControl,
  runSupplyChainRiskPooling,
  runVisualDecisionFrontier,
  runWorkforceServiceOperations,
} from '../general/domain-application-models';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function metric(result: DomainModelResult<unknown>, key: string): number {
  const v = result.best.metrics[key];
  if (typeof v !== 'number') throw new Error(`metric ${key} is not numeric`);
  return v;
}

function commonGraphChecks(id: string, result: DomainModelResult<unknown>): void {
  check(`${id} has at least three candidate plans`, result.candidates.length >= 3, `n=${result.candidates.length}`);
  check(`${id} returns a feasible incumbent`, result.best.feasible, result.best.candidateId);
  check(`${id} uses source/generator/evaluator/sink stations`,
    result.topology.stations[0] === `${id}-scenario-source`
    && result.topology.stations.includes(`${id}-candidate-generator`)
    && result.topology.stations.includes(`${id}-plan-evaluator`)
    && result.topology.stations.includes(`${id}-result-sink`));
  check(`${id} exposes domain movables`,
    ['DomainScenarioToken', 'DomainPlanToken', 'DomainEvaluationToken'].every(t => result.topology.movables.includes(t)));
}

async function main(): Promise<void> {
  console.log('\n-- adaptive-fuzzy-control --');
  {
    const r = runAdaptiveFuzzyControl({});
    commonGraphChecks('adaptive-fuzzy-control', r);
    check('adaptive fuzzy control gets tracking RMS below 4C', metric(r, 'rmsError') < 4, `rms=${metric(r, 'rmsError')}`);
  }

  console.log('\n-- logistics-routing-heuristics --');
  {
    const r = runLogisticsRoutingHeuristics({});
    commonGraphChecks('logistics-routing-heuristics', r);
    check('routing incumbent has no capacity violation', metric(r, 'capacityViolation') === 0);
    check('routing distance is finite and positive', Number.isFinite(metric(r, 'routeDistance')) && metric(r, 'routeDistance') > 0);
  }

  console.log('\n-- bottleneck-production-control --');
  {
    const r = runBottleneckProductionControl({});
    commonGraphChecks('bottleneck-production-control', r);
    check('manufacturing incumbent ships most demand', metric(r, 'service') > 0.9, `service=${metric(r, 'service')}`);
  }

  console.log('\n-- supply-chain-risk-pooling --');
  {
    const r = runSupplyChainRiskPooling({});
    commonGraphChecks('supply-chain-risk-pooling', r);
    check('supply-chain incumbent achieves high fill rate', metric(r, 'fillRate') > 0.9, `fill=${metric(r, 'fillRate')}`);
  }

  console.log('\n-- workforce-service-operations --');
  {
    const r = runWorkforceServiceOperations({});
    commonGraphChecks('workforce-service-operations', r);
    check('operations incumbent meets service feasibility', metric(r, 'serviceLevel') >= 0.9, `service=${metric(r, 'serviceLevel')}`);
  }

  console.log('\n-- portfolio-drawdown-control --');
  {
    const r = runPortfolioDrawdownControl({});
    commonGraphChecks('portfolio-drawdown-control', r);
    check('financial control keeps drawdown bounded', metric(r, 'maxDrawdown') < 0.1, `drawdown=${metric(r, 'maxDrawdown')}`);
  }

  console.log('\n-- dynamic-pricing-revenue --');
  {
    const r = runDynamicPricingRevenue({});
    commonGraphChecks('dynamic-pricing-revenue', r);
    check('dynamic pricing earns positive revenue', metric(r, 'revenue') > 0, `revenue=${metric(r, 'revenue')}`);
    check('dynamic pricing sells material capacity', metric(r, 'sellThrough') > 0.5, `sellThrough=${metric(r, 'sellThrough')}`);
  }

  console.log('\n-- buyer-aware-dynamic-pricing --');
  {
    const r = runBuyerAwareDynamicPricing({});
    commonGraphChecks('buyer-aware-dynamic-pricing', r);
    check('buyer-aware pricing respects default privacy budget', metric(r, 'privacyViolations') === 0);
    check('buyer-aware pricing keeps perceived fairness bounded', metric(r, 'avgFairnessSpread') <= 0.205, `spread=${metric(r, 'avgFairnessSpread')}`);
    check('buyer-aware pricing preserves retention', metric(r, 'retentionIndex') >= 0.78, `retention=${metric(r, 'retentionIndex')}`);
    check('buyer-aware pricing exposes period trace for animation', r.best.trace !== undefined && r.best.trace.t.length > 0);
  }

  console.log('\n-- energy-storage-dispatch --');
  {
    const r = runEnergyStorageDispatch({});
    commonGraphChecks('energy-storage-dispatch', r);
    check('energy dispatch has no unserved load', metric(r, 'unserved') === 0);
    check('energy dispatch cost is finite', Number.isFinite(metric(r, 'cost')));
  }

  console.log('\n-- active-learning-acquisition --');
  {
    const r = runActiveLearningAcquisition({});
    commonGraphChecks('active-learning-acquisition', r);
    check('active learning produces meaningful error reduction',
      metric(r, 'expectedErrorReduction') > 0.3, `reduction=${metric(r, 'expectedErrorReduction')}`);
  }

  console.log('\n-- visual-decision-frontier --');
  {
    const r = runVisualDecisionFrontier({});
    commonGraphChecks('visual-decision-frontier', r);
    check('decision frontier emits visualization-ready metrics', r.best.metrics.visualizationReady === true);
    check('decision frontier identifies a top alternative', typeof r.best.metrics.topAlternative === 'string');
  }

  console.log('\n-- registry and JSON smoke --');
  const ids = [
    'adaptive-fuzzy-control',
    'logistics-routing-heuristics',
    'bottleneck-production-control',
    'supply-chain-risk-pooling',
    'workforce-service-operations',
    'portfolio-drawdown-control',
    'dynamic-pricing-revenue',
    'buyer-aware-dynamic-pricing',
    'energy-storage-dispatch',
    'active-learning-acquisition',
    'visual-decision-frontier',
  ];
  for (const id of ids) check(`registry has ${id}`, getModel(id).id === id);
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'dynamic-pricing-revenue',
    parameters: {capacity: 120},
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('runFromSpec executes domain application model', summary.modelId === 'dynamic-pricing-revenue');

  console.log('\n========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`domain-application-test: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
