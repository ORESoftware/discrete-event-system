'use strict';

// =============================================================================
// Tests for statistical + stochastic optimisation extensions.
// =============================================================================

import * as fs from 'fs';
import {
  fitDistribution,
  runAdaptiveSimOpt,
  runCapacityExpansionSDDP,
  runDistributionFit,
  runRiskCapacity,
  sampleFittedDistribution,
} from '../general/statistical-optimization';
import {getModel, runFromSpec} from '../general/des-registry';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

async function checkThrows(label: string, fn: () => unknown | Promise<unknown>, contains?: string): Promise<void> {
  let threw = false;
  let message = '';
  try {
    await fn();
  } catch (e) {
    threw = true;
    message = e instanceof Error ? e.message : String(e);
  }
  const matched = contains === undefined || message.includes(contains);
  check(label, threw && matched, threw ? `message=${JSON.stringify(message)}` : 'did not throw');
}

async function main(): Promise<void> {
console.log('\n[1] Distribution fitting: MLE vs method of moments');
{
  const samples = [8, 9, 10, 11, 12, 9, 10, 11];
  const normalMLE = fitDistribution(samples, 'normal', 'mle');
  const normalMOM = fitDistribution(samples, 'normal', 'moments');
  check('1.1 normal MLE mean equals sample mean', close(normalMLE.mean, 10));
  check('1.2 MLE variance uses n denominator', close(normalMLE.variance, 1.5));
  check('1.3 method-of-moments variance uses unbiased variance', close(normalMOM.variance, 12 / 7));

  const fit = runDistributionFit({samples, families: ['normal', 'gamma', 'empirical'], methods: ['mle', 'moments']});
  check('1.4 distribution-fit validators pass', fit.validation.every(c => c.passed));
  check('1.5 AIC sorting puts best first', fit.fits[0].aic <= fit.fits[fit.fits.length - 1].aic);
}

console.log('\n[2] Risk-aware scenario optimisation');
{
  const chance = runRiskCapacity({
    cost: [10, 12],
    price: [25, 28],
    demand: {kind: 'uniform', ranges: [{low: 50, high: 100}, {low: 40, high: 80}]},
    numScenarios: 60,
    seed: 5,
    xMax: 120,
    step: 20,
    risk: {kind: 'chance', minServiceLevel: 0.8, shortfallLimit: 10},
  });
  check('2.1 chance model validators pass', chance.validation.every(c => c.passed));
  check('2.2 selected candidate satisfies service level', chance.best.serviceLevel >= 0.8 - 1e-12);

  const dro = runRiskCapacity({
    cost: [10],
    price: [25],
    demand: {kind: 'uniform', ranges: [{low: 20, high: 80}]},
    numScenarios: 50,
    seed: 9,
    xMax: 100,
    step: 10,
    risk: {kind: 'dro', radius: 1.0},
  });
  check('2.3 DRO-lite objective penalises volatility', dro.best.robustObjective <= dro.best.meanProfit + 1e-9);

  const empirical = runRiskCapacity({
    cost: [10],
    price: [25],
    demand: {kind: 'empirical', empirical: [[{value: 20, prob: 0.5}, {value: 60, prob: 0.5}]]},
    numScenarios: 30,
    seed: 4,
    xMax: 80,
    step: 20,
    risk: {kind: 'expectation'},
  });
  check('2.4 empirical point-mass demand is accepted', empirical.validation.every(c => c.passed));
}

console.log('\n[3] Multi-stage SDDP-style capacity expansion');
{
  const r = runCapacityExpansionSDDP({
    horizon: 3,
    demand: [{low: 20, high: 50}, {low: 30, high: 70}, {low: 40, high: 90}],
    price: [25, 24, 23],
    expansionCost: [12, 10, 8],
    initialCapacity: 0,
    xMax: 100,
    step: 10,
    samplesPerStage: 30,
    seed: 7,
    maxIter: 25,
    tol: 0.01,
  });
  check('3.1 SDDP validators pass', r.validation.every(c => c.passed));
  check('3.2 upper bound dominates exact grid objective', r.finalUpperBound + 1e-6 >= r.exactObjective);
  check('3.3 lower bound no better than exact objective', r.finalLowerBound <= r.exactObjective + 1e-6);
  check('3.4 final gap closes on small sampled-grid problem', r.gap <= 1e-5, `gap=${r.gap}`);
}

console.log('\n[4] Adaptive simulation optimisation');
{
  const r = runAdaptiveSimOpt({
    cost: [10, 12],
    price: [25, 28],
    demand: {kind: 'uniform', ranges: [{low: 50, high: 100}, {low: 40, high: 80}]},
    alternatives: [
      {name: 'lean', x: [60, 50]},
      {name: 'balanced', x: [80, 65]},
      {name: 'buffered', x: [100, 80]},
    ],
    seed: 11,
    initialSamples: 3,
    budget: 45,
    batchSize: 3,
    exploration: 1.5,
  });
  const total = r.stats.reduce((s, a) => s + a.n, 0);
  check('4.1 adaptive validators pass', r.validation.every(c => c.passed));
  check('4.2 budget is consumed exactly for divisible batch', total === 45, `total=${total}`);
  check('4.3 each alternative received initial samples', r.stats.every(s => s.n >= 3));
  check('4.4 best has finite standard error', Number.isFinite(r.best.stderr));
}

console.log('\n[5] JSON registry smoke tests');
{
  for (const id of ['stochastic-lp', 'distribution-fit', 'risk-capacity', 'sddp-capacity', 'adaptive-simopt']) {
    const reg = getModel(id);
    check(`5.registry has ${id}`, reg.id === id);
  }
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'risk-capacity',
    parameters: {
      cost: [10],
      price: [25],
      demand: {kind: 'uniform', ranges: [{low: 20, high: 50}]},
      numScenarios: 20,
      seed: 3,
      xMax: 60,
      step: 20,
      risk: {kind: 'expectation'},
    },
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('5.runFromSpec executes risk-capacity JSON without animation', summary.modelId === 'risk-capacity');

  const animSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'distribution-fit',
    parameters: {samples: [1, 2, 3, 4, 5], families: ['normal'], methods: ['mle']},
    runtime: {verbose: false},
  }, {verbose: false});
  check('5.default animation writes html output', animSummary.outputs.some(o => o.kind === 'html' && o.path === 'out/distribution-fit.html'));
  check('5.default animation writes frames output', animSummary.outputs.some(o => o.kind === 'frames' && o.path === 'out/distribution-fit.frames.jsonl'));

  const fittedDemandSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'risk-capacity',
    parameters: {
      cost: [10],
      price: [25],
      demand: {kind: 'fitted', fitted: [{family: 'normal', method: 'mle', params: {mu: 40, sigma: 5}}]},
      numScenarios: 20,
      seed: 8,
      xMax: 60,
      step: 20,
      risk: {kind: 'expectation'},
    },
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('5.fitted distribution demand works through JSON schema', fittedDemandSummary.modelId === 'risk-capacity');
}

console.log('\n[6] Fail-fast preconditions');
{
  let threw = false;
  try {
    runCapacityExpansionSDDP({
      horizon: 1,
      demand: [{low: 10, high: 5}],
      price: [20],
      expansionCost: [10],
      initialCapacity: 0,
      xMax: 30,
      step: 10,
      samplesPerStage: 5,
      seed: 1,
    });
  } catch {
    threw = true;
  }
  check('6.1 SDDP rejects inverted demand range', threw);

  threw = false;
  try {
    runAdaptiveSimOpt({
      cost: [10],
      price: [20],
      demand: {kind: 'uniform', ranges: [{low: 0, high: 10}]},
      alternatives: [{name: 'a', x: [5]}, {name: 'b', x: [6]}],
      seed: 1,
      initialSamples: 5,
      budget: 2,
      batchSize: 1,
      exploration: 1,
    });
  } catch {
    threw = true;
  }
  check('6.2 adaptive simopt rejects budget below initial allocation', threw);
}

console.log('\n[7] Hardening regressions');
{
  await checkThrows('7.1 empirical probabilities must sum to one', () => runRiskCapacity({
    cost: [10],
    price: [25],
    demand: {kind: 'empirical', empirical: [[{value: 20, prob: 0.4}, {value: 60, prob: 0.4}]]},
    numScenarios: 5,
    seed: 4,
    xMax: 80,
    step: 20,
    risk: {kind: 'expectation'},
  }), 'sum to 1');

  await checkThrows('7.2 fitted JSON demand requires family parameters', () => runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'risk-capacity',
    parameters: {
      cost: [10],
      price: [25],
      demand: {kind: 'fitted', fitted: [{family: 'normal', params: {mu: 40}}]},
      numScenarios: 5,
      seed: 8,
      xMax: 60,
      step: 20,
      risk: {kind: 'expectation'},
    },
    runtime: {animate: false, verbose: false},
  }, {verbose: false}), 'params.sigma');

  await checkThrows('7.3 oversized risk grid is rejected before enumeration', () => runRiskCapacity({
    cost: [1, 1, 1, 1],
    price: [2, 2, 2, 2],
    demand: {kind: 'uniform', ranges: [{low: 0, high: 1}, {low: 0, high: 1}, {low: 0, high: 1}, {low: 0, high: 1}]},
    numScenarios: 1,
    seed: 1,
    xMax: 100,
    step: 1,
    risk: {kind: 'expectation'},
  }), 'grid candidate count');

  await checkThrows('7.4 stochastic-lp rejects dimension mismatches', () => runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'stochastic-lp',
    parameters: {cost: [10, 12], price: [25, 28], ranges: [[50, 100]], numScenarios: 5, seed: 42},
    runtime: {animate: false, verbose: false},
  }, {verbose: false}), 'same length');

  const csvPath = 'out/statopt-risk-hardening.csv';
  const logPath = 'out/statopt-risk-hardening.jsonl';
  const observedSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'risk-capacity',
    parameters: {
      cost: [1, 1],
      price: [3, 4],
      demand: {kind: 'uniform', ranges: [{low: 0, high: 2}, {low: 0, high: 2}]},
      numScenarios: 4,
      seed: 3,
      xMax: 2,
      step: 2,
      risk: {kind: 'expectation'},
    },
    runtime: {animate: false, verbose: false, outputs: {csv: csvPath, log: logPath}},
  }, {verbose: false});
  const csv = fs.readFileSync(csvPath, 'utf8');
  const log = fs.readFileSync(logPath, 'utf8');
  check('7.5 CSV writer quotes vector fields', csv.includes('"[0,0]"'));
  check('7.6 observability log output is reported', observedSummary.outputs.some(o => o.kind === 'log' && o.path === logPath));
  check('7.7 observability log captures start and finish', log.includes('"kind":"risk-capacity-start"') && log.includes('"kind":"risk-capacity-finish"'));

  const htmlPath = 'out/statopt-custom-animation';
  const animSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'distribution-fit',
    parameters: {samples: [1, 2, 3, 4, 5], families: ['normal'], methods: ['mle']},
    runtime: {verbose: false, outputs: {html: htmlPath}},
  }, {verbose: false});
  check('7.8 custom animation HTML path is preserved', animSummary.outputs.some(o => o.kind === 'html' && o.path === htmlPath));
  check('7.9 custom animation frames path is distinct and suffixed', animSummary.outputs.some(o => o.kind === 'frames' && o.path === `${htmlPath}.frames.jsonl`));

  await checkThrows('7.10 distribution-fit requires at least two samples', () => runDistributionFit({
    samples: [1],
    families: ['normal'],
    methods: ['mle'],
  }), 'at least 2');

  await checkThrows('7.11 risk-capacity rejects unknown risk kind', () => runRiskCapacity({
    cost: [10],
    price: [25],
    demand: {kind: 'uniform', ranges: [{low: 20, high: 50}]},
    numScenarios: 5,
    seed: 3,
    xMax: 60,
    step: 20,
    risk: {kind: 'mystery'} as any,
  }), 'risk.kind');

  await checkThrows('7.12 SDDP rejects non-finite demand highs', () => runCapacityExpansionSDDP({
    horizon: 1,
    demand: [{low: 10, high: Infinity}],
    price: [20],
    expansionCost: [10],
    initialCapacity: 0,
    xMax: 30,
    step: 10,
    samplesPerStage: 5,
    seed: 1,
  }), 'demand[0].high');

  await checkThrows('7.13 adaptive simopt rejects duplicate names', () => runAdaptiveSimOpt({
    cost: [10],
    price: [25],
    demand: {kind: 'uniform', ranges: [{low: 20, high: 50}]},
    alternatives: [{name: 'same', x: [20]}, {name: 'same', x: [40]}],
    seed: 1,
    initialSamples: 1,
    budget: 2,
    batchSize: 1,
    exploration: 1,
  }), 'unique');

  await checkThrows('7.14 public fitted sampler validates required params', () => sampleFittedDistribution({
    family: 'normal',
    method: 'mle',
    params: {mu: 0},
    logLikelihood: 0,
    aic: 0,
    mean: 0,
    variance: 1,
    support: 'real',
  }, () => 0.5), 'params.sigma');
}

console.log(`\nstatistical-optimization-test summary: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
