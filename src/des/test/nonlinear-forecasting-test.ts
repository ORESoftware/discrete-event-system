// RUST MIGRATION: Port file-for-file to `tests/nonlinear_forecasting_test.rs` as integration coverage for nonlinear MDP/POMDP forecast specs and registry execution.
// Test-port notes: translate forecast cases into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, and approximate-float helpers; keep scenario fixtures deterministic.

'use strict';

// Tests for nonlinear MDP/POMDP forecasting.

import {getModel, runFromSpec} from '../general/des-registry';
import {NonlinearMDPPOMDPForecastResult, runNonlinearMDPPOMDPForecast} from '../general/nonlinear-forecasting-model';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function hasStation(result: NonlinearMDPPOMDPForecastResult, id: string): boolean {
  return result.topology.stations.includes(id);
}

async function main(): Promise<void> {
  console.log('\n-- nonlinear-mdp-pomdp-forecast --');
  const r = runNonlinearMDPPOMDPForecast({});
  check('forecast model uses explicit data source station', hasStation(r, 'nonlinear-forecast-data-source'));
  check('forecast model uses explicit POMDP station', hasStation(r, 'pomdp-latent-variable-station'));
  check('forecast model uses explicit MDP station', hasStation(r, 'mdp-variable-discovery-station'));
  check('forecast model uses equation tuning station', hasStation(r, 'nonlinear-equation-tuning-station'));
  check('forecast model uses projection sink path', hasStation(r, 'forecast-projection-station') && hasStation(r, 'forecast-result-sink'));
  check('forecast model exposes all expected movables',
    ['ForecastDataToken', 'LatentBeliefTraceToken', 'DiscoveredVariablesToken', 'FineTunedEquationToken', 'ForecastProjectionToken']
      .every(m => r.topology.movables.includes(m)));
  check('POMDP belief trace covers training horizon', r.pomdp.points.length === 42, `points=${r.pomdp.points.length}`);
  check('MDP enumerates feature discovery state space', r.mdp.states >= 512 && r.mdp.actions >= 8, `states=${r.mdp.states} actions=${r.mdp.actions}`);
  check('MDP selects at least one latent POMDP variable', r.discoveredVariables.some(v => v.source === 'pomdp'), r.selectedVariables.join(','));
  check('validation MSE improves over baseline', r.metrics.validationMse < r.metrics.baselineValidationMse, `${r.metrics.validationMse} < ${r.metrics.baselineValidationMse}`);
  check('forecast MSE improves over naive baseline', r.metrics.forecastMse < r.metrics.baselineForecastMse, `${r.metrics.forecastMse} < ${r.metrics.baselineForecastMse}`);
  check('projection length matches default horizon', r.projection.length === 8);
  check('forecast values are finite', r.projection.every(p => Number.isFinite(p.forecast) && Number.isFinite(p.lower) && Number.isFinite(p.upper)));
  check('fine-tuning trace improves MSE',
    r.equation.trace[r.equation.trace.length - 1].mse < r.equation.trace[0].mse,
    `${r.equation.trace[0].mse} -> ${r.equation.trace[r.equation.trace.length - 1].mse}`);

  console.log('\n-- registry and JSON smoke --');
  check('registry has nonlinear-mdp-pomdp-forecast', getModel('nonlinear-mdp-pomdp-forecast').id === 'nonlinear-mdp-pomdp-forecast');
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'nonlinear-mdp-pomdp-forecast',
    parameters: {trainingPeriods: 42, forecastHorizon: 8},
    runtime: {animate: false, verbose: false},
  }, {verbose: false});
  check('runFromSpec executes nonlinear forecast model', summary.modelId === 'nonlinear-mdp-pomdp-forecast');

  console.log('\n========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`nonlinear-forecasting-test: ${passed}/${checks.length} checks passed.`);
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
