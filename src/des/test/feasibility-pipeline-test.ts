'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/feasibility_pipeline_test.rs   (integration test crate)
// 1:1 file move. Tests the optimization feasibility-checker pipeline, so it is
// an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - async checkThrows()/await -> assert on Result::Err / #[should_panic].
//   - runFromJsonFile / fs + JSON -> serde_json + std::fs (or `tempfile`).
//   - variable-type union ('binary'|...) -> a Rust enum.
// =============================================================================

// =============================================================================
// Tests for the general optimization feasibility checker pipeline.
// =============================================================================

import * as fs from 'fs';
import {getModel, runFromJsonFile, runFromSpec} from '../general/des-registry';
import {
  StructuredOptimizationProblem,
  evaluateCandidate,
  runFeasibilityPipeline,
} from '../general/feasibility-pipeline';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
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
  check(label, threw && (contains === undefined || message.includes(contains)), threw ? `message=${JSON.stringify(message)}` : 'did not throw');
}

const knapsack: StructuredOptimizationProblem = {
  sense: 'max',
  variables: [
    {name: 'x0', type: 'binary'},
    {name: 'x1', type: 'binary'},
    {name: 'x2', type: 'binary'},
  ],
  objective: {coefficients: {x0: 60, x1: 100, x2: 120}},
  constraints: [{name: 'capacity', coefficients: {x0: 10, x1: 20, x2: 30}, sense: '<=', rhs: 50}],
  tolerance: 1e-8,
};

async function main(): Promise<void> {
console.log('\n[1] Direct feasibility evaluation');
{
  const ok = evaluateCandidate(knapsack, {values: {x0: 1, x1: 1, x2: 0}});
  check('1.1 feasible candidate is recognized', ok.feasible && ok.objectiveValue === 160, `objective=${ok.objectiveValue}`);

  const bad = evaluateCandidate(knapsack, {values: {x0: 1, x1: 1, x2: 1.25}});
  check('1.2 domain and constraint violations are reported',
    !bad.feasible && bad.domainViolations.length >= 1 && bad.constraintViolations.length === 1,
    `domain=${bad.domainViolations.length} constraints=${bad.constraintViolations.length}`);

  await checkThrows('1.3 unknown objective coefficient is rejected', () => evaluateCandidate({
    ...knapsack,
    objective: {coefficients: {x0: 1, missing: 2}},
  }, {values: {x0: 1, x1: 0, x2: 0}}), 'reference a declared variable');
}

console.log('\n[2] Pipeline checks and improves candidate solutions');
{
  const r = runFeasibilityPipeline({
    problem: knapsack,
    candidate: {id: 'user-best', values: {x0: 1, x1: 1, x2: 0}},
    improvement: {enabled: true, maxIterations: 80, seed: 4, integerStep: 1},
  });
  check('2.1 pipeline improves binary incumbent', r.status === 'improved' && r.best.objectiveValue === 220, `status=${r.status} objective=${r.best.objectiveValue}`);
  check('2.2 best candidate remains feasible', r.best.feasible && r.best.totalViolation === 0);
  check('2.3 network exposes all pipeline stations', r.network.stationaryEntities.length === 7 && r.network.edges.length === 7);
  check('2.4 validators pass', r.validation.every(c => c.passed), r.validation.filter(c => !c.passed).map(c => c.name).join(', '));

  const checkOnly = runFeasibilityPipeline({
    problem: knapsack,
    candidate: {values: {x0: 1, x1: 1, x2: 1}},
    improvement: {enabled: false},
  });
  check('2.5 checker-only mode reports infeasible incumbent', checkOnly.status === 'infeasible' && !checkOnly.initial.feasible, `status=${checkOnly.status}`);
}

console.log('\n[3] Continuous problem improvement and time limits');
{
  const production: StructuredOptimizationProblem = {
    sense: 'min',
    variables: [
      {name: 'regular', type: 'continuous', lb: 0, ub: 100, step: 5},
      {name: 'overtime', type: 'continuous', lb: 0, ub: 50, step: 5},
    ],
    objective: {coefficients: {regular: 4, overtime: 7}},
    constraints: [
      {name: 'demand', coefficients: {regular: 1, overtime: 1}, sense: '>=', rhs: 80},
      {name: 'regular-cap', coefficients: {regular: 1}, sense: '<=', rhs: 60},
    ],
  };
  const prod = runFeasibilityPipeline({
    problem: production,
    candidate: {values: {regular: 50, overtime: 40}},
    improvement: {enabled: true, maxIterations: 60, seed: 9, continuousStep: 5},
  });
  check('3.1 continuous incumbent improves cost', prod.status === 'improved' && prod.best.objectiveValue < prod.initial.objectiveValue, `initial=${prod.initial.objectiveValue} best=${prod.best.objectiveValue}`);
  check('3.2 continuous best remains feasible', prod.best.feasible);

  const timed = runFeasibilityPipeline({
    problem: production,
    candidate: {values: {regular: 50, overtime: 40}},
    improvement: {enabled: true, maxIterations: 1000, seed: 1},
    timeLimitMs: 0,
  });
  check('3.3 zero wall-clock budget stops through checker', timed.status === 'time-limit' && timed.stopSignals.length >= 1, `status=${timed.status} signals=${timed.stopSignals.length}`);
}

console.log('\n[4] Registry, JSON input, logs, and animation');
{
  check('4.1 registry has feasibility-pipeline', getModel('feasibility-pipeline').id === 'feasibility-pipeline');
  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'feasibility-pipeline',
    parameters: {
      problem: knapsack,
      candidate: {id: 'json-user', values: {x0: 1, x1: 1, x2: 0}},
      improvement: {enabled: true, maxIterations: 40, seed: 4},
    },
    runtime: {verbose: false, outputs: {csv: 'out/feasibility-pipeline-test.csv', html: 'out/feasibility-pipeline-test.html', log: 'out/feasibility-pipeline-test.jsonl'}},
  }, {verbose: false});
  const result = summary.result as any;
  check('4.2 JSON run writes default frames output', summary.outputs.some(o => o.kind === 'frames' && o.path === 'out/feasibility-pipeline-test.frames.jsonl'));
  check('4.3 JSON run writes observability log', fs.readFileSync('out/feasibility-pipeline-test.jsonl', 'utf8').includes('"kind":"feasibility-pipeline-finish"'));
  check('4.4 JSON best solution improves candidate', result.best.objectiveValue === 220, `objective=${result.best.objectiveValue}`);

  const fromFile = await runFromJsonFile('examples/feasibility-pipeline-knapsack.json', {verbose: false});
  check('4.5 checked-in example runs through registry', fromFile.modelId === 'feasibility-pipeline' && fromFile.outputs.some(o => o.kind === 'frames'));
}

console.log(`\nfeasibility-pipeline-test summary: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
