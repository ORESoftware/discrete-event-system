// RUST MIGRATION: Port file-for-file to `tests/internal_solver_network_test.rs` for GA/SA/knapsack/shortest-path/TSP solver-network coverage.
// Test-port notes: translate solver scenarios into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, approximate-float helpers, and deterministic fixtures.

'use strict';

// =============================================================================
// Tests for internal solver networks: GA, SA, knapsack, shortest path, TSP, and
// the wall-clock checker station.
// =============================================================================

import * as fs from 'fs';
import {getModel, runFromJsonFile, runFromSpec} from '../general/des-registry';
import {buildPentagonTSP, heldKarpExact} from '../general/genetic-tsp';
import {runInternalSolverNetwork} from '../general/internal-solver-network';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

async function main(): Promise<void> {
console.log('\n[1] Shortest path as a DES solver network');
{
  const r = runInternalSolverNetwork({
    kind: 'shortest-path',
    shortestPath: {algorithm: 'dijkstra', source: 0, builtin: 'small-chain'},
  });
  const state = r.best.bestState as any;
  check('1.1 run completes', r.status === 'complete', `status=${r.status}`);
  check('1.2 distance to sink is 6', close(state.distance[4], 6), `d4=${state.distance[4]}`);
  check('1.3 network exposes solver/checker/sink stationary entities',
    r.network.stationaryEntities.some(n => n.id === 'wall-clock-checker') &&
    r.network.stationaryEntities.some(n => n.id === 'solution-sink') &&
    r.network.edges.some(e => e.movingEntity === 'SolverSolutionToken'));
  check('1.4 validators pass', r.validation.every(c => c.passed), r.validation.filter(c => !c.passed).map(c => c.name).join(', '));
}

console.log('\n[2] Knapsack exact DP and simulated annealing');
{
  const dp = runInternalSolverNetwork({
    kind: 'knapsack-dp',
    knapsack: {
      values: [60, 100, 120],
      weights: [10, 20, 30],
      capacity: 50,
    },
  });
  const best = dp.best.bestState as any;
  check('2.1 DP finds the exact value 220', best.value === 220, `value=${best.value}`);
  check('2.2 DP solution is feasible', best.weight <= best.capacity && dp.best.feasible, `weight=${best.weight}`);
  check('2.3 DP emits incumbent tokens', dp.trace.length >= 3 && dp.trace.some(t => t.done), `trace=${dp.trace.length}`);

  const sa = runInternalSolverNetwork({
    kind: 'knapsack-sa',
    knapsack: {
      values: [60, 100, 120],
      weights: [10, 20, 30],
      capacity: 50,
      maxIterations: 80,
      seed: 4,
      cooling: {kind: 'geometric', T0: 30, alpha: 0.97, Tmin: 1e-6},
    },
  });
  const saBest = sa.best.bestState as any;
  check('2.4 SA returns a feasible knapsack incumbent', saBest.weight <= saBest.capacity && sa.best.feasible, `weight=${saBest.weight}`);
  check('2.5 SA keeps at least the greedy incumbent quality', saBest.value >= 160, `value=${saBest.value}`);
}

console.log('\n[3] Traveling salesman internal solvers');
{
  const ga = runInternalSolverNetwork({
    kind: 'tsp-ga',
    tsp: {
      builtin: 'pentagon',
      n: 7,
      seed: 7,
      ga: {popSize: 28, numGenerations: 25, seed: 9, init: 'nearest-neighbor'},
    },
  });
  const gaBest = ga.best.bestState as any;
  check('3.1 GA completes with a feasible tour', ga.status === 'complete' && ga.best.feasible, `status=${ga.status}`);
  check('3.2 GA tour length is finite', Number.isFinite(gaBest.length), `length=${gaBest.length}`);

  const sa = runInternalSolverNetwork({
    kind: 'tsp-sa',
    tsp: {
      builtin: 'pentagon',
      n: 7,
      seed: 5,
      sa: {maxIterations: 80, seed: 5, cooling: {kind: 'geometric', T0: 100, alpha: 0.97, Tmin: 1e-6}, init: 'nearest-neighbor'},
    },
  });
  const saBest = sa.best.bestState as any;
  check('3.3 TSP SA completes with a feasible tour', sa.status === 'complete' && sa.best.feasible, `status=${sa.status}`);
  check('3.4 TSP SA tour length is finite', Number.isFinite(saBest.length), `length=${saBest.length}`);

  const inst = buildPentagonTSP(5);
  const exact = heldKarpExact(inst);
  const hk = runInternalSolverNetwork({
    kind: 'tsp-held-karp',
    tsp: {builtin: 'pentagon', n: 5},
  });
  const hkBest = hk.best.bestState as any;
  check('3.5 Held-Karp station matches exact length', close(hkBest.length, exact.length, 1e-10), `best=${hkBest.length} exact=${exact.length}`);
}

console.log('\n[4] Wall-clock checker station');
{
  const timed = runInternalSolverNetwork({
    kind: 'tsp-sa',
    timeLimitMs: 0,
    checkEveryTicks: 1,
    tsp: {
      builtin: 'pentagon',
      n: 6,
      seed: 6,
      sa: {maxIterations: 1000, seed: 6, cooling: {kind: 'geometric', T0: 100, alpha: 0.999, Tmin: 1e-9}},
    },
  });
  check('4.1 zero budget stops on the first checker pass', timed.status === 'time-limit', `status=${timed.status}`);
  check('4.2 stop signal is emitted and observed by the sink', timed.stopSignals.length >= 1 && timed.wallClock.expired, `signals=${timed.stopSignals.length}`);
}

console.log('\n[5] Registry, JSON input, observability, and animation');
{
  check('5.1 registry has internal-solver-network', getModel('internal-solver-network').id === 'internal-solver-network');

  const summary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'internal-solver-network',
    parameters: {
      kind: 'knapsack-dp',
      knapsack: {
        values: [60, 100, 120],
        weights: [10, 20, 30],
        capacity: 50,
      },
    },
    runtime: {
      verbose: false,
      outputs: {
        csv: 'out/internal-solver-network-test.csv',
        html: 'out/internal-solver-network-test.html',
        log: 'out/internal-solver-network-test.jsonl',
      },
    },
  }, {verbose: false});
  const result = summary.result as any;
  check('5.2 JSON run writes default frames when animation is enabled',
    summary.outputs.some(o => o.kind === 'frames' && o.path === 'out/internal-solver-network-test.frames.jsonl'));
  check('5.3 JSON run writes observability log',
    fs.readFileSync('out/internal-solver-network-test.jsonl', 'utf8').includes('"kind":"internal-solver-finish"'));
  check('5.4 JSON result preserves best value', result.best.bestState.value === 220, `value=${result.best.bestState.value}`);

  const fromFile = await runFromJsonFile('examples/internal-solver-knapsack-dp.json', {verbose: false});
  check('5.5 checked-in JSON example runs through registry', fromFile.modelId === 'internal-solver-network' && fromFile.outputs.some(o => o.kind === 'frames'));
}

console.log(`\ninternal-solver-network-test summary: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
