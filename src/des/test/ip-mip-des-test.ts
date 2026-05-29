'use strict';

// =============================================================================
// test/ip-mip-des-test.ts -- explicit station-graph IP/MIP solver tests.
// =============================================================================

import {
  buildBinaryKnapsackIP,
  buildIPMIPSolverTechniquePlan,
  buildSmallMixedIP,
  IPMIPProblem,
  solveIPMIPWithDES,
} from '../general/ip-mip-des';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS  ${label}${detail ? ' -- ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL  ${label}${detail ? ' -- ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
function feasible(p: IPMIPProblem, x: number[]): boolean {
  if (x.length !== p.c.length) return false;
  for (let j = 0; j < x.length; j++) {
    if (x[j] < -1e-7) return false;
    if (p.ub && Number.isFinite(p.ub[j]) && x[j] > p.ub[j] + 1e-7) return false;
    if (p.integerVars[j] && Math.abs(x[j] - Math.round(x[j])) > 1e-7) return false;
  }
  for (let i = 0; i < p.A.length; i++) {
    let lhs = 0;
    for (let j = 0; j < x.length; j++) lhs += p.A[i][j] * x[j];
    if (lhs > p.b[i] + 1e-7) return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 -- Station-graph knapsack with incremental LP backend');
// -----------------------------------------------------------------------------
{
  const p = buildBinaryKnapsackIP([10, 40, 30, 50], [5, 4, 6, 3], 10);
  const r = solveIPMIPWithDES(p, {lpAlgorithm: 'incremental-primal-dual', maxCutRounds: 1});
  check('status optimal', r.status === 'optimal', r.status);
  check('z = 90', close(r.z, 90), `z=${r.z}`);
  check('x feasible and integer', feasible(p, r.x), `x=[${r.x.join(',')}]`);
  check('station topology exposes LP relaxation block',
    r.topology.some(n => n.id === 'ip-lp-relaxation' && n.role.includes('LP solver')),
    r.topology.map(n => n.id).join(','));
  check('solver is explicitly single-threaded for now', r.executionMode === 'single-threaded', r.executionMode);
  check('topology exposes composite solver parent',
    r.topology.some(n => n.id === r.compositeStationId && n.role.includes('composite')),
    r.topology.map(n => n.id).join(','));
  check('substations are nested under the composite solver',
    r.topology.some(n => n.id === 'ip-node-decision' && n.parentId === r.compositeStationId),
    JSON.stringify(r.topology));
  check('movable solver tokens carry state and lineage',
    r.tokenStats.stateful > 0 && r.tokenStats.maxGeneration > 0 && r.trace.some(e => !!e.nodeTokenId),
    JSON.stringify(r.tokenStats));
  check('shared token registry counts solver payload kinds',
    (r.tokenStats.byKind['ip-node'] ?? 0) > 0
      && (r.tokenStats.byKind['ip-relaxation'] ?? 0) > 0
      && (r.tokenStats.byKind['ip-candidate'] ?? 0) > 0,
    JSON.stringify(r.tokenStats.byKind));
  check('shared token registry keeps state transition totals',
    r.tokenStats.stateTransitions >= r.tokenStats.stateful,
    JSON.stringify(r.tokenStats));
  check('trace preserves child-token lineage roots',
    r.trace.some(e => e.lineageRoot === 'ip-node-0' && (e.tokenGeneration ?? 0) > 0),
    JSON.stringify(r.trace.map(e => ({root: e.lineageRoot, gen: e.tokenGeneration}))));
  check('completion movables can be stateless',
    r.tokenStats.stateless > 0,
    JSON.stringify(r.tokenStats));
  check('rounding/repair tried candidates', r.candidatesTried > 0, `candidates=${r.candidatesTried}`);
  check('default solve is in-house only', r.inHouseOnly && !r.usesExternalSolvers, JSON.stringify(r.lpAlgorithmUsage));
  check('performance telemetry is populated',
    r.performance.elapsedMs >= 0 && r.performance.nodesPerSecond >= 0 && r.performance.tokensCreated === r.tokenStats.created,
    JSON.stringify(r.performance));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 -- Selectable LP algorithms agree');
// -----------------------------------------------------------------------------
{
  const p = buildBinaryKnapsackIP([10, 40, 30, 50], [5, 4, 6, 3], 10);
  for (const alg of ['internal-simplex', 'des-simplex-dantzig', 'des-simplex-bland'] as const) {
    const r = solveIPMIPWithDES(p, {lpAlgorithm: alg, maxCutRounds: 1});
    check(`${alg}: status optimal`, r.status === 'optimal', r.status);
    check(`${alg}: z = 90`, close(r.z, 90), `z=${r.z}`);
    check(`${alg}: backend recorded`, r.lpAlgorithm === alg, r.lpAlgorithm);
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 -- Mixed integer/continuous program');
// -----------------------------------------------------------------------------
{
  const p = buildSmallMixedIP();
  const r = solveIPMIPWithDES(p, {lpAlgorithm: 'incremental-primal-dual', maxCutRounds: 0});
  check('status optimal', r.status === 'optimal', r.status);
  check('z = 13', close(r.z, 13), `z=${r.z}`);
  check('integer variables integral, continuous allowed', feasible(p, r.x), `x=[${r.x.join(',')}]`);
  check('continuous variable reaches upper bound', close(r.x[2], 10), `x2=${r.x[2]}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 -- Cover cut strengthens binary relaxation');
// -----------------------------------------------------------------------------
{
  const p = buildBinaryKnapsackIP([10, 10, 10], [2, 2, 2], 3);
  const rNoCuts = solveIPMIPWithDES(p, {lpAlgorithm: 'incremental-primal-dual', maxCutRounds: 0, maxNodes: 50});
  const rCuts = solveIPMIPWithDES(p, {lpAlgorithm: 'incremental-primal-dual', maxCutRounds: 1, maxNodes: 50});
  check('no-cuts solve optimal', rNoCuts.status === 'optimal', rNoCuts.status);
  check('cuts solve optimal', rCuts.status === 'optimal', rCuts.status);
  check('optimal z = 10', close(rCuts.z, 10), `z=${rCuts.z}`);
  check('cover cut station adds at least one cut', rCuts.cutsAdded > 0, `cuts=${rCuts.cutsAdded}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 -- Auto technique selection');
// -----------------------------------------------------------------------------
{
  const p = buildBinaryKnapsackIP([10, 40, 30, 50], [5, 4, 6, 3], 10);
  const r = solveIPMIPWithDES(p, {lpAlgorithm: 'auto', maxCutRounds: 1});
  check('auto solve remains optimal', r.status === 'optimal', r.status);
  check('auto request is recorded', r.lpAlgorithm === 'auto', r.lpAlgorithm);
  check('auto picked in-engine incremental LP for small branch-cut model',
    (r.lpAlgorithmUsage['incremental-primal-dual'] ?? 0) > 0,
    JSON.stringify(r.lpAlgorithmUsage));
  check('technique plan records binary features',
    r.techniquePlan.features.allBinary && r.techniquePlan.rootLPAlgorithm === 'incremental-primal-dual',
    JSON.stringify(r.techniquePlan.features));
  check('auto default refuses external solvers', r.inHouseOnly && !r.techniquePlan.externalSolversAllowed,
    JSON.stringify(r.techniquePlan));
}

{
  const dense: IPMIPProblem = {
    sense: 'max',
    c: Array.from({length: 80}, (_, i) => 1 + (i % 5)),
    A: Array.from({length: 40}, (_, r) => Array.from({length: 80}, (_, c) => ((r + c) % 7) + 1)),
    b: Array.from({length: 40}, () => 500),
    integerVars: Array.from({length: 80}, () => false),
    ub: Array.from({length: 80}, () => 10),
  };
  const inHousePlan = buildIPMIPSolverTechniquePlan(dense, 'auto');
  check('auto plan keeps large dense root relaxation in-house by default',
    !inHousePlan.externalCandidate && inHousePlan.rootLPAlgorithm === 'incremental-primal-dual',
    JSON.stringify(inHousePlan));
  const externalPlan = buildIPMIPSolverTechniquePlan(dense, 'auto', true);
  check('auto plan can opt in to external LP for large dense root relaxation',
    externalPlan.externalCandidate && externalPlan.rootLPAlgorithm === 'external-highs-ipm',
    JSON.stringify(externalPlan));
  let threwExternal = false;
  try {
    buildIPMIPSolverTechniquePlan(dense, 'external-highs');
  } catch (e) {
    threwExternal = true;
  }
  check('explicit external backend requires opt-in',
    threwExternal,
    JSON.stringify(inHousePlan));
}

{
  const separable: IPMIPProblem = {
    sense: 'max',
    c: [5, 4, 7, 6],
    A: [[1, 1, 0, 0], [0, 0, 1, 1]],
    b: [1, 1],
    integerVars: [true, true, true, true],
    ub: [1, 1, 1, 1],
  };
  const plan = buildIPMIPSolverTechniquePlan(separable, 'auto');
  check('auto plan detects separable decomposition candidate',
    plan.decompositionCandidate && plan.features.constraintVariableComponents === 2,
    JSON.stringify(plan));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 -- Preconditions and limits');
// -----------------------------------------------------------------------------
{
  let threw = false;
  try {
    solveIPMIPWithDES({sense: 'max', c: [1, 2], A: [[1]], b: [1], integerVars: [true, true]});
  } catch (e) {
    threw = true;
  }
  check('rejects malformed A row', threw);

  const p = buildBinaryKnapsackIP(
    Array.from({length: 12}, (_, i) => i + 1),
    Array.from({length: 12}, (_, i) => i + 1),
    20,
  );
  const r = solveIPMIPWithDES(p, {maxNodes: 1, maxCutRounds: 0});
  check('small maxNodes returns optimal or maxnodes', r.status === 'optimal' || r.status === 'maxnodes', r.status);
  check('node cap respected', r.nodesExplored <= 1, `nodes=${r.nodesExplored}`);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
