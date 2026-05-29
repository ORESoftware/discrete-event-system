'use strict';

// =============================================================================
// Tests for advanced optimization and decision/control station-graph models.
// =============================================================================

import {getModel, runFromSpec} from '../general/des-registry';
import {
  ParetoArchiveStation,
  ParetoCandidateToken,
} from '../general/des-base/advanced-optimization';
import {runIterativeDES} from '../general/des-base';
import {
  paretoFrontIsNondominated,
  runAntColonyTSP,
  runMapColoringCSP,
  runMaxSATLocalSearch,
  runParetoPortfolio,
  runParticleSwarm,
  runSDPMaxCutRelaxation,
} from '../general/advanced-optimization-models';
import {
  runHInfinityRobustControl,
  runPursuitEvasionGame,
} from '../general/advanced-control-models';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function nearDiagOne(matrix: readonly number[][], tol = 1e-9): boolean {
  return matrix.every((row, i) => Math.abs(row[i] - 1) <= tol);
}

async function main(): Promise<void> {
  console.log('\n-- particle-swarm --');
  {
    const r = runParticleSwarm({objective: 'sphere', dimension: 3, particles: 32, iterations: 120, seed: 11});
    check('PSO drives sphere objective near zero', r.bestValue < 1e-8, `best=${r.bestValue}`);
    check('PSO records requested iterations', r.iterations === 120, `iters=${r.iterations}`);
    check('PSO uses shared particle movables', r.topology.movables.includes('NumericSwarmParticle'));
    check('PSO starts from source and ends at result sink',
      r.topology.stations[0] === 'particle-swarm-source' && r.topology.stations.includes('particle-swarm-result-sink'));
    check('PSO rejects degenerate bounds before running',
      throws(() => runParticleSwarm({lower: 1, upper: 1})));
    check('PSO rejects invalid dimensions through framework preconditions',
      throws(() => runParticleSwarm({dimension: Number.NaN})));
  }

  console.log('\n-- ant-colony-tsp --');
  {
    const r = runAntColonyTSP({iterations: 80, seed: 5});
    check('ACO returns a closed tour', r.bestTour.length >= 3 && r.bestTour[0] === r.bestTour[r.bestTour.length - 1],
      `tour=${r.bestTour.join('->')}`);
    check('ACO best length is finite and positive', Number.isFinite(r.bestLength) && r.bestLength > 0, `len=${r.bestLength}`);
    check('ACO uses graph walk tokens', r.topology.movables.includes('GraphWalkToken'));
    check('ACO starts from source and ends at result sink',
      r.topology.stations[0] === 'ant-colony-tsp-source' && r.topology.stations.includes('ant-colony-tsp-result-sink'));
    check('ACO rejects duplicate coordinates before graph search',
      throws(() => runAntColonyTSP({points: [{x: 0, y: 0}, {x: 0, y: 0}]})));
  }

  console.log('\n-- map-coloring-csp --');
  {
    const r = runMapColoringCSP({});
    check('CSP finds a satisfying map coloring', r.satisfied, JSON.stringify(r.assignment));
    check('CSP processes at least one tree node', r.nodesProcessed > 0, `nodes=${r.nodesProcessed}`);
    check('CSP uses assignment movables', r.topology.movables.includes('ConstraintAssignmentToken'));
    check('CSP starts from source and ends at result sink',
      r.topology.stations[0] === 'map-coloring-csp-source' && r.topology.stations.includes('map-coloring-csp-result-sink'));
    check('CSP rejects constraints that reference unknown variables',
      throws(() => runMapColoringCSP({variables: ['A'], colors: ['red'], edges: [['A', 'B']]})));
  }

  console.log('\n-- max-sat-local-search --');
  {
    const r = runMaxSATLocalSearch({});
    check('MAX-SAT satisfies the default formula', r.allSatisfied, `${r.satisfiedClauses}/${r.totalClauses}`);
    check('MAX-SAT assignment has boolean variables', r.assignment.every(v => typeof v === 'boolean'));
    check('MAX-SAT uses shared single-state style movables', r.topology.movables.some(v => v.includes('OptimizationCandidateToken')));
    check('MAX-SAT starts from source and ends at result sink',
      r.topology.stations[0] === 'max-sat-local-search-source' && r.topology.stations.includes('max-sat-local-search-result-sink'));
  }

  console.log('\n-- sdp-maxcut-relaxation --');
  {
    const r = runSDPMaxCutRelaxation({});
    check('SDP relaxation value upper-bounds rounded cut', r.sdpValue + 1e-9 >= r.roundedCutValue,
      `sdp=${r.sdpValue}, cut=${r.roundedCutValue}`);
    check('SDP Gram matrix has unit diagonal', nearDiagOne(r.gramMatrix));
    check('SDP uses unit-vector relaxation station', r.topology.stations.includes('sdp-maxcut-relaxation-station'));
    check('SDP starts from source and ends at result sink',
      r.topology.stations[0] === 'sdp-maxcut-relaxation-source' && r.topology.stations.includes('sdp-maxcut-relaxation-result-sink'));
    const fallback = runSDPMaxCutRelaxation({edges: []});
    check('SDP empty edge input falls back to reusable default graph', fallback.roundedCutValue > 0);
  }

  console.log('\n-- pareto-portfolio --');
  {
    const r = runParetoPortfolio({});
    check('Pareto archive keeps multiple tradeoff points', r.paretoFront.length >= 2, `front=${r.paretoFront.length}`);
    check('Pareto front is nondominated', paretoFrontIsNondominated(r.paretoFront));
    check('Pareto archive processed all candidates', r.candidateCount >= 200, `candidates=${r.candidateCount}`);
    check('Pareto portfolio starts from candidate source into archive sink',
      r.topology.stations[0] === 'pareto-portfolio-source' && r.topology.stations.includes('pareto-portfolio-archive'));
    const archive = new ParetoArchiveStation<number>('pareto-reuse-test');
    runIterativeDES([archive], {shuffle: false, maxTicks: 2, runValidators: false});
    archive.enqueue(new ParetoCandidateToken(1, [1, -1]));
    runIterativeDES([archive], {shuffle: false, maxTicks: 2, runValidators: false});
    check('Pareto archive can be reused after going idle', archive.getProcessedCount() === 1);
    const dupArchive = new ParetoArchiveStation<number>('pareto-duplicate-test', [
      new ParetoCandidateToken(1, [1, -1]),
      new ParetoCandidateToken(2, [1, -1]),
    ]);
    runIterativeDES([dupArchive], {shuffle: false, maxTicks: 4, runValidators: false});
    check('Pareto archive de-duplicates equivalent objective movables', dupArchive.getArchive().length === 1);
  }

  console.log('\n-- hinfinity-robust-control --');
  {
    const r = runHInfinityRobustControl({});
    check('H-infinity graph keeps disturbance-to-state gain below gamma', r.boundedByGamma,
      `gain=${r.l2GainEstimate}, gamma=${r.gamma}`);
    check('H-infinity controller keeps state bounded', Math.abs(r.finalState) < 0.5,
      `final=${r.finalState}`);
    check('H-infinity graph has controller/adversary/plant stations', r.topology.stations.length === 3);
  }

  console.log('\n-- pursuit-evasion-game --');
  {
    const r = runPursuitEvasionGame({});
    check('pursuit/evasion differential game captures the evader', r.captureTick !== null,
      `capture=${r.captureTick}`);
    check('pursuit/evasion distance decreases', r.finalDistance < r.distanceHistory[0],
      `d0=${r.distanceHistory[0]}, df=${r.finalDistance}`);
    check('pursuit/evasion uses command movables', r.topology.movables.includes('ControlMoveToken') && r.topology.movables.includes('DisturbanceMoveToken'));
  }

  console.log('\n-- registry coverage --');
  {
    const ids = [
      'simulated-annealing',
      'internal-solver-network',
      'particle-swarm',
      'ant-colony-tsp',
      'map-coloring-csp',
      'max-sat-local-search',
      'sdp-maxcut-relaxation',
      'stochastic-lp',
      'pareto-portfolio',
      'policy-gradient-corridor',
      'actor-critic-grid',
      'inventory-dp',
      'tiger-pomdp',
      'grid-localization-pomdp',
      'pontryagin-bang-bang',
      'mpc-double-integrator',
      'double-integrator-lqr',
      'kalman-filter',
      'hinfinity-robust-control',
      'mrac',
      'iterative-learning-control',
      'pursuit-evasion-game',
      'stochastic-flow-mdp',
    ];
    for (const id of ids) check(`registry has ${id}`, getModel(id).id === id);

    const psoSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'particle-swarm',
      parameters: {objective: 'sphere', iterations: 30, particles: 16, seed: 3},
      runtime: {animate: false, verbose: false},
    }, {verbose: false});
    check('runFromSpec executes particle-swarm', psoSummary.modelId === 'particle-swarm');

    const robustSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'hinfinity-robust-control',
      parameters: {numSteps: 50},
      runtime: {animate: false, verbose: false},
    }, {verbose: false});
    check('runFromSpec executes H-infinity robust control', robustSummary.modelId === 'hinfinity-robust-control');
  }

  console.log('\n========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`advanced-optimization-control-test: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
