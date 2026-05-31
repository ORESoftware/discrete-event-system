// RUST MIGRATION: Target module `src/des/general/adapters/advanced_optimization_control_adapter.rs`.
// RUST MIGRATION: Convert the advanced optimization/control registrations into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map parameter schemas, optimizer configs, and run summaries to `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Replace validation throws or rejected params with `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// JSON adapters for advanced optimization and decision/control station graphs.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  AntColonyTSPParams,
  AntColonyTSPResult,
  MapColoringCSPParams,
  MapColoringCSPResult,
  MaxSATParams,
  MaxSATResult,
  ParetoPortfolioParams,
  ParetoPortfolioResult,
  ParticleSwarmParams,
  ParticleSwarmResult,
  SDPMaxCutParams,
  SDPMaxCutResult,
  WeightedEdge,
  runAntColonyTSP,
  runMapColoringCSP,
  runMaxSATLocalSearch,
  runParetoPortfolio,
  runParticleSwarm,
  runSDPMaxCutRelaxation,
} from '../advanced-optimization-models';
import {
  HInfinityRobustControlParams,
  HInfinityRobustControlResult,
  PursuitEvasionGameParams,
  PursuitEvasionGameResult,
  runHInfinityRobustControl,
  runPursuitEvasionGame,
} from '../advanced-control-models';
import {csvRow, writeCsvLines} from './adapter-utils';

const numberVectorSchema: ParamSchema = {kind: 'array', items: {kind: 'number'}, minLength: 1};

const pointSchema: ParamSchema = {
  kind: 'object',
  fields: {x: {kind: 'number'}, y: {kind: 'number'}},
  required: ['x', 'y'],
};

const stringPairSchema: ParamSchema = {
  kind: 'array',
  items: {kind: 'string'},
  minLength: 2,
  maxLength: 2,
};

const weightedEdgeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    i: {kind: 'number', integer: true, min: 0},
    j: {kind: 'number', integer: true, min: 0},
    weight: {kind: 'number', min: 1e-12},
  },
  required: ['i', 'j', 'weight'],
};

const portfolioAssetSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    expectedReturn: {kind: 'number'},
    risk: {kind: 'number', min: 0},
  },
  required: ['name', 'expectedReturn', 'risk'],
};

registerModel<ParticleSwarmParams, ParticleSwarmResult>({
  id: 'particle-swarm',
  description: 'Particle Swarm Optimization using a shared numeric-swarm station and particle movables.',
  schema: {
    kind: 'object',
    fields: {
      objective: {kind: 'string', enum: ['sphere', 'rastrigin', 'rosenbrock'], default: 'sphere'},
      dimension: {kind: 'number', integer: true, min: 1, default: 3},
      particles: {kind: 'number', integer: true, min: 1, default: 32},
      iterations: {kind: 'number', integer: true, min: 1, default: 120},
      lower: {kind: 'number', default: -5},
      upper: {kind: 'number', default: 5},
      inertia: {kind: 'number', min: 0, default: 0.68},
      cognitive: {kind: 'number', min: 0, default: 1.45},
      social: {kind: 'number', min: 0, default: 1.45},
      seed: {kind: 'number', integer: true, default: 11},
    },
    required: [],
  },
  run(params) { return runParticleSwarm(params); },
  summarize(result) {
    return [
      'PARTICLE SWARM OPTIMIZATION (DES)',
      '---------------------------------',
      `  Best value:     ${result.bestValue.toExponential(4)}`,
      `  Best position:  [${result.bestPosition.map(v => v.toFixed(4)).join(', ')}]`,
      `  Iterations:     ${result.iterations}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iteration', 'best_value', 'mean_value', 'worst_value'])];
    for (const row of result.trace) lines.push(csvRow([row.iteration, row.bestValue, row.meanValue, row.worstValue]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<AntColonyTSPParams, AntColonyTSPResult>({
  id: 'ant-colony-tsp',
  description: 'Ant Colony Optimization on TSP using pheromone graph-search stations and walk tokens.',
  schema: {
    kind: 'object',
    fields: {
      points: {kind: 'array', items: pointSchema, minLength: 2},
      ants: {kind: 'number', integer: true, min: 1, default: 18},
      iterations: {kind: 'number', integer: true, min: 1, default: 80},
      alpha: {kind: 'number', min: 0, default: 1},
      beta: {kind: 'number', min: 0, default: 3},
      evaporation: {kind: 'number', min: 0, max: 1, default: 0.28},
      deposit: {kind: 'number', min: 1e-12, default: 1},
      seed: {kind: 'number', integer: true, default: 5},
    },
    required: [],
  },
  run(params) {
    return runAntColonyTSP({...params, points: params.points && params.points.length > 0 ? params.points : undefined});
  },
  summarize(result) {
    return [
      'ANT COLONY TSP (DES)',
      '--------------------',
      `  Best length:    ${result.bestLength.toFixed(6)}`,
      `  Best tour:      ${result.bestTour.join(' -> ')}`,
      `  Iterations:     ${result.iterations}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iteration', 'best_length', 'mean_length', 'worst_length'])];
    for (const row of result.trace) lines.push(csvRow([row.iteration, row.bestLength, row.meanLength, row.worstLength]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<MapColoringCSPParams, MapColoringCSPResult>({
  id: 'map-coloring-csp',
  description: 'Constraint Satisfaction Problem solved by shared MRV backtracking tree-search station.',
  schema: {
    kind: 'object',
    fields: {
      variables: {kind: 'array', items: {kind: 'string'}, minLength: 1},
      colors: {kind: 'array', items: {kind: 'string'}, minLength: 1},
      edges: {kind: 'array', items: stringPairSchema, minLength: 1},
      maxNodes: {kind: 'number', integer: true, min: 1, default: 10000},
    },
    required: [],
  },
  run(params) {
    return runMapColoringCSP({
      ...params,
      variables: params.variables && params.variables.length > 0 ? params.variables : undefined,
      colors: params.colors && params.colors.length > 0 ? params.colors : undefined,
      edges: params.edges && params.edges.length > 0 ? params.edges as Array<[string, string]> : undefined,
    });
  },
  summarize(result) {
    return [
      'MAP COLORING CSP (DES)',
      '----------------------',
      `  Satisfied:      ${result.satisfied ? 'yes' : 'no'}`,
      `  Assignment:     ${Object.entries(result.assignment).map(([k, v]) => `${k}=${v}`).join(', ')}`,
      `  Nodes:          ${result.nodesProcessed}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
});

registerModel<MaxSATParams, MaxSATResult>({
  id: 'max-sat-local-search',
  description: 'SAT/MAX-SAT local search using the shared single-state optimizer station.',
  schema: {
    kind: 'object',
    fields: {
      numVars: {kind: 'number', integer: true, min: 1},
      clauses: {kind: 'array', items: numberVectorSchema, minLength: 1},
      iterations: {kind: 'number', integer: true, min: 1, default: 300},
      noise: {kind: 'number', min: 0, max: 1, default: 0.25},
      seed: {kind: 'number', integer: true, default: 13},
    },
    required: [],
  },
  run(params) {
    return runMaxSATLocalSearch({...params, clauses: params.clauses && params.clauses.length > 0 ? params.clauses : undefined});
  },
  summarize(result) {
    return [
      'MAX-SAT LOCAL SEARCH (DES)',
      '--------------------------',
      `  Satisfied:      ${result.satisfiedClauses}/${result.totalClauses}`,
      `  Complete SAT:   ${result.allSatisfied ? 'yes' : 'no'}`,
      `  Iterations:     ${result.iterations}`,
      `  Assignment:     [${result.assignment.map(v => v ? 'T' : 'F').join(', ')}]`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iteration', 'unsatisfied'])];
    for (const row of result.trace) lines.push(csvRow([row.iteration, row.unsatisfied]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<SDPMaxCutParams, SDPMaxCutResult>({
  id: 'sdp-maxcut-relaxation',
  description: 'Semidefinite Max-Cut relaxation through rank-constrained unit-vector station updates.',
  schema: {
    kind: 'object',
    fields: {
      nodes: {kind: 'number', integer: true, min: 2, default: 5},
      edges: {kind: 'array', items: weightedEdgeSchema, minLength: 1},
      rank: {kind: 'number', integer: true, min: 1, default: 3},
      iterations: {kind: 'number', integer: true, min: 1, default: 250},
      stepSize: {kind: 'number', min: 1e-12, default: 0.08},
      seed: {kind: 'number', integer: true, default: 17},
    },
    required: [],
  },
  run(params) {
    return runSDPMaxCutRelaxation({...params, edges: params.edges && params.edges.length > 0 ? params.edges as WeightedEdge[] : undefined});
  },
  summarize(result) {
    return [
      'SDP MAX-CUT RELAXATION (DES)',
      '----------------------------',
      `  SDP value:      ${result.sdpValue.toFixed(6)}`,
      `  Rounded cut:    ${result.roundedCutValue.toFixed(6)}`,
      `  Cut signs:      [${result.cut.join(', ')}]`,
      `  Iterations:     ${result.iterations}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iteration', 'objective'])];
    for (const row of result.trace) lines.push(csvRow([row.iteration, row.objective]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<ParetoPortfolioParams, ParetoPortfolioResult>({
  id: 'pareto-portfolio',
  description: 'Multi-objective risk/return optimization with a reusable Pareto archive station.',
  schema: {
    kind: 'object',
    fields: {
      assets: {kind: 'array', items: portfolioAssetSchema, minLength: 1},
      samples: {kind: 'number', integer: true, min: 1, default: 240},
      seed: {kind: 'number', integer: true, default: 19},
    },
    required: [],
  },
  run(params) {
    return runParetoPortfolio({...params, assets: params.assets && params.assets.length > 0 ? params.assets : undefined});
  },
  summarize(result) {
    return [
      'PARETO PORTFOLIO (DES)',
      '----------------------',
      `  Candidates:     ${result.candidateCount}`,
      `  Pareto points:  ${result.paretoFront.length}`,
      `  Hypervolume:    ${result.hypervolume.toExponential(4)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['risk', 'expected_return', 'weights'])];
    for (const point of result.paretoFront) lines.push(csvRow([point.risk, point.expectedReturn, JSON.stringify(point.weights)]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<HInfinityRobustControlParams, HInfinityRobustControlResult>({
  id: 'hinfinity-robust-control',
  description: 'H-infinity-style robust control against a worst-case bounded disturbance station.',
  schema: {
    kind: 'object',
    fields: {
      x0: {kind: 'number', default: 2},
      a: {kind: 'number', default: 0.25},
      b: {kind: 'number', default: 1},
      gain: {kind: 'number', min: 1e-12, default: 3.2},
      disturbanceMax: {kind: 'number', min: 0, default: 0.45},
      controlMax: {kind: 'number', min: 1e-12, default: 5},
      gamma: {kind: 'number', min: 1e-12, default: 2.5},
      dt: {kind: 'number', min: 1e-12, default: 0.03},
      numSteps: {kind: 'number', integer: true, min: 1, default: 260},
    },
    required: [],
  },
  run(params) { return runHInfinityRobustControl(params); },
  summarize(result) {
    return [
      'H-INFINITY ROBUST CONTROL (DES)',
      '-------------------------------',
      `  Final state:    ${result.finalState.toFixed(6)}`,
      `  Peak |state|:   ${result.peakAbsState.toFixed(6)}`,
      `  L2 gain est.:   ${result.l2GainEstimate.toFixed(6)} <= gamma ${result.gamma}`,
      `  Bounded:        ${result.boundedByGamma ? 'yes' : 'no'}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['tick', 'time', 'state', 'control', 'disturbance', 'cost'])];
    for (const row of result.trace) lines.push(csvRow([row.tick, row.time, row.state[0], row.control[0], row.disturbance[0], row.cost]));
    writeCsvLines(csvPath, lines);
  },
});

registerModel<PursuitEvasionGameParams, PursuitEvasionGameResult>({
  id: 'pursuit-evasion-game',
  description: 'Differential game: pursuit/evasion as plant, pursuer policy, and evader policy stations.',
  schema: {
    kind: 'object',
    fields: {
      pursuer: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
      evader: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
      pursuerSpeed: {kind: 'number', min: 1e-12, default: 1.25},
      evaderSpeed: {kind: 'number', min: 0, default: 0.6},
      captureRadius: {kind: 'number', min: 1e-12, default: 0.25},
      dt: {kind: 'number', min: 1e-12, default: 0.1},
      numSteps: {kind: 'number', integer: true, min: 1, default: 120},
    },
    required: [],
  },
  run(params) {
    return runPursuitEvasionGame({
      ...params,
      pursuer: params.pursuer && params.pursuer.length === 2 ? [params.pursuer[0], params.pursuer[1]] : undefined,
      evader: params.evader && params.evader.length === 2 ? [params.evader[0], params.evader[1]] : undefined,
    });
  },
  summarize(result) {
    return [
      'PURSUIT/EVASION DIFFERENTIAL GAME (DES)',
      '---------------------------------------',
      `  Capture tick:   ${result.captureTick === null ? 'not captured' : result.captureTick}`,
      `  Final distance: ${result.finalDistance.toFixed(6)}`,
      `  Steps recorded: ${result.distanceHistory.length}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['tick', 'time', 'px', 'py', 'ex', 'ey', 'ux', 'uy', 'wx', 'wy', 'distance'])];
    for (const row of result.trace) {
      const s = row.state;
      lines.push(csvRow([row.tick, row.time, s[0], s[1], s[2], s[3], row.control[0], row.control[1], row.disturbance[0], row.disturbance[1],
        Math.hypot(s[2] - s[0], s[3] - s[1])]));
    }
    writeCsvLines(csvPath, lines);
  },
});
