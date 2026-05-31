// RUST MIGRATION: Target module `src/des/general/adapters/network_flow_adapters.rs`.
// RUST MIGRATION: Convert the grouped max-flow, stochastic-flow MDP, and traffic adapters into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map flow graphs, MDP transition payloads, traffic params, and results to `serde` config/result structs; output paths become `PathBuf`.
// RUST MIGRATION: Return `Result<_, ValidationError>` for malformed capacities, probabilities, and network topology.
'use strict';

// =============================================================================
// JSON adapters for network-flow models:
//   - max-flow: Edmonds-Karp augmenting paths as fixed-point DES ticks
//   - traffic-flow: small continuous-position traffic simulation on a grid
// =============================================================================

import {DESModelRegistration, DESRuntimeConfig, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  buildTextbookMaxFlowProblem,
  MaxFlowProblem,
  MaxFlowResult,
  solveMaxFlow,
} from '../max-flow';
import {
  buildDefaultTrafficProblem,
  buildTrafficMaxFlowProblem,
  runTrafficSimulation,
  TrafficProblem,
  TrafficSimulationResult,
} from '../traffic-flow';
import {
  buildDefaultStochasticFlowMDPProblem,
  solveStochasticFlowMDP,
  StochasticFlowMDPProblem,
  StochasticFlowMDPResult,
} from '../stochastic-flow-mdp';
import {csvRow, jsonCsvRow, writeCsvLines} from './adapter-utils';

// -----------------------------------------------------------------------------
// Max flow
// -----------------------------------------------------------------------------

interface MaxFlowParams {
  builtin?: 'textbook';
  problem?: MaxFlowProblem;
}

const maxFlowEdgeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    from: {kind: 'number', integer: true, min: 0},
    to: {kind: 'number', integer: true, min: 0},
    capacity: {kind: 'number', min: 0},
    name: {kind: 'string'},
  },
  required: ['from', 'to', 'capacity'],
};

const maxFlowProblemSchema: ParamSchema = {
  kind: 'object',
  description: 'Directed capacitated network with a single source and sink.',
  fields: {
    numNodes: {kind: 'number', integer: true, min: 2},
    source: {kind: 'number', integer: true, min: 0},
    sink: {kind: 'number', integer: true, min: 0},
    edges: {kind: 'array', items: maxFlowEdgeSchema, minLength: 1},
  },
  required: ['numNodes', 'source', 'sink', 'edges'],
};

const maxFlowSchema: ParamSchema = {
  kind: 'object',
  description: 'Maximum flow solved by DES augmenting-path iterations.',
  fields: {
    builtin: {kind: 'string', enum: ['textbook'], default: 'textbook'},
    problem: maxFlowProblemSchema,
  },
  required: [],
};

const maxFlowAdapter: DESModelRegistration<MaxFlowParams, MaxFlowResult> = {
  id: 'max-flow',
  description: 'Maximum flow/min-cut optimisation via Edmonds-Karp DES ticks.',
  schema: maxFlowSchema,
  run(params) {
    return solveMaxFlow(params.problem ?? buildTextbookMaxFlowProblem());
  },
  summarize(result) {
    return [
      'MAX-FLOW OPTIMISATION',
      '---------------------',
      `  Status:          ${result.status}`,
      `  Max flow:        ${result.maxFlow.toFixed(6)}`,
      `  Iterations:      ${result.iterations}`,
      `  Augmentations:   ${result.trace.length}`,
      `  Min-cut cap:     ${result.minCut.capacity.toFixed(6)}`,
      `  Source side:     {${result.minCut.sourceSide.join(', ')}}`,
      `  Cut edges:       ${result.minCut.cutEdges.map(e => e.name ?? `${e.from}->${e.to}`).join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['from,to,name,capacity,flow'];
    for (const e of result.edgeFlows) {
      lines.push(csvRow([e.from, e.to, e.name ?? '', e.capacity, e.flow]));
    }
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'textbook',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'max-flow',
      description: 'Textbook six-node maximum-flow/min-cut example.',
      parameters: {builtin: 'textbook'},
    },
  }],
};

registerModel(maxFlowAdapter);

// -----------------------------------------------------------------------------
// Stochastic flow MDP
// -----------------------------------------------------------------------------

interface StochasticFlowMDPParams {
  builtin?: 'small-stochastic-network';
  problem?: StochasticFlowMDPProblem;
  seed?: number;
  maxPolicyRows?: number;
}

const stochasticFlowEdgeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    from: {kind: 'number', integer: true, min: 0},
    to: {kind: 'number', integer: true, min: 0},
    capacity: {kind: 'number', integer: true, min: 0},
    successProb: {kind: 'number', min: 0, max: 1},
    cost: {kind: 'number', min: 0},
    name: {kind: 'string'},
  },
  required: ['from', 'to', 'capacity', 'successProb'],
};

const stochasticFlowProblemSchema: ParamSchema = {
  kind: 'object',
  description: 'Finite-horizon stochastic flow-control MDP on a directed network.',
  fields: {
    numNodes: {kind: 'number', integer: true, min: 2},
    source: {kind: 'number', integer: true, min: 0},
    sink: {kind: 'number', integer: true, min: 0},
    edges: {kind: 'array', items: stochasticFlowEdgeSchema, minLength: 1},
    horizon: {kind: 'number', integer: true, min: 1},
    deliveredReward: {kind: 'number', min: 1e-9},
    waitPenalty: {kind: 'number', min: 0},
    failurePenalty: {kind: 'number', min: 0},
    discount: {kind: 'number', min: 0, max: 1},
    maxStates: {kind: 'number', integer: true, min: 1},
  },
  required: ['numNodes', 'source', 'sink', 'edges', 'horizon'],
};

const stochasticFlowMDPSchema: ParamSchema = {
  kind: 'object',
  description: 'MDP interpretation of max-flow when edge availability/capacity is stochastic.',
  fields: {
    builtin: {kind: 'string', enum: ['small-stochastic-network'], default: 'small-stochastic-network'},
    problem: stochasticFlowProblemSchema,
    seed: {kind: 'number', integer: true, default: 7},
    maxPolicyRows: {kind: 'number', integer: true, min: 1, default: 24},
  },
  required: [],
};

const stochasticFlowMDPAdapter: DESModelRegistration<StochasticFlowMDPParams, StochasticFlowMDPResult> = {
  id: 'stochastic-flow-mdp',
  description: 'MDP interpretation of max-flow: stochastic capacities/availability with sequential routing control.',
  schema: stochasticFlowMDPSchema,
  run(params, runtime: DESRuntimeConfig) {
    return solveStochasticFlowMDP(params.problem ?? buildDefaultStochasticFlowMDPProblem(), {
      seed: runtime.seed ?? params.seed ?? 7,
      maxPolicyRows: params.maxPolicyRows ?? 24,
    });
  },
  summarize(result) {
    const first = result.initialPolicy.slice(0, 5)
      .map(row => `t${row.stage}:${row.action.label}`)
      .join(' -> ');
    return [
      'STOCHASTIC FLOW-CONTROL MDP',
      '---------------------------',
      `  Horizon:         ${result.horizon}`,
      `  States:          ${result.numStates}`,
      `  E[reward]*:      ${result.expectedReward.toFixed(6)}`,
      `  Static max-flow: ${result.deterministicMaxFlow.toFixed(6)}  (deterministic upper bound)`,
      `  First policy:    ${first}`,
      `  Sim delivered:   ${result.simulation.delivered}`,
      `  Sim reward:      ${result.simulation.totalReward.toFixed(6)}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['stage,state_index,node,capacities,action,value'];
    for (const row of result.policy) {
      lines.push(jsonCsvRow([
        row.stage,
        row.stateIndex,
        row.state.node,
        row.state.capacities,
        row.action.label,
        row.value,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'small-stochastic-network',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'stochastic-flow-mdp',
      description: 'MDP interpretation of max-flow with stochastic edge availability.',
      parameters: {builtin: 'small-stochastic-network', seed: 7},
      runtime: {seed: 7},
    },
  }],
};

registerModel(stochasticFlowMDPAdapter);

// -----------------------------------------------------------------------------
// Traffic flow
// -----------------------------------------------------------------------------

interface TrafficParams {
  builtin?: 'five-intersection';
  problem?: TrafficProblem;
}

const trafficNodeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'number', integer: true, min: 0},
    name: {kind: 'string'},
    x: {kind: 'number'},
    y: {kind: 'number'},
    signalOffsetSec: {kind: 'number', min: 0},
  },
  required: ['id', 'name', 'x', 'y'],
};

const trafficLinkSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    from: {kind: 'number', integer: true, min: 0},
    to: {kind: 'number', integer: true, min: 0},
    lengthM: {kind: 'number', min: 1e-9},
    speedLimitMps: {kind: 'number', min: 1e-9},
    capacity: {kind: 'number', integer: true, min: 1, max: 299},
    dischargePerMin: {kind: 'number', min: 1e-9},
  },
  required: ['id', 'from', 'to', 'lengthM', 'speedLimitMps'],
};

const trafficSourceSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    node: {kind: 'number', integer: true, min: 0},
    destNode: {kind: 'number', integer: true, min: 0},
    ratePerMin: {kind: 'number', min: 0},
    maxGenerated: {kind: 'number', integer: true, min: 0},
    startSec: {kind: 'number', min: 0},
    endSec: {kind: 'number', min: 0},
  },
  required: ['id', 'node', 'destNode', 'ratePerMin'],
};

const trafficProblemSchema: ParamSchema = {
  kind: 'object',
  description: 'Small directed road network with signalized intersections and moving cars.',
  fields: {
    nodes: {kind: 'array', items: trafficNodeSchema, minLength: 2},
    links: {kind: 'array', items: trafficLinkSchema, minLength: 1},
    sources: {kind: 'array', items: trafficSourceSchema, minLength: 1},
    durationSec: {kind: 'number', min: 1e-9},
    dtSec: {kind: 'number', min: 1e-9},
    maxCars: {kind: 'number', integer: true, min: 1, max: 299},
    minGapM: {kind: 'number', min: 1e-9},
    accelMps2: {kind: 'number', min: 1e-9},
    signalCycleSec: {kind: 'number', min: 1e-9},
    drainAfterSourcesSec: {kind: 'number', min: 0},
    seed: {kind: 'number', integer: true},
  },
  required: [
    'nodes', 'links', 'sources', 'durationSec', 'dtSec', 'maxCars',
    'minGapM', 'accelMps2', 'signalCycleSec',
  ],
};

const trafficSchema: ParamSchema = {
  kind: 'object',
  description: 'Traffic-flow simulation with stationary grid/link/intersection entities and moving cars.',
  fields: {
    builtin: {kind: 'string', enum: ['five-intersection'], default: 'five-intersection'},
    problem: trafficProblemSchema,
  },
  required: [],
};

const trafficAdapter: DESModelRegistration<TrafficParams, TrafficSimulationResult> = {
  id: 'traffic-flow',
  description: 'Continuous-position traffic simulation on a five-intersection grid with max-flow upper bound.',
  schema: trafficSchema,
  run(params, runtime: DESRuntimeConfig) {
    const problem = {...(params.problem ?? buildDefaultTrafficProblem())};
    if (runtime.seed !== undefined) problem.seed = runtime.seed;
    return runTrafficSimulation(problem);
  },
  summarize(result) {
    return [
      'TRAFFIC-FLOW DES',
      '----------------',
      `  Generated:       ${result.generatedCars}`,
      `  Completed:       ${result.completedCars}`,
      `  Active at stop:  ${result.activeCars}`,
      `  Max active:      ${result.maxActiveCars}`,
      `  Blocked tries:   ${result.blockedSourceAttempts}`,
      `  Mean travel:     ${result.meanTravelTimeSec.toFixed(3)} sec`,
      `  P95 travel:      ${result.p95TravelTimeSec.toFixed(3)} sec`,
      `  Throughput:      ${result.throughputPerHour.toFixed(3)} cars/hour`,
      `  Max-flow bound:  ${result.maxFlowUpperBoundPerMin.toFixed(3)} cars/min`,
      `  Throughput/bnd:  ${result.throughputVsMaxFlow.toFixed(3)}`,
      `  Invariants:      ${result.invariantViolations.length === 0 ? 'ok' : `${result.invariantViolations.length} violations`}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['id,from,to,capacity,entered,exited,final_occupancy,max_occupancy,avg_occupancy'];
    for (const l of result.linkStats) {
      lines.push(csvRow([
        l.id, l.from, l.to, l.capacity, l.entered, l.exited,
        l.finalOccupancy, l.maxOccupancy, l.avgOccupancy,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'five-intersection',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'traffic-flow',
      description: 'Five-intersection traffic-flow scenario with fewer than 300 cars.',
      parameters: {builtin: 'five-intersection'},
      runtime: {seed: 7},
    },
  }],
};

registerModel(trafficAdapter);
