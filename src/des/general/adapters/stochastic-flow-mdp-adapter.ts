// RUST MIGRATION: Target module `src/des/general/adapters/stochastic_flow_mdp_adapter.rs`.
// RUST MIGRATION: Convert stochastic-flow MDP adapter registration into Rust adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map graph edges, stochastic transition params, policies, and results to `serde` config/result structs; output paths become `PathBuf`.
// RUST MIGRATION: Return `Result<_, ValidationError>` for invalid capacities, probabilities, states, and action definitions.
'use strict';

// =============================================================================
// JSON adapter for stochastic-flow MDP.
// =============================================================================

import {DESModelRegistration, DESRuntimeConfig, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  buildDefaultStochasticFlowMDPProblem,
  solveStochasticFlowMDP,
  StochasticFlowMDPProblem,
  StochasticFlowMDPResult,
} from '../stochastic-flow-mdp';
import {jsonCsvRow as csvRow, writeCsvLines} from './adapter-utils';

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
      lines.push(csvRow([
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
