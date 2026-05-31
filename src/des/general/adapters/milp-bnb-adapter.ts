'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/milp-bnb-adapter.rs
//   (module des::general::adapters::milp_bnb_adapter)
// 1:1 file move. Registers MILP branch-and-bound + explicit IP/MIP-DES JSON adapters.
//
// Declarations → Rust:
//   interface MILPParams / IPMIPDESParams      -> struct (#[derive(Deserialize)];
//             nested raw?/knapsack?/options? -> Option<struct>)
//   const milpSchema / ipMipDESSchema: ParamSchema -> serde + validator metadata
//   const adapter / ipMipDESAdapter: DESModelRegistration<P,R> -> struct + impl trait;
//             registerModel(...) -> explicit registration calls
//
// Conversion notes (file-specific):
//   - `sense: 'max'|'min'`, `branchRule`, `nodeSelection`, and the long
//     `lpAlgorithm` literal union (LPRelaxationAlgorithm) -> #[serde] enums.
//   - GotChA: `lpAlgorithm: 'auto' as LPRelaxationAlgorithm` cast in the example —
//     in Rust this is just the enum variant, no cast.
//   - Empty optional arrays coerced to undefined (`ub.length > 0 ? ub : undefined`)
//     so the LP solver sees absent (not length-0) vectors -> Option + is_empty guard.
//   - `Object.entries(result.lpAlgorithmUsage)` (string-keyed map) -> HashMap iterate.
//   - `throw new Error` for missing {raw, knapsack} -> Result/validation.
//   - CSV writes optional trace fields with `?? ''` and `children?.join('|')` ->
//     Option formatting + iterator join.
// =============================================================================

// =============================================================================
// general/adapters/milp-bnb-adapter.ts — JSON adapter for the MILP-via-
// branch-and-bound solver. Demonstrates registry support for OPTIMISATION
// models with structured outputs.
// =============================================================================

import {
  solveMILP, buildKnapsackMILP, MILPProblem, MILPSolution,
} from '../milp-bnb';
import {
  buildBinaryKnapsackIP,
  IPMIPProblem,
  IPMIPSolution,
  IPMIPSolveOptions,
  LPRelaxationAlgorithm,
  solveIPMIPWithDES,
} from '../ip-mip-des';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {csvRow, writeCsvLines} from './adapter-utils';

interface MILPParams {
  /** Either provide a "raw" MILP block, or a "knapsack" convenience builder. */
  raw?: {
    sense: 'max' | 'min';
    c: number[];
    A: number[][];
    b: number[];
    integerVars: boolean[];
    ub?: number[];
    varNames?: string[];
    conNames?: string[];
  };
  knapsack?: {
    values: number[];
    weights: number[];
    capacity: number;
  };
  options?: {
    maxNodes?: number;
    lpMaxIters?: number;
    intTol?: number;
    branchRule?: 'most-fractional' | 'first-fractional';
    initialIncumbentZ?: number;
  };
}

const milpSchema: ParamSchema = {
  kind: 'object',
  description: 'MILP solved via branch-and-bound with IncrementalLP relaxations at each node.',
  fields: {
    raw: {kind: 'object', fields: {
      sense: {kind: 'string', enum: ['max', 'min']},
      c: {kind: 'array', items: {kind: 'number'}},
      A: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}}},
      b: {kind: 'array', items: {kind: 'number'}},
      integerVars: {kind: 'array', items: {kind: 'boolean'}},
      ub: {kind: 'array', items: {kind: 'number'}},
      varNames: {kind: 'array', items: {kind: 'string'}},
      conNames: {kind: 'array', items: {kind: 'string'}},
    }, required: ['sense', 'c', 'A', 'b', 'integerVars']},
    knapsack: {kind: 'object', fields: {
      values: {kind: 'array', items: {kind: 'number'}},
      weights: {kind: 'array', items: {kind: 'number'}},
      capacity: {kind: 'number'},
    }, required: ['values', 'weights', 'capacity']},
    options: {kind: 'object', fields: {
      maxNodes: {kind: 'number', integer: true, min: 1},
      lpMaxIters: {kind: 'number', integer: true, min: 1},
      intTol: {kind: 'number', min: 0},
      branchRule: {kind: 'string', enum: ['most-fractional', 'first-fractional']},
      initialIncumbentZ: {kind: 'number'},
    }, required: []},
  },
  required: [],
};

const adapter: DESModelRegistration<MILPParams, MILPSolution> = {
  id: 'milp-bnb',
  description: 'Mixed-integer LP via branch-and-bound, composing IncrementalLP for relaxations.',
  schema: milpSchema,

  run(params: MILPParams) {
    let problem: MILPProblem;
    if (params.knapsack) {
      problem = buildKnapsackMILP(
        params.knapsack.values, params.knapsack.weights, params.knapsack.capacity);
    } else if (params.raw) {
      // Treat missing/empty optional arrays as undefined so the LP solver
      // doesn't see a length-0 ub mismatching c.length.
      const ub = params.raw.ub && params.raw.ub.length > 0 ? params.raw.ub : undefined;
      const varNames = params.raw.varNames && params.raw.varNames.length > 0 ? params.raw.varNames : undefined;
      const conNames = params.raw.conNames && params.raw.conNames.length > 0 ? params.raw.conNames : undefined;
      problem = {
        sense: params.raw.sense, c: params.raw.c, A: params.raw.A, b: params.raw.b,
        integerVars: params.raw.integerVars, ub, varNames, conNames,
      };
    } else {
      throw new Error('milp-bnb: provide one of {raw, knapsack}');
    }
    return solveMILP(problem, params.options ?? {});
  },

  summarize(result: MILPSolution, _params: MILPParams): string {
    const xPretty = result.x.slice(0, 16).map(v => Number.isFinite(v) ? v.toFixed(3) : 'N/A').join(', ');
    return [
      'MILP-BRANCH-AND-BOUND RUN SUMMARY',
      '──────────────────────────────────',
      `  Status:           ${result.status}`,
      `  z*:               ${Number.isFinite(result.z) ? result.z.toFixed(6) : result.z}`,
      `  Best bound:       ${result.bestBound.toFixed(6)}`,
      `  Optimality gap:   ${result.gap.toExponential(2)}`,
      `  Nodes explored:   ${result.nodesExplored}`,
      `  LP pivots total:  ${result.totalPivots}`,
      '',
      `  x* (first 16):    [${xPretty}${result.x.length > 16 ? ', …' : ''}]`,
    ].join('\n');
  },

  writeCsv(result: MILPSolution, csvPath: string): void {
    const lines = ['var_index,value'];
    for (let i = 0; i < result.x.length; i++) lines.push(csvRow([i, result.x[i]]));
    lines.push('');
    lines.push('node_id,parent_id,depth,branch_var,branch_type,branch_value,lp_status,lp_z,fractional_count,pruned,pruned_reason,incumbent_updated');
    for (const e of result.trace) {
      lines.push(csvRow([
        e.nodeId, e.parentId ?? '', e.depth,
        e.branchVar ?? '', e.branchType ?? '', e.branchValue ?? '',
        e.lpStatus, e.lpZ ?? '', e.fractional.length,
        e.pruned, e.prunedReason ?? '', e.incumbentUpdated,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },

  examples: [
    {
      name: 'milp-knapsack-4item',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'milp-bnb',
        description: '4-item textbook knapsack',
        parameters: {
          knapsack: {values: [10, 40, 30, 50], weights: [5, 4, 6, 3], capacity: 10},
        },
      },
    },
  ],
};

registerModel(adapter);

// -----------------------------------------------------------------------------
// Explicit station-graph IP/MIP solver adapter
// -----------------------------------------------------------------------------

interface IPMIPDESParams {
  raw?: IPMIPProblem;
  knapsack?: {
    values: number[];
    weights: number[];
    capacity: number;
  };
  options?: IPMIPSolveOptions;
}

const ipMipDESSchema: ParamSchema = {
  kind: 'object',
  description: 'Integer / mixed-integer program solved by an explicit DES station graph.',
  fields: {
    raw: {kind: 'object', fields: {
      sense: {kind: 'string', enum: ['max', 'min']},
      c: {kind: 'array', items: {kind: 'number'}},
      A: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}}},
      b: {kind: 'array', items: {kind: 'number'}},
      integerVars: {kind: 'array', items: {kind: 'boolean'}},
      ub: {kind: 'array', items: {kind: 'number'}},
      varNames: {kind: 'array', items: {kind: 'string'}},
      conNames: {kind: 'array', items: {kind: 'string'}},
    }, required: ['sense', 'c', 'A', 'b', 'integerVars']},
    knapsack: {kind: 'object', fields: {
      values: {kind: 'array', items: {kind: 'number'}},
      weights: {kind: 'array', items: {kind: 'number'}},
      capacity: {kind: 'number'},
    }, required: ['values', 'weights', 'capacity']},
    options: {kind: 'object', fields: {
      maxNodes: {kind: 'number', integer: true, min: 1},
      maxTicks: {kind: 'number', integer: true, min: 1},
      lpMaxIters: {kind: 'number', integer: true, min: 1},
      intTol: {kind: 'number', min: 0},
      branchRule: {kind: 'string', enum: ['most-fractional', 'first-fractional']},
      nodeSelection: {kind: 'string', enum: ['dfs', 'best-bound']},
      lpAlgorithm: {kind: 'string', enum: [
        'auto',
        'incremental-primal-dual',
        'des-simplex-dantzig',
        'des-simplex-bland',
        'internal-simplex',
        'external-highs',
        'external-highs-ds',
        'external-highs-ipm',
      ]},
      maxCutRounds: {kind: 'number', integer: true, min: 0},
      maxCutsPerNode: {kind: 'number', integer: true, min: 0},
      heuristicPasses: {kind: 'number', integer: true, min: 0},
      allowExternalSolvers: {kind: 'boolean'},
    }, required: []},
  },
  required: [],
};

const ipMipDESAdapter: DESModelRegistration<IPMIPDESParams, IPMIPSolution> = {
  id: 'ip-mip-des',
  description: 'Integer/MIP solver graph: LP relaxation, rounding/repair, cuts, incumbent, and branching stations.',
  schema: ipMipDESSchema,

  run(params) {
    let problem: IPMIPProblem;
    if (params.knapsack) {
      problem = buildBinaryKnapsackIP(params.knapsack.values, params.knapsack.weights, params.knapsack.capacity);
    } else if (params.raw) {
      problem = {
        ...params.raw,
        ub: params.raw.ub && params.raw.ub.length > 0 ? params.raw.ub : undefined,
        varNames: params.raw.varNames && params.raw.varNames.length > 0 ? params.raw.varNames : undefined,
        conNames: params.raw.conNames && params.raw.conNames.length > 0 ? params.raw.conNames : undefined,
      };
    } else {
      throw new Error('ip-mip-des: provide one of {raw, knapsack}');
    }
    return solveIPMIPWithDES(problem, params.options ?? {});
  },

  summarize(result) {
    const xPretty = result.x.slice(0, 16).map(v => Number.isFinite(v) ? v.toFixed(3) : 'N/A').join(', ');
    return [
      'IP/MIP DES SOLVER GRAPH',
      '-----------------------',
      `  Status:           ${result.status}`,
      `  Execution mode:   ${result.executionMode}`,
      `  In-house only:    ${result.inHouseOnly}`,
      `  LP backend:       ${result.lpAlgorithm}`,
      `  LP backend usage: ${Object.entries(result.lpAlgorithmUsage).map(([k, v]) => `${k}=${v}`).join(', ') || 'none'}`,
      `  Technique plan:   ${result.techniquePlan.rootLPAlgorithm}${result.techniquePlan.externalCandidate ? ' (external candidate)' : ''}`,
      `  z*:               ${Number.isFinite(result.z) ? result.z.toFixed(6) : result.z}`,
      `  Best bound:       ${Number.isFinite(result.bestBound) ? result.bestBound.toFixed(6) : result.bestBound}`,
      `  Gap:              ${result.gap.toExponential(2)}`,
      `  Nodes explored:   ${result.nodesExplored}`,
      `  Elapsed:          ${result.performance.elapsedMs} ms (${result.performance.nodesPerSecond.toFixed(2)} nodes/s)`,
      `  LP solves:        ${result.lpSolves}`,
      `  LP solver time:   ${result.performance.totalLPSolverMs} ms`,
      `  LP iterations:    ${result.totalLPIterations}`,
      `  Cuts added:       ${result.cutsAdded}`,
      `  Candidates tried: ${result.candidatesTried}`,
      `  Solver tokens:    ${result.tokenStats.created} (${result.tokenStats.stateful} stateful / ${result.tokenStats.stateless} stateless)`,
      `  Incumbent source: ${result.incumbentSource ?? 'none'}`,
      `  x* (first 16):    [${xPretty}${result.x.length > 16 ? ', ...' : ''}]`,
    ].join('\n');
  },

  writeCsv(result, csvPath) {
    const lines = ['node_id,parent_id,depth,lp_status,lp_z,solver,fractional_count,action,reason,children,cuts_added,node_token_id,lineage_root,token_generation,state_mode'];
    for (const e of result.trace) {
      lines.push(csvRow([
        e.nodeId, e.parentId ?? '', e.depth,
        e.lpStatus, e.lpZ ?? '', e.solver, e.fractional.length,
        e.action, e.reason ?? '', e.children?.join('|') ?? '', e.cutsAdded ?? '',
        e.nodeTokenId ?? '', e.lineageRoot ?? '', e.tokenGeneration ?? '', e.stateMode ?? '',
      ]));
    }
    writeCsvLines(csvPath, lines);
  },

  examples: [{
    name: 'ip-mip-des-knapsack',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'ip-mip-des',
      description: '4-item knapsack solved by the explicit IP/MIP DES station graph.',
      parameters: {
        knapsack: {values: [10, 40, 30, 50], weights: [5, 4, 6, 3], capacity: 10},
        options: {lpAlgorithm: 'auto' as LPRelaxationAlgorithm, maxCutRounds: 1},
      },
    },
  }],
};

registerModel(ipMipDESAdapter);
