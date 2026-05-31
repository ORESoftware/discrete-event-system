// RUST MIGRATION: Target module `src/des/general/adapters/multistage_sddp_adapter.rs`.
// RUST MIGRATION: Convert the multi-stage SDDP adapter registration into Rust adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map stage demand outcomes, stochastic program config, cuts, and run results to `serde` structs; filesystem paths become `PathBuf`.
// RUST MIGRATION: Express bad stage data, probabilities, and scenario validation as `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/multistage-sddp-adapter.rs
//   (module des::general::adapters::multistage_sddp_adapter)
// 1:1 file move. JSON adapter registering the multistage-SDDP inventory model.
//
// Declarations → Rust:
//   interface MultiStageParams                 -> struct (problem?/options? -> Option)
//   const demandOutcomeSchema/multiStageProblemSchema/multiStageSchema: ParamSchema
//                                        -> serde + validator metadata
//   const multiStageAdapter: DESModelRegistration<P,R> -> struct + impl trait;
//             registerModel(...) -> explicit registration
//
// Conversion notes (file-specific):
//   - `demands` is `number[][]` of demand/prob outcomes (per-stage scenario lists)
//     -> Vec<Vec<DemandOutcome>>.
//   - `params.problem ?? buildDefault…()`, `params.options ?? {}` -> Option::unwrap_or(_else).
//   - CSV writes `tr.policyValue ?? ''` / `tr.gapToExact ?? ''` -> Option formatting;
//     `gapToExact?.toExponential(3) ?? 'n/a'` in summary likewise.
//   - NOTE: this is a near-duplicate of the multistage block in
//     stochastic-optimization-adapters.ts; both register id 'multistage-sddp'.
// =============================================================================

import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  buildDefaultMultiStageInventoryProblem,
  MultiStageInventoryProblem,
  MultiStageRunResult,
  runMultiStageInventoryDemo,
  SDDPOptions,
} from '../multistage-stochastic';
import {csvRow, writeCsvLines} from './adapter-utils';

interface MultiStageParams {
  problem?: MultiStageInventoryProblem;
  options?: SDDPOptions;
}

const demandOutcomeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    demand: {kind: 'number', min: 0},
    prob: {kind: 'number', min: 0, max: 1},
  },
  required: ['demand', 'prob'],
};

const multiStageProblemSchema: ParamSchema = {
  kind: 'object',
  description: 'Multi-stage inventory/storage stochastic program.',
  fields: {
    horizon: {kind: 'number', integer: true, min: 1},
    initialInventory: {kind: 'number', min: 0},
    capacity: {kind: 'number', min: 1e-9},
    maxOrder: {kind: 'array', items: {kind: 'number', min: 0}},
    price: {kind: 'array', items: {kind: 'number', min: 0}},
    orderCost: {kind: 'array', items: {kind: 'number', min: 0}},
    holdCost: {kind: 'array', items: {kind: 'number', min: 0}},
    stockoutCost: {kind: 'array', items: {kind: 'number', min: 0}},
    salvageValue: {kind: 'number', min: 0},
    demands: {kind: 'array', items: {kind: 'array', items: demandOutcomeSchema}},
  },
  required: [
    'horizon', 'initialInventory', 'capacity', 'maxOrder', 'price',
    'orderCost', 'holdCost', 'stockoutCost', 'salvageValue', 'demands',
  ],
};

const multiStageSchema: ParamSchema = {
  kind: 'object',
  description: 'Multi-stage stochastic inventory solved by SDDP and exact scenario tree validation.',
  fields: {
    problem: multiStageProblemSchema,
    options: {kind: 'object', fields: {
      maxIter: {kind: 'number', integer: true, min: 1, default: 80},
      tol: {kind: 'number', min: 0, default: 1e-4},
      seed: {kind: 'number', integer: true, default: 1},
      evaluatePolicyEvery: {kind: 'number', integer: true, min: 1, default: 80},
      finiteDiffStep: {kind: 'number', min: 1e-9},
      cutGridSize: {kind: 'number', integer: true, min: 2, default: 21},
    }, required: []},
  },
  required: [],
};

const multiStageAdapter: DESModelRegistration<MultiStageParams, MultiStageRunResult> = {
  id: 'multistage-sddp',
  description: 'Multi-stage stochastic inventory/storage optimisation via SDDP cut recursion.',
  schema: multiStageSchema,
  run(params) {
    const problem = params.problem ?? buildDefaultMultiStageInventoryProblem();
    return runMultiStageInventoryDemo(problem, params.options ?? {});
  },
  summarize(result) {
    return [
      'MULTI-STAGE STOCHASTIC PROGRAM (SDDP)',
      '-------------------------------------',
      `  Exact tree:      ${result.exact.status}  z=${result.exact.objective.toFixed(6)}  nodes=${result.exact.nodeCount}`,
      `  SDDP status:     ${result.sddp.status}`,
      `  SDDP iters:      ${result.sddp.iterations}`,
      `  Upper bound:     ${result.sddp.upperBound.toFixed(6)}`,
      `  Policy value:    ${result.sddp.policyValue.toFixed(6)}`,
      `  Gap to exact:    ${result.sddp.gapToExact?.toExponential(3) ?? 'n/a'}`,
      `  Cuts/stage:      [${result.sddp.cutsPerStage.join(', ')}]`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['iter,upper_bound,policy_value,gap_to_exact,terminal_inventory,cuts_added'];
    for (const tr of result.sddp.trace) {
      lines.push(csvRow([
        tr.iter,
        tr.upperBound,
        tr.policyValue ?? '',
        tr.gapToExact ?? '',
        tr.terminalInventory,
        tr.cutsAdded.length,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
};

registerModel(multiStageAdapter);
