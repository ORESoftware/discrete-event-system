// RUST MIGRATION: Target module `src/des/general/adapters/stochastic_optimization_adapters.rs`.
// RUST MIGRATION: Convert stochastic LP and multi-stage optimization adapter registrations into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Promote pair schemas, demand outcomes, stochastic configs, cuts, and run results to `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Use `Result<_, ValidationError>` for probability, scenario, vector-shape, and solver input validation.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/stochastic-optimization-adapters.rs
//   (module des::general::adapters::stochastic_optimization_adapters)
// 1:1 file move. Registers two-stage stochastic-LP (SAA/Benders) + multistage-SDDP adapters.
//
// Declarations → Rust:
//   interface StochasticLPParams / StochasticLPRunResult / MultiStageParams -> struct
//             (#[derive(Deserialize)]; optional fields -> Option<T>)
//   const pairSchema / *Schema: ParamSchema    -> serde + validator metadata
//   const stochasticLPAdapter / multiStageAdapter: DESModelRegistration<P,R> ->
//             struct + impl ModelAdapter trait; registerModel(...) -> explicit calls
//   fn fmtVec(x: readonly number[])             -> plain `fn` formatting helper
//
// Conversion notes (file-specific):
//   - GotChA: `closedForm?: ReturnType<typeof solveProductionClosedForm>` derives a
//     type from a function's return — name that result type explicitly in Rust
//     (Option<ClosedFormResult>); TS `ReturnType<>` has no analogue.
//   - `ranges: Array<[number, number]>` -> Vec<(f64, f64)>; validated to equal-length
//     c/p/ranges with `throw new Error` (invariant -> Result/validation).
//   - The `evalX` closure captures `params`/`oos` -> a closure or local fn; watch
//     borrow of captured slices.
//   - `params.N ?? 200`, `gapToExact?.toExponential(3) ?? 'n/a'` -> unwrap_or chains;
//     `tr.policyValue ?? ''` in CSV -> Option formatting.
//   - `params.budget === undefined` gates the closed-form branch -> Option::is_none.
// =============================================================================

// =============================================================================
// JSON adapters for stochastic-optimisation models:
//   - stochastic-lp: existing two-stage SAA + Benders model, now first-class
//   - multistage-sddp: multi-stage inventory/storage SDDP model
// =============================================================================

import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  buildProductionSLP, buildProductionScenarios,
  solveSLPMonolithic, solveSLPBenders, solveProductionClosedForm,
  SLPSolveResult,
} from '../stochastic-lp';
import {
  buildDefaultMultiStageInventoryProblem,
  MultiStageInventoryProblem,
  MultiStageRunResult,
  runMultiStageInventoryDemo,
  SDDPOptions,
} from '../multistage-stochastic';
import {csvRow, jsonCsvRow, writeCsvLines} from './adapter-utils';

// -----------------------------------------------------------------------------
// Two-stage stochastic LP adapter
// -----------------------------------------------------------------------------

interface StochasticLPParams {
  c: number[];
  p: number[];
  ranges: Array<[number, number]>;
  N?: number;
  seed?: number;
  budget?: number;
  tol?: number;
  maxIter?: number;
  oosN?: number;
}

interface StochasticLPRunResult {
  closedForm?: ReturnType<typeof solveProductionClosedForm>;
  monolithic: SLPSolveResult;
  benders: SLPSolveResult;
  outOfSample?: {
    N: number;
    monolithic: number;
    benders: number;
    closedForm?: number;
  };
}

const pairSchema: ParamSchema = {
  kind: 'array',
  items: {kind: 'number'},
  minLength: 2,
  maxLength: 2,
};

const stochasticLPSchema: ParamSchema = {
  kind: 'object',
  description: 'Two-stage stochastic LP: production capacity under demand uncertainty.',
  fields: {
    c: {kind: 'array', items: {kind: 'number'}},
    p: {kind: 'array', items: {kind: 'number'}},
    ranges: {kind: 'array', items: pairSchema},
    N: {kind: 'number', integer: true, min: 1, default: 200},
    seed: {kind: 'number', integer: true, default: 42},
    budget: {kind: 'number', min: 0},
    tol: {kind: 'number', min: 0, default: 1e-7},
    maxIter: {kind: 'number', integer: true, min: 1, default: 200},
    oosN: {kind: 'number', integer: true, min: 0, default: 0},
  },
  required: ['c', 'p', 'ranges'],
};

const stochasticLPAdapter: DESModelRegistration<StochasticLPParams, StochasticLPRunResult> = {
  id: 'stochastic-lp',
  description: 'Two-stage stochastic LP via SAA and Benders/L-shaped decomposition.',
  schema: stochasticLPSchema,
  run(params) {
    if (params.c.length !== params.p.length || params.c.length !== params.ranges.length) {
      throw new Error('stochastic-lp: c, p, and ranges must have the same length');
    }
    const N = params.N ?? 200;
    const seed = params.seed ?? 42;
    const slp = buildProductionSLP(params.c, params.p, params.budget);
    const scenarios = buildProductionScenarios({ranges: params.ranges, seed}, N);
    const closedForm = params.budget === undefined
      ? solveProductionClosedForm(params.c, params.p, params.ranges)
      : undefined;
    const monolithic = solveSLPMonolithic(slp, scenarios);
    const benders = solveSLPBenders(slp, scenarios, {tol: params.tol ?? 1e-7, maxIter: params.maxIter ?? 200});
    let outOfSample: StochasticLPRunResult['outOfSample'];
    const oosN = params.oosN ?? 0;
    if (oosN > 0) {
      const oos = buildProductionScenarios({ranges: params.ranges, seed: seed + 99991}, oosN);
      const evalX = (x: number[]): number => {
        let z = 0;
        for (let i = 0; i < params.c.length; i++) z += -params.c[i] * x[i];
        let q = 0;
        for (const sc of oos) {
          for (let i = 0; i < params.p.length; i++) q += params.p[i] * Math.min(x[i], sc.meta.D[i]);
        }
        return z + q / oos.length;
      };
      outOfSample = {
        N: oosN,
        monolithic: evalX(monolithic.x),
        benders: evalX(benders.x),
        closedForm: closedForm ? evalX(closedForm.x) : undefined,
      };
    }
    return {closedForm, monolithic, benders, outOfSample};
  },
  summarize(result, params) {
    const lines = [
      'STOCHASTIC LP (two-stage SAA + Benders)',
      '---------------------------------------',
      `  Scenarios:       ${params.N ?? 200}`,
      `  Budget:          ${params.budget ?? 'none'}`,
      `  Monolithic:      ${result.monolithic.status}  z=${result.monolithic.objective.toFixed(6)}  x=[${fmtVec(result.monolithic.x)}]`,
      `  Benders:         ${result.benders.status}  z=${result.benders.objective.toFixed(6)}  x=[${fmtVec(result.benders.x)}]`,
      `  |Delta z|:       ${Math.abs(result.benders.objective - result.monolithic.objective).toExponential(2)}`,
      `  Benders iters:   ${result.benders.iterations}`,
    ];
    if (result.closedForm) {
      lines.push(`  Closed-form:     z=${result.closedForm.objective.toFixed(6)}  x=[${fmtVec(result.closedForm.x)}]`);
    }
    if (result.outOfSample) {
      lines.push(`  OOS N=${result.outOfSample.N}: monolithic=${result.outOfSample.monolithic.toFixed(4)}  benders=${result.outOfSample.benders.toFixed(4)}`);
    }
    return lines.join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = ['method,status,objective,iterations,x'];
    for (const row of [result.monolithic, result.benders]) {
      lines.push(jsonCsvRow([row.method, row.status, row.objective, row.iterations, row.x]));
    }
    writeCsvLines(csvPath, lines);
  },
};

registerModel(stochasticLPAdapter);

// -----------------------------------------------------------------------------
// Multi-stage SDDP adapter
// -----------------------------------------------------------------------------

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

function fmtVec(x: readonly number[]): string {
  return x.map(v => v.toFixed(4)).join(', ');
}
