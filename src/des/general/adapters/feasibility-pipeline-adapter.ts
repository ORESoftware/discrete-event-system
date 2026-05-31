// RUST MIGRATION: Target module `src/des/general/adapters/feasibility_pipeline_adapter.rs`.
// RUST MIGRATION: Convert feasibility pipeline adapter registration and drawing helpers into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Promote variables, constraints, candidate evaluations, and improvements to `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Make constraint validation and infeasible candidate handling explicit with `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/feasibility-pipeline-adapter.rs
//   (module des::general::adapters::feasibility_pipeline_adapter)
// 1:1 file move. JSON adapter for the feasibility-checker/improver pipeline, with
// an animated station-graph scene builder.
//
// Declarations → Rust:
//   const coefficientMapSchema/variableSchema/objectiveSchema/constraintSchema/
//         problemSchema/candidateSchema/improvementSchema/feasibilitySchema: ParamSchema
//                                        -> serde + validator metadata
//   fn formatNumber / valuesSummary / drawPipeline -> plain `fn` helpers
//   const adapter: DESModelRegistration<P,R> -> struct + impl ModelAdapter trait;
//             registerModel(adapter) -> explicit registration
//
// Conversion notes (file-specific):
//   - GotChA: objective/constraint `coefficients` and candidate `values` are open
//     `{kind:'object', fields:{}}` maps (variable-name → number) -> HashMap<String, f64>;
//     `Object.entries(e.values)` iteration order is not guaranteed in Rust.
//   - `type: 'continuous'|'integer'|'binary'`, `sense: '<='|'>='|'='` / `'min'|'max'`
//     literal unions -> enums.
//   - Shapes pushed into `Shape[]` (animation/types) -> Vec<Shape>; Shape -> enum;
//     animation derives from the result trace (no RNG).
//   - async run/animate via withLogger + FrameRecorder -> async fns; `?? default`
//     (constraints?.length ?? 0, parentId ?? '') -> Option handling.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {framesPath, jsonCsvRow as csvRow, validationLine, withLogger, writeCsvLines} from './adapter-utils';
import {
  FeasibilityEvaluation,
  FeasibilityPipelineParams,
  FeasibilityPipelineResult,
  runFeasibilityPipeline,
} from '../feasibility-pipeline';

const coefficientMapSchema: ParamSchema = {
  kind: 'object',
  fields: {},
  required: [],
  description: 'Map from variable name to finite coefficient.',
};

const variableSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    type: {kind: 'string', enum: ['continuous', 'integer', 'binary'], default: 'continuous'},
    lb: {kind: 'number'},
    ub: {kind: 'number'},
    step: {kind: 'number', min: 0},
  },
  required: ['name'],
};

const objectiveSchema: ParamSchema = {
  kind: 'object',
  fields: {
    constant: {kind: 'number', default: 0},
    coefficients: coefficientMapSchema,
  },
  required: ['coefficients'],
};

const constraintSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    coefficients: coefficientMapSchema,
    sense: {kind: 'string', enum: ['<=', '>=', '=']},
    rhs: {kind: 'number'},
    tolerance: {kind: 'number', min: 0},
  },
  required: ['coefficients', 'sense', 'rhs'],
};

const problemSchema: ParamSchema = {
  kind: 'object',
  fields: {
    sense: {kind: 'string', enum: ['min', 'max']},
    variables: {kind: 'array', items: variableSchema, minLength: 1},
    objective: objectiveSchema,
    constraints: {kind: 'array', items: constraintSchema},
    tolerance: {kind: 'number', min: 0, default: 1e-8},
  },
  required: ['sense', 'variables', 'objective'],
};

const candidateSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string', default: 'user-candidate'},
    values: {kind: 'object', fields: {}, required: []},
    vector: {kind: 'array', items: {kind: 'number'}},
  },
  required: [],
};

const improvementSchema: ParamSchema = {
  kind: 'object',
  fields: {
    enabled: {kind: 'boolean', default: true},
    maxIterations: {kind: 'number', integer: true, min: 0, default: 200},
    seed: {kind: 'number', integer: true, default: 1},
    continuousStep: {kind: 'number', min: 0, default: 1},
    integerStep: {kind: 'number', min: 0, default: 1},
    penalty: {kind: 'number', min: 0, default: 1000000},
    allowRepair: {kind: 'boolean', default: true},
  },
  required: [],
};

const feasibilitySchema: ParamSchema = {
  kind: 'object',
  description: 'Check a user candidate for a structured optimization problem and optionally improve it internally.',
  fields: {
    problem: problemSchema,
    candidate: candidateSchema,
    improvement: improvementSchema,
    timeLimitMs: {kind: 'number', min: 0, default: 180000},
    maxTicks: {kind: 'number', integer: true, min: 1},
    checkEveryTicks: {kind: 'number', integer: true, min: 1, default: 1},
  },
  required: ['problem', 'candidate'],
};

function formatNumber(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (Math.abs(x) >= 1e9 || Math.abs(x) < 1e-3 && x !== 0) return x.toExponential(3);
  return x.toFixed(4);
}

function valuesSummary(e: FeasibilityEvaluation): string {
  return Object.entries(e.values).slice(0, 6).map(([k, v]) => `${k}=${formatNumber(v)}`).join('  ');
}

function drawPipeline(row: FeasibilityEvaluation, index: number, result: FeasibilityPipelineResult): Shape[] {
  const nodes = [
    {id: 'source', x: 84, y: 190, label: 'candidate', fill: '#dbeafe'},
    {id: 'domain', x: 230, y: 190, label: 'domain', fill: row.domainViolations.length ? '#fee2e2' : '#dcfce7'},
    {id: 'constraint', x: 385, y: 190, label: 'constraints', fill: row.constraintViolations.length ? '#fee2e2' : '#dcfce7'},
    {id: 'objective', x: 550, y: 190, label: 'objective', fill: '#fef3c7'},
    {id: 'sink', x: 720, y: 190, label: 'sink', fill: row.feasible ? '#dcfce7' : '#fee2e2'},
    {id: 'improver', x: 550, y: 335, label: 'improver', fill: '#ede9fe'},
  ];
  const q = ((index % 12) + 1) / 13;
  const tokenX = 84 + q * (720 - 84);
  const shapes: Shape[] = [
    {kind: 'rect', x: 0, y: 0, w: 900, h: 560, fill: '#f8fafc'},
    {kind: 'text', x: 450, y: 36, text: `Feasibility pipeline  ${result.status}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'line', x1: 142, y1: 190, x2: 184, y2: 190, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: 290, y1: 190, x2: 325, y2: 190, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: 455, y1: 190, x2: 484, y2: 190, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: 620, y1: 190, x2: 655, y2: 190, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: 550, y1: 222, x2: 550, y2: 304, stroke: '#7c3aed', strokeWidth: 2, dasharray: '6,5'},
    {kind: 'line', x1: 505, y1: 335, x2: 230, y2: 222, stroke: '#7c3aed', strokeWidth: 2, dasharray: '6,5', opacity: 0.65},
    {kind: 'circle', x: tokenX, y: 155, r: 9, fill: row.feasible ? '#22c55e' : '#ef4444', stroke: '#ffffff', strokeWidth: 2},
  ];
  for (const n of nodes) {
    shapes.push({kind: 'rect', x: n.x - 58, y: n.y - 28, w: 116, h: 56, fill: n.fill, stroke: '#1e293b', rx: 6});
    shapes.push({kind: 'text', x: n.x, y: n.y + 4, text: n.label, fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
  }
  shapes.push({kind: 'text', x: 78, y: 82, text: `candidate ${row.candidateId}`, fontSize: 14, fill: '#334155'});
  shapes.push({kind: 'text', x: 78, y: 108, text: `objective ${formatNumber(row.objectiveValue)}`, fontSize: 14, fill: '#334155'});
  shapes.push({kind: 'text', x: 78, y: 134, text: `violation total ${formatNumber(row.totalViolation)} max ${formatNumber(row.maxViolation)}`, fontSize: 14, fill: row.feasible ? '#047857' : '#b91c1c'});
  shapes.push({kind: 'text', x: 78, y: 440, text: valuesSummary(row), fontSize: 14, fill: '#0f172a'});
  if (result.wallClock.expired) {
    shapes.push({kind: 'text', x: 720, y: 335, text: 'time stop', fontSize: 14, anchor: 'middle', fontWeight: 'bold', fill: '#b91c1c'});
  }
  return shapes;
}

const adapter: DESModelRegistration<FeasibilityPipelineParams, FeasibilityPipelineResult> = {
  id: 'feasibility-pipeline',
  description: 'General optimization feasibility checker and internal improvement pipeline.',
  schema: feasibilitySchema,

  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'feasibility-pipeline-start', level: 'info', variables: params.problem.variables.length, constraints: params.problem.constraints?.length ?? 0});
      const result = runFeasibilityPipeline(params);
      const stride = Math.max(1, Math.floor(result.trace.length / 50));
      for (let i = 0; i < result.trace.length; i += stride) {
        const row = result.trace[i];
        logger?.log({kind: 'feasibility-pipeline-trace', level: 'debug', candidateId: row.candidateId, objective: row.objectiveValue, feasible: row.feasible, totalViolation: row.totalViolation});
      }
      logger?.log({kind: 'feasibility-pipeline-finish', level: 'info', status: result.status, bestCandidate: result.best.candidateId, feasible: result.best.feasible, objective: result.best.objectiveValue, totalViolation: result.best.totalViolation});
      return result;
    });
  },

  summarize(result) {
    return [
      'FEASIBILITY PIPELINE',
      '------------------------',
      `  status=${result.status} trace=${result.trace.length} improvements=${result.improvements.length}`,
      `  initial feasible=${result.initial.feasible} objective=${formatNumber(result.initial.objectiveValue)} violation=${formatNumber(result.initial.totalViolation)}`,
      `  best    feasible=${result.best.feasible} objective=${formatNumber(result.best.objectiveValue)} violation=${formatNumber(result.best.totalViolation)} candidate=${result.best.candidateId}`,
      `  wall-clock=${formatNumber(result.wallClock.elapsedMs)} / ${formatNumber(result.wallClock.budgetMs)} ms checks=${result.wallClock.checks}`,
      `  network stationary=${result.network.stationaryEntities.length} moving=${result.network.movingEntities.length} edges=${result.network.edges.length}`,
      `  validation: ${validationLine(result.validation)}`,
      `  values: ${valuesSummary(result.best)}`,
    ].join('\n');
  },

  writeCsv(result, csvPath) {
    const lines = ['candidate_id,parent_id,iteration,origin,objective,comparable_objective,feasible,total_violation,max_violation,values,violations'];
    for (const row of result.trace) {
      lines.push(csvRow([row.candidateId, row.parentId ?? '', row.iteration, row.origin, row.objectiveValue, row.comparableObjective, row.feasible, row.totalViolation, row.maxViolation, row.values, row.violations]));
    }
    writeCsvLines(csvPath, lines);
  },

  async animate(result, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'feasibility-pipeline');
    const rows = result.trace.length > 0 ? result.trace : [result.best];
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 900,
      height: 560,
      fps: 8,
      title: 'Feasibility checker pipeline',
      subtitle: 'Candidate tokens move through domain, constraint, objective, and improvement stations',
      recordEveryTicks: Math.max(1, Math.ceil(rows.length / 250)),
    });
    rows.forEach((row, i) => {
      rec.frame(row.iteration, i, () => ({
        shapes: drawPipeline(row, i, result),
        caption: `${row.candidateId}: feasible=${row.feasible} objective=${formatNumber(row.objectiveValue)} violation=${formatNumber(row.totalViolation)}`,
      }));
    });
    rec.setCharts([
      {x: 70, y: 460, w: 360, h: 74, title: 'Objective', series: [{label: 'objective', color: '#2563eb', t: rows.map((_, i) => i), y: rows.map(r => r.objectiveValue)}]},
      {x: 480, y: 460, w: 350, h: 74, title: 'Violation', series: [{label: 'total violation', color: '#dc2626', t: rows.map((_, i) => i), y: rows.map(r => r.totalViolation)}]},
    ]);
    await rec.finish();
  },

  examples: [
    {
      name: 'repair and improve a binary knapsack candidate',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'feasibility-pipeline',
        parameters: {
          problem: {
            sense: 'max',
            variables: [
              {name: 'x0', type: 'binary'},
              {name: 'x1', type: 'binary'},
              {name: 'x2', type: 'binary'},
            ],
            objective: {coefficients: {x0: 60, x1: 100, x2: 120}},
            constraints: [{name: 'capacity', coefficients: {x0: 10, x1: 20, x2: 30}, sense: '<=', rhs: 50}],
          },
          candidate: {values: {x0: 1, x1: 1, x2: 0}},
          improvement: {enabled: true, maxIterations: 60, seed: 4, integerStep: 1},
        },
        runtime: {animate: true},
      },
    },
  ],
};

registerModel(adapter);
