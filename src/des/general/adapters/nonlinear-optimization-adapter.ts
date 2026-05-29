'use strict';

// =============================================================================
// JSON adapters for Newton/quasi-Newton and nonlinear least-squares DES models.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  NonlinearLeastSquaresParams,
  NonlinearLeastSquaresResult,
  UnconstrainedOptParams,
  UnconstrainedOptResult,
  runBFGSRosenbrock,
  runGaussNewtonCurveFit,
  runLevenbergMarquardtCurveFit,
  runNewtonRosenbrock,
} from '../nonlinear-optimization-models';
import {csvRow, writeCsvLines} from './adapter-utils';

const vectorSchema: ParamSchema = {kind: 'array', items: {kind: 'number'}, minLength: 1};
const unconstrainedSchema: ParamSchema = {
  kind: 'object',
  fields: {
    x0: vectorSchema,
    maxIter: {kind: 'number', integer: true, min: 1, default: 100},
    tol: {kind: 'number', min: 0, default: 1e-8},
  },
  required: [],
};

registerModel<UnconstrainedOptParams, UnconstrainedOptResult>({
  id: 'newton-rosenbrock',
  description: 'Newton minimization of Rosenbrock through movable optimization state tokens.',
  schema: unconstrainedSchema,
  run(params) { return runNewtonRosenbrock(params); },
  summarize(result) { return unconstrainedSummary('NEWTON ROSENBROCK (DES)', result); },
  writeCsv(result, csvPath) { writeUnconstrainedCsv(result, csvPath); },
  examples: [{
    name: 'rosenbrock',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'newton-rosenbrock',
      description: 'Newton method on Rosenbrock as a DES state-token loop.',
      parameters: {x0: [-1.2, 1], maxIter: 50, tol: 1e-8},
    },
  }],
});

registerModel<UnconstrainedOptParams, UnconstrainedOptResult>({
  id: 'bfgs-rosenbrock',
  description: 'BFGS quasi-Newton minimization of Rosenbrock through movable optimization state tokens.',
  schema: unconstrainedSchema,
  run(params) { return runBFGSRosenbrock(params); },
  summarize(result) { return unconstrainedSummary('BFGS ROSENBROCK (DES)', result); },
  writeCsv(result, csvPath) { writeUnconstrainedCsv(result, csvPath); },
  examples: [{
    name: 'rosenbrock',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'bfgs-rosenbrock',
      description: 'BFGS on Rosenbrock as a DES state-token loop.',
      parameters: {x0: [-1.2, 1], maxIter: 100, tol: 1e-6},
    },
  }],
});

const curvePointSchema: ParamSchema = {
  kind: 'object',
  fields: {x: {kind: 'number'}, y: {kind: 'number'}},
  required: ['x', 'y'],
};

const nlsSchema: ParamSchema = {
  kind: 'object',
  fields: {
    points: {kind: 'array', items: curvePointSchema, minLength: 2},
    initial: vectorSchema,
    maxIter: {kind: 'number', integer: true, min: 1, default: 30},
    tol: {kind: 'number', min: 0, default: 1e-8},
    lambda: {kind: 'number', min: 0, default: 0.1},
  },
  required: [],
};

registerModel<NonlinearLeastSquaresParams, NonlinearLeastSquaresResult>({
  id: 'gauss-newton-curve-fit',
  description: 'Nonlinear exponential curve fitting with Gauss-Newton state-token updates.',
  schema: nlsSchema,
  run(params) { return runGaussNewtonCurveFit(params); },
  summarize(result) { return nlsSummary('GAUSS-NEWTON CURVE FIT (DES)', result); },
  writeCsv(result, csvPath) { writeNLSCsv(result, csvPath); },
  examples: [{
    name: 'exp-decay',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'gauss-newton-curve-fit',
      description: 'Fit y = a exp(bx) with Gauss-Newton state tokens.',
      parameters: {initial: [1, -0.2], maxIter: 20},
    },
  }],
});

registerModel<NonlinearLeastSquaresParams, NonlinearLeastSquaresResult>({
  id: 'levenberg-marquardt-curve-fit',
  description: 'Nonlinear exponential curve fitting with Levenberg-Marquardt damped state-token updates.',
  schema: nlsSchema,
  run(params) { return runLevenbergMarquardtCurveFit(params); },
  summarize(result) { return nlsSummary('LEVENBERG-MARQUARDT CURVE FIT (DES)', result); },
  writeCsv(result, csvPath) { writeNLSCsv(result, csvPath); },
  examples: [{
    name: 'exp-decay',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'levenberg-marquardt-curve-fit',
      description: 'Fit y = a exp(bx) with LM damped state tokens.',
      parameters: {initial: [1, -0.2], lambda: 0.1, maxIter: 30},
    },
  }],
});

function unconstrainedSummary(title: string, result: UnconstrainedOptResult): string {
  return [
    title,
    '----------------------------------------',
    `  Objective:      ${result.objective.toExponential(3)}`,
    `  x*:             [${result.x.map(v => v.toFixed(6)).join(', ')}]`,
    `  Gradient norm:  ${result.gradientNorm.toExponential(3)}`,
    `  Iterations:     ${result.iterations}`,
    `  Stations:       ${result.topology.stations.join(' -> ')}`,
    `  Movables:       ${result.topology.movables.join(', ')}`,
  ].join('\n');
}

function nlsSummary(title: string, result: NonlinearLeastSquaresResult): string {
  return [
    title,
    '----------------------------------------',
    `  SSE:            ${result.sse.toExponential(3)}`,
    `  Params:         [${result.params.map(v => v.toFixed(6)).join(', ')}]`,
    `  Gradient norm:  ${result.gradientNorm.toExponential(3)}`,
    `  Iterations:     ${result.iterations}`,
    `  Stations:       ${result.topology.stations.join(' -> ')}`,
    `  Movables:       ${result.topology.movables.join(', ')}`,
  ].join('\n');
}

function writeUnconstrainedCsv(result: UnconstrainedOptResult, csvPath: string): void {
  const lines = [csvRow(['iter', 'objective', 'gradient_norm', 'x'])];
  for (const row of result.trace) lines.push(csvRow([row.iter, row.objective, row.gradientNorm, JSON.stringify(row.x)]));
  writeCsvLines(csvPath, lines);
}

function writeNLSCsv(result: NonlinearLeastSquaresResult, csvPath: string): void {
  const lines = [csvRow(['iter', 'sse', 'gradient_norm', 'params'])];
  for (const row of result.trace) lines.push(csvRow([row.iter, row.sse, row.gradientNorm, JSON.stringify(row.params)]));
  writeCsvLines(csvPath, lines);
}
