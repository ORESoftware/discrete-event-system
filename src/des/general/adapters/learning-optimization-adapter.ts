'use strict';

// =============================================================================
// JSON adapters for station-graph learning and optimization models.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  BackpropMLPParams,
  GradientTrainingResult,
  LinearRegressionParams,
  LinearRegressionResult,
  LogisticRegressionSGDParams,
  RidgeRegressionParams,
  runBackpropMLPClassifier,
  runLinearRegressionLS,
  runLogisticRegressionSGD,
  runRidgeRegressionLS,
} from '../learning-optimization-models';
import {
  ExpectedSarsaGridParams,
  ExpectedSarsaGridResult,
  PolicyGradientCorridorParams,
  PolicyGradientCorridorResult,
  runExpectedSarsaGridworld,
  runPolicyGradientCorridor,
} from '../rl-learning-models';
import {csvRow, writeCsvLines} from './adapter-utils';

const supervisedSampleSchema: ParamSchema = {
  kind: 'object',
  fields: {
    x: {kind: 'array', items: {kind: 'number'}, minLength: 1},
    y: {kind: 'number'},
  },
  required: ['x', 'y'],
};

const samplesSchema: ParamSchema = {
  kind: 'array',
  items: supervisedSampleSchema,
  minLength: 1,
};

registerModel<LinearRegressionParams, LinearRegressionResult>({
  id: 'linear-regression-ls',
  description: 'Least-squares linear regression as DES sample tokens, normal-equation accumulator, and fit sink.',
  schema: {
    kind: 'object',
    fields: {
      samples: samplesSchema,
      fitIntercept: {kind: 'boolean', default: true},
      ridge: {kind: 'number', min: 0, default: 0},
    },
    required: [],
  },
  run(params) { return runLinearRegressionLS(params); },
  summarize(result) {
    return [
      'LINEAR REGRESSION (least squares DES)',
      '------------------------------------',
      `  Samples:        ${result.sampleCount}`,
      `  Coefficients:   [${result.coefficients.map(v => v.toFixed(6)).join(', ')}]`,
      `  Intercept:      ${result.intercept.toFixed(6)}`,
      `  MSE:            ${result.mse.toExponential(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['sample', 'prediction', 'residual'])];
    result.predictions.forEach((p, i) => lines.push(csvRow([i, p, result.residuals[i]])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'line-y-2x-plus-1',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'linear-regression-ls',
      description: 'Fit y = 2x + 1 through DES sample and fit tokens.',
      parameters: {},
    },
  }],
});

registerModel<RidgeRegressionParams, LinearRegressionResult>({
  id: 'ridge-regression-ls',
  description: 'Ridge-regularized least squares using the shared DES sample and normal-equation stations.',
  schema: {
    kind: 'object',
    fields: {
      samples: samplesSchema,
      fitIntercept: {kind: 'boolean', default: true},
      ridge: {kind: 'number', min: 0, default: 0.1},
    },
    required: [],
  },
  run(params) { return runRidgeRegressionLS(params); },
  summarize(result) {
    return [
      'RIDGE REGRESSION (least squares DES)',
      '------------------------------------',
      `  Samples:        ${result.sampleCount}`,
      `  Coefficients:   [${result.coefficients.map(v => v.toFixed(6)).join(', ')}]`,
      `  Intercept:      ${result.intercept.toFixed(6)}`,
      `  MSE:            ${result.mse.toExponential(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['sample', 'prediction', 'residual'])];
    result.predictions.forEach((p, i) => lines.push(csvRow([i, p, result.residuals[i]])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'regularized-line-fit',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'ridge-regression-ls',
      description: 'Fit a regularized line through the shared DES least-squares station graph.',
      parameters: {ridge: 0.1},
    },
  }],
});

registerModel<LogisticRegressionSGDParams, GradientTrainingResult>({
  id: 'logistic-regression-sgd',
  description: 'Binary logistic regression trained by mini-batch gradient tokens and SGD/Adam updates.',
  schema: {
    kind: 'object',
    fields: {
      samples: samplesSchema,
      epochs: {kind: 'number', integer: true, min: 1, default: 120},
      batchSize: {kind: 'number', integer: true, min: 1, default: 4},
      learningRate: {kind: 'number', min: 1e-9, default: 0.2},
      optimizer: {kind: 'string', enum: ['sgd', 'adam'], default: 'sgd'},
      l2: {kind: 'number', min: 0, default: 0},
    },
    required: [],
  },
  run(params) { return runLogisticRegressionSGD(params); },
  summarize(result) {
    return gradientSummary('LOGISTIC REGRESSION (mini-batch gradient DES)', result);
  },
  writeCsv(result, csvPath) { writeGradientCsv(result, csvPath); },
  examples: [{
    name: 'separable-points',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'logistic-regression-sgd',
      description: 'Train a binary linear classifier through sample, batch, and gradient-update stations.',
      parameters: {epochs: 120, batchSize: 3, learningRate: 0.2},
    },
  }],
});

registerModel<BackpropMLPParams, GradientTrainingResult>({
  id: 'backprop-mlp-classifier',
  description: 'One-hidden-layer MLP trained by explicit backprop gradient tokens over mini-batches.',
  schema: {
    kind: 'object',
    fields: {
      samples: samplesSchema,
      hiddenUnits: {kind: 'number', integer: true, min: 1, default: 4},
      epochs: {kind: 'number', integer: true, min: 1, default: 800},
      batchSize: {kind: 'number', integer: true, min: 1, default: 4},
      learningRate: {kind: 'number', min: 1e-9, default: 0.08},
      optimizer: {kind: 'string', enum: ['sgd', 'adam'], default: 'adam'},
      seed: {kind: 'number', integer: true, default: 7},
    },
    required: [],
  },
  run(params) { return runBackpropMLPClassifier(params); },
  summarize(result) {
    return gradientSummary('BACKPROP MLP CLASSIFIER (DES)', result);
  },
  writeCsv(result, csvPath) { writeGradientCsv(result, csvPath); },
  examples: [{
    name: 'xor',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'backprop-mlp-classifier',
      description: 'Train XOR with explicit mini-batch and backprop update stations.',
      parameters: {hiddenUnits: 4, epochs: 800, batchSize: 4, learningRate: 0.08, optimizer: 'adam', seed: 7},
    },
  }],
});

registerModel<PolicyGradientCorridorParams, PolicyGradientCorridorResult>({
  id: 'policy-gradient-corridor',
  description: 'REINFORCE-style softmax policy-gradient agent on a corridor environment with train/resume tokens.',
  schema: {
    kind: 'object',
    fields: {
      numEpisodes: {kind: 'number', integer: true, min: 1, default: 300},
      maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 40},
      rolloutLen: {kind: 'number', integer: true, min: 1, default: 12},
      alpha: {kind: 'number', min: 1e-9, default: 0.04},
      gamma: {kind: 'number', min: 0, max: 1, default: 0.95},
      seed: {kind: 'number', integer: true, default: 1},
      length: {kind: 'number', integer: true, min: 2, default: 7},
    },
    required: [],
  },
  run(params) { return runPolicyGradientCorridor(params); },
  summarize(result) {
    return rlSummary('POLICY GRADIENT CORRIDOR (DES)', result.rewardHistory.length, result.greedySuccessRate, result.greedyMeanLength, result.updates, result.topology);
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['episode', 'reward', 'length'])];
    result.rewardHistory.forEach((r, i) => lines.push(csvRow([i, r, result.lengthHistory[i]])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'corridor',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'policy-gradient-corridor',
      description: 'Softmax policy-gradient agent connected to a corridor environment through DES action/transition tokens.',
      parameters: {numEpisodes: 300, rolloutLen: 12, alpha: 0.04, gamma: 0.95, seed: 1, length: 7},
    },
  }],
});

registerModel<ExpectedSarsaGridParams, ExpectedSarsaGridResult>({
  id: 'expected-sarsa-gridworld',
  description: 'Expected SARSA control on GridWorld using environment state/action/transition tokens.',
  schema: {
    kind: 'object',
    fields: {
      numEpisodes: {kind: 'number', integer: true, min: 1, default: 900},
      maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 80},
      alpha: {kind: 'number', min: 1e-9, default: 0.2},
      gamma: {kind: 'number', min: 0, max: 1, default: 0.95},
      epsilon: {kind: 'number', min: 0, max: 1, default: 0.35},
      epsilonDecay: {kind: 'number', min: 0, max: 1, default: 0.995},
      epsilonMin: {kind: 'number', min: 0, max: 1, default: 0.02},
      seed: {kind: 'number', integer: true, default: 1},
    },
    required: [],
  },
  run(params) { return runExpectedSarsaGridworld(params); },
  summarize(result) {
    return [
      'EXPECTED SARSA GRIDWORLD (DES)',
      '----------------------------------------',
      `  Episodes:       ${result.rewardHistory.length}`,
      `  Greedy reaches: ${result.greedyReached ? 'yes' : 'no'} in ${result.greedyLen} steps`,
      `  Q(start):       [${result.qStart.map(v => v.toFixed(3)).join(', ')}]`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['episode', 'reward', 'length'])];
    result.rewardHistory.forEach((r, i) => lines.push(csvRow([i, r, result.lengthHistory[i]])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'gridworld',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'expected-sarsa-gridworld',
      description: 'Expected SARSA learns GridWorld through DES state/action/transition movables.',
      parameters: {numEpisodes: 900, alpha: 0.2, gamma: 0.95, epsilon: 0.35, epsilonDecay: 0.995, seed: 1},
    },
  }],
});

function gradientSummary(title: string, result: GradientTrainingResult): string {
  return [
    title,
    '----------------------------------------',
    `  Steps:          ${result.lossHistory.length}`,
    `  Final loss:     ${result.finalLoss.toFixed(6)}`,
    `  Accuracy:       ${(100 * result.accuracy).toFixed(1)}%`,
    `  Bias:           ${result.bias.toFixed(6)}`,
    `  Weights:        [${result.weights.slice(0, 8).map(v => v.toFixed(4)).join(', ')}${result.weights.length > 8 ? ', ...' : ''}]`,
    `  Stations:       ${result.topology.stations.join(' -> ')}`,
    `  Movables:       ${result.topology.movables.join(', ')}`,
  ].join('\n');
}

function writeGradientCsv(result: GradientTrainingResult, csvPath: string): void {
  const lines = [csvRow(['step', 'loss', 'gradient_norm'])];
  result.lossHistory.forEach((loss, i) => lines.push(csvRow([i + 1, loss, result.gradientNormHistory[i]])));
  writeCsvLines(csvPath, lines);
}

function rlSummary(
  title: string,
  episodes: number,
  successRate: number,
  meanLength: number,
  updates: number,
  topology: {stations: string[]; movables: string[]},
): string {
  return [
    title,
    '----------------------------------------',
    `  Episodes:       ${episodes}`,
    `  Greedy success: ${(100 * successRate).toFixed(1)}%`,
    `  Greedy length:  ${meanLength.toFixed(2)}`,
    `  Updates:        ${updates}`,
    `  Stations:       ${topology.stations.join(' -> ')}`,
    `  Movables:       ${topology.movables.join(', ')}`,
  ].join('\n');
}
