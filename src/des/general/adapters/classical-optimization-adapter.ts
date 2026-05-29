'use strict';

// =============================================================================
// JSON adapters for classic optimization station-graph models.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  AssignmentParams,
  AssignmentResult,
  FlowShopNEHParams,
  FlowShopNEHResult,
  JobShopDispatchParams,
  JobShopDispatchResult,
  QPProjectedGradientParams,
  QPProjectedGradientResult,
  VRPSavingsParams,
  VRPSavingsResult,
  runAuctionAssignment,
  runFlowShopNEH,
  runHungarianAssignment,
  runJobShopDispatch,
  runQPCoordinateDescent,
  runQPProjectedGradient,
  runVRPNearestNeighbor,
  runVRPSavings,
} from '../classical-optimization-models';
import {csvRow, writeCsvLines} from './adapter-utils';

const numberVectorSchema: ParamSchema = {kind: 'array', items: {kind: 'number'}, minLength: 1};
const numberMatrixSchema: ParamSchema = {kind: 'array', items: numberVectorSchema, minLength: 1};

registerModel<QPProjectedGradientParams, QPProjectedGradientResult>({
  id: 'qp-projected-gradient',
  description: 'Box-constrained quadratic programming via projected-gradient state tokens.',
  schema: {
    kind: 'object',
    fields: {
      Q: numberMatrixSchema,
      c: numberVectorSchema,
      lower: numberVectorSchema,
      upper: numberVectorSchema,
      x0: numberVectorSchema,
      stepSize: {kind: 'number', min: 1e-12, default: 0.12},
      maxIter: {kind: 'number', integer: true, min: 1, default: 200},
      tol: {kind: 'number', min: 0, default: 1e-8},
    },
    required: [],
  },
  run(params) { return runQPProjectedGradient(params); },
  summarize(result) {
    return [
      'QP PROJECTED GRADIENT (DES)',
      '---------------------------',
      `  Objective:      ${result.objective.toFixed(8)}`,
      `  x*:             [${result.x.map(v => v.toFixed(6)).join(', ')}]`,
      `  Iterations:     ${result.iterations}`,
      `  Gradient norm:  ${result.gradientNorm.toExponential(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iter', 'objective', 'gradient_norm', 'x'])];
    for (const row of result.trace) lines.push(csvRow([row.iter, row.objective, row.gradientNorm, JSON.stringify(row.x)]));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'box-constrained-q2',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'qp-projected-gradient',
      description: 'Small box-constrained quadratic program solved by movable state tokens.',
      parameters: {},
    },
  }],
});

registerModel<QPProjectedGradientParams, QPProjectedGradientResult>({
  id: 'qp-coordinate-descent',
  description: 'Box-constrained quadratic programming via coordinate-descent state tokens.',
  schema: {
    kind: 'object',
    fields: {
      Q: numberMatrixSchema,
      c: numberVectorSchema,
      lower: numberVectorSchema,
      upper: numberVectorSchema,
      x0: numberVectorSchema,
      maxIter: {kind: 'number', integer: true, min: 1, default: 100},
      tol: {kind: 'number', min: 0, default: 1e-8},
    },
    required: [],
  },
  run(params) { return runQPCoordinateDescent(params); },
  summarize(result) {
    return [
      'QP COORDINATE DESCENT (DES)',
      '---------------------------',
      `  Objective:      ${result.objective.toFixed(8)}`,
      `  x*:             [${result.x.map(v => v.toFixed(6)).join(', ')}]`,
      `  Iterations:     ${result.iterations}`,
      `  Gradient norm:  ${result.gradientNorm.toExponential(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['iter', 'objective', 'gradient_norm', 'x'])];
    for (const row of result.trace) lines.push(csvRow([row.iter, row.objective, row.gradientNorm, JSON.stringify(row.x)]));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'box-constrained-coordinate',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'qp-coordinate-descent',
      description: 'Small box-constrained quadratic program solved by coordinate-descent state tokens.',
      parameters: {},
    },
  }],
});

registerModel<AssignmentParams, AssignmentResult>({
  id: 'hungarian-assignment',
  description: 'Assignment problem with row/column reduction stations and assignment-result tokens.',
  schema: {
    kind: 'object',
    fields: {cost: numberMatrixSchema},
    required: [],
  },
  run(params) { return runHungarianAssignment(params); },
  summarize(result) {
    return [
      'HUNGARIAN-STYLE ASSIGNMENT (DES)',
      '--------------------------------',
      `  Objective:      ${result.objective.toFixed(6)}`,
      `  Assignment:     [${result.assignment.join(', ')}]`,
      `  Row reductions: [${result.rowReductions.map(v => v.toFixed(2)).join(', ')}]`,
      `  Col reductions: [${result.colReductions.map(v => v.toFixed(2)).join(', ')}]`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['worker', 'job', 'objective'])];
    result.assignment.forEach((job, worker) => lines.push(csvRow([worker, job, result.objective])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'three-by-three',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'hungarian-assignment',
      description: '3x3 assignment through row reduction, column reduction, and assignment builder stations.',
      parameters: {},
    },
  }],
});

registerModel<AssignmentParams & {epsilon?: number; maxIter?: number}, AssignmentResult>({
  id: 'auction-assignment',
  description: 'Assignment problem using movable auction price/assignment state tokens.',
  schema: {
    kind: 'object',
    fields: {
      cost: numberMatrixSchema,
      epsilon: {kind: 'number', min: 1e-12, default: 0.01},
      maxIter: {kind: 'number', integer: true, min: 1},
    },
    required: [],
  },
  run(params) { return runAuctionAssignment(params); },
  summarize(result) {
    return [
      'AUCTION ASSIGNMENT (DES)',
      '------------------------',
      `  Objective:      ${result.objective.toFixed(6)}`,
      `  Assignment:     [${result.assignment.join(', ')}]`,
      `  Price vector:   [${result.colReductions.map(v => v.toFixed(3)).join(', ')}]`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['worker', 'job', 'objective'])];
    result.assignment.forEach((job, worker) => lines.push(csvRow([worker, job, result.objective])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'three-by-three-auction',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'auction-assignment',
      description: '3x3 assignment through movable auction state tokens.',
      parameters: {epsilon: 0.01},
    },
  }],
});

const customerSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    x: {kind: 'number'},
    y: {kind: 'number'},
    demand: {kind: 'number', min: 0},
  },
  required: ['id', 'x', 'y', 'demand'],
};

registerModel<VRPSavingsParams, VRPSavingsResult>({
  id: 'vrp-savings',
  description: 'Capacitated vehicle routing with Clarke-Wright savings and route-merge tokens.',
  schema: {
    kind: 'object',
    fields: {
      depot: {kind: 'object', fields: {x: {kind: 'number'}, y: {kind: 'number'}}, required: ['x', 'y']},
      customers: {kind: 'array', items: customerSchema, minLength: 1},
      vehicleCapacity: {kind: 'number', min: 1e-12, default: 5},
    },
    required: [],
  },
  run(params) { return runVRPSavings(params); },
  summarize(result) {
    return [
      'VRP SAVINGS HEURISTIC (DES)',
      '---------------------------',
      `  Routes:         ${result.routes.length}`,
      `  Total distance: ${result.totalDistance.toFixed(6)}`,
      `  Savings pairs:  ${result.savingsConsidered}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['route', 'customers', 'load', 'distance'])];
    result.routes.forEach((r, i) => lines.push(csvRow([i, r.customers.join('|'), r.load, r.distance])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'small-cvrp',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'vrp-savings',
      description: 'Small capacitated VRP using savings and route-merge stations.',
      parameters: {vehicleCapacity: 5},
    },
  }],
});

registerModel<VRPSavingsParams, VRPSavingsResult>({
  id: 'vrp-nearest-neighbor',
  description: 'Capacitated vehicle routing with nearest-neighbor route-construction tokens.',
  schema: {
    kind: 'object',
    fields: {
      depot: {kind: 'object', fields: {x: {kind: 'number'}, y: {kind: 'number'}}, required: ['x', 'y']},
      customers: {kind: 'array', items: customerSchema, minLength: 1},
      vehicleCapacity: {kind: 'number', min: 1e-12, default: 5},
    },
    required: [],
  },
  run(params) { return runVRPNearestNeighbor(params); },
  summarize(result) {
    return [
      'VRP NEAREST NEIGHBOR (DES)',
      '--------------------------',
      `  Routes:         ${result.routes.length}`,
      `  Total distance: ${result.totalDistance.toFixed(6)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['route', 'customers', 'load', 'distance'])];
    result.routes.forEach((r, i) => lines.push(csvRow([i, r.customers.join('|'), r.load, r.distance])));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'small-cvrp-nearest',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'vrp-nearest-neighbor',
      description: 'Small capacitated VRP using a nearest-neighbor route-construction station.',
      parameters: {vehicleCapacity: 5},
    },
  }],
});

const operationSchema: ParamSchema = {
  kind: 'object',
  fields: {
    machine: {kind: 'string'},
    duration: {kind: 'number', min: 0},
  },
  required: ['machine', 'duration'],
};

const jobSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    due: {kind: 'number'},
    operations: {kind: 'array', items: operationSchema, minLength: 1},
  },
  required: ['id', 'operations'],
};

registerModel<JobShopDispatchParams, JobShopDispatchResult>({
  id: 'job-shop-dispatch',
  description: 'Job-shop scheduling via job tokens and a dispatch-rule scheduler station.',
  schema: {
    kind: 'object',
    fields: {
      jobs: {kind: 'array', items: jobSchema, minLength: 1},
      rule: {kind: 'string', enum: ['fifo', 'spt', 'edd'], default: 'spt'},
    },
    required: [],
  },
  run(params) { return runJobShopDispatch(params); },
  summarize(result) {
    return [
      'JOB-SHOP DISPATCH (DES)',
      '-----------------------',
      `  Operations:     ${result.schedule.length}`,
      `  Makespan:       ${result.makespan.toFixed(3)}`,
      `  Total flow:     ${result.totalFlowTime.toFixed(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['job', 'operation', 'machine', 'start', 'finish'])];
    for (const op of result.schedule) lines.push(csvRow([op.jobId, op.opIndex, op.machine, op.start, op.finish]));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'three-job-spt',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'job-shop-dispatch',
      description: 'Three-job two-machine schedule using a shortest-processing-time dispatch station.',
      parameters: {rule: 'spt'},
    },
  }],
});

const flowShopJobSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    processingTimes: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    due: {kind: 'number'},
  },
  required: ['id', 'processingTimes'],
};

registerModel<FlowShopNEHParams, FlowShopNEHResult>({
  id: 'flow-shop-neh',
  description: 'Flow-shop scheduling with NEH sequence tokens and a schedule-builder station.',
  schema: {
    kind: 'object',
    fields: {
      jobs: {kind: 'array', items: flowShopJobSchema, minLength: 1},
    },
    required: [],
  },
  run(params) { return runFlowShopNEH(params); },
  summarize(result) {
    return [
      'FLOW-SHOP NEH (DES)',
      '-------------------',
      `  Sequence:       ${result.sequence.join(' -> ')}`,
      `  Operations:     ${result.schedule.length}`,
      `  Makespan:       ${result.makespan.toFixed(3)}`,
      `  Total flow:     ${result.totalFlowTime.toFixed(3)}`,
      `  Stations:       ${result.topology.stations.join(' -> ')}`,
      `  Movables:       ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['job', 'operation', 'machine', 'start', 'finish'])];
    for (const op of result.schedule) lines.push(csvRow([op.jobId, op.opIndex, op.machine, op.start, op.finish]));
    writeCsvLines(csvPath, lines);
  },
  examples: [{
    name: 'four-job-flow-shop',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'flow-shop-neh',
      description: 'Four-job flow-shop schedule using NEH sequence and schedule stations.',
      parameters: {},
    },
  }],
});
