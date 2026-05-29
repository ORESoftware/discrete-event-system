'use strict';

// =============================================================================
// general/adapters/simulated-annealing-adapter.ts — JSON adapter for the
// simulated-annealing solver. Supports built-in TSP and knapsack problem
// adapters; users can extend by writing TS subclasses for other state types.
// =============================================================================

import {
  runSimulatedAnnealing, buildTSPSAProblem, buildKnapsackSAProblem,
  CoolingSchedule, SAResult,
} from '../simulated-annealing';
import {buildRandomTSP, buildPentagonTSP, TSPInstance, tourLength} from '../genetic-tsp';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {writeCsvLines} from './adapter-utils';

interface SAParams {
  /** Which built-in problem adapter to run. */
  problem: 'tsp' | 'knapsack';

  tsp?: {
    /** Either a built-in instance or an explicit one. */
    builtin?: 'pentagon' | 'random';
    n?: number;
    seed?: number;
    /** Explicit coordinates and distance matrix (skip if builtin used). */
    coordinates?: Array<[number, number]>;
    distance?: number[][];
    precedence?: Array<[number, number]>;
    init?: 'random' | 'nearest-neighbor';
    moves?: '2-opt' | 'or-opt' | 'mixed';
    penaltyPerViolation?: number;
  };

  knapsack?: {
    values: number[];
    weights: number[];
    capacity: number;
  };

  cooling: CoolingSchedule;

  options: {
    maxIterations: number;
    seed?: number;
    stallLimit?: number;
    recordTrace?: boolean;
    traceStride?: number;
  };
}

const coolingSchema: ParamSchema = {
  kind: 'oneOf',
  variants: [
    {tag: 'geometric', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['geometric']},
      T0: {kind: 'number', min: 0},
      alpha: {kind: 'number', min: 0, max: 1},
      Tmin: {kind: 'number', min: 0},
    }, required: ['kind', 'T0', 'alpha']}},
    {tag: 'logarithmic', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['logarithmic']},
      T0: {kind: 'number', min: 0},
      Tmin: {kind: 'number', min: 0},
    }, required: ['kind', 'T0']}},
    {tag: 'linear', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['linear']},
      T0: {kind: 'number', min: 0},
      rate: {kind: 'number', min: 0},
      Tmin: {kind: 'number', min: 0},
    }, required: ['kind', 'T0', 'rate']}},
    {tag: 'exp-restart', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['exp-restart']},
      T0: {kind: 'number', min: 0},
      alpha: {kind: 'number', min: 0, max: 1},
      period: {kind: 'number', integer: true, min: 1},
      Tmin: {kind: 'number', min: 0},
    }, required: ['kind', 'T0', 'alpha', 'period']}},
  ],
};

const saSchema: ParamSchema = {
  kind: 'object',
  description: 'Simulated annealing on a generic combinatorial problem (TSP / knapsack built-in).',
  fields: {
    problem: {kind: 'string', enum: ['tsp', 'knapsack']},
    tsp: {kind: 'object', fields: {
      builtin: {kind: 'string', enum: ['pentagon', 'random']},
      n: {kind: 'number', integer: true, min: 3},
      seed: {kind: 'number', integer: true},
      coordinates: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}}},
      distance: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}}},
      precedence: {kind: 'array', items: {kind: 'array', items: {kind: 'number', integer: true, min: 0}}},
      init: {kind: 'string', enum: ['random', 'nearest-neighbor']},
      moves: {kind: 'string', enum: ['2-opt', 'or-opt', 'mixed']},
      penaltyPerViolation: {kind: 'number', min: 0},
    }, required: []},
    knapsack: {kind: 'object', fields: {
      values: {kind: 'array', items: {kind: 'number'}},
      weights: {kind: 'array', items: {kind: 'number'}},
      capacity: {kind: 'number', min: 0},
    }, required: ['values', 'weights', 'capacity']},
    cooling: coolingSchema,
    options: {kind: 'object', fields: {
      maxIterations: {kind: 'number', integer: true, min: 1},
      seed: {kind: 'number', integer: true},
      stallLimit: {kind: 'number', integer: true, min: 0},
      recordTrace: {kind: 'boolean'},
      traceStride: {kind: 'number', integer: true, min: 1},
    }, required: ['maxIterations']},
  },
  required: ['problem', 'cooling', 'options'],
};

interface SAAdapterResult {
  problem: 'tsp' | 'knapsack';
  raw: SAResult<unknown>;
  /** TSP-specific extras: tour-length and instance metadata. */
  tspExtras?: {tourLength: number; n: number};
  /** Knapsack extras: value, weight. */
  knapExtras?: {value: number; weight: number; capacity: number};
}

const adapter: DESModelRegistration<SAParams, SAAdapterResult> = {
  id: 'simulated-annealing',
  description: 'Simulated annealing on built-in TSP / knapsack problems (extensible to others via TS subclassing).',
  schema: saSchema,

  run(params: SAParams) {
    if (params.problem === 'tsp') {
      let inst: TSPInstance;
      if (params.tsp?.builtin === 'pentagon') {
        inst = buildPentagonTSP(params.tsp.n ?? 5);
      } else if (params.tsp?.builtin === 'random') {
        inst = buildRandomTSP(params.tsp.n ?? 20, params.tsp.seed ?? 42);
      } else if (params.tsp?.coordinates && params.tsp.distance) {
        inst = {
          n: params.tsp.coordinates.length,
          coordinates: params.tsp.coordinates,
          distance: params.tsp.distance,
          precedence: params.tsp.precedence,
        };
      } else {
        throw new Error('simulated-annealing: tsp params must specify builtin or (coordinates + distance)');
      }
      const saP = buildTSPSAProblem(inst, {
        init: params.tsp?.init ?? 'nearest-neighbor',
        moves: params.tsp?.moves ?? 'mixed',
        penaltyPerViolation: params.tsp?.penaltyPerViolation,
      });
      const r = runSimulatedAnnealing(saP, {
        maxIterations: params.options.maxIterations,
        cooling: params.cooling,
        seed: params.options.seed,
        stallLimit: params.options.stallLimit,
        recordTrace: params.options.recordTrace,
        traceStride: params.options.traceStride,
      });
      return {
        problem: 'tsp', raw: r as SAResult<unknown>,
        tspExtras: {tourLength: tourLength(inst, r.bestState as number[]), n: inst.n},
      };
    } else {
      if (!params.knapsack) throw new Error('simulated-annealing: knapsack params required');
      const saP = buildKnapsackSAProblem(params.knapsack);
      const r = runSimulatedAnnealing(saP, {
        maxIterations: params.options.maxIterations,
        cooling: params.cooling,
        seed: params.options.seed,
        stallLimit: params.options.stallLimit,
        recordTrace: params.options.recordTrace,
        traceStride: params.options.traceStride,
      });
      const x = r.bestState as number[];
      let v = 0, w = 0;
      for (let i = 0; i < x.length; i++) { v += params.knapsack.values[i] * x[i]; w += params.knapsack.weights[i] * x[i]; }
      return {
        problem: 'knapsack', raw: r as SAResult<unknown>,
        knapExtras: {value: v, weight: w, capacity: params.knapsack.capacity},
      };
    }
  },

  summarize(result: SAAdapterResult, params: SAParams): string {
    const lines = [
      'SIMULATED-ANNEALING RUN SUMMARY',
      '──────────────────────────────────',
      `  Problem:           ${result.problem}`,
      `  Cooling:           ${params.cooling.kind}`,
      `  Iterations:        ${result.raw.iterations}`,
      `  Accepted:          ${result.raw.acceptedCount}`,
      `  Improvements:      ${result.raw.improveCount}`,
      `  Best cost:         ${result.raw.bestCost.toFixed(4)}`,
      `  Final cost:        ${result.raw.finalCost.toFixed(4)}`,
    ];
    if (result.tspExtras) {
      lines.push(`  Tour length (n=${result.tspExtras.n}):  ${result.tspExtras.tourLength.toFixed(4)}`);
    }
    if (result.knapExtras) {
      lines.push(`  Knapsack value:    ${result.knapExtras.value.toFixed(2)}    weight: ${result.knapExtras.weight.toFixed(2)} / ${result.knapExtras.capacity}`);
    }
    return lines.join('\n');
  },

  writeCsv(result: SAAdapterResult, csvPath: string): void {
    const lines = ['k,T,best_cost,current_cost'];
    const T = result.raw.temperatureHistory;
    const b = result.raw.bestHistory;
    const c = result.raw.currentHistory;
    for (let i = 0; i < b.length; i++) {
      lines.push(`${i},${T[i] ?? ''},${b[i]},${c[i]}`);
    }
    writeCsvLines(csvPath, lines);
  },

  examples: [
    {
      name: 'sa-tsp-random20',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'simulated-annealing',
        description: 'SA on a 20-city random TSP',
        parameters: {
          problem: 'tsp',
          tsp: {builtin: 'random', n: 20, seed: 5, init: 'nearest-neighbor', moves: 'mixed'},
          cooling: {kind: 'geometric', T0: 100, alpha: 0.999},
          options: {maxIterations: 30000, seed: 1},
        },
      },
    },
  ],
};

registerModel(adapter);
