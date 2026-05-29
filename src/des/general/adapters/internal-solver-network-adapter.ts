'use strict';

// =============================================================================
// JSON adapter for internal solver networks.
//
// This adapter exposes GA, simulated annealing, dynamic-programming knapsack,
// shortest path, and exact small TSP as DES station networks. It also records
// the solver/checker/sink graph as animation frames whenever runtime animation
// is enabled, which is the registry default for models with an animator.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {framesPath, jsonCsvRow as csvRow, validationLine, withLogger, writeCsvLines} from './adapter-utils';
import {
  InternalSolverRunParams,
  InternalSolverRunResult,
  SolverProgressPayload,
  runInternalSolverNetwork,
} from '../internal-solver-network';

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

const graphEdgeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    to: {kind: 'number', integer: true, min: 0},
    weight: {kind: 'number'},
  },
  required: ['to', 'weight'],
};

const graphSchema: ParamSchema = {
  kind: 'object',
  fields: {
    numNodes: {kind: 'number', integer: true, min: 1},
    edges: {kind: 'array', items: {kind: 'array', items: graphEdgeSchema}},
    coordinates: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2}},
    nodeNames: {kind: 'array', items: {kind: 'string'}},
  },
  required: ['numNodes', 'edges'],
};

const shortestPathSchema: ParamSchema = {
  kind: 'object',
  fields: {
    algorithm: {kind: 'string', enum: ['bellman-ford', 'dijkstra'], default: 'dijkstra'},
    source: {kind: 'number', integer: true, min: 0, default: 0},
    builtin: {kind: 'string', enum: ['small-chain']},
    graph: graphSchema,
    randomGraph: {kind: 'object', fields: {
      numNodes: {kind: 'number', integer: true, min: 2, max: 100000},
      edgeProb: {kind: 'number', min: 0, max: 1},
      wMin: {kind: 'number'},
      wMax: {kind: 'number'},
      seed: {kind: 'number', integer: true},
    }, required: ['numNodes', 'edgeProb', 'wMin', 'wMax', 'seed']},
  },
  required: ['algorithm', 'source'],
};

const knapsackSchema: ParamSchema = {
  kind: 'object',
  fields: {
    values: {kind: 'array', items: {kind: 'number'}, minLength: 1},
    weights: {kind: 'array', items: {kind: 'number'}, minLength: 1},
    capacity: {kind: 'number', integer: true, min: 0},
    seed: {kind: 'number', integer: true, default: 1},
    maxIterations: {kind: 'number', integer: true, min: 1, default: 5000},
    cooling: coolingSchema,
    stallLimit: {kind: 'number', integer: true, min: 0, default: 0},
    penalty: {kind: 'number', min: 0, default: 1000000},
  },
  required: ['values', 'weights', 'capacity'],
};

const tspSASchema: ParamSchema = {
  kind: 'object',
  fields: {
    cooling: coolingSchema,
    maxIterations: {kind: 'number', integer: true, min: 1, default: 5000},
    seed: {kind: 'number', integer: true, default: 1},
    init: {kind: 'string', enum: ['random', 'nearest-neighbor'], default: 'nearest-neighbor'},
    moves: {kind: 'string', enum: ['2-opt', 'or-opt', 'mixed'], default: 'mixed'},
    penaltyPerViolation: {kind: 'number', min: 0, default: 1000000},
    traceStride: {kind: 'number', integer: true, min: 1},
    stallLimit: {kind: 'number', integer: true, min: 0, default: 0},
  },
  required: ['maxIterations', 'seed'],
};

const tspGASchema: ParamSchema = {
  kind: 'object',
  fields: {
    popSize: {kind: 'number', integer: true, min: 2, default: 60},
    numGenerations: {kind: 'number', integer: true, min: 1, default: 200},
    tournamentSize: {kind: 'number', integer: true, min: 1, default: 3},
    crossoverProb: {kind: 'number', min: 0, max: 1, default: 0.95},
    mutationProb: {kind: 'number', min: 0, max: 1, default: 0.3},
    elitism: {kind: 'number', integer: true, min: 0, default: 2},
    seed: {kind: 'number', integer: true, default: 1},
    init: {kind: 'string', enum: ['random', 'nearest-neighbor'], default: 'nearest-neighbor'},
    penaltyPerViolation: {kind: 'number', min: 0, default: 1000000},
  },
  required: ['popSize', 'numGenerations', 'seed'],
};

const tspSchema: ParamSchema = {
  kind: 'object',
  fields: {
    builtin: {kind: 'string', enum: ['pentagon', 'random'], default: 'pentagon'},
    n: {kind: 'number', integer: true, min: 3, default: 5},
    seed: {kind: 'number', integer: true, default: 1},
    coordinates: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2}},
    distance: {kind: 'array', items: {kind: 'array', items: {kind: 'number'}}},
    precedence: {kind: 'array', items: {kind: 'array', items: {kind: 'number', integer: true, min: 0}, minLength: 2, maxLength: 2}},
    sa: tspSASchema,
    ga: tspGASchema,
  },
  required: [],
};

const internalSolverSchema: ParamSchema = {
  kind: 'object',
  description: 'Internal optimization/search solvers represented as DES station networks.',
  fields: {
    kind: {kind: 'string', enum: ['shortest-path', 'knapsack-dp', 'knapsack-sa', 'tsp-sa', 'tsp-ga', 'tsp-held-karp']},
    timeLimitMs: {kind: 'number', min: 0, default: 180000},
    maxTicks: {kind: 'number', integer: true, min: 1},
    checkEveryTicks: {kind: 'number', integer: true, min: 1, default: 1},
    shortestPath: shortestPathSchema,
    knapsack: knapsackSchema,
    tsp: tspSchema,
  },
  required: ['kind'],
};

function formatNumber(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (Math.abs(x) >= 1e9 || Math.abs(x) < 1e-3 && x !== 0) return x.toExponential(3);
  return x.toFixed(4);
}

function summarizeBestState(row: SolverProgressPayload): string {
  const state = row.bestState as Record<string, unknown>;
  if (state && Array.isArray(state.distance)) {
    const distances = (state.distance as unknown[]).slice(0, 8).map(v => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : 'inf');
    return `dist=[${distances.join(', ')}${(state.distance as unknown[]).length > 8 ? ', ...' : ''}]`;
  }
  if (state && typeof state.value === 'number' && typeof state.weight === 'number') {
    return `value=${formatNumber(state.value)} weight=${formatNumber(state.weight)}/${formatNumber(state.capacity as number)}`;
  }
  if (state && Array.isArray(state.tour) && typeof state.length === 'number') {
    return `length=${formatNumber(state.length)} tour=[${(state.tour as unknown[]).slice(0, 9).join(' -> ')}${(state.tour as unknown[]).length > 9 ? ' -> ...' : ''}]`;
  }
  return JSON.stringify(row.bestState).slice(0, 120);
}

function drawSolverNetwork(row: SolverProgressPayload, frameIndex: number, result: InternalSolverRunResult): Shape[] {
  const source = {x: 90, y: 185, label: 'source', fill: '#dbeafe'};
  const solver = {x: 275, y: 185, label: row.solverKind, fill: '#ede9fe'};
  const checker = {x: 275, y: 330, label: 'time checker', fill: result.wallClock.expired ? '#fee2e2' : '#dcfce7'};
  const sink = {x: 600, y: 185, label: 'solution sink', fill: '#fef3c7'};
  const q = ((frameIndex % 16) + 1) / 17;
  const sx = solver.x + q * (sink.x - solver.x);
  const sy = solver.y + q * (sink.y - solver.y);
  const elapsed = Math.min(1, result.wallClock.budgetMs === 0 ? 1 : result.wallClock.elapsedMs / result.wallClock.budgetMs);
  const shapes: Shape[] = [
    {kind: 'rect', x: 0, y: 0, w: 900, h: 560, fill: '#f8fafc'},
    {kind: 'text', x: 450, y: 36, text: `${result.kind}  ${result.status}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'line', x1: source.x + 52, y1: source.y, x2: solver.x - 62, y2: solver.y, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: solver.x + 70, y1: solver.y, x2: sink.x - 76, y2: sink.y, stroke: '#64748b', strokeWidth: 2},
    {kind: 'line', x1: checker.x + 76, y1: checker.y - 10, x2: sink.x - 76, y2: sink.y + 24, stroke: '#ef4444', strokeWidth: 2, dasharray: '6,6', opacity: result.wallClock.expired ? 0.9 : 0.25},
    {kind: 'circle', x: sx, y: sy, r: 9, fill: row.done ? '#22c55e' : '#2563eb', stroke: '#ffffff', strokeWidth: 2, title: 'SolverSolutionToken'},
    {kind: 'rect', x: source.x - 58, y: source.y - 26, w: 116, h: 52, fill: source.fill, stroke: '#1e293b', rx: 6},
    {kind: 'rect', x: solver.x - 78, y: solver.y - 32, w: 156, h: 64, fill: solver.fill, stroke: '#1e293b', rx: 6},
    {kind: 'rect', x: checker.x - 84, y: checker.y - 28, w: 168, h: 56, fill: checker.fill, stroke: '#1e293b', rx: 6},
    {kind: 'rect', x: sink.x - 86, y: sink.y - 30, w: 172, h: 60, fill: sink.fill, stroke: '#1e293b', rx: 6},
    {kind: 'text', x: source.x, y: source.y + 4, text: source.label, fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'text', x: solver.x, y: solver.y + 4, text: solver.label, fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'text', x: checker.x, y: checker.y + 4, text: checker.label, fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'text', x: sink.x, y: sink.y + 4, text: sink.label, fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'text', x: 90, y: 82, text: `iteration ${row.iteration}`, fontSize: 15, fill: '#334155'},
    {kind: 'text', x: 90, y: 108, text: `objective ${formatNumber(row.objective)}`, fontSize: 15, fill: '#334155'},
    {kind: 'text', x: 90, y: 134, text: row.feasible ? 'feasible' : 'infeasible', fontSize: 15, fill: row.feasible ? '#047857' : '#b91c1c', fontWeight: 'bold'},
    {kind: 'rect', x: 520, y: 318, w: 260, h: 14, fill: '#e2e8f0', stroke: '#94a3b8', rx: 3},
    {kind: 'rect', x: 520, y: 318, w: 260 * elapsed, h: 14, fill: result.wallClock.expired ? '#ef4444' : '#22c55e', rx: 3},
    {kind: 'text', x: 650, y: 356, text: `wall clock ${formatNumber(result.wallClock.elapsedMs)} / ${formatNumber(result.wallClock.budgetMs)} ms`, fontSize: 13, anchor: 'middle', fill: '#334155'},
    {kind: 'text', x: 90, y: 430, text: summarizeBestState(row), fontSize: 14, fill: '#0f172a'},
  ];
  if (result.wallClock.expired) {
    shapes.push({kind: 'circle', x: 438, y: 270, r: 9, fill: '#ef4444', stroke: '#ffffff', strokeWidth: 2, title: 'StopSignalToken'});
  }
  return shapes;
}

const adapter: DESModelRegistration<InternalSolverRunParams, InternalSolverRunResult> = {
  id: 'internal-solver-network',
  description: 'Internal GA, SA, knapsack, shortest-path, and TSP solvers as DES station/movable networks.',
  schema: internalSolverSchema,

  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'internal-solver-start', level: 'info', solverKind: params.kind, timeLimitMs: params.timeLimitMs ?? 180000});
      const result = runInternalSolverNetwork(params);
      const stride = Math.max(1, Math.floor(result.trace.length / 50));
      for (let i = 0; i < result.trace.length; i += stride) {
        const row = result.trace[i];
        logger?.log({kind: 'internal-solver-trace', level: 'debug', solverKind: row.solverKind, iteration: row.iteration, objective: row.objective, feasible: row.feasible, done: row.done});
      }
      logger?.log({kind: 'internal-solver-finish', level: 'info', solverKind: result.kind, status: result.status, objective: result.best.objective, iterations: result.best.iteration, validationOk: result.runSummary.validationOk ?? true});
      return result;
    });
  },

  summarize(result) {
    return [
      'INTERNAL SOLVER NETWORK',
      '------------------------',
      `  kind=${result.kind} status=${result.status}`,
      `  iterations=${result.best.iteration} ticks=${result.runSummary.ticks} reason=${result.runSummary.reason}`,
      `  objective=${formatNumber(result.best.objective)} feasible=${result.best.feasible} done=${result.best.done}`,
      `  wall-clock=${formatNumber(result.wallClock.elapsedMs)} / ${formatNumber(result.wallClock.budgetMs)} ms checks=${result.wallClock.checks}`,
      `  network stationary=${result.network.stationaryEntities.length} moving=${result.network.movingEntities.length} edges=${result.network.edges.length}`,
      `  validation: ${validationLine(result.validation)}`,
      `  best: ${summarizeBestState(result.best)}`,
    ].join('\n');
  },

  writeCsv(result, csvPath) {
    const lines = ['tick,iteration,solver_kind,objective,feasible,done,best_state,metadata'];
    for (const row of result.trace) {
      lines.push(csvRow([row.tick, row.iteration, row.solverKind, row.objective, row.feasible, row.done, row.bestState, row.metadata ?? {}]));
    }
    writeCsvLines(csvPath, lines);
  },

  async animate(result, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'internal-solver-network');
    const rows = result.trace.length > 0 ? result.trace : [result.best];
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 900,
      height: 560,
      fps: 8,
      title: 'Internal solver network',
      subtitle: 'Solver station emits incumbent tokens; wall-clock checker emits stop tokens',
      recordEveryTicks: Math.max(1, Math.ceil(rows.length / 250)),
    });
    rows.forEach((row, i) => {
      rec.frame(row.iteration, i, () => ({
        shapes: drawSolverNetwork(row, i, result),
        caption: `${row.solverKind}: objective=${formatNumber(row.objective)} ${row.feasible ? 'feasible' : 'infeasible'}`,
      }));
    });
    rec.setCharts([
      {
        x: 70, y: 455, w: 760, h: 80,
        title: 'Objective by solver iteration',
        series: [{label: 'objective', color: '#2563eb', t: rows.map(r => r.iteration), y: rows.map(r => r.objective)}],
      },
    ]);
    await rec.finish();
  },

  examples: [
    {
      name: 'knapsack dynamic programming',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'internal-solver-network',
        parameters: {
          kind: 'knapsack-dp',
          timeLimitMs: 180000,
          knapsack: {
            values: [20, 30, 35, 12, 3],
            weights: [2, 5, 7, 3, 1],
            capacity: 10,
          },
        },
        runtime: {animate: true},
      },
    },
    {
      name: 'tsp genetic algorithm',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'internal-solver-network',
        parameters: {
          kind: 'tsp-ga',
          timeLimitMs: 180000,
          tsp: {
            builtin: 'pentagon',
            n: 8,
            seed: 7,
            ga: {popSize: 40, numGenerations: 80, seed: 11, init: 'nearest-neighbor'},
          },
        },
        runtime: {animate: true},
      },
    },
  ],
};

registerModel(adapter);
