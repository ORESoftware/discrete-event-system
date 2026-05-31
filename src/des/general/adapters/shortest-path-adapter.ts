// RUST MIGRATION: Target module `src/des/general/adapters/shortest_path_adapter.rs`.
// RUST MIGRATION: Convert shortest-path adapter registration and builtin graph handling into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map graph edges, source/target params, paths, and metrics to `serde` config/result structs; output paths become `PathBuf`.
// RUST MIGRATION: Use `Result<_, ValidationError>` for malformed graph, missing node, and negative/invalid weight errors.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/shortest-path-adapter.rs
//   (module des::general::adapters::shortest_path_adapter)
// 1:1 file move. JSON adapter registering the Bellman-Ford / Dijkstra DES solver.
//
// Declarations → Rust:
//   interface SPParams                         -> struct (#[derive(Deserialize)];
//             graph?/randomGraph?/builtin? -> Option fields; nested graph -> struct)
//   const spSchema: ParamSchema                -> serde + validator metadata
//   const adapter: DESModelRegistration<SPParams, SPResult> -> struct + impl trait
//   registerModel(adapter)                     -> explicit registration call
//
// Conversion notes (file-specific):
//   - `algorithm: 'bellman-ford' | 'dijkstra'` -> enum; dispatch via match (not ternary).
//   - GotChA: distances use JS `Infinity` for unreachable + `Number.isFinite` checks
//     -> f64::INFINITY / .is_finite() (or model as Option<f64>).
//   - `throw new Error` when none of {builtin, graph, randomGraph} given -> Result/validation.
//   - Graph.edges is `Array<Array<{to,weight}>>` (adjacency list) -> Vec<Vec<Edge>>.
// =============================================================================

// =============================================================================
// general/adapters/shortest-path-adapter.ts — JSON adapter for the
// Bellman-Ford-DES / Dijkstra-DES shortest-path solver.
//
// Demonstrates that the registry can host a wholly different model
// without changes to the registry itself — the adapter pattern scales
// to as many models as you want to register.
// =============================================================================

import {
  Graph, SPResult,
  shortestPathBellmanFordDES, shortestPathDijkstraDES,
  buildRandomGraph, buildSmallChainGraph,
} from '../shortest-path-des';
import {
  DESModelRegistration, ParamSchema, DESRuntimeConfig,
} from '../des-spec';
import {registerModel} from '../des-registry';
import {writeCsvLines} from './adapter-utils';

interface SPParams {
  algorithm: 'bellman-ford' | 'dijkstra';
  source: number;
  /** Either an explicit graph or a random-graph spec — exactly one of these. */
  graph?: {
    numNodes: number;
    edges: Array<Array<{to: number; weight: number}>>;
    coordinates?: Array<[number, number]>;
    nodeNames?: string[];
  };
  randomGraph?: {
    numNodes: number;
    edgeProb: number;
    wMin: number;
    wMax: number;
    seed: number;
  };
  builtin?: 'small-chain';
}

const spSchema: ParamSchema = {
  kind: 'object',
  description: 'Shortest path on a directed graph using a DES wave-propagation solver.',
  fields: {
    algorithm: {kind: 'string', enum: ['bellman-ford', 'dijkstra'], description: 'Which DES variant to run'},
    source: {kind: 'number', integer: true, min: 0, description: 'Source node id'},
    graph: {kind: 'object', fields: {
      numNodes: {kind: 'number', integer: true, min: 1},
      edges: {kind: 'array', items: {kind: 'array', items: {kind: 'object', fields: {
        to: {kind: 'number', integer: true, min: 0},
        weight: {kind: 'number'},
      }, required: ['to', 'weight']}}},
    }, required: []},
    randomGraph: {kind: 'object', fields: {
      numNodes: {kind: 'number', integer: true, min: 2, max: 1000},
      edgeProb: {kind: 'number', min: 0, max: 1},
      wMin: {kind: 'number'},
      wMax: {kind: 'number'},
      seed: {kind: 'number', integer: true},
    }, required: []},
    builtin: {kind: 'string', enum: ['small-chain']},
  },
  required: ['algorithm', 'source'],
};

const adapter: DESModelRegistration<SPParams, SPResult> = {
  id: 'shortest-path',
  description: 'Shortest path solved by DES wave-propagation (Bellman-Ford or Dijkstra).',
  schema: spSchema,

  run(params: SPParams) {
    let g: Graph;
    if (params.builtin === 'small-chain') g = buildSmallChainGraph();
    else if (params.randomGraph) g = buildRandomGraph(
      params.randomGraph.numNodes, params.randomGraph.edgeProb,
      params.randomGraph.wMin, params.randomGraph.wMax, params.randomGraph.seed);
    else if (params.graph) g = {numNodes: params.graph.numNodes, edges: params.graph.edges,
                                coordinates: params.graph.coordinates, nodeNames: params.graph.nodeNames};
    else throw new Error('shortest-path: provide one of {builtin, graph, randomGraph}');

    const r = params.algorithm === 'bellman-ford'
      ? shortestPathBellmanFordDES(g, params.source)
      : shortestPathDijkstraDES(g, params.source);
    return r;
  },

  summarize(result: SPResult, params: SPParams): string {
    const reachable = result.distance.filter(d => Number.isFinite(d)).length;
    const lines = [
      'SHORTEST-PATH RUN SUMMARY',
      '──────────────────────────────────',
      `  Algorithm:       ${params.algorithm}`,
      `  Source:          ${params.source}`,
      `  Iterations:      ${result.iterations}`,
      `  Waves emitted:   ${result.wavesEmitted}`,
      `  Reachable nodes: ${reachable} / ${result.distance.length}`,
      `  Negative cycle:  ${result.hasNegativeCycleFromSource ? 'YES (from source)' : 'no'}`,
      '',
      `  Distances (first 12 nodes):  ${result.distance.slice(0, 12).map(d => Number.isFinite(d) ? d.toFixed(2) : '∞').join(', ')}`,
    ];
    return lines.join('\n');
  },

  writeCsv(result: SPResult, csvPath: string): void {
    const lines = ['node,distance,predecessor'];
    for (let v = 0; v < result.distance.length; v++) {
      lines.push(`${v},${Number.isFinite(result.distance[v]) ? result.distance[v].toFixed(6) : 'inf'},${result.predecessor[v]}`);
    }
    writeCsvLines(csvPath, lines);
  },
};

registerModel(adapter);
