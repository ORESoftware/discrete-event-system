'use strict';

// =============================================================================
// main-shortest-path.ts — Shortest-path-as-DES on a directed weighted
// graph. Each node IS a stationary entity holding its current best
// distance estimate; each "wave" message IS a movable carrying a
// distance update along an edge.
//
// Two algorithms, both expressed inside the DES:
//   - Bellman-Ford-DES: iterative relaxation, terminates in ≤ |V|-1 ticks
//                       and naturally distributed (every dirty node
//                       broadcasts to its neighbours each tick)
//   - Dijkstra-DES:     priority-queue scheduling of the same fixed-point
//                       computation; one "active" station per tick
//
// USAGE
// ─────
//     node dist/des/main-shortest-path.js                       # 5-node chain
//     N_NODES=10 EDGE_PROB=0.35 node …                          # random graph
//     ALGO=dijkstra node …                                      # Dijkstra mode
//     ALGO=both ANIMATE=1 N_NODES=12 node …                     # both algorithms,
//                                                                #   animation written
// =============================================================================

import * as path from 'path';
import {
  buildSmallChainGraph, buildRandomGraph,
  shortestPathBellmanFordDES, shortestPathDijkstraDES,
  reconstructPath, Graph, SPResult,
} from './general/shortest-path-des';
import {FrameRecorder} from './animation/frame-recorder';
import {STAGE_W, STAGE_H, buildShortestPathFrame, buildShortestPathCharts} from './animation/scenes/shortest-path-scene';

async function main(): Promise<void> {
  const algo = (process.env.ALGO ?? 'bellman-ford') as 'bellman-ford' | 'dijkstra' | 'both';
  const nNodes = Number(process.env.N_NODES ?? 0);
  const edgeProb = Number(process.env.EDGE_PROB ?? 0.35);
  const seed = Number(process.env.SEED ?? 13);
  const source = Number(process.env.SOURCE ?? 0);
  const animate = process.env.ANIMATE === '1';

  const graph: Graph = nNodes > 0
    ? buildRandomGraph(nNodes, edgeProb, 1, 10, seed)
    : buildSmallChainGraph();

  // ── Banner ──
  console.log('# Shortest-path solver as DES (each node is a station, waves are movables)');
  console.log(`# graph: ${graph.numNodes} nodes, source = ${graph.nodeNames?.[source] ?? source}`);
  let edgeCount = 0;
  for (const e of graph.edges) edgeCount += e.length;
  console.log(`# edges: ${edgeCount}`);
  console.log('');

  // ── Run requested algorithm(s) ──
  const runs: Array<{name: string; result: SPResult}> = [];
  if (algo === 'bellman-ford' || algo === 'both') {
    const t0 = Date.now();
    const r = shortestPathBellmanFordDES(graph, source);
    console.log(`# Bellman-Ford-DES finished in ${Date.now() - t0}ms`);
    console.log(`#   iterations  = ${r.iterations}`);
    console.log(`#   waves emitted = ${r.wavesEmitted}`);
    console.log(`#   negative cycle = ${r.hasNegativeCycleFromSource}`);
    runs.push({name: 'bellman-ford', result: r});
  }
  if (algo === 'dijkstra' || algo === 'both') {
    const t0 = Date.now();
    const r = shortestPathDijkstraDES(graph, source);
    console.log(`# Dijkstra-DES       finished in ${Date.now() - t0}ms`);
    console.log(`#   priority-queue pops = ${r.iterations}`);
    console.log(`#   waves emitted       = ${r.wavesEmitted}`);
    runs.push({name: 'dijkstra', result: r});
  }
  console.log('');

  // ── Cross-validate Bellman-Ford and Dijkstra distances if both ran ──
  if (runs.length === 2) {
    let maxDiff = 0;
    for (let v = 0; v < graph.numNodes; v++) {
      const a = runs[0].result.distance[v];
      const b = runs[1].result.distance[v];
      if (isFinite(a) && isFinite(b)) maxDiff = Math.max(maxDiff, Math.abs(a - b));
      else if (a !== b) maxDiff = Infinity;
    }
    console.log(`# Bellman-Ford vs Dijkstra:  max |Δ distance| = ${maxDiff.toExponential(2)}`);
    console.log('');
  }

  // ── Per-node distance + path report from the first run ──
  const r = runs[0].result;
  console.log(`# Distances from source ${graph.nodeNames?.[source] ?? source}:`);
  for (let v = 0; v < graph.numNodes; v++) {
    const name = graph.nodeNames?.[v] ?? String(v);
    if (!isFinite(r.distance[v])) {
      console.log(`#   ${name.padEnd(6)} d = ∞       (unreachable)`);
    } else {
      const path = reconstructPath(r, source, v);
      const pathStr = path?.map(p => graph.nodeNames?.[p] ?? p).join(' → ') ?? '-';
      console.log(`#   ${name.padEnd(6)} d = ${r.distance[v].toFixed(2).padStart(6)}   path: ${pathStr}`);
    }
  }
  console.log('');

  // ── Animation ──
  if (animate) {
    const targetRun = runs[runs.length - 1];        // animate the last (so 'both' shows Dijkstra)
    const outDir = path.join(__dirname, '..', '..', 'out');
    const framesPath = path.join(outDir, `shortest-path-${targetRun.name}.frames.jsonl`);
    const htmlPath   = path.join(outDir, `shortest-path-${targetRun.name}.html`);
    const rec = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H, fps: 2,
      title: `Shortest-path-DES (${targetRun.name})`,
      subtitle: `${graph.numNodes} nodes, ${edgeCount} edges, ${targetRun.result.iterations} iterations`,
      background: '#020617',
    });
    const ticks: number[] = [];
    const minD: number[] = [];
    const maxD: number[] = [];
    for (let i = 0; i < targetRun.result.trace.length; i++) {
      const distNow = targetRun.result.trace[i];
      const events = i > 0 ? targetRun.result.waveEvents[i - 1] ?? [] : [];
      ticks.push(i);
      const finite = distNow.filter(d => isFinite(d));
      minD.push(finite.length ? Math.min(...finite) : 0);
      maxD.push(finite.length ? Math.max(...finite) : 0);
      rec.frame(i, i, () => buildShortestPathFrame(i, i, {
        graph, distanceNow: distNow, waveEvents: events,
        source, iteration: i, algorithm: targetRun.result.algorithm,
      }));
    }
    rec.setCharts(buildShortestPathCharts(ticks, minD, maxD));
    rec.finish();
    console.log(`# Animation written to ${htmlPath}`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
