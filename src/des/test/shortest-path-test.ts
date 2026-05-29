'use strict';

// =============================================================================
// test/shortest-path-test.ts — unit tests for the shortest-path-DES module.
// =============================================================================

import {
  buildSmallChainGraph, buildRandomGraph, Graph,
  shortestPathBellmanFordDES, shortestPathDijkstraDES, reconstructPath,
} from '../general/shortest-path-des';

let pass = 0, fail = 0;
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' — ' + detail : ''));
  cond ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-12): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 — Small chain graph distances (BF)');
// -----------------------------------------------------------------------------
{
  const g = buildSmallChainGraph();
  const r = shortestPathBellmanFordDES(g, 0);
  close('d(s) = 0', r.distance[0], 0);
  close('d(a) = 1', r.distance[1], 1);
  close('d(b) = 3', r.distance[2], 3);
  close('d(c) = 5', r.distance[3], 5);
  close('d(t) = 6', r.distance[4], 6);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 — Small chain graph distances (Dijkstra)');
// -----------------------------------------------------------------------------
{
  const g = buildSmallChainGraph();
  const r = shortestPathDijkstraDES(g, 0);
  close('d(s) = 0', r.distance[0], 0);
  close('d(a) = 1', r.distance[1], 1);
  close('d(b) = 3', r.distance[2], 3);
  close('d(c) = 5', r.distance[3], 5);
  close('d(t) = 6', r.distance[4], 6);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 — Path reconstruction');
// -----------------------------------------------------------------------------
{
  const g = buildSmallChainGraph();
  const r = shortestPathDijkstraDES(g, 0);
  const path = reconstructPath(r, 0, 4);
  expect('path s → t exists', path !== null);
  expect('path = [s, a, b, c, t]', JSON.stringify(path) === JSON.stringify([0, 1, 2, 3, 4]),
    `path = ${JSON.stringify(path)}`);
  // Source-to-itself
  const pSelf = reconstructPath(r, 0, 0);
  expect('path s → s = [s]', JSON.stringify(pSelf) === JSON.stringify([0]));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 — Unreachable nodes have ∞ distance');
// -----------------------------------------------------------------------------
{
  const g: Graph = {
    numNodes: 4,
    edges: [
      [{to: 1, weight: 5}],
      [],
      [{to: 3, weight: 1}],
      [],
    ],
  };
  const r = shortestPathBellmanFordDES(g, 0);
  expect('node 2 unreachable from 0', !isFinite(r.distance[2]));
  expect('node 3 unreachable from 0', !isFinite(r.distance[3]));
  expect('node 1 reachable, d=5', r.distance[1] === 5);
  expect('reconstructPath unreachable returns null',
    reconstructPath(r, 0, 3) === null);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 — BF and Dijkstra agree on random non-negative graphs');
// -----------------------------------------------------------------------------
{
  for (const seed of [1, 7, 13, 42, 99]) {
    const n = 12;
    const g = buildRandomGraph(n, 0.4, 1, 9, seed);
    const bf = shortestPathBellmanFordDES(g, 0);
    const dj = shortestPathDijkstraDES(g, 0);
    let ok = true;
    for (let v = 0; v < n; v++) {
      const a = bf.distance[v], b = dj.distance[v];
      if (!isFinite(a) && !isFinite(b)) continue;
      if (!isFinite(a) || !isFinite(b)) { ok = false; break; }
      if (Math.abs(a - b) > 1e-12) { ok = false; break; }
    }
    expect(`seed=${seed}: BF and Dijkstra agree`, ok);
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 — Negative weights handled correctly');
// -----------------------------------------------------------------------------
{
  const g: Graph = {
    numNodes: 3,
    edges: [
      [{to: 1, weight: 5}, {to: 2, weight: -2}],
      [{to: 2, weight: 1}],
      [],
    ],
  };
  const bf = shortestPathBellmanFordDES(g, 0);
  close('d(2) = -2 (direct negative edge dominates)', bf.distance[2], -2);
  expect('no negative cycle', !bf.hasNegativeCycleFromSource);
  // Dijkstra must throw.
  let threw = false;
  try { shortestPathDijkstraDES(g, 0); } catch (e) { threw = true; }
  expect('Dijkstra throws on negative weight', threw);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 — Negative cycle detection');
// -----------------------------------------------------------------------------
{
  const g: Graph = {
    numNodes: 3,
    edges: [
      [{to: 1, weight: 1}],
      [{to: 2, weight: -3}],
      [{to: 1, weight: 1}],     // 1↔2 has cycle weight = -3 + 1 = -2
    ],
  };
  const bf = shortestPathBellmanFordDES(g, 0);
  expect('negative cycle reachable from source flagged',
    bf.hasNegativeCycleFromSource);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 8 — Trace and waveEvents have consistent shape');
// -----------------------------------------------------------------------------
{
  const g = buildSmallChainGraph();
  const bf = shortestPathBellmanFordDES(g, 0);
  expect('trace.length = iterations + 1 (initial + per-tick snapshots)',
    bf.trace.length === bf.iterations + 1,
    `trace=${bf.trace.length}, iterations=${bf.iterations}`);
  expect('waveEvents.length = iterations',
    bf.waveEvents.length === bf.iterations);
  // Each snapshot has exactly numNodes entries.
  for (const snap of bf.trace) expect('trace entry has numNodes distances',
    snap.length === g.numNodes);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 9 — Reproducibility: same graph + source ⇒ same result');
// -----------------------------------------------------------------------------
{
  const g = buildRandomGraph(15, 0.4, 1, 5, 99);
  const r1 = shortestPathBellmanFordDES(g, 0);
  const r2 = shortestPathBellmanFordDES(g, 0);
  expect('same distances', JSON.stringify(r1.distance) === JSON.stringify(r2.distance));
  expect('same predecessor', JSON.stringify(r1.predecessor) === JSON.stringify(r2.predecessor));
  expect('same iteration count', r1.iterations === r2.iterations);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 10 — Bellman-Ford respects |V|-1 worst-case bound');
// -----------------------------------------------------------------------------
{
  // Linear chain: 0 → 1 → 2 → 3 → 4. With dirty-node optimisation we
  // converge in n-1 = 4 iterations, tight on this graph.
  const n = 6;
  const edges: any[] = [];
  for (let i = 0; i < n - 1; i++) edges.push([{to: i + 1, weight: 1}]);
  edges.push([]);
  const g: Graph = {numNodes: n, edges};
  const bf = shortestPathBellmanFordDES(g, 0);
  for (let v = 0; v < n; v++) close(`linear chain d(${v}) = ${v}`, bf.distance[v], v);
  expect(`linear chain converges in ${bf.iterations} ≤ ${n} iterations`,
    bf.iterations <= n);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
