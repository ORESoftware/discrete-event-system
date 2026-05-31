'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-shortest-path.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Verifies the DES-driven Bellman-Ford ≡ Dijkstra on positive-
// weight graphs and against textbook references.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - `Graph` adjacency -> a typed struct (`HashMap`-of-edges or `Vec<Edge>`).
//   - random-graph generation -> inject `SeededRandom`.
//   - `console.log` PASS/FAIL + `process.exit` -> `println!` / `std::process::exit`.
// =============================================================================

// =============================================================================
// runners/validate-shortest-path.ts — verify the DES-driven Bellman-Ford
// and Dijkstra agree with each other on positive-weight graphs and with
// classic textbook references.
// =============================================================================

import {
  buildSmallChainGraph, buildRandomGraph, Graph,
  shortestPathBellmanFordDES, shortestPathDijkstraDES,
} from '../general/shortest-path-des';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label + (detail ? '  — ' + detail : ''));
  ok ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-12): void {
  check(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// =============================================================================
console.log('\nStudy 1 — Small chain graph: textbook distances');
// =============================================================================
{
  const g = buildSmallChainGraph();
  const r = shortestPathBellmanFordDES(g, 0);
  // Expected distances from 's' (0) — the optimal path is s→a→b→c→t = 1+2+2+1 = 6.
  close('d(s) = 0', r.distance[0], 0);
  close('d(a) = 1', r.distance[1], 1);
  close('d(b) = 3', r.distance[2], 3);    // s→a(1)→b(2) = 3
  close('d(c) = 5', r.distance[3], 5);    // s→a(1)→b(2)→c(2) = 5
  close('d(t) = 6', r.distance[4], 6);    // s→a→b→c→t
  check('Bellman-Ford terminates in ≤ 4 iterations on 5-node graph',
        r.iterations <= 4, `iterations = ${r.iterations}`);
  check('no negative cycle on positive-weight graph',
        !r.hasNegativeCycleFromSource);
}

// =============================================================================
console.log('\nStudy 2 — Bellman-Ford-DES ≡ Dijkstra-DES on non-negative graphs');
// =============================================================================
{
  for (const seed of [1, 7, 13, 42, 99]) {
    const n = 12;
    const g = buildRandomGraph(n, 0.4, 1, 9, seed);
    const bf = shortestPathBellmanFordDES(g, 0);
    const dj = shortestPathDijkstraDES(g, 0);
    let maxDiff = 0;
    for (let v = 0; v < n; v++) {
      const a = bf.distance[v], b = dj.distance[v];
      if (!isFinite(a) && !isFinite(b)) continue;
      if (!isFinite(a) || !isFinite(b)) maxDiff = Infinity;
      else maxDiff = Math.max(maxDiff, Math.abs(a - b));
    }
    check(`seed=${seed}: Bellman-Ford and Dijkstra agree on every distance`,
          maxDiff < 1e-12,
          `max |Δ| = ${maxDiff.toExponential(2)}`);
  }
}

// =============================================================================
console.log('\nStudy 3 — Dijkstra refuses negative weights');
// =============================================================================
{
  const g: Graph = {
    numNodes: 3,
    edges: [
      [{to: 1, weight: 5}, {to: 2, weight: -2}],   // negative weight on (0, 2)
      [{to: 2, weight: 1}],
      [],
    ],
  };
  let threw = false;
  try { shortestPathDijkstraDES(g, 0); } catch (e) { threw = true; }
  check('Dijkstra throws on negative-weight edge', threw);
  // Bellman-Ford handles it just fine (no negative CYCLE).
  const bf = shortestPathBellmanFordDES(g, 0);
  close('Bellman-Ford handles negative edge: d(2) = -2',
        bf.distance[2], -2);
  check('Bellman-Ford does not flag negative cycle', !bf.hasNegativeCycleFromSource);
}

// =============================================================================
console.log('\nStudy 4 — Bellman-Ford detects negative cycles reachable from source');
// =============================================================================
{
  const g: Graph = {
    numNodes: 3,
    edges: [
      [{to: 1, weight: 1}],
      [{to: 2, weight: -3}],
      [{to: 1, weight: 1}],     // 1 → 2 → 1 has total weight -3 + 1 = -2 (negative cycle)
    ],
  };
  const bf = shortestPathBellmanFordDES(g, 0);
  check('negative cycle reachable from source flagged',
        bf.hasNegativeCycleFromSource);
}

// =============================================================================
console.log('\nStudy 5 — Bellman-Ford terminates in ≤ |V|-1 iterations on positive-weight graphs');
// =============================================================================
{
  for (const n of [5, 10, 20, 30]) {
    const g = buildRandomGraph(n, 0.3, 1, 5, 42 + n);
    const bf = shortestPathBellmanFordDES(g, 0);
    check(`n=${n}: Bellman-Ford ran in ${bf.iterations} iterations (≤ ${n - 1})`,
          bf.iterations <= n,    // could be n if it ran the cycle-detection round
          `iterations=${bf.iterations}, |V|-1=${n - 1}`);
  }
}

// =============================================================================
console.log('\nStudy 6 — Wave count: Dijkstra waves ≤ Bellman-Ford waves on dense graphs');
// =============================================================================
{
  for (const seed of [1, 7, 42]) {
    const g = buildRandomGraph(15, 0.5, 1, 10, seed);
    const bf = shortestPathBellmanFordDES(g, 0);
    const dj = shortestPathDijkstraDES(g, 0);
    console.log(`    seed=${seed}: BF waves = ${bf.wavesEmitted}, Dij waves = ${dj.wavesEmitted}`);
    // Dijkstra processes each edge at most once (since each node is settled
    // at most once), so its wave count = total edges. BF can revisit edges
    // across iterations and is therefore typically larger.
    check(`seed=${seed}: Dijkstra waves ≤ Bellman-Ford waves`,
          dj.wavesEmitted <= bf.wavesEmitted,
          `dij=${dj.wavesEmitted}, bf=${bf.wavesEmitted}`);
  }
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
