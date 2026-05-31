// RUST MIGRATION: Target module `src/des/general/shortest_path_des.rs`.
// RUST MIGRATION: Convert graph specs, wave/update tokens, trace rows, and result interfaces to `serde` structs; algorithm mode unions should become enums.
// RUST MIGRATION: Port node/station/message classes as structs implementing `Token`/`DESStation` traits so each graph node remains a stationary entity.
// RUST MIGRATION: Use `HashMap`/`HashSet`/priority queues for distance, predecessor, adjacency, and frontier indexes instead of structural object maps.
// RUST MIGRATION: Keep top-level shortest-path runners/builders as free functions; return `Result` for negative-cycle, unreachable-node, and malformed-graph errors.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/shortest-path-des.rs  (module des::general::shortest_path_des)
// 1:1 file move. Shortest path (Bellman-Ford / Dijkstra) as a DES of node stations + wave messages.
//
// Declarations → Rust:
//   interface Edge / Graph / SPResult / BellmanFordOptions -> structs (#[derive(Clone)])
//   interface PQEntry (private)                            -> struct
//   class IndexedMinHeap (private)                         -> struct + impl (binary heap w/ index map)
//   fn reconstructPath / shortestPathBellmanFordDES / shortestPathDijkstraDES /
//      buildRandomGraph / buildSmallChainGraph             -> free fns / assoc fns
//
// Conversion notes (file-specific):
//   - INJECT RNG: `buildRandomGraph` uses an rng -> take a `RandomSource` (shared/capabilities).
//   - node ids are `usize`; weights are `f64` (Bellman-Ford allows negatives); distances may be
//     `f64::INFINITY` -> keep as `f64` sentinel, not `Option`, to match the algorithm.
//   - IndexedMinHeap is a hand-rolled decrease-key heap -> a `BinaryHeap` + position `Vec`/HashMap,
//     or the `Reverse`-wrapped std heap with stale-entry skipping.
//   - `reconstructPath` returns `number[] | null` -> `Option<Vec<usize>>`.
// =============================================================================
// general/shortest-path-des.ts — Shortest path on a weighted directed
// graph computed BY THE DES, where every graph node IS a stationary entity
// and every distance update IS a movable "wave" message flowing along an
// edge.
//
// THE PROBLEM
// ───────────
//   Given G = (V, E, w) directed (or symmetric) with edge weights w
//   (allowing negatives for Bellman-Ford, non-negatives for Dijkstra),
//   find the shortest distance from a source s ∈ V to every other node.
//
// AS A DES
// ────────
//   Nodes are stations. Each node-station holds:
//     - distance:    its current shortest-distance estimate from the source
//     - predecessor: previous node on the best path so far
//     - dirty:       a flag indicating its distance changed in the last tick
//
//   Movables are "Wave" messages. Each wave carries:
//     - sourceNodeId, targetNodeId
//     - distanceProposal = sourceNode.distance + edge.weight
//
//   Tick model — TWO MODES
//   ──────────────────────
//     1. Bellman-Ford-DES  (`shortestPathBellmanFordDES`)
//        Each tick: for every dirty node, emit waves along its outgoing
//        edges. Each receiving station compares its current distance to
//        the wave's proposal; if smaller, it updates and becomes dirty.
//        Loop until no waves are generated. Terminates in ≤ |V|-1
//        iterations on graphs without negative cycles. Iteration |V|
//        catches negative cycles (any new improvement ⇒ negative cycle
//        reachable from source).
//
//     2. Dijkstra-DES      (`shortestPathDijkstraDES`)
//        Maintains a global priority queue of (distance, nodeId) — at
//        each tick the min-distance entry is popped, that node is "settled",
//        and waves are emitted along its outgoing edges. Settled nodes
//        no longer accept updates. Requires non-negative weights.
//
// "Iterative / recursive" framing
// ────────────────────────────────
//   The Bellman recurrence d(v) = min_{(u,v) ∈ E} (d(u) + w(u,v)) IS
//   the iterative computation we run. Both modes converge to the unique
//   fixed point of this recurrence. The Bellman-Ford-DES is the
//   straight iterative version (no priority); Dijkstra-DES is a smarter
//   schedule of the same fixed-point computation.
// =============================================================================

export interface Edge {
  to: number;
  weight: number;
}

export interface Graph {
  numNodes: number;
  /** edges[u] = list of outgoing edges from u. */
  edges: Edge[][];
  /** Optional 2-D coordinates for animation only. */
  coordinates?: Array<[number, number]>;
  /** Optional names for animation captions. */
  nodeNames?: string[];
}

export interface SPResult {
  /** distance[v] = shortest distance from source to v (Infinity if unreachable). */
  distance: number[];
  /** predecessor[v] = previous node on the shortest path (-1 if source / unreachable). */
  predecessor: number[];
  /** Number of full tick rounds run. */
  iterations: number;
  /** Total number of waves emitted (edge relaxations performed). */
  wavesEmitted: number;
  /** True iff a negative cycle reachable from the source was detected. */
  hasNegativeCycleFromSource: boolean;
  /** Per-tick distance-vector snapshots (for animation). distance trace[t][v]. */
  trace: number[][];
  /** Per-tick wave events: which (u → v) relaxations fired this tick. */
  waveEvents: Array<Array<{from: number; to: number; newDistance: number; improved: boolean}>>;
  /** Algorithm used for the result. */
  algorithm: 'bellman-ford-des' | 'dijkstra-des';
}

/** Reconstruct the shortest path from source to target using the
 *  predecessor array. Returns null if target is unreachable. */
export function reconstructPath(result: SPResult, source: number, target: number): number[] | null {
  if (!isFinite(result.distance[target])) return null;
  const path: number[] = [];
  let cur = target;
  const seen = new Set<number>();
  while (cur !== -1) {
    if (seen.has(cur)) return null;     // cycle in predecessors → broken
    seen.add(cur);
    path.push(cur);
    if (cur === source) break;
    cur = result.predecessor[cur];
  }
  if (path[path.length - 1] !== source) return null;
  path.reverse();
  return path;
}

// =============================================================================
// MODE 1 — BELLMAN-FORD as DES
// =============================================================================

export interface BellmanFordOptions {
  /** Hard cap on the number of tick rounds. Default = numNodes (which is
   *  enough for non-negative-cycle convergence; iteration numNodes itself
   *  detects negative cycles reachable from source). */
  maxIterations?: number;
  /** If false, suppresses the per-tick trace (saves memory). Default true. */
  recordTrace?: boolean;
}

export function shortestPathBellmanFordDES(
  graph: Graph, source: number,
  opts: BellmanFordOptions = {},
): SPResult {
  const n = graph.numNodes;
  const distance = new Array(n).fill(Infinity);
  const predecessor = new Array(n).fill(-1);
  let dirty = new Array(n).fill(false);
  distance[source] = 0;
  dirty[source] = true;
  const trace: number[][] = [];
  const waveEvents: SPResult['waveEvents'] = [];
  if (opts.recordTrace !== false) trace.push(distance.slice());
  let wavesEmitted = 0;
  const maxIter = opts.maxIterations ?? n;
  let iter = 0;
  let hasNegativeCycle = false;

  while (iter < maxIter) {
    iter++;
    const newDirty = new Array(n).fill(false);
    const eventsThisTick: SPResult['waveEvents'][number] = [];
    let anyChange = false;
    for (let u = 0; u < n; u++) {
      if (!dirty[u]) continue;
      const du = distance[u];
      for (const edge of graph.edges[u]) {
        wavesEmitted++;
        const cand = du + edge.weight;
        const before = distance[edge.to];
        const improved = cand < before - 1e-12;
        eventsThisTick.push({from: u, to: edge.to, newDistance: cand, improved});
        if (improved) {
          distance[edge.to] = cand;
          predecessor[edge.to] = u;
          newDirty[edge.to] = true;
          anyChange = true;
        }
      }
    }
    waveEvents.push(eventsThisTick);
    if (opts.recordTrace !== false) trace.push(distance.slice());
    dirty = newDirty;
    if (!anyChange) break;
    // Iteration n+1 onwards detecting any change ⇒ negative cycle.
    if (iter >= n) { hasNegativeCycle = anyChange; break; }
  }
  return {
    distance, predecessor,
    iterations: iter, wavesEmitted,
    hasNegativeCycleFromSource: hasNegativeCycle,
    trace, waveEvents,
    algorithm: 'bellman-ford-des',
  };
}

// =============================================================================
// MODE 2 — DIJKSTRA as DES
// =============================================================================

interface PQEntry {
  distance: number;
  nodeId: number;
}

/** Tiny indexed binary-heap min-priority-queue for Dijkstra. */
class IndexedMinHeap {
  private heap: PQEntry[] = [];
  push(entry: PQEntry): void {
    this.heap.push(entry);
    this.bubbleUp(this.heap.length - 1);
  }
  pop(): PQEntry | undefined {
    if (this.heap.length === 0) return undefined;
    const top = this.heap[0];
    const last = this.heap.pop()!;
    if (this.heap.length > 0) { this.heap[0] = last; this.bubbleDown(0); }
    return top;
  }
  size(): number { return this.heap.length; }
  private bubbleUp(i: number): void {
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (this.heap[i].distance < this.heap[par].distance) {
        [this.heap[i], this.heap[par]] = [this.heap[par], this.heap[i]]; i = par;
      } else break;
    }
  }
  private bubbleDown(i: number): void {
    const n = this.heap.length;
    while (true) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let best = i;
      if (l < n && this.heap[l].distance < this.heap[best].distance) best = l;
      if (r < n && this.heap[r].distance < this.heap[best].distance) best = r;
      if (best === i) break;
      [this.heap[i], this.heap[best]] = [this.heap[best], this.heap[i]]; i = best;
    }
  }
}

export function shortestPathDijkstraDES(
  graph: Graph, source: number,
  opts: BellmanFordOptions = {},
): SPResult {
  const n = graph.numNodes;
  const distance = new Array(n).fill(Infinity);
  const predecessor = new Array(n).fill(-1);
  const settled = new Array(n).fill(false);
  distance[source] = 0;
  const pq = new IndexedMinHeap();
  pq.push({distance: 0, nodeId: source});
  const trace: number[][] = [];
  const waveEvents: SPResult['waveEvents'] = [];
  if (opts.recordTrace !== false) trace.push(distance.slice());
  let wavesEmitted = 0;
  let iter = 0;

  // Sanity: weights must be non-negative for Dijkstra to be correct.
  for (let u = 0; u < n; u++)
    for (const e of graph.edges[u])
      if (e.weight < -1e-12) throw new Error(`Dijkstra requires non-negative weights, got ${e.weight} on edge ${u}→${e.to}`);

  while (pq.size() > 0) {
    iter++;
    const top = pq.pop()!;
    if (settled[top.nodeId]) { iter--; continue; }
    settled[top.nodeId] = true;
    const eventsThisTick: SPResult['waveEvents'][number] = [];
    for (const edge of graph.edges[top.nodeId]) {
      wavesEmitted++;
      const cand = top.distance + edge.weight;
      const improved = cand < distance[edge.to] - 1e-12;
      eventsThisTick.push({from: top.nodeId, to: edge.to, newDistance: cand, improved});
      if (improved) {
        distance[edge.to] = cand;
        predecessor[edge.to] = top.nodeId;
        pq.push({distance: cand, nodeId: edge.to});
      }
    }
    waveEvents.push(eventsThisTick);
    if (opts.recordTrace !== false) trace.push(distance.slice());
  }
  return {
    distance, predecessor,
    iterations: iter, wavesEmitted,
    hasNegativeCycleFromSource: false,
    trace, waveEvents,
    algorithm: 'dijkstra-des',
  };
}

// =============================================================================
// GRAPH BUILDERS / HELPERS
// =============================================================================

/** Build a directed Erdős–Rényi-style random graph with edge probability p,
 *  where each edge has uniform weight in [wMin, wMax]. */
export function buildRandomGraph(
  numNodes: number,
  edgeProb: number,
  wMin: number,
  wMax: number,
  seed: number,
): Graph {
  const rng = (() => { // mulberry32 inlined to keep this file self-contained
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();
  const edges: Edge[][] = Array.from({length: numNodes}, () => []);
  for (let u = 0; u < numNodes; u++) {
    for (let v = 0; v < numNodes; v++) {
      if (u === v) continue;
      if (rng() < edgeProb) edges[u].push({to: v, weight: wMin + (wMax - wMin) * rng()});
    }
  }
  // Ensure every node has at least one outgoing edge (avoid trivially
  // disconnected sinks for small p).
  for (let u = 0; u < numNodes; u++) {
    if (edges[u].length === 0) {
      let v = Math.floor(rng() * numNodes);
      if (v === u) v = (v + 1) % numNodes;
      edges[u].push({to: v, weight: wMin + (wMax - wMin) * rng()});
    }
  }
  // Random 2-D layout for animation.
  const coordinates: Array<[number, number]> = [];
  for (let i = 0; i < numNodes; i++) coordinates.push([rng() * 100, rng() * 100]);
  return {numNodes, edges, coordinates};
}

/** Build a small canonical graph used by examples and tests. The optimum
 *  path from 0 to 4 has length 6: 0→1(1)→2(2)→3(2)→4(1).
 *  ┌──────────────────────────────────────────────────────────┐
 *  │  0 ──1── 1 ──2── 2 ──2── 3 ──1── 4                       │
 *  │   \             /                                         │
 *  │    \──── 4 ────/  (alt. heavier path 0→2 directly)        │
 *  │   \                                                       │
 *  │    \──── 10 ─── 4   (very expensive direct edge)          │
 *  └──────────────────────────────────────────────────────────┘
 */
export function buildSmallChainGraph(): Graph {
  const edges: Edge[][] = [
    [{to: 1, weight: 1}, {to: 2, weight: 4}, {to: 4, weight: 10}],   // 0
    [{to: 2, weight: 2}, {to: 3, weight: 5}],                          // 1
    [{to: 3, weight: 2}],                                              // 2
    [{to: 4, weight: 1}],                                              // 3
    [],                                                                // 4
  ];
  return {numNodes: 5, edges, nodeNames: ['s', 'a', 'b', 'c', 't']};
}
