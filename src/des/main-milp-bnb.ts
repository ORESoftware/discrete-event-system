// RUST MIGRATION: target src/bin/main_milp_bnb.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-milp-bnb.rs   (fn main)
// 1:1 file move. CLI driver for MILP via Branch-and-Bound (knapsack with B&B
// trace, brute-force sanity check, generic MILP, pure-LP fallback, scaling).
//
// Conversion notes (file-specific):
//   - bitmask brute enumeration (1 << n) -> u64/usize bitsets.
//   - use crate::des::general::milp_bnb; top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-milp-bnb.ts — CLI driver for MILP via Branch-and-Bound.
//
// Demonstrates:
//   1. 0/1 knapsack — small textbook instance solved with full B&B trace.
//   2. Larger knapsack vs brute-force enumeration (sanity check).
//   3. Generic MILP — minimisation example with mixed integer/continuous vars.
//   4. Pure-LP fallback — when integerVars is all-false the solver
//      reduces to a single root-node LP solve. Useful sanity check.
//   5. Performance scaling — knapsack with growing n, showing how B&B
//      avoids the O(2^n) blow-up of brute force.
// =============================================================================

import {solveMILP, buildKnapsackMILP, MILPProblem, MILPSolution} from './general/milp-bnb';

function header(s: string): void {
  console.log();
  console.log('═'.repeat(96));
  console.log('  ' + s);
  console.log('═'.repeat(96));
}

function bruteKnapsack(values: number[], weights: number[], capacity: number): {z: number; x: number[]} {
  const n = values.length;
  let bestZ = 0, bestX = new Array(n).fill(0);
  for (let mask = 0; mask < (1 << n); mask++) {
    let v = 0, w = 0;
    for (let i = 0; i < n; i++) if (mask & (1 << i)) { v += values[i]; w += weights[i]; }
    if (w <= capacity && v > bestZ) {
      bestZ = v; bestX = Array.from({length: n}, (_, i) => (mask & (1 << i)) ? 1 : 0);
    }
  }
  return {z: bestZ, x: bestX};
}

function printSolution(label: string, r: MILPSolution): void {
  const xPretty = r.x.map(v => Number.isFinite(v) ? v.toFixed(3) : 'N/A').slice(0, 16).join(', ');
  console.log(`  ${label}`);
  console.log(`    status:   ${r.status}`);
  console.log(`    z*:       ${Number.isFinite(r.z) ? r.z.toFixed(4) : r.z}`);
  console.log(`    bestBound:${r.bestBound.toFixed(4)}    gap: ${r.gap.toExponential(2)}`);
  console.log(`    x* (first 16):  [${xPretty}${r.x.length > 16 ? ', …' : ''}]`);
  console.log(`    nodes:    ${r.nodesExplored}    LP pivots: ${r.totalPivots}`);
}

function main(): void {

  header('STUDY 1 — Textbook 0/1 knapsack (4 items)');
  console.log('  values  v = [10, 40, 30, 50]');
  console.log('  weights w = [ 5,  4,  6,  3]');
  console.log('  capacity W = 10');
  {
    const milp = buildKnapsackMILP([10, 40, 30, 50], [5, 4, 6, 3], 10);
    const t0 = Date.now();
    const r = solveMILP(milp, {verbose: true});
    const dt = Date.now() - t0;
    console.log();
    printSolution(`B&B solution (wall=${dt}ms):`, r);
  }

  header('STUDY 2 — 12-item knapsack vs brute force (4096 enumerations)');
  {
    const v = [12, 18,  9,  4, 21, 35, 14, 25, 30,  8, 17,  6];
    const w = [ 5,  8,  4,  3,  9, 13,  6,  7, 11,  3,  6,  4];
    const cap = 30;
    const milp = buildKnapsackMILP(v, w, cap);
    const t0 = Date.now();
    const r = solveMILP(milp);
    const t1 = Date.now();
    const brute = bruteKnapsack(v, w, cap);
    const t2 = Date.now();
    printSolution('B&B', r);
    console.log(`    brute force z=${brute.z}, x=[${brute.x.join(', ')}]`);
    const match = Math.abs(r.z - brute.z) < 1e-6;
    console.log(`    match: ${match ? 'YES' : 'NO'}    (B&B=${(t1 - t0)}ms, brute=${(t2 - t1)}ms)`);
  }

  header('STUDY 3 — Generic MILP (2 integer + 1 continuous var, min sense)');
  console.log('  min  3 x_0 + 5 x_1 + 7 x_2');
  console.log('  s.t. x_0 + x_1 + x_2 ≤ 10');
  console.log('       2 x_0 + x_1     ≤ 8');
  console.log('       x_0 + 2 x_1 + 3 x_2 ≤ 15');
  console.log('  x_0, x_1 ∈ ℤ_≥0,  x_2 ∈ ℝ_≥0');
  {
    // To MAXIMIZE we negate. To MINIMIZE — but the MILP solver only
    // supports b ≥ 0 (no Phase-1). Use a max problem for diversity:
    //   max  3 x_0 + 5 x_1 + 7 x_2
    //   s.t. x_0 + x_1 + x_2 ≤ 10
    //        2 x_0 + x_1     ≤ 8
    //        x_0 + 2 x_1 + 3 x_2 ≤ 15
    //   x_0, x_1 ∈ ℤ_≥0, x_2 ∈ ℝ_≥0.
    const milp: MILPProblem = {
      sense: 'max',
      c: [3, 5, 7],
      A: [[1, 1, 1], [2, 1, 0], [1, 2, 3]],
      b: [10, 8, 15],
      integerVars: [true, true, false],
    };
    const r = solveMILP(milp, {verbose: false});
    printSolution('B&B', r);
  }

  header('STUDY 4 — Pure LP (all variables continuous) — should run only the root');
  {
    const lp: MILPProblem = {
      sense: 'max',
      c: [3, 5],
      A: [[1, 0], [0, 2], [3, 2]],
      b: [4, 12, 18],
      integerVars: [false, false],
    };
    const r = solveMILP(lp);
    printSolution('B&B (no integrality)', r);
    console.log(`    expected: z = 36, x = (2, 6) — classic textbook 2-D LP.`);
  }

  header('STUDY 5 — Knapsack scaling: B&B nodes vs n');
  console.log('  n         nodes  pivots   z*       wall(ms)');
  for (const n of [6, 10, 14, 18, 22, 26]) {
    const v: number[] = []; const w: number[] = [];
    let s = 1; const rng = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 0x100000000; };
    for (let i = 0; i < n; i++) { v.push(Math.floor(rng() * 40 + 1)); w.push(Math.floor(rng() * 25 + 1)); }
    const cap = Math.floor(w.reduce((a, b) => a + b, 0) * 0.4);
    const milp = buildKnapsackMILP(v, w, cap);
    const t0 = Date.now();
    const r = solveMILP(milp, {maxNodes: 50000});
    const dt = Date.now() - t0;
    console.log(`  ${n.toString().padStart(2)}     ${r.nodesExplored.toString().padStart(7)}  ${r.totalPivots.toString().padStart(7)}  ${r.z.toFixed(2).padStart(7)}    ${dt.toString().padStart(5)}`);
  }
}

main();
