'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-simulated-annealing.rs   (fn main)
// 1:1 file move. CLI driver for Simulated Annealing (TSP at several sizes,
// cooling-schedule comparison, knapsack vs MILP-B&B/brute).
//
// Conversion notes (file-specific):
//   - SA acceptance uses Math.random -> inject RandomSource/SeededRandom.
//   - CoolingSchedule type -> enum.
//   - use crate::des::general::{simulated_annealing, genetic_tsp, milp_bnb}.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-simulated-annealing.ts — CLI driver for Simulated Annealing.
//
// Demonstrates SA on:
//   1. Small TSP (n=5 pentagon) — should hit the exact optimum.
//   2. Medium TSP (n=12) — compared against Held–Karp exact and the GA.
//   3. Larger TSP (n=30) — head-to-head with GA, both starting from
//      nearest-neighbour, equal compute budget.
//   4. Cooling-schedule comparison — geometric vs logarithmic vs linear.
//   5. 0/1 knapsack — SA vs MILP-B&B vs brute force.
// =============================================================================

import {
  runSimulatedAnnealing, buildTSPSAProblem, buildKnapsackSAProblem,
  CoolingSchedule,
} from './general/simulated-annealing';
import {
  buildPentagonTSP, buildRandomTSP, tourLength, heldKarpExact, runGeneticTSP,
} from './general/genetic-tsp';
import {solveMILP, buildKnapsackMILP} from './general/milp-bnb';

function header(s: string): void {
  console.log();
  console.log('═'.repeat(96));
  console.log('  ' + s);
  console.log('═'.repeat(96));
}

function main(): void {

  header('STUDY 1 — Pentagon TSP (n=5): SA finds the exact optimum');
  {
    const inst = buildPentagonTSP(5, 50);
    const optimum = tourLength(inst, [0, 1, 2, 3, 4]);
    const saP = buildTSPSAProblem(inst);
    const r = runSimulatedAnnealing(saP, {
      maxIterations: 2000,
      cooling: {kind: 'geometric', T0: 50, alpha: 0.998},
      seed: 1,
    });
    console.log(`  optimum (perimeter) = ${optimum.toFixed(4)}`);
    console.log(`  SA best             = ${r.bestCost.toFixed(4)}    ratio = ${(r.bestCost / optimum).toFixed(6)}`);
    console.log(`  iters = ${r.iterations}, accepted = ${r.acceptedCount}, improvements = ${r.improveCount}`);
  }

  header('STUDY 2 — n=12 random TSP: SA vs Held–Karp (exact) vs GA');
  {
    const inst = buildRandomTSP(12, 17);
    const t0 = Date.now();
    const exact = heldKarpExact(inst);
    const dtExact = Date.now() - t0;
    const t1 = Date.now();
    const sa = runSimulatedAnnealing(buildTSPSAProblem(inst), {
      maxIterations: 20000,
      cooling: {kind: 'geometric', T0: 50, alpha: 0.9995},
      seed: 1,
    });
    const dtSA = Date.now() - t1;
    const t2 = Date.now();
    const ga = runGeneticTSP(inst, {populationSize: 80, numGenerations: 200, seed: 1, init: 'nearest-neighbor'});
    const dtGA = Date.now() - t2;
    console.log(`  Held–Karp (exact)    z = ${exact.length.toFixed(4)}    wall = ${dtExact} ms`);
    console.log(`  SA       z = ${sa.bestCost.toFixed(4)}  ratio = ${(sa.bestCost / exact.length).toFixed(6)}    wall = ${dtSA} ms (${sa.iterations} iters)`);
    console.log(`  GA       z = ${ga.bestLength.toFixed(4)}  ratio = ${(ga.bestLength / exact.length).toFixed(6)}    wall = ${dtGA} ms (${ga.generations} generations)`);
  }

  header('STUDY 3 — n=30 random TSP: SA vs GA, equal compute');
  {
    const inst = buildRandomTSP(30, 99);
    const t0 = Date.now();
    const sa = runSimulatedAnnealing(buildTSPSAProblem(inst), {
      maxIterations: 100000,
      cooling: {kind: 'geometric', T0: 200, alpha: 0.99995},
      seed: 1,
    });
    const dtSA = Date.now() - t0;
    const t1 = Date.now();
    const ga = runGeneticTSP(inst, {populationSize: 200, numGenerations: 500, seed: 1, init: 'nearest-neighbor'});
    const dtGA = Date.now() - t1;
    console.log(`  SA   z = ${sa.bestCost.toFixed(4)}    wall = ${dtSA} ms (${sa.iterations} iters)`);
    console.log(`  GA   z = ${ga.bestLength.toFixed(4)}    wall = ${dtGA} ms (${ga.generations} generations)`);
    const winner = sa.bestCost < ga.bestLength ? 'SA' : 'GA';
    console.log(`  winner: ${winner}    margin = ${Math.abs(sa.bestCost - ga.bestLength).toFixed(4)}`);
  }

  header('STUDY 4 — Cooling schedules on the same TSP');
  {
    const inst = buildRandomTSP(20, 5);
    const exact = inst.n <= 14 ? heldKarpExact(inst) : null;
    if (exact) console.log(`  exact = ${exact.length.toFixed(4)}`);
    console.log(`  ${'schedule'.padEnd(28)}${'best'.padStart(10)}${'iters'.padStart(8)}${'wall(ms)'.padStart(10)}`);
    const schedules: Array<{name: string; sched: CoolingSchedule; iters: number}> = [
      {name: 'geometric  α=0.999, T0=50',   sched: {kind: 'geometric',   T0: 50,  alpha: 0.999},  iters: 30000},
      {name: 'geometric  α=0.9995, T0=200', sched: {kind: 'geometric',   T0: 200, alpha: 0.9995}, iters: 30000},
      {name: 'logarithmic T0=200',           sched: {kind: 'logarithmic', T0: 200},                iters: 30000},
      {name: 'linear     rate=0.005',         sched: {kind: 'linear',      T0: 100, rate: 0.005},   iters: 20000},
      {name: 'exp-restart α=0.99, p=2000',   sched: {kind: 'exp-restart', T0: 50,  alpha: 0.99, period: 2000}, iters: 30000},
    ];
    for (const s of schedules) {
      const t0 = Date.now();
      const r = runSimulatedAnnealing(buildTSPSAProblem(inst), {
        maxIterations: s.iters, cooling: s.sched, seed: 7,
      });
      const dt = Date.now() - t0;
      console.log(`  ${s.name.padEnd(28)}${r.bestCost.toFixed(2).padStart(10)}${r.iterations.toString().padStart(8)}${dt.toString().padStart(10)}`);
    }
  }

  header('STUDY 5 — 0/1 knapsack: SA heuristic vs MILP-B&B exact');
  {
    let s = 1234;
    const rng = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 0x100000000; };
    const n = 15;
    const v = Array.from({length: n}, () => Math.floor(rng() * 50 + 1));
    const w = Array.from({length: n}, () => Math.floor(rng() * 25 + 1));
    const cap = Math.floor(w.reduce((a, b) => a + b, 0) * 0.4);
    console.log(`  n=${n}, capacity=${cap}`);
    const t0 = Date.now();
    const exact = solveMILP(buildKnapsackMILP(v, w, cap));
    const dtExact = Date.now() - t0;
    const t1 = Date.now();
    const sa = runSimulatedAnnealing(buildKnapsackSAProblem({values: v, weights: w, capacity: cap}), {
      maxIterations: 5000, cooling: {kind: 'geometric', T0: 30, alpha: 0.999}, seed: 11,
    });
    const dtSA = Date.now() - t1;
    console.log(`  MILP-B&B (exact):  z = ${exact.z.toFixed(2)}    wall = ${dtExact} ms    nodes = ${exact.nodesExplored}`);
    console.log(`  SA (heuristic):    z = ${(-sa.bestCost).toFixed(2)}    wall = ${dtSA} ms    iters = ${sa.iterations}`);
    const ratio = (-sa.bestCost) / exact.z;
    console.log(`  SA / exact = ${ratio.toFixed(6)}    (1.0 = found exact optimum)`);
  }
}

main();
