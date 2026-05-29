'use strict';

// =============================================================================
// main-genetic-tsp.ts — solve the Travelling Salesman Problem with a
// genetic algorithm modelled inside the DES engine, with optional
// precedence constraints that exercise the branch-cutting pathway.
//
// THE GA AS A DES
// ───────────────
//   Each generation is one DES tick. Inside the tick a breeding pipeline
//   of stations runs:
//
//      Selection  →  Crossover  →  Mutation  →  Feasibility  →  Fitness  →  Replacement
//
//   Movables are CHROMOSOMES (city permutations). Constraints (precedence)
//   are enforced at the Feasibility station: infeasible offspring are
//   either CUT (dropped, with bounded retry), PENALISED (objective +∞),
//   or REPAIRED (swap-fix until feasible).
//
// USAGE
// ─────
//     node dist/des/main-genetic-tsp.js                     # 30 cities, no constraints
//     N_CITIES=12 INSTANCE=pentagon node …                  # 12-city pentagon, GA + Held-Karp
//     PRECEDENCE=1 FEASIBILITY=cut node …                   # GA with branch cutting
//     PRECEDENCE=1 FEASIBILITY=penalize node …              # comparison policy
//     ANIMATE=1 N_CITIES=20 GENERATIONS=80 node …           # writes out HTML
// =============================================================================

import * as path from 'path';
import {
  buildRandomTSP, buildPentagonTSP, runGeneticTSP,
  tourLength, checkPrecedence, isPermutation,
  heldKarpExact, oneTreeLowerBound, TSPInstance,
} from './general/genetic-tsp';
import {FrameRecorder} from './animation/frame-recorder';
import {STAGE_W, STAGE_H, buildGeneticTSPFrame, buildGeneticTSPCharts} from './animation/scenes/genetic-tsp-scene';

async function main(): Promise<void> {
  const n = Number(process.env.N_CITIES ?? 25);
  const seed = Number(process.env.SEED ?? 7);
  const generations = Number(process.env.GENERATIONS ?? 200);
  const popSize = Number(process.env.POP ?? 100);
  const usePrecedence = process.env.PRECEDENCE === '1';
  const feasibility = (process.env.FEASIBILITY ?? 'cut') as 'cut' | 'penalize' | 'repair';
  const animate = process.env.ANIMATE === '1';
  const instanceKind = process.env.INSTANCE ?? 'random';

  // ── Build the instance ──
  let instance: TSPInstance;
  if (instanceKind === 'pentagon') {
    instance = buildPentagonTSP(n, 50);
  } else {
    instance = buildRandomTSP(n, seed);
  }
  if (usePrecedence) {
    // 4 precedence pairs: (0→last), (1→second-last), …
    const pp: Array<[number, number]> = [];
    for (let i = 0; i < Math.min(4, Math.floor(n / 3)); i++) pp.push([i, n - 1 - i]);
    instance.precedence = pp;
  }

  // ── Banner ──
  console.log('# Genetic-TSP solver (GA inside the DES engine)');
  console.log(`# n=${n} cities, instance=${instanceKind}, seed=${seed}`);
  console.log(`# population=${popSize}, generations=${generations}`);
  if (usePrecedence) console.log(`# precedence pairs: ${JSON.stringify(instance.precedence)}, feasibility=${feasibility}`);
  else console.log('# no precedence constraints');
  console.log('');
  console.log('# Lower bound (1-tree relaxation): ' + oneTreeLowerBound(instance).toFixed(3));
  if (n <= 14 && !usePrecedence) {
    process.stdout.write('# Computing exact Held–Karp optimum ... ');
    const t0 = Date.now();
    const hk = heldKarpExact(instance);
    console.log(`length = ${hk.length.toFixed(3)} in ${Date.now() - t0}ms`);
  } else {
    console.log('# (Held–Karp skipped: n > 14 or precedence active)');
  }
  console.log('');

  // ── Run the GA ──
  process.stdout.write('# Running GA ... ');
  const t0 = Date.now();
  const result = runGeneticTSP(instance, {
    populationSize: popSize,
    numGenerations: generations,
    seed: seed + 1000,
    feasibility,
    elitism: 4,
    init: 'nearest-neighbor',
  });
  console.log(`done in ${Date.now() - t0}ms`);
  console.log('');

  // ── Report ──
  console.log(`# Best tour length found  = ${result.bestLength.toFixed(3)}`);
  console.log(`# Best tour valid permutation? ${isPermutation(result.bestTour, instance.n)}`);
  console.log(`# Best tour feasible (precedence)? ${checkPrecedence(instance, result.bestTour) === null}`);
  console.log(`# Total feasible children evaluated  = ${result.totalFeasibleEvaluated}`);
  console.log(`# Total infeasible children cut      = ${result.totalInfeasibleCut}`);
  console.log('');
  console.log(`# Tour: ${result.bestTour.join(' → ')} → ${result.bestTour[0]}`);
  console.log('');

  // Convergence summary: best per generation, sampled.
  const sampled = [];
  const step = Math.max(1, Math.floor(generations / 10));
  for (let g = 0; g < generations; g += step) sampled.push([g, result.perGenerationBest[g], result.perGenerationMean[g]]);
  console.log('# Convergence (sampled):');
  console.log('#   gen     best        mean');
  for (const [g, b, m] of sampled) console.log(`#   ${String(g).padStart(4)}  ${b.toFixed(3).padStart(8)}  ${m.toFixed(3).padStart(8)}`);
  console.log('');

  // ── Animation ──
  if (animate) {
    const outDir = path.join(__dirname, '..', '..', 'out');
    const framesPath = path.join(outDir, `genetic-tsp.frames.jsonl`);
    const htmlPath   = path.join(outDir, `genetic-tsp.html`);
    // Six sub-frames per generation (one per station in the GA chain),
    // showing chromosomes flowing along the pipeline.
    const rec = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H, fps: 12,    // 12 fps × 6 phases ≈ 2 generations/sec
      title: `Genetic-TSP — ${n} cities, ${feasibility} branch-cutting`,
      subtitle: `Each generation has 6 phases (Selection → Crossover → Mutation → Feasibility → Fitness → Replacement). Movables = chromosomes.`,
      background: '#020617',
    });
    const gens: number[] = [];
    let subTick = 0;
    for (let g = 0; g < generations; g++) {
      gens.push(g);
      const tour = result.perGenerationElite[g];
      for (let phase = 0 as 0|1|2|3|4|5; phase < 6; phase = (phase + 1) as 0|1|2|3|4|5) {
        const tCapture = g + phase / 6;
        const sub = subTick++;
        rec.frame(tCapture, sub, () => buildGeneticTSPFrame(tCapture, sub, {
          instance, eliteTour: tour,
          best: result.perGenerationBest[g],
          mean: result.perGenerationMean[g],
          worst: result.perGenerationMean[g] * 1.5,
          generation: g,
          numFeasibleChildren: result.totalFeasibleEvaluated,
          numInfeasibleChildren: result.totalInfeasibleCut,
          precedenceCount: instance.precedence?.length ?? 0,
          arch: {generation: g, phase, cutThisGen: 0, acceptThisGen: 0},
        }));
      }
    }
    rec.setCharts(buildGeneticTSPCharts(gens, result.perGenerationBest, result.perGenerationMean));
    rec.finish();
    console.log(`# Animation written to ${htmlPath}`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
