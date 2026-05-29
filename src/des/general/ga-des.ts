'use strict';

// =============================================================================
// general/ga-des.ts — Genetic Algorithm as a DES, built on the
// PopulationOptimizer<I> base class. Concrete leaf implements ONLY the
// hooks; the per-generation breeding loop is the base's template method.
// =============================================================================

import {
  PopulationOptimizer, PopulationSinkStation, PopulationSourceStation,
  runIterativeDES, IterativeRunOptions,
  intrinsicCheck, monotonicityValidator,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {
  TSPInstance, Tour, tourLength,
  tournamentSelect, orderCrossover, inversionMutate, swapMutate,
  isPermutation, heldKarpExact,
} from './genetic-tsp';
import {mulberry32} from './prng';

export interface TSPGAOptions {
  popSize: number;
  numGenerations: number;
  tournamentSize?: number;
  crossoverProb?: number;
  mutationProb?: number;
  elitism?: number;
  seed: number;
  init?: 'random' | 'nearest-neighbor';
  /** Steepness of fitness inverse — lower = flatter selection pressure. */
  penaltyPerViolation?: number;
}

export class TSPGAOptimizer extends PopulationOptimizer<Tour> {
  private readonly inst: TSPInstance;
  private readonly numGenerations: number;
  private readonly tournamentK: number;
  private readonly crossoverProb: number;
  private readonly mutationProb: number;
  private readonly elite: number;
  private readonly initMode: 'random' | 'nearest-neighbor';
  private readonly penalty: number;

  constructor(
    id: string,
    inst: TSPInstance,
    opts: TSPGAOptions,
    lifecycle: {deferBootstrap?: boolean; rng?: () => number} = {},
  ) {
    super(id, {popSize: opts.popSize, rng: lifecycle.rng ?? mulberry32(opts.seed)});
    this.inst = inst;
    this.numGenerations = opts.numGenerations;
    this.tournamentK = opts.tournamentSize ?? 3;
    this.crossoverProb = opts.crossoverProb ?? 0.95;
    this.mutationProb = opts.mutationProb ?? 0.3;
    this.elite = Math.min(opts.elitism ?? 2, opts.popSize);
    this.initMode = opts.init ?? 'random';
    this.penalty = opts.penaltyPerViolation ?? 1e6;
    if (!lifecycle.deferBootstrap) this.bootstrap();

    // ── Intrinsic invariants ─────────────────────────────────────────────
    // With elitism ≥ 1, best-so-far history is monotone non-increasing.
    if (this.elite >= 1) {
      this.addValidator(monotonicityValidator<TSPGAOptimizer>({
        name: 'ga.bestHistory.monotone',
        group: 'ga-intrinsic',
        extract: st => st.bestHistory,
        direction: 'non-increasing',
      }));
    }
    // best is a valid permutation of n cities.
    this.addValidator(intrinsicCheck<TSPGAOptimizer>({
      name: 'ga.best-is-valid-permutation',
      group: 'ga-intrinsic',
      predicate: st => isPermutation(st.getBestTour(), st.inst.n),
      expected: 'permutation of [0..n-1]',
      observedFn: st => `n=${st.inst.n}  bestLen=${st.getBestTour().length}`,
    }));

    // ── Ground-truth: Held-Karp lower bound (small instances only) ───────
    if (inst.n <= 12 && !inst.precedence) {
      let exactLen: number | null = null;
      this.addValidator(intrinsicCheck<TSPGAOptimizer>({
        name: 'ga.bestLength-vs-heldKarp-LB',
        group: 'ga-ground-truth',
        predicate: st => {
          if (exactLen === null) exactLen = heldKarpExact(st.inst).length;
          return st.getBestLength() >= exactLen - 1e-9;
        },
        expected: 'bestLength ≥ heldKarp.length',
        observedFn: st => {
          if (exactLen === null) exactLen = heldKarpExact(st.inst).length;
          return `best=${st.getBestLength().toFixed(4)}  heldKarp=${exactLen.toFixed(4)}`;
        },
        details: 'best length is below the global optimum',
      }));
    }
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected initialPopulation(size: number, rng: () => number): Tour[] {
    return initialTourPopulation(this.inst, size, this.initMode, rng);
  }

  private nearestNeighbor(start: number): Tour {
    const n = this.inst.n;
    const tour: Tour = [start];
    const seen = new Set<number>([start]);
    let cur = start;
    while (tour.length < n) {
      let bestNext = -1; let bestD = Infinity;
      for (let j = 0; j < n; j++) {
        if (seen.has(j)) continue;
        const d = this.inst.distance[cur][j];
        if (d < bestD) { bestD = d; bestNext = j; }
      }
      tour.push(bestNext); seen.add(bestNext); cur = bestNext;
    }
    return tour;
  }

  protected evaluate(tour: Tour): number {
    let c = tourLength(this.inst, tour);
    if (this.inst.precedence) {
      for (const [a, b] of this.inst.precedence) {
        let posA = -1, posB = -1;
        for (let i = 0; i < tour.length; i++) {
          if (tour[i] === a) posA = i;
          if (tour[i] === b) posB = i;
        }
        if (posA >= 0 && posB >= 0 && posA >= posB) c += this.penalty;
      }
    }
    return c;
  }

  /** Tournament selection of TWO parents (for one offspring). */
  protected select(_pop: readonly Tour[], fitness: readonly number[], rng: () => number): Tour[] {
    const lengths = fitness as readonly number[];
    const i = tournamentSelect(lengths as number[], this.tournamentK, rng);
    let j = tournamentSelect(lengths as number[], this.tournamentK, rng);
    let tries = 0;
    while (j === i && tries < 8) { j = tournamentSelect(lengths as number[], this.tournamentK, rng); tries++; }
    return [this.population[i], this.population[j]];
  }

  /** Order crossover with probability `crossoverProb`; otherwise clone p1. */
  protected recombine(parents: readonly Tour[], rng: () => number): Tour {
    if (rng() < this.crossoverProb) return orderCrossover(parents[0], parents[1], rng);
    return parents[0].slice();
  }

  /** Apply mutation with `mutationProb`. Use inversion (preserves perm). */
  protected mutate(child: Tour, rng: () => number): Tour {
    if (rng() < this.mutationProb) {
      // Mix of inversion and swap.
      return rng() < 0.6 ? inversionMutate(child, rng) : swapMutate(child, rng);
    }
    return child;
  }

  protected clone(t: Tour): Tour { return t.slice(); }

  protected shouldStop(gen: number): boolean { return gen >= this.numGenerations; }

  protected override eliteCount(): number { return this.elite; }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getBestTour(): Tour { return this.getBest(); }
  getBestLength(): number { return this.getBestFitness(); }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

export interface GADESResult {
  bestTour: Tour;
  bestLength: number;
  generations: number;
  bestHistory: readonly number[];
  meanHistory: readonly number[];
  ticks: number;
}

export function runTSPGADES(
  inst: TSPInstance, opts: TSPGAOptions,
  desOptions?: IterativeRunOptions,
): GADESResult {
  const rng = mulberry32(opts.seed);
  const initMode = opts.init ?? 'random';
  const source = new PopulationSourceStation<Tour>(
    'ga-source',
    () => initialTourPopulation(inst, opts.popSize, initMode, rng),
    population => validateInitialTourPopulation('ga-source', inst, opts.popSize, population),
  );
  const opt = new TSPGAOptimizer('ga', inst, opts, {deferBootstrap: true, rng});
  const sink = new PopulationSinkStation<Tour>('ga-sink');
  source.pipe(opt, PopulationSourceStation.CH_INITIAL_POPULATION, PopulationOptimizer.CH_INITIAL_POPULATION);
  opt.pipe(sink, PopulationOptimizer.CH_RESULT, PopulationSinkStation.CH_RESULT);
  const summary = runIterativeDES([source, opt, sink], {rng, ...desOptions, shuffle: desOptions?.shuffle ?? false});
  const result = sink.latest?.snapshot;
  if (!result) throw new Error('ga-des: result sink did not receive a final population');
  const best = result.best;
  if (!isPermutation(best, inst.n)) throw new Error('ga-des: best is not a valid permutation');
  return {
    bestTour: best, bestLength: result.bestFitness,
    generations: result.generation,
    bestHistory: opt.bestHistory,
    meanHistory: opt.meanHistory,
    ticks: summary.ticks,
  };
}

function initialTourPopulation(
  inst: TSPInstance,
  size: number,
  initMode: 'random' | 'nearest-neighbor',
  rng: () => number,
): Tour[] {
  const n = inst.n;
  const out: Tour[] = [];
  if (initMode === 'nearest-neighbor') {
    const k = Math.min(n, size);
    for (let s = 0; s < k; s++) out.push(nearestNeighborTour(inst, s));
    while (out.length < size) out.push(randomTour(n, rng));
    return out;
  }
  for (let p = 0; p < size; p++) out.push(randomTour(n, rng));
  return out;
}

function nearestNeighborTour(inst: TSPInstance, start: number): Tour {
  const n = inst.n;
  const tour: Tour = [start];
  const seen = new Set<number>([start]);
  let cur = start;
  while (tour.length < n) {
    let bestNext = -1; let bestD = Infinity;
    for (let j = 0; j < n; j++) {
      if (seen.has(j)) continue;
      const d = inst.distance[cur][j];
      if (d < bestD) { bestD = d; bestNext = j; }
    }
    tour.push(bestNext); seen.add(bestNext); cur = bestNext;
  }
  return tour;
}

function randomTour(n: number, rng: () => number): Tour {
  const t: Tour = Array.from({length: n}, (_, i) => i);
  for (let i = t.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [t[i], t[j]] = [t[j], t[i]];
  }
  return t;
}

function validateInitialTourPopulation(
  sourceId: string,
  inst: TSPInstance,
  expectedSize: number,
  population: readonly Tour[],
): void {
  Preconditions.lengthEq(sourceId, 'initial population', population, expectedSize);
  for (let i = 0; i < population.length; i++) {
    Preconditions.check(sourceId, `initial population[${i}]`, `be a permutation of ${inst.n} cities`,
      isPermutation(population[i], inst.n), population[i]);
    Preconditions.finite(sourceId, `initial population[${i}] length`, tourLength(inst, population[i]));
  }
}
