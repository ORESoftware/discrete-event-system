'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/genetic-tsp.rs  (module des::general::genetic_tsp)
// 1:1 file move. Genetic algorithm for TSP modelled as a DES (each generation = one tick).
//
// Declarations → Rust:
//   interface TSPInstance / GASolverOptions / GenerationInfo / GASolverResult / GAPerformanceStats -> structs
//   type Tour = number[]          -> `type Tour = Vec<usize>;`
//   class GeneticTSPOptimizer extends PopulationOptimizer<Tour> -> struct + impl (base -> trait)
//   fn buildRandomTSP/buildPentagonTSP/tourLength/checkPrecedence/isPermutation/
//      tournamentSelect/orderCrossover/inversionMutate/swapMutate/repairPrecedence/
//      twoOptImprove/runGeneticTSP/heldKarpExact/oneTreeLowerBound -> assoc/free fns
//
// Conversion notes (file-specific):
//   - GA operators (select/crossover/mutate) take an `rng: () => number` closure (mulberry32)
//     -> inject `RandomSource`; vanilla operators map to PureTransform/assoc fns.
//   - `checkPrecedence` returns `[number, number] | null` -> `Option<(usize, usize)>`.
//   - `heldKarpExact` is bitmask DP -> index DP by `u32`/`usize` masks (no JS-number bit limits in Rust).
//   - branch-cut policy 'cut'|'penalize'|'repair' string union -> enum.
// =============================================================================

// =============================================================================
// general/genetic-tsp.ts — Genetic Algorithm for the Travelling Salesman
// Problem, modelled as a discrete-event system (every generation is a
// tick; selection, crossover, mutation, feasibility, fitness, replacement
// are stations; chromosomes are the movables flowing between them).
//
// THE PROBLEM
// ───────────
//   Given n cities at 2-D coordinates, find a permutation π of the cities
//   minimising  Σ_{i=0}^{n-1}  ‖coord(π_i) - coord(π_{i+1 mod n})‖
//   i.e. the closed Euclidean Hamiltonian cycle of minimum length.
//   Optionally, precedence constraints (must-visit-A-before-B) may
//   eliminate parts of the search space — that's where branch cutting
//   becomes meaningful.
//
// AS A DES
// ────────
//   Stations:
//     - PopulationStation       holds the current population of P tours
//     - SelectionStation        picks parents (tournament)
//     - CrossoverStation        Order-Crossover (OX), returns 1 child
//     - MutationStation         inversion / swap mutation
//     - FeasibilityStation      checks precedence constraints; cuts if violated
//     - FitnessStation          computes tour length → fitness
//     - ReplacementStation      μ + λ elitist + tournament-deletion
//
//   Movables:
//     - Chromosome (a tour)
//
//   Tick model:
//     One DES tick = one generation. Inside a tick the breeding pipeline
//     produces P_offspring chromosomes; ReplacementStation merges them
//     with the current population and keeps the best P. After every tick
//     we record (best, mean, worst) fitness and the elite tour for
//     animation.
//
// BRANCH CUTTING
// ──────────────
//   When precedence constraints are present, OX can produce children
//   that violate them. We support three policies:
//     'cut'      — silently drop infeasible children, retry up to retryLimit
//     'penalize' — accept but inflate tour length by P_penalty
//     'repair'   — rotate the violating segment until feasible (best-effort)
//   The validation runner shows that 'cut' converges to better mean tour
//   length than 'penalize' on instances with non-trivial constraint sets.
// =============================================================================

import {mulberry32} from './prng';
import {PureTransform} from '../shared/transform';
import {
  PopulationOptimizer, runIterativeDES,
  intrinsicCheck, monotonicityValidator,
} from './des-base';

export interface TSPInstance {
  n: number;
  /** 2-D coordinates, one per city. */
  coordinates: Array<[number, number]>;
  /** distance[i][j] = Euclidean distance between city i and city j. */
  distance: number[][];
  /** Optional precedence constraints: each (i, j) means "city i must appear
   *  somewhere before city j in the tour". Tours that violate any pair
   *  are infeasible. */
  precedence?: Array<[number, number]>;
}

export type Tour = number[];

// -----------------------------------------------------------------------------
// Instance builders
// -----------------------------------------------------------------------------

export interface BuildRandomTSPInput {
  n: number;
  seed?: number;
  precedence?: Array<[number, number]>;
}

export class BuildRandomTSP extends PureTransform<BuildRandomTSPInput, TSPInstance> {
  transform(input: BuildRandomTSPInput): TSPInstance {
    const n = input.n;
    const seed = input.seed ?? 42;
    const rng = mulberry32(seed);
    const coords: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) coords.push([rng() * 100, rng() * 100]);
    const dist: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        const dx = coords[i][0] - coords[j][0];
        const dy = coords[i][1] - coords[j][1];
        row.push(Math.sqrt(dx * dx + dy * dy));
      }
      dist.push(row);
    }
    return {n, coordinates: coords, distance: dist, precedence: input.precedence};
  }
}

/** @deprecated Use `new BuildRandomTSP().transform({n, seed, precedence})`. */
export function buildRandomTSP(n: number, seed: number = 42, opts: {
  precedence?: Array<[number, number]>;
} = {}): TSPInstance {
  return new BuildRandomTSP().transform({n, seed, precedence: opts.precedence});
}

export interface BuildPentagonTSPInput {
  n?: number;
  radius?: number;
}

/** A small, well-known TSP instance for unit testing: 5 cities arranged
 *  on a regular pentagon, where the optimal tour visits them in order
 *  with length n × side-length. */
export class BuildPentagonTSP extends PureTransform<BuildPentagonTSPInput, TSPInstance> {
  transform(input: BuildPentagonTSPInput): TSPInstance {
    const n = input.n ?? 5;
    const radius = input.radius ?? 50;
    const coords: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * i) / n;
      coords.push([50 + radius * Math.cos(a), 50 + radius * Math.sin(a)]);
    }
    const dist: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        const dx = coords[i][0] - coords[j][0];
        const dy = coords[i][1] - coords[j][1];
        row.push(Math.sqrt(dx * dx + dy * dy));
      }
      dist.push(row);
    }
    return {n, coordinates: coords, distance: dist};
  }
}

/** @deprecated Use `new BuildPentagonTSP().transform({n, radius})`. */
export function buildPentagonTSP(n: number = 5, radius: number = 50): TSPInstance {
  return new BuildPentagonTSP().transform({n, radius});
}

// -----------------------------------------------------------------------------
// Tour evaluation + feasibility
// -----------------------------------------------------------------------------

export interface InstanceTourInput { instance: TSPInstance; tour: Tour; }

export class TourLength extends PureTransform<InstanceTourInput, number> {
  transform(input: InstanceTourInput): number {
    const {instance, tour} = input;
    let s = 0;
    const n = instance.n;
    for (let i = 0; i < n; i++) s += instance.distance[tour[i]][tour[(i + 1) % n]];
    return s;
  }
}

/** @deprecated Use `new TourLength().transform({instance, tour})`. */
export function tourLength(instance: TSPInstance, tour: Tour): number {
  return new TourLength().transform({instance, tour});
}

/** Returns null if feasible, otherwise the violating (i, j) precedence pair. */
export class CheckPrecedence extends PureTransform<InstanceTourInput, [number, number] | null> {
  transform(input: InstanceTourInput): [number, number] | null {
    const {instance, tour} = input;
    if (!instance.precedence) return null;
    // Build position map: where does city c appear in the tour?
    const pos = new Array(instance.n).fill(-1);
    for (let i = 0; i < tour.length; i++) pos[tour[i]] = i;
    for (const [a, b] of instance.precedence) {
      if (pos[a] >= 0 && pos[b] >= 0 && pos[a] >= pos[b]) return [a, b];
    }
    return null;
  }
}

/** @deprecated Use `new CheckPrecedence().transform({instance, tour})`. */
export function checkPrecedence(instance: TSPInstance, tour: Tour): [number, number] | null {
  return new CheckPrecedence().transform({instance, tour});
}

export interface IsPermutationInput { tour: Tour; n: number; }

export class IsPermutation extends PureTransform<IsPermutationInput, boolean> {
  transform(input: IsPermutationInput): boolean {
    const {tour, n} = input;
    if (tour.length !== n) return false;
    const seen = new Set<number>();
    for (const c of tour) {
      if (c < 0 || c >= n) return false;
      if (seen.has(c)) return false;
      seen.add(c);
    }
    return true;
  }
}

/** @deprecated Use `new IsPermutation().transform({tour, n})`. */
export function isPermutation(tour: Tour, n: number): boolean {
  return new IsPermutation().transform({tour, n});
}

// -----------------------------------------------------------------------------
// GA OPERATORS (stations in the DES)
// -----------------------------------------------------------------------------

/** Tournament selection: sample `size` chromosomes uniformly at random from
 *  the population, return the index of the lowest-tour-length one. */
export interface TournamentSelectInput {
  populationLengths: number[];
  size: number;
  rng: () => number;
}

export class TournamentSelect extends PureTransform<TournamentSelectInput, number> {
  transform(input: TournamentSelectInput): number {
    const {populationLengths, size, rng} = input;
    let bestIdx = Math.floor(rng() * populationLengths.length);
    let bestLen = populationLengths[bestIdx];
    for (let k = 1; k < size; k++) {
      const idx = Math.floor(rng() * populationLengths.length);
      if (populationLengths[idx] < bestLen) { bestLen = populationLengths[idx]; bestIdx = idx; }
    }
    return bestIdx;
  }
}

/** @deprecated Use `new TournamentSelect().transform({populationLengths, size, rng})`. */
export function tournamentSelect(populationLengths: number[], size: number, rng: () => number): number {
  return new TournamentSelect().transform({populationLengths, size, rng});
}

/** Order-Crossover (OX) for TSP: copy a random sub-segment of parent 1
 *  to the child, then fill the remaining positions with parent 2's order
 *  (skipping any cities already in the child). The result is always a
 *  permutation. */
export interface OrderCrossoverInput { parent1: Tour; parent2: Tour; rng: () => number; }

export class OrderCrossover extends PureTransform<OrderCrossoverInput, Tour> {
  transform(input: OrderCrossoverInput): Tour {
    const {parent1, parent2, rng} = input;
    const n = parent1.length;
    let a = Math.floor(rng() * n);
    let b = Math.floor(rng() * n);
    if (a > b) [a, b] = [b, a];
    const child: Tour = new Array(n).fill(-1);
    const inChild = new Set<number>();
    for (let i = a; i <= b; i++) { child[i] = parent1[i]; inChild.add(parent1[i]); }
    let p2cursor = (b + 1) % n;
    let cursor = (b + 1) % n;
    while (cursor !== a) {
      while (inChild.has(parent2[p2cursor])) p2cursor = (p2cursor + 1) % n;
      child[cursor] = parent2[p2cursor];
      inChild.add(parent2[p2cursor]);
      p2cursor = (p2cursor + 1) % n;
      cursor = (cursor + 1) % n;
    }
    return child;
  }
}

/** @deprecated Use `new OrderCrossover().transform({parent1, parent2, rng})`. */
export function orderCrossover(parent1: Tour, parent2: Tour, rng: () => number): Tour {
  return new OrderCrossover().transform({parent1, parent2, rng});
}

/** Inversion mutation: pick a random sub-segment and reverse it. This is
 *  exactly a 2-opt move on the cycle; it preserves permutation-validity
 *  by construction. */
export interface MutateInput { tour: Tour; rng: () => number; }

export class InversionMutate extends PureTransform<MutateInput, Tour> {
  transform(input: MutateInput): Tour {
    const {tour, rng} = input;
    const n = tour.length;
    let a = Math.floor(rng() * n);
    let b = Math.floor(rng() * n);
    if (a > b) [a, b] = [b, a];
    const out = tour.slice();
    while (a < b) { [out[a], out[b]] = [out[b], out[a]]; a++; b--; }
    return out;
  }
}

/** @deprecated Use `new InversionMutate().transform({tour, rng})`. */
export function inversionMutate(tour: Tour, rng: () => number): Tour {
  return new InversionMutate().transform({tour, rng});
}

/** Swap mutation: pick two random positions, swap them. */
export class SwapMutate extends PureTransform<MutateInput, Tour> {
  transform(input: MutateInput): Tour {
    const {tour, rng} = input;
    const n = tour.length;
    const a = Math.floor(rng() * n);
    let b = Math.floor(rng() * n);
    while (b === a) b = Math.floor(rng() * n);
    const out = tour.slice();
    [out[a], out[b]] = [out[b], out[a]];
    return out;
  }
}

/** @deprecated Use `new SwapMutate().transform({tour, rng})`. */
export function swapMutate(tour: Tour, rng: () => number): Tour {
  return new SwapMutate().transform({tour, rng});
}

/** Repair attempt: if `tour` violates the precedence (i, j) with i later
 *  than j, swap them. Repeat at most `maxRounds` times. Returns the
 *  repaired tour and a feasibility flag (it may still be infeasible after
 *  the repair budget is exhausted). */
export interface RepairPrecedenceInput { instance: TSPInstance; tour: Tour; maxRounds?: number; }

export interface RepairPrecedenceResult { tour: Tour; feasible: boolean; }

export class RepairPrecedence extends PureTransform<RepairPrecedenceInput, RepairPrecedenceResult> {
  transform(input: RepairPrecedenceInput): RepairPrecedenceResult {
    const {instance, tour} = input;
    const maxRounds = input.maxRounds ?? 4;
    if (!instance.precedence) return {tour, feasible: true};
    const checker = new CheckPrecedence();
    const out = tour.slice();
    for (let r = 0; r < maxRounds; r++) {
      const v = checker.transform({instance, tour: out});
      if (v === null) return {tour: out, feasible: true};
      const [a, b] = v;
      const pa = out.indexOf(a), pb = out.indexOf(b);
      [out[pa], out[pb]] = [out[pb], out[pa]];
    }
    return {tour: out, feasible: checker.transform({instance, tour: out}) === null};
  }
}

/** @deprecated Use `new RepairPrecedence().transform({instance, tour, maxRounds})`. */
export function repairPrecedence(instance: TSPInstance, tour: Tour, maxRounds: number = 4): {
  tour: Tour; feasible: boolean;
} {
  return new RepairPrecedence().transform({instance, tour, maxRounds});
}

// -----------------------------------------------------------------------------
// GA SOLVER (= DES driver: each tick is one generation)
// -----------------------------------------------------------------------------

export interface GASolverOptions {
  populationSize?: number;     // default 100
  numGenerations?: number;     // default 200
  tournamentSize?: number;     // default 3
  crossoverProb?: number;      // default 0.95
  mutationProb?: number;       // default 0.3
  /** Number of top chromosomes copied unchanged to the next generation. */
  elitism?: number;            // default 2
  seed?: number;               // default 1
  /** Constraint-handling policy: */
  feasibility?: 'cut' | 'penalize' | 'repair';
  /** Penalty added to tour length per violated precedence pair. */
  penaltyPerViolation?: number;
  /** Max retries to produce a feasible child before giving up (cut). */
  retryLimit?: number;
  /** Initial population: 'random' (default) or 'nearest-neighbor' (heuristic). */
  init?: 'random' | 'nearest-neighbor';
  /** Optional memetic improvement applied to offspring after mutation. */
  localSearch?: 'none' | 'two-opt';
  /** Probability of applying local search to each offspring. */
  localSearchProb?: number;
  /** Number of first-improvement 2-opt passes per local-search application. */
  localSearchPasses?: number;
  /** Optional callback invoked after every generation with (gen, best, mean, eliteTour). */
  onGeneration?: (gen: number, info: GenerationInfo) => void;
}

export interface GenerationInfo {
  best: number;
  mean: number;
  worst: number;
  eliteTour: Tour;
  numFeasibleChildren: number;
  numInfeasibleChildren: number;
}

export interface GASolverResult {
  bestTour: Tour;
  bestLength: number;
  perGenerationBest: number[];
  perGenerationMean: number[];
  perGenerationElite: Tour[];        // for animation
  totalFeasibleEvaluated: number;
  totalInfeasibleCut: number;
  localSearchApplications: number;
  elapsedMs: number;
  performance: GAPerformanceStats;
  generations: number;
}

export interface GAPerformanceStats {
  elapsedMs: number;
  generationsPerSecond: number;
  estimatedEvaluations: number;
  evaluationsPerSecond: number;
  initialBest: number;
  finalBest: number;
  absoluteImprovement: number;
  relativeImprovement: number;
}

function nearestNeighborTour(instance: TSPInstance, start: number): Tour {
  const n = instance.n;
  const tour: Tour = [start];
  const visited = new Set<number>([start]);
  let cur = start;
  while (tour.length < n) {
    let bestNext = -1;
    let bestD = Infinity;
    for (let j = 0; j < n; j++) {
      if (visited.has(j)) continue;
      const d = instance.distance[cur][j];
      if (d < bestD) { bestD = d; bestNext = j; }
    }
    tour.push(bestNext);
    visited.add(bestNext);
    cur = bestNext;
  }
  return tour;
}

export interface TwoOptImproveInput { instance: TSPInstance; tour: Tour; maxPasses?: number; }

export class TwoOptImprove extends PureTransform<TwoOptImproveInput, Tour> {
  transform(input: TwoOptImproveInput): Tour {
    const {instance, tour} = input;
    const maxPasses = input.maxPasses ?? 1;
    const n = tour.length;
    if (n < 4 || maxPasses <= 0) return tour.slice();
    const out = tour.slice();
    const d = instance.distance;
    for (let pass = 0; pass < maxPasses; pass++) {
      let improved = false;
      for (let i = 0; i < n - 1 && !improved; i++) {
        const a = out[i];
        const b = out[(i + 1) % n];
        for (let k = i + 2; k < n; k++) {
          if (i === 0 && k === n - 1) continue;
          const c = out[k];
          const e = out[(k + 1) % n];
          const delta = d[a][c] + d[b][e] - d[a][b] - d[c][e];
          if (delta < -1e-12) {
            reverseSegment(out, i + 1, k);
            improved = true;
            break;
          }
        }
      }
      if (!improved) break;
    }
    return out;
  }
}

/** @deprecated Use `new TwoOptImprove().transform({instance, tour, maxPasses})`. */
export function twoOptImprove(instance: TSPInstance, tour: Tour, maxPasses: number = 1): Tour {
  return new TwoOptImprove().transform({instance, tour, maxPasses});
}

function reverseSegment(tour: Tour, lo: number, hi: number): void {
  while (lo < hi) {
    [tour[lo], tour[hi]] = [tour[hi], tour[lo]];
    lo++;
    hi--;
  }
}

// =============================================================================
// GeneticTSPOptimizer — concrete leaf of PopulationOptimizer<Tour>.
//
// Hooks:
//   initialPopulation → random or nearest-neighbour seeded
//   evaluate          → tourLength + precedence penalty (penalize feasibility)
//   select            → two tournament-selected parents
//   recombine         → orderCrossover with prob cxProb (else clone p1)
//   mutate            → inversion or swap with prob mutProb
//   clone             → tour.slice()
//   shouldStop        → gen ≥ numGenerations
//   acceptChild       → for 'cut' feasibility: keep retrying until child
//                        respects precedence
//   childRetryLimit   → opts.retryLimit when feasibility='cut', else 1
// =============================================================================

class GeneticTSPOptimizer extends PopulationOptimizer<Tour> {
  private readonly inst: TSPInstance;
  private readonly numGen: number;
  private readonly tournamentK: number;
  private readonly cxProb: number;
  private readonly mutProb: number;
  private readonly eliteN: number;
  private readonly initMode: 'random' | 'nearest-neighbor';
  private readonly feas: 'cut' | 'penalize' | 'repair';
  private readonly penalty: number;
  private readonly retryLimit: number;
  private readonly localSearch: 'none' | 'two-opt';
  private readonly localSearchProb: number;
  private readonly localSearchPasses: number;
  feasCount = 0;
  infeasCount = 0;
  localSearchApplications = 0;
  /** Per-generation history (matches the legacy GASolverResult fields). */
  readonly perGenBest: number[] = [];
  readonly perGenMean: number[] = [];
  readonly perGenElite: Tour[] = [];
  private readonly onGenerationCb?: (gen: number, info: GenerationInfo) => void;

  constructor(inst: TSPInstance, opts: Required<Omit<GASolverOptions, 'onGeneration'>> & {onGeneration?: GASolverOptions['onGeneration']}) {
    super('genetic-tsp', {popSize: opts.populationSize, rng: mulberry32(opts.seed)});
    this.inst = inst;
    this.numGen = opts.numGenerations;
    this.tournamentK = opts.tournamentSize;
    this.cxProb = opts.crossoverProb;
    this.mutProb = opts.mutationProb;
    this.eliteN = Math.min(opts.elitism, opts.populationSize);
    this.initMode = opts.init;
    this.feas = opts.feasibility;
    this.penalty = opts.penaltyPerViolation;
    this.retryLimit = opts.retryLimit;
    this.localSearch = opts.localSearch;
    this.localSearchProb = opts.localSearchProb;
    this.localSearchPasses = opts.localSearchPasses;
    this.onGenerationCb = opts.onGeneration;
    this.bootstrap();

    // Intrinsic invariants of the legacy GA driver.
    if (this.eliteN >= 1) {
      this.addValidator(monotonicityValidator<GeneticTSPOptimizer>({
        name: 'genetic-tsp.bestHistory.monotone',
        group: 'genetic-tsp-intrinsic',
        extract: st => st.bestHistory,
        direction: 'non-increasing',
      }));
    }
    this.addValidator(intrinsicCheck<GeneticTSPOptimizer>({
      name: 'genetic-tsp.best-is-valid-permutation',
      group: 'genetic-tsp-intrinsic',
      predicate: st => isPermutation(st.getBest(), st.inst.n),
      expected: 'permutation of [0..n-1]',
      observedFn: st => `n=${st.inst.n}  bestLen=${st.getBest().length}`,
    }));
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected initialPopulation(size: number, rng: () => number): Tour[] {
    const out: Tour[] = [];
    if (this.initMode === 'nearest-neighbor') {
      const seedCount = Math.min(this.inst.n, size);
      for (let i = 0; i < seedCount; i++) out.push(nearestNeighborTour(this.inst, i));
      while (out.length < size) {
        const t: Tour = [];
        const remaining = Array.from({length: this.inst.n}, (_, i) => i);
        while (remaining.length > 0) t.push(remaining.splice(Math.floor(rng() * remaining.length), 1)[0]);
        out.push(t);
      }
      return out;
    }
    for (let p = 0; p < size; p++) {
      const t = Array.from({length: this.inst.n}, (_, i) => i);
      for (let i = t.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [t[i], t[j]] = [t[j], t[i]];
      }
      out.push(t);
    }
    return out;
  }

  protected evaluate(t: Tour): number {
    let len = tourLength(this.inst, t);
    if (this.feas === 'penalize') {
      const v = checkPrecedence(this.inst, t);
      if (v !== null) len += this.penalty;
    }
    return len;
  }

  protected select(_pop: readonly Tour[], fitness: readonly number[], rng: () => number): Tour[] {
    const i1 = tournamentSelect(fitness as number[], this.tournamentK, rng);
    const i2 = tournamentSelect(fitness as number[], this.tournamentK, rng);
    return [this.population[i1], this.population[i2]];
  }

  protected recombine(parents: readonly Tour[], rng: () => number): Tour {
    if (rng() < this.cxProb) return orderCrossover(parents[0], parents[1], rng);
    return parents[0].slice();
  }

  protected mutate(child: Tour, rng: () => number): Tour {
    let out = child;
    if (rng() < this.mutProb) {
      out = rng() < 0.5 ? inversionMutate(child, rng) : swapMutate(child, rng);
    }
    if (this.localSearch === 'two-opt' && rng() < this.localSearchProb) {
      const improved = twoOptImprove(this.inst, out, this.localSearchPasses);
      if (tourLength(this.inst, improved) < tourLength(this.inst, out) - 1e-12) {
        this.localSearchApplications++;
      }
      out = improved;
    }
    return out;
  }

  protected clone(t: Tour): Tour { return t.slice(); }
  protected shouldStop(gen: number): boolean { return gen >= this.numGen; }
  protected override eliteCount(): number { return this.eliteN; }

  /** Constraint hook for feasibility='cut': only accept a child if it
   *  respects the precedence DAG. For 'repair' we patch the child in
   *  place; for 'penalize' we always accept (penalty handled in evaluate). */
  protected override acceptChild(child: Tour): boolean {
    if (!this.inst.precedence || this.inst.precedence.length === 0) {
      this.feasCount += 1;
      return true;
    }
    if (this.feas === 'penalize') {
      this.feasCount += 1;
      return true;
    }
    if (this.feas === 'repair') {
      const r = repairPrecedence(this.inst, child);
      // mutate-style in-place patch.
      for (let i = 0; i < r.tour.length; i++) child[i] = r.tour[i];
      if (r.feasible) { this.feasCount += 1; return true; }
      this.infeasCount += 1;
      return true;        // accept the partial-repair child anyway
    }
    // 'cut' — accept iff feasible.
    if (checkPrecedence(this.inst, child) === null) { this.feasCount += 1; return true; }
    return false;
  }

  protected override childRetryLimit(): number {
    if (!this.inst.precedence || this.inst.precedence.length === 0) return 1;
    return this.feas === 'cut' ? Math.max(1, this.retryLimit) : 1;
  }

  protected override onChildRejected(_child: Tour, attempt: number): void {
    // Only count as "infeasible cut" when the entire retry budget was used up.
    if (attempt >= this.childRetryLimit()) this.infeasCount += 1;
  }

  protected override onGeneration(_gen: number): void {
    const lengths = this.fitness;
    const minLen = Math.min(...lengths);
    const meanLen = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const maxLen = Math.max(...lengths);
    const eliteIdx = lengths.indexOf(minLen);
    this.perGenBest.push(minLen);
    this.perGenMean.push(meanLen);
    this.perGenElite.push(this.population[eliteIdx].slice());
    this.onGenerationCb?.(_gen - 1, {
      best: minLen, mean: meanLen, worst: maxLen, eliteTour: this.population[eliteIdx].slice(),
      numFeasibleChildren: this.feasCount, numInfeasibleChildren: this.infeasCount,
    });
  }
}

/** Run the GA. Internally orchestrated by GeneticTSPOptimizer running on
 *  the runIterativeDES engine; each tick is exactly one generation. */
export function runGeneticTSP(
  instance: TSPInstance,
  options: GASolverOptions = {},
): GASolverResult {
  const t0 = Date.now();
  const filled = {
    populationSize: options.populationSize ?? 100,
    numGenerations: options.numGenerations ?? 200,
    tournamentSize: options.tournamentSize ?? 3,
    crossoverProb:  options.crossoverProb ?? 0.95,
    mutationProb:   options.mutationProb ?? 0.3,
    elitism:        options.elitism ?? 2,
    seed:           options.seed ?? 1,
    feasibility:    (options.feasibility ?? 'cut') as 'cut' | 'penalize' | 'repair',
    penaltyPerViolation: options.penaltyPerViolation ?? 1e6,
    retryLimit:     options.retryLimit ?? 8,
    init:           (options.init ?? 'random') as 'random' | 'nearest-neighbor',
    localSearch:    (options.localSearch ?? 'none') as 'none' | 'two-opt',
    localSearchProb: options.localSearchProb ?? 1,
    localSearchPasses: options.localSearchPasses ?? 1,
    onGeneration:   options.onGeneration,
  };
  const opt = new GeneticTSPOptimizer(instance, filled);
  runIterativeDES([opt]);
  const elapsedMs = Date.now() - t0;
  const best = opt.getBest();
  const estimatedEvaluations = filled.populationSize * (opt.getGeneration() + 1);
  const initialBest = opt.perGenBest[0] ?? opt.getBestFitness();
  const finalBest = tourLength(instance, best);
  const absoluteImprovement = initialBest - finalBest;
  return {
    bestTour: best.slice(),
    bestLength: finalBest,
    perGenerationBest: opt.perGenBest,
    perGenerationMean: opt.perGenMean,
    perGenerationElite: opt.perGenElite,
    totalFeasibleEvaluated: opt.feasCount,
    totalInfeasibleCut: opt.infeasCount,
    localSearchApplications: opt.localSearchApplications,
    elapsedMs,
    performance: {
      elapsedMs,
      generationsPerSecond: opt.getGeneration() / Math.max(elapsedMs / 1000, 1e-9),
      estimatedEvaluations,
      evaluationsPerSecond: estimatedEvaluations / Math.max(elapsedMs / 1000, 1e-9),
      initialBest,
      finalBest,
      absoluteImprovement,
      relativeImprovement: absoluteImprovement / Math.max(Math.abs(initialBest), 1e-9),
    },
    generations: opt.getGeneration(),
  };
}

// -----------------------------------------------------------------------------
// LOWER BOUNDS / REFERENCE SOLVERS
// -----------------------------------------------------------------------------

/** Held–Karp dynamic-programming exact solver for small TSPs (n ≤ 16).
 *  Bitmask DP over (visited subset, current city). Returns optimal tour
 *  and length. Used to validate the GA on small instances. */
export interface HeldKarpResult { tour: Tour; length: number; }

export class HeldKarpExact extends PureTransform<TSPInstance, HeldKarpResult> {
  transform(instance: TSPInstance): HeldKarpResult {
  const n = instance.n;
  if (n > 16) throw new Error(`Held–Karp only practical for n ≤ 16, got ${n}`);
  const N = 1 << n;
  // dp[mask][i] = (min length of path starting at 0, visiting exactly `mask`, ending at i)
  // We store as Float64Array packed by (mask * n + i).
  const dp = new Float64Array(N * n).fill(Infinity);
  const parent = new Int32Array(N * n).fill(-1);
  dp[(1) * n + 0] = 0;       // start at city 0, mask = {0}
  for (let mask = 1; mask < N; mask++) {
    if (!(mask & 1)) continue;
    for (let i = 0; i < n; i++) {
      if (!(mask & (1 << i))) continue;
      const cur = dp[mask * n + i];
      if (!isFinite(cur)) continue;
      for (let j = 0; j < n; j++) {
        if (mask & (1 << j)) continue;
        const newMask = mask | (1 << j);
        const cand = cur + instance.distance[i][j];
        if (cand < dp[newMask * n + j]) {
          dp[newMask * n + j] = cand;
          parent[newMask * n + j] = i;
        }
      }
    }
  }
  const fullMask = N - 1;
  let bestEnd = 1;
  let bestLen = Infinity;
  for (let i = 1; i < n; i++) {
    const cand = dp[fullMask * n + i] + instance.distance[i][0];
    if (cand < bestLen) { bestLen = cand; bestEnd = i; }
  }
  // Reconstruct
  const tour: number[] = [];
  let mask = fullMask;
  let cur = bestEnd;
  while (cur !== -1) {
    tour.push(cur);
    const prev = parent[mask * n + cur];
    mask ^= (1 << cur);
    cur = prev;
  }
  tour.reverse();
  return {tour, length: bestLen};
  }
}

/** @deprecated Use `new HeldKarpExact().transform(instance)`. */
export function heldKarpExact(instance: TSPInstance): HeldKarpResult {
  return new HeldKarpExact().transform(instance);
}

/** 1-tree lower bound (Held–Karp's relaxation cousin; cheap): the cost of
 *  a minimum spanning tree on cities {1, …, n-1} plus the two cheapest
 *  edges from city 0 is a valid lower bound on the optimal TSP cost. */
export class OneTreeLowerBound extends PureTransform<TSPInstance, number> {
  transform(instance: TSPInstance): number {
  const n = instance.n;
  // Prim's MST on {1, …, n-1}.
  const inTree = new Array(n).fill(false);
  const minEdge = new Array(n).fill(Infinity);
  inTree[0] = true;          // exclude city 0
  let mstCost = 0;
  // Initialise frontier from city 1.
  if (n < 2) return 0;
  inTree[1] = true;
  for (let j = 2; j < n; j++) minEdge[j] = instance.distance[1][j];
  for (let count = 1; count < n - 1; count++) {
    let best = -1;
    let bestVal = Infinity;
    for (let j = 2; j < n; j++) {
      if (!inTree[j] && minEdge[j] < bestVal) { bestVal = minEdge[j]; best = j; }
    }
    if (best === -1) break;
    mstCost += bestVal;
    inTree[best] = true;
    for (let k = 2; k < n; k++) {
      if (!inTree[k] && instance.distance[best][k] < minEdge[k])
        minEdge[k] = instance.distance[best][k];
    }
  }
  // Two cheapest edges from city 0.
  const edges0 = [];
  for (let j = 1; j < n; j++) edges0.push(instance.distance[0][j]);
  edges0.sort((a, b) => a - b);
  return mstCost + (edges0[0] ?? 0) + (edges0[1] ?? 0);
  }
}

/** @deprecated Use `new OneTreeLowerBound().transform(instance)`. */
export function oneTreeLowerBound(instance: TSPInstance): number {
  return new OneTreeLowerBound().transform(instance);
}
