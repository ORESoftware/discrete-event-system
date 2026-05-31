// RUST MIGRATION: Target module `src/des/general/sa_des.rs`.
// RUST MIGRATION: Port simulated-annealing DES leaf classes as structs implementing the `SingleStateOptimizer` trait/template hooks.
// RUST MIGRATION: Convert optimizer options, tick events, and run results to `serde` structs; discriminated problem-specific moves should become enums.
// RUST MIGRATION: Keep DES runner/builders as free functions, but expose graph-visible optimizer steps through trait impls rather than closure-heavy helpers.
// RUST MIGRATION: Inject RNG for candidate generation/acceptance and return `Result` for invalid temperature schedules, dimensions, or TSP/knapsack inputs.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/sa-des.rs  (module des::general::sa_des)
// 1:1 file move. Simulated annealing (and hill-climbing) over TSP tours as a DES.
//
// Declarations → Rust:
//   type CoolingSchedule = {kind:'geometric'|...}  -> enum CoolingSchedule { Geometric{..}, ... }
//   fn temperatureAt(s, k)                          -> match on the CoolingSchedule enum
//   interface TSPSAOptions / SADESResult            -> structs
//   class TSPSAOptimizer extends SingleStateOptimizer<Tour> -> struct + impl optimizer trait
//   class TSPHillClimber extends TSPSAOptimizer     -> struct reusing the SA struct (see note)
//   fn runTSPSADES / runTSPHillClimberDES / initialTour / validateInitialTour -> fns
//
// Conversion notes (file-specific):
//   - CoolingSchedule is a DISCRIMINATED UNION on `kind` -> `enum` with per-variant fields,
//     matched in `temperatureAt`.
//   - INHERITANCE: TSPHillClimber EXTENDS TSPSAOptimizer (class-on-class). Rust has no
//     inheritance: factor the shared SA template into a trait/base struct and have HillClimber
//     override only the `accept` rule (compose, don't extend).
//   - INJECT RNG: `mulberry32` + initialTour('random') + Metropolis accept -> `RandomSource`.
//   - SingleStateOptimizer is a template-method base -> trait with default driver fn.
//   - depends on genetic-tsp.ts (Tour/TSPInstance/tourLength/...) -> use crate::...::genetic_tsp.
//   - validateInitialTour throws -> `panic!` (invariant) or `Result`.
// =============================================================================
// general/sa-des.ts — Simulated Annealing as a DES, built on the
// SingleStateOptimizer<S> base class. Concrete leaf classes implement
// ONLY the hooks; the iteration loop is the base's template method.
//
// Two leaves:
//   • TSPSAOptimizer — SA on a TSPInstance with 2-opt + or-opt moves.
//   • HillClimber<S>  — same template, different `accept` rule.
// =============================================================================

import {
  SingleStateOptimizer, SingleStateSinkStation, SingleStateSourceStation,
  runIterativeDES, IterativeRunOptions,
  intrinsicCheck, monotonicityValidator, numericValidator,
} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {TSPInstance, Tour, tourLength, isPermutation, heldKarpExact} from './genetic-tsp';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// COOLING SCHEDULES
// -----------------------------------------------------------------------------

export type CoolingSchedule =
  | {kind: 'geometric'; T0: number; alpha: number; Tmin?: number}
  | {kind: 'logarithmic'; T0: number; Tmin?: number}
  | {kind: 'linear'; T0: number; rate: number; Tmin?: number}
  | {kind: 'exp-restart'; T0: number; alpha: number; period: number; Tmin?: number};

export function temperatureAt(s: CoolingSchedule, k: number): number {
  switch (s.kind) {
    case 'geometric':   return Math.max(s.Tmin ?? 0, s.T0 * Math.pow(s.alpha, k));
    case 'logarithmic': return Math.max(s.Tmin ?? 0, s.T0 / Math.log(2 + k));
    case 'linear':      return Math.max(s.Tmin ?? 0, s.T0 - s.rate * k);
    case 'exp-restart': return Math.max(s.Tmin ?? 0, s.T0 * Math.pow(s.alpha, k % s.period));
  }
}

// -----------------------------------------------------------------------------
// TSP — SA leaf
// -----------------------------------------------------------------------------

export interface TSPSAOptions {
  cooling: CoolingSchedule;
  maxIterations: number;
  seed: number;
  init?: 'random' | 'nearest-neighbor';
  moves?: '2-opt' | 'or-opt' | 'mixed';
  /** Cost penalty per violated precedence pair. */
  penaltyPerViolation?: number;
  traceStride?: number;
  /** Stop after this many iterations without best improvement. 0 = off. */
  stallLimit?: number;
}

export class TSPSAOptimizer extends SingleStateOptimizer<Tour> {
  private readonly inst: TSPInstance;
  private readonly cooling: CoolingSchedule;
  private readonly maxIters: number;
  private readonly initMode: 'random' | 'nearest-neighbor';
  private readonly moves: '2-opt' | 'or-opt' | 'mixed';
  private readonly penalty: number;
  private readonly stallLimit: number;
  private stallSince = 0;
  private bestSeen = Infinity;

  constructor(
    id: string,
    inst: TSPInstance,
    opts: TSPSAOptions,
    lifecycle: {deferBootstrap?: boolean; rng?: () => number} = {},
  ) {
    super(id, {rng: lifecycle.rng ?? mulberry32(opts.seed), traceStride: opts.traceStride});
    this.inst = inst;
    this.cooling = opts.cooling;
    this.maxIters = opts.maxIterations;
    this.initMode = opts.init ?? 'nearest-neighbor';
    this.moves = opts.moves ?? 'mixed';
    this.penalty = opts.penaltyPerViolation ?? 1e6;
    this.stallLimit = opts.stallLimit ?? 0;
    // bootstrap MUST be called by every concrete subclass after super().
    if (!lifecycle.deferBootstrap) this.bootstrap();

    // ── Intrinsic invariants ─────────────────────────────────────────────
    // Best-so-far history is monotone non-increasing by definition of SA.
    this.addValidator(monotonicityValidator<TSPSAOptimizer>({
      name: 'sa.bestHistory.monotone',
      group: 'sa-intrinsic',
      extract: st => st.bestHistory,
      direction: 'non-increasing',
    }));
    // Best is a valid permutation of n cities.
    this.addValidator(intrinsicCheck<TSPSAOptimizer>({
      name: 'sa.best-is-valid-permutation',
      group: 'sa-intrinsic',
      predicate: st => isPermutation(st.getBest(), st.inst.n),
      expected: 'permutation of [0..n-1]',
      observedFn: st => `n=${st.inst.n}  bestLen=${st.getBest().length}`,
    }));
    // bestCost ≥ 0 for any valid Euclidean TSP.
    this.addValidator(intrinsicCheck<TSPSAOptimizer>({
      name: 'sa.best-cost-nonnegative',
      group: 'sa-intrinsic',
      predicate: st => st.getBestCost() >= 0,
      expected: '≥ 0',
      observedFn: st => `bestCost=${st.getBestCost()}`,
    }));

    // ── Ground-truth validator: Held-Karp exact lower bound ──────────────
    // Only auto-attached for small instances where DP is tractable. For
    // a feasible permutation, bestCost ≥ heldKarp.length is a HARD
    // invariant of the problem.
    if (inst.n <= 12 && !inst.precedence) {
      let exactLen: number | null = null;
      this.addValidator(intrinsicCheck<TSPSAOptimizer>({
        name: 'sa.bestCost-vs-heldKarp-LB',
        group: 'sa-ground-truth',
        predicate: st => {
          if (exactLen === null) exactLen = heldKarpExact(st.inst).length;
          return st.getBestCost() >= exactLen - 1e-9;
        },
        expected: 'bestCost ≥ heldKarp.length',
        observedFn: st => {
          if (exactLen === null) exactLen = heldKarpExact(st.inst).length;
          return `bestCost=${st.getBestCost().toFixed(4)}  heldKarp=${exactLen.toFixed(4)}`;
        },
        details: 'bestCost is below the true global optimum — would indicate a bug',
      }));
    }
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected initialState(rng: () => number): Tour {
    return initialTour(this.inst, this.initMode, rng);
  }

  protected cost(tour: Tour): number {
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

  protected propose(tour: Tour, rng: () => number): Tour {
    const move = this.moves === 'mixed' ? (rng() < 0.7 ? '2-opt' : 'or-opt') : this.moves;
    const n = tour.length;
    if (move === '2-opt') {
      let i = Math.floor(rng() * n);
      let j = Math.floor(rng() * n);
      if (i > j) [i, j] = [j, i];
      if (j - i < 1) j = Math.min(n - 1, i + 1);
      const next = tour.slice();
      for (let a = i, b = j; a < b; a++, b--) {
        const tmp = next[a]; next[a] = next[b]; next[b] = tmp;
      }
      return next;
    }
    // or-opt: extract a sub-segment of length L ∈ {1,2,3} and reinsert.
    const L = 1 + Math.floor(rng() * 3);
    if (L >= n) return tour.slice();
    const i = Math.floor(rng() * (n - L + 1));
    const seg = tour.slice(i, i + L);
    const remain = tour.slice(0, i).concat(tour.slice(i + L));
    const insertAt = Math.floor(rng() * (remain.length + 1));
    if (insertAt === i) return tour.slice();
    return remain.slice(0, insertAt).concat(seg).concat(remain.slice(insertAt));
  }

  /** Metropolis acceptance with the configured cooling schedule. */
  protected accept(
    _current: Tour, _candidate: Tour,
    currentCost: number, candidateCost: number,
    iter: number, rng: () => number,
  ): boolean {
    const delta = candidateCost - currentCost;
    if (delta <= 0) return true;
    const T = temperatureAt(this.cooling, iter);
    if (T <= 0) return false;
    return rng() < Math.exp(-delta / T);
  }

  protected clone(t: Tour): Tour { return t.slice(); }

  protected shouldStop(iter: number): boolean {
    if (iter >= this.maxIters) return true;
    if (this.stallLimit > 0 && this.stallSince >= this.stallLimit) return true;
    return false;
  }

  // ── HOOKS (optional override) — track stall ─────────────────────────────

  protected onAccept(_c: Tour, _delta: number, _iter: number): void {
    if (this.bestCost < this.bestSeen) {
      this.bestSeen = this.bestCost; this.stallSince = 0;
    } else {
      this.stallSince += 1;
    }
  }
  protected onReject(_c: Tour, _delta: number, _iter: number): void {
    this.stallSince += 1;
  }

  protected override onBootstrap(): void {
    this.bestSeen = this.bestCost;
  }
}

// -----------------------------------------------------------------------------
// HILL CLIMBER — same template, different acceptance.
// -----------------------------------------------------------------------------

export class TSPHillClimber extends TSPSAOptimizer {
  /** Hill climbing: accept ONLY strict improvements. */
  protected override accept(
    _c: Tour, _cand: Tour,
    cur: number, candCost: number,
    _iter: number, _rng: () => number,
  ): boolean {
    return candCost < cur;
  }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

export interface SADESResult {
  bestTour: Tour;
  bestCost: number;
  iterations: number;
  acceptedCount: number;
  improveCount: number;
  bestHistory: readonly number[];
  currentHistory: readonly number[];
  ticks: number;
}

export function runTSPSADES(
  inst: TSPInstance, opts: TSPSAOptions,
  desOptions?: IterativeRunOptions,
): SADESResult {
  const rng = mulberry32(opts.seed);
  const source = new SingleStateSourceStation<Tour>(
    'sa-source',
    () => initialTour(inst, opts.init ?? 'nearest-neighbor', rng),
    tour => validateInitialTour('sa-source', inst, tour),
  );
  const opt = new TSPSAOptimizer('sa', inst, opts, {deferBootstrap: true, rng});
  const sink = new SingleStateSinkStation<Tour>('sa-sink');
  source.pipe(opt, SingleStateSourceStation.CH_INITIAL_STATE, SingleStateOptimizer.CH_INITIAL_STATE);
  opt.pipe(sink, SingleStateOptimizer.CH_RESULT, SingleStateSinkStation.CH_RESULT);
  const summary = runIterativeDES([source, opt, sink], {rng, ...desOptions, shuffle: desOptions?.shuffle ?? false});
  const result = sink.latest?.snapshot;
  if (!result) throw new Error('sa-des: result sink did not receive a final tour');
  const best = result.best;
  if (!isPermutation(best, inst.n)) throw new Error('sa-des: best is not a valid permutation');
  return {
    bestTour: best, bestCost: result.bestCost,
    iterations: result.iteration,
    acceptedCount: result.acceptedCount,
    improveCount: result.improveCount,
    bestHistory: opt.bestHistory,
    currentHistory: opt.currentHistory,
    ticks: summary.ticks,
  };
}

export function runTSPHillClimberDES(
  inst: TSPInstance, opts: TSPSAOptions,
  desOptions?: IterativeRunOptions,
): SADESResult {
  const rng = mulberry32(opts.seed);
  const source = new SingleStateSourceStation<Tour>(
    'hc-source',
    () => initialTour(inst, opts.init ?? 'nearest-neighbor', rng),
    tour => validateInitialTour('hc-source', inst, tour),
  );
  const opt = new TSPHillClimber('hc', inst, opts, {deferBootstrap: true, rng});
  const sink = new SingleStateSinkStation<Tour>('hc-sink');
  source.pipe(opt, SingleStateSourceStation.CH_INITIAL_STATE, SingleStateOptimizer.CH_INITIAL_STATE);
  opt.pipe(sink, SingleStateOptimizer.CH_RESULT, SingleStateSinkStation.CH_RESULT);
  const summary = runIterativeDES([source, opt, sink], {rng, ...desOptions, shuffle: desOptions?.shuffle ?? false});
  const result = sink.latest?.snapshot;
  if (!result) throw new Error('hc-des: result sink did not receive a final tour');
  return {
    bestTour: result.best, bestCost: result.bestCost,
    iterations: result.iteration,
    acceptedCount: result.acceptedCount,
    improveCount: result.improveCount,
    bestHistory: opt.bestHistory,
    currentHistory: opt.currentHistory,
    ticks: summary.ticks,
  };
}

function initialTour(inst: TSPInstance, initMode: 'random' | 'nearest-neighbor', rng: () => number): Tour {
  const n = inst.n;
  if (initMode === 'nearest-neighbor') {
    const start = Math.floor(rng() * n);
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
  const t: Tour = Array.from({length: n}, (_, i) => i);
  for (let i = t.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [t[i], t[j]] = [t[j], t[i]];
  }
  return t;
}

function validateInitialTour(sourceId: string, inst: TSPInstance, tour: Tour): void {
  Preconditions.check(sourceId, 'initial tour', `be a permutation of ${inst.n} cities`, isPermutation(tour, inst.n), tour);
  Preconditions.finite(sourceId, 'initial tour length', tourLength(inst, tour));
}
