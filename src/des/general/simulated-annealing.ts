'use strict';

// =============================================================================
// general/simulated-annealing.ts — Simulated Annealing as a discrete-event
// system, with a TSP problem adapter and a generic interface so other
// combinatorial problems can plug in.
//
// THE ALGORITHM
// ─────────────
//   Single-walker probabilistic local search inspired by metallurgy.
//   At each tick:
//     1. Generate a CANDIDATE neighbour s' of the current state s.
//     2. Compute Δ = cost(s') − cost(s).
//     3. ACCEPT s' if Δ ≤ 0 (improvement) or with probability exp(−Δ/T)
//        (Metropolis criterion). Otherwise stay at s.
//     4. UPDATE the temperature T by the cooling schedule.
//     5. Track the best state ever seen (separate from the walker).
//
//   Convergence: with logarithmic cooling T_k = T_0 / log(1 + k),
//   global optimum is found with probability 1 in the limit (Hajek 1988).
//   Geometric cooling T_k = α · T_{k-1} (α ≈ 0.99) is faster but loses
//   the theoretical guarantee — the textbook practical default.
//
// AS A DES
// ────────
//   Stations:
//     CandidateGenerator — emits a neighbour of the current state.
//     CostEvaluator      — computes Δ.
//     MetropolisAccept   — accepts/rejects, emits the next-state movable.
//     TemperatureSched   — global thermal schedule, ticked each iteration.
//     BestTracker        — sink-station holding the global-best state seen.
//
//   Movables: candidate-state, decision (accept/reject), Δ, current-T.
//
//   Each tick is one proposal+decision. The trace of (T, current-cost,
//   best-cost) per tick is what you'd plot or animate.
//
// PROBLEM-INTERFACE GENERICITY
// ────────────────────────────
//   The runner takes an `SAProblem<S>` — any state type S, with three
//   user-supplied functions: cost, neighbour, and (optionally) initial
//   state. The TSP adapter at the bottom of this file is the canonical
//   example; one could plug in a knapsack, vertex-cover, scheduling, etc.
// =============================================================================

import {mulberry32} from './prng';
import {
  SingleStateOptimizer, SingleStateSinkStation, SingleStateSourceStation, runIterativeDES,
  intrinsicCheck, monotonicityValidator,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// GENERIC PROBLEM INTERFACE
// -----------------------------------------------------------------------------

export interface SAProblem<S> {
  /** Compute the (real-valued) cost of a state. Lower = better. */
  cost: (s: S) => number;
  /** Generate a NEIGHBOUR of the current state. Must NOT mutate s. */
  neighbour: (s: S, rng: () => number) => S;
  /** Build the initial state. */
  initial: (rng: () => number) => S;
  /** Optional cheap clone (otherwise we use structuredClone for safety). */
  clone?: (s: S) => S;
}

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
// SOLVER OPTIONS, EVENTS, RESULTS
// -----------------------------------------------------------------------------

export interface SASolverOptions {
  /** Maximum number of iterations (ticks). */
  maxIterations: number;
  /** Cooling schedule. */
  cooling: CoolingSchedule;
  /** Random seed. */
  seed?: number;
  /** Stop if best has not improved for this many ticks. 0 = no early stop. */
  stallLimit?: number;
  /** Print every tick to stderr (only for tiny problems). */
  verbose?: boolean;
  /** Record full trace (every tick) — costs O(maxIterations) memory. */
  recordTrace?: boolean;
  /** Record-trace stride: keep one in N trace entries. Default 1. */
  traceStride?: number;
}

export interface SATickEvent {
  k: number;
  T: number;
  currentCost: number;
  candidateCost: number;
  delta: number;
  accept: boolean;
  acceptProb: number;
  bestCost: number;
}

export interface SAResult<S> {
  bestState: S;
  bestCost: number;
  finalState: S;
  finalCost: number;
  iterations: number;
  acceptedCount: number;
  improveCount: number;
  /** per-tick trace if `recordTrace=true`. */
  trace?: SATickEvent[];
  /** per-record best-cost history (always present, downsampled by traceStride). */
  bestHistory: number[];
  /** per-record current-cost history (always present, downsampled). */
  currentHistory: number[];
  temperatureHistory: number[];
}

// -----------------------------------------------------------------------------
// MAIN SOLVER
// -----------------------------------------------------------------------------

// =============================================================================
// SAOptimizer<S> — concrete leaf of SingleStateOptimizer<S> in des-base/.
// Hooks: initialState, cost, propose, accept (Metropolis), clone, shouldStop.
// =============================================================================

class SAOptimizer<S> extends SingleStateOptimizer<S> {
  private readonly problem: SAProblem<S>;
  private readonly cooling: CoolingSchedule;
  private readonly maxIters: number;
  private readonly stallLimit: number;
  private readonly verbose: boolean;
  private readonly recordTrace: boolean;
  private readonly cloneFn: (s: S) => S;
  /** Track stall (ticks since last best improvement). */
  private stallCount = 0;
  private prevBest = Infinity;
  /** If recordTrace, captured every iteration. */
  readonly trace: SATickEvent[] = [];
  readonly temperatureHistory: number[] = [];
  /** Current iteration's temperature (for trace recording). */
  private currentT = 0;
  /** Current iteration's candidate cost (for trace + acceptance prob). */
  private currentCandCost = 0;
  private currentAcceptProb = 1;
  private currentAccepted = false;

  constructor(
    problem: SAProblem<S>,
    options: SASolverOptions,
    lifecycle: {deferBootstrap?: boolean; rng?: () => number} = {},
  ) {
    super('simulated-annealing', {
      rng: lifecycle.rng ?? mulberry32(options.seed ?? 42),
      traceStride: Math.max(1, options.traceStride ?? 1),
    });
    this.problem = problem;
    this.cooling = options.cooling;
    this.maxIters = options.maxIterations;
    this.stallLimit = options.stallLimit ?? 0;
    this.verbose = options.verbose ?? false;
    this.recordTrace = options.recordTrace ?? false;
    this.cloneFn = problem.clone ?? ((s: S) => structuredClone(s));
    if (!lifecycle.deferBootstrap) this.bootstrap();
    if (this.traceStride > 0) this.temperatureHistory.push(temperatureAt(this.cooling, 0));

    // Intrinsic invariant: best-so-far is monotone non-increasing.
    this.addValidator(monotonicityValidator<SAOptimizer<S>>({
      name: 'sa.bestHistory.monotone',
      group: 'simulated-annealing-intrinsic',
      extract: st => st.bestHistory,
      direction: 'non-increasing',
    }));
    this.addValidator(intrinsicCheck<SAOptimizer<S>>({
      name: 'sa.acceptedCount-le-iterations',
      group: 'simulated-annealing-intrinsic',
      predicate: st => st.getAcceptedCount() <= st.getIteration(),
      expected: 'accepted ≤ iter',
      observedFn: st => `accepted=${st.getAcceptedCount()}  iter=${st.getIteration()}`,
    }));
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected initialState(rng: () => number): S { return this.problem.initial(rng); }
  protected cost(s: S): number { return this.problem.cost(s); }
  protected propose(s: S, rng: () => number): S { return this.problem.neighbour(s, rng); }
  protected clone(s: S): S { return this.cloneFn(s); }

  protected accept(_cur: S, _cand: S, ccur: number, ccand: number,
                   iter: number, rng: () => number): boolean {
    const T = temperatureAt(this.cooling, iter);
    this.currentT = T;
    this.currentCandCost = ccand;
    if (T <= 0) {
      this.currentAcceptProb = 0; this.currentAccepted = false;
      return false;
    }
    const delta = ccand - ccur;
    if (delta <= 0) {
      this.currentAcceptProb = 1; this.currentAccepted = true;
      return true;
    }
    const p = Math.exp(-delta / T);
    this.currentAcceptProb = p;
    this.currentAccepted = rng() < p;
    return this.currentAccepted;
  }

  protected shouldStop(iter: number): boolean {
    if (iter >= this.maxIters) return true;
    if (this.stallLimit > 0 && this.stallCount >= this.stallLimit) return true;
    if (iter > 0 && temperatureAt(this.cooling, iter) <= 0) return true;
    return false;
  }

  // ── HOOKS (optional override) ────────────────────────────────────────────

  protected onAccept(_c: S, _delta: number, iter: number): void {
    if (this.bestCost < this.prevBest) {
      this.prevBest = this.bestCost; this.stallCount = 0;
    } else {
      this.stallCount += 1;
    }
    this.recordTickEvent(iter);
  }
  protected onReject(_c: S, _delta: number, iter: number): void {
    this.stallCount += 1;
    this.recordTickEvent(iter);
  }
  protected override onBootstrap(): void {
    this.prevBest = this.bestCost;
  }
  private recordTickEvent(k: number): void {
    if (k % this.traceStride === 0) this.temperatureHistory.push(this.currentT);
    if (this.recordTrace) {
      this.trace.push({
        k, T: this.currentT,
        currentCost: this.currentCost, candidateCost: this.currentCandCost,
        delta: this.currentCandCost - this.currentCost + (this.currentAccepted ? 0 : 0),
        accept: this.currentAccepted, acceptProb: this.currentAcceptProb,
        bestCost: this.bestCost,
      });
    }
    if (this.verbose) {
      console.error(`SA  k=${k.toString().padStart(6)}  T=${this.currentT.toExponential(2)}  cur=${this.currentCost.toFixed(4)}  cand=${this.currentCandCost.toFixed(4)}  Δ=${(this.currentCandCost - this.currentCost).toFixed(4)}  p=${this.currentAcceptProb.toFixed(3)}  ${this.currentAccepted ? 'ACC' : 'rej'}  best=${this.bestCost.toFixed(4)}`);
    }
  }
}

/**
 * Run simulated annealing on a generic problem. Internally orchestrated
 * by an SAOptimizer<S> (concrete leaf of SingleStateOptimizer<S>) running
 * on the runIterativeDES engine — each tick is one proposal + Metropolis
 * accept/reject.
 */
export function runSimulatedAnnealing<S>(
  problem: SAProblem<S>,
  options: SASolverOptions,
): SAResult<S> {
  const rng = mulberry32(options.seed ?? 42);
  const source = new SingleStateSourceStation<S>(
    'simulated-annealing-source',
    () => problem.initial(rng),
    state => {
      const initialCost = problem.cost(state);
      Preconditions.finite('simulated-annealing-source', 'initialCost', initialCost);
    },
  );
  const opt = new SAOptimizer<S>(problem, options, {deferBootstrap: true, rng});
  const sink = new SingleStateSinkStation<S>('simulated-annealing-sink');
  source.pipe(opt, SingleStateSourceStation.CH_INITIAL_STATE, SingleStateOptimizer.CH_INITIAL_STATE);
  opt.pipe(sink, SingleStateOptimizer.CH_RESULT, SingleStateSinkStation.CH_RESULT);
  runIterativeDES([source, opt, sink], {shuffle: false});
  const result = sink.latest?.snapshot;
  if (!result) throw new Error('simulated-annealing: result sink did not receive a final state');
  return {
    bestState: result.best,
    bestCost: result.bestCost,
    finalState: result.current,
    finalCost: result.currentCost,
    iterations: result.iteration,
    acceptedCount: result.acceptedCount,
    improveCount: result.improveCount,
    trace: options.recordTrace ? opt.trace : undefined,
    // SingleStateOptimizer.bootstrap() pushes one initial entry; legacy
    // semantics records exactly `iterations` entries (one per Metropolis
    // tick) — drop the bootstrap entry to match.
    bestHistory: opt.bestHistory.slice(1),
    currentHistory: opt.currentHistory.slice(1),
    temperatureHistory: opt.temperatureHistory.slice(0, opt.bestHistory.length - 1),
  };
}

// -----------------------------------------------------------------------------
// TSP ADAPTER
// -----------------------------------------------------------------------------

import {TSPInstance, Tour, tourLength, isPermutation, checkPrecedence} from './genetic-tsp';

/** Build an SAProblem<Tour> for a TSP instance, using the 2-opt move family.
 *  2-opt: reverse a sub-segment [i..j] of the tour. Classic neighbourhood
 *  for TSP; a single 2-opt move changes exactly two edges. */
export function buildTSPSAProblem(
  instance: TSPInstance,
  opts: {
    /** Penalty per violated precedence pair (added to cost). */
    penaltyPerViolation?: number;
    /** Initial-state heuristic. */
    init?: 'random' | 'nearest-neighbor';
    /** Move set. Default 'mixed' (2-opt + or-opt). */
    moves?: '2-opt' | 'or-opt' | 'mixed';
  } = {},
): SAProblem<Tour> {
  const penalty = opts.penaltyPerViolation ?? 1e6;
  const init = opts.init ?? 'nearest-neighbor';
  const moves = opts.moves ?? 'mixed';
  const n = instance.n;
  return {
    cost(tour: Tour): number {
      let c = tourLength(instance, tour);
      if (instance.precedence) {
        for (const [a, b] of instance.precedence) {
          // Find positions; if a appears AFTER b, that's a violation.
          let posA = -1, posB = -1;
          for (let i = 0; i < tour.length; i++) {
            if (tour[i] === a) posA = i;
            if (tour[i] === b) posB = i;
          }
          if (posA >= 0 && posB >= 0 && posA >= posB) c += penalty;
        }
      }
      return c;
    },
    neighbour(tour: Tour, rng: () => number): Tour {
      const useMove = moves === 'mixed' ? (rng() < 0.7 ? '2-opt' : 'or-opt') : moves;
      if (useMove === '2-opt') {
        // Reverse tour[i..j].
        let i = Math.floor(rng() * n);
        let j = Math.floor(rng() * n);
        if (i > j) [i, j] = [j, i];
        if (j - i < 1) j = Math.min(n - 1, i + 1);
        const next = tour.slice();
        // reverse in place
        for (let a = i, b = j; a < b; a++, b--) {
          const tmp = next[a]; next[a] = next[b]; next[b] = tmp;
        }
        return next;
      } else {
        // or-opt: extract a sub-segment of length L ∈ {1,2,3} and reinsert
        // at a different random position.
        const L = 1 + Math.floor(rng() * 3);
        if (L >= n) return tour.slice();
        const i = Math.floor(rng() * (n - L + 1));
        const seg = tour.slice(i, i + L);
        const remaining = tour.slice(0, i).concat(tour.slice(i + L));
        const insertAt = Math.floor(rng() * (remaining.length + 1));
        if (insertAt === i) return tour.slice();   // null-move; return original
        return remaining.slice(0, insertAt).concat(seg).concat(remaining.slice(insertAt));
      }
    },
    initial(rng: () => number): Tour {
      if (init === 'nearest-neighbor') {
        const start = Math.floor(rng() * n);
        const tour: Tour = [start];
        const visited = new Set<number>([start]);
        let cur = start;
        while (tour.length < n) {
          let bestNext = -1; let bestD = Infinity;
          for (let j = 0; j < n; j++) {
            if (visited.has(j)) continue;
            const d = instance.distance[cur][j];
            if (d < bestD) { bestD = d; bestNext = j; }
          }
          tour.push(bestNext); visited.add(bestNext); cur = bestNext;
        }
        return tour;
      }
      const t: Tour = Array.from({length: n}, (_, i) => i);
      for (let i = t.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [t[i], t[j]] = [t[j], t[i]];
      }
      return t;
    },
    clone(tour: Tour): Tour {
      return tour.slice();
    },
  };
}

// -----------------------------------------------------------------------------
// KNAPSACK ADAPTER (bonus: shows the framework's genericity)
// -----------------------------------------------------------------------------

export interface KnapsackInstance {
  values: number[];
  weights: number[];
  capacity: number;
}

/** SA over 0/1 knapsack. State is a length-n bitstring. Neighbours flip
 *  a single bit. Cost is −value, with a steep penalty for over-capacity. */
export function buildKnapsackSAProblem(inst: KnapsackInstance, penalty = 1e6): SAProblem<number[]> {
  const n = inst.values.length;
  return {
    cost(x: number[]): number {
      let v = 0, w = 0;
      for (let i = 0; i < n; i++) { v += inst.values[i] * x[i]; w += inst.weights[i] * x[i]; }
      return -v + penalty * Math.max(0, w - inst.capacity);
    },
    neighbour(x: number[], rng: () => number): number[] {
      const j = Math.floor(rng() * n);
      const next = x.slice();
      next[j] = 1 - next[j];
      return next;
    },
    initial(rng: () => number): number[] {
      // Greedy heuristic by value/weight ratio, capped at capacity.
      const order = Array.from({length: n}, (_, i) => i)
        .sort((a, b) => (inst.values[b] / inst.weights[b]) - (inst.values[a] / inst.weights[a]));
      const x = new Array(n).fill(0);
      let w = 0;
      for (const i of order) {
        if (w + inst.weights[i] <= inst.capacity) { x[i] = 1; w += inst.weights[i]; }
      }
      return x;
    },
    clone(x: number[]): number[] { return x.slice(); },
  };
}

export {checkPrecedence, isPermutation};
