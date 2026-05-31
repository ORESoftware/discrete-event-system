'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/population_optimizer.rs
// - Keep file-for-file. Channel constants become pub consts; initial/result
//   token classes and PopulationResultSnapshot become token/data structs.
// - PopulationSourceStation and PopulationSinkStation become concrete DESStation
//   structs; PopulationOptimizer becomes a trait plus shared optimizer-state
//   struct for population, fitness, best, and history.
// - Pure proposal/fitness helpers can stay trait methods; if exposed as graph
//   transforms, implement PureTransform/PureTransformEntity with transform().
// - Convert duplicate seed, uninitialized optimizer, and invalid fitness throws
//   to Result.

// =============================================================================
// general/des-base/population-optimizer.ts — base class for POPULATION-BASED
// metaheuristics: genetic algorithm, particle swarm, differential evolution,
// ant colony optimization, evolution strategies, …
//
// PROBLEM SHAPE
// ─────────────
//   minimise f(x) over x ∈ X
//   by maintaining a POPULATION P_t = {x_1, …, x_K}, generating a new
//   population P_{t+1} via:
//
//      parents  = SELECT(P_t, fitness)
//      child    = RECOMBINE(parents) ∘ MUTATE
//      P_{t+1}  = REPLACE(P_t, child)        (with optional ELITISM)
//
//   The DIFFERENTIATOR among algorithms in this family is which
//   operators you plug in. PSO uses velocity update and global-best
//   instead of crossover; DE uses difference-vector mutation; ACO uses
//   pheromone-weighted construction. The base class accepts ALL of
//   these via its hooks.
//
// TEMPLATE METHOD (final)
// ───────────────────────
//   runTimeStep():
//     if shouldStop()                  → finalise
//     newPop = []
//     copy top-K elites unchanged
//     while newPop.length < popSize:
//       parents = select(...)
//       child   = recombine(parents)
//       child   = mutate(child)
//       newPop.push(child)
//     evaluate population
//     update best, history, generation
//
// HOOKS (abstract)
// ────────────────
//   initialPopulation, evaluate, select, recombine, mutate, clone, shouldStop
//
// HOOKS (optional override)
// ─────────────────────────
//   eliteCount (default 0), onGeneration, onFinish
// =============================================================================

import {ChannelName, DESStation, Token} from './station';

export const POPULATION_INITIAL_CHANNEL: ChannelName = 'population-initial';
export const POPULATION_RESULT_CHANNEL: ChannelName = 'population-result';

export class PopulationInitialToken<I> implements Token {
  constructor(readonly population: readonly I[]) {}
}

export interface PopulationResultSnapshot<I> {
  best: I;
  bestFitness: number;
  population: I[];
  fitness: number[];
  generation: number;
}

export class PopulationResultToken<I> implements Token {
  constructor(readonly snapshot: PopulationResultSnapshot<I>) {}
}

export class PopulationSourceStation<I> extends DESStation {
  static readonly CH_INITIAL_POPULATION = POPULATION_INITIAL_CHANNEL;
  private emitted = false;

  constructor(
    id: string,
    private readonly initialPopulation: () => readonly I[],
    private readonly validateInitialPopulation: (population: readonly I[]) => void = () => {},
  ) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    const population = this.initialPopulation();
    this.validateInitialPopulation(population);
    this.emit(new PopulationInitialToken(population), PopulationSourceStation.CH_INITIAL_POPULATION);
    this.emitted = true;
  }
}

export class PopulationSinkStation<I> extends DESStation {
  static readonly CH_RESULT = POPULATION_RESULT_CHANNEL;
  latest: PopulationResultToken<I> | undefined;

  constructor(id: string) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(PopulationSinkStation.CH_RESULT) > 0; }

  runTimeStep(): void {
    const tokens = this.drain<PopulationResultToken<I>>(PopulationSinkStation.CH_RESULT);
    if (tokens.length > 0) this.latest = tokens[tokens.length - 1];
  }
}

export abstract class PopulationOptimizer<I> extends DESStation {
  static readonly CH_INITIAL_POPULATION = POPULATION_INITIAL_CHANNEL;
  static readonly CH_RESULT = POPULATION_RESULT_CHANNEL;

  protected population: I[] = [];
  protected fitness: number[] = [];   // lower is better
  protected generation = 0;
  protected best!: I;
  protected bestFitness: number = Infinity;
  protected finished = false;
  protected initialized = false;
  private resultEmitted = false;

  readonly bestHistory: number[] = [];
  readonly meanHistory: number[] = [];
  readonly worstHistory: number[] = [];

  protected readonly popSize: number;
  protected readonly rng: () => number;

  constructor(id: string, opts: {popSize: number; rng: () => number}) {
    super(id);
    this.popSize = opts.popSize;
    this.rng = opts.rng;
    // Subclass MUST call this.bootstrap() after own constructor work.
  }

  /** Seed initial population + fitness. Called once by the subclass after
   *  its own ctor work, since we can't call abstract methods from base ctor. */
  protected bootstrap(): void {
    this.bootstrapFromPopulation(this.initialPopulation(this.popSize, this.rng));
  }

  /** Source-driven bootstrap used when the initial population is emitted by
   *  an explicit source station instead of hidden in the optimizer ctor. */
  protected bootstrapFromPopulation(initialPopulation: readonly I[]): void {
    if (this.initialized) throw new Error(`${this.id}: initial population already supplied`);
    this.population = initialPopulation.map(x => this.clone(x));
    if (this.population.length !== this.popSize) {
      throw new Error(`initialPopulation returned ${this.population.length} individuals, expected ${this.popSize}`);
    }
    this.fitness = this.population.map(x => this.evaluate(x));
    for (let i = 0; i < this.fitness.length; i++) {
      if (!Number.isFinite(this.fitness[i])) {
        throw new Error(`${this.id}: initial population fitness[${i}] must be finite; got ${this.fitness[i]}`);
      }
    }
    this.recordBest();
    this.initialized = true;
    this.onBootstrap();
  }

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  protected abstract initialPopulation(size: number, rng: () => number): I[];
  protected abstract evaluate(individual: I): number;
  /** Pick parents (≥ 1) for one offspring. */
  protected abstract select(pop: readonly I[], fitness: readonly number[], rng: () => number): I[];
  /** Combine parents into a child. (For PSO/DE: this is the velocity-
   *  or difference-update; for GA: crossover.) */
  protected abstract recombine(parents: readonly I[], rng: () => number): I;
  /** Apply mutation. */
  protected abstract mutate(child: I, rng: () => number): I;
  protected abstract clone(individual: I): I;
  protected abstract shouldStop(generation: number): boolean;

  // ── HOOKS (optional override) ────────────────────────────────────────────

  /** Number of best individuals copied unchanged to the next generation. */
  protected eliteCount(): number { return 0; }
  protected onBootstrap(): void {}
  protected onGeneration(_gen: number): void {}
  protected onFinish(): void {}

  /** Constraint-handling hook: return true to accept a freshly-bred child,
   *  false to retry. The default accepts every child (penalty-only
   *  feasibility). Subclasses with hard constraints (precedence,
   *  knapsack, …) can implement "cut" semantics by returning false until
   *  a feasible child appears or the retry budget is exhausted. */
  protected acceptChild(_child: I): boolean { return true; }
  /** Maximum number of breeding attempts per offspring slot. After the
   *  budget is exhausted the LAST attempt is pushed to the population
   *  even if `acceptChild` returned false (so popSize is preserved). */
  protected childRetryLimit(): number { return 1; }
  /** Instrumentation hook fired when `acceptChild` returns false. */
  protected onChildRejected(_child: I, _attempt: number): void {}

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  runTimeStep(): void {
    if (this.finished) return;
    if (!this.initialized) {
      const seeds = this.drain<PopulationInitialToken<I>>(PopulationOptimizer.CH_INITIAL_POPULATION);
      if (seeds.length === 0) return;
      if (seeds.length > 1) throw new Error(`${this.id}: expected exactly one initial-population token, got ${seeds.length}`);
      this.bootstrapFromPopulation(seeds[0].population);
      return;
    }
    if (this.inboxSize(PopulationOptimizer.CH_INITIAL_POPULATION) > 0) {
      throw new Error(`${this.id}: received an initial-population token after initialization`);
    }
    if (this.shouldStop(this.generation)) {
      this.finished = true; this.onFinish(); this.emitResult(); return;
    }
    const newPop: I[] = [];
    const newFit: number[] = [];
    // Elitism — copy best k unchanged.
    const eliteK = Math.max(0, Math.min(this.popSize, this.eliteCount()));
    if (eliteK > 0) {
      const order = this.fitness.map((f, i) => [f, i] as const).sort((a, b) => a[0] - b[0]);
      for (let k = 0; k < eliteK; k++) {
        const idx = order[k][1];
        newPop.push(this.clone(this.population[idx]));
        newFit.push(order[k][0]);
      }
    }
    const retryBudget = Math.max(1, this.childRetryLimit());
    while (newPop.length < this.popSize) {
      let child!: I;
      let accepted = false;
      for (let attempt = 0; attempt < retryBudget; attempt++) {
        const parents = this.select(this.population, this.fitness, this.rng);
        child = this.recombine(parents, this.rng);
        child = this.mutate(child, this.rng);
        if (this.acceptChild(child)) { accepted = true; break; }
        this.onChildRejected(child, attempt);
      }
      if (!accepted) this.onChildRejected(child, retryBudget);
      newPop.push(child);
      newFit.push(this.evaluate(child));
    }
    this.population = newPop;
    this.fitness = newFit;
    this.recordBest();
    this.generation++;
    this.onGeneration(this.generation);
  }

  override hasWork(): boolean {
    return this.inboxSize(PopulationOptimizer.CH_INITIAL_POPULATION) > 0 ||
      (this.initialized && !this.finished);
  }

  // ── INTERNALS ────────────────────────────────────────────────────────────

  private recordBest(): void {
    let bestIdx = 0; let bestF = this.fitness[0];
    let mean = 0; let worst = -Infinity;
    for (let i = 0; i < this.fitness.length; i++) {
      const f = this.fitness[i];
      mean += f;
      if (f < bestF) { bestF = f; bestIdx = i; }
      if (f > worst) worst = f;
    }
    mean /= this.fitness.length;
    if (bestF < this.bestFitness) {
      this.bestFitness = bestF;
      this.best = this.clone(this.population[bestIdx]);
    }
    this.bestHistory.push(this.bestFitness);
    this.meanHistory.push(mean);
    this.worstHistory.push(worst);
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getPopulation(): readonly I[] { this.assertInitializedForRead(); return this.population; }
  getFitness(): readonly number[] { this.assertInitializedForRead(); return this.fitness; }
  getBest(): I { this.assertInitializedForRead(); return this.best; }
  getBestFitness(): number { this.assertInitializedForRead(); return this.bestFitness; }
  getGeneration(): number { return this.generation; }
  isFinished(): boolean { return this.finished; }
  isInitialized(): boolean { return this.initialized; }

  private emitResult(): void {
    if (this.resultEmitted) return;
    this.emit(new PopulationResultToken({
      best: this.clone(this.best),
      bestFitness: this.bestFitness,
      population: this.population.map(x => this.clone(x)),
      fitness: this.fitness.slice(),
      generation: this.generation,
    }), PopulationOptimizer.CH_RESULT);
    this.resultEmitted = true;
  }

  private assertInitializedForRead(): void {
    if (!this.initialized) throw new Error(`${this.id}: optimizer has not received an initial population`);
  }
}
