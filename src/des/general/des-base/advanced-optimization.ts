'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/advanced_optimization.rs
// - Keep file-for-file. OptimizationCandidateToken, GraphWalkToken,
//   ConstraintAssignmentToken, and ParetoCandidateToken become token structs;
//   trace/options/archive interfaces become data structs.
// - NumericSwarmOptimizerStation, PheromoneGraphSearchStation,
//   ConstraintSatisfactionSearchStation, SourceDrivenConstraintSatisfactionSearchStation,
//   and UnitVectorRelaxationStation become traits plus reusable state structs.
// - ParetoArchiveStation remains a concrete struct implementing DESStation;
//   Set usage maps to HashSet/BTreeSet depending on deterministic ordering needs.
// - Keep pure helpers such as dominates, normalize, vectorDot, and gram as module
//   functions unless lifted into the DES graph, where they should be PureTransform
//   or PureTransformEntity implementations. Replace thrown errors with Result.

// =============================================================================
// Shared station/token bases for advanced optimization families.
//
// These are deliberately framework-level pieces: concrete algorithms plug into
// template methods, while candidates, tours, assignments, Pareto points, and SDP
// iterates move as typed tokens. That keeps the models renderable by the future
// visual editor as station graphs instead of hidden one-off loops.
// =============================================================================

import {ChannelName, DESStation, Token, DEFAULT_CHANNEL} from './station';
import {TreeSearchStation, NodeEvaluation} from './tree-search';
import {Preconditions} from './preconditions';

// -----------------------------------------------------------------------------
// Generic movable tokens
// -----------------------------------------------------------------------------

export class OptimizationCandidateToken<T> implements Token {
  constructor(
    readonly kind: string,
    readonly candidate: T,
    readonly score: number,
    readonly iteration: number,
  ) {}
}

export class GraphWalkToken<N = number> implements Token {
  constructor(
    readonly nodes: N[],
    readonly cost: number,
    readonly iteration: number,
  ) {}
}

export class ConstraintAssignmentToken<D = string> implements Token {
  constructor(
    readonly assignment: Record<string, D>,
    readonly depth: number,
  ) {}
}

export class ParetoCandidateToken<T> implements Token {
  constructor(
    readonly candidate: T,
    readonly objectives: number[],
    readonly generation = 0,
  ) {}
}

export interface OptimizationTraceRow {
  iteration: number;
  bestScore: number;
  meanScore?: number;
  worstScore?: number;
}

// -----------------------------------------------------------------------------
// Numeric swarm optimization
// -----------------------------------------------------------------------------

export interface NumericSwarmParticle {
  id: string;
  position: number[];
  velocity: number[];
  bestPosition: number[];
  bestScore: number;
  score: number;
}

export interface NumericSwarmOptions {
  particleCount: number;
  dimension: number;
  iterations: number;
  lowerBound: number[];
  upperBound: number[];
  rng: () => number;
}

export abstract class NumericSwarmOptimizerStation extends DESStation {
  protected readonly particleCount: number;
  protected readonly dimension: number;
  protected readonly iterations: number;
  protected readonly lowerBound: number[];
  protected readonly upperBound: number[];
  protected readonly rng: () => number;
  protected particles: NumericSwarmParticle[] = [];
  protected iteration = 0;
  protected finished = false;
  protected bestPosition: number[] = [];
  protected bestScore = Infinity;

  readonly trace: OptimizationTraceRow[] = [];

  constructor(id: string, opts: NumericSwarmOptions) {
    super(id);
    Preconditions.integerInRange(id, 'particleCount', opts.particleCount, 1, 1e9);
    Preconditions.integerInRange(id, 'dimension', opts.dimension, 1, 1e6);
    Preconditions.integerInRange(id, 'iterations', opts.iterations, 1, 1e9);
    Preconditions.lengthEq(id, 'lowerBound', opts.lowerBound, opts.dimension);
    Preconditions.lengthEq(id, 'upperBound', opts.upperBound, opts.dimension);
    Preconditions.allFinite(id, 'lowerBound', opts.lowerBound);
    Preconditions.allFinite(id, 'upperBound', opts.upperBound);
    for (let i = 0; i < opts.dimension; i++) {
      Preconditions.check(id, `lowerBound[${i}] < upperBound[${i}]`, 'satisfy lower < upper',
        opts.lowerBound[i] < opts.upperBound[i], [opts.lowerBound[i], opts.upperBound[i]]);
    }
    this.particleCount = opts.particleCount;
    this.dimension = opts.dimension;
    this.iterations = opts.iterations;
    this.lowerBound = opts.lowerBound.slice();
    this.upperBound = opts.upperBound.slice();
    this.rng = opts.rng;
  }

  override assertPreconditions(): void {
    const cls = this.constructor.name;
    Preconditions.integerInRange(cls, 'particleCount', this.particleCount, 1, 1e9);
    Preconditions.integerInRange(cls, 'dimension', this.dimension, 1, 1e6);
    Preconditions.integerInRange(cls, 'iterations', this.iterations, 1, 1e9);
    Preconditions.lengthEq(cls, 'lowerBound', this.lowerBound, this.dimension);
    Preconditions.lengthEq(cls, 'upperBound', this.upperBound, this.dimension);
    Preconditions.allFinite(cls, 'lowerBound', this.lowerBound);
    Preconditions.allFinite(cls, 'upperBound', this.upperBound);
    for (let i = 0; i < this.dimension; i++) {
      Preconditions.check(cls, `lowerBound[${i}] < upperBound[${i}]`, 'satisfy lower < upper',
        this.lowerBound[i] < this.upperBound[i], [this.lowerBound[i], this.upperBound[i]]);
    }
  }

  protected abstract objective(position: readonly number[]): number;
  protected abstract updateParticle(
    particle: NumericSwarmParticle,
    globalBest: readonly number[],
    iteration: number,
  ): NumericSwarmParticle;

  protected bootstrap(): void {
    this.assertPreconditions();
    this.particles = [];
    for (let i = 0; i < this.particleCount; i++) {
      const position = this.randomPosition();
      const velocity = this.randomVelocity();
      const score = this.objective(position);
      Preconditions.finite(this.id, `initial particle score ${i}`, score);
      const p: NumericSwarmParticle = {
        id: `particle-${i}`,
        position,
        velocity,
        bestPosition: position.slice(),
        bestScore: score,
        score,
      };
      this.particles.push(p);
      this.considerBest(p);
    }
    this.recordTrace();
  }

  runTimeStep(): void {
    if (this.finished) return;
    if (this.iteration >= this.iterations) {
      this.finished = true;
      return;
    }

    const nextParticles: NumericSwarmParticle[] = [];
    for (const particle of this.particles) {
      const next = this.updateParticle(this.cloneParticle(particle), this.bestPosition, this.iteration);
      Preconditions.lengthEq(this.id, `${next.id}.position`, next.position, this.dimension);
      Preconditions.lengthEq(this.id, `${next.id}.velocity`, next.velocity, this.dimension);
      Preconditions.allFinite(this.id, `${next.id}.position`, next.position);
      Preconditions.allFinite(this.id, `${next.id}.velocity`, next.velocity);
      next.position = this.clampPosition(next.position);
      next.score = this.objective(next.position);
      Preconditions.finite(this.id, `${next.id}.score`, next.score);
      if (next.score < next.bestScore) {
        next.bestScore = next.score;
        next.bestPosition = next.position.slice();
      }
      this.considerBest(next);
      nextParticles.push(next);
      this.emit(new OptimizationCandidateToken('swarm-particle', next.position.slice(), next.score, this.iteration));
    }
    this.particles = nextParticles;
    this.iteration += 1;
    this.recordTrace();
  }

  override hasWork(): boolean { return !this.finished; }

  getBestPosition(): number[] { return this.bestPosition.slice(); }
  getBestScore(): number { return this.bestScore; }
  getParticles(): NumericSwarmParticle[] { return this.particles.map(p => this.cloneParticle(p)); }
  getIteration(): number { return this.iteration; }

  protected randomPosition(): number[] {
    return this.lowerBound.map((lo, i) => lo + this.rng() * (this.upperBound[i] - lo));
  }

  protected randomVelocity(): number[] {
    return this.lowerBound.map((lo, i) => {
      const span = this.upperBound[i] - lo;
      return (this.rng() * 2 - 1) * 0.2 * span;
    });
  }

  protected clampPosition(x: readonly number[]): number[] {
    return x.map((v, i) => Math.max(this.lowerBound[i], Math.min(this.upperBound[i], v)));
  }

  protected cloneParticle(p: NumericSwarmParticle): NumericSwarmParticle {
    return {
      id: p.id,
      position: p.position.slice(),
      velocity: p.velocity.slice(),
      bestPosition: p.bestPosition.slice(),
      bestScore: p.bestScore,
      score: p.score,
    };
  }

  private considerBest(p: NumericSwarmParticle): void {
    if (p.bestScore < this.bestScore) {
      this.bestScore = p.bestScore;
      this.bestPosition = p.bestPosition.slice();
    }
  }

  private recordTrace(): void {
    const scores = this.particles.map(p => p.score);
    const mean = scores.reduce((a, b) => a + b, 0) / Math.max(1, scores.length);
    const worst = scores.reduce((a, b) => Math.max(a, b), -Infinity);
    this.trace.push({iteration: this.iteration, bestScore: this.bestScore, meanScore: mean, worstScore: worst});
  }
}

// -----------------------------------------------------------------------------
// Pheromone-guided constructive graph search
// -----------------------------------------------------------------------------

export interface PheromoneGraphOptions {
  nodeCount: number;
  ants: number;
  iterations: number;
  alpha: number;
  beta: number;
  evaporation: number;
  deposit: number;
  rng: () => number;
}

export abstract class PheromoneGraphSearchStation extends DESStation {
  protected readonly nodeCount: number;
  protected readonly ants: number;
  protected readonly iterations: number;
  protected readonly alpha: number;
  protected readonly beta: number;
  protected readonly evaporation: number;
  protected readonly deposit: number;
  protected readonly rng: () => number;
  protected pheromone: number[][];
  protected iteration = 0;
  protected finished = false;
  protected bestPath: number[] = [];
  protected bestCost = Infinity;

  readonly trace: OptimizationTraceRow[] = [];

  constructor(id: string, opts: PheromoneGraphOptions) {
    super(id);
    Preconditions.integerInRange(id, 'nodeCount', opts.nodeCount, 2, 1e6);
    Preconditions.integerInRange(id, 'ants', opts.ants, 1, 1e9);
    Preconditions.integerInRange(id, 'iterations', opts.iterations, 1, 1e9);
    Preconditions.nonNegative(id, 'alpha', opts.alpha);
    Preconditions.nonNegative(id, 'beta', opts.beta);
    Preconditions.inRange(id, 'evaporation', opts.evaporation, 0, 1);
    Preconditions.positive(id, 'deposit', opts.deposit);
    this.nodeCount = opts.nodeCount;
    this.ants = opts.ants;
    this.iterations = opts.iterations;
    this.alpha = opts.alpha;
    this.beta = opts.beta;
    this.evaporation = opts.evaporation;
    this.deposit = opts.deposit;
    this.rng = opts.rng;
    this.pheromone = Array.from({length: this.nodeCount}, () => new Array(this.nodeCount).fill(1));
  }

  override assertPreconditions(): void {
    const cls = this.constructor.name;
    Preconditions.integerInRange(cls, 'nodeCount', this.nodeCount, 2, 1e6);
    Preconditions.integerInRange(cls, 'ants', this.ants, 1, 1e9);
    Preconditions.integerInRange(cls, 'iterations', this.iterations, 1, 1e9);
    Preconditions.nonNegative(cls, 'alpha', this.alpha);
    Preconditions.nonNegative(cls, 'beta', this.beta);
    Preconditions.inRange(cls, 'evaporation', this.evaporation, 0, 1);
    Preconditions.positive(cls, 'deposit', this.deposit);
  }

  protected abstract pathCost(path: readonly number[]): number;
  protected abstract heuristic(from: number, to: number): number;

  runTimeStep(): void {
    if (this.finished) return;
    if (this.iteration >= this.iterations) {
      this.finished = true;
      return;
    }

    const walks: Array<{path: number[]; cost: number}> = [];
    for (let ant = 0; ant < this.ants; ant++) {
      const path = this.constructPath(ant % this.nodeCount);
      const cost = this.pathCost(path);
      Preconditions.positive(this.id, 'pathCost', cost);
      walks.push({path, cost});
      if (cost < this.bestCost) {
        this.bestCost = cost;
        this.bestPath = path.slice();
      }
      this.emit(new GraphWalkToken(path.slice(), cost, this.iteration));
    }
    this.evaporate();
    for (const walk of walks) this.depositWalk(walk.path, walk.cost);
    const mean = walks.reduce((acc, w) => acc + w.cost, 0) / walks.length;
    const worst = walks.reduce((acc, w) => Math.max(acc, w.cost), -Infinity);
    this.trace.push({iteration: this.iteration, bestScore: this.bestCost, meanScore: mean, worstScore: worst});
    this.iteration += 1;
  }

  override hasWork(): boolean { return !this.finished; }

  getBestPath(): number[] { return this.bestPath.slice(); }
  getBestCost(): number { return this.bestCost; }
  getPheromone(): number[][] { return this.pheromone.map(r => r.slice()); }
  getIteration(): number { return this.iteration; }

  protected constructPath(start: number): number[] {
    const path = [start];
    const unvisited = new Set<number>();
    for (let i = 0; i < this.nodeCount; i++) if (i !== start) unvisited.add(i);
    while (unvisited.size > 0) {
      const current = path[path.length - 1];
      const next = this.pickNext(current, [...unvisited]);
      path.push(next);
      unvisited.delete(next);
    }
    path.push(start);
    return path;
  }

  protected pickNext(from: number, options: readonly number[]): number {
    const weights = options.map(to => {
      const tau = Math.pow(this.pheromone[from][to], this.alpha);
      const eta = Math.pow(Math.max(1e-12, this.heuristic(from, to)), this.beta);
      return tau * eta;
    });
    const total = weights.reduce((a, b) => a + b, 0);
    if (!Number.isFinite(total) || total <= 0) {
      return options[Math.floor(this.rng() * options.length)];
    }
    let r = this.rng() * total;
    for (let i = 0; i < options.length; i++) {
      r -= weights[i];
      if (r <= 0) return options[i];
    }
    return options[options.length - 1];
  }

  private evaporate(): void {
    const keep = 1 - this.evaporation;
    for (let i = 0; i < this.nodeCount; i++) {
      for (let j = 0; j < this.nodeCount; j++) this.pheromone[i][j] = Math.max(1e-9, this.pheromone[i][j] * keep);
    }
  }

  private depositWalk(path: readonly number[], cost: number): void {
    const amount = this.deposit / Math.max(1e-12, cost);
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1];
      const b = path[i];
      this.pheromone[a][b] += amount;
      this.pheromone[b][a] += amount;
    }
  }
}

// -----------------------------------------------------------------------------
// Constraint-satisfaction tree search
// -----------------------------------------------------------------------------

export interface ConstraintSearchNode<D = string> {
  assignment: Record<string, D>;
  depth: number;
}

export abstract class ConstraintSatisfactionSearchStation<D = string>
  extends TreeSearchStation<ConstraintSearchNode<D>> {
  protected readonly variables: string[];
  protected readonly domains: Record<string, D[]>;
  protected readonly frontier: Array<ConstraintSearchNode<D>> = [];
  protected solution: Record<string, D> | null = null;

  constructor(id: string, variables: readonly string[], domains: Record<string, readonly D[]>, maxNodes = Infinity) {
    super(id, {objective: 'maximise', maxNodes});
    this.variables = variables.slice();
    this.domains = {};
    for (const v of variables) this.domains[v] = (domains[v] ?? []).slice();
    this.frontier.push({assignment: {}, depth: 0});
  }

  override assertPreconditions(): void {
    const cls = this.constructor.name;
    Preconditions.nonEmpty(cls, 'variables', this.variables);
    for (const v of this.variables) {
      Preconditions.nonEmpty(cls, `domains.${v}`, this.domains[v] ?? []);
    }
  }

  protected abstract isConsistent(assignment: Readonly<Record<string, D>>): boolean;

  protected pickNext(): ConstraintSearchNode<D> | null {
    return this.frontier.pop() ?? null;
  }

  protected evaluate(node: ConstraintSearchNode<D>): NodeEvaluation {
    if (!this.isConsistent(node.assignment)) return {bound: -Infinity, isLeaf: true, isFeasible: false};
    const complete = this.variables.every(v => node.assignment[v] !== undefined);
    return {
      bound: this.variables.length,
      isLeaf: complete,
      isFeasible: complete,
      value: complete ? this.variables.length : undefined,
    };
  }

  protected expand(node: ConstraintSearchNode<D>): ConstraintSearchNode<D>[] {
    const variable = this.chooseVariable(node.assignment);
    if (!variable) return [];
    const out: Array<ConstraintSearchNode<D>> = [];
    for (const value of this.domains[variable]) {
      const assignment = {...node.assignment, [variable]: value};
      if (this.isConsistent(assignment)) out.push({assignment, depth: node.depth + 1});
    }
    return out.reverse();
  }

  protected pushChildren(children: ConstraintSearchNode<D>[]): void {
    this.frontier.push(...children);
  }

  protected override currentBestBound(): number {
    return this.frontier.length > 0 ? this.variables.length : -Infinity;
  }

  protected override shouldPrune(node: ConstraintSearchNode<D>, ev: NodeEvaluation): boolean {
    return !this.isConsistent(node.assignment) || super.shouldPrune(node, ev);
  }

  protected override onIncumbentUpdate(node: ConstraintSearchNode<D>): void {
    this.solution = {...node.assignment};
    this.emit(new ConstraintAssignmentToken(this.solution, node.depth), DEFAULT_CHANNEL);
  }

  getSolution(): Record<string, D> | null { return this.solution ? {...this.solution} : null; }
  getVariables(): string[] { return this.variables.slice(); }
  getDomains(): Record<string, D[]> {
    const out: Record<string, D[]> = {};
    for (const [k, v] of Object.entries(this.domains)) out[k] = v.slice();
    return out;
  }

  private chooseVariable(assignment: Readonly<Record<string, D>>): string | null {
    let best: string | null = null;
    let bestCount = Infinity;
    for (const variable of this.variables) {
      if (assignment[variable] !== undefined) continue;
      let count = 0;
      for (const value of this.domains[variable]) {
        if (this.isConsistent({...assignment, [variable]: value})) count += 1;
      }
      if (count < bestCount) {
        best = variable;
        bestCount = count;
      }
    }
    return best;
  }
}

/** Source-driven CSP wrapper for models that should enter the run loop through
 *  an explicit start movable and exit through a result movable. The search
 *  itself still uses the shared TreeSearchStation template method. */
export abstract class SourceDrivenConstraintSatisfactionSearchStation<
  D,
  Start extends Token,
  Result extends Token,
> extends ConstraintSatisfactionSearchStation<D> {
  private started = false;
  private resultEmitted = false;

  constructor(
    id: string,
    variables: readonly string[],
    domains: Record<string, readonly D[]>,
    maxNodes: number,
    private readonly startChannel: ChannelName,
    private readonly resultChannel: ChannelName,
  ) {
    super(id, variables, domains, maxNodes);
  }

  protected abstract acceptStartToken(token: Start): void;
  protected abstract makeResultToken(): Result;

  override hasWork(): boolean {
    if (!this.started) return this.inboxSize(this.startChannel) > 0;
    return !this.resultEmitted;
  }

  override runTimeStep(): void {
    if (!this.started) {
      const starts = this.drain<Start>(this.startChannel);
      Preconditions.check(this.id, 'start token count', 'receive exactly one start token',
        starts.length === 1, starts.length);
      this.acceptStartToken(starts[0]);
      this.started = true;
      return;
    }

    if (!this.isFinished()) super.runTimeStep();
    if (this.isFinished() && !this.resultEmitted) {
      this.emit(this.makeResultToken(), this.resultChannel);
      this.resultEmitted = true;
    }
  }
}

// -----------------------------------------------------------------------------
// Pareto archive station
// -----------------------------------------------------------------------------

export interface ParetoArchiveRow<T> {
  candidate: T;
  objectives: number[];
  generation: number;
}

export class ParetoArchiveStation<T> extends DESStation {
  private readonly archive: Array<ParetoArchiveRow<T>> = [];
  private readonly pending: Array<ParetoCandidateToken<T>> = [];
  private processed = 0;
  private finished = false;

  constructor(id: string, candidates: readonly ParetoCandidateToken<T>[] = []) {
    super(id);
    this.pending.push(...candidates);
  }

  enqueue(candidate: ParetoCandidateToken<T>): void {
    this.pending.push(candidate);
    this.finished = false;
  }

  runTimeStep(): void {
    const inbox = this.drain<ParetoCandidateToken<T>>();
    this.pending.push(...inbox);
    if (this.pending.length > 0) this.finished = false;
    if (this.finished) return;
    const next = this.pending.shift();
    if (!next) {
      this.finished = true;
      return;
    }
    this.processed += 1;
    this.consider(next);
  }

  override hasWork(): boolean {
    return !this.finished || this.pending.length > 0 || this.inboxSize() > 0;
  }

  getArchive(): Array<ParetoArchiveRow<T>> {
    return this.archive.map(row => ({
      candidate: row.candidate,
      objectives: row.objectives.slice(),
      generation: row.generation,
    }));
  }

  getProcessedCount(): number { return this.processed; }

  private consider(token: ParetoCandidateToken<T>): void {
    Preconditions.nonEmpty(this.id, 'objectives', token.objectives);
    Preconditions.allFinite(this.id, 'objectives', token.objectives);
    for (const row of this.archive) {
      if (sameObjectives(row.objectives, token.objectives)) return;
      if (dominates(row.objectives, token.objectives)) return;
    }
    for (let i = this.archive.length - 1; i >= 0; i--) {
      if (dominates(token.objectives, this.archive[i].objectives)) this.archive.splice(i, 1);
    }
    this.archive.push({
      candidate: token.candidate,
      objectives: token.objectives.slice(),
      generation: token.generation,
    });
  }
}

export function dominates(a: readonly number[], b: readonly number[]): boolean {
  Preconditions.lengthEq('dominates', 'objective vector b', b, a.length);
  Preconditions.allFinite('dominates', 'objective vector a', a);
  Preconditions.allFinite('dominates', 'objective vector b', b);
  let strictlyBetter = false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i] + 1e-12) return false;
    if (a[i] < b[i] - 1e-12) strictlyBetter = true;
  }
  return strictlyBetter;
}

function sameObjectives(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 1e-12) return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Rank-constrained SDP / unit-vector relaxation station
// -----------------------------------------------------------------------------

export interface UnitVectorRelaxationOptions {
  nodes: number;
  rank: number;
  iterations: number;
  stepSize: number;
  rng: () => number;
}

export interface UnitVectorRelaxationTraceRow {
  iteration: number;
  objective: number;
}

export abstract class UnitVectorRelaxationStation extends DESStation {
  protected readonly nodes: number;
  protected readonly rank: number;
  protected readonly iterations: number;
  protected readonly stepSize: number;
  protected readonly rng: () => number;
  protected vectors: number[][];
  protected bestVectors: number[][];
  protected bestObjective = -Infinity;
  protected iteration = 0;
  protected finished = false;

  readonly trace: UnitVectorRelaxationTraceRow[] = [];

  constructor(id: string, opts: UnitVectorRelaxationOptions) {
    super(id);
    Preconditions.integerInRange(id, 'nodes', opts.nodes, 2, 1e6);
    Preconditions.integerInRange(id, 'rank', opts.rank, 1, 1e6);
    Preconditions.integerInRange(id, 'iterations', opts.iterations, 1, 1e9);
    Preconditions.positive(id, 'stepSize', opts.stepSize);
    this.nodes = opts.nodes;
    this.rank = opts.rank;
    this.iterations = opts.iterations;
    this.stepSize = opts.stepSize;
    this.rng = opts.rng;
    this.vectors = Array.from({length: this.nodes}, () => this.randomUnitVector());
    this.bestVectors = this.vectors.map(v => v.slice());
  }

  override assertPreconditions(): void {
    const cls = this.constructor.name;
    Preconditions.integerInRange(cls, 'nodes', this.nodes, 2, 1e6);
    Preconditions.integerInRange(cls, 'rank', this.rank, 1, 1e6);
    Preconditions.integerInRange(cls, 'iterations', this.iterations, 1, 1e9);
    Preconditions.positive(cls, 'stepSize', this.stepSize);
  }

  protected abstract objective(vectors: readonly number[][]): number;
  protected abstract gradient(vectors: readonly number[][]): number[][];

  protected bootstrap(): void {
    this.recordBest();
  }

  runTimeStep(): void {
    if (this.finished) return;
    if (this.iteration >= this.iterations) {
      this.finished = true;
      return;
    }
    const grad = this.gradient(this.vectors);
    Preconditions.lengthEq(this.id, 'gradient', grad, this.nodes);
    for (let i = 0; i < this.nodes; i++) {
      Preconditions.lengthEq(this.id, `gradient[${i}]`, grad[i], this.rank);
      Preconditions.allFinite(this.id, `gradient[${i}]`, grad[i]);
      for (let j = 0; j < this.rank; j++) this.vectors[i][j] += this.stepSize * grad[i][j];
      this.vectors[i] = normalize(this.vectors[i]);
    }
    this.iteration += 1;
    this.recordBest();
  }

  override hasWork(): boolean { return !this.finished; }

  getVectors(): number[][] { return this.vectors.map(v => v.slice()); }
  getBestVectors(): number[][] { return this.bestVectors.map(v => v.slice()); }
  getBestObjective(): number { return this.bestObjective; }
  getGramMatrix(): number[][] { return gram(this.bestVectors); }
  getIteration(): number { return this.iteration; }

  protected randomUnitVector(): number[] {
    const v = new Array<number>(this.rank);
    for (let i = 0; i < this.rank; i++) v[i] = this.rng() * 2 - 1;
    return normalize(v);
  }

  private recordBest(): void {
    const value = this.objective(this.vectors);
    Preconditions.finite(this.id, 'objective', value);
    this.trace.push({iteration: this.iteration, objective: value});
    if (value > this.bestObjective) {
      this.bestObjective = value;
      this.bestVectors = this.vectors.map(v => v.slice());
    }
  }
}

export function normalize(v: readonly number[]): number[] {
  Preconditions.nonEmpty('normalize', 'v', v);
  Preconditions.allFinite('normalize', 'v', v);
  const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
  if (norm < 1e-12) {
    const out = new Array(v.length).fill(0);
    out[0] = 1;
    return out;
  }
  return v.map(x => x / norm);
}

export function vectorDot(a: readonly number[], b: readonly number[]): number {
  Preconditions.lengthEq('vectorDot', 'b', b, a.length);
  Preconditions.allFinite('vectorDot', 'a', a);
  Preconditions.allFinite('vectorDot', 'b', b);
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

export function gram(vectors: readonly number[][]): number[][] {
  return vectors.map(a => vectors.map(b => vectorDot(a, b)));
}
