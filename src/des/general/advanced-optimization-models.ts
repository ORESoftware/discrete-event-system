'use strict';

// =============================================================================
// Concrete advanced-optimization models built on shared DES station bases.
// =============================================================================

import {
  NumericSwarmOptimizerStation,
  NumericSwarmParticle,
  ParetoArchiveStation,
  ParetoCandidateToken,
  PheromoneGraphSearchStation,
  SourceDrivenConstraintSatisfactionSearchStation,
  UnitVectorRelaxationStation,
  dominates,
  vectorDot,
} from './des-base/advanced-optimization';
import {
  ChannelName,
  DESStation,
  LatestTokenSinkStation,
  SingleStateOptimizer,
  SingleTokenSourceStation,
  Token,
  runIterativeDES,
} from './des-base';
import {StationGraphTopology, stationGraphTopology} from './des-base/model-topology';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

function sq(x: number): number { return x * x; }

function dist(a: Point2, b: Point2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

// -----------------------------------------------------------------------------
// 21/23. Particle Swarm Optimization
// -----------------------------------------------------------------------------

export type ContinuousObjectiveName = 'sphere' | 'rastrigin' | 'rosenbrock';

export interface ParticleSwarmParams {
  objective?: ContinuousObjectiveName;
  dimension?: number;
  particles?: number;
  iterations?: number;
  lower?: number;
  upper?: number;
  inertia?: number;
  cognitive?: number;
  social?: number;
  seed?: number;
}

export interface ParticleSwarmResult {
  bestPosition: number[];
  bestValue: number;
  iterations: number;
  trace: Array<{iteration: number; bestValue: number; meanValue: number; worstValue: number}>;
  topology: StationGraphTopology;
}

class OptimizationStartToken<P> implements Token {
  constructor(readonly model: string, readonly params: P) {}
}

class ParticleSwarmResultToken implements Token {
  constructor(readonly result: ParticleSwarmResult) {}
}

class ParticleSwarmStation extends NumericSwarmOptimizerStation {
  static readonly CH_START: ChannelName = 'particle-swarm-start';
  static readonly CH_RESULT: ChannelName = 'particle-swarm-result';
  private readonly objectiveName: ContinuousObjectiveName;
  private readonly inertia: number;
  private readonly cognitive: number;
  private readonly social: number;
  private readonly maxVelocity: number;
  private started = false;
  private resultEmitted = false;

  constructor(params: Required<ParticleSwarmParams>) {
    const lower = new Array(params.dimension).fill(params.lower);
    const upper = new Array(params.dimension).fill(params.upper);
    super('particle-swarm-station', {
      particleCount: params.particles,
      dimension: params.dimension,
      iterations: params.iterations,
      lowerBound: lower,
      upperBound: upper,
      rng: mulberry32(params.seed),
    });
    this.objectiveName = params.objective;
    this.inertia = params.inertia;
    this.cognitive = params.cognitive;
    this.social = params.social;
    this.maxVelocity = 0.25 * (params.upper - params.lower);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    Preconditions.nonNegative(this.id, 'inertia', this.inertia);
    Preconditions.nonNegative(this.id, 'cognitive', this.cognitive);
    Preconditions.nonNegative(this.id, 'social', this.social);
  }

  protected objective(position: readonly number[]): number {
    if (this.objectiveName === 'rastrigin') {
      return 10 * position.length + position.reduce((acc, x) => acc + x * x - 10 * Math.cos(2 * Math.PI * x), 0);
    }
    if (this.objectiveName === 'rosenbrock') {
      let value = 0;
      for (let i = 0; i + 1 < position.length; i++) value += 100 * sq(position[i + 1] - sq(position[i])) + sq(1 - position[i]);
      return value;
    }
    return position.reduce((acc, x) => acc + x * x, 0);
  }

  protected updateParticle(
    particle: NumericSwarmParticle,
    globalBest: readonly number[],
    _iteration: number,
  ): NumericSwarmParticle {
    const next = {...particle, position: particle.position.slice(), velocity: particle.velocity.slice()};
    for (let i = 0; i < next.position.length; i++) {
      const rp = this.rng();
      const rg = this.rng();
      const velocity =
        this.inertia * next.velocity[i] +
        this.cognitive * rp * (next.bestPosition[i] - next.position[i]) +
        this.social * rg * (globalBest[i] - next.position[i]);
      next.velocity[i] = clamp(velocity, -this.maxVelocity, this.maxVelocity);
      next.position[i] += next.velocity[i];
    }
    return next;
  }

  override hasWork(): boolean {
    if (!this.started) return this.inboxSize(ParticleSwarmStation.CH_START) > 0;
    return !this.resultEmitted;
  }

  override runTimeStep(): void {
    if (!this.started) {
      const starts = this.drain<OptimizationStartToken<Required<ParticleSwarmParams>>>(ParticleSwarmStation.CH_START);
      if (starts.length === 0) return;
      validateParticleSwarmParams('particle-swarm-source', starts[starts.length - 1].params);
      this.bootstrap();
      this.started = true;
      return;
    }
    if (!this.finished) super.runTimeStep();
    if (this.finished && !this.resultEmitted) {
      this.emit(new ParticleSwarmResultToken(makeParticleSwarmResult(this)), ParticleSwarmStation.CH_RESULT);
      this.resultEmitted = true;
    }
  }
}

function validateParticleSwarmParams(model: string, params: Required<ParticleSwarmParams>): void {
  Preconditions.integerInRange(model, 'dimension', params.dimension, 1, 1e6);
  Preconditions.integerInRange(model, 'particles', params.particles, 1, 1e9);
  Preconditions.integerInRange(model, 'iterations', params.iterations, 1, 1e9);
  Preconditions.finite(model, 'lower', params.lower);
  Preconditions.finite(model, 'upper', params.upper);
  Preconditions.check(model, 'bounds', 'satisfy lower < upper',
    params.lower < params.upper, [params.lower, params.upper]);
  Preconditions.nonNegative(model, 'inertia', params.inertia);
  Preconditions.nonNegative(model, 'cognitive', params.cognitive);
  Preconditions.nonNegative(model, 'social', params.social);
}

function makeParticleSwarmResult(station: ParticleSwarmStation): ParticleSwarmResult {
  return {
    bestPosition: station.getBestPosition(),
    bestValue: station.getBestScore(),
    iterations: station.getIteration(),
    trace: station.trace.map(row => ({
      iteration: row.iteration,
      bestValue: row.bestScore,
      meanValue: row.meanScore ?? NaN,
      worstValue: row.worstScore ?? NaN,
    })),
    topology: stationGraphTopology(
      ['particle-swarm-source', 'particle-swarm-station', 'particle-swarm-result-sink'],
      [
        'OptimizationStartToken<particle-swarm>',
        'OptimizationCandidateToken<swarm-particle>',
        'NumericSwarmParticle',
        'ParticleSwarmResultToken',
      ],
    ),
  };
}

export function runParticleSwarm(params: ParticleSwarmParams = {}): ParticleSwarmResult {
  const defaults: Required<ParticleSwarmParams> = {
    objective: params.objective ?? 'sphere',
    dimension: params.dimension ?? 3,
    particles: params.particles ?? 32,
    iterations: params.iterations ?? 120,
    lower: params.lower ?? -5,
    upper: params.upper ?? 5,
    inertia: params.inertia ?? 0.68,
    cognitive: params.cognitive ?? 1.45,
    social: params.social ?? 1.45,
    seed: params.seed ?? 11,
  };
  validateParticleSwarmParams('runParticleSwarm', defaults);
  const source = new SingleTokenSourceStation<OptimizationStartToken<Required<ParticleSwarmParams>>>(
    'particle-swarm-source',
    ParticleSwarmStation.CH_START,
    () => new OptimizationStartToken('particle-swarm', defaults),
    token => validateParticleSwarmParams('particle-swarm-source', token.params),
  );
  const station = new ParticleSwarmStation(defaults);
  const sink = new LatestTokenSinkStation<ParticleSwarmResultToken>('particle-swarm-result-sink', ParticleSwarmStation.CH_RESULT);
  source.pipe(station, ParticleSwarmStation.CH_START, ParticleSwarmStation.CH_START);
  station.pipe(sink, ParticleSwarmStation.CH_RESULT, ParticleSwarmStation.CH_RESULT);
  runIterativeDES([source, station, sink], {shuffle: false, maxTicks: defaults.iterations + 5, runValidators: false});
  if (!sink.latest) throw new Error('particle-swarm did not produce a result');
  return sink.latest.result;
}

// -----------------------------------------------------------------------------
// 24. Ant Colony Optimization on a TSP graph
// -----------------------------------------------------------------------------

export interface Point2 {x: number; y: number}

export interface AntColonyTSPParams {
  points?: Point2[];
  ants?: number;
  iterations?: number;
  alpha?: number;
  beta?: number;
  evaporation?: number;
  deposit?: number;
  seed?: number;
}

export interface AntColonyTSPResult {
  bestTour: number[];
  bestLength: number;
  iterations: number;
  trace: Array<{iteration: number; bestLength: number; meanLength: number; worstLength: number}>;
  topology: StationGraphTopology;
}

class AntColonyTSPResultToken implements Token {
  constructor(readonly result: AntColonyTSPResult) {}
}

class AntColonyTSPStation extends PheromoneGraphSearchStation {
  static readonly CH_START: ChannelName = 'ant-colony-tsp-start';
  static readonly CH_RESULT: ChannelName = 'ant-colony-tsp-result';
  private readonly points: Point2[];
  private started = false;
  private resultEmitted = false;

  constructor(params: Required<AntColonyTSPParams>) {
    super('ant-colony-tsp-station', {
      nodeCount: params.points.length,
      ants: params.ants,
      iterations: params.iterations,
      alpha: params.alpha,
      beta: params.beta,
      evaporation: params.evaporation,
      deposit: params.deposit,
      rng: mulberry32(params.seed),
    });
    this.points = params.points.map(p => ({x: p.x, y: p.y}));
  }

  protected pathCost(path: readonly number[]): number {
    let value = 0;
    for (let i = 1; i < path.length; i++) value += dist(this.points[path[i - 1]], this.points[path[i]]);
    return value;
  }

  protected heuristic(from: number, to: number): number {
    return 1 / Math.max(1e-9, dist(this.points[from], this.points[to]));
  }

  override hasWork(): boolean {
    if (!this.started) return this.inboxSize(AntColonyTSPStation.CH_START) > 0;
    return !this.resultEmitted;
  }

  override runTimeStep(): void {
    if (!this.started) {
      const starts = this.drain<OptimizationStartToken<Required<AntColonyTSPParams>>>(AntColonyTSPStation.CH_START);
      if (starts.length === 0) return;
      validateAntColonyTSPParams('ant-colony-tsp-source', starts[starts.length - 1].params);
      this.started = true;
      return;
    }
    if (!this.finished) super.runTimeStep();
    if (this.finished && !this.resultEmitted) {
      this.emit(new AntColonyTSPResultToken(makeAntColonyTSPResult(this)), AntColonyTSPStation.CH_RESULT);
      this.resultEmitted = true;
    }
  }
}

function validateAntColonyTSPParams(model: string, params: Required<AntColonyTSPParams>): void {
  const points = params.points;
  Preconditions.check(model, 'points.length', 'be at least 2', points.length >= 2, points.length);
  const seenPoints = new Set<string>();
  for (let i = 0; i < points.length; i++) {
    Preconditions.finite(model, `points[${i}].x`, points[i].x);
    Preconditions.finite(model, `points[${i}].y`, points[i].y);
    const key = `${points[i].x}:${points[i].y}`;
    Preconditions.check(model, `points[${i}]`, 'be a unique coordinate',
      !seenPoints.has(key), points[i]);
    seenPoints.add(key);
  }
  Preconditions.integerInRange(model, 'ants', params.ants, 1, 1e9);
  Preconditions.integerInRange(model, 'iterations', params.iterations, 1, 1e9);
  Preconditions.nonNegative(model, 'alpha', params.alpha);
  Preconditions.nonNegative(model, 'beta', params.beta);
  Preconditions.inRange(model, 'evaporation', params.evaporation, 0, 1);
  Preconditions.positive(model, 'deposit', params.deposit);
}

function makeAntColonyTSPResult(station: AntColonyTSPStation): AntColonyTSPResult {
  return {
    bestTour: station.getBestPath(),
    bestLength: station.getBestCost(),
    iterations: station.getIteration(),
    trace: station.trace.map(row => ({
      iteration: row.iteration,
      bestLength: row.bestScore,
      meanLength: row.meanScore ?? NaN,
      worstLength: row.worstScore ?? NaN,
    })),
    topology: stationGraphTopology(
      ['ant-colony-tsp-source', 'ant-colony-tsp-station', 'ant-colony-tsp-result-sink'],
      ['OptimizationStartToken<ant-colony-tsp>', 'GraphWalkToken', 'pheromone-matrix-state', 'AntColonyTSPResultToken'],
    ),
  };
}

export function runAntColonyTSP(params: AntColonyTSPParams = {}): AntColonyTSPResult {
  const points = params.points ?? defaultTSPPoints();
  const defaults: Required<AntColonyTSPParams> = {
    points,
    ants: params.ants ?? Math.max(12, points.length * 3),
    iterations: params.iterations ?? 80,
    alpha: params.alpha ?? 1,
    beta: params.beta ?? 3,
    evaporation: params.evaporation ?? 0.28,
    deposit: params.deposit ?? 1,
    seed: params.seed ?? 5,
  };
  validateAntColonyTSPParams('runAntColonyTSP', defaults);
  const source = new SingleTokenSourceStation<OptimizationStartToken<Required<AntColonyTSPParams>>>(
    'ant-colony-tsp-source',
    AntColonyTSPStation.CH_START,
    () => new OptimizationStartToken('ant-colony-tsp', defaults),
    token => validateAntColonyTSPParams('ant-colony-tsp-source', token.params),
  );
  const station = new AntColonyTSPStation(defaults);
  const sink = new LatestTokenSinkStation<AntColonyTSPResultToken>('ant-colony-tsp-result-sink', AntColonyTSPStation.CH_RESULT);
  source.pipe(station, AntColonyTSPStation.CH_START, AntColonyTSPStation.CH_START);
  station.pipe(sink, AntColonyTSPStation.CH_RESULT, AntColonyTSPStation.CH_RESULT);
  runIterativeDES([source, station, sink], {shuffle: false, maxTicks: defaults.iterations + 5, runValidators: false});
  if (!sink.latest) throw new Error('ant-colony-tsp did not produce a result');
  return sink.latest.result;
}

function defaultTSPPoints(): Point2[] {
  return [
    {x: 0, y: 0},
    {x: 1.5, y: 0.3},
    {x: 2.4, y: 1.7},
    {x: 1.4, y: 2.8},
    {x: -0.2, y: 2.2},
    {x: -0.8, y: 0.9},
  ];
}

// -----------------------------------------------------------------------------
// 25. Constraint Satisfaction Problem: map coloring
// -----------------------------------------------------------------------------

export interface MapColoringCSPParams {
  variables?: string[];
  colors?: string[];
  edges?: Array<[string, string]>;
  maxNodes?: number;
}

export interface MapColoringCSPResult {
  assignment: Record<string, string>;
  satisfied: boolean;
  nodesProcessed: number;
  topology: StationGraphTopology;
}

class MapColoringCSPResultToken implements Token {
  constructor(readonly result: MapColoringCSPResult) {}
}

class MapColoringCSPStation extends SourceDrivenConstraintSatisfactionSearchStation<
  string,
  OptimizationStartToken<Required<MapColoringCSPParams>>,
  MapColoringCSPResultToken
> {
  static readonly CH_START: ChannelName = 'map-coloring-csp-start';
  static readonly CH_RESULT: ChannelName = 'map-coloring-csp-result';
  private readonly edges: Array<[string, string]>;

  constructor(params: Required<MapColoringCSPParams>) {
    const domains: Record<string, string[]> = {};
    for (const variable of params.variables) domains[variable] = params.colors.slice();
    super(
      'map-coloring-csp-station',
      params.variables,
      domains,
      params.maxNodes,
      MapColoringCSPStation.CH_START,
      MapColoringCSPStation.CH_RESULT,
    );
    this.edges = params.edges.map(edge => [edge[0], edge[1]]);
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    const known = new Set(this.getVariables());
    for (const [a, b] of this.edges) {
      Preconditions.check(this.id, `edge ${a}-${b}`, 'reference known variables',
        known.has(a) && known.has(b), [a, b]);
    }
  }

  protected isConsistent(assignment: Readonly<Record<string, string>>): boolean {
    for (const [a, b] of this.edges) {
      if (assignment[a] !== undefined && assignment[b] !== undefined && assignment[a] === assignment[b]) return false;
    }
    return true;
  }

  checkAssignment(assignment: Readonly<Record<string, string>>): boolean {
    return this.getVariables().every(v => assignment[v] !== undefined) && this.isConsistent(assignment);
  }

  protected acceptStartToken(token: OptimizationStartToken<Required<MapColoringCSPParams>>): void {
    Preconditions.check(this.id, 'start model', 'match map-coloring-csp',
      token.model === 'map-coloring-csp', token.model);
    validateMapColoringCSPParams(`${this.id}.start`, token.params);
  }

  protected makeResultToken(): MapColoringCSPResultToken {
    return new MapColoringCSPResultToken(this.result());
  }

  private result(): MapColoringCSPResult {
    const assignment = this.getSolution() ?? {};
    return {
      assignment,
      satisfied: this.checkAssignment(assignment),
      nodesProcessed: this.getNodesProcessed(),
      topology: stationGraphTopology(
        ['map-coloring-csp-source', 'map-coloring-csp-station', 'map-coloring-csp-result-sink'],
        ['OptimizationStartToken<map-coloring-csp>', 'ConstraintAssignmentToken', 'MapColoringCSPResultToken'],
      ),
    };
  }
}

function validateMapColoringCSPParams(model: string, params: Required<MapColoringCSPParams>): void {
  Preconditions.nonEmpty(model, 'variables', params.variables);
  Preconditions.nonEmpty(model, 'colors', params.colors);
  Preconditions.integerInRange(model, 'maxNodes', params.maxNodes, 1, 1e9);
  Preconditions.check(model, 'variables', 'be unique',
    new Set(params.variables).size === params.variables.length, params.variables);
  Preconditions.check(model, 'colors', 'be unique',
    new Set(params.colors).size === params.colors.length, params.colors);
  const known = new Set(params.variables);
  for (const [i, edge] of params.edges.entries()) {
    Preconditions.lengthEq(model, `edges[${i}]`, edge, 2);
    const [a, b] = edge;
    Preconditions.check(model, `edges[${i}]`, 'reference known variables',
      known.has(a) && known.has(b), edge);
    Preconditions.check(model, `edges[${i}]`, 'connect two distinct variables', a !== b, edge);
  }
}

export function runMapColoringCSP(params: MapColoringCSPParams = {}): MapColoringCSPResult {
  const variables = params.variables ?? ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T'];
  const colors = params.colors ?? ['red', 'green', 'blue'];
  const edges = params.edges ?? [
    ['WA', 'NT'], ['WA', 'SA'], ['NT', 'SA'], ['NT', 'Q'], ['SA', 'Q'],
    ['SA', 'NSW'], ['SA', 'V'], ['Q', 'NSW'], ['NSW', 'V'],
  ] as Array<[string, string]>;
  const defaults: Required<MapColoringCSPParams> = {
    variables,
    colors,
    edges,
    maxNodes: params.maxNodes ?? 10_000,
  };
  validateMapColoringCSPParams('runMapColoringCSP', defaults);
  const source = new SingleTokenSourceStation<OptimizationStartToken<Required<MapColoringCSPParams>>>(
    'map-coloring-csp-source',
    MapColoringCSPStation.CH_START,
    () => new OptimizationStartToken('map-coloring-csp', defaults),
    token => validateMapColoringCSPParams('map-coloring-csp-source', token.params),
  );
  const station = new MapColoringCSPStation(defaults);
  const sink = new LatestTokenSinkStation<MapColoringCSPResultToken>(
    'map-coloring-csp-result-sink',
    MapColoringCSPStation.CH_RESULT,
  );
  source.pipe(station, MapColoringCSPStation.CH_START, MapColoringCSPStation.CH_START);
  station.pipe(sink, MapColoringCSPStation.CH_RESULT, MapColoringCSPStation.CH_RESULT);
  runIterativeDES([source, station, sink], {shuffle: false, maxTicks: defaults.maxNodes + 4, runValidators: false});
  const resultToken = sink.latest;
  Preconditions.check('runMapColoringCSP', 'result token', 'be emitted by map-coloring-csp-station',
    resultToken !== undefined, sink.id);
  if (!resultToken) throw new Error('runMapColoringCSP: result token was not emitted');
  return resultToken.result;
}

// -----------------------------------------------------------------------------
// 26. SAT / MAX-SAT local search
// -----------------------------------------------------------------------------

export interface MaxSATParams {
  numVars?: number;
  clauses?: number[][];
  iterations?: number;
  noise?: number;
  seed?: number;
}

export interface MaxSATResult {
  assignment: boolean[];
  satisfiedClauses: number;
  totalClauses: number;
  allSatisfied: boolean;
  iterations: number;
  trace: Array<{iteration: number; unsatisfied: number}>;
  topology: StationGraphTopology;
}

class MaxSATResultToken implements Token {
  constructor(readonly result: MaxSATResult) {}
}

class MaxSATLocalSearchStation extends SingleStateOptimizer<boolean[]> {
  static readonly CH_START: ChannelName = 'max-sat-local-search-start';
  static readonly CH_RESULT: ChannelName = 'max-sat-local-search-result';
  private readonly numVars: number;
  private readonly clauses: number[][];
  private readonly iterations: number;
  private readonly noise: number;
  private started = false;
  private maxSatResultEmitted = false;

  constructor(params: Required<MaxSATParams>) {
    super('max-sat-local-search-station', {rng: mulberry32(params.seed)});
    this.numVars = params.numVars;
    this.clauses = params.clauses.map(c => c.slice());
    this.iterations = params.iterations;
    this.noise = params.noise;
    this.assertPreconditions();
  }

  override assertPreconditions(): void {
    Preconditions.integerInRange(this.id, 'numVars', this.numVars, 1, 1e9);
    Preconditions.nonEmpty(this.id, 'clauses', this.clauses);
    Preconditions.integerInRange(this.id, 'iterations', this.iterations, 1, 1e9);
    Preconditions.inRange(this.id, 'noise', this.noise, 0, 1);
    for (let i = 0; i < this.clauses.length; i++) {
      Preconditions.nonEmpty(this.id, `clauses[${i}]`, this.clauses[i]);
      for (const lit of this.clauses[i]) {
        Preconditions.integer(this.id, `literal ${lit}`, lit);
        Preconditions.check(this.id, `literal ${lit}`, 'refer to a variable in [1, numVars]',
          Math.abs(lit) >= 1 && Math.abs(lit) <= this.numVars, lit);
      }
    }
  }

  protected initialState(rng: () => number): boolean[] {
    return randomBooleanAssignment(this.numVars, rng);
  }

  protected cost(state: boolean[]): number {
    return this.clauses.length - countSatisfied(this.clauses, state);
  }

  protected propose(state: boolean[], rng: () => number): boolean[] {
    const unsat = this.clauses.filter(clause => !clauseSatisfied(clause, state));
    const next = state.slice();
    if (unsat.length === 0) return next;
    const clause = unsat[Math.floor(rng() * unsat.length)];
    let variable = Math.abs(clause[Math.floor(rng() * clause.length)]) - 1;
    if (rng() >= this.noise) {
      let bestVar = variable;
      let bestScore = -Infinity;
      for (const lit of clause) {
        const idx = Math.abs(lit) - 1;
        const trial = state.slice();
        trial[idx] = !trial[idx];
        const score = countSatisfied(this.clauses, trial);
        if (score > bestScore) {
          bestScore = score;
          bestVar = idx;
        }
      }
      variable = bestVar;
    }
    next[variable] = !next[variable];
    return next;
  }

  protected accept(): boolean { return true; }
  protected clone(state: boolean[]): boolean[] { return state.slice(); }
  protected shouldStop(iter: number): boolean { return iter >= this.iterations || this.getBestCost() === 0; }

  override hasWork(): boolean {
    if (!this.started) return this.inboxSize(MaxSATLocalSearchStation.CH_START) > 0;
    return !this.maxSatResultEmitted;
  }

  override runTimeStep(): void {
    if (!this.started) {
      const starts = this.drain<OptimizationStartToken<Required<MaxSATParams>>>(MaxSATLocalSearchStation.CH_START);
      if (starts.length === 0) return;
      validateMaxSATParams('max-sat-local-search-source', starts[starts.length - 1].params);
      this.bootstrap();
      this.started = true;
      return;
    }
    if (!this.finished) super.runTimeStep();
    if (this.finished && !this.maxSatResultEmitted) {
      this.emit(new MaxSATResultToken(this.result()), MaxSATLocalSearchStation.CH_RESULT);
      this.maxSatResultEmitted = true;
    }
  }

  private result(): MaxSATResult {
    const assignment = this.getBest();
    const satisfiedClauses = countSatisfied(this.clauses, assignment);
    return {
      assignment,
      satisfiedClauses,
      totalClauses: this.clauses.length,
      allSatisfied: satisfiedClauses === this.clauses.length,
      iterations: this.getIteration(),
      trace: this.bestHistory.map((unsatisfied, i) => ({iteration: i, unsatisfied})),
      topology: stationGraphTopology(
        ['max-sat-local-search-source', 'max-sat-local-search-station', 'max-sat-local-search-result-sink'],
        ['OptimizationStartToken<max-sat>', 'boolean-assignment-state', 'OptimizationCandidateToken<boolean[]>', 'MaxSATResultToken'],
      ),
    };
  }
}

function validateMaxSATParams(model: string, params: Required<MaxSATParams>): void {
  Preconditions.integerInRange(model, 'numVars', params.numVars, 1, 1e9);
  Preconditions.nonEmpty(model, 'clauses', params.clauses);
  Preconditions.integerInRange(model, 'iterations', params.iterations, 1, 1e9);
  Preconditions.inRange(model, 'noise', params.noise, 0, 1);
  for (let i = 0; i < params.clauses.length; i++) {
    Preconditions.nonEmpty(model, `clauses[${i}]`, params.clauses[i]);
    for (const lit of params.clauses[i]) {
      Preconditions.integer(model, `literal ${lit}`, lit);
      Preconditions.check(model, `literal ${lit}`, 'refer to a variable in [1, numVars]',
        Math.abs(lit) >= 1 && Math.abs(lit) <= params.numVars, lit);
    }
  }
}

export function runMaxSATLocalSearch(params: MaxSATParams = {}): MaxSATResult {
  const clauses = params.clauses && params.clauses.length > 0 ? params.clauses : defaultMaxSATClauses();
  const inferredVars = clauses.reduce((acc, clause) => Math.max(acc, ...clause.map(lit => Math.abs(lit))), 0);
  const defaults: Required<MaxSATParams> = {
    numVars: params.numVars ?? inferredVars,
    clauses,
    iterations: params.iterations ?? 300,
    noise: params.noise ?? 0.25,
    seed: params.seed ?? 13,
  };
  validateMaxSATParams('runMaxSATLocalSearch', defaults);
  const source = new SingleTokenSourceStation<OptimizationStartToken<Required<MaxSATParams>>>(
    'max-sat-local-search-source',
    MaxSATLocalSearchStation.CH_START,
    () => new OptimizationStartToken('max-sat-local-search', defaults),
    token => validateMaxSATParams('max-sat-local-search-source', token.params),
  );
  const station = new MaxSATLocalSearchStation(defaults);
  const sink = new LatestTokenSinkStation<MaxSATResultToken>('max-sat-local-search-result-sink', MaxSATLocalSearchStation.CH_RESULT);
  source.pipe(station, MaxSATLocalSearchStation.CH_START, MaxSATLocalSearchStation.CH_START);
  station.pipe(sink, MaxSATLocalSearchStation.CH_RESULT, MaxSATLocalSearchStation.CH_RESULT);
  runIterativeDES([source, station, sink], {shuffle: false, maxTicks: defaults.iterations + 5, runValidators: false});
  if (!sink.latest) throw new Error('max-sat-local-search did not produce a result');
  return sink.latest.result;
}

function randomBooleanAssignment(numVars: number, rng: () => number): boolean[] {
  return Array.from({length: numVars}, () => rng() < 0.5);
}

function defaultMaxSATClauses(): number[][] {
  return [
    [1, 2, -3],
    [-1, 3],
    [2, 4],
    [-2, -4],
    [1, -4],
    [-1, -2, 3],
  ];
}

function literalSatisfied(lit: number, assignment: readonly boolean[]): boolean {
  const value = assignment[Math.abs(lit) - 1];
  return lit > 0 ? value : !value;
}

function clauseSatisfied(clause: readonly number[], assignment: readonly boolean[]): boolean {
  return clause.some(lit => literalSatisfied(lit, assignment));
}

function countSatisfied(clauses: readonly number[][], assignment: readonly boolean[]): number {
  return clauses.filter(clause => clauseSatisfied(clause, assignment)).length;
}

// -----------------------------------------------------------------------------
// 27. SDP relaxation: Max-Cut via rank-constrained unit vectors
// -----------------------------------------------------------------------------

export interface WeightedEdge {
  i: number;
  j: number;
  weight: number;
}

export interface SDPMaxCutParams {
  nodes?: number;
  edges?: WeightedEdge[];
  rank?: number;
  iterations?: number;
  stepSize?: number;
  seed?: number;
}

export interface SDPMaxCutResult {
  sdpValue: number;
  roundedCutValue: number;
  cut: number[];
  gramMatrix: number[][];
  iterations: number;
  trace: Array<{iteration: number; objective: number}>;
  topology: StationGraphTopology;
}

class SDPMaxCutResultToken implements Token {
  constructor(readonly result: SDPMaxCutResult) {}
}

class MaxCutSDPStation extends UnitVectorRelaxationStation {
  static readonly CH_START: ChannelName = 'sdp-maxcut-relaxation-start';
  static readonly CH_RESULT: ChannelName = 'sdp-maxcut-relaxation-result';
  private readonly edges: WeightedEdge[];
  private started = false;
  private resultEmitted = false;

  constructor(params: Required<SDPMaxCutParams>) {
    super('sdp-maxcut-relaxation-station', {
      nodes: params.nodes,
      rank: params.rank,
      iterations: params.iterations,
      stepSize: params.stepSize,
      rng: mulberry32(params.seed),
    });
    this.edges = params.edges.map(edge => ({...edge}));
    this.assertPreconditions();
  }

  override assertPreconditions(): void {
    super.assertPreconditions();
    for (const edge of this.edges) {
      Preconditions.integerInRange(this.id, 'edge.i', edge.i, 0, this.nodes - 1);
      Preconditions.integerInRange(this.id, 'edge.j', edge.j, 0, this.nodes - 1);
      Preconditions.positive(this.id, 'edge.weight', edge.weight);
    }
  }

  protected objective(vectors: readonly number[][]): number {
    let value = 0;
    for (const edge of this.edges) {
      value += edge.weight * (1 - vectorDot(vectors[edge.i], vectors[edge.j])) / 2;
    }
    return value;
  }

  protected gradient(vectors: readonly number[][]): number[][] {
    const grad = Array.from({length: this.nodes}, () => new Array(this.rank).fill(0));
    for (const edge of this.edges) {
      for (let k = 0; k < this.rank; k++) {
        grad[edge.i][k] += -0.5 * edge.weight * vectors[edge.j][k];
        grad[edge.j][k] += -0.5 * edge.weight * vectors[edge.i][k];
      }
    }
    return grad;
  }

  override hasWork(): boolean {
    if (!this.started) return this.inboxSize(MaxCutSDPStation.CH_START) > 0;
    return !this.resultEmitted;
  }

  override runTimeStep(): void {
    if (!this.started) {
      const starts = this.drain<OptimizationStartToken<Required<SDPMaxCutParams>>>(MaxCutSDPStation.CH_START);
      if (starts.length === 0) return;
      validateSDPMaxCutParams('sdp-maxcut-relaxation-source', starts[starts.length - 1].params);
      this.bootstrap();
      this.started = true;
      return;
    }
    if (!this.finished) super.runTimeStep();
    if (this.finished && !this.resultEmitted) {
      this.emit(new SDPMaxCutResultToken(this.result()), MaxCutSDPStation.CH_RESULT);
      this.resultEmitted = true;
    }
  }

  private result(): SDPMaxCutResult {
    const rounded = bestHyperplaneCut(this.getBestVectors(), this.edges);
    return {
      sdpValue: this.getBestObjective(),
      roundedCutValue: rounded.value,
      cut: rounded.cut,
      gramMatrix: this.getGramMatrix(),
      iterations: this.getIteration(),
      trace: this.trace,
      topology: stationGraphTopology(
        ['sdp-maxcut-relaxation-source', 'sdp-maxcut-relaxation-station', 'sdp-maxcut-relaxation-result-sink'],
        ['OptimizationStartToken<sdp-maxcut>', 'unit-vector-state', 'GramMatrixToken', 'SDPMaxCutResultToken'],
      ),
    };
  }
}

function validateSDPMaxCutParams(model: string, params: Required<SDPMaxCutParams>): void {
  Preconditions.integerInRange(model, 'nodes', params.nodes, 2, 1e6);
  Preconditions.integerInRange(model, 'rank', params.rank, 1, 1e6);
  Preconditions.integerInRange(model, 'iterations', params.iterations, 1, 1e9);
  Preconditions.positive(model, 'stepSize', params.stepSize);
  Preconditions.nonEmpty(model, 'edges', params.edges);
  for (const edge of params.edges) {
    Preconditions.integerInRange(model, 'edge.i', edge.i, 0, params.nodes - 1);
    Preconditions.integerInRange(model, 'edge.j', edge.j, 0, params.nodes - 1);
    Preconditions.positive(model, 'edge.weight', edge.weight);
  }
}

export function runSDPMaxCutRelaxation(params: SDPMaxCutParams = {}): SDPMaxCutResult {
  const edges = params.edges && params.edges.length > 0 ? params.edges : defaultMaxCutEdges();
  const nodes = params.nodes ?? 5;
  const defaults: Required<SDPMaxCutParams> = {
    nodes,
    edges,
    rank: params.rank ?? 3,
    iterations: params.iterations ?? 250,
    stepSize: params.stepSize ?? 0.08,
    seed: params.seed ?? 17,
  };
  validateSDPMaxCutParams('runSDPMaxCutRelaxation', defaults);
  const source = new SingleTokenSourceStation<OptimizationStartToken<Required<SDPMaxCutParams>>>(
    'sdp-maxcut-relaxation-source',
    MaxCutSDPStation.CH_START,
    () => new OptimizationStartToken('sdp-maxcut-relaxation', defaults),
    token => validateSDPMaxCutParams('sdp-maxcut-relaxation-source', token.params),
  );
  const station = new MaxCutSDPStation(defaults);
  const sink = new LatestTokenSinkStation<SDPMaxCutResultToken>('sdp-maxcut-relaxation-result-sink', MaxCutSDPStation.CH_RESULT);
  source.pipe(station, MaxCutSDPStation.CH_START, MaxCutSDPStation.CH_START);
  station.pipe(sink, MaxCutSDPStation.CH_RESULT, MaxCutSDPStation.CH_RESULT);
  runIterativeDES([source, station, sink], {shuffle: false, maxTicks: defaults.iterations + 5, runValidators: false});
  if (!sink.latest) throw new Error('sdp-maxcut-relaxation did not produce a result');
  return sink.latest.result;
}

function defaultMaxCutEdges(): WeightedEdge[] {
  return [
    {i: 0, j: 1, weight: 1},
    {i: 1, j: 2, weight: 1},
    {i: 2, j: 3, weight: 1},
    {i: 3, j: 4, weight: 1},
    {i: 4, j: 0, weight: 1},
    {i: 0, j: 2, weight: 0.7},
    {i: 1, j: 3, weight: 0.5},
    {i: 2, j: 4, weight: 0.8},
  ];
}

function bestHyperplaneCut(vectors: readonly number[][], edges: readonly WeightedEdge[]): {cut: number[]; value: number} {
  const directions = vectors.concat(basisDirections(vectors[0]?.length ?? 1));
  let bestCut = new Array(vectors.length).fill(0);
  let bestValue = -Infinity;
  for (const dir of directions) {
    const cut = vectors.map(v => vectorDot(v, dir) >= 0 ? 1 : -1);
    const value = cutValue(cut, edges);
    if (value > bestValue) {
      bestValue = value;
      bestCut = cut;
    }
  }
  return {cut: bestCut, value: bestValue};
}

function basisDirections(rank: number): number[][] {
  return Array.from({length: rank}, (_, i) => {
    const v = new Array(rank).fill(0);
    v[i] = 1;
    return v;
  });
}

function cutValue(cut: readonly number[], edges: readonly WeightedEdge[]): number {
  let value = 0;
  for (const edge of edges) if (cut[edge.i] !== cut[edge.j]) value += edge.weight;
  return value;
}

// -----------------------------------------------------------------------------
// 29. Multi-objective optimization: Pareto portfolio archive
// -----------------------------------------------------------------------------

export interface PortfolioAsset {
  name: string;
  expectedReturn: number;
  risk: number;
}

export interface ParetoPortfolioParams {
  assets?: PortfolioAsset[];
  samples?: number;
  seed?: number;
}

export interface ParetoPortfolioPoint {
  weights: number[];
  expectedReturn: number;
  risk: number;
}

export interface ParetoPortfolioResult {
  paretoFront: ParetoPortfolioPoint[];
  candidateCount: number;
  hypervolume: number;
  topology: StationGraphTopology;
}

class ParetoCandidateSourceStation<T> extends DESStation {
  private emitted = false;

  constructor(id: string, private readonly candidates: readonly ParetoCandidateToken<T>[]) {
    super(id);
  }

  override assertPreconditions(): void {
    Preconditions.nonEmpty(this.id, 'candidates', this.candidates);
    for (const candidate of this.candidates) {
      Preconditions.nonEmpty(this.id, 'candidate.objectives', candidate.objectives);
      Preconditions.allFinite(this.id, 'candidate.objectives', candidate.objectives);
    }
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    for (const candidate of this.candidates) this.emit(candidate);
    this.emitted = true;
  }
}

export function runParetoPortfolio(params: ParetoPortfolioParams = {}): ParetoPortfolioResult {
  const assets = params.assets && params.assets.length > 0 ? params.assets : defaultPortfolioAssets();
  const samples = params.samples ?? 240;
  Preconditions.integerInRange('runParetoPortfolio', 'samples', samples, 1, 1e9);
  for (const asset of assets) {
    Preconditions.finite('runParetoPortfolio', `${asset.name}.expectedReturn`, asset.expectedReturn);
    Preconditions.nonNegative('runParetoPortfolio', `${asset.name}.risk`, asset.risk);
  }
  const rng = mulberry32(params.seed ?? 19);
  const candidates: Array<ParetoCandidateToken<ParetoPortfolioPoint>> = [];
  for (let i = 0; i < samples; i++) {
    const weights = randomSimplex(assets.length, rng);
    const point = portfolioPoint(assets, weights);
    candidates.push(new ParetoCandidateToken(point, [point.risk, -point.expectedReturn], i));
  }
  for (let i = 0; i < assets.length; i++) {
    const weights = new Array(assets.length).fill(0);
    weights[i] = 1;
    const point = portfolioPoint(assets, weights);
    candidates.push(new ParetoCandidateToken(point, [point.risk, -point.expectedReturn], samples + i));
  }
  const source = new ParetoCandidateSourceStation<ParetoPortfolioPoint>('pareto-portfolio-source', candidates);
  const station = new ParetoArchiveStation<ParetoPortfolioPoint>('pareto-portfolio-archive');
  source.pipe(station);
  runIterativeDES([source, station], {shuffle: false, maxTicks: candidates.length + 3, runValidators: false});
  const paretoFront = station.getArchive()
    .map(row => row.candidate)
    .sort((a, b) => a.risk - b.risk || a.expectedReturn - b.expectedReturn);
  return {
    paretoFront,
    candidateCount: station.getProcessedCount(),
    hypervolume: portfolioHypervolume(paretoFront),
    topology: stationGraphTopology(['pareto-portfolio-source', 'pareto-portfolio-archive'], ['ParetoCandidateToken<portfolio>', 'ParetoArchiveRow']),
  };
}

function defaultPortfolioAssets(): PortfolioAsset[] {
  return [
    {name: 'cash', expectedReturn: 0.02, risk: 0.01},
    {name: 'bonds', expectedReturn: 0.045, risk: 0.06},
    {name: 'equity', expectedReturn: 0.09, risk: 0.18},
    {name: 'growth', expectedReturn: 0.13, risk: 0.30},
  ];
}

function randomSimplex(n: number, rng: () => number): number[] {
  const draws = Array.from({length: n}, () => -Math.log(Math.max(1e-12, rng())));
  const total = draws.reduce((a, b) => a + b, 0);
  return draws.map(x => x / total);
}

function portfolioPoint(assets: readonly PortfolioAsset[], weights: readonly number[]): ParetoPortfolioPoint {
  let expectedReturn = 0;
  let variance = 0;
  for (let i = 0; i < assets.length; i++) {
    expectedReturn += weights[i] * assets[i].expectedReturn;
    variance += sq(weights[i] * assets[i].risk);
  }
  return {weights: weights.slice(), expectedReturn, risk: Math.sqrt(variance)};
}

function portfolioHypervolume(front: readonly ParetoPortfolioPoint[]): number {
  if (front.length === 0) return 0;
  const maxRisk = Math.max(...front.map(p => p.risk)) * 1.1;
  const minReturn = Math.min(...front.map(p => p.expectedReturn)) * 0.9;
  let hv = 0;
  let prevRisk = 0;
  for (const point of front) {
    const width = Math.max(0, point.risk - prevRisk);
    const height = Math.max(0, point.expectedReturn - minReturn);
    hv += width * height;
    prevRisk = point.risk;
  }
  const tailWidth = Math.max(0, maxRisk - prevRisk);
  const last = front[front.length - 1];
  hv += tailWidth * Math.max(0, last.expectedReturn - minReturn);
  return hv;
}

export function paretoFrontIsNondominated(front: readonly ParetoPortfolioPoint[]): boolean {
  const objectives = front.map(p => [p.risk, -p.expectedReturn]);
  for (let i = 0; i < objectives.length; i++) {
    for (let j = 0; j < objectives.length; j++) {
      if (i !== j && dominates(objectives[j], objectives[i])) return false;
    }
  }
  return true;
}
