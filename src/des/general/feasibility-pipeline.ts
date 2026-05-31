'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/feasibility-pipeline.rs  (module des::general::feasibility_pipeline)
// 1:1 file move. Runnable DES pipeline: domain/constraint/objective checks + optional local improver.
//
// Declarations → Rust:
//   type VariableKind/ConstraintSense/ObjectiveSense = '...'|'...' -> enums (+#[serde(rename_all)])
//   const *_CHANNEL                -> `const &str` channel ids (or a ChannelId enum)
//   interface OptimizationVariable/LinearObjective/LinearConstraint/StructuredOptimizationProblem/
//             CandidateSolutionInput/Feasibility{ImprovementOptions,PipelineParams,Violation,Evaluation,
//             PipelineNode,PipelineEdge,PipelineNetwork,PipelineResult}/CandidatePayload -> structs
//   class *Token (impl Token)      -> structs `impl Token`
//   class *Station (extend DESStation) -> structs `impl` the station trait
//   fn runFeasibilityPipeline / evaluateCandidate -> fns
//
// Conversion notes (file-specific):
//   - `ConstraintSense '<='|'>='|'='` / VariableKind / ObjectiveSense -> enums matched with `match`.
//   - `mulberry32(seed)` in the local improver -> inject `RandomSource`.
//   - Tokens/stations are nominal `impl Trait`; channels -> typed queues; `Preconditions` throw -> `Result`.
//   - Depends on internal-solver-network (WallClockCheckerStation/StopSignalToken) — see that header.
// =============================================================================

// =============================================================================
// General optimization feasibility checker pipeline.
//
// A user supplies a structured optimization problem plus their incumbent
// solution. The pipeline checks domains, constraints, and objective value, then
// optionally feeds evaluated candidates into a local internal improver. This is
// intentionally a runnable DES program, not a single monolithic helper:
//
//   CandidateSource -> DomainChecker -> ConstraintChecker -> ObjectiveEvaluator
//                                                      |                 |
//                                                      v                 v
//                                                 SolutionSink <- Improvement
//
// The wall-clock checker station from internal solver networks caps the run.
// =============================================================================

import {DESStation, Token} from './des-base/station';
import {IterativeRunSummary, runIterativeDES} from './des-base/runner';
import {ValidationCheck, intrinsicCheck} from './des-base/validation';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';
import {
  STOP_CHANNEL,
  StopSignalToken,
  WallClockCheckerStation,
} from './internal-solver-network';

export const CANDIDATE_CHANNEL = 'candidate';
export const DOMAIN_CHANNEL = 'domain-checked';
export const CONSTRAINT_CHANNEL = 'constraint-checked';
export const EVALUATION_CHANNEL = 'evaluation';

export type VariableKind = 'continuous' | 'integer' | 'binary';
export type ConstraintSense = '<=' | '>=' | '=';
export type ObjectiveSense = 'min' | 'max';

export interface OptimizationVariable {
  name: string;
  type?: VariableKind;
  lb?: number;
  ub?: number;
  step?: number;
}

export interface LinearObjective {
  constant?: number;
  coefficients: Record<string, number>;
}

export interface LinearConstraint {
  name?: string;
  coefficients: Record<string, number>;
  sense: ConstraintSense;
  rhs: number;
  tolerance?: number;
}

export interface StructuredOptimizationProblem {
  sense: ObjectiveSense;
  variables: OptimizationVariable[];
  objective: LinearObjective;
  constraints?: LinearConstraint[];
  tolerance?: number;
}

export interface CandidateSolutionInput {
  id?: string;
  values?: Record<string, number>;
  vector?: number[];
}

export interface FeasibilityImprovementOptions {
  enabled?: boolean;
  maxIterations?: number;
  seed?: number;
  continuousStep?: number;
  integerStep?: number;
  penalty?: number;
  allowRepair?: boolean;
}

export interface FeasibilityPipelineParams {
  problem: StructuredOptimizationProblem;
  candidate: CandidateSolutionInput;
  improvement?: FeasibilityImprovementOptions;
  timeLimitMs?: number;
  maxTicks?: number;
  checkEveryTicks?: number;
}

export interface CandidatePayload {
  id: string;
  parentId?: string;
  iteration: number;
  origin: 'user' | 'repair' | 'neighbor';
  values: Record<string, number>;
}

export interface FeasibilityViolation {
  kind: 'domain' | 'constraint';
  name: string;
  violation: number;
  message: string;
  variable?: string;
  constraint?: string;
  activity?: number;
  rhs?: number;
}

export interface FeasibilityEvaluation {
  candidateId: string;
  parentId?: string;
  iteration: number;
  origin: CandidatePayload['origin'];
  values: Record<string, number>;
  objectiveValue: number;
  comparableObjective: number;
  totalViolation: number;
  maxViolation: number;
  feasible: boolean;
  domainViolations: FeasibilityViolation[];
  constraintViolations: FeasibilityViolation[];
  violations: FeasibilityViolation[];
  merit: number;
}

export interface FeasibilityPipelineNode {
  id: string;
  kind: string;
  role: 'source' | 'checker' | 'evaluator' | 'improver' | 'sink';
}

export interface FeasibilityPipelineEdge {
  from: string;
  to: string;
  movingEntity: string;
  channel: string;
}

export interface FeasibilityPipelineNetwork {
  stationaryEntities: FeasibilityPipelineNode[];
  movingEntities: Array<{id: string; kind: string; tokenType: string}>;
  edges: FeasibilityPipelineEdge[];
}

export interface FeasibilityPipelineResult {
  status: 'feasible' | 'infeasible' | 'improved' | 'infeasible-improved' | 'time-limit' | 'tick-limit';
  initial: FeasibilityEvaluation;
  best: FeasibilityEvaluation;
  trace: FeasibilityEvaluation[];
  improvements: FeasibilityEvaluation[];
  stopSignals: StopSignalToken['payload'][];
  wallClock: {
    budgetMs: number;
    elapsedMs: number;
    checks: number;
    expired: boolean;
  };
  runSummary: IterativeRunSummary;
  network: FeasibilityPipelineNetwork;
  validation: ValidationCheck[];
}

export class CandidateToken implements Token {
  readonly kind = 'candidate';
  constructor(readonly payload: CandidatePayload) {}
}

export class DomainCheckedToken implements Token {
  readonly kind = 'domain-checked-candidate';
  constructor(
    readonly candidate: CandidatePayload,
    readonly domainViolations: FeasibilityViolation[],
  ) {}
}

export class ConstraintCheckedToken implements Token {
  readonly kind = 'constraint-checked-candidate';
  constructor(
    readonly candidate: CandidatePayload,
    readonly domainViolations: FeasibilityViolation[],
    readonly constraintViolations: FeasibilityViolation[],
  ) {}
}

export class FeasibilityEvaluationToken implements Token {
  readonly kind = 'feasibility-evaluation';
  constructor(readonly payload: FeasibilityEvaluation) {}
}

export class CandidateSourceStation extends DESStation {
  private emitted = false;

  constructor(id: string, private readonly candidate: CandidatePayload) {
    super(id);
  }

  override hasWork(): boolean {
    return !this.emitted;
  }

  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(new CandidateToken(this.candidate), CANDIDATE_CHANNEL);
    this.emitted = true;
  }
}

export class DomainCheckerStation extends DESStation {
  constructor(id: string, private readonly problem: StructuredOptimizationProblem) {
    super(id);
    this.addValidator(intrinsicCheck<DomainCheckerStation>({
      name: 'domain-checker.problem-has-variables',
      group: 'feasibility-pipeline',
      predicate: s => s.problem.variables.length > 0,
      expected: 'at least one variable',
      observedFn: s => String(s.problem.variables.length),
    }));
  }

  override assertPreconditions(): void {
    validateProblem(this.problem);
  }

  override hasWork(): boolean {
    return this.inboxSize(CANDIDATE_CHANNEL) > 0;
  }

  runTimeStep(): void {
    for (const token of this.drain<CandidateToken>(CANDIDATE_CHANNEL)) {
      this.emit(new DomainCheckedToken(token.payload, checkDomain(this.problem, token.payload.values)), DOMAIN_CHANNEL);
    }
  }
}

export class ConstraintCheckerStation extends DESStation {
  constructor(id: string, private readonly problem: StructuredOptimizationProblem) {
    super(id);
  }

  override assertPreconditions(): void {
    validateProblem(this.problem);
  }

  override hasWork(): boolean {
    return this.inboxSize(DOMAIN_CHANNEL) > 0;
  }

  runTimeStep(): void {
    for (const token of this.drain<DomainCheckedToken>(DOMAIN_CHANNEL)) {
      this.emit(new ConstraintCheckedToken(
        token.candidate,
        token.domainViolations,
        checkConstraints(this.problem, token.candidate.values),
      ), CONSTRAINT_CHANNEL);
    }
  }
}

export class ObjectiveEvaluatorStation extends DESStation {
  constructor(
    id: string,
    private readonly problem: StructuredOptimizationProblem,
    private readonly penalty: number,
  ) {
    super(id);
    this.addValidator(intrinsicCheck<ObjectiveEvaluatorStation>({
      name: 'objective-evaluator.penalty-positive',
      group: 'feasibility-pipeline',
      predicate: s => s.penalty > 0,
      expected: 'penalty > 0',
      observedFn: s => String(s.penalty),
    }));
  }

  override assertPreconditions(): void {
    validateProblem(this.problem);
    Preconditions.positive('ObjectiveEvaluatorStation', 'penalty', this.penalty);
  }

  override hasWork(): boolean {
    return this.inboxSize(CONSTRAINT_CHANNEL) > 0;
  }

  runTimeStep(): void {
    for (const token of this.drain<ConstraintCheckedToken>(CONSTRAINT_CHANNEL)) {
      this.emit(new FeasibilityEvaluationToken(finalizeEvaluation(
        this.problem,
        token.candidate,
        token.domainViolations,
        token.constraintViolations,
        this.penalty,
      )), EVALUATION_CHANNEL);
    }
  }
}

export class ImprovementStation extends DESStation {
  private readonly enabled: boolean;
  private readonly maxIterations: number;
  private readonly continuousStep: number;
  private readonly integerStep: number;
  private readonly allowRepair: boolean;
  private readonly rng: () => number;
  private initialized = false;
  private done = false;
  private waiting = false;
  private repairTried = false;
  private proposalCount = 0;
  private bestEval: FeasibilityEvaluation | undefined;

  constructor(
    id: string,
    private readonly problem: StructuredOptimizationProblem,
    opts: FeasibilityImprovementOptions = {},
  ) {
    super(id);
    this.enabled = opts.enabled ?? true;
    this.maxIterations = opts.maxIterations ?? 200;
    this.continuousStep = opts.continuousStep ?? 1;
    this.integerStep = opts.integerStep ?? 1;
    this.allowRepair = opts.allowRepair ?? true;
    this.rng = mulberry32(opts.seed ?? 1);
    this.addValidator(intrinsicCheck<ImprovementStation>({
      name: 'improvement-station.best-evaluation-exists',
      group: 'feasibility-pipeline',
      predicate: s => s.bestEval !== undefined,
      expected: 'at least one evaluated candidate',
      observedFn: s => s.bestEval ? s.bestEval.candidateId : 'missing',
    }));
  }

  override assertPreconditions(): void {
    validateProblem(this.problem);
    Preconditions.integerInRange('ImprovementStation', 'maxIterations', this.maxIterations, 0, Number.MAX_SAFE_INTEGER);
    Preconditions.positive('ImprovementStation', 'continuousStep', this.continuousStep);
    Preconditions.positive('ImprovementStation', 'integerStep', this.integerStep);
  }

  override hasWork(): boolean {
    if (this.inboxSize(EVALUATION_CHANNEL) > 0) return true;
    if (!this.enabled || this.done || !this.initialized || this.waiting) return false;
    return this.proposalCount < this.maxIterations || (this.allowRepair && !this.repairTried);
  }

  runTimeStep(): void {
    for (const token of this.drain<FeasibilityEvaluationToken>(EVALUATION_CHANNEL)) {
      this.waiting = false;
      this.initialized = true;
      if (!this.bestEval || evaluationBetter(token.payload, this.bestEval, this.problem)) {
        this.bestEval = token.payload;
      }
    }
    if (!this.enabled || this.done || !this.bestEval || this.waiting) return;
    if (this.allowRepair && !this.repairTried) {
      this.repairTried = true;
      const repaired = repairValues(this.problem, this.bestEval.values);
      if (!sameValues(this.problem, repaired, this.bestEval.values)) {
        this.emitCandidate(repaired, 'repair');
        return;
      }
    }
    if (this.proposalCount >= this.maxIterations) {
      this.done = true;
      return;
    }
    this.emitCandidate(proposeNeighbor(
      this.problem,
      this.bestEval.values,
      this.rng,
      this.continuousStep,
      this.integerStep,
    ), 'neighbor');
  }

  private emitCandidate(values: Record<string, number>, origin: CandidatePayload['origin']): void {
    const candidate: CandidatePayload = {
      id: `${origin}-${this.proposalCount + 1}`,
      parentId: this.bestEval?.candidateId,
      iteration: this.proposalCount + 1,
      origin,
      values,
    };
    this.proposalCount++;
    this.waiting = true;
    this.emit(new CandidateToken(candidate), CANDIDATE_CHANNEL);
  }
}

export class FeasibilitySinkStation extends DESStation {
  readonly trace: FeasibilityEvaluation[] = [];
  readonly stops: StopSignalToken['payload'][] = [];

  constructor(id = 'feasibility-sink', private readonly problem: StructuredOptimizationProblem) {
    super(id);
    this.addValidator(intrinsicCheck<FeasibilitySinkStation>({
      name: 'feasibility-sink.trace-nonempty',
      group: 'feasibility-pipeline',
      predicate: s => s.trace.length > 0,
      expected: 'at least one evaluation',
      observedFn: s => String(s.trace.length),
    }));
  }

  override hasWork(): boolean {
    return false;
  }

  runTimeStep(): void {
    for (const token of this.drain<FeasibilityEvaluationToken>(EVALUATION_CHANNEL)) {
      this.trace.push(token.payload);
    }
    for (const token of this.drain<StopSignalToken>(STOP_CHANNEL)) {
      this.stops.push(token.payload);
    }
  }

  best(): FeasibilityEvaluation | undefined {
    let best: FeasibilityEvaluation | undefined;
    for (const row of this.trace) {
      if (!best || evaluationBetter(row, best, this.problem)) best = row;
    }
    return best;
  }
}

export function runFeasibilityPipeline(params: FeasibilityPipelineParams): FeasibilityPipelineResult {
  validateProblem(params.problem);
  const improvement = params.improvement ?? {};
  const penalty = improvement.penalty ?? 1000000;
  const source = new CandidateSourceStation('candidate-source', candidatePayloadFromInput(params.problem, params.candidate));
  const domain = new DomainCheckerStation('domain-checker', params.problem);
  const constraints = new ConstraintCheckerStation('constraint-checker', params.problem);
  const objective = new ObjectiveEvaluatorStation('objective-evaluator', params.problem, penalty);
  const improver = new ImprovementStation('improvement-station', params.problem, improvement);
  const sink = new FeasibilitySinkStation('feasibility-sink', params.problem);
  const budgetMs = params.timeLimitMs ?? 180000;
  const checker = new WallClockCheckerStation('wall-clock-checker', budgetMs, params.checkEveryTicks ?? 1);

  source.pipe(domain, CANDIDATE_CHANNEL, CANDIDATE_CHANNEL);
  improver.pipe(domain, CANDIDATE_CHANNEL, CANDIDATE_CHANNEL);
  domain.pipe(constraints, DOMAIN_CHANNEL, DOMAIN_CHANNEL);
  constraints.pipe(objective, CONSTRAINT_CHANNEL, CONSTRAINT_CHANNEL);
  objective.pipe(sink, EVALUATION_CHANNEL, EVALUATION_CHANNEL);
  objective.pipe(improver, EVALUATION_CHANNEL, EVALUATION_CHANNEL);
  checker.pipe(sink, STOP_CHANNEL, STOP_CHANNEL);

  const maxTicks = params.maxTicks ?? defaultMaxTicks(improvement);
  const stations = [source, domain, constraints, objective, improver, checker, sink];
  const runSummary = runIterativeDES(stations, {
    shuffle: false,
    maxTicks,
    stopWhen: () => checker.expired(),
  });
  const initial = sink.trace[0] ?? evaluateCandidate(params.problem, params.candidate, penalty);
  const best = sink.best() ?? initial;
  const improvements = sink.trace.filter(row => row.candidateId !== initial.candidateId && evaluationBetter(row, initial, params.problem));
  return {
    status: pipelineStatus(params.problem, runSummary, checker, initial, best),
    initial,
    best,
    trace: sink.trace.slice(),
    improvements,
    stopSignals: sink.stops.slice(),
    wallClock: {
      budgetMs,
      elapsedMs: checker.elapsedMs(),
      checks: checker.numChecks(),
      expired: checker.expired(),
    },
    runSummary,
    network: describeFeasibilityPipelineNetwork(),
    validation: runSummary.validation ?? [],
  };
}

export function evaluateCandidate(
  problem: StructuredOptimizationProblem,
  candidate: CandidateSolutionInput,
  penalty = 1000000,
): FeasibilityEvaluation {
  validateProblem(problem);
  const payload = candidatePayloadFromInput(problem, candidate);
  return finalizeEvaluation(problem, payload, checkDomain(problem, payload.values), checkConstraints(problem, payload.values), penalty);
}

function candidatePayloadFromInput(problem: StructuredOptimizationProblem, input: CandidateSolutionInput): CandidatePayload {
  const values: Record<string, number> = {};
  if (input.values) {
    for (const v of problem.variables) values[v.name] = input.values[v.name];
  } else if (input.vector) {
    for (let i = 0; i < problem.variables.length; i++) values[problem.variables[i].name] = input.vector[i];
  } else {
    for (const v of problem.variables) values[v.name] = NaN;
  }
  return {id: input.id ?? 'user-candidate', iteration: 0, origin: 'user', values};
}

function checkDomain(problem: StructuredOptimizationProblem, values: Record<string, number>): FeasibilityViolation[] {
  const tol = problem.tolerance ?? 1e-8;
  const out: FeasibilityViolation[] = [];
  for (const v of problem.variables) {
    const x = values[v.name];
    const lb = lowerBound(v);
    const ub = upperBound(v);
    if (!Number.isFinite(x)) {
      out.push({kind: 'domain', name: `${v.name}.finite`, variable: v.name, violation: Number.POSITIVE_INFINITY, message: `${v.name} is missing or not finite`});
      continue;
    }
    if (x < lb - tol) out.push({kind: 'domain', name: `${v.name}.lb`, variable: v.name, violation: lb - x, message: `${v.name}=${x} below lower bound ${lb}`});
    if (x > ub + tol) out.push({kind: 'domain', name: `${v.name}.ub`, variable: v.name, violation: x - ub, message: `${v.name}=${x} above upper bound ${ub}`});
    if ((v.type === 'integer' || v.type === 'binary') && Math.abs(x - Math.round(x)) > tol) {
      out.push({kind: 'domain', name: `${v.name}.integer`, variable: v.name, violation: Math.abs(x - Math.round(x)), message: `${v.name}=${x} is not integral`});
    }
    if (v.type === 'binary' && Math.min(Math.abs(x), Math.abs(x - 1)) > tol) {
      out.push({kind: 'domain', name: `${v.name}.binary`, variable: v.name, violation: Math.min(Math.abs(x), Math.abs(x - 1)), message: `${v.name}=${x} is not binary`});
    }
  }
  return out;
}

function checkConstraints(problem: StructuredOptimizationProblem, values: Record<string, number>): FeasibilityViolation[] {
  const constraints = problem.constraints ?? [];
  const out: FeasibilityViolation[] = [];
  for (let i = 0; i < constraints.length; i++) {
    const c = constraints[i];
    const tol = c.tolerance ?? problem.tolerance ?? 1e-8;
    const activity = evaluateLinear(c.coefficients, values, 0);
    let violation = 0;
    if (c.sense === '<=') violation = Math.max(0, activity - c.rhs - tol);
    else if (c.sense === '>=') violation = Math.max(0, c.rhs - activity - tol);
    else violation = Math.max(0, Math.abs(activity - c.rhs) - tol);
    if (violation > 0) {
      const name = c.name ?? `constraint-${i}`;
      out.push({kind: 'constraint', name, constraint: name, activity, rhs: c.rhs, violation, message: `${name}: activity ${activity} ${c.sense} ${c.rhs} violated by ${violation}`});
    }
  }
  return out;
}

function finalizeEvaluation(
  problem: StructuredOptimizationProblem,
  candidate: CandidatePayload,
  domainViolations: FeasibilityViolation[],
  constraintViolations: FeasibilityViolation[],
  penalty: number,
): FeasibilityEvaluation {
  const objectiveValue = evaluateLinear(problem.objective.coefficients, candidate.values, problem.objective.constant ?? 0);
  const comparableObjective = problem.sense === 'min' ? objectiveValue : -objectiveValue;
  const violations = domainViolations.concat(constraintViolations);
  const totalViolation = violations.reduce((s, v) => s + safeViolation(v.violation), 0);
  const maxViolation = violations.reduce((m, v) => Math.max(m, safeViolation(v.violation)), 0);
  return {
    candidateId: candidate.id,
    parentId: candidate.parentId,
    iteration: candidate.iteration,
    origin: candidate.origin,
    values: {...candidate.values},
    objectiveValue,
    comparableObjective,
    totalViolation,
    maxViolation,
    feasible: violations.length === 0,
    domainViolations,
    constraintViolations,
    violations,
    merit: totalViolation * penalty + comparableObjective,
  };
}

function evaluateLinear(coefficients: Record<string, number>, values: Record<string, number>, constant: number): number {
  let out = constant;
  for (const [name, coeff] of Object.entries(coefficients)) out += coeff * (values[name] ?? NaN);
  return out;
}

function safeViolation(x: number): number {
  return Number.isFinite(x) ? x : 1e12;
}

function evaluationBetter(a: FeasibilityEvaluation, b: FeasibilityEvaluation, problem: StructuredOptimizationProblem): boolean {
  const tol = problem.tolerance ?? 1e-8;
  if (a.feasible && !b.feasible) return true;
  if (!a.feasible && b.feasible) return false;
  if (a.feasible && b.feasible) return a.comparableObjective < b.comparableObjective - tol;
  if (a.totalViolation < b.totalViolation - tol) return true;
  if (Math.abs(a.totalViolation - b.totalViolation) <= tol && a.merit < b.merit - tol) return true;
  return false;
}

function repairValues(problem: StructuredOptimizationProblem, input: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const v of problem.variables) {
    let x = input[v.name];
    const lb = lowerBound(v);
    const ub = upperBound(v);
    if (!Number.isFinite(x)) {
      if (Number.isFinite(lb) && Number.isFinite(ub)) x = (lb + ub) / 2;
      else if (Number.isFinite(lb)) x = lb;
      else if (Number.isFinite(ub)) x = ub;
      else x = 0;
    }
    x = clamp(x, lb, ub);
    if (v.type === 'binary') x = x >= 0.5 ? 1 : 0;
    else if (v.type === 'integer') x = Math.round(x);
    out[v.name] = clamp(x, lb, ub);
  }
  return out;
}

function proposeNeighbor(
  problem: StructuredOptimizationProblem,
  input: Record<string, number>,
  rng: () => number,
  continuousStep: number,
  integerStep: number,
): Record<string, number> {
  const base = repairValues(problem, input);
  const variables = problem.variables;
  const binary = variables.filter(v => v.type === 'binary');
  if (binary.length >= 2 && rng() < 0.5) {
    const ones = binary.filter(v => base[v.name] >= 0.5);
    const zeros = binary.filter(v => base[v.name] < 0.5);
    if (ones.length > 0 && zeros.length > 0) {
      const out = {...base};
      const drop = ones[Math.floor(rng() * ones.length)];
      const add = zeros[Math.floor(rng() * zeros.length)];
      out[drop.name] = 0;
      out[add.name] = 1;
      return out;
    }
  }
  for (let attempt = 0; attempt < variables.length; attempt++) {
    const v = variables[Math.floor(rng() * variables.length)];
    const out = {...base};
    const lb = lowerBound(v);
    const ub = upperBound(v);
    if (v.type === 'binary') {
      out[v.name] = base[v.name] >= 0.5 ? 0 : 1;
    } else {
      const sign = rng() < 0.5 ? -1 : 1;
      const step = v.step ?? (v.type === 'integer' ? integerStep : continuousStep);
      out[v.name] = base[v.name] + sign * step;
      if (v.type === 'integer') out[v.name] = Math.round(out[v.name]);
    }
    out[v.name] = clamp(out[v.name], lb, ub);
    if (!sameValues(problem, out, base)) return out;
  }
  return base;
}

function sameValues(problem: StructuredOptimizationProblem, a: Record<string, number>, b: Record<string, number>): boolean {
  const tol = problem.tolerance ?? 1e-8;
  return problem.variables.every(v => Math.abs((a[v.name] ?? NaN) - (b[v.name] ?? NaN)) <= tol);
}

function lowerBound(v: OptimizationVariable): number {
  if (v.type === 'binary') return Math.max(v.lb ?? 0, 0);
  return v.lb ?? Number.NEGATIVE_INFINITY;
}

function upperBound(v: OptimizationVariable): number {
  if (v.type === 'binary') return Math.min(v.ub ?? 1, 1);
  return v.ub ?? Number.POSITIVE_INFINITY;
}

function clamp(x: number, lb: number, ub: number): number {
  return Math.min(ub, Math.max(lb, x));
}

function validateProblem(problem: StructuredOptimizationProblem): void {
  Preconditions.check('FeasibilityPipeline', 'problem', 'be an object', problem !== null && typeof problem === 'object' && !Array.isArray(problem), problem);
  Preconditions.nonEmpty('FeasibilityPipeline', 'variables', problem.variables);
  Preconditions.check('FeasibilityPipeline', 'sense', 'be "min" or "max"', problem.sense === 'min' || problem.sense === 'max', problem.sense);
  const names = new Set<string>();
  for (const v of problem.variables) {
    Preconditions.check('FeasibilityPipeline', 'variable.name', 'be non-empty', typeof v.name === 'string' && v.name.length > 0, v.name);
    Preconditions.check('FeasibilityPipeline', `variable.${v.name}.unique`, 'be unique', !names.has(v.name), v.name);
    names.add(v.name);
    if (v.lb !== undefined) Preconditions.finite('FeasibilityPipeline', `${v.name}.lb`, v.lb);
    if (v.ub !== undefined) Preconditions.finite('FeasibilityPipeline', `${v.name}.ub`, v.ub);
    Preconditions.check('FeasibilityPipeline', `${v.name}.bounds`, 'satisfy lb <= ub', lowerBound(v) <= upperBound(v), [lowerBound(v), upperBound(v)]);
    if (v.step !== undefined) Preconditions.positive('FeasibilityPipeline', `${v.name}.step`, v.step);
    if (v.type !== undefined) {
      Preconditions.check('FeasibilityPipeline', `${v.name}.type`, 'be continuous, integer, or binary', v.type === 'continuous' || v.type === 'integer' || v.type === 'binary', v.type);
    }
  }
  Preconditions.check('FeasibilityPipeline', 'objective', 'be an object', problem.objective !== null && typeof problem.objective === 'object' && !Array.isArray(problem.objective), problem.objective);
  validateCoefficients('objective.coefficients', problem.objective.coefficients, names);
  if (problem.objective.constant !== undefined) Preconditions.finite('FeasibilityPipeline', 'objective.constant', problem.objective.constant);
  const constraints = problem.constraints ?? [];
  for (let i = 0; i < constraints.length; i++) {
    const c = constraints[i];
    Preconditions.check('FeasibilityPipeline', `constraints[${i}].sense`, 'be <=, >=, or =', c.sense === '<=' || c.sense === '>=' || c.sense === '=', c.sense);
    Preconditions.finite('FeasibilityPipeline', `constraints[${i}].rhs`, c.rhs);
    if (c.tolerance !== undefined) Preconditions.nonNegative('FeasibilityPipeline', `constraints[${i}].tolerance`, c.tolerance);
    validateCoefficients(`constraints[${i}].coefficients`, c.coefficients, names);
  }
  if (problem.tolerance !== undefined) Preconditions.nonNegative('FeasibilityPipeline', 'tolerance', problem.tolerance);
}

function validateCoefficients(param: string, coeffs: Record<string, number>, variableNames: Set<string>): void {
  Preconditions.check('FeasibilityPipeline', param, 'be an object', coeffs !== null && typeof coeffs === 'object' && !Array.isArray(coeffs), coeffs);
  for (const [name, coeff] of Object.entries(coeffs)) {
    Preconditions.check('FeasibilityPipeline', `${param}.${name}`, 'reference a declared variable', variableNames.has(name), name);
    Preconditions.finite('FeasibilityPipeline', `${param}.${name}`, coeff);
  }
}

function defaultMaxTicks(opts: FeasibilityImprovementOptions): number {
  return (opts.enabled ?? true) ? (opts.maxIterations ?? 200) * 4 + 16 : 16;
}

function pipelineStatus(
  problem: StructuredOptimizationProblem,
  summary: IterativeRunSummary,
  checker: WallClockCheckerStation,
  initial: FeasibilityEvaluation,
  best: FeasibilityEvaluation,
): FeasibilityPipelineResult['status'] {
  if (checker.expired()) return 'time-limit';
  if (summary.reason === 'maxticks') return 'tick-limit';
  const improved = best.candidateId !== initial.candidateId && evaluationBetter(best, initial, problem);
  if (best.feasible && improved) return 'improved';
  if (best.feasible) return 'feasible';
  return improved ? 'infeasible-improved' : 'infeasible';
}

function describeFeasibilityPipelineNetwork(): FeasibilityPipelineNetwork {
  return {
    stationaryEntities: [
      {id: 'candidate-source', kind: 'candidate-source', role: 'source'},
      {id: 'domain-checker', kind: 'domain-checker', role: 'checker'},
      {id: 'constraint-checker', kind: 'constraint-checker', role: 'checker'},
      {id: 'objective-evaluator', kind: 'objective-evaluator', role: 'evaluator'},
      {id: 'improvement-station', kind: 'local-improver', role: 'improver'},
      {id: 'wall-clock-checker', kind: 'wall-clock-checker', role: 'checker'},
      {id: 'feasibility-sink', kind: 'feasibility-sink', role: 'sink'},
    ],
    movingEntities: [
      {id: 'CandidateToken', kind: 'candidate-solution', tokenType: 'CandidateToken'},
      {id: 'DomainCheckedToken', kind: 'domain-checked-candidate', tokenType: 'DomainCheckedToken'},
      {id: 'ConstraintCheckedToken', kind: 'constraint-checked-candidate', tokenType: 'ConstraintCheckedToken'},
      {id: 'FeasibilityEvaluationToken', kind: 'evaluation', tokenType: 'FeasibilityEvaluationToken'},
      {id: 'StopSignalToken', kind: 'stop-signal', tokenType: 'StopSignalToken'},
    ],
    edges: [
      {from: 'candidate-source', to: 'domain-checker', movingEntity: 'CandidateToken', channel: CANDIDATE_CHANNEL},
      {from: 'improvement-station', to: 'domain-checker', movingEntity: 'CandidateToken', channel: CANDIDATE_CHANNEL},
      {from: 'domain-checker', to: 'constraint-checker', movingEntity: 'DomainCheckedToken', channel: DOMAIN_CHANNEL},
      {from: 'constraint-checker', to: 'objective-evaluator', movingEntity: 'ConstraintCheckedToken', channel: CONSTRAINT_CHANNEL},
      {from: 'objective-evaluator', to: 'feasibility-sink', movingEntity: 'FeasibilityEvaluationToken', channel: EVALUATION_CHANNEL},
      {from: 'objective-evaluator', to: 'improvement-station', movingEntity: 'FeasibilityEvaluationToken', channel: EVALUATION_CHANNEL},
      {from: 'wall-clock-checker', to: 'feasibility-sink', movingEntity: 'StopSignalToken', channel: STOP_CHANNEL},
    ],
  };
}
