// RUST MIGRATION: target module src/des/general/domain_application_models.rs.
// RUST MIGRATION: DomainModelResult, DomainTrace, DomainEvaluation, scenario/plan/params/result types become serde structs; exported result aliases should become concrete type aliases.
// RUST MIGRATION: Generic DomainScenarioSource/CandidateGenerator/PlanEvaluator/ResultSink stations become generic structs implementing Station traits with explicit trait bounds.
// RUST MIGRATION: runDomainPipeline is graph-visible shared orchestration; port it as a reusable PureTransform/trait-backed pipeline, with each run* application as a thin PureTransform wrapper.
// RUST MIGRATION: Record-like plan fields map to HashMap<String, f64> where names are dynamic, and validation helpers return Result.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/domain-application-models.rs  (module des::general::domain_application_models)
// 1:1 file move. Applied OR/control/DS models on ONE generic DES topology (Scenario->Generate->Evaluate->Sink).
//
// Declarations → Rust:
//   interface DomainModelResult<P>/DomainTrace/DomainEvaluation<P>/DomainScenario<S> -> generic structs
//   class DomainScenarioToken<S>/DomainPlanToken<S,P>/DomainEvaluationToken<P> (impl Token) -> generic structs `impl Token`
//   class DomainScenarioSource/CandidateGenerator/PlanEvaluator/ResultSink Station<..> (extend DESStation)
//                                    -> generic structs `impl` the station trait (base -> trait)
//   per-domain interface *Params / *Scenario / *Plan + `type *Result = DomainModelResult<*Plan>` -> structs + type aliases
//   fn runAdaptiveFuzzyControl/runLogisticsRoutingHeuristics/runBottleneckProductionControl/
//      runSupplyChainRiskPooling/runWorkforceServiceOperations/runPortfolioDrawdownControl/
//      runDynamicPricingRevenue/runBuyerAwareDynamicPricing/... -> fns
//
// Conversion notes (file-specific):
//   - The generic Scenario/Plan station pipeline carries over as Rust generics `<S, P>`.
//   - `interface FooScenario extends Required<FooParams>` -> a fully-populated struct (resolve Option defaults up front).
//   - Generator/evaluator logic per domain -> trait methods or `Fn` params; `Preconditions` throw -> `Result`.
//   - `DomainModelResult<P = unknown>` default -> generic with a concrete payload type per domain (avoid `unknown`).
// =============================================================================

// =============================================================================
// domain-application-models.ts
//
// Applied operations/control/data-science models, each expressed as the
// same explicit DES topology:
//
//   ScenarioSource -> CandidateGenerator -> PlanEvaluator -> ResultSink
//
// Scenarios, candidate plans, and evaluated plans are moving tokens. The
// generator/evaluator/sink are stationary entities. This gives third-party
// users a consistent validation surface across very different domains while
// leaving each domain free to encode its own plan representation and objective.
// =============================================================================

import {
  ChannelName,
  DESStation,
  StationGraphSummary,
  Token,
  channelEdge,
  runIterativeDES,
  stationGraph,
} from './des-base';
import {Preconditions} from './des-base/preconditions';

export interface DomainModelResult<P = unknown> {
  modelId: string;
  category: string;
  best: DomainEvaluation<P>;
  candidates: DomainEvaluation<P>[];
  topology: StationGraphSummary;
}

export interface DomainTrace {
  t: number[];
  series: Record<string, number[]>;
  captions?: string[];
}

export interface DomainEvaluation<P = unknown> {
  candidateId: string;
  plan: P;
  objective: number;
  feasible: boolean;
  metrics: Record<string, number | string | boolean>;
  trace?: DomainTrace;
}

interface DomainScenario<S> {
  modelId: string;
  category: string;
  scenario: S;
}

class DomainScenarioToken<S> implements Token {
  constructor(readonly payload: DomainScenario<S>) {}
}

class DomainPlanToken<S, P> implements Token {
  constructor(
    readonly modelId: string,
    readonly category: string,
    readonly scenario: S,
    readonly candidateId: string,
    readonly plan: P,
  ) {}
}

class DomainEvaluationToken<P> implements Token {
  constructor(readonly evaluation: DomainEvaluation<P>) {}
}

class DomainScenarioSourceStation<S> extends DESStation {
  static readonly CH_SCENARIO: ChannelName = 'scenario';
  private emitted = false;

  constructor(id: string, private readonly payload: DomainScenario<S>) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(new DomainScenarioToken(this.payload), DomainScenarioSourceStation.CH_SCENARIO);
    this.emitted = true;
  }
}

class DomainCandidateGeneratorStation<S, P> extends DESStation {
  static readonly CH_SCENARIO: ChannelName = DomainScenarioSourceStation.CH_SCENARIO;
  static readonly CH_PLAN: ChannelName = 'candidate-plan';

  constructor(
    id: string,
    private readonly generate: (scenario: S) => Array<{candidateId: string; plan: P}>,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(DomainCandidateGeneratorStation.CH_SCENARIO) > 0; }

  runTimeStep(): void {
    const scenarios = this.drain<DomainScenarioToken<S>>(DomainCandidateGeneratorStation.CH_SCENARIO);
    for (const token of scenarios) {
      for (const candidate of this.generate(token.payload.scenario)) {
        this.emit(
          new DomainPlanToken(
            token.payload.modelId,
            token.payload.category,
            token.payload.scenario,
            candidate.candidateId,
            candidate.plan,
          ),
          DomainCandidateGeneratorStation.CH_PLAN,
        );
      }
    }
  }
}

class DomainPlanEvaluatorStation<S, P> extends DESStation {
  static readonly CH_PLAN: ChannelName = DomainCandidateGeneratorStation.CH_PLAN;
  static readonly CH_EVALUATION: ChannelName = 'evaluation';

  constructor(
    id: string,
    private readonly evaluate: (scenario: S, plan: P, candidateId: string) => DomainEvaluation<P>,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(DomainPlanEvaluatorStation.CH_PLAN) > 0; }

  runTimeStep(): void {
    const plans = this.drain<DomainPlanToken<S, P>>(DomainPlanEvaluatorStation.CH_PLAN);
    for (const token of plans) {
      this.emit(
        new DomainEvaluationToken(this.evaluate(token.scenario, token.plan, token.candidateId)),
        DomainPlanEvaluatorStation.CH_EVALUATION,
      );
    }
  }
}

class DomainResultSinkStation<P> extends DESStation {
  static readonly CH_EVALUATION: ChannelName = DomainPlanEvaluatorStation.CH_EVALUATION;
  readonly evaluations: DomainEvaluation<P>[] = [];

  constructor(id: string) { super(id); }

  override hasWork(): boolean { return this.inboxSize(DomainResultSinkStation.CH_EVALUATION) > 0; }

  runTimeStep(): void {
    const incoming = this.drain<DomainEvaluationToken<P>>(DomainResultSinkStation.CH_EVALUATION);
    for (const token of incoming) this.evaluations.push(token.evaluation);
  }

  best(): DomainEvaluation<P> {
    const feasible = this.evaluations.filter(row => row.feasible);
    if (feasible.length === 0) throw new Error(`${this.id}: no feasible domain plans were evaluated`);
    return feasible.reduce((best, row) => row.objective > best.objective ? row : best, feasible[0]);
  }
}

function runDomainPipeline<S, P>(opts: {
  modelId: string;
  category: string;
  scenario: S;
  generate: (scenario: S) => Array<{candidateId: string; plan: P}>;
  evaluate: (scenario: S, plan: P, candidateId: string) => DomainEvaluation<P>;
}): DomainModelResult<P> {
  const source = new DomainScenarioSourceStation(`${opts.modelId}-scenario-source`, {
    modelId: opts.modelId,
    category: opts.category,
    scenario: opts.scenario,
  });
  const generator = new DomainCandidateGeneratorStation(`${opts.modelId}-candidate-generator`, opts.generate);
  const evaluator = new DomainPlanEvaluatorStation(`${opts.modelId}-plan-evaluator`, opts.evaluate);
  const sink = new DomainResultSinkStation<P>(`${opts.modelId}-result-sink`);
  source.pipe(generator, DomainScenarioSourceStation.CH_SCENARIO, DomainCandidateGeneratorStation.CH_SCENARIO);
  generator.pipe(evaluator, DomainCandidateGeneratorStation.CH_PLAN, DomainPlanEvaluatorStation.CH_PLAN);
  evaluator.pipe(sink, DomainPlanEvaluatorStation.CH_EVALUATION, DomainResultSinkStation.CH_EVALUATION);
  runIterativeDES([source, generator, evaluator, sink], {shuffle: false, maxTicks: 8, runValidators: false});
  const candidates = sink.evaluations.slice().sort((a, b) => b.objective - a.objective);
  return {
    modelId: opts.modelId,
    category: opts.category,
    best: sink.best(),
    candidates,
    topology: stationGraph(
      [source, generator, evaluator, sink],
      ['DomainScenarioToken', 'DomainPlanToken', 'DomainEvaluationToken'],
      [
        channelEdge(source, DomainScenarioSourceStation.CH_SCENARIO, generator, DomainCandidateGeneratorStation.CH_SCENARIO),
        channelEdge(generator, DomainCandidateGeneratorStation.CH_PLAN, evaluator, DomainPlanEvaluatorStation.CH_PLAN),
        channelEdge(evaluator, DomainPlanEvaluatorStation.CH_EVALUATION, sink, DomainResultSinkStation.CH_EVALUATION),
      ],
    ),
  };
}

// -----------------------------------------------------------------------------
// 1. Control systems: adaptive/fuzzy/intelligent control
// -----------------------------------------------------------------------------

export interface AdaptiveFuzzyControlParams {
  steps?: number;
  dt?: number;
  setpoint?: number;
  initialTemp?: number;
  outsideTemp?: number;
  disturbance?: number;
}

interface FuzzyControlScenario extends Required<AdaptiveFuzzyControlParams> {
  plantLoss: number;
  plantGain: number;
  controlMax: number;
}

interface FuzzyControlPlan {
  errorGain: number;
  derivativeGain: number;
  outputGain: number;
  adaptiveBoost: number;
}

export type AdaptiveFuzzyControlResult = DomainModelResult<FuzzyControlPlan>;

export function runAdaptiveFuzzyControl(params: AdaptiveFuzzyControlParams = {}): AdaptiveFuzzyControlResult {
  const scenario: FuzzyControlScenario = {
    steps: params.steps ?? 140,
    dt: params.dt ?? 0.1,
    setpoint: params.setpoint ?? 22,
    initialTemp: params.initialTemp ?? 16,
    outsideTemp: params.outsideTemp ?? 8,
    disturbance: params.disturbance ?? 0.15,
    plantLoss: 0.06,
    plantGain: 0.42,
    controlMax: 6,
  };
  checkPositiveInt('runAdaptiveFuzzyControl', 'steps', scenario.steps);
  Preconditions.positive('runAdaptiveFuzzyControl', 'dt', scenario.dt);
  return runDomainPipeline({
    modelId: 'adaptive-fuzzy-control',
    category: 'Control systems (adaptive; fuzzy; intelligent)',
    scenario,
    generate: fuzzyControlCandidates,
    evaluate: evaluateFuzzyControl,
  });
}

function fuzzyControlCandidates(_scenario: FuzzyControlScenario): Array<{candidateId: string; plan: FuzzyControlPlan}> {
  return [
    {candidateId: 'calm-fuzzy', plan: {errorGain: 0.35, derivativeGain: 0.10, outputGain: 2.8, adaptiveBoost: 0}},
    {candidateId: 'balanced-adaptive-fuzzy', plan: {errorGain: 0.55, derivativeGain: 0.20, outputGain: 4.2, adaptiveBoost: 0.8}},
    {candidateId: 'aggressive-fuzzy', plan: {errorGain: 0.85, derivativeGain: 0.25, outputGain: 5.8, adaptiveBoost: 0.4}},
    {candidateId: 'energy-saver-fuzzy', plan: {errorGain: 0.45, derivativeGain: 0.35, outputGain: 3.3, adaptiveBoost: 0.2}},
  ];
}

function evaluateFuzzyControl(
  scenario: FuzzyControlScenario,
  plan: FuzzyControlPlan,
  candidateId: string,
): DomainEvaluation<FuzzyControlPlan> {
  let temp = scenario.initialTemp;
  let prevError = scenario.setpoint - temp;
  let energy = 0;
  let sqErr = 0;
  let settlingTick = scenario.steps;
  for (let k = 0; k < scenario.steps; k++) {
    const error = scenario.setpoint - temp;
    const dError = error - prevError;
    const boost = Math.abs(error) > 1.5 ? plan.adaptiveBoost * Math.min(1.5, Math.abs(error) / 4) : 0;
    const fuzzySignal = Math.tanh(plan.errorGain * error + plan.derivativeGain * dError);
    const control = clamp(plan.outputGain * (fuzzySignal + boost), 0, scenario.controlMax);
    const outdoorLeak = scenario.plantLoss * (scenario.outsideTemp - temp);
    const seasonalDisturbance = scenario.disturbance * Math.sin(0.15 * k);
    temp += scenario.dt * (outdoorLeak + scenario.plantGain * control + seasonalDisturbance);
    energy += control * scenario.dt;
    sqErr += error * error;
    if (settlingTick === scenario.steps && Math.abs(error) < 0.25) settlingTick = k;
    prevError = error;
  }
  const rmsError = Math.sqrt(sqErr / scenario.steps);
  const objective = -rmsError - 0.025 * energy - 0.001 * settlingTick;
  return {
    candidateId,
    plan,
    objective,
    feasible: true,
    metrics: {rmsError, energy, settlingTick, finalTemp: temp},
  };
}

// -----------------------------------------------------------------------------
// 2. Logistics/transportation: optimal routing, heuristics, scheduling
// -----------------------------------------------------------------------------

export interface LogisticsRoutingParams {
  vehicleCapacity?: number;
}

interface Customer {
  id: number;
  x: number;
  y: number;
  demand: number;
}

interface RoutingScenario {
  depot: {x: number; y: number};
  customers: Customer[];
  vehicleCapacity: number;
}

interface RoutingPlan {
  heuristic: 'nearest-neighbor' | 'sweep' | 'savings' | 'balanced-savings';
  routes: number[][];
}

export type LogisticsRoutingResult = DomainModelResult<RoutingPlan>;

export function runLogisticsRoutingHeuristics(params: LogisticsRoutingParams = {}): LogisticsRoutingResult {
  const scenario: RoutingScenario = {
    depot: {x: 0, y: 0},
    vehicleCapacity: params.vehicleCapacity ?? 7,
    customers: [
      {id: 1, x: 2, y: 1, demand: 2},
      {id: 2, x: 3, y: 4, demand: 2},
      {id: 3, x: -1, y: 3, demand: 1},
      {id: 4, x: -3, y: 2, demand: 3},
      {id: 5, x: -2, y: -2, demand: 2},
      {id: 6, x: 3, y: -2, demand: 2},
      {id: 7, x: 5, y: 1, demand: 1},
    ],
  };
  Preconditions.positive('runLogisticsRoutingHeuristics', 'vehicleCapacity', scenario.vehicleCapacity);
  return runDomainPipeline({
    modelId: 'logistics-routing-heuristics',
    category: 'Logistics/transportation (optimal routing, heuristics, scheduling)',
    scenario,
    generate: routingCandidates,
    evaluate: evaluateRoutingPlan,
  });
}

function routingCandidates(scenario: RoutingScenario): Array<{candidateId: string; plan: RoutingPlan}> {
  const nearest = buildNearestNeighborRoutes(scenario);
  const sweep = splitSequenceByCapacity(
    scenario.customers.slice().sort((a, b) => Math.atan2(a.y, a.x) - Math.atan2(b.y, b.x)).map(c => c.id),
    scenario,
  );
  const savings = buildSavingsRoutes(scenario, false);
  const balancedSavings = buildSavingsRoutes(scenario, true);
  return [
    {candidateId: 'nearest-neighbor', plan: {heuristic: 'nearest-neighbor', routes: nearest}},
    {candidateId: 'polar-sweep', plan: {heuristic: 'sweep', routes: sweep}},
    {candidateId: 'clarke-wright-savings', plan: {heuristic: 'savings', routes: savings}},
    {candidateId: 'balanced-savings', plan: {heuristic: 'balanced-savings', routes: balancedSavings}},
  ];
}

function buildNearestNeighborRoutes(scenario: RoutingScenario): number[][] {
  const remaining = new Set(scenario.customers.map(c => c.id));
  const routes: number[][] = [];
  while (remaining.size > 0) {
    const route: number[] = [];
    let load = 0;
    let cur = scenario.depot;
    while (true) {
      let best: Customer | null = null;
      let bestD = Infinity;
      for (const id of remaining) {
        const c = customerById(scenario, id);
        if (load + c.demand > scenario.vehicleCapacity) continue;
        const d = dist(cur, c);
        if (d < bestD) { bestD = d; best = c; }
      }
      if (!best) break;
      route.push(best.id);
      load += best.demand;
      remaining.delete(best.id);
      cur = best;
    }
    routes.push(route);
  }
  return routes;
}

function buildSavingsRoutes(scenario: RoutingScenario, balance: boolean): number[][] {
  let routes = scenario.customers.map(c => [c.id]);
  const loadOf = (route: readonly number[]) => route.reduce((sum, id) => sum + customerById(scenario, id).demand, 0);
  const savings = [];
  for (const a of scenario.customers) {
    for (const b of scenario.customers) {
      if (a.id >= b.id) continue;
      const baseSaving = dist(scenario.depot, a) + dist(scenario.depot, b) - dist(a, b);
      const balancePenalty = balance ? 0.04 * Math.abs(a.demand - b.demand) : 0;
      savings.push({a: a.id, b: b.id, saving: baseSaving - balancePenalty});
    }
  }
  savings.sort((a, b) => b.saving - a.saving);
  for (const s of savings) {
    const ia = routes.findIndex(r => r[0] === s.a || r[r.length - 1] === s.a);
    const ib = routes.findIndex(r => r[0] === s.b || r[r.length - 1] === s.b);
    if (ia < 0 || ib < 0 || ia === ib) continue;
    const ra = routes[ia];
    const rb = routes[ib];
    if (loadOf(ra) + loadOf(rb) > scenario.vehicleCapacity) continue;
    const merged = mergeRouteEnds(ra, rb, s.a, s.b);
    routes = routes.filter((_r, i) => i !== ia && i !== ib);
    routes.push(merged);
  }
  return routes;
}

function mergeRouteEnds(a: number[], b: number[], aId: number, bId: number): number[] {
  const aa = a[0] === aId ? a.slice().reverse() : a.slice();
  const bb = b[b.length - 1] === bId ? b.slice().reverse() : b.slice();
  return [...aa, ...bb];
}

function splitSequenceByCapacity(sequence: number[], scenario: RoutingScenario): number[][] {
  const routes: number[][] = [];
  let cur: number[] = [];
  let load = 0;
  for (const id of sequence) {
    const d = customerById(scenario, id).demand;
    if (cur.length > 0 && load + d > scenario.vehicleCapacity) {
      routes.push(cur);
      cur = [];
      load = 0;
    }
    cur.push(id);
    load += d;
  }
  if (cur.length > 0) routes.push(cur);
  return routes;
}

function evaluateRoutingPlan(scenario: RoutingScenario, plan: RoutingPlan, candidateId: string): DomainEvaluation<RoutingPlan> {
  const routeDistance = plan.routes.reduce((sum, route) => sum + routeLength(scenario, route), 0);
  const capacityViolation = plan.routes.reduce((sum, route) => sum + Math.max(0, routeLoad(scenario, route) - scenario.vehicleCapacity), 0);
  const objective = -routeDistance - 1000 * capacityViolation - 0.2 * plan.routes.length;
  return {
    candidateId,
    plan,
    objective,
    feasible: capacityViolation === 0,
    metrics: {routeDistance, vehicles: plan.routes.length, capacityViolation},
  };
}

// -----------------------------------------------------------------------------
// 3. Manufacturing: production planning/control, novel algorithms
// -----------------------------------------------------------------------------

export interface ManufacturingParams {
  horizon?: number;
  dailyDemand?: number;
}

interface ManufacturingScenario {
  horizon: number;
  dailyDemand: number;
  stage1Rate: number;
  stage2Rate: number;
}

interface ManufacturingPlan {
  releaseLot: number;
  wipCap: number;
  expediteThreshold: number;
}

export type ManufacturingResult = DomainModelResult<ManufacturingPlan>;

export function runBottleneckProductionControl(params: ManufacturingParams = {}): ManufacturingResult {
  const scenario: ManufacturingScenario = {
    horizon: params.horizon ?? 18,
    dailyDemand: params.dailyDemand ?? 8,
    stage1Rate: 12,
    stage2Rate: 9,
  };
  checkPositiveInt('runBottleneckProductionControl', 'horizon', scenario.horizon);
  Preconditions.positive('runBottleneckProductionControl', 'dailyDemand', scenario.dailyDemand);
  return runDomainPipeline({
    modelId: 'bottleneck-production-control',
    category: 'Manufacturing (production planning and control, novel algorithms)',
    scenario,
    generate: manufacturingCandidates,
    evaluate: evaluateManufacturingPlan,
  });
}

function manufacturingCandidates(_scenario: ManufacturingScenario): Array<{candidateId: string; plan: ManufacturingPlan}> {
  return [
    {candidateId: 'push-large-lots', plan: {releaseLot: 16, wipCap: 50, expediteThreshold: 30}},
    {candidateId: 'lean-kanban', plan: {releaseLot: 8, wipCap: 20, expediteThreshold: 14}},
    {candidateId: 'bottleneck-buffer-rope', plan: {releaseLot: 10, wipCap: 28, expediteThreshold: 18}},
    {candidateId: 'adaptive-expedite-control', plan: {releaseLot: 12, wipCap: 32, expediteThreshold: 10}},
  ];
}

function evaluateManufacturingPlan(
  scenario: ManufacturingScenario,
  plan: ManufacturingPlan,
  candidateId: string,
): DomainEvaluation<ManufacturingPlan> {
  let raw = 0;
  let buffer = 0;
  let finished = 0;
  let backlog = 0;
  let wipArea = 0;
  let shipped = 0;
  let expedites = 0;
  for (let t = 0; t < scenario.horizon; t++) {
    const wip = raw + buffer;
    const release = wip < plan.wipCap ? Math.min(plan.releaseLot, plan.wipCap - wip) : 0;
    raw += release;
    if (backlog > plan.expediteThreshold) { raw += 2; expedites += 1; }
    const m1 = Math.min(raw, scenario.stage1Rate);
    raw -= m1;
    buffer += m1;
    const m2 = Math.min(buffer, scenario.stage2Rate);
    buffer -= m2;
    finished += m2;
    const demand = scenario.dailyDemand + backlog;
    const ship = Math.min(finished, demand);
    finished -= ship;
    backlog = demand - ship;
    shipped += ship;
    wipArea += raw + buffer + finished;
  }
  const avgWip = wipArea / scenario.horizon;
  const service = shipped / (scenario.horizon * scenario.dailyDemand);
  const objective = 15 * shipped - 3.5 * backlog - 0.25 * avgWip - 1.2 * expedites;
  return {candidateId, plan, objective, feasible: true, metrics: {shipped, service, backlog, avgWip, expedites}};
}

// -----------------------------------------------------------------------------
// 4. Supply chain management: novel algorithms
// -----------------------------------------------------------------------------

export interface SupplyChainParams {
  horizon?: number;
}

interface SupplyChainScenario {
  horizon: number;
  demand: number[];
  leadTime: number;
}

interface SupplyChainPlan {
  baseStock: number;
  reviewPeriod: number;
  riskPooling: number;
}

export type SupplyChainResult = DomainModelResult<SupplyChainPlan>;

export function runSupplyChainRiskPooling(params: SupplyChainParams = {}): SupplyChainResult {
  const horizon = params.horizon ?? 20;
  const demand = Array.from({length: horizon}, (_v, t) => 12 + 4 * Math.sin(0.65 * t) + (t % 5 === 0 ? 5 : 0));
  const scenario: SupplyChainScenario = {horizon, demand, leadTime: 2};
  checkPositiveInt('runSupplyChainRiskPooling', 'horizon', horizon);
  return runDomainPipeline({
    modelId: 'supply-chain-risk-pooling',
    category: 'Supply chain management (novel algorithms)',
    scenario,
    generate: supplyChainCandidates,
    evaluate: evaluateSupplyChainPlan,
  });
}

function supplyChainCandidates(_scenario: SupplyChainScenario): Array<{candidateId: string; plan: SupplyChainPlan}> {
  return [
    {candidateId: 'local-minmax', plan: {baseStock: 28, reviewPeriod: 1, riskPooling: 0}},
    {candidateId: 'pooled-safety-stock', plan: {baseStock: 36, reviewPeriod: 2, riskPooling: 0.45}},
    {candidateId: 'service-first-pooling', plan: {baseStock: 44, reviewPeriod: 1, riskPooling: 0.7}},
    {candidateId: 'inventory-lean-pooling', plan: {baseStock: 32, reviewPeriod: 3, riskPooling: 0.55}},
  ];
}

function evaluateSupplyChainPlan(
  scenario: SupplyChainScenario,
  plan: SupplyChainPlan,
  candidateId: string,
): DomainEvaluation<SupplyChainPlan> {
  let invA = plan.baseStock;
  let invB = plan.baseStock;
  const pipeline: Array<{t: number; qtyA: number; qtyB: number}> = [];
  let served = 0;
  let demandTotal = 0;
  let holding = 0;
  let stockout = 0;
  for (let t = 0; t < scenario.horizon; t++) {
    for (const order of pipeline.filter(o => o.t === t)) {
      invA += order.qtyA;
      invB += order.qtyB;
    }
    const dA = scenario.demand[t] * (0.9 + 0.1 * Math.sin(t));
    const dB = scenario.demand[t] * (1.1 - 0.1 * Math.sin(t));
    const transfer = plan.riskPooling * Math.max(0, invA - invB) / 2;
    invA -= transfer;
    invB += transfer;
    const sA = Math.min(invA, dA);
    const sB = Math.min(invB, dB);
    invA -= sA;
    invB -= sB;
    served += sA + sB;
    demandTotal += dA + dB;
    stockout += (dA - sA) + (dB - sB);
    holding += invA + invB;
    if (t % plan.reviewPeriod === 0) {
      pipeline.push({
        t: t + scenario.leadTime,
        qtyA: Math.max(0, plan.baseStock - invA),
        qtyB: Math.max(0, plan.baseStock - invB),
      });
    }
  }
  const fillRate = served / demandTotal;
  const objective = 1000 * fillRate - 0.18 * holding - 5 * stockout;
  return {candidateId, plan, objective, feasible: true, metrics: {fillRate, holding, stockout, served}};
}

// -----------------------------------------------------------------------------
// 5. Operations management: novel algorithms
// -----------------------------------------------------------------------------

export interface OperationsParams {
  overtimeCost?: number;
}

interface OperationsScenario {
  demand: number[];
  overtimeCost: number;
}

interface OperationsPlan {
  staffing: number[];
  flexPool: number;
}

export type OperationsResult = DomainModelResult<OperationsPlan>;

export function runWorkforceServiceOperations(params: OperationsParams = {}): OperationsResult {
  const scenario: OperationsScenario = {
    demand: [7, 11, 15, 12, 9, 6],
    overtimeCost: params.overtimeCost ?? 18,
  };
  Preconditions.positive('runWorkforceServiceOperations', 'overtimeCost', scenario.overtimeCost);
  return runDomainPipeline({
    modelId: 'workforce-service-operations',
    category: 'Operations management (novel algorithms)',
    scenario,
    generate: operationsCandidates,
    evaluate: evaluateOperationsPlan,
  });
}

function operationsCandidates(_scenario: OperationsScenario): Array<{candidateId: string; plan: OperationsPlan}> {
  return [
    {candidateId: 'lean-fixed-roster', plan: {staffing: [7, 9, 11, 10, 8, 6], flexPool: 1}},
    {candidateId: 'service-buffer-roster', plan: {staffing: [8, 11, 14, 12, 10, 7], flexPool: 2}},
    {candidateId: 'risk-pooled-flex-roster', plan: {staffing: [7, 10, 13, 11, 9, 6], flexPool: 4}},
    {candidateId: 'overlap-wave-roster', plan: {staffing: [8, 12, 13, 13, 9, 6], flexPool: 1}},
  ];
}

function evaluateOperationsPlan(scenario: OperationsScenario, plan: OperationsPlan, candidateId: string): DomainEvaluation<OperationsPlan> {
  let covered = 0;
  let demand = 0;
  let idle = 0;
  let overtime = 0;
  for (let i = 0; i < scenario.demand.length; i++) {
    const available = plan.staffing[i] + plan.flexPool * (scenario.demand[i] > plan.staffing[i] ? 0.85 : 0.25);
    covered += Math.min(available, scenario.demand[i]);
    demand += scenario.demand[i];
    idle += Math.max(0, available - scenario.demand[i]);
    overtime += Math.max(0, scenario.demand[i] - available);
  }
  const serviceLevel = covered / demand;
  const laborCost = plan.staffing.reduce((sum, n) => sum + n, 0) * 12 + plan.flexPool * 20 + overtime * scenario.overtimeCost;
  const objective = 900 * serviceLevel - laborCost - 2 * idle;
  return {candidateId, plan, objective, feasible: serviceLevel >= 0.9, metrics: {serviceLevel, laborCost, overtime, idle}};
}

// -----------------------------------------------------------------------------
// 6. Financial engineering: applied control theory
// -----------------------------------------------------------------------------

export interface FinancialControlParams {
  initialWealth?: number;
}

interface FinancialScenario {
  initialWealth: number;
  returns: number[];
}

interface FinancialPlan {
  floorFraction: number;
  multiplier: number;
  volTarget: number;
}

export type FinancialControlResult = DomainModelResult<FinancialPlan>;

export function runPortfolioDrawdownControl(params: FinancialControlParams = {}): FinancialControlResult {
  const scenario: FinancialScenario = {
    initialWealth: params.initialWealth ?? 100,
    returns: [0.012, 0.008, -0.018, 0.015, -0.025, 0.010, 0.006, -0.010, 0.020, -0.012, 0.011, 0.007],
  };
  Preconditions.positive('runPortfolioDrawdownControl', 'initialWealth', scenario.initialWealth);
  return runDomainPipeline({
    modelId: 'portfolio-drawdown-control',
    category: 'Financial engineering (applied control theory, novel algorithms)',
    scenario,
    generate: financialCandidates,
    evaluate: evaluateFinancialPlan,
  });
}

function financialCandidates(_scenario: FinancialScenario): Array<{candidateId: string; plan: FinancialPlan}> {
  return [
    {candidateId: 'buy-and-hold', plan: {floorFraction: 0, multiplier: 1, volTarget: 1}},
    {candidateId: 'conservative-cppi', plan: {floorFraction: 0.88, multiplier: 2.2, volTarget: 0.7}},
    {candidateId: 'adaptive-drawdown-control', plan: {floorFraction: 0.9, multiplier: 3.4, volTarget: 0.55}},
    {candidateId: 'growth-cppi', plan: {floorFraction: 0.82, multiplier: 4.1, volTarget: 0.9}},
  ];
}

function evaluateFinancialPlan(scenario: FinancialScenario, plan: FinancialPlan, candidateId: string): DomainEvaluation<FinancialPlan> {
  let wealth = scenario.initialWealth;
  let peak = wealth;
  let maxDrawdown = 0;
  let turnover = 0;
  let prevRisk = 0;
  for (const r of scenario.returns) {
    const floor = scenario.initialWealth * plan.floorFraction;
    const cushion = Math.max(0, wealth - floor);
    const riskyWeight = clamp(plan.multiplier * cushion / Math.max(wealth, 1e-12), 0, plan.volTarget);
    wealth *= 1 + riskyWeight * r + (1 - riskyWeight) * 0.001;
    peak = Math.max(peak, wealth);
    maxDrawdown = Math.max(maxDrawdown, (peak - wealth) / peak);
    turnover += Math.abs(riskyWeight - prevRisk);
    prevRisk = riskyWeight;
  }
  const objective = wealth - 85 * maxDrawdown - 0.8 * turnover;
  return {candidateId, plan, objective, feasible: wealth > 0, metrics: {finalWealth: wealth, maxDrawdown, turnover}};
}

// -----------------------------------------------------------------------------
// 7. Revenue management: dynamic pricing
// -----------------------------------------------------------------------------

export interface RevenueManagementParams {
  capacity?: number;
}

interface RevenueScenario {
  capacity: number;
  periods: number;
  basePrice: number;
  baseDemand: number;
  elasticity: number;
}

interface PricingPlan {
  priceFloor: number;
  priceCeiling: number;
  scarcityGain: number;
  smoothing: number;
}

export type RevenueManagementResult = DomainModelResult<PricingPlan>;

export function runDynamicPricingRevenue(params: RevenueManagementParams = {}): RevenueManagementResult {
  const scenario: RevenueScenario = {
    capacity: params.capacity ?? 120,
    periods: 16,
    basePrice: 100,
    baseDemand: 10,
    elasticity: 1.35,
  };
  Preconditions.positive('runDynamicPricingRevenue', 'capacity', scenario.capacity);
  return runDomainPipeline({
    modelId: 'dynamic-pricing-revenue',
    category: 'Revenue management (novel dynamic pricing algorithms)',
    scenario,
    generate: pricingCandidates,
    evaluate: evaluatePricingPlan,
  });
}

function pricingCandidates(_scenario: RevenueScenario): Array<{candidateId: string; plan: PricingPlan}> {
  return [
    {candidateId: 'fixed-reference-price', plan: {priceFloor: 100, priceCeiling: 100, scarcityGain: 0, smoothing: 1}},
    {candidateId: 'scarcity-surge', plan: {priceFloor: 82, priceCeiling: 150, scarcityGain: 0.45, smoothing: 0.55}},
    {candidateId: 'bayesian-demand-smoothing', plan: {priceFloor: 88, priceCeiling: 140, scarcityGain: 0.32, smoothing: 0.78}},
    {candidateId: 'sellout-protection-pricing', plan: {priceFloor: 90, priceCeiling: 170, scarcityGain: 0.70, smoothing: 0.45}},
  ];
}

function evaluatePricingPlan(scenario: RevenueScenario, plan: PricingPlan, candidateId: string): DomainEvaluation<PricingPlan> {
  let inventory = scenario.capacity;
  let price = scenario.basePrice;
  let revenue = 0;
  let sold = 0;
  for (let t = 0; t < scenario.periods; t++) {
    const scarcity = 1 - inventory / scenario.capacity;
    const targetPrice = clamp(scenario.basePrice * (1 + plan.scarcityGain * scarcity), plan.priceFloor, plan.priceCeiling);
    price = plan.smoothing * price + (1 - plan.smoothing) * targetPrice;
    const season = 1 + 0.35 * Math.sin(Math.PI * t / Math.max(1, scenario.periods - 1));
    const demand = scenario.baseDemand * season * Math.exp(-scenario.elasticity * (price / scenario.basePrice - 1));
    const qty = Math.min(inventory, demand);
    inventory -= qty;
    sold += qty;
    revenue += qty * price;
  }
  const sellThrough = sold / scenario.capacity;
  const objective = revenue - 8 * inventory - 250 * Math.max(0, sellThrough - 0.995);
  return {candidateId, plan, objective, feasible: true, metrics: {revenue, sold, inventory, sellThrough, finalPrice: price}};
}

// -----------------------------------------------------------------------------
// 8. Revenue management: buyer-aware dynamic pricing
// -----------------------------------------------------------------------------

export interface BuyerAwareDynamicPricingParams {
  horizon?: number;
  initialInventory?: number;
  privacyBudget?: number;
  fairnessTolerance?: number;
  sustainabilityWeight?: number;
}

interface BuyerSegment {
  id: string;
  size: number;
  willingnessToPay: number;
  priceSensitivity: number;
  onlineSignal: number;
  consentRate: number;
  fairnessExpectation: number;
  retentionValue: number;
  sustainabilityPreference: number;
}

interface BuyerAwarePricingScenario extends Required<BuyerAwareDynamicPricingParams> {
  basePrice: number;
  unitCost: number;
  replenishment: number[];
  demandPulse: number[];
  segments: BuyerSegment[];
}

interface BuyerAwarePricingPlan {
  priceFloor: number;
  priceCeiling: number;
  scarcityGain: number;
  demandSignalGain: number;
  personalizationGain: number;
  consentGate: boolean;
  fairnessClamp: number;
  smoothing: number;
  maxPriceChanges: number;
  retentionCare: number;
  wastePenalty: number;
  sustainabilityCredit: number;
}

interface PeriodPricingState {
  t: number;
  publicPrice: number;
  averagePrice: number;
  inventory: number;
  sold: number;
  revenue: number;
  fairnessSpread: number;
  retentionIndex: number;
}

export type BuyerAwareDynamicPricingResult = DomainModelResult<BuyerAwarePricingPlan>;

export function runBuyerAwareDynamicPricing(params: BuyerAwareDynamicPricingParams = {}): BuyerAwareDynamicPricingResult {
  const horizon = params.horizon ?? 12;
  checkPositiveInt('runBuyerAwareDynamicPricing', 'horizon', horizon);
  const scenario: BuyerAwarePricingScenario = {
    horizon,
    initialInventory: params.initialInventory ?? 160,
    privacyBudget: params.privacyBudget ?? 0,
    fairnessTolerance: params.fairnessTolerance ?? 0.18,
    sustainabilityWeight: params.sustainabilityWeight ?? 120,
    basePrice: 100,
    unitCost: 42,
    replenishment: Array.from({length: horizon}, (_v, t) => t === Math.floor(horizon / 2) ? 34 : 0),
    demandPulse: Array.from({length: horizon}, (_v, t) => 1 + 0.18 * Math.sin(Math.PI * t / Math.max(1, horizon - 1)) + (t > horizon * 0.58 ? 0.06 : 0)),
    segments: [
      {id: 'value-seekers', size: 18, willingnessToPay: 82, priceSensitivity: 1.70, onlineSignal: 0.45, consentRate: 0.40, fairnessExpectation: 0.86, retentionValue: 8, sustainabilityPreference: 0.55},
      {id: 'convenience-buyers', size: 14, willingnessToPay: 118, priceSensitivity: 1.10, onlineSignal: 0.65, consentRate: 0.64, fairnessExpectation: 0.58, retentionValue: 12, sustainabilityPreference: 0.35},
      {id: 'premium-loyalists', size: 8, willingnessToPay: 148, priceSensitivity: 0.76, onlineSignal: 0.72, consentRate: 0.82, fairnessExpectation: 0.46, retentionValue: 18, sustainabilityPreference: 0.42},
      {id: 'privacy-protective', size: 10, willingnessToPay: 105, priceSensitivity: 1.30, onlineSignal: 0.50, consentRate: 0.18, fairnessExpectation: 0.92, retentionValue: 15, sustainabilityPreference: 0.65},
      {id: 'sustainability-led', size: 7, willingnessToPay: 126, priceSensitivity: 0.95, onlineSignal: 0.58, consentRate: 0.55, fairnessExpectation: 0.70, retentionValue: 16, sustainabilityPreference: 0.95},
    ],
  };
  Preconditions.positive('runBuyerAwareDynamicPricing', 'initialInventory', scenario.initialInventory);
  Preconditions.nonNegative('runBuyerAwareDynamicPricing', 'privacyBudget', scenario.privacyBudget);
  Preconditions.nonNegative('runBuyerAwareDynamicPricing', 'fairnessTolerance', scenario.fairnessTolerance);
  Preconditions.nonNegative('runBuyerAwareDynamicPricing', 'sustainabilityWeight', scenario.sustainabilityWeight);
  return runDomainPipeline({
    modelId: 'buyer-aware-dynamic-pricing',
    category: 'Revenue management (novel dynamic pricing algorithms)',
    scenario,
    generate: buyerAwarePricingCandidates,
    evaluate: evaluateBuyerAwarePricingPlan,
  });
}

function buyerAwarePricingCandidates(_scenario: BuyerAwarePricingScenario): Array<{candidateId: string; plan: BuyerAwarePricingPlan}> {
  return [
    {
      candidateId: 'static-reference-price',
      plan: {priceFloor: 100, priceCeiling: 100, scarcityGain: 0, demandSignalGain: 0, personalizationGain: 0, consentGate: true, fairnessClamp: 0, smoothing: 1, maxPriceChanges: 0, retentionCare: 0.55, wastePenalty: 8, sustainabilityCredit: 0.30},
    },
    {
      candidateId: 'limited-inventory-public-price',
      plan: {priceFloor: 82, priceCeiling: 138, scarcityGain: 0.38, demandSignalGain: 0.22, personalizationGain: 0, consentGate: true, fairnessClamp: 0.05, smoothing: 0.72, maxPriceChanges: 2, retentionCare: 0.66, wastePenalty: 9, sustainabilityCredit: 0.38},
    },
    {
      candidateId: 'consent-aware-buyer-signals',
      plan: {priceFloor: 80, priceCeiling: 145, scarcityGain: 0.34, demandSignalGain: 0.30, personalizationGain: 0.22, consentGate: true, fairnessClamp: 0.13, smoothing: 0.62, maxPriceChanges: 3, retentionCare: 0.78, wastePenalty: 8, sustainabilityCredit: 0.48},
    },
    {
      candidateId: 'aggressive-personalized-yield',
      plan: {priceFloor: 75, priceCeiling: 185, scarcityGain: 0.58, demandSignalGain: 0.42, personalizationGain: 0.55, consentGate: false, fairnessClamp: 0.36, smoothing: 0.35, maxPriceChanges: 8, retentionCare: 0.25, wastePenalty: 5, sustainabilityCredit: 0.10},
    },
    {
      candidateId: 'fair-sustainable-lifecycle',
      plan: {priceFloor: 86, priceCeiling: 132, scarcityGain: 0.28, demandSignalGain: 0.18, personalizationGain: 0.12, consentGate: true, fairnessClamp: 0.09, smoothing: 0.78, maxPriceChanges: 2, retentionCare: 0.95, wastePenalty: 13, sustainabilityCredit: 0.85},
    },
  ];
}

function evaluateBuyerAwarePricingPlan(
  scenario: BuyerAwarePricingScenario,
  plan: BuyerAwarePricingPlan,
  candidateId: string,
): DomainEvaluation<BuyerAwarePricingPlan> {
  let inventory = scenario.initialInventory;
  const maxInventory = scenario.initialInventory + scenario.replenishment.reduce((sum, x) => sum + x, 0);
  let publicPrice = scenario.basePrice;
  let priceChanges = 0;
  let revenue = 0;
  let grossMargin = 0;
  let soldTotal = 0;
  let priceWeightedSold = 0;
  let privacyViolations = 0;
  let fairnessSpreadSum = 0;
  let fairnessPenalty = 0;
  let retentionNumerator = 0;
  let retentionDenominator = 0;
  const trace: PeriodPricingState[] = [];

  for (let t = 0; t < scenario.horizon; t++) {
    inventory += scenario.replenishment[t] ?? 0;
    const demandPulse = scenario.demandPulse[t] ?? 1;
    const scarcity = 1 - inventory / Math.max(maxInventory, 1e-12);
    const target = clamp(
      scenario.basePrice * (1 + plan.scarcityGain * scarcity + plan.demandSignalGain * (demandPulse - 1)),
      plan.priceFloor,
      plan.priceCeiling,
    );
    const wouldChange = Math.abs(target - publicPrice) / Math.max(publicPrice, 1e-12) > 0.025;
    const changeAllowed = wouldChange && priceChanges < plan.maxPriceChanges;
    const effectiveTarget = changeAllowed ? target : publicPrice;
    if (changeAllowed) priceChanges++;
    publicPrice = plan.smoothing * publicPrice + (1 - plan.smoothing) * effectiveTarget;

    const prices = scenario.segments.map(segmentPriceFactory(scenario, plan, publicPrice));
    const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);
    const fairnessSpread = (maxPrice - minPrice) / Math.max(avgPrice, 1e-12);
    const fairnessExcess = Math.max(0, fairnessSpread - scenario.fairnessTolerance);
    const fairnessWeight = scenario.segments.reduce((sum, s) => sum + s.fairnessExpectation, 0) / scenario.segments.length;
    fairnessSpreadSum += fairnessSpread;
    fairnessPenalty += fairnessExcess * fairnessWeight * 950;

    const expectedDemand = scenario.segments.map((segment, i) => {
      const price = prices[i];
      const affordability = segment.willingnessToPay / Math.max(price, 1e-12);
      const response = Math.exp(segment.priceSensitivity * (affordability - 1));
      const privacyExposure = plan.consentGate ? 0 : (1 - segment.consentRate) * plan.personalizationGain;
      const fairnessDrag = 1 - clamp(fairnessExcess * segment.fairnessExpectation * 1.7, 0, 0.60);
      const privacyDrag = 1 - clamp(privacyExposure * 0.32, 0, 0.45);
      const sustainabilityLift = 1 + 0.07 * plan.sustainabilityCredit * segment.sustainabilityPreference;
      return segment.size * demandPulse * response * fairnessDrag * privacyDrag * sustainabilityLift;
    });
    const periodDemand = expectedDemand.reduce((sum, x) => sum + x, 0);
    const periodSold = Math.min(inventory, periodDemand);

    if (periodDemand > 0) {
      for (let i = 0; i < scenario.segments.length; i++) {
        const segment = scenario.segments[i];
        const sold = periodSold * expectedDemand[i] / periodDemand;
        const price = prices[i];
        const privacyExposure = plan.consentGate ? 0 : (1 - segment.consentRate) * plan.personalizationGain;
        const retentionFactor = clamp(
          0.92
            + 0.08 * plan.retentionCare
            + 0.04 * plan.sustainabilityCredit * segment.sustainabilityPreference
            - fairnessExcess * segment.fairnessExpectation * 1.15
            - privacyExposure * 0.38,
          0,
          1.08,
        );
        const segmentRevenue = sold * price;
        revenue += segmentRevenue;
        grossMargin += sold * (price - scenario.unitCost);
        soldTotal += sold;
        priceWeightedSold += sold * price;
        retentionNumerator += sold * segment.retentionValue * retentionFactor;
        retentionDenominator += sold * segment.retentionValue;
        privacyViolations += plan.consentGate ? 0 : sold * (1 - segment.consentRate) * plan.personalizationGain;
      }
    }
    inventory -= periodSold;
    trace.push({
      t,
      publicPrice,
      averagePrice: avgPrice,
      inventory,
      sold: periodSold,
      revenue,
      fairnessSpread,
      retentionIndex: retentionDenominator > 0 ? retentionNumerator / retentionDenominator : 1,
    });
  }

  const avgFairnessSpread = fairnessSpreadSum / scenario.horizon;
  const avgPrice = priceWeightedSold / Math.max(soldTotal, 1e-12);
  const sellThrough = soldTotal / Math.max(maxInventory, 1e-12);
  const finalInventory = inventory;
  const retentionIndex = retentionDenominator > 0 ? retentionNumerator / retentionDenominator : 1;
  const wasteShare = finalInventory / Math.max(maxInventory, 1e-12);
  const sustainabilityScore = clamp(1 - wasteShare + 0.08 * plan.sustainabilityCredit - 0.03 * priceChanges, 0, 1.1);
  const privacyCost = privacyViolations * 18 * (0.6 + plan.personalizationGain);
  const wasteCost = plan.wastePenalty * finalInventory;
  const objective = grossMargin
    + 0.20 * retentionNumerator
    + scenario.sustainabilityWeight * sustainabilityScore
    - privacyCost
    - fairnessPenalty
    - wasteCost
    - 35 * priceChanges;
  const feasible = privacyViolations <= scenario.privacyBudget + 1e-9
    && avgFairnessSpread <= scenario.fairnessTolerance + 0.025
    && retentionIndex >= 0.78;

  return {
    candidateId,
    plan,
    objective,
    feasible,
    metrics: {
      revenue,
      grossMargin,
      unitsSold: soldTotal,
      finalInventory,
      sellThrough,
      avgPrice,
      avgFairnessSpread,
      privacyViolations,
      retentionIndex,
      sustainabilityScore,
      priceChanges,
    },
    trace: {
      t: trace.map(row => row.t),
      series: {
        publicPrice: trace.map(row => row.publicPrice),
        averagePrice: trace.map(row => row.averagePrice),
        inventory: trace.map(row => row.inventory),
        sold: trace.map(row => row.sold),
        cumulativeRevenue: trace.map(row => row.revenue),
        fairnessSpread: trace.map(row => row.fairnessSpread),
        retentionIndex: trace.map(row => row.retentionIndex),
      },
      captions: trace.map(row => `t=${row.t}: price=${row.averagePrice.toFixed(2)} inventory=${row.inventory.toFixed(1)} fairness=${row.fairnessSpread.toFixed(3)}`),
    },
  };
}

function segmentPriceFactory(
  scenario: BuyerAwarePricingScenario,
  plan: BuyerAwarePricingPlan,
  publicPrice: number,
): (segment: BuyerSegment) => number {
  return (segment: BuyerSegment) => {
    const consentShare = plan.consentGate ? segment.consentRate : 1;
    const personalComponent = scenario.basePrice * plan.personalizationGain * (segment.willingnessToPay / scenario.basePrice - 1) * consentShare;
    const signalComponent = scenario.basePrice * plan.demandSignalGain * 0.12 * (segment.onlineSignal - 0.55) * consentShare;
    const raw = publicPrice + personalComponent + signalComponent;
    const lo = Math.max(plan.priceFloor, publicPrice * (1 - plan.fairnessClamp));
    const hi = Math.min(plan.priceCeiling, publicPrice * (1 + plan.fairnessClamp));
    return clamp(raw, lo, hi);
  };
}

// -----------------------------------------------------------------------------
// 9. Energy: optimization of power systems
// -----------------------------------------------------------------------------

export interface EnergyParams {
  batteryCapacity?: number;
}

interface EnergyScenario {
  demand: number[];
  renewable: number[];
  price: number[];
  batteryCapacity: number;
  maxCharge: number;
}

interface EnergyPlan {
  chargeBelow: number;
  dischargeAbove: number;
  reserve: number;
}

export type EnergyResult = DomainModelResult<EnergyPlan>;

export function runEnergyStorageDispatch(params: EnergyParams = {}): EnergyResult {
  const scenario: EnergyScenario = {
    demand: [42, 40, 38, 36, 45, 58, 67, 72, 68, 54, 48, 44],
    renewable: [8, 9, 12, 20, 30, 34, 28, 18, 12, 9, 8, 7],
    price: [36, 32, 28, 24, 18, 22, 42, 68, 74, 55, 44, 38],
    batteryCapacity: params.batteryCapacity ?? 50,
    maxCharge: 12,
  };
  Preconditions.positive('runEnergyStorageDispatch', 'batteryCapacity', scenario.batteryCapacity);
  return runDomainPipeline({
    modelId: 'energy-storage-dispatch',
    category: 'Energy (optimization of power systems)',
    scenario,
    generate: energyCandidates,
    evaluate: evaluateEnergyPlan,
  });
}

function energyCandidates(_scenario: EnergyScenario): Array<{candidateId: string; plan: EnergyPlan}> {
  return [
    {candidateId: 'no-storage-reference', plan: {chargeBelow: -Infinity, dischargeAbove: Infinity, reserve: 0}},
    {candidateId: 'price-arbitrage-dispatch', plan: {chargeBelow: 30, dischargeAbove: 55, reserve: 8}},
    {candidateId: 'renewable-first-dispatch', plan: {chargeBelow: 42, dischargeAbove: 62, reserve: 15}},
    {candidateId: 'reliability-reserve-dispatch', plan: {chargeBelow: 34, dischargeAbove: 48, reserve: 22}},
  ];
}

function evaluateEnergyPlan(scenario: EnergyScenario, plan: EnergyPlan, candidateId: string): DomainEvaluation<EnergyPlan> {
  let soc = scenario.batteryCapacity / 2;
  let cost = 0;
  let curtailment = 0;
  let unserved = 0;
  let emissions = 0;
  for (let t = 0; t < scenario.demand.length; t++) {
    let netLoad = scenario.demand[t] - scenario.renewable[t];
    if (scenario.price[t] < plan.chargeBelow) {
      const charge = Math.min(scenario.maxCharge, scenario.batteryCapacity - soc);
      soc += charge;
      netLoad += charge / 0.92;
    }
    if (scenario.price[t] > plan.dischargeAbove && soc > plan.reserve) {
      const discharge = Math.min(scenario.maxCharge, soc - plan.reserve, Math.max(0, netLoad));
      soc -= discharge;
      netLoad -= 0.92 * discharge;
    }
    if (netLoad < 0) curtailment += -netLoad;
    const thermal = Math.max(0, netLoad);
    cost += thermal * scenario.price[t] + 0.08 * thermal * thermal;
    emissions += 0.45 * thermal;
    unserved += Math.max(0, netLoad - 75);
  }
  const objective = -cost - 1000 * unserved - 8 * curtailment - 2 * emissions;
  return {candidateId, plan, objective, feasible: unserved < 1e-9, metrics: {cost, curtailment, unserved, emissions, finalSoc: soc}};
}

// -----------------------------------------------------------------------------
// 10. Machine learning/statistical learning: novel algorithms/use cases
// -----------------------------------------------------------------------------

export interface ActiveLearningParams {
  budget?: number;
}

interface ActiveLearningScenario {
  budget: number;
  pool: Array<{id: number; uncertainty: number; diversity: number; cost: number; value: number}>;
}

interface ActiveLearningPlan {
  uncertaintyWeight: number;
  diversityWeight: number;
  costWeight: number;
}

export type ActiveLearningResult = DomainModelResult<ActiveLearningPlan>;

export function runActiveLearningAcquisition(params: ActiveLearningParams = {}): ActiveLearningResult {
  const scenario: ActiveLearningScenario = {
    budget: params.budget ?? 9,
    pool: [
      {id: 1, uncertainty: 0.92, diversity: 0.35, cost: 2, value: 0.9},
      {id: 2, uncertainty: 0.65, diversity: 0.80, cost: 3, value: 0.85},
      {id: 3, uncertainty: 0.74, diversity: 0.72, cost: 2, value: 0.78},
      {id: 4, uncertainty: 0.40, diversity: 0.95, cost: 2, value: 0.66},
      {id: 5, uncertainty: 0.88, diversity: 0.45, cost: 4, value: 0.95},
      {id: 6, uncertainty: 0.55, diversity: 0.60, cost: 1, value: 0.60},
    ],
  };
  Preconditions.positive('runActiveLearningAcquisition', 'budget', scenario.budget);
  return runDomainPipeline({
    modelId: 'active-learning-acquisition',
    category: 'Machine learning and statistical learning (novel algorithms and novel use cases)',
    scenario,
    generate: activeLearningCandidates,
    evaluate: evaluateActiveLearningPlan,
  });
}

function activeLearningCandidates(_scenario: ActiveLearningScenario): Array<{candidateId: string; plan: ActiveLearningPlan}> {
  return [
    {candidateId: 'uncertainty-sampling', plan: {uncertaintyWeight: 1, diversityWeight: 0, costWeight: 0}},
    {candidateId: 'diversity-regularized-active-learning', plan: {uncertaintyWeight: 0.7, diversityWeight: 0.55, costWeight: 0.1}},
    {candidateId: 'cost-aware-information-gain', plan: {uncertaintyWeight: 0.75, diversityWeight: 0.35, costWeight: 0.45}},
    {candidateId: 'balanced-portfolio-acquisition', plan: {uncertaintyWeight: 0.55, diversityWeight: 0.65, costWeight: 0.25}},
  ];
}

function evaluateActiveLearningPlan(
  scenario: ActiveLearningScenario,
  plan: ActiveLearningPlan,
  candidateId: string,
): DomainEvaluation<ActiveLearningPlan> {
  const ranked = scenario.pool.slice().sort((a, b) => scoreActive(b, plan) - scoreActive(a, plan));
  let cost = 0;
  let infoGain = 0;
  const selected: number[] = [];
  for (const item of ranked) {
    if (cost + item.cost > scenario.budget) continue;
    selected.push(item.id);
    cost += item.cost;
    infoGain += item.value * (0.65 * item.uncertainty + 0.35 * item.diversity);
  }
  const expectedErrorReduction = 1 - Math.exp(-infoGain / 2.8);
  const objective = 100 * expectedErrorReduction - 0.8 * cost + 2 * selected.length;
  return {
    candidateId,
    plan,
    objective,
    feasible: selected.length > 0,
    metrics: {selected: selected.join('|'), cost, infoGain, expectedErrorReduction},
  };
}

function scoreActive(item: ActiveLearningScenario['pool'][number], plan: ActiveLearningPlan): number {
  return plan.uncertaintyWeight * item.uncertainty + plan.diversityWeight * item.diversity - plan.costWeight * item.cost;
}

// -----------------------------------------------------------------------------
// 11. Decision science: data science + visualization
// -----------------------------------------------------------------------------

export interface DecisionScienceParams {
  riskWeight?: number;
}

interface DecisionScenario {
  alternatives: Array<{name: string; cost: number; impact: number; risk: number; adoption: number}>;
  riskWeight: number;
}

interface DecisionPlan {
  impactWeight: number;
  adoptionWeight: number;
  costWeight: number;
  riskWeight: number;
}

export type DecisionScienceResult = DomainModelResult<DecisionPlan>;

export function runVisualDecisionFrontier(params: DecisionScienceParams = {}): DecisionScienceResult {
  const scenario: DecisionScenario = {
    riskWeight: params.riskWeight ?? 0.35,
    alternatives: [
      {name: 'pilot automation', cost: 42, impact: 78, risk: 22, adoption: 74},
      {name: 'full platform rebuild', cost: 88, impact: 96, risk: 65, adoption: 58},
      {name: 'targeted workflow redesign', cost: 35, impact: 70, risk: 18, adoption: 82},
      {name: 'analytics copilot', cost: 54, impact: 86, risk: 35, adoption: 76},
      {name: 'status quo plus training', cost: 18, impact: 42, risk: 9, adoption: 90},
    ],
  };
  Preconditions.nonNegative('runVisualDecisionFrontier', 'riskWeight', scenario.riskWeight);
  return runDomainPipeline({
    modelId: 'visual-decision-frontier',
    category: 'Decision science (using data science combined with visualization)',
    scenario,
    generate: decisionCandidates,
    evaluate: evaluateDecisionPlan,
  });
}

function decisionCandidates(scenario: DecisionScenario): Array<{candidateId: string; plan: DecisionPlan}> {
  return [
    {candidateId: 'impact-led-view', plan: {impactWeight: 0.60, adoptionWeight: 0.20, costWeight: 0.12, riskWeight: scenario.riskWeight}},
    {candidateId: 'adoption-led-view', plan: {impactWeight: 0.38, adoptionWeight: 0.42, costWeight: 0.12, riskWeight: scenario.riskWeight}},
    {candidateId: 'risk-adjusted-frontier', plan: {impactWeight: 0.48, adoptionWeight: 0.28, costWeight: 0.08, riskWeight: scenario.riskWeight + 0.2}},
    {candidateId: 'lean-value-frontier', plan: {impactWeight: 0.42, adoptionWeight: 0.25, costWeight: 0.25, riskWeight: scenario.riskWeight}},
  ];
}

function evaluateDecisionPlan(scenario: DecisionScenario, plan: DecisionPlan, candidateId: string): DomainEvaluation<DecisionPlan> {
  const scored = scenario.alternatives.map(alt => {
    const score = plan.impactWeight * alt.impact
      + plan.adoptionWeight * alt.adoption
      - plan.costWeight * alt.cost
      - plan.riskWeight * alt.risk;
    return {...alt, score};
  }).sort((a, b) => b.score - a.score);
  const top = scored[0];
  const separation = scored.length > 1 ? top.score - scored[1].score : top.score;
  const frontierCount = scored.filter(alt => alt.impact >= 70 && alt.risk <= 40).length;
  const objective = top.score + 0.15 * separation + frontierCount;
  return {
    candidateId,
    plan,
    objective,
    feasible: true,
    metrics: {topAlternative: top.name, topScore: top.score, separation, frontierCount, visualizationReady: true},
  };
}

// -----------------------------------------------------------------------------
// Shared math helpers
// -----------------------------------------------------------------------------

function customerById(scenario: RoutingScenario, id: number): Customer {
  const c = scenario.customers.find(item => item.id === id);
  if (!c) throw new Error(`unknown customer ${id}`);
  return c;
}

function routeLoad(scenario: RoutingScenario, route: readonly number[]): number {
  return route.reduce((sum, id) => sum + customerById(scenario, id).demand, 0);
}

function routeLength(scenario: RoutingScenario, route: readonly number[]): number {
  let total = 0;
  let cur = scenario.depot;
  for (const id of route) {
    const c = customerById(scenario, id);
    total += dist(cur, c);
    cur = c;
  }
  return total + dist(cur, scenario.depot);
}

function dist(a: {x: number; y: number}, b: {x: number; y: number}): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function checkPositiveInt(model: string, param: string, value: number): void {
  Preconditions.integer(model, param, value);
  Preconditions.check(model, param, 'be >= 1', value >= 1, value);
}
