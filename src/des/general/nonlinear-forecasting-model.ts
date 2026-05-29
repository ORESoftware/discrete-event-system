'use strict';

// =============================================================================
// nonlinear-forecasting-model.ts
//
// Nonlinear prediction/forecasting as an explicit DES station graph:
//
//   ForecastDataSource -> POMDPLatentVariable -> MDPVariableDiscovery
//     -> NonlinearEquationTuning -> ForecastProjection -> ResultSink
//
// The POMDP station infers hidden regime beliefs from noisy observations. The
// MDP station treats candidate variables as discovery actions and uses value
// iteration to decide which observed, nonlinear, lagged, and latent-belief
// variables are worth adding. The equation station fine-tunes a nonlinear basis
// expansion, and the projection station rolls it forward for forecasting.
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
import {DiscreteBelief} from './belief';
import {POMDPSpec, beliefUpdate, mdpValueIteration} from './pomdp';

type RegimeId = 'baseline' | 'expansion' | 'contraction' | 'shock';
type RegimeObservation = 'low' | 'flat' | 'high' | 'volatile';
type VariableSource = 'observed' | 'lagged' | 'nonlinear' | 'pomdp';

const REGIMES: RegimeId[] = ['baseline', 'expansion', 'contraction', 'shock'];
const REGIME_OBSERVATIONS: RegimeObservation[] = ['low', 'flat', 'high', 'volatile'];
const OBSERVE_ACTION = ['observe'];

const CH_DATA: ChannelName = 'forecast-data';
const CH_BELIEF: ChannelName = 'latent-belief-trace';
const CH_VARIABLES: ChannelName = 'discovered-variables';
const CH_EQUATION: ChannelName = 'fine-tuned-equation';
const CH_PROJECTION: ChannelName = 'forecast-projection';

export interface NonlinearMDPPOMDPForecastParams {
  trainingPeriods?: number;
  forecastHorizon?: number;
  mdpBudget?: number;
  ridge?: number;
  fineTuneIterations?: number;
  validationShare?: number;
}

interface NormalizedForecastParams {
  trainingPeriods: number;
  forecastHorizon: number;
  mdpBudget: number;
  ridge: number;
  fineTuneIterations: number;
  validationShare: number;
}

interface ForecastObservation {
  t: number;
  demand: number;
  supply: number;
  price: number;
  y: number;
  hiddenRegime: RegimeId;
}

interface ForecastScenario {
  params: NormalizedForecastParams;
  observations: ForecastObservation[];
  featureCandidates: FeatureCandidate[];
  pomdpSpec: POMDPSpec<RegimeId, string, RegimeObservation>;
}

interface FeatureContext {
  t: number;
  demand: number;
  supplyGap: number;
  price: number;
  lagY: number;
  momentum: number;
  trend: number;
  beliefBaseline: number;
  beliefExpansion: number;
  beliefContraction: number;
  beliefShock: number;
}

interface FeatureCandidate {
  id: string;
  label: string;
  source: VariableSource;
  cost: number;
  compute: (ctx: FeatureContext) => number;
}

interface ForecastRow {
  t: number;
  target: number;
  context: FeatureContext;
  split: 'train' | 'validation';
}

export interface LatentBeliefPoint {
  t: number;
  observation: RegimeObservation;
  prior: number[];
  posterior: number[];
  mode: RegimeId;
  entropy: number;
}

export interface LatentBeliefTrace {
  states: RegimeId[];
  points: LatentBeliefPoint[];
  finalBelief: number[];
  transitionMatrix: number[][];
}

export interface MDPDiscoveryStep {
  step: number;
  stateMask: number;
  action: string;
  reward: number;
  validationMseBefore: number;
  validationMseAfter: number;
  selectedAfter: string[];
}

export interface VariableDiscoveryResult {
  selectedFeatureIndices: number[];
  selectedVariables: Array<{id: string; label: string; source: VariableSource; cost: number}>;
  rows: ForecastRow[];
  trainRows: ForecastRow[];
  validationRows: ForecastRow[];
  baselineValidationMse: number;
  validationMse: number;
  trainMse: number;
  mdpStates: number;
  mdpActions: number;
  mdpIterations: number;
  mdpFinalDelta: number;
  actionTrace: MDPDiscoveryStep[];
}

export interface FineTuneTraceRow {
  iter: number;
  mse: number;
  validationMse: number;
  coefficients: number[];
}

export interface TunedEquation {
  featureIndices: number[];
  featureIds: string[];
  featureLabels: string[];
  coefficients: number[];
  means: number[];
  scales: number[];
  intercept: number;
  equationText: string;
  inSampleMse: number;
  validationMse: number;
  trace: FineTuneTraceRow[];
  fitted: Array<{t: number; actual: number; predicted: number; split: 'train' | 'validation'}>;
}

export interface ForecastProjectionPoint {
  t: number;
  horizonStep: number;
  forecast: number;
  actual: number;
  lower: number;
  upper: number;
  beliefMode: RegimeId;
  beliefEntropy: number;
}

export interface NonlinearMDPPOMDPForecastResult {
  modelId: 'nonlinear-mdp-pomdp-forecast';
  selectedVariables: string[];
  discoveredVariables: VariableDiscoveryResult['selectedVariables'];
  equation: TunedEquation;
  pomdp: LatentBeliefTrace;
  mdp: {
    states: number;
    actions: number;
    iterations: number;
    finalDelta: number;
    actionTrace: MDPDiscoveryStep[];
  };
  projection: ForecastProjectionPoint[];
  metrics: {
    baselineValidationMse: number;
    validationMse: number;
    trainMse: number;
    inSampleMse: number;
    forecastMse: number;
    baselineForecastMse: number;
    finalBeliefEntropy: number;
    selectedVariableCount: number;
  };
  topology: StationGraphSummary;
}

class ForecastDataToken implements Token {
  constructor(readonly scenario: ForecastScenario) {}
}

class LatentBeliefTraceToken implements Token {
  constructor(readonly scenario: ForecastScenario, readonly beliefTrace: LatentBeliefTrace) {}
}

class DiscoveredVariablesToken implements Token {
  constructor(
    readonly scenario: ForecastScenario,
    readonly beliefTrace: LatentBeliefTrace,
    readonly discovery: VariableDiscoveryResult,
  ) {}
}

class FineTunedEquationToken implements Token {
  constructor(
    readonly scenario: ForecastScenario,
    readonly beliefTrace: LatentBeliefTrace,
    readonly discovery: VariableDiscoveryResult,
    readonly equation: TunedEquation,
  ) {}
}

class ForecastProjectionToken implements Token {
  constructor(readonly result: NonlinearMDPPOMDPForecastResult) {}
}

class ForecastDataSourceStation extends DESStation {
  static readonly CH_DATA: ChannelName = CH_DATA;
  private emitted = false;

  constructor(id: string, private readonly scenario: ForecastScenario) {
    super(id);
  }

  override hasWork(): boolean { return !this.emitted; }

  runTimeStep(): void {
    if (this.emitted) return;
    this.emit(new ForecastDataToken(this.scenario), ForecastDataSourceStation.CH_DATA);
    this.emitted = true;
  }
}

class POMDPLatentVariableStation extends DESStation {
  static readonly CH_DATA: ChannelName = CH_DATA;
  static readonly CH_BELIEF: ChannelName = CH_BELIEF;

  override hasWork(): boolean { return this.inboxSize(POMDPLatentVariableStation.CH_DATA) > 0; }

  runTimeStep(): void {
    for (const token of this.drain<ForecastDataToken>(POMDPLatentVariableStation.CH_DATA)) {
      this.emit(
        new LatentBeliefTraceToken(token.scenario, inferLatentRegimeBeliefs(token.scenario)),
        POMDPLatentVariableStation.CH_BELIEF,
      );
    }
  }
}

class MDPVariableDiscoveryStation extends DESStation {
  static readonly CH_BELIEF: ChannelName = CH_BELIEF;
  static readonly CH_VARIABLES: ChannelName = CH_VARIABLES;

  override hasWork(): boolean { return this.inboxSize(MDPVariableDiscoveryStation.CH_BELIEF) > 0; }

  runTimeStep(): void {
    for (const token of this.drain<LatentBeliefTraceToken>(MDPVariableDiscoveryStation.CH_BELIEF)) {
      const discovery = discoverVariablesByMDP(token.scenario, token.beliefTrace);
      this.emit(
        new DiscoveredVariablesToken(token.scenario, token.beliefTrace, discovery),
        MDPVariableDiscoveryStation.CH_VARIABLES,
      );
    }
  }
}

class NonlinearEquationTuningStation extends DESStation {
  static readonly CH_VARIABLES: ChannelName = CH_VARIABLES;
  static readonly CH_EQUATION: ChannelName = CH_EQUATION;

  override hasWork(): boolean { return this.inboxSize(NonlinearEquationTuningStation.CH_VARIABLES) > 0; }

  runTimeStep(): void {
    for (const token of this.drain<DiscoveredVariablesToken>(NonlinearEquationTuningStation.CH_VARIABLES)) {
      const equation = fineTuneEquation(token.scenario, token.discovery);
      this.emit(
        new FineTunedEquationToken(token.scenario, token.beliefTrace, token.discovery, equation),
        NonlinearEquationTuningStation.CH_EQUATION,
      );
    }
  }
}

class ForecastProjectionStation extends DESStation {
  static readonly CH_EQUATION: ChannelName = CH_EQUATION;
  static readonly CH_PROJECTION: ChannelName = CH_PROJECTION;

  override hasWork(): boolean { return this.inboxSize(ForecastProjectionStation.CH_EQUATION) > 0; }

  runTimeStep(): void {
    for (const token of this.drain<FineTunedEquationToken>(ForecastProjectionStation.CH_EQUATION)) {
      const projection = projectForecast(token.scenario, token.beliefTrace, token.equation);
      const actual = projection.map(row => row.actual);
      const predicted = projection.map(row => row.forecast);
      const lastTrainingY = token.scenario.observations[token.scenario.params.trainingPeriods - 1].y;
      const baselineForecastMse = mse(actual, actual.map(() => lastTrainingY));
      const result: NonlinearMDPPOMDPForecastResult = {
        modelId: 'nonlinear-mdp-pomdp-forecast',
        selectedVariables: token.discovery.selectedVariables.map(v => v.id),
        discoveredVariables: token.discovery.selectedVariables,
        equation: token.equation,
        pomdp: token.beliefTrace,
        mdp: {
          states: token.discovery.mdpStates,
          actions: token.discovery.mdpActions,
          iterations: token.discovery.mdpIterations,
          finalDelta: token.discovery.mdpFinalDelta,
          actionTrace: token.discovery.actionTrace,
        },
        projection,
        metrics: {
          baselineValidationMse: token.discovery.baselineValidationMse,
          validationMse: token.discovery.validationMse,
          trainMse: token.discovery.trainMse,
          inSampleMse: token.equation.inSampleMse,
          forecastMse: mse(actual, predicted),
          baselineForecastMse,
          finalBeliefEntropy: entropy(token.beliefTrace.finalBelief),
          selectedVariableCount: token.discovery.selectedVariables.length,
        },
        topology: stationGraph([], [], []),
      };
      this.emit(new ForecastProjectionToken(result), ForecastProjectionStation.CH_PROJECTION);
    }
  }
}

class ForecastResultSinkStation extends DESStation {
  static readonly CH_PROJECTION: ChannelName = CH_PROJECTION;
  result: NonlinearMDPPOMDPForecastResult | undefined;

  override hasWork(): boolean { return this.inboxSize(ForecastResultSinkStation.CH_PROJECTION) > 0; }

  runTimeStep(): void {
    const tokens = this.drain<ForecastProjectionToken>(ForecastResultSinkStation.CH_PROJECTION);
    if (tokens.length > 0) this.result = tokens[tokens.length - 1].result;
  }
}

export function runNonlinearMDPPOMDPForecast(params: NonlinearMDPPOMDPForecastParams = {}): NonlinearMDPPOMDPForecastResult {
  const scenario = buildForecastScenario(params);
  const source = new ForecastDataSourceStation('nonlinear-forecast-data-source', scenario);
  const pomdp = new POMDPLatentVariableStation('pomdp-latent-variable-station');
  const mdp = new MDPVariableDiscoveryStation('mdp-variable-discovery-station');
  const tuning = new NonlinearEquationTuningStation('nonlinear-equation-tuning-station');
  const projection = new ForecastProjectionStation('forecast-projection-station');
  const sink = new ForecastResultSinkStation('forecast-result-sink');

  source.pipe(pomdp, ForecastDataSourceStation.CH_DATA, POMDPLatentVariableStation.CH_DATA);
  pomdp.pipe(mdp, POMDPLatentVariableStation.CH_BELIEF, MDPVariableDiscoveryStation.CH_BELIEF);
  mdp.pipe(tuning, MDPVariableDiscoveryStation.CH_VARIABLES, NonlinearEquationTuningStation.CH_VARIABLES);
  tuning.pipe(projection, NonlinearEquationTuningStation.CH_EQUATION, ForecastProjectionStation.CH_EQUATION);
  projection.pipe(sink, ForecastProjectionStation.CH_PROJECTION, ForecastResultSinkStation.CH_PROJECTION);

  runIterativeDES([source, pomdp, mdp, tuning, projection, sink], {shuffle: false, maxTicks: 12, runValidators: false});
  if (!sink.result) throw new Error('nonlinear-mdp-pomdp-forecast did not produce a result');
  sink.result.topology = stationGraph(
    [source, pomdp, mdp, tuning, projection, sink],
    ['ForecastDataToken', 'LatentBeliefTraceToken', 'DiscoveredVariablesToken', 'FineTunedEquationToken', 'ForecastProjectionToken'],
    [
      channelEdge(source, ForecastDataSourceStation.CH_DATA, pomdp, POMDPLatentVariableStation.CH_DATA),
      channelEdge(pomdp, POMDPLatentVariableStation.CH_BELIEF, mdp, MDPVariableDiscoveryStation.CH_BELIEF),
      channelEdge(mdp, MDPVariableDiscoveryStation.CH_VARIABLES, tuning, NonlinearEquationTuningStation.CH_VARIABLES),
      channelEdge(tuning, NonlinearEquationTuningStation.CH_EQUATION, projection, ForecastProjectionStation.CH_EQUATION),
      channelEdge(projection, ForecastProjectionStation.CH_PROJECTION, sink, ForecastResultSinkStation.CH_PROJECTION),
    ],
  );
  return sink.result;
}

function buildForecastScenario(params: NonlinearMDPPOMDPForecastParams): ForecastScenario {
  const actual = normalizeParams(params);
  const observations = syntheticForecastSeries(actual.trainingPeriods, actual.forecastHorizon);
  return {
    params: actual,
    observations,
    featureCandidates: featureCandidates(),
    pomdpSpec: buildRegimePOMDP(),
  };
}

function normalizeParams(params: NonlinearMDPPOMDPForecastParams): NormalizedForecastParams {
  const actual: NormalizedForecastParams = {
    trainingPeriods: params.trainingPeriods ?? 42,
    forecastHorizon: params.forecastHorizon ?? 8,
    mdpBudget: params.mdpBudget ?? 6,
    ridge: params.ridge ?? 0.03,
    fineTuneIterations: params.fineTuneIterations ?? 18,
    validationShare: params.validationShare ?? 0.25,
  };
  Preconditions.integerInRange('runNonlinearMDPPOMDPForecast', 'trainingPeriods', actual.trainingPeriods, 18, 200);
  Preconditions.integerInRange('runNonlinearMDPPOMDPForecast', 'forecastHorizon', actual.forecastHorizon, 1, 80);
  Preconditions.integerInRange('runNonlinearMDPPOMDPForecast', 'mdpBudget', actual.mdpBudget, 1, 10);
  Preconditions.nonNegative('runNonlinearMDPPOMDPForecast', 'ridge', actual.ridge);
  Preconditions.integerInRange('runNonlinearMDPPOMDPForecast', 'fineTuneIterations', actual.fineTuneIterations, 1, 200);
  Preconditions.inRange('runNonlinearMDPPOMDPForecast', 'validationShare', actual.validationShare, 0.1, 0.5);
  return actual;
}

function syntheticForecastSeries(trainingPeriods: number, forecastHorizon: number): ForecastObservation[] {
  const total = trainingPeriods + forecastHorizon;
  const out: ForecastObservation[] = [];
  let y = 58;
  for (let t = 0; t < total; t++) {
    const hiddenRegime = hiddenRegimeAt(t, trainingPeriods);
    const shock = hiddenRegime === 'shock' ? 1 : 0;
    const contraction = hiddenRegime === 'contraction' ? 1 : 0;
    const expansion = hiddenRegime === 'expansion' ? 1 : 0;
    const demand = 1.18 + 0.018 * t + 0.22 * Math.sin(t / 3.2) + 0.12 * expansion - 0.13 * shock;
    const supply = 1.02 + 0.17 * Math.cos(t / 4.6) + 0.04 * expansion - 0.11 * contraction - 0.23 * shock;
    const price = 1.00 + 0.07 * Math.sin(t / 5.1) + Math.max(0, 0.94 - supply) * 0.24 + 0.10 * shock;
    const regimeLift = hiddenRegime === 'expansion' ? 5.8 : hiddenRegime === 'contraction' ? -5.2 : hiddenRegime === 'shock' ? -11.5 : 0;
    const deterministicNoise = 1.1 * Math.sin(1.7 * t) + 0.55 * Math.cos(0.61 * t);
    if (t === 0) {
      y = 56 + 8 * Math.tanh(0.9 * demand - 0.65 * price) + regimeLift + deterministicNoise;
    } else {
      const lagY = out[t - 1].y;
      y = 17.5
        + 0.64 * lagY
        + 14.5 * Math.tanh(0.92 * demand - 0.68 * price)
        + 6.4 * demand * (1.12 - supply)
        + 2.2 * Math.sin(t / 5.7)
        + regimeLift
        + deterministicNoise;
    }
    out.push({t, demand, supply, price, y, hiddenRegime});
  }
  return out;
}

function hiddenRegimeAt(t: number, trainingPeriods: number): RegimeId {
  if (t < Math.floor(trainingPeriods * 0.28)) return 'baseline';
  if (t < Math.floor(trainingPeriods * 0.52)) return 'expansion';
  if (t < Math.floor(trainingPeriods * 0.64)) return 'shock';
  if (t < trainingPeriods) return 'contraction';
  return t < trainingPeriods + 3 ? 'baseline' : 'expansion';
}

function buildRegimePOMDP(): POMDPSpec<RegimeId, string, RegimeObservation> {
  const transitionMatrix = regimeTransitionMatrix();
  return {
    states: REGIMES,
    actions: OBSERVE_ACTION,
    observations: REGIME_OBSERVATIONS,
    transition: (sIdx: number, _aIdx: number) => transitionMatrix[sIdx],
    observation: (sNextIdx: number, _aIdx: number) => observationLikelihood(REGIMES[sNextIdx]),
    reward: (_sIdx: number, _aIdx: number) => -0.01,
    discount: 0.94,
    initialBelief: [0.64, 0.14, 0.14, 0.08],
  };
}

function regimeTransitionMatrix(): number[][] {
  return [
    [0.72, 0.16, 0.08, 0.04],
    [0.18, 0.70, 0.04, 0.08],
    [0.24, 0.04, 0.66, 0.06],
    [0.20, 0.10, 0.25, 0.45],
  ];
}

function observationLikelihood(regime: RegimeId): number[] {
  switch (regime) {
    case 'baseline': return [0.17, 0.56, 0.18, 0.09];
    case 'expansion': return [0.07, 0.20, 0.62, 0.11];
    case 'contraction': return [0.60, 0.21, 0.07, 0.12];
    case 'shock': return [0.25, 0.10, 0.10, 0.55];
  }
}

function inferLatentRegimeBeliefs(scenario: ForecastScenario): LatentBeliefTrace {
  const spec = scenario.pomdpSpec;
  let posterior = new DiscreteBelief(spec.states, spec.initialBelief);
  const points: LatentBeliefPoint[] = [{
    t: 0,
    observation: 'flat',
    prior: posterior.asArray(),
    posterior: posterior.asArray(),
    mode: posterior.mode(),
    entropy: posterior.entropy(),
  }];
  for (let t = 1; t < scenario.params.trainingPeriods; t++) {
    const obs = classifyRegimeObservation(scenario.observations[t - 1], scenario.observations[t]);
    const prior = posterior.clone();
    prior.propagate((_state, index) => spec.transition(index, 0));
    posterior = beliefUpdate(spec, posterior, 0, REGIME_OBSERVATIONS.indexOf(obs));
    points.push({
      t,
      observation: obs,
      prior: prior.asArray(),
      posterior: posterior.asArray(),
      mode: posterior.mode(),
      entropy: posterior.entropy(),
    });
  }
  return {
    states: REGIMES.slice(),
    points,
    finalBelief: posterior.asArray(),
    transitionMatrix: regimeTransitionMatrix(),
  };
}

function classifyRegimeObservation(prev: ForecastObservation, cur: ForecastObservation): RegimeObservation {
  const dy = cur.y - prev.y;
  const expected = 8.0 * (cur.demand - prev.demand) - 5.0 * (cur.price - prev.price) + 3.0 * (cur.supply - prev.supply);
  const residual = dy - expected;
  if (Math.abs(residual) > 7.0 || Math.abs(cur.supply - prev.supply) > 0.18) return 'volatile';
  if (residual > 2.1) return 'high';
  if (residual < -2.1) return 'low';
  return 'flat';
}

function featureCandidates(): FeatureCandidate[] {
  return [
    {id: 'observed-demand-index', label: 'observed demand index', source: 'observed', cost: 0.25, compute: ctx => ctx.demand},
    {id: 'observed-supply-gap', label: 'observed supply gap', source: 'observed', cost: 0.22, compute: ctx => ctx.supplyGap},
    {id: 'observed-price-pressure', label: 'observed price pressure', source: 'observed', cost: 0.28, compute: ctx => ctx.price},
    {id: 'lagged-outcome', label: 'lagged outcome', source: 'lagged', cost: 0.30, compute: ctx => ctx.lagY},
    {id: 'lagged-momentum', label: 'lagged momentum', source: 'lagged', cost: 0.34, compute: ctx => ctx.momentum},
    {id: 'nonlinear-demand-saturation', label: 'nonlinear demand saturation', source: 'nonlinear', cost: 0.38, compute: ctx => Math.tanh(0.92 * ctx.demand - 0.68 * ctx.price)},
    {id: 'nonlinear-demand-supply-coupling', label: 'nonlinear demand/supply coupling', source: 'nonlinear', cost: 0.42, compute: ctx => ctx.demand * ctx.supplyGap},
    {id: 'latent-expansion-belief', label: 'POMDP expansion belief', source: 'pomdp', cost: 0.18, compute: ctx => ctx.beliefExpansion},
    {id: 'latent-contraction-belief', label: 'POMDP contraction belief', source: 'pomdp', cost: 0.20, compute: ctx => ctx.beliefContraction},
    {id: 'latent-shock-belief', label: 'POMDP shock belief', source: 'pomdp', cost: 0.20, compute: ctx => ctx.beliefShock},
  ];
}

function discoverVariablesByMDP(scenario: ForecastScenario, beliefTrace: LatentBeliefTrace): VariableDiscoveryResult {
  const rows = buildForecastRows(scenario, beliefTrace);
  const split = splitRows(rows, scenario.params.validationShare);
  const numFeatures = scenario.featureCandidates.length;
  const states = Array.from({length: 1 << numFeatures}, (_v, mask) => mask);
  const actions = scenario.featureCandidates.map(f => f.id).concat('stop');
  const stopAction = actions.length - 1;
  const evalCache = new Map<number, FitEvaluation>();
  const evaluate = (mask: number): FitEvaluation => {
    let cached = evalCache.get(mask);
    if (!cached) {
      cached = evaluateFeatureMask(maskToIndices(mask, numFeatures), scenario, split.trainRows, split.validationRows);
      evalCache.set(mask, cached);
    }
    return cached;
  };
  const transitionTo = (mask: number, action: number): number => {
    if (action === stopAction) return mask;
    if (countBits(mask) >= scenario.params.mdpBudget) return mask;
    if ((mask & (1 << action)) !== 0) return mask;
    return mask | (1 << action);
  };
  const rewardOf = (mask: number, action: number): number => {
    if (action === stopAction) return 0;
    const next = transitionTo(mask, action);
    if (next === mask) return -5;
    const before = evaluate(mask);
    const after = evaluate(next);
    const featureCost = scenario.featureCandidates[action].cost * 0.55;
    const overfitPenalty = 0.05 * Math.max(0, after.validationMse - after.trainMse);
    return before.validationMse - after.validationMse - featureCost - overfitPenalty;
  };
  const mdpSpec: POMDPSpec<number, string, string> = {
    states,
    actions,
    observations: ['none'],
    transition: (sIdx, aIdx) => {
      const row = new Array<number>(states.length).fill(0);
      row[transitionTo(states[sIdx], aIdx)] = 1;
      return row;
    },
    observation: (_sNextIdx, _aIdx) => [1],
    reward: (sIdx, aIdx) => rewardOf(states[sIdx], aIdx),
    discount: 0.92,
  };
  const vi = mdpValueIteration(mdpSpec, {tol: 1e-7, maxIter: 250});
  let mask = 0;
  const actionTrace: MDPDiscoveryStep[] = [];
  for (let step = 0; step < scenario.params.mdpBudget + 2; step++) {
    const action = vi.policy[mask];
    const next = transitionTo(mask, action);
    const before = evaluate(mask);
    const after = evaluate(next);
    actionTrace.push({
      step,
      stateMask: mask,
      action: actions[action],
      reward: rewardOf(mask, action),
      validationMseBefore: before.validationMse,
      validationMseAfter: after.validationMse,
      selectedAfter: maskToIndices(next, numFeatures).map(i => scenario.featureCandidates[i].id),
    });
    if (action === stopAction || next === mask) break;
    mask = next;
  }
  const selected = maskToIndices(mask, numFeatures);
  const finalEval = evaluate(mask);
  const baseline = evaluate(0);
  return {
    selectedFeatureIndices: selected,
    selectedVariables: selected.map(i => {
      const f = scenario.featureCandidates[i];
      return {id: f.id, label: f.label, source: f.source, cost: f.cost};
    }),
    rows,
    trainRows: split.trainRows,
    validationRows: split.validationRows,
    baselineValidationMse: baseline.validationMse,
    validationMse: finalEval.validationMse,
    trainMse: finalEval.trainMse,
    mdpStates: states.length,
    mdpActions: actions.length,
    mdpIterations: vi.iterations,
    mdpFinalDelta: vi.finalDelta,
    actionTrace,
  };
}

function buildForecastRows(scenario: ForecastScenario, beliefTrace: LatentBeliefTrace): ForecastRow[] {
  const rows: ForecastRow[] = [];
  const trainN = scenario.params.trainingPeriods;
  const validationCount = Math.max(2, Math.floor((trainN - 2) * scenario.params.validationShare));
  const validationStart = trainN - validationCount;
  for (let t = 2; t < trainN; t++) {
    rows.push({
      t,
      target: scenario.observations[t].y,
      context: featureContextForTraining(scenario, beliefTrace, t),
      split: t >= validationStart ? 'validation' : 'train',
    });
  }
  return rows;
}

function featureContextForTraining(scenario: ForecastScenario, beliefTrace: LatentBeliefTrace, t: number): FeatureContext {
  const cur = scenario.observations[t];
  const lag = scenario.observations[t - 1].y;
  const prev = scenario.observations[t - 2].y;
  const point = beliefTrace.points[t] ?? beliefTrace.points[beliefTrace.points.length - 1];
  return featureContext(cur, lag, prev, point.prior, scenario.params.trainingPeriods);
}

function featureContext(
  obs: ForecastObservation,
  lagY: number,
  prevY: number,
  belief: readonly number[],
  trainingPeriods: number,
): FeatureContext {
  return {
    t: obs.t,
    demand: obs.demand,
    supplyGap: 1.12 - obs.supply,
    price: obs.price,
    lagY,
    momentum: lagY - prevY,
    trend: obs.t / Math.max(1, trainingPeriods - 1),
    beliefBaseline: belief[0],
    beliefExpansion: belief[1],
    beliefContraction: belief[2],
    beliefShock: belief[3],
  };
}

function splitRows(rows: ForecastRow[], validationShare: number): {trainRows: ForecastRow[]; validationRows: ForecastRow[]} {
  const validationCount = Math.max(2, Math.floor(rows.length * validationShare));
  return {
    trainRows: rows.slice(0, rows.length - validationCount),
    validationRows: rows.slice(rows.length - validationCount),
  };
}

interface FitEvaluation {
  trainMse: number;
  validationMse: number;
}

function evaluateFeatureMask(
  featureIndices: readonly number[],
  scenario: ForecastScenario,
  trainRows: readonly ForecastRow[],
  validationRows: readonly ForecastRow[],
): FitEvaluation {
  const fit = ridgeFit(featureIndices, scenario.featureCandidates, trainRows, scenario.params.ridge);
  return {
    trainMse: predictionMse(fit, scenario.featureCandidates, trainRows),
    validationMse: predictionMse(fit, scenario.featureCandidates, validationRows),
  };
}

function fineTuneEquation(scenario: ForecastScenario, discovery: VariableDiscoveryResult): TunedEquation {
  const allRows = discovery.rows;
  const target = ridgeFit(discovery.selectedFeatureIndices, scenario.featureCandidates, allRows, scenario.params.ridge);
  const startCoefficients = [mean(allRows.map(row => row.target)), ...new Array(discovery.selectedFeatureIndices.length).fill(0)];
  const trace: FineTuneTraceRow[] = [];
  for (let iter = 0; iter <= scenario.params.fineTuneIterations; iter++) {
    const alpha = iter === scenario.params.fineTuneIterations ? 1 : 1 - Math.exp(-0.32 * iter);
    const coeffs = target.coefficients.map((v, i) => startCoefficients[i] + alpha * (v - startCoefficients[i]));
    const iterFit: RidgeFit = {...target, coefficients: coeffs};
    trace.push({
      iter,
      mse: predictionMse(iterFit, scenario.featureCandidates, allRows),
      validationMse: predictionMse(iterFit, scenario.featureCandidates, discovery.validationRows),
      coefficients: coeffs.slice(),
    });
  }
  const fitted = allRows.map(row => ({
    t: row.t,
    actual: row.target,
    predicted: predictWithFit(target, scenario.featureCandidates, row.context),
    split: row.split,
  }));
  return {
    featureIndices: discovery.selectedFeatureIndices.slice(),
    featureIds: discovery.selectedVariables.map(v => v.id),
    featureLabels: discovery.selectedVariables.map(v => v.label),
    coefficients: target.coefficients.slice(1),
    means: target.means.slice(),
    scales: target.scales.slice(),
    intercept: target.coefficients[0],
    equationText: equationText(discovery.selectedVariables.map(v => v.id), target.coefficients),
    inSampleMse: predictionMse(target, scenario.featureCandidates, allRows),
    validationMse: predictionMse(target, scenario.featureCandidates, discovery.validationRows),
    trace,
    fitted,
  };
}

interface RidgeFit {
  featureIndices: number[];
  coefficients: number[];
  means: number[];
  scales: number[];
}

function ridgeFit(
  featureIndices: readonly number[],
  candidates: readonly FeatureCandidate[],
  rows: readonly ForecastRow[],
  ridge: number,
): RidgeFit {
  const p = featureIndices.length;
  const raw = rows.map(row => featureIndices.map(i => candidates[i].compute(row.context)));
  const means = new Array<number>(p).fill(0);
  const scales = new Array<number>(p).fill(1);
  for (let j = 0; j < p; j++) {
    means[j] = mean(raw.map(row => row[j]));
    const variance = mean(raw.map(row => Math.pow(row[j] - means[j], 2)));
    scales[j] = Math.sqrt(Math.max(variance, 1e-10));
  }
  const dim = p + 1;
  const A = Array.from({length: dim}, () => new Array<number>(dim).fill(0));
  const b = new Array<number>(dim).fill(0);
  for (let r = 0; r < rows.length; r++) {
    const x = [1, ...raw[r].map((v, j) => (v - means[j]) / scales[j])];
    for (let i = 0; i < dim; i++) {
      b[i] += x[i] * rows[r].target;
      for (let j = 0; j < dim; j++) A[i][j] += x[i] * x[j];
    }
  }
  for (let i = 1; i < dim; i++) A[i][i] += ridge;
  return {
    featureIndices: featureIndices.slice(),
    coefficients: solveLinearSystem(A, b),
    means,
    scales,
  };
}

function predictionMse(
  fit: RidgeFit,
  candidates: readonly FeatureCandidate[],
  rows: readonly ForecastRow[],
): number {
  return mse(rows.map(row => row.target), rows.map(row => predictWithFit(fit, candidates, row.context)));
}

function predictWithFit(fit: RidgeFit, candidates: readonly FeatureCandidate[], context: FeatureContext): number {
  let y = fit.coefficients[0];
  for (let j = 0; j < fit.featureIndices.length; j++) {
    const raw = candidates[fit.featureIndices[j]].compute(context);
    y += fit.coefficients[j + 1] * (raw - fit.means[j]) / fit.scales[j];
  }
  return y;
}

function projectForecast(
  scenario: ForecastScenario,
  beliefTrace: LatentBeliefTrace,
  equation: TunedEquation,
): ForecastProjectionPoint[] {
  const featureFit: RidgeFit = {
    featureIndices: equation.featureIndices,
    coefficients: [equation.intercept, ...equation.coefficients],
    means: equation.means,
    scales: equation.scales,
  };
  const out: ForecastProjectionPoint[] = [];
  let belief = new DiscreteBelief(REGIMES, beliefTrace.finalBelief);
  let lagY = scenario.observations[scenario.params.trainingPeriods - 1].y;
  let prevY = scenario.observations[scenario.params.trainingPeriods - 2].y;
  const residualScale = Math.sqrt(Math.max(equation.inSampleMse, 1e-9));
  for (let h = 1; h <= scenario.params.forecastHorizon; h++) {
    belief.propagate((_state, index) => scenario.pomdpSpec.transition(index, 0));
    const t = scenario.params.trainingPeriods + h - 1;
    const obs = scenario.observations[t];
    const ctx = featureContext(obs, lagY, prevY, belief.asArray(), scenario.params.trainingPeriods);
    const forecast = predictWithFit(featureFit, scenario.featureCandidates, ctx);
    const band = residualScale * (1.1 + 0.07 * h + 0.22 * belief.entropy());
    out.push({
      t,
      horizonStep: h,
      forecast,
      actual: obs.y,
      lower: forecast - 1.96 * band,
      upper: forecast + 1.96 * band,
      beliefMode: belief.mode(),
      beliefEntropy: belief.entropy(),
    });
    prevY = lagY;
    lagY = forecast;
  }
  return out;
}

function maskToIndices(mask: number, numFeatures: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < numFeatures; i++) if ((mask & (1 << i)) !== 0) out.push(i);
  return out;
}

function countBits(x: number): number {
  let n = 0;
  let v = x;
  while (v > 0) {
    n += v & 1;
    v = v >> 1;
  }
  return n;
}

function mean(xs: readonly number[]): number {
  return xs.reduce((sum, x) => sum + x, 0) / Math.max(1, xs.length);
}

function mse(actual: readonly number[], predicted: readonly number[]): number {
  if (actual.length !== predicted.length) throw new Error('mse: length mismatch');
  if (actual.length === 0) return 0;
  let total = 0;
  for (let i = 0; i < actual.length; i++) total += Math.pow(actual[i] - predicted[i], 2);
  return total / actual.length;
}

function entropy(weights: readonly number[]): number {
  let h = 0;
  for (const w of weights) if (w > 0) h -= w * Math.log(w);
  return h;
}

function solveLinearSystem(A: number[][], b: readonly number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => row.slice().concat(b[i]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-10) {
      M[col][col] += 1e-8;
      pivot = col;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map(row => row[n]);
}

function equationText(featureIds: readonly string[], coefficients: readonly number[]): string {
  const terms = [`${coefficients[0].toFixed(3)}`];
  for (let i = 0; i < featureIds.length; i++) {
    terms.push(`${coefficients[i + 1] >= 0 ? '+' : '-'} ${Math.abs(coefficients[i + 1]).toFixed(3)}*z(${featureIds[i]})`);
  }
  return `y_hat = ${terms.join(' ')}`;
}
