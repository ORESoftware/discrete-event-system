#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_factmachine_markets.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-factmachine-markets.rs   (fn main)
// 1:1 file move. Platform-level multi-market scheduler above the single-market
// factmachine model (which markets to open, when, how, for how long).
//
// Conversion notes (file-specific):
//   - scheduler policy union (fixed-daily/greedy-buzz/mdp-oracle/pomdp-belief)
//     -> enum.
//   - the operator action is multi-field (schedule/contract/duration/pricing/
//     rewards/verification) -> a struct of enums.
//   - Math.random (buzz/votes/belief) -> inject RandomSource/SeededRandom.
//   - LARGE file: many station classes -> struct + impl trait; top-level run
//     -> fn main.
// =============================================================================

// =============================================================================
// FactMachine multi-market simulator.
//
// This is a platform-level layer above the existing single-market
// `main-factmachine.ts` POMDP/LMSR work. The single-market model asks:
//   "how should a bettor act inside one opinion market?"
//
// This model asks:
//   "which opinion markets should the platform open, when, for how long,
//    and should they be binary or scalar?"
//
// The scheduler policies compared here:
//   fixed-daily  — one 24h binary market at a time.
//   greedy-buzz  — open high-observed-buzz topics with a simple heuristic.
//   mdp-oracle   — MDP value iteration with full latent topic state.
//   pomdp-belief — QMDP-style policy over a belief state inferred from noisy
//                  buzz/ambiguity observations and updated after resolution.
//
// The operator action is intentionally richer than just "open/close":
//   1. schedule: wait vs open now
//   2. contract: majority binary, scalar vote-distribution, or over/under
//   3. duration: 15m, 1h, 6h, 24h
//   4. pricing: trading fee + LMSR liquidity depth
//   5. voter rewards: point-payout intensity
//   6. verification: open/basic/proof-of-personhood friction
//   7. information design: price-only, delayed votes, live votes, demographic
//      slices, or momentum indicators
//
// In every market, voting and betting run together. A user must first cast a
// vote + predicted distribution before they can bet.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {mulberry32} from './general/prng';
import {TimeSteppedStation} from './general/time-stepped-station';
import {MDPSpec, valueIteration, qValue} from './general/value-iteration';
import {LMSR} from './main-factmachine';

export type SchedulerPolicy = 'fixed-daily' | 'greedy-buzz' | 'mdp-oracle' | 'pomdp-belief';
export type MarketKind = 'binary' | 'scalar' | 'threshold';
type Category = 'politics' | 'culture' | 'sports' | 'conspiracy' | 'breaking';
type VerificationTier = 'open' | 'basic' | 'proof';
type InformationMode = 'price-only' | 'delayed-votes' | 'live-votes' | 'demographic-slices' | 'momentum-signals';

interface CandidateTopic {
  id: number;
  category: Category;
  createdAt: number;
  expiresAt: number;
  trueHotness: number;
  ambiguity: number;
  trueTheta: number;
  manipulationRisk: number;
  newsCycleIntensity: number;
  socialVirality: number;
  influencerActivity: number;
  memeMomentum: number;
  demographicPolarization: number;
  opinionEventCoupling: number;
  eventProbability: number;
  turnoutSkew: number;
  botPressure: number;
  referralElasticity: number;
  observedBuzz: number;
  observedAmbiguity: number;
}

interface SchedulerAction {
  label: string;
  kind: MarketKind | 'wait';
  durationH: number;
  feeRate: number;
  liquidityMultiplier: number;
  rewardMultiplier: number;
  verification: VerificationTier;
  informationMode: InformationMode;
  timingDecay: number;
  threshold?: number;
  description: string;
}

interface OpenMarket {
  id: number;
  topic: CandidateTopic;
  action: SchedulerAction;
  openAt: number;
  closeAt: number;
}

export interface ClosedMarket {
  id: number;
  topic: CandidateTopic;
  kind: MarketKind;
  contractLabel: string;
  durationH: number;
  openAt: number;
  closeAt: number;
  feeRate: number;
  liquidity: number;
  rewardMultiplier: number;
  verification: VerificationTier;
  informationMode: InformationMode;
  timingDecay: number;
  threshold?: number;
  finalVoteFraction: number;
  outcomeIndex: number;
  votes: number;
  suspectedSybilVotes: number;
  avgVoteTimeFraction: number;
  avgTimingMultiplier: number;
  bettors: number;
  trades: number;
  buyVolume: number;
  sellVolume: number;
  feeRevenue: number;
  voterPoints: number;
  raffleEntries: number;
  avgPredictionError: number;
  opinionSamplingError: number;
  predictionBrierScore: number;
  externalOutcome: number;
  avgTraderBeliefError: number;
  traderBeliefEntropy: number;
  herdingIndex: number;
  priceOpinionGap: number;
  marketMakerRiskBound: number;
  fraudPressure: number;
  referralAdds: number;
  churnRisk: number;
  rewardInflationPressure: number;
  liquidityUtilization: number;
  whaleTradeShare: number;
  traderPnl: number;
  lmsrLoss: number;
}

export interface PortfolioConfig {
  scenarioLabel: string;
  horizonH: number;
  stepH: number;
  maxConcurrent: number;
  minDailyMarkets: number;
  maxDailyMarkets: number;
  dailyMarketCaps: number[];
  seed: number;
  liquidity: number;
  feeRate: number;
  scalarBins: number;
  minMarketParticipants: number;
}

export interface MarketKindAggregate {
  kind: MarketKind;
  markets: number;
  votes: number;
  bettors: number;
  trades: number;
  buyVolume: number;
  sellVolume: number;
  feeRevenue: number;
  voterPoints: number;
  suspectedSybilVotes: number;
  avgDurationH: number;
  avgLiquidity: number;
  avgFeeRate: number;
  avgPredictionError: number;
  avgOpinionSamplingError: number;
  avgPredictionBrierScore: number;
  avgTraderBeliefError: number;
  traderBeliefEntropy: number;
  herdingIndex: number;
  priceOpinionGap: number;
  fraudPressure: number;
  liquidityUtilization: number;
  whaleTradeShare: number;
  platformSurplus: number;
}

export interface DailySummary {
  day: number;
  marketCap: number;
  opened: number;
  closed: number;
  activeEnd: number;
  queuedEnd: number;
  votes: number;
  bettors: number;
  trades: number;
  feeRevenue: number;
  voterPoints: number;
  binaryClosed: number;
  scalarClosed: number;
  thresholdClosed: number;
  avgPredictionError: number;
  avgOpinionSamplingError: number;
  avgPredictionBrierScore: number;
  fraudPressure: number;
  herdingIndex: number;
}

export interface PolicyAggregate {
  scenarioLabel: string;
  minMarketParticipants: number;
  policy: SchedulerPolicy;
  marketsOpened: number;
  marketsClosed: number;
  binaryMarkets: number;
  scalarMarkets: number;
  thresholdMarkets: number;
  avgDurationH: number;
  avgFeeRate: number;
  avgLiquidity: number;
  avgRewardMultiplier: number;
  proofMarkets: number;
  avgTimingDecay: number;
  votes: number;
  suspectedSybilVotes: number;
  avgVoteTimeFraction: number;
  avgTimingMultiplier: number;
  bettors: number;
  trades: number;
  buyVolume: number;
  sellVolume: number;
  feeRevenue: number;
  voterPoints: number;
  raffleEntries: number;
  avgPredictionError: number;
  avgOpinionSamplingError: number;
  avgPredictionBrierScore: number;
  avgTraderBeliefError: number;
  traderBeliefEntropy: number;
  herdingIndex: number;
  priceOpinionGap: number;
  marketMakerRiskBound: number;
  fraudPressure: number;
  referralAdds: number;
  churnRisk: number;
  rewardInflationPressure: number;
  liquidityUtilization: number;
  whaleTradeShare: number;
  avgNewsCycleIntensity: number;
  avgSocialVirality: number;
  avgInfluencerActivity: number;
  avgDemographicPolarization: number;
  traderPnl: number;
  lmsrLoss: number;
  platformSurplus: number;
  engagementScore: number;
  avgBeliefEntropy?: number;
  avgBeliefError?: number;
}

export interface PolicyRun {
  scenarioLabel: string;
  minMarketParticipants: number;
  policy: SchedulerPolicy;
  aggregate: PolicyAggregate;
  kindBreakdown: MarketKindAggregate[];
  daily: DailySummary[];
  closedMarkets: ClosedMarket[];
  actionCounts: Array<{action: string; count: number}>;
  timeline: Array<{
    t: number;
    day: number;
    open: number;
    closed: number;
    queued: number;
    votes: number;
    bettors: number;
    trades: number;
    fees: number;
    marketCap: number;
    openedToday: number;
    openedTotal: number;
  }>;
  beliefTrace?: Array<{t: number; entropy: number; expectedHotness: number; error: number}>;
}

export interface OperatorMDP {
  spec: MDPSpec;
  V: Float64Array;
  policy: Int32Array;
  q: number[][];
  actions: SchedulerAction[];
  iterations: number;
  finalDelta: number;
  gamma: number;
  stateLabel: (s: number) => string;
}

class FactMachinePortfolioStation extends TimeSteppedStation {
  readonly pending: CandidateTopic[] = [];
  readonly active: OpenMarket[] = [];
  readonly closed: ClosedMarket[] = [];
  readonly actionCounts = new Map<string, number>();
  readonly timeline: PolicyRun['timeline'] = [];
  readonly beliefTrace: NonNullable<PolicyRun['beliefTrace']> = [];
  private nextTopicId = 0;
  private nextMarketId = 0;
  private fixedNextOpen = 0;
  private votesSoFar = 0;
  private bettorsSoFar = 0;
  private tradesSoFar = 0;
  private feesSoFar = 0;
  private openedTotal = 0;
  private readonly openedByDay = new Map<number, number>();
  private readonly categoryBelief = new CategoryBelief();

  constructor(
    readonly config: PortfolioConfig,
    readonly scheduler: SchedulerPolicy,
    readonly mdp: OperatorMDP,
    readonly rng: () => number,
  ) {
    super(`factmachine-${scheduler}`);
  }

  runTimeStep(_stepSize: number, tick: number): void {
    const now = tick * this.config.stepH;
    const acceptingNewMarkets = now < this.config.horizonH - 1e-9;
    if (acceptingNewMarkets) this.emitCandidateTopics(now);
    this.closeMarkets(now);
    this.expireCandidates(now);
    if (acceptingNewMarkets) this.openMarkets(now);
    const day = dayIndex(now);
    this.timeline.push({
      t: now,
      day,
      open: this.active.length,
      closed: this.closed.length,
      queued: this.pending.length,
      votes: this.votesSoFar,
      bettors: this.bettorsSoFar,
      trades: this.tradesSoFar,
      fees: this.feesSoFar,
      marketCap: dailyMarketCapForDay(day, this.config),
      openedToday: this.openedByDay.get(day) ?? 0,
      openedTotal: this.openedTotal,
    });
  }

  toRun(): PolicyRun {
    const aggregate = aggregateRun(this.scheduler, this.closed, this.config);
    const beliefTrace = this.scheduler === 'pomdp-belief' ? this.beliefTrace.slice() : undefined;
    if (beliefTrace && beliefTrace.length > 0) {
      aggregate.avgBeliefEntropy = mean(beliefTrace.map(x => x.entropy));
      aggregate.avgBeliefError = mean(beliefTrace.map(x => x.error));
    }
    return {
      scenarioLabel: this.config.scenarioLabel,
      minMarketParticipants: this.config.minMarketParticipants,
      policy: this.scheduler,
      aggregate,
      kindBreakdown: aggregateByKind(this.closed),
      daily: buildDailySummaries(this.closed, this.timeline, this.config),
      closedMarkets: this.closed.slice(),
      actionCounts: Array.from(this.actionCounts.entries())
        .map(([action, count]) => ({action, count}))
        .sort((a, b) => b.count - a.count),
      timeline: this.timeline.slice(),
      beliefTrace,
    };
  }

  private emitCandidateTopics(now: number): void {
    const baseRatePerHour = 1.35;
    const lambda = baseRatePerHour * this.config.stepH;
    const n = samplePoisson(lambda, this.rng);
    for (let i = 0; i < n; i++) {
      const topic = sampleTopic(this.nextTopicId++, now, this.rng);
      this.pending.push(topic);
    }
    this.pending.sort((a, b) => b.observedBuzz - a.observedBuzz);
  }

  private closeMarkets(now: number): void {
    for (let i = this.active.length - 1; i >= 0; i--) {
      const market = this.active[i];
      if (market.closeAt > now + 1e-9) continue;
      this.active.splice(i, 1);
      const closed = simulateMarket(market, this.config, this.rng);
      this.closed.push(closed);
      this.votesSoFar += closed.votes;
      this.bettorsSoFar += closed.bettors;
      this.tradesSoFar += closed.trades;
      this.feesSoFar += closed.feeRevenue;
      this.categoryBelief.observeMarket(closed);
    }
  }

  private expireCandidates(now: number): void {
    for (let i = this.pending.length - 1; i >= 0; i--) {
      if (this.pending[i].expiresAt < now) this.pending.splice(i, 1);
    }
  }

  private openMarkets(now: number): void {
    const day = dayIndex(now);
    const dayCap = dailyMarketCapForDay(day, this.config);
    while (
      this.active.length < this.config.maxConcurrent
      && (this.openedByDay.get(day) ?? 0) < dayCap
      && this.pending.length > 0
    ) {
      const candidate = this.chooseCandidate(now);
      if (!candidate) return;
      const action = this.chooseAction(candidate, now);
      bump(this.actionCounts, action.label);
      if (action.kind === 'wait') return;
      this.pending.splice(this.pending.indexOf(candidate), 1);
      this.active.push({
        id: this.nextMarketId++,
        topic: candidate,
        action,
        openAt: now,
        closeAt: now + action.durationH,
      });
      this.openedByDay.set(day, (this.openedByDay.get(day) ?? 0) + 1);
      this.openedTotal++;
      if (this.scheduler === 'fixed-daily') this.fixedNextOpen = now + 24;
    }
  }

  private chooseCandidate(now: number): CandidateTopic | null {
    if (this.scheduler === 'fixed-daily' && now + 1e-9 < this.fixedNextOpen) return null;
    if (this.scheduler === 'fixed-daily' && this.active.length > 0) return null;
    return this.pending[0] ?? null;
  }

  private chooseAction(topic: CandidateTopic, now: number): SchedulerAction {
    const fatigueBin = fatigueBinFor(this.active.length, this.config.maxConcurrent);
  if (this.scheduler === 'fixed-daily') {
      return actionBy(this.mdp.actions, 'binary', 24, 'baseline');
    }
    if (this.scheduler === 'greedy-buzz') {
      if (topic.observedBuzz < 0.38 && this.active.length > 0) return this.mdp.actions[0];
      if (topic.observedAmbiguity > 0.72) {
        const durationH = topic.observedBuzz > 0.78 ? 1 : 6;
        return actionBy(this.mdp.actions, 'scalar', durationH, topic.observedBuzz > 0.78 ? 'growth' : 'deep');
      }
      if (topic.observedBuzz > 0.74 && topic.observedAmbiguity > 0.42) {
        return actionBy(this.mdp.actions, 'threshold', 1, 'over55');
      }
      const durationH = topic.observedBuzz > 0.76 ? 1 : topic.observedBuzz > 0.55 ? 6 : 24;
      return actionBy(this.mdp.actions, 'binary', durationH, durationH === 24 ? 'baseline' : 'growth');
    }
    if (this.scheduler === 'mdp-oracle') {
      const s = encodeOperatorState(
        bin3(topic.trueHotness),
        bin3(topic.ambiguity),
        fatigueBin,
      );
      return this.mdp.actions[this.mdp.policy[s]];
    }

    const belief = this.categoryBelief.beliefFor(topic);
    const qByAction = new Array(this.mdp.actions.length).fill(0);
    let expectedHotness = 0;
    let error = 0;
    for (const b of belief) {
      const s = encodeOperatorState(b.hotBin, b.ambBin, fatigueBin);
      expectedHotness += b.prob * hotnessMidpoint(b.hotBin);
      error += b.prob * Math.abs(hotnessMidpoint(b.hotBin) - topic.trueHotness);
      for (let a = 0; a < this.mdp.actions.length; a++) qByAction[a] += b.prob * this.mdp.q[s][a];
    }
    const best = argMax(qByAction);
    this.beliefTrace.push({
      t: now,
      entropy: entropy(belief.map(b => b.prob)),
      expectedHotness,
      error,
    });
    return this.mdp.actions[best];
  }
}

class CategoryBelief {
  private readonly beta: Record<Category, {a: number; b: number}> = {
    politics: {a: 2.0, b: 1.8},
    culture: {a: 1.6, b: 2.0},
    sports: {a: 1.5, b: 2.1},
    conspiracy: {a: 1.8, b: 2.0},
    breaking: {a: 2.2, b: 1.4},
  };

  beliefFor(topic: CandidateTopic): Array<{hotBin: number; ambBin: number; prob: number}> {
    const priorMean = this.beta[topic.category].a / (this.beta[topic.category].a + this.beta[topic.category].b);
    const hotScores = [0, 1, 2].map(h => {
      const mid = hotnessMidpoint(h);
      const obs = -Math.pow(topic.observedBuzz - mid, 2) / 0.045;
      const prior = -Math.pow(priorMean - mid, 2) / 0.18;
      return obs + prior;
    });
    const ambScores = [0, 1, 2].map(a => -Math.pow(topic.observedAmbiguity - ambiguityMidpoint(a), 2) / 0.055);
    const hotP = softmax(hotScores);
    const ambP = softmax(ambScores);
    const out: Array<{hotBin: number; ambBin: number; prob: number}> = [];
    for (let h = 0; h < 3; h++) {
      for (let a = 0; a < 3; a++) out.push({hotBin: h, ambBin: a, prob: hotP[h] * ambP[a]});
    }
    return out;
  }

  observeMarket(market: ClosedMarket): void {
    const success = market.votes >= 180 || market.feeRevenue >= 45 || market.trades >= 90;
    const b = this.beta[market.topic.category];
    b.a += success ? 1 : 0.25;
    b.b += success ? 0.25 : 1;
  }
}

const OPERATOR_ACTIONS: SchedulerAction[] = [
  waitAction(),
  marketAction('binary-baseline-24h', 'binary', 24, {
    feeRate: 0.01,
    liquidityMultiplier: 1.0,
    rewardMultiplier: 1.0,
    verification: 'basic',
    informationMode: 'delayed-votes',
    timingDecay: 1.0,
    description: 'baseline majority market',
  }),
  marketAction('binary-growth-15m', 'binary', 0.25, {
    feeRate: 0.005,
    liquidityMultiplier: 1.35,
    rewardMultiplier: 1.35,
    verification: 'open',
    informationMode: 'momentum-signals',
    timingDecay: 1.45,
    description: 'fast low-fee launch for hot topics',
  }),
  marketAction('binary-growth-1h', 'binary', 1, {
    feeRate: 0.005,
    liquidityMultiplier: 1.25,
    rewardMultiplier: 1.25,
    verification: 'open',
    informationMode: 'live-votes',
    timingDecay: 1.35,
    description: 'one-hour majority market optimized for participation',
  }),
  marketAction('binary-surplus-6h', 'binary', 6, {
    feeRate: 0.02,
    liquidityMultiplier: 0.75,
    rewardMultiplier: 0.85,
    verification: 'basic',
    informationMode: 'price-only',
    timingDecay: 0.85,
    description: 'higher-fee majority market optimized for margin',
  }),
  marketAction('binary-proof-24h', 'binary', 24, {
    feeRate: 0.01,
    liquidityMultiplier: 1.15,
    rewardMultiplier: 1.05,
    verification: 'proof',
    informationMode: 'demographic-slices',
    timingDecay: 1.05,
    description: 'longer majority market with proof-of-personhood trust',
  }),
  marketAction('scalar-growth-1h', 'scalar', 1, {
    feeRate: 0.005,
    liquidityMultiplier: 1.35,
    rewardMultiplier: 1.25,
    verification: 'open',
    informationMode: 'live-votes',
    timingDecay: 1.25,
    description: 'distribution market for ambiguous fast-moving topics',
  }),
  marketAction('scalar-deep-6h', 'scalar', 6, {
    feeRate: 0.01,
    liquidityMultiplier: 1.55,
    rewardMultiplier: 1.1,
    verification: 'basic',
    informationMode: 'demographic-slices',
    timingDecay: 1.0,
    description: 'deeper-liquidity distribution market',
  }),
  marketAction('scalar-proof-24h', 'scalar', 24, {
    feeRate: 0.01,
    liquidityMultiplier: 1.4,
    rewardMultiplier: 1.0,
    verification: 'proof',
    informationMode: 'demographic-slices',
    timingDecay: 0.95,
    description: 'long-form scalar sentiment read with verified voting',
  }),
  marketAction('over55-growth-1h', 'threshold', 1, {
    feeRate: 0.005,
    liquidityMultiplier: 1.25,
    rewardMultiplier: 1.2,
    verification: 'open',
    informationMode: 'momentum-signals',
    timingDecay: 1.35,
    threshold: 0.55,
    description: 'over/under 55% agree, optimized for rapid debate',
  }),
  marketAction('over60-surplus-6h', 'threshold', 6, {
    feeRate: 0.02,
    liquidityMultiplier: 0.9,
    rewardMultiplier: 0.9,
    verification: 'basic',
    informationMode: 'price-only',
    timingDecay: 0.85,
    threshold: 0.60,
    description: 'over/under 60% agree, optimized for fee capture',
  }),
  marketAction('over55-proof-24h', 'threshold', 24, {
    feeRate: 0.01,
    liquidityMultiplier: 1.25,
    rewardMultiplier: 1.05,
    verification: 'proof',
    informationMode: 'delayed-votes',
    timingDecay: 1.1,
    threshold: 0.55,
    description: 'verified over/under sentiment threshold',
  }),
];

export function buildOperatorMDP(): OperatorMDP {
  const gamma = 0.88;
  const spec: MDPSpec = {
    numStates: 27,
    numActions: () => OPERATOR_ACTIONS.length,
    outcomes: (s, a) => operatorOutcomes(s, OPERATOR_ACTIONS[a]),
    stateLabel: operatorStateLabel,
    actionLabel: a => OPERATOR_ACTIONS[a].label,
  };
  const vi = valueIteration(spec, {gamma, tol: 1e-8, maxIter: 10000, randomTieBreak: false});
  const q: number[][] = [];
  for (let s = 0; s < spec.numStates; s++) {
    const row: number[] = [];
    for (let a = 0; a < OPERATOR_ACTIONS.length; a++) row.push(qValue(spec, vi.V, s, a, gamma));
    q.push(row);
  }
  return {
    spec,
    V: vi.V,
    policy: vi.policy,
    q,
    actions: OPERATOR_ACTIONS,
    iterations: vi.iterations,
    finalDelta: vi.finalDelta,
    gamma,
    stateLabel: operatorStateLabel,
  };
}

function operatorOutcomes(s: number, action: SchedulerAction): Array<{prob: number; reward: number; nextState: number}> {
  const {hotBin, ambBin, fatigueBin} = decodeOperatorState(s);
  if (action.kind === 'wait') {
    const nextFatigue = Math.max(0, fatigueBin - 1);
    return [{prob: 1, reward: -1.5 + hotBin * 0.2, nextState: encodeOperatorState(hotBin, ambBin, nextFatigue)}];
  }
  const reward = expectedMarketUtility(hotBin, ambBin, fatigueBin, action);
  const fatigueUp = Math.min(2, fatigueBin + (action.durationH >= 6 ? 1 : 0));
  const fatigueSame = fatigueBin;
  const fatigueDown = Math.max(0, fatigueBin - 1);
  return [
    {prob: 0.58, reward, nextState: encodeOperatorState(hotBin, ambBin, fatigueUp)},
    {prob: 0.28, reward: reward * 0.88, nextState: encodeOperatorState(Math.max(0, hotBin - 1), ambBin, fatigueSame)},
    {prob: 0.14, reward: reward * 1.12, nextState: encodeOperatorState(Math.min(2, hotBin + 1), ambBin, fatigueDown)},
  ];
}

function expectedMarketUtility(hotBin: number, ambBin: number, fatigueBin: number, action: SchedulerAction): number {
  const hot = hotnessMidpoint(hotBin);
  const amb = ambiguityMidpoint(ambBin);
  const reach = 1 - Math.exp(-action.durationH / 5.5);
  const urgency = action.durationH <= 1 ? 1.22 : action.durationH <= 6 ? 1.06 : 0.92;
  const fatiguePenalty = 1 - fatigueBin * 0.16;
  const kindFit = action.kind === 'scalar'
    ? (0.70 + amb * 0.95)
    : action.kind === 'threshold'
      ? (0.82 + amb * 0.18 + hot * 0.12)
      : (1.34 - amb * 0.42 + hot * 0.08);
  const feeDrag = clamp(1.14 - action.feeRate * 18, 0.66, 1.08);
  const liquidityBoost = 0.86 + action.liquidityMultiplier * 0.14;
  const rewardBoost = 0.82 + action.rewardMultiplier * 0.25;
  const verificationParticipation = verificationParticipationMultiplier(action.verification);
  const verificationTrust = verificationTrustMultiplier(action.verification);
  const informationEngagement = informationEngagementMultiplier(action.informationMode);
  const informationTrust = informationTrustMultiplier(action.informationMode);
  const timingUrgency = clamp(0.86 + action.timingDecay * 0.18, 0.78, 1.24);
  const votes = 45 * hot * (0.45 + 2.4 * reach) * urgency * fatiguePenalty * kindFit
    * rewardBoost * verificationParticipation * informationEngagement * timingUrgency;
  const traders = votes * (0.18 + 0.24 * hot + 0.08 * amb) * marketTraderFit(action.kind as MarketKind) * feeDrag * liquidityBoost;
  const avgTradeSize = (7 + 9 * hot) * (0.86 + action.liquidityMultiplier * 0.18);
  const fees = traders * avgTradeSize * action.feeRate;
  const integrityPenalty = Math.pow(hot * amb, 1.2)
    * (action.durationH >= 24 ? 10 : 4)
    * manipulationMultiplier(action.verification);
  const rewardCost = votes * (0.06 + action.rewardMultiplier * 0.08);
  const thresholdPenalty = action.kind === 'threshold'
    ? 3.5 + Math.max(0, (action.threshold ?? 0.55) - 0.55) * (1 - amb) * 8
    : 0;
  const scalarResolutionBonus = action.kind === 'scalar' ? 8.5 * amb * (0.45 + hot) : 0;
  const binaryClarityBonus = action.kind === 'binary' ? 7 * (1 - amb) * (0.55 + hot) : 0;
  const thresholdHeadlineBonus = action.kind === 'threshold' ? 4.2 * hot * (0.55 + amb) : 0;
  const cascadePenalty = Math.pow(hot * amb, 1.1) * informationHerdingMultiplier(action.informationMode) * 6;
  return votes * 0.28 + traders * 0.55 + fees * 2.4 + verificationTrust * 3
    + informationTrust * 2 + scalarResolutionBonus + binaryClarityBonus + thresholdHeadlineBonus
    - rewardCost - integrityPenalty - thresholdPenalty - cascadePenalty - fatigueBin * 4;
}

function simulateMarket(market: OpenMarket, cfg: PortfolioConfig, rng: () => number): ClosedMarket {
  const {topic, action} = market;
  const kind = action.kind as MarketKind;
  const N = kind === 'scalar' ? cfg.scalarBins : 2;
  const populationScale = Math.max(1, cfg.minMarketParticipants / 1000);
  const liquidity = cfg.liquidity * action.liquidityMultiplier * Math.sqrt(populationScale);
  const lmsr = new LMSR(liquidity, N);
  const duration = action.durationH;
  const reach = 1 - Math.exp(-duration / 6);
  const urgency = duration <= 1 ? 1.22 : duration <= 6 ? 1.05 : 0.9;
  const kindVoteFit = kind === 'scalar'
    ? 0.92 + topic.ambiguity * 0.22
    : kind === 'threshold'
      ? 0.96 + thresholdDrama(topic.trueTheta, action.threshold ?? 0.55) * 0.14
      : 1.06 - topic.ambiguity * 0.10;
  const rewardBoost = 0.82 + action.rewardMultiplier * 0.25;
  const verificationParticipation = verificationParticipationMultiplier(action.verification);
  const informationEngagement = informationEngagementMultiplier(action.informationMode);
  const timingUrgency = clamp(0.86 + action.timingDecay * 0.18, 0.78, 1.24);
  const expectedVotes = Math.max(
    6,
    55 * topic.trueHotness * (0.35 + 2.5 * reach) * urgency * kindVoteFit
    * rewardBoost * verificationParticipation * informationEngagement * timingUrgency
    * (0.82 + topic.newsCycleIntensity * 0.22 + topic.socialVirality * 0.18 + topic.referralElasticity * 0.10),
  );
  const votes = Math.max(cfg.minMarketParticipants, samplePoisson(expectedVotes, rng));
  const manipMult = manipulationMultiplier(action.verification);
  const majorityDirection = topic.trueTheta >= 0.5 ? 1 : -1;
  const manipulationPush = topic.manipulationRisk * manipMult * majorityDirection * 0.085;
  const turnoutPush = topic.turnoutSkew * topic.demographicPolarization * majorityDirection * 0.045;
  const influencerPush = topic.influencerActivity * topic.memeMomentum * majorityDirection * 0.025;
  const effectiveTheta = clamp(topic.trueTheta + manipulationPush + turnoutPush + influencerPush, 0.03, 0.97);
  const suspectedSybilVotes = Math.min(
    votes,
    samplePoisson(
      votes
      * (topic.manipulationRisk * 0.12 + topic.botPressure * 0.10 + topic.influencerActivity * 0.03)
      * manipMult
      * (1 + Math.log10(populationScale) * 0.06),
      rng,
    ),
  );
  const externalOutcome = rng() < topic.eventProbability ? 1 : 0;
  const opinionFactGap = Math.abs(topic.trueTheta - topic.eventProbability);
  const resolutionConfusionRate = clamp(
    0.04
    + opinionFactGap * 0.18
    + topic.ambiguity * 0.07
    + (action.informationMode === 'price-only' ? 0.06 : 0)
    + (action.informationMode === 'momentum-signals' ? 0.05 : 0)
    - (action.informationMode === 'demographic-slices' ? 0.04 : 0),
    0.02,
    0.34,
  );

  let yesVotes = 0;
  let bettorCount = 0;
  let trades = 0;
  let buyVolume = 0;
  let sellVolume = 0;
  let feeRevenue = 0;
  let voterPoints = 0;
  let raffleEntries = 0;
  let predictionError = 0;
  let voteTimeFraction = 0;
  let timingMultiplierSum = 0;
  let traderBeliefError = 0;
  let traderBeliefEntropy = 0;
  let herdingMass = 0;
  let whaleVolume = 0;
  let traderCash = 0;
  const traderShares = new Array(N).fill(0);
  const marketPublicSignal = clamp(
    effectiveTheta + normal(rng) * informationObservationNoise(action.informationMode, topic.ambiguity),
    0.01,
    0.99,
  );

  for (let i = 0; i < votes; i++) {
    const timingExponent = clamp(
      1.05 + action.timingDecay * 0.62 - informationWaitPressure(action.informationMode) * 0.32 + action.rewardMultiplier * 0.08,
      0.75,
      2.55,
    );
    const voteTime = duration * Math.pow(rng(), timingExponent);
    voteTimeFraction += duration > 0 ? voteTime / duration : 0;
    const voteYes = rng() < effectiveTheta;
    if (voteYes) yesVotes++;
    const voterPrivateSignal = clamp(effectiveTheta + normal(rng) * (0.11 + 0.09 * topic.ambiguity), 0.01, 0.99);
    const voterPublicSignal = clamp(
      marketPublicSignal + normal(rng) * informationObservationNoise(action.informationMode, topic.ambiguity) * 0.55,
      0.01,
      0.99,
    );
    const voterInfoWeight = informationSignalWeight(action.informationMode) * (voteTime / Math.max(duration, 1e-9));
    const predictedTheta = clamp(
      voterPrivateSignal * (1 - voterInfoWeight) + voterPublicSignal * voterInfoWeight,
      0.01,
      0.99,
    );
    const predictedAgree = voteYes ? predictedTheta : 1 - predictedTheta;
    const actualAgreePlaceholder = voteYes ? effectiveTheta : 1 - effectiveTheta;
    const err = Math.abs(predictedAgree - actualAgreePlaceholder);
    predictionError += err;
    const timingBoost = 1 + Math.exp(-action.timingDecay * voteTime / Math.max(0.35, duration * 0.32));
    timingMultiplierSum += timingBoost;
    const accuracyPoints = err <= 0.05 ? 18 : Math.max(0, 12 * (1 - err / 0.25));
    voterPoints += accuracyPoints * timingBoost * action.rewardMultiplier;
    if (err <= 0.20) raffleEntries++;

    const feeDrag = clamp(1.14 - action.feeRate * 18, 0.66, 1.08);
    const liquidityBoost = 0.86 + action.liquidityMultiplier * 0.14;
    const socialTradingBoost = 1 + topic.socialVirality * 0.16 + topic.influencerActivity * 0.10 + topic.memeMomentum * 0.10;
    const tradeProb = clamp(
      (0.14 + topic.trueHotness * 0.28 + topic.ambiguity * 0.08 + (kind === 'binary' ? 0.04 : kind === 'threshold' ? 0.02 : -0.02))
      * feeDrag * liquidityBoost * socialTradingBoost,
      0.04,
      0.68,
    );
    if (rng() >= tradeProb) continue;
    bettorCount++;
    const tradesAsPredictionMarket = rng() < resolutionConfusionRate;
    const traderTarget = tradesAsPredictionMarket ? topic.eventProbability : effectiveTheta;
    const privateSignal = clamp(
      traderTarget + normal(rng) * (0.16 - 0.05 * topic.trueHotness + (tradesAsPredictionMarket ? 0.04 : 0)),
      0.01,
      0.99,
    );
    const publicSignal = clamp(
      (tradesAsPredictionMarket ? topic.eventProbability : marketPublicSignal)
      + normal(rng) * informationObservationNoise(action.informationMode, topic.ambiguity) * (tradesAsPredictionMarket ? 0.72 : 0.45),
      0.01,
      0.99,
    );
    const traderInfoWeight = clamp(
      informationSignalWeight(action.informationMode)
      + topic.influencerActivity * 0.08
      + topic.memeMomentum * 0.07,
      0,
      0.86,
    );
    const signal = clamp(privateSignal * (1 - traderInfoWeight) + publicSignal * traderInfoWeight, 0.01, 0.99);
    traderBeliefError += Math.abs(signal - effectiveTheta);
    traderBeliefEntropy += bernoulliEntropy(signal);
    herdingMass += Math.abs(signal - privateSignal);
    const outcome = kind === 'binary'
      ? (signal >= 0.5 ? 0 : 1)
      : kind === 'threshold'
        ? (signal >= (action.threshold ?? 0.55) ? 0 : 1)
        : clamp(Math.floor(signal * N), 0, N - 1);
    const whaleProb = clamp(0.012 + topic.influencerActivity * 0.024 + topic.socialVirality * 0.014 + Math.log10(populationScale) * 0.006, 0.008, 0.08);
    const isWhale = rng() < whaleProb;
    const whaleMultiplier = isWhale ? 6 + 18 * rng() : 1;
    const baseBudget = clamp(
      (4 + expSample(1 / (7 + 10 * topic.trueHotness), rng))
      * (0.92 + action.liquidityMultiplier * 0.10)
      * clamp(1.08 - action.feeRate * 10, 0.78, 1.03),
      3,
      72,
    );
    const budget = clamp(baseBudget * whaleMultiplier, 3, isWhale ? 900 : 72);
    const buy = buyBudget(lmsr, outcome, budget, action.feeRate);
    trades++;
    buyVolume += budget;
    feeRevenue += buy.fee;
    traderCash -= buy.cost + buy.fee;
    traderShares[outcome] += buy.shares;
    if (isWhale) whaleVolume += budget;

    if (rng() < 0.26 + 0.10 * topic.manipulationRisk) {
      const sharesOut = traderShares[outcome] * (0.25 + 0.35 * rng());
      if (sharesOut > 1e-9) {
        const sell = sellShares(lmsr, outcome, sharesOut, action.feeRate);
        trades++;
        sellVolume += sell.gross;
        feeRevenue += sell.fee;
        traderCash += sell.gross - sell.fee;
        traderShares[outcome] -= sharesOut;
      }
    }
  }

  const finalVoteFraction = yesVotes / votes;
  const outcomeIndex = kind === 'binary'
    ? (finalVoteFraction >= 0.5 ? 0 : 1)
    : kind === 'threshold'
      ? (finalVoteFraction >= (action.threshold ?? 0.55) ? 0 : 1)
      : clamp(Math.floor(finalVoteFraction * N), 0, N - 1);
  const payout = traderShares[outcomeIndex];
  traderCash += payout;
  const traderPnl = traderCash;
  const lmsrLoss = Math.max(0, payout - buyVolume + sellVolume);
  const finalPredictionError = votes > 0 ? predictionError / votes : 0;
  const finalPrices = lmsr.prices();
  const marketImpliedVoteFraction = kind === 'scalar'
    ? sum(finalPrices.map((p, i) => p * scalarBinMidpoint(i, N)))
    : finalPrices[0];
  const observedEvent = kind === 'scalar'
    ? finalVoteFraction
    : outcomeIndex === 0 ? 1 : 0;
  const priceOpinionGap = kind === 'scalar'
    ? Math.abs(marketImpliedVoteFraction - finalVoteFraction)
    : Math.abs(finalPrices[0] - observedEvent);
  const opinionSamplingError = Math.abs(finalVoteFraction - topic.trueTheta);
  const predictionBrierScore = Math.pow(marketImpliedVoteFraction - externalOutcome, 2);
  const referralAdds = Math.round(
    votes * topic.referralElasticity * (0.012 + topic.socialVirality * 0.034 + topic.memeMomentum * 0.024)
    * informationEngagementMultiplier(action.informationMode),
  );
  const fraudPressure = votes > 0 ? suspectedSybilVotes / votes : 0;
  const rewardInflationPressure = clamp((voterPoints / Math.max(1, votes)) / 24 + action.rewardMultiplier * 0.10 + Math.log10(populationScale) * 0.025, 0, 1);
  const liquidityUtilization = buyVolume / Math.max(1, liquidity);
  const whaleTradeShare = buyVolume > 0 ? whaleVolume / buyVolume : 0;
  const churnRisk = clamp(
    0.04 + rewardInflationPressure * 0.20 + fraudPressure * 0.26 + Math.max(0, priceOpinionGap - 0.25) * 0.20
    + topic.demographicPolarization * 0.08 - topic.referralElasticity * 0.05,
    0,
    1,
  );

  return {
    id: market.id,
    topic,
    kind,
    contractLabel: contractLabel(action),
    durationH: duration,
    openAt: market.openAt,
    closeAt: market.closeAt,
    feeRate: action.feeRate,
    liquidity,
    rewardMultiplier: action.rewardMultiplier,
    verification: action.verification,
    informationMode: action.informationMode,
    timingDecay: action.timingDecay,
    threshold: action.threshold,
    finalVoteFraction,
    outcomeIndex,
    votes,
    suspectedSybilVotes,
    avgVoteTimeFraction: votes > 0 ? voteTimeFraction / votes : 0,
    avgTimingMultiplier: votes > 0 ? timingMultiplierSum / votes : 0,
    bettors: bettorCount,
    trades,
    buyVolume,
    sellVolume,
    feeRevenue,
    voterPoints,
    raffleEntries,
    avgPredictionError: finalPredictionError,
    opinionSamplingError,
    predictionBrierScore,
    externalOutcome,
    avgTraderBeliefError: bettorCount > 0 ? traderBeliefError / bettorCount : 0,
    traderBeliefEntropy: bettorCount > 0 ? traderBeliefEntropy / bettorCount : 0,
    herdingIndex: bettorCount > 0 ? clamp(herdingMass / (bettorCount * 0.5), 0, 1) : 0,
    priceOpinionGap,
    marketMakerRiskBound: lmsr.b * Math.log(N),
    fraudPressure,
    referralAdds,
    churnRisk,
    rewardInflationPressure,
    liquidityUtilization,
    whaleTradeShare,
    traderPnl,
    lmsrLoss,
  };
}

export function runPortfolio(policy: SchedulerPolicy, cfg: PortfolioConfig, mdp: OperatorMDP): PolicyRun {
  const station = new FactMachinePortfolioStation(cfg, policy, mdp, mulberry32(cfg.seed + policySeed(policy)));
  const ticks = Math.round(cfg.horizonH / cfg.stepH);
  for (let tick = 0; tick <= ticks; tick++) station.runTimeStep(cfg.stepH, tick);
  while (station.active.length > 0) {
    const nextClose = Math.min(...station.active.map(m => m.closeAt));
    station.runTimeStep(cfg.stepH, Math.ceil(nextClose / cfg.stepH));
  }
  return station.toRun();
}

function aggregateRun(policy: SchedulerPolicy, markets: ClosedMarket[], cfg: PortfolioConfig): PolicyAggregate {
  const marketsOpened = markets.length;
  const binaryMarkets = markets.filter(m => m.kind === 'binary').length;
  const scalarMarkets = markets.filter(m => m.kind === 'scalar').length;
  const thresholdMarkets = markets.filter(m => m.kind === 'threshold').length;
  const feeRevenue = sum(markets.map(m => m.feeRevenue));
  const voterPoints = sum(markets.map(m => m.voterPoints));
  const platformSurplus = feeRevenue - voterPoints * 0.012 - sum(markets.map(m => m.lmsrLoss));
  const votes = sum(markets.map(m => m.votes));
  const bettors = sum(markets.map(m => m.bettors));
  return {
    scenarioLabel: cfg.scenarioLabel,
    minMarketParticipants: cfg.minMarketParticipants,
    policy,
    marketsOpened,
    marketsClosed: markets.length,
    binaryMarkets,
    scalarMarkets,
    thresholdMarkets,
    avgDurationH: mean(markets.map(m => m.durationH)),
    avgFeeRate: mean(markets.map(m => m.feeRate)),
    avgLiquidity: mean(markets.map(m => m.liquidity)),
    avgRewardMultiplier: mean(markets.map(m => m.rewardMultiplier)),
    proofMarkets: markets.filter(m => m.verification === 'proof').length,
    avgTimingDecay: mean(markets.map(m => m.timingDecay)),
    votes,
    suspectedSybilVotes: sum(markets.map(m => m.suspectedSybilVotes)),
    avgVoteTimeFraction: weightedMean(markets.map(m => [m.avgVoteTimeFraction, m.votes])),
    avgTimingMultiplier: weightedMean(markets.map(m => [m.avgTimingMultiplier, m.votes])),
    bettors,
    trades: sum(markets.map(m => m.trades)),
    buyVolume: sum(markets.map(m => m.buyVolume)),
    sellVolume: sum(markets.map(m => m.sellVolume)),
    feeRevenue,
    voterPoints,
    raffleEntries: sum(markets.map(m => m.raffleEntries)),
    avgPredictionError: weightedMean(markets.map(m => [m.avgPredictionError, m.votes])),
    avgOpinionSamplingError: weightedMean(markets.map(m => [m.opinionSamplingError, m.votes])),
    avgPredictionBrierScore: weightedMean(markets.map(m => [m.predictionBrierScore, Math.max(1, m.trades)])),
    avgTraderBeliefError: weightedMean(markets.map(m => [m.avgTraderBeliefError, m.bettors])),
    traderBeliefEntropy: weightedMean(markets.map(m => [m.traderBeliefEntropy, m.bettors])),
    herdingIndex: weightedMean(markets.map(m => [m.herdingIndex, m.bettors])),
    priceOpinionGap: weightedMean(markets.map(m => [m.priceOpinionGap, Math.max(1, m.trades)])),
    marketMakerRiskBound: sum(markets.map(m => m.marketMakerRiskBound)),
    fraudPressure: weightedMean(markets.map(m => [m.fraudPressure, m.votes])),
    referralAdds: sum(markets.map(m => m.referralAdds)),
    churnRisk: weightedMean(markets.map(m => [m.churnRisk, m.votes])),
    rewardInflationPressure: weightedMean(markets.map(m => [m.rewardInflationPressure, m.votes])),
    liquidityUtilization: weightedMean(markets.map(m => [m.liquidityUtilization, Math.max(1, m.trades)])),
    whaleTradeShare: weightedMean(markets.map(m => [m.whaleTradeShare, Math.max(1, m.trades)])),
    avgNewsCycleIntensity: weightedMean(markets.map(m => [m.topic.newsCycleIntensity, m.votes])),
    avgSocialVirality: weightedMean(markets.map(m => [m.topic.socialVirality, m.votes])),
    avgInfluencerActivity: weightedMean(markets.map(m => [m.topic.influencerActivity, m.votes])),
    avgDemographicPolarization: weightedMean(markets.map(m => [m.topic.demographicPolarization, m.votes])),
    traderPnl: sum(markets.map(m => m.traderPnl)),
    lmsrLoss: sum(markets.map(m => m.lmsrLoss)),
    platformSurplus,
    engagementScore: votes + 1.8 * bettors + 0.35 * marketsOpened,
  };
}

function aggregateByKind(markets: ClosedMarket[]): MarketKindAggregate[] {
  const kinds: MarketKind[] = ['binary', 'scalar', 'threshold'];
  return kinds.map(kind => {
    const subset = markets.filter(m => m.kind === kind);
    const feeRevenue = sum(subset.map(m => m.feeRevenue));
    const voterPoints = sum(subset.map(m => m.voterPoints));
    return {
      kind,
      markets: subset.length,
      votes: sum(subset.map(m => m.votes)),
      bettors: sum(subset.map(m => m.bettors)),
      trades: sum(subset.map(m => m.trades)),
      buyVolume: sum(subset.map(m => m.buyVolume)),
      sellVolume: sum(subset.map(m => m.sellVolume)),
      feeRevenue,
      voterPoints,
      suspectedSybilVotes: sum(subset.map(m => m.suspectedSybilVotes)),
      avgDurationH: mean(subset.map(m => m.durationH)),
      avgLiquidity: mean(subset.map(m => m.liquidity)),
      avgFeeRate: mean(subset.map(m => m.feeRate)),
      avgPredictionError: weightedMean(subset.map(m => [m.avgPredictionError, m.votes])),
      avgOpinionSamplingError: weightedMean(subset.map(m => [m.opinionSamplingError, m.votes])),
      avgPredictionBrierScore: weightedMean(subset.map(m => [m.predictionBrierScore, Math.max(1, m.trades)])),
      avgTraderBeliefError: weightedMean(subset.map(m => [m.avgTraderBeliefError, m.bettors])),
      traderBeliefEntropy: weightedMean(subset.map(m => [m.traderBeliefEntropy, m.bettors])),
      herdingIndex: weightedMean(subset.map(m => [m.herdingIndex, m.bettors])),
      priceOpinionGap: weightedMean(subset.map(m => [m.priceOpinionGap, Math.max(1, m.trades)])),
      fraudPressure: weightedMean(subset.map(m => [m.fraudPressure, m.votes])),
      liquidityUtilization: weightedMean(subset.map(m => [m.liquidityUtilization, Math.max(1, m.trades)])),
      whaleTradeShare: weightedMean(subset.map(m => [m.whaleTradeShare, Math.max(1, m.trades)])),
      platformSurplus: feeRevenue - voterPoints * 0.012 - sum(subset.map(m => m.lmsrLoss)),
    };
  });
}

function buildDailySummaries(
  markets: ClosedMarket[],
  timeline: PolicyRun['timeline'],
  cfg: PortfolioConfig,
): DailySummary[] {
  const horizonDays = Math.ceil(cfg.horizonH / 24);
  const maxCloseDay = markets.length === 0 ? 0 : Math.max(...markets.map(m => dayIndex(m.closeAt)));
  const maxTimelineDay = timeline.length === 0 ? 0 : Math.max(...timeline.map(x => x.day));
  const days = Math.max(horizonDays, maxCloseDay + 1, maxTimelineDay + 1);
  const lastTimelineByDay = new Map<number, PolicyRun['timeline'][number]>();
  for (const frame of timeline) lastTimelineByDay.set(frame.day, frame);

  const summaries: DailySummary[] = [];
  for (let day = 0; day < days; day++) {
    const opened = markets.filter(m => dayIndex(m.openAt) === day);
    const closed = markets.filter(m => dayIndex(m.closeAt) === day);
    const lastFrame = lastTimelineByDay.get(day);
    summaries.push({
      day,
      marketCap: dailyMarketCapForDay(day, cfg),
      opened: opened.length,
      closed: closed.length,
      activeEnd: lastFrame?.open ?? 0,
      queuedEnd: lastFrame?.queued ?? 0,
      votes: sum(closed.map(m => m.votes)),
      bettors: sum(closed.map(m => m.bettors)),
      trades: sum(closed.map(m => m.trades)),
      feeRevenue: sum(closed.map(m => m.feeRevenue)),
      voterPoints: sum(closed.map(m => m.voterPoints)),
      binaryClosed: closed.filter(m => m.kind === 'binary').length,
      scalarClosed: closed.filter(m => m.kind === 'scalar').length,
      thresholdClosed: closed.filter(m => m.kind === 'threshold').length,
      avgPredictionError: weightedMean(closed.map(m => [m.avgPredictionError, m.votes])),
      avgOpinionSamplingError: weightedMean(closed.map(m => [m.opinionSamplingError, m.votes])),
      avgPredictionBrierScore: weightedMean(closed.map(m => [m.predictionBrierScore, Math.max(1, m.trades)])),
      fraudPressure: weightedMean(closed.map(m => [m.fraudPressure, m.votes])),
      herdingIndex: weightedMean(closed.map(m => [m.herdingIndex, m.bettors])),
    });
  }
  return summaries;
}

function sampleTopic(id: number, now: number, rng: () => number): CandidateTopic {
  const categories: Category[] = ['politics', 'culture', 'sports', 'conspiracy', 'breaking'];
  const weights = [0.26, 0.22, 0.16, 0.16, 0.20];
  const category = categories[categorical(weights, rng)];
  const categoryHotBoost: Record<Category, number> = {
    politics: 0.08,
    culture: -0.02,
    sports: -0.04,
    conspiracy: 0.02,
    breaking: 0.14,
  };
  const trueHotness = clamp(betaLike(rng, 2.0, 2.2) + categoryHotBoost[category], 0.05, 0.98);
  const ambiguity = clamp(betaLike(rng, 2.1, 2.1), 0.05, 0.95);
  const lean = (rng() < 0.5 ? -1 : 1) * (0.08 + 0.35 * (1 - ambiguity) * rng());
  const trueTheta = clamp(0.5 + lean, 0.03, 0.97);
  const newsCycleIntensity = clamp(betaLike(rng, category === 'breaking' ? 3.2 : 2.1, 2.0) + (category === 'breaking' ? 0.10 : 0), 0.02, 1);
  const socialVirality = clamp(0.22 + 0.48 * trueHotness + 0.22 * newsCycleIntensity + normal(rng) * 0.13, 0, 1);
  const influencerActivity = clamp(betaLike(rng, 1.8, 2.4) + (category === 'politics' || category === 'culture' ? 0.08 : 0), 0, 1);
  const memeMomentum = clamp(0.18 + 0.44 * socialVirality + 0.20 * ambiguity + normal(rng) * 0.12, 0, 1);
  const demographicPolarization = clamp(0.12 + 0.55 * (1 - ambiguity) + (category === 'politics' || category === 'conspiracy' ? 0.14 : 0) + normal(rng) * 0.10, 0, 1);
  const couplingPrior: Record<Category, number> = {
    politics: 0.48,
    culture: 0.32,
    sports: 0.78,
    conspiracy: 0.26,
    breaking: 0.66,
  };
  const opinionEventCoupling = clamp(
    couplingPrior[category] + newsCycleIntensity * 0.12 - ambiguity * 0.12 - demographicPolarization * 0.16,
    0.12,
    0.92,
  );
  const eventProbability = clamp(
    0.5
    + (trueTheta - 0.5) * opinionEventCoupling
    + normal(rng) * (0.12 + ambiguity * 0.10 + (category === 'conspiracy' ? 0.08 : 0)),
    0.03,
    0.97,
  );
  const turnoutSkew = clamp(Math.abs(normal(rng)) * 0.18 + demographicPolarization * 0.24 + socialVirality * 0.10, 0, 0.75);
  const botPressure = clamp(0.03 + 0.20 * socialVirality + 0.22 * influencerActivity + (category === 'politics' || category === 'conspiracy' ? 0.08 : 0) + normal(rng) * 0.06, 0, 0.75);
  const referralElasticity = clamp(0.10 + 0.46 * memeMomentum + 0.22 * socialVirality + normal(rng) * 0.10, 0, 1);
  const manipulationRisk = clamp(
    0.06 + 0.18 * trueHotness * ambiguity + 0.16 * botPressure + 0.08 * demographicPolarization
    + (category === 'politics' || category === 'conspiracy' ? 0.08 : 0),
    0,
    0.68,
  );
  const observedBuzz = clamp(trueHotness + 0.18 * newsCycleIntensity + 0.16 * socialVirality + 0.10 * influencerActivity + normal(rng) * 0.14, 0, 1);
  const observedAmbiguity = clamp(ambiguity + normal(rng) * 0.16, 0, 1);
  return {
    id,
    category,
    createdAt: now,
    expiresAt: now + 5.5,
    trueHotness,
    ambiguity,
    trueTheta,
    manipulationRisk,
    newsCycleIntensity,
    socialVirality,
    influencerActivity,
    memeMomentum,
    demographicPolarization,
    opinionEventCoupling,
    eventProbability,
    turnoutSkew,
    botPressure,
    referralElasticity,
    observedBuzz,
    observedAmbiguity,
  };
}

function buyBudget(lmsr: LMSR, outcome: number, grossBudget: number, feeRate: number): {shares: number; cost: number; fee: number} {
  const fee = grossBudget * feeRate;
  const budget = grossBudget - fee;
  const prices = lmsr.prices();
  const p = Math.max(1e-9, prices[outcome]);
  const shares = lmsr.b * Math.log1p(Math.expm1(budget / lmsr.b) / p);
  const dq = new Array(lmsr.N).fill(0);
  dq[outcome] = shares;
  const cost = lmsr.trade(dq);
  return {shares, cost, fee};
}

function sellShares(lmsr: LMSR, outcome: number, shares: number, feeRate: number): {gross: number; fee: number} {
  const dq = new Array(lmsr.N).fill(0);
  dq[outcome] = -shares;
  const gross = -lmsr.cost(dq);
  lmsr.trade(dq);
  return {gross, fee: gross * feeRate};
}

function buildHTML(runs: PolicyRun[], mdp: OperatorMDP, cfg: PortfolioConfig): string {
  const data = JSON.stringify({runs, mdp: summarizeOperatorMDP(mdp), cfg})
    .replace(/<\/(?=script)/gi, '<\\/')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FactMachine multi-market MDP/POMDP simulation</title>
<style>
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:#f5f5f7;color:#171717}
header{padding:16px 22px 10px;background:#fff;border-bottom:1px solid #ddd}
h1{font-size:19px;margin:0 0 4px}.sub{color:#666;font-size:13px;margin:0}
main{padding:16px 22px;max-width:1280px;margin:auto}
.controls,.time-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
select,button{padding:6px 8px;border:1px solid #bbb;border-radius:4px;background:white;font:inherit}
button{cursor:pointer}.grid{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin:12px 0}
.metric{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px}.metric .k{font-size:12px;color:#666}.metric .v{font-size:20px;font-weight:650;margin-top:2px}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin:12px 0;overflow:auto}.panel h3{font-size:14px;margin:0 0 10px}
.plot-grid{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:12px}.summary-grid{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:10px}
.summary-card{border:1px solid #e2e2e2;border-radius:6px;padding:10px;background:#fafafa;line-height:1.35}.summary-card h4{margin:0 0 6px;font-size:14px}.summary-card p{margin:5px 0;font-size:12px;color:#333}.summary-card b{color:#111}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #eee;padding:7px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
svg{width:100%;height:auto;display:block;background:#fff}.note{font-size:12px;color:#555;line-height:1.45}.pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eee;margin-right:4px}
input[type=range]{min-width:260px;flex:1}.readout{font-family:SF Mono,Menlo,Consolas,monospace;color:#333;font-size:12px;min-width:360px}
@media(max-width:860px){main{padding:12px}.grid{grid-template-columns:repeat(2,minmax(120px,1fr))}.summary-grid,.plot-grid{grid-template-columns:1fr}.readout{min-width:100%}}
</style>
</head>
<body>
<header>
  <h1>FactMachine multi-market MDP/POMDP simulation</h1>
  <p class="sub">Multiple opinion markets open/close over ${cfg.horizonH / 24} days with daily launch capacity from ${cfg.minDailyMarkets} to ${cfg.maxDailyMarkets}. Betting is gated by voting; scalar, binary, and over/under contracts are compared across 1k and 10k participant-scale scenarios.</p>
</header>
<main>
  <div class="panel intro-panel">
    <h3>What This Page Is Attempting To Do</h3>
    <div class="summary-grid">
      <section class="summary-card">
        <h4>Goal</h4>
        <p>This is an operator simulation for FactMachine-style opinion markets. It asks which topics to open, when to open them, how long to run them, and whether each market should resolve as majority binary, scalar vote distribution, or over/under threshold.</p>
        <p>The output is not a claim that one policy is universally best; it is a controlled comparison of launch policies under the same synthetic participation, fraud, liquidity, timing, and information assumptions.</p>
      </section>
      <section class="summary-card">
        <h4>How MDP Works</h4>
        <p>The MDP has ${mdp.spec.numStates} operator states: topic hotness, topic ambiguity, and market-load fatigue. For each state it evaluates ${mdp.actions.length} action recipes covering contract type, duration, fees, liquidity, rewards, verification, and information visibility.</p>
        <p>Value iteration estimates the long-run value of each action and chooses the recipe with the best expected future reward.</p>
      </section>
      <section class="summary-card">
        <h4>How POMDP Works</h4>
        <p>The POMDP version does not get to see true topic hotness or ambiguity. It observes noisy buzz and ambiguity signals, maintains a belief over possible states, and chooses actions by averaging MDP Q-values under that belief.</p>
        <p>After markets close, the belief updates from realized engagement, fees, and trades. This makes the POMDP closer to what a live operator would actually know.</p>
      </section>
      <section class="summary-card">
        <h4>Opinion vs Prediction</h4>
        <p>Opinion markets resolve to the final participant vote distribution. Prediction-market accuracy is tracked as a counterfactual Brier score against a latent external event outcome, because public opinion can diverge from what later turns out to be true.</p>
        <p>The dashboard now separates opinion sampling error from prediction Brier score so a market can be good at measuring opinion while still being a poor forecast of external reality.</p>
      </section>
      <section class="summary-card">
        <h4>Shortcomings</h4>
        <p>The model is synthetic. It assumes simple parametric behavior for voters, bettors, manipulation, liquidity response, and information exposure. Real behavior may have regime changes, coordinated campaigns, platform shocks, and feedback loops not captured here.</p>
      </section>
      <section class="summary-card">
        <h4>Missing Wildcards</h4>
        <p>Important unknowns include acquisition channel quality, real identity verification failure rates, creator/influencer incentives, legal or moderation constraints, market maker capital limits, off-platform coordination, news shocks, bot adaptation, and whether users understand that opinion markets do not necessarily resolve to factual truth.</p>
      </section>
    </div>
  </div>
  <div class="controls">
    <label>scale <select id="scenario"></select></label>
    <label>policy <select id="policy"></select></label>
    <label>contract metric <select id="contractMetric"></select></label>
    <label>scale metric <select id="scaleMetric"></select></label>
    <span id="badges"></span>
  </div>
  <div class="grid" id="metrics"></div>
  <div class="panel">
    <h3>Time-Step Simulation</h3>
    <div class="time-controls">
      <button id="play">Play</button>
      <label>speed <select id="playbackSpeed"></select></label>
      <button id="stepBack">Step -</button>
      <button id="stepForward">Step +</button>
      <input id="timeScrub" type="range" min="0" value="0" step="1">
      <span class="readout" id="timeReadout"></span>
    </div>
    <svg id="statePlot" viewBox="0 0 1160 360" aria-label="time step simulation state"></svg>
  </div>
  <div class="plot-grid">
    <div class="panel"><h3>Plot 1: Daily Market Capacity vs Opens</h3><svg id="dailyPlot" viewBox="0 0 580 320" aria-label="daily market capacity"></svg></div>
    <div class="panel"><h3>Plot 2: Cumulative Votes, Bettors, Fees</h3><svg id="throughputPlot" viewBox="0 0 580 320" aria-label="cumulative throughput"></svg></div>
    <div class="panel"><h3>Plot 3: Binary vs Scalar Contract Variables</h3><svg id="contractPlot" viewBox="0 0 580 320" aria-label="contract comparison"></svg></div>
    <div class="panel"><h3>Plot 4: 1k vs 10k Scale Comparison</h3><svg id="scalePlot" viewBox="0 0 580 320" aria-label="scale comparison"></svg></div>
  </div>
  <div class="panel"><h3>Executive Summary</h3><div id="execSummary"></div></div>
  <div class="panel"><h3>1k vs 10k Scale Comparison</h3><div id="scaleComparison"></div></div>
  <div class="panel"><h3>Nonlinearity Diagnostics</h3><div id="nonlinear"></div></div>
  <div class="panel"><h3>Policy Comparison</h3><div id="comparison"></div></div>
  <div class="panel"><h3>Decision Levers Learned</h3><div id="levers"></div></div>
  <div class="panel"><h3>Agent Model Decomposition</h3><div id="agents"></div></div>
  <div class="panel"><h3>Selected Policy Market Mix</h3><div id="mix"></div></div>
  <div class="panel"><h3>MDP/POMDP Notes</h3><div class="note" id="notes"></div></div>
</main>
<script type="application/json" id="data">${data}</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const runs=DATA.runs;
const fmt=n=>Number(n||0).toLocaleString(undefined,{maximumFractionDigits:1});
const pct=n=>(100*Number(n||0)).toFixed(1)+'%';
const money=n=>'$'+fmt(n);
const actionMeta=Object.fromEntries(DATA.mdp.actions.map(a=>[a.label,a]));
const policySel=document.getElementById('policy');
const scenarioSel=document.getElementById('scenario');
const contractMetricSel=document.getElementById('contractMetric');
const scaleMetricSel=document.getElementById('scaleMetric');
const scrub=document.getElementById('timeScrub');
const playBtn=document.getElementById('play');
const speedSel=document.getElementById('playbackSpeed');
const stepBackBtn=document.getElementById('stepBack');
const stepForwardBtn=document.getElementById('stepForward');
const policies=[...new Set(runs.map(r=>r.policy))];
const scenarios=[...new Set(runs.map(r=>r.scenarioLabel))];
const contractMetrics=[
 ['markets','markets'],['votes','votes'],['bettors','bettors'],['trades','trades'],['feeRevenue','fee revenue'],['platformSurplus','surplus'],['avgPredictionError','voter prediction error'],['avgOpinionSamplingError','opinion sampling error'],['avgPredictionBrierScore','prediction Brier score'],['herdingIndex','herding'],['fraudPressure','fraud']
];
const scaleMetrics=[
 ['votes','votes'],['bettors','bettors'],['trades','trades'],['feeRevenue','fee revenue'],['platformSurplus','surplus'],['avgOpinionSamplingError','opinion sampling error'],['avgPredictionBrierScore','prediction Brier score'],['fraudPressure','fraud'],['herdingIndex','herding'],['liquidityUtilization','liquidity use'],['whaleTradeShare','whale share']
];
for(const s of scenarios){const o=document.createElement('option');o.value=s;o.textContent=s;o.selected=s.includes('10,000');scenarioSel.appendChild(o);}
for(const p of policies){const o=document.createElement('option');o.value=p;o.textContent=p;o.selected=p==='pomdp-belief';policySel.appendChild(o);}
for(const m of contractMetrics){const o=document.createElement('option');o.value=m[0];o.textContent=m[1];contractMetricSel.appendChild(o);}
for(const m of scaleMetrics){const o=document.createElement('option');o.value=m[0];o.textContent=m[1];scaleMetricSel.appendChild(o);}
for(const speed of [0.5,1,2,3,5,10]){const o=document.createElement('option');o.value=String(speed);o.textContent=speed+'x';o.selected=speed===2;speedSel.appendChild(o);}
scaleMetricSel.value='votes';
let frameIndex=0;
let playTimer=null;
policySel.addEventListener('change',()=>{frameIndex=0;render();});
scenarioSel.addEventListener('change',()=>{frameIndex=0;render();});
contractMetricSel.addEventListener('change',render);
scaleMetricSel.addEventListener('change',render);
speedSel.addEventListener('change',()=>{if(playTimer) startPlayback(); render();});
scrub.addEventListener('input',()=>{frameIndex=Number(scrub.value);render();});
stepBackBtn.addEventListener('click',()=>{const r=selected();frameIndex=Math.max(0,frameIndex-1);scrub.value=String(frameIndex);render();});
stepForwardBtn.addEventListener('click',()=>{const r=selected();frameIndex=Math.min(r.timeline.length-1,frameIndex+1);scrub.value=String(frameIndex);render();});
playBtn.addEventListener('click',()=>{if(playTimer) stopPlayback(); else startPlayback();});
function playbackDelayMs(){return Math.max(16,Math.round(260/Number(speedSel.value||1)));}
function advanceFrame(){const r=selected();frameIndex=(frameIndex+1)%Math.max(1,r.timeline.length);scrub.value=String(frameIndex);render();}
function stopPlayback(){if(playTimer) clearInterval(playTimer);playTimer=null;playBtn.textContent='Play';}
function startPlayback(){stopPlayback();playBtn.textContent='Pause';playTimer=setInterval(advanceFrame,playbackDelayMs());}
function currentRuns(){return runs.filter(r=>r.scenarioLabel===scenarioSel.value);}
function selected(){return runs.find(r=>r.policy===policySel.value&&r.scenarioLabel===scenarioSel.value)||currentRuns()[0]||runs[0];}
function render(){
	 const r=selected(), a=r.aggregate;
 if(frameIndex>=r.timeline.length) frameIndex=Math.max(0,r.timeline.length-1);
 scrub.max=String(Math.max(0,r.timeline.length-1));
 scrub.value=String(frameIndex);
 const f=r.timeline[frameIndex]||r.timeline[0]||{t:0,day:0,open:0,closed:0,queued:0,votes:0,bettors:0,trades:0,fees:0,marketCap:0,openedToday:0,openedTotal:0};
 document.getElementById('badges').innerHTML='<span class="pill">'+r.scenarioLabel+'</span><span class="pill">'+a.marketsClosed+' closed markets</span><span class="pill">'+a.binaryMarkets+' binary</span><span class="pill">'+a.scalarMarkets+' scalar</span><span class="pill">'+a.thresholdMarkets+' over/under</span><span class="pill">day '+(f.day+1)+'</span>';
	 document.getElementById('metrics').innerHTML=[
  ['Votes',fmt(a.votes)],['Bettors',fmt(a.bettors)],['Trades',fmt(a.trades)],['Fee revenue','$'+fmt(a.feeRevenue)],
  ['Voter points',fmt(a.voterPoints)],['Raffle entries',fmt(a.raffleEntries)],['Platform surplus','$'+fmt(a.platformSurplus)],['Avg pred error',pct(a.avgPredictionError)],
  ['Opinion sampling err',pct(a.avgOpinionSamplingError)],['Prediction Brier',fmt(a.avgPredictionBrierScore)],['Opinion/price gap',pct(a.priceOpinionGap)],['External forecast miss',fmt(Math.sqrt(a.avgPredictionBrierScore||0))],
  ['Avg fee',pct(a.avgFeeRate)],['Avg liquidity','$'+fmt(a.avgLiquidity)],['Avg reward',fmt(a.avgRewardMultiplier)+'x'],['Suspect votes',fmt(a.suspectedSybilVotes)],
  ['Avg vote time',pct(a.avgVoteTimeFraction)],['Timing boost',fmt(a.avgTimingMultiplier)+'x'],['Herding index',pct(a.herdingIndex)],['Fraud pressure',pct(a.fraudPressure)],
  ['Trader belief err',pct(a.avgTraderBeliefError)],['MM risk bound','$'+fmt(a.marketMakerRiskBound)],['Timing decay',fmt(a.avgTimingDecay)],
  ['Referrals',fmt(a.referralAdds)],['Churn risk',pct(a.churnRisk)],['Reward inflation',pct(a.rewardInflationPressure)],['Liq utilization',fmt(a.liquidityUtilization)+'x'],
  ['Whale share',pct(a.whaleTradeShare)],['News intensity',pct(a.avgNewsCycleIntensity)],['Social virality',pct(a.avgSocialVirality)],['Demo polarization',pct(a.avgDemographicPolarization)]
	 ].map(([k,v])=>'<div class="metric"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');
	 drawStatePlot(r,f);
	 drawDailyPlot(r);
	 drawThroughputPlot(r);
	 drawContractPlot(r);
	 drawScalePlot();
	 drawExecutiveSummary();
	 drawScaleComparison();
	 drawNonlinearityDiagnostics();
	 drawComparison();
	 drawLevers(r);
 drawAgents(r);
 drawMix(r);
 drawNotes(r);
}
function drawStatePlot(r,f){
 const svg=document.getElementById('statePlot'); clear(svg);
 const W=1160,H=360,pad=42;
 rect(svg,0,0,W,H,'#fbfbfc','#ddd');
 text(svg,24,30,'day '+(f.day+1)+' / t='+f.t.toFixed(2)+'h','#111',16,'bold');
 text(svg,24,54,'cap '+f.marketCap+', opened today '+f.openedToday+', active '+f.open+', queued '+f.queued+', closed '+f.closed,'#555',12);
 document.getElementById('timeReadout').textContent='step '+frameIndex+' of '+Math.max(0,r.timeline.length-1)+' | day '+(f.day+1)+' | votes '+fmt(f.votes)+' | bettors '+fmt(f.bettors)+' | trades '+fmt(f.trades)+' | fees '+money(f.fees);
 const closed=marketsUntil(r,f.t);
 const kinds=['binary','scalar','threshold'];
 const kindColors={binary:'#2563eb',scalar:'#7c3aed',threshold:'#e11d48'};
 const kindCounts=Object.fromEntries(kinds.map(k=>[k,closed.filter(m=>m.kind===k).length]));
 const totalKind=Math.max(1,kinds.reduce((s,k)=>s+kindCounts[k],0));
 let x=24;
 for(const k of kinds){
   const w=280*(kindCounts[k]/totalKind);
   rect(svg,x,82,w,24,kindColors[k],null,0.88);
   text(svg,x+5,99,k+' '+kindCounts[k],'#fff',11,'bold');
   x+=w;
 }
 const bars=[
  ['active',f.open,Math.max(1,DATA.cfg.maxConcurrent),'#0f766e'],
  ['queued',f.queued,Math.max(1,...r.timeline.map(x=>x.queued)),'#f59e0b'],
  ['closed',f.closed,Math.max(1,r.aggregate.marketsClosed),'#334155'],
  ['votes',f.votes,Math.max(1,r.aggregate.votes),'#2563eb'],
  ['bettors',f.bettors,Math.max(1,r.aggregate.bettors),'#7c3aed'],
  ['fees',f.fees,Math.max(1,r.aggregate.feeRevenue),'#16a34a']
 ];
 const barX=24,barY=145,barW=106,barGap=22,barH=170;
 for(let i=0;i<bars.length;i++){
  const b=bars[i], h=barH*(b[1]/b[2]);
  rect(svg,barX+i*(barW+barGap),barY+barH-h,barW,h,b[3],null,0.9);
  rect(svg,barX+i*(barW+barGap),barY,barW,barH,'none','#ddd');
  text(svg,barX+i*(barW+barGap)+barW/2,barY+barH+20,b[0],'#444',12,'normal','middle');
  text(svg,barX+i*(barW+barGap)+barW/2,barY+barH-h-8,formatMetric(b[0]==='fees'?'feeRevenue':b[0],b[1]),b[3],12,'bold','middle');
 }
 const recent=closed.slice(-10).reverse();
 const listX=850;
 text(svg,listX,82,'latest closed markets','#111',13,'bold');
 for(let i=0;i<recent.length;i++){
  const m=recent[i], y=108+i*22;
  rect(svg,listX,y-13,10,10,kindColors[m.kind],null);
  text(svg,listX+18,y,'#'+m.id+' '+m.kind+' '+m.topic.category+' votes '+fmt(m.votes)+' fees '+money(m.feeRevenue),'#333',11);
 }
}
function drawDailyPlot(r){
 const svg=document.getElementById('dailyPlot'); clear(svg);
 const W=580,H=320,pad=38,days=r.daily.filter(d=>d.day<Math.ceil(DATA.cfg.horizonH/24));
 const maxY=Math.max(1,...days.map(d=>Math.max(d.marketCap,d.opened,d.binaryClosed+d.scalarClosed+d.thresholdClosed)));
 axes(svg,W,H,pad,'days','markets');
 const bw=(W-pad*1.6)/Math.max(1,days.length);
 for(const d of days){
  const x=pad+d.day*bw, y=scaleY(d.opened,maxY,H,pad);
  rect(svg,x+1,y,Math.max(1,bw-2),H-pad-y,'#38bdf8',null,0.55);
 }
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.marketCap,maxY,H,pad)]),'#0f172a',2);
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.scalarClosed,maxY,H,pad)]),'#7c3aed',2);
 path(svg,days.map(d=>[pad+(d.day+0.5)*bw,scaleY(d.binaryClosed,maxY,H,pad)]),'#2563eb',2);
 const currentDay=(r.timeline[frameIndex]||{}).day||0;
 line(svg,pad+(currentDay+0.5)*bw,pad,pad+(currentDay+0.5)*bw,H-pad,'#ef4444');
 legend(svg,48,20,[['opened','#38bdf8'],['cap','#0f172a'],['scalar closed','#7c3aed'],['binary closed','#2563eb']]);
}
function drawThroughputPlot(r){
 const svg=document.getElementById('throughputPlot'); clear(svg);
 const W=580,H=320,pad=38,tl=r.timeline;
 const maxT=Math.max(1,...tl.map(x=>x.t)), maxY=Math.max(1,r.aggregate.votes,r.aggregate.bettors,r.aggregate.feeRevenue);
 axes(svg,W,H,pad,'hours','cumulative');
 const pts=(field,scale)=>tl.map(x=>[scaleX(x.t,maxT,W,pad),scaleY(x[field]*scale,maxY,H,pad)]);
 path(svg,pts('votes',1),'#2563eb',2);
 path(svg,pts('bettors',1),'#7c3aed',2);
 path(svg,pts('fees',1),'#16a34a',2);
 const f=tl[frameIndex]||tl[0];
 if(f) line(svg,scaleX(f.t,maxT,W,pad),pad,scaleX(f.t,maxT,W,pad),H-pad,'#ef4444');
 legend(svg,48,20,[['votes','#2563eb'],['bettors','#7c3aed'],['fee revenue','#16a34a']]);
}
function drawContractPlot(r){
 const svg=document.getElementById('contractPlot'); clear(svg);
 const W=580,H=320,pad=48,metric=contractMetricSel.value;
 const rows=r.kindBreakdown.filter(x=>x.kind!=='threshold'||x.markets>0);
 const maxY=Math.max(1,...rows.map(x=>Math.abs(x[metric]||0)));
 axes(svg,W,H,pad,'contract','value');
 const colors={binary:'#2563eb',scalar:'#7c3aed',threshold:'#e11d48'};
 const bw=(W-pad*2)/Math.max(1,rows.length)/1.6;
 for(let i=0;i<rows.length;i++){
  const row=rows[i], v=row[metric]||0, x=pad+30+i*((W-pad*2)/Math.max(1,rows.length)), h=(H-pad*2)*(Math.abs(v)/maxY);
  rect(svg,x,H-pad-h,bw,h,colors[row.kind],null,0.9);
  text(svg,x+bw/2,H-pad+18,row.kind,'#333',11,'normal','middle');
  text(svg,x+bw/2,H-pad-h-8,formatMetric(metric,v),colors[row.kind],11,'bold','middle');
 }
 text(svg,48,22,'selected metric: '+metricLabel(metric),'#111',12,'bold');
}
function drawScalePlot(){
 const svg=document.getElementById('scalePlot'); clear(svg);
 const W=580,H=320,pad=48,metric=scaleMetricSel.value;
 const byPolicy=policies.map(p=>runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants));
 const vals=[];
 for(const pair of byPolicy){for(const r of pair) vals.push(Math.abs(r.aggregate[metric]||0));}
 const maxY=Math.max(1,...vals);
 axes(svg,W,H,pad,'policy','value');
 const groupW=(W-pad*2)/Math.max(1,policies.length), bw=groupW*0.28;
 for(let i=0;i<byPolicy.length;i++){
  const pair=byPolicy[i];
  for(let j=0;j<pair.length;j++){
   const r=pair[j], v=r.aggregate[metric]||0, x=pad+i*groupW+groupW*0.22+j*bw*1.2, h=(H-pad*2)*(Math.abs(v)/maxY);
   rect(svg,x,H-pad-h,bw,h,j===0?'#60a5fa':'#1d4ed8',null,0.9);
   text(svg,x+bw/2,H-pad-h-8,formatMetric(metric,v),j===0?'#2563eb':'#1e3a8a',10,'bold','middle');
  }
  text(svg,pad+i*groupW+groupW/2,H-pad+18,policies[i].replace('-',' '),'#333',10,'normal','middle');
 }
 legend(svg,48,20,[[scenarios[0]||'low scale','#60a5fa'],[scenarios[scenarios.length-1]||'high scale','#1d4ed8']]);
}
		function drawExecutiveSummary(){
		 document.getElementById('execSummary').innerHTML='<div class="summary-grid">'+
	 currentRuns().map(r=>'<section class="summary-card"><h4>'+r.policy+'</h4><p>'+policyRead(r)+'</p><p><b>Recipe:</b> '+recipeRead(r)+'</p><p><b>Tradeoff:</b> '+tradeoffRead(r)+'</p><p><b>Optimality:</b> '+optimalityRead(r)+'</p></section>').join('')+
	 '</div>';
	}
	function drawScaleComparison(){
	 const rows=[];
	 for(const p of policies){
	  const byScale=runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants);
	  if(byScale.length<2) continue;
	  const low=byScale[0].aggregate, high=byScale[byScale.length-1].aggregate;
	  rows.push({policy:p,low,high,lowLabel:byScale[0].scenarioLabel,highLabel:byScale[byScale.length-1].scenarioLabel});
	 }
	 document.getElementById('scaleComparison').innerHTML='<table><thead><tr><th>policy</th><th>scale move</th><th>votes</th><th>bettors</th><th>fees</th><th>surplus</th><th>opinion err</th><th>prediction Brier</th><th>fraud</th><th>herding</th><th>liquidity use</th><th>whale share</th><th>executive read</th></tr></thead><tbody>'+
	 rows.map(x=>'<tr><td>'+x.policy+'</td><td>'+x.lowLabel+' -> '+x.highLabel+'</td><td>'+fmt(x.low.votes)+' -> '+fmt(x.high.votes)+'</td><td>'+fmt(x.low.bettors)+' -> '+fmt(x.high.bettors)+'</td><td>$'+fmt(x.low.feeRevenue)+' -> $'+fmt(x.high.feeRevenue)+'</td><td>$'+fmt(x.low.platformSurplus)+' -> $'+fmt(x.high.platformSurplus)+'</td><td>'+pct(x.low.avgOpinionSamplingError)+' -> '+pct(x.high.avgOpinionSamplingError)+'</td><td>'+fmt(x.low.avgPredictionBrierScore)+' -> '+fmt(x.high.avgPredictionBrierScore)+'</td><td>'+pct(x.low.fraudPressure)+' -> '+pct(x.high.fraudPressure)+'</td><td>'+pct(x.low.herdingIndex)+' -> '+pct(x.high.herdingIndex)+'</td><td>'+fmt(x.low.liquidityUtilization)+'x -> '+fmt(x.high.liquidityUtilization)+'x</td><td>'+pct(x.low.whaleTradeShare)+' -> '+pct(x.high.whaleTradeShare)+'</td><td style="text-align:left">'+scaleRead(x.low,x.high)+'</td></tr>').join('')+
	 '</tbody></table>';
	}
	function drawNonlinearityDiagnostics(){
	 const policy=DATA.mdp.fullPolicy||[];
	 const byKey=Object.fromEntries(policy.map(x=>[x.hotBin+'-'+x.ambBin+'-'+x.fatigueBin,x]));
	 const switches=[];
	 for(const s of policy){
	  for(const axis of ['hotBin','ambBin','fatigueBin']){
	   const n={hotBin:s.hotBin,ambBin:s.ambBin,fatigueBin:s.fatigueBin};
	   n[axis]+=1;
	   if(n[axis]>2) continue;
	   const other=byKey[n.hotBin+'-'+n.ambBin+'-'+n.fatigueBin];
	   if(other&&other.action!==s.action){
	    switches.push({from:s,to:other,axis,gap:Math.min(s.qGap,other.qGap)});
	   }
	  }
	 }
	 switches.sort((a,b)=>a.gap-b.gap);
	 const fragile=[...policy].sort((a,b)=>a.qGap-b.qGap).slice(0,6);
	 const scaleRows=[];
	 for(const p of policies){
	  const rs=runs.filter(r=>r.policy===p).sort((a,b)=>a.minMarketParticipants-b.minMarketParticipants);
	  if(rs.length<2) continue;
	  const low=rs[0].aggregate, high=rs[rs.length-1].aggregate;
	  scaleRows.push({
	   policy:p,
	   fees:elasticity(low.feeRevenue,high.feeRevenue,rs[0].minMarketParticipants,rs[rs.length-1].minMarketParticipants),
	   surplus:elasticity(Math.max(1,Math.abs(low.platformSurplus)),Math.max(1,Math.abs(high.platformSurplus)),rs[0].minMarketParticipants,rs[rs.length-1].minMarketParticipants),
	   fraud:high.fraudPressure-low.fraudPressure,
	   liq:high.liquidityUtilization-low.liquidityUtilization,
	  });
	 }
	 document.getElementById('nonlinear').innerHTML=
	  '<div class="summary-grid">'+
	  '<section class="summary-card"><h4>Policy Switch Boundaries</h4><p>Adjacent MDP states where one-bin changes flip the operator action. These are discovered nonlinear thresholds in the learned value surface.</p><p>'+switches.slice(0,6).map(x=>x.axis+': '+stateShort(x.from)+' '+x.from.action+' -> '+stateShort(x.to)+' '+x.to.action+' (Q gap '+fmt(x.gap)+')').join('<br>')+'</p></section>'+
	  '<section class="summary-card"><h4>Fragile Optima</h4><p>Small Q gaps mean small parameter changes can change the optimal policy.</p><p>'+fragile.map(x=>stateShort(x)+' -> '+x.action+' (Q gap '+fmt(x.qGap)+')').join('<br>')+'</p></section>'+
	  '<section class="summary-card"><h4>Scale Elasticities</h4><p>Elasticity > 1 means superlinear scale effects; < 1 means sublinear.</p><p>'+scaleRows.map(x=>x.policy+': fees '+fmt(x.fees)+'x, surplus magnitude '+fmt(x.surplus)+'x, fraud Δ '+pct(x.fraud)+', liquidity-use Δ '+fmt(x.liq)+'x').join('<br>')+'</p></section>'+
	  '<section class="summary-card"><h4>Where The POMDP Helps</h4><p>The POMDP discovers nonlinearities by tracking belief entropy/error and testing whether noisy observations cross an operator action boundary. The most important discoveries here are over/under becoming optimal after belief mass moves into hot/debatable states, and proof/delayed-vote recipes reducing fraud/herding enough to dominate at scale.</p></section>'+
	  '</div>';
	}
	function drawComparison(){
 const sorted=[...currentRuns()].sort((a,b)=>b.aggregate.engagementScore-a.aggregate.engagementScore);
 document.getElementById('comparison').innerHTML='<table><thead><tr><th>policy</th><th>markets</th><th>binary</th><th>scalar</th><th>o/u</th><th>avg fee</th><th>votes</th><th>bettors</th><th>fees</th><th>surplus</th><th>engagement</th><th>opinion err</th><th>prediction Brier</th><th>vote time</th><th>herding</th><th>fraud</th><th>operator belief error</th></tr></thead><tbody>'+
 sorted.map(r=>{const a=r.aggregate;return '<tr><td>'+r.policy+'</td><td>'+a.marketsClosed+'</td><td>'+a.binaryMarkets+'</td><td>'+a.scalarMarkets+'</td><td>'+a.thresholdMarkets+'</td><td>'+pct(a.avgFeeRate)+'</td><td>'+fmt(a.votes)+'</td><td>'+fmt(a.bettors)+'</td><td>$'+fmt(a.feeRevenue)+'</td><td>$'+fmt(a.platformSurplus)+'</td><td>'+fmt(a.engagementScore)+'</td><td>'+pct(a.avgOpinionSamplingError)+'</td><td>'+fmt(a.avgPredictionBrierScore)+'</td><td>'+pct(a.avgVoteTimeFraction)+'</td><td>'+pct(a.herdingIndex)+'</td><td>'+pct(a.fraudPressure)+'</td><td>'+(a.avgBeliefError===undefined?'':pct(a.avgBeliefError))+'</td></tr>'}).join('')+
 '</tbody></table>';
	}
	function policyRead(r){
	 const a=r.aggregate;
	 if(r.policy==='fixed-daily') return 'Conservative launch baseline: simple to operate, easy to explain, and useful as the reference floor.';
	 if(r.policy==='greedy-buzz') return 'Exploration and traffic-maximizing heuristic: chases observed buzz and tests many market designs.';
	 if(r.policy==='mdp-oracle') return 'Upper-bound operator benchmark: solves the MDP as if latent topic quality were directly visible.';
	 if(r.policy==='pomdp-belief') return 'Deployable learned operator: chooses actions from noisy buzz/ambiguity beliefs and updates after outcomes.';
	 return 'Policy run with '+fmt(a.marketsClosed)+' closed markets.';
	}
	function recipeRead(r){
	 const opened=r.actionCounts.filter(x=>x.action!=='wait');
	 const top=opened[0];
	 if(!top) return 'Mostly waits; no stable market-opening recipe emerged.';
	 const meta=actionMeta[top.action]||{};
	 const mix=opened.slice(0,3).map(x=>x.action+' x'+x.count).join(', ');
	 return 'Top recipe: '+top.action+' ('+contractLabelJS(meta)+', '+(meta.durationH||0)+'h, '+pct(meta.feeRate)+', '+(meta.informationMode||'no info mode')+'). Mix: '+mix+'.';
	}
	function tradeoffRead(r){
	 const a=r.aggregate;
	 const engagementRank=rankOf(r,'engagementScore',true);
	 const surplusRank=rankOf(r,'platformSurplus',true);
	 const fraudRank=rankOf(r,'fraudPressure',false);
	 const herdRank=rankOf(r,'herdingIndex',false);
	 const voterCost=a.votes>0?a.voterPoints/a.votes:0;
	 return 'Ranks #'+engagementRank+' engagement, #'+surplusRank+' surplus, #'+fraudRank+' fraud pressure, #'+herdRank+' herding. Reward cost is '+fmt(voterCost)+' points/voter; avg vote time is '+pct(a.avgVoteTimeFraction)+'.';
	}
	function optimalityRead(r){
	 const a=r.aggregate;
	 const tags=[];
	 if(isBest(r,'engagementScore',true)) tags.push('best engagement/traffic');
	 if(isBest(r,'feeRevenue',true)) tags.push('best gross fee revenue');
	 if(isBest(r,'platformSurplus',true)) tags.push('best surplus control');
	 if(isBest(r,'fraudPressure',false)) tags.push('lowest fraud pressure');
	 if(isBest(r,'herdingIndex',false)) tags.push('lowest herding');
	 if(isBest(r,'avgOpinionSamplingError',false)) tags.push('best opinion representativeness');
	 if(isBest(r,'avgPredictionBrierScore',false)) tags.push('best counterfactual prediction score');
	 if(isBest(r,'avgTraderBeliefError',false)) tags.push('best trader belief accuracy');
	 if(r.aggregate.avgBeliefError!==undefined && isBestDefined(r,'avgBeliefError',false)) tags.push('best deployable operator belief read');
	 if(tags.length===0) tags.push('benchmark or middle-ground comparison point');
	 return tags.join('; ')+'.';
	}
	function rankOf(run,metric,higher){
	 const sorted=[...currentRuns()].sort((x,y)=>higher?y.aggregate[metric]-x.aggregate[metric]:x.aggregate[metric]-y.aggregate[metric]);
	 return sorted.findIndex(x=>x.policy===run.policy)+1;
	}
	function isBest(run,metric,higher){
	 return rankOf(run,metric,higher)===1;
	}
	function isBestDefined(run,metric,higher){
	 const filtered=currentRuns().filter(x=>x.aggregate[metric]!==undefined);
	 const sorted=filtered.sort((x,y)=>higher?y.aggregate[metric]-x.aggregate[metric]:x.aggregate[metric]-y.aggregate[metric]);
	 return sorted[0]&&sorted[0].policy===run.policy;
	}
	function scaleRead(low,high){
	 const feeMultiple=high.feeRevenue/Math.max(1,low.feeRevenue);
	 const fraudDelta=high.fraudPressure-low.fraudPressure;
	 const liqDelta=high.liquidityUtilization-low.liquidityUtilization;
	 const parts=['fees scale '+fmt(feeMultiple)+'x'];
	 parts.push(fraudDelta>0.002?'fraud pressure rises':fraudDelta<-0.002?'fraud pressure improves':'fraud pressure is stable');
	 parts.push(liqDelta>1?'liquidity gets stressed':liqDelta<-1?'liquidity headroom improves':'liquidity load is stable');
	 if(high.avgOpinionSamplingError<low.avgOpinionSamplingError-0.002) parts.push('opinion sampling improves');
	 if(high.avgPredictionBrierScore>low.avgPredictionBrierScore+0.02) parts.push('external prediction score worsens');
	 if(high.whaleTradeShare>low.whaleTradeShare+0.02) parts.push('whale concentration increases');
	 if(high.referralAdds>low.referralAdds*5) parts.push('referral loop becomes material');
	 return parts.join('; ')+'.';
	}
	function elasticity(low,high,lowScale,highScale){
	 if(low<=0||high<=0||lowScale<=0||highScale<=0||lowScale===highScale) return 0;
	 return Math.log(high/low)/Math.log(highScale/lowScale);
	}
	function stateShort(s){return 'h'+s.hotBin+'/a'+s.ambBin+'/f'+s.fatigueBin;}
	function drawLevers(r){
 const rows=r.actionCounts.slice(0,12);
 document.getElementById('levers').innerHTML='<table><thead><tr><th>action</th><th>count</th><th>contract</th><th>duration</th><th>fee</th><th>liquidity</th><th>rewards</th><th>verification</th><th>info</th><th>decay</th><th>description</th></tr></thead><tbody>'+
 rows.map(x=>{const a=actionMeta[x.action]||{};return '<tr><td>'+x.action+'</td><td>'+x.count+'</td><td>'+contractLabelJS(a)+'</td><td>'+(a.durationH||0)+'h</td><td>'+pct(a.feeRate)+'</td><td>'+fmt((DATA.cfg.liquidity||0)*(a.liquidityMultiplier||1))+'</td><td>'+fmt(a.rewardMultiplier||1)+'x</td><td>'+(a.verification||'')+'</td><td>'+(a.informationMode||'')+'</td><td>'+fmt(a.timingDecay||0)+'</td><td style="text-align:left">'+(a.description||'')+'</td></tr>'}).join('')+
 '</tbody></table>';
}
function drawAgents(r){
 const a=r.aggregate;
 const rows=[
  ['Platform operator','MDP/POMDP policy learner','Chooses topic timing, contract, duration, fee, liquidity, rewards, verification, and information reveal policy.','engagement '+fmt(a.engagementScore)+', surplus $'+fmt(a.platformSurplus)],
  ['Voters','Partial-observation timing/accuracy model','Must vote before betting; trade off early exponential multiplier against more public information later.','avg vote time '+pct(a.avgVoteTimeFraction)+', opinion sampling error '+pct(a.avgOpinionSamplingError)],
  ['Bettors','Partial-observation trader model','Observe noisy public/private signals, prices, and information mode; choose whether/what to buy after voting. Some fraction mistakenly trades as though the opinion market were an external prediction market.','bettors '+fmt(a.bettors)+', prediction Brier '+fmt(a.avgPredictionBrierScore)+', herding '+pct(a.herdingIndex)],
  ['Market maker','LMSR liquidity/risk model','Quotes binary, scalar, and over/under contracts with configurable b and bounded loss.','risk bound $'+fmt(a.marketMakerRiskBound)+', price gap '+pct(a.priceOpinionGap)],
  ['Trust & safety','Fraud-pressure response model','Verification tier reduces suspected Sybil vote pressure but also lowers participation.','suspect votes '+fmt(a.suspectedSybilVotes)+', fraud pressure '+pct(a.fraudPressure)]
 ];
 document.getElementById('agents').innerHTML='<table><thead><tr><th>layer</th><th>model</th><th>role in operator transition/reward</th><th>current selected policy signal</th></tr></thead><tbody>'+
 rows.map(row=>'<tr><td>'+row[0]+'</td><td>'+row[1]+'</td><td style="text-align:left">'+row[2]+'</td><td style="text-align:left">'+row[3]+'</td></tr>').join('')+
 '</tbody></table>';
}
function drawMix(r){
 const markets=[...r.closedMarkets].slice(-16).reverse();
 document.getElementById('mix').innerHTML='<table><thead><tr><th>market</th><th>cat</th><th>contract</th><th>dur</th><th>info</th><th>fee</th><th>liq</th><th>reward</th><th>verify</th><th>votes</th><th>bettors</th><th>vote time</th><th>herding</th><th>suspect</th><th>fees</th><th>opinion θ</th><th>final vote</th><th>event p</th><th>event</th><th>opinion err</th><th>Brier</th></tr></thead><tbody>'+
 markets.map(m=>'<tr><td>#'+m.id+'</td><td>'+m.topic.category+'</td><td>'+m.contractLabel+'</td><td>'+m.durationH+'h</td><td>'+m.informationMode+'</td><td>'+pct(m.feeRate)+'</td><td>'+fmt(m.liquidity)+'</td><td>'+fmt(m.rewardMultiplier)+'x</td><td>'+m.verification+'</td><td>'+m.votes+'</td><td>'+m.bettors+'</td><td>'+pct(m.avgVoteTimeFraction)+'</td><td>'+pct(m.herdingIndex)+'</td><td>'+m.suspectedSybilVotes+'</td><td>$'+fmt(m.feeRevenue)+'</td><td>'+(100*m.topic.trueTheta).toFixed(0)+'%</td><td>'+(100*m.finalVoteFraction).toFixed(0)+'%</td><td>'+(100*m.topic.eventProbability).toFixed(0)+'%</td><td>'+(m.externalOutcome?'yes':'no')+'</td><td>'+pct(m.opinionSamplingError)+'</td><td>'+fmt(m.predictionBrierScore)+'</td></tr>').join('')+
 '</tbody></table>';
}
function drawNotes(r){
 const topActions=r.actionCounts.slice(0,5).map(x=>x.action+'='+x.count).join(', ');
 let s='Operator MDP: '+DATA.mdp.numStates+' states, '+DATA.mdp.actions.length+' action recipes, gamma='+DATA.mdp.gamma+'. Top scheduler actions for this policy: '+topActions+'. ';
 s+='Daily launch capacity is explicit in the state trace: min '+DATA.cfg.minDailyMarkets+', max '+DATA.cfg.maxDailyMarkets+', horizon '+(DATA.cfg.horizonH/24)+' days. ';
 s+='Each action recipe combines open/wait scheduling, contract type, duration, fee rate, LMSR liquidity, reward multiplier, verification tier, information visibility, and exponential timing decay. ';
 s+='Bettor/voter motives are not the top-level learned policy here; they are behavioral submodels that generate transition and reward signals for the operator policy. ';
 s+='Opinion accuracy is measured as sampling error against latent public opinion; prediction accuracy is measured as a counterfactual Brier score against a latent external event outcome. ';
 if(r.policy==='mdp-oracle') s+='The MDP oracle sees latent hotness and ambiguity before opening a market; this is a useful upper-bound, not a deployable assumption.';
 if(r.policy==='pomdp-belief') s+='The POMDP scheduler sees noisy buzz/ambiguity, maintains category-level beliefs, and updates them after market resolution. Its belief error/entropy are reported in the comparison table.';
 if(r.policy==='fixed-daily') s+='Fixed daily approximates the initial one-market-every-24h operating mode.';
 if(r.policy==='greedy-buzz') s+='Greedy buzz is a transparent heuristic baseline: open high observed-buzz topics and choose scalar when ambiguity looks high.';
 document.getElementById('notes').textContent=s;
}
function marketsUntil(r,t){return r.closedMarkets.filter(m=>m.closeAt<=t+1e-9)}
function metricLabel(metric){const all=contractMetrics.concat(scaleMetrics);const row=all.find(x=>x[0]===metric);return row?row[1]:metric}
function formatMetric(metric,value){if(['avgPredictionError','avgOpinionSamplingError','herdingIndex','fraudPressure','avgTraderBeliefError','whaleTradeShare'].includes(metric))return pct(value);if(['feeRevenue','platformSurplus','fees'].includes(metric))return money(value);return fmt(value)}
function clear(svg){while(svg.firstChild)svg.removeChild(svg.firstChild)}
function axes(svg,W,H,pad,xLabel,yLabel){line(svg,pad,H-pad,W-pad,H-pad,'#aaa');line(svg,pad,pad,pad,H-pad,'#aaa');text(svg,W-pad,H-10,xLabel,'#666',10,'normal','end');text(svg,10,pad,yLabel,'#666',10)}
function scaleX(x,max,W,pad){return pad+(W-pad*2)*(x/Math.max(1,max))}
function scaleY(y,max,H,pad){return H-pad-(H-pad*2)*(y/Math.max(1,max))}
function legend(svg,x,y,items){let dx=0;for(const it of items){rect(svg,x+dx,y-10,10,10,it[1],null);text(svg,x+dx+14,y,it[0],'#444',11);dx+=it[0].length*7+42}}
function line(svg,x1,y1,x2,y2,c){const e=document.createElementNS('http://www.w3.org/2000/svg','line');e.setAttribute('x1',String(x1));e.setAttribute('y1',String(y1));e.setAttribute('x2',String(x2));e.setAttribute('y2',String(y2));e.setAttribute('stroke',c);e.setAttribute('stroke-width','1.5');svg.appendChild(e)}
function rect(svg,x,y,w,h,fill,stroke,opacity){const e=document.createElementNS('http://www.w3.org/2000/svg','rect');e.setAttribute('x',String(x));e.setAttribute('y',String(y));e.setAttribute('width',String(Math.max(0,w)));e.setAttribute('height',String(Math.max(0,h)));e.setAttribute('fill',fill||'none');if(stroke)e.setAttribute('stroke',stroke);if(opacity!==undefined)e.setAttribute('opacity',String(opacity));svg.appendChild(e)}
function path(svg,pts,c,width){if(pts.length===0)return;const e=document.createElementNS('http://www.w3.org/2000/svg','path');e.setAttribute('d',pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' '));e.setAttribute('fill','none');e.setAttribute('stroke',c);e.setAttribute('stroke-width',String(width||2));svg.appendChild(e)}
function text(svg,x,y,s,c,size,weight,anchor){const e=document.createElementNS('http://www.w3.org/2000/svg','text');e.setAttribute('x',String(x));e.setAttribute('y',String(y));e.setAttribute('font-size',String(size||12));e.setAttribute('fill',c);if(weight)e.setAttribute('font-weight',weight);if(anchor)e.setAttribute('text-anchor',anchor);e.textContent=s;svg.appendChild(e)}
function contractLabelJS(a){if(a.kind==='threshold')return 'over/under '+Math.round((a.threshold||0.55)*100)+'%';if(a.kind==='scalar')return 'scalar distribution';if(a.kind==='binary')return 'majority binary';return 'wait'}
render();
</script>
</body>
</html>`;
}

function summarizeOperatorMDP(mdp: OperatorMDP) {
  const fullPolicy = Array.from({length: mdp.spec.numStates}, (_, s) => {
    const qs = mdp.q[s].slice().sort((a, b) => b - a);
    const d = decodeOperatorState(s);
    return {
      state: mdp.stateLabel(s),
      hotBin: d.hotBin,
      ambBin: d.ambBin,
      fatigueBin: d.fatigueBin,
      action: mdp.actions[mdp.policy[s]].label,
      value: mdp.V[s],
      qGap: qs.length >= 2 ? qs[0] - qs[1] : 0,
      bestQ: qs[0] ?? 0,
      secondQ: qs[1] ?? 0,
    };
  });
  return {
    numStates: mdp.spec.numStates,
    actions: mdp.actions,
    iterations: mdp.iterations,
    finalDelta: mdp.finalDelta,
    gamma: mdp.gamma,
    fullPolicy,
    samplePolicy: fullPolicy.slice(0, 12),
  };
}

export function buildDailyMarketCaps(
  horizonDays: number,
  minDailyMarkets: number,
  maxDailyMarkets: number,
  seed: number,
): number[] {
  const days = Math.max(1, Math.ceil(horizonDays));
  const lo = Math.max(0, Math.floor(minDailyMarkets));
  const hi = Math.max(lo, Math.floor(maxDailyMarkets));
  if (lo === hi) return new Array(days).fill(lo);
  const rng = mulberry32(seed + 0x9e3779b9);
  const caps: number[] = [];
  for (let day = 0; day < days; day++) {
    const weeklyPulse = 0.5 + 0.5 * Math.sin((2 * Math.PI * (day + (seed % 7))) / 7);
    const newsShock = rng();
    const weekendDrag = day % 7 === 5 || day % 7 === 6 ? -0.12 : 0.04;
    const level = clamp(0.52 * weeklyPulse + 0.40 * newsShock + weekendDrag, 0, 1);
    caps.push(Math.round(lo + (hi - lo) * level));
  }
  caps[0] = lo;
  if (days > 1) caps[Math.floor(days * 0.62)] = hi;
  return caps;
}

export function dailyMarketCapForDay(day: number, cfg: PortfolioConfig): number {
  if (cfg.dailyMarketCaps.length === 0) return cfg.maxDailyMarkets;
  const idx = clamp(Math.floor(day), 0, cfg.dailyMarketCaps.length - 1);
  return cfg.dailyMarketCaps[idx];
}

export function dayIndex(tHours: number): number {
  return Math.max(0, Math.floor(tHours / 24));
}

function parseDailyMarketCaps(raw: string | undefined, horizonDays: number): number[] | null {
  if (!raw) return null;
  const parsed = raw.split(',')
    .map(x => Math.floor(Number(x.trim())))
    .filter(x => Number.isFinite(x) && x >= 0);
  if (parsed.length === 0) return null;
  while (parsed.length < horizonDays) parsed.push(parsed[parsed.length - 1]);
  return parsed.slice(0, horizonDays);
}

export function defaultConfig(): PortfolioConfig {
  const minMarketParticipants = Number(process.env.MIN_MARKET_PARTICIPANTS ?? 1000);
  const horizonH = Number(process.env.HORIZON_H ?? 24 * 50);
  const horizonDays = Math.ceil(horizonH / 24);
  const seed = Number(process.env.SEED ?? 42);
  const minDailyMarkets = Math.max(0, Math.floor(Number(process.env.MIN_DAILY_MARKETS ?? 2)));
  const maxDailyMarkets = Math.max(minDailyMarkets, Math.floor(Number(process.env.MAX_DAILY_MARKETS ?? 10)));
  const dailyMarketCaps = parseDailyMarketCaps(process.env.DAILY_MARKET_CAPS, horizonDays)
    ?? buildDailyMarketCaps(horizonDays, minDailyMarkets, maxDailyMarkets, seed);
  return {
    scenarioLabel: process.env.SCENARIO_LABEL ?? `${minMarketParticipants.toLocaleString()} participants`,
    horizonH,
    stepH: Number(process.env.STEP_H ?? 0.25),
    maxConcurrent: Number(process.env.MAX_CONCURRENT ?? maxDailyMarkets),
    minDailyMarkets,
    maxDailyMarkets,
    dailyMarketCaps,
    seed,
    liquidity: Number(process.env.LIQUIDITY ?? 500),
    feeRate: Number(process.env.FEE_RATE ?? 0.01),
    scalarBins: Number(process.env.SCALAR_BINS ?? 7),
    minMarketParticipants,
  };
}

export function scenarioConfigs(base: PortfolioConfig): PortfolioConfig[] {
  const raw = process.env.PARTICIPANT_SCALES;
  const scales = raw
    ? raw.split(',').map(x => Number(x.trim())).filter(x => Number.isFinite(x) && x > 0)
    : [1000, 10000];
  return scales.map(scale => ({
    ...base,
    scenarioLabel: `${scale.toLocaleString()} participants`,
    minMarketParticipants: scale,
  }));
}

function main(): void {
  const cfg = defaultConfig();
  const scenarios = scenarioConfigs(cfg);
  const mdp = buildOperatorMDP();
  const policies: SchedulerPolicy[] = ['fixed-daily', 'greedy-buzz', 'mdp-oracle', 'pomdp-belief'];
  const runs = scenarios.flatMap(scenario => policies.map(p => runPortfolio(p, scenario, mdp)));
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const htmlPath = path.join(outDir, 'factmachine-markets.html');
  const jsonPath = path.join(outDir, 'factmachine-markets-results.json');
  fs.writeFileSync(htmlPath, buildHTML(runs, mdp, cfg));
  fs.writeFileSync(jsonPath, JSON.stringify({config: cfg, mdp: summarizeOperatorMDP(mdp), runs}, null, 2));

  console.log('# FactMachine multi-market simulation');
  console.log(`#   horizon=${cfg.horizonH}h, maxConcurrent=${cfg.maxConcurrent}, scenarios=${scenarios.map(s => s.minMarketParticipants).join(',')}`);
  console.log(`#   operator MDP: ${mdp.spec.numStates} states, ${mdp.actions.length} actions, ${mdp.iterations} VI sweeps`);
  for (const r of runs) {
    const a = r.aggregate;
    console.log('');
    console.log(`# ${r.scenarioLabel} / ${r.policy}`);
    console.log(`#   markets ${a.marketsClosed} (${a.binaryMarkets} binary, ${a.scalarMarkets} scalar, ${a.thresholdMarkets} over/under), avg duration ${a.avgDurationH.toFixed(2)}h`);
    console.log(`#   avg fee ${(100 * a.avgFeeRate).toFixed(2)}%, avg liquidity $${a.avgLiquidity.toFixed(0)}, avg reward ${a.avgRewardMultiplier.toFixed(2)}x, timing decay ${a.avgTimingDecay.toFixed(2)}, proof markets ${a.proofMarkets}`);
    console.log(`#   votes ${a.votes}, suspect votes ${a.suspectedSybilVotes}, bettors ${a.bettors}, trades ${a.trades}, raffle entries ${a.raffleEntries}`);
    console.log(`#   user submodels: avg vote time ${(100 * a.avgVoteTimeFraction).toFixed(1)}%, timing boost ${a.avgTimingMultiplier.toFixed(2)}x, trader belief error ${(100 * a.avgTraderBeliefError).toFixed(1)}%, herding ${(100 * a.herdingIndex).toFixed(1)}%`);
    console.log(`#   opinion vs prediction: opinion sampling error ${(100 * a.avgOpinionSamplingError).toFixed(1)}%, prediction Brier ${a.avgPredictionBrierScore.toFixed(3)}, price/opinion gap ${(100 * a.priceOpinionGap).toFixed(1)}%`);
    console.log(`#   fees $${a.feeRevenue.toFixed(2)}, voter points ${a.voterPoints.toFixed(1)}, surplus $${a.platformSurplus.toFixed(2)}, engagement ${a.engagementScore.toFixed(1)}`);
    if (a.avgBeliefError !== undefined) {
      console.log(`#   POMDP belief entropy ${a.avgBeliefEntropy!.toFixed(3)}, hotness error ${(100 * a.avgBeliefError).toFixed(1)}%`);
    }
  }
  console.log('');
  console.log(`# wrote ${htmlPath}`);
  console.log(`# wrote ${jsonPath}`);
}

function encodeOperatorState(hotBin: number, ambBin: number, fatigueBin: number): number {
  return (hotBin * 3 + ambBin) * 3 + fatigueBin;
}
function decodeOperatorState(s: number): {hotBin: number; ambBin: number; fatigueBin: number} {
  const fatigueBin = s % 3; s = Math.floor(s / 3);
  const ambBin = s % 3; s = Math.floor(s / 3);
  return {hotBin: s, ambBin, fatigueBin};
}
function operatorStateLabel(s: number): string {
  const d = decodeOperatorState(s);
  return `hot=${d.hotBin}/amb=${d.ambBin}/fatigue=${d.fatigueBin}`;
}
function fatigueBinFor(active: number, cap: number): number {
  const x = cap <= 0 ? 1 : active / cap;
  return x < 0.34 ? 0 : x < 0.67 ? 1 : 2;
}
function waitAction(): SchedulerAction {
  return {
    label: 'wait',
    kind: 'wait',
    durationH: 0,
    feeRate: 0,
    liquidityMultiplier: 1,
    rewardMultiplier: 1,
    verification: 'basic',
    informationMode: 'price-only',
    timingDecay: 1,
    description: 'do not open a new market this step',
  };
}
function marketAction(
  label: string,
  kind: MarketKind,
  durationH: number,
  opts: {
    feeRate: number;
    liquidityMultiplier: number;
    rewardMultiplier: number;
    verification: VerificationTier;
    informationMode: InformationMode;
    timingDecay: number;
    threshold?: number;
    description: string;
  },
): SchedulerAction {
  return {label, kind, durationH, ...opts};
}
function actionBy(actions: SchedulerAction[], kind: MarketKind, durationH: number, labelHint?: string): SchedulerAction {
  return actions.find(a => a.kind === kind && a.durationH === durationH && (!labelHint || a.label.includes(labelHint)))
    ?? actions.find(a => a.kind === kind && a.durationH === durationH)
    ?? actions[0];
}
function contractLabel(action: SchedulerAction): string {
  if (action.kind === 'threshold') return `over/under ${Math.round((action.threshold ?? 0.55) * 100)}%`;
  if (action.kind === 'scalar') return 'scalar distribution';
  if (action.kind === 'binary') return 'majority binary';
  return 'wait';
}
function scalarBinMidpoint(i: number, bins: number): number {
  return clamp((i + 0.5) / Math.max(1, bins), 0, 1);
}
function verificationParticipationMultiplier(tier: VerificationTier): number {
  return tier === 'proof' ? 0.84 : tier === 'basic' ? 0.96 : 1.08;
}
function verificationTrustMultiplier(tier: VerificationTier): number {
  return tier === 'proof' ? 1.18 : tier === 'basic' ? 1.0 : 0.88;
}
function manipulationMultiplier(tier: VerificationTier): number {
  return tier === 'proof' ? 0.32 : tier === 'basic' ? 0.62 : 1.0;
}
function marketTraderFit(kind: MarketKind): number {
  return kind === 'binary' ? 1.08 : kind === 'threshold' ? 1.14 : 0.88;
}
function thresholdDrama(theta: number, threshold: number): number {
  return Math.exp(-Math.pow(theta - threshold, 2) / 0.045);
}
function informationEngagementMultiplier(mode: InformationMode): number {
  return ({
    'price-only': 0.88,
    'delayed-votes': 0.98,
    'live-votes': 1.12,
    'demographic-slices': 1.08,
    'momentum-signals': 1.18,
  })[mode];
}
function informationTrustMultiplier(mode: InformationMode): number {
  return ({
    'price-only': 0.88,
    'delayed-votes': 1.04,
    'live-votes': 0.95,
    'demographic-slices': 1.15,
    'momentum-signals': 0.92,
  })[mode];
}
function informationHerdingMultiplier(mode: InformationMode): number {
  return ({
    'price-only': 0.55,
    'delayed-votes': 0.72,
    'live-votes': 1.15,
    'demographic-slices': 0.95,
    'momentum-signals': 1.35,
  })[mode];
}
function informationSignalWeight(mode: InformationMode): number {
  return ({
    'price-only': 0.18,
    'delayed-votes': 0.34,
    'live-votes': 0.54,
    'demographic-slices': 0.48,
    'momentum-signals': 0.62,
  })[mode];
}
function informationWaitPressure(mode: InformationMode): number {
  return ({
    'price-only': 0.15,
    'delayed-votes': 0.34,
    'live-votes': 0.62,
    'demographic-slices': 0.48,
    'momentum-signals': 0.55,
  })[mode];
}
function informationObservationNoise(mode: InformationMode, ambiguity: number): number {
  const base = ({
    'price-only': 0.16,
    'delayed-votes': 0.11,
    'live-votes': 0.08,
    'demographic-slices': 0.07,
    'momentum-signals': 0.10,
  })[mode];
  return base + ambiguity * 0.05;
}
function bernoulliEntropy(p: number): number {
  const q = clamp(p, 1e-9, 1 - 1e-9);
  return -(q * Math.log(q) + (1 - q) * Math.log(1 - q));
}
function bin3(x: number): number {
  return x < 0.34 ? 0 : x < 0.67 ? 1 : 2;
}
function hotnessMidpoint(bin: number): number {
  return [0.20, 0.52, 0.84][bin] ?? 0.52;
}
function ambiguityMidpoint(bin: number): number {
  return [0.18, 0.50, 0.82][bin] ?? 0.5;
}
function betaLike(rng: () => number, a: number, b: number): number {
  const x = -Math.log(Math.max(1e-12, rng())) / a;
  const y = -Math.log(Math.max(1e-12, rng())) / b;
  return x / (x + y);
}
function samplePoisson(lambda: number, rng: () => number): number {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}
function expSample(rate: number, rng: () => number): number {
  return -Math.log(Math.max(1e-12, 1 - rng())) / Math.max(1e-12, rate);
}
function normal(rng: () => number): number {
  const u = Math.max(1e-12, rng());
  const v = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function categorical(weights: number[], rng: () => number): number {
  const total = sum(weights);
  let u = rng() * total;
  for (let i = 0; i < weights.length; i++) {
    u -= weights[i];
    if (u <= 0) return i;
  }
  return weights.length - 1;
}
function softmax(scores: number[]): number[] {
  const m = Math.max(...scores);
  const exps = scores.map(x => Math.exp(x - m));
  const z = sum(exps);
  return exps.map(x => x / z);
}
function entropy(ps: number[]): number {
  return -ps.reduce((s, p) => p > 0 ? s + p * Math.log(p) : s, 0);
}
function argMax(xs: number[]): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}
function weightedMean(rows: Array<[number, number]>): number {
  const w = rows.reduce((s, r) => s + r[1], 0);
  return w === 0 ? 0 : rows.reduce((s, r) => s + r[0] * r[1], 0) / w;
}
function sum(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0);
}
function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : sum(xs) / xs.length;
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
function bump<K>(m: Map<K, number>, key: K, by = 1): void {
  m.set(key, (m.get(key) ?? 0) + by);
}
function policySeed(policy: SchedulerPolicy): number {
  return {'fixed-daily': 10, 'greedy-buzz': 20, 'mdp-oracle': 30, 'pomdp-belief': 40}[policy];
}

if (require.main === module) main();
