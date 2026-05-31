'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/collaborative-inference.rs  (module des::general::collaborative_inference)
// 1:1 file move. Sparse subjective preference learning (ratings+pairwise -> global rank) as a DES station graph.
//
// Declarations → Rust:
//   type CollaborativeInferenceScenario = ... -> enum
//   interface CollaborativeInference{Item,Response,Params,Coverage,Result}/CollaborativeItemScore/
//             CredibilityWeightSummary/ScenarioPreset/NormalizedConfig/... -> structs
//   class RespondentToken/RatingEvidenceToken/PairwisePreferenceToken/EvidenceSnapshotToken/RankingToken
//                                    -> structs `impl Token`
//   class RespondentSource/SurveyEncoder/EvidenceAggregator/RankingInference/InferenceResultSink Station
//                                    -> structs `impl` the station trait
//   fn runCollaborativeInference   -> fn (or StatefulTransform)
//
// Conversion notes (file-specific):
//   - HEAVY `Map`/`Set` usage (evidence keyed by item id, pairwise counts) -> `HashMap`/`HashSet`
//     (item-id keys need `Hash + Eq`; iteration order is N/A — sort explicitly where output order matters).
//   - `CollaborativeInferenceScenario` string union -> enum; empirical-Bayes shrinkage math stays f64.
//   - Tokens/stations are nominal `impl Trait`; channels -> typed queues; `Preconditions` throw -> `Result`.
// =============================================================================

// =============================================================================
// general/collaborative-inference.ts
//
// Sparse subjective preference learning as a DES station graph.
//
// Model idea
// ----------
// A population knows only small overlapping subsets of a large option set. Each
// respondent rates and ranks the subset they have actually experienced. The
// station graph converts those local opinions into rating evidence and pairwise
// preference evidence, then infers a global ranking with empirical-Bayes
// shrinkage so lightly observed items do not jump to the top on one lucky vote.
//
// DES mapping
// -----------
//   RespondentSource      source station for respondent movables
//   SurveyEncoder         station that turns one respondent into evidence
//   EvidenceAggregator    station that accumulates rating and pairwise tokens
//   RankingInference      station that converts evidence snapshots to scores
//   InferenceResultSink   sink station for the final inferred ranking
//
// Movables:
//   RespondentToken, RatingEvidenceToken, PairwisePreferenceToken,
//   EvidenceSnapshotToken, RankingToken.
// =============================================================================

import {
  ChannelName,
  DESStation,
  StationGraphSummary,
  Token,
  ValidationCheck,
  channelEdge,
  runIterativeDES,
  stationGraph,
} from './des-base';
import {mulberry32} from './prng';

export type CollaborativeInferenceScenario =
  | 'programming-languages'
  | 'model-validation'
  | 'learning-resources'
  | 'custom';

export interface CollaborativeInferenceItem {
  id: string;
  label?: string;
  group?: string;
  /** Used only by synthetic scenario generation, never by the inference step. */
  latentUtility?: number;
  /** Relative chance that a synthetic respondent has experience with this item. */
  exposure?: number;
  /** Optional prior score in [0, 1] used when evidence is sparse. */
  priorScore?: number;
}

export interface CollaborativeInferenceResponse {
  id?: string;
  itemIds?: string[];
  ratings?: Record<string, number>;
  ranking?: string[];
  /** Respondent age. Used to cap impossible experience claims. */
  age?: number;
  /** Per-item claimed years of experience, keyed by item id. */
  experienceYears?: Record<string, number>;
  weight?: number;
  segment?: string;
}

export interface CollaborativeInferenceParams {
  scenario?: CollaborativeInferenceScenario;
  items?: CollaborativeInferenceItem[];
  responses?: CollaborativeInferenceResponse[];
  respondentCount?: number;
  /** Alias accepted for natural JSON specs. */
  respondents?: number;
  minItemsPerRespondent?: number;
  maxItemsPerRespondent?: number;
  respondentsPerTick?: number;
  ratingMin?: number;
  ratingMax?: number;
  noiseStd?: number;
  seed?: number;
  ratingWeight?: number;
  rankingWeight?: number;
  shrinkage?: number;
  topK?: number;
  credibilityWeighting?: boolean;
  credibilityPasses?: number;
  minCredibleAge?: number;
  referenceAge?: number;
  referenceExperienceYears?: number;
  ageWeightStrength?: number;
  experienceWeightStrength?: number;
  highRatedBreadthStrength?: number;
  highRatedScoreThreshold?: number;
  minHighRatedItems?: number;
  maxCredibilityMultiplier?: number;
}

interface ScenarioPreset {
  scenario: CollaborativeInferenceScenario;
  label: string;
  defaultRespondents: number;
  minItemsPerRespondent: number;
  maxItemsPerRespondent: number;
  ratingMin: number;
  ratingMax: number;
  noiseStd: number;
  items: CollaborativeInferenceItem[];
}

interface NormalizedConfig {
  scenario: CollaborativeInferenceScenario;
  scenarioLabel: string;
  items: CollaborativeInferenceItem[];
  itemById: Map<string, CollaborativeInferenceItem>;
  responses: CollaborativeInferenceResponse[];
  respondentCount: number;
  minItemsPerRespondent: number;
  maxItemsPerRespondent: number;
  respondentsPerTick: number;
  ratingMin: number;
  ratingMax: number;
  noiseStd: number;
  seed: number;
  ratingWeight: number;
  rankingWeight: number;
  shrinkage: number;
  topK: number;
  synthetic: boolean;
  credibility: CredibilityWeightingConfig;
}

interface CredibilityWeightingConfig {
  enabled: boolean;
  passes: number;
  minCredibleAge: number;
  referenceAge: number;
  referenceExperienceYears: number;
  ageWeightStrength: number;
  experienceWeightStrength: number;
  highRatedBreadthStrength: number;
  highRatedScoreThreshold: number;
  minHighRatedItems: number;
  maxMultiplier: number;
}

export interface CredibilityWeightSummary {
  enabled: boolean;
  passes: number;
  minCredibleAge: number;
  highRatedScoreThreshold: number;
  minHighRatedItems: number;
  meanRespondentWeight: number;
  maxRespondentWeight: number;
  cappedExperienceClaims: number;
  highRatedBonusRespondents: number;
}

interface ItemEvidenceStats {
  itemId: string;
  ratingSum: number;
  ratingWeight: number;
  ratingCount: number;
  pairwiseWins: number;
  pairwiseLosses: number;
}

export interface CollaborativeItemScore {
  rank: number;
  itemId: string;
  label: string;
  group?: string;
  score: number;
  confidence: number;
  uncertainty: number;
  ratingMean: number;
  ratingCount: number;
  comparisonCount: number;
  pairwiseWinRate: number;
  support: number;
}

export interface CollaborativeInferenceCoverage {
  items: number;
  itemsWithRatings: number;
  itemsWithComparisons: number;
  minRatingsPerItem: number;
  meanRatingsPerItem: number;
  maxRatingsPerItem: number;
  minComparisonsPerItem: number;
  meanComparisonsPerItem: number;
  maxComparisonsPerItem: number;
}

export interface CollaborativeInferenceResult {
  scenario: CollaborativeInferenceScenario;
  scenarioLabel: string;
  synthetic: boolean;
  respondentCount: number;
  respondentsProcessed: number;
  ratingEvidenceCount: number;
  pairwiseEvidenceCount: number;
  invalidEvidence: string[];
  credibility: CredibilityWeightSummary;
  coverage: CollaborativeInferenceCoverage;
  rankings: CollaborativeItemScore[];
  top: CollaborativeItemScore[];
  validation: ValidationCheck[];
  topology: StationGraphSummary;
  stationRoles: {
    sources: string[];
    stations: string[];
    sinks: string[];
    movables: string[];
  };
}

class RespondentToken implements Token {
  constructor(readonly response: CollaborativeInferenceResponse) {}
}

class RatingEvidenceToken implements Token {
  constructor(
    readonly respondentId: string,
    readonly itemId: string,
    readonly rating: number,
    readonly weight: number,
  ) {}
}

class PairwisePreferenceToken implements Token {
  constructor(
    readonly respondentId: string,
    readonly winnerId: string,
    readonly loserId: string,
    readonly weight: number,
  ) {}
}

class EvidenceSnapshotToken implements Token {
  constructor(
    readonly itemStats: ReadonlyMap<string, ItemEvidenceStats>,
    readonly respondentsProcessed: number,
    readonly ratingEvidenceCount: number,
    readonly pairwiseEvidenceCount: number,
  ) {}
}

class RankingToken implements Token {
  constructor(readonly rankings: CollaborativeItemScore[]) {}
}

interface RespondentWeightProfile {
  respondentWeight: number;
  itemWeights: Map<string, number>;
  highRatedItemCount: number;
  breadthMultiplier: number;
  cappedExperienceClaims: number;
}

class RespondentSourceStation extends DESStation {
  static readonly CH_RESPONDENT: ChannelName = 'respondent';
  emittedCount = 0;

  constructor(
    id: string,
    private readonly responses: readonly CollaborativeInferenceResponse[],
    private readonly respondentsPerTick: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.emittedCount < this.responses.length; }

  runTimeStep(): void {
    const n = Math.min(this.respondentsPerTick, this.responses.length - this.emittedCount);
    for (let k = 0; k < n; k++) {
      this.emit(new RespondentToken(this.responses[this.emittedCount]), RespondentSourceStation.CH_RESPONDENT);
      this.emittedCount += 1;
    }
  }
}

class SurveyEncoderStation extends DESStation {
  static readonly CH_RESPONDENT: ChannelName = RespondentSourceStation.CH_RESPONDENT;
  static readonly CH_RATING: ChannelName = 'rating-evidence';
  static readonly CH_PAIRWISE: ChannelName = 'pairwise-evidence';

  respondentsProcessed = 0;
  ratingEvidenceCount = 0;
  pairwiseEvidenceCount = 0;
  respondentWeightSum = 0;
  maxRespondentWeight = 0;
  cappedExperienceClaims = 0;
  highRatedBonusRespondents = 0;
  readonly invalidEvidence: string[] = [];

  constructor(
    id: string,
    private readonly validItemIds: ReadonlySet<string>,
    private readonly ratingMin: number,
    private readonly ratingMax: number,
    private readonly credibility: CredibilityWeightingConfig,
    private readonly preliminaryScores?: ReadonlyMap<string, number>,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(SurveyEncoderStation.CH_RESPONDENT) > 0; }

  runTimeStep(): void {
    const respondents = this.drain<RespondentToken>(SurveyEncoderStation.CH_RESPONDENT);
    for (const token of respondents) this.process(token.response);
  }

  private process(response: CollaborativeInferenceResponse): void {
    const respondentId = response.id ?? `respondent-${this.respondentsProcessed}`;
    const seen = new Set<string>();
    for (const itemId of response.itemIds ?? []) {
      if (this.validItemIds.has(itemId)) seen.add(itemId);
      else this.invalidEvidence.push(`${respondentId}: unknown item ${itemId}`);
    }
    for (const itemId of Object.keys(response.ratings ?? {})) if (this.validItemIds.has(itemId)) seen.add(itemId);
    for (const itemId of response.ranking ?? []) if (this.validItemIds.has(itemId)) seen.add(itemId);
    const profile = respondentWeightProfile(response, seen, this.credibility, this.preliminaryScores);
    this.respondentWeightSum += profile.respondentWeight;
    this.maxRespondentWeight = Math.max(this.maxRespondentWeight, profile.respondentWeight);
    this.cappedExperienceClaims += profile.cappedExperienceClaims;
    if (profile.breadthMultiplier > 1) this.highRatedBonusRespondents += 1;

    const ratingEntries = Object.entries(response.ratings ?? {});
    for (const [itemId, rawRating] of ratingEntries) {
      if (!this.validItemIds.has(itemId)) {
        this.invalidEvidence.push(`${respondentId}: rating references unknown item ${itemId}`);
        continue;
      }
      if (!Number.isFinite(rawRating) || rawRating < this.ratingMin || rawRating > this.ratingMax) {
        this.invalidEvidence.push(`${respondentId}: rating for ${itemId} outside [${this.ratingMin}, ${this.ratingMax}]`);
        continue;
      }
      seen.add(itemId);
      this.emit(new RatingEvidenceToken(respondentId, itemId, rawRating, profile.itemWeights.get(itemId) ?? profile.respondentWeight), SurveyEncoderStation.CH_RATING);
      this.ratingEvidenceCount += 1;
    }

    const ranking = this.validRanking(response, seen, respondentId);
    for (let i = 0; i < ranking.length; i++) {
      for (let j = i + 1; j < ranking.length; j++) {
        const wi = profile.itemWeights.get(ranking[i]) ?? profile.respondentWeight;
        const wj = profile.itemWeights.get(ranking[j]) ?? profile.respondentWeight;
        this.emit(new PairwisePreferenceToken(respondentId, ranking[i], ranking[j], (wi + wj) / 2), SurveyEncoderStation.CH_PAIRWISE);
        this.pairwiseEvidenceCount += 1;
      }
    }
    this.respondentsProcessed += 1;
  }

  private validRanking(response: CollaborativeInferenceResponse, seen: Set<string>, respondentId: string): string[] {
    const rawRanking = response.ranking && response.ranking.length > 0
      ? response.ranking
      : [...seen].sort((a, b) => {
          const ar = response.ratings?.[a] ?? 0;
          const br = response.ratings?.[b] ?? 0;
          return br - ar || a.localeCompare(b);
        });
    const out: string[] = [];
    const used = new Set<string>();
    for (const itemId of rawRanking) {
      if (!this.validItemIds.has(itemId)) {
        this.invalidEvidence.push(`${respondentId}: ranking references unknown item ${itemId}`);
        continue;
      }
      if (used.has(itemId)) continue;
      used.add(itemId);
      out.push(itemId);
    }
    return out;
  }
}

class EvidenceAggregatorStation extends DESStation {
  static readonly CH_RATING: ChannelName = SurveyEncoderStation.CH_RATING;
  static readonly CH_PAIRWISE: ChannelName = SurveyEncoderStation.CH_PAIRWISE;
  static readonly CH_SNAPSHOT: ChannelName = 'evidence-snapshot';

  readonly stats: Map<string, ItemEvidenceStats>;
  ratingEvidenceCount = 0;
  pairwiseEvidenceCount = 0;

  constructor(id: string, itemIds: readonly string[], private readonly survey: SurveyEncoderStation) {
    super(id);
    this.stats = new Map(itemIds.map(itemId => [itemId, {
      itemId,
      ratingSum: 0,
      ratingWeight: 0,
      ratingCount: 0,
      pairwiseWins: 0,
      pairwiseLosses: 0,
    }]));
  }

  override hasWork(): boolean {
    return this.inboxSize(EvidenceAggregatorStation.CH_RATING) > 0
        || this.inboxSize(EvidenceAggregatorStation.CH_PAIRWISE) > 0;
  }

  runTimeStep(): void {
    let changed = false;
    const ratings = this.drain<RatingEvidenceToken>(EvidenceAggregatorStation.CH_RATING);
    for (const rating of ratings) {
      const s = this.stats.get(rating.itemId);
      if (!s) continue;
      s.ratingSum += rating.rating * rating.weight;
      s.ratingWeight += rating.weight;
      s.ratingCount += 1;
      this.ratingEvidenceCount += 1;
      changed = true;
    }

    const comparisons = this.drain<PairwisePreferenceToken>(EvidenceAggregatorStation.CH_PAIRWISE);
    for (const cmp of comparisons) {
      const winner = this.stats.get(cmp.winnerId);
      const loser = this.stats.get(cmp.loserId);
      if (!winner || !loser) continue;
      winner.pairwiseWins += cmp.weight;
      loser.pairwiseLosses += cmp.weight;
      this.pairwiseEvidenceCount += 1;
      changed = true;
    }

    if (changed) {
      this.emit(new EvidenceSnapshotToken(
        cloneStats(this.stats),
        this.survey.respondentsProcessed,
        this.ratingEvidenceCount,
        this.pairwiseEvidenceCount,
      ), EvidenceAggregatorStation.CH_SNAPSHOT);
    }
  }
}

class RankingInferenceStation extends DESStation {
  static readonly CH_SNAPSHOT: ChannelName = EvidenceAggregatorStation.CH_SNAPSHOT;
  static readonly CH_RANKING: ChannelName = 'ranking';

  constructor(
    id: string,
    private readonly items: readonly CollaborativeInferenceItem[],
    private readonly ratingMin: number,
    private readonly ratingMax: number,
    private readonly ratingWeight: number,
    private readonly rankingWeight: number,
    private readonly shrinkage: number,
  ) {
    super(id);
  }

  override hasWork(): boolean { return this.inboxSize(RankingInferenceStation.CH_SNAPSHOT) > 0; }

  runTimeStep(): void {
    const snapshots = this.drain<EvidenceSnapshotToken>(RankingInferenceStation.CH_SNAPSHOT);
    if (snapshots.length === 0) return;
    const latest = snapshots[snapshots.length - 1];
    this.emit(new RankingToken(this.rank(latest)), RankingInferenceStation.CH_RANKING);
  }

  private rank(snapshot: EvidenceSnapshotToken): CollaborativeItemScore[] {
    const width = this.ratingMax - this.ratingMin;
    const midpoint = (this.ratingMin + this.ratingMax) / 2;

    const rows = this.items.map(item => {
      const s = snapshot.itemStats.get(item.id) ?? emptyStats(item.id);
      const ratingMean = s.ratingWeight > 0 ? s.ratingSum / s.ratingWeight : midpoint;
      const ratingScore = clamp01((ratingMean - this.ratingMin) / width);
      const comparisons = s.pairwiseWins + s.pairwiseLosses;
      const pairwiseWinRate = comparisons > 0 ? s.pairwiseWins / comparisons : 0.5;
      const pairwiseScore = (s.pairwiseWins + 0.5 * this.shrinkage) / (comparisons + this.shrinkage);
      const ratingConfidence = s.ratingCount / (s.ratingCount + this.shrinkage);
      const rankingConfidence = comparisons / (comparisons + this.shrinkage);
      const evidenceWeight = this.ratingWeight * ratingConfidence + this.rankingWeight * rankingConfidence;
      const priorScore = clamp01(item.priorScore ?? 0.5);
      const empirical = evidenceWeight > 0
        ? (this.ratingWeight * ratingConfidence * ratingScore + this.rankingWeight * rankingConfidence * pairwiseScore) / evidenceWeight
        : priorScore;
      const support = s.ratingCount + comparisons;
      const confidence = support / (support + this.shrinkage);
      const score = clamp01(confidence * empirical + (1 - confidence) * priorScore);
      const uncertainty = Math.sqrt(Math.max(0, score * (1 - score)) / Math.max(1, support + this.shrinkage));
      return {
        rank: 0,
        itemId: item.id,
        label: item.label ?? item.id,
        group: item.group,
        score,
        confidence,
        uncertainty,
        ratingMean,
        ratingCount: s.ratingCount,
        comparisonCount: comparisons,
        pairwiseWinRate,
        support,
      };
    });

    rows.sort((a, b) => b.score - a.score || b.confidence - a.confidence || a.label.localeCompare(b.label));
    rows.forEach((row, i) => row.rank = i + 1);
    return rows;
  }
}

class InferenceResultSinkStation extends DESStation {
  static readonly CH_RANKING: ChannelName = RankingInferenceStation.CH_RANKING;
  latest: RankingToken | undefined;

  constructor(id: string) { super(id); }

  override hasWork(): boolean { return this.inboxSize(InferenceResultSinkStation.CH_RANKING) > 0; }

  runTimeStep(): void {
    const rankings = this.drain<RankingToken>(InferenceResultSinkStation.CH_RANKING);
    if (rankings.length > 0) this.latest = rankings[rankings.length - 1];
  }
}

export function runCollaborativeInference(params: CollaborativeInferenceParams): CollaborativeInferenceResult {
  const cfg = normalizeCollaborativeInferenceParams(params);
  const passes = cfg.credibility.enabled ? cfg.credibility.passes : 1;
  let preliminaryScores: Map<string, number> | undefined;
  let finalRun: CollaborativeInferencePassResult | undefined;
  for (let pass = 0; pass < passes; pass++) {
    finalRun = runCollaborativeInferencePass(cfg, preliminaryScores);
    preliminaryScores = new Map(finalRun.rankings.map(row => [row.itemId, row.score]));
  }
  if (!finalRun) throw new Error('collaborative-inference did not run');
  return passToResult(cfg, finalRun, passes);
}

interface CollaborativeInferencePassResult {
  source: RespondentSourceStation;
  survey: SurveyEncoderStation;
  aggregator: EvidenceAggregatorStation;
  ranker: RankingInferenceStation;
  sink: InferenceResultSinkStation;
  rankings: CollaborativeItemScore[];
  coverage: CollaborativeInferenceCoverage;
  validation: ValidationCheck[];
  topology: StationGraphSummary;
}

function runCollaborativeInferencePass(
  cfg: NormalizedConfig,
  preliminaryScores?: ReadonlyMap<string, number>,
): CollaborativeInferencePassResult {
  const source = new RespondentSourceStation('respondent-source', cfg.responses, cfg.respondentsPerTick);
  const survey = new SurveyEncoderStation(
    'survey-encoder',
    new Set(cfg.items.map(i => i.id)),
    cfg.ratingMin,
    cfg.ratingMax,
    cfg.credibility,
    preliminaryScores,
  );
  const aggregator = new EvidenceAggregatorStation('evidence-aggregator', cfg.items.map(i => i.id), survey);
  const ranker = new RankingInferenceStation(
    'ranking-inference',
    cfg.items,
    cfg.ratingMin,
    cfg.ratingMax,
    cfg.ratingWeight,
    cfg.rankingWeight,
    cfg.shrinkage,
  );
  const sink = new InferenceResultSinkStation('inference-result-sink');

  source.pipe(survey, RespondentSourceStation.CH_RESPONDENT, SurveyEncoderStation.CH_RESPONDENT);
  survey.pipe(aggregator, SurveyEncoderStation.CH_RATING, EvidenceAggregatorStation.CH_RATING);
  survey.pipe(aggregator, SurveyEncoderStation.CH_PAIRWISE, EvidenceAggregatorStation.CH_PAIRWISE);
  aggregator.pipe(ranker, EvidenceAggregatorStation.CH_SNAPSHOT, RankingInferenceStation.CH_SNAPSHOT);
  ranker.pipe(sink, RankingInferenceStation.CH_RANKING, InferenceResultSinkStation.CH_RANKING);

  runIterativeDES([source, survey, aggregator, ranker, sink], {
    shuffle: false,
    maxTicks: Math.ceil(cfg.responses.length / cfg.respondentsPerTick) + 5,
    runValidators: false,
  });

  if (!sink.latest) throw new Error('collaborative-inference did not produce a ranking');
  const rankings = sink.latest.rankings;
  const coverage = coverageSummary(aggregator.stats);
  const validation = validationChecks(cfg, source, survey, aggregator, rankings, coverage);
  const topology = stationGraph(
    [source, survey, aggregator, ranker, sink],
    ['RespondentToken', 'RatingEvidenceToken', 'PairwisePreferenceToken', 'EvidenceSnapshotToken', 'RankingToken'],
    [
      channelEdge(source, RespondentSourceStation.CH_RESPONDENT, survey, SurveyEncoderStation.CH_RESPONDENT),
      channelEdge(survey, SurveyEncoderStation.CH_RATING, aggregator, EvidenceAggregatorStation.CH_RATING),
      channelEdge(survey, SurveyEncoderStation.CH_PAIRWISE, aggregator, EvidenceAggregatorStation.CH_PAIRWISE),
      channelEdge(aggregator, EvidenceAggregatorStation.CH_SNAPSHOT, ranker, RankingInferenceStation.CH_SNAPSHOT),
      channelEdge(ranker, RankingInferenceStation.CH_RANKING, sink, InferenceResultSinkStation.CH_RANKING),
    ],
  );
  return {source, survey, aggregator, ranker, sink, rankings, coverage, validation, topology};
}

function passToResult(
  cfg: NormalizedConfig,
  pass: CollaborativeInferencePassResult,
  passes: number,
): CollaborativeInferenceResult {
  return {
    scenario: cfg.scenario,
    scenarioLabel: cfg.scenarioLabel,
    synthetic: cfg.synthetic,
    respondentCount: cfg.responses.length,
    respondentsProcessed: pass.survey.respondentsProcessed,
    ratingEvidenceCount: pass.aggregator.ratingEvidenceCount,
    pairwiseEvidenceCount: pass.aggregator.pairwiseEvidenceCount,
    invalidEvidence: pass.survey.invalidEvidence.slice(0, 25),
    credibility: {
      enabled: cfg.credibility.enabled,
      passes,
      minCredibleAge: cfg.credibility.minCredibleAge,
      highRatedScoreThreshold: cfg.credibility.highRatedScoreThreshold,
      minHighRatedItems: cfg.credibility.minHighRatedItems,
      meanRespondentWeight: pass.survey.respondentsProcessed > 0 ? pass.survey.respondentWeightSum / pass.survey.respondentsProcessed : 0,
      maxRespondentWeight: pass.survey.maxRespondentWeight,
      cappedExperienceClaims: pass.survey.cappedExperienceClaims,
      highRatedBonusRespondents: pass.survey.highRatedBonusRespondents,
    },
    coverage: pass.coverage,
    rankings: pass.rankings,
    top: pass.rankings.slice(0, cfg.topK),
    validation: pass.validation,
    topology: pass.topology,
    stationRoles: {
      sources: ['respondent-source'],
      stations: ['survey-encoder', 'evidence-aggregator', 'ranking-inference'],
      sinks: ['inference-result-sink'],
      movables: ['RespondentToken', 'RatingEvidenceToken', 'PairwisePreferenceToken', 'EvidenceSnapshotToken', 'RankingToken'],
    },
  };
}

function normalizeCollaborativeInferenceParams(params: CollaborativeInferenceParams = {}): NormalizedConfig {
  const scenario = params.scenario ?? 'programming-languages';
  const preset = scenarioPreset(scenario);
  const items = normalizeItems(params.items && params.items.length > 0 ? params.items : preset.items);
  const itemById = new Map(items.map(item => [item.id, item]));
  if (itemById.size !== items.length) throw new Error('collaborative-inference: item ids must be unique');

  const respondentCount = params.respondentCount ?? params.respondents ?? preset.defaultRespondents;
  const minItems = params.minItemsPerRespondent ?? preset.minItemsPerRespondent;
  const maxItems = params.maxItemsPerRespondent ?? preset.maxItemsPerRespondent;
  const ratingMin = params.ratingMin ?? preset.ratingMin;
  const ratingMax = params.ratingMax ?? preset.ratingMax;
  const seed = params.seed ?? 1;
  const credibility = normalizeCredibilityConfig(params);
  if (ratingMax <= ratingMin) throw new Error('collaborative-inference: ratingMax must be greater than ratingMin');
  if (minItems < 1 || maxItems < minItems) throw new Error('collaborative-inference: require 1 <= minItemsPerRespondent <= maxItemsPerRespondent');

  const responses = params.responses && params.responses.length > 0
    ? params.responses.map((r, i) => normalizeResponse(r, i))
    : generateSyntheticResponses({
        items,
        respondentCount,
        minItems,
        maxItems,
        ratingMin,
        ratingMax,
        noiseStd: params.noiseStd ?? preset.noiseStd,
        seed,
        minCredibleAge: credibility.minCredibleAge,
      });

  return {
    scenario,
    scenarioLabel: preset.label,
    items,
    itemById,
    responses,
    respondentCount: responses.length,
    minItemsPerRespondent: minItems,
    maxItemsPerRespondent: maxItems,
    respondentsPerTick: Math.max(1, Math.floor(params.respondentsPerTick ?? 100)),
    ratingMin,
    ratingMax,
    noiseStd: params.noiseStd ?? preset.noiseStd,
    seed,
    ratingWeight: Math.max(0, params.ratingWeight ?? 0.55),
    rankingWeight: Math.max(0, params.rankingWeight ?? 0.45),
    shrinkage: Math.max(1e-9, params.shrinkage ?? 12),
    topK: Math.max(1, Math.floor(params.topK ?? 10)),
    synthetic: !(params.responses && params.responses.length > 0),
    credibility,
  };
}

function normalizeItems(items: readonly CollaborativeInferenceItem[]): CollaborativeInferenceItem[] {
  if (items.length === 0) throw new Error('collaborative-inference: items must be non-empty');
  return items.map((item, i) => {
    const id = item.id || `item-${i + 1}`;
    return {
      ...item,
      id,
      label: item.label ?? id,
      latentUtility: clamp01(item.latentUtility ?? 0.5),
      exposure: Math.max(0, item.exposure ?? 1),
      priorScore: item.priorScore === undefined ? undefined : clamp01(item.priorScore),
    };
  });
}

function normalizeResponse(response: CollaborativeInferenceResponse, index: number): CollaborativeInferenceResponse {
  const ratingIds = Object.keys(response.ratings ?? {});
  const itemIds = response.itemIds && response.itemIds.length > 0
    ? response.itemIds
    : response.ranking && response.ranking.length > 0
      ? response.ranking
      : ratingIds;
  return {
    ...response,
    id: response.id ?? `respondent-${index + 1}`,
    itemIds: unique(itemIds),
    ranking: response.ranking ? unique(response.ranking) : undefined,
    weight: finitePositive(response.weight, 1),
  };
}

function normalizeCredibilityConfig(params: CollaborativeInferenceParams): CredibilityWeightingConfig {
  return {
    enabled: params.credibilityWeighting ?? true,
    passes: Math.max(1, Math.floor(params.credibilityPasses ?? 2)),
    minCredibleAge: Math.max(0, params.minCredibleAge ?? 15),
    referenceAge: Math.max(1, params.referenceAge ?? 50),
    referenceExperienceYears: Math.max(1, params.referenceExperienceYears ?? 15),
    ageWeightStrength: Math.max(0, params.ageWeightStrength ?? 0.35),
    experienceWeightStrength: Math.max(0, params.experienceWeightStrength ?? 0.60),
    highRatedBreadthStrength: Math.max(0, params.highRatedBreadthStrength ?? 0.40),
    highRatedScoreThreshold: Math.max(0, Math.min(1, params.highRatedScoreThreshold ?? 0.72)),
    minHighRatedItems: Math.max(1, Math.floor(params.minHighRatedItems ?? 2)),
    maxMultiplier: Math.max(1, params.maxCredibilityMultiplier ?? 3),
  };
}

function respondentWeightProfile(
  response: CollaborativeInferenceResponse,
  seenItems: ReadonlySet<string>,
  cfg: CredibilityWeightingConfig,
  preliminaryScores?: ReadonlyMap<string, number>,
): RespondentWeightProfile {
  const explicitWeight = finitePositive(response.weight, 1);
  if (!cfg.enabled) {
    return {
      respondentWeight: explicitWeight,
      itemWeights: new Map([...seenItems].map(itemId => [itemId, explicitWeight])),
      highRatedItemCount: 0,
      breadthMultiplier: 1,
      cappedExperienceClaims: 0,
    };
  }

  const age = finiteNonNegative(response.age);
  const maxCredibleYears = age === undefined ? Infinity : Math.max(0, age - cfg.minCredibleAge);
  const ageMultiplier = age === undefined
    ? 1
    : 1 + cfg.ageWeightStrength * normalizedLog(Math.max(0, age - cfg.minCredibleAge), Math.max(1, cfg.referenceAge - cfg.minCredibleAge));

  const highRatedItemCount = preliminaryScores
    ? [...seenItems].filter(itemId => (preliminaryScores.get(itemId) ?? 0) >= cfg.highRatedScoreThreshold).length
    : 0;
  const breadthMultiplier = highRatedItemCount >= cfg.minHighRatedItems
    ? 1 + cfg.highRatedBreadthStrength * Math.min(2, highRatedItemCount / cfg.minHighRatedItems)
    : 1;

  const base = capMultiplier(explicitWeight * ageMultiplier * breadthMultiplier, explicitWeight, cfg.maxMultiplier);
  const itemWeights = new Map<string, number>();
  let cappedExperienceClaims = 0;
  for (const itemId of seenItems) {
    const rawYears = finiteNonNegative(response.experienceYears?.[itemId]) ?? 0;
    const cappedYears = Math.min(rawYears, maxCredibleYears);
    if (rawYears > cappedYears + 1e-9) cappedExperienceClaims += 1;
    const experienceMultiplier = 1 + cfg.experienceWeightStrength * normalizedLog(cappedYears, cfg.referenceExperienceYears);
    itemWeights.set(itemId, capMultiplier(base * experienceMultiplier, explicitWeight, cfg.maxMultiplier));
  }

  return {
    respondentWeight: base,
    itemWeights,
    highRatedItemCount,
    breadthMultiplier,
    cappedExperienceClaims,
  };
}

function generateSyntheticResponses(opts: {
  items: readonly CollaborativeInferenceItem[];
  respondentCount: number;
  minItems: number;
  maxItems: number;
  ratingMin: number;
  ratingMax: number;
  noiseStd: number;
  seed: number;
  minCredibleAge: number;
}): CollaborativeInferenceResponse[] {
  const rng = mulberry32(opts.seed);
  const out: CollaborativeInferenceResponse[] = [];
  for (let r = 0; r < opts.respondentCount; r++) {
    const k = randomInt(rng, opts.minItems, opts.maxItems);
    const selected = weightedSampleWithoutReplacement(opts.items, k, rng);
    const age = syntheticAge(rng);
    const maxCredibleYears = Math.max(0, age - opts.minCredibleAge);
    const personalOffset = normal01(rng) * 0.12;
    const ratings: Record<string, number> = {};
    const experienceYears: Record<string, number> = {};
    for (const item of selected) {
      const latent = item.latentUtility ?? 0.5;
      const noisyUtility = clamp01(latent + personalOffset + normal01(rng) * opts.noiseStd / 10);
      ratings[item.id] = roundTo(opts.ratingMin + noisyUtility * (opts.ratingMax - opts.ratingMin), 2);
      const seniority = 0.35 + 0.65 * rng();
      experienceYears[item.id] = roundTo(Math.min(maxCredibleYears, seniority * Math.min(18, maxCredibleYears)), 1);
    }
    const ranking = selected
      .map(item => item.id)
      .sort((a, b) => ratings[b] - ratings[a] || a.localeCompare(b));
    out.push({
      id: `respondent-${r + 1}`,
      age,
      itemIds: selected.map(item => item.id),
      experienceYears,
      ratings,
      ranking,
    });
  }
  return out;
}

function scenarioPreset(scenario: CollaborativeInferenceScenario): ScenarioPreset {
  switch (scenario) {
    case 'programming-languages':
      return programmingLanguagePreset();
    case 'model-validation':
      return modelValidationPreset();
    case 'learning-resources':
      return learningResourcesPreset();
    case 'custom':
      return {
        scenario,
        label: 'Custom collaborative inference scenario',
        defaultRespondents: 200,
        minItemsPerRespondent: 3,
        maxItemsPerRespondent: 5,
        ratingMin: 1,
        ratingMax: 10,
        noiseStd: 1,
        items: [
          {id: 'option-a', label: 'Option A', latentUtility: 0.58, exposure: 1},
          {id: 'option-b', label: 'Option B', latentUtility: 0.50, exposure: 1},
          {id: 'option-c', label: 'Option C', latentUtility: 0.42, exposure: 1},
        ],
      };
  }
}

function programmingLanguagePreset(): ScenarioPreset {
  const names = [
    ['python', 'Python', 0.88, 1.7],
    ['typescript', 'TypeScript', 0.84, 1.45],
    ['rust', 'Rust', 0.83, 0.75],
    ['go', 'Go', 0.80, 1.05],
    ['kotlin', 'Kotlin', 0.77, 0.70],
    ['swift', 'Swift', 0.75, 0.66],
    ['javascript', 'JavaScript', 0.74, 1.9],
    ['csharp', 'C#', 0.73, 1.15],
    ['java', 'Java', 0.70, 1.55],
    ['scala', 'Scala', 0.69, 0.42],
    ['elixir', 'Elixir', 0.68, 0.32],
    ['clojure', 'Clojure', 0.67, 0.25],
    ['julia', 'Julia', 0.66, 0.30],
    ['ruby', 'Ruby', 0.65, 0.75],
    ['fsharp', 'F#', 0.64, 0.20],
    ['haskell', 'Haskell', 0.63, 0.22],
    ['php', 'PHP', 0.61, 1.1],
    ['c', 'C', 0.60, 1.05],
    ['cpp', 'C++', 0.59, 1.15],
    ['r', 'R', 0.58, 0.70],
    ['dart', 'Dart', 0.57, 0.42],
    ['lua', 'Lua', 0.56, 0.28],
    ['erlang', 'Erlang', 0.55, 0.18],
    ['ocaml', 'OCaml', 0.54, 0.16],
    ['zig', 'Zig', 0.53, 0.24],
    ['nim', 'Nim', 0.52, 0.12],
    ['perl', 'Perl', 0.51, 0.25],
    ['shell', 'Shell', 0.50, 1.25],
    ['sql', 'SQL', 0.49, 1.35],
    ['matlab', 'MATLAB', 0.48, 0.35],
    ['groovy', 'Groovy', 0.47, 0.22],
    ['powershell', 'PowerShell', 0.46, 0.65],
    ['objective-c', 'Objective-C', 0.45, 0.20],
    ['visual-basic', 'Visual Basic', 0.44, 0.25],
    ['fortran', 'Fortran', 0.43, 0.12],
    ['cobol', 'COBOL', 0.42, 0.10],
    ['delphi', 'Delphi', 0.41, 0.12],
    ['smalltalk', 'Smalltalk', 0.40, 0.08],
    ['elm', 'Elm', 0.39, 0.12],
    ['reason', 'ReasonML', 0.38, 0.08],
    ['solidity', 'Solidity', 0.37, 0.24],
    ['apex', 'Apex', 0.36, 0.15],
    ['abap', 'ABAP', 0.35, 0.14],
    ['assembly', 'Assembly', 0.34, 0.42],
    ['racket', 'Racket', 0.33, 0.10],
    ['prolog', 'Prolog', 0.32, 0.08],
    ['ada', 'Ada', 0.31, 0.08],
    ['vba', 'VBA', 0.30, 0.28],
    ['scratch', 'Scratch', 0.29, 0.18],
    ['coffeescript', 'CoffeeScript', 0.28, 0.10],
  ] as const;
  return {
    scenario: 'programming-languages',
    label: 'Programming languages ranked from sparse developer experience',
    defaultRespondents: 10000,
    minItemsPerRespondent: 4,
    maxItemsPerRespondent: 5,
    ratingMin: 1,
    ratingMax: 10,
    noiseStd: 1.1,
    items: names.map(([id, label, latentUtility, exposure]) => ({id, label, latentUtility, exposure, group: 'language'})),
  };
}

function modelValidationPreset(): ScenarioPreset {
  const rows = [
    ['des-station-graph', 'DES station graph', 0.86, 1.2, 'execution'],
    ['fel-reference', 'Future-event-list reference', 0.78, 0.9, 'validation'],
    ['monte-carlo', 'Monte Carlo replication', 0.76, 1.1, 'validation'],
    ['analytical-baseline', 'Analytical baseline', 0.74, 0.8, 'validation'],
    ['simpy-reference', 'SimPy reference model', 0.71, 0.7, 'external'],
    ['ciw-reference', 'Ciw queueing reference', 0.68, 0.5, 'external'],
    ['agent-based', 'Agent-based model', 0.66, 0.8, 'execution'],
    ['digital-twin', 'Hybrid digital twin', 0.64, 0.45, 'execution'],
    ['neural-surrogate', 'Neural surrogate', 0.60, 0.55, 'approximation'],
    ['spreadsheet', 'Spreadsheet prototype', 0.48, 0.9, 'baseline'],
    ['ad-hoc-script', 'Ad-hoc script', 0.38, 1.0, 'baseline'],
    ['manual-review', 'Manual review only', 0.30, 0.6, 'baseline'],
  ] as const;
  return {
    scenario: 'model-validation',
    label: 'Model validation workflows ranked by external reviewers',
    defaultRespondents: 800,
    minItemsPerRespondent: 3,
    maxItemsPerRespondent: 5,
    ratingMin: 1,
    ratingMax: 7,
    noiseStd: 0.9,
    items: rows.map(([id, label, latentUtility, exposure, group]) => ({id, label, latentUtility, exposure, group})),
  };
}

function learningResourcesPreset(): ScenarioPreset {
  const rows = [
    ['worked-examples', 'Worked examples', 0.84, 1.3, 'practice'],
    ['interactive-notebooks', 'Interactive notebooks', 0.81, 1.1, 'practice'],
    ['project-builds', 'Small project builds', 0.79, 1.0, 'practice'],
    ['office-hours', 'Office hours', 0.74, 0.8, 'support'],
    ['visual-simulations', 'Visual simulations', 0.72, 0.7, 'exploration'],
    ['short-videos', 'Short videos', 0.68, 1.4, 'content'],
    ['textbook-chapters', 'Textbook chapters', 0.62, 1.1, 'content'],
    ['flashcards', 'Flashcards', 0.55, 0.8, 'review'],
    ['long-lectures', 'Long lectures', 0.50, 1.0, 'content'],
    ['discussion-board', 'Discussion board', 0.46, 0.7, 'support'],
  ] as const;
  return {
    scenario: 'learning-resources',
    label: 'Learning resources ranked from sparse student feedback',
    defaultRespondents: 1200,
    minItemsPerRespondent: 3,
    maxItemsPerRespondent: 4,
    ratingMin: 1,
    ratingMax: 5,
    noiseStd: 0.8,
    items: rows.map(([id, label, latentUtility, exposure, group]) => ({id, label, latentUtility, exposure, group})),
  };
}

function validationChecks(
  cfg: NormalizedConfig,
  source: RespondentSourceStation,
  survey: SurveyEncoderStation,
  aggregator: EvidenceAggregatorStation,
  rankings: readonly CollaborativeItemScore[],
  coverage: CollaborativeInferenceCoverage,
): ValidationCheck[] {
  return [
    {
      name: 'respondent conservation',
      group: 'collaborative inference',
      passed: source.emittedCount === survey.respondentsProcessed && survey.respondentsProcessed === cfg.responses.length,
      observed: String(survey.respondentsProcessed),
      expected: String(cfg.responses.length),
    },
    {
      name: 'rating evidence conservation',
      group: 'collaborative inference',
      passed: survey.ratingEvidenceCount === aggregator.ratingEvidenceCount,
      observed: String(aggregator.ratingEvidenceCount),
      expected: String(survey.ratingEvidenceCount),
    },
    {
      name: 'pairwise evidence conservation',
      group: 'collaborative inference',
      passed: survey.pairwiseEvidenceCount === aggregator.pairwiseEvidenceCount,
      observed: String(aggregator.pairwiseEvidenceCount),
      expected: String(survey.pairwiseEvidenceCount),
    },
    {
      name: 'all items ranked',
      group: 'collaborative inference',
      passed: rankings.length === cfg.items.length,
      observed: String(rankings.length),
      expected: String(cfg.items.length),
    },
    {
      name: 'scores are finite probabilities',
      group: 'collaborative inference',
      passed: rankings.every(r => Number.isFinite(r.score) && r.score >= 0 && r.score <= 1),
      expected: 'score in [0, 1] for every item',
    },
    {
      name: 'coverage reaches every item',
      group: 'collaborative inference',
      passed: coverage.itemsWithRatings === cfg.items.length && coverage.itemsWithComparisons === cfg.items.length,
      observed: `${coverage.itemsWithRatings}/${cfg.items.length} rated, ${coverage.itemsWithComparisons}/${cfg.items.length} compared`,
      expected: 'each item has rating and comparison evidence',
    },
    {
      name: 'no invalid evidence',
      group: 'collaborative inference',
      passed: survey.invalidEvidence.length === 0,
      observed: String(survey.invalidEvidence.length),
      expected: '0',
      details: survey.invalidEvidence.slice(0, 3).join('; '),
    },
    {
      name: 'credibility weights are finite',
      group: 'collaborative inference',
      passed: Number.isFinite(survey.respondentWeightSum) && Number.isFinite(survey.maxRespondentWeight),
      observed: `mean=${survey.respondentsProcessed > 0 ? survey.respondentWeightSum / survey.respondentsProcessed : 0}, max=${survey.maxRespondentWeight}`,
      expected: 'finite respondent weights',
    },
  ];
}

function coverageSummary(stats: ReadonlyMap<string, ItemEvidenceStats>): CollaborativeInferenceCoverage {
  const rows = [...stats.values()];
  const ratingCounts = rows.map(r => r.ratingCount);
  const comparisonCounts = rows.map(r => r.pairwiseWins + r.pairwiseLosses);
  return {
    items: rows.length,
    itemsWithRatings: ratingCounts.filter(v => v > 0).length,
    itemsWithComparisons: comparisonCounts.filter(v => v > 0).length,
    minRatingsPerItem: Math.min(...ratingCounts),
    meanRatingsPerItem: mean(ratingCounts),
    maxRatingsPerItem: Math.max(...ratingCounts),
    minComparisonsPerItem: Math.min(...comparisonCounts),
    meanComparisonsPerItem: mean(comparisonCounts),
    maxComparisonsPerItem: Math.max(...comparisonCounts),
  };
}

function cloneStats(stats: ReadonlyMap<string, ItemEvidenceStats>): Map<string, ItemEvidenceStats> {
  return new Map([...stats.entries()].map(([k, v]) => [k, {...v}]));
}

function emptyStats(itemId: string): ItemEvidenceStats {
  return {itemId, ratingSum: 0, ratingWeight: 0, ratingCount: 0, pairwiseWins: 0, pairwiseLosses: 0};
}

function weightedSampleWithoutReplacement(
  items: readonly CollaborativeInferenceItem[],
  count: number,
  rng: () => number,
): CollaborativeInferenceItem[] {
  const pool = items.slice();
  const out: CollaborativeInferenceItem[] = [];
  const n = Math.min(count, pool.length);
  for (let k = 0; k < n; k++) {
    const total = pool.reduce((s, item) => s + Math.max(0, item.exposure ?? 1), 0);
    let draw = rng() * (total > 0 ? total : pool.length);
    let idx = 0;
    for (; idx < pool.length; idx++) {
      draw -= total > 0 ? Math.max(0, pool[idx].exposure ?? 1) : 1;
      if (draw <= 0) break;
    }
    const picked = pool.splice(Math.min(idx, pool.length - 1), 1)[0];
    out.push(picked);
  }
  return out;
}

function normal01(rng: () => number): number {
  const u1 = Math.max(1e-12, rng());
  const u2 = Math.max(1e-12, rng());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randomInt(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

function finitePositive(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) && (value as number) > 0 ? value as number : fallback;
}

function finiteNonNegative(value: number | undefined): number | undefined {
  return Number.isFinite(value) && (value as number) >= 0 ? value as number : undefined;
}

function normalizedLog(value: number, reference: number): number {
  return Math.min(1, Math.log1p(Math.max(0, value)) / Math.log1p(Math.max(1, reference)));
}

function capMultiplier(value: number, base: number, maxMultiplier: number): number {
  const cap = Math.max(base, base * maxMultiplier);
  return Math.max(0, Math.min(value, cap));
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0.5;
  return Math.max(0, Math.min(1, x));
}

function roundTo(x: number, digits: number): number {
  const m = 10 ** digits;
  return Math.round(x * m) / m;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function syntheticAge(rng: () => number): number {
  return Math.floor(18 + Math.pow(rng(), 1.45) * 47);
}
