// RUST MIGRATION: Target module `src/des/general/adapters/collaborative_inference_adapter.rs`.
// RUST MIGRATION: Convert collaborative inference adapter registration plus scene rendering helpers into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Encode inference params, responses, and animation frame payloads as `serde` config/result structs; output paths should be `PathBuf`.
// RUST MIGRATION: Return `Result<_, ValidationError>` for invalid item/response data and rendering setup failures.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/collaborative-inference-adapter.rs
//   (module des::general::adapters::collaborative_inference_adapter)
// 1:1 file move. JSON adapter for sparse collaborative preference inference, with
// an animated station-graph scene builder.
//
// Declarations → Rust:
//   const itemSchema/responseSchema/schema: ParamSchema -> serde + validator metadata
//   registerModel<P,R>({ id, schema, run, summarize, writeCsv, animate }) -> struct +
//             impl ModelAdapter trait
//   fn collaborativeInferenceScene / stationCount / credibilityCaption / text /
//      rankColor / truncate / lerp / easeOutCubic / mean -> plain `fn` helpers
//
// Conversion notes (file-specific):
//   - GotChA: `top: CollaborativeInferenceResult['top']` is an indexed-access type
//     (the element type of the `top` field) — name that element struct explicitly
//     in Rust; indexed-access types have no analogue.
//   - response.ratings / experienceYears are open `{kind:'object', fields:{}}` maps
//     (string→number) -> HashMap<String, f64>; `scenario` literal union -> enum.
//   - `.toLocaleString()` thousands formatting -> a num-format helper (not built in).
//   - Shapes pushed into `Shape[]` (animation/types) -> Vec<Shape>; Shape -> enum;
//     the animation is purely derived from the result (eased interpolation), no RNG.
//   - `Number.isInteger(value)` formatting branch -> f64::fract()==0 check.
// =============================================================================

// =============================================================================
// JSON adapter for collaborative sparse preference inference.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {
  CollaborativeInferenceParams,
  CollaborativeInferenceResult,
  runCollaborativeInference,
} from '../collaborative-inference';
import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {csvRow, framesPath, validationLine, writeCsvLines} from './adapter-utils';

const itemSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    label: {kind: 'string'},
    group: {kind: 'string'},
    latentUtility: {kind: 'number', min: 0, max: 1, default: 0.5},
    exposure: {kind: 'number', min: 0, default: 1},
    priorScore: {kind: 'number', min: 0, max: 1},
  },
  required: ['id'],
};

const responseSchema: ParamSchema = {
  kind: 'object',
  fields: {
    id: {kind: 'string'},
    itemIds: {kind: 'array', items: {kind: 'string'}, minLength: 0},
    ratings: {kind: 'object', fields: {}, required: []},
    ranking: {kind: 'array', items: {kind: 'string'}, minLength: 0},
    age: {kind: 'number', min: 0},
    experienceYears: {kind: 'object', fields: {}, required: []},
    weight: {kind: 'number', min: 0, default: 1},
    segment: {kind: 'string'},
  },
  required: [],
};

const schema: ParamSchema = {
  kind: 'object',
  fields: {
    scenario: {kind: 'string', enum: ['programming-languages', 'model-validation', 'learning-resources', 'custom'], default: 'programming-languages'},
    items: {kind: 'array', items: itemSchema, minLength: 0},
    responses: {kind: 'array', items: responseSchema, minLength: 0},
    respondentCount: {kind: 'number', integer: true, min: 1},
    respondents: {kind: 'number', integer: true, min: 1},
    minItemsPerRespondent: {kind: 'number', integer: true, min: 1},
    maxItemsPerRespondent: {kind: 'number', integer: true, min: 1},
    respondentsPerTick: {kind: 'number', integer: true, min: 1, default: 100},
    ratingMin: {kind: 'number'},
    ratingMax: {kind: 'number'},
    noiseStd: {kind: 'number', min: 0},
    seed: {kind: 'number', integer: true, default: 1},
    ratingWeight: {kind: 'number', min: 0, default: 0.55},
    rankingWeight: {kind: 'number', min: 0, default: 0.45},
    shrinkage: {kind: 'number', min: 0, default: 12},
    topK: {kind: 'number', integer: true, min: 1, default: 10},
    credibilityWeighting: {kind: 'boolean', default: true},
    credibilityPasses: {kind: 'number', integer: true, min: 1, default: 2},
    minCredibleAge: {kind: 'number', min: 0, default: 15},
    referenceAge: {kind: 'number', min: 1, default: 50},
    referenceExperienceYears: {kind: 'number', min: 1, default: 15},
    ageWeightStrength: {kind: 'number', min: 0, default: 0.35},
    experienceWeightStrength: {kind: 'number', min: 0, default: 0.6},
    highRatedBreadthStrength: {kind: 'number', min: 0, default: 0.4},
    highRatedScoreThreshold: {kind: 'number', min: 0, max: 1, default: 0.72},
    minHighRatedItems: {kind: 'number', integer: true, min: 1, default: 2},
    maxCredibilityMultiplier: {kind: 'number', min: 1, default: 3},
  },
  required: [],
};

registerModel<CollaborativeInferenceParams, CollaborativeInferenceResult>({
  id: 'collaborative-inference',
  description: 'Sparse subjective ratings/rankings fused into a global item ranking with station-graph evidence aggregation.',
  schema,
  run(params) { return runCollaborativeInference(params); },
  summarize(result) {
    const lines = [
      'COLLABORATIVE INFERENCE (sparse preference learning DES)',
      '-------------------------------------------------------',
      `  Scenario:       ${result.scenarioLabel}`,
      `  Respondents:    ${result.respondentsProcessed.toLocaleString()}`,
      `  Ratings:        ${result.ratingEvidenceCount.toLocaleString()}`,
      `  Comparisons:    ${result.pairwiseEvidenceCount.toLocaleString()}`,
      `  Coverage:       ratings ${result.coverage.itemsWithRatings}/${result.coverage.items}, comparisons ${result.coverage.itemsWithComparisons}/${result.coverage.items}`,
      `  Credibility:    ${result.credibility.enabled ? `${result.credibility.passes} pass(es), mean weight=${result.credibility.meanRespondentWeight.toFixed(3)}, max=${result.credibility.maxRespondentWeight.toFixed(3)}, capped claims=${result.credibility.cappedExperienceClaims}, breadth bonuses=${result.credibility.highRatedBonusRespondents}` : 'disabled'}`,
      `  Validation:     ${validationLine(result.validation)}`,
      '',
      '  Top inferred items:',
      ...result.top.map(row => `    ${row.rank}. ${row.label.padEnd(24)} score=${row.score.toFixed(3)} confidence=${row.confidence.toFixed(3)} ratings=${row.ratingCount}`),
      '',
      `  Sources:        ${result.stationRoles.sources.join(', ')}`,
      `  Stations:       ${result.stationRoles.stations.join(' -> ')}`,
      `  Sinks:          ${result.stationRoles.sinks.join(', ')}`,
      `  Movables:       ${result.stationRoles.movables.join(', ')}`,
    ];
    return lines.join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow([
      'rank',
      'item_id',
      'label',
      'group',
      'score',
      'confidence',
      'uncertainty',
      'rating_mean',
      'rating_count',
      'comparison_count',
      'pairwise_win_rate',
    ])];
    for (const row of result.rankings) {
      lines.push(csvRow([
        row.rank,
        row.itemId,
        row.label,
        row.group ?? '',
        row.score,
        row.confidence,
        row.uncertainty,
        row.ratingMean,
        row.ratingCount,
        row.comparisonCount,
        row.pairwiseWinRate,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(result, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'collaborative-inference');
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 980,
      height: 620,
      fps: 12,
      title: 'Collaborative inference',
      subtitle: `${result.scenarioLabel}: sparse respondent opinions become global rankings`,
      background: '#fbfcfd',
    });

    const frameCount = 90;
    const top = result.top.slice(0, Math.min(8, result.top.length));
    const respondentsT: number[] = [];
    const respondentsY: number[] = [];
    const ratingsT: number[] = [];
    const ratingsY: number[] = [];
    const comparisonsT: number[] = [];
    const comparisonsY: number[] = [];
    const confidenceT: number[] = [];
    const confidenceY: number[] = [];

    for (let tick = 0; tick < frameCount; tick++) {
      const p = tick / (frameCount - 1);
      const eased = easeOutCubic(p);
      respondentsT.push(tick);
      respondentsY.push(result.respondentsProcessed * eased);
      ratingsT.push(tick);
      ratingsY.push(result.ratingEvidenceCount * eased);
      comparisonsT.push(tick);
      comparisonsY.push(result.pairwiseEvidenceCount * eased);
      confidenceT.push(tick);
      confidenceY.push(mean(top.map(row => row.confidence)) * eased);

      rec.frame(tick, tick, () => ({
        shapes: collaborativeInferenceScene(result, top, tick, p, eased),
        caption: `respondents=${Math.round(result.respondentsProcessed * eased).toLocaleString()}  ratings=${Math.round(result.ratingEvidenceCount * eased).toLocaleString()}  comparisons=${Math.round(result.pairwiseEvidenceCount * eased).toLocaleString()}`,
      }));
    }

    rec.setCharts([
      {
        x: 48, y: 474, w: 410, h: 96, title: 'Evidence accumulation', yLabel: 'count',
        series: [
          {label: 'respondents', color: '#2f6f73', t: respondentsT, y: respondentsY},
          {label: 'ratings', color: '#3867d6', t: ratingsT, y: ratingsY},
          {label: 'comparisons', color: '#b65c2f', t: comparisonsT, y: comparisonsY},
        ],
      },
      {
        x: 520, y: 474, w: 360, h: 96, title: 'Top-rank confidence', yMin: 0, yMax: 1, yLabel: 'mean',
        series: [{label: 'confidence', color: '#4b8f46', t: confidenceT, y: confidenceY}],
      },
    ]);
    await rec.finish();
  },
  examples: [
    {
      name: 'programming-languages',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'collaborative-inference',
        description: 'Rank 50 programming languages from 10,000 sparse developer ratings/rankings.',
        parameters: {
          scenario: 'programming-languages',
          respondentCount: 10000,
          minItemsPerRespondent: 4,
          maxItemsPerRespondent: 5,
          seed: 7,
          topK: 12,
        },
      },
    },
    {
      name: 'model-validation',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'collaborative-inference',
        description: 'Rank model execution and validation workflows from sparse external reviewer feedback.',
        parameters: {
          scenario: 'model-validation',
          respondentCount: 800,
          minItemsPerRespondent: 3,
          maxItemsPerRespondent: 5,
          seed: 11,
          topK: 8,
        },
      },
    },
    {
      name: 'learning-resources',
      spec: {
        $schema: 'des/model-spec/v1',
        model: 'collaborative-inference',
        description: 'Rank learning resources from sparse student ratings/rankings.',
        parameters: {
          scenario: 'learning-resources',
          respondentCount: 1200,
          minItemsPerRespondent: 3,
          maxItemsPerRespondent: 4,
          seed: 5,
          topK: 8,
        },
      },
    },
  ],
});

function collaborativeInferenceScene(
  result: CollaborativeInferenceResult,
  top: CollaborativeInferenceResult['top'],
  tick: number,
  p: number,
  eased: number,
): Shape[] {
  const shapes: Shape[] = [];
  const stations = [
    {id: 'Source', x: 92, y: 124, color: '#cfe7e4'},
    {id: 'Survey', x: 290, y: 124, color: '#d7e3ff'},
    {id: 'Evidence', x: 500, y: 124, color: '#f7ddc8'},
    {id: 'Ranker', x: 710, y: 124, color: '#dff0d8'},
    {id: 'Sink', x: 884, y: 124, color: '#eadcf8'},
  ];

  shapes.push(
    text(48, 44, result.scenarioLabel, 18, '#17202a', 'bold'),
    text(48, 70, `${result.respondentsProcessed.toLocaleString()} respondents, ${result.coverage.items} items, ${validationLine(result.validation)}`, 13, '#566573'),
    text(48, 92, credibilityCaption(result, eased), 12, '#566573'),
  );

  for (let i = 0; i < stations.length - 1; i++) {
    shapes.push({
      kind: 'line',
      x1: stations[i].x + 68,
      y1: stations[i].y + 34,
      x2: stations[i + 1].x - 68,
      y2: stations[i + 1].y + 34,
      stroke: '#85929e',
      strokeWidth: 3,
      opacity: 0.8,
    });
  }

  for (const s of stations) {
    shapes.push({
      kind: 'rect',
      x: s.x - 68,
      y: s.y,
      w: 136,
      h: 68,
      rx: 6,
      fill: s.color,
      stroke: '#5d6d7e',
      strokeWidth: 1.2,
    });
    shapes.push(text(s.x, s.y + 28, s.id, 14, '#1f2d3d', 'bold', 'middle'));
    const count = stationCount(s.id, result, eased);
    shapes.push(text(s.x, s.y + 49, count, 11, '#34495e', 'normal', 'middle'));
  }

  const tokenColors = ['#2f6f73', '#3867d6', '#b65c2f', '#4b8f46', '#7d5ab6'];
  for (let i = 0; i < 12; i++) {
    const local = (p * 4 + i / 12 + tick * 0.005) % 1;
    const x = lerp(160, 816, local);
    const y = 158 + Math.sin((local * Math.PI * 2) + i) * 18;
    shapes.push({kind: 'circle', x, y, r: 5, fill: tokenColors[i % tokenColors.length], opacity: 0.28 + 0.55 * eased});
  }

  shapes.push(text(48, 236, 'Top inferred ranking', 16, '#17202a', 'bold'));
  shapes.push(text(640, 236, 'Coverage', 16, '#17202a', 'bold'));

  const maxScore = Math.max(0.01, ...top.map(row => row.score));
  top.forEach((row, i) => {
    const y = 264 + i * 25;
    const barW = 430 * (row.score / maxScore) * eased;
    const label = `${row.rank}. ${truncate(row.label, 24)}`;
    shapes.push(text(48, y + 14, label, 12, '#273746'));
    shapes.push({kind: 'rect', x: 218, y, w: 438, h: 16, rx: 4, fill: '#edf1f5', stroke: '#d5dce3'});
    shapes.push({kind: 'rect', x: 218, y, w: barW, h: 16, rx: 4, fill: rankColor(i), opacity: 0.9});
    shapes.push(text(670, y + 13, `${(row.score * eased).toFixed(3)}  conf ${(row.confidence * eased).toFixed(2)}`, 11, '#566573'));
  });

  const coverageRows = [
    ['Items rated', result.coverage.itemsWithRatings, result.coverage.items],
    ['Items compared', result.coverage.itemsWithComparisons, result.coverage.items],
    ['Mean ratings/item', result.coverage.meanRatingsPerItem, result.coverage.maxRatingsPerItem],
    ['Mean comps/item', result.coverage.meanComparisonsPerItem, result.coverage.maxComparisonsPerItem],
  ] as const;
  coverageRows.forEach((row, i) => {
    const y = 264 + i * 44;
    const value = typeof row[1] === 'number' ? row[1] : 0;
    const max = Math.max(1, typeof row[2] === 'number' ? row[2] : 1);
    shapes.push(text(640, y + 12, row[0], 12, '#273746'));
    shapes.push({kind: 'rect', x: 640, y: y + 18, w: 250, h: 12, rx: 4, fill: '#edf1f5', stroke: '#d5dce3'});
    shapes.push({kind: 'rect', x: 640, y: y + 18, w: 250 * Math.min(1, value / max) * eased, h: 12, rx: 4, fill: '#2f6f73', opacity: 0.8});
    shapes.push(text(900, y + 29, Number.isInteger(value) ? `${Math.round(value)}` : value.toFixed(1), 11, '#566573', 'normal', 'end'));
  });

  return shapes;
}

function stationCount(id: string, result: CollaborativeInferenceResult, p: number): string {
  if (id === 'Source') return `${Math.round(result.respondentsProcessed * p).toLocaleString()} emitted`;
  if (id === 'Survey') return `${Math.round(result.ratingEvidenceCount * p).toLocaleString()} ratings`;
  if (id === 'Evidence') return `${Math.round(result.pairwiseEvidenceCount * p).toLocaleString()} pairs`;
  if (id === 'Ranker') return `${Math.round(result.coverage.items * p)}/${result.coverage.items} ranked`;
  return `${result.top.length} top items`;
}

function credibilityCaption(result: CollaborativeInferenceResult, p: number): string {
  if (!result.credibility.enabled) return 'credibility weighting disabled';
  const capped = Math.round(result.credibility.cappedExperienceClaims * p);
  const bonuses = Math.round(result.credibility.highRatedBonusRespondents * p);
  return `credibility: ${result.credibility.passes} passes, age-capped experience, mean weight ${(result.credibility.meanRespondentWeight * p).toFixed(2)}, capped claims ${capped}, breadth bonuses ${bonuses}`;
}

function text(
  x: number,
  y: number,
  value: string,
  fontSize = 12,
  fill = '#111',
  fontWeight: 'normal' | 'bold' = 'normal',
  anchor: 'start' | 'middle' | 'end' = 'start',
): Shape {
  return {kind: 'text', x, y, text: value, fontSize, fill, fontWeight, anchor};
}

function rankColor(i: number): string {
  const colors = ['#3867d6', '#4b8f46', '#b65c2f', '#7d5ab6', '#2f6f73', '#d4a017', '#566573', '#a44952'];
  return colors[i % colors.length];
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : `${s.slice(0, Math.max(0, max - 3))}...`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}
