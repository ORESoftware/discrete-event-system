'use strict';

// =============================================================================
// JSON adapters for statistical + stochastic optimisation models.
// =============================================================================

import {registerModel} from '../des-registry';
import {DESModelRegistration, ParamSchema} from '../des-spec';
import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {csvRow, framesPath, validationLine, withLogger, writeCsvLines} from './adapter-utils';
import {
  AdaptiveSimOptParams, AdaptiveSimOptResult, DistributionFitParams,
  DistributionFitResult, RiskCapacityParams, RiskCapacityResult,
  runAdaptiveSimOpt, runCapacityExpansionSDDP, runDistributionFit,
  runRiskCapacity, SDDPParams, SDDPResult,
} from '../statistical-optimization';
import {
  buildProductionScenarios, buildProductionSLP, solveProductionClosedForm,
  solveSLPBenders, solveSLPMonolithic, SLPSolveResult,
} from '../stochastic-lp';

const rangeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    low: {kind: 'number', min: 0},
    high: {kind: 'number', min: 0},
  },
  required: ['low', 'high'],
};

const fittedDistributionSchema: ParamSchema = {
  kind: 'object',
  fields: {
    family: {kind: 'string', enum: ['normal', 'lognormal', 'exponential', 'gamma', 'poisson', 'empirical']},
    method: {kind: 'string', enum: ['mle', 'moments'], default: 'mle'},
    params: {kind: 'object', fields: {}, required: []},
    logLikelihood: {kind: 'number', default: 0},
    aic: {kind: 'number', default: 0},
    mean: {kind: 'number', default: 0},
    variance: {kind: 'number', default: 0},
    support: {kind: 'string', enum: ['real', 'positive', 'nonnegative-integer', 'empirical'], default: 'positive'},
  },
  required: ['family', 'params'],
};

const empiricalPointSchema: ParamSchema = {
  kind: 'object',
  fields: {
    value: {kind: 'number'},
    prob: {kind: 'number', min: 0, max: 1},
  },
  required: ['value', 'prob'],
};

const demandSchema: ParamSchema = {
  kind: 'oneOf',
  variants: [
    {tag: 'uniform', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['uniform'], default: 'uniform'},
      ranges: {kind: 'array', items: rangeSchema, minLength: 1},
    }, required: ['kind', 'ranges']}},
    {tag: 'fitted', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['fitted']},
      fitted: {kind: 'array', items: fittedDistributionSchema, minLength: 1},
    }, required: ['kind', 'fitted']}},
    {tag: 'empirical', schema: {kind: 'object', fields: {
      kind: {kind: 'string', enum: ['empirical']},
      empirical: {kind: 'array', items: {kind: 'array', items: empiricalPointSchema, minLength: 1}, minLength: 1},
    }, required: ['kind', 'empirical']}},
  ],
};

const fitParamsSchema: ParamSchema = {
  kind: 'object',
  fields: {
    samples: {kind: 'array', items: {kind: 'number'}, minLength: 2},
    families: {kind: 'array', items: {kind: 'string', enum: ['normal', 'lognormal', 'exponential', 'gamma', 'poisson', 'empirical']}},
    methods: {kind: 'array', items: {kind: 'string', enum: ['mle', 'moments']}},
  },
  required: ['samples'],
};

const riskParamsSchema: ParamSchema = {
  kind: 'object',
  fields: {
    cost: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    price: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    demand: demandSchema,
    numScenarios: {kind: 'number', integer: true, min: 1, default: 200},
    seed: {kind: 'number', integer: true, default: 42},
    xMax: {kind: 'number', min: 0},
    step: {kind: 'number', min: 0},
    risk: {
      kind: 'object',
      fields: {
        kind: {kind: 'string', enum: ['expectation', 'cvar', 'chance', 'dro'], default: 'expectation'},
        alpha: {kind: 'number', min: 0.5, max: 0.999, default: 0.9},
        lambda: {kind: 'number', min: 0, default: 1},
        minServiceLevel: {kind: 'number', min: 0, max: 1, default: 0.9},
        shortfallLimit: {kind: 'number', min: 0, default: 0},
        radius: {kind: 'number', min: 0, default: 1},
      },
      required: ['kind'],
    },
  },
  required: ['cost', 'price', 'demand', 'xMax', 'step', 'risk'],
};

const sddpParamsSchema: ParamSchema = {
  kind: 'object',
  fields: {
    horizon: {kind: 'number', integer: true, min: 1},
    demand: {kind: 'array', items: rangeSchema, minLength: 1},
    price: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    expansionCost: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    initialCapacity: {kind: 'number', min: 0, default: 0},
    xMax: {kind: 'number', min: 0},
    step: {kind: 'number', min: 0},
    samplesPerStage: {kind: 'number', integer: true, min: 1, default: 40},
    seed: {kind: 'number', integer: true, default: 7},
    maxIter: {kind: 'number', integer: true, min: 1, default: 40},
    tol: {kind: 'number', min: 0, default: 1e-3},
  },
  required: ['horizon', 'demand', 'price', 'expansionCost', 'xMax', 'step'],
};

const altSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    x: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
  },
  required: ['name', 'x'],
};

const adaptiveParamsSchema: ParamSchema = {
  kind: 'object',
  fields: {
    cost: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    price: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    demand: demandSchema,
    alternatives: {kind: 'array', items: altSchema, minLength: 2},
    seed: {kind: 'number', integer: true, default: 11},
    initialSamples: {kind: 'number', integer: true, min: 1, default: 5},
    budget: {kind: 'number', integer: true, min: 1, default: 120},
    batchSize: {kind: 'number', integer: true, min: 1, default: 5},
    exploration: {kind: 'number', min: 0, default: 1.5},
  },
  required: ['cost', 'price', 'demand', 'alternatives'],
};

interface StochasticLPParams {
  cost?: number[];
  price?: number[];
  c?: number[];
  p?: number[];
  ranges: Array<[number, number]>;
  numScenarios?: number;
  N?: number;
  seed?: number;
  budget?: number;
  maxIter?: number;
  tol?: number;
  oosN?: number;
}

interface NormalizedStochasticLPParams {
  cost: number[];
  price: number[];
  ranges: Array<[number, number]>;
  numScenarios: number;
  seed: number;
  budget?: number;
  maxIter?: number;
  tol?: number;
  oosN?: number;
}

interface StochasticLPAdapterResult {
  closedForm?: SLPSolveResult;
  monolithic: SLPSolveResult;
  benders: SLPSolveResult;
  outOfSample?: {
    N: number;
    monolithic: number;
    benders: number;
    closedForm?: number;
  };
}

const stochasticLPSchema: ParamSchema = {
  kind: 'object',
  fields: {
    cost: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    price: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    c: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    p: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 1},
    ranges: {kind: 'array', items: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 2, maxLength: 2}, minLength: 1},
    numScenarios: {kind: 'number', integer: true, min: 1, default: 200},
    N: {kind: 'number', integer: true, min: 1, default: 200},
    seed: {kind: 'number', integer: true, default: 42},
    budget: {kind: 'number', min: 0},
    maxIter: {kind: 'number', integer: true, min: 1, default: 200},
    tol: {kind: 'number', min: 0, default: 1e-7},
    oosN: {kind: 'number', integer: true, min: 0, default: 0},
  },
  required: ['ranges'],
};

function normalizeStochasticLPParams(params: StochasticLPParams): NormalizedStochasticLPParams {
  const cost = params.cost && params.cost.length > 0 ? params.cost : params.c;
  const price = params.price && params.price.length > 0 ? params.price : params.p;
  if (!cost || !price) {
    throw new Error('stochastic-lp: provide cost/price or c/p arrays');
  }
  return {
    cost,
    price,
    ranges: params.ranges,
    numScenarios: params.numScenarios ?? params.N ?? 200,
    seed: params.seed ?? 42,
    budget: params.budget,
    maxIter: params.maxIter,
    tol: params.tol,
    oosN: params.oosN,
  };
}

function assertStochasticLPParams(params: NormalizedStochasticLPParams): void {
  if (params.cost.length !== params.price.length || params.cost.length !== params.ranges.length) {
    throw new Error(`stochastic-lp: cost, price, and ranges must have the same length`);
  }
  for (let i = 0; i < params.ranges.length; i++) {
    const [lo, hi] = params.ranges[i];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < 0 || hi < lo) {
      throw new Error(`stochastic-lp: ranges[${i}] must satisfy 0 <= low <= high`);
    }
  }
}

function addBar(shapes: Shape[], x: number, yBase: number, w: number, h: number, fill: string, label: string, title?: string): void {
  shapes.push({kind: 'rect', x, y: yBase - h, w, h, fill, stroke: '#334155', rx: 3, title});
  shapes.push({kind: 'text', x: x + w / 2, y: yBase + 14, text: label, fontSize: 10, anchor: 'middle', fill: '#334155'});
}

function lineChartSeries(trace: readonly number[], label: string, color: string): {label: string; color: string; t: number[]; y: number[]} {
  return {label, color, t: trace.map((_, i) => i + 1), y: trace.slice()};
}

const stochasticLPAdapter: DESModelRegistration<StochasticLPParams, StochasticLPAdapterResult> = {
  id: 'stochastic-lp',
  description: 'Two-stage stochastic LP via SAA monolithic solve and Benders/L-shaped DES cuts.',
  schema: stochasticLPSchema,
  async run(params, runtime) {
    const actual = normalizeStochasticLPParams(params);
    return withLogger(runtime, logger => {
      assertStochasticLPParams(actual);
      const slp = buildProductionSLP(actual.cost, actual.price, actual.budget);
      const scenarios = buildProductionScenarios({ranges: actual.ranges, seed: actual.seed}, actual.numScenarios);
      logger?.log({kind: 'stochastic-lp-start', level: 'info', numScenarios: actual.numScenarios, budget: actual.budget ?? null});
      const closedForm = actual.budget === undefined ? solveProductionClosedForm(actual.cost, actual.price, actual.ranges) : undefined;
      const monolithic = solveSLPMonolithic(slp, scenarios);
      const benders = solveSLPBenders(slp, scenarios, {tol: actual.tol ?? 1e-7, maxIter: actual.maxIter ?? 200});
      let outOfSample: StochasticLPAdapterResult['outOfSample'];
      if ((actual.oosN ?? 0) > 0) {
        const oos = buildProductionScenarios({ranges: actual.ranges, seed: actual.seed + 99991}, actual.oosN!);
        const evalX = (x: number[]): number => {
          let z = 0;
          for (let i = 0; i < actual.cost.length; i++) z += -actual.cost[i] * x[i];
          let q = 0;
          for (const sc of oos) {
            for (let i = 0; i < actual.price.length; i++) q += actual.price[i] * Math.min(x[i], sc.meta.D[i]);
          }
          return z + q / oos.length;
        };
        outOfSample = {
          N: actual.oosN!,
          monolithic: evalX(monolithic.x),
          benders: evalX(benders.x),
          closedForm: closedForm ? evalX(closedForm.x) : undefined,
        };
      }
      logger?.log({kind: 'stochastic-lp-finish', level: 'info', monoObjective: monolithic.objective, bendersObjective: benders.objective, iterations: benders.iterations});
      return {closedForm, monolithic, benders, outOfSample};
    });
  },
  summarize(r) {
    return [
      'STOCHASTIC LP',
      '------------------------',
      `  Monolithic: status=${r.monolithic.status} z=${r.monolithic.objective.toFixed(4)} iters=${r.monolithic.iterations}`,
      `  Benders:    status=${r.benders.status} z=${r.benders.objective.toFixed(4)} cuts=${r.benders.bendersTrace?.filter(t => t.cutAdded).length ?? 0}`,
      `  |Delta z|:  ${Math.abs(r.monolithic.objective - r.benders.objective).toExponential(3)}`,
      ...(r.closedForm ? [`  Closed form z*: ${r.closedForm.objective.toFixed(4)}`] : []),
      ...(r.outOfSample ? [`  OOS N=${r.outOfSample.N}: monolithic=${r.outOfSample.monolithic.toFixed(4)} benders=${r.outOfSample.benders.toFixed(4)}`] : []),
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['iter,upper_bound,lower_bound,gap,theta,expected_q'];
    for (const it of r.benders.bendersTrace ?? []) {
      lines.push(csvRow([it.iter, it.upperBound, it.lowerBound, it.gap, it.thetaMaster, it.expectedQ].map(v => Number(v).toFixed(8))));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'stochastic-lp');
    const trace = r.benders.bendersTrace ?? [];
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 900, height: 520, fps: 8, title: 'Two-stage stochastic LP', subtitle: 'Benders cuts as DES ticks'});
    const maxGap = Math.max(1, ...trace.map(t => t.gap));
    for (const it of trace) {
      rec.frame(it.iter, it.iter, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 900, h: 520, fill: '#f8fafc'},
          {kind: 'text', x: 450, y: 36, text: `Benders iteration ${it.iter}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
        ];
        addBar(shapes, 180, 390, 90, 260 * Math.max(0, it.upperBound) / Math.max(1, it.upperBound, it.lowerBound), '#60a5fa', 'UB');
        addBar(shapes, 320, 390, 90, 260 * Math.max(0, it.lowerBound) / Math.max(1, it.upperBound, it.lowerBound), '#34d399', 'LB');
        addBar(shapes, 520, 390, 90, 260 * Math.max(0, it.gap) / maxGap, '#f97316', 'gap');
        shapes.push({kind: 'text', x: 450, y: 450, text: `x=[${it.xMaster.map(v => v.toFixed(2)).join(', ')}]  gap=${it.gap.toExponential(2)}`, fontSize: 14, anchor: 'middle', fill: '#334155'});
        return {shapes, caption: `UB=${it.upperBound.toFixed(3)} LB=${it.lowerBound.toFixed(3)} E[Q]=${it.expectedQ.toFixed(3)}`};
      });
    }
    rec.setCharts([{x: 70, y: 70, w: 760, h: 150, title: 'Benders bounds', series: [
      {label: 'UB', color: '#2563eb', t: trace.map(t => t.iter), y: trace.map(t => t.upperBound)},
      {label: 'LB', color: '#059669', t: trace.map(t => t.iter), y: trace.map(t => t.lowerBound)},
    ]}]);
    await rec.finish();
  },
  examples: [{name: '2-product capacity planning', spec: {$schema: 'des/model-spec/v1', model: 'stochastic-lp', parameters: {cost: [10, 12], price: [25, 28], ranges: [[50, 100], [40, 80]], numScenarios: 200, seed: 42}, runtime: {animate: true}}}],
};

registerModel(stochasticLPAdapter);

registerModel<DistributionFitParams, DistributionFitResult>({
  id: 'distribution-fit',
  description: 'Fit demand/service samples by MLE and method of moments, then rank by AIC.',
  schema: fitParamsSchema,
  run(params) { return runDistributionFit(params); },
  summarize(r) {
    return [
      'DISTRIBUTION FIT',
      '------------------------',
      `  n=${r.samples.length} mean=${r.sampleMean.toFixed(4)} var=${r.sampleVariance.toFixed(4)}`,
      `  best=${r.bestByAIC.family}/${r.bestByAIC.method} AIC=${r.bestByAIC.aic.toFixed(3)}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['rank,family,method,aic,log_likelihood,mean,variance,params'];
    r.fits.forEach((f, i) => lines.push(csvRow([i + 1, f.family, f.method, f.aic.toFixed(8), f.logLikelihood.toFixed(8), f.mean.toFixed(8), f.variance.toFixed(8), JSON.stringify(f.params)])));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'distribution-fit');
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 900, height: 480, fps: 3, title: 'Distribution fitting', subtitle: 'MLE vs method of moments'});
    const aics = r.fits.map(f => f.aic);
    const aMin = Math.min(...aics), aMax = Math.max(...aics);
    rec.frame(0, 0, () => {
      const shapes: Shape[] = [
        {kind: 'rect', x: 0, y: 0, w: 900, h: 480, fill: '#f8fafc'},
        {kind: 'text', x: 450, y: 34, text: `Best fit: ${r.bestByAIC.family}/${r.bestByAIC.method}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
      ];
      r.fits.slice(0, 10).forEach((f, i) => {
        const h = 220 * (1 - (f.aic - aMin) / Math.max(1e-9, aMax - aMin));
        addBar(shapes, 70 + i * 80, 360, 44, Math.max(8, h), i === 0 ? '#22c55e' : '#60a5fa', `${i + 1}`, `${f.family}/${f.method} AIC=${f.aic.toFixed(2)}`);
        shapes.push({kind: 'text', x: 92 + i * 80, y: 383, text: f.family.slice(0, 4), fontSize: 9, anchor: 'middle', fill: '#334155'});
      });
      return {shapes, caption: `sample mean=${r.sampleMean.toFixed(3)} sample var=${r.sampleVariance.toFixed(3)}`};
    });
    await rec.finish();
  },
  examples: [{name: 'positive service times', spec: {$schema: 'des/model-spec/v1', model: 'distribution-fit', parameters: {samples: [8.2, 9.1, 10.4, 7.6, 12.3, 9.9, 11.1, 8.7, 10.8, 9.4], families: ['normal', 'lognormal', 'gamma', 'exponential'], methods: ['mle', 'moments']}, runtime: {animate: true}}}],
});

registerModel<RiskCapacityParams, RiskCapacityResult>({
  id: 'risk-capacity',
  description: 'Scenario capacity planning with expectation, CVaR, chance, or DRO-lite objectives.',
  schema: riskParamsSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'risk-capacity-start', level: 'info', risk: params.risk.kind, scenarios: params.numScenarios});
      const result = runRiskCapacity(params);
      logger?.log({kind: 'risk-capacity-finish', level: 'info', best: result.best});
      return result;
    });
  },
  summarize(r) {
    return [
      'RISK CAPACITY',
      '------------------------',
      `  risk=${r.params.risk.kind} scenarios=${r.scenarios.length}`,
      `  best x=[${r.best.x.join(', ')}] objective=${r.best.robustObjective.toFixed(3)}`,
      `  mean=${r.best.meanProfit.toFixed(3)} sd=${r.best.sdProfit.toFixed(3)} service=${(100 * r.best.serviceLevel).toFixed(1)}% CVaR(loss)=${r.best.cvarLoss.toFixed(3)}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['x,mean_profit,sd_profit,cvar_loss,service_level,robust_objective,feasible'];
    for (const c of r.candidates) lines.push(csvRow([JSON.stringify(c.x), c.meanProfit, c.sdProfit, c.cvarLoss, c.serviceLevel, c.robustObjective, c.feasible ? 1 : 0]));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'risk-capacity');
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 900, height: 500, fps: 5, title: 'Risk-aware capacity planning', subtitle: `${r.params.risk.kind} objective over scenario grid`});
    const sorted = r.candidates.slice().sort((a, b) => b.robustObjective - a.robustObjective).slice(0, 12);
    const maxObj = Math.max(...sorted.map(c => c.robustObjective), 1);
    rec.frame(0, 0, () => {
      const shapes: Shape[] = [{kind: 'rect', x: 0, y: 0, w: 900, h: 500, fill: '#f8fafc'}, {kind: 'text', x: 450, y: 34, text: `Best x=[${r.best.x.join(', ')}]`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'}];
      sorted.forEach((c, i) => addBar(shapes, 55 + i * 68, 370, 42, 260 * Math.max(0, c.robustObjective) / maxObj, c.feasible ? '#22c55e' : '#ef4444', `${i + 1}`, `x=${JSON.stringify(c.x)}`));
      return {shapes, caption: `objective=${r.best.robustObjective.toFixed(3)} service=${(100 * r.best.serviceLevel).toFixed(1)}%`};
    });
    await rec.finish();
  },
  examples: [{name: 'CVaR capacity', spec: {$schema: 'des/model-spec/v1', model: 'risk-capacity', parameters: {cost: [10, 12], price: [25, 28], demand: {kind: 'uniform', ranges: [{low: 50, high: 100}, {low: 40, high: 80}]}, numScenarios: 250, seed: 5, xMax: 120, step: 10, risk: {kind: 'cvar', alpha: 0.9, lambda: 0.2}}, runtime: {animate: true}}}],
});

registerModel<SDDPParams, SDDPResult>({
  id: 'sddp-capacity',
  description: 'Multi-stage stochastic capacity expansion via SDDP-style value-function cuts.',
  schema: sddpParamsSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => runCapacityExpansionSDDP(params, logger));
  },
  summarize(r) {
    return [
      'SDDP CAPACITY',
      '------------------------',
      `  horizon=${r.params.horizon} iterations=${r.trace.length}`,
      `  exact sampled-grid objective=${r.exactObjective.toFixed(4)}`,
      `  upper=${r.finalUpperBound.toFixed(4)} lower=${r.finalLowerBound.toFixed(4)} gap=${r.gap.toFixed(4)}`,
      `  cuts by stage=[${r.cuts.map(c => c.length).join(', ')}]`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['iter,upper_bound,lower_bound,exact_objective,gap,cut_counts,forward_states,forward_return'];
    for (const t of r.trace) lines.push(csvRow([t.iter, t.upperBound, t.lowerBound, t.exactObjective, t.gap, JSON.stringify(t.cutCounts), JSON.stringify(t.forwardStates), t.forwardReturn]));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'sddp-capacity');
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 920, height: 540, fps: 8, title: 'SDDP capacity expansion', subtitle: 'Multi-stage cuts, bounds, and forward sampled states'});
    const maxCuts = Math.max(1, ...r.trace.flatMap(t => t.cutCounts));
    for (const row of r.trace) {
      rec.frame(row.iter, row.iter, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 920, h: 540, fill: '#f8fafc'},
          {kind: 'text', x: 460, y: 34, text: `SDDP iteration ${row.iter}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
        ];
        row.cutCounts.forEach((c, i) => addBar(shapes, 90 + i * 95, 390, 48, 250 * c / maxCuts, '#60a5fa', `t${i}`, `${c} cuts`));
        shapes.push({kind: 'text', x: 460, y: 450, text: `states: ${row.forwardStates.map(v => v.toFixed(1)).join(' -> ')}`, fontSize: 13, anchor: 'middle', fill: '#334155'});
        return {shapes, caption: `upper=${row.upperBound.toFixed(3)} lower=${row.lowerBound.toFixed(3)} exact=${row.exactObjective.toFixed(3)} gap=${row.gap.toFixed(3)}`};
      });
    }
    rec.setCharts([{x: 80, y: 70, w: 760, h: 150, title: 'Bounds', series: [
      lineChartSeries(r.trace.map(t => t.upperBound), 'upper', '#2563eb'),
      lineChartSeries(r.trace.map(t => t.lowerBound), 'lower', '#059669'),
      lineChartSeries(r.trace.map(t => t.exactObjective), 'exact', '#111827'),
    ]}]);
    await rec.finish();
  },
  examples: [{name: '3-stage capacity expansion', spec: {$schema: 'des/model-spec/v1', model: 'sddp-capacity', parameters: {horizon: 3, demand: [{low: 20, high: 50}, {low: 30, high: 70}, {low: 40, high: 90}], price: [25, 24, 23], expansionCost: [12, 10, 8], initialCapacity: 0, xMax: 100, step: 10, samplesPerStage: 50, seed: 7, maxIter: 35, tol: 0.01}, runtime: {animate: true}}}],
});

registerModel<AdaptiveSimOptParams, AdaptiveSimOptResult>({
  id: 'adaptive-simopt',
  description: 'Adaptive simulation optimisation with sequential UCB allocation across candidate policies.',
  schema: adaptiveParamsSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => runAdaptiveSimOpt(params, logger));
  },
  summarize(r) {
    return [
      'ADAPTIVE SIMOPT',
      '------------------------',
      `  best=${r.best.name} x=[${r.best.x.join(', ')}] mean=${r.best.mean.toFixed(3)} stderr=${r.best.stderr.toFixed(3)} n=${r.best.n}`,
      `  total samples=${r.stats.reduce((s, a) => s + a.n, 0)} alternatives=${r.stats.length}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['name,x,n,mean,sd,stderr,ucb'];
    for (const s of r.stats) lines.push(csvRow([s.name, JSON.stringify(s.x), s.n, s.mean, s.sd, s.stderr, s.ucb]));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'adaptive-simopt');
    const rec = new FrameRecorder({framesPath: frames, htmlPath, width: 920, height: 520, fps: 10, title: 'Adaptive simulation optimisation', subtitle: 'Sequential sampling concentrates on promising candidates'});
    const maxMean = Math.max(1, ...r.stats.map(s => s.mean));
    for (let k = 0; k < Math.max(1, r.trace.length); k++) {
      const row = r.trace[Math.min(k, r.trace.length - 1)];
      rec.frame(k + 1, k + 1, () => {
        const shapes: Shape[] = [{kind: 'rect', x: 0, y: 0, w: 920, h: 520, fill: '#f8fafc'}, {kind: 'text', x: 460, y: 34, text: `Best: ${row?.bestName ?? r.best.name}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'}];
        r.stats.forEach((s, i) => addBar(shapes, 80 + i * 95, 370, 50, 250 * Math.max(0, s.mean) / maxMean, s.name === r.best.name ? '#22c55e' : '#60a5fa', s.name, `n=${s.n} mean=${s.mean.toFixed(2)}`));
        return {shapes, caption: row ? `iter=${row.iter} sampled=${row.sampled} total=${row.totalSamples} bestMean=${row.bestMean.toFixed(3)}` : ''};
      });
    }
    rec.setCharts([{x: 80, y: 70, w: 760, h: 130, title: 'Best sampled mean over time', series: [{label: 'best mean', color: '#059669', t: r.trace.map(t => t.iter), y: r.trace.map(t => t.bestMean)}]}]);
    await rec.finish();
  },
  examples: [{name: 'adaptive capacity candidates', spec: {$schema: 'des/model-spec/v1', model: 'adaptive-simopt', parameters: {cost: [10, 12], price: [25, 28], demand: {kind: 'uniform', ranges: [{low: 50, high: 100}, {low: 40, high: 80}]}, alternatives: [{name: 'lean', x: [60, 50]}, {name: 'balanced', x: [80, 65]}, {name: 'buffered', x: [100, 80]}], seed: 11, initialSamples: 5, budget: 120, batchSize: 5, exploration: 1.5}, runtime: {animate: true}}}],
});
