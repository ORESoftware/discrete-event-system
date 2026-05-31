// RUST MIGRATION: Target module `src/des/general/adapters/nonlinear_forecasting_adapter.rs`.
// RUST MIGRATION: Convert nonlinear forecasting adapter registration and animation helpers into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map forecast params, station panels, traces, and frame captions to `serde` config/result structs; runtime paths become `PathBuf`.
// RUST MIGRATION: Return `Result<_, ValidationError>` for invalid horizon, station, and forecast input validation.
'use strict';

// JSON adapter for nonlinear MDP/POMDP forecasting.

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {registerModel} from '../des-registry';
import {DESRuntimeConfig, ParamSchema} from '../des-spec';
import {
  NonlinearMDPPOMDPForecastParams,
  NonlinearMDPPOMDPForecastResult,
  runNonlinearMDPPOMDPForecast,
} from '../nonlinear-forecasting-model';
import {csvRow, framesPath, writeCsvLines} from './adapter-utils';

const nonlinearForecastSchema: ParamSchema = {
  kind: 'object',
  fields: {
    trainingPeriods: {kind: 'number', integer: true, min: 18, max: 200, default: 42},
    forecastHorizon: {kind: 'number', integer: true, min: 1, max: 80, default: 8},
    mdpBudget: {kind: 'number', integer: true, min: 1, max: 10, default: 6},
    ridge: {kind: 'number', min: 0, default: 0.03},
    fineTuneIterations: {kind: 'number', integer: true, min: 1, max: 200, default: 18},
    validationShare: {kind: 'number', min: 0.1, max: 0.5, default: 0.25},
  },
  required: [],
};

registerModel<NonlinearMDPPOMDPForecastParams, NonlinearMDPPOMDPForecastResult>({
  id: 'nonlinear-mdp-pomdp-forecast',
  description: 'Nonlinear forecasting: POMDP latent-variable discovery plus MDP feature selection and equation fine-tuning.',
  schema: nonlinearForecastSchema,
  run(params) { return runNonlinearMDPPOMDPForecast(params); },
  summarize(result) {
    return [
      'NONLINEAR MDP/POMDP FORECAST',
      '----------------------------------------',
      `  Selected variables:       ${result.selectedVariables.join(', ')}`,
      `  Validation MSE:           ${result.metrics.validationMse.toFixed(4)} (baseline ${result.metrics.baselineValidationMse.toFixed(4)})`,
      `  Forecast MSE:             ${result.metrics.forecastMse.toFixed(4)} (baseline ${result.metrics.baselineForecastMse.toFixed(4)})`,
      `  POMDP final entropy:      ${result.metrics.finalBeliefEntropy.toFixed(4)}`,
      `  MDP states/actions:       ${result.mdp.states}/${result.mdp.actions}`,
      `  Equation:                 ${result.equation.equationText}`,
      `  Stations:                 ${result.topology.stations.join(' -> ')}`,
      `  Movables:                 ${result.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(result, csvPath) {
    const lines = [csvRow(['kind', 't', 'actual', 'predicted', 'lower', 'upper', 'split_or_belief_mode'])];
    for (const row of result.equation.fitted) {
      lines.push(csvRow(['fit', row.t, row.actual, row.predicted, '', '', row.split]));
    }
    for (const row of result.projection) {
      lines.push(csvRow(['forecast', row.t, row.actual, row.forecast, row.lower, row.upper, row.beliefMode]));
    }
    writeCsvLines(csvPath, lines);
  },
  animate(result, _params, runtime) {
    return animateNonlinearForecast(result, runtime);
  },
});

async function animateNonlinearForecast(result: NonlinearMDPPOMDPForecastResult, runtime: DESRuntimeConfig): Promise<void> {
  const {htmlPath, frames} = framesPath(runtime, 'nonlinear-mdp-pomdp-forecast');
  const rec = new FrameRecorder({
    framesPath: frames,
    htmlPath,
    width: 960,
    height: 580,
    fps: 4,
    title: 'Nonlinear MDP/POMDP Forecast',
    subtitle: 'Hidden-state beliefs, MDP variable discovery, equation tuning, and projection as DES movables',
    background: '#f8fafc',
  });
  const frameCount = Math.max(result.equation.trace.length, result.projection.length, result.mdp.actionTrace.length, 1);
  for (let i = 0; i < frameCount; i++) {
    rec.frame(i, i, () => buildForecastFrame(result, i));
  }
  rec.setCharts([
    {
      x: 38, y: 410, w: 270, h: 130, title: 'Equation fine-tuning', yLabel: 'MSE',
      series: [
        {label: 'all rows', color: '#7c3aed', t: result.equation.trace.map(r => r.iter), y: result.equation.trace.map(r => r.mse)},
        {label: 'validation', color: '#dc2626', t: result.equation.trace.map(r => r.iter), y: result.equation.trace.map(r => r.validationMse)},
      ],
    },
    {
      x: 345, y: 410, w: 270, h: 130, title: 'Fit and projection', yLabel: 'outcome',
      series: [
        {label: 'actual fit', color: '#334155', t: result.equation.fitted.map(r => r.t), y: result.equation.fitted.map(r => r.actual)},
        {label: 'fitted', color: '#2563eb', t: result.equation.fitted.map(r => r.t), y: result.equation.fitted.map(r => r.predicted)},
        {label: 'forecast', color: '#059669', t: result.projection.map(r => r.t), y: result.projection.map(r => r.forecast)},
      ],
    },
    {
      x: 652, y: 410, w: 270, h: 130, title: 'POMDP latent beliefs', yLabel: 'probability',
      yMin: 0, yMax: 1,
      series: [
        {label: 'expansion', color: '#059669', t: result.pomdp.points.map(p => p.t), y: result.pomdp.points.map(p => p.posterior[1])},
        {label: 'contraction', color: '#f97316', t: result.pomdp.points.map(p => p.t), y: result.pomdp.points.map(p => p.posterior[2])},
        {label: 'shock', color: '#dc2626', t: result.pomdp.points.map(p => p.t), y: result.pomdp.points.map(p => p.posterior[3])},
      ],
    },
  ]);
  await rec.finish();
}

function buildForecastFrame(result: NonlinearMDPPOMDPForecastResult, frame: number): {shapes: Shape[]; caption: string} {
  const shapes: Shape[] = [{kind: 'rect', x: 0, y: 0, w: 960, h: 580, fill: '#f8fafc'}];
  shapes.push({kind: 'text', x: 38, y: 34, text: 'Nonlinear MDP/POMDP forecast', fontSize: 21, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: 38, y: 58, text: `validation MSE ${result.metrics.validationMse.toFixed(3)} vs baseline ${result.metrics.baselineValidationMse.toFixed(3)}   forecast MSE ${result.metrics.forecastMse.toFixed(3)}`, fontSize: 13, fill: '#475569'});
  drawStationPipeline(shapes, frame);
  drawVariables(shapes, result);
  drawActivePanels(shapes, result, frame);
  const tune = result.equation.trace[Math.min(frame, result.equation.trace.length - 1)];
  return {shapes, caption: `frame ${frame + 1}: MSE ${tune ? tune.mse.toFixed(3) : result.metrics.inSampleMse.toFixed(3)} with ${result.selectedVariables.length} variables`};
}

function drawStationPipeline(shapes: Shape[], frame: number): void {
  const nodes = [
    {label: 'DataSource', x: 42, y: 94, w: 124, fill: '#dbeafe'},
    {label: 'POMDP belief', x: 205, y: 94, w: 132, fill: '#dcfce7'},
    {label: 'MDP discovery', x: 380, y: 94, w: 132, fill: '#ede9fe'},
    {label: 'Equation tune', x: 555, y: 94, w: 132, fill: '#ffedd5'},
    {label: 'Projection', x: 730, y: 94, w: 124, fill: '#f1f5f9'},
  ];
  for (let i = 0; i < nodes.length - 1; i++) {
    const active = i === frame % (nodes.length - 1);
    shapes.push({kind: 'line', x1: nodes[i].x + nodes[i].w, y1: 123, x2: nodes[i + 1].x, y2: 123, stroke: active ? '#7c3aed' : '#94a3b8', strokeWidth: active ? 4 : 2, opacity: active ? 0.95 : 0.65});
  }
  for (const node of nodes) {
    shapes.push({kind: 'rect', x: node.x, y: node.y, w: node.w, h: 58, fill: node.fill, stroke: '#334155', strokeWidth: 1.2, rx: 7});
    shapes.push({kind: 'text', x: node.x + node.w / 2, y: 121, text: node.label, fontSize: 12, anchor: 'middle', fill: '#0f172a', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: node.x + node.w / 2, y: 139, text: 'station', fontSize: 10, anchor: 'middle', fill: '#475569'});
  }
  const edge = frame % (nodes.length - 1);
  shapes.push({kind: 'circle', x: (nodes[edge].x + nodes[edge].w + nodes[edge + 1].x) / 2, y: 123, r: 10, fill: '#7c3aed', stroke: '#ffffff', strokeWidth: 2});
}

function drawVariables(shapes: Shape[], result: NonlinearMDPPOMDPForecastResult): void {
  shapes.push({kind: 'rect', x: 42, y: 190, w: 360, h: 180, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 7});
  shapes.push({kind: 'text', x: 60, y: 216, text: 'Variables discovered by MDP', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  for (let i = 0; i < result.discoveredVariables.length; i++) {
    const v = result.discoveredVariables[i];
    const y = 242 + i * 20;
    const color = v.source === 'pomdp' ? '#059669' : v.source === 'nonlinear' ? '#7c3aed' : v.source === 'lagged' ? '#f97316' : '#2563eb';
    shapes.push({kind: 'circle', x: 62, y: y - 4, r: 5, fill: color});
    shapes.push({kind: 'text', x: 76, y, text: `${v.id} (${v.source})`, fontSize: 11, fill: '#334155'});
  }
}

function drawActivePanels(shapes: Shape[], result: NonlinearMDPPOMDPForecastResult, frame: number): void {
  shapes.push({kind: 'rect', x: 430, y: 190, w: 230, h: 180, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 7});
  shapes.push({kind: 'text', x: 448, y: 216, text: 'MDP action trace', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  for (let i = 0; i < Math.min(6, result.mdp.actionTrace.length); i++) {
    const row = result.mdp.actionTrace[i];
    const active = i === Math.min(frame, result.mdp.actionTrace.length - 1);
    shapes.push({kind: 'text', x: 448, y: 242 + i * 20, text: `${i + 1}. ${row.action}`, fontSize: 11, fill: active ? '#7c3aed' : '#334155', fontWeight: active ? 'bold' : 'normal'});
  }
  shapes.push({kind: 'rect', x: 690, y: 190, w: 230, h: 180, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 7});
  shapes.push({kind: 'text', x: 708, y: 216, text: 'Forecast projection', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  const row = result.projection[Math.min(frame, result.projection.length - 1)];
  const lines = row ? [
    `t=${row.t}  h=${row.horizonStep}`,
    `forecast ${row.forecast.toFixed(2)}`,
    `actual ${row.actual.toFixed(2)}`,
    `band [${row.lower.toFixed(1)}, ${row.upper.toFixed(1)}]`,
    `belief ${row.beliefMode}`,
    `entropy ${row.beliefEntropy.toFixed(3)}`,
  ] : [];
  for (let i = 0; i < lines.length; i++) {
    shapes.push({kind: 'text', x: 708, y: 244 + i * 20, text: lines[i], fontSize: 12, fill: '#334155'});
  }
}
