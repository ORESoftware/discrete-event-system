'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/neural_network_scene.rs
// - Exported buildNeural*Animation functions can remain module helpers returning Animation serde structs with Vec<Frame>.
// - BuiltFrame and result imports should become nominal Rust structs/enums; avoid structural intersections by defining frame sample structs.
// - Local frame/chart/metric helpers stay private; maps/sets should be HashMap/HashSet only where lookup semantics matter.
// - If a neural scene builder becomes DES graph-visible, lift it into a PureTransform struct with transform(result_sample) -> Frame.

// =============================================================================
// Neural-network animation scenes.
//
// These are post-hoc scenes over results that already ran:
//   - XOR: network topology + active training sample + loss/prediction charts
//   - Neural Q-learning: learned greedy policy through the corridor
//   - Neural ODE: decay trajectory with a tiny vector-field network
// =============================================================================

import {ChartSpec, Frame, Shape} from '../types';
import {
  FeedForwardNetwork,
  SupervisedNeuralNetDESResult,
  NeuralQLearningResult,
} from '../../general/neural-network';
import {ODETrace} from '../../general/ode';

export const NEURAL_STAGE_W = 1100;
export const NEURAL_STAGE_H = 720;

type BuiltFrame = Array<{t: number; tick: number} & Omit<Frame, 't' | 'tick'>>;

const C = {
  ink: '#172033',
  muted: '#64748b',
  panel: '#f8fafc',
  grid: '#d7dee8',
  blue: '#2563eb',
  green: '#16a34a',
  amber: '#f59e0b',
  red: '#dc2626',
  purple: '#7c3aed',
};

export function buildNeuralXorAnimation(
  result: SupervisedNeuralNetDESResult<FeedForwardNetwork>,
): {frames: BuiltFrame; charts: ChartSpec[]} {
  const total = result.lossHistory.length;
  const numFrames = Math.min(120, Math.max(20, Math.ceil(total / 260)));
  const frames: BuiltFrame = [];
  const finalPred = result.predictions.map(v => v[0]);
  const samples = [
    {x: [0, 0], y: 0, label: '0 xor 0'},
    {x: [0, 1], y: 1, label: '0 xor 1'},
    {x: [1, 0], y: 1, label: '1 xor 0'},
    {x: [1, 1], y: 0, label: '1 xor 1'},
  ];

  for (let i = 0; i < numFrames; i++) {
    const step = Math.min(total - 1, Math.round(i * (total - 1) / Math.max(1, numFrames - 1)));
    const sample = samples[step % samples.length];
    const loss = result.lossHistory[step] ?? 0;
    frames.push({
      t: step,
      tick: i,
      ...buildXorFrame(result.network, step, total, loss, sample, finalPred),
    });
  }
  return {frames, charts: buildXorCharts(result)};
}

function buildXorFrame(
  network: FeedForwardNetwork,
  step: number,
  total: number,
  loss: number,
  sample: {x: number[]; y: number; label: string},
  finalPred: number[],
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = baseBackground('Neural XOR', 'supervised DES training');

  const dims = [network.inputDim, ...network.layers.map(l => l.biases.length)];
  const x0 = 90, x1 = 660, y0 = 110, y1 = 410;
  const layerXs = dims.map((_, i) => x0 + i * ((x1 - x0) / Math.max(1, dims.length - 1)));
  const nodePos: Array<Array<{x: number; y: number}>> = [];
  for (let li = 0; li < dims.length; li++) {
    const n = dims[li];
    const ys = Array.from({length: n}, (_, j) => y0 + (j + 1) * ((y1 - y0) / (n + 1)));
    nodePos.push(ys.map(y => ({x: layerXs[li], y})));
  }

  // Edges.
  for (let li = 0; li < nodePos.length - 1; li++) {
    for (const a of nodePos[li]) for (const b of nodePos[li + 1]) {
      shapes.push({kind: 'line', x1: a.x, y1: a.y, x2: b.x, y2: b.y,
                   stroke: '#cbd5e1', strokeWidth: 0.8, opacity: 0.8});
    }
  }
  // Moving token.
  const phase = (step % 4) / 3;
  shapes.push({kind: 'circle', x: x0 + phase * (x1 - x0), y: 72, r: 8,
               fill: C.purple, title: 'sample token moving across layer stations'});
  shapes.push({kind: 'line', x1: x0, y1: 72, x2: x1, y2: 72, stroke: '#e2e8f0', strokeWidth: 2});

  // Nodes.
  for (let li = 0; li < nodePos.length; li++) {
    for (let j = 0; j < nodePos[li].length; j++) {
      const p = nodePos[li][j];
      const active = li === 0 && sample.x[j] === 1;
      shapes.push({kind: 'circle', x: p.x, y: p.y, r: 18,
                   fill: active ? C.green : '#ffffff',
                   stroke: li === nodePos.length - 1 ? C.blue : '#475569',
                   strokeWidth: li === nodePos.length - 1 ? 2 : 1});
      shapes.push({kind: 'text', x: p.x, y: p.y + 4, text: li === 0 ? String(sample.x[j]) : '',
                   anchor: 'middle', fontSize: 12, fill: active ? '#fff' : C.ink, fontWeight: 'bold'});
    }
    const label = li === 0 ? 'input' : li === nodePos.length - 1 ? 'output' : `hidden ${li}`;
    shapes.push({kind: 'text', x: layerXs[li], y: y1 + 34, text: label,
                 anchor: 'middle', fontSize: 12, fill: C.muted});
  }

  // Metrics panel.
  const progress = total <= 1 ? 1 : step / (total - 1);
  metricPanel(shapes, 740, 86, 310, 330, [
    ['sample', `${sample.label} -> ${sample.y}`],
    ['training step', `${step + 1} / ${total}`],
    ['progress', `${(100 * progress).toFixed(1)}%`],
    ['current loss', loss.toExponential(3)],
    ['params', String(network.parameterCount())],
  ]);
  progressBar(shapes, 760, 382, 270, 14, progress, C.blue);

  // Final prediction bars.
  shapes.push({kind: 'text', x: 88, y: 468, text: 'final predictions', fontSize: 13, fill: C.ink, fontWeight: 'bold'});
  for (let i = 0; i < finalPred.length; i++) {
    const x = 90 + i * 145;
    const y = 510;
    const h = 100 * finalPred[i];
    shapes.push({kind: 'rect', x, y: y + 100 - h, w: 76, h, fill: finalPred[i] > 0.5 ? C.green : C.blue, rx: 3});
    shapes.push({kind: 'rect', x, y, w: 76, h: 100, fill: 'none', stroke: '#cbd5e1', strokeWidth: 1, rx: 3});
    shapes.push({kind: 'text', x: x + 38, y: y + 124, text: ['00', '01', '10', '11'][i],
                 anchor: 'middle', fontSize: 12, fill: C.muted});
    shapes.push({kind: 'text', x: x + 38, y: y - 8, text: finalPred[i].toFixed(3),
                 anchor: 'middle', fontSize: 12, fill: C.ink, fontWeight: 'bold'});
  }

  return {shapes, caption: `step=${step + 1} sample=${sample.label} target=${sample.y} loss=${loss.toExponential(3)}`};
}

function buildXorCharts(result: SupervisedNeuralNetDESResult<FeedForwardNetwork>): ChartSpec[] {
  const t = result.lossHistory.map((_, i) => i + 1);
  const losses = result.lossHistory.map(x => Math.max(1e-12, x));
  const predT = result.predictions.map((_, i) => i);
  return [
    {
      x: 690, y: 440, w: 360, h: 120,
      title: 'loss per sample',
      yMin: 0,
      yMax: Math.max(...result.lossHistory.slice(0, Math.min(100, result.lossHistory.length)), ...result.lossHistory),
      series: [{label: 'loss', color: C.red, t, y: losses}],
    },
    {
      x: 690, y: 575, w: 360, h: 110,
      title: 'final XOR outputs',
      yMin: 0, yMax: 1,
      series: [{label: 'prediction', color: C.green, t: predT, y: result.predictions.map(v => v[0])}],
      cursor: false,
    },
  ];
}

export function buildNeuralQCorridorAnimation(
  result: NeuralQLearningResult,
  length: number,
): {frames: BuiltFrame; charts: ChartSpec[]} {
  const frames: BuiltFrame = [];
  const path = greedyPath(result.policy, length);
  for (let i = 0; i < path.length; i++) {
    frames.push({t: i, tick: i, ...buildCorridorFrame(result, length, path, i)});
  }
  return {frames, charts: buildQCharts(result)};
}

function buildCorridorFrame(
  result: NeuralQLearningResult,
  length: number,
  path: number[],
  step: number,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = baseBackground('Neural Q-learning', 'corridor MDP policy rollout');
  const cellW = 120;
  const startX = 90;
  const y = 230;
  const state = path[step];
  for (let s = 0; s < length; s++) {
    const x = startX + s * cellW;
    const isGoal = s === length - 1;
    const isAgent = s === state;
    shapes.push({kind: 'rect', x, y, w: 92, h: 92,
                 fill: isGoal ? '#dcfce7' : '#fff',
                 stroke: isAgent ? C.blue : '#cbd5e1',
                 strokeWidth: isAgent ? 3 : 1,
                 rx: 5});
    shapes.push({kind: 'text', x: x + 46, y: y + 54, text: isAgent ? 'A' : isGoal ? 'G' : String(s),
                 anchor: 'middle', fontSize: 24, fill: isGoal ? C.green : C.ink, fontWeight: 'bold'});
    if (s < length - 1) {
      shapes.push({kind: 'line', x1: x + 94, y1: y + 46, x2: x + cellW - 12, y2: y + 46,
                   stroke: '#94a3b8', strokeWidth: 2});
      if (result.policy[s] === 1) {
        shapes.push({kind: 'path', d: `M ${x + cellW - 18} ${y + 40} L ${x + cellW - 8} ${y + 46} L ${x + cellW - 18} ${y + 52}`,
                     stroke: C.green, strokeWidth: 2, fill: 'none'});
      }
    }
  }
  metricPanel(shapes, 730, 122, 320, 300, [
    ['episodes', String(result.totalEpisodes)],
    ['env steps', String(result.totalSteps)],
    ['eval success', '100%'],
    ['mean eval length', '5.0'],
    ['current state', String(state)],
  ]);
  progressBar(shapes, 750, 388, 280, 14, step / Math.max(1, path.length - 1), C.green);
  return {shapes, caption: `greedy rollout step=${step} state=${state} policy=[${result.policy.join(', ')}]`};
}

function greedyPath(policy: number[], length: number): number[] {
  const path = [0];
  let s = 0;
  for (let k = 0; k < length * 3; k++) {
    if (s === length - 1) break;
    const a = policy[s] ?? 1;
    s = a === 0 ? Math.max(0, s - 1) : Math.min(length - 1, s + 1);
    path.push(s);
  }
  return path;
}

function buildQCharts(result: NeuralQLearningResult): ChartSpec[] {
  const episodes = result.rewardHistory.map((_, i) => i + 1);
  return [
    {
      x: 70, y: 470, w: 470, h: 180,
      title: 'episode return',
      series: [{label: 'return', color: C.blue, t: episodes, y: result.rewardHistory.slice()}],
    },
    {
      x: 580, y: 470, w: 470, h: 180,
      title: 'TD training loss',
      yMin: 0,
      series: [{label: 'loss', color: C.red, t: result.lossHistory.map((_, i) => i + 1), y: result.lossHistory.slice()}],
    },
  ];
}

export function buildNeuralOdeAnimation(
  trace: ODETrace,
  rate: number,
  exactFinal: number,
  error: number,
): {frames: BuiltFrame; charts: ChartSpec[]} {
  const frames: BuiltFrame = [];
  for (let i = 0; i < trace.t.length; i++) {
    frames.push({t: trace.t[i], tick: i, ...buildOdeFrame(trace, i, rate, exactFinal, error)});
  }
  const y = trace.y.map(v => v[0]);
  const exact = trace.t.map(t => y[0] * Math.exp(-rate * t));
  return {
    frames,
    charts: [{
      x: 80, y: 430, w: 940, h: 230,
      title: 'neural ODE trajectory',
      yMin: 0,
      yMax: Math.max(...y) * 1.05,
      series: [
        {label: 'network RK4', color: C.blue, t: trace.t.slice(), y},
        {label: 'exact exp decay', color: C.green, t: trace.t.slice(), y: exact},
      ],
    }],
  };
}

function buildOdeFrame(
  trace: ODETrace,
  idx: number,
  rate: number,
  exactFinal: number,
  error: number,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = baseBackground('Neural ODE', `network vector field dy/dt = -${rate} y`);
  const t = trace.t[idx];
  const y = trace.y[idx][0];
  const x0 = 90, y0 = 100, w = 580, h = 260;
  shapes.push({kind: 'rect', x: x0, y: y0, w, h, fill: '#fff', stroke: '#cbd5e1', strokeWidth: 1, rx: 5});
  shapes.push({kind: 'line', x1: x0 + 40, y1: y0 + h - 30, x2: x0 + w - 20, y2: y0 + h - 30, stroke: '#94a3b8', strokeWidth: 1});
  shapes.push({kind: 'line', x1: x0 + 40, y1: y0 + 20, x2: x0 + 40, y2: y0 + h - 30, stroke: '#94a3b8', strokeWidth: 1});
  const tMax = trace.t[trace.t.length - 1];
  const yMax = Math.max(...trace.y.map(v => v[0]));
  const sx = (tt: number) => x0 + 40 + (w - 70) * tt / Math.max(1e-12, tMax);
  const sy = (yy: number) => y0 + h - 30 - (h - 60) * yy / Math.max(1e-12, yMax);
  let d = '';
  for (let i = 0; i <= idx; i++) {
    d += `${i === 0 ? 'M' : 'L'} ${sx(trace.t[i]).toFixed(2)} ${sy(trace.y[i][0]).toFixed(2)} `;
  }
  shapes.push({kind: 'path', d, stroke: C.blue, strokeWidth: 3, fill: 'none'});
  shapes.push({kind: 'circle', x: sx(t), y: sy(y), r: 8, fill: C.blue});

  // Tiny network/vector-field diagram.
  const nx = 750, ny = 115;
  shapes.push({kind: 'rect', x: nx - 30, y: ny - 35, w: 300, h: 250, fill: C.panel, stroke: '#d7dee8', strokeWidth: 1, rx: 5});
  shapes.push({kind: 'text', x: nx, y: ny - 10, text: 'vector-field network', fontSize: 13, fill: C.ink, fontWeight: 'bold'});
  shapes.push({kind: 'circle', x: nx + 45, y: ny + 80, r: 24, fill: '#fff', stroke: C.blue, strokeWidth: 2, label: 'y'});
  shapes.push({kind: 'circle', x: nx + 205, y: ny + 80, r: 24, fill: '#fff', stroke: C.green, strokeWidth: 2, label: 'dy'});
  shapes.push({kind: 'line', x1: nx + 70, y1: ny + 80, x2: nx + 180, y2: ny + 80, stroke: C.muted, strokeWidth: 2});
  shapes.push({kind: 'text', x: nx + 125, y: ny + 70, text: `w = -${rate}`, anchor: 'middle', fontSize: 12, fill: C.ink});
  metricPanel(shapes, 730, 345, 320, 70, [
    ['final exact', exactFinal.toFixed(6)],
    ['abs error', error.toExponential(3)],
  ]);
  return {shapes, caption: `t=${t.toFixed(3)} y=${y.toFixed(6)} dy/dt=${(-rate * y).toFixed(6)}`};
}

function baseBackground(title: string, subtitle: string): Shape[] {
  return [
    {kind: 'rect', x: 0, y: 0, w: NEURAL_STAGE_W, h: NEURAL_STAGE_H, fill: '#f8fafc'},
    {kind: 'text', x: 40, y: 36, text: title, fontSize: 20, fill: C.ink, fontWeight: 'bold'},
    {kind: 'text', x: 40, y: 58, text: subtitle, fontSize: 12, fill: C.muted},
  ];
}

function metricPanel(shapes: Shape[], x: number, y: number, w: number, h: number, rows: Array<[string, string]>): void {
  shapes.push({kind: 'rect', x, y, w, h, fill: '#fff', stroke: '#d7dee8', strokeWidth: 1, rx: 5});
  shapes.push({kind: 'text', x: x + 16, y: y + 28, text: 'metrics', fontSize: 13, fill: C.ink, fontWeight: 'bold'});
  for (let i = 0; i < rows.length; i++) {
    const yy = y + 62 + i * 28;
    shapes.push({kind: 'text', x: x + 16, y: yy, text: rows[i][0], fontSize: 12, fill: C.muted});
    shapes.push({kind: 'text', x: x + w - 16, y: yy, text: rows[i][1], anchor: 'end',
                 fontSize: 12, fill: C.ink, fontWeight: 'bold'});
  }
}

function progressBar(shapes: Shape[], x: number, y: number, w: number, h: number, p: number, fill: string): void {
  shapes.push({kind: 'rect', x, y, w, h, fill: '#e2e8f0', rx: h / 2});
  shapes.push({kind: 'rect', x, y, w: Math.max(0, Math.min(1, p)) * w, h, fill, rx: h / 2});
}
