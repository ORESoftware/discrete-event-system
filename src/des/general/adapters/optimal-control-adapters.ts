// RUST MIGRATION: Target module `src/des/general/adapters/optimal_control_adapters.rs`.
// RUST MIGRATION: Convert LQR/MPC/optimal-control adapter registrations and render helpers into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Encode state-space configs, trajectories, controls, costs, and solver results as `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Return `Result<_, ValidationError>` for matrix dimensions, horizon bounds, and controller validation.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/optimal-control-adapters.rs
//   (module des::general::adapters::optimal_control_adapters)
// 1:1 file move. Registers seven optimal-control JSON adapters (Pontryagin,
// Kalman, sliding-mode, MRAC, ILC, feedback-linearization, MPC) + an ILC animation.
//
// Declarations → Rust:
//   const pontryaginSchema/kalmanSchema/slidingModeSchema/mracSchema/ilcSchema/
//         feedbackLinSchema/mpcSchema: ParamSchema -> serde + validator metadata
//   interface FlatFeedbackLinParams            -> struct (Option fields)
//   registerModel(...) x7                       -> one struct + impl ModelAdapter trait each
//   fn buildILCFrame / metricBar / drawMiniSeries / formatILCNumber -> plain `fn` helpers
//
// Conversion notes (file-specific):
//   - GotChA: many `run` bodies coerce decoded arrays via numberPair(p.x0, ..)/
//     optionalNumberPair(p.Q, ..) into `[number, number]` — in Rust deserialize
//     directly into `[f64; 2]` / `Option<[f64; 2]>`, no coercion.
//   - GotChA: feedback-linearization remaps a FLAT FlatFeedbackLinParams into the
//     nested FeedbackLinearizationOpts{params:{m,l,g,c}, ..} inside `run` — model as
//     two structs + a From/build step.
//   - `disturbanceType: 'sin'|'square'|'random'`, `referenceKind`, `quadrature`
//     literal unions -> enums; 'random' disturbance + seed -> inject seeded RNG.
//   - drawMiniSeries/edges emit SVG `path` `d` strings -> a path-builder; Shapes ->
//     Vec<Shape> (enum). Animations derive from result traces (no RNG).
// =============================================================================

// =============================================================================
// general/adapters/optimal-control-adapters.ts — JSON adapters for the
// entity-based optimal-control models added in this batch:
//
//   • pontryagin-bang-bang  — time-optimal control via Pontryagin's MP
//   • kalman-filter         — linear KF on a noisy radar tracking problem
//   • sliding-mode          — robust control under matched disturbance
//   • mrac                  — model-reference adaptive control
//   • iterative-learning-control — repeated-trial feedforward learning
//   • feedback-linearization — nonlinear pendulum tracking
//   • mpc-double-integrator — receding-horizon QP-based MPC
//
// Each adapter follows the `DESModelRegistration<P, R>` contract from
// `des-spec.ts`, identical to the patterns in the MDP-adjacent and
// queueing adapters.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';

import {runPontryaginBangBang, PontryaginOpts, PontryaginResult}
  from '../pontryagin-bang-bang';
import {runRadarTracking, RadarTrackingOpts, RadarTrackingResult}
  from '../kalman-filter';
import {runSlidingMode, SlidingModeOpts, SlidingModeResult}
  from '../sliding-mode-control';
import {runMRAC, MRACOpts, MRACResult} from '../mrac';
import {
  IterativeLearningControlParams,
  IterativeLearningControlResult,
  runIterativeLearningControl,
} from '../iterative-learning-control';
import {runFeedbackLinearization, FeedbackLinearizationOpts, FeedbackLinearizationResult}
  from '../feedback-linearization';
import {runMPCDoubleIntegrator, MPCDoubleIntOpts, MPCDoubleIntResult}
  from '../mpc-double-integrator';
import {csvRow, framesPath, numberPair, optionalNumberPair, writeCsvLines} from './adapter-utils';

// -----------------------------------------------------------------------------
// 1. pontryagin-bang-bang
// -----------------------------------------------------------------------------

const pontryaginSchema: ParamSchema = {
  kind: 'object',
  description: 'Time-optimal control of a double integrator via Pontryagin\'s Maximum Principle (bang-bang).',
  fields: {
    x0: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
    uMax: {kind: 'number', min: 0, default: 1},
    dt: {kind: 'number', min: 1e-6, default: 0.02},
    numSteps: {kind: 'number', integer: true, min: 1, default: 500},
    deadband: {kind: 'number', min: 0, default: 0.1},
  },
  required: [],
};

registerModel<PontryaginOpts, PontryaginResult>({
  id: 'pontryagin-bang-bang',
  description: 'Pontryagin Maximum Principle: time-optimal bang-bang control on a double integrator.',
  schema: pontryaginSchema,
  run(p) {
    return runPontryaginBangBang({...p, x0: numberPair(p.x0, [3, 0], 'x0')});
  },
  summarize(r, p) {
    const dt = p.dt ?? 0.02;
    return [
      'PONTRYAGIN BANG-BANG (time-optimal)',
      '────────────────────────────────────',
      `  Initial state:           [${(p.x0 ?? [3, 0]).join(', ')}]`,
      `  |u| bound:               ${p.uMax ?? 1}`,
      `  Bang-bang switches:      ${r.switchCount}  (PMP predicts ≤ 1)`,
      `  Arrival tick:            ${r.arrivalTick}    (entered deadband)`,
      `  Arrival time:            ${(r.arrivalTick * dt).toFixed(3)} s`,
      `  Theoretical optimum t*:  ${r.theoreticalArrivalTime.toFixed(3)} s`,
      `  Final state:             [${r.trajectory[r.trajectory.length - 1].map(x => x.toFixed(3)).join(', ')}]`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,x,v,u'];
    for (let i = 0; i < r.controls.length; i++) {
      lines.push(csvRow([i, r.trajectory[i][0].toFixed(6), r.trajectory[i][1].toFixed(6), r.controls[i][0].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 2. kalman-filter
// -----------------------------------------------------------------------------

const kalmanSchema: ParamSchema = {
  kind: 'object',
  description: 'Linear Kalman filter on a noisy 1-D radar tracking problem.',
  fields: {
    x0: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
    dt: {kind: 'number', min: 1e-6, default: 0.1},
    numSteps: {kind: 'number', integer: true, min: 1, default: 200},
    procNoiseStd: {kind: 'number', min: 0, default: 0.1},
    measNoiseStd: {kind: 'number', min: 0, default: 1.0},
    P0Scale: {kind: 'number', min: 0, default: 10},
    seed: {kind: 'number', integer: true, default: 1},
  },
  required: [],
};

registerModel<RadarTrackingOpts, RadarTrackingResult>({
  id: 'kalman-filter',
  description: 'Linear Kalman filter — radar tracking of a constant-velocity target with position-only sensor.',
  schema: kalmanSchema,
  run(p) {
    return runRadarTracking({...p, x0: numberPair(p.x0, [0, 1], 'x0')});
  },
  summarize(r, p) {
    return [
      'KALMAN FILTER — RADAR TRACKING',
      '────────────────────────────────────',
      `  Process noise σ_w:       ${p.procNoiseStd ?? 0.1}`,
      `  Sensor noise σ_v:        ${p.measNoiseStd ?? 1.0}`,
      `  Steps:                   ${r.numSteps}`,
      `  RMSE (KF estimate):      ${r.rmsePos.toFixed(3)} m`,
      `  RMSE (raw measurement):  ${r.rmseMeasPos.toFixed(3)} m`,
      `  Final cov trace:         ${r.finalCovTrace.toFixed(3)}`,
      `  KF beats raw sensor by:  ${(100 * (1 - r.rmsePos / r.rmseMeasPos)).toFixed(1)} %`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,truePos,trueVel,measPos,estPos,estVel'];
    for (let i = 0; i < r.estimates.length; i++) {
      const tp = r.trueTrajectory[i + 1];
      lines.push(csvRow([i, tp[0].toFixed(6), tp[1].toFixed(6), r.measurements[i][0].toFixed(6), r.estimates[i][0].toFixed(6), r.estimates[i][1].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 3. sliding-mode
// -----------------------------------------------------------------------------

const slidingModeSchema: ParamSchema = {
  kind: 'object',
  description: 'Sliding-mode control of an uncertain double integrator with bounded matched disturbance.',
  fields: {
    x0: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
    dt: {kind: 'number', min: 1e-6, default: 0.05},
    numSteps: {kind: 'number', integer: true, min: 1, default: 400},
    lambda: {kind: 'number', min: 0, default: 2},
    eta: {kind: 'number', min: 0, default: 3},
    boundary: {kind: 'number', min: 0, default: 0.05},
    uBound: {kind: 'number', min: 0, default: 5},
    disturbanceAmp: {kind: 'number', min: 0, default: 1},
    disturbanceType: {kind: 'string', enum: ['sin', 'square', 'random'], default: 'sin'},
    seed: {kind: 'number', integer: true, default: 1},
  },
  required: [],
};

registerModel<SlidingModeOpts, SlidingModeResult>({
  id: 'sliding-mode',
  description: 'Sliding-mode (robust) control of an uncertain plant with bounded disturbance.',
  schema: slidingModeSchema,
  run(p) {
    return runSlidingMode({...p, x0: numberPair(p.x0, [3, 0], 'x0')});
  },
  summarize(r, p) {
    return [
      'SLIDING-MODE CONTROL (robust)',
      '────────────────────────────────────',
      `  Disturbance:             type=${p.disturbanceType ?? 'sin'} amp=${p.disturbanceAmp ?? 1}`,
      `  Reaching tick:           ${r.reachingTick}    (s = 0 hit)`,
      `  Stayed near origin?      ${r.stayedNearOrigin ? 'YES' : 'no'}`,
      `  Final |x|+|v|:           ${r.finalDistanceFromOrigin.toFixed(3)}`,
      `  λ:                       ${p.lambda ?? 2}`,
      `  η:                       ${p.eta ?? 3}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,x,v,u'];
    for (let i = 0; i < r.controls.length; i++) {
      lines.push(csvRow([i, r.trajectory[i][0].toFixed(6), r.trajectory[i][1].toFixed(6), r.controls[i][0].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 4. mrac
// -----------------------------------------------------------------------------

const mracSchema: ParamSchema = {
  kind: 'object',
  description: 'Model-Reference Adaptive Control on a first-order plant with unknown a, b > 0.',
  fields: {
    a: {kind: 'number', default: 1},
    b: {kind: 'number', default: 2},
    am: {kind: 'number', max: -1e-9, default: -2},
    bm: {kind: 'number', default: 2},
    x0: {kind: 'number', default: 0},
    xm0: {kind: 'number', default: 0},
    gamma: {kind: 'number', min: 0, default: 5},
    dt: {kind: 'number', min: 1e-6, default: 0.01},
    numSteps: {kind: 'number', integer: true, min: 1, default: 4000},
    uBound: {kind: 'number', min: 0},
  },
  required: [],
};

registerModel<MRACOpts, MRACResult>({
  id: 'mrac',
  description: 'Model-Reference Adaptive Control with unknown plant gain (Lyapunov-based MIT rule).',
  schema: mracSchema,
  run(p) { return runMRAC(p); },
  summarize(r, p) {
    return [
      'MRAC (Model-Reference Adaptive Control)',
      '────────────────────────────────────',
      `  True plant:              a=${p.a ?? 1} b=${p.b ?? 2}`,
      `  Reference model:         a_m=${p.am ?? -2} b_m=${p.bm ?? 2}`,
      `  Adaptation gain γ:       ${p.gamma ?? 5}`,
      `  Final θ_x, θ_r:          [${r.finalTheta.map(x => x.toFixed(3)).join(', ')}]`,
      `  Ideal θ*_x, θ*_r:        [${r.idealTheta.map(x => x.toFixed(3)).join(', ')}]`,
      `  Steady-state RMS error:  ${r.rmsErrorSteadyState.toFixed(4)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,x,xm,r,theta_x,theta_r,error'];
    for (let i = 0; i < r.trackingError.length; i++) {
      lines.push(csvRow([i, r.trajectory[i + 1][0].toFixed(6), r.referenceTrajectory[i + 1].toFixed(6), r.rHistory[i].toFixed(6), r.thetaXHistory[i].toFixed(6), r.thetaRHistory[i].toFixed(6), r.trackingError[i].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 5. iterative-learning-control
// -----------------------------------------------------------------------------

const ilcSchema: ParamSchema = {
  kind: 'object',
  description: 'Iterative Learning Control on a repeated first-order tracking task.',
  fields: {
    trials: {kind: 'number', integer: true, min: 1, default: 30},
    horizon: {kind: 'number', integer: true, min: 2, default: 80},
    dt: {kind: 'number', min: 1e-6, default: 0.1},
    plantRate: {kind: 'number', min: 1e-12, default: 1.2},
    plantGain: {kind: 'number', min: 1e-12, default: 1},
    learningGain: {kind: 'number', min: 0, max: 2, default: 0.8},
    feedbackGain: {kind: 'number', min: 0, default: 0.8},
    controlMax: {kind: 'number', min: 1e-12, default: 5},
    referenceKind: {kind: 'string', enum: ['sine', 'step', 'ramp'], default: 'sine'},
    referenceAmplitude: {kind: 'number', min: 0, default: 1},
    initialOutput: {kind: 'number', default: 0},
  },
  required: [],
};

registerModel<IterativeLearningControlParams, IterativeLearningControlResult>({
  id: 'iterative-learning-control',
  description: 'Iterative Learning Control: repeated-trial feedforward adaptation over source/station/sink movables.',
  schema: ilcSchema,
  run(p) { return runIterativeLearningControl(p); },
  summarize(r, p) {
    return [
      'ITERATIVE LEARNING CONTROL (DES)',
      '--------------------------------',
      `  Trials:         ${r.trialSummaries.length}`,
      `  Reference:      ${p.referenceKind ?? 'sine'}  amplitude=${p.referenceAmplitude ?? 1}`,
      `  Initial RMS:    ${r.initialRmsError.toFixed(6)}`,
      `  Final RMS:      ${r.finalRmsError.toFixed(6)}`,
      `  Improvement:    ${(100 * (1 - r.improvementRatio)).toFixed(1)}% RMS reduction`,
      `  Stations:       ${r.topology.stations.join(' -> ')}`,
      `  Movables:       ${r.topology.movables.join(', ')}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['trial,rms_error,max_abs_error,max_abs_control,final_output,final_reference'];
    for (const row of r.trialSummaries) {
      lines.push(csvRow([row.trial, row.rmsError, row.maxAbsError, row.maxAbsControl, row.finalOutput, row.finalReference]));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(r, _p, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'iterative-learning-control');
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 960,
      height: 560,
      fps: 6,
      title: 'Iterative Learning Control',
      subtitle: 'Trial plans, plant responses, and learning updates flow as DES movables',
      background: '#f8fafc',
    });
    const yValues = [...r.referenceTrajectory, ...r.finalOutputTrajectory];
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    for (const row of r.trialSummaries) {
      rec.frame(row.trial, row.trial, () => buildILCFrame(r, row.trial, yMin, yMax));
    }
    const trialT = r.trialSummaries.map(row => row.trial);
    const timeT = r.referenceTrajectory.map((_v, i) => i);
    rec.setCharts([
      {
        x: 40, y: 360, w: 270, h: 150, title: 'RMS error by trial', yLabel: 'RMS',
        yMin: 0, yMax: Math.max(...r.trialSummaries.map(row => row.rmsError)) * 1.05,
        series: [{label: 'tracking error', color: '#dc2626', t: trialT, y: r.trialSummaries.map(row => row.rmsError)}],
      },
      {
        x: 345, y: 360, w: 270, h: 150, title: 'Final trial tracking', yLabel: 'y',
        yMin: yMin - 0.1 * Math.max(1, yMax - yMin), yMax: yMax + 0.1 * Math.max(1, yMax - yMin),
        series: [
          {label: 'reference', color: '#2563eb', t: timeT, y: r.referenceTrajectory},
          {label: 'output', color: '#059669', t: r.finalOutputTrajectory.map((_v, i) => i), y: r.finalOutputTrajectory},
        ],
      },
      {
        x: 650, y: 360, w: 270, h: 150, title: 'Learned feedforward', yLabel: 'u_ff',
        series: [{label: 'feedforward', color: '#7c3aed', t: r.finalFeedforwardSequence.map((_v, i) => i), y: r.finalFeedforwardSequence}],
      },
    ]);
    await rec.finish();
  },
});

function buildILCFrame(
  result: IterativeLearningControlResult,
  trialIndex: number,
  yMin: number,
  yMax: number,
): {shapes: Shape[]; caption: string} {
  const row = result.trialSummaries[trialIndex];
  const shapes: Shape[] = [];
  const initial = result.initialRmsError;
  const reduction = 100 * (1 - row.rmsError / Math.max(initial, 1e-12));
  const nodes = [
    {id: 'source', label: 'trial source', x: 92, y: 96, w: 128, h: 58, fill: '#dbeafe'},
    {id: 'controller', label: 'controller program', x: 270, y: 96, w: 148, h: 58, fill: '#ede9fe'},
    {id: 'plant', label: 'plant trial', x: 470, y: 96, w: 128, h: 58, fill: '#dcfce7'},
    {id: 'learner', label: 'learning update', x: 650, y: 96, w: 144, h: 58, fill: '#ffedd5'},
    {id: 'sink', label: 'result sink', x: 830, y: 96, w: 120, h: 58, fill: '#f1f5f9'},
  ];
  const edgePoints = [
    {from: nodes[0], to: nodes[1], label: 'ILCTrialPlanToken'},
    {from: nodes[1], to: nodes[2], label: 'ILCControlProgramToken'},
    {from: nodes[2], to: nodes[3], label: 'ILCTrialResultToken'},
    {from: nodes[2], to: nodes[4], label: 'ILCTrialResultToken'},
    {from: nodes[3], to: nodes[1], label: 'next ILCTrialPlanToken'},
  ];
  const activeEdge = edgePoints[trialIndex % edgePoints.length];

  shapes.push({kind: 'rect', x: 0, y: 0, w: 960, h: 560, fill: '#f8fafc'});
  shapes.push({kind: 'text', x: 40, y: 34, text: `Trial ${row.trial + 1} of ${result.trialSummaries.length}`, fontSize: 22, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: 40, y: 58, text: `RMS ${row.rmsError.toFixed(5)}  reduction ${reduction.toFixed(1)}%  max |u| ${row.maxAbsControl.toFixed(3)}`, fontSize: 13, fill: '#475569'});

  for (const edge of edgePoints) {
    const active = edge === activeEdge;
    const y = edge.from.id === 'learner' ? edge.from.y + edge.from.h + 22 : edge.from.y + edge.from.h / 2;
    const y2 = edge.from.id === 'learner' ? edge.to.y + edge.to.h + 22 : edge.to.y + edge.to.h / 2;
    const x1 = edge.from.id === 'learner' ? edge.from.x + edge.from.w / 2 : edge.from.x + edge.from.w;
    const x2 = edge.from.id === 'learner' ? edge.to.x + edge.to.w / 2 : edge.to.x;
    if (edge.from.id === 'learner') {
      shapes.push({
        kind: 'path',
        d: `M ${x1} ${y} C ${x1} 190, ${x2} 190, ${x2} ${y2}`,
        stroke: active ? '#7c3aed' : '#94a3b8',
        strokeWidth: active ? 4 : 2,
        fill: 'none',
        opacity: active ? 0.95 : 0.55,
      });
    } else {
      shapes.push({kind: 'line', x1, y1: y, x2, y2, stroke: active ? '#7c3aed' : '#94a3b8', strokeWidth: active ? 4 : 2, opacity: active ? 0.95 : 0.65});
    }
    shapes.push({kind: 'text', x: (x1 + x2) / 2, y: Math.min(y, y2) - 10, text: edge.label, fontSize: 9, anchor: 'middle', fill: active ? '#5b21b6' : '#64748b'});
  }

  for (const n of nodes) {
    shapes.push({kind: 'rect', x: n.x, y: n.y, w: n.w, h: n.h, fill: n.fill, stroke: '#334155', strokeWidth: 1.3, rx: 7});
    shapes.push({kind: 'text', x: n.x + n.w / 2, y: n.y + 28, text: n.label, fontSize: 12, anchor: 'middle', fill: '#0f172a', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: n.x + n.w / 2, y: n.y + 46, text: 'station', fontSize: 10, anchor: 'middle', fill: '#475569'});
  }

  const tokenX = activeEdge.from.id === 'learner'
    ? (activeEdge.from.x + activeEdge.from.w / 2 + activeEdge.to.x + activeEdge.to.w / 2) / 2
    : (activeEdge.from.x + activeEdge.from.w + activeEdge.to.x) / 2;
  const tokenY = activeEdge.from.id === 'learner'
    ? 190
    : activeEdge.from.y + activeEdge.from.h / 2;
  shapes.push({kind: 'circle', x: tokenX, y: tokenY, r: 10, fill: '#7c3aed', stroke: '#ffffff', strokeWidth: 2, title: activeEdge.label});
  shapes.push({kind: 'text', x: tokenX, y: tokenY + 4, text: 'm', fontSize: 10, fill: '#ffffff', anchor: 'middle', fontWeight: 'bold'});

  metricBar(shapes, 40, 230, 270, 18, 'current RMS error', row.rmsError, initial, '#dc2626');
  metricBar(shapes, 40, 275, 270, 18, 'max absolute error', row.maxAbsError, Math.max(...result.trialSummaries.map(x => x.maxAbsError)), '#f97316');
  metricBar(shapes, 40, 320, 270, 18, 'learning progress', trialIndex + 1, result.trialSummaries.length, '#059669');

  shapes.push({kind: 'rect', x: 370, y: 218, w: 530, h: 120, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 7});
  shapes.push({kind: 'text', x: 386, y: 242, text: 'Final trial output vs reference', fontSize: 13, fill: '#0f172a', fontWeight: 'bold'});
  drawMiniSeries(shapes, result.referenceTrajectory, 392, 260, 480, 58, yMin, yMax, '#2563eb');
  drawMiniSeries(shapes, result.finalOutputTrajectory, 392, 260, 480, 58, yMin, yMax, '#059669');
  shapes.push({kind: 'line', x1: 392, y1: 318, x2: 872, y2: 318, stroke: '#94a3b8', strokeWidth: 1});
  shapes.push({kind: 'text', x: 392, y: 335, text: 'reference', fontSize: 10, fill: '#2563eb'});
  shapes.push({kind: 'text', x: 472, y: 335, text: 'output', fontSize: 10, fill: '#059669'});

  return {
    shapes,
    caption: `trial ${row.trial + 1}: RMS ${row.rmsError.toFixed(5)} (${reduction.toFixed(1)}% reduction from trial 1)`,
  };
}

function metricBar(
  shapes: Shape[],
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  value: number,
  maxValue: number,
  fill: string,
): void {
  const frac = Math.max(0, Math.min(1, value / Math.max(maxValue, 1e-12)));
  shapes.push({kind: 'text', x, y: y - 8, text: `${label}: ${formatILCNumber(value)}`, fontSize: 12, fill: '#334155', fontWeight: 'bold'});
  shapes.push({kind: 'rect', x, y, w, h, fill: '#e2e8f0', stroke: '#cbd5e1', rx: 4});
  shapes.push({kind: 'rect', x, y, w: w * frac, h, fill, rx: 4, opacity: 0.88});
}

function drawMiniSeries(
  shapes: Shape[],
  values: readonly number[],
  x: number,
  y: number,
  w: number,
  h: number,
  yMin: number,
  yMax: number,
  stroke: string,
): void {
  const denom = Math.max(1, values.length - 1);
  const yRange = Math.max(1e-12, yMax - yMin);
  const d = values.map((v, i) => {
    const px = x + w * i / denom;
    const py = y + h - h * (v - yMin) / yRange;
    return `${i === 0 ? 'M' : 'L'} ${px.toFixed(2)} ${py.toFixed(2)}`;
  }).join(' ');
  shapes.push({kind: 'path', d, stroke, strokeWidth: 2.4, fill: 'none'});
}

function formatILCNumber(x: number): string {
  if (Math.abs(x) < 0.001) return x.toExponential(2);
  return x.toFixed(4);
}

// -----------------------------------------------------------------------------
// 6. feedback-linearization
// -----------------------------------------------------------------------------

const feedbackLinSchema: ParamSchema = {
  kind: 'object',
  description: 'Feedback-linearization (computed-torque) tracking control of a pendulum.',
  fields: {
    m: {kind: 'number', min: 0, default: 1},
    l: {kind: 'number', min: 0, default: 1},
    g: {kind: 'number', min: 0, default: 9.81},
    c: {kind: 'number', min: 0, default: 0.1},
    theta0: {kind: 'number', default: 3.141592653589793},
    thetaDot0: {kind: 'number', default: 0},
    kp: {kind: 'number', min: 0, default: 25},
    kv: {kind: 'number', min: 0, default: 10},
    uBound: {kind: 'number', min: 0, default: 100},
    dt: {kind: 'number', min: 1e-6, default: 0.01},
    numSteps: {kind: 'number', integer: true, min: 1, default: 1000},
  },
  required: [],
};

interface FlatFeedbackLinParams {
  m?: number; l?: number; g?: number; c?: number;
  theta0?: number; thetaDot0?: number;
  kp?: number; kv?: number; uBound?: number;
  dt?: number; numSteps?: number;
}

registerModel<FlatFeedbackLinParams, FeedbackLinearizationResult>({
  id: 'feedback-linearization',
  description: 'Feedback linearization (nonlinear control): pendulum tracking via computed torque.',
  schema: feedbackLinSchema,
  run(p) {
    const opts: FeedbackLinearizationOpts = {
      params: {m: p.m ?? 1, l: p.l ?? 1, g: p.g ?? 9.81, c: p.c ?? 0.1},
      theta0: p.theta0,
      thetaDot0: p.thetaDot0,
      kp: p.kp,
      kv: p.kv,
      uBound: p.uBound,
      dt: p.dt,
      numSteps: p.numSteps,
    };
    return runFeedbackLinearization(opts);
  },
  summarize(r, p) {
    return [
      'FEEDBACK LINEARIZATION (pendulum)',
      '────────────────────────────────────',
      `  Mass / length / g / damping: ${p.m ?? 1} / ${p.l ?? 1} / ${p.g ?? 9.81} / ${p.c ?? 0.1}`,
      `  PD gains kp / kv:        ${p.kp ?? 25} / ${p.kv ?? 10}`,
      `  Steps:                   ${r.numSteps}`,
      `  Steady-state RMS error:  ${r.rmsErrorSteadyState.toExponential(2)} rad`,
      `  Final angle:             ${r.trajectory[r.trajectory.length - 1][0].toFixed(4)} rad`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,theta,thetaDot,thetaRef,torque'];
    for (let i = 0; i < r.controls.length; i++) {
      lines.push(csvRow([i, r.trajectory[i + 1][0].toFixed(6), r.trajectory[i + 1][1].toFixed(6), r.thetaDHistory[i].toFixed(6), r.controls[i][0].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 7. mpc-double-integrator
// -----------------------------------------------------------------------------

const mpcSchema: ParamSchema = {
  kind: 'object',
  description: 'Constrained MPC on a double integrator: receding-horizon QP via projected gradient.',
  fields: {
    x0: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
    uMax: {kind: 'number', min: 0, default: 1},
    N: {kind: 'number', integer: true, min: 1, default: 15},
    Q: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 2, maxLength: 2},
    Qf: {kind: 'array', items: {kind: 'number', min: 0}, minLength: 2, maxLength: 2},
    R: {kind: 'number', min: 1e-9, default: 0.1},
    dt: {kind: 'number', min: 1e-6, default: 0.1},
    numSteps: {kind: 'number', integer: true, min: 1, default: 100},
  },
  required: [],
};

registerModel<MPCDoubleIntOpts, MPCDoubleIntResult>({
  id: 'mpc-double-integrator',
  description: 'MPC: constrained receding-horizon QP on a double integrator.',
  schema: mpcSchema,
  run(p) {
    return runMPCDoubleIntegrator({
      ...p,
      x0: numberPair(p.x0, [3, 0], 'x0'),
      Q: optionalNumberPair(p.Q, 'Q'),
      Qf: optionalNumberPair(p.Qf, 'Qf'),
    });
  },
  summarize(r, p) {
    const dt = p.dt ?? 0.1;
    return [
      'MPC — DOUBLE INTEGRATOR (constrained QP)',
      '──────────────────────────────────────────',
      `  Initial state:           [${(p.x0 ?? [3, 0]).join(', ')}]`,
      `  |u| bound:               ${p.uMax ?? 1}`,
      `  Horizon N:               ${p.N ?? 15}`,
      `  Sample period dt:        ${dt}`,
      `  Arrival tick:            ${r.arrivalTick}    (~ ${(r.arrivalTick * dt).toFixed(2)} s)`,
      `  Max realised |u|:        ${r.maxAbsU.toFixed(4)}`,
      `  Final state:             [${r.trajectory[r.trajectory.length - 1].map(x => x.toFixed(3)).join(', ')}]`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,x,v,u'];
    for (let i = 0; i < r.controls.length; i++) {
      lines.push(csvRow([i, r.trajectory[i][0].toFixed(6), r.trajectory[i][1].toFixed(6), r.controls[i][0].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});
