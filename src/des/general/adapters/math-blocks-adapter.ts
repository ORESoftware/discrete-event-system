// RUST MIGRATION: Target module `src/des/general/adapters/math_blocks_adapter.rs`.
// RUST MIGRATION: Convert math-block adapter registrations for ODE/PDE/equation demos into adapter structs/functions around `DESModelSpec`.
// RUST MIGRATION: Represent numeric maps, state vectors, traces, and chart frames as `serde` config/result structs; output paths become `PathBuf`.
// RUST MIGRATION: Make expression, range, and numeric validation explicit with `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/math-blocks-adapter.rs
//   (module des::general::adapters::math_blocks_adapter)
// 1:1 file move. Registers math-ODE-blocks / heat1d-blocks / math-equation JSON
// adapters (block-diagram numerics), all with animations.
//
// Declarations → Rust:
//   const numericMapSchema/odeStateSchema/odeSchema/heatSchema/equationSchema:
//         ParamSchema                          -> serde + validator metadata
//   fn palette / finiteRange / heatColor       -> plain `fn` helpers
//   registerModel(...) x3                       -> one struct + impl ModelAdapter trait each
//
// Conversion notes (file-specific):
//   - GotChA: `state`/`derivatives` keyed by variable name and `constants`/`initial`
//     are open `{kind:'object', fields:{}}` numeric maps -> HashMap<String, f64>;
//     state derivatives are expression STRINGS evaluated at runtime (the math
//     expression engine) -> port the expr evaluator, not a literal closure.
//   - `format: 'json'|'latex'|'xml'`, `kind: 'ode'|'heat1d'`, `method:
//     'euler'|'trapezoid'` literal unions -> enums; math-equation result is a tagged
//     union over ode/heat1d -> enum matched in summarize/writeCsv/animate.
//   - `r.ode!`/`r.heat1d?` non-null/optional chaining -> Option match.
//   - Shapes pushed into `Shape[]` (animation/types) -> Vec<Shape>; heatColor builds
//     `rgb(r,g,b)` strings -> a colour helper; animations derive from result traces.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {registerModel} from '../des-registry';
import {ParamSchema} from '../des-spec';
import {csvRow, framesPath, validationLine, withLogger, writeCsvLines} from './adapter-utils';
import {
  MathEquationInputParams,
  MathEquationResult,
  runMathEquationProblem,
} from '../math-equation-input';
import {
  Heat1DBlockParams,
  Heat1DBlockResult,
  ODEBlockSystemParams,
  ODEBlockSystemResult,
  runHeat1DBlockGrid,
  runODEBlockSystem,
} from '../math-blocks';

const numericMapSchema: ParamSchema = {kind: 'object', fields: {}, required: []};

const odeStateSchema: ParamSchema = {
  kind: 'object',
  fields: {
    name: {kind: 'string'},
    initial: {kind: 'number'},
    derivative: {kind: 'string'},
  },
  required: ['name', 'initial', 'derivative'],
};

const odeSchema: ParamSchema = {
  kind: 'object',
  fields: {
    states: {kind: 'array', items: odeStateSchema, minLength: 1, maxLength: 100},
    t0: {kind: 'number', default: 0},
    t1: {kind: 'number'},
    dt: {kind: 'number', min: 1e-12},
    method: {kind: 'string', enum: ['euler', 'trapezoid'], default: 'euler'},
    constants: numericMapSchema,
  },
  required: ['states', 't1', 'dt'],
};

const heatSchema: ParamSchema = {
  kind: 'object',
  fields: {
    cells: {kind: 'number', integer: true, min: 3, max: 1000},
    length: {kind: 'number', min: 1e-12},
    alpha: {kind: 'number', min: 0},
    t0: {kind: 'number', default: 0},
    t1: {kind: 'number'},
    dt: {kind: 'number', min: 1e-12},
    initialExpression: {kind: 'string', default: 'sin(pi*x/length)'},
    initialValues: {kind: 'array', items: {kind: 'number'}, minLength: 3},
    leftBoundary: {kind: 'number'},
    rightBoundary: {kind: 'number'},
    constants: numericMapSchema,
  },
  required: ['cells', 'length', 'alpha', 't1', 'dt'],
};

const equationSchema: ParamSchema = {
  kind: 'object',
  fields: {
    format: {kind: 'string', enum: ['json', 'latex', 'xml']},
    kind: {kind: 'string', enum: ['ode', 'heat1d']},
    equation: {kind: 'string'},
    ode: {kind: 'object', fields: {}, required: []},
    heat1d: {kind: 'object', fields: {}, required: []},
    states: {kind: 'array', items: odeStateSchema, minLength: 1},
    constants: numericMapSchema,
    initial: numericMapSchema,
    t0: {kind: 'number', default: 0},
    t1: {kind: 'number', default: 1},
    dt: {kind: 'number', min: 1e-12},
    method: {kind: 'string', enum: ['euler', 'trapezoid'], default: 'euler'},
    cells: {kind: 'number', integer: true, min: 3, max: 1000},
    length: {kind: 'number', min: 1e-12},
    alpha: {kind: 'number', min: 0},
    initialExpression: {kind: 'string'},
    initialValues: {kind: 'array', items: {kind: 'number'}, minLength: 3},
    leftBoundary: {kind: 'number'},
    rightBoundary: {kind: 'number'},
  },
  required: ['format'],
};

function palette(i: number): string {
  const colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c', '#0891b2', '#be123c'];
  return colors[i % colors.length];
}

function finiteRange(values: readonly number[]): {min: number; max: number} {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return {min: 0, max: 1};
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return min === max ? {min: min - 1, max: max + 1} : {min, max};
}

function heatColor(v: number, min: number, max: number): string {
  const q = Math.max(0, Math.min(1, (v - min) / Math.max(1e-12, max - min)));
  const mid = 0.5;
  if (q <= mid) {
    const a = q / mid;
    const r = Math.round(37 + a * (248 - 37));
    const g = Math.round(99 + a * (250 - 99));
    const b = Math.round(235 + a * (252 - 235));
    return `rgb(${r},${g},${b})`;
  }
  const a = (q - mid) / mid;
  const r = Math.round(248 + a * (220 - 248));
  const g = Math.round(250 + a * (38 - 250));
  const b = Math.round(252 + a * (38 - 252));
  return `rgb(${r},${g},${b})`;
}

registerModel<ODEBlockSystemParams, ODEBlockSystemResult>({
  id: 'math-ode-blocks',
  description: 'ODE system assembled from stationary math blocks, integrators, expression RHS blocks, sources, and sinks.',
  schema: odeSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'math-ode-run-start', level: 'info', states: params.states.map(s => s.name), dt: params.dt});
      const result = runODEBlockSystem(params, logger);
      logger?.log({kind: 'math-ode-run-finish', level: 'info', finalState: result.finalState, steps: result.steps});
      return result;
    });
  },
  summarize(r) {
    const final = Object.entries(r.finalState).map(([k, v]) => `${k}=${v.toPrecision(6)}`).join(', ');
    return [
      'MATH ODE BLOCKS',
      '------------------------',
      `  states=${r.params.states.map(s => s.name).join(', ')} steps=${r.steps} dt=${r.params.dt}`,
      `  final state: ${final}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const names = r.params.states.map(s => s.name);
    const lines = [csvRow(['tick', 'time', ...names, ...names.map(n => `d_${n}`)])];
    for (const row of r.trace) {
      lines.push(csvRow([
        row.tick,
        row.time,
        ...names.map(n => row.state[n]),
        ...names.map(n => row.derivatives[n]),
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  async animate(r, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'math-ode-blocks');
    const names = params.states.map(s => s.name);
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 900,
      height: 540,
      fps: 12,
      title: 'ODE math blocks',
      subtitle: 'Stationary integrator and RHS expression blocks exchange moving scalar signals',
      visualBlocks: r.visualBlocks,
    });
    const allValues = r.trace.flatMap(row => names.map(n => row.state[n]));
    const range = finiteRange(allValues);
    for (const row of r.trace) {
      rec.frame(row.time, row.tick, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 900, h: 540, fill: '#f8fafc'},
          {kind: 'text', x: 450, y: 34, text: `t=${row.time.toFixed(3)}`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
        ];
        names.forEach((name, i) => {
          const y = 86 + i * 76;
          const c = palette(i);
          shapes.push({kind: 'rect', x: 92, y, w: 190, h: 48, rx: 6, fill: '#dbeafe', stroke: c, strokeWidth: 2, title: `Integrator for ${name}`});
          shapes.push({kind: 'text', x: 187, y: y + 20, text: `integrator ${name}`, fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
          shapes.push({kind: 'text', x: 187, y: y + 38, text: row.state[name].toPrecision(5), fontSize: 12, anchor: 'middle', fill: '#334155'});
          shapes.push({kind: 'line', x1: 282, y1: y + 24, x2: 535, y2: y + 24, stroke: c, strokeWidth: 3, opacity: 0.85});
          shapes.push({kind: 'rect', x: 535, y, w: 270, h: 48, rx: 6, fill: '#fef3c7', stroke: '#92400e', strokeWidth: 2, title: params.states[i].derivative});
          shapes.push({kind: 'text', x: 670, y: y + 20, text: `d${name}/dt = ${params.states[i].derivative}`, fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
          shapes.push({kind: 'text', x: 670, y: y + 38, text: row.derivatives[name].toPrecision(5), fontSize: 12, anchor: 'middle', fill: '#334155'});
          shapes.push({kind: 'line', x1: 535, y1: y + 44, x2: 282, y2: y + 44, stroke: '#64748b', strokeWidth: 2, dasharray: '5,6', opacity: 0.7});
          const barW = 180 * (row.state[name] - range.min) / Math.max(1e-12, range.max - range.min);
          shapes.push({kind: 'rect', x: 92, y: y + 56, w: 180, h: 8, fill: '#e2e8f0'});
          shapes.push({kind: 'rect', x: 92, y: y + 56, w: Math.max(0, Math.min(180, barW)), h: 8, fill: c});
        });
        return {shapes, caption: `final target time ${params.t1}; current state ${names.map(n => `${n}=${row.state[n].toPrecision(4)}`).join(', ')}`};
      });
    }
    rec.setCharts([
      {
        x: 70,
        y: 380,
        w: 760,
        h: 120,
        title: 'State trajectories',
        series: names.map((name, i) => ({label: name, color: palette(i), t: r.trace.map(row => row.time), y: r.trace.map(row => row.state[name])})),
      },
    ]);
    await rec.finish();
  },
  examples: [{
    name: 'exponential decay',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'math-ode-blocks',
      parameters: {
        states: [{name: 'y', initial: 1, derivative: '-k*y'}],
        constants: {k: 1},
        t0: 0,
        t1: 1,
        dt: 0.01,
        method: 'euler',
      },
      runtime: {animate: true},
    },
  }],
});

registerModel<Heat1DBlockParams, Heat1DBlockResult>({
  id: 'math-heat1d-blocks',
  description: '1D heat equation PDE as a stationary grid of cell integrators and Laplacian blocks.',
  schema: heatSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      const normalized: Heat1DBlockParams = {
        ...params,
        initialValues: params.initialValues && params.initialValues.length > 0 ? params.initialValues : undefined,
      };
      logger?.log({kind: 'math-heat1d-run-start', level: 'info', cells: normalized.cells, dt: normalized.dt});
      const result = runHeat1DBlockGrid(normalized, logger);
      logger?.log({kind: 'math-heat1d-run-finish', level: 'info', steps: result.steps, finalMax: Math.max(...result.finalValues)});
      return result;
    });
  },
  summarize(r) {
    const last = r.trace[r.trace.length - 1];
    return [
      'MATH HEAT1D BLOCKS',
      '------------------------',
      `  cells=${r.params.cells} steps=${r.steps} dt=${r.params.dt} dx=${r.dx.toPrecision(5)} cfl=${r.cfl.toPrecision(5)}`,
      `  final min=${last.min.toPrecision(6)} max=${last.max.toPrecision(6)} mean=${last.mean.toPrecision(6)}`,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = [csvRow(['tick', 'time', 'min', 'max', 'mean', ...r.x.map((_, i) => `cell_${i}`)])];
    for (const row of r.trace) lines.push(csvRow([row.tick, row.time, row.min, row.max, row.mean, ...row.values]));
    writeCsvLines(csvPath, lines);
  },
  async animate(r, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'math-heat1d-blocks');
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 900,
      height: 520,
      fps: 14,
      title: 'Heat equation block grid',
      subtitle: 'Cell integrators exchange Laplacian-derived heat-flow signals',
      visualBlocks: r.visualBlocks,
    });
    const range = finiteRange(r.trace.flatMap(row => row.values));
    const cellW = 760 / params.cells;
    for (const row of r.trace) {
      rec.frame(row.time, row.tick, () => {
        const shapes: Shape[] = [
          {kind: 'rect', x: 0, y: 0, w: 900, h: 520, fill: '#f8fafc'},
          {kind: 'text', x: 450, y: 34, text: `t=${row.time.toFixed(3)}  max=${row.max.toPrecision(4)}`, fontSize: 19, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
        ];
        for (let i = 0; i < row.values.length; i++) {
          const x = 70 + i * cellW;
          const color = heatColor(row.values[i], range.min, range.max);
          shapes.push({kind: 'rect', x, y: 90, w: Math.max(1, cellW - 1), h: 120, fill: color, stroke: '#ffffff', strokeWidth: 1, title: `cell ${i}: ${row.values[i].toPrecision(5)}`});
          if (i === 0 || i === row.values.length - 1 || i % Math.max(1, Math.floor(row.values.length / 6)) === 0) {
            shapes.push({kind: 'text', x: x + cellW / 2, y: 230, text: String(i), fontSize: 10, anchor: 'middle', fill: '#475569'});
          }
        }
        const midY = 286;
        for (let i = 1; i < row.values.length - 1; i++) {
          const x = 70 + i * cellW + cellW / 2;
          shapes.push({kind: 'circle', x, y: midY, r: 4, fill: '#f97316', opacity: 0.6, title: `laplacian block ${i}`});
          if (i > 1) shapes.push({kind: 'line', x1: x - cellW, y1: midY, x2: x, y2: midY, stroke: '#94a3b8', strokeWidth: 1, opacity: 0.5});
        }
        shapes.push({kind: 'text', x: 450, y: 260, text: `stationary grid: ${params.cells} cell blocks, ${params.cells - 2} Laplacian blocks`, fontSize: 13, anchor: 'middle', fill: '#334155'});
        return {shapes, caption: `min=${row.min.toPrecision(4)} mean=${row.mean.toPrecision(4)} max=${row.max.toPrecision(4)} cfl=${r.cfl.toPrecision(3)}`};
      });
    }
    rec.setCharts([
      {x: 70, y: 330, w: 760, h: 130, title: 'Min, mean, and max temperature', series: [
        {label: 'min', color: '#2563eb', t: r.trace.map(row => row.time), y: r.trace.map(row => row.min)},
        {label: 'mean', color: '#059669', t: r.trace.map(row => row.time), y: r.trace.map(row => row.mean)},
        {label: 'max', color: '#dc2626', t: r.trace.map(row => row.time), y: r.trace.map(row => row.max)},
      ]},
    ]);
    await rec.finish();
  },
  examples: [{
    name: 'cooling sine pulse',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'math-heat1d-blocks',
      parameters: {
        cells: 31,
        length: 1,
        alpha: 0.02,
        t0: 0,
        t1: 0.5,
        dt: 0.005,
        initialExpression: 'sin(pi*x/length)',
        leftBoundary: 0,
        rightBoundary: 0,
      },
      runtime: {animate: true},
    },
  }],
});

registerModel<MathEquationInputParams, MathEquationResult>({
  id: 'math-equation',
  description: 'Parse a math equation supplied as LaTeX, XML, or structured JSON, generate a stationary/moving block network, and solve it numerically.',
  schema: equationSchema,
  async run(params, runtime) {
    return withLogger(runtime, logger => {
      logger?.log({kind: 'math-equation-run-start', level: 'info', format: params.format, problemKind: params.kind});
      const result = runMathEquationProblem(params, logger);
      logger?.log({kind: 'math-equation-run-finish', level: 'info', problemKind: result.kind, nodes: result.network.nodes.length, edges: result.network.edges.length});
      return result;
    });
  },
  summarize(r) {
    const modelLine = r.kind === 'ode' && r.ode
      ? `  ODE final state: ${Object.entries(r.ode.finalState).map(([k, v]) => `${k}=${v.toPrecision(6)}`).join(', ')}`
      : `  heat final max: ${r.heat1d?.trace[r.heat1d.trace.length - 1].max.toPrecision(6) ?? 'n/a'}`;
    return [
      'MATH EQUATION INPUT',
      '------------------------',
      `  format=${r.inputFormat} kind=${r.kind}`,
      `  generated network: nodes=${r.network.nodes.length} edges=${r.network.edges.length}`,
      modelLine,
      `  validation: ${validationLine(r.validation)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    if (r.kind === 'ode' && r.ode) {
      const names = r.ode.params.states.map(s => s.name);
      const lines = [csvRow(['tick', 'time', ...names, ...names.map(n => `d_${n}`)])];
      for (const row of r.ode.trace) lines.push(csvRow([row.tick, row.time, ...names.map(n => row.state[n]), ...names.map(n => row.derivatives[n])]));
      writeCsvLines(csvPath, lines);
      return;
    }
    if (r.heat1d) {
      const lines = [csvRow(['tick', 'time', 'min', 'max', 'mean', ...r.heat1d.x.map((_, i) => `cell_${i}`)])];
      for (const row of r.heat1d.trace) lines.push(csvRow([row.tick, row.time, row.min, row.max, row.mean, ...row.values]));
      writeCsvLines(csvPath, lines);
    }
  },
  async animate(r, params, runtime) {
    const {htmlPath, frames} = framesPath(runtime, 'math-equation');
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 900,
      height: 540,
      fps: 12,
      title: 'Math equation block network',
      subtitle: `${r.inputFormat} input -> stationary blocks + moving MathSignal edges`,
      visualBlocks: r.ode?.visualBlocks ?? r.heat1d?.visualBlocks ?? [],
    });
    if (r.kind === 'ode' && r.ode) {
      const names = r.ode.params.states.map(s => s.name);
      for (const row of r.ode.trace) {
        rec.frame(row.time, row.tick, () => {
          const shapes: Shape[] = [
            {kind: 'rect', x: 0, y: 0, w: 900, h: 540, fill: '#f8fafc'},
            {kind: 'text', x: 450, y: 34, text: `ODE block network at t=${row.time.toFixed(3)}`, fontSize: 19, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
          ];
          names.forEach((name, i) => {
            const y = 90 + i * 82;
            const color = palette(i);
            shapes.push({kind: 'rect', x: 90, y, w: 180, h: 50, rx: 6, fill: '#dbeafe', stroke: color, strokeWidth: 2});
            shapes.push({kind: 'text', x: 180, y: y + 22, text: `source/state ${name}`, fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
            shapes.push({kind: 'text', x: 180, y: y + 40, text: row.state[name].toPrecision(5), fontSize: 12, anchor: 'middle', fill: '#334155'});
            shapes.push({kind: 'line', x1: 270, y1: y + 25, x2: 570, y2: y + 25, stroke: color, strokeWidth: 3});
            shapes.push({kind: 'rect', x: 570, y, w: 240, h: 50, rx: 6, fill: '#fef3c7', stroke: '#92400e', strokeWidth: 2});
            shapes.push({kind: 'text', x: 690, y: y + 22, text: `rhs ${name}`, fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
            shapes.push({kind: 'text', x: 690, y: y + 40, text: row.derivatives[name].toPrecision(5), fontSize: 12, anchor: 'middle', fill: '#334155'});
            shapes.push({kind: 'line', x1: 570, y1: y + 47, x2: 270, y2: y + 47, stroke: '#64748b', strokeWidth: 2, dasharray: '5,6'});
          });
          return {shapes, caption: `${r.network.nodes.length} stationary nodes and ${r.network.edges.length} moving-signal edges generated from ${r.inputFormat}`};
        });
      }
      rec.setCharts([{x: 70, y: 380, w: 760, h: 120, title: 'State trajectories', series: names.map((name, i) => ({label: name, color: palette(i), t: r.ode!.trace.map(row => row.time), y: r.ode!.trace.map(row => row.state[name])}))}]);
    } else if (r.heat1d) {
      const heat = r.heat1d;
      const range = finiteRange(heat.trace.flatMap(row => row.values));
      const cellW = 760 / heat.params.cells;
      for (const row of heat.trace) {
        rec.frame(row.time, row.tick, () => {
          const shapes: Shape[] = [
            {kind: 'rect', x: 0, y: 0, w: 900, h: 540, fill: '#f8fafc'},
            {kind: 'text', x: 450, y: 34, text: `Heat PDE block grid at t=${row.time.toFixed(3)}`, fontSize: 19, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
          ];
          for (let i = 0; i < row.values.length; i++) {
            const x = 70 + i * cellW;
            shapes.push({kind: 'rect', x, y: 90, w: Math.max(1, cellW - 1), h: 118, fill: heatColor(row.values[i], range.min, range.max), stroke: '#ffffff', strokeWidth: 1});
          }
          shapes.push({kind: 'text', x: 450, y: 245, text: `${r.network.nodes.length} stationary nodes, ${r.network.edges.length} MathSignal edges`, fontSize: 13, anchor: 'middle', fill: '#334155'});
          return {shapes, caption: `min=${row.min.toPrecision(4)} mean=${row.mean.toPrecision(4)} max=${row.max.toPrecision(4)}`};
        });
      }
      rec.setCharts([{x: 70, y: 330, w: 760, h: 130, title: 'Heat summary', series: [
        {label: 'min', color: '#2563eb', t: heat.trace.map(row => row.time), y: heat.trace.map(row => row.min)},
        {label: 'mean', color: '#059669', t: heat.trace.map(row => row.time), y: heat.trace.map(row => row.mean)},
        {label: 'max', color: '#dc2626', t: heat.trace.map(row => row.time), y: heat.trace.map(row => row.max)},
      ]}]);
    }
    await rec.finish();
  },
  examples: [{
    name: 'latex ODE decay',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'math-equation',
      parameters: {
        format: 'latex',
        kind: 'ode',
        equation: '\\frac{dy}{dt} = -k y; y(0)=1',
        constants: {k: 1},
        t1: 1,
        dt: 0.01,
      },
      runtime: {animate: true},
    },
  }],
});
