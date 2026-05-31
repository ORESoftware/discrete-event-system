// RUST MIGRATION: Target module `src/des/general/adapters/signal_transforms_adapter.rs`.
// RUST MIGRATION: Convert z/Laplace/Fourier transform adapters, zod validation, and animation helpers into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Map transform configs, complex points, contributions, traces, and frames to `serde` config/result structs; paths become `PathBuf`.
// RUST MIGRATION: Replace zod/schema failures and expression/sample validation with `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// JSON adapters for signal transform station graphs.
// =============================================================================

import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {z} from 'zod';
import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {csvRow, framesPath, validationLine, writeCsvLines} from './adapter-utils';
import {
  ComplexValue,
  FourierTransformParams,
  LaplaceTransformParams,
  TransformContributionRecord,
  TransformOutputPoint,
  TransformRunResult,
  ZTransformParams,
  formatComplex,
  runFourierTransform,
  runLaplaceTransform,
  runZTransform,
} from '../signal-transforms';

const numberVectorSchema: ParamSchema = {kind: 'array', items: {kind: 'number'}, minLength: 1};
const numericMapSchema: ParamSchema = {kind: 'object', fields: {}, required: []};
const numberVectorZodSchema = z.array(z.number().finite()).min(1);
const numericMapZodSchema = z.record(z.number().finite());

const complexPointSchema: ParamSchema = {
  kind: 'object',
  fields: {
    label: {kind: 'string'},
    re: {kind: 'number'},
    im: {kind: 'number', default: 0},
  },
  required: ['re'],
};

const complexPointArraySchema: ParamSchema = {
  kind: 'array',
  items: complexPointSchema,
  minLength: 1,
};
const complexPointZodSchema = z.object({
  label: z.string().optional(),
  re: z.number().finite(),
  im: z.number().finite().default(0),
}).strict();
const complexPointArrayZodSchema = z.array(complexPointZodSchema).min(1);

const zTransformSchema: ParamSchema = {
  kind: 'object',
  fields: {
    sequence: numberVectorSchema,
    expression: {kind: 'string'},
    constants: numericMapSchema,
    terms: {kind: 'number', integer: true, min: 1, max: 1000000, default: 8},
    startIndex: {kind: 'number', integer: true, default: 0},
    zValues: complexPointArraySchema,
    tolerance: {kind: 'number', min: 0, default: 1e-9},
  },
  required: ['zValues'],
};
const zTransformZodSchema = z.object({
  sequence: numberVectorZodSchema.optional(),
  expression: z.string().min(1).optional(),
  constants: numericMapZodSchema.optional(),
  terms: z.number().int().min(1).max(1000000).default(8),
  startIndex: z.number().int().default(0),
  zValues: complexPointArrayZodSchema,
  tolerance: z.number().finite().min(0).default(1e-9),
}).strict().refine(
  params => (params.sequence !== undefined && params.sequence.length > 0) || params.expression !== undefined,
  {path: ['sequence'], message: 'expected either a non-empty sequence or an expression'},
);

const continuousTransformFields: Record<string, ParamSchema> = {
  samples: numberVectorSchema,
  expression: {kind: 'string'},
  constants: numericMapSchema,
  t0: {kind: 'number', default: 0},
  t1: {kind: 'number', default: 1},
  dt: {kind: 'number', min: 1e-12, default: 0.01},
  quadrature: {kind: 'string', enum: ['rectangular', 'trapezoid'], default: 'trapezoid'},
  tolerance: {kind: 'number', min: 0, default: 1e-9},
};
const continuousTransformZodFields = {
  samples: numberVectorZodSchema.optional(),
  expression: z.string().min(1).optional(),
  constants: numericMapZodSchema.optional(),
  t0: z.number().finite().default(0),
  t1: z.number().finite().default(1),
  dt: z.number().finite().positive().default(0.01),
  quadrature: z.enum(['rectangular', 'trapezoid']).default('trapezoid'),
  tolerance: z.number().finite().min(0).default(1e-9),
};

function hasSamplesOrExpression(params: {samples?: readonly number[]; expression?: string}): boolean {
  return (params.samples !== undefined && params.samples.length > 0) || params.expression !== undefined;
}

const laplaceTransformSchema: ParamSchema = {
  kind: 'object',
  fields: {
    ...continuousTransformFields,
    sValues: complexPointArraySchema,
  },
  required: ['sValues'],
};
const laplaceTransformZodSchema = z.object({
  ...continuousTransformZodFields,
  sValues: complexPointArrayZodSchema,
}).strict().refine(hasSamplesOrExpression, {
  path: ['samples'],
  message: 'expected either non-empty samples or an expression',
});

const fourierTransformSchema: ParamSchema = {
  kind: 'object',
  fields: {
    ...continuousTransformFields,
    omegaValues: numberVectorSchema,
  },
  required: ['omegaValues'],
};
const fourierTransformZodSchema = z.object({
  ...continuousTransformZodFields,
  omegaValues: numberVectorZodSchema,
}).strict().refine(hasSamplesOrExpression, {
  path: ['samples'],
  message: 'expected either non-empty samples or an expression',
});

function summarizeTransform(title: string, result: TransformRunResult): string {
  const lines = [
    title,
    '-'.repeat(title.length),
    `  convention: ${result.convention}`,
    `  samples:    ${result.samples.length}`,
    `  points:     ${result.outputs.length}`,
    `  entities:   sources=${result.entityFramework.sources.length} stations=${result.entityFramework.stations.length} sinks=${result.entityFramework.sinks.length}`,
    `  movables:   ${result.entityFramework.movableEntities.join(', ')}`,
    `  validation: ${validationLine(result.validation)}`,
  ];
  for (const output of result.outputs.slice(0, 6)) {
    lines.push(`  ${output.label}: ${formatComplex(output.value)}  |.|=${output.magnitude.toPrecision(6)}`);
  }
  if (result.outputs.length > 6) lines.push(`  ... ${result.outputs.length - 6} more point(s)`);
  return lines.join('\n');
}

function writeTransformCsv(result: TransformRunResult, csvPath: string): void {
  const lines = [csvRow([
    'point_index',
    'label',
    'point_re',
    'point_im',
    'value_re',
    'value_im',
    'magnitude',
    'phase',
    'direct_reference_re',
    'direct_reference_im',
    'absolute_error',
    'samples_used',
  ])];
  for (const output of result.outputs) {
    lines.push(csvRow([
      output.pointIndex,
      output.label,
      output.point.re,
      output.point.im,
      output.value.re,
      output.value.im,
      output.magnitude,
      output.phase,
      output.directReference.re,
      output.directReference.im,
      output.absoluteError,
      output.samplesUsed,
    ]));
  }
  writeCsvLines(csvPath, lines);
}

function color(i: number): string {
  const colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c', '#0891b2', '#be123c', '#4b5563'];
  return colors[i % colors.length];
}

function finiteRange(values: readonly number[]): {min: number; max: number} {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return {min: -1, max: 1};
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return min === max ? {min: min - 1, max: max + 1} : {min, max};
}

function mapRange(value: number, from: {min: number; max: number}, toMin: number, toMax: number): number {
  const q = (value - from.min) / Math.max(1e-12, from.max - from.min);
  return toMin + Math.max(0, Math.min(1, q)) * (toMax - toMin);
}

function complexSnapshot(
  outputs: readonly TransformOutputPoint[],
  trace: readonly TransformContributionRecord[],
  traceIndex: number,
): ComplexValue[] {
  const values = outputs.map(() => ({re: 0, im: 0}));
  for (let i = 0; i <= traceIndex && i < trace.length; i++) {
    const row = trace[i];
    values[row.pointIndex] = row.cumulative;
  }
  return values;
}

function contributionRowsForPoint(result: TransformRunResult, pointIndex: number): TransformContributionRecord[] {
  return result.trace.filter(row => row.pointIndex === pointIndex);
}

function transformFrame(result: TransformRunResult, traceIndex: number): {shapes: Shape[]; caption: string} {
  const row = result.trace[Math.min(traceIndex, result.trace.length - 1)];
  const current = complexSnapshot(result.outputs, result.trace, traceIndex);
  const sampleValues = result.samples.map(s => s.value);
  const sampleRange = finiteRange(sampleValues);
  const maxMag = Math.max(1e-12, ...result.outputs.map(o => o.magnitude), ...current.map(v => Math.hypot(v.re, v.im)));
  const shapes: Shape[] = [
    {kind: 'rect', x: 0, y: 0, w: 920, h: 560, fill: '#f8fafc'},
    {kind: 'text', x: 460, y: 34, text: `${result.kind.toUpperCase()} transform`, fontSize: 20, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'},
    {kind: 'text', x: 460, y: 58, text: `sample ${row.sampleIndex} at ${result.samples[0].abscissaName}=${row.abscissa.toPrecision(5)} -> ${row.pointLabel}`, fontSize: 13, anchor: 'middle', fill: '#334155'},
  ];

  const nodes = [
    {id: 'source', label: 'source', x: 92, y: 126, fill: '#dbeafe'},
    {id: 'kernel', label: 'kernel', x: 270, y: 126, fill: '#fef3c7'},
    {id: 'accumulator', label: 'accumulator', x: 486, y: 126, fill: '#dcfce7'},
    {id: 'sink', label: 'sink', x: 706, y: 126, fill: '#ede9fe'},
  ];
  for (let i = 0; i < nodes.length - 1; i++) {
    shapes.push({kind: 'line', x1: nodes[i].x + 58, y1: nodes[i].y, x2: nodes[i + 1].x - 58, y2: nodes[i + 1].y, stroke: '#64748b', strokeWidth: 3});
  }
  for (const node of nodes) {
    shapes.push({kind: 'rect', x: node.x - 58, y: node.y - 28, w: 116, h: 56, rx: 6, fill: node.fill, stroke: '#1e293b', strokeWidth: 1.5});
    shapes.push({kind: 'text', x: node.x, y: node.y + 4, text: node.label, fontSize: 12, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
  }
  const tokenQ = result.trace.length <= 1 ? 1 : traceIndex / (result.trace.length - 1);
  shapes.push({kind: 'circle', x: 92 + tokenQ * (706 - 92), y: 88, r: 8, fill: color(row.pointIndex), stroke: '#ffffff', strokeWidth: 2, title: 'current movable contribution token'});

  const barX = 64;
  const barY = 220;
  const barW = 760 / result.samples.length;
  for (let i = 0; i < result.samples.length; i++) {
    if (result.samples.length > 160 && i % Math.ceil(result.samples.length / 160) !== 0) continue;
    const x = barX + i * barW;
    const h = mapRange(sampleValues[i], sampleRange, 4, 78);
    const active = i === row.sampleIndex;
    shapes.push({kind: 'rect', x, y: barY + 82 - h, w: Math.max(1, barW - 1), h, fill: active ? '#f97316' : '#cbd5e1', opacity: active ? 1 : 0.8, title: `sample ${i}: ${sampleValues[i].toPrecision(5)}`});
  }
  shapes.push({kind: 'text', x: 64, y: 206, text: 'input samples', fontSize: 13, fontWeight: 'bold', fill: '#0f172a'});

  const planeCx = 235;
  const planeCy = 415;
  const planeR = 112;
  shapes.push({kind: 'circle', x: planeCx, y: planeCy, r: planeR, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1});
  shapes.push({kind: 'line', x1: planeCx - planeR, y1: planeCy, x2: planeCx + planeR, y2: planeCy, stroke: '#94a3b8', strokeWidth: 1});
  shapes.push({kind: 'line', x1: planeCx, y1: planeCy - planeR, x2: planeCx, y2: planeCy + planeR, stroke: '#94a3b8', strokeWidth: 1});
  shapes.push({kind: 'text', x: planeCx, y: planeCy - planeR - 18, text: 'complex accumulation', fontSize: 13, anchor: 'middle', fontWeight: 'bold', fill: '#0f172a'});
  current.forEach((v, i) => {
    const x = planeCx + (v.re / maxMag) * (planeR - 12);
    const y = planeCy - (v.im / maxMag) * (planeR - 12);
    shapes.push({kind: 'line', x1: planeCx, y1: planeCy, x2: x, y2: y, stroke: color(i), strokeWidth: 2, opacity: 0.8});
    shapes.push({kind: 'circle', x, y, r: 5, fill: color(i), stroke: '#ffffff', strokeWidth: 1.5, title: result.outputs[i].label});
  });

  const tableX = 450;
  const tableY = 342;
  shapes.push({kind: 'text', x: tableX, y: tableY - 22, text: 'current totals', fontSize: 13, fontWeight: 'bold', fill: '#0f172a'});
  result.outputs.slice(0, 6).forEach((output, i) => {
    const y = tableY + i * 28;
    const value = current[output.pointIndex];
    shapes.push({kind: 'circle', x: tableX, y: y - 4, r: 5, fill: color(output.pointIndex)});
    shapes.push({kind: 'text', x: tableX + 16, y, text: output.label, fontSize: 12, fill: '#334155'});
    shapes.push({kind: 'text', x: tableX + 128, y, text: formatComplex(value, 4), fontSize: 12, fill: '#0f172a'});
    shapes.push({kind: 'text', x: tableX + 330, y, text: `target ${formatComplex(output.value, 4)}`, fontSize: 12, fill: '#475569'});
  });

  return {
    shapes,
    caption: `${result.entityFramework.sources[0]} -> ${result.entityFramework.stations.join(' -> ')} -> ${result.entityFramework.sinks[0]}`,
  };
}

async function animateTransform(result: TransformRunResult, runtime: {outputs?: {html?: string; frames?: string}}): Promise<void> {
  const {htmlPath, frames} = framesPath(runtime, `${result.kind}-transform`);
  const stride = Math.max(1, Math.ceil(Math.max(1, result.trace.length) / 240));
  const rec = new FrameRecorder({
    framesPath: frames,
    htmlPath,
    width: 920,
    height: 560,
    fps: 16,
    title: `${result.kind.toUpperCase()} transform`,
    subtitle: 'Source, transform kernel station, accumulator station, and sink exchange movable sample and contribution tokens',
    recordEveryTicks: stride,
  });
  for (let i = 0; i < result.trace.length; i++) {
    rec.frame(i, i, () => transformFrame(result, i));
  }
  if (result.trace.length > 0 && (result.trace.length - 1) % stride !== 0) {
    const last = result.trace.length - 1;
    rec.frame(last, Math.ceil(last / stride) * stride, () => transformFrame(result, last));
  }
  rec.setCharts(result.outputs.slice(0, 5).map((output, i) => {
    const rows = contributionRowsForPoint(result, output.pointIndex);
    return {
      x: 60,
      y: 36 + i * 92,
      w: 250,
      h: 66,
      title: output.label,
      series: [
        {label: 'real', color: color(output.pointIndex), t: rows.map((_, j) => j), y: rows.map(r => r.cumulative.re)},
        {label: 'imag', color: '#111827', t: rows.map((_, j) => j), y: rows.map(r => r.cumulative.im)},
      ],
    };
  }));
  await rec.finish();
}

registerModel<ZTransformParams, TransformRunResult>({
  id: 'z-transform',
  description: 'Finite Z-transform as source, kernel, accumulator, and sink stations exchanging movable contribution tokens.',
  schema: zTransformSchema,
  zodSchema: zTransformZodSchema,
  run(params) {
    return runZTransform(params);
  },
  summarize(result) {
    return summarizeTransform('Z-TRANSFORM (DES)', result);
  },
  writeCsv: writeTransformCsv,
  animate(result, _params, runtime) {
    return animateTransform(result, runtime);
  },
  examples: [{
    name: 'finite geometric sequence',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'z-transform',
      description: 'Finite geometric sequence evaluated at several z-plane points.',
      parameters: {
        sequence: [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        startIndex: 0,
        zValues: [
          {label: 'z=2', re: 2},
          {label: 'z=1', re: 1},
          {label: 'z=-1', re: -1},
        ],
      },
    },
  }],
});

registerModel<LaplaceTransformParams, TransformRunResult>({
  id: 'laplace-transform',
  description: 'Numerical Laplace transform with function samples moving through transform kernel stations.',
  schema: laplaceTransformSchema,
  zodSchema: laplaceTransformZodSchema,
  run(params) {
    return runLaplaceTransform(params);
  },
  summarize(result) {
    return summarizeTransform('LAPLACE TRANSFORM (DES)', result);
  },
  writeCsv: writeTransformCsv,
  animate(result, _params, runtime) {
    return animateTransform(result, runtime);
  },
  examples: [{
    name: 'decaying exponential',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'laplace-transform',
      description: 'Laplace transform of exp(-a t) over a finite integration window.',
      parameters: {
        expression: 'exp(-a*t)',
        constants: {a: 2},
        t0: 0,
        t1: 8,
        dt: 0.01,
        quadrature: 'trapezoid',
        sValues: [
          {label: 's=1', re: 1},
          {label: 's=0.5+i', re: 0.5, im: 1},
        ],
      },
    },
  }],
});

registerModel<FourierTransformParams, TransformRunResult>({
  id: 'fourier-transform',
  description: 'Numerical Fourier transform using angular frequencies and movable weighted sample tokens.',
  schema: fourierTransformSchema,
  zodSchema: fourierTransformZodSchema,
  run(params) {
    return runFourierTransform(params);
  },
  summarize(result) {
    return summarizeTransform('FOURIER TRANSFORM (DES)', result);
  },
  writeCsv: writeTransformCsv,
  animate(result, _params, runtime) {
    return animateTransform(result, runtime);
  },
  examples: [{
    name: 'windowed sinusoid',
    spec: {
      $schema: 'des/model-spec/v1',
      model: 'fourier-transform',
      description: 'Fourier transform of sin(2t) on one period with angular frequency probes.',
      parameters: {
        expression: 'sin(omega0*t)',
        constants: {omega0: 2},
        t0: 0,
        t1: 6.283185307179586,
        dt: 0.0031415926535897933,
        quadrature: 'trapezoid',
        omegaValues: [0, 1, 2, 3, -2],
      },
    },
  }],
});
