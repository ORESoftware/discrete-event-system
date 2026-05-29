'use strict';

// =============================================================================
// Universal DES model spec.
//
// This is the portable JSON document shape for the modeling layer. It captures:
//   1. original user input,
//   2. normalized mathematics,
//   3. generated DES stationary entities and moving-entity edges,
//   4. solver/runtime intent.
//
// The existing `des/model-spec/v1` registry envelope remains the execution
// envelope. A universal document can be converted to one when the solver target
// is a registered model such as `math-equation`.
// =============================================================================

import {ValidationCheck} from './des-base/validation';
import {DESModelSpec, DESRuntimeConfig} from './des-spec';
import {
  EquationInputFormat,
  MathEquationInputParams,
  MathEquationResult,
} from './math-equation-input';
import {
  BlockGraphEdge,
  BlockGraphNode,
  Heat1DBlockParams,
  ODEBlockSystemParams,
} from './math-blocks';

export type UniversalModelKind =
  | 'ode'
  | 'pde'
  | 'optimization'
  | 'network-flow'
  | 'traffic-flow'
  | 'queueing'
  | 'agent'
  | 'custom';

export type UniversalInputFormat =
  | EquationInputFormat
  | 'json'
  | 'xml'
  | 'text'
  | 'manual';

export interface UniversalDESModelSpec {
  $schema: 'des/universal-model/v1';
  id: string;
  description?: string;
  originalInput: UniversalOriginalInput;
  math: UniversalMathSpec;
  des: UniversalDESNetworkSpec;
  solver: UniversalSolverSpec;
  runtime?: DESRuntimeConfig;
  metadata?: {
    author?: string;
    createdAt?: string;
    tags?: string[];
    notes?: string;
    [key: string]: unknown;
  };
}

export interface UniversalOriginalInput {
  format: UniversalInputFormat;
  content?: string;
  uri?: string;
  contentType?: string;
  language?: string;
  capturedAt?: string;
  metadata?: Record<string, unknown>;
}

export interface UniversalMathSpec {
  kind: UniversalModelKind;
  independentVariables: UniversalMathVariable[];
  stateVariables: UniversalMathVariable[];
  parameters?: UniversalMathParameter[];
  equations: UniversalMathEquation[];
  initialConditions?: UniversalMathCondition[];
  boundaryConditions?: UniversalMathCondition[];
  constraints?: UniversalMathEquation[];
  objectives?: UniversalMathEquation[];
  domain?: Record<string, unknown>;
  numerics?: UniversalNumericsSpec;
  normalized?: UniversalNormalizedMath;
}

export interface UniversalMathVariable {
  id: string;
  role: 'independent' | 'state' | 'field' | 'input' | 'output' | 'algebraic' | 'parameter';
  initial?: number | string | number[];
  units?: string;
  domain?: Record<string, unknown>;
  description?: string;
}

export interface UniversalMathParameter {
  id: string;
  value: number | string | number[];
  units?: string;
  description?: string;
}

export interface UniversalMathEquation {
  id: string;
  kind: 'ode' | 'pde' | 'algebraic' | 'constraint' | 'objective' | 'boundary' | 'initial';
  lhs?: string;
  rhs?: string;
  expression?: string;
  normalizedExpression?: string;
  variables?: string[];
  metadata?: Record<string, unknown>;
}

export interface UniversalMathCondition {
  id: string;
  variable: string;
  at?: Record<string, number | string>;
  value: number | string | number[];
  kind: 'initial' | 'dirichlet' | 'neumann' | 'periodic' | 'source' | 'sink';
  expression?: string;
}

export interface UniversalNumericsSpec {
  time?: {
    t0?: number;
    t1?: number;
    dt?: number;
    steps?: number;
  };
  space?: {
    dimensions?: number;
    cells?: number | number[];
    length?: number | number[];
    dx?: number | number[];
  };
  method?: string;
  tolerances?: Record<string, number>;
}

export interface UniversalNormalizedMath {
  targetModel: 'math-equation' | 'math-ode-blocks' | 'math-heat1d-blocks' | string;
  parameters: Record<string, unknown>;
}

export interface UniversalDESNetworkSpec {
  time?: UniversalNumericsSpec['time'];
  stationaryEntities: UniversalStationaryEntity[];
  movingEntities: UniversalMovingEntity[];
  graph: {
    nodes: UniversalStationaryEntity[];
    edges: UniversalGraphEdge[];
  };
  sources?: UniversalEndpointSpec[];
  sinks?: UniversalEndpointSpec[];
  observability?: {
    recordSignals?: boolean;
    recordState?: boolean;
    recordGraph?: boolean;
  };
}

export interface UniversalStationaryEntity {
  id: string;
  kind: string;
  role?: 'source' | 'sink' | 'processor' | 'integrator' | 'field-cell' | 'boundary' | 'logic' | 'optimizer' | 'observer';
  className?: string;
  parameters?: Record<string, unknown>;
  ports?: {
    inputs?: string[];
    outputs?: string[];
  };
  position?: {
    x?: number;
    y?: number;
    z?: number;
    index?: number | number[];
  };
  metadata?: Record<string, unknown>;
}

export interface UniversalMovingEntity {
  id: string;
  kind: string;
  tokenType: string;
  payloadSchema?: Record<string, unknown>;
  semantics?: string;
}

export interface UniversalGraphEdge {
  id: string;
  from: UniversalPortRef;
  to: UniversalPortRef;
  movingEntity: string;
  delayTicks?: number;
  transform?: string;
  metadata?: Record<string, unknown>;
}

export interface UniversalPortRef {
  entityId: string;
  port: string;
}

export interface UniversalEndpointSpec {
  id: string;
  entityId: string;
  port?: string;
  role?: string;
  variable?: string;
  value?: unknown;
  record?: boolean;
}

export interface UniversalSolverSpec {
  targetModel: 'math-equation' | 'math-ode-blocks' | 'math-heat1d-blocks' | string;
  method?: string;
  options?: Record<string, unknown>;
}

export function isUniversalDESModelSpec(value: unknown): value is UniversalDESModelSpec {
  return typeof value === 'object' &&
    value !== null &&
    (value as { $schema?: unknown }).$schema === 'des/universal-model/v1';
}

export function validateUniversalDESModelSpec(spec: UniversalDESModelSpec): ValidationCheck[] {
  const checks: ValidationCheck[] = [];
  push(checks, 'universal-schema', spec.$schema === 'des/universal-model/v1', String(spec.$schema), 'des/universal-model/v1');
  push(checks, 'universal-id', isNonEmpty(spec.id), spec.id, 'non-empty string');
  push(checks, 'original-input-present', !!spec.originalInput && isNonEmpty(spec.originalInput.format), spec.originalInput?.format, 'format string');
  push(checks, 'original-input-content-or-uri', !!spec.originalInput && (isNonEmpty(spec.originalInput.content) || isNonEmpty(spec.originalInput.uri)), 'content/uri', 'one must be present');
  push(checks, 'math-kind', !!spec.math && isNonEmpty(spec.math.kind), spec.math?.kind, 'math kind');
  push(checks, 'math-equations-non-empty', Array.isArray(spec.math?.equations) && spec.math.equations.length > 0, String(spec.math?.equations?.length ?? 0), 'at least one equation');
  push(checks, 'stationary-entities-non-empty', Array.isArray(spec.des?.stationaryEntities) && spec.des.stationaryEntities.length > 0, String(spec.des?.stationaryEntities?.length ?? 0), 'at least one stationary entity');
  push(checks, 'moving-entities-non-empty', Array.isArray(spec.des?.movingEntities) && spec.des.movingEntities.length > 0, String(spec.des?.movingEntities?.length ?? 0), 'at least one moving entity kind');
  push(checks, 'solver-target-model', !!spec.solver && isNonEmpty(spec.solver.targetModel), spec.solver?.targetModel, 'registered target model id');

  const nodeIds = spec.des?.stationaryEntities?.map(n => n.id) ?? [];
  const movingIds = spec.des?.movingEntities?.map(m => m.id) ?? [];
  push(checks, 'stationary-ids-unique', unique(nodeIds), duplicate(nodeIds) ?? 'unique', 'unique stationary ids');
  push(checks, 'moving-ids-unique', unique(movingIds), duplicate(movingIds) ?? 'unique', 'unique moving ids');
  const nodeSet = new Set(nodeIds);
  const movingSet = new Set(movingIds);
  for (const edge of spec.des?.graph?.edges ?? []) {
    push(checks, `edge-from-ref/${edge.id}`, nodeSet.has(edge.from.entityId), edge.from.entityId, 'known stationary entity');
    push(checks, `edge-to-ref/${edge.id}`, nodeSet.has(edge.to.entityId), edge.to.entityId, 'known stationary entity');
    push(checks, `edge-moving-ref/${edge.id}`, movingSet.has(edge.movingEntity), edge.movingEntity, 'known moving entity');
  }
  for (const source of spec.des?.sources ?? []) {
    push(checks, `source-ref/${source.id}`, nodeSet.has(source.entityId), source.entityId, 'known stationary entity');
  }
  for (const sink of spec.des?.sinks ?? []) {
    push(checks, `sink-ref/${sink.id}`, nodeSet.has(sink.entityId), sink.entityId, 'known stationary entity');
  }
  const dt = spec.math?.numerics?.time?.dt ?? spec.des?.time?.dt;
  if (dt !== undefined) push(checks, 'time-dt-positive', typeof dt === 'number' && Number.isFinite(dt) && dt > 0, String(dt), 'finite dt > 0');
  return checks;
}

export function assertUniversalDESModelSpec(spec: UniversalDESModelSpec): void {
  const failed = validateUniversalDESModelSpec(spec).filter(c => !c.passed);
  if (failed.length === 0) return;
  throw new Error(`invalid universal DES model spec:\n  ${failed.map(c => `${c.name}: observed=${c.observed ?? ''} expected=${c.expected ?? ''}`).join('\n  ')}`);
}

export function universalToDESModelSpec(spec: UniversalDESModelSpec): DESModelSpec<Record<string, unknown>> {
  assertUniversalDESModelSpec(spec);
  if (spec.solver.targetModel !== 'math-equation') {
    throw new Error(`universalToDESModelSpec: unsupported targetModel "${spec.solver.targetModel}"`);
  }
  const params = universalToMathEquationInput(spec) as unknown as Record<string, unknown>;
  return {
    $schema: 'des/model-spec/v1',
    model: 'math-equation',
    description: spec.description,
    parameters: params,
    runtime: spec.runtime,
    metadata: spec.metadata,
  };
}

export function universalToMathEquationInput(spec: UniversalDESModelSpec): MathEquationInputParams {
  const normalized = spec.math.normalized;
  if (normalized?.targetModel === 'math-equation') {
    return normalized.parameters as unknown as MathEquationInputParams;
  }
  const format = spec.originalInput.format;
  if (format !== 'json' && format !== 'latex' && format !== 'xml') {
    throw new Error(`universalToMathEquationInput: original input format "${format}" cannot be run by math-equation`);
  }
  return {
    format,
    kind: spec.math.kind === 'pde' ? 'heat1d' : 'ode',
    equation: spec.originalInput.content,
    t0: spec.math.numerics?.time?.t0,
    t1: spec.math.numerics?.time?.t1,
    dt: spec.math.numerics?.time?.dt,
  };
}

export function universalFromMathEquationResult(
  input: MathEquationInputParams,
  result: MathEquationResult,
  opts: {id?: string; description?: string; runtime?: DESRuntimeConfig; metadata?: UniversalDESModelSpec['metadata']} = {},
): UniversalDESModelSpec {
  const stationaryEntities = result.network.nodes.map(node => universalStationaryFromBlock(node));
  const edges = result.network.edges.map((edge, i) => universalEdgeFromBlock(edge, i));
  const spec: UniversalDESModelSpec = {
    $schema: 'des/universal-model/v1',
    id: opts.id ?? `universal-${result.kind}`,
    description: opts.description,
    originalInput: originalFromInput(input),
    math: result.kind === 'ode'
      ? mathFromODEResult(input, result.normalized as ODEBlockSystemParams)
      : mathFromHeatResult(input, result.normalized as Heat1DBlockParams),
    des: {
      time: result.kind === 'ode'
        ? timeFromODE(result.normalized as ODEBlockSystemParams)
        : timeFromHeat(result.normalized as Heat1DBlockParams),
      stationaryEntities,
      movingEntities: [{
        id: 'MathSignal',
        kind: 'signal-token',
        tokenType: 'MathSignal',
        payloadSchema: {value: 'number', time: 'number', tick: 'integer', sourceId: 'string', channel: 'string'},
        semantics: 'Scalar value token moving between stationary DES math blocks.',
      }],
      graph: {nodes: stationaryEntities, edges},
      sources: sourceEndpoints(result.network.nodes, result.normalized),
      sinks: sinkEndpoints(result.network.nodes),
      observability: {recordSignals: true, recordState: true, recordGraph: true},
    },
    solver: {targetModel: 'math-equation', method: methodFromNormalized(result.normalized)},
    runtime: opts.runtime,
    metadata: opts.metadata,
  };
  spec.math.normalized = {
    targetModel: 'math-equation',
    parameters: input as unknown as Record<string, unknown>,
  };
  assertUniversalDESModelSpec(spec);
  return spec;
}

function mathFromODEResult(input: MathEquationInputParams, params: ODEBlockSystemParams): UniversalMathSpec {
  const constants = params.constants ?? {};
  return {
    kind: 'ode',
    independentVariables: [{id: 't', role: 'independent'}],
    stateVariables: params.states.map(s => ({id: s.name, role: 'state', initial: s.initial})),
    parameters: Object.entries(constants).map(([id, value]) => ({id, value})),
    equations: params.states.map(s => ({
      id: `ode:${s.name}`,
      kind: 'ode',
      lhs: `d${s.name}/dt`,
      rhs: s.derivative,
      normalizedExpression: s.derivative,
      variables: params.states.map(x => x.name),
    })),
    initialConditions: params.states.map(s => ({id: `initial:${s.name}`, variable: s.name, at: {t: params.t0 ?? 0}, value: s.initial, kind: 'initial'})),
    numerics: {time: timeFromODE(params), method: params.method ?? input.method ?? 'euler'},
  };
}

function mathFromHeatResult(input: MathEquationInputParams, params: Heat1DBlockParams): UniversalMathSpec {
  return {
    kind: 'pde',
    independentVariables: [{id: 't', role: 'independent'}, {id: 'x', role: 'independent', domain: {length: params.length}}],
    stateVariables: [{id: 'u', role: 'field', initial: params.initialValues ?? params.initialExpression ?? 'sin(pi*x/length)'}],
    parameters: [
      {id: 'alpha', value: params.alpha},
      {id: 'length', value: params.length},
      {id: 'cells', value: params.cells},
      ...Object.entries(params.constants ?? {}).map(([id, value]) => ({id, value})),
    ],
    equations: [{
      id: 'pde:heat1d',
      kind: 'pde',
      lhs: 'du/dt',
      rhs: 'alpha*d2u/dx2',
      normalizedExpression: 'alpha*(u[i-1] - 2*u[i] + u[i+1]) / dx^2',
      variables: ['u', 'x', 't'],
      metadata: {sourceEquation: input.equation},
    }],
    initialConditions: [{id: 'initial:u', variable: 'u', at: {t: params.t0 ?? 0}, value: params.initialValues ?? params.initialExpression ?? 'sin(pi*x/length)', kind: 'initial'}],
    boundaryConditions: [
      {id: 'boundary:left', variable: 'u', at: {x: 0}, value: params.leftBoundary ?? 0, kind: 'dirichlet'},
      {id: 'boundary:right', variable: 'u', at: {x: params.length}, value: params.rightBoundary ?? 0, kind: 'dirichlet'},
    ],
    domain: {x: [0, params.length]},
    numerics: {time: timeFromHeat(params), space: {dimensions: 1, cells: params.cells, length: params.length}, method: 'explicit-euler-laplacian'},
  };
}

function universalStationaryFromBlock(node: BlockGraphNode): UniversalStationaryEntity {
  const inputs = Array.isArray(node.inputs) ? node.inputs : node.inputs ? Object.keys(node.inputs) : [];
  const outputs = node.output ? [node.output] : [];
  return {
    id: node.id,
    kind: node.kind,
    role: roleFromBlockKind(node.kind),
    className: node.kind,
    parameters: node.expression ? {expression: node.expression} : undefined,
    ports: {inputs, outputs},
  };
}

function universalEdgeFromBlock(edge: BlockGraphEdge, index: number): UniversalGraphEdge {
  return {
    id: `edge:${index}:${edge.from}->${edge.to}`,
    from: {entityId: edge.from, port: edge.fromChannel},
    to: {entityId: edge.to, port: edge.toChannel},
    movingEntity: edge.signal,
  };
}

function originalFromInput(input: MathEquationInputParams): UniversalOriginalInput {
  if (input.equation) {
    return {
      format: input.format,
      content: input.equation,
      contentType: input.format === 'latex' ? 'application/x-latex' : input.format === 'xml' ? 'application/xml' : 'application/json',
    };
  }
  return {
    format: 'json',
    content: JSON.stringify(input.ode ?? input.heat1d ?? input, null, 2),
    contentType: 'application/json',
  };
}

function roleFromBlockKind(kind: string): UniversalStationaryEntity['role'] {
  if (kind.includes('integrator')) return 'integrator';
  if (kind.includes('boundary')) return 'boundary';
  if (kind.includes('laplacian')) return 'processor';
  if (kind.includes('expression')) return 'processor';
  if (kind.includes('source')) return 'source';
  if (kind.includes('sink')) return 'sink';
  return 'processor';
}

function sourceEndpoints(nodes: readonly BlockGraphNode[], normalized: ODEBlockSystemParams | Heat1DBlockParams): UniversalEndpointSpec[] {
  const endpoints: UniversalEndpointSpec[] = [];
  if (isODEParams(normalized)) {
    for (const s of normalized.states) {
      endpoints.push({id: `source:initial:${s.name}`, entityId: `integrator:${s.name}`, port: 'out', role: 'initial-condition', variable: s.name, value: s.initial});
    }
    return endpoints;
  }
  for (const node of nodes) {
    if (node.kind === 'constant-boundary') endpoints.push({id: `source:${node.id}`, entityId: node.id, port: 'out', role: 'boundary-condition'});
  }
  return endpoints;
}

function sinkEndpoints(nodes: readonly BlockGraphNode[]): UniversalEndpointSpec[] {
  return nodes
    .filter(n => n.kind.includes('integrator') || n.kind === 'constant-boundary')
    .map(n => ({id: `sink:trace:${n.id}`, entityId: n.id, port: 'out', role: 'trace-recorder', record: true}));
}

function methodFromNormalized(normalized: ODEBlockSystemParams | Heat1DBlockParams): string | undefined {
  return isODEParams(normalized) ? normalized.method : 'explicit-euler-laplacian';
}

function timeFromODE(params: ODEBlockSystemParams): UniversalNumericsSpec['time'] {
  return {t0: params.t0 ?? 0, t1: params.t1, dt: params.dt, steps: Math.round((params.t1 - (params.t0 ?? 0)) / params.dt)};
}

function timeFromHeat(params: Heat1DBlockParams): UniversalNumericsSpec['time'] {
  return {t0: params.t0 ?? 0, t1: params.t1, dt: params.dt, steps: Math.round((params.t1 - (params.t0 ?? 0)) / params.dt)};
}

function isODEParams(value: ODEBlockSystemParams | Heat1DBlockParams): value is ODEBlockSystemParams {
  return Array.isArray((value as ODEBlockSystemParams).states);
}

function push(checks: ValidationCheck[], name: string, passed: boolean, observed?: string, expected?: string): void {
  checks.push({name, passed, observed, expected, group: 'universal-model'});
}

function isNonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function unique(values: readonly string[]): boolean {
  return new Set(values).size === values.length;
}

function duplicate(values: readonly string[]): string | undefined {
  const seen = new Set<string>();
  for (const v of values) {
    if (seen.has(v)) return v;
    seen.add(v);
  }
  return undefined;
}
