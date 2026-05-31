// RUST MIGRATION: target module src/des/general/math_equation_input.rs.
// RUST MIGRATION: EquationInputFormat, EquationProblemKind, and normalizeMathEquationProblem's union return become enums; params/network/result structs become serde structs.
// RUST MIGRATION: runMathEquationProblem is a graph-visible input transform and should be a PureTransform entry struct; latexToExpression can remain a free parser helper.
// RUST MIGRATION: Record<string, unknown/number> maps become serde_json::Map/HashMap<String, f64>, and Attrs/ExprToken become private structs/enums.
// RUST MIGRATION: Manual JSON/XML/LaTeX parsing should return Result with typed parse errors; consider quick-xml and a small expression tokenizer module in Rust.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/math-equation-input.rs  (module des::general::math_equation_input)
// 1:1 file move. Normalizes JSON/LaTeX/XML equation input into math-block ODE/PDE params, then runs.
//
// Declarations → Rust:
//   type EquationInputFormat = 'json'|'latex'|'xml' -> enum; type EquationProblemKind = 'ode'|'heat1d' -> enum
//   interface MathEquationInputParams/MathEquationNetwork/MathEquationResult/Attrs/ExprToken -> structs
//   fn runMathEquationProblem / normalizeMathEquationProblem / latexToExpression -> fns
//   (+ many private parse/coerce helpers: stateFromJson, *FromXml, *OrDefault, ... -> assoc fns)
//
// Conversion notes (file-specific):
//   - `normalizeMathEquationProblem` returns `{kind:'ode',..} | {kind:'heat1d',..}` -> enum + `match`.
//   - `unknown` / `Record<string, unknown>` inputs -> `serde_json::Value`; coercion helpers -> typed parsing.
//   - Hand-rolled LaTeX/XML scanning (textBetween/parseAttrs/xmlDecode/...) -> a real parser or `regex`/`serde`.
//   - `logger?: { log(event) }` callback -> optional `&dyn Logger` trait; `Preconditions` throws -> `Result`.
//   - Builds on expr.rs + math-blocks.rs (see those headers).
// =============================================================================

// =============================================================================
// Equation input normalizer for math-block DES models.
//
// User-facing input can be structured JSON, constrained LaTeX, or a tiny XML
// dialect. This module converts those formats into the existing stationary
// math-block ODE/PDE model parameters, runs the numerical solver, and returns
// the generated node/edge network.
// =============================================================================

import {Preconditions} from './des-base/preconditions';
import {evaluate, parse} from './expr';
import {
  BlockGraphEdge,
  BlockGraphNode,
  Heat1DBlockParams,
  Heat1DBlockResult,
  IntegratorMethod,
  ODEBlockSystemParams,
  ODEBlockSystemResult,
  ODEStateSpec,
  runHeat1DBlockGrid,
  runODEBlockSystem,
} from './math-blocks';

export type EquationInputFormat = 'json' | 'latex' | 'xml';
export type EquationProblemKind = 'ode' | 'heat1d';

export interface MathEquationInputParams {
  format: EquationInputFormat;
  kind?: EquationProblemKind;
  equation?: string;
  ode?: Record<string, unknown>;
  heat1d?: Record<string, unknown>;
  states?: unknown[];
  constants?: Record<string, unknown>;
  initial?: Record<string, unknown>;
  t0?: number;
  t1?: number;
  dt?: number;
  method?: IntegratorMethod;
  cells?: number;
  length?: number;
  alpha?: number;
  initialExpression?: string;
  initialValues?: number[];
  leftBoundary?: number;
  rightBoundary?: number;
}

export interface MathEquationNetwork {
  nodes: BlockGraphNode[];
  edges: BlockGraphEdge[];
}

export interface MathEquationResult {
  inputFormat: EquationInputFormat;
  kind: EquationProblemKind;
  equation?: string;
  normalized: ODEBlockSystemParams | Heat1DBlockParams;
  network: MathEquationNetwork;
  ode?: ODEBlockSystemResult;
  heat1d?: Heat1DBlockResult;
  validation: readonly {name: string; passed: boolean; group?: string}[];
}

interface Attrs {
  [key: string]: string;
}

const FUNCTIONS = new Set(['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'ln', 'sqrt', 'abs']);
const GREEK: Record<string, string> = {
  '\\alpha': 'alpha',
  '\\beta': 'beta',
  '\\gamma': 'gamma',
  '\\delta': 'delta',
  '\\epsilon': 'epsilon',
  '\\varepsilon': 'epsilon',
  '\\theta': 'theta',
  '\\lambda': 'lambda',
  '\\mu': 'mu',
  '\\sigma': 'sigma',
  '\\tau': 'tau',
  '\\omega': 'omega',
  '\\pi': 'pi',
};

export function runMathEquationProblem(params: MathEquationInputParams, logger?: {log(event: Record<string, unknown>): void}): MathEquationResult {
  console.debug(`[math-equation-input] running problem: format=${params.format}, kind=${params.kind ?? 'inferred'}, equation=${params.equation ? JSON.stringify(params.equation) : '(none)'}`);
  const normalized = normalizeMathEquationProblem(params);
  logger?.log({kind: 'math-equation-normalized', level: 'info', format: params.format, problemKind: normalized.kind});
  if (normalized.kind === 'ode') {
    const ode = runODEBlockSystem(normalized.params, logger);
    return {
      inputFormat: params.format,
      kind: 'ode',
      equation: params.equation,
      normalized: normalized.params,
      network: {nodes: ode.blockGraph, edges: ode.blockGraphEdges},
      ode,
      validation: ode.validation,
    };
  }
  const heat1d = runHeat1DBlockGrid(normalized.params, logger);
  return {
    inputFormat: params.format,
    kind: 'heat1d',
    equation: params.equation,
    normalized: normalized.params,
    network: {nodes: heat1d.blockGraph, edges: heat1d.blockGraphEdges},
    heat1d,
    validation: heat1d.validation,
  };
}

export function normalizeMathEquationProblem(params: MathEquationInputParams): {kind: 'ode'; params: ODEBlockSystemParams} | {kind: 'heat1d'; params: Heat1DBlockParams} {
  Preconditions.check('MathEquationInput', 'format', 'be json, latex, or xml', ['json', 'latex', 'xml'].includes(params.format), params.format);
  const kind = inferProblemKind(params);
  if (params.format === 'json') {
    return kind === 'ode'
      ? {kind: 'ode', params: normalizeJsonODE(params)}
      : {kind: 'heat1d', params: normalizeJsonHeat(params)};
  }
  if (params.format === 'latex') {
    Preconditions.check('MathEquationInput', 'equation', 'be a non-empty LaTeX string', typeof params.equation === 'string' && params.equation.trim().length > 0, params.equation);
    return kind === 'ode'
      ? {kind: 'ode', params: normalizeLatexODE(params)}
      : {kind: 'heat1d', params: normalizeLatexHeat(params)};
  }
  Preconditions.check('MathEquationInput', 'equation', 'be a non-empty XML string', typeof params.equation === 'string' && params.equation.trim().length > 0, params.equation);
  return kind === 'ode'
    ? {kind: 'ode', params: normalizeXmlODE(params)}
    : {kind: 'heat1d', params: normalizeXmlHeat(params)};
}

function inferProblemKind(params: MathEquationInputParams): EquationProblemKind {
  if (params.kind !== undefined) {
    Preconditions.check('MathEquationInput', 'kind', 'be ode or heat1d', params.kind === 'ode' || params.kind === 'heat1d', params.kind);
    return params.kind;
  }
  if (params.format === 'json') {
    if (params.heat1d) return 'heat1d';
    return 'ode';
  }
  const equation = params.equation ?? '';
  if (params.format === 'xml') {
    const root = rootName(equation);
    if (root === 'heat1d' || root === 'pde') return 'heat1d';
    return 'ode';
  }
  return /\\partial|partial/.test(equation) ? 'heat1d' : 'ode';
}

function normalizeJsonODE(params: MathEquationInputParams): ODEBlockSystemParams {
  const src = objectRecord(params.ode) ?? (params as unknown as Record<string, unknown>);
  const constants = mergeConstants(params.constants, objectRecord(src.constants));
  const initial = mergeInitial(params.initial, objectRecord(src.initial));
  const statesRaw = arrayValue(src.states ?? params.states, 'MathEquationInput.states');
  const states = statesRaw.map((raw, i) => stateFromJson(raw, i, initial, constants));
  const t0 = numberOrDefault(src.t0, params.t0 ?? 0, 'MathEquationInput.t0');
  const t1 = numberOrDefault(src.t1, params.t1 ?? 1, 'MathEquationInput.t1');
  const dt = numberOrDefault(src.dt, params.dt ?? defaultDt(t0, t1, 100), 'MathEquationInput.dt');
  const method = methodOrDefault(src.method, params.method ?? 'euler');
  return {states, constants, t0, t1, dt, method};
}

function normalizeJsonHeat(params: MathEquationInputParams): Heat1DBlockParams {
  const src = objectRecord(params.heat1d) ?? (params as unknown as Record<string, unknown>);
  const constants = mergeConstants(params.constants, objectRecord(src.constants));
  const cells = integerOrDefault(src.cells, params.cells ?? 31, 'MathEquationInput.cells');
  const length = numberOrDefault(src.length, params.length ?? 1, 'MathEquationInput.length');
  const alpha = numberOrDefault(src.alpha, params.alpha ?? constants.alpha ?? 0.01, 'MathEquationInput.alpha');
  const t0 = numberOrDefault(src.t0, params.t0 ?? 0, 'MathEquationInput.t0');
  const t1 = numberOrDefault(src.t1, params.t1 ?? 1, 'MathEquationInput.t1');
  const dt = numberOrDefault(src.dt, params.dt ?? stableHeatDt(t0, t1, cells, length, alpha), 'MathEquationInput.dt');
  return {
    cells,
    length,
    alpha,
    t0,
    t1,
    dt,
    constants,
    initialExpression: stringOrDefault(src.initialExpression, params.initialExpression ?? 'sin(pi*x/length)'),
    initialValues: numericArrayOrUndefined(src.initialValues ?? params.initialValues, 'MathEquationInput.initialValues'),
    leftBoundary: numberOrUndefined(src.leftBoundary ?? params.leftBoundary, 'MathEquationInput.leftBoundary'),
    rightBoundary: numberOrUndefined(src.rightBoundary ?? params.rightBoundary, 'MathEquationInput.rightBoundary'),
  };
}

function normalizeLatexODE(params: MathEquationInputParams): ODEBlockSystemParams {
  const constants = mergeConstants(params.constants);
  const initial = mergeInitial(params.initial);
  const states = parseLatexODE(params.equation ?? '', initial, constants);
  const t0 = params.t0 ?? 0;
  const t1 = params.t1 ?? 1;
  const dt = params.dt ?? defaultDt(t0, t1, 100);
  return {states, constants, t0, t1, dt, method: params.method ?? 'euler'};
}

function normalizeLatexHeat(params: MathEquationInputParams): Heat1DBlockParams {
  const constants = mergeConstants(params.constants);
  const cells = params.cells ?? 31;
  const length = params.length ?? 1;
  const alpha = params.alpha ?? constants.alpha ?? 0.01;
  const t0 = params.t0 ?? 0;
  const t1 = params.t1 ?? 1;
  const dt = params.dt ?? stableHeatDt(t0, t1, cells, length, alpha);
  return {
    cells,
    length,
    alpha,
    t0,
    t1,
    dt,
    constants,
    initialExpression: params.initialExpression ?? 'sin(pi*x/length)',
    initialValues: numericArrayOrUndefined(params.initialValues, 'MathEquationInput.initialValues'),
    leftBoundary: params.leftBoundary ?? 0,
    rightBoundary: params.rightBoundary ?? 0,
  };
}

function normalizeXmlODE(params: MathEquationInputParams): ODEBlockSystemParams {
  const xml = safeXml(params.equation ?? '');
  const attrs = rootAttrs(xml, 'ode');
  const constants = mergeConstants(params.constants, constantsFromXml(xml));
  const initial = mergeInitial(params.initial);
  const states: ODEStateSpec[] = [];
  for (const match of xml.matchAll(/<state\b([^>]*)>([\s\S]*?)<\/state>/g)) {
    const stateAttrs = parseAttrs(match[1]);
    const body = match[2];
    const name = requiredString(stateAttrs.name, 'MathEquationInput.xml.state.name');
    const rhsRaw = stateAttrs.derivative ?? stateAttrs.rhs ?? textBetween(body, 'derivative') ?? textBetween(body, 'rhs') ?? textBetween(body, 'equation');
    Preconditions.check('MathEquationInput', `xml.state.${name}.derivative`, 'be present', rhsRaw !== undefined, stateAttrs);
    const initialValue = numberOrDefault(stateAttrs.initial, initial[name] ?? 0, `MathEquationInput.xml.state.${name}.initial`);
    states.push({name, initial: initialValue, derivative: expressionText(rhsRaw ?? '')});
  }
  for (const match of xml.matchAll(/<equation\b[^>]*>([\s\S]*?)<\/equation>/g)) {
    const parsed = parseDerivativeEquation(xmlDecode(match[1]), constants, initial);
    if (parsed) states.push(parsed);
  }
  Preconditions.nonEmpty('MathEquationInput', 'xml.ode.states', states);
  const t0 = numberOrDefault(attrs.t0, params.t0 ?? 0, 'MathEquationInput.xml.t0');
  const t1 = numberOrDefault(attrs.t1, params.t1 ?? 1, 'MathEquationInput.xml.t1');
  const dt = numberOrDefault(attrs.dt, params.dt ?? defaultDt(t0, t1, 100), 'MathEquationInput.xml.dt');
  const method = methodOrDefault(attrs.method, params.method ?? 'euler');
  return {states, constants, t0, t1, dt, method};
}

function normalizeXmlHeat(params: MathEquationInputParams): Heat1DBlockParams {
  const xml = safeXml(params.equation ?? '');
  const attrs = rootAttrs(xml, 'heat1d');
  const constants = mergeConstants(params.constants, constantsFromXml(xml));
  const cells = integerOrDefault(attrs.cells, params.cells ?? 31, 'MathEquationInput.xml.cells');
  const length = numberOrDefault(attrs.length, params.length ?? 1, 'MathEquationInput.xml.length');
  const alpha = numberOrDefault(attrs.alpha, params.alpha ?? constants.alpha ?? 0.01, 'MathEquationInput.xml.alpha');
  const t0 = numberOrDefault(attrs.t0, params.t0 ?? 0, 'MathEquationInput.xml.t0');
  const t1 = numberOrDefault(attrs.t1, params.t1 ?? 1, 'MathEquationInput.xml.t1');
  const dt = numberOrDefault(attrs.dt, params.dt ?? stableHeatDt(t0, t1, cells, length, alpha), 'MathEquationInput.xml.dt');
  const initial = textBetween(xml, 'initial');
  return {
    cells,
    length,
    alpha,
    t0,
    t1,
    dt,
    constants,
    initialExpression: initial ? expressionText(xmlDecode(initial)) : params.initialExpression ?? 'sin(pi*x/length)',
    initialValues: numericArrayOrUndefined(params.initialValues, 'MathEquationInput.xml.initialValues'),
    leftBoundary: numberOrUndefined(attrs.leftBoundary ?? params.leftBoundary, 'MathEquationInput.xml.leftBoundary') ?? 0,
    rightBoundary: numberOrUndefined(attrs.rightBoundary ?? params.rightBoundary, 'MathEquationInput.xml.rightBoundary') ?? 0,
  };
}

function parseLatexODE(equation: string, initial: Record<string, number>, constants: Record<string, number>): ODEStateSpec[] {
  const states = new Map<string, ODEStateSpec>();
  const parsedInitials: Record<string, number> = {...initial};
  for (const statement of equationStatements(equation)) {
    const eq = statement.split('=');
    if (eq.length < 2) continue;
    const lhs = eq[0].trim();
    const rhs = eq.slice(1).join('=').trim();
    const initName = initialConditionName(lhs);
    if (initName) {
      parsedInitials[initName] = evaluate(parse(expressionText(rhs)), {...constants, ...parsedInitials});
      continue;
    }
    const parsed = parseDerivativeEquation(`${lhs}=${rhs}`, constants, parsedInitials);
    if (parsed) states.set(parsed.name, parsed);
  }
  Preconditions.nonEmpty('MathEquationInput', 'latex.ode.derivatives', Array.from(states.values()));
  return Array.from(states.values()).map(s => ({
    ...s,
    initial: parsedInitials[s.name] ?? s.initial,
  }));
}

function parseDerivativeEquation(equation: string, constants: Record<string, number>, initial: Record<string, number>): ODEStateSpec | undefined {
  const parts = equation.split('=');
  if (parts.length < 2) return undefined;
  const lhs = parts[0].trim();
  const rhs = parts.slice(1).join('=').trim();
  const name = derivativeStateName(lhs);
  if (!name) return undefined;
  const derivative = expressionText(rhs);
  parse(derivative);
  return {name, initial: initial[name] ?? 0, derivative};
}

function derivativeStateName(lhs: string): string | undefined {
  const compact = lhs.replace(/\s+/g, '');
  let match = compact.match(/^\\frac\{d([A-Za-z_][A-Za-z0-9_]*)\}\{dt\}$/);
  if (match) return match[1];
  match = compact.match(/^\\dot\{([A-Za-z_][A-Za-z0-9_]*)\}$/);
  if (match) return match[1];
  match = compact.match(/^([A-Za-z_][A-Za-z0-9_]*)'$/);
  if (match) return match[1];
  match = compact.match(/^d([A-Za-z_][A-Za-z0-9_]*)\/dt$/);
  if (match) return match[1];
  return undefined;
}

function initialConditionName(lhs: string): string | undefined {
  const compact = lhs.replace(/\s+/g, '');
  const match = compact.match(/^([A-Za-z_][A-Za-z0-9_]*)\((?:0|t0)\)$/);
  return match?.[1];
}

function expressionText(raw: string): string {
  const decoded = xmlDecode(raw).trim();
  if (looksLatex(decoded)) return latexToExpression(decoded);
  return insertImplicitMultiplication(rewriteMathAliases(decoded));
}

export function latexToExpression(input: string): string {
  let s = input.trim();
  s = s.replace(/\$\$/g, '').replace(/\$/g, '');
  s = s.replace(/\\left/g, '').replace(/\\right/g, '');
  s = s.replace(/\\,/g, '').replace(/&/g, '');
  s = s.replace(/\\begin\{[^}]+\}/g, '').replace(/\\end\{[^}]+\}/g, '');
  s = replaceFractions(s);
  for (const [from, to] of Object.entries(GREEK)) s = s.split(from).join(to);
  s = s.replace(/\\cdot|\\times/g, '*');
  s = s.replace(/\\ln/g, 'log');
  for (const fn of FUNCTIONS) s = s.replace(new RegExp(`\\\\${fn}\\b`, 'g'), fn === 'ln' ? 'log' : fn);
  s = s.replace(/\^\{([^{}]+)\}/g, '^($1)');
  s = s.replace(/_\{([A-Za-z0-9]+)\}/g, '_$1');
  s = s.replace(/[{}]/g, match => match === '{' ? '(' : ')');
  s = rewriteMathAliases(s);
  return insertImplicitMultiplication(s);
}

function replaceFractions(input: string): string {
  let s = input;
  let idx = s.indexOf('\\frac');
  while (idx >= 0) {
    const numerator = readBraced(s, idx + '\\frac'.length);
    if (!numerator) break;
    const denominator = readBraced(s, numerator.end);
    if (!denominator) break;
    const replacement = `((${replaceFractions(numerator.value)})/(${replaceFractions(denominator.value)}))`;
    s = s.slice(0, idx) + replacement + s.slice(denominator.end);
    idx = s.indexOf('\\frac');
  }
  return s;
}

function readBraced(s: string, start: number): {value: string; end: number} | undefined {
  let i = start;
  while (i < s.length && /\s/.test(s[i])) i++;
  if (s[i] !== '{') return undefined;
  let depth = 0;
  const begin = i + 1;
  for (; i < s.length; i++) {
    if (s[i] === '{') depth++;
    else if (s[i] === '}') {
      depth--;
      if (depth === 0) return {value: s.slice(begin, i), end: i + 1};
    }
  }
  return undefined;
}

function insertImplicitMultiplication(input: string): string {
  const tokens = tokenizeExpression(input);
  let out = '';
  for (let i = 0; i < tokens.length; i++) {
    const prev = tokens[i - 1];
    const cur = tokens[i];
    if (prev && needsMultiplication(prev, cur)) out += '*';
    out += cur.text;
  }
  return out;
}

interface ExprToken {
  kind: 'num' | 'id' | 'op' | 'lparen' | 'rparen';
  text: string;
}

function tokenizeExpression(input: string): ExprToken[] {
  const tokens: ExprToken[] = [];
  let i = 0;
  while (i < input.length) {
    const c = input[i];
    if (/\s/.test(c)) { i++; continue; }
    if (/[0-9.]/.test(c)) {
      let j = i + 1;
      while (j < input.length && /[0-9.]/.test(input[j])) j++;
      if (j < input.length && /[eE]/.test(input[j])) {
        j++;
        if (/[+-]/.test(input[j])) j++;
        while (j < input.length && /[0-9]/.test(input[j])) j++;
      }
      tokens.push({kind: 'num', text: input.slice(i, j)});
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i + 1;
      while (j < input.length && /[A-Za-z0-9_]/.test(input[j])) j++;
      tokens.push({kind: 'id', text: input.slice(i, j)});
      i = j;
      continue;
    }
    if (c === '(') { tokens.push({kind: 'lparen', text: c}); i++; continue; }
    if (c === ')') { tokens.push({kind: 'rparen', text: c}); i++; continue; }
    if ('+-*/^'.includes(c)) { tokens.push({kind: 'op', text: c}); i++; continue; }
    console.warn(`[math-equation-input] unsupported expression character "${c}" while tokenizing ${JSON.stringify(input)} — check for stray symbols or an unmapped LaTeX command.`);
    throw new Error(`MathEquationInput: unsupported expression character "${c}" in ${JSON.stringify(input)}`);
  }
  return tokens;
}

function needsMultiplication(a: ExprToken, b: ExprToken): boolean {
  const aEnds = a.kind === 'num' || a.kind === 'id' || a.kind === 'rparen';
  const bStarts = b.kind === 'num' || b.kind === 'id' || b.kind === 'lparen';
  if (!aEnds || !bStarts) return false;
  if (a.kind === 'id' && b.kind === 'lparen' && FUNCTIONS.has(a.text)) return false;
  return true;
}

function rewriteMathAliases(input: string): string {
  return input
    .replace(/\bln\b/g, 'log')
    .replace(/\bPI\b/g, 'pi')
    .replace(/\bPi\b/g, 'pi')
    .replace(/\beuler\b/g, 'e');
}

function equationStatements(equation: string): string[] {
  return equation
    .replace(/\\begin\{cases\}/g, '')
    .replace(/\\end\{cases\}/g, '')
    .split(/\\\\|\n|;/g)
    .map(s => s.replace(/\$/g, '').replace(/&/g, '').trim())
    .filter(Boolean);
}

function looksLatex(s: string): boolean {
  return /\\frac|\\dot|\\partial|\\alpha|\\beta|\\gamma|\\lambda|\\pi|\\sin|\\cos|\\exp|\^\{/.test(s);
}

function stateFromJson(raw: unknown, index: number, initial: Record<string, number>, constants: Record<string, number>): ODEStateSpec {
  const obj = objectRecord(raw);
  Preconditions.check('MathEquationInput', `states[${index}]`, 'be an object', obj !== undefined, raw);
  const name = requiredString(obj?.name, `MathEquationInput.states[${index}].name`);
  const derivativeRaw = obj?.derivative ?? obj?.rhs ?? obj?.equation;
  Preconditions.check('MathEquationInput', `states[${index}].derivative`, 'be present', derivativeRaw !== undefined, obj);
  const initialValue = numberOrDefault(obj?.initial, initial[name] ?? 0, `MathEquationInput.states[${index}].initial`);
  let derivative = expressionText(String(derivativeRaw));
  if (String(derivativeRaw).includes('=')) {
    const parsed = parseDerivativeEquation(String(derivativeRaw), constants, {[name]: initialValue});
    derivative = parsed?.derivative ?? expressionText(String(derivativeRaw).split('=').slice(1).join('='));
  }
  return {name, initial: initialValue, derivative};
}

function mergeConstants(...records: Array<Record<string, unknown> | undefined>): Record<string, number> {
  const out: Record<string, number> = {pi: Math.PI, e: Math.E};
  for (const record of records) {
    if (!record) continue;
    for (const [key, value] of Object.entries(record)) out[key] = numberOrDefault(value, out[key], `MathEquationInput.constants.${key}`);
  }
  return out;
}

function mergeInitial(...records: Array<Record<string, unknown> | undefined>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const record of records) {
    if (!record) continue;
    for (const [key, value] of Object.entries(record)) out[key] = numberOrDefault(value, 0, `MathEquationInput.initial.${key}`);
  }
  return out;
}

function constantsFromXml(xml: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const match of xml.matchAll(/<constant\b([^/>]*)(?:\/>|>[\s\S]*?<\/constant>)/g)) {
    const attrs = parseAttrs(match[1]);
    const name = requiredString(attrs.name, 'MathEquationInput.xml.constant.name');
    out[name] = numberOrDefault(attrs.value, 0, `MathEquationInput.xml.constant.${name}.value`);
  }
  return out;
}

function safeXml(xml: string): string {
  Preconditions.check('MathEquationInput', 'xml', 'not contain DOCTYPE or ENTITY declarations', !/<!DOCTYPE|<!ENTITY/i.test(xml), 'blocked XML declaration');
  return xml.trim();
}

function rootName(xml: string): string | undefined {
  return xml.match(/<([A-Za-z_][A-Za-z0-9_-]*)\b/)?.[1];
}

function rootAttrs(xml: string, fallbackRoot: string): Attrs {
  const root = rootName(xml) ?? fallbackRoot;
  const match = xml.match(new RegExp(`<${root}\\b([^>]*)>`));
  return match ? parseAttrs(match[1]) : {};
}

function parseAttrs(raw: string): Attrs {
  const attrs: Attrs = {};
  for (const match of raw.matchAll(/([A-Za-z_][A-Za-z0-9_:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) {
    attrs[match[1]] = xmlDecode(match[2] ?? match[3] ?? '');
  }
  return attrs;
}

function textBetween(raw: string, tag: string): string | undefined {
  const match = raw.match(new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}>`));
  return match?.[1]?.trim();
}

function xmlDecode(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function arrayValue(value: unknown, param: string): unknown[] {
  Preconditions.check('MathEquationInput', param, 'be a non-empty array', Array.isArray(value) && value.length > 0, value);
  return value as unknown[];
}

function requiredString(value: unknown, param: string): string {
  Preconditions.check('MathEquationInput', param, 'be a non-empty string', typeof value === 'string' && value.trim().length > 0, value);
  return String(value).trim();
}

function stringOrDefault(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : fallback;
}

function numberOrDefault(value: unknown, fallback: number, param: string): number {
  if (value === undefined || value === null || value === '') return fallback;
  const n = typeof value === 'number' ? value : Number(value);
  Preconditions.finite('MathEquationInput', param, n);
  return n;
}

function numberOrUndefined(value: unknown, param: string): number | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  return numberOrDefault(value, 0, param);
}

function integerOrDefault(value: unknown, fallback: number, param: string): number {
  const n = numberOrDefault(value, fallback, param);
  Preconditions.integer('MathEquationInput', param, n);
  return n;
}

function methodOrDefault(value: unknown, fallback: IntegratorMethod): IntegratorMethod {
  if (value === undefined || value === null || value === '') return fallback;
  Preconditions.check('MathEquationInput', 'method', 'be euler or trapezoid', value === 'euler' || value === 'trapezoid', value);
  return value as IntegratorMethod;
}

function numericArrayOrUndefined(value: unknown, param: string): number[] | undefined {
  if (value === undefined || value === null) return undefined;
  Preconditions.check('MathEquationInput', param, 'be an array', Array.isArray(value), value);
  const arr = value as unknown[];
  if (arr.length === 0) return undefined;
  return arr.map((v, i) => numberOrDefault(v, 0, `${param}[${i}]`));
}

function defaultDt(t0: number, t1: number, steps: number): number {
  Preconditions.check('MathEquationInput', 'time horizon', 'satisfy t1 > t0', t1 > t0, {t0, t1});
  return (t1 - t0) / steps;
}

function stableHeatDt(t0: number, t1: number, cells: number, length: number, alpha: number): number {
  Preconditions.integerInRange('MathEquationInput', 'cells', cells, 3, 100000);
  Preconditions.positive('MathEquationInput', 'length', length);
  Preconditions.nonNegative('MathEquationInput', 'alpha', alpha);
  if (alpha === 0) return defaultDt(t0, t1, 100);
  const dx = length / (cells - 1);
  const target = 0.45 * dx * dx / alpha;
  const steps = Math.max(1, Math.ceil((t1 - t0) / target));
  return (t1 - t0) / steps;
}
