'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des-spec.rs  (module des::general::des_spec)
// 1:1 file move. JSON spec envelope + tiny declarative param validator for runnable models.
//
// Declarations → Rust:
//   interface DESModelSpec/DESRuntimeConfig/DESModelMetadata/ValidationResult/
//             DESRunSummary           -> structs (#[derive(Serialize, Deserialize)])
//   type ParamSchema = {kind:...}|... -> enum ParamSchema { Number{..}, String{..}, ... } (match on kind)
//   interface DESModelRegistration<P,R> -> trait (run/summarize/animate/writeCsv methods)
//   fn validate / validateInner / typeOf -> assoc fns; validateInner matches ParamSchema
//
// Conversion notes (file-specific):
//   - `ParamSchema` is a discriminated union (`kind`) -> Rust enum + `match`.
//   - `unknown` / `Record<string, unknown>` values -> `serde_json::Value`.
//   - `zod` (`ZodType<P>`) + `$schema` literal -> serde/validator; literal type -> const/enum.
//   - `run: (params, runtime) => R | Promise<R>` -> sync fn or async (return a Future);
//     registration callbacks (run/summarize/animate/writeCsv) -> trait methods, not field closures.
// =============================================================================

import type {ZodType} from 'zod';

// =============================================================================
// general/des-spec.ts — JSON specification format for runnable DES models.
//
// Purpose
// -------
// A DES model in this codebase is a topology of stations + movables. Two
// pieces define a runnable program:
//
//   1. The TOPOLOGY  (which stations exist, how they're connected, what
//                     movables flow). This is fixed by a "model class" —
//                     a chunk of code that you import.
//   2. The PARAMETERS (numeric configuration: τ, K_p, T_target, durations,
//                      seeds, …). These are what you typically vary.
//
// JSON is GREAT for (2) and BAD for (1). So the spec format we use is a
// thin envelope:
//
//   {
//     "$schema": "des/model-spec/v1",
//     "model": "temp-control",            // <-- a registered model id
//     "description": "winter day, ±2°F", // <-- free-form
//     "parameters": { ... },              // <-- model-specific
//     "runtime": {                        // <-- common: how to run
//       "seed": 42,
//       "outputs": {
//         "csv": "out/foo.csv",
//         "html": "out/foo.html",
//         "log": "out/foo.jsonl"
//       }
//     },
//     "metadata": {                       // <-- bookkeeping
//       "author": "Alice",
//       "createdAt": "2026-05-25T20:00:00Z",
//       "tags": ["winter", "demo"]
//     }
//   }
//
// CODE OR JSON?
// -------------
// Both work. The same envelope is a valid TypeScript object literal:
//
//   const spec: DESModelSpec<TempControlParams> = {
//     $schema: 'des/model-spec/v1',
//     model: 'temp-control',
//     parameters: { T_target: 70, ... },
//     runtime: { seed: 42 },
//   };
//   const result = runFromSpec(spec);
//
// Users who want WHOLLY new behaviour write a model class in TypeScript
// and register it under a new id; users who want a new VARIANT of an
// existing model just write a new JSON.
//
// TWO-WAY SERIALIZATION
// ---------------------
// runFromSpec() returns a result that the model knows how to summarise
// and (optionally) animate. The on-disk JSON spec is the EXACT input
// that produced any saved output, so re-running the same spec
// reproduces the same result (subject to the seed).
// =============================================================================

/** Top-level envelope. P is the model-specific parameter type. */
export interface DESModelSpec<P = Record<string, unknown>> {
  /** Spec format version. Must equal "des/model-spec/v1". */
  $schema: 'des/model-spec/v1';
  /** Registered model id (looked up in DESModelRegistry). */
  model: string;
  /** Optional human-readable description. */
  description?: string;
  /** Model-specific parameters. Validated against the registered schema. */
  parameters: P;
  /** Optional runtime/execution settings. */
  runtime?: DESRuntimeConfig;
  /** Optional bookkeeping. */
  metadata?: DESModelMetadata;
}

export interface DESRuntimeConfig {
  /** Deterministic random seed for the run. */
  seed?: number;
  /** If false, suppress animation even when the model supports it. Defaults to true. */
  animate?: boolean;
  /** Where to write outputs. */
  outputs?: {
    /** CSV trace path. */
    csv?: string;
    /** HTML animation path. If set, the model's animator is invoked. */
    html?: string;
    /** JSONL frames file path (defaults to html with .frames.jsonl extension). */
    frames?: string;
    /** JSON summary path. */
    summary?: string;
    /** JSONL observability log path. */
    log?: string;
  };
  /** If false, suppress informational console output. Defaults to true. */
  verbose?: boolean;
}

export interface DESModelMetadata {
  author?: string;
  /** ISO 8601 timestamp. */
  createdAt?: string;
  /** Free-form tags. */
  tags?: string[];
  /** Free-form notes. */
  notes?: string;
}

// -----------------------------------------------------------------------------
// Parameter schema — a tiny declarative validator that the registry uses to
// type-check the JSON params before the model runs. Keeps deps zero.
// -----------------------------------------------------------------------------

export type ParamSchema =
  | {kind: 'number'; min?: number; max?: number; integer?: boolean; default?: number; description?: string}
  | {kind: 'string'; enum?: string[]; default?: string; description?: string}
  | {kind: 'boolean'; default?: boolean; description?: string}
  | {kind: 'array'; items: ParamSchema; minLength?: number; maxLength?: number; description?: string}
  | {kind: 'object'; fields: Record<string, ParamSchema>; required?: string[]; description?: string}
  | {kind: 'oneOf'; variants: Array<{tag: string; tagField?: string; schema: ParamSchema; description?: string}>; description?: string};

/** Result of validating params against a schema. */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  /** Validated parameters with defaults filled in. Only present if valid. */
  value?: unknown;
}

export function validate(value: unknown, schema: ParamSchema, path = '$'): ValidationResult {
  const errors: string[] = [];
  const v = validateInner(value, schema, path, errors);
  return {valid: errors.length === 0, errors, value: errors.length === 0 ? v : undefined};
}

function validateInner(value: unknown, schema: ParamSchema, path: string, errors: string[]): unknown {
  switch (schema.kind) {
    case 'number': {
      const v = (value === undefined || value === null) ? schema.default : value;
      if (typeof v !== 'number' || Number.isNaN(v)) { errors.push(`${path}: expected number, got ${typeOf(value)}`); return v; }
      if (schema.integer && !Number.isInteger(v))   errors.push(`${path}: expected integer, got ${v}`);
      if (schema.min !== undefined && v < schema.min) errors.push(`${path}: ${v} < min ${schema.min}`);
      if (schema.max !== undefined && v > schema.max) errors.push(`${path}: ${v} > max ${schema.max}`);
      return v;
    }
    case 'string': {
      const v = (value === undefined || value === null) ? schema.default : value;
      if (typeof v !== 'string') { errors.push(`${path}: expected string, got ${typeOf(value)}`); return v; }
      if (schema.enum && !schema.enum.includes(v)) errors.push(`${path}: ${JSON.stringify(v)} not in [${schema.enum.map(s => JSON.stringify(s)).join(', ')}]`);
      return v;
    }
    case 'boolean': {
      const v = (value === undefined || value === null) ? schema.default : value;
      if (typeof v !== 'boolean') { errors.push(`${path}: expected boolean, got ${typeOf(value)}`); return v; }
      return v;
    }
    case 'array': {
      if (value === undefined || value === null) {
        errors.push(`${path}: required array missing`);
        return [];
      }
      if (!Array.isArray(value)) { errors.push(`${path}: expected array, got ${typeOf(value)}`); return value; }
      if (schema.minLength !== undefined && value.length < schema.minLength) errors.push(`${path}: length ${value.length} < ${schema.minLength}`);
      if (schema.maxLength !== undefined && value.length > schema.maxLength) errors.push(`${path}: length ${value.length} > ${schema.maxLength}`);
      return value.map((item, i) => validateInner(item, schema.items, `${path}[${i}]`, errors));
    }
    case 'object': {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        errors.push(`${path}: expected object, got ${typeOf(value)}`); return value;
      }
      const obj = value as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      const required = schema.required ?? Object.keys(schema.fields);
      for (const [key, sub] of Object.entries(schema.fields)) {
        if (!(key in obj) && !required.includes(key)) {
          // missing optional, set default if any
          out[key] = validateInner(undefined, sub, `${path}.${key}`, []);
          continue;
        }
        if (!(key in obj) && required.includes(key)) {
          // Try default; otherwise error.
          const probeErrs: string[] = [];
          const probed = validateInner(undefined, sub, `${path}.${key}`, probeErrs);
          if (probeErrs.length === 0) { out[key] = probed; continue; }
          errors.push(`${path}: missing required field ".${key}"`);
          continue;
        }
        out[key] = validateInner(obj[key], sub, `${path}.${key}`, errors);
      }
      // Allow unknown fields but warn
      for (const key of Object.keys(obj)) {
        if (!(key in schema.fields)) out[key] = obj[key];
      }
      return out;
    }
    case 'oneOf': {
      if (typeof value !== 'object' || value === null) {
        errors.push(`${path}: expected one of (object), got ${typeOf(value)}`); return value;
      }
      const obj = value as Record<string, unknown>;
      const tagField = schema.variants[0].tagField ?? 'kind';
      const tag = obj[tagField];
      const variant = schema.variants.find(v => v.tag === tag);
      if (!variant) {
        errors.push(`${path}: ${tagField} ${JSON.stringify(tag)} not in [${schema.variants.map(v => JSON.stringify(v.tag)).join(', ')}]`);
        return value;
      }
      return validateInner(value, variant.schema, `${path}<${variant.tag}>`, errors);
    }
  }
}

function typeOf(v: unknown): string {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v;
}

// -----------------------------------------------------------------------------
// Model registration interface — what an adapter must provide to be usable
// from JSON.
// -----------------------------------------------------------------------------

export interface DESModelRegistration<P, R> {
  /** Stable id used in JSON's "model" field. */
  id: string;
  /** One-line summary. */
  description: string;
  /** Schema for validating the parameters object. */
  schema: ParamSchema;
  /** Optional stricter runtime schema. When present, runFromSpec() uses this
   *  first so models can attach richer JSON validation while the lightweight
   *  ParamSchema remains available for CLI introspection and older adapters. */
  zodSchema?: ZodType<P>;
  /** Run the model. */
  run: (params: P, runtime: DESRuntimeConfig) => R | Promise<R>;
  /** Render a one-page human-readable summary of the result. */
  summarize: (result: R, params: P) => string;
  /** Optional animation hook. Receives the result and writes outputs. */
  animate?: (result: R, params: P, runtime: DESRuntimeConfig) => Promise<void>;
  /** Optional CSV writer. */
  writeCsv?: (result: R, csvPath: string) => void;
  /** Optional examples (each is a complete spec the user can copy). */
  examples?: Array<{name: string; spec: DESModelSpec<P>}>;
}

// -----------------------------------------------------------------------------
// Result type returned by runFromSpec.
// -----------------------------------------------------------------------------

export interface DESRunSummary {
  modelId: string;
  /** The params that were actually used (after defaults filled in). */
  params: unknown;
  /** Wall-clock run time in ms. */
  runtimeMs: number;
  /** Model-specific result (whatever .run returned). */
  result: unknown;
  /** Human-readable summary lines. */
  summaryText: string;
  /** Files that were written. */
  outputs: Array<{kind: 'csv' | 'html' | 'frames' | 'summary' | 'log'; path: string}>;
}
