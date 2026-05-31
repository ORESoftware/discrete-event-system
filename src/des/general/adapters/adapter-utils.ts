// RUST MIGRATION: Target module `src/des/general/adapters/adapter_utils.rs`.
// RUST MIGRATION: Convert these shared adapter helpers to free functions around `DESModelSpec`/registration modules; use `serde` structs for any config/result records.
// RUST MIGRATION: Represent CSV/output paths as `PathBuf`, filesystem writes as `std::fs`/`std::io::Result`, and validation failures as `Result<_, ValidationError>`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/adapters/adapter-utils.rs
//   (module des::general::adapters::adapter_utils)
// 1:1 file move. Shared helpers for the JSON model-spec adapters (CSV, logging).
//
// Declarations → Rust:
//   export fn validationLine / csvCell / csvRow / jsonCsvCell / jsonCsvRow /
//             writeCsvLines / numberPair / optionalNumberPair / defaultFramesPath /
//             framesPath                       -> plain `pub fn` (genuinely stateless
//             utilities — these stay free functions, NOT transform classes)
//   export async fn withLogger<T>             -> `pub async fn` (or generic fn taking
//             a closure) returning the closure's result
//
// Conversion notes (file-specific):
//   - `fs.writeFileSync` -> std::fs::write; `JsonlLogger` -> the ported logger type.
//   - `JSON.stringify(v)` in jsonCsvCell -> serde_json::to_string.
//   - CSV quoting uses a regex `/[",\n]/` -> the `regex` crate or a manual scan.
//   - `numberPair`/`optionalNumberPair` throw on length != 2 -> return Result or a
//     fixed-size `[f64; 2]`; the `throw` is an invariant (panic candidate).
//   - `runtime.outputs?.log` / `?? fallback` -> Option chaining + unwrap_or.
//   - withLogger is generic over T and may return `T | Promise<T>` -> a single
//     async fn; the try/finally that closes the logger -> RAII Drop or explicit close.
// =============================================================================

import * as fs from 'fs';
import {JsonlLogger} from '../../observability/logger';
import {DESRuntimeConfig} from '../des-spec';

export function validationLine(checks: readonly {passed: boolean}[]): string {
  const pass = checks.filter(c => c.passed).length;
  return `${pass}/${checks.length} checks passed`;
}

export function csvCell(v: unknown): string {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function csvRow(values: readonly unknown[]): string {
  return values.map(csvCell).join(',');
}

export function jsonCsvCell(v: unknown): string {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function jsonCsvRow(values: readonly unknown[]): string {
  return values.map(jsonCsvCell).join(',');
}

export function writeCsvLines(csvPath: string, lines: readonly string[]): void {
  fs.writeFileSync(csvPath, lines.join('\n'));
}

export function numberPair(
  values: readonly number[] | undefined,
  fallback: readonly [number, number],
  name: string,
): [number, number] {
  const pair = values ?? fallback;
  if (pair.length !== 2) throw new Error(`${name} must have length 2`);
  return [pair[0], pair[1]];
}

export function optionalNumberPair(values: readonly number[] | undefined, name: string): [number, number] | undefined {
  if (!values) return undefined;
  if (values.length !== 2) throw new Error(`${name} must have length 2`);
  return [values[0], values[1]];
}

export async function withLogger<T>(
  runtime: DESRuntimeConfig,
  fn: (logger?: JsonlLogger) => T | Promise<T>,
): Promise<T> {
  const logPath = runtime.outputs?.log;
  const logger = logPath ? new JsonlLogger(logPath, 'debug') : undefined;
  try {
    return await fn(logger);
  } finally {
    if (logger) await logger.close();
  }
}

export function defaultFramesPath(htmlPath: string): string {
  return htmlPath.endsWith('.html') ? htmlPath.replace(/\.html$/, '.frames.jsonl') : `${htmlPath}.frames.jsonl`;
}

export function framesPath(runtime: DESRuntimeConfig, model: string): {htmlPath?: string; frames: string} {
  const out = runtime.outputs ?? {};
  const htmlPath = out.html;
  const frames = out.frames ?? (htmlPath ? defaultFramesPath(htmlPath) : `out/${model}.frames.jsonl`);
  return {htmlPath, frames};
}
