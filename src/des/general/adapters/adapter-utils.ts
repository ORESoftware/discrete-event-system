'use strict';

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
