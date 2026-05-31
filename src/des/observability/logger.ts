'use strict';

// RUST MIGRATION:
// - Target: src/des/observability/logger.rs
// - LogLevel becomes an enum with Ord; BaseEvent becomes a trait or enum-backed
//   event struct serialized with serde_json.
// - JsonlLogger maps to a struct owning a BufWriter<File>, min-level, counters,
//   and HashMap<String, usize>; close should be Drop/flush plus explicit Result.
// - readEvents is a pure IO transform from path -> Vec<Event>; replace
//   Record<string, any>, synchronous Node fs calls, and thrown parse errors with
//   serde_json::Value/typed events and anyhow/thiserror Results.

// =============================================================================
// RUST MIGRATION  —  target: src/des/observability/logger.rs  (module des::observability::logger)
// 1:1 file move. Append-only JSONL event logger + reader.
//
// Declarations → Rust:
//   type LogLevel = 'trace'|'debug'|'info'|'warn'|'error'
//                              -> enum LogLevel (#[serde(rename_all="lowercase")]; derive PartialOrd)
//   const LEVEL_ORDER          -> rank via `as u8` on the enum / a match fn
//   interface BaseEvent        -> struct BaseEvent { kind, level: Option<LogLevel>, t: Option<f64> }
//   class JsonlLogger          -> struct + impl
//   function readEvents        -> free fn `fn read_events(path) -> Result<Vec<Value>, E>`
//
// Conversion notes (file-specific):
//   - `fs.createWriteStream(.., {flags:'w'})` + `mkdirSync` + sync `.write` ->
//     `std::fs::File`/`BufWriter` (truncating); `create_dir_all`. No streams.
//   - `BaseEvent & Record<string, any>` is an OPEN shape -> `serde_json::Value`
//     or a struct with `#[serde(flatten)] extra: Map<String, Value>`.
//   - `JSON.stringify` / `JSON.parse` -> `serde_json`.
//   - `byKind: Map<string, number>` -> `HashMap<String, u64>`; `getKindCounts` -> clone.
//   - `close(): Promise<void>` -> `Drop`/explicit `flush()`; no async needed.
//   - `readEvents` throws on malformed line -> return `Result` with line number.
// =============================================================================

// =============================================================================
// JSONL file logger for the DES simulator.
//
// Design goals (matched to the user's request "just logging tools, this is a
// CLI program, no telemetry"):
//   - Append-only line-delimited JSON so the file can be tail'd, jq'd, or
//     replayed by an offline validator.
//   - Cheap when filtered out (level check before stringify).
//   - Synchronous .write so we don't have to await every event.
//   - One-shot file path; run-to-run files don't get appended on top of
//     each other (we open with 'w', truncating).
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';

export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error';

const LEVEL_ORDER: Record<LogLevel, number> = {
  trace: 0, debug: 1, info: 2, warn: 3, error: 4,
};

export interface BaseEvent {
  kind: string;
  level?: LogLevel;
  t?: number;
}

export class JsonlLogger {
  private stream: fs.WriteStream;
  private minLevel: number;
  private filePath: string;
  private eventCount = 0;
  private byKind = new Map<string, number>();

  constructor(filePath: string, minLevel: LogLevel = 'info') {
    this.filePath = filePath;
    fs.mkdirSync(path.dirname(filePath), {recursive: true});
    this.stream = fs.createWriteStream(filePath, {flags: 'w'});
    this.minLevel = LEVEL_ORDER[minLevel];
  }

  log(event: BaseEvent & Record<string, any>): void {
    const level = event.level ?? 'info';
    if (LEVEL_ORDER[level] < this.minLevel) return;
    this.stream.write(JSON.stringify({level, ...event}) + '\n');
    this.eventCount++;
    this.byKind.set(event.kind, (this.byKind.get(event.kind) ?? 0) + 1);
  }

  getEventCount(): number {
    return this.eventCount;
  }

  getKindCounts(): Record<string, number> {
    return Object.fromEntries(this.byKind.entries());
  }

  getFilePath(): string {
    return this.filePath;
  }

  close(): Promise<void> {
    return new Promise<void>(resolve => this.stream.end(() => resolve()));
  }
}

// Convenience reader for offline validators / comparators.
export function readEvents(filePath: string): Array<Record<string, any>> {
  const raw = fs.readFileSync(filePath, 'utf8');
  const lines = raw.split('\n');
  const events: Array<Record<string, any>> = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    try {
      events.push(JSON.parse(line));
    } catch (err) {
      throw new Error(`malformed JSONL at line ${i + 1} of ${filePath}: ${(err as Error).message}`);
    }
  }
  return events;
}
