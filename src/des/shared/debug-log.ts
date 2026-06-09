'use strict';

// RUST MIGRATION:
// - Target: src/des/shared/debug_log.rs
// - Maps to the `log`/`tracing` crates: `debugLog(() => msg)` -> `log::debug!(...)`
//   (the macro is already lazy, so the thunk is unnecessary in Rust).
// - DES_DEBUG env gate -> `RUST_LOG=debug` or a `tracing_subscriber` filter.

// =============================================================================
// shared/debug-log.ts — opt-in debug logging for HOT engine paths.
//
// Several engine stations (sources, processors, sinks, decisions) emit a
// per-entity / per-step debug line. In Node `console.debug` aliases
// `console.log`, so left unguarded these flood stdout and measurably slow a
// large simulation (millions of entities → millions of lines, plus the cost of
// building each interpolated string).
//
// `debugLog` takes a THUNK so the message is only constructed — and printed —
// when debugging is enabled. It is OFF by default and turned on with the
// `DES_DEBUG` environment variable ("1" / "true" / "yes" / "on").
// =============================================================================

const TRUTHY = new Set(['1', 'true', 'yes', 'on']);

function readFlag(): boolean {
  const v = (typeof process !== 'undefined' && process.env && process.env.DES_DEBUG) || '';
  return TRUTHY.has(String(v).toLowerCase());
}

let enabled = readFlag();

/** Whether hot-path debug logging is currently enabled (via `DES_DEBUG`). */
export function debugEnabled(): boolean {
  return enabled;
}

/** Override the debug flag at runtime (e.g. from a test or a CLI flag). */
export function setDebugEnabled(on: boolean): void {
  enabled = on;
}

/**
 * Gated debug log for hot paths. The `make` thunk is invoked only when
 * debugging is enabled, so neither the string interpolation nor the write
 * happens on the default (disabled) path.
 */
export function debugLog(make: () => string): void {
  if (enabled) {
    console.debug(make());
  }
}
