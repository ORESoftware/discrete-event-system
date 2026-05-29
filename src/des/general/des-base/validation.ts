'use strict';

// =============================================================================
// general/des-base/validation.ts — VALIDATOR PROTOCOL for DES stations.
//
// Every iterative-algorithm base in `des-base/` exposes `addValidator(v)`,
// `runValidation()`, and `validationReport()`. `runIterativeDES` calls
// `onFinalize()` and then `runValidation()` on every station after the loop
// terminates and aggregates the resulting checks into its summary, so
// algorithms can ship with intrinsic invariants AND external-reference
// comparisons attached to the model itself, not bolted on by the runner.
//
// CHECK SHAPE
// ───────────
//   { name, passed, observed?, expected?, group?, details? }
//
// FACTORY HELPERS (used by leaf classes when self-registering):
//   numericValidator       — compare a scalar against a known value (abs/rel tol)
//   boundValidator         — assert value ∈ [low, high]
//   monotonicityValidator  — assert series is non-increasing / non-decreasing
//   groundTruthValidator   — generic compare-against-reference with custom equality
//   externalReferenceValidator — read JSON from disk + user-supplied compare
//   intrinsicCheck         — convert a () → boolean predicate into a Validator
//
// All factories return `Validator<S>` where `S` is the station type. Multiple
// validators can be registered per station; failures are reported individually
// with `passed: false` (validators that throw are captured similarly).
// =============================================================================

import * as fs from 'fs';

/** A single pass/fail check produced by a Validator. */
export interface ValidationCheck {
  /** Human-readable check name (printed in reports). */
  readonly name: string;
  /** True if the check passed. */
  readonly passed: boolean;
  /** Stringified observed value (printed when failing). */
  readonly observed?: string;
  /** Stringified expected value or constraint. */
  readonly expected?: string;
  /** Optional grouping label for report formatting. */
  readonly group?: string;
  /** Optional extra details (printed when failing). */
  readonly details?: string;
}

/** A pluggable validator for a DES station. The runner invokes `validate`
 *  after the algorithm terminates; the returned checks are aggregated into
 *  the run summary. */
export interface Validator<S = unknown> {
  readonly name: string;
  validate(station: S): ValidationCheck[];
}

/** Run a list of validators and capture exceptions as failed checks so
 *  one buggy validator never blocks the rest. */
export function runValidators<S>(
  station: S, validators: ReadonlyArray<Validator<S>>,
): ValidationCheck[] {
  const out: ValidationCheck[] = [];
  for (const v of validators) {
    try {
      out.push(...v.validate(station));
    } catch (e) {
      console.warn(`[validation] validator "${v.name}" threw during validate(): ${e instanceof Error ? e.message : String(e)} — recording as a failed check.`);
      out.push({
        name: v.name + '/threw',
        passed: false,
        details: e instanceof Error ? e.message : String(e),
      });
    }
  }
  return out;
}

/** Format a ValidationCheck list as a multi-line report. */
export function formatValidationReport(checks: ReadonlyArray<ValidationCheck>): string {
  if (checks.length === 0) return '(no validators registered)';
  const lines: string[] = [];
  let pass = 0, fail = 0;
  let curGroup: string | undefined = '';
  for (const c of checks) {
    const g = c.group ?? '';
    if (g !== curGroup) {
      if (g) lines.push(`  ─── ${g} ───`);
      curGroup = g;
    }
    const tag = c.passed ? 'PASS' : 'FAIL';
    const obsPart = c.observed !== undefined ? `  observed=${c.observed}` : '';
    const expPart = c.expected !== undefined ? `  expected=${c.expected}` : '';
    const detPart = c.passed || !c.details ? '' : `  (${c.details})`;
    lines.push(`  ${tag}  ${c.name}${obsPart}${expPart}${detPart}`);
    if (c.passed) pass += 1; else fail += 1;
  }
  lines.push(`  ${'-'.repeat(64)}`);
  lines.push(`  ${pass} passed, ${fail} failed`);
  return lines.join('\n');
}

// =============================================================================
// FACTORIES
// =============================================================================

/** Wrap an arbitrary `(station) → boolean` predicate into a Validator. */
export function intrinsicCheck<S>(opts: {
  name: string;
  predicate: (station: S) => boolean;
  expected?: string;
  observedFn?: (station: S) => string;
  group?: string;
  details?: string;
}): Validator<S> {
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      const passed = opts.predicate(s);
      return [{
        name: opts.name,
        passed,
        observed: opts.observedFn ? opts.observedFn(s) : undefined,
        expected: opts.expected,
        group: opts.group,
        details: passed ? undefined : opts.details,
      }];
    },
  };
}

/** Compare a scalar extracted from the station against a known reference.
 *  `mode='absolute'` checks |obs − exp| ≤ tol; `mode='relative'` checks
 *  |obs − exp| / max(1e-12, |exp|) ≤ tol. */
export function numericValidator<S>(opts: {
  name: string;
  extract: (s: S) => number;
  expected: number | ((s: S) => number);
  tol?: number;
  mode?: 'absolute' | 'relative';
  group?: string;
}): Validator<S> {
  const tol = opts.tol ?? 1e-9;
  const mode = opts.mode ?? 'absolute';
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      const obs = opts.extract(s);
      const exp = typeof opts.expected === 'function'
        ? (opts.expected as (s: S) => number)(s) : opts.expected;
      if (!Number.isFinite(obs) || !Number.isFinite(exp)) {
        return [{
          name: opts.name, passed: false,
          observed: String(obs), expected: String(exp),
          group: opts.group, details: 'non-finite value',
        }];
      }
      const diff = Math.abs(obs - exp);
      const denom = mode === 'relative' ? Math.max(1e-12, Math.abs(exp)) : 1;
      const err = diff / denom;
      const passed = err <= tol;
      return [{
        name: opts.name, passed,
        observed: obs.toPrecision(8), expected: exp.toPrecision(8),
        group: opts.group,
        details: passed ? undefined : `${mode}-err=${err.toExponential(2)} > tol=${tol}`,
      }];
    },
  };
}

/** Assert a numeric extract is in [low, high] (closed by default). */
export function boundValidator<S>(opts: {
  name: string;
  extract: (s: S) => number;
  low?: number;
  high?: number;
  inclusive?: boolean;
  group?: string;
}): Validator<S> {
  const lo = opts.low ?? -Infinity;
  const hi = opts.high ?? Infinity;
  const inc = opts.inclusive ?? true;
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      const v = opts.extract(s);
      const inLo = inc ? v >= lo : v > lo;
      const inHi = inc ? v <= hi : v < hi;
      const passed = inLo && inHi;
      return [{
        name: opts.name, passed,
        observed: String(v),
        expected: `${inc ? '[' : '('}${lo}, ${hi}${inc ? ']' : ')'}`,
        group: opts.group,
        details: passed ? undefined :
          `value ${v} outside ${inc ? 'closed' : 'open'} interval [${lo}, ${hi}]`,
      }];
    },
  };
}

/** Assert that a sequence is monotone in `direction`. */
export function monotonicityValidator<S>(opts: {
  name: string;
  extract: (s: S) => readonly number[];
  direction: 'non-increasing' | 'non-decreasing';
  tol?: number;
  group?: string;
}): Validator<S> {
  const tol = opts.tol ?? 1e-12;
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      const xs = opts.extract(s);
      let firstViolation = -1;
      for (let i = 1; i < xs.length; i++) {
        const d = xs[i] - xs[i - 1];
        const ok = opts.direction === 'non-increasing' ? d <= tol : d >= -tol;
        if (!ok) { firstViolation = i; break; }
      }
      const passed = firstViolation === -1;
      return [{
        name: opts.name, passed,
        observed: passed ? `${opts.direction} (n=${xs.length})`
                         : `breaks at i=${firstViolation}`,
        expected: opts.direction,
        group: opts.group,
        details: passed ? undefined :
          `xs[${firstViolation - 1}]=${xs[firstViolation - 1]}  xs[${firstViolation}]=${xs[firstViolation]}`,
      }];
    },
  };
}

/** Custom-equality version of `numericValidator` for non-numeric quantities
 *  (vectors, structs). The user supplies the comparator that returns a
 *  failure-string-or-null. */
export function groundTruthValidator<S, T>(opts: {
  name: string;
  extract: (s: S) => T;
  expected: T | ((s: S) => T);
  compare: (observed: T, expected: T) => string | null;
  format?: (v: T) => string;
  group?: string;
}): Validator<S> {
  const fmt = opts.format ?? ((x: T) => String(x));
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      const obs = opts.extract(s);
      const exp = typeof opts.expected === 'function'
        ? (opts.expected as (s: S) => T)(s) : opts.expected;
      const failure = opts.compare(obs, exp);
      const passed = failure === null;
      return [{
        name: opts.name, passed,
        observed: fmt(obs), expected: fmt(exp),
        group: opts.group,
        details: passed ? undefined : (failure ?? undefined),
      }];
    },
  };
}

/** External-reference validator: load a JSON file from disk and run a
 *  user-supplied comparator. Used to plug Python/scipy/SimPy references
 *  directly into the algorithm's lifecycle.
 *
 *  When the reference file is absent the validator reports a single
 *  failed check (`<name>/reference-missing`) — but the algorithm still
 *  ran, so the rest of the run summary is unaffected. Pass
 *  `silentIfMissing: true` to skip the failure entirely (useful for
 *  optional references that may or may not exist). */
export function externalReferenceValidator<S>(opts: {
  name: string;
  referencePath: string;
  compare: (station: S, ref: any) => ValidationCheck[];
  silentIfMissing?: boolean;
  group?: string;
}): Validator<S> {
  return {
    name: opts.name,
    validate(s: S): ValidationCheck[] {
      if (!fs.existsSync(opts.referencePath)) {
        if (opts.silentIfMissing) return [];
        console.warn(`[validation] external reference file not found for "${opts.name}": ${opts.referencePath} — comparison check will fail. Generate the reference or set silentIfMissing.`);
        return [{
          name: opts.name + '/reference-missing',
          passed: false,
          observed: 'absent', expected: 'present',
          group: opts.group,
          details: `reference file ${opts.referencePath} not found`,
        }];
      }
      const ref = JSON.parse(fs.readFileSync(opts.referencePath, 'utf8'));
      const checks = opts.compare(s, ref);
      // Tag with the validator's group if not already set.
      return opts.group
        ? checks.map(c => c.group === undefined ? {...c, group: opts.group} : c)
        : checks;
    },
  };
}
