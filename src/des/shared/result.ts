'use strict';

// =============================================================================
// RUST MIGRATION — target: src/des/shared/result.rs   (module des::shared::result)
// 1:1 file move. See the module doc below for the full Rust mapping.
// =============================================================================

// =============================================================================
// shared/result.ts — Rust-shaped Result<T, E> and Option<T>.
//
// These mirror Rust's `std::result::Result` and `std::option::Option` so that
// error/optional handling written in TypeScript translates 1:1 to Rust during
// the migration. Use these instead of `throw` for *expected*, recoverable
// outcomes (infeasible solve, singular matrix, parse failure, …). Reserve
// `throw` for programmer errors / invariant violations (which map to Rust
// `panic!`).
//
// Rust mapping:
//   Result<T, E>  ->  enum Result<T, E> { Ok(T), Err(E) }
//   Option<T>     ->  enum Option<T>    { Some(T), None }
//
// The tagged-union encoding (`kind: 'ok' | 'err'`) is deliberate: discriminated
// unions are the idiomatic TypeScript stand-in for Rust enums and the migration
// can pattern-match on `kind` exactly like a Rust `match`.
// =============================================================================

export interface Ok<T> {
  readonly kind: 'ok';
  readonly value: T;
}

export interface Err<E> {
  readonly kind: 'err';
  readonly error: E;
}

export type Result<T, E = string> = Ok<T> | Err<E>;

export const ok = <T>(value: T): Ok<T> => ({kind: 'ok', value});
export const err = <E>(error: E): Err<E> => ({kind: 'err', error});

export const isOk = <T, E>(r: Result<T, E>): r is Ok<T> => r.kind === 'ok';
export const isErr = <T, E>(r: Result<T, E>): r is Err<E> => r.kind === 'err';

/** Unwrap an Ok value or throw (maps to Rust `Result::unwrap`, which panics). */
export const unwrap = <T, E>(r: Result<T, E>): T => {
  if (r.kind === 'ok') return r.value;
  throw new Error(`called unwrap() on an Err value: ${String((r as Err<E>).error)}`);
};

/** Unwrap an Ok value or return a fallback (maps to Rust `Result::unwrap_or`). */
export const unwrapOr = <T, E>(r: Result<T, E>, fallback: T): T =>
  r.kind === 'ok' ? r.value : fallback;

/** Map the Ok value (maps to Rust `Result::map`). */
export const mapResult = <T, U, E>(r: Result<T, E>, fn: (v: T) => U): Result<U, E> =>
  r.kind === 'ok' ? ok(fn(r.value)) : r;

/** Map the Err value (maps to Rust `Result::map_err`). */
export const mapErr = <T, E, F>(r: Result<T, E>, fn: (e: E) => F): Result<T, F> =>
  r.kind === 'err' ? err(fn(r.error)) : r;

// -----------------------------------------------------------------------------
// Option<T>
// -----------------------------------------------------------------------------

export interface Some<T> {
  readonly kind: 'some';
  readonly value: T;
}

export interface None {
  readonly kind: 'none';
}

export type Option<T> = Some<T> | None;

export const some = <T>(value: T): Some<T> => ({kind: 'some', value});
export const NONE: None = {kind: 'none'};
export const none = <T>(): Option<T> => NONE;

export const isSome = <T>(o: Option<T>): o is Some<T> => o.kind === 'some';
export const isNone = <T>(o: Option<T>): o is None => o.kind === 'none';

/** Convert a possibly-undefined/null value into an Option. */
export const fromNullable = <T>(v: T | null | undefined): Option<T> =>
  v === null || v === undefined ? NONE : some(v);

/** Unwrap a Some value or return a fallback (maps to Rust `Option::unwrap_or`). */
export const optionOr = <T>(o: Option<T>, fallback: T): T =>
  o.kind === 'some' ? o.value : fallback;
