'use strict';

// =============================================================================
// RUST MIGRATION — target: src/des/shared/transform.rs   (module des::shared::transform)
// 1:1 file move. See the module doc below for the full Rust mapping.
// =============================================================================

// =============================================================================
// shared/transform.ts — the core "function as a type" abstraction.
//
// MIGRATION CONTRACT
// ------------------
// Rust has no classes, only structs + traits. To make the TypeScript -> Rust
// migration mechanical, every meaningful unit of behaviour is modeled as a
// class implementing a small interface (the "trait"). A plain/vanilla function
// `f(input) -> output` becomes a `PureTransform` subclass with a `transform`
// method instead of a free `function`.
//
//   TypeScript                              Rust
//   ----------------------------------      ------------------------------------
//   interface Transform<I, O>          ->   trait Transform<I, O>
//   abstract class PureTransform<I,O>  ->   struct + impl Transform for ...
//   transform(input: I): O             ->   fn transform(&self, input: I) -> O
//
// CONVENTIONS
//   * One input, one output. If a function takes several arguments, bundle them
//     into a single `Input` interface (named fields => Rust struct fields). This
//     avoids positional-argument ambiguity that does not survive translation.
//   * Configuration/parameters live as constructor fields (`readonly`), exactly
//     like fields on a Rust struct. `transform` then reads `this.<field>`.
//   * `PureTransform` must be pure: deterministic, no I/O, no global mutable
//     state, no `Math.random()`/`Date.now()`. Inject those via capability ports
//     (see `shared/capabilities.ts`) so behaviour stays reproducible and the
//     Rust impl can take the same dependencies.
//   * Use `StatefulTransform` when the transform carries mutable internal state
//     across calls (maps to a `&mut self` method in Rust).
//   * Use `FallibleTransform` when failure is an expected, recoverable outcome;
//     it returns `Result<O, E>` instead of throwing (see `shared/result.ts`).
// =============================================================================

import {Result} from './result';

/** The fundamental trait: turn an `I` into an `O`. */
export interface Transform<I, O> {
  transform(input: I): O;
}

/** A transform whose failures are values, not exceptions (Rust `Result`). */
export interface FallibleTransformTrait<I, O, E = string> {
  transform(input: I): Result<O, E>;
}

/**
 * Base class for pure, deterministic, side-effect-free functions modeled as a
 * type. Subclasses implement only `transform`. Configuration belongs in the
 * constructor as `readonly` fields.
 */
export abstract class PureTransform<I, O> implements Transform<I, O> {
  abstract transform(input: I): O;
}

/**
 * Base class for a transform that mutates internal state across invocations
 * (running accumulators, iterative solvers stepping in place, RNG-backed
 * samplers, …). Equivalent to a Rust method taking `&mut self`.
 */
export abstract class StatefulTransform<I, O> implements Transform<I, O> {
  abstract transform(input: I): O;
}

/**
 * Base class for transforms whose failure is an expected outcome. Returns a
 * `Result<O, E>` rather than throwing, matching Rust's fallible functions.
 */
export abstract class FallibleTransform<I, O, E = string>
  implements FallibleTransformTrait<I, O, E> {
  abstract transform(input: I): Result<O, E>;
}

/**
 * Concrete adapter for callers/tests that still want to pass a closure while we
 * migrate. Wrapping a closure in a `Transform` keeps call sites uniform; the
 * closure itself is the only thing that needs a hand-translation to a Rust
 * struct later. Prefer a named `PureTransform` subclass for anything reused.
 */
export class FnTransform<I, O> extends PureTransform<I, O> {
  constructor(private readonly fn: (input: I) => O) {
    super();
  }

  transform(input: I): O {
    return this.fn(input);
  }
}
