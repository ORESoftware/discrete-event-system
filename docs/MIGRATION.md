# TypeScript → Rust Migration Guide

This document records the conventions and the structural work used to harden the
TypeScript source so it can be migrated to a Rust crate with as little
re-design as possible. Rust has **no classes** — only `struct` + `trait` — so the
TypeScript is being shaped to mirror that model.

The machine-enforced summary lives in `.cursor/rules/rust-migration-conventions.mdc`.

---

## 1. The core mapping

| TypeScript                                   | Rust                                            |
| -------------------------------------------- | ----------------------------------------------- |
| `interface Trait<I, O> { m(i: I): O }`       | `trait Trait<I, O> { fn m(&self, i: I) -> O }`  |
| `abstract class Base implements Trait`       | `struct Base; impl Trait for Base`              |
| `class Foo extends Base` (config in ctor)    | `struct Foo { ...fields }`                      |
| method `transform(input): output`            | `fn transform(&self, input) -> output`          |
| discriminated union `type T = A \| B`        | `enum T { A(..), B(..) }`                        |
| `switch (x.kind) { case 'a': ... }`          | `match x { T::A(..) => ... }`                    |
| `Result<T, E>` (`shared/result.ts`)          | `Result<T, E>`                                  |
| `Option<T>` (`shared/result.ts`)             | `Option<T>`                                     |
| `throw new Error(...)`                       | `panic!(...)` (bugs only)                       |
| `RandomSource` / `Clock` ports               | injected `rand::Rng` / clock trait              |

---

## 2. The `shared/` foundation (translate this first)

`src/des/shared/` is a **dependency-free leaf**. It encodes the conventions and
should be the first thing ported to Rust.

- **`transform.ts`** — `Transform<I, O>` (the trait) plus the base classes a
  "vanilla function" should become:
  - `PureTransform<I, O>` — pure, deterministic, no I/O / globals / RNG.
  - `StatefulTransform<I, O>` — carries mutable state across calls (`&mut self`).
  - `FallibleTransform<I, O, E>` — returns `Result<O, E>` instead of throwing.
  - `FnTransform<I, O>` — wraps a closure during migration.
- **`result.ts`** — Rust-shaped `Result<T, E>` and `Option<T>` as tagged unions
  (`ok`/`err`/`some`/`none`, `map`, `unwrapOr`, …).
- **`capabilities.ts`** — `RandomSource` / `Clock` traits with `SystemRandom`,
  `SystemClock`, and a deterministic `SeededRandom` (mulberry32).
- **`linalg.ts`** — dense linear algebra as classes: `LinAlg` / `VecOps` (static
  arithmetic), `MatrixInverse`, `LinearSystem`, `MatrixRank`, `SymmetricEigen`.
  Canonical home; `control-systems/linear-algebra.ts` now re-exports it.

---

## 3. Rules for new and refactored code

### 3.1 Functions become transforms

A free function is replaced by a class. Parameters that configure the algorithm
go on the constructor (`readonly` fields = struct fields); the thing being
processed is the single `transform` input.

```typescript
// before
export function bfgs(f, grad, x0, opts) { ... }

// after
export class Bfgs extends PureTransform<FirstOrderProblem, OptimResult> {
  constructor(private readonly opts: OptimOptions = {}) { super(); }
  transform(p: FirstOrderProblem): OptimResult { ... }
}
```

Multi-argument calls are collapsed into a single named `Input` interface so there
are no positional-only signatures to untangle in Rust.

During migration, the old free function may remain as a one-line `@deprecated`
shim delegating to the class, so existing call sites keep compiling. New code
should use the class.

### 3.2 Enums and pattern matching

Model closed sets of variants as discriminated unions with a `kind` tag and match
on `kind`. This is already used for the symbolic AST (`expr.ts`) and for
`Result`/`Option`.

### 3.3 Errors

- Expected, recoverable failure → return `Result<T, E>`.
- Programmer error / broken invariant → `throw` (becomes `panic!`).

### 3.4 Determinism

No direct `Math.random()` / `Date.now()`. Take a `RandomSource` / `Clock`
parameter (default to `DEFAULT_RANDOM` / `DEFAULT_CLOCK` only at the very edge).

### 3.5 Types

- No `any` / `as any` in new code.
- Prefer `readonly` inputs and returning fresh values over mutation.

---

## 4. Status / roadmap

**Done**
- `shared/` foundation (`transform`, `result`, `capabilities`, `linalg`).
- `general/optim.ts` → `GradientDescent`, `NewtonOptim`, `Bfgs`, `AutoGradient`.
- `general/expr.ts` → `ExprParser`, `ExprEvaluator`, `ExprCompiler`,
  `ExprPrinter`, `ExprDifferentiator`, `ExprSimplifier`, plus numerical-derivative
  transforms.
- Linear-algebra toolkit relocated to `shared/linalg.ts`.

**Remaining (prioritized)**
1. Convert the rest of the pure-math / algorithm modules in `general/` that still
   expose free functions (LP, ODE, random-variables, etc.) to transform classes.
2. Route the ~150 direct `Math.random()` / `Date.now()` call sites through
   `RandomSource` / `Clock`.
3. Replace recoverable `throw` sites with `Result` where the caller can react.
4. Drive down `any` / `as any` usage in core modules.
5. Drop `@deprecated` free-function shims once their call sites use the classes.

The `des-base/` hierarchy is already class/trait-shaped (template-method bases
with abstract hooks) and migrates with minimal change.

---

## 5. Per-file migration headers (the 1:1 plan)

The migration is a **file-for-file move**: every `src/des/**/X.ts` becomes
`src/des/**/X.rs` (a Rust module). Barrels (`index.ts`) become `mod.rs`. To make
that mechanical, **every source file carries a `RUST MIGRATION` header** placed
directly under the leading `'use strict';` line.

### 5.1 Header template

```typescript
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/<same-path>.rs   (module des::<...>)
// 1:1 file move. <one line: what this module is>.
//
// Declarations → Rust:
//   class Foo            -> struct Foo { fields } + impl (+ impl Trait for Foo)
//   abstract class Base  -> trait Base { default fns } (state via fields on impls)
//   interface Behavior   -> trait Behavior
//   interface DataShape  -> struct DataShape   (#[derive(Clone, ...)])
//   type U = A | B        -> enum U { A(..), B(..) }   (match on `kind`)
//   enum E               -> enum E
//   const fn-table       -> match in an associated fn
//
// Imports → use:
//   import {X} from '../shared/...'  ->  use crate::des::shared::...::X;
//
// Conversion notes (file-specific):
//   - <gotcha 1>
//   - <gotcha 2>
// =============================================================================
```

Keep it concise and **file-specific** in the notes — only call out things a Rust
author actually needs (overloads, `as any`, RNG/clock, `Map`/`Set` keys,
closures, cyclic types, BigNumber, etc.). Do **not** restate the whole template
when nothing tricky applies.

### 5.2 Global gotchas to reference (don't repeat in every file)

| TypeScript thing                     | Rust treatment                                            |
| ------------------------------------ | --------------------------------------------------------- |
| `class A extends B`                  | no inheritance: trait `B` with defaults + struct `A`      |
| structural `interface`               | nominal `trait`/`struct` — write explicit `impl`          |
| `any` / `as any`                     | pick a concrete type, generic, or `enum`; `dyn Any` last  |
| function overloads                   | one generic fn, or an input `enum`, or distinct fns       |
| getters/setters                      | plain methods `fn x(&self)` / `fn set_x(&mut self, ..)`   |
| `throw`                              | `Result<_, E>` (recoverable) or `panic!` (bug)            |
| `Math.random()` / `Date.now()`       | inject `RandomSource` / `Clock` (`shared/capabilities`)   |
| `Map<K,V>` / `Set<V>`                | `HashMap`/`HashSet` — `K`/`V` need `Hash + Eq`; order N/A |
| `number`                             | `f64` (math) / `i64` / `usize` (indices) — choose per use |
| `mathjs.BigNumber`                   | **`rust_decimal::Decimal`** via `des::shared::precision` (exact compound bookkeeping: clock/money/probs/RV) — NOT for numerical kernels, which stay `f64`. See §5.4 |
| `Fraction` / exact `p/q`             | `num_rational::Rational64` / `BigRational` (`precision::frac`) |
| long `f64` accumulation (hot loop)   | `precision::KahanAccumulator` / `kahan_sum` (compensated)  |
| float `==` / `!=`                    | `precision::approx_eq` / `approx_eq_rel` / `almost_zero` — EXCEPT exact-zero divide guards & integer (`fract()==0.0`) checks, which stay exact |
| `uuid` / `ws` / `zod`                | `uuid` / `tokio-tungstenite` / `serde` + `validator`      |
| `JSON` / `toJSON` / serialize        | `serde` + `serde_json` (`#[derive(Serialize)]`)           |
| closures capturing mutable state     | `FnMut` + `move`; watch the borrow checker                |
| union of string literals             | `enum` with `#[serde(rename_all = ...)]`                  |

### 5.3 Rules when adding headers

- **Only add comments.** Never change code, signatures, imports, or behaviour.
- Headers go right under `'use strict';` (or at line 1 if absent).
- Files that already have a rich top doc-comment: prepend the `RUST MIGRATION`
  block above it (keep the existing doc).
- The build must stay green (`tsc --noEmit`). Comments can't break it, but verify.

### 5.4 Numeric precision policy (FINALIZED)

The engine runs a deliberate **two-tier** precision strategy. The Rust port
encodes it in `des::shared::precision` (read that module's doc). Pick the tier
per the *role* of the number, not by habit.

**Tier 1 — numerical kernels stay `f64`.** Linear algebra, optimization, ODE
integration, LP/simplex, eigen-solvers, quadrature, statistics, RL value
functions. Rationale:
- Every model cross-validates against five `float64` references (Python /
  numpy / scipy / R / Octave). Decimals would stop matching them.
- These are iterative approximations whose truncation error dwarfs `f64`
  round-off; exact arithmetic buys nothing and costs ~100×.
- `test/float-bias-test.ts` measures plain-`f64` 1M-tick drift at ~1e-6
  (relative ~2e-10) — negligible.
- For long accumulation in a hot loop use `precision::KahanAccumulator` /
  `kahan_sum`. Never `==`/`!=` on `f64` — use `approx_eq` / `approx_eq_rel` /
  `almost_zero`. (EXCEPTION: exact-zero divide guards and `fract()==0.0`
  integer checks legitimately use exact equality.)

**Tier 2 — exact base-10 `Decimal` for compound bookkeeping** (the
`mathjs.BigNumber` domains): the DES clock / time accrual, money &
prediction-market balances (`factmachine`), routing probabilities that must
sum to 1 (`probability-decision`), and the RV algebra (`random-variables/rv`).
Use `rust_decimal::Decimal`; construct with `precision::bgn` (matches TS
`math.bignumber(String(x))` — exact for `0.05`, `0.1`, …), coerce on read-out
with `precision::to_f64` (matches `Number(bn.toString())`). Reference
implementation: `general/time_accrued.rs` (the `SimClock`).

**Tier 3 — exact rationals** (`p/q`, where TS used `Fraction`):
`num_rational::Rational64` / `BigRational`, via `precision::frac`.

