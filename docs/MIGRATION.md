# TypeScript to Rust Migration Notes

## 1. Goal

This repository should be prepared for a future file-for-file Rust migration. The intended mapping is direct:

- `src/des/foo.ts` becomes `src/des/foo.rs`
- `src/des/folder/index.ts` becomes `src/des/folder/mod.rs`
- `src/des/main-thing.ts` becomes either `src/bin/main_thing.rs` or a thin binary calling `src/des/main_thing.rs`

The TypeScript source remains the implementation of record for now. Comments added during this pass are migration scaffolding only.

## 2. Rust Shape

- TypeScript classes map to Rust structs with one or more trait impl blocks.
- TypeScript abstract classes map to a Rust trait plus, where shared state exists, a reusable state struct.
- TypeScript interfaces map to Rust traits when they describe behavior and Rust structs/enums when they describe data.
- TypeScript discriminated unions map to Rust enums.
- TypeScript `Map` and `Set` map to `HashMap` and `HashSet`, or `BTreeMap` and `BTreeSet` where deterministic iteration matters.
- TypeScript `throw` paths map to `Result<T, E>` unless the current behavior is deliberately unrecoverable.

## 3. Pure Transform Rule

When a plain function participates in the DES graph, prefer a `PureTransform` / `PureTransformEntity` class with a `transform(...)` method. That shape ports cleanly to Rust as:

```rust
pub trait PureTransform<I, O> {
    fn transform(&self, input: I, ctx: &TransformContext) -> TransformResult<O>;
}
```

Standalone math helpers may remain helpers during this TypeScript cleanup, but each file header should say whether they become inherent methods, trait methods, or private module functions in Rust.

## 4. Global Gotchas

| TypeScript pattern | Rust migration target |
| --- | --- |
| `any` / loose structural values | concrete structs, enums, or `serde_json::Value` at boundary only |
| optional object fields | `Option<T>` |
| object spreads | explicit constructors or builder structs |
| callbacks | generic `Fn` traits, boxed trait objects, or concrete strategy structs |
| inheritance with shared mutable fields | state struct plus trait impl |
| getters | methods such as `short_uuid()` |
| `Math.random()` | injected RNG trait/object |
| `process.env` | configuration struct populated at boundary |
| `fs` / `path` | `std::fs`, `std::path::PathBuf` |
| `mathjs.BigNumber` | `rust_decimal`, `bigdecimal`, or `f64` after domain review |
| `uuid` | `uuid` crate |
| `ws` | `tokio-tungstenite` or `axum` websocket extractors |
| `zod` schemas | `serde` structs plus validation constructors |

## 5. Inline Header Template

Each source file should start with a short `RUST MIGRATION` header:

```ts
// RUST MIGRATION:
// - Target: src/des/path/to/file.rs
// - Keep file-for-file. Convert exported interfaces/types to Rust structs/enums/traits in this module.
// - Convert exported classes to structs plus trait impls; preserve template-method hooks as traits.
// - Keep free helpers private unless they are part of the public API; DES graph functions should become PureTransform implementors.
// - Replace thrown errors with Result-returning constructors or validators.
```

Specialized headers should add file-specific notes for tests, runners, animation, servers, and external adapters.
