'use strict';

// =============================================================================
// RUST MIGRATION — target: src/des/shared/mod.rs   (module des::shared)
// 1:1 file move. Barrel/`index.ts` -> `mod.rs` re-exporting the child modules.
// See the module doc below for the full Rust mapping.
// =============================================================================

// =============================================================================
// shared/index.ts — public API of the dependency-free shared foundation.
//
// These modules are the FIRST things to translate to Rust: they have no
// dependency on any feature module and encode the engine-wide conventions
// (Transform trait, Result/Option, capability ports, linear algebra).
// =============================================================================

export * from './result';
export * from './transform';
export * from './capabilities';
export * from './linalg';
