// RUST MIGRATION: Target module `src/des/general/control_systems/linear_algebra.rs`.
// RUST MIGRATION: Convert `Vec`/`Mat` aliases, `LinAlg`, inverse, eigen, and rank helpers into nominal structs/traits over `Vec<f64>` and `Vec<Vec<f64>>`.
// RUST MIGRATION: Keep plant/controller/estimator users on explicit `f64` matrix/vector APIs and pass solver tolerances/config instead of globals.
// RUST MIGRATION: Any graph-visible pure matrix evaluator should be wrapped as a PureTransform-style struct with a `transform` method returning `Result`.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/control-systems/linear-algebra.rs
//   (module des::general::control_systems::linear_algebra)
// 1:1 file move. COMPATIBILITY RE-EXPORT SHIM — declares NO items of its own.
//
// Declarations → Rust:
//   export * from '../../shared/linalg'  ->  pub use crate::des::shared::linalg::*;
//
// Conversion notes (file-specific):
//   - The dense linear-algebra toolkit now lives in `shared/linalg.ts`
//     (LinAlg, VecOps, MatrixInverse, LinearSystem, MatrixRank, SymmetricEigen).
//     This file is ONLY a re-export so legacy `control-systems/linear-algebra`
//     imports keep resolving; in Rust it is a single `pub use` (mod.rs-style
//     re-export), NOT a second copy of the algebra.
//   - New code should depend on `shared::linalg` directly; this shim can be
//     deleted once all call sites are repointed.
// =============================================================================

// =============================================================================
// control-systems/linear-algebra.ts — COMPATIBILITY RE-EXPORT.
//
// The dense linear-algebra toolkit now lives at `shared/linalg.ts` (a
// dependency-free leaf module). This file is retained so existing imports
// (`../general/control-systems/linear-algebra`) keep working; new code should
// import from `shared/linalg` directly.
// =============================================================================

export * from '../../shared/linalg';
