'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/mod.rs
// - This is the Rust `mod.rs` boundary for all des_base modules. Each export
//   below should become a `pub mod ...;` plus selective `pub use ...;` re-export.
// - Preserve the class-family hierarchy as trait families: DESStation as the
//   base trait/state pair, then optimizer, RL, controller, transform, and visual
//   block traits layered over it.
// - Keep the module list one-to-one with the TypeScript files during the first
//   migration. Consolidation can happen only after behavioral parity tests pass.

// =============================================================================
// general/des-base/index.ts — public API of the DES base hierarchy.
//
// Hierarchy:
//
//   DESStation                                   — foundation
//   ├── TransformEntity<I, O>                    — zero-queue function station
//   ├── SingleStateOptimizer<S>                  — SA / hill climb / tabu
//   ├── PopulationOptimizer<I>                   — GA / PSO / DE / ACO
//   ├── RLAgentStation<S, A>                     — Q-learning / SARSA
//   ├── PolicyGradientAgent<S, A>                — REINFORCE / A2C / PPO
//   │   PolicyUpdateStation                      — counterpart for the above
//   ├── EnvironmentStation<S, A>                 — environment wrapper
//   ├── NeuralNetworkStation                     — inference / supervised learning
//   ├── TreeSearchStation<N>                     — MILP-B&B / MCTS / A* / beam
//   ├── FixedPointIterationStation<S>            — value iter / Benders / Jacobi
//   └── ControllerStation<O, U>                  — bang-bang / PID / fuzzy / MPC
//
// All algorithm-family bases enforce a TEMPLATE-METHOD `runTimeStep` and
// expose ABSTRACT hooks for the pieces that differ across algorithms in
// that family. Concrete leaf algorithms implement only the hooks.
// =============================================================================

export * from './validation';
export * from './station';
export * from './stateful-token';
export * from './smart-movable';
export * from './transform-entity';
export * from './episode-accounting';
export * from './composite-station';
export * from './runner';
export * from './rl-tokens';
export * from './single-state-optimizer';
export * from './population-optimizer';
export * from './rl-agent';
export * from './policy-gradient-agent';
export * from './environment';
export * from './tree-search';
export * from './fixed-point';
export * from './controller';
export * from './finite-horizon-dp';
export * from './linear-vfa';
export * from './belief-state';
export * from './semi-mdp';
export * from './actor-critic';
export * from './monte-carlo-rl';
export * from './multi-agent';
export * from './lqr-controller';
export * from './control-blocks';
export * from './preconditions';
export * from './argmax';
export * from './cut-pool';
export * from './neural-network';
export * from './learning-optimization';
export * from './model-topology';
export * from './advanced-optimization';
export * from './adversarial-control';
