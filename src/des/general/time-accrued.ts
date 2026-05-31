// RUST MIGRATION: Target module `src/des/general/time_accrued.rs`.
// RUST MIGRATION: Replace the module-level mutable object with an explicit `TimeAccrued` struct whose current time and step size are owned by the simulation context.
// RUST MIGRATION: Map `math.BigNumber` to a chosen Rust numeric type (`f64`, `rust_decimal`, or big rational) consistently with the rest of the time model.
// RUST MIGRATION: Convert getter/setter arrow functions into methods on `TimeAccrued`; avoid global mutable state unless guarded by an explicit runtime handle.
// RUST MIGRATION: Return `Result` from setters/bump methods for negative, zero, or non-finite step sizes.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/time-accrued.rs  (module des::general::time_accrued)
// 1:1 file move. A global simulation clock: accrued time + step size, mutated in place.
//
// Declarations → Rust:
//   const timeAccrued (module-level mutable singleton)  -> see note (NO global mutable analogue)
//   const getStepSize / setStepSize / bumpTimeAccruedByTimeStep / getTimeAccrued (arrow fns)
//        -> methods on a `SimClock` struct: `step_size()`, `set_step_size()`, `bump(dt)`, `now()`
//
// Conversion notes (file-specific):
//   - GLOBAL MUTABLE STATE: `timeAccrued` is a module-level singleton mutated by setters. Rust
//     forbids ergonomic global mutables — model this as the `Clock` capability (shared/capabilities):
//     an owned `SimClock` struct threaded through the simulation, not a static.
//   - `math.BigNumber` (currentTime/stepSize via `bgn`) -> a decimal/bignum crate (rust_decimal)
//     or `f64`; pick ONE time representation engine-wide. `math.add` -> the chosen type's `+`.
//   - getter/setter pattern -> `fn now(&self)` / `fn set_step_size(&mut self, ..)` (§5.2).
//   - `bgn` factory comes from general.ts (see that file's header).
// =============================================================================

import {bgn} from "./general";
import * as math from "mathjs";

const timeAccrued = {
  currentTime: bgn(0),
  stepSizeMillis: bgn(10)
};

export const getStepSize = () => {
  return timeAccrued.stepSizeMillis;
};

export const setStepSize = (v: math.BigNumber) => {
   timeAccrued.stepSizeMillis = v;
};

export const bumpTimeAccruedByTimeStep = (timeStep: math.BigNumber) => {
  timeAccrued.currentTime = math.add(timeAccrued.currentTime, timeStep);
};

export const getTimeAccrued = () => {
  return timeAccrued.currentTime;
}
