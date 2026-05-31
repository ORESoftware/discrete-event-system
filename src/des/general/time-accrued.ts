// RUST MIGRATION: Target module `src/des/general/time_accrued.rs`.
// RUST MIGRATION: Replace the module-level mutable object with an explicit `TimeAccrued` struct whose current time and step size are owned by the simulation context.
// RUST MIGRATION: Map `math.BigNumber` to a chosen Rust numeric type (`f64`, `rust_decimal`, or big rational) consistently with the rest of the time model.
// RUST MIGRATION: Convert getter/setter arrow functions into methods on `TimeAccrued`; avoid global mutable state unless guarded by an explicit runtime handle.
// RUST MIGRATION: Return `Result` from setters/bump methods for negative, zero, or non-finite step sizes.
'use strict';

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
