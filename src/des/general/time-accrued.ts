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