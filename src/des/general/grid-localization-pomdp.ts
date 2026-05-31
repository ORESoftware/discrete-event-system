// RUST MIGRATION: target module src/des/general/grid_localization_pomdp.rs.
// RUST MIGRATION: GridLocalizationActionKind and GridLocalizationObservation become enums; params/actions/trace/result/model become serde structs.
// RUST MIGRATION: buildGridLocalizationPOMDP and runGridLocalizationPOMDP are POMDP model/transformation entrypoints; expose graph-visible ones as PureTransform structs.
// RUST MIGRATION: Dense transition/observation/belief tables should use Vec matrices; sampling takes injected rand::Rng and validation returns Result.
'use strict';

// =============================================================================
// grid-localization-pomdp
//
// A multi-dimensional POMDP over a hidden target location (x, y). The agent can
// scan rows, scan columns, or inspect a specific cell. Scans are noisy binary
// observations; inspecting the right cell ends the episode with a reward.
//
// This is deliberately small enough to run exact belief-tree lookahead, while
// still exercising the framework's multi-dimensional state-space utilities.
// =============================================================================

import {DiscreteBelief} from './belief';
import {CartesianStateSpace} from './cartesian-state-space';
import {
  BeliefLookaheadSolver,
  beliefUpdate,
  POMDPSpec,
} from './pomdp';
import {mulberry32} from './prng';
import {Preconditions} from './des-base/preconditions';

export type GridLocalizationActionKind = 'scan-row' | 'scan-column' | 'inspect';
export type GridLocalizationObservation = 'no' | 'yes';

export interface GridLocalizationParams {
  width: number;
  height: number;
  horizon?: number;
  numSteps?: number;
  seed?: number;
  hiddenTarget?: readonly [number, number];
  initialBelief?: readonly number[];
  scanAccuracy?: number;
  inspectAccuracy?: number;
  scanCost?: number;
  inspectCorrectReward?: number;
  inspectWrongPenalty?: number;
  discount?: number;
}

export interface GridLocalizationAction {
  kind: GridLocalizationActionKind;
  index: number;
  x?: number;
  y?: number;
  label: string;
}

export interface GridLocalizationTraceRow {
  step: number;
  action: GridLocalizationAction;
  observation: GridLocalizationObservation;
  hiddenTarget: [number, number];
  entropy: number;
  mode: [number, number];
  modeProbability: number;
  hiddenProbability: number;
  found: boolean;
}

export interface GridLocalizationResult {
  params: Required<Omit<GridLocalizationParams, 'initialBelief' | 'hiddenTarget'>> & {
    hiddenTarget: [number, number];
  };
  stateSpace: {
    dimensions: Array<{name: string; size: number}>;
    numStates: number;
  };
  actions: GridLocalizationAction[];
  observations: GridLocalizationObservation[];
  trace: GridLocalizationTraceRow[];
  finalBelief: number[];
  finalEntropy: number;
  found: boolean;
  foundAtStep?: number;
  totalReturn: number;
}

interface GridLocalizationModel {
  space: CartesianStateSpace;
  actions: GridLocalizationAction[];
  spec: POMDPSpec<number, GridLocalizationAction, GridLocalizationObservation>;
}

const OBSERVATIONS: GridLocalizationObservation[] = ['no', 'yes'];

export function buildGridLocalizationPOMDP(params: GridLocalizationParams): GridLocalizationModel {
  validateParams(params);
  const width = params.width;
  const height = params.height;
  const space = new CartesianStateSpace([
    {name: 'x', size: width},
    {name: 'y', size: height},
  ]);
  const actions = buildActions(width, height);
  const states = Array.from({length: space.numStates}, (_, i) => i);
  const scanAccuracy = params.scanAccuracy ?? 0.9;
  const inspectAccuracy = params.inspectAccuracy ?? 0.99;
  const scanCost = params.scanCost ?? -0.2;
  const inspectCorrectReward = params.inspectCorrectReward ?? 20;
  const inspectWrongPenalty = params.inspectWrongPenalty ?? -12;
  const discount = params.discount ?? 0.95;
  const spec: POMDPSpec<number, GridLocalizationAction, GridLocalizationObservation> = {
    states,
    actions,
    observations: OBSERVATIONS,
    transition: sIdx => {
      const row = new Array<number>(space.numStates).fill(0);
      row[sIdx] = 1;
      return row;
    },
    observation: (sIdx, aIdx) => {
      const action = actions[aIdx];
      const [x, y] = space.decode(sIdx);
      const trueYes = action.kind === 'scan-row'
        ? y === action.y
        : action.kind === 'scan-column'
          ? x === action.x
          : x === action.x && y === action.y;
      const acc = action.kind === 'inspect' ? inspectAccuracy : scanAccuracy;
      const pYes = trueYes ? acc : 1 - acc;
      return [1 - pYes, pYes];
    },
    reward: (sIdx, aIdx) => {
      const action = actions[aIdx];
      if (action.kind !== 'inspect') return scanCost;
      const [x, y] = space.decode(sIdx);
      return x === action.x && y === action.y ? inspectCorrectReward : inspectWrongPenalty;
    },
    discount,
    initialBelief: params.initialBelief?.slice() ?? states.map(() => 1 / states.length),
  };
  return {space, actions, spec};
}

export function runGridLocalizationPOMDP(params: GridLocalizationParams): GridLocalizationResult {
  validateParams(params);
  const model = buildGridLocalizationPOMDP(params);
  const p = normaliseParams(params, model.space);
  const rng = mulberry32(p.seed);
  const hiddenTarget = p.hiddenTarget;
  const hiddenIndex = model.space.encode(hiddenTarget);
  let belief = new DiscreteBelief(model.spec.states, model.spec.initialBelief);
  const planner = new BeliefLookaheadSolver(model.spec, {
    horizon: p.horizon,
    leaf: 'qmdp',
    maxNodes: 500000,
  });
  const trace: GridLocalizationTraceRow[] = [];
  let discount = 1;
  let totalReturn = 0;
  let found = false;
  let foundAtStep: number | undefined;

  for (let step = 0; step < p.numSteps && !found; step++) {
    const actionIdx = planner.act(belief);
    const action = model.actions[actionIdx];
    const obsDist = model.spec.observation(hiddenIndex, actionIdx);
    const obsIdx = sampleIndex(obsDist, rng);
    const observation = OBSERVATIONS[obsIdx];
    totalReturn += discount * model.spec.reward(hiddenIndex, actionIdx);
    discount *= p.discount;
    belief = beliefUpdate(model.spec, belief, actionIdx, obsIdx);
    found = action.kind === 'inspect' && observation === 'yes';
    if (found) foundAtStep = step;
    const modeIndex = belief.modeIndex();
    const mode = model.space.decode(modeIndex) as [number, number];
    trace.push({
      step,
      action,
      observation,
      hiddenTarget,
      entropy: belief.entropy(),
      mode,
      modeProbability: belief.weights[modeIndex],
      hiddenProbability: belief.weights[hiddenIndex],
      found,
    });

  }

  return {
    params: p,
    stateSpace: {
      dimensions: model.space.dimensions.map(d => ({name: d.name, size: d.size})),
      numStates: model.space.numStates,
    },
    actions: model.actions,
    observations: OBSERVATIONS.slice(),
    trace,
    finalBelief: belief.asArray(),
    finalEntropy: belief.entropy(),
    found,
    foundAtStep,
    totalReturn,
  };
}

function buildActions(width: number, height: number): GridLocalizationAction[] {
  const actions: GridLocalizationAction[] = [];
  for (let y = 0; y < height; y++) actions.push({kind: 'scan-row', index: actions.length, y, label: `scan row ${y}`});
  for (let x = 0; x < width; x++) actions.push({kind: 'scan-column', index: actions.length, x, label: `scan column ${x}`});
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      actions.push({kind: 'inspect', index: actions.length, x, y, label: `inspect (${x},${y})`});
    }
  }
  return actions;
}

function validateParams(params: GridLocalizationParams): void {
  const cls = 'GridLocalizationPOMDP';
  Preconditions.integerInRange(cls, 'width', params.width, 2, 8);
  Preconditions.integerInRange(cls, 'height', params.height, 2, 8);
  if (params.horizon !== undefined) Preconditions.integerInRange(cls, 'horizon', params.horizon, 0, 6);
  if (params.numSteps !== undefined) Preconditions.integerInRange(cls, 'numSteps', params.numSteps, 1, 100);
  if (params.seed !== undefined) Preconditions.integer(cls, 'seed', params.seed);
  if (params.scanAccuracy !== undefined) Preconditions.inRange(cls, 'scanAccuracy', params.scanAccuracy, 0.5, 1);
  if (params.inspectAccuracy !== undefined) Preconditions.inRange(cls, 'inspectAccuracy', params.inspectAccuracy, 0.5, 1);
  if (params.discount !== undefined) Preconditions.inRange(cls, 'discount', params.discount, 0, 1);
  if (params.initialBelief !== undefined) {
    Preconditions.lengthEq(cls, 'initialBelief', params.initialBelief, params.width * params.height);
    Preconditions.probabilityVector(cls, 'initialBelief', params.initialBelief);
  }
  if (params.hiddenTarget !== undefined) {
    Preconditions.lengthEq(cls, 'hiddenTarget', params.hiddenTarget, 2);
    Preconditions.integerInRange(cls, 'hiddenTarget.x', params.hiddenTarget[0], 0, params.width - 1);
    Preconditions.integerInRange(cls, 'hiddenTarget.y', params.hiddenTarget[1], 0, params.height - 1);
  }
}

function normaliseParams(
  params: GridLocalizationParams,
  space: CartesianStateSpace,
): Required<Omit<GridLocalizationParams, 'initialBelief' | 'hiddenTarget'>> & {hiddenTarget: [number, number]} {
  const seed = params.seed ?? 1;
  const rng = mulberry32(seed + 10007);
  const hiddenTarget = params.hiddenTarget
    ? [params.hiddenTarget[0], params.hiddenTarget[1]] as [number, number]
    : space.decode(Math.floor(rng() * space.numStates)) as [number, number];
  return {
    width: params.width,
    height: params.height,
    horizon: params.horizon ?? 3,
    numSteps: params.numSteps ?? 8,
    seed,
    hiddenTarget,
    scanAccuracy: params.scanAccuracy ?? 0.9,
    inspectAccuracy: params.inspectAccuracy ?? 0.99,
    scanCost: params.scanCost ?? -0.2,
    inspectCorrectReward: params.inspectCorrectReward ?? 20,
    inspectWrongPenalty: params.inspectWrongPenalty ?? -12,
    discount: params.discount ?? 0.95,
  };
}

function sampleIndex(probabilities: readonly number[], rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probabilities.length; i++) {
    acc += probabilities[i];
    if (u <= acc) return i;
  }
  return probabilities.length - 1;
}
