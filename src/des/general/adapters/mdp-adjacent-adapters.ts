// RUST MIGRATION: Target module `src/des/general/adapters/mdp_adjacent_adapters.rs`.
// RUST MIGRATION: Convert inventory, mountain-car, tiger, localization, actor-critic, blackjack, and LQR adapters into structs/functions around `DESModelSpec`.
// RUST MIGRATION: Encode MDP/POMDP params, policies, traces, and summaries as `serde` config/result structs; runtime/output paths become `PathBuf`.
// RUST MIGRATION: Use `Result<_, ValidationError>` for probability, grid, transition, and action-space validation.
'use strict';

// =============================================================================
// general/adapters/mdp-adjacent-adapters.ts — JSON adapters for the
// nine MDP-adjacent models added in this batch:
//
//   • inventory-dp           — finite-horizon dynamic programming
//   • mountain-car-vfa       — approximate dynamic programming (linear VFA)
//   • tiger-pomdp            — POMDP (belief-state value iteration / QMDP)
//   • grid-localization-pomdp — multi-dimensional POMDP belief lookahead
//   • four-rooms-smdp        — Semi-MDP / options framework
//   • actor-critic-grid      — actor-critic on tabular GridWorld
//   • blackjack-mc           — Monte Carlo on-policy control
//   • stag-hunt              — multi-agent independent Q-learning
//   • double-integrator-lqr  — LQR / stochastic control
//
// Each adapter follows the `DESModelRegistration<P, R>` contract from
// `des-spec.ts`.
// =============================================================================

import {ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';

import {InventoryProblem, solveInventoryDP, InventoryDPResult} from '../inventory-dp';
import {runMountainCar, MountainCarResult, MountainCarTrainOpts} from '../mountain-car';
import {simulateTiger, buildTigerSpec, TigerSimResult, TigerOpts} from '../tiger-pomdp';
import {runGridLocalizationPOMDP, GridLocalizationParams, GridLocalizationResult} from '../grid-localization-pomdp';
import {runFourRoomsSMDP, FourRoomsResult, FourRoomsTrainOpts} from '../four-rooms';
import {runActorCriticGridworld, ActorCriticResult, ActorCriticTrainOpts} from '../actor-critic-gridworld';
import {runBlackjackMC, BlackjackResult, BlackjackTrainOpts} from '../blackjack';
import {runStagHunt, StagHuntResult, StagHuntOpts} from '../stag-hunt';
import {runDoubleIntegratorLQR, DoubleIntegratorOpts, DoubleIntegratorResult} from '../double-integrator-lqr';
import {csvRow, numberPair, writeCsvLines} from './adapter-utils';

// -----------------------------------------------------------------------------
// 1. inventory-dp
// -----------------------------------------------------------------------------

const inventorySchema: ParamSchema = {
  kind: 'object',
  description: 'Multi-period stochastic inventory by finite-horizon DP.',
  fields: {
    horizon: {kind: 'number', integer: true, min: 1},
    S_max: {kind: 'number', integer: true, min: 0},
    demandPmf: {kind: 'array', items: {kind: 'number', min: 0, max: 1}},
    price: {kind: 'number', min: 0},
    cost: {kind: 'number', min: 0},
    fixedCost: {kind: 'number', min: 0, default: 0},
    holdCost: {kind: 'number', min: 0, default: 0.5},
    stockoutCost: {kind: 'number', min: 0, default: 5},
    salvageValue: {kind: 'number', default: 0},
    discount: {kind: 'number', min: 0, max: 1, default: 1},
    initialInventory: {kind: 'number', integer: true, min: 0},
  },
  required: ['horizon', 'S_max', 'demandPmf', 'price', 'cost', 'initialInventory'],
};

registerModel<InventoryProblem & {seed?: number}, InventoryDPResult>({
  id: 'inventory-dp',
  description: 'Multi-period stochastic inventory solved by finite-horizon DP (backward induction).',
  schema: inventorySchema,
  run(p, _runtime) { return solveInventoryDP(p, {seed: p.seed ?? 1}); },
  summarize(r, p) {
    return [
      'INVENTORY DP',
      '────────────────────────────',
      `  Horizon:      ${p.horizon}`,
      `  S_max:        ${p.S_max}`,
      `  E[demand]:    ${r.meanDemand.toFixed(2)}`,
      `  V*(t=0, s=${p.initialInventory}): ${r.expectedReward.toFixed(3)}`,
      `  Sim total reward (seed=${p.seed ?? 1}): ${r.simulation.totalReward.toFixed(3)}`,
      `  Orders:       ${r.simulation.orders.join(', ')}`,
      `  Demands:      ${r.simulation.demands.join(', ')}`,
      `  Inventory:    ${r.simulation.inventory.join(', ')}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,inventory,order,demand,reward'];
    for (let t = 0; t < r.simulation.orders.length; t++) {
      lines.push(csvRow([t, r.simulation.inventory[t], r.simulation.orders[t], r.simulation.demands[t], r.simulation.rewards[t].toFixed(4)]));
    }
    writeCsvLines(csvPath, lines);
  },
});

// -----------------------------------------------------------------------------
// 2. mountain-car-vfa
// -----------------------------------------------------------------------------

const mountainCarSchema: ParamSchema = {
  kind: 'object',
  description: 'Mountain Car solved by linear VFA with tile coding (Sutton-Albus).',
  fields: {
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 200},
    maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 1000},
    alpha: {kind: 'number', min: 0, default: 0.5},
    gamma: {kind: 'number', min: 0, max: 1, default: 1.0},
    epsilon: {kind: 'number', min: 0, max: 1, default: 0},
    epsilonDecay: {kind: 'number', min: 0, max: 1, default: 1},
    epsilonMin: {kind: 'number', min: 0, max: 1, default: 0},
    seed: {kind: 'number', integer: true, default: 1},
    numTilings: {kind: 'number', integer: true, min: 1, default: 8},
    numTilesPerDim: {kind: 'number', integer: true, min: 2, default: 8},
  },
  required: ['numEpisodes'],
};

registerModel<MountainCarTrainOpts, MountainCarResult>({
  id: 'mountain-car-vfa',
  description: 'Mountain Car (continuous control) by linear VFA with Sutton-Albus tile coding.',
  schema: mountainCarSchema,
  run(p) { return runMountainCar(p); },
  summarize(r) {
    const meanLast = (xs: readonly number[], n: number) => {
      const a = xs.slice(-Math.min(n, xs.length));
      return a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
    };
    return [
      'MOUNTAIN CAR (linear VFA + tile coding)',
      '─────────────────────────────────────────',
      `  Episodes:               ${r.rewardHistory.length}`,
      `  Mean return (last 20):  ${meanLast(r.rewardHistory, 20).toFixed(2)}`,
      `  Mean length (last 20):  ${meanLast(r.lengthHistory, 20).toFixed(1)}`,
      `  Mean |TD-err| (last 20):${meanLast(r.tdErrorHistory, 20).toFixed(4)}`,
      `  Greedy from x=-0.5:     ${r.greedySolves ? `solves in ${r.greedyEpisodeLength} steps` : `does NOT solve in ${r.greedyEpisodeLength} steps`}`,
      `  ‖θ‖₂:                   ${r.thetaNorm.toFixed(2)}`,
    ].join('\n');
  },
});

// -----------------------------------------------------------------------------
// 3. tiger-pomdp
// -----------------------------------------------------------------------------

const tigerSchema: ParamSchema = {
  kind: 'object',
  description: 'Cassandra-Kaelbling-Littman 1994 Tiger problem with QMDP / 1-step look-ahead.',
  fields: {
    listenAccuracy: {kind: 'number', min: 0.5, max: 1, default: 0.85},
    openGood: {kind: 'number', default: 10},
    openBad: {kind: 'number', default: -100},
    listenCost: {kind: 'number', default: -1},
    discount: {kind: 'number', min: 0, max: 1, default: 0.95},
    solver: {kind: 'string', enum: ['qmdp', 'one-step-lookahead'], default: 'one-step-lookahead'},
    numSteps: {kind: 'number', integer: true, min: 1, default: 50},
    seed: {kind: 'number', integer: true, default: 1},
  },
  required: [],
};

registerModel<TigerOpts & {solver: 'qmdp' | 'one-step-lookahead'; numSteps: number; seed: number}, TigerSimResult>({
  id: 'tiger-pomdp',
  description: 'Tiger POMDP — belief-state planning under partial observability.',
  schema: tigerSchema,
  run(p) {
    return simulateTiger({
      spec: buildTigerSpec(p),
      solver: p.solver, numSteps: p.numSteps, seed: p.seed,
    });
  },
  summarize(r, p) {
    const actionNames = ['LISTEN', 'OPEN-LEFT', 'OPEN-RIGHT'];
    return [
      'TIGER POMDP',
      '───────────────────────────────',
      `  Solver:       ${p.solver}`,
      `  Steps:        ${r.steps}`,
      `  Total return: ${r.totalReturn.toFixed(2)}`,
      `  # opens:      ${r.numOpens}`,
      `  # bad opens:  ${r.numBadOpens}  (catastrophic open of tiger door)`,
      `  Action mix:   ${countActions(r.actions, actionNames)}`,
    ].join('\n');
  },
});

function countActions(actions: number[], names: string[]): string {
  const counts = new Array(names.length).fill(0);
  for (const a of actions) counts[a] += 1;
  return names.map((n, i) => `${n}=${counts[i]}`).join(', ');
}

// -----------------------------------------------------------------------------
// 4. grid-localization-pomdp
// -----------------------------------------------------------------------------

const gridLocalizationSchema: ParamSchema = {
  kind: 'object',
  description: '2D hidden-target localization POMDP with row/column scans and inspect actions.',
  fields: {
    width: {kind: 'number', integer: true, min: 2, max: 8, default: 3},
    height: {kind: 'number', integer: true, min: 2, max: 8, default: 3},
    horizon: {kind: 'number', integer: true, min: 0, max: 6, default: 3},
    numSteps: {kind: 'number', integer: true, min: 1, max: 100, default: 8},
    seed: {kind: 'number', integer: true, default: 1},
    hiddenTarget: {kind: 'array', items: {kind: 'number', integer: true}, minLength: 2, maxLength: 2},
    initialBelief: {kind: 'array', items: {kind: 'number', min: 0, max: 1}},
    scanAccuracy: {kind: 'number', min: 0.5, max: 1, default: 0.9},
    inspectAccuracy: {kind: 'number', min: 0.5, max: 1, default: 0.99},
    scanCost: {kind: 'number', default: -0.2},
    inspectCorrectReward: {kind: 'number', default: 20},
    inspectWrongPenalty: {kind: 'number', default: -12},
    discount: {kind: 'number', min: 0, max: 1, default: 0.95},
  },
  required: [],
};

registerModel<GridLocalizationParams, GridLocalizationResult>({
  id: 'grid-localization-pomdp',
  description: 'Multi-dimensional POMDP: localize a hidden target on a 2D grid by belief lookahead.',
  schema: gridLocalizationSchema,
  run(p) { return runGridLocalizationPOMDP(normalizeGridLocalizationParams(p)); },
  summarize(r) {
    const first = r.trace[0];
    const last = r.trace[r.trace.length - 1];
    return [
      'GRID LOCALIZATION POMDP',
      '───────────────────────────────',
      `  State space:    ${r.params.width} x ${r.params.height} = ${r.stateSpace.numStates} hidden cells`,
      `  Planner:        belief lookahead horizon=${r.params.horizon}`,
      `  Hidden target:  (${r.params.hiddenTarget[0]}, ${r.params.hiddenTarget[1]})`,
      `  First action:   ${first ? first.action.label + ' -> ' + first.observation : 'n/a'}`,
      `  Found target:   ${r.found ? `YES at step ${r.foundAtStep}` : 'no'}`,
      `  Entropy:        ${first ? first.entropy.toFixed(3) : 'n/a'} -> ${last ? last.entropy.toFixed(3) : 'n/a'}`,
      `  P(hidden):      ${last ? last.hiddenProbability.toFixed(3) : 'n/a'}`,
      `  Total return:   ${r.totalReturn.toFixed(3)}`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['step,action,observation,mode_x,mode_y,mode_probability,hidden_probability,entropy,found'];
    for (const row of r.trace) {
      lines.push(csvRow([
        row.step,
        row.action.label,
        row.observation,
        row.mode[0],
        row.mode[1],
        row.modeProbability,
        row.hiddenProbability,
        row.entropy,
        row.found,
      ]));
    }
    writeCsvLines(csvPath, lines);
  },
  examples: [{name: '3x3 hidden target search', spec: {$schema: 'des/model-spec/v1', model: 'grid-localization-pomdp', parameters: {
    width: 3,
    height: 3,
    horizon: 3,
    numSteps: 8,
    seed: 7,
    hiddenTarget: [2, 1],
    scanAccuracy: 1,
    inspectAccuracy: 1,
  }, runtime: {outputs: {csv: 'out/grid-localization-pomdp.csv', summary: 'out/grid-localization-pomdp.summary.json'}}}}],
});

function normalizeGridLocalizationParams(p: GridLocalizationParams): GridLocalizationParams {
  const hiddenTarget = p.hiddenTarget && p.hiddenTarget.length === 2
    ? [p.hiddenTarget[0], p.hiddenTarget[1]] as [number, number]
    : undefined;
  const initialBelief = p.initialBelief && p.initialBelief.length > 0 ? p.initialBelief : undefined;
  return {...p, hiddenTarget, initialBelief};
}

// -----------------------------------------------------------------------------
// 5. four-rooms-smdp
// -----------------------------------------------------------------------------

const fourRoomsSchema: ParamSchema = {
  kind: 'object',
  description: 'Four-Rooms gridworld with hallway options (Sutton, Precup, Singh 1999).',
  fields: {
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 200},
    maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 5000},
    alpha: {kind: 'number', min: 0, default: 0.25},
    gamma: {kind: 'number', min: 0, max: 1, default: 0.99},
    epsilon: {kind: 'number', min: 0, max: 1, default: 0.1},
    epsilonDecay: {kind: 'number', min: 0, max: 1, default: 1},
    epsilonMin: {kind: 'number', min: 0, max: 1, default: 0.01},
    seed: {kind: 'number', integer: true, default: 1},
    slip: {kind: 'number', min: 0, max: 1, default: 0},
    includePrimitive: {kind: 'boolean', default: true},
    initQ: {kind: 'number', default: 1.0},
  },
  required: [],
};

registerModel<FourRoomsTrainOpts, FourRoomsResult>({
  id: 'four-rooms-smdp',
  description: 'Four-Rooms with hallway OPTIONS — SMDP Q-learning over temporally extended actions.',
  schema: fourRoomsSchema,
  run(p) { return runFourRoomsSMDP(p as FourRoomsTrainOpts); },
  summarize(r) {
    const lastL = r.lengthHistory.slice(-Math.min(20, r.lengthHistory.length));
    const meanL = lastL.reduce((s, x) => s + x, 0) / Math.max(1, lastL.length);
    return [
      'FOUR ROOMS (Semi-MDP, options)',
      '────────────────────────────────',
      `  Episodes trained:        ${r.rewardHistory.length}`,
      `  Mean episode len (last20): ${meanL.toFixed(1)}`,
      `  Greedy reaches goal:     ${r.greedyReachedGoal ? `YES in ${r.greedyEpisodeLength} steps` : 'no'}`,
      `  Optimal-path lower bound: 20 steps (no walls / one path)`,
    ].join('\n');
  },
});

// -----------------------------------------------------------------------------
// 6. actor-critic-grid
// -----------------------------------------------------------------------------

const actorCriticSchema: ParamSchema = {
  kind: 'object',
  description: 'One-step tabular actor-critic on GridWorld.',
  fields: {
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 1000},
    maxStepsPerEpisode: {kind: 'number', integer: true, min: 1, default: 100},
    alphaV: {kind: 'number', min: 0, default: 0.1},
    alphaP: {kind: 'number', min: 0, default: 0.05},
    gamma: {kind: 'number', min: 0, max: 1, default: 0.95},
    entropyCoef: {kind: 'number', default: 0},
    seed: {kind: 'number', integer: true, default: 1},
    width: {kind: 'number', integer: true, min: 2, default: 4},
    height: {kind: 'number', integer: true, min: 2, default: 4},
  },
  required: [],
};

registerModel<ActorCriticTrainOpts, ActorCriticResult>({
  id: 'actor-critic-grid',
  description: 'One-step Actor-Critic with tabular softmax policy + tabular V on GridWorld.',
  schema: actorCriticSchema,
  run(p) { return runActorCriticGridworld(p as ActorCriticTrainOpts); },
  summarize(r) {
    const lastR = r.rewardHistory.slice(-Math.min(20, r.rewardHistory.length));
    const meanR = lastR.reduce((s, x) => s + x, 0) / Math.max(1, lastR.length);
    return [
      'ACTOR-CRITIC (tabular) on GridWorld',
      '─────────────────────────────────────',
      `  Episodes:                ${r.rewardHistory.length}`,
      `  Mean return (last 20):   ${meanR.toFixed(2)}`,
      `  V(start) (critic):       ${r.Vstart.toFixed(3)}`,
      `  Greedy from start:       ${r.greedyReached ? `reaches goal in ${r.greedyLen} steps` : `fails (len=${r.greedyLen})`}`,
    ].join('\n');
  },
});

// -----------------------------------------------------------------------------
// 7. blackjack-mc
// -----------------------------------------------------------------------------

const blackjackSchema: ParamSchema = {
  kind: 'object',
  description: 'Sutton & Barto §5.1 Blackjack with first-visit Monte Carlo control.',
  fields: {
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 50_000},
    seed: {kind: 'number', integer: true, default: 1},
    epsilon: {kind: 'number', min: 0, max: 1, default: 0.1},
    epsilonDecay: {kind: 'number', min: 0, max: 1, default: 1},
    epsilonMin: {kind: 'number', min: 0, max: 1, default: 0.05},
    firstVisit: {kind: 'boolean', default: true},
    gamma: {kind: 'number', min: 0, max: 1, default: 1.0},
    evalEpisodes: {kind: 'number', integer: true, min: 1, default: 5000},
  },
  required: [],
};

registerModel<BlackjackTrainOpts & {evalEpisodes?: number}, BlackjackResult>({
  id: 'blackjack-mc',
  description: 'Blackjack solved by on-policy first-visit Monte Carlo control (Sutton & Barto §5.1).',
  schema: blackjackSchema,
  run(p) { return runBlackjackMC(p); },
  summarize(r) {
    return [
      'BLACKJACK MC',
      '──────────────────────',
      `  Cells visited:          ${r.visitedCells} / 400`,
      `  Greedy mean return:     ${r.greedyMeanReturn.toFixed(3)}  (theoretical optimum ≈ -0.04)`,
      `  Baseline (stick≥20):    ${r.baselineMeanReturn.toFixed(3)}  (≈ -0.27)`,
      `  Improvement over base:  ${(r.greedyMeanReturn - r.baselineMeanReturn).toFixed(3)}`,
    ].join('\n');
  },
});

// -----------------------------------------------------------------------------
// 8. stag-hunt
// -----------------------------------------------------------------------------

const stagHuntSchema: ParamSchema = {
  kind: 'object',
  description: 'Stag Hunt coordination game with two independent Q-learners (Tan 1993 IQL).',
  fields: {
    numEpisodes: {kind: 'number', integer: true, min: 1, default: 5000},
    alpha: {kind: 'number', min: 0, default: 0.05},
    gamma: {kind: 'number', min: 0, max: 1, default: 0},
    epsilon: {kind: 'number', min: 0, max: 1, default: 0.2},
    epsilonDecay: {kind: 'number', min: 0, max: 1, default: 0.999},
    epsilonMin: {kind: 'number', min: 0, max: 1, default: 0.01},
    seed: {kind: 'number', integer: true, default: 1},
  },
  required: [],
};

registerModel<StagHuntOpts, StagHuntResult>({
  id: 'stag-hunt',
  description: 'Stag Hunt — two independent Q-learners coordinate on a Nash equilibrium.',
  schema: stagHuntSchema,
  run(p) { return runStagHunt(p); },
  summarize(r) {
    const acts = ['STAG', 'HARE'];
    return [
      'STAG HUNT (Independent Q-Learning, 2 agents)',
      '──────────────────────────────────────────────',
      `  Episodes:               ${r.rewardHistory.length}`,
      `  Recent mean return:     [${r.recentMeanReturn[0].toFixed(2)}, ${r.recentMeanReturn[1].toFixed(2)}]`,
      `  Final greedy actions:   [${acts[r.finalJointAction[0]]}, ${acts[r.finalJointAction[1]]}]`,
      `  Coordinated on Stag?    ${r.coordinatedOnStag ? 'YES (payoff-dominant)' : 'no'}`,
      `  Coordinated on Hare?    ${r.coordinatedOnHare ? 'YES (risk-dominant)'   : 'no'}`,
    ].join('\n');
  },
});

// -----------------------------------------------------------------------------
// 9. double-integrator-lqr
// -----------------------------------------------------------------------------

const lqrSchema: ParamSchema = {
  kind: 'object',
  description: 'Discrete-time LQR on a double integrator, computed by Riccati iteration.',
  fields: {
    dt: {kind: 'number', min: 1e-6, default: 0.1},
    qPos: {kind: 'number', min: 0, default: 1},
    qVel: {kind: 'number', min: 0, default: 0.1},
    rU: {kind: 'number', min: 1e-6, default: 0.01},
    noiseStd: {kind: 'number', min: 0, default: 0.05},
    x0: {kind: 'array', items: {kind: 'number'}, minLength: 2, maxLength: 2},
    numSteps: {kind: 'number', integer: true, min: 1, default: 100},
    uSat: {kind: 'number', min: 0, default: Infinity},
    gamma: {kind: 'number', min: 0, max: 1, default: 1},
    seed: {kind: 'number', integer: true, default: 1},
  },
  required: [],
};

registerModel<DoubleIntegratorOpts, DoubleIntegratorResult>({
  id: 'double-integrator-lqr',
  description: 'Discrete-time LQR feedback control of a double integrator (Riccati DARE).',
  schema: lqrSchema,
  run(p) {
    // The schema may have decoded `x0: [number, number]` as `number[]`
    // — coerce with a length check so the LQR runner gets the right type.
    return runDoubleIntegratorLQR({...p, x0: numberPair((p as DoubleIntegratorOpts).x0, [3, 0], 'x0')});
  },
  summarize(r) {
    const finalState = r.trajectory[r.trajectory.length - 1];
    return [
      'DOUBLE-INTEGRATOR LQR',
      '────────────────────────────────────',
      `  Riccati iters:           ${r.riccatiIters}  (residual ${r.riccatiResidual.toExponential(2)})`,
      `  Optimal feedback K:      [${r.K[0].map(x => x.toFixed(3)).join(', ')}]`,
      `  Cost-to-go (DARE):       ${r.riccatiCostFromX0.toFixed(3)}`,
      `  Realised cumulative cost ${r.totalCost.toFixed(3)}`,
      `  Final state:             [${finalState[0].toFixed(3)}, ${finalState[1].toFixed(3)}]`,
    ].join('\n');
  },
  writeCsv(r, csvPath) {
    const lines = ['t,pos,vel,u,stage_cost'];
    for (let t = 0; t < r.controls.length; t++) {
      lines.push(csvRow([t, r.trajectory[t][0].toFixed(6), r.trajectory[t][1].toFixed(6), r.controls[t].toFixed(6), r.stageCosts[t].toFixed(6)]));
    }
    writeCsvLines(csvPath, lines);
  },
});
