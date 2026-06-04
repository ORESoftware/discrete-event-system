#!/usr/bin/env ts-node
'use strict';

import {DiscreteBelief} from '../general/belief';
import {
  BeliefLookaheadSolver,
  beliefUpdate,
  mdpValueIteration,
  pomdpExactFiniteHorizon,
} from '../general/pomdp';
import {buildTigerSpec} from '../general/tiger-pomdp';
import {MDPSpec, valueIteration} from '../general/value-iteration';
import {
  HighriseElevatorConfig,
  ScheduledArrival,
  optimizeHighriseDispatchMDP,
  runHighriseElevators,
} from '../main-elevator-highrise';

interface CheckRow {
  name: string;
  passed: boolean;
  detail?: string;
}

const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function expectThrows(name: string, fn: () => unknown, needle?: string): void {
  try {
    fn();
    check(name, false, 'expected throw');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    check(name, needle === undefined || message.includes(needle), message);
  }
}

function tinyMDP(): MDPSpec {
  return {
    numStates: 2,
    numActions: () => 1,
    outcomes: s => s === 0
      ? [{prob: 1, reward: 1, nextState: 1}]
      : [{prob: 1, reward: 0, nextState: 1}],
    isTerminal: s => s === 1,
  };
}

console.log('\n- generic MDP value-iteration hardening -');
{
  const valid = valueIteration(tinyMDP(), {gamma: 0.9, tol: 1e-10, randomTieBreak: false});
  check('valid tiny MDP still solves', valid.policy[0] === 0 && Number.isFinite(valid.V[0]));
  expectThrows('rejects gamma outside [0, 1]', () => valueIteration(tinyMDP(), {gamma: 1.2}), 'gamma');
  expectThrows('rejects negative outcome probability', () => valueIteration({
    ...tinyMDP(),
    outcomes: () => [{prob: -0.1, reward: 0, nextState: 0}],
  }), 'prob');
  expectThrows('rejects outcome probabilities that do not sum to one', () => valueIteration({
    ...tinyMDP(),
    outcomes: () => [{prob: 0.5, reward: 0, nextState: 0}],
  }), 'sum to 1');
  expectThrows('rejects nextState outside the state space', () => valueIteration({
    ...tinyMDP(),
    outcomes: () => [{prob: 1, reward: 0, nextState: 2}],
  }), 'nextState');
  expectThrows('rejects nonterminal states with no legal outcomes', () => valueIteration({
    numStates: 1,
    numActions: () => 1,
    outcomes: () => [],
  }), 'legal');
}

console.log('\n- generic POMDP and belief hardening -');
{
  const tiger = buildTigerSpec();
  const updated = beliefUpdate(tiger, new DiscreteBelief(tiger.states, [0.5, 0.5]), 0, 0);
  const total = updated.weights.reduce((s, x) => s + x, 0);
  check('valid Tiger belief update remains normalized', Math.abs(total - 1) < 1e-9);
  expectThrows('rejects negative belief prior', () => new DiscreteBelief(['a', 'b'], [1, -0.1]), 'prior');
  expectThrows('rejects malformed POMDP transition rows', () => mdpValueIteration({
    ...tiger,
    transition: () => [0.7, 0.7],
  }), 'sum to 1');
  expectThrows('rejects malformed POMDP observation rows', () => beliefUpdate({
    ...tiger,
    observation: () => [1],
  }, new DiscreteBelief(tiger.states), 0, 0), 'length');
  expectThrows('rejects invalid lookahead horizon', () => new BeliefLookaheadSolver(tiger, {horizon: -1}), 'horizon');
  expectThrows('caps exact POMDP finite-horizon growth', () => pomdpExactFiniteHorizon(tiger, 7), 'horizon');
}

console.log('\n- highrise elevator MDP hardening -');
{
  const callOnly = optimizeHighriseDispatchMDP('call-only');
  const destination = optimizeHighriseDispatchMDP('destination-dispatch');
  check('call-only MDP state count matches encoder', callOnly.numStates === 24, `states=${callOnly.numStates}`);
  check('destination-dispatch MDP state count matches encoder', destination.numStates === 648, `states=${destination.numStates}`);
  check('highrise policies contain only legal actions',
    [...Array.from(callOnly.policy), ...Array.from(destination.policy)]
      .every(a => Number.isInteger(a) && a >= 0 && a < destination.actions.length));

  const cfg: HighriseElevatorConfig = {
    nFloors: 12,
    nElevators: 3,
    capacity: 6,
    floorTravelTime: 0.8,
    serviceTime: 1,
    arrivalRate: 0.1,
    simT: 20,
    drainT: 80,
    stepSize: 0.5,
    seed: 3,
    localSensorRadius: 5,
    urgentWaitThreshold: 15,
  };
  const schedule: ScheduledArrival[] = [
    {t: 0, fromFloor: 0, toFloor: 9},
    {t: 1, fromFloor: 0, toFloor: 8},
    {t: 2, fromFloor: 1, toFloor: 9},
    {t: 3, fromFloor: 8, toFloor: 0},
    {t: 4, fromFloor: 7, toFloor: 0},
    {t: 5, fromFloor: 0, toFloor: 6},
    {t: 6, fromFloor: 5, toFloor: 11},
    {t: 8, fromFloor: 11, toFloor: 0},
  ];
  const tuned = runHighriseElevators(cfg, 'mdp-tuned', schedule, {
    authority: 'central',
    mdpTuning: destination,
    recordEveryTicks: 10,
  }).result;
  check('highrise MDP run records learned decisions',
    (tuned.mdpRun?.totalDecisions ?? 0) > 0,
    `decisions=${tuned.mdpRun?.totalDecisions ?? 0}`);
  check('highrise MDP diagnostics use known action labels',
    tuned.mdpRun?.actionCounts.every(row => destination.actionLabels.includes(row.action)) ?? false);
  expectThrows('MDP policy requires explicit tuning', () => runHighriseElevators(cfg, 'mdp-tuned', schedule, {
    authority: 'central',
    recordEveryTicks: 10,
  }), 'requires matching MDP tuning');
  expectThrows('MDP policy rejects mismatched observability tuning', () => runHighriseElevators(cfg, 'mdp-tuned', schedule, {
    authority: 'central',
    mdpTuning: callOnly,
    recordEveryTicks: 10,
  }), 'observability');
  expectThrows('highrise schedule must be sorted by time', () => runHighriseElevators(cfg, 'lowest-total-time', [
    schedule[1],
    schedule[0],
  ], {authority: 'central', recordEveryTicks: 10}), 'nondecreasing');
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`mdp-pomdp-hardening-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
