// RUST MIGRATION: Port file-for-file to `tests/preconditions_test.rs` as integration coverage for pre-run guards and model precondition errors.
// Test-port notes: translate error-path checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and `matches!`; keep seeded fixtures deterministic.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/preconditions_test.rs   (integration test crate)
// 1:1 file move. Verifies pre-run guards fire on bad inputs across many models.
// Keep the doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - the central pattern is "bad input throws PreconditionError naming a param"
//     -> map PreconditionError to a Result::Err (assert the message), or use
//     `#[should_panic(expected = "<param>")]` if the Rust port panics.
//   - Preconditions.* edge cases (NaN, non-PSD, prob vector ≠ 1) -> assert on the
//     guard's Result; float checks use approx tolerances.
// =============================================================================

// =============================================================================
// test/preconditions-test.ts — verifies that the new pre-run guards
// (Preconditions.* + each model's `assertPreconditions()` override + each
// public `runX(opts)` parameter validator) actually FIRE on bad inputs.
//
// The strategy is:
//   1. For each new model, run it with at least one obviously-broken
//      parameter and assert that the call throws a `PreconditionError`
//      whose message names the offending parameter.
//   2. Also test the underlying `Preconditions.*` utility on edge cases
//      (NaN, negative, divide-by-zero, non-PSD, probability vector that
//      doesn't sum to 1, etc.).
// =============================================================================

import {
  Preconditions, PreconditionError,
} from '../general/des-base/preconditions';

import {runPontryaginBangBang} from '../general/pontryagin-bang-bang';
import {runRadarTracking} from '../general/kalman-filter';
import {runSlidingMode} from '../general/sliding-mode-control';
import {runMRAC} from '../general/mrac';
import {runFeedbackLinearization} from '../general/feedback-linearization';
import {runMPCDoubleIntegrator} from '../general/mpc-double-integrator';
import {solveInventoryDP} from '../general/inventory-dp';
import {runMountainCar} from '../general/mountain-car';
import {runFourRoomsSMDP} from '../general/four-rooms';
import {runActorCriticGridworld} from '../general/actor-critic-gridworld';
import {runBlackjackMC} from '../general/blackjack';
import {runStagHunt} from '../general/stag-hunt';
import {simulateTiger, buildTigerSpec} from '../general/tiger-pomdp';
import {runDoubleIntegratorLQR} from '../general/double-integrator-lqr';
import {runTempControl} from '../general/temp-control';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

/** Assert that `fn` throws `PreconditionError` whose message contains
 *  `paramFragment` (case-insensitive substring match). */
function expectPreconditionError(name: string, fn: () => void, paramFragment: string): void {
  let err: unknown = null;
  try { fn(); } catch (e) { err = e; }
  if (!err) {
    check(name, false, `expected throw, got success`);
    return;
  }
  if (!(err instanceof PreconditionError)) {
    check(name, false, `expected PreconditionError, got ${(err as Error).constructor.name}: ${(err as Error).message}`);
    return;
  }
  if (!err.message.toLowerCase().includes(paramFragment.toLowerCase())) {
    check(name, false, `error did not mention "${paramFragment}": ${err.message}`);
    return;
  }
  check(name, true, `correctly threw on ${paramFragment}`);
}

// =============================================================================
// 1. PRECONDITIONS UTILITY (low-level guards)
// =============================================================================

console.log('\n— Preconditions.* low-level guards —');
{
  expectPreconditionError('finite catches NaN',
    () => Preconditions.finite('m', 'x', NaN), 'x');
  expectPreconditionError('finite catches +Inf',
    () => Preconditions.finite('m', 'x', Infinity), 'x');
  expectPreconditionError('positive catches 0',
    () => Preconditions.positive('m', 'x', 0), 'x');
  expectPreconditionError('positive catches negative',
    () => Preconditions.positive('m', 'x', -0.5), 'x');
  expectPreconditionError('inRange catches above bound',
    () => Preconditions.inRange('m', 'x', 1.2, 0, 1), 'x');
  expectPreconditionError('inRange catches below bound',
    () => Preconditions.inRange('m', 'x', -0.2, 0, 1), 'x');
  expectPreconditionError('integer catches float',
    () => Preconditions.integer('m', 'k', 3.7), 'k');
  expectPreconditionError('probabilityVector catches sum 0.99',
    () => Preconditions.probabilityVector('m', 'p', [0.5, 0.4, 0.09]), 'p');
  expectPreconditionError('probabilityVector catches negative entry',
    () => Preconditions.probabilityVector('m', 'p', [0.5, -0.1, 0.6]), 'p');
  expectPreconditionError('symmetricMatrix catches asymmetry',
    () => Preconditions.symmetricMatrix('m', 'M', [[1, 2], [3, 4]]), 'M');
  expectPreconditionError('positiveDefiniteCholesky catches zero',
    () => Preconditions.positiveDefiniteCholesky('m', 'M', [[0, 0], [0, 1]]), 'M');
  expectPreconditionError('positiveDefiniteCholesky catches indefinite',
    () => Preconditions.positiveDefiniteCholesky('m', 'M', [[1, 2], [2, 1]]), 'M');
  expectPreconditionError('notDivByZero catches 0',
    () => Preconditions.notDivByZero('m', 'd', 0), 'd');
  expectPreconditionError('lengthEq catches mismatch',
    () => Preconditions.lengthEq('m', 'arr', [1, 2], 3), 'arr');
  expectPreconditionError('integerInRange catches out of range',
    () => Preconditions.integerInRange('m', 'k', 11, 0, 10), 'k');

  // Positive cases (should NOT throw).
  let ok = true;
  try {
    Preconditions.finite('m', 'x', 1.5);
    Preconditions.positive('m', 'x', 0.001);
    Preconditions.inRange('m', 'x', 0.5, 0, 1);
    Preconditions.probabilityVector('m', 'p', [0.4, 0.3, 0.3]);
    Preconditions.symmetricMatrix('m', 'M', [[1, 2], [2, 3]]);
    Preconditions.positiveDefiniteCholesky('m', 'M', [[2, 1], [1, 2]]);
    Preconditions.notDivByZero('m', 'd', 0.5);
  } catch (e) { ok = false; }
  check('valid inputs do not throw', ok);
}

// =============================================================================
// 2. ENTITY-BASED CONTROL MODELS
// =============================================================================

console.log('\n— Pontryagin bang-bang preconditions —');
{
  expectPreconditionError('rejects uMax = 0',
    () => runPontryaginBangBang({uMax: 0}), 'uMax');
  expectPreconditionError('rejects dt = 0',
    () => runPontryaginBangBang({dt: 0}), 'dt');
  expectPreconditionError('rejects negative numSteps',
    () => runPontryaginBangBang({numSteps: -1}), 'numSteps');
  expectPreconditionError('rejects NaN x0',
    () => runPontryaginBangBang({x0: [NaN, 0]}), 'x0');
}

console.log('\n— Kalman filter preconditions —');
{
  expectPreconditionError('rejects measNoiseStd = 0 (R singular)',
    () => runRadarTracking({measNoiseStd: 0}), 'measNoiseStd');
  expectPreconditionError('rejects negative procNoiseStd',
    () => runRadarTracking({procNoiseStd: -0.1}), 'procNoiseStd');
  expectPreconditionError('rejects dt = 0',
    () => runRadarTracking({dt: 0}), 'dt');
}

console.log('\n— Sliding-mode preconditions —');
{
  // SMC reaching condition: η > D required.
  expectPreconditionError('rejects eta <= disturbanceAmp (reaching condition)',
    () => runSlidingMode({eta: 0.5, disturbanceAmp: 1, numSteps: 10}), 'eta');
  expectPreconditionError('rejects lambda = 0',
    () => runSlidingMode({lambda: 0}), 'lambda');
  expectPreconditionError('rejects boundary = 0',
    () => runSlidingMode({boundary: 0}), 'boundary');
  expectPreconditionError('rejects unknown disturbanceType',
    () => runSlidingMode({disturbanceType: 'tri' as any}), 'disturbanceType');
}

console.log('\n— MRAC preconditions —');
{
  expectPreconditionError('rejects b <= 0 (sign-known assumption)',
    () => runMRAC({b: 0}), 'b');
  expectPreconditionError('rejects am >= 0 (reference must be Hurwitz)',
    () => runMRAC({am: 0.1}), 'am');
  expectPreconditionError('rejects gamma*dt > 1 (numerical stability)',
    () => runMRAC({gamma: 1000, dt: 0.01, numSteps: 10}), 'gamma*dt');
}

console.log('\n— Feedback linearization preconditions —');
{
  expectPreconditionError('rejects mass m = 0 (divide by zero in 1/(m·l²))',
    () => runFeedbackLinearization({params: {m: 0}}), 'params.m');
  expectPreconditionError('rejects length l = 0',
    () => runFeedbackLinearization({params: {l: 0}}), 'params.l');
  expectPreconditionError('rejects negative kp',
    () => runFeedbackLinearization({kp: -1}), 'kp');
}

console.log('\n— MPC preconditions —');
{
  expectPreconditionError('rejects R = 0 (gradient ill-posed)',
    () => runMPCDoubleIntegrator({R: 0}), 'R');
  expectPreconditionError('rejects N = 0',
    () => runMPCDoubleIntegrator({N: 0}), 'N (horizon)');
  expectPreconditionError('rejects uMax = 0',
    () => runMPCDoubleIntegrator({uMax: 0}), 'uMax');
}

// =============================================================================
// 3. MDP-ADJACENT MODELS
// =============================================================================

console.log('\n— Inventory DP preconditions —');
{
  const goodPmf = [0.3, 0.3, 0.4];
  const baseProblem = {
    horizon: 3, S_max: 5, demandPmf: goodPmf,
    price: 8, cost: 3, fixedCost: 1, holdCost: 0.5, stockoutCost: 5,
    salvageValue: 0, discount: 1, initialInventory: 0,
  };
  expectPreconditionError('rejects horizon = 0',
    () => solveInventoryDP({...baseProblem, horizon: 0}), 'horizon');
  expectPreconditionError('rejects PMF that does not sum to 1',
    () => solveInventoryDP({...baseProblem, demandPmf: [0.2, 0.3, 0.4]}), 'demandPmf');
  expectPreconditionError('rejects negative price',
    () => solveInventoryDP({...baseProblem, price: -1}), 'price');
  expectPreconditionError('rejects discount > 1',
    () => solveInventoryDP({...baseProblem, discount: 1.5}), 'discount');
  expectPreconditionError('rejects initialInventory > S_max',
    () => solveInventoryDP({...baseProblem, initialInventory: 999}), 'initialInventory');
  // Positive control: valid inputs run successfully.
  let ok = true;
  try { solveInventoryDP(baseProblem); } catch (e) { ok = false; console.log((e as Error).message); }
  check('valid inputs run successfully', ok);
}

console.log('\n— Mountain Car preconditions —');
{
  expectPreconditionError('rejects numEpisodes = 0',
    () => runMountainCar({numEpisodes: 0}), 'numEpisodes');
  expectPreconditionError('rejects alpha = 0',
    () => runMountainCar({numEpisodes: 1, alpha: 0}), 'alpha');
  expectPreconditionError('rejects gamma > 1',
    () => runMountainCar({numEpisodes: 1, gamma: 1.5}), 'gamma');
  expectPreconditionError('rejects epsilon > 1',
    () => runMountainCar({numEpisodes: 1, epsilon: 2}), 'epsilon');
}

console.log('\n— Four Rooms preconditions —');
{
  expectPreconditionError('rejects numEpisodes = 0',
    () => runFourRoomsSMDP({numEpisodes: 0}), 'numEpisodes');
  expectPreconditionError('rejects gamma > 1',
    () => runFourRoomsSMDP({numEpisodes: 1, gamma: 1.5}), 'gamma');
  expectPreconditionError('rejects slip > 1',
    () => runFourRoomsSMDP({numEpisodes: 1, slip: 1.5}), 'slip');
}

console.log('\n— Actor-Critic preconditions —');
{
  expectPreconditionError('rejects numEpisodes = 0',
    () => runActorCriticGridworld({numEpisodes: 0}), 'numEpisodes');
  expectPreconditionError('rejects alphaV = 0',
    () => runActorCriticGridworld({numEpisodes: 1, alphaV: 0}), 'alphaV');
  expectPreconditionError('rejects negative width',
    () => runActorCriticGridworld({numEpisodes: 1, width: -1}), 'width');
}

console.log('\n— Blackjack preconditions —');
{
  expectPreconditionError('rejects numEpisodes = 0',
    () => runBlackjackMC({numEpisodes: 0}), 'numEpisodes');
  expectPreconditionError('rejects gamma > 1',
    () => runBlackjackMC({numEpisodes: 1, gamma: 2}), 'gamma');
}

console.log('\n— Stag Hunt preconditions —');
{
  expectPreconditionError('rejects numEpisodes = 0',
    () => runStagHunt({numEpisodes: 0}), 'numEpisodes');
  expectPreconditionError('rejects negative alpha',
    () => runStagHunt({numEpisodes: 1, alpha: -0.1}), 'alpha');
}

console.log('\n— Tiger POMDP preconditions —');
{
  const spec = buildTigerSpec();
  expectPreconditionError('rejects numSteps = 0',
    () => simulateTiger({spec, solver: 'qmdp', numSteps: 0}), 'numSteps');
  expectPreconditionError('rejects unknown solver',
    () => simulateTiger({spec, solver: 'mcts' as any, numSteps: 5}), 'solver');
  expectPreconditionError('rejects belief that does not sum to 1',
    () => simulateTiger({spec, solver: 'qmdp', numSteps: 5, initialBelief: [0.3, 0.3]}), 'initialBelief');
}

console.log('\n— Double-integrator LQR preconditions —');
{
  expectPreconditionError('rejects rU = 0 (R singular, DARE blows up)',
    () => runDoubleIntegratorLQR({rU: 0}), 'rU');
  expectPreconditionError('rejects negative qPos',
    () => runDoubleIntegratorLQR({qPos: -1}), 'qPos');
  expectPreconditionError('rejects dt = 0',
    () => runDoubleIntegratorLQR({dt: 0}), 'dt');
  expectPreconditionError('rejects gamma = 0',
    () => runDoubleIntegratorLQR({gamma: 0}), 'gamma');
}

// =============================================================================
// 4. TEMP CONTROL
// =============================================================================

console.log('\n— Temp control preconditions —');
{
  const baseCfg: any = {
    dt_min: 1, duration_h: 1, T_target: 20, cost_per_kWh: 1, comfort_penalty: 1,
    controller: {kind: 'pid', Kp: 100, Ki: 5, Kd: 10},
  };
  expectPreconditionError('rejects dt_min = 0',
    () => runTempControl({...baseCfg, dt_min: 0}), 'dt_min');
  expectPreconditionError('rejects duration_h = 0',
    () => runTempControl({...baseCfg, duration_h: 0}), 'duration_h');
  expectPreconditionError('rejects negative cost_per_kWh',
    () => runTempControl({...baseCfg, cost_per_kWh: -1}), 'cost_per_kWh');
  expectPreconditionError('rejects negative sensorNoiseStd',
    () => runTempControl({...baseCfg, sensorNoiseStd: -0.1}), 'sensorNoiseStd');
}

// =============================================================================
// SUMMARY
// =============================================================================

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`preconditions-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
