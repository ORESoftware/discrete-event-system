'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/optimal_control_test.rs   (integration test crate)
// 1:1 file move. End-to-end tests of the entity-based optimal-control models
// (PMP, Kalman, SMC, MRAC, ILC, feedback-lin, MPC). Keep the doc-block below.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - each algorithm's theoretical invariant is a float comparison -> the
//     `approx` crate (assert_relative_eq! / assert_abs_diff_eq!).
//   - the control-blocks entity framework -> StationaryEntity-style trait + the
//     VectorSignal token struct; no inheritance.
// =============================================================================

// =============================================================================
// test/optimal-control-test.ts — end-to-end tests for the entity-based
// optimal-control models added in this batch:
//
//   1. pontryagin-bang-bang   — Pontryagin's Maximum Principle (PMP)
//   2. kalman-filter          — linear Kalman filter (radar tracking)
//   3. sliding-mode           — robust SMC under bounded disturbance
//   4. mrac                   — model-reference adaptive control
//   5. iterative-learning-control — repeated-trial feedforward learning
//   6. feedback-linearization — nonlinear pendulum tracking
//   7. mpc-double-integrator  — MPC with hard input constraints
//
// All examples are built on the entity-based control framework
// (`des-base/control-blocks.ts`, which extends StationaryEntity-style
// `MultiDirectionalSignalEntity` blocks and AbstractMovingEntity-style
// `VectorSignal` tokens). Each test verifies the canonical theoretical
// invariant the algorithm is supposed to satisfy.
// =============================================================================

import {runPontryaginBangBang, optimalTimeDoubleIntegrator}
  from '../general/pontryagin-bang-bang';
import {runRadarTracking} from '../general/kalman-filter';
import {runSlidingMode} from '../general/sliding-mode-control';
import {runMRAC} from '../general/mrac';
import {runIterativeLearningControl} from '../general/iterative-learning-control';
import {runFeedbackLinearization} from '../general/feedback-linearization';
import {runMPCDoubleIntegrator} from '../general/mpc-double-integrator';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// =============================================================================
// 1. PONTRYAGIN BANG-BANG (TIME-OPTIMAL DOUBLE INTEGRATOR)
// =============================================================================

console.log('\n— pontryagin-bang-bang (time-optimal control via PMP) —');
{
  const r = runPontryaginBangBang({x0: [3, 0], uMax: 1, dt: 0.02, numSteps: 500, deadband: 0.1});
  check('exactly one bang-bang switch (PMP optimum)',
        r.switchCount === 1, `switches=${r.switchCount}`);
  // Closed-form: t* = 2√(|x0|/u_max) for v0 = 0.
  const tArrival = r.arrivalTick * 0.02;
  check('arrival time near closed-form t*',
        Math.abs(tArrival - r.theoreticalArrivalTime) < 0.5,
        `arrived=${tArrival.toFixed(3)}s, t*=${r.theoreticalArrivalTime.toFixed(3)}s`);
  check('|u(t)| respects bound u_max = 1',
        r.controls.every(u => Math.abs(u[0]) <= 1.0001),
        `max|u|=${Math.max(...r.controls.map(u => Math.abs(u[0]))).toFixed(4)}`);
  check('trajectory ends near origin',
        (() => { const f = r.trajectory[r.trajectory.length - 1]; return Math.abs(f[0]) + Math.abs(f[1]) < 0.05; })(),
        `final=${r.trajectory[r.trajectory.length - 1].map(x => x.toFixed(3)).join(', ')}`);
  // Closed-form formula matches: 2 √3 ≈ 3.464.
  const tForm = optimalTimeDoubleIntegrator(3, 0, 1);
  check('closed-form t* matches 2√(x0/u_max) from rest',
        Math.abs(tForm - 2 * Math.sqrt(3)) < 0.05,
        `t*=${tForm.toFixed(3)}, theoretical=${(2*Math.sqrt(3)).toFixed(3)}`);
}

// =============================================================================
// 2. KALMAN FILTER (RADAR TRACKING)
// =============================================================================

console.log('\n— kalman-filter (state estimation) —');
{
  const r = runRadarTracking({
    x0: [0, 1], dt: 0.1, numSteps: 200,
    procNoiseStd: 0.1, measNoiseStd: 1.0, P0Scale: 10, seed: 7,
  });
  check('KF beats raw sensor RMSE',
        r.rmsePos < r.rmseMeasPos,
        `rmseEst=${r.rmsePos.toFixed(3)} < rmseMeas=${r.rmseMeasPos.toFixed(3)}`);
  check('KF reduces RMSE by ≥ 50% vs raw measurement',
        r.rmsePos < 0.5 * r.rmseMeasPos,
        `reduction=${(100*(1-r.rmsePos/r.rmseMeasPos)).toFixed(1)}%`);
  check('posterior covariance shrinks (cov trace < P0 trace)',
        r.finalCovTrace < 20,
        `trace=${r.finalCovTrace.toFixed(3)}`);
  check('full trajectory has expected length',
        r.trueTrajectory.length === r.numSteps + 1 && r.estimates.length === r.numSteps);
}

// =============================================================================
// 3. SLIDING MODE (ROBUST CONTROL)
// =============================================================================

console.log('\n— sliding-mode (robust control) —');
{
  const r = runSlidingMode({
    x0: [3, 0], dt: 0.05, numSteps: 400,
    lambda: 2, eta: 3, boundary: 0.05, uBound: 5,
    disturbanceAmp: 1, disturbanceType: 'sin', seed: 1,
  });
  check('reaches sliding surface in finite time',
        r.reachingTick < 100,
        `reached at tick ${r.reachingTick}`);
  check('stays near origin under disturbance (after t = T/2)',
        r.stayedNearOrigin);
  check('final |x|+|v| small despite disturbance',
        r.finalDistanceFromOrigin < 0.5,
        `final=${r.finalDistanceFromOrigin.toFixed(3)}`);
  // Robust to a different disturbance type.
  const r2 = runSlidingMode({...{x0: [3, 0] as [number, number], dt: 0.05, numSteps: 400,
                                  lambda: 2, eta: 3, boundary: 0.05, uBound: 5,
                                  disturbanceAmp: 1}, disturbanceType: 'square', seed: 2});
  check('robust to square-wave disturbance too',
        r2.stayedNearOrigin && r2.finalDistanceFromOrigin < 0.5,
        `square: final=${r2.finalDistanceFromOrigin.toFixed(3)}, stayed=${r2.stayedNearOrigin}`);
}

// =============================================================================
// 4. MRAC (ADAPTIVE CONTROL)
// =============================================================================

console.log('\n— mrac (adaptive control) —');
{
  const r = runMRAC({
    a: 1, b: 2, am: -2, bm: 2, x0: 0, xm0: 0,
    gamma: 5, dt: 0.01, numSteps: 4000,
  });
  check('steady-state RMS tracking error < 0.05',
        r.rmsErrorSteadyState < 0.05,
        `rms=${r.rmsErrorSteadyState.toExponential(2)}`);
  check('θ_x converges near ideal θ*_x',
        Math.abs(r.finalTheta[0] - r.idealTheta[0]) < 0.2,
        `θ_x=${r.finalTheta[0].toFixed(3)}, ideal=${r.idealTheta[0].toFixed(3)}`);
  check('θ_r converges near ideal θ*_r',
        Math.abs(r.finalTheta[1] - r.idealTheta[1]) < 0.2,
        `θ_r=${r.finalTheta[1].toFixed(3)}, ideal=${r.idealTheta[1].toFixed(3)}`);
  // MRAC works with different unknown a, b too.
  const r2 = runMRAC({a: -0.5, b: 1.5, am: -3, bm: 3, gamma: 8, dt: 0.01, numSteps: 4000});
  check('also adapts with different (a, b) parameters',
        r2.rmsErrorSteadyState < 0.05,
        `rms=${r2.rmsErrorSteadyState.toExponential(2)}`);
}

// =============================================================================
// 5. ITERATIVE LEARNING CONTROL (REPEATED-TRIAL FEEDFORWARD LEARNING)
// =============================================================================

console.log('\n- iterative-learning-control (learning-based control) -');
{
  const r = runIterativeLearningControl({
    trials: 30,
    horizon: 80,
    dt: 0.1,
    plantRate: 1.2,
    plantGain: 1,
    learningGain: 0.8,
    feedbackGain: 0.8,
    controlMax: 5,
    referenceKind: 'sine',
  });
  check('ILC reduces repeated-trial RMS error by at least 95%',
        r.finalRmsError < 0.05 * r.initialRmsError,
        `initial=${r.initialRmsError.toFixed(4)}, final=${r.finalRmsError.toFixed(4)}`);
  check('ILC final tracking RMS is small',
        r.finalRmsError < 0.01,
        `final=${r.finalRmsError.toExponential(2)}`);
  check('ILC topology has source, learning station, and sink',
        r.topology.stations[0] === 'ilc-trial-source'
        && r.topology.stations.includes('ilc-learning-update-station')
        && r.topology.stations.includes('ilc-result-sink'));
  check('ILC graph uses explicit trial/program/result movables',
        ['ILCTrialPlanToken', 'ILCControlProgramToken', 'ILCTrialResultToken']
          .every(t => r.topology.movables.includes(t)));
}

// =============================================================================
// 6. FEEDBACK LINEARIZATION (NONLINEAR PENDULUM)
// =============================================================================

console.log('\n— feedback-linearization (nonlinear pendulum tracking) —');
{
  const r = runFeedbackLinearization({
    theta0: Math.PI, thetaDot0: 0,
    kp: 25, kv: 10, dt: 0.01, numSteps: 1000,
  });
  check('steady-state RMS tracking error < 1e-3 rad',
        r.rmsErrorSteadyState < 1e-3,
        `rms=${r.rmsErrorSteadyState.toExponential(2)}`);
  check('trajectory stable (no blow-up)',
        r.trajectory.every(x => Math.abs(x[0]) < 100 && Math.abs(x[1]) < 100));
  // Feedback linearization should track ANY smooth reference: a step.
  const stepRef = (_t: number) => ({theta: 0, thetaDot: 0, thetaDDot: 0});
  const r2 = runFeedbackLinearization({
    theta0: Math.PI, thetaDot0: 0, kp: 25, kv: 10,
    reference: stepRef, dt: 0.01, numSteps: 500,
  });
  // With a step reference, settle to θ = 0.
  const finalAngle = r2.trajectory[r2.trajectory.length - 1][0];
  check('settles to step reference θ_d = 0 within 5 s',
        Math.abs(finalAngle) < 0.05,
        `final θ=${finalAngle.toFixed(4)} rad`);
}

// =============================================================================
// 7. MPC (CONSTRAINED RECEDING-HORIZON QP)
// =============================================================================

console.log('\n— mpc-double-integrator (constrained QP / receding horizon) —');
{
  const r = runMPCDoubleIntegrator({
    x0: [3, 0], uMax: 1, N: 15,
    Q: [10, 1], Qf: [50, 5], R: 0.1, dt: 0.1, numSteps: 100,
  });
  check('respects |u| ≤ uMax constraint',
        r.maxAbsU <= 1.0001,
        `max|u|=${r.maxAbsU.toFixed(4)}`);
  check('reaches origin within budget',
        r.arrivalTick < 100,
        `arrived at tick ${r.arrivalTick}`);
  check('saturates control on the boundary (active constraint)',
        r.maxAbsU > 0.95,
        `max|u|=${r.maxAbsU.toFixed(4)} (should saturate)`);
  // Tighter constraint = slower arrival (basic monotonicity check).
  const rTight = runMPCDoubleIntegrator({x0: [3, 0], uMax: 0.5, N: 20,
                                          Q: [10, 1], Qf: [50, 5], R: 0.1, dt: 0.1, numSteps: 200});
  check('tighter |u| bound → larger arrival tick',
        rTight.arrivalTick > r.arrivalTick,
        `tight=${rTight.arrivalTick}, loose=${r.arrivalTick}`);
}

// =============================================================================
// SUMMARY
// =============================================================================

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`optimal-control-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
