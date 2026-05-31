'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-empirical-control.rs   (fn main)
// 1:1 file move. Measures the DEGREE of controllability/observability via
// Gramians and Monte-Carlo trials (not the binary Kalman verdict).
//
// Conversion notes (file-specific):
//   - Monte-Carlo trials (random controls / measurement noise) -> inject
//     RandomSource/SeededRandom (Mulberry32 here).
//   - Gramian eigenvalue spread -> shared::linalg (SymmetricEigen).
//   - many station/token types imported -> use crate::des::general::
//     control_systems::...; top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-empirical-control.ts — measure the DEGREE of controllability /
// observability numerically (Gramians) and empirically (Monte-Carlo trials),
// rather than the binary Kalman rank verdict.
//
//   npm run empirical-control
//
// For each linear system it reports:
//   • W_c / W_o eigenvalue spread (min = weakest direction, max = strongest)
//   • the empirical reached-state covariance spread from random control trials
//     (should be ∝ W_c) and the least-squares targeting success rate
//   • the noisy least-squares state-reconstruction error (∝ 1/√W_o weak axis)
// For the MDP it reports random-policy reach degree per target; for the POMDP
// the belief-tracking hit-probability / residual entropy per true state.
// =============================================================================

import {runIterativeDES} from './general/des-base/runner';
import {StateSpaceModel, MarkovDecisionProcess, PartiallyObservableProcess} from './general/control-systems/observability-controllability';
import {
  ControllabilityGramian,
  DegreeReportSinkStation,
  DegreeReportToken,
  DiscreteLinearSystem,
  DiscreteSystemSourceStation,
  DiscreteSystemToken,
  EmpiricalChannels,
  LtiDegreeEvaluatorStation,
  MdpControllabilityDegree,
  MdpDegreeEvaluatorStation,
  MdpDegreeSourceStation,
  MdpDegreeToken,
  MonteCarloControllability,
  MonteCarloDistinguishability,
  MonteCarloObservability,
  ObservabilityGramian,
  PomdpDegreeEvaluatorStation,
  PomdpDegreeSourceStation,
  PomdpDegreeToken,
} from './general/control-systems/empirical-control';

class EmpiricalControlDemo {
  private readonly horizon = 40;
  private readonly dt = 0.02;

  /** The "real system": DC motor, state [i, ω], input V, output ω. */
  private dcMotor(): DiscreteLinearSystem {
    const R = 2, L = 0.5, Ke = 0.1, Kt = 0.1, J = 0.02, B = 0.002;
    const model = new StateSpaceModel({
      A: [[-R / L, -Ke / L], [Kt / J, -B / J]],
      B: [[1 / L], [0]],
      C: [[0, 1]],
    });
    return DiscreteLinearSystem.fromContinuous(model, this.dt);
  }

  /** Double integrator (controllable + observable). */
  private doubleIntegrator(): DiscreteLinearSystem {
    const model = new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]});
    return DiscreteLinearSystem.fromContinuous(model, this.dt);
  }

  /** Decoupled modes: input/output only touch mode 1 ⇒ mode 2 is invisible
   *  and undrivable, so min controllability / observability → 0. */
  private decoupled(): DiscreteLinearSystem {
    const model = new StateSpaceModel({A: [[-1, 0], [0, -2]], B: [[1], [0]], C: [[1, 0]]});
    return DiscreteLinearSystem.fromContinuous(model, this.dt);
  }

  run(): void {
    const systems: Array<{label: string; sys: DiscreteLinearSystem}> = [
      {label: 'DC motor (real system)', sys: this.dcMotor()},
      {label: 'double integrator', sys: this.doubleIntegrator()},
      {label: 'decoupled modes', sys: this.decoupled()},
    ];

    // ── DES pipeline: Gramian min/max degree reports ────────────────────────
    const ltiSource = new DiscreteSystemSourceStation('lti-src',
      systems.map(s => new DiscreteSystemToken(s.label, s.sys, this.horizon)));
    const ltiEval = new LtiDegreeEvaluatorStation('lti-degree');
    const mdpSource = new MdpDegreeSourceStation('mdp-src', [
      new MdpDegreeToken('ring MDP', new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]})),
      new MdpDegreeToken('trap MDP', new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]]})),
    ]);
    const mdpEval = new MdpDegreeEvaluatorStation('mdp-degree');
    const pomdpSource = new PomdpDegreeSourceStation('pomdp-src', [
      new PomdpDegreeToken('distinct sensors', new PartiallyObservableProcess({
        numStates: 2, numActions: 1, transition: [[[0.5, 0.5], [0.5, 0.5]]], numObservations: 2, observation: [[1, 0], [0, 1]],
      })),
      new PomdpDegreeToken('aliased sensors', new PartiallyObservableProcess({
        numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]], numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]],
      })),
    ]);
    const pomdpEval = new PomdpDegreeEvaluatorStation('pomdp-degree');
    const sink = new DegreeReportSinkStation('sink');
    ltiSource.pipe(ltiEval, EmpiricalChannels.SYSTEM); ltiEval.pipe(sink, EmpiricalChannels.REPORT);
    mdpSource.pipe(mdpEval, EmpiricalChannels.MDP); mdpEval.pipe(sink, EmpiricalChannels.REPORT);
    pomdpSource.pipe(pomdpEval, EmpiricalChannels.POMDP); pomdpEval.pipe(sink, EmpiricalChannels.REPORT);
    runIterativeDES([ltiSource, ltiEval, mdpSource, mdpEval, pomdpSource, pomdpEval, sink],
      {shuffle: false, maxTicks: 20});

    console.log('================ Gramian degree reports (DES pipeline) ================');
    for (const r of sink.reports) this.printReport(r);

    // ── Direct empirical comparison: analytic Gramian vs Monte-Carlo trials ──
    console.log('\n================ Empirical (trial-based) vs analytic ================');
    for (const {label, sys} of systems) {
      const wc = new ControllabilityGramian(sys, this.horizon);
      const wo = new ObservabilityGramian(sys, this.horizon);
      const mcC = new MonteCarloControllability(sys, this.horizon, {trials: 3000, inputBound: 1, seed: 1}).run();
      const mcO = new MonteCarloObservability(sys, this.horizon, {trials: 1500, noiseStd: 0.02, seed: 2}).run();
      console.log(`\n--- ${label} ---`);
      console.log(`  CONTROLLABILITY`);
      console.log(`    W_c eigenvalues (min..max) : [${this.vec(wc.eigenvalues())}]`);
      console.log(`    empirical reach-cloud var  : [${this.vec(mcC.spreadEigenvalues)}]  (∝ W_c)`);
      console.log(`    least-squares target hit % : ${(mcC.targetSuccessRate * 100).toFixed(1)}%   reachRadius=${mcC.reachRadius.toFixed(3)}`);
      console.log(`    min/max controllability    : ${wc.min().toExponential(2)} / ${wc.max().toExponential(2)}  (cond ${this.cond(wc.conditionNumber())})`);
      console.log(`  OBSERVABILITY`);
      console.log(`    W_o eigenvalues (min..max) : [${this.vec(wo.eigenvalues())}]`);
      console.log(`    recon error (mean/worst)   : ${mcO.meanReconstructionError.toFixed(4)} / ${mcO.worstReconstructionError.toFixed(4)}  @ noise 0.02`);
      console.log(`    min/max observability      : ${wo.min().toExponential(2)} / ${wo.max().toExponential(2)}  (cond ${this.cond(wo.conditionNumber())})`);
    }

    // ── MDP hitting times (numerical planning) ──────────────────────────────
    console.log('\n================ MDP controllability via value iteration ================');
    const ring = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]});
    const trap = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]]});
    for (const [name, mdp] of [['ring', ring], ['trap', trap]] as Array<[string, MarkovDecisionProcess]>) {
      const planner = new MdpControllabilityDegree(mdp);
      console.log(`  ${name}: expected steps to reach s0 from [s0,s1,s2] = [${this.vec(planner.expectedHittingTimes(0))}]`);
      console.log(`        random-policy reach degree per target = [${this.vec(planner.perTargetDegree({episodes: 600}))}]`);
    }

    // ── POMDP distinguishability (belief tracking) ──────────────────────────
    console.log('\n================ POMDP observability via belief tracking ================');
    const distinct = new PartiallyObservableProcess({numStates: 2, numActions: 1, transition: [[[0.5, 0.5], [0.5, 0.5]]], numObservations: 2, observation: [[1, 0], [0, 1]]});
    const aliased = new PartiallyObservableProcess({numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]], numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]]});
    for (const [name, pomdp] of [['distinct', distinct], ['aliased', aliased]] as Array<[string, PartiallyObservableProcess]>) {
      const r = new MonteCarloDistinguishability(pomdp).run({episodes: 800});
      console.log(`  ${name}: belief hit-prob per state = [${this.vec(r.hitProbability)}]   residual entropy = [${this.vec(r.residualEntropy)}] bits`);
    }
  }

  private printReport(r: DegreeReportToken): void {
    console.log(`[${r.kind}] ${r.label}`);
    console.log(`    ${r.detail}`);
  }
  private vec(v: number[]): string { return v.map(x => (Number.isFinite(x) ? x.toFixed(4) : '∞')).join(', '); }
  private cond(c: number): string { return Number.isFinite(c) ? c.toExponential(1) : '∞'; }
}

new EmpiricalControlDemo().run();
