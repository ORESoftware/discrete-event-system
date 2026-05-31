// RUST MIGRATION: Prefer moving these unit checks into `src/des/general/control_systems/empirical_control.rs` under `#[cfg(test)] mod tests`.
// Test-port notes: translate DES/control-system scenarios into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, and approximate-float helpers; keep RNG seeds deterministic.

'use strict';

// =============================================================================
// Unit tests for general/control-systems/empirical-control.ts.
// Run with:
//   ./node_modules/.bin/ts-node src/des/test/empirical-control-test.ts
// =============================================================================

import {runIterativeDES} from '../general/des-base/runner';
import {DESStation} from '../general/des-base/station';
import {LinAlg, MatrixInverse, SymmetricEigen} from '../general/control-systems/linear-algebra';
import {MarkovDecisionProcess, PartiallyObservableProcess, StateSpaceModel} from '../general/control-systems/observability-controllability';
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
  MinEnergyController,
  MonteCarloControllability,
  MonteCarloDistinguishability,
  MonteCarloObservability,
  Mulberry32,
  ObservabilityGramian,
} from '../general/control-systems/empirical-control';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol: number): boolean { return Math.abs(a - b) <= tol; }

// -----------------------------------------------------------------------------
console.log('\n[1] LinAlg — inverse & symmetric eigen');
// -----------------------------------------------------------------------------
{
  const A = [[4, 1], [1, 3]];
  const Ai = new MatrixInverse(A).inverse();
  const prod = LinAlg.matMul(A, Ai);
  check('1.1 A·A⁻¹ = I', close(prod[0][0], 1, 1e-9) && close(prod[1][1], 1, 1e-9) && close(prod[0][1], 0, 1e-9));
  check('1.2 singular matrix throws', (() => { try { new MatrixInverse([[1, 2], [2, 4]]).inverse(); return false; } catch { return true; } })());

  // Diagonal eigen.
  const eig = new SymmetricEigen([[2, 0], [0, 5]]);
  check('1.3 eigenvalues ascending', close(eig.minEigenvalue(), 2, 1e-9) && close(eig.maxEigenvalue(), 5, 1e-9));
  // Known 2x2 symmetric: [[2,1],[1,2]] → eigenvalues 1 and 3.
  const eig2 = new SymmetricEigen([[2, 1], [1, 2]]);
  check('1.4 [[2,1],[1,2]] → {1,3}', close(eig2.minEigenvalue(), 1, 1e-8) && close(eig2.maxEigenvalue(), 3, 1e-8));
  // Eigenvectors orthogonal & reconstruct.
  const v = eig2.vectors();
  const dot = v[0][0] * v[0][1] + v[1][0] * v[1][1];
  check('1.5 eigenvectors orthonormal', close(dot, 0, 1e-8));
}

// -----------------------------------------------------------------------------
console.log('\n[2] RNG determinism');
// -----------------------------------------------------------------------------
{
  const a = new Mulberry32(42), b = new Mulberry32(42);
  check('2.1 same seed → same stream', a.next() === b.next() && a.next() === b.next());
  const c = new Mulberry32(7);
  let inRange = true;
  for (let i = 0; i < 1000; i++) { const u = c.uniform(2); if (u < -2 || u > 2) inRange = false; }
  check('2.2 uniform(2) within [-2,2]', inRange);
  // categorical respects pmf roughly.
  const r = new Mulberry32(1); let ones = 0;
  for (let i = 0; i < 5000; i++) if (r.categorical([0.2, 0.8]) === 1) ones++;
  check('2.3 categorical ≈ pmf', close(ones / 5000, 0.8, 0.03), `got ${(ones / 5000).toFixed(3)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[3] Gramians — controllable vs deficient');
// -----------------------------------------------------------------------------
{
  const di = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}), 0.05);
  const wc = new ControllabilityGramian(di, 30);
  const wo = new ObservabilityGramian(di, 30);
  check('3.1 controllable ⇒ W_c min > 0', wc.min() > 1e-9, `min=${wc.min().toExponential(2)}`);
  check('3.2 observable ⇒ W_o min > 0', wo.min() > 1e-9, `min=${wo.min().toExponential(2)}`);
  check('3.3 finite condition numbers', Number.isFinite(wc.conditionNumber()) && Number.isFinite(wo.conditionNumber()));

  // Decoupled: second mode untouched ⇒ a zero eigenvalue on both Gramians.
  const dec = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[-1, 0], [0, -2]], B: [[1], [0]], C: [[1, 0]]}), 0.05);
  const wcD = new ControllabilityGramian(dec, 30);
  const woD = new ObservabilityGramian(dec, 30);
  check('3.4 decoupled ⇒ W_c min ≈ 0 (uncontrollable dir)', wcD.min() < 1e-9, `min=${wcD.min().toExponential(2)}`);
  check('3.5 decoupled ⇒ W_o min ≈ 0 (unobservable dir)', woD.min() < 1e-9, `min=${woD.min().toExponential(2)}`);
  check('3.6 decoupled ⇒ cond = ∞', !Number.isFinite(wcD.conditionNumber()) && !Number.isFinite(woD.conditionNumber()));
}

// -----------------------------------------------------------------------------
console.log('\n[4] MinEnergyController — reach error detects uncontrollable subspace');
// -----------------------------------------------------------------------------
{
  const di = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}), 0.05);
  const ctl = new MinEnergyController(di, 30);
  check('4.1 controllable target reached (err≈0)', ctl.reachError([0.5, -0.3]) < 1e-6, `err=${ctl.reachError([0.5, -0.3]).toExponential(2)}`);

  const dec = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[-1, 0], [0, -2]], B: [[1], [0]], C: [[1, 0]]}), 0.05);
  const ctlD = new MinEnergyController(dec, 30);
  check('4.2 reachable-mode target ok', ctlD.reachError([1, 0]) < 1e-6);
  check('4.3 unreachable-mode target has residual', ctlD.reachError([0, 1]) > 0.9, `err=${ctlD.reachError([0, 1]).toFixed(3)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[5] Monte-Carlo controllability — recovers Gramian directions');
// -----------------------------------------------------------------------------
{
  const di = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}), 0.05);
  const wc = new ControllabilityGramian(di, 30);
  const mc = new MonteCarloControllability(di, 30, {trials: 4000, inputBound: 1, seed: 1}).run();
  // Cov = (uBound²/3)·W_c ⇒ eigenvalue ratios should match.
  const wcRatio = wc.eigenvalues()[1] / wc.eigenvalues()[0];
  const mcRatio = mc.spreadEigenvalues[1] / mc.spreadEigenvalues[0];
  check('5.1 empirical spread ratio ≈ Gramian ratio', close(wcRatio, mcRatio, wcRatio * 0.25),
    `gram=${wcRatio.toFixed(2)} mc=${mcRatio.toFixed(2)}`);
  check('5.2 controllable ⇒ high target hit rate', mc.targetSuccessRate > 0.95, `rate=${mc.targetSuccessRate.toFixed(2)}`);

  const dec = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[-1, 0], [0, -2]], B: [[1], [0]], C: [[1, 0]]}), 0.05);
  const mcD = new MonteCarloControllability(dec, 30, {trials: 2000, seed: 3}).run();
  check('5.3 deficient ⇒ low target hit rate', mcD.targetSuccessRate < 0.3, `rate=${mcD.targetSuccessRate.toFixed(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[6] Monte-Carlo observability — reconstruction error tracks W_o');
// -----------------------------------------------------------------------------
{
  const di = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}), 0.05);
  const obs = new MonteCarloObservability(di, 30, {trials: 1500, noiseStd: 0.01, seed: 2}).run();
  check('6.1 observable ⇒ small recon error', obs.meanReconstructionError < 0.1, `err=${obs.meanReconstructionError.toFixed(4)}`);

  const dec = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[-1, 0], [0, -2]], B: [[1], [0]], C: [[1, 0]]}), 0.05);
  const obsD = new MonteCarloObservability(dec, 30, {trials: 1500, noiseStd: 0.01, seed: 2}).run();
  check('6.2 unobservable dir ⇒ large recon error', obsD.meanReconstructionError > 0.5, `err=${obsD.meanReconstructionError.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[7] MDP controllability degree — value iteration + rollouts');
// -----------------------------------------------------------------------------
{
  const ring = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]]});
  const trap = new MarkovDecisionProcess({numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]]});
  const ringP = new MdpControllabilityDegree(ring);
  const trapP = new MdpControllabilityDegree(trap);

  const hitRing = ringP.expectedHittingTimes(0);
  check('7.1 ring: hitting time s2→s0 = 1', close(hitRing[2], 1, 1e-6), `=${hitRing[2]}`);
  check('7.2 ring: hitting time s1→s0 = 2', close(hitRing[1], 2, 1e-6), `=${hitRing[1]}`);

  const hitTrap = trapP.expectedHittingTimes(0);
  check('7.3 trap: s1,s2 cannot reach s0 (∞)', !Number.isFinite(hitTrap[1]) && !Number.isFinite(hitTrap[2]));

  const degRing = ringP.perTargetDegree({episodes: 400, seed: 1});
  check('7.4 ring fully controllable (all degree 1)', degRing.every(d => d > 0.99));
  const degTrap = trapP.perTargetDegree({episodes: 400, seed: 1});
  check('7.5 trap not fully controllable (some degree < 1)', degTrap.some(d => d < 0.99));
}

// -----------------------------------------------------------------------------
console.log('\n[8] POMDP observability degree — belief tracking');
// -----------------------------------------------------------------------------
{
  const distinct = new PartiallyObservableProcess({numStates: 2, numActions: 1, transition: [[[0.5, 0.5], [0.5, 0.5]]], numObservations: 2, observation: [[1, 0], [0, 1]]});
  const aliased = new PartiallyObservableProcess({numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]], numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]]});

  const rD = new MonteCarloDistinguishability(distinct).run({episodes: 600, seed: 5});
  check('8.1 distinct ⇒ hit-prob ≈ 1', rD.minDegree > 0.95, `min=${rD.minDegree.toFixed(3)}`);
  check('8.2 distinct ⇒ residual entropy ≈ 0', rD.residualEntropy.every(e => e < 0.05));

  const rA = new MonteCarloDistinguishability(aliased).run({episodes: 600, seed: 5});
  check('8.3 aliased ⇒ hit-prob ≈ 0.5', close(rA.minDegree, 0.5, 0.1), `min=${rA.minDegree.toFixed(3)}`);
  check('8.4 aliased ⇒ residual entropy ≈ 1 bit', rA.residualEntropy.every(e => close(e, 1, 0.1)));
}

// -----------------------------------------------------------------------------
console.log('\n[9] DES pipeline — Gramian degree reports flow to sink');
// -----------------------------------------------------------------------------
{
  const di = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]}), 0.05);
  const src = new DiscreteSystemSourceStation('src', [new DiscreteSystemToken('di', di, 30)]);
  const evalr = new LtiDegreeEvaluatorStation('eval');
  const sink = new DegreeReportSinkStation('sink');
  src.pipe(evalr, EmpiricalChannels.SYSTEM);
  evalr.pipe(sink, EmpiricalChannels.REPORT);
  runIterativeDES([src, evalr, sink] as DESStation[], {shuffle: false, maxTicks: 10});
  check('9.1 one report emitted', sink.reports.length === 1, `got ${sink.reports.length}`);
  const r: DegreeReportToken | undefined = sink.reports[0];
  check('9.2 report is lti-degree with positive min controllability',
    !!r && r.kind === 'lti-degree' && r.minControllability > 0 && r.minObservability > 0);
}

// -----------------------------------------------------------------------------
console.log('\n[10] Cross-validation invariants — Gramians, eigen, rollouts');
// -----------------------------------------------------------------------------
{
  const sys = DiscreteLinearSystem.fromContinuous(
    new StateSpaceModel({A: [[0, 1], [0, -0.5]], B: [[0], [1]], C: [[1, 0]]}), 0.1);
  const H = 12;
  // W_c = R Rᵀ  and  W_o = Oᵀ O.
  const R = sys.reachabilityMap(H);
  const RRt = LinAlg.matMul(R, LinAlg.transpose(R));
  const wc = new ControllabilityGramian(sys, H).matrix();
  const O = sys.observabilityMap(H);
  const OtO = LinAlg.matMul(LinAlg.transpose(O), O);
  const wo = new ObservabilityGramian(sys, H).matrix();
  const matClose = (A: number[][], B: number[][], tol: number) => {
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) if (Math.abs(A[i][j] - B[i][j]) > tol) return false;
    return true;
  };
  check('10.1 W_c = R·Rᵀ', matClose(wc, RRt, 1e-9));
  check('10.2 W_o = Oᵀ·O', matClose(wo, OtO, 1e-9));
  check('10.3 Gramians symmetric', close(wc[0][1], wc[1][0], 1e-12) && close(wo[0][1], wo[1][0], 1e-12));

  // minEnergyToReach(x) == xᵀ W_c⁻¹ x.
  const gram = new ControllabilityGramian(sys, H);
  const x = [0.3, -0.7];
  const Winv = new MatrixInverse(gram.matrix()).inverse();
  const wInvX = LinAlg.matVec(Winv, x);
  const quad = x[0] * wInvX[0] + x[1] * wInvX[1];
  check('10.4 minEnergyToReach = xᵀW⁻¹x', close(gram.minEnergyToReach(x), quad, 1e-6 * Math.abs(quad) + 1e-9));

  // SymmetricEigen reconstruction: V Λ Vᵀ = M.
  const M = [[3, 1, 0], [1, 2, 1], [0, 1, 4]];
  const eig = new SymmetricEigen(M);
  const V = eig.vectors();
  const Lam = LinAlg.zeros(3, 3);
  eig.values().forEach((l, i) => Lam[i][i] = l);
  const recon = LinAlg.matMul(LinAlg.matMul(V, Lam), LinAlg.transpose(V));
  check('10.5 V·Λ·Vᵀ reconstructs M', matClose(recon, M, 1e-7));
  check('10.6 eigenvalues ascending', eig.values()[0] <= eig.values()[1] && eig.values()[1] <= eig.values()[2]);

  // rollout matches manual stepping; outputs length = H.
  const x0 = [1, 0], u = [[0.5], [-0.2], [0.1]];
  let manual = x0;
  for (const uk of u) manual = sys.step(manual, uk);
  const rolled = sys.rollout(x0, u);
  check('10.7 rollout == manual stepping', close(rolled[0], manual[0], 1e-12) && close(rolled[1], manual[1], 1e-12));
  check('10.8 outputs() length = H', sys.outputs(x0, 7).length === 7);
}

// -----------------------------------------------------------------------------
console.log('\n[11] MDP/POMDP degree — extra structural cases');
// -----------------------------------------------------------------------------
{
  // Fully-connected MDP (every state can self-loop & jump) ⇒ all targets degree ~1.
  const full = new MarkovDecisionProcess({numStates: 3, numActions: 1,
    transition: [[[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]]});
  const deg = new MdpControllabilityDegree(full).perTargetDegree({episodes: 400, seed: 2});
  check('11.1 fully-connected ⇒ every target degree = 1', deg.every(d => d > 0.99), `[${deg.map(d => d.toFixed(2))}]`);

  // Two actions: action 0 goes right, action 1 goes left ⇒ strongly connected.
  const bidir = new MarkovDecisionProcess({numStates: 3, numActions: 2,
    transition: [
      [[0, 1, 0], [0, 0, 1], [0, 0, 1]],   // a0: →
      [[1, 0, 0], [1, 0, 0], [0, 1, 0]],   // a1: ←
    ]});
  const planner = new MdpControllabilityDegree(bidir);
  const hit = planner.expectedHittingTimes(0);
  check('11.2 bidirectional: all states can reach s0 (finite)', hit.every(h => Number.isFinite(h)));

  // POMDP needing >1 step: obs aliases s0,s1 instantly but transitions differ.
  const multiStep = new PartiallyObservableProcess({
    numStates: 3, numActions: 1,
    transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]],
    numObservations: 2,
    observation: [[1, 0], [1, 0], [0, 1]],   // s0,s1 share obs0; s2 obs1
  });
  const r = new MonteCarloDistinguishability(multiStep).run({episodes: 600, seed: 3});
  check('11.3 multi-step POMDP: s2 well identified', r.hitProbability[2] > 0.9, `p(s2)=${r.hitProbability[2].toFixed(2)}`);
}

// -----------------------------------------------------------------------------
console.log(`\n──────────────────────────────────────────────`);
console.log(`  empirical-control: ${pass} passed, ${fail} failed`);
console.log(`──────────────────────────────────────────────`);
if (fail > 0) process.exit(1);
