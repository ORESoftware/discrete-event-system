'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/observability_controllability_test.rs   (integration test crate)
// 1:1 file move. Drives control-systems/observability-controllability stations
// plus shared LinAlg, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - matrix rank uses an epsilon threshold -> compare via approx::assert_*_eq!
//     with an explicit tolerance; matrices -> the shared linalg types.
// =============================================================================

// =============================================================================
// Unit tests for general/control-systems/observability-controllability.ts.
// Run with:
//   ./node_modules/.bin/ts-node src/des/test/observability-controllability-test.ts
// =============================================================================

import {runIterativeDES} from '../general/des-base/runner';
import {DESStation} from '../general/des-base/station';
import {LinAlg} from '../general/control-systems/linear-algebra';
import {
  ControllabilityEvaluatorStation,
  EvaluationSinkStation,
  EvaluationToken,
  MarkovDecisionProcess,
  MdpControllabilityEvaluatorStation,
  MdpSourceStation,
  MdpToken,
  ObservabilityEvaluatorStation,
  PartiallyObservableProcess,
  PomdpObservabilityEvaluatorStation,
  PomdpSourceStation,
  PomdpToken,
  StateSpaceModel,
  StateSpaceSourceStation,
  StateSpaceToken,
} from '../general/control-systems/observability-controllability';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

// -----------------------------------------------------------------------------
console.log('\n[1] LinAlg — rank, products, stacking');
// -----------------------------------------------------------------------------
{
  check('1.1 identity rank = n', LinAlg.rank(LinAlg.identity(4)) === 4);
  check('1.2 rank-deficient (two equal rows)', LinAlg.rank([[1, 2], [1, 2]]) === 1);
  check('1.3 zero matrix rank 0', LinAlg.rank(LinAlg.zeros(3, 3)) === 0);
  const A = [[1, 2], [3, 4]];
  const I = LinAlg.identity(2);
  check('1.4 A·I = A', JSON.stringify(LinAlg.matMul(A, I)) === JSON.stringify(A));
  check('1.5 hstack concatenates columns',
    JSON.stringify(LinAlg.hstack([[[1], [2]], [[3], [4]]])) === JSON.stringify([[1, 3], [2, 4]]));
  check('1.6 vstack concatenates rows',
    JSON.stringify(LinAlg.vstack([[[1, 2]], [[3, 4]]])) === JSON.stringify([[1, 2], [3, 4]]));
  check('1.7 A² via power', JSON.stringify(LinAlg.power(A, 2)) === JSON.stringify(LinAlg.matMul(A, A)));
}

// -----------------------------------------------------------------------------
console.log('\n[2] Linear state-space — the query worked example');
// -----------------------------------------------------------------------------
{
  // A=[[0,1],[0,0]], B=[[0],[1]], C=[[1,0]]  →  both rank 2.
  const m = new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]});
  // 𝒞 = [B AB] = [[0,1],[1,0]]
  check('2.1 controllability matrix = [[0,1],[1,0]]',
    JSON.stringify(m.controllabilityMatrix()) === JSON.stringify([[0, 1], [1, 0]]));
  // 𝒪 = [C; CA] = [[1,0],[0,1]]
  check('2.2 observability matrix = [[1,0],[0,1]]',
    JSON.stringify(m.observabilityMatrix()) === JSON.stringify([[1, 0], [0, 1]]));
  check('2.3 rank 𝒞 = 2 → controllable', m.controllabilityRank() === 2 && m.isControllable());
  check('2.4 rank 𝒪 = 2 → observable', m.observabilityRank() === 2 && m.isObservable());
}

// -----------------------------------------------------------------------------
console.log('\n[3] Linear state-space — deficient cases');
// -----------------------------------------------------------------------------
{
  // Decoupled modes, input/ output touch only mode 1 → neither.
  const neither = new StateSpaceModel({A: [[1, 0], [0, 2]], B: [[1], [0]], C: [[1, 0]]});
  check('3.1 uncontrollable (rank 1)', !neither.isControllable() && neither.controllabilityRank() === 1);
  check('3.2 unobservable (rank 1)', !neither.isObservable() && neither.observabilityRank() === 1);
  // Drive both modes but only observe mode 1 → controllable, not observable.
  const cNotO = new StateSpaceModel({A: [[1, 0], [0, 2]], B: [[1], [1]], C: [[1, 0]]});
  check('3.3 controllable but not observable', cNotO.isControllable() && !cNotO.isObservable());
  // Observe both modes, drive only mode 1 → observable, not controllable.
  const oNotC = new StateSpaceModel({A: [[1, 0], [0, 2]], B: [[1], [0]], C: [[1, 0], [0, 1]]});
  check('3.4 observable but not controllable', oNotC.isObservable() && !oNotC.isControllable());
}

// -----------------------------------------------------------------------------
console.log('\n[4] MDP — reachability (controllability analog)');
// -----------------------------------------------------------------------------
{
  const ring = new MarkovDecisionProcess({
    numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],
  });
  check('4.1 ring is strongly connected → controllable', ring.isStructurallyControllable());
  check('4.2 ring reaches all S² ordered pairs', ring.reachablePairCount() === 9);
  const trap = new MarkovDecisionProcess({
    numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]],
  });
  check('4.3 absorbing trap → not controllable', !trap.isStructurallyControllable());
  check('4.4 trap loses reachable pairs', trap.reachablePairCount() < 9);
}

// -----------------------------------------------------------------------------
console.log('\n[5] POMDP — distinguishability (observability analog)');
// -----------------------------------------------------------------------------
{
  // Distinct sensors → immediately observable.
  const distinct = new PartiallyObservableProcess({
    numStates: 2, numActions: 1, transition: [[[0.5, 0.5], [0.5, 0.5]]],
    numObservations: 2, observation: [[1, 0], [0, 1]],
  });
  check('5.1 distinct sensors → observable', distinct.isStructurallyObservable());
  check('5.2 distinct → 2 classes, no aliasing',
    distinct.classCount() === 2 && distinct.indistinguishablePairs().length === 0);

  // Same observation + stay put → never observable.
  const aliased = new PartiallyObservableProcess({
    numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]],
    numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]],
  });
  check('5.3 aliased + absorbing → not observable', !aliased.isStructurallyObservable());
  check('5.4 aliased → pair (0,1) flagged',
    JSON.stringify(aliased.indistinguishablePairs()) === JSON.stringify([[0, 1]]));

  // MULTI-STEP: s0 and s1 share the observation, but s0 transitions to the
  // distinctly-observed s2 while s1 stays — refinement must separate them
  // (the POMDP cousin of using the CA term, not just C).
  const multiStep = new PartiallyObservableProcess({
    numStates: 3, numActions: 1,
    transition: [[[0, 0, 1], [0, 1, 0], [0, 0, 1]]],
    numObservations: 2,
    observation: [[1, 0], [1, 0], [0, 1]],
  });
  check('5.5 multi-step refinement → observable', multiStep.isStructurallyObservable(),
        `classes=${multiStep.classCount()}`);
  check('5.6 multi-step → no residual aliasing', multiStep.indistinguishablePairs().length === 0);
}

// -----------------------------------------------------------------------------
console.log('\n[6] DES pipeline — evaluators emit correct verdicts as tokens');
// -----------------------------------------------------------------------------
{
  const lti = [
    new StateSpaceToken('di', new StateSpaceModel({A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]]})),
    new StateSpaceToken('neither', new StateSpaceModel({A: [[1, 0], [0, 2]], B: [[1], [0]], C: [[1, 0]]})),
  ];
  const mdp = [new MdpToken('ring', new MarkovDecisionProcess({
    numStates: 3, numActions: 1, transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],
  }))];
  const pomdp = [new PomdpToken('aliased', new PartiallyObservableProcess({
    numStates: 2, numActions: 1, transition: [[[1, 0], [0, 1]]],
    numObservations: 2, observation: [[0.5, 0.5], [0.5, 0.5]],
  }))];

  const ltiSrc = new StateSpaceSourceStation('lti', lti);
  const mdpSrc = new MdpSourceStation('mdp', mdp);
  const pomdpSrc = new PomdpSourceStation('pomdp', pomdp);
  const ctrl = new ControllabilityEvaluatorStation('ctrl');
  const obs = new ObservabilityEvaluatorStation('obs');
  const mdpEval = new MdpControllabilityEvaluatorStation('mdp-eval');
  const pomdpEval = new PomdpObservabilityEvaluatorStation('pomdp-eval');
  const sink = new EvaluationSinkStation('sink');
  ltiSrc.pipe(ctrl, 'model-lti', 'model-lti');
  ltiSrc.pipe(obs, 'model-lti', 'model-lti');
  mdpSrc.pipe(mdpEval, 'model-mdp', 'model-mdp');
  pomdpSrc.pipe(pomdpEval, 'model-pomdp', 'model-pomdp');
  for (const ev of [ctrl, obs, mdpEval, pomdpEval] as DESStation[]) ev.pipe(sink, 'evaluation', 'evaluation');

  runIterativeDES([ltiSrc, mdpSrc, pomdpSrc, ctrl, obs, mdpEval, pomdpEval, sink], {shuffle: false, maxTicks: 10});

  const find = (label: string, kind: string) =>
    sink.results.find(r => r.label === label && r.kind === kind) as EvaluationToken | undefined;
  check('6.1 emits 2 verdicts per LTI model + 1 MDP + 1 POMDP', sink.results.length === 2 * 2 + 1 + 1,
        `got ${sink.results.length}`);
  check('6.2 di controllable verdict', find('di', 'controllability')?.full === true);
  check('6.3 di observable verdict', find('di', 'observability')?.full === true);
  check('6.4 neither uncontrollable verdict', find('neither', 'controllability')?.full === false);
  check('6.5 neither unobservable verdict', find('neither', 'observability')?.full === false);
  check('6.6 ring MDP controllable verdict', find('ring', 'mdp-controllability')?.full === true);
  check('6.7 aliased POMDP unobservable verdict', find('aliased', 'pomdp-observability')?.full === false);
}

console.log(`\n  ─────────────────────────────────────────────────────────────────────────`);
console.log(`  ${pass} passed, ${fail} failed`);
process.exitCode = fail === 0 ? 0 : 1;
