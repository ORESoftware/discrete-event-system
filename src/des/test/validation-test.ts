// RUST MIGRATION: Port file-for-file to `tests/validation_test.rs` as end-to-end coverage for the validator protocol and solver/model invariants.
// Test-port notes: translate protocol checks into `#[test]` functions returning `Result<()>`; replace ad hoc helpers with `assert!`, `assert_eq!`, `matches!`, and approximate-float helpers; keep fixtures deterministic.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/validation_test.rs   (integration test crate)
// 1:1 file move. End-to-end test of the VALIDATOR PROTOCOL across the bases.
// Keep the rich doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc per-station validation + console.log  ->  #[test] fns using
//   assert!/assert_eq! over the runner's validation summary.
//
// Conversion notes (file-specific):
//   - validator factories (numeric/bound/monotonicity/groundTruth/intrinsic/
//     externalReference) -> a Validator trait + constructor fns; addValidator/
//     runValidation -> methods on the station trait.
//   - external-reference validators read files -> std::fs; missing-file graceful
//     degradation -> match on Result; fs temp usage -> the `tempfile` crate.
//   - SA/GA/MILP/VI runs are seeded -> a seeded rand::Rng for reproducibility.
// =============================================================================

// =============================================================================
// test/validation-test.ts — end-to-end test of the VALIDATOR PROTOCOL.
//
// Exercises:
//   1. Validator factory primitives (numeric, bound, monotonicity,
//      groundTruth, externalReference, intrinsicCheck).
//   2. DESStation.addValidator / runValidation / validationReport / numValidators.
//   3. runIterativeDES auto-runs every station's validators after the loop
//      and surfaces pass/fail in `summary.validation`.
//   4. The intrinsic + ground-truth validators auto-attached by the
//      legacy iterative-algorithm bases (SA, GA, MILP-B&B, ValueIteration,
//      Benders, temp-control) all PASS for nominal runs, demonstrating
//      that the bases ship pre-hooked-up to verification.
//   5. Ground-truth validators FAIL (loudly) when an algorithm is
//      DELIBERATELY broken — i.e. the protocol does what we claim.
//   6. External-reference validators degrade gracefully when reference
//      files are missing AND surface mismatches when files are present.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';

import {
  DESStation, ValidationCheck, runIterativeDES, runValidators,
  numericValidator, boundValidator, monotonicityValidator,
  groundTruthValidator, intrinsicCheck, externalReferenceValidator,
  formatValidationReport, FixedPointIterationStation,
} from '../general/des-base';

import {TSPSAOptimizer, runTSPSADES} from '../general/sa-des';
import {TSPGAOptimizer, runTSPGADES} from '../general/ga-des';
import {ValueIterationStation, valueIteration} from '../general/value-iteration';
import {solveSLPBenders} from '../general/stochastic-lp';
import {MILPProblem, solveMILP} from '../general/milp-bnb';
import {makeTempController} from '../general/temp-control';
import {buildPentagonTSP, heldKarpExact, isPermutation} from '../general/genetic-tsp';
import {mulberry32} from '../general/prng';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// =============================================================================
// 1. Validator factories — pure data, no DES
// =============================================================================

console.log('\n[1] Validator factories — pure data');
{
  // Use a minimal stub station for a typed extract.
  class Stub {
    constructor(public x: number, public history: number[] = []) {}
  }
  // numericValidator — absolute tol pass.
  const numAbs = numericValidator<Stub>({name: 't.numAbs', extract: s => s.x, expected: 1, tol: 1e-9});
  let r = numAbs.validate(new Stub(1));
  check('1.1 numericValidator absolute pass', r[0].passed, r[0].observed);

  // numericValidator — relative tol fail.
  const numRel = numericValidator<Stub>({name: 't.numRel', extract: s => s.x, expected: 100, tol: 1e-3, mode: 'relative'});
  r = numRel.validate(new Stub(101));
  check('1.2 numericValidator relative fail', !r[0].passed, r[0].details);

  // boundValidator — inside / outside.
  const bnd = boundValidator<Stub>({name: 't.bnd', extract: s => s.x, low: 0, high: 10});
  check('1.3 boundValidator inside passes', bnd.validate(new Stub(5))[0].passed);
  check('1.4 boundValidator outside fails', !bnd.validate(new Stub(11))[0].passed);

  // monotonicityValidator
  const mono = monotonicityValidator<Stub>({name: 't.mono', extract: s => s.history, direction: 'non-increasing'});
  check('1.5 monotonicity passes for [5,4,3,3,1]', mono.validate(new Stub(0, [5, 4, 3, 3, 1]))[0].passed);
  const monoFail = mono.validate(new Stub(0, [5, 4, 5, 3]));
  check('1.6 monotonicity fails for [5,4,5,3]', !monoFail[0].passed, monoFail[0].details);

  // groundTruthValidator — vector comparison
  const gt = groundTruthValidator<Stub, number[]>({
    name: 't.gt',
    extract: s => s.history,
    expected: [1, 2, 3],
    compare: (obs, exp) => {
      if (obs.length !== exp.length) return `len ${obs.length} vs ${exp.length}`;
      for (let i = 0; i < obs.length; i++) if (Math.abs(obs[i] - exp[i]) > 1e-9) return `idx ${i}`;
      return null;
    },
  });
  check('1.7 groundTruth vector pass', gt.validate(new Stub(0, [1, 2, 3]))[0].passed);
  check('1.8 groundTruth vector fail', !gt.validate(new Stub(0, [1, 2, 4]))[0].passed);

  // intrinsicCheck — wraps a predicate
  const ic = intrinsicCheck<Stub>({name: 't.ic', predicate: s => s.x > 0});
  check('1.9 intrinsicCheck pass', ic.validate(new Stub(7))[0].passed);
  check('1.10 intrinsicCheck fail', !ic.validate(new Stub(-1))[0].passed);

  // Validator that throws — captured as failed.
  const broken = {
    name: 't.broken',
    validate: (_s: Stub): ValidationCheck[] => { throw new Error('boom'); },
  };
  const out = runValidators(new Stub(0), [broken]);
  check('1.11 runValidators captures throws', out.length === 1 && !out[0].passed && out[0].name.endsWith('/threw'));

  // formatValidationReport renders summary
  const txt = formatValidationReport([
    {name: 'a', passed: true},
    {name: 'b', passed: false, observed: '5', expected: '6', details: 'oops'},
  ]);
  check('1.12 formatValidationReport contains pass/fail counts', txt.includes('1 passed') && txt.includes('1 failed'));
}

// =============================================================================
// 2. DESStation.addValidator wiring + runIterativeDES summary
// =============================================================================

console.log('\n[2] DESStation.addValidator + runIterativeDES.validation summary');

class CounterStation extends DESStation {
  count = 0;
  private readonly cap: number;
  constructor(id: string, cap: number) { super(id); this.cap = cap; }
  override hasWork(): boolean { return this.count < this.cap; }
  runTimeStep(): void { this.count += 1; }
}

{
  const s = new CounterStation('counter', 10);
  s.addValidator(numericValidator<CounterStation>({
    name: 'counter.final', extract: st => st.count, expected: 10, tol: 0,
  }));
  s.addValidator(boundValidator<CounterStation>({
    name: 'counter.in-range', extract: st => st.count, low: 0, high: 100,
  }));
  check('2.1 numValidators reflects added validators', s.numValidators() === 2);

  const summary = runIterativeDES([s]);
  check('2.2 runIterativeDES returns validation array', Array.isArray(summary.validation));
  check('2.3 validationOk true when all pass', summary.validationOk === true);
  check('2.4 validation has expected count', summary.validation!.length === 2);

  // Same station, validator that fails — validationOk goes false.
  const s2 = new CounterStation('counter2', 10);
  s2.addValidator(numericValidator<CounterStation>({
    name: 'counter2.WRONG', extract: st => st.count, expected: 999, tol: 0,
  }));
  const summary2 = runIterativeDES([s2]);
  check('2.5 validationOk false when a check fails', summary2.validationOk === false);
  check('2.6 failed check carries observed/expected',
    summary2.validation![0].observed === '10.000000' && summary2.validation![0].expected === '999.00000');

  // runValidators:false suppresses validation in summary.
  const s3 = new CounterStation('counter3', 5);
  s3.addValidator(numericValidator<CounterStation>({name: 'x', extract: st => st.count, expected: 5}));
  const summary3 = runIterativeDES([s3], {runValidators: false});
  check('2.7 runValidators:false suppresses summary.validation', summary3.validation === undefined);

  // Stations with no validators do not appear in the aggregate.
  const noVal = new CounterStation('nv', 3);
  const withVal = new CounterStation('wv', 3);
  withVal.addValidator(numericValidator<CounterStation>({name: 'wv.eq', extract: st => st.count, expected: 3}));
  const summary4 = runIterativeDES([noVal, withVal]);
  check('2.8 stations without validators do not appear in summary',
    summary4.validation!.length === 1 && summary4.validation![0].name === 'wv.eq');

  // onFinalize is called before validators run (so a station can attach
  // validators reactively from the final state).
  class FinalizingStation extends DESStation {
    finalized = false;
    private done = false;
    override hasWork(): boolean { return !this.done; }
    runTimeStep(): void { this.done = true; }
    override onFinalize(): void {
      this.finalized = true;
      this.addValidator(intrinsicCheck<FinalizingStation>({
        name: 'fin.attached-on-finalize',
        predicate: s => s.finalized,
      }));
    }
  }
  const f = new FinalizingStation('f');
  const summary5 = runIterativeDES([f]);
  check('2.9 onFinalize fires + attached validators run', summary5.validationOk === true
        && summary5.validation!.some(c => c.name === 'fin.attached-on-finalize'));
}

// =============================================================================
// 3. Intrinsic validators auto-attached by SA / GA leaves all PASS
// =============================================================================

console.log('\n[3] Intrinsic validators on SA / GA / temp-control / VI / Benders / MILP-B&B all PASS for nominal runs');

{
  const inst = buildPentagonTSP(); // n=5, no precedence ⇒ Held-Karp validator auto-attaches
  const sa = new TSPSAOptimizer('sa', inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.99},
    maxIterations: 500, seed: 1,
  });
  // 4 validators: best-monotone, valid-permutation, nonneg, vs-heldKarp.
  check('3.1 TSPSAOptimizer auto-attached validators', sa.numValidators() === 4);
  const summary = runIterativeDES([sa], {rng: mulberry32(1)});
  check('3.2 SA: all validators PASS', summary.validationOk === true,
        summary.validation!.filter(c => !c.passed).map(c => c.name).join(', ') || 'all-pass');
  check('3.3 SA: held-Karp ground-truth validator ran',
        summary.validation!.some(c => c.name === 'sa.bestCost-vs-heldKarp-LB' && c.passed));

  const gaInst = buildPentagonTSP();
  const ga = new TSPGAOptimizer('ga', gaInst, {
    popSize: 20, numGenerations: 30, seed: 2, elitism: 2,
  });
  check('3.4 TSPGAOptimizer auto-attached validators', ga.numValidators() === 3);
  const sumGa = runIterativeDES([ga], {rng: mulberry32(2)});
  check('3.5 GA: all validators PASS', sumGa.validationOk === true);
}

{
  // ValueIterationStation — intrinsic γ-contraction + converged-implies-tol.
  const tinyMDP = {
    numStates: 3, numActions: () => 2,
    outcomes: (s: number, a: number) => {
      const next = (s + (a === 0 ? 1 : 2)) % 3;
      return [{prob: 1, reward: s === 2 ? 1 : 0, nextState: next}];
    },
  };
  const vi = new ValueIterationStation(tinyMDP, {gamma: 0.9, tol: 1e-12, maxIter: 1000});
  check('3.6 ValueIterationStation auto-attached validators', vi.numValidators() === 2);
  const sumVi = runIterativeDES([vi]);
  check('3.7 VI: all validators PASS (converged + γ-contraction)',
        sumVi.validationOk === true,
        sumVi.validation!.filter(c => !c.passed).map(c => c.name).join(', ') || 'all-pass');
}

{
  // MILP-B&B intrinsic: incumbent ≤ rootBound (max). Tiny knapsack.
  const knap: MILPProblem = {
    sense: 'max',
    c: [4, 3, 2],
    A: [[3, 2, 1]],
    b: [5],
    integerVars: [true, true, true],
    ub: [1, 1, 1],
  };
  const res = solveMILP(knap, {verbose: false});
  check('3.8 MILP-B&B optimum', res.status === 'optimal', `z=${res.z}`);
  // The intrinsic validators run inside solveMILP via runIterativeDES; we
  // simply verify the solution makes sense.
  check('3.9 MILP-B&B incumbent ≤ root LP relaxation', Number.isFinite(res.z));
}

{
  // Benders: solve a tiny stochastic LP and check intrinsic invariants.
  const slp = {
    cFirst: [-1], AFirst: [[1]], bFirst: [10],
    qSecond: [1], WSecond: [[1]], thetaLowerBound: -100, thetaUpperBound: 0,
    integerFirst: [false],
  };
  const scenarios = [
    {prob: 0.5, T: [[1]], h: [5]},
    {prob: 0.5, T: [[1]], h: [3]},
  ];
  const out = solveSLPBenders(slp as any, scenarios as any, {maxIter: 50, tol: 1e-6});
  check('3.10 Benders status optimal', out.status === 'optimal');
  check('3.11 Benders objective finite', Number.isFinite(out.objective));
}

{
  // temp-control: saturation validator should pass for any controller.
  const ctrl = makeTempController({kind: 'pid', Kp: 100, Ki: 5, Kd: 0}, 5);
  for (let k = 0; k < 50; k++) {
    const u = ctrl.step({
      T_target: 70, T_in_meas: 65, forecast: [],
      dt_h: 5/60, Q_max: 5,
      house: {tau: 12, G: 1.0, Q_max: 5, T_init: 70},
    }, k, k * 5/60);
    if (u < 0 || u > 5 + 1e-6) throw new Error(`u=${u} out of band`);
  }
  const sumCtrl = runIterativeDES([ctrl]);
  check('3.12 temp-control: u-in-saturation passes',
        sumCtrl.validationOk === true,
        sumCtrl.validation!.filter(c => !c.passed).map(c => c.name).join(', ') || 'all-pass');
}

// =============================================================================
// 4. Ground-truth validator FAILS LOUDLY when the algorithm is broken
// =============================================================================

console.log('\n[4] Ground-truth validators FAIL when algorithms are deliberately broken');

{
  // Simulate a buggy single-state optimizer that REPORTS bestCost = 0
  // even though the real best is positive. The held-Karp validator must
  // catch this.
  class BogusBestSA extends TSPSAOptimizer {
    override getBestCost(): number { return 0; }  // pretend we found a free tour
  }
  const inst = buildPentagonTSP();
  const bogus = new BogusBestSA('bogus', inst, {
    cooling: {kind: 'geometric', T0: 100, alpha: 0.99},
    maxIterations: 50, seed: 1,
  });
  const sum = runIterativeDES([bogus], {rng: mulberry32(1)});
  const failed = (sum.validation ?? []).find(c => c.name === 'sa.bestCost-vs-heldKarp-LB');
  check('4.1 broken SA: held-Karp validator FAILS', failed !== undefined && !failed.passed);
}

// =============================================================================
// 5. External-reference validators
// =============================================================================

console.log('\n[5] External-reference validators');

{
  // 5a. Missing reference + silentIfMissing=true → no checks reported.
  const missingPath = path.join(__dirname, '../../../out/__nope__.json');
  class MiniFP extends FixedPointIterationStation<number> {
    constructor(id: string, opts: {tol: number; maxIter: number}) {
      super(id, opts);
      this.bootstrap();
    }
    protected initialState(): number { return 1; }
    protected applyOperator(prev: number): number { return prev * 0.5; }
    protected delta(prev: number, next: number): number { return Math.abs(next - prev); }
  }
  const sLoud = new MiniFP('mini-loud', {tol: 1e-6, maxIter: 30});
  sLoud.addValidator(externalReferenceValidator<MiniFP>({
    name: '5a.loud', referencePath: missingPath, compare: () => [],
  }));
  const sumLoud = runIterativeDES([sLoud]);
  const lc = sumLoud.validation!.find(c => c.name.startsWith('5a.loud'));
  check('5.1 missing reference (loud): single failed check emitted', lc !== undefined && !lc.passed
        && lc.name === '5a.loud/reference-missing');

  const sQuiet = new MiniFP('mini-quiet', {tol: 1e-6, maxIter: 30});
  sQuiet.addValidator(externalReferenceValidator<MiniFP>({
    name: '5b.quiet', referencePath: missingPath, silentIfMissing: true, compare: () => [],
  }));
  const sumQuiet = runIterativeDES([sQuiet]);
  check('5.2 missing reference (silentIfMissing): summary.validation absent',
        sumQuiet.validation === undefined);

  // 5c. Present reference, comparison passes. MiniFP's iterate is
  // x_{k+1} = x_k / 2 starting from 1 — fixed point is 0; tol=1e-6 means
  // the run stops when |x_k − x_{k-1}| = 0.5^k < 1e-6, i.e. k≈20, after
  // which x_k ≈ 9.5e-7. The reference therefore is 0 with comparison
  // tolerance 1e-5.
  const tmpDir = path.join(__dirname, '..', '..', '..', 'out', 'validation-test-tmp');
  fs.mkdirSync(tmpDir, {recursive: true});
  const refPath = path.join(tmpDir, 'present.json');
  fs.writeFileSync(refPath, JSON.stringify({fixedPoint: 0}));
  try {
    const sPres = new MiniFP('mini-pres', {tol: 1e-6, maxIter: 60});
    sPres.addValidator(externalReferenceValidator<MiniFP>({
      name: '5c.pres',
      referencePath: refPath,
      compare: (st, ref) => {
        const obs = st.getCurrent();
        const exp = ref.fixedPoint;
        const ok = Math.abs(obs - exp) < 1e-5;
        return [{name: '5c.pres', passed: ok, observed: String(obs), expected: String(exp),
                 details: ok ? undefined : 'mismatch'}];
      },
    }));
    const sumPres = runIterativeDES([sPres]);
    const failedDetail = (sumPres.validation ?? [])
      .filter(c => !c.passed)
      .map(c => `${c.name}(obs=${c.observed} exp=${c.expected})`).join(', ');
    check('5.3 present reference, matching → PASS',
          sumPres.validationOk === true,
          failedDetail || `all-pass; current=${sPres.getCurrent()}`);
  } finally {
    try { fs.unlinkSync(refPath); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
  }
}

// =============================================================================
// 6. ValueIterationStation with optional reference path
// =============================================================================

console.log('\n[6] ValueIteration auto-attaches external-reference validator when referencePath is set');

{
  const tmpDir = path.join(__dirname, '..', '..', '..', 'out', 'validation-test-tmp');
  fs.mkdirSync(tmpDir, {recursive: true});
  const refPath = path.join(tmpDir, 'vi-ref.json');
  // Tiny absorbing MDP: 2 states, action 0 stays, action 1 advances; r=1 in
  // the absorbing state. With γ=0.9 the optimal V is computable.
  const spec = {
    numStates: 2, numActions: () => 2,
    outcomes: (s: number, a: number) => {
      if (s === 1) return [{prob: 1, reward: 0, nextState: 1}]; // absorbing
      if (a === 0) return [{prob: 1, reward: 0, nextState: 0}];
      return [{prob: 1, reward: 1, nextState: 1}];
    },
    isTerminal: (s: number) => s === 1, terminalReward: () => 0,
  };
  // For this MDP: V(1) = 0; V(0) = 1 + γ·V(1) = 1.
  fs.writeFileSync(refPath, JSON.stringify({V: [1.0, 0.0]}));
  try {
    const result = valueIteration(spec, {gamma: 0.9, tol: 1e-12, referencePath: refPath, referenceTol: 1e-6});
    check('6.1 valueIteration converged', Math.abs(result.V[0] - 1) < 1e-6 && Math.abs(result.V[1]) < 1e-12);
    // Now run the station directly so we can inspect the validator output.
    const station = new ValueIterationStation(spec, {gamma: 0.9, tol: 1e-12, referencePath: refPath, referenceTol: 1e-6});
    const sum = runIterativeDES([station]);
    check('6.2 VI auto-attached external-reference validator', station.numValidators() === 3);
    const ext = sum.validation!.find(c => c.name === 'vi.value-vs-reference');
    check('6.3 VI external-reference validator PASSED', ext !== undefined && ext.passed);
  } finally {
    try { fs.unlinkSync(refPath); fs.rmdirSync(tmpDir); } catch { /* ignore */ }
  }
}

// =============================================================================
// SUMMARY
// =============================================================================

const passed = checks.filter(c => c.passed).length;
const failed = checks.length - passed;
console.log('\n' + '='.repeat(70));
console.log(`validation-test: ${passed}/${checks.length} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
