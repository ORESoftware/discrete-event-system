'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/multistage_stochastic_test.rs   (integration test crate)
// 1:1 file move. Exercises multistage-stochastic plus the shared AffineCutPool /
// PreconditionError from des-base, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) relative float comparison -> approx::assert_relative_eq!.
//   - expectPrecondition() asserts a thrown PreconditionError naming a param ->
//     map to `Result::Err` matching, or `#[should_panic(expected = "...")]`.
//   - SDDP scenario sampling -> a seeded rand::Rng for reproducible cuts.
// =============================================================================

// =============================================================================
// Unit tests for multi-stage stochastic programming / SDDP and the shared
// affine cut-pool base class.
// =============================================================================

import {AffineCutPool, PreconditionError} from '../general/des-base';
import {
  buildDefaultMultiStageInventoryProblem,
  evaluatePolicyExact,
  solveExactScenarioTree,
  solveMultiStageSDDP,
  solveStageDecision,
  validateMultiStageProblem,
} from '../general/multistage-stochastic';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function close(a: number, b: number, tol = 1e-7): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
function expectPrecondition(label: string, fn: () => void, param: string): void {
  let err: unknown = null;
  try { fn(); } catch (e) { err = e; }
  check(label,
    err instanceof PreconditionError && err.message.toLowerCase().includes(param.toLowerCase()),
    err ? (err as Error).message : 'no error');
}

console.log('\n[1] AffineCutPool base utility');
{
  const upper = new AffineCutPool(1, 'upper');
  upper.add({alpha: 10, beta: [0], source: 'constant'});
  upper.add({alpha: 4, beta: [1], source: 'slope'});
  check('1.1 upper envelope takes min cut', close(upper.evaluate([3]), 7));
  check('1.2 active cut provenance preserved', upper.activeCut([3])?.source === 'slope');

  const lower = new AffineCutPool(1, 'lower', [
    {alpha: 1, beta: [1]},
    {alpha: 5, beta: [-1]},
  ]);
  check('1.3 lower envelope takes max cut', close(lower.evaluate([2]), 3));
  expectPrecondition('1.4 rejects wrong beta dimension',
    () => upper.add({alpha: 1, beta: [1, 2]}), 'cut.beta');
}

console.log('\n[2] Stage LP mechanics');
{
  const p = buildDefaultMultiStageInventoryProblem();
  const terminal = new AffineCutPool(1, 'upper', [{alpha: 0, beta: [p.salvageValue]}]);
  const dec = solveStageDecision(p, p.horizon - 1, 4, 6, terminal);
  check('2.1 stage LP optimal', dec.status === 'optimal');
  check('2.2 inventory balance holds',
    close(dec.nextInventory, 4 + dec.order - dec.sell), `next=${dec.nextInventory}`);
  check('2.3 demand balance holds',
    close(dec.sell + dec.stockout, 6), `sell+stockout=${dec.sell + dec.stockout}`);
  check('2.4 state and order bounds hold',
    dec.nextInventory >= -1e-8 && dec.nextInventory <= p.capacity + 1e-8 &&
    dec.order >= -1e-8 && dec.order <= p.maxOrder[p.horizon - 1] + 1e-8);
}

console.log('\n[3] Exact extensive-form scenario tree');
{
  const p = buildDefaultMultiStageInventoryProblem();
  const exact = solveExactScenarioTree(p);
  check('3.1 exact tree status optimal', exact.status === 'optimal');
  check('3.2 node count = 2 + 4 + 8 + 16', exact.nodeCount === 30, `nodes=${exact.nodeCount}`);
  check('3.3 exact objective finite and positive', Number.isFinite(exact.objective) && exact.objective > 0,
    `z=${exact.objective}`);
}

console.log('\n[4] SDDP converges to exact tree on the default 4-stage problem');
{
  const p = buildDefaultMultiStageInventoryProblem();
  const exact = solveExactScenarioTree(p);
  const sddp = solveMultiStageSDDP(p, {
    maxIter: 80,
    tol: 1e-4,
    seed: 3,
    exactObjective: exact.objective,
    evaluatePolicyEvery: 20,
    cutGridSize: 21,
  });
  check('4.1 SDDP declares optimal', sddp.status === 'optimal', `status=${sddp.status}`);
  check('4.2 upper bound is valid', sddp.upperBound + 1e-5 >= exact.objective,
    `upper=${sddp.upperBound}, exact=${exact.objective}`);
  check('4.3 exact-policy value matches extensive form',
    Math.abs(sddp.policyValue - exact.objective) <= 1e-4,
    `policy=${sddp.policyValue}, exact=${exact.objective}`);
  check('4.4 each nonterminal stage accumulated cuts',
    sddp.cutsPerStage.slice(0, p.horizon).every(n => n >= 2), `[${sddp.cutsPerStage.join(',')}]`);
  check('4.5 terminal stage has only salvage cut', sddp.cutsPerStage[p.horizon] === 1);
  check('4.6 trace has one row per iteration', sddp.trace.length === sddp.iterations);
}

console.log('\n[5] Policy evaluator and preconditions');
{
  const p = buildDefaultMultiStageInventoryProblem();
  const exact = solveExactScenarioTree(p);
  const sddp = solveMultiStageSDDP(p, {maxIter: 30, seed: 5, exactObjective: exact.objective});
  const pools = sddp.cuts.map(cuts => new AffineCutPool(1, 'upper', cuts));
  const policyValue = evaluatePolicyExact(p, pools);
  check('5.1 exported cuts reproduce reported policy value',
    close(policyValue, sddp.policyValue, 1e-6), `${policyValue} vs ${sddp.policyValue}`);
  expectPrecondition('5.2 rejects probability mass that does not sum to one',
    () => validateMultiStageProblem({
      ...p,
      demands: [
        [{demand: 1, prob: 0.2}, {demand: 2, prob: 0.2}],
        ...p.demands.slice(1),
      ],
    }), 'prob');
  expectPrecondition('5.3 rejects initial inventory above capacity',
    () => validateMultiStageProblem({...p, initialInventory: p.capacity + 1}), 'initialInventory');
}

console.log(`\n${pass + fail} tests: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
