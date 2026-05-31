'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/dispatch_test.rs   (integration test crate)
// 1:1 file move. Spans dispatch / lp / lp-des / mcts, so it is an integration
// test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc eq()/expect()/close()/pass-fail counters + console.log  ->  #[test]
//   fns using assert!/assert_eq!; drop the manual tally and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - eq() compares via JSON.stringify -> derive PartialEq and use assert_eq!.
//   - close(a,b,tol) float comparison -> approx::assert_relative_eq!.
//   - policies use mulberry32 / MCTS rollouts -> a seeded rand::Rng so the
//     simulated dispatch outcomes are reproducible.
// =============================================================================

// =============================================================================
// test/dispatch-test.ts — unit tests for the multi-class dispatch combo.
// =============================================================================

import {
  DispatchProblem, simulateDispatch,
  policyRandom, policyRoundRobin, policyShortestQueue, policySECT,
  policyFluidLP, policyMDPVI,
  buildDispatchFluidLP, evaluatePolicy,
} from '../general/dispatch';
import {solveLPInternal} from '../general/lp';
import {solveLPViaDES} from '../general/lp-des';
import {mcts, MCTSEnv} from '../general/mcts';
import {mulberry32} from '../general/prng';

let pass = 0;
let fail = 0;
function eq(label: string, a: any, b: any): void {
  const ok = JSON.stringify(a) === JSON.stringify(b);
  console.log((ok ? '  PASS' : '  FAIL') + '  ' + label
    + (ok ? '' : `  expected ${JSON.stringify(b)}  got ${JSON.stringify(a)}`));
  ok ? pass++ : fail++;
}
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' — ' + detail : ''));
  cond ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-9): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} − ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

const problem2x2: DispatchProblem = {
  M: 2, K: 2,
  arrivalRate: 1.6,
  classProb: [0.6, 0.4],
  serviceRate: [[2.0, 0.8], [0.8, 2.0]],
};

const problem3x3: DispatchProblem = {
  M: 3, K: 3,
  arrivalRate: 2.4,
  classProb: [1 / 3, 1 / 3, 1 / 3],
  serviceRate: [[1.6, 0.9, 0.7], [0.7, 1.6, 0.9], [0.9, 0.7, 1.6]],
};

// -----------------------------------------------------------------------------
console.log('\nGroup 1 — DES simulator basic invariants');
// -----------------------------------------------------------------------------
{
  const r = simulateDispatch(problem2x2, policyRoundRobin(), 1000, 42, 0);
  expect('completedJobs is positive', r.completedJobs > 0, `${r.completedJobs}`);
  expect('per-machine job counts sum to total arrivals',
    r.perMachineJobs.reduce((a, b) => a + b, 0) === 1000,
    `Σ = ${r.perMachineJobs.reduce((a, b) => a + b, 0)}`);
  expect('mean sojourn is finite and positive',
    Number.isFinite(r.meanSojourn) && r.meanSojourn > 0,
    `meanSojourn = ${r.meanSojourn}`);
  expect('utilisation is in [0, 1] per machine',
    r.perMachineUtilisation.every(u => u >= 0 && u <= 1),
    `util = ${JSON.stringify(r.perMachineUtilisation.map(u => u.toFixed(3)))}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 — Round-robin distributes evenly');
// -----------------------------------------------------------------------------
{
  const r = simulateDispatch(problem3x3, policyRoundRobin(), 3000, 7, 0);
  const counts = r.perMachineJobs;
  // Each machine should receive within ±1 of N/M jobs.
  expect('round-robin per-machine counts within ±1 of N/M',
    counts.every(c => Math.abs(c - 1000) <= 1),
    `counts = ${JSON.stringify(counts)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 — SECT chooses the class-aligned machine when queues are empty');
// -----------------------------------------------------------------------------
{
  const policy = policySECT(problem2x2);
  // For class 0: μ_{0,0} = 2.0, μ_{0,1} = 0.8. Empty queues. SECT should pick machine 0.
  const a0 = policy.pick({M: 2, K: 2, q: [0, 0], idleUntil: [0, 0], inService: [-1, -1], now: 0}, 0);
  expect('SECT class-0 → machine 0 when both queues empty', a0 === 0, `got ${a0}`);
  const a1 = policy.pick({M: 2, K: 2, q: [0, 0], idleUntil: [0, 0], inService: [-1, -1], now: 0}, 1);
  expect('SECT class-1 → machine 1 when both queues empty', a1 === 1, `got ${a1}`);
  // Now suppose class-0's home machine has 5 jobs queued. SECT should still pick
  // it because (5+1)/2.0 = 3.0 < (0+1)/0.8 = 1.25? Wait, 3.0 > 1.25 so SECT
  // should pick machine 1 in this case.
  const a2 = policy.pick({M: 2, K: 2, q: [5, 0], idleUntil: [0, 0], inService: [-1, -1], now: 0}, 0);
  expect('SECT class-0 with q=[5,0] picks machine 1 (overflow)',
    a2 === 1, `got ${a2}, (5+1)/2.0=3.0 vs (0+1)/0.8=1.25`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 — Shortest-queue picks the lower queue index ties');
// -----------------------------------------------------------------------------
{
  const policy = policyShortestQueue();
  const a = policy.pick({M: 3, K: 1, q: [3, 1, 5], idleUntil: [0, 0, 0], inService: [-1, -1, -1], now: 0}, 0);
  expect('SQ q=[3,1,5] → machine 1', a === 1, `got ${a}`);
  // Tie at 0 between machines 0 and 1 — should pick machine 0 (deterministic).
  const a2 = policy.pick({M: 3, K: 1, q: [0, 0, 2], idleUntil: [0, 0, 0], inService: [-1, -1, -1], now: 0}, 0);
  expect('SQ q=[0,0,2] → machine 0 (tie-break low index)', a2 === 0, `got ${a2}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 — Fluid LP shape and feasibility');
// -----------------------------------------------------------------------------
{
  const lp = buildDispatchFluidLP(problem3x3);
  // K=3, M=3 ⇒ K*M+1 = 10 vars. K=3 equalities + M=3 inequalities.
  expect('fluid LP has K*M+1 = 10 variables', lp.c.length === 10, `got ${lp.c.length}`);
  expect('fluid LP has K = 3 equality rows', (lp.A_eq?.length ?? 0) === 3,
    `got ${lp.A_eq?.length}`);
  expect('fluid LP has M = 3 inequality rows', (lp.A_ub?.length ?? 0) === 3,
    `got ${lp.A_ub?.length}`);
  // Objective is c[t] = 1, c[x_{c,m}] = 0.
  for (let i = 0; i < 9; i++) eq(`fluid LP c[x_${i}] = 0`, lp.c[i], 0);
  eq('fluid LP c[t] = 1', lp.c[9], 1);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 — Fluid LP cross-solver bit-exact agreement');
// -----------------------------------------------------------------------------
{
  const lp = buildDispatchFluidLP(problem3x3);
  const a = solveLPInternal(lp);
  const b = solveLPViaDES(lp);
  expect('fluid LP — internal simplex returns optimal',
    a.status === 'optimal', `status=${a.status}`);
  expect('fluid LP — DES-engine simplex returns optimal',
    b.status === 'optimal', `status=${b.status}`);
  if (a.status === 'optimal' && b.status === 'optimal') {
    close('fluid LP — internal vs DES simplex objectives within 1e-9',
      a.objective, b.objective, 1e-9);
    // Check x agreement.
    const dx = Math.max(...a.x.map((v, i) => Math.abs(v - b.x[i])));
    expect('fluid LP — internal vs DES simplex x within 1e-6',
      dx < 1e-6, `max |Δx| = ${dx.toExponential(2)}`);
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 — Fluid LP policy: x* sums to 1 per class, ≥ 0');
// -----------------------------------------------------------------------------
{
  const r = policyFluidLP(problem3x3);
  for (let c = 0; c < problem3x3.K; c++) {
    const sum = r.x[c].reduce((a, b) => a + b, 0);
    close(`fluid-LP class ${c}: Σ_m x_{c,m} = 1`, sum, 1, 1e-6);
    expect(`fluid-LP class ${c}: x_{c,m} ≥ 0 ∀ m`,
      r.x[c].every(v => v >= -1e-9), `x[${c}] = ${JSON.stringify(r.x[c])}`);
  }
  expect('fluid-LP bottleneck load t* in (0, 1]',
    r.bottleneckLoad > 0 && r.bottleneckLoad <= 1, `t* = ${r.bottleneckLoad}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 8 — MDP-VI policy returns a legal action at every state');
// -----------------------------------------------------------------------------
{
  const r = policyMDPVI(problem2x2, {qMax: 3, gamma: 0.95, rolloutsPerSA: 30, seed: 1});
  expect('MDP-VI |S| = (qMax+1)^M × K', r.numStates === Math.pow(4, 2) * 2,
    `|S| = ${r.numStates}`);
  // Probe a few states for legal actions.
  for (let q1 = 0; q1 <= 3; q1++) {
    for (let q2 = 0; q2 <= 3; q2++) {
      for (let c = 0; c < 2; c++) {
        const a = r.policy.pick({M: 2, K: 2, q: [q1, q2], idleUntil: [0, 0], inService: [-1, -1], now: 0}, c);
        expect(`MDP-VI policy.pick(q=[${q1},${q2}], c=${c}) ∈ [0, 2)`,
          a === 0 || a === 1, `got ${a}`);
      }
    }
  }
}

// -----------------------------------------------------------------------------
console.log('\nGroup 9 — MCTS finds the obvious optimal action in a tiny deterministic env');
// -----------------------------------------------------------------------------
{
  // Two-action, single-state, deterministic env: action 0 → reward 1, action 1 → reward 10.
  // MCTS should overwhelmingly visit action 1 after a handful of iterations.
  type S = {step: number};
  const env: MCTSEnv<S> = {
    numActions: () => 2,
    applyAction: (s, a) => ({next: {step: s.step + 1}, reward: a === 0 ? 1 : 10, done: s.step + 1 >= 1}),
    isTerminal: (s) => s.step >= 1,
    rolloutDepth: 0,
    gamma: 1.0,
  };
  const rng = mulberry32(7);
  const result = mcts(env, {step: 0}, {iterations: 50, rng: () => rng()});
  expect('MCTS picks action 1 (reward 10 > 1)', result.action === 1,
    `got ${result.action}, visits=${JSON.stringify(Array.from(result.visits.entries()))}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 10 — Reproducibility: same seed ⇒ same result');
// -----------------------------------------------------------------------------
{
  const a = simulateDispatch(problem2x2, policySECT(problem2x2), 500, 1234, 0);
  const b = simulateDispatch(problem2x2, policySECT(problem2x2), 500, 1234, 0);
  close('seed=1234 ⇒ same mean sojourn', a.meanSojourn, b.meanSojourn, 0);
  eq('seed=1234 ⇒ same per-machine counts', a.perMachineJobs, b.perMachineJobs);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 11 — MDP-VI reward-model alignment');
// In a tiny deterministic instance where one machine clearly dominates
// for a class, the MDP-VI policy should pick that machine when both
// queues are empty. (Same property SECT has — but MDP-VI is the
// solver, not the heuristic.)
// -----------------------------------------------------------------------------
{
  const r = policyMDPVI(problem2x2, {qMax: 4, gamma: 0.95, rolloutsPerSA: 100, seed: 2});
  const a = r.policy.pick({M: 2, K: 2, q: [0, 0], idleUntil: [0, 0], inService: [-1, -1], now: 0}, 0);
  const b = r.policy.pick({M: 2, K: 2, q: [0, 0], idleUntil: [0, 0], inService: [-1, -1], now: 0}, 1);
  expect('MDP-VI: empty system, class 0 → machine 0 (μ=2.0 vs 0.8)', a === 0, `got ${a}`);
  expect('MDP-VI: empty system, class 1 → machine 1 (μ=2.0 vs 0.8)', b === 1, `got ${b}`);
}

console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
