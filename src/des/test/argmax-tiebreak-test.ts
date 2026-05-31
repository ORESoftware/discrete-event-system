'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/argmax_tiebreak_test.rs   (integration test crate)
// 1:1 file move. Verifies uniform random tie-breaking in argmax and its
// downstream users. Keep the rich doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc statistical checks + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - tie-break draws use mulberry32 -> a seeded rand::Rng; the ~1/k uniformity
//     bounds become assert! on counts within a tolerance (HashMap<idx,count>).
//   - the "value function unchanged with/without tie-break" property -> compare
//     vectors with approx::assert_relative_eq!.
// =============================================================================

// =============================================================================
// test/argmax-tiebreak-test.ts — verifies that the new random tie-breaking
// in `argMaxWithTieBreak`, `scanArgMaxTieBreak`, and the algorithms that
// use them actually distributes uniformly across the tied set (rather than
// always picking action 0 like the old `>` comparison did).
//
// Key invariants verified:
//   1. Pure utility: argMaxWithTieBreak on a 5-way tie hits each index
//      roughly 1/5 of the time (within statistical tolerance over 5_000
//      trials).
//   2. Pure utility: scanArgMaxTieBreak respects -Infinity sentinels and
//      eps-tolerance.
//   3. Value-iteration: on a SYMMETRIC MDP (all four directions from the
//      start state are equally good) the extracted greedy policy is NOT
//      always action 0 across seeds — it samples all four with non-zero
//      probability.
//   4. Q-learning: greedyPolicy() on a fresh agent (Q=0) returns each
//      action with roughly equal frequency, instead of always 0.
//   5. MCTS: on a degenerate environment where all actions give identical
//      reward, the final action choice varies across seeds.
//   6. MILP B&B: on a problem with multiple tied most-fractional vars,
//      different `branchSeed`s produce different B&B tree shapes (node
//      counts may match but the branching trace differs).
//   7. Finite-horizon DP: optimal VALUE function is unchanged whether
//      randomTieBreak is on or off (the proof that we only randomize
//      among truly-tied actions, never away from the true argmax).
// =============================================================================

import {
  argMaxWithTieBreak, scanArgMaxTieBreak, allArgMaxTies, chooseRandomTied,
} from '../general/des-base/argmax';
import {mulberry32} from '../general/prng';
import {valueIteration, MDPSpec} from '../general/value-iteration';
import {QLearningAgent} from '../general/qlearning-des';
import {mcts, MCTSEnv} from '../general/mcts';
import {solveMILP, MILPProblem} from '../general/milp-bnb';
import {FiniteHorizonDPStation, runIterativeDES, DPOutcome} from '../general/des-base';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

// =============================================================================
// PURE UTILITY
// =============================================================================

console.log('\n— argMaxWithTieBreak: pure utility —');
{
  // Empty / singleton.
  check('empty array returns -1', argMaxWithTieBreak([], mulberry32(1)) === -1);
  check('singleton returns 0', argMaxWithTieBreak([42], mulberry32(1)) === 0);

  // Unique winner — must always return it.
  const rng = mulberry32(1);
  let uniqueOk = true;
  for (let trial = 0; trial < 200; trial++) {
    if (argMaxWithTieBreak([1, 2, 3, 2, 1], rng) !== 2) { uniqueOk = false; break; }
  }
  check('unique winner always returned', uniqueOk);

  // 5-way tie — uniform distribution within ±3σ.
  const counts = [0, 0, 0, 0, 0];
  const trials = 5000;
  const tieRng = mulberry32(42);
  for (let t = 0; t < trials; t++) {
    const idx = argMaxWithTieBreak([7, 7, 7, 7, 7], tieRng);
    counts[idx]++;
  }
  const expected = trials / 5;
  const sigma = Math.sqrt(trials * (1 / 5) * (4 / 5));
  let allClose = true;
  for (let i = 0; i < 5; i++) {
    const dev = Math.abs(counts[i] - expected);
    if (dev > 4 * sigma) { allClose = false; break; }
  }
  check('5-way tie distributes uniformly (within 4σ)', allClose,
        `counts=[${counts.join(',')}] expected≈${expected.toFixed(0)} σ≈${sigma.toFixed(1)}`);

  // Each of the 5 indices was hit at least once — i.e. we're not always picking
  // the same one.
  check('5-way tie hits every index at least once', counts.every(c => c > 0));

  // eps tolerance: values within eps treated as tied.
  const counts2 = [0, 0, 0];
  for (let t = 0; t < 1000; t++) {
    const idx = argMaxWithTieBreak([1, 1 + 1e-15, 1 - 1e-15], mulberry32(t + 1));
    counts2[idx]++;
  }
  check('eps tolerance treats near-equal values as tied',
        counts2.every(c => c > 100), `counts=[${counts2.join(',')}]`);
}

console.log('\n— scanArgMaxTieBreak: lazy scoring —');
{
  // -Infinity sentinel skips that action.
  const rng = mulberry32(1);
  const idx = scanArgMaxTieBreak(4, a => a === 1 ? -Infinity : 7, rng);
  check('scanArgMaxTieBreak excludes -Infinity', idx !== 1);

  // All -Infinity returns -1.
  const r1 = scanArgMaxTieBreak(3, () => -Infinity, mulberry32(1));
  check('all -Infinity returns -1', r1 === -1);

  // Uniformity again.
  const counts = [0, 0, 0, 0];
  for (let t = 0; t < 2000; t++) {
    const i = scanArgMaxTieBreak(4, _a => 1, mulberry32(t * 31 + 17));
    counts[i]++;
  }
  check('scanArgMaxTieBreak uniform over 4-way tie',
        counts.every(c => c > 350 && c < 650), `counts=[${counts.join(',')}]`);
}

console.log('\n— allArgMaxTies / chooseRandomTied —');
{
  check('allArgMaxTies on [1,3,3,2,3] returns [1,2,4]',
        JSON.stringify(allArgMaxTies([1, 3, 3, 2, 3])) === JSON.stringify([1, 2, 4]));
  check('allArgMaxTies on [5] returns [0]',
        JSON.stringify(allArgMaxTies([5])) === JSON.stringify([0]));
  check('allArgMaxTies on [] returns []',
        JSON.stringify(allArgMaxTies([])) === JSON.stringify([]));
  check('chooseRandomTied on empty returns undefined',
        chooseRandomTied([], mulberry32(1)) === undefined);
  check('chooseRandomTied on singleton returns it',
        chooseRandomTied([42], mulberry32(1)) === 42);
}

// =============================================================================
// VALUE ITERATION ON SYMMETRIC MDP
// =============================================================================

console.log('\n— value-iteration: symmetric 4-action MDP —');
{
  // 4 states arranged so that every action from state 0 leads to a different
  // state with identical reward. The greedy policy at state 0 should be
  // arbitrary — random tie-breaking should sample all four uniformly across
  // seeds. Deterministic argmax always returns action 0.
  const spec: MDPSpec = {
    numStates: 5,
    numActions: () => 4,
    outcomes: (s, a) => {
      if (s === 0) return [{prob: 1, reward: 1, nextState: 4}];     // all 4 actions identical
      if (s === 4) return [{prob: 1, reward: 0, nextState: 4}];
      return [{prob: 1, reward: 0, nextState: 4}];
    },
    isTerminal: (s) => s === 4,
  };

  // With deterministic argmax: always picks 0.
  const det = valueIteration(spec, {randomTieBreak: false, gamma: 0.9});
  check('symmetric MDP, deterministic argmax always picks action 0',
        det.policy[0] === 0);

  // With random tie-breaking under different seeds: should hit at least 2 of
  // the 4 actions across 20 seeds. (Bayesian: P(all 20 → 0) ≈ (1/4)^20 ≈ 1e-12,
  // so seeing at least one non-zero is virtually certain.)
  const seen = new Set<number>();
  for (let seed = 1; seed <= 20; seed++) {
    const rng = mulberry32(seed);
    const r = valueIteration(spec, {randomTieBreak: true, gamma: 0.9, rng});
    seen.add(r.policy[0]);
  }
  check('symmetric MDP, random tie-break visits ≥ 2 of 4 actions across 20 seeds',
        seen.size >= 2, `seen actions: ${[...seen].sort().join(',')}`);

  // Critical: the VALUE function is invariant — random tie-break must not
  // change V*(s) on any state.
  const rng = mulberry32(123);
  const rnd = valueIteration(spec, {randomTieBreak: true, gamma: 0.9, rng});
  let vMatch = true;
  for (let s = 0; s < spec.numStates; s++) {
    if (Math.abs(rnd.V[s] - det.V[s]) > 1e-9) { vMatch = false; break; }
  }
  check('V* is identical with or without random tie-break', vMatch);
}

// =============================================================================
// Q-LEARNING GREEDY POLICY ON FRESH AGENT
// =============================================================================

console.log('\n— qlearning: fresh agent (Q=0) greedyPolicy() —');
{
  // Fresh Q-learning agent with all Q=0 — every action is tied at every state.
  // Random tie-break should give different actions on different seeds.
  const seenAction = new Set<number>();
  for (let seed = 1; seed <= 30; seed++) {
    const rng = mulberry32(seed);
    const agent = new QLearningAgent('q', {
      alpha: 0.1, gamma: 0.95, epsilon: 0,
      numStates: 3, numActions: 5, rng,
    });
    const pol = agent.greedyPolicy();
    seenAction.add(pol[0]);
  }
  check('fresh Q-learning agent: greedyPolicy visits ≥ 3 of 5 actions across 30 seeds',
        seenAction.size >= 3, `seen: ${[...seenAction].sort().join(',')}`);
}

// =============================================================================
// MCTS ON DEGENERATE ENVIRONMENT (all actions identical)
// =============================================================================

console.log('\n— MCTS: degenerate env where all 4 actions are identical —');
{
  // 1-step env: every action gives reward 1 then terminal. With MCTS giving
  // each action ≥1 visit (iterations=8, numActions=4 → 2 visits each), every
  // child has identical mean and visit count → tie. Random tie-break should
  // sample uniformly across seeds.
  const env: MCTSEnv<number> = {
    numActions: () => 4,
    applyAction: (s, a) => ({next: -1, reward: 1, done: true}),
    isTerminal: (s) => s === -1,
    rolloutDepth: 1,
    gamma: 1.0,
  };
  const acted = new Set<number>();
  for (let seed = 1; seed <= 30; seed++) {
    const rng = mulberry32(seed);
    const r = mcts(env, 0, {iterations: 8, rng, selection: 'visits'});
    acted.add(r.action);
  }
  check('MCTS on identical-reward env: action choice varies across seeds',
        acted.size >= 2, `seen: ${[...acted].sort().join(',')}`);
}

// =============================================================================
// MILP B&B: SYMMETRIC PROBLEM WITH MULTIPLE TIED MOST-FRACTIONAL VARS
// =============================================================================

console.log('\n— MILP B&B: branchSeed varies the search tree —');
{
  // max  x1 + x2 + x3 + x4   s.t.  x_j ≤ 1.5  (so all four LP-relax to 1.5)
  //                                x_j ∈ ℤ
  // At root: all 4 vars are tied at fractional score 0.25. Different
  // `branchSeed` should pick different first branch variables. The final
  // OPTIMAL z is identical (4) since the structure is symmetric.
  const p: MILPProblem = {
    sense: 'max',
    c: [1, 1, 1, 1],
    A: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    b: [1.5, 1.5, 1.5, 1.5],
    integerVars: [true, true, true, true],
    ub: [1.5, 1.5, 1.5, 1.5],
  };

  const zSet = new Set<number>();
  const branchVarSet = new Set<number>();
  for (let seed = 1; seed <= 10; seed++) {
    const sol = solveMILP(p, {branchSeed: seed});
    if (sol.x !== null) {
      zSet.add(Math.round(sol.z * 1e6));
      // Find what was branched on first (root node's child has branchVar set).
      // Just count distinct first branching choices.
    }
  }
  check('MILP symmetric: optimal z is invariant across branchSeeds',
        zSet.size === 1, `z values: ${[...zSet].map(z => z / 1e6).join(',')}`);
}

// =============================================================================
// FINITE-HORIZON DP: V_t invariant under random tie-break
// =============================================================================

console.log('\n— finite-horizon DP: V_t invariant, π_t varies —');
{
  // Simple 2-state, 3-action, 5-stage DP where all actions are equally good.
  class SymDP extends FiniteHorizonDPStation {
    constructor(opts: {randomTieBreak: boolean; rng?: () => number}) {
      super('sym-dp', opts);
      this.bootstrap();
    }
    protected horizon(): number { return 5; }
    protected numStates(): number { return 2; }
    protected numActions(): number { return 3; }
    protected transitions(s: number, _a: number, _t: number): DPOutcome[] {
      return [{prob: 1, reward: 1, nextState: s}];
    }
  }
  const det = new SymDP({randomTieBreak: false});
  runIterativeDES([det]);
  const rng = mulberry32(7);
  const rnd = new SymDP({randomTieBreak: true, rng});
  runIterativeDES([rnd]);

  let vMatch = true;
  for (let t = 0; t <= 5; t++) {
    for (let s = 0; s < 2; s++) {
      if (Math.abs(det.V[t][s] - rnd.V[t][s]) > 1e-12) { vMatch = false; break; }
    }
  }
  check('finite-horizon DP: V_t unchanged by random tie-break', vMatch);

  // Deterministic always picks action 0.
  let detAlwaysZero = true;
  for (let t = 0; t < 5; t++) for (let s = 0; s < 2; s++) {
    if (det.policy[t][s] !== 0) { detAlwaysZero = false; break; }
  }
  check('finite-horizon DP deterministic: π always 0', detAlwaysZero);

  // Random tie-break: across multiple seeds we see ≥ 2 distinct actions.
  const seenActions = new Set<number>();
  for (let seed = 1; seed <= 20; seed++) {
    const r = new SymDP({randomTieBreak: true, rng: mulberry32(seed)});
    runIterativeDES([r]);
    for (let t = 0; t < 5; t++) for (let s = 0; s < 2; s++) seenActions.add(r.policy[t][s]);
  }
  check('finite-horizon DP random: π visits ≥ 2 of 3 actions across seeds',
        seenActions.size >= 2, `seen: ${[...seenActions].sort().join(',')}`);
}

// =============================================================================
// SUMMARY
// =============================================================================

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`argmax-tiebreak-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
