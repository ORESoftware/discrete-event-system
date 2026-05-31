// RUST MIGRATION: target src/bin/main_dispatch_combo.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-dispatch-combo.rs   (fn main)
// 1:1 file move. One dispatch problem solved by DES + MDP + LP + MCTS plus
// heuristics, layered (engines / decision abstraction / DES).
//
// Conversion notes (file-specific):
//   - engines (simplex/value-iteration/MCTS/heuristics) -> use crate::des::
//     general::...
//   - MCTS rollouts + stochastic arrivals -> inject RandomSource/SeededRandom.
//   - heuristic set string union -> enum; top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-dispatch-combo.ts — DES + MDP + LP + MCTS on a single combinatorial
// optimisation problem (multi-class parallel-server dispatch).
//
// The user observation that motivated this:
//
//   "I am interested in using DES to help reformulate / reframe problems
//    that can be solved with LP or interior point algos etc, and basically
//    using DES to solve almost any problem via combinatorial methods
//    (slower but effective and handles stochastics well!)"
//
// The architectural pattern, from the user's question, instantiated here:
//
//   ┌────────────────────────────────────────────────────────────────┐
//   │                                                                │
//   │  Layer 3 — OPTIMISATION ENGINES (combinatorial / continuous)   │
//   │                                                                │
//   │   - simplex / interior-point (LP fluid relaxation)            │
//   │   - value iteration on the truncated MDP                       │
//   │   - Monte Carlo Tree Search (UCT) on the DES tree              │
//   │   - heuristics (random / round-robin / shortest-queue / SECT)  │
//   │                                                                │
//   ├────────────────────────────────────────────────────────────────┤
//   │                                                                │
//   │  Layer 2 — DECISION ABSTRACTION                                │
//   │                                                                │
//   │   MDP at arrival decision epochs; transition probabilities     │
//   │   estimated by running the DES from each (s, a) pair → the    │
//   │   "DES-as-MDP-simulator" pattern in plain code                 │
//   │                                                                │
//   ├────────────────────────────────────────────────────────────────┤
//   │                                                                │
//   │  Layer 1 — PHYSICAL / DYNAMIC WORLD                            │
//   │                                                                │
//   │   DES simulator: arrival events + service-completion events,   │
//   │   exponential inter-event times, FIFO per-machine queues,      │
//   │   class-dependent service rates                                │
//   │                                                                │
//   └────────────────────────────────────────────────────────────────┘
//
// Every policy is evaluated by the SAME DES with the SAME seeds, so
// head-to-head comparison is fair. The expected ordering on this
// problem (lower mean sojourn time = better):
//
//   MDP-VI ≤ MCTS ≤ Fluid-LP ≤ SECT ≤ SQ ≤ Round-Robin ≤ Random
//
// USAGE
// ─────
//   node dist/des/main-dispatch-combo.js
//   N_REPS=50 N_ARRIVALS=4000 node dist/des/main-dispatch-combo.js
//   LP_SOLVER=scipy:highs-ipm node dist/des/main-dispatch-combo.js
//   LP_SOLVER=internal node dist/des/main-dispatch-combo.js
//   SKIP_MDP=1 node dist/des/main-dispatch-combo.js     # skip exact VI (slow build)
// =============================================================================

import {
  DispatchProblem, DispatchPolicy,
  simulateDispatch, evaluatePolicy, welchT,
  policyRandom, policyRoundRobin, policyShortestQueue, policySECT,
  policyFluidLP, policyMDPVI, policyMCTS,
  buildDispatchFluidLP,
} from './general/dispatch';
import {lpToString} from './general/lp';

async function main(): Promise<void> {
  // ---------- Problem instance (small enough for exact MDP-VI) ----------
  const problem: DispatchProblem = {
    M: 2, K: 2,
    arrivalRate: 1.6,
    classProb: [0.6, 0.4],
    serviceRate: [
      // class-1 row (machine-1 fast for this class)
      [2.0, 0.8],
      // class-2 row (machine-2 fast for this class)
      [0.8, 2.0],
    ],
  };
  const numReps = Number(process.env.N_REPS ?? 30);
  const numArrivals = Number(process.env.N_ARRIVALS ?? 3000);
  const warmup = Math.floor(numArrivals * 0.1);
  const seedBase = Number(process.env.SEED_BASE ?? 1000);
  const skipMDP = process.env.SKIP_MDP === '1';

  // ---------- Banner ----------
  console.log('# Multi-class dispatch — DES + MDP + LP + MCTS combo');
  console.log(`# M=${problem.M} machines, K=${problem.K} classes`);
  console.log(`# arrival rate λ = ${problem.arrivalRate}`);
  console.log(`# class probs    = [${problem.classProb.join(', ')}]`);
  console.log(`# service rates μ_{c,m}:`);
  for (let c = 0; c < problem.K; c++) {
    console.log(`#   class ${c + 1}: [${problem.serviceRate[c].map(v => v.toFixed(2)).join(', ')}]`);
  }
  // Theoretical traffic intensity if perfectly balanced:
  let totalCapacity = 0;
  for (let m = 0; m < problem.M; m++) {
    let cap = 0;
    for (let c = 0; c < problem.K; c++) cap += problem.classProb[c] * problem.serviceRate[c][m];
    totalCapacity += cap;
  }
  console.log(`# total capacity Σ_m Σ_c p_c μ_{c,m} = ${totalCapacity.toFixed(3)}`);
  console.log(`# theoretical ρ_avg if balanced = λ / capacity = ${(problem.arrivalRate / totalCapacity).toFixed(3)}`);
  console.log(`# (should be < 1 for stability)`);
  console.log('');
  console.log(`# Replications per policy = ${numReps}`);
  console.log(`# Arrivals per replication = ${numArrivals}  (first ${warmup} discarded as warmup)`);
  console.log('');

  // ---------- Show the LP relaxation ----------
  console.log('# Layer-3 LP fluid relaxation (solved via simplex / interior-point):');
  const lp = buildDispatchFluidLP(problem);
  for (const line of lpToString(lp).split('\n')) console.log('#   ' + line);
  console.log('');

  // ---------- Policy registry ----------
  type Entry = {name: string; factory: () => DispatchPolicy; note?: string};
  const fluidResult = policyFluidLP(problem);
  console.log(`# Fluid LP solved via ${fluidResult.solver} in ${fluidResult.iters} iterations`);
  console.log(`#   bottleneck load t* = max_m ρ_m = ${fluidResult.bottleneckLoad.toFixed(4)}`);
  for (let c = 0; c < problem.K; c++) {
    console.log(`#   class ${c + 1} → x* = [${fluidResult.x[c].map(v => v.toFixed(3)).join(', ')}]`);
  }
  console.log('');

  let mdpResult: ReturnType<typeof policyMDPVI> | null = null;
  if (!skipMDP) {
    process.stdout.write('# Building MDP via DES rollouts and running value iteration ... ');
    const t0 = Date.now();
    mdpResult = policyMDPVI(problem, {qMax: 5, gamma: 0.95, rolloutsPerSA: 50});
    console.log(`done in ${Date.now() - t0}ms (|S|=${mdpResult.numStates}, qMax=${mdpResult.qMax})`);
    console.log('');
  }

  const policies: Entry[] = [
    {name: 'random',          factory: () => policyRandom(13), note: 'Layer 3: trivial baseline'},
    {name: 'round-robin',     factory: () => policyRoundRobin(), note: 'Layer 3: state-blind heuristic'},
    {name: 'shortest-queue',  factory: () => policyShortestQueue(), note: 'Layer 3: queue-aware heuristic'},
    {name: 'SECT',            factory: () => policySECT(problem), note: 'Layer 3: class-aware heuristic'},
    {name: 'fluid-LP',        factory: () => policyFluidLP(problem).policy, note: 'Layer 3: simplex / interior-point on the fluid relaxation'},
  ];
  if (mdpResult) policies.push({name: 'MDP-VI', factory: () => mdpResult!.policy, note: 'Layer 3 ∘ Layer 2: value iteration on the empirical MDP whose transitions came from DES rollouts'});
  policies.push({name: 'MCTS',           factory: () => policyMCTS(problem, {iterations: 200, rolloutDepth: 35}), note: 'Layer 3 ∘ Layer 1: tree search using DES as the rollout oracle'});

  // ---------- Evaluation ----------
  const results = policies.map(p => {
    process.stdout.write(`# Evaluating '${p.name}' (${numReps} reps × ${numArrivals} arrivals) ... `);
    const t0 = Date.now();
    const r = evaluatePolicy(problem, p.factory, p.name, numReps, numArrivals, seedBase, warmup);
    console.log(`done in ${Date.now() - t0}ms,  mean sojourn = ${r.meanWait.toFixed(4)}`);
    return r;
  });
  console.log('');

  // ---------- Comparison table ----------
  console.log('# ' + '─'.repeat(78));
  console.log('# Mean sojourn time per policy (lower is better):');
  console.log('# ' + '─'.repeat(78));
  console.log(`#   ${'policy'.padEnd(18)}${'mean sojourn'.padStart(15)}${'sd'.padStart(11)}${'utilisation'.padStart(20)}`);
  for (const r of results) {
    const utilStr = '[' + r.utilisation.map(u => (u * 100).toFixed(1) + '%').join(', ') + ']';
    console.log(`#   ${r.policyName.padEnd(18)}${r.meanWait.toFixed(4).padStart(15)}${r.sdWait.toFixed(4).padStart(11)}${utilStr.padStart(20)}`);
  }
  console.log('');

  // ---------- Welch t-tests against the random baseline ----------
  console.log('# ' + '─'.repeat(78));
  console.log('# Welch t-statistic vs random (large positive ⇒ policy is significantly better):');
  console.log('# ' + '─'.repeat(78));
  const random = results.find(r => r.policyName === 'random')!;
  for (const r of results) {
    if (r.policyName === 'random') continue;
    const t = welchT(random.rawWaits, r.rawWaits);
    console.log(`#   random vs ${r.policyName.padEnd(18)} t = ${t.toFixed(2)}    Δmean = ${(random.meanWait - r.meanWait).toFixed(4)}`);
  }
  console.log('');

  // ---------- Architectural recap ----------
  console.log('# ' + '─'.repeat(78));
  console.log('# Architectural recap:');
  console.log('# ' + '─'.repeat(78));
  for (const p of policies) {
    console.log(`#   ${p.name.padEnd(18)} → ${p.note ?? ''}`);
  }
  console.log('');
  console.log('#   The same DES (`simulateDispatch`) evaluates EVERY policy.');
  console.log('#   The MDP-VI policy uses the DES as its transition oracle.');
  console.log('#   The MCTS policy uses the DES as its rollout simulator.');
  console.log('#   The fluid-LP policy is a randomised assignment from the LP relaxation,');
  console.log('#     which the simplex / interior-point of choice solves in milliseconds.');
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
