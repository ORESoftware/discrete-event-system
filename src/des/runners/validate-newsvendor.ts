#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-newsvendor.rs  (a `fn main` binary;
//                    an `examples/…rs` also works)
// 1:1 file move. Validates the newsvendor and multi-period inventory MDP
// (critical-fractile vs brute search vs value iteration; (s,S) policy structure).
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env + `child_process` (reference) -> `std::env::var` / `std::process::Command`.
//   - `JSON` -> `serde_json`.
//   - `as any` on results -> concrete typed structs.
//   - demand sampling (Poisson/Uniform) -> inject `SeededRandom`.
//   - `process.exit(code)` -> `std::process::exit(code)`.
// =============================================================================

// =============================================================================
// Validate the newsvendor and multi-period inventory MDP.
//
//   STUDY 1: Single-period newsvendor — three solution methods agree
//     (a) Critical-fractile analytical formula
//     (b) Brute-search over q ∈ [0, qMax]
//     (c) MDP value iteration (γ = 0, 1-step)
//     All three should produce the same q* and the same E[profit(q*)].
//     Tested on multiple parameter regimes (low/high price, with/without
//     salvage, Poisson and Uniform demand).
//
//   STUDY 2: Multi-period MDP recovers single-period at γ → 0
//     With γ = 0 and inventory cap forcing zero leftover, the
//     multi-period MDP must produce q* = newsvendor q*.
//
//   STUDY 3: Multi-period optimal policy structure
//     With fixedCost = 0, the optimal policy is base-stock.
//     With fixedCost > 0, the optimal policy is (s, S).
//     Both are pinned without any structural assumption — VI just
//     finds the optimum over all action mappings, and we check the
//     output has the expected shape.
//
//   STUDY 4: Simulation matches Bellman value
//     Long-run average reward of a long simulation under the
//     discovered policy ≈ V(0) · (1 − γ). Tied to within stochastic
//     tolerance.
// =============================================================================

import {execFileSync} from 'child_process';
import * as path from 'path';
import {
  analyticalOptimalQ, bruteSearchOptimalQ, demandPoissonPMF, demandUniformPMF,
  expectedProfit, mdpOptimalQ, NewsvendorParams, simulate,
} from '../main-newsvendor';
import {
  detectPolicyStructure, inventoryMDPSpec, InventoryParams,
  simulateInventoryMDP,
} from '../main-inventory-mdp';
import {valueIteration} from '../general/value-iteration';

const PYTHON = process.env.NEWSVENDOR_PY ?? 'python3';
const PY_SCRIPT = path.join(__dirname, '..', '..', '..', 'external-references', 'newsvendor', 'newsvendor.py');

function runPython(args: string[]): any | null {
  try {
    const out = execFileSync(PYTHON, [PY_SCRIPT, ...args], {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
    return JSON.parse(out);
  } catch (err) {
    return null;
  }
}

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string) {
  if (ok) { pass++; console.log(`  PASS    ${label}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}
function approx(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

// =============================================================================
console.log('\nStudy 1  Newsvendor: critical-fractile ≡ brute-search ≡ MDP');
console.log('==========================================================================');

const SCENARIOS: Array<{name: string; params: NewsvendorParams}> = [
  {
    name: 'classic Poisson, λ=50, p/c/s = 1.0/0.5/0.1',
    params: {
      unitCost: 0.5, unitPrice: 1.0, unitSalvage: 0.1,
      demand: demandPoissonPMF(50, 125), qMax: 125,
    },
  },
  {
    name: 'high margin, low salvage, λ=20',
    params: {
      unitCost: 0.3, unitPrice: 2.0, unitSalvage: 0,
      demand: demandPoissonPMF(20, 60), qMax: 60,
    },
  },
  {
    name: 'low margin, high salvage, λ=100',
    params: {
      unitCost: 0.9, unitPrice: 1.0, unitSalvage: 0.7,
      demand: demandPoissonPMF(100, 200), qMax: 200,
    },
  },
  {
    name: 'uniform demand U[10, 30]',
    params: {
      unitCost: 0.5, unitPrice: 1.0, unitSalvage: 0.1,
      demand: demandUniformPMF(10, 30, 40), qMax: 40,
    },
  },
];

for (const scen of SCENARIOS) {
  console.log(`\n  ${scen.name}`);
  const a = analyticalOptimalQ(scen.params);
  const b = bruteSearchOptimalQ(scen.params);
  const m = mdpOptimalQ(scen.params);
  console.log(`    analytical q*=${a.qStar} (CR=${a.criticalRatio.toFixed(4)}),  ` +
              `brute q*=${b.qStar},  MDP q*=${m.qStar}`);
  console.log(`    E[profit(q*)] analytical=${expectedProfit(a.qStar, scen.params).toFixed(4)}  ` +
              `brute=${b.profileEP[b.qStar].toFixed(4)}  MDP V=${m.V0.toFixed(4)}`);
  check(`analytical q* ≡ brute q*`, a.qStar === b.qStar);
  check(`analytical q* ≡ MDP q*`,    a.qStar === m.qStar);
  check(`E[profit] analytical ≡ brute`, approx(expectedProfit(a.qStar, scen.params), b.profileEP[b.qStar], 1e-9));
  check(`E[profit] analytical ≡ MDP V`,  approx(expectedProfit(a.qStar, scen.params), m.V0, 1e-9));
}

// =============================================================================
console.log('\nStudy 2  Multi-period MDP at γ=0 reduces to newsvendor');
console.log('==========================================================================');
console.log('  With γ=0 and "salvage at end of day" by setting unitCost = (effective);');
console.log('  the multi-period MDP at state x=0 should pick the newsvendor q*.');

{
  const np: NewsvendorParams = SCENARIOS[0].params;
  // Build an inventory MDP whose myopic per-step reward IS the newsvendor profit:
  //   newsvendor: r = p·min(q,D) + s·(q−D)+ − c·q
  //   multi-period: r = p·min(q,D) − c·q − holdCost·(q−D)+ − lostCost·(D−q)+
  //   ⇒ holdCost = −s    (negative holding cost = salvage revenue per leftover)
  //     lostCost = 0     (foregone revenue already in min(q, D) term)
  const ip: InventoryParams = {
    xMax: np.qMax,
    aMax: np.qMax,
    demand: np.demand,
    unitCost:  np.unitCost,
    fixedCost: 0,
    unitPrice: np.unitPrice,
    holdCost:  -np.unitSalvage,
    lostCost:  0,
    gamma:     0,
  };
  const spec = inventoryMDPSpec(ip);
  const result = valueIteration(spec, {gamma: 0, tol: 1e-12});
  const policyAtZero = result.policy[0];
  const newsvendorQStar = analyticalOptimalQ(np).qStar;
  console.log(`    multi-period MDP π(0) = ${policyAtZero}    newsvendor q* = ${newsvendorQStar}`);
  check('γ=0 multi-period MDP π(0) = newsvendor q*',
        policyAtZero === newsvendorQStar);
}

// =============================================================================
console.log('\nStudy 3  Optimal policy structure: base-stock vs (s, S)');
console.log('==========================================================================');

const INV_BASE: InventoryParams = {
  xMax: 50, aMax: 50,
  demand: demandPoissonPMF(20, 50),
  unitCost: 1.0, fixedCost: 0,
  unitPrice: 2.0, holdCost: 0.1, lostCost: 0.5,
  gamma: 0.95,
};

{
  const params = {...INV_BASE, fixedCost: 0};
  const spec = inventoryMDPSpec(params);
  const result = valueIteration(spec, {gamma: params.gamma, tol: 1e-9});
  const policy = Array.from(result.policy).map(v => Math.max(0, v));
  const struct = detectPolicyStructure(policy);
  console.log(`  fixedCost = 0:  structure=${struct.kind}  S*=${struct.S}  s*=${struct.reorderPoint}`);
  check('fixedCost=0 ⇒ base-stock policy', struct.kind === 'base-stock');
  check('base-stock S* > 0',                struct.S > 0);
  check('base-stock S* ≤ xMax',             struct.S <= params.xMax);
}

{
  const params = {...INV_BASE, fixedCost: 10};
  const spec = inventoryMDPSpec(params);
  const result = valueIteration(spec, {gamma: params.gamma, tol: 1e-9});
  const policy = Array.from(result.policy).map(v => Math.max(0, v));
  const struct = detectPolicyStructure(policy);
  console.log(`  fixedCost = 10: structure=${struct.kind}  S*=${struct.S}  s*=${struct.reorderPoint}`);
  check('fixedCost>0 ⇒ (s, S) policy',   struct.kind === 's-S');
  check('s* < S* − 1 (gap due to setup cost)',
        struct.reorderPoint < struct.S - 1,
        `s=${struct.reorderPoint} S=${struct.S}`);
}

{
  // Sweep K: as fixed cost grows, the gap S − s should grow.
  const Ks = [0, 1, 5, 10, 25, 50];
  const gaps: number[] = [];
  console.log('\n  Sweep over fixedCost K:');
  console.log('    K       S*    s*    S − s    structure');
  for (const K of Ks) {
    const p = {...INV_BASE, fixedCost: K};
    const spec = inventoryMDPSpec(p);
    const r = valueIteration(spec, {gamma: p.gamma, tol: 1e-9});
    const policy = Array.from(r.policy).map(v => Math.max(0, v));
    const struct = detectPolicyStructure(policy);
    const gap = struct.S - struct.reorderPoint;
    gaps.push(gap);
    console.log(`    ${K.toString().padStart(2)}      ${struct.S.toString().padStart(3)}   ${struct.reorderPoint.toString().padStart(3)}     ${gap.toString().padStart(3)}      ${struct.kind}`);
  }
  // Gap should be weakly monotonic non-decreasing in K (apart from boundary effects).
  let monotonic = true;
  for (let i = 1; i < gaps.length; i++) {
    if (gaps[i] < gaps[i - 1] - 1) { monotonic = false; break; }
  }
  check('S − s gap is (weakly) increasing in K', monotonic);
}

// =============================================================================
console.log('\nStudy 4  Simulation matches Bellman value');
console.log('==========================================================================');

{
  const params = {...INV_BASE, fixedCost: 0, gamma: 0.95};
  const spec = inventoryMDPSpec(params);
  const result = valueIteration(spec, {gamma: params.gamma, tol: 1e-9});
  const policy = Array.from(result.policy).map(v => Math.max(0, v));

  // Long-run average reward under discounted policy ≈ V(0) · (1 − γ).
  const days = 50000;
  const sim = simulateInventoryMDP(params, x => policy[x], days, 42, 0);
  const expectedAvg = result.V[0] * (1 - params.gamma);
  console.log(`    V(0) = ${result.V[0].toFixed(3)},  V(0)·(1−γ) = ${expectedAvg.toFixed(3)}`);
  console.log(`    simulated mean reward over ${days} days = ${sim.meanReward.toFixed(3)}`);
  // Tolerance scales with stochastic noise (~1/sqrt(days)) and the gamma residual.
  // 5% relative is comfortable here.
  const tol = 0.05 * Math.abs(expectedAvg);
  check('simulation mean ≈ V(0)·(1−γ) within 5%',
        Math.abs(sim.meanReward - expectedAvg) < tol,
        `sim=${sim.meanReward.toFixed(3)} expected=${expectedAvg.toFixed(3)}`);
}

// =============================================================================
console.log('\nStudy 5  Cross-validation against Python (scipy / numpy) reference');
console.log('==========================================================================');

{
  // Newsvendor: same parameters as scenario 1.
  const py = runPython(['--lambda', '50', '--c', '0.5', '--p', '1.0', '--s', '0.1']);
  if (py === null) {
    console.log('  SKIP    Python reference not runnable (set NEWSVENDOR_PY=/path/to/python or install numpy)');
  } else {
    const tsResult = analyticalOptimalQ(SCENARIOS[0].params);
    const tsEP = expectedProfit(tsResult.qStar, SCENARIOS[0].params);
    console.log(`  newsvendor: TS q*=${tsResult.qStar} EP=${tsEP.toFixed(4)};  ` +
                `Py q*=${py.newsvendor.q_star} EP=${py.newsvendor.expected_profit_at_qstar.toFixed(4)}`);
    check('newsvendor q* matches Python',
          tsResult.qStar === py.newsvendor.q_star);
    check('newsvendor E[profit] matches Python within 1e-6',
          approx(tsEP, py.newsvendor.expected_profit_at_qstar, 1e-6),
          `|diff|=${Math.abs(tsEP - py.newsvendor.expected_profit_at_qstar).toExponential(2)}`);
  }
}

{
  // Multi-period: same parameters as the (s, S) sweep.
  const params: InventoryParams = {
    xMax: 50, aMax: 50,
    demand: demandPoissonPMF(20, 51),  // match Python's d_max = int(20*2.5+1)=51
    unitCost: 1.0, fixedCost: 10,
    unitPrice: 2.0, holdCost: 0.1, lostCost: 0.5,
    gamma: 0.95,
  };
  const spec = inventoryMDPSpec(params);
  const tsResult = valueIteration(spec, {gamma: params.gamma, tol: 1e-9});
  const tsPolicy = Array.from(tsResult.policy).map(v => Math.max(0, v));
  const py = runPython(['--multi', '--lambda', '20', '--c', '1.0', '--K', '10',
                        '--p', '2.0', '--h', '0.1', '--L', '0.5',
                        '--gamma', '0.95', '--x-max', '50', '--a-max', '50']);
  if (py === null) {
    console.log('  SKIP    Python reference not runnable for multi-period');
  } else {
    const tsV0 = tsResult.V[0];
    const pyV0 = py.inventory_mdp.V_at_zero;
    console.log(`  multi-period: TS V(0)=${tsV0.toFixed(4)},  Py V(0)=${pyV0.toFixed(4)}`);
    console.log(`  TS policy[0..19] = [${tsPolicy.slice(0, 20).join(', ')}]`);
    console.log(`  Py policy[0..19] = [${py.inventory_mdp.policy_first_20.join(', ')}]`);
    check('multi-period V(0) matches Python within 1e-3',
          approx(tsV0, pyV0, 1e-3),
          `|diff|=${Math.abs(tsV0 - pyV0).toExponential(2)}`);
    let policyMatch = true;
    for (let x = 0; x < 20; x++) {
      if (tsPolicy[x] !== py.inventory_mdp.policy_first_20[x]) { policyMatch = false; break; }
    }
    check('multi-period policy[0..19] matches Python', policyMatch);
  }
}

// =============================================================================
console.log('\nsummary: ' + pass + ' pass, ' + fail + ' fail');
process.exit(fail === 0 ? 0 : 1);
