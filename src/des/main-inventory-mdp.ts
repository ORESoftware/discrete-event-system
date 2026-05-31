#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_inventory_mdp.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-inventory-mdp.rs   (fn main)
// 1:1 file move. Multi-period inventory MDP: optimal ordering policy via
// value iteration (leftover stock carries over).
//
// Conversion notes (file-specific):
//   - demand distribution + value iteration -> pure fns; any sampling ->
//     SeededRandom.
//   - state/value arrays over inventory levels -> Vec<f64>.
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// MULTI-PERIOD INVENTORY MDP — discovering optimal policies via value iteration.
//
// Generalisation of the newsvendor: leftover stock CARRIES OVER to the
// next day. This converts a 1-step problem into an infinite-horizon MDP
// where memory matters and the optimal policy depends on current
// inventory.
//
// MDP formulation
// ---------------
//   State    s = current inventory level x ∈ {0, …, X_max}
//   Action   a = order quantity (received before demand realises)
//                a ∈ {0, …, A_max}, with x + a ≤ X_max (storage cap)
//   Demand   D ~ DemandDist  (independent across days)
//   Sales    sold = min(x + a, D)
//   Reward   r(x, a, D) = unitPrice · sold
//                       − unitCost  · a            (procurement)
//                       − fixedCost · 1{a > 0}     (fixed setup)
//                       − holdCost  · (x + a − D)+ (carried at end of day)
//                       − lostCost  · (D − x − a)+ (under "lost-sales" model)
//   Next     x' = (x + a − D)+   (lost-sales)
//                            or x + a − D (uncapped) if backorders allowed
//
// Discount γ ∈ (0, 1) for infinite-horizon discounted-sum reward.
//
// KNOWN OPTIMAL POLICY STRUCTURES (used as analytic anchors)
//
//   1. No fixed cost (fixedCost = 0): the optimal policy is **base-stock**:
//      "order up to S* whenever x < S*". S* depends on demand distribution
//      and the cost ratios via a multi-period generalisation of the
//      newsvendor critical fractile. For γ = 0 this reduces to the
//      single-period q* (with x = 0 acting as start-of-day).
//
//   2. Positive fixed cost K > 0: optimal policy is **(s, S)**:
//      "if x ≤ s, order up to S; else don't order". This is the
//      celebrated Scarf result. Both s and S require value iteration to
//      compute in general.
//
// The value iteration in `general/value-iteration.ts` discovers the
// optimal policy from scratch, with no structural assumption — and we
// then check that the discovered policy matches the expected structure.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {mulberry32, withSeed} from './general/prng';
import {MDPSpec, Outcome, valueIteration} from './general/value-iteration';
import {DemandDist, demandPoissonPMF, sampleDemand} from './main-newsvendor';

export interface InventoryParams {
  /** Maximum on-hand inventory (cap on storage). */
  xMax: number;
  /** Maximum order quantity per period. */
  aMax: number;
  /** Demand distribution. */
  demand: DemandDist;
  /** Variable cost per unit ordered. */
  unitCost: number;
  /** Fixed (setup) cost incurred on any positive order. */
  fixedCost: number;
  /** Revenue per unit sold. */
  unitPrice: number;
  /** Holding cost per unit carried at end of period. */
  holdCost: number;
  /** Penalty per unit of unmet demand (lost-sales model). */
  lostCost: number;
  /** Discount factor in (0, 1). */
  gamma: number;
}

// -----------------------------------------------------------------------------
// MDP spec (lost-sales). Backorder variant left as a pure cost-model swap.
// -----------------------------------------------------------------------------

export function inventoryMDPSpec(p: InventoryParams): MDPSpec {
  const numStates = p.xMax + 1;
  const numActions = (s: number) => Math.max(0, Math.min(p.aMax, p.xMax - s)) + 1;
  const isTerminal = (_: number) => false;

  // Pre-cache rewards and successor distributions per (x, a) for speed.
  const cache: Outcome[][][] = new Array(numStates);
  for (let x = 0; x < numStates; x++) {
    const A = numActions(x);
    const perAction: Outcome[][] = new Array(A);
    for (let a = 0; a < A; a++) {
      const after = x + a;                  // inventory available before demand
      const ol: Outcome[] = [];
      const orderCost = p.unitCost * a + (a > 0 ? p.fixedCost : 0);
      for (let d = 0; d < p.demand.pmf.length; d++) {
        const pr = p.demand.pmf[d];
        if (pr === 0) continue;
        const sold = Math.min(after, d);
        const leftover = Math.max(0, after - d);
        const lost = Math.max(0, d - after);
        const reward = p.unitPrice * sold
                     - orderCost
                     - p.holdCost * leftover
                     - p.lostCost * lost;
        const nextX = leftover;             // lost-sales: nextX = (x+a-D)+
        // Coalesce outcomes by next-state: same nextX may appear from
        // multiple (d) values when demand is large enough that all
        // collapse to nextX = 0.
        const idx = ol.findIndex(o => o.nextState === nextX && Math.abs(o.reward - reward) < 1e-12);
        if (idx >= 0) ol[idx].prob += pr;
        else          ol.push({prob: pr, reward, nextState: nextX});
      }
      perAction[a] = ol;
    }
    cache[x] = perAction;
  }

  return {
    numStates, numActions, isTerminal,
    outcomes: (s, a) => cache[s][a] ?? [],
    stateLabel: x => `x=${x}`,
    actionLabel: a => `order=${a}`,
  };
}

// -----------------------------------------------------------------------------
// Simulate the MDP under a given policy.
// -----------------------------------------------------------------------------

export function simulateInventoryMDP(
  p: InventoryParams,
  policy: (x: number) => number,
  days: number,
  seed: number,
  initialInventory = 0,
): {meanReward: number; meanInventory: number; meanLost: number; meanLeftover: number;
    history: Array<{day: number; x: number; a: number; d: number; sold: number; reward: number; nextX: number}>} {
  return withSeed(seed, () => {
    const rng = mulberry32(seed);
    let x = initialInventory;
    let totalReward = 0, totalInv = 0, totalLost = 0, totalLeftover = 0;
    const history: Array<{day: number; x: number; a: number; d: number; sold: number; reward: number; nextX: number}> = [];
    for (let day = 0; day < days; day++) {
      const a = Math.max(0, Math.min(p.aMax, p.xMax - x, policy(x)));
      const d = sampleDemand(p.demand, rng);
      const after = x + a;
      const sold = Math.min(after, d);
      const leftover = Math.max(0, after - d);
      const lost = Math.max(0, d - after);
      const orderCost = p.unitCost * a + (a > 0 ? p.fixedCost : 0);
      const reward = p.unitPrice * sold - orderCost - p.holdCost * leftover - p.lostCost * lost;
      totalReward += reward;
      totalInv += x;
      totalLost += lost;
      totalLeftover += leftover;
      history.push({day, x, a, d, sold, reward, nextX: leftover});
      x = leftover;
    }
    return {
      meanReward: totalReward / days,
      meanInventory: totalInv / days,
      meanLost: totalLost / days,
      meanLeftover: totalLeftover / days,
      history,
    };
  });
}

// -----------------------------------------------------------------------------
// Policy structure detection. Given a discovered π(x), check whether it
// is a base-stock or (s, S) policy and extract the parameters.
// -----------------------------------------------------------------------------

export interface PolicyStructure {
  kind: 'base-stock' | 's-S' | 'irregular';
  /** Order-up-to level S (defined for both base-stock and (s,S)). */
  S: number;
  /** Reorder point s (defined only for (s,S); equals S − 1 for base-stock). */
  reorderPoint: number;
  /** Detailed per-state action map (x → a). */
  perState: number[];
}

export function detectPolicyStructure(policy: ArrayLike<number>): PolicyStructure {
  const xMax = policy.length - 1;
  const perState = Array.from({length: policy.length}, (_, x) => policy[x]);
  // For each x, the "order-up-to" target T(x) = x + a(x).
  const T = perState.map((a, x) => x + a);
  // Base-stock: T(x) = S for all x ≤ S, and a(x) = 0 for x ≥ S.
  // (s, S):     T(x) = S for x ≤ s, and a(x) = 0 for x > s.
  // The non-zero ordering region is contiguous from 0 up to some s*; in
  // that region T(x) = some constant S. Detect.
  let s = -1;
  let Stargets = new Set<number>();
  for (let x = 0; x <= xMax; x++) {
    if (perState[x] > 0) {
      s = x;
      Stargets.add(T[x]);
    }
  }
  if (s === -1) return {kind: 'irregular', S: 0, reorderPoint: -1, perState};
  if (Stargets.size > 1) return {kind: 'irregular', S: Math.max(...Stargets), reorderPoint: s, perState};
  const S = [...Stargets][0];
  // Verify the "no order above s" half.
  for (let x = s + 1; x <= xMax; x++) {
    if (perState[x] > 0) return {kind: 'irregular', S, reorderPoint: s, perState};
  }
  // Base-stock if s = S − 1 (you order whenever below S, including at S − 1).
  if (s === S - 1) return {kind: 'base-stock', S, reorderPoint: S - 1, perState};
  return {kind: 's-S', S, reorderPoint: s, perState};
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

async function main() {
  const lambda = Number(process.env.LAMBDA ?? 20);
  const dMax = Number(process.env.D_MAX ?? Math.ceil(lambda * 2.5));
  const xMax = Number(process.env.X_MAX ?? Math.ceil(lambda * 2.5));
  const aMax = Number(process.env.A_MAX ?? Math.ceil(lambda * 2.5));
  const params: InventoryParams = {
    xMax, aMax,
    demand: demandPoissonPMF(lambda, dMax),
    unitCost:    Number(process.env.UNIT_COST    ?? 1.0),
    fixedCost:   Number(process.env.FIXED_COST   ?? 0),
    unitPrice:   Number(process.env.UNIT_PRICE   ?? 2.0),
    holdCost:    Number(process.env.HOLD_COST    ?? 0.1),
    lostCost:    Number(process.env.LOST_COST    ?? 0.5),
    gamma:       Number(process.env.GAMMA        ?? 0.95),
  };
  const days = Number(process.env.DAYS ?? 5000);
  const seed = Number(process.env.SEED ?? 1);

  console.log(`# Multi-period inventory MDP`);
  console.log(`#   demand = Poisson(λ=${lambda})  truncated at ${dMax}`);
  console.log(`#   xMax=${params.xMax}, aMax=${params.aMax}, γ=${params.gamma}`);
  console.log(`#   unitCost=${params.unitCost}, fixedCost=${params.fixedCost}, unitPrice=${params.unitPrice}`);
  console.log(`#   holdCost=${params.holdCost}, lostCost=${params.lostCost}`);

  const t0 = Date.now();
  const spec = inventoryMDPSpec(params);
  const result = valueIteration(spec, {gamma: params.gamma, tol: 1e-8, maxIter: 5000});
  const ms = Date.now() - t0;

  console.log(`\n# Value iteration converged in ${result.iterations} sweeps  ` +
              `(final ΔV = ${result.finalDelta.toExponential(2)}, ${ms} ms)`);

  const policy = Array.from(result.policy).map(v => Math.max(0, v));
  const struct = detectPolicyStructure(policy);
  console.log(`\n# Discovered policy structure: ${struct.kind}`);
  console.log(`#   S (order-up-to) = ${struct.S}`);
  console.log(`#   reorder point s = ${struct.reorderPoint}`);
  if (struct.kind === 's-S') console.log(`#   (s, S) = (${struct.reorderPoint}, ${struct.S})`);

  const previewN = Math.min(20, params.xMax + 1);
  console.log(`\n# π(x) and V(x) for x ∈ [0, ${previewN - 1}]`);
  console.log(`     x   action a   x+a (target)    V(x)`);
  for (let x = 0; x < previewN; x++) {
    console.log(`  ${x.toString().padStart(4)}    ${policy[x].toString().padStart(4)}        ${(x + policy[x]).toString().padStart(4)}        ${result.V[x].toFixed(3).padStart(10)}`);
  }

  const sim = simulateInventoryMDP(params, x => policy[x], days, seed, 0);
  console.log(`\n# ${days}-day simulation under discovered policy (initial x=0)`);
  console.log(`    mean reward/day      = ${sim.meanReward.toFixed(3)}`);
  console.log(`    mean inventory       = ${sim.meanInventory.toFixed(2)}`);
  console.log(`    mean lost demand     = ${sim.meanLost.toFixed(2)}`);
  console.log(`    mean leftover        = ${sim.meanLeftover.toFixed(2)}`);
  console.log(`    long-run avg reward  ≈ V(0) · (1−γ) = ${(result.V[0] * (1 - params.gamma)).toFixed(3)}`);

  // Optional animation.
  if (process.env.ANIMATE === '1') {
    const {FrameRecorder} = await import('./animation/frame-recorder');
    const {STAGE_W, STAGE_H, buildNewsvendorFrame, buildNewsvendorChart} =
      await import('./animation/scenes/newsvendor-scene');
    const outDir2 = path.join(__dirname, '..', '..', 'out');
    const animPath = path.join(outDir2, `inventory-mdp.html`);
    const framesPath = path.join(outDir2, `inventory-mdp.frames.jsonl`);
    const animDays = Number(process.env.ANIM_DAYS ?? Math.min(days, 200));
    const policyLabel = struct.kind === 's-S'
      ? `(s, S) = (${struct.reorderPoint}, ${struct.S})`
      : struct.kind === 'base-stock'
        ? `base-stock S* = ${struct.S}`
        : `irregular`;
    const rec = new FrameRecorder({
      framesPath, htmlPath: animPath,
      width: STAGE_W, height: STAGE_H, fps: 4,
      title: `Inventory MDP — discovered policy`,
      subtitle: `λ=${lambda}  c=${params.unitCost}  K=${params.fixedCost}  ` +
                `p=${params.unitPrice}  h=${params.holdCost}  L=${params.lostCost}  γ=${params.gamma}  ${policyLabel}`,
      liveTickLine: false,
    });
    const trace = {t: [] as number[], inv: [] as number[], profit: [] as number[], cumProfit: [] as number[]};
    let cum = 0;
    const animSim = simulateInventoryMDP(params, x => policy[x], animDays, seed, 0);
    const qScale = Math.max(...animSim.history.map(h => h.x + h.a));
    for (let i = 0; i < animSim.history.length; i++) {
      const h = animSim.history[i];
      cum += h.reward;
      trace.t.push(h.day);
      trace.inv.push(h.x);
      trace.profit.push(h.reward);
      trace.cumProfit.push(cum);
      const lost = Math.max(0, h.d - (h.x + h.a));
      rec.frame(h.day, i, () => buildNewsvendorFrame({
        day: h.day, startInv: h.x, ordered: h.a, demand: h.d,
        sold: h.sold, leftover: h.nextX, lost, profit: h.reward, cumProfit: cum,
        qScale, policy: policyLabel,
      }));
    }
    rec.setCharts(buildNewsvendorChart(trace));
    await rec.finish();
    console.log(`# wrote ${animPath} (${rec.getFrameCount()} frames)`);
  }

  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'inventory-mdp.json');
  fs.writeFileSync(outPath, JSON.stringify({
    params: {...params, demand: undefined, demandLambda: lambda, dMax},
    days, seed,
    iterations: result.iterations, finalDelta: result.finalDelta,
    V: Array.from(result.V), policy,
    structure: struct,
    simulation: {meanReward: sim.meanReward, meanInventory: sim.meanInventory,
                 meanLost: sim.meanLost, meanLeftover: sim.meanLeftover},
  }));
  console.log(`# wrote ${outPath}`);
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
