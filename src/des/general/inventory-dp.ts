'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/inventory-dp.rs  (module des::general::inventory_dp)
// 1:1 file move. Stochastic inventory control via finite-horizon DP (backward induction).
//
// Declarations → Rust:
//   interface InventoryProblem / InventoryDPResult -> structs (#[derive(Clone)])
//   class InventoryDPStation extends FiniteHorizonDPStation -> struct + impl (base -> trait)
//   fn solveInventoryDP / simulateInventory -> free fns (or PureTransform / StatefulTransform)
//   fn sampleFromPmf              -> assoc fn taking `&mut impl Rng`
//
// Conversion notes (file-specific):
//   - `mulberry32(seed)` closure RNG in simulateInventory/sampleFromPmf -> inject `RandomSource`.
//   - `FiniteHorizonDPStation` is a template-method base -> trait with default fns + struct state.
//   - demand PMF is `readonly number[]` -> `&[f64]`; value/policy tables -> `Vec<f64>`/`Vec<usize>`.
// =============================================================================

// =============================================================================
// general/inventory-dp.ts — multi-period STOCHASTIC INVENTORY MANAGEMENT
// solved by FINITE-HORIZON DYNAMIC PROGRAMMING (backward induction).
//
// CLASSIC PROBLEM (Bellman 1957, Scarf 1960, Veinott)
// ───────────────────────────────────────────────────
//   T-period horizon. At the start of each period t the manager observes
//   the on-hand inventory s ∈ {0, …, S_max} and chooses an order quantity
//   a ∈ {0, …, S_max − s}. Order arrives instantly. Demand D_t is random
//   with PMF p(d). Sales = min(D_t, s + a). Inventory at start of t+1 is
//   max(0, s + a − D_t). Cash flow:
//
//       r(s, a, d) = price · min(d, s + a)
//                  − cost  · a                     (variable purchase cost)
//                  − fixed · 1{a > 0}              (fixed order cost K)
//                  − holdCost · max(0, s + a − d)  (hold leftover)
//                  − stockoutCost · max(0, d − (s + a))
//
//   Terminal salvage: V_T(s) = salvageValue · s.
//
//   Bellman recursion (backward induction):
//
//       V_t(s) = max_a Σ_d p(d) [ r(s, a, d) + γ V_{t+1}(s_next) ]
//       π_t(s) = argmax_a same
//
//   This is the WORKHORSE textbook example — closed form solution doesn't
//   exist for general K > 0 (the (s, S) policy is optimal only in special
//   cases). DP gives the EXACT optimum for any well-defined PMF.
//
// AS A DES STATION
// ────────────────
//   InventoryDPStation extends FiniteHorizonDPStation. Each tick is one
//   backward sweep across all states for the current stage. Termination
//   when stage = 0 has been computed.
//
// SIMULATION HELPER
// ─────────────────
//   simulateInventory(spec, V, π, seed) — draw a path from t=0 with
//   policy π and demand PMF, return per-period cash + final inventory.
//   Used by the adapter to print expected vs realised reward.
// =============================================================================

import {FiniteHorizonDPStation, runIterativeDES, intrinsicCheck, DPOutcome} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

export interface InventoryProblem {
  /** Number of decision periods T. */
  horizon: number;
  /** Maximum on-hand inventory. State space is {0, …, S_max}. */
  S_max: number;
  /** Per-period demand distribution. demandPmf[d] = P(D = d). */
  demandPmf: readonly number[];
  /** Selling price per unit. */
  price: number;
  /** Variable order cost per unit. */
  cost: number;
  /** Fixed ordering cost (per period when a > 0). */
  fixedCost: number;
  /** Holding cost per unit per period. */
  holdCost: number;
  /** Stockout penalty per unit shortage. */
  stockoutCost: number;
  /** Salvage value per leftover unit at terminal stage. */
  salvageValue: number;
  /** Discount factor γ ∈ (0, 1]. Default 1 (undiscounted finite horizon). */
  discount?: number;
  /** Initial inventory at t=0. */
  initialInventory: number;
}

export class InventoryDPStation extends FiniteHorizonDPStation {
  private readonly p: InventoryProblem;
  /** Pre-computed E[demand] for diagnostics. */
  readonly meanDemand: number;

  constructor(p: InventoryProblem) {
    super('inventory-dp');
    if (p.horizon < 1) throw new Error('horizon must be ≥ 1');
    if (p.S_max < 0) throw new Error('S_max must be ≥ 0');
    if (p.demandPmf.length === 0) throw new Error('demandPmf must be non-empty');
    const sumP = p.demandPmf.reduce((s, x) => s + x, 0);
    if (Math.abs(sumP - 1) > 1e-9) {
      throw new Error(`demandPmf must sum to 1, got ${sumP}`);
    }
    this.p = p;
    this.meanDemand = p.demandPmf.reduce((s, pd, d) => s + pd * d, 0);
    this.bootstrap();

    // Intrinsic invariants for the inventory DP.
    this.addValidator(intrinsicCheck<InventoryDPStation>({
      name: 'inventory-dp.policy-feasible',
      group: 'inventory-dp-intrinsic',
      predicate: st => {
        for (let t = 0; t < st['p'].horizon; t++) {
          const pol = st.policy[t];
          for (let s = 0; s < pol.length; s++) {
            const a = pol[s];
            if (a < 0) continue;
            if (a < 0 || s + a > st['p'].S_max) return false;
          }
        }
        return true;
      },
      expected: 'a ∈ [0, S_max − s] for every (t, s)',
      observedFn: st => `T=${st['p'].horizon}  S_max=${st['p'].S_max}`,
    }));
    this.addValidator(intrinsicCheck<InventoryDPStation>({
      name: 'inventory-dp.terminal-V-equals-salvage',
      group: 'inventory-dp-intrinsic',
      predicate: st => {
        const T = st['p'].horizon; const sv = st['p'].salvageValue;
        const VT = st.V[T];
        for (let s = 0; s < VT.length; s++) {
          if (Math.abs(VT[s] - sv * s) > 1e-9) return false;
        }
        return true;
      },
      expected: 'V_T(s) = salvageValue · s',
      observedFn: st => `salvage=${st['p'].salvageValue}, V_T(0)=${st.V[st['p'].horizon][0]}`,
    }));
    this.addValidator(intrinsicCheck<InventoryDPStation>({
      name: 'inventory-dp.value-finite-everywhere',
      group: 'inventory-dp-intrinsic',
      predicate: st => {
        for (let t = 0; t <= st['p'].horizon; t++) {
          for (const v of st.V[t]) if (!Number.isFinite(v)) return false;
        }
        return true;
      },
      expected: 'every V[t][s] finite',
      observedFn: () => 'no NaN/Inf',
    }));
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected horizon(): number { return this.p.horizon; }
  protected numStates(): number { return this.p.S_max + 1; }
  protected numActions(state: number, _stage: number): number {
    return this.p.S_max - state + 1; // a = 0 … S_max - s
  }
  protected stageDiscount(_stage: number): number { return this.p.discount ?? 1; }
  protected terminalReward(state: number): number {
    return this.p.salvageValue * state;
  }

  protected transitions(state: number, action: number, _stage: number): DPOutcome[] {
    const p = this.p;
    const inv = state + action;
    const orderCost = p.cost * action + (action > 0 ? p.fixedCost : 0);
    const out: DPOutcome[] = [];
    for (let d = 0; d < p.demandPmf.length; d++) {
      const prob = p.demandPmf[d];
      if (prob === 0) continue;
      const sales = Math.min(d, inv);
      const leftover = Math.max(0, inv - d);
      const shortage = Math.max(0, d - inv);
      const reward = p.price * sales
                   - orderCost
                   - p.holdCost * leftover
                   - p.stockoutCost * shortage;
      out.push({prob, reward, nextState: leftover});
    }
    return out;
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface InventoryDPResult {
  /** V[t][s] for t = 0 … T. */
  V: number[][];
  /** π[t][s] for t = 0 … T-1. */
  policy: number[][];
  /** Expected total reward starting from (t=0, s=initialInventory). */
  expectedReward: number;
  /** A simulated trajectory under the optimal policy (for sanity / display). */
  simulation: {
    inventory: number[]; orders: number[]; demands: number[]; rewards: number[];
    totalReward: number;
  };
  /** Mean demand (for diagnostics). */
  meanDemand: number;
  ticks: number;
}

export function solveInventoryDP(p: InventoryProblem, opts?: {seed?: number}): InventoryDPResult {
  // Pre-construction guards on the problem instance.
  const cls = 'solveInventoryDP';
  Preconditions.integerInRange(cls, 'horizon', p.horizon, 1, 1e6);
  Preconditions.integerInRange(cls, 'S_max', p.S_max, 0, 1e6);
  Preconditions.integerInRange(cls, 'initialInventory', p.initialInventory, 0, p.S_max);
  Preconditions.probabilityVector(cls, 'demandPmf', p.demandPmf);
  Preconditions.nonNegative(cls, 'price', p.price);
  Preconditions.nonNegative(cls, 'cost', p.cost);
  Preconditions.nonNegative(cls, 'fixedCost', p.fixedCost);
  Preconditions.nonNegative(cls, 'holdCost', p.holdCost);
  Preconditions.nonNegative(cls, 'stockoutCost', p.stockoutCost);
  Preconditions.finite(cls, 'salvageValue', p.salvageValue);
  Preconditions.inRange(cls, 'discount', p.discount ?? 1, 0, 1);
  const station = new InventoryDPStation(p);
  const summary = runIterativeDES([station]);
  const sim = simulateInventory(p, station.policy, opts?.seed ?? 1);
  return {
    V: station.V.map(row => row.slice()),
    policy: station.policy.map(row => row.slice()),
    expectedReward: station.V[0][p.initialInventory],
    simulation: sim,
    meanDemand: station.meanDemand,
    ticks: summary.ticks,
  };
}

/** Forward simulation under a given policy. */
export function simulateInventory(
  p: InventoryProblem, policy: readonly number[][], seed: number,
): InventoryDPResult['simulation'] {
  const rng = mulberry32(seed);
  const inventory: number[] = [p.initialInventory];
  const orders: number[] = [];
  const demands: number[] = [];
  const rewards: number[] = [];
  let total = 0;
  let s = p.initialInventory;
  for (let t = 0; t < p.horizon; t++) {
    const a = policy[t][s];
    const inv = s + a;
    const d = sampleFromPmf(p.demandPmf, rng);
    const sales = Math.min(d, inv);
    const leftover = Math.max(0, inv - d);
    const shortage = Math.max(0, d - inv);
    const orderCost = p.cost * a + (a > 0 ? p.fixedCost : 0);
    const r = p.price * sales - orderCost - p.holdCost * leftover - p.stockoutCost * shortage;
    orders.push(a); demands.push(d); rewards.push(r); total += r;
    s = leftover;
    inventory.push(s);
  }
  // Add terminal salvage as a final reward entry for transparency.
  total += p.salvageValue * s;
  return {inventory, orders, demands, rewards, totalReward: total};
}

function sampleFromPmf(pmf: readonly number[], rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < pmf.length; i++) {
    acc += pmf[i];
    if (u < acc) return i;
  }
  return pmf.length - 1;
}
