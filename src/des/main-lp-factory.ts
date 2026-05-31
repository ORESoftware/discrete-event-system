'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-lp-factory.rs   (fn main)
// 1:1 file move. Simulation-optimisation: solve a factory LP, then stress the
// nominal plan with a DES (log-normal times, breakdowns, finite buffers).
//
// Conversion notes (file-specific):
//   - log-normal processing-time + breakdown sampling -> inject RandomSource/
//     SeededRandom.
//   - LP solve + DES wiring -> use crate::des::general::...
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// main-lp-factory.ts — DES + LP simulation-optimisation pattern.
//
// SCENARIO
// ────────
// A factory makes 3 products (A, B, C). Each product needs a fixed amount
// of time on each of 4 machines (M1..M4). Each machine has a weekly time
// budget (in minutes). Profit per unit is known. Find the production
// plan that maximises weekly profit:
//
//     max  Σ_p profit_p · x_p
//     s.t. Σ_p tau_{m,p} · x_p ≤ capacity_m     ∀ m
//          x_p ≥ 0,   x_p continuous
//
// This is a textbook 3-variable, 4-constraint LP. Simplex / HiGHS solves
// it in 2–4 pivots. The optimum is the WEEKLY NOMINAL PLAN.
//
// PROBLEM
// ───────
// The nominal LP assumes:
//   - deterministic processing times,
//   - no machine breakdowns,
//   - infinite buffers between machines,
//   - no order-of-operations bottlenecks.
//
// Real factories violate every one of these. So we run a DES simulation
// where each unit of each product flows through M1 → M2 → M3 → M4 with
//   - log-normal processing times around the LP's tau_{m,p} mean,
//   - per-machine breakdown probability p_break per minute (Poisson),
//   - finite per-machine buffer capacity (queue overflow → block upstream),
// and measure the REALISED weekly throughput.
//
// The pattern: LP gives the nominal upper bound; DES gives operational
// reality. The gap is usually 10–30%. Robust LP planning (shrink the
// capacity RHS by a safety factor) closes the gap by trading nominal
// optimality for realised reliability — this is the simulation-
// optimisation feedback loop.
//
// USAGE
// ─────
//   # baseline — LP plan, run via DES under realistic dynamics, 30 reps
//   node dist/des/main-lp-factory.js
//
//   # use external simplex
//   LP_SOLVER=scipy:highs node dist/des/main-lp-factory.js
//   LP_SOLVER=scipy:highs-ipm node dist/des/main-lp-factory.js   # interior-point
//   LP_SOLVER=internal node dist/des/main-lp-factory.js          # in-process simplex
//
//   # robust planning: shrink RHS by 10% to leave headroom for variability
//   ROBUST=0.9 node dist/des/main-lp-factory.js
//
//   # vary disturbance strength
//   PROC_CV=0.30 BREAK_PROB=0.005 node dist/des/main-lp-factory.js
// =============================================================================

import {LPProblem, LPSolution} from './general/lp';
import {solveLPThenSimulate} from './general/des-lp-bridge';
import {mulberry32} from './general/prng';

// -----------------------------------------------------------------------------
// Problem data — products × machines.
// -----------------------------------------------------------------------------
interface FactoryProblem {
  products: string[];                    // length P
  machines: string[];                    // length M
  /** Mean processing time machine m × product p, in minutes. tau[m][p]. */
  tau: number[][];
  /** Weekly capacity per machine (minutes). */
  capacity: number[];
  /** Profit per finished unit. */
  profit: number[];
}

const FACTORY: FactoryProblem = {
  products: ['Widget-A', 'Widget-B', 'Widget-C'],
  machines: ['Lathe', 'Mill', 'Drill', 'Press'],
  tau: [
    /* Lathe */ [3.0, 5.0, 2.5],
    /* Mill  */ [2.5, 1.5, 4.0],
    /* Drill */ [1.0, 2.0, 1.5],
    /* Press */ [4.0, 3.0, 2.0],
  ],
  profit: [40, 30, 50],     // dollars per unit
  // 5 days × 8 hours/day × 60 min = 2400 min per machine per week.
  capacity: [2400, 2400, 2400, 2400],
};

// -----------------------------------------------------------------------------
// Build the LP.
// -----------------------------------------------------------------------------
function buildFactoryLP(prob: FactoryProblem, robustFactor = 1.0): LPProblem {
  const P = prob.products.length;
  const M = prob.machines.length;
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  for (let m = 0; m < M; m++) {
    A_ub.push(prob.tau[m].slice());
    b_ub.push(prob.capacity[m] * robustFactor);
  }
  return {
    sense: 'max',
    c: prob.profit.slice(),
    A_ub, b_ub,
    varNames: prob.products,
    conNames: prob.machines.map(m => m + ' capacity'),
  };
}

// -----------------------------------------------------------------------------
// DES — flow units through 4 machines in series, with stochastic processing
//       times and per-machine breakdowns.
//
// Structure: each machine is a "Station" holding a single unit-in-progress.
// Each tick is 1 minute. A unit's flow:  Lathe → Mill → Drill → Press.
//
// We do not need the full DES base class hierarchy here; this file
// demonstrates the bridge pattern, so the simulator is intentionally
// minimal and self-contained. The "stationary entity / movable entity"
// architecture is fully exercised in main-elevator.ts, main-two-disease.ts,
// and main-factmachine.ts.
// -----------------------------------------------------------------------------
interface SimResult {
  realisedThroughput: number[];          // units of each product produced
  realisedRevenue: number;               // dollars
  utilisation: number[];                 // per-machine
  breakdowns: number[];                  // per-machine outages
  wallClockMin: number;                  // minutes simulated
}

interface SimParams {
  procCV: number;                        // log-normal coefficient of variation
  breakProbPerMin: number;               // probability of a breakdown per machine per minute
  breakDurationMin: number;              // average outage duration
  totalMin: number;                      // simulated minutes (default 1 work-week = 2400)
  seed: number;
}

function simulateFactory(prob: FactoryProblem, plan: number[],
                          params: SimParams): SimResult {
  const M = prob.machines.length;
  const P = prob.products.length;
  const rng = mulberry32(params.seed);

  // Job queue: a sequence of product indices to push through the line, in
  // round-robin proportions matching the plan, capped at the planned total.
  const sched: number[] = [];
  const planRounded = plan.map(x => Math.floor(x));
  const remaining = planRounded.slice();
  while (remaining.some(r => r > 0)) {
    for (let p = 0; p < P; p++) {
      if (remaining[p] > 0) { sched.push(p); remaining[p]--; }
    }
  }

  // Per-machine state: the unit currently on it (product id), remaining
  // processing time, and a "down until" timer for breakdowns.
  type Slot = {productId: number; remaining: number} | null;
  const slot: Slot[] = new Array(M).fill(null);
  // FIFO buffers between machines (machine M takes from buffer[M], emits to buffer[M+1]).
  // buffer[0] = inbound queue (job list), buffer[M] = finished output.
  const buffer: number[][] = new Array(M + 1).fill(null).map(() => []);
  for (const p of sched) buffer[0].push(p);

  const downUntil: number[] = new Array(M).fill(-1);
  const breakdowns: number[] = new Array(M).fill(0);
  const utilisation: number[] = new Array(M).fill(0);

  const lognormalSample = (mean: number, cv: number, rng: () => number): number => {
    if (cv <= 0) return mean;
    // Log-normal with given mean and cv (=sd/mean): sigma² = ln(1+cv²); mu = ln(mean) − σ²/2.
    const sigma2 = Math.log(1 + cv * cv);
    const sigma = Math.sqrt(sigma2);
    const mu = Math.log(mean) - 0.5 * sigma2;
    // Box-Muller.
    const u1 = Math.max(1e-12, rng());
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.max(0.1, Math.exp(mu + sigma * z));
  };

  for (let t = 0; t < params.totalMin; t++) {
    // Step machines from last → first (so that one machine emitting to the
    // next does not re-process within the same tick).
    for (let m = M - 1; m >= 0; m--) {
      // Breakdown event?
      if (t >= downUntil[m] && slot[m] !== null && rng() < params.breakProbPerMin) {
        breakdowns[m]++;
        downUntil[m] = t + Math.max(1, Math.round(-Math.log(Math.max(1e-12, rng())) * params.breakDurationMin));
      }
      const isDown = t < downUntil[m];
      if (slot[m] !== null && !isDown) {
        slot[m]!.remaining--;
        utilisation[m]++;
        if (slot[m]!.remaining <= 0) {
          // Try to push to next buffer (or output if last machine).
          buffer[m + 1].push(slot[m]!.productId);
          slot[m] = null;
        }
      }
      // If slot is empty, pull from upstream buffer.
      if (slot[m] === null && buffer[m].length > 0) {
        const productId = buffer[m].shift()!;
        const meanT = prob.tau[m][productId];
        const procT = lognormalSample(meanT, params.procCV, rng);
        slot[m] = {productId, remaining: Math.ceil(procT)};
      }
    }
  }

  const realisedThroughput = new Array(P).fill(0);
  for (const p of buffer[M]) realisedThroughput[p]++;
  let realisedRevenue = 0;
  for (let p = 0; p < P; p++) realisedRevenue += prob.profit[p] * realisedThroughput[p];

  return {
    realisedThroughput,
    realisedRevenue,
    utilisation: utilisation.map(u => u / params.totalMin),
    breakdowns,
    wallClockMin: params.totalMin,
  };
}

// -----------------------------------------------------------------------------
// Welch t statistic, just for nominal-vs-realised comparison.
// -----------------------------------------------------------------------------
function welchT(a: number[], b: number[]): {t: number; mean_a: number; mean_b: number; sd_a: number; sd_b: number} {
  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const variance = (xs: number[], mu: number) => xs.reduce((s, v) => s + (v - mu) ** 2, 0) / Math.max(1, xs.length - 1);
  const ma = mean(a), mb = mean(b);
  const va = variance(a, ma), vb = variance(b, mb);
  const t = (ma - mb) / Math.sqrt(va / a.length + vb / b.length);
  return {t, mean_a: ma, mean_b: mb, sd_a: Math.sqrt(va), sd_b: Math.sqrt(vb)};
}

// -----------------------------------------------------------------------------
// Main entry.
// -----------------------------------------------------------------------------
async function main(): Promise<void> {
  const robustFactor = Number(process.env.ROBUST ?? 1.0);
  const procCV = Number(process.env.PROC_CV ?? 0.25);
  const breakProb = Number(process.env.BREAK_PROB ?? 0.002);
  const breakDuration = Number(process.env.BREAK_DUR ?? 30);
  const totalMin = Number(process.env.TOTAL_MIN ?? 2400);
  const N_REPS = Number(process.env.N_REPS ?? 30);

  const lp = buildFactoryLP(FACTORY, robustFactor);
  console.log('# Factory scheduling LP + DES bridge');
  console.log(`# solver=${process.env.LP_SOLVER ?? 'scipy:highs'}  robustFactor=${robustFactor}  procCV=${procCV}  breakProb=${breakProb}/min`);
  console.log('');

  const result = solveLPThenSimulate(lp,
    (plan: LPSolution) => {
      // For the realised side, run N replications and average.
      const reps: SimResult[] = [];
      for (let r = 0; r < N_REPS; r++) {
        reps.push(simulateFactory(FACTORY, plan.x, {
          procCV, breakProbPerMin: breakProb, breakDurationMin: breakDuration,
          totalMin, seed: 1000 + r,
        }));
      }
      return reps;
    });

  const plan = result.plan;
  const reps = result.realised;
  console.log(`# LP solver:    ${plan.solver}    iters=${plan.iters}    elapsed=${plan.elapsedMs}ms`);
  console.log(`# LP plan x = [ ${plan.x.map(v => v.toFixed(2)).join(', ')} ]`);
  console.log(`# LP NOMINAL revenue = $${plan.objective.toFixed(2)}`);
  if (plan.dualUB && plan.dualUB.length) {
    console.log(`# Shadow prices on capacity (machine $/min):`);
    for (let m = 0; m < FACTORY.machines.length; m++) {
      console.log(`#   ${FACTORY.machines[m].padEnd(8)} : $${plan.dualUB[m].toFixed(4)}/min`);
    }
  }
  if (plan.reducedCosts && plan.reducedCosts.length) {
    console.log(`# Reduced costs (binding ⇒ x = 0): ${plan.reducedCosts.map(v => v.toFixed(4)).join(', ')}`);
  }
  console.log('');

  const revenues = reps.map(r => r.realisedRevenue);
  const meanRev = revenues.reduce((s, v) => s + v, 0) / revenues.length;
  const sdRev = Math.sqrt(revenues.map(v => (v - meanRev) ** 2).reduce((s, v) => s + v, 0) / Math.max(1, revenues.length - 1));
  const gap = (plan.objective - meanRev) / plan.objective * 100;

  console.log(`# DES realised over ${N_REPS} reps (1-week sim @ ${totalMin} min):`);
  console.log(`#   mean revenue = $${meanRev.toFixed(2)}    sd = $${sdRev.toFixed(2)}`);
  console.log(`#   throughput   = [ ${reps[0].realisedThroughput.map((_, p) => {
    const mean = reps.reduce((s, r) => s + r.realisedThroughput[p], 0) / reps.length;
    return mean.toFixed(1);
  }).join(', ')} ]`);
  console.log(`#   utilisation  = [ ${reps[0].utilisation.map((_, m) => {
    const mean = reps.reduce((s, r) => s + r.utilisation[m], 0) / reps.length;
    return (mean * 100).toFixed(1) + '%';
  }).join(', ')} ]`);
  console.log(`#   breakdowns   = [ ${reps[0].breakdowns.map((_, m) => {
    const mean = reps.reduce((s, r) => s + r.breakdowns[m], 0) / reps.length;
    return mean.toFixed(2);
  }).join(', ')} ]`);
  console.log('');
  console.log(`# Plan-vs-realised gap: $${(plan.objective - meanRev).toFixed(2)} (${gap.toFixed(1)}% of nominal)`);
  console.log(`#   ↑ this is the cost of believing a deterministic LP in a stochastic factory`);
  console.log('');

  // Sweep robust factor to show simulation-optimisation feedback loop.
  if (process.env.SWEEP === '1') {
    console.log('# === Robustness sweep: shrink LP RHS by various factors ===');
    console.log('#   robust    LP nominal      mean realised      realised sd      net gain over plan-as-is');
    const baselineRev = meanRev;
    for (const rf of [1.00, 0.95, 0.90, 0.85, 0.80, 0.75]) {
      const lp2 = buildFactoryLP(FACTORY, rf);
      const sub = solveLPThenSimulate(lp2,
        (planSub: LPSolution) => {
          const r2: SimResult[] = [];
          for (let r = 0; r < N_REPS; r++) {
            r2.push(simulateFactory(FACTORY, planSub.x, {
              procCV, breakProbPerMin: breakProb, breakDurationMin: breakDuration,
              totalMin, seed: 1000 + r,
            }));
          }
          return r2;
        });
      const rev = sub.realised.map(r => r.realisedRevenue);
      const mr = rev.reduce((s, v) => s + v, 0) / rev.length;
      const sr = Math.sqrt(rev.map(v => (v - mr) ** 2).reduce((s, v) => s + v, 0) / Math.max(1, rev.length - 1));
      const delta = mr - baselineRev;
      console.log(`#   ${rf.toFixed(2)}      $${sub.plan.objective.toFixed(2).padStart(9)}      $${mr.toFixed(2).padStart(9)}      $${sr.toFixed(2).padStart(7)}      ${delta >= 0 ? '+' : ''}$${delta.toFixed(2)}`);
    }
    console.log('');
    console.log('#   Lower robust factor = lower nominal but possibly higher realised because');
    console.log('#   the plan no longer overcommits machines that breakdown / vary.');
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}

export {buildFactoryLP, simulateFactory, FACTORY};
