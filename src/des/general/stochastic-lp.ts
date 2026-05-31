'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/stochastic-lp.rs  (module des::general::stochastic_lp)
// 1:1 file move. Two-stage stochastic LP via SAA + Benders/L-shaped cuts as a DES.
//
// Declarations → Rust:
//   interface SLPProblem/Scenario/BendersIteration/SLPSolveResult/BendersOpts/UniformDemandSpec -> structs
//   interface BendersIterState (private)               -> struct
//   class BendersStation extends FixedPointIterationStation<BendersIterState> -> struct + impl trait
//   fn solveSubproblemWithDuals/solveSLPMonolithic/solveSLPBenders/buildProductionScenarios/
//      buildProductionSLP/solveProductionClosedForm   -> free fns / assoc fns
//   fn mulberry32(seed)                                -> see note (duplicate of prng.ts)
//
// Conversion notes (file-specific):
//   - INJECT RNG: this file RE-DECLARES its own `mulberry32` for scenario sampling. In Rust use the
//     single `SeededRandom` from shared/capabilities — do NOT port two copies.
//   - master LP is a long-lived `IncrementalLP` (warm-started) -> struct field; cuts arrive as
//     movable tokens, accumulated in a `Vec` cut pool.
//   - constraint matrices `number[][]` -> `Vec<Vec<f64>>` (or shared/linalg); duals are `Vec<f64>`.
//   - infeasible/unbounded subproblem outcomes -> `Result`/status enum, not bare throws.
// =============================================================================
// general/stochastic-lp.ts — TWO-STAGE STOCHASTIC LINEAR PROGRAMMING
// expressed as a discrete-event SYSTEM that BLENDS the simulation half
// of DES (scenario sampling — Monte-Carlo over ω) with the algorithmic
// half (piecewise-linear value-function approximation via Benders /
// L-shaped cuts, executed on the same tick clock). The master is a
// long-lived station owning an `IncrementalLP` instance; cuts arrive as
// movables; one Benders iteration is one tick.
//
// THE PROBLEM
// ───────────
// Two-stage SLP in MAXIMISATION form:
//
//     max  c · x + E_ω[ Q(x, ω) ]
//     s.t. A x ≤ b,   x ≥ 0
//
// where the recourse function Q is itself an LP:
//
//     Q(x, ω) = max  q · y
//               s.t. T(ω) x + W y ≤ h(ω),   y ≥ 0
//
// (The expectation is over a known distribution of ω, sampled by DES.)
//
// THREE WAYS WE SOLVE IT
// ──────────────────────
//   1. solveSLPMonolithic  — Sample Average Approximation (SAA). Sample N
//      scenarios, build ONE giant LP with first-stage variables and N copies
//      of second-stage variables, solve from scratch with `solveLPInternal`.
//      Grows linearly in N — fine for N up to a few thousand.
//
//   2. solveSLPBenders     — Benders / L-shaped decomposition AS A DES.
//      Stations:
//        MasterStation   — owns an `IncrementalLP` instance for the master.
//                          Each Benders iteration is a tick. Each cut is a
//                          movable arriving at the master via applyAddConstraint().
//        ScenarioStation — N stations (one per scenario), each solves its own
//                          small LP and emits a (Q_s, π_s*) movable.
//        AggregatorStation — averages (π_s · h_s) and (π_s T_s) across
//                          scenarios to form the optimality cut, sends it to
//                          the master.
//      Convergence: master_obj − feasible_obj ≤ tol.
//
//   3. solveSLPClosedForm  — analytical newsvendor-style oracle for the
//      simple production-planning case (used for validation).
//
// WHY PIVOTING + WARM-STARTING IS THE WHOLE POINT
// ───────────────────────────────────────────────
// Every Benders iteration adds ONE constraint to the master LP. Without
// warm-starting, each iteration's master solve is O(n³). With warm-starting
// from the previous optimal basis, dual simplex repairs the broken row in
// O(few) pivots — typically 1 or 2. That's why our `IncrementalLP` is the
// right substrate: `applyAddConstraint` followed by `solveToOptimum`
// is exactly what L-shaped methods need for the master.
// =============================================================================

import {LPProblem, solveLPInternal} from './lp';
import {IncrementalLP} from './incremental-lp';
import {
  FixedPointIterationStation, runIterativeDES,
  intrinsicCheck, externalReferenceValidator, ValidationCheck,
} from './des-base';
import {PureTransform} from '../shared/transform';

// -----------------------------------------------------------------------------
// PROBLEM TYPES
// -----------------------------------------------------------------------------

export interface SLPProblem {
  /** First-stage objective coefficients (length n_first). */
  cFirst: number[];
  /** First-stage constraint matrix A (length m_first × n_first). May be []. */
  AFirst: number[][];
  /** First-stage RHS (length m_first). */
  bFirst: number[];
  /** Second-stage objective coefficients (length n_second). */
  qSecond: number[];
  /** Second-stage technology matrix W (length m_second × n_second). */
  WSecond: number[][];
  /** Lower bound on Q(x, ω) — used to translate θ to a non-negative variable. */
  thetaLowerBound: number;
  /** Upper bound on Q(x, ω) — keeps the master bounded before any cuts arrive. */
  thetaUpperBound: number;
  /** Optional names. */
  varNames?: string[];
}

export interface Scenario {
  /** Scenario-specific T matrix (m_second × n_first). */
  T: number[][];
  /** Scenario-specific RHS h(ω) (length m_second). */
  h: number[];
  /** Probability mass (default 1/N). */
  prob?: number;
  meta?: any;
}

export interface BendersIteration {
  iter: number;
  /** First-stage decision the master proposed this iteration. */
  xMaster: number[];
  /** Master's belief about θ (i.e. about E[Q(x*, ω)]). Always an upper bound on the truth. */
  thetaMaster: number;
  /** Per-scenario subproblem objective Q(x*, ω_s). */
  scenarioValues: number[];
  /** Per-scenario subproblem dual π_s* (length m_second). */
  scenarioDuals: number[][];
  /** Empirical E[Q] from this iteration's subproblem solves. */
  expectedQ: number;
  /** Cut added to the master at the end of this iteration: a · x + θ_var ≤ b. */
  cutAdded?: {coefs: number[]; rhs: number; pi_h: number; pi_T: number[]};
  /** Master objective at x*: c·x + θ — a valid UPPER BOUND on the optimum. */
  upperBound: number;
  /** Feasible objective c·x + E[Q] — a valid LOWER BOUND on the optimum. */
  lowerBound: number;
  /** UB − LB. */
  gap: number;
  /** Stop reason if this iteration ended Benders. */
  stopReason?: 'converged' | 'iter-limit' | 'subproblem-error';
}

export interface SLPSolveResult {
  status: 'optimal' | 'unbounded' | 'infeasible' | 'iter-limit';
  /** First-stage decision. */
  x: number[];
  /** Total objective: c·x + E[Q] (empirical, on these scenarios). */
  objective: number;
  cFirstX: number;
  expectedQ: number;
  /** Per-scenario second-stage decisions. */
  yByScenario: number[][];
  /** Per-scenario Q values. */
  scenarioValues: number[];
  /** Pivots / iterations. */
  iterations: number;
  /** Solution method. */
  method: 'monolithic' | 'benders' | 'closed-form';
  /** Per-iteration trace (Benders only). */
  bendersTrace?: BendersIteration[];
  /** Wall-clock ms. */
  elapsedMs: number;
}

// -----------------------------------------------------------------------------
// SUBPROBLEM SOLVER — uses our IncrementalLP and reads duals from the
// row-0 reduced costs of the slack columns (a textbook fact: the dual of
// constraint i in `max c·y s.t. A y ≤ b, y ≥ 0` equals the reduced cost
// of slack i in the optimal tableau).
// -----------------------------------------------------------------------------

/** A recourse subproblem `max q·y s.t. W y ≤ rhs, y ≥ 0`. */
export interface SubproblemDualsInput {
  q: number[];
  W: number[][];
  rhs: number[];
}

export interface SubproblemDualsResult {
  status: 'optimal' | 'unbounded' | 'infeasible';
  y: number[];
  obj: number;
  duals: number[];
}

/** Solve a recourse subproblem and recover its dual prices from the slack
 *  reduced costs at the optimum. */
export class SubproblemWithDualsSolver extends PureTransform<SubproblemDualsInput, SubproblemDualsResult> {
  transform({q, W, rhs}: SubproblemDualsInput): SubproblemDualsResult {
    // Validate the warm-start precondition.
    for (let i = 0; i < rhs.length; i++) {
      if (rhs[i] < -1e-9) {
        // Negative RHS would require Phase-1 simplex in IncrementalLP. Fall back
        // to the general two-phase solver in lp.ts (it doesn't expose duals
        // directly, but we can recompute them manually by re-solving with the
        // optimal basis). For our test problems we never hit this case.
        throw new Error(`solveSubproblemWithDuals: rhs[${i}] = ${rhs[i]} < 0; would require Phase-1 simplex`);
      }
    }
    const lp = new IncrementalLP({sense: 'max', c: q, A: W, b: rhs});
    lp.solveToOptimum();
    if (lp.status === 'unbounded')   return {status: 'unbounded',   y: [], obj: NaN, duals: []};
    if (lp.status === 'infeasible')  return {status: 'infeasible',  y: [], obj: NaN, duals: []};
    const y     = lp.getX();
    const obj   = lp.getZ();
    // Duals = reduced costs of slack columns at the optimum.
    // For max LP, slack reduced cost ≥ 0 corresponds to dual π_i ≥ 0.
    const rc    = lp.getReducedCosts();
    const duals = rc.slice(q.length, q.length + W.length);
    return {status: 'optimal', y, obj, duals};
  }
}

/** @deprecated Use `new SubproblemWithDualsSolver().transform({q, W, rhs})`. */
export function solveSubproblemWithDuals(
  q: number[], W: number[][], rhs: number[],
): {status: 'optimal' | 'unbounded' | 'infeasible'; y: number[]; obj: number; duals: number[]} {
  return new SubproblemWithDualsSolver().transform({q, W, rhs});
}

// -----------------------------------------------------------------------------
// METHOD 1: Sample Average Approximation — monolithic LP via solveLPInternal
// -----------------------------------------------------------------------------

/** Sample Average Approximation: build and solve ONE monolithic LP over all
 *  scenarios. The problem is configuration; the scenario set is the input. */
export class SLPMonolithicSolver extends PureTransform<Scenario[], SLPSolveResult> {
  constructor(private readonly p: SLPProblem) {
    super();
  }

  transform(scenarios: Scenario[]): SLPSolveResult {
    const p = this.p;
    const t0 = Date.now();
    const N = scenarios.length;
  const nFirst  = p.cFirst.length;
  const nSecond = p.qSecond.length;
  const mFirst  = p.AFirst.length;
  const mSecond = p.WSecond.length;
  const totalVars = nFirst + N * nSecond;

  const c = new Array(totalVars).fill(0);
  for (let j = 0; j < nFirst; j++) c[j] = p.cFirst[j];
  for (let s = 0; s < N; s++) {
    const w = scenarios[s].prob ?? 1 / N;
    for (let j = 0; j < nSecond; j++) c[nFirst + s * nSecond + j] = w * p.qSecond[j];
  }
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  // First-stage constraints.
  for (let i = 0; i < mFirst; i++) {
    const row = new Array(totalVars).fill(0);
    for (let j = 0; j < nFirst; j++) row[j] = p.AFirst[i][j];
    A_ub.push(row); b_ub.push(p.bFirst[i]);
  }
  // Second-stage constraints, scenario-by-scenario.
  for (let s = 0; s < N; s++) {
    for (let i = 0; i < mSecond; i++) {
      const row = new Array(totalVars).fill(0);
      for (let j = 0; j < nFirst;  j++) row[j]                       = scenarios[s].T[i][j];
      for (let j = 0; j < nSecond; j++) row[nFirst + s * nSecond + j] = p.WSecond[i][j];
      A_ub.push(row); b_ub.push(scenarios[s].h[i]);
    }
  }
  const lp: LPProblem = {sense: 'max', c, A_ub, b_ub};
  const sol = solveLPInternal(lp, {maxIter: 50000});

  const x = sol.x.slice(0, nFirst);
  const yByScenario: number[][] = [];
  const scenarioValues: number[] = [];
  for (let s = 0; s < N; s++) {
    const y_s = sol.x.slice(nFirst + s * nSecond, nFirst + (s + 1) * nSecond);
    yByScenario.push(y_s);
    let qy = 0;
    for (let j = 0; j < nSecond; j++) qy += p.qSecond[j] * y_s[j];
    scenarioValues.push(qy);
  }
  let cFirstX = 0; for (let j = 0; j < nFirst; j++) cFirstX += p.cFirst[j] * x[j];
  let expectedQ = 0;
  for (let s = 0; s < N; s++) expectedQ += (scenarios[s].prob ?? 1 / N) * scenarioValues[s];

  return {
    status: sol.status as any,
    x, objective: sol.objective, cFirstX, expectedQ,
    yByScenario, scenarioValues,
    iterations: sol.iters ?? 0,
    method: 'monolithic',
    elapsedMs: Date.now() - t0,
  };
  }
}

/** @deprecated Use `new SLPMonolithicSolver(p).transform(scenarios)`. */
export function solveSLPMonolithic(p: SLPProblem, scenarios: Scenario[]): SLPSolveResult {
  return new SLPMonolithicSolver(p).transform(scenarios);
}

// -----------------------------------------------------------------------------
// METHOD 2: Benders Decomposition (L-shaped) AS A DES
//
// Master station owns an IncrementalLP with variables [x_1..x_n, θ_var]
// where θ_var = θ − thetaLowerBound (so θ_var ≥ 0 is enforced). The
// objective is max c · x + θ_var + thetaLowerBound, encoded as objective
// vector [c, 1] (we add the constant separately at the end).
//
// At each iteration:
//   1. The MasterStation produces (x*, θ_var*).
//   2. N ScenarioStations, in parallel, solve their subproblem at x* and
//      emit (Q_s, π_s*) movables.
//   3. The AggregatorStation averages (π · h) and (π · T) across scenarios,
//      forms an optimality cut θ ≤ E[π·h] − E[π T] · x  (equivalently
//      θ_var + E[π T] · x ≤ E[π·h] − thetaLowerBound) and ships it to
//      the master via applyAddConstraint.
//   4. Convergence test: gap = master_obj − feasible_obj ≤ tol.
// -----------------------------------------------------------------------------

export interface BendersOpts {
  maxIter?: number;
  tol?: number;
  verbose?: boolean;
  /** Optional path to a JSON shaped like the scipy SLP reference output:
   *  `{ x: number[], objective: number }`. When provided, the Benders
   *  station auto-attaches an external-reference validator that compares
   *  the first-stage solution element-wise within `referenceTol`. */
  referencePath?: string;
  referenceTol?: number;
  silentIfMissing?: boolean;
}

// =============================================================================
// BendersStation — concrete leaf of FixedPointIterationStation<BendersIterState>.
//
//   initialState  → empty bookkeeping (iter 0, gap = ∞)
//   applyOperator → one Benders iteration: master solve → scenario duals →
//                   build optimality cut → applyAddConstraint to master
//   delta(prev,next) → next.gap (UB − LB) — feeds shouldStop ≤ tol
// =============================================================================

interface BendersIterState {
  iter: number;
  gap: number;
  upperBound: number;
  lowerBound: number;
  /** Sentinel set when the master problem becomes infeasible/unbounded mid-loop. */
  status: 'running' | 'optimal' | 'infeasible' | 'unbounded' | 'subproblem-error';
}

class BendersStation extends FixedPointIterationStation<BendersIterState> {
  private readonly p: SLPProblem;
  private readonly scenarios: Scenario[];
  private readonly N: number;
  private readonly nFirst: number;
  private readonly mSecond: number;
  private readonly verbose: boolean;
  private readonly master: IncrementalLP;
  readonly trace: BendersIteration[] = [];
  bestLowerBound = -Infinity;
  bestX: number[];
  bestY: number[][] = [];
  bestScenarioValues: number[] = [];
  pivotsTotal = 0;
  /** Latest scenario values / duals / Y, captured during applyOperator so
   *  the surrounding solveSLPBenders wrapper can return them. */
  lastScenarioValues: number[] = [];
  lastScenarioY: number[][] = [];
  lastXMaster: number[];
  finalStatus: 'optimal' | 'iter-limit' | 'infeasible' | 'unbounded' | 'subproblem-error' = 'iter-limit';

  constructor(p: SLPProblem, scenarios: Scenario[], opts: Required<BendersOpts>) {
    super('benders', {tol: opts.tol, maxIter: opts.maxIter});
    this.p = p;
    this.scenarios = scenarios;
    this.N = scenarios.length;
    this.nFirst = p.cFirst.length;
    this.mSecond = p.WSecond.length;
    this.verbose = opts.verbose;
    this.bestX = new Array(this.nFirst).fill(0);
    this.lastXMaster = new Array(this.nFirst).fill(0);

    // Build the master.
    const masterC = [...p.cFirst, 1];
    const masterA: number[][] = [];
    const masterB: number[] = [];
    for (let i = 0; i < p.AFirst.length; i++) {
      masterA.push([...p.AFirst[i], 0]);
      masterB.push(p.bFirst[i]);
    }
    const thetaSpan = p.thetaUpperBound - p.thetaLowerBound;
    if (thetaSpan <= 0) throw new Error('thetaUpperBound must exceed thetaLowerBound');
    masterA.push([...new Array(this.nFirst).fill(0), 1]);
    masterB.push(thetaSpan);
    this.master = new IncrementalLP({
      sense: 'max', c: masterC, A: masterA, b: masterB,
      varNames: [...(p.varNames?.slice() ?? p.cFirst.map((_, i) => `x${i + 1}`)), 'theta'],
    });
    this.bootstrap();

    // Intrinsic invariants for any Benders run.
    this.addValidator(intrinsicCheck<BendersStation>({
      name: 'benders.optimal-implies-gap-le-tol',
      group: 'benders-intrinsic',
      predicate: st => st.finalStatus !== 'optimal' || st.getCurrent().gap <= st['tol'] + 1e-9,
      expected: 'gap ≤ tol when optimal',
      observedFn: st => `status=${st.finalStatus}  gap=${st.getCurrent().gap}  tol=${st['tol']}`,
      details: 'optimality declared but UB − LB exceeds tol',
    }));
    this.addValidator(intrinsicCheck<BendersStation>({
      name: 'benders.lower-bound-le-upper-bound',
      group: 'benders-intrinsic',
      predicate: st => {
        const cur = st.getCurrent();
        if (!cur || !Number.isFinite(cur.upperBound) || !Number.isFinite(cur.lowerBound)) return true;
        return cur.lowerBound <= cur.upperBound + 1e-6;
      },
      expected: 'LB ≤ UB',
      observedFn: st => `LB=${st.getCurrent()?.lowerBound}  UB=${st.getCurrent()?.upperBound}`,
      details: 'lower bound exceeds upper bound — would indicate a duality bug',
    }));

    // Optional external-reference validator (e.g. scipy extensive-form LP).
    if (opts.referencePath) {
      const refTol = opts.referenceTol ?? 1e-3;
      this.addValidator(externalReferenceValidator<BendersStation>({
        name: 'benders.solution-vs-reference',
        group: 'benders-external',
        referencePath: opts.referencePath,
        silentIfMissing: opts.silentIfMissing,
        compare: (st, ref) => {
          const x = st.lastXMaster;
          const refX = ref.x as number[];
          const refObj = ref.objective as number;
          const out: ValidationCheck[] = [];
          if (Array.isArray(refX) && refX.length === x.length) {
            let maxAbs = 0; let argmax = -1;
            for (let i = 0; i < x.length; i++) {
              const e = Math.abs(x[i] - refX[i]);
              if (e > maxAbs) { maxAbs = e; argmax = i; }
            }
            const passed = maxAbs <= refTol;
            out.push({
              name: 'benders.x-vs-reference', passed,
              observed: `max|Δx|=${maxAbs.toExponential(3)} at i=${argmax}`,
              expected: `≤ ${refTol}`,
              details: passed ? undefined : `x[${argmax}]=${x[argmax]}  ref=${refX[argmax]}`,
            });
          }
          if (Number.isFinite(refObj)) {
            const cur = st.getCurrent();
            const obj = Number.isFinite(cur?.lowerBound) ? cur.lowerBound : NaN;
            const e = Math.abs(obj - refObj) / Math.max(1e-12, Math.abs(refObj));
            const passed = e <= refTol;
            out.push({
              name: 'benders.objective-vs-reference', passed,
              observed: obj.toPrecision(8), expected: refObj.toPrecision(8),
              details: passed ? undefined : `rel-err=${e.toExponential(3)} > ${refTol}`,
            });
          }
          return out;
        },
      }));
    }
  }

  protected initialState(): BendersIterState {
    return {iter: 0, gap: Infinity, upperBound: Infinity, lowerBound: -Infinity, status: 'running'};
  }

  protected applyOperator(prev: BendersIterState): BendersIterState {
    const iter = prev.iter + 1;
    // 1. Master solve.
    const pivotsBefore = this.master.tick;
    this.master.solveToOptimum();
    this.pivotsTotal += this.master.tick - pivotsBefore;
    if (this.master.status !== 'optimal') {
      this.finalStatus = this.master.status as 'infeasible' | 'unbounded';
      return {iter, gap: 0, upperBound: NaN, lowerBound: this.bestLowerBound,
              status: this.master.status as 'infeasible' | 'unbounded'};
    }
    const xMaster = this.master.getX().slice(0, this.nFirst);
    const thetaVarMaster = this.master.getX()[this.nFirst];
    const thetaMaster = thetaVarMaster + this.p.thetaLowerBound;
    const cTx = this.p.cFirst.reduce((s, ci, i) => s + ci * xMaster[i], 0);
    this.lastXMaster = xMaster;

    // 2. Scenario subproblems.
    const scenarioValues: number[] = [];
    const scenarioDuals: number[][] = [];
    const scenarioY: number[][] = [];
    for (let s = 0; s < this.N; s++) {
      const sc = this.scenarios[s];
      const rhs = sc.h.map((hi, i) => {
        let v = hi;
        for (let j = 0; j < this.nFirst; j++) v -= sc.T[i][j] * xMaster[j];
        return v;
      });
      const sub = solveSubproblemWithDuals(this.p.qSecond, this.p.WSecond, rhs);
      if (sub.status !== 'optimal') {
        this.finalStatus = 'subproblem-error';
        const stop: BendersIteration = {
          iter, xMaster, thetaMaster,
          scenarioValues, scenarioDuals,
          expectedQ: NaN,
          upperBound: cTx + thetaMaster, lowerBound: this.bestLowerBound,
          gap: cTx + thetaMaster - this.bestLowerBound,
          stopReason: 'subproblem-error',
        };
        this.trace.push(stop);
        return {iter, gap: 0, upperBound: NaN, lowerBound: NaN, status: 'subproblem-error'};
      }
      scenarioValues.push(sub.obj);
      scenarioDuals.push(sub.duals);
      scenarioY.push(sub.y);
    }
    let expectedQ = 0;
    for (let s = 0; s < this.N; s++) expectedQ += (this.scenarios[s].prob ?? 1 / this.N) * scenarioValues[s];
    const upperBound = cTx + thetaMaster;
    const lowerBound = cTx + expectedQ;
    if (lowerBound > this.bestLowerBound) {
      this.bestLowerBound = lowerBound;
      this.bestX = xMaster.slice();
      this.bestY = scenarioY.map(y => y.slice());
      this.bestScenarioValues = scenarioValues.slice();
    }
    this.lastScenarioValues = scenarioValues;
    this.lastScenarioY = scenarioY;
    const gap = upperBound - lowerBound;
    if (this.verbose) {
      console.log(`[Benders] iter=${iter}  x=[${xMaster.map(v => v.toFixed(2)).join(',')}]  ` +
                  `θ=${thetaMaster.toFixed(3)}  E[Q]=${expectedQ.toFixed(3)}  ` +
                  `UB=${upperBound.toFixed(3)}  LB=${lowerBound.toFixed(3)}  gap=${gap.toExponential(2)}`);
    }

    // 3. Convergence?
    if (gap <= this.tol) {
      this.finalStatus = 'optimal';
      this.trace.push({
        iter, xMaster, thetaMaster, scenarioValues, scenarioDuals,
        expectedQ, upperBound, lowerBound, gap, stopReason: 'converged',
      });
      return {iter, gap, upperBound, lowerBound, status: 'optimal'};
    }

    // 4. Build the optimality cut and add it to the master.
    let pi_h_avg = 0;
    const pi_T_avg = new Array(this.nFirst).fill(0);
    for (let s = 0; s < this.N; s++) {
      const w = this.scenarios[s].prob ?? 1 / this.N;
      const pi = scenarioDuals[s];
      const sc = this.scenarios[s];
      for (let i = 0; i < this.mSecond; i++) pi_h_avg += w * pi[i] * sc.h[i];
      for (let j = 0; j < this.nFirst; j++) {
        let s_val = 0;
        for (let i = 0; i < this.mSecond; i++) s_val += pi[i] * sc.T[i][j];
        pi_T_avg[j] += w * s_val;
      }
    }
    const cutCoefs = [...pi_T_avg, 1];
    const cutRhs   = pi_h_avg - this.p.thetaLowerBound;
    this.master.applyAddConstraint(cutCoefs, cutRhs, `cut${iter}`);
    this.trace.push({
      iter, xMaster, thetaMaster, scenarioValues, scenarioDuals,
      expectedQ,
      cutAdded: {coefs: cutCoefs, rhs: cutRhs, pi_h: pi_h_avg, pi_T: pi_T_avg.slice()},
      upperBound, lowerBound, gap,
    });
    return {iter, gap, upperBound, lowerBound, status: 'running'};
  }

  protected delta(_prev: BendersIterState, next: BendersIterState): number {
    return next.gap;
  }

  protected override shouldStop(iter: number, lastDelta: number): boolean {
    if (this.current && this.current.status !== 'running' && iter > 0) {
      this.convergenceReason = this.current.status === 'optimal' ? 'converged' : 'maxiter';
      return true;
    }
    return super.shouldStop(iter, lastDelta);
  }
}

export function solveSLPBenders(
  p: SLPProblem, scenarios: Scenario[], opts: BendersOpts = {},
): SLPSolveResult {
  const t0 = Date.now();
  const filled: Required<BendersOpts> = {
    maxIter: opts.maxIter ?? 100,
    tol: opts.tol ?? 1e-6,
    verbose: opts.verbose ?? false,
    referencePath: opts.referencePath ?? '',
    referenceTol: opts.referenceTol ?? 1e-3,
    silentIfMissing: opts.silentIfMissing ?? true,
  };
  const N = scenarios.length;
  const nFirst = p.cFirst.length;
  // unused now (kept on the original signature for clarity)
  void p.qSecond.length;
  void p.WSecond.length;
  void N; void nFirst;

  const station = new BendersStation(p, scenarios, filled);
  runIterativeDES([station]);
  const finalState = station.getCurrent();
  const status = station.finalStatus;
  if (status === 'optimal') {
    const cTx = p.cFirst.reduce((s, ci, i) => s + ci * station.lastXMaster[i], 0);
    return {
      status: 'optimal',
      x: station.lastXMaster.slice(),
      objective: finalState.lowerBound,
      cFirstX: cTx,
      expectedQ: finalState.lowerBound - cTx,
      yByScenario: station.lastScenarioY.map(y => y.slice()),
      scenarioValues: station.lastScenarioValues.slice(),
      iterations: finalState.iter,
      method: 'benders', bendersTrace: station.trace,
      elapsedMs: Date.now() - t0,
    };
  }
  if (status === 'infeasible' || status === 'unbounded') {
    return {
      status, x: station.bestX, objective: NaN, cFirstX: NaN, expectedQ: NaN,
      yByScenario: [], scenarioValues: [],
      iterations: finalState.iter, method: 'benders',
      bendersTrace: station.trace, elapsedMs: Date.now() - t0,
    };
  }
  if (status === 'subproblem-error') {
    return {
      status: 'infeasible', x: station.bestX, objective: NaN, cFirstX: NaN, expectedQ: NaN,
      yByScenario: station.bestY, scenarioValues: station.bestScenarioValues,
      iterations: finalState.iter, method: 'benders',
      bendersTrace: station.trace, elapsedMs: Date.now() - t0,
    };
  }
  return {
    status: 'iter-limit',
    x: station.bestX, objective: station.bestLowerBound,
    cFirstX: p.cFirst.reduce((s, ci, i) => s + ci * station.bestX[i], 0),
    expectedQ: station.bestLowerBound - p.cFirst.reduce((s, ci, i) => s + ci * station.bestX[i], 0),
    yByScenario: station.bestY, scenarioValues: station.bestScenarioValues,
    iterations: filled.maxIter, method: 'benders', bendersTrace: station.trace,
    elapsedMs: Date.now() - t0,
  };
}

// -----------------------------------------------------------------------------
// SCENARIO SAMPLING UTILITIES
// -----------------------------------------------------------------------------

export interface UniformDemandSpec {
  /** Range [a_i, b_i] per second-stage product. */
  ranges: Array<[number, number]>;
  /** Random seed. */
  seed: number;
}

/** mulberry32 PRNG. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample N uniform-demand scenarios for the production-planning template:
 *  W = [I; I], T = [-I; 0], h = [0...0, D_1, ..., D_n_second]. The scenario
 *  count `N` is configuration; the demand spec is the `transform` input. */
export class ProductionScenarioBuilder extends PureTransform<UniformDemandSpec, Scenario[]> {
  constructor(private readonly N: number) {
    super();
  }

  transform(spec: UniformDemandSpec): Scenario[] {
    const N = this.N;
    const r = mulberry32(spec.seed);
    const n = spec.ranges.length;
    const scenarios: Scenario[] = [];
    for (let s = 0; s < N; s++) {
      const D = new Array(n);
      for (let i = 0; i < n; i++) {
        const [a, b] = spec.ranges[i];
        D[i] = a + r() * (b - a);
      }
      // T = [-I; 0]  (capacity rows put -x; demand rows are scenario-independent of x)
      const T: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row = new Array(n).fill(0); row[i] = -1; T.push(row);
      }
      for (let i = 0; i < n; i++) {
        T.push(new Array(n).fill(0));
      }
      // h = [0...0, D_1, ..., D_n]
      const h = [...new Array(n).fill(0), ...D];
      scenarios.push({T, h, prob: 1 / N, meta: {D}});
    }
    return scenarios;
  }
}

/** @deprecated Use `new ProductionScenarioBuilder(N).transform(spec)`. */
export function buildProductionScenarios(spec: UniformDemandSpec, N: number): Scenario[] {
  return new ProductionScenarioBuilder(N).transform(spec);
}

/** Cost/revenue (and optional budget) for the production-planning template. */
export interface ProductionSLPInput {
  c: number[];
  p: number[];
  budget?: number;
}

/** Build the production-planning SLPProblem template with cost c and revenue p. */
export class ProductionSLPBuilder extends PureTransform<ProductionSLPInput, SLPProblem> {
  transform({c, p, budget}: ProductionSLPInput): SLPProblem {
    const n = c.length;
    if (p.length !== n) throw new Error('cost and revenue must have same length');
  // First-stage objective is -c · x (cost minimisation in max-form).
  const cFirst = c.map(ci => -ci);
  // First-stage constraints: optional budget x_1 + ... + x_n ≤ B.
  const AFirst: number[][] = [];
  const bFirst: number[] = [];
  if (budget !== undefined) { AFirst.push(new Array(n).fill(1)); bFirst.push(budget); }
  // Second stage: max p · y, with W = [I; I] (capacity y_i ≤ x_i ; demand y_i ≤ D_i).
  const W: number[][] = [];
  for (let i = 0; i < n; i++) { const row = new Array(n).fill(0); row[i] = 1; W.push(row); }
  for (let i = 0; i < n; i++) { const row = new Array(n).fill(0); row[i] = 1; W.push(row); }
  // Q ≥ 0 always (y = 0 feasible). Bound above by  Σ p_i · max(D_i, x_i_max).
  const thetaLB = 0;
  const thetaUB = p.reduce((acc, pi, i) => acc + pi * 10000, 0);   // large but finite
  return {
    cFirst, AFirst, bFirst,
    qSecond: p, WSecond: W,
    thetaLowerBound: thetaLB, thetaUpperBound: thetaUB,
    varNames: c.map((_, i) => `x${i + 1}`),
  };
  }
}

/** @deprecated Use `new ProductionSLPBuilder().transform({c, p, budget})`. */
export function buildProductionSLP(c: number[], p: number[], budget?: number): SLPProblem {
  return new ProductionSLPBuilder().transform({c, p, budget});
}

// -----------------------------------------------------------------------------
// METHOD 3: Closed-form newsvendor solution (validation oracle)
// -----------------------------------------------------------------------------

/** Closed-form optimum of the multi-product newsvendor problem WITHOUT a budget
 *  constraint, with uniform demand: x_i* = a_i + (b_i − a_i) · (p_i − c_i)/p_i.
 *  Returns the optimum first-stage decision and the analytical expected
 *  objective (using the formula E[min(x, D)] = x − (x−a)²/(2(b−a)) for x ∈ [a,b]). */
/** Cost, revenue, and per-product uniform demand ranges for the closed-form oracle. */
export interface ProductionClosedFormInput {
  c: number[];
  p: number[];
  ranges: Array<[number, number]>;
}

/** Analytical newsvendor-style oracle for the production-planning case. */
export class ProductionClosedFormSolver extends PureTransform<ProductionClosedFormInput, SLPSolveResult> {
  transform({c, p, ranges}: ProductionClosedFormInput): SLPSolveResult {
    const t0 = Date.now();
    const n = c.length;
  const x = new Array(n);
  let zVal = 0;
  for (let i = 0; i < n; i++) {
    const [a, b] = ranges[i];
    if (p[i] <= c[i]) x[i] = 0;        // not profitable
    else {
      const xi = a + (b - a) * (p[i] - c[i]) / p[i];
      x[i] = Math.max(0, Math.min(xi, b));
    }
    // E[min(x_i, D_i)] for uniform [a,b]:
    let eMin: number;
    if (x[i] <= a)        eMin = x[i];
    else if (x[i] >= b)   eMin = (a + b) / 2;
    else                  eMin = x[i] - (x[i] - a) * (x[i] - a) / (2 * (b - a));
    zVal += -c[i] * x[i] + p[i] * eMin;
  }
  let cFirstX = 0; for (let i = 0; i < n; i++) cFirstX += -c[i] * x[i];
  return {
    status: 'optimal',
    x, objective: zVal,
    cFirstX, expectedQ: zVal - cFirstX,
    yByScenario: [], scenarioValues: [],
    iterations: 0, method: 'closed-form',
    elapsedMs: Date.now() - t0,
  };
  }
}

/** @deprecated Use `new ProductionClosedFormSolver().transform({c, p, ranges})`. */
export function solveProductionClosedForm(
  c: number[], p: number[], ranges: Array<[number, number]>,
): SLPSolveResult {
  return new ProductionClosedFormSolver().transform({c, p, ranges});
}
