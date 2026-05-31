// RUST MIGRATION: target module src/des/general/lp_des.rs.
// RUST MIGRATION: DESSimplexTrace, DESSimplexOptions, DESSimplexSolution, and Preprocessed become serde structs; SimplexState is a private mutable state struct.
// RUST MIGRATION: SimplexRoleStation and concrete phase/entering/leaving/pivot/observer stations become structs implementing Station traits instead of inheritance.
// RUST MIGRATION: solveLPViaDES is DES-visible solver orchestration and should be a PureTransform entry struct returning Result<DESSimplexSolution, Error>.
// RUST MIGRATION: Tableau mutation uses Vec<Vec<f64>> with explicit borrowing; preprocessing and pivot errors should be Result/status values.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/lp-des.rs  (module des::general::lp_des)
// 1:1 file move. Two-phase simplex driven as a DES (entering/leaving/pivot/observer stations; one pivot = one tick).
//
// Declarations → Rust:
//   interface DESSimplexTrace / DESSimplexOptions -> structs (Default; optionals -> Option<T>)
//   interface DESSimplexSolution extends LPSolution -> struct (compose/flatten LPSolution fields)
//   class SimplexState / Preprocessed -> structs (shared mutable tableau + preprocessing data)
//   class Entering/Leaving/Pivot/PhaseTransition/Observer Station (extend SimplexRoleStation)
//                                    -> structs `impl` a SimplexRole/station trait (base -> trait)
//   fn solveLPViaDES               -> FallibleTransform<LPProblem, DESSimplexSolution> or fn
//
// Conversion notes (file-specific):
//   - Dense tableau + basis live in SimplexState, mutated by stations each tick -> shared `&mut`
//     state (Rc<RefCell<SimplexState>> or pass &mut); `number[][]` -> matrix/Vec<Vec<f64>>.
//   - Pivot rule (Dantzig/Bland) -> enum; phase-1/phase-2 share the loop with a swapped cost row.
//   - `extends LPSolution` -> no interface inheritance; compose. Failure via LP status, not throw.
// =============================================================================

// =============================================================================
// LP solver implemented as a DES.
//
// The user observed:
//
//   "we can mimic an LP solver using our own DES system - it's basically
//    recursive and finds steepest path and keeps going, we can use our
//    existing engine to model and solve LP problems"
//
// This file makes that concrete. Simplex IS naturally a discrete-event
// process: it walks vertex-to-vertex along the boundary of the feasible
// polytope, and each PIVOT is a discrete event. The DES engine drives
// the walk; LP-specific logic is encapsulated in four stationary
// entities:
//
//   ┌────────────────────┐    ┌────────────────────┐    ┌──────────────┐
//   │ EnteringStation    │───▶│ LeavingStation     │───▶│ PivotStation │
//   │ "steepest reduced  │    │ "min-ratio test:   │    │ "elementary  │
//   │  cost direction"   │    │  furthest along the│    │  row ops"    │
//   │ (Dantzig or Bland) │    │  edge before exit" │    │              │
//   └────────────────────┘    └────────────────────┘    └──────────────┘
//                                                              │
//   ┌──────────────────────────────────────────────────────────┘
//   ▼
//   ┌────────────────────┐
//   │ ObserverStation    │  (records vertex + objective into the trace,
//   │                    │   exactly the same way Census does in our
//   │                    │   compartmental SEIR model)
//   └────────────────────┘
//
// One pivot = one tick. Two phases of simplex (phase-1 for feasibility
// when the origin is not a basic-feasible solution, phase-2 for
// optimality) share the same DES loop with a different cost row in the
// tableau. The simulation halts when the EnteringStation reports no
// improving direction (optimality), no positive ratio is available
// (unboundedness), or the iteration cap is hit.
//
// Numerical equivalence with the in-process simplex (`solveLPInternal`)
// and scipy:highs is verified in `runners/validate-lp.ts` — the DES
// driver pivots through the SAME sequence of vertices and converges to
// the SAME optimum bit-for-bit on non-degenerate LPs.
//
// IS THIS EFFICIENT?
// ──────────────────
// No. A direct simplex implementation in 80 lines would beat this
// because the DES per-tick scaffolding (status checks, history
// captures, stage transitions) is overhead. The point is to
// demonstrate that the DES engine is generic enough to host an LP
// solver — i.e. that "events", "queues", "stations", and "movables"
// are a sufficient computational substrate for vertex-walking
// optimisation. The animation pipeline gives free visualisation of
// the polytope walk for 2-D problems.
// =============================================================================

import {LPProblem, LPSolution, LPStatus} from './lp';
import {DESStation, runIterativeDES} from './des-base';

// -----------------------------------------------------------------------------
// Shared simulation state.
//
// Every station reads / writes this single object. This is exactly the
// pattern used by Census + FieldStation in the calculus engine: the
// stations are functional roles; the state lives outside them; the
// simulation just orchestrates the order of access.
// -----------------------------------------------------------------------------
export interface DESSimplexTrace {
  pivotHistory: Array<{
    tick: number;
    phase: 1 | 2;
    enter: number;
    leave: number;
    obj: number;
    pivotElt: number;
  }>;
  vertexHistory: number[][];
  objHistory: number[];
}

class SimplexState {
  T!: number[][];                  // tableau, (m+1) × (ncols+1)
  basis!: number[];                // length m, the column index basic in each row
  m!: number;
  ncols!: number;
  iters = 0;
  maxIter!: number;
  tol!: number;
  status: LPStatus | 'in-progress' | 'phase-done' = 'in-progress';
  phase: 1 | 2 = 1;
  pivotRule!: 'dantzig' | 'bland';
  // Pending pivot info — set by EnteringStation, consumed by LeavingStation, then PivotStation.
  pendingEntering = -1;
  pendingLeaving = -1;
  // Phase 1 bookkeeping.
  artificialCols: Set<number> = new Set();
  hasArtificials!: boolean;
  // Saved phase-2 cost row for when we transition.
  phase2CostRow!: number[];
  // Original problem mapping (y-space → x-space).
  n_orig!: number;
  shifts!: number[];
  yIndexOfPos!: number[];
  freeNeg!: number[];
  sense!: 'max' | 'min';
  // Trace for animation / observability.
  trace: DESSimplexTrace = {pivotHistory: [], vertexHistory: [], objHistory: []};
  snapshotVersion = 0;
  observedSnapshotVersion = -1;
  skipSelectionThisTick = false;

  constructor(args: {
    T: number[][]; basis: number[]; m: number; ncols: number;
    maxIter: number; tol: number;
    phase2CostRow: number[];
    artificialCols: Set<number>;
    n_orig: number; shifts: number[]; yIndexOfPos: number[]; freeNeg: number[];
    sense: 'max' | 'min';
    pivotRule: 'dantzig' | 'bland';
  }) {
    Object.assign(this, args);
    this.hasArtificials = args.artificialCols.size > 0;
    this.phase = this.hasArtificials ? 1 : 2;
  }

  /** Reconstruct the original x vector from the current basic feasible solution. */
  currentVertex(): number[] {
    const y = new Array(this.ncols).fill(0);
    for (let r = 0; r < this.m; r++) y[this.basis[r]] = this.T[r][this.ncols];
    const x = new Array(this.n_orig).fill(0);
    for (let i = 0; i < this.n_orig; i++) {
      const yp = y[this.yIndexOfPos[i]];
      const yn = this.freeNeg[i] >= 0 ? y[this.freeNeg[i]] : 0;
      x[i] = yp - yn + this.shifts[i];
    }
    return x;
  }

  /** Current objective in the original (max/min) sense.
   *
   * In our cost-row convention T[m][j] = c_B^T B^{-1} A_j − c_j (the
   * "negative reduced cost" in the working-max sense), and so T[m][ncols]
   * = c_B^T B^{-1} b′ = current objective value in the working-max sense.
   * We always work in MAX form internally (callers passing `sense:'min'`
   * have their c flipped at the API boundary), so the working-max
   * objective IS the original-max objective; for original-min we return
   * its negation. */
  currentObjective(): number {
    if (this.phase === 1) return NaN; // phase-1 objective is sum-of-artificials, not user obj
    const z = this.T[this.m][this.ncols];
    return this.sense === 'max' ? z : -z;
  }

  markSnapshot(): void {
    this.snapshotVersion++;
  }
}

// -----------------------------------------------------------------------------
// Stations. Each is a functional role; the per-tick orchestration stays
// inside the DES driver.
// -----------------------------------------------------------------------------

abstract class SimplexRoleStation extends DESStation {
  constructor(id: string, protected readonly state: SimplexState) {
    super(id);
  }

  override hasWork(): boolean {
    return this.state.status === 'in-progress' || this.state.status === 'phase-done';
  }
}

function pivotTableau(s: SimplexState, enter: number, leave: number): number {
  const ncols = s.T[0].length;
  const pv = s.T[leave][enter];
  for (let j = 0; j < ncols; j++) s.T[leave][j] /= pv;
  for (let r = 0; r < s.T.length; r++) {
    if (r === leave) continue;
    const factor = s.T[r][enter];
    if (factor === 0) continue;
    for (let j = 0; j < ncols; j++) s.T[r][j] -= factor * s.T[leave][j];
  }
  s.basis[leave] = enter;
  s.iters++;
  s.pendingEntering = -1;
  s.pendingLeaving = -1;
  if (s.iters >= s.maxIter) s.status = 'iter-limit';
  return pv;
}

class PhaseTransitionStation extends SimplexRoleStation {
  constructor(state: SimplexState) {
    super('lp-phase-transition-station', state);
  }

  runTimeStep(): void {
    const s = this.state;
    if (s.status !== 'phase-done') return;
    if (s.phase === 2) {
      s.status = 'optimal';
      return;
    }

    let sumArt = 0;
    for (let r = 0; r < s.m; r++) {
      if (s.artificialCols.has(s.basis[r])) sumArt += s.T[r][s.ncols];
    }
    if (sumArt > 1e-7) {
      s.status = 'infeasible';
      return;
    }

    for (let r = 0; r < s.m; r++) {
      if (!s.artificialCols.has(s.basis[r])) continue;
      for (let j = 0; j < s.ncols; j++) {
        if (!s.artificialCols.has(j) && Math.abs(s.T[r][j]) > s.tol) {
          pivotTableau(s, j, r);
          if (s.iters >= s.maxIter) return;
          break;
        }
      }
    }

    s.T[s.m] = s.phase2CostRow.slice();
    for (let r = 0; r < s.m; r++) {
      const cb = s.T[s.m][s.basis[r]];
      if (cb !== 0) {
        for (let j = 0; j <= s.ncols; j++) s.T[s.m][j] -= cb * s.T[r][j];
      }
    }
    s.phase = 2;
    s.status = 'in-progress';
    s.skipSelectionThisTick = true;
    s.markSnapshot();
  }
}

class EnteringStation extends SimplexRoleStation {
  constructor(state: SimplexState) {
    super('lp-entering-station', state);
  }

  /**
   * Scan the cost row for a column with negative reduced cost (entering = a
   * direction along which the objective improves). Two pivot rules:
   *
   *   - 'dantzig': pick the column with the MOST negative reduced cost
   *     (steepest immediate improvement; usually fewest pivots)
   *   - 'bland'  : pick the FIRST column with strictly negative reduced
   *     cost (lowest index); guarantees termination on degenerate LPs
   */
  runTimeStep(): void {
    const s = this.state;
    if (s.status !== 'in-progress') return;
    if (s.skipSelectionThisTick) {
      s.skipSelectionThisTick = false;
      return;
    }
    const m = s.m;
    const ncols = s.ncols;
    let entering = -1;
    if (s.pivotRule === 'dantzig') {
      let bestRC = -s.tol;
      for (let j = 0; j < ncols; j++) {
        if (s.artificialCols.has(j)) continue; // artificials never re-enter, in either phase
        if (s.T[m][j] < bestRC) { bestRC = s.T[m][j]; entering = j; }
      }
    } else {
      for (let j = 0; j < ncols; j++) {
        if (s.artificialCols.has(j)) continue;
        if (s.T[m][j] < -s.tol) { entering = j; break; }
      }
    }
    if (entering === -1) {
      // No improving direction → optimal for the current phase.
      s.pendingEntering = -1;
      s.status = 'phase-done';
    } else {
      s.pendingEntering = entering;
    }
  }
}

class LeavingStation extends SimplexRoleStation {
  constructor(state: SimplexState) {
    super('lp-leaving-station', state);
  }

  /**
   * Min-ratio test on the entering column: the ratio b_r / a_{r,enter}
   * tells how far along the entering edge we can travel before some basic
   * variable hits zero. Pick the row with the smallest positive ratio.
   * Bland's tie-break (lowest basis index) ensures finite termination on
   * degenerate problems.
   */
  runTimeStep(): void {
    const s = this.state;
    if (s.pendingEntering === -1) return;
    const m = s.m;
    const ncols = s.ncols;
    const entering = s.pendingEntering;
    let leaving = -1;
    let bestRatio = +Infinity;
    for (let r = 0; r < m; r++) {
      if (s.T[r][entering] > s.tol) {
        const ratio = s.T[r][ncols] / s.T[r][entering];
        if (ratio < bestRatio - s.tol
            || (Math.abs(ratio - bestRatio) <= s.tol
                && (leaving < 0 || s.basis[r] < s.basis[leaving]))) {
          bestRatio = ratio;
          leaving = r;
        }
      }
    }
    if (leaving === -1) {
      // No positive entry in the entering column ⇒ unbounded direction.
      s.status = 'unbounded';
      s.pendingLeaving = -1;
    } else {
      s.pendingLeaving = leaving;
    }
  }
}

class PivotStation extends SimplexRoleStation {
  constructor(state: SimplexState) {
    super('lp-pivot-station', state);
  }

  /** Elementary row operations: scale the pivot row to make pivot=1, then
   *  subtract appropriate multiples from every other row to zero out the
   *  pivot column. This is the SAME operation Gauss-Jordan does. */
  runTimeStep(): void {
    const s = this.state;
    if (s.pendingEntering === -1 || s.pendingLeaving === -1) return;
    const enter = s.pendingEntering;
    const leave = s.pendingLeaving;
    const pivotElt = pivotTableau(s, enter, leave);
    s.markSnapshot();
    s.trace.pivotHistory.push({
      tick: s.iters, phase: s.phase,
      enter, leave, obj: s.currentObjective(), pivotElt,
    });
  }
}

class ObserverStation extends SimplexRoleStation {
  constructor(state: SimplexState) {
    super('lp-observer-station', state);
  }

  /** Capture the current vertex and objective into the trace. Exactly
   *  the role Census plays in our compartmental SEIR model: a read-only
   *  snapshot for downstream observability/animation. */
  runTimeStep(): void {
    const s = this.state;
    if (s.observedSnapshotVersion === s.snapshotVersion) return;
    s.trace.objHistory.push(s.currentObjective());
    s.trace.vertexHistory.push(s.currentVertex());
    s.observedSnapshotVersion = s.snapshotVersion;
  }

  override hasWork(): boolean {
    return super.hasWork() || this.state.observedSnapshotVersion < this.state.snapshotVersion;
  }
}

// -----------------------------------------------------------------------------
// Build initial tableau in y-space (preprocess: shift bounds, split free
// variables, add slacks / surpluses / artificials, fix b ≥ 0, install
// phase-1 cost row).
// -----------------------------------------------------------------------------
interface Preprocessed {
  T: number[][]; basis: number[]; m: number; ncols: number;
  artificialCols: Set<number>; phase2CostRow: number[];
  n_orig: number; shifts: number[]; yIndexOfPos: number[]; freeNeg: number[];
}

function preprocess(p: LPProblem): Preprocessed {
  const n = p.c.length;
  const A_ub = p.A_ub ?? [];
  const b_ub = p.b_ub ?? [];
  const A_eq = p.A_eq ?? [];
  const b_eq = p.b_eq ?? [];
  const lb = p.lb ?? new Array(n).fill(0);
  const ub = p.ub ?? new Array(n).fill(null);

  // Shift / split: x_i = (y_pos − y_neg) + shift_i.
  const shifts: number[] = new Array(n).fill(0);
  const freeNeg: number[] = [];
  const yIndexOfPos: number[] = [];
  let yCount = 0;
  for (let i = 0; i < n; i++) {
    if (lb[i] === null) {
      yIndexOfPos.push(yCount++);
      freeNeg.push(yCount++);
      shifts[i] = 0;
    } else {
      yIndexOfPos.push(yCount++);
      freeNeg.push(-1);
      shifts[i] = lb[i] as number;
    }
  }
  const ny = yCount;

  // Convert to working sense (we maximise z = c^T y; minimisation is handled
  // by the caller flipping `sense`). Cost in y-space:
  const cY: number[] = new Array(ny).fill(0);
  for (let i = 0; i < n; i++) {
    cY[yIndexOfPos[i]] += p.c[i];
    if (freeNeg[i] >= 0) cY[freeNeg[i]] += -p.c[i];
  }

  // Build A y ≤ b' and A y = b'' lists.
  const Ay: number[][] = [];
  const by: number[] = [];
  const eqRows: boolean[] = [];
  for (let r = 0; r < A_ub.length; r++) {
    const row = new Array(ny).fill(0);
    let rhs = b_ub[r];
    for (let i = 0; i < n; i++) {
      row[yIndexOfPos[i]] += A_ub[r][i];
      if (freeNeg[i] >= 0) row[freeNeg[i]] += -A_ub[r][i];
      rhs -= A_ub[r][i] * shifts[i];
    }
    Ay.push(row); by.push(rhs); eqRows.push(false);
  }
  for (let r = 0; r < A_eq.length; r++) {
    const row = new Array(ny).fill(0);
    let rhs = b_eq[r];
    for (let i = 0; i < n; i++) {
      row[yIndexOfPos[i]] += A_eq[r][i];
      if (freeNeg[i] >= 0) row[freeNeg[i]] += -A_eq[r][i];
      rhs -= A_eq[r][i] * shifts[i];
    }
    Ay.push(row); by.push(rhs); eqRows.push(true);
  }
  // Upper bounds → extra ≤ rows.
  for (let i = 0; i < n; i++) {
    if (ub[i] !== null) {
      const row = new Array(ny).fill(0);
      row[yIndexOfPos[i]] = 1;
      if (freeNeg[i] >= 0) row[freeNeg[i]] = -1;
      Ay.push(row); by.push((ub[i] as number) - shifts[i]); eqRows.push(false);
    }
  }
  const m = Ay.length;

  // Sign-fix b ≥ 0 by flipping rows. After flip, an originally-≤ row
  // becomes ≥, which needs a surplus + artificial; an originally-= row
  // (or already-≥-equivalent flipped) needs an artificial.
  const slackSign: number[] = [];        // +1 for ≤ row, −1 for ≥ row, 0 for equality
  for (let r = 0; r < m; r++) {
    if (by[r] < 0) {
      for (let j = 0; j < ny; j++) Ay[r][j] = -Ay[r][j];
      by[r] = -by[r];
      // After flip: ≤ becomes ≥ (needs surplus + artificial); = stays =.
      slackSign.push(eqRows[r] ? 0 : -1);
    } else {
      slackSign.push(eqRows[r] ? 0 : +1);
    }
  }

  // Allocate columns for slacks/surpluses (one per non-equality row) and
  // artificials (one per ≥ or = row).
  const ny_total = ny + m;                 // y + slacks/surpluses
  const artificialCols = new Set<number>();
  let artCount = 0;
  const artCol: number[] = [];             // -1 if no artificial in this row
  for (let r = 0; r < m; r++) {
    if (slackSign[r] === +1) {
      artCol.push(-1);
    } else {
      const c = ny_total + artCount;
      artCol.push(c);
      artificialCols.add(c);
      artCount++;
    }
  }
  const totalCols = ny_total + artCount;

  // Build the tableau: m+1 rows × (totalCols+1).
  const T: number[][] = [];
  for (let r = 0; r < m; r++) {
    const row = new Array(totalCols + 1).fill(0);
    for (let j = 0; j < ny; j++) row[j] = Ay[r][j];
    if (slackSign[r] === +1) row[ny + r] = +1;
    else if (slackSign[r] === -1) row[ny + r] = -1;       // surplus
    if (artCol[r] >= 0) row[artCol[r]] = 1;
    row[totalCols] = by[r];
    T.push(row);
  }

  // Initial basis: artificial if present, else slack.
  const basis: number[] = [];
  for (let r = 0; r < m; r++) basis.push(artCol[r] >= 0 ? artCol[r] : ny + r);

  // Phase-1 cost row: minimise sum of artificials, i.e. set cost row to
  // c1[j] = +1 for artificial columns, then row-reduce so that basic
  // artificials have reduced cost 0 (subtract those rows from cost row).
  const phase1Cost = new Array(totalCols).fill(0);
  for (const c of artificialCols) phase1Cost[c] = 1;
  // Build cost row in tableau form.
  const costRow = new Array(totalCols + 1).fill(0);
  for (let j = 0; j < totalCols; j++) costRow[j] = phase1Cost[j];
  // Row-reduce: subtract row r from cost row for each artificial in basis.
  for (let r = 0; r < m; r++) {
    if (artCol[r] >= 0) {
      for (let j = 0; j <= totalCols; j++) costRow[j] -= T[r][j];
    }
  }
  T.push(costRow);

  // Phase-2 cost row: maximise c^T y = z. We maintain T[m][j] = c_B B^-1 A_j − c_j
  // (i.e. negative reduced cost in maximisation). Equivalent encoding:
  //   reduced cost r_j = T[m][j];  optimum when r_j ≥ 0 ∀ j.
  // Start with c2[j] = -c[j] for original y columns, 0 elsewhere.
  const phase2Cost = new Array(totalCols).fill(0);
  for (let j = 0; j < ny; j++) phase2Cost[j] = -cY[j];
  // No need to reduce now; we apply this row when we transition to phase 2.
  // We pad to totalCols+1 with the RHS (initial 0).
  const phase2CostRow = phase2Cost.concat([0]);

  return {
    T, basis, m, ncols: totalCols,
    artificialCols, phase2CostRow,
    n_orig: n, shifts, yIndexOfPos, freeNeg,
  };
}

// -----------------------------------------------------------------------------
// Public API.
// -----------------------------------------------------------------------------

export interface DESSimplexOptions {
  /** 'dantzig' (default; steepest reduced cost) or 'bland' (anti-cycling, first improving). */
  pivotRule?: 'dantzig' | 'bland';
  maxIter?: number;
  tol?: number;
}

export interface DESSimplexSolution extends LPSolution {
  /** Per-pivot trace usable for animation / observability. */
  trace: DESSimplexTrace;
}

/**
 * Solve an LP using the DES engine. Each pivot is one tick. The
 * algorithm walks vertex-to-vertex along the steepest improving edge of
 * the feasible polytope, exactly the way classical simplex does — but
 * the orchestration goes through the DES tick loop and four
 * stations.
 *
 * Returns the standard `LPSolution` plus a `trace` field that records
 * the objective and current vertex at every pivot, suitable for direct
 * playback through the animation plugin.
 */
export function solveLPViaDES(p: LPProblem, opts: DESSimplexOptions = {}): DESSimplexSolution {
  const t0 = Date.now();
  const tol = opts.tol ?? 1e-9;
  const maxIter = opts.maxIter ?? 5000;
  const pivotRule = opts.pivotRule ?? 'dantzig';
  // Convert min → max by flipping the objective.
  const sense = p.sense;
  const c = sense === 'max' ? p.c.slice() : p.c.map(v => -v);
  const lpWorking: LPProblem = {...p, sense: 'max', c};

  const pp = preprocess(lpWorking);

  const state = new SimplexState({
    T: pp.T, basis: pp.basis, m: pp.m, ncols: pp.ncols,
    maxIter, tol,
    phase2CostRow: pp.phase2CostRow,
    artificialCols: pp.artificialCols,
    n_orig: pp.n_orig, shifts: pp.shifts, yIndexOfPos: pp.yIndexOfPos, freeNeg: pp.freeNeg,
    sense,
    pivotRule,
  });
  // No artificials → skip phase 1 entirely; install phase-2 cost row,
  // row-reduced against any non-zero basic-variable reduced cost.
  if (state.artificialCols.size === 0) {
    state.T[state.m] = state.phase2CostRow.slice();
    for (let r = 0; r < state.m; r++) {
      const cb = state.T[state.m][state.basis[r]];
      if (cb !== 0) {
        for (let j = 0; j <= state.ncols; j++) {
          state.T[state.m][j] -= cb * state.T[r][j];
        }
      }
    }
    state.phase = 2;
  }

  // Build the simplex stations. Same lifecycle as our SEIR / FactMachine stations:
  // the shared DES runner owns the fixed tick order, and each stationary role
  // mutates the shared tableau state only through its own runTimeStep hook.
  const observer = new ObserverStation(state);
  const phase    = new PhaseTransitionStation(state);
  const entering = new EnteringStation(state);
  const leaving  = new LeavingStation(state);
  const pivot    = new PivotStation(state);
  runIterativeDES([observer, phase, entering, leaving, pivot], {
    shuffle: false,
    maxTicks: maxIter + state.m + 10,
    runValidators: false,
  });

  const solStatus: LPStatus =
    (state.status === 'optimal' || state.status === 'infeasible'
     || state.status === 'unbounded' || state.status === 'iter-limit')
      ? state.status : 'numerical-error';

  if (solStatus !== 'optimal') {
    return {
      status: solStatus, x: [], objective: NaN,
      solver: `des-simplex(${pivotRule})`,
      elapsedMs: Date.now() - t0,
      iters: state.iters,
      trace: state.trace,
      message: solStatus === 'unbounded'
        ? 'unbounded direction at LeavingStation'
        : (solStatus === 'infeasible' ? 'phase-1 sum of artificials > 0' : ''),
    };
  }
  const x = state.currentVertex();
  let obj = 0;
  for (let i = 0; i < p.c.length; i++) obj += p.c[i] * x[i];
  return {
    status: 'optimal',
    x, objective: obj,
    solver: `des-simplex(${pivotRule})`,
    elapsedMs: Date.now() - t0,
    iters: state.iters,
    trace: state.trace,
    message: `DES simplex: ${state.trace.pivotHistory.length} pivots across ${state.phase === 2 ? 'two phases' : 'phase 2 only'}`,
  };
}
