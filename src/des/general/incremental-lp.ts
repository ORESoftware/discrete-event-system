'use strict';

// =============================================================================
// general/incremental-lp.ts — INCREMENTAL Linear Programming solver
// expressed as a discrete-event SYSTEM (the "S" in DES taken in its
// broader sense — this isn't a simulation, it's an iterative algorithm
// running on the same station/movable substrate). Each pivot is one
// tick. The solver maintains a warm-startable basis across modifications:
//
//    • Add constraint  a·x ≤ b
//    • Remove constraint by index
//    • Change the objective vector c
//    • Add a new variable (with its column and objective coefficient)
//    • Remove a variable
//
// The recovery method after each modification uses the *parametric simplex*
// idea: every modification breaks at most ONE of the two simplex
// invariants (primal feasibility x_B ≥ 0, dual feasibility c̄_N ≤ 0 for max),
// and we restart the right flavour of simplex (primal or dual) to restore
// the broken invariant while preserving the other.
//
// MAPPING TO THE DES FRAMEWORK
// ────────────────────────────
//   Stations
//     LPTableauStation  — owns the dense tableau + basis. Mutated by
//                         events arriving from the EventQueueStation.
//     EventQueueStation — heap of pending modification events keyed by
//                         tick. Drains into LPTableauStation each tick.
//     PivotStation      — runs ONE simplex pivot per tick (primal if dual
//                         feasibility holds; dual if primal feasibility
//                         holds; nothing if optimal/infeasible/unbounded).
//     CensusStation     — captures per-tick (basis, x, z, mode, last
//                         event applied) for animation.
//   Movables
//     ConstraintAddEvent, ConstraintRemoveEvent,
//     ObjectiveChangeEvent, VariableAddEvent, VariableRemoveEvent
//                       — flow from EventQueueStation into LPTableauStation.
//     PivotEvent        — emitted by PivotStation, recorded for the trace.
//
// TABLEAU CONVENTION (max-form throughout)
// ────────────────────────────────────────
//   columns 0 … numStruct-1                : structural variables x_1..x_n
//   columns numStruct … numStruct+m-1      : slack variables       s_1..s_m
//   columns numStruct+m                    : RHS column             (last)
//   row 0                                  : z-row, holding reduced costs
//                                            (z_j − c_j); rightmost cell = z
//   rows 1..m                              : constraint rows
//   basis[i]                               : column index basic in row (i+1)
//
//   Initial basis = all slack columns. So x = 0, z = 0, slacks = b.
//   This is feasible iff every b_i ≥ 0; we accept that requirement at
//   construction time (Phase-1 / Big-M is out of scope for this module).
//
//   For min-LPs, callers should negate `c` and flip the result's z; the
//   API stores `senseSign` to do this transparently.
// =============================================================================

const EPS = 1e-9;

// -----------------------------------------------------------------------------
// MOVABLES — modification events flowing into the LP tableau station.
// -----------------------------------------------------------------------------

export type LPEvent =
  | {tick: number; kind: 'add-constraint';    coefs: number[]; rhs: number; name?: string}
  | {tick: number; kind: 'remove-constraint'; index: number;                 name?: string}
  | {tick: number; kind: 'change-objective';  newC: number[];                name?: string}
  | {tick: number; kind: 'add-variable';      column: number[]; cNew: number; name?: string}
  | {tick: number; kind: 'remove-variable';   structIndex: number;            name?: string};

export interface PivotEvent {
  tick: number;
  mode: 'primal' | 'dual' | 'optimal' | 'infeasible' | 'unbounded' | 'idle';
  entering?: number;     // column index
  leaving?: number;      // row index (1..m)
  enteringName?: string;
  leavingName?: string;
}

// -----------------------------------------------------------------------------
// PROBLEM SETUP
// -----------------------------------------------------------------------------

export interface IncrementalLPInit {
  sense: 'max' | 'min';
  /** Initial objective. Length numStruct. */
  c: number[];
  /** Initial constraints `A·x ≤ b`. Each row length numStruct. */
  A: number[][];
  /** Initial RHS. Length = A.length. Must be non-negative for warm-start. */
  b: number[];
  /** Optional names. */
  varNames?: string[];
  conNames?: string[];
}

export interface LPSnapshot {
  tick: number;
  numStruct: number;
  numConstraints: number;
  basis: number[];                  // length m, column indices
  x: number[];                      // length numStruct
  slacks: number[];                 // length m (slack values)
  z: number;                        // current objective value (in original sense)
  reducedCosts: number[];           // length numStruct + m (row 0 excluding rhs)
  rhs: number[];                    // length m (current basic feasible values)
  primalFeasible: boolean;
  dualFeasible: boolean;
  isOptimal: boolean;
  varNames: string[];
  conNames: string[];
  /** The event applied at this tick (if any). */
  appliedEvent?: LPEvent;
  /** The pivot that fired at this tick (if any). */
  pivot?: PivotEvent;
  /** Mode the solver is currently in. */
  mode: 'primal' | 'dual' | 'optimal' | 'infeasible' | 'unbounded';
}

// -----------------------------------------------------------------------------
// SOLVER CLASS
// -----------------------------------------------------------------------------

export class IncrementalLP {
  /** Tableau: shape (m+1) × (numStruct + m + 1). The last column is rhs;
   *  row 0 is the z-row (reduced costs in the variable cells, current z
   *  in the rhs cell). */
  tab: number[][] = [];
  basis: number[] = [];        // length m, each entry is a column index
  numStruct = 0;
  /** +1 for max, -1 for min. We always solve internally as a max LP by
   *  negating c at construction; getZ() flips the sign back. */
  senseSign: 1 | -1 = 1;
  varNames: string[] = [];
  conNames: string[] = [];
  /** Detected status. */
  status: 'primal' | 'dual' | 'optimal' | 'infeasible' | 'unbounded' = 'primal';
  /** Current tick counter; advances by exactly one each `runOneTick`. */
  tick = 0;

  // ---------------------------------------------------------------------
  constructor(init: IncrementalLPInit) {
    const n = init.c.length;
    const m = init.A.length;
    if (m !== init.b.length) throw new Error(`A.length (${m}) ≠ b.length (${init.b.length})`);
    for (let i = 0; i < m; i++) {
      if (init.A[i].length !== n) throw new Error(`A[${i}].length (${init.A[i].length}) ≠ c.length (${n})`);
      if (init.b[i] < -EPS) throw new Error(`b[${i}] = ${init.b[i]} < 0; warm-start requires non-negative RHS (use Phase-1 elsewhere)`);
    }
    this.numStruct = n;
    this.senseSign = init.sense === 'max' ? 1 : -1;
    this.varNames = init.varNames?.slice() ?? [];
    while (this.varNames.length < n) this.varNames.push(`x${this.varNames.length + 1}`);
    this.conNames = init.conNames?.slice() ?? [];
    while (this.conNames.length < m) this.conNames.push(`c${this.conNames.length + 1}`);
    // Build the tableau. n_struct + m slacks + 1 rhs.
    const totalCols = n + m + 1;
    this.tab = [];
    // Row 0 (z-row). Reduced cost initial: z_j − c_j = −c_j for non-basic structural;
    // 0 for basic slacks; 0 for rhs (z = 0).
    const z = new Array(totalCols).fill(0);
    for (let j = 0; j < n; j++) z[j] = -this.senseSign * init.c[j];
    this.tab.push(z);
    // Constraint rows: [A_i | I row | b_i]
    for (let i = 0; i < m; i++) {
      const row = new Array(totalCols).fill(0);
      for (let j = 0; j < n; j++) row[j] = init.A[i][j];
      row[n + i] = 1;                // slack column
      row[totalCols - 1] = init.b[i];
      this.tab.push(row);
    }
    this.basis = [];
    for (let i = 0; i < m; i++) this.basis.push(n + i);    // initial basis = slacks
    this.refreshStatus();
  }

  // ---------------------------------------------------------------------
  // CORE PIVOT MACHINERY
  // ---------------------------------------------------------------------

  /** Run primal pivot: most-negative-reduced-cost rule for entering, min-ratio
   *  for leaving. Returns the pivot info or a terminal status. */
  private primalPivot(): PivotEvent {
    const m = this.tab.length - 1;
    const totalCols = this.tab[0].length;
    const rhsCol = totalCols - 1;
    // Entering: most negative reduced cost (Dantzig's rule).
    let entering = -1;
    let mostNeg = -EPS;
    for (let j = 0; j < rhsCol; j++) {
      if (this.tab[0][j] < mostNeg) { mostNeg = this.tab[0][j]; entering = j; }
    }
    if (entering === -1) return {tick: this.tick, mode: 'optimal'};
    // Leaving: min ratio.
    let leaving = -1;
    let minRatio = Infinity;
    for (let i = 1; i <= m; i++) {
      if (this.tab[i][entering] > EPS) {
        const ratio = this.tab[i][rhsCol] / this.tab[i][entering];
        if (ratio < minRatio - EPS || (Math.abs(ratio - minRatio) < EPS && (leaving === -1 || this.basis[i - 1] < this.basis[leaving - 1]))) {
          minRatio = ratio; leaving = i;
        }
      }
    }
    if (leaving === -1) return {tick: this.tick, mode: 'unbounded'};
    const leavingCol = this.basis[leaving - 1];     // capture BEFORE the pivot updates the basis
    this.doPivot(leaving, entering);
    this.basis[leaving - 1] = entering;
    return {
      tick: this.tick, mode: 'primal',
      entering, leaving,
      enteringName: this.colName(entering),
      leavingName: this.colName(leavingCol),
    };
  }

  /** Run dual pivot: most-negative-RHS rule for leaving, then ratio test
   *  on the z-row for entering. Maintains dual feasibility. */
  private dualPivot(): PivotEvent {
    const m = this.tab.length - 1;
    const totalCols = this.tab[0].length;
    const rhsCol = totalCols - 1;
    // Leaving: most negative basic value.
    let leaving = -1;
    let mostNeg = -EPS;
    for (let i = 1; i <= m; i++) {
      if (this.tab[i][rhsCol] < mostNeg) { mostNeg = this.tab[i][rhsCol]; leaving = i; }
    }
    if (leaving === -1) return {tick: this.tick, mode: 'optimal'};
    // Entering: argmin over j with tab[leaving][j] < 0 of (tab[0][j] / -tab[leaving][j]).
    let entering = -1;
    let minRatio = Infinity;
    for (let j = 0; j < rhsCol; j++) {
      if (this.tab[leaving][j] < -EPS) {
        const ratio = this.tab[0][j] / -this.tab[leaving][j];
        if (ratio < minRatio - EPS) { minRatio = ratio; entering = j; }
      }
    }
    if (entering === -1) return {tick: this.tick, mode: 'infeasible'};
    const leavingCol = this.basis[leaving - 1];     // capture BEFORE the pivot updates the basis
    this.doPivot(leaving, entering);
    this.basis[leaving - 1] = entering;
    return {
      tick: this.tick, mode: 'dual',
      entering, leaving,
      enteringName: this.colName(entering),
      leavingName: this.colName(leavingCol),
    };
  }

  /** Gauss–Jordan elimination on (pivot row, pivot column). */
  private doPivot(r: number, c: number): void {
    const pivotVal = this.tab[r][c];
    if (Math.abs(pivotVal) < EPS) throw new Error(`degenerate pivot ${pivotVal} at row ${r} col ${c}`);
    for (let j = 0; j < this.tab[r].length; j++) this.tab[r][j] /= pivotVal;
    for (let i = 0; i < this.tab.length; i++) {
      if (i === r) continue;
      const factor = this.tab[i][c];
      if (Math.abs(factor) < EPS) continue;
      for (let j = 0; j < this.tab[i].length; j++) {
        this.tab[i][j] -= factor * this.tab[r][j];
      }
    }
  }

  /** One tick of the DES: apply one pivot if not optimal. Returns the
   *  pivot event recorded, or an "idle" event if optimal/infeasible/unbounded. */
  step(): PivotEvent {
    this.tick++;
    if (this.status === 'optimal' || this.status === 'infeasible' || this.status === 'unbounded') {
      return {tick: this.tick, mode: 'idle'};
    }
    let pivot: PivotEvent;
    if (this.status === 'dual') pivot = this.dualPivot();
    else                        pivot = this.primalPivot();
    // Update solver mode.
    if (pivot.mode === 'optimal' || pivot.mode === 'infeasible' || pivot.mode === 'unbounded') {
      this.status = pivot.mode;
    } else {
      this.refreshStatus();
    }
    return pivot;
  }

  /** Recompute the primal/dual feasibility flags and choose the appropriate mode. */
  private refreshStatus(): void {
    const m = this.tab.length - 1;
    const totalCols = this.tab[0].length;
    const rhsCol = totalCols - 1;
    let primalFeas = true, dualFeas = true;
    for (let i = 1; i <= m; i++) if (this.tab[i][rhsCol] < -EPS) { primalFeas = false; break; }
    for (let j = 0; j < rhsCol; j++) if (this.tab[0][j] < -EPS) { dualFeas = false; break; }
    if (primalFeas && dualFeas)           this.status = 'optimal';
    else if (primalFeas && !dualFeas)     this.status = 'primal';
    else if (!primalFeas && dualFeas)     this.status = 'dual';
    else                                  this.status = 'primal';   // both broken: bias to primal
  }

  // ---------------------------------------------------------------------
  // MODIFICATION OPERATIONS — each one is the body of a station that
  // consumes a movable arriving from the EventQueueStation.
  // ---------------------------------------------------------------------

  /** Add `coefs · x ≤ rhs`. After the call the system is dual-feasible;
   *  if the new slack is negative (i.e. primal infeasible) the next
   *  step()'s will run dual simplex. */
  applyAddConstraint(coefs: number[], rhs: number, name?: string): void {
    if (coefs.length !== this.numStruct) throw new Error(`coefs.length=${coefs.length}, numStruct=${this.numStruct}`);
    const m = this.tab.length - 1;
    const oldTotal = this.tab[0].length;
    // 1. Insert a new slack column at index numStruct + m (just before rhs).
    const newCol = this.numStruct + m;
    for (let i = 0; i < this.tab.length; i++) {
      this.tab[i].splice(newCol, 0, 0);
    }
    // 2. Append a new constraint row in standard form.
    const totalCols = oldTotal + 1;
    const newRow = new Array(totalCols).fill(0);
    for (let j = 0; j < this.numStruct; j++) newRow[j] = coefs[j];
    newRow[newCol] = 1;                    // its slack
    newRow[totalCols - 1] = rhs;
    // 3. Reduce the new row against all currently basic columns to maintain
    //    the basic feasible structure: subtract `newRow[basis[k]] · row k+1`
    //    from newRow so the new row has zeros under every existing basic column.
    for (let k = 0; k < this.basis.length; k++) {
      const bj = this.basis[k];
      const factor = newRow[bj];
      if (Math.abs(factor) < EPS) continue;
      for (let j = 0; j < totalCols; j++) newRow[j] -= factor * this.tab[k + 1][j];
    }
    this.tab.push(newRow);
    this.basis.push(newCol);                                     // new slack is basic
    this.conNames.push(name ?? `c${this.conNames.length + 1}`);
    this.refreshStatus();
  }

  /** Remove the constraint at `index` (0-based).
   *
   *  In a tableau that has been pivoted any number of times, the row that
   *  "encodes" constraint i is NOT row i+1 — it's the row whose basic
   *  variable is slack_i (or any row containing the only contribution
   *  from slack_i). We therefore:
   *    1. Locate `r*`, the row where slack_i is currently basic. If
   *       slack_i is non-basic, pivot it in at any row with a non-zero
   *       coefficient in column slack_i (this MAY break primal feasibility,
   *       which the next dual-simplex step will repair).
   *    2. Drop row `r*` and column slack_i. Because slack_i was basic at
   *       r*, the column had a 1 at r* and 0 elsewhere, so dropping it
   *       does not perturb the remaining tableau. */
  applyRemoveConstraint(index: number): void {
    if (index < 0 || index >= this.tab.length - 1) throw new Error(`remove-constraint: index ${index} out of range`);
    const slackCol = this.numStruct + index;
    const m = this.tab.length - 1;
    // 1a. Find the row where slack_i is currently basic.
    let rStar = -1;
    for (let r = 1; r <= m; r++) if (this.basis[r - 1] === slackCol) { rStar = r; break; }
    // 1b. If slack_i is non-basic, force it into the basis.
    if (rStar === -1) {
      for (let r = 1; r <= m; r++) {
        if (Math.abs(this.tab[r][slackCol]) > EPS) {
          this.doPivot(r, slackCol);
          this.basis[r - 1] = slackCol;
          rStar = r;
          break;
        }
      }
      if (rStar === -1) {
        // Column is identically zero in every row — constraint was redundant.
        // Pick row corresponding to `index+1` arbitrarily.
        rStar = Math.min(index + 1, m);
      }
    }
    // 2. Drop the row and the slack column.
    this.tab.splice(rStar, 1);
    for (let i = 0; i < this.tab.length; i++) this.tab[i].splice(slackCol, 1);
    this.basis.splice(rStar - 1, 1);
    for (let k = 0; k < this.basis.length; k++) if (this.basis[k] > slackCol) this.basis[k] -= 1;
    this.conNames.splice(index, 1);
    this.refreshStatus();
  }

  /** Replace the objective with `newC`. The dual feasibility may break; the
   *  next step()'s will run primal simplex. */
  applyChangeObjective(newC: number[]): void {
    if (newC.length !== this.numStruct) throw new Error(`change-objective: length ${newC.length}, expected ${this.numStruct}`);
    const m = this.tab.length - 1;
    const totalCols = this.tab[0].length;
    const rhsCol = totalCols - 1;
    // Reset the z-row. New row 0 = -senseSign * newC for structural cols, 0 for slack cols, 0 for rhs.
    for (let j = 0; j < rhsCol; j++) {
      this.tab[0][j] = (j < this.numStruct) ? -this.senseSign * newC[j] : 0;
    }
    this.tab[0][rhsCol] = 0;
    // Now re-zero the z-row beneath every basic column by row-reduction.
    for (let k = 0; k < this.basis.length; k++) {
      const bj = this.basis[k];
      const factor = this.tab[0][bj];
      if (Math.abs(factor) < EPS) continue;
      for (let j = 0; j < totalCols; j++) this.tab[0][j] -= factor * this.tab[k + 1][j];
    }
    this.refreshStatus();
  }

  /** Append a new structural variable with column `column` (length m, in
   *  ORIGINAL untransformed standard-form coordinates) and objective
   *  coefficient `cNew`.
   *
   *  CRITICAL: the inserted column must be expressed in the CURRENT
   *  tableau's coordinate system, which is `B^{-1}·column`, not the raw
   *  `column`. The slack columns at positions numStruct..numStruct+m-1
   *  jointly store `B^{-1}` (because they started as the identity at init
   *  and have been transformed by every pivot since), so we can compute
   *  `B^{-1}·column` as a linear combination over those slack columns.
   *
   *  Similarly the row-0 entry is `c_B^T B^{-1} a_new − c_new`, which
   *  using the same trick equals `Σ_k column[k] · tab[0][slack_k] − senseSign·cNew`. */
  applyAddVariable(column: number[], cNew: number, name?: string): void {
    const m = this.tab.length - 1;
    if (column.length !== m) throw new Error(`add-variable: column length ${column.length}, expected ${m}`);
    // Compute the transformed column = B^{-1} · column, plus the row-0 reduced cost.
    const transformed = new Array(m + 1).fill(0);
    for (let i = 1; i <= m; i++) {
      let v = 0;
      for (let k = 0; k < m; k++) v += column[k] * this.tab[i][this.numStruct + k];
      transformed[i] = v;
    }
    let zRowEntry = -this.senseSign * cNew;     // start with −c_new term
    for (let k = 0; k < m; k++) zRowEntry += column[k] * this.tab[0][this.numStruct + k];
    transformed[0] = zRowEntry;
    // Insert the column at position `numStruct` (just after the existing structural
    // variables, before the slack block).
    const insertAt = this.numStruct;
    for (let i = 0; i < this.tab.length; i++) this.tab[i].splice(insertAt, 0, transformed[i]);
    this.numStruct += 1;
    for (let k = 0; k < this.basis.length; k++) if (this.basis[k] >= insertAt) this.basis[k] += 1;
    this.varNames.splice(insertAt, 0, name ?? `x${insertAt + 1}`);
    this.refreshStatus();
  }

  /** Remove a structural variable. If non-basic, drop its column directly;
   *  if basic, force-pivot it out first (primal pivot in its current basic row),
   *  then drop. */
  applyRemoveVariable(structIndex: number): void {
    if (structIndex < 0 || structIndex >= this.numStruct) throw new Error(`remove-variable: index ${structIndex} out of range`);
    const dropCol = structIndex;
    // Is it basic somewhere?
    let basicRow = -1;
    for (let k = 0; k < this.basis.length; k++) if (this.basis[k] === dropCol) { basicRow = k + 1; break; }
    if (basicRow !== -1) {
      // Find any non-basic column with a non-zero entry in `basicRow` (preferring slack)
      // to force a pivot that knocks dropCol out of the basis.
      let entering = -1;
      const totalCols = this.tab[0].length;
      const rhsCol = totalCols - 1;
      for (let j = 0; j < rhsCol; j++) {
        if (j === dropCol) continue;
        if (this.basis.includes(j)) continue;
        if (Math.abs(this.tab[basicRow][j]) > EPS) { entering = j; break; }
      }
      if (entering !== -1) {
        this.doPivot(basicRow, entering);
        this.basis[basicRow - 1] = entering;
      } else {
        // If no pivot is possible, the variable is degenerate; we drop the row entirely.
        this.tab.splice(basicRow, 1);
        this.basis.splice(basicRow - 1, 1);
      }
    }
    // Drop the column from every row.
    for (let i = 0; i < this.tab.length; i++) this.tab[i].splice(dropCol, 1);
    this.numStruct -= 1;
    for (let k = 0; k < this.basis.length; k++) if (this.basis[k] > dropCol) this.basis[k] -= 1;
    this.varNames.splice(dropCol, 1);
    this.refreshStatus();
  }

  /** Convenience: apply an LPEvent. */
  applyEvent(e: LPEvent): void {
    switch (e.kind) {
      case 'add-constraint':    this.applyAddConstraint(e.coefs, e.rhs, e.name); break;
      case 'remove-constraint': this.applyRemoveConstraint(e.index); break;
      case 'change-objective':  this.applyChangeObjective(e.newC); break;
      case 'add-variable':      this.applyAddVariable(e.column, e.cNew, e.name); break;
      case 'remove-variable':   this.applyRemoveVariable(e.structIndex); break;
    }
  }

  /** Pivot until optimal/infeasible/unbounded. Returns the pivot trace. */
  solveToOptimum(maxIters = 1000): PivotEvent[] {
    const trace: PivotEvent[] = [];
    for (let k = 0; k < maxIters; k++) {
      const before = this.status;
      const ev = this.step();
      trace.push(ev);
      if (this.status === 'optimal' || this.status === 'infeasible' || this.status === 'unbounded') return trace;
      if (ev.mode === 'idle') return trace;
    }
    return trace;
  }

  // ---------------------------------------------------------------------
  // INSPECTION / SNAPSHOT
  // ---------------------------------------------------------------------

  /** Resolved x for the structural variables. */
  getX(): number[] {
    const x = new Array(this.numStruct).fill(0);
    const m = this.tab.length - 1;
    const rhsCol = this.tab[0].length - 1;
    for (let k = 0; k < this.basis.length; k++) {
      const j = this.basis[k];
      if (j < this.numStruct) x[j] = this.tab[k + 1][rhsCol];
    }
    return x;
  }
  /** Slack values. */
  getSlacks(): number[] {
    const m = this.tab.length - 1;
    const s = new Array(m).fill(0);
    const rhsCol = this.tab[0].length - 1;
    for (let k = 0; k < this.basis.length; k++) {
      const j = this.basis[k];
      if (j >= this.numStruct) s[j - this.numStruct] = this.tab[k + 1][rhsCol];
    }
    return s;
  }
  /** Current objective value, in the original sense. */
  getZ(): number {
    const rhsCol = this.tab[0].length - 1;
    return this.senseSign * this.tab[0][rhsCol];
  }
  /** Reduced-cost vector (length numStruct + m). */
  getReducedCosts(): number[] {
    const rhsCol = this.tab[0].length - 1;
    return this.tab[0].slice(0, rhsCol);
  }
  colName(j: number): string {
    return j < this.numStruct ? this.varNames[j] : this.conNames[j - this.numStruct] + '_slack';
  }
  snapshot(appliedEvent?: LPEvent, pivot?: PivotEvent): LPSnapshot {
    const m = this.tab.length - 1;
    const rhsCol = this.tab[0].length - 1;
    return {
      tick: this.tick,
      numStruct: this.numStruct,
      numConstraints: m,
      basis: this.basis.slice(),
      x: this.getX(),
      slacks: this.getSlacks(),
      z: this.getZ(),
      reducedCosts: this.getReducedCosts(),
      rhs: this.tab.slice(1).map(r => r[rhsCol]),
      primalFeasible: this.tab.slice(1).every(r => r[rhsCol] >= -EPS),
      dualFeasible:   this.tab[0].slice(0, rhsCol).every(v => v >= -EPS),
      isOptimal: this.status === 'optimal',
      varNames: this.varNames.slice(),
      conNames: this.conNames.slice(),
      appliedEvent, pivot,
      mode: this.status,
    };
  }
}
