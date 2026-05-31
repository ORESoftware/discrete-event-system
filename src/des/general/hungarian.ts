// RUST MIGRATION: target module src/des/general/hungarian.rs.
// RUST MIGRATION: AssignmentDirection becomes an enum and AssignmentResult becomes a serde struct.
// RUST MIGRATION: hungarian is a pure solver and should stay a free function returning Result<AssignmentResult, Error> for malformed matrices.
// RUST MIGRATION: Use Vec<Vec<f64>> for cost matrices and keep rectangular padding/dual arrays explicit.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/hungarian.rs  (module des::general::hungarian)
// 1:1 file move. Hungarian (Jonker-Volgenant) O(n^3) bipartite assignment solver.
//
// Declarations → Rust:
//   type AssignmentDirection = 'min' | 'max'  -> enum AssignmentDirection { Min, Max }
//   interface AssignmentResult                -> struct AssignmentResult (#[derive(Clone)])
//   fn hungarian(matrix, dir)                 -> PureTransform<AssignmentInput, AssignmentResult> (vanilla algorithm)
//
// Conversion notes (file-specific):
//   - `number[][]` cost matrix -> `Vec<Vec<f64>>` (or an (n,m) matrix type); pure & deterministic.
//   - `-1` sentinel in rows/cols (unmatched) -> `Option<usize>` (or keep `i64` if mirroring exactly).
// =============================================================================

// =============================================================================
// general/hungarian.ts — Hungarian algorithm for square / rectangular
// bipartite assignment.
//
// Solves   max  Σ x_{i,j} · w_{i,j}        (or min for cost matrices)
//          s.t. Σ_j x_{i,j} = 1   ∀ i
//               Σ_i x_{i,j} = 1   ∀ j
//               x_{i,j} ∈ {0, 1}
//
// in O(n^3) where n = max(rows, cols). For our 7v12 player→position
// problem n ≤ 12, so this is microseconds — but it's the inner solver
// invoked at every period boundary by greedy-Hungarian, the LP-relaxation
// rounder, and the MDP-VI reward function, so it has to be exact.
//
// Implementation: O(n^3) Jonker–Volgenant style "shortest augmenting path"
// matrix variant. Correctness verified bit-exactly against
// scipy.optimize.linear_sum_assignment in the test suite.
// =============================================================================

export type AssignmentDirection = 'min' | 'max';

export interface AssignmentResult {
  /** rows[i] = j  iff agent i is assigned to job j; -1 if rectangular and unmatched. */
  rows: number[];
  /** cols[j] = i  iff job j is filled by agent i; -1 if unfilled. */
  cols: number[];
  /** Total cost (or weight) of the optimal assignment. */
  total: number;
}

/**
 * Solve a (possibly rectangular) bipartite assignment.
 *
 * @param matrix  rows × cols numeric matrix (must be a rectangular array)
 * @param dir     'min' to minimise total cost (default), 'max' to maximise total weight
 */
export function hungarian(matrix: number[][], dir: AssignmentDirection = 'min'): AssignmentResult {
  if (matrix.length === 0) return {rows: [], cols: [], total: 0};
  const nRows = matrix.length;
  const nCols = matrix[0].length;
  const n = Math.max(nRows, nCols);
  const sign = dir === 'max' ? -1 : 1;
  // Pad to square matrix with a constant fill (so dummy rows/cols don't
  // distort the optimal pairing). For 'max', use 0; for 'min' use a large
  // finite number. We pick fills so the dummy rows/cols don't dominate
  // any real assignment.
  let fill = 0;
  if (dir === 'min') {
    let mx = matrix[0][0];
    for (const r of matrix) for (const v of r) if (v > mx) mx = v;
    fill = mx + 1;     // dummy entries cost more than any real one
  } else {
    let mn = matrix[0][0];
    for (const r of matrix) for (const v of r) if (v < mn) mn = v;
    fill = mn - 1;     // dummy entries weigh less than any real one
  }
  const a: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = new Array(n);
    for (let j = 0; j < n; j++) {
      const v = i < nRows && j < nCols ? matrix[i][j] : fill;
      row[j] = sign * v;
    }
    a.push(row);
  }
  // Jonker–Volgenant via shortest-path "u, v, p" with column potentials.
  // u[i] = row potential, v[j] = column potential, p[j] = row matched to col j.
  const INF = Infinity;
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0);
  const way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(INF);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (!used[j]) {
          const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  // p[j] = row assigned to column j (1-indexed). Build rows[] and cols[].
  const rows = new Array(nRows).fill(-1);
  const cols = new Array(nCols).fill(-1);
  let total = 0;
  for (let j = 1; j <= n; j++) {
    const i = p[j];
    if (i === 0) continue;
    const ri = i - 1;
    const cj = j - 1;
    if (ri < nRows && cj < nCols) {
      rows[ri] = cj;
      cols[cj] = ri;
      total += matrix[ri][cj];
    }
  }
  return {rows, cols, total};
}
