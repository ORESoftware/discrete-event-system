'use strict';

// =============================================================================
// control-systems/linear-algebra.ts — small dense linear-algebra toolkit used
// by the control-systems family (wind-MPPT, DC-motor, observability/
// controllability evaluator).
//
// DESIGN CONSTRAINT
// ─────────────────
//   Everything in the control-systems family is expressed as CLASSES with
//   METHODS — no free/standalone functions. This file therefore exposes a
//   single `LinAlg` class whose STATIC methods implement matrix/vector
//   arithmetic, plus a `MatrixRank` class that owns the numeric rank
//   computation (Gaussian elimination with partial pivoting) as state-
//   carrying instance methods.
//
//   `Mat` is row-major `number[][]`, `Vec` is `number[]`.
// =============================================================================

export type Vec = number[];
export type Mat = number[][];

/** Stateless matrix/vector arithmetic. All entry points are static methods so
 *  the family never reaches for a bare function. */
export class LinAlg {
  /** n×n identity. */
  static identity(n: number): Mat {
    const out: Mat = Array.from({length: n}, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) out[i][i] = 1;
    return out;
  }

  /** r×c zero matrix. */
  static zeros(r: number, c: number): Mat {
    return Array.from({length: r}, () => new Array<number>(c).fill(0));
  }

  /** Deep copy. */
  static copy(M: Mat): Mat {
    return M.map(row => row.slice());
  }

  /** Number of rows. */
  static rows(M: Mat): number {
    return M.length;
  }

  /** Number of columns (0 for an empty matrix). */
  static cols(M: Mat): number {
    return M.length === 0 ? 0 : M[0].length;
  }

  /** Transpose. */
  static transpose(M: Mat): Mat {
    const r = LinAlg.rows(M);
    const c = LinAlg.cols(M);
    const out: Mat = Array.from({length: c}, () => new Array<number>(r).fill(0));
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = M[i][j];
    return out;
  }

  /** Matrix product A·B. Throws on a shape mismatch. */
  static matMul(A: Mat, B: Mat): Mat {
    const ra = LinAlg.rows(A);
    const ca = LinAlg.cols(A);
    const rb = LinAlg.rows(B);
    const cb = LinAlg.cols(B);
    if (ca !== rb) throw new Error(`LinAlg.matMul: shape mismatch ${ra}x${ca} · ${rb}x${cb}`);
    const out: Mat = LinAlg.zeros(ra, cb);
    for (let i = 0; i < ra; i++) {
      for (let k = 0; k < ca; k++) {
        const a = A[i][k];
        if (a === 0) continue;
        for (let j = 0; j < cb; j++) out[i][j] += a * B[k][j];
      }
    }
    return out;
  }

  /** Matrix·vector M·v. */
  static matVec(M: Mat, v: Vec): Vec {
    const r = LinAlg.rows(M);
    const c = LinAlg.cols(M);
    if (c !== v.length) throw new Error(`LinAlg.matVec: shape mismatch ${r}x${c} · ${v.length}`);
    const out = new Array<number>(r).fill(0);
    for (let i = 0; i < r; i++) {
      let acc = 0;
      for (let j = 0; j < c; j++) acc += M[i][j] * v[j];
      out[i] = acc;
    }
    return out;
  }

  /** A + B. */
  static add(A: Mat, B: Mat): Mat {
    const r = LinAlg.rows(A);
    const c = LinAlg.cols(A);
    const out = LinAlg.zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] + B[i][j];
    return out;
  }

  /** A − B. */
  static sub(A: Mat, B: Mat): Mat {
    const r = LinAlg.rows(A);
    const c = LinAlg.cols(A);
    const out = LinAlg.zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] - B[i][j];
    return out;
  }

  /** s·M. */
  static scale(M: Mat, s: number): Mat {
    return M.map(row => row.map(x => x * s));
  }

  /** A^k via repeated multiplication (k ≥ 0; k = 0 → identity). */
  static power(A: Mat, k: number): Mat {
    const n = LinAlg.rows(A);
    let acc = LinAlg.identity(n);
    for (let i = 0; i < k; i++) acc = LinAlg.matMul(acc, A);
    return acc;
  }

  /** Horizontal block concatenation [A | B | …]: same row count, columns
   *  summed. Used to assemble the controllability matrix. */
  static hstack(blocks: readonly Mat[]): Mat {
    if (blocks.length === 0) throw new Error('LinAlg.hstack: no blocks');
    const r = LinAlg.rows(blocks[0]);
    for (const b of blocks) {
      if (LinAlg.rows(b) !== r) throw new Error('LinAlg.hstack: row-count mismatch');
    }
    const out: Mat = Array.from({length: r}, () => [] as number[]);
    for (const b of blocks) for (let i = 0; i < r; i++) out[i].push(...b[i]);
    return out;
  }

  /** Vertical block concatenation [A; B; …]: same column count, rows
   *  appended. Used to assemble the observability matrix. */
  static vstack(blocks: readonly Mat[]): Mat {
    if (blocks.length === 0) throw new Error('LinAlg.vstack: no blocks');
    const c = LinAlg.cols(blocks[0]);
    const out: Mat = [];
    for (const b of blocks) {
      if (LinAlg.cols(b) !== c) throw new Error('LinAlg.vstack: column-count mismatch');
      for (const row of b) out.push(row.slice());
    }
    return out;
  }

  /** Largest absolute entry — used to scale the rank tolerance. */
  static maxAbs(M: Mat): number {
    let m = 0;
    for (const row of M) for (const x of row) {
      const a = Math.abs(x);
      if (a > m) m = a;
    }
    return m;
  }

  /** Numeric rank via `MatrixRank`. `tol` defaults to a scale-aware
   *  threshold derived from the matrix size and magnitude. */
  static rank(M: Mat, tol?: number): number {
    return new MatrixRank(M, tol).rank();
  }
}

/** Dense matrix inverse via Gauss–Jordan elimination with partial pivoting.
 *  Kept as a class so the elimination scratch state never leaks into a bare
 *  helper function. Throws if the matrix is singular to the given tolerance. */
export class MatrixInverse {
  private readonly n: number;
  private readonly aug: Mat;
  private readonly tol: number;
  private result: Mat | null = null;

  constructor(M: Mat, tol?: number) {
    const n = LinAlg.rows(M);
    if (LinAlg.cols(M) !== n) throw new Error('MatrixInverse: matrix must be square');
    this.n = n;
    this.tol = tol ?? Math.max(1, LinAlg.maxAbs(M)) * n * 1e-14;
    // [ M | I ]
    this.aug = M.map((row, i) => row.concat(LinAlg.identity(n)[i]));
  }

  inverse(): Mat {
    if (this.result !== null) return this.result;
    const n = this.n;
    const a = this.aug;
    for (let col = 0; col < n; col++) {
      let best = col;
      for (let i = col + 1; i < n; i++) if (Math.abs(a[i][col]) > Math.abs(a[best][col])) best = i;
      if (Math.abs(a[best][col]) <= this.tol) throw new Error('MatrixInverse: matrix is singular');
      if (best !== col) { const t = a[best]; a[best] = a[col]; a[col] = t; }
      const piv = a[col][col];
      for (let j = 0; j < 2 * n; j++) a[col][j] /= piv;
      for (let i = 0; i < n; i++) {
        if (i === col) continue;
        const f = a[i][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) a[i][j] -= f * a[col][j];
      }
    }
    this.result = a.map(row => row.slice(n));
    return this.result;
  }

  /** Solve M·X = B (B is r×c) using the computed inverse. */
  solve(B: Mat): Mat {
    return LinAlg.matMul(this.inverse(), B);
  }
}

/** Eigen-decomposition of a SYMMETRIC matrix via the cyclic Jacobi rotation
 *  method. Eigenvalues are returned in ASCENDING order with their eigenvectors
 *  as the columns of `vectors()`. Symmetric PSD Gramians are the intended
 *  input, so the iteration converges quickly for the small matrices used by
 *  the control-systems family. */
export class SymmetricEigen {
  private readonly n: number;
  private vals: Vec | null = null;
  private vecs: Mat | null = null;
  private readonly source: Mat;
  private readonly sweeps: number;

  constructor(M: Mat, sweeps = 100) {
    const n = LinAlg.rows(M);
    if (LinAlg.cols(M) !== n) throw new Error('SymmetricEigen: matrix must be square');
    this.n = n;
    this.source = LinAlg.copy(M);
    this.sweeps = sweeps;
  }

  private compute(): void {
    const n = this.n;
    const a = LinAlg.copy(this.source);
    const v = LinAlg.identity(n);
    for (let sweep = 0; sweep < this.sweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-30) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1);
          const s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const idx = Array.from({length: n}, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
    this.vals = idx.map(i => a[i][i]);
    this.vecs = Array.from({length: n}, (_, r) => idx.map(i => v[r][i]));
  }

  /** Eigenvalues in ascending order. */
  values(): Vec {
    if (this.vals === null) this.compute();
    return this.vals!.slice();
  }

  /** Eigenvectors as matrix columns, aligned with `values()`. */
  vectors(): Mat {
    if (this.vecs === null) this.compute();
    return LinAlg.copy(this.vecs!);
  }

  /** Smallest eigenvalue (clamped at 0 for numerical PSD noise). */
  minEigenvalue(): number {
    return Math.max(0, this.values()[0]);
  }

  /** Largest eigenvalue. */
  maxEigenvalue(): number {
    const v = this.values();
    return v[v.length - 1];
  }

  /** Eigenvector (column) for the smallest eigenvalue. */
  minEigenvector(): Vec {
    return LinAlg.transpose(this.vectors())[0];
  }

  /** Eigenvector (column) for the largest eigenvalue. */
  maxEigenvector(): Vec {
    const cols = LinAlg.transpose(this.vectors());
    return cols[cols.length - 1];
  }

  /** λ_max / λ_min — anisotropy of the Gramian (∞-ish when near-singular). */
  conditionNumber(): number {
    const lo = this.minEigenvalue();
    if (lo <= 0) return Infinity;
    return this.maxEigenvalue() / lo;
  }
}

/** Owns a single numeric-rank computation. Holding the working copy and the
 *  tolerance as instance state keeps the Gaussian-elimination pivots out of
 *  any standalone helper function. */
export class MatrixRank {
  private readonly work: Mat;
  private readonly tol: number;
  private rankValue: number | null = null;

  constructor(M: Mat, tol?: number) {
    this.work = LinAlg.copy(M);
    const r = LinAlg.rows(M);
    const c = LinAlg.cols(M);
    const scale = Math.max(1, LinAlg.maxAbs(M));
    this.tol = tol ?? Math.max(r, c) * scale * 1e-12;
  }

  /** Row-reduce (partial pivoting) and count pivots above tolerance. */
  rank(): number {
    if (this.rankValue !== null) return this.rankValue;
    const A = this.work;
    const r = LinAlg.rows(A);
    const c = LinAlg.cols(A);
    let pivotRow = 0;
    for (let col = 0; col < c && pivotRow < r; col++) {
      let best = pivotRow;
      for (let i = pivotRow + 1; i < r; i++) {
        if (Math.abs(A[i][col]) > Math.abs(A[best][col])) best = i;
      }
      if (Math.abs(A[best][col]) <= this.tol) continue;
      if (best !== pivotRow) {
        const tmp = A[best];
        A[best] = A[pivotRow];
        A[pivotRow] = tmp;
      }
      const piv = A[pivotRow][col];
      for (let i = 0; i < r; i++) {
        if (i === pivotRow) continue;
        const factor = A[i][col] / piv;
        if (factor === 0) continue;
        for (let j = col; j < c; j++) A[i][j] -= factor * A[pivotRow][j];
      }
      pivotRow++;
    }
    this.rankValue = pivotRow;
    return pivotRow;
  }

  /** True iff the rank equals the requested full rank `n`. */
  isFullRank(n: number): boolean {
    return this.rank() === n;
  }
}
