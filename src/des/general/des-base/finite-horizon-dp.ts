'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/finite_horizon_dp.rs  (module des::general::des_base::finite_horizon_dp)
// 1:1 file move. Template-method base for finite-horizon DP via backward induction.
//
// Declarations → Rust:
//   interface DPOutcome             -> struct DPOutcome { prob, reward, next_state }
//   interface DPOptions             -> struct (#[derive(Default)])
//   abstract class FiniteHorizonDPStation -> trait FiniteHorizonDPStation: DESStation
//   fn maxArr / minArr (private)    -> fn max_arr / min_arr (or iterator .fold with total_cmp)
//
// Conversion notes (file-specific):
//   - TEMPLATE METHOD: `runTimeStep` (one backward sweep) is final; required hooks
//     horizon/numStates/numActions/transitions -> required trait fns; terminalReward/
//     stageDiscount/onStageComputed -> provided defaults.
//   - `rng: () => number` defaulting to `Math.random` -> inject `RandomSource`
//     (shared/capabilities); the reservoir tie-break must use it.
//   - `V: number[][]` / `policy: number[][]` -> `Vec<Vec<f64>>` / `Vec<Vec<i64>>`
//     (policy uses -1 sentinel -> consider `Option<usize>`).
//   - `stageHistory: Array<{stage,maxV,minV}>` -> `Vec<StageStat>` (named struct).
//   - non-ASCII `γ` identifier -> `gamma`.
//   - `throw new Error` (horizon<1) + Preconditions.* -> `Result`/`panic!`; reuse
//     preconditions.rs.
// =============================================================================

// =============================================================================
// general/des-base/finite-horizon-dp.ts — base class for FINITE-HORIZON
// DYNAMIC PROGRAMMING via backward induction.
//
// PROBLEM SHAPE
// ─────────────
//   Horizon T < ∞, terminal value V_T(s) known. Compute
//
//     V_t(s) = max_a Σ_{s'} p(s'|s,a) [ r(s, a, s', t) + V_{t+1}(s') ],   t < T
//     π_t(s) = argmax_a same expression
//
//   by walking BACKWARDS from t = T-1 down to t = 0. This is the canonical
//   finite-horizon Bellman recursion (multi-period inventory, multi-stage
//   stopping, finite-horizon investment, …) and is *distinct* from the
//   infinite-horizon fixed-point operator handled by `value-iteration.ts`.
//
// AS A DES STATION
// ────────────────
//   `runTimeStep()` performs ONE backward sweep at the current `t`. The
//   station finishes when `t` reaches 0. Each backward sweep is a tick.
//
// HOOKS (abstract)
// ────────────────
//   horizon()             → T (≥ 1)
//   numStates()           → |S|
//   numActions(s, t)      → A(s, t) (legal action count; 0-indexed; can vary by t)
//   transitions(s, a, t)  → list of {prob, reward, nextState} for STAGE t
//   terminalReward(s)     → V_T(s) (boundary condition)
//
// HOOKS (optional override)
// ─────────────────────────
//   stageDiscount(t)      → γ_t (default 1)
//   onStageComputed(t, V) → instrumentation
// =============================================================================

import {DESStation} from './station';
import {Preconditions} from './preconditions';
import {ARGMAX_EPS_DEFAULT} from './argmax';

export interface DPOutcome {
  prob: number;
  reward: number;
  nextState: number;
}

export interface DPOptions {
  /** Optional cap on max history length. */
  maxHistoryLen?: number;
  /**
   * If true (default), break argmax ties uniformly at random in the
   * backward induction. Use `false` to revert to first-action-wins
   * (matches the deterministic textbook DP and most reference outputs;
   * useful when comparing against a fixed reference table).
   *
   * NOTE: the VALUE function V_t(s) is identical either way (since it
   * only depends on the max, not the argmax); only the optimal policy
   * π_t(s) is affected, and only on states where multiple actions
   * achieve the optimum.
   */
  randomTieBreak?: boolean;
  tieBreakEps?: number;
  rng?: () => number;
}

export abstract class FiniteHorizonDPStation extends DESStation {
  /** V[t][s] for t = 0 … T (length T+1). Built incrementally. */
  readonly V: number[][] = [];
  /** π[t][s] for t = 0 … T-1 (length T). −1 in terminal positions. */
  readonly policy: number[][] = [];
  /** True after stage 0 has been computed. */
  protected finished = false;
  /** Current stage being processed (counts down from T-1 to 0). */
  protected currentStage = 0;
  /** History of {stage, maxV, minV} for diagnostics. */
  readonly stageHistory: Array<{stage: number; maxV: number; minV: number}> = [];

  protected readonly maxHistoryLen: number;
  protected readonly randomTieBreak: boolean;
  protected readonly tieBreakEps: number;
  protected readonly rng: () => number;

  constructor(id: string, opts: DPOptions = {}) {
    super(id);
    this.maxHistoryLen = opts.maxHistoryLen ?? Infinity;
    this.randomTieBreak = opts.randomTieBreak ?? true;
    this.tieBreakEps = opts.tieBreakEps ?? ARGMAX_EPS_DEFAULT;
    this.rng = opts.rng ?? Math.random;
  }

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  protected abstract horizon(): number;
  protected abstract numStates(): number;
  protected abstract numActions(state: number, stage: number): number;
  protected abstract transitions(state: number, action: number, stage: number): DPOutcome[];
  /** Terminal reward V_T(s). Default 0; override for non-trivial boundary. */
  protected terminalReward(_state: number): number { return 0; }

  // ── HOOKS (optional override) ────────────────────────────────────────────

  /** Per-stage discount factor. Default 1 (undiscounted finite horizon). */
  protected stageDiscount(_stage: number): number { return 1; }
  protected onStageComputed(_stage: number, _V: readonly number[]): void {}

  // ── TEMPLATE METHOD ──────────────────────────────────────────────────────

  /** Subclasses MUST call this at the end of their constructor — it
   *  installs V_T from the terminal reward and sets currentStage = T-1. */
  protected bootstrap(): void {
    const T = this.horizon();
    const N = this.numStates();
    if (T < 1) throw new Error(`finite-horizon-dp: horizon must be ≥ 1, got ${T}`);
    const VT = new Array<number>(N);
    for (let s = 0; s < N; s++) VT[s] = this.terminalReward(s);
    this.V[T] = VT;
    this.currentStage = T - 1;
    if (this.stageHistory.length < this.maxHistoryLen) {
      this.stageHistory.push({stage: T, maxV: maxArr(VT), minV: minArr(VT)});
    }
  }

  override hasWork(): boolean { return !this.finished; }

  runTimeStep(): void {
    if (this.finished) return;
    const t = this.currentStage;
    const N = this.numStates();
    const Vnext = this.V[t + 1];
    const Vt = new Array<number>(N);
    const pol = new Array<number>(N);
    const γ = this.stageDiscount(t);
    const eps = this.tieBreakEps;
    const useTieBreak = this.randomTieBreak;
    for (let s = 0; s < N; s++) {
      const A = this.numActions(s, t);
      let bestQ = -Infinity;
      let bestA = -1;
      let tieCount = 0;
      for (let a = 0; a < A; a++) {
        const outs = this.transitions(s, a, t);
        if (outs.length === 0) continue;
        let q = 0;
        for (let i = 0; i < outs.length; i++) {
          const o = outs[i];
          q += o.prob * (o.reward + γ * Vnext[o.nextState]);
        }
        if (bestA < 0 || q > bestQ + eps) {
          bestQ = q; bestA = a; tieCount = 1;
        } else if (useTieBreak && q >= bestQ - eps) {
          tieCount++;
          // Reservoir sampling: keep the new index with probability 1/tieCount.
          if (this.rng() * tieCount < 1) bestA = a;
        }
      }
      Vt[s] = bestA >= 0 ? bestQ : 0;
      pol[s] = bestA;
    }
    this.V[t] = Vt;
    this.policy[t] = pol;
    if (this.stageHistory.length < this.maxHistoryLen) {
      this.stageHistory.push({stage: t, maxV: maxArr(Vt), minV: minArr(Vt)});
    }
    this.onStageComputed(t, Vt);
    if (t === 0) { this.finished = true; return; }
    this.currentStage = t - 1;
  }

  /** Pre-run guards: horizon ≥ 1, |S| ≥ 1, every stage's transition list
   *  is a proper probability distribution (probs ≥ 0, sum to 1 within
   *  tol), every reward is finite. Subclasses can override and call
   *  `super.assertPreconditions()` to keep these. */
  override assertPreconditions(): void {
    super.assertPreconditions();
    const T = this.horizon();
    const N = this.numStates();
    Preconditions.check(this.constructor.name, 'horizon()', 'be an integer >= 1',
                        Number.isInteger(T) && T >= 1, T);
    Preconditions.check(this.constructor.name, 'numStates()', 'be an integer >= 1',
                        Number.isInteger(N) && N >= 1, N);
    // Sample-validate ONE state per stage (full validation is O(T·|S|·|A|·max|outcomes|);
    // the runner is the right place for that, here we keep guards O(T·|S|)).
    for (let t = 0; t < T; t++) {
      for (let s = 0; s < N; s++) {
        const A = this.numActions(s, t);
        Preconditions.check(this.constructor.name, `numActions(${s},${t})`, 'be >= 0',
                            Number.isInteger(A) && A >= 0, A);
        for (let a = 0; a < A; a++) {
          const outs = this.transitions(s, a, t);
          if (outs.length === 0) continue;
          let sum = 0;
          for (let i = 0; i < outs.length; i++) {
            const o = outs[i];
            Preconditions.check(this.constructor.name, `transitions(${s},${a},${t})[${i}].prob`,
              'be in [0, 1]',
              Number.isFinite(o.prob) && o.prob >= 0 && o.prob <= 1 + 1e-9, o.prob);
            Preconditions.finite(this.constructor.name, `transitions(${s},${a},${t})[${i}].reward`, o.reward);
            Preconditions.check(this.constructor.name, `transitions(${s},${a},${t})[${i}].nextState`,
              'be a valid state index',
              Number.isInteger(o.nextState) && o.nextState >= 0 && o.nextState < N, o.nextState);
            sum += o.prob;
          }
          if (Math.abs(sum - 1) > 1e-6) {
            Preconditions.check(this.constructor.name, `transitions(${s},${a},${t}) probs`,
              'sum to 1', false, sum);
          }
        }
      }
    }
    for (let t = 0; t < T; t++) {
      const γ = this.stageDiscount(t);
      Preconditions.inRange(this.constructor.name, `stageDiscount(${t})`, γ, 0, 1);
    }
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  /** Value function at stage t. Stage T returns the terminal rewards. */
  getV(t: number): readonly number[] { return this.V[t]; }
  /** Optimal action at stage t (0 ≤ t ≤ T-1) in state s. */
  getAction(t: number, s: number): number { return this.policy[t][s]; }
  isFinished(): boolean { return this.finished; }
  getCurrentStage(): number { return this.currentStage; }
}

function maxArr(a: readonly number[]): number {
  let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m;
}
function minArr(a: readonly number[]): number {
  let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m;
}
