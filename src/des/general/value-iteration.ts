'use strict';

// =============================================================================
// Generic value iteration for finite-state, finite-action MDPs.
//
// Bellman optimality equation:
//   V*(s) = max_a Σ_s' T(s'|s,a) · [ r(s, a, s') + γ · V*(s') ]
//   π*(s) = argmax_a Σ_s' T(s'|s,a) · [ r(s, a, s') + γ · V*(s') ]
//
// Caller supplies an `MDPSpec`: number of states, number of actions per
// state (variable), and an `outcomes(s, a)` callable that returns the
// list of {prob, reward, nextState} triples for taking action `a` in
// state `s`.
//
// Convergence: max_s |V_next(s) − V(s)| < tol. Default tol = 1e-9.
//
// Difference from the USACC solver in `mdp/value-iteration.ts`: this
// is parameterised by an MDP spec object rather than hard-coded against
// the court MDP. Use this one for the newsvendor / inventory MDPs and
// any other generic discrete MDP the framework grows.
//
// AS A DES: orchestrated by ValueIterationStation, a leaf of
// FixedPointIterationStation<Float64Array> in `general/des-base/`.
// runTimeStep() is the template method; the abstract hooks supply:
//   • initialState() → V_0 (zeros, with terminal pinning)
//   • applyOperator(V_k) → V_{k+1} via the Bellman backup
//   • delta(V_k, V_{k+1}) → max-norm change, used for convergence
// =============================================================================

import {
  FixedPointIterationStation, runIterativeDES,
  intrinsicCheck, externalReferenceValidator, ValidationCheck,
  scanArgMaxTieBreak,
} from './des-base';

export interface Outcome {
  /** Probability of this outcome, in [0, 1]. */
  prob: number;
  /** Immediate reward received on this transition. */
  reward: number;
  /** Next state index after this outcome. */
  nextState: number;
}

export interface MDPSpec {
  numStates: number;
  /**
   * Number of legal actions in state s. May be the same for every state
   * or vary; if action `a` is illegal in state `s`, return [] from
   * `outcomes(s, a)` and it will be skipped. For simplicity callers
   * usually use a constant action count.
   */
  numActions: (s: number) => number;
  /** Outcomes for taking action `a` in state `s`. Probabilities must sum to 1 (within `tol`). */
  outcomes: (s: number, a: number) => Outcome[];
  /** Predicate marking absorbing terminal states. Default: never terminal. */
  isTerminal?: (s: number) => boolean;
  /** Reward associated with being in a terminal state (V is pinned here). Default: 0. */
  terminalReward?: (s: number) => number;
  /** Optional human-readable label for state s, used by toString helpers. */
  stateLabel?: (s: number) => string;
  /** Optional human-readable label for action a, used by toString helpers. */
  actionLabel?: (a: number) => string;
}

export interface VIOptions {
  /** Discount factor in [0, 1]. Default 0.95. */
  gamma?: number;
  /** Convergence tolerance on max |ΔV|. Default 1e-9. */
  tol?: number;
  /** Hard cap on iterations. Default 5000. */
  maxIter?: number;
  /** If true, validate that outcome probabilities sum to 1 ± probTol. Default true. */
  validateProbs?: boolean;
  probTol?: number;
  /** Optional path to a reference JSON shaped like the court-MDP scipy
   *  output: `{ V: number[], gamma?: number }`. When provided, the
   *  station auto-attaches an external-reference validator that compares
   *  `V` element-wise within `referenceTol`. Missing files emit a single
   *  failed check unless `silentIfMissing` is true. */
  referencePath?: string;
  referenceTol?: number;
  silentIfMissing?: boolean;
  /**
   * Random tie-breaking for `greedyPolicy()`. When multiple actions achieve
   * the maximum Q-value within `tieBreakEps`, one is chosen uniformly at
   * random. Defaults to true; pass `false` to revert to first-action-wins
   * (e.g. when comparing against a reference implementation that uses
   * deterministic argmax). Default rng is `Math.random`, so wrap in
   * `withSeed(...)` for reproducibility.
   */
  randomTieBreak?: boolean;
  tieBreakEps?: number;
  rng?: () => number;
}

export interface VIResult {
  V: Float64Array;
  policy: Int32Array;
  iterations: number;
  finalDelta: number;
  gamma: number;
}

/**
 * ValueIterationStation — leaf of FixedPointIterationStation<Float64Array>.
 *
 *   initialState  → V_0 (zeros, terminal states pinned to terminalReward)
 *   applyOperator → V_{k+1}(s) = max_a Σ p(s'|s,a) [r + γ V_k(s')]
 *   delta         → max_s |V_{k+1}(s) − V_k(s)|
 */
export class ValueIterationStation extends FixedPointIterationStation<Float64Array> {
  private readonly spec: MDPSpec;
  private readonly gamma: number;
  private readonly T: Outcome[][][];
  private readonly aCount: number[];
  private readonly isTerminal: (s: number) => boolean;
  private readonly terminalReward: (s: number) => number;
  private readonly randomTieBreak: boolean;
  private readonly tieBreakEps: number;
  private readonly rng: () => number;

  constructor(spec: MDPSpec, opts: VIOptions = {}) {
    super('value-iteration', {
      tol: opts.tol ?? 1e-9,
      maxIter: opts.maxIter ?? 5000,
    });
    this.spec = spec;
    this.gamma = opts.gamma ?? 0.95;
    this.isTerminal = spec.isTerminal ?? (() => false);
    this.terminalReward = spec.terminalReward ?? (() => 0);
    this.randomTieBreak = opts.randomTieBreak ?? true;
    this.tieBreakEps = opts.tieBreakEps ?? 1e-12;
    this.rng = opts.rng ?? Math.random;
    const validateProbs = opts.validateProbs ?? true;
    const probTol = opts.probTol ?? 1e-9;

    // Pre-build the transition table once (50× speedup on iterates).
    this.T = new Array(spec.numStates);
    this.aCount = new Array(spec.numStates);
    for (let s = 0; s < spec.numStates; s++) {
      if (this.isTerminal(s)) { this.T[s] = []; this.aCount[s] = 0; continue; }
      const A = spec.numActions(s);
      this.aCount[s] = A;
      const perAction: Outcome[][] = new Array(A);
      for (let a = 0; a < A; a++) {
        const ol = spec.outcomes(s, a);
        if (validateProbs && ol.length > 0) {
          let total = 0;
          for (const o of ol) total += o.prob;
          if (Math.abs(total - 1) > probTol) {
            console.warn(`[value-iteration] outcomes(${s}, ${a}) probabilities sum to ${total} (expected 1 ± ${probTol}) — transition model is not a valid distribution.`);
            throw new Error(
              `outcomes(${s}, ${a}) probabilities sum to ${total}, expected 1 (tol=${probTol})`);
          }
        }
        perAction[a] = ol;
      }
      this.T[s] = perAction;
    }
    this.bootstrap();

    // Intrinsic invariants: when the run terminates with reason 'converged',
    // `lastDelta` must be ≤ tol; the delta history must be non-increasing
    // up to numerical noise (Bellman operator is a γ-contraction in
    // sup-norm).
    this.addValidator(intrinsicCheck<ValueIterationStation>({
      name: 'vi.converged-implies-delta-le-tol',
      group: 'value-iteration-intrinsic',
      predicate: st => st.getReason() !== 'converged' || st.getLastDelta() <= st['tol'] + 1e-15,
      expected: 'lastDelta ≤ tol when converged',
      observedFn: st => `reason=${st.getReason()}  lastDelta=${st.getLastDelta()}  tol=${st['tol']}`,
      details: 'fixed-point declared converged but lastDelta exceeds tol',
    }));
    this.addValidator(intrinsicCheck<ValueIterationStation>({
      name: 'vi.delta-history-bounded-by-gamma',
      group: 'value-iteration-intrinsic',
      // Bellman is γ-contraction: deltaHistory[k+1] ≤ γ · deltaHistory[k].
      // Allow a 1e-9 absolute slack for floating-point noise.
      predicate: st => {
        const dh = st.deltaHistory;
        const gamma = st['gamma'];
        for (let k = 1; k < dh.length; k++) {
          if (dh[k] > gamma * dh[k - 1] + 1e-9) return false;
        }
        return true;
      },
      expected: `Δ_{k+1} ≤ γ·Δ_k  (γ=${this.gamma})`,
      observedFn: st => `deltaHistory.length=${st.deltaHistory.length}`,
      details: 'Bellman operator should be a γ-contraction in sup-norm',
    }));

    // Optional external-reference validator (e.g. scipy / court-MDP).
    if (opts.referencePath) {
      const refTol = opts.referenceTol ?? 1e-4;
      this.addValidator(externalReferenceValidator<ValueIterationStation>({
        name: 'vi.value-vs-reference',
        group: 'value-iteration-external',
        referencePath: opts.referencePath,
        silentIfMissing: opts.silentIfMissing,
        compare: (st, ref) => {
          const V = st.getCurrent();
          const refV = ref.V as number[];
          if (!Array.isArray(refV) || refV.length !== V.length) {
            return [{name: 'vi.value-vs-reference', passed: false,
              observed: `V.length=${V.length}`, expected: `V.length=${refV?.length ?? 'NaN'}`,
              details: 'reference V missing or wrong length'}];
          }
          let maxAbs = 0; let argmax = -1;
          for (let s = 0; s < V.length; s++) {
            const e = Math.abs(V[s] - refV[s]);
            if (e > maxAbs) { maxAbs = e; argmax = s; }
          }
          const passed = maxAbs <= refTol;
          const out: ValidationCheck[] = [{
            name: 'vi.value-vs-reference', passed,
            observed: `max|ΔV|=${maxAbs.toExponential(3)} at s=${argmax}`,
            expected: `≤ ${refTol}`,
            details: passed ? undefined :
              `V[${argmax}]=${V[argmax]}  ref=${refV[argmax]}`,
          }];
          return out;
        },
      }));
    }
  }

  // ── HOOKS ────────────────────────────────────────────────────────────────

  protected initialState(): Float64Array {
    const V = new Float64Array(this.spec.numStates);
    for (let s = 0; s < this.spec.numStates; s++) {
      if (this.isTerminal(s)) V[s] = this.terminalReward(s);
    }
    return V;
  }

  protected applyOperator(V: Float64Array): Float64Array {
    const Vn = new Float64Array(this.spec.numStates);
    for (let s = 0; s < this.spec.numStates; s++) {
      if (this.isTerminal(s)) { Vn[s] = V[s]; continue; }
      const A = this.aCount[s];
      let best = -Infinity;
      let any = false;
      for (let a = 0; a < A; a++) {
        const ol = this.T[s][a];
        if (ol.length === 0) continue;
        let q = 0;
        for (let i = 0; i < ol.length; i++) {
          const o = ol[i];
          q += o.prob * (o.reward + this.gamma * V[o.nextState]);
        }
        if (q > best) { best = q; any = true; }
      }
      Vn[s] = any ? best : 0;
    }
    return Vn;
  }

  protected delta(prev: Float64Array, next: Float64Array): number {
    let d = 0;
    for (let s = 0; s < prev.length; s++) {
      const di = Math.abs(next[s] - prev[s]);
      if (di > d) d = di;
      if (this.isTerminal(s)) continue;
    }
    return d;
  }

  // ── PUBLIC ───────────────────────────────────────────────────────────────

  /**
   * Extract the greedy policy from the current value function.
   *
   * When multiple actions tie for the max Q-value (within `tieBreakEps`),
   * one is chosen uniformly at random (controlled by `opts.randomTieBreak`,
   * default true). Without this, the algorithm would always pick the lowest-
   * index tied action, which biases symmetric MDPs (e.g. gridworlds with
   * equally-good directions from a state) and breaks reproducibility vs.
   * textbook implementations.
   */
  greedyPolicy(): Int32Array {
    const V = this.current;
    const policy = new Int32Array(this.spec.numStates);
    for (let s = 0; s < this.spec.numStates; s++) {
      if (this.isTerminal(s)) { policy[s] = -1; continue; }
      const A = this.aCount[s];
      if (this.randomTieBreak) {
        policy[s] = scanArgMaxTieBreak(A, a => {
          const ol = this.T[s][a];
          if (ol.length === 0) return -Infinity;
          let q = 0;
          for (let i = 0; i < ol.length; i++) {
            const o = ol[i];
            q += o.prob * (o.reward + this.gamma * V[o.nextState]);
          }
          return q;
        }, this.rng, this.tieBreakEps);
      } else {
        let bestA = -1; let bestQ = -Infinity;
        for (let a = 0; a < A; a++) {
          const ol = this.T[s][a];
          if (ol.length === 0) continue;
          let q = 0;
          for (let i = 0; i < ol.length; i++) {
            const o = ol[i];
            q += o.prob * (o.reward + this.gamma * V[o.nextState]);
          }
          if (q > bestQ) { bestQ = q; bestA = a; }
        }
        policy[s] = bestA;
      }
    }
    return policy;
  }
}

/**
 * Run value iteration on an MDPSpec. Returns optimal value function V
 * and greedy policy π extracted from V.
 *
 * The transition table is built once at the start (one call to
 * `outcomes(s, a)` per (s, a) pair) and reused on every Bellman sweep,
 * which is typically a 50× speedup over recomputing on the fly.
 */
export function valueIteration(spec: MDPSpec, opts: VIOptions = {}): VIResult {
  const station = new ValueIterationStation(spec, opts);
  runIterativeDES([station]);
  const finalDelta = station.getLastDelta();
  const tol = opts.tol ?? 1e-9;
  if (finalDelta > tol) {
    console.warn(`[value-iteration] stopped after ${station.getIteration()} iterations with max|ΔV|=${finalDelta} > tol=${tol} (reason=${station.getReason?.() ?? 'unknown'}); value function may not be optimal.`);
  }
  const V = station.getCurrent();
  const policy = station.greedyPolicy();
  return {
    V, policy,
    iterations: station.getIteration(),
    finalDelta: station.getLastDelta(),
    gamma: opts.gamma ?? 0.95,
  };
}

/**
 * Q-value of (s, a) under value function V. Useful for inspecting how
 * close the second-best action is to the optimum (policy fragility).
 */
export function qValue(spec: MDPSpec, V: Float64Array, s: number, a: number, gamma: number): number {
  const ol = spec.outcomes(s, a);
  let q = 0;
  for (const o of ol) q += o.prob * (o.reward + gamma * V[o.nextState]);
  return q;
}

/**
 * For each state, return all action Q-values sorted descending. Useful
 * for diagnosing how distinguishable the optimal action is.
 */
export function qValuesAll(spec: MDPSpec, V: Float64Array, s: number, gamma: number): Array<{action: number; q: number}> {
  const A = spec.numActions(s);
  const out: Array<{action: number; q: number}> = [];
  for (let a = 0; a < A; a++) {
    out.push({action: a, q: qValue(spec, V, s, a, gamma)});
  }
  out.sort((x, y) => y.q - x.q);
  return out;
}
