'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/semi_mdp.rs  (module des::general::des_base::semi_mdp)
// 1:1 file move. Options-framework Semi-MDP + intra-option SMDP Q-learning;
// extends RLAgentStation.
//
// Declarations → Rust:
//   interface Option<S,A>           -> trait Opt<S, A> { fn init; fn policy; fn terminate; }
//                                      (NOTE: `Option` collides with std::option::Option —
//                                       rename, e.g. `SmdpOption`/`Opt`)
//   interface SemiMDPOptions        -> struct (#[derive(Default)] bar `rng`)
//   abstract class SemiMDPAgentStation<S,A> -> trait/struct: RLAgentStation<S, A>
//
// Conversion notes (file-specific):
//   - NAME CLASH: the `Option<S,A>` interface MUST be renamed in Rust to avoid the
//     prelude `Option`.
//   - INHERITANCE: implements RLAgentStation's pickAction/update/endOfEpisode and
//     adds abstract options()/stateKey() hooks -> required trait fns.
//   - `Q: number[][]` sparse-ish keyed by stateKey -> `Vec<Vec<f64>>` or
//     `HashMap<usize, Vec<f64>>` (stateKey returns the index).
//   - `optionStartState!: S` definite-assignment -> `Option<S>`.
//   - `rng: () => number` -> inject `RandomSource`; tie-break via argmax.rs.
//   - `Required<SemiMDPOptions>` resolved opts -> concrete config struct.
//   - non-ASCII `ω`, `γ` -> `omega`, `gamma`. `Math.pow(γ, tau)` -> `gamma.powi(tau)`.
//   - `throw new Error` (no legal option) -> `Result`/`panic!`.
// =============================================================================

// =============================================================================
// general/des-base/semi-mdp.ts — base classes for SEMI-MARKOV DECISION
// PROCESSES under the OPTIONS FRAMEWORK (Sutton, Precup, Singh 1999).
//
// PROBLEM SHAPE
// ─────────────
//   A standard MDP fixes a one-step decision granularity. A Semi-MDP
//   instead allows decisions to span MULTIPLE primitive time steps: the
//   agent picks an OPTION ω (a sub-policy), executes it until the
//   option's TERMINATION condition fires, and only then chooses the
//   next option. Three pieces define an option:
//
//       ω = ⟨ I_ω,  π_ω,  β_ω ⟩
//       I_ω(s) ∈ {0, 1}   "initiation set" — can ω start in s?
//       π_ω(a|s)          option's INTERNAL policy
//       β_ω(s) ∈ [0, 1]   probability of TERMINATING in s
//
//   The Semi-MDP Bellman equation gives an exact value over options:
//
//       Q(s, ω) = E[ Σ_{k=0}^{τ−1} γ^k r_{t+k} + γ^τ max_{ω'} Q(s_{t+τ}, ω') ]
//
//   so options DO NOT change the optimal value but can dramatically
//   speed up planning/learning by introducing temporal abstraction.
//
// THIS BASE PROVIDES
// ──────────────────
//   • `Option<S, A>` interface with init/policy/terminate hooks.
//   • `SemiMDPAgentStation<S, A>` — extends RLAgentStation, performs
//     INTRA-OPTION SMDP Q-LEARNING:
//
//       Q(s, ω) ← Q(s, ω) + α [ r̄ + γ^τ max_{ω'} Q(s', ω') − Q(s, ω) ]
//
//     while a primitive policy executes inside the option.
//   • Frame-rate runner pattern: every primitive step the agent decides
//     whether the option is still in effect; on termination it picks a
//     new option.
//
// HOOKS (abstract on subclass)
// ────────────────────────────
//   options()       → readonly Option<S, A>[]    (full option library)
//   pickOption(s)   → ω index                    (e.g. ε-greedy over Q)
//
// HOOKS (default on Option)
// ─────────────────────────
//   The user passes plain {init, policy, terminate} objects.
// =============================================================================

import {RLAgentStation} from './rl-agent';
import {StateToken, ActionToken, TransitionToken} from './rl-tokens';
import {ARGMAX_EPS_DEFAULT} from './argmax';

export interface Option<S = number, A = number> {
  /** Human-readable label (used in debug output). */
  readonly name: string;
  /** Initiation set: true iff this option can start in state `s`. */
  init(s: S): boolean;
  /** Internal policy of the option. */
  policy(s: S, rng: () => number): A;
  /** Termination probability β_ω(s). 1 = terminate; 0 = continue. */
  terminate(s: S): number;
}

export interface SemiMDPOptions {
  rng: () => number;
  /** Step size α. Default 0.1. */
  alpha?: number;
  /** Discount γ. Default 0.95. */
  gamma?: number;
  /** ε-greedy probability over the OPTION level. Default 0.1. */
  epsilon?: number;
  /** ε-decay multiplier per primitive episode. Default 1. */
  epsilonDecay?: number;
  /** ε-floor. Default 0.01. */
  epsilonMin?: number;
}

/** SMDP Q-learning at the option level on top of a discrete-state MDP. */
export abstract class SemiMDPAgentStation<S = number, A = number> extends RLAgentStation<S, A> {
  protected readonly opts: Required<SemiMDPOptions>;
  /** Q[s][ω]. Subclasses size this in their constructor. */
  protected Q: number[][] = [];
  /** Currently executing option (−1 = none). */
  protected currentOption = -1;
  /** State at which the current option began. */
  protected optionStartState!: S;
  /** Cumulative discounted reward inside the current option. */
  protected optionReward = 0;
  /** Number of primitive steps inside the current option. */
  protected optionTau = 0;

  constructor(id: string, semiOpts: SemiMDPOptions) {
    super(id, {rng: semiOpts.rng});
    this.opts = {
      rng: semiOpts.rng,
      alpha: semiOpts.alpha ?? 0.1,
      gamma: semiOpts.gamma ?? 0.95,
      epsilon: semiOpts.epsilon ?? 0.1,
      epsilonDecay: semiOpts.epsilonDecay ?? 1,
      epsilonMin: semiOpts.epsilonMin ?? 0.01,
    };
  }

  // ── HOOKS (abstract) ─────────────────────────────────────────────────────

  /** Library of available options. */
  protected abstract options(): readonly Option<S, A>[];
  /** State-key for indexing Q[s]. Required because S is generic — for
   *  integer states return the integer; for tuples return a hash. */
  protected abstract stateKey(s: S): number;

  // ── HOOKS (optional override) ────────────────────────────────────────────

  /**
   * ε-greedy over options at state s, with UNIFORM RANDOM TIE-BREAKING
   * on the greedy argmax. Necessary because Q starts as all-zero, so
   * deterministic argmax would always pick the lowest-index legal option
   * on the very first call (or after any symmetric state).
   */
  protected pickOption(s: S, rng: () => number): number {
    const opts = this.options();
    const legal: number[] = [];
    for (let i = 0; i < opts.length; i++) if (opts[i].init(s)) legal.push(i);
    if (legal.length === 0) throw new Error(`no option available in state`);
    if (rng() < this.opts.epsilon) return legal[Math.floor(rng() * legal.length)];
    const eps = ARGMAX_EPS_DEFAULT;
    let bestQ = -Infinity; let best = -1; let tieCount = 0;
    for (const i of legal) {
      const q = this.Q[this.stateKey(s)]?.[i] ?? 0;
      if (best < 0 || q > bestQ + eps) {
        bestQ = q; best = i; tieCount = 1;
      } else if (q >= bestQ - eps) {
        tieCount++;
        if (rng() * tieCount < 1) best = i;
      }
    }
    return best;
  }

  // ── ACTION / UPDATE WIRED THROUGH RLAgentStation ────────────────────────

  /** Inside-option primitive action selection. The option's `policy`
   *  hook decides the next primitive a. If we're between options we
   *  pick a new option first. */
  protected pickAction(state: S, rng: () => number): A {
    if (this.currentOption < 0 || this.options()[this.currentOption].terminate(state) >= rng()) {
      // Option terminated (or none active). Backup Q(s_start, ω) if we
      // had one running, then choose a new option.
      if (this.currentOption >= 0) {
        this.backup(state, false);
      }
      this.currentOption = this.pickOption(state, rng);
      this.optionStartState = state;
      this.optionReward = 0;
      this.optionTau = 0;
    }
    return this.options()[this.currentOption].policy(state, rng);
  }

  protected update(_state: S, _action: A, reward: number, nextState: S, done: boolean): void {
    if (this.currentOption < 0) return;
    const γ = this.opts.gamma;
    this.optionReward += Math.pow(γ, this.optionTau) * reward;
    this.optionTau += 1;
    if (done) this.backup(nextState, true);
  }

  /** Apply the SMDP Q-learning update for the option that just ended. */
  protected backup(sNext: S, terminalEpisode: boolean): void {
    const ω = this.currentOption;
    const sStart = this.optionStartState;
    const sk = this.stateKey(sStart);
    const γ = this.opts.gamma;
    let bootstrap = 0;
    if (!terminalEpisode) {
      const opts = this.options();
      let best = -Infinity;
      const skn = this.stateKey(sNext);
      for (let j = 0; j < opts.length; j++) {
        if (!opts[j].init(sNext)) continue;
        const q = this.Q[skn]?.[j] ?? 0;
        if (q > best) best = q;
      }
      if (Number.isFinite(best)) bootstrap = Math.pow(γ, this.optionTau) * best;
    }
    if (!this.Q[sk]) this.Q[sk] = new Array(this.options().length).fill(0);
    const target = this.optionReward + bootstrap;
    this.Q[sk][ω] += this.opts.alpha * (target - this.Q[sk][ω]);
    this.currentOption = -1;
  }

  protected override endOfEpisode(_episodeId: number): void {
    this.opts.epsilon = Math.max(this.opts.epsilonMin, this.opts.epsilon * this.opts.epsilonDecay);
    this.currentOption = -1;
  }

  // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

  getQ(): readonly number[][] { return this.Q; }
  getEpsilon(): number { return this.opts.epsilon; }
}
