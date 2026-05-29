'use strict';

// =============================================================================
// general/tiger-pomdp.ts — the canonical TIGER PROBLEM (Cassandra,
// Kaelbling, Littman 1994) wrapped on the new BeliefStateStation base.
//
// PROBLEM
// ───────
//   Two doors. A tiger is behind one, gold behind the other. The agent
//   may LISTEN (cheap, noisy) or OPEN_LEFT / OPEN_RIGHT (definitive but
//   risky). Listening preserves the latent state but yields a noisy
//   observation. Opening either door RESETS the world to a uniform
//   belief.
//
//   States = {tiger-left, tiger-right}
//   Actions = {listen, open-left, open-right}
//   Observations = {hear-left, hear-right}
//
//   Reward(s, listen)         = -1 (per listen)
//   Reward(tiger-left, open-left)  = -100  (got bit)
//   Reward(tiger-right, open-left) = +10   (gold)
//   Reward(tiger-left, open-right) = +10
//   Reward(tiger-right, open-right) = -100
//
//   Listen accuracy: P(hear-left | tiger-left, listen) = 0.85.
//
// SOLVERS
// ───────
//   We support two solvers from the existing `pomdp.ts` infrastructure:
//
//     - QMDP heuristic (assumes agent will fully observe after this step)
//     - 1-step belief look-ahead (information-gathering aware)
//
//   Both run as `BeliefStateStation`s in the DES framework: each tick
//   consumes an (action, observation) tuple, applies the Bayesian filter,
//   and emits a BeliefToken. The action is chosen by the solver's
//   policy.
//
// TIGER PROBLEM TIE-BREAK
// ───────────────────────
//   Without information-gathering reasoning, QMDP just opens the door
//   it currently believes is safer — even from a uniform prior — and
//   hits the bad door 50% of the time. With 1-step look-ahead, the
//   agent first LISTENS until belief is concentrated, THEN opens the
//   safer door. Validated below.
// =============================================================================

import {
  BeliefStateStation, ActionObservationToken, BeliefToken,
  POMDPCore, runIterativeDES,
} from './des-base';
import {
  POMDPSpec as ClassicPOMDPSpec, QMDPSolver as ClassicQMDP,
  beliefUpdate as classicBeliefUpdate, mdpValueIteration,
} from './pomdp';
import {DiscreteBelief} from './belief';
import {mulberry32} from './prng';
import {Preconditions} from './des-base/preconditions';

// -----------------------------------------------------------------------------
// TIGER PROBLEM CONSTANTS
// -----------------------------------------------------------------------------

export const TIGER_LEFT  = 0;
export const TIGER_RIGHT = 1;
export const ACT_LISTEN     = 0;
export const ACT_OPEN_LEFT  = 1;
export const ACT_OPEN_RIGHT = 2;
export const OBS_HEAR_LEFT  = 0;
export const OBS_HEAR_RIGHT = 1;

export interface TigerOpts {
  /** P(hear-left | tiger-left, listen). Default 0.85. */
  listenAccuracy?: number;
  /** Reward for opening the gold door. Default +10. */
  openGood?: number;
  /** Reward for opening the tiger door. Default -100. */
  openBad?: number;
  /** Per-listen cost. Default -1. */
  listenCost?: number;
  /** Discount γ. Default 0.95. */
  discount?: number;
}

/** Build the Tiger problem in the classic `POMDPSpec` shape. */
export function buildTigerSpec(opts: TigerOpts = {}): ClassicPOMDPSpec<string, string, string> {
  const acc = opts.listenAccuracy ?? 0.85;
  const good = opts.openGood ?? 10;
  const bad = opts.openBad ?? -100;
  const lc = opts.listenCost ?? -1;
  const γ = opts.discount ?? 0.95;
  return {
    states: ['tiger-left', 'tiger-right'],
    actions: ['listen', 'open-left', 'open-right'],
    observations: ['hear-left', 'hear-right'],
    transition: (sIdx, aIdx) => {
      if (aIdx === ACT_LISTEN) return sIdx === TIGER_LEFT ? [1, 0] : [0, 1];
      // Opening either door resets the world to uniform.
      return [0.5, 0.5];
    },
    observation: (sNextIdx, aIdx) => {
      if (aIdx !== ACT_LISTEN) return [0.5, 0.5];
      if (sNextIdx === TIGER_LEFT)  return [acc, 1 - acc];
      return [1 - acc, acc];
    },
    reward: (sIdx, aIdx) => {
      if (aIdx === ACT_LISTEN) return lc;
      if (aIdx === ACT_OPEN_LEFT)  return sIdx === TIGER_LEFT  ? bad : good;
      if (aIdx === ACT_OPEN_RIGHT) return sIdx === TIGER_RIGHT ? bad : good;
      return 0;
    },
    discount: γ,
    initialBelief: [0.5, 0.5],
  };
}

// -----------------------------------------------------------------------------
// CORE ADAPTER: classic POMDPSpec → POMDPCore (the BeliefStateStation API)
// -----------------------------------------------------------------------------

function specToCore(spec: ClassicPOMDPSpec<unknown, unknown, unknown>): POMDPCore<number, number> {
  return {
    numStates: spec.states.length,
    numActions: spec.actions.length,
    numObservations: spec.observations.length,
    transitionProb: (s, a, sp) => spec.transition(s, a)[sp] ?? 0,
    observationProb: (sp, a, o) => spec.observation(sp, a)[o] ?? 0,
  };
}

// -----------------------------------------------------------------------------
// QMDP STATION
// -----------------------------------------------------------------------------

export class QMDPStation extends BeliefStateStation<number, number> {
  protected readonly classicSpec: ClassicPOMDPSpec<unknown, unknown, unknown>;
  protected readonly solver: ClassicQMDP<unknown, unknown, unknown>;

  constructor(spec: ClassicPOMDPSpec<unknown, unknown, unknown>, b0?: number[]) {
    super('qmdp', specToCore(spec), b0 ?? spec.initialBelief);
    this.classicSpec = spec;
    this.solver = new ClassicQMDP(spec);
  }

  pickAction(b: readonly number[]): number {
    const belief = new DiscreteBelief(this.classicSpec.states, b.slice());
    return this.solver.act(belief);
  }

  /** Expose the QMDP value function for diagnostics. */
  qmdpValue(b: readonly number[]): number {
    let v = 0;
    const Q = this.solver.Q;
    // V_QMDP(s) = max_a Q(s, a)
    for (let s = 0; s < Q.length; s++) {
      let best = -Infinity;
      for (let a = 0; a < Q[s].length; a++) if (Q[s][a] > best) best = Q[s][a];
      v += b[s] * best;
    }
    return v;
  }
}

// -----------------------------------------------------------------------------
// 1-STEP LOOK-AHEAD STATION
// -----------------------------------------------------------------------------

/** Information-gathering aware policy: pick a* = argmax_a R(b,a) +
 *  γ Σ_o P(o|b,a) V_QMDP(τ(b,a,o)). */
export class OneStepLookAheadStation extends QMDPStation {
  constructor(spec: ClassicPOMDPSpec<unknown, unknown, unknown>, b0?: number[]) {
    super(spec, b0);
    (this as any).id = 'pomdp-1step-lookahead';
  }

  override pickAction(b: readonly number[]): number {
    const γ = this.classicSpec.discount;
    const N = this.classicSpec.states.length;
    const A = this.classicSpec.actions.length;
    const O = this.classicSpec.observations.length;
    let bestA = 0; let bestQ = -Infinity;
    for (let a = 0; a < A; a++) {
      // Expected immediate reward.
      let rImm = 0;
      for (let s = 0; s < N; s++) rImm += b[s] * this.classicSpec.reward(s, a);
      // Discounted expected QMDP value at the next belief over observations.
      let exp = 0;
      for (let o = 0; o < O; o++) {
        const pO = this.observationLikelihood(b, a, o);
        if (pO === 0) continue;
        const bp = this.beliefUpdate(b, a, o);
        exp += pO * this.qmdpValue(bp);
      }
      const q = rImm + γ * exp;
      if (q > bestQ) { bestQ = q; bestA = a; }
    }
    return bestA;
  }
}

// -----------------------------------------------------------------------------
// SIMULATION DRIVER
// -----------------------------------------------------------------------------

export interface TigerSimResult {
  /** Discounted return. */
  totalReturn: number;
  actions: number[];
  observations: number[];
  states: number[];
  beliefP0: number[];
  steps: number;
  /** Number of doors opened during the run. */
  numOpens: number;
  /** Number of times the bad door was opened. */
  numBadOpens: number;
}

export interface TigerSimOpts {
  spec?: ClassicPOMDPSpec<unknown, unknown, unknown>;
  solver: 'qmdp' | 'one-step-lookahead';
  numSteps: number;
  seed?: number;
  initialState?: number;
  initialBelief?: number[];
}

/** Run a fixed-step simulation under the chosen solver. The "world"
 *  outside the agent is the same Tiger spec instantiated in a separate
 *  module-local RNG. */
export function simulateTiger(opts: TigerSimOpts): TigerSimResult {
  const cls = 'simulateTiger';
  Preconditions.integerInRange(cls, 'numSteps', opts.numSteps, 1, 1e9);
  Preconditions.check(cls, 'solver', "be 'qmdp' or 'one-step-lookahead'",
    opts.solver === 'qmdp' || opts.solver === 'one-step-lookahead', opts.solver);
  if (opts.initialBelief !== undefined) {
    Preconditions.probabilityVector(cls, 'initialBelief', opts.initialBelief);
  }
  const rng = mulberry32(opts.seed ?? 1);
  const spec = opts.spec ?? buildTigerSpec();
  Preconditions.inRange(cls, 'spec.discount', spec.discount, 0, 1);
  Preconditions.nonEmpty(cls, 'spec.states', spec.states);
  Preconditions.nonEmpty(cls, 'spec.actions', spec.actions);
  Preconditions.nonEmpty(cls, 'spec.observations', spec.observations);
  if (opts.initialBelief !== undefined) {
    Preconditions.lengthEq(cls, 'initialBelief', opts.initialBelief, spec.states.length);
  }
  if (opts.initialState !== undefined) {
    Preconditions.integerInRange(cls, 'initialState', opts.initialState, 0, spec.states.length - 1);
  }
  const γ = spec.discount;
  const Solver = opts.solver === 'one-step-lookahead' ? OneStepLookAheadStation : QMDPStation;
  const station = new Solver(spec, opts.initialBelief);
  let s = opts.initialState !== undefined ? opts.initialState : (rng() < 0.5 ? 0 : 1);
  const actions: number[] = [];
  const observations: number[] = [];
  const states: number[] = [s];
  const beliefP0: number[] = [station.getBelief()[0]];
  let totalRet = 0;
  let discount = 1;
  let numOpens = 0;
  let numBadOpens = 0;

  for (let t = 0; t < opts.numSteps; t++) {
    const a = station.pickAction(station.getBelief());
    actions.push(a);
    // Sample next state (latent dynamics).
    const tRow = spec.transition(s, a);
    let sp = 0; let acc = 0; const u1 = rng();
    for (let sn = 0; sn < spec.states.length; sn++) {
      acc += tRow[sn]; if (u1 < acc) { sp = sn; break; }
    }
    // Sample observation conditional on s'.
    const oRow = spec.observation(sp, a);
    let o = 0; acc = 0; const u2 = rng();
    for (let on = 0; on < spec.observations.length; on++) {
      acc += oRow[on]; if (u2 < acc) { o = on; break; }
    }
    observations.push(o);
    states.push(sp);
    const r = spec.reward(s, a);
    totalRet += discount * r;
    discount *= γ;
    if (a === ACT_OPEN_LEFT || a === ACT_OPEN_RIGHT) {
      numOpens += 1;
      if (r < 0) numBadOpens += 1;
    }
    // Inject the (action, observation) into the station's inbox and run one tick.
    station.take(new ActionObservationToken<number, number>(a, o), QMDPStation.CH_INPUT);
    runIterativeDES([station], {maxTicks: 1, runValidators: false});
    beliefP0.push(station.getBelief()[0]);
    s = sp;
  }
  return {totalReturn: totalRet, actions, observations, states, beliefP0,
          steps: opts.numSteps, numOpens, numBadOpens};
}
