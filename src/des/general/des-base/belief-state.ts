'use strict';

// =============================================================================
// general/des-base/belief-state.ts — base class for BELIEF-STATE iterative
// algorithms (POMDPs, hidden Markov filtering, simultaneous localisation
// and mapping, dual control, …).
//
// PROBLEM SHAPE
// ─────────────
//   The agent maintains a belief b ∈ Δ(S) over latent state space S.
//   Every interaction with the environment supplies an action a and an
//   observation o; the belief evolves via
//
//     b_{k+1}(s') = (η · O(o|s', a)) · Σ_s T(s'|s, a) b_k(s)
//
//   This base class wraps the belief representation + the Bayesian
//   filter step + a `step(action, observation)` API that any POMDP /
//   filtering algorithm uses. Concrete algorithms (QMDP look-ahead,
//   point-based VI, particle filtering) plug in by extending this and
//   adding their planning logic.
//
// AS A DES STATION
// ────────────────
//   `runTimeStep()` consumes the latest (action, observation) tuple
//   from the inboxes and updates the belief in place; emits the new
//   belief on the OUT channel for downstream stations.
//
// The "tick" of this station is one belief update — the granularity
// at which a real-world tracker would refresh its posterior estimate.
// =============================================================================

import {DESStation, ChannelName, Token} from './station';

export class ActionObservationToken<A = number, O = number> implements Token {
  constructor(public readonly action: A, public readonly observation: O) {}
}

export class BeliefToken implements Token {
  constructor(public readonly belief: readonly number[]) {}
}

export interface POMDPCore<A = number, O = number> {
  /** |S|. */
  numStates: number;
  /** |A|. */
  numActions: number;
  /** |Ω|. */
  numObservations: number;
  /** T(s, a, s') as a function (avoids forcing dense arrays for big S). */
  transitionProb(s: number, a: A, sp: number): number;
  /** O(s', a, o) — observation likelihood after taking a, ending in s'. */
  observationProb(sp: number, a: A, o: O): number;
}

export abstract class BeliefStateStation<A = number, O = number> extends DESStation {
  static readonly CH_INPUT: ChannelName = 'ao';
  static readonly CH_BELIEF: ChannelName = 'belief';

  protected belief: number[];
  protected readonly core: POMDPCore<A, O>;
  /** Trace of beliefs (always recorded). */
  readonly beliefHistory: number[][] = [];

  constructor(id: string, core: POMDPCore<A, O>, initialBelief?: readonly number[]) {
    super(id);
    this.core = core;
    if (initialBelief) {
      if (initialBelief.length !== core.numStates)
        throw new Error(`initial belief length ${initialBelief.length} != numStates ${core.numStates}`);
      this.belief = initialBelief.slice();
    } else {
      this.belief = new Array(core.numStates).fill(1 / core.numStates);
    }
    this.beliefHistory.push(this.belief.slice());
  }

  override hasWork(): boolean { return this.inboxSize(BeliefStateStation.CH_INPUT) > 0; }

  runTimeStep(): void {
    const tokens = this.drain<ActionObservationToken<A, O>>(BeliefStateStation.CH_INPUT);
    for (const t of tokens) {
      this.belief = this.beliefUpdate(this.belief, t.action, t.observation);
      this.beliefHistory.push(this.belief.slice());
      this.emit(new BeliefToken(this.belief), BeliefStateStation.CH_BELIEF);
    }
  }

  /** Bayesian belief update b' = η · O(o|s',a) Σ_s T(s'|s,a) b(s). */
  beliefUpdate(b: readonly number[], a: A, o: O): number[] {
    const N = this.core.numStates;
    const bp = new Array<number>(N).fill(0);
    let total = 0;
    for (let sp = 0; sp < N; sp++) {
      let pTrans = 0;
      for (let s = 0; s < N; s++) pTrans += this.core.transitionProb(s, a, sp) * b[s];
      const v = this.core.observationProb(sp, a, o) * pTrans;
      bp[sp] = v;
      total += v;
    }
    if (total > 0) for (let i = 0; i < N; i++) bp[i] /= total;
    else for (let i = 0; i < N; i++) bp[i] = 1 / N;
    return bp;
  }

  /** P(o | b, a). */
  observationLikelihood(b: readonly number[], a: A, o: O): number {
    let total = 0;
    for (let sp = 0; sp < this.core.numStates; sp++) {
      let pTrans = 0;
      for (let s = 0; s < this.core.numStates; s++) pTrans += this.core.transitionProb(s, a, sp) * b[s];
      total += this.core.observationProb(sp, a, o) * pTrans;
    }
    return total;
  }

  getBelief(): readonly number[] { return this.belief; }
  setBelief(b: readonly number[]): void {
    if (b.length !== this.core.numStates)
      throw new Error(`belief length ${b.length} != numStates ${this.core.numStates}`);
    this.belief = b.slice();
  }

  /** Subclasses MUST implement: pick the next action given the current
   *  belief. This is the "policy" half of any POMDP algorithm. */
  abstract pickAction(b: readonly number[]): A;
}
