'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/blackjack.rs  (module des::general::blackjack)
// 1:1 file move. Blackjack environment solved by on-policy Monte-Carlo control.
//
// Declarations → Rust:
//   fn drawCard / handTotal       -> assoc fns (vanilla helpers; handTotal returns a small struct)
//   class Blackjack (impl Environment) -> struct Blackjack { rng, dealer/player cards } + impl Environment trait
//   interface BlackjackTrainOpts / BlackjackResult -> structs (Default; optionals -> Option<T>)
//   fn runBlackjackMC             -> free fn (or PureTransform<BlackjackTrainOpts, BlackjackResult>)
//   static Blackjack.dealerStickPolicy -> associated fn
//
// Conversion notes (file-specific):
//   - `this.rng = opts.rng ?? Math.random` -> inject a `RandomSource`; default only at the edge.
//     RNG is threaded as a closure `() => number` here; pass `&mut impl Rng` in Rust.
//   - `step` returns an inline object -> a `StepResult { next_state, reward, done }` struct.
//   - `implements Environment` is structural -> write explicit `impl Environment for Blackjack`.
// =============================================================================

// =============================================================================
// general/blackjack.ts — the textbook BLACKJACK environment from
// Sutton & Barto §5.1, solved by ON-POLICY MONTE CARLO CONTROL.
//
// RULES
// ─────
//   - Infinite deck (cards drawn with replacement, P(face) = 4/13,
//     ace = 1 or 11).
//   - Player sees own hand sum + dealer's UPCARD; chooses HIT or STICK.
//   - On STICK, dealer reveals hole card and hits while sum < 17, else
//     stands. Comparison: closer to 21 wins; ties draw; bust = -1.
//   - Reward only at terminal: +1 win, 0 draw, -1 lose.
//
// STATE ENCODING (200 states, the canonical S&B coding)
// ─────────────────────────────────────────────────────
//     player_sum ∈ {12, …, 21}              (10 values; below 12 always hit)
//     dealer_up  ∈ {1, …, 10}               (Ace-up = 1 by convention)
//     usable_ace ∈ {0, 1}
//
//   stateId = (player_sum - 12) * 20 + (dealer_up - 1) * 2 + usable_ace
//   numStates = 10 * 10 * 2 = 200
//   numActions = 2  (0 = STICK, 1 = HIT)
//
// REFERENCE PERFORMANCE
// ─────────────────────
//   The standard rule "always stick on 20+, hit otherwise" earns
//   ≈ -0.27 EV per hand. The MC-optimal policy earns ≈ -0.04 EV. Our
//   500k-episode run reaches ≈ -0.05 ± 0.02 with first-visit MC.
// =============================================================================

import {Environment} from './rl-environments';
import {MonteCarloAgent, EnvironmentStation, runIterativeDES} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// CARD UTILITIES
// -----------------------------------------------------------------------------

/** Draw a single card from an infinite deck. Card values: 1–10 (10's quad-weighted). */
function drawCard(rng: () => number): number {
  const u = Math.floor(rng() * 13) + 1;
  return Math.min(10, u); // 11=J, 12=Q, 13=K all become 10
}

/** Sum a hand interpreting any ace as 11 if it doesn't bust. Returns
 *  {sum, usableAce}. */
function handTotal(cards: readonly number[]): {sum: number; usableAce: boolean} {
  let s = 0; let aces = 0;
  for (const c of cards) {
    if (c === 1) { aces += 1; s += 11; } else s += c;
  }
  while (s > 21 && aces > 0) { s -= 10; aces -= 1; }
  return {sum: s, usableAce: aces > 0};
}

// -----------------------------------------------------------------------------
// ENVIRONMENT
// -----------------------------------------------------------------------------

export class Blackjack implements Environment {
  readonly numStates = 200;
  readonly numActions = 2;
  private readonly rng: () => number;
  /** Internal "deal" cache for the current episode. */
  private dealerCards: number[] = [];
  private playerCards: number[] = [];

  constructor(opts: {rng?: () => number} = {}) {
    this.rng = opts.rng ?? Math.random;
  }

  reset(): number {
    // Deal until player_sum ≥ 12 (below-12 actions are deterministic hit).
    this.playerCards = [drawCard(this.rng), drawCard(this.rng)];
    this.dealerCards = [drawCard(this.rng), drawCard(this.rng)];
    while (handTotal(this.playerCards).sum < 12) this.playerCards.push(drawCard(this.rng));
    return this.encodeState();
  }

  /** Reset using an injected RNG (for reproducibility). */
  resetWithRng(rng: () => number): number {
    this.playerCards = [drawCard(rng), drawCard(rng)];
    this.dealerCards = [drawCard(rng), drawCard(rng)];
    while (handTotal(this.playerCards).sum < 12) this.playerCards.push(drawCard(rng));
    return this.encodeState();
  }

  step(_state: number, action: number): {nextState: number; reward: number; done: boolean} {
    if (action === 1) {
      // HIT: draw a card.
      this.playerCards.push(drawCard(this.rng));
      const t = handTotal(this.playerCards);
      if (t.sum > 21) return {nextState: this.encodeState(true), reward: -1, done: true};
      return {nextState: this.encodeState(), reward: 0, done: false};
    }
    // STICK: dealer plays, then settle.
    const playerSum = handTotal(this.playerCards).sum;
    let dealerSum = handTotal(this.dealerCards).sum;
    while (dealerSum < 17) {
      this.dealerCards.push(drawCard(this.rng));
      dealerSum = handTotal(this.dealerCards).sum;
    }
    let r: number;
    if (dealerSum > 21)              r =  1;
    else if (playerSum > dealerSum)  r =  1;
    else if (playerSum === dealerSum) r = 0;
    else                              r = -1;
    return {nextState: this.encodeState(true), reward: r, done: true};
  }

  /** Compact 200-state encoding. Bust collapses to a sentinel. */
  encodeState(busted = false): number {
    const t = handTotal(this.playerCards);
    if (busted || t.sum > 21) {
      // Map all bust/terminal states to a single dummy index. Doesn't
      // affect Q since updates only fire on visited states, but it must
      // be in [0, numStates). Choose state 0 (player_sum=12, ua=0,
      // dealer=1) — won't ever be VISITED again because episode is done.
      return 0;
    }
    const ps = t.sum;
    const playerIdx = ps - 12;       // 0..9
    const ua = t.usableAce ? 1 : 0;
    const dealerUp = this.dealerCards[0];   // 1..10
    return playerIdx * 20 + (dealerUp - 1) * 2 + ua;
  }

  /** Deterministic STICK-on-20+ baseline policy. */
  static dealerStickPolicy(state: number): number {
    const ps = Math.floor(state / 20) + 12;
    return ps >= 20 ? 0 : 1;
  }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface BlackjackTrainOpts {
  numEpisodes: number;
  seed?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  firstVisit?: boolean;
  gamma?: number;
}

export interface BlackjackResult {
  rewardHistory: readonly number[];
  /** Final mean return over the last `evalEpisodes` greedy episodes. */
  greedyMeanReturn: number;
  /** Baseline mean return over the same eval window (stick on 20+). */
  baselineMeanReturn: number;
  /** Number of (s, a) cells visited at least once during training. */
  visitedCells: number;
  finalEpsilon: number;
  ticks: number;
}

export function runBlackjackMC(
  opts: BlackjackTrainOpts & {evalEpisodes?: number} = {numEpisodes: 100_000},
): BlackjackResult {
  const cls = 'runBlackjackMC';
  Preconditions.integerInRange(cls, 'numEpisodes', opts.numEpisodes, 1, 1e9);
  if (opts.evalEpisodes !== undefined)
    Preconditions.integerInRange(cls, 'evalEpisodes', opts.evalEpisodes, 1, 1e9);
  if (opts.gamma !== undefined) Preconditions.inRange(cls, 'gamma', opts.gamma, 0, 1);
  if (opts.epsilon !== undefined) Preconditions.inRange(cls, 'epsilon', opts.epsilon, 0, 1);
  if (opts.epsilonDecay !== undefined) Preconditions.inRange(cls, 'epsilonDecay', opts.epsilonDecay, 0, 1);
  if (opts.epsilonMin !== undefined) Preconditions.inRange(cls, 'epsilonMin', opts.epsilonMin, 0, 1);
  const rng = mulberry32(opts.seed ?? 1);
  const env = new Blackjack({rng});
  const agent = new MonteCarloAgent('blackjack-mc', {
    rng, numStates: 200, numActions: 2,
    firstVisit: opts.firstVisit ?? true,
    gamma: opts.gamma ?? 1.0,
    epsilon: opts.epsilon ?? 0.1,
    epsilonDecay: opts.epsilonDecay ?? 1,
    epsilonMin: opts.epsilonMin ?? 0.05,
  });
  const envStation = new EnvironmentStation<number, number>('env', env, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: 50,    // a hand never exceeds ~10 hits in practice
  });
  envStation.pipe(agent, EnvironmentStation.CH_STATE, MonteCarloAgent.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, MonteCarloAgent.CH_TRANSITION);
  agent.pipe(envStation, MonteCarloAgent.CH_ACTION, EnvironmentStation.CH_ACTION);
  const summary = runIterativeDES([envStation, agent], {rng});

  // Greedy + baseline evaluation.
  const evalN = opts.evalEpisodes ?? 10_000;
  const evalEnv = new Blackjack({rng});
  let greedyTotal = 0;
  for (let e = 0; e < evalN; e++) {
    let s = evalEnv.resetWithRng(rng);
    let stepCount = 0;
    while (stepCount < 50) {
      const a = agent.greedyAction(s);
      const r = evalEnv.step(s, a);
      stepCount += 1;
      if (r.done) { greedyTotal += r.reward; break; }
      s = r.nextState;
    }
  }
  let baselineTotal = 0;
  const baseEnv = new Blackjack({rng});
  for (let e = 0; e < evalN; e++) {
    let s = baseEnv.resetWithRng(rng);
    let stepCount = 0;
    while (stepCount < 50) {
      const a = Blackjack.dealerStickPolicy(s);
      const r = baseEnv.step(s, a);
      stepCount += 1;
      if (r.done) { baselineTotal += r.reward; break; }
      s = r.nextState;
    }
  }
  let visited = 0;
  const counts = agent.getVisitCounts();
  for (let i = 0; i < counts.length; i++) if (counts[i] > 0) visited += 1;
  return {
    rewardHistory: agent.rewardHistory.slice(),
    greedyMeanReturn: greedyTotal / evalN,
    baselineMeanReturn: baselineTotal / evalN,
    visitedCells: visited,
    finalEpsilon: agent.getEpsilon(),
    ticks: summary.ticks,
  };
}
