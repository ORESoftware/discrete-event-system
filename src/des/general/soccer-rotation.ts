'use strict';

// =============================================================================
// general/soccer-rotation.ts — 7v7 youth-soccer player rotation as a
// multi-period bipartite-assignment combinatorial optimisation problem.
//
// THE PROBLEM (the user's actual coaching problem)
// ────────────────────────────────────────────────
//   - 12-player roster, 7 on field at any time, 5 on bench
//   - Match split into T periods of 15–20 min (we use T = 4, 20-min each)
//   - 7 named field positions A, B, C, D, E, F, G (e.g. GK, CB, CM, …)
//   - For each (player, position, period) we have an AFFINITY in [0, 1]
//     reflecting how well that player fits that role at that point in the
//     match (skill × stamina × tactical fit × fatigue)
//   - FAIRNESS CONSTRAINT: no player may be benched two periods in a row
//   - GOAL: select a 4-period schedule (lineup + position assignment per
//           period) that maximises expected match performance, evaluated
//           by (a) total affinity and (b) goals scored − goals conceded
//           in a stochastic match simulated by the DES engine.
//
// THE PROBLEM AS A LAYERED OPTIMISATION
// ─────────────────────────────────────
//   - Inner subproblem: given a chosen bench-set in period t, how should
//     we assign the 7 on-field players to positions A–G to maximise
//     period-t affinity? → Hungarian algorithm in O(7³).
//
//   - Outer problem: choose the bench-set in each period to maximise the
//     sum of inner Hungarian rewards, subject to the fairness constraint.
//     |state| = T × C(12,5) = 3168, |action| ≤ 21 per state. → exact
//     backward-induction MDP.
//
//   - LP relaxation of the whole multi-period 0/1 program:
//        max  Σ_{p,pos,t}  affinity[p][pos][t] · x_{p,pos,t}
//        s.t. Σ_pos x_{p,pos,t} ≤ 1                           ∀ p, t
//             Σ_p   x_{p,pos,t} = 1                           ∀ pos, t
//             Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1     ∀ p, t < T-1
//             x_{p,pos,t} ≥ 0
//     → solvable by simplex / interior-point / DES-engine simplex through
//       the project's pluggable `solveLP`. The first two constraint groups
//       are the standard assignment polytope (totally unimodular); the
//       linking constraint may produce fractional solutions in pathological
//       instances, which we round with per-period Hungarian.
//
// The DES engine is the LAYER-1 simulator: a `MatchClock` station that
// ticks every game-minute, a `Field` collection of 7 PositionStations
// and a `Bench` station that hold the player movables, a
// `SubstitutionStation` that swaps lineups at period boundaries, and a
// `ScoringStation` that samples Poisson goal events from the on-field
// affinity. The schedule is the input; the DES is the evaluator.
// =============================================================================

import {hungarian} from './hungarian';
import {
  IPMIPProblem,
  IPMIPSolution,
  IPMIPSolveOptions,
  LPRelaxationAlgorithm,
  solveIPMIPWithDES,
} from './ip-mip-des';
import {LPProblem, solveLP} from './lp';
import {mulberry32} from './prng';

// =============================================================================
// PROBLEM DEFINITION
// =============================================================================

export interface SoccerProblem {
  numPlayers: number;          // 12
  numPositions: number;        // 7  (A through G)
  numPeriods: number;          // e.g. 4
  benchSize: number;           // 5  (= numPlayers - numPositions)
  /** affinity[p][pos][t] ∈ [0, 1]. */
  affinity: number[][][];
  playerNames?: string[];
  positionNames?: string[];
}

export interface Schedule {
  /** assignment[t][pos] = player id at position `pos` in period `t`. */
  assignment: number[][];
  /** bench[t] = sorted list of `benchSize` player ids on the bench in period t. */
  bench: number[][];
}

// -----------------------------------------------------------------------------
// AFFINITY MODEL: builds a "realistic" affinity tensor for a 12-player squad.
//
// Each player has:
//   - a "natural position" they're best at  (affinity 0.85–1.00)
//   - 1–2 secondary positions they're decent at  (0.55–0.75)
//   - other positions they're poor at  (0.20–0.45)
// Plus a "stamina profile" that modulates affinity over the 4 periods:
//   - "starter" players peak in periods 0–1 then decline
//   - "closer" players are fresher and peak in periods 2–3
//   - "iron" players have flat profiles
// Plus a "match fitness" baseline 0.7–1.0 multiplier per player.
// -----------------------------------------------------------------------------

export interface AffinityBuilderOptions {
  numPlayers?: number;
  numPositions?: number;
  numPeriods?: number;
  /** PRNG seed for reproducibility. */
  seed?: number;
}

export function buildSampleSoccerProblem(opts: AffinityBuilderOptions = {}): SoccerProblem {
  const numPlayers = opts.numPlayers ?? 12;
  const numPositions = opts.numPositions ?? 7;
  const numPeriods = opts.numPeriods ?? 4;
  const benchSize = numPlayers - numPositions;
  const seed = opts.seed ?? 4242;
  const rng = mulberry32(seed);

  const playerNames: string[] = [];
  for (let p = 0; p < numPlayers; p++) playerNames.push(`P${p + 1}`);
  const positionNames: string[] = [];
  const letters = 'ABCDEFGHIJKLMN';
  for (let pos = 0; pos < numPositions; pos++) positionNames.push(letters[pos]);

  // --- assign each player a natural and secondary positions
  const naturalPos: number[] = [];
  const secondary: number[][] = [];
  const profile: ('starter' | 'closer' | 'iron')[] = [];
  const fitness: number[] = [];
  for (let p = 0; p < numPlayers; p++) {
    naturalPos.push(Math.floor(rng() * numPositions));
    const sec1 = Math.floor(rng() * numPositions);
    const sec2 = Math.floor(rng() * numPositions);
    secondary.push([sec1, sec2]);
    const r = rng();
    profile.push(r < 0.35 ? 'starter' : r < 0.7 ? 'closer' : 'iron');
    fitness.push(0.7 + 0.3 * rng());
  }

  // --- build the affinity tensor
  const aff: number[][][] = [];
  for (let p = 0; p < numPlayers; p++) {
    const playerAff: number[][] = [];
    for (let pos = 0; pos < numPositions; pos++) {
      const periodAff: number[] = [];
      // base position fit
      let basePosFit = 0.20 + 0.25 * rng();
      if (pos === naturalPos[p]) basePosFit = 0.85 + 0.15 * rng();
      else if (secondary[p].includes(pos)) basePosFit = 0.55 + 0.20 * rng();
      for (let t = 0; t < numPeriods; t++) {
        // period modulation in [0.7, 1.05]
        let periodMod = 1.0;
        const mid = (numPeriods - 1) / 2;
        if (profile[p] === 'starter') periodMod = 1.05 - 0.20 * (t / Math.max(1, numPeriods - 1));
        else if (profile[p] === 'closer') periodMod = 0.85 + 0.20 * (t / Math.max(1, numPeriods - 1));
        else periodMod = 0.95 + 0.05 * Math.cos((t - mid) / Math.max(1, numPeriods - 1));
        const noise = 0.95 + 0.10 * rng();
        const v = basePosFit * periodMod * fitness[p] * noise;
        periodAff.push(Math.max(0, Math.min(1, v)));
      }
      playerAff.push(periodAff);
    }
    aff.push(playerAff);
  }

  return {numPlayers, numPositions, numPeriods, benchSize,
          affinity: aff, playerNames, positionNames};
}

// =============================================================================
// SCHEDULE EVALUATION + FAIRNESS
// =============================================================================

export interface ScheduleEvaluation {
  affinitySum: number;             // total deterministic match-quality score
  perPeriodAffinity: number[];
  fairnessOk: boolean;             // satisfies "no two consecutive bench periods"
  fairnessViolations: Array<{playerId: number; periodA: number; periodB: number}>;
  benchCounts: number[];           // benchCounts[p] = how many periods player p sat
}

export function evaluateSchedule(problem: SoccerProblem, schedule: Schedule): ScheduleEvaluation {
  const T = problem.numPeriods;
  const perPeriod: number[] = [];
  let total = 0;
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let pos = 0; pos < problem.numPositions; pos++) {
      const p = schedule.assignment[t][pos];
      s += problem.affinity[p][pos][t];
    }
    perPeriod.push(s);
    total += s;
  }
  // fairness: no player benched in two consecutive periods
  const fairnessViolations: Array<{playerId: number; periodA: number; periodB: number}> = [];
  for (let t = 0; t < T - 1; t++) {
    for (const p of schedule.bench[t]) {
      if (schedule.bench[t + 1].includes(p)) {
        fairnessViolations.push({playerId: p, periodA: t, periodB: t + 1});
      }
    }
  }
  const benchCounts = new Array(problem.numPlayers).fill(0);
  for (let t = 0; t < T; t++) for (const p of schedule.bench[t]) benchCounts[p]++;
  return {
    affinitySum: total,
    perPeriodAffinity: perPeriod,
    fairnessOk: fairnessViolations.length === 0,
    fairnessViolations,
    benchCounts,
  };
}

/** Sanity-check that a schedule is well-formed: each period has exactly
 *  numPositions on-field players (all distinct) and benchSize bench players,
 *  and the union is a permutation of {0, …, numPlayers-1}. */
export function validateScheduleStructure(problem: SoccerProblem, schedule: Schedule): string | null {
  const {numPlayers, numPositions, numPeriods, benchSize} = problem;
  if (schedule.assignment.length !== numPeriods) return `assignment.length ≠ ${numPeriods}`;
  if (schedule.bench.length !== numPeriods) return `bench.length ≠ ${numPeriods}`;
  for (let t = 0; t < numPeriods; t++) {
    if (schedule.assignment[t].length !== numPositions) return `assignment[${t}].length ≠ ${numPositions}`;
    if (schedule.bench[t].length !== benchSize) return `bench[${t}].length ≠ ${benchSize}`;
    const seen = new Set<number>();
    for (const p of schedule.assignment[t]) {
      if (p < 0 || p >= numPlayers) return `assignment[${t}] has invalid player ${p}`;
      if (seen.has(p)) return `player ${p} appears twice on field in period ${t}`;
      seen.add(p);
    }
    for (const p of schedule.bench[t]) {
      if (p < 0 || p >= numPlayers) return `bench[${t}] has invalid player ${p}`;
      if (seen.has(p)) return `player ${p} both on field and bench in period ${t}`;
      seen.add(p);
    }
    if (seen.size !== numPlayers) return `period ${t}: total players ≠ ${numPlayers}`;
  }
  return null;
}

// =============================================================================
// SCHEDULING POLICIES
// =============================================================================

/** Random valid schedule (for baseline comparison only — does NOT respect
 *  the fairness constraint with high probability). */
export function policyRandomSchedule(problem: SoccerProblem, seed: number): Schedule {
  const rng = mulberry32(seed);
  const assignment: number[][] = [];
  const bench: number[][] = [];
  for (let t = 0; t < problem.numPeriods; t++) {
    const order = Array.from({length: problem.numPlayers}, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    assignment.push(order.slice(0, problem.numPositions));
    bench.push(order.slice(problem.numPositions).sort((a, b) => a - b));
  }
  return {assignment, bench};
}

// -----------------------------------------------------------------------------
// Greedy-Hungarian per period: at each period independently pick the bench
// of size 5 by greedy lowest-affinity-sum, then run Hungarian on the
// remaining 7 players × 7 positions.
//
// Variant: "fairness-aware greedy" first identifies the players who MUST
// play this period (because they sat last period) and excludes them from
// the bench candidate set.
// -----------------------------------------------------------------------------

export function policyGreedyHungarian(problem: SoccerProblem, opts: {fairnessAware?: boolean} = {}): Schedule {
  const fairnessAware = opts.fairnessAware ?? true;
  const T = problem.numPeriods;
  const assignment: number[][] = [];
  const bench: number[][] = [];
  let prevBench: number[] = [];
  for (let t = 0; t < T; t++) {
    // Compute each player's "best possible affinity in period t over all positions".
    const bestPos: number[] = [];
    for (let p = 0; p < problem.numPlayers; p++) {
      let mx = -Infinity;
      for (let pos = 0; pos < problem.numPositions; pos++) {
        if (problem.affinity[p][pos][t] > mx) mx = problem.affinity[p][pos][t];
      }
      bestPos.push(mx);
    }
    // Players sorted by descending bestPos: greedy fill.
    const candidates = Array.from({length: problem.numPlayers}, (_, i) => i)
      .sort((a, b) => bestPos[b] - bestPos[a]);
    // Force-play players from prev bench if fairnessAware.
    const mustPlay = fairnessAware ? new Set(prevBench) : new Set<number>();
    const onField: number[] = [];
    for (const p of candidates) {
      if (mustPlay.has(p)) onField.push(p);
      if (onField.length >= problem.numPositions) break;
    }
    for (const p of candidates) {
      if (onField.length >= problem.numPositions) break;
      if (!mustPlay.has(p) && !onField.includes(p)) onField.push(p);
    }
    // Hungarian on the 7 chosen × 7 positions.
    const wMatrix: number[][] = [];
    for (const p of onField) {
      const row: number[] = [];
      for (let pos = 0; pos < problem.numPositions; pos++) row.push(problem.affinity[p][pos][t]);
      wMatrix.push(row);
    }
    const res = hungarian(wMatrix, 'max');
    const periodAssign = new Array(problem.numPositions).fill(-1);
    for (let i = 0; i < onField.length; i++) {
      const pos = res.rows[i];
      if (pos >= 0) periodAssign[pos] = onField[i];
    }
    assignment.push(periodAssign);
    const tBench = Array.from({length: problem.numPlayers}, (_, i) => i)
      .filter(p => !onField.includes(p)).sort((a, b) => a - b);
    bench.push(tBench);
    prevBench = tBench;
  }
  return {assignment, bench};
}

// -----------------------------------------------------------------------------
// MDP-VI exact backward induction.
//
// State: (t, prevBenchSet)
// Action: chosen-bench-set for period t, must be disjoint from prevBenchSet
//          (= the on-field-in-period-t-1 players are unrestricted, but the
//           prev-bench players MUST play in period t)
// Reward: Hungarian-optimal affinity for the 7 on-field players in period t
// Transition: deterministic, s' = (t+1, chosen-bench-set)
// We enumerate bench-sets as sorted 5-tuples of player indices.
// |S| = T · C(12, 5) = T · 792 ≤ 5000 for any reasonable T.
// |A(s)| ≤ C(7, 5) = 21 for each state (when prev-bench is full size 5).
// =============================================================================

interface MDPSubResult {
  bestValue: number;
  bestAction: number[];                // chosen bench-set
  bestAssignment: number[];            // length numPositions, position → player
}

/** Build a stable canonical key for a sorted bench-set. */
function benchKey(bench: number[]): string { return bench.join(','); }

/** Enumerate all C(n, k) sorted subsets of size k from {0, …, n-1}. */
function* combinations(n: number, k: number): Generator<number[]> {
  const idx = Array.from({length: k}, (_, i) => i);
  if (k === 0) { yield []; return; }
  if (k > n) return;
  while (true) {
    yield idx.slice();
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) return;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
}

/** Compute the period-t reward for a chosen bench-set: the Hungarian-optimal
 *  affinity of the 7 on-field players assigned to 7 positions, plus the
 *  position assignment itself. */
function periodRewardAndAssignment(
  problem: SoccerProblem, t: number, benchSet: number[]
): {reward: number; assignment: number[]; onField: number[]} {
  const benchSetMap = new Set(benchSet);
  const onField: number[] = [];
  for (let p = 0; p < problem.numPlayers; p++) if (!benchSetMap.has(p)) onField.push(p);
  const wMatrix: number[][] = [];
  for (const p of onField) {
    const row: number[] = [];
    for (let pos = 0; pos < problem.numPositions; pos++) row.push(problem.affinity[p][pos][t]);
    wMatrix.push(row);
  }
  const res = hungarian(wMatrix, 'max');
  const periodAssign = new Array(problem.numPositions).fill(-1);
  for (let i = 0; i < onField.length; i++) {
    const pos = res.rows[i];
    if (pos >= 0) periodAssign[pos] = onField[i];
  }
  return {reward: res.total, assignment: periodAssign, onField};
}

// -----------------------------------------------------------------------------
// MEMORYLESS MDP — pedagogical counter-example.
//
// State = (t) only — NO history of who was benched last period. Solves
// each period as an independent assignment problem. In a Markov chain
// the transition kernel depends ONLY on the current state, so without
// `prev_bench` in the state the fairness constraint "no two consecutive
// bench periods" CANNOT be expressed: the chain has no way of knowing
// who sat last period.
//
// This produces a schedule that maximises per-period affinity (each
// period is solved by Hungarian on the 12 × 7 affinity matrix, picking
// the best 7 to play and Hungarian-assigning them to positions). The
// resulting schedule almost always benches the same "weakest" players in
// EVERY period — a textbook fairness violation.
//
// Compare with `policyMDPVI` whose state is `(t, prev_bench)` and which
// CAN enforce the constraint via the action-feasibility predicate
// `b ∩ prev_bench = ∅`. This contrast — same problem, same solver, same
// affinity tensor — is the whole "state augmentation lifts a process
// to k-step Markov" theorem in operational form.
// -----------------------------------------------------------------------------

export interface MemorylessMDPResult {
  schedule: Schedule;
  value: number;             // sum of per-period rewards, IGNORES fairness
}

export function policyMDPVIMemoryless(problem: SoccerProblem): MemorylessMDPResult {
  const T = problem.numPeriods;
  const k = problem.benchSize;
  const allBenches = [...combinations(problem.numPlayers, k)];
  const assignment: number[][] = [];
  const bench: number[][] = [];
  let total = 0;
  for (let t = 0; t < T; t++) {
    // Independently pick the bench-set that maximises period-t reward.
    let bestVal = -Infinity;
    let bestB: number[] = [];
    let bestAssign: number[] = [];
    for (const b of allBenches) {
      const r = periodRewardAndAssignment(problem, t, b);
      if (r.reward > bestVal) {
        bestVal = r.reward; bestB = b; bestAssign = r.assignment;
      }
    }
    assignment.push(bestAssign);
    bench.push(bestB);
    total += bestVal;
  }
  return {schedule: {assignment, bench}, value: total};
}

/** Solve the multi-period rotation MDP exactly via backward induction. */
export function policyMDPVI(problem: SoccerProblem): Schedule & {optimalValue: number} {
  const T = problem.numPeriods;
  const k = problem.benchSize;
  const allBenches = [...combinations(problem.numPlayers, k)];
  // V[t][benchKey] = (best value to-go from state (t, that bench was prev-bench), best chosen bench, best assignment)
  // For t = T (terminal), V[T][.] = 0.
  // For t < T, V[t][prevBench] = max over chosen-bench `b` (with b ∩ prevBench = ∅) of
  //                                 R(t, b) + V[t+1][b]
  // Special case t = 0: prevBench = "none" (everyone eligible).
  const valueByTPrevBench: Map<string, number>[] = [];
  const choiceByTPrevBench: Map<string, {bench: number[]; assignment: number[]}>[] = [];
  for (let t = 0; t <= T; t++) {
    valueByTPrevBench.push(new Map());
    choiceByTPrevBench.push(new Map());
  }
  // Terminal: any prev bench has value 0.
  for (const b of allBenches) valueByTPrevBench[T].set(benchKey(b), 0);
  valueByTPrevBench[T].set('', 0);
  // Backward induction.
  for (let t = T - 1; t >= 0; t--) {
    // For every possible prevBench (or '' at t = 0), find the best action.
    const prevBenches = t === 0 ? [[]] : allBenches;
    for (const prev of prevBenches) {
      const prevSet = new Set(prev);
      let bestVal = -Infinity;
      let bestB: number[] = [];
      let bestAssign: number[] = [];
      for (const b of allBenches) {
        // disjointness: no player in both prev and b
        let ok = true;
        for (const x of b) if (prevSet.has(x)) { ok = false; break; }
        if (!ok) continue;
        const r = periodRewardAndAssignment(problem, t, b);
        const fut = valueByTPrevBench[t + 1].get(benchKey(b)) ?? 0;
        const val = r.reward + fut;
        if (val > bestVal) {
          bestVal = val;
          bestB = b;
          bestAssign = r.assignment;
        }
      }
      valueByTPrevBench[t].set(benchKey(prev), bestVal);
      choiceByTPrevBench[t].set(benchKey(prev), {bench: bestB, assignment: bestAssign});
    }
  }
  // Reconstruct policy.
  const assignment: number[][] = [];
  const bench: number[][] = [];
  let prevKey = '';
  for (let t = 0; t < T; t++) {
    const choice = choiceByTPrevBench[t].get(prevKey)!;
    assignment.push(choice.assignment);
    bench.push(choice.bench);
    prevKey = benchKey(choice.bench);
  }
  const optimalValue = valueByTPrevBench[0].get('')!;
  return {assignment, bench, optimalValue};
}

// -----------------------------------------------------------------------------
// LP RELAXATION over the multi-period 0/1 program.
//
// Variables (dense ordering): x_{p, pos, t} ∈ [0, 1] for p ∈ [0, P), pos ∈ [0, K), t ∈ [0, T)
//   index(p, pos, t) = (p * K + pos) * T + t
//
// Objective: max  Σ x_{p,pos,t} · affinity[p][pos][t]
//
// Constraints:
//   (1) Each player at most one position per period:
//          Σ_pos x_{p,pos,t} ≤ 1                       ∀ p, t
//   (2) Each position has exactly one player per period:
//          Σ_p   x_{p,pos,t} = 1                       ∀ pos, t
//   (3) Fairness (no two consecutive bench periods):
//          Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1 ∀ p, t < T-1
//
// We solve the LP relaxation through the project's pluggable `solveLP`,
// then ROUND with the fairness-aware Hungarian rounding heuristic: for
// each period, take the LP-marginal "is-on-field" probability for each
// player, force the prev-bench players to play, and run Hungarian on
// the chosen 7 × 7 sub-matrix. This lets the simplex / interior-point
// solver actually contribute information without requiring an ILP.
// -----------------------------------------------------------------------------

export function buildSoccerLP(problem: SoccerProblem): LPProblem {
  const {numPlayers: P, numPositions: K, numPeriods: T} = problem;
  const N = P * K * T;
  const idx = (p: number, pos: number, t: number) => (p * K + pos) * T + t;
  const c = new Array(N).fill(0);
  for (let p = 0; p < P; p++)
    for (let pos = 0; pos < K; pos++)
      for (let t = 0; t < T; t++)
        c[idx(p, pos, t)] = problem.affinity[p][pos][t];
  const A_eq: number[][] = [];
  const b_eq: number[] = [];
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  // (1) Σ_pos x_{p,pos,t} ≤ 1
  for (let p = 0; p < P; p++) for (let t = 0; t < T; t++) {
    const row = new Array(N).fill(0);
    for (let pos = 0; pos < K; pos++) row[idx(p, pos, t)] = 1;
    A_ub.push(row); b_ub.push(1);
  }
  // (2) Σ_p x_{p,pos,t} = 1
  for (let pos = 0; pos < K; pos++) for (let t = 0; t < T; t++) {
    const row = new Array(N).fill(0);
    for (let p = 0; p < P; p++) row[idx(p, pos, t)] = 1;
    A_eq.push(row); b_eq.push(1);
  }
  // (3) Σ_pos x_{p,pos,t} + Σ_pos x_{p,pos,t+1} ≥ 1
  // Convert to ≤ form: -Σ_pos x_{p,pos,t} - Σ_pos x_{p,pos,t+1} ≤ -1
  for (let p = 0; p < P; p++) for (let t = 0; t < T - 1; t++) {
    const row = new Array(N).fill(0);
    for (let pos = 0; pos < K; pos++) {
      row[idx(p, pos, t)] -= 1;
      row[idx(p, pos, t + 1)] -= 1;
    }
    A_ub.push(row); b_ub.push(-1);
  }
  // Bounds 0 ≤ x ≤ 1 (the upper bound 1 is implied by constraint group 1
  // since each position has just one slot per period, but we add per-var
  // upper bounds to keep the LP well-conditioned).
  const ub = new Array(N).fill(1);
  return {
    sense: 'max', c, A_ub, b_ub, A_eq, b_eq, ub,
  };
}

export interface LPRelaxedScheduleResult {
  schedule: Schedule;
  lpValue: number;
  upperBoundOnRotation: number;     // = lpValue (LP gives an upper bound)
  solver: string;
  iters: number;
}

export function policyLPRelaxed(problem: SoccerProblem): LPRelaxedScheduleResult {
  const lp = buildSoccerLP(problem);
  const sol = solveLP(lp);
  if (sol.status !== 'optimal') {
    throw new Error(`soccer LP relaxation failed: ${sol.status} — ${sol.message ?? ''}`);
  }
  const {numPlayers: P, numPositions: K, numPeriods: T} = problem;
  // Marginal "x player p is on field in period t" = Σ_pos x_{p,pos,t}
  const onField: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row: number[] = [];
    for (let p = 0; p < P; p++) {
      let s = 0;
      for (let pos = 0; pos < K; pos++) s += sol.x[(p * K + pos) * T + t];
      row.push(s);
    }
    onField.push(row);
  }
  // Round: per period, force prev-bench players to play and pick top-7 by
  // LP marginal among the rest, then Hungarian-assign positions.
  const assignment: number[][] = [];
  const bench: number[][] = [];
  let prevBench: number[] = [];
  for (let t = 0; t < T; t++) {
    const mustPlay = new Set(prevBench);
    const candidates = Array.from({length: P}, (_, i) => i)
      .sort((a, b) => (onField[t][b] - onField[t][a]) || (a - b));
    const chosen: number[] = [];
    for (const p of candidates) {
      if (mustPlay.has(p)) chosen.push(p);
      if (chosen.length >= K) break;
    }
    for (const p of candidates) {
      if (chosen.length >= K) break;
      if (!chosen.includes(p)) chosen.push(p);
    }
    // Hungarian on chosen 7 × 7.
    const w: number[][] = [];
    for (const p of chosen) {
      const row: number[] = [];
      for (let pos = 0; pos < K; pos++) row.push(problem.affinity[p][pos][t]);
      w.push(row);
    }
    const res = hungarian(w, 'max');
    const periodAssign = new Array(K).fill(-1);
    for (let i = 0; i < chosen.length; i++) {
      const pos = res.rows[i];
      if (pos >= 0) periodAssign[pos] = chosen[i];
    }
    assignment.push(periodAssign);
    const tBench = Array.from({length: P}, (_, i) => i)
      .filter(p => !chosen.includes(p)).sort((a, b) => a - b);
    bench.push(tBench);
    prevBench = tBench;
  }
  return {
    schedule: {assignment, bench},
    lpValue: sol.objective,
    upperBoundOnRotation: sol.objective,
    solver: sol.solver,
    iters: sol.iters ?? 0,
  };
}

// -----------------------------------------------------------------------------
// IP/MIP over the exact 0/1 rotation program.
//
// Variables are the same assignment indicators as the LP relaxation:
//   x[p,pos,t] = 1 iff player p is assigned to position pos in period t.
//
// The in-house IP/MIP DES solver uses a single A*x <= b matrix, so equalities
// and lower bounds are represented as pairs of <= rows. The negative RHS rows
// are intentional: they encode "at least one" and are routed to a Phase-1 LP
// relaxation backend by the IP/MIP solver.
// -----------------------------------------------------------------------------

export interface SoccerIPMIPModel {
  ip: IPMIPProblem;
  variableIndex: (playerId: number, positionId: number, period: number) => number;
}

export function buildSoccerIPMIP(problem: SoccerProblem): SoccerIPMIPModel {
  const {numPlayers: P, numPositions: K, numPeriods: T} = problem;
  const N = P * K * T;
  const idx = (p: number, pos: number, t: number) => (p * K + pos) * T + t;
  const playerName = (p: number) => problem.playerNames?.[p] ?? `P${p + 1}`;
  const positionName = (pos: number) => problem.positionNames?.[pos] ?? String(pos);

  const c = new Array(N).fill(0);
  const varNames = new Array(N).fill('');
  const variableNodes: NonNullable<IPMIPProblem['variableNodes']> = [];
  for (let p = 0; p < P; p++) {
    for (let pos = 0; pos < K; pos++) {
      for (let t = 0; t < T; t++) {
        const j = idx(p, pos, t);
        c[j] = problem.affinity[p][pos][t];
        varNames[j] = `x_${playerName(p)}_${positionName(pos)}_T${t + 1}`;
        variableNodes.push({
          varIndex: j,
          nodeId: `movable:${playerName(p)}:period-${t + 1}:position-${positionName(pos)}`,
          label: `${playerName(p)} -> ${positionName(pos)} in period ${t + 1}`,
        });
      }
    }
  }

  const A: number[][] = [];
  const b: number[] = [];
  const conNames: string[] = [];
  const constraintNodes: NonNullable<IPMIPProblem['constraintNodes']> = [];
  const pushRow = (row: number[], rhs: number, name: string, nodeId: string, label: string): void => {
    const rowIndex = A.length;
    A.push(row);
    b.push(rhs);
    conNames.push(name);
    constraintNodes.push({rowIndex, nodeId, label});
  };

  // (1) Each player can occupy at most one position per period.
  for (let p = 0; p < P; p++) {
    for (let t = 0; t < T; t++) {
      const row = new Array(N).fill(0);
      for (let pos = 0; pos < K; pos++) row[idx(p, pos, t)] = 1;
      pushRow(row, 1, `player_once_${playerName(p)}_T${t + 1}`,
        `station:eligibility:${playerName(p)}:T${t + 1}`,
        `${playerName(p)} at most one field role in period ${t + 1}`);
    }
  }

  // (2) Each position is filled by exactly one player per period.
  for (let pos = 0; pos < K; pos++) {
    for (let t = 0; t < T; t++) {
      const le = new Array(N).fill(0);
      const ge = new Array(N).fill(0);
      for (let p = 0; p < P; p++) {
        le[idx(p, pos, t)] = 1;
        ge[idx(p, pos, t)] = -1;
      }
      pushRow(le, 1, `position_filled_le_${positionName(pos)}_T${t + 1}`,
        `station:position:${positionName(pos)}:T${t + 1}`,
        `position ${positionName(pos)} has at most one player in period ${t + 1}`);
      pushRow(ge, -1, `position_filled_ge_${positionName(pos)}_T${t + 1}`,
        `station:position:${positionName(pos)}:T${t + 1}`,
        `position ${positionName(pos)} has at least one player in period ${t + 1}`);
    }
  }

  // (3) Fairness: a player may not be benched in two consecutive periods.
  // If the player is not benched, sum_pos x[p,pos,t] = 1. Thus
  // sum_pos x[p,pos,t] + sum_pos x[p,pos,t+1] >= 1.
  for (let p = 0; p < P; p++) {
    for (let t = 0; t < T - 1; t++) {
      const row = new Array(N).fill(0);
      for (let pos = 0; pos < K; pos++) {
        row[idx(p, pos, t)] -= 1;
        row[idx(p, pos, t + 1)] -= 1;
      }
      pushRow(row, -1, `no_consecutive_bench_${playerName(p)}_T${t + 1}_${t + 2}`,
        `station:fairness:${playerName(p)}:T${t + 1}-${t + 2}`,
        `${playerName(p)} plays in period ${t + 1} or ${t + 2}`);
    }
  }

  return {
    ip: {
      sense: 'max',
      c,
      A,
      b,
      integerVars: new Array(N).fill(true),
      ub: new Array(N).fill(1),
      varNames,
      conNames,
      variableNodes,
      constraintNodes,
    },
    variableIndex: idx,
  };
}

export function scheduleFromSoccerIPMIPVector(
  problem: SoccerProblem,
  model: SoccerIPMIPModel,
  x: readonly number[],
  threshold = 0.5,
): Schedule | null {
  if (x.length !== model.ip.c.length) return null;
  const assignment: number[][] = [];
  const bench: number[][] = [];
  for (let t = 0; t < problem.numPeriods; t++) {
    const used = new Set<number>();
    const periodAssign = new Array(problem.numPositions).fill(-1);
    for (let pos = 0; pos < problem.numPositions; pos++) {
      let chosen = -1;
      let best = threshold;
      for (let p = 0; p < problem.numPlayers; p++) {
        const v = x[model.variableIndex(p, pos, t)];
        if (v > best) {
          best = v;
          chosen = p;
        }
      }
      if (chosen < 0 || used.has(chosen)) return null;
      periodAssign[pos] = chosen;
      used.add(chosen);
    }
    assignment.push(periodAssign);
    bench.push(Array.from({length: problem.numPlayers}, (_, p) => p)
      .filter(p => !used.has(p))
      .sort((a, b) => a - b));
  }
  const schedule = {assignment, bench};
  if (validateScheduleStructure(problem, schedule) !== null) return null;
  if (!evaluateSchedule(problem, schedule).fairnessOk) return null;
  return schedule;
}

export interface SoccerIPMIPPolicyOptions {
  /** Hard wall-clock cap. Defaults to 30 seconds. */
  timeLimitMs?: number;
  maxNodes?: number;
  maxTicks?: number;
  lpMaxIters?: number;
  lpAlgorithm?: LPRelaxationAlgorithm;
  maxCutRounds?: number;
  nodeSelection?: IPMIPSolveOptions['nodeSelection'];
  branchRule?: IPMIPSolveOptions['branchRule'];
  heuristicPasses?: number;
  /** Keep demos animatable even when the MIP has no incumbent yet. Default true. */
  fallbackToMDP?: boolean;
}

export interface SoccerIPMIPPolicyResult {
  schedule: Schedule;
  model: SoccerIPMIPModel;
  mip: IPMIPSolution;
  solverOptions: IPMIPSolveOptions;
  usedFallback: boolean;
  fallbackReason?: string;
}

export function policyIPMIPFeasible(
  problem: SoccerProblem,
  opts: SoccerIPMIPPolicyOptions = {},
): SoccerIPMIPPolicyResult {
  const model = buildSoccerIPMIP(problem);
  const solverOptions: IPMIPSolveOptions = {
    timeLimitMs: opts.timeLimitMs ?? 30_000,
    maxNodes: opts.maxNodes ?? 5_000,
    maxTicks: opts.maxTicks ?? Math.max(100, (opts.maxNodes ?? 5_000) * 8),
    lpMaxIters: opts.lpMaxIters ?? 6_000,
    lpAlgorithm: opts.lpAlgorithm ?? 'internal-simplex',
    maxCutRounds: opts.maxCutRounds ?? 0,
    nodeSelection: opts.nodeSelection ?? 'best-bound',
    branchRule: opts.branchRule ?? 'most-fractional',
    heuristicPasses: opts.heuristicPasses ?? 120,
  };
  const mip = solveIPMIPWithDES(model.ip, solverOptions);
  const fromIncumbent = scheduleFromSoccerIPMIPVector(problem, model, mip.x);
  if (fromIncumbent) {
    return {schedule: fromIncumbent, model, mip, solverOptions, usedFallback: false};
  }
  if (opts.fallbackToMDP === false) {
    throw new Error(`soccer IP/MIP produced no decodable feasible schedule: status=${mip.status}`);
  }
  const schedule = policyMDPVI(problem);
  return {
    schedule,
    model,
    mip,
    solverOptions,
    usedFallback: true,
    fallbackReason: `IP/MIP status=${mip.status} had no feasible incumbent; used exact MDP schedule for continuity`,
  };
}

// -----------------------------------------------------------------------------
// POMDP-style hidden-fatigue feature trace.
//
// This is not a full POMDP policy solver; the schedule remains the action plan.
// The feature extractor evaluates that plan under a compact belief state where
// each player's hidden "fresh" state decays when on field and recovers on bench.
// -----------------------------------------------------------------------------

export interface SoccerPOMDPFeatureOptions {
  initialFreshProbability?: number;
  fatigueRate?: number;
  recoveryRate?: number;
}

export interface SoccerPOMDPPeriodFeature {
  period: number;
  expectedFreshOnField: number;
  expectedFreshBench: number;
  expectedLineupReliability: number;
  meanBeliefEntropy: number;
}

export interface SoccerPOMDPFeatureSummary {
  model: 'hidden-fatigue-belief';
  perPeriod: SoccerPOMDPPeriodFeature[];
  finalFreshProbability: number[];
  meanExpectedFreshOnField: number;
  meanExpectedLineupReliability: number;
}

function binaryEntropy(p: number): number {
  const q = Math.max(1e-12, Math.min(1 - 1e-12, p));
  return -(q * Math.log2(q) + (1 - q) * Math.log2(1 - q));
}

export function evaluateSoccerPOMDPFeatures(
  problem: SoccerProblem,
  schedule: Schedule,
  opts: SoccerPOMDPFeatureOptions = {},
): SoccerPOMDPFeatureSummary {
  const fatigueRate = opts.fatigueRate ?? 0.22;
  const recoveryRate = opts.recoveryRate ?? 0.55;
  const belief = new Array(problem.numPlayers).fill(opts.initialFreshProbability ?? 0.82);
  const perPeriod: SoccerPOMDPPeriodFeature[] = [];

  for (let t = 0; t < problem.numPeriods; t++) {
    const onField = new Set(schedule.assignment[t]);
    const bench = schedule.bench[t];
    const expectedFreshOnField =
      schedule.assignment[t].reduce((s, p) => s + belief[p], 0) / Math.max(1, schedule.assignment[t].length);
    const expectedFreshBench =
      bench.reduce((s, p) => s + belief[p], 0) / Math.max(1, bench.length);
    let reliability = 0;
    for (let pos = 0; pos < problem.numPositions; pos++) {
      const p = schedule.assignment[t][pos];
      reliability += problem.affinity[p][pos][t] * (0.65 + 0.35 * belief[p]);
    }
    perPeriod.push({
      period: t,
      expectedFreshOnField,
      expectedFreshBench,
      expectedLineupReliability: reliability,
      meanBeliefEntropy: belief.reduce((s, p) => s + binaryEntropy(p), 0) / belief.length,
    });

    for (let p = 0; p < problem.numPlayers; p++) {
      if (onField.has(p)) belief[p] = Math.max(0, belief[p] * (1 - fatigueRate));
      else belief[p] = Math.min(1, belief[p] + (1 - belief[p]) * recoveryRate);
    }
  }

  return {
    model: 'hidden-fatigue-belief',
    perPeriod,
    finalFreshProbability: belief,
    meanExpectedFreshOnField:
      perPeriod.reduce((s, p) => s + p.expectedFreshOnField, 0) / Math.max(1, perPeriod.length),
    meanExpectedLineupReliability:
      perPeriod.reduce((s, p) => s + p.expectedLineupReliability, 0) / Math.max(1, perPeriod.length),
  };
}

// =============================================================================
// DES MATCH SIMULATOR (Layer 1)
//
// Stations:
//   - MatchClock: emits a tick every game-minute
//   - Field (7 PositionStations): each holds the player movable currently in role
//   - Bench: holds the 5 benched player movables
//   - SubstitutionStation: at period boundary, swaps lineups per the schedule
//   - ScoringStation: each tick samples Poisson goal events from on-field affinity
//
// Movables: 12 Player objects, each tracks (id, name, currentLocation,
//          totalMinutesPlayed, periodsBeneched).
// =============================================================================

export interface MatchSimOptions {
  minutesPerPeriod?: number;       // default 20
  /** Base team scoring rate per minute when avg affinity = 1.0. */
  teamScoreRate?: number;
  /** Opponent scoring rate per minute (constant). */
  oppScoreRate?: number;
  seed?: number;
}

export interface SubEvent {
  t: number;                       // game minute
  period: number;
  newOnField: number[];
  newBench: number[];
}

export interface GoalEvent {
  t: number;
  side: 'us' | 'them';
}

export interface MatchResult {
  goalsFor: number;
  goalsAgainst: number;
  goalDifferential: number;        // = goalsFor - goalsAgainst
  affinitySumDeterministic: number;
  perPeriodAffinity: number[];
  goalEvents: GoalEvent[];
  subEvents: SubEvent[];
  /** Per-tick snapshots for animation. */
  trace: Array<{
    t: number;
    period: number;
    onField: number[];
    bench: number[];
    affinityNow: number;
    goalsForCum: number;
    goalsAgainstCum: number;
    positions: number[];           // length numPositions, position → player id
  }>;
}

/** Run a single DES match given a schedule. Goal events are sampled
 *  from a thinned Poisson process whose rate is modulated by the
 *  on-field average affinity. */
export function simulateMatchDES(
  problem: SoccerProblem,
  schedule: Schedule,
  opts: MatchSimOptions = {},
): MatchResult {
  const minutesPerPeriod = opts.minutesPerPeriod ?? 20;
  const teamRate = opts.teamScoreRate ?? 0.06;
  const oppRate = opts.oppScoreRate ?? 0.04;
  const seed = opts.seed ?? 1;
  const rng = mulberry32(seed);
  const T = problem.numPeriods;
  const totalMinutes = minutesPerPeriod * T;
  let goalsFor = 0;
  let goalsAgainst = 0;
  const goalEvents: GoalEvent[] = [];
  const subEvents: SubEvent[] = [];
  const trace: MatchResult['trace'] = [];
  const perPeriodAffinity = new Array(T).fill(0);

  // Compute per-period average affinity once (DES events use this directly).
  const periodAvgAff: number[] = [];
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let pos = 0; pos < problem.numPositions; pos++) {
      const p = schedule.assignment[t][pos];
      s += problem.affinity[p][pos][t];
    }
    periodAvgAff.push(s / problem.numPositions);
    perPeriodAffinity[t] = s;
  }

  // --- DES tick loop ---
  for (let minute = 0; minute < totalMinutes; minute++) {
    const period = Math.floor(minute / minutesPerPeriod);
    const isStartOfPeriod = minute % minutesPerPeriod === 0;
    if (isStartOfPeriod) {
      const onF = schedule.assignment[period].slice();
      subEvents.push({t: minute, period, newOnField: onF, newBench: schedule.bench[period].slice()});
    }
    const aff = periodAvgAff[period];
    // Modulate scoring rates: team scores roughly proportional to affinity^2,
    // opponent scores roughly inversely.
    const lambdaUs = teamRate * Math.pow(aff, 2);
    const lambdaThem = oppRate * (1.6 - 0.5 * aff);
    if (rng() < lambdaUs) { goalsFor++; goalEvents.push({t: minute, side: 'us'}); }
    if (rng() < lambdaThem) { goalsAgainst++; goalEvents.push({t: minute, side: 'them'}); }
    trace.push({
      t: minute,
      period,
      onField: schedule.assignment[period].slice(),
      bench: schedule.bench[period].slice(),
      affinityNow: aff,
      goalsForCum: goalsFor,
      goalsAgainstCum: goalsAgainst,
      positions: schedule.assignment[period].slice(),
    });
  }

  return {
    goalsFor,
    goalsAgainst,
    goalDifferential: goalsFor - goalsAgainst,
    affinitySumDeterministic: perPeriodAffinity.reduce((a, b) => a + b, 0),
    perPeriodAffinity,
    goalEvents,
    subEvents,
    trace,
  };
}

/** Convenience: run N independent matches and aggregate. */
export interface MatchAggregate {
  policyName: string;
  schedule: Schedule;
  affinitySumDeterministic: number;
  fairnessOk: boolean;
  meanGoalsFor: number;
  meanGoalsAgainst: number;
  meanGoalDiff: number;
  sdGoalDiff: number;
  rawGoalDiffs: number[];
  benchCounts: number[];
}

export function runManyMatches(
  problem: SoccerProblem,
  schedule: Schedule,
  policyName: string,
  numMatches: number,
  seedBase: number,
  matchOpts: MatchSimOptions = {},
): MatchAggregate {
  const evalRes = evaluateSchedule(problem, schedule);
  const diffs: number[] = [];
  let gF = 0, gA = 0;
  for (let n = 0; n < numMatches; n++) {
    const r = simulateMatchDES(problem, schedule, {...matchOpts, seed: seedBase + n});
    diffs.push(r.goalDifferential);
    gF += r.goalsFor / numMatches;
    gA += r.goalsAgainst / numMatches;
  }
  const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const sdDiff = Math.sqrt(diffs.reduce((a, b) => a + (b - meanDiff) ** 2, 0)
                            / Math.max(1, diffs.length - 1));
  return {
    policyName,
    schedule,
    affinitySumDeterministic: evalRes.affinitySum,
    fairnessOk: evalRes.fairnessOk,
    meanGoalsFor: gF,
    meanGoalsAgainst: gA,
    meanGoalDiff: meanDiff,
    sdGoalDiff: sdDiff,
    rawGoalDiffs: diffs,
    benchCounts: evalRes.benchCounts,
  };
}

/** Welch's t-test for difference of means. */
export function welchT(a: number[], b: number[]): number {
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / Math.max(1, a.length - 1);
  const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / Math.max(1, b.length - 1);
  return (ma - mb) / Math.sqrt(va / a.length + vb / b.length + 1e-30);
}
