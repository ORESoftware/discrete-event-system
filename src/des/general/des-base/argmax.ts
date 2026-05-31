'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/argmax.rs
// - Keep file-for-file. ARGMAX_EPS_DEFAULT becomes a pub const and the argmax
//   routines can stay as pure module functions returning indices/options.
// - Replace rng callbacks with an injected RNG trait or generic R: Rng, keeping
//   deterministic tie-breaking explicit at call sites.
// - These helpers are not DES stations today; if an argmax scorer is lifted into
//   the graph, wrap it in a PureTransform/PureTransformEntity with transform().
// - Prefer Result/Option for empty inputs or invalid scores instead of throws.

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/argmax.rs  (module des::general::des_base::argmax)
// 1:1 file move. Random-tie-breaking argmax helpers for greedy value selection.
//
// Declarations → Rust:
//   const ARGMAX_EPS_DEFAULT      -> const ARGMAX_EPS_DEFAULT: f64 = 1e-12;
//   fn argMaxWithTieBreak         -> fn arg_max_with_tie_break(&[f64], &mut dyn RandomSource, eps) -> isize
//   fn scanArgMaxTieBreak         -> fn scan_arg_max_tie_break(n, impl Fn(usize)->f64, rng, eps) -> isize
//   fn chooseRandomTied<T>        -> fn choose_random_tied<T>(&[T], rng) -> Option<&T>
//   fn allArgMaxTies              -> fn all_arg_max_ties(&[f64], eps) -> Vec<usize>
//
// Conversion notes (file-specific):
//   - `rng: () => number` -> inject `RandomSource`/`rand::Rng` (shared/capabilities);
//     reservoir tie-break must use the SAME seeded source for reproducibility.
//   - Sentinel `-1` for "no winner" -> return `Option<usize>` (preferred) rather
//     than `isize` with -1.
//   - `values: ArrayLike<number>` -> `&[f64]`.
//   - Pure deterministic given the rng; no I/O.
// =============================================================================

// =============================================================================
// general/des-base/argmax.ts
//
// Random-tie-breaking argmax for value-based decision making.
//
// Many algorithms in the codebase (value iteration, Q-learning, finite-horizon
// DP, linear VFA, semi-MDP, MCTS, MILP branching, POMDP QMDP, etc.) extract a
// greedy decision via:
//
//     let bestA = -1, bestQ = -Infinity;
//     for (let a = 0; a < A; a++) {
//       const q = score(a);
//       if (q > bestQ) { bestQ = q; bestA = a; }    // ← strict > = first wins
//     }
//
// Strict `>` means whichever action is encountered FIRST with the maximum
// value wins. When ties exist — and they routinely do (initial Q=0 across
// all actions, symmetric MDPs, identical UCT for fresh nodes, MILP branching
// at the root) — the algorithm has a permanent bias toward low-index
// actions. That bias hurts:
//
//   1. Exploration. With ε=0 greedy you NEVER try the other equally-good
//      actions, so you can never break out of a symmetric initialization.
//   2. Convergence speed on symmetric problems (2-10× slower).
//   3. Reproducibility against textbook / paper numbers, which generally
//      assume uniform random tie-breaking.
//   4. Robustness: the algorithm shouldn't depend on the order in which
//      the user happened to enumerate the actions.
//
// The fix is conceptually a one-shot Fisher-Yates over the tied set: collect
// every index whose score matches the best (within `eps`), then sample one
// uniformly. This module provides three flavors:
//
//   - argMaxWithTieBreak(values, rng)      — array of scores → winning index
//   - scanArgMaxTieBreak(A, score, rng)    — A actions, scored lazily
//   - chooseRandomTied(candidates, rng)    — explicit collected ties → pick
//
// All three respect a small `eps` (1e-12 by default) when comparing floats
// so that genuine numerical drift doesn't accidentally break ties.
//
// IMPORTANT: random tie-breaking is reproducible — pass the same seeded
// rng() callback every run and you'll get the same trajectory.
// =============================================================================

/** Default float-comparison epsilon. Two scores within ±eps are tied. */
export const ARGMAX_EPS_DEFAULT = 1e-12;

/**
 * Return the index of the maximum value in `values`, breaking ties uniformly
 * at random via `rng()`.
 *
 * If `values` is empty, returns -1.
 *
 * Treats two scores as tied if |a - b| <= eps. This is important because
 * many of our value updates accumulate floating-point error (eg. value
 * iteration's Bellman backup), so an exact `==` would miss legitimate ties.
 */
export function argMaxWithTieBreak(
  values: ArrayLike<number>,
  rng: () => number,
  eps: number = ARGMAX_EPS_DEFAULT,
): number {
  const n = values.length;
  if (n === 0) return -1;
  if (n === 1) return 0;

  let best = values[0];
  let tieCount = 1;
  let winner = 0;

  for (let i = 1; i < n; i++) {
    const v = values[i];
    if (v > best + eps) {
      best = v;
      winner = i;
      tieCount = 1;
    } else if (v >= best - eps) {
      // tie: reservoir-sample uniformly across all ties seen so far.
      tieCount++;
      if (rng() * tieCount < 1) winner = i;
    }
  }
  return winner;
}

/**
 * Same as `argMaxWithTieBreak` but the scores are produced on-the-fly by a
 * scoring function. Avoids materializing the full score array when only the
 * argmax is needed.
 *
 * `score(a)` may return -Infinity for actions that should be excluded; those
 * are never selected as winners (unless ALL actions return -Infinity, in
 * which case the result is -1).
 *
 * @returns the winning index in [0, n), or -1 if no action has a finite score.
 */
export function scanArgMaxTieBreak(
  n: number,
  score: (a: number) => number,
  rng: () => number,
  eps: number = ARGMAX_EPS_DEFAULT,
): number {
  if (n <= 0) return -1;

  let best = -Infinity;
  let tieCount = 0;
  let winner = -1;

  for (let a = 0; a < n; a++) {
    const v = score(a);
    if (!Number.isFinite(v)) continue;
    if (winner === -1 || v > best + eps) {
      best = v;
      winner = a;
      tieCount = 1;
    } else if (v >= best - eps) {
      tieCount++;
      if (rng() * tieCount < 1) winner = a;
    }
  }
  return winner;
}

/**
 * Given an array of candidate indices that are already known to be tied for
 * the best, pick one uniformly at random. Used by algorithms that build the
 * tied set explicitly (e.g. MCTS collecting children with identical UCT,
 * MILP B&B collecting variables with identical fractional score).
 *
 * Returns -1 if `candidates` is empty.
 */
export function chooseRandomTied<T>(
  candidates: readonly T[],
  rng: () => number,
): T | undefined {
  if (candidates.length === 0) return undefined;
  if (candidates.length === 1) return candidates[0];
  return candidates[Math.floor(rng() * candidates.length)];
}

/**
 * Convenience: return ALL indices tied for the maximum (within eps).
 * Useful when an algorithm needs to know "how many ways are there to act
 * optimally?" — e.g. when computing a softmax over the tied set, or when
 * spawning multiple parallel rollouts.
 */
export function allArgMaxTies(
  values: ArrayLike<number>,
  eps: number = ARGMAX_EPS_DEFAULT,
): number[] {
  const n = values.length;
  if (n === 0) return [];
  let best = values[0];
  for (let i = 1; i < n; i++) if (values[i] > best) best = values[i];
  const ties: number[] = [];
  for (let i = 0; i < n; i++) if (values[i] >= best - eps) ties.push(i);
  return ties;
}
