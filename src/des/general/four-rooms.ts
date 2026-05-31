// RUST MIGRATION: target module src/des/general/four_rooms.rs.
// RUST MIGRATION: FourRoomsEnv becomes a struct implementing Environment; FourRoomsSMDPAgent becomes a station struct implementing the SemiMDPAgent trait.
// RUST MIGRATION: FourRoomsOpts, FourRoomsTrainOpts, and FourRoomsResult become serde structs; option/action identifiers should use enums or usize newtypes.
// RUST MIGRATION: buildFourRoomsOptions and hallway helpers are free builders; runFourRoomsSMDP is a training PureTransform returning Result.
// RUST MIGRATION: HALLWAY_FIRST_ACTION uses Map/Set today; port to HashMap/HashSet or precomputed Vec<Option<Action>> for deterministic grid indexing.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/four-rooms.rs  (module des::general::four_rooms)
// 1:1 file move. Four-Rooms benchmark + hallway OPTIONS + SMDP Q-learning agent.
//
// Declarations → Rust:
//   interface FourRoomsOpts / FourRoomsTrainOpts / FourRoomsResult -> structs
//   class FourRoomsEnv implements Environment -> struct + `impl Environment`
//   class FourRoomsSMDPAgent extends SemiMDPAgentStation -> struct + impl (base -> trait)
//   const FOUR_ROOMS_MAP/HALLWAYS/GOAL/DR/DC -> `const`/`static` arrays
//   fn buildFourRoomsOptions / makeHallwayOption / rcToIdx/idxToRC/isFree/room -> assoc fns
//
// Conversion notes (file-specific):
//   - `Option<number, number>` here is the RL OPTION type (init/policy/terminate closures),
//     NOT Rust's `Option` — rename to `RlOption`/`SkillOption` and model fields as trait/`Box<dyn Fn>`.
//   - `this.rng = opts.rng ?? Math.random` -> inject `RandomSource`; `() => 0` deterministic eval RNG.
//   - `Map<number,number>` first-action tables -> dense `Vec<usize>` (keys are 0..121) or HashMap.
//   - `Set<number>` ownerRooms -> `HashSet<usize>`; BFS `queue.shift()` -> `VecDeque`.
//   - `(evalAgent as any).Q[i][j] = ...` reaches into private state -> expose a setter; avoid `as any`.
//   - `Infinity` sentinels -> `f64::INFINITY`.
// =============================================================================

// =============================================================================
// general/four-rooms.ts — the canonical FOUR ROOMS benchmark (Sutton,
// Precup, Singh 1999) for SEMI-MDPs / OPTIONS framework.
//
// MAP (11 × 11)
// ─────────────
//
//      0 1 2 3 4 5 6 7 8 9 10
//   0  . . . . . │ . . . . .
//   1  . . . . . │ . . . . .
//   2  . . . . . . . . . . .       ← horizontal corridor (row 2)
//   3  . . . . . │ . . . . .
//   4  . . . . . │ . . . . .
//   5  ─────.───── ─────.─────     ← vertical corridor (col 4 / 5)
//   6  . . . . . │ . . . . .
//   7  . . . . . │ . . . . .
//   8  . . . . . . . . . . .
//   9  . . . . . │ . . . . .
//  10  . . . . . │ . . . . G
//
//   Walls are denoted by │ and ─. There are 4 hallway cells linking
//   the rooms (we follow the standard layout from the original paper:
//   row-2 col-5, col-5 row-5, row-8 col-5, col-5 row-2 — ish; we
//   define them concretely below).
//
//   States = 11 × 11 = 121 minus walls. Actions = {N, E, S, W} with
//   slip probability p_slip (default 0). Reward = +1 at goal,
//   0 otherwise. Episode terminates at goal.
//
// HALLWAY OPTIONS
// ───────────────
//   Eight options (two per room): "go to NORTH/SOUTH hallway" or "go
//   to EAST/WEST hallway", whichever applies. Each option's INTERNAL
//   policy is a hard-coded shortest-path-to-hallway; termination
//   probability β is 1 at the hallway cell and 0 elsewhere.
//
//   With these temporally-extended options, SMDP Q-learning over the
//   8-action OPTION-MDP learns to navigate to the goal in O(few)
//   episodes. With primitive 4-action Q-learning, learning takes 10×
//   longer (commonly cited as the headline result).
// =============================================================================

import {Environment} from './rl-environments';
import {Option, SemiMDPAgentStation, EnvironmentStation, runIterativeDES, scanArgMaxTieBreak} from './des-base';
import {Preconditions} from './des-base/preconditions';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// MAP DEFINITION
// -----------------------------------------------------------------------------

/** 1 = wall, 0 = free. Hallways are free cells embedded in walls. */
const FOUR_ROOMS_MAP: number[][] = (() => {
  const m: number[][] = Array.from({length: 11}, () => new Array(11).fill(0));
  // Vertical wall at col 5 except hallways at row 2 (top half) and row 6 (bottom half).
  for (let r = 0; r < 11; r++) m[r][5] = 1;
  m[2][5] = 0;   // top hallway
  m[6][5] = 0;   // bottom hallway
  // Horizontal wall at row 5 except hallways at col 1 (left half) and col 8 (right half).
  for (let c = 0; c < 11; c++) m[5][c] = 1;
  m[5][1] = 0;
  m[5][8] = 0;
  // Re-clear the corner where two walls cross — the (5, 5) cell.
  m[5][5] = 1;
  return m;
})();

const HALLWAYS: ReadonlyArray<[number, number]> = [
  [2, 5],   // 0: top  vertical hallway
  [6, 5],   // 1: bottom vertical hallway
  [5, 1],   // 2: left   horizontal hallway
  [5, 8],   // 3: right  horizontal hallway
];

const GOAL: [number, number] = [10, 10];

// 0=N, 1=E, 2=S, 3=W
const DR = [-1, 0, 1, 0];
const DC = [0, 1, 0, -1];

function rcToIdx(r: number, c: number): number { return r * 11 + c; }
function idxToRC(i: number): [number, number] { return [Math.floor(i / 11), i % 11]; }
function isFree(r: number, c: number): boolean {
  if (r < 0 || r > 10 || c < 0 || c > 10) return false;
  return FOUR_ROOMS_MAP[r][c] === 0;
}
function room(r: number, c: number): number {
  // 0=NW, 1=NE, 2=SW, 3=SE.
  return (r >= 5 ? 2 : 0) + (c >= 5 ? 1 : 0);
}

// -----------------------------------------------------------------------------
// ENVIRONMENT
// -----------------------------------------------------------------------------

export interface FourRoomsOpts {
  slip?: number;       // probability the actuator slips perpendicularly. Default 0.
  startState?: number; // default: top-left corner (0, 0).
  rng?: () => number;
}

export class FourRoomsEnv implements Environment {
  readonly numStates = 121;
  readonly numActions = 4;
  private readonly slip: number;
  private readonly start: number;
  private readonly rng: () => number;

  constructor(opts: FourRoomsOpts = {}) {
    this.slip = opts.slip ?? 0;
    this.start = opts.startState ?? rcToIdx(0, 0);
    this.rng = opts.rng ?? Math.random;
  }

  reset(): number { return this.start; }

  step(state: number, action: number): {nextState: number; reward: number; done: boolean} {
    const [r, c] = idxToRC(state);
    let aEff = action;
    if (this.slip > 0 && this.rng() < this.slip) {
      aEff = (this.rng() < 0.5 ? (action + 1) % 4 : (action + 3) % 4);
    }
    const nr = r + DR[aEff], nc = c + DC[aEff];
    const nextState = isFree(nr, nc) ? rcToIdx(nr, nc) : state;
    const [gr, gc] = GOAL;
    const done = (nr === gr && nc === gc);
    const reward = done ? 1 : 0;
    return {nextState, reward, done};
  }

  render(state: number): string {
    const [r, c] = idxToRC(state);
    const lines: string[] = [];
    for (let i = 0; i < 11; i++) {
      let line = '';
      for (let j = 0; j < 11; j++) {
        if (i === r && j === c) line += '@';
        else if (FOUR_ROOMS_MAP[i][j] === 1) line += '█';
        else if (i === GOAL[0] && j === GOAL[1]) line += 'G';
        else line += '.';
      }
      lines.push(line);
    }
    return lines.join('\n');
  }
}

// -----------------------------------------------------------------------------
// HALLWAY OPTIONS
// -----------------------------------------------------------------------------

/** Pre-compute first-step-action lookup tables via BFS to each
 *  hallway. For each hallway we run BFS backwards from the hallway
 *  cell, then for every reachable free cell we record which of the
 *  four primitive actions takes one step closer. This is the
 *  RIGHT WAY to do hallway options on a non-trivial grid: we never
 *  hit a wall, never need a fallback, and the option always
 *  terminates at the hallway. */
const HALLWAY_FIRST_ACTION: ReadonlyArray<Map<number, number>> = HALLWAYS.map(([hr, hc]) => {
  const dist: number[] = new Array(121).fill(Infinity);
  const parent: number[] = new Array(121).fill(-1);
  const start = rcToIdx(hr, hc);
  dist[start] = 0;
  const queue: number[] = [start];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    const [r, c] = idxToRC(cur);
    for (let a = 0; a < 4; a++) {
      const nr = r + DR[a], nc = c + DC[a];
      if (!isFree(nr, nc)) continue;
      const ni = rcToIdx(nr, nc);
      if (dist[ni] === Infinity) {
        dist[ni] = dist[cur] + 1;
        parent[ni] = cur;
        queue.push(ni);
      }
    }
  }
  // For each cell with parent[cell] valid, infer the primitive action
  // that goes from cell → parent[cell] (one step closer to hallway).
  const m = new Map<number, number>();
  for (let s = 0; s < 121; s++) {
    if (s === start) { m.set(s, 0); continue; }
    if (parent[s] < 0) continue;
    const [sr, sc] = idxToRC(s);
    const [pr, pc] = idxToRC(parent[s]);
    const dr = pr - sr, dc = pc - sc;
    let a = 0;
    if (dr === -1) a = 0; else if (dc === 1) a = 1;
    else if (dr === 1) a = 2; else if (dc === -1) a = 3;
    m.set(s, a);
  }
  return m;
});

function makeHallwayOption(name: string, hallwayIdx: number, ownerRooms: ReadonlySet<number>): Option<number, number> {
  const [hr, hc] = HALLWAYS[hallwayIdx];
  const lookup = HALLWAY_FIRST_ACTION[hallwayIdx];
  return {
    name,
    init: (s) => ownerRooms.has(room(...idxToRC(s))) || (idxToRC(s)[0] === hr && idxToRC(s)[1] === hc),
    policy: (s, _rng) => lookup.get(s) ?? 0,
    terminate: (s) => {
      const [r, c] = idxToRC(s);
      if (r === hr && c === hc) return 1;
      if (r === GOAL[0] && c === GOAL[1]) return 1;
      if (!ownerRooms.has(room(r, c))) return 1;
      return 0;
    },
  };
}

/** Eight options: each room has the two adjacent hallways as targets,
 *  plus one "primitive" option per direction for completeness. */
export function buildFourRoomsOptions(includePrimitive = true): Option<number, number>[] {
  const opts: Option<number, number>[] = [];
  // Hallway options.
  opts.push(makeHallwayOption('NW→top',    0, new Set([0])));
  opts.push(makeHallwayOption('NW→left',   2, new Set([0])));
  opts.push(makeHallwayOption('NE→top',    0, new Set([1])));
  opts.push(makeHallwayOption('NE→right',  3, new Set([1])));
  opts.push(makeHallwayOption('SW→left',   2, new Set([2])));
  opts.push(makeHallwayOption('SW→bottom', 1, new Set([2])));
  opts.push(makeHallwayOption('SE→right',  3, new Set([3])));
  opts.push(makeHallwayOption('SE→bottom', 1, new Set([3])));
  if (includePrimitive) {
    for (let a = 0; a < 4; a++) {
      const aFinal = a;
      opts.push({
        name: `prim-${['N', 'E', 'S', 'W'][a]}`,
        init: () => true,
        policy: () => aFinal,
        terminate: () => 1,   // primitive options terminate after 1 step
      });
    }
  }
  return opts;
}

// -----------------------------------------------------------------------------
// SMDP Q-LEARNING AGENT
// -----------------------------------------------------------------------------

class FourRoomsSMDPAgent extends SemiMDPAgentStation<number, number> {
  private readonly _options: Option<number, number>[];

  constructor(opts: {rng: () => number; alpha: number; gamma: number; epsilon: number;
                     epsilonDecay: number; epsilonMin: number; options: Option<number, number>[];
                     initQ?: number}) {
    super('four-rooms-smdp', {
      rng: opts.rng, alpha: opts.alpha, gamma: opts.gamma,
      epsilon: opts.epsilon, epsilonDecay: opts.epsilonDecay, epsilonMin: opts.epsilonMin,
    });
    this._options = opts.options;
    const init = opts.initQ ?? 1.0;       // optimistic init: drive exploration
    for (let s = 0; s < 121; s++) this.Q[s] = new Array(this._options.length).fill(init);
  }
  protected options(): readonly Option<number, number>[] { return this._options; }
  protected stateKey(s: number): number { return s; }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

export interface FourRoomsTrainOpts {
  numEpisodes: number;
  maxStepsPerEpisode?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  seed?: number;
  /** Slip probability of the env. Default 0. */
  slip?: number;
  /** Include primitive 4-direction options. Default true. */
  includePrimitive?: boolean;
  /** Optimistic Q initial value. Default 1.0 (helps drive exploration). */
  initQ?: number;
}

export interface FourRoomsResult {
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  /** Number of steps used by the GREEDY policy from start to goal
   *  (Infinity if it never reaches the goal within maxStepsPerEpisode). */
  greedyEpisodeLength: number;
  greedyReachedGoal: boolean;
  finalEpsilon: number;
  ticks: number;
}

export function runFourRoomsSMDP(opts: FourRoomsTrainOpts): FourRoomsResult {
  // Pre-run guards.
  const cls = 'runFourRoomsSMDP';
  Preconditions.integerInRange(cls, 'numEpisodes', opts.numEpisodes, 1, 1e9);
  if (opts.maxStepsPerEpisode !== undefined)
    Preconditions.integerInRange(cls, 'maxStepsPerEpisode', opts.maxStepsPerEpisode, 1, 1e9);
  if (opts.alpha !== undefined) Preconditions.positive(cls, 'alpha', opts.alpha);
  if (opts.gamma !== undefined) Preconditions.inRange(cls, 'gamma', opts.gamma, 0, 1);
  if (opts.epsilon !== undefined) Preconditions.inRange(cls, 'epsilon', opts.epsilon, 0, 1);
  if (opts.epsilonDecay !== undefined) Preconditions.inRange(cls, 'epsilonDecay', opts.epsilonDecay, 0, 1);
  if (opts.epsilonMin !== undefined) Preconditions.inRange(cls, 'epsilonMin', opts.epsilonMin, 0, 1);
  if (opts.slip !== undefined) Preconditions.inRange(cls, 'slip', opts.slip, 0, 1);
  if (opts.initQ !== undefined) Preconditions.finite(cls, 'initQ', opts.initQ);
  const rng = mulberry32(opts.seed ?? 1);
  const env = new FourRoomsEnv({slip: opts.slip ?? 0, rng});
  const options = buildFourRoomsOptions(opts.includePrimitive ?? true);
  const agent = new FourRoomsSMDPAgent({
    rng, alpha: opts.alpha ?? 0.25, gamma: opts.gamma ?? 0.99,
    epsilon: opts.epsilon ?? 0.1,
    epsilonDecay: opts.epsilonDecay ?? 1, epsilonMin: opts.epsilonMin ?? 0.01,
    options, initQ: opts.initQ ?? 1.0,
  });
  const envStation = new EnvironmentStation<number, number>('env', env, {
    numEpisodes: opts.numEpisodes,
    maxStepsPerEpisode: opts.maxStepsPerEpisode ?? 5000,
  });
  envStation.pipe(agent, EnvironmentStation.CH_STATE, FourRoomsSMDPAgent.CH_STATE);
  envStation.pipe(agent, EnvironmentStation.CH_TRANSITION, FourRoomsSMDPAgent.CH_TRANSITION);
  agent.pipe(envStation, FourRoomsSMDPAgent.CH_ACTION, EnvironmentStation.CH_ACTION);
  const summary = runIterativeDES([envStation, agent], {rng});

  // Greedy rollout from start.
  const evalEnv = new FourRoomsEnv({slip: 0, rng});
  let s = evalEnv.reset();
  let len = 0;
  let reached = false;
  const evalAgent = new FourRoomsSMDPAgent({
    rng: () => 0,  // deterministic — never take ε branch
    alpha: 0, gamma: opts.gamma ?? 0.99,
    epsilon: 0,
    epsilonDecay: 1, epsilonMin: 0,
    options,
  });
  // Copy the trained Q.
  const Qtrained = agent.getQ();
  for (let i = 0; i < 121; i++) for (let j = 0; j < options.length; j++) {
    (evalAgent as any).Q[i][j] = Qtrained[i][j];
  }
  const max = opts.maxStepsPerEpisode ?? 5000;
  let curOption = -1;
  const evalRng = mulberry32(((opts.seed ?? 1) + 17) >>> 0);
  for (let t = 0; t < max; t++) {
    if (curOption < 0 || options[curOption].terminate(s) >= 1) {
      // Pick the greedy option (random tie-breaking so symmetric Q-rows
      // don't always collapse to option 0).
      curOption = scanArgMaxTieBreak(options.length,
        i => options[i].init(s) ? Qtrained[s][i] : -Infinity,
        evalRng);
      if (curOption < 0) curOption = 0;
    }
    const a = options[curOption].policy(s, () => 0);
    const r = evalEnv.step(s, a);
    len += 1;
    if (r.done) { reached = true; break; }
    s = r.nextState;
  }

  return {
    rewardHistory: agent.rewardHistory.slice(),
    lengthHistory: agent.lengthHistory.slice(),
    greedyEpisodeLength: reached ? len : Infinity,
    greedyReachedGoal: reached,
    finalEpsilon: agent.getEpsilon(),
    ticks: summary.ticks,
  };
}
