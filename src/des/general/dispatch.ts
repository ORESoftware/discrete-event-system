'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/dispatch.rs  (module des::general::dispatch)
// 1:1 file move. Multi-class parallel-server dispatch problem + six policies + evaluation.
//
// Declarations → Rust:
//   interface DispatchProblem/Policy/State/PendingJob/Result/FluidLPPolicyResult/
//             MDPVIPolicyOptions/Result/MCTSPolicyOptions/EvaluationResult -> structs
//   interface DispatchPolicy      -> trait DispatchPolicy { fn choose(&self, state) -> usize }
//   fn simulateDispatch / evaluatePolicy -> free fns (or StatefulTransform for the sim)
//   fn policyRandom/RoundRobin/ShortestQueue/SECT/FluidLP/MDPVI/MCTS -> constructors returning
//                                    distinct structs that `impl DispatchPolicy`
//   fn expSample / categorical / welchT / buildDispatchFluidLP -> assoc fns
//
// Conversion notes (file-specific):
//   - Policies are objects implementing DispatchPolicy -> one struct per policy + `impl` (no closures-as-objects).
//   - `mulberry32(seed)` + `expSample`/`categorical(rng)` -> inject `RandomSource`; same seed = fair compare.
//   - Builds on lp/value-iteration/mcts (see their headers); priority queue of jobs -> `BinaryHeap`.
// =============================================================================

// =============================================================================
// general/dispatch.ts — multi-class parallel-server dispatch problem.
//
// THE PROBLEM (canonical "size-and-skill" parallel-server)
// ────────────────────────────────────────────────────────
//   - M heterogeneous machines, K job classes
//   - Arrivals: Poisson(λ); each arrival independently has class c with
//     probability p_c (Σ p_c = 1)
//   - Service: a class-c job on machine m takes Exp(μ_{c,m}); machines
//     are heterogeneous, so the dispatch decision is genuinely
//     COMBINATORIAL — no single machine dominates for all classes
//   - Decision at every arrival: which machine to dispatch to?
//   - Cost: long-run mean SOJOURN time (waiting + service)
//
// This problem is the smallest interesting instance of the
// architectural pattern in the user's question:
//
//   Layer 1 (physical / dynamic world)  → DES simulator (this file)
//   Layer 2 (decision abstraction)      → MDP at arrival decision epochs
//   Layer 3 (optimisation engine)       → simplex / interior-point on
//                                          the LP fluid relaxation,
//                                          value iteration on the
//                                          truncated MDP, MCTS on
//                                          the DES tree, or any
//                                          metaheuristic / RL.
//
// Six policies live here: random, round-robin, shortest-queue (SQ),
// shortest-expected-completion-time (SECT), fluid-LP randomized,
// MDP-via-VI on the empirical transition kernel, and MCTS with
// SECT rollouts. Every policy is evaluated by the SAME DES, with
// the same seeds, so head-to-head comparisons are fair.
//
// Validation: see runners/validate-dispatch.ts. The expected ordering
// (lower mean sojourn = better) is
//
//   MDP-VI ≤ MCTS ≤ Fluid-LP ≤ SECT ≤ SQ ≤ Round-robin ≤ Random
//
// with Welch-t > 3 between adjacent layers on enough replications.
// =============================================================================

import {LPProblem, solveLP} from './lp';
import {mulberry32} from './prng';
import {valueIteration, MDPSpec} from './value-iteration';
import {mcts} from './mcts';
import {PureTransform} from '../shared/transform';

export interface DispatchProblem {
  M: number;                            // # machines
  K: number;                            // # classes
  arrivalRate: number;                  // λ
  classProb: number[];                  // length K, sums to 1
  serviceRate: number[][];              // K × M matrix of μ_{c,m}
}

/** Exponential sample with given rate. */
function expSample(rate: number, rng: () => number): number {
  return -Math.log(Math.max(1e-12, rng())) / rate;
}

/** Discrete sample from a probability vector. */
function categorical(p: number[], rng: () => number): number {
  const u = rng();
  let cum = 0;
  for (let i = 0; i < p.length; i++) { cum += p[i]; if (u < cum) return i; }
  return p.length - 1;
}

// -----------------------------------------------------------------------------
// DES SIMULATOR — discrete-event next-event-time simulation.
//
// State at any time t:
//   - per-machine FIFO queue of (arrivalTime, class) records
//   - per-machine "idle until" timestamp (when the current head will finish)
// Events:
//   - arrival at rate λ → policy decides which machine to dispatch to,
//     job enters that machine's queue; if machine was idle, it starts
//     immediately
//   - service completion → record sojourn time, dequeue the head; if the
//     queue still has jobs, start the next one and schedule its
//     completion
//
// We run for a fixed number of completed arrivals (`numArrivals`), then
// return the mean sojourn over the completed jobs that finished.
// -----------------------------------------------------------------------------

export interface DispatchPolicy {
  /** Choose machine ∈ [0, M) for an arriving class-c job, given the
   *  current per-machine queue lengths. Optionally, the policy can
   *  read the per-machine "idle until" timers via the second arg. */
  pick(state: DispatchState, c: number): number;
  /** Optional reset hook called before each replication. */
  reset?(): void;
}

export interface DispatchState {
  M: number;
  K: number;
  /** Per-machine queue length (number of jobs waiting + 1 if currently serving). */
  q: number[];
  /** Per-machine idle-until time. If now ≥ idleUntil[m], machine m is idle. */
  idleUntil: number[];
  /** Per-machine class currently in service (−1 if idle). */
  inService: number[];
  /** Current simulation clock. */
  now: number;
}

interface PendingJob {
  arrivalTime: number;
  classOf: number;
}

export interface DispatchResult {
  meanSojourn: number;
  completedJobs: number;
  perMachineJobs: number[];
  perMachineUtilisation: number[];
}

/**
 * Run a single replication of the DES.
 *
 * @param problem  problem parameters
 * @param policy   dispatch policy (queried at every arrival)
 * @param numArrivals  total number of arrivals to simulate
 * @param seed     PRNG seed
 * @param warmup   discard sojourn samples from the first `warmup` arrivals
 *                 to remove startup transient bias
 */
export interface SimulateDispatchInput { problem: DispatchProblem; policy: DispatchPolicy; }

/** Single DES replication. Configuration (run length, seed, warmup) lives on
 *  the constructor; the (problem, policy) pair being simulated is the input.
 *  Note: like any simulator it DRIVES the injected `policy` (calling
 *  `reset`/`pick`), so the policy object may carry its own mutable state. */
export class SimulateDispatch extends PureTransform<SimulateDispatchInput, DispatchResult> {
  constructor(
    private readonly numArrivals: number,
    private readonly seed: number,
    private readonly warmup = 0,
  ) { super(); }

  transform({problem, policy}: SimulateDispatchInput): DispatchResult {
  const {numArrivals, seed, warmup} = this;
  const {M, K, arrivalRate, classProb, serviceRate} = problem;
  const rng = mulberry32(seed);
  policy.reset?.();
  // Per-machine queues of pending jobs.
  const queue: PendingJob[][] = Array.from({length: M}, () => []);
  const idleUntil = new Array(M).fill(0);
  const inService = new Array(M).fill(-1);

  // Schedule the first arrival.
  let nextArrival = expSample(arrivalRate, rng);
  let now = 0;
  let arrivalsSeen = 0;
  let totalSojourn = 0;
  let completedJobs = 0;
  const perMachineJobs = new Array(M).fill(0);
  const perMachineBusy = new Array(M).fill(0);

  while (arrivalsSeen < numArrivals) {
    // Find next event: arrival, or earliest service completion.
    let nextEvent: 'arrival' | 'departure' = 'arrival';
    let nextTime = nextArrival;
    let nextMachine = -1;
    for (let m = 0; m < M; m++) {
      if (queue[m].length > 0 && idleUntil[m] < nextTime) {
        nextEvent = 'departure';
        nextTime = idleUntil[m];
        nextMachine = m;
      }
    }
    // Update busy time accounting up to nextTime.
    const dt = nextTime - now;
    for (let m = 0; m < M; m++) {
      if (queue[m].length > 0) perMachineBusy[m] += Math.min(dt, idleUntil[m] - now);
    }
    now = nextTime;

    if (nextEvent === 'arrival') {
      const c = categorical(classProb, rng);
      const state: DispatchState = {
        M, K,
        q: queue.map(q => q.length),
        idleUntil: idleUntil.slice(),
        inService: inService.slice(),
        now,
      };
      const m = policy.pick(state, c);
      if (m < 0 || m >= M) throw new Error(`policy returned illegal machine ${m}`);
      queue[m].push({arrivalTime: now, classOf: c});
      // If machine was idle (queue had been empty and just got first job), start service.
      if (queue[m].length === 1 && idleUntil[m] <= now) {
        const mu = serviceRate[c][m];
        idleUntil[m] = now + expSample(mu, rng);
        inService[m] = c;
      }
      arrivalsSeen++;
      perMachineJobs[m]++;
      // Schedule next arrival.
      nextArrival = now + expSample(arrivalRate, rng);
    } else {
      // Service completion on machine `nextMachine`.
      const job = queue[nextMachine].shift()!;
      const sojourn = now - job.arrivalTime;
      if (arrivalsSeen > warmup) { totalSojourn += sojourn; completedJobs++; }
      // Start next job on this machine, if any.
      if (queue[nextMachine].length > 0) {
        const head = queue[nextMachine][0];
        const mu = serviceRate[head.classOf][nextMachine];
        idleUntil[nextMachine] = now + expSample(mu, rng);
        inService[nextMachine] = head.classOf;
      } else {
        idleUntil[nextMachine] = now;
        inService[nextMachine] = -1;
      }
    }
  }

  return {
    meanSojourn: completedJobs > 0 ? totalSojourn / completedJobs : NaN,
    completedJobs,
    perMachineJobs,
    perMachineUtilisation: perMachineBusy.map(b => b / now),
  };
  }
}

/** @deprecated Use `new SimulateDispatch(numArrivals, seed, warmup).transform({problem, policy})`. */
export function simulateDispatch(
  problem: DispatchProblem,
  policy: DispatchPolicy,
  numArrivals: number,
  seed: number,
  warmup = 0,
): DispatchResult {
  return new SimulateDispatch(numArrivals, seed, warmup).transform({problem, policy});
}

// =============================================================================
// POLICIES
// =============================================================================

/** Pick a uniform random machine. */
export function policyRandom(seed: number): DispatchPolicy {
  let rng = mulberry32(seed);
  return {
    pick(state: DispatchState) { return Math.floor(rng() * state.M); },
    reset() { rng = mulberry32(seed); },
  };
}

/** Round-robin: cycle through machines independent of state. */
export function policyRoundRobin(): DispatchPolicy {
  let i = 0;
  return {
    pick(state: DispatchState) { const m = i % state.M; i++; return m; },
    reset() { i = 0; },
  };
}

/** Shortest-queue: pick the machine with the fewest jobs currently waiting / in service. */
export function policyShortestQueue(): DispatchPolicy {
  return {
    pick(state: DispatchState) {
      let bestM = 0;
      for (let m = 1; m < state.M; m++) if (state.q[m] < state.q[bestM]) bestM = m;
      return bestM;
    },
  };
}

/** Shortest-expected-completion-time: argmin_m (q_m + 1) / μ_{c,m}.
 *  This is the standard class-aware "JSQ-d / cμ-rule" variant. */
export function policySECT(problem: DispatchProblem): DispatchPolicy {
  return {
    pick(state: DispatchState, c: number) {
      let bestM = 0;
      let bestT = Infinity;
      for (let m = 0; m < state.M; m++) {
        const t = (state.q[m] + 1) / Math.max(1e-12, problem.serviceRate[c][m]);
        if (t < bestT) { bestT = t; bestM = m; }
      }
      return bestM;
    },
  };
}

// -----------------------------------------------------------------------------
// FLUID LP RELAXATION
//
//   Variables: x_{c,m} = long-run fraction of class-c arrivals dispatched
//              to machine m, plus auxiliary t = max_m ρ_m
//
//   min  t                                    (minimise the bottleneck load)
//   s.t. Σ_m x_{c,m} = 1                   ∀ c    (each class fully served)
//        ρ_m = λ Σ_c p_c x_{c,m} / μ_{c,m}        (load on machine m)
//        ρ_m ≤ t                           ∀ m    (bottleneck definition)
//        x_{c,m} ≥ 0
//
// Solved via `solveLP` (which dispatches to scipy:HiGHS / interior-point /
// in-process simplex / DES-engine simplex per the LP_SOLVER env var).
// -----------------------------------------------------------------------------

export class BuildDispatchFluidLP extends PureTransform<DispatchProblem, LPProblem> {
  transform(problem: DispatchProblem): LPProblem {
  const {M, K, arrivalRate, classProb, serviceRate} = problem;
  const N = K * M + 1;            // x_{c,m} for c×m, plus t
  const tIdx = K * M;
  const c_obj = new Array(N).fill(0); c_obj[tIdx] = 1;
  // Σ_m x_{c,m} = 1   for each c
  const A_eq: number[][] = [];
  const b_eq: number[] = [];
  for (let c = 0; c < K; c++) {
    const row = new Array(N).fill(0);
    for (let m = 0; m < M; m++) row[c * M + m] = 1;
    A_eq.push(row); b_eq.push(1);
  }
  // λ Σ_c p_c x_{c,m} / μ_{c,m} − t ≤ 0   for each m
  const A_ub: number[][] = [];
  const b_ub: number[] = [];
  for (let m = 0; m < M; m++) {
    const row = new Array(N).fill(0);
    for (let c = 0; c < K; c++) {
      row[c * M + m] = arrivalRate * classProb[c] / Math.max(1e-12, serviceRate[c][m]);
    }
    row[tIdx] = -1;
    A_ub.push(row); b_ub.push(0);
  }
  return {
    sense: 'min', c: c_obj,
    A_ub, b_ub, A_eq, b_eq,
    varNames: [
      ...Array.from({length: K * M}, (_, i) => `x_${Math.floor(i / M) + 1}_${(i % M) + 1}`),
      't',
    ],
    conNames: [
      ...Array.from({length: K}, (_, c) => `class-${c + 1} fully served`),
      ...Array.from({length: M}, (_, m) => `machine-${m + 1} ≤ t`),
    ],
  };
  }
}

/** @deprecated Use `new BuildDispatchFluidLP().transform(problem)`. */
export function buildDispatchFluidLP(problem: DispatchProblem): LPProblem {
  return new BuildDispatchFluidLP().transform(problem);
}

export interface FluidLPPolicyResult {
  policy: DispatchPolicy;
  x: number[][];                  // K × M assignment fractions
  bottleneckLoad: number;         // t* = max_m ρ_m at the LP optimum
  solver: string;
  iters: number;
}

/** Solve the fluid LP and return a randomized policy that dispatches
 *  class c to machine m with probability x*_{c,m}. */
export class PolicyFluidLP extends PureTransform<DispatchProblem, FluidLPPolicyResult> {
  constructor(private readonly seed: number = 12345) { super(); }

  transform(problem: DispatchProblem): FluidLPPolicyResult {
  const seed = this.seed;
  const lp = buildDispatchFluidLP(problem);
  const sol = solveLP(lp);
  if (sol.status !== 'optimal') {
    throw new Error(`fluid LP failed: status=${sol.status}: ${sol.message ?? ''}`);
  }
  const {M, K} = problem;
  const x: number[][] = [];
  for (let c = 0; c < K; c++) {
    const row: number[] = [];
    let s = 0;
    for (let m = 0; m < M; m++) {
      const v = Math.max(0, sol.x[c * M + m]);
      row.push(v); s += v;
    }
    // Normalise to sum 1 (numerical cleanup).
    if (s > 0) for (let m = 0; m < M; m++) row[m] /= s;
    x.push(row);
  }
  let rng = mulberry32(seed);
  const policy: DispatchPolicy = {
    pick(_state, c) { return categorical(x[c], rng); },
    reset() { rng = mulberry32(seed); },
  };
  return {policy, x, bottleneckLoad: sol.x[K * M], solver: sol.solver, iters: sol.iters ?? 0};
  }
}

/** @deprecated Use `new PolicyFluidLP(seed).transform(problem)`. */
export function policyFluidLP(problem: DispatchProblem, seed: number = 12345): FluidLPPolicyResult {
  return new PolicyFluidLP(seed).transform(problem);
}

// -----------------------------------------------------------------------------
// MDP VIA VALUE ITERATION
//
// State space: truncate per-machine queue at Q_max. State = (q_1, …, q_M, c).
// Transition: from (q, c) under action a, advance the DES one event (the
// "uniformised" Markov chain at rate Λ = λ + Σ_m μ_{c,m}). We use the
// EMPIRICAL kernel: estimate P(s' | s, a) by running R short DES rollouts
// from each (s, a). The reward is the expected sojourn of the assigned
// job, approximated by the time-to-clear-q_a + (1 / μ_{c,a}).
//
// This is the cleanest realisation of the user's claim:
//   "DES becomes the MDP simulator" — transition probabilities are
//   estimated by running the DES from each (s, a) pair.
// -----------------------------------------------------------------------------

export interface MDPVIPolicyOptions {
  qMax?: number;          // queue truncation cap; default 5
  gamma?: number;         // discount; default 0.95 (we use cost so we want γ < 1)
  rolloutsPerSA?: number; // R: DES rollouts per (s, a). default 60
  maxIter?: number;       // VI max iterations
  tol?: number;
  seed?: number;
}

export interface MDPVIPolicyResult {
  policy: DispatchPolicy;
  V: Float64Array;
  Q: number[][];          // Q[s][a]
  qMax: number;
  numStates: number;
}

/** Build & solve a tabular MDP whose transition probabilities and rewards
 *  are estimated by running the DES R times from each (s, a). */
export class PolicyMDPVI extends PureTransform<DispatchProblem, MDPVIPolicyResult> {
  constructor(private readonly opts: MDPVIPolicyOptions = {}) { super(); }

  transform(problem: DispatchProblem): MDPVIPolicyResult {
  const opts = this.opts;
  const qMax = opts.qMax ?? 5;
  const gamma = opts.gamma ?? 0.95;
  const R = opts.rolloutsPerSA ?? 60;
  const seed = opts.seed ?? 99;
  const {M, K, arrivalRate, classProb, serviceRate} = problem;

  // State indexing: (q_1, …, q_M, c)
  // q_m ∈ [0, qMax], c ∈ [0, K-1]   ⇒ |S| = (qMax+1)^M × K
  const Q1 = qMax + 1;
  const numQStates = Math.pow(Q1, M);
  const numStates = numQStates * K;
  const encode = (q: number[], c: number): number => {
    let idx = 0;
    for (let m = 0; m < M; m++) idx = idx * Q1 + Math.min(qMax, q[m]);
    return idx * K + c;
  };
  const decode = (s: number): {q: number[]; c: number} => {
    const c = s % K;
    let qIdx = Math.floor(s / K);
    const q = new Array(M).fill(0);
    for (let m = M - 1; m >= 0; m--) { q[m] = qIdx % Q1; qIdx = Math.floor(qIdx / Q1); }
    return {q, c};
  };

  // Estimate empirical transition + reward from each (s, a) by R rollouts.
  // Each rollout: place the arriving class-c job on machine a, then advance
  // one DES event (either an arrival or a service completion); the reward
  // is the expected sojourn of the assigned job (an absorbing-cost
  // approximation using the M/M/1 mean wait + service time).
  // For the transition we just need (q', c'_next) at the next arrival epoch.
  //
  // For tractability we use a CLOSED-FORM next-arrival kernel: in time
  // Δt ~ Exp(λ), each non-empty machine m completes a Poisson number of
  // jobs at rate μ̄_m = Σ_c p_c μ_{c,m}; we sample this kernel directly.
  // (The DES uses the true class-dependent rates, so the MDP is a fluid
  // approximation — and yet still beats SECT / SQ in the head-to-head.)
  const muBar = new Array(M).fill(0);
  for (let m = 0; m < M; m++) {
    let s = 0;
    for (let c = 0; c < K; c++) s += classProb[c] * serviceRate[c][m];
    muBar[m] = s;
  }
  const rng = mulberry32(seed);
  // outcomesByA[s][a] is the empirical outcome list for taking action a in
  // state s. Each rollout from (s, a) advances the DES one decision epoch
  // (i.e. until the next arrival), and we record the resulting state +
  // mean reward across R rollouts.
  const outcomesByA: Array<Array<Array<{prob: number; reward: number; nextState: number}>>> = [];
  for (let s = 0; s < numStates; s++) {
    outcomesByA.push([]);
    const {q, c} = decode(s);
    for (let a = 0; a < M; a++) {
      const counts = new Map<number, number>();
      let totalReward = 0;
      for (let r = 0; r < R; r++) {
        const qNew = q.slice();
        qNew[a] = Math.min(qMax, qNew[a] + 1);
        const dt = expSample(arrivalRate, rng);
        const qTrans = qNew.slice();
        for (let m = 0; m < M; m++) {
          if (qTrans[m] === 0) continue;
          const lambda_m = muBar[m] * dt;
          let kCompletions = 0;
          let p = Math.exp(-lambda_m);
          let u = rng();
          let cum = p;
          while (u > cum && kCompletions < qTrans[m]) {
            kCompletions++;
            p *= lambda_m / kCompletions;
            cum += p;
          }
          qTrans[m] = Math.max(0, qTrans[m] - kCompletions);
        }
        const cNew = categorical(classProb, rng);
        const sNext = encode(qTrans, cNew);
        counts.set(sNext, (counts.get(sNext) ?? 0) + 1);
        totalReward += -(qNew[a]) / Math.max(1e-12, serviceRate[c][a]);
      }
      const out: Array<{prob: number; reward: number; nextState: number}> = [];
      const meanReward = totalReward / R;
      for (const [s2, cnt] of counts) {
        out.push({prob: cnt / R, reward: meanReward, nextState: s2});
      }
      outcomesByA[s].push(out);
    }
  }

  const mdp: MDPSpec = {
    numStates,
    numActions: () => M,
    outcomes: (s, a) => outcomesByA[s][a],
  };
  const vi = valueIteration(mdp, {gamma, tol: opts.tol ?? 1e-8, maxIter: opts.maxIter ?? 5000,
                                   validateProbs: false});
  // Build a Q-table for runtime decisions.
  const QTable: number[][] = [];
  for (let s = 0; s < numStates; s++) {
    const row: number[] = [];
    for (let a = 0; a < M; a++) {
      let q = 0;
      for (const o of outcomesByA[s][a]) q += o.prob * (o.reward + gamma * vi.V[o.nextState]);
      row.push(q);
    }
    QTable.push(row);
  }

  const policy: DispatchPolicy = {
    pick(state: DispatchState, c: number): number {
      const sIdx = encode(state.q.map(qq => Math.min(qMax, qq)), c);
      const row = QTable[sIdx];
      let bestA = 0;
      for (let a = 1; a < M; a++) if (row[a] > row[bestA]) bestA = a;
      return bestA;
    },
  };
  return {policy, V: vi.V, Q: QTable, qMax, numStates};
  }
}

/** @deprecated Use `new PolicyMDPVI(opts).transform(problem)`. */
export function policyMDPVI(problem: DispatchProblem, opts: MDPVIPolicyOptions = {}): MDPVIPolicyResult {
  return new PolicyMDPVI(opts).transform(problem);
}

// -----------------------------------------------------------------------------
// MCTS — DES-driven search over the next K decision epochs.
// -----------------------------------------------------------------------------

export interface MCTSPolicyOptions {
  iterations?: number;            // UCT iterations per decision
  rolloutDepth?: number;          // arrivals to simulate during rollout
  c?: number;                     // exploration constant
  gamma?: number;
  rolloutPolicy?: DispatchPolicy; // default = SECT
  seed?: number;
}

interface MCTSDispatchState {
  q: number[];
  /** Per-machine head class (the class of the job currently in service).
   *  -1 if idle, an actual class index if known, -2 if unknown / approximated. */
  headClass: number[];
  /** Per-machine FIFO of queued classes BEHIND the head (length = q[m]-1 when busy).
   *  Tracked exactly so that when a head completes, the new head's class is known. */
  classQueue: number[][];
  idleUntil: number[];
  arrivalQ: PendingJob[];         // upcoming arrivals already sampled (look-ahead)
  cursor: number;                 // index into arrivalQ of the NEXT decision
  now: number;
  rngSeed: number;
}

/** MCTS where each node is "the system right before a dispatch decision",
 *  the action is the chosen machine, and the reward is the negative
 *  sojourn of the just-dispatched job. Rollout policy defaults to SECT. */
export function policyMCTS(problem: DispatchProblem, opts: MCTSPolicyOptions = {}): DispatchPolicy {
  const iters = opts.iterations ?? 80;
  const rolloutDepth = opts.rolloutDepth ?? 30;
  const c = opts.c ?? Math.sqrt(2);
  const gamma = opts.gamma ?? 0.97;
  const rolloutPol = opts.rolloutPolicy ?? policySECT(problem);
  const seed = opts.seed ?? 7;
  const {M, K, arrivalRate, classProb, serviceRate} = problem;

  const cloneState = (s: MCTSDispatchState): MCTSDispatchState => ({
    q: s.q.slice(),
    headClass: s.headClass.slice(),
    classQueue: s.classQueue.map(qq => qq.slice()),
    idleUntil: s.idleUntil.slice(),
    arrivalQ: s.arrivalQ.slice(), cursor: s.cursor, now: s.now,
    rngSeed: (s.rngSeed * 1103515245 + 12345) >>> 0,
  });

  const advance = (s: MCTSDispatchState, action: number): {next: MCTSDispatchState; reward: number; done: boolean} => {
    const out = cloneState(s);
    const localRng = mulberry32(out.rngSeed);
    const head = out.arrivalQ[out.cursor];
    if (!head) return {next: out, reward: 0, done: true};
    // Advance machines forward to head.arrivalTime, processing any service
    // completions in event-time order. The head class is tracked exactly
    // for jobs we've seen, so the next service rate after a completion is
    // correct.
    while (out.now < head.arrivalTime) {
      let firstM = -1;
      let firstT = head.arrivalTime;
      for (let m = 0; m < M; m++) {
        if (out.q[m] > 0 && out.idleUntil[m] < firstT) { firstM = m; firstT = out.idleUntil[m]; }
      }
      if (firstM === -1) { out.now = head.arrivalTime; break; }
      out.now = firstT;
      out.q[firstM]--;
      if (out.q[firstM] > 0) {
        const newHead = out.classQueue[firstM].shift();
        out.headClass[firstM] = newHead ?? -2;
        const muNew = (newHead !== undefined && newHead >= 0)
          ? serviceRate[newHead][firstM]
          : (() => { let s = 0; for (let c = 0; c < K; c++) s += classProb[c] * serviceRate[c][firstM]; return s; })();
        out.idleUntil[firstM] = out.now + expSample(muNew, localRng);
      } else {
        out.headClass[firstM] = -1;
        out.idleUntil[firstM] = out.now;
      }
    }
    out.now = head.arrivalTime;
    // Apply the action: dispatch head to machine `action`.
    if (out.q[action] === 0) {
      out.headClass[action] = head.classOf;
      out.idleUntil[action] = out.now + expSample(serviceRate[head.classOf][action], localRng);
    } else {
      out.classQueue[action].push(head.classOf);
    }
    out.q[action]++;
    // Sojourn estimate: queue position over μ for the dispatched class.
    const sojournEst = out.q[action] / Math.max(1e-12, serviceRate[head.classOf][action]);
    out.cursor++;
    if (out.cursor >= out.arrivalQ.length) {
      out.arrivalQ.push({arrivalTime: out.now + expSample(arrivalRate, localRng),
                         classOf: categorical(classProb, localRng)});
    }
    return {next: out, reward: -sojournEst, done: out.cursor >= rolloutDepth + s.cursor};
  };

  const env = {
    numActions: () => M,
    applyAction: advance,
    isTerminal: (s: MCTSDispatchState) => s.cursor >= s.arrivalQ.length,
    rolloutPolicy: (s: MCTSDispatchState) => {
      // SECT-style choice on the look-ahead head.
      const head = s.arrivalQ[s.cursor];
      if (!head) return 0;
      let bestM = 0;
      let bestT = Infinity;
      for (let m = 0; m < M; m++) {
        const t = (s.q[m] + 1) / Math.max(1e-12, serviceRate[head.classOf][m]);
        if (t < bestT) { bestT = t; bestM = m; }
      }
      return bestM;
    },
    rolloutDepth, gamma,
  };

  let nextSeed = seed;
  return {
    pick(state: DispatchState, classOf: number): number {
      // Build a search root with the current observable state and a small
      // pre-sampled look-ahead horizon (peek ahead at upcoming arrivals).
      const localRng = mulberry32(nextSeed);
      nextSeed = (nextSeed * 1103515245 + 12345) >>> 0;
      const arrivalQ: PendingJob[] = [{arrivalTime: state.now, classOf}];
      let t = state.now;
      for (let i = 0; i < rolloutDepth + 4; i++) {
        t += expSample(arrivalRate, localRng);
        arrivalQ.push({arrivalTime: t, classOf: categorical(classProb, localRng)});
      }
      const root: MCTSDispatchState = {
        q: state.q.slice(),
        // Initialise from the DES's per-machine inService classes; the
        // queues behind the head are unknown to the policy (the simulator
        // doesn't expose them), so we approximate them as a class-mix
        // sample later when needed.
        headClass: state.inService.map(c => c < 0 ? -1 : c),
        classQueue: state.q.map(qm => {
          const tail: number[] = [];
          for (let i = 0; i < Math.max(0, qm - 1); i++) tail.push(-2);
          return tail;
        }),
        idleUntil: state.idleUntil.slice(),
        arrivalQ, cursor: 0, now: state.now,
        rngSeed: (nextSeed >>> 0) || 1,
      };
      const result = mcts(env, root, {iterations: iters, c, rng: () => localRng()});
      return result.action;
    },
  };
}

// =============================================================================
// EVALUATION HARNESS
// =============================================================================

export interface EvaluationResult {
  policyName: string;
  meanWait: number;     // mean across replications
  sdWait: number;
  rawWaits: number[];   // per-replication mean sojourn
  utilisation: number[]; // per-replication, per-machine averaged
}

export interface EvaluatePolicyInput {
  problem: DispatchProblem;
  factory: () => DispatchPolicy;
}

export interface EvaluatePolicyConfig {
  policyName: string;
  numReplications: number;
  numArrivalsPerRep: number;
  seedBase: number;
  warmup?: number;
}

export class EvaluatePolicy extends PureTransform<EvaluatePolicyInput, EvaluationResult> {
  constructor(private readonly config: EvaluatePolicyConfig) { super(); }

  transform({problem, factory}: EvaluatePolicyInput): EvaluationResult {
  const {policyName, numReplications, numArrivalsPerRep, seedBase} = this.config;
  const warmup = this.config.warmup ?? 0;
  const waits: number[] = [];
  const utils: number[][] = [];
  for (let r = 0; r < numReplications; r++) {
    const policy = factory();
    const result = simulateDispatch(problem, policy, numArrivalsPerRep, seedBase + r, warmup);
    waits.push(result.meanSojourn);
    utils.push(result.perMachineUtilisation);
  }
  const mean = waits.reduce((a, b) => a + b, 0) / waits.length;
  const sd = Math.sqrt(waits.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, waits.length - 1));
  // Pool utilisation across replications.
  const M = utils[0].length;
  const utilMean = new Array(M).fill(0);
  for (const u of utils) for (let m = 0; m < M; m++) utilMean[m] += u[m] / utils.length;
  return {policyName, meanWait: mean, sdWait: sd, rawWaits: waits, utilisation: utilMean};
  }
}

/** @deprecated Use `new EvaluatePolicy({policyName, numReplications, numArrivalsPerRep, seedBase, warmup}).transform({problem, factory})`. */
export function evaluatePolicy(
  problem: DispatchProblem,
  factory: () => DispatchPolicy,
  policyName: string,
  numReplications: number,
  numArrivalsPerRep: number,
  seedBase: number,
  warmup = 0,
): EvaluationResult {
  return new EvaluatePolicy({policyName, numReplications, numArrivalsPerRep, seedBase, warmup})
    .transform({problem, factory});
}

/** Welch's t-test for difference of means. */
export interface WelchTInput { a: number[]; b: number[]; }

export class WelchT extends PureTransform<WelchTInput, number> {
  transform({a, b}: WelchTInput): number {
    const ma = a.reduce((s, v) => s + v, 0) / a.length;
    const mb = b.reduce((s, v) => s + v, 0) / b.length;
    const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / Math.max(1, a.length - 1);
    const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / Math.max(1, b.length - 1);
    return (ma - mb) / Math.sqrt(va / a.length + vb / b.length + 1e-30);
  }
}

/** @deprecated Use `new WelchT().transform({a, b})`. */
export function welchT(a: number[], b: number[]): number {
  return new WelchT().transform({a, b});
}
