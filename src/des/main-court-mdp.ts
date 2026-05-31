#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_court_mdp.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-court-mdp.rs   (fn main)
// 1:1 file move. USACC case-flow MDP: cases flow through stage stations, each
// applying an interchangeable Policy.
//
// Conversion notes (file-specific):
//   - the policy set (AlwaysEscalate/RejectAll/NaiveThreshold/Optimal) -> trait
//     Policy with one struct impl each (not subclasses).
//   - stochastic factor updates / stage transitions -> inject RandomSource/
//     SeededRandom.
//   - imports mdp/usacc-mdp + entity modules -> use crate::des::...
//   - top-level run -> fn main.
// =============================================================================

// =============================================================================
// USACC MDP simulation in the framework.
//
// Architecture:
//
//                              ┌───────────────────► AcceptedSink
//   CaseSource ─▶ Submission ─┐
//                  │          ├──▶ Validation ─┐
//                  │          ▼                ▼
//                  └────────► CloseSink     Admission ─┐
//                                              │       │
//                                              ▼       ▼
//                                            Trial ─▶ AcceptedSink
//                                              │
//                                              ▼
//                                          CloseSink
//
// Case = moving entity carrying a `CaseState` (per `mdp/usacc-mdp.ts`).
// Each Stage station applies a `Policy` to the case's current state, then
// performs the transition: stochastic factor updates plus stage advance,
// terminal close, or terminal accept depending on the action chosen.
//
// Policies are interchangeable. We ship four:
//   - `AlwaysEscalatePolicy`  → naively push every case to trial.
//   - `RejectAllPolicy`       → reject every case immediately.
//   - `NaiveThresholdPolicy`  → escalate if evidence ≥ MED, else investigate.
//   - `OptimalPolicy`         → looks up π* from value iteration.
//
// The framework's role: run N=1000 cases through the station graph, log per-
// case trajectories, aggregate reward / acceptance / closure / exhaustion
// rates, and compare across policies.
//
// Validation: `external-references/court-mdp/court_mdp.py` runs the SAME
// value iteration on the SAME MDP and dumps V* and π*. The TS validator
// checks both match within 1e-9 per state.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {fisherYatesShuffle} from './general/general';
import {mulberry32, withSeed} from './general/prng';
import {BufferedTimeSteppedStation} from './general/time-stepped-station';
import {
  ACCEPTED, ACTIONS, Action, CLOSED, CaseState, EXHAUSTED, FUND_ACTIVE,
  N_ACTIONS, decode, encode, isTerminal, outcomes, sampleInitialState,
  STAGES, EVIDENCE, CORROBORATION, MANIPULATION, CONFLICT, FUNDING,
} from './mdp/usacc-mdp';
import {valueIteration} from './mdp/value-iteration';

// -----------------------------------------------------------------------------
// Policy interface.
// -----------------------------------------------------------------------------

export interface Policy {
  name: string;
  /** Returns the action index (0..N_ACTIONS-1) for the given state. */
  pick(s: CaseState, stateId: number): number;
}

export class AlwaysEscalatePolicy implements Policy {
  name = 'always-escalate';
  pick(_s: CaseState): number { return ACTIONS.indexOf('escalate_to_next_stage'); }
}

export class RejectAllPolicy implements Policy {
  name = 'reject-all';
  pick(): number { return ACTIONS.indexOf('reject_or_close'); }
}

/**
 * A hand-tuned heuristic policy for comparison. Roughly: gather evidence
 * if it's low, push escrow if unfunded, audit if manipulation looks high,
 * escalate if the case looks strong, close if it looks weak.
 */
export class NaiveThresholdPolicy implements Policy {
  name = 'naive-threshold';
  pick(s: CaseState): number {
    if (s.funding === 0)        return ACTIONS.indexOf('release_escrow');
    if (s.manipulation >= 2)    return ACTIONS.indexOf('hold_for_audit');
    if (s.conflict === 1)       return ACTIONS.indexOf('assign_reviewers');
    if (s.evidence === 0)       return ACTIONS.indexOf('request_more_evidence');
    if (s.corroboration === 0)  return ACTIONS.indexOf('verify_identity');
    if (s.evidence + s.corroboration - s.manipulation - 1.5 * s.conflict >= 2.0)
      return ACTIONS.indexOf('escalate_to_next_stage');
    return ACTIONS.indexOf('reject_or_close');
  }
}

export class OptimalPolicy implements Policy {
  name = 'optimal';
  constructor(public action: Int32Array) {}
  pick(_s: CaseState, stateId: number): number {
    return this.action[stateId];
  }
}

// -----------------------------------------------------------------------------
// Moving entity.
// -----------------------------------------------------------------------------

class Case {
  history: Array<{stateId: number; action: number; reward: number}> = [];
  totalReward = 0;
  arrivalTime: number;
  exitTime: number = -1;
  terminal: number = -1;
  steps = 0;
  constructor(public id: number, public state: CaseState, t: number) {
    this.arrivalTime = t;
  }
}

// -----------------------------------------------------------------------------
// Stationary entities.
// -----------------------------------------------------------------------------

class CaseSource extends BufferedTimeSteppedStation<Case> {
  private idx = 0;
  constructor(id: string, public total: number, public arrivalsPerTick: number,
              public seed: number, public floor1: StageStation) { super(id); }
  runTimeStep(_stepSize: number, t: number): void {
    if (this.idx >= this.total) return;
    const rng = mulberry32(this.seed + this.idx);
    for (let k = 0; k < this.arrivalsPerTick && this.idx < this.total; k++, this.idx++) {
      const init = sampleInitialState(rng);
      const c = new Case(this.idx, init, t);
      this.floor1.takeItem(c);
    }
  }
}

class StageStation extends BufferedTimeSteppedStation<Case> {
  // Each stage station handles cases at a particular stage. It applies the
  // policy, performs the transition, and routes the case forward.
  constructor(
    id: string,
    public stageNum: number,                  // 0..3
    public policy: Policy,
    public rng: () => number,
    public nextStages: StageStation[],        // index by destination stage num
    public acceptedSink: TerminalSink,
    public closedSink:   TerminalSink,
    public exhaustedSink: TerminalSink,
  ) { super(id); }

  runTimeStep(_stepSize: number, t: number): void {
    if (this.inbox.length === 0) return;
    const todo = this.inbox;
    this.inbox = [];
    for (const c of todo) {
      if (c.state.stage !== this.stageNum) {
        // Routing mistake; defensive — should not happen.
        throw new Error(`StageStation ${this.id} got case at stage ${c.state.stage}`);
      }
      const stateId = encode(c.state);
      const action = this.policy.pick(c.state, stateId);
      const ol = outcomes(stateId, action);
      // Sample outcome.
      const r = this.rng();
      let cum = 0;
      let chosen = ol[0];
      for (const o of ol) {
        cum += o.prob;
        if (r <= cum) { chosen = o; break; }
      }
      c.history.push({stateId, action, reward: chosen.reward});
      c.totalReward += chosen.reward;
      c.steps++;
      if (chosen.nextState === ACCEPTED) {
        c.terminal = ACCEPTED; c.exitTime = t;
        this.acceptedSink.collect(c);
      } else if (chosen.nextState === CLOSED) {
        c.terminal = CLOSED; c.exitTime = t;
        this.closedSink.collect(c);
      } else if (chosen.nextState === EXHAUSTED) {
        c.terminal = EXHAUSTED; c.exitTime = t;
        this.exhaustedSink.collect(c);
      } else {
        const sNext = decode(chosen.nextState)!;
        c.state = sNext;
        if (sNext.stage === this.stageNum) {
          // Same stage; re-enqueue here for the next tick.
          this.inbox.push(c);
        } else {
          this.nextStages[sNext.stage].takeItem(c);
        }
      }
    }
  }
}

class TerminalSink extends BufferedTimeSteppedStation<Case> {
  collected: Case[] = [];
  constructor(id: string) { super(id); }
  collect(c: Case): void { this.collected.push(c); }
  runTimeStep(_stepSize: number, _t: number): void { /* nothing */ }
}

// -----------------------------------------------------------------------------
// Public API.
// -----------------------------------------------------------------------------

export interface CourtMDPConfig {
  totalCases: number;
  arrivalsPerTick: number;
  maxTicks: number;
  seed: number;
}

export interface CourtMDPResult {
  policy: string;
  config: CourtMDPConfig;
  cases: Array<{
    id: number; arrivalTime: number; exitTime: number; terminal: number;
    steps: number; totalReward: number;
  }>;
  aggregates: {
    n: number;
    nAccepted: number;
    nClosed: number;
    nExhausted: number;
    nTimedOut: number;
    meanReward: number;
    meanSteps: number;
    p95Steps: number;
    fractionAccepted: number;
    fractionClosed: number;
    fractionExhausted: number;
  };
}

export function runCourtSim(cfg: CourtMDPConfig, policy: Policy): CourtMDPResult {
  return withSeed(cfg.seed, () => {
    const rng = mulberry32(cfg.seed ^ 0xCAFE);
    const acceptedSink  = new TerminalSink('accepted');
    const closedSink    = new TerminalSink('closed');
    const exhaustedSink = new TerminalSink('exhausted');
    const stages: StageStation[] = [];
    for (let i = 0; i < 4; i++) {
      stages.push(new StageStation(`stage${i}`, i, policy, rng, stages, acceptedSink, closedSink, exhaustedSink));
    }
    // Stage station references each other (forward routing): wire the array
    // (already shared by reference above).
    const source = new CaseSource('src', cfg.totalCases, cfg.arrivalsPerTick, cfg.seed, stages[0]);
    const stations: BufferedTimeSteppedStation<Case>[] = [source, ...stages, acceptedSink, closedSink, exhaustedSink];

    const allCases: Case[] = [];
    let t = 0;
    while (t < cfg.maxTicks) {
      // Source first so newly-arrived cases enter the graph this same tick.
      source.runTimeStep(1.0, t);
      // Process stages in shuffled order.
      const stageOrder = [...stages];
      for (const _ of fisherYatesShuffle(stageOrder)) { /* generator */ }
      for (const s of stageOrder) s.runTimeStep(1.0, t);
      t++;
      const collected = acceptedSink.collected.length + closedSink.collected.length + exhaustedSink.collected.length;
      if (collected === cfg.totalCases) break;
    }

    const allFinished: Case[] = [
      ...acceptedSink.collected,
      ...closedSink.collected,
      ...exhaustedSink.collected,
    ];
    const nTimedOut = cfg.totalCases - allFinished.length;
    allCases.push(...allFinished);

    const rewards = allFinished.map(c => c.totalReward);
    const steps = allFinished.map(c => c.steps);
    const sortedSteps = [...steps].sort((a, b) => a - b);
    const p95 = sortedSteps.length > 0 ? sortedSteps[Math.floor(0.95 * (sortedSteps.length - 1))] : 0;
    const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length);

    return {
      policy: policy.name,
      config: cfg,
      cases: allFinished.map(c => ({
        id: c.id, arrivalTime: c.arrivalTime, exitTime: c.exitTime,
        terminal: c.terminal, steps: c.steps, totalReward: c.totalReward,
      })),
      aggregates: {
        n: cfg.totalCases,
        nAccepted: acceptedSink.collected.length,
        nClosed: closedSink.collected.length,
        nExhausted: exhaustedSink.collected.length,
        nTimedOut,
        meanReward: mean(rewards),
        meanSteps:  mean(steps),
        p95Steps:   p95,
        fractionAccepted:  acceptedSink.collected.length  / cfg.totalCases,
        fractionClosed:    closedSink.collected.length    / cfg.totalCases,
        fractionExhausted: exhaustedSink.collected.length / cfg.totalCases,
      },
    };
  });
}

// -----------------------------------------------------------------------------
// CLI: run all four policies, plus dump V* and π* for the validator.
// -----------------------------------------------------------------------------

function dumpVI() {
  const t0 = Date.now();
  const vi = valueIteration({gamma: 0.95, tol: 1e-10, maxIter: 5000});
  console.log(`# value iteration: ${vi.iterations} sweeps, max|ΔV| = ${vi.finalDelta.toExponential(3)}, ${Date.now() - t0} ms`);
  return vi;
}

function main() {
  const N = Number(process.env.CASES ?? 5000);
  const seed = Number(process.env.SEED ?? 42);
  const arrivalsPerTick = Number(process.env.ARRIVALS_PER_TICK ?? 5);
  const maxTicks = Number(process.env.MAX_TICKS ?? 10000);

  console.log('# USACC MDP simulation');
  console.log(`#   ${N} cases, ${arrivalsPerTick} arrivals/tick, maxTicks=${maxTicks}, seed=${seed}`);

  const vi = dumpVI();
  const optimal = new OptimalPolicy(vi.policy);
  const policies: Policy[] = [
    new RejectAllPolicy(),
    new AlwaysEscalatePolicy(),
    new NaiveThresholdPolicy(),
    optimal,
  ];

  const cfg: CourtMDPConfig = {totalCases: N, arrivalsPerTick, maxTicks, seed};

  const results: CourtMDPResult[] = [];
  for (const p of policies) {
    const t0 = Date.now();
    const r = runCourtSim(cfg, p);
    const ms = Date.now() - t0;
    results.push(r);
    const a = r.aggregates;
    console.log('');
    console.log(`# policy = ${p.name}    (${ms} ms)`);
    console.log(`#   meanReward = ${a.meanReward.toFixed(2)}    meanSteps = ${a.meanSteps.toFixed(2)}    p95Steps = ${a.p95Steps}`);
    console.log(`#   accepted = ${(a.fractionAccepted * 100).toFixed(1)}%    closed = ${(a.fractionClosed * 100).toFixed(1)}%    exhausted = ${(a.fractionExhausted * 100).toFixed(1)}%`);
    if (a.nTimedOut > 0) console.log(`#   WARNING: ${a.nTimedOut} cases timed out (raise MAX_TICKS)`);
  }

  // Dump V*, π*, and per-policy aggregates for the validator.
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'court-mdp-framework.json');
  fs.writeFileSync(outPath, JSON.stringify({
    config: cfg,
    vi: {
      gamma: vi.gamma,
      iterations: vi.iterations,
      finalDelta: vi.finalDelta,
      V: Array.from(vi.V),
      policy: Array.from(vi.policy),
    },
    results: results.map(r => ({policy: r.policy, aggregates: r.aggregates})),
  }));
  console.log(`# wrote ${outPath}`);

  // Show the optimal action at each stage (averaged over factor combos).
  console.log('');
  console.log('# Optimal action distribution by stage (from π*):');
  for (let stage = 0; stage < 4; stage++) {
    const counts = new Array(N_ACTIONS).fill(0);
    let total = 0;
    for (let s = 0; s < 864; s++) {
      const cs = decode(s)!;
      if (cs.stage === stage) {
        counts[vi.policy[s]]++;
        total++;
      }
    }
    console.log(`#   ${STAGES[stage].padEnd(4)} (${total} states): ` +
      counts.map((c, i) => c > 0 ? `${ACTIONS[i].slice(0,4)}=${c}` : '').filter(Boolean).join(' '));
  }
}

if (require.main === module) main();
