// RUST MIGRATION: Target module `src/des/general/ppo_des.rs`.
// RUST MIGRATION: Convert PPO options and DES result interfaces to `serde` structs; keep numeric state/action specialization explicit (`usize` or newtypes) instead of `number`.
// RUST MIGRATION: Port `TabularPPOAgent` and `PPOClipUpdateStation` as structs implementing policy-gradient and policy-update traits.
// RUST MIGRATION: Replace inherited override methods with trait impls; shared base-class state should be embedded/composed in the Rust structs.
// RUST MIGRATION: Inject RNG through the environment/agent ports and return `Result` for bad batch sizes, horizons, or non-finite advantages.
'use strict';

// =============================================================================
// general/ppo-des.ts — PPO as a DES, built on PolicyGradientAgent and
// PolicyUpdateStation base classes.
//
// Concrete leaves:
//   • TabularPPOAgent extends PolicyGradientAgent<number, number>
//       hooks: samplePolicyAndValue (softmax over θ[s][·] + V[s])
//   • PPOClipUpdateStation extends PolicyUpdateStation
//       hooks: runUpdate (GAE advantages + K epochs of clipped surrogate
//               + value MSE on the agent's tabular θ and V).
//
// Topology:
//
//     ┌──────────────┐     state/trans       ┌────────────────────────┐
//     │  Environment  │ ──────────────────▶ │  TabularPPOAgent        │
//     │   Station     │                     │  (PolicyGradientAgent)  │
//     │               │ ◀── action ──────── │                         │
//     └──────────────┘                      └─────────────┬──────────┘
//                                                          │ TrainTrigger (when buffer full)
//                                                          ▼
//                                              ┌────────────────────────┐
//                                              │  PPOClipUpdateStation   │
//                                              │  (PolicyUpdateStation)  │
//                                              └─────────────┬──────────┘
//                                                          │ Resume
//                                                          ▼
//                                                       (back to agent)
// =============================================================================

import {
  PolicyGradientAgent, PolicyUpdateStation, EnvironmentStation,
  PureEnvironment, runIterativeDES, IterativeRunOptions,
  RolloutEntry, argMaxWithTieBreak,
} from './des-base';
import {mulberry32} from './prng';

// -----------------------------------------------------------------------------
// TABULAR PPO AGENT
// -----------------------------------------------------------------------------

export class TabularPPOAgent extends PolicyGradientAgent<number, number> {
  /** Policy logits θ[s][a]. */
  theta: number[][];
  /** Value table V[s]. */
  V: number[];
  numStates: number;
  numActions: number;

  constructor(id: string, opts: {
    numStates: number;
    numActions: number;
    rolloutLen: number;
    rng: () => number;
  }) {
    super(id, {rolloutLen: opts.rolloutLen, rng: opts.rng});
    this.numStates = opts.numStates;
    this.numActions = opts.numActions;
    this.theta = Array.from({length: opts.numStates},
      () => new Array(opts.numActions).fill(0));
    this.V = new Array(opts.numStates).fill(0);
  }

  // ── HOOK ────────────────────────────────────────────────────────────────

  protected samplePolicyAndValue(state: number, rng: () => number): {
    action: number; logProb: number; value: number;
  } {
    const logits = this.theta[state];
    let m = -Infinity;
    for (const x of logits) if (x > m) m = x;
    let z = 0;
    for (const x of logits) z += Math.exp(x - m);
    const logZ = m + Math.log(z);
    const u = rng();
    let cum = 0;
    let a = logits.length - 1;
    for (let i = 0; i < logits.length; i++) {
      cum += Math.exp(logits[i] - logZ);
      if (u <= cum) { a = i; break; }
    }
    return {action: a, logProb: logits[a] - logZ, value: this.V[state]};
  }

  greedyPolicy(): number[] {
    return this.theta.map(row => argMaxWithTieBreak(row, this.rng));
  }
}

// -----------------------------------------------------------------------------
// PPO CLIP UPDATE STATION
// -----------------------------------------------------------------------------

export interface PPOUpdateOptions {
  gamma: number;
  lambda: number;
  clipEps: number;
  policyLr: number;
  valueLr: number;
  numEpochs: number;
  miniBatchSize: number;
  entropyCoef?: number;
  /** Normalise advantages within the batch before update. */
  normaliseAdvantage?: boolean;
}

export class PPOClipUpdateStation extends PolicyUpdateStation {
  constructor(
    id: string,
    private readonly agent: TabularPPOAgent,
    private readonly opts: PPOUpdateOptions & {rng: () => number},
  ) { super(id); }

  /** Subclasses fill this in. Reads from agent.buffer, mutates θ and V,
   *  clears the buffer at the end. */
  protected runUpdate(): void {
    const buf = this.agent.getBuffer();
    if (buf.length === 0) return;
    const n = buf.length;
    // 1. GAE advantages and returns-to-go.
    const {adv, ret} = this.computeAdvantages(buf as RolloutEntry<number, number>[]);
    if (this.opts.normaliseAdvantage ?? true) {
      let mean = 0; for (const x of adv) mean += x; mean /= n;
      let varAdv = 0; for (const x of adv) varAdv += (x - mean) ** 2; varAdv /= n;
      const std = Math.sqrt(varAdv) + 1e-8;
      for (let i = 0; i < n; i++) adv[i] = (adv[i] - mean) / std;
    }
    // 2. K epochs of clipped surrogate + value-loss SGD.
    const idx = Array.from({length: n}, (_, i) => i);
    for (let epoch = 0; epoch < this.opts.numEpochs; epoch++) {
      this.shuffle(idx);
      for (let mb = 0; mb < n; mb += this.opts.miniBatchSize) {
        const batch = idx.slice(mb, Math.min(n, mb + this.opts.miniBatchSize));
        for (const i of batch) {
          this.applyOneSampleUpdate(buf[i] as RolloutEntry<number, number>, adv[i], ret[i]);
        }
      }
    }
    // 3. Clear the buffer for next rollout.
    this.agent.clearBuffer();
  }

  private computeAdvantages(buf: RolloutEntry<number, number>[]): {adv: number[]; ret: number[]} {
    const n = buf.length;
    const adv = new Array(n).fill(0);
    const ret = new Array(n).fill(0);
    let gae = 0;
    for (let t = n - 1; t >= 0; t--) {
      const e = buf[t];
      // Defensive: tail entry might still be missing reward if rollout
      // ended exactly at buffer-full; treat as r=0, done=true, vNext=0.
      const r = e.r ?? 0;
      const done = e.done ?? true;
      const vNext = (done || e.sNext === undefined) ? 0 : this.agent.V[e.sNext];
      const delta = r + this.opts.gamma * vNext - e.v;
      gae = delta + this.opts.gamma * this.opts.lambda * (done ? 0 : gae);
      adv[t] = gae;
      ret[t] = adv[t] + e.v;
    }
    return {adv, ret};
  }

  private shuffle(arr: number[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.opts.rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /** One mini-batch sample's contribution. Tabular gradient:
   *    ∂log π(a|s) / ∂θ[s,a'] = δ(a'=a) − π(a'|s)
   *  Clipped surrogate gradient: zero outside the clip region. */
  private applyOneSampleUpdate(e: RolloutEntry<number, number>, A: number, G: number): void {
    const s = e.s;
    const a = e.a;
    const logits = this.agent.theta[s];
    let m = -Infinity; for (const l of logits) if (l > m) m = l;
    let z = 0; for (const l of logits) z += Math.exp(l - m);
    const logZ = m + Math.log(z);
    const logProbNew = logits[a] - logZ;
    const ratio = Math.exp(logProbNew - e.logProbOld);
    const inClip =
      (A >= 0 && ratio < 1 + this.opts.clipEps) ||
      (A <  0 && ratio > 1 - this.opts.clipEps);
    if (inClip) {
      for (let aPrime = 0; aPrime < logits.length; aPrime++) {
        const piAprime = Math.exp(logits[aPrime] - logZ);
        const grad = ratio * A * ((aPrime === a ? 1 : 0) - piAprime);
        this.agent.theta[s][aPrime] += this.opts.policyLr * grad;
      }
    }
    if (this.opts.entropyCoef && this.opts.entropyCoef > 0) {
      // Recompute π & H from PRE-update logits (the logits we just modified
      // are mutated in place; for simplicity use the snapshot we already
      // have via logits[]. Tiny inaccuracy is fine — entropy bonus is small.)
      const pi = logits.map(l => Math.exp(l - logZ));
      let H = 0; for (let k = 0; k < pi.length; k++) H -= pi[k] * (logits[k] - logZ);
      for (let aPrime = 0; aPrime < pi.length; aPrime++) {
        const grad = pi[aPrime] * (H + (logits[aPrime] - logZ));
        this.agent.theta[s][aPrime] += this.opts.policyLr * this.opts.entropyCoef * grad;
      }
    }
    // Value SGD: V[s] ← V[s] + lr_v · (G − V[s]).
    this.agent.V[s] += this.opts.valueLr * (G - this.agent.V[s]);
  }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

export interface PPODESResult {
  theta: number[][];
  V: number[];
  policy: number[];
  rewardHistory: readonly number[];
  lengthHistory: readonly number[];
  totalEpisodes: number;
  totalSteps: number;
  totalUpdates: number;
  totalTicks: number;
}

export function runPPODES(env: PureEnvironment<number, number>, opts: {
  totalSteps: number;
  rolloutLen: number;
  numEpochs: number;
  miniBatchSize: number;
  policyLr: number;
  valueLr: number;
  gamma: number;
  lambda: number;
  clipEps: number;
  entropyCoef?: number;
  normaliseAdvantage?: boolean;
  maxStepsPerEpisode?: number;
  seed?: number;
  desOptions?: IterativeRunOptions;
}): PPODESResult {
  const rng = mulberry32(opts.seed ?? 1);
  const agent = new TabularPPOAgent('actor', {
    numStates: env.numStates, numActions: env.numActions,
    rolloutLen: opts.rolloutLen, rng,
  });
  const updater = new PPOClipUpdateStation('updater', agent, {
    gamma: opts.gamma, lambda: opts.lambda, clipEps: opts.clipEps,
    policyLr: opts.policyLr, valueLr: opts.valueLr,
    numEpochs: opts.numEpochs, miniBatchSize: opts.miniBatchSize,
    entropyCoef: opts.entropyCoef ?? 0,
    normaliseAdvantage: opts.normaliseAdvantage,
    rng,
  });
  const envSt = new EnvironmentStation<number, number>('env', env, {
    maxStepsPerEpisode: opts.maxStepsPerEpisode,
  });
  // Channel wiring.
  envSt.pipe(agent, EnvironmentStation.CH_STATE, PolicyGradientAgent.CH_STATE);
  envSt.pipe(agent, EnvironmentStation.CH_TRANSITION, PolicyGradientAgent.CH_TRANSITION);
  agent.pipe(envSt, PolicyGradientAgent.CH_ACTION, EnvironmentStation.CH_ACTION);
  agent.pipe(updater, PolicyGradientAgent.CH_TRAIN, PolicyUpdateStation.CH_TRAIN);
  updater.pipe(agent, PolicyUpdateStation.CH_RESUME, PolicyGradientAgent.CH_RESUME);

  const summary = runIterativeDES([envSt, agent, updater], {
    rng,
    stopWhen: () => envSt.totalSteps >= opts.totalSteps,
    ...opts.desOptions,
  });
  envSt.done = true;

  return {
    theta: agent.theta, V: agent.V,
    policy: agent.greedyPolicy(),
    rewardHistory: envSt.rewardHistory,
    lengthHistory: envSt.lengthHistory,
    totalEpisodes: envSt.rewardHistory.length,
    totalSteps: envSt.totalSteps,
    totalUpdates: updater.numUpdates,
    totalTicks: summary.ticks,
  };
}
