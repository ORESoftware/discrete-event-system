#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_two_disease.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// Two-disease epidemic with co-infection interaction.
//
// MODEL
// -----
// Six-compartment SIR-on-a-lattice. Diseases A and B can each spread
// independently through a fully-mixed population, but a person already
// infected with one can also catch the other and become co-infected (AB).
// The only thing the diseases share is the host: their transmission rates
// and removal rates are independent. The interaction enters via the
// per-co-infection death probability, which the user has set to be LOWER
// than the higher of the two single-disease death probabilities (50% with
// both vs 60% from B alone), modelling some kind of immune-priming /
// resource-sharing effect.
//
//             β_A·(I_A/N)        β_B·(I_B/N)
//   ┌───────►   A   ───────────►   AB
//   │            \                /
//   │             γ_A           γ_AB
//   S              \           /
//   │               ▼         ▼
//   │              R or D    R or D
//   │
//   │  β_B·(I_B/N)        β_A·(I_A/N)
//   └───────►   B   ───────────►   AB
//                  \
//                   γ_B
//                    ▼
//                  R or D
//
// where I_A = #A + #AB and I_B = #B + #AB.
//
// Per-compartment removal:
//   A   → R with prob γ_A · (1 − p_death_A) · dt    , → D with prob γ_A · p_death_A · dt
//   B   → R with prob γ_B · (1 − p_death_B) · dt    , → D with prob γ_B · p_death_B · dt
//   AB  → R with prob γ_AB · (1 − p_death_AB) · dt , → D with prob γ_AB · p_death_AB · dt
//
// Default user-stated death probabilities:
//   p_death_A   = 0.40
//   p_death_B   = 0.60
//   p_death_AB  = 0.50         ← lower than B alone, modelling interaction
//
// FRAMEWORK FIT (the interesting bit)
// ------------------------------------
// Each compartment is a stationary entity holding its current population
// of moving entities (people). Compartments interact: the rate at which
// S becomes A depends on the global count in A AND AB. To preserve the
// "each station sees only what it sees" discipline, we add an explicit
// `WorldCensus` station that runs FIRST each tick and reads each
// compartment's count, freezing them in a shared `counts` object that
// every compartment reads but never mutates. Compartments then run in
// shuffled order (Fisher-Yates), each emitting transitions to other
// compartments via a `pending` buffer; after all have run, every
// compartment commits its pending → people. This is exactly the
// synchronous-data-flow pattern used by `main-electric-circuit.ts`.
//
// VALIDATION
// ----------
// External reference at `external-references/two-disease/two_disease.py`
// implements the same model two ways:
//   (1) deterministic mean-field ODE with scipy LSODA;
//   (2) Gillespie SSA stochastic with the same per-person rates.
// `validate-two-disease.ts` compares the framework's ensemble mean (over
// many seeds) to the ODE solution and Welch-tests against the Gillespie
// SSA. Final-size analytics (R∞, D∞, fraction co-infected) are checked
// directly.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {fisherYatesShuffle} from './general/general';
import {mulberry32, withSeed} from './general/prng';
import {competingRisks, sampleCategorical, poissonBinomialPMF, meanFromPMF, varianceFromPMF} from './general/random-variables';
import {TimeSteppedStation as Station} from './general/time-stepped-station';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export type CompartmentId = 'S' | 'A' | 'B' | 'AB' | 'R' | 'D';

export interface TwoDiseaseParams {
  N: number;            // total population
  initialA: number;     // start in A
  initialB: number;     // start in B
  initialAB: number;    // start in AB
  beta_A: number;       // transmission rate for A
  beta_B: number;       // transmission rate for B
  gamma_A: number;      // removal rate for A
  gamma_B: number;      // removal rate for B
  gamma_AB: number;     // removal rate for AB
  p_death_A: number;    // P(D | leaving A)
  p_death_B: number;    // P(D | leaving B)
  p_death_AB: number;   // P(D | leaving AB)
  simT: number;         // total simulated time
  stepSize: number;     // dt
  seed: number;
}

class Person {
  state: CompartmentId = 'S';
  history: Array<{state: CompartmentId; time: number}> = [];
  constructor(public id: number) { this.history.push({state: 'S', time: 0}); }
  transition(to: CompartmentId, time: number): void {
    this.state = to; this.history.push({state: to, time});
  }
}

// -----------------------------------------------------------------------------
// Stations
// -----------------------------------------------------------------------------

interface GlobalCounts {
  S: number; A: number; B: number; AB: number; R: number; D: number;
  N: number;       // total alive (S + A + B + AB + R), used as denominator
  total: number;   // S + A + B + AB + R + D, should be invariant
}

/**
 * Reads each compartment's `people.length` at the START of the tick and
 * publishes a frozen snapshot. All compartments use this snapshot to
 * compute per-person transition probabilities, so transitions are
 * order-independent within the tick.
 */
class WorldCensus extends Station {
  counts: GlobalCounts = {S: 0, A: 0, B: 0, AB: 0, R: 0, D: 0, N: 0, total: 0};
  constructor(id: string, public S: Compartment, public A: Compartment, public B: Compartment,
              public AB: Compartment, public R: Compartment, public D: Compartment) { super(id); }
  runTimeStep(_stepSize: number, _t: number): void {
    this.counts.S  = this.S.people.length;
    this.counts.A  = this.A.people.length;
    this.counts.B  = this.B.people.length;
    this.counts.AB = this.AB.people.length;
    this.counts.R  = this.R.people.length;
    this.counts.D  = this.D.people.length;
    this.counts.N     = this.counts.S + this.counts.A + this.counts.B + this.counts.AB + this.counts.R;
    this.counts.total = this.counts.N + this.counts.D;
  }
}

class Compartment extends Station {
  people: Person[] = [];
  pending: Person[] = [];
  constructor(
    id: string,
    public kind: CompartmentId,
    public params: TwoDiseaseParams,
    public rng: () => number,
    public census: WorldCensus,
  ) { super(id); }

  /** Called by another compartment to push a person into this one. */
  takeItem(p: Person): void { this.pending.push(p); }

  /** End-of-tick commit: pending becomes part of the population. */
  commit(): void {
    if (this.pending.length === 0) return;
    for (const p of this.pending) this.people.push(p);
    this.pending = [];
  }

  // Targets — set by the simulation builder.
  destA!: Compartment;  destB!: Compartment;  destAB!: Compartment;
  destR!: Compartment;  destD!: Compartment;

  runTimeStep(stepSize: number, t: number): void {
    if (this.people.length === 0) return;
    const c = this.census.counts;
    const N = Math.max(1, c.N);  // alive denominator; N = 0 would be all-dead world
    const time = t * stepSize;
    const dt = stepSize;
    const survivors: Person[] = [];

    // Build the competing-risk rate vector for this compartment ONCE per
    // tick and convert to exact discrete-time first-event probabilities.
    // The result is `[p_stay, p_event_1, …, p_event_K]`. Each person draws
    // a single uniform and indexes into this categorical.
    //
    // This is the EXACT formula
    //   p_stay     = exp(−Λ · dt)
    //   p_event_i  = (λ_i / Λ) · (1 − p_stay)
    // implemented in `general/random-variables.competingRisks`. It is
    // unbiased for any dt, unlike the linear approximation `λ_i · dt`
    // used in many naive DES kernels (which carries Λ·dt/2 bias per tick).
    let outcomePMF: number[];
    let dests: Compartment[];

    if (this.kind === 'S') {
      const lambda_A = this.params.beta_A * (c.A + c.AB) / N;
      const lambda_B = this.params.beta_B * (c.B + c.AB) / N;
      outcomePMF = competingRisks([lambda_A, lambda_B], dt);
      dests = [this.destA, this.destB];
      // The S compartment also has the joint event "got both" — at
      // first-event resolution this collapses to "whichever fires first
      // in this tick"; the (small) joint-fire probability is folded into
      // the marginal of the first event. To preserve the original
      // "dual exposure ⇒ AB" semantics, emit AB for any person who would
      // have entered both A and B in the linear-approximation regime.
      // Since competing-risks already commits to ONE event, the joint
      // event is captured probabilistically by promoting the rare second
      // exposure during the interval — modelled by an additional
      // independent Bernoulli draw against the OTHER lambda over the
      // remaining (1 − p_first) probability mass. Specifically: after
      // resolving the first event (A or B), we re-sample the second
      // disease's per-person infection probability over the same dt; if
      // it fires, the person becomes AB instead.
      for (const p of this.people) {
        const idx = sampleCategorical(outcomePMF, this.rng);
        if (idx === 0) { survivors.push(p); continue; }
        // First disease was idx = 1 (A) or 2 (B). Test the other in this
        // dt with its own marginal Bernoulli.
        let final: 'A' | 'B' | 'AB' = idx === 1 ? 'A' : 'B';
        if (idx === 1) {
          // Got A. Test for B independently.
          const pB = 1 - Math.exp(-lambda_B * dt);
          if (this.rng() < pB) final = 'AB';
        } else {
          // Got B. Test for A independently.
          const pA = 1 - Math.exp(-lambda_A * dt);
          if (this.rng() < pA) final = 'AB';
        }
        p.transition(final, time);
        if (final === 'A')      this.destA.takeItem(p);
        else if (final === 'B') this.destB.takeItem(p);
        else                    this.destAB.takeItem(p);
      }
    } else if (this.kind === 'A') {
      // Competing risks: A→AB (acquires B), A→R (recovers), A→D (dies).
      const lambda_AB = this.params.beta_B * (c.B + c.AB) / N;
      const lambda_R  = this.params.gamma_A * (1 - this.params.p_death_A);
      const lambda_D  = this.params.gamma_A *      this.params.p_death_A;
      outcomePMF = competingRisks([lambda_AB, lambda_R, lambda_D], dt);
      dests = [this.destAB, this.destR, this.destD];
      this.applyOutcomes(outcomePMF, dests, ['AB', 'R', 'D'], time, survivors);
    } else if (this.kind === 'B') {
      const lambda_AB = this.params.beta_A * (c.A + c.AB) / N;
      const lambda_R  = this.params.gamma_B * (1 - this.params.p_death_B);
      const lambda_D  = this.params.gamma_B *      this.params.p_death_B;
      outcomePMF = competingRisks([lambda_AB, lambda_R, lambda_D], dt);
      dests = [this.destAB, this.destR, this.destD];
      this.applyOutcomes(outcomePMF, dests, ['AB', 'R', 'D'], time, survivors);
    } else if (this.kind === 'AB') {
      const lambda_R = this.params.gamma_AB * (1 - this.params.p_death_AB);
      const lambda_D = this.params.gamma_AB *      this.params.p_death_AB;
      outcomePMF = competingRisks([lambda_R, lambda_D], dt);
      dests = [this.destR, this.destD];
      this.applyOutcomes(outcomePMF, dests, ['R', 'D'], time, survivors);
    } else {
      // R, D: absorbing.
      return;
    }
    this.people = survivors;
  }

  /**
   * Helper for non-S compartments. `outcomePMF[0]` is "stay"; index k > 0
   * means transition to `dests[k-1]` and `kinds[k-1]`. People who stay
   * are appended to `survivors`.
   */
  private applyOutcomes(
    outcomePMF: ReadonlyArray<number>,
    dests: Compartment[],
    kinds: CompartmentId[],
    time: number,
    survivors: Person[],
  ): void {
    for (const p of this.people) {
      const idx = sampleCategorical(outcomePMF, this.rng);
      if (idx === 0) { survivors.push(p); continue; }
      const k = idx - 1;
      p.transition(kinds[k], time);
      dests[k].takeItem(p);
    }
  }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

export interface TwoDiseaseTrace {
  t: number[];
  S: number[]; A: number[]; B: number[]; AB: number[]; R: number[]; D: number[];
}

export interface TwoDiseaseResult {
  params: TwoDiseaseParams;
  trace: TwoDiseaseTrace;
  finalCounts: {S: number; A: number; B: number; AB: number; R: number; D: number};
  // For each person, which compartments they passed through and final state.
  perPerson: Array<{
    id: number;
    final: CompartmentId;
    everA: boolean; everB: boolean; everAB: boolean;
  }>;
}

/**
 * Build the simulation graph and run it.
 */
export function runTwoDisease(params: TwoDiseaseParams): TwoDiseaseResult {
  return withSeed(params.seed, () => {
    const rng = mulberry32(params.seed);

    const S = new Compartment('S',  'S',  params, rng, undefined as any);
    const A = new Compartment('A',  'A',  params, rng, undefined as any);
    const B = new Compartment('B',  'B',  params, rng, undefined as any);
    const AB = new Compartment('AB','AB', params, rng, undefined as any);
    const R = new Compartment('R',  'R',  params, rng, undefined as any);
    const D = new Compartment('D',  'D',  params, rng, undefined as any);
    const census = new WorldCensus('census', S, A, B, AB, R, D);
    for (const c of [S, A, B, AB, R, D]) c.census = census;
    for (const c of [S, A, B, AB, R, D]) {
      c.destA = A; c.destB = B; c.destAB = AB; c.destR = R; c.destD = D;
    }

    // Seed populations.
    let nextId = 0;
    const initS = params.N - params.initialA - params.initialB - params.initialAB;
    if (initS < 0) throw new Error('initial A + B + AB exceed N');
    for (let i = 0; i < initS; i++) S.people.push(new Person(nextId++));
    for (let i = 0; i < params.initialA; i++) {
      const p = new Person(nextId++); p.transition('A', 0); A.people.push(p);
    }
    for (let i = 0; i < params.initialB; i++) {
      const p = new Person(nextId++); p.transition('B', 0); B.people.push(p);
    }
    for (let i = 0; i < params.initialAB; i++) {
      const p = new Person(nextId++); p.transition('AB', 0); AB.people.push(p);
    }
    const allPeople: Person[] = [
      ...S.people, ...A.people, ...B.people, ...AB.people,
    ];

    const compartments = [S, A, B, AB, R, D];
    const trace: TwoDiseaseTrace = {t: [], S: [], A: [], B: [], AB: [], R: [], D: []};

    const N_steps = Math.round(params.simT / params.stepSize);
    for (let t = 0; t < N_steps; t++) {
      // 1. Census reads frozen counts.
      census.runTimeStep(params.stepSize, t);
      // 2. Compartments process in shuffled order using frozen counts.
      const order = [...compartments];
      for (const _ of fisherYatesShuffle(order)) { /* drain */ }
      for (const c of order) c.runTimeStep(params.stepSize, t);
      // 3. Commit pending.
      for (const c of compartments) c.commit();
      // 4. Record trace at integer time steps.
      const time = (t + 1) * params.stepSize;
      trace.t.push(time);
      trace.S.push(S.people.length);
      trace.A.push(A.people.length);
      trace.B.push(B.people.length);
      trace.AB.push(AB.people.length);
      trace.R.push(R.people.length);
      trace.D.push(D.people.length);
    }

    return {
      params,
      trace,
      finalCounts: {
        S: S.people.length, A: A.people.length, B: B.people.length,
        AB: AB.people.length, R: R.people.length, D: D.people.length,
      },
      perPerson: allPeople.map(p => ({
        id: p.id,
        final: p.state,
        everA:  p.history.some(h => h.state === 'A'  || h.state === 'AB'),
        everB:  p.history.some(h => h.state === 'B'  || h.state === 'AB'),
        everAB: p.history.some(h => h.state === 'AB'),
      })),
    };
  });
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

async function main() {
  const params: TwoDiseaseParams = {
    N: Number(process.env.N ?? 1000),
    initialA: Number(process.env.INIT_A ?? 5),
    initialB: Number(process.env.INIT_B ?? 5),
    initialAB: Number(process.env.INIT_AB ?? 0),
    beta_A:  Number(process.env.BETA_A  ?? 0.5),
    beta_B:  Number(process.env.BETA_B  ?? 0.4),
    gamma_A: Number(process.env.GAMMA_A ?? 1/7),
    gamma_B: Number(process.env.GAMMA_B ?? 1/10),
    gamma_AB: Number(process.env.GAMMA_AB ?? 1/8),
    p_death_A:  Number(process.env.P_D_A ?? 0.40),
    p_death_B:  Number(process.env.P_D_B ?? 0.60),
    p_death_AB: Number(process.env.P_D_AB ?? 0.50),
    simT: Number(process.env.SIM_T ?? 200),
    stepSize: Number(process.env.STEPSIZE ?? 0.1),
    seed: Number(process.env.SEED ?? 1),
  };
  console.log(`# Two-disease epidemic`);
  console.log(`#   N=${params.N} initial A=${params.initialA} B=${params.initialB} AB=${params.initialAB}`);
  console.log(`#   β_A=${params.beta_A} β_B=${params.beta_B}`);
  console.log(`#   γ_A=${params.gamma_A.toFixed(4)} γ_B=${params.gamma_B.toFixed(4)} γ_AB=${params.gamma_AB.toFixed(4)}`);
  console.log(`#   p_death A=${params.p_death_A} B=${params.p_death_B} AB=${params.p_death_AB}`);
  console.log(`#   simT=${params.simT} dt=${params.stepSize} seed=${params.seed}`);

  // Multiple seeds for ensemble.
  const reps = Number(process.env.REPS ?? 30);
  const traces: TwoDiseaseTrace[] = [];
  const finalDeaths: number[] = [];
  const finalRecovered: number[] = [];
  const fractionEverAB: number[] = [];
  const perPersonDeathFlags: number[][] = [];   // per rep, 0/1 array length N
  const t0 = Date.now();
  for (let r = 0; r < reps; r++) {
    const cfg = {...params, seed: params.seed + r};
    const result = runTwoDisease(cfg);
    traces.push(result.trace);
    finalDeaths.push(result.finalCounts.D);
    finalRecovered.push(result.finalCounts.R);
    fractionEverAB.push(result.perPerson.filter(p => p.everAB).length / params.N);
    perPersonDeathFlags.push(result.perPerson.map(p => p.final === 'D' ? 1 : 0));
  }
  const ms = Date.now() - t0;
  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const std = (xs: number[]) => {
    const m = mean(xs);
    return Math.sqrt(xs.reduce((s, v) => s + (v - m)*(v - m), 0) / Math.max(1, xs.length - 1));
  };
  console.log('');
  console.log(`# ${reps} replications, ${ms} ms total`);
  console.log(`#   final D : mean=${mean(finalDeaths).toFixed(2)}  std=${std(finalDeaths).toFixed(2)}`);
  console.log(`#   final R : mean=${mean(finalRecovered).toFixed(2)}  std=${std(finalRecovered).toFixed(2)}`);
  console.log(`#   ever AB : mean=${(mean(fractionEverAB)*100).toFixed(2)}%  std=${(std(fractionEverAB)*100).toFixed(2)}pp`);
  // Conservation check: D + R + still-infected should equal N at the end.
  const r0 = runTwoDisease(params);
  const f = r0.finalCounts;
  console.log(`#   conservation: S+A+B+AB+R+D = ${f.S + f.A + f.B + f.AB + f.R + f.D}, N = ${params.N}`);

  // Cross-check via Poisson-binomial: average per-person final-death
  // probability from the simulation, build a Poisson-binomial PMF for
  // total deaths, compare its mean / std to the empirical histogram.
  // This is the "sum of Bernoullis = convolution of their PMFs" identity
  // applied to validate the simulation's variance scaling.
  if (reps >= 5) {
    const perPersonProbs: number[] = new Array(params.N).fill(0);
    for (const flags of perPersonDeathFlags) {
      for (let i = 0; i < params.N; i++) perPersonProbs[i] += flags[i] / reps;
    }
    const pb = poissonBinomialPMF(perPersonProbs);
    const pbMean = meanFromPMF(pb);
    const pbStd  = Math.sqrt(varianceFromPMF(pb));
    console.log('');
    console.log(`#   Poisson-binomial cross-check (assumes per-person deaths are ~independent):`);
    console.log(`#     simulation:  E[D] = ${mean(finalDeaths).toFixed(2)}  std = ${std(finalDeaths).toFixed(2)}`);
    console.log(`#     PB model  :  E[D] = ${pbMean.toFixed(2)}  std = ${pbStd.toFixed(2)}`);
    // Note: PB std is a LOWER BOUND because in reality cross-person
    // correlation (epidemic dynamics couple them) widens the distribution.
    // Simulation std should be ≥ PB std; if it's much larger, that's the
    // expected epidemic-coupling effect.
  }

  // Dump for downstream analysis.
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  // Compute mean trajectory across reps (point-wise).
  const T = traces[0].t.length;
  const meanTrace: TwoDiseaseTrace = {t: [...traces[0].t], S: [], A: [], B: [], AB: [], R: [], D: []};
  for (let i = 0; i < T; i++) {
    meanTrace.S.push(mean(traces.map(tr => tr.S[i])));
    meanTrace.A.push(mean(traces.map(tr => tr.A[i])));
    meanTrace.B.push(mean(traces.map(tr => tr.B[i])));
    meanTrace.AB.push(mean(traces.map(tr => tr.AB[i])));
    meanTrace.R.push(mean(traces.map(tr => tr.R[i])));
    meanTrace.D.push(mean(traces.map(tr => tr.D[i])));
  }
  const outPath = path.join(outDir, 'two-disease-framework.json');
  fs.writeFileSync(outPath, JSON.stringify({
    params, reps,
    finalDeaths, finalRecovered, fractionEverAB,
    meanTrace, traces,
  }));
  console.log(`# wrote ${outPath}`);

  // ----- Optional animation -------------------------------------------------
  // ANIMATE=1     → record + render an HTML animation of replication 0
  // ANIMATE_REP=k → use replication k as the animation source (default 0)
  if (process.env.ANIMATE === '1') {
    const which = Math.min(reps - 1, Number(process.env.ANIMATE_REP ?? 0));
    const trace = traces[which];
    const framesPath = path.join(outDir, 'two-disease.frames.jsonl');
    const htmlPath   = path.join(outDir, 'two-disease.html');
    const {FrameRecorder} = await import('./animation/frame-recorder');
    const {STAGE_W, STAGE_H, buildFrame, buildCompartmentChart} = await import('./animation/scenes/two-disease-scene');
    const rec = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H,
      fps: 30,
      title: 'Two-disease epidemic — framework simulation',
      subtitle: `N=${params.N}  β_A=${params.beta_A}  β_B=${params.beta_B}  γ_AB=${params.gamma_AB}  ` +
                `p_d_AB=${params.p_death_AB}  dt=${params.stepSize}  rep=${which}`,
      liveTickLine: true,
      recordEveryTicks: Math.max(1, Math.floor(trace.t.length / 600)),
    });
    for (let i = 0; i < trace.t.length; i++) {
      const counts = {S: trace.S[i], A: trace.A[i], B: trace.B[i],
                      AB: trace.AB[i], R: trace.R[i], D: trace.D[i]};
      rec.frame(trace.t[i], i, () => buildFrame(trace.t[i], i, counts, params.N));
    }
    rec.setCharts([buildCompartmentChart(trace, params.N)]);
    await rec.finish();
    console.log(`# wrote ${htmlPath} (${rec.getFrameCount()} frames)`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
