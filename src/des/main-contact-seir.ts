#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-contact-seir.rs   (fn main)
// 1:1 file move. Contact-based SEIR: explicit pairwise/triplet interactions
// vs mass-action mean field. Defines the model AND runs it.
//
// Conversion notes (file-specific):
//   - Contact-kernel selector 'mass-action'|'pairwise'|'triplet' -> enum
//     ContactKernel (match on kind).
//   - Heavy sampling (Poisson contacts, Gamma per-person c_i, Bernoulli
//     transmission) -> inject RandomSource/SeededRandom (shared::capabilities);
//     Gamma/Poisson draws -> a rand_distr crate.
//   - Top-level run -> fn main(); the Population station + kernels are reusable
//     model types that could also live in a des::... module.
// =============================================================================

// =============================================================================
// CONTACT-BASED SEIR — explicit pairwise (and triplet) interactions.
//
// THE QUESTION
// ------------
// Classical compartmental SEIR uses MASS-ACTION INCIDENCE: each S has
// per-tick infection probability 1 − exp(−β · I/N · dt), where β = c · p
// is the product of "average contacts per unit time" and "transmission
// probability per S–I contact". This *implicitly* assumes pairwise
// interaction; the actual contact pairs are marginalised away. It is
// correct in the homogeneous, large-N limit, but loses three things:
//
//   1. Finite-size variance: actual contact-partner sampling has more
//      variance than the deterministic I/N expectation.
//   2. Heterogeneous contact rates: if c_i varies across people
//      (super-spreaders), R₀ depends on the FULL distribution of c
//      via R₀ = (E[c²]/E[c]) · p · 1/γ, not just the mean.
//   3. Higher-order interactions: complex contagion (require ≥2
//      simultaneous infectious contacts to transmit) scales as I²/N²,
//      giving a sharper threshold than I/N.
//
// THIS MODULE
// -----------
// One Population station holds all people. Each tick the station calls
// `runContactKernel(...)` which is one of:
//
//   - 'mass-action': classical mean-field, identical to the existing
//     SEIR engines. P(S → E in dt) = 1 − exp(−β · I/N · dt).
//   - 'pairwise':    each S samples Poisson(c_i · dt) contacts, each
//     contact partner is uniformly random from the other N−1 people,
//     each S–I contact transmits with prob p.
//   - 'triplet':     each S samples Poisson(c_i · dt) triplet meetings,
//     each meeting picks two random others. Transmission requires BOTH
//     partners to be infectious (complex contagion); prob p applies if
//     so. Models e.g. social contagion ("adopt if ≥2 friends adopt").
//
// All three drive the same E → I (rate σ) and I → R (rate γ) dynamics.
// All three accept HETEROGENEOUS contact rates: the per-person c_i is
// drawn from Gamma(shape, scale) with mean = `contactRate` and
// coefficient of variation `contactRateCV` (cv = 0 ⇒ homogeneous).
//
// All three are unbiased (in expectation, mass-action and pairwise
// produce the same number of new infections per tick when the
// population is well-mixed and homogeneous), but their VARIANCE and
// behaviour under heterogeneity differ — see CHANGELOG and the
// validate-contact-vs-meanfield runner.
//
// FRAMEWORK FIT
// -------------
// The Population station holds the people; the contact kernel runs
// inside its `runTimeStep(dt)`. There is exactly one stationary
// entity. People are technically moving entities (carry state) but
// they don't actually transit between stations in this model — they
// all live in the Population. If we wanted spatial structure, we'd
// add multiple Populations and a migration kernel.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {mulberry32, withSeed} from './general/prng';
import {sampleGamma, samplePoisson} from './general/random-variables';

export type Kernel = 'mass-action' | 'pairwise' | 'triplet';
export type State = 'S' | 'E' | 'I' | 'R';

export interface ContactSEIRParams {
  /** Total population size. */
  N: number;
  /** Initial number of infectious individuals. */
  initialI: number;
  /** Mean contact rate per person per unit time (across the population). */
  contactRate: number;
  /** Coefficient of variation for contact rate (0 = homogeneous). */
  contactRateCV: number;
  /** Per-S-I-contact transmission probability. */
  pTransmit: number;
  /** E → I rate (1/incubation). */
  sigma: number;
  /** I → R rate (1/duration of infectiousness). */
  gamma: number;
  simT: number;
  stepSize: number;
  seed: number;
  kernel: Kernel;
}

export interface Person {
  id: number;
  state: State;
  /** Per-person contact rate (potentially heterogeneous). */
  c: number;
  /** Time first infected (exposed). NaN if never. */
  tExposed: number;
  /** Time became infectious. */
  tInfectious: number;
  /** Time recovered. */
  tRecovered: number;
  /** Number of secondary infections this person caused. */
  offspring: number;
  /** ID of the person that infected this one (or -1 for index cases / -2 for never infected). */
  infectedBy: number;
}

export interface ContactSEIRResult {
  params: ContactSEIRParams;
  trace: {t: number[]; S: number[]; E: number[]; I: number[]; R: number[]};
  totalContacts: number;
  totalTransmissions: number;
  finalAttackRate: number;
  /** Mean offspring of all individuals who ever became infectious. */
  R0_empirical: number;
  /** Mean offspring of just the index cases (the seed I). */
  R0_indexOnly: number;
  /**
   * Mean offspring of generation-2 cases (those directly infected by an
   * index seed). Bias-corrected R₀ when E[c] is heterogeneous: gen-2
   * cases were selected through contacts so their E[c | infected] is
   * inflated by (1 + CV²) over E[c]. R₀(gen-2) ≈ R₀_hom · (1 + CV²) for
   * pairwise; mass-action cannot reproduce this because infectors are
   * picked uniformly.
   */
  R0_secondGen: number;
  /** Per-person contact rates, offspring count, and infector. */
  perPerson: Array<{id: number; c: number; offspring: number; ever: boolean; infectedBy: number}>;
}

// -----------------------------------------------------------------------------
// Population: the one stationary entity.
// -----------------------------------------------------------------------------

class Population {
  people: Person[];
  rng: () => number;
  params: ContactSEIRParams;
  totalContacts = 0;
  totalTransmissions = 0;

  constructor(params: ContactSEIRParams) {
    this.params = params;
    this.rng = mulberry32(params.seed);
    this.people = [];
    // Per-person contact rates: Gamma(shape, scale) so mean = contactRate
    // and CV = contactRateCV. cv = 0 returns the deterministic mean.
    let drawC: () => number;
    if (params.contactRateCV === 0) {
      drawC = () => params.contactRate;
    } else {
      const cv2 = params.contactRateCV * params.contactRateCV;
      const shape = 1 / cv2;
      const scale = params.contactRate * cv2;
      drawC = () => sampleGamma(shape, scale, this.rng);
    }
    for (let i = 0; i < params.N; i++) {
      this.people.push({
        id: i,
        state: 'S',
        c: drawC(),
        tExposed: NaN,
        tInfectious: NaN,
        tRecovered: NaN,
        offspring: 0,
        infectedBy: -2,
      });
    }
    // Seed initialI random individuals as infectious.
    for (let i = 0; i < params.initialI; i++) {
      const p = this.people[i];
      p.state = 'I';
      p.tExposed = 0;
      p.tInfectious = 0;
      p.infectedBy = -1;
    }
  }

  runTimeStep(t: number): void {
    const dt = this.params.stepSize;

    // E → I and I → R updates (independent of contact kernel).
    // Order matters at the per-person level: we do E→I first because
    // a fresh infection in this tick should NOT immediately become
    // recoverable. We use a snapshot of state at the start of the tick
    // for both decisions.
    const stateNow = this.people.map(p => p.state);
    for (let i = 0; i < this.people.length; i++) {
      const p = this.people[i];
      if (stateNow[i] === 'E') {
        if (this.rng() < 1 - Math.exp(-this.params.sigma * dt)) {
          p.state = 'I'; p.tInfectious = (t + 1) * dt;
        }
      } else if (stateNow[i] === 'I') {
        if (this.rng() < 1 - Math.exp(-this.params.gamma * dt)) {
          p.state = 'R'; p.tRecovered = (t + 1) * dt;
        }
      }
    }

    // S → E via the chosen contact kernel.
    // We use the tick-start snapshot of who's infectious so the order
    // of S processing within the tick doesn't matter (synchronous data
    // flow, same pattern as two-disease).
    const isInfectiousAtTickStart = stateNow.map(s => s === 'I');

    if (this.params.kernel === 'mass-action') {
      this.runMassActionKernel(t, isInfectiousAtTickStart);
    } else if (this.params.kernel === 'pairwise') {
      this.runPairwiseKernel(t, isInfectiousAtTickStart);
    } else if (this.params.kernel === 'triplet') {
      this.runTripletKernel(t, isInfectiousAtTickStart);
    }
  }

  /** Mean-field. P(S → E) = 1 − exp(−β · I/N · dt) where β = c · p. */
  private runMassActionKernel(t: number, isI: boolean[]): void {
    const N = this.people.length;
    let I = 0; for (const x of isI) if (x) I++;
    if (I === 0) return;
    const beta = this.params.contactRate * this.params.pTransmit;
    const lambda = beta * I / N;
    const pInfect = 1 - Math.exp(-lambda * this.params.stepSize);
    const dt = this.params.stepSize;
    for (let i = 0; i < this.people.length; i++) {
      const p = this.people[i];
      if (p.state !== 'S') continue;
      // Per-person infection draw with the per-person contact rate so
      // mass-action correctly handles heterogeneity in expectation
      // (note: it doesn't reproduce the variance — that's the point).
      const lambdaI = (p.c / this.params.contactRate) * lambda;
      const pi = 1 - Math.exp(-lambdaI * dt);
      if (this.rng() < pi) {
        // Infector is unknown in mass-action; assign uniformly at
        // random from the infectious set to enable R₀ accounting.
        const infectorIdx = this.pickRandomInfectious(isI);
        if (infectorIdx >= 0) {
          this.people[infectorIdx].offspring++;
          p.infectedBy = infectorIdx;
        }
        p.state = 'E'; p.tExposed = (t + 1) * dt;
        this.totalTransmissions++;
      }
    }
  }

  /**
   * Explicit pairwise (SYMMETRIC). Both S and I initiate contacts at
   * half rate (so the total expected contact rate per pair (i, j) is
   * (c_i + c_j) / (2N), summing both sides). Each contact i→j with
   * (S, I) state at tick start gives an infection with probability
   * pTransmit. This is the standard symmetric pair-contact model and
   * is the right thing to compare to mass-action: in the homogeneous
   * limit β_eff = c · p · I/N · 1 = β · I/N (mass-action). Under
   * heterogeneity, a high-c person both initiates more contacts AND
   * is contacted more, so their offspring count is inflated linearly
   * in c — this is what produces the over-dispersed offspring
   * distribution ("super-spreaders") observed in real epidemics.
   *
   * The S-initiated-only variant (`pairwise-asymmetric`) is also
   * supported as a sanity check; it produces homogeneous offspring
   * even with heterogeneous c because the I's c never matters for
   * its own transmission.
   */
  private runPairwiseKernel(t: number, isI: boolean[]): void {
    const N = this.people.length;
    const dt = this.params.stepSize;
    const stateSnap = this.people.map(p => p.state);
    for (let i = 0; i < this.people.length; i++) {
      const p = this.people[i];
      // Half-rate symmetric: each side accounts for half the pair-contact rate.
      const k = samplePoisson(p.c * 0.5 * dt, this.rng);
      this.totalContacts += k;
      const myState = stateSnap[i];
      for (let j = 0; j < k; j++) {
        let other = Math.floor(this.rng() * N);
        if (other === i) other = (other + 1) % N;
        const partState = stateSnap[other];
        // Two transmission directions: S→I-partner means partner infects S;
        // I→S-partner means I infects partner. Apply each independently.
        if (myState === 'S' && partState === 'I' && this.rng() < this.params.pTransmit) {
          if (p.state === 'S') {
            this.people[other].offspring++;
            p.state = 'E'; p.tExposed = (t + 1) * dt; p.infectedBy = other;
            this.totalTransmissions++;
          }
        } else if (myState === 'I' && partState === 'S' && this.rng() < this.params.pTransmit) {
          const o = this.people[other];
          if (o.state === 'S') {
            p.offspring++;
            o.state = 'E'; o.tExposed = (t + 1) * dt; o.infectedBy = i;
            this.totalTransmissions++;
          }
        }
      }
    }
  }

  /**
   * Triplet (complex contagion). Each S samples Poisson(c_i · dt) triplet
   * meetings. Each meeting picks two random others. Transmission requires
   * BOTH partners to be infectious; if so, prob `pTransmit` applies.
   */
  private runTripletKernel(t: number, isI: boolean[]): void {
    const N = this.people.length;
    const dt = this.params.stepSize;
    for (let i = 0; i < this.people.length; i++) {
      const p = this.people[i];
      if (p.state !== 'S') continue;
      const k = samplePoisson(p.c * dt, this.rng);
      this.totalContacts += k;
      for (let j = 0; j < k; j++) {
        let a = Math.floor(this.rng() * N); if (a === i) a = (a + 1) % N;
        let b = Math.floor(this.rng() * N); if (b === i || b === a) b = (b + 2) % N;
        if (isI[a] && isI[b] && this.rng() < this.params.pTransmit) {
          // Credit the offspring evenly between both infectors.
          this.people[a].offspring += 0.5;
          this.people[b].offspring += 0.5;
          p.state = 'E'; p.tExposed = (t + 1) * dt; p.infectedBy = a;
          this.totalTransmissions++;
          break;
        }
      }
    }
  }

  private pickRandomInfectious(isI: boolean[]): number {
    let count = 0; for (const x of isI) if (x) count++;
    if (count === 0) return -1;
    let pick = Math.floor(this.rng() * count);
    for (let i = 0; i < isI.length; i++) {
      if (isI[i]) {
        if (pick === 0) return i;
        pick--;
      }
    }
    return -1;
  }
}

// -----------------------------------------------------------------------------
// Driver
// -----------------------------------------------------------------------------

export function runContactSEIR(
  params: ContactSEIRParams,
  onTick?: (people: ReadonlyArray<Person>, t: number, tick: number, totalContacts: number, totalTransmissions: number) => void,
): ContactSEIRResult {
  return withSeed(params.seed, () => {
    const pop = new Population(params);
    const nTicks = Math.round(params.simT / params.stepSize);
    const trace = {t: [] as number[], S: [] as number[], E: [] as number[],
                   I: [] as number[], R: [] as number[]};
    for (let t = 0; t < nTicks; t++) {
      pop.runTimeStep(t);
      if (onTick) onTick(pop.people, (t + 1) * params.stepSize, t + 1, pop.totalContacts, pop.totalTransmissions);
      // Record at end of tick (matches two-disease).
      let S = 0, E = 0, I = 0, R = 0;
      for (const p of pop.people) {
        if      (p.state === 'S') S++;
        else if (p.state === 'E') E++;
        else if (p.state === 'I') I++;
        else                       R++;
      }
      trace.t.push((t + 1) * params.stepSize);
      trace.S.push(S); trace.E.push(E); trace.I.push(I); trace.R.push(R);
    }
    const ever = pop.people.filter(p => p.state !== 'S' || !isNaN(p.tExposed));
    const finalAttackRate = (params.N - trace.S[trace.S.length - 1]) / params.N;
    const everInfectious = pop.people.filter(p => !isNaN(p.tInfectious));
    const R0_empirical = everInfectious.length === 0
      ? 0 : everInfectious.reduce((s, p) => s + p.offspring, 0) / everInfectious.length;
    const indexCases = pop.people.filter(p => p.infectedBy === -1);
    const indexIds = new Set(indexCases.map(p => p.id));
    const R0_indexOnly = indexCases.length === 0
      ? 0 : indexCases.reduce((s, p) => s + p.offspring, 0) / indexCases.length;
    // Generation-2: directly infected by an index seed. They were
    // selected through contacts so their E[c] is inflated → R₀ inflated.
    const gen2 = pop.people.filter(p => indexIds.has(p.infectedBy));
    const R0_secondGen = gen2.length === 0 ? 0 :
      gen2.reduce((s, p) => s + p.offspring, 0) / gen2.length;
    const perPerson = pop.people.map(p => ({
      id: p.id, c: p.c, offspring: p.offspring,
      ever: !isNaN(p.tExposed) || p.infectedBy === -1,
      infectedBy: p.infectedBy,
    }));
    return {
      params, trace,
      totalContacts: pop.totalContacts,
      totalTransmissions: pop.totalTransmissions,
      finalAttackRate,
      R0_empirical, R0_indexOnly, R0_secondGen,
      perPerson,
    };
  });
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

async function main() {
  const params: ContactSEIRParams = {
    N:             Number(process.env.N             ?? 2000),
    initialI:      Number(process.env.INITIAL_I     ?? 5),
    contactRate:   Number(process.env.CONTACT_RATE  ?? 6),
    contactRateCV: Number(process.env.CONTACT_CV    ?? 0),
    pTransmit:     Number(process.env.P_TRANSMIT    ?? 0.05),
    sigma:         Number(process.env.SIGMA         ?? 1 / 5.2),
    gamma:         Number(process.env.GAMMA         ?? 1 / 7.0),
    simT:          Number(process.env.SIM_T         ?? 120),
    stepSize:      Number(process.env.STEPSIZE      ?? 0.1),
    seed:          Number(process.env.SEED          ?? 1),
    kernel:        (process.env.KERNEL as Kernel)   ?? 'pairwise',
  };
  const reps = Number(process.env.REPS ?? 1);

  console.log(`# Contact-SEIR: kernel=${params.kernel}, N=${params.N}, ` +
              `c=${params.contactRate} (cv=${params.contactRateCV}), ` +
              `p=${params.pTransmit}, σ=${params.sigma.toFixed(3)}, γ=${params.gamma.toFixed(3)}`);
  console.log(`#   simT=${params.simT}, dt=${params.stepSize}, reps=${reps}`);

  // β = c · p, R₀ ≈ β / γ for homogeneous mass-action; for pairwise with
  // CV > 0, R₀ has a multiplicative factor (1 + CV²).
  const betaTheory = params.contactRate * params.pTransmit;
  const R0Theory = betaTheory / params.gamma;
  const R0HetFactor = 1 + params.contactRateCV * params.contactRateCV;
  console.log(`#   β = ${betaTheory.toFixed(3)},  R₀(homogeneous) ≈ ${R0Theory.toFixed(2)},  ` +
              `(1+CV²) factor = ${R0HetFactor.toFixed(2)},  R₀(heterogeneous) ≈ ${(R0Theory * R0HetFactor).toFixed(2)}`);

  const t0 = Date.now();
  const results: ContactSEIRResult[] = [];
  for (let r = 0; r < reps; r++) {
    results.push(runContactSEIR({...params, seed: params.seed + r}));
  }
  const ms = Date.now() - t0;

  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const std  = (xs: number[]) => {
    const m = mean(xs);
    return Math.sqrt(xs.reduce((s, v) => s + (v - m) * (v - m), 0) / Math.max(1, xs.length - 1));
  };
  const attackRates = results.map(r => r.finalAttackRate);
  const R0s         = results.map(r => r.R0_empirical);
  const R0idx       = results.map(r => r.R0_indexOnly);
  const totalC      = results.map(r => r.totalContacts);
  const totalT      = results.map(r => r.totalTransmissions);

  console.log('');
  console.log(`# ${reps} replication(s) in ${ms} ms`);
  console.log(`#   final attack rate : mean=${(mean(attackRates) * 100).toFixed(2)}%   std=${(std(attackRates) * 100).toFixed(2)}pp`);
  console.log(`#   R₀ (all infectives): mean=${mean(R0s).toFixed(2)}   std=${std(R0s).toFixed(2)}`);
  console.log(`#   R₀ (index cases)  : mean=${mean(R0idx).toFixed(2)}   std=${std(R0idx).toFixed(2)}`);
  if (params.kernel !== 'mass-action') {
    console.log(`#   total contacts    : mean=${mean(totalC).toFixed(0)}   std=${std(totalC).toFixed(0)}`);
    console.log(`#   total transmissions: mean=${mean(totalT).toFixed(0)}   std=${std(totalT).toFixed(0)}`);
  }

  // Dump for downstream analysis.
  const outDir = path.join(__dirname, '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, `contact-seir-${params.kernel}.json`);
  // Compute the mean trajectory across replications.
  const T = results[0].trace.t.length;
  const meanTrace = {t: [...results[0].trace.t],
                     S: new Array(T).fill(0), E: new Array(T).fill(0),
                     I: new Array(T).fill(0), R: new Array(T).fill(0)};
  for (const r of results) {
    for (let i = 0; i < T; i++) {
      meanTrace.S[i] += r.trace.S[i] / reps;
      meanTrace.E[i] += r.trace.E[i] / reps;
      meanTrace.I[i] += r.trace.I[i] / reps;
      meanTrace.R[i] += r.trace.R[i] / reps;
    }
  }
  fs.writeFileSync(outPath, JSON.stringify({
    params, reps,
    meanTrace,
    finalAttackRates: attackRates,
    R0_empirical:     R0s,
    R0_indexOnly:     R0idx,
    traces: results.map(r => r.trace),
    perPerson: results[0].perPerson,
  }));
  console.log(`# wrote ${outPath}`);

  // Optional animation: re-run rep 0 with snapshot capture and write HTML.
  if (process.env.ANIMATE === '1') {
    const {FrameRecorder} = await import('./animation/frame-recorder');
    const {STAGE_W, STAGE_H, layoutGrid, buildContactFrame, buildContactChart} =
      await import('./animation/scenes/contact-seir-scene');
    const animPath = path.join(outDir, `contact-seir-${params.kernel}.html`);
    const framesPath = path.join(outDir, `contact-seir-${params.kernel}.frames.jsonl`);
    const rec = new FrameRecorder({
      framesPath, htmlPath: animPath,
      width: STAGE_W, height: STAGE_H, fps: 30,
      title: `Contact-SEIR (${params.kernel})`,
      subtitle: `N=${params.N}  c=${params.contactRate} (cv=${params.contactRateCV})  ` +
                `p=${params.pTransmit}  σ=${params.sigma.toFixed(3)}  γ=${params.gamma.toFixed(3)}  ` +
                `dt=${params.stepSize}`,
      liveTickLine: true,
      // Cap frames at ~200 so the HTML stays under ~10 MB even at N ≈ 1000.
      // For higher fidelity, set ANIM_FRAMES env var.
      recordEveryTicks: Math.max(1, Math.floor(params.simT / params.stepSize / Number(process.env.ANIM_FRAMES ?? 200))),
    });
    const pos = layoutGrid(params.N);
    const meanC = params.contactRate;
    const trace = {t: [] as number[], S: [] as number[], E: [] as number[],
                   I: [] as number[], R: [] as number[]};
    runContactSEIR(params, (people, t, tick, contacts, transmissions) => {
      let S = 0, E = 0, I = 0, R = 0;
      for (const p of people) {
        if (p.state === 'S') S++;
        else if (p.state === 'E') E++;
        else if (p.state === 'I') I++;
        else R++;
      }
      trace.t.push(t);
      trace.S.push(S); trace.E.push(E); trace.I.push(I); trace.R.push(R);
      rec.frame(t, tick, () =>
        buildContactFrame(t, tick, people, pos, meanC, contacts, transmissions, params.kernel));
    });
    rec.setCharts([buildContactChart(trace, params.N)]);
    await rec.finish();
    console.log(`# wrote ${animPath} (${rec.getFrameCount()} frames)`);
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
