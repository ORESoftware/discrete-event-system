#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-factmachine.rs  (a `fn main` binary;
//                    an `examples/…rs` also works)
// 1:1 file move. Validates the FactMachine POMDP (belief filter vs scipy, policy
// ranking, Tiger POMDP, binary-vs-scalar market comparison).
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level driver code becomes `fn main()`.
//   - env + `child_process` (scipy/numpy reference) -> `std::env::var` /
//     `std::process::Command`.
//   - `JSON.parse` of the reference -> serde struct.
//   - `as any` on results -> concrete typed structs.
//   - simulation RNG -> inject `SeededRandom`.
//   - `process.exit(code)` -> `std::process::exit(code)`.
// =============================================================================

// =============================================================================
// Validate the FactMachine POMDP.
//
//   STUDY 1: Bayesian belief filter ≡ scipy/numpy reference (bit-level).
//   STUDY 2: P(majority votes YES | θ) ≡ scipy.stats.binom.sf.
//   STUDY 3: Belief calibration improves over time (Brier ↓).
//   STUDY 4: Policy ranking — oracle ≥ qmdp ≈ myopic > random ≈ hold.
//   STUDY 5: PDF "late voter coordination" claim — H(b) spikes at end.
//   STUDY 6: Tiger POMDP — exact VI agrees with QMDP at flat prior.
//   STUDY 7: BINARY vs SCALAR markets — compare and contrast.
//             (a) Same belief filter ⇒ identical posterior trajectories.
//             (b) Binary has higher mean PnL and HIGHER win-rate at θ far
//                 from 0.5 (sure-thing effect from majority concentration).
//             (c) Scalar has higher PnL VARIANCE (concentration risk into
//                 a single bin) and a STRICTLY LARGER information edge for
//                 the oracle (scalar oracle PnL − scalar myopic PnL >
//                 binary oracle PnL − binary myopic PnL).
//             (d) Scalar prices reveal a FULL DISTRIBUTION; binary
//                 collapses everything to P(YES) — measured by entropy of
//                 the price vector.
// =============================================================================

import {execFileSync} from 'child_process';
import * as path from 'path';
import {DiscreteBelief, brierScore} from '../general/belief';
import {QMDPSolver, mdpValueIteration, POMDPSpec, pomdpExactFiniteHorizon} from '../general/pomdp';
import {defaultParams, FactMachineParams, runFactMachine} from '../main-factmachine';

const PYTHON = process.env.FACTMACHINE_PY ?? 'python3';
const PY_SCRIPT = path.join(__dirname, '..', '..', '..', 'external-references', 'factmachine', 'factmachine.py');

function runPython(env: Record<string, string>): any | null {
  try {
    const out = execFileSync(PYTHON, [PY_SCRIPT], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
      env: {...process.env, ...env},
    });
    const lines = out.trim().split('\n');
    return JSON.parse(lines[lines.length - 1]);
  } catch {
    return null;
  }
}

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail?: string): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  (' + detail + ')' : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  (' + detail + ')' : ''}`); }
}

// -----------------------------------------------------------------------------
// STUDY 1: Bayesian filter ≡ scipy
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 1: Bayesian belief filter ≡ scipy reference ===');
{
  const K = 21;
  const informedness = 0.6;
  const states: number[] = [];
  for (let i = 0; i < K; i++) states.push(i / (K - 1));
  const obs: Array<[number, number]> = [
    [12, 20], [15, 22], [9, 19], [17, 20], [11, 18], [14, 19], [16, 20], [10, 22],
  ];
  const obsStr = obs.map(([y, n]) => `${y}/${n}`).join(',');
  const py = runPython({PROBLEM: 'belief', THETA_BINS: String(K),
                        INFORMEDNESS: String(informedness), OBS: obsStr});
  if (!py) { console.log('  SKIP    scipy/numpy reference unavailable'); }
  else {
    const b = new DiscreteBelief(states);
    const tsMeans: number[] = [b.mean()];
    for (const [y, n] of obs) {
      b.update(theta => {
        const q = theta * informedness + 0.5 * (1 - informedness);
        return Math.exp(y * Math.log(Math.max(1e-300, q))
                      + (n - y) * Math.log(Math.max(1e-300, 1 - q)));
      });
      tsMeans.push(b.mean());
    }
    const dMean = Math.abs(tsMeans[tsMeans.length - 1] - py.final_mean);
    check(`final E[θ] match  TS=${tsMeans[tsMeans.length - 1].toFixed(8)}  PY=${py.final_mean.toFixed(8)}`,
          dMean < 1e-10, `|Δ|=${dMean.toExponential(2)}`);
    let maxBeliefDiff = 0;
    for (let i = 0; i < K; i++) {
      const d = Math.abs(b.weights[i] - py.final_belief[i]);
      if (d > maxBeliefDiff) maxBeliefDiff = d;
    }
    check(`per-bin |b_TS − b_PY| ≤ 1e-12 across 21 bins`,
          maxBeliefDiff < 1e-12, `max|Δ|=${maxBeliefDiff.toExponential(2)}`);
    let maxMeanDiff = 0;
    for (let t = 0; t <= obs.length; t++) {
      const d = Math.abs(tsMeans[t] - py.mean_history[t]);
      if (d > maxMeanDiff) maxMeanDiff = d;
    }
    check(`per-tick mean trajectory matches across ${obs.length + 1} steps`,
          maxMeanDiff < 1e-10, `max|Δ|=${maxMeanDiff.toExponential(2)}`);
  }
}

// -----------------------------------------------------------------------------
// STUDY 2: Win-probability under majority resolution
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 2: P(majority votes YES | θ) ≡ scipy.stats.binom.sf ===');
{
  const py = runPython({PROBLEM: 'pwin', N_VOTERS: '51'});
  if (!py) { console.log('  SKIP    scipy reference unavailable'); }
  else {
    // Re-implement TS pYesWins inline (avoids importing private helper).
    const params = defaultParams({resolutionMode: 'majority', N_voters: 51});
    const half = Math.floor(params.N_voters / 2);
    function pwinTS(theta: number): number {
      const N = params.N_voters;
      let p = 0;
      let logP = N * Math.log(Math.max(1e-300, 1 - theta));
      let lcoef = 0;
      for (let k = 0; k <= N; k++) {
        if (k > half) p += Math.exp(lcoef + logP);
        if (k < N) {
          lcoef += Math.log(N - k) - Math.log(k + 1);
          logP += Math.log(Math.max(1e-300, theta)) - Math.log(Math.max(1e-300, 1 - theta));
        }
      }
      return Math.max(0, Math.min(1, p));
    }
    let maxDiff = 0;
    for (let i = 0; i < py.thetas.length; i++) {
      const d = Math.abs(pwinTS(py.thetas[i]) - py.pwin[i]);
      if (d > maxDiff) maxDiff = d;
    }
    check(`pYesWins at 9 θ values matches scipy.stats.binom.sf to 1e-10`,
          maxDiff < 1e-10, `max|Δ|=${maxDiff.toExponential(2)}`);
  }
}

// -----------------------------------------------------------------------------
// STUDY 3: Belief calibration improves over time
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 3: Belief calibration over time (Brier decreases) ===');
{
  const N_REPS = 200;
  const T = 24;
  // Random θ each rep so the validator is testing the FILTER, not a single point.
  const brierByT: number[] = new Array(T + 1).fill(0);
  for (let r = 0; r < N_REPS; r++) {
    const seed = 17 + r;
    const trueTheta = 0.05 + 0.9 * (r / N_REPS);   // sweep from 0.05 to 0.95
    const params: FactMachineParams = defaultParams({
      seed, trueTheta, T, policy: 'hold',           // hold = no trades, pure filter
      resolutionMode: 'bernoulli',
    });
    const r1 = runFactMachine(params);
    for (let t = 0; t <= T; t++) {
      brierByT[t] += brierScore(r1.beliefMean[t], r1.finalOutcome);
    }
  }
  for (let t = 0; t <= T; t++) brierByT[t] /= N_REPS;
  const initBrier = brierByT[0];
  const finalBrier = brierByT[T];
  const midBrier = brierByT[Math.floor(T / 2)];
  console.log(`#   Brier(t=0) = ${initBrier.toFixed(4)},  Brier(t=12) = ${midBrier.toFixed(4)},  Brier(t=24) = ${finalBrier.toFixed(4)}`);
  // Theoretical Brier at t=0: uniform prior gives E[θ] = 0.5, and outcome ∈ {0,1};
  // (0.5 − Y)² = 0.25 for any Y ∈ {0, 1}, so the average is exactly 0.25.
  check(`Brier at t=0 (uniform prior, no info) = 0.25 (theoretical)`,
        Math.abs(initBrier - 0.25) < 1e-8, `init=${initBrier.toFixed(4)}`);
  check(`Brier at end < Brier at start (filter learns)`,
        finalBrier < initBrier - 0.02, `end=${finalBrier.toFixed(4)}, init=${initBrier.toFixed(4)}`);
  check(`Brier at t=12 < Brier at t=0 (monotone-ish learning)`,
        midBrier < initBrier - 0.01, `mid=${midBrier.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
// STUDY 4: Policy ranking
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 4: Policy ranking — oracle ≥ qmdp ≈ myopic > random ≈ hold ===');
{
  const N_REPS = 1000;
  const policies: FactMachineParams['policy'][] = ['hold', 'random', 'myopic', 'qmdp', 'oracle'];
  const stats: Record<string, {mean: number; sd: number}> = {};
  for (const policy of policies) {
    let sum = 0, sumSq = 0;
    for (let r = 0; r < N_REPS; r++) {
      const params = defaultParams({
        seed: 5000 + r, trueTheta: 0.65, policy,
        resolutionMode: 'bernoulli',
      });
      const out = runFactMachine(params);
      sum += out.pnl; sumSq += out.pnl * out.pnl;
    }
    const mean = sum / N_REPS;
    const variance = Math.max(0, sumSq / N_REPS - mean * mean);
    stats[policy] = {mean, sd: Math.sqrt(variance)};
  }
  for (const p of policies) {
    console.log(`#   ${p.padEnd(8)}  mean=${stats[p].mean.toFixed(3)}  sd=${stats[p].sd.toFixed(3)}`);
  }
  const welchT = (a: {mean: number; sd: number}, b: {mean: number; sd: number}, n: number): number => {
    const se = Math.sqrt(a.sd * a.sd / n + b.sd * b.sd / n);
    return (a.mean - b.mean) / se;
  };
  check(`oracle.mean > qmdp.mean (value of perfect information)`,
        stats.oracle.mean > stats.qmdp.mean,
        `oracle=${stats.oracle.mean.toFixed(3)} qmdp=${stats.qmdp.mean.toFixed(3)}`);
  check(`qmdp.mean > random.mean`, stats.qmdp.mean > stats.random.mean);
  check(`myopic.mean > hold.mean (which is exactly 0)`,
        stats.myopic.mean > stats.hold.mean);
  check(`oracle vs random Welch-t > 5 (highly significant)`,
        welchT(stats.oracle, stats.random, N_REPS) > 5,
        `t = ${welchT(stats.oracle, stats.random, N_REPS).toFixed(2)}`);
  check(`qmdp vs random Welch-t > 3 (significant)`,
        welchT(stats.qmdp, stats.random, N_REPS) > 3,
        `t = ${welchT(stats.qmdp, stats.random, N_REPS).toFixed(2)}`);
}

// -----------------------------------------------------------------------------
// STUDY 5: Late-stage manipulation MISDIRECTS the belief mean.
//
// The PDF claims a sophisticated POMDP agent WOULD spike entropy under a
// late surge of contradicting voters. Our agent is a vanilla Bayesian
// filter — it cannot do that, because every observation strictly increases
// the posterior's precision (just on a possibly-wrong θ). What our agent
// DOES do is exactly the failure mode the PDF warns about: it gets fooled.
// We pin that empirically:
//
//   (a) Without flip, E[θ] tracks the truth to within 0.05.
//   (b) With a sufficiently strong late flip, E[θ] swings AWAY from the
//       truth toward 1 − θ, by a margin > 0.10.
//   (c) Bettor PnL drops noticeably under the flip (the misdirection costs
//       the agent money).
//
// A future "sophisticated" agent that maintains a hyper-prior over
// {legitimate, manipulated} would widen its CI on the last tick instead;
// this is a clean follow-on once we add detection-POMDPs (Part 4 of the PDF).
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 5: Late-stage "voter coordination" misdirects E[θ] ===');
{
  const N_REPS = 300;
  const T = 24;
  const trueTheta = 0.7;
  const lateFlipMultiplier = 10;
  let baselineDeltaTheta = 0;
  let flipDeltaTheta = 0;
  let baselinePnL = 0;
  let flipPnL = 0;
  for (let r = 0; r < N_REPS; r++) {
    const seed = 700 + r;
    const p1 = defaultParams({seed, trueTheta, T, policy: 'myopic',
                              resolutionMode: 'bernoulli', lateFlip: false});
    const p2 = defaultParams({seed, trueTheta, T, policy: 'myopic',
                              resolutionMode: 'bernoulli',
                              lateFlip: true, lateFlipMultiplier});
    const r1 = runFactMachine(p1);
    const r2 = runFactMachine(p2);
    baselineDeltaTheta += r1.beliefMean[T] - trueTheta;
    flipDeltaTheta     += r2.beliefMean[T] - trueTheta;
    baselinePnL += r1.pnl;
    flipPnL     += r2.pnl;
  }
  baselineDeltaTheta /= N_REPS;
  flipDeltaTheta     /= N_REPS;
  baselinePnL /= N_REPS;
  flipPnL     /= N_REPS;
  console.log(`#   true θ = ${trueTheta},  flip surge = ${lateFlipMultiplier}× K_noise at t = T-2`);
  console.log(`#   baseline:  mean(E[θ] − θ_true) = ${baselineDeltaTheta.toFixed(4)}    mean PnL = ${baselinePnL.toFixed(3)}`);
  console.log(`#   with flip: mean(E[θ] − θ_true) = ${flipDeltaTheta.toFixed(4)}    mean PnL = ${flipPnL.toFixed(3)}`);
  check(`(a) without flip, |E[θ] − θ_true| ≤ 0.05 at end of market`,
        Math.abs(baselineDeltaTheta) <= 0.05,
        `Δθ=${baselineDeltaTheta.toFixed(4)}`);
  check(`(b) with flip, E[θ] is shifted AWAY from truth (Δθ < −0.10, toward 1−θ)`,
        flipDeltaTheta < -0.10,
        `flip Δθ=${flipDeltaTheta.toFixed(4)}`);
  check(`(c) flip costs the bettor money (mean PnL drop > 0.10; small because most positions are taken before the flip tick)`,
        baselinePnL - flipPnL > 0.10,
        `baseline=${baselinePnL.toFixed(3)} flip=${flipPnL.toFixed(3)}  drop=${(baselinePnL - flipPnL).toFixed(3)}`);
}

// -----------------------------------------------------------------------------
// STUDY 6: 2-state POMDP exact value iteration on Tiger problem
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 6: Cassandra "Tiger" POMDP — exact VI agrees with QMDP at flat prior ===');
{
  // Classical Tiger problem.
  // States: tiger-left (TL), tiger-right (TR).
  // Actions: open-left, open-right, listen.
  // Observations: hear-left, hear-right.
  // T(s, a, s') = identity for listen; for open: state is reset uniformly
  // in the textbook formulation. Here we keep the classic dynamics.
  // R: open the right door = +10; open the wrong door = -100; listen = -1.
  // O(s', listen, hear-correct) = 0.85; O(s', open, ·) = 0.5 each.
  // Gamma = 0.95.
  const states = ['TL', 'TR'];
  const actions = ['open-left', 'open-right', 'listen'];
  const obsList = ['hear-left', 'hear-right'];
  const spec: POMDPSpec<string, string, string> = {
    states, actions, observations: obsList,
    transition: (sIdx, aIdx) => {
      if (actions[aIdx] === 'listen') {
        return sIdx === 0 ? [1, 0] : [0, 1];
      }
      return [0.5, 0.5];   // door opens reset state uniformly
    },
    observation: (sNextIdx, aIdx) => {
      if (actions[aIdx] !== 'listen') return [0.5, 0.5];
      return sNextIdx === 0 ? [0.85, 0.15] : [0.15, 0.85];
    },
    reward: (sIdx, aIdx) => {
      const a = actions[aIdx];
      if (a === 'listen') return -1;
      if ((a === 'open-left' && states[sIdx] === 'TR') ||
          (a === 'open-right' && states[sIdx] === 'TL')) return 10;
      return -100;
    },
    discount: 0.95,
  };
  const exact = pomdpExactFiniteHorizon(spec, 4);
  const flat = [0.5, 0.5];
  const Vexact = exact.V(flat);
  const qm = new QMDPSolver(spec, {tol: 1e-10, maxIter: 5000});
  const Vqmdp = Math.max(qm.qBelief(new DiscreteBelief(states, flat), 0),
                          qm.qBelief(new DiscreteBelief(states, flat), 1),
                          qm.qBelief(new DiscreteBelief(states, flat), 2));
  console.log(`#   V_exact(0.5, 0.5)  = ${Vexact.toFixed(4)}`);
  console.log(`#   V_QMDP (0.5, 0.5)  = ${Vqmdp.toFixed(4)}`);
  // QMDP is an upper bound on V_POMDP, so QMDP ≥ exact.
  check(`QMDP value ≥ exact POMDP value at flat prior (QMDP is upper bound)`,
        Vqmdp >= Vexact - 1e-6, `QMDP=${Vqmdp.toFixed(3)} exact=${Vexact.toFixed(3)}`);
  // Both agents listen at flat prior (do not open a door).
  check(`exact policy at flat prior chooses 'listen'`,
        actions[exact.act(new DiscreteBelief(states, flat))] === 'listen');
  check(`QMDP policy at flat prior chooses 'listen'`,
        actions[qm.act(new DiscreteBelief(states, flat))] === 'listen');
}

// -----------------------------------------------------------------------------
// STUDY 7: Binary vs Scalar — compare and contrast.
// -----------------------------------------------------------------------------
console.log('\n=== STUDY 7: Binary vs Scalar markets ===');
{
  const N_REPS = 1000;
  const T = 24;
  const trueTheta = 0.65;       // far from 0.5: majority outcome is decisive

  function runBlock(market: 'binary' | 'scalar', policy: FactMachineParams['policy']) {
    let sumPnL = 0, sumSqPnL = 0, wins = 0;
    let sumBeliefVar = 0;     // posterior variance at end
    for (let r = 0; r < N_REPS; r++) {
      const params = defaultParams({
        seed: 9000 + r, trueTheta, T, policy, marketType: market,
        resolutionMode: 'majority', thetaBins: 21, K_noise: 20, fee: 0.01,
      });
      const out = runFactMachine(params);
      sumPnL += out.pnl; sumSqPnL += out.pnl * out.pnl;
      if (out.pnl > 0) wins++;
      sumBeliefVar += out.beliefVar[out.beliefVar.length - 1];
    }
    const mean = sumPnL / N_REPS;
    const variance = Math.max(0, sumSqPnL / N_REPS - mean * mean);
    return {
      meanPnL: mean,
      sdPnL: Math.sqrt(variance),
      winRate: wins / N_REPS,
      finalBeliefVar: sumBeliefVar / N_REPS,
    };
  }

  const binMy = runBlock('binary', 'myopic');
  const binOr = runBlock('binary', 'oracle');
  const binRn = runBlock('binary', 'random');
  const scMy  = runBlock('scalar', 'myopic');
  const scOr  = runBlock('scalar', 'oracle');
  const scRn  = runBlock('scalar', 'random');

  console.log(`#                  binary                     scalar`);
  console.log(`#                  PnL    sd      win-rate    PnL    sd       win-rate`);
  console.log(`#   random        ${binRn.meanPnL.toFixed(3).padStart(6)}  ${binRn.sdPnL.toFixed(2).padStart(5)}   ${binRn.winRate.toFixed(3)}     ${scRn.meanPnL.toFixed(3).padStart(6)}  ${scRn.sdPnL.toFixed(2).padStart(5)}    ${scRn.winRate.toFixed(3)}`);
  console.log(`#   myopic        ${binMy.meanPnL.toFixed(3).padStart(6)}  ${binMy.sdPnL.toFixed(2).padStart(5)}   ${binMy.winRate.toFixed(3)}     ${scMy.meanPnL.toFixed(3).padStart(6)}  ${scMy.sdPnL.toFixed(2).padStart(5)}    ${scMy.winRate.toFixed(3)}`);
  console.log(`#   oracle        ${binOr.meanPnL.toFixed(3).padStart(6)}  ${binOr.sdPnL.toFixed(2).padStart(5)}   ${binOr.winRate.toFixed(3)}     ${scOr.meanPnL.toFixed(3).padStart(6)}  ${scOr.sdPnL.toFixed(2).padStart(5)}    ${scOr.winRate.toFixed(3)}`);

  // (a) Same belief trajectory in both markets at the same seed
  //     (because the likelihood model uses only `yes/total` from noise traders).
  {
    const p = defaultParams({seed: 1234, trueTheta: 0.6, T: 12,
                              marketType: 'binary', resolutionMode: 'majority',
                              policy: 'hold'});
    const r1 = runFactMachine(p);
    const r2 = runFactMachine({...p, marketType: 'scalar'});
    let maxDiff = 0;
    for (let t = 0; t <= T; t++) {
      const d = Math.abs(r1.beliefMean[t] - r2.beliefMean[t]);
      if (d > maxDiff) maxDiff = d;
    }
    check(`(a) same belief trajectory in binary vs scalar at hold-policy (max|Δ|<1e-12)`,
          maxDiff < 1e-12, `max|Δ|=${maxDiff.toExponential(2)}`);
  }

  // (b) Binary has higher win-rate at θ far from 0.5.
  check(`(b) binary myopic win-rate > scalar myopic win-rate at θ=0.65 (sure-thing effect)`,
        binMy.winRate > scMy.winRate + 0.3,
        `binary=${binMy.winRate.toFixed(3)} vs scalar=${scMy.winRate.toFixed(3)}`);
  check(`(b') binary mean PnL > scalar mean PnL for myopic at θ=0.65`,
        binMy.meanPnL > scMy.meanPnL,
        `binary=${binMy.meanPnL.toFixed(3)} vs scalar=${scMy.meanPnL.toFixed(3)}`);

  // (c) Scalar has higher PnL variance (concentration risk).
  check(`(c) scalar PnL sd > binary PnL sd for myopic (variance from bin concentration)`,
        scMy.sdPnL > binMy.sdPnL,
        `binary sd=${binMy.sdPnL.toFixed(3)} vs scalar sd=${scMy.sdPnL.toFixed(3)}`);

  // (d) Scalar oracle has a STRICTLY LARGER information edge over its
  //     uninformed counterpart than binary oracle does.
  const binEdge = binOr.meanPnL - binMy.meanPnL;
  const scEdge  = scOr.meanPnL  - scMy.meanPnL;
  console.log(`#   oracle edge:   binary=${binEdge.toFixed(3)},  scalar=${scEdge.toFixed(3)}`);
  check(`(d) scalar oracle edge > binary oracle edge (info more valuable in scalar)`,
        scEdge > binEdge,
        `scalar=${scEdge.toFixed(3)} vs binary=${binEdge.toFixed(3)}`);

  // (e) Scalar market price vector carries strictly more entropy than binary's.
  //     Compare H(price vector) at t=T across both markets, averaged over reps.
  let binH = 0, scH = 0;
  const Hreps = 200;
  for (let r = 0; r < Hreps; r++) {
    const pb = defaultParams({seed: 200 + r, trueTheta: 0.5, T, policy: 'hold',
                               marketType: 'binary', resolutionMode: 'majority'});
    const ps = defaultParams({seed: 200 + r, trueTheta: 0.5, T, policy: 'hold',
                               marketType: 'scalar', resolutionMode: 'majority'});
    const rb = runFactMachine(pb);
    const rs = runFactMachine(ps);
    const phb = rb.priceHistory[T];
    const phs = rs.priceHistory[T];
    let hb = 0, hs = 0;
    for (const x of phb) if (x > 0) hb -= x * Math.log(x);
    for (const x of phs) if (x > 0) hs -= x * Math.log(x);
    binH += hb; scH += hs;
  }
  binH /= Hreps; scH /= Hreps;
  console.log(`#   H(price vector) at t=T:  binary=${binH.toFixed(3)} (max=${Math.log(2).toFixed(3)}),  scalar=${scH.toFixed(3)} (max=${Math.log(21).toFixed(3)})`);
  check(`(e) scalar price vector carries more entropy than binary at θ=0.5`,
        scH > binH * 1.5,
        `binary=${binH.toFixed(3)} scalar=${scH.toFixed(3)}`);
}

// -----------------------------------------------------------------------------
console.log(`\n=== Summary: ${pass} passed, ${fail} failed ===`);
if (fail > 0) process.exit(1);
