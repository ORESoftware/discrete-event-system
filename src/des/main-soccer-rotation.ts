// RUST MIGRATION: target src/bin/main_soccer_rotation.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// main-soccer-rotation.ts — 7v7 player rotation: assignment + scheduling
// solved every which way (random, per-period Hungarian, multi-period LP
// relaxation, time-boxed IP/MIP, exact MDP backward induction), match
// outcome simulated by the DES engine, animated as a pitch + bench scene.
//
// THE PROBLEM (the user's coaching problem)
// ─────────────────────────────────────────
//   - 12-player roster, 7 on field, 5 on bench, every 20-min period
//   - Each (player, position, period) has an affinity in [0, 1]
//   - Fairness: no player benched two consecutive periods
//   - Goal: choose the 4-period schedule that maximises expected match
//           performance (deterministic affinity + simulated goal diff)
//
// THE LAYERED ARCHITECTURE
// ────────────────────────
//   Layer 1 (DES, simulateMatchDES):
//     Match clock ticks every game-minute, scoring station samples
//     Poisson goal events from on-field affinity, substitution station
//     swaps lineups at period boundaries.
//
//   Layer 2 (MDP):
//     State = (t, prev-bench-set), action = chosen bench-set (disjoint
//     from prev-bench by fairness), reward = Hungarian-optimal affinity
//     of the 7 on-field players assigned to 7 positions. Solved by
//     exact backward induction.
//
//   Layer 3 (combinatorial optimisers):
//     - random schedule
//     - greedy-Hungarian per period (fairness-aware)
//     - multi-period LP relaxation rounded with Hungarian
//     - time-boxed IP/MIP branch-and-cut over the exact 0/1 program
//     - exact MDP backward induction (provably optimal for the
//       deterministic affinity objective)
//
// USAGE
// ─────
//   node dist/des/main-soccer-rotation.js
//   ANIMATE=1 node dist/des/main-soccer-rotation.js
//   N_MATCHES=200 SEED=99 node dist/des/main-soccer-rotation.js
//   LP_SOLVER=scipy:highs-ipm node dist/des/main-soccer-rotation.js
//   MIP_LP_ALGO=internal-simplex POLICY=mip node dist/des/main-soccer-rotation.js
//   POLICY=mip ANIMATE=1 node dist/des/main-soccer-rotation.js
// =============================================================================

import * as path from 'path';
import {
  buildSampleSoccerProblem, evaluateSchedule, validateScheduleStructure,
  policyRandomSchedule, policyGreedyHungarian, policyMDPVI, policyMDPVIMemoryless, policyLPRelaxed,
  policyIPMIPFeasible, evaluateSoccerPOMDPFeatures, SoccerIPMIPPolicyResult,
  runManyMatches, simulateMatchDES, welchT,
  Schedule, MatchAggregate,
} from './general/soccer-rotation';
import {LPRelaxationAlgorithm} from './general/ip-mip-des';
import {FrameRecorder} from './animation/frame-recorder';
import {STAGE_W, STAGE_H, buildSoccerFrame, buildSoccerCharts} from './animation/scenes/soccer-scene';
import {
  SOCCER_IPMIP_SOLVER_H,
  SOCCER_IPMIP_SOLVER_W,
  buildSoccerIPMIPSolverCharts,
  buildSoccerIPMIPSolverFrame,
  soccerIPMIPSolverFrameCount,
} from './animation/scenes/soccer-ipmip-solver-scene';

async function main(): Promise<void> {
  const seed = Number(process.env.SEED ?? 4242);
  const numMatches = Number(process.env.N_MATCHES ?? 100);
  const animate = process.env.ANIMATE === '1';
  const policyFilter = (process.env.POLICY ?? '').toLowerCase().trim();
  const mipTimeLimitMs = Number(process.env.MIP_TIME_LIMIT_MS ?? 30_000);
  const mipMaxNodes = Number(process.env.MIP_MAX_NODES ?? 5_000);
  const mipLPAlgorithm = (process.env.MIP_LP_ALGO ?? 'internal-simplex') as LPRelaxationAlgorithm;

  const problem = buildSampleSoccerProblem({seed});
  const mipState: {latest: SoccerIPMIPPolicyResult | null} = {latest: null};

  // ─── Banner ─────────────────────────────────────────────────────────
  console.log('# 7v7 youth soccer player rotation as combinatorial optimisation');
  console.log(`# ${problem.numPlayers} players, ${problem.numPositions} positions, ${problem.numPeriods} periods of 20 min each`);
  console.log(`# fairness constraint: no player benched two consecutive periods`);
  console.log(`# affinity tensor seed = ${seed}`);
  console.log('');
  console.log('# Per-player best position and period peak (sample of affinity tensor):');
  for (let p = 0; p < problem.numPlayers; p++) {
    let bestPos = 0, bestVal = -Infinity, bestT = 0;
    for (let pos = 0; pos < problem.numPositions; pos++)
      for (let t = 0; t < problem.numPeriods; t++)
        if (problem.affinity[p][pos][t] > bestVal) {
          bestVal = problem.affinity[p][pos][t];
          bestPos = pos;
          bestT = t;
        }
    const posName = problem.positionNames?.[bestPos] ?? bestPos.toString();
    console.log(`#   ${problem.playerNames?.[p] ?? `P${p}`}  best @ pos ${posName}, period ${bestT + 1}, affinity ${bestVal.toFixed(2)}`);
  }
  console.log('');

  // ─── Run policies ────────────────────────────────────────────────────
  const policies: Array<{name: string; build: () => Schedule; note: string}> = [
    {name: 'random', build: () => policyRandomSchedule(problem, seed + 1),
     note: 'L3: trivial baseline, ignores fairness'},
    {name: 'MDP-memoryless', build: () => policyMDPVIMemoryless(problem).schedule,
     note: 'L2 with state=(t,) — Markov but NO history → cannot express fairness'},
    {name: 'greedy-Hungarian', build: () => policyGreedyHungarian(problem, {fairnessAware: true}),
     note: 'L3: per-period bipartite assignment with fairness pre-fill'},
    {name: 'LP-relaxation', build: () => policyLPRelaxed(problem).schedule,
     note: 'L3: simplex / interior-point on the LP relaxation, Hungarian-rounded'},
    {name: 'IP/MIP-feasible', build: () => {
      mipState.latest = policyIPMIPFeasible(problem, {
        timeLimitMs: mipTimeLimitMs,
        maxNodes: mipMaxNodes,
        maxTicks: Math.max(100, mipMaxNodes * 8),
        lpAlgorithm: mipLPAlgorithm,
      });
      return mipState.latest.schedule;
    }, note: 'L3: exact 0/1 IP/MIP branch-and-cut DES; time-boxed, returns best feasible incumbent'},
    {name: 'MDP-VI-exact', build: () => policyMDPVI(problem),
     note: 'L2 with state=(t, prev-bench) — Markov + 1-period memory ⇒ fairness'},
  ];
  const filtered = policyFilter
    ? policies.filter(p => p.name.toLowerCase().includes(policyFilter))
    : policies;

  // Compute the LP upper bound separately for context.
  const lp = policyLPRelaxed(problem);
  console.log(`# LP relaxation upper bound on total deterministic affinity = ${lp.lpValue.toFixed(4)}`);
  console.log(`# (solved via ${lp.solver}, ${lp.iters} iterations)`);
  console.log('');

  const aggs: MatchAggregate[] = [];
  for (const p of filtered) {
    process.stdout.write(`# Building schedule '${p.name}' ... `);
    const t0 = Date.now();
    const schedule = p.build();
    const buildMs = Date.now() - t0;
    const err = validateScheduleStructure(problem, schedule);
    if (err) { console.log(`FAIL: ${err}`); continue; }
    const evalRes = evaluateSchedule(problem, schedule);
    const belief = evaluateSoccerPOMDPFeatures(problem, schedule);
    process.stdout.write(
      `affinity=${evalRes.affinitySum.toFixed(2)}, fairness=${evalRes.fairnessOk ? 'OK' : 'VIOLATED'}, ` +
      `beliefFresh=${belief.meanExpectedFreshOnField.toFixed(3)}, build=${buildMs}ms\n`
    );
    const latestMIP = mipState.latest;
    if (p.name === 'IP/MIP-feasible' && latestMIP) {
      console.log(
        `#   IP/MIP status=${latestMIP.mip.status}, gap=${latestMIP.mip.gap.toExponential(2)}, ` +
        `nodes=${latestMIP.mip.nodesExplored}, lpSolves=${latestMIP.mip.lpSolves}, ` +
        `lpUsage=${Object.entries(latestMIP.mip.lpAlgorithmUsage).map(([k, v]) => `${k}=${v}`).join(',') || 'none'}, ` +
        `elapsed=${latestMIP.mip.elapsedMs}ms, incumbent=${latestMIP.mip.incumbentSource ?? 'none'}` +
        `${latestMIP.usedFallback ? `, fallback=${latestMIP.fallbackReason}` : ''}`
      );
    }
    process.stdout.write(`#   simulating ${numMatches} matches ... `);
    const tSim = Date.now();
    const agg = runManyMatches(problem, schedule, p.name, numMatches, seed + 1000);
    console.log(`done in ${Date.now() - tSim}ms`);
    aggs.push(agg);
  }
  console.log('');

  // ─── Comparison table ────────────────────────────────────────────────
  console.log('# ' + '─'.repeat(94));
  console.log('# Policy comparison: deterministic affinity (offline) + simulated match outcome (DES)');
  console.log('# ' + '─'.repeat(94));
  const colName = 'policy'.padEnd(20);
  const colAff  = 'affinity'.padStart(11);
  const colGD   = 'goal diff'.padStart(11);
  const colSD   = 'sd'.padStart(8);
  const colFor  = 'gF'.padStart(7);
  const colAg   = 'gA'.padStart(7);
  const colFair = 'fairness'.padStart(11);
  const colT    = 't vs random'.padStart(13);
  console.log(`#   ${colName}${colAff}${colGD}${colSD}${colFor}${colAg}${colFair}${colT}`);
  const random = aggs.find(a => a.policyName === 'random');
  for (const a of aggs) {
    const t = random ? welchT(a.rawGoalDiffs, random.rawGoalDiffs) : NaN;
    console.log(
      `#   ${a.policyName.padEnd(20)}` +
      `${a.affinitySumDeterministic.toFixed(3).padStart(11)}` +
      `${a.meanGoalDiff.toFixed(3).padStart(11)}` +
      `${a.sdGoalDiff.toFixed(3).padStart(8)}` +
      `${a.meanGoalsFor.toFixed(2).padStart(7)}` +
      `${a.meanGoalsAgainst.toFixed(2).padStart(7)}` +
      `${(a.fairnessOk ? 'OK' : 'VIOLATED').padStart(11)}` +
      `${(Number.isNaN(t) ? '   —' : t.toFixed(2)).padStart(13)}`
    );
  }
  console.log('');

  // ─── Player-fairness audit ──────────────────────────────────────────
  console.log('# Per-player periods on bench (out of ' + problem.numPeriods + '):');
  for (const a of aggs) {
    const counts = a.benchCounts.map((c, i) => `${problem.playerNames?.[i] ?? 'P' + i}=${c}`).join(' ');
    const flag = a.fairnessOk ? '' : '  ← VIOLATES fairness';
    console.log(`#   ${a.policyName.padEnd(20)}  ${counts}${flag}`);
  }
  console.log('');

  // ─── Architectural recap ────────────────────────────────────────────
  console.log('# Architectural recap:');
  for (const p of filtered) console.log(`#   ${p.name.padEnd(20)} → ${p.note}`);
  console.log('#');
  console.log('#   Layer 1 (DES): simulateMatchDES runs 80 game-minutes per match,');
  console.log('#                  samples Poisson goal events from on-field affinity,');
  console.log('#                  triggers a substitution event at every period boundary.');
  console.log('#   Layer 2 (MDP): exact backward induction; |S| = 4 periods × C(12,5) = 3168,');
  console.log('#                  reward at each (s, a) is the Hungarian-optimal assignment.');
  console.log('#   POMDP feature: hidden fatigue belief is carried across periods for audit metrics.');
  console.log('#   Layer 3:       random / greedy-Hungarian / LP-relaxation / IP-MIP / MDP-VI.');
  console.log('');

  // ─── Optional animation of the best policy ──────────────────────────
  if (animate) {
    const best = aggs.reduce((acc, x) => x.meanGoalDiff > acc.meanGoalDiff ? x : acc, aggs[0]);
    console.log(`# Animating policy '${best.policyName}' (best mean goal diff = ${best.meanGoalDiff.toFixed(3)})`);
    const outDir = path.join(__dirname, '..', '..', 'out');
    const safePolicyName = best.policyName.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '');
    const framesPath = path.join(outDir, `soccer-${safePolicyName}.frames.jsonl`);
    const htmlPath = path.join(outDir, `soccer-${safePolicyName}.html`);
    const rec = new FrameRecorder({
      framesPath, htmlPath,
      width: STAGE_W, height: STAGE_H, fps: 6,
      title: `7v7 — ${best.policyName}`,
      subtitle: `affinity ${best.affinitySumDeterministic.toFixed(2)}, goal diff ${best.meanGoalDiff.toFixed(2)}`,
      background: '#0b1220',
    });
    const match = simulateMatchDES(problem, best.schedule, {seed: seed + 1000});
    const ts: number[] = [];
    const affs: number[] = [];
    const gFs: number[] = [];
    const gAs: number[] = [];
    for (let i = 0; i < match.trace.length; i++) {
      const tr = match.trace[i];
      let goalThisTick: 'us' | 'them' | null = null;
      for (const ev of match.goalEvents) {
        if (ev.t === tr.t) { goalThisTick = ev.side; break; }
      }
      ts.push(tr.t); affs.push(tr.affinityNow);
      gFs.push(tr.goalsForCum); gAs.push(tr.goalsAgainstCum);
      rec.frame(tr.t, i, () => buildSoccerFrame(tr.t, i, {
        t: tr.t, period: tr.period, positions: tr.positions, bench: tr.bench,
        goalsFor: tr.goalsForCum, goalsAgainst: tr.goalsAgainstCum,
        affinityNow: tr.affinityNow, goalThisTick, problem,
      }));
    }
    rec.setCharts(buildSoccerCharts(ts, affs, gFs, gAs));
    await rec.finish();
    console.log(`# Animation written to ${htmlPath}`);

    const latestMIP = mipState.latest;
    if (latestMIP) {
      const solverFramesPath = path.join(outDir, 'soccer-IP-MIP-feasible-solver.frames.jsonl');
      const solverHtmlPath = path.join(outDir, 'soccer-IP-MIP-feasible-solver.html');
      const solverRec = new FrameRecorder({
        framesPath: solverFramesPath,
        htmlPath: solverHtmlPath,
        width: SOCCER_IPMIP_SOLVER_W,
        height: SOCCER_IPMIP_SOLVER_H,
        fps: 5,
        title: '7v7 IP/MIP Solver Entities',
        subtitle: `status ${latestMIP.mip.status}, LP ${latestMIP.mip.lpAlgorithm}, nodes ${latestMIP.mip.nodesExplored}`,
        background: '#f8fafc',
      });
      const totalSolverFrames = soccerIPMIPSolverFrameCount(latestMIP.mip);
      for (let i = 0; i < totalSolverFrames; i++) {
        solverRec.frame(i, i, () => buildSoccerIPMIPSolverFrame(latestMIP.mip, i));
      }
      solverRec.setCharts(buildSoccerIPMIPSolverCharts(latestMIP.mip));
      await solverRec.finish();
      console.log(`# Solver entity animation written to ${solverHtmlPath}`);
    }
  }
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}
