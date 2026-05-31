#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/compare_elevator_dispatch.rs.
// - Keep this as a CLI runner with a Result-returning main; map SEEDS/LAMBDAS/SIM_T parsing to clap or std::env.
// - Convert TrialAggregate to a serde-serializable struct and keep JSON output behind serde_json plus std::fs/std::path.
// - Preserve the elevator comparison loop as plain orchestration over the migrated build_schedule/run_elevator APIs.
'use strict';

// Sweep across seeds and arrival rates to quantify the value of coordinated
// dispatch versus the uncoordinated baseline. Produces a clean table and
// a JSON dump of the per-trial aggregates.
//
// Why this matters: the coordinated policy is a one-line MDP policy —
// "myopically minimise the expected number of redundant stops by skipping
// any (floor, direction) that another elevator has already claimed". This
// driver demonstrates how much of the elevator system's tail latency comes
// purely from lack of coordination.
//
// Usage:
//   npm run build
//   node dist/des/runners/compare-elevator-dispatch.js
//   # configurable:
//   SEEDS=1,2,3,4,5 LAMBDAS=0.2,0.3,0.4,0.5 SIM_T=3600 \
//     node dist/des/runners/compare-elevator-dispatch.js

import * as fs from 'fs';
import * as path from 'path';
import {ElevatorConfig, buildSchedule, runElevator} from '../main-elevator';

interface TrialAggregate {
  seed: number;
  lambda: number;
  uncoord: ReturnType<typeof runElevator>['aggregates'];
  coord:   ReturnType<typeof runElevator>['aggregates'];
}

function parseList(env: string | undefined, def: number[]): number[] {
  if (!env) return def;
  return env.split(',').map(s => Number(s.trim())).filter(x => Number.isFinite(x));
}

function pct(a: number, b: number): string {
  if (b === 0) return 'n/a';
  return ((a / b - 1) * 100).toFixed(1) + '%';
}

function main() {
  const seeds   = parseList(process.env.SEEDS,   [1, 2, 3, 4, 5]);
  const lambdas = parseList(process.env.LAMBDAS, [0.2, 0.4]);
  const simT    = Number(process.env.SIM_T ?? 1800);

  const trials: TrialAggregate[] = [];
  console.log(`# elevator dispatch comparison`);
  console.log(`#   seeds = ${seeds.join(',')}`);
  console.log(`#   λ     = ${lambdas.join(',')} arrivals/s`);
  console.log(`#   simT  = ${simT}s`);
  console.log('');
  console.log('  λ     seed   meanWait u→c     p95Wait u→c     meanTotal u→c    Δmean   Δp95');
  console.log('  ────  ────  ─────────────────  ─────────────────  ──────────────────  ──────  ──────');

  for (const lambda of lambdas) {
    for (const seed of seeds) {
      const baseCfg: Omit<ElevatorConfig, 'dispatchMode'> = {
        nFloors: 4, nElevators: 3, capacity: 8,
        floorTravelTime: 4, serviceTime: 3,
        arrivalRate: lambda, simT, stepSize: 0.5, seed,
      };
      const schedule = buildSchedule({...baseCfg, dispatchMode: 'uncoordinated'});
      const u = runElevator({...baseCfg, dispatchMode: 'uncoordinated'}, schedule);
      const c = runElevator({...baseCfg, dispatchMode: 'coordinated'},  schedule);
      const trial: TrialAggregate = {seed, lambda, uncoord: u.aggregates, coord: c.aggregates};
      trials.push(trial);
      const mw = `${u.aggregates.meanWait.toFixed(2).padStart(5)} → ${c.aggregates.meanWait.toFixed(2).padStart(5)}`;
      const pw = `${u.aggregates.p95Wait.toFixed(1).padStart(5)} → ${c.aggregates.p95Wait.toFixed(1).padStart(5)}`;
      const mt = `${u.aggregates.meanTotal.toFixed(2).padStart(6)} → ${c.aggregates.meanTotal.toFixed(2).padStart(6)}`;
      const dm = pct(c.aggregates.meanWait, u.aggregates.meanWait).padStart(7);
      const dp = pct(c.aggregates.p95Wait,  u.aggregates.p95Wait).padStart(7);
      console.log(`  ${lambda.toFixed(2)}  ${String(seed).padStart(4)}  ${mw}   ${pw}   ${mt}  ${dm} ${dp}`);
    }
  }

  // Aggregate by lambda.
  const byLambda = new Map<number, TrialAggregate[]>();
  for (const t of trials) {
    if (!byLambda.has(t.lambda)) byLambda.set(t.lambda, []);
    byLambda.get(t.lambda)!.push(t);
  }
  console.log('');
  console.log('  Per-rate summary (mean over seeds):');
  for (const [lambda, ts] of byLambda) {
    const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;
    const u_mw = mean(ts.map(t => t.uncoord.meanWait));
    const c_mw = mean(ts.map(t => t.coord.meanWait));
    const u_p95 = mean(ts.map(t => t.uncoord.p95Wait));
    const c_p95 = mean(ts.map(t => t.coord.p95Wait));
    const u_mt = mean(ts.map(t => t.uncoord.meanTotal));
    const c_mt = mean(ts.map(t => t.coord.meanTotal));
    console.log(`    λ=${lambda.toFixed(2)}: meanWait ${u_mw.toFixed(2)}→${c_mw.toFixed(2)} (${pct(c_mw,u_mw)}), p95Wait ${u_p95.toFixed(1)}→${c_p95.toFixed(1)} (${pct(c_p95,u_p95)}), meanTotal ${u_mt.toFixed(2)}→${c_mt.toFixed(2)} (${pct(c_mt,u_mt)})`);
  }

  const outDir = path.join(__dirname, '..', '..', '..', 'out');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, {recursive: true});
  const outPath = path.join(outDir, 'elevator-dispatch-sweep.json');
  fs.writeFileSync(outPath, JSON.stringify({seeds, lambdas, simT, trials}, null, 2));
  console.log('');
  console.log(`# wrote ${outPath}`);
}

main();
