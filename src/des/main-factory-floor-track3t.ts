#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-factory-floor-track3t.rs   (fn main)
// 1:1 file move. Warehouse/factory-floor sim comparing a conventional floor vs
// a Track3t-enabled floor; the smart forklift is a QMDP POMDP controller.
//
// Conversion notes (file-specific):
//   - POMDP belief over the hidden pallet location -> Vec<f64>; QMDP solve ->
//     value iteration on the MDP.
//   - stochastic location/noise sampling -> inject RandomSource/SeededRandom.
//   - imports animation + warehouse comparison modules -> use crate::des::...
//   - fs/path output -> std::fs; top-level run -> fn main.
// =============================================================================

// =============================================================================
// Track3t-style warehouse/factory-floor simulation.
//
// Runs the same job plan through:
//   1. a conventional WMS/manual lookup floor with stale/noisy location data
//   2. a Track3t-enabled floor with high-frequency, high-accuracy location data
//
// The smart forklift is modeled as a POMDP controller. It plans with QMDP:
// solve the fully observable MDP, then choose actions against the current
// belief over the hidden pallet location.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {FrameRecorder} from './animation/frame-recorder';
import {
  buildWarehouseComparisonCharts,
  buildWarehouseComparisonFrame,
  WAREHOUSE_TRACK3T_STAGE_H,
  WAREHOUSE_TRACK3T_STAGE_W,
  warehouseComparisonFrameCount,
  warehouseComparisonFrameTime,
} from './animation/scenes/warehouse-track3t-scene';
import {
  runWarehouseComparison,
  summarizeWarehouseComparison,
  TRACK3T_ARCHIVE_GROUNDING,
} from './general/factory-floor-track3t';

async function main(): Promise<void> {
  const jobs = Number(process.env.JOBS ?? 120);
  const seed = Number(process.env.SEED ?? 7);
  const animate = process.env.ANIMATE !== '0';
  const result = runWarehouseComparison({jobs, seed, recordTrace: true});

  console.log('# Factory-floor / warehouse Track3t comparison');
  console.log(`# jobs=${jobs}, seed=${seed}`);
  console.log('# model: source -> movable pallets -> smart-movable forklift -> stationary sinks');
  console.log('# controller: POMDP belief updates + QMDP over the underlying MDP');
  console.log('');
  console.log(summarizeWarehouseComparison(result));
  console.log('');
  console.log('# Improvement deltas');
  console.log(`mean cycle time reduction = ${result.deltas.meanCycleTimeReductionPct.toFixed(1)}%`);
  console.log(`throughput lift           = ${result.deltas.throughputLiftPct.toFixed(1)}%`);
  console.log(`search miss reduction     = ${result.deltas.searchMissReductionPct.toFixed(1)}%`);
  console.log(`shipping error reduction  = ${result.deltas.errorReductionPct.toFixed(1)}%`);
  console.log(`belief entropy reduction  = ${result.deltas.entropyReductionPct.toFixed(1)}%`);
  console.log('');
  console.log('# Archived Track3t grounding');
  for (const source of TRACK3T_ARCHIVE_GROUNDING) {
    console.log(`- ${source.label}: ${source.url}`);
  }

  const outDir = path.join(__dirname, '..', '..', 'out');
  fs.mkdirSync(outDir, {recursive: true});
  const jsonPath = path.join(outDir, 'factory-floor-track3t.json');
  fs.writeFileSync(jsonPath, JSON.stringify(result, null, 2));
  console.log(`# wrote ${jsonPath}`);

  if (animate) {
    const framesPath = path.join(outDir, 'factory-floor-track3t.frames.jsonl');
    const htmlPath = path.join(outDir, 'factory-floor-track3t.html');
    const framesPerTraceStep = Number(process.env.MOTION_FRAMES_PER_STEP ?? 6);
    const frameCount = Math.min(
      Number(process.env.ANIM_FRAMES ?? Number.POSITIVE_INFINITY),
      warehouseComparisonFrameCount(result, framesPerTraceStep),
    );
    const rec = new FrameRecorder({
      framesPath,
      htmlPath,
      width: WAREHOUSE_TRACK3T_STAGE_W,
      height: WAREHOUSE_TRACK3T_STAGE_H,
      fps: Number(process.env.FPS ?? 10),
      title: 'Warehouse floor: Track3t comparison',
      subtitle: '2D smart-movable forklift and pallet motion; default visual dt = 0.1 sec at 1x',
      background: '#f8fafc',
      liveTickLine: false,
    });
    for (let i = 0; i < frameCount; i++) {
      rec.frame(
        warehouseComparisonFrameTime(result, i, framesPerTraceStep),
        i,
        () => buildWarehouseComparisonFrame(result, i, framesPerTraceStep),
      );
    }
    rec.setCharts(buildWarehouseComparisonCharts(result));
    await rec.finish();
    console.log(`# wrote ${framesPath}`);
    console.log(`# wrote ${htmlPath}`);
  }
}

if (require.main === module) {
  main().catch(err => {
    console.error(err);
    process.exit(1);
  });
}
