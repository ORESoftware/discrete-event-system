#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-external-fel-models.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Runs non-epidemic DES models from one JSON spec and compares
// them to source-only external FEL references.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level code becomes `fn main()`.
//   - `DESModelSpec` + `runFromJsonFile` (dynamic spec loading) -> typed structs +
//     serde; no `as any`.
//   - `fs`/`path` + `child_process` -> `std::fs` / `std::process::Command`.
//   - `JSON.parse` of external payloads (TrafficFelPayload, …) -> typed `serde`
//     structs with `#[serde(rename_all=...)]` for the status unions.
//   - `console.log` results table + `process.exit` -> `println!` /
//     `std::process::exit`.
// =============================================================================

// =============================================================================
// validate-external-fel-models.ts
//
// Runs representative non-epidemic DES models from one JSON spec, then sends
// that same spec to source-only external FEL references. The goal is not to
// prove bit-for-bit equivalence for every implementation detail; it is to catch
// semantic drift in the model contracts and make aggregate stats easy to review.
// =============================================================================

import './external-modules';
import * as fs from 'fs';
import * as path from 'path';
import {DESModelSpec} from '../general/des-spec';
import {runFromJsonFile} from '../general/des-registry';
import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  ComputerNetworkProblem,
  ComputerNetworkResult,
} from '../general/computer-network';
import {TrafficNetwork} from '../general/network-flow';
import {SmartTrafficParams, SmartTrafficResult} from '../general/smart-traffic-flow';
import {
  COMPUTER_NETWORK_FEL_REFERENCE_ID,
  TRAFFIC_FEL_REFERENCE_ID,
} from './external-modules';
import {repoRootFromRunner, runExternalModule} from './external-program';

interface CheckRow {name: string; passed: boolean; detail?: string}

interface TrafficFelPayload {
  kernel: string;
  status: 'ok' | 'error';
  message?: string;
  result?: TrafficFelResult;
}

interface TrafficFelResult {
  generatedDemand: number;
  entered: number;
  exited: number;
  dropped: number;
  activeAtEnd: number;
  maxActiveCars: number;
  completionRatio: number;
  meanTravelTimeSec: number;
  p95TravelTimeSec: number;
  meanSpeedMps: number;
  eventCount: number;
}

const ROOT = repoRootFromRunner();
const OUT_DIR = path.join(ROOT, 'out', 'external-fel');
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function sameCount(name: string, actual: number, expected: number): void {
  check(name, actual === expected, `actual=${actual} expected=${expected}`);
}

function closeAbs(name: string, actual: number, expected: number, tolerance: number): void {
  const diff = Math.abs(actual - expected);
  check(name, diff <= tolerance, `actual=${fmt(actual)} expected=${fmt(expected)} diff=${fmt(diff)} tol=${fmt(tolerance)}`);
}

function closeRel(name: string, actual: number, expected: number, tolerance: number): void {
  const diff = Math.abs(actual - expected);
  const rel = diff / Math.max(1, Math.abs(actual), Math.abs(expected));
  check(name, rel <= tolerance, `actual=${fmt(actual)} expected=${fmt(expected)} rel=${rel.toFixed(3)} tol=${tolerance}`);
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x !== 0 && Math.abs(x) < 1e-4) return x.toExponential(3);
  return x.toFixed(6).replace(/\.?0+$/, '');
}

function scenarioPath(name: string, file: string): string {
  return path.join(OUT_DIR, name, file);
}

async function runInternalFromSameJson<T, P = unknown>(name: string, spec: DESModelSpec<P>): Promise<{specPath: string; result: T}> {
  const specPath = scenarioPath(name, 'input.json');
  fs.mkdirSync(path.dirname(specPath), {recursive: true});
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
  const summary = await runFromJsonFile(specPath, {verbose: false});
  check(`${name}: internal registry ran same JSON`, summary.modelId === spec.model, `model=${summary.modelId}`);
  return {specPath, result: summary.result as T};
}

function runExternalJson<T>(moduleId: string, specPath: string, outPath: string): T {
  const ext = runExternalModule(moduleId, {problem: specPath, out: outPath});
  check(`${moduleId}: process exits cleanly`, ext.status === 0, `status=${ext.status}`);
  if (ext.stdout.trim()) console.log(`  external stdout: ${ext.stdout.trim()}`);
  if (ext.stderr.trim()) console.error(ext.stderr.trim());
  check(`${moduleId}: writes output JSON`, fs.existsSync(outPath), outPath);
  return JSON.parse(fs.readFileSync(outPath, 'utf8')) as T;
}

async function compareComputerNetworkScenario(name: string, problem: ComputerNetworkProblem): Promise<void> {
  console.log('');
  console.log(`-- computer-network/${name} --`);
  const spec: DESModelSpec<{problem: ComputerNetworkProblem}> = {
    $schema: 'des/model-spec/v1',
    model: 'computer-network',
    description: `External FEL validation: computer network ${name}`,
    parameters: {problem},
    runtime: {animate: false, verbose: false},
  };
  const {specPath, result: internal} =
    await runInternalFromSameJson<ComputerNetworkResult>(`computer-network-${name}`, spec);
  const out = scenarioPath(`computer-network-${name}`, 'external-fel.json');
  const payload = runExternalJson<{kernel: string; result: ComputerNetworkResult}>(
    COMPUTER_NETWORK_FEL_REFERENCE_ID,
    specPath,
    out,
  );
  check(`${name}: external reports FEL kernel`, payload.kernel === 'python-computer-network-fel-reference', payload.kernel);
  const external = payload.result;

  sameCount(`${name}: generated packets`, internal.generatedPackets, external.generatedPackets);
  sameCount(`${name}: delivered packets`, internal.deliveredPackets, external.deliveredPackets);
  sameCount(`${name}: dropped packets`, internal.droppedPackets, external.droppedPackets);
  sameCount(`${name}: active packets`, internal.activePackets, external.activePackets);
  sameCount(`${name}: max active packets`, internal.maxActivePackets, external.maxActivePackets);
  closeAbs(`${name}: delivery ratio`, internal.deliveryRatio, external.deliveryRatio, 1e-12);
  closeAbs(`${name}: offered load Mbps`, internal.offeredLoadMbps, external.offeredLoadMbps, 1e-12);
  closeAbs(`${name}: goodput Mbps`, internal.goodputMbps, external.goodputMbps, 1e-12);
  closeAbs(`${name}: mean latency ms`, internal.meanLatencyMs, external.meanLatencyMs, 1e-9);
  closeAbs(`${name}: p95 latency ms`, internal.p95LatencyMs, external.p95LatencyMs, 1e-9);
  closeAbs(`${name}: total cost`, internal.totalCost, external.totalCost, 1e-12);
  check(
    `${name}: top bottleneck agrees`,
    internal.bottlenecks[0]?.kind === external.bottlenecks[0]?.kind &&
      internal.bottlenecks[0]?.id === external.bottlenecks[0]?.id,
    `internal=${internal.bottlenecks[0]?.kind}:${internal.bottlenecks[0]?.id} external=${external.bottlenecks[0]?.kind}:${external.bottlenecks[0]?.id}`,
  );
  check(`${name}: invariant violation lists agree`,
        JSON.stringify(internal.invariantViolations) === JSON.stringify(external.invariantViolations));
}

async function compareTrafficScenario(): Promise<void> {
  console.log('');
  console.log('-- smart-traffic-flow/signalized-corridor --');
  const params = buildSignalizedCorridorParams();
  const spec: DESModelSpec<SmartTrafficParams> = {
    $schema: 'des/model-spec/v1',
    model: 'smart-traffic-flow',
    description: 'External FEL validation: scheduled cars through a signalized corridor',
    parameters: params,
    runtime: {animate: false, verbose: false},
  };
  const {specPath, result: internal} =
    await runInternalFromSameJson<SmartTrafficResult>('smart-traffic-signalized-corridor', spec);
  const out = scenarioPath('smart-traffic-signalized-corridor', 'external-fel.json');
  const payload = runExternalJson<TrafficFelPayload>(TRAFFIC_FEL_REFERENCE_ID, specPath, out);
  check('traffic FEL payload is ok', payload.status === 'ok', payload.message ?? payload.status);
  if (!payload.result) return;
  const external = payload.result;
  const scheduled = params.scheduledTrips?.length ?? 0;

  sameCount('traffic: external reads scheduled demand', external.generatedDemand, scheduled);
  sameCount('traffic: internal entered scheduled demand', internal.entered, scheduled);
  sameCount('traffic: external entered scheduled demand', external.entered, scheduled);
  sameCount('traffic: internal has no drops in comparison scenario', internal.dropped, 0);
  sameCount('traffic: external has no drops in comparison scenario', external.dropped, 0);
  closeAbs('traffic: completed cars align', internal.exited, external.exited, 2);
  closeAbs('traffic: active-at-end aligns', internal.finalCars.length, external.activeAtEnd, 2);
  closeRel('traffic: mean travel times same broad band',
           internal.meanTravelTimeSec, external.meanTravelTimeSec, 0.65);
  closeRel('traffic: mean speeds same broad band',
           internal.meanSpeedMps, external.meanSpeedMps, 0.75);
  closeRel('traffic: max active cars same broad band',
           internal.maxActiveCars, external.maxActiveCars, 0.75);
  check('traffic: external FEL processed events', external.eventCount >= scheduled, `events=${external.eventCount}`);
  check('traffic: internal validators pass', internal.validation.every(c => c.passed));
}

function buildSignalizedCorridorParams(): SmartTrafficParams {
  const network: TrafficNetwork = {
    nodes: [
      {id: 'W', kind: 'source', x: 0, y: 1},
      {id: 'S', kind: 'source', x: 1, y: 2},
      {id: 'I', kind: 'intersection', x: 1, y: 1},
      {id: 'E', kind: 'sink', x: 2, y: 1},
      {id: 'N', kind: 'sink', x: 1, y: 0},
    ],
    lanes: [
      {id: 'W-I', from: 'W', to: 'I', lengthM: 80, speedLimitMps: 12, capacity: 8},
      {id: 'I-E', from: 'I', to: 'E', lengthM: 90, speedLimitMps: 12, capacity: 9},
      {id: 'S-I', from: 'S', to: 'I', lengthM: 70, speedLimitMps: 10, capacity: 7},
      {id: 'I-N', from: 'I', to: 'N', lengthM: 70, speedLimitMps: 10, capacity: 7},
    ],
    signals: [{
      nodeId: 'I',
      phases: [
        {name: 'main', greenLanes: ['W-I'], durationSec: 18},
        {name: 'side', greenLanes: ['S-I'], durationSec: 12},
      ],
    }],
    sources: [
      {id: 'west', nodeId: 'W', ratePerMin: 0, destinationSinkIds: ['east']},
      {id: 'south', nodeId: 'S', ratePerMin: 0, destinationSinkIds: ['north']},
    ],
    sinks: [
      {id: 'east', nodeId: 'E'},
      {id: 'north', nodeId: 'N'},
    ],
  };
  const scheduledTrips = [
    ...Array.from({length: 12}, (_, i) => ({
      departSec: i * 12,
      sourceId: 'west',
      destinationSinkId: 'east',
    })),
    ...Array.from({length: 6}, (_, i) => ({
      departSec: 6 + i * 24,
      sourceId: 'south',
      destinationSinkId: 'north',
    })),
  ].sort((a, b) => a.departSec - b.departSec || a.sourceId.localeCompare(b.sourceId));

  return {
    network,
    durationSec: 210,
    dtSec: 1,
    seed: 19,
    actorShuffleSeed: 2026,
    maxCars: 80,
    smartCarPoolSize: 100,
    spawnRateMultiplier: 0,
    scheduledTrips,
    carLengthM: 4.8,
    carWidthM: 1.8,
    laneWidthM: 3.7,
    minGapM: 2.5,
    maxAccelMps2: 2.2,
    maxDecelMps2: 4,
    maxJerkMps3: 6,
    reactionTimeSec: 0.6,
    timeHeadwaySec: 1.0,
    gridCellSizeM: 0.3048,
    accidentRiskScale: 0,
    accidentProbability: 0,
    distancePreferenceSpread: 0,
    startPreferenceSpread: 0,
  };
}

async function main(): Promise<void> {
  console.log('External FEL comparison suite');
  console.log('=============================');
  await compareComputerNetworkScenario('small-enterprise', buildDefaultComputerNetworkProblem());
  await compareComputerNetworkScenario('bottleneck-lab', buildBottleneckComputerNetworkProblem());
  await compareTrafficScenario();

  console.log('');
  console.log('========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`validate-external-fel-models: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main().catch(e => {
  console.error(e instanceof Error ? e.stack ?? e.message : String(e));
  process.exit(1);
});
