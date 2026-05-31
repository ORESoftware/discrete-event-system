#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/validate-computer-network.rs  (a `fn main`
//                    binary; an `examples/…rs` also works)
// 1:1 file move. Runs the computer-network DES and cross-checks it against a
// dependency-free Python reference.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level code becomes `fn main()`.
//   - `fs`/`path` + external program -> `std::fs` / `std::process::Command`.
//   - `JSON.parse`/`stringify` of problem + result -> serde structs; no `as any`.
//   - `process.exit(code)` -> `std::process::exit(code)`.
// =============================================================================

// =============================================================================
// validate-computer-network.ts
//
// Runs the computer-network DES in TypeScript and cross-checks the same problem
// with a dependency-free Python reference, invoked through the sanctioned
// external-program module system.
// =============================================================================

import './external-modules';
import * as fs from 'fs';
import * as path from 'path';
import {COMPUTER_NETWORK_REFERENCE_ID} from './external-modules';
import {repoRootFromRunner, runExternalModule} from './external-program';
import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  ComputerNetworkProblem,
  ComputerNetworkResult,
  runComputerNetworkSimulation,
} from '../general/computer-network';

interface CheckRow {name: string; passed: boolean; detail?: string}

const ROOT = repoRootFromRunner();
const OUT_DIR = path.join(ROOT, 'out', 'external', 'computer-network');
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  - ' + detail : ''}`);
}

function close(name: string, actual: number, expected: number, tol = 1e-9): void {
  const diff = Math.abs(actual - expected);
  check(name, diff <= tol, `actual=${fmt(actual)} expected=${fmt(expected)} diff=${diff.toExponential(3)} tol=${tol}`);
}

function sameCount(name: string, actual: number, expected: number): void {
  check(name, actual === expected, `actual=${actual} expected=${expected}`);
}

function byId<T extends {id: string}>(xs: readonly T[]): Map<string, T> {
  return new Map(xs.map(x => [x.id, x]));
}

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toFixed(12).replace(/\.?0+$/, '') : String(n);
}

function writeProblem(name: string, problem: ComputerNetworkProblem): string {
  fs.mkdirSync(OUT_DIR, {recursive: true});
  const problemPath = path.join(OUT_DIR, `${name}-problem.json`);
  fs.writeFileSync(problemPath, JSON.stringify(problem, null, 2));
  return problemPath;
}

function runExternal(name: string, problem: ComputerNetworkProblem): ComputerNetworkResult {
  const problemPath = writeProblem(name, problem);
  const out = path.join(OUT_DIR, `${name}-reference.json`);
  const ext = runExternalModule(COMPUTER_NETWORK_REFERENCE_ID, {problem: problemPath, out});
  console.log(`  external command: ${ext.command} ${ext.args.map(a => JSON.stringify(a)).join(' ')}`);
  if (ext.stdout.trim()) console.log(ext.stdout.trim());
  if (ext.stderr.trim()) console.error(ext.stderr.trim());
  if (ext.status !== 0) {
    throw new Error(`external computer-network reference exited with status ${ext.status}`);
  }
  return JSON.parse(fs.readFileSync(out, 'utf8')).result as ComputerNetworkResult;
}

function compareScenario(name: string, problem: ComputerNetworkProblem): void {
  console.log('');
  console.log(`-- ${name} --`);
  const internal = runComputerNetworkSimulation(problem);
  const external = runExternal(name, problem);

  sameCount(`${name}: generated packets`, internal.generatedPackets, external.generatedPackets);
  sameCount(`${name}: delivered packets`, internal.deliveredPackets, external.deliveredPackets);
  sameCount(`${name}: dropped packets`, internal.droppedPackets, external.droppedPackets);
  sameCount(`${name}: active packets`, internal.activePackets, external.activePackets);
  sameCount(`${name}: max active packets`, internal.maxActivePackets, external.maxActivePackets);
  close(`${name}: delivery ratio`, internal.deliveryRatio, external.deliveryRatio);
  close(`${name}: offered load Mbps`, internal.offeredLoadMbps, external.offeredLoadMbps);
  close(`${name}: wire throughput Mbps`, internal.throughputMbps, external.throughputMbps);
  close(`${name}: goodput Mbps`, internal.goodputMbps, external.goodputMbps);
  close(`${name}: mean latency ms`, internal.meanLatencyMs, external.meanLatencyMs);
  close(`${name}: p95 latency ms`, internal.p95LatencyMs, external.p95LatencyMs);
  close(`${name}: total cost`, internal.totalCost, external.totalCost);
  close(`${name}: total simulated ms`, internal.totalSimulatedMs, external.totalSimulatedMs);

  const topInternal = internal.bottlenecks[0];
  const topExternal = external.bottlenecks[0];
  check(
    `${name}: top bottleneck agrees`,
    topInternal?.kind === topExternal?.kind && topInternal?.id === topExternal?.id && topInternal?.reason === topExternal?.reason,
    `internal=${topInternal ? `${topInternal.kind}:${topInternal.id} ${topInternal.reason}` : 'none'} external=${topExternal ? `${topExternal.kind}:${topExternal.id} ${topExternal.reason}` : 'none'}`,
  );

  const extFlows = byId(external.flowStats);
  for (const flow of internal.flowStats) {
    const ref = extFlows.get(flow.id);
    check(`${name}/${flow.id}: external flow present`, !!ref);
    if (!ref) continue;
    sameCount(`${name}/${flow.id}: generated`, flow.generatedPackets, ref.generatedPackets);
    sameCount(`${name}/${flow.id}: delivered`, flow.deliveredPackets, ref.deliveredPackets);
    sameCount(`${name}/${flow.id}: dropped`, flow.droppedPackets, ref.droppedPackets);
    close(`${name}/${flow.id}: goodput`, flow.goodputMbps, ref.goodputMbps);
    close(`${name}/${flow.id}: mean latency`, flow.meanLatencyMs, ref.meanLatencyMs);
  }

  const extLinks = byId(external.linkStats);
  for (const link of internal.linkStats) {
    const ref = extLinks.get(link.id);
    check(`${name}/${link.id}: external link present`, !!ref);
    if (!ref) continue;
    sameCount(`${name}/${link.id}: enqueued`, link.enqueuedPackets, ref.enqueuedPackets);
    sameCount(`${name}/${link.id}: dropped`, link.droppedPackets, ref.droppedPackets);
    close(`${name}/${link.id}: utilization`, link.utilization, ref.utilization, 1e-9);
    close(`${name}/${link.id}: mean queue delay`, link.meanQueueDelayMs, ref.meanQueueDelayMs, 1e-9);
  }

  check(`${name}: invariant violation lists agree`, JSON.stringify(internal.invariantViolations) === JSON.stringify(external.invariantViolations));
}

function main(): void {
  console.log('Computer-network DES: framework vs external Python reference');
  console.log('===========================================================');
  compareScenario('small-enterprise', buildDefaultComputerNetworkProblem());
  compareScenario('bottleneck-lab', buildBottleneckComputerNetworkProblem());

  console.log('');
  console.log('========================================');
  const passed = checks.filter(c => c.passed).length;
  console.log(`validate-computer-network: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main();
