#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-computer-network.rs   (fn main)
// 1:1 file move. Thin runner: packet-switched computer-network DES, prints a
// flow summary for a chosen scenario.
//
// Conversion notes (file-specific):
//   - top-level main() -> fn main(); process.env.SCENARIO -> std::env::var.
//   - delegates to general/computer-network -> use crate::des::general::
//     computer_network.
// =============================================================================

// =============================================================================
// main-computer-network.ts -- runnable packet-switched computer network DES.
// =============================================================================

import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  runComputerNetworkSimulation,
} from './general/computer-network';

function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : 'n/a';
}

function main(): void {
  const scenario = process.env.SCENARIO ?? 'bottleneck';
  const problem = scenario === 'baseline'
    ? buildDefaultComputerNetworkProblem()
    : buildBottleneckComputerNetworkProblem();
  const result = runComputerNetworkSimulation(problem);

  console.log('# Computer-network DES');
  console.log('# stationary hosts/routers/switches/links + moving packets');
  console.log(`# scenario=${scenario === 'baseline' ? 'baseline' : 'bottleneck'}`);
  console.log(`# nodes=${problem.nodes.length}, links=${problem.links.length}, flows=${problem.flows.length}`);
  console.log(`# routing=${result.routingMetric}, simulated=${fmt(result.totalSimulatedMs, 1)} ms`);
  console.log('');

  console.log('## Flow summary');
  console.log(`  generated packets: ${result.generatedPackets}`);
  console.log(`  delivered packets: ${result.deliveredPackets}`);
  console.log(`  dropped packets:   ${result.droppedPackets}`);
  console.log(`  active at stop:    ${result.activePackets}`);
  console.log(`  max active:        ${result.maxActivePackets}`);
  console.log(`  delivery ratio:    ${fmt(result.deliveryRatio, 4)}`);
  console.log(`  offered load:      ${fmt(result.offeredLoadMbps, 4)} Mbps`);
  console.log(`  wire throughput:   ${fmt(result.throughputMbps, 4)} Mbps`);
  console.log(`  goodput:           ${fmt(result.goodputMbps, 4)} Mbps`);
  console.log(`  total cost:        ${fmt(result.totalCost, 6)}`);
  console.log('');

  console.log('## Latency');
  console.log(`  mean: ${fmt(result.meanLatencyMs, 2)} ms`);
  console.log(`  p95:  ${fmt(result.p95LatencyMs, 2)} ms`);
  console.log('');

  console.log('## Per-flow stats');
  for (const f of result.flowStats) {
    console.log(`  ${f.id.padEnd(14)} ${f.protocol.padEnd(4)} ${f.source} -> ${f.destination} delivered=${String(f.deliveredPackets).padStart(4)}/${String(f.generatedPackets).padStart(4)} drops=${String(f.droppedPackets).padStart(4)} mean=${fmt(f.meanTimeInSystemMs, 2)}ms goodput=${fmt(f.goodputMbps, 3)}Mbps cost=${fmt(f.totalCost, 6)}`);
  }
  console.log('');

  console.log('## Link stats');
  for (const l of result.linkStats) {
    console.log(`  ${l.id.padEnd(18)} ${l.from} -> ${l.to} delivered=${String(l.deliveredPackets).padStart(4)} util=${fmt(l.utilization, 3)} avgInFlight=${fmt(l.avgInFlight, 2)} meanQ=${fmt(l.meanQueueDelayMs, 2)}ms maxQ=${fmt(l.maxQueueDelayMs, 2)}ms`);
  }
  console.log('');

  console.log('## Bottlenecks');
  for (const b of result.bottlenecks.slice(0, 5)) {
    const util = b.utilization === undefined ? '' : ` util=${fmt(b.utilization, 3)}`;
    console.log(`  ${b.kind.padEnd(4)} ${b.id.padEnd(18)} ${b.reason.padEnd(16)} score=${fmt(b.score, 3)}${util} avgQ=${fmt(b.avgQueue, 2)} maxQ=${fmt(b.maxQueue, 0)} meanQ=${fmt(b.meanQueueDelayMs, 2)}ms drops=${b.droppedPackets}`);
  }

  console.log('');
  console.log('## Traffic build-up samples');
  for (const s of result.timeSeries.slice(0, 6)) {
    console.log(`  t=${String(s.tMs).padStart(4)}ms active=${String(s.activePackets).padStart(4)} delivered=${String(s.deliveredPackets).padStart(4)} dropped=${String(s.droppedPackets).padStart(4)}`);
  }
  if (result.timeSeries.length > 8) {
    console.log('  ...');
    for (const s of result.timeSeries.slice(-4)) {
      console.log(`  t=${String(s.tMs).padStart(4)}ms active=${String(s.activePackets).padStart(4)} delivered=${String(s.deliveredPackets).padStart(4)} dropped=${String(s.droppedPackets).padStart(4)}`);
    }
  }

  if (result.invariantViolations.length > 0) {
    console.log('');
    console.log('## Invariant violations');
    for (const v of result.invariantViolations.slice(0, 10)) console.log(`  ${v}`);
  }
}

if (require.main === module) {
  main();
}
