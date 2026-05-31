// RUST MIGRATION: Port file-for-file to `tests/computer_network_test.rs` as integration coverage for packet-switched network DES specs.
// Test-port notes: translate scenario checks into `#[test]` functions returning `Result<()>`; replace helper assertions with `assert!`, `assert_eq!`, and approximate-float helpers; keep network fixtures and seeds deterministic.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/computer_network_test.rs   (integration test crate)
// 1:1 file move. Tests the packet-switched network DES + its registry spec,
// so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc expect()/close()/pass-fail counters + console.log  ->  #[test] fns
//   using assert!/assert_eq!; drop the manual tally and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - close(a,b,tol) float comparison -> approx::assert_relative_eq!.
//   - packet-count conservation (generated == delivered + dropped) -> assert_eq!.
// =============================================================================

// =============================================================================
// test/computer-network-test.ts -- packet-switched network DES tests.
// =============================================================================

import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  ComputerNetworkProblem,
  NetworkHostStation,
  NetworkLinkStation,
  NetworkPacket,
  NetworkRouterStation,
  NetworkSwitchStation,
  runComputerNetworkSimulation,
} from '../general/computer-network';
import {runFromSpec} from '../general/des-registry';

let pass = 0, fail = 0;
function expect(label: string, cond: boolean, detail?: string): void {
  console.log((cond ? '  PASS' : '  FAIL') + '  ' + label + (detail ? ' -- ' + detail : ''));
  cond ? pass++ : fail++;
}
function close(label: string, a: number, b: number, tol = 1e-9): void {
  expect(label, Math.abs(a - b) <= tol, `|${a} - ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 1 -- Default computer-network topology');
// -----------------------------------------------------------------------------
{
  const p = buildDefaultComputerNetworkProblem();
  const r = runComputerNetworkSimulation(p);
  expect('generates packets', r.generatedPackets > 0, `generated=${r.generatedPackets}`);
  expect('delivers all default packets', r.deliveredPackets === r.generatedPackets,
    `delivered=${r.deliveredPackets}, generated=${r.generatedPackets}, dropped=${r.droppedPackets}`);
  expect('no packets remain active', r.activePackets === 0, `active=${r.activePackets}`);
  expect('no drops in provisioned default', r.droppedPackets === 0, `dropped=${r.droppedPackets}`);
  expect('mean latency is finite', Number.isFinite(r.meanLatencyMs), `mean=${r.meanLatencyMs}`);
  expect('p95 latency is finite', Number.isFinite(r.p95LatencyMs), `p95=${r.p95LatencyMs}`);
  expect('cost accumulates from traversed links', r.totalCost > 0, `cost=${r.totalCost}`);
  expect('throughput is positive', r.throughputMbps > 0, `throughput=${r.throughputMbps}`);
  expect('goodput is positive', r.goodputMbps > 0, `goodput=${r.goodputMbps}`);
  expect('flow stats emitted for both flows', r.flowStats.length === 2, `flows=${r.flowStats.length}`);
  expect('link stats include generated reverse links', r.linkStats.some(l => l.id === 'edge-core:rev'));
  expect('time series records traffic samples', r.timeSeries.length > 0, `samples=${r.timeSeries.length}`);
  expect('bottleneck report is available', r.bottlenecks.length > 0, `bottlenecks=${r.bottlenecks.length}`);
  expect('no invariant violations', r.invariantViolations.length === 0, r.invariantViolations.slice(0, 3).join('; '));
}

// -----------------------------------------------------------------------------
console.log('\nGroup 2 -- Stationary/movable class surface');
// -----------------------------------------------------------------------------
{
  const host = new NetworkHostStation({id: 'h'});
  const router = new NetworkRouterStation({id: 'r'});
  const sw = new NetworkSwitchStation({id: 's'});
  const link = new NetworkLinkStation({id: 'l', from: 'h', to: 'r', bandwidthMbps: 10, latencyMs: 1});
  const packet = new NetworkPacket(1, 'f', 'tcp', 'h', 'r', 1000, 1040, 0, 8);
  expect('host accepts packet movable', host.receivePacket(packet));
  expect('router has router kind', router.kind === 'router');
  expect('switch has switch kind', sw.kind === 'switch');
  expect('link accepts packet movable', link.canAcceptPacket());
  link.enqueuePacket(packet, 0);
  expect('packet records link hop', packet.hops.join('->') === 'h->r', packet.hops.join('->'));
  expect('link schedules in-flight packet', link.scheduledCount() === 1, `scheduled=${link.scheduledCount()}`);
  expect('packet records protocol', packet.snapshot().protocol === 'tcp');
}

// -----------------------------------------------------------------------------
console.log('\nGroup 3 -- Explicit cost-routing topology');
// -----------------------------------------------------------------------------
{
  const p: ComputerNetworkProblem = {
    nodes: [
      {id: 'a', kind: 'host', forwardingRatePps: 1000, queueLimitPackets: 64},
      {id: 'cheap', kind: 'router', forwardingRatePps: 1000, queueLimitPackets: 64},
      {id: 'fast', kind: 'router', forwardingRatePps: 1000, queueLimitPackets: 64},
      {id: 'b', kind: 'host', forwardingRatePps: 1000, queueLimitPackets: 64},
    ],
    links: [
      {id: 'a-cheap', from: 'a', to: 'cheap', bandwidthMbps: 100, latencyMs: 10, costPerMb: 0.001},
      {id: 'cheap-b', from: 'cheap', to: 'b', bandwidthMbps: 100, latencyMs: 10, costPerMb: 0.001},
      {id: 'a-fast', from: 'a', to: 'fast', bandwidthMbps: 100, latencyMs: 1, costPerMb: 0.100},
      {id: 'fast-b', from: 'fast', to: 'b', bandwidthMbps: 100, latencyMs: 1, costPerMb: 0.100},
    ],
    flows: [{id: 'f', source: 'a', destination: 'b', ratePps: 100, packetSizeBytes: 1000, maxPackets: 10}],
    durationMs: 100,
    dtMs: 1,
    routingMetric: 'cost',
    drainAfterSourcesMs: 500,
  };
  const r = runComputerNetworkSimulation(p);
  expect('delivers all packets on cost route', r.deliveredPackets === 10, `delivered=${r.deliveredPackets}`);
  expect('uses cheap path', r.deliveredPacketsTrace.every(pkt => pkt.hops.join('->') === 'a->cheap->b'),
    r.deliveredPacketsTrace.map(pkt => pkt.hops.join('->')).join(','));
  close('cost is two cheap 1000-byte links per packet', r.totalCost, 10 * 2 * 0.001 * 0.001, 1e-12);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 4 -- Congestion and drops');
// -----------------------------------------------------------------------------
{
  const p = buildDefaultComputerNetworkProblem();
  p.links = [
    {id: 'tiny', from: 'client-a', to: 'server', bandwidthMbps: 0.1, latencyMs: 1, queueLimitPackets: 2, costPerMb: 0},
  ];
  p.flows = [{id: 'hot', source: 'client-a', destination: 'server', ratePps: 2000, packetSizeBytes: 1500, maxPackets: 200}];
  p.durationMs = 100;
  p.drainAfterSourcesMs = 200;
  p.maxPacketsInSystem = 1000;
  const r = runComputerNetworkSimulation(p);
  expect('overloaded tiny link drops packets', r.droppedPackets > 0, `dropped=${r.droppedPackets}`);
  expect('drop trace names link overflow', r.droppedPacketsTrace.some(pkt => pkt.dropReason === 'link-queue-overflow'));
  expect('conservation holds after congestion', r.generatedPackets === r.deliveredPackets + r.droppedPackets + r.activePackets,
    `g=${r.generatedPackets}, d=${r.deliveredPackets}, x=${r.droppedPackets}, a=${r.activePackets}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 5 -- Bottleneck lab traffic buildup');
// -----------------------------------------------------------------------------
{
  const r = runComputerNetworkSimulation(buildBottleneckComputerNetworkProblem());
  expect('bottleneck lab drops some packets under overload', r.droppedPackets > 0, `dropped=${r.droppedPackets}`);
  expect('bottleneck lab records active packet buildup', r.maxActivePackets > 100, `maxActive=${r.maxActivePackets}`);
  expect('top bottleneck is the narrow WAN link', r.bottlenecks[0]?.id === 'edge-wan',
    `top=${r.bottlenecks[0]?.kind}:${r.bottlenecks[0]?.id}`);
  expect('top bottleneck reports saturation or drops',
    ['saturated link', 'drops observed', 'queueing delay', 'queue buildup'].includes(r.bottlenecks[0]?.reason ?? ''),
    `reason=${r.bottlenecks[0]?.reason}`);
  expect('time series shows buildup above initial sample',
    Math.max(...r.timeSeries.map(s => s.activePackets)) > r.timeSeries[0].activePackets,
    `first=${r.timeSeries[0]?.activePackets}, max=${Math.max(...r.timeSeries.map(s => s.activePackets))}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 6 -- Protocol overhead and goodput');
// -----------------------------------------------------------------------------
{
  const p: ComputerNetworkProblem = {
    nodes: [
      {id: 'a', kind: 'host', forwardingRatePps: 1000, queueLimitPackets: 64},
      {id: 'b', kind: 'host', forwardingRatePps: 1000, queueLimitPackets: 64},
    ],
    links: [{id: 'a-b', from: 'a', to: 'b', bandwidthMbps: 100, latencyMs: 1, queueLimitPackets: 64}],
    flows: [
      {id: 'udp', source: 'a', destination: 'b', protocol: 'udp', ratePps: 100, packetSizeBytes: 1000, maxPackets: 10},
      {id: 'http', source: 'a', destination: 'b', protocol: 'http', ratePps: 100, packetSizeBytes: 1000, maxPackets: 10},
    ],
    durationMs: 200,
    dtMs: 1,
    drainAfterSourcesMs: 500,
  };
  const r = runComputerNetworkSimulation(p);
  const udp = r.flowStats.find(f => f.id === 'udp')!;
  const http = r.flowStats.find(f => f.id === 'http')!;
  expect('UDP flow records UDP protocol', udp.protocol === 'udp');
  expect('HTTP flow records HTTP protocol', http.protocol === 'http');
  expect('HTTP wire bytes exceed UDP wire bytes for same payload',
    http.generatedBytes / Math.max(1, http.generatedPackets) > udp.generatedBytes / Math.max(1, udp.generatedPackets),
    `http=${http.generatedBytes}, udp=${udp.generatedBytes}`);
  expect('goodput does not exceed wire throughput', r.goodputMbps <= r.throughputMbps + 1e-12,
    `goodput=${r.goodputMbps}, wire=${r.throughputMbps}`);
}

// -----------------------------------------------------------------------------
console.log('\nGroup 7 -- JSON registry adapter');
// -----------------------------------------------------------------------------
{
  const spec = {
    $schema: 'des/model-spec/v1' as const,
    model: 'computer-network',
    description: 'registry smoke test',
    parameters: {builtin: 'small-enterprise' as const},
    runtime: {verbose: false},
  };
  runFromSpec(spec, {verbose: false}).then(summary => {
    const r = summary.result as {deliveredPackets: number};
    expect('registry runs computer-network model', summary.modelId === 'computer-network');
    expect('registry result delivers packets', r.deliveredPackets > 0, `delivered=${r.deliveredPackets}`);
    finish();
  }).catch(e => {
    expect('registry runs computer-network model', false, e instanceof Error ? e.message : String(e));
    finish();
  });
}

function finish(): void {
  // -----------------------------------------------------------------------------
  console.log('\nGroup 8 -- Hard precondition failures');
  // -----------------------------------------------------------------------------
  {
    let threw = false;
    try {
      const p = buildDefaultComputerNetworkProblem();
      p.flows[0] = {...p.flows[0], destination: 'missing'};
      runComputerNetworkSimulation(p);
    } catch (e) {
      threw = true;
    }
    expect('rejects flow destination outside topology', threw);

    threw = false;
    try {
      const p = buildDefaultComputerNetworkProblem();
      p.flows[0] = {...p.flows[0], source: 'edge-1'};
      runComputerNetworkSimulation(p);
    } catch (e) {
      threw = true;
    }
    expect('rejects packet flow sourced from a transit router', threw);

    threw = false;
    try {
      const p = buildDefaultComputerNetworkProblem();
      p.links[0] = {...p.links[0], bandwidthMbps: 0};
      runComputerNetworkSimulation(p);
    } catch (e) {
      threw = true;
    }
    expect('rejects zero-bandwidth link', threw);
  }

  console.log(`\n${pass + fail} checks: ${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}
