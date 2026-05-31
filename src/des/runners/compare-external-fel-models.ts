#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/compare_external_fel_models.rs.
// - Keep this as a CLI comparison binary with Result-returning main; route env/config through clap or std::env.
// - Convert SharedTrafficInput, CheckRow, and EngineReport into serde structs and keep report JSON/Markdown I/O in std::fs.
// - Model runExternalModule calls as external adapter ports using std::process or tokio::process, with serde_json payload boundaries.
'use strict';

// =============================================================================
// compare-external-fel-models.ts
//
// Source/sink driven DES comparisons against external reference models. Every
// scenario writes one shared JSON input and then feeds that same file back into
// the internal runner and the external program.
// =============================================================================

import './external-modules';
import * as fs from 'fs';
import * as path from 'path';
import {mulberry32} from '../general/prng';
import {
  buildBottleneckComputerNetworkProblem,
  buildDefaultComputerNetworkProblem,
  ComputerNetworkProblem,
  ComputerNetworkResult,
  runComputerNetworkSimulation,
  validateComputerNetworkProblem,
} from '../general/computer-network';
import {
  buildFiveIntersectionTrafficNetwork,
  TrafficLane,
  TrafficNetwork,
  TrafficScheduledTrip,
} from '../general/network-flow';
import {runSmartTrafficFlow, SmartTrafficParams, SmartTrafficResult} from '../general/smart-traffic-flow';
import {
  COMPUTER_NETWORK_FEL_REFERENCE_ID,
  TRAFFIC_CIW_REFERENCE_ID,
  TRAFFIC_FEL_REFERENCE_ID,
  TRAFFIC_SIMPY_REFERENCE_ID,
  TRAFFIC_SUMO_REFERENCE_ID,
} from './external-modules';
import {repoRootFromRunner, runExternalModule} from './external-program';

interface SharedTrafficTrip extends TrafficScheduledTrip {
  id: string;
  route: string[];
  sourceNodeId: string;
  sinkNodeId: string;
}

interface SharedTrafficInput {
  schema: 'des/shared-traffic-source-sink/v1';
  model: 'smart-traffic-flow';
  params: Omit<SmartTrafficParams, 'network' | 'scheduledTrips'>;
  network: TrafficNetwork;
  trips: SharedTrafficTrip[];
  sourceInitialConditions: {
    sourceCount: number;
    sinkCount: number;
    scheduledTrips: number;
    demandEndSec: number;
  };
}

interface CheckRow {
  name: string;
  passed: boolean;
  detail: string;
}

interface EngineReport {
  domain: 'traffic' | 'computer-network';
  scenario: string;
  engine: string;
  status: 'passed' | 'failed' | 'skipped';
  inputPath: string;
  outputPath?: string;
  checks: CheckRow[];
  notes: string[];
}

const ROOT = repoRootFromRunner();
const OUT_DIR = path.join(ROOT, 'out', 'external-fel-comparison');

function main(): void {
  fs.mkdirSync(OUT_DIR, {recursive: true});
  const reports = [
    ...compareTraffic(),
    ...compareComputerNetwork(),
  ];
  const report = {
    generatedAt: new Date().toISOString(),
    sharedInputContract: 'Internal and external runs read the same JSON scenario files from out/external-fel-comparison.',
    reports,
  };
  const jsonPath = path.join(OUT_DIR, 'comparison-report.json');
  const mdPath = path.join(OUT_DIR, 'comparison-report.md');
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  fs.writeFileSync(mdPath, renderMarkdown(reports));
  console.log(renderMarkdown(reports));
  console.log(`Wrote ${path.relative(ROOT, jsonPath)}`);
  const failed = reports.filter(r => r.status === 'failed');
  if (failed.length > 0) process.exit(1);
}

function compareTraffic(): EngineReport[] {
  const network = buildFiveIntersectionTrafficNetwork();
  const params: SmartTrafficParams = {
    network,
    durationSec: 180,
    dtSec: 0.1,
    seed: 19,
    actorShuffleSeed: 2026,
    maxCars: 180,
    smartCarPoolSize: 260,
    spawnRateMultiplier: 0.35,
    carLengthM: 4.8,
    carWidthM: 1.8,
    laneWidthM: 3.7,
    minGapM: 2.5,
    maxAccelMps2: 2.2,
    maxDecelMps2: 4,
    maxJerkMps3: 6,
    reactionTimeSec: 0.8,
    timeHeadwaySec: 1.1,
    gridCellSizeM: 0.3048,
    accidentRiskScale: 0,
    accidentProbability: 0,
    distancePreferenceSpread: 0,
    startPreferenceSpread: 0,
  };
  const demandEndSec = 120;
  const trips = generateScheduledTrips(network, params, params.seed + 4242, demandEndSec);
  const input: SharedTrafficInput = {
    schema: 'des/shared-traffic-source-sink/v1',
    model: 'smart-traffic-flow',
    params: stripTrafficRuntimeInputs(params),
    network,
    trips,
    sourceInitialConditions: {
      sourceCount: network.sources.length,
      sinkCount: network.sinks.length,
      scheduledTrips: trips.length,
      demandEndSec,
    },
  };
  const inputPath = path.join(OUT_DIR, 'traffic-shared-input.json');
  fs.writeFileSync(inputPath, JSON.stringify(input, null, 2));

  const shared = JSON.parse(fs.readFileSync(inputPath, 'utf8')) as SharedTrafficInput;
  const sourceChecks = validateSharedTrafficInput(shared);
  if (sourceChecks.some(c => !c.passed)) {
    return [{
      domain: 'traffic',
      scenario: 'five-intersection-scheduled-trips',
      engine: 'source/sink input',
      status: 'failed',
      inputPath,
      checks: sourceChecks,
      notes: ['shared traffic input failed before simulation'],
    }];
  }

  const internalTrips = shared.trips.map(({departSec, sourceId, destinationSinkId}) => ({
    departSec,
    sourceId,
    destinationSinkId,
  }));
  const internal = runSmartTrafficFlow({...shared.params, network: shared.network, scheduledTrips: internalTrips});
  const reports: EngineReport[] = [
    runTrafficExternal('Python traffic FEL', TRAFFIC_FEL_REFERENCE_ID, inputPath, internal),
    runTrafficExternal('SimPy', TRAFFIC_SIMPY_REFERENCE_ID, inputPath, internal),
    runTrafficExternal('Ciw', TRAFFIC_CIW_REFERENCE_ID, inputPath, internal),
    runTrafficExternal('SUMO', TRAFFIC_SUMO_REFERENCE_ID, inputPath, internal),
  ];
  reports.unshift({
    domain: 'traffic',
    scenario: 'five-intersection-scheduled-trips',
    engine: 'source/sink input',
    status: 'passed',
    inputPath,
    checks: sourceChecks,
    notes: [`internal DES entered=${internal.entered} exited=${internal.exited} dropped=${internal.dropped}`],
  });
  return reports;
}

function runTrafficExternal(engine: string, moduleId: string, inputPath: string, internal: SmartTrafficResult): EngineReport {
  const outputPath = path.join(OUT_DIR, `${engine.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.json`);
  const ext = runExternalModule(moduleId, {problem: inputPath, out: outputPath, collisionAction: 'warn'});
  const notes = [`external command status=${ext.status}`];
  if (ext.stderr.trim()) notes.push(ext.stderr.trim().slice(0, 500));
  if (ext.status !== 0 || !fs.existsSync(outputPath)) {
    return {
      domain: 'traffic',
      scenario: 'five-intersection-scheduled-trips',
      engine,
      status: 'failed',
      inputPath,
      outputPath,
      checks: [checkRow('external process writes output JSON', false, `status=${ext.status}`)],
      notes,
    };
  }
  const payload = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
  if (payload.status === 'unavailable') {
    return {
      domain: 'traffic',
      scenario: 'five-intersection-scheduled-trips',
      engine,
      status: 'skipped',
      inputPath,
      outputPath,
      checks: [checkRow('external dependency reported unavailable cleanly', true, payload.message ?? 'unavailable')],
      notes,
    };
  }
  if (payload.status && payload.status !== 'ok') {
    return {
      domain: 'traffic',
      scenario: 'five-intersection-scheduled-trips',
      engine,
      status: 'failed',
      inputPath,
      outputPath,
      checks: [checkRow('external payload status is ok', false, payload.message ?? String(payload.status))],
      notes,
    };
  }
  const result = payload.result;
  const checks = compareTrafficStats(internal, result);
  return {
    domain: 'traffic',
    scenario: 'five-intersection-scheduled-trips',
    engine,
    status: checks.every(c => c.passed) ? 'passed' : 'failed',
    inputPath,
    outputPath,
    checks,
    notes: result.notes ?? notes,
  };
}

function compareComputerNetwork(): EngineReport[] {
  const scenarios: Array<{name: string; problem: ComputerNetworkProblem}> = [
    {name: 'small-enterprise', problem: buildDefaultComputerNetworkProblem()},
    {name: 'bottleneck-lab', problem: buildBottleneckComputerNetworkProblem()},
  ];
  const reports: EngineReport[] = [];
  for (const scenario of scenarios) {
    validateComputerNetworkProblem(scenario.problem);
    const input = {
      $schema: 'des/model-spec/v1',
      model: 'computer-network',
      description: `${scenario.name} shared source/sink packet-flow comparison`,
      parameters: {problem: scenario.problem},
      runtime: {verbose: false},
    };
    const inputPath = path.join(OUT_DIR, `computer-network-${scenario.name}.json`);
    const outputPath = path.join(OUT_DIR, `computer-network-${scenario.name}-python-fel-reference.json`);
    fs.writeFileSync(inputPath, JSON.stringify(input, null, 2));

    const shared = JSON.parse(fs.readFileSync(inputPath, 'utf8')) as typeof input;
    const internal = runComputerNetworkSimulation(shared.parameters.problem);
    const ext = runExternalModule(COMPUTER_NETWORK_FEL_REFERENCE_ID, {problem: inputPath, out: outputPath});
    const notes = [`external command status=${ext.status}`];
    if (ext.stderr.trim()) notes.push(ext.stderr.trim().slice(0, 500));
    if (ext.status !== 0 || !fs.existsSync(outputPath)) {
      reports.push({
        domain: 'computer-network',
        scenario: scenario.name,
        engine: 'Python computer-network FEL',
        status: 'failed',
        inputPath,
        outputPath,
        checks: [checkRow('external process writes output JSON', false, `status=${ext.status}`)],
        notes,
      });
      continue;
    }
    const external = JSON.parse(fs.readFileSync(outputPath, 'utf8')).result as ComputerNetworkResult;
    const checks = compareComputerNetworkStats(internal, external);
    reports.push({
      domain: 'computer-network',
      scenario: scenario.name,
      engine: 'Python computer-network FEL',
      status: checks.every(c => c.passed) ? 'passed' : 'failed',
      inputPath,
      outputPath,
      checks,
      notes,
    });
  }
  return reports;
}

function validateSharedTrafficInput(input: SharedTrafficInput): CheckRow[] {
  const checks: CheckRow[] = [];
  const sourceById = new Map(input.network.sources.map(s => [s.id, s]));
  const sinkById = new Map(input.network.sinks.map(s => [s.id, s]));
  const laneById = new Map(input.network.lanes.map(l => [l.id, l]));
  checks.push(checkRow('has source entities', input.network.sources.length > 0, `sources=${input.network.sources.length}`));
  checks.push(checkRow('has sink entities', input.network.sinks.length > 0, `sinks=${input.network.sinks.length}`));
  checks.push(checkRow('has scheduled source trips', input.trips.length > 0, `trips=${input.trips.length}`));
  for (const trip of input.trips) {
    const source = sourceById.get(trip.sourceId);
    const sink = sinkById.get(trip.destinationSinkId);
    const routeOk = source !== undefined && sink !== undefined && routeConnects(trip.route, source.nodeId, sink.nodeId, laneById);
    const allowed = source ? (source.destinationSinkIds ?? input.network.sinks.map(s => s.id)).includes(trip.destinationSinkId) : false;
    checks.push(checkRow(`${trip.id}: source exists`, source !== undefined && source.nodeId === trip.sourceNodeId, `${trip.sourceId}@${trip.sourceNodeId}`));
    checks.push(checkRow(`${trip.id}: sink exists`, sink !== undefined && sink.nodeId === trip.sinkNodeId, `${trip.destinationSinkId}@${trip.sinkNodeId}`));
    checks.push(checkRow(`${trip.id}: sink allowed by source`, allowed, `${trip.sourceId}->${trip.destinationSinkId}`));
    checks.push(checkRow(`${trip.id}: route connects source to sink`, routeOk, trip.route.join('->')));
    checks.push(checkRow(`${trip.id}: departSec is in horizon`, trip.departSec >= 0 && trip.departSec <= input.params.durationSec, `depart=${trip.departSec}`));
  }
  return checks;
}

function compareTrafficStats(internal: SmartTrafficResult, external: any): CheckRow[] {
  const checks = [
    exactNumber('generated demand matches internal scheduled input', external.generatedDemand, internal.params.scheduledTrips?.length ?? external.generatedDemand),
    relativeNumber('departures align with internal entered count', external.departed, internal.entered, 0.2),
    relativeNumber('arrivals align with internal exited count', external.arrived, internal.exited, 0.45),
    closeNumber('active-at-end aligns with internal final cars', external.activeAtEnd, internal.finalCars.length, 2),
    finiteNumber('external mean travel time is finite', external.meanTravelTimeSec),
    finiteNumber('external mean speed is finite', external.meanSpeedMps),
  ];
  if (internal.meanTravelTimeSec > 0 && external.meanTravelTimeSec > 0) {
    checks.push(ratioBand('mean travel time same order of magnitude', external.meanTravelTimeSec, internal.meanTravelTimeSec, 0.2, 5));
  }
  return checks;
}

function compareComputerNetworkStats(internal: ComputerNetworkResult, external: ComputerNetworkResult): CheckRow[] {
  const checks = [
    exactNumber('generated packets', external.generatedPackets, internal.generatedPackets),
    exactNumber('delivered packets', external.deliveredPackets, internal.deliveredPackets),
    exactNumber('dropped packets', external.droppedPackets, internal.droppedPackets),
    exactNumber('active packets', external.activePackets, internal.activePackets),
    exactNumber('max active packets', external.maxActivePackets, internal.maxActivePackets),
    closeNumber('delivery ratio', external.deliveryRatio, internal.deliveryRatio, 1e-9),
    closeNumber('offered load Mbps', external.offeredLoadMbps, internal.offeredLoadMbps, 1e-9),
    closeNumber('wire throughput Mbps', external.throughputMbps, internal.throughputMbps, 1e-9),
    closeNumber('goodput Mbps', external.goodputMbps, internal.goodputMbps, 1e-9),
    closeNumber('mean latency ms', external.meanLatencyMs, internal.meanLatencyMs, 1e-9),
    closeNumber('p95 latency ms', external.p95LatencyMs, internal.p95LatencyMs, 1e-9),
    closeNumber('total cost', external.totalCost, internal.totalCost, 1e-9),
  ];
  checks.push(checkRow(
    'top bottleneck agrees',
    external.bottlenecks[0]?.kind === internal.bottlenecks[0]?.kind &&
      external.bottlenecks[0]?.id === internal.bottlenecks[0]?.id &&
      external.bottlenecks[0]?.reason === internal.bottlenecks[0]?.reason,
    `internal=${bottleneckLabel(internal)} external=${bottleneckLabel(external)}`,
  ));
  return checks;
}

function generateScheduledTrips(network: TrafficNetwork, params: SmartTrafficParams, seed: number, demandEndSec: number): SharedTrafficTrip[] {
  const rng = mulberry32(seed);
  const accumulators = new Map<string, number>();
  const trips: SharedTrafficTrip[] = [];
  const ticks = Math.ceil(demandEndSec / params.dtSec);
  const sinkById = new Map(network.sinks.map(s => [s.id, s]));
  for (const source of network.sources) accumulators.set(source.id, 0);
  for (let tick = 0; tick < ticks; tick++) {
    const departSec = roundTime(tick * params.dtSec);
    for (const source of network.sources) {
      const expected = source.ratePerMin * (params.spawnRateMultiplier ?? 1) * params.dtSec / 60;
      let acc = (accumulators.get(source.id) ?? 0) + expected;
      const count = Math.floor(acc);
      acc -= count;
      accumulators.set(source.id, acc);
      for (let k = 0; k < count; k++) {
        const sinkIds = source.destinationSinkIds ?? network.sinks.map(s => s.id);
        const destinationSinkId = sinkIds[Math.floor(rng() * sinkIds.length)];
        const sink = sinkById.get(destinationSinkId);
        if (!sink) continue;
        const route = shortestLanePath(network, source.nodeId, sink.nodeId);
        if (route.length === 0) continue;
        trips.push({
          id: `trip-${trips.length + 1}`,
          departSec,
          sourceId: source.id,
          destinationSinkId,
          route,
          sourceNodeId: source.nodeId,
          sinkNodeId: sink.nodeId,
        });
      }
    }
  }
  return trips;
}

function shortestLanePath(network: TrafficNetwork, fromNodeId: string, toNodeId: string): string[] {
  const outgoing = new Map<string, TrafficLane[]>();
  for (const lane of network.lanes) {
    const lanes = outgoing.get(lane.from) ?? [];
    lanes.push(lane);
    outgoing.set(lane.from, lanes);
  }
  const queue: Array<{nodeId: string; route: string[]}> = [{nodeId: fromNodeId, route: []}];
  const seen = new Set<string>([fromNodeId]);
  while (queue.length > 0) {
    const item = queue.shift()!;
    if (item.nodeId === toNodeId) return item.route;
    for (const lane of outgoing.get(item.nodeId) ?? []) {
      if (seen.has(lane.to)) continue;
      seen.add(lane.to);
      queue.push({nodeId: lane.to, route: [...item.route, lane.id]});
    }
  }
  return [];
}

function routeConnects(route: string[], sourceNodeId: string, sinkNodeId: string, laneById: Map<string, TrafficLane>): boolean {
  if (route.length === 0) return false;
  let current = sourceNodeId;
  for (const laneId of route) {
    const lane = laneById.get(laneId);
    if (!lane || lane.from !== current) return false;
    current = lane.to;
  }
  return current === sinkNodeId;
}

function stripTrafficRuntimeInputs(params: SmartTrafficParams): Omit<SmartTrafficParams, 'network' | 'scheduledTrips'> {
  const {network: _network, scheduledTrips: _scheduledTrips, ...rest} = params;
  return rest;
}

function exactNumber(name: string, actual: number, expected: number): CheckRow {
  return checkRow(name, actual === expected, `actual=${fmt(actual)} expected=${fmt(expected)}`);
}

function closeNumber(name: string, actual: number, expected: number, tolerance: number): CheckRow {
  const diff = Math.abs(actual - expected);
  return checkRow(name, diff <= tolerance, `actual=${fmt(actual)} expected=${fmt(expected)} diff=${diff.toExponential(3)} tol=${tolerance}`);
}

function relativeNumber(name: string, actual: number, expected: number, tolerance: number): CheckRow {
  const diff = Math.abs(actual - expected);
  const rel = diff / Math.max(1, Math.abs(actual), Math.abs(expected));
  return checkRow(name, rel <= tolerance, `actual=${fmt(actual)} expected=${fmt(expected)} rel=${rel.toFixed(3)} tol=${tolerance}`);
}

function ratioBand(name: string, actual: number, expected: number, minRatio: number, maxRatio: number): CheckRow {
  const ratio = actual / Math.max(1e-9, expected);
  return checkRow(name, ratio >= minRatio && ratio <= maxRatio, `actual=${fmt(actual)} expected=${fmt(expected)} ratio=${ratio.toFixed(3)}`);
}

function finiteNumber(name: string, actual: number): CheckRow {
  return checkRow(name, Number.isFinite(actual), `actual=${fmt(actual)}`);
}

function checkRow(name: string, passed: boolean, detail: string): CheckRow {
  return {name, passed, detail};
}

function bottleneckLabel(result: ComputerNetworkResult): string {
  const b = result.bottlenecks[0];
  return b ? `${b.kind}:${b.id}:${b.reason}` : 'none';
}

function fmt(n: number): string {
  return Number.isFinite(n) ? String(Math.round(n * 1e6) / 1e6) : String(n);
}

function roundTime(value: number): number {
  return Math.round(value * 10) / 10;
}

function renderMarkdown(reports: EngineReport[]): string {
  const lines = [
    '# External FEL Comparison',
    '',
    '| Domain | Scenario | Engine | Status | Checks | Input | Output |',
    '| --- | --- | --- | --- | ---: | --- | --- |',
  ];
  for (const report of reports) {
    const passed = report.checks.filter(c => c.passed).length;
    lines.push([
      report.domain,
      report.scenario,
      report.engine,
      report.status,
      `${passed}/${report.checks.length}`,
      path.relative(ROOT, report.inputPath),
      report.outputPath ? path.relative(ROOT, report.outputPath) : '',
    ].map(escapeMarkdownCell).join(' | ').replace(/^/, '| ').replace(/$/, ' |'));
  }
  const failed = reports.flatMap(report => report.checks
    .filter(check => !check.passed)
    .map(check => `- ${report.domain}/${report.scenario}/${report.engine}: ${check.name} (${check.detail})`));
  if (failed.length > 0) {
    lines.push('', '## Failed Checks', '', ...failed);
  }
  const skipped = reports.filter(report => report.status === 'skipped');
  if (skipped.length > 0) {
    lines.push('', '## Skipped Optional Engines', '', ...skipped.map(report => `- ${report.engine}: ${report.checks[0]?.detail ?? 'skipped'}`));
  }
  lines.push('');
  return lines.join('\n');
}

function escapeMarkdownCell(value: string): string {
  return value.replace(/\|/g, '\\|');
}

main();
