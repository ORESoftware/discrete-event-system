// RUST MIGRATION:
// - Target: src/bin/compare_traffic_engines.rs.
// - Keep this as a CLI runner with a Result-returning main; map TRAFFIC_ENGINE_VENV and paths to clap/std::env plus PathBuf.
// - Convert SharedTrip, EngineStats, and XmlAttrs to nominal structs; keep XML/JSON render helpers private module functions.
// - Replace spawnSync with std::process::Command or tokio::process and put SUMO/UXsim adapters behind explicit external-engine traits.
'use strict';

import * as fs from 'fs';
import * as path from 'path';
import {spawnSync} from 'child_process';
import {mulberry32} from '../general/prng';
import {
  buildFiveIntersectionTrafficNetwork,
  TrafficLane,
  TrafficNetwork,
  TrafficScheduledTrip,
} from '../general/network-flow';
import {runSmartTrafficFlow, SmartTrafficParams, SmartTrafficResult} from '../general/smart-traffic-flow';

interface SharedTrip extends TrafficScheduledTrip {
  id: string;
  route: string[];
  sourceNodeId: string;
  sinkNodeId: string;
}

interface EngineStats {
  engine: string;
  version?: string;
  generated: number;
  entered?: number;
  completed: number;
  activeAtEnd: number;
  dropped?: number;
  maxActive?: number;
  meanTravelTimeSec?: number;
  meanSpeedMps?: number;
  meanAbsJerkMps3?: number;
  minHeadwayM?: number;
  notes?: string[];
}

interface XmlAttrs {
  [key: string]: string;
}

const projectRoot = path.resolve(__dirname, '../../..');
const outDir = path.join(projectRoot, 'out', 'traffic-engine-comparison');
const venvDir = process.env.TRAFFIC_ENGINE_VENV
  ? path.resolve(projectRoot, process.env.TRAFFIC_ENGINE_VENV)
  : path.join(projectRoot, 'out', 'traffic-engine-venv');
const sumoBin = path.join(venvDir, 'bin', 'sumo');
const netconvertBin = path.join(venvDir, 'bin', 'netconvert');
const pythonBin = path.join(venvDir, 'bin', 'python');

function main(): void {
  fs.mkdirSync(outDir, {recursive: true});
  const network = buildFiveIntersectionTrafficNetwork();
  const params: SmartTrafficParams = {
    network,
    durationSec: 200,
    dtSec: 0.1,
    seed: 19,
    maxCars: 180,
    spawnRateMultiplier: 1,
    carLengthM: 4.8,
    carWidthM: 1.8,
    laneWidthM: 3.7,
    minGapM: 2.5,
    maxAccelMps2: 2.2,
    maxDecelMps2: 4,
    timeHeadwaySec: 1.2,
    reactionTimeSec: 1,
    maxJerkMps3: 6,
    gridCellSizeM: 0.3048,
    smartCarPoolSize: 240,
    actorShuffleSeed: 2026,
    accidentRiskScale: 0,
  };
  const trips = generateScheduledTrips(network, params, params.seed + 4242);
  writeSharedInput(network, params, trips);

  const desStats = runDes(params, trips);
  const sumoStats = runSumo(network, params, trips);
  const uxsimStats = runUxsim(network, params, trips);
  const comparison = {
    generatedAt: new Date().toISOString(),
    scenario: {
      network: 'five-intersection',
      durationSec: params.durationSec,
      dtSec: params.dtSec,
      scheduledTrips: trips.length,
      lanes: network.lanes.length,
      intersections: network.nodes.filter(n => n.kind === 'intersection').length,
    },
    engines: [desStats, sumoStats, uxsimStats],
  };

  fs.writeFileSync(path.join(outDir, 'traffic-engine-comparison.json'), JSON.stringify(comparison, null, 2));
  fs.writeFileSync(path.join(outDir, 'traffic-engine-comparison.md'), renderMarkdown(comparison.engines, comparison.scenario));
  console.log(renderMarkdown(comparison.engines, comparison.scenario));
  console.log(`\nWrote ${path.relative(projectRoot, path.join(outDir, 'traffic-engine-comparison.json'))}`);
}

function generateScheduledTrips(network: TrafficNetwork, params: SmartTrafficParams, seed: number): SharedTrip[] {
  const rng = mulberry32(seed);
  const accumulators = new Map<string, number>();
  const trips: SharedTrip[] = [];
  const ticks = Math.ceil(params.durationSec / params.dtSec);
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

function runDes(params: SmartTrafficParams, trips: SharedTrip[]): EngineStats {
  const scheduledTrips = trips.map(({departSec, sourceId, destinationSinkId}) => ({departSec, sourceId, destinationSinkId}));
  const result = runSmartTrafficFlow({...params, scheduledTrips});
  return {
    engine: 'DES smart traffic',
    version: 'local',
    generated: trips.length,
    entered: result.entered,
    completed: result.exited,
    activeAtEnd: result.finalCars.length,
    dropped: result.dropped,
    maxActive: result.maxActiveCars,
    meanTravelTimeSec: roundMetric(result.meanTravelTimeSec),
    meanSpeedMps: roundMetric(result.meanSpeedMps),
    meanAbsJerkMps3: roundMetric(meanAbsJerk(result)),
    minHeadwayM: roundMetric(minLeaderGap(result)),
    notes: ['uses one-foot cell stations and smart movable car runTimeStep decisions'],
  };
}

function runSumo(network: TrafficNetwork, params: SmartTrafficParams, trips: SharedTrip[]): EngineStats {
  if (!fs.existsSync(sumoBin) || !fs.existsSync(netconvertBin)) {
    return {
      engine: 'SUMO',
      generated: trips.length,
      completed: 0,
      activeAtEnd: trips.length,
      notes: [`SUMO binaries not found under ${venvDir}`],
    };
  }
  const dir = path.join(outDir, 'sumo');
  fs.mkdirSync(dir, {recursive: true});
  const nodesFile = path.join(dir, 'five-intersection.nod.xml');
  const edgesFile = path.join(dir, 'five-intersection.edg.xml');
  const routesFile = path.join(dir, 'five-intersection.rou.xml');
  const netFile = path.join(dir, 'five-intersection.net.xml');
  const tripinfoFile = path.join(dir, 'tripinfo.xml');
  const summaryFile = path.join(dir, 'summary.xml');

  fs.writeFileSync(nodesFile, sumoNodesXml(network));
  fs.writeFileSync(edgesFile, sumoEdgesXml(network));
  fs.writeFileSync(routesFile, sumoRoutesXml(params, trips));
  runCommand(netconvertBin, [
    '--node-files', nodesFile,
    '--edge-files', edgesFile,
    '--output-file', netFile,
    '--no-turnarounds',
    '--xml-validation', 'never',
  ], dir);
  const version = commandVersion(sumoBin);
  runCommand(sumoBin, [
    '-n', netFile,
    '-r', routesFile,
    '--begin', '0',
    '--end', String(params.durationSec),
    '--step-length', String(params.dtSec),
    '--tripinfo-output', tripinfoFile,
    '--summary-output', summaryFile,
    '--no-step-log', 'true',
    '--duration-log.disable', 'true',
    '--time-to-teleport', '-1',
    '--xml-validation', 'never',
    '--collision.action', 'warn',
  ], dir);
  const tripInfos = parseXmlRecords(fs.readFileSync(tripinfoFile, 'utf8'), 'tripinfo');
  const summarySteps = parseXmlRecords(fs.readFileSync(summaryFile, 'utf8'), 'step');
  const last = summarySteps[summarySteps.length - 1] ?? {};
  const durations = tripInfos.map(t => Number(t.duration)).filter(Number.isFinite);
  const speeds = tripInfos
    .map(t => Number(t.routeLength) / Math.max(1e-9, Number(t.duration)))
    .filter(Number.isFinite);
  const maxActive = Math.max(0, ...summarySteps.map(s => Number(s.running) || 0));
  const inserted = Number(last.inserted);
  const ended = Number(last.ended);
  return {
    engine: 'SUMO',
    version,
    generated: trips.length,
    entered: Number.isFinite(inserted) ? inserted : undefined,
    completed: tripInfos.length,
    activeAtEnd: Number.isFinite(inserted) && Number.isFinite(ended) ? Math.max(0, inserted - ended) : Math.max(0, trips.length - tripInfos.length),
    maxActive,
    meanTravelTimeSec: meanRounded(durations),
    meanSpeedMps: meanRounded(speeds),
    notes: ['microscopic, space-continuous SUMO run with the shared scheduled trip table'],
  };
}

function runUxsim(network: TrafficNetwork, params: SmartTrafficParams, trips: SharedTrip[]): EngineStats {
  if (!fs.existsSync(pythonBin)) {
    return {
      engine: 'UXsim',
      generated: trips.length,
      completed: 0,
      activeAtEnd: trips.length,
      notes: [`UXsim virtualenv Python not found under ${venvDir}`],
    };
  }
  const dir = path.join(outDir, 'uxsim');
  fs.mkdirSync(dir, {recursive: true});
  const inputFile = path.join(dir, 'input.json');
  const scriptFile = path.join(dir, 'run_uxsim.py');
  const outputFile = path.join(dir, 'stats.json');
  fs.writeFileSync(inputFile, JSON.stringify({network, params, trips}, null, 2));
  fs.writeFileSync(scriptFile, uxsimScript());
  runCommand(pythonBin, [scriptFile, inputFile, outputFile], dir);
  const stats = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
  return {
    engine: 'UXsim',
    version: stats.version,
    generated: stats.generated,
    entered: stats.entered,
    completed: stats.completed,
    activeAtEnd: stats.activeAtEnd,
    maxActive: stats.maxActive,
    meanTravelTimeSec: roundMetric(stats.meanTravelTimeSec),
    meanSpeedMps: roundMetric(stats.meanSpeedMps),
    notes: ['mesoscopic UXsim run with exact shared departure times and OD pairs; jerk is not exposed by UXsim'],
  };
}

function writeSharedInput(network: TrafficNetwork, params: SmartTrafficParams, trips: SharedTrip[]): void {
  fs.writeFileSync(path.join(outDir, 'shared-traffic-input.json'), JSON.stringify({
    network,
    params: {...params, network: undefined},
    trips,
  }, null, 2));
}

function sumoNodesXml(network: TrafficNetwork): string {
  const signalNodes = new Set((network.signals ?? []).map(s => s.nodeId));
  const lines = ['<nodes>'];
  for (const node of network.nodes) {
    const type = signalNodes.has(node.id) ? 'traffic_light' : 'priority';
    lines.push(`  <node id="${xml(node.id)}" x="${node.x * 120}" y="${-node.y * 120}" type="${type}"/>`);
  }
  lines.push('</nodes>');
  return lines.join('\n') + '\n';
}

function sumoEdgesXml(network: TrafficNetwork): string {
  const lines = ['<edges>'];
  for (const lane of network.lanes) {
    lines.push(`  <edge id="${xml(lane.id)}" from="${xml(lane.from)}" to="${xml(lane.to)}" numLanes="1" speed="${lane.speedLimitMps}" length="${lane.lengthM}" priority="1"/>`);
  }
  lines.push('</edges>');
  return lines.join('\n') + '\n';
}

function sumoRoutesXml(params: SmartTrafficParams, trips: SharedTrip[]): string {
  const routes = new Map<string, string>();
  for (const trip of trips) {
    const key = trip.route.join(' ');
    if (!routes.has(key)) routes.set(key, `route-${routes.size + 1}`);
  }
  const lines = [
    '<routes>',
    `  <vType id="car" accel="${params.maxAccelMps2 ?? 2.2}" decel="${params.maxDecelMps2 ?? 4}" apparentDecel="${params.maxDecelMps2 ?? 4}" emergencyDecel="${Math.max(8, (params.maxDecelMps2 ?? 4) * 2)}" length="${params.carLengthM ?? 4.8}" minGap="${params.minGapM ?? 2.5}" maxSpeed="13.5" tau="${params.reactionTimeSec ?? 1}" sigma="0.5"/>`,
  ];
  for (const [edges, id] of routes.entries()) lines.push(`  <route id="${id}" edges="${xml(edges)}"/>`);
  for (const trip of trips) {
    const routeId = routes.get(trip.route.join(' '));
    lines.push(`  <vehicle id="${xml(trip.id)}" type="car" route="${routeId}" depart="${trip.departSec.toFixed(1)}" departLane="best" departSpeed="max"/>`);
  }
  lines.push('</routes>');
  return lines.join('\n') + '\n';
}

function uxsimScript(): string {
  return `
import json
import sys

from uxsim import World
import uxsim

input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path) as f:
    data = json.load(f)

network = data["network"]
params = data["params"]
trips = data["trips"]
vehicle_space = max(1e-9, params.get("carLengthM", 4.8) + params.get("minGapM", 2.5))
W = World(
    name="five-intersection-cross-check",
    deltan=1,
    reaction_time=params.get("dtSec", 0.1),
    tmax=params["durationSec"],
    random_seed=params.get("seed", 19),
    print_mode=0,
    save_mode=0,
    show_mode=0,
    show_progress=0,
    vehicle_logging_timestep_interval=1,
    hard_deterministic_mode=True,
)
for node in network["nodes"]:
    W.addNode(node["id"], node["x"] * 120, -node["y"] * 120)
for lane in network["lanes"]:
    W.addLink(
        lane["id"],
        lane["from"],
        lane["to"],
        length=lane["lengthM"],
        free_flow_speed=lane["speedLimitMps"],
        number_of_lanes=1,
        jam_density=1 / vehicle_space,
    )
for trip in trips:
    W.addVehicle(
        trip["sourceNodeId"],
        trip["sinkNodeId"],
        trip["departSec"],
        name=trip["id"],
        links_prefer=trip["route"],
    )
W.exec_simulation()

vehicles = list(W.VEHICLES.values())
completed = [v for v in vehicles if getattr(v, "state", None) == "end" and getattr(v, "travel_time", None) is not None]
def safe_float(value, default=0):
    try:
        return float(value)
    except Exception:
        return default

durations = [safe_float(v.travel_time) for v in completed]
speeds = [safe_float(getattr(v, "distance_traveled", 0)) / max(1e-9, safe_float(v.travel_time)) for v in completed]
max_active = 0
dt = params.get("dtSec", 0.1)
ticks = int(params["durationSec"] / dt + 0.5)
for tick in range(ticks):
    t = tick * dt
    active = 0
    for v in vehicles:
        depart = safe_float(getattr(v, "departure_time_in_second", getattr(v, "departure_time", 0)))
        arrival = getattr(v, "arrival_time", None)
        if depart <= t and (arrival is None or safe_float(arrival, params["durationSec"] + 1) > t):
            active += 1
    max_active = max(max_active, active)

out = {
    "version": getattr(uxsim, "__version__", "unknown"),
    "generated": len(trips),
    "entered": len(vehicles),
    "completed": len(completed),
    "activeAtEnd": max(0, len(vehicles) - len(completed)),
    "maxActive": max_active,
    "meanTravelTimeSec": sum(durations) / len(durations) if durations else None,
    "meanSpeedMps": sum(speeds) / len(speeds) if speeds else None,
}
with open(output_path, "w") as f:
    json.dump(out, f, indent=2)
`;
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

function runCommand(command: string, args: string[], cwd: string): string {
  const result = spawnSync(command, args, {
    cwd,
    encoding: 'utf8',
    env: {...process.env, PATH: `${path.dirname(command)}:${process.env.PATH ?? ''}`},
  });
  if (result.status !== 0) {
    throw new Error([
      `${path.basename(command)} failed with status ${result.status}`,
      result.stdout,
      result.stderr,
    ].filter(Boolean).join('\n'));
  }
  return `${result.stdout ?? ''}${result.stderr ?? ''}`;
}

function commandVersion(command: string): string {
  const text = runCommand(command, ['--version'], outDir);
  return text.split('\n').find(line => line.trim().length > 0)?.trim() ?? 'unknown';
}

function parseXmlRecords(text: string, tag: string): XmlAttrs[] {
  const records: XmlAttrs[] = [];
  const re = new RegExp(`<${tag}\\b([^>]*)\\/?>`, 'g');
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) records.push(parseXmlAttrs(match[1]));
  return records;
}

function parseXmlAttrs(text: string): XmlAttrs {
  const attrs: XmlAttrs = {};
  const re = /([A-Za-z0-9_.:-]+)="([^"]*)"/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) attrs[match[1]] = match[2];
  return attrs;
}

function renderMarkdown(engines: EngineStats[], scenario: any): string {
  const header = [
    '# Traffic Engine Cross-Check',
    '',
    `Scenario: ${scenario.network}, ${scenario.durationSec}s at dt=${scenario.dtSec}s, ${scenario.scheduledTrips} scheduled trips.`,
    '',
    '| Engine | Version | Generated | Entered | Completed | Active @ end | Dropped | Max active | Mean travel (s) | Mean speed (m/s) | Mean |jerk| | Notes |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
  ];
  const rows = engines.map(e => [
    e.engine,
    e.version ?? '',
    String(e.generated),
    fmtCell(e.entered),
    String(e.completed),
    String(e.activeAtEnd),
    fmtCell(e.dropped),
    fmtCell(e.maxActive),
    fmtCell(e.meanTravelTimeSec),
    fmtCell(e.meanSpeedMps),
    fmtCell(e.meanAbsJerkMps3),
    (e.notes ?? []).join('; '),
  ].map(escapeMarkdownCell).join(' | '));
  return [...header, ...rows.map(r => `| ${r} |`), ''].join('\n');
}

function fmtCell(value: number | undefined): string {
  return value === undefined || value === null || !Number.isFinite(value) ? '' : String(roundMetric(value));
}

function escapeMarkdownCell(value: string): string {
  return value.replace(/\|/g, '\\|');
}

function meanRounded(xs: number[]): number | undefined {
  return xs.length === 0 ? undefined : roundMetric(xs.reduce((a, b) => a + b, 0) / xs.length);
}

function meanAbsJerk(result: SmartTrafficResult): number {
  const jerks = result.trace.flatMap(row => row.cars.map(car => Math.abs(car.jerkMps3)));
  return jerks.length === 0 ? 0 : jerks.reduce((sum, x) => sum + x, 0) / jerks.length;
}

function minLeaderGap(result: SmartTrafficResult): number {
  const gaps = result.trace.flatMap(row => row.cars.map(car => car.leaderGapM).filter((x): x is number => x !== undefined));
  return gaps.length === 0 ? 0 : Math.min(...gaps);
}

function roundMetric(value: number | null | undefined): number | undefined {
  return value === undefined || value === null || !Number.isFinite(value) ? undefined : Math.round(value * 1000) / 1000;
}

function roundTime(value: number): number {
  return Math.round(value * 10) / 10;
}

function xml(value: string): string {
  return value.replace(/[<>&"']/g, c => ({'<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;'}[c]!));
}

main();
