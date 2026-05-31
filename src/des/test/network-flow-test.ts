'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/network_flow_test.rs   (integration test crate)
// 1:1 file move. Spans network-flow / smart-traffic-flow / max-flow /
// traffic-flow / stochastic-flow-mdp, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - float flow/throughput comparisons -> approx::assert_relative_eq!.
//   - stochastic-flow-MDP sampling -> a seeded rand::Rng; fs usage -> `tempfile`.
// =============================================================================

// =============================================================================
// Tests for network-flow, traffic-flow, and stochastic-flow DES models.
// =============================================================================

import * as fs from 'fs';
import {getModel, runFromSpec} from '../general/des-registry';
import {
  buildFiveIntersectionTrafficNetwork,
  runMaxFlow,
  runTrafficFlow,
} from '../general/network-flow';
import {runSmartTrafficFlow} from '../general/smart-traffic-flow';
import {
  buildTextbookMaxFlowProblem,
  MaxFlowProblem,
  solveMaxFlow,
} from '../general/max-flow';
import {
  buildDefaultTrafficProblem,
  buildTrafficMaxFlowProblem,
  runTrafficSimulation,
} from '../general/traffic-flow';
import {
  buildDefaultStochasticFlowMDPProblem,
  solveStochasticFlowMDP,
} from '../general/stochastic-flow-mdp';

let pass = 0, fail = 0;

function check(label: string, ok: boolean, detail = ''): void {
  if (ok) {
    pass++;
    console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`);
  } else {
    fail++;
    console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`);
  }
}

function closeValue(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function checkClose(label: string, a: number, b: number, tol = 1e-8): void {
  check(label, closeValue(a, b, tol), `|${a} - ${b}| = ${Math.abs(a - b).toExponential(2)}`);
}

function minTrafficLeaderGap(r: ReturnType<typeof runTrafficFlow>): number {
  const gaps = r.trace.flatMap(row => row.cars.map(car => car.leaderGapM).filter((x): x is number => x !== undefined));
  return gaps.length > 0 ? Math.min(...gaps) : 0;
}

function minSmartLeaderGap(r: ReturnType<typeof runSmartTrafficFlow>): number {
  const gaps = r.trace.flatMap(row => row.cars.map(car => car.leaderGapM).filter((x): x is number => x !== undefined));
  return gaps.length > 0 ? Math.min(...gaps) : 0;
}

async function checkThrows(label: string, fn: () => unknown | Promise<unknown>, contains?: string): Promise<void> {
  let threw = false;
  let message = '';
  try {
    await fn();
  } catch (e) {
    threw = true;
    message = e instanceof Error ? e.message : String(e);
  }
  check(label, threw && (contains === undefined || message.includes(contains)), threw ? `message=${JSON.stringify(message)}` : 'did not throw');
}

const teachingNetwork = {
  numNodes: 6,
  source: 0,
  sink: 5,
  nodeNames: ['s', 'a', 'b', 'c', 'd', 't'],
  nodeCoordinates: [[90, 260], [260, 160], [260, 360], [520, 160], [520, 360], [760, 260]] as Array<[number, number]>,
  edges: [
    {from: 0, to: 1, capacity: 16, name: 's-a'},
    {from: 0, to: 2, capacity: 13, name: 's-b'},
    {from: 1, to: 2, capacity: 10, name: 'a-b'},
    {from: 2, to: 1, capacity: 4, name: 'b-a'},
    {from: 1, to: 3, capacity: 12, name: 'a-c'},
    {from: 3, to: 2, capacity: 9, name: 'c-b'},
    {from: 2, to: 4, capacity: 14, name: 'b-d'},
    {from: 4, to: 3, capacity: 7, name: 'd-c'},
    {from: 3, to: 5, capacity: 20, name: 'c-t'},
    {from: 4, to: 5, capacity: 4, name: 'd-t'},
  ],
};

async function main(): Promise<void> {
  console.log('\n[1] Animated/logged max-flow DES optimization');
  {
    const r = runMaxFlow(teachingNetwork);
    checkClose('1.1 max flow equals textbook value 23', r.maxFlow, 23);
    checkClose('1.2 min-cut capacity equals max flow', r.minCut.capacity, r.maxFlow);
    check('1.3 validators pass', r.validation.every(c => c.passed));
    check('1.4 augmentation trace is non-empty', r.trace.length > 0, `len=${r.trace.length}`);
    check('1.5 augmenting path values are non-decreasing', r.trace.every((row, i, xs) => i === 0 || row.value >= xs[i - 1].value - 1e-12));
    check('1.6 edge flows satisfy capacity bounds', r.edgeFlows.every(e => e.flow >= -1e-9 && e.flow <= e.capacity + 1e-9));

    await checkThrows('1.7 rejects negative capacities', () => runMaxFlow({
      numNodes: 2,
      source: 0,
      sink: 1,
      edges: [{from: 0, to: 1, capacity: -1}],
    }), 'capacity');
  }

  console.log('\n[2] Animated/logged continuous-time traffic flow');
  {
    const network = buildFiveIntersectionTrafficNetwork();
    check('2.1 builtin network has five intersections', network.nodes.filter(n => n.kind === 'intersection').length === 5);
    check('2.2 builtin network has sources and sinks', network.sources.length === 3 && network.sinks.length === 3);

    const r = runTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 180,
      dtSec: 1,
      seed: 19,
      maxCars: 250,
      spawnRateMultiplier: 1,
      carLengthM: 4.8,
      minGapM: 2.5,
      maxAccelMps2: 2.2,
      maxDecelMps2: 4,
    });
    check('2.3 traffic validators pass', r.validation.every(c => c.passed));
    check('2.4 simulation produced moving cars', r.entered > 0 && r.maxActiveCars > 0, `entered=${r.entered} maxActive=${r.maxActiveCars}`);
    check('2.5 some cars exit through sinks', r.exited > 0, `exited=${r.exited}`);
    check('2.6 active cars stay below maxCars and 300', r.maxActiveCars <= 250 && r.maxActiveCars < 300, `maxActive=${r.maxActiveCars}`);
    check('2.7 trace length equals duration/dt', r.trace.length === 180, `len=${r.trace.length}`);
    check('2.8 conservation holds in result', r.entered === r.exited + r.finalCars.length);
    check('2.9 signal phases are observable', Object.keys(r.trace[0].signalPhases).length === 5);
    check('2.10 traffic cells are stationized at roughly one-foot resolution', r.cellStats.createdCellStations > r.network.lanes.length * 100 && Math.abs(r.cellStats.cellSizeM - 0.3048) < 1e-12, `cells=${r.cellStats.createdCellStations} cell=${r.cellStats.cellSizeM}`);
    check('2.11 car kinematics include acceleration, jerk, and current grid tiles',
      r.trace.some(row => row.cars.some(car =>
        Number.isFinite(car.accelerationMps2) &&
        Number.isFinite(car.jerkMps3) &&
        car.gridCellIds.length > 0,
      )),
      'expected at least one active car with kinematic/grid state');
    check('2.12 observed car-to-car clearance stays nonnegative', minTrafficLeaderGap(r) >= -1e-7, `clearance=${minTrafficLeaderGap(r)}`);

    const smartCoarse = runSmartTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 90,
      dtSec: 1,
      seed: 19,
      maxCars: 120,
      spawnRateMultiplier: 1,
      carLengthM: 4.8,
      minGapM: 2.5,
      maxAccelMps2: 2.2,
      maxDecelMps2: 4,
    });
    check('2.13 smart traffic validators pass', smartCoarse.validation.every(c => c.passed));
    check('2.14 smart traffic enters and exits cars', smartCoarse.entered > 0 && smartCoarse.exited > 0, `entered=${smartCoarse.entered} exited=${smartCoarse.exited}`);
    check('2.15 smart movables own runTimeStep decisions',
      smartCoarse.trace.some(row => row.cars.some(car => car.actorId.startsWith('smart-car-') && car.runCount > 0)),
      'expected at least one smart movable car with decision ticks');
    check('2.16 smart traffic preserves grid ids and nonnegative observed clearance',
      minSmartLeaderGap(smartCoarse) >= -1e-7 && smartCoarse.trace.some(row => row.cars.some(car => car.gridCellIds.length > 0)),
      `clearance=${minSmartLeaderGap(smartCoarse)}`);

    const capped = runTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 60,
      dtSec: 1,
      seed: 5,
      maxCars: 20,
      spawnRateMultiplier: 10,
    });
    check('2.17 maxCars cap is enforced under high demand', capped.maxActiveCars <= 20, `maxActive=${capped.maxActiveCars}`);
    check('2.18 high demand drops cars when grid is saturated', capped.dropped > 0, `dropped=${capped.dropped}`);
    check('2.19 capped traffic validators pass', capped.validation.every(c => c.passed));

    await checkThrows('2.20 rejects over-300 car systems', () => runTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 10,
      dtSec: 1,
      seed: 1,
      maxCars: 301,
    }), 'maxCars');

    const micro = runTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 10,
      dtSec: 0.1,
      seed: 11,
      maxCars: 40,
      spawnRateMultiplier: 12,
      gridCellSizeM: 0.3048,
      reactionTimeSec: 0.8,
      maxJerkMps3: 6,
    });
    const movingSnap = micro.trace.find(row => row.cars.length > 0)?.cars[0];
    check('2.14 small-dt traffic produces expected trace length', micro.trace.length === 100, `len=${micro.trace.length}`);
    check('2.15 car snapshots expose acceleration, jerk, and target acceleration',
      !!movingSnap
        && Number.isFinite(movingSnap.accelerationMps2)
        && Number.isFinite(movingSnap.jerkMps3)
        && Number.isFinite(movingSnap.targetAccelerationMps2),
      movingSnap ? JSON.stringify({a: movingSnap.accelerationMps2, j: movingSnap.jerkMps3, target: movingSnap.targetAccelerationMps2}) : 'no car');
    check('2.16 cars carry occupied one-foot grid cells',
      !!movingSnap && movingSnap.gridCellCount > 0 && movingSnap.gridCellIds.length === movingSnap.gridCellCount,
      movingSnap ? `cells=${movingSnap.gridCellCount}` : 'no car');
    check('2.17 traffic grid reports stationary cell stations',
      micro.cellStats.cellSizeM <= 0.3048 + 1e-12 && micro.cellStats.createdCellStations > 0,
      JSON.stringify(micro.cellStats));
    check('2.18 trace reports active grid-cell occupancy',
      micro.trace.some(row => row.activeGridCells > 0),
      `max=${Math.max(...micro.trace.map(row => row.activeGridCells))}`);
    check('2.19 micro traffic validators pass', micro.validation.every(c => c.passed));

    const smart = runSmartTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 30,
      dtSec: 0.1,
      seed: 13,
      actorShuffleSeed: 99,
      maxCars: 60,
      smartCarPoolSize: 200,
      spawnRateMultiplier: 4,
      gridCellSizeM: 0.3048,
      reactionTimeSec: 0.8,
      maxJerkMps3: 6,
      distancePreferenceSpread: 0.4,
      startPreferenceSpread: 0.6,
    });
    const preferenceSamples = smart.trace.flatMap(row => row.cars.map(car => car.distancePreference));
    const startPreferenceSamples = smart.trace.flatMap(row => row.cars.map(car => car.startPreference));
    check('2.20 smart traffic validators pass', smart.validation.every(c => c.passed));
    check('2.21 smart cars are runner participants, not stations', smart.execution.participantCount === 201 && smart.execution.smartMovableCount === 200, JSON.stringify(smart.execution));
    check('2.22 active smart cars receive exactly one runTimeStep per tick',
      smart.trace.every(row => row.scheduledSmartCars === row.smartMovableRuns),
      `maxRuns=${smart.execution.maxSmartMovableRunsPerTick}`);
    check('2.23 smart traffic produced shuffled actor order samples',
      smart.trace.some(row => row.actorRunOrder.length > 1),
      smart.trace.find(row => row.actorRunOrder.length > 1)?.actorRunOrder.join(' ') ?? 'no order');
    check('2.24 smart traffic keeps active cars under 300', smart.maxActiveCars < 300 && smart.maxActiveCars <= 60, `maxActive=${smart.maxActiveCars}`);
    check('2.25 smart traffic uses tenth-second ticks in trace output', smart.trace.length === 300 && closeValue(smart.trace[0].timeSec, 0.1), `len=${smart.trace.length} first=${smart.trace[0].timeSec}`);
    check('2.26 smart cars sample heterogeneous following-distance preferences',
      preferenceSamples.some(x => x < 0.9) && preferenceSamples.some(x => x > 1.1),
      `min=${Math.min(...preferenceSamples).toFixed(3)} max=${Math.max(...preferenceSamples).toFixed(3)}`);
    check('2.27 smart cars sample heterogeneous startup hesitation preferences',
      startPreferenceSamples.some(x => x < 0.85) && startPreferenceSamples.some(x => x > 1.15),
      `min=${Math.min(...startPreferenceSamples).toFixed(3)} max=${Math.max(...startPreferenceSamples).toFixed(3)}`);

    const accidentEvents: any[] = [];
    const accident = runSmartTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 20,
      dtSec: 0.1,
      seed: 19,
      actorShuffleSeed: 2026,
      maxCars: 250,
      smartCarPoolSize: 400,
      spawnRateMultiplier: 3,
      gridCellSizeM: 0.3048,
      accidentRiskScale: 16,
      accidentAccelBoostMps2: 20,
      accidentFaultDurationSec: 2,
    }, {log: e => { if (e.kind === 'smart-traffic-accident') accidentEvents.push(e); }});
    const firstAccident = accident.accidents[0];
    check('2.28 smart traffic can generate deterministic rear-end contact accidents', accident.crashed > 0 && accident.accidents.length === accident.crashed, `crashed=${accident.crashed}`);
    check('2.29 accident run validators pass', accident.validation.every(c => c.passed));
    check('2.30 accident events name the striking car, struck car, and grid cell',
      !!firstAccident && firstAccident.actorId.startsWith('smart-car-') && firstAccident.otherActorId.startsWith('smart-car-') && firstAccident.cellId.includes('#'),
      firstAccident ? JSON.stringify(firstAccident) : 'no accident');
    check('2.31 accidents are behavior-risk-triggered but emitted only at body contact',
      !!firstAccident && firstAccident.reason === 'body-contact-rear-end' && firstAccident.riskScore > 0 && firstAccident.hazardPerSec > 0,
      firstAccident ? JSON.stringify({reason: firstAccident.reason, riskScore: firstAccident.riskScore, hazardPerSec: firstAccident.hazardPerSec}) : 'no accident');
    check('2.32 road grid cell stations record accident hits',
      accident.cellStats.accidentCellStations > 0 && accident.cellStats.accidentCellHits === accident.accidents.length,
      JSON.stringify(accident.cellStats));
    check('2.33 trace exposes per-tick accident flashes', accident.trace.some(row => row.accidents.length > 0 && row.crashed > 0));
    check('2.34 accident conservation counts crashed cars as terminal', accident.entered === accident.exited + accident.crashed + accident.finalCars.length);
    check('2.35 accident logger emits sanctioned review events', accidentEvents.length === accident.accidents.length, `events=${accidentEvents.length}`);
  }

  console.log('\n[3] JSON registry and output paths');
  {
    check('3.1 registry has max-flow', getModel('max-flow').id === 'max-flow');
    check('3.2 registry has traffic-flow', getModel('traffic-flow').id === 'traffic-flow');
    check('3.3 registry has stochastic-flow-mdp', getModel('stochastic-flow-mdp').id === 'stochastic-flow-mdp');
    check('3.4 registry has smart-traffic-flow', getModel('smart-traffic-flow').id === 'smart-traffic-flow');

    const logPath = 'out/network-flow-test-max-flow.jsonl';
    const maxSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'max-flow',
      parameters: teachingNetwork,
      runtime: {verbose: false, animate: true, outputs: {html: 'out/network-flow-test-max-flow.html', log: logPath}},
    }, {verbose: false});
    checkClose('3.5 JSON direct max-flow returns expected value', (maxSummary.result as any).maxFlow, 23);
    check('3.6 default max-flow frames output is reported', maxSummary.outputs.some(o => o.kind === 'frames' && o.path === 'out/network-flow-test-max-flow.frames.jsonl'));
    check('3.7 max-flow observability log is written', fs.readFileSync(logPath, 'utf8').includes('"kind":"max-flow-finish"'));

    const maxBuiltin = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'max-flow',
      parameters: {builtin: 'textbook'},
      runtime: {verbose: false, animate: false},
    }, {verbose: false});
    checkClose('3.8 JSON builtin max-flow returns expected value', (maxBuiltin.result as any).maxFlow, 23);

    const trafficSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'traffic-flow',
      parameters: {builtin: 'five-intersection', durationSec: 30, dtSec: 1, seed: 7, maxCars: 80, spawnRateMultiplier: 1},
      runtime: {verbose: false, animate: false},
    }, {verbose: false});
    const trafficResult = trafficSummary.result as any;
    check('3.9 JSON traffic-flow runs without animation when disabled', trafficSummary.modelId === 'traffic-flow' && trafficSummary.outputs.every(o => o.kind !== 'html'));
    check('3.10 JSON traffic-flow returns trace rows', trafficResult.trace.length === 30, `len=${trafficResult.trace.length}`);

    const smartSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'smart-traffic-flow',
      parameters: {builtin: 'five-intersection', durationSec: 20, dtSec: 0.1, seed: 7, actorShuffleSeed: 55, maxCars: 40, smartCarPoolSize: 120, spawnRateMultiplier: 3},
      runtime: {verbose: false, animate: false},
    }, {verbose: false});
    const smartResult = smartSummary.result as any;
    check('3.11 JSON smart-traffic-flow runs without animation when disabled', smartSummary.modelId === 'smart-traffic-flow' && smartSummary.outputs.every(o => o.kind !== 'html'));
    check('3.12 JSON smart-traffic-flow reports smart movable execution', smartResult.execution.smartMovableCount === 120 && smartResult.trace.length === 200, JSON.stringify(smartResult.execution));

    const stochasticSummary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'stochastic-flow-mdp',
      parameters: {builtin: 'small-stochastic-network', seed: 7, maxPolicyRows: 8},
      runtime: {verbose: false, seed: 7},
    }, {verbose: false});
    check('3.13 JSON stochastic-flow-mdp returns finite value', Number.isFinite((stochasticSummary.result as any).expectedReward));
  }

  console.log('\n[4] Modular max-flow/min-cut implementation');
  {
    const r = solveMaxFlow(buildTextbookMaxFlowProblem());
    checkClose('4.1 max flow = 23', r.maxFlow, 23);
    checkClose('4.2 min-cut capacity = 23', r.minCut.capacity, 23);
    check('4.3 augmenting trace is non-empty', r.trace.length > 0);
    check('4.4 source side contains source', r.minCut.sourceSide.includes(r.source));
    check('4.5 sink side contains sink', r.minCut.sinkSide.includes(r.sink));
    check('4.6 all edge flows respect capacity', r.edgeFlows.every(e => e.flow >= -1e-9 && e.flow <= e.capacity + 1e-9));

    const p: MaxFlowProblem = {
      numNodes: 4,
      source: 0,
      sink: 3,
      edges: [
        {from: 0, to: 1, capacity: 4, name: 's-a'},
        {from: 0, to: 2, capacity: 3, name: 's-b'},
        {from: 1, to: 2, capacity: 1, name: 'a-b'},
        {from: 1, to: 3, capacity: 2, name: 'a-t'},
        {from: 2, to: 3, capacity: 4, name: 'b-t'},
      ],
    };
    const bottleneck = solveMaxFlow(p);
    checkClose('4.7 small network max flow = sink bottleneck 6', bottleneck.maxFlow, 6);
    checkClose('4.8 small network cut equals flow', bottleneck.minCut.capacity, bottleneck.maxFlow);
  }

  console.log('\n[5] Modular traffic simulation and max-flow bound');
  {
    const p = buildDefaultTrafficProblem();
    const r = runTrafficSimulation(p);
    check('5.1 traffic completes at least one car', r.completedCars > 0, `completed=${r.completedCars}`);
    check('5.2 generated bounded by configured source caps', r.generatedCars <= 240, `generated=${r.generatedCars}`);
    check('5.3 conservation holds at stop', r.generatedCars === r.completedCars + r.activeCars, `generated=${r.generatedCars}, completed=${r.completedCars}, active=${r.activeCars}`);
    check('5.4 max active stays below 300', r.maxActiveCars < 300, `maxActive=${r.maxActiveCars}`);
    check('5.5 max active stays under configured cap', r.maxActiveCars <= p.maxCars, `maxActive=${r.maxActiveCars}, cap=${p.maxCars}`);
    check('5.6 no traffic invariant violations', r.invariantViolations.length === 0, r.invariantViolations.slice(0, 3).join('; '));
    check('5.7 mean travel time finite', Number.isFinite(r.meanTravelTimeSec), `mean=${r.meanTravelTimeSec}`);
    check('5.8 p95 travel time finite', Number.isFinite(r.p95TravelTimeSec), `p95=${r.p95TravelTimeSec}`);
    check('5.9 time series recorded', r.timeSeries.length > 0, `samples=${r.timeSeries.length}`);

    const mf = solveMaxFlow(buildTrafficMaxFlowProblem(p));
    checkClose('5.10 reported max-flow bound matches solver', r.maxFlowUpperBoundPerMin, mf.maxFlow);
    check('5.11 simulated throughput does not exceed demand bound', r.throughputVsMaxFlow <= 1.05, `ratio=${r.throughputVsMaxFlow}`);
    check('5.12 max-flow bound is positive', mf.maxFlow > 0, `maxFlow=${mf.maxFlow}`);
  }

  console.log('\n[6] Stochastic-flow MDP interpretation');
  {
    const p = buildDefaultStochasticFlowMDPProblem();
    const r = solveStochasticFlowMDP(p, {seed: 7});
    check('6.1 MDP value is finite', Number.isFinite(r.expectedReward), `V0=${r.expectedReward}`);
    check('6.2 MDP state space is enumerated', r.numStates > 0, `states=${r.numStates}`);
    check('6.3 initial action is a routing action', r.initialPolicy[0].action.kind === 'edge', `action=${r.initialPolicy[0].action.label}`);
    check('6.4 expected reward below deterministic max-flow bound', r.expectedReward <= r.deterministicMaxFlow + 1e-9, `V0=${r.expectedReward}, maxFlow=${r.deterministicMaxFlow}`);
    check('6.5 simulated delivered units respect deterministic bound', r.simulation.delivered <= r.deterministicMaxFlow, `delivered=${r.simulation.delivered}, maxFlow=${r.deterministicMaxFlow}`);
    check('6.6 stage history includes terminal plus each Bellman stage', r.stageHistory.length === p.horizon + 1, `history=${r.stageHistory.length}, horizon=${p.horizon}`);

    p.edges = p.edges.map(e => ({...e, successProb: 1, cost: 0}));
    p.waitPenalty = 0;
    p.failurePenalty = 0;
    const deterministic = solveStochasticFlowMDP(p, {seed: 1});
    checkClose('6.7 deterministic MDP value = static max flow', deterministic.expectedReward, deterministic.deterministicMaxFlow);
    checkClose('6.8 deterministic simulation delivers static max flow', deterministic.simulation.delivered, deterministic.deterministicMaxFlow);
    check('6.9 deterministic policy uses all route attempts', deterministic.initialPolicy.filter(x => x.action.kind === 'edge').length === p.horizon);
  }

  console.log('\n[7] Hard precondition failures');
  {
    await checkThrows('7.1 modular traffic rejects maxCars >= 300', () => runTrafficSimulation({...buildDefaultTrafficProblem(), maxCars: 300}));
    await checkThrows('7.2 modular max-flow rejects source == sink', () => solveMaxFlow({numNodes: 2, source: 0, sink: 0, edges: [{from: 0, to: 1, capacity: 1}]}));
    await checkThrows('7.3 stochastic-flow MDP rejects invalid transition probability', () => {
      const p = buildDefaultStochasticFlowMDPProblem();
      p.edges[0] = {...p.edges[0], successProb: 1.5};
      return solveStochasticFlowMDP(p);
    });
    await checkThrows('7.4 smart traffic rejects scheduled trips with unknown source', () => runSmartTrafficFlow({
      builtin: 'five-intersection',
      durationSec: 10,
      dtSec: 0.1,
      seed: 1,
      maxCars: 20,
      scheduledTrips: [{departSec: 0, sourceId: 'missing', destinationSinkId: 'east'}],
    }), 'source id');
    await checkThrows('7.5 smart traffic rejects sources not anchored on source nodes', () => {
      const network = buildFiveIntersectionTrafficNetwork();
      network.sources[0] = {...network.sources[0], nodeId: 'I0'};
      return runSmartTrafficFlow({
        network,
        durationSec: 10,
        dtSec: 0.1,
        seed: 1,
        maxCars: 20,
      });
    }, 'source node');
  }

  console.log(`\nnetwork-flow-test summary: ${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
