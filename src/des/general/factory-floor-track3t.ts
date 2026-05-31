'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/factory-floor-track3t.rs  (module des::general::factory_floor_track3t)
// 1:1 file move. Warehouse/forklift QMDP+POMDP comparison model as a DES station graph.
//
// Declarations → Rust:
//   type WarehouseStationKind = 'source'|'storage'|'aisle'|'sink' -> enum
//   interface Warehouse*/StationDefinition -> structs; const TRACK3T_ARCHIVE_GROUNDING/*_SCENARIO -> `static`/`const`
//   class WarehousePallet (impl Token) -> struct `impl Token`
//   class WarehouseStation/Source/Sink (extend DESStation) / WarehouseForklift (extend SmartMovable)
//                                    -> structs `impl` station/movable traits (bases -> traits)
//   class WarehouseQMDPSolver / WarehousePlanner -> structs + impl
//   fn buildWarehousePOMDP/simulateWarehouseScenario/runWarehouseComparison/... -> fns
//
// Conversion notes (file-specific):
//   - `mulberry32(seed)` RNG (noisy observations, sampling) -> inject `RandomSource`.
//   - `Map`/`Set` over station/pallet ids -> `HashMap`/`HashSet` (String/usize keys: Hash+Eq).
//   - `WarehouseStationKind` union -> enum; builds on DiscreteBelief + POMDP (see those headers).
//   - Station/movable inheritance (Source/Sink <- WarehouseStation <- DESStation) -> trait + composition.
// =============================================================================

// =============================================================================
// factory-floor-track3t.ts
//
// A warehouse/factory-floor comparison model grounded in the archived Track3t
// product claims: continuous indoor material tracking, high ID/location
// accuracy, cloud analytics, and fewer production/shipping errors.
//
// Framework mapping:
//   - WarehouseSource emits movable WarehousePallet jobs.
//   - WarehouseStation / WarehouseSink are stationary floor entities.
//   - WarehousePallet is the movable load.
//   - WarehouseForklift is a SmartMovable that uses a POMDP/QMDP planner.
//
// Decision model:
//   - Hidden state: forklift position, pallet true location, carrying flag.
//   - Action: drive to a stationary entity.
//   - MDP: value iteration inside QMDP solves the fully-observable version.
//   - POMDP: noisy observations update a belief over hidden pallet locations.
// =============================================================================

import {DiscreteBelief} from './belief';
import {beliefUpdate, POMDPSpec} from './pomdp';
import {mulberry32} from './prng';
import {DESStation, Token} from './des-base/station';
import {SmartMovable} from './des-base/smart-movable';
import {Preconditions} from './des-base/preconditions';

export const TRACK3T_ARCHIVE_GROUNDING = [
  {
    label: 'Track3t archived home, 2018-03-31',
    url: 'https://web.archive.org/web/20180331125107/https://www.track3t.com/',
    modelUse: 'Motivates material-flow visibility, lower transit time, fewer bottlenecks, and fewer production/shipping errors.',
  },
  {
    label: 'Track3t archived solution page, 2018-08-15',
    url: 'https://web.archive.org/web/20180815170011/https://www.track3t.com/the-solution',
    modelUse: 'Motivates high location accuracy, high ID accuracy, continuous sensing, dashboards, forensics, and predictive analytics.',
  },
  {
    label: 'Track3t archived about page, 2018-08-15',
    url: 'https://web.archive.org/web/20180815174909/https://www.track3t.com/about-us/',
    modelUse: 'Motivates RFID/wireless/cloud architecture and the move beyond dock-gate-only RFID observations.',
  },
] as const;

export type WarehouseStationKind = 'source' | 'storage' | 'aisle' | 'sink';

export interface StationDefinition {
  id: string;
  label: string;
  kind: WarehouseStationKind;
  x: number;
  y: number;
  canHoldPallet?: boolean;
}

export interface WarehouseLayout {
  stations: StationDefinition[];
  sourceStationId: string;
  sinkStationIds: string[];
  gridMeters?: number;
  routeEdges?: Array<readonly [string, string]>;
}

export interface WarehouseScenarioConfig {
  id: 'baseline' | 'track3t' | string;
  label: string;
  locationAccuracy: number;
  idAccuracy: number;
  initialMisplacementProbability: number;
  placementErrorProbability: number;
  forkliftSpeedMetersPerMinute: number;
  routeInflation: number;
  handlingMinutes: number;
  confirmationDelayMinutes: number;
  searchPenaltyMinutes: number;
  reworkPenaltyMinutes: number;
  deliveryReward: number;
  wrongDeliveryPenalty: number;
  discount: number;
  qmdpTol: number;
  qmdpMaxIter: number;
  dueMinutes: number;
  sensorRefreshSeconds: number;
}

export interface WarehouseSimulationOptions {
  jobs?: number;
  seed?: number;
  maxStepsPerJob?: number;
  layout?: WarehouseLayout;
  recordTrace?: boolean;
  destinationPlan?: string[];
}

export interface WarehouseAction {
  kind: 'go-to';
  target: number;
  label: string;
}

export interface WarehouseObservation {
  kind: 'location' | 'carrying' | 'complete';
  station?: number;
  label: string;
}

export interface WarehouseDecisionState {
  forklift: number;
  pallet: number;
  carrying: boolean;
  terminal: boolean;
}

export interface WarehousePOMDPModel {
  layout: WarehouseLayout;
  scenario: WarehouseScenarioConfig;
  destinationIndex: number;
  states: number[];
  actions: WarehouseAction[];
  observations: WarehouseObservation[];
  terminalState: number;
  spec: POMDPSpec<number, WarehouseAction, WarehouseObservation>;
  encodeState: (forklift: number, pallet: number, carrying: boolean) => number;
  decodeState: (stateId: number) => WarehouseDecisionState;
  nextState: (stateId: number, actionIdx: number) => number;
  observationIndexForLocation: (stationIdx: number) => number;
}

export interface WarehouseStepTrace {
  scenarioId: string;
  jobId: string;
  jobIndex: number;
  step: number;
  timeStart: number;
  timeEnd: number;
  action: string;
  actionTarget: string;
  observation: string;
  event: 'search-miss' | 'pickup' | 'move-loaded' | 'delivered' | 'delivery-error' | 'failed';
  destination: string;
  forkliftBefore: string;
  forkliftAfter: string;
  palletBefore: string;
  palletAfter: string;
  carryingBefore: boolean;
  carryingAfter: boolean;
  beliefEntropy: number;
  beliefByStation: number[];
  cumulativeDelivered: number;
  cumulativeErrors: number;
  cumulativeSearchMisses: number;
  cycleTimeSoFar: number;
}

export interface WarehouseJobSummary {
  jobId: string;
  destination: string;
  completed: boolean;
  shippingError: boolean;
  cycleTime: number;
  steps: number;
  searchMisses: number;
  onTime: boolean;
}

export interface WarehouseMetrics {
  jobsCreated: number;
  completedJobs: number;
  failedJobs: number;
  shippingErrors: number;
  shippingErrorRate: number;
  totalTime: number;
  meanCycleTime: number;
  throughputPerHour: number;
  onTimeRate: number;
  meanStepsPerJob: number;
  meanSearchMissesPerJob: number;
  meanBeliefEntropy: number;
}

export interface WarehouseScenarioResult {
  scenario: WarehouseScenarioConfig;
  layout: WarehouseLayout;
  metrics: WarehouseMetrics;
  jobs: WarehouseJobSummary[];
  trace: WarehouseStepTrace[];
}

export interface WarehouseComparisonResult {
  layout: WarehouseLayout;
  baseline: WarehouseScenarioResult;
  track3t: WarehouseScenarioResult;
  deltas: {
    meanCycleTimeReductionPct: number;
    throughputLiftPct: number;
    searchMissReductionPct: number;
    errorReductionPct: number;
    entropyReductionPct: number;
  };
  sourceNotes: typeof TRACK3T_ARCHIVE_GROUNDING;
}

export class WarehousePallet implements Token {
  constructor(
    readonly id: string,
    readonly destinationId: string,
    public locationId: string,
    readonly createdAt: number,
  ) {}
}

export class WarehouseStation extends DESStation {
  readonly queue: WarehousePallet[] = [];

  constructor(readonly def: StationDefinition) {
    super(def.id);
  }

  receive(pallet: WarehousePallet): void {
    pallet.locationId = this.def.id;
    this.queue.push(pallet);
  }

  remove(pallet: WarehousePallet): boolean {
    const idx = this.queue.findIndex(p => p.id === pallet.id);
    if (idx < 0) return false;
    this.queue.splice(idx, 1);
    return true;
  }

  runTimeStep(): void {}
}

export class WarehouseSource extends WarehouseStation {
  emitPallet(id: string, destinationId: string, createdAt: number): WarehousePallet {
    const pallet = new WarehousePallet(id, destinationId, this.def.id, createdAt);
    this.receive(pallet);
    return pallet;
  }
}

export class WarehouseSink extends WarehouseStation {
  readonly collected: Array<{pallet: WarehousePallet; time: number; correct: boolean}> = [];

  collect(pallet: WarehousePallet, time: number, correct: boolean): void {
    pallet.locationId = this.def.id;
    this.collected.push({pallet, time, correct});
  }
}

export class WarehouseForklift extends SmartMovable {
  carrying?: WarehousePallet;

  constructor(id: string, public stationId: string) {
    super(id);
  }

  moveTo(stationId: string): void {
    this.stationId = stationId;
    if (this.carrying) this.carrying.locationId = stationId;
  }

  pickup(pallet: WarehousePallet): void {
    this.carrying = pallet;
    pallet.locationId = this.stationId;
  }

  drop(): WarehousePallet | undefined {
    const pallet = this.carrying;
    this.carrying = undefined;
    return pallet;
  }

  runTimeStep(): void {}
}

export function defaultWarehouseLayout(): WarehouseLayout {
  const stations: StationDefinition[] = [
    {id: 'receiving', label: 'Recv', kind: 'source', x: 0, y: 3, canHoldPallet: true},
    {id: 'staging', label: 'Stage', kind: 'storage', x: 2, y: 3, canHoldPallet: true},
  ];
  const routeEdges: Array<readonly [string, string]> = [['receiving', 'staging']];
  const rowNames = ['a', 'b', 'c', 'd'];
  for (let r = 0; r < rowNames.length; r++) {
    const row = rowNames[r];
    const y = r * 2;
    let prevId = 'staging';
    for (let c = 1; c <= 3; c++) {
      const id = `reserve-${row}${c}`;
      stations.push({id, label: `${row.toUpperCase()}${c}`, kind: 'storage', x: 4 + (c - 1) * 2, y, canHoldPallet: true});
      routeEdges.push([prevId, id]);
      prevId = id;
    }
  }
  stations.push(
    {id: 'aisle-main', label: 'Aisle', kind: 'aisle', x: 10, y: 3, canHoldPallet: true},
    {id: 'line-a', label: 'Line A', kind: 'sink', x: 12, y: 0},
    {id: 'line-b', label: 'Line B', kind: 'sink', x: 12, y: 2},
    {id: 'line-c', label: 'Line C', kind: 'sink', x: 12, y: 4},
    {id: 'shipping', label: 'Ship', kind: 'sink', x: 12, y: 6},
  );
  for (const row of rowNames) routeEdges.push([`reserve-${row}3`, 'aisle-main']);
  routeEdges.push(
    ['aisle-main', 'line-a'],
    ['aisle-main', 'line-b'],
    ['aisle-main', 'line-c'],
    ['aisle-main', 'shipping'],
  );
  return {
    sourceStationId: 'receiving',
    sinkStationIds: ['line-a', 'line-b', 'line-c', 'shipping'],
    gridMeters: 12,
    stations,
    routeEdges,
  };
}

export const BASELINE_WAREHOUSE_SCENARIO: WarehouseScenarioConfig = {
  id: 'baseline',
  label: 'Conventional WMS / manual lookup',
  locationAccuracy: 0.64,
  idAccuracy: 0.94,
  initialMisplacementProbability: 0.24,
  placementErrorProbability: 0.12,
  forkliftSpeedMetersPerMinute: 72,
  routeInflation: 1.26,
  handlingMinutes: 1.8,
  confirmationDelayMinutes: 2.2,
  searchPenaltyMinutes: 6.5,
  reworkPenaltyMinutes: 18,
  deliveryReward: 110,
  wrongDeliveryPenalty: 65,
  discount: 0.96,
  qmdpTol: 1e-6,
  qmdpMaxIter: 1200,
  dueMinutes: 22,
  sensorRefreshSeconds: 900,
};

export const TRACK3T_WAREHOUSE_SCENARIO: WarehouseScenarioConfig = {
  id: 'track3t',
  label: 'Track3t-enabled floor',
  locationAccuracy: 0.985,
  idAccuracy: 0.999,
  initialMisplacementProbability: 0.055,
  placementErrorProbability: 0.02,
  forkliftSpeedMetersPerMinute: 78,
  routeInflation: 0.94,
  handlingMinutes: 1.5,
  confirmationDelayMinutes: 0.25,
  searchPenaltyMinutes: 0.9,
  reworkPenaltyMinutes: 18,
  deliveryReward: 110,
  wrongDeliveryPenalty: 65,
  discount: 0.96,
  qmdpTol: 1e-6,
  qmdpMaxIter: 1200,
  dueMinutes: 22,
  sensorRefreshSeconds: 0.5,
};

export function buildWarehousePOMDP(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  destinationIndex: number,
): WarehousePOMDPModel {
  validateLayout(layout);
  validateScenario(scenario);
  Preconditions.integerInRange('WarehousePOMDP', 'destinationIndex', destinationIndex, 0, layout.stations.length - 1);
  const n = layout.stations.length;
  const terminalState = n * n * 2;
  const states = Array.from({length: terminalState + 1}, (_, i) => i);
  const actions: WarehouseAction[] = layout.stations.map((s, target) => ({
    kind: 'go-to',
    target,
    label: `go to ${s.label}`,
  }));
  const observations: WarehouseObservation[] = [
    ...layout.stations.map((s, station) => ({kind: 'location' as const, station, label: `sensor says ${s.label}`})),
    {kind: 'carrying', label: 'forklift carrying pallet'},
    {kind: 'complete', label: 'delivery complete'},
  ];

  const encodeState = (forklift: number, pallet: number, carrying: boolean): number => {
    Preconditions.integerInRange('WarehousePOMDP', 'forklift', forklift, 0, n - 1);
    Preconditions.integerInRange('WarehousePOMDP', 'pallet', pallet, 0, n - 1);
    return (forklift * n + pallet) * 2 + (carrying ? 1 : 0);
  };

  const decodeState = (stateId: number): WarehouseDecisionState => {
    if (stateId === terminalState) {
      return {forklift: -1, pallet: -1, carrying: false, terminal: true};
    }
    const carrying = stateId % 2 === 1;
    let rest = Math.floor(stateId / 2);
    const pallet = rest % n;
    rest = Math.floor(rest / n);
    const forklift = rest;
    return {forklift, pallet, carrying, terminal: false};
  };

  const nextState = (stateId: number, actionIdx: number): number => {
    if (stateId === terminalState) return terminalState;
    const s = decodeState(stateId);
    const target = actions[actionIdx].target;
    if (s.carrying) {
      if (target === destinationIndex) return terminalState;
      return encodeState(target, target, true);
    }
    if (target === s.pallet) return encodeState(target, target, true);
    return encodeState(target, s.pallet, false);
  };

  const observationIndexForLocation = (stationIdx: number): number => stationIdx;
  const carryingObsIdx = n;
  const completeObsIdx = n + 1;

  const spec: POMDPSpec<number, WarehouseAction, WarehouseObservation> = {
    states,
    actions,
    observations,
    transition: (sIdx, aIdx) => {
      const row = new Array<number>(states.length).fill(0);
      row[nextState(sIdx, aIdx)] = 1;
      return row;
    },
    observation: (sNextIdx, _aIdx) => {
      const row = new Array<number>(observations.length).fill(0);
      if (sNextIdx === terminalState) {
        row[completeObsIdx] = 1;
        return row;
      }
      const s = decodeState(sNextIdx);
      if (s.carrying) {
        row[carryingObsIdx] = 1;
        return row;
      }
      const wrongMass = (1 - scenario.locationAccuracy) / Math.max(1, n - 1);
      for (let i = 0; i < n; i++) row[observationIndexForLocation(i)] = i === s.pallet
        ? scenario.locationAccuracy
        : wrongMass;
      return row;
    },
    reward: (sIdx, aIdx) => {
      if (sIdx === terminalState) return 0;
      const s = decodeState(sIdx);
      const target = actions[aIdx].target;
      const travel = travelMinutes(layout, scenario, s.forklift, target);
      const confirm = scenario.confirmationDelayMinutes;
      if (s.carrying && target === destinationIndex) {
        return scenario.deliveryReward
          - travel
          - scenario.handlingMinutes
          - confirm
          - scenario.placementErrorProbability * scenario.wrongDeliveryPenalty;
      }
      if (s.carrying) return -travel - 0.25 * scenario.handlingMinutes;
      if (target === s.pallet) return -travel - scenario.handlingMinutes - confirm;
      return -travel - scenario.searchPenaltyMinutes - confirm;
    },
    discount: scenario.discount,
    isTerminal: sIdx => sIdx === terminalState,
  };

  return {
    layout,
    scenario,
    destinationIndex,
    states,
    actions,
    observations,
    terminalState,
    spec,
    encodeState,
    decodeState,
    nextState,
    observationIndexForLocation,
  };
}

export class WarehouseQMDPSolver {
  readonly Q: number[][];
  readonly iterations: number;
  readonly finalDelta: number;

  constructor(readonly model: WarehousePOMDPModel) {
    const solved = this.solve();
    this.Q = solved.Q;
    this.iterations = solved.iterations;
    this.finalDelta = solved.finalDelta;
  }

  act(belief: DiscreteBelief<number>, rng: () => number): number {
    let bestA = 0;
    let bestQ = -Infinity;
    let ties = 0;
    for (let a = 0; a < this.model.actions.length; a++) {
      let q = 0;
      for (let s = 0; s < belief.weights.length; s++) {
        const w = belief.weights[s];
        if (w !== 0) q += w * this.Q[s][a];
      }
      if (q > bestQ + 1e-12) {
        bestA = a;
        bestQ = q;
        ties = 1;
      } else if (Math.abs(q - bestQ) <= 1e-12) {
        ties++;
        if (rng() < 1 / ties) bestA = a;
      }
    }
    return bestA;
  }

  private solve(): {Q: number[][]; iterations: number; finalDelta: number} {
    const K = this.model.states.length;
    const A = this.model.actions.length;
    const gamma = this.model.scenario.discount;
    const maxIter = this.model.scenario.qmdpMaxIter;
    const tol = this.model.scenario.qmdpTol;
    let V = new Array<number>(K).fill(0);
    let iterations = 0;
    let finalDelta = Infinity;
    for (let iter = 0; iter < maxIter; iter++) {
      const next = new Array<number>(K).fill(0);
      let delta = 0;
      for (let s = 0; s < K; s++) {
        if (s === this.model.terminalState) continue;
        let best = -Infinity;
        for (let a = 0; a < A; a++) {
          const sp = this.model.nextState(s, a);
          const q = this.model.spec.reward(s, a) + gamma * V[sp];
          if (q > best) best = q;
        }
        next[s] = best;
        const d = Math.abs(next[s] - V[s]);
        if (d > delta) delta = d;
      }
      V = next;
      iterations = iter + 1;
      finalDelta = delta;
      if (delta <= tol) break;
    }

    const Q = Array.from({length: K}, () => new Array<number>(A).fill(0));
    for (let s = 0; s < K; s++) {
      if (s === this.model.terminalState) continue;
      for (let a = 0; a < A; a++) {
        const sp = this.model.nextState(s, a);
        Q[s][a] = this.model.spec.reward(s, a) + gamma * V[sp];
      }
    }
    return {Q, iterations, finalDelta};
  }
}

export class WarehousePlanner {
  private readonly cache = new Map<number, {model: WarehousePOMDPModel; solver: WarehouseQMDPSolver}>();

  constructor(readonly layout: WarehouseLayout, readonly scenario: WarehouseScenarioConfig) {}

  forDestination(destinationIndex: number): {model: WarehousePOMDPModel; solver: WarehouseQMDPSolver} {
    const cached = this.cache.get(destinationIndex);
    if (cached) return cached;
    const model = buildWarehousePOMDP(this.layout, this.scenario, destinationIndex);
    const solver = new WarehouseQMDPSolver(model);
    const packed = {model, solver};
    this.cache.set(destinationIndex, packed);
    return packed;
  }

  chooseAction(
    destinationIndex: number,
    belief: DiscreteBelief<number>,
    rng: () => number,
  ): {model: WarehousePOMDPModel; actionIndex: number; action: WarehouseAction} {
    const {model, solver} = this.forDestination(destinationIndex);
    const actionIndex = solver.act(belief, rng);
    return {model, actionIndex, action: model.actions[actionIndex]};
  }
}

export function simulateWarehouseScenario(
  scenario: WarehouseScenarioConfig,
  opts: WarehouseSimulationOptions = {},
): WarehouseScenarioResult {
  const layout = opts.layout ?? defaultWarehouseLayout();
  validateLayout(layout);
  validateScenario(scenario);
  const jobs = opts.jobs ?? 120;
  const seed = opts.seed ?? 7;
  const maxStepsPerJob = opts.maxStepsPerJob ?? 24;
  Preconditions.integerInRange('WarehouseSimulation', 'jobs', jobs, 1, 10000);
  Preconditions.integerInRange('WarehouseSimulation', 'maxStepsPerJob', maxStepsPerJob, 1, 200);

  const floor = buildWarehouseFloor(layout);
  const rng = mulberry32(seed);
  const planner = new WarehousePlanner(layout, scenario);
  const forklift = new WarehouseForklift(`${scenario.id}-forklift-1`, layout.sourceStationId);
  forklift.activate();
  const stationToIndex = stationIndexMap(layout);
  const sourceIndex = stationToIndex.get(layout.sourceStationId)!;
  const sinkPlan = opts.destinationPlan ?? makeDestinationPlan(layout, jobs, rng);
  let now = 0;
  let completedJobs = 0;
  let failedJobs = 0;
  let shippingErrors = 0;
  let cumulativeSearchMisses = 0;
  let entropySum = 0;
  let entropyCount = 0;
  const jobSummaries: WarehouseJobSummary[] = [];
  const trace: WarehouseStepTrace[] = [];

  for (let j = 0; j < jobs; j++) {
    const destinationId = sinkPlan[j % sinkPlan.length];
    const destinationIndex = stationToIndex.get(destinationId);
    if (destinationIndex === undefined) throw new Error(`unknown destination in plan: ${destinationId}`);
    const jobId = `${scenario.id}-p${j + 1}`;
    const pallet = floor.source.emitPallet(jobId, destinationId, now);
    const initialPalletIndex = sampleInitialPalletLocation(layout, scenario, rng);
    pallet.locationId = layout.stations[initialPalletIndex].id;
    const jobStart = now;
    const initialForkliftIndex = stationToIndex.get(forklift.stationId) ?? sourceIndex;
    const {model} = planner.forDestination(destinationIndex);
    let actualState = model.encodeState(initialForkliftIndex, initialPalletIndex, false);
    let observedLocation = sampleLocationObservation(layout, scenario, initialPalletIndex, rng);
    let belief = initialWarehouseBelief(layout, scenario, model, initialForkliftIndex, observedLocation);
    let searchMisses = 0;
    let completed = false;
    let shippingError = false;
    let stepsTaken = 0;

    for (let step = 0; step < maxStepsPerJob; step++) {
      stepsTaken = step + 1;
      const before = model.decodeState(actualState);
      const beforeStationId = before.terminal ? forklift.stationId : layout.stations[before.forklift].id;
      const palletBeforeId = before.terminal ? destinationId : layout.stations[before.pallet].id;
      const {model: actionModel, actionIndex, action} = planner.chooseAction(destinationIndex, belief, rng);
      const target = action.target;
      const nextState = actionModel.nextState(actualState, actionIndex);
      const after = actionModel.decodeState(nextState);
      const duration = actionDurationMinutes(layout, scenario, before, target, destinationIndex);
      const timeStart = now;
      now += duration;

      let event: WarehouseStepTrace['event'];
      if (before.carrying && target === destinationIndex) {
        completed = true;
        completedJobs++;
        shippingError = rng() > scenario.idAccuracy || rng() < scenario.placementErrorProbability;
        if (shippingError) {
          shippingErrors++;
          now += scenario.reworkPenaltyMinutes;
          event = 'delivery-error';
        } else {
          event = 'delivered';
        }
        const delivered = forklift.drop() ?? pallet;
        const sink = floor.sinks.get(destinationId);
        if (sink) sink.collect(delivered, now, !shippingError);
        forklift.moveTo(destinationId);
      } else if (before.carrying) {
        event = 'move-loaded';
        forklift.moveTo(layout.stations[target].id);
      } else if (target === before.pallet) {
        event = 'pickup';
        forklift.moveTo(layout.stations[target].id);
        forklift.pickup(pallet);
      } else {
        event = 'search-miss';
        searchMisses++;
        cumulativeSearchMisses++;
        forklift.moveTo(layout.stations[target].id);
      }

      const obsDist = actionModel.spec.observation(nextState, actionIndex);
      const obsIdx = sampleIndex(obsDist, rng);
      observedLocation = observationToLocation(actionModel, obsIdx, observedLocation);
      belief = beliefUpdate(actionModel.spec, belief, actionIndex, obsIdx);
      const entropy = belief.entropy();
      entropySum += entropy;
      entropyCount++;

      if (opts.recordTrace ?? true) {
        const afterStationId = after.terminal
          ? destinationId
          : layout.stations[after.forklift].id;
        const palletAfterId = after.terminal
          ? destinationId
          : layout.stations[after.pallet].id;
        trace.push({
          scenarioId: scenario.id,
          jobId,
          jobIndex: j,
          step,
          timeStart,
          timeEnd: now,
          action: action.label,
          actionTarget: layout.stations[target].id,
          observation: actionModel.observations[obsIdx].label,
          event,
          destination: destinationId,
          forkliftBefore: beforeStationId,
          forkliftAfter: afterStationId,
          palletBefore: palletBeforeId,
          palletAfter: palletAfterId,
          carryingBefore: before.carrying,
          carryingAfter: after.terminal ? false : after.carrying,
          beliefEntropy: entropy,
          beliefByStation: beliefByStation(actionModel, belief),
          cumulativeDelivered: completedJobs,
          cumulativeErrors: shippingErrors,
          cumulativeSearchMisses,
          cycleTimeSoFar: now - jobStart,
        });
      }

      actualState = nextState;
      if (completed) break;
    }

    if (!completed) {
      failedJobs++;
      now += scenario.reworkPenaltyMinutes;
      trace.push({
        scenarioId: scenario.id,
        jobId,
        jobIndex: j,
        step: stepsTaken,
        timeStart: now,
        timeEnd: now,
        action: 'manual escalation',
        actionTarget: forklift.stationId,
        observation: 'job failed before delivery',
        event: 'failed',
        destination: destinationId,
        forkliftBefore: forklift.stationId,
        forkliftAfter: forklift.stationId,
        palletBefore: pallet.locationId,
        palletAfter: pallet.locationId,
        carryingBefore: !!forklift.carrying,
        carryingAfter: !!forklift.carrying,
        beliefEntropy: belief.entropy(),
        beliefByStation: beliefByStation(model, belief),
        cumulativeDelivered: completedJobs,
        cumulativeErrors: shippingErrors,
        cumulativeSearchMisses,
        cycleTimeSoFar: now - jobStart,
      });
      forklift.carrying = undefined;
    }

    const cycleTime = now - jobStart;
    jobSummaries.push({
      jobId,
      destination: destinationId,
      completed,
      shippingError,
      cycleTime,
      steps: stepsTaken,
      searchMisses,
      onTime: completed && cycleTime <= scenario.dueMinutes,
    });
  }

  const totalCycle = sum(jobSummaries.map(j => j.cycleTime));
  const completed = jobSummaries.filter(j => j.completed);
  const metrics: WarehouseMetrics = {
    jobsCreated: jobs,
    completedJobs,
    failedJobs,
    shippingErrors,
    shippingErrorRate: completedJobs > 0 ? shippingErrors / completedJobs : 0,
    totalTime: now,
    meanCycleTime: jobSummaries.length > 0 ? totalCycle / jobSummaries.length : 0,
    throughputPerHour: now > 0 ? completedJobs / now * 60 : 0,
    onTimeRate: completed.length > 0 ? completed.filter(j => j.onTime).length / completed.length : 0,
    meanStepsPerJob: mean(jobSummaries.map(j => j.steps)),
    meanSearchMissesPerJob: mean(jobSummaries.map(j => j.searchMisses)),
    meanBeliefEntropy: entropyCount > 0 ? entropySum / entropyCount : 0,
  };

  return {scenario, layout, metrics, jobs: jobSummaries, trace};
}

export function runWarehouseComparison(opts: WarehouseSimulationOptions = {}): WarehouseComparisonResult {
  const layout = opts.layout ?? defaultWarehouseLayout();
  const seed = opts.seed ?? 7;
  const jobs = opts.jobs ?? 120;
  const destinationPlan = opts.destinationPlan ?? makeDestinationPlan(layout, jobs, mulberry32(seed + 404));
  const baseline = simulateWarehouseScenario(BASELINE_WAREHOUSE_SCENARIO, {
    ...opts,
    layout,
    jobs,
    seed,
    destinationPlan,
  });
  const track3t = simulateWarehouseScenario(TRACK3T_WAREHOUSE_SCENARIO, {
    ...opts,
    layout,
    jobs,
    seed,
    destinationPlan,
  });
  return {
    layout,
    baseline,
    track3t,
    deltas: {
      meanCycleTimeReductionPct: pctReduction(baseline.metrics.meanCycleTime, track3t.metrics.meanCycleTime),
      throughputLiftPct: pctLift(baseline.metrics.throughputPerHour, track3t.metrics.throughputPerHour),
      searchMissReductionPct: pctReduction(baseline.metrics.meanSearchMissesPerJob, track3t.metrics.meanSearchMissesPerJob),
      errorReductionPct: pctReduction(baseline.metrics.shippingErrorRate, track3t.metrics.shippingErrorRate),
      entropyReductionPct: pctReduction(baseline.metrics.meanBeliefEntropy, track3t.metrics.meanBeliefEntropy),
    },
    sourceNotes: TRACK3T_ARCHIVE_GROUNDING,
  };
}

export function summarizeWarehouseComparison(result: WarehouseComparisonResult): string {
  const rows = [
    ['metric', 'baseline', 'track3t'],
    ['completed', String(result.baseline.metrics.completedJobs), String(result.track3t.metrics.completedJobs)],
    ['mean cycle min', fmt(result.baseline.metrics.meanCycleTime), fmt(result.track3t.metrics.meanCycleTime)],
    ['throughput jobs/hr', fmt(result.baseline.metrics.throughputPerHour), fmt(result.track3t.metrics.throughputPerHour)],
    ['search misses/job', fmt(result.baseline.metrics.meanSearchMissesPerJob), fmt(result.track3t.metrics.meanSearchMissesPerJob)],
    ['shipping error rate', pct(result.baseline.metrics.shippingErrorRate), pct(result.track3t.metrics.shippingErrorRate)],
    ['on-time rate', pct(result.baseline.metrics.onTimeRate), pct(result.track3t.metrics.onTimeRate)],
    ['mean belief entropy', fmt(result.baseline.metrics.meanBeliefEntropy), fmt(result.track3t.metrics.meanBeliefEntropy)],
  ];
  const widths = rows[0].map((_, i) => Math.max(...rows.map(r => r[i].length)));
  return rows.map((r, idx) => {
    const line = r.map((c, i) => c.padEnd(widths[i])).join('  ');
    return idx === 0 ? line + '\n' + widths.map(w => '-'.repeat(w)).join('  ') : line;
  }).join('\n');
}

export function buildWarehouseFloor(layout: WarehouseLayout): {
  source: WarehouseSource;
  sinks: Map<string, WarehouseSink>;
  stations: Map<string, WarehouseStation>;
} {
  validateLayout(layout);
  const stations = new Map<string, WarehouseStation>();
  const sinks = new Map<string, WarehouseSink>();
  let source: WarehouseSource | undefined;
  for (const def of layout.stations) {
    if (def.id === layout.sourceStationId) {
      source = new WarehouseSource(def);
      stations.set(def.id, source);
    } else if (layout.sinkStationIds.includes(def.id)) {
      const sink = new WarehouseSink(def);
      sinks.set(def.id, sink);
      stations.set(def.id, sink);
    } else {
      stations.set(def.id, new WarehouseStation(def));
    }
  }
  if (!source) throw new Error(`source station not found: ${layout.sourceStationId}`);
  return {source, sinks, stations};
}

export function initialWarehouseBelief(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  model: WarehousePOMDPModel,
  forkliftIndex: number,
  observedLocationIndex: number,
): DiscreteBelief<number> {
  const n = layout.stations.length;
  const prior = initialLocationPrior(layout, scenario);
  const locationPosterior = new Array<number>(n).fill(0);
  for (let loc = 0; loc < n; loc++) {
    locationPosterior[loc] = prior[loc] * locationObservationProbability(n, scenario, loc, observedLocationIndex);
  }
  normalizeInPlace(locationPosterior);
  const weights = new Array<number>(model.states.length).fill(0);
  for (let loc = 0; loc < n; loc++) {
    weights[model.encodeState(forkliftIndex, loc, false)] = locationPosterior[loc];
  }
  return new DiscreteBelief(model.states, weights);
}

export function beliefByStation(model: WarehousePOMDPModel, belief: DiscreteBelief<number>): number[] {
  const n = model.layout.stations.length;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < belief.weights.length; i++) {
    const w = belief.weights[i];
    if (w === 0 || i === model.terminalState) continue;
    const s = model.decodeState(i);
    if (s.terminal) continue;
    out[s.pallet] += w;
  }
  return out;
}

export function travelMinutes(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  fromStationIdx: number,
  toStationIdx: number,
): number {
  if (fromStationIdx === toStationIdx) return 0;
  const meters = manhattanDistance(layout, fromStationIdx, toStationIdx) * (layout.gridMeters ?? 12);
  return meters / scenario.forkliftSpeedMetersPerMinute * scenario.routeInflation;
}

function actionDurationMinutes(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  before: WarehouseDecisionState,
  target: number,
  destinationIndex: number,
): number {
  if (before.terminal) return 0;
  const travel = travelMinutes(layout, scenario, before.forklift, target);
  if (before.carrying && target === destinationIndex) {
    return travel + scenario.handlingMinutes + scenario.confirmationDelayMinutes;
  }
  if (before.carrying) return travel;
  if (target === before.pallet) {
    return travel + scenario.handlingMinutes + scenario.confirmationDelayMinutes;
  }
  return travel + scenario.searchPenaltyMinutes + scenario.confirmationDelayMinutes;
}

function sampleInitialPalletLocation(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  rng: () => number,
): number {
  const stationToIndex = stationIndexMap(layout);
  const sourceIndex = stationToIndex.get(layout.sourceStationId)!;
  if (rng() >= scenario.initialMisplacementProbability) return sourceIndex;
  const candidates = palletCandidateIndexes(layout).filter(i => i !== sourceIndex);
  return candidates[Math.floor(rng() * candidates.length)] ?? sourceIndex;
}

function sampleLocationObservation(
  layout: WarehouseLayout,
  scenario: WarehouseScenarioConfig,
  trueLocationIndex: number,
  rng: () => number,
): number {
  const n = layout.stations.length;
  if (rng() < scenario.locationAccuracy) return trueLocationIndex;
  let obs = Math.floor(rng() * (n - 1));
  if (obs >= trueLocationIndex) obs++;
  return obs;
}

function observationToLocation(model: WarehousePOMDPModel, obsIdx: number, fallback: number): number {
  const obs = model.observations[obsIdx];
  return obs.kind === 'location' && obs.station !== undefined ? obs.station : fallback;
}

function initialLocationPrior(layout: WarehouseLayout, scenario: WarehouseScenarioConfig): number[] {
  const n = layout.stations.length;
  const stationToIndex = stationIndexMap(layout);
  const sourceIndex = stationToIndex.get(layout.sourceStationId)!;
  const candidates = palletCandidateIndexes(layout);
  const prior = new Array<number>(n).fill(0);
  prior[sourceIndex] = 1 - scenario.initialMisplacementProbability;
  const others = candidates.filter(i => i !== sourceIndex);
  const share = scenario.initialMisplacementProbability / Math.max(1, others.length);
  for (const i of others) prior[i] = share;
  normalizeInPlace(prior);
  return prior;
}

function locationObservationProbability(
  numStations: number,
  scenario: WarehouseScenarioConfig,
  trueLocationIndex: number,
  observedLocationIndex: number,
): number {
  if (trueLocationIndex === observedLocationIndex) return scenario.locationAccuracy;
  return (1 - scenario.locationAccuracy) / Math.max(1, numStations - 1);
}

function makeDestinationPlan(layout: WarehouseLayout, jobs: number, rng: () => number): string[] {
  const out: string[] = [];
  for (let i = 0; i < jobs; i++) {
    out.push(layout.sinkStationIds[Math.floor(rng() * layout.sinkStationIds.length)]);
  }
  return out;
}

function palletCandidateIndexes(layout: WarehouseLayout): number[] {
  return layout.stations
    .map((s, i) => ({s, i}))
    .filter(({s}) => s.kind !== 'sink' && s.canHoldPallet !== false)
    .map(({i}) => i);
}

function manhattanDistance(layout: WarehouseLayout, aIdx: number, bIdx: number): number {
  const a = layout.stations[aIdx];
  const b = layout.stations[bIdx];
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function stationIndexMap(layout: WarehouseLayout): Map<string, number> {
  return new Map(layout.stations.map((s, i) => [s.id, i]));
}

function validateLayout(layout: WarehouseLayout): void {
  Preconditions.nonEmpty('WarehouseLayout', 'stations', layout.stations);
  const ids = new Set<string>();
  for (const [i, s] of layout.stations.entries()) {
    if (ids.has(s.id)) throw new Error(`WarehouseLayout: duplicate station id ${s.id}`);
    ids.add(s.id);
    Preconditions.finite('WarehouseLayout', `stations[${i}].x`, s.x);
    Preconditions.finite('WarehouseLayout', `stations[${i}].y`, s.y);
  }
  if (!ids.has(layout.sourceStationId)) throw new Error(`WarehouseLayout: source missing: ${layout.sourceStationId}`);
  for (const id of layout.sinkStationIds) {
    if (!ids.has(id)) throw new Error(`WarehouseLayout: sink missing: ${id}`);
  }
  if (layout.gridMeters !== undefined) Preconditions.positive('WarehouseLayout', 'gridMeters', layout.gridMeters);
}

function validateScenario(s: WarehouseScenarioConfig): void {
  const model = `WarehouseScenario(${s.id})`;
  Preconditions.inRange(model, 'locationAccuracy', s.locationAccuracy, 0.5, 1);
  Preconditions.inRange(model, 'idAccuracy', s.idAccuracy, 0, 1);
  Preconditions.inRange(model, 'initialMisplacementProbability', s.initialMisplacementProbability, 0, 1);
  Preconditions.inRange(model, 'placementErrorProbability', s.placementErrorProbability, 0, 1);
  Preconditions.positive(model, 'forkliftSpeedMetersPerMinute', s.forkliftSpeedMetersPerMinute);
  Preconditions.positive(model, 'routeInflation', s.routeInflation);
  Preconditions.nonNegative(model, 'handlingMinutes', s.handlingMinutes);
  Preconditions.nonNegative(model, 'confirmationDelayMinutes', s.confirmationDelayMinutes);
  Preconditions.nonNegative(model, 'searchPenaltyMinutes', s.searchPenaltyMinutes);
  Preconditions.nonNegative(model, 'reworkPenaltyMinutes', s.reworkPenaltyMinutes);
  Preconditions.positive(model, 'deliveryReward', s.deliveryReward);
  Preconditions.nonNegative(model, 'wrongDeliveryPenalty', s.wrongDeliveryPenalty);
  Preconditions.inRange(model, 'discount', s.discount, 0, 1);
  Preconditions.positive(model, 'qmdpTol', s.qmdpTol);
  Preconditions.integerInRange(model, 'qmdpMaxIter', s.qmdpMaxIter, 1, 100000);
  Preconditions.positive(model, 'dueMinutes', s.dueMinutes);
  Preconditions.positive(model, 'sensorRefreshSeconds', s.sensorRefreshSeconds);
}

function sampleIndex(probabilities: ReadonlyArray<number>, rng: () => number): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probabilities.length; i++) {
    acc += probabilities[i];
    if (u <= acc) return i;
  }
  return probabilities.length - 1;
}

function normalizeInPlace(xs: number[]): void {
  const total = sum(xs);
  if (!Number.isFinite(total) || total <= 0) {
    const u = 1 / xs.length;
    for (let i = 0; i < xs.length; i++) xs[i] = u;
    return;
  }
  for (let i = 0; i < xs.length; i++) xs[i] /= total;
}

function sum(xs: readonly number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

function mean(xs: readonly number[]): number {
  return xs.length > 0 ? sum(xs) / xs.length : 0;
}

function pctReduction(a: number, b: number): number {
  if (Math.abs(a) < 1e-12) return b < a ? 100 : 0;
  return (a - b) / Math.abs(a) * 100;
}

function pctLift(a: number, b: number): number {
  if (Math.abs(a) < 1e-12) return b > a ? 100 : 0;
  return (b - a) / Math.abs(a) * 100;
}

function fmt(x: number): string {
  return Number.isFinite(x) ? x.toFixed(3) : 'n/a';
}

function pct(x: number): string {
  return Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : 'n/a';
}
