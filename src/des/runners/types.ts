// RUST MIGRATION:
// - Target: src/des/runners/types.rs.
// - Keep this as the shared data-model module; Kernel becomes an enum and SimConfig/RunOpts/RunResult become serde structs.
// - Replace Record<string, ...> with HashMap/BTreeMap or typed compartment enums where deterministic order matters.
// - Constants such as COMPARTMENT_ORDER, DEFAULT_CONFIG, and EDGES should become const/static constructors with explicit ownership.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/runners/types.rs   (module des::runners::types)
// 1:1 file move. Shared config/result types for the SEIR kernel runners.
//
// Declarations → Rust:
//   type Kernel = 'framework' | 'fel' | ...  -> enum Kernel { Framework, Fel, .. }
//                                               (#[serde(rename_all="kebab-case")])
//   interface SimConfig / RunOpts / RunResult -> struct (#[derive(Clone, Serialize, Deserialize)])
//
// Conversion notes (file-specific):
//   - `[number, number]` interarrival/residence tuples -> `(f64, f64)`.
//   - `Record<string, number>` population maps -> `HashMap<String, f64>` (or a
//     fixed-size array indexed by COMPARTMENT_ORDER for speed).
//   - This is a pure data module: no I/O, no RNG — translate first.
// =============================================================================

// =============================================================================
// Shared types used by all three kernel runners (framework, FEL reference,
// per-individual-clock framework variant). Keeping these in one place makes
// it easy to swap kernels in the replication / stepsize-sweep drivers.
// =============================================================================

export type Kernel = 'framework' | 'fel' | 'per-individual' | 'gillespie' | 'ode' | 'difference';

export interface SimConfig {
  /** Days per discrete step. Only the framework kernel uses it. */
  stepSize: number;
  /** Total simulation horizon in days. */
  horizonDays: number;
  /** Days during which the source is active. After this we drain. */
  phase1Days: number;
  /** How many entities the source emits before quiescing. */
  sourceCap: number;
  /** Inter-arrival uniform [a, b] (days) for the source. */
  arrivalsInterarrival: [number, number];
  /** Per-station service-clock uniform [a, b] (days) at every station. */
  residence: Record<string, [number, number]>;
  probabilities: {
    asymptomaticShare: number;
    hospitalizationGivenSymptom: number;
    caseFatalityGivenHospital: number;
  };
}

export interface RunOpts {
  /** Seed for the seedable PRNG; deterministic when supplied. */
  seed?: number;
  /** If true, dump JSONL events to logPath. Defaults to false (cheap). */
  logEvents?: boolean;
  /** Where to write the JSONL log if logEvents=true. */
  logPath?: string;
  /** Sample populations only every N days (default 1). */
  sampleEveryDays?: number;
  /**
   * FEL service discipline.
   *  'fifo'        - Single per-station service clock (M/M/1, matches the
   *                  framework's three-queue EntityProcessor). Default.
   *  'individual'  - Per-entity exit clock at arrival (M/M/inf, matches the
   *                  new framework PerIndividualProcessor).
   */
  service?: 'fifo' | 'individual';
}

export interface RunResult {
  kernel: Kernel;
  config: SimConfig;
  seed: number;
  totals: {
    created: number;
    absorbed: number;
  };
  finalPopulations: Record<string, number>;
  /** Counts of (from -> to) transitions, decision nodes flattened away. */
  transitionCounts: Record<string, Record<string, number>>;
  /** Empirical row-stochastic transition matrix. */
  splitProbs: Record<string, Record<string, number>>;
  /** Mean over per-day samples. */
  timeAvgPopulations: Record<string, number>;
  peakPopulations: Record<string, number>;
  elapsedMs: number;
}

export const COMPARTMENT_ORDER = ['S', 'E', 'I-P', 'I-A', 'I-S', 'I-H', 'R'];

export const COMPARTMENT_GROUPS: Record<string, string[]> = {
  'S':   ['S'],
  'E':   ['E'],
  'I-P': ['I-P', 'I-P-Decision'],
  'I-A': ['I-A'],
  'I-S': ['I-S', 'I-S-Decision'],
  'I-H': ['I-H', 'I-H-Decision'],
  'R':   ['R'],
};

export const DEFAULT_RESIDENCE: Record<string, [number, number]> = {
  'S':   [0.20, 0.40], 'E':   [0.20, 0.40], 'I-P': [0.20, 0.40],
  'I-A': [0.20, 0.40], 'I-S': [0.20, 0.40], 'I-H': [0.20, 0.40],
  'R':   [1.50, 2.50], 'D':   [0.10, 0.30],
  'I-P-Decision': [0.05, 0.15],
  'I-S-Decision': [0.05, 0.15],
  'I-H-Decision': [0.05, 0.15],
};

export const DEFAULT_CONFIG: SimConfig = {
  stepSize: 1.0,
  horizonDays: 1200,
  phase1Days: 800,
  sourceCap: 500,
  arrivalsInterarrival: [0.7, 1.3],
  residence: DEFAULT_RESIDENCE,
  probabilities: {
    asymptomaticShare:           0.40,
    hospitalizationGivenSymptom: 0.20,
    caseFatalityGivenHospital:   0.12,
  },
};

export const EDGES: Array<[string, string]> = [
  ['main-source',   'S'],
  ['S',             'E'],
  ['E',             'I-P'],
  ['I-P',           'I-P-Decision'],
  ['I-P-Decision',  'I-A'],
  ['I-P-Decision',  'I-S'],
  ['I-A',           'R'],
  ['I-S',           'I-S-Decision'],
  ['I-S-Decision',  'R'],
  ['I-S-Decision',  'I-H'],
  ['I-H',           'I-H-Decision'],
  ['I-H-Decision',  'R'],
  ['I-H-Decision',  'D'],
  ['D',             'main-sink'],
  ['R',             'S'],
];

/**
 * Successor map (used by FEL and per-individual kernels). Mirrors EDGES but
 * folds branching probabilities into the from-station whose successors are
 * the decision node.
 */
export function buildSuccessors(probs: SimConfig['probabilities']):
    Record<string, Array<{prob: number, to: string}>> {
  return {
    'main-source':  [{prob: 1, to: 'S'}],
    'S':            [{prob: 1, to: 'E'}],
    'E':            [{prob: 1, to: 'I-P'}],
    'I-P':          [{prob: 1, to: 'I-P-Decision'}],
    'I-P-Decision': [
      {prob:     probs.asymptomaticShare, to: 'I-A'},
      {prob: 1 - probs.asymptomaticShare, to: 'I-S'},
    ],
    'I-A':          [{prob: 1, to: 'R'}],
    'I-S':          [{prob: 1, to: 'I-S-Decision'}],
    'I-S-Decision': [
      {prob: 1 - probs.hospitalizationGivenSymptom, to: 'R'},
      {prob:     probs.hospitalizationGivenSymptom, to: 'I-H'},
    ],
    'I-H':          [{prob: 1, to: 'I-H-Decision'}],
    'I-H-Decision': [
      {prob: 1 - probs.caseFatalityGivenHospital, to: 'R'},
      {prob:     probs.caseFatalityGivenHospital, to: 'D'},
    ],
    'R':            [{prob: 1, to: 'S'}],
    'D':            [{prob: 1, to: 'main-sink'}],
  };
}
