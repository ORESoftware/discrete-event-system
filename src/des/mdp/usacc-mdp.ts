'use strict';

// RUST MIGRATION:
// - Target: src/des/mdp/usacc_mdp.rs
// - The const arrays/types should become Rust enums plus `TryFrom<usize>` or
//   fixed lookup tables; CaseState and Outcome become Copy/Clone structs.
// - encode/decode/quality/reward/outcomes/sampleInitialState are pure functions;
//   if keeping the TS migration pattern, group them behind PureTransform-style
//   structs before porting, then translate to trait impls or associated fns.
// - Replace TS union-string action types, Record<Action, number>, non-null
//   assertions, and floating probability checks with enums, arrays indexed by
//   discriminants, Option/Result, and deterministic f64/decimal tolerances.
// - Inject RNG for sampleInitialState instead of accepting an untyped closure.

// =============================================================================
// RUST MIGRATION  —  target: src/des/mdp/usacc-mdp.rs  (module des::mdp::usacc_mdp)
// 1:1 file move. The USACC court-case MDP: states, actions, transitions, rewards.
//
// Declarations → Rust:
//   const STAGES/EVIDENCE/CORROBORATION/MANIPULATION/CONFLICT/FUNDING (`as const`)
//     + type Stage/Evidence/... = typeof X[number]   -> string-literal unions -> enums
//   interface CaseState                              -> struct CaseState { stage, evidence, ... } (all u8)
//   const N_STATES/ACCEPTED/CLOSED/EXHAUSTED/N_ACTIONS/FUND_*  -> consts
//   const ACTIONS (`as const`) + type Action          -> enum Action (+ VARIANTS array)
//   interface Outcome                                 -> struct Outcome { next_state, prob, reward }
//   const ACTION_COST / drawPctPerAction: Record<Action,number> -> match on Action / array[Action as usize]
//   fn encode/decode/isTerminal/quality/terminalReward/rewardOfAccept/rewardOfClose/outcomes -> free fns
//   fn sampleInitialState(rng)                         -> fn taking an injected RandomSource
//
// Conversion notes (file-specific):
//   - The factor unions (Stage/Evidence/…) and the `... as const` arrays -> Rust
//     enums; CaseState stores them as small ints (0..N) exactly as here.
//   - `decode(id)!` non-null -> `Option<CaseState>` + `expect`.
//   - `Record<Action, number>` cost/draw tables -> `match` on the Action enum (or an
//     array indexed by `action as usize`), not a HashMap.
//   - Local `type Edge` built via `{...stay, ...e}` spread + `Partial<Edge>`/`Omit` ->
//     a struct with `#[derive(Default,Clone)]` + struct-update syntax (`Edge { ..stay }`).
//   - `Map<number, Outcome>` coalescing -> `HashMap<u32, Outcome>` (sum probs).
//   - Integer-percent probs divided by 10000 at the end (kept so TS/py/Rust round
//     identically) -> do the integer math in `u32`, divide into `f64` once.
//   - `throw new Error('probability sum != 1')` -> `panic!`/`debug_assert!` (invariant).
//   - `sampleInitialState(rng: () => number)` -> inject `RandomSource`; PRESERVE the
//     exact number/order of `rng()` draws (each call advances RNG state — sequencing matters).
//   - `switch (a)` on the action string -> `match` on the Action enum.
// =============================================================================

// =============================================================================
// MDP for the USACC (US Anti-Corruption Court) project.
//
// Source spec: https://oresoftware.github.io/us-anti-corruption-court-project/mdp
//
// We model the MDP version (NOT the POMDP version): the visible state IS
// the ground truth that the policy acts on. The realistic system is a POMDP
// because the most important variables (collusion, witness reliability,
// strategic delay, etc.) are hidden — but the project page itself notes that
// the simplified MDP is "useful for queue management, reviewer allocation,
// and funding logic" and that's the part we model.
//
// State factors and their domain sizes:
//
//   case_stage         ∈ {SUB, VAL, ADM, TRI}                 — 4
//   evidence_strength  ∈ {LO, MED, HI}                        — 3
//   corroboration      ∈ {NONE, SINGLE, MULTI}                — 3
//   manipulation_risk  ∈ {LO, MED, HI}                        — 3
//   conflict_risk      ∈ {LO, HI}                             — 2
//   funding_status     ∈ {UNFUNDED, ESCROWED, ACTIVE, EXHAUSTED} — 4
//
//   non-terminal states     = 4·3·3·3·2·4 = 864
//   terminal states         = 3 (ACCEPTED, CLOSED, EXHAUSTED)
//   total                   = 867
//
// State encoding (sequential integer):
//
//   id(s) = ((((stage·3 + ev)·3 + corr)·3 + man)·2 + conf)·4 + fund
//   ACCEPTED  = 864
//   CLOSED    = 865
//   EXHAUSTED = 866
//
// Actions (per the project page):
//
//   0 request_more_evidence
//   1 verify_identity
//   2 normalize_record
//   3 assign_reviewers
//   4 hold_for_audit
//   5 escalate_to_next_stage
//   6 release_escrow
//   7 reject_or_close
//
// Transition model: each (s, a) yields a list of (s', p, r) outcomes, where
// p sums to 1 and r is the immediate reward of the (s, a, s') triple.
// Terminal rewards live entirely in the (s, a, terminal) outcome — there
// are no per-step rewards once a terminal state is reached.
//
// Reward model (per-action small negative cost, per-terminal large signed):
//
//   per-action cost            -1 .. -5  (encoded in transitions below)
//   ACCEPTED  reward = 50 * (Q - 0.5)    where Q = ev + corr - man - 1.5*conf
//   CLOSED    reward = 50 * (0.5 - Q)
//   EXHAUSTED reward = -150
// =============================================================================

// -----------------------------------------------------------------------------
// State factor enumerations.
// -----------------------------------------------------------------------------

export const STAGES = ['SUB', 'VAL', 'ADM', 'TRI'] as const;
export const EVIDENCE = ['LO', 'MED', 'HI'] as const;
export const CORROBORATION = ['NONE', 'SINGLE', 'MULTI'] as const;
export const MANIPULATION = ['LO', 'MED', 'HI'] as const;
export const CONFLICT = ['LO', 'HI'] as const;
export const FUNDING = ['UNFUNDED', 'ESCROWED', 'ACTIVE', 'EXHAUSTED'] as const;

export type Stage         = typeof STAGES[number];
export type Evidence      = typeof EVIDENCE[number];
export type Corroboration = typeof CORROBORATION[number];
export type Manipulation  = typeof MANIPULATION[number];
export type Conflict      = typeof CONFLICT[number];
export type Funding       = typeof FUNDING[number];

export interface CaseState {
  stage: number;        // 0..3
  evidence: number;     // 0..2
  corroboration: number;// 0..2
  manipulation: number; // 0..2
  conflict: number;     // 0..1
  funding: number;      // 0..3
}

export const N_STATES = 4 * 3 * 3 * 3 * 2 * 4 + 3; // 867
export const ACCEPTED  = 864;
export const CLOSED    = 865;
export const EXHAUSTED = 866;

export const ACTIONS = [
  'request_more_evidence',
  'verify_identity',
  'normalize_record',
  'assign_reviewers',
  'hold_for_audit',
  'escalate_to_next_stage',
  'release_escrow',
  'reject_or_close',
] as const;
export type Action = typeof ACTIONS[number];
export const N_ACTIONS = ACTIONS.length;

export const FUND_UNFUNDED  = 0;
export const FUND_ESCROWED  = 1;
export const FUND_ACTIVE    = 2;
export const FUND_EXHAUSTED = 3;

// -----------------------------------------------------------------------------
// State encoding / decoding.
// -----------------------------------------------------------------------------

export function encode(s: CaseState): number {
  return ((((s.stage * 3 + s.evidence) * 3 + s.corroboration) * 3 + s.manipulation) * 2 + s.conflict) * 4 + s.funding;
}

export function decode(id: number): CaseState | null {
  if (id >= 864) return null;
  const funding       = id % 4; id = Math.floor(id / 4);
  const conflict      = id % 2; id = Math.floor(id / 2);
  const manipulation  = id % 3; id = Math.floor(id / 3);
  const corroboration = id % 3; id = Math.floor(id / 3);
  const evidence      = id % 3; id = Math.floor(id / 3);
  const stage         = id;
  return {stage, evidence, corroboration, manipulation, conflict, funding};
}

export function isTerminal(id: number): boolean { return id >= 864; }

// -----------------------------------------------------------------------------
// Reward model.
// -----------------------------------------------------------------------------

/**
 * Quality score Q ∈ [-3.5, +4]: how genuinely strong / clean / honest the
 * case is, given its visible factors. Used to compute terminal rewards.
 * High Q → escalating to ACCEPTED is good and closing is bad. Low Q → vice
 * versa.
 */
export function quality(s: CaseState): number {
  return s.evidence + s.corroboration - s.manipulation - 1.5 * s.conflict;
}

export function terminalReward(id: number): number {
  if (id === ACCEPTED)  return 0;   // populated at transition time
  if (id === CLOSED)    return 0;   // populated at transition time
  if (id === EXHAUSTED) return -150;
  return 0;
}

/** Reward of arriving at ACCEPTED from a (presumably non-terminal) state s. */
export function rewardOfAccept(s: CaseState): number { return 50 * (quality(s) - 0.5); }
/** Reward of arriving at CLOSED from state s. */
export function rewardOfClose(s: CaseState):  number { return 50 * (0.5 - quality(s)); }

// -----------------------------------------------------------------------------
// Transition model.
// -----------------------------------------------------------------------------

export interface Outcome {
  nextState: number;    // state id (0..866)
  prob:      number;
  reward:    number;
}

const ACTION_COST: Record<Action, number> = {
  request_more_evidence:  -2,
  verify_identity:        -2,
  normalize_record:       -1,
  assign_reviewers:       -3,
  hold_for_audit:         -5,
  escalate_to_next_stage: -2,
  release_escrow:         -1,
  reject_or_close:         0,
};

/**
 * Build the outcome list for (s, a). Pure function — same input always
 * produces the same list. Probabilities sum to 1.
 *
 * The transition model is hand-tuned but designed so:
 *   - Each helpful action has a meaningful chance of advancing exactly the
 *     factor it is named for, but a smaller chance of perturbing other
 *     factors.
 *   - Funding monotonically degrades (UNFUNDED ← ESCROWED ← ACTIVE ← never
 *     refills). Each action draws down funding with action-specific prob.
 *   - escalate from TRI = ACCEPTED. reject_or_close from anything = CLOSED.
 *   - If funding hits EXHAUSTED at any point, the case becomes EXHAUSTED.
 *
 * Probabilities are encoded as integer percents and divided at the end so
 * floats round identically in TS and Python.
 */
export function outcomes(stateId: number, action: number): Outcome[] {
  if (isTerminal(stateId)) {
    // Terminal states absorb.
    return [{nextState: stateId, prob: 1.0, reward: 0}];
  }
  const s = decode(stateId)!;
  const a = ACTIONS[action];
  const cost = ACTION_COST[a];

  // Funding is special: the case becomes EXHAUSTED if funding drops below 0
  // (i.e. tries to draw from UNFUNDED).
  type Edge = {
    nextStage: number; nextEv: number; nextCorr: number;
    nextMan: number; nextConf: number; nextFund: number;
    pct: number;     // probability in integer percent (sum to 100)
    extraReward: number;
    targetTerminal?: number;
  };
  const edges: Edge[] = [];

  const stay: Omit<Edge, 'pct' | 'extraReward'> = {
    nextStage: s.stage, nextEv: s.evidence, nextCorr: s.corroboration,
    nextMan: s.manipulation, nextConf: s.conflict, nextFund: s.funding,
  };

  const funded = s.funding > FUND_UNFUNDED;

  // Helper: build edges by splitting probability across factor changes.
  const addEdge = (e: Partial<Edge> = {}, pct: number, extraReward = 0) => {
    edges.push({
      ...stay,
      ...e,
      pct,
      extraReward,
    } as Edge);
  };

  switch (a) {
    case 'request_more_evidence': {
      // 60% evidence +1, 30% nothing, 10% reveal manipulation +1.
      // Funding draw 30% of the time (if funded).
      const evUp = Math.min(s.evidence + 1, 2);
      const manUp = Math.min(s.manipulation + 1, 2);
      addEdge({nextEv: evUp},   60);
      addEdge({},               30);
      addEdge({nextMan: manUp}, 10);
      break;
    }
    case 'verify_identity': {
      // 50% corroboration +1, 30% nothing, 20% manipulation -1 (if > LO).
      const corrUp = Math.min(s.corroboration + 1, 2);
      const manDn  = Math.max(s.manipulation - 1, 0);
      addEdge({nextCorr: corrUp},  50);
      addEdge({},                  30);
      addEdge({nextMan: manDn},    20);
      break;
    }
    case 'normalize_record': {
      // 30% evidence +1, 40% conflict resolves to LO (if HI), 30% nothing.
      const evUp = Math.min(s.evidence + 1, 2);
      addEdge({nextEv: evUp},                                 30);
      addEdge({nextConf: 0},                                  40);
      addEdge({},                                             30);
      break;
    }
    case 'assign_reviewers': {
      // 60% conflict resolves to LO, 30% evidence +1, 10% nothing.
      const evUp = Math.min(s.evidence + 1, 2);
      addEdge({nextConf: 0},  60);
      addEdge({nextEv: evUp}, 30);
      addEdge({},             10);
      break;
    }
    case 'hold_for_audit': {
      // 70% manipulation collapses to LO, 20% evidence +1, 10% nothing.
      const evUp = Math.min(s.evidence + 1, 2);
      addEdge({nextMan: 0},   70);
      addEdge({nextEv: evUp}, 20);
      addEdge({},             10);
      break;
    }
    case 'escalate_to_next_stage': {
      if (s.stage === 3) {
        // From TRI, escalation = ACCEPTED.
        return [{
          nextState: ACCEPTED,
          prob: 1,
          reward: cost + rewardOfAccept(s),
        }];
      }
      // 80% stage advances, 20% fails (conflict surfaces) — stage stays,
      // conflict goes HI.
      addEdge({nextStage: s.stage + 1},       80);
      addEdge({nextConf: 1},                  20);
      break;
    }
    case 'release_escrow': {
      // funding advances by 1 (UNFUNDED→ESC→ACT→EXH not allowed; we cap at
      // ACTIVE). 100% deterministic.
      const fundNext = Math.min(s.funding + 1, FUND_ACTIVE);
      addEdge({nextFund: fundNext}, 100);
      break;
    }
    case 'reject_or_close': {
      return [{
        nextState: CLOSED,
        prob: 1,
        reward: cost + rewardOfClose(s),
      }];
    }
  }

  // Apply funding draw to every edge by the action's draw probability.
  // For simplicity we apply a flat 25% funding-draw across all edges of
  // every action except release_escrow and reject_or_close. release_escrow
  // adds rather than subtracts, and reject_or_close is terminal.
  const drawPctPerAction: Record<Action, number> = {
    request_more_evidence:  25,
    verify_identity:        25,
    normalize_record:       10,
    assign_reviewers:       30,
    hold_for_audit:         50,
    escalate_to_next_stage: 25,
    release_escrow:          0,
    reject_or_close:         0,
  };
  const drawPct = drawPctPerAction[a];

  // Build the final outcome list. For each base edge, fork it into
  // (no-draw, with-draw). Funding decrease: ACT→ESC→UNF→EXHAUSTED.
  const out: Outcome[] = [];
  for (const e of edges) {
    const baseProb = e.pct * (100 - drawPct) / 10000;
    const drawProb = e.pct * drawPct        / 10000;
    if (baseProb > 0) {
      const sNext: CaseState = {
        stage: e.nextStage, evidence: e.nextEv, corroboration: e.nextCorr,
        manipulation: e.nextMan, conflict: e.nextConf, funding: e.nextFund,
      };
      out.push({nextState: encode(sNext), prob: baseProb, reward: cost});
    }
    if (drawProb > 0) {
      const fAfterDraw = e.nextFund - 1;
      if (fAfterDraw < FUND_UNFUNDED) {
        // Funding underflow → EXHAUSTED.
        out.push({nextState: EXHAUSTED, prob: drawProb, reward: cost - 150});
      } else {
        const sNext: CaseState = {
          stage: e.nextStage, evidence: e.nextEv, corroboration: e.nextCorr,
          manipulation: e.nextMan, conflict: e.nextConf, funding: fAfterDraw,
        };
        out.push({nextState: encode(sNext), prob: drawProb, reward: cost});
      }
    }
  }

  // Coalesce duplicates (same nextState gets prob summed; reward must
  // already be the same since we always emit `cost` as the per-step reward
  // for non-terminal transitions).
  const map = new Map<number, Outcome>();
  for (const o of out) {
    const cur = map.get(o.nextState);
    if (cur) cur.prob += o.prob;
    else map.set(o.nextState, {...o});
  }
  const coalesced = [...map.values()];

  // Sanity: probabilities sum to ~1.
  let p = 0; for (const o of coalesced) p += o.prob;
  if (Math.abs(p - 1) > 1e-9) {
    throw new Error(`outcomes(${stateId}, ${action}) probability sum ${p} != 1`);
  }
  return coalesced;
}

// -----------------------------------------------------------------------------
// Initial state distribution: how cases enter the system.
// -----------------------------------------------------------------------------

/**
 * Returns the starting state for a freshly-filed case under random "real
 * world" conditions. Most cases enter at SUB stage with messy partial info.
 */
export function sampleInitialState(rng: () => number): CaseState {
  const evRoll = rng();
  const evidence      = evRoll < 0.5 ? 0 : (evRoll < 0.85 ? 1 : 2);
  const corroboration = rng() < 0.6 ? 0 : (rng() < 0.5 ? 1 : 2);
  const manipulation  = rng() < 0.5 ? 0 : (rng() < 0.6 ? 1 : 2);
  const conflict      = rng() < 0.7 ? 0 : 1;
  const funding       = rng() < 0.5 ? FUND_UNFUNDED : FUND_ESCROWED;
  return {stage: 0, evidence, corroboration, manipulation, conflict, funding};
}
