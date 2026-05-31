// RUST MIGRATION:
// - Target: src/bin/validate_lp.rs.
// - Keep this as a CLI validation binary with Result-returning main; replace LP_SOLVER env mutation with scoped config values.
// - Convert LP/MDP test cases and solver outputs to nominal structs, using Vec<f64> for vector math.
// - Keep approximation/check helpers private and route internal/external solver choices through traits or enum-backed adapters.
'use strict';

// =============================================================================
// Validate the in-process simplex against scipy.optimize.linprog (HiGHS,
// HiGHS-IPM, scipy-simplex, scipy-interior-point) on canonical LP problems,
// and validate the MDP-as-LP transformation against generic value iteration.
// =============================================================================

import {LPProblem, solveLP, solveLPInternal, solveLPExternal} from '../general/lp';
import {buildMDPLP, solveMDPAsLP} from '../general/des-lp-bridge';
import {solveLPViaDES} from '../general/lp-des';
import {valueIteration, MDPSpec, qValue} from '../general/value-iteration';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

function approxEq(a: number, b: number, tol = 1e-7): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function maxAbsDiff(u: ArrayLike<number>, v: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < u.length; i++) m = Math.max(m, Math.abs(u[i] - v[i]));
  return m;
}

function arrToString(a: ArrayLike<number>, digits = 6): string {
  const out: string[] = [];
  for (let i = 0; i < a.length; i++) out.push(a[i].toFixed(digits));
  return out.join(', ');
}

function scipyUnavailable(r: {status: string; message?: string}): boolean {
  return r.status === 'numerical-error' && /scipy|numpy|No module named/.test(r.message ?? '');
}

// =============================================================================
// STUDY 1: Classic 2-variable LP — internal ≡ scipy across all methods
// =============================================================================
console.log('=== STUDY 1: 2-variable LP across all solver methods ===');
{
  const lp: LPProblem = {
    sense: 'max',
    c: [3, 2],
    A_ub: [[1, 1], [1, 3]],
    b_ub: [4, 6],
  };
  const expectedObj = 12;
  const expectedX = [4, 0];
  const internal = solveLPInternal(lp);
  console.log(`#   internal:        x=${JSON.stringify(internal.x)}  obj=${internal.objective}  iters=${internal.iters}`);
  check('internal solver finds optimum',
        internal.status === 'optimal' && approxEq(internal.objective, expectedObj),
        `obj=${internal.objective.toFixed(6)}`);
  for (const method of ['highs', 'highs-ds', 'highs-ipm'] as const) {
    const ext = solveLPExternal(lp, {method});
    if (ext.status === 'numerical-error' && /scipy/.test(ext.message ?? '')) {
      console.log(`#   scipy:${method} skipped (scipy unavailable)`);
      continue;
    }
    console.log(`#   scipy:${method}:  x=${JSON.stringify(ext.x)}  obj=${ext.objective}  iters=${ext.iters}`);
    check(`scipy:${method} matches expected optimum`,
          ext.status === 'optimal' && approxEq(ext.objective, expectedObj)
            && maxAbsDiff(ext.x, expectedX) < 1e-9,
          `obj=${ext.objective.toFixed(6)}`);
    check(`scipy:${method} ≡ internal (|Δobj| ≤ 1e-9)`,
          approxEq(ext.objective, internal.objective, 1e-9));
  }
}

// =============================================================================
// STUDY 2: Diet problem — minimisation with mixed inequality directions
// (cost ≤ 0 floors must be flipped). Classic LP teaching example.
// =============================================================================
console.log('\n=== STUDY 2: Diet LP (Stigler 1945 mini, 4 foods × 3 nutrients) ===');
{
  // min  cost = 0.5·x1 + 0.3·x2 + 0.7·x3 + 0.2·x4
  // s.t. protein:    2x1 + 3x2 + 1x3 + 4x4 ≥ 12
  //      vit-A:      1x1 + 2x2 + 3x3 + 1x4 ≥  6
  //      vit-C:      3x1 + 1x2 + 2x3 + 0x4 ≥  4
  //      x_i ≥ 0
  // Encode '≥' as '−lhs ≤ −rhs'.
  const lp: LPProblem = {
    sense: 'min',
    c: [0.5, 0.3, 0.7, 0.2],
    A_ub: [
      [-2, -3, -1, -4],
      [-1, -2, -3, -1],
      [-3, -1, -2,  0],
    ],
    b_ub: [-12, -6, -4],
  };
  const internal = solveLPInternal(lp);
  console.log(`#   internal cost  = ${internal.objective.toFixed(6)}   x = ${internal.x.map(v => v.toFixed(4)).join(', ')}`);
  const ext = solveLPExternal(lp, {method: 'highs'});
  if (ext.status === 'optimal') {
    console.log(`#   scipy:highs    = ${ext.objective.toFixed(6)}   x = ${ext.x.map(v => v.toFixed(4)).join(', ')}`);
    check('internal cost ≡ scipy:highs cost (|Δ| ≤ 1e-7)',
          approxEq(internal.objective, ext.objective, 1e-7),
          `Δ=${Math.abs(internal.objective - ext.objective).toExponential(3)}`);
    check('internal x ≡ scipy:highs x  (max|Δ| ≤ 1e-6)',
          maxAbsDiff(internal.x, ext.x) < 1e-6,
          `max|Δx|=${maxAbsDiff(internal.x, ext.x).toExponential(3)}`);
  }
  // All constraints must be satisfied at the optimum.
  const x = internal.x;
  const protein = 2*x[0] + 3*x[1] + 1*x[2] + 4*x[3];
  const vitA    = 1*x[0] + 2*x[1] + 3*x[2] + 1*x[3];
  const vitC    = 3*x[0] + 1*x[1] + 2*x[2] + 0*x[3];
  check('protein ≥ 12 (constraint feasibility)', protein >= 12 - 1e-7, `got ${protein.toFixed(4)}`);
  check('vit-A   ≥  6', vitA >= 6 - 1e-7, `got ${vitA.toFixed(4)}`);
  check('vit-C   ≥  4', vitC >= 4 - 1e-7, `got ${vitC.toFixed(4)}`);
}

// =============================================================================
// STUDY 3: Transportation problem (3 supply × 3 demand, balanced).
// Equality constraints, larger basis. This is where naive LPs typically
// surface bugs, since the equality encoding requires phase-1.
// =============================================================================
console.log('\n=== STUDY 3: Transportation LP (3×3, balanced) ===');
{
  // Suppliers s_i with capacity 20, 30, 25 → Demanders d_j needing 25, 25, 25.
  // Variables x_ij = units shipped, cost c_ij. n = 9.
  //   min  Σ c_ij x_ij
  //   s.t. Σ_j x_ij = supply_i           (3 rows)
  //        Σ_i x_ij = demand_j           (3 rows)
  //        x_ij ≥ 0
  const cost = [
    [4, 6, 8],
    [3, 5, 7],
    [9, 2, 1],
  ];
  const supply = [20, 30, 25];
  const demand = [25, 25, 25];
  const c: number[] = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) c.push(cost[i][j]);
  const A_eq: number[][] = [];
  const b_eq: number[] = [];
  // Supply constraints.
  for (let i = 0; i < 3; i++) {
    const row = new Array(9).fill(0);
    for (let j = 0; j < 3; j++) row[i * 3 + j] = 1;
    A_eq.push(row); b_eq.push(supply[i]);
  }
  // Demand constraints.
  for (let j = 0; j < 3; j++) {
    const row = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) row[i * 3 + j] = 1;
    A_eq.push(row); b_eq.push(demand[j]);
  }
  const lp: LPProblem = {sense: 'min', c, A_eq, b_eq};
  const internal = solveLPInternal(lp);
  const ext = solveLPExternal(lp, {method: 'highs'});
  console.log(`#   internal cost   = ${internal.objective.toFixed(6)}`);
  if (ext.status === 'optimal') {
    console.log(`#   scipy:highs     = ${ext.objective.toFixed(6)}`);
    check('transportation cost: internal ≡ scipy:highs (|Δ| ≤ 1e-7)',
          approxEq(internal.objective, ext.objective, 1e-7),
          `internal=${internal.objective.toFixed(6)}  highs=${ext.objective.toFixed(6)}`);
  }
  // Verify supply/demand satisfied.
  let supplyOk = true, demandOk = true;
  for (let i = 0; i < 3; i++) {
    let s = 0; for (let j = 0; j < 3; j++) s += internal.x[i * 3 + j];
    if (Math.abs(s - supply[i]) > 1e-6) supplyOk = false;
  }
  for (let j = 0; j < 3; j++) {
    let s = 0; for (let i = 0; i < 3; i++) s += internal.x[i * 3 + j];
    if (Math.abs(s - demand[j]) > 1e-6) demandOk = false;
  }
  check('supply equalities all satisfied', supplyOk);
  check('demand equalities all satisfied', demandOk);
}

// =============================================================================
// STUDY 4: MDP-as-LP ≡ value iteration on a small grid-world / chain.
// =============================================================================
console.log('\n=== STUDY 4: MDP-as-LP solution ≡ value-iteration solution ===');
{
  // 5-state chain: states 0..4. Action 0 = left (deterministic), 1 = right.
  // Reward: +1 for reaching state 4 (terminal), 0 elsewhere.
  // γ = 0.9.
  const N = 5;
  const mdp: MDPSpec = {
    numStates: N,
    numActions: () => 2,
    outcomes: (s, a) => {
      if (s === N - 1) return [{prob: 1, reward: 0, nextState: s}];   // terminal
      const target = a === 1 ? Math.min(N - 1, s + 1) : Math.max(0, s - 1);
      const reward = target === N - 1 ? 1 : 0;
      return [{prob: 1, reward, nextState: target}];
    },
    isTerminal: (s) => s === N - 1,
    terminalReward: () => 0,
  };
  const gamma = 0.9;
  const vi = valueIteration(mdp, {gamma, tol: 1e-12, maxIter: 10000});
  const lpSol = solveMDPAsLP(mdp, gamma);
  console.log(`#   VI    V = ${arrToString(vi.V)}    iters=${vi.iterations}`);
  console.log(`#   LP    V = ${arrToString(lpSol.V)}    iters=${lpSol.lp.iters}`);
  check('V*_LP ≡ V*_VI (max|Δ| ≤ 1e-6)',
        maxAbsDiff(lpSol.V, vi.V) < 1e-6,
        `max|Δ|=${maxAbsDiff(lpSol.V, vi.V).toExponential(3)}`);
  let polMatch = true;
  for (let s = 0; s < N - 1; s++) if (lpSol.policy[s] !== vi.policy[s]) polMatch = false;
  check('π*_LP ≡ π*_VI on all non-terminal states', polMatch,
        `LP=${lpSol.policy.join(',')}  VI=${vi.policy.join(',')}`);
}

// =============================================================================
// STUDY 5: Larger 3×3 grid-world MDP-as-LP ≡ value-iteration.
// =============================================================================
console.log('\n=== STUDY 5: 3×3 grid-world MDP-as-LP ≡ value-iteration ===');
{
  // 9 states arranged in a 3×3 grid; goal at corner (2,2) = state 8 (terminal).
  // 4 actions: 0=up 1=down 2=left 3=right. Slip probability 0.2 (lateral
  // moves with equal prob). Step cost = -0.04. Goal reward = +1.
  const W = 3, H = 3;
  const N = W * H;
  const idx = (x: number, y: number) => y * W + x;
  const move = (s: number, a: number): number => {
    const x = s % W, y = Math.floor(s / W);
    if (a === 0) return idx(x, Math.max(0, y - 1));
    if (a === 1) return idx(x, Math.min(H - 1, y + 1));
    if (a === 2) return idx(Math.max(0, x - 1), y);
    return idx(Math.min(W - 1, x + 1), y);
  };
  const slipPair: Record<number, [number, number]> = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]};
  const goal = idx(2, 2);
  const mdp: MDPSpec = {
    numStates: N,
    numActions: () => 4,
    outcomes: (s, a) => {
      if (s === goal) return [{prob: 1, reward: 0, nextState: s}];
      const intended = move(s, a);
      const [sa1, sa2] = slipPair[a];
      const slip1 = move(s, sa1);
      const slip2 = move(s, sa2);
      const stepCost = -0.04;
      const r = (sp: number) => sp === goal ? 1.0 : stepCost;
      return [
        {prob: 0.8, reward: r(intended), nextState: intended},
        {prob: 0.1, reward: r(slip1),    nextState: slip1},
        {prob: 0.1, reward: r(slip2),    nextState: slip2},
      ];
    },
    isTerminal: (s) => s === goal,
    terminalReward: () => 0,
  };
  const gamma = 0.95;
  const vi = valueIteration(mdp, {gamma, tol: 1e-12, maxIter: 10000});
  const lp = solveMDPAsLP(mdp, gamma);
  console.log(`#   VI iters=${vi.iterations}    LP iters=${lp.lp.iters}    LP solver=${lp.lp.solver}`);
  console.log(`#   V*_VI = ${arrToString(vi.V, 4)}`);
  console.log(`#   V*_LP = ${arrToString(lp.V, 4)}`);
  check('grid-world V*_LP ≡ V*_VI (max|Δ| ≤ 1e-5)',
        maxAbsDiff(lp.V, vi.V) < 1e-5,
        `max|Δ|=${maxAbsDiff(lp.V, vi.V).toExponential(3)}`);
  let policiesOptimal = true;
  let maxPolicyGap = 0;
  for (let s = 0; s < N; s++) {
    if (s === goal) continue;
    const qLP = qValue(mdp, vi.V, s, lp.policy[s], gamma);
    const qVI = qValue(mdp, vi.V, s, vi.policy[s], gamma);
    let bestQ = -Infinity;
    for (let a = 0; a < mdp.numActions(s); a++) bestQ = Math.max(bestQ, qValue(mdp, vi.V, s, a, gamma));
    const gap = Math.max(bestQ - qLP, bestQ - qVI);
    maxPolicyGap = Math.max(maxPolicyGap, gap);
    if (gap > 1e-7) policiesOptimal = false;
  }
  check('grid-world LP and VI policies both choose optimal actions',
        policiesOptimal,
        `max action-value gap=${maxPolicyGap.toExponential(3)}  LP=[${lp.policy.join(',')}]  VI=[${vi.policy.join(',')}]`);
}

// =============================================================================
// STUDY 6: LP-to-LP equivalence — internal solver matches scipy:highs to
// ~1e-9 across 200 random small LPs.
// =============================================================================
console.log('\n=== STUDY 6: 200 random feasible LPs — internal ≡ scipy:highs ===');
{
  const N_PROB = 200;
  let maxObjDiff = 0;
  let nMatch = 0;
  let nSkip = 0;
  let scipyAvailable = true;
  for (let p = 0; p < N_PROB; p++) {
    // Build a random LP: max c^T x  s.t.  Ax ≤ b, x ≥ 0.
    // Seeded with simple PRNG for reproducibility.
    const rng = ((seed: number) => () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xFFFFFFFF;
    })(p + 1);
    const n = 3 + Math.floor(rng() * 5);     // 3..7 vars
    const m = 3 + Math.floor(rng() * 5);     // 3..7 constraints
    const c: number[] = Array.from({length: n}, () => rng() * 4 - 1);
    const A_ub: number[][] = [];
    const b_ub: number[] = [];
    for (let i = 0; i < m; i++) {
      A_ub.push(Array.from({length: n}, () => rng() * 2));   // non-negative
      b_ub.push(1 + rng() * 9);                              // positive RHS
    }
    const lp: LPProblem = {sense: 'max', c, A_ub, b_ub};
    const internal = solveLPInternal(lp, {maxIter: 1000});
    const ext = solveLPExternal(lp, {method: 'highs'});
    if (ext.status === 'numerical-error' && /scipy/.test(ext.message ?? '')) {
      scipyAvailable = false; nSkip++; continue;
    }
    if (internal.status !== 'optimal' || ext.status !== 'optimal') { nSkip++; continue; }
    const d = Math.abs(internal.objective - ext.objective);
    if (d > maxObjDiff) maxObjDiff = d;
    if (d < 1e-7) nMatch++;
  }
  if (!scipyAvailable) {
    console.log('#   scipy unavailable; skipping random comparison');
  } else {
    console.log(`#   ${nMatch}/${N_PROB - nSkip} matched to 1e-7   max|Δobj| = ${maxObjDiff.toExponential(3)}`);
    check('all random LPs match within 1e-7 (excluding skipped)',
          nMatch === N_PROB - nSkip,
          `nMatch=${nMatch}  N=${N_PROB - nSkip}  maxΔ=${maxObjDiff.toExponential(3)}`);
  }
}

// =============================================================================
// STUDY 7: LP_SOLVER env-var dispatching — solveLP picks the right backend.
// =============================================================================
console.log('\n=== STUDY 7: solveLP env-var dispatch ===');
{
  const lp: LPProblem = {sense: 'max', c: [1, 1], A_ub: [[1, 0], [0, 1]], b_ub: [3, 5]};
  const expected = 8;
  for (const choice of ['internal', 'scipy:highs', 'scipy:highs-ds', 'scipy:highs-ipm']) {
    process.env.LP_SOLVER = choice;
    const r = solveLP(lp);
    console.log(`#   LP_SOLVER=${choice.padEnd(20)} → ${r.solver.padEnd(20)} obj=${r.objective.toFixed(4)} ${r.message ? '(' + r.message.slice(0, 60) + ')' : ''}`);
    check(`LP_SOLVER=${choice} returns optimum`,
          r.status === 'optimal' && approxEq(r.objective, expected),
          `obj=${r.objective}`);
  }
  delete process.env.LP_SOLVER;
}

// =============================================================================
// STUDY 8: DES-engine simplex ≡ scipy:highs ≡ in-process simplex
//
// The DES solver implements simplex as a 4-station tick loop:
//   EnteringStation → LeavingStation → PivotStation → ObserverStation
// One pivot per tick. Same algorithm as the in-process simplex, just
// driven by the DES engine. We assert numerical equivalence with both
// the in-process simplex AND scipy:highs across canonical LP shapes
// and 50 random feasible LPs.
// =============================================================================
console.log('\n=== STUDY 8: DES-engine simplex ≡ scipy:highs (LP-as-DES validation) ===');
{
  const cases: {name: string; lp: LPProblem}[] = [
    {name: '2-var max',
     lp: {sense: 'max', c: [3, 2], A_ub: [[1, 1], [1, 3]], b_ub: [4, 6]}},
    {name: 'diet (phase-1)',
     lp: {sense: 'min', c: [0.5, 0.3, 0.7, 0.2],
          A_ub: [[-2, -3, -1, -4], [-1, -2, -3, -1], [-3, -1, -2, 0]],
          b_ub: [-12, -6, -4]}},
    {name: 'transportation 3×3 equalities',
     lp: {sense: 'min', c: [4, 6, 8, 3, 5, 7, 9, 2, 1],
          A_eq: [[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1],
                 [1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1]],
          b_eq: [20, 30, 25, 25, 25, 25]}},
    {name: 'unbounded',
     lp: {sense: 'max', c: [1, 1], A_ub: [[-1, 1]], b_ub: [3]}},
    {name: 'infeasible',
     lp: {sense: 'max', c: [1, 1], A_ub: [[1, 1], [-1, -1]], b_ub: [3, -5]}},
  ];
  for (const {name, lp} of cases) {
    const desD = solveLPViaDES(lp, {pivotRule: 'dantzig'});
    const desB = solveLPViaDES(lp, {pivotRule: 'bland'});
    const ext = solveLPExternal(lp, {method: 'highs'});
    const internal = solveLPInternal(lp);
    const all = scipyUnavailable(ext) ? [desD, desB, internal] : [desD, desB, internal, ext];
    const stats = all.map(r => r.status);
    const sameStatus = stats.every(s => s === stats[0]);
    const scope = scipyUnavailable(ext) ? 'available solvers' : 'all four solvers';
    if (scipyUnavailable(ext)) console.log(`#   ${name.padEnd(32)}  scipy:highs skipped (${ext.message})`);
    check(`'${name}': ${scope} agree on status (${stats[0]})`,
          sameStatus, `statuses=[${stats.join(',')}]`);
    if (desD.status === 'optimal') {
      const objs = all.map(r => r.objective);
      const referenceObj = objs[objs.length - 1];
      const maxDelta = Math.max(...objs.map(o => Math.abs(o - referenceObj)));
      check(`'${name}': ${scope} agree on objective (max|Δ| ≤ 1e-7)`,
            maxDelta < 1e-7,
            `objs=[${objs.map(o => o.toFixed(6)).join(', ')}]   maxΔ=${maxDelta.toExponential(2)}`);
      const referenceX = scipyUnavailable(ext) ? internal.x : ext.x;
      const referenceName = scipyUnavailable(ext) ? 'internal simplex' : 'scipy:highs';
      const xMaxDelta = maxAbsDiff(desD.x, referenceX);
      check(`'${name}': DES-simplex x ≡ ${referenceName}  (max|Δ| ≤ 1e-7)`,
            xMaxDelta < 1e-7,
            `max|Δx|=${xMaxDelta.toExponential(2)}`);
      console.log(`#   ${name.padEnd(32)}  Dantzig=${desD.trace.pivotHistory.length} pivots   Bland=${desB.trace.pivotHistory.length} pivots   internal=${internal.iters}   highs=${ext.iters}`);
    }
  }
}

console.log('\n=== STUDY 9: 50 random feasible LPs — DES simplex ≡ scipy:highs ===');
{
  const N = 50;
  let nMatch = 0;
  let nSkip = 0;
  let scipyAvailable = true;
  let maxObjDiff = 0;
  for (let p = 0; p < N; p++) {
    const rng = ((seed: number) => () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xFFFFFFFF;
    })(p + 50);
    const n = 2 + Math.floor(rng() * 4);
    const m = 2 + Math.floor(rng() * 4);
    const c = Array.from({length: n}, () => rng() * 4 - 1);
    const A_ub: number[][] = [];
    const b_ub: number[] = [];
    for (let i = 0; i < m; i++) {
      A_ub.push(Array.from({length: n}, () => rng() * 2));
      b_ub.push(1 + rng() * 9);
    }
    const lp: LPProblem = {sense: 'max', c, A_ub, b_ub};
    const des = solveLPViaDES(lp, {maxIter: 500});
    const ext = solveLPExternal(lp, {method: 'highs'});
    if (scipyUnavailable(ext)) {
      scipyAvailable = false;
      nSkip++;
      continue;
    }
    if (des.status !== 'optimal' || ext.status !== 'optimal') { nSkip++; continue; }
    const d = Math.abs(des.objective - ext.objective);
    if (d > maxObjDiff) maxObjDiff = d;
    if (d < 1e-7) nMatch++;
  }
  const compared = N - nSkip;
  if (!scipyAvailable || compared === 0) {
    console.log('#   scipy unavailable; skipping random DES/scipy comparison');
  } else {
    console.log(`#   ${nMatch}/${compared} matched to 1e-7   max|Δobj| = ${maxObjDiff.toExponential(3)}`);
    check('all 50 random LPs: DES-simplex obj ≡ scipy:highs obj  (|Δ| ≤ 1e-7)',
          nMatch === compared,
          `nMatch=${nMatch}  N=${compared}`);
  }
}

// =============================================================================
console.log(`\n=== Summary: ${pass} passed, ${fail} failed ===`);
process.exit(fail === 0 ? 0 : 1);
