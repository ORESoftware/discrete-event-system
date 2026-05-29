#!/usr/bin/env ts-node
'use strict';

// Compares the framework's USACC MDP value iteration (out/court-mdp-framework.json)
// against the Python reference (out/external/court-mdp/python.json).
//
// HOW TO RUN
// ----------
//   npm run build
//   node dist/des/main-court-mdp.js                # writes V*, π* to out/court-mdp-framework.json
//   bash external-references/run-all.sh            # writes out/external/court-mdp/python.json
//   node dist/des/runners/validate-court-mdp.js
//
// Reports max-abs-error on V* and policy-disagreement count. Asserts both
// match within 1e-7 (absolute V difference) and 0 policy disagreements
// (since both implementations evaluate the same Bellman backup with the
// same coalesced transitions). The implementations were ported one-to-one
// from each other, so any disagreement is a bug.

import * as fs from 'fs';
import * as path from 'path';
import {ACTIONS, N_STATES, isTerminal, decode, STAGES, EVIDENCE, CORROBORATION, MANIPULATION} from '../mdp/usacc-mdp';

const ROOT = path.join(__dirname, '..', '..', '..');
const tsPath = path.join(ROOT, 'out', 'court-mdp-framework.json');
const pyPath = path.join(ROOT, 'out', 'external', 'court-mdp', 'python.json');

function loadJson(p: string): any {
  if (!fs.existsSync(p)) {
    console.error(`[validate-court-mdp] missing ${p}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function main() {
  const ts = loadJson(tsPath);
  const py = loadJson(pyPath);

  const Vts: number[] = ts.vi.V;
  const Vpy: number[] = py.V;
  const piTs: number[] = ts.vi.policy;
  const piPy: number[] = py.policy;

  console.log('USACC MDP: framework value iteration vs Python value iteration');
  console.log('==================================================================');
  console.log(`  γ = ${ts.vi.gamma}    framework iters = ${ts.vi.iterations}    python iters = ${py.iterations}`);
  console.log(`  framework final |ΔV| = ${ts.vi.finalDelta.toExponential(3)}    python = ${py.finalDelta.toExponential(3)}`);

  let maxV = 0; let maxAtState = -1;
  for (let s = 0; s < N_STATES; s++) {
    const d = Math.abs(Vts[s] - Vpy[s]);
    if (d > maxV) { maxV = d; maxAtState = s; }
  }
  let pDisagree = 0;
  let firstDisagreeState = -1;
  for (let s = 0; s < N_STATES; s++) {
    if (isTerminal(s)) continue;
    if (piTs[s] !== piPy[s]) {
      pDisagree++;
      if (firstDisagreeState < 0) firstDisagreeState = s;
    }
  }

  console.log(`  max |V_ts(s) - V_py(s)|       = ${maxV.toExponential(3)}  (at state ${maxAtState})`);
  console.log(`  policy disagreement count    = ${pDisagree} / ${N_STATES - 3}`);
  if (pDisagree > 0 && firstDisagreeState >= 0) {
    const cs = decode(firstDisagreeState)!;
    console.log(`    first disagree: state ${firstDisagreeState} = (${STAGES[cs.stage]}, ev=${EVIDENCE[cs.evidence]}, corr=${CORROBORATION[cs.corroboration]}, man=${MANIPULATION[cs.manipulation]}, conf=${cs.conflict ? 'HI' : 'LO'}, fund=${cs.funding})`);
    console.log(`      framework picks ${ACTIONS[piTs[firstDisagreeState]]}, python picks ${ACTIONS[piPy[firstDisagreeState]]}`);
  }

  // Per-policy aggregate sanity (just print, don't assert).
  console.log('');
  console.log('  Policy comparison (framework simulation, last run):');
  for (const r of ts.results) {
    const a = r.aggregates;
    console.log(`    ${r.policy.padEnd(18)}  meanReward=${a.meanReward.toFixed(2).padStart(8)}    accepted=${(a.fractionAccepted*100).toFixed(1).padStart(5)}%    closed=${(a.fractionClosed*100).toFixed(1).padStart(5)}%    exhausted=${(a.fractionExhausted*100).toFixed(1).padStart(5)}%`);
  }

  const tolV = 1e-7;
  const ok = maxV < tolV && pDisagree === 0;
  console.log('');
  console.log(`  max V diff < ${tolV.toExponential(0)}: ${maxV < tolV ? 'yes' : 'NO'}`);
  console.log(`  policies identical: ${pDisagree === 0 ? 'yes' : 'NO'}`);
  console.log(ok ? '  PASS' : '  FAIL');
  process.exit(ok ? 0 : 1);
}

main();
