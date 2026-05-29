#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// validate-ip-mip-external.ts
//
// Cross-checks the DES IP/MIP station graph against a sanctioned external
// source-only Python reference.  No solver binary is vendored in the repo.
// The external invocation goes through runners/external-program.ts and the
// module metadata in runners/external-modules.ts.
// =============================================================================

import './external-modules';
import * as fs from 'fs';
import * as path from 'path';
import {IP_MIP_REFERENCE_ID} from './external-modules';
import {repoRootFromRunner, runExternalModule} from './external-program';
import {
  buildBinaryKnapsackIP,
  IPMIPProblem,
  IPMIPSolution,
  solveIPMIPWithDES,
} from '../general/ip-mip-des';

interface CheckRow {name: string; passed: boolean; detail?: string}
interface ExternalPayload {
  result: {
    status: string;
    solver: string;
    x?: number[];
    objective?: number | null;
    message?: string;
    enumerated?: number;
  };
}

const ROOT = repoRootFromRunner();
const OUT_DIR = path.join(ROOT, 'out', 'external', 'ip-mip');
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function close(name: string, actual: number, expected: number, tol = 1e-8): void {
  const diff = Math.abs(actual - expected);
  check(name, diff <= tol, `actual=${actual} expected=${expected} diff=${diff.toExponential(3)} tol=${tol}`);
}

function writeProblem(name: string, problem: IPMIPProblem): string {
  fs.mkdirSync(OUT_DIR, {recursive: true});
  const p = path.join(OUT_DIR, `${name}-problem.json`);
  fs.writeFileSync(p, JSON.stringify(problem, null, 2));
  return p;
}

function runExternal(name: string, problem: IPMIPProblem, solver = 'brute-force'): ExternalPayload {
  const problemPath = writeProblem(name, problem);
  const out = path.join(OUT_DIR, `${name}-reference.json`);
  const ext = runExternalModule(IP_MIP_REFERENCE_ID, {
    problem: problemPath,
    out,
    solver,
    maxEnumerations: 2000000,
  });
  console.log(`  external command: ${ext.command} ${ext.args.map(a => JSON.stringify(a)).join(' ')}`);
  if (ext.stdout.trim()) console.log(`  external stdout: ${ext.stdout.trim()}`);
  if (ext.stderr.trim()) console.error(ext.stderr.trim());
  if (ext.status !== 0) throw new Error(`external IP/MIP reference exited with status ${ext.status}`);
  return JSON.parse(fs.readFileSync(out, 'utf8')) as ExternalPayload;
}

function compareScenario(name: string, problem: IPMIPProblem): void {
  console.log('');
  console.log(`-- ${name} --`);
  const internal = solveIPMIPWithDES(problem, {lpAlgorithm: 'incremental-primal-dual', maxCutRounds: 1});
  const external = runExternal(name, problem, 'brute-force');
  compare(name, problem, internal, external);
}

function compare(name: string, problem: IPMIPProblem, internal: IPMIPSolution, external: ExternalPayload): void {
  check(`${name}: external reference available`,
    external.result.status !== 'unavailable',
    external.result.message);
  check(`${name}: statuses agree optimal`,
    internal.status === 'optimal' && external.result.status === 'optimal',
    `internal=${internal.status} external=${external.result.status}`);
  if (external.result.status !== 'optimal' || external.result.objective === null || external.result.objective === undefined) return;
  close(`${name}: objective`, internal.z, external.result.objective);
  check(`${name}: internal incumbent feasible`, feasible(problem, internal.x), `x=[${internal.x.join(',')}]`);
  check(`${name}: external incumbent feasible`, feasible(problem, external.result.x ?? []), `x=[${external.result.x?.join(',') ?? ''}]`);
}

function feasible(p: IPMIPProblem, x: readonly number[], tol = 1e-8): boolean {
  if (x.length !== p.c.length) return false;
  for (let j = 0; j < x.length; j++) {
    if (x[j] < -tol) return false;
    const ub = p.ub?.[j];
    if (ub !== undefined && Number.isFinite(ub) && x[j] > ub + tol) return false;
    if (p.integerVars[j] && Math.abs(x[j] - Math.round(x[j])) > tol) return false;
  }
  for (let i = 0; i < p.A.length; i++) {
    let lhs = 0;
    for (let j = 0; j < x.length; j++) lhs += p.A[i][j] * x[j];
    if (lhs > p.b[i] + tol) return false;
  }
  return true;
}

function main(): void {
  console.log('IP/MIP DES: framework vs sanctioned external Python reference');
  console.log('=============================================================');

  compareScenario('knapsack-4item',
    buildBinaryKnapsackIP([10, 40, 30, 50], [5, 4, 6, 3], 10));
  compareScenario('cover-cut-lab',
    buildBinaryKnapsackIP([10, 10, 10], [2, 2, 2], 3));
  compareScenario('integer-bounded',
    {
      sense: 'max',
      c: [3, 5],
      A: [[2, 3]],
      b: [12],
      integerVars: [true, true],
      ub: [6, 6],
      varNames: ['a', 'b'],
      conNames: ['resource'],
    });

  console.log('');
  const passed = checks.filter(c => c.passed).length;
  console.log(`validate-ip-mip-external: ${passed}/${checks.length} checks passed.`);
  if (passed < checks.length) {
    console.log('FAILED:');
    for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
    process.exit(1);
  }
}

main();
