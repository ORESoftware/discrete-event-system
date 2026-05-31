#!/usr/bin/env ts-node
// RUST MIGRATION: target src/bin/main_ip_mip_des.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// main-ip-mip-des.ts -- integer/MIP solver graph demo.
// =============================================================================

import {
  buildBinaryKnapsackIP,
  LPRelaxationAlgorithm,
  solveIPMIPWithDES,
} from './general/ip-mip-des';

function fmt(x: number, digits = 4): string {
  return Number.isFinite(x) ? x.toFixed(digits) : String(x);
}

function main(): void {
  const lpAlgorithm = (process.env.LP_ALGO ?? 'auto') as LPRelaxationAlgorithm;
  const problem = buildBinaryKnapsackIP(
    [10, 40, 30, 50],
    [5, 4, 6, 3],
    10,
  );
  const result = solveIPMIPWithDES(problem, {
    lpAlgorithm,
    allowExternalSolvers: process.env.ALLOW_EXTERNAL_SOLVERS === '1',
    maxNodes: Number(process.env.MAX_NODES ?? 200),
    maxCutRounds: Number(process.env.MAX_CUT_ROUNDS ?? 1),
    nodeSelection: (process.env.NODE_SELECTION ?? 'dfs') as 'dfs' | 'best-bound',
  });

  console.log('# IP/MIP solver graph as DES');
  console.log(`# LP backend:       ${result.lpAlgorithm}`);
  console.log(`# LP usage:         ${Object.entries(result.lpAlgorithmUsage).map(([k, v]) => `${k}=${v}`).join(', ')}`);
  console.log(`# root plan:        ${result.techniquePlan.rootLPAlgorithm}`);
  console.log(`# execution mode:   ${result.executionMode}`);
  console.log(`# in-house only:    ${result.inHouseOnly}`);
  console.log(`# status:           ${result.status}`);
  console.log(`# z*:               ${fmt(result.z)}`);
  console.log(`# best bound:       ${fmt(result.bestBound)}`);
  console.log(`# gap:              ${result.gap.toExponential(2)}`);
  console.log(`# x*:               [${result.x.map(v => fmt(v, 3)).join(', ')}]`);
  console.log(`# nodes explored:   ${result.nodesExplored}`);
  console.log(`# elapsed:          ${result.performance.elapsedMs} ms (${result.performance.nodesPerSecond.toFixed(2)} nodes/s)`);
  console.log(`# LP solves:        ${result.lpSolves}`);
  console.log(`# LP solver time:   ${result.performance.totalLPSolverMs} ms`);
  console.log(`# LP iterations:    ${result.totalLPIterations}`);
  console.log(`# cuts added:       ${result.cutsAdded}`);
  console.log(`# candidates tried: ${result.candidatesTried}`);
  console.log(`# solver tokens:    ${result.tokenStats.created} created (${result.tokenStats.stateful} stateful, ${result.tokenStats.stateless} stateless)`);
  console.log(`# incumbent source: ${result.incumbentSource ?? 'none'}`);
  console.log('');

  console.log('## Station graph');
  for (const n of result.topology) {
    console.log(`  ${n.id.padEnd(22)} ${n.role}`);
  }
  console.log('');

  console.log('## First trace events');
  for (const ev of result.trace.slice(0, 12)) {
    const z = ev.lpZ === null ? 'n/a' : fmt(ev.lpZ, 3);
    const frac = ev.fractional.length ? ` frac={${ev.fractional.join(',')}}` : '';
    const kids = ev.children ? ` children=[${ev.children.join(',')}]` : '';
    console.log(`  node=${ev.nodeId} d=${ev.depth} z=${z} ${ev.action}${frac}${kids}${ev.reason ? ` -- ${ev.reason}` : ''}`);
  }
}

if (require.main === module) {
  main();
}
