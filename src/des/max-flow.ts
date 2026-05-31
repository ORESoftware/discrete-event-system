#!/usr/bin/env ts-node
// RUST MIGRATION: target src/des/max_flow.rs.
// RUST MIGRATION: Keep max-flow logic in this library module; if runnable behavior remains, add a thin src/bin/max_flow.rs wrapper.
// RUST MIGRATION: Map problem/solver shapes to structs/traits, keep the algorithmic core pure, and return Result for fallible setup.
// RUST MIGRATION: Use clap/std::env/PathBuf only at wrapper boundaries and keep JSON examples/config as serde-deserialized structs.
'use strict';

// =============================================================================
// max-flow.ts -- runnable maximum-flow optimiser expressed as a DES.
// =============================================================================

import {buildTextbookMaxFlowProblem, solveMaxFlow} from './general/max-flow';

function fmt(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(4);
}

function main(): void {
  const result = solveMaxFlow(buildTextbookMaxFlowProblem());

  console.log('# Maximum-flow optimiser as DES');
  console.log('# one augmenting path = one DES fixed-point tick');
  console.log(`# source=${result.source}, sink=${result.sink}, nodes=${result.numNodes}`);
  console.log(`# max flow = ${fmt(result.maxFlow)}`);
  console.log(`# augmentations = ${result.trace.length}, iterations = ${result.iterations}`);
  console.log('');

  console.log('## Augmenting-path trace');
  for (const t of result.trace) {
    console.log(`  iter ${String(t.iter).padStart(2)}: path ${t.path.join(' -> ')}  bottleneck=${fmt(t.bottleneck)}  flow=${fmt(t.flowAfter)}`);
  }
  console.log('');

  console.log('## Edge flows');
  for (const e of result.edgeFlows) {
    const name = e.name ? `${e.name} ` : '';
    console.log(`  ${name}${e.from} -> ${e.to}: ${fmt(e.flow)} / ${fmt(e.capacity)}`);
  }
  console.log('');

  console.log('## Min cut');
  console.log(`  S = {${result.minCut.sourceSide.join(', ')}}`);
  console.log(`  T = {${result.minCut.sinkSide.join(', ')}}`);
  console.log(`  capacity = ${fmt(result.minCut.capacity)}`);
}

if (require.main === module) {
  main();
}
