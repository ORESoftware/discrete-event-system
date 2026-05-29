#!/usr/bin/env ts-node
'use strict';

// =============================================================================
// main-stochastic-flow-mdp.ts -- MDP interpretation of stochastic max-flow.
// =============================================================================

import {
  buildDefaultStochasticFlowMDPProblem,
  solveStochasticFlowMDP,
} from './general/stochastic-flow-mdp';

function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : 'n/a';
}

function main(): void {
  const seed = Number(process.env.SEED ?? 7);
  const result = solveStochasticFlowMDP(buildDefaultStochasticFlowMDPProblem(), {seed, maxPolicyRows: 16});

  console.log('# Stochastic flow control MDP');
  console.log('# state=(current node, remaining capacities), action=edge attempt or wait');
  console.log(`# horizon=${result.horizon}, states=${result.numStates}`);
  console.log(`# deterministic max-flow upper bound=${result.deterministicMaxFlow}`);
  console.log(`# optimal expected reward=${fmt(result.expectedReward)}`);
  console.log(`# simulated delivered units (seed=${seed})=${result.simulation.delivered}`);
  console.log(`# simulated total reward=${fmt(result.simulation.totalReward)}`);
  console.log('');

  console.log('## Initial-state policy path (success branch)');
  for (const row of result.initialPolicy) {
    console.log(`  t=${row.stage}: node=${row.state.node}, caps=[${row.state.capacities.join(',')}] -> ${row.action.label}  V=${fmt(row.value)}`);
  }
  console.log('');

  console.log('## Simulated trajectory');
  for (const step of result.simulation.steps) {
    const ok = step.action.kind === 'wait' ? 'wait' : (step.success ? 'success' : 'fail');
    console.log(`  t=${step.stage}: ${step.nodeBefore} --${step.action.label}/${ok}--> ${step.nodeAfter}  r=${fmt(step.reward)}  delivered=${step.deliveredSoFar}`);
  }
}

if (require.main === module) {
  main();
}
