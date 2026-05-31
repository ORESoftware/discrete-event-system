'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/external_module_test.rs   (integration test crate)
// 1:1 file move. Tests the external solver/validator module registry +
// script resolution, so it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/throws()/pass-fail counters + console.log  ->  #[test] fns
//   using assert!/assert_eq!; drop the manual tally and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - fs/os/path + repo-root resolution -> std::fs / std::path::Path.
//   - runExternalModule shells out -> std::process::Command.
//   - throws(fn) -> assert on Result::Err (or #[should_panic]).
// =============================================================================

// =============================================================================
// test/external-module-test.ts — external solver/validator module registry.
// =============================================================================

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  listExternalModules,
  resolveExternalScript,
  repoRootFromRunner,
  runExternalModule,
} from '../runners/external-program';
import {
  COMPUTER_NETWORK_FEL_REFERENCE_ID,
  NEURAL_NETWORK_REFERENCE_ID,
  TRAFFIC_CIW_REFERENCE_ID,
  TRAFFIC_FEL_REFERENCE_ID,
  TRAFFIC_SIMPY_REFERENCE_ID,
  TRAFFIC_SUMO_REFERENCE_ID,
} from '../runners/external-modules';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];

function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ' - ' + detail : ''}`);
}

function throws(fn: () => void): boolean {
  try { fn(); return false; } catch { return true; }
}

console.log('\n-- external module registry --');
{
  const modules = listExternalModules();
  check('neural-network-reference is registered',
        modules.some(m => m.id === NEURAL_NETWORK_REFERENCE_ID));
  check('traffic-sumo-reference is registered',
        modules.some(m => m.id === TRAFFIC_SUMO_REFERENCE_ID));
  check('computer-network-fel-reference is registered',
        modules.some(m => m.id === COMPUTER_NETWORK_FEL_REFERENCE_ID));
  check('traffic-fel-reference is registered',
        modules.some(m => m.id === TRAFFIC_FEL_REFERENCE_ID));
  check('traffic-simpy-reference is registered',
        modules.some(m => m.id === TRAFFIC_SIMPY_REFERENCE_ID));
  check('traffic-ciw-reference is registered',
        modules.some(m => m.id === TRAFFIC_CIW_REFERENCE_ID));
  const mod = modules.find(m => m.id === NEURAL_NETWORK_REFERENCE_ID)!;
  check('registered module uses PYTHON_BIN interpreter override',
        mod.interpreter.envVar === 'PYTHON_BIN');
  check('registered source path lives under external-references',
        mod.sourcePath.startsWith('external-references/'));
}

console.log('\n-- external source path guard --');
{
  const root = repoRootFromRunner();
  check('valid external source resolves',
        fs.existsSync(resolveExternalScript(root, 'external-references/neural-network/nn_reference.py')));
  check('path outside external-references is rejected',
        throws(() => resolveExternalScript(root, 'src/des/main-neural-net.ts')));
}

console.log('\n-- run source-only external module --');
{
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'des-ext-module-'));
  const out = path.join(tmpDir, 'neural-reference.json');
  const r = runExternalModule(NEURAL_NETWORK_REFERENCE_ID, {
    out,
    xorEpochs: 12,
    seed: 7,
  });
  check('external module exits 0', r.status === 0, `status=${r.status}`);
  check('external module reports module id', r.moduleId === NEURAL_NETWORK_REFERENCE_ID);
  check('external module writes output JSON', fs.existsSync(out));
  const payload = JSON.parse(fs.readFileSync(out, 'utf8'));
  check('output has xor/corridor/ode sections',
        !!payload.xor && !!payload.corridor && !!payload.neuralOdeDecay);
  check('short XOR run used requested epochs',
        payload.xor.lossHistory.length === 12 * 4,
        `len=${payload.xor.lossHistory.length}`);
  check('corridor reference has right-moving nonterminal policy',
        JSON.stringify(payload.corridor.policy.slice(0, 5)) === '[1,1,1,1,1]');

  const trafficProblem = path.join(tmpDir, 'traffic-problem.json');
  const trafficOut = path.join(tmpDir, 'traffic-sumo-reference.json');
  fs.writeFileSync(trafficProblem, JSON.stringify({
    schema: 'des/external-traffic-sumo/v1',
    params: {durationSec: 10, dtSec: 1, carLengthM: 4.8, minGapM: 2.5, laneWidthM: 3.7, maxAccelMps2: 2.2, maxDecelMps2: 4},
    network: {
      nodes: [
        {id: 'src', kind: 'source', x: 0, y: 0},
        {id: 'sink', kind: 'sink', x: 1, y: 0},
      ],
      lanes: [{id: 'src-sink', from: 'src', to: 'sink', lengthM: 50, speedLimitMps: 13.5}],
      sources: [{id: 'src', nodeId: 'src', ratePerMin: 12, destinationSinkIds: ['sink']}],
      sinks: [{id: 'sink', nodeId: 'sink'}],
    },
    demand: [{id: 'src-sink', sourceId: 'src', sinkId: 'sink', route: ['src-sink'], vehicles: 2, beginSec: 0, endSec: 10}],
  }, null, 2));
  const traffic = runExternalModule(TRAFFIC_SUMO_REFERENCE_ID, {problem: trafficProblem, out: trafficOut});
  check('SUMO traffic module exits 0 even when simulator is optional', traffic.status === 0, `status=${traffic.status}`);
  check('SUMO traffic module writes JSON payload', fs.existsSync(trafficOut));
  const trafficPayload = JSON.parse(fs.readFileSync(trafficOut, 'utf8'));
  check('SUMO traffic payload reports known status',
        ['ok', 'unavailable', 'error'].includes(trafficPayload.status),
        `status=${trafficPayload.status}`);

  const trafficFelProblem = path.join(tmpDir, 'traffic-fel-problem.json');
  const trafficFelOut = path.join(tmpDir, 'traffic-fel-reference.json');
  fs.writeFileSync(trafficFelProblem, JSON.stringify({
    $schema: 'des/model-spec/v1',
    model: 'smart-traffic-flow',
    parameters: {
      durationSec: 30,
      dtSec: 1,
      seed: 1,
      maxCars: 20,
      smartCarPoolSize: 20,
      spawnRateMultiplier: 0,
      network: {
        nodes: [
          {id: 'src', kind: 'source', x: 0, y: 0},
          {id: 'sink', kind: 'sink', x: 1, y: 0},
        ],
        lanes: [{id: 'src-sink', from: 'src', to: 'sink', lengthM: 40, speedLimitMps: 10}],
        sources: [{id: 'src', nodeId: 'src', ratePerMin: 0, destinationSinkIds: ['sink']}],
        sinks: [{id: 'sink', nodeId: 'sink'}],
      },
      scheduledTrips: [
        {departSec: 0, sourceId: 'src', destinationSinkId: 'sink'},
        {departSec: 5, sourceId: 'src', destinationSinkId: 'sink'},
      ],
    },
  }, null, 2));
  const trafficFel = runExternalModule(TRAFFIC_FEL_REFERENCE_ID, {problem: trafficFelProblem, out: trafficFelOut});
  check('traffic FEL module exits 0', trafficFel.status === 0, `status=${trafficFel.status}`);
  check('traffic FEL module writes JSON payload', fs.existsSync(trafficFelOut));
  const trafficFelPayload = JSON.parse(fs.readFileSync(trafficFelOut, 'utf8'));
  check('traffic FEL payload is ok', trafficFelPayload.status === 'ok', `status=${trafficFelPayload.status}`);
  check('traffic FEL consumed scheduled trips',
        trafficFelPayload.result.generatedDemand === 2 && trafficFelPayload.result.entered === 2,
        JSON.stringify(trafficFelPayload.result));

  const simpyOut = path.join(tmpDir, 'traffic-simpy-reference.json');
  const simpy = runExternalModule(TRAFFIC_SIMPY_REFERENCE_ID, {problem: trafficProblem, out: simpyOut});
  check('SimPy traffic module exits 0 even when package is optional', simpy.status === 0, `status=${simpy.status}`);
  check('SimPy traffic module writes JSON payload', fs.existsSync(simpyOut));
  const simpyPayload = JSON.parse(fs.readFileSync(simpyOut, 'utf8'));
  check('SimPy traffic payload reports known status',
        ['ok', 'unavailable', 'error'].includes(simpyPayload.status),
        `status=${simpyPayload.status}`);

  const ciwOut = path.join(tmpDir, 'traffic-ciw-reference.json');
  const ciw = runExternalModule(TRAFFIC_CIW_REFERENCE_ID, {problem: trafficProblem, out: ciwOut});
  check('Ciw traffic module exits 0 even when package is optional', ciw.status === 0, `status=${ciw.status}`);
  check('Ciw traffic module writes JSON payload', fs.existsSync(ciwOut));
  const ciwPayload = JSON.parse(fs.readFileSync(ciwOut, 'utf8'));
  check('Ciw traffic payload reports known status',
        ['ok', 'unavailable', 'error'].includes(ciwPayload.status),
        `status=${ciwPayload.status}`);
}

console.log('\n========================================');
const passed = checks.filter(c => c.passed).length;
console.log(`external-module-test: ${passed}/${checks.length} checks passed.`);
if (passed < checks.length) {
  console.log('FAILED:');
  for (const c of checks) if (!c.passed) console.log(`  - ${c.name}${c.detail ? ': ' + c.detail : ''}`);
  process.exit(1);
}
