'use strict';

// =============================================================================
// main-from-json.ts — Run any registered DES model from a JSON spec file.
//
// Usage:
//   node dist/des/main-from-json.js examples/temp-control-pid.json
//   node dist/des/main-from-json.js --list                            # show models
//   node dist/des/main-from-json.js --schema temp-control             # show param schema
//   node dist/des/main-from-json.js --example temp-control            # print example spec
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {runFromJsonFile, listModels, getModel} from './general/des-registry';

function printHelp(): void {
  console.log('Usage:');
  console.log('  node dist/des/main-from-json.js <path-to-spec.json>');
  console.log('  node dist/des/main-from-json.js --list');
  console.log('  node dist/des/main-from-json.js --schema   <model-id>');
  console.log('  node dist/des/main-from-json.js --example  <model-id>');
  console.log('');
  console.log('A spec file is a JSON object with at least:');
  console.log('  { "$schema": "des/model-spec/v1", "model": "<id>", "parameters": { ... } }');
  console.log('or a universal modeling document:');
  console.log('  { "$schema": "des/universal-model/v1", "originalInput": ..., "math": ..., "des": ... }');
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === '-h' || argv[0] === '--help') {
    printHelp();
    process.exit(argv.length === 0 ? 1 : 0);
  }

  if (argv[0] === '--list') {
    const models = listModels();
    console.log(`Registered models (${models.length}):`);
    for (const m of models) console.log(`  ${m.id.padEnd(24)} — ${m.description}`);
    return;
  }
  if (argv[0] === '--schema' && argv[1]) {
    const reg = getModel(argv[1]);
    console.log(`Model: ${reg.id}`);
    console.log(`Description: ${reg.description}`);
    console.log('Schema:');
    console.log(JSON.stringify(reg.schema, null, 2));
    return;
  }
  if (argv[0] === '--example' && argv[1]) {
    const reg = getModel(argv[1]);
    if (!reg.examples || reg.examples.length === 0) {
      console.error(`No examples registered for "${argv[1]}".`);
      process.exit(1);
    }
    console.log(JSON.stringify(reg.examples[0].spec, null, 2));
    return;
  }

  const specPath = argv[0];
  if (!fs.existsSync(specPath)) {
    console.error(`Spec file not found: ${specPath}`);
    process.exit(1);
  }
  const summary = await runFromJsonFile(specPath, {verbose: true});
  if (summary.outputs.length > 0) {
    console.log('');
    console.log('Outputs written:');
    for (const o of summary.outputs) console.log(`  [${o.kind}] ${o.path}`);
  }
  console.log('');
  console.log(`Total wall-clock time: ${summary.runtimeMs} ms`);
}

main().catch(e => {
  console.error('Error:', e instanceof Error ? e.message : e);
  process.exit(1);
});
