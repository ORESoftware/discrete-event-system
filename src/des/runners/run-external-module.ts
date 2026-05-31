#!/usr/bin/env ts-node
// RUST MIGRATION:
// - Target: src/bin/run_external_module.rs.
// - Keep this as the external-adapter CLI with Result-returning main; replace process.argv parsing with clap.
// - Preserve JSON-like key=value parameter parsing as serde_json::Value or typed params at the boundary.
// - Route process execution through the migrated external_program adapter using std::process or tokio::process.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/run-external-module.rs  (a `fn main` binary)
// 1:1 file move. CLI front-end for invoking sanctioned external modules.
//
// Conversion notes (file-specific):
//   - CLI entry point: top-level code becomes `fn main()`.
//   - `process.argv` parsing (`--list`, `<module-id>`, `--key=value`) ->
//     `std::env::args()` (or a `clap` parser).
//   - param value union `string | number | boolean` -> `enum ParamValue { Str,
//     Num, Bool }`.
//   - `throw new Error(..)` on bad args -> return `Result`/exit non-zero (user
//     error, not a panic).
//   - `JSON` output + `console.log` -> `serde_json` + `println!`.
// =============================================================================

// =============================================================================
// CLI for sanctioned external modules.
//
// Usage:
//   ts-node src/des/runners/run-external-module.ts --list
//   ts-node src/des/runners/run-external-module.ts neural-network-reference
//   ts-node src/des/runners/run-external-module.ts neural-network-reference --seed=11 --out out/external/neural-network/reference.json
// =============================================================================

import './external-modules';
import {ExternalModuleParams, listExternalModules, runExternalModule} from './external-program';

function printHelp(): void {
  console.log('Usage:');
  console.log('  ts-node src/des/runners/run-external-module.ts --list');
  console.log('  ts-node src/des/runners/run-external-module.ts <module-id> [--key=value ...]');
  console.log('');
  console.log('External module invocations are shell-free and source paths must live under external-references/.');
}

function parseParams(args: string[]): ExternalModuleParams {
  const out: ExternalModuleParams = {};
  for (const arg of args) {
    if (!arg.startsWith('--')) throw new Error(`unexpected argument "${arg}"`);
    const eq = arg.indexOf('=');
    if (eq < 0) throw new Error(`expected --key=value, got "${arg}"`);
    const key = arg.slice(2, eq);
    const raw = arg.slice(eq + 1);
    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(key)) throw new Error(`invalid param key "${key}"`);
    out[key] = parseValue(raw);
  }
  return out;
}

function parseValue(raw: string): string | number | boolean {
  if (raw === 'true') return true;
  if (raw === 'false') return false;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(raw)) return Number(raw);
  return raw;
}

function main(): void {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === '-h' || argv[0] === '--help') {
    printHelp();
    process.exit(argv.length === 0 ? 1 : 0);
  }

  if (argv[0] === '--list') {
    const modules = listExternalModules();
    console.log(`External modules (${modules.length}):`);
    for (const m of modules) {
      const env = m.interpreter.envVar;
      const cmd = process.env[env] ?? m.interpreter.defaultCommand;
      console.log(`  ${m.id.padEnd(30)} ${m.kind.padEnd(10)} ${m.interpreter.label} via ${env}=${cmd}`);
      console.log(`    ${m.description}`);
    }
    return;
  }

  const id = argv[0];
  const params = parseParams(argv.slice(1));
  const r = runExternalModule(id, params);
  console.log(`external module: ${id}`);
  console.log(`command: ${r.command} ${r.args.map(a => JSON.stringify(a)).join(' ')}`);
  if (r.stdout.trim()) console.log(r.stdout.trim());
  if (r.stderr.trim()) console.error(r.stderr.trim());
  process.exit(r.status ?? 1);
}

main();
