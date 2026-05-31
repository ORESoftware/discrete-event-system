// RUST MIGRATION:
// - Target: src/des/runners/external_program.rs.
// - Keep this as the process-adapter module; interfaces become serde-friendly structs/enums plus an ExternalProgramRunner trait.
// - Replace spawnSync with std::process::Command or tokio::process::Command and return Result<ExternalProgramResult, ExternalProgramError>.
// - Use PathBuf canonicalization for repo-root/script guards and keep external parameter values explicit instead of structural Record types.
'use strict';

// =============================================================================
// Sanctioned external-program invocation helpers and module registry for
// validators / reference solvers.
//
// Rules:
//   - source scripts must live under external-references/
//   - no shell is used; arguments are passed as an argv array
//   - the interpreter is explicit (env-var override, stable default)
//   - stdout/stderr are captured for diagnostics
//   - binaries/interpreters are NEVER checked into this repo; only source
//     scripts and module metadata live here
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {spawnSync} from 'child_process';

export interface ExternalProgramResult {
  command: string;
  args: string[];
  status: number | null;
  stdout: string;
  stderr: string;
  moduleId?: string;
}

export type ExternalModuleKind = 'reference' | 'solver' | 'validator';
export type ExternalParamValue = string | number | boolean | undefined;
export type ExternalModuleParams = Record<string, ExternalParamValue>;

export interface ExternalInterpreterSpec {
  /** Environment variable that points to the interpreter/binary. */
  envVar: string;
  /** Fallback command resolved from PATH. */
  defaultCommand: string;
  /** Human label used in --list output. */
  label: string;
}

export interface ExternalModuleContext {
  root: string;
  outRoot: string;
  moduleOutDir: string;
}

export interface ExternalProgramModule {
  /** Stable id, e.g. "neural-network-reference". */
  id: string;
  kind: ExternalModuleKind;
  description: string;
  /** Source file under external-references/. */
  sourcePath: string;
  interpreter: ExternalInterpreterSpec;
  defaultParams?: ExternalModuleParams;
  timeoutMs?: number;
  maxBufferBytes?: number;
  buildArgs(params: ExternalModuleParams, ctx: ExternalModuleContext): string[];
}

const EXTERNAL_MODULES = new Map<string, ExternalProgramModule>();

export function repoRootFromRunner(): string {
  return path.join(__dirname, '..', '..', '..');
}

export function resolveExternalScript(root: string, relativeScript: string): string {
  const externalRoot = path.resolve(root, 'external-references');
  const script = path.resolve(root, relativeScript);
  if (!script.startsWith(externalRoot + path.sep)) {
    throw new Error(`external script must live under ${externalRoot}: ${script}`);
  }
  if (!fs.existsSync(script)) {
    throw new Error(`external script not found: ${script}`);
  }
  return script;
}

export function registerExternalModule(module: ExternalProgramModule): void {
  if (EXTERNAL_MODULES.has(module.id)) {
    throw new Error(`external module "${module.id}" already registered`);
  }
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(module.id)) {
    throw new Error(`invalid external module id "${module.id}"`);
  }
  // Validate source path at registration time when possible.
  resolveExternalScript(repoRootFromRunner(), module.sourcePath);
  EXTERNAL_MODULES.set(module.id, module);
}

export function getExternalModule(id: string): ExternalProgramModule {
  const module = EXTERNAL_MODULES.get(id);
  if (!module) {
    throw new Error(`unknown external module "${id}". Registered: [${listExternalModules().map(m => m.id).join(', ')}]`);
  }
  return module;
}

export function listExternalModules(): ExternalProgramModule[] {
  return [...EXTERNAL_MODULES.values()].sort((a, b) => a.id.localeCompare(b.id));
}

export function runExternalProgram(
  command: string,
  args: string[],
  opts: {
    cwd?: string;
    timeoutMs?: number;
    maxBufferBytes?: number;
    moduleId?: string;
  } = {},
): ExternalProgramResult {
  const r = spawnSync(command, args, {
    cwd: opts.cwd ?? repoRootFromRunner(),
    shell: false,
    encoding: 'utf8',
    timeout: opts.timeoutMs ?? Number(process.env.EXTERNAL_TIMEOUT_MS ?? 120_000),
    maxBuffer: opts.maxBufferBytes ?? 10 * 1024 * 1024,
  });
  if (r.error) {
    throw new Error(`failed to run ${command}: ${r.error.message}`);
  }
  return {
    command,
    args,
    status: r.status,
    stdout: r.stdout ?? '',
    stderr: r.stderr ?? '',
    moduleId: opts.moduleId,
  };
}

export function runExternalModule(
  id: string,
  params: ExternalModuleParams = {},
): ExternalProgramResult {
  const module = getExternalModule(id);
  const root = repoRootFromRunner();
  const script = resolveExternalScript(root, module.sourcePath);
  const outRoot = path.join(root, 'out', 'external');
  const moduleOutDir = path.join(outRoot, id.replace(/-reference$/, ''));
  const merged = {...(module.defaultParams ?? {}), ...params};
  const command = process.env[module.interpreter.envVar] ?? module.interpreter.defaultCommand;
  const args = [script, ...module.buildArgs(merged, {root, outRoot, moduleOutDir})];
  return runExternalProgram(command, args, {
    cwd: root,
    timeoutMs: module.timeoutMs,
    maxBufferBytes: module.maxBufferBytes,
    moduleId: id,
  });
}

/** Backward-compatible convenience for source-only Python references. */
export function runPythonReference(relativeScript: string, args: string[] = []): ExternalProgramResult {
  const root = repoRootFromRunner();
  const script = resolveExternalScript(root, relativeScript);
  const python = process.env.PYTHON_BIN ?? 'python3';
  return runExternalProgram(python, [script, ...args], {cwd: root});
}
