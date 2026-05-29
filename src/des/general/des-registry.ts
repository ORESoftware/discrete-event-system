'use strict';

// =============================================================================
// general/des-registry.ts — runtime registry that maps model ids to
// runnable adapters, plus the runFromSpec() driver that JSON files (and
// the main-from-json.ts CLI) call into.
//
// Adding a model:
//   1. Build the model's pure run(...) function in general/<your-model>.ts.
//   2. Build an adapter in this file (or a sibling) that implements
//      DESModelRegistration<P, R>.
//   3. Call registerModel(adapter) at module load time.
//   4. Save a JSON example to examples/<your-model>.json.
//   5. Run with `node main-from-json.js examples/<your-model>.json`.
//
// Registered models so far:
//   • temp-control               : 24-hour HVAC control DES
//   • shortest-path              : graph SPP via DES wave propagation
//   • milp-bnb                   : MILP via branch-and-bound + IncrementalLP
//   • simulated-annealing        : SA on TSP / knapsack (extensible)
//   • internal-solver-network    : GA/SA/knapsack/SP/TSP DES solver networks
//   • feasibility-pipeline       : candidate feasibility + improvement pipeline
//   • collaborative-inference    : sparse subjective ratings/rankings to global rank
//
// More can be wrapped over time without changing this file's design.
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {
  DESModelSpec, DESModelRegistration, DESRuntimeConfig, DESRunSummary,
  ParamSchema, validate,
} from './des-spec';
import {isUniversalDESModelSpec, universalToDESModelSpec} from './universal-model-spec';
import {defaultFramesPath} from './adapters/adapter-utils';

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

const REGISTRY: Map<string, DESModelRegistration<any, any>> = new Map();

export function registerModel<P, R>(reg: DESModelRegistration<P, R>): void {
  if (REGISTRY.has(reg.id)) throw new Error(`model "${reg.id}" already registered`);
  REGISTRY.set(reg.id, reg);
}

export function getModel(id: string): DESModelRegistration<any, any> {
  const reg = REGISTRY.get(id);
  if (!reg) throw new Error(`unknown model "${id}". Registered: [${[...REGISTRY.keys()].join(', ')}]`);
  return reg;
}

export function listModels(): Array<{id: string; description: string}> {
  return [...REGISTRY.values()].map(r => ({id: r.id, description: r.description}));
}

// -----------------------------------------------------------------------------
// Spec runner — the central driver for JSON-driven runs.
// -----------------------------------------------------------------------------

export interface RunFromSpecOptions {
  /** If true, log progress to stderr. Defaults to spec.runtime?.verbose ?? true. */
  verbose?: boolean;
}

/**
 * Validate, run, render, and (optionally) animate a DESModelSpec.
 * Returns a DESRunSummary describing what happened.
 *
 * Side effects: writes any files configured under spec.runtime.outputs.
 */
export async function runFromSpec(
  spec: DESModelSpec<any>,
  opts: RunFromSpecOptions = {},
): Promise<DESRunSummary> {
  if (spec.$schema !== 'des/model-spec/v1') {
    throw new Error(`unknown $schema "${spec.$schema}". Expected "des/model-spec/v1".`);
  }
  const reg = getModel(spec.model);
  const runtime: DESRuntimeConfig = spec.runtime ?? {};
  const outCfg: NonNullable<DESRuntimeConfig['outputs']> = {...(runtime.outputs ?? {})};
  if (reg.animate && runtime.animate !== false) {
    if (!outCfg.html) outCfg.html = path.join('out', `${spec.model}.html`);
    if (!outCfg.frames) outCfg.frames = defaultFramesPath(outCfg.html);
  }
  const runtimeForRun: DESRuntimeConfig = {...runtime, outputs: outCfg};
  const verbose = opts.verbose ?? runtime.verbose ?? true;

  // Validate parameters. Newer adapters can provide a Zod schema for richer
  // JSON validation; older adapters continue to use the lightweight ParamSchema.
  const params = validateModelParameters(spec.model, spec.parameters, reg);

  if (verbose) console.error(`[runFromSpec] model="${spec.model}"  description=${JSON.stringify(spec.description ?? reg.description)}`);

  // Execute.
  const t0 = Date.now();
  const result = await reg.run(params, runtimeForRun);
  const runtimeMs = Date.now() - t0;

  // Summarise.
  const summaryText = reg.summarize(result, params);
  if (verbose) {
    console.error(`[runFromSpec] completed in ${runtimeMs} ms`);
    console.error('');
    console.error(summaryText);
  }

  // Write outputs.
  const outputs: DESRunSummary['outputs'] = [];
  if (outCfg.csv && reg.writeCsv) {
    fs.mkdirSync(path.dirname(outCfg.csv), {recursive: true});
    reg.writeCsv(result, outCfg.csv);
    outputs.push({kind: 'csv', path: outCfg.csv});
    if (verbose) console.error(`[runFromSpec] wrote CSV: ${outCfg.csv}`);
  }
  if ((outCfg.html || outCfg.frames) && reg.animate && runtime.animate !== false) {
    await reg.animate(result, params, runtimeForRun);
    if (outCfg.html) outputs.push({kind: 'html', path: outCfg.html});
    if (outCfg.frames) outputs.push({kind: 'frames', path: outCfg.frames});
    if (verbose && outCfg.html) console.error(`[runFromSpec] wrote HTML: ${outCfg.html}`);
  }
  if (outCfg.log && fs.existsSync(outCfg.log)) {
    outputs.push({kind: 'log', path: outCfg.log});
    if (verbose) console.error(`[runFromSpec] wrote log: ${outCfg.log}`);
  }
  if (outCfg.summary) {
    fs.mkdirSync(path.dirname(outCfg.summary), {recursive: true});
    const payload = {
      modelId: spec.model,
      params,
      runtimeMs,
      summaryText,
      result: serialiseResult(result),
    };
    fs.writeFileSync(outCfg.summary, JSON.stringify(payload, null, 2));
    outputs.push({kind: 'summary', path: outCfg.summary});
    if (verbose) console.error(`[runFromSpec] wrote summary: ${outCfg.summary}`);
  }

  return {modelId: spec.model, params, runtimeMs, result, summaryText, outputs};
}

function validateModelParameters<P>(
  modelId: string,
  value: unknown,
  reg: DESModelRegistration<P, unknown>,
): P {
  if (reg.zodSchema) {
    const parsed = reg.zodSchema.safeParse(value);
    if (!parsed.success) {
      const errors = parsed.error.issues.map(issue => {
        const pathText = issue.path.length > 0 ? '$.' + issue.path.join('.') : '$';
        return `${pathText}: ${issue.message}`;
      });
      throw new Error(`invalid parameters for model "${modelId}":\n  ${errors.join('\n  ')}`);
    }
    return parsed.data;
  }
  const v = validate(value, reg.schema);
  if (!v.valid) {
    throw new Error(`invalid parameters for model "${modelId}":\n  ${v.errors.join('\n  ')}`);
  }
  return v.value as P;
}

/** Strip non-serialisable fields for JSON dumps. */
function serialiseResult(r: unknown): unknown {
  if (r === null || r === undefined) return r;
  if (Array.isArray(r)) return r.map(serialiseResult);
  if (typeof r === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(r as Record<string, unknown>)) {
      if (typeof v === 'function') continue;
      // Skip giant arrays
      if (Array.isArray(v) && v.length > 1000 && typeof v[0] === 'number') {
        out[k] = `<array length=${v.length} (omitted from summary)>`;
        continue;
      }
      out[k] = serialiseResult(v);
    }
    return out;
  }
  return r;
}

// -----------------------------------------------------------------------------
// Convenience: load a JSON file and run it.
// -----------------------------------------------------------------------------

export async function runFromJsonFile(specPath: string, opts: RunFromSpecOptions = {}): Promise<DESRunSummary> {
  const text = fs.readFileSync(specPath, 'utf8');
  const parsed = JSON.parse(text) as unknown;
  const spec = isUniversalDESModelSpec(parsed)
    ? universalToDESModelSpec(parsed)
    : parsed as DESModelSpec<unknown>;
  return runFromSpec(spec, opts);
}

// -----------------------------------------------------------------------------
// Auto-register the built-in models. Each registration lives in its own
// adapter file to keep this registry file small and to make it obvious
// where to add new ones.
// -----------------------------------------------------------------------------

import './adapters/temp-control-adapter';
import './adapters/shortest-path-adapter';
import './adapters/milp-bnb-adapter';
import './adapters/simulated-annealing-adapter';
import './adapters/network-flow-adapter';
import './adapters/stochastic-flow-mdp-adapter';
import './adapters/computer-network-adapter';
import './adapters/mdp-adjacent-adapters';
import './adapters/optimal-control-adapters';
import './adapters/statistical-optimization-adapter';
import './adapters/multistage-sddp-adapter';
import './adapters/math-blocks-adapter';
import './adapters/internal-solver-network-adapter';
import './adapters/feasibility-pipeline-adapter';
import './adapters/neural-network-adapters';
import './adapters/learning-optimization-adapter';
import './adapters/collaborative-inference-adapter';
import './adapters/classical-optimization-adapter';
import './adapters/nonlinear-optimization-adapter';
import './adapters/nonlinear-forecasting-adapter';
import './adapters/advanced-optimization-control-adapter';
import './adapters/signal-transforms-adapter';
import './adapters/domain-application-adapter';

export {ParamSchema};
