'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/runners/external_modules.rs
//                    (module des::runners::external_modules — hyphen → underscore)
// 1:1 file move. Metadata registry of the built-in external reference modules.
//
// Conversion notes (file-specific):
//   - Pure metadata + `registerBuiltInExternalModules()` mutating a global
//     registry; the `let registered = false` idempotency guard -> `std::sync::Once`
//     (or a `OnceLock<()>`), or build the table eagerly as a `static`.
//   - module-id string consts -> `const NEURAL_NETWORK_REFERENCE_ID: &str = ..`
//     (or an `enum ModuleId`).
//   - `path.join` -> `std::path::Path`/`PathBuf`.
//   - env-var interpreter lookups (`PYTHON_BIN`, …) -> `std::env::var`.
// =============================================================================

// =============================================================================
// Built-in external solver/validator modules.
//
// This file contains metadata only. It registers source scripts that live under
// external-references/ and describes how to invoke them. Interpreters are found
// from environment variables or PATH; no binaries are vendored in git.
// =============================================================================

import * as path from 'path';
import {
  ExternalModuleParams,
  ExternalParamValue,
  registerExternalModule,
} from './external-program';

export const NEURAL_NETWORK_REFERENCE_ID = 'neural-network-reference';
export const COMPUTER_NETWORK_REFERENCE_ID = 'computer-network-reference';
export const COMPUTER_NETWORK_FEL_REFERENCE_ID = 'computer-network-fel-reference';
export const IP_MIP_REFERENCE_ID = 'ip-mip-reference';
export const TRAFFIC_FEL_REFERENCE_ID = 'traffic-fel-reference';
export const TRAFFIC_SIMPY_REFERENCE_ID = 'traffic-simpy-reference';
export const TRAFFIC_CIW_REFERENCE_ID = 'traffic-ciw-reference';
export const TRAFFIC_SUMO_REFERENCE_ID = 'traffic-sumo-reference';

let registered = false;

export function registerBuiltInExternalModules(): void {
  if (registered) return;
  registered = true;

  registerExternalModule({
    id: NEURAL_NETWORK_REFERENCE_ID,
    kind: 'reference',
    description: 'Dependency-free Python reference for neural XOR, corridor value iteration, and neural ODE decay.',
    sourcePath: 'external-references/neural-network/nn_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    defaultParams: {
      seed: 7,
      xorEpochs: 8000,
      xorLr: 0.3,
      corridorLength: 6,
      corridorGamma: 0.95,
      odeRate: 0.5,
      odeY0: 1,
      odeT1: 2,
      odeDt: 0.05,
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'reference.json'));
      return [
        '--out', out,
        '--seed', numberParam(params.seed, 7),
        '--xor-epochs', numberParam(params.xorEpochs, 8000),
        '--xor-lr', numberParam(params.xorLr, 0.3),
        '--corridor-length', numberParam(params.corridorLength, 6),
        '--corridor-gamma', numberParam(params.corridorGamma, 0.95),
        '--ode-rate', numberParam(params.odeRate, 0.5),
        '--ode-y0', numberParam(params.odeY0, 1),
        '--ode-t1', numberParam(params.odeT1, 2),
        '--ode-dt', numberParam(params.odeDt, 0.05),
      ];
    },
  });

  registerExternalModule({
    id: COMPUTER_NETWORK_REFERENCE_ID,
    kind: 'validator',
    description: 'Dependency-free Python reference simulator for computer-network topology, queueing, drops, and bottleneck metrics.',
    sourcePath: 'external-references/computer-network/network_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    defaultParams: {
      builtin: 'bottleneck-lab',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'reference.json'));
      const args = ['--out', out];
      if (params.problem !== undefined) {
        args.push('--problem', stringParam(params.problem, ''));
      } else {
        args.push('--builtin', stringParam(params.builtin, 'bottleneck-lab'));
      }
      return args;
    },
  });

  registerExternalModule({
    id: COMPUTER_NETWORK_FEL_REFERENCE_ID,
    kind: 'validator',
    description: 'Dependency-free Python FEL-style packet-network reference; consumes the same computer-network JSON model spec as the internal registry.',
    sourcePath: 'external-references/computer-network/network_fel_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    defaultParams: {
      builtin: 'bottleneck-lab',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'fel-reference.json'));
      const args = ['--out', out];
      if (params.problem !== undefined) {
        args.push('--problem', stringParam(params.problem, ''));
      } else {
        args.push('--builtin', stringParam(params.builtin, 'bottleneck-lab'));
      }
      return args;
    },
  });

  registerExternalModule({
    id: IP_MIP_REFERENCE_ID,
    kind: 'solver',
    description: 'Source-only external IP/MIP reference: Python brute force for bounded integer models, optional scipy.optimize.milp when installed.',
    sourcePath: 'external-references/ip-mip/ip_mip_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    defaultParams: {
      solver: 'auto',
      maxEnumerations: 1000000,
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'reference.json'));
      return [
        '--problem', stringParam(params.problem, ''),
        '--out', out,
        '--solver', stringParam(params.solver, 'auto'),
        '--max-enumerations', numberParam(params.maxEnumerations, 1000000),
      ];
    },
  });

  registerExternalModule({
    id: TRAFFIC_SIMPY_REFERENCE_ID,
    kind: 'validator',
    description: 'Optional SimPy process-oriented traffic FEL reference for shared source/sink scheduled trips.',
    sourcePath: 'external-references/traffic/simpy_traffic_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'traffic-simpy-reference.json'));
      return [
        '--problem', stringParam(params.problem, ''),
        '--out', out,
      ];
    },
  });

  registerExternalModule({
    id: TRAFFIC_CIW_REFERENCE_ID,
    kind: 'validator',
    description: 'Optional Ciw queueing-network traffic FEL reference for shared source/sink scheduled trips.',
    sourcePath: 'external-references/traffic/ciw_traffic_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'traffic-ciw-reference.json'));
      return [
        '--problem', stringParam(params.problem, ''),
        '--out', out,
      ];
    },
  });

  registerExternalModule({
    id: TRAFFIC_FEL_REFERENCE_ID,
    kind: 'validator',
    description: 'Dependency-free Python Future Event List traffic reference for model-spec traffic flows and shared source/sink scheduled trips.',
    sourcePath: 'external-references/traffic/fel_traffic_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'traffic-fel-reference.json'));
      return [
        '--problem', stringParam(params.problem, ''),
        '--out', out,
      ];
    },
  });

  registerExternalModule({
    id: TRAFFIC_SUMO_REFERENCE_ID,
    kind: 'validator',
    description: 'Optional SUMO black-box traffic simulator cross-check; calls SUMO/netconvert from PATH or SUMO_BIN without vendoring binaries.',
    sourcePath: 'external-references/traffic/sumo_traffic_reference.py',
    interpreter: {
      envVar: 'PYTHON_BIN',
      defaultCommand: 'python3',
      label: 'Python 3',
    },
    buildArgs(params, ctx) {
      const out = stringParam(params.out, path.join(ctx.moduleOutDir, 'sumo-reference.json'));
      const args = [
        '--problem', stringParam(params.problem, ''),
        '--out', out,
      ];
      if (params.workdir !== undefined) args.push('--workdir', stringParam(params.workdir, ''));
      if (params.sumoBin !== undefined) args.push('--sumo-bin', stringParam(params.sumoBin, ''));
      if (params.netconvertBin !== undefined) args.push('--netconvert-bin', stringParam(params.netconvertBin, ''));
      if (params.collisionAction !== undefined) args.push('--collision-action', stringParam(params.collisionAction, 'warn'));
      return args;
    },
  });
}

function stringParam(v: ExternalParamValue, fallback: string): string {
  if (v === undefined) return fallback;
  return String(v);
}

function numberParam(v: ExternalParamValue, fallback: number): string {
  if (v === undefined) return String(fallback);
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`expected finite numeric external param, got ${String(v)}`);
  return String(n);
}

registerBuiltInExternalModules();
