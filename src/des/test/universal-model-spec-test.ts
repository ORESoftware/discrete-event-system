// RUST MIGRATION: Port file-for-file to `tests/universal_model_spec_test.rs` as integration coverage for universal DES JSON documents and registry execution.
// Test-port notes: translate JSON/spec cases into `#[test]` functions returning `Result<()>`; use `assert!`, `assert_eq!`, approximate helpers where needed, and `tempfile`/serde fixtures.

'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/universal_model_spec_test.rs   (integration test crate)
// 1:1 file move. Tests the universal DES JSON document shape + conversions, so
// it is an integration test under `tests/`.
//
// Test harness → Rust:
//   ad-hoc check()/pass-fail counters + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual tally and the PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - JSON spec + runFromJsonFile -> serde + serde_json (#[derive(Serialize,
//     Deserialize)] on UniversalDESModelSpec).
//   - close(a,b,tol) relative float comparison -> approx::assert_relative_eq!.
//   - the `as const` literal-union fields ('latex'|'ode'|...) -> Rust enums.
//   - async main()/await -> a plain sync #[test].
// =============================================================================

// =============================================================================
// Tests for the universal DES JSON document shape.
// =============================================================================

import * as fs from 'fs';
import {runFromJsonFile, runFromSpec} from '../general/des-registry';
import {runMathEquationProblem} from '../general/math-equation-input';
import {
  UniversalDESModelSpec,
  universalFromMathEquationResult,
  universalToDESModelSpec,
  validateUniversalDESModelSpec,
} from '../general/universal-model-spec';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

async function main(): Promise<void> {
console.log('\n[1] Universal spec generated from equation input');
{
  const input = {
    format: 'latex' as const,
    kind: 'ode' as const,
    equation: String.raw`\frac{dy}{dt} = -k y; y(0)=1`,
    constants: {k: 1},
    t0: 0,
    t1: 1,
    dt: 0.01,
    method: 'euler' as const,
  };
  const equation = runMathEquationProblem(input);
  const universal = universalFromMathEquationResult(input, equation, {
    id: 'generated-latex-ode',
    description: 'Generated universal spec from a LaTeX ODE.',
  });
  const checks = validateUniversalDESModelSpec(universal);
  check('1.1 generated universal spec validates', checks.every(c => c.passed), checks.filter(c => !c.passed).map(c => c.name).join(', '));
  check('1.2 original input is preserved', universal.originalInput.format === 'latex' && universal.originalInput.content?.includes('\\frac') === true);
  check('1.3 math section declares ODE equation and initial condition', universal.math.kind === 'ode' && universal.math.equations.length === 1 && universal.math.initialConditions?.length === 1);
  check('1.4 DES graph declares stationary and moving entities', universal.des.stationaryEntities.length === 2 && universal.des.movingEntities.some(e => e.id === 'MathSignal'));
  check('1.5 graph edges reference the generated block network', universal.des.graph.edges.length === 2 && universal.des.graph.edges.every(e => e.movingEntity === 'MathSignal'));

  const runnable = universalToDESModelSpec(universal);
  const summary = await runFromSpec({...runnable, runtime: {verbose: false, animate: false}}, {verbose: false});
  const result = summary.result as any;
  check('1.6 universal spec converts back to runnable DES model', summary.modelId === 'math-equation' && close(result.ode.finalState.y, Math.exp(-1), 0.006));
}

console.log('\n[2] Canonical universal JSON example');
{
  const raw = JSON.parse(fs.readFileSync('examples/universal-math-equation-latex-ode.json', 'utf8')) as UniversalDESModelSpec;
  const checks = validateUniversalDESModelSpec(raw);
  check('2.1 checked-in universal example validates', checks.every(c => c.passed), checks.filter(c => !c.passed).map(c => c.name).join(', '));
  check('2.2 universal example has all four contract sections', !!raw.originalInput && !!raw.math && !!raw.des && !!raw.solver);
  const summary = await runFromJsonFile('examples/universal-math-equation-latex-ode.json', {verbose: false});
  const result = summary.result as any;
  check('2.3 runFromJsonFile accepts universal schema directly', summary.modelId === 'math-equation' && close(result.ode.finalState.y, Math.exp(-1), 0.006));
  check('2.4 universal example writes animation frames by default', summary.outputs.some(o => o.kind === 'frames' && o.path === 'out/universal-math-equation-latex-ode.frames.jsonl'));
}

console.log('\n[3] Hardening invalid graph references');
{
  const raw = JSON.parse(fs.readFileSync('examples/universal-math-equation-latex-ode.json', 'utf8')) as UniversalDESModelSpec;
  raw.des.graph.edges[0] = {
    ...raw.des.graph.edges[0],
    from: {entityId: 'missing-node', port: 'out'},
  };
  const checks = validateUniversalDESModelSpec(raw);
  check('3.1 invalid edge source is reported', checks.some(c => c.name === 'edge-from-ref/edge:state-to-rhs' && !c.passed));
}

console.log(`\nuniversal-model-spec-test summary: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
