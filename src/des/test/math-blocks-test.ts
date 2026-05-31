// RUST MIGRATION: Port file-for-file to `tests/math_blocks_test.rs` for stationary math blocks, ODE block systems, and heat PDE grid coverage.
// Test-port notes: translate model scenarios into `#[test]` functions returning `Result<()>`; replace ad hoc checks with `assert!`, `assert_eq!`, and approximate-float helpers; use `tempfile` for file fixtures.

'use strict';

// =============================================================================
// Tests for stationary math blocks, ODE block systems, and heat PDE grids.
// =============================================================================

import * as fs from 'fs';
import {getModel, runFromSpec} from '../general/des-registry';
import {runMathEquationProblem} from '../general/math-equation-input';
import {
  ComparatorBlock,
  ConstantSourceBlock,
  DerivativeBlock,
  FirstOrderFilterBlock,
  FunctionSourceBlock,
  GainBlock,
  IntegratorBlock,
  LogicBlock,
  ProductBlock,
  SaturationBlock,
  SinkBlock,
  SubtractBlock,
  SumBlock,
  runHeat1DBlockGrid,
  runMathBlockDiagram,
  runODEBlockSystem,
} from '../general/math-blocks';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}

function close(a: number, b: number, tol = 1e-8): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

async function checkThrows(label: string, fn: () => unknown | Promise<unknown>, contains?: string): Promise<void> {
  let threw = false;
  let message = '';
  try {
    await fn();
  } catch (e) {
    threw = true;
    message = e instanceof Error ? e.message : String(e);
  }
  check(label, threw && (contains === undefined || message.includes(contains)), threw ? `message=${JSON.stringify(message)}` : 'did not throw');
}

async function main(): Promise<void> {
console.log('\n[1] Scalar math blocks as stationary DES stations');
{
  const opts = {dt: 1, ticks: 4};
  const a = new ConstantSourceBlock('a', 2, opts);
  const b = new ConstantSourceBlock('b', 5, opts);
  const sum = new SumBlock('sum', ['a', 'b'], opts);
  const sub = new SubtractBlock('sub', 'a', 'b', opts);
  const prod = new ProductBlock('prod', ['a', 'b'], opts);
  const gain = new GainBlock('gain', 10, opts);
  const sink = new SinkBlock('sink', opts);
  a.pipe(sum, 'out', 'a').pipe(sub, 'out', 'a').pipe(prod, 'out', 'a');
  b.pipe(sum, 'out', 'b').pipe(sub, 'out', 'b').pipe(prod, 'out', 'b');
  sum.pipe(gain, 'out', 'in');
  gain.pipe(sink, 'out', 'in');
  const r = runMathBlockDiagram([a, b, sum, sub, prod, gain, sink]);
  check('1.1 arithmetic validators pass', r.validation.every(c => c.passed));
  check('1.2 sum emits 2 + 5', sum.outputHistory.every(s => s.value === 7));
  check('1.3 subtraction emits 2 - 5', sub.outputHistory.every(s => s.value === -3));
  check('1.4 product emits 2 * 5', prod.outputHistory.every(s => s.value === 10));
  check('1.5 gain and sink receive routed signal', sink.received.length === 4 && sink.received.every(s => s.value === 70), `received=${sink.received.length}`);
}

console.log('\n[2] Integration, differentiation, filters, and logic');
{
  const iopts = {dt: 0.1, ticks: 11};
  const u = new ConstantSourceBlock('u', 2, iopts);
  const int = new IntegratorBlock('integral', 0, iopts);
  u.pipe(int, 'out', 'in');
  runMathBlockDiagram([u, int]);
  check('2.1 integrator accumulates constant input', close(int.outputHistory[10].value, 2, 1e-10), `final=${int.outputHistory[10].value}`);

  const dopts = {dt: 0.1, ticks: 8};
  const ramp = new FunctionSourceBlock('ramp', t => 3 * t, dopts);
  const deriv = new DerivativeBlock('derivative', dopts);
  ramp.pipe(deriv, 'out', 'in');
  runMathBlockDiagram([ramp, deriv]);
  check('2.2 differentiator recovers ramp slope after initial tick', deriv.outputHistory.slice(1).every(s => close(s.value, 3, 1e-9)));

  const fsrc = new ConstantSourceBlock('step', 10, iopts);
  const filt = new FirstOrderFilterBlock('filter', 0.5, 0, iopts);
  fsrc.pipe(filt, 'out', 'in');
  runMathBlockDiagram([fsrc, filt]);
  const fvals = filt.outputHistory.map(s => s.value);
  check('2.3 low-pass filter moves toward step without overshoot', fvals[fvals.length - 1] > fvals[0] && fvals.every(v => v >= 0 && v <= 10));

  const logicOpts = {dt: 1, ticks: 3};
  const high = new ConstantSourceBlock('high', 3, logicOpts);
  const low = new ConstantSourceBlock('low', -1, logicOpts);
  const sat = new SaturationBlock('sat', 0, 1, logicOpts);
  const cmp = new ComparatorBlock('cmp', 'gt', logicOpts, 'left', undefined, 2);
  const logic = new LogicBlock('logic', 'and', ['cmp', 'sat'], logicOpts);
  high.pipe(cmp, 'out', 'left');
  high.pipe(sat, 'out', 'in');
  cmp.pipe(logic, 'out', 'cmp');
  sat.pipe(logic, 'out', 'sat');
  low.pipe(new SinkBlock('low-sink', logicOpts), 'out', 'in');
  runMathBlockDiagram([high, low, sat, cmp, logic]);
  check('2.4 comparator and logic blocks emit boolean numerics', cmp.outputHistory.every(s => s.value === 1) && logic.outputHistory.every(s => s.value === 1));
}

console.log('\n[3] ODE block systems');
{
  const r = runODEBlockSystem({
    states: [{name: 'y', initial: 1, derivative: '-k*y'}],
    constants: {k: 1},
    t0: 0,
    t1: 1,
    dt: 0.01,
    method: 'euler',
  });
  check('3.1 ODE validators pass', r.validation.every(c => c.passed));
  check('3.2 exponential decay approximates exp(-1)', close(r.finalState.y, Math.exp(-1), 0.006), `final=${r.finalState.y}`);
  check('3.3 ODE block graph exposes integrator and RHS blocks', r.blockGraph.some(b => b.kind === 'integrator') && r.blockGraph.some(b => b.kind === 'expression'));

  const osc = runODEBlockSystem({
    states: [
      {name: 'x', initial: 1, derivative: 'v'},
      {name: 'v', initial: 0, derivative: '-x'},
    ],
    t0: 0,
    t1: 0.5,
    dt: 0.01,
  });
  check('3.4 multi-state feedback system stays finite', osc.validation.every(c => c.passed) && Number.isFinite(osc.finalState.x) && Number.isFinite(osc.finalState.v));
  await checkThrows('3.5 duplicate ODE state names are rejected', () => runODEBlockSystem({
    states: [
      {name: 'y', initial: 0, derivative: '1'},
      {name: 'y', initial: 1, derivative: '1'},
    ],
    t1: 1,
    dt: 0.1,
  }), 'unique');
}

console.log('\n[4] PDE heat equation block grid');
{
  const heat = runHeat1DBlockGrid({
    cells: 21,
    length: 1,
    alpha: 0.02,
    t0: 0,
    t1: 0.2,
    dt: 0.0025,
    initialExpression: 'sin(pi*x/length)',
    leftBoundary: 0,
    rightBoundary: 0,
  });
  check('4.1 heat-grid validators pass', heat.validation.every(c => c.passed));
  check('4.2 heat peak decays under zero boundary conditions', heat.trace[heat.trace.length - 1].max < heat.trace[0].max, `initial=${heat.trace[0].max} final=${heat.trace[heat.trace.length - 1].max}`);
  check('4.3 boundary source blocks remain pinned', close(heat.finalValues[0], 0) && close(heat.finalValues[heat.finalValues.length - 1], 0));
  check('4.4 heat graph uses cell and Laplacian stations', heat.blockGraph.filter(b => b.kind === 'laplacian-1d').length === 19);
  await checkThrows('4.5 unstable heat CFL is rejected', () => runHeat1DBlockGrid({
    cells: 11,
    length: 1,
    alpha: 1,
    t1: 1,
    dt: 0.1,
  }), 'stability');
}

console.log('\n[5] JSON registry, observability, and animation outputs');
{
  check('5.1 registry has math-ode-blocks', getModel('math-ode-blocks').id === 'math-ode-blocks');
  check('5.2 registry has math-heat1d-blocks', getModel('math-heat1d-blocks').id === 'math-heat1d-blocks');

  const logPath = 'out/math-blocks-test-ode.jsonl';
  const odeSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'math-ode-blocks',
    parameters: {
      states: [{name: 'y', initial: 1, derivative: '-y'}],
      t0: 0,
      t1: 0.2,
      dt: 0.05,
      method: 'euler',
    },
    runtime: {verbose: false, animate: true, outputs: {html: 'out/math-blocks-test-ode.html', log: logPath}},
  }, {verbose: false});
  check('5.3 JSON ODE emits default frames output', odeSummary.outputs.some(o => o.kind === 'frames' && o.path === 'out/math-blocks-test-ode.frames.jsonl'));
  check('5.4 JSON ODE observability log is written', fs.readFileSync(logPath, 'utf8').includes('"kind":"math-ode-run-finish"'));

  const heatSummary = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'math-heat1d-blocks',
    parameters: {
      cells: 9,
      length: 1,
      alpha: 0.01,
      t1: 0.05,
      dt: 0.005,
      initialExpression: 'sin(pi*x/length)',
      leftBoundary: 0,
      rightBoundary: 0,
    },
    runtime: {verbose: false, animate: false},
  }, {verbose: false});
  const heatResult = heatSummary.result as any;
  check('5.5 JSON heat model runs without animation when disabled', heatSummary.outputs.every(o => o.kind !== 'html') && heatResult.trace.length === 11);
}

console.log('\n[6] Equation ingestion to generated block networks');
{
  const latex = runMathEquationProblem({
    format: 'latex',
    kind: 'ode',
    equation: String.raw`\frac{dy}{dt} = -k y; y(0)=1`,
    constants: {k: 1},
    t1: 1,
    dt: 0.01,
  });
  check('6.1 LaTeX ODE input solves numerically', latex.kind === 'ode' && latex.ode !== undefined && close(latex.ode.finalState.y, Math.exp(-1), 0.006));
  check('6.2 LaTeX ODE generates stationary nodes and moving-signal edges', latex.network.nodes.length === 2 && latex.network.edges.length === 2);

  const json = runMathEquationProblem({
    format: 'json',
    kind: 'ode',
    ode: {
      states: [
        {name: 'x', initial: 1, derivative: 'v'},
        {name: 'v', initial: 0, derivative: '-x'},
      ],
      t1: 0.5,
      dt: 0.01,
    },
  });
  check('6.3 structured JSON ODE input builds dense feedback RHS graph', json.kind === 'ode' && json.network.nodes.length === 4 && json.network.edges.length === 6);

  const xml = runMathEquationProblem({
    format: 'xml',
    kind: 'heat1d',
    equation: '<heat1d cells="11" length="1" alpha="0.02" t1="0.1" dt="0.005" leftBoundary="0" rightBoundary="0"><initial>sin(pi*x/length)</initial></heat1d>',
  });
  check('6.4 XML heat PDE input solves on a generated grid', xml.kind === 'heat1d' && xml.heat1d !== undefined && xml.heat1d.trace[xml.heat1d.trace.length - 1].max < xml.heat1d.trace[0].max);
  check('6.5 XML heat PDE exposes cell/laplacian network edges', xml.network.nodes.length === 20 && xml.network.edges.length === 36, `nodes=${xml.network.nodes.length} edges=${xml.network.edges.length}`);

  const latexJson = await runFromSpec({
    $schema: 'des/model-spec/v1',
    model: 'math-equation',
    parameters: {
      format: 'latex',
      kind: 'ode',
      equation: String.raw`\frac{dy}{dt} = -k y; y(0)=1`,
      constants: {k: 1},
      t1: 0.2,
      dt: 0.05,
    },
    runtime: {verbose: false, animate: true, outputs: {html: 'out/math-equation-test-latex.html'}},
  }, {verbose: false});
  check('6.6 JSON runner writes animation for equation input by default', latexJson.outputs.some(o => o.kind === 'frames' && o.path === 'out/math-equation-test-latex.frames.jsonl'));
}

console.log(`\nmath-blocks-test summary: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
