'use strict';

// =============================================================================
// RUST MIGRATION  —  target: tests/optimization_as_des_test.rs   (integration test crate)
// 1:1 file move. Tests the des-base optimizer/RL hierarchy and the four leaves
// (SA, GA, Q-learning, PPO). Keep the doc-block below; this header sits above it.
//
// Test harness → Rust:
//   ad-hoc check()/CheckRow + console.log  ->  #[test] fns using
//   assert!/assert_eq!; drop the manual rows and PASS/FAIL printing.
//
// Conversion notes (file-specific):
//   - SA/GA/Q-learning/PPO all use mulberry32 -> a seeded rand::Rng so the
//     learned policies / tours are reproducible across runs.
//   - throws(fn) -> assert on Result::Err / #[should_panic].
//   - objective/value comparisons -> approx::assert_relative_eq!.
// =============================================================================

// =============================================================================
// test/optimization-as-des-test.ts — unit tests for the des-base hierarchy
// (DESStation, runner, SingleStateOptimizer, PopulationOptimizer,
// RLAgentStation, PolicyGradientAgent, EnvironmentStation) AND the four
// concrete leaves (SA, GA, Q-learning, PPO).
// =============================================================================

import {
  DESStation, runIterativeDES, SingleStateOptimizer, PopulationOptimizer,
  SingleStateSinkStation, SingleStateSourceStation,
  PopulationSinkStation, PopulationSourceStation,
  RLAgentStation, PolicyGradientAgent, PolicyUpdateStation,
  EnvironmentStation, StateToken, ActionToken, TransitionToken,
  EpisodeAccounting, VectorEpisodeAccounting,
  Token, ChannelName,
} from '../general/des-base';
import {
  TSPSAOptimizer, TSPHillClimber, runTSPSADES, runTSPHillClimberDES,
} from '../general/sa-des';
import {TSPGAOptimizer, runTSPGADES} from '../general/ga-des';
import {QLearningAgent, runQLearningDES} from '../general/qlearning-des';
import {TabularPPOAgent, runPPODES} from '../general/ppo-des';
import {
  buildPentagonTSP, tourLength, isPermutation, buildRandomTSP, heldKarpExact,
} from '../general/genetic-tsp';
import {GridWorld, Corridor, evalPolicy} from '../general/rl-environments';
import {mulberry32} from '../general/prng';

interface CheckRow {name: string; passed: boolean; detail?: string}
const checks: CheckRow[] = [];
function check(name: string, passed: boolean, detail?: string): void {
  checks.push({name, passed, detail});
  console.log(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

function throws(fn: () => unknown): boolean {
  try {
    fn();
    return false;
  } catch (_e) {
    return true;
  }
}

// =============================================================================
// 1. DESStation channel mechanics
// =============================================================================

class DummyToken implements Token { constructor(public n: number) {} }

class CountingStation extends DESStation {
  received: number[] = [];
  override hasWork(): boolean { return this.inboxSizes()['default'] > 0 || this.inboxSizes()['x'] > 0; }
  runTimeStep(): void {
    for (const t of this.drainPub('default') as DummyToken[]) this.received.push(t.n);
    for (const t of this.drainPub('x') as DummyToken[]) this.received.push(100 + t.n);
  }
  drainPub(ch: ChannelName) { return (this as any).drain(ch); }
  emitPub(t: Token, ch: ChannelName) { (this as any).emit(t, ch); }
}

console.log('\n— DESStation channel mechanics —');
{
  const a = new CountingStation('a');
  const b = new CountingStation('b');
  a.pipe(b, 'default', 'default');
  a.pipe(b, 'x', 'x');
  // Manual emits to drive a.
  a.take(new DummyToken(1));
  a.take(new DummyToken(2), 'x');
  a.runTimeStep();
  check('drain consumes inbox', JSON.stringify(a.received) === '[1,102]');
  check('inbox empty after drain', !a.hasWork());

  // pipe routes to target with target-channel.
  a.emitPub(new DummyToken(7), 'default');
  a.emitPub(new DummyToken(8), 'x');
  const sizes = b['inboxes'] as Map<string, Token[]>;
  check('pipe routes default channel', (sizes.get('default')?.length ?? 0) === 1);
  check('pipe routes named channel', (sizes.get('x')?.length ?? 0) === 1);
}

// =============================================================================
// 2. runIterativeDES termination & shuffling
// =============================================================================

console.log('\n— runIterativeDES termination —');
{
  // Quiescent at tick 0 → done immediately.
  class IdleStation extends DESStation {
    override hasWork(): boolean { return false; }
    runTimeStep(): void {}
  }
  const s = new IdleStation('idle');
  const r = runIterativeDES([s]);
  check('quiescent → reason=done', r.reason === 'done');
  check('quiescent → ticks=0', r.ticks === 0);
}
{
  class CounterStation extends DESStation {
    public ticks = 0;
    override hasWork(): boolean { return this.ticks < 5; }
    runTimeStep(): void { this.ticks++; }
  }
  const s = new CounterStation('c');
  const r = runIterativeDES([s], {maxTicks: 10});
  check('runs until !hasWork', s.ticks === 5 && r.reason === 'done');
}
{
  class InfiniteStation extends DESStation {
    public ticks = 0;
    override hasWork(): boolean { return true; }
    runTimeStep(): void { this.ticks++; }
  }
  const s = new InfiniteStation('inf');
  const r = runIterativeDES([s], {maxTicks: 7});
  check('maxTicks honoured', s.ticks === 7 && r.reason === 'maxticks');
}
{
  class InfiniteStation extends DESStation {
    override hasWork(): boolean { return true; }
    runTimeStep(): void {}
  }
  const s = new InfiniteStation('s');
  let calls = 0;
  const r = runIterativeDES([s], {stopWhen: () => { calls++; return calls > 3; }});
  check('stopWhen fires before tick', r.reason === 'stop-when' && r.ticks === 3);
}

// =============================================================================
// 3. SingleStateOptimizer template-method behaviour
// =============================================================================
//
// Test by instantiating a TINY toy: minimise f(x) = (x - 7)^2 over integers,
// neighbour = ±1, accept = always (so it does a random walk; we verify only
// machinery: bootstrap, history monotonicity of `best`, finished flag).

class IntegerHillClimb extends SingleStateOptimizer<number> {
  constructor(seed: number, public maxIter: number, public target: number) {
    super('hc', {rng: mulberry32(seed)});
    this.bootstrap();
  }
  protected initialState(rng: () => number): number { return Math.floor(rng() * 100); }
  protected cost(s: number): number { return (s - this.target) ** 2; }
  protected propose(s: number, rng: () => number): number { return s + (rng() < 0.5 ? -1 : 1); }
  protected accept(_a: number, _b: number, ca: number, cb: number): boolean { return cb < ca; }
  protected clone(s: number): number { return s; }
  protected shouldStop(iter: number): boolean { return iter >= this.maxIter; }
}

console.log('\n— SingleStateOptimizer mechanics —');
{
  const opt = new IntegerHillClimb(7, 200, 7);
  runIterativeDES([opt], {rng: mulberry32(7)});
  check('hill-climber reaches optimum (cost=0)', opt.getBestCost() === 0);
  check('best-history monotone non-increasing',
    opt.bestHistory.every((x, i, a) => i === 0 || x <= a[i - 1]));
  check('finished flag set', opt.isFinished());
  check('ran exactly maxIter iterations', opt.getIteration() === 200);
}
{
  class SourceSeededHillClimb extends SingleStateOptimizer<number> {
    constructor(public maxIter: number, public target: number) {
      super('source-seeded-hc', {rng: mulberry32(11)});
    }
    protected initialState(): number { return 0; }
    protected cost(s: number): number { return (s - this.target) ** 2; }
    protected propose(s: number): number { return s > this.target ? s - 1 : s; }
    protected accept(_a: number, _b: number, ca: number, cb: number): boolean { return cb <= ca; }
    protected clone(s: number): number { return s; }
    protected shouldStop(iter: number): boolean { return iter >= this.maxIter; }
  }
  const source = new SingleStateSourceStation('hc-source', () => 10, s => {
    if (!Number.isFinite(s)) throw new Error('bad source seed');
  });
  const opt = new SourceSeededHillClimb(3, 7);
  const sink = new SingleStateSinkStation<number>('hc-sink');
  source.pipe(opt, SingleStateSourceStation.CH_INITIAL_STATE, SingleStateOptimizer.CH_INITIAL_STATE);
  opt.pipe(sink, SingleStateOptimizer.CH_RESULT, SingleStateSinkStation.CH_RESULT);
  runIterativeDES([source, opt, sink], {shuffle: false});
  check('single-state optimizer can be source-seeded', opt.isInitialized() && opt.getBest() === 7);
  check('single-state optimizer emits final state to sink', sink.latest?.snapshot.best === 7);

  const badSource = new SingleStateSourceStation('bad-hc-source', () => Number.NaN, s => {
    if (!Number.isFinite(s)) throw new Error('bad source seed');
  });
  const badOpt = new SourceSeededHillClimb(1, 0);
  badSource.pipe(badOpt, SingleStateSourceStation.CH_INITIAL_STATE, SingleStateOptimizer.CH_INITIAL_STATE);
  check('single-state source rejects invalid initial conditions',
    throws(() => runIterativeDES([badSource, badOpt], {shuffle: false})));
}
{
  // SA on pentagon TSP — exact optimum reachable.
  const inst = buildPentagonTSP(5, 50);
  const opt = tourLength(inst, [0, 1, 2, 3, 4]);
  const sa = runTSPSADES(inst, {
    cooling: {kind: 'geometric', T0: 50, alpha: 0.998}, maxIterations: 2000, seed: 1,
  });
  check('SA finds pentagon optimum', Math.abs(sa.bestCost - opt) < 1e-9);
  check('SA bestHistory monotone', sa.bestHistory.every((x, i, a) => i === 0 || x <= a[i - 1] + 1e-12));
  check('SA bestTour valid permutation', isPermutation(sa.bestTour, inst.n));
}
{
  // HC: finds local optimum quickly.
  const inst = buildPentagonTSP(5, 50);
  const opt = tourLength(inst, [0, 1, 2, 3, 4]);
  const hc = runTSPHillClimberDES(inst, {
    cooling: {kind: 'geometric', T0: 50, alpha: 0.998}, maxIterations: 2000, seed: 1,
  });
  check('HC reaches pentagon optimum (NN init)', Math.abs(hc.bestCost - opt) < 1e-9);
  check('HC accepted only strict improvements (acc == impr)',
    hc.acceptedCount === hc.improveCount);
}

// =============================================================================
// 4. PopulationOptimizer template-method behaviour
// =============================================================================

console.log('\n— PopulationOptimizer mechanics —');
{
  class TinyPopulation extends PopulationOptimizer<number> {
    constructor() { super('tiny-pop', {popSize: 2, rng: mulberry32(5)}); }
    protected initialPopulation(): number[] { return [4, 6]; }
    protected evaluate(x: number): number { return Math.abs(x); }
    protected select(pop: readonly number[]): number[] { return [pop[0]]; }
    protected recombine(parents: readonly number[]): number { return parents[0] - 1; }
    protected mutate(child: number): number { return child; }
    protected clone(x: number): number { return x; }
    protected shouldStop(generation: number): boolean { return generation >= 3; }
  }
  const source = new PopulationSourceStation<number>('tiny-pop-source', () => [3, 5], pop => {
    if (pop.length !== 2 || pop.some(x => !Number.isFinite(x))) throw new Error('bad population source');
  });
  const opt = new TinyPopulation();
  const sink = new PopulationSinkStation<number>('tiny-pop-sink');
  source.pipe(opt, PopulationSourceStation.CH_INITIAL_POPULATION, PopulationOptimizer.CH_INITIAL_POPULATION);
  opt.pipe(sink, PopulationOptimizer.CH_RESULT, PopulationSinkStation.CH_RESULT);
  runIterativeDES([source, opt, sink], {shuffle: false});
  check('population optimizer can be source-seeded', opt.isInitialized() && opt.getGeneration() === 3);
  check('population optimizer emits final population to sink', sink.latest?.snapshot.generation === 3);
}
{
  const inst = buildRandomTSP(8, 17);
  const exact = heldKarpExact(inst);
  const ga = runTSPGADES(inst, {popSize: 40, numGenerations: 80, seed: 1, init: 'nearest-neighbor', elitism: 2});
  check('GA finds good tour on n=8', ga.bestLength <= exact.length * 1.05);
  check('GA bestHistory monotone', ga.bestHistory.every((x, i, a) => i === 0 || x <= a[i - 1] + 1e-12));
  check('GA bestTour is valid permutation', isPermutation(ga.bestTour, inst.n));
  check('GA used N+1 history entries (init + N gens)', ga.bestHistory.length === ga.generations + 1);
}

// =============================================================================
// 5. RLAgent + Environment topology
// =============================================================================

console.log('\n— RL agent + environment topology —');
{
  const env = new GridWorld();
  const opt = env.optimalV(0.95);
  const ql = runQLearningDES(env, {
    numEpisodes: 500, alpha: 0.3, gamma: 0.95,
    epsilon: 0.8, epsilonDecay: 0.99, epsilonMin: 0.05,
    maxStepsPerEpisode: 50, seed: 1,
  });
  check('Q-learning matches optimal V(0)', Math.abs(Math.max(...ql.Q[0]) - opt.V[0]) < 0.01);
  const evalQ = evalPolicy(env, (s) => ql.policy[s], {numEpisodes: 100, maxStepsPerEpisode: 100, gamma: 0.95});
  check('Q-learning greedy policy 100% success', evalQ.successRate === 1);
  check('Q-learning total episodes == numEpisodes', ql.totalEpisodes === 500);
  check('Q-learning produced reward history', ql.rewardHistory.length === 500);
}

// =============================================================================
// 6. PolicyGradientAgent + PolicyUpdateStation
// =============================================================================

console.log('\n— PolicyGradientAgent rollout & update —');
{
  const cor = new Corridor(8);
  const opt = cor.optimalV(0.95);
  const ppo = runPPODES(cor, {
    totalSteps: 8000, rolloutLen: 64,
    numEpochs: 6, miniBatchSize: 16,
    policyLr: 0.05, valueLr: 0.1,
    gamma: 0.95, lambda: 0.95, clipEps: 0.2,
    entropyCoef: 0.01, maxStepsPerEpisode: 30, seed: 1,
  });
  check('PPO V(0) close to optimal', Math.abs(ppo.V[0] - opt.V[0]) < 0.05);
  check('PPO greedy policy moves right (action=1) on s=0', ppo.policy[0] === 1);
  const evalP = evalPolicy(cor, (s) => ppo.policy[s], {numEpisodes: 50, maxStepsPerEpisode: 30, gamma: 0.95});
  check('PPO greedy policy 100% success', evalP.successRate === 1);
  check('PPO performed multiple updates', ppo.totalUpdates >= 100);
  check('PPO totalSteps >= budget', ppo.totalSteps >= 8000);
}

// =============================================================================
// 7. EnvironmentStation token semantics
// =============================================================================

console.log('\n— EnvironmentStation semantics —');
{
  // Tiny env: 2 states, 2 actions; action 1 always goes to state 1 (terminal,
  // reward +1); action 0 stays. Episode ends when state=1.
  const env = {
    numStates: 2, numActions: 2,
    reset: () => 0,
    step: (s: number, a: number) => a === 1
      ? {nextState: 1, reward: 1, done: true}
      : {nextState: 0, reward: 0, done: false},
  };
  const sink = new (class extends DESStation {
    transitions: TransitionToken<number, number>[] = [];
    states: StateToken<number>[] = [];
    override hasWork(): boolean {
      return this.inboxSizes().state > 0 || this.inboxSizes().transition > 0;
    }
    runTimeStep(): void {
      for (const s of (this as any).drain('state') as StateToken<number>[]) this.states.push(s);
      for (const t of (this as any).drain('transition') as TransitionToken<number, number>[]) this.transitions.push(t);
      // Drive: always emit action=1 (which terminates) on the most-recent state.
      const fresh = this.states.slice(-1)[0];
      if (fresh) (this as any).emit(new ActionToken<number, number>(fresh.state, 1, fresh.episodeId), 'action');
      // Continue an in-progress episode (post-transition that wasn't done).
      const lastT = this.transitions.slice(-1)[0];
      if (lastT && !lastT.done) (this as any).emit(new ActionToken<number, number>(lastT.nextState, 1, lastT.episodeId), 'action');
    }
  })('sink');

  const eSt = new EnvironmentStation<number, number>('e', env, {numEpisodes: 3});
  eSt.pipe(sink, EnvironmentStation.CH_STATE, 'state');
  eSt.pipe(sink, EnvironmentStation.CH_TRANSITION, 'transition');
  sink.pipe(eSt, 'action', EnvironmentStation.CH_ACTION);

  runIterativeDES([eSt, sink], {maxTicks: 20});
  check('env emitted exactly 3 StateTokens (one per episode)', sink.states.length === 3);
  check('env emitted exactly 3 TransitionTokens (one terminal each)', sink.transitions.length === 3);
  check('all transitions are terminal', sink.transitions.every(t => t.done));
  check('env.totalSteps == 3', eSt.totalSteps === 3);
  check('env.rewardHistory has 3 ones', JSON.stringify(eSt.rewardHistory) === '[1,1,1]');
  check('env.lengthHistory has 3 ones', JSON.stringify(eSt.lengthHistory) === '[1,1,1]');
}

console.log('\n— Shared episode accounting helpers —');
{
  const scalar = new EpisodeAccounting();
  scalar.recordStep(2);
  scalar.recordStep(-0.5);
  const done = scalar.finishEpisode();
  check('scalar episode accounting records return and length',
    done.reward === 1.5 && done.length === 2
      && scalar.rewardHistory[0] === 1.5
      && scalar.lengthHistory[0] === 2
      && scalar.totalSteps === 2);

  const vector = new VectorEpisodeAccounting(2);
  vector.recordStep([1, -1]);
  vector.recordStep([0.5, 2]);
  const vDone = vector.finishEpisode(2);
  check('vector episode accounting records per-agent returns',
    JSON.stringify(vDone.rewards) === '[1.5,1]'
      && JSON.stringify(vector.rewardHistory) === '[[1.5,1]]'
      && vector.lengthHistory[0] === 2
      && vector.totalSteps === 2);
}

// =============================================================================
// SUMMARY
// =============================================================================

const passed = checks.filter(c => c.passed).length;
const failed = checks.length - passed;
console.log(`\n=== optimization-as-DES test summary: ${passed}/${checks.length} passed, ${failed} failed`);
if (failed > 0) {
  console.log('Failures:');
  for (const c of checks) if (!c.passed) console.log('  - ' + c.name + (c.detail ? ': ' + c.detail : ''));
  process.exit(1);
}
