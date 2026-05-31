// RUST MIGRATION: target src/bin/main_observability_controllability.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// main-observability-controllability.ts — runnable demo of the general
// controllability / observability evaluator across linear state-space models,
// an MDP (reachability), and POMDPs (distinguishability).
//
//   npm run obs-ctrl
// =============================================================================

import {runIterativeDES} from './general/des-base/runner';
import {DESStation} from './general/des-base/station';
import {DcMotorDynamics} from './general/control-systems/dc-motor';
import {
  ControllabilityEvaluatorStation,
  EvaluationSinkStation,
  EvaluationToken,
  MarkovDecisionProcess,
  MdpControllabilityEvaluatorStation,
  MdpSourceStation,
  MdpToken,
  ObservabilityEvaluatorStation,
  PartiallyObservableProcess,
  PomdpObservabilityEvaluatorStation,
  PomdpSourceStation,
  PomdpToken,
  StateSpaceModel,
  StateSpaceSourceStation,
  StateSpaceToken,
} from './general/control-systems/observability-controllability';

class ObsCtrlDemo {
  run(): void {
    const ltiTokens = this.buildLtiModels();
    const mdpTokens = this.buildMdpModels();
    const pomdpTokens = this.buildPomdpModels();

    const ltiSource = new StateSpaceSourceStation('lti-source', ltiTokens);
    const mdpSource = new MdpSourceStation('mdp-source', mdpTokens);
    const pomdpSource = new PomdpSourceStation('pomdp-source', pomdpTokens);

    const ctrlEval = new ControllabilityEvaluatorStation('ctrl-eval');
    const obsEval = new ObservabilityEvaluatorStation('obs-eval');
    const mdpEval = new MdpControllabilityEvaluatorStation('mdp-eval');
    const pomdpEval = new PomdpObservabilityEvaluatorStation('pomdp-eval');
    const sink = new EvaluationSinkStation('sink');

    ltiSource.pipe(ctrlEval, 'model-lti', 'model-lti');
    ltiSource.pipe(obsEval, 'model-lti', 'model-lti');
    mdpSource.pipe(mdpEval, 'model-mdp', 'model-mdp');
    pomdpSource.pipe(pomdpEval, 'model-pomdp', 'model-pomdp');
    for (const ev of [ctrlEval, obsEval, mdpEval, pomdpEval] as DESStation[]) {
      ev.pipe(sink, 'evaluation', 'evaluation');
    }

    runIterativeDES(
      [ltiSource, mdpSource, pomdpSource, ctrlEval, obsEval, mdpEval, pomdpEval, sink],
      {shuffle: false, maxTicks: 10},
    );

    this.report(sink, ltiTokens, mdpTokens, pomdpTokens);
  }

  private buildLtiModels(): StateSpaceToken[] {
    const tokens: StateSpaceToken[] = [];
    // The query's worked example: double integrator. Both rank 2.
    tokens.push(new StateSpaceToken('double-integrator (query example)', new StateSpaceModel({
      A: [[0, 1], [0, 0]], B: [[0], [1]], C: [[1, 0]],
    })));
    // Diagonal plant, input only reaches x1 → uncontrollable; output only sees
    // x1 → unobservable. A clean "neither" example.
    tokens.push(new StateSpaceToken('decoupled modes (B,C reach one mode)', new StateSpaceModel({
      A: [[1, 0], [0, 2]], B: [[1], [0]], C: [[1, 0]],
    })));
    // Controllable but NOT observable: both modes driven, output sees only x1.
    tokens.push(new StateSpaceToken('controllable, not observable', new StateSpaceModel({
      A: [[1, 0], [0, 2]], B: [[1], [1]], C: [[1, 0]],
    })));
    // The DC motor (R,L,Ke,Kt,J,B) — physically controllable & observable.
    const motor = new DcMotorDynamics({
      resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002,
    }).stateSpace();
    tokens.push(new StateSpaceToken('DC motor (V → ω)', new StateSpaceModel(motor)));
    return tokens;
  }

  private buildMdpModels(): MdpToken[] {
    // 3-state controllable ring: action 0 advances s→s+1 (mod 3), so every
    // state reaches every state → strongly connected.
    const ring = new MarkovDecisionProcess({
      numStates: 3, numActions: 1,
      transition: [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],
    });
    // 3-state with an absorbing trap (state 2): once there you cannot leave →
    // not strongly connected → not structurally controllable.
    const trap = new MarkovDecisionProcess({
      numStates: 3, numActions: 1,
      transition: [[[0, 1, 0], [0, 0, 1], [0, 0, 1]]],
    });
    return [
      new MdpToken('ring MDP (strongly connected)', ring),
      new MdpToken('trap MDP (absorbing state 2)', trap),
    ];
  }

  private buildPomdpModels(): PomdpToken[] {
    // Observable: distinct observations per state (identity-like sensor).
    const distinct = new PartiallyObservableProcess({
      numStates: 2, numActions: 1,
      transition: [[[0.5, 0.5], [0.5, 0.5]]],
      numObservations: 2,
      observation: [[1, 0], [0, 1]],
    });
    // Aliased: both states emit the same observation distribution AND stay put
    // under the only action → never distinguishable → not observable.
    const aliased = new PartiallyObservableProcess({
      numStates: 2, numActions: 1,
      transition: [[[1, 0], [0, 1]]],
      numObservations: 2,
      observation: [[0.5, 0.5], [0.5, 0.5]],
    });
    return [
      new PomdpToken('distinct-sensor POMDP', distinct),
      new PomdpToken('aliased-sensor POMDP', aliased),
    ];
  }

  private report(
    sink: EvaluationSinkStation,
    lti: readonly StateSpaceToken[],
    mdp: readonly MdpToken[],
    pomdp: readonly PomdpToken[],
  ): void {
    console.log('\n================================================================================');
    console.log(' Observability & Controllability — general structural evaluator');
    console.log('================================================================================');

    console.log('\n LINEAR STATE-SPACE  (Kalman rank tests)');
    console.log(' --------------------------------------------------------------------------------');
    for (const t of lti) {
      const rs = sink.forLabel(t.label);
      const ctrl = rs.find(r => r.kind === 'controllability') as EvaluationToken;
      const obs = rs.find(r => r.kind === 'observability') as EvaluationToken;
      console.log(`   ${t.label}`);
      console.log(`      controllable : ${this.verdict(ctrl.full)}   (${ctrl.detail})`);
      console.log(`      observable   : ${this.verdict(obs.full)}   (${obs.detail})`);
    }

    console.log('\n MDP  (reachability ≈ controllability)');
    console.log(' --------------------------------------------------------------------------------');
    for (const t of mdp) {
      const r = sink.forLabel(t.label).find(x => x.kind === 'mdp-controllability') as EvaluationToken;
      console.log(`   ${t.label}`);
      console.log(`      controllable : ${this.verdict(r.full)}   (${r.detail})`);
    }

    console.log('\n POMDP  (distinguishability ≈ observability)');
    console.log(' --------------------------------------------------------------------------------');
    for (const t of pomdp) {
      const r = sink.forLabel(t.label).find(x => x.kind === 'pomdp-observability') as EvaluationToken;
      const aliasing = t.pomdp.indistinguishablePairs();
      console.log(`   ${t.label}`);
      console.log(`      observable   : ${this.verdict(r.full)}   (${r.detail})`);
      if (aliasing.length) console.log(`      aliased state pairs: ${aliasing.map(p => `(${p[0]},${p[1]})`).join(' ')}`);
    }
    console.log('\n================================================================================\n');
  }

  private verdict(ok: boolean): string {
    return ok ? 'YES' : 'NO ';
  }
}

new ObsCtrlDemo().run();
