'use strict';

// =============================================================================
// control-systems/observability-controllability.ts — a general evaluator for
// the two structural properties of dynamical systems:
//
//   CONTROLLABILITY — can an input drive the state anywhere?
//   OBSERVABILITY   — can the output reveal the full internal state?
//
// THREE LENSES
// ────────────
//   1. LINEAR STATE-SPACE  ẋ = Ax + Bu,  y = Cx + Du   (`StateSpaceModel`)
//        Controllable ⟺ rank[ B  AB  A²B … Aⁿ⁻¹B ] = n   (Kalman rank test)
//        Observable   ⟺ rank[ C; CA; CA²; …; CAⁿ⁻¹ ] = n
//
//   2. MDP (fully observed)  (`MarkovDecisionProcess`)
//        The MDP analog of controllability is REACHABILITY: from every state,
//        can some action sequence reach every other state? That is exactly the
//        transitive closure of the "∃ action a with P_a(s→s') > 0" graph being
//        complete — the discrete cousin of the controllability matrix spanning
//        the whole state space.
//
//   3. POMDP (partially observed)  (`PartiallyObservableProcess`)
//        The POMDP analog of observability is STATE DISTINGUISHABILITY: can the
//        observation stream eventually separate every pair of states? We answer
//        it with partition refinement — start by splitting states on their
//        observation distribution, then iteratively split blocks whose members
//        transition (under some action) to different blocks. Reaching all-
//        singletons is the discrete cousin of the observability matrix having
//        full rank (using outputs over time, not just the instantaneous C).
//
// Everything here is expressed as CLASSES with METHODS (LinAlg/MatrixRank do
// the numeric linear algebra), and the DES pipeline below wires sources →
// `PureTransformEntity` evaluators → sink so the structural verdicts flow as
// tokens on named channels.
// =============================================================================

import {ChannelName, DESStation, Token} from '../des-base/station';
import {PureTransformEntity} from '../des-base/transform-entity';
import {Preconditions} from '../des-base/preconditions';
import {LinAlg, Mat, MatrixRank} from './linear-algebra';

// =============================================================================
// 1. LINEAR STATE-SPACE MODEL
// =============================================================================

export interface StateSpaceSpec {
  /** state matrix A (n×n) */
  A: Mat;
  /** input matrix B (n×m) */
  B: Mat;
  /** output matrix C (p×n) */
  C: Mat;
  /** feedthrough D (p×m). Default zero. */
  D?: Mat;
}

/** A linear time-invariant model with the Kalman controllability/observability
 *  tests as methods. */
export class StateSpaceModel {
  readonly A: Mat;
  readonly B: Mat;
  readonly C: Mat;
  readonly D: Mat;

  constructor(spec: StateSpaceSpec) {
    const cls = 'StateSpaceModel';
    Preconditions.squareMatrix(cls, 'A', spec.A);
    const n = spec.A.length;
    Preconditions.rectangularMatrix(cls, 'B', spec.B);
    Preconditions.lengthEq(cls, 'B', spec.B, n);
    Preconditions.rectangularMatrix(cls, 'C', spec.C);
    Preconditions.lengthEq(cls, 'C[0]', spec.C[0], n);
    this.A = LinAlg.copy(spec.A);
    this.B = LinAlg.copy(spec.B);
    this.C = LinAlg.copy(spec.C);
    this.D = spec.D ? LinAlg.copy(spec.D) : LinAlg.zeros(spec.C.length, spec.B[0].length);
  }

  /** State dimension n. */
  stateDim(): number {
    return this.A.length;
  }

  /** Input dimension m. */
  inputDim(): number {
    return LinAlg.cols(this.B);
  }

  /** Output dimension p. */
  outputDim(): number {
    return this.C.length;
  }

  /** Controllability matrix 𝒞 = [ B  AB  A²B … Aⁿ⁻¹B ]  (n × n·m). */
  controllabilityMatrix(): Mat {
    const n = this.stateDim();
    const blocks: Mat[] = [];
    let aPowerB = this.B;            // A⁰B = B
    blocks.push(aPowerB);
    for (let k = 1; k < n; k++) {
      aPowerB = LinAlg.matMul(this.A, aPowerB);   // Aᵏ⁻¹B → AᵏB
      blocks.push(aPowerB);
    }
    return LinAlg.hstack(blocks);
  }

  /** Observability matrix 𝒪 = [ C; CA; CA²; …; CAⁿ⁻¹ ]  (n·p × n). */
  observabilityMatrix(): Mat {
    const n = this.stateDim();
    const blocks: Mat[] = [];
    let cAPower = this.C;            // C·A⁰ = C
    blocks.push(cAPower);
    for (let k = 1; k < n; k++) {
      cAPower = LinAlg.matMul(cAPower, this.A);    // C·Aᵏ⁻¹ → C·Aᵏ
      blocks.push(cAPower);
    }
    return LinAlg.vstack(blocks);
  }

  controllabilityRank(): number {
    return LinAlg.rank(this.controllabilityMatrix());
  }

  observabilityRank(): number {
    return LinAlg.rank(this.observabilityMatrix());
  }

  isControllable(): boolean {
    return new MatrixRank(this.controllabilityMatrix()).isFullRank(this.stateDim());
  }

  isObservable(): boolean {
    return new MatrixRank(this.observabilityMatrix()).isFullRank(this.stateDim());
  }
}

// =============================================================================
// 2. MARKOV DECISION PROCESS — reachability ("controllability")
// =============================================================================

export interface MdpSpec {
  /** number of states S */
  numStates: number;
  /** number of actions A */
  numActions: number;
  /** transition[a][s][s'] = P(s' | s, a). Each transition[a][s] is a pmf. */
  transition: number[][][];
}

/** A finite MDP whose structural-controllability test is reachability of the
 *  controlled transition graph (transitive closure). */
export class MarkovDecisionProcess {
  readonly numStates: number;
  readonly numActions: number;
  readonly transition: number[][][];

  constructor(spec: MdpSpec) {
    const cls = 'MarkovDecisionProcess';
    Preconditions.integerInRange(cls, 'numStates', spec.numStates, 1, 100_000);
    Preconditions.integerInRange(cls, 'numActions', spec.numActions, 1, 100_000);
    Preconditions.lengthEq(cls, 'transition', spec.transition, spec.numActions);
    for (let a = 0; a < spec.numActions; a++) {
      Preconditions.lengthEq(cls, `transition[${a}]`, spec.transition[a], spec.numStates);
      for (let s = 0; s < spec.numStates; s++) {
        Preconditions.probabilityVector(cls, `transition[${a}][${s}]`, spec.transition[a][s], 1e-6);
      }
    }
    this.numStates = spec.numStates;
    this.numActions = spec.numActions;
    this.transition = spec.transition.map(a => a.map(row => row.slice()));
  }

  /** One-step adjacency: edge s → s' iff some action gives positive
   *  probability of that transition. */
  oneStepAdjacency(): boolean[][] {
    const adj = Array.from({length: this.numStates}, () => new Array<boolean>(this.numStates).fill(false));
    for (let a = 0; a < this.numActions; a++) {
      for (let s = 0; s < this.numStates; s++) {
        for (let t = 0; t < this.numStates; t++) {
          if (this.transition[a][s][t] > 1e-12) adj[s][t] = true;
        }
      }
    }
    return adj;
  }

  /** Reachability closure R[s][t] = "t reachable from s in ≥ 0 steps"
   *  (Floyd–Warshall transitive closure; diagonal forced true). */
  reachabilityClosure(): boolean[][] {
    const n = this.numStates;
    const reach = this.oneStepAdjacency();
    for (let s = 0; s < n; s++) reach[s][s] = true;
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) {
        if (!reach[i][k]) continue;
        for (let j = 0; j < n; j++) {
          if (reach[k][j]) reach[i][j] = true;
        }
      }
    }
    return reach;
  }

  /** Structurally controllable ⟺ every state is reachable from every state
   *  (the controlled graph is strongly connected). */
  isStructurallyControllable(): boolean {
    const reach = this.reachabilityClosure();
    for (let i = 0; i < this.numStates; i++) {
      for (let j = 0; j < this.numStates; j++) {
        if (!reach[i][j]) return false;
      }
    }
    return true;
  }

  /** Count of reachable ordered (s, t) pairs — a controllability "degree". */
  reachablePairCount(): number {
    const reach = this.reachabilityClosure();
    let c = 0;
    for (const row of reach) for (const v of row) if (v) c++;
    return c;
  }
}

// =============================================================================
// 3. PARTIALLY OBSERVABLE PROCESS — distinguishability ("observability")
// =============================================================================

export interface PomdpSpec extends MdpSpec {
  /** number of distinct observations O */
  numObservations: number;
  /** observation[s][o] = P(o | s). Each observation[s] is a pmf. */
  observation: number[][];
}

/** A finite POMDP whose structural-observability test is whether the
 *  observation process can eventually distinguish every pair of states. */
export class PartiallyObservableProcess {
  readonly mdp: MarkovDecisionProcess;
  readonly numObservations: number;
  readonly observation: number[][];

  constructor(spec: PomdpSpec) {
    this.mdp = new MarkovDecisionProcess(spec);
    const cls = 'PartiallyObservableProcess';
    Preconditions.integerInRange(cls, 'numObservations', spec.numObservations, 1, 100_000);
    Preconditions.lengthEq(cls, 'observation', spec.observation, spec.numStates);
    for (let s = 0; s < spec.numStates; s++) {
      Preconditions.probabilityVector(cls, `observation[${s}]`, spec.observation[s], 1e-6);
    }
    this.numObservations = spec.numObservations;
    this.observation = spec.observation.map(row => row.slice());
  }

  /** Partition-refinement labels: states sharing a label are not (yet)
   *  distinguishable. Starts from the observation distribution and refines by
   *  successor-block distributions under each action until stable. */
  distinguishabilityClasses(tol = 1e-9): number[] {
    const n = this.mdp.numStates;
    // Initial signature = quantised observation distribution.
    let labels = this.labelBySignature(
      Array.from({length: n}, (_, s) => this.quantise(this.observation[s], tol)),
    );
    for (let iter = 0; iter < n + 1; iter++) {
      const signatures: string[] = [];
      for (let s = 0; s < n; s++) {
        // Signature = own label + per-action distribution over current labels.
        const parts: string[] = [String(labels[s])];
        for (let a = 0; a < this.mdp.numActions; a++) {
          const blockMass = new Array<number>(n).fill(0);
          for (let t = 0; t < n; t++) blockMass[labels[t]] += this.mdp.transition[a][s][t];
          parts.push(this.quantise(blockMass, tol));
        }
        signatures.push(parts.join('|'));
      }
      const next = this.labelBySignature(signatures);
      if (this.samePartition(labels, next)) break;
      labels = next;
    }
    return labels;
  }

  /** Structurally observable ⟺ refinement yields all-singleton classes
   *  (every pair of states is eventually distinguishable). */
  isStructurallyObservable(): boolean {
    const labels = this.distinguishabilityClasses();
    return new Set(labels).size === this.mdp.numStates;
  }

  /** Pairs (s, t), s < t, that remain indistinguishable (perceptual aliasing). */
  indistinguishablePairs(): Array<[number, number]> {
    const labels = this.distinguishabilityClasses();
    const out: Array<[number, number]> = [];
    for (let s = 0; s < labels.length; s++) {
      for (let t = s + 1; t < labels.length; t++) {
        if (labels[s] === labels[t]) out.push([s, t]);
      }
    }
    return out;
  }

  /** Number of distinguishability classes (full-rank analog: equals S). */
  classCount(): number {
    return new Set(this.distinguishabilityClasses()).size;
  }

  private quantise(v: readonly number[], tol: number): string {
    const digits = Math.max(0, Math.round(-Math.log10(tol)));
    return v.map(x => x.toFixed(digits)).join(',');
  }

  private labelBySignature(signatures: readonly string[]): number[] {
    const map = new Map<string, number>();
    const labels = new Array<number>(signatures.length);
    for (let i = 0; i < signatures.length; i++) {
      let id = map.get(signatures[i]);
      if (id === undefined) {
        id = map.size;
        map.set(signatures[i], id);
      }
      labels[i] = id;
    }
    return labels;
  }

  private samePartition(a: readonly number[], b: readonly number[]): boolean {
    if (a.length !== b.length) return false;
    // Compare induced partitions (label values themselves may renumber).
    const mapAB = new Map<number, number>();
    const mapBA = new Map<number, number>();
    for (let i = 0; i < a.length; i++) {
      const ea = mapAB.get(a[i]);
      if (ea === undefined) mapAB.set(a[i], b[i]); else if (ea !== b[i]) return false;
      const eb = mapBA.get(b[i]);
      if (eb === undefined) mapBA.set(b[i], a[i]); else if (eb !== a[i]) return false;
    }
    return true;
  }
}

// =============================================================================
// DES PIPELINE
// =============================================================================

export class ObsCtrlChannels {
  static readonly MODEL_LTI: ChannelName = 'model-lti';
  static readonly MODEL_MDP: ChannelName = 'model-mdp';
  static readonly MODEL_POMDP: ChannelName = 'model-pomdp';
  static readonly RESULT: ChannelName = 'evaluation';
}

export class StateSpaceToken implements Token {
  constructor(readonly label: string, readonly model: StateSpaceModel) {}
}

export class MdpToken implements Token {
  constructor(readonly label: string, readonly mdp: MarkovDecisionProcess) {}
}

export class PomdpToken implements Token {
  constructor(readonly label: string, readonly pomdp: PartiallyObservableProcess) {}
}

export type EvaluationKind =
  | 'controllability'
  | 'observability'
  | 'mdp-controllability'
  | 'pomdp-observability';

/** A single structural verdict produced by an evaluator station. */
export class EvaluationToken implements Token {
  constructor(
    readonly label: string,
    readonly kind: EvaluationKind,
    /** measured rank / class-count / reachable-degree */
    readonly measure: number,
    /** target value for a "full"/positive verdict */
    readonly target: number,
    readonly full: boolean,
    readonly detail: string,
  ) {}
}

/** Emits a fixed list of linear state-space models once. */
export class StateSpaceSourceStation extends DESStation {
  private emitted = false;

  constructor(id: string, private readonly models: readonly StateSpaceToken[]) {
    super(id);
  }

  override hasWork(): boolean {
    return !this.emitted;
  }

  runTimeStep(): void {
    if (this.emitted) return;
    for (const m of this.models) this.emit(m, ObsCtrlChannels.MODEL_LTI);
    this.emitted = true;
  }
}

/** Emits a fixed list of MDPs once. */
export class MdpSourceStation extends DESStation {
  private emitted = false;

  constructor(id: string, private readonly models: readonly MdpToken[]) {
    super(id);
  }

  override hasWork(): boolean {
    return !this.emitted;
  }

  runTimeStep(): void {
    if (this.emitted) return;
    for (const m of this.models) this.emit(m, ObsCtrlChannels.MODEL_MDP);
    this.emitted = true;
  }
}

/** Emits a fixed list of POMDPs once. */
export class PomdpSourceStation extends DESStation {
  private emitted = false;

  constructor(id: string, private readonly models: readonly PomdpToken[]) {
    super(id);
  }

  override hasWork(): boolean {
    return !this.emitted;
  }

  runTimeStep(): void {
    if (this.emitted) return;
    for (const m of this.models) this.emit(m, ObsCtrlChannels.MODEL_POMDP);
    this.emitted = true;
  }
}

/** Kalman controllability test as a zero-backlog transform. */
export class ControllabilityEvaluatorStation extends PureTransformEntity<StateSpaceToken, EvaluationToken> {
  constructor(id: string) {
    super(id, {inputChannels: ObsCtrlChannels.MODEL_LTI, outputChannel: ObsCtrlChannels.RESULT});
  }

  transform(token: StateSpaceToken): EvaluationToken {
    const n = token.model.stateDim();
    const rank = token.model.controllabilityRank();
    return new EvaluationToken(
      token.label, 'controllability', rank, n, rank === n,
      `rank 𝒞 = ${rank} / n = ${n}`,
    );
  }
}

/** Kalman observability test as a zero-backlog transform. */
export class ObservabilityEvaluatorStation extends PureTransformEntity<StateSpaceToken, EvaluationToken> {
  constructor(id: string) {
    super(id, {inputChannels: ObsCtrlChannels.MODEL_LTI, outputChannel: ObsCtrlChannels.RESULT});
  }

  transform(token: StateSpaceToken): EvaluationToken {
    const n = token.model.stateDim();
    const rank = token.model.observabilityRank();
    return new EvaluationToken(
      token.label, 'observability', rank, n, rank === n,
      `rank 𝒪 = ${rank} / n = ${n}`,
    );
  }
}

/** MDP reachability ("controllability") test as a zero-backlog transform. */
export class MdpControllabilityEvaluatorStation extends PureTransformEntity<MdpToken, EvaluationToken> {
  constructor(id: string) {
    super(id, {inputChannels: ObsCtrlChannels.MODEL_MDP, outputChannel: ObsCtrlChannels.RESULT});
  }

  transform(token: MdpToken): EvaluationToken {
    const s = token.mdp.numStates;
    const target = s * s;
    const reachable = token.mdp.reachablePairCount();
    return new EvaluationToken(
      token.label, 'mdp-controllability', reachable, target, token.mdp.isStructurallyControllable(),
      `reachable ordered pairs = ${reachable} / S² = ${target}`,
    );
  }
}

/** POMDP distinguishability ("observability") test as a zero-backlog transform. */
export class PomdpObservabilityEvaluatorStation extends PureTransformEntity<PomdpToken, EvaluationToken> {
  constructor(id: string) {
    super(id, {inputChannels: ObsCtrlChannels.MODEL_POMDP, outputChannel: ObsCtrlChannels.RESULT});
  }

  transform(token: PomdpToken): EvaluationToken {
    const s = token.pomdp.mdp.numStates;
    const classes = token.pomdp.classCount();
    return new EvaluationToken(
      token.label, 'pomdp-observability', classes, s, token.pomdp.isStructurallyObservable(),
      `distinguishability classes = ${classes} / S = ${s}`,
    );
  }
}

/** Collects evaluation verdicts. */
export class EvaluationSinkStation extends DESStation {
  readonly results: EvaluationToken[] = [];

  constructor(id: string) {
    super(id);
  }

  override hasWork(): boolean {
    return this.inboxSize(ObsCtrlChannels.RESULT) > 0;
  }

  runTimeStep(): void {
    this.results.push(...this.drain<EvaluationToken>(ObsCtrlChannels.RESULT));
  }

  /** Verdicts for one label, in arrival order. */
  forLabel(label: string): EvaluationToken[] {
    return this.results.filter(r => r.label === label);
  }
}
