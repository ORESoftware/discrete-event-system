'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/stateful_token.rs  (module des::general::des_base::stateful_token)
// 1:1 file move. Lineage-tracked tokens with optional state machine + a registry.
//
// Declarations → Rust:
//   type TokenStateMode = 'stateless' | 'stateful'  -> enum TokenStateMode { Stateless, Stateful }
//   interface TokenLineage              -> struct TokenLineage (#[derive(Clone)]; Option fields)
//   interface TokenStateTransition<S>   -> struct (generic S: state label)
//   interface StatefulToken<S> : Token  -> trait StatefulToken: Token  (or a struct)
//   interface StatefulTokenRegistryStats -> struct (byKind -> HashMap<String, u64>)
//   fn makeStatefulToken/makeStatelessToken/transitionToken/spawnStatefulChildToken/
//      childLineage/isStatefulToken      -> free fns (or assoc constructors)
//   class PayloadStatefulToken<S,P>     -> struct + impl StatefulToken
//   class StatefulTokenRegistry          -> struct (tokens: HashMap<String, ..>)
//
// Conversion notes (file-specific):
//   - `S extends string` is a string-literal label param -> a generic `S: ..` or
//     an `enum` of states; `StateOf<T>` conditional type has no Rust analogue.
//   - `isStatefulToken` is a TS type guard (`t is StatefulToken`) -> downcast via
//     `dyn Any`/enum match; there is no structural narrowing in Rust.
//   - `transitionToken` mutates the token in place (`token.currentState = ..`,
//     push history) -> `&mut self`; returns the same handle.
//   - Registry `Map<string, StatefulToken<any>>` -> `HashMap<String, Box<dyn StatefulToken>>`.
//   - `StatefulToken<any>` / `as S | undefined` casts -> concrete generics; avoid `dyn Any`.
// =============================================================================

import {Token} from './station';

export type TokenStateMode = 'stateless' | 'stateful';

export interface TokenLineage {
  tokenId: string;
  parentTokenId?: string;
  rootTokenId: string;
  causationTokenId?: string;
  generation: number;
}

export interface TokenStateTransition<S extends string = string> {
  tick: number;
  stationId: string;
  from?: S;
  to: S;
  event: string;
  detail?: string;
}

export interface StatefulToken<S extends string = string> extends Token {
  kind: string;
  lineage: TokenLineage;
  stateMode: TokenStateMode;
  currentState?: S;
  stateHistory?: TokenStateTransition<S>[];
}

export interface StatefulTokenRegistryStats {
  created: number;
  stateful: number;
  stateless: number;
  stateTransitions: number;
  maxGeneration: number;
  byKind: Record<string, number>;
}

export function makeStatefulToken<S extends string>(opts: {
  kind: string;
  tokenId: string;
  initialState: S;
  tick: number;
  stationId: string;
  event?: string;
  detail?: string;
}): StatefulToken<S> {
  return {
    kind: opts.kind,
    lineage: {
      tokenId: opts.tokenId,
      rootTokenId: opts.tokenId,
      generation: 0,
    },
    stateMode: 'stateful',
    currentState: opts.initialState,
    stateHistory: [{
      tick: opts.tick,
      stationId: opts.stationId,
      to: opts.initialState,
      event: opts.event ?? 'created',
      detail: opts.detail,
    }],
  };
}

export function makeStatelessToken(opts: {
  kind: string;
  tokenId: string;
  parent?: StatefulToken<any>;
  causationTokenId?: string;
}): StatefulToken<never> {
  return {
    kind: opts.kind,
    lineage: childLineage(opts.tokenId, opts.parent, opts.causationTokenId),
    stateMode: 'stateless',
  };
}

type StateOf<T> = T extends StatefulToken<infer S> ? S : string;

export function transitionToken<T extends StatefulToken<any>>(
  token: T,
  nextState: StateOf<T>,
  opts: {
    tick: number;
    stationId: string;
    event: string;
    detail?: string;
  },
): T {
  if (token.stateMode !== 'stateful') return token;
  const from = token.currentState;
  token.currentState = nextState;
  const history = token.stateHistory ?? [];
  history.push({
    tick: opts.tick,
    stationId: opts.stationId,
    from,
    to: nextState,
    event: opts.event,
    detail: opts.detail,
  });
  token.stateHistory = history;
  return token;
}

export function spawnStatefulChildToken<S extends string>(parent: StatefulToken<any>, opts: {
  kind: string;
  tokenId: string;
  initialState: S;
  tick: number;
  stationId: string;
  event?: string;
  detail?: string;
}): StatefulToken<S> {
  return {
    kind: opts.kind,
    lineage: childLineage(opts.tokenId, parent, parent.lineage.tokenId),
    stateMode: 'stateful',
    currentState: opts.initialState,
    stateHistory: [{
      tick: opts.tick,
      stationId: opts.stationId,
      to: opts.initialState,
      event: opts.event ?? 'spawned',
      detail: opts.detail,
    }],
  };
}

export class PayloadStatefulToken<S extends string, P> implements StatefulToken<S> {
  kind: string;
  lineage: TokenLineage;
  stateMode: TokenStateMode;
  currentState?: S;
  stateHistory?: TokenStateTransition<S>[];
  readonly payload: P;

  constructor(opts: {
    kind: string;
    tokenId: string;
    payload: P;
    initialState: S;
    tick: number;
    stationId: string;
    event?: string;
    detail?: string;
    parent?: StatefulToken<any>;
    causationTokenId?: string;
    stateMode?: TokenStateMode;
  }) {
    const base = opts.stateMode === 'stateless'
      ? makeStatelessToken({
        kind: opts.kind,
        tokenId: opts.tokenId,
        parent: opts.parent,
        causationTokenId: opts.causationTokenId,
      })
      : opts.parent
        ? spawnStatefulChildToken(opts.parent, {
          kind: opts.kind,
          tokenId: opts.tokenId,
          initialState: opts.initialState,
          tick: opts.tick,
          stationId: opts.stationId,
          event: opts.event,
          detail: opts.detail,
        })
        : makeStatefulToken({
          kind: opts.kind,
          tokenId: opts.tokenId,
          initialState: opts.initialState,
          tick: opts.tick,
          stationId: opts.stationId,
          event: opts.event,
          detail: opts.detail,
        });
    this.kind = base.kind;
    this.lineage = base.lineage;
    this.stateMode = base.stateMode;
    this.currentState = base.currentState as S | undefined;
    this.stateHistory = base.stateHistory as TokenStateTransition<S>[] | undefined;
    this.payload = opts.payload;
  }
}

export class StatefulTokenRegistry {
  private readonly tokens = new Map<string, StatefulToken<any>>();
  private readonly byKind: Record<string, number> = {};
  private created = 0;
  private stateful = 0;
  private stateless = 0;
  private maxGeneration = 0;

  track(t: StatefulToken<any>): void {
    if (this.tokens.has(t.lineage.tokenId)) {
      this.tokens.set(t.lineage.tokenId, t);
      return;
    }
    this.tokens.set(t.lineage.tokenId, t);
    this.created++;
    this.byKind[t.kind] = (this.byKind[t.kind] ?? 0) + 1;
    if (t.stateMode === 'stateful') this.stateful++;
    else this.stateless++;
    this.maxGeneration = Math.max(this.maxGeneration, t.lineage.generation);
  }

  snapshot(): StatefulTokenRegistryStats {
    let stateTransitions = 0;
    for (const t of this.tokens.values()) stateTransitions += t.stateHistory?.length ?? 0;
    return {
      created: this.created,
      stateful: this.stateful,
      stateless: this.stateless,
      stateTransitions,
      maxGeneration: this.maxGeneration,
      byKind: {...this.byKind},
    };
  }
}

export function childLineage(
  tokenId: string,
  parent?: StatefulToken<any>,
  causationTokenId?: string,
): TokenLineage {
  if (!parent) return {tokenId, rootTokenId: tokenId, generation: 0, causationTokenId};
  return {
    tokenId,
    parentTokenId: parent.lineage.tokenId,
    rootTokenId: parent.lineage.rootTokenId,
    causationTokenId,
    generation: parent.lineage.generation + 1,
  };
}

export function isStatefulToken<S extends string = string>(t: Token): t is StatefulToken<S> {
  const maybe = t as Partial<StatefulToken<S>>;
  return !!maybe.lineage && (maybe.stateMode === 'stateful' || maybe.stateMode === 'stateless');
}
