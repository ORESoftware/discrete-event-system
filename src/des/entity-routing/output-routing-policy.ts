'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/entity-routing/output-routing-policy.rs  (module des::entity_routing::output_routing_policy)
// 1:1 file move. Output-connection ordering policies for single-target routing.
//
// Declarations → Rust:
//   type OutputRoutingPolicy = 'random'|'round-robin'|'ordered'
//                                  -> enum OutputRoutingPolicy { Random, RoundRobin, Ordered }
//                                     (#[serde(rename_all = "kebab-case")])
//   interface HasOutputRoutingPolicy -> trait or a `Option<OutputRoutingPolicy>` field
//   class OutputConnectionRouter<C>  -> struct { policy, cursor } + impl
//
// Conversion notes (file-specific):
//   - `fisherYatesShuffle` consumes randomness -> the router must hold an injected
//     `RandomSource` (shared/capabilities); do NOT reach for ambient rng/Math.random.
//   - `cursor` for round-robin is mutable state -> `order`/`markAccepted` take `&mut self`.
//   - `connections.indexOf(accepted)` needs identity/equality -> require `C: PartialEq`
//     or route by index/id rather than by value.
//   - `order()` returns slice/concat COPIES -> `Vec<C>` (require `C: Clone`) or
//     return a permutation of indices to avoid cloning edges.
// =============================================================================

// =============================================================================
// entity-routing/output-routing-policy.ts
//
// Shared output-connection ordering policies for queueing-style stations that
// choose ONE accepting downstream target. This is deliberately not used by
// DESStation.emit(), which broadcasts to every listener and therefore has no
// competitive routing bias to resolve.
// =============================================================================

import {fisherYatesShuffle} from '../general/general';

export type OutputRoutingPolicy = 'random' | 'round-robin' | 'ordered';

export interface HasOutputRoutingPolicy {
  outputRouting?: OutputRoutingPolicy;
}

export class OutputConnectionRouter<C> {
  private cursor = 0;

  constructor(readonly policy: OutputRoutingPolicy = 'random') {}

  order(connections: readonly C[]): C[] {
    if (connections.length <= 1) return connections.slice();
    switch (this.policy) {
      case 'ordered':
        return connections.slice();
      case 'round-robin': {
        const start = this.cursor % connections.length;
        return connections.slice(start).concat(connections.slice(0, start));
      }
      case 'random':
      default:
        return Array.from(fisherYatesShuffle(connections.slice()));
    }
  }

  markAccepted(connections: readonly C[], accepted: C): void {
    if (this.policy !== 'round-robin' || connections.length === 0) return;
    const ix = connections.indexOf(accepted);
    if (ix < 0) {
      console.warn(`[output-router] round-robin markAccepted: accepted connection not found in the ${connections.length}-element connection list; cursor left unchanged (rotation may stall).`);
      return;
    }
    this.cursor = (ix + 1) % connections.length;
  }

  getCursor(): number {
    return this.cursor;
  }
}
