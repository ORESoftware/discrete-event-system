'use strict';

// RUST MIGRATION:
// - Target: src/des/entity_routing/output_routing_policy.rs
// - OutputRoutingPolicy should be a Rust enum; HasOutputRoutingPolicy becomes
//   either a trait with `output_routing()` or a config struct embedded by
//   stations that route to one downstream target.
// - OutputConnectionRouter<C> maps cleanly to a generic struct with cursor state
//   and impl methods; `order` can return Vec<C> or ordered indices to avoid
//   cloning heavy connection handles.
// - `random` currently delegates to fisherYatesShuffle/Math.random; inject an
//   RNG trait so deterministic tests and Rust rand usage are explicit.

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

  constructor(
    readonly policy: OutputRoutingPolicy = 'random',
    private readonly rng: () => number = Math.random,
  ) {}

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
        return Array.from(fisherYatesShuffle(connections.slice(), this.rng));
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
