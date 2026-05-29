'use strict';

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
    if (ix < 0) return;
    this.cursor = (ix + 1) % connections.length;
  }

  getCursor(): number {
    return this.cursor;
  }
}
