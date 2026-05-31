'use strict';

// RUST MIGRATION:
// - Target: src/des/general/des_base/cut_pool.rs
// - Keep file-for-file. CutEnvelopeSense should become an enum and AffineCut a
//   data struct with Vec<f64> coefficients.
// - AffineCutPool becomes a concrete struct with inherent methods; any public
//   evaluation/update behavior can also implement a small trait for optimizers.
// - Keep scalar/vector helper logic private or associated with AffineCutPool; if
//   a cut evaluator is made into a DES graph node, expose it via PureTransform.
// - Convert invalid dimensions or envelope misuse to Result-returning methods.

// =============================================================================
// general/des-base/cut-pool.ts -- reusable affine cut pools for decomposition
// algorithms: Benders/L-shaped, SDDP, Kelley cutting planes, outer
// approximation, and other value-function approximation schemes.
//
// A cut is an affine function
//
//     f(x) = alpha + beta*x
//
// and a pool represents either an UPPER envelope (min of cuts, used for
// concave maximisation value functions such as the SDDP model below) or a
// LOWER envelope (max of cuts, used for convex minimisation recourse costs).
// The class is deliberately small: validation, dimension checks, envelope
// evaluation, and cheap snapshots for tests / JSON output.
// =============================================================================

import {Preconditions} from './preconditions';

export type CutEnvelopeSense = 'upper' | 'lower';

export interface AffineCut {
  alpha: number;
  beta: number[];
  /** Optional provenance string: "terminal", "iter=12 stage=2", ... */
  source?: string;
}

export class AffineCutPool {
  private readonly cuts: AffineCut[] = [];

  constructor(
    readonly dimension: number,
    readonly sense: CutEnvelopeSense,
    initialCuts: ReadonlyArray<AffineCut> = [],
  ) {
    Preconditions.integerInRange('AffineCutPool', 'dimension', dimension, 1, 1e6);
    for (const c of initialCuts) this.add(c);
  }

  add(cut: AffineCut): void {
    this.assertCut(cut);
    this.cuts.push({alpha: cut.alpha, beta: cut.beta.slice(), source: cut.source});
  }

  size(): number { return this.cuts.length; }

  all(): AffineCut[] {
    return this.cuts.map(c => ({alpha: c.alpha, beta: c.beta.slice(), source: c.source}));
  }

  evaluateCut(cut: AffineCut, x: ReadonlyArray<number>): number {
    this.assertPoint(x);
    let v = cut.alpha;
    for (let i = 0; i < this.dimension; i++) v += cut.beta[i] * x[i];
    return v;
  }

  evaluate(x: ReadonlyArray<number>): number {
    this.assertPoint(x);
    if (this.cuts.length === 0) {
      return this.sense === 'upper' ? Infinity : -Infinity;
    }
    let best = this.sense === 'upper' ? Infinity : -Infinity;
    for (const cut of this.cuts) {
      const v = this.evaluateCut(cut, x);
      if (this.sense === 'upper') {
        if (v < best) best = v;
      } else {
        if (v > best) best = v;
      }
    }
    return best;
  }

  activeCut(x: ReadonlyArray<number>): AffineCut | null {
    this.assertPoint(x);
    if (this.cuts.length === 0) return null;
    let bestIdx = 0;
    let best = this.evaluateCut(this.cuts[0], x);
    for (let i = 1; i < this.cuts.length; i++) {
      const v = this.evaluateCut(this.cuts[i], x);
      if ((this.sense === 'upper' && v < best) || (this.sense === 'lower' && v > best)) {
        best = v; bestIdx = i;
      }
    }
    const c = this.cuts[bestIdx];
    return {alpha: c.alpha, beta: c.beta.slice(), source: c.source};
  }

  private assertCut(cut: AffineCut): void {
    Preconditions.finite('AffineCutPool', 'cut.alpha', cut.alpha);
    Preconditions.lengthEq('AffineCutPool', 'cut.beta', cut.beta, this.dimension);
    Preconditions.allFinite('AffineCutPool', 'cut.beta', cut.beta);
  }

  private assertPoint(x: ReadonlyArray<number>): void {
    Preconditions.lengthEq('AffineCutPool', 'x', x, this.dimension);
    Preconditions.allFinite('AffineCutPool', 'x', x);
  }
}
