// RUST MIGRATION: target module src/des/general/cartesian_state_space.rs.
// RUST MIGRATION: CartesianDimension, CoordinateTransition, and CoordinateMDPSpec become serde structs; optional fields become Option<T>.
// RUST MIGRATION: CartesianStateSpace becomes a nominal struct with impl methods for coordinate/index conversion; preserve checked conversions with Result.
// RUST MIGRATION: coordinateMDPToSpec is a pure adapter and can stay a free function unless exposed as a PureTransform in the Rust graph API.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/cartesian-state-space.rs  (module des::general::cartesian_state_space)
// 1:1 file move. Reversible index<->coordinate bridge for multi-dim discrete MDP/POMDP state spaces.
//
// Declarations → Rust:
//   interface CartesianDimension / CoordinateTransition -> structs (#[derive(Clone)])
//   interface CoordinateMDPSpec  -> struct holding boxed callbacks (see notes)
//   class CartesianStateSpace    -> struct { dimensions, strides, num_states } + impl
//   fn coordinateMDPToSpec       -> fn (CoordinateMDPSpec) -> MDPSpec
//
// Conversion notes (file-specific):
//   - `CoordinateMDPSpec` fields `numActions`/`outcomes`/`isTerminal`/... are closures;
//     model as trait methods or `Box<dyn Fn(&[usize], usize) -> _>` (optionals -> Option<...>).
//   - dup-name guard uses `Set<string>` -> `HashSet<String>` (String: Hash+Eq).
//   - integer indices are `usize`; `Math.floor(index/stride)%size` is integer division.
//   - constructor `throw` are invariant violations -> `panic!` (or a fallible `new`).
// =============================================================================

// =============================================================================
// CartesianStateSpace
//
// Shared indexing for multi-dimensional discrete MDP/POMDP state spaces.
// The algorithms in this repo mostly consume compact integer state IDs, while
// models are easier to read as coordinates like (x, y, inventory, backlog).
// This helper is the reversible bridge between those two views.
// =============================================================================

import {MDPSpec, Outcome} from './value-iteration';

export interface CartesianDimension {
  name: string;
  size: number;
  labels?: readonly string[];
}

export interface CoordinateTransition {
  prob: number;
  reward: number;
  next: readonly number[];
}

export interface CoordinateMDPSpec {
  space: CartesianStateSpace;
  numActions: (coords: readonly number[], stateIndex: number) => number;
  outcomes: (coords: readonly number[], action: number, stateIndex: number) => CoordinateTransition[];
  isTerminal?: (coords: readonly number[], stateIndex: number) => boolean;
  terminalReward?: (coords: readonly number[], stateIndex: number) => number;
  actionLabel?: (action: number) => string;
}

export class CartesianStateSpace {
  readonly dimensions: readonly CartesianDimension[];
  readonly strides: readonly number[];
  readonly numStates: number;

  constructor(dimensions: readonly CartesianDimension[]) {
    if (dimensions.length === 0) throw new Error('CartesianStateSpace: at least one dimension is required');
    const names = new Set<string>();
    const strides: number[] = [];
    let n = 1;
    for (const dim of dimensions) {
      if (!dim.name) throw new Error('CartesianStateSpace: dimension names must be non-empty');
      if (names.has(dim.name)) throw new Error(`CartesianStateSpace: duplicate dimension "${dim.name}"`);
      names.add(dim.name);
      if (!Number.isInteger(dim.size) || dim.size <= 0) {
        throw new Error(`CartesianStateSpace: dimension "${dim.name}" size must be a positive integer`);
      }
      if (dim.labels !== undefined && dim.labels.length !== dim.size) {
        throw new Error(`CartesianStateSpace: dimension "${dim.name}" labels length must equal size`);
      }
      strides.push(n);
      n *= dim.size;
    }
    this.dimensions = dimensions.map(d => ({...d, labels: d.labels?.slice()}));
    this.strides = strides;
    this.numStates = n;
  }

  rank(): number {
    return this.dimensions.length;
  }

  encode(coords: readonly number[]): number {
    if (coords.length !== this.dimensions.length) {
      throw new Error(`CartesianStateSpace.encode: coordinate rank ${coords.length} != ${this.dimensions.length}`);
    }
    let index = 0;
    for (let i = 0; i < coords.length; i++) {
      const c = coords[i];
      const dim = this.dimensions[i];
      if (!Number.isInteger(c) || c < 0 || c >= dim.size) {
        throw new Error(`CartesianStateSpace.encode: ${dim.name}=${c} outside [0, ${dim.size - 1}]`);
      }
      index += c * this.strides[i];
    }
    return index;
  }

  decode(index: number): number[] {
    if (!Number.isInteger(index) || index < 0 || index >= this.numStates) {
      throw new Error(`CartesianStateSpace.decode: index ${index} outside [0, ${this.numStates - 1}]`);
    }
    const coords = new Array<number>(this.dimensions.length);
    for (let i = this.dimensions.length - 1; i >= 0; i--) {
      coords[i] = Math.floor(index / this.strides[i]) % this.dimensions[i].size;
    }
    return coords;
  }

  label(index: number): string {
    return this.coordLabel(this.decode(index));
  }

  coordLabel(coords: readonly number[]): string {
    if (coords.length !== this.dimensions.length) {
      throw new Error(`CartesianStateSpace.coordLabel: coordinate rank ${coords.length} != ${this.dimensions.length}`);
    }
    return coords.map((c, i) => {
      const dim = this.dimensions[i];
      const value = dim.labels?.[c] ?? String(c);
      return `${dim.name}=${value}`;
    }).join(',');
  }

  allCoords(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.numStates; i++) out.push(this.decode(i));
    return out;
  }
}

export function coordinateMDPToSpec(spec: CoordinateMDPSpec): MDPSpec {
  const space = spec.space;
  return {
    numStates: space.numStates,
    numActions: s => spec.numActions(space.decode(s), s),
    outcomes: (s, a): Outcome[] => spec.outcomes(space.decode(s), a, s).map(o => ({
      prob: o.prob,
      reward: o.reward,
      nextState: space.encode(o.next),
    })),
    isTerminal: spec.isTerminal ? s => spec.isTerminal!(space.decode(s), s) : undefined,
    terminalReward: spec.terminalReward ? s => spec.terminalReward!(space.decode(s), s) : undefined,
    stateLabel: s => space.label(s),
    actionLabel: spec.actionLabel,
  };
}
