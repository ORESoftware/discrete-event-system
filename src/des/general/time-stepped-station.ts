'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/time-stepped-station.rs  (module des::general::time_stepped_station)
// 1:1 file move. Lightweight base classes for fixed-step (tick-driven) DES stations.
//
// Declarations → Rust:
//   abstract class TimeSteppedStation                         -> trait TimeSteppedStation { fn run_time_step(..) }
//   abstract class BufferedTimeSteppedStation<T>              -> trait + a base struct holding the inbox
//   abstract class RoutedTimeSteppedStation<T>                -> trait extending Buffered (see note)
//   abstract class BidirectionalTimeSteppedStation<F,B>       -> trait (fwd/back channels)
//   abstract class SynchronousDataflowStation<V>              -> trait (SDF firing)
//   interface SynchronousDataflowConnection<V>                -> struct
//
// Conversion notes (file-specific):
//   - INHERITANCE CHAIN: Routed -> Buffered -> TimeSteppedStation. Rust has no class inheritance:
//     express as a `trait TimeSteppedStation` with default fns + concrete structs that COMPOSE a
//     shared base (e.g. an `inbox: VecDeque<T>` field) and `impl` the traits. Do not mirror `extends`.
//   - `inbox: T[]` FIFO -> `VecDeque<T>`; `id: string` -> `String`; generics <T,F,B,V> carry over.
//   - pure scheduling primitives: no RNG/clock/Map here.
// =============================================================================
// Shared station primitives for fixed-step DES models.
//
// The full queueing framework (`abstract/`, entity-source/processor/sink, and
// entity-moving/) is the preferred substrate when a model is naturally a
// source -> station -> sink network of movable entities. These lighter bases
// cover the fixed-step simulations that still need the same discipline:
// explicit stationary entities, movable payloads/tokens, stable IDs for
// visualisation, and order-independent tick loops.
// =============================================================================

/** Common contract for stationary entities driven by a fixed-size tick. */
export abstract class TimeSteppedStation {
  constructor(public id: string) {}
  abstract runTimeStep(stepSize: number, t: number): void;
}

/** A station with a FIFO inbox of movable items/tokens. */
export abstract class BufferedTimeSteppedStation<T> extends TimeSteppedStation {
  protected inbox: T[] = [];

  takeItem(item: T): void {
    this.inbox.push(item);
  }

  inboxSize(): number {
    return this.inbox.length;
  }

  protected drainInbox(): T[] {
    const items = this.inbox;
    this.inbox = [];
    return items;
  }
}

/** A buffered station that emits each item to all declared downstream stations. */
export abstract class RoutedTimeSteppedStation<T> extends BufferedTimeSteppedStation<T> {
  protected out: Array<BufferedTimeSteppedStation<T>> = [];

  addOutConnection(station: BufferedTimeSteppedStation<T>): this {
    this.out.push(station);
    return this;
  }

  protected emit(item: T): void {
    for (const target of this.out) target.takeItem(item);
  }
}

/**
 * Bidirectional token station for models with a forward and backward pass
 * over the same graph, such as backpropagation.
 */
export abstract class BidirectionalTimeSteppedStation<F, B> extends TimeSteppedStation {
  forwardInbox: F[] = [];
  backwardInbox: B[] = [];
  forwardOut: Array<BidirectionalTimeSteppedStation<F, B>> = [];
  backwardOut: Array<BidirectionalTimeSteppedStation<F, B>> = [];

  takeForward(token: F): void {
    this.forwardInbox.push(token);
  }

  takeBackward(token: B): void {
    this.backwardInbox.push(token);
  }

  addForwardOut(station: BidirectionalTimeSteppedStation<F, B>): this {
    this.forwardOut.push(station);
    return this;
  }

  addBackwardOut(station: BidirectionalTimeSteppedStation<F, B>): this {
    this.backwardOut.push(station);
    return this;
  }

  protected emitForward(token: F): void {
    for (const target of this.forwardOut) target.takeForward(token);
  }

  protected emitBackward(token: B): void {
    for (const target of this.backwardOut) target.takeBackward(token);
  }
}

export interface SynchronousDataflowConnection<V> {
  kind: string;
  target: SynchronousDataflowStation<V>;
}

/**
 * Station base for synchronous data-flow networks. Emissions are staged in a
 * target's pending map and become visible only after commit(), giving every
 * station a frozen view of inputs for the current tick.
 */
export abstract class SynchronousDataflowStation<V = number> extends TimeSteppedStation {
  inbox: Map<string, V> = new Map();
  pending: Map<string, V> = new Map();
  outConnections: Array<SynchronousDataflowConnection<V>> = [];

  addOut(kind: string, target: SynchronousDataflowStation<V>): this {
    this.outConnections.push({kind, target});
    return this;
  }

  protected emit(kind: string, value: V): void {
    for (const c of this.outConnections) {
      if (c.kind === kind) c.target.pending.set(kind, value);
    }
  }

  /** Move every pending emission into the next-tick inbox. */
  commit(): void {
    for (const [key, value] of this.pending) this.inbox.set(key, value);
    this.pending.clear();
  }
}
