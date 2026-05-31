'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/des/general/des_base/composite_station.rs  (module des::general::des_base::composite_station)
// 1:1 file move. A DESStation that owns an internal station sub-graph and
// exposes it through explicit input/output ports.
//
// Declarations → Rust:
//   class CompositePortBridgeStation (private) -> struct + impl DESStation (egress buffer)
//   interface CompositeInputPort / CompositeOutputPort -> structs
//   interface CompositeStationSnapshot   -> struct (#[derive(Clone)])
//   class CompositeDESStation            -> struct + impl DESStation (composes children)
//
// Conversion notes (file-specific):
//   - COMPOSITION over inheritance already: `children: DESStation[]` ->
//     `Vec<Rc<RefCell<dyn DESStation>>>` (or an arena); `addSubstation` returns the
//     handle. `pipe`/`take`/`emit` route via those handles.
//   - `override hasWork/onFinalize/numValidators/runValidation` recurse into
//     children + `super.*` -> in Rust call the default-method body explicitly
//     (no `super`); aggregate child results.
//   - `Record<string, number>` / nested `Record<..>` (snapshot) -> `HashMap`.
//   - Borrow checker: ticking children while the parent holds them needs
//     `RefCell` borrows scoped per child, or index-based iteration over an arena.
// =============================================================================

import {ChannelName, DEFAULT_CHANNEL, DESStation, Token} from './station';
import {ValidationCheck} from './validation';

class CompositePortBridgeStation extends DESStation {
  runTimeStep(): void {}

  drainPort<T extends Token = Token>(channel: ChannelName): T[] {
    return this.drain<T>(channel);
  }

  hasPortWork(): boolean {
    return this.hasWork();
  }
}

interface CompositeInputPort {
  outerChannel: ChannelName;
  target: DESStation;
  targetChannel: ChannelName;
}

interface CompositeOutputPort {
  outerChannel: ChannelName;
  bridge: CompositePortBridgeStation;
}

export interface CompositeStationSnapshot {
  id: string;
  childIds: string[];
  inboxes: Record<string, number>;
  childInboxes: Record<string, Record<string, number>>;
}

/**
 * A DES station that owns an internal station graph.
 *
 * CompositeDESStation is useful when a model has meaningful internal
 * queueing/protocol structure, but callers should still see one station in the
 * outer topology. Outer tokens are routed through explicit input/output ports;
 * internal substations run one tick per parent tick in declared order.
 */
export class CompositeDESStation extends DESStation {
  protected readonly children: DESStation[] = [];
  private readonly inputPorts: CompositeInputPort[] = [];
  private readonly outputPorts: CompositeOutputPort[] = [];
  private tick = 0;

  addSubstation<T extends DESStation>(station: T): T {
    this.children.push(station);
    return station;
  }

  exposeInput(
    outerChannel: ChannelName,
    target: DESStation,
    targetChannel: ChannelName = outerChannel,
  ): this {
    this.inputPorts.push({outerChannel, target, targetChannel});
    return this;
  }

  exposeOutput(
    source: DESStation,
    sourceChannel: ChannelName = DEFAULT_CHANNEL,
    outerChannel: ChannelName = sourceChannel,
  ): this {
    const bridge = this.addSubstation(new CompositePortBridgeStation(`${this.id}:out:${outerChannel}:${this.outputPorts.length}`));
    source.pipe(bridge, sourceChannel, outerChannel);
    this.outputPorts.push({outerChannel, bridge});
    return this;
  }

  childStations(): readonly DESStation[] {
    return this.children;
  }

  override assertPreconditions(): void {
    for (const child of this.children) child.assertPreconditions();
  }

  override hasWork(): boolean {
    if (super.hasWork()) return true;
    for (const child of this.children) if (child.hasWork()) return true;
    for (const port of this.outputPorts) if (port.bridge.hasPortWork()) return true;
    return false;
  }

  runTimeStep(): void {
    this.routeIngress();
    for (const child of this.children) child.runTimeStep();
    this.routeEgress();
    this.tick++;
  }

  override onFinalize(): void {
    for (const child of this.children) child.onFinalize();
  }

  override numValidators(): number {
    return super.numValidators() + this.children.reduce((n, child) => n + child.numValidators(), 0);
  }

  override runValidation(): ValidationCheck[] {
    const out = super.runValidation();
    for (const child of this.children) out.push(...child.runValidation());
    return out;
  }

  snapshotComposite(): CompositeStationSnapshot {
    const childInboxes: Record<string, Record<string, number>> = {};
    for (const child of this.children) childInboxes[child.id] = child.inboxSizes();
    return {
      id: this.id,
      childIds: this.children.map(c => c.id),
      inboxes: this.inboxSizes(),
      childInboxes,
    };
  }

  protected compositeTick(): number {
    return this.tick;
  }

  private routeIngress(): void {
    for (const port of this.inputPorts) {
      for (const token of this.drain(port.outerChannel)) {
        port.target.take(token, port.targetChannel);
      }
    }
  }

  private routeEgress(): void {
    for (const port of this.outputPorts) {
      for (const token of port.bridge.drainPort(port.outerChannel)) {
        this.emit(token, port.outerChannel);
      }
    }
  }
}
