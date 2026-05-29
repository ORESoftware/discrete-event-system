'use strict';

import type {Shape} from '../../animation/types';
import {CompositeDESStation} from './composite-station';
import {DESRunLoopEntity, DESStation} from './station';

export type VisualBlockRole = 'source' | 'sink' | 'transform' | 'station' | 'composite' | 'observer';
export type VisualPortDirection = 'in' | 'out';

export interface VisualPortOptions {
  id: string;
  kind?: string;
  label?: string;
  dataType?: string;
  required?: boolean;
  capacity?: number;
  metadata?: Record<string, unknown>;
}

export type VisualPortInput = string | VisualPortOptions;

export interface VisualBlockPort {
  id: string;
  direction: VisualPortDirection;
  kind: string;
  label: string;
  dataType?: string;
  required: boolean;
  capacity?: number;
  metadata?: Record<string, unknown>;
}

export interface VisualBlockPortSpec {
  inputs?: readonly VisualPortInput[];
  outputs?: readonly VisualPortInput[];
}

export interface VisualBlockLayout {
  x?: number;
  y?: number;
  w?: number;
  h?: number;
}

export interface VisualBlockStyle {
  fill?: string;
  stroke?: string;
  text?: string;
}

export interface VisualBlockSpec {
  id: string;
  kind: string;
  role: VisualBlockRole;
  label: string;
  alwaysRenderInHtml: true;
  layout: Required<VisualBlockLayout>;
  ports: {
    inputs: VisualBlockPort[];
    outputs: VisualBlockPort[];
  };
  connectionsIn: VisualBlockConnectionSpec[];
  connectionsOut: VisualBlockConnectionSpec[];
  contains: Array<{
    id?: string;
    kind: string;
  }>;
  style: Required<VisualBlockStyle>;
  metadata?: Record<string, unknown>;
}

export interface VisualBlockOptions {
  kind?: string;
  role?: VisualBlockRole;
  label?: string;
  layout?: VisualBlockLayout;
  ports?: VisualBlockPortSpec;
  style?: VisualBlockStyle;
  metadata?: Record<string, unknown>;
}

export interface VisualBlockRenderContext {
  tick?: number;
  time?: number;
  index?: number;
  stageWidth?: number;
  stageHeight?: number;
}

export type VisualBlockMember = DESStation | DESRunLoopEntity | {id?: string} | unknown;
export type VisualBlockRenderable = VisualBlock | VisualBlockSpec;

export interface VisualBlockConnectionOptions {
  id?: string;
  fromPort?: string;
  toPort?: string;
  kind?: string;
  metadata?: Record<string, unknown>;
  wireDES?: boolean;
}

export interface VisualBlockConnectionSpec {
  id: string;
  kind: string;
  from: {
    blockId: string;
    portId: string;
  };
  to: {
    blockId: string;
    portId: string;
  };
  metadata?: Record<string, unknown>;
}

interface VisualBlockInternalOptions {
  kind: string;
  role: VisualBlockRole;
  label: string;
  layout: Required<VisualBlockLayout>;
  ports: {
    inputs: VisualBlockPort[];
    outputs: VisualBlockPort[];
  };
  style: Required<VisualBlockStyle>;
  metadata?: Record<string, unknown>;
}

const DEFAULT_LAYOUT: Required<VisualBlockLayout> = {x: 24, y: 24, w: 180, h: 64};
const DEFAULT_STYLE: Required<VisualBlockStyle> = {
  fill: '#eef2ff',
  stroke: '#4f46e5',
  text: '#111827',
};

export class VisualBlock extends CompositeDESStation {
  readonly alwaysRenderInHtml = true;
  private readonly visualMembers: VisualBlockMember[] = [];
  private readonly visualConnectionsIn: VisualBlockConnectionSpec[] = [];
  private readonly visualConnectionsOut: VisualBlockConnectionSpec[] = [];
  private readonly visualOptions: VisualBlockInternalOptions;

  constructor(id: string, opts: VisualBlockOptions = {}) {
    super(id);
    const role = opts.role ?? 'station';
    this.visualOptions = {
      kind: opts.kind ?? 'visual-block',
      role,
      label: opts.label ?? id,
      layout: {...DEFAULT_LAYOUT, ...(opts.layout ?? {})},
      ports: {
        inputs: normalizePorts(opts.ports?.inputs ?? [], 'in'),
        outputs: normalizePorts(opts.ports?.outputs ?? [], 'out'),
      },
      style: {...DEFAULT_STYLE, ...(opts.style ?? {})},
      metadata: opts.metadata,
    };
    this.assertRolePorts();
  }

  static source(id: string, outputs: readonly VisualPortInput[], opts: Omit<VisualBlockOptions, 'role' | 'ports'> = {}): VisualBlock {
    return new VisualBlock(id, {...opts, role: 'source', ports: {outputs}});
  }

  static sink(id: string, inputs: readonly VisualPortInput[], opts: Omit<VisualBlockOptions, 'role' | 'ports'> = {}): VisualBlock {
    return new VisualBlock(id, {...opts, role: 'sink', ports: {inputs}});
  }

  override addSubstation<T extends DESStation>(station: T): T {
    const child = super.addSubstation(station);
    this.addVisualMember(child);
    return child;
  }

  addVisualMember<T extends VisualBlockMember>(member: T): T {
    if (!this.visualMembers.includes(member)) this.visualMembers.push(member);
    return member;
  }

  containedVisualMembers(): readonly VisualBlockMember[] {
    return this.visualMembers;
  }

  visualInputPorts(): readonly VisualBlockPort[] {
    return this.visualOptions.ports.inputs;
  }

  visualOutputPorts(): readonly VisualBlockPort[] {
    return this.visualOptions.ports.outputs;
  }

  addInputPort(port: VisualPortInput): VisualBlockPort {
    if (this.visualOptions.role === 'source') throw new Error(`VisualBlock(${this.id}): source blocks cannot have input ports`);
    const normalized = normalizePort(port, 'in');
    assertUniquePort(this.visualOptions.ports.inputs, normalized.id, this.id, 'input');
    this.visualOptions.ports.inputs.push(normalized);
    return normalized;
  }

  addOutputPort(port: VisualPortInput): VisualBlockPort {
    if (this.visualOptions.role === 'sink') throw new Error(`VisualBlock(${this.id}): sink blocks cannot have output ports`);
    const normalized = normalizePort(port, 'out');
    assertUniquePort(this.visualOptions.ports.outputs, normalized.id, this.id, 'output');
    this.visualOptions.ports.outputs.push(normalized);
    return normalized;
  }

  connectTo(target: VisualBlock, opts: VisualBlockConnectionOptions = {}): VisualBlockConnectionSpec {
    const output = this.resolvePort('out', opts.fromPort);
    const input = target.resolvePort('in', opts.toPort);
    const kind = opts.kind ?? output.kind;
    if (output.kind !== kind) {
      throw new Error(`VisualBlock(${this.id}): output port "${output.id}" has kind "${output.kind}", not "${kind}"`);
    }
    if (input.kind !== kind) {
      throw new Error(`VisualBlock(${target.id}): input port "${input.id}" has kind "${input.kind}", not "${kind}"`);
    }
    const connection: VisualBlockConnectionSpec = {
      id: opts.id ?? `${this.id}:${output.id}->${target.id}:${input.id}`,
      kind,
      from: {blockId: this.id, portId: output.id},
      to: {blockId: target.id, portId: input.id},
      metadata: opts.metadata,
    };
    this.visualConnectionsOut.push(connection);
    target.receiveVisualConnection(connection);
    if (opts.wireDES ?? true) this.pipe(target, output.id, input.id);
    return connection;
  }

  visualConnectionsIncoming(): readonly VisualBlockConnectionSpec[] {
    return this.visualConnectionsIn;
  }

  visualConnectionsOutgoing(): readonly VisualBlockConnectionSpec[] {
    return this.visualConnectionsOut;
  }

  setVisualLayout(layout: VisualBlockLayout): this {
    this.visualOptions.layout = {...this.visualOptions.layout, ...layout};
    return this;
  }

  visualBlockSpec(overrides: {layout?: VisualBlockLayout; index?: number} = {}): VisualBlockSpec {
    const layout = {
      ...this.visualOptions.layout,
      ...(overrides.layout ?? {}),
    };
    return {
      id: this.id,
      kind: this.visualOptions.kind,
      role: this.visualOptions.role,
      label: this.visualOptions.label,
      alwaysRenderInHtml: true,
      layout,
      ports: {
        inputs: clonePorts(this.visualOptions.ports.inputs),
        outputs: clonePorts(this.visualOptions.ports.outputs),
      },
      connectionsIn: this.visualConnectionsIn.map(cloneConnection),
      connectionsOut: this.visualConnectionsOut.map(cloneConnection),
      contains: this.visualMembers.map(member => ({
        id: memberId(member),
        kind: memberKind(member),
      })),
      style: {...this.visualOptions.style},
      metadata: this.visualOptions.metadata,
    };
  }

  renderVisualBlock(ctx: VisualBlockRenderContext = {}): Shape[] {
    return renderVisualBlockSpec(this.visualBlockSpec({
      layout: defaultLayoutFor(ctx.index ?? 0, this.visualOptions.layout),
    }));
  }

  private receiveVisualConnection(connection: VisualBlockConnectionSpec): void {
    this.visualConnectionsIn.push(connection);
  }

  private resolvePort(direction: VisualPortDirection, portId: string | undefined): VisualBlockPort {
    const ports = direction === 'in' ? this.visualOptions.ports.inputs : this.visualOptions.ports.outputs;
    if (portId) {
      const found = ports.find(port => port.id === portId);
      if (!found) throw new Error(`VisualBlock(${this.id}): unknown ${direction} port "${portId}"`);
      return found;
    }
    if (ports.length !== 1) {
      throw new Error(`VisualBlock(${this.id}): expected exactly one ${direction} port, found ${ports.length}; pass ${direction === 'in' ? 'toPort' : 'fromPort'}`);
    }
    return ports[0];
  }

  private assertRolePorts(): void {
    if (this.visualOptions.role === 'source' && this.visualOptions.ports.inputs.length > 0) {
      throw new Error(`VisualBlock(${this.id}): source blocks can only define output ports`);
    }
    if (this.visualOptions.role === 'sink' && this.visualOptions.ports.outputs.length > 0) {
      throw new Error(`VisualBlock(${this.id}): sink blocks can only define input ports`);
    }
  }
}

export function isVisualBlock(value: unknown): value is VisualBlock {
  return value instanceof VisualBlock ||
    (typeof value === 'object' && value !== null && (value as {alwaysRenderInHtml?: unknown}).alwaysRenderInHtml === true);
}

export function renderVisualBlocks(
  blocks: readonly VisualBlockRenderable[],
  ctx: Omit<VisualBlockRenderContext, 'index'> = {},
): Shape[] {
  return blocks.flatMap((block, index) => {
    if (block instanceof VisualBlock) return block.renderVisualBlock({...ctx, index});
    return renderVisualBlockSpec({...block, layout: defaultLayoutFor(index, block.layout)});
  });
}

export function visualBlockSpecs(blocks: readonly VisualBlock[]): VisualBlockSpec[] {
  return blocks.map((block, index) => block.visualBlockSpec({layout: defaultLayoutFor(index, block.visualBlockSpec().layout)}));
}

export function renderVisualBlockSpec(spec: VisualBlockSpec): Shape[] {
  const {x, y, w, h} = spec.layout;
  const portY = y + h / 2;
  const shapes: Shape[] = [
    {
      kind: 'rect',
      x,
      y,
      w,
      h,
      rx: 6,
      fill: spec.style.fill,
      stroke: spec.style.stroke,
      strokeWidth: 2,
      title: `${spec.kind}: ${spec.id}`,
      visualBlockId: spec.id,
    },
    {
      kind: 'text',
      x: x + w / 2,
      y: y + 23,
      text: spec.label,
      fontSize: 13,
      anchor: 'middle',
      fontWeight: 'bold',
      fill: spec.style.text,
      visualBlockId: spec.id,
    },
    {
      kind: 'text',
      x: x + w / 2,
      y: y + 43,
      text: spec.contains.length > 0 ? `${spec.kind} (${spec.contains.length})` : spec.kind,
      fontSize: 11,
      anchor: 'middle',
      fill: '#475569',
      visualBlockId: spec.id,
    },
  ];

  spec.ports.inputs.forEach((_port, i) => {
    shapes.push({
      kind: 'circle',
      x,
      y: portY - (spec.ports.inputs.length - 1) * 5 + i * 10,
      r: 4,
      fill: '#ffffff',
      stroke: spec.style.stroke,
      strokeWidth: 1.5,
      title: `${_port.id}: ${_port.kind}`,
      visualBlockId: spec.id,
    });
  });
  spec.ports.outputs.forEach((_port, i) => {
    shapes.push({
      kind: 'circle',
      x: x + w,
      y: portY - (spec.ports.outputs.length - 1) * 5 + i * 10,
      r: 4,
      fill: spec.style.stroke,
      stroke: '#ffffff',
      strokeWidth: 1.5,
      title: `${_port.id}: ${_port.kind}`,
      visualBlockId: spec.id,
    });
  });

  return shapes;
}

function defaultLayoutFor(index: number, layout: Required<VisualBlockLayout>): Required<VisualBlockLayout> {
  const explicit = layout.x !== DEFAULT_LAYOUT.x || layout.y !== DEFAULT_LAYOUT.y;
  if (explicit) return layout;
  return {...layout, x: DEFAULT_LAYOUT.x, y: DEFAULT_LAYOUT.y + index * (layout.h + 12)};
}

function normalizePorts(ports: readonly VisualPortInput[], direction: VisualPortDirection): VisualBlockPort[] {
  const normalized = ports.map(port => normalizePort(port, direction));
  const seen = new Set<string>();
  for (const port of normalized) {
    if (seen.has(port.id)) throw new Error(`VisualBlock: duplicate ${direction} port "${port.id}"`);
    seen.add(port.id);
  }
  return normalized;
}

function normalizePort(port: VisualPortInput, direction: VisualPortDirection): VisualBlockPort {
  const raw: VisualPortOptions = typeof port === 'string' ? {id: port} : port;
  if (!raw.id || raw.id.trim().length === 0) throw new Error('VisualBlock ports require non-empty ids');
  if (raw.capacity !== undefined && (!Number.isInteger(raw.capacity) || raw.capacity < 0)) {
    throw new Error(`VisualBlock port "${raw.id}" capacity must be a non-negative integer`);
  }
  const kind = raw.kind ?? 'token';
  return {
    id: raw.id,
    direction,
    kind,
    label: raw.label ?? raw.id,
    dataType: raw.dataType,
    required: raw.required ?? false,
    capacity: raw.capacity,
    metadata: raw.metadata,
  };
}

function assertUniquePort(ports: readonly VisualBlockPort[], id: string, blockId: string, directionLabel: string): void {
  if (ports.some(port => port.id === id)) throw new Error(`VisualBlock(${blockId}): duplicate ${directionLabel} port "${id}"`);
}

function clonePorts(ports: readonly VisualBlockPort[]): VisualBlockPort[] {
  return ports.map(port => ({
    ...port,
    metadata: port.metadata ? {...port.metadata} : undefined,
  }));
}

function cloneConnection(connection: VisualBlockConnectionSpec): VisualBlockConnectionSpec {
  return {
    ...connection,
    from: {...connection.from},
    to: {...connection.to},
    metadata: connection.metadata ? {...connection.metadata} : undefined,
  };
}

function memberId(member: VisualBlockMember): string | undefined {
  if (typeof member === 'object' && member !== null && 'id' in member) {
    const id = (member as {id?: unknown}).id;
    return typeof id === 'string' ? id : undefined;
  }
  return undefined;
}

function memberKind(member: VisualBlockMember): string {
  if (typeof member === 'object' && member !== null && member.constructor?.name) {
    return member.constructor.name;
  }
  return typeof member;
}
