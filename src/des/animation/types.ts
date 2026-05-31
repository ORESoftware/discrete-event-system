'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/types.rs
// - Keep file-for-file. Convert Animation, Frame, ChartSpec, and ChartSeries to serde Serialize/Deserialize structs.
// - Convert Shape to a tagged serde enum with circle/rect/line/text/path variants; optional TS fields become Option<T>.
// - Preserve visualBlockId as an optional string boundary field until the visual block model has typed Rust ids.

// =============================================================================
// Core types for the animation plugin.
//
// DESIGN
// ------
// Every simulation in this engine runs on a fixed-step tick clock. So the
// natural animation primitive is a `Frame` per tick (or per Nth tick). A
// simulation hooks into the FrameRecorder, declares "this is what the
// scene looks like right now", and the recorder serialises it to a JSONL
// stream. Later, the HtmlPlayer reads those frames and emits a self-
// contained HTML file with play/pause/scrub controls.
//
// REAL-TIME vs POST-HOC
// ---------------------
// Because the simulation is single-threaded the recorder can also tee
// each frame to stderr as a one-line tick summary, giving a "live" feel
// without needing a second process. The frames file produced in this
// mode is identical to the one produced offline.
//
// SCHEMA
// ------
// Animation:
//   { width, height, fps, title?, frames: Frame[], charts?: ChartSpec[] }
//
// Frame (per-tick scene state):
//   { t, tick, shapes: Shape[], caption?: string, marks?: Mark[] }
//
// Shape kinds: circle, rect, line, text. Each carries minimal SVG-ish
// attributes (x/y/r/w/h/fill/stroke). Coordinates are in the animation's
// width × height pixel grid; renderers don't transform them.
//
// ChartSpec (global time-series panels rendered alongside frames):
//   { x, y, w, h, title?, yMin?, yMax?, series: ChartSeries[] }
//
// ChartSeries:
//   { label, color, t: number[], y: number[] }   (parallel arrays)
//
// The player draws each chart's series clipped to t ≤ frames[i].t,
// producing the "watch the line draw itself" effect without the
// per-frame data being O(N²).
// =============================================================================

export interface Animation {
  /** Pixel width of the SVG stage. */
  width: number;
  /** Pixel height of the SVG stage. */
  height: number;
  /** Default playback rate (frames per second). */
  fps: number;
  /** Optional title displayed in the page header. */
  title?: string;
  /** Optional one-line caption shown beneath the title. */
  subtitle?: string;
  /** Per-tick scene snapshots. */
  frames: Frame[];
  /** Optional global time series panels rendered alongside frames. */
  charts?: ChartSpec[];
  /** Optional CSS background color for the stage. Defaults to white. */
  background?: string;
}

export interface Frame {
  /** Simulation time. */
  t: number;
  /** Tick index (integer). */
  tick: number;
  /** SVG shapes that make up this frame. */
  shapes: Shape[];
  /** Optional caption shown beneath the stage for this frame. */
  caption?: string;
}

export type Shape =
  | CircleShape
  | RectShape
  | LineShape
  | TextShape
  | PathShape;

export interface CircleShape {
  kind: 'circle';
  x: number;
  y: number;
  r: number;
  fill: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
  label?: string;
  /** Optional SVG-style title (hover text). */
  title?: string;
  /** VisualBlock id when this shape is part of an always-rendered block. */
  visualBlockId?: string;
}

export interface RectShape {
  kind: 'rect';
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
  label?: string;
  /** Optional rounded-corner radius. */
  rx?: number;
  title?: string;
  /** VisualBlock id when this shape is part of an always-rendered block. */
  visualBlockId?: string;
}

export interface LineShape {
  kind: 'line';
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  stroke: string;
  strokeWidth?: number;
  opacity?: number;
  /** "5,3" for dashed, omit for solid. */
  dasharray?: string;
  /** VisualBlock id when this shape is part of an always-rendered block. */
  visualBlockId?: string;
}

export interface TextShape {
  kind: 'text';
  x: number;
  y: number;
  text: string;
  fontSize?: number;
  fill?: string;
  /** "start" | "middle" | "end". */
  anchor?: 'start' | 'middle' | 'end';
  fontWeight?: 'normal' | 'bold';
  fontFamily?: string;
  /** VisualBlock id when this shape is part of an always-rendered block. */
  visualBlockId?: string;
}

export interface PathShape {
  kind: 'path';
  /** SVG path data. */
  d: string;
  stroke?: string;
  strokeWidth?: number;
  fill?: string;
  opacity?: number;
  /** VisualBlock id when this shape is part of an always-rendered block. */
  visualBlockId?: string;
}

export interface ChartSpec {
  x: number;
  y: number;
  w: number;
  h: number;
  title?: string;
  yMin?: number;
  yMax?: number;
  /** Display name for the y-axis. */
  yLabel?: string;
  series: ChartSeries[];
  /** If true, draws a vertical "current time" cursor line. Defaults to true. */
  cursor?: boolean;
}

export interface ChartSeries {
  label: string;
  color: string;
  /** Parallel arrays. t and y must have the same length. */
  t: number[];
  y: number[];
}
