'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/frame_recorder.rs
// - Keep FrameRecorderOpts as a config struct and FrameRecorder as a stateful writer around PathBuf + std::fs::File/BufWriter.
// - frame(...) can stay an inherent method accepting an FnOnce builder; if graph-visible later, split a PureTransform adapter out.
// - readAnimation should return Result<Animation, AnimationReadError>; replace throw/any JSON events with serde-tagged structs/enums.
// - HTML output remains delegated to html_player::build_html and filesystem work should use std::fs plus PathBuf.

// =============================================================================
// FrameRecorder — emits per-tick scene snapshots.
//
// Three sinks:
//   1. JSONL frames file       (always)
//   2. Optional ANSI tick line (real-time feel; written to stderr)
//   3. Optional HTML output    (one-shot post-hoc rendering at finish())
//
// Because the simulation is single-threaded, the recorder is called
// inline from the tick loop. Calling .frame() is cheap (O(|shapes|)
// stringify + one stream.write). The HTML output is generated only
// when finish() is called, by reading back the JSONL.
//
// Usage:
//   const rec = new FrameRecorder({
//     framesPath: 'out/two-disease.frames.jsonl',
//     htmlPath:   'out/two-disease.html',
//     width: 900, height: 600, fps: 30, title: 'Two-disease',
//     liveTickLine: true,
//   });
//   for each tick t:
//     rec.frame(t, tickIndex, () => buildSceneFor(...));
//   rec.setCharts([{ ... }]);
//   rec.finish();
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';
import {Animation, ChartSpec, Frame} from './types';
import {buildHTML} from './html-player';
import {VisualBlockRenderable, renderVisualBlocks} from '../general/des-base/visual-block';

export interface FrameRecorderOpts {
  /** Path to write a JSONL frames file. Required. */
  framesPath: string;
  /** Path to write the standalone HTML file at finish(). Optional. */
  htmlPath?: string;
  /** SVG stage width in pixels. */
  width: number;
  /** SVG stage height in pixels. */
  height: number;
  /** Playback rate (frames per second). Defaults to 30. */
  fps?: number;
  /** Page title. */
  title?: string;
  /** One-line caption shown under the title. */
  subtitle?: string;
  /** Background color for the SVG stage. Defaults to '#fff'. */
  background?: string;
  /** If true, writes a one-line tick summary to stderr each frame. */
  liveTickLine?: boolean;
  /** Record only every Nth tick (default 1). */
  recordEveryTicks?: number;
  /** Visual blocks are appended to every HTML/animation frame. */
  visualBlocks?: readonly VisualBlockRenderable[];
}

export class FrameRecorder {
  private readonly opts: Required<Omit<FrameRecorderOpts, 'htmlPath' | 'subtitle' | 'visualBlocks'>> &
                          Pick<FrameRecorderOpts, 'htmlPath' | 'subtitle'> & {
                            visualBlocks: readonly VisualBlockRenderable[];
                          };
  private readonly stream: fs.WriteStream;
  private charts: ChartSpec[] = [];
  private frameCount = 0;
  private lastLiveLine = '';

  constructor(opts: FrameRecorderOpts) {
    this.opts = {
      htmlPath: opts.htmlPath,
      subtitle: opts.subtitle,
      framesPath: opts.framesPath,
      width: opts.width,
      height: opts.height,
      fps: opts.fps ?? 30,
      title: opts.title ?? 'Simulation',
      background: opts.background ?? '#ffffff',
      liveTickLine: opts.liveTickLine ?? false,
      recordEveryTicks: opts.recordEveryTicks ?? 1,
      visualBlocks: opts.visualBlocks ?? [],
    };
    fs.mkdirSync(path.dirname(this.opts.framesPath), {recursive: true});
    this.stream = fs.createWriteStream(this.opts.framesPath, {flags: 'w'});
    // Write the header line: kind=animation-header.
    this.stream.write(JSON.stringify({
      kind: 'animation-header',
      width: this.opts.width,
      height: this.opts.height,
      fps: this.opts.fps,
      title: this.opts.title,
      subtitle: this.opts.subtitle,
      background: this.opts.background,
    }) + '\n');
  }

  /**
   * Record a frame. The build callback is only invoked if this tick is
   * eligible for recording (`tick % recordEveryTicks === 0`); cheap when
   * filtered out. Pass a builder that returns the shapes (and optional
   * caption) for THIS tick only — the recorder fills in t and tick.
   */
  frame(t: number, tick: number, build: () => {shapes: Frame['shapes']; caption?: string}): void {
    if (tick % this.opts.recordEveryTicks !== 0) return;
    const built = build();
    const visualShapes = this.opts.visualBlocks.length > 0
      ? renderVisualBlocks(this.opts.visualBlocks, {tick, time: t, stageWidth: this.opts.width, stageHeight: this.opts.height})
      : [];
    const f: Frame = {t, tick, shapes: built.shapes.concat(visualShapes), ...(built.caption ? {caption: built.caption} : {})};
    this.stream.write(JSON.stringify({kind: 'animation-frame', ...f}) + '\n');
    this.frameCount++;
    if (this.opts.liveTickLine) this.writeLiveLine(t, tick, built.caption);
  }

  /** Set the global time-series chart panels. May be called any time before finish(). */
  setCharts(charts: ChartSpec[]): void {
    this.charts = charts;
  }

  /** Add a single chart panel. */
  addChart(c: ChartSpec): void {
    this.charts.push(c);
  }

  /** Number of frames recorded so far. */
  getFrameCount(): number {
    return this.frameCount;
  }

  /**
   * Flush the JSONL stream, write the HTML output (if configured), and
   * return the in-memory Animation object.
   */
  async finish(): Promise<Animation> {
    if (this.charts.length > 0) {
      this.stream.write(JSON.stringify({kind: 'animation-charts', charts: this.charts}) + '\n');
    }
    await new Promise<void>(resolve => this.stream.end(() => resolve()));

    const anim = readAnimation(this.opts.framesPath);
    if (this.opts.htmlPath) {
      fs.mkdirSync(path.dirname(this.opts.htmlPath), {recursive: true});
      fs.writeFileSync(this.opts.htmlPath, buildHTML(anim));
    }

    if (this.opts.liveTickLine) {
      // End the live line with a newline so subsequent stderr output is clean.
      process.stderr.write('\n');
    }
    return anim;
  }

  private writeLiveLine(t: number, tick: number, caption?: string): void {
    if (!process.stderr.isTTY) return;
    const cap = caption ? `  ${caption}` : '';
    const line = `[anim] t=${t.toFixed(2)}  tick=${tick}  frames=${this.frameCount}${cap}`;
    // Pad to overwrite previous line if shorter.
    const padding = Math.max(0, this.lastLiveLine.length - line.length);
    process.stderr.write(`\r${line}${' '.repeat(padding)}`);
    this.lastLiveLine = line;
  }
}

/**
 * Reconstruct an Animation from a JSONL frames file. Tolerant of
 * unknown event kinds (so simulations can interleave their normal
 * observability events into the same file in the future).
 */
export function readAnimation(framesPath: string): Animation {
  const raw = fs.readFileSync(framesPath, 'utf8');
  const lines = raw.split('\n');
  let header: any = null;
  const frames: Frame[] = [];
  let charts: ChartSpec[] | undefined;
  for (const line of lines) {
    if (!line) continue;
    let ev: any;
    try { ev = JSON.parse(line); } catch (e) {
      throw new Error(`malformed JSONL in ${framesPath}: ${(e as Error).message}`);
    }
    if (ev.kind === 'animation-header') header = ev;
    else if (ev.kind === 'animation-frame') frames.push({t: ev.t, tick: ev.tick, shapes: ev.shapes, caption: ev.caption});
    else if (ev.kind === 'animation-charts') charts = ev.charts;
  }
  if (header === null) throw new Error(`${framesPath} contains no animation-header event`);
  return {
    width: header.width,
    height: header.height,
    fps: header.fps,
    title: header.title,
    subtitle: header.subtitle,
    background: header.background,
    frames,
    charts,
  };
}
