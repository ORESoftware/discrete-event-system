'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/scenes/factmachine_scene.rs
// - Keep buildFactMachineFrame/buildFactMachineCharts as module helpers that return Frame/ChartSpec serde structs.
// - ArchitectureFrameArgs, FactMachineParams, and FactMachineResult should be nominal Rust structs instead of structural object shapes.
// - If a scene builder is wired into the DES graph, introduce a FactMachineSceneTransform implementing PureTransform::transform.
// - Local color/draw helpers stay private; arrays of Shape become Vec<Shape> and optional labels become Option<String>.

// =============================================================================
// FactMachine animation scene — REWRITTEN to put the DES ARCHITECTURE
// front-and-centre.
//
// Each frame is divided into two regions:
//
//   LEFT (560 × 720) — STATION GRAPH
//     Five stations laid out as labelled boxes:
//       NoiseTrader → Market ← Bettor      (Census taps Market)
//                              ↓
//                         Resolution        (fires at t = T only)
//     Movables (orders, votes) travel along the edges as small coloured
//     dots whose position is interpolated by an intra-tick "phase"
//     parameter ∈ [0, 1] so the player visibly sees them flow.
//     The active station for the current phase is highlighted.
//
//   RIGHT (520 × 720) — ANALYTICS
//     Belief histogram, price + E[θ] line chart, order-flow & entropy
//     panels are kept (just compacted) so the analytics story still reads.
//
// SUB-TICK FRAMES
// ───────────────
// `main-factmachine.ts` now emits 5 sub-frames per simulation tick, each
// representing one PHASE of the per-tick orchestration:
//   phase 0 → noise orders leaving NoiseTrader, in flight to Market
//   phase 1 → noise batch settled; Census reads prices/belief
//   phase 2 → Bettor reads prices; emits its order, in flight to Market
//   phase 3 → bettor's order settled; settlement applied
//   phase 4 → (only at t=T) ResolutionStation fires; voter ballots flow
// =============================================================================

import type {ChartSpec, Frame, Shape} from '../types';
import type {FactMachineParams, FactMachineResult} from '../../main-factmachine';

export const STAGE_W = 1080;
export const STAGE_H = 720;

// Architecture panel.
const ARCH_X = 20, ARCH_Y = 40, ARCH_W = 540, ARCH_H = 660;
// Analytics panel (right side).
const HIST_X = 580, HIST_Y = 60, HIST_W = 480, HIST_H = 220;
const PRICE_X = 580, PRICE_Y = 300, PRICE_W = 480, PRICE_H = 200;
const FLOW_X = 580, FLOW_Y = 520, FLOW_W = 232, FLOW_H = 180;
const ENT_X = 828, ENT_Y = 520, ENT_W = 232, ENT_H = 180;

// Station boxes inside ARCH panel.
const NOISE_X = ARCH_X + 30,  NOISE_Y = ARCH_Y + 60,  STATION_W = 150, STATION_H = 90;
const MARKET_X = ARCH_X + 280, MARKET_Y = ARCH_Y + 240;
const BETTOR_X = ARCH_X + 30,  BETTOR_Y = ARCH_Y + 240;
const CENSUS_X = ARCH_X + 280, CENSUS_Y = ARCH_Y + 60;
const RESOL_X  = ARCH_X + 280, RESOL_Y  = ARCH_Y + 440;

function viridis(t: number): string {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + (253 - 68) * t);
  const g = Math.round(1 + (231 - 1) * Math.sqrt(t));
  const b = Math.round(84 + (37 - 84) * t * t);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

/** Draw a labelled station box. Optionally highlighted (active this phase). */
function drawStation(shapes: Shape[], x: number, y: number, w: number, h: number,
                     title: string, lines: string[], active: boolean): void {
  shapes.push({kind: 'rect', x, y, w, h,
               fill: active ? '#fef3c7' : '#1e293b',
               stroke: active ? '#f59e0b' : '#475569',
               strokeWidth: active ? 3 : 1.5, rx: 8});
  shapes.push({kind: 'text', x: x + w / 2, y: y + 22, text: title,
               fontSize: 13, fill: active ? '#92400e' : '#fde68a',
               fontWeight: 'bold', anchor: 'middle'});
  for (let i = 0; i < lines.length; i++) {
    shapes.push({kind: 'text', x: x + w / 2, y: y + 44 + i * 16, text: lines[i],
                 fontSize: 11, fill: active ? '#1f2937' : '#cbd5e1', anchor: 'middle'});
  }
}

/** Draw an edge with optional in-flight movables. progress ∈ [0, 1] is how
 *  far along the edge each dot has travelled. */
function drawEdge(shapes: Shape[], x1: number, y1: number, x2: number, y2: number,
                  label: string, dotCount: number, dotColor: string, progress: number): void {
  shapes.push({kind: 'line', x1, y1, x2, y2, stroke: '#64748b', strokeWidth: 1.5, opacity: 0.7});
  // Arrow tip.
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len > 0) {
    const ux = dx / len, uy = dy / len;
    const ax = x2 - 8 * ux, ay = y2 - 8 * uy;
    const px = -uy, py = ux;
    shapes.push({kind: 'line', x1: ax, y1: ay, x2: ax + 4 * px, y2: ay + 4 * py,
                 stroke: '#64748b', strokeWidth: 1.5});
    shapes.push({kind: 'line', x1: ax, y1: ay, x2: ax - 4 * px, y2: ay - 4 * py,
                 stroke: '#64748b', strokeWidth: 1.5});
  }
  // Label at midpoint.
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  shapes.push({kind: 'rect', x: mx - 50, y: my - 8, w: 100, h: 16,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 3, opacity: 0.9});
  shapes.push({kind: 'text', x: mx, y: my + 4, text: label, fontSize: 10,
               fill: '#fde68a', anchor: 'middle'});
  // Movable dots: spread along the edge, centred at `progress`.
  const N = Math.min(dotCount, 8);    // cap visual count
  for (let i = 0; i < N; i++) {
    const t = Math.max(0.02, Math.min(0.98,
      progress + (i - (N - 1) / 2) * 0.04));
    const cx = x1 + (x2 - x1) * t;
    const cy = y1 + (y2 - y1) * t;
    shapes.push({kind: 'circle', x: cx, y: cy, r: 3.5,
                 fill: dotColor, stroke: '#0b1220', strokeWidth: 0.5});
  }
}

export interface ArchitectureFrameArgs {
  /** integer simulation tick t ∈ [0, T] this frame belongs to. */
  tick: number;
  /** sub-tick phase index 0..4 (see header comment). */
  phase: 0 | 1 | 2 | 3 | 4;
  /** Number of noise orders that flow this tick. */
  noiseOrderCount: number;
  /** Yes orders / total this tick (informational only). */
  noiseYes: number;
  /** Total orders this tick. */
  noiseTotal: number;
  /** Bettor's action this tick (-1 = hold). */
  bettorAction: number;
  /** Number of voters at resolution. */
  voterCount: number;
  /** Final resolution outcome (only meaningful at phase 4 of last tick). */
  resolutionOutcome?: number;
  /** Final voteFraction (phase 4 of last tick only). */
  voteFraction?: number;
  /** Belief weights at this tick. */
  beliefWeights: ReadonlyArray<number>;
  /** Prices vector at this tick. */
  prices: ReadonlyArray<number>;
}

export function buildFactMachineFrame(
  tick: number,
  beliefWeights: ReadonlyArray<number>,
  result: FactMachineResult,
  params: FactMachineParams,
  /** Optional architecture sub-tick info; if absent, defaults to phase=1
   *  (between-phases steady state) and no movables in flight. */
  arch?: Partial<ArchitectureFrameArgs>,
): Omit<Frame, 't' | 'tick'> {
  const shapes: Shape[] = [];

  // ============ ARCHITECTURE (left panel) ============
  const phase = (arch?.phase ?? 1) as 0 | 1 | 2 | 3 | 4;
  const isLast = tick >= params.T;
  // Activity flags per phase.
  const noiseActive    = phase === 0;
  const marketActive   = phase === 1 || phase === 3 || phase === 4;
  const censusActive   = phase === 1;
  const bettorActive   = phase === 2;
  const resolutionActive = phase === 4 && isLast;

  // Background panel.
  shapes.push({kind: 'rect', x: ARCH_X, y: ARCH_Y, w: ARCH_W, h: ARCH_H,
               fill: '#0b1220', stroke: '#334155', strokeWidth: 1.5, rx: 6});
  const phaseLabel = ['phase 0: noise → market',
                      'phase 1: noise settles · census reads',
                      'phase 2: bettor reads · sends order',
                      'phase 3: bettor settles',
                      'phase 4: RESOLUTION (votes flow)'][phase];
  shapes.push({kind: 'text', x: ARCH_X + ARCH_W / 2, y: ARCH_Y + 24,
               text: `DES architecture — t = ${tick}/${params.T} — ${phaseLabel}`,
               fontSize: 14, fill: '#f1f5f9', fontWeight: 'bold', anchor: 'middle'});

  // ── Stations ──
  const noiseOrderCount = arch?.noiseOrderCount ?? 0;
  const bettorAction = arch?.bettorAction ?? -1;
  const yes = arch?.noiseYes ?? 0;
  const tot = arch?.noiseTotal ?? 0;
  drawStation(shapes, NOISE_X, NOISE_Y, STATION_W, STATION_H,
    'NoiseTrader', [`K = ${params.K_noise}`, `informedness ${params.informedness.toFixed(2)}`,
                    tot > 0 ? `last batch ${yes}/${tot}` : 'idle'], noiseActive);
  drawStation(shapes, MARKET_X, MARKET_Y, STATION_W, STATION_H,
    'Market (LMSR)', [`P(YES) = ${(arch?.prices?.[0] ?? 0.5).toFixed(3)}`,
                      `liq L = ${params.liquidity}`,
                      `b = ${(params.liquidity / Math.log(params.marketType === 'binary' ? 2 : params.thetaBins)).toFixed(2)}`], marketActive);
  let bMean = 0;
  for (let i = 0; i < beliefWeights.length; i++) bMean += beliefWeights[i] * (i / (beliefWeights.length - 1));
  drawStation(shapes, BETTOR_X, BETTOR_Y, STATION_W, STATION_H,
    'Bettor', [`policy = ${params.policy}`, `E[θ] = ${bMean.toFixed(3)}`,
               bettorAction < 0 ? 'action = hold' : `action = buy ${bettorAction}`], bettorActive);
  drawStation(shapes, CENSUS_X, CENSUS_Y, STATION_W, STATION_H,
    'Census', [`snapshots`, `prices + b(θ)`, `for trace`], censusActive);
  drawStation(shapes, RESOL_X, RESOL_Y, STATION_W, STATION_H,
    'Resolution', isLast && phase === 4
      ? [`fires NOW`, `${params.N_voters} votes`,
         arch?.resolutionOutcome !== undefined
           ? `outcome = ${arch.resolutionOutcome === 0 ? 'YES' : 'NO'}` : '']
      : [`pending`, `fires at t=${params.T}`, `${params.N_voters} voters`],
    resolutionActive);

  // ── Edges with movables ──
  const cx = (x: number, w: number) => x + w / 2;
  const cy = (y: number, h: number) => y + h / 2;
  // 1. NoiseTrader → Market
  if (phase === 0) {
    drawEdge(shapes, cx(NOISE_X, STATION_W), NOISE_Y + STATION_H,
             cx(MARKET_X, STATION_W), MARKET_Y,
             `${noiseOrderCount} noise orders`, noiseOrderCount, '#22d3ee', 0.5);
  } else {
    drawEdge(shapes, cx(NOISE_X, STATION_W), NOISE_Y + STATION_H,
             cx(MARKET_X, STATION_W), MARKET_Y,
             tot > 0 ? `${tot} settled` : `idle`, 0, '#22d3ee', 0);
  }
  // 2. Market → Census (read).
  if (phase === 1) {
    drawEdge(shapes, MARKET_X + STATION_W / 2, MARKET_Y,
             CENSUS_X + STATION_W / 2, CENSUS_Y + STATION_H,
             'snapshot', 1, '#a78bfa', 0.4);
  } else {
    drawEdge(shapes, MARKET_X + STATION_W / 2, MARKET_Y,
             CENSUS_X + STATION_W / 2, CENSUS_Y + STATION_H,
             'reads', 0, '#a78bfa', 0);
  }
  // 3. Market → Bettor (read prices)
  if (phase === 2) {
    drawEdge(shapes, MARKET_X, MARKET_Y + STATION_H / 2,
             BETTOR_X + STATION_W, BETTOR_Y + STATION_H / 2,
             'reads prices', 1, '#fb7185', 0.5);
  }
  // 4. Bettor → Market (send order)
  if (phase === 2 && bettorAction >= 0) {
    drawEdge(shapes, BETTOR_X + STATION_W, BETTOR_Y + 20,
             MARKET_X, MARKET_Y + 20,
             `1 order (buy ${bettorAction === 0 ? 'YES' : bettorAction === 1 ? 'NO' : '#' + bettorAction})`,
             1, '#f472b6', 0.5);
  }
  // 5. Resolution receives votes (only on the final phase 4 of the last tick)
  if (resolutionActive) {
    drawEdge(shapes, MARKET_X + STATION_W / 2, MARKET_Y + STATION_H,
             RESOL_X + STATION_W / 2, RESOL_Y,
             `${params.N_voters} votes`, params.N_voters, '#fda4af', 0.6);
  }

  // ============ ANALYTICS (right panel) ============
  const K = beliefWeights.length;

  // Belief histogram (compact).
  shapes.push({kind: 'rect', x: HIST_X - 6, y: HIST_Y - 6, w: HIST_W + 12, h: HIST_H + 12,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: HIST_X, y: HIST_Y - 12,
               text: `Bettor's belief b(θ) — t = ${tick}`,
               fontSize: 12, fill: '#e2e8f0', fontWeight: 'bold'});
  let maxW = 0;
  for (const w of beliefWeights) if (w > maxW) maxW = w;
  maxW = Math.max(maxW, 1.2 / K);
  const barW = HIST_W / K;
  for (let i = 0; i < K; i++) {
    const θ = i / (K - 1);
    const h = (beliefWeights[i] / maxW) * (HIST_H - 30);
    shapes.push({kind: 'rect', x: HIST_X + i * barW + barW * 0.05,
                 y: HIST_Y + (HIST_H - 30) - h + 4,
                 w: barW * 0.9, h: Math.max(0.5, h),
                 fill: viridis(θ)});
  }
  // True θ marker.
  const truthX = HIST_X + params.trueTheta * HIST_W;
  shapes.push({kind: 'line', x1: truthX, y1: HIST_Y + 4, x2: truthX, y2: HIST_Y + HIST_H - 26,
               stroke: '#dc2626', strokeWidth: 2, dasharray: '4 3'});
  shapes.push({kind: 'text', x: truthX + 4, y: HIST_Y + 16, fontSize: 10, fill: '#fca5a5',
               text: `true θ = ${params.trueTheta.toFixed(2)}`});
  // E_b[θ] marker.
  const meanX = HIST_X + bMean * HIST_W;
  shapes.push({kind: 'line', x1: meanX, y1: HIST_Y + 4, x2: meanX, y2: HIST_Y + HIST_H - 26,
               stroke: '#60a5fa', strokeWidth: 2});
  shapes.push({kind: 'text', x: meanX + 4, y: HIST_Y + 32, fontSize: 10, fill: '#93c5fd',
               text: `E[θ] = ${bMean.toFixed(3)}`});

  // Price + E[θ] line chart (compact).
  shapes.push({kind: 'rect', x: PRICE_X - 6, y: PRICE_Y - 6, w: PRICE_W + 12, h: PRICE_H + 12,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: PRICE_X, y: PRICE_Y - 12,
               text: `P(YES) red, E[θ] blue — through t = ${tick}`,
               fontSize: 12, fill: '#e2e8f0', fontWeight: 'bold'});
  const yTrue = PRICE_Y + (1 - params.trueTheta) * (PRICE_H - 18) + 4;
  shapes.push({kind: 'line', x1: PRICE_X, y1: yTrue, x2: PRICE_X + PRICE_W, y2: yTrue,
               stroke: '#dc2626', strokeWidth: 1, dasharray: '3 3'});
  const T = result.priceHistory.length - 1;
  const xAt = (i: number) => PRICE_X + (i / Math.max(1, T)) * PRICE_W;
  const yAt = (v: number) => PRICE_Y + (1 - v) * (PRICE_H - 18) + 4;
  const upTo = Math.min(tick, T);
  const priceScalar = (t: number): number => {
    const ph = result.priceHistory[t];
    if (ph.length === 2) return ph[0];
    let bestJ = 0; for (let j = 1; j < ph.length; j++) if (ph[j] > ph[bestJ]) bestJ = j;
    return (bestJ + 0.5) / ph.length;
  };
  let priceD = '', meanD = '';
  for (let i = 0; i <= upTo; i++) {
    priceD += (i === 0 ? 'M' : ' L') + xAt(i).toFixed(1) + ' ' + yAt(priceScalar(i)).toFixed(1);
    meanD  += (i === 0 ? 'M' : ' L') + xAt(i).toFixed(1) + ' ' + yAt(result.beliefMean[i]).toFixed(1);
  }
  shapes.push({kind: 'path', d: priceD, stroke: '#dc2626', strokeWidth: 2, fill: 'none'});
  shapes.push({kind: 'path', d: meanD,  stroke: '#60a5fa', strokeWidth: 2, fill: 'none'});

  // Order flow ratio panel.
  shapes.push({kind: 'rect', x: FLOW_X - 6, y: FLOW_Y - 6, w: FLOW_W + 12, h: FLOW_H + 12,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: FLOW_X, y: FLOW_Y - 12,
               text: `Order flow YES/total`,
               fontSize: 11, fill: '#e2e8f0', fontWeight: 'bold'});
  const flowTrue = FLOW_Y + (1 - params.trueTheta) * (FLOW_H - 18) + 4;
  shapes.push({kind: 'line', x1: FLOW_X, y1: flowTrue, x2: FLOW_X + FLOW_W, y2: flowTrue,
               stroke: '#dc2626', strokeWidth: 1, dasharray: '3 3'});
  for (let i = 0; i <= Math.min(tick - 1, result.yesOrdersHistory.length - 1); i++) {
    const ratio = result.totalOrdersHistory[i] > 0
      ? result.yesOrdersHistory[i] / result.totalOrdersHistory[i] : 0.5;
    const x = FLOW_X + ((i + 0.5) / Math.max(1, T)) * FLOW_W;
    const y = FLOW_Y + (1 - ratio) * (FLOW_H - 18) + 4;
    shapes.push({kind: 'circle', x, y, r: 2.5, fill: '#fbbf24'});
  }

  // Entropy panel.
  shapes.push({kind: 'rect', x: ENT_X - 6, y: ENT_Y - 6, w: ENT_W + 12, h: ENT_H + 12,
               fill: '#0f172a', stroke: '#334155', strokeWidth: 1, rx: 4});
  shapes.push({kind: 'text', x: ENT_X, y: ENT_Y - 12,
               text: `Belief entropy H(b)`,
               fontSize: 11, fill: '#e2e8f0', fontWeight: 'bold'});
  let maxH = Math.log(K);
  let entD = '';
  for (let i = 0; i <= upTo; i++) {
    const x = ENT_X + (i / Math.max(1, T)) * ENT_W;
    const y = ENT_Y + (1 - result.beliefEntropy[i] / maxH) * (ENT_H - 18) + 4;
    entD += (i === 0 ? 'M' : ' L') + x.toFixed(1) + ' ' + y.toFixed(1);
  }
  shapes.push({kind: 'path', d: entD, stroke: '#a78bfa', strokeWidth: 2, fill: 'none'});

  return {
    shapes,
    caption: `t=${tick} ${phaseLabel}`,
  };
}

export function buildFactMachineCharts(_r: FactMachineResult): ChartSpec[] {
  return [];
}
