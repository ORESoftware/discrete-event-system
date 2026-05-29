'use strict';

// JSON adapters for the domain application model pack.

import {DESRuntimeConfig, ParamSchema} from '../des-spec';
import {registerModel} from '../des-registry';
import {FrameRecorder} from '../../animation/frame-recorder';
import {Shape} from '../../animation/types';
import {
  ActiveLearningParams,
  ActiveLearningResult,
  AdaptiveFuzzyControlParams,
  AdaptiveFuzzyControlResult,
  BuyerAwareDynamicPricingParams,
  BuyerAwareDynamicPricingResult,
  DecisionScienceParams,
  DecisionScienceResult,
  DomainModelResult,
  DomainTrace,
  EnergyParams,
  EnergyResult,
  FinancialControlParams,
  FinancialControlResult,
  LogisticsRoutingParams,
  LogisticsRoutingResult,
  ManufacturingParams,
  ManufacturingResult,
  OperationsParams,
  OperationsResult,
  RevenueManagementParams,
  RevenueManagementResult,
  SupplyChainParams,
  SupplyChainResult,
  runActiveLearningAcquisition,
  runAdaptiveFuzzyControl,
  runBottleneckProductionControl,
  runBuyerAwareDynamicPricing,
  runDynamicPricingRevenue,
  runEnergyStorageDispatch,
  runLogisticsRoutingHeuristics,
  runPortfolioDrawdownControl,
  runSupplyChainRiskPooling,
  runVisualDecisionFrontier,
  runWorkforceServiceOperations,
} from '../domain-application-models';
import {csvRow, framesPath, jsonCsvCell, writeCsvLines} from './adapter-utils';

const emptySchema: ParamSchema = {kind: 'object', fields: {}, required: []};

registerModel<AdaptiveFuzzyControlParams, AdaptiveFuzzyControlResult>({
  id: 'adaptive-fuzzy-control',
  description: 'Adaptive fuzzy control: tune fuzzy controller candidates over a first-order plant station graph.',
  schema: {
    kind: 'object',
    fields: {
      steps: {kind: 'number', integer: true, min: 1, default: 140},
      dt: {kind: 'number', min: 1e-9, default: 0.1},
      setpoint: {kind: 'number', default: 22},
      initialTemp: {kind: 'number', default: 16},
      outsideTemp: {kind: 'number', default: 8},
      disturbance: {kind: 'number', min: 0, default: 0.15},
    },
    required: [],
  },
  run: runAdaptiveFuzzyControl,
  summarize: domainSummary('ADAPTIVE FUZZY CONTROL'),
  animate: animateDomainModel('Adaptive fuzzy control'),
  writeCsv: writeDomainCsv,
});

registerModel<LogisticsRoutingParams, LogisticsRoutingResult>({
  id: 'logistics-routing-heuristics',
  description: 'Logistics routing: compare nearest-neighbor, sweep, and savings heuristics as movable candidate plans.',
  schema: {
    kind: 'object',
    fields: {vehicleCapacity: {kind: 'number', min: 1e-9, default: 7}},
    required: [],
  },
  run: runLogisticsRoutingHeuristics,
  summarize: domainSummary('LOGISTICS ROUTING HEURISTICS'),
  animate: animateDomainModel('Logistics routing heuristics'),
  writeCsv: writeDomainCsv,
});

registerModel<ManufacturingParams, ManufacturingResult>({
  id: 'bottleneck-production-control',
  description: 'Manufacturing production control: bottleneck-buffer-rope and adaptive expedite policies.',
  schema: {
    kind: 'object',
    fields: {
      horizon: {kind: 'number', integer: true, min: 1, default: 18},
      dailyDemand: {kind: 'number', min: 0, default: 8},
    },
    required: [],
  },
  run: runBottleneckProductionControl,
  summarize: domainSummary('BOTTLENECK PRODUCTION CONTROL'),
  animate: animateDomainModel('Bottleneck production control'),
  writeCsv: writeDomainCsv,
});

registerModel<SupplyChainParams, SupplyChainResult>({
  id: 'supply-chain-risk-pooling',
  description: 'Supply chain management: multi-echelon risk-pooling reorder policy candidates.',
  schema: {
    kind: 'object',
    fields: {horizon: {kind: 'number', integer: true, min: 1, default: 20}},
    required: [],
  },
  run: runSupplyChainRiskPooling,
  summarize: domainSummary('SUPPLY CHAIN RISK POOLING'),
  animate: animateDomainModel('Supply chain risk pooling'),
  writeCsv: writeDomainCsv,
});

registerModel<OperationsParams, OperationsResult>({
  id: 'workforce-service-operations',
  description: 'Operations management: service-risk workforce roster heuristics with flex-pool control.',
  schema: {
    kind: 'object',
    fields: {overtimeCost: {kind: 'number', min: 1e-9, default: 18}},
    required: [],
  },
  run: runWorkforceServiceOperations,
  summarize: domainSummary('WORKFORCE SERVICE OPERATIONS'),
  animate: animateDomainModel('Workforce service operations'),
  writeCsv: writeDomainCsv,
});

registerModel<FinancialControlParams, FinancialControlResult>({
  id: 'portfolio-drawdown-control',
  description: 'Financial engineering: CPPI-style portfolio drawdown control candidates.',
  schema: {
    kind: 'object',
    fields: {initialWealth: {kind: 'number', min: 1e-9, default: 100}},
    required: [],
  },
  run: runPortfolioDrawdownControl,
  summarize: domainSummary('PORTFOLIO DRAWDOWN CONTROL'),
  animate: animateDomainModel('Portfolio drawdown control'),
  writeCsv: writeDomainCsv,
});

registerModel<RevenueManagementParams, RevenueManagementResult>({
  id: 'dynamic-pricing-revenue',
  description: 'Revenue management: dynamic pricing policies using scarcity and demand smoothing.',
  schema: {
    kind: 'object',
    fields: {capacity: {kind: 'number', min: 1e-9, default: 120}},
    required: [],
  },
  run: runDynamicPricingRevenue,
  summarize: domainSummary('DYNAMIC PRICING REVENUE MANAGEMENT'),
  animate: animateDomainModel('Dynamic pricing revenue management'),
  writeCsv: writeDomainCsv,
});

registerModel<BuyerAwareDynamicPricingParams, BuyerAwareDynamicPricingResult>({
  id: 'buyer-aware-dynamic-pricing',
  description: 'Revenue management: buyer-aware dynamic pricing with privacy, fairness, inventory, and retention guardrails.',
  schema: {
    kind: 'object',
    fields: {
      horizon: {kind: 'number', integer: true, min: 1, default: 12},
      initialInventory: {kind: 'number', min: 1e-9, default: 160},
      privacyBudget: {kind: 'number', min: 0, default: 0},
      fairnessTolerance: {kind: 'number', min: 0, default: 0.18},
      sustainabilityWeight: {kind: 'number', min: 0, default: 120},
    },
    required: [],
  },
  run: runBuyerAwareDynamicPricing,
  summarize: domainSummary('BUYER-AWARE DYNAMIC PRICING'),
  animate: animateDomainModel('Buyer-aware dynamic pricing'),
  writeCsv: writeDomainCsv,
});

registerModel<EnergyParams, EnergyResult>({
  id: 'energy-storage-dispatch',
  description: 'Energy optimization: storage dispatch candidates for renewable integration and price arbitrage.',
  schema: {
    kind: 'object',
    fields: {batteryCapacity: {kind: 'number', min: 1e-9, default: 50}},
    required: [],
  },
  run: runEnergyStorageDispatch,
  summarize: domainSummary('ENERGY STORAGE DISPATCH'),
  animate: animateDomainModel('Energy storage dispatch'),
  writeCsv: writeDomainCsv,
});

registerModel<ActiveLearningParams, ActiveLearningResult>({
  id: 'active-learning-acquisition',
  description: 'Machine/statistical learning: active-learning acquisition policies over unlabeled data movables.',
  schema: {
    kind: 'object',
    fields: {budget: {kind: 'number', min: 1e-9, default: 9}},
    required: [],
  },
  run: runActiveLearningAcquisition,
  summarize: domainSummary('ACTIVE LEARNING ACQUISITION'),
  animate: animateDomainModel('Active learning acquisition'),
  writeCsv: writeDomainCsv,
});

registerModel<DecisionScienceParams, DecisionScienceResult>({
  id: 'visual-decision-frontier',
  description: 'Decision science: MCDA frontier scoring with visualization-ready alternatives and weights.',
  schema: {
    ...emptySchema,
    fields: {riskWeight: {kind: 'number', min: 0, default: 0.35}},
  },
  run: runVisualDecisionFrontier,
  summarize: domainSummary('VISUAL DECISION FRONTIER'),
  animate: animateDomainModel('Visual decision frontier'),
  writeCsv: writeDomainCsv,
});

function domainSummary(title: string): (result: DomainModelResult<unknown>) => string {
  return (result: DomainModelResult<unknown>) => [
    title,
    '----------------------------------------',
    `  Category:       ${result.category}`,
    `  Best plan:      ${result.best.candidateId}`,
    `  Objective:      ${result.best.objective.toFixed(6)}`,
    `  Metrics:        ${metricsLine(result.best.metrics)}`,
    `  Candidates:     ${result.candidates.length}`,
    `  Stations:       ${result.topology.stations.join(' -> ')}`,
    `  Movables:       ${result.topology.movables.join(', ')}`,
  ].join('\n');
}

function writeDomainCsv(result: DomainModelResult<unknown>, csvPath: string): void {
  const lines = [csvRow(['candidate_id', 'objective', 'feasible', 'metrics', 'plan'])];
  for (const row of result.candidates) {
    lines.push([
      csvRow([row.candidateId, row.objective, row.feasible]),
      jsonCsvCell(row.metrics),
      jsonCsvCell(row.plan),
    ].join(','));
  }
  writeCsvLines(csvPath, lines);
}

function metricsLine(metrics: Record<string, number | string | boolean>): string {
  return Object.entries(metrics).slice(0, 4).map(([k, v]) => {
    if (typeof v === 'number') return `${k}=${formatMetric(v)}`;
    return `${k}=${v}`;
  }).join(', ');
}

function formatMetric(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  if (Math.abs(v) >= 0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function animateDomainModel(title: string): (result: DomainModelResult<unknown>, params: unknown, runtime: DESRuntimeConfig) => Promise<void> {
  return async (result: DomainModelResult<unknown>, _params: unknown, runtime: DESRuntimeConfig): Promise<void> => {
    const {htmlPath, frames} = framesPath(runtime, result.modelId);
    const rec = new FrameRecorder({
      framesPath: frames,
      htmlPath,
      width: 940,
      height: 560,
      fps: 3,
      title,
      subtitle: 'Scenario, candidate plans, evaluated plans, and result sink as DES movables',
      background: '#f8fafc',
    });
    const candidates = result.candidates.slice().sort((a, b) => b.objective - a.objective);
    for (let i = 0; i < candidates.length; i++) {
      rec.frame(i, i, () => buildDomainFrame(result, candidates, i));
    }
    rec.setCharts(domainCharts(result, candidates));
    await rec.finish();
  };
}

function buildDomainFrame(
  result: DomainModelResult<unknown>,
  candidates: DomainModelResult<unknown>['candidates'],
  activeIndex: number,
): {shapes: Shape[]; caption: string} {
  const shapes: Shape[] = [{kind: 'rect', x: 0, y: 0, w: 940, h: 560, fill: '#f8fafc'}];
  const active = candidates[activeIndex];
  const best = result.best;
  const nodes = [
    {id: 'source', label: 'ScenarioSource', x: 50, y: 98, w: 150, h: 58, fill: '#dbeafe'},
    {id: 'generator', label: 'CandidateGenerator', x: 255, y: 98, w: 172, h: 58, fill: '#ede9fe'},
    {id: 'evaluator', label: 'PlanEvaluator', x: 485, y: 98, w: 150, h: 58, fill: '#dcfce7'},
    {id: 'sink', label: 'ResultSink', x: 700, y: 98, w: 130, h: 58, fill: '#f1f5f9'},
  ];
  const edges = [
    {from: nodes[0], to: nodes[1], label: 'DomainScenarioToken'},
    {from: nodes[1], to: nodes[2], label: 'DomainPlanToken'},
    {from: nodes[2], to: nodes[3], label: 'DomainEvaluationToken'},
  ];
  shapes.push({kind: 'text', x: 44, y: 34, text: result.modelId, fontSize: 21, fill: '#0f172a', fontWeight: 'bold'});
  shapes.push({kind: 'text', x: 44, y: 58, text: `active=${active.candidateId}  best=${best.candidateId}  objective=${active.objective.toFixed(2)}`, fontSize: 13, fill: '#475569'});

  for (const edge of edges) {
    const edgeActive = edges.indexOf(edge) === activeIndex % edges.length;
    const x1 = edge.from.x + edge.from.w;
    const x2 = edge.to.x;
    const y = edge.from.y + edge.from.h / 2;
    shapes.push({kind: 'line', x1, y1: y, x2, y2: y, stroke: edgeActive ? '#7c3aed' : '#94a3b8', strokeWidth: edgeActive ? 4 : 2, opacity: edgeActive ? 0.95 : 0.65});
    shapes.push({kind: 'text', x: (x1 + x2) / 2, y: y - 13, text: edge.label, fontSize: 10, anchor: 'middle', fill: edgeActive ? '#5b21b6' : '#64748b'});
  }
  for (const node of nodes) {
    shapes.push({kind: 'rect', x: node.x, y: node.y, w: node.w, h: node.h, fill: node.fill, stroke: '#334155', strokeWidth: 1.2, rx: 7});
    shapes.push({kind: 'text', x: node.x + node.w / 2, y: node.y + 28, text: node.label, fontSize: 12, anchor: 'middle', fill: '#0f172a', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: node.x + node.w / 2, y: node.y + 46, text: 'station', fontSize: 10, anchor: 'middle', fill: '#475569'});
  }
  const activeEdge = edges[activeIndex % edges.length];
  shapes.push({
    kind: 'circle',
    x: (activeEdge.from.x + activeEdge.from.w + activeEdge.to.x) / 2,
    y: activeEdge.from.y + activeEdge.from.h / 2,
    r: 10,
    fill: '#7c3aed',
    stroke: '#ffffff',
    strokeWidth: 2,
    title: activeEdge.label,
  });

  drawCandidateBars(shapes, candidates, activeIndex);
  drawMetricPanel(shapes, active);
  return {shapes, caption: `${active.candidateId}: ${active.feasible ? 'feasible' : 'infeasible'} objective ${active.objective.toFixed(2)}`};
}

function drawCandidateBars(shapes: Shape[], candidates: DomainModelResult<unknown>['candidates'], activeIndex: number): void {
  const x = 48;
  const yBase = 315;
  const w = 540;
  const maxAbs = Math.max(1, ...candidates.map(row => Math.abs(row.objective)));
  shapes.push({kind: 'text', x, y: 202, text: 'Candidate objective comparison', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  for (let i = 0; i < candidates.length; i++) {
    const row = candidates[i];
    const barW = w * Math.abs(row.objective) / maxAbs;
    const y = 226 + i * 28;
    const fill = i === activeIndex ? '#7c3aed' : row.feasible ? '#059669' : '#dc2626';
    shapes.push({kind: 'rect', x, y, w: Math.max(2, barW), h: 18, fill, rx: 4, opacity: i === activeIndex ? 0.96 : 0.72});
    shapes.push({kind: 'text', x: x + 8, y: y + 13, text: row.candidateId, fontSize: 10, fill: '#ffffff', fontWeight: 'bold'});
    shapes.push({kind: 'text', x: x + Math.min(w + 12, Math.max(70, barW + 8)), y: y + 13, text: row.objective.toFixed(1), fontSize: 11, fill: '#334155'});
  }
  shapes.push({kind: 'line', x1: x, y1: yBase + 54, x2: x + w, y2: yBase + 54, stroke: '#cbd5e1', strokeWidth: 1});
}

function drawMetricPanel(shapes: Shape[], row: DomainModelResult<unknown>['candidates'][number]): void {
  const x = 630;
  const y = 214;
  shapes.push({kind: 'rect', x, y, w: 260, h: 172, fill: '#ffffff', stroke: '#cbd5e1', strokeWidth: 1, rx: 7});
  shapes.push({kind: 'text', x: x + 16, y: y + 25, text: 'Active evaluation', fontSize: 14, fill: '#0f172a', fontWeight: 'bold'});
  const entries = Object.entries(row.metrics).slice(0, 6);
  for (let i = 0; i < entries.length; i++) {
    const [key, value] = entries[i];
    const v = typeof value === 'number' ? formatMetric(value) : String(value);
    shapes.push({kind: 'text', x: x + 16, y: y + 52 + i * 19, text: `${key}: ${v}`, fontSize: 11, fill: '#334155'});
  }
}

function domainCharts(result: DomainModelResult<unknown>, candidates: DomainModelResult<unknown>['candidates']): Array<{x: number; y: number; w: number; h: number; title?: string; yLabel?: string; series: Array<{label: string; color: string; t: number[]; y: number[]}>}> {
  const charts = [
    {
      x: 48, y: 404, w: 250, h: 120, title: 'Candidate objective', yLabel: 'objective',
      series: [{label: 'objective', color: '#7c3aed', t: candidates.map((_row, i) => i), y: candidates.map(row => row.objective)}],
    },
  ];
  const trace = result.best.trace;
  if (!trace) return charts;
  const price = seriesIfPresent(trace, 'averagePrice', 'average price', '#2563eb');
  const inventory = seriesIfPresent(trace, 'inventory', 'inventory', '#059669');
  if (price || inventory) {
    charts.push({
      x: 345, y: 404, w: 250, h: 120, title: 'Best plan price/inventory', yLabel: 'level',
      series: [price, inventory].filter((x): x is {label: string; color: string; t: number[]; y: number[]} => x !== undefined),
    });
  }
  const fairness = seriesIfPresent(trace, 'fairnessSpread', 'fairness spread', '#dc2626');
  const retention = seriesIfPresent(trace, 'retentionIndex', 'retention index', '#0d9488');
  if (fairness || retention) {
    charts.push({
      x: 642, y: 404, w: 250, h: 120, title: 'Best plan guardrails', yLabel: 'index',
      series: [fairness, retention].filter((x): x is {label: string; color: string; t: number[]; y: number[]} => x !== undefined),
    });
  }
  return charts;
}

function seriesIfPresent(trace: DomainTrace, key: string, label: string, color: string): {label: string; color: string; t: number[]; y: number[]} | undefined {
  const y = trace.series[key];
  if (!y) return undefined;
  return {label, color, t: trace.t, y};
}
