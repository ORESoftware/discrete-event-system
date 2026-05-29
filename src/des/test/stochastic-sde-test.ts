'use strict';

// =============================================================================
// Unit tests for stochastic-sde.ts and sde-learning.ts.
// Run with:
//   ./node_modules/.bin/ts-node src/des/test/stochastic-sde-test.ts
// =============================================================================

import {runIterativeDES} from '../general/des-base/runner';
import {DESStation} from '../general/des-base/station';
import {Mulberry32} from '../general/control-systems/empirical-control';
import {
  EulerMaruyamaIntegrator,
  GeometricBrownianMotion,
  OrnsteinUhlenbeck,
  SdeChannels,
  SdeEstimateSinkStation,
  SdePlantStation,
  StochasticDcMotor,
} from '../general/control-systems/stochastic-sde';
import {
  DenoisingDiffusionModel,
  EnsembleKalmanFilter,
  EnsembleKalmanFilterStation,
  GbmFamily,
  Mlp,
  OuFamily,
  SdeMaximumLikelihoodEstimator,
} from '../general/control-systems/sde-learning';

let pass = 0, fail = 0;
function check(label: string, ok: boolean, detail = ''): void {
  if (ok) { pass++; console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`); }
  else    { fail++; console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`); }
}
function rel(a: number, b: number): number { return Math.abs(a - b) / Math.max(1e-9, Math.abs(b)); }

// -----------------------------------------------------------------------------
console.log('\n[1] Euler–Maruyama — GBM moments vs analytic');
// -----------------------------------------------------------------------------
{
  const gbm = new GeometricBrownianMotion(0.1, 0.3);
  const em = new EulerMaruyamaIntegrator();
  const x0 = 1, dt = 0.002, steps = 800, paths = 3000;
  let sum = 0, sumSq = 0;
  for (let p = 0; p < paths; p++) {
    const {path} = em.simulate(gbm, [x0], dt, steps, new Mulberry32(500 + p));
    const xT = path[path.length - 1][0];
    sum += xT; sumSq += xT * xT;
  }
  const T = dt * steps;
  const empMean = sum / paths;
  const empVar = sumSq / paths - empMean * empMean;
  check('1.1 E[X_T] ≈ analytic', rel(empMean, gbm.meanAt(x0, T)) < 0.03, `emp=${empMean.toFixed(4)} ana=${gbm.meanAt(x0, T).toFixed(4)}`);
  check('1.2 Var[X_T] ≈ analytic', rel(empVar, gbm.varAt(x0, T)) < 0.12, `emp=${empVar.toFixed(4)} ana=${gbm.varAt(x0, T).toFixed(4)}`);
  // Deterministic with a fixed seed.
  const a = em.simulate(gbm, [1], 0.01, 10, new Mulberry32(1)).path;
  const b = em.simulate(gbm, [1], 0.01, 10, new Mulberry32(1)).path;
  check('1.3 seeded paths reproducible', JSON.stringify(a) === JSON.stringify(b));
}

// -----------------------------------------------------------------------------
console.log('\n[2] Ornstein–Uhlenbeck — stationary variance');
// -----------------------------------------------------------------------------
{
  const ou = new OrnsteinUhlenbeck(1.0, 0, 0.5);
  const {path} = new EulerMaruyamaIntegrator().simulate(ou, [0], 0.01, 60000, new Mulberry32(3));
  const tail = path.slice(path.length / 2).map(x => x[0]);
  const mean = tail.reduce((a, v) => a + v, 0) / tail.length;
  const variance = tail.reduce((a, v) => a + (v - mean) ** 2, 0) / tail.length;
  check('2.1 stationary mean ≈ μ', Math.abs(mean - ou.stationaryMean()) < 0.05, `mean=${mean.toFixed(4)}`);
  check('2.2 stationary var ≈ σ²/2θ', rel(variance, ou.stationaryVariance()) < 0.15, `var=${variance.toFixed(4)} target=${ou.stationaryVariance().toFixed(4)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[3] ML-1: maximum-likelihood SDE parameter recovery');
// -----------------------------------------------------------------------------
{
  const gbm = new GeometricBrownianMotion(0.15, 0.25);
  const {times, path} = new EulerMaruyamaIntegrator().simulate(gbm, [1], 0.004, 6000, new Mulberry32(11));
  const fit = new SdeMaximumLikelihoodEstimator({iterations: 1000, learningRate: 0.05}).fit(new GbmFamily(), times, path);
  // σ (diffusion) is reliably identifiable from a single path; μ (drift) is not.
  check('3.1 recovers σ within 10%', rel(fit.params.sigma, 0.25) < 0.1, `sigma=${fit.params.sigma.toFixed(4)}`);
  check('3.2 μ estimate finite & near truth (loose)', Math.abs(fit.params.mu - 0.15) < 0.15, `mu=${fit.params.mu.toFixed(4)}`);
  // NLL at truth should not beat the fitted optimum.
  const est = new SdeMaximumLikelihoodEstimator();
  const nllTruth = est.negLogLikelihood(new GbmFamily(), [0.15, Math.log(0.25)], times, path);
  check('3.3 fitted NLL ≤ NLL(truth)', fit.finalNegLogLik <= nllTruth + 1e-6, `fit=${fit.finalNegLogLik.toFixed(2)} truth=${nllTruth.toFixed(2)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[4] ML-2: Ensemble Kalman Filter recovers a hidden state');
// -----------------------------------------------------------------------------
{
  const motor = new StochasticDcMotor({
    resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002,
    voltage: 12, currentNoise: 0.4, speedNoise: 0.5,
  });
  const dt = 0.01, steps = 400;
  const H = [[0, 1]];
  const plant = new SdePlantStation('plant', {system: motor, x0: [0, 0], dt, steps, observationMatrix: H, observationNoiseStd: [0.6], seed: 5});
  const filter = new EnsembleKalmanFilter(motor, dt, {ensembleSize: 150, observationMatrix: H, observationNoiseVar: [0.36], initialMean: [0, 0], initialStd: [2, 5], seed: 9});
  const enkf = new EnsembleKalmanFilterStation('enkf', filter);
  const sink = new SdeEstimateSinkStation('sink');
  plant.pipe(enkf, SdeChannels.OBSERVATION);
  plant.pipe(sink, SdeChannels.STATE);
  enkf.pipe(sink, SdeChannels.ESTIMATE);
  runIterativeDES([plant, enkf, sink] as DESStation[], {shuffle: false, maxTicks: steps + 5});

  check('4.1 emits one estimate per step', sink.estimates.length === steps, `got ${sink.estimates.length}`);
  const rmse = sink.rmseByDimension();
  const meanI = sink.truth.reduce((a, t) => a + t.state[0], 0) / sink.truth.length;
  const baseI = Math.sqrt(sink.truth.reduce((a, t) => a + (t.state[0] - meanI) ** 2, 0) / sink.truth.length);
  check('4.2 hidden current RMSE < ½ baseline', rmse[0] < 0.5 * baseI, `rmse=${rmse[0].toFixed(3)} base=${baseI.toFixed(3)}`);
  check('4.3 observed-speed RMSE below sensor noise (0.6)', rmse[1] < 0.6, `rmse=${rmse[1].toFixed(3)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[5] MLP — learns a nonlinear function');
// -----------------------------------------------------------------------------
{
  const rng = new Mulberry32(1);
  const net = new Mlp(1, 32, rng);
  for (let it = 0; it < 8000; it++) {
    const x = rng.uniform(2);
    net.trainExample([x], Math.sin(x), 0.02);
  }
  let sse = 0; const m = 200;
  const r2 = new Mulberry32(99);
  for (let i = 0; i < m; i++) { const x = r2.uniform(2); const d = net.predict([x]) - Math.sin(x); sse += d * d; }
  check('5.1 MLP fits sin(x) (MSE < 0.01)', sse / m < 0.01, `mse=${(sse / m).toFixed(5)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[6] ML-3: denoising diffusion learns a unimodal target');
// -----------------------------------------------------------------------------
{
  const rng = new Mulberry32(5);
  const data = Array.from({length: 2000}, () => 3 + rng.normal() * 0.6);
  const model = new DenoisingDiffusionModel({steps: 80, betaMax: 0.2, hidden: 64, seed: 2});
  model.train(data, {iterations: 25000, learningRate: 0.006});
  const s = DenoisingDiffusionModel.summarise(model.sample(2000));
  check('6.1 generated mean ≈ 3', Math.abs(s.mean - 3) < 0.5, `mean=${s.mean.toFixed(3)}`);
  check('6.2 generated std ≈ 0.6', Math.abs(s.std - 0.6) < 0.35, `std=${s.std.toFixed(3)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[7] Deterministic limit — zero-noise motor → analytic steady state');
// -----------------------------------------------------------------------------
{
  // With σ=0 the SDE is the deterministic ODE. Steady state: V=Ri+K_eω, K_t i=Bω.
  const R = 2, L = 0.5, Ke = 0.1, Kt = 0.1, J = 0.02, Bf = 0.002, V = 12;
  const motor = new StochasticDcMotor({resistance: R, inductance: L, backEmfConstant: Ke, torqueConstant: Kt, inertia: J, friction: Bf, voltage: V, currentNoise: 0, speedNoise: 0});
  const {path} = new EulerMaruyamaIntegrator().simulate(motor, [0, 0], 0.001, 20000, new Mulberry32(1));
  const [iEnd, wEnd] = path[path.length - 1];
  const wStar = V / (Ke + (R * Bf) / Kt);
  const iStar = (Bf * wStar) / Kt;
  check('7.1 ω → analytic steady state', rel(wEnd, wStar) < 0.01, `ω=${wEnd.toFixed(3)} target=${wStar.toFixed(3)}`);
  check('7.2 i → analytic steady state', rel(iEnd, iStar) < 0.02, `i=${iEnd.toFixed(4)} target=${iStar.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[8] EnKF — posterior uncertainty shrinks');
// -----------------------------------------------------------------------------
{
  const motor = new StochasticDcMotor({resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002, voltage: 12, currentNoise: 0.4, speedNoise: 0.5});
  const dt = 0.01;
  const H = [[0, 1]];
  const filter = new EnsembleKalmanFilter(motor, dt, {ensembleSize: 200, observationMatrix: H, observationNoiseVar: [0.36], initialMean: [0, 0], initialStd: [3, 6], seed: 1});
  const var0 = filter.variance();
  // Feed consistent speed observations near the steady state for a while.
  for (let k = 0; k < 200; k++) filter.step([80 + (k % 5) * 0.1]);
  const varN = filter.variance();
  check('8.1 speed variance shrinks', varN[1] < var0[1], `${var0[1].toFixed(2)} → ${varN[1].toFixed(3)}`);
  check('8.2 hidden-current variance shrinks', varN[0] < var0[0], `${var0[0].toFixed(2)} → ${varN[0].toFixed(3)}`);
  check('8.3 posterior means finite', Number.isFinite(filter.mean()[0]) && Number.isFinite(filter.mean()[1]));
}

// -----------------------------------------------------------------------------
console.log('\n[9] ML-1 — Ornstein–Uhlenbeck parameter recovery');
// -----------------------------------------------------------------------------
{
  const ou = new OrnsteinUhlenbeck(0.8, 1.5, 0.4);
  const {times, path} = new EulerMaruyamaIntegrator().simulate(ou, [0], 0.01, 8000, new Mulberry32(21));
  const fit = new SdeMaximumLikelihoodEstimator({iterations: 600, learningRate: 0.05}).fit(new OuFamily(), times, path);
  // σ (diffusion) reliably identifiable; θ, μ recover well from a long mean-reverting path.
  check('9.1 recovers σ within 10%', rel(fit.params.sigma, 0.4) < 0.1, `sigma=${fit.params.sigma.toFixed(4)}`);
  check('9.2 recovers θ within 30%', rel(fit.params.theta, 0.8) < 0.3, `theta=${fit.params.theta.toFixed(4)}`);
  check('9.3 recovers μ within 0.3', Math.abs(fit.params.mu - 1.5) < 0.3, `mu=${fit.params.mu.toFixed(4)}`);
}

// -----------------------------------------------------------------------------
console.log('\n[10] Diffusion schedule + Brownian increment statistics');
// -----------------------------------------------------------------------------
{
  const model = new DenoisingDiffusionModel({steps: 100, betaMax: 0.2, seed: 1});
  check('10.1 forward process reaches prior (√ᾱ_T ≈ 0)', model.terminalSignalRetention() < 0.05, `√ᾱ_T=${model.terminalSignalRetention().toFixed(4)}`);
  check('10.2 step count exposed', model.numSteps() === 100);

  // dW ~ N(0, dt·I): sample variance ≈ dt.
  const em = new EulerMaruyamaIntegrator();
  const rng = new Mulberry32(8);
  const dt = 0.05; const N = 20000;
  let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) { const dW = em.brownianIncrement(1, dt, rng)[0]; s += dW; s2 += dW * dW; }
  const mean = s / N, variance = s2 / N - mean * mean;
  check('10.3 E[dW] ≈ 0', Math.abs(mean) < 0.01, `mean=${mean.toFixed(4)}`);
  check('10.4 Var[dW] ≈ dt', rel(variance, dt) < 0.05, `var=${variance.toFixed(4)} dt=${dt}`);
}

// -----------------------------------------------------------------------------
console.log(`\n──────────────────────────────────────────────`);
console.log(`  stochastic-sde: ${pass} passed, ${fail} failed`);
console.log(`──────────────────────────────────────────────`);
if (fail > 0) process.exit(1);
