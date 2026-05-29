'use strict';

// =============================================================================
// main-stochastic-sde.ts — model SDEs and run THREE ML algorithms on them.
//
//   npm run stochastic-sde
//
//   0. Euler–Maruyama vs analytic GBM moments (the random-process engine).
//   1. ML system-id : recover μ, σ of a GBM path by maximum likelihood.
//   2. ML filtering  : Ensemble Kalman Filter estimates the hidden current i of
//                      a stochastic DC motor from noisy SPEED-only measurements
//                      (wired as a DES pipeline: plant → EnKF station → sink).
//   3. ML generative : a denoising-diffusion model learns a bimodal target and
//                      generates samples by integrating the reverse-time SDE.
// =============================================================================

import {runIterativeDES} from './general/des-base/runner';
import {DESStation} from './general/des-base/station';
import {Mulberry32} from './general/control-systems/empirical-control';
import {
  EulerMaruyamaIntegrator,
  GeometricBrownianMotion,
  SdeChannels,
  SdeEstimateSinkStation,
  SdePlantStation,
  StochasticDcMotor,
} from './general/control-systems/stochastic-sde';
import {
  DenoisingDiffusionModel,
  EnsembleKalmanFilter,
  EnsembleKalmanFilterStation,
  GbmFamily,
  SdeMaximumLikelihoodEstimator,
} from './general/control-systems/sde-learning';

class StochasticSdeDemo {
  run(): void {
    this.engineCheck();
    this.mleSystemId();
    this.enkfFiltering();
    this.diffusionGenerative();
  }

  // 0. Engine — empirical moments vs the closed-form GBM solution.
  private engineCheck(): void {
    console.log('================ 0. SDE engine: GBM Euler–Maruyama vs analytic ================');
    const gbm = new GeometricBrownianMotion(0.1, 0.3);
    const em = new EulerMaruyamaIntegrator();
    const x0 = 1, dt = 0.002, steps = 1000, paths = 4000;
    let sum = 0, sumSq = 0;
    for (let p = 0; p < paths; p++) {
      const {path} = em.simulate(gbm, [x0], dt, steps, new Mulberry32(1000 + p));
      const xT = path[path.length - 1][0];
      sum += xT; sumSq += xT * xT;
    }
    const empMean = sum / paths;
    const empVar = sumSq / paths - empMean * empMean;
    const T = dt * steps;
    console.log(`  T=${T.toFixed(2)}  E[X_T]: empirical ${empMean.toFixed(4)} vs analytic ${gbm.meanAt(x0, T).toFixed(4)}`);
    console.log(`           Var[X_T]: empirical ${empVar.toFixed(4)} vs analytic ${gbm.varAt(x0, T).toFixed(4)}`);
  }

  // 1. ML system identification — MLE of GBM drift/diffusion from one path.
  private mleSystemId(): void {
    console.log('\n================ 1. ML system-id: maximum-likelihood SDE fit ================');
    const trueMu = 0.12, trueSigma = 0.3;
    const gbm = new GeometricBrownianMotion(trueMu, trueSigma);
    const {times, path} = new EulerMaruyamaIntegrator().simulate(gbm, [1], 0.004, 6000, new Mulberry32(77));
    const fit = new SdeMaximumLikelihoodEstimator({iterations: 1500, learningRate: 0.05}).fit(new GbmFamily(), times, path);
    console.log(`  true   : mu=${trueMu}, sigma=${trueSigma}`);
    console.log(`  learned: mu=${fit.params.mu.toFixed(4)}, sigma=${fit.params.sigma.toFixed(4)}   (NLL=${fit.finalNegLogLik.toFixed(1)}, ${fit.iterations} Adam steps)`);
  }

  // 2. ML filtering — EnKF recovers hidden current from speed-only measurements.
  private enkfFiltering(): void {
    console.log('\n================ 2. ML filtering: Ensemble Kalman Filter (DES pipeline) ================');
    const motor = new StochasticDcMotor({
      resistance: 2, inductance: 0.5, backEmfConstant: 0.1, torqueConstant: 0.1, inertia: 0.02, friction: 0.002,
      voltage: 12, currentNoise: 0.4, speedNoise: 0.5,
    });
    const dt = 0.01, steps = 500;
    const H = [[0, 1]];                 // observe ω only; current i is hidden
    const plant = new SdePlantStation('motor-plant', {
      system: motor, x0: [0, 0], dt, steps, observationMatrix: H, observationNoiseStd: [0.6], seed: 5,
    });
    const filter = new EnsembleKalmanFilter(motor, dt, {
      ensembleSize: 150, observationMatrix: H, observationNoiseVar: [0.36],
      initialMean: [0, 0], initialStd: [2, 5], seed: 9,
    });
    const enkf = new EnsembleKalmanFilterStation('enkf', filter);
    const sink = new SdeEstimateSinkStation('sink');
    plant.pipe(enkf, SdeChannels.OBSERVATION);
    plant.pipe(sink, SdeChannels.STATE);
    enkf.pipe(sink, SdeChannels.ESTIMATE);
    runIterativeDES([plant, enkf, sink] as DESStation[], {shuffle: false, maxTicks: steps + 5});

    const rmse = sink.rmseByDimension();
    // Baseline current RMSE if you just guessed the mean current (no filter).
    const meanI = sink.truth.reduce((a, t) => a + t.state[0], 0) / sink.truth.length;
    const baseI = Math.sqrt(sink.truth.reduce((a, t) => a + (t.state[0] - meanI) ** 2, 0) / sink.truth.length);
    console.log(`  observed: speed ω (noisy, σ=0.6);  hidden: current i`);
    console.log(`  EnKF RMSE  → current i = ${rmse[0].toFixed(4)},  speed ω = ${rmse[1].toFixed(4)}`);
    console.log(`  baseline   → current i (guess mean) = ${baseI.toFixed(4)}   ⇒ filter recovers the hidden state`);
  }

  // 3. ML generative — denoising diffusion learns a bimodal target.
  private diffusionGenerative(): void {
    console.log('\n================ 3. ML generative: score-based diffusion (reverse SDE) ================');
    const rng = new Mulberry32(2024);
    const data: number[] = [];
    for (let i = 0; i < 3000; i++) {
      const mode = rng.next() < 0.5 ? -2 : 2;
      data.push(mode + rng.normal() * 0.4);
    }
    const model = new DenoisingDiffusionModel({steps: 100, betaMax: 0.2, hidden: 128, seed: 3});
    const loss = model.train(data, {iterations: 60000, learningRate: 0.004});
    const samples = model.sample(3000);
    const dataStats = DenoisingDiffusionModel.summarise(data);
    const genStats = DenoisingDiffusionModel.summarise(samples);
    const nearNeg = samples.filter(s => s < 0).length / samples.length;
    console.log(`  target  : bimodal N(±2, 0.4²)   data mean/std = ${dataStats.mean.toFixed(3)} / ${dataStats.std.toFixed(3)}`);
    console.log(`  learned : sample mean/std = ${genStats.mean.toFixed(3)} / ${genStats.std.toFixed(3)}   (final DSM loss ${loss.toFixed(4)})`);
    console.log(`  modes   : ${(nearNeg * 100).toFixed(0)}% near −2, ${((1 - nearNeg) * 100).toFixed(0)}% near +2  (target ≈ 50/50)`);
  }
}

new StochasticSdeDemo().run();
