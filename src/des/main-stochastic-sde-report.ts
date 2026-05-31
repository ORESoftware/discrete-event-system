// RUST MIGRATION: target src/bin/main_stochastic_sde_report.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// RUST MIGRATION  —  target: src/bin/main-stochastic-sde-report.rs   (fn main)
// 1:1 file move. Report tool: runs main-stochastic-sde and writes a styled
// HTML report into out/.
//
// Conversion notes (file-specific):
//   - execFileSync(ts-node, [main-stochastic-sde.ts]) -> std::process::Command
//     invoking that sibling binary.
//   - class StochasticSdeReport -> struct + impl; fs write -> std::fs.
//   - RunReportPage -> use crate::des::animation::run_report.
// =============================================================================

// =============================================================================
// main-stochastic-sde-report.ts — run the stochastic-SDE + 3-ML-algorithm demo
// and write a styled HTML report into out/.
//
//   npm run stochastic-sde-report
// =============================================================================

import {execFileSync} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import {RunReportPage} from './animation/run-report';

class StochasticSdeReport {
  run(): void {
    const tsNode = path.join('node_modules', '.bin', 'ts-node');
    let log: string;
    try {
      log = execFileSync(tsNode, ['src/des/main-stochastic-sde.ts'], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
    } catch (e) {
      log = `run failed:\n${(e as Error).message}`;
    }
    const page = new RunReportPage(
      'Stochastic Differential Equations + 3 ML algorithms',
      'Euler–Maruyama SDE engine with system identification, ensemble filtering, and score-based diffusion.',
    );
    page.addSection({
      heading: 'What this run covers',
      description: 'Models dX = f(X,t)dt + g(X,t)dW where the solution is a random process. Three machine-learning paradigms run on it: ' +
        '(1) maximum-likelihood SDE parameter recovery (system identification); (2) an Ensemble Kalman Filter that estimates a hidden ' +
        'motor current from noisy speed-only measurements (filtering/inference); (3) a denoising-diffusion generative model that learns a ' +
        'bimodal target and samples it by integrating the reverse-time SDE.',
    });
    page.addSection({heading: 'Run output', log});
    const out = path.join('out', 'stochastic-sde', 'report.html');
    fs.mkdirSync(path.dirname(out), {recursive: true});
    fs.writeFileSync(out, page.toHtml());
    console.log(`Stochastic-SDE report: ${path.resolve(out)}`);
  }
}

new StochasticSdeReport().run();
