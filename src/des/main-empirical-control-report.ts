// RUST MIGRATION: target src/bin/main_empirical_control_report.rs.
// RUST MIGRATION: Keep this binary thin: parse CLI/env/path inputs with clap/std::env/PathBuf, then call library orchestration.
// RUST MIGRATION: Port the runnable body as fn main() -> Result<()> and move reusable DES setup into src/des modules/traits.
// RUST MIGRATION: Keep JSON examples/config as serde-deserialized structs instead of ad-hoc JS objects.
'use strict';

// =============================================================================
// main-empirical-control-report.ts — run the empirical controllability /
// observability demo and write a styled HTML report into out/.
//
//   npm run empirical-control-report
// =============================================================================

import {execFileSync} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import {RunReportPage} from './animation/run-report';

class EmpiricalControlReport {
  run(): void {
    const tsNode = path.join('node_modules', '.bin', 'ts-node');
    let log: string;
    try {
      log = execFileSync(tsNode, ['src/des/main-empirical-control.ts'], {encoding: 'utf8', maxBuffer: 8 * 1024 * 1024});
    } catch (e) {
      log = `run failed:\n${(e as Error).message}`;
    }
    const page = new RunReportPage(
      'Empirical Controllability & Observability',
      'Quantitative degree (Gramian eigenvalues) and trial-based estimates vs the analytic Kalman tests.',
    );
    page.addSection({
      heading: 'What this run measures',
      description: 'Instead of the binary Kalman rank verdict, this computes how controllable/observable each direction is: ' +
        'controllability/observability Gramian eigenvalues (min = weakest direction, max = strongest), the empirical reached-state ' +
        'covariance from thousands of random control rollouts (∝ W_c), least-squares target hit rate, noisy state-reconstruction error, ' +
        'MDP random-policy reach degree, and POMDP belief-tracking hit-probability / residual entropy.',
    });
    page.addSection({heading: 'Run output', log});
    const out = path.join('out', 'empirical-control', 'report.html');
    fs.mkdirSync(path.dirname(out), {recursive: true});
    fs.writeFileSync(out, page.toHtml());
    console.log(`Empirical-control report: ${path.resolve(out)}`);
  }
}

new EmpiricalControlReport().run();
