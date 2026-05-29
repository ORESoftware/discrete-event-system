'use strict';

// =============================================================================
// main-build-site.ts — regenerate every simulation HTML page into out/ and
// write the curated landing index (out/index.html).
//
//   npm run build-site
//
// The cluster's dd-des-simulator serves out/ at /des/out/, so committing the
// regenerated artifacts is the deploy step (GitOps: commit → host pull → serve).
// =============================================================================

import {execFileSync} from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import {IndexEntry, SimulationIndexPage} from './animation/run-report';

class SimulationSiteBuilder {
  private readonly tsNode = path.join('node_modules', '.bin', 'ts-node');

  private run(script: string, env: Record<string, string> = {}): void {
    process.stderr.write(`  • ${script} ${Object.entries(env).map(([k, v]) => `${k}=${v}`).join(' ')}\n`);
    execFileSync(this.tsNode, [script], {stdio: 'inherit', env: {...process.env, ...env}, maxBuffer: 8 * 1024 * 1024});
  }

  build(): void {
    process.stderr.write('Regenerating animations...\n');
    this.run('src/des/main-wind-mppt-anim.ts');
    this.run('src/des/main-wind-mppt-anim.ts', {CONTROLLER: 'pi'});
    this.run('src/des/main-dc-motor-anim.ts');
    this.run('src/des/main-dc-motor-anim.ts', {MODE: 'open'});
    this.run('src/des/main-observability-controllability-anim.ts');

    process.stderr.write('Generating run reports...\n');
    this.run('src/des/main-empirical-control-report.ts');
    this.run('src/des/main-stochastic-sde-report.ts');

    this.writeIndex();
  }

  private linkIfExists(entry: IndexEntry): IndexEntry | null {
    return fs.existsSync(path.join('out', entry.href)) ? entry : null;
  }

  private writeIndex(): void {
    const animations: IndexEntry[] = [
      {kind: 'animation', title: 'Wind MPPT — optimal torque', href: 'wind-mppt/animation-optimal-torque.html',
        description: 'Variable-speed PMSG turbine tracking optimal tip-speed ratio via T = K_opt·ω².'},
      {kind: 'animation', title: 'Wind MPPT — PI speed loop', href: 'wind-mppt/animation-pi.html',
        description: 'Same turbine driven by a PI controller tracking ω* = λ*·V/R.'},
      {kind: 'animation', title: 'DC motor — closed-loop PI', href: 'dc-motor/animation-closed.html',
        description: 'Back-EMF ODE motor; PI speed control tracking 60→100 rad/s with a load step.'},
      {kind: 'animation', title: 'DC motor — open loop', href: 'dc-motor/animation-open.html',
        description: 'Step-voltage response showing back-EMF rise throttling armature current.'},
      {kind: 'animation', title: 'Controllability & Observability', href: 'obs-ctrl/animation.html',
        description: 'Kalman rank tests, MDP reachability, and POMDP distinguishability storyboard.'},
    ];
    const runs: IndexEntry[] = [
      {kind: 'run report', title: 'Empirical controllability & observability', href: 'empirical-control/report.html',
        description: 'Gramian degree (min/max directions) and Monte-Carlo trial estimates vs analytic Kalman tests.'},
      {kind: 'run report', title: 'Stochastic SDEs + 3 ML algorithms', href: 'stochastic-sde/report.html',
        description: 'Euler–Maruyama engine with MLE system-id, Ensemble Kalman filtering, and a diffusion model.'},
    ];

    const page = new SimulationIndexPage(
      'Discrete-Event-System — Simulations & Runs',
      'Control-system animations and numerical / machine-learning runs, generated from the discrete-event-system submodule.',
    );
    const present = (es: IndexEntry[]) => es.map(e => this.linkIfExists(e)).filter((e): e is IndexEntry => e !== null);
    page.addGroup({
      heading: 'Control-system animations',
      blurb: 'Interactive HTML players (play / pause / scrub / speed) built on the DES animation engine.',
      entries: present(animations),
    });
    page.addGroup({
      heading: 'Numerical & machine-learning runs',
      blurb: 'Reproducible run reports with the full console output of each simulation.',
      entries: present(runs),
    });

    const out = path.join('out', 'index.html');
    fs.mkdirSync(path.dirname(out), {recursive: true});
    fs.writeFileSync(out, page.toHtml(new Date().toISOString()));
    process.stderr.write(`\nLanding page: ${path.resolve(out)}\n`);
    process.stderr.write('Served on cluster at /des/out/ (and /des/out/index.html).\n');
  }
}

new SimulationSiteBuilder().build();
