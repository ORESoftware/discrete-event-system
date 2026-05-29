'use strict';

// =============================================================================
// general/des-base/runner.ts — the iterative DES runner.
//
// Calls every run-loop participant's runTimeStep() in shuffled order each tick
// until no participant has work (or a stop predicate fires, or maxTicks hit).
// =============================================================================

import {DESRunLoopEntity, DESStation} from './station';
import {ValidationCheck} from './validation';

export type IterativeDESParticipant = DESRunLoopEntity;

export interface IterativeRunOptions {
  /** Maximum simulation ticks. Default Infinity (run until quiescent). */
  maxTicks?: number;
  /** Stop predicate run BEFORE each tick. Returns true -> terminate. */
  stopWhen?: (tick: number, stations: readonly IterativeDESParticipant[]) => boolean;
  /** Optional dynamic roster. Use when smart movables enter/exit while
   *  the simulation is running and should participate directly in ticks. */
  getRunLoopEntities?: () => IterativeDESParticipant[];
  /** RNG for tick-order shuffling. Defaults to Math.random. Pass a
   *  seeded mulberry32 for reproducible runs. */
  rng?: () => number;
  /** Whether to randomise station-execution order each tick. Default true. */
  shuffle?: boolean;
  /** Optional per-tick callback (instrumentation, animation, etc.). */
  onTick?: (tick: number, stations: readonly IterativeDESParticipant[]) => void;
  /** Run validators registered on every station after the loop terminates.
   *  Default true. Each station's `onFinalize()` runs first, then
   *  `runValidation()` aggregates all checks into the summary. */
  runValidators?: boolean;
}

export interface IterativeRunSummary {
  ticks: number;
  reason: 'done' | 'maxticks' | 'stop-when';
  /** Aggregated validator output, present iff `runValidators !== false` AND
   *  at least one station had validators registered. Stations are visited
   *  in first-seen run-loop order. */
  validation?: ValidationCheck[];
  /** True iff all entries in `validation` had `passed: true` (or there
   *  were no validators). */
  validationOk?: boolean;
}

export type DESResultStation<R> = DESStation & {
  result(validation?: ValidationCheck[]): R;
};

/** Mutate `arr` in place via Fisher-Yates with the given RNG. */
function shuffleInPlace<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/** Run the simulation until a terminal condition fires.
 *
 *  Termination order:
 *    1. stopWhen(tick) returns true  -> 'stop-when'
 *    2. tick >= maxTicks             -> 'maxticks'
 *    3. no station has work          -> 'done'
 *
 *  Each tick:
 *    1. compute "any participant has work?" via participant.hasWork()
 *    2. shuffle(participants) (if shuffle is true)
 *    3. for each participant: participant.runTimeStep()
 *    4. onTick(tick, participants)
 *    5. tick++
 */
export function runIterativeDES(
  stations: IterativeDESParticipant[],
  opts: IterativeRunOptions = {},
): IterativeRunSummary {
  const maxTicks = opts.maxTicks ?? Infinity;
  const rng = opts.rng ?? Math.random;
  const shuffle = opts.shuffle ?? true;
  const wantValidate = opts.runValidators ?? true;

  const seen = new Set<IterativeDESParticipant>();
  const currentEntities = (): IterativeDESParticipant[] => {
    const entities = opts.getRunLoopEntities?.() ?? stations;
    for (const s of entities) seen.add(s);
    return entities;
  };

  for (const s of currentEntities()) s.assertPreconditions?.();

  let tick = 0;
  let reason: IterativeRunSummary['reason'];
  while (true) {
    const entities = currentEntities();
    if (opts.stopWhen?.(tick, entities)) { reason = 'stop-when'; break; }
    if (tick >= maxTicks) {
      console.warn(`[runIterativeDES] hit maxTicks=${maxTicks} before the system went quiescent (${entities.length} participants still active) — increase maxTicks or check for a non-terminating model.`);
      reason = 'maxticks';  break;
    }
    let anyWork = false;
    for (const s of entities) {
      const hasWork = s.hasWork ? s.hasWork() : true;
      if (hasWork) { anyWork = true; break; }
    }
    if (!anyWork) { reason = 'done'; break; }
    const order = entities.slice();
    if (shuffle) shuffleInPlace(order, rng);
    for (const s of order) s.runTimeStep();
    opts.onTick?.(tick, entities);
    tick++;
  }

  for (const s of seen) s.onFinalize?.();
  const summary: IterativeRunSummary = {ticks: tick, reason};
  if (wantValidate) {
    const allChecks: ValidationCheck[] = [];
    for (const s of seen) {
      if (!s.numValidators || !s.runValidation || s.numValidators() === 0) continue;
      allChecks.push(...s.runValidation());
    }
    if (allChecks.length > 0) {
      summary.validation = allChecks;
      summary.validationOk = allChecks.every(c => c.passed);
      if (!summary.validationOk) {
        const failed = allChecks.filter(c => !c.passed).map(c => c.name);
        console.warn(`[runIterativeDES] ${failed.length}/${allChecks.length} validators FAILED after ${tick} ticks: ${failed.join(', ')}`);
      }
    }
  }
  return summary;
}

export function runResultStation<R>(
  station: DESResultStation<R>,
  opts: IterativeRunOptions = {},
): R {
  const summary = runIterativeDES([station], opts);
  return station.result(summary.validation ?? []);
}

export function failedValidationChecks(summary: Pick<IterativeRunSummary, 'validation'>): ValidationCheck[] {
  return summary.validation?.filter(c => !c.passed) ?? [];
}

export function validationFailureNames(summary: Pick<IterativeRunSummary, 'validation'>): string {
  return failedValidationChecks(summary).map(c => c.name).join(', ');
}

export function assertNoValidationFailures(
  summary: Pick<IterativeRunSummary, 'validation'>,
  modelName: string,
): void {
  const names = validationFailureNames(summary);
  if (names.length > 0) {
    console.warn(`[${modelName}] post-run validation failed (${failedValidationChecks(summary).length} checks): ${names}`);
    throw new Error(`${modelName} validation failed: ${names}`);
  }
}
