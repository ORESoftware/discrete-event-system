// RUST MIGRATION: Port file-for-file to `tests/collaborative_inference_test.rs` for sparse preference inference and registry-driven DES execution.
// Test-port notes: convert scenarios into `#[test]` functions returning `Result<()>`; replace ad hoc check helpers with `assert!`, `assert_eq!`, and approximate-float helpers; preserve deterministic fixtures.

'use strict';

// =============================================================================
// Tests for collaborative sparse preference inference.
// =============================================================================

import {getModel, runFromSpec} from '../general/des-registry';
import {runCollaborativeInference} from '../general/collaborative-inference';

let pass = 0;
let fail = 0;

function check(label: string, ok: boolean, detail = ''): void {
  if (ok) {
    pass += 1;
    console.log(`  PASS    ${label}${detail ? '  ' + detail : ''}`);
  } else {
    fail += 1;
    console.log(`  FAIL    ${label}${detail ? '  ' + detail : ''}`);
  }
}

async function main(): Promise<void> {
  console.log('\n[1] Explicit sparse responses');
  {
    const r = runCollaborativeInference({
      scenario: 'custom',
      items: [
        {id: 'a', label: 'A'},
        {id: 'b', label: 'B'},
        {id: 'c', label: 'C'},
      ],
      responses: [
        {id: 'r1', age: 20, experienceYears: {a: 4, b: 12}, ratings: {a: 5, b: 3}, ranking: ['a', 'b']},
        {id: 'r2', age: 35, experienceYears: {a: 15, c: 2}, ratings: {a: 4, c: 2}, ranking: ['a', 'c']},
        {id: 'r3', age: 30, experienceYears: {b: 10, c: 5}, ratings: {b: 4, c: 3}, ranking: ['b', 'c']},
      ],
      ratingMin: 1,
      ratingMax: 5,
      shrinkage: 1,
      minCredibleAge: 15,
      topK: 3,
    });
    check('processes every explicit respondent', r.respondentsProcessed === 3, `processed=${r.respondentsProcessed}`);
    check('creates rating evidence', r.ratingEvidenceCount === 6, `ratings=${r.ratingEvidenceCount}`);
    check('creates pairwise evidence', r.pairwiseEvidenceCount === 3, `comparisons=${r.pairwiseEvidenceCount}`);
    check('ranks the clearly preferred item first', r.rankings[0].itemId === 'a', `top=${r.rankings[0].itemId}`);
    check('validates conservation and finite scores', r.validation.every(c => c.passed), r.validation.filter(c => !c.passed).map(c => c.name).join(', '));
    check('caps impossible experience claims', r.credibility.cappedExperienceClaims >= 1, `capped=${r.credibility.cappedExperienceClaims}`);
    check('credibility weights increase experienced respondents', r.credibility.maxRespondentWeight > 1, `max=${r.credibility.maxRespondentWeight}`);
    check('exposes source/station/sink roles',
      r.stationRoles.sources.includes('respondent-source')
      && r.stationRoles.stations.includes('evidence-aggregator')
      && r.stationRoles.sinks.includes('inference-result-sink'));
  }

  console.log('\n[2] Programming-language scenario');
  {
    const r = runCollaborativeInference({
      scenario: 'programming-languages',
      respondentCount: 1000,
      minItemsPerRespondent: 4,
      maxItemsPerRespondent: 5,
      respondentsPerTick: 200,
      seed: 7,
      topK: 10,
    });
    check('uses 50 language alternatives', r.rankings.length === 50, `items=${r.rankings.length}`);
    check('processes synthetic developer panel', r.respondentsProcessed === 1000, `processed=${r.respondentsProcessed}`);
    check('ratings roughly match 4-5 per respondent', r.ratingEvidenceCount >= 4000 && r.ratingEvidenceCount <= 5000, `ratings=${r.ratingEvidenceCount}`);
    check('coverage reaches every language', r.coverage.itemsWithRatings === 50 && r.coverage.itemsWithComparisons === 50);
    check('top scores are sorted', r.rankings.every((row, i, arr) => i === 0 || arr[i - 1].score >= row.score));
    check('iterative breadth weighting ran', r.credibility.passes === 2 && r.credibility.highRatedBonusRespondents > 0,
      `passes=${r.credibility.passes}, bonuses=${r.credibility.highRatedBonusRespondents}`);
  }

  console.log('\n[3] Other built-in scenarios');
  for (const scenario of ['model-validation', 'learning-resources'] as const) {
    const r = runCollaborativeInference({scenario, respondentCount: 300, seed: 3, topK: 5});
    check(`${scenario} produces a ranking`, r.rankings.length > 0);
    check(`${scenario} validation passes`, r.validation.every(c => c.passed), r.validation.filter(c => !c.passed).map(c => c.name).join(', '));
  }

  console.log('\n[4] Registry smoke');
  {
    const reg = getModel('collaborative-inference');
    check('registry has collaborative-inference', reg.id === 'collaborative-inference');
    const summary = await runFromSpec({
      $schema: 'des/model-spec/v1',
      model: 'collaborative-inference',
      parameters: {
        scenario: 'learning-resources',
        respondentCount: 120,
        seed: 9,
        topK: 4,
      },
      runtime: {animate: false, verbose: false},
    }, {verbose: false});
    check('runFromSpec executes collaborative-inference', summary.modelId === 'collaborative-inference');
  }

  console.log('\n========================================');
  console.log(`collaborative-inference-test: ${pass}/${pass + fail} checks passed.`);
  if (fail > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
