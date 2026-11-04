# mod-109-minimal-eval-and-regression-checks — Solutions

Reference solutions for the **Minimal Eval and Regression Checks** module. The four
exercises build a small, boring, automation-friendly evaluation harness on top of
the `support-agent` app from mod-105 / mod-106. The arc is: write a golden set,
run it under pytest in CI, pick the right scoring method per field, then emit a
regression report when the model changes.

The scenario — a customer-support agent that answers questions about returns,
shipping, and account access — is reused across all four exercises so the golden
set and scoring code grow together.

## Index

| Exercise | Solution | Focus |
|----------|----------|-------|
| exercise-01 | [exercise-01-author-a-golden-set](exercise-01-author-a-golden-set/SOLUTION.md) | Design a ~30-example golden set with intent, difficulty, risk-tier, and expected-output fields |
| exercise-02 | [exercise-02-pytest-eval-in-ci](exercise-02-pytest-eval-in-ci/SOLUTION.md) | Drive the golden set from `pytest`, parametrize per-case, publish a GitHub Actions job that gates PRs |
| exercise-03 | [exercise-03-rule-vs-similarity-vs-judge-scoring](exercise-03-rule-vs-similarity-vs-judge-scoring/SOLUTION.md) | Pick the right scorer per field: exact/regex, embedding similarity, LLM-as-judge with a calibrated rubric |
| exercise-04 | [exercise-04-regression-report-for-a-model-swap](exercise-04-regression-report-for-a-model-swap/SOLUTION.md) | Produce a side-by-side baseline-vs-candidate regression report with pass-rate deltas, per-tag breakdowns, and a merge recommendation |

## Conventions

- All code blocks are runnable. The `support-agent` uses the `AGENT_FAKE=1` stub
  from mod-106 so the golden set can be exercised without a provider API key.
- The golden set is a single JSONL file (`evals/golden.jsonl`) that is the input
  to every exercise. New examples are appended, never silently rewritten, so a
  regression report can be reproduced from git history.
- Scoring code lives in `evals/scoring.py`. Each scorer is a pure function
  `(expected, actual) -> Score` so pytest can call them without a fixture.
- CI results are stored as JSON (`evals/results/<git-sha>.json`) so exercise-04's
  regression report is a diff between two of those files, not a re-run.

## Sources used across the module

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (AI RMF 1.0) — <https://www.nist.gov/itl/ai-risk-management-framework>
