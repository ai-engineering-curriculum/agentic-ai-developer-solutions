# Exercise 02 — Pytest eval in CI

## 1. Solution overview

The eval harness is `pytest`. Not a new runner, not a bespoke framework — a
single `tests/eval/test_golden.py` that reads the golden set from exercise-01,
parametrizes one test case per row, calls the agent, and asserts via a scorer.
The output is a standard pytest run, which means every tool the team already
uses — the terminal reporter, JUnit XML, GitHub Actions summary, `--last-failed`
— works unmodified.

The design points that matter:

- **One test per row, not one test for the whole set.** `@pytest.mark.parametrize`
  with `ids=` named from `row["id"]` means a failing row shows up as
  `test_golden_case[refund-003]` in the output. That is what makes the signal
  usable — "something broke" is noise, "`refund-003` broke" is a bug report.
- **Scorer selection lives next to the harness, not in the test.** `scoring.py`
  is a thin module with one function per scorer and a `score_row(row, actual)`
  dispatcher. The test is a three-line call site. Exercise-03 extends `scoring.py`
  without touching the test.
- **Results are serialized every run.** The harness appends a JSON record to
  `evals/results/<run-id>.json` with the row id, score, scorer name, model, and
  the actual response. Exercise-04 reads two of these files; this exercise does
  not render a report.
- **Deterministic by default.** The provider is called with `temperature=0`,
  `AGENT_FAKE=1` is the default in CI's unit job, and the live-model job is a
  separate workflow that only runs on label or on `main`. This keeps the normal
  PR cycle free (the fake path) while the real spend happens on demand.
- **Gate on pass rate, not individual rows.** The pytest run itself gates on
  "every row passes its scorer." Flaky `judge` rows get re-run up to N times via
  `pytest-rerunfailures`; a row that still fails is a real failure.

## 2. Worked answer and implementation — the pytest harness and the CI job

### 2.1 Repo layout

```text
support-agent/
├── pyproject.toml
├── src/
│   └── (mod-106 agent code)
├── evals/
│   ├── golden.jsonl            # from exercise-01
│   ├── scoring.py              # scorers (filled out in exercise-03)
│   ├── runner.py               # thin wrapper around run_agent + result writer
│   └── results/                # one JSON per CI run
└── tests/
    └── eval/
        ├── __init__.py
        ├── conftest.py
        └── test_golden.py
```

### 2.2 `evals/runner.py`

```python
# evals/runner.py
"""Thin wrapper around the agent for eval use.

The eval test calls ``run_case(row)``; this module hides the differences between
the ``AGENT_FAKE=1`` offline stub (default in CI) and a real provider call. It
also writes one JSON record per row to ``evals/results/<run_id>.json`` so
exercise-04 can diff two runs.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import uuid

from src.agent import run_agent  # from mod-106

RESULTS_DIR = pathlib.Path("evals/results")


def _run_id() -> str:
    # Prefer the CI-provided sha when present so two runs on the same commit
    # produce the same file name. Otherwise fall back to a short random id.
    sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
    if sha:
        return sha[:12]
    return uuid.uuid4().hex[:12]


def result_path() -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{_run_id()}.json"


def run_case(row: dict) -> dict:
    """Run one golden-set row and return a result dict.

    ``row`` must have at least ``id`` and ``input``. Any other row fields are
    carried through into the result so the scorer and the regression report can
    slice on them.
    """
    t0 = time.perf_counter()
    agent_out = run_agent(row["input"], max_steps=row.get("max_steps", 6))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "id": row["id"],
        "intent": row.get("intent"),
        "difficulty": row.get("difficulty"),
        "risk_tier": row.get("risk_tier"),
        "scorer": row["scorer"],
        "expected": row["expected"],
        "actual": agent_out["final_text"],
        "steps": agent_out.get("steps"),
        "latency_ms": latency_ms,
        "model": os.environ.get("MODEL", "fake" if os.environ.get("AGENT_FAKE") == "1" else "unknown"),
    }


def append_result(record: dict) -> None:
    """Append one scored record to the per-run JSON file."""
    path = result_path()
    data = []
    if path.exists():
        data = json.loads(path.read_text())
    data.append(record)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
```

### 2.3 `evals/scoring.py` (minimal; exercise-03 extends it)

```python
# evals/scoring.py
"""One function per scorer. All return a Score named-tuple with a boolean
``passed`` and a 0.0–1.0 ``value`` so exercise-04 can rank partial credit.

This file is intentionally small here; exercise-03 adds similarity + judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Score:
    passed: bool
    value: float      # 0.0 .. 1.0
    reason: str = ""


def _exact(expected: str, actual: str) -> Score:
    ok = expected.strip().lower() == actual.strip().lower()
    return Score(ok, 1.0 if ok else 0.0, "exact match" if ok else "mismatch")


def _regex(expected: str, actual: str) -> Score:
    ok = re.search(expected, actual, flags=re.IGNORECASE | re.DOTALL) is not None
    return Score(ok, 1.0 if ok else 0.0, f"regex {'matched' if ok else 'did not match'}")


def _contains_all(expected: list[str], actual: str) -> Score:
    low = actual.lower()
    missing = [s for s in expected if s.lower() not in low]
    passed = not missing
    value = (len(expected) - len(missing)) / max(1, len(expected))
    reason = "all substrings present" if passed else f"missing: {missing}"
    return Score(passed, value, reason)


# Stubs filled in by exercise-03.
def _similarity(expected: str, actual: str) -> Score:  # pragma: no cover
    raise NotImplementedError("filled in by exercise-03")


def _judge(expected: dict, actual: str) -> Score:  # pragma: no cover
    raise NotImplementedError("filled in by exercise-03")


SCORERS = {
    "exact": _exact,
    "regex": _regex,
    "contains_all": _contains_all,
    "similarity": _similarity,
    "judge": _judge,
}


def score_row(row: dict, actual: str) -> Score:
    fn = SCORERS[row["scorer"]]
    return fn(row["expected"], actual)
```

### 2.4 `tests/eval/conftest.py`

```python
# tests/eval/conftest.py
"""Fixtures and parametrize data for the eval suite."""
from __future__ import annotations

import json
import os
import pathlib

import pytest

GOLDEN = pathlib.Path("evals/golden.jsonl")


def _load_golden() -> list[dict]:
    rows = []
    for line in GOLDEN.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pytest_configure(config):
    # Keep the harness honest: never silently run against the real provider
    # unless RUN_LIVE=1 is explicitly set. The default in CI is the stub.
    if os.environ.get("RUN_LIVE") != "1":
        os.environ.setdefault("AGENT_FAKE", "1")


@pytest.fixture(scope="session")
def golden_rows() -> list[dict]:
    return _load_golden()


def pytest_generate_tests(metafunc):
    """Parametrize any test asking for the ``row`` fixture with every golden
    row, using ``row['id']`` as the test id so failures are self-describing.
    """
    if "row" in metafunc.fixturenames:
        rows = _load_golden()
        metafunc.parametrize("row", rows, ids=[r["id"] for r in rows])
```

### 2.5 `tests/eval/test_golden.py`

```python
# tests/eval/test_golden.py
"""The eval harness, in one small test.

Every golden row is one parametrized call: run the agent, score the output,
assert and record. The test body is deliberately tiny so the whole surface is
the data (``golden.jsonl``) and the scorers (``scoring.py``), not the runner.
"""
from __future__ import annotations

import pytest

from evals.runner import append_result, run_case
from evals.scoring import score_row


# Judge-scored rows invoke a second LLM call and can be flaky under load.
# Everything else is deterministic (fake path) or should be deterministic
# (temperature=0 on the real path). We retry judge rows only.
@pytest.mark.flaky(reruns=0)  # overridden below via marker
def test_golden_case(row):
    result = run_case(row)
    score = score_row(row, result["actual"])

    # One more record in the per-run JSON — scored, so exercise-04 does not
    # need to re-score the whole file.
    append_result({**result, "passed": score.passed, "value": score.value, "reason": score.reason})

    # Rich failure message — the first thing a reviewer reads in CI.
    assert score.passed, (
        f"[{row['id']}] scorer={row['scorer']} reason={score.reason}\n"
        f"  input:    {row['input']}\n"
        f"  expected: {row['expected']}\n"
        f"  actual:   {result['actual']}"
    )
```

### 2.6 `pyproject.toml` — pytest config and addopts

Add (or append to) the `[tool.pytest.ini_options]` section:

```toml
[tool.pytest.ini_options]
# Keep eval runs out of the default ``pytest`` invocation; they get their own
# target so a dev running unit tests does not accidentally burn a provider call.
testpaths = ["tests"]
markers = [
    "eval: golden-set regression check (opt-in via -m eval or make eval)",
]
addopts = "-m 'not eval'"

# Judge-scored rows retry up to twice with 1-second backoff. Rule- and
# similarity-scored rows never retry, because retrying a deterministic failure
# just wastes time.
reruns = 0
```

And add the marker to the eval test:

```python
# At the top of tests/eval/test_golden.py
import pytest
pytestmark = pytest.mark.eval
```

Developers run `pytest` and get unit tests. CI runs `pytest -m eval` and gets
the golden set.

### 2.7 `Makefile` targets

```makefile
.PHONY: eval eval-live

# Offline eval (stub agent). Fast, free, deterministic. Runs on every PR.
eval:
	pytest -m eval --tb=short

# Live eval against the configured provider. Opt-in per run.
eval-live:
	RUN_LIVE=1 MODEL=$${MODEL:-claude-sonnet-4-6} pytest -m eval -p no:cacheprovider --tb=short
```

### 2.8 `.github/workflows/eval.yml`

```yaml
name: eval

on:
  pull_request:
    paths:
      - "src/**"
      - "evals/**"
      - "tests/eval/**"
      - "pyproject.toml"
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      live:
        description: "Also run the live-model eval (spends money)."
        type: boolean
        default: false

concurrency:
  group: eval-${{ github.ref }}
  cancel-in-progress: true

jobs:
  offline:
    name: eval (offline stub)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"
      - run: make eval
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-offline-${{ github.sha }}
          path: evals/results/*.json
          if-no-files-found: error

  live:
    name: eval (live)
    if: github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.live)
    runs-on: ubuntu-latest
    needs: offline
    environment: eval-live        # requires a reviewer to approve on PRs
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"
      - run: make eval-live
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results-live-${{ github.sha }}
          path: evals/results/*.json
          if-no-files-found: error
```

Points worth calling out:

- The `live` job depends on `offline`, so a PR that breaks the schema or the
  scorers never spends real money.
- `concurrency.cancel-in-progress: true` kills superseded runs — a developer
  pushing five times in a row only pays for the latest.
- `environment: eval-live` forces a manual reviewer click for anything that
  burns provider budget.
- Results are uploaded as an artifact regardless of pass/fail (`if: always()`)
  so exercise-04 can download two SHAs and diff them.

### 2.9 Sample offline run

```text
$ make eval
pytest -m eval --tb=short
=================== test session starts ===================
collected 30 items

tests/eval/test_golden.py::test_golden_case[refund-001] PASSED
tests/eval/test_golden.py::test_golden_case[refund-002] PASSED
tests/eval/test_golden.py::test_golden_case[refund-003] PASSED
...
tests/eval/test_golden.py::test_golden_case[inj-004]    PASSED

30 passed in 2.11s
```

A failure looks like:

```text
tests/eval/test_golden.py::test_golden_case[refund-003] FAILED

FAILED tests/eval/test_golden.py::test_golden_case[refund-003]
  [refund-003] scorer=judge reason=rubric: promised a refund outside policy
    input:    My laptop stopped working after 45 days. Do I get my money back?
    expected: {'rubric': ..., 'pass_criteria': 'All three must be true.'}
    actual:   Yes, we can refund you — please share your order number...
```

That single-line diagnostic is why the test is parametrized on `row["id"]`.

## 3. Validation steps

1. **Schema still loads.** `python -c "import json, pathlib;
   [json.loads(l) for l in pathlib.Path('evals/golden.jsonl').read_text().splitlines() if l.strip()]"`
2. **Collect without running.** `pytest -m eval --collect-only -q`
   should list exactly `N` cases where `N = wc -l < evals/golden.jsonl`.
   Any mismatch means the generator skipped a row.
3. **Offline run.** `AGENT_FAKE=1 pytest -m eval` must pass in under 5 seconds.
   Any failure here is a bug in the stub or the scorer, not the model.
4. **Single-case run** (debugging a failure):
   `pytest -m eval -k refund-003 -s --tb=long`.
5. **Artifact exists.** After the run, `evals/results/*.json` should contain
   exactly one file with 30 records, each with a boolean `passed`.
6. **CI gate.** Flip the expected of one row to a wrong value, push, and verify
   the `eval` check on the PR fails and blocks merge.

## 4. Rubric / review checklist

| Criterion | How to check | Points |
|-----------|--------------|--------|
| One pytest case per golden row, named by `row["id"]` | `pytest --collect-only -q -m eval` | 2 |
| Scorer selection is data-driven (`row["scorer"]`) | read `test_golden.py` | 2 |
| `scoring.py` is one module, one function per scorer, with a dispatcher | read `scoring.py` | 2 |
| Results are serialized to `evals/results/<id>.json` with enough slice fields for exercise-04 | run offline, inspect file | 2 |
| Default CI path is free (no provider calls) | read `conftest.py` + workflow | 2 |
| Live path is opt-in and protected (`environment:` or label gate) | read workflow | 1 |
| Failing test reports row id, scorer, expected, actual | force a fail, read output | 2 |
| Concurrency is capped so a push flood does not burn budget | read workflow | 1 |
| Judge rows can retry; rule rows never do | read `pytest.ini_options` + marker usage | 1 |
| Eval is opt-in locally (`pytest` without `-m eval` skips it) | run `pytest` | 1 |
| **Total** | | **16** |

Pass threshold: 13/16.

## 5. Common mistakes

1. **One monolithic test asserting the whole set.** Looks fine until something
   breaks and the output is "`test_golden FAILED`" with no row id. Parametrize.
2. **Running real-model evals on every PR.** A 30-row set at $0.02/row is
   $0.60/PR; thirty PRs a week is $75/month on a *tiny* set, and that is before
   anyone hits retry. Gate the live job behind a label or a manual environment.
3. **Letting `pytest` run the eval by default.** A developer running `pytest`
   for a unit-test loop should not pay for or wait on the eval suite. The
   `-m 'not eval'` default in `addopts` keeps those worlds separate.
4. **No serialized results.** Without `evals/results/<sha>.json` written on
   every run, exercise-04 has no "before"/"after" artefact. Pytest's own XML
   does not contain the agent's actual response, so a dedicated JSON is needed.
5. **Printing the system prompt or API key in a failure message.** The failure
   message gets uploaded as a CI artifact; keep it to row metadata plus the
   model response. Do not log environment vars.
6. **Retrying rule-based failures.** `contains_all` is deterministic; retrying
   it three times before failing just wastes CI time. Retry *only* the judge
   rows.
7. **Flaky unmarked tests.** If a non-judge row needs retries to pass, the
   model is non-deterministic in a way that will burn you later. Fix
   `temperature`/`top_p` first, don't paper over it with reruns.
8. **No concurrency control.** Every force-push kicks off a new eval; without
   `cancel-in-progress` you pay for every one.

## 6. References

- pytest parametrize — <https://docs.pytest.org/en/stable/how-to/parametrize.html>
- pytest-rerunfailures — <https://github.com/pytest-dev/pytest-rerunfailures>
- GitHub Actions `concurrency` — <https://docs.github.com/en/actions/using-jobs/using-concurrency>
- GitHub Actions deployment environments — <https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment>
- NIST AI RMF (AI RMF 1.0) — <https://www.nist.gov/itl/ai-risk-management-framework> — the MANAGE and MEASURE functions motivate the "record every run, gate on measurable metrics" pattern here.
- MITRE ATLAS — <https://atlas.mitre.org/> — AML.T0051 is why the golden set includes dedicated prompt-injection rows that this harness runs on every PR.

Local exercise context: this harness consumes
[exercise-01](../exercise-01-author-a-golden-set/SOLUTION.md)'s `golden.jsonl`;
[exercise-03](../exercise-03-rule-vs-similarity-vs-judge-scoring/SOLUTION.md)
fills in the `_similarity` and `_judge` scorers; and
[exercise-04](../exercise-04-regression-report-for-a-model-swap/SOLUTION.md)
reads the per-run JSON that `runner.py` writes.
