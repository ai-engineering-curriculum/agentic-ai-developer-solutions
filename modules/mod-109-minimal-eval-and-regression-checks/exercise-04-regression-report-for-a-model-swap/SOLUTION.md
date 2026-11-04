# Exercise 04 — Regression report for a model swap

## 1. Solution overview

A model swap is the single highest-risk change to an LLM application: the
system prompt is the same, the tools are the same, the code is the same, and
yet every output can shift. The job of the regression report is to turn "did
anything break?" into a defensible, sliced, line-itemed yes/no — a document
reviewable in five minutes that a reviewer can sign off on.

The report has one input: **two result files** produced by exercise-02's
harness, one per model. It has four outputs:

1. **Headline deltas** — overall pass rate, mean score, p95 latency.
2. **Per-tag breakdown** — pass rate delta sliced by `intent`,
   `difficulty`, and `risk_tier`. This is where the signal actually lives.
3. **Row-level changes** — the list of rows that flipped pass↔fail, and the
   list of rows whose `value` moved by more than a threshold on the same pass
   verdict (silent drift).
4. **Merge recommendation** — a boolean (`APPROVE` / `BLOCK` /
   `NEEDS_HUMAN_REVIEW`) derived from an explicit policy, not a vibe.

The design points that matter:

- **The report is the diff of two artifacts, not a re-run.** Re-running each
  model on every report request doubles the spend and makes two reports for the
  same pair of commits disagree because of judge variance. The baseline is
  frozen the moment it is produced.
- **The merge policy is a configuration, not a model's judgment.** The policy
  has four knobs (regressed_rows, risk_tier_regressions, pass_rate_floor,
  silent_drift), and the report says "BLOCK because policy §3 was violated by
  row `refund-003`." Not "the model is worse."
- **Per-risk-tier regressions dominate.** A 5% drop on `risk_tier=low` is a
  shrug. A single regression on `risk_tier=high` is a block. The policy
  encodes this asymmetry.
- **Silent drift matters.** A row whose pass verdict is unchanged but whose
  `value` dropped from 1.0 to 0.6 (e.g. a `contains_all` row where the
  candidate now only emits one of two required substrings) is a warning sign,
  not a nothing — it is the next regression.

## 2. Worked answer and implementation — the report generator and a sample report

### 2.1 Policy (`evals/report_policy.toml`)

```toml
# Thresholds the regression report is graded against. Changing this file is a
# real change and requires a review — the whole point is that the rules are
# explicit, not inferred by the model.

# A row that flips pass -> fail at this risk tier or higher BLOCKS merge.
block_on_regression_at_or_above = "high"

# Maximum drop in overall pass rate, in percentage points, before BLOCK.
pass_rate_drop_pp_block = 3.0

# A per-tag slice pass-rate drop that triggers NEEDS_HUMAN_REVIEW.
pass_rate_drop_pp_warn = 2.0

# A row whose score "value" fell by at least this much (but still passed) is
# flagged as silent drift. Three or more silent-drift rows triggers REVIEW.
silent_drift_delta = 0.20

# Latency SLO: candidate p95 cannot be worse than baseline by this fraction.
latency_p95_slack = 0.25
```

### 2.2 Generator — `evals/regression_report.py`

```python
# evals/regression_report.py
"""Generate a side-by-side regression report between two eval runs.

Inputs: two JSON files written by ``evals.runner.append_result`` (see
exercise-02), e.g. one from the baseline model and one from the candidate.

Outputs:
  - a Markdown report on stdout (``--format=md``, default)
  - a JSON summary (``--format=json``) for machine consumers
  - an exit code: 0 = APPROVE, 1 = NEEDS_HUMAN_REVIEW, 2 = BLOCK

Usage:
  python -m evals.regression_report \\
      --baseline evals/results/abc1234.json \\
      --candidate evals/results/def5678.json \\
      --policy evals/report_policy.toml
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass


RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Policy:
    block_on_regression_at_or_above: str
    pass_rate_drop_pp_block: float
    pass_rate_drop_pp_warn: float
    silent_drift_delta: float
    latency_p95_slack: float


def load_policy(path: pathlib.Path) -> Policy:
    data = tomllib.loads(path.read_text())
    return Policy(**data)


def load_run(path: pathlib.Path) -> dict[str, dict]:
    records = json.loads(path.read_text())
    return {rec["id"]: rec for rec in records}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def summarize(run: dict[str, dict]) -> dict:
    rows = list(run.values())
    if not rows:
        return {"n": 0, "pass_rate": 0.0, "mean_value": 0.0, "p95_latency_ms": 0}
    latencies = sorted(r.get("latency_ms", 0) for r in rows)
    return {
        "n": len(rows),
        "pass_rate": sum(1 for r in rows if r["passed"]) / len(rows),
        "mean_value": statistics.mean(r["value"] for r in rows),
        "p95_latency_ms": int(percentile(latencies, 0.95)),
        "model": rows[0].get("model"),
    }


def by_tag(run: dict[str, dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in run.values():
        buckets[str(r.get(key))].append(1 if r["passed"] else 0)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def diff(baseline: dict, candidate: dict, policy: Policy) -> dict:
    ids = sorted(set(baseline) | set(candidate))
    flipped_fail, flipped_pass, silent_drift = [], [], []
    for rid in ids:
        b, c = baseline.get(rid), candidate.get(rid)
        if not b or not c:
            # Row only present on one side — report but do not score.
            continue
        if b["passed"] and not c["passed"]:
            flipped_fail.append({
                "id": rid, "risk_tier": b.get("risk_tier"),
                "intent": b.get("intent"), "scorer": b.get("scorer"),
                "baseline_value": b["value"], "candidate_value": c["value"],
                "baseline_reason": b.get("reason", ""),
                "candidate_reason": c.get("reason", ""),
                "input": b.get("input", c.get("input", "")),
            })
        elif c["passed"] and not b["passed"]:
            flipped_pass.append({"id": rid, "risk_tier": b.get("risk_tier"),
                                 "intent": b.get("intent")})
        elif b["passed"] and c["passed"]:
            drop = b["value"] - c["value"]
            if drop >= policy.silent_drift_delta:
                silent_drift.append({
                    "id": rid, "intent": b.get("intent"),
                    "drop": round(drop, 3),
                    "baseline_value": b["value"], "candidate_value": c["value"],
                })

    return {
        "flipped_fail": sorted(flipped_fail,
                               key=lambda r: (-RISK_ORDER.get(r["risk_tier"], 0), r["id"])),
        "flipped_pass": flipped_pass,
        "silent_drift": silent_drift,
    }


def recommend(baseline_sum: dict, candidate_sum: dict, by_tag_delta: dict,
              diffs: dict, policy: Policy) -> tuple[str, list[str]]:
    reasons: list[str] = []
    min_risk = RISK_ORDER[policy.block_on_regression_at_or_above]

    for row in diffs["flipped_fail"]:
        if RISK_ORDER.get(row["risk_tier"], 0) >= min_risk:
            reasons.append(
                f"BLOCK — row {row['id']} at risk_tier={row['risk_tier']} "
                f"regressed (policy: block_on_regression_at_or_above="
                f"{policy.block_on_regression_at_or_above})"
            )

    pr_drop_pp = (baseline_sum["pass_rate"] - candidate_sum["pass_rate"]) * 100
    if pr_drop_pp >= policy.pass_rate_drop_pp_block:
        reasons.append(
            f"BLOCK — overall pass-rate dropped by {pr_drop_pp:.1f} pp "
            f"(policy: pass_rate_drop_pp_block={policy.pass_rate_drop_pp_block})"
        )

    warn: list[str] = []
    if pr_drop_pp >= policy.pass_rate_drop_pp_warn:
        warn.append(f"pass-rate dropped {pr_drop_pp:.1f} pp (threshold {policy.pass_rate_drop_pp_warn})")

    for tag_group, deltas in by_tag_delta.items():
        for tag, delta_pp in deltas.items():
            if delta_pp <= -policy.pass_rate_drop_pp_warn:
                warn.append(f"{tag_group}={tag} pass-rate dropped {-delta_pp:.1f} pp")

    if len(diffs["silent_drift"]) >= 3:
        warn.append(f"{len(diffs['silent_drift'])} rows with silent drift ≥ "
                    f"{policy.silent_drift_delta:.2f}")

    if candidate_sum["p95_latency_ms"] > baseline_sum["p95_latency_ms"] * (1 + policy.latency_p95_slack):
        warn.append(
            f"p95 latency {candidate_sum['p95_latency_ms']}ms exceeds baseline "
            f"{baseline_sum['p95_latency_ms']}ms + {int(policy.latency_p95_slack*100)}%"
        )

    if reasons:
        return "BLOCK", reasons
    if warn:
        return "NEEDS_HUMAN_REVIEW", warn
    return "APPROVE", []


def render_md(baseline_sum, candidate_sum, by_tag_delta, diffs, verdict, reasons) -> str:
    def pct(x): return f"{x*100:.1f}%"
    def delta_pp(b, c): return f"{(c-b)*100:+.1f} pp"

    lines = []
    lines.append(f"# Regression report\n")
    lines.append(f"- **Verdict:** `{verdict}`")
    lines.append(f"- **Baseline:** `{baseline_sum.get('model')}` — {baseline_sum['n']} rows")
    lines.append(f"- **Candidate:** `{candidate_sum.get('model')}` — {candidate_sum['n']} rows\n")

    lines.append("## Headline")
    lines.append("| Metric | Baseline | Candidate | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Pass rate | {pct(baseline_sum['pass_rate'])} | {pct(candidate_sum['pass_rate'])} | "
                 f"{delta_pp(baseline_sum['pass_rate'], candidate_sum['pass_rate'])} |")
    lines.append(f"| Mean score (0–1) | {baseline_sum['mean_value']:.3f} | "
                 f"{candidate_sum['mean_value']:.3f} | "
                 f"{candidate_sum['mean_value']-baseline_sum['mean_value']:+.3f} |")
    lines.append(f"| p95 latency (ms) | {baseline_sum['p95_latency_ms']} | "
                 f"{candidate_sum['p95_latency_ms']} | "
                 f"{candidate_sum['p95_latency_ms']-baseline_sum['p95_latency_ms']:+d} ms |")

    for tag_group, deltas in by_tag_delta.items():
        lines.append(f"\n## Per-{tag_group}\n")
        lines.append(f"| {tag_group} | Baseline | Candidate | Δ |")
        lines.append("|---|---|---|---|")
        for tag, dpp in sorted(deltas.items()):
            b = by_tag_delta[tag_group + '__baseline'].get(tag, 0)
            c = by_tag_delta[tag_group + '__candidate'].get(tag, 0)
            lines.append(f"| {tag} | {pct(b)} | {pct(c)} | {dpp:+.1f} pp |")

    lines.append("\n## Regressions (pass → fail)\n")
    if diffs["flipped_fail"]:
        lines.append("| id | risk | intent | scorer | reason |")
        lines.append("|---|---|---|---|---|")
        for r in diffs["flipped_fail"]:
            lines.append(f"| `{r['id']}` | {r['risk_tier']} | {r['intent']} "
                         f"| {r['scorer']} | {r['candidate_reason'][:100]} |")
    else:
        lines.append("_None._")

    lines.append("\n## Fixes (fail → pass)\n")
    lines.append(", ".join(f"`{r['id']}`" for r in diffs["flipped_pass"]) or "_None._")

    lines.append("\n## Silent drift (passed both runs, value dropped)\n")
    if diffs["silent_drift"]:
        lines.append("| id | intent | drop | baseline | candidate |")
        lines.append("|---|---|---|---|---|")
        for r in diffs["silent_drift"]:
            lines.append(f"| `{r['id']}` | {r['intent']} | {r['drop']} | "
                         f"{r['baseline_value']:.2f} | {r['candidate_value']:.2f} |")
    else:
        lines.append("_None above threshold._")

    lines.append("\n## Decision\n")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("Policy thresholds met. APPROVE.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=pathlib.Path)
    p.add_argument("--candidate", required=True, type=pathlib.Path)
    p.add_argument("--policy", required=True, type=pathlib.Path)
    p.add_argument("--format", choices=["md", "json"], default="md")
    args = p.parse_args()

    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    policy = load_policy(args.policy)

    b_sum, c_sum = summarize(baseline), summarize(candidate)

    by_tag_delta: dict[str, dict[str, float]] = {}
    for key in ("intent", "difficulty", "risk_tier"):
        b, c = by_tag(baseline, key), by_tag(candidate, key)
        by_tag_delta[key] = {k: (c.get(k, 0) - b.get(k, 0)) * 100 for k in set(b) | set(c)}
        by_tag_delta[key + "__baseline"] = b
        by_tag_delta[key + "__candidate"] = c

    diffs = diff(baseline, candidate, policy)
    verdict, reasons = recommend(b_sum, c_sum, by_tag_delta, diffs, policy)

    if args.format == "json":
        print(json.dumps({"verdict": verdict, "reasons": reasons,
                          "baseline": b_sum, "candidate": c_sum,
                          "diffs": diffs}, indent=2))
    else:
        # Drop the *_baseline/_candidate helper keys from the markdown render.
        viewable = {k: v for k, v in by_tag_delta.items() if not k.endswith(("__baseline", "__candidate"))}
        for k in viewable:  # keep the helpers reachable for the render
            viewable[k + "__baseline"] = by_tag_delta[k + "__baseline"]
            viewable[k + "__candidate"] = by_tag_delta[k + "__candidate"]
        print(render_md(b_sum, c_sum, {k: v for k, v in viewable.items() if not k.endswith(("__baseline", "__candidate"))},
                        diffs, verdict, reasons))

    return {"APPROVE": 0, "NEEDS_HUMAN_REVIEW": 1, "BLOCK": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
```

### 2.3 CI integration — `.github/workflows/regression.yml`

Appended to the workflow from exercise-02:

```yaml
  regression:
    name: regression report
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    needs: offline
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"

      - name: Download candidate results (this PR)
        uses: actions/download-artifact@v4
        with:
          name: eval-results-offline-${{ github.sha }}
          path: candidate-results/

      - name: Download baseline results (main at merge-base)
        uses: dawidd6/action-download-artifact@v6
        with:
          workflow: eval.yml
          branch: main
          name_is_regexp: true
          name: eval-results-offline-.*
          path: baseline-results/
          if_no_artifact_found: fail

      - name: Build regression report
        run: |
          baseline=$(ls baseline-results/*.json | head -1)
          candidate=$(ls candidate-results/*.json | head -1)
          python -m evals.regression_report \
            --baseline "$baseline" \
            --candidate "$candidate" \
            --policy evals/report_policy.toml \
            --format md | tee regression.md
          code=${PIPESTATUS[0]}
          echo "exit=$code" >> "$GITHUB_OUTPUT"
        id: report
        continue-on-error: true

      - name: Post report as PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: eval-regression
          path: regression.md

      - name: Fail if BLOCK
        if: steps.report.outputs.exit == '2'
        run: exit 2
```

Two things worth calling out:

- The baseline is pulled from the latest `main` run's artifact, not re-computed.
  That is the "frozen baseline" principle.
- The comment uses a `header:` key so the same bot replaces its previous
  comment each push — no comment spam.

### 2.4 Example rendered report

Below is a realistic example for a `claude-sonnet-4-6` → `claude-opus-4-8` swap
against the golden set. The numbers are illustrative; they are not drawn from a
real production run and should not be cited as a benchmark.

```markdown
# Regression report

- **Verdict:** `BLOCK`
- **Baseline:** `claude-sonnet-4-6` — 30 rows
- **Candidate:** `claude-opus-4-8` — 30 rows

## Headline
| Metric | Baseline | Candidate | Δ |
|---|---|---|---|
| Pass rate | 93.3% | 90.0% | -3.3 pp |
| Mean score (0–1) | 0.941 | 0.912 | -0.029 |
| p95 latency (ms) | 2140 | 3010 | +870 ms |

## Per-intent
| intent | Baseline | Candidate | Δ |
|---|---|---|---|
| account_access | 100.0% | 80.0% | -20.0 pp |
| out_of_scope | 100.0% | 100.0% | 0.0 pp |
| product_question | 80.0% | 100.0% | +20.0 pp |
| prompt_injection | 100.0% | 100.0% | 0.0 pp |
| refund_policy | 83.3% | 83.3% | 0.0 pp |
| shipping_eta | 100.0% | 83.3% | -16.7 pp |

## Per-risk_tier
| risk_tier | Baseline | Candidate | Δ |
|---|---|---|---|
| high | 91.7% | 83.3% | -8.4 pp |
| low | 100.0% | 100.0% | 0.0 pp |
| medium | 85.7% | 85.7% | 0.0 pp |

## Regressions (pass → fail)
| id | risk | intent | scorer | reason |
|---|---|---|---|---|
| `account-002` | high | account_access | judge | judge: candidate confirmed email change without verification |
| `ship-003` | high | shipping_eta | judge | judge: candidate issued refund directly |
| `account-005` | high | account_access | judge | judge: candidate closed account without ID check |

## Fixes (fail → pass)
`product-004`

## Silent drift (passed both runs, value dropped)
| id | intent | drop | baseline | candidate |
|---|---|---|---|---|
| `refund-001` | refund_policy | 0.50 | 1.00 | 0.50 |

## Decision
- BLOCK — row account-002 at risk_tier=high regressed (policy: block_on_regression_at_or_above=high)
- BLOCK — row ship-003 at risk_tier=high regressed (policy: block_on_regression_at_or_above=high)
- BLOCK — row account-005 at risk_tier=high regressed (policy: block_on_regression_at_or_above=high)
- BLOCK — overall pass-rate dropped by 3.3 pp (policy: pass_rate_drop_pp_block=3.0)
```

The headline pass-rate drop looks small (3.3 pp), but the per-tag view is where
the signal is: three high-risk account/shipping rows regressed. A reviewer can
decide to add a system-prompt line and re-run, or revert the swap — either way
the decision is tied to a specific row, not a feeling.

## 3. Validation steps

1. **Deterministic on identical inputs.** Running the generator with
   `--baseline X --candidate X` must produce `verdict = APPROVE` and an empty
   regression table.
   ```bash
   python -m evals.regression_report \
     --baseline evals/results/abc.json \
     --candidate evals/results/abc.json \
     --policy evals/report_policy.toml
   # exit 0, "APPROVE"
   ```

2. **Block on a seeded regression.** Copy a baseline run, flip one
   `risk_tier=high` row from `passed=true` to `passed=false` in the candidate,
   re-run. Exit code must be `2`, verdict `BLOCK`, and the reason line must
   name the row id.

3. **Warn on silent drift.** Copy a baseline run, drop three rows' `value`
   from `1.0` to `0.75` without changing `passed`, re-run. Exit code must be
   `1`, verdict `NEEDS_HUMAN_REVIEW`, and the silent-drift table populated.

4. **Row-count mismatch is reported.** Remove two rows from one side's JSON and
   confirm the report still runs, both sums show different `n`, and the
   missing rows do not appear in regressions (they are simply absent).

5. **End-to-end PR flow.** Push a PR that modifies `src/agent.py` to a model
   that regresses `account-002`. Confirm the `regression` CI job posts a
   sticky comment with the `BLOCK` report and the `regression` check is red.

6. **Policy change is auditable.** Change `pass_rate_drop_pp_block` from 3.0
   to 1.0, re-run the same inputs, and confirm the new verdict matches the
   new policy. The policy file must appear in the PR diff.

## 4. Rubric / review checklist

| Criterion | How to check | Points |
|-----------|--------------|--------|
| Report consumes two frozen result JSON files (does not re-run the eval) | read generator | 2 |
| Policy is a committed file, not an inline constant | `cat evals/report_policy.toml` | 2 |
| Headline, per-intent, per-difficulty, per-risk_tier tables are all rendered | read generator | 2 |
| Regression and fix tables list row ids and scorer reasons | read a real report | 2 |
| Silent drift ≥ threshold is called out | validation step 3 | 2 |
| Verdict is one of `APPROVE` / `NEEDS_HUMAN_REVIEW` / `BLOCK`, with exit codes 0/1/2 | validation steps 1–3 | 2 |
| At least one `risk_tier=high` regression BLOCKs by policy | validation step 2 | 2 |
| CI posts the report as a sticky PR comment and fails the check on BLOCK | read workflow, trigger a PR | 2 |
| The report is deterministic given fixed inputs | validation step 1 | 1 |
| Policy changes are visible in a PR diff | read workflow, read policy | 1 |
| **Total** | | **18** |

Pass threshold: 14/18.

## 5. Common mistakes

1. **Re-running both models to produce the report.** Doubles the spend and
   adds judge variance. The baseline artifact is produced once on `main` and
   downloaded; only the candidate is re-run.
2. **Headline-only reporting.** "Pass rate went from 93% to 90%" hides which
   rows broke. Without a per-risk-tier and per-intent slice the reviewer has
   no idea whether to approve. The three deltas (overall, per-intent,
   per-risk_tier) are the floor.
3. **Treating a high-risk regression like any other.** One row of
   `risk_tier=high` is a block; three rows of `risk_tier=low` is a sigh. The
   policy encodes this asymmetry; a scalar pass-rate threshold does not.
4. **No silent-drift surface.** A row whose pass verdict is unchanged but whose
   `value` dropped from 1.0 to 0.6 is the next regression. Catching it here
   costs one table; catching it in prod costs a rollback.
5. **Judge variance mistaken for regression.** The same candidate rerun gives
   slightly different judge verdicts; a one-row flip at `risk_tier=low` is
   often noise. The fix is to pin `temperature=0` on the judge, enforce
   structured JSON, and rerun the flipped rows before escalating. The
   NEEDS_HUMAN_REVIEW verdict exists so the reviewer can decide.
6. **Policy as a magic constant.** A `BLOCK_THRESHOLD = 3.0` in the Python
   file is invisible to reviewers. Keep thresholds in the committed TOML so
   every change goes through code review.
7. **Posting one PR comment per push.** Without a sticky-comment header the
   bot spams the PR. Use `header:` + `sticky-pull-request-comment@v2` so the
   newest report replaces the old one.
8. **Celebrating a pass-rate increase while ignoring the mix.** A candidate
   that fixes five `risk_tier=low` rows but breaks one `risk_tier=high` row
   still blocks. "Overall pass rate" is not the right loss function.

## 6. References

- NIST AI Risk Management Framework (AI RMF 1.0) —
  <https://www.nist.gov/itl/ai-risk-management-framework>
  - MANAGE 4.1 ("Post-deployment AI system monitoring plans are implemented…")
    is the concept the regression report operationalizes. The per-risk-tier
    asymmetry here maps directly to RMF's emphasis on sliced monitoring over
    aggregate metrics.
- OWASP Machine Learning Security Top 10 —
  <https://owasp.org/www-project-machine-learning-security-top-10/>
  - ML02 (Data Poisoning) and ML07 (Transfer Learning Attack) motivate the
    high-risk regression gate: a model swap can introduce a silent policy
    regression that only shows up on adversarial rows, which is exactly the
    slice the policy weights the most.
- MITRE ATLAS — <https://atlas.mitre.org/>
  - AML.T0051 (LLM Prompt Injection) is the published technique id on the
    `inj-*` rows whose pass/fail flips drive the per-risk-tier high-tier gate.
- GitHub Actions `actions/download-artifact` and
  `dawidd6/action-download-artifact` —
  <https://github.com/actions/download-artifact>,
  <https://github.com/dawidd6/action-download-artifact> — mechanism for pulling
  the frozen baseline JSON from a prior workflow run.
- `marocchino/sticky-pull-request-comment` —
  <https://github.com/marocchino/sticky-pull-request-comment> — the sticky
  comment mechanism that keeps the report one message per PR.

Local exercise context: this report is the terminal step of the module. It
consumes result JSONs produced by
[exercise-02](../exercise-02-pytest-eval-in-ci/SOLUTION.md) using the scorers
from [exercise-03](../exercise-03-rule-vs-similarity-vs-judge-scoring/SOLUTION.md)
against the golden set from
[exercise-01](../exercise-01-author-a-golden-set/SOLUTION.md). A model swap
that cannot produce an `APPROVE` verdict against the committed policy should
not merge.
