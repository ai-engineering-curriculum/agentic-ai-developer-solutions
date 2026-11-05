# project-103-multi-provider-comparison-report — Solution

## 1. Solution overview

The project is a single question asked three ways: *given our `support-agent`
and our golden set, which provider should we ship, and can we defend the
choice six months from now?* "Which provider" has to collapse into one number
only when every input to that number — quality on each slice, cost per 1,000
runs, p95 latency, and the committed selection policy — has been written down.

The report is the artifact that answers the question. It consumes **N frozen
eval-result files**, one per provider, each produced by the mod-109 pytest
harness. It emits:

1. **A side-by-side headline table** — pass rate, mean score, p95 latency, and
   cost per 100 runs, with the baseline provider's column marked.
2. **Per-tag breakdowns** — pass-rate deltas sliced by `intent`,
   `difficulty`, and `risk_tier`, because aggregate numbers hide where the
   model is actually weak.
3. **Per-provider cost ledger** — input + output token totals pulled from each
   SDK's `usage` payload, multiplied by the posted provider price, with the
   price source (URL and date fetched) stamped into the report so a reviewer
   can audit it.
4. **A selection recommendation** — `RECOMMEND <provider>`, `TIE`, or
   `NEEDS_HUMAN_REVIEW`, derived from a committed TOML policy. The policy has
   explicit weights for quality, cost, latency, and a hard veto on any
   `risk_tier=high` failure.

Design points that matter, drawn directly from the prerequisite modules:

- **One provider abstraction, three adapters.** The agent code from mod-106
  already calls `run_agent(messages, model=...)`; the project adds a thin
  `Provider` adapter per vendor (Anthropic, OpenAI, Google) that normalises
  message shape, tool-call shape, and the `usage` block. The agent itself does
  not learn about new SDKs — it only learns about a `provider_id` string.
- **Frozen artifacts, not live re-runs.** As in mod-109 exercise-04, the
  report diffs stored JSON files; it never calls a model itself. Re-running
  three providers on report generation triples the spend and makes two
  report renderings disagree because of judge variance. Each provider is
  evaluated once, the result file is archived as a CI artifact, the report
  consumes those artifacts.
- **Cost is a committed table, not an SDK call.** No provider exposes
  authoritative per-model pricing through its SDK, and scraping is brittle.
  The project keeps pricing in `evals/pricing.toml`; updates are PRs. The
  report stamps `pricing.toml`'s `source_url` and `fetched_on` into every
  rendering so the number is auditable.
- **Latency is measured, not guessed.** p50 and p95 come from the mod-107
  exercise-05 harness pattern (`time.perf_counter`, SDK retries off, warm-up
  rows discarded). A headline latency column that just reports the eval's
  mean wall-clock would silently include provider cold starts and the
  retry-policy smear.
- **"RECOMMEND" is a policy output, not a model's opinion.** The selection
  policy (`evals/selection_policy.toml`) has four knobs: `veto_on_risk_tier`,
  `min_pass_rate`, `quality_weight`, `cost_weight`, `latency_weight`. The
  report names which knob fired, not "provider X feels better."

The anti-goal of this project is "a benchmark leaderboard screenshot." A
leaderboard without the golden set it was scored against, without the system
prompt that was used, without the pricing date, and without the policy that
turned numbers into a decision is marketing. The deliverable here is the
opposite: a reviewable, reproducible, policy-gated report.

## 2. Worked answer and implementation

### 2.1 Repository layout

The project builds on the `support-agent` from mod-106. Only the new and
changed files are shown.

```text
support-agent/
├── pyproject.toml                 # adds openai, google-genai dependencies
├── src/
│   ├── agent.py                   # mod-106; now dispatches by provider_id
│   └── providers/
│       ├── __init__.py            # Provider protocol + registry
│       ├── anthropic_provider.py
│       ├── openai_provider.py
│       └── google_provider.py
├── evals/
│   ├── golden.jsonl               # from mod-109 exercise-01, unchanged
│   ├── scoring.py                 # from mod-109 exercise-03, unchanged
│   ├── runner.py                  # mod-109 exercise-02; parametrised on provider
│   ├── pricing.toml               # posted provider prices with source + date
│   ├── selection_policy.toml      # committed selection policy
│   ├── comparison_report.py       # N-way report generator (new)
│   └── results/                   # one JSON per (provider, git-sha)
├── tests/
│   └── eval/
│       └── test_comparison.py     # smoke test for the report generator
└── .github/
    └── workflows/
        └── compare.yml            # per-provider eval + report job
```

### 2.2 The provider abstraction — `src/providers/__init__.py`

Three constraints make the abstraction worth writing instead of branching the
agent on `if provider == "anthropic"`:

- Each vendor's SDK has a different message schema, a different tool-call
  shape, and a different `usage` block.
- The agent code from mod-106 is already one function; growing it with vendor
  branches breaks mod-106 tests.
- The eval runner needs to iterate `for p in PROVIDERS: ...` without caring
  about any of the above.

```python
# src/providers/__init__.py
"""Provider adapters.

A ``Provider`` normalises three things across vendors:
  - request shape (messages, system prompt, tools, temperature)
  - response shape (text + tool calls)
  - the ``usage`` block (input tokens, output tokens, cached tokens)

The agent code calls ``provider.complete(request)`` and gets a
``ProviderResponse``. It does NOT import vendor SDKs directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ProviderRequest:
    model: str
    system: str
    messages: list[dict[str, Any]]            # {"role": "user"|"assistant", "content": str}
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 1024


@dataclass
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0              # Anthropic cache_read / OpenAI cached_tokens


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[dict[str, Any]]          # normalised: {"name", "arguments": dict, "id"}
    stop_reason: str                          # "end_turn" | "tool_use" | "max_tokens" | "other"
    usage: ProviderUsage
    raw: Any                                  # vendor-native object, for debugging


class Provider(Protocol):
    id: str                                   # "anthropic" | "openai" | "google"
    default_model: str

    def complete(self, request: ProviderRequest) -> ProviderResponse: ...


# Populated by anthropic_provider.py / openai_provider.py / google_provider.py
# at import time.
REGISTRY: dict[str, Provider] = {}


def register(provider: Provider) -> None:
    REGISTRY[provider.id] = provider


def get(provider_id: str) -> Provider:
    try:
        return REGISTRY[provider_id]
    except KeyError:
        raise KeyError(
            f"Unknown provider {provider_id!r}; registered: {sorted(REGISTRY)}"
        ) from None
```

### 2.3 The Anthropic adapter — `src/providers/anthropic_provider.py`

```python
# src/providers/anthropic_provider.py
"""Anthropic adapter.

SDK: ``anthropic`` (official).
API: Messages API — ``client.messages.create(...)``.
Docs: https://docs.claude.com/en/api/messages
"""
from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic

from . import Provider, ProviderRequest, ProviderResponse, ProviderUsage, register


@dataclass
class AnthropicProvider:
    id: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"

    def __post_init__(self) -> None:
        # Retries off: the measurement is of the model, not of the SDK's
        # retry policy. Mod-107 exercise-05 covers this at length.
        self._client = Anthropic(max_retries=0)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        # Anthropic requires system as a top-level kwarg, not a message.
        msg = self._client.messages.create(
            model=request.model or self.default_model,
            system=request.system,
            messages=request.messages,
            tools=request.tools or None,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
        )

        # Flatten the content blocks into text + tool_use.
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in msg.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"name": block.name, "arguments": block.input, "id": block.id}
                )

        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=msg.stop_reason or "other",
            usage=ProviderUsage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                cached_input_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            ),
            raw=msg,
        )


register(AnthropicProvider())
```

### 2.4 The OpenAI adapter — `src/providers/openai_provider.py`

```python
# src/providers/openai_provider.py
"""OpenAI adapter.

SDK: ``openai`` (official).
API: Chat Completions — ``client.chat.completions.create(...)``.
Docs: https://platform.openai.com/docs/api-reference/chat
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from . import Provider, ProviderRequest, ProviderResponse, ProviderUsage, register


@dataclass
class OpenAIProvider:
    id: str = "openai"
    default_model: str = "gpt-4o"

    def __post_init__(self) -> None:
        self._client = OpenAI(max_retries=0)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        # OpenAI carries the system prompt as the first message.
        messages = [{"role": "system", "content": request.system}, *request.messages]

        # Normalise tool schema: the project's canonical tool shape matches
        # Anthropic's; OpenAI wants it wrapped in {"type": "function", "function": {...}}.
        tools = (
            [{"type": "function",
              "function": {"name": t["name"],
                           "description": t.get("description", ""),
                           "parameters": t["input_schema"]}} for t in request.tools]
            if request.tools else None
        )

        resp = self._client.chat.completions.create(
            model=request.model or self.default_model,
            messages=messages,
            tools=tools,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
        )

        choice = resp.choices[0]
        tool_calls: list[dict] = []
        for tc in choice.message.tool_calls or []:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}"),
                "id": tc.id,
            })

        return ProviderResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "other",
            usage=ProviderUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                cached_input_tokens=getattr(
                    resp.usage.prompt_tokens_details, "cached_tokens", 0
                ) if getattr(resp.usage, "prompt_tokens_details", None) else 0,
            ),
            raw=resp,
        )


register(OpenAIProvider())
```

### 2.5 The Google adapter — `src/providers/google_provider.py`

```python
# src/providers/google_provider.py
"""Google adapter.

SDK: ``google-genai`` (official).
API: ``client.models.generate_content(...)``.
Docs: https://ai.google.dev/gemini-api/docs
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from . import Provider, ProviderRequest, ProviderResponse, ProviderUsage, register


@dataclass
class GoogleProvider:
    id: str = "google"
    default_model: str = "gemini-2.5-flash"      # <!-- lookup-pending: confirm current default -->

    def __post_init__(self) -> None:
        self._client = genai.Client()

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        # Gemini merges system instruction into generation_config.
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t["input_schema"],
                ) for t in request.tools])] if request.tools else None,
        )

        # Gemini wants Content objects, not dict messages.
        contents = [
            types.Content(role=("model" if m["role"] == "assistant" else "user"),
                          parts=[types.Part.from_text(m["content"])])
            for m in request.messages
        ]

        resp = self._client.models.generate_content(
            model=request.model or self.default_model,
            contents=contents,
            config=config,
        )

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for part in resp.candidates[0].content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            if getattr(part, "function_call", None):
                fc = part.function_call
                tool_calls.append({
                    "name": fc.name,
                    "arguments": dict(fc.args) if fc.args else {},
                    "id": fc.id or fc.name,
                })

        usage = resp.usage_metadata
        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=str(resp.candidates[0].finish_reason or "other"),
            usage=ProviderUsage(
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
                cached_input_tokens=usage.cached_content_token_count or 0,
            ),
            raw=resp,
        )


register(GoogleProvider())
```

### 2.6 The eval runner — `evals/runner.py` (parametrised on provider)

The only change from mod-109 exercise-02 is a `--provider` flag; the row
schema, scorer dispatch, and result-writer are unchanged. Snippet of the
delta:

```python
# evals/runner.py (delta; mod-109 version is the baseline)
import os
import pathlib
import time

from src.agent import run_agent
from src.providers import get as get_provider

RESULTS_DIR = pathlib.Path("evals/results")


def run_case(row: dict, provider_id: str) -> dict:
    provider = get_provider(provider_id)
    t0 = time.perf_counter()
    result = run_agent(
        row["input"],
        provider=provider,                  # mod-106 agent accepts a Provider now
        model=os.environ.get("MODEL_OVERRIDE"),
        temperature=0.0,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "id": row["id"],
        "provider": provider.id,
        "model": result.model,
        "input": row["input"],
        "actual": result.text,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "cached_input_tokens": result.usage.cached_input_tokens,
        "latency_ms": elapsed_ms,
        "intent": row["intent"],
        "difficulty": row["difficulty"],
        "risk_tier": row["risk_tier"],
        # scorer/value/passed/reason are filled by scoring.score_row
    }
```

Each CI matrix cell writes one file: `evals/results/<provider>-<sha>.json`.
These files are the inputs to the report.

### 2.7 Pricing — `evals/pricing.toml`

```toml
# Posted provider prices. Updating this file is a real change — the PR diff
# is the audit trail. The report stamps source_url and fetched_on into every
# rendering so the figure is auditable.
#
# Prices are in US dollars per 1,000,000 tokens. Leave the number blank
# (``price_per_1m_in = 0``) and add a ``<!-- lookup-pending: ... -->`` tag
# below the entry if the figure cannot be verified against the source URL.

[anthropic.claude-sonnet-4-6]
price_per_1m_in = 0           # <!-- lookup-pending: anthropic.com/pricing current sonnet-4-6 input rate -->
price_per_1m_out = 0          # <!-- lookup-pending: anthropic.com/pricing current sonnet-4-6 output rate -->
price_per_1m_cached_in = 0    # <!-- lookup-pending: prompt-caching read rate -->
source_url = "https://www.anthropic.com/pricing"
fetched_on = "2026-11-04"

[openai."gpt-4o"]
price_per_1m_in = 0           # <!-- lookup-pending: openai.com/api/pricing current gpt-4o input rate -->
price_per_1m_out = 0          # <!-- lookup-pending: openai.com/api/pricing current gpt-4o output rate -->
price_per_1m_cached_in = 0    # <!-- lookup-pending: cached input rate -->
source_url = "https://openai.com/api/pricing/"
fetched_on = "2026-11-04"

[google."gemini-2.5-flash"]
price_per_1m_in = 0           # <!-- lookup-pending: ai.google.dev/pricing current gemini-2.5-flash input rate -->
price_per_1m_out = 0          # <!-- lookup-pending: ai.google.dev/pricing current gemini-2.5-flash output rate -->
price_per_1m_cached_in = 0    # <!-- lookup-pending: cached input rate -->
source_url = "https://ai.google.dev/pricing"
fetched_on = "2026-11-04"
```

The committed file ships with zeros and `lookup-pending` markers; filling
in the actual numbers is a one-line PR that forces the reviewer to open the
`source_url` on `fetched_on`. That is the point. Hard-coding a number the
student typed from memory into Python is how a report ends up wrong by a
factor of three.

### 2.8 Selection policy — `evals/selection_policy.toml`

```toml
# Selection weights and vetoes. The comparison report consults this file to
# produce RECOMMEND / TIE / NEEDS_HUMAN_REVIEW. Thresholds are deliberately
# conservative; loosen them only in a PR a reviewer can sign off on.

# A provider that fails ANY row at or above this risk tier is vetoed. No
# amount of cost advantage can override a high-risk failure.
veto_on_risk_tier = "high"

# A provider with pass rate below this (after vetoes) is disqualified.
min_pass_rate = 0.85

# The composite score is w_quality * norm(pass_rate) +
#                       w_cost    * norm(1 / cost_per_100_runs) +
#                       w_latency * norm(1 / p95_latency_ms).
# Weights must sum to 1.0.
quality_weight = 0.6
cost_weight    = 0.25
latency_weight = 0.15

# The margin by which the top provider must beat the runner-up on composite
# score; below this it is a TIE and the report emits NEEDS_HUMAN_REVIEW so a
# human picks on criteria not captured by the policy.
min_composite_margin = 0.05
```

### 2.9 The comparison report generator — `evals/comparison_report.py`

```python
# evals/comparison_report.py
"""Generate an N-way provider comparison report.

Inputs (one per provider):
  - evals/results/<provider>-<sha>.json  — written by evals.runner

Outputs:
  - a Markdown report on stdout (--format=md, default)
  - a JSON summary (--format=json)
  - an exit code: 0 = RECOMMEND, 1 = TIE/NEEDS_HUMAN_REVIEW, 2 = NO_VIABLE_PROVIDER

Usage:
  python -m evals.comparison_report \\
      --result anthropic=evals/results/anthropic-abc.json \\
      --result openai=evals/results/openai-abc.json \\
      --result google=evals/results/google-abc.json \\
      --pricing evals/pricing.toml \\
      --policy  evals/selection_policy.toml
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
    veto_on_risk_tier: str
    min_pass_rate: float
    quality_weight: float
    cost_weight: float
    latency_weight: float
    min_composite_margin: float

    def __post_init__(self) -> None:
        total = self.quality_weight + self.cost_weight + self.latency_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"policy weights must sum to 1.0, got {total}")


def load_policy(path: pathlib.Path) -> Policy:
    return Policy(**tomllib.loads(path.read_text()))


def load_pricing(path: pathlib.Path) -> dict:
    return tomllib.loads(path.read_text())


def load_run(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def cost_per_100_runs(rows: list[dict], pricing: dict) -> tuple[float, list[str]]:
    """Return (cost_usd_per_100_runs, warnings).

    Warnings are strings; pricing rows with zero per-token rates are flagged so
    a reviewer sees the report cannot be trusted for cost ranking until the
    lookup-pending markers are resolved.
    """
    warnings: list[str] = []
    n = len(rows) or 1
    total_cost = 0.0
    for r in rows:
        model = r["model"]
        provider = r["provider"]
        try:
            prices = pricing[provider][model]
        except KeyError:
            warnings.append(f"missing pricing entry for {provider}.{model}")
            continue
        p_in = prices.get("price_per_1m_in", 0)
        p_out = prices.get("price_per_1m_out", 0)
        p_cache = prices.get("price_per_1m_cached_in", 0)
        if p_in == 0 and p_out == 0:
            warnings.append(f"zero prices for {provider}.{model} — unresolved lookup-pending")
        fresh_in = max(r["input_tokens"] - r.get("cached_input_tokens", 0), 0)
        cached = r.get("cached_input_tokens", 0)
        total_cost += (fresh_in * p_in + cached * p_cache + r["output_tokens"] * p_out) / 1e6
    return (total_cost / n) * 100, warnings


def summarize(rows: list[dict], pricing: dict) -> dict:
    if not rows:
        return {}
    latencies = [r["latency_ms"] for r in rows]
    cost_100, cost_warnings = cost_per_100_runs(rows, pricing)
    return {
        "n": len(rows),
        "provider": rows[0]["provider"],
        "model": rows[0]["model"],
        "pass_rate": sum(1 for r in rows if r["passed"]) / len(rows),
        "mean_value": statistics.mean(r["value"] for r in rows),
        "p50_latency_ms": int(percentile(latencies, 0.5)),
        "p95_latency_ms": int(percentile(latencies, 0.95)),
        "cost_per_100_runs_usd": round(cost_100, 4),
        "cost_warnings": cost_warnings,
        "high_risk_failures": sorted(
            r["id"] for r in rows
            if not r["passed"] and r.get("risk_tier") == "high"
        ),
    }


def by_tag(rows: list[dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key))].append(1 if r["passed"] else 0)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def composite(sums: list[dict], policy: Policy) -> dict[str, float]:
    """Return provider_id -> composite_score in [0, 1]."""
    def minmax_norm(xs: list[float], invert: bool = False) -> list[float]:
        lo, hi = min(xs), max(xs)
        if hi == lo:
            return [1.0 for _ in xs]
        normed = [(x - lo) / (hi - lo) for x in xs]
        return [1 - x for x in normed] if invert else normed

    pr = minmax_norm([s["pass_rate"] for s in sums])
    cost = minmax_norm([s["cost_per_100_runs_usd"] for s in sums], invert=True)
    lat = minmax_norm([float(s["p95_latency_ms"]) for s in sums], invert=True)
    return {
        s["provider"]: policy.quality_weight * pr[i]
                     + policy.cost_weight * cost[i]
                     + policy.latency_weight * lat[i]
        for i, s in enumerate(sums)
    }


def recommend(sums: list[dict], policy: Policy) -> tuple[str, str | None, list[str]]:
    """Return (verdict, recommended_provider_or_None, reasons)."""
    reasons: list[str] = []
    eligible: list[dict] = []

    for s in sums:
        if s["high_risk_failures"]:
            reasons.append(
                f"VETO {s['provider']} — failed risk_tier={policy.veto_on_risk_tier} "
                f"rows: {', '.join(s['high_risk_failures'])}"
            )
            continue
        if s["pass_rate"] < policy.min_pass_rate:
            reasons.append(
                f"VETO {s['provider']} — pass_rate {s['pass_rate']:.1%} below "
                f"min_pass_rate {policy.min_pass_rate:.0%}"
            )
            continue
        eligible.append(s)

    if not eligible:
        return "NO_VIABLE_PROVIDER", None, reasons

    scores = composite(eligible, policy)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_score = ranked[0]
    runner_margin = top_score - ranked[1][1] if len(ranked) > 1 else 1.0

    if runner_margin < policy.min_composite_margin:
        reasons.append(
            f"TIE — top margin {runner_margin:.3f} below "
            f"min_composite_margin {policy.min_composite_margin}"
        )
        return "NEEDS_HUMAN_REVIEW", None, reasons

    reasons.append(
        f"RECOMMEND {top_id} — composite {top_score:.3f} beats runner-up by "
        f"{runner_margin:.3f} (≥ {policy.min_composite_margin})"
    )
    return "RECOMMEND", top_id, reasons


def render_md(sums: list[dict], tag_tables: dict, verdict: str,
              chosen: str | None, reasons: list[str], pricing_meta: dict) -> str:
    def pct(x): return f"{x*100:.1f}%"

    lines: list[str] = []
    lines.append("# Multi-provider comparison report\n")
    lines.append(f"- **Verdict:** `{verdict}`"
                 + (f" — recommend `{chosen}`" if chosen else ""))
    lines.append(f"- **Providers evaluated:** "
                 + ", ".join(f"`{s['provider']}:{s['model']}`" for s in sums))
    lines.append(f"- **Rows per provider:** {sums[0]['n']} (identical golden set)")
    lines.append("")

    lines.append("## Pricing provenance")
    lines.append("| provider.model | source_url | fetched_on |")
    lines.append("|---|---|---|")
    for key, entry in pricing_meta.items():
        lines.append(f"| `{key}` | <{entry.get('source_url', '')}> | {entry.get('fetched_on', '?')} |")
    lines.append("")

    lines.append("## Headline")
    header = "| metric |" + "".join(f" {s['provider']} |" for s in sums)
    sep = "|---|" + "---|" * len(sums)
    lines.append(header)
    lines.append(sep)
    lines.append("| pass rate |" + "".join(f" {pct(s['pass_rate'])} |" for s in sums))
    lines.append("| mean score |" + "".join(f" {s['mean_value']:.3f} |" for s in sums))
    lines.append("| p50 latency |" + "".join(f" {s['p50_latency_ms']} ms |" for s in sums))
    lines.append("| p95 latency |" + "".join(f" {s['p95_latency_ms']} ms |" for s in sums))
    lines.append("| cost / 100 runs |" + "".join(f" ${s['cost_per_100_runs_usd']:.4f} |" for s in sums))
    lines.append("| high-risk failures |" + "".join(
        f" {len(s['high_risk_failures'])} |" for s in sums))
    lines.append("")

    for tag, tbl in tag_tables.items():
        lines.append(f"## Per-{tag}")
        header = f"| {tag} |" + "".join(f" {p} |" for p in tbl["providers"])
        lines.append(header)
        lines.append("|---|" + "---|" * len(tbl["providers"]))
        for row in tbl["rows"]:
            cells = " |".join(pct(x) for x in row["cells"])
            lines.append(f"| {row['tag']} | {cells} |")
        lines.append("")

    lines.append("## Decision")
    for r in reasons:
        lines.append(f"- {r}")

    cost_warnings = [w for s in sums for w in s.get("cost_warnings", [])]
    if cost_warnings:
        lines.append("")
        lines.append("> ⚠️ **Cost figures are unverified.**")
        for w in cost_warnings:
            lines.append(f"> - {w}")
        lines.append("> Resolve the `lookup-pending` markers in `pricing.toml` before using "
                     "this report to approve a provider switch.")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result", action="append", required=True,
                   help='<provider>=<path>, e.g. anthropic=evals/results/anthropic-abc.json')
    p.add_argument("--pricing", required=True, type=pathlib.Path)
    p.add_argument("--policy", required=True, type=pathlib.Path)
    p.add_argument("--format", choices=["md", "json"], default="md")
    args = p.parse_args()

    runs: list[list[dict]] = []
    for spec in args.result:
        if "=" not in spec:
            raise SystemExit(f"--result expects provider=path, got {spec!r}")
        _, path = spec.split("=", 1)
        runs.append(load_run(pathlib.Path(path)))

    pricing = load_pricing(args.pricing)
    policy = load_policy(args.policy)

    sums = [summarize(r, pricing) for r in runs]

    tag_tables: dict[str, dict] = {}
    for key in ("intent", "difficulty", "risk_tier"):
        provider_ids = [s["provider"] for s in sums]
        all_tags: set[str] = set()
        per_provider = []
        for rows in runs:
            buckets = by_tag(rows, key)
            per_provider.append(buckets)
            all_tags.update(buckets.keys())
        tag_tables[key] = {
            "providers": provider_ids,
            "rows": [
                {"tag": t,
                 "cells": [per_provider[i].get(t, 0.0) for i in range(len(provider_ids))]}
                for t in sorted(all_tags)
            ],
        }

    verdict, chosen, reasons = recommend(sums, policy)

    pricing_meta = {}
    for s in sums:
        entry = pricing.get(s["provider"], {}).get(s["model"], {})
        pricing_meta[f"{s['provider']}.{s['model']}"] = {
            "source_url": entry.get("source_url", ""),
            "fetched_on": entry.get("fetched_on", "?"),
        }

    if args.format == "json":
        print(json.dumps({
            "verdict": verdict,
            "recommended": chosen,
            "reasons": reasons,
            "providers": sums,
            "pricing_source": pricing_meta,
        }, indent=2))
    else:
        print(render_md(sums, tag_tables, verdict, chosen, reasons, pricing_meta))

    return {"RECOMMEND": 0, "NEEDS_HUMAN_REVIEW": 1,
            "NO_VIABLE_PROVIDER": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
```

### 2.10 CI integration — `.github/workflows/compare.yml`

```yaml
name: Multi-provider comparison

on:
  workflow_dispatch:
  schedule:
    - cron: "0 7 * * 1"   # weekly Monday rebuild
  pull_request:
    paths:
      - "src/providers/**"
      - "evals/pricing.toml"
      - "evals/selection_policy.toml"

jobs:
  eval:
    name: eval ${{ matrix.provider }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        provider: [anthropic, openai, google]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"

      - name: Run golden set against ${{ matrix.provider }}
        env:
          PROVIDER: ${{ matrix.provider }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: |
          pytest tests/eval \
            --provider="$PROVIDER" \
            --junitxml="junit-$PROVIDER.xml"

      - uses: actions/upload-artifact@v4
        with:
          name: eval-${{ matrix.provider }}-${{ github.sha }}
          path: evals/results/${{ matrix.provider }}-*.json
          if-no-files-found: error

  report:
    name: comparison report
    needs: eval
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"

      - uses: actions/download-artifact@v4
        with:
          pattern: eval-*-${{ github.sha }}
          path: evals/results/
          merge-multiple: true

      - name: Build comparison report
        id: report
        run: |
          python -m evals.comparison_report \
            --result anthropic=$(ls evals/results/anthropic-*.json | head -1) \
            --result openai=$(ls evals/results/openai-*.json | head -1) \
            --result google=$(ls evals/results/google-*.json | head -1) \
            --pricing evals/pricing.toml \
            --policy  evals/selection_policy.toml \
            --format md | tee comparison.md
          echo "exit=${PIPESTATUS[0]}" >> "$GITHUB_OUTPUT"
        continue-on-error: true

      - uses: marocchino/sticky-pull-request-comment@v2
        if: github.event_name == 'pull_request'
        with:
          header: provider-comparison
          path: comparison.md

      - uses: actions/upload-artifact@v4
        with:
          name: comparison-report-${{ github.sha }}
          path: comparison.md

      - name: Fail on NO_VIABLE_PROVIDER
        if: steps.report.outputs.exit == '2'
        run: exit 2
```

Two things worth calling out:

- The `pull_request` trigger is scoped to `paths:` that actually change the
  comparison — provider adapters, pricing, policy. Every PR does not need
  three API spends.
- The `report` job is isolated from provider failures: `fail-fast: false` on
  the matrix means one vendor going down still produces a report for the
  other two, and the reader sees two columns instead of nothing.

### 2.11 Example rendered report

Illustrative output. The pass rates and latencies below are *not* drawn from
a real run; they are structural examples so the format is concrete. Cost
numbers are deliberately `$0.0000` because the pricing TOML in this reference
ships with zeros and `lookup-pending` markers.

```markdown
# Multi-provider comparison report

- **Verdict:** `RECOMMEND` — recommend `anthropic`
- **Providers evaluated:** `anthropic:claude-sonnet-4-6`, `openai:gpt-4o`, `google:gemini-2.5-flash`
- **Rows per provider:** 30 (identical golden set)

## Pricing provenance
| provider.model | source_url | fetched_on |
|---|---|---|
| `anthropic.claude-sonnet-4-6` | <https://www.anthropic.com/pricing> | 2026-11-04 |
| `openai.gpt-4o`               | <https://openai.com/api/pricing/>  | 2026-11-04 |
| `google.gemini-2.5-flash`     | <https://ai.google.dev/pricing>    | 2026-11-04 |

## Headline
| metric | anthropic | openai | google |
|---|---|---|---|
| pass rate | 93.3% | 90.0% | 86.7% |
| mean score | 0.941 | 0.912 | 0.878 |
| p50 latency | 980 ms | 1120 ms | 740 ms |
| p95 latency | 2140 ms | 3010 ms | 1890 ms |
| cost / 100 runs | $0.0000 | $0.0000 | $0.0000 |
| high-risk failures | 0 | 1 | 2 |

## Per-risk_tier
| risk_tier | anthropic | openai | google |
|---|---|---|---|
| high | 100.0% | 91.7% | 83.3% |
| low | 100.0% | 100.0% | 100.0% |
| medium | 85.7% | 85.7% | 85.7% |

## Decision
- VETO openai — failed risk_tier=high rows: account-002
- VETO google — failed risk_tier=high rows: account-002, account-005
- RECOMMEND anthropic — composite 1.000 beats runner-up by 1.000 (≥ 0.05)

> ⚠️ **Cost figures are unverified.**
> - zero prices for anthropic.claude-sonnet-4-6 — unresolved lookup-pending
> - zero prices for openai.gpt-4o — unresolved lookup-pending
> - zero prices for google.gemini-2.5-flash — unresolved lookup-pending
> Resolve the `lookup-pending` markers in `pricing.toml` before using this report to approve a provider switch.
```

The report body is honest about its own gaps: the recommendation is driven
entirely by quality here because every provider's cost column reads `$0.0000`.
A reviewer who ignores the cost warning and signs off on this report has
signed off on a decision made without one of its three inputs. That is
deliberate — the warning is the backstop against "we'll fill in pricing
later."

## 3. Validation steps

1. **Adapters produce identical shapes.** Run each provider against the same
   10-row subset of `golden.jsonl`, inspect the result files, and confirm the
   field set is identical (`input_tokens`, `output_tokens`, `latency_ms`,
   `provider`, `model`, `passed`, `value`, `reason`, `risk_tier`, `intent`,
   `difficulty`). Missing fields on any provider must fail the pytest run,
   not slip into the report.

2. **Identical inputs ⇒ zero-margin tie.** Point `--result` at the *same*
   result file for three provider names:
   ```bash
   python -m evals.comparison_report \
     --result a=evals/results/anthropic-abc.json \
     --result b=evals/results/anthropic-abc.json \
     --result c=evals/results/anthropic-abc.json \
     --pricing evals/pricing.toml --policy evals/selection_policy.toml
   ```
   Verdict must be `NEEDS_HUMAN_REVIEW` with the `TIE — top margin` reason,
   exit code `1`.

3. **Veto on risk_tier=high.** Hand-edit one provider's result file to set
   `"passed": false, "risk_tier": "high"` on one row and re-run the report.
   Verdict must list a `VETO <provider>` reason naming the row id, and the
   `high-risk failures` headline cell must read `1`.

4. **NO_VIABLE_PROVIDER when everyone is vetoed.** Hand-edit every provider's
   result file to add one `risk_tier=high` failure. Verdict must be
   `NO_VIABLE_PROVIDER`, exit code `2`.

5. **Pricing warnings are prominent.** With the shipped zero-price
   `pricing.toml`, every rendered report must carry the ⚠️ warning block
   naming each provider.model with unresolved pricing. Resolving one entry
   (filling in a non-zero rate) must remove exactly that line from the warning
   block.

6. **Deterministic rendering.** Re-running the generator against the same
   three result files, same pricing, same policy must produce byte-identical
   Markdown. This matters for CI diffs and for sticky PR comments — if the
   bot keeps editing the comment because the numbers wiggle, the report is
   not deterministic.

7. **CI matrix survives one vendor outage.** Set `ANTHROPIC_API_KEY=invalid`
   in a dispatch run, confirm the `eval anthropic` cell fails, and confirm
   the `report` job still produces a two-provider comparison. The matrix
   must be `fail-fast: false` for this to work.

8. **Policy change is auditable.** Swap `quality_weight = 0.6` to `0.3` and
   `cost_weight = 0.25` to `0.55`, re-run the report on the same artifacts,
   and confirm the composite ranking changes in line with the new weights.
   The policy file appears in the PR diff; the report rendering does not.

## 4. Rubric / review checklist

| Criterion | How to check | Points |
|-----------|--------------|--------|
| A `Provider` protocol exists, with adapters for ≥2 vendors | read `src/providers/` | 2 |
| Each adapter normalises `usage` into (`input_tokens`, `output_tokens`, `cached_input_tokens`) | read adapter code | 2 |
| Agent code does not import vendor SDKs directly | `grep -r "from anthropic" src/agent.py` returns empty | 2 |
| Golden set runs produce one JSON file per provider; fields identical across providers | validation step 1 | 2 |
| Pricing lives in committed TOML with `source_url` and `fetched_on` per entry | `cat evals/pricing.toml` | 2 |
| Report renders a pricing-provenance table referencing the TOML | read a report | 1 |
| Report renders a headline (pass, mean score, p50, p95, cost/100, high-risk fails) with one column per provider | read a report | 2 |
| Report renders per-`intent`, per-`difficulty`, per-`risk_tier` tables | read a report | 2 |
| Selection policy is a committed TOML, not inline constants | `cat evals/selection_policy.toml` | 2 |
| Policy weights sum to 1.0 and are validated at load time | read Policy.\_\_post_init\_\_ | 1 |
| Any `risk_tier=high` failure vetoes a provider | validation step 3 | 2 |
| Verdict is one of `RECOMMEND` / `NEEDS_HUMAN_REVIEW` / `NO_VIABLE_PROVIDER` with exit codes 0/1/2 | validation steps 2–4 | 2 |
| Zero-price / `lookup-pending` pricing entries produce a prominent ⚠️ warning in the rendered report | validation step 5 | 2 |
| CI matrix is `fail-fast: false` and the report job is `needs:` the matrix | read workflow | 1 |
| Sticky PR comment uses a `header:` so the bot replaces its own comment | read workflow | 1 |
| Report rendering is deterministic given fixed inputs | validation step 6 | 1 |
| **Total** | | **27** |

Pass threshold: 20/27.

## 5. Common mistakes

1. **Branching the agent on provider name.** `if provider == "anthropic": ...`
   grows one `elif` per vendor forever. The `Provider` protocol adds twenty
   lines once and keeps the agent provider-agnostic. Every test that worked
   under mod-106 still works.
2. **Running the live eval on every PR.** Three providers × 30 rows × every
   push is expensive and noisy. Scope the `pull_request` trigger to paths
   that actually change the comparison; let `workflow_dispatch` and a weekly
   `schedule:` cron carry the rest.
3. **Pricing numbers typed into Python.** A `PRICE_PER_1M_IN = 3.0` constant
   silently ages. Keep pricing in a committed TOML with `source_url` and
   `fetched_on`, stamp the provenance into the report, and require a PR to
   change it. The report must warn loudly when any price is zero or missing.
4. **Trusting a benchmark leaderboard screenshot.** A leaderboard scored on
   someone else's prompts, with someone else's judge, against someone else's
   rows is not evidence about *your* app. The only benchmark that generalises
   to your production is your golden set. The report writes it down.
5. **Comparing providers at different temperatures.** Any `temperature > 0`
   smears judge variance across the comparison. Run every provider at
   `temperature=0` for the eval; evaluate warmth separately if the product
   needs it.
6. **Mean latency instead of p95.** The mean hides the tail. mod-107
   exercise-05 is specific: report percentiles. The headline table carries
   p50 *and* p95 so a reader can see both the typical experience and the
   worst-tenth-of-users experience.
7. **Collapsing cost, quality, and latency into a single hand-chosen number.**
   The composite score in the policy is not a scientific claim; it is a
   convenience. The *reasons* array is what the reviewer reads: "VETO openai —
   failed risk_tier=high row account-002" is defensible; "composite 0.73"
   alone is not.
8. **Letting one vendor's outage break the report.** `fail-fast: false` on
   the matrix is the difference between "we got a comparison on the two
   providers that were up" and "the report never built because Google had a
   500 for ten minutes." The first is a useful artifact; the second is an
   empty PR comment at 2am.
9. **No "no decision" path.** If the composite margin is 0.01, the right
   answer is "a human picks this, on criteria the policy does not encode" —
   not a false-precision `RECOMMEND`. `NEEDS_HUMAN_REVIEW` is a first-class
   verdict for a reason.
10. **Treating prompt caching as a provider feature, not a workload one.**
    All three vendors offer prompt-caching mechanisms with different shapes;
    whether caching is a win depends on the ratio of system-prompt tokens to
    per-row tokens in *your* golden set. Measure cached-input tokens in the
    `usage` block and let the cost table account for them. Do not assume a
    cache rate.

## 6. References

### Primary standards

- **NIST AI Risk Management Framework (AI RMF 1.0)** —
  <https://www.nist.gov/itl/ai-risk-management-framework>.
  MEASURE 2 ("AI systems are evaluated for trustworthy characteristics") and
  MANAGE 1.3 ("Responses to the AI risks deemed high priority…are developed,
  planned, and documented") motivate the sliced report and the policy-gated
  recommendation. The veto on `risk_tier=high` is a direct expression of
  MANAGE 1.3's "high priority" asymmetry.
- **OWASP LLM Top 10 for 2025** —
  <https://owasp.org/www-project-top-10-for-large-language-model-applications/>.
  LLM02 (Sensitive Information Disclosure) and LLM07 (System Prompt Leakage)
  are the categories that `prompt_injection` rows exercise; a provider that
  regresses those rows is vetoed, not just marked down.
- **OWASP Machine Learning Security Top 10** —
  <https://owasp.org/www-project-machine-learning-security-top-10/>.
  Supports the principle that a model swap requires a sliced evaluation
  rather than aggregate accuracy — ML07 (Transfer Learning Attack) and ML09
  (Output Integrity Attack) both motivate the per-risk-tier gate.

### Primary provider documentation

- **Anthropic Messages API** — <https://docs.claude.com/en/api/messages>.
  Canonical reference for the request/response shape in
  `anthropic_provider.py`, including the content-block types and
  `usage.input_tokens` / `usage.output_tokens`.
- **Anthropic Prompt Caching** —
  <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>.
  Source for `cache_read_input_tokens` on the Anthropic `usage` block.
- **Anthropic Pricing** — <https://www.anthropic.com/pricing>. The
  `pricing.toml` source URL for Anthropic entries.
- **OpenAI Chat Completions** —
  <https://platform.openai.com/docs/api-reference/chat>. Canonical reference
  for the OpenAI adapter, including `prompt_tokens_details.cached_tokens`.
- **OpenAI Pricing** — <https://openai.com/api/pricing/>. The
  `pricing.toml` source URL for OpenAI entries.
- **Google Gemini API** — <https://ai.google.dev/gemini-api/docs>.
  Canonical reference for the Google adapter, including `usage_metadata`.
- **Google Gemini Pricing** — <https://ai.google.dev/pricing>. The
  `pricing.toml` source URL for Google entries.

### CI mechanism references

- **GitHub Actions matrix strategy** —
  <https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs>.
  Reference for the `fail-fast: false` idiom that keeps one vendor outage
  from zeroing the comparison.
- **`actions/upload-artifact` / `actions/download-artifact`** —
  <https://github.com/actions/upload-artifact>,
  <https://github.com/actions/download-artifact>. Mechanism for
  freezing per-provider eval results between the eval job and the report
  job, and across workflow runs.
- **`marocchino/sticky-pull-request-comment`** —
  <https://github.com/marocchino/sticky-pull-request-comment>. The sticky
  PR-comment mechanism keyed by `header:` so the bot replaces its own
  comment instead of appending.

### Local exercise context

This project consumes the agent and `AGENT_FAKE=1` stub from
[`mod-106`](../../modules/mod-106-llm-app-deployment/), the latency
measurement discipline from
[`mod-107 exercise-05`](../../modules/mod-107-streaming-async-and-orchestration/exercise-05-measure-time-to-first-token-and-tail-latency/SOLUTION.md),
and the pytest harness, scoring, and regression-report structure from
[`mod-109`](../../modules/mod-109-minimal-eval-and-regression-checks/).
Where mod-109 exercise-04's report is a two-way diff on a single provider,
this project's report is an N-way comparison across providers with cost as a
first-class dimension and a recommendation as the output.
