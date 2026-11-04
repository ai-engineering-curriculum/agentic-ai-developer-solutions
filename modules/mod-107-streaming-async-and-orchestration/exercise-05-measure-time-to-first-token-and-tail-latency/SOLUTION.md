# mod-107-streaming-async-and-orchestration/exercise-05-measure-time-to-first-token-and-tail-latency — Solution

## Solution overview

The deliverable is a load-testing harness — `measure_stream.py` — that fans
out N streaming requests, records per-request timings in `samples.csv`, and
prints the one-page percentile report chapter 5 prescribes. Everything about
this exercise is in service of making the number honest: percentiles only,
`perf_counter` only, SDK retries off, and the first few requests discarded
to keep cold-starts out of the steady-state p95.

Design choices:

- **Reuse `fanout` from exercise 03.** The semaphore, timeout, and alignment
  properties are already proven; here the harness only supplies the
  per-item `stream_and_measure` coroutine and a prompt list.
- **One client per run, `max_retries=0`.** Chapter 5 is explicit: measuring
  the model means not measuring the retry policy. Hidden SDK retries
  smear latency across a small share of requests and pollute the tail.
- **CSV is the raw data, not a formatted table.** Future modules re-run
  this harness against newer models and compare. Writing a format we can
  diff / `pandas.read_csv(...)` across runs is what makes that possible.
- **`tokens_per_s = output_tokens / (t_last_chunk - t_first_chunk)`,** not
  `output_tokens / ttc`. Dividing by TTC charges throughput for the pre-first-
  token setup (connect, upload, server-side scheduling) and under-reports
  the real generation rate. Chapter 5 flags this specifically.

The reference uses Anthropic through `AsyncAnthropic`. The OpenAI adapter
(sketched inline) is one function; the rest of the harness is identical.

## Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "measure-stream"
version = "0.1.0"
description = "Harness for TTFT / TTC / tokens-per-sec percentile reporting on streaming LLM calls."
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.39,<1.0",
    "httpx>=0.27,<1.0",
    "numpy>=1.26,<3.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

### `src/measure.py`

```python
# src/measure.py
"""Per-request streaming timing instrument.

Every timestamp is perf_counter(); wall-clock time is NOT used so an NTP
adjustment mid-stream cannot produce a negative latency. Output tokens come
from the SDK's final usage (Anthropic: final.usage.output_tokens; OpenAI:
chunk.usage.completion_tokens on a stream_options={"include_usage": True}
stream).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic


@dataclass
class Sample:
    idx: int
    prompt: str
    t0: float
    t_first_chunk: float | None = None
    t_last_chunk: float | None = None
    output_tokens: int | None = None
    outcome: str = "ok"
    failure_reason: str = ""

    @property
    def ttft_s(self) -> float | None:
        if self.t_first_chunk is None:
            return None
        return self.t_first_chunk - self.t0

    @property
    def ttc_s(self) -> float | None:
        if self.t_last_chunk is None:
            return None
        return self.t_last_chunk - self.t0

    @property
    def tokens_per_s(self) -> float | None:
        if self.output_tokens is None or self.t_first_chunk is None or self.t_last_chunk is None:
            return None
        # Denominator is the GENERATION phase only, not the whole wall-clock.
        # chapter 5 calls this out: dividing by TTC under-reports throughput.
        gen = max(self.t_last_chunk - self.t_first_chunk, 1e-6)
        return self.output_tokens / gen


async def stream_and_measure_anthropic(
    client: AsyncAnthropic,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    idx: int,
) -> Sample:
    sample = Sample(idx=idx, prompt=prompt, t0=time.perf_counter())
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _text in stream.text_stream:
                if sample.t_first_chunk is None:
                    sample.t_first_chunk = time.perf_counter()
            final = await stream.get_final_message()
        sample.t_last_chunk = time.perf_counter()
        sample.output_tokens = final.usage.output_tokens
    except Exception as exc:  # noqa: BLE001
        sample.t_last_chunk = time.perf_counter()
        sample.outcome = _classify_failure(exc)
        sample.failure_reason = f"{type(exc).__name__}: {exc}"
    return sample


def _classify_failure(exc: Exception) -> str:
    # Compact failure buckets for the report's composition line.
    name = type(exc).__name__
    if "RateLimit" in name or "429" in str(exc):
        return "429"
    if "Status" in name or "HTTPStatusError" in name:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            return str(status)
    if "Timeout" in name:
        return "timeout"
    return "error"
```

### `src/report.py`

```python
# src/report.py
"""Percentile aggregation + the one-page report.

Uses numpy.percentile. For N under a few hundred, statistics.quantiles is
also fine; numpy is faster when we grow to multi-thousand-request harnesses.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .measure import Sample


def _p(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarise(samples: list[Sample], *, model: str, concurrency: int, warmup: int) -> str:
    # Warm-up guard: drop the first `warmup` samples before aggregation.
    if warmup > 0:
        effective = samples[warmup:]
    else:
        effective = samples
    successes = [s for s in effective if s.outcome == "ok"]
    failures = [s for s in effective if s.outcome != "ok"]

    ttft_ms = [s.ttft_s * 1000 for s in successes if s.ttft_s is not None]
    ttc_s = [s.ttc_s for s in successes if s.ttc_s is not None]
    tps = [s.tokens_per_s for s in successes if s.tokens_per_s is not None]

    breakdown = Counter(s.outcome for s in failures)
    failures_text = ", ".join(f"{count}× {code}" for code, count in sorted(breakdown.items()))
    if not failures_text:
        failures_text = "0 terminal"

    lines: list[str] = []
    lines.append(f"Warm-up: discarded first {warmup} requests from aggregation")
    lines.append(f"Sample: {len(effective)} requests, model {model}, concurrency {concurrency}")
    if ttft_ms:
        lines.append(f"TTFT   p50: {_p(ttft_ms, 50):.0f} ms   p95: {_p(ttft_ms, 95):.0f} ms   p99: {_p(ttft_ms, 99):.0f} ms")
    if ttc_s:
        lines.append(f"TTC    p50: {_p(ttc_s, 50):.1f} s    p95: {_p(ttc_s, 95):.1f} s    p99: {_p(ttc_s, 99):.1f} s")
    if tps:
        lines.append(f"Tokens per s   p50: {_p(tps, 50):.0f}   p95: {_p(tps, 95):.0f}   p99: {_p(tps, 99):.0f}   (output-only)")
    rate = (len(failures) / len(effective) * 100.0) if effective else 0.0
    lines.append(f"Failures: {len(failures)} / {len(effective)} ({rate:.1f}%)  — {failures_text}")
    return "\n".join(lines)
```

### `src/prompts.py`

```python
# src/prompts.py
SHORT_PROMPTS = [
    "In one sentence, what does TTFT mean for a streaming LLM?",
    "In one sentence, define Server-Sent Events.",
    "In one sentence, what is an async semaphore for?",
    "In one sentence, why do we use jitter in retry backoff?",
    "In one sentence, what is a percentile?",
    # ... extend to 30 short prompts ...
] * 6  # 30 total

LONG_PROMPTS = [
    "Write three paragraphs on why streaming LLM responses change perceived latency.",
    "Write three paragraphs on the trade-offs between threads and asyncio for LLM I/O.",
    "Write three paragraphs on p50/p95/p99 and why the mean lies for LLM latency.",
    # ... extend to 20 long prompts ...
] * 7  # 20 total, truncated below
LONG_PROMPTS = LONG_PROMPTS[:20]
```

### `src/csv_writer.py`

```python
# src/csv_writer.py
import csv
from pathlib import Path

from .measure import Sample


FIELDS = [
    "idx", "outcome", "failure_reason",
    "t0", "t_first_chunk", "t_last_chunk",
    "ttft_s", "ttc_s", "output_tokens", "tokens_per_s",
    "prompt",
]


def write_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in samples:
            w.writerow({
                "idx": s.idx,
                "outcome": s.outcome,
                "failure_reason": s.failure_reason,
                "t0": f"{s.t0:.6f}",
                "t_first_chunk": "" if s.t_first_chunk is None else f"{s.t_first_chunk:.6f}",
                "t_last_chunk": "" if s.t_last_chunk is None else f"{s.t_last_chunk:.6f}",
                "ttft_s": "" if s.ttft_s is None else f"{s.ttft_s:.4f}",
                "ttc_s": "" if s.ttc_s is None else f"{s.ttc_s:.4f}",
                "output_tokens": "" if s.output_tokens is None else s.output_tokens,
                "tokens_per_s": "" if s.tokens_per_s is None else f"{s.tokens_per_s:.2f}",
                "prompt": s.prompt,
            })
```

### `src/measure_stream.py` — the entry point

```python
# src/measure_stream.py
"""The harness. Reusable in every subsequent module when someone asks
"how fast is it?"

Usage:
    python -m src.measure_stream --profile short --out samples_short.csv
    python -m src.measure_stream --profile long  --out samples_long.csv
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

# Reuse the fan-out from exercise 03. In this repo we import from the
# already-shipped package; if you are iterating standalone, vendor fanout.py
# alongside src/measure_stream.py.
from src.fanout import fanout  # noqa: E402
from src.measure import Sample, stream_and_measure_anthropic  # noqa: E402
from src.csv_writer import write_csv  # noqa: E402
from src.report import summarise  # noqa: E402
from src.prompts import LONG_PROMPTS, SHORT_PROMPTS  # noqa: E402

WARMUP = 2  # Discard the first N from aggregation — cold-start guard.


async def _run(args) -> int:
    prompts = SHORT_PROMPTS if args.profile == "short" else LONG_PROMPTS
    max_tokens = 256 if args.profile == "short" else 1800

    # IMPORTANT: max_retries=0 — chapter 5 is explicit that SDK retries
    # pollute the tail. Measure the model, not the retry policy.
    async with AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                              max_retries=0) as client:
        async def one(idx_prompt: tuple[int, str]) -> Sample:
            idx, prompt = idx_prompt
            return await stream_and_measure_anthropic(
                client, args.model, prompt, max_tokens=max_tokens, idx=idx,
            )

        indexed = list(enumerate(prompts))
        results = await fanout(
            indexed, one,
            concurrency=args.concurrency,
            per_item_timeout=120.0,
        )

    # fanout returns a list aligned with `indexed`; any exceptions are per-item.
    samples: list[Sample] = []
    for i, r in enumerate(results):
        if isinstance(r, Sample):
            samples.append(r)
        else:
            # A Sample was never constructed — the coroutine raised before
            # `stream_and_measure_anthropic` could record. Create one as a
            # failure row so the CSV has a line for every attempted request.
            samples.append(Sample(
                idx=i, prompt=prompts[i], t0=0.0,
                outcome="error",
                failure_reason=f"{type(r).__name__}: {r}",
            ))

    out_csv = Path(args.out)
    write_csv(out_csv, samples)

    report = summarise(samples,
                       model=args.model,
                       concurrency=args.concurrency,
                       warmup=WARMUP)
    print(report)
    print(f"\nRaw samples: {out_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["short", "long"], required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--out", default="samples.csv")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
```

### Sample `LATENCY_REPORT.md` — the artefact your future self reads

```markdown
# Streaming latency — mod-107 exercise 05

Date run: 2026-11-03
Provider: Anthropic
Model: claude-opus-4-7
Concurrency: 6

## Short profile (30 prompts, ~200 output tokens)

```
Warm-up: discarded first 2 requests from aggregation
Sample: 28 requests, model claude-opus-4-7, concurrency 6
TTFT   p50: 320 ms   p95: 540 ms   p99: 900 ms
TTC    p50: 2.1 s    p95: 3.4 s    p99: 4.2 s
Tokens per s   p50: 95   p95: 70   p99: 55   (output-only)
Failures: 0 / 28 (0.0%)  — 0 terminal
```

## Long profile (20 prompts, ~1500 output tokens)

```
Warm-up: discarded first 2 requests from aggregation
Sample: 18 requests, model claude-opus-4-7, concurrency 6
TTFT   p50: 340 ms   p95: 610 ms   p99: 1250 ms
TTC    p50: 12.4 s   p95: 16.1 s   p99: 19.8 s
Tokens per s   p50: 125  p95: 92   p99: 78   (output-only)
Failures: 0 / 18 (0.0%)  — 0 terminal
```

## Answers to the three questions

1. **TTFT p50 / p95 / p99 for ~200-token response:** 320 ms / 540 ms / 900 ms.
2. **TTC p50 / p95 / p99 for ~1500-token response:** 12.4 s / 16.1 s / 19.8 s.
3. **Mean / p99 ratio for long TTC:** mean ≈ 13.1 s, p99 ≈ 19.8 s, ratio ≈
   0.66. Ratio is not dramatic but the tail does carry meaningful weight — a
   user in the 99th percentile waits ~50% longer than the median user.

## What surprised me

The TTFT distribution is tight across both profiles — within 50 ms at p50
across short and long — which confirms chapter 5's observation that
first-token latency is a property of the request setup, not the generation
length. The variance lives in TTC, which rises almost linearly with output
tokens.
```

> Numbers above are placeholders for the report's shape. When you run the
> harness yourself, replace with the numbers you measured; the grading check
> is that your report looks like this, not that your numbers match mine.

## Validation steps

```bash
pip install -e ".[test]"

# 1) Short profile — ~1 minute wall-clock.
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.measure_stream --profile short --out samples_short.csv
# Prints the one-page report; writes samples_short.csv.

# 2) Long profile — ~3-4 minutes wall-clock.
python -m src.measure_stream --profile long --out samples_long.csv

# 3) Sanity: long TTC p99 at least 3x short TTC p99.
python - <<'PY'
import pandas as pd
short = pd.read_csv("samples_short.csv")
long  = pd.read_csv("samples_long.csv")
import numpy as np
def p99(df, col): return np.nanpercentile(df[col].astype(float), 99)
assert p99(long, "ttc_s") >= 3 * p99(short, "ttc_s"), "long profile should dominate TTC tail"
print("ok")
PY

# 4) Stability: re-run short profile. p50 TTC between the two runs is within
#    ~20%. If it is wildly different, extend WARMUP or raise N.

# 5) Grep: SDK retries off, perf_counter everywhere.
grep -n "max_retries" src/measure_stream.py          # expect max_retries=0
grep -rn "time.time\|time.monotonic" src/            # expect no matches
```

The report's shape is the evaluation surface: **no mean-latency line**;
percentiles only; sample size, model, concurrency named explicitly; failure
composition broken down.

## Rubric or review checklist

| Criterion | Pass if |
|-----------|---------|
| `time.perf_counter()` everywhere | No `time.time()` or `time.monotonic()` in the measurement path |
| Correct denominator for throughput | `output_tokens / (t_last_chunk - t_first_chunk)`, not `/ ttc` |
| SDK retries disabled | Client built with `max_retries=0` |
| Warm-up applied | First 2 samples are discarded from aggregation and named in the report |
| One-page report exactly shaped | Lines for Sample / TTFT / TTC / Tokens per s / Failures — no mean |
| `samples.csv` is complete | Every attempted request has a row; failed rows have outcome + failure_reason |
| Short vs long are distinguishable | `long` profile TTC p99 ≥ 3× `short` profile TTC p99 |
| Stability across back-to-back runs | p50 within ~20% between two consecutive runs of the same profile |
| Report answers questions with numbers | Each of the three questions in `LATENCY_REPORT.md` has a numeric answer, not an adjective |
| Rate-limit retries not polluting the tail | Either observed failure rate is 0% or 429s are visible in the composition line (not silently absorbed by retries) |

## Common mistakes

1. **Reporting mean latency.** The mean is dominated by the tail and tells
   you nothing about what a typical user sees. The one-page report has no
   mean line. Percentiles only.
2. **Not discarding warm-up samples.** The first one or two requests include
   cold-start costs (connection pool construction, TLS handshake, provider-
   side scheduling). Averaging them into the steady-state p95 falsely
   inflates it.
3. **`output_tokens / ttc_s` for throughput.** Charges throughput for pre-
   generation setup. The honest denominator is the generation phase only:
   `t_last_chunk - t_first_chunk`.
4. **Leaving `max_retries` at the SDK default.** One retried 429 adds a
   couple of seconds to a specific request and smears your tail. Turn SDK
   retries off; measure the model.
5. **Sample sizes too small.** p99 on 20 samples is "the worst of 20"; it
   moves wildly between runs. The short/long sample sizes here (30/20) give
   barely-stable p50; honest p99 needs >100 samples. State the sample size
   and do not over-interpret.
6. **Mixing client-side and server-side measurements silently.** If your
   harness runs in the same process as the endpoint under test, you are
   measuring server-side latency. Browser latency (what the PM asks about)
   includes network to the server. Keep the two separate when you report.
7. **`time.time()` on a long-lived stream.** An NTP adjustment mid-stream
   can produce a negative TTFT. `perf_counter` is monotonic; always use it.
8. **Including retry sleeps in the measurement.** If your harness wraps
   requests in exercise 04's retry helper, your "latency" number includes
   `asyncio.sleep` time. For a model-latency measurement, turn retries off;
   for a feature-latency measurement, include retries but flag the retry-
   attributable share separately.
9. **Writing only aggregated numbers to the CSV.** The CSV is the raw
   evidence; later analyses (histograms, bucketing by model, A/B against a
   new provider) need one row per request with every timestamp intact.
10. **Running with many concurrent streams to "get more samples faster".**
    High concurrency tickles rate limits; your tail now reflects your
    retry policy, not the model. Chapter 5 recommends concurrency 4–8.

## References

- Chapter 5 of this module — `../../../agentic-ai-developer-learning/lessons/mod-107-streaming-async-and-orchestration/05-measuring-streaming-latency.md`
- Prometheus histograms and quantile queries: <https://prometheus.io/docs/practices/histograms/>
- OpenTelemetry metrics data model: <https://opentelemetry.io/docs/specs/otel/metrics/data-model/>
- HdrHistogram — low-overhead, high-fidelity latency histograms: <https://www.hdrhistogram.org/>
- Gil Tene — "How NOT to Measure Latency": <https://www.infoq.com/presentations/latency-response-time/>
- `numpy.percentile`: <https://numpy.org/doc/stable/reference/generated/numpy.percentile.html>
- Anthropic streaming reference (final usage): <https://docs.anthropic.com/en/api/messages-streaming>
- OpenAI Chat Completions streaming (`stream_options={"include_usage": True}`): <https://platform.openai.com/docs/api-reference/chat/streaming>
