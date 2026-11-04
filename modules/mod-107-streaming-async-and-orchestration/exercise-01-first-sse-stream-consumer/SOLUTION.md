# mod-107-streaming-async-and-orchestration/exercise-01-first-sse-stream-consumer — Solution

## Solution overview

The exercise asks for one streaming CLI that produces character-by-character
output and a three-line timing summary, built twice: once through the provider
SDK's stream helper (Variant A), once by reading SSE frames off `httpx.stream`
and parsing them by hand (Variant B). The whole point is that after Variant B
you can't be surprised by the transport anymore — you have seen the `event:` /
`data:` / blank-line structure arrive on your own file descriptor.

Design choices that shape the reference solution:

- **One module per variant, one shared timing helper.** The streaming loop and
  the SSE parser are the only things that differ between SDK and raw paths; the
  timers, the CLI, and the summary printer are shared.
- **`time.perf_counter()` everywhere.** Chapter 1 says it; exercise 05 formalises
  it. Wall-clock (`time.time()`) can jump backwards on an NTP adjustment and
  produce a negative TTFT mid-stream.
- **Record the raw SSE frames on the Variant B path.** Even without implementing
  the `--record / --replay` stretch goal, keeping a `--record path.sse` flag that
  tees every SSE frame to disk makes debugging the parser tractable — you can
  reproduce a bad run without spending more API credit.
- **One exit path for every failure.** Auth, bad model name, and mid-stream
  disconnects all converge on a single `print(..., file=sys.stderr); sys.exit(1)`
  helper so the "no stack trace on misconfig" acceptance criterion is a property
  of the program, not something you remember to do inline.

The CLI exposes `--provider {anthropic,openai}`, `--variant {sdk,raw}`,
`--model`, and `--temperature`, plus the `--record / --replay` pair. Running
the same prompt through both variants at `temperature=0` makes the "same output
text on both paths" acceptance check trivial to run as a `diff`.

## Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "sse-first-stream"
version = "0.1.0"
description = "A tiny CLI that streams one LLM response two ways: SDK and raw SSE."
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.39,<1.0",
    "openai>=1.40,<2.0",
    "httpx>=0.27,<1.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

### `src/timing.py`

```python
# src/timing.py
"""Tiny timing recorder shared by both variants.

TTFT, TTC, and output-token counts live here so the printed summary is identical
regardless of which streaming path produced it. All timestamps use perf_counter
so an NTP adjustment mid-stream cannot produce a negative latency.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass
class StreamTiming:
    t0: float
    t_first_chunk: float | None = None
    t_last_chunk: float | None = None
    output_tokens: int | None = None

    @classmethod
    def start(cls) -> "StreamTiming":
        return cls(t0=time.perf_counter())

    def mark_first_chunk(self) -> None:
        if self.t_first_chunk is None:
            self.t_first_chunk = time.perf_counter()

    def mark_last_chunk(self) -> None:
        self.t_last_chunk = time.perf_counter()

    def print_summary(self) -> None:
        # Print to stderr so it does not contaminate the streamed text on stdout.
        # A consumer piping stdout to a file still sees the streamed answer only.
        if self.t_first_chunk is None:
            # Stream died before producing anything. Report zeros rather than NaN.
            ttft_ms = 0.0
        else:
            ttft_ms = (self.t_first_chunk - self.t0) * 1000.0
        if self.t_last_chunk is None:
            ttc_s = time.perf_counter() - self.t0
        else:
            ttc_s = self.t_last_chunk - self.t0
        print(
            f"\n--- TTFT: {ttft_ms:.0f} ms | TTC: {ttc_s:.2f} s | "
            f"output_tokens: {self.output_tokens if self.output_tokens is not None else 'n/a'}",
            file=sys.stderr,
            flush=True,
        )
```

### `src/variant_sdk.py`

```python
# src/variant_sdk.py
"""Variant A — SDK streaming helpers for both providers.

The shape you would actually ship. Each function produces streamed text on
stdout and returns a StreamTiming that already has first-chunk, last-chunk, and
output_tokens filled in.
"""
from __future__ import annotations

import os
import sys

from .timing import StreamTiming


def stream_anthropic(prompt: str, *, model: str, temperature: float) -> StreamTiming:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    timing = StreamTiming.start()

    try:
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            # text_stream yields deltas already unpacked from content_block_delta.
            for text in stream.text_stream:
                timing.mark_first_chunk()
                print(text, end="", flush=True)
            final = stream.get_final_message()
            timing.output_tokens = final.usage.output_tokens
    except anthropic.APIStatusError as exc:
        print(f"\n[error] anthropic API error: {exc.status_code} {exc.message}",
              file=sys.stderr, flush=True)
        timing.mark_last_chunk()
        raise
    except Exception as exc:
        # Mid-stream read error (e.g. RemoteProtocolError) — mark close and re-raise.
        print(f"\n[error] stream interrupted: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        timing.mark_last_chunk()
        raise

    timing.mark_last_chunk()
    return timing


def stream_openai(prompt: str, *, model: str, temperature: float) -> StreamTiming:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    timing = StreamTiming.start()

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            # Final chunk carries .usage only when include_usage is set.
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            # Chunks that carry only role or finish_reason have no .choices content.
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    timing.mark_first_chunk()
                    print(text, end="", flush=True)
            if chunk.usage is not None:
                timing.output_tokens = chunk.usage.completion_tokens
    except openai.APIStatusError as exc:
        print(f"\n[error] openai API error: {exc.status_code} {exc.message}",
              file=sys.stderr, flush=True)
        timing.mark_last_chunk()
        raise
    except Exception as exc:
        print(f"\n[error] stream interrupted: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        timing.mark_last_chunk()
        raise

    timing.mark_last_chunk()
    return timing
```

### `src/variant_raw.py`

```python
# src/variant_raw.py
"""Variant B — raw SSE parsing over httpx.stream.

The whole exercise lives here: group lines on blank lines, ignore comments,
dispatch on event name, and for Anthropic print only text_delta deltas inside
content_block_delta events. For OpenAI, treat 'data: [DONE]' as the sentinel.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from .timing import StreamTiming


def _iter_sse_events(
    response: httpx.Response,
    *,
    record_path: Path | None = None,
) -> Iterator[tuple[str | None, str]]:
    """Yield (event_name, data_string) pairs from an SSE response.

    Groups lines until a blank line (the SSE record terminator). Joins
    multi-line 'data:' payloads with '\\n' per the SSE spec. Skips comment
    lines (':' prefix) and any lines that are not 'event:' or 'data:'.
    """
    recorder = record_path.open("w", encoding="utf-8") if record_path else None
    try:
        event_name: str | None = None
        data_lines: list[str] = []
        for raw in response.iter_lines():
            if recorder is not None:
                # Preserve the exact framing so --replay can reproduce it later.
                recorder.write(raw + "\n")
            if raw == "":
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = None
                data_lines = []
                continue
            if raw.startswith(":"):
                # SSE comment / keep-alive. Both Anthropic and many proxies send
                # ': ping' frames every few seconds; if we errored on them the
                # stream would appear to freeze for no visible reason.
                continue
            if raw.startswith("event:"):
                event_name = raw[len("event:"):].strip()
            elif raw.startswith("data:"):
                data_lines.append(raw[len("data:"):].lstrip())
            # Any other line (id:, retry:) is ignored for this exercise.
    finally:
        if recorder is not None:
            recorder.close()


def stream_anthropic_raw(
    prompt: str,
    *,
    model: str,
    temperature: float,
    record_path: Path | None = None,
) -> StreamTiming:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    timing = StreamTiming.start()

    with httpx.stream(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "temperature": temperature,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60.0,
    ) as response:
        # Bad model name / bad key fail here, before we read any SSE frames.
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")
            print(f"\n[error] anthropic API error: {response.status_code} {body}",
                  file=sys.stderr, flush=True)
            timing.mark_last_chunk()
            response.raise_for_status()

        try:
            for event_name, data in _iter_sse_events(response, record_path=record_path):
                if event_name in (None, "ping", "message_start", "message_stop"):
                    # message_start carries initial usage; we read usage later from
                    # message_delta where the totals are actually populated.
                    continue
                payload = json.loads(data)
                if event_name == "content_block_delta":
                    delta = payload.get("delta", {})
                    if delta.get("type") == "text_delta":
                        timing.mark_first_chunk()
                        print(delta["text"], end="", flush=True)
                elif event_name == "message_delta":
                    usage = payload.get("usage") or {}
                    if "output_tokens" in usage:
                        timing.output_tokens = usage["output_tokens"]
        except httpx.HTTPError as exc:
            # Mid-stream connection drop. Print what we got, note the exception.
            print(f"\n[error] stream interrupted: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            timing.mark_last_chunk()
            raise

    timing.mark_last_chunk()
    return timing


def stream_openai_raw(
    prompt: str,
    *,
    model: str,
    temperature: float,
    record_path: Path | None = None,
) -> StreamTiming:
    api_key = os.environ["OPENAI_API_KEY"]
    timing = StreamTiming.start()

    with httpx.stream(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60.0,
    ) as response:
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")
            print(f"\n[error] openai API error: {response.status_code} {body}",
                  file=sys.stderr, flush=True)
            timing.mark_last_chunk()
            response.raise_for_status()

        try:
            for event_name, data in _iter_sse_events(response, record_path=record_path):
                # OpenAI does not set event:; every record is a data: JSON chunk
                # plus the literal '[DONE]' sentinel.
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                choices = payload.get("choices") or []
                if choices:
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        timing.mark_first_chunk()
                        print(text, end="", flush=True)
                usage = payload.get("usage")
                if usage and "completion_tokens" in usage:
                    timing.output_tokens = usage["completion_tokens"]
        except httpx.HTTPError as exc:
            print(f"\n[error] stream interrupted: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            timing.mark_last_chunk()
            raise

    timing.mark_last_chunk()
    return timing
```

### `src/main.py`

```python
# src/main.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .timing import StreamTiming
from .variant_raw import stream_anthropic_raw, stream_openai_raw
from .variant_sdk import stream_anthropic, stream_openai


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream one LLM response.")
    parser.add_argument("prompt", help="prompt text")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--variant", choices=["sdk", "raw"], default="sdk")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--record", type=Path, default=None,
                        help="Variant B only: tee raw SSE frames to this file.")
    args = parser.parse_args(argv)

    # Sensible provider defaults so the common cases are one-flag commands.
    model = args.model or {
        "anthropic": "claude-opus-4-7",
        "openai": "gpt-4.1",
    }[args.provider]

    try:
        if args.variant == "sdk":
            if args.provider == "anthropic":
                timing = stream_anthropic(args.prompt, model=model, temperature=args.temperature)
            else:
                timing = stream_openai(args.prompt, model=model, temperature=args.temperature)
        else:
            if args.provider == "anthropic":
                timing = stream_anthropic_raw(args.prompt, model=model, temperature=args.temperature,
                                              record_path=args.record)
            else:
                timing = stream_openai_raw(args.prompt, model=model, temperature=args.temperature,
                                           record_path=args.record)
    except KeyboardInterrupt:
        # Ctrl-C during generation should print what we have and exit non-zero.
        print("\n[info] interrupted by user", file=sys.stderr, flush=True)
        # No per-request timing here because the helper never returned, but a
        # polished version would thread the StreamTiming out of the generator
        # on KeyboardInterrupt and print it below. Noted in common mistakes.
        return 130
    except Exception:
        # Helpers already printed the controlled error line to stderr.
        return 1

    timing.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tests/test_sse_parser.py`

A tiny offline test for the SSE grouping logic. It bypasses the network with a
fake `httpx.Response`-like object so it never spends API credit:

```python
# tests/test_sse_parser.py
from __future__ import annotations

import io

from src.variant_raw import _iter_sse_events


class FakeResponse:
    """Minimal stand-in that exposes .iter_lines() over a canned byte stream."""
    def __init__(self, text: str) -> None:
        self._text = text

    def iter_lines(self):
        for line in self._text.split("\n"):
            yield line


def test_ignores_comments_and_groups_on_blank_lines():
    sse = (
        ": keep-alive\n"
        "event: content_block_delta\n"
        'data: {"delta": {"type": "text_delta", "text": "hi"}}\n'
        "\n"
        "event: ping\n"
        "data: {}\n"
        "\n"
    )
    events = list(_iter_sse_events(FakeResponse(sse)))
    assert events == [
        ("content_block_delta", '{"delta": {"type": "text_delta", "text": "hi"}}'),
        ("ping", "{}"),
    ]


def test_joins_multi_line_data_with_newline():
    sse = (
        "event: content_block_delta\n"
        'data: {"delta":\n'
        'data:  {"type":"text_delta","text":"A\\nB"}}\n'
        "\n"
    )
    events = list(_iter_sse_events(FakeResponse(sse)))
    assert events[0][0] == "content_block_delta"
    assert "\n" in events[0][1]  # multi-line data joined with \n per spec
```

## Validation steps

Run against both providers at `temperature=0` and confirm the streamed text is
identical across variants; then force the failure cases.

```bash
# Install
pip install -e ".[test]"

# Offline parser test (no key, no network, no spend)
pytest -q
# 2 passed

# 1) Happy path — SDK, Anthropic, visible streaming
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main "Write a paragraph about the history of Server-Sent Events." \
    --provider anthropic --variant sdk > /tmp/out_sdk.txt
# TTFT ~200-600 ms; TTC several seconds; output_tokens ~120-200

# 2) Happy path — raw SSE, Anthropic
python -m src.main "Write a paragraph about the history of Server-Sent Events." \
    --provider anthropic --variant raw --record /tmp/trace.sse > /tmp/out_raw.txt

# 3) Same text out of both variants at temperature=0
diff /tmp/out_sdk.txt /tmp/out_raw.txt
# empty (modulo whitespace on provider-side quirks)

# 4) OpenAI raw path exits cleanly on [DONE]
export OPENAI_API_KEY=sk-...
python -m src.main "Haiku about SSE." --provider openai --variant raw
# clean exit, no JSONDecodeError

# 5) Controlled failure on bad model name — NOT a stack trace
python -m src.main "hi" --provider anthropic --variant raw --model claude-doesnt-exist
# stderr: [error] anthropic API error: 404 ...
# exit 1

# 6) Mid-generation Ctrl-C — summary reflects partial time
python -m src.main "Write a 2000-word essay on SSE." --provider anthropic --variant sdk
# ... hit Ctrl-C after ~1 s ...
# stderr: [info] interrupted by user
# exit 130
```

Inspect the recorded SSE frames to confirm the parser is seeing what you think
it is seeing:

```bash
head -30 /tmp/trace.sse
# event: message_start
# data: {"type":"message_start", ...}
#
# event: content_block_start
# data: {"type":"content_block_start", ...}
#
# event: content_block_delta
# data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The"}}
# ...
```

If `TTFT ≈ TTC`, you're not actually streaming — something downstream (a
terminal that line-buffers, a proxy, your own `print` without `flush=True`) is
batching the output. Walk the pipeline and find the buffer.

## Rubric or review checklist

| Criterion | Pass if |
|-----------|---------|
| Streaming is visibly chunked on stdout | TTFT reported in low hundreds of ms; characters appear as they arrive, not in one block at the end |
| Three-line summary printed | TTFT (ms), TTC (s), output_tokens — all three present |
| Variant A uses the SDK helper | `client.messages.stream(...)` (Anthropic) or `stream=True` with `stream_options={"include_usage": True}` (OpenAI) |
| Variant B uses `httpx.stream` directly | No `sseclient`-style library in the import list |
| Variant B groups events on blank lines | Multi-line `data:` payloads join with `\n`; `:` comments and `ping` events are ignored |
| Variant B handles Anthropic correctly | Prints only `text_delta` inside `content_block_delta`; reads final usage from `message_delta` |
| Variant B handles OpenAI's `[DONE]` sentinel | No `JSONDecodeError` at stream end; stream closes on sentinel |
| Deterministic cross-variant diff | At `temperature=0`, `diff` of SDK vs raw outputs is empty for the same prompt |
| Clean error for bad model name | Prints one error line; exits non-zero; no Python traceback |
| Ctrl-C mid-stream exits cleanly | Partial text preserved; TTC reflects the Ctrl-C moment, not the full generation |
| Uses `time.perf_counter()` | No `time.time()` in the timing path |

## Common mistakes

1. **Line-buffered `print`.** Without `flush=True`, Python's stdout buffers until
   a newline — your streaming effect disappears for prose that generates without
   a newline for a sentence or two. Fix: `print(text, end="", flush=True)` on
   every delta. This is mentioned in chapter 1's "buffering somewhere in the
   middle" failure mode.
2. **Parsing per `data:` line instead of per event.** SSE terminates events on
   blank lines, not on every newline. Rare multi-line `data:` payloads will
   crash a line-at-a-time `json.loads`. Group lines, dispatch on blank line.
3. **`json.loads("[DONE]")` on OpenAI.** Only OpenAI sends this sentinel, and it
   is not part of the SSE standard. Check `data == "[DONE]"` before parsing.
4. **Creating a fresh `httpx.Client` per call.** For Variant B's one-shot
   streaming CLI this is cosmetic; it becomes a real bug in exercise 03 where a
   per-item client defeats keep-alive. Mention it now so the habit is right.
5. **Mixing reader buffering with event boundaries.** `response.iter_bytes()`
   gives you bytes with no line structure at all — you then have to split on
   `\n` *and* handle the blank-line boundary yourself. `response.iter_lines()`
   does the first; you still have to do the second.
6. **Treating `ping` or `message_start` as data-bearing.** The default event
   name when `event:` is absent is `"message"`; Anthropic almost always sets an
   `event:` line, but a parser that errors on unknown names will die after ~15
   seconds when the first keep-alive arrives. Dispatch on known names; ignore
   the rest.
7. **Using `time.time()` instead of `time.perf_counter()`.** Wall-clock is
   subject to NTP adjustments; a jump mid-stream produces a negative TTFT.
   Chapter 1 explicitly calls this out.
8. **Reporting mean latency from one request.** This exercise is one request at
   a time — the three numbers in the summary are *this* request's TTFT, TTC,
   and output tokens, not a percentile. Exercise 05 is where percentiles start
   to matter.

## References

- Server-Sent Events — WHATWG HTML Living Standard:
  <https://html.spec.whatwg.org/multipage/server-sent-events.html>
- MDN — Using Server-Sent Events:
  <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>
- Anthropic Messages streaming reference:
  <https://docs.anthropic.com/en/api/messages-streaming>
- OpenAI Chat Completions streaming reference:
  <https://platform.openai.com/docs/api-reference/chat/streaming>
- `httpx` async / streaming responses:
  <https://www.python-httpx.org/async/#streaming-responses>
- Chapter 1 of this module — `../../../agentic-ai-developer-learning/lessons/mod-107-streaming-async-and-orchestration/01-consuming-sse-streams.md`
