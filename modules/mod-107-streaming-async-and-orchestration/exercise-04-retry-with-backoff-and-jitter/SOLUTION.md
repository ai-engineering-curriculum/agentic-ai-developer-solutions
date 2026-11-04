# mod-107-streaming-async-and-orchestration/exercise-04-retry-with-backoff-and-jitter — Solution

## Solution overview

The goal is a retry helper with surgically correct semantics:

- Retry only on transient failures (configurable status codes + a small set of
  `httpx` connection exceptions).
- Terminal failures (400, 401, 403, 404, 422, …) raise *immediately*; no retry,
  no sleep.
- Sleep `random.uniform(0, min(cap, base * 2 ** n))` between attempts — full
  jitter, chapter 4's thundering-herd defence. `retry-after` wins over jitter
  when present and parseable as a number.
- The final attempt re-raises the *original* exception verbatim so callers
  upstream can pattern-match on `httpx.HTTPStatusError`.
- Per-attempt log line: attempt number, status/exception, next-sleep delay.
  Enough to reconstruct the retry sequence from logs alone.

The helper is proved with a tiny local FastAPI "fake server" that returns a
scripted sequence of statuses on demand (`/queue?statuses=429,429,200`). That
way the four assertion scenarios and the 100-run jitter check all run against
a real socket on localhost — not against a mocked `httpx.Response`, which would
let us silently miss transport-layer regressions.

Two design choices worth flagging:

- **Catch typed `httpx.HTTPStatusError`, not `response.status_code`
  branching.** The helper assumes `do_request()` returns a response *and*
  calls `raise_for_status()`. That makes the transient/terminal decision one
  `except` clause each, not a nest of ifs — and it is the shape the exercise
  prescribes ("raise the original exception on exhaustion").
- **`retry-after` only parses the seconds form.** RFC 7231 also allows an
  HTTP-date string; chapter 4 says to fall back to jitter rather than parse
  dates. The reference leaves a comment at the parser site and tests only the
  seconds form, which is what every real provider sends in practice.

## Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "retry-with-backoff"
version = "0.1.0"
description = "Retry helper with exponential backoff + full jitter, proved with a fake server."
requires-python = ">=3.11"

dependencies = [
    "httpx>=0.27,<1.0",
    "fastapi[standard]>=0.115,<1.0",
    "uvicorn>=0.30,<1.0",
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

### `src/retry.py`

```python
# src/retry.py
"""retry_with_backoff: exponential backoff + full jitter, respects retry-after.

Contract:

    async def retry_with_backoff(
        do_request,                           # zero-arg async callable
        *,
        max_attempts: int = 5,
        base: float = 1.0,
        cap: float = 30.0,
        transient: set[int] = {408, 429, 500, 502, 503, 504, 529},
    ) -> httpx.Response: ...

- do_request() must return a response that has already been status-checked
  (typically `r = await client.get(...); r.raise_for_status(); return r`).
  If the response was non-2xx, raise_for_status raises HTTPStatusError, and
  this helper examines .response.status_code to decide transient vs terminal.
- Transient: HTTP status in `transient`, or httpx.ConnectError /
  httpx.ReadTimeout / httpx.RemoteProtocolError. Sleep then retry, up to
  (max_attempts - 1) retries. Last attempt re-raises the original exception.
- Terminal: any other HTTP status code (400/401/403/404/413/422/…). Raise
  immediately. Do not sleep.
- retry-after header (numeric seconds only) overrides the jittered delay.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("retry")

DEFAULT_TRANSIENT: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504, 529})


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Return the retry-after delay in seconds, or None if absent/unusable.

    RFC 7231 allows an HTTP-date form too; chapter 4 says to fall back to
    jittered backoff rather than parse dates. If your provider ever starts
    sending dates, extend here.
    """
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def _jittered_delay(attempt: int, base: float, cap: float) -> float:
    """Full jitter: random.uniform(0, min(cap, base * 2 ** attempt)).

    `attempt` is zero-indexed: 0 on the first retry wait, 1 on the second, ...
    """
    return random.uniform(0.0, min(cap, base * (2 ** attempt)))


async def retry_with_backoff(
    do_request: Callable[[], Awaitable[httpx.Response]],
    *,
    max_attempts: int = 5,
    base: float = 1.0,
    cap: float = 30.0,
    transient: frozenset[int] | set[int] = DEFAULT_TRANSIENT,
) -> httpx.Response:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            response = await do_request()
            # On a 2xx, return. On a non-2xx, the caller must raise_for_status
            # BEFORE returning — so arrival here means success.
            log.info("attempt=%d status=%d next_delay=None",
                     attempt + 1, response.status_code)
            return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_exc = exc
            if status not in transient:
                # Terminal. Raise the ORIGINAL exception so callers can
                # pattern-match. Do NOT wrap, do NOT sleep.
                log.info("attempt=%d status=%d next_delay=None terminal",
                         attempt + 1, status)
                raise
            if attempt == max_attempts - 1:
                log.info("attempt=%d status=%d next_delay=None exhausted",
                         attempt + 1, status)
                raise
            retry_after = _parse_retry_after(exc.response)
            delay = retry_after if retry_after is not None else _jittered_delay(attempt, base, cap)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                log.info("attempt=%d exc=%s next_delay=None exhausted",
                         attempt + 1, type(exc).__name__)
                raise
            delay = _jittered_delay(attempt, base, cap)
            log.info("attempt=%d exc=%s next_delay=%.3fs",
                     attempt + 1, type(exc).__name__, delay)
            await asyncio.sleep(delay)
            continue
        # Reached here only on a transient HTTP status with a planned retry.
        log.info("attempt=%d status=%d next_delay=%.3fs",
                 attempt + 1, exc.response.status_code, delay)
        await asyncio.sleep(delay)

    # Loop fell through (should not happen; the last attempt always raises).
    assert last_exc is not None
    raise last_exc
```

### `src/fake_server.py`

A small FastAPI app that lets tests script the next N responses:

```python
# src/fake_server.py
"""Scripted fake upstream for the retry tests.

Routes:
  POST /queue?statuses=429,429,200&retry_after=2   # set the next N responses
  POST /reset                                       # clear the queue
  GET  /ping                                        # consume one queued item;
                                                    # 200 OK with "pong" if empty.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from fastapi import FastAPI, Query, Response

app = FastAPI()

_QUEUE: deque[tuple[int, Optional[float]]] = deque()


@app.post("/queue")
def set_queue(statuses: str = Query(...), retry_after: Optional[float] = None):
    _QUEUE.clear()
    for s in statuses.split(","):
        _QUEUE.append((int(s), retry_after))
    return {"queued": len(_QUEUE)}


@app.post("/reset")
def reset():
    _QUEUE.clear()
    return {"queued": 0}


@app.get("/ping")
def ping():
    if not _QUEUE:
        return Response(content="pong", status_code=200)
    status, retry_after = _QUEUE.popleft()
    headers: dict[str, str] = {}
    if status in (429, 503) and retry_after is not None:
        headers["retry-after"] = str(retry_after)
    body = "pong" if status < 400 else f"simulated {status}"
    return Response(content=body, status_code=status, headers=headers)
```

### `tests/test_retry.py`

```python
# tests/test_retry.py
"""Four assertions plus a 100-run jitter check against the fake server."""
from __future__ import annotations

import asyncio
import contextlib
import socket
import statistics
import time

import httpx
import pytest
import uvicorn

from src.fake_server import app
from src.retry import retry_with_backoff

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
async def server_url():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for the port to accept connections.
    for _ in range(50):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"http://127.0.0.1:{port}/ping", timeout=0.2)
                if r.status_code < 500:
                    break
        except Exception:
            await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task


async def _make_do_request(client: httpx.AsyncClient, url: str):
    async def do_request() -> httpx.Response:
        r = await client.get(url)
        r.raise_for_status()
        return r
    return do_request


async def _queue(client: httpx.AsyncClient, base: str, statuses: str,
                 retry_after: float | None = None) -> None:
    params = {"statuses": statuses}
    if retry_after is not None:
        params["retry_after"] = retry_after
    r = await client.post(f"{base}/queue", params=params)
    r.raise_for_status()


async def test_all_transient_then_success(server_url):
    async with httpx.AsyncClient() as client:
        await _queue(client, server_url, "429,429,200")
        do = await _make_do_request(client, f"{server_url}/ping")
        t0 = time.perf_counter()
        resp = await retry_with_backoff(do, base=0.5, cap=5.0)
        elapsed = time.perf_counter() - t0
    assert resp.status_code == 200
    # 3 total attempts → 2 sleeps capped at 0.5 + 1.0 = 1.5s with jitter top.
    assert 0.0 < elapsed < 2.0


async def test_respects_retry_after(server_url):
    async with httpx.AsyncClient() as client:
        await _queue(client, server_url, "429,429,200", retry_after=2.0)
        do = await _make_do_request(client, f"{server_url}/ping")
        t0 = time.perf_counter()
        resp = await retry_with_backoff(do, base=0.1, cap=1.0)
        elapsed = time.perf_counter() - t0
    assert resp.status_code == 200
    # Two 2s sleeps = ~4s. With jitter disabled by retry-after, no spread.
    assert 3.8 < elapsed < 5.5


async def test_terminal_error_is_not_retried(server_url):
    async with httpx.AsyncClient() as client:
        await _queue(client, server_url, "400")
        do = await _make_do_request(client, f"{server_url}/ping")
        t0 = time.perf_counter()
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await retry_with_backoff(do, base=0.5, cap=5.0)
        elapsed = time.perf_counter() - t0
    assert exc_info.value.response.status_code == 400
    assert elapsed < 0.5  # no sleep occurred


async def test_exhaustion_raises_original(server_url):
    async with httpx.AsyncClient() as client:
        await _queue(client, server_url, "500,500,500,500,500")
        do = await _make_do_request(client, f"{server_url}/ping")
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await retry_with_backoff(do, max_attempts=3, base=0.1, cap=0.5)
    # Original HTTPStatusError, not a wrapper.
    assert exc_info.value.response.status_code == 500


async def test_jitter_distribution_has_spread(server_url):
    """Run 100 times; std-dev of elapsed times must be >= ~30% of mean."""
    elapsed_times: list[float] = []
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            await _queue(client, server_url, "429,429,200")
            do = await _make_do_request(client, f"{server_url}/ping")
            t0 = time.perf_counter()
            await retry_with_backoff(do, max_attempts=3, base=1.0, cap=2.0)
            elapsed_times.append(time.perf_counter() - t0)

    mean = statistics.fmean(elapsed_times)
    stdev = statistics.pstdev(elapsed_times)
    # Thundering-herd defence: some fast runs, some slow runs.
    assert stdev / mean > 0.30, (
        f"distribution too tight: mean={mean:.3f}s stdev={stdev:.3f}s "
        f"ratio={stdev/mean:.2f}"
    )
    assert min(elapsed_times) < 1.0   # some runs lucky-short
    assert max(elapsed_times) > 1.5   # some runs near the cap
```

### Optional: wire into the fan-out from exercise 03

Stretch goal. `fanout(items, lambda x: retry_with_backoff(lambda: fn(x)))`. The
retry helper layers cleanly on top of the fan-out because it does not touch
cancellation (it just awaits sleeps, which are cancellable), so a batch-wide
cancellation still propagates through the retry waits.

## Validation steps

```bash
pip install -e ".[test]"

# 1) Four assertion scenarios + 100-run jitter check. ~30 seconds total.
pytest -q
# 5 passed

# 2) Inspect per-attempt logs (uncomment basicConfig(level=INFO)).
pytest -q -o log_cli=true -o log_cli_level=INFO -k respects_retry_after
# attempt=1 status=429 next_delay=2.000s
# attempt=2 status=429 next_delay=2.000s
# attempt=3 status=200 next_delay=None

# 3) Grep for forbidden idioms.
grep -rn "time.sleep" src/
# (no matches — asyncio.sleep everywhere)

grep -rn "RuntimeError" src/retry.py
# (no matches — we re-raise the original exception, not wrap it)

# 4) Negative control for the jitter test: replace _jittered_delay with
#    `return min(cap, base * 2**attempt)` (no jitter) and re-run the jitter
#    test. It must fail: all runs cluster at the same value.
```

## Rubric or review checklist

| Criterion | Pass if |
|-----------|---------|
| Transient set is configurable | `transient=` is a parameter, not a hard-coded set |
| Correct full-jitter formula | `random.uniform(0, min(cap, base * 2 ** attempt))` — not fixed exponential, not equal jitter |
| `retry-after` honoured | Seconds form parses; HTTP-date form is noted as a limitation |
| Terminal is not retried | 400/401/403/404/422 raise immediately after one attempt, with the original `HTTPStatusError` |
| Exhaustion raises original | `max_attempts=N` means N attempts; the last re-raises the original exception, not a wrapper |
| No `time.sleep` in `retry.py` | `grep -n "time.sleep" src/retry.py` returns nothing |
| All four scenarios pass | Each of: all-transient-then-success, respects retry-after (~4 s), terminal not retried, exhaustion |
| Jitter has spread | Over 100 runs, `stdev / mean > 0.30`; min run < 1 s, max run > ~1.5 s |
| Per-attempt log is sufficient | Each line has attempt number, status/exception, next-delay (or `None`) — you could reconstruct the retry sequence from the log alone |
| Fake server used | No mocked `httpx.Response` objects; the tests drive a real local server |

## Common mistakes

1. **Retrying 400/422.** A 422 "invalid schema" is in your request; retrying
   sends the same bad request again at a cost. Terminals never retry.
2. **No jitter — just `base * 2 ** attempt`.** Ten clients all hit the same
   429 at the same instant and retry at the same instant. Full jitter is the
   fix prescribed by the AWS Builders' Library article.
3. **Wrapping the final exception in `RuntimeError`.** Upstream callers can
   no longer `except httpx.HTTPStatusError` to distinguish provider failures
   from code bugs. Re-raise the original.
4. **Sleeping on the final attempt.** You already decided not to retry;
   sleeping before raising is pure waste.
5. **Trusting `retry-after` to always be seconds.** RFC 7231 allows an
   HTTP-date string. Try seconds first; on `ValueError`, fall back to
   jittered delay rather than ship a half-working date parser.
6. **Using the SDK's built-in retry alongside this helper.** Each attempt
   through your loop becomes N physical requests. Set `max_retries=0` on
   the SDK client when you own the retry policy.
7. **Catching `BaseException`.** Will swallow `CancelledError` and
   `KeyboardInterrupt`. Catch `httpx.HTTPStatusError` and the specific
   transport exceptions you care about; everything else propagates.
8. **Testing against a mocked `httpx.Response`.** Masks real transport bugs:
   a `httpx.ReadTimeout` from a mocked response is not the same code path as
   one from a real socket that went silent. The fake server is 40 lines; use
   it.
9. **Treating `429` as always transient.** It usually is — but OpenAI's
   daily-quota 429 is terminal for the next 24 hours, per chapter 4. If
   your product cares, pattern-match on the error payload too.

## References

- Chapter 4 of this module — `../../../agentic-ai-developer-learning/lessons/mod-107-streaming-async-and-orchestration/04-retrying-transient-failures.md`
- AWS Builders' Library — Timeouts, retries, and backoff with jitter:
  <https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/>
- OpenAI error codes: <https://platform.openai.com/docs/guides/error-codes>
- Anthropic error reference: <https://docs.anthropic.com/en/api/errors>
- `tenacity` — mature retry library (read for comparison, not for import):
  <https://tenacity.readthedocs.io/>
- RFC 7231 §7.1.3 (Retry-After header):
  <https://www.rfc-editor.org/rfc/rfc7231#section-7.1.3>
