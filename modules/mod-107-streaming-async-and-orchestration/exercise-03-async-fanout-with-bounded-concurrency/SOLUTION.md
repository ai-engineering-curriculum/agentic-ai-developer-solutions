# mod-107-streaming-async-and-orchestration/exercise-03-async-fanout-with-bounded-concurrency — Solution

## Solution overview

The deliverable is a small, reusable `fanout` coroutine — the one every
subsequent module in this track will import. The contract is modest: run `fn`
over each `item` in `items` with at most `concurrency` in flight, hold a
per-item timeout, return a list aligned with the input, keep exceptions
in-place at their index, and cancel the whole batch promptly when the outer
task is cancelled.

The reference solution leans on three Python 3.11+ features because they make
the invariants tractable:

- `asyncio.TaskGroup` — gives structured concurrency: the `async with` block
  does not return until every task has completed or been cancelled. No task
  leaks past the boundary. `TaskGroup` also propagates cancellation into the
  children for free; a `task.cancel()` from the outside turns into a
  `CancelledError` inside every active `fn` call without extra glue.
- `asyncio.Semaphore` — the primary cap. Acquired *inside* `fn`'s wrapper so
  the cap bounds in-flight work, not queued work.
- `asyncio.wait_for` — per-item deadline. Wraps `fn(item)` so one misbehaving
  item cannot hold a semaphore slot indefinitely.

Two decisions that look like detail but matter:

- **The semaphore is acquired strictly *inside* the task, not outside.**
  Acquiring it outside the task body would serialise all scheduling, defeat
  the "aligned list" requirement, and make cancellation leaky. The
  `tg.create_task` loop stays tight: schedule everything; each task blocks
  on the semaphore before issuing its request.
- **`fanout` does not own the provider client.** The caller passes `fn`,
  which already closes over a shared `AsyncAnthropic` / `AsyncOpenAI` /
  `httpx.AsyncClient`. The exercise shows the wiring in `main.py` for
  completeness, and a `FanoutRunner` context manager that owns the client is
  sketched in the stretch-goal section — but the core helper is
  client-agnostic so it also wraps non-LLM I/O (database reads, HTTP fetches)
  without special-casing.

An observability helper (`on_event`) lets the caller log per-item start, end,
success/failure, and the current in-flight count. The default is a quiet
no-op; the smoke tests use a recording callback.

## Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "async-fanout"
version = "0.1.0"
description = "Bounded, cancellable async fan-out helper for LLM calls."
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.39,<1.0",
    "httpx>=0.27,<1.0",
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

### `src/fanout.py`

```python
# src/fanout.py
"""Bounded, cancellable async fan-out.

Usage:

    async def summarise(headline: str) -> str:
        resp = await client.messages.create(...)
        return resp.content[0].text

    results = await fanout(headlines, summarise, concurrency=8, per_item_timeout=20.0)
    # results is aligned with `headlines`; a failing item is the exception
    # object at its index (not raised).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class _FanoutCallback(Protocol):
    def __call__(self, event: "FanoutEvent") -> None: ...


@dataclass
class FanoutEvent:
    kind: str                 # "start" | "end"
    index: int
    item: object
    elapsed_s: float | None = None
    outcome: str | None = None  # "ok" | "timeout" | "error" | None (on start)
    in_flight: int | None = None


async def fanout(
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int = 8,
    per_item_timeout: float = 30.0,
    on_event: _FanoutCallback | None = None,
) -> list[R | BaseException]:
    """Run fn on every item with bounded concurrency and per-item timeout.

    Returns a list the same length as `items`. For item i, results[i] is the
    successful return value of fn(items[i]) or the exception that terminated
    it (TimeoutError, httpx error, provider error, ...). CancelledError is
    NOT returned in-place — if the caller cancels the fanout, we re-raise.
    """
    if not items:
        return []
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    sem = asyncio.Semaphore(concurrency)
    results: list[R | BaseException] = [None] * len(items)  # type: ignore[list-item]
    in_flight = 0

    def _emit(event: FanoutEvent) -> None:
        if on_event is not None:
            on_event(event)

    async def one(idx: int, item: T) -> None:
        nonlocal in_flight
        # Acquire INSIDE the task body — the cap is on in-flight work, not
        # on scheduled tasks. Acquiring outside would serialise the whole
        # batch because gather/tg would await each acquire sequentially.
        async with sem:
            in_flight += 1
            t_start = time.perf_counter()
            _emit(FanoutEvent("start", idx, item, in_flight=in_flight))
            outcome = "ok"
            try:
                results[idx] = await asyncio.wait_for(fn(item), timeout=per_item_timeout)
            except asyncio.CancelledError:
                # Propagate cancellation up the TaskGroup. Do NOT turn this
                # into a per-item result — the caller cancelled the fan-out
                # and expects the whole task to raise CancelledError.
                outcome = "cancelled"
                raise
            except asyncio.TimeoutError as exc:
                # wait_for already cancelled fn(item); record the timeout.
                results[idx] = exc
                outcome = "timeout"
            except Exception as exc:
                # Any other error becomes an in-place result for this item;
                # the rest of the batch still runs.
                results[idx] = exc
                outcome = "error"
            finally:
                elapsed = time.perf_counter() - t_start
                _emit(FanoutEvent("end", idx, item,
                                  elapsed_s=elapsed, outcome=outcome,
                                  in_flight=in_flight))
                in_flight -= 1

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(one(i, item))

    return results
```

### `src/runner.py` — stretch: a `FanoutRunner` that owns the client

```python
# src/runner.py
"""Optional convenience wrapper that owns a provider client.

Lets a server reuse the same client across many fanout batches rather than
building a new one per request. Still delegates the hard work to
`fanout()`; this type is purely wiring.
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from .fanout import fanout


class FanoutRunner:
    def __init__(self, *, concurrency: int = 8, per_item_timeout: float = 30.0) -> None:
        self.concurrency = concurrency
        self.per_item_timeout = per_item_timeout
        self._stack: AsyncExitStack | None = None
        self.client: AsyncAnthropic | None = None

    async def __aenter__(self) -> "FanoutRunner":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        # The AsyncAnthropic client itself is an async context manager;
        # entering it lets us tear its httpx pool down cleanly on exit.
        self.client = await self._stack.enter_async_context(AsyncAnthropic())
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._stack is not None
        await self._stack.__aexit__(*exc)
        self._stack = None
        self.client = None

    async def run(self, items, fn: Callable[[object], Awaitable[object]]):
        return await fanout(items, fn,
                            concurrency=self.concurrency,
                            per_item_timeout=self.per_item_timeout)
```

### `src/main.py` — an example driver with a real workload

```python
# src/main.py
"""Driver that fans out N short summarisations under bounded concurrency.

Run:
    python -m src.main --n 40 --concurrency 8 --model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

from anthropic import AsyncAnthropic

from .fanout import FanoutEvent, fanout

HEADLINES = [
    f"Headline {i}: a terse 5-10 word news-style line to summarise."
    for i in range(1, 101)
]


def _make_summariser(client: AsyncAnthropic, model: str):
    async def summarise(headline: str) -> str:
        resp = await client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user",
                       "content": f"Summarise in one sentence: {headline}"}],
        )
        return resp.content[0].text
    return summarise


def _logger():
    """Return (callback, done_counter, max_in_flight_tracker)."""
    counters = {"done": 0, "ok": 0, "err": 0, "max_in_flight": 0, "sum_elapsed": 0.0}
    def _cb(ev: FanoutEvent) -> None:
        if ev.kind == "start":
            if ev.in_flight is not None and ev.in_flight > counters["max_in_flight"]:
                counters["max_in_flight"] = ev.in_flight
        else:
            counters["done"] += 1
            counters["sum_elapsed"] += ev.elapsed_s or 0.0
            if ev.outcome == "ok":
                counters["ok"] += 1
            else:
                counters["err"] += 1
            print(f"[fanout] idx={ev.index:>3} outcome={ev.outcome:<8} "
                  f"elapsed={ev.elapsed_s:.2f}s in_flight={ev.in_flight}")
    return _cb, counters


async def _main(args) -> int:
    headlines = HEADLINES[:args.n]
    async with AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) as client:
        fn = _make_summariser(client, args.model)
        cb, counters = _logger()
        t0 = time.perf_counter()
        results = await fanout(
            headlines, fn,
            concurrency=args.concurrency,
            per_item_timeout=args.timeout,
            on_event=cb,
        )
        wall = time.perf_counter() - t0

    assert len(results) == len(headlines)
    print(f"\n[fanout] wall_clock={wall:.2f}s  sum_elapsed={counters['sum_elapsed']:.2f}s  "
          f"ratio={counters['sum_elapsed']/wall:.1f}x  max_in_flight={counters['max_in_flight']}  "
          f"ok={counters['ok']}  err={counters['err']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--model", default="claude-opus-4-7")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tests/test_fanout.py` — offline, no API credit

```python
# tests/test_fanout.py
from __future__ import annotations

import asyncio
import time

import pytest

from src.fanout import FanoutEvent, fanout

pytestmark = pytest.mark.asyncio


async def test_aligned_output_and_exceptions_in_place():
    async def fn(x: int) -> int:
        if x == 5:
            raise RuntimeError("boom")
        await asyncio.sleep(0.01)
        return x * 2

    items = list(range(10))
    results = await fanout(items, fn, concurrency=4)

    assert len(results) == len(items)
    for i, r in enumerate(results):
        if i == 5:
            assert isinstance(r, RuntimeError)
        else:
            assert r == i * 2


async def test_concurrency_actually_parallel():
    """Each item sleeps 0.1s. With concurrency=10, 10 items should finish in
    ~0.1s, not 1.0s."""
    async def fn(_: int) -> int:
        await asyncio.sleep(0.1)
        return 1

    t0 = time.perf_counter()
    await fanout(list(range(10)), fn, concurrency=10)
    assert (time.perf_counter() - t0) < 0.3  # generous bound


async def test_concurrency_one_is_serial():
    async def fn(_: int) -> int:
        await asyncio.sleep(0.05)
        return 1

    t0 = time.perf_counter()
    await fanout(list(range(10)), fn, concurrency=1)
    assert (time.perf_counter() - t0) > 0.4  # close to 10 * 0.05


async def test_per_item_timeout_returns_timeout_error():
    async def fn(x: int) -> int:
        await asyncio.sleep(1.0)  # longer than the timeout
        return x

    results = await fanout([1, 2, 3], fn, concurrency=3, per_item_timeout=0.05)
    assert all(isinstance(r, asyncio.TimeoutError) for r in results)


async def test_cancellation_is_prompt():
    """Caller cancels the fanout task; it should raise CancelledError within
    ~500ms and leave no in-flight work."""
    started = 0
    finished = 0

    async def fn(_: int) -> int:
        nonlocal started, finished
        started += 1
        try:
            await asyncio.sleep(5.0)
        finally:
            finished += 1
        return 1

    task = asyncio.create_task(fanout(list(range(20)), fn, concurrency=5))
    await asyncio.sleep(0.2)  # let some work start
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    # Every task that started must have reached its finally — if finished
    # trails started, the TaskGroup is still cleaning up and we have a bug.
    assert finished == started


async def test_event_stream_in_flight_cap():
    """in_flight count never exceeds concurrency."""
    events: list[FanoutEvent] = []

    async def fn(_: int) -> int:
        await asyncio.sleep(0.02)
        return 1

    await fanout(list(range(30)), fn, concurrency=5, on_event=events.append)
    peak = max((e.in_flight or 0) for e in events if e.kind == "start")
    assert peak <= 5
```

## Validation steps

```bash
pip install -e ".[test]"

# 1) Offline tests — proves alignment, timeout, cancellation, and the cap
#    without spending a cent.
pytest -q
# 6 passed

# 2) Live smoke: 40 headlines at concurrency=8 should be ~40/8 = 5x median
#    single-request latency, not close to 40x.
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main --n 40 --concurrency 8
# [fanout] wall_clock=5.4s sum_elapsed=38.1s ratio=7.1x max_in_flight=8 ok=40 err=0

# 3) Sanity check: concurrency=1 serialises.
python -m src.main --n 10 --concurrency 1
# wall_clock ≈ sum_elapsed, ratio ≈ 1.0x

# 4) Injected failure: inject one item with a system prompt large enough to
#    400. The run still returns 40 items (39 ok + 1 error at the failing
#    index). Edit main.py to insert a 1 MB headline at index 7 for this.

# 5) Aggressive per-item timeout: everything times out quickly and the batch
#    finishes under ~2 seconds.
python -m src.main --n 40 --concurrency 8 --timeout 0.001
# Expect: every result is TimeoutError, wall_clock < 2s.

# 6) Cancellation test (ad-hoc): hit Ctrl-C mid-run. The process should exit
#    within ~1s and the in-flight count in the log should drop to 0 before
#    exit. If requests continue to print "end" events after Ctrl-C, the
#    async with on the client is wrong somewhere.

# 7) Client-reuse grep: confirm exactly one AsyncAnthropic() construction
#    runs per process.
grep -n "AsyncAnthropic(" src/main.py
# Expect: exactly one line in the async context manager, not inside summarise.
```

## Rubric or review checklist

| Criterion | Pass if |
|-----------|---------|
| Aligned output | `fanout(items, fn)` returns a list with `results[i]` corresponding to `items[i]`; verified by a test that fails one specific index |
| Exceptions in-place | A failing item appears as the raised exception at its index; the rest of the batch still runs |
| `CancelledError` not swallowed | Cancelling the outer task raises `CancelledError`; it is *not* returned as a per-item result |
| Concurrency cap honoured | Observed `max_in_flight` over the run is `<= concurrency` |
| Serial sanity check | `concurrency=1` produces `ratio ≈ 1.0` of sum / wall |
| Parallel sanity check | `concurrency=8` on 40 items with similar per-item latency yields `ratio ≈ 5–8x` |
| Per-item timeout works | With `per_item_timeout=0.001` every result is `TimeoutError` and wall-clock stays bounded |
| Cancellation is prompt | After `task.cancel()` the fanout task raises within ~500 ms and no new in-flight increments appear |
| One client per batch | Grepping for `AsyncAnthropic(` / `AsyncOpenAI(` in the driver finds one construction, outside the per-item function |
| No `time.sleep` inside coroutines | Grep returns no hits inside `src/`; `asyncio.sleep` everywhere |
| `TaskGroup` preferred over bare `gather` | `async with asyncio.TaskGroup()` is used; if `gather` is used, `return_exceptions=True` is set and exceptions are post-processed into aligned positions |

## Common mistakes

1. **Acquiring the semaphore outside the task body.** Something like
   `async with sem: tg.create_task(...)` serialises the whole batch because
   the semaphore holds across the `create_task` boundary. The cap belongs
   *inside* the task's coroutine.
2. **Shuffling by completion order with `gather(*coros)`.** `gather`'s return
   order is argument order, which looks fine until you process the result
   list without threading indices through — a `return_exceptions=True` path
   can quietly drop the position. Prefer `TaskGroup` + assigning into an
   indexed list.
3. **Creating one `AsyncAnthropic`/`AsyncOpenAI`/`httpx.AsyncClient` per
   item.** Defeats keep-alive, re-does the TLS handshake every call, and
   under load exhausts sockets. One client per batch (or, better, one per
   process, shared between batches).
4. **Omitting `httpx.Timeout`.** A hung request holds its semaphore slot
   forever; the batch hangs on one stuck item. `asyncio.wait_for` catches
   the slow case at the fanout layer, but set a transport timeout on the
   client too so `fn` fails fast when the socket itself is dead.
5. **Swallowing `CancelledError`.** `except Exception:` is safe; `except
   BaseException:` is where the bug hides. If you catch `BaseException`
   around `await fn(item)`, re-raise `CancelledError` immediately — anything
   else wedges shutdown.
6. **`time.sleep` inside a coroutine.** Blocks the whole event loop. If you
   need to pace requests, `await asyncio.sleep(...)`.
7. **Treating `fanout` as "tail-latency bounded" because it has a per-item
   timeout.** `per_item_timeout` caps *one item*; the batch can still take
   `(len(items) / concurrency) * per_item_timeout` wall-clock. For a global
   deadline, add a stretch-goal outer `asyncio.wait_for(fanout(...),
   timeout=T)` and handle `CancelledError` inside `fn` to stop gracefully.
8. **Logging in-flight count from inside `fn` without a lock.** The counter
   `in_flight` in the reference is updated inside the `async with sem`
   region; async tasks do not preempt each other, so a plain counter is
   safe. If you move the counter update outside the sem region, you lose
   that guarantee.

## References

- Chapter 3 of this module — `../../../agentic-ai-developer-learning/lessons/mod-107-streaming-async-and-orchestration/03-async-fanout-with-asyncio-and-httpx.md`
- `asyncio.TaskGroup`: <https://docs.python.org/3/library/asyncio-task.html#task-groups>
- `asyncio.Semaphore`: <https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore>
- `asyncio.wait_for`: <https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for>
- `httpx.AsyncClient` + `Limits`: <https://www.python-httpx.org/async/> and <https://www.python-httpx.org/advanced/resource-limits/>
- Anthropic Python SDK (AsyncAnthropic): <https://github.com/anthropics/anthropic-sdk-python>
- OpenAI Python SDK (AsyncOpenAI): <https://github.com/openai/openai-python>
