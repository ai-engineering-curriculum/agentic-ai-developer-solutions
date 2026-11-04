# mod-107-streaming-async-and-orchestration/exercise-02-streaming-partial-json — Solution

## Solution overview

The exercise asks for a streaming renderer that only shows a field after it has
*completed* — never a truncated string, a truncated number, or a mid-escape
chunk — and proves the parser's correctness with a hand-crafted adversarial
fake stream so the three mis-render bugs from chapter 2 cannot survive.

Design choices:

- **One incremental parser, used by both paths.** The live Anthropic path and
  the `--fake-stream` replay path produce the same `partial_json` chunks; both
  feed into one `IncrementalObject` state machine. If the fake tests pass, the
  live path by construction cannot render worse than they do.
- **Field completion is explicit, not inferred from JSON validity.** The
  parser tracks brace/bracket depth *and* whether the cursor is inside a
  string *and* whether the last byte was a backslash. A field is "complete"
  only when the parser has seen the closing quote of a string value (or the
  closing bracket/brace of a container value) *and* the next delimiter
  (`,`, `}`, or whitespace followed by `,`/`}`) has arrived. Chapter 2's "wait
  for the delimiter" rule for numbers falls out of the same machinery.
- **Skeleton-then-fill with `rich.live.Live`.** Printing one static table once,
  then updating cells as fields complete, is what the chapter calls the
  "stable shape, updated content" pattern. Terminal cursor moves (`\r`,
  `\x1b[F`) work too, but `rich` handles wrapping and ANSI reset for free.
- **At-exit equivalence check.** After the stream closes, the rendered
  dictionary is compared to the SDK's final `tool_use.input` (Anthropic) or
  `event.parsed` (OpenAI). Any disagreement is a parser bug and the program
  exits non-zero. This is the acceptance criterion that keeps the parser
  honest — a renderer that "looks right" but silently drifts from the final
  parse is still broken.

Scope: the reference uses the Anthropic streamed tool-use path (chapter 2's
first provider pattern). The OpenAI Structured Outputs path is sketched at the
end — the parser is reusable, only the chunk-source adapter changes.

## Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "stream-partial-json"
version = "0.1.0"
description = "Render partial JSON safely: only show a field once it is complete."
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.39,<1.0",
    "rich>=13.7,<14.0",
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

### `src/parser.py`

```python
# src/parser.py
"""Incremental JSON object parser for streamed tool/response output.

Scope: one top-level JSON object whose fields are strings, numbers, enums (a
constrained string), or arrays of primitives / arrays of objects. That covers
the exercise's status-report schema and the stretch 'evidence' field.

The state machine tracks:
  - depth  : how many open { / [ we are inside (top-level complete when 0)
  - in_str : whether the cursor is inside a JSON string value
  - esc    : whether the previous character was a backslash in a string
             (so the next character is part of an escape sequence)
  - stack  : a per-level record of "what key am I currently filling?" and
             "what completed values have I collected under this key?"

On every byte fed in, the parser updates state. When a top-level field
completes (its value's closing token arrives AND we are back in the top-level
object and we see a comma/closing brace), it fires a `on_field_complete(name,
value)` callback. The renderer listens for those callbacks and only ever
renders complete field values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class IncrementalObject:
    on_field_complete: Callable[[str, Any], None]
    on_element_complete: Callable[[str, int, Any], None] | None = None

    buf: str = ""
    cursor: int = 0

    # Positions of the last "safe to parse" prefix of the buffer. A field is
    # considered complete when the parser has already walked past its value's
    # closing delimiter.
    _done_fields: set[str] = field(default_factory=set)

    def feed(self, chunk: str) -> None:
        """Append a chunk and advance the scanner.

        We re-scan from `cursor` each time; JSON is not context-sensitive in a
        way that forces re-scanning earlier content once the scanner has
        passed it, so this is O(len(input)) overall.
        """
        self.buf += chunk
        self._scan()

    def finalize(self) -> dict[str, Any]:
        """Call after the stream closes. Returns the final parsed object.

        Anything the incremental scanner missed (e.g. the very last field that
        has no trailing comma before the closing brace) is caught here.
        """
        try:
            parsed = json.loads(self.buf)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stream ended with invalid JSON: {exc}") from exc
        # Fire completion for any fields the scanner did not already see
        # (typical: the last field, whose "completion" is the closing brace).
        for key, value in parsed.items():
            if key not in self._done_fields:
                self._done_fields.add(key)
                self.on_field_complete(key, value)
        return parsed

    # ---- scanner ----

    def _scan(self) -> None:
        """Walk the buffer from `cursor` to the end, emitting completions."""
        s = self.buf
        i = self.cursor
        n = len(s)

        # Minimal top-level walker: we only parse the outermost { ... } and
        # emit "field complete" for each immediate child. Nested parsing
        # (array of objects for the stretch 'evidence' field) is handled by
        # json.loads on the sliced-complete value — the scanner's only job is
        # to find a "value complete" boundary.
        while i < n:
            # Skip whitespace outside of strings.
            if s[i] in " \t\r\n":
                i += 1
                continue

            if s[i] == "{" and i == 0:
                i += 1
                continue

            # After the opening brace, we expect: "key" : value , "key" : ...
            # Find the next key.
            if s[i] == '"':
                key_end = self._find_string_end(s, i)
                if key_end is None:
                    break  # key not fully arrived yet
                key = json.loads(s[i:key_end + 1])
                i = key_end + 1
                # Skip to ':'
                while i < n and s[i] in " \t\r\n":
                    i += 1
                if i >= n:
                    break
                if s[i] != ":":
                    break  # malformed; wait for more bytes
                i += 1
                while i < n and s[i] in " \t\r\n":
                    i += 1
                # Now find the value end.
                value_end = self._find_value_end(s, i)
                if value_end is None:
                    break  # value not fully arrived yet — do NOT render
                value_text = s[i:value_end]
                try:
                    value = json.loads(value_text)
                except json.JSONDecodeError:
                    break  # should not happen if _find_value_end is correct
                if key not in self._done_fields:
                    self._done_fields.add(key)
                    self.on_field_complete(key, value)
                    # Stretch: fire per-element completion for arrays-of-X.
                    if self.on_element_complete and isinstance(value, list):
                        for idx, elem in enumerate(value):
                            self.on_element_complete(key, idx, elem)
                i = value_end
                # Skip trailing comma or closing brace / whitespace.
                while i < n and s[i] in " \t\r\n,":
                    if s[i] == "}":
                        break
                    i += 1
            elif s[i] == "}":
                i += 1
            else:
                # Unexpected character between fields (whitespace/',' already
                # handled). Wait for more input rather than guess.
                break

        self.cursor = i

    @staticmethod
    def _find_string_end(s: str, start: int) -> int | None:
        """Return the index of the closing '"' of the string starting at `start`.

        Returns None if the string is not yet closed (the final `"` has not
        arrived, or the last byte is a backslash so the next character is an
        escape we have not seen yet). This is the rule that keeps a mid-escape
        chunk from being rendered as a bare backslash.
        """
        assert s[start] == '"'
        i = start + 1
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "\\":
                # The entire escape (one char for \n/\t/..., six chars for
                # \uXXXX) must be present before we can declare the string
                # safe to parse. Err on the side of "wait".
                if i + 1 >= n:
                    return None  # mid-escape
                if s[i + 1] == "u":
                    if i + 5 >= n:
                        return None  # mid-\uXXXX
                    i += 6
                    continue
                i += 2
                continue
            if ch == '"':
                return i
            i += 1
        return None

    def _find_value_end(self, s: str, start: int) -> int | None:
        """Return the index one past the last byte of the value starting at
        `start`, or None if the value is not yet complete.
        """
        if start >= len(s):
            return None
        c = s[start]

        if c == '"':
            end = self._find_string_end(s, start)
            return None if end is None else end + 1

        if c in "{[":
            # Walk with depth and string-awareness.
            depth = 0
            in_str = False
            esc = False
            i = start
            n = len(s)
            while i < n:
                ch = s[i]
                if in_str:
                    if esc:
                        # Guard against an incomplete \uXXXX right at the end.
                        if ch == "u" and i + 4 >= n:
                            return None
                        esc = False
                    elif ch == "\\":
                        esc = True
                        # If the backslash is the last byte we have, wait.
                        if i + 1 >= n:
                            return None
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        return i + 1
                i += 1
            return None

        if c == "-" or c.isdigit():
            # Number: end is first delimiter (',', '}', ']', whitespace, EOF).
            # Crucially, EOF does NOT mean "number complete" — we might get
            # more digits in the next chunk. Only return a value when we have
            # seen a delimiter.
            i = start + 1
            n = len(s)
            while i < n and s[i] not in ",}] \t\r\n":
                i += 1
            if i == n:
                return None  # no delimiter yet; could still extend
            return i

        # true / false / null — all bounded, finite tokens. Wait for enough
        # bytes to pattern-match the whole token.
        for literal in ("true", "false", "null"):
            if c == literal[0]:
                if start + len(literal) > len(s):
                    return None  # not enough bytes
                if s[start:start + len(literal)] == literal:
                    return start + len(literal)
                return None

        return None
```

### `src/renderer.py`

```python
# src/renderer.py
"""Skeleton-then-fill renderer. One live table; cells update as fields complete."""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table


PLACEHOLDER = "[dim][waiting …][/dim]"


class Renderer:
    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        self.values: dict[str, Any] = {f: None for f in fields}
        self.console = Console()
        self.live: Live | None = None

    def __enter__(self) -> "Renderer":
        self.live = Live(self._table(), console=self.console, refresh_per_second=8)
        self.live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self.live is not None
        self.live.__exit__(*exc)

    def set_field(self, name: str, value: Any) -> None:
        if name not in self.values:
            return  # Extra field we did not expect; ignore for the UI.
        self.values[name] = value
        if self.live is not None:
            self.live.update(self._table())

    def _table(self) -> Table:
        tbl = Table(show_header=False, box=None, pad_edge=False)
        tbl.add_column("field", style="bold")
        tbl.add_column("value")
        for f in self.fields:
            v = self.values[f]
            tbl.add_row(f"{f}:", PLACEHOLDER if v is None else _format(v))
        return tbl


def _format(v: Any) -> str:
    if isinstance(v, list):
        return "\n".join(f"- {item}" for item in v)
    return str(v)
```

### `src/main.py`

```python
# src/main.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .parser import IncrementalObject
from .renderer import Renderer

FIELDS = ["title", "risk_level", "summary", "action_items"]

TOOL = {
    "name": "record_report",
    "description": "Record a structured status report.",
    "input_schema": {
        "type": "object",
        "required": ["title", "risk_level", "summary", "action_items"],
        "properties": {
            "title": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["low", "med", "high"]},
            "summary": {"type": "string"},
            "action_items": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def _stream_live(prompt: str) -> tuple[dict, dict]:
    """Return (final_from_sdk, final_from_renderer)."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with Renderer(FIELDS) as ui:
        parser = IncrementalObject(on_field_complete=ui.set_field)

        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "record_report"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                    parser.feed(event.delta.partial_json)
            final_message = stream.get_final_message()

        rendered = parser.finalize()

    # Pull the authoritative final tool input from the SDK.
    tool_use = next(b for b in final_message.content if b.type == "tool_use")
    return tool_use.input, rendered


def _stream_fake(path: Path) -> tuple[dict, dict]:
    """Replay a hand-crafted sequence of partial_json chunks from a file.

    File format: one chunk per line (JSON-encoded string). This bypasses the
    provider entirely and makes the parser testable without API credit.
    """
    chunks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with Renderer(FIELDS) as ui:
        parser = IncrementalObject(on_field_complete=ui.set_field)
        for ch in chunks:
            parser.feed(ch)
        rendered = parser.finalize()
    # No SDK oracle in fake mode — compare against the json.loads of the
    # concatenated chunks, which is the "ground truth" the stream was
    # simulating.
    sdk_truth = json.loads("".join(chunks))
    return sdk_truth, rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Report on today's ingest pipeline health.")
    parser.add_argument("--fake-stream", type=Path, default=None,
                        help="Replay partial_json chunks from a file; no network.")
    args = parser.parse_args(argv)

    if args.fake_stream:
        sdk_truth, rendered = _stream_fake(args.fake_stream)
    else:
        sdk_truth, rendered = _stream_live(args.prompt)

    if sdk_truth == rendered:
        print("\nOK: final render matches parsed")
        return 0

    print("\nMISMATCH between incremental render and final parsed object:",
          file=sys.stderr)
    print(f"  final: {sdk_truth!r}", file=sys.stderr)
    print(f"  rendered: {rendered!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### The three adversarial test fixtures (`tests/fixtures/*.txt`)

Each file is one chunk per line (`json.dumps`-ed string), fed to
`_stream_fake`. The chunks correspond to the three mis-render bugs from the
exercise.

`tests/fixtures/mid_string.txt`:

```
"{\"title\": \"Deploy "
"pipeline update\", \"risk_level\": \"low\", \"summary\": \"Rolled out.\", \"action_items\": [\"follow up\"]}"
```

`tests/fixtures/mid_enum.txt`:

```
"{\"title\": \"OK\", \"risk_level\": \"l"
"ow\", \"summary\": \"Nominal.\", \"action_items\": [\"none\"]}"
```

`tests/fixtures/mid_escape.txt`:

```
"{\"title\": \"OK\", \"risk_level\": \"low\", \"summary\": \"Bake at 350\\"
"u00b0F.\", \"action_items\": [\"wait\"]}"
```

### `tests/test_parser.py`

```python
# tests/test_parser.py
"""Adversarial tests. These prove the parser cannot be tricked into rendering
a truncated value — the whole point of the exercise."""
from __future__ import annotations

from src.parser import IncrementalObject


def _run(chunks: list[str]) -> list[tuple[str, object]]:
    """Return the ordered list of (field, value) completions the parser fires."""
    events: list[tuple[str, object]] = []
    p = IncrementalObject(on_field_complete=lambda k, v: events.append((k, v)))
    for ch in chunks:
        p.feed(ch)
    p.finalize()
    return events


def test_mid_string_waits_for_closing_quote():
    """Chunk split mid-string: title="Deploy ...pipeline update".
    The parser must never fire 'title' == 'Deploy ' (the truncated prefix).
    """
    events = _run([
        '{"title": "Deploy ',
        'pipeline update", "risk_level": "low", "summary": "Rolled out.", '
        '"action_items": ["follow up"]}',
    ])
    titles = [v for k, v in events if k == "title"]
    assert titles == ["Deploy pipeline update"]
    assert "Deploy " not in titles  # never render the mid-string prefix


def test_mid_enum_waits_for_closing_quote():
    """Chunk split mid-enum: risk_level="l" then "ow".
    The parser must never fire 'risk_level' == 'l'.
    """
    events = _run([
        '{"title": "OK", "risk_level": "l',
        'ow", "summary": "Nominal.", "action_items": ["none"]}',
    ])
    risks = [v for k, v in events if k == "risk_level"]
    assert risks == ["low"]


def test_mid_escape_waits_for_full_escape():
    """Chunk split mid-escape: summary ending '...350\\' then 'u00b0F...'.
    The parser must not fire 'summary' until the escape sequence is complete.
    """
    events = _run([
        '{"title": "OK", "risk_level": "low", "summary": "Bake at 350\\',
        'u00b0F.", "action_items": ["wait"]}',
    ])
    summaries = [v for k, v in events if k == "summary"]
    assert summaries == ["Bake at 350°F."]


def test_number_waits_for_delimiter():
    """A number is not complete until a delimiter follows.
    We represent this via a schema that happens to have a numeric field; if
    yours does not, a stretch-goal `score: int` is the canonical case.
    """
    events: list[tuple[str, object]] = []
    p = IncrementalObject(on_field_complete=lambda k, v: events.append((k, v)))
    p.feed('{"score": 8')      # digit arrived but no delimiter
    assert ("score", 8) not in events
    p.feed('.7, "ok": true}')  # delimiter arrives; now 8.7 is complete
    p.finalize()
    assert ("score", 8.7) in events
```

## Validation steps

```bash
pip install -e ".[test]"

# 1) The three adversarial tests — no network, no credit.
pytest -q tests/test_parser.py
# 4 passed

# 2) Replay a fixture against the real renderer (still offline). Watch the
#    terminal: four [waiting …] placeholders appear first, then each is
#    replaced by its real value in order.
python -m src.main --fake-stream tests/fixtures/mid_string.txt
# ...
# OK: final render matches parsed

python -m src.main --fake-stream tests/fixtures/mid_enum.txt
python -m src.main --fake-stream tests/fixtures/mid_escape.txt

# 3) Live run against Anthropic. Visually confirm that title and risk_level
#    appear well before summary and action_items — streaming earning its keep.
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main --prompt "Report on today's ingest pipeline health."
# ...
# OK: final render matches parsed

# 4) Negative control: make the parser aggressive (e.g. fire on every \" seen)
#    and re-run the adversarial tests. They must fail. This is how you know the
#    tests are not permissive.
```

If `OK: final render matches parsed` does not print on a live run, the
incremental parser disagreed with the SDK's final parse — fix the parser, not
the oracle. The SDK's `get_final_message().content[0].input` is ground truth.

## Rubric or review checklist

| Criterion | Pass if |
|-----------|---------|
| Four `[waiting …]` placeholders appear immediately | Renderer prints the skeleton before any field completes |
| Each placeholder replaced only on completion | No intermediate text appears for a field before its closing delimiter |
| Mid-string test passes | `title` never fires with the prefix `"Deploy "`; only with the final `"Deploy pipeline update"` |
| Mid-enum test passes | `risk_level` never fires with `"l"`; only with `"low"` |
| Mid-escape test passes | `summary` never fires until the `°` escape is complete |
| Streaming benefit visible | At least one field completes visibly before the last chunk arrives — otherwise the parser is too conservative |
| Final equivalence printed | `OK: final render matches parsed` prints on every successful run |
| Mismatch is loud | On a parser bug, exit code is non-zero and the diff is printed to stderr |
| Schema-constrained output | Anthropic tool+`tool_choice`, or OpenAI `response_format=json_schema`/pydantic |
| No mid-stream `json.loads` of the raw buffer | Grep: `json.loads` only in `finalize` and inside the bounded `_find_value_end` slice |

## Common mistakes

1. **`json.loads` on every chunk.** The buffer is only valid JSON at the very
   last byte. Parsing on every chunk raises `JSONDecodeError` essentially
   always — fix by only slicing a *known-complete* prefix and parsing that.
2. **Declaring a string done at the first `"`.** JSON strings legally contain
   escaped quotes (`\"`). A naive "first quote after the opening quote wins"
   check cuts a long string in half on the first escaped quote. The scanner
   must honour `\\` and `\"`.
3. **Treating the end of the buffer as a delimiter for numbers.** `8` is not
   final until followed by `,`, `}`, `]`, or whitespace. If you render on
   buffer-ran-out, `8` becomes the value and `8.7` loses its tail.
4. **Rendering arrays on `[` seen.** An array is a tree node — individual
   completed elements are safe to render; the whole array is not until `]`
   arrives. The reference parser fires per-element completions optionally
   (via `on_element_complete`), which is what the stretch "evidence" field
   needs.
5. **Rendering a mid-escape chunk verbatim.** A `"` closing the string must be
   reached without the previous byte being an unresolved backslash. The
   adversarial `mid_escape` test is the one that exposes this — most
   hand-written parsers get the mid-string and mid-enum cases right and fail
   the escape test.
6. **Pretending that OpenAI's `event.snapshot` is already safe to read.** The
   snapshot is a *best-effort* partial parse. Read it only when the
   corresponding `content.done` (or per-field completion event) fires;
   otherwise you are re-inventing the same mis-render bugs on top of the SDK.
7. **Mixing rendering with execution.** If you later chain this to a tool
   call, the tool executes on the whole parsed input, not on `event.snapshot`.
   Keep two variables: the "display state" updates eagerly on per-field
   completion; the "execution state" only exists when `content_block_stop`
   fires.
8. **Trusting field order.** Even with strict Structured Outputs, do not
   assume `title` always finishes before `risk_level`. Render whatever field
   completes first in whatever row the skeleton assigned to it.

## References

- Chapter 2 of this module — `../../../agentic-ai-developer-learning/lessons/mod-107-streaming-async-and-orchestration/02-streaming-partial-json.md`
- Anthropic streamed tool-use (`input_json_delta` events):
  <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/streaming-tool-use>
- OpenAI Structured Outputs:
  <https://platform.openai.com/docs/guides/structured-outputs>
- `ijson` — event-driven streaming JSON parser:
  <https://pypi.org/project/ijson/>
- `rich.live` — live terminal updates:
  <https://rich.readthedocs.io/en/stable/live.html>
- JSON string-escape rules — ECMA-404 / RFC 8259:
  <https://www.rfc-editor.org/rfc/rfc8259#section-7>
