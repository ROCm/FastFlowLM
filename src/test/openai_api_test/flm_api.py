"""Shared helpers for the OpenAI API tests.

Everything here uses only the Python 3 standard library, so the tests run
anywhere a FastFlowLM server is reachable:

* ``Api``: posts chat completion requests and reads SSE streams.
* ``fold_stream``: folds a stream of chunks into one assistant message,
  the way an OpenAI client should (tool calls merged by ``index``).
* ``validate``: a small JSON Schema checker covering the keywords the
  tests use.
* ``case`` / ``run_cases`` / ``main``: a minimal runner printing
  PASS / FAIL / SKIP per case and exiting non-zero on any failure.

Every request the tests send pins ``think``, ``top_k``, ``temperature`` and
``repeat_last_n`` explicitly: FastFlowLM applies those to the loaded model and
keeps them for later requests, so a test must never rely on what an earlier one
left set.
"""

# Every check here reports on the server under test, not on a caller
# passing bad arguments, so a wrong type in the response is a test
# failure rather than a TypeError.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_ENDPOINT = "http://127.0.0.1:52625"
DEFAULT_TIMEOUT = 300.0

# Raw markup any supported format may write; none of it may reach the client.
MARKERS = (
    "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<function=", "<parameter=",
    "<|tool_call_start|>", "<|tool_call_end|>",
    "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|tool_response>",
    "<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|call|>", "<|return|>",
)

TOOL_CALL_ID = re.compile(r"^call_[A-Za-z0-9]{24}$")


class Skip(Exception):
    """The case does not apply to the model under test."""


# ---------------------------------------------------------------- HTTP


@dataclass
class Response:
    """A non-streaming response: HTTP status and decoded JSON body."""

    status: int
    body: dict[str, Any]

    def error_message(self) -> str:
        err = self.body.get("error")
        if isinstance(err, dict):
            return str(err.get("message", ""))
        return str(err or "")


@dataclass
class Stream:
    """A streaming response: every ``data:`` payload before ``[DONE]``."""

    status: int
    chunks: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False  # saw "data: [DONE]"
    error: dict[str, Any] | None = None  # body of a non-200 reply


class Api:
    """Talks to one FastFlowLM server about one model."""

    def __init__(self, endpoint: str, model: str, timeout: float, always_thinks: bool = False) -> None:
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        # the model reasons whatever think says (qwen3-tk, deepseek-r1, gpt-oss, ...)
        self.always_thinks = always_thinks

    def body(self, messages: list[dict[str, Any]], *, think: bool = False,
             max_tokens: int = 256, **fields: Any) -> dict[str, Any]:
        """Build a request body with sampling pinned (see the module note)."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "think": think,
            "top_k": 1,
            "temperature": 0.0,
            "repeat_last_n": 64,
        }
        body.update(fields)
        return body

    def _open(self, body: dict[str, Any]):
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def post(self, body: dict[str, Any]) -> Response:
        """Send a non-streaming request; HTTP errors are returned, not raised."""
        body = dict(body, stream=False)
        try:
            with self._open(body) as resp:
                return Response(resp.status, json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return Response(e.code, json.loads(raw))
            except json.JSONDecodeError:
                return Response(e.code, {"error": {"message": raw}})

    def stream(self, body: dict[str, Any]) -> Stream:
        """Send a streaming request and collect its SSE payloads."""
        body = dict(body, stream=True)
        try:
            resp = self._open(body)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                error = json.loads(raw)
            except json.JSONDecodeError:
                error = {"error": {"message": raw}}
            return Stream(e.code, error=error)
        out = Stream(resp.status)
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    out.done = True
                    break
                out.chunks.append(json.loads(payload))
        return out

    def ok(self, body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming request that must succeed; returns ``choices[0]``."""
        resp = self.post(body)
        if resp.status != 200:
            raise_for_capability(resp.error_message())
            raise AssertionError(f"HTTP {resp.status}: {resp.error_message()}")
        expect(isinstance(resp.body.get("choices"), list) and len(resp.body["choices"]) == 1,
               f"expected one choice, got {resp.body.get('choices')!r}")
        return resp.body["choices"][0]

    def ok_stream(self, body: dict[str, Any]) -> Stream:
        """Streaming request that must succeed and end with ``[DONE]``."""
        s = self.stream(body)
        if s.status != 200:
            message = str((s.error or {}).get("error", {}).get("message", s.error))
            raise_for_capability(message)
            raise AssertionError(f"HTTP {s.status}: {message}")
        expect(s.done, "stream ended without data: [DONE]")
        return s


def raise_for_capability(message: str) -> None:
    """Turn a 'this model cannot do that' rejection into a skip."""
    if "has no tool call syntax" in message:
        raise Skip("the model's chat format has no tool call syntax")
    if "does not support tool calling" in message:
        raise Skip("the model is not shown tools")
    if "only supports single-turn" in message:
        raise Skip("the model only supports single-turn requests")
    if "logprobs is not supported" in message:
        raise Skip("the model cannot report logprobs")


# ---------------------------------------------------------------- stream folding


@dataclass
class Folded:
    """An assistant turn rebuilt from stream chunks."""

    reasoning: str
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, Any] | None
    # kind of each delta in order: "reasoning", "content" or "tool:<index>"
    order: list[str]


def fold_stream(chunks: list[dict[str, Any]]) -> Folded:
    """Fold chunks into one message, merging tool call deltas by index."""
    reasoning: list[str] = []
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage = None
    order: list[str] = []
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                order.append("reasoning")
            if delta.get("content"):
                content.append(delta["content"])
                order.append("content")
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {
                    "id": None, "type": None, "function": {"name": "", "arguments": ""},
                })
                if tc.get("id"):
                    call["id"] = tc["id"]
                if tc.get("type"):
                    call["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] = fn["name"]
                call["function"]["arguments"] += fn.get("arguments") or ""
                order.append(f"tool:{tc['index']}")
    return Folded(
        reasoning="".join(reasoning),
        content="".join(content),
        tool_calls=[calls[i] for i in sorted(calls)],
        finish_reason=finish_reason,
        usage=usage,
        order=order,
    )


# ---------------------------------------------------------------- checks


def expect(condition: bool, message: str) -> None:
    """Assert with a message that survives ``python -O``."""
    if not condition:
        raise AssertionError(message)


def assert_no_markers(text: str | None, where: str) -> None:
    """No raw template markup may leak into what the client sees."""
    for marker in MARKERS:
        expect(marker not in (text or ""), f"raw marker {marker!r} leaked into {where}: {text!r}")


def parse_arguments(call: dict[str, Any]) -> Any:
    """Tool arguments must be a JSON string; return the decoded value."""
    args = call["function"]["arguments"]
    expect(isinstance(args, str), f"arguments must be a JSON string, got {type(args).__name__}")
    try:
        return json.loads(args)
    except json.JSONDecodeError as e:
        raise AssertionError(f"arguments are not valid JSON ({e}): {args!r}") from None


def tool_schema(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tools:
        if tool["function"]["name"] == name:
            return tool["function"].get("parameters", {})
    raise AssertionError(f"model called {name!r}, which is not one of the request's tools")


def check_tool_calls(calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
    """Every call is well formed, uses a declared tool and matches its schema."""
    ids = set()
    for i, call in enumerate(calls):
        expect(call.get("type") == "function", f"call {i}: type is {call.get('type')!r}")
        expect(isinstance(call.get("id"), str) and TOOL_CALL_ID.match(call["id"]) is not None,
               f"call {i}: id {call.get('id')!r} is not call_ + 24 alphanumerics")
        expect(call["id"] not in ids, f"call {i}: id {call['id']} is not unique")
        ids.add(call["id"])
        args = parse_arguments(call)
        validate(args, tool_schema(tools, call["function"]["name"]), f"call {i} arguments")


def check_message_contract(choice: dict[str, Any]) -> dict[str, Any]:
    """The non-streaming message shape every response must have."""
    msg = choice.get("message")
    expect(isinstance(msg, dict), f"choice has no message: {choice!r}")
    expect(msg.get("role") == "assistant", f"role is {msg.get('role')!r}")
    expect(choice.get("finish_reason") in ("stop", "length", "tool_calls"),
           f"unexpected finish_reason {choice.get('finish_reason')!r}")
    if "reasoning_content" in msg:
        expect(isinstance(msg["reasoning_content"], str) and msg["reasoning_content"] != "",
               "reasoning_content, when present, must be a non-empty string")
    assert_no_markers(msg.get("content"), "content")
    assert_no_markers(msg.get("reasoning_content"), "reasoning_content")
    if msg.get("tool_calls"):
        expect(msg.get("content") is None or isinstance(msg["content"], str),
               "content must be a string or null")
        for i, call in enumerate(msg["tool_calls"]):
            expect(call.get("index") == i, f"tool call {i} has index {call.get('index')!r}")
    else:
        expect(isinstance(msg.get("content"), str), "content must be a string when there are no tool calls")
        expect(choice.get("finish_reason") != "tool_calls", "finish_reason is tool_calls without tool calls")
    return msg


def check_stream_contract(s: Stream, *, include_usage: bool) -> Folded:
    """The chunk-level shape every stream must have; returns the folded turn."""
    expect(len(s.chunks) > 0, "stream has no chunks")
    ids = {c.get("id") for c in s.chunks}
    expect(len(ids) == 1, f"chunks carry different ids: {ids}")
    seen_tool_ids: dict[int, str] = {}
    finish_chunks = 0
    for n, chunk in enumerate(s.chunks):
        expect(chunk.get("object") == "chat.completion.chunk", f"chunk {n}: object is {chunk.get('object')!r}")
        if include_usage:
            expect("usage" in chunk, f"chunk {n}: include_usage asked for a usage key on every chunk")
        else:
            expect("usage" not in chunk, f"chunk {n}: usage sent although include_usage was not asked")
        choices = chunk.get("choices")
        expect(isinstance(choices, list), f"chunk {n}: choices is {choices!r}")
        if not choices:
            expect(include_usage and n == len(s.chunks) - 1,
                   f"chunk {n}: only the last chunk, with include_usage, may have no choices")
            expect(isinstance(chunk.get("usage"), dict), "the usage chunk has no usage")
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") is not None:
            finish_chunks += 1
            continue
        parts = [k for k in ("reasoning_content", "content", "tool_calls") if delta.get(k)]
        expect(len(parts) == 1, f"chunk {n}: a delta must carry exactly one part, got {parts}: {delta}")
        for tc in delta.get("tool_calls") or []:
            index = tc.get("index")
            expect(isinstance(index, int), f"chunk {n}: tool call delta without an index")
            if index not in seen_tool_ids:
                expect(index == len(seen_tool_ids), f"chunk {n}: tool call index {index} skips ahead")
                expect(bool(tc.get("id")) and tc.get("type") == "function" and bool(tc["function"].get("name")),
                       f"chunk {n}: the first delta of a call must carry id, type and name: {tc}")
                seen_tool_ids[index] = tc["id"]
            else:
                expect("id" not in tc and "name" not in (tc.get("function") or {}),
                       f"chunk {n}: a later delta of call {index} repeats its id or name: {tc}")
    expect(finish_chunks == 1, f"expected one chunk with a finish_reason, got {finish_chunks}")
    folded = fold_stream(s.chunks)
    assert_no_markers(folded.content, "streamed content")
    assert_no_markers(folded.reasoning, "streamed reasoning_content")
    if include_usage:
        expect(folded.usage is not None, "include_usage was asked but no usage arrived")
    return folded


def assert_parity(api: Api, body: dict[str, Any], key: Callable[[dict[str, Any]], Any],
                  attempts: int = 3) -> Folded:
    """The stream folds to the non-streaming message, compared through key.

    key maps a message dict (content, reasoning_content, tool_calls) to the
    value to compare. Greedy decoding on the NPU is not always bit-reproducible
    between identical requests, mostly over long outputs, so each mode is tried
    up to ``attempts`` times: any matching pair passes (a parser disagreement
    would be systematic), and no match is a failure only when the
    non-streaming request reproduced itself; otherwise the case is skipped.
    """

    def from_message(msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": msg.get("content") or None,
            "reasoning_content": msg.get("reasoning_content", ""),
            "tool_calls": msg.get("tool_calls") or [],
        }

    plain: list[Any] = []
    streamed: list[Any] = []
    for _ in range(attempts):
        plain.append(key(from_message(api.ok(body)["message"])))
        folded = check_stream_contract(api.ok_stream(body), include_usage=False)
        streamed.append(key({
            "content": folded.content or None,
            "reasoning_content": folded.reasoning,
            "tool_calls": folded.tool_calls,
        }))
        if any(s == p for s in streamed for p in plain):
            return folded
    if any(p != plain[0] for p in plain):
        raise Skip(f"greedy decoding did not reproduce over {attempts} identical requests; "
                   "cannot compare stream and non-stream")
    raise AssertionError(f"stream and non-stream differ:\n  stream:     {streamed[0]!r}\n"
                         f"  non-stream: {plain[0]!r}")


# ---------------------------------------------------------------- JSON Schema


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate(value: Any, schema: Any, where: str = "value", root: Any = None) -> None:
    """Check value against the JSON Schema keywords these tests use.

    Covers type, enum, const, properties, required, additionalProperties,
    items, anyOf, minimum/maximum, pattern and local $ref.
    """
    root = schema if root is None else root
    if schema is True or not isinstance(schema, dict):
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        expect(ref.startswith("#/"), f"{where}: only local $ref is supported, got {ref}")
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        validate(value, target, where, root)
        return
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                validate(value, option, where, root)
                break
            except AssertionError as e:
                errors.append(str(e))
        else:
            raise AssertionError(f"{where}: matches no anyOf option: {errors}")
    kind = schema.get("type")
    if isinstance(kind, list):
        expect(any(_type_ok(value, k) for k in kind), f"{where}: {value!r} is not any of {kind}")
    elif isinstance(kind, str):
        expect(_type_ok(value, kind), f"{where}: {value!r} is not of type {kind}")
    if "enum" in schema:
        expect(value in schema["enum"], f"{where}: {value!r} is not one of {schema['enum']}")
    if "const" in schema:
        expect(value == schema["const"], f"{where}: {value!r} != const {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            expect(value >= schema["minimum"], f"{where}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema:
            expect(value <= schema["maximum"], f"{where}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str) and "pattern" in schema:
        expect(re.search(schema["pattern"], value) is not None,
               f"{where}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            expect(name in value, f"{where}: required key {name!r} is missing from {value!r}")
        for name, item in value.items():
            if name in props:
                validate(item, props[name], f"{where}.{name}", root)
            elif schema.get("additionalProperties") is False:
                raise AssertionError(f"{where}: unexpected key {name!r}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{where}[{i}]", root)


# ---------------------------------------------------------------- runner

Case = Callable[[Api], None]


def case(func: Case) -> Case:
    """Mark a test function; its docstring's first line is its description."""
    func.is_case = True  # type: ignore[attr-defined]
    return func


def collect(module_globals: dict[str, Any]) -> list[Case]:
    """The cases of a module, in definition order."""
    return [f for f in module_globals.values() if callable(f) and getattr(f, "is_case", False)]


@dataclass
class Tally:
    passed: int = 0
    failed: int = 0
    skipped: int = 0

    def add(self, other: Tally) -> None:
        self.passed += other.passed
        self.failed += other.failed
        self.skipped += other.skipped


def run_cases(cases: list[Case], api: Api, *, only: str | None = None, verbose: bool = False) -> Tally:
    """Run cases and print one line per case."""
    tally = Tally()
    for func in cases:
        module = os.path.splitext(os.path.basename(func.__globals__.get("__file__", func.__module__)))[0]
        name = f"{module}.{func.__name__}"
        if only and only not in name:
            continue
        start = time.monotonic()
        try:
            func(api)
        except Skip as e:
            tally.skipped += 1
            print(f"SKIP  {name}: {e}")
            continue
        except AssertionError as e:
            tally.failed += 1
            print(f"FAIL  {name} ({time.monotonic() - start:.1f}s): {e}")
            if verbose:
                traceback.print_exc()
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            tally.failed += 1
            print(f"FAIL  {name}: cannot reach the server: {e}")
            continue
        tally.passed += 1
        print(f"PASS  {name} ({time.monotonic() - start:.1f}s)")
    return tally


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", help="model tag currently loaded by the FastFlowLM server")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="server base URL (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--always-thinks", action="store_true",
                        help="the model cannot turn thinking off (e.g. qwen3-tk, deepseek-r1, gpt-oss): "
                             "do not require think=false to drop reasoning_content")
    parser.add_argument("-k", dest="only", help="run only cases whose name contains this text")
    parser.add_argument("-v", "--verbose", action="store_true", help="print tracebacks of failures")
    return parser.parse_args()


def make_api(args: argparse.Namespace) -> Api:
    return Api(args.endpoint, args.model, args.timeout, always_thinks=args.always_thinks)


def summarize(tally: Tally) -> int:
    print(f"\n{tally.passed} passed, {tally.failed} failed, {tally.skipped} skipped")
    return 1 if tally.failed else 0


def main(cases: list[Case], description: str) -> None:
    """Entry point for running one test module on its own."""
    args = parse_args(description)
    api = make_api(args)
    sys.exit(summarize(run_cases(cases, api, only=args.only, verbose=args.verbose)))
