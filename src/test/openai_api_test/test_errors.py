#!/usr/bin/env python3
"""Requests the server must refuse with an HTTP 4xx instead of serving loosely.

Each malformed tool_choice / response_format is rejected with a 400, and a
model name the catalog does not have with a 404, before any generation
starts, with an OpenAI style error body
(``{"error": {"message", "type": "invalid_request_error", "code": 400}}``),
for streaming requests too. A good request afterwards must still work:
a rejection must not leave the server in a bad state.

A prompt that cannot be prefilled is refused too, on /v1/chat/completions and
/v1/completions: a prompt longer than the context gets a 4xx naming the
problem, and a prefill that fails in the engine an HTTP 500, never a 200 that
reads as an empty reply. A streamed request gets the same status, or, once
its stream has started, an error event followed by ``[DONE]``.

The engine failure is injected: a server started with
``FLM_TEST_PREFILL_FAULT=throw`` (or ``false``) fails the prefill of every
prompt containing the text ``FLM_TEST_PREFILL_FAULT``, and only those. The
``prefill_fault`` cases skip against a server without it.

Usage::

    python3 test_errors.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import flm_api
from flm_api import Api, case, expect
from fixtures import WEATHER_TOOLS, function_choice, json_schema_format, user


def expect_400(api: Api, fields: dict[str, Any], message_part: str) -> None:
    """Non-streaming and streaming both refuse, naming the problem."""
    body = api.body(user("hi"), max_tokens=8, **fields)
    resp = api.post(body)
    expect(resp.status == 400, f"expected HTTP 400, got {resp.status}: {resp.body}")
    err = resp.body.get("error")
    expect(isinstance(err, dict), f"error body is not an object: {resp.body}")
    expect(err.get("type") == "invalid_request_error", f"error type is {err.get('type')!r}")
    expect(message_part in str(err.get("message")), f"message {err.get('message')!r} lacks {message_part!r}")

    s = api.stream(body)
    expect(s.status == 400, f"streaming: expected HTTP 400, got {s.status}")
    expect(not s.chunks, "streaming: chunks were sent before the rejection")


@case
def bad_tool_choice_string(api: Api) -> None:
    """An unknown tool_choice string."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": "sometimes"}, "tool_choice must be")


@case
def bad_tool_choice_object(api: Api) -> None:
    """A tool_choice object without function.name."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": {"type": "function"}}, "tool_choice must be")


@case
def required_without_tools(api: Api) -> None:
    """required with no tools to call."""
    expect_400(api, {"tool_choice": "required"}, "has no tools")


@case
def function_without_tools(api: Api) -> None:
    """A forced function with no tools at all."""
    expect_400(api, {"tool_choice": function_choice("get_weather")}, "has no tools")


@case
def unknown_function(api: Api) -> None:
    """A forced function that is not among the tools."""
    expect_400(api, {"tools": WEATHER_TOOLS, "tool_choice": function_choice("launch_rocket")},
               "not in tools")


@case
def bad_response_format_type(api: Api) -> None:
    """An unknown response_format type."""
    expect_400(api, {"response_format": {"type": "xml"}}, "response_format type must be")


@case
def json_schema_without_schema(api: Api) -> None:
    """json_schema without its schema."""
    expect_400(api, {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
               "needs json_schema.schema")


@case
def uncompilable_schema(api: Api) -> None:
    """A schema the grammar compiler cannot build: a dangling $ref."""
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}, "required": ["a"]}
    expect_400(api, {"response_format": json_schema_format("bad", schema)}, "cannot build the decoding grammar")


@case
def unknown_model(api: Api) -> None:
    """A model name the catalog does not have is a 404 on both endpoints, as OpenAI's."""
    model = "no-such-model:1b"

    def expect_404(status: int, body: dict[str, Any], where: str) -> None:
        expect(status == 404, f"{where}: expected HTTP 404, got {status}: {body}")
        err = body.get("error")
        expect(isinstance(err, dict), f"{where}: error body is not an object: {body}")
        expect(err.get("type") == "invalid_request_error", f"{where}: error type is {err.get('type')!r}")
        expect(err.get("param") == "model", f"{where}: error param is {err.get('param')!r}")
        expect(model in str(err.get("message")), f"{where}: message {err.get('message')!r} lacks {model!r}")

    body = api.body(user("hi"), max_tokens=8, model=model)
    resp = api.post(body)
    expect_404(resp.status, resp.body, "chat")
    s = api.stream(body)
    expect_404(s.status, s.error or {}, "streaming chat")
    expect(not s.chunks, "streaming chat: chunks were sent before the rejection")

    url = api.url.replace("/v1/chat/completions", "/v1/completions")
    req = urllib.request.Request(url, data=json.dumps({"model": model, "prompt": "hi", "max_tokens": 8}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=api.timeout) as r:
            status, reply = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, reply = e.code, json.loads(e.read())
    expect_404(status, reply, "completions")


@case
def null_options_are_unset(api: Api) -> None:
    """Options sent as null, as some clients do for unset ones, are not an error."""
    body = api.body(user("Say hi."), max_tokens=16, tools=WEATHER_TOOLS, tool_choice=None,
                    parallel_tool_calls=None, response_format=None, reasoning_effort=None,
                    presence_penalty=None, stream=None, max_completion_tokens=None)
    flm_api.check_message_contract(api.ok(body))


def stream_error(s: flm_api.Stream, where: str) -> dict[str, Any]:
    """A refused stream: an error status before any chunk, or an error event and then [DONE]."""
    if s.status >= 400:
        err = (s.error or {}).get("error")
        expect(isinstance(err, dict), f"{where}: error body is not an object: {s.error}")
        expect(not s.chunks, f"{where}: chunks were sent before the rejection")
        return err
    expect(s.status == 200 and s.error is None, f"{where}: HTTP {s.status}: {s.error}")
    errors = [c["error"] for c in s.chunks if "error" in c]
    expect(errors, f"{where}: HTTP 200 stream without an error event ({len(s.chunks)} chunks)")
    expect(isinstance(errors[0], dict), f"{where}: error event is not an object: {errors[0]}")
    expect(s.done, f"{where}: error event not followed by data: [DONE]")
    return errors[0]


def error_of(resp: flm_api.Response, where: str) -> dict[str, Any]:
    """The OpenAI error object of a refused non-streaming request, whose code is its status."""
    expect(resp.status >= 400, f"{where}: expected an error status, got {resp.status}: {str(resp.body)[:300]}")
    err = resp.body.get("error")
    expect(isinstance(err, dict), f"{where}: error body is not an object: {resp.body}")
    expect(err.get("code") == resp.status, f"{where}: error code {err.get('code')!r} is not the status {resp.status}")
    return err


def check_too_long(err: dict[str, Any], where: str) -> None:
    message = str(err.get("message") or "")
    expect(message, f"{where}: error without a message: {err}")
    expect("Max length reached" not in message, f"{where}: the fixed message, not the reason: {message!r}")
    expect("token" in message, f"{where}: message {message!r} does not say how long the prompt is")


def context_words(api: Api) -> int:
    """Words (about one token each) in a prompt surely longer than the model's context."""
    resp = api.post(api.body(user("Say hi."), max_tokens=8))
    usage = (resp.body.get("usage") if resp.status == 200 else None) or {}
    held = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    pct = usage.get("kv_token_occupancy_rate_percentage")
    if held and isinstance(pct, (int, float)) and pct > 0:
        return max(8192, int(1.25 * held * 100 / pct))  # a quarter over: enough, and the server logs every body
    return 1 << 18


@case
def prompt_longer_than_context(api: Api) -> None:
    """A prompt the context cannot hold gets an error status naming the problem, on both endpoints."""
    text = "hello " * context_words(api)
    body = api.body(user(text), max_tokens=8)
    check_too_long(error_of(api.post(body), "chat"), "chat")
    check_too_long(stream_error(api.stream(body), "streaming chat"), "streaming chat")
    completion = {"model": api.model, "prompt": text, "max_tokens": 8}
    url = api.completions_url()
    check_too_long(error_of(api.post(completion, url), "completions"), "completions")
    check_too_long(stream_error(api.stream(completion, url), "streaming completions"), "streaming completions")
    flm_api.check_message_contract(api.ok(api.body(user("Say hi."), max_tokens=16)))


FAULT_PROMPT = "FLM_TEST_PREFILL_FAULT: say hi."


@case
def prefill_fault(api: Api) -> None:
    """A prefill the engine fails is an HTTP 500 with an OpenAI error body, streamed or not, on both endpoints."""
    body = api.body(user(FAULT_PROMPT), max_tokens=8)
    resp = api.post(body)
    if resp.status == 200:
        # a 200 that carries an error is the bug this case is about, not a server without the hook
        expect(not (isinstance(resp.body, dict) and "error" in resp.body),
               f"chat: HTTP 200 with an error body: {str(resp.body)[:300]}")
        raise flm_api.Skip("the server was not started with FLM_TEST_PREFILL_FAULT")
    for where, r in (("chat", resp),
                     ("completions", api.post({"model": api.model, "prompt": FAULT_PROMPT, "max_tokens": 8},
                                              api.completions_url()))):
        err = error_of(r, where)
        expect(r.status == 500, f"{where}: expected HTTP 500, got {r.status}: {err}")
        expect(err.get("message"), f"{where}: error without a message: {err}")
        expect(err.get("type"), f"{where}: error without a type: {err}")
    for where, s in (("streaming chat", api.stream(body)),
                     ("streaming completions", api.stream({"model": api.model, "prompt": FAULT_PROMPT, "max_tokens": 8},
                                                          api.completions_url()))):
        err = stream_error(s, where)
        expect(s.status in (200, 500), f"{where}: expected HTTP 500, got {s.status}: {err}")
        expect(err.get("message"), f"{where}: error without a message: {err}")
    # the failed prefill must not leave anything behind for the next request
    flm_api.check_message_contract(api.ok(api.body(user("Say hi."), max_tokens=16)))
    flm_api.check_stream_contract(api.ok_stream(api.body(user("Say hi."), max_tokens=16)), include_usage=False)


@case
def server_still_serves(api: Api) -> None:
    """After the rejections a normal request still succeeds."""
    flm_api.check_message_contract(api.ok(api.body(user("Say hi."), max_tokens=16)))


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
