#!/usr/bin/env python3
"""Tool calling: parsing, streaming deltas and every tool_choice mode.

What the server guarantees, and so what is checked, per mode:

* ``auto``      calls, if any, name a declared tool and follow its schema
* ``required``  at least one such call
* function      exactly one call, to that function, and no content
* ``none``      no calls at all; the tools are not even shown to the model
* ``parallel_tool_calls: false``  at most one call

Every call must have an OpenAI-style id, a JSON-string ``arguments`` that
validates against the tool's ``parameters``, and ``finish_reason``
``tool_calls``. Streamed calls must open with id/type/name and only
append arguments after that.

Whether a model *chooses* to call a tool under ``auto`` is up to the model,
so those cases check the shape only. Models whose chat format has no tool
call syntax skip everything that forces a call.

Usage::

    python3 test_tool_calls.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import BOOK_FLIGHT, GET_WEATHER, WEATHER_TOOLS, function_choice, user

WEATHER_PROMPT = "What's the weather in Paris right now?"
TWO_CITIES_PROMPT = "What's the weather in Paris and in Tokyo?"


def expect_calls(choice: dict, tools: list[dict], *, at_least: int = 1) -> list[dict]:
    """A finished turn with valid calls; returns them."""
    msg = flm_api.check_message_contract(choice)
    expect(choice["finish_reason"] != "length", "ran out of max_tokens inside a tool call")
    calls = msg.get("tool_calls") or []
    expect(len(calls) >= at_least, f"expected at least {at_least} tool call(s), got {len(calls)}: {msg}")
    flm_api.check_tool_calls(calls, tools)
    expect(choice["finish_reason"] == "tool_calls", f"finish_reason is {choice['finish_reason']!r}")
    return calls


def expect_streamed_calls(api: Api, body: dict, tools: list[dict], *, at_least: int = 1) -> list[dict]:
    """The streamed version of expect_calls."""
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    expect(folded.finish_reason != "length", "ran out of max_tokens inside a tool call")
    expect(len(folded.tool_calls) >= at_least,
           f"expected at least {at_least} streamed tool call(s), got {len(folded.tool_calls)}")
    flm_api.check_tool_calls(folded.tool_calls, tools)
    expect(folded.finish_reason == "tool_calls", f"finish_reason is {folded.finish_reason!r}")
    return folded.tool_calls


# ---------------------------------------------------------------- auto


@case
def auto_nonstream(api: Api) -> None:
    """tool_choice auto: any calls the model makes are valid."""
    choice = api.ok(api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS))
    msg = flm_api.check_message_contract(choice)
    if msg.get("tool_calls"):
        expect_calls(choice, WEATHER_TOOLS)
    else:
        expect(choice["finish_reason"] in ("stop", "length"), f"finish_reason {choice['finish_reason']!r}")


@case
def auto_stream(api: Api) -> None:
    """tool_choice auto, streamed: any calls are well formed and valid."""
    folded = flm_api.check_stream_contract(
        api.ok_stream(api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS)), include_usage=False)
    if folded.tool_calls and folded.finish_reason != "length":
        flm_api.check_tool_calls(folded.tool_calls, WEATHER_TOOLS)
        expect(folded.finish_reason == "tool_calls", f"finish_reason {folded.finish_reason!r}")


@case
def no_tools_no_calls(api: Api) -> None:
    """Without tools in the request nothing is parsed as a tool call."""
    msg = flm_api.check_message_contract(api.ok(api.body(user(WEATHER_PROMPT), max_tokens=96)))
    expect("tool_calls" not in msg, f"a request without tools returned tool_calls: {msg}")


# ---------------------------------------------------------------- required


@case
def required_nonstream(api: Api) -> None:
    """tool_choice required: at least one valid call."""
    expect_calls(api.ok(api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice="required")),
                 WEATHER_TOOLS)


@case
def required_stream(api: Api) -> None:
    """tool_choice required, streamed: at least one valid call."""
    expect_streamed_calls(api, api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice="required"),
                          WEATHER_TOOLS)


@case
def required_even_when_off_topic(api: Api) -> None:
    """required forces a call even when the prompt does not need a tool."""
    expect_calls(api.ok(api.body(user("Tell me a joke about cats."), tools=WEATHER_TOOLS, tool_choice="required")),
                 WEATHER_TOOLS)


@case
def required_single_call(api: Api) -> None:
    """required + parallel_tool_calls=false: exactly one call, even when two are asked for."""
    body = api.body(user(TWO_CITIES_PROMPT), tools=WEATHER_TOOLS, tool_choice="required",
                    parallel_tool_calls=False)
    calls = expect_calls(api.ok(body), WEATHER_TOOLS)
    expect(len(calls) == 1, f"parallel_tool_calls=false allowed {len(calls)} calls")
    calls = expect_streamed_calls(api, body, WEATHER_TOOLS)
    expect(len(calls) == 1, f"parallel_tool_calls=false allowed {len(calls)} streamed calls")


# ---------------------------------------------------------------- forced function


@case
def function_nonstream(api: Api) -> None:
    """Forcing a function returns exactly one call to it and no content."""
    body = api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice=function_choice("get_time"))
    choice = api.ok(body)
    calls = expect_calls(choice, WEATHER_TOOLS)
    expect(len(calls) == 1, f"a forced function produced {len(calls)} calls")
    expect(calls[0]["function"]["name"] == "get_time", f"forced get_time, got {calls[0]['function']['name']}")
    expect(choice["message"].get("content") is None,
           f"a forced call must leave content null, got {choice['message'].get('content')!r}")


@case
def function_stream(api: Api) -> None:
    """Forcing a function, streamed: one call to it, arguments sent as growing JSON."""
    body = api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice=function_choice("get_time"))
    s = api.ok_stream(body)
    folded = flm_api.check_stream_contract(s, include_usage=False)
    expect(folded.content == "", f"a forced call streamed content: {folded.content!r}")
    expect([c["function"]["name"] for c in folded.tool_calls] == ["get_time"],
           f"expected one get_time call, got {folded.tool_calls}")
    flm_api.check_tool_calls(folded.tool_calls, WEATHER_TOOLS)
    # each prefix of the streamed arguments is a prefix of the final JSON
    pieces = [tc["function"].get("arguments", "")
              for chunk in s.chunks for choice in chunk.get("choices") or []
              for tc in (choice.get("delta") or {}).get("tool_calls") or []]
    expect("".join(pieces) == folded.tool_calls[0]["function"]["arguments"], "argument pieces do not add up")


@case
def function_typed_arguments(api: Api) -> None:
    """Integers, numbers, booleans, enums and typed arrays all follow the schema."""
    prompt = ("Book 2 passengers from Paris to Rome in business class, refundable, "
              "at most 450.50 euros, one vegan and one standard meal.")
    choice = api.ok(api.body(user(prompt), tools=[BOOK_FLIGHT], tool_choice=function_choice("book_flight")))
    calls = expect_calls(choice, [BOOK_FLIGHT])
    args = json.loads(calls[0]["function"]["arguments"])
    expect(isinstance(args["passengers"], int) and not isinstance(args["passengers"], bool),
           f"passengers is not an integer: {args['passengers']!r}")
    expect(isinstance(args["refundable"], bool), f"refundable is not a boolean: {args['refundable']!r}")


@case
def function_with_thinking(api: Api) -> None:
    """A forced call still works when the model thinks first."""
    body = api.body(user(WEATHER_PROMPT), think=True, max_tokens=1536, tools=WEATHER_TOOLS,
                    tool_choice=function_choice("get_weather"))
    calls = expect_calls(api.ok(body), WEATHER_TOOLS)
    expect(calls[0]["function"]["name"] == "get_weather", "wrong function called")


@case
def non_strict_tool(api: Api) -> None:
    """strict=false only requires well-formed JSON arguments."""
    loose = json.loads(json.dumps(GET_WEATHER))
    loose["function"]["strict"] = False
    loose["function"]["parameters"]["properties"]["city"] = {"type": "integer"}
    choice = api.ok(api.body(user(WEATHER_PROMPT), tools=[loose], tool_choice="required"))
    msg = flm_api.check_message_contract(choice)
    calls = msg.get("tool_calls") or []
    expect(len(calls) >= 1, "required produced no call")
    flm_api.parse_arguments(calls[0])  # valid JSON; the schema is not enforced


# ---------------------------------------------------------------- none


@case
def none_nonstream(api: Api) -> None:
    """tool_choice none: plain text, no calls, no tool markup."""
    choice = api.ok(api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice="none", max_tokens=96))
    msg = flm_api.check_message_contract(choice)
    expect("tool_calls" not in msg, f"tool_choice none returned tool calls: {msg}")
    expect(choice["finish_reason"] in ("stop", "length"), f"finish_reason {choice['finish_reason']!r}")


@case
def none_stream(api: Api) -> None:
    """tool_choice none, streamed: no tool call deltas."""
    s = api.ok_stream(api.body(user(WEATHER_PROMPT), tools=WEATHER_TOOLS, tool_choice="none", max_tokens=96))
    folded = flm_api.check_stream_contract(s, include_usage=False)
    expect(folded.tool_calls == [], f"tool_choice none streamed tool calls: {folded.tool_calls}")


# ---------------------------------------------------------------- multi turn


@case
def round_trip(api: Api) -> None:
    """A call, its tool result, then a final answer that uses the result."""
    messages = user(WEATHER_PROMPT)
    first = api.ok(api.body(messages, tools=WEATHER_TOOLS, tool_choice=function_choice("get_weather")))
    call = expect_calls(first, WEATHER_TOOLS)[0]
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call["id"], "type": "function", "function": call["function"]}]},
        {"role": "tool", "tool_call_id": call["id"],
         "content": json.dumps({"temperature_c": 23, "sky": "sunny"})},
    ]
    resp = api.post(api.body(messages, tools=WEATHER_TOOLS, max_tokens=128))
    if resp.status != 200:
        flm_api.raise_for_capability(resp.error_message())
        raise AssertionError(f"HTTP {resp.status}: {resp.error_message()}")
    choice = resp.body["choices"][0]
    msg = flm_api.check_message_contract(choice)
    if msg.get("tool_calls"):
        raise Skip("the model called another tool instead of answering; nothing to check")
    expect("23" in (msg["content"] or ""), f"the answer does not use the tool result: {msg['content']!r}")



@case
def reasoning_kept_within_turn(api: Api) -> None:
    """Reasoning sent back with a tool call stays in the prompt until the next user message."""
    call = {"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}
    reasoning = "The user wants the weather in Paris, so I call get_weather for Paris."

    def prompt_tokens(follow_up: bool, **fields) -> int:
        messages = user(WEATHER_PROMPT) + [
            {"role": "assistant", "content": None, "tool_calls": [call], **fields},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temperature_c": 23})},
        ]
        if follow_up:
            messages += [{"role": "assistant", "content": "It is 23 degrees."}] + user("And tomorrow?")
        resp = api.post(api.body(messages, tools=WEATHER_TOOLS, max_tokens=1))
        expect(resp.status == 200, f"HTTP {resp.status}: {resp.error_message()}")
        return resp.body["usage"]["prompt_tokens"]

    without = prompt_tokens(False)
    kept = prompt_tokens(False, reasoning_content=reasoning)
    if kept == without:
        raise Skip("the chat template does not render reasoning on a tool call")
    for fields in ({"reasoning": reasoning}, {"thinking": reasoning}, {"reasoning_content": None, "thinking": reasoning}):
        expect(prompt_tokens(False, **fields) == kept, f"reasoning sent as {list(fields)} was not kept")
    expect(prompt_tokens(True, reasoning_content=reasoning) == prompt_tokens(True),
           "reasoning from an earlier turn was fed back to the model")

CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
