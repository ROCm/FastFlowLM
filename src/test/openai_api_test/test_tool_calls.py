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

The definitions a request offers are part of the prompt, so its
``prompt_tokens`` grow with them. A model whose prompt ignores them skips
that case, unless it is one of the packages checked to render them.

Whether a model *chooses* to call a tool under ``auto`` is up to the model,
so those cases check the shape only. Models whose chat format has no tool
call syntax skip everything that forces a call.

Usage::

    python3 test_tool_calls.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import copy
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
def function_unanchored_pattern(api: Api) -> None:
    """An unanchored string pattern only has to match somewhere in the string, as JSON Schema reads it."""
    tool = json.loads(json.dumps(GET_WEATHER))
    tool["function"]["parameters"]["properties"]["city"] = {
        "type": "string", "minLength": 1, "maxLength": 64, "pattern": "\\S"}
    choice = api.ok(api.body(user(WEATHER_PROMPT), tools=[tool], tool_choice=function_choice("get_weather")))
    city = json.loads(expect_calls(choice, [tool])[0]["function"]["arguments"])["city"]
    expect("paris" in city.lower(), f"the pattern \\S did not leave room for the city: {city!r}")


NOTE = ("The quarterly review moved to Thursday at three in the afternoon, "
        "in the large conference room on the fourth floor.")


@case
def function_pattern_keeps_length_bounds(api: Api) -> None:
    """A one-character pattern such as \\S still leaves maxLength in force."""
    tool = {"type": "function", "function": {
        "name": "save_note", "description": "Save a note",
        "parameters": {"type": "object", "required": ["text"], "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 20, "pattern": "\\S"}}}}}
    body = api.body(user(f"Save this note word for word: {NOTE}"), tools=[tool],
                    tool_choice=function_choice("save_note"))
    text = json.loads(expect_calls(api.ok(body), [tool])[0]["function"]["arguments"])["text"]
    expect(0 < len(text) <= 20 and not text.isspace(), f"text {text!r} is not 1..20 characters with a non-blank one")


@case
def function_nested_long_bounded_pattern(api: Api) -> None:
    """A \\S pattern with a long maxLength inside a nested object still gets a grammar."""
    caption = {"type": "string", "minLength": 1, "maxLength": 512, "pattern": "\\S"}
    tool = {"type": "function", "function": {
        "name": "draw_chart", "description": "Draw a chart",
        "parameters": {"type": "object", "required": ["spec"], "additionalProperties": False,
                       "properties": {"spec": {"type": "object", "required": ["caption"],
                                               "additionalProperties": False,
                                               "properties": {"caption": caption}}}}}}
    choice = api.ok(api.body(user("Draw a chart of monthly sales with a short caption."), tools=[tool],
                             tool_choice=function_choice("draw_chart")))
    spec = json.loads(expect_calls(choice, [tool])[0]["function"]["arguments"])["spec"]
    expect(0 < len(spec["caption"]) <= 512, f"caption length {len(spec['caption'])} is outside 1..512")


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


# ---------------------------------------------------------------- definitions

SUM_PROMPT = "What is two plus four?"
SHORT_DESCRIPTION = "Shelf inventory."
LONG_DESCRIPTION = SHORT_DESCRIPTION + " Amber birch cobalt dune elm frost." * 64
# Packages checked to render the tools they are offered: on these a prompt without the
# definitions is a failure. Any other model whose prompt ignores them skips (see the README).
SHOWN_DEFINITIONS = ("qwen2.5-it-oga-hipep:14b", "qwen3.5-oga-hipep:9b")


def described(description: str) -> list[dict]:
    """One tool the question does not need, with this description."""
    tool = copy.deepcopy(GET_WEATHER)
    tool["function"]["description"] = description
    return [tool]


def prompt_usage(api: Api, **fields) -> tuple[int, int | None]:
    """(prompt_tokens, cached_tokens) of SUM_PROMPT sent with these fields."""
    resp = api.post(api.body(user(SUM_PROMPT), max_tokens=1, **fields))
    if resp.status != 200:
        flm_api.raise_for_capability(resp.error_message())
        raise AssertionError(f"HTTP {resp.status}: {resp.error_message()}")
    usage = resp.body.get("usage") or {}
    expect(isinstance(usage.get("prompt_tokens"), int), f"no usage.prompt_tokens: {usage}")
    return usage["prompt_tokens"], (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


@case
def definitions_reach_the_prompt(api: Api) -> None:
    """Offered definitions are in the prompt: a longer description is a longer prompt; none shows nothing."""
    plain, _ = prompt_usage(api)
    short, _ = prompt_usage(api, tools=described(SHORT_DESCRIPTION))
    # The next prompt parts from the last one inside the definitions: reusing all of
    # the last one would be answering with the old description.
    changed, cached = prompt_usage(api, tools=described(SHORT_DESCRIPTION + " Checked daily."))
    long, _ = prompt_usage(api, tools=described(LONG_DESCRIPTION))
    hidden, _ = prompt_usage(api, tools=described(SHORT_DESCRIPTION), tool_choice="none")
    expect(hidden == plain,
           f"tool_choice none must not show the tools: {hidden} prompt tokens, {plain} without tools")
    if short == plain and long == plain:
        if api.model in SHOWN_DEFINITIONS:
            raise AssertionError(f"the definitions never reach the prompt: {plain} prompt tokens "
                                 "with no tool, a short description and a long one")
        raise Skip("the model's prompt does not carry tool definitions")
    expect(plain < short < changed < long,
           f"prompt tokens must grow with the definitions: none {plain}, short {short}, "
           f"a longer description {changed}, long {long}")
    expect(cached is None or cached < short,
           f"a changed definition reused the whole earlier prompt: {cached} of its {short} tokens cached")


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


def positional_shelf_history(api: Api) -> tuple[list[dict], list[dict]]:
    if api.model not in ("qwen3.5-oga-hipep:9b", "qwen2.5-it-oga-hipep:14b", "laguna2.1-oga-hipep:33b-a3b",
                         "laguna2.1-oga-hipep:118b-a8b", "nemotron3.5-lightning-oga-hipep:30b-a3b"):
        raise Skip("only the verified positional OGA packages use this adapter")
    turns = user("Read the red and blue shelves. Report each shelf and its numeric value as JSON.")
    calls = [{"id": ident, "type": "function", "function": {
        "name": "read_shelf", "arguments": json.dumps({"shelf": shelf})}}
        for ident, shelf in (("call_z", "red"), ("call_a", "blue"))]
    turns.append({"role": "assistant", "content": None, "tool_calls": calls})
    results = [{"role": "tool", "tool_call_id": ident, "content": value}
               for ident, value in (("call_z", "731"), ("call_a", "482"))]
    return turns, results


@case
def positional_parallel_results(api: Api) -> None:
    """Cold reversed delivery, repeated history, and a genuine ID-only remap, over JSON and SSE."""
    turns, results = positional_shelf_history(api)
    schema = {"type": "object", "properties": {k: {"type": "integer"} for k in ("red", "blue")},
              "required": ["red", "blue"], "additionalProperties": False}
    fmt = {"type": "json_schema", "json_schema": {"name": "shelves", "strict": True, "schema": schema}}
    remapped = [dict(r, tool_call_id=other["tool_call_id"]) for r, other in zip(results, reversed(results))]
    mismatches = []
    for stream in (False, True):
        for name, delivery, values in (("reversed", results[::-1], (731, 482)),
                                       ("ordered", results, (731, 482)),
                                       ("remapped", remapped, (482, 731))):
            body = api.body(turns + delivery, response_format=fmt, tool_choice="none", max_tokens=64)
            if stream:
                reply = api.stream(dict(body, stream_options={"include_usage": True}))
                expect(reply.status == 200 and reply.done, f"{name}: incomplete stream: {reply.error}")
                folded = flm_api.check_stream_contract(reply, include_usage=True)
                text, usage = folded.content, folded.usage
            else:
                reply = api.post(body)
                expect(reply.status == 200, f"{name}: HTTP {reply.status}: {reply.error_message()}")
                text = flm_api.check_message_contract(reply.body["choices"][0])["content"]
                usage = reply.body["usage"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            print(f"  {name}, stream={stream}: {cached}/{usage['prompt_tokens']} prompt tokens cached")
            try:
                mapping = json.loads(text)
            except (ValueError, TypeError):
                mapping = None
            if mapping != dict(zip(("red", "blue"), values)):
                mismatches.append(f"{name}, stream={stream}: expected red/blue={values}, got {text!r}")
    expect(not mismatches, "; ".join(mismatches))


@case
def positional_parallel_errors(api: Api) -> None:
    """Unresolvable parallel histories get indexed JSON errors, even when streaming was requested."""
    turns, (red, blue) = positional_shelf_history(api)
    calls = turns[1]["tool_calls"]
    for stream in (False, True):
        for messages, path in (
            (turns + [red, dict(blue, tool_call_id="other")], "messages[3].tool_call_id"),
            (turns + [red, red], "messages[3].tool_call_id"),
            (turns + [red, blue, red], "messages[4].tool_call_id"),
            (turns + [red, {"role": "tool", "content": "482"}], "messages[3].tool_call_id"),
            (turns + [red], "messages[1].tool_calls[1].id"),
            (turns + [red, *user("Next task"), blue], "messages[1].tool_calls[1].id"),
            ([turns[0], dict(turns[1], tool_calls=[calls[0], calls[0]]), red, blue],
             "messages[1].tool_calls[1].id"),
            (user("First message") + turns + [red, red], "messages[4].tool_call_id"),
        ):
            body = api.body(messages, tool_choice="none", max_tokens=1)
            if stream:
                s = api.stream(body)
                status, error = s.status, s.error or {}
                expect(not s.chunks and not s.done, "error arrived after SSE was committed")
            else:
                r = api.post(body)
                status, error = r.status, r.body
            expect(status == 400, f"{path}: expected HTTP 400, got {status}")
            detail = error.get("error", {})
            expect(detail.get("type") == "invalid_request_error" and detail.get("param") == path,
                   f"{path}: wrong error: {detail}")
    bad_part = dict(blue, content=[{"type": "image_url", "image_url": {"url": "invalid"}}])
    r = api.post(api.body(turns + [bad_part, red], tool_choice="none"))
    expect(r.status == 400 and "messages[2].content[0]" in r.error_message(),
           f"the content error must name the original result: {r.error_message()}")
    expect(api.post(api.body(turns + [blue, red], tool_choice="none", max_tokens=1)).status == 200,
           "valid request failed after the errors")


@case
def positional_single_round_trips(api: Api) -> None:
    """Replay a real named call using the ID returned on each wire format."""
    positional_shelf_history(api)
    for stream in (False, True):
        messages = user(WEATHER_PROMPT)
        body = api.body(messages, tools=WEATHER_TOOLS, tool_choice=function_choice("get_weather"))
        try:
            calls = (expect_streamed_calls(api, body, WEATHER_TOOLS) if stream
                     else expect_calls(api.ok(body), WEATHER_TOOLS))
        except Skip as error:
            raise AssertionError(f"this package must support named calls: {error}") from error
        expect(len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", f"unexpected calls: {calls}")
        messages += [{"role": "assistant", "content": None, "tool_calls": calls},
                     {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temperature_c": 23}'}]
        follow = api.body(messages, tools=WEATHER_TOOLS, tool_choice="none", max_tokens=128)
        try:
            text = (flm_api.check_stream_contract(api.ok_stream(follow), include_usage=False).content
                    if stream else flm_api.check_message_contract(api.ok(follow))["content"])
        except Skip as error:
            raise AssertionError(f"this package must accept the named call result: {error}") from error
        expect("23" in text, f"single-call result missing from the answer: {text!r}")


# The model families whose chat templates render reasoning_content on a tool call
# (Gemma 4, Qwen3 and later Qwen); elsewhere the reasoning is not part of the prompt.
REASONING_ON_TOOL_CALLS = ("gemma4", "qwen3")


@case
def reasoning_kept_within_turn(api: Api) -> None:
    """Reasoning sent back with a tool call stays in the prompt until the next user message."""
    if not api.model.startswith(REASONING_ON_TOOL_CALLS):
        raise Skip("the chat template does not render reasoning on a tool call")
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
        return api.ok_body(api.body(messages, tools=WEATHER_TOOLS, max_tokens=1))["usage"]["prompt_tokens"]

    without = prompt_tokens(False)
    kept = prompt_tokens(False, reasoning_content=reasoning)
    expect(kept > without, f"reasoning sent back with the tool call was dropped ({kept} vs {without} prompt tokens)")
    for fields in ({"reasoning": reasoning}, {"thinking": reasoning},
                   {"reasoning_content": None, "thinking": reasoning},
                   {"reasoning_content": "", "reasoning": reasoning}):
        expect(prompt_tokens(False, **fields) == kept, f"reasoning sent as {fields} was not kept")
    expect(prompt_tokens(True, reasoning_content=reasoning) == prompt_tokens(True),
           "reasoning from an earlier turn was fed back to the model")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
