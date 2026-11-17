#!/usr/bin/env python3
"""response_format: JSON mode and JSON schema, streamed and not.

The grammar guarantees the shape of ``content`` whatever the model, so
every finished answer (``finish_reason`` ``stop``) must parse as JSON and
validate against the request's schema. Covered:

* ``json_object`` gives a JSON object;
* ``json_schema`` with flat properties, a string enum, Pydantic style
  ``$defs``/``$ref`` nesting, and const / anyOf / bounds / pattern;
* the same through a stream;
* with thinking on, reasoning stays free text and only content is JSON;
* with thinking off, there is no reasoning at all;
* ``text`` is the same as sending nothing.

Usage::

    python3 test_response_format.py <model> [--always-thinks] [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
from typing import Any

import flm_api
from flm_api import Api, case, expect
from fixtures import (
    ORDER_SCHEMA,
    PERSON_SCHEMA,
    REVIEW_SCHEMA,
    SENTIMENT_SCHEMA,
    WEATHER_TOOLS,
    json_schema_format,
    user,
)


def expect_json(content: str | None, finish_reason: str | None, schema: Any = None) -> Any:
    """content is complete JSON matching schema; returns the decoded value."""
    expect(finish_reason == "stop", f"finish_reason is {finish_reason!r}, the JSON may be cut off")
    expect(isinstance(content, str), f"content is {content!r}")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as e:
        raise AssertionError(f"content is not valid JSON ({e}): {content!r}") from None
    if schema is not None:
        flm_api.validate(value, schema, "content")
    return value


def nonstream_json(api: Api, body: dict, schema: Any) -> tuple[dict, Any]:
    choice = api.ok(body)
    msg = flm_api.check_message_contract(choice)
    expect("tool_calls" not in msg, "a response_format answer has tool_calls")
    return msg, expect_json(msg["content"], choice["finish_reason"], schema)


# ---------------------------------------------------------------- json_object


@case
def json_object_nonstream(api: Api) -> None:
    """json_object returns a JSON object."""
    body = api.body(user("Describe the color red as JSON with keys name and hex."),
                    response_format={"type": "json_object"})
    _, value = nonstream_json(api, body, {"type": "object"})
    expect(isinstance(value, dict), f"json_object returned {type(value).__name__}")


@case
def json_object_stream(api: Api) -> None:
    """json_object, streamed: the folded content is a JSON object."""
    body = api.body(user("Describe the color red as JSON with keys name and hex."),
                    response_format={"type": "json_object"})
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    expect_json(folded.content, folded.finish_reason, {"type": "object"})


# ---------------------------------------------------------------- json_schema


@case
def schema_flat(api: Api) -> None:
    """A flat schema with required string and integer fields."""
    body = api.body(user("Extract the person: Alice is 31 years old and lives in Berlin."),
                    response_format=json_schema_format("person", PERSON_SCHEMA))
    nonstream_json(api, body, PERSON_SCHEMA)


@case
def schema_enum(api: Api) -> None:
    """A string enum only lets the listed labels through."""
    body = api.body(user("Classify the sentiment of: 'I loved this movie!'"),
                    response_format=json_schema_format("sentiment", SENTIMENT_SCHEMA), max_tokens=48)
    nonstream_json(api, body, SENTIMENT_SCHEMA)


@case
def schema_nested_refs(api: Api) -> None:
    """Pydantic style $defs/$ref nesting, a list of objects."""
    body = api.body(user("Bob orders 2 apples and 3 pears. Extract the order."),
                    response_format=json_schema_format("Order", ORDER_SCHEMA))
    nonstream_json(api, body, ORDER_SCHEMA)


@case
def schema_keywords(api: Api) -> None:
    """const, anyOf, integer bounds and a string pattern."""
    body = api.body(user("Review a margherita pizza eaten on 2026-03-14: score it 1-5 and say if it is vegetarian."),
                    response_format=json_schema_format("review", REVIEW_SCHEMA))
    nonstream_json(api, body, REVIEW_SCHEMA)


@case
def schema_unanchored_pattern(api: Api) -> None:
    """An unanchored pattern only has to match somewhere in the string."""
    schema = {"type": "object", "properties": {"city": {"type": "string", "pattern": "\\S"}},
              "required": ["city"], "additionalProperties": False}
    body = api.body(user("Which city is the capital of France? Answer as JSON."),
                    response_format=json_schema_format("city", schema), max_tokens=64)
    _, value = nonstream_json(api, body, schema)
    expect("paris" in value["city"].lower(), f"the pattern \\S did not leave room for the city: {value['city']!r}")


@case
def schema_stream(api: Api) -> None:
    """json_schema, streamed: the folded content validates."""
    body = api.body(user("Bob orders 2 apples and 3 pears. Extract the order."),
                    response_format=json_schema_format("Order", ORDER_SCHEMA))
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    expect(folded.tool_calls == [], "a response_format answer streamed tool calls")
    expect_json(folded.content, folded.finish_reason, ORDER_SCHEMA)


@case
def schema_with_thinking(api: Api) -> None:
    """Thinking stays free text; only content is held to the schema."""
    body = api.body(user("Extract the person: Alice is 31 years old and lives in Berlin."),
                    think=True, max_tokens=2048,
                    response_format=json_schema_format("person", PERSON_SCHEMA))
    msg, _ = nonstream_json(api, body, PERSON_SCHEMA)
    flm_api.assert_no_markers(msg.get("reasoning_content"), "reasoning_content")


@case
def schema_without_thinking(api: Api) -> None:
    """With thinking off the grammar leaves no room for a reasoning block."""
    body = api.body(user("Extract the person: Alice is 31 years old and lives in Berlin."),
                    think=False, response_format=json_schema_format("person", PERSON_SCHEMA))
    msg, _ = nonstream_json(api, body, PERSON_SCHEMA)
    if not api.always_thinks:
        expect("reasoning_content" not in msg, "think=false with a schema still produced reasoning_content")


@case
def schema_beats_auto_tools(api: Api) -> None:
    """With tool_choice auto, response_format wins: JSON content and no calls."""
    body = api.body(user("Extract the person: Alice is 31 years old and lives in Berlin."),
                    tools=WEATHER_TOOLS, response_format=json_schema_format("person", PERSON_SCHEMA))
    nonstream_json(api, body, PERSON_SCHEMA)


@case
def text_format(api: Api) -> None:
    """response_format text is plain, unconstrained output."""
    choice = api.ok(api.body(user("Say hello."), response_format={"type": "text"}, max_tokens=32))
    flm_api.check_message_contract(choice)


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
