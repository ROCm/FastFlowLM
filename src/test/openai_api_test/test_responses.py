#!/usr/bin/env python3
"""Wire checks for stateless POST /v1/responses. Run against a live FLM server."""

from __future__ import annotations

import urllib.error
import urllib.request

import flm_api
from flm_api import Api, case, expect


def responses_url(api: Api) -> str:
    return api.url.removesuffix("/chat/completions") + "/responses"


def expect_field_error(api: Api, field: str, value: object) -> None:
    reply = api.post({"model": api.model, "input": "Hi", field: value}, url=responses_url(api))
    expect(reply.status == 400, f"{field}: expected HTTP 400, got {reply.status}: {reply.body}")
    err = reply.body.get("error")
    expect(isinstance(err, dict) and err.get("param") == field,
           f"expected error.param={field!r}, got {reply.body}")


@case
def previous_response_id_is_refused(api: Api) -> None:
    expect_field_error(api, "previous_response_id", "resp_not_stored")


@case
def conversation_is_refused(api: Api) -> None:
    expect_field_error(api, "conversation", "conv_not_stored")


@case
def background_is_refused(api: Api) -> None:
    expect_field_error(api, "background", True)


@case
def rejects_mixed_tool_lists(api: Api) -> None:
    function = {"type": "function", "name": "read", "parameters": {"type": "object"}}
    for tools, field in (([function, {"type": "web_search"}], "tools[1].type"),
                         ([{"type": "namespace", "name": "group"}, function], "tools[0].type")):
        reply = api.post({"model": api.model, "input": "Hi", "tools": tools}, url=responses_url(api))
        err = reply.body.get("error")
        expect(reply.status == 400 and isinstance(err, dict) and err.get("param") == field,
               f"expected 400 naming {field}: {reply.status} {reply.body}")


@case
def unknown_model_is_not_a_route_miss(api: Api) -> None:
    reply = api.post({"model": "wp49-no-such-model:1b", "input": "Hi"}, url=responses_url(api))
    err = reply.body.get("error")
    expect(reply.status == 404 and isinstance(err, dict) and err.get("param") == "model",
           f"expected a routed 404 naming model, got {reply.status}: {reply.body}")


@case
def creates_a_response(api: Api) -> None:
    reply = api.post({"model": api.model, "input": "Reply with the word OK.",
                      "max_output_tokens": 32, "store": False}, url=responses_url(api))
    expect(reply.status == 200, f"expected HTTP 200, got {reply.status}: {reply.body}")
    obj = reply.body
    expect(obj.get("object") == "response" and "choices" not in obj, f"unexpected object: {obj}")
    expect(isinstance(obj.get("output"), list) and obj.get("status") in ("completed", "incomplete"),
           f"missing output or terminal status: {obj}")
    expect(isinstance(obj.get("usage"), dict) and obj["usage"].get("input_tokens", 0) > 0,
           f"missing input token usage: {obj}")


@case
def store_does_not_create_a_lookup_route(api: Api) -> None:
    root = responses_url(api)
    created = api.post({"model": api.model, "input": "Say OK.", "max_output_tokens": 8, "store": True}, url=root)
    expect(created.status == 200 and created.body.get("store") is False, f"not stateless: {created.body}")
    try:
        with urllib.request.urlopen(root + "/" + created.body["id"], timeout=api.timeout) as r:
            expect(False, f"unexpected retrieval: {r.status}")
    except urllib.error.HTTPError as e:
        expect(e.code == 404, f"lookup should be unrouted, got {e.code}")


@case
def stream_has_one_terminal_response(api: Api) -> None:
    stream = api.stream({"model": api.model, "input": "Say OK.", "max_output_tokens": 32},
                        url=responses_url(api))
    expect(stream.status == 200 and bool(stream.chunks),
           f"expected Responses SSE events, got HTTP {stream.status}: {stream.error}")
    types = [e.get("type") for e in stream.chunks]
    expect(types[0] == "response.created", f"unexpected first event: {types[0]}")
    terminals = [t for t in types if t in ("response.completed", "response.incomplete", "response.failed")]
    expect(terminals == [types[-1]] and types[-1] != "response.failed", f"bad terminal events: {types}")
    seq = [e.get("sequence_number") for e in stream.chunks]
    expect(all(type(n) is int for n in seq) and seq == sorted(set(seq)), f"bad sequence numbers: {seq}")
    first = stream.chunks[0].get("response") or {}
    last = stream.chunks[-1].get("response") or {}
    expect(first.get("id") and first["id"] == last.get("id") and last.get("object") == "response",
           f"response identity changed: {first.get('id')} to {last.get('id')}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
