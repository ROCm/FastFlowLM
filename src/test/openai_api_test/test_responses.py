#!/usr/bin/env python3
"""Wire checks for stateless POST /v1/responses. Run against a live FLM server."""

from __future__ import annotations

import json
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
    expect_field_error(api, "previous_response_id", None)


@case
def conversation_is_refused(api: Api) -> None:
    expect_field_error(api, "conversation", "conv_not_stored")
    expect_field_error(api, "conversation", None)


@case
def background_is_refused(api: Api) -> None:
    expect_field_error(api, "background", True)
    expect_field_error(api, "background", False)


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
def nullable_function_strict(api: Api) -> None:
    tool = {"type": "function", "name": "read_marker", "parameters": {"type": "object"}, "strict": None}
    body = {"model": api.model, "input": "Say OK.", "tools": [tool], "max_output_tokens": 4}
    reply = api.post(body, url=responses_url(api))
    expect(reply.status == 200 and reply.body.get("object") == "response",
           f"nullable strict: {reply.status} {reply.body}")
    tool["strict"] = "true"
    reply = api.post(body, url=responses_url(api))
    expect(reply.status == 400 and reply.body.get("error", {}).get("param") == "tools[0].strict",
           f"invalid strict: {reply.status} {reply.body}")


@case
def unfinished_parallel_group_rejects_new_call(api: Api) -> None:
    def call(i):
        return {"type":"function_call", "name":"read", "call_id":i, "arguments":"{}"}
    def result(i):
        return {"type":"function_call_output", "call_id":i, "output":"ok"}
    history = [call("A"), call("B"), result("A"), call("C"), result("B"), result("C")]
    reply = api.post({"model":api.model,"input":history}, url=responses_url(api))
    expect(reply.status == 400 and reply.body.get("error", {}).get("param") == "input[3].type",
           f"incorrect tool group: {reply.status} {reply.body}")



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
    expect_field_error(api, "previous_response_id", created.body["id"])
    try:
        with urllib.request.urlopen(root + "/" + created.body["id"], timeout=api.timeout) as r:
            expect(False, f"unexpected retrieval: {r.status}")
    except urllib.error.HTTPError as e:
        expect(e.code == 404, f"lookup should be unrouted, got {e.code}")


@case
def stream_has_one_terminal_response(api: Api) -> None:
    body = {"model": api.model, "input": "Count from 1 to 100.", "max_output_tokens": 1, "stream": True}
    req = urllib.request.Request(responses_url(api), data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=api.timeout) as r:
        expect(r.headers.get_content_type() == "text/event-stream", "wrong stream media type")
        frames = r.read().decode().strip().split("\n\n")
    events = []
    for frame in frames:
        lines = frame.splitlines()
        expect(len(lines) == 2 and lines[1].startswith("data: "), f"not a named SSE frame: {frame}")
        event = json.loads(lines[1][6:])
        expect(lines[0] == "event: " + event["type"], "SSE name disagrees with payload")
        events.append(event)
    types = [e.get("type") for e in events]
    expect(types[0] == "response.created", f"unexpected first event: {types[0]}")
    terminals = [t for t in types if t in ("response.completed", "response.incomplete", "response.failed")]
    expect(terminals == [types[-1]] and types[-1] == "response.incomplete", f"bad terminal events: {types}")
    seq = [e.get("sequence_number") for e in events]
    expect(seq == list(range(len(seq))), f"bad sequence numbers: {seq}")
    first = events[0].get("response") or {}
    last = events[-1].get("response") or {}
    expect(last.get("incomplete_details") == {"reason":"max_output_tokens"}, f"wrong limit reason: {last}")
    expect(first.get("id") and first["id"] == last.get("id") and last.get("object") == "response",
           f"response identity changed: {first.get('id')} to {last.get('id')}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
