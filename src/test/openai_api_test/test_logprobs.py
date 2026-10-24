#!/usr/bin/env python3
"""logprobs and top_logprobs on /v1/chat/completions, streamed and not.

Checks what a client of OpenAI's logprobs relies on:

* ``logprobs: true`` puts ``{"content": [...], "refusal": null}`` on the
  choice; without it ``logprobs`` stays null;
* every entry has ``token``, ``logprob`` <= 0, ``bytes`` and
  ``top_logprobs`` with exactly ``top_logprobs`` alternatives, most likely
  first; under greedy decoding the chosen token is the first of them;
* the entries cover the content and nothing else: their bytes put together
  are the content, so reasoning and tool call markup have no entries;
* when streaming, only content chunks carry logprobs, and the entries of all
  chunks are the same tokens the non-streaming response reports;
* malformed logprobs / top_logprobs get HTTP 400.

Models that cannot report logprobs (the MTP speculative model) answer 400
"logprobs is not supported", which turns the cases into skips.

Usage::

    python3 test_logprobs.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import WEATHER_TOOLS, function_choice, user

PROMPT = "In one short sentence, what is the capital of France?"
TOP = 3


def check_entry(entry: dict[str, Any], top_n: int, where: str) -> None:
    """One logprobs.content entry has OpenAI's shape."""
    expect(isinstance(entry.get("token"), str), f"{where}: token is {entry.get('token')!r}")
    lp = entry.get("logprob")
    expect(isinstance(lp, (int, float)) and lp <= 0.0, f"{where}: logprob is {lp!r}")
    data = entry.get("bytes")
    expect(isinstance(data, list) and all(isinstance(b, int) and 0 <= b <= 255 for b in data),
           f"{where}: bytes is {data!r}")
    top = entry.get("top_logprobs")
    expect(isinstance(top, list) and len(top) == top_n,
           f"{where}: expected {top_n} top_logprobs, got {top!r}")
    for i, alt in enumerate(top):
        expect(isinstance(alt.get("token"), str), f"{where}.top_logprobs[{i}]: token is {alt.get('token')!r}")
        expect(isinstance(alt.get("logprob"), (int, float)) and alt["logprob"] <= 0.0,
               f"{where}.top_logprobs[{i}]: logprob is {alt.get('logprob')!r}")
        expect(alt.get("bytes") is None or isinstance(alt["bytes"], list),
               f"{where}.top_logprobs[{i}]: bytes is {alt.get('bytes')!r}")
        expect("top_logprobs" not in alt, f"{where}.top_logprobs[{i}] nests top_logprobs")
    for i in range(1, len(top)):
        expect(top[i - 1]["logprob"] >= top[i]["logprob"], f"{where}: top_logprobs are not most likely first")
    expect(sum(2.718281828459045 ** alt["logprob"] for alt in top) <= 1.0 + 1e-3,
           f"{where}: top_logprobs add up to more than probability 1")


def check_greedy(entry: dict[str, Any], where: str) -> None:
    """Greedy decoding picks the most likely token, so it heads top_logprobs."""
    top = entry["top_logprobs"]
    expect(top[0]["token"] == entry["token"],
           f"{where}: greedy picked {entry['token']!r} but the most likely is {top[0]['token']!r}")
    expect(abs(top[0]["logprob"] - entry["logprob"]) < 1e-4,
           f"{where}: {entry['logprob']} differs from its own top_logprobs entry {top[0]['logprob']}")


def joined(entries: list[dict[str, Any]]) -> str:
    """The text the entries' bytes spell."""
    return b"".join(bytes(e["bytes"]) for e in entries).decode("utf-8", "replace")


def expect_covers(entries: list[dict[str, Any]], content: str | None, where: str) -> None:
    """The entries are the content's tokens: no more (reasoning, markup), no fewer."""
    text = joined(entries)
    expect(text.strip() == (content or "").strip(),
           f"{where}: the logprobs tokens spell {text!r}, the content is {content!r}")
    flm_api.assert_no_markers(text, f"{where} logprobs tokens")


def logprobs_body(api: Api, messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """A request asking for logprobs, penalties off.

    logprobs are the raw distribution, before penalties; pinning them off keeps
    greedy's pick the raw distribution's most likely token, which check_greedy relies on.
    """
    fields.setdefault("max_tokens", 48)
    fields.setdefault("logprobs", True)
    fields.setdefault("top_logprobs", TOP)
    fields.setdefault("repetition_penalty", 1.0)
    fields.setdefault("presence_penalty", 0.0)
    fields.setdefault("frequency_penalty", 0.0)
    return api.body(messages, **fields)


def nonstream_entries(api: Api, body: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    choice = api.ok(body)
    msg = flm_api.check_message_contract(choice)
    lp = choice.get("logprobs")
    expect(isinstance(lp, dict), f"logprobs was asked for, got {lp!r}")
    expect(lp.get("refusal") is None, f"logprobs.refusal is {lp.get('refusal')!r}")
    expect(isinstance(lp.get("content"), list), f"logprobs.content is {lp.get('content')!r}")
    return msg, lp["content"]


def stream_entries(api: Api, body: dict[str, Any]) -> tuple[flm_api.Folded, list[dict[str, Any]]]:
    s = api.ok_stream(body)
    folded = flm_api.check_stream_contract(s, include_usage=False)
    entries: list[dict[str, Any]] = []
    for n, chunk in enumerate(s.chunks):
        for choice in chunk.get("choices") or []:
            expect("logprobs" in choice, f"chunk {n}: logprobs was asked for but the choice has no logprobs key")
            delta = choice.get("delta") or {}
            lp = choice["logprobs"]
            if delta.get("content"):
                expect(isinstance(lp, dict) and isinstance(lp.get("content"), list),
                       f"chunk {n}: a content chunk has logprobs {lp!r}")
                entries.extend(lp["content"])
            else:
                expect(lp is None, f"chunk {n}: a chunk without content has logprobs {lp!r}")
    return folded, entries


@case
def nonstream_logprobs(api: Api) -> None:
    """Every content token has an entry with top_logprobs, and they spell the content."""
    msg, entries = nonstream_entries(api, logprobs_body(api, user(PROMPT)))
    expect(len(entries) > 0, "no logprobs entries for a non-empty answer")
    for i, entry in enumerate(entries):
        check_entry(entry, TOP, f"content[{i}]")
        check_greedy(entry, f"content[{i}]")
    expect_covers(entries, msg["content"], "non-stream")


@case
def stream_logprobs(api: Api) -> None:
    """Content chunks carry their tokens' logprobs, the others carry null."""
    folded, entries = stream_entries(api, logprobs_body(api, user(PROMPT)))
    expect(len(entries) > 0, "no logprobs entries in the stream")
    for i, entry in enumerate(entries):
        check_entry(entry, TOP, f"streamed content[{i}]")
        check_greedy(entry, f"streamed content[{i}]")
    expect_covers(entries, folded.content, "stream")


@case
def stream_matches_nonstream(api: Api) -> None:
    """Greedy streaming and non-streaming report the same tokens."""
    body = logprobs_body(api, user("Name three primary colors, comma separated."))
    plain: list[list[str]] = []
    streamed: list[list[str]] = []
    for _ in range(3):
        plain.append([e["token"] for e in nonstream_entries(api, body)[1]])
        streamed.append([e["token"] for e in stream_entries(api, body)[1]])
        if any(s == p for s in streamed for p in plain):
            return
    if any(p != plain[0] for p in plain):
        raise Skip("greedy decoding did not reproduce over 3 identical requests")
    raise AssertionError(f"stream and non-stream differ:\n  stream:     {streamed[0]!r}\n"
                         f"  non-stream: {plain[0]!r}")


@case
def top_logprobs_default(api: Api) -> None:
    """logprobs without top_logprobs gives every entry an empty top_logprobs."""
    body = logprobs_body(api, user(PROMPT), max_tokens=16)
    del body["top_logprobs"]
    _, entries = nonstream_entries(api, body)
    expect(len(entries) > 0, "no logprobs entries")
    for i, entry in enumerate(entries):
        check_entry(entry, 0, f"content[{i}]")


@case
def off_by_default(api: Api) -> None:
    """Without logprobs the choice's logprobs is null and chunks carry none."""
    body = api.body(user(PROMPT), max_tokens=16)
    choice = api.ok(body)
    expect(choice.get("logprobs") is None, f"logprobs was not asked for, got {choice.get('logprobs')!r}")
    s = api.ok_stream(body)
    for n, chunk in enumerate(s.chunks):
        for c in chunk.get("choices") or []:
            expect(c.get("logprobs") is None, f"chunk {n}: logprobs was not asked for, got {c.get('logprobs')!r}")


@case
def reasoning_has_no_logprobs(api: Api) -> None:
    """With thinking on, the reasoning's tokens are left out of logprobs.content."""
    body = logprobs_body(api, user("What is 17 + 25? Answer with the number."), think=True, max_tokens=512)
    msg, entries = nonstream_entries(api, body)
    if not msg.get("reasoning_content"):
        raise Skip("no reasoning_content with think=true: the model does not reason")
    if msg.get("content") is None or not msg["content"].strip():
        raise Skip("the model used up max_tokens reasoning")
    expect_covers(entries, msg["content"], "non-stream with reasoning")
    folded, streamed = stream_entries(api, body)
    if folded.content.strip():
        expect_covers(streamed, folded.content, "stream with reasoning")


@case
def tool_call_has_no_logprobs(api: Api) -> None:
    """A forced tool call is not content, so its tokens have no entries."""
    body = logprobs_body(api, user("What's the weather in Paris?"), tools=WEATHER_TOOLS,
                         tool_choice=function_choice("get_weather"), max_tokens=128)
    msg, entries = nonstream_entries(api, body)
    expect(bool(msg.get("tool_calls")), "the forced call is missing")
    expect_covers(entries, msg.get("content"), "non-stream tool call")
    folded, streamed = stream_entries(api, body)
    expect_covers(streamed, folded.content, "stream tool call")


def expect_400(api: Api, fields: dict[str, Any], message_part: str) -> None:
    """Non-streaming and streaming both refuse, naming the problem."""
    body = api.body(user("hi"), max_tokens=8, **fields)
    resp = api.post(body)
    expect(resp.status == 400, f"expected HTTP 400, got {resp.status}: {resp.body}")
    err = resp.body.get("error")
    expect(isinstance(err, dict) and err.get("type") == "invalid_request_error",
           f"not an invalid_request_error: {resp.body}")
    expect(message_part in str(err.get("message")), f"message {err.get('message')!r} lacks {message_part!r}")
    s = api.stream(body)
    expect(s.status == 400, f"streaming: expected HTTP 400, got {s.status}")
    expect(not s.chunks, "streaming: chunks were sent before the rejection")


@case
def top_logprobs_needs_logprobs(api: Api) -> None:
    """top_logprobs without logprobs: true is refused."""
    expect_400(api, {"top_logprobs": 2}, "requires logprobs")
    expect_400(api, {"logprobs": False, "top_logprobs": 2}, "requires logprobs")


@case
def bad_top_logprobs(api: Api) -> None:
    """top_logprobs outside 0..20, or not an integer, is refused."""
    expect_400(api, {"logprobs": True, "top_logprobs": 21}, "between 0 and 20")
    expect_400(api, {"logprobs": True, "top_logprobs": -1}, "between 0 and 20")
    expect_400(api, {"logprobs": True, "top_logprobs": 1.5}, "between 0 and 20")


@case
def bad_logprobs(api: Api) -> None:
    """A logprobs that is not a boolean is refused."""
    expect_400(api, {"logprobs": "yes"}, "must be a boolean")


@case
def null_logprobs_is_unset(api: Api) -> None:
    """logprobs and top_logprobs sent as null mean not asked for."""
    choice = api.ok(api.body(user("Say hi."), max_tokens=8, logprobs=None, top_logprobs=None))
    expect(choice.get("logprobs") is None, f"logprobs was not asked for, got {choice.get('logprobs')!r}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
