#!/usr/bin/env python3
"""Ollama routes: a thinking model's reasoning goes to ``thinking``, not into the answer.

* ``/api/chat`` and ``/api/generate`` with ``think: true`` return the reasoning in
  ``message.thinking`` / ``thinking`` and the answer without ``<think>`` tags.
* The same holds when streamed, and the streamed text folds to the non-stream result.
* An empty ``thinking`` is left out of every packet (Ollama's ``omitempty``).
* ``think: false`` gives no ``thinking`` and no tags.
* ``/v1/chat/completions`` on the same request is unchanged.

Models that never reason skip the thinking-on cases. Raw HTTP, no Ollama client needed.

Usage::

    python3 test_ollama_thinking.py <model> [--always-thinks] [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import user

PROMPT = "Is 7 a prime number? Answer with yes or no."
THINK_BUDGET = 1536
TAGS = ("<think>", "</think>")
# sampling pinned as the /v1 tests do, so two requests for the same text give the same text
PINS = {"temperature": 0.0, "top_k": 1, "repeat_last_n": 64, "seed": 1}


def ollama_url(api: Api, path: str) -> str:
    return api.url.replace("/v1/chat/completions", path)


def post(api: Api, path: str, body: dict[str, Any], stream: bool) -> list[dict[str, Any]]:
    """POST to an Ollama route; return the JSON objects of the reply (one if not streamed)."""
    body = dict(body, stream=stream)
    req = urllib.request.Request(ollama_url(api, path), data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=api.timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise AssertionError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    parts = [json.loads(line) for line in raw.splitlines() if line.strip()]
    expect(bool(parts), "empty reply")
    for part in parts:
        expect("error" not in part, f"error in reply: {part.get('error')}")
    return parts


def options(api: Api) -> dict[str, Any]:
    budget = THINK_BUDGET + (api.reasoning_budget if api.always_thinks else 0)
    return {"temperature": 0, "top_k": 1, "repeat_last_n": 64, "seed": 1, "num_predict": budget}


def chat(api: Api, stream: bool, **fields: Any) -> list[dict[str, Any]]:
    body = {"model": api.model, "messages": user(PROMPT), "options": options(api), **PINS, **fields}
    return post(api, "/api/chat", body, stream)


def generate(api: Api, stream: bool, **fields: Any) -> list[dict[str, Any]]:
    body = {"model": api.model, "prompt": PROMPT, "options": options(api), **PINS, **fields}
    return post(api, "/api/generate", body, stream)


def fold(parts: list[dict[str, Any]], is_chat: bool) -> tuple[str, str]:
    """(answer, thinking) of a reply; checks that no packet carries an empty thinking."""
    answer, thinking = [], []
    for part in parts:
        holder = part.get("message", {}) if is_chat else part
        text = holder.get("content" if is_chat else "response", "")
        answer.append(text)
        if "thinking" in holder:
            expect(holder["thinking"] != "", f"an empty thinking was sent: {part}")
            thinking.append(holder["thinking"])
    return "".join(answer), "".join(thinking)


def check_split(answer: str, thinking: str, where: str) -> None:
    for tag in TAGS:
        expect(tag not in answer, f"{where}: {tag} in the answer: {answer[:120]!r}")
        expect(tag not in thinking, f"{where}: {tag} in thinking: {thinking[:120]!r}")


def require_reasoning(api: Api) -> None:
    """Skip unless the model reasons on /v1 with think=true; the Ollama routes must then reason too."""
    msg = flm_api.check_message_contract(api.ok(api.body(user(PROMPT), think=True, max_tokens=THINK_BUDGET)))
    if not msg.get("reasoning_content"):
        raise Skip("no reasoning_content on /v1 with think=true: the model does not reason")


def require_thinking(thinking: str, where: str) -> None:
    expect(thinking != "", f"{where}: think=true returned no thinking, though the model reasons on /v1")


@case
def chat_think_on_nonstream(api: Api) -> None:
    """/api/chat, think=true: thinking in message.thinking, the answer without tags."""
    require_reasoning(api)
    (part,) = chat(api, False, think=True)
    answer, thinking = fold([part], True)
    require_thinking(thinking, "/api/chat")
    check_split(answer, thinking, "/api/chat")
    expect(part["message"]["role"] == "assistant", f"role: {part['message']}")
    if part.get("done_reason") == "stop":
        expect(answer.strip() != "", "the model stopped after thinking without an answer")


@case
def chat_think_on_stream(api: Api) -> None:
    """/api/chat streamed, think=true: the same split, no tag in any packet."""
    require_reasoning(api)
    answer, thinking = fold(chat(api, True, think=True), True)
    require_thinking(thinking, "/api/chat stream")
    check_split(answer, thinking, "/api/chat stream")


@case
def generate_think_on(api: Api) -> None:
    """/api/generate, think=true, stream and not: no tag in the answer, and any thinking is split off (whether the route reasons at all is WP-21)."""
    for stream in (False, True):
        answer, thinking = fold(generate(api, stream, think=True), False)
        check_split(answer, thinking, f"/api/generate stream={stream}")


@case
def stream_equals_nonstream(api: Api) -> None:
    """The streamed answer and thinking fold to the non-stream ones for the same request."""
    require_reasoning(api)
    (whole,) = chat(api, False, think=True)
    whole_answer, whole_thinking = fold([whole], True)
    require_thinking(whole_thinking, "/api/chat")
    streamed = fold(chat(api, True, think=True), True)
    expect(streamed == (whole_answer, whole_thinking),
           f"stream {streamed!r} differs from non-stream {(whole_answer, whole_thinking)!r}")


@case
def no_empty_thinking_key(api: Api) -> None:
    """No packet of either route, stream or not, carries an empty thinking."""
    for stream in (False, True):
        fold(chat(api, stream, think=True), True)
        fold(generate(api, stream, think=True), False)
        fold(chat(api, stream, think=False), True)


@case
def think_off_no_thinking(api: Api) -> None:
    """think=false: no thinking and no tags, unless the model cannot stop reasoning."""
    if api.always_thinks:
        raise Skip("the model reasons whatever think says")
    for stream in (False, True):
        answer, thinking = fold(chat(api, stream, think=False), True)
        check_split(answer, thinking, f"/api/chat think=false stream={stream}")
        expect(thinking == "", f"think=false returned thinking: {thinking[:120]!r}")


@case
def v1_unchanged(api: Api) -> None:
    """/v1/chat/completions on the same request still splits into reasoning_content and content."""
    require_reasoning(api)
    msg = flm_api.check_message_contract(api.ok(api.body(user(PROMPT), think=True, max_tokens=THINK_BUDGET)))
    check_split(msg.get("content") or "", msg["reasoning_content"], "/v1")


CASES = flm_api.collect(globals())


if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
