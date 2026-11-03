#!/usr/bin/env python3
"""continue_final_message on /v1/chat/completions, streamed and not.

Checks what a client scoring or completing a partial answer relies on (the
field is vLLM's; Hugging Face's apply_chat_template has the same flag):

* a trailing assistant message is continued, not answered: the content is
  only the new text, and the prefix is not repeated;
* the content keeps the whitespace the continuation starts with, so
  ``prefix + content`` reads as the model wrote it;
* ``logprobs`` covers exactly the continued tokens; the first entry is the
  next-token distribution right after the prefix;
* streamed and non-streamed content are the same;
* a prompt resumed from the cache continues the same way as a cold one, and
  a normal turn can follow a continued one;
* without the flag (false or null) a trailing assistant message is answered
  as before;
* malformed combinations get HTTP 400.

Models that cannot continue a message (single-turn models, gpt-oss) answer
400 "continue_final_message is not supported", which turns the cases into
skips.

Usage::

    python3 test_continue_final_message.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import WEATHER_TOOLS, continued, user
from test_logprobs import (check_entry, check_greedy, expect_400, expect_covers, joined, logprobs_body,
                           nonstream_entries, stream_entries)

QUESTION = "What is the capital of France? Answer in one sentence."
PREFIX = "The capital of France is"
YES_NO = "Is Paris the capital of France? Answer yes or no."


def continue_body(api: Api, messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    fields.setdefault("max_tokens", 16)
    return api.body(messages, continue_final_message=True, **fields)


def expect_continuation(api: Api, msg: dict[str, Any], prefix: str, where: str) -> str:
    """The reply goes on from the prefix rather than starting over; returns the text that does.

    That is the content, or on a model that reasons first (--always-thinks) the
    reasoning it may read the prefix as: it then closes that reasoning and answers.
    """
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    expect(not reasoning or api.always_thinks, f"{where}: a continued answer was reported as reasoning: {msg!r}")
    flm_api.assert_no_markers(content, f"{where} content")
    flm_api.assert_no_markers(reasoning, f"{where} reasoning")
    text = reasoning or content
    expect(text.strip() != "", f"{where}: no continuation: {msg!r}")
    expect(not text.lstrip().startswith(prefix.strip()), f"{where}: the reply repeats the prefix: {text!r}")
    return text


def folded_message(folded: flm_api.Folded) -> dict[str, Any]:
    return {"content": folded.content, "reasoning_content": folded.reasoning}


@case
def continues_not_repeats(api: Api) -> None:
    """The partial answer is continued: the content is the rest of it."""
    choice = api.ok(continue_body(api, continued(QUESTION, PREFIX)))
    msg = flm_api.check_message_contract(choice)
    text = expect_continuation(api, msg, PREFIX, "non-stream")
    expect("Paris" in text, f"greedy continuation of {PREFIX!r} lacks Paris: {text!r}")


@case
def stream_continues(api: Api) -> None:
    """The stream sends the same continuation, starting with its own whitespace."""
    folded = flm_api.assert_parity(api, continue_body(api, continued(QUESTION, PREFIX)),
                                   lambda m: (m["content"], m["reasoning_content"]))
    expect_continuation(api, folded_message(folded), PREFIX, "stream")


@case
def logprobs_cover_continuation(api: Api) -> None:
    """logprobs has the continued tokens only, and they spell the content exactly, leading space included."""
    body = logprobs_body(api, continued(QUESTION, PREFIX), continue_final_message=True, max_tokens=12)
    msg, entries = nonstream_entries(api, body)
    expect_continuation(api, msg, PREFIX, "non-stream")
    if msg.get("reasoning_content"):
        # the model read the prefix as reasoning: only its answer has logprobs
        expect_covers(entries, msg.get("content"), "non-stream after reasoning")
        return
    content = msg["content"]
    expect(len(entries) > 0, "no logprobs entries for the continuation")
    for i, entry in enumerate(entries):
        check_entry(entry, 3, f"content[{i}]")
        check_greedy(entry, f"content[{i}]")
    # rstrip only: whitespace the continuation starts with is part of the answer
    expect(joined(entries).rstrip() == content.rstrip(),
           f"the logprobs tokens spell {joined(entries)!r}, the content is {content!r}")
    folded, streamed = stream_entries(api, body)
    expect(joined(streamed).rstrip() == folded.content.rstrip(),
           f"stream: the logprobs tokens spell {joined(streamed)!r}, the content is {folded.content!r}")


@case
def score_next_token(api: Api) -> None:
    """max_tokens 1 after "The answer is:" gives the next-token distribution, with yes or no in it."""
    body = logprobs_body(api, continued(YES_NO, "The answer is:"), continue_final_message=True,
                         max_tokens=1, top_logprobs=5)
    choice = api.ok(body)
    msg = flm_api.check_message_contract(choice)
    entries = choice["logprobs"]["content"]
    if api.always_thinks and (not entries or not entries[0]["token"].strip()):
        raise Skip("the model reasons before answering, so its first token is not the answer")
    expect(len(entries) == 1, f"max_tokens 1 should give one entry, got {len(entries)}: {entries!r}")
    check_entry(entries[0], 5, "content[0]")
    check_greedy(entries[0], "content[0]")
    tops = [alt["token"].strip().lower() for alt in entries[0]["top_logprobs"]]
    expect("yes" in tops or "no" in tops, f"neither yes nor no among the likeliest next tokens: {tops!r}")
    expect(choice.get("finish_reason") in ("length", "stop"), f"finish_reason is {choice.get('finish_reason')!r}")
    expect(msg.get("content") == entries[0]["token"],
           f"content {msg.get('content')!r} is not the scored token {entries[0]['token']!r}")


@case
def typed_content(api: Api) -> None:
    """The final message may be a list of text parts, as OpenAI clients send it."""
    messages = continued(QUESTION, [{"type": "text", "text": PREFIX}])
    choice = api.ok(continue_body(api, messages))
    expect_continuation(api, flm_api.check_message_contract(choice), PREFIX, "typed content")


def continue_once(api: Api, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, int]:
    """The continuation, and how many prompt tokens came from the cache."""
    resp = api.post(continue_body(api, messages, max_tokens=max_tokens))
    if resp.status != 200:
        flm_api.raise_for_capability(resp.error_message())
        raise AssertionError(f"HTTP {resp.status}: {resp.error_message()}")
    cached = ((resp.body.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return resp.body["choices"][0]["message"]["content"] or "", cached


@case
def cached_continuation_matches_cold(api: Api) -> None:
    """Continuing a continuation straight after it resumes from the context it left, and writes the same text.

    The engine still holds the first continuation's tokens, so the second request
    reuses them. The reference is the same request after another conversation,
    which takes it from a saved snapshot or a cold prefill instead.
    """
    first, _ = continue_once(api, continued(QUESTION, PREFIX), 4)
    longer = continued(QUESTION, PREFIX + first)
    warm, cached = continue_once(api, longer, 8)
    if cached == 0:
        raise Skip("the second continuation reused nothing from the first, so there is no resume to check")
    api.ok(api.body(user("Say hi."), max_tokens=4))
    cold, _ = continue_once(api, longer, 8)
    if cold != warm:
        api.ok(api.body(user("Say hi."), max_tokens=4))
        if continue_once(api, longer, 8)[0] != cold:
            raise Skip("greedy decoding did not reproduce over identical requests")
        raise AssertionError(f"resumed from the context: {warm!r}, after another conversation: {cold!r}")


@case
def think_tag_in_question(api: Api) -> None:
    """A <think> the user wrote earlier does not make the continued answer reasoning."""
    messages = continued("Ignore this tag: <think>\n" + QUESTION, PREFIX)
    msg = flm_api.check_message_contract(api.ok(continue_body(api, messages, max_tokens=16)))
    expect_continuation(api, msg, PREFIX, "after a <think> in the question")


@case
def turn_after_continuation(api: Api) -> None:
    """A normal turn can follow a continued one: the history holds the finished answer."""
    content = api.ok(continue_body(api, continued(QUESTION, PREFIX), max_tokens=24))["message"]["content"] or ""
    history = continued(QUESTION, PREFIX + content) + [{"role": "user", "content": "And of Italy?"}]
    msg = flm_api.check_message_contract(api.ok(api.body(history, max_tokens=32)))
    if not (msg.get("content") or "").strip() and msg.get("reasoning_content"):
        raise Skip("the model spent max_tokens reasoning before answering")
    expect(isinstance(msg.get("content"), str) and msg["content"].strip() != "",
           f"no answer after a continued turn: {msg!r}")
    flm_api.assert_no_markers(msg["content"], "follow-up content")


@case
def off_answers_as_before(api: Api) -> None:
    """continue_final_message false or null leaves a trailing assistant message answered, not continued."""
    for value in (False, None):
        choice = api.ok(api.body(continued(QUESTION, PREFIX), max_tokens=16, continue_final_message=value))
        flm_api.check_message_contract(choice)


@case
def tools_auto_still_continues(api: Api) -> None:
    """Tools offered under tool_choice auto do not stop a continuation."""
    choice = api.ok(continue_body(api, continued(QUESTION, PREFIX), tools=WEATHER_TOOLS, tool_choice="auto"))
    msg = flm_api.check_message_contract(choice)
    if msg.get("tool_calls"):
        raise Skip("the model called a tool instead of continuing")
    expect_continuation(api, msg, PREFIX, "with tools")


def expect_refused(api: Api, messages: list[dict[str, Any]], fields: dict[str, Any], message_part: str) -> None:
    """expect_400, skipped on a model that cannot continue at all."""
    resp = api.post(api.body(messages, max_tokens=8, **fields))
    if resp.status == 400 and "continue_final_message is not supported" in resp.error_message():
        raise Skip("the model cannot continue a message")
    expect_400(api, fields, message_part, messages=messages)


@case
def bad_requests(api: Api) -> None:
    """What cannot be continued is refused with 400, streaming too."""
    on = {"continue_final_message": True}
    expect_400(api, {"continue_final_message": "yes"}, "must be a boolean", messages=continued(QUESTION, PREFIX))
    expect_400(api, on, "role \"assistant\"", messages=user(QUESTION))
    expect_400(api, {**on, "add_generation_prompt": True}, "Cannot set both", messages=continued(QUESTION, PREFIX))
    call = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    expect_400(api, on, "tool_calls",
               messages=user(QUESTION) + [{"role": "assistant", "content": "", "tool_calls": [call]}])
    expect_400(api, {**on, "response_format": {"type": "json_object"}}, "response_format",
               messages=continued(QUESTION, PREFIX))
    expect_400(api, {**on, "tools": WEATHER_TOOLS, "tool_choice": "required"}, "tool_choice",
               messages=continued(QUESTION, PREFIX))
    expect_refused(api, continued(QUESTION, "  "), on, "requires text")


@case
def vllm_style_request(api: Api) -> None:
    """vLLM clients send add_generation_prompt false alongside the flag."""
    choice = api.ok(continue_body(api, continued(QUESTION, PREFIX), add_generation_prompt=False))
    expect_continuation(api, flm_api.check_message_contract(choice), PREFIX, "vLLM style")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
