#!/usr/bin/env python3
"""Response shape of /v1/chat/completions, streamed and not.

Checks the contract every client relies on, whatever the model:

* a non-streaming message has the OpenAI fields, no raw template markup,
  and a finish_reason that agrees with its tool calls;
* every stream chunk carries exactly one kind of delta, one stream id,
  one finish chunk, then ``data: [DONE]``;
* usage is only streamed when ``stream_options.include_usage`` asks, as
  one last chunk with no choices;
* streaming and non-streaming produce the same message: the server
  parses both with one parser, so with greedy sampling they must agree
  (skipped when greedy decoding itself does not reproduce).

Usage::

    python3 test_response_shape.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid

import flm_api
from flm_api import Api, case, expect
from fixtures import WEATHER_TOOLS, function_choice, user

PROMPT = "In one short sentence, what is the capital of France?"


@case
def nonstream_message(api: Api) -> None:
    """A plain answer has role, string content and a usage block that adds up."""
    resp = api.post(api.body(user(PROMPT), max_tokens=64))
    expect(resp.status == 200, f"HTTP {resp.status}: {resp.error_message()}")
    expect(resp.body.get("object") == "chat.completion", f"object is {resp.body.get('object')!r}")
    choice = resp.body["choices"][0]
    msg = flm_api.check_message_contract(choice)
    expect(msg["content"].strip() != "", "content is empty")
    expect("tool_calls" not in msg, "no tools were offered, yet the message has tool_calls")
    usage = resp.body.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        expect(isinstance(usage.get(key), int), f"usage.{key} missing: {usage}")
    expect(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
           f"usage does not add up: {usage}")


@case
def stream_without_usage(api: Api) -> None:
    """Without include_usage no chunk carries a usage key."""
    s = api.ok_stream(api.body(user(PROMPT), max_tokens=64))
    folded = flm_api.check_stream_contract(s, include_usage=False)
    expect(folded.content.strip() != "", "streamed content is empty")
    expect(folded.finish_reason in ("stop", "length"), f"finish_reason {folded.finish_reason!r}")


@case
def stream_with_usage(api: Api) -> None:
    """include_usage puts usage: null on every chunk and the totals on a last, choice-less chunk."""
    s = api.ok_stream(api.body(user(PROMPT), max_tokens=64, stream_options={"include_usage": True}))
    folded = flm_api.check_stream_contract(s, include_usage=True)
    last = s.chunks[-1]
    expect(last["choices"] == [], "the usage chunk must be the last one, with no choices")
    for chunk in s.chunks[:-1]:
        expect(chunk["usage"] is None, "usage must be null on every chunk but the last")
    expect(folded.usage["completion_tokens"] > 0, f"usage reports no completion tokens: {folded.usage}")


@case
def length_finish(api: Api) -> None:
    """Running out of max_tokens reports finish_reason "length" in both modes."""
    body = api.body(user("Count from 1 to 100, separated by commas."), max_tokens=8)
    choice = api.ok(body)
    expect(choice["finish_reason"] == "length", f"non-stream finish_reason is {choice['finish_reason']!r}")
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    expect(folded.finish_reason == "length", f"stream finish_reason is {folded.finish_reason!r}")


@case
def stream_matches_nonstream(api: Api) -> None:
    """Greedy streaming and non-streaming answers are the same text."""
    body = api.body(user("Name three primary colors, comma separated."), max_tokens=48)
    flm_api.assert_parity(api, body, lambda m: (m["reasoning_content"], m["content"]))


@case
def stream_matches_nonstream_tools(api: Api) -> None:
    """With a forced call, both modes return the same call name and arguments."""
    body = api.body(user("What's the weather in Paris?"), tools=WEATHER_TOOLS,
                    tool_choice=function_choice("get_weather"))
    flm_api.assert_parity(
        api, body, lambda m: [(c["function"]["name"], c["function"]["arguments"]) for c in m["tool_calls"]])


# The frontend, chosen by the catalog's details.family, takes the prompt-end checkpoint that an exact
# retry rewinds to. These families do, and the cases passed on a model of each
# (qwen2.5-it-oga-hipep:14b, qwen3.8-oga-hipep:27b).
EXACT_RETRY_FAMILIES = ("qwen2", "qwen3.5")
# These do not yet: Gemma3 on a text-only prompt, Gemma4e (gemma4-it-oga-hipep:26b-a4b) with thinking
# off. An exact retry there replays the prompt (cached_tokens 0). Follow-up of WP-33.
EXACT_RETRY_MISSING_FAMILIES = ("gemma3", "gemma4e")
# Not known either way: the cases have not run on a model of these families.
EXACT_RETRY_UNVERIFIED_FAMILIES = ("gemma4-12b", "gpt-oss", "laguna", "nemotron-h")
# check_exact_retry_families.py (CPU CI) holds the three lists to the catalog: every family with an
# oga-hipep model is in exactly one of them, and none names a family the catalog does not have.


def _require_oga_exact_retry(api: Api) -> str:
    """Skip unless the server's loaded model runs on oga-hipep with prompt-end rewind support."""
    base_url = api.url.removesuffix("/v1/chat/completions")
    try:
        with urllib.request.urlopen(base_url + "/api/ps", timeout=api.timeout) as reply:
            if reply.status != 200:
                raise flm_api.Skip(f"/api/ps returned HTTP {reply.status}")
            ps = json.load(reply)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise flm_api.Skip(f"server does not expose /api/ps backend metadata ({exc})")

    models = ps.get("models") or []
    loaded = next((m for m in models if m.get("name") == api.model), None)
    if loaded is None and len(models) == 1:
        loaded = models[0]
    if loaded is None:
        raise flm_api.Skip(f"{api.model} is not loaded in /api/ps; cannot verify OGA backend")

    details = loaded.get("details") or {}
    backend = details.get("backend")
    if backend != "oga-hipep":
        raise flm_api.Skip(f"loaded backend is {backend!r}, not 'oga-hipep'")

    canonical = str(loaded.get("name") or "")
    family = details.get("family")
    if family in EXACT_RETRY_MISSING_FAMILIES:
        raise flm_api.Skip(f"{canonical}: the {family} frontend does not yet checkpoint the prompt end")
    if family in EXACT_RETRY_UNVERIFIED_FAMILIES:
        raise flm_api.Skip(f"{canonical}: exact retry is not verified for the {family} frontend")
    if family not in EXACT_RETRY_FAMILIES:
        raise flm_api.Skip(f"{canonical}: family {family!r} is in none of the exact-retry lists")
    return canonical


def _check_retry_usage(
    resp_usage: dict,
    label: str,
    *,
    expect_hit: bool,
    expected_prompt_tokens: int | None = None,
) -> int:
    prompt_tokens = resp_usage.get("prompt_tokens")
    completion_tokens = resp_usage.get("completion_tokens")
    cached = (resp_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    expect(isinstance(prompt_tokens, int) and prompt_tokens > 0, f"{label} missing prompt_tokens: {resp_usage}")
    expect(
        isinstance(completion_tokens, int) and completion_tokens > 1,
        f"{label} decoded only {completion_tokens} token(s); retry must run after decode advances past prompt end",
    )
    expect(isinstance(cached, int), f"{label} missing integer prompt_tokens_details.cached_tokens: {resp_usage}")
    if expected_prompt_tokens is not None:
        expect(
            prompt_tokens == expected_prompt_tokens,
            f"{label} prompt_tokens changed from {expected_prompt_tokens} to {prompt_tokens}",
        )
    if expect_hit:
        expect(
            cached == prompt_tokens,
            f"{label} exact retry cached {cached}/{prompt_tokens} tokens (expected full reuse)",
        )
    else:
        expect(cached == 0, f"{label} cold run unexpectedly cached {cached}/{prompt_tokens} tokens")
    return prompt_tokens


@case
def exact_retry_reuses_prompt(api: Api) -> None:
    """Single-slot exact retries (JSON and SSE include_usage) reuse the full prompt."""
    _require_oga_exact_retry(api)
    nonce = uuid.uuid4().hex[:8]
    prompt = user(f"[run {nonce}] Write a short sentence explaining why two plus four equals six.")
    body = api.body(prompt, max_tokens=32)

    cold = api.post(body)
    expect(cold.status == 200, f"plain cold HTTP {cold.status}: {cold.error_message()}")
    prompt_n = _check_retry_usage(cold.body.get("usage") or {}, "plain-cold", expect_hit=False)
    cold_msg = flm_api.check_message_contract(cold.body["choices"][0])

    retry = api.post(body)
    expect(retry.status == 200, f"plain retry HTTP {retry.status}: {retry.error_message()}")
    _check_retry_usage(retry.body.get("usage") or {}, "plain-retry", expect_hit=True, expected_prompt_tokens=prompt_n)
    retry_msg = flm_api.check_message_contract(retry.body["choices"][0])
    expect(
        retry_msg["content"] == cold_msg["content"],
        f"plain retry changed greedy output: {cold_msg['content']!r} vs {retry_msg['content']!r}",
    )

    sse = flm_api.check_stream_contract(
        api.ok_stream(api.body(prompt, max_tokens=32, stream_options={"include_usage": True})),
        include_usage=True,
    )
    _check_retry_usage(sse.usage or {}, "plain-sse", expect_hit=True, expected_prompt_tokens=prompt_n)
    expect(
        sse.content == cold_msg["content"],
        f"plain SSE retry changed greedy output: {cold_msg['content']!r} vs {sse.content!r}",
    )


@case
def exact_retry_multi_chunk_reuses_prompt(api: Api) -> None:
    """A >4,096-token prompt (spanning default 4,096-token prefill chunks) reuses the full prompt on retry.

    Requires server --ctx-len > ~4,850 (skips cleanly if the server rejects the prompt for context capacity)
    and default --prefill-chunk-len <= 4096 to exercise multi-chunk expect_prefill + rewind.
    """
    _require_oga_exact_retry(api)
    nonce = uuid.uuid4().hex[:8]
    long_system = (
        f"[run {nonce}] You are a concise warehouse assistant. Follow the SOP exactly.\n"
        + "Section note: verify bin tag, pallet count, aisle label, and seal code before dispatch. " * 250
    )
    chunked = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "Write a five-word sentence confirming you have read the warehouse SOP."},
    ]
    body_long = api.body(chunked, max_tokens=32)

    cold = api.post(body_long)
    if cold.status == 400:
        err = cold.error_message()
        if "exceeds the" in err.lower() or "context" in err.lower():
            raise flm_api.Skip(f"server context window cannot hold the >4096-token prompt: {err}")
    expect(cold.status == 200, f"chunked cold HTTP {cold.status}: {cold.error_message()}")
    long_n = _check_retry_usage(cold.body.get("usage") or {}, "chunked-cold", expect_hit=False)
    expect(long_n > 4096, f"chunked prompt was only {long_n} tokens (expected >4096 to span chunks)")
    cold_msg = flm_api.check_message_contract(cold.body["choices"][0])

    retry = api.post(body_long)
    expect(retry.status == 200, f"chunked retry HTTP {retry.status}: {retry.error_message()}")
    _check_retry_usage(retry.body.get("usage") or {}, "chunked-retry", expect_hit=True, expected_prompt_tokens=long_n)
    retry_msg = flm_api.check_message_contract(retry.body["choices"][0])
    expect(
        retry_msg["content"] == cold_msg["content"],
        f"chunked retry changed greedy output: {cold_msg['content']!r} vs {retry_msg['content']!r}",
    )


@case
def exact_retry_interleaved_aba_reuses_prompt(api: Api) -> None:
    """Interleaved A/B/A exact retry reuses slot A when the server is configured with >= 2 effective slots."""
    _require_oga_exact_retry(api)
    if os.environ.get("FLM_TEST_MULTI_SLOT") != "1":
        raise flm_api.Skip("set FLM_TEST_MULTI_SLOT=1 when the server is configured with >= 2 effective OGA slots")
    nonce = uuid.uuid4().hex[:8]
    body_a = api.body(
        user(f"[run {nonce}] Write a short sentence explaining why two plus four equals six."),
        max_tokens=32,
    )
    body_b = api.body(
        user(f"[run {nonce}] Write a short sentence explaining why three plus five equals eight."),
        max_tokens=32,
    )

    cold_a = api.post(body_a)
    expect(cold_a.status == 200, f"A/B/A cold A HTTP {cold_a.status}: {cold_a.error_message()}")
    n_a = _check_retry_usage(cold_a.body.get("usage") or {}, "aba-cold-A", expect_hit=False)
    cold_a_msg = flm_api.check_message_contract(cold_a.body["choices"][0])

    cold_b = api.post(body_b)
    expect(cold_b.status == 200, f"A/B/A cold B HTTP {cold_b.status}: {cold_b.error_message()}")
    _check_retry_usage(cold_b.body.get("usage") or {}, "aba-cold-B", expect_hit=False)

    retry_a = api.post(body_a)
    expect(retry_a.status == 200, f"A/B/A retry A HTTP {retry_a.status}: {retry_a.error_message()}")
    _check_retry_usage(retry_a.body.get("usage") or {}, "aba-retry-A", expect_hit=True, expected_prompt_tokens=n_a)
    retry_a_msg = flm_api.check_message_contract(retry_a.body["choices"][0])
    expect(
        retry_a_msg["content"] == cold_a_msg["content"],
        f"A/B/A retry changed greedy output: {cold_a_msg['content']!r} vs {retry_a_msg['content']!r}",
    )


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
