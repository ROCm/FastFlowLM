#!/usr/bin/env python3
"""Integration test for the OpenAI-compatible usage/cache contract.

FastFlowLM's engine keeps a KV cache across turns of the same
conversation and only prefills the tokens that were not already in
that cache. A regression made ``usage.prompt_tokens`` report only the
newly prefilled tail instead of the whole prompt, so a client that
tracks context growth from ``usage`` saw it collapse to near-zero on
every turn after the first.

This script guards the fix by driving a live, already-running
FastFlowLM server through two sequential chat turns -- once with a
plain JSON response and once with ``stream=True`` -- and checking
that, on both endpoints:

* ``usage.prompt_tokens`` is the *whole* prompt (cached prefix + new
  tail), so it strictly grows from turn 1 to turn 2.
* ``usage.prompt_tokens_details.cached_tokens`` is present on turn 2,
  is a positive integer, and never exceeds ``prompt_tokens`` (it is a
  subset of it, not an alternate count).
* ``usage.total_tokens == usage.prompt_tokens +
  usage.completion_tokens`` on every turn.

It deliberately does NOT assert anything about
``stream_options.include_usage`` or about ``usage`` being present (as
``null`` or otherwise) on intermediate streaming chunks -- that
behaviour belongs to a different, unmerged branch.

Usage
-----
Start the FastFlowLM server yourself (e.g. ``flm serve <model>``),
then run::

    python3 test_usage_contract.py <model>
    python3 test_usage_contract.py <model> --endpoint URL --timeout T

The script requires only the Python 3 standard library. It exits 0
and prints a pass summary (with the observed token counts for every
turn) on success, or exits non-zero with a diagnosable
``AssertionError`` message on failure.
"""

# Every check here reports on the server under test, not on a caller
# passing bad arguments, so a wrong type in the response is a test
# failure rather than a TypeError. main() catches AssertionError alone
# to turn any violation into the pass/fail summary and exit code.
# ruff: noqa: TRY004

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_ENDPOINT = "http://127.0.0.1:52625"
DEFAULT_TIMEOUT = 300.0
MAX_TOKENS = 64
TEMPERATURE = 0.1
PROMPT_1 = "In one short sentence, what is the capital of France?"
PROMPT_2 = "In one short sentence, name its most famous landmark."

# Single-turn models reject conversation history, so their cache reuse
# is a pinned system prefix shared by successive one-shot requests.
SINGLE_TURN_SYSTEM = (
    "You are a terse assistant. Answer in one short sentence. "
    "Never elaborate. Stay factual."
)
SINGLE_TURN_PROMPT_1 = "What is the capital of France?"
SINGLE_TURN_PROMPT_2 = "What is the capital of Japan?"

# Substring of the server's rejection when history is not allowed.
SINGLE_TURN_MARKER = b"only supports single-turn requests"


class SingleTurnModel(Exception):
    """The model under test rejects multi-turn conversation history."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this integration test."""
    parser = argparse.ArgumentParser(
        description=(
            "Integration test for the OpenAI usage/cache contract: "
            "usage.prompt_tokens must report the whole prompt "
            "(cached prefix + new tail), and "
            "usage.prompt_tokens_details.cached_tokens must report "
            "the cached subset of it."
        ),
    )
    parser.add_argument(
        "model",
        help="model tag currently loaded by the FastFlowLM server",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-request timeout in seconds (default: %(default)s)",
    )
    return parser.parse_args()


def chat_completions_url(endpoint: str) -> str:
    """Build the chat completions URL from a server base ``endpoint``."""
    return endpoint.rstrip("/") + "/v1/chat/completions"


def build_turn1_messages() -> list[dict[str, str]]:
    """Build the message list for the first turn of the conversation."""
    return [{"role": "user", "content": PROMPT_1}]


def build_turn2_messages(
    turn1_messages: list[dict[str, str]],
    assistant_reply: str,
) -> list[dict[str, str]]:
    """Append the turn-1 assistant reply and a new user message.

    ``turn1_messages`` is not mutated; a new list is returned that
    contains the original turn, the assistant's reply, and a follow
    up user message, so the server sees a growing conversation.
    """
    messages = copy.deepcopy(turn1_messages)
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": PROMPT_2})
    return messages


def build_single_turn_messages(prompt: str) -> list[dict[str, str]]:
    """Build a one-shot exchange sharing the pinned system prefix."""
    return [
        {"role": "system", "content": SINGLE_TURN_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> bytes:
    """POST ``payload`` as JSON to ``url`` and return the raw body.

    Raises ``AssertionError`` (with the response body, if any) when
    the request fails, so a failure is diagnosable without a
    debugger.
    """
    data = json.dumps(payload).encode("utf-8")
    accept = (
        "text/event-stream" if payload.get("stream") else "application/json"
    )
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        if SINGLE_TURN_MARKER in raw:
            raise SingleTurnModel(raw.decode("utf-8", errors="replace"))
        text = raw.decode("utf-8", errors="replace")
        raise AssertionError(
            f"POST {url} failed with HTTP {error.code}: {text}"
        ) from error
    except urllib.error.URLError as error:
        raise AssertionError(f"POST {url} failed: {error.reason}") from error
    # The server answers 200 with an error body in some builds, so the
    # marker has to be checked on the success path too.
    if SINGLE_TURN_MARKER in body:
        raise SingleTurnModel(body.decode("utf-8", errors="replace"))
    return body


def extract_message_and_usage(
    response: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Extract the assistant reply and usage from a JSON response.

    ``response`` is the decoded body of a non-streaming chat
    completion response.
    """
    choices = response.get("choices")
    if not choices:
        raise AssertionError(f"response has no 'choices': {response!r}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise AssertionError(
            f"assistant message has no string 'content': {message!r}"
        )
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError(f"response has no 'usage' object: {response!r}")
    return content, usage


def parse_sse_chunks(raw_body: bytes) -> list[dict[str, Any]]:
    """Parse an SSE response body into a list of decoded JSON chunks.

    Lines that are not ``data: ...`` are ignored, and the terminal
    literal ``data: [DONE]`` line is dropped rather than parsed as
    JSON.
    """
    chunks = []
    text = raw_body.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        if not payload:
            continue
        chunks.append(json.loads(payload))
    return chunks


def extract_streamed_reply_and_usage(
    chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Reassemble the assistant reply and find the final usage chunk.

    The assistant reply is reassembled by concatenating every
    ``choices[0].delta.content`` piece in order. The final usage is
    taken from the last chunk that carries a non-null ``usage``
    object, per the OpenAI streaming contract.
    """
    content_parts = []
    final_usage = None
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                content_parts.append(piece)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            final_usage = usage
    if final_usage is None:
        raise AssertionError(
            "no streamed chunk carried a non-null 'usage' object; "
            f"chunks were: {chunks!r}"
        )
    return "".join(content_parts), final_usage


def print_usage(turn_label: str, usage: dict[str, Any]) -> None:
    """Print the observed token counts for one turn, for PR evidence."""
    details = usage.get("prompt_tokens_details")
    cached_tokens = None
    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    print(
        f"  {turn_label}: prompt_tokens={prompt_tokens} "
        f"cached_tokens={cached_tokens} "
        f"completion_tokens={completion_tokens} "
        f"total_tokens={total_tokens}"
    )


def assert_usage_consistency(usage: dict[str, Any], turn_label: str) -> None:
    """Assert ``total_tokens == prompt_tokens + completion_tokens``.

    Raises ``AssertionError`` with the offending usage object when
    fields are missing, not integers, or inconsistent.
    """
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    for name, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("total_tokens", total_tokens),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise AssertionError(
                f"{turn_label}: usage.{name} must be an int, got {value!r} in "
                f"{usage!r}"
            )
    if total_tokens != prompt_tokens + completion_tokens:
        raise AssertionError(
            f"{turn_label}: total_tokens ({total_tokens}) != "
            f"prompt_tokens ({prompt_tokens}) + "
            f"completion_tokens ({completion_tokens})"
        )


def assert_cache_growth(
    turn1_prompt_tokens: int,
    turn2_usage: dict[str, Any],
    turn_label: str,
) -> None:
    """Assert turn 2 grew the prompt and reused a cached prefix.

    Checks that turn 2's ``prompt_tokens`` is strictly greater than
    turn 1's, and that ``prompt_tokens_details.cached_tokens`` is a
    positive integer no larger than turn 2's ``prompt_tokens``.
    Nothing is asserted about turn 1's ``cached_tokens``: the server
    is long-lived and may already hold a cached prefix from earlier
    activity.
    """
    turn2_prompt_tokens = turn2_usage.get("prompt_tokens")
    if not isinstance(turn2_prompt_tokens, int):
        raise AssertionError(
            f"{turn_label}: turn 2 usage.prompt_tokens must be an int, got "
            f"{turn2_prompt_tokens!r} in {turn2_usage!r}"
        )
    if turn2_prompt_tokens <= turn1_prompt_tokens:
        raise AssertionError(
            f"{turn_label}: expected turn 2 prompt_tokens "
            f"({turn2_prompt_tokens}) > turn 1 prompt_tokens "
            f"({turn1_prompt_tokens}); the KV cache prefix reuse bug "
            "would make usage collapse instead of grow"
        )
    assert_prefix_reused(turn2_usage, turn_label)


def assert_prefix_reused(usage: dict[str, Any], turn_label: str) -> None:
    """Assert a cached prefix was reused and counted in the prompt.

    ``prompt_tokens_details.cached_tokens`` must be a positive integer
    no larger than ``prompt_tokens``: the cached prefix is reported as
    a subset of the whole prompt, never instead of it.
    """
    prompt_tokens = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        raise AssertionError(
            f"{turn_label}: usage is missing "
            f"prompt_tokens_details.cached_tokens: {usage!r}"
        )
    cached_tokens = details["cached_tokens"]
    if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool):
        raise AssertionError(
            f"{turn_label}: cached_tokens must be an int, got "
            f"{cached_tokens!r} ({type(cached_tokens).__name__})"
        )
    if cached_tokens <= 0:
        raise AssertionError(
            f"{turn_label}: expected cached_tokens > 0, got {cached_tokens}"
        )
    if cached_tokens > prompt_tokens:
        raise AssertionError(
            f"{turn_label}: cached_tokens ({cached_tokens}) exceeds "
            f"prompt_tokens ({prompt_tokens}); cached_tokens must be a "
            "subset of prompt_tokens"
        )


def run_pinned_prefix_checks(
    url: str,
    model: str,
    timeout: float,
    stream: bool,
    label: str,
) -> None:
    """Check cache accounting on a single-turn model.

    Single-turn models reject conversation history, so the growing
    transcript used elsewhere does not apply. Their cache reuse is the
    pinned system prefix shared by successive one-shot requests: two
    requests share one system message, and the second must report that
    prefix as reused and counted inside ``prompt_tokens``.
    """
    prompts = (SINGLE_TURN_PROMPT_1, SINGLE_TURN_PROMPT_2)
    suffix = " (stream)" if stream else ""
    usages = []
    for index, prompt in enumerate(prompts, start=1):
        payload: dict[str, Any] = {
            "model": model,
            "messages": build_single_turn_messages(prompt),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        if stream:
            payload["stream"] = True
        body = post_json(url, payload, timeout)
        if stream:
            _, usage = extract_streamed_reply_and_usage(parse_sse_chunks(body))
        else:
            _, usage = extract_message_and_usage(json.loads(body))
        print_usage(f"request {index}{suffix}", usage)
        assert_usage_consistency(usage, f"{label} request {index}")
        usages.append(usage)
    # Only the second request is required to show a reused prefix: the
    # first may or may not already sit on the pin, depending on whether
    # the model was just loaded.
    assert_prefix_reused(usages[1], f"{label} request 2")


def run_non_streaming_test(endpoint: str, model: str, timeout: float) -> None:
    """Run Test 1: the two-turn, non-streaming usage contract check."""
    print("Test 1: non-streaming usage contract")
    url = chat_completions_url(endpoint)

    turn1_messages = build_turn1_messages()
    turn1_payload = {
        "model": model,
        "messages": turn1_messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    turn1_body = post_json(url, turn1_payload, timeout)
    turn1_response = json.loads(turn1_body)
    turn1_reply, turn1_usage = extract_message_and_usage(turn1_response)
    print_usage("turn 1", turn1_usage)
    assert_usage_consistency(turn1_usage, "test 1 turn 1")

    turn2_messages = build_turn2_messages(turn1_messages, turn1_reply)
    turn2_payload = copy.deepcopy(turn1_payload)
    turn2_payload["messages"] = turn2_messages
    try:
        turn2_body = post_json(url, turn2_payload, timeout)
    except SingleTurnModel:
        print(
            "  model rejects multi-turn history; checking the pinned "
            "system prefix instead"
        )
        run_pinned_prefix_checks(
            url,
            model,
            timeout,
            stream=False,
            label="test 1",
        )
        print("Test 1 PASSED")
        return
    turn2_response = json.loads(turn2_body)
    _, turn2_usage = extract_message_and_usage(turn2_response)
    print_usage("turn 2", turn2_usage)
    assert_usage_consistency(turn2_usage, "test 1 turn 2")
    assert_cache_growth(
        turn1_usage["prompt_tokens"],
        turn2_usage,
        "test 1",
    )
    print("Test 1 PASSED")


def run_streaming_test(endpoint: str, model: str, timeout: float) -> None:
    """Run Test 2: the two-turn, streaming usage contract check."""
    print("\nTest 2: streaming usage contract")
    url = chat_completions_url(endpoint)

    turn1_messages = build_turn1_messages()
    turn1_payload = {
        "model": model,
        "messages": turn1_messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    turn1_body = post_json(url, turn1_payload, timeout)
    turn1_chunks = parse_sse_chunks(turn1_body)
    turn1_reply, turn1_usage = extract_streamed_reply_and_usage(
        turn1_chunks,
    )
    print_usage("turn 1 (stream)", turn1_usage)
    assert_usage_consistency(turn1_usage, "test 2 turn 1")

    turn2_messages = build_turn2_messages(turn1_messages, turn1_reply)
    turn2_payload = copy.deepcopy(turn1_payload)
    turn2_payload["messages"] = turn2_messages
    try:
        turn2_body = post_json(url, turn2_payload, timeout)
    except SingleTurnModel:
        print(
            "  model rejects multi-turn history; checking the pinned "
            "system prefix instead"
        )
        run_pinned_prefix_checks(
            url,
            model,
            timeout,
            stream=True,
            label="test 2",
        )
        print("Test 2 PASSED")
        return
    turn2_chunks = parse_sse_chunks(turn2_body)
    _, turn2_usage = extract_streamed_reply_and_usage(turn2_chunks)
    print_usage("turn 2 (stream)", turn2_usage)
    assert_usage_consistency(turn2_usage, "test 2 turn 2")
    assert_cache_growth(
        turn1_usage["prompt_tokens"],
        turn2_usage,
        "test 2",
    )
    print("Test 2 PASSED")


def main() -> int:
    """Run both usage-contract tests and print a pass/fail summary."""
    args = parse_args()
    try:
        run_non_streaming_test(args.endpoint, args.model, args.timeout)
        run_streaming_test(args.endpoint, args.model, args.timeout)
    except AssertionError as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 1
    print("\nAll usage contract checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
