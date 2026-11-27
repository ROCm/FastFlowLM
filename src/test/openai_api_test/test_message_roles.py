#!/usr/bin/env python3
"""Instruction messages on /v1/chat/completions: ``developer``, and ``system`` after the first turn.

OpenAI defines ``developer`` as ``system`` under a new name and allows both
anywhere in the conversation; agents (pi, OpenClaw) send their system prompt
as ``developer`` to a model they mark as reasoning, and add instructions on
later turns. Checks:

* a ``developer`` instruction is rendered exactly as the same ``system``
  instruction (same ``prompt_tokens``), alone or followed by ``system``;
* a ``system`` or ``developer`` message after the first turn answers 200 and
  reaches the prompt (``prompt_tokens`` grow by it), on a template that
  renders it in place (Qwen3) and on one that rejects it (Qwen3.5, where the
  server sends it as a ``<system-reminder>`` user turn);
* a conversation that only grows, with a later instruction, keeps reusing the
  earlier prompt (``cached_tokens``), so nothing earlier moved.

``prompt_tokens`` is the signal: a dropped instruction answers 200 too.

Usage::

    python3 test_message_roles.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect

RULE = "The secret word is PINEAPPLE. When asked for the secret word, answer with it."
STYLE = "Answer in one word."
QUESTION = "What is the secret word?"


def msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def history() -> list[dict[str, Any]]:
    return [msg("system", "You are a helpful assistant."), msg("user", "Hello."),
            msg("assistant", "Hello! How can I help?")]


def usage_of(api: Api, messages: list[dict[str, Any]], max_tokens: int = 16) -> tuple[dict[str, Any], dict[str, Any]]:
    """Send a request that must answer 200; return its message and usage."""
    resp = api.post(api.body(messages, max_tokens=max_tokens))
    if resp.status != 200:
        flm_api.raise_for_capability(resp.error_message())
        raise AssertionError(f"HTTP {resp.status}: {resp.error_message()}")
    choice = resp.body["choices"][0]
    usage = resp.body.get("usage") or {}
    expect(isinstance(usage.get("prompt_tokens"), int), f"no usage.prompt_tokens: {resp.body!r}")
    return flm_api.check_message_contract(choice), usage


def prompt_tokens(api: Api, messages: list[dict[str, Any]]) -> int:
    return usage_of(api, messages, max_tokens=1)[1]["prompt_tokens"]


@case
def developer_renders_as_system(api: Api) -> None:
    """A leading developer instruction gives the prompt of the same system instruction."""
    as_system = prompt_tokens(api, [msg("system", RULE), msg("user", QUESTION)])
    as_developer = prompt_tokens(api, [msg("developer", RULE), msg("user", QUESTION)])
    expect(as_developer == as_system,
           f"developer prompt has {as_developer} tokens, the same instruction as system {as_system}")


@case
def developer_then_system_is_one_message(api: Api) -> None:
    """developer followed by system at the start render as one system message, in order."""
    merged = prompt_tokens(api, [msg("system", RULE + " " + STYLE), msg("user", QUESTION)])
    split = prompt_tokens(api, [msg("developer", RULE + " "), msg("system", STYLE), msg("user", QUESTION)])
    expect(split == merged, f"developer + system prompt has {split} tokens, one system message {merged}")


def later_instruction(api: Api, role: str) -> None:
    without = prompt_tokens(api, history() + [msg("user", QUESTION)])
    message, usage = usage_of(api, history() + [msg(role, RULE), msg("user", QUESTION)])
    expect(usage["prompt_tokens"] > without,
           f"a later {role} message did not reach the prompt: {usage['prompt_tokens']} tokens, {without} without it")
    flm_api.assert_no_markers(message.get("content"), "content")


@case
def later_system_answers(api: Api) -> None:
    """A system message after the first turn answers 200 and reaches the prompt."""
    later_instruction(api, "system")


@case
def later_developer_answers(api: Api) -> None:
    """A developer message after the first turn answers 200 and reaches the prompt."""
    later_instruction(api, "developer")


@case
def later_instruction_with_image_is_400(api: Api) -> None:
    """An image in a later instruction is refused, naming the part, whether or not the template takes the message."""
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    later = msg("system", [{"type": "text", "text": RULE}, image])
    resp = api.post(api.body(history() + [later, msg("user", QUESTION)], max_tokens=1))
    expect(resp.status == 400, f"expected 400, got {resp.status}: {resp.error_message()}")
    expect("messages[3].content[1]" in resp.error_message(), f"the error does not name the part: {resp.error_message()}")


@case
def later_instruction_keeps_prefix(api: Api) -> None:
    """A conversation that adds an instruction on a later turn reuses the whole earlier prompt."""
    first = history() + [msg("user", "Name a fruit.")]
    reply, usage = usage_of(api, first)
    answered = first + [msg("assistant", reply.get("content") or "")]

    def reused(messages: list[dict[str, Any]]) -> int:
        usage_of(api, first)  # the earlier prompt is what the cache holds when the next turn comes
        cached = (usage_of(api, messages)[1].get("prompt_tokens_details") or {}).get("cached_tokens")
        return cached if isinstance(cached, int) else 0

    # control: the same next turn without the instruction; only a model that reuses nothing there skips
    control = reused(answered + [msg("user", QUESTION)])
    if control == 0:
        raise Skip("the server reports no reused prompt tokens for this model")
    # A template that opens reasoning after the reply header, and renders the reply
    # without it on the next turn, reuses the earlier prompt only up to that header
    # (the control's count): the instruction must not cost anything beyond it.
    expected = min(usage["prompt_tokens"], control)
    cached = reused(answered + [msg("system", RULE), msg("user", QUESTION)])
    expect(cached >= expected,
           f"the turn with a later instruction reused {cached} tokens, fewer than the earlier prompt "
           f"({usage['prompt_tokens']}, {control} reused without the instruction): something before it moved")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
