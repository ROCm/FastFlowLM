#!/usr/bin/env python3
"""Reasoning output and the per-request thinking switch.

* With ``think: true`` a reasoning model's thinking comes back in
  ``reasoning_content``, never as ``<think>`` markup in ``content``.
* With ``think: false`` (or ``reasoning_effort: "none"``) there is no
  ``reasoning_content``, unless the model cannot stop thinking: pass
  ``--always-thinks`` for those.
* A request without ``think`` gets the model's default, whatever an earlier
  request set.
* A stream sends every reasoning delta before the first content delta and
  folds to exactly the non-streaming message, so no half-written tag ever
  went out as text.

Models that never reason skip the thinking-on cases.

Usage::

    python3 test_reasoning.py <model> [--always-thinks] [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import user

# Short to think about, so a small budget still reaches the answer.
PROMPT = "Is 7 a prime number? Answer with yes or no."
THINK_BUDGET = 1536


def require_reasoning(msg: dict) -> None:
    if not msg.get("reasoning_content"):
        raise Skip("no reasoning_content with think=true: the model does not reason")


@case
def think_on_nonstream(api: Api) -> None:
    """think=true puts the thinking in reasoning_content and only the answer in content."""
    choice = api.ok(api.body(user(PROMPT), think=True, max_tokens=THINK_BUDGET))
    msg = flm_api.check_message_contract(choice)
    require_reasoning(msg)
    if choice["finish_reason"] == "stop":
        expect(msg["content"].strip() != "", "the model stopped after thinking without an answer")


@case
def think_on_stream(api: Api) -> None:
    """A stream sends all reasoning before any content, and folds to the non-streaming message."""
    body = api.body(user(PROMPT), think=True, max_tokens=THINK_BUDGET)
    folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
    if folded.reasoning == "":
        raise Skip("no reasoning_content with think=true: the model does not reason")
    if "content" in folded.order:
        first_content = folded.order.index("content")
        expect("reasoning" not in folded.order[first_content:],
               "a reasoning delta arrived after content had started")
    # long outputs do not reproduce bit for bit between requests: compare a short one
    short = api.body(user(PROMPT), think=True, max_tokens=64)
    flm_api.assert_parity(api, short, lambda m: (m["reasoning_content"], m["content"]))


@case
def think_off_nonstream(api: Api) -> None:
    """think=false returns no reasoning_content."""
    choice = api.ok(api.body(user(PROMPT), think=False, max_tokens=128))
    msg = flm_api.check_message_contract(choice)
    if api.always_thinks:
        return
    expect("reasoning_content" not in msg,
           f"think=false still returned reasoning_content (pass --always-thinks for models that "
           f"cannot stop thinking): {msg.get('reasoning_content', '')[:120]!r}")
    expect(msg["content"].strip() != "", "content is empty")


@case
def think_off_stream(api: Api) -> None:
    """think=false streams no reasoning_content deltas."""
    s = api.ok_stream(api.body(user(PROMPT), think=False, max_tokens=128))
    folded = flm_api.check_stream_contract(s, include_usage=False)
    if not api.always_thinks:
        expect(folded.reasoning == "", f"think=false streamed reasoning: {folded.reasoning[:120]!r}")


@case
def reasoning_effort_none(api: Api) -> None:
    """reasoning_effort "none" turns thinking off like think=false."""
    body = api.body(user(PROMPT), max_tokens=128, reasoning_effort="none")
    del body["think"]
    msg = flm_api.check_message_contract(api.ok(body))
    if not api.always_thinks:
        expect("reasoning_content" not in msg, "reasoning_effort=none still returned reasoning_content")


@case
def think_switch_is_per_request(api: Api) -> None:
    """Off, on, off again: each request gets what it asked for."""
    off = api.ok(api.body(user(PROMPT), think=False, max_tokens=128))["message"]
    on = api.ok(api.body(user(PROMPT), think=True, max_tokens=THINK_BUDGET))["message"]
    require_reasoning(on)
    off_again = api.ok(api.body(user(PROMPT), think=False, max_tokens=128))["message"]
    if not api.always_thinks:
        expect("reasoning_content" not in off and "reasoning_content" not in off_again,
               "think=false returned reasoning_content after a think=true request")


@case
def history_with_reasoning(api: Api) -> None:
    """A previous turn's reasoning_content sent back in history is accepted (and dropped)."""
    first = api.ok(api.body(user("Pick a random fruit. One word."), think=True, max_tokens=THINK_BUDGET))["message"]
    messages = user("Pick a random fruit. One word.") + [
        {"role": "assistant", "content": first.get("content") or "apple",
         "reasoning_content": first.get("reasoning_content", "")},
        {"role": "user", "content": "What color is it usually? One word."},
    ]
    choice = api.ok(api.body(messages, think=False, max_tokens=64))
    flm_api.check_message_contract(choice)


@case
def think_does_not_carry_over(api: Api) -> None:
    """A request without think gets the model's default, whatever an earlier request set."""
    def thinks(**fields) -> bool:
        body = api.body(user(PROMPT), max_tokens=THINK_BUDGET, **fields)
        if "think" not in fields:
            del body["think"]
        return bool(flm_api.check_message_contract(api.ok(body)).get("reasoning_content"))

    default = thinks()
    for think in (True, False):
        thinks(think=think)
        expect(thinks() == default, f"after think={think} a request without think "
               f"{'reasoned' if not default else 'did not reason'}, unlike the model's default")


CASES = flm_api.collect(globals())


if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
