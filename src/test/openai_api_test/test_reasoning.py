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

import json
import os
import urllib.request

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


def reads_nested_switch(api: Api) -> bool:
    """Whether this model is declared to read chat_template_kwargs.enable_thinking.

    FLM_TEST_NESTED_THINKING=on or off declares it; unset, the Qwen3.5 packages are taken to read it.
    """
    declared = os.environ.get("FLM_TEST_NESTED_THINKING", "").lower()
    if declared in ("on", "off"):
        return declared == "on"
    family = loaded_family(api)
    if family is not None:
        return family == "qwen3.5"
    return "qwen3.5" in api.model.lower()


_families: dict[str, str | None] = {}


def loaded_family(api: Api) -> str | None:
    """The catalog family /api/ps reports for api.model, or None if it is not the loaded model.

    The server reads the switch for every model on the Qwen3.5 frontend, whose catalog family
    is qwen3.5 whatever the tag says (qwen3.6-oga-hipep, qwen3.8-oga-hipep, ...).
    """
    if _families.get(api.model) is None:
        family = None
        try:
            url = api.url.removesuffix("/v1/chat/completions") + "/api/ps"
            with urllib.request.urlopen(url, timeout=api.timeout) as reply:
                for loaded in json.load(reply).get("models", []):
                    if loaded.get("model") == api.model or loaded.get("name") == api.model:
                        family = (loaded.get("details") or {}).get("family") or None
        except (OSError, ValueError):
            family = None
        # Not cached while unknown: the model may not be loaded yet.
        _families[api.model] = family
    return _families[api.model]


def not_nested(api: Api, reason: str) -> None:
    """Skip a nested-switch case on a model that ignores the switch.

    A model declared to read it fails instead of skipping, so a server that ignores the
    field cannot pass vacuously.
    """
    if reads_nested_switch(api):
        raise AssertionError(reason + " (this model is declared to read the nested switch)")
    raise Skip(reason)


def skip_always_thinking(api: Api) -> None:
    """Skip a nested-switch case on a model that always reasons, unless it is declared to read the switch.

    The cases tell that a model reads the switch by its reasoning with enable_thinking=true,
    which such a model shows either way.
    """
    if api.always_thinks and not reads_nested_switch(api):
        raise Skip("the model always reasons, so its reasoning does not show that it reads the nested switch")


def nested_body(api: Api, enable_thinking, **fields) -> dict:
    """A request in the Qwen model-card shape: no native think, the switch in chat_template_kwargs."""
    body = api.body(user(PROMPT), max_tokens=THINK_BUDGET, **fields)
    del body["think"]
    body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return body


@case
def nested_thinking_switch(api: Api) -> None:
    """chat_template_kwargs.enable_thinking is think: on gives reasoning_content, off gives none.

    Only the Qwen3.5 packages read it; on any other model the field is ignored and this skips.
    """
    skip_always_thinking(api)
    on = api.ok(nested_body(api, True))["message"]
    if not on.get("reasoning_content"):
        not_nested(api, "no reasoning_content with enable_thinking=true: the model does not read the nested switch")
    off = api.ok(nested_body(api, False))["message"]
    if not api.always_thinks:
        expect("reasoning_content" not in off,
               f"enable_thinking=false still returned reasoning_content: {off.get('reasoning_content', '')[:120]!r}")
    expect(off["content"].strip() != "", "content is empty")


@case
def nested_thinking_switch_stream(api: Api) -> None:
    """The same switch on a stream: reasoning deltas with it on, none with it off."""
    skip_always_thinking(api)
    s = api.ok_stream(nested_body(api, True))
    on = flm_api.check_stream_contract(s, include_usage=False)
    if on.reasoning == "":
        not_nested(api, "no reasoning with enable_thinking=true: the model does not read the nested switch")
    off = flm_api.check_stream_contract(api.ok_stream(nested_body(api, False)), include_usage=False)
    if not api.always_thinks:
        expect(off.reasoning == "", f"enable_thinking=false streamed reasoning: {off.reasoning[:120]!r}")


@case
def nested_thinking_is_per_request(api: Api) -> None:
    """Off, on, then no switch at all: a request without it gets the model's default again."""
    skip_always_thinking(api)
    default = api.body(user(PROMPT), max_tokens=THINK_BUDGET)
    del default["think"]
    before = api.ok(default)["message"]
    if not api.ok(nested_body(api, True))["message"].get("reasoning_content"):
        not_nested(api, "the model does not read the nested switch")
    api.ok(nested_body(api, False))
    after = api.ok(default)["message"]
    expect(bool(before.get("reasoning_content")) == bool(after.get("reasoning_content")),
           "a request without the switch did not get the default back after enable_thinking=false")


@case
def nested_thinking_wrong_type_is_400(api: Api) -> None:
    """A non-boolean value, a non-object container or a contradicting native control is a 400 that names the field."""
    skip_always_thinking(api)
    probe = api.post(nested_body(api, True))
    if probe.status != 200 or not probe.body["choices"][0]["message"].get("reasoning_content"):
        not_nested(api, "the model does not read the nested switch")
    for bad in ("yes", 1):
        resp = api.post(nested_body(api, bad))
        expect(resp.status == 400, f"enable_thinking={bad!r}: expected 400, got {resp.status}")
        expect("chat_template_kwargs.enable_thinking" in resp.error_message(), resp.error_message())
    body = nested_body(api, True)
    body["chat_template_kwargs"] = "yes"
    resp = api.post(body)
    expect(resp.status == 400 and "chat_template_kwargs" in resp.error_message(), f"{resp.status} {resp.error_message()}")
    conflict = nested_body(api, True)
    conflict["think"] = False
    resp = api.post(conflict)
    expect(resp.status == 400 and "think" in resp.error_message(), f"{resp.status} {resp.error_message()}")
    # the stream is refused before any frame, and the next valid request still works
    s = api.stream(nested_body(api, "yes"))
    expect(s.status == 400, f"stream with a bad switch: expected 400, got {s.status}")
    api.ok(nested_body(api, True))


@case
def nested_thinking_off_with_json_object(api: Api) -> None:
    """A grammar-constrained request sees the effective setting: off leaves no room to reason."""
    skip_always_thinking(api)
    if not api.ok(nested_body(api, True))["message"].get("reasoning_content"):
        not_nested(api, "the model does not read the nested switch")
    body = nested_body(api, False, response_format={"type": "json_object"})
    body["messages"] = user('Reply with a JSON object {"answer": "yes" or "no"}: is 7 a prime number?')
    msg = flm_api.check_message_contract(api.ok(body))
    if not api.always_thinks:
        expect("reasoning_content" not in msg,
               f"enable_thinking=false with json_object still reasoned: {msg.get('reasoning_content', '')[:120]!r}")
    import json
    try:
        parsed = json.loads(msg["content"])
    except ValueError:
        raise AssertionError(f"content is not JSON: {msg['content'][:120]!r}")
    expect(isinstance(parsed, dict), f"content is not a JSON object: {msg['content'][:120]!r}")


@case
def nested_thinking_error_leaves_no_state(api: Api) -> None:
    """Off, then a refused conflict, then no switch: the refused request changed nothing."""
    skip_always_thinking(api)
    default = api.body(user(PROMPT), max_tokens=THINK_BUDGET)
    del default["think"]
    if not api.ok(nested_body(api, True))["message"].get("reasoning_content"):
        not_nested(api, "the model does not read the nested switch")
    before = api.ok(default)["message"]
    api.ok(nested_body(api, False))
    conflict = nested_body(api, False)
    conflict["think"] = True
    resp = api.post(conflict)
    expect(resp.status == 400, f"conflicting switches: expected 400, got {resp.status}")
    after = api.ok(default)["message"]
    expect(bool(before.get("reasoning_content")) == bool(after.get("reasoning_content")),
           "a request without the switch did not get the default back after a refused conflict")


@case
def nested_thinking_ignored_elsewhere(api: Api) -> None:
    """A model that does not read the nested switch ignores the field, whatever it holds."""
    if reads_nested_switch(api):
        raise Skip("this model reads the nested switch")
    for value in ("yes", {"enable_thinking": "yes"}):
        body = api.body(user(PROMPT), max_tokens=64)
        body["chat_template_kwargs"] = value
        resp = api.post(body)
        expect(resp.status == 200, f"chat_template_kwargs={value!r}: expected 200, got {resp.status} {resp.error_message()}")


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
