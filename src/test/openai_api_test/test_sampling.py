#!/usr/bin/env python3
"""seed and the penalty window (repeat_last_n) on /v1/chat/completions.

Checks:

* the same request with the same ``seed`` samples the same text, even with an
  unseeded request in between, streamed or not;
* ``repeat_last_n: 0`` turns the penalties off: large penalties then change
  nothing;
* ``repeat_last_n: -1`` (the whole reply) and a window longer than any reply
  give the same text;
* malformed values get HTTP 400, and null means not sent.

Decoding on the NPU isn't always bit-reproducible between identical requests,
so the comparisons retry, and skip when plain greedy decoding doesn't reproduce
itself either.

Usage::

    python3 test_sampling.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect
from fixtures import user
from test_logprobs import expect_400

STORY = "Invent a name for a pet dragon and describe it in two sentences."
REPEAT = "Write the word apple twenty times, separated by spaces."
TRIES = 3


def no_penalties(**fields: Any) -> dict[str, Any]:
    """Penalty fields set to off, overridable; they stick, so every case sends them."""
    out: dict[str, Any] = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
    out.update(fields)
    return out


def sampled(api: Api, prompt: str, **fields: Any) -> dict[str, Any]:
    """A request that really samples: wide top_k, temperature 1."""
    return api.body(user(prompt), max_tokens=48,
                    **no_penalties(top_k=40, top_p=1.0, temperature=1.0, **fields))


def content(api: Api, body: dict[str, Any]) -> str:
    return flm_api.check_message_contract(api.ok(body)).get("content") or ""


def require_reproducible(api: Api) -> None:
    """Skip unless the model computes the same logits for identical requests.

    Logits that differ by a rounding step from run to run change the pick
    wherever two tokens are close, yet greedy text can still match over a short
    answer, so compare the logprobs.
    """
    body = api.body(user(STORY), max_tokens=48, logprobs=True, **no_penalties())
    resp = api.post(body)
    if resp.status != 200:
        # no logprobs on this model (the MTP speculative model): compare the text
        body.pop("logprobs")
        first = content(api, body)
        if all(content(api, body) == first for _ in range(TRIES)):
            return
        raise Skip("greedy decoding did not reproduce over identical requests")
    first = resp.body["choices"][0]["logprobs"]["content"]
    second = api.ok(body)["logprobs"]["content"]
    if [(e["token"], e["logprob"]) for e in first] != [(e["token"], e["logprob"]) for e in second]:
        raise Skip("the model computes different logits for identical requests, so no output is repeatable")


def expect_same(api: Api, body_a: dict[str, Any], body_b: dict[str, Any], what: str,
                between: dict[str, Any] | None = None) -> None:
    """body_a and body_b give the same content, allowing for NPU non-determinism."""
    seen: list[tuple[str, str]] = []
    for _ in range(TRIES):
        a = content(api, body_a)
        if between is not None:
            content(api, between)
        b = content(api, body_b)
        if a == b:
            return
        seen.append((a, b))
    require_reproducible(api)
    raise AssertionError(f"{what}:\n  first:  {seen[0][0]!r}\n  second: {seen[0][1]!r}")


@case
def seed_reproduces(api: Api) -> None:
    """The same seed samples the same text, whatever ran in between."""
    seeded = sampled(api, STORY, seed=1234)
    expect_same(api, seeded, seeded, "the same seed sampled different text",
                between=sampled(api, STORY))


@case
def seed_stream_matches_nonstream(api: Api) -> None:
    """A seeded stream folds to the seeded non-streaming answer."""
    body = sampled(api, STORY, seed=99)
    for _ in range(TRIES):
        plain = content(api, body)
        folded = flm_api.check_stream_contract(api.ok_stream(body), include_usage=False)
        if folded.content == plain:
            return
    require_reproducible(api)
    raise AssertionError(f"stream {folded.content!r} differs from non-stream {plain!r}")


@case
def window_zero_turns_penalties_off(api: Api) -> None:
    """With repeat_last_n 0, large penalties change nothing."""
    off = api.body(user(REPEAT), max_tokens=48, **no_penalties(repeat_last_n=0))
    heavy = api.body(user(REPEAT), max_tokens=48,
                     **no_penalties(repeat_last_n=0, presence_penalty=2.0, frequency_penalty=2.0))
    expect_same(api, off, heavy, "penalties applied with repeat_last_n 0")


@case
def window_whole_reply(api: Api) -> None:
    """-1 and a window longer than any reply both cover the whole reply."""
    fields = {"presence_penalty": 0.5, "frequency_penalty": 0.5}
    whole = api.body(user(REPEAT), max_tokens=48, **no_penalties(repeat_last_n=-1, **fields))
    huge = api.body(user(REPEAT), max_tokens=48, **no_penalties(repeat_last_n=2 ** 40, **fields))
    expect_same(api, whole, huge, "repeat_last_n -1 and 2**40 differ")


@case
def bad_repeat_last_n(api: Api) -> None:
    """repeat_last_n below -1, or not an integer, is refused."""
    expect_400(api, {"repeat_last_n": -2}, "repeat_last_n must be an integer")
    expect_400(api, {"repeat_last_n": 1.5}, "repeat_last_n must be an integer")
    expect_400(api, {"repeat_last_n": "64"}, "repeat_last_n must be an integer")


@case
def bad_seed(api: Api) -> None:
    """A seed that is not an integer is refused."""
    expect_400(api, {"seed": "7"}, "seed must be an integer")
    expect_400(api, {"seed": 1.5}, "seed must be an integer")


@case
def null_is_unset(api: Api) -> None:
    """seed and repeat_last_n sent as null mean not sent."""
    api.ok(api.body(user("Say hi."), max_tokens=8, seed=None, repeat_last_n=None))


@case
def negative_and_large_seeds(api: Api) -> None:
    """Any 64-bit integer is a seed, as at OpenAI."""
    for seed in (-1, 0, 2 ** 63 - 1, 2 ** 64 - 1):
        api.ok(api.body(user("Say hi."), max_tokens=8, seed=seed))


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
