"""Every chat model answers, streams, holds a second turn and, if it has a
vision encoder, sees.

Replaces the old per-model sweeps, which only checked that a reply arrived:
each check here has a known answer, so a model that loads but emits garbage
fails. One plain server serves the whole suite; the model switches with each
request, which also exercises unload/load between engines.
"""

from __future__ import annotations

import base64
from pathlib import Path

from ..harness import ServerGone, expect, skip, warn
from ..server import ServerError

NAME = "chat"
DESCRIPTION = "per-model answer, streaming, two turns, non-streamed usage, vision"

# Greedy, so the content checks see the model's single most likely answer.
# FLM keeps sampling and think settings on the loaded model between requests,
# so every request sends them rather than relying on what came before.
GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0}

ANSWER_PROMPT = "What is the capital of France? Answer with one word."
ANSWER_WORD = "paris"

TURNS = ("Name one primary color. Answer in one short sentence.",
         "Now name a different primary color. Answer in one short sentence.")
NON_STREAM_PROMPT = "Name the largest planet in the solar system. Answer in one short sentence."

# Translators answer every prompt with a translation, so they get one to do.
TRANSLATE_TEXT = "Thank you very much."
TRANSLATE_WORD = "merci"
TRANSLATE_TURNS = ("Thank you very much.", "Good night.")

COLOR_PROMPT = "What color is the square? Answer with one word."
COLOR_IMAGES = {"red": "vision/red_square.png", "blue": "vision/blue_square.png"}
PHOTO = "test_image1.jpg"
PHOTO_PROMPT = "Describe this image in one sentence."

# Token budgets. A model that always reasons spends hundreds of tokens before
# any content, so it gets a larger one; everything else stays short to keep
# the NPU runners free.
ANSWER_TOKENS = 64
TURN_TOKENS = 128
THINKING_TOKENS = 1024

# lfm2-trans is LFM2-2.6B-Transcript, a meeting-transcript summarizer with
# its own input format, not a translator; plan.Model.translation matches it
# by id prefix anyway. It gets the plain question, and a wrong answer is only
# a warning, since the question is outside what it was tuned for.
OFF_FORMAT_PREFIXES = ("lfm2-trans",)

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def run(ctx) -> None:
    models = ctx.plan.chat()
    if not models:
        return
    try:
        with ctx.server("chat") as server:
            for index, model_id in enumerate(models):
                try:
                    _model_cases(ctx, server, ctx.plan.model(model_id))
                except ServerGone as exc:
                    ctx.not_attempted(NAME, models[index + 1:], str(exc))
                    raise
    except ServerError as exc:
        ctx.not_attempted(NAME, models, f"server did not start: {exc}")
        raise


def _model_cases(ctx, server, model) -> None:
    client = server.client
    # A model that crashed the server once is likely to again; its remaining
    # cases are dropped to leave the restart budget for the other models.
    restarts = server.restarts

    def crashed():
        return server.restarts > restarts

    thinks = model.always_thinks
    translator = model.translation and not model.id.startswith(OFF_FORMAT_PREFIXES)

    def chat(messages, max_tokens, stream=True):
        return client.chat(model.id, messages, stream=stream, think=False,
                           max_tokens=THINKING_TOKENS if thinks else max_tokens, **GREEDY)

    def check_reply(reply, what):
        expect(reply.finish_reason, f"{what}: no finish_reason")
        if thinks:
            expect(reply.content.strip() or reply.reasoning.strip(),
                   f"{what}: no content or reasoning")
        else:
            expect(reply.content.strip(), f"{what}: empty content")

    with ctx.case(NAME, "answer", model.id, server) as case:
        if translator:
            prompt, word = _translate_prompt(model.id, TRANSLATE_TEXT), TRANSLATE_WORD
        else:
            prompt, word = ANSWER_PROMPT, ANSWER_WORD
        reply = chat([{"role": "user", "content": prompt}], ANSWER_TOKENS)
        case.data.update(_timing(reply), content=reply.content[:200])
        if not reply.content.strip() and thinks and reply.finish_reason == "length":
            # Whether the reasoning fits the budget is the model's doing, not
            # the server's; a reasoning stream that ends cleanly with no
            # answer still fails below.
            warn(f"reasoning used all {THINKING_TOKENS} tokens before an answer")
        expect(reply.content.strip(), "empty content")
        if word not in reply.content.lower():
            message = f"expected {word!r} in {reply.content[:120]!r}"
            if model.id.startswith(OFF_FORMAT_PREFIXES):
                warn(message)
            raise AssertionError(message)

    if crashed():
        return
    with ctx.case(NAME, "stream-two-turns", model.id, server) as case:
        turns = ([_translate_prompt(model.id, t) for t in TRANSLATE_TURNS] if translator
                 else list(TURNS))
        messages = [{"role": "user", "content": turns[0]}]
        first = chat(messages, TURN_TOKENS)
        case.data["turn1"] = _timing(first)
        check_reply(first, "turn 1")
        if model.single_turn:
            skip("turn 1 passed; turn 2 not sent to a single-turn model")
        messages += [{"role": "assistant", "content": first.content or first.reasoning[:500]},
                     {"role": "user", "content": turns[1]}]
        second = chat(messages, TURN_TOKENS)
        case.data["turn2"] = _timing(second)
        check_reply(second, "turn 2")
        case.data["cached_tokens"] = second.cached_tokens
        if not second.cached_tokens:
            # Reusing the first turn's KV cache is an optimisation not every
            # engine implements, so its absence is worth seeing, not failing.
            warn(f"turn 2 reported cached_tokens={second.cached_tokens}")

    if crashed():
        return
    with ctx.case(NAME, "non-stream", model.id, server) as case:
        prompt = (_translate_prompt(model.id, TRANSLATE_TEXT) if translator
                  else NON_STREAM_PROMPT)
        reply = chat([{"role": "user", "content": prompt}], TURN_TOKENS, stream=False)
        case.data.update(_timing(reply))
        check_reply(reply, "non-stream")
        usage = reply.usage
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        expect(isinstance(prompt_tokens, int) and prompt_tokens > 0,
               f"usage.prompt_tokens is {prompt_tokens!r}")
        expect(isinstance(completion_tokens, int) and completion_tokens > 0,
               f"usage.completion_tokens is {completion_tokens!r}")
        expect(usage.get("total_tokens") == prompt_tokens + completion_tokens,
               f"usage.total_tokens {usage.get('total_tokens')!r} != "
               f"{prompt_tokens} + {completion_tokens}")

    if not model.vlm or crashed():
        return
    with ctx.case(NAME, "vision", model.id, server) as case:
        if model.translation:
            # A translator answers the question with its translation, so the
            # colour word is not a fair check; the photo case still runs.
            skip("translation model: does not answer questions about images")
        for color, relative in COLOR_IMAGES.items():
            reply = chat([_image_message(ctx.asset(relative), COLOR_PROMPT)], ANSWER_TOKENS)
            case.data[color] = {**_timing(reply), "content": reply.content[:120]}
            expect(reply.content.strip(), f"{color} square: empty content")
            expect(color in reply.content.lower(),
                   f"{color} square: answer was {reply.content[:120]!r}")

    if crashed():
        return
    with ctx.case(NAME, "vision-photo", model.id, server) as case:
        reply = chat([_image_message(ctx.asset(PHOTO), PHOTO_PROMPT)], TURN_TOKENS)
        case.data.update(_timing(reply), content=reply.content[:200])
        check_reply(reply, "photo")


def _translate_prompt(model_id: str, text: str) -> str:
    """The prompt format each translator's model card documents."""
    if model_id.startswith("translategemma"):
        return ("You are a professional English (en) to French (fr) translator. Your goal "
                "is to accurately convey the meaning and nuances of the original English "
                "text while adhering to French grammar, vocabulary, and cultural "
                "sensitivities.\nProduce only the French translation, without any "
                "additional explanations or commentary. Please translate the following "
                f"English text into French:\n\n\n{text}")
    return f"Translate the following segment into French, without additional explanation.\n\n{text}"


def _image_message(path: Path, prompt: str) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    url = f"data:{MIME.get(path.suffix.lower(), 'image/jpeg')};base64,{encoded}"
    return {"role": "user", "content": [{"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": url}}]}


def _timing(reply) -> dict:
    """The server's load/prefill/decode accounting for one request."""
    keys = ("prompt_tokens", "completion_tokens", "load_duration", "prefill_duration_ttft",
            "decoding_duration", "decoding_speed_tps")
    out = {k: reply.usage[k] for k in keys if isinstance(reply.usage.get(k), (int, float))}
    out["seconds"] = round(reply.seconds, 2)
    return out
