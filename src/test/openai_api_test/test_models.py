#!/usr/bin/env python3
"""What `GET /v1/models` says each model can do, and that the server does what it says.

A bring-your-own-key client fills in a model's properties from this listing: whether it calls
tools, takes images and thinks, and how long its context is. Each entry keeps the OpenAI fields
(`id`, `object`, `created`, `owned_by`) and adds:

* ``capabilities``: Ollama's words, ``completion`` first (``tools``, ``vision``, ``audio``,
  ``thinking``, ``embedding``);
* ``context_length``: the context the server will use, `--ctx-len` when it was started with it
  (a single-turn model keeps its own);
* ``architecture.input_modalities`` / ``output_modalities`` and ``supported_parameters``
  (``tools``, ``tool_choice``, ``reasoning``): the same facts in OpenRouter's names.

The listing cases need no model and also run on the HTTP harness. The last two cases send a
request to the model named on the command line and need the engine: set
``FLM_TEST_NO_ENGINE=1`` to skip them on the harness. ``FLM_TEST_CTX_LEN=<n>`` declares the
`--ctx-len` the server was started with, so a server that ignores it fails instead of the case
being skipped.

Usage::

    python3 test_models.py <model> [--endpoint URL] [-k NAME] [-v]
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import flm_api
from fixtures import user
from flm_api import Api, Skip, case, expect

# single-turn models: their context is fixed (src/model_list.json, label "single-turn")
SINGLE_TURN = {"gemma4e-flash:e2b", "gemma4e-flash:e4b", "qwen3vl-flash:4b",
               "nemotron3.5-bolt-dflash-flash:e4b", "hy-mt2-flash:1.8b"}
WORDS = {"completion", "tools", "vision", "audio", "thinking", "embedding"}
PARAMETERS = {"tools", "tool_choice", "reasoning"}
# a 1x1 PNG
PIXEL = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
         "hKmMIQAAAABJRU5ErkJggg==")
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}


def listing(api: Api) -> list[dict[str, Any]]:
    url = api.url.removesuffix("/chat/completions") + "/models"
    with urllib.request.urlopen(url, timeout=api.timeout) as reply:
        expect(reply.status == 200, f"/v1/models: expected HTTP 200, got {reply.status}")
        body = json.load(reply)
    expect(body.get("object") == "list" and isinstance(body.get("data"), list),
           f"unexpected models envelope: {str(body)[:200]}")
    expect(len(body["data"]) > 0, "/v1/models lists no model")
    return body["data"]


def entry_of(api: Api, model: str) -> dict[str, Any] | None:
    return next((m for m in listing(api) if m.get("id") == model), None)


def strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


@case
def every_entry_has_the_fields(api: Api) -> None:
    for entry in listing(api):
        name = entry.get("id")
        expect(isinstance(name, str) and name, f"entry without an id: {entry}")
        expect(entry.get("object") == "model" and entry.get("owned_by") == "FastFlowLM"
               and isinstance(entry.get("created"), int),
               f"{name}: the OpenAI fields changed: {entry}")
        capabilities = entry.get("capabilities")
        expect(strings(capabilities) and set(capabilities) <= WORDS,
               f"{name}: capabilities is not a list of known words: {capabilities}")
        architecture = entry.get("architecture")
        expect(isinstance(architecture, dict) and strings(architecture.get("input_modalities"))
               and strings(architecture.get("output_modalities")),
               f"{name}: architecture has no modality lists: {architecture}")
        parameters = entry.get("supported_parameters")
        expect(strings(parameters) and set(parameters) <= PARAMETERS,
               f"{name}: supported_parameters is not a list of known names: {parameters}")
        expect("api_capabilities" not in entry, f"{name}: api_capabilities is reserved")
        if "completion" in capabilities:
            expect(capabilities[0] == "completion", f"{name}: completion is not first")
            context = entry.get("context_length")
            expect(isinstance(context, int) and not isinstance(context, bool) and context > 0,
                   f"{name}: context_length is not a positive integer: {context!r}")
            expect("text" in architecture["input_modalities"]
                   and architecture["output_modalities"] == ["text"],
                   f"{name}: a chat model reads and writes text: {architecture}")
        # the two vocabularies are one record: they cannot disagree
        expect(("vision" in capabilities) == ("image" in architecture["input_modalities"]),
               f"{name}: vision and input_modalities disagree: {entry}")
        expect(("audio" in capabilities) == ("audio" in architecture["input_modalities"]),
               f"{name}: audio and input_modalities disagree: {entry}")
        expect(("tools" in capabilities) == ("tools" in parameters)
               == ("tool_choice" in parameters),
               f"{name}: tools and supported_parameters disagree: {entry}")
        expect(("thinking" in capabilities) == ("reasoning" in parameters),
               f"{name}: thinking and supported_parameters disagree: {entry}")


@case
def a_vision_model_lists_images_and_a_text_model_does_not(api: Api) -> None:
    seen = 0
    for model, image, tools in (("qwen3.5:4b", True, True), ("llama3.2:1b", False, False)):
        entry = entry_of(api, model)
        if entry is None:      # not offered by this build or on this silicon
            continue
        seen += 1
        modalities = entry.get("architecture", {}).get("input_modalities")
        capabilities = entry.get("capabilities")
        expect(isinstance(modalities, list) and isinstance(capabilities, list),
               f"{model}: no capability fields: {entry}")
        expect(("image" in modalities) == image, f"{model}: input_modalities {modalities}")
        expect(("vision" in capabilities) == image, f"{model}: capabilities {capabilities}")
        expect(("tools" in capabilities) == tools, f"{model}: capabilities {capabilities}")
    if not seen:
        raise Skip("this server lists neither qwen3.5:4b nor llama3.2:1b")


@case
def context_length_follows_ctx_len(api: Api) -> None:
    declared = os.environ.get("FLM_TEST_CTX_LEN")
    if not declared:
        raise Skip("set FLM_TEST_CTX_LEN to the --ctx-len the server was started with")
    followed = 0
    for entry in listing(api):
        if "completion" not in entry.get("capabilities", ["completion"]):
            continue
        if entry["id"] in SINGLE_TURN:   # the loader keeps these at their own context
            continue
        followed += 1
        expect(entry.get("context_length") == int(declared),
               f"{entry['id']}: context_length {entry.get('context_length')}, "
               f"the server was started with --ctx-len {declared}")
    expect(followed > 0, "the listing has no chat model to check")


def needs_engine(api: Api) -> dict[str, Any]:
    if os.environ.get("FLM_TEST_NO_ENGINE") == "1":
        raise Skip("FLM_TEST_NO_ENGINE=1: no model can be loaded on this server")
    entry = entry_of(api, api.model)
    expect(entry is not None, f"/v1/models omits the served model {api.model}")
    return entry


@case
def a_model_listed_with_tools_answers_a_forced_call(api: Api) -> None:
    entry = needs_engine(api)
    reply = api.post(api.body(user("What is the weather in Paris?"), max_tokens=128,
                              tools=[WEATHER_TOOL], tool_choice="required"))
    if "tools" in entry["capabilities"]:
        expect(reply.status == 200, f"listed with tools, but a forced call got HTTP "
               f"{reply.status}: {str(reply.body)[:200]}")
        calls = reply.body["choices"][0]["message"].get("tool_calls")
        expect(bool(calls), f"listed with tools, but no tool call came back: {reply.body}")
    elif reply.status == 200:
        # families whose prompt never shows the tools still answer a forced call from the
        # grammar alone; whether they should refuse it is tool calling's change to make
        raise Skip("not listed with tools, yet a forced call was answered with 200")
    else:
        expect(reply.status == 400, f"listed without tools, but a forced call got HTTP "
               f"{reply.status}: {str(reply.body)[:200]}")


@case
def a_model_listed_with_image_input_accepts_an_image(api: Api) -> None:
    entry = needs_engine(api)
    message = {"role": "user", "content": [
        {"type": "text", "text": "Describe the image in one word."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PIXEL}},
    ]}
    if "image" not in entry["architecture"]["input_modalities"]:
        # today a text model answers 200 and never sees the image (test_images.py); refusing
        # it is media validation's change to make, so there is nothing to hold it to yet
        raise Skip("not listed with image input; a text model still answers an image with 200")
    reply = api.post(api.body([message], max_tokens=16))
    expect(reply.status == 200, f"listed with image input, but an image got HTTP "
           f"{reply.status}: {str(reply.body)[:200]}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
