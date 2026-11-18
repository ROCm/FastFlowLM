#!/usr/bin/env python3
"""Server log privacy: user text and model replies stay out of the log unless FLM_LOG_CONTENT=1.

Each case sends a request carrying a marker string where a client puts text (message content,
system prompt, prompt, input, response_format, metadata, the URL query), then reads the server's
log and looks for it.

The test reads the log the server writes, so it needs to know where that is and how the
server was started. Both are declared, so a server in the other mode fails instead of passing:

* ``FLM_TEST_SERVER_LOG=<file>``: the file the server's stdout goes to;
* ``FLM_TEST_LOG_CONTENT=0``: started without ``FLM_LOG_CONTENT`` (the default): no marker and
  no ``Model RAW Output:`` dump may reach the log;
* ``FLM_TEST_LOG_CONTENT=1``: started with ``FLM_LOG_CONTENT=1``: the markers and the dump do.

The embeddings case needs an embedding model (``flm serve <model> --embed 1``); it is skipped
when the server answers 4xx for it.

Usage::

    FLM_TEST_SERVER_LOG=serve.log FLM_TEST_LOG_CONTENT=0 python3 test_log_privacy.py qwen3:0.6b [--endpoint URL]
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect

RAW_MARK = "Model RAW Output:"


def declared() -> tuple[str, bool]:
    path = os.environ.get("FLM_TEST_SERVER_LOG", "")
    mode = os.environ.get("FLM_TEST_LOG_CONTENT", "")
    if not path or mode not in ("0", "1"):
        raise Skip("set FLM_TEST_SERVER_LOG and FLM_TEST_LOG_CONTENT=0|1 (see the module docstring)")
    return path, mode == "1"


def log_size(path: str) -> int:
    return os.path.getsize(path)


def log_since(path: str, offset: int) -> str:
    time.sleep(0.5)  # the server prints its last lines after the response
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def post(api: Api, path: str, body: dict[str, Any]) -> int:
    base = api.url[: -len("/v1/chat/completions")]
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=api.timeout) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        e.read()
        return e.code


def check(api: Api, path: str, make_body, *, generates: bool, query: bool = False) -> None:
    log, content_on = declared()
    marker = "zqx" + uuid.uuid4().hex[:12]
    start = log_size(log)
    status = post(api, path + ("?note=" + marker if query else ""), make_body(marker))
    if status >= 400 and path.endswith("embeddings"):
        raise Skip(f"{path} answered {status} (no embedding model loaded?)")
    expect(status == 200, f"{path} answered {status}")
    text = log_since(log, start)
    # the request log may cut long text to its ends: look for the marker's first half too
    found = marker[:9] in text
    if content_on:
        expect(found, f"FLM_LOG_CONTENT=1 but the marker of {path} is not in the log")
        if generates:
            expect(RAW_MARK in text, f"FLM_LOG_CONTENT=1 but {path} added no '{RAW_MARK}' dump")
    else:
        expect(not found, f"the log carries user text from {path}")
        expect(RAW_MARK not in text, f"the log carries a '{RAW_MARK}' dump of the {path} reply")


@case
def test_chat_text_is_not_logged(api: Api) -> None:
    check(api, "/v1/chat/completions", lambda m: {
        "model": api.model, "max_tokens": 4,
        "messages": [{"role": "system", "content": m + " be brief"}, {"role": "user", "content": "say " + m}]},
        generates=True)


@case
def test_chat_parts_and_schema_are_not_logged(api: Api) -> None:
    check(api, "/v1/chat/completions", lambda m: {
        "model": api.model, "max_tokens": 4, "stream": True,
        "response_format": {"type": "json_schema", "json_schema": {"name": m, "schema": {"type": "object", "description": m}}},
        "messages": [{"role": "user", "content": [{"type": "text", "text": m}]}]},
        generates=True)


@case
def test_setting_named_keys_and_query_are_not_logged(api: Api) -> None:
    # "role" and "detail" name settings elsewhere in a request; here they carry text
    check(api, "/v1/chat/completions", lambda m: {
        "model": api.model, "max_tokens": 4, "metadata": {"role": m, "detail": m + " more"},
        "messages": [{"role": "user", "content": "hi"}]},
        generates=True, query=True)


@case
def test_completions_prompt_is_not_logged(api: Api) -> None:
    check(api, "/v1/completions", lambda m: {"model": api.model, "max_tokens": 4, "prompt": "say " + m},
          generates=True)


@case
def test_generate_prompt_and_system_are_not_logged(api: Api) -> None:
    check(api, "/api/generate", lambda m: {
        "model": api.model, "stream": False, "options": {"num_predict": 4}, "system": m, "prompt": "say " + m},
        generates=True)


@case
def test_embeddings_input_is_not_logged(api: Api) -> None:
    check(api, "/v1/embeddings", lambda m: {"model": "embed-gemma:300m", "input": m}, generates=False)


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
