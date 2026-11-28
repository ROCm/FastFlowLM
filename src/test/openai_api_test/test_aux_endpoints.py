#!/usr/bin/env python3
"""Inspect model availability and a resident model's prompt without inference."""
import os
from urllib.parse import quote

import flm_api
from flm_api import Api, Skip, case, expect
from test_http_safety import call


@case
def discovery(api: Api):
    def get(path):
        status, _, body = call(api, "GET", path)
        expect(status == 200, f"{path}: {status} {str(body)[:180]}")
        return body

    catalog = get("/v1/models")["data"]
    expect(all(isinstance(r.get("installed"), bool) for r in catalog), "missing installed flag")
    installed = get("/v1/models?installed=true")["data"]
    expect([r["id"] for r in installed] == [r["id"] for r in catalog if r["installed"]],
           "filter and item flags disagree")
    status, _, body = call(api, "GET", "/v1/models?installed=maybe")
    expect(status == 400 and body["error"]["param"] == "installed", str(body))
    row = catalog[0]
    detail = get("/v1/models/" + quote(row["id"], safe=""))
    expect({k: v for k, v in detail.items() if k != "created"} ==
           {k: v for k, v in row.items() if k != "created"}, "detail differs from list")
    family = row["id"].split(":", 1)[0]
    expect(get("/v1/models/" + quote(family, safe=""))["id"].startswith(family + ":"),
           "family alias does not resolve to a catalog ID")
    status, _, body = call(api, "GET", "/v1/models/no-such-wp11-model")
    expect(status == 404 and body["error"]["param"] == "model", str(body))


@case
def resident_prompt(api: Api):
    if api.model not in ("qwen3:4b", "qwen3:0.6b") or os.getenv("FLM_TEST_NO_ENGINE") == "1":
        raise Skip("requires a loaded built-in Qwen3 frontend")
    status, _, body = call(api, "GET", "/health")
    expect(status == 200 and body == {"status": "ok"}, str(body))

    def post(path, **fields):
        status, _, body = call(api, "POST", path, {"model": api.model, **fields})
        expect(status == 200, f"{path}: {status} {body}")
        return body

    prompt = post("/apply-template", messages=[{"role": "user", "content": "Hello 世界!"}])["prompt"]
    expect("Hello 世界!" in prompt, prompt)
    nulls = post("/apply-template", messages=[{"role": "user", "content": "Hello 世界!"}],
                 tools=None, think=None, model=None)["prompt"]
    expect(nulls == prompt, "null options changed the prompt")
    if api.model == "qwen3:4b":
        messages = [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Hello"}]
        tool = {"type": "function", "function": {"name": "diagnostic_lookup",
                "parameters": {"type": "object", "properties": {}}}}
        shown = post("/apply-template", messages=messages, tools=[tool], think=False)["prompt"]
        hidden = post("/apply-template", messages=messages, tools=[tool], tool_choice="none")["prompt"]
        expect("diagnostic_lookup" in shown and "diagnostic_lookup" not in hidden, "tool visibility")
        expect("Be brief." in shown, "lost developer instruction")
    text = "Hello 世界! 🤗"
    tokens = post("/tokenize", content=text)["tokens"]
    expect(post("/tokenize", content=text, with_pieces=None, parse_special=None,
                add_special=None)["tokens"] == tokens, "null flags changed token IDs")
    pieces = post("/tokenize", content=text, with_pieces=True)["tokens"]
    expect(tokens and [p["id"] for p in pieces] == tokens, str(pieces))
    expect(post("/detokenize", tokens=tokens)["content"] == text, str(tokens))
    for fields, param in (({"tool_choice": "bad", "messages": [{"role": "user", "content": "hi"}]}, "tool_choice"),
                          ({"messages": [{"role": "user", "content": [{"type": 123, "text": "hi"}]}]}, "messages"),
                          ({"messages": [{"role": "user", "content": "hi", "images": ["unused"]}]}, "messages")):
        status, _, body = call(api, "POST", "/apply-template", fields)
        expect(status == 400 and body["error"]["param"] == param, str(body))
    for path, fields in (("/tokenize", {"parse_special": False, "content": "hello"}),
                         ("/tokenize", {"add_special": True, "content": "hello"}),
                         ("/detokenize", {"tokens": [-1]}),
                         ("/apply-template", {"messages": [{"role": "user", "content":
                             [{"type": "image_url", "image_url": {"url": "unused"}}]}]})):
        status, _, body = call(api, "POST", path, fields)
        expect(status == 400 and body["error"].get("param"), f"{status}: {body}")


CASES = flm_api.collect(globals())
if __name__ == "__main__":
    flm_api.main(CASES, __doc__)
