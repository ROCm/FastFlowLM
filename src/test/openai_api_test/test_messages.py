"""Anthropic Messages acceptance and route-specific errors."""
import urllib.request

import flm_api
from flm_api import Api, case, expect
from test_http_safety import cors_mode


def post(api, **changes):
    body = {"model": api.model, "max_tokens": 48, "messages": [{"role": "user", "content": "Say OK."}]}
    return api.post({**body, **changes}, url=api.url.removesuffix("/chat/completions") + "/messages")


@case
def messages_errors(api: Api):
    for field, changes in [
        ("thinking.budget_tokens", {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 7}}),
        ("thinking.budget_tokens", {"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        ("stop_sequences", {"stop_sequences": ["STOP"]}),
        ("tools[0].type", {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        ("messages[0].content[0]", {"messages": [{"role":"user","content":[{"type":"document"}]}]}),
    ]:
        reply = post(api, stream=True, **changes)
        expect(reply.status == 400 and reply.body.get("type") == "error" and
               reply.body.get("error", {}).get("type") == "invalid_request_error" and
               field in reply.body["error"]["message"], f"{field}: {reply.status} {reply.body}")


@case
def messages_text(api: Api):
    reply = post(api)
    expect(reply.status == 200 and reply.body["type"] == "message" and
           reply.body["role"] == "assistant" and reply.body["content"], str(reply.body))


@case
def messages_unknown_model(api: Api):
    reply = post(api, model="wp47-no-such-model:1b", stream=True)
    expect(reply.status == 404 and reply.body.get("type") == "error" and
           reply.body.get("error", {}).get("type") == "not_found_error" and
           "model" in reply.body["error"]["message"], str(reply.body))


@case
def messages_browser_preflight(api: Api):
    url = api.url.removesuffix("/chat/completions") + "/messages"
    req = urllib.request.Request(url, method="OPTIONS", headers={
        "Origin": "http://localhost", "Access-Control-Request-Method": "POST"})
    with urllib.request.urlopen(req, timeout=10) as reply:
        headers = reply.headers.get("Access-Control-Allow-Headers", "").lower()
        if cors_mode(api) == "off":
            expect(not headers, headers)
        else:
            expect("anthropic-version" in headers and "x-api-key" in headers, headers)


CASES = flm_api.collect(globals())
if __name__ == "__main__":
    flm_api.main(CASES, __doc__)
