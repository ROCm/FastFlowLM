"""Anthropic Messages acceptance and route-specific errors."""
import flm_api
import os
from flm_api import Api, case, expect
from test_http_safety import call, cors_mode


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
        ("messages[1].role", {"messages": [{"role":"user","content":"Hello"},{"role":"assistant","content":"Pre"}]}),
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
def messages_manual_budget(api: Api):
    """Slow NPU fixture: both requests must use the same prompt and total output limit."""
    if os.environ.get("FLM_TEST_MESSAGES_BUDGET") != "1":
        raise flm_api.Skip("set FLM_TEST_MESSAGES_BUDGET=1 on the Qwen3-4B FLM NPU backend")
    prompt = ("Find the last three digits of 7^777. Work through the modular arithmetic carefully and "
              "independently verify it with a second method in your reasoning. In your final answer, "
              "output ONLY the three digits, no explanation or working.")
    def run(n):
        reply = post(api, max_tokens=4096, temperature=0, messages=[{"role":"user","content":prompt}],
                     thinking={"type":"enabled","budget_tokens":n})
        expect(reply.status==200, f"budget {n} returned {reply.status}")
        blocks=reply.body["content"]
        reason="".join(b["thinking"] for b in blocks if b["type"]=="thinking")
        answer="".join(b["text"] for b in blocks if b["type"]=="text")
        expect(reason and answer and reply.body["usage"]["output_tokens"]>n, "did not return both stages")
        return reason
    low,high=run(1024),run(1536)
    expect(high.startswith(low) and len(high)>len(low), "the numeric budget did not change the reasoning boundary")


@case
def messages_unknown_model(api: Api):
    reply = post(api, model="wp47-no-such-model:1b", stream=True)
    expect(reply.status == 404 and reply.body.get("type") == "error" and
           reply.body.get("error", {}).get("type") == "not_found_error" and
           "model" in reply.body["error"]["message"], str(reply.body))


@case
def messages_unsupported_budget(api: Api):
    # Must refuse at ingress; this model must not be downloaded or made resident.
    reply = post(api, model="gpt-oss:20b", stream=True, max_tokens=2048,
                 thinking={"type": "enabled", "budget_tokens": 1024})
    if reply.status == 404:  # Not in this platform's catalog.
        raise flm_api.Skip("gpt-oss:20b not in the local catalog")
    expect(reply.status == 400 and reply.body.get("type") == "error" and
           "thinking.budget_tokens" in reply.body["error"]["message"], str(reply.body))


@case
def messages_browser_preflight(api: Api):
    status, headers, _ = call(api, "OPTIONS", "/v1/messages", headers={
        "Origin": "http://localhost", "Access-Control-Request-Method": "POST"})
    allowed = headers.get("access-control-allow-headers", "").lower()
    if cors_mode(api) == "off":
        expect(not allowed, allowed)
    else:
        expect(status == 200 and "anthropic-version" in allowed and "x-api-key" in allowed,
               f"{status}: {allowed}")


CASES = flm_api.collect(globals())
if __name__ == "__main__":
    flm_api.main(CASES, __doc__)
