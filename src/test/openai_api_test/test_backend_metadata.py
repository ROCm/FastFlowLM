"""FLM-specific backend metadata for a server running an OGA GPU model.

Run against an OGA-enabled FLM started with the named model. Not an L2h test.
"""

import json
import os
import urllib.request

import flm_api
from fixtures import user
from flm_api import Api, case, expect


def get(api: Api, path: str) -> dict:
    with urllib.request.urlopen(api.url.removesuffix("/v1/chat/completions") + path,
                                timeout=api.timeout) as reply:
        expect(reply.status == 200, f"{path}: expected HTTP 200, got {reply.status}")
        return json.load(reply)


@case
def oga_chat(api: Api) -> None:
    reply = api.post(api.body(user("Say hello."), max_tokens=16))
    expect(reply.status == 200, f"chat returned HTTP {reply.status}: {reply.body}")
    expect(reply.body.get("usage", {}).get("completion_tokens", 0) > 0, "no generated tokens")


@case
def ps_reports_gpu(api: Api) -> None:
    models = get(api, "/api/ps")["models"]
    loaded = next((m for m in models if m["name"] == api.model), None)
    expect(loaded is not None, f"{api.model} is not listed as loaded: {models}")
    expect(loaded["model"] == api.model and loaded["size"] > 0 and loaded["expires_at"],
           f"missing existing /api/ps fields: {loaded}")
    details = loaded["details"]
    expect(details.get("backend") == "oga-hipep" and details.get("device") == "gpu",
           f"unexpected /api/ps details: {details}")


@case
def models_reports_gpu(api: Api) -> None:
    reply = get(api, "/v1/models")
    expect(reply["object"] == "list", f"unexpected models envelope: {reply}")
    entry = next((m for m in reply["data"] if m["id"] == api.model), None)
    expect(entry is not None, f"/v1/models omits {api.model}")
    expect(entry["object"] == "model" and entry["owned_by"] == "FastFlowLM" and entry["created"] > 0,
           f"missing existing model fields: {entry}")
    expect(entry.get("backend") == "oga-hipep" and entry.get("device") == "gpu",
           f"unexpected /v1/models entry: {entry}")


@case
def refused_npu_switch_keeps_gpu(api: Api) -> None:
    npu_tag = os.environ.get("FLM_TEST_UNAVAILABLE_NPU_MODEL")
    if not npu_tag:
        raise flm_api.Skip("set FLM_TEST_UNAVAILABLE_NPU_MODEL on an NPU-disabled host")
    body = api.body(user("Say hello."), max_tokens=16)
    warm = api.post(body)
    expect(warm.status == 200, f"could not warm GPU conversation: {warm}")
    ps_reports_gpu(api)
    refused = api.post(dict(body, model=npu_tag))
    expect(refused.status == 500, f"expected a known but unavailable model: {refused}")
    ps_reports_gpu(api)  # Check before any subsequent chat could reload the GPU model.
    expect("No NPU device available" in refused.body["error"]["message"], str(refused.body))
    del body["model"]
    reply = api.post(dict(body, backend="oga-hipep"))
    expect(reply.status == 200 and reply.body.get("usage", {}).get("completion_tokens", 0) > 0,
           f"chat without a model did not continue on the loaded GPU: {reply}")
    expect(reply.body["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0,
           "naming the already-selected backend or refusing the switch dropped the warm GPU cache")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
