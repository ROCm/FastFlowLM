"""Opt-in real NPU/GPU residency test. Start one server with both models installed."""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def request(endpoint, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(endpoint + path, data,
                                  {"Content-Type": "application/json"}), timeout=180)


def inventory(ep):
    with request(ep, "/api/ps") as r:
        return {m["name"]: m["details"].get("device") for m in json.load(r)["models"]}


def chat(ep, model, stream=False, prompt="What is two plus two? Answer in one word."):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 12,
            "temperature": 0, "think": False, "stream": stream}
    if model:
        body["model"] = model
    begin = time.monotonic()
    with request(ep, "/v1/chat/completions", body) as response:
        if not stream:
            reply = json.load(response)
            assert reply["choices"][0]["message"]["content"] and reply["usage"]["completion_tokens"] > 0, reply
            return reply["model"], reply["choices"][0]["message"]["content"], None
        first, text, served, done, finished = None, "", None, False, False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            assert not chunk.get("error"), chunk
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    first = time.monotonic() - begin if first is None else first
                    served = chunk["model"]
                    text += delta.get("content") or ""
                finished |= choice.get("finish_reason") is not None
        assert done and finished and text and first is not None, (done, finished, text)
        return served, text, first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--npu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--limit", type=int, choices=(1, 2), required=True, help="expected server setting")
    p.add_argument("--gpu-first", action="store_true")
    p.add_argument("--third-gpu", help="installed model on the GPU, to replace the first GPU model")
    p.add_argument("--third-npu", help="installed replacement on the NPU; the GPU must survive")
    p.add_argument("--bad-frontend", help="installed fixture with a nonexistent frontend_provider")
    p.add_argument("--pressure", action="store_true", help="third GPU package deliberately exceeds coexistence budget")
    p.add_argument("--repeat", type=int, default=5)
    a = p.parse_args()
    ep = a.endpoint.rstrip("/")
    devices = {a.npu: "npu", a.gpu: "gpu"}
    order = list(devices)
    if a.gpu_first:
        order.reverse()
    reference = {}
    for m in order:
        served, reference[m], _ = chat(ep, m)
        assert served == m
    expected = devices if a.limit == 2 else {order[-1]: devices[order[-1]]}
    got = inventory(ep)
    assert got == expected, f"after warming both: expected {expected}, got {got}"
    assert chat(ep, None)[0] == order[-1]
    try:
        chat(ep, "unknown-wp42:not-installed")
    except urllib.error.HTTPError as err:
        assert err.code == 404, err.code
    else:
        raise AssertionError("unknown model succeeded")
    assert inventory(ep) == expected
    if a.bad_frontend:
        try:
            chat(ep, a.bad_frontend)
        except urllib.error.HTTPError as err:
            assert err.code in (400, 500), err.code
        else:
            raise AssertionError("invalid frontend provider succeeded")
        assert inventory(ep) == expected
    times = {m: [] for m in order}
    for _ in range(a.repeat):
        for m in order:
            served, text, ttft = chat(ep, m, True)
            assert served == m and text == reference[m], (m, served, text, reference)
            assert inventory(ep) == (devices if a.limit == 2 else {m: devices[m]})
            times[m].append(ttft)
    for m, values in times.items():
        print(f"{m} first token median={statistics.median(values):.3f}s "
              f"range={min(values):.3f}..{max(values):.3f}s n={len(values)}")
        if a.limit == 2:
            assert max(values) < 3, values
    if a.limit == 2:
        # Plain completions invalidates its own conversation. It must not reset the
        # other model, including on the cancelled stream's cleanup path.
        cached_body = {"model": a.gpu, "messages": [{"role": "user", "content":
                       "GPU cache isolation check. Write a sentence explaining why two plus four is six."}],
                       "think": False, "temperature": 0, "max_tokens": 24}
        with request(ep, "/v1/chat/completions", cached_body) as r:
            primed = json.load(r)["usage"]["prompt_tokens"]
        with request(ep, "/v1/completions", {"model": a.npu, "prompt": "Count:", "max_tokens": 4}) as r:
            assert json.load(r)["choices"]
        with request(ep, "/v1/chat/completions", cached_body) as r:
            usage = json.load(r)["usage"]
            assert usage["prompt_tokens_details"]["cached_tokens"] == primed == usage["prompt_tokens"] > 0, usage
        with request(ep, "/v1/chat/completions",
                     {"model": a.gpu, "stream": True, "max_tokens": 180,
                      "messages": [{"role": "user", "content": "List the integers from 1 to 100."}]}) as r:
            for line in r:
                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                    choices = json.loads(line[5:]).get("choices", [])
                    if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content") for c in choices):
                        break
            else:
                raise AssertionError("expected a generated token before disconnect")
        served, text, _ = chat(ep, a.npu)
        assert (served, text) == (a.npu, reference[a.npu]) and inventory(ep) == devices
        with request(ep, "/v1/responses", {"model": a.gpu, "input": "Say hello.", "max_output_tokens": 8}) as r:
            assert json.load(r)["model"] == a.gpu
    if a.third_gpu or a.third_npu:
        replacement, device, survivor = ((a.third_gpu, "gpu", a.npu) if a.third_gpu else (a.third_npu, "npu", a.gpu))
        assert chat(ep, replacement)[0] == replacement
        expected = {replacement: device}
        if a.limit == 2 and not a.pressure:
            expected[survivor] = devices[survivor]
        assert inventory(ep) == expected, (inventory(ep), expected)
        served, text, _ = chat(ep, survivor)
        assert (served, text) == (survivor, reference[survivor])
        assert inventory(ep).get(survivor) == devices[survivor]
    print("PASS residency and per-model output")


if __name__ == "__main__":
    main()
