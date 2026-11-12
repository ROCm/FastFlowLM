# Long-context tests for FLM#1001 against a running `flm serve`.
# usage: long_tests.py <port> <out.json> <case,case,...>
# Each case prints one line: name, prompt/completion tokens, finish reason, pass/fail, and why.
import json, sys, time, random, urllib.request, urllib.error

PORT, OUT, CASES = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
MODEL = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-mtp:27b"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def chat(messages, max_tokens, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "think": False}
    body.update(extra)
    t = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:400], "secs": time.time() - t}
    except Exception as e:
        return {"exception": repr(e), "secs": time.time() - t}
    r["secs"] = time.time() - t
    return r

# Filler: numbered records with distinct facts, so the model cannot guess a hidden one.
rng = random.Random(1001)
TOWNS = ["Arden", "Belmont", "Corvin", "Dunmore", "Elsing", "Farrow", "Galt", "Hollis", "Ivers", "Jarrow"]
def record(i):
    return (f"Record {i:05d}: the harbour office in {rng.choice(TOWNS)} logged {rng.randint(2, 98)} ships, "
            f"{rng.randint(10, 900)} tonnes of grain and a wind of {rng.randint(1, 40)} knots.\n")
RECORDS = [record(i) for i in range(4000)]
TOK_PER_RECORD = 40.0   # refined by calibrate()

def calibrate():
    global TOK_PER_RECORD
    base = chat([{"role": "user", "content": "Say OK."}], 1)
    r = chat([{"role": "user", "content": "".join(RECORDS[:200]) + "Say OK."}], 1)
    TOK_PER_RECORD = (r["usage"]["prompt_tokens"] - base["usage"]["prompt_tokens"]) / 200
    print(f"calibrate: {TOK_PER_RECORD:.2f} tokens per record", flush=True)

def haystack(tokens, needle=None, at=0.5):
    n = int(tokens / TOK_PER_RECORD)
    recs = RECORDS[:n]
    if needle:
        recs = recs[:int(n * at)] + [needle] + recs[int(n * at):]
    return "".join(recs)

NEEDLE = "Special note: the passphrase for the north gate is AMBER-7351-QUILL.\n"
results = []
def report(name, r, ok, why=""):
    u = r.get("usage", {})
    fin = r["choices"][0]["finish_reason"] if "choices" in r else None
    text = r["choices"][0]["message"].get("content") or "" if "choices" in r else ""
    line = (f"{name}: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
            f"finish={fin} secs={r.get('secs', 0):.0f} {'PASS' if ok else 'FAIL'} {why}")
    print(line, flush=True)
    results.append({"case": name, "ok": ok, "why": why, "usage": u, "finish": fin, "secs": r.get("secs"),
                    "text": text[-600:], "error": r.get("http_error") or r.get("exception"), "body": r.get("body")})
    json.dump(results, open(OUT, "w"), indent=1)

def sane(text):
    # garbage check: no long runs of one character, mostly printable
    return text and not any(c * 12 in text for c in "/!.-_ ?") and sum(ch.isprintable() or ch in "\n\t" for ch in text) > 0.98 * len(text)

def total(r):
    u = r.get("usage", {})
    return (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)

def needle_case(name, tokens, at):
    q = haystack(tokens, NEEDLE, at) + "\nWhat is the passphrase for the north gate? Answer with the passphrase only."
    r = chat([{"role": "user", "content": q}], 32)
    ok = "choices" in r and "AMBER-7351-QUILL" in (r["choices"][0]["message"].get("content") or "")
    report(name, r, ok, "" if ok else "needle not found")

def fill_case(name, tokens, max_tokens, limit, **extra):
    # write until the context is full: must end with finish=length at the limit, text sane
    q = haystack(tokens) + "\nNow write a very long story about one of these harbours; keep going for as long as you can."
    r = chat([{"role": "user", "content": q}], max_tokens, **extra)
    if "choices" not in r:
        return report(name, r, False, f"no reply: {r.get('http_error') or r.get('exception')} {r.get('body', '')[:200]}")
    text = r["choices"][0]["message"].get("content") or ""
    fin = r["choices"][0]["finish_reason"]
    # the last sampled token is returned but never written to the cache, so usage can be limit + 1
    ok = fin == "length" and limit - 16 <= total(r) <= limit + 1 and sane(text)
    report(name, r, ok, f"total={total(r)} limit={limit}")

def run(case):
    if case == "calibrate": calibrate()
    elif case == "short":
        r = chat([{"role": "user", "content": "Name the three planets closest to the Sun."}], 64)
        report(case, r, "choices" in r and "Mercury" in r["choices"][0]["message"]["content"])
    elif case.startswith("needle"):        # needle<K>k, needle at 25 / 50 / 90 %
        k = int(case[6:-1]) * 1000
        for at in (0.25, 0.5, 0.9):
            needle_case(f"{case}@{int(at*100)}", k, at)
    elif case == "reporter":                # ~8.4K few-shot prompt + up to 1.5K answer (#1001)
        q = ("You will read harbour records and then write a detailed report.\n" + haystack(8400) +
             "\nWrite a detailed report (about 1500 tokens) on the records above: busiest towns, grain totals, wind.")
        r = chat([{"role": "user", "content": q}], 1500)
        ok = "choices" in r and r["choices"][0]["finish_reason"] in ("stop", "length") and sane(r["choices"][0]["message"]["content"])
        report(case, r, ok, f"total={total(r)}")
    elif case.startswith("fill"):           # fill<limit>[_json|_tools]: prompt ends ~600 below the top
        parts = case[4:].split("_"); limit = int(parts[0])
        extra = {}
        if "json" in parts: extra["response_format"] = {"type": "json_object"}
        if "tools" in parts:
            extra["tools"] = [{"type": "function", "function": {"name": "log_story", "description": "Save a story",
                               "parameters": {"type": "object", "properties": {"story": {"type": "string"}}, "required": ["story"]}}}]
        fill_case(case, limit - 700, 4000, limit, **extra)
    elif case.startswith("tail"):           # tail<limit>_<gap>: prompt ends <gap> tokens below the top (verify-tail prefill)
        limit, gap = map(int, case[4:].split("_"))
        q = haystack(limit - gap - 60, NEEDLE, 0.5) + "\nWhat is the passphrase for the north gate? Answer with the passphrase only."
        r = chat([{"role": "user", "content": q}], 64)
        if "choices" not in r:
            ok = r.get("http_error") is not None and gap < 0
            return report(case, r, ok, f"error reply {r.get('http_error')} {r.get('body', '')[:200]}")
        text = r["choices"][0]["message"].get("content") or ""
        ok = total(r) <= limit + 1 and ("AMBER-7351-QUILL" in text or r["choices"][0]["finish_reason"] == "length")
        report(case, r, ok, f"total={total(r)} text={text[:60]!r}")
    elif case.startswith("over"):           # over<limit>: prompt larger than the context -> clean error, server lives
        limit = int(case[4:])
        r = chat([{"role": "user", "content": haystack(limit + 400)}], 16)
        alive = chat([{"role": "user", "content": "Say OK."}], 4)
        ok = "choices" in alive and ("http_error" in r or (r.get("choices") and r["choices"][0]["finish_reason"] == "length"))
        report(case, r, ok, f"after: {'alive' if 'choices' in alive else 'DEAD'} {r.get('body', '')[:200]}")
    elif case.startswith("grow"):           # grow<limit>: multi-turn chat 2K per turn to the top; needle from turn 1
        limit = int(case[4:])
        msgs = [{"role": "user", "content": haystack(2000, NEEDLE, 0.5) + "\nRemember these records. Reply 'noted'."}]
        start = 0; turn = 0
        while True:
            r = chat(msgs, 16)
            turn += 1
            if "choices" not in r:
                ok = total(r) == 0 and turn > 1
                return report(f"{case}_turn{turn}", r, False, f"error {r.get('http_error')} {r.get('body', '')[:200]}")
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
            used = r["usage"]["prompt_tokens"]
            if used > limit - 2600:
                break
            start += int(2000 / TOK_PER_RECORD)
            msgs.append({"role": "user", "content": "".join(RECORDS[(start + 100) % 3000:(start + 100) % 3000 + int(2000 / TOK_PER_RECORD)]) + "\nMore records. Reply 'noted'."})
        msgs.append({"role": "user", "content": "What is the passphrase for the north gate? Answer with the passphrase only."})
        r = chat(msgs, 32)
        ok = "choices" in r and "AMBER-7351-QUILL" in (r["choices"][0]["message"].get("content") or "")
        report(f"{case}_turns{turn}", r, ok, f"total={total(r)}")
    else:
        print("unknown case", case)

for c in CASES:
    run(c)
print("ALL_PASS" if all(x["ok"] for x in results) else "SOME_FAIL", flush=True)
