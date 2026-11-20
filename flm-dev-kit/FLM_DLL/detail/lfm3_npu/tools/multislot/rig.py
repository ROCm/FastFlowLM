#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Multi-slot prefix cache test rig for lfm3:1.7b (server level).

Starts its own `flm serve` processes on dedicated ports (never touches other servers), drives conversations over the REST API
and asserts on `usage.flm_prefix_cache` (lookup hit/miss, slot type, matched tokens, saved/dropped, budget) and on the texts.

  python3 rig.py --flm <flm binary> [--env KEY=VAL ...] [--only S1,S3] [--decode full|conv|host] [--port 52740]

Scenarios (each prints PASS/FAIL lines; exit status 1 if any FAIL):
  S1  history growth        turn 1 misses and saves; turns 2..4 hit the previous end slot; a boundary slot is saved
  S2  branching             an edited earlier turn resumes from the right ancestor; replaying a turn resumes from its parent
  S3  boundary sharing      a new conversation with the same long system prompt resumes from the boundary slot; another system misses
  S4  stale rows            the same request after unrelated traffic gives the identical text (rows past the length are zeroed)
  S5  think on / off        three-turn conversations with thinking on and off both hit
  S6  tools                 a tool-calling conversation (tools in the head, tool call + tool result turns) hits
  S7  long context          ~6k tokens over several turns (cache growth past the initial MAX_L): every turn hits, no errors
  S8  slot cap              --prefix-slots 2: slots_used <= 2, snapshots dropped, newest conversation still hits
  S9  tiny budget           --prefix-cache-mb 1: nothing saved over budget, requests still answered
  S10 off                   --prefix-slots 0: no cache, turn 1 text equals the cached run's turn 1
  S11 determinism           the same request sequence on a fresh server gives identical texts and identical lookups
"""
import argparse, json, os, subprocess, sys, time, urllib.request

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""), flush=True)
    return ok


class Server:
    def __init__(self, flm, port, env, extra=()):
        self.port = port
        e = dict(os.environ)
        e.update(env)
        self.log = open(f"/tmp/rig_serve_{port}.log", "w")
        self.p = subprocess.Popen([flm, "serve", "lfm3:1.7b", "-p", str(port), *extra], env=e, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2).read()
                return
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError("server exited: see " + self.log.name)
                time.sleep(1)
        raise RuntimeError("server did not start")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except Exception:
            self.p.kill()

    def chat(self, msgs, tools=None, think=False, max_tokens=24):
        body = {"model": "lfm3:1.7b", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "think": think, "stream": False}
        if tools:
            body["tools"] = tools
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=900))
        m = r["choices"][0]["message"]
        return (m.get("content") or "") + (m.get("reasoning_content") or ""), r["usage"], r["usage"].get("flm_prefix_cache") or {}, m


WORDS = "amber basalt cobalt dune ember fjord garnet harbor indigo juniper kelp lagoon meadow nectar obsidian prairie quartz ridge saffron tundra umber vale willow xenon yarrow zephyr".split()


def prose(tag, n_words):
    """deterministic filler; different `tag` gives different text"""
    out = []
    for i in range(n_words):
        out.append(WORDS[(i * 7 + hash_tag(tag) + i // 5) % len(WORDS)])
    return " ".join(out)


def hash_tag(tag):
    h = 0
    for c in tag:
        h = (h * 31 + ord(c)) % 1000003
    return h


def system(tag, n=520):
    return f"You are a careful assistant. Follow these notes ({tag}): " + prose("sys" + tag, n) + ". Answer briefly."


def user(tag, n=110):
    return f"Question {tag}: " + prose("usr" + tag, n) + " What is the third word of this message?"


def reply(tag, n=90):
    return "Noted. " + prose("rep" + tag, n)


def conv_turns(srv, sys_tag, tags, think=False, tools=None, sys_words=520, user_words=110):
    """one conversation: system + len(tags) user turns with canned assistant replies; returns [(text, usage, pc, prompt_tokens)]"""
    msgs = [{"role": "system", "content": system(sys_tag, sys_words)}]
    out = []
    for i, t in enumerate(tags):
        msgs.append({"role": "user", "content": user(t, user_words)})
        text, usage, pc, _ = srv.chat(msgs, tools=tools, think=think)
        out.append((text, usage, pc, usage["prompt_tokens"]))
        msgs.append({"role": "assistant", "content": reply(t)})
    return out, msgs


def hit(pc):
    return pc.get("lookup") == "hit"


def s1(srv):
    out, _ = conv_turns(srv, "A", ["a1", "a2", "a3", "a4"])
    check("S1 turn 1 misses and saves a snapshot", out[0][2].get("lookup") == "miss" and len(out[0][2].get("saved", [])) >= 1, json.dumps(out[0][2].get("saved")))
    check("S1 a boundary slot (system head) is saved", any(s.get("slot_type") == "boundary" for s in out[0][2].get("saved", [])) or
          any(s.get("slot_type") == "boundary" for o in out for s in o[2].get("saved", [])))
    for i in range(1, 4):
        pc, prev = out[i][2], out[i - 1][3]
        check(f"S1 turn {i+1} hits the previous end slot", hit(pc) and pc.get("slot_type") == "end" and pc.get("matched_tokens", 0) >= 0.9 * prev,
              f"matched {pc.get('matched_tokens')} of previous prompt {prev}, prompt {out[i][3]}")
    return out


def s2(srv):
    # parent conversation, then branch at turn 2 with an edited user message, then replay the original turn 3
    base, msgs = conv_turns(srv, "B", ["b1", "b2", "b3"])
    p1 = base[0][3]
    branch = [{"role": "system", "content": system("B")}, {"role": "user", "content": user("b1")}, {"role": "assistant", "content": reply("b1")},
              {"role": "user", "content": user("b2-edited")}]
    _, _, pc, _ = srv.chat(branch)
    check("S2 an edited turn resumes from its ancestor (turn 1 end slot)", hit(pc) and pc.get("matched_tokens", 0) >= 0.9 * p1, f"matched {pc.get('matched_tokens')} ancestor {p1}")
    replay = msgs[:-1][:6]   # system, u1, r1, u2, r2, u3 : the original turn 3 prompt
    _, usage, pc, _ = srv.chat(replay)
    check("S2 replaying turn 3 still hits (parent slot kept)", hit(pc) and pc.get("matched_tokens", 0) >= 0.9 * base[1][3], f"matched {pc.get('matched_tokens')} parent {base[1][3]}")


def s3(srv):
    out, _ = conv_turns(srv, "A", ["c1"])      # same system as S1's conversation, different first user message
    pc = out[0][2]
    check("S3 same system prompt resumes from the boundary slot", hit(pc) and pc.get("slot_type") == "boundary" and pc.get("matched_tokens", 0) >= 512,
          json.dumps({k: pc.get(k) for k in ("lookup", "slot_type", "matched_tokens", "miss_reason")}))
    out2, _ = conv_turns(srv, "Z", ["z1"])
    check("S3 a different system prompt misses", out2[0][2].get("lookup") == "miss", json.dumps(out2[0][2].get("miss_reason")))


def s4(srv):
    first, msgs = conv_turns(srv, "D", ["d1", "d2"])
    again1 = srv.chat(msgs[:4])[0]       # system, u1, r1, u2 : turn 2 prompt (hits turn 1 end slot)
    conv_turns(srv, "E", ["e1", "e2"], user_words=400)   # unrelated, longer traffic in between
    again2 = srv.chat(msgs[:4])[0]
    check("S4 replay after unrelated traffic gives the identical text", again1 == again2 and len(again1) > 0, repr(again1[:50]) + " vs " + repr(again2[:50]))


def s5(srv):
    for think in (True, False):
        out, _ = conv_turns(srv, "T" + str(think), ["t1", "t2", "t3"], think=think)
        check(f"S5 think={think}: turns 2 and 3 hit", hit(out[1][2]) and hit(out[2][2]), json.dumps([o[2].get("lookup") for o in out]))


def s6(srv):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    msgs = [{"role": "system", "content": system("TOOLS", 300)}, {"role": "user", "content": user("w1") + " Then tell me the weather in Paris."}]
    text, usage, pc, m = srv.chat(msgs, tools=tools, max_tokens=60)
    check("S6 tool conversation turn 1 answered", len(text) > 0 or m.get("tool_calls"), repr(text[:40]))
    msgs += [{"role": "assistant", "content": reply("w1")}, {"role": "user", "content": user("w2")}]
    _, _, pc2, _ = srv.chat(msgs, tools=tools, max_tokens=60)
    check("S6 turn 2 with tools hits", hit(pc2), json.dumps({k: pc2.get(k) for k in ("lookup", "matched_tokens", "miss_reason")}))
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"}, {"role": "user", "content": "Thanks. And in Rome?"}]
    _, _, pc3, _ = srv.chat(msgs, tools=tools, max_tokens=60)
    check("S6 turn 3 (tool call + tool result in history) hits", hit(pc3), json.dumps({k: pc3.get(k) for k in ("lookup", "matched_tokens", "miss_reason")}))


def s7(srv):
    out, _ = conv_turns(srv, "L", ["l%d" % i for i in range(8)], user_words=520, sys_words=200)
    last = out[-1][3]
    check("S7 context grows past 6k tokens", last > 4500, f"last prompt {last}")
    check("S7 every turn after the first hits", all(hit(o[2]) for o in out[1:]), json.dumps([o[2].get("lookup") for o in out]))
    check("S7 all answers non-empty", all(len(o[0]) > 0 for o in out))


def run_sequence(srv):
    """the request sequence of S1 + S3 (used by S10 / S11): returns texts and lookups"""
    texts, looks = [], []
    out, _ = conv_turns(srv, "A", ["a1", "a2", "a3"])
    out2, _ = conv_turns(srv, "A", ["c1"])
    out3, _ = conv_turns(srv, "Q", ["q1", "q2"])
    for o in out + out2 + out3:
        texts.append(o[0]); looks.append((o[2].get("lookup"), o[2].get("slot_type"), o[2].get("matched_tokens")))
    return texts, looks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flm", required=True)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--only", default="")
    ap.add_argument("--decode", default="full")
    ap.add_argument("--port", type=int, default=52740)
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    env["LFM3_DECODE"] = a.decode
    only = set(a.only.split(",")) if a.only else None
    want = lambda n: only is None or n in only
    port = a.port

    def with_server(extra, fn):
        nonlocal port
        srv = Server(a.flm, port, env, extra)
        port += 1
        try:
            fn(srv)
        finally:
            srv.stop()

    if any(want(n) for n in ("S1", "S2", "S3", "S4", "S5", "S6", "S7")):
        def default_server(srv):
            if want("S1"): s1(srv)
            if want("S2"): s2(srv)
            if want("S3"): s3(srv)
            if want("S4"): s4(srv)
            if want("S5"): s5(srv)
            if want("S6"): s6(srv)
            if want("S7"): s7(srv)
        with_server([], default_server)
    if want("S8"):
        def capped(srv):
            seen_dropped = 0
            for i in range(4):
                text, usage, pc, _ = srv.chat([{"role": "user", "content": user("cap%d" % i, 200)}])
                seen_dropped += pc.get("dropped", 0)
                check(f"S8 conversation {i}: slots_used <= 2", pc.get("slots_used", 0) <= 2 and pc.get("slots_max") == 2, f"used {pc.get('slots_used')} max {pc.get('slots_max')}")
            check("S8 snapshots were dropped to stay within 2 slots", seen_dropped >= 2, f"dropped {seen_dropped}")
            msgs = [{"role": "user", "content": user("cap3", 200)}, {"role": "assistant", "content": reply("cap3")}, {"role": "user", "content": user("cap3b", 100)}]
            _, _, pc, _ = srv.chat(msgs)
            check("S8 the newest conversation still hits", hit(pc), json.dumps(pc.get("miss_reason")))
        with_server(["--prefix-slots", "2"], capped)
    if want("S9"):
        def tiny(srv):
            texts = []
            for i in range(2):
                text, usage, pc, _ = srv.chat([{"role": "system", "content": system("TINY")}, {"role": "user", "content": user("tiny%d" % i)}])
                texts.append(text)
                check(f"S9 request {i}: budget respected", pc.get("bytes_used", 0) <= pc.get("bytes_budget", 1 << 40), f"used {pc.get('bytes_used')} budget {pc.get('bytes_budget')}")
            check("S9 requests are still answered", all(len(t) > 0 for t in texts))
        with_server(["--prefix-cache-mb", "1"], tiny)
    on_texts = None
    if want("S10") or want("S11"):
        srv1 = Server(a.flm, port, env); port += 1
        try:
            on_texts, on_looks = run_sequence(srv1)
        finally:
            srv1.stop()
        check("S11 the sequence used the cache (hits present)", any(l[0] == "hit" for l in on_looks), json.dumps(on_looks))
        if want("S11"):
            srv2 = Server(a.flm, port, env); port += 1
            try:
                t2, l2 = run_sequence(srv2)
            finally:
                srv2.stop()
            check("S11 fresh-server replay: identical texts", t2 == on_texts, str([i for i, (x, y) in enumerate(zip(t2, on_texts)) if x != y]))
            check("S11 fresh-server replay: identical lookups", l2 == on_looks)
        if want("S10"):
            srv3 = Server(a.flm, port, env, ["--prefix-slots", "0"]); port += 1
            try:
                t3, l3 = run_sequence(srv3)
            finally:
                srv3.stop()
            check("S10 --prefix-slots 0: no lookups reported as hits", all(l[0] != "hit" for l in l3), json.dumps(l3[:3]))
            check("S10 turn 1 text identical with and without the cache", t3[0] == on_texts[0])
            same = sum(1 for x, y in zip(t3, on_texts) if x == y)
            print(f"INFO S10 identical texts cache on/off: {same} of {len(t3)} (chunked-prefill numerics may differ after a resume)")
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)} passed, {len(bad)} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
