"""Pressure test of minicpm-v-4.7:1b batch image input (image_raw RGB8 and PNG) on one flm package.

  python pressure.py --flm-bin PATH --repo CHECKOUT --out DIR [--phases A,P,B,C,D,E,F]
                     [--baseline-bin PATH] [--golden FILE] [--quick]

Phases (each starts its own server; the run stops at the first failed phase):
  A correctness   P perf          B soak          C shape sweep
  D concurrency + client abuse    E file input    F final health
Writes DIR/results.json, DIR/summary.md (also appended to $GITHUB_STEP_SUMMARY),
DIR/golden.json (texts and token counts, to compare one platform against another)
and one server log per phase. Exit code 1 on any failed check.
"""

import argparse
import base64
import json
import os
import re
import shutil
import socket
import statistics as S
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

import proc
from clip import FH, FW, Clips
from pngio import write_rgb

MODEL = "minicpm-v-4.7:1b"
OTHER = "qwen3:0.6b"
MODEL_DIR = "MiniCPM-V-4.7-1B-NPU2"
BASELINE_MODEL_VERSION = "1.0.12"
CTX = ["--ctx-len", "32768"]
FRAME = FW * FH * 3
HD = 1920 * 1080 * 3
# Our own prompt in the style of the customer request (theirs is not ours to publish).
PROMPT = ("请用中文概述这段视频片段，按画面出现的先后顺序，描述能看到的主要物体、人物、动作和场景。"
          "要求：相似或重复的画面合并描述，只写有意义的新变化；最多3条，每条一句，总计不超过120字；"
          "不输出时间戳，不逐帧列举，不抄写画面中的文字，不猜测片段之外的内容。")


# -- requests ------------------------------------------------------------------

def raw_part(data, n, w=FW, h=FH):
    return {"type": "image_raw", "image_raw": {"pixel_format": "RGB8", "layout": "THWC", "dtype": "uint8",
                                               "shape": [n, h, w, 3], "data": base64.b64encode(data).decode()}}


def png_parts(data, n, w=FW, h=FH):
    size = w * h * 3
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(
        write_rgb(w, h, data[i * size:(i + 1) * size])).decode()}} for i in range(n)]


def txt(s):
    return {"type": "text", "text": s}


def body_of(content, model=MODEL, max_tokens=4096, stream=False):
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "stream": stream, "think": False,
            "chat_template_kwargs": {"enable_thinking": False}}


class Reply:
    def __init__(self, code, body, wall, size, encode_s=0.0):
        self.code, self.body, self.wall, self.size, self.encode_s = code, body, wall, size, encode_s

    @property
    def text(self):
        try:
            return self.body["choices"][0]["message"]["content"]
        except Exception:
            return None

    @property
    def finish(self):
        try:
            return self.body["choices"][0]["finish_reason"]
        except Exception:
            return None

    @property
    def usage(self):
        return self.body.get("usage") or {} if isinstance(self.body, dict) else {}

    @property
    def error(self):
        e = self.body.get("error") if isinstance(self.body, dict) else None
        return (e.get("message") if isinstance(e, dict) else e) or ""

    def row(self):
        u = self.usage
        dec = u.get("decoding_duration") or 0
        return {"code": self.code, "wall": round(self.wall, 3), "ttft": u.get("prefill_duration_ttft"),
                "decode_s": dec, "tok_s": round(u.get("completion_tokens", 0) / dec, 2) if dec else None,
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "finish": self.finish, "body_mb": round(self.size / 1e6, 2), "encode_s": round(self.encode_s, 3)}


def post(port, content=None, raw_body=None, timeout=1800, **kw):
    t0 = time.monotonic()
    data = raw_body if raw_body is not None else json.dumps(body_of(content, **kw)).encode()
    enc = time.monotonic() - t0
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    wall = time.monotonic() - t0
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = {"error": raw[:300].decode("utf-8", "replace")}
    return Reply(code, body, wall, len(data), enc)


def post_stream(port, content, timeout=1800, **kw):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body_of(content, stream=True, **kw)).encode(),
                                 {"Content-Type": "application/json"})
    text, chunks = "", 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                chunks += 1
                for ch in json.loads(line[6:]).get("choices", []):
                    text += (ch.get("delta") or {}).get("content") or ""
    return text, chunks


def abandon(port, raw_body, after_s):
    """Sends a whole request, then hangs up after `after_s` without reading the reply."""
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(raw_body)}\r\n\r\n").encode()
    s.sendall(head + raw_body)
    time.sleep(after_s)
    s.close()


# -- run state -----------------------------------------------------------------

class Run:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        os.makedirs(self.out, exist_ok=True)
        sys.path.insert(0, os.path.join(args.repo, ".github", "script"))
        from npu_test.server import FlmServer  # CI's own start/stop logic
        self.FlmServer = FlmServer
        self.clips = Clips(args.repo, os.path.join(self.out, "clips"))
        self.results = {"flm_bin": args.flm_bin, "baseline_bin": args.baseline_bin, "phases": {}}
        self.golden = json.load(open(args.golden, encoding="utf-8")) if args.golden else None
        self.ref = {}          # texts from phase A, used as expected output later
        self.phase = None
        self.servers = []

    # checks ------------------------------------------------------------------
    def check(self, name, ok, detail="", soft=False):
        rec = {"name": name, "ok": bool(ok), "soft": soft, "detail": detail}
        self.results["phases"][self.phase]["checks"].append(rec)
        tag = "PASS" if ok else ("WARN" if soft else "FAIL")
        print(f"  [{tag}] {name} {detail}", flush=True)
        return ok

    def data(self, key, value):
        self.results["phases"][self.phase].setdefault("data", {})[key] = value

    # servers -----------------------------------------------------------------
    def server(self, name, bin_=None, extra=(), env=None):
        s = self.FlmServer(bin_ or self.args.flm_bin, name=name, log_dir=self.out, args=CTX + list(extra),
                           ready_timeout=600, request_timeout=1800, max_restarts=0, env=env)
        s.start()
        self.servers.append(s)
        path = proc.image_path(s.proc.pid)
        want = bin_ or self.args.flm_bin
        # The Linux package's flm is a script that execs flm-real next to it.
        ok = path and (proc.same_file(path, want) or
                       (os.name != "nt" and proc.same_file(os.path.dirname(path), os.path.dirname(want))))
        if not self.check(f"{name}: server process is the package under test", ok, f"pid {s.proc.pid} image {path}"):
            raise SystemExit("wrong flm process")
        return s

    def stop(self, s):
        s.stop()
        self.shrink_log(s)
        if s.unkillable:
            self.check(f"{s.name}: server stopped", False, "process survived kill (stuck in NPU driver?)")
            raise SystemExit("unkillable server")
        return s.proc.returncode

    def shrink_log(self, s, keep=300):
        """flm logs each request body, and does not shorten image_raw data (only image_url, input_audio
        and images), so a 120-frame request writes ~43 MB. Cut such lines so the artifact stays small,
        and count them: the server writing the frames to its log is itself a finding."""
        try:
            with open(s.log_path, "rb") as f:
                lines = f.read().split(b"\n")
        except OSError:
            return
        big = [len(x) for x in lines if len(x) > 100_000]
        if not big:
            return
        out = [x if len(x) <= 100_000 else x[:keep] + b" ...[%d bytes cut by pressure.py]" % (len(x) - keep)
               for x in lines]
        with open(s.log_path, "wb") as f:
            f.write(b"\n".join(out))
        logged = self.results.setdefault("raw_data_logged", {"lines": 0, "bytes": 0, "max": 0})
        logged["lines"] += len(big)
        logged["bytes"] += sum(big)
        logged["max"] = max(logged["max"], *big)

    def log_text(self, s):
        try:
            return open(s.log_path, encoding="utf-8", errors="replace").read()
        except OSError:
            return ""


# -- phases --------------------------------------------------------------------

def phase_A(r):
    c, q = r.clips, txt(PROMPT)
    s = r.server("A")
    rows = {}
    try:
        for n in (1, 8, 32, 120):
            frames = c.frames(n)
            raw = post(s.port, [raw_part(frames, n), q])
            png = post(s.port, png_parts(frames, n) + [q])
            rows[n] = {"raw": raw.row(), "png": png.row()}
            r.check(f"N={n} raw 200", raw.code == 200, f"{raw.code} {raw.error[:120]}")
            r.check(f"N={n} png 200", png.code == 200, f"{png.code} {png.error[:120]}")
            r.check(f"N={n} raw text == png text", raw.text is not None and raw.text == png.text,
                    f"raw {raw.text!r:.80} | png {png.text!r:.80}")
            r.check(f"N={n} same prompt tokens", raw.usage.get("prompt_tokens") == png.usage.get("prompt_tokens"),
                    f"{raw.usage.get('prompt_tokens')} vs {png.usage.get('prompt_tokens')}")
            r.check(f"N={n} finish_reason stop", raw.finish == "stop", str(raw.finish))
            r.ref[n] = {"text": raw.text, "prompt_tokens": raw.usage.get("prompt_tokens")}
        again = post(s.port, [raw_part(c.frames(120), 120), q])
        r.check("N=120 raw repeat is deterministic", again.text == r.ref[120]["text"], f"{again.text!r:.80}")
        if r.golden:
            for n in (1, 8, 32, 120):
                g = r.golden["A"].get(str(n), {})
                r.check(f"N={n} prompt tokens == golden ({r.golden.get('platform')})",
                        g.get("prompt_tokens") == r.ref[n]["prompt_tokens"],
                        f"{r.ref[n]['prompt_tokens']} vs {g.get('prompt_tokens')}")
                r.check(f"N={n} text == golden ({r.golden.get('platform')})", g.get("text") == r.ref[n]["text"],
                        "" if g.get("text") == r.ref[n]["text"] else f"got {r.ref[n]['text']!r:.100}", soft=True)
    finally:
        r.stop(s)
    log = r.log_text(s)
    r.check("no stride_1 log spam", log.count("stride_1") == 0, f"{log.count('stride_1')} lines")
    r.data("rows", rows)
    json.dump({"platform": sys.platform, "clip_sha256": c.digest(), "A": {str(k): v for k, v in r.ref.items()}},
              open(os.path.join(r.out, "golden.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _sweep_ns(r):
    return (1, 8, 32, 64, 120, "hd12")


def _content(r, n, kind, q):
    if n == "hd12":
        d = r.clips.hd(12)
        return [raw_part(d, 12, 1920, 1080), q] if kind == "raw" else png_parts(d, 12, 1920, 1080) + [q]
    d = r.clips.frames(n)
    return [raw_part(d, n), q] if kind == "raw" else png_parts(d, n) + [q]


def _bodies(r, n, q):
    return {k: json.dumps(body_of(_content(r, n, k, q))).encode() for k in ("raw", "png")}


def _baseline_models(r):
    """A model folder the baseline release accepts: the same files (hard links, else copies) with
    config.json's flm_version lowered. The current files declare 1.0.13, which a 1.0.12 flm refuses
    to load although the weights are unchanged (the 2026-11-08 config.json update only bumped it)."""
    src_root = os.environ.get("FLM_MODEL_PATH") or os.path.expanduser("~/.flm" if os.name == "nt" else "~/.config/flm")
    src = os.path.join(src_root, "models", MODEL_DIR)  # flm reads <FLM_MODEL_PATH>/models/<name>
    root = os.path.join(r.out, "base-models")
    dst = os.path.join(root, "models", MODEL_DIR)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f == "config.json" or os.path.exists(os.path.join(dst, f)):
            continue
        try:
            os.link(os.path.join(src, f), os.path.join(dst, f))
        except OSError:
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    r.data("baseline_model_flm_version_was", cfg.get("flm_version"))
    cfg["flm_version"] = BASELINE_MODEL_VERSION
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
    return root


def _sweep(r, s, kinds, reps, label, soft=False):
    q, table = txt(PROMPT), {}
    post(s.port, [raw_part(r.clips.frames(2), 2), txt("hi")], max_tokens=4)
    for n in _sweep_ns(r):
        bodies = _bodies(r, n, q)
        post(s.port, raw_body=bodies[kinds[0]])  # warm-up for this shape
        runs = {k: [] for k in kinds}
        for _ in range(reps):
            for k in kinds:
                rep = post(s.port, raw_body=bodies[k])
                r.check(f"{label} N={n} {k} 200", rep.code == 200, f"{rep.code} {rep.error[:100]}", soft=soft)
                runs[k].append(rep.row())
        table[str(n)] = {k: _summ(v) for k, v in runs.items()}
        med = lambda k, key: f"{table[str(n)][k][key]['med']:.2f}s" if table[str(n)][k][key] else "-"
        line = "  ".join(f"{k}: ttft {med(k, 'ttft')} wall {med(k, 'wall')}" for k in kinds)
        print(f"  {label} N={n} tokens={runs[kinds[0]][0]['prompt_tokens']}  {line}", flush=True)
    return table


def _summ(rows):
    out = {"n": len(rows), "prompt_tokens": rows[0]["prompt_tokens"], "body_mb": rows[0]["body_mb"],
           "completion_tokens": [x["completion_tokens"] for x in rows]}
    for key in ("wall", "ttft", "decode_s", "tok_s", "encode_s"):
        vals = [x[key] for x in rows if x[key] is not None]
        out[key] = {"med": round(S.median(vals), 3), "min": round(min(vals), 3), "max": round(max(vals), 3),
                    "all": vals} if vals else None
    return out


def _profile(r, bin_, label, kinds, env=None):
    """Tower and language-prefill times from FLM_MINICPM_V_VISION_PROFILE, in a pass of its own."""
    s = r.server(f"P-{label}-profile", bin_, env=dict(env or {}, FLM_MINICPM_V_VISION_PROFILE="1"))
    q, out = txt(PROMPT), {}
    try:
        for n in (32, 120):
            for k in kinds:
                body = json.dumps(body_of(_content(r, n, k, q))).encode()
                mark = os.path.getsize(s.log_path)
                for _ in range(3):
                    post(s.port, raw_body=body)
                log = open(s.log_path, encoding="utf-8", errors="replace").read()[mark:]
                tower = [int(x) for x in re.findall(r"vision: .*? rows in (\d+) ms", log)]
                llm = [int(x) for x in re.findall(r"language prefill: \d+ tokens in (\d+) ms", log)]
                out[f"{n}-{k}"] = {"tower_ms": tower, "llm_prefill_ms": llm}
                print(f"  {label} profile N={n} {k}: tower {tower} ms, llm prefill {llm} ms", flush=True)
    finally:
        r.stop(s)
    return out


def phase_P(r):
    reps = 1 if r.args.quick else r.args.perf_reps
    s = r.server("P-main")
    try:
        main = _sweep(r, s, ("raw", "png"), reps, "main")
        # cold first request is measured with fresh servers below; model switch here
        switch = []
        for _ in range(2):
            post(s.port, [txt("hi")], model=OTHER, max_tokens=4)
            rep = post(s.port, [raw_part(r.clips.frames(120), 120), txt(PROMPT)])
            switch.append(rep.row())
        r.data("after_switch_N120_raw", switch)
    finally:
        r.stop(s)
    r.data("main", main)
    cold = []
    for i in range(1 if r.args.quick else 3):
        t0 = time.monotonic()
        s = r.server(f"P-cold{i}")
        ready = time.monotonic() - t0
        try:
            rep = post(s.port, [raw_part(r.clips.frames(120), 120), txt(PROMPT)])
            cold.append(dict(rep.row(), server_ready_s=round(ready, 2)))
            print(f"  cold {i}: ready {ready:.1f}s, first N=120 raw wall {rep.wall:.2f}s", flush=True)
        finally:
            r.stop(s)
    r.data("cold_first_N120_raw", cold)
    r.data("profile_main", _profile(r, None, "main", ("raw", "png")))
    if r.args.baseline_bin:
        env, usable = {"FLM_MODEL_PATH": _baseline_models(r)}, False
        s = r.server("P-baseline", r.args.baseline_bin, env=env)
        try:
            r.data("baseline_version", s.version)
            probe = post(s.port, [txt("hi")], max_tokens=4)
            usable = r.check("baseline loads the model", probe.code == 200,
                             f"{probe.code} {probe.error[:160]}", soft=True)
            if usable:
                base = _sweep(r, s, ("png",), reps, "baseline", soft=True)
                r.data("baseline", base)
        finally:
            r.stop(s)
        if usable:
            r.data("profile_baseline", _profile(r, r.args.baseline_bin, "baseline", ("png",), env=env))
        # drift check: main again on the largest case after the baseline
        s = r.server("P-main-again")
        try:
            b = _bodies(r, 120, txt(PROMPT))
            post(s.port, raw_body=b["raw"])
            again = [post(s.port, raw_body=b[k]).row() for k in ("raw", "png") for _ in range(2)]
            r.data("main_again_N120", again)
        finally:
            r.stop(s)
    _perf_verdicts(r)


def _perf_verdicts(r):
    d = r.results["phases"]["P"]["data"]
    m = d["main"]["120"]
    raw, png = m["raw"]["ttft"]["med"], m["png"]["ttft"]["med"]
    r.check("perf: N=120 raw TTFT within 5% of PNG (or faster)", raw <= png * 1.05,
            f"raw {raw:.2f}s png {png:.2f}s", soft=True)
    bp = d.get("baseline", {}).get("120", {}).get("png", {})
    if bp.get("ttft") and bp.get("tok_s") and m["png"]["tok_s"]:
        b = bp["ttft"]["med"]
        r.check("perf: main PNG N=120 TTFT faster than baseline", png < b,
                f"main png {png:.2f}s raw {raw:.2f}s vs baseline png {b:.2f}s ({b / raw:.2f}x for raw)", soft=True)
        bt = bp["tok_s"]["med"]
        mt = m["png"]["tok_s"]["med"]
        r.check("perf: decode tok/s not lower than baseline (-5%)", mt >= bt * 0.95,
                f"main {mt:.1f} vs baseline {bt:.1f} tok/s", soft=True)


def phase_B(r):
    n_req = 6 if r.args.quick else 30
    s = r.server("B")
    q = txt(PROMPT)
    b = _bodies(r, 120, q)
    sampler = proc.Sampler(s.proc.pid, 5.0)
    sampler.start()
    rows, idle = [], []
    try:
        for i in range(n_req):
            kind = ("raw", "png")[i % 2]
            rep = post(s.port, raw_body=b[kind])
            row = dict(rep.row(), kind=kind, same=rep.text == r.ref[120]["text"])
            rows.append(row)
            idle.append(proc.stats(s.proc.pid))
            print(f"  B {i:2d} {kind} {rep.code} ttft {row['ttft']} wall {row['wall']} "
                  f"rss {idle[-1]['rss'] / 2**30:.2f} GiB handles {idle[-1]['handles']} same={row['same']}", flush=True)
            if rep.code != 200:
                break
    finally:
        sampler.stop()
        r.stop(s)
    r.data("rows", rows)
    r.data("samples", sampler.samples)
    r.data("idle", idle)
    r.check("all 200", all(x["code"] == 200 for x in rows) and len(rows) == n_req)
    r.check("every text == phase A N=120 text", all(x["same"] for x in rows),
            f"{sum(not x['same'] for x in rows)} differ")
    for kind in ("raw", "png"):
        t = [x["ttft"] for x in rows[2:] if x["kind"] == kind and x["ttft"]]
        if t:
            med = S.median(t)
            slow = [round(v, 2) for v in t if v > med * 1.15]
            r.check(f"{kind} TTFT: no request >15% over median", len(slow) == 0,
                    f"median {med:.2f}s slow {slow}", soft=len(slow) <= 2)
    if len(idle) >= 8:
        base = S.median(x["rss"] for x in idle[2:5])
        end = S.median(x["rss"] for x in idle[-3:])
        r.check("RSS drift < 3% after warm-up", end <= base * 1.03,
                f"{base / 2**30:.2f} -> {end / 2**30:.2f} GiB ({(end / base - 1) * 100:+.1f}%)")
        hb, he = S.median(x["handles"] for x in idle[2:5]), S.median(x["handles"] for x in idle[-3:])
        r.check("handle count flat (+50 max)", he <= hb + 50, f"{hb} -> {he}")


def phase_C(r):
    c, q = r.clips, txt("Describe these frames in one sentence.")
    s = r.server("C")

    def twice(name, content, want=200):
        a = post(s.port, content, max_tokens=64)
        if want != 200:
            return r.check(f"{name} -> {want}", a.code == want, f"{a.code} {a.error[:100]}")
        b = post(s.port, content, max_tokens=64)
        r.check(f"{name} 200 + deterministic", a.code == b.code == 200 and a.text == b.text,
                f"{a.code}/{b.code} tokens {a.usage.get('prompt_tokens')} {a.error[:80]}")
        return a

    try:
        for n in (1, 2, 3, 7, 16, 33, 64, 100, 119, 120, 127, 128):
            twice(f"raw N={n}", [raw_part(c.frames(n), n), q])
        twice("raw N=129", [raw_part(c.frames(129), 129), q], want=400)
        hd = twice("12 x 1080p raw", [raw_part(c.hd(12), 12, 1920, 1080), q])
        r.data("hd12_prompt_tokens", hd.usage.get("prompt_tokens"))
        f6 = c.frames(6)
        twice("mixed png + raw + png", png_parts(f6[:FRAME], 1) + [raw_part(f6[FRAME:5 * FRAME], 4)]
              + png_parts(f6[5 * FRAME:], 1) + [q])
        twice("portrait 224x398 x8 raw", [raw_part(c.portrait(8), 8, FH, FW), q])
        twice("two raw parts, different sizes", [raw_part(c.frames(4), 4), raw_part(c.hd(12)[:2 * HD], 2, 1920, 1080), q])
        for name, content in (("size change: 398x224 x8", [raw_part(c.frames(8), 8), q]),
                              ("size change: 1080p x2", [raw_part(c.hd(12)[:2 * HD], 2, 1920, 1080), q]),
                              ("size change: 224x398 x8", [raw_part(c.portrait(8), 8, FH, FW), q])):
            twice(name, content)
    finally:
        r.stop(s)


def phase_D(r):
    c, q = r.clips, txt(PROMPT)
    s = r.server("D")
    sampler = proc.Sampler(s.proc.pid, 5.0)
    sampler.start()
    small = [raw_part(c.frames(8), 8), q]
    try:
        body120 = json.dumps(body_of([raw_part(c.frames(120), 120), q])).encode()
        out = [None] * 4

        def one(i):
            out[i] = post(s.port, raw_body=body120)
        th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        [t.start() for t in th]
        [t.join() for t in th]
        r.check("4 parallel N=120: all 200", all(o and o.code == 200 for o in out), str([o and o.code for o in out]))
        r.check("4 parallel N=120: text == serial", all(o and o.text == r.ref[120]["text"] for o in out))
        r.data("parallel_walls", [o and round(o.wall, 2) for o in out])

        text, chunks = post_stream(s.port, [raw_part(c.frames(120), 120), q])
        r.check("stream N=120 == non-stream text", text == r.ref[120]["text"], f"{chunks} chunks")

        for i in range(3 if r.args.quick else 10):
            abandon(s.port, body120, 1.5 + 0.5 * (i % 4))
        good = post(s.port, small)
        r.check("good request after 10 client hang-ups", good.code == 200 and good.text == r.ref[8]["text"],
                f"{good.code} {good.error[:100]}")
        codes = [post(s.port, [raw_part(c.frames(1), 1), q], max_tokens=4) for _ in range(20)]
        r.check("20 quick requests after hang-ups: all 200 (FLM#911)", all(x.code == 200 for x in codes),
                str(sorted({(x.code, x.error[:60]) for x in codes})))

        big = post(s.port, [raw_part(c.hd(21)[:21 * HD], 21, 1920, 1080), txt("Describe in one sentence.")],
                   max_tokens=16)
        r.check("21 x 1080p (130.6 MB, under the 128 MiB cap) accepted", big.code == 200,
                f"{big.code} tokens {big.usage.get('prompt_tokens')} {big.error[:100]}", soft=True)
        over = post(s.port, [raw_part(c.hd(22)[:22 * HD], 22, 1920, 1080), txt("hi")], max_tokens=4)
        r.check("22 x 1080p (over 128 MiB) -> 400", over.code == 400, f"{over.code} {over.error[:100]}")

        g = raw_part(c.frames(2), 2)

        def mod(**kw):
            p = json.loads(json.dumps(g))
            p["image_raw"].update(kw)
            return p
        bad = {"pixel_format BGR8": mod(pixel_format="BGR8"), "layout TCHW": mod(layout="TCHW"),
               "dtype float32": mod(dtype="float32"), "shape 3-dim": mod(shape=[224, 398, 3]),
               "shape C=4": mod(shape=[2, 224, 398, 4]), "shape too many frames": mod(shape=[3, 224, 398, 3]),
               "shape zero": mod(shape=[0, 224, 398, 3]), "shape negative": mod(shape=[2, -224, 398, 3]),
               "shape float": mod(shape=[2.5, 224, 398, 3]), "shape overflow": mod(shape=[2**24, 2**24, 2**24, 3]),
               "data not string": mod(data=123), "data bad base64": mod(data="!!!!" * 10),
               "data empty": mod(data=""), "image_raw missing": {"type": "image_raw"},
               "image_raw not object": {"type": "image_raw", "image_raw": "x"}}
        for name, part in bad.items():
            rep = post(s.port, [part, txt("hi")], max_tokens=4)
            r.check(f"malformed: {name} -> 400", rep.code == 400, f"{rep.code} {rep.error[:80]}")
        trunc = json.dumps(body_of(small)).encode()[:100000]
        rep = post(s.port, raw_body=trunc)
        r.check("truncated JSON body -> 4xx", 400 <= rep.code < 500, str(rep.code))

        cycles = 3 if r.args.quick else 20
        bad_codes = []
        for i in range(cycles):
            for content, model in (([raw_part(c.frames(1), 1), txt("hi")], MODEL), ([txt("hi")], OTHER)):
                rep = post(s.port, content, model=model, max_tokens=4)
                if rep.code != 200:
                    bad_codes.append((i, model, rep.code, rep.error[:60]))
        r.check(f"{cycles} model switch cycles ({MODEL} <-> {OTHER}) all 200", not bad_codes, str(bad_codes[:3]))
        final = post(s.port, small)
        r.check("minicpm N=8 after switches == phase A", final.code == 200 and final.text == r.ref[8]["text"],
                f"{final.code}")
    finally:
        sampler.stop()
        r.stop(s)
    r.data("samples", sampler.samples)


def phase_E(r):
    c = r.clips
    allow = os.path.join(r.out, "rawdir")
    outside = os.path.join(r.out, "outside")
    os.makedirs(allow, exist_ok=True)
    os.makedirs(outside, exist_ok=True)
    good = os.path.join(allow, "clip.rgb")
    with open(good, "wb") as f:
        f.write(c.frames(120))
    secret = os.path.join(outside, "s.rgb")
    with open(secret, "wb") as f:
        f.write(c.frames(1))
    s = r.server("E", extra=["--allow-raw-path", allow])
    q = txt(PROMPT)

    def pp(path, n=120):
        return {"type": "image_raw", "image_raw": {"pixel_format": "RGB8", "layout": "THWC", "dtype": "uint8",
                                                   "shape": [n, FH, FW, 3], "path": path}}

    def must(name, path, want, n=120, soft=False):
        rep = post(s.port, [pp(path, n), q], max_tokens=4096 if want == 200 else 8, timeout=120)
        ok = rep.code == want and (want != 200 or rep.text == r.ref[120]["text"])
        r.check(f"{name} -> {want}", ok, f"{rep.code} {rep.error[:100]} [{path}]", soft=soft)
        return rep

    try:
        must("path inside allowed dir (text == base64 text)", good, 200)
        must("outside file", secret, 400, 1)
        must("traversal ..", os.path.join(allow, "..", "outside", "s.rgb"), 400, 1)
        must("missing file", os.path.join(allow, "none.rgb"), 400)
        must("directory", allow, 400)
        must("wrong size", good, 400, 50)
        both = {"type": "image_raw", "image_raw": dict(raw_part(c.frames(1), 1)["image_raw"], path=good)}
        rep = post(s.port, [both, q], max_tokens=8)
        r.check("path + data both -> 400", rep.code == 400, str(rep.code))
        if os.name == "nt":
            drive, tail = os.path.splitdrive(os.path.abspath(secret))
            gdrive, gtail = os.path.splitdrive(os.path.abspath(good))
            unc = lambda d, t: "\\\\localhost\\" + d[0] + "$" + t
            must("UNC to outside file", unc(drive, tail), 400, 1)
            must("\\\\?\\ prefix to outside file", "\\\\?\\" + os.path.abspath(secret), 400, 1)
            must("\\\\?\\ prefix .. traversal", "\\\\?\\" + allow + "\\..\\outside\\s.rgb", 400, 1)
            must("alternate data stream", good + ":x", 400)
            must("device name CON", os.path.join(allow, "CON"), 400)
            must("device name NUL", os.path.join(allow, "NUL"), 400)
            # Inside the dir but spelled differently: either answer is safe; record which.
            must("UNC to allowed file", unc(gdrive, gtail), 200, soft=True)
            must("\\\\?\\ prefix to allowed file", "\\\\?\\" + os.path.abspath(good), 200, soft=True)
            must("forward slashes, allowed file", good.replace("\\", "/"), 200, soft=True)
            must("upper-case path, allowed file", good.upper(), 200, soft=True)
        else:
            must("/etc/passwd", "/etc/passwd", 400, 1)
            link = os.path.join(allow, "link.rgb")
            if not os.path.lexists(link):
                os.symlink(secret, link)
            must("symlink inside dir to outside file", link, 400, 1)
        rep = post(s.port, [raw_part(c.frames(1), 1), q], max_tokens=8)
        r.check("server healthy after path probes", rep.code == 200, str(rep.code))
    finally:
        r.stop(s)


def phase_F(r):
    s = r.server("F")
    try:
        rep = post(s.port, [raw_part(r.clips.frames(8), 8), txt(PROMPT)])
        r.check("final N=8 request == phase A", rep.code == 200 and rep.text == r.ref[8]["text"], str(rep.code))
    finally:
        code = r.stop(s)
    r.check("server exits after stop", s.proc.poll() is not None, f"exit code {code}")
    left = []
    for x in r.servers:
        if x.proc.poll() is None:
            left.append(x.proc.pid)
    r.check("no server process left behind", not left, str(left))
    if os.name == "nt":
        ps = subprocess.run(["tasklist", "/FI", "IMAGENAME eq flm.exe", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True).stdout.strip()
        r.check("no flm.exe running", "flm.exe" not in ps.lower(), ps[:200])


COLOURS = ("red", "green", "blue", "yellow", "white", "black", "orange")


def _text_prompt(facts):
    """A text-only prompt of about 20 tokens per fact, deterministic, with one question about the facts."""
    if not facts:
        return "Name three primary colours."
    lines = [f"Item {i}: weighs {i * 37 % 97} grams, is {COLOURS[i % 7]}, and sits on shelf {i % 11}." for i in range(facts)]
    return "\n".join(lines) + "\nWhich three items are the heaviest? Answer in one short sentence."


def _rtm_arm(r, label, env):
    """One server with FLM_MINICPM_MM_RTM set as `env` says. Image prompts go longest first (full 512-row matmul
    dispatches), text prompts after them (shorter, so more dispatches with fewer rows). Stops at the first error."""
    env = dict(env, FLM_MINICPM_V_VISION_PROFILE="1", FLM_MINICPM_V_PREFILL_PROFILE="1")
    s = r.server(f"R-{label}", env=env)
    out, q = {"cases": {}, "failed": False}, txt(PROMPT)
    cases = [(f"img{n}", json.dumps(body_of(_content(r, n, "raw", q))).encode()) for n in (120, 64, 32, 8, 1)]
    cases += [(f"text{f}", json.dumps(body_of([txt(_text_prompt(f))], max_tokens=256)).encode())
              for f in (150, 40, 10, 0)]
    try:
        for name, body in cases:
            mark = os.path.getsize(s.log_path)
            reps = [post(s.port, raw_body=body) for _ in range(3)]
            log = open(s.log_path, encoding="utf-8", errors="replace").read()[mark:]
            rows = [x.row() for x in reps]
            texts = [x.text for x in reps]
            out["cases"][name] = {
                "prompt_tokens": rows[0]["prompt_tokens"], "ttft": [x["ttft"] for x in rows],
                "tok_s": [x["tok_s"] for x in rows], "text": texts[0], "texts_same": len(set(texts)) == 1,
                "llm_prefill_ms": [int(x) for x in re.findall(r"language prefill: \d+ tokens in (\d+) ms", log)],
                "layer_split_ms": [(float(a), float(b)) for a, b in
                                   re.findall(r"\[minicpm_v prefill\] L=\d+\s+\d+ attn ([\d.]+) ms\s+\d+ linear ([\d.]+) ms", log)]}
            c = out["cases"][name]
            print(f"  R {label} {name}: tokens {c['prompt_tokens']} ttft {c['ttft']} prefill {c['llm_prefill_ms']} ms "
                  f"same={c['texts_same']} text {c['text']!r:.60}", flush=True)
            ok = r.check(f"{label} {name} 200 x3", all(x.code == 200 for x in reps),
                         " ".join(f"{x.code} {x.error[:80]}" for x in reps if x.code != 200))
            r.check(f"{label} {name} deterministic", c["texts_same"], "", soft=True)
            if not ok:
                out["failed"] = True
                break
    finally:
        r.stop(s)
    log = r.log_text(s)
    out["errors_in_log"] = len(re.findall(r"(?i)\b(error|exception|timeout|hang)\b", log))
    return out


def phase_R(r):
    """Old fixed 512-row matmul (FLM_MINICPM_MM_RTM=0) against the runtime-m matmul (=1) in the same job.
    texts go to rtm.json, to compare with the other platform's run of this phase."""
    arms = {}
    for label, val in (("mm0", "0"), ("rtm1", "1")):
        arms[label] = _rtm_arm(r, label, {"FLM_MINICPM_MM_RTM": val})
        r.data(label, arms[label])
        if arms[label]["failed"]:
            break
    if all(k in arms and not arms[k]["failed"] for k in ("mm0", "rtm1")):
        a, b = arms["mm0"]["cases"], arms["rtm1"]["cases"]
        for name in a:
            pa, pb = a[name]["llm_prefill_ms"], b[name]["llm_prefill_ms"]
            ma = S.median(pa) if pa else None
            mb = S.median(pb) if pb else None
            print(f"  R {name}: prefill mm0 {ma} ms rtm1 {mb} ms, ttft mm0 {S.median(a[name]['ttft'])} "
                  f"rtm1 {S.median(b[name]['ttft'])} s, same text {a[name]['text'] == b[name]['text']}", flush=True)
        big = "img120"
        if a[big]["llm_prefill_ms"] and b[big]["llm_prefill_ms"]:
            ma, mb = S.median(a[big]["llm_prefill_ms"]), S.median(b[big]["llm_prefill_ms"])
            r.check("perf: N=120 language prefill faster with runtime-m mm", mb < ma, f"rtm1 {mb} ms vs mm0 {ma} ms",
                    soft=True)
    json.dump({"platform": sys.platform, "clip_sha256": r.clips.digest(),
               "arms": {k: {n: {"text": c["text"], "prompt_tokens": c["prompt_tokens"]} for n, c in v["cases"].items()}
                        for k, v in arms.items()}},
              open(os.path.join(r.out, "rtm.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


PHASES = {"A": phase_A, "P": phase_P, "B": phase_B, "C": phase_C, "D": phase_D, "E": phase_E, "F": phase_F,
          "R": phase_R}


def summary(r):
    lines = [f"## minicpm-v-4.7:1b batch image pressure test", "",
             f"- flm: `{r.args.flm_bin}`", f"- baseline: `{r.args.baseline_bin}`",
             f"- clip sha256: `{r.clips.digest()}`"]
    rl = r.results.get("raw_data_logged")
    if rl:
        lines.append(f"- server log echoed image_raw data in full: {rl['lines']} lines, {rl['bytes'] / 1e6:.0f} MB "
                     f"(largest {rl['max'] / 1e6:.1f} MB); cut from the uploaded logs")
    lines.append("")
    for ph, d in r.results["phases"].items():
        fails = [c for c in d["checks"] if not c["ok"] and not c["soft"]]
        warns = [c for c in d["checks"] if not c["ok"] and c["soft"]]
        lines.append(f"### Phase {ph}: {'FAIL' if fails or d.get('error') else 'PASS'} "
                     f"({len(d['checks'])} checks, {len(fails)} failed, {len(warns)} warnings, {d.get('seconds', 0):.0f} s)")
        for c in fails + warns:
            lines.append(f"- {'FAIL' if c in fails else 'WARN'} {c['name']}: {c['detail'][:200]}")
        if d.get("error"):
            lines.append(f"- error: `{d['error'][:300]}`")
        if ph == "P" and "data" in d and "main" in d["data"]:
            dd = d["data"]
            # raw frames are one view each by default (max_slice_nums 1), PNG frames are sliced: token counts differ
            lines += ["", "| N | tokens raw / PNG / base PNG | main raw TTFT | main PNG TTFT | baseline PNG TTFT | main raw wall | decode tok/s main / base |",
                      "|---|---|---|---|---|---|---|"]
            for n, row in dd["main"].items():
                base = dd.get("baseline", {}).get(n, {}).get("png")
                f = lambda x, k: f"{x[k]['med']:.2f}" if x and x.get(k) else "-"
                toks = f"{row['raw']['prompt_tokens']} / {row['png']['prompt_tokens']} / {(base or {}).get('prompt_tokens', '-')}"
                lines.append(f"| {n} | {toks} | {f(row['raw'], 'ttft')} | {f(row['png'], 'ttft')} | "
                             f"{f(base, 'ttft')} | {f(row['raw'], 'wall')} | {f(row['png'], 'tok_s')} / {f(base, 'tok_s')} |")
            for key in ("profile_main", "profile_baseline"):
                if key in dd:
                    lines.append("")
                    lines += [f"- {key} {k}: tower {v['tower_ms']} ms, LLM prefill {v['llm_prefill_ms']} ms"
                              for k, v in dd[key].items()]
            if dd.get("cold_first_N120_raw"):
                lines.append(f"- cold first N=120 raw wall: {[x['wall'] for x in dd['cold_first_N120_raw']]} s")
        lines.append("")
    text = "\n".join(lines)
    open(os.path.join(r.out, "summary.md"), "w", encoding="utf-8").write(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flm-bin", required=True)
    ap.add_argument("--baseline-bin")
    ap.add_argument("--repo", required=True, help="FastFlowLM checkout (images + npu_test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="A,P,B,C,D,E,F")
    ap.add_argument("--golden", help="golden.json from another platform's run, to compare phase A against")
    ap.add_argument("--perf-reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="fewer repetitions, for a dry run")
    args = ap.parse_args()
    args.flm_bin = os.path.abspath(args.flm_bin)
    if args.baseline_bin:
        args.baseline_bin = os.path.abspath(args.baseline_bin)
    r = Run(args)
    print(f"clip sha256 {r.clips.digest()}", flush=True)
    for name in (OTHER, MODEL):
        subprocess.run([args.flm_bin, "pull", name], check=False)
    failed = False
    phases = args.phases.split(",")
    if "A" not in phases:
        phases = ["A"] + phases  # later phases compare against phase A's texts
    for ph in phases:
        r.phase = ph
        r.results["phases"][ph] = {"checks": []}
        print(f"=== phase {ph} ===", flush=True)
        t0 = time.monotonic()
        try:
            PHASES[ph](r)
        except BaseException as e:  # noqa: BLE001 - record and stop
            r.results["phases"][ph]["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            failed = True
        r.results["phases"][ph]["seconds"] = round(time.monotonic() - t0, 1)
        failed |= any(not c["ok"] and not c["soft"] for c in r.results["phases"][ph]["checks"])
        json.dump(r.results, open(os.path.join(r.out, "results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=str)
        if failed:
            print(f"phase {ph} failed; stopping", flush=True)
            break
    for s in r.servers:
        if s.proc and s.proc.poll() is None:
            s.stop()
    summary(r)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
