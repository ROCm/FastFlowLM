"""Fixtures for test_d3_compose, rendered by the upstream d3-turbo package modules themselves.

    python gen_compose_fixtures.py --package <vllm-sr/d3-turbo dir> --out fixtures [--seed 11] [--units 4000] [--requests 400]

Run it with a Python that has transformers, torch (CPU is enough), numpy and pillow (a venv with the upstream d3 requirements).
It never loads the model: the request fixtures run the package's own ``D3._system_one_composed`` on an instance
built without weights, whose encoder returns seeded random hidden rows, so the whole composed code path (parts,
rendering, tokenization, processor, state split, readout, Box decode, confidence, calibration, usage, errors)
is upstream's and only the 27B forward is replaced by a recorded stand-in.

Writes
  compose_units.jsonl     one record per call of a d3_compose function (check_references, split_string,
                          describe_parts, hoist_state, user_parts + render, tail/suffix, window_plan, group_times,
                          expected / order_edges / decode / confidence / calibrate on synthetic distributions, ...)
  compose_offsets.jsonl   rendered Box prompts with the tokenizer's ids, per-token UTF-8 byte lengths and upstream's
                          readout_offsets (or its error)
  compose_requests.jsonl  composed requests (assets, Box, choice / noul with images in options, errors) with the
                          upstream response and, per answered question, the recorded rows the C++ side replays
  decision_config.json    the package's config (answer codes, box_calibration)
  vocab_bytes.json        UTF-8 byte length of every token id the fixtures use
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------------------
# random material
# ------------------------------------------------------------------------------------------------------------
WORDS = ["the", "cat", "State", "x", "Question:", "0", "1", "true", "None", "A", "B", "option", "image", "[image 1]",
         "[image", "k]", "<a>", "<b>", "<ab>", "<a>b", "@@", "img", "ünïcödé", "é", "é", "中文", "日本語", "한국어",
         "ไทย", "हिन्दी", "🙂", "🤖🚀", "𝔘𝔫𝔦", "tab\t", "nl\n", "quote\"", "back\\slash", "ctrl\x01", " ", " ",
         "<|im_start|>", "<|im_end|>", "</think>", "  ", "\n\n", "Options:", "A: ", "top:", "bottom:", "visible:"]
# combining marks (Thai, Devanagari): transformers' Qwen2Tokenizer and tokenizer.json (FLM) split them differently
# (pre-existing, as for the smaller d3 models), so most request fixtures avoid them and FastFlowLM's real tokenizer can be checked
WORDS_PLAIN = [w for w in WORDS if w not in ("ไทย", "हिन्दी")]
WORD_POOL = [WORDS]
REFS = ["<a>", "<b>", "<ab>", "<a>b", "img", "[x]", "🙂", "é", "a", "ab", "中", "<c> d", "@@", "key:1", "x" * 40,
        "<é>", "😀😀", "  x ", "　z", "a.b", "{}", '"q"', "\\", "/", "[image 1]"]
SENTINELS = ["", "", ""]


def rand_text(rng: random.Random, refs, n: int | None = None) -> str:
    n = rng.randint(0, 6) if n is None else n
    pieces = []
    for _ in range(n):
        r = rng.random()
        if refs and r < 0.3:
            pieces.append(rng.choice(refs))
        else:
            pieces.append(rng.choice(WORD_POOL[0]))
        if rng.random() < 0.3:
            pieces.append(rng.choice([" ", "", "-", ", "]))
    return "".join(pieces)


def rand_value(rng: random.Random, refs, depth: int = 0):
    r = rng.random()
    if depth >= 3 or r < 0.45:
        r2 = rng.random()
        if r2 < 0.1 and refs:
            return rng.choice(refs)
        if r2 < 0.6:
            return rand_text(rng, refs)
        if r2 < 0.7:
            return rng.choice([0, 1, -3, 12345678901234567890, 2 ** 70])
        if r2 < 0.8:
            return rng.choice([0.5, 1e-05, 1e16, 100000.0, -2.5, 3.14159])
        if r2 < 0.9:
            return rng.choice([True, False, None])
        return ""
    if r < 0.7:
        return [rand_value(rng, refs, depth + 1) for _ in range(rng.randint(0, 4))]
    out = {}
    for _ in range(rng.randint(0, 4)):
        key = rand_text(rng, refs, rng.randint(0, 2)) or "k"
        out[key] = rand_value(rng, refs, depth + 1)
    return out


def jd(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def pack_edges(edges):
    """Edges as base64 float32 when every value is exactly a float32 (the runtime's softmax output), else a list."""
    flat = np.asarray(edges, dtype=np.float64)
    f32 = flat.astype(np.float32)
    if flat.ndim == 2 and bool(np.all(f32.astype(np.float64) == flat)):
        return {"edges_f32": base64.b64encode(f32.tobytes()).decode(), "n": int(flat.shape[1])}
    return {"edges": edges}


def part_json(part):
    kind, item = part
    if kind == "text":
        return ["t", item]
    if kind == "video":
        return ["v", item]
    return ["i", item] if isinstance(item, int) else ["a", item]


def call(fn, *args):
    """(ok, result) of a d3_compose call; ValueError text on failure, other exceptions by name."""
    try:
        return {"ok": fn(*args)}
    except ValueError as exc:
        return {"error": str(exc), "etype": "ValueError"}
    except Exception as exc:  # noqa: BLE001 - recorded: C++ must not raise ContractError for these
        return {"error": str(exc), "etype": type(exc).__name__}


# ------------------------------------------------------------------------------------------------------------
# synthetic distributions
# ------------------------------------------------------------------------------------------------------------
def softmax32(logits):
    t = np.asarray(logits, dtype=np.float32)
    t = t - t.max()
    e = np.exp(t)
    return (e / e.sum(dtype=np.float32)).astype(np.float32)


def rand_dist(rng: random.Random, nprng, n: int) -> list[float]:
    """A probability vector as the runtime hands it over: float32 softmax values widened to Python floats."""
    kind = rng.choice(["soft", "soft", "peak", "uniform", "twin", "onehot", "tail", "dyadic", "bin0", "binlast", "zeros"])
    if kind == "soft":
        p = softmax32(nprng.normal(0, rng.uniform(0.3, 6), n))
    elif kind == "peak":
        logits = nprng.normal(0, 0.5, n)
        logits[rng.randrange(n)] += rng.uniform(5, 30)
        p = softmax32(logits)
    elif kind == "uniform":
        p = np.full(n, 1.0 / n, dtype=np.float32)
    elif kind == "twin":
        logits = nprng.normal(0, 0.3, n)
        logits[rng.randrange(n)] += 12
        logits[rng.randrange(n)] += 12
        p = softmax32(logits)
    elif kind == "onehot":
        p = np.zeros(n, dtype=np.float32)
        p[rng.randrange(n)] = 1.0
    elif kind == "tail":
        p = softmax32(-np.arange(n) * rng.uniform(0.01, 0.6))
    elif kind == "dyadic":
        p = np.zeros(n, dtype=np.float64)
        k = rng.randint(1, min(n, 8))
        idx = sorted(rng.sample(range(n), k))
        for i in idx:
            p[i] = 1.0 / k
        p = p.astype(np.float32)
    elif kind == "bin0":
        p = softmax32(np.r_[40.0, np.zeros(n - 1)])
    elif kind == "binlast":
        p = softmax32(np.r_[np.zeros(n - 1), 40.0])
    else:
        p = np.zeros(n, dtype=np.float32)
        p[0] = p[-1] = 0.5
    return [float(x) for x in p]


# ------------------------------------------------------------------------------------------------------------
# unit fixtures
# ------------------------------------------------------------------------------------------------------------
def write_units(out: Path, dc, df, processor, codes, calibration, rng, nprng, n_units: int) -> None:
    from_json = json.loads
    records: list[dict] = []

    def add(kind, **fields):
        records.append({"kind": kind, **fields})

    for _ in range(n_units // 8):
        refs_in = rng.sample(REFS, rng.randint(0, 6))
        if rng.random() < 0.08:
            refs_in.append(rng.choice(["", " ", "\t\n", "　", "  ", SENTINELS[rng.randrange(3)], "x" + SENTINELS[0], "y" * 1100,
                                       "é" * 600]))
        res = call(dc.check_references, refs_in)
        add("check_references", refs=refs_in, **res)
    for count_refs in (60, 64, 65, 72):   # the 65,536-byte total: 1,000-byte references
        refs_in = [f"r{i:03d}-" + "q" * 994 for i in range(count_refs)]
        add("check_references", refs=refs_in, **call(dc.check_references, refs_in))

    for _ in range(n_units):
        refs_in = rng.sample(REFS, rng.randint(0, 6))
        refs = dc.check_references(refs_in)
        which = rng.random()
        if which < 0.25:
            text = rand_text(rng, refs, rng.randint(0, 9))
            if refs and rng.random() < 0.15:
                text = rng.choice(refs)
            add("split_string", refs=refs, text=text, parts=[part_json(p) for p in dc.split_string(text, refs)])
        elif which < 0.55:
            value = rand_value(rng, refs)
            if rng.random() < 0.05:
                value = rng.choice([[rng.choice(refs)] if refs else [], {"k": rng.choice(refs)} if refs else {}, ["a" + SENTINELS[rng.randrange(3)]]])
            res = call(dc.describe_parts, value, refs)
            if "ok" in res:
                res = {"ok": [part_json(p) for p in res["ok"]]}
            add("describe_parts", refs=refs, value=jd(value), **res)
        elif which < 0.8:
            value = rand_value(rng, refs)
            front = rng.randint(0, 4)
            hoisted, uses = dc.hoist_state(value, refs, front)
            add("hoist_state", refs=refs, state=jd(value), front=front, hoisted=jd(hoisted), uses=uses)
        else:
            state = rand_value(rng, refs)
            kind = rng.choice(["choice", "noul", "box", "box", "track"])
            instr = rng.choice([rand_text(rng, refs), rand_text(rng, refs), "", None, rand_value(rng, refs), rand_text(rng, refs, 3)])
            q: dict = {"type": kind}
            if instr is not None or rng.random() < 0.5:
                q["instructions"] = instr
            if kind == "choice":
                crit = {}
                for _ in range(rng.randint(1, 5)):
                    k = rand_text(rng, [], 1) or "k"
                    crit[k] = rng.choice([None, rand_text(rng, refs), rand_text(rng, refs), rand_value(rng, refs), "", 0])
                q["criteria"] = crit
            elif kind == "noul" and rng.random() < 0.7:
                q["criteria"] = {key: rng.choice([rand_text(rng, refs), rand_value(rng, refs), "", None]) for key in rng.sample(["false", "true"], rng.randint(0, 2))}
            at = None
            if kind == "track" or (kind == "box" and rng.random() < 0.3):
                at = round(rng.uniform(0, 70), rng.choice([0, 1, 3, 6]))
                if rng.random() < 0.1:
                    at = rng.choice([0.05, 0.25, 2.25, 0.15, 1e-3])
            question = dict(q)
            if at is not None:
                question["at"] = at
            n_front, n_videos = rng.randint(0, 3), rng.randint(0, 2)
            res = call(dc.user_parts, state, question, codes, n_front, n_videos, refs)
            rec = {"refs": refs, "state": jd(state), "question": jd(q), "at": at, "n_front": n_front, "n_videos": n_videos}
            if "ok" in res:
                parts = res["ok"]
                presence = kind == "track"
                suffix = dc.suffix(presence) if kind in ("box", "track") else ""
                rec["parts"] = [part_json(p) for p in parts]
                rec["text"] = dc.render(processor, parts, suffix)
                rec["suffix"] = suffix
            else:
                rec["error"], rec["etype"] = res["error"], res["etype"]
            add("user_parts", **rec)

    # tail / suffix / seconds
    for presence in (False, True):
        add("suffix", presence=presence, text=dc.suffix(presence), markers=list(dc.markers(presence)))
        for at in (None, 0.0, 0.25, 0.05, 1.25, 59.95, 12.345):
            add("tail_text", presence=presence, at=at, text=dc.tail_text(codes, at, presence))
    for _ in range(200):
        v = rng.choice([rng.uniform(0, 100), round(rng.uniform(0, 100), 2) + 0.05, rng.randint(0, 900) / 20, rng.uniform(0, 3e5)])
        add("format_seconds", value=v, text=dc.format_seconds(v))
        r3 = round(v, 3)
        add("round3", value=v, rounded=r3)

    # math
    points_ref = [list(p) for p in dc._POINTS]
    add("points", points=points_ref)
    sizes = [128] * 12 + [2, 7, 8, 9, 100, 129, 255, 300]
    for _ in range(max(40, n_units // 3)):
        n = rng.choice(sizes)
        edges = [rand_dist(rng, nprng, n) for _ in range(4)]
        if rng.random() < 0.1:
            edges[1] = list(edges[0])
        if rng.random() < 0.05:
            edges[3] = list(edges[2])
        try:
            box = dc.decode(edges)
        except Exception as exc:  # noqa: BLE001
            add("decode", **pack_edges(edges), error=str(exc))
            continue
        raw = dc.confidence(edges, box)
        cal = dc.calibrate(raw, calibration)
        # a second, arbitrary reported box against the same edges (the calibration/IoU chain on its own)
        other = dict(box)
        if rng.random() < 0.4:
            other = {"top": rng.random() * 0.5, "bottom": 0.5 + rng.random() * 0.5, "left": rng.random() * 0.5, "right": 0.5 + rng.random() * 0.5}
        add("decode", **pack_edges(edges), box=box, raw=raw, calibrated=cal, other=other, raw_other=dc.confidence(edges, other))
    # exact cdf ties with the quasi-random points
    for _ in range(80):
        edges = []
        for d in range(4):
            x = dc._POINTS[rng.randrange(512)][d]
            p = [0.0] * 128
            p[rng.randrange(0, 100)] = x
            p[rng.randrange(100, 128)] += 1.0 - x
            edges.append(p)
        box = dc.decode(edges)
        add("decode", **pack_edges(edges), box=box, raw=dc.confidence(edges, box), calibrated=dc.calibrate(dc.confidence(edges, box), calibration), other=box,
            raw_other=dc.confidence(edges, box))
    for _ in range(200):
        a, b = rng.random(), rng.random()
        if rng.random() < 0.3:
            b = a
        if rng.random() < 0.1:
            a, b = rng.choice([0.0, 1.0, 0.001, 0.999]), rng.choice([0.0, 1.0, 0.0005, 0.9995])
        add("order_edges", a=a, b=b, out=list(dc.order_edges(a, b)))
    table = [list(k) for k in calibration]
    for _ in range(300):
        v = rng.choice([rng.random(), k[0] if (k := rng.choice(table)) else 0.0, rng.randint(0, 512) / 512, 0.0, 1.0, -0.1, 1.2])
        add("calibrate", value=v, table=table, out=dc.calibrate(v, table))
    for _ in range(40):
        t = sorted({round(rng.random(), 3) for _ in range(rng.randint(1, 6))})
        t = [[x, round(rng.random(), 3)] for x in t]
        v = rng.random()
        add("calibrate", value=v, table=t, out=dc.calibrate(v, t))
    add("calibrate", value=0.4, table=[], out=0.4)
    add("calibrate", value=0.4, table=None, out=0.4)
    for _ in range(100):
        p = rand_dist(rng, nprng, 2)
        if sum(p) > 0:
            add("visible", p=p, out=float(p[1] / sum(p)))

    # windows
    fpss = [0.1, 6, 12, 23.976, 24, 25, 29.97, 30, 60, 120, 1000, 15.0, 7.5]
    for _ in range(300):
        fps = rng.choice(fpss + [rng.uniform(0.5, 120)])
        rate = rng.choice([1, 2])
        total = rng.choice([1, 2, 3, 5, 31, 32, 33, 64, 65, rng.randint(1, 9000), rng.randint(1, 400), max(1, int(rng.uniform(0.1, 120) * fps))])
        if int(total / (fps / (2.0 * rate))) > 700:      # keep the fixture small (a 300-s clip is allowed upstream)
            continue
        plan = dc.window_plan(total, fps, rate)
        add("window_plan", total=total, fps=fps, rate=rate, windows=plan, times=[dc.group_times(w, fps) for w in plan])

    with open(out / "compose_units.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("units:", len(records))


# ------------------------------------------------------------------------------------------------------------
# tokenizer byte lengths and offsets
# ------------------------------------------------------------------------------------------------------------
def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class ByteLengths:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.dec = byte_decoder()
        self.added = {t.content: len(t.content.encode()) for t in tokenizer.added_tokens_decoder.values()}
        self.cache: dict[int, int] = {}

    def __call__(self, i: int) -> int:
        if i not in self.cache:
            token = self.tok.convert_ids_to_tokens(i)
            if token in self.added:
                self.cache[i] = self.added[token]
            else:
                self.cache[i] = sum(1 if ch in self.dec else len(ch.encode()) for ch in token)
        return self.cache[i]


def write_offsets(out: Path, dc, processor, codes, rng, n: int, lengths: ByteLengths) -> None:
    tok = processor.tokenizer
    n_ok = n_err = 0
    with open(out / "compose_offsets.jsonl", "w") as f:
        for i in range(n):
            presence = rng.random() < 0.3
            refs = rng.sample(REFS, rng.randint(0, 3)) if rng.random() < 0.3 else []
            refs = dc.check_references(refs)
            state = rand_value(rng, [])
            instr = rand_text(rng, refs, rng.randint(0, 8))
            q = {"type": "track" if presence else "box", "instructions": instr or "the thing"}
            at = round(rng.uniform(0, 70), 1) if presence or rng.random() < 0.2 else None
            hoisted, uses = dc.hoist_state(state, refs, 0)
            parts = dc.user_parts(hoisted, dict(q, **({"at": at} if at is not None else {})), codes, 0, 0, refs)
            text = dc.render(processor, parts, dc.suffix(presence))
            if rng.random() < 0.05:
                text = text + rng.choice([" ", "\n", "x"])      # no longer ends with the markers
            if rng.random() < 0.05:
                text = text.replace("top:", "top: ", 1) if rng.random() < 0.5 else text[:-1]
            ids = tok(text, add_special_tokens=False)["input_ids"]
            res = call(dc.readout_offsets, tok, text, presence)
            rec = {"text": text, "presence": presence, "ids": ids, "bytes": [lengths(i) for i in ids]}
            if "ok" in res:
                rec["offsets"] = res["ok"]
                n_ok += 1
            else:
                rec["error"] = res["error"]
                n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("offsets:", n_ok, "ok", n_err, "errors")


# ------------------------------------------------------------------------------------------------------------
# composed requests through the package's own _system_one_composed (model replaced by a recorded stand-in)
# ------------------------------------------------------------------------------------------------------------
def png(rng: random.Random, w: int, h: int, fmt: str = "PNG") -> str:
    """A blocky random image (low-entropy, so the fixtures stay small); PNG, JPEG or WebP data URL."""
    from PIL import Image

    cell = rng.choice([4, 8, 16])
    gw, gh = (w + cell - 1) // cell, (h + cell - 1) // cell
    grid = np.frombuffer(bytes(rng.randrange(256) for _ in range(gw * gh * 3)), dtype=np.uint8).reshape(gh, gw, 3)
    arr = np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)[:h, :w]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr), "RGB").save(buf, format=fmt)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[fmt]
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def build_model(rt, package: Path, hidden: int, seed: int):
    import torch

    from d3_format import state_end_char as rt_state_end_char

    cfg = json.loads((package / "decision_config.json").read_text())
    model = object.__new__(rt.D3)
    model.torch = torch
    model.device = torch.device("cpu")
    model.processor = rt.load_processor(package)
    model.codes = cfg["codes"]
    model.model_name = "d3-turbo"
    model.blocked = True
    model.max_length = 32768
    model.image_unavailable = None
    systemone = cfg["systemone"]
    model.question_types = ("choice", "noul", "score") + tuple(k for k in ("box", "track") if k in systemone["question_types"])
    model.image_placements = ("state",) + tuple(p for p in ("instructions", "criteria") if p in systemone["image_placements"])
    model.box_calibration = systemone["box_calibration"]
    model.temperature = 1.0
    model.readout_dtype = "float32"
    g = torch.Generator().manual_seed(seed)
    model.readout = torch.randn(255, hidden, generator=g, dtype=torch.float32)
    model.hidden = hidden
    orig_load = model.load_images
    model.load_images = lambda images, strict=True: orig_load(images, strict=True)   # the REST server's strict loading
    model.blocks = lambda width, widths, states: None
    model.captured = []
    model.tokenized = {}
    tok = model.processor.tokenizer

    class Recording(type(tok)):
        def __call__(self, text, *a, **kw):
            result = super().__call__(text, *a, **kw)
            if kw.get("return_offsets_mapping") and isinstance(text, str):
                model.tokenized[text] = result["input_ids"]
            return result

    tok.__class__ = Recording

    def media(inputs):
        model._last_ids = inputs["input_ids"][0].tolist()
        return None, None

    model._media_embeddings = media

    class Encoder:
        def __call__(self, embeds, mask, mrope, blocks=None):
            ids = model._last_ids
            gen = torch.Generator().manual_seed(hash(tuple(ids)) % (2 ** 31))
            scale = float(torch.rand(1, generator=gen) * 2.5 + 0.4)
            h = (torch.randn(len(ids), model.hidden, generator=gen) * scale).to(torch.bfloat16)
            model._hidden = h
            return (h,)

    model.encoder = Encoder()
    orig_row = model._composed_row

    def row(text, hoisted, row_images, n_front, clips, offsets, counts):
        values, state, block = orig_row(text, hoisted, row_images, n_front, clips, offsets, counts)
        # the state segment's own tokenization (FastFlowLM measures the state block with it): record it too
        segment = text[: rt_state_end_char(text, hoisted)]
        model.tokenized[segment] = model.processor.tokenizer(segment, add_special_tokens=False)["input_ids"]
        ids = model._last_ids
        width = len(ids)
        picked = model._hidden[[width - 1 - o for o in offsets]].float().tolist()
        model.captured.append({"text": text, "width": width, "crc": zlib.crc32(np.asarray(ids, dtype=np.int64).tobytes()), "state": state,
                               "block": block, "offsets": offsets, "counts": counts, "values": values, "hidden": picked})
        return values, state, block

    model._composed_row = row
    return model, cfg


def rand_question(rng: random.Random, assets: list[str], codes) -> dict:
    kind = rng.choice(["choice", "choice", "noul", "box", "box", "box", "score", "bad", "track", "boxat"])
    def mention(prob):
        return rng.choice(assets) if assets and rng.random() < prob else None
    def text(prob=0.35):
        pieces = [rng.choice(WORDS[:12] + ["pick", "find the", "which"]) for _ in range(rng.randint(0, 3))]
        m = mention(prob)
        if m:
            pieces.insert(rng.randint(0, len(pieces)), m)
        return " ".join(pieces)
    if kind == "box":
        q = {"type": "box", "instructions": text(0.3) if rng.random() < 0.9 else rng.choice(["", 5, None])}
        if rng.random() < 0.08:
            q["extra"] = 1
        return q
    if kind == "choice":
        crit = {}
        for _ in range(rng.randint(1, 5)):
            r = rng.random()
            key = rng.choice(["a", "b", "c", "dd", "opt1", "é", "x y"])
            if r < 0.3:
                crit[key] = None
            elif r < 0.7:
                crit[key] = mention(0.7) or text(0.1)
            elif r < 0.85:
                crit[key] = {"img": mention(0.8), "n": 1} if assets else {"n": 1}
            else:
                crit[key] = [mention(0.8), "z"] if assets else ["z"]
        return {"type": "choice", "instructions": text(), "criteria": crit}
    if kind == "noul":
        q = {"type": "noul", "instructions": text()}
        if rng.random() < 0.5:
            q["criteria"] = {"true": mention(0.6) or "yes"}
        return q
    if kind == "score":
        return {"type": "score", "instructions": text(0.2), "criteria": ["bad", "ok", "good"]}
    if kind == "track":
        return {"type": "track", "instructions": "the cat"}
    if kind == "boxat":
        return {"type": "box", "instructions": "the cat", "at": 1.5}
    return rng.choice([{"type": "nope"}, {"type": "choice"}, 5, {"type": "noul", "criteria": {"maybe": 1}}, {"type": "choice", "criteria": {}}])


def write_requests(out: Path, rt, dc, package: Path, rng, n: int, lengths: ByteLengths, hidden: int) -> None:
    import torch

    WORD_POOL[0] = WORDS_PLAIN
    model, cfg = build_model(rt, package, hidden, 1234)
    model.readout_dtype = "float32"
    refusal_model, _ = build_model(rt, package, hidden, 1234)
    # a package that does not declare the Box / image-placement features: the per-question refusals
    refusal_model.question_types = ("choice", "noul", "score")
    refusal_model.image_placements = ("state",)
    n_ok = n_err = 0
    vocab: dict[int, int] = {}
    with open(out / "compose_requests.jsonl", "w") as f:
        for i in range(n):
            WORD_POOL[0] = WORDS if rng.random() < 0.12 else WORDS_PLAIN
            use_refusal = i % 17 == 16
            m = refusal_model if use_refusal else model
            n_assets = rng.randint(0, 4)
            pool = ["<a>", "<b>", "<c>", "[img]", "🙂", "ünï", "X1"]
            names = rng.sample(pool, n_assets)
            assets = {}
            for name in names:
                assets[name] = png(rng, rng.choice([16, 40, 64, 100, 130, 257]), rng.choice([16, 33, 64, 120, 300]), rng.choice(["PNG", "PNG", "JPEG", "WEBP"]))
            unused = {}
            if rng.random() < 0.2:
                unused = {"<unused>": "not an image at all"}
            used_names = names
            state = rand_value(rng, used_names, 1) if rng.random() < 0.9 else rng.choice(["", None, 0])
            if used_names and rng.random() < 0.6:
                state = {"here": rng.choice(used_names), "list": [rng.choice(used_names), "text " + rng.choice(used_names)], "n": rand_text(rng, [])}
            if rng.random() < 0.15:
                state = rand_text(rng, used_names, 5)
            n_images = rng.choice([0, 0, 1, 2])
            images = [png(rng, rng.choice([32, 77, 200]), rng.choice([32, 90, 150])) for _ in range(n_images)]
            questions = {}
            for j in range(rng.randint(1, 6)):
                questions[f"q{j}"] = rand_question(rng, used_names, codes := cfg["codes"])
            body = {"state": state, "questions": questions}
            if images:
                body["images"] = images
            if assets or unused or rng.random() < 0.1:
                body["assets"] = {**assets, **unused} if rng.random() < 0.8 else {**unused, **assets}
            rec = {"id": f"req{i}", "body": json.dumps(body, ensure_ascii=False), "refusal_model": use_refusal}
            if rng.random() < 0.03:   # request-level corner cases
                body["assets"] = rng.choice([{"": png(rng, 32, 32)}, {"a" * 2000: png(rng, 32, 32)}, {"x": png(rng, 32, 32)},
                                             {"ok": "data:image/gif;base64,AAAA"}, {"ok": 5}, {"ok": "data:image/png;base64,@@"}])
                rec["body"] = json.dumps(body, ensure_ascii=False)
            m.captured.clear()
            m.tokenized.clear()
            try:
                resp = m._system_one(body.get("state"), body["questions"], body.get("images"), None, body.get("assets"))
                rec["response"] = resp
                rec["rows"] = list(m.captured)
                n_ok += 1
            except ValueError as exc:
                rec["detail"] = str(exc)
                n_err += 1
            except Exception as exc:  # noqa: BLE001 - upstream crashes (HTTP 500); FLM answers the question inline instead
                rec["crash"] = type(exc).__name__
                n_err += 1
            rec["tok"] = dict(m.tokenized)
            for ids in m.tokenized.values():
                for t in ids:
                    vocab[t] = lengths(t)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # plain token bytes for every id the C++ test may meet, and the readout of the stand-in model
    (out / "readout.json").write_text(json.dumps({"hidden": hidden, "weight": model.readout.tolist()}))
    (out / "request_vocab_bytes.json").write_text(json.dumps({str(i): b for i, b in sorted(vocab.items())}))
    print("requests:", n_ok, "answered", n_err, "refused at request level")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--units", type=int, default=4000)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--offsets", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    package, out = Path(args.package), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(package))
    import d3_compose as dc
    import d3_format as df
    import d3_runtime as rt

    cfg = json.loads((package / "decision_config.json").read_text())
    shutil.copy(package / "decision_config.json", out / "decision_config.json")
    codes, calibration = cfg["codes"], cfg["systemone"]["box_calibration"]
    processor = rt.load_processor(package)
    lengths = ByteLengths(processor.tokenizer)
    rng, nprng = random.Random(args.seed), np.random.default_rng(args.seed)
    write_units(out, dc, df, processor, codes, calibration, rng, nprng, args.units)
    write_offsets(out, dc, processor, codes, rng, args.offsets, lengths)
    write_requests(out, rt, dc, package, rng, args.requests, lengths, args.hidden)
    # byte lengths of every id of the fixtures
    ids = set()
    for line in open(out / "compose_offsets.jsonl"):
        ids.update(json.loads(line)["ids"])
    merged = {int(k): v for k, v in json.loads((out / "request_vocab_bytes.json").read_text()).items()}
    merged.update({i: lengths(i) for i in ids})
    (out / "request_vocab_bytes.json").unlink()
    (out / "vocab_bytes.json").write_text(json.dumps({str(i): merged[i] for i in sorted(merged)}))


if __name__ == "__main__":
    main()
