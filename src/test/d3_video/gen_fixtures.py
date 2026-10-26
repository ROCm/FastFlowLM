#!/usr/bin/env python
"""Goldens for the d3 video input side, produced by the real upstream code.

Run with transformers 5.17, torchvision 0.24 and opencv-python-headless, CPU only (D3_UPSTREAM = the upstream
d3-turbo package's python modules, D3_TURBO_ROOT = the package folder):

    HIP_VISIBLE_DEVICES= python gen_fixtures.py OUTDIR --compact  # the committed set (small frame budget; resized
                                                                  # goldens above 210 kB dropped: md5 decides)
    HIP_VISIBLE_DEVICES= python gen_fixtures.py OUTDIR            # the full small set (all resized goldens, ~4 MB)
    HIP_VISIBLE_DEVICES= python gen_fixtures.py OUTDIR --big      # real 0.2 MP budget, not committed (tens of MB)

What it records (cases.json + <case>.resized.u8):
  * the sampled frame indices / padded frame list (upstream video_sample_indices, _video, _frames_video);
  * the processor's output for the clip: video_grid_thw, md5 of pixel_values_videos (fp32 bytes), the
    resized uint8 frames (what torchvision's resize gives before rescale/normalize/patchify);
  * the per-video text (replace_video_token), the rendered user prompt with its placeholders and the
    expanded prompt (get_text_with_replacements), with token ids of the expanded video text;
  * tables for video_sample_indices / d3_compose.window_plan / group_times;
  * decode goldens: small clips written with the ffmpeg CLI, decoded by upstream's _decode_video (OpenCV).

The synthetic frames are a function of (seed, noise, frame, y, x, c) in 32-bit integer arithmetic that
test_d3_video.cpp repeats, so no pixel data has to be stored for the inputs.
"""
import argparse, hashlib, json, math, os, struct, subprocess, sys
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

UP = os.environ.get("D3_UPSTREAM", ".")
ROOT = os.environ.get("D3_TURBO_ROOT", "vllm-sr/d3-turbo")
sys.path.insert(0, UP)

import numpy as np
import torch

M32 = 0xFFFFFFFF
IMAGE_PADS = 64


def pattern(seed, noise, frames, h, w):
    """uint8 [len(frames), h, w, 3]; the C++ twin is synth_frame() in test_d3_video.cpp."""
    f = np.asarray(frames, dtype=np.uint32).reshape(-1, 1, 1, 1)
    y = np.arange(h, dtype=np.uint32).reshape(1, -1, 1, 1)
    x = np.arange(w, dtype=np.uint32).reshape(1, 1, -1, 1)
    c = np.arange(3, dtype=np.uint32).reshape(1, 1, 1, 3)
    with np.errstate(over="ignore"):
        s = np.uint32(seed)
        hv = (s * np.uint32(0x9E3779B1)) ^ (f * np.uint32(0x85EBCA6B)) ^ (y * np.uint32(0xC2B2AE35)) \
            ^ (x * np.uint32(0x27D4EB2F)) ^ (c * np.uint32(0x165667B1))
        hv = hv ^ (hv >> np.uint32(15))
        hv = hv * np.uint32(0x2C1B3C6D)
        hv = hv ^ (hv >> np.uint32(12))
        hv = hv * np.uint32(0x297A2D39)
        hv = hv ^ (hv >> np.uint32(15))
        n = ((hv >> np.uint32(24)) * np.uint32(noise)) >> np.uint32(8)
        base = (x * np.uint32(3) + y * np.uint32(5) + f * np.uint32(11) + c * np.uint32(40)) % np.uint32(512)
        tri = np.where(base < 256, base, np.uint32(511) - base)
        block = ((x // np.uint32(16) + y // np.uint32(16)) & np.uint32(1)) * np.uint32(40)
        v = np.minimum(tri + n + block, np.uint32(255))
    return np.broadcast_to(v, (len(frames), h, w, 3)).astype(np.uint8).copy()


def md5(b):
    return hashlib.md5(b).hexdigest()


def fnv64(data: bytes, h=0xCBF29CE484222325):
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load():
    from transformers import AutoProcessor
    import d3_runtime as R
    import d3_compose as DC
    processor = AutoProcessor.from_pretrained(ROOT)
    R.configure_video_processor(processor)
    return processor, R, DC


def run_case(processor, R, DC, case, outdir, max_video_tokens):
    vp = processor.video_processor
    vp.max_video_tokens = max_video_tokens
    out = {"name": case["name"], "h": case["h"], "w": case["w"], "seed": case["seed"], "noise": case["noise"],
           "fps": case["fps"], "total": case["total"], "mode": case["mode"], "max_video_tokens": max_video_tokens}
    # --- sampling and the Video object, as upstream builds it
    if case["mode"] == "decode":
        indices = R.video_sample_indices(case["total"], case["fps"])
        frames = [f for f in pattern(case["seed"], case["noise"], indices, case["h"], case["w"])]
        video = R._video(frames, list(indices), case["fps"], case["total"])
    else:  # frame list: [T, h, w, 3] array, thinned to 32, 2 fps
        arr = pattern(case["seed"], case["noise"], list(range(case["total"])), case["h"], case["w"])
        video = R._frames_video(arr)
        out["fps"] = R.VIDEO_FPS
    out["indices"] = list(video.indices)
    out["n_frames"] = len(video.frames)
    out["video_fps"] = video.fps
    out["video_total"] = video.total_frames
    # --- the processor
    try:
        planned = vp.get_num_of_video_patches(len(video.frames), video.size[1], video.size[0]) // vp.merge_size ** 2
        media = dict(vp(videos=[video.frames], video_metadata=[video.metadata()], return_metadata=True, return_tensors="pt"))
    except ValueError as exc:
        out["error"] = str(exc)
        return out
    grid = media["video_grid_thw"][0].tolist()
    out["grid"] = grid
    out["planned_tokens"] = planned
    pv = media["pixel_values_videos"].contiguous()
    assert pv.dtype == torch.float32
    out["pixel_md5"] = md5(pv.numpy().tobytes())
    out["pixel_count"] = pv.numel()
    # resized uint8 frames (the resize the processor applies), the +-1 LSB reference
    t = torch.from_numpy(np.ascontiguousarray(video.frames)).permute(0, 3, 1, 2).unsqueeze(0)
    nf = t.shape[1]
    max_pixels = vp.size["longest_edge"]
    frame_cap = vp.max_video_tokens * 32 * 32
    ppf = max(min(frame_cap, vp.size["longest_edge"] // nf), int(vp.size["shortest_edge"] * 1.05))
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    rh, rw = smart_resize(num_frames=nf, height=video.size[1], width=video.size[0], factor=32, temporal_factor=2,
                          min_pixels=vp.size["shortest_edge"], max_pixels=ppf * nf)
    from torchvision.transforms.v2 import functional as tvF
    resized = tvF.resize(t, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    assert [grid[1], grid[2]] == [rh // 16, rw // 16]
    out["resized"] = [rh, rw]
    r = resized[0].permute(0, 2, 3, 1).contiguous().numpy()
    with open(os.path.join(outdir, case["name"] + ".resized.u8"), "wb") as f:
        f.write(r.tobytes())
    # self-check of the layout: patchify(normalize(resized)) == the processor's tensor
    x = (resized.to(torch.float32) - 127.5) / 127.5
    patches, gt, gh, gw = vp.patchify(x, 16, 2, 2)
    assert torch.equal(patches[0], pv), "normalize/patchify model differs from the processor"
    # --- text
    video_text = processor.replace_video_token(dict(media), 0)
    out["video_text"] = video_text
    out["group_stamps"] = [s for s in __import__("re").findall(r"<([0-9.]+) seconds>", video_text)]
    tok = processor.tokenizer
    ids_full = tok(video_text, add_special_tokens=False)["input_ids"]
    out["video_text_tokens"] = len(ids_full)
    # one pad per group, tokenized, then expanded by run length
    per_group = (grid[1] * grid[2]) // 4
    import re
    short = video_text.replace("<|video_pad|>" * per_group, "<|video_pad|>")
    ids_short = tok(short, add_special_tokens=False)["input_ids"]
    pad = tok.convert_tokens_to_ids("<|video_pad|>")
    assert len(ids_short) - short.count("<|video_pad|>") + short.count("<|video_pad|>") * per_group == len(ids_full)
    out["video_pad_id"] = pad
    out["ids_one_pad_per_group"] = ids_short
    # --- prompt: chat template with the video placeholder, then the processor's replacement
    codes = json.load(open(os.path.join(ROOT, "decision_config.json")))["codes"]
    prompts = []
    for p in case.get("prompts", []):
        q = R.normalize_question(p["question"])
        msgs = R.video_messages("d3", p["state"], q.rendered, codes, p.get("images", 0), 1)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image_texts = ["<|image_pad|>" * IMAGE_PADS] * p.get("images", 0)   # one 256 x 256 image: 64 tokens
        expanded = processor.get_text_with_replacements([text], image_texts, [video_text])[0][0]
        prompts.append({"state": p["state"], "question": p["question"], "images": p.get("images", 0),
                        "text": text, "expanded": expanded})
    out["prompts"] = prompts
    return out


def mrope_goldens(processor, R, cases):
    """Qwen3.5's get_rope_index on real tokenized prompts with images and videos (positions are axis-major [3][n])."""
    import types
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    cfg = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))
    fake = types.SimpleNamespace(config=cfg)
    fake.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, fake)
    tok = processor.tokenizer
    image_id = tok.convert_tokens_to_ids("<|image_pad|>")
    video_id = tok.convert_tokens_to_ids("<|video_pad|>")
    codes = json.load(open(os.path.join(ROOT, "decision_config.json")))["codes"]
    good = [c for c in cases if "grid" in c]
    picks = [("video", [good[0]], 0), ("image+video", [good[2]], 1), ("two videos", [good[1], good[3]], 0),
             ("two videos + 2 images", [good[4], good[5]], 2), ("portrait", [good[3]], 0)]
    out = []
    for name, vids, n_images in picks:
        q = R.normalize_question({"type": "noul", "instructions": "Is it there?"})
        msgs = R.video_messages("d3", "state", q.rendered, codes, n_images, len(vids))
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image_texts = ["<|image_pad|>" * IMAGE_PADS] * n_images
        expanded = processor.get_text_with_replacements([text], image_texts, [c["video_text"] for c in vids])[0][0]
        ids = tok(expanded, add_special_tokens=False, return_tensors="pt")["input_ids"]
        types_ = processor.create_mm_token_type_ids(ids)
        image_grid = torch.tensor([[1, 16, 16]] * n_images) if n_images else None
        video_grid = torch.tensor([c["grid"] for c in vids])
        pos, _ = Qwen3_5Model.get_rope_index(fake, ids, torch.as_tensor(types_), image_grid, video_grid.clone())
        groups = [[16, 16]] * n_images
        for c in vids:
            groups += [[c["grid"][1], c["grid"][2]]] * c["grid"][0]
        out.append({"name": name, "ids": ids[0].tolist(), "image_id": image_id, "video_id": video_id, "groups": groups,
                    "positions": pos[:, 0].reshape(-1).tolist(), "next": int(pos.max()) + 1})
    return {"cases": out}


def sampling_tables(R, DC):
    explicit = []
    for total, fps in [(1, 24), (2, 24), (3, 24), (4, 24), (5, 2), (9, 24), (33, 2), (64, 2), (65, 2), (300, 30),
                       (324, 29.97), (324, 30000 / 1001), (7, 0.1), (1000, 1000), (45, 6), (100, 12.5), (3, 120)]:
        explicit.append({"total": total, "fps": fps, "sample": R.video_sample_indices(total, fps),
                         "windows": {str(r): [{"indices": w, "times": DC.group_times(w, fps)} for w in DC.window_plan(total, fps, r)]
                                     for r in (1, 2)}})
    # exhaustive sweeps, hashed: FNV-1a over the little-endian bytes of every result
    h = 0xCBF29CE484222325
    fps_list = [0.1, 6, 12, 23.976, 24, 25, 29.97, 30, 60, 120, 1000]
    count = 0
    h_s, h_w = h, h
    for fps in fps_list:
        for total in range(1, 3001):
            s = R.video_sample_indices(total, fps)
            h_s = fnv64(struct.pack("<ii", total, len(s)) + struct.pack("<%di" % len(s), *s), h_s)
            for rate in (1, 2):
                if total > 400 and total % 7:  # thin the larger windows to keep generation fast
                    continue
                ws = DC.window_plan(total, fps, rate)
                for w in ws:
                    t = DC.group_times(w, fps)
                    h_w = fnv64(struct.pack("<iii", total, rate, len(w)) + struct.pack("<%di" % len(w), *w)
                                + struct.pack("<%dd" % len(t), *t), h_w)
                count += 1
    return {"explicit": explicit, "fps_list": fps_list, "sweep_total_max": 3001, "sample_fnv": h_s, "window_fnv": h_w}


def decode_goldens(R, outdir):
    """Tiny clips through the ffmpeg CLI, decoded by upstream's _decode_video (OpenCV) as the reference."""
    clips = [
        ("h264.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18"], 48, 32, 40, 10),
        ("vp9.webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "30", "-deadline", "realtime"], 48, 32, 36, 12),
        ("h264.mkv", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18"], 64, 48, 30, 15),
        ("h264.mov", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18"], 40, 56, 24, 12),
        ("anim.gif", [], 48, 32, 28, 8),
    ]
    res = []
    for name, codec, w, h, nframes, fps in clips:
        path = os.path.join(outdir, name)
        raw = pattern(7, 12, list(range(nframes)), h, w)
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(fps), "-i", "-"] + codec + [path], input=raw.tobytes(), capture_output=True)
        if p.returncode:
            print("ffmpeg failed for", name, p.stderr.decode(), file=sys.stderr)
            continue
        data = open(path, "rb").read()
        try:
            v = R._decode_video(data, strict=True)
        except ValueError as exc:
            res.append({"file": name, "error": str(exc)})
            continue
        fr = np.ascontiguousarray(v.frames)
        with open(os.path.join(outdir, name + ".frames.u8"), "wb") as f:
            f.write(fr.tobytes())
        entry = {"file": name, "w": w, "h": h, "n_frames": len(fr), "indices": v.indices, "fps": v.fps,
                 "total": v.total_frames, "frames_file": name + ".frames.u8", "frames_md5": md5(fr.tobytes()), "windows": {}}
        # upstream's one-pass window decode (track), at both rates
        for rate in (1, 2):
            try:
                ws = R._decode_windows(data, rate)
                entry["windows"][str(rate)] = [{"indices": x.indices, "fps": x.fps, "total": x.total_frames,
                                                "md5": md5(np.ascontiguousarray(x.frames).tobytes())} for x in ws]
            except ValueError as exc:
                entry["windows"][str(rate)] = {"error": str(exc)}
        res.append(entry)
    # clips with a display matrix: OpenCV rotates them upright
    for angle in (90, 180, 270):
        name = f"rot{angle}.mp4"
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(angle), "-i", os.path.join(outdir, "h264.mp4"),
                            "-c", "copy", os.path.join(outdir, name)], capture_output=True)
        if p.returncode:
            continue
        v = R._decode_video(open(os.path.join(outdir, name), "rb").read(), strict=True)
        fr = np.ascontiguousarray(v.frames)
        res.append({"file": name, "w": fr.shape[2], "h": fr.shape[1], "n_frames": len(fr), "indices": v.indices, "fps": v.fps,
                    "total": v.total_frames, "frames_md5": md5(fr.tobytes()), "rotation": angle})
    return res


SMALL = [
    # name, mode, total, fps, h, w, seed, noise
    ("c01_min4", "decode", 4, 24.0, 96, 160, 1, 0),
    ("c02_odd_dims", "decode", 7, 6.0, 123, 77, 2, 24),
    ("c03_cap32", "decode", 33, 2.0, 64, 96, 3, 255),
    ("c04_portrait", "decode", 20, 4.0, 240, 135, 4, 40),
    ("c05_narrow", "decode", 60, 30.0, 250, 100, 5, 8),
    ("c06_20frames", "decode", 100, 10.0, 45, 71, 6, 100),
    ("c07_odd_count", "decode", 5, 1.0, 31, 40, 7, 16),
    ("c08_list_thin", "list", 33, 2.0, 80, 48, 8, 60),
    ("c09_list_12_wide", "list", 12, 2.0, 17, 300, 9, 0),
    ("c10_list_3_pad", "list", 3, 2.0, 200, 200, 10, 200),
    ("c11_wide_ratio8", "decode", 30, 3.0, 64, 512, 11, 30),
    ("c12_tiny", "decode", 9, 24.0, 2, 40, 12, 20),
    ("c13_aspect_error", "decode", 9, 24.0, 8, 1700, 13, 0),
    ("c14_12frames", "decode", 48, 4.0, 90, 160, 14, 12),
    ("c15_big_for_cap", "decode", 9, 24.0, 300, 420, 15, 20),
]
BIG = [
    ("b01_480x640_n8", "decode", 40, 8.0, 480, 640, 21, 20),
    ("b02_1080x1920_n4", "decode", 8, 24.0, 1080, 1920, 22, 20),
    ("b03_360x640_n8", "decode", 8, 1.0, 360, 640, 23, 20),
    ("b04_720x1280_n32", "decode", 640, 10.0, 720, 1280, 24, 12),
    ("b05_portrait_1280x720_n6", "list", 6, 2.0, 1280, 720, 25, 12),
    ("b06_odd_481x853_n10", "decode", 100, 10.0, 481, 853, 26, 30),
    ("b07_small_64x64_n4", "decode", 4, 24.0, 64, 64, 27, 40),
    ("b08_tall_1600x300_n12", "list", 12, 2.0, 1600, 300, 28, 12),
]
PROMPTS = [
    {"state": "The street is empty at night.", "question": {"type": "choice", "instructions": "What moves?", "criteria": {"car": "a car", "bird": "a bird"}}},
    {"state": {"camera": 3, "note": "ünïcode"}, "question": {"type": "noul", "instructions": "Is there a person?"}, "images": 1},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--compact", action="store_true",
                    help="drop the resized-frame goldens above 210 kB (what is committed; the tests then rely on the pixel md5)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    processor, R, DC = load()
    cases = []
    spec = BIG if a.big else SMALL
    cap_tokens = 196 if a.big else 16
    for name, mode, total, fps, h, w, seed, noise in spec:
        case = {"name": name, "mode": mode, "total": total, "fps": fps, "h": h, "w": w, "seed": seed, "noise": noise,
                "prompts": PROMPTS}
        cases.append(run_case(processor, R, DC, case, a.outdir, cap_tokens))
        print(name, cases[-1].get("grid"), cases[-1].get("error", ""))
    doc = {"max_video_tokens": cap_tokens, "cases": cases}
    if not a.big:
        doc["tables"] = sampling_tables(R, DC)
        doc["decode"] = decode_goldens(R, a.outdir)
        # request level: tokens over the per-request limit
        vp = processor.video_processor
        vp.max_video_tokens = 196
        one = vp.get_num_of_video_patches(32, 720, 1280) // 4
        doc["limit"] = {"per_video_tokens": one, "frames": 32, "h": 720, "w": 1280,
                        "videos_over": 16384 // one + 1, "videos_ok": 16384 // one}
    else:
        # the shipped example clip, decoded by upstream
        path = os.path.join(ROOT, "assets", "example-hummingbird.mp4")
        v = R._decode_video(open(path, "rb").read(), strict=True)
        fr = np.ascontiguousarray(v.frames)
        open(os.path.join(a.outdir, "hummingbird.frames.u8"), "wb").write(fr.tobytes())
        doc["hummingbird"] = {"n_frames": len(fr), "indices": v.indices, "fps": v.fps, "total": v.total_frames,
                              "w": fr.shape[2], "h": fr.shape[1], "file": path}
    json.dump(mrope_goldens(processor, R, cases), open(os.path.join(a.outdir, "mrope.json"), "w"))
    json.dump(doc, open(os.path.join(a.outdir, "cases.json"), "w"), indent=1)
    if a.compact:
        for n in os.listdir(a.outdir):
            f = os.path.join(a.outdir, n)
            if n.endswith(".resized.u8") and os.path.getsize(f) > 210000:
                os.remove(f)


if __name__ == "__main__":
    main()
