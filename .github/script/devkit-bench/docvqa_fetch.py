#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Build the DocVQA evaluation subset from a pinned public source.

Adapted from FastFlowLM_IRON's .github/scripts/vlm-acc-bench/build_dataset.py.

Runs on a GitHub-hosted runner (devkit-bench.yml's docvqa job), never on an
NPU runner: it downloads one validation shard of the Hugging Face dataset
lmms-lab/DocVQA at a pinned commit (REVISION), checks its SHA-256 and size
against SHARD, samples N questions with a fixed seed (the same sample as
IRON's build_dataset.py, which read only this shard for N <= 200) and writes

  <out>/eval.jsonl        one record per question (question_id, image, prompt, answers)
  <out>/images/<qid>.jpg  the image of each question
  <out>/manifest.json     source, revision, sha256, seed and N

which the workflow hands to the NPU jobs as an artifact of the same run.
Nothing of DocVQA is committed to this repository: its terms beyond the
dataset card's are unresolved, so the bench is off unless enabled (see
devkit-bench.yml, docvqa).

Needs pyarrow and pillow.

Usage:
  python docvqa_fetch.py --n 50 --out data/docvqa [--cache dir]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import urllib.request

DATASET = "lmms-lab/DocVQA"
# The dataset's commit (Hugging Face API "sha"), so the shard cannot change
# under the bench; its hash is checked as well.
REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
SHARD = {
    "path": "DocVQA/validation-00000-of-00006.parquet",
    "sha256": "a31507f0b700ac64f3ead52057c5dc3ccfb0baadcc62bc0d19d17159f080a4d8",
    "size": 114817746,
}
SEED = 3407
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
QID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def shard_url(revision: str = REVISION, path: str = SHARD["path"]) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: str, sha256: str = SHARD["sha256"], size: int = SHARD["size"]) -> str:
    """"" when the file is the pinned one, else why not."""
    try:
        actual_size = os.path.getsize(path)
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    if actual_size != size:
        return f"{path} is {actual_size} bytes, expected {size}"
    digest = sha256_file(path)
    if digest != sha256:
        return f"{path} has SHA-256 {digest}, expected {sha256}"
    return ""


def fetch(cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(SHARD["path"]))
    if os.path.exists(path) and not verify(path):
        print(f"[cache] {path} verified")
        return path
    url = shard_url()
    print(f"[download] {url}", flush=True)
    partial = path + ".partial"
    with urllib.request.urlopen(url, timeout=600) as resp, open(partial, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    why = verify(partial)
    if why:
        os.remove(partial)
        sys.exit(f"[error] the downloaded shard is not the pinned one: {why}")
    os.replace(partial, path)
    return path


def sample(qids: list[str], n: int, seed: int = SEED) -> list[int]:
    """Indices of the N questions to ask, in shard order (IRON's sampling)."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(qids)), min(n, len(qids))))


def record(qid: str, question: str, answers) -> dict:
    answers = answers if isinstance(answers, list) else [answers]
    return {"question_id": qid, "image": f"{qid}.jpg", "prompt": f"{question}{PROMPT_SUFFIX}",
            "answers": [str(a) for a in answers]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50, help="number of questions to sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="", help="where to keep the shard (default <out>/_parquet)")
    args = ap.parse_args(argv)
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError:
        sys.exit("Need pyarrow and pillow:  pip install pyarrow pillow")
    Image.MAX_IMAGE_PIXELS = None

    shard = fetch(args.cache or os.path.join(args.out, "_parquet"))
    table = pq.read_table(shard)
    qid_col = "questionId" if "questionId" in table.column_names else "question_id"
    qids = [str(v) for v in table.column(qid_col).to_pylist()]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    out = []
    for i in sample(qids, args.n):
        qid = qids[i]
        if not QID.match(qid):
            sys.exit(f"[error] unexpected question id {qid!r}")
        img = table.column("image")[i].as_py()
        im = Image.open(io.BytesIO(img["bytes"]))
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(os.path.join(img_dir, f"{qid}.jpg"), quality=95)
        out.append(record(qid, table.column("question")[i].as_py(), table.column("answers")[i].as_py()))
    with open(os.path.join(args.out, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "revision": REVISION, **SHARD, "seed": SEED, "n": len(out)}, f, indent=2)
    if not args.cache:
        shutil.rmtree(os.path.join(args.out, "_parquet"), ignore_errors=True)
    print(f"[done] wrote {len(out)} questions -> {args.out}/eval.jsonl (+ images/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
