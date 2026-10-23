#!/usr/bin/env python3
"""Pick the MMMU-Pro vision questions that d3_mmmu_test.py asks d3.

MMMU-Pro's vision setting (MMMU/MMMU_Pro on Hugging Face, config "vision",
split "test") puts each question and its lettered options inside one
screenshot; the text carries neither. This downloads that split's Parquet
files, keeps the questions d3 can be asked, sorts them by id and draws --count
of them with random.Random(--seed), so the same revision and seed always give
the same questions. Kept are questions with:

    one PNG, JPEG or WebP image of at most 8 MB (what the server accepts)
    2 to 26 options, and an answer letter among them

Writes <out>/manifest.json and one image per question next to it. The
manifest records the dataset commit the questions came from, so a run can be
reproduced after the dataset moves on.

Needs huggingface_hub and pyarrow, which the test itself does not. The Parquet
files stay in the Hugging Face cache ($HF_HOME), so a runner downloads them
once per dataset revision.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_OPTIONS = 26


def image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def parse_options(value) -> list[str] | None:
    """The options column holds a Python list, as a string in some revisions."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else None


def image_bytes(value) -> bytes | None:
    if isinstance(value, dict):
        value = value.get("bytes")
    return value if isinstance(value, bytes) and value else None


def shard_files(api: HfApi, repo: str, config: str, split: str, revision: str) -> list[str]:
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                   if f.endswith(".parquet") and f.split("/")[0] == config
                   and Path(f).name.startswith(split))
    if not files:
        raise SystemExit(f"No {config}/{split}*.parquet files in {repo} at {revision}.")
    return files


def scan(path: str) -> tuple[list[dict], dict[str, int]]:
    """Every question in one shard d3 can be asked, without its image bytes."""
    columns = [c for c in ("id", "options", "answer", "subject", "image")
               if c in pq.ParquetFile(path).schema_arrow.names]
    usable, excluded = [], {}
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=32):
        for row in batch.to_pylist():
            options = parse_options(row.get("options"))
            data = image_bytes(row.get("image"))
            answer = str(row.get("answer") or "").strip()
            reason = None
            if not options or not 2 <= len(options) <= MAX_OPTIONS:
                reason = "options"
            elif answer not in [chr(65 + i) for i in range(len(options))]:
                reason = "answer"
            elif data is None or image_mime(data) is None:
                reason = "image format"
            elif len(data) > MAX_IMAGE_BYTES:
                reason = "image size"
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            usable.append({"id": str(row["id"]), "subject": row.get("subject") or "",
                           "options": len(options), "answer": answer, "shard": path})
    return usable, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory to write the questions to.")
    parser.add_argument("--repo", default="MMMU/MMMU_Pro")
    parser.add_argument("--config", default="vision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default="main",
                        help="Dataset branch, tag or commit (default: main).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()

    api = HfApi()
    commit = api.dataset_info(args.repo, revision=args.revision).sha
    print(f"{args.repo} at {args.revision} is commit {commit}")

    usable, excluded = [], {}
    for name in shard_files(api, args.repo, args.config, args.split, commit):
        path = hf_hub_download(args.repo, name, repo_type="dataset", revision=commit)
        rows, skipped = scan(path)
        usable += rows
        for reason, count in skipped.items():
            excluded[reason] = excluded.get(reason, 0) + count
        print(f"  {name}: {len(rows)} usable, {sum(skipped.values())} excluded")

    usable.sort(key=lambda r: r["id"])
    if len({r["id"] for r in usable}) != len(usable):
        raise SystemExit("Question ids are not unique; the draw would not be reproducible.")
    if len(usable) < args.count:
        raise SystemExit(f"Only {len(usable)} usable questions, {args.count} wanted.")
    chosen = sorted(random.Random(args.seed).sample(usable, args.count), key=lambda r: r["id"])
    print(f"{len(usable)} usable questions (excluded: {excluded or 'none'}); "
          f"drew {args.count} with seed {args.seed}")

    # Second pass for the image bytes, so only the chosen ones are held in memory.
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {r["id"]: (index, r) for index, r in enumerate(chosen)}
    for shard in sorted({r["shard"] for r in chosen}):
        for batch in pq.ParquetFile(shard).iter_batches(columns=["id", "image"], batch_size=32):
            for row in batch.to_pylist():
                index, item = wanted.get(str(row["id"]), (None, None))
                if item is None or "file" in item:
                    continue
                data = image_bytes(row["image"])
                item["mime"] = image_mime(data)
                item["file"] = f"q{index:02d}.{item['mime'].split('/')[1]}"
                (args.out / item["file"]).write_bytes(data)
    missing = [r["id"] for r in chosen if "file" not in r]
    if missing:
        raise SystemExit(f"Images not found on the second pass: {missing}")

    manifest = {
        "source": f"{args.repo} ({args.config}, {args.split} split)",
        "revision": commit,
        "seed": args.seed,
        "count": args.count,
        "usable": len(usable),
        "excluded": excluded,
        "items": [{k: v for k, v in r.items() if k != "shard"} for r in chosen],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} questions to {args.out}")


if __name__ == "__main__":
    main()
