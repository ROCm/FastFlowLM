#!/usr/bin/env python3
"""Build the ASR evaluation set in assets/asr_eval/ from FLEURS.

Run this by hand, not in CI: it reads the FLEURS test splits from local
Parquet files and needs pyarrow, numpy and soundfile, none of which the test
itself uses. Download the splits from the dataset's Parquet conversion first:

    https://huggingface.co/api/datasets/google/fleurs/parquet/<config>/test

The selection is deterministic: rows sorted by sentence id, one recording per
sentence, 4 to 25 s long, the first ones taken. So rerunning against the same
Parquet revision reproduces the same set. Sets:

    en-short       20 English clips, each inside one 30 s Whisper window
    en-long        3 files of ~80 s, English clips joined with short gaps, so
                   each crosses two window boundaries
    en-longform    1 file of ~300 s, built the same way
    <lang>-fleurs  10 clips each for de, es, fr, ja, zh and ar

Clips are written as 16 kHz mono MP3, the format the server already decodes
for the bundled test clip, to keep the set a few MB.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SAMPLE_RATE = 16000
GAP_SECONDS = 0.6
MIN_SECONDS, MAX_SECONDS = 4.0, 25.0

# FLEURS config, ISO 639-1 code for the request's language field, and the
# metric: CER for languages written without spaces between words.
LANGUAGES = {
    "en": ("en_us", "wer"),
    "de": ("de_de", "wer"),
    "es": ("es_419", "wer"),
    "fr": ("fr_fr", "wer"),
    "ja": ("ja_jp", "cer"),
    "zh": ("cmn_hans_cn", "cer"),
    "ar": ("ar_eg", "wer"),
}


def load_clips(parquet: Path) -> list[dict]:
    """Rows usable as clips, sorted by sentence id, one per sentence."""
    table = pq.read_table(
        parquet, columns=["id", "num_samples", "transcription", "raw_transcription", "audio"]
    )
    rows = sorted(table.to_pylist(), key=lambda r: (r["id"], r["audio"]["path"]))
    clips, seen = [], set()
    for row in rows:
        seconds = row["num_samples"] / SAMPLE_RATE
        if row["id"] in seen or not MIN_SECONDS <= seconds <= MAX_SECONDS:
            continue
        seen.add(row["id"])
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError(f"{parquet.name} id {row['id']}: {rate} Hz")
        clips.append({
            "id": row["id"],
            "audio": audio,
            "reference": row["transcription"].strip(),
            "raw_reference": row["raw_transcription"].strip(),
        })
    return clips


def take_until(clips: list[dict], seconds: float) -> list[dict]:
    """Pops clips off the front until their joined length reaches `seconds`."""
    taken, total = [], 0.0
    while clips and total < seconds:
        clip = clips.pop(0)
        taken.append(clip)
        total += len(clip["audio"]) / SAMPLE_RATE + GAP_SECONDS
    return taken


def write_item(out: Path, name: str, set_name: str, language: str, metric: str,
               clips: list[dict]) -> dict:
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype="float32")
    parts = []
    for index, clip in enumerate(clips):
        if index:
            parts.append(gap)
        parts.append(clip["audio"])
    audio = np.concatenate(parts)
    path = out / f"{name}.mp3"
    sf.write(path, audio, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III")
    return {
        "file": path.name,
        "set": set_name,
        "language": language,
        "metric": metric,
        "duration": round(len(audio) / SAMPLE_RATE, 3),
        "fleurs_ids": [clip["id"] for clip in clips],
        "reference": " ".join(clip["reference"] for clip in clips),
        "raw_reference": " ".join(clip["raw_reference"] for clip in clips),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet_dir", type=Path, help="Directory of <config>.parquet files")
    parser.add_argument("--revision", required=True,
                        help="FLEURS Parquet revision the files came from, recorded in the manifest")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "assets" / "asr_eval")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    items = []
    for language, (config, metric) in LANGUAGES.items():
        clips = load_clips(args.parquet_dir / f"{config}.parquet")
        print(f"{config}: {len(clips)} usable clips")
        if language == "en":
            for index, clip in enumerate(clips[:20]):
                items.append(write_item(args.out, f"en-short-{index:02d}", "en-short",
                                        language, metric, [clip]))
            rest = clips[20:]
            for index in range(3):
                items.append(write_item(args.out, f"en-long-{index}", "en-long",
                                        language, metric, take_until(rest, 80)))
            items.append(write_item(args.out, "en-longform-0", "en-longform",
                                    language, metric, take_until(rest, 300)))
        else:
            for index, clip in enumerate(clips[:10]):
                items.append(write_item(args.out, f"{language}-fleurs-{index:02d}",
                                        f"{language}-fleurs", language, metric, [clip]))

    manifest = {
        "source": "FLEURS test split (google/fleurs), CC BY 4.0",
        "parquet_revision": args.revision,
        "sample_rate": SAMPLE_RATE,
        "items": items,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(item["duration"] for item in items)
    print(f"{len(items)} files, {total:.0f} s of audio, written to {args.out}")


if __name__ == "__main__":
    main()
