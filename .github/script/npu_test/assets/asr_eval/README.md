# ASR evaluation set

The audio and reference transcripts used by `.github/script/asr_test.py`.

## Source and license

Every clip and transcript here comes from the test split of **FLEURS**
(`google/fleurs` on Hugging Face), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The clips were
re-encoded to 16 kHz mono MP3, and the English ones in `en-long` and
`en-longform` were joined end to end with 0.6 s of silence between them. The
transcripts are unchanged.

> Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S.,
> Riesa, J., Rivera, C., and Bapna, A. *FLEURS: Few-shot Learning Evaluation
> of Universal Representations of Speech.* IEEE SLT 2022.
> https://arxiv.org/abs/2205.12446

`manifest.json` records the FLEURS Parquet revision and, for every file, the
FLEURS sentence ids it was built from.

## Contents

| Set | Files | Audio | Metric | What it exercises |
|---|---|---|---|---|
| `en-short` | 20 | 194 s | WER | single 30 s window |
| `en-long` | 3 | 252 s | WER | ~83 s each, two window boundaries |
| `en-longform` | 1 | 308 s | WER | ten window boundaries |
| `de-fleurs`, `es-fleurs`, `fr-fleurs`, `ar-fleurs` | 10 each | 105-139 s each | WER | other languages |
| `ja-fleurs`, `zh-fleurs` | 10 each | 104-142 s each | CER | languages written without spaces |

`expected.json` holds the error rates main reached, per platform and set, for
runs that have no baseline build to compare against. Platforms are kept apart
because their builds decode differently: Windows and Linux differ by up to
5 points on some sets with identical sources.

## Rebuilding

`.github/script/npu_test/tools/build_asr_eval.py` regenerates this directory from the FLEURS
Parquet files; its docstring says where to get them. The selection is
deterministic, so the same Parquet revision gives the same set.
