# hunyuan_npu test harness

Three programs, built by `make` (each has its own `main()`):

| program | what it measures |
|---|---|
| `test_hunyuan` | the 130-line Chinese→English translation benchmark (short prompts) |
| `bench_prefill` | prefill time for prompts of 12 … 1004 tokens, through the chat template |
| `test_long` | long context: prefill + decode time per prompt length, and logits dumps |

## test_long

Added 2026-11-03 for the long-context work (engine change: `FLM_DLL/detail/hunyuan_npu`,
"Long context" in its README).

```
python3 make_long_text.py /scratch/michyu/datasets/locomo-mc10/raw/locomo10.json long.txt
test_long --text long.txt --lengths 512,1024,2048,4096,8192,16384,32768 --decode 16 --dump out/
```

- The prompt is the first N tokens of the text file, fed to the engine as raw token ids
  (`causal_lm::prefill`), so N is exact and there is no chat template. `make_long_text.py`
  flattens the 10 LoCoMo conversations (natural English dialogue, ~188K tokens) into one file.
  The file is not committed (LoCoMo is CC BY-NC).
- After the prompt it runs `--decode` steps with **teacher forcing**: step i feeds the text's
  token N+i, not the model's own pick. Then two builds see exactly the same tokens and their
  logits can be compared step by step, even if they would pick different words.
- `--context 0` (default) loads the engine with longest prompt + decode steps.
- The catalog marks the model `single-turn`, which makes FastFlowLM force the 1K default
  context; `test_long` drops that label in memory only. `model_list.json` is unchanged.
- `HunyuanProbe` is a subclass of `Hunyuan` whose only job is to reach the protected
  `lm_engine` and `tokenizer`.
- Columns: prefill ms and tok/s for the whole prompt; decode ms for all steps and tok/s;
  `top1` = the token id the prefill logits pick.
- `--dump dir` writes `L<N>_prefill.bin` and `L<N>_decode<i>.bin` (raw bf16 logits).
- `--hash` (WP12) adds a last column: an FNV-1a hash of the prefill logits bytes. The engine is
  deterministic, so the same length must give the same hash in every prompt and every run; a
  different hash means that prompt's logits changed (used to check output after a run failure).

## test_hunyuan --order (WP12)

`--order forward` (default, the corpus order), `reverse`, or `shuffle:<seed>`. Every turn rewinds
to the pinned system prompt, so a turn's text should not depend on the order. It did (14/130 lines
in reverse order) until the engine zeroed the cache rows left past the pin (`HY_ZERO_STALE_KV=1`,
engine README "Stale KV rows after restore"). The turns are printed in the order they ran; pair
them by source text to compare two orders (`hymt2_plan/logs/WP12/g2cmp.py`).

## compare_logits.py

`python3 compare_logits.py <ref_dir> <test_dir>` prints, per prompt length, the cosine of
the prefill logits, whether the top-1 token matches, the top-5 overlap, and over the decode
steps the lowest / mean cosine and how many top-1 tokens match.

The reference is the same engine built with attention on the host in fp32:

```
cd FLM_DLL/detail/hunyuan_npu
make HOST_ATTN=1 BUILD_DIR=../../build/hunyuan_npu_ref
```

Everything else (embedding, matmuls on the NPU, lm_head) is the same in both builds, so the
difference is the NPU attention alone. One known offset: the reference decodes through the
batched prefill path at L = 1, the NPU build through the decode overlay, whose matmuls round
differently. So decode cosines sit near 0.995 even at 512 tokens; what matters is whether they
fall as the prompt grows.

The accuracy gate of the Hy-MT2 plan (G4) uses the all-CPU reference instead
(`make HOST_ATTN=1 HOST_MM=1 BUILD_DIR=../../build/hunyuan_npu_host`): on Krackan the
`HOST_ATTN=1`-only build hangs from ~4K. Dumps for 512 ... 24K are in
`/scratch/michyu/hy_mt2_stage/long/host_acc`.

## prof_summary.py

Added 2026-11-05 (WP0). Sums the `[prefill-prof]` lines of a `test_long` run made with the
engine's profile build:

```
cd FLM_DLL/detail/hunyuan_npu && make PREFILL_PROFILE=1 BUILD_DIR=../../build/hunyuan_npu_prof
LD_LIBRARY_PATH=.../build/hunyuan_npu_prof/lib test_long --text long.txt --lengths 12,12,12,12,512,512,512,512 --decode 0 > prof.log
python3 prof_summary.py prof.log [--skip 1] [--steps] [--md]
```

- Each result row of `test_long` closes one prompt; the profile lines before it belong to it.
  Every step is summed over the 32 layers (and over all 512-token pieces of a long prompt).
- Repeat each length (as above) and keep `--skip 1`: the first call at a new padded size also
  builds its run sequences and buffers. The table shows the median of the rest.
- Rows group the steps: NPU GEMM runs, NPU attention, each kind of CPU work, syncs, lm_head.
  `not covered by ticks` = the measured prefill time minus all ticks. `--steps` prints every
  single step too.
- An NPU run's time includes the overlay switch in front of it (`attn:mha` from GEMM to
  attention, `attn:o_run` back) and, in layer 0, the reload after the decode xclbin.
- The profile prints ~30 lines per layer, which costs ~5% at short prompts: use it for the
  split, `bench_prefill` / `test_long` without it for the totals.
- Since WP1 the syncs are ranged (group "syncs (ranged)") and K's rope runs inside
  `attn:fill_kv_cache`. Logs from before WP1 still have `attn:k_rope_norm` and
  `attn:kv_sync_to`; they show up under "other ticks".
- Since WP2 the matmul inputs go to the NPU as bfp16 tiles: the two norms and `mlp:glu_mul`
  include packing + flushing them (no `layer:x_sync_to` / `mlp:hid_sync_to` in a normal NPU
  build any more), and `attn:o_x_tiles` (own group) reads the attention output back and packs
  it for the o run.
