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
