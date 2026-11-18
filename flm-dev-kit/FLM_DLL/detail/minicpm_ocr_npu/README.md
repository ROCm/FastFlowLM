<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# minicpm_ocr_npu — MiniCPM-OCR-1.0

Engine for `openbmb/MiniCPM-OCR-1.0` (revision `165531f9ead5c7f09fba35a9a89fd2e255acf99a`):
a 1.3 B VLM, 24-layer hybrid text tower plus a 27-block SigLIP vision tower.

**Forked from `detail/minicpm_v_4_7_npu/` on 2026-10-28.** Read that engine's docs for anything
not mentioned here — `image/README.md` in particular is unchanged and still accurate.

## What this model is

Upstream, `MiniCPMOcrForConditionalGeneration` is `MiniCPMV4_6ForConditionalGeneration` with
`pass`. MiniCPM-V-4.7 is *also* 4.6, plus ~600 lines of video grouping and 3D-RoPE. So
MiniCPM-OCR is not a 4.7 variant; it is the **4.6 subset that 4.7 extends**, and every difference
below is therefore a removal.

The text tower is config-identical to Qwen3.5-0.8B, so the **Qwen3.5-0.8B xclbins are reused
unchanged**. `layer_types` is 3 × `linear_attention` (GatedDeltaNet) then 1 × `full_attention`,
repeating six times.

## The four deltas from minicpm_v_4_7_npu

| Delta | Where it is handled | Engine code change |
|---|---|---|
| `downsample_mode: "4x"` | `image/vision_model_def.hpp` already reads it | **none** — `insert_layer_id` becomes −1, `use_vit_merger()` false |
| `mtp_num_hidden_layers: 0` | nowhere | **none** — no engine source reads `mtp_*` |
| `vocab_size: 248100` | `vocab_size_padded` | **none** — pads to the same 248832 as 4.7's 248094 |
| `tie_word_embeddings: true` | the **converter** (`q4nx/models/minicpm_ocr.py`) | **none** — `model.q4nx` ships a real `lm_head.weight` |
| **no mRoPE** | `minicpm_ocr_npu.cpp::_prefill_with_mm()` | **the only one** — positions are sequential for image prompts too |

That last one is ~5 lines, and the branch it needs already existed as 4.7's text-only path.
`image/mrope.hpp` keeps `canvas_token_ids` (for `image_pad`) but `canvas_mrope()` is dead on every
path this model takes; the header says so.

**Do not restore the mRoPE branch for image prompts.** The language model was trained on whatever
position rule its own checkpoint uses; 4.7's would be wrong in a way that reads as a slightly worse
model rather than as an error.

## The cost of 4x

Skipping the vit_merger is not a speed win. Blocks 7..26 run on **4× the patches**, and
`token_divisor()` drops from 16 to 4, so one image costs **256 LLM rows (a 225×225 icon) to 2560
(anything from a 1280×1280 page to a 12 MP photo)** — the ceiling comes from `max_slice_nums: 9`,
not from pixel count. One image fits inside `max_prefill_len: 4096`; two in a turn do not.

The projector is unaffected: its input is still `D * merge² = 4608`.

## xclbins it registers

From `xclbins/<model_name>/`, by filename only — the engine fails at load if one is missing.

| File | Stage | Source |
|---|---|---|
| `layer.xclbin`, `lm_head.xclbin` | decode | Qwen3.5-0.8B's, unchanged |
| `layer_with_lm_head.xclbin` (optional) | decode + every lm_head run, replaces the two above when present | `FLM_Xclbin/minicpm_ocr/decoding` (`MODEL_TYPE=QWEN3_5_08B`); see "lm_head inside layer.xclbin" |
| `layer_with_lm_head_kvb.xclbin` (optional) | the same, attention reads K / V as bfp16; preferred over `layer_with_lm_head.xclbin` when present | `FLM_Xclbin/minicpm_ocr/decoding`, `make bitstream KV_BFP16=1`; see "Decode attention on bfp16 K and V" |
| `mm.xclbin`, `attn.xclbin` | prefill | Qwen3.5-0.8B's, unchanged |
| `attn_bfp16.xclbin` (optional) | prefill attention, replaces `attn.xclbin` when present | `FLM_Xclbin/minicpm_ocr/gated_attention_prefill` (the Qwen3.5-2B flash kernel); see "Prefill attention on bfp16 tiles" |
| `mm_bfp16_glu.xclbin` or `mm_bfp16.xclbin` (optional) | every prefill mm, replaces `mm.xclbin` when present (`_glu` first) | `FLM_Xclbin/minicpm_ocr/dequant_mm_prefill` (the Qwen3.5-2B flash mm); one build serves both names, see "Prefill mm on bfp16 tiles" |
| `conv8.xclbin` | prefill, only when built with `PREFILL_CONV=npu` | `FLM_Xclbin/minicpm_ocr/conv1d_prefill` (all 8 columns, one q/k head + 64 v channels per core; laid out for D = 6144). Its runtime sequence is `gen_seq_conv1d()`; the old `conv.xclbin` does not work with it, hence the new name |
| `vision_attn.xclbin`, `vision_mm_a.xclbin`, `vision_mm_b.xclbin` | vision tower | Qwen3.8-27B's, unchanged |

No new `MODEL_TYPE` was needed for this port. `conv8.xclbin` is the one new kernel; it is not in
`kernel_assignment.json` yet, so a packaged model folder has to get it by hand until it is.

## Config

`model_config.template.json` is the contract; `FLM_Q4NX_Converter/package_minicpm_ocr.py` copies it
to the packaged `config.json` with the `_comment*` keys stripped. The engine reads a **flattened**
config — `LM_Config::_normalize_multi_modal()` lifts only `vocab_size` out of `text_config`, so
every text dimension must be hoisted to the top level. Upstream nests them; do not copy that shape.

`MINICPM_OCR_INSERT_LAYER_ID` and `MINICPM_OCR_VIT_MERGER_INTERMEDIATE_SIZE` are deliberately
**absent** rather than set: at 4x that tower does not exist, and `vision_weight.q4nx` carries none
of its 12 tensors.

## Build

```shell
bash ../../build_minicpm_ocr.sh                 # engine + its shared deps
make -j8                                        # this engine only
make PREFILL_MM=cpu PREFILL_ATTN=cpu -j8        # host-only: no xclbin, no NPU
```

Output is `../../build/lib/libminicpm_ocr_npu.so`. Keep `-Wl,-Bsymbolic-functions`: the app links
every engine at once, and without it this `.so`'s internal calls bind to whichever engine comes
first in `DT_NEEDED`. The fork renamed every model-specific symbol (`minicpm_v_4_7_npu` →
`minicpm_ocr_npu`, `mcv` → `mco`, `minicpm_v_cpu` → `minicpm_ocr_cpu`, `MINICPM_V_*` →
`MINICPM_OCR_*`, including the `FLM_MINICPM_OCR_*` env vars), so `nm -DC` shows **zero**
`minicpm_v` symbols — the two engines can be loaded together.

Windows (`CMakeLists.txt`, or `make` under WSL) builds `minicpm_ocr_npu.dll` + `.lib` and copies
both to `../../build/lib`. It builds only the shipped case, the Makefile's defaults: everything on the
array except conv1d, so it sets `FLM_PREFILL_CONV_ON_CPU` and needs no `conv8.xclbin`. The other
Makefile switches are for debugging on Linux and have no CMake equivalent.

## Where each stage runs, and why GDN/conv stay on the host

`PREFILL_GDN` and `PREFILL_CONV` default to `cpu`. The Makefile explains the choice and quotes
MiniCPM-V-4.7's numbers; this model was measured separately, because its prefill carries 4x the
vision tokens and the mix could have differed. It does not.

Measured 2026-10-28, idle box, interleaved A/B with the `.so` physically swapped each time -- the
harness has an RPATH, so `LD_LIBRARY_PATH` would have silently run one build twice. Time in the 18
GatedDeltaNet layers (`FLM_MINICPM_OCR_PREFILL_PROFILE=1`), driven by `--prompt-file`:

| prompt tokens | `GDN/CONV=cpu` (default) | `GDN/CONV=npu` | ratio |
|---|---|---|---|
| 1,011 | 282 ms | 513 ms | 1.82x |
| 2,009 | 563 ms | 827 ms | 1.47x |
| 8,015 | 2,288 ms | 2,695 ms | 1.18x |
| 16,023 | 4,642 ms | 5,200 ms | 1.12x |
| 24,031 | 6,827 ms | 7,497 ms | 1.10x |

**The array's penalty is mostly fixed overhead, and it does amortise** -- 82% at 1k falls to 10% at
24k. It is worth knowing that a single short-prompt measurement badly overstates the gap. But the
curves do not cross: fitting 8k -> 24k gives 0.283 ms/token on the host against 0.300 on the array,
so the array stays ~6% more expensive *per token* and the ratio flattens near 1.06-1.09x rather
than dipping below 1. Both stages are token-serial on the array, so they pay a bitstream switch and
a full activation round trip for arithmetic the host does with AVX-512.

The 6 attention layers are unchanged either way (126 ms vs 130 ms at 1287 tokens), as they must be
-- they contain neither stage.

Generated text differs trivially between the two builds: the three text turns are byte-identical
and the image caption diverges in one phrase ("painted in hues" vs "painted with the hues") -- one
early token flip from different arithmetic, not a correctness failure. Each build is deterministic
across runs.

### Update 2026-10-29: the rewritten GDN kernel changes this above ~1.5k tokens

`FLM_Xclbin/minicpm_ocr/gate_delta_net_prefill/` holds a rewrite of the GDN kernel (kernel `.cc`
only — same signature, IRON, DMA and host sequence, so this engine takes it with no change; build
with `PREFILL_GDN=npu`, conv stays on the host). Tested here against the shipping host-GDN build
and against the old kernel, medians of 3, interleaved:

| tokens | host GDN | new kernel | old kernel |
|---|---|---|---|
| 12 | 168 ms | 280 ms (+67%) | 275 ms |
| 1,020 | 411 ms | 446 ms (+9%) | 551 ms |
| 4,092 | 1,504 ms | **1,256 ms (-16%)** | 1,719 ms |
| 8,015 | 3,079 ms | **2,474 ms (-20%)** | 3,358 ms |
| 24,031 | 10,913 ms | **8,805 ms (-19%)** | 11,292 ms |

(whole language prefill.) Fitting the 18 GDN layers: host **0.285 ms/token + 16 ms**, new kernel
**0.192 ms/token + 155 ms**, old kernel 0.293 + 210. The new kernel is a third cheaper per token
than the host, so it wins past a **crossover near 1.5k tokens** and loses below on a fixed ~140 ms.
That fixed cost is the same for the old and new kernels, so it is per-layer overhead (switching
and DMA setup), not the kernel's maths — it is the thing to attack next for short prompts.

Accuracy improves too: `FLM_XREF_LAYERS=all`, all 18 layers, `gdn_out` rel L2 4.5e-3 -> 2.1e-3,
`gdn_state` 1.9e-5 -> 3.2e-6, better on every layer including a 125-token partial chunk. End to
end it gives the same text as the host-GDN build on 13/15 frozen prompts (old kernel 12/15), and
both misses are ties in the q-ref (top-2 margins 0.0 and 0.5).

For this model the crossover sits right where image prompts land: one picture is 256-2560
tokens, a typical page ~1.3k. So the choice is genuinely workload-dependent.

### Length dispatch: short prefills on the host, long ones on the array

Built with `PREFILL_GDN=npu`, the engine now picks the GDN backend **per prefill call**: calls of
at least `gdn_npu_min_tokens()` = **1536** tokens (5120 since 2026-11-06, see "GDN and attention placement
re-tuned") run on the array, shorter ones decline and run
the host kernel. `FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS` overrides it without a rebuild (`0` = always
the array, a huge value = always the host), which also makes a backend A/B possible on one `.so`.

"Per call" matters because MiniCPM_OCR never splits a prompt -- a 24k prompt is one call. In
practice the split is therefore between turns: a long first prompt runs on the array, and a cached
follow-up prefills only its new tokens, usually on the host, starting from the state the array
left. Each path leaves the recurrent state current on both host and device, so any order is safe.

Verified 2026-10-29 (`models_simple/prompts/minicpm_ocr/dispatch/`):

* **Routing** -- on one `.so`, default output is identical to forced-host below 1536 and to
  forced-array at or above it, checked at 12, 1024, **1535, 1536**, 2009, 4092 and 4200 prefill
  tokens. Below the threshold the fixed cost is really gone (12 tokens: 84 ms of GDN layers, not 190).
* **Handoff** -- a fact stated in one turn and asked for in the next is recalled in all three
  settings, in both directions: array -> host (3,142-token head on the array, 21-token tail on the
  host) and host -> array (988-token head on the host, 2,043-token tail on the array). The
  follow-up split happens only when the cached head is at least `min_boundary` = 512 tokens.
* **Speed** -- medians of 3, the default lands within +/-3% (noise) of the faster backend at every
  length from 12 to 24k:

| tokens | all host | all array | default |
|---|---|---|---|
| 12 | 166 ms | 283 ms | 164 ms |
| 1,024 | 417 | 452 | 418 |
| 1,535 / 1,536 | 582 / 569 | 583 / 574 | 565 / 581 |
| 2,009 | 735 | 702 | 703 |
| 4,092 | 1,506 | 1,257 | 1,269 |
| 24,031 | 11,018 | 8,868 | 8,912 |

**This needs the rewritten kernel staged as `GateDeltaNet_prefill.xclbin`.** With the older Qwen3_5
kernel the array is slower than the host at every length, so `PREFILL_GDN=npu` would make long
prompts slower. That is why the committed Makefile default stays `PREFILL_GDN=cpu` until the
kernel source is committed; the default flips in the same change that adds the new xclbin.

### Conv1d stays on the host (decided 2026-10-29)

The rewritten conv kernel (`conv8.xclbin`, all 8 columns) is correct in the engine -- per layer
cos >= 0.99997 and ~0.1% relative L2 on all 18 layers, every row checked including partial tail
blocks -- and about 2x faster than the host per call (4.93 vs ~10.6 ms a layer at 8k). It still
loses end to end below ~9k tokens, because running it on the array adds one more context switch
to every layer (mm -> conv -> GDN instead of mm -> GDN, ~2.8 ms), on top of its launch.

The four conv/GDN combinations, whole language prefill, medians of 3, on one `.so` driven by
`FLM_MINICPM_OCR_CONV_NPU_MIN_TOKENS` / `FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS`:

| tokens | conv host, GDN host | conv host, GDN array | conv array, GDN host | conv array, GDN array |
|---|---|---|---|---|
| 12 | **163** | 281 | 269 | 339 |
| 1,024 | **388** | 448 | 493 | 501 |
| 2,009 | 712 | **700** | 819 | 746 |
| 8,015 | 3,064 | **2,485** | 3,099 | 2,501 |
| 16,023 | 6,676 | 5,439 | 6,639 | **5,354** (-1.6%) |
| 24,031 | 10,693 | 8,855 | 10,704 | **8,645** (-2.4%) |

So conv on the array only wins above ~9k tokens, by 1.6-2.4%. It stays on the host
(`PREFILL_CONV=cpu`, the default) because the gain is small and lands where OCR prompts rarely go,
and because it would get worse with a matmul + GDN prefill overlay: conv would then cost two
switches a layer, and at 7 KB of program memory it cannot join that overlay (16 KB limit). The
length switch exists for experiments -- `make PREFILL_CONV=npu`, then the env var above (default
12288) -- and needs `conv8.xclbin` from `FLM_Xclbin/minicpm_ocr/conv1d_prefill` in the model folder.

**conv_qkv is now made valid only where it is consumed.** Its one reader is the gated delta rule,
so the conv paths copy it only to the side the GDN will run on (`_conv_qkv_needed_on_host()`),
instead of both ways every layer -- up to ~98 MB a layer at 8k tokens. Output is bit-identical
to before in all four combinations (20/20 prompts) and across the host <-> array handoff tests;
the host-only path gets ~3% faster at 1k-4k tokens (1,496 -> 1,447 ms at 4k).

### Host-side prefill speed-ups (2026-10-29)

No kernel changed. Prefill, medians of 5-7 interleaved processes, each step against the one
before it:

| step | change | 128 tok | 512 | 2,048 | 8,192 | output |
|---|---|---|---|---|---|---|
| 1+2 | host helpers on every core; needless syncs dropped | -8% | -7% | -7% | -6% | bit-identical |
| 3 | host work under the in_proj array runs | -6% | -5% | -4% | -5% | bit-identical |
| 5 | attention on the host when that is cheaper | -25% | -10% | 0 | 0 | see below |

1. **Threads.** The `cpu_functions.hpp` helpers -- norms, residual adds, the alpha/beta matmuls,
   gated norm, GLU -- were pinned to 4 threads, and `_elementwise_mul` / `_sigmoid` were
   single-threaded, although all of them run while the array waits. They now use
   `prefill_threads()` (one per physical core). Rechecked with two busy-loop processes running
   alongside: no worse than before; an 8-thread cap was not better anywhere.
2. **Syncs.** `qkv` is pushed only when the conv runs on the array; linear o_proj no longer pushes
   `x` (the run only writes it); the MLP no longer pulls `hid` (the array never writes it);
   `fill_kv_cache` flushes only the rows it wrote instead of a pull + push of the whole 64 MB
   cache (~1.1 ms per attention layer at any length). Host attention pulls only the rows it reads.
3. **Overlap.** `_in_proj_overlapped_npu()` starts the qkv and g projections asynchronously and
   does alpha/beta under the qkv run, the conv-state carry + host conv under the g run.
5. **Attention placement**, per prefill call. `_attn_on_npu()` estimates one layer both ways: the
   host costs ~33 ns per query-key pair it really scores; the array costs two context switches
   (~6.5 ms) plus ~0.93 ns per pair of its *padded* rows against the whole cache.
   `FLM_MINICPM_OCR_ATTN_BACKEND=npu|host` forces a side (default auto). Below 1536 tokens the GDN
   is on the host too, so a short prompt now never leaves `mm.xclbin`.

| fresh prompt, tokens | 124 | 361 | 508 | 620 | 758 | 888 | 1,019 | 2,044 |
|---|---|---|---|---|---|---|---|---|
| attention on the array, ms | 160 | 193 | 213 | 306 | 332 | 356 | 374 | 630 |
| attention on the host, ms | 124 | 164 | 190 | 284 | 324 | 368 | 395 | 928 |
| auto, ms | **123** | **162** | **188** | **284** | **322** | 359 | **368** | **626** |

Short cached follow-ups (`--follow-up ... --resume`, 34 new tokens) stay on the host even on a
long conversation, because the array would still run 512 padded rows against all of it: 235 vs
317 ms on 2k cached, 318 vs 437 ms on 8k. A single "length >= N" threshold fitted on fresh
prompts would have sent the 8k case to the array.

Checks for step 5: auto's output equals forced-host where it picks the host and forced-array
(= the previous build) where it picks the array; the invariance run is unchanged (the same one
`mm.xclbin` miss at 1024); `--multi-slot` passes. Against the bf16 / Q-ref prompts the score is
unchanged (3/6 exact), but `multiturn_recall` now leaves the Q-ref at token 8 (Q-ref margin
0.75) instead of 16 (margin 0.000): host attention computes in fp32 where the array kernel is
mostly bf16, so it is a different rounding, not a fault -- but it is not a tie either.

**Not done:** keeping scratch buffers across calls (grow-only). It would save ~15 ms once per
conversation (the first short follow-up after a long turn reallocates), and every later small call
would then sync the large buffers unless all ~25 syncs became row-ranged -- more cost than gain as
a simple change. **Seen, not fixed:** a resumed turn spends ~25-90 ms in the slot restore/save
(`write_state` zeroes and syncs all six 64 MB attention caches); row-ranged syncs there would
likely remove most of it. The GDN threshold was re-swept after these changes and stays at 1536
(within +-1% of the faster backend from 1,019 to 2,044 tokens).

## lm_head inside layer.xclbin (2026-11-05)

Before, every decoded token switched xclbins twice: `layer.xclbin` -> `lm_head.xclbin` -> `layer.xclbin`, about
2.9 + 3.2 ms per token. `layer_with_lm_head.xclbin` (built in `FLM_Xclbin/minicpm_ocr/decoding`, a copy of the
Qwen3.5-2B design from `origin/main`) runs the lm_head on the layer design's projection cores, so a token never
switches.

How it works:

* The projection cores and the rms core read a **mode** at the start of every run (0 GDN layer, 1 attention layer,
  2/3 lm_head). The host writes it into each core's RTP buffer and releases the core's lock 6
  (`minicpm_ocr_npu_sequence::_set_modes`, called at the start of every layer sequence when `write_modes` is on).
  Addresses: projection cores 33280, rms core **34816** (36864 on the 2B; the build checks it).
* The lm_head is a separate run on the same xclbin (`gen_lm_head_in_layer_seq`): x, the Q8_0 lm_head re-laid out as
  32 x 128 half-chunks padded to 4,736 B (294 MB, `lm_head_in_layer.hpp`), `model.norm` in its own buffer, logits
  out of shim 1. It stays a separate run after the layer run list, so `set_context_length()` (the next token's run
  list) still runs while the lm_head runs.
* The same run gives the prefill's final logits.
* Choice at load time: `layer_with_lm_head.xclbin` present in the xclbin folder -> new path;
  absent, or `FLM_MINICPM_OCR_LM_HEAD_IN_LAYER=0` -> the old `layer.xclbin` + `lm_head.xclbin`, unchanged
  (the old `layer.xclbin` has no mode RTPs, so the modes are only written on the new path).

Correctness:

* lm_head alone, same final x: **bit-identical** to `lm_head.xclbin` (kernel-folder test). Prefill last-row logits of
  10 cases: bit-identical.
* The decode layers are a **different build** (Peano CI toolchain, the 2B folder's sources), so decode is not
  bit-identical: same-history decode logits vs the old path cos >= 0.9993 (median 0.99998). 6 of 12 golden replies
  change, each at a near tie (top-2 within one bf16 step, one exact tie). `--check` vs the fp32 CPU reference is as
  close as the shipped `layer.xclbin`. Image answers, `--resume`, clear and `--multi-slot` unchanged.

Decode speed, 128 generated tokens, median of 5, A/B alternated (shipped -> new):

| prompt tokens | decode tok/s | ms/token: layers + lm_head (profile build) |
|---|---|---|
| 124 | 47.7 -> **66.5** | 12.86 + 8.14 -> 9.72 + 5.29 |
| 1,019 | 46.3 -> **64.4** | 13.47 + 8.10 -> 10.23 + 5.29 |
| 8,188 | 38.0 -> **52.4** | 18.14 + 8.12 -> 13.75 + 5.31 |
| 24,041 | 26.7 -> **36.3** | 29.11 + 8.36 -> 21.94 + 5.48 |

The lm_head now reads 294 MB at ~55 GB/s on 8 shim channels (the separate `lm_head.xclbin` used 16, 4.9 ms alone);
the whole token reads at ~37 GB/s at short context (was ~27). Prefill is unchanged.

## Prefill attention on bfp16 tiles (2026-11-05)

`attn_bfp16.xclbin` is the Qwen3.5-2B flash prefill attention (`FLM_Xclbin/minicpm_ocr/gated_attention_prefill`, a
copy of `origin/main:FLM_Xclbin/Qwen3_5_2B/gated_attention_prefill`; same head shapes). It reads q, K and V^T as
bfp16 tiles made on the host and gives each core 32 query rows instead of 16.

How the engine drives it (`attn_bfp16.hpp`, the attention block in `minicpm_ocr_prefill.hpp`):

* **q tiles** (`q_tiles`): made right after the q RoPE, while the array runs the g projection, but only for calls
  that will run on the array.
* **A bfp16 copy of each attention layer's KV cache** (`kv_store`): K tiles, then V^T tiles, `rows` rows per kv head
  (grows by 1.5x in 4096-row steps, not past the bf16 cache). Before each array call, rows from
  `min(valid, L_begin)` up to the last 128-key block the kernel reads are converted again. Rows the host did not just
  write are **pulled from the device first**: the rows decode added since the last array call, and the rows past
  the prompt that the kernel also reads. (The 2B engine did not need this because its `fill_kv_cache` pulls the
  whole cache; ours does not, see "Host-side prefill speed-ups".)
* **`kv_cache_changed(from_row)`** lowers `valid` whenever the host rewrites the bf16 cache: `clear_context` (0),
  `restore` (the checkpoint length), `write_state` / `write_state_kv` (`kv_from`), `update_max_length` (the old
  size). So after a slot restore, the next array call converts the whole copy again (~3.8 ms per layer at 8K).
* **Choice at load time**: `attn_bfp16.xclbin` in the xclbin folder -> new path; absent, or
  `FLM_MINICPM_OCR_ATTN_BFP16=0` -> `attn.xclbin`, unchanged. Host / array placement per call
  (`FLM_MINICPM_OCR_ATTN_BACKEND`, `_attn_on_npu`) works on both; the new path has its own array cost.

**Not bit-identical to `attn.xclbin`, by design of the 2B kernel.** Its softmax differs in four places (fp32 shift
`s * log2(e) - m * log2(e)`, `c` exactly 1 where the max stayed, a lazy running max, `y` rescaled only when a max
moved). A test build that undoes only those four (`ATTN_SHARED_SOFTMAX` in the kernel folder) is bit-identical to
`attn.xclbin`: standalone on random and on real q / K / V, with and without history, and in this engine (golden
text 12/12 incl. `--resume`, clear and images; the 10 prefill logits bit-identical; `--multi-slot` passes). So the
bfp16 tiles, the 32 rows per core and the KV copy add no difference. (Bit-identical needs the same host / array
placement too: a call the two cost models place differently runs on the host in fp32 on one side.)

The 2B kernel itself vs `attn.xclbin`: attention output cos
0.99996-0.99999 on real data (rel L2 0.4-0.9%), equally close to an fp32 reference; golden text 11/12 identical,
the 12th (`turns4_german_ocr`) changes its 128th token at an exact tie after the sampler's penalties; prefill
logits: 6/10 bit-identical, the rest cos >= 0.99977 with the same top-1 except one exact top-2 tie.

Array cost, one attention layer (switches + conversions + kernel, `attn:mha` + the switch back), forced to the
array, median of 3:

| call | `attn.xclbin` | `attn_bfp16.xclbin` |
|---|---|---|
| 508 fresh | 7.1 ms | 7.0 ms |
| 1,019 fresh | 8.5 ms | 7.9 ms |
| 2,044 fresh | 10.6 ms | 8.7 ms |
| 4,092 fresh | 21.8 ms | 14.4 ms |
| 8,188 fresh | 65.8 ms | 35.6 ms |
| 24,031 fresh | 499.8 ms | 223.9 ms |
| 19 new on 8,195 cached | 13.7 ms | 8.6 ms |

Fit for `_attn_on_npu` (new path): 6,575 us + 0.000399 us per padded query-key pair + 0.13 us per cache row (the old
path keeps 6,500 + 0.00093 per pair). Placement barely moves: fresh prompts go to the array from ~660 tokens (was
~670), short follow-ups still stay on the host. The q tiles are not fully hidden under the g projection
(`attn:qkvg_proj` +0.2 ms per layer at 1K, +1.7 ms at 24K).

End to end, prefill ms, median of 5, A / B / C alternated (A = shipped engine + `attn.xclbin`; B = this engine +
`attn_bfp16.xclbin`; C = this engine + the exact build), 128 generated tokens, decode unchanged:

| prompt tokens | A shipped | B 2B kernel | C exact build |
|---|---|---|---|
| 1,019 | 394 | 397 | 405 |
| 2,044 | 627 | 619 | 633 |
| 4,092 | 1,124 | 1,072 (-5%) | 1,101 |
| 8,188 | 2,212 | 2,016 (-9%) | 2,120 (-4%) |
| 24,041 | 7,960 | **6,345 (-20%)** | 7,150 (-10%) |

24K stage profile (median of 3): attention 2,983 -> 1,323 ms, language prefill 7,949 -> 6,288 ms; no other stage moved
by more than ~16 ms.

## Prefill mm on bfp16 tiles (2026-11-05)

`mm_bfp16.xclbin` is the Qwen3.5-2B flash prefill matmul (`FLM_Xclbin/minicpm_ocr/dequant_mm_prefill`): x arrives as
bfp16 tiles made on the host, each W tile is converted to bfp16 once, each core sends its own y, k goes in passes of
256. Same weights in DDR, same y layout, so only x changes for the engine. **Bit-identical to `mm.xclbin`.**

How the engine drives it (`mm_bfp16.hpp`, `gen_dequant_mm_bfp16`):

* **x tiles made where x is made.** The five host steps that write an mm input (input RMS norm, post-attention RMS
  norm, the linear layers' gated norm, the attention layers' `g * sigmoid` product, the GLU) now go through
  `tiles::publish`: per 16-row group, the rows are computed into a small per-thread scratch and turned into bfp16
  tiles while they are still in cache (the 2B's `publish()`). The arithmetic is the old one: `_rms_norm` and
  `_exp_glu` were split into row helpers that the old loops call too.
* **bf16 rows only where they are read again.** The normed x of a linear layer is still written as bf16 (alpha /
  beta read it on the host); everything else only as tiles. A debug build (`DEBUG_LEVEL >= 1`) keeps every bf16
  copy, for the dumps and cross-checks.
* **Range syncs, no dirty lines.** Only the tile bytes written are flushed (and the bf16 rows when kept). Buffers the
  array writes later (x, after_mix, g) are no longer written by the host at all on this path, so no host cache line
  can land on array output. The old whole-buffer `memset` + flush of x per layer is gone on this path.
* **Padding rows stay zero.** Tile groups a longer call left past this call's L are zeroed, so padded rows read zeros
  as before.
* **Choice at load time**: `mm_bfp16_glu.xclbin`, else `mm_bfp16.xclbin` in the xclbin folder -> new path;
  `FLM_MINICPM_OCR_MM_BFP16=0` (or neither file) -> `mm.xclbin`, unchanged.

Checked: kernel bit-identical to `mm.xclbin` on the 7 shapes at L
512 / 2048 / 8192; in this engine the golden text 12/12 (incl. `--resume`, clear and images), the 10 prefill logits
bit-identical, `--multi-slot` passes. Prefill, median of 5, shipped engine vs this path, alternated:

| prompt tokens | `mm.xclbin` | `mm_bfp16.xclbin` |
|---|---|---|
| 124 | 123 ms | 103 ms (-16%) |
| 1,019 | 384 ms | 340 ms (-11%) |
| 2,044 | 618 ms | 550 ms (-11%) |
| 8,188 | 2,041 ms | 1,757 ms (-14%) |
| 24,041 | 6,445 ms | 5,538 ms (-14%) |

8K stage profile (median of 5): GDN in-proj 440 -> 375 ms, GDN o_proj 125 -> 93, attention projections 102 -> 77,
attention o_proj 53 -> 41, MLP up/gate + host GLU 448 -> 328, MLP down 162 -> 102; the host steps that now also make
tiles grew a little (norms +6, gated norm +5 ms). Decode is not touched.

### GLU on the array (`mm_bfp16_glu.xclbin`)

The same build installed as `mm_bfp16_glu.xclbin` also runs the GLU: the up/gate run (`gen_dequant_mm_bfp16_glu`,
`rtp_act` 2) writes `silu(gate) * up` itself, straight in the down projection's x tile format, into one 16 KB slot per
(128 rows, 32 features); the down run reads the slots (`x_from_glu_slots`). There is no host step and no sync between
the two runs, and no `upgate` / `hid` buffers on the host. `FLM_MINICPM_OCR_GLU_NPU=0` keeps the host GLU (then the
path is bit-identical to `mm.xclbin`). **Not bit-identical**: the SiLU is the array's hardware-tanh one (the same the
z projection's act 1 uses), rel L2 of the down projection's input 1.86e-2 vs the host GLU (1.50e-2 from the bfp16
tiles alone). The file name says the build has the GLU epilogue, so a build without it can never be sent `rtp_act`
2. Kernel side: `FLM_Xclbin/minicpm_ocr/dequant_mm_prefill/README.md` "GLU epilogue".

Checked (WP3 report): with `FLM_MINICPM_OCR_GLU_NPU=0` the same xclbin gives the golden text 12/12 and the 10 prefill
logits bit-identical. With the GLU on: prefill logits cos 0.99949-0.99997 (same top-1 except the photo prompt's exact
top-2 tie), `--multi-slot` passes, golden text 5/12 identical; the 7 changed replies come from 4 places, each an
exact or near tie in the old path (decode top-2 margins 0 / 0.0625 / 0 in raw logits, and `text_long_2k`'s first
word at 0.50, which WP1 alone also flipped). Prefill, median of 5, alternated, host GLU (`mm_bfp16.xclbin`) vs GLU on
the array:

| prompt tokens | host GLU | GLU on the array |
|---|---|---|
| 124 | 102 ms | 96 ms (-6%) |
| 1,019 | 349 ms | 329 ms (-6%) |
| 2,044 | 545 ms | 514 ms (-6%) |
| 8,188 | 1,732 ms | 1,610 ms (-7%) |
| 24,041 | 5,427 ms | 5,051 ms (-7%) |

8K profile: MLP up/gate + GLU 336 -> 227 ms (24 layers), language prefill 1,748 -> 1,600 ms.

## Host prefill work (WP4, 2026-11-05)

After WP1-WP3 the array part of a prefill had become much faster, and a fresh stage profile showed how much CPU work
was still on the critical path (the array waiting for the host): about 27% of a 124-token prefill, 42% at 1K, 25%
at 8K, 22% at 24K. The largest pieces: the host conv (it ran "under" the g projection, but that run is much
shorter), the host gated delta rule below 1,536 tokens, the norms / residual adds / gated norm between the
projections, the buffer allocations of a new length, and a few whole-buffer syncs. WP4 removed most of it with
seven changes, ported from the Qwen3.5-2B flash engine where one existed. **Every change is bit-identical**: same
arithmetic per value, only the order of independent work, the threads and the syncs changed.

| step | change | where |
|---|---|---|
| 1 | conv1d + SiLU + q/k L2 norm in one pass per row | `minicpm_ocr_prefill_cpu.hpp` `conv1d_silu_l2norm` |
| 2 | qkv projection in row chunks; the host conv of a chunk runs under the next chunk's run | `_in_proj_overlapped_npu`, `prefill_chunks.hpp` |
| 3 | o_proj and MLP in row chunks; residual add fused into the norm after it; the next layer's input norm under this layer's last down run | `_forward_chunked`, `run_o_chunks`, `forward_chunked` |
| 4 | host gated delta rule: a whole head per work item | `gated_delta_net_slabs<8>` |
| 5 | no lm_head weight sync per prefill | `_prefill_with_mm` |
| 6 | row-range syncs for the scratch the host reads or writes | linear / attention blocks |
| 7 | the MLP's and the attention block's buffers for a new length made on a second thread | `_reserve_ahead` |

### 1. Conv, SiLU and the q/k L2 norm in one pass

`conv1d_silu()` then `l2norm_heads()` read and wrote the whole conv output twice, and the L2 norm's scaling loop was
scalar (its `static_cast<bf16>` blocked vectorizing). `conv1d_silu_l2norm()` normalizes each row right after making
it, while it is still in the core's cache, and scales 16 values at a time.

The trick is the bf16 rounding of that scaling: the scalar `static_cast<bf16>` (biovault) rounds to nearest even,
but turns zero and subnormal inputs into a signed zero, and the vector rounding used elsewhere does not.
`f32_bf16_like_scalar_512()` does both, so it gives the same bits. Checked by brute force over all 2^32 float
inputs, compiled with the engine's flags (`logs/WP4/bench/cvt_engine.cpp`: 0 mismatches). Under `-ffast-math` the
scalar path does not treat NaN specially either (a first version that did had 12.6 M NaN mismatches), so neither
does the vector one. Standalone, 16 threads: 8K rows 7.8 -> 4.2 ms per layer, 1K 1.0 -> 0.43 ms. Streaming
(non-temporal) stores and 32 threads were tried: no gain (8 -> 16 threads gives only 5.3 -> 4.1 ms, it is close to
memory-bound now).

### 2. The qkv projection in row chunks

The conv of a row reads only its own qkv row and the 3 rows before it. So once the array has projected the first
rows, the host can already run their conv. `prefill_chunks.hpp` splits a call's rows into up to
`FLM_MINICPM_OCR_PREFILL_CHUNKS` chunks (default 3, `1` = one run as before) on 512-row steps, the first chunk the
larger one:

```
array:  [ qkv rows c0 ][ qkv rows c1 ......... ][ qkv rows c2 ......... ][ g run ............. ]
host:   [ alpha/beta ]  [ carry in + conv c0 ]  [ conv c1 ]               [ conv c2 + carry out ]
```

`gen_dequant_mm_bfp16(_glu)` got a last parameter `x_row_begin` (default 0): a run over one chunk reads x from that
row on. Measured at 8K, in-proj stage over the 18 GDN layers: 1 / 2 / 3 / 4 chunks = 301 / 262 / 253 / 250 ms
(24K: 833 / 734 / 675 / 654). A prompt of up to 512 tokens has one chunk and runs as before.

**Lesson (a bug found by the golden set, fixed before the commit):** the "carry out" copies this call's last 3 qkv
rows into the conv-state cache. A first version did it before the last chunk's range sync, so it could read stale
host cache lines. Prefill logits were still bit-identical (they do not use that copy), but the next call (decode, a
follow-up turn) started from a wrong conv state: 2 of the 12 golden replies changed. Every host read of array output
must come after the sync of exactly those rows.

### 3. o_proj and the MLP in row chunks (the 2B's layer pipeline)

The same chunks, for the rest of the layer (`_forward_chunked`, mm_bfp16.xclbin only):

```
array:  o(c0)     o(c1)      o(c2)     up(c0)    up(c1)  up(c2)  down(c0)  down(c1)  down(c2)  | next layer ...
host:   prep(c1)  prep(c2)   post(c1)  post(c2)                            tail(c0)  tail(c1)  | tail(c2)
                  post(c0)
```
(prep(c0) runs before o(c0); with one chunk everything runs in order.)

* `prep(c)`: the gated norm (GDN layers) or the gate product (attention layers) of chunk c, the o_proj's input tiles.
* `post(c)`: residual add + post-attention RMS norm of chunk c, into the up/gate run's tiles.
* `tail(c)`: residual add + the **next layer's input RMS norm** of chunk c. So the next layer starts with its input
  norm done (`input_normed_layer`); the last layer only adds.
* Each residual add and the norm after it are one pass per row (`cpu_func::_residual_add_rms_norm_row`): add, round
  to bf16, store, then normalize the stored value, exactly as `_residual_add` then `_rms_norm_row` did.
* `tiles::publish_rows()` makes one chunk's rows of an mm input; the call that covers the last row also zeroes the
  stale tile groups past L (as `publish()` did).
* With the GLU on the host (`FLM_MINICPM_OCR_GLU_NPU=0`) the GLU of chunk c - 1 runs under up(c), the last one
  under down(c0).
* Debug builds: the pipeline overwrites x chunk by chunk before the dumps and cross-checks of the whole tensor run,
  so the row-chunk path keeps copies of x as the array wrote it (`debug_block_out`, `debug_mlp_out`) for
  `linear_out` / `attn_o_proj` / `mlp_down` and their cross-checks.

The 2B engine does this with 2 chunks; here the same 3 chunks as step 2 are used. Effect: 8K 1,483 -> 1,431 ms,
24K 4,711 -> 4,506 ms (A/B, 3 rounds); about neutral at 1K (two 512-row chunks, little host work to hide).

### 4. The host gated delta rule, a whole head per work item

Below 1,536 tokens the GDN runs on the host (placement unchanged). Its loop over one 16-value slab of the state is
two chains of 128 dependent FMAs per token: each FMA waits for the one before (4 cycles), so the core mostly waits.
`gated_delta_net_slabs<8>` updates all 8 slabs of a 128-wide head side by side: 8 independent chains to fill the
FMA pipes, and each token's q / k converted once per head instead of once per slab. Each slab keeps its own order of
operations, so outputs and the final state are bit-identical (checked at L 1 .. 1,535). Standalone: 1K 8.3 -> 3.0 ms
per layer, 2.8x. In the engine at 1K: GDN 94 -> 39 ms over the 18 layers, **prefill 319 -> 230 ms**.

Note: the 1,536-token host/array threshold was fitted to the old host speed; the host is now ~2.5x faster, so the
real crossover is higher. It was not changed here (WP4 keeps placement); WP4c moved it to 5120.

### 5. No lm_head weight sync per prefill

`_prefill_with_mm` synced the whole 294 MB lm_head BO on every call (~2 ms). `load_weights()` already writes and syncs
it once and nothing writes it later, so the sync is gone.

### 6. Row-range syncs

The remaining whole-buffer syncs of prefill scratch (qkv on the one-chunk path, gate, after_mix on both GDN and
attention sides, the attention q / g / k / v) now cover rows [0, L) only. q keeps a whole sync when its tiles for the
array are made (the tile converter reads every padded row). `scratch_push_range()` is a no-op when scratch is host
memory. This matters only when L is much smaller than L_padded (124 tokens in 512 rows): the sync stages at 124 tokens
shrink by ~1.9 ms in the profile; the whole prefill moves within the box's noise.

### 7. Buffers of a new length on a second thread

The first call at a new padded length creates new BOs (~0.1 ms per MB) and array sequences: 8K ~67 ms, 24K ~190 ms
in all. `_reserve_ahead()` runs the MLP's and the attention block's `reserve()` on a second thread (`std::async`)
while layer 0's block runs, started after that block's own `reserve()` (two threads allocating at once slowed each
other); `_wait_reserved()` joins it before either block is used (measured wait ~0.1 ms). The two reserves touch only
their own block's buffers and sequences; the sequence generator is read-only and has no shared mutable state. At 8K
the prefill drops by ~35 ms; short prompts gain nothing measurable.

**XRT only.** Under HRX (`FLM_USE_HRX`) the same three `reserve()` calls run on the calling thread. HRX allocates every
BO on the one process-wide stream (`hrx_cpp.hpp`, `rt().stream`) that also dispatches and waits for the array runs,
and nothing serializes the two threads on it: an allocation from the second thread while the first one waits broke
the stream's timeline (`run::wait ... semaphore already signaled past N`, then `hrx_buffer_allocate failed` for this
and every later model in the process; FastFlowLM PR #983's HRX tests). Measured with the CI HRX package on Strix
Halo: the npu_test chat suite passes (text and images), an 8K prompt prefills in 1.45 s and its follow-up resumes
from the prompt cache. The XRT library is unchanged (same bytes).

### Row chunks on HRX

Under HRX the prefill runs every projection as one run (`prefill_chunks.hpp`: `allowed_chunks = 1`, the env
switch cannot raise it). The row-chunk pipeline (steps 2, 3 and 6) has the host write and read some rows of a BO while
the array works on other rows of the same BO, and hands rows over with row-range syncs. HRX has no partial flush or
invalidate (`buffer.hpp`: the range syncs sync the whole BO), so a sync for one chunk also flushes or drops the rows
of the other chunk. Measured with the CI HRX package: an 8K prompt gave a different, wrong reply on every run with
chunks on (`FLM_MINICPM_OCR_MM_BFP16=0` or `FLM_MINICPM_OCR_PREFILL_CHUNKS=1` made it match XRT); with one chunk HRX
gives XRT's reply at 8K (twice) and 24K. Cost: 8K prefill ~1.5 s, 24K 5.25 s (XRT 4.87 s). The XRT library is unchanged
(same bytes).

### Results

Prefill, `so/wp4.so` with the installed xclbins vs the shipped engine (WP1-WP3), median of 5 processes, alternated,
128 generated tokens (`logs/WP4/speed_final`):

| prompt tokens | shipped | WP4 | change | decode tok/s (shipped / WP4) |
|---|---|---|---|---|
| 124 | 95 ms | 81 ms | -15% | 66.4 / 66.5 |
| 1,019 | 317 ms | 224 ms | -29% | 64.3 / 63.8 |
| 2,044 | 506 ms | 471 ms | -7% | 62.3 / 61.9 |
| 8,188 | 1,612 ms | 1,409 ms | -13% | 51.5 / 51.6 |
| 24,041 | 5,019 ms | 4,453 ms | -11% | 36.4 / 36.4 |

2K gains least: there the GDN runs on the array, so step 4 does not apply. Checked: golden text 12/12 (incl.
`--resume`, clear, images), the 10 prefill logits and fresh prompts of 512 .. 24K tokens bit-identical to the
installed build, `--multi-slot` passes, every A/B reply identical; `FLM_MINICPM_OCR_PREFILL_CHUNKS=1`,
`FLM_MINICPM_OCR_GLU_NPU=0` and `FLM_MINICPM_OCR_MM_BFP16=0` still give their old golden text 12/12.

### Not done (measured, left for later)

* **Attention layers' host steps** (q rope + tiles, sigmoid, fill KV, the bfp16 KV copy update): ~45 ms exposed at
  8K, ~140 ms at 24K. A reorder (k and v runs first with their KV copy work under the q / g runs, the sigmoid under
  the attention run) would hide an estimated ~18 / ~55 ms; it needs `kv_store::update` split into its K and V halves. Done in WP4d (section "Attention-layer host steps").
* **The bfp16 KV copy and `fill_kv_cache`** stay where they are (part of the item above).
* **Grow-only scratch** across calls (README "Host-side prefill speed-ups"): not done; step 7 hides most of the
  allocation of a first call at a new length instead.

## GDN and attention placement re-tuned (WP4c, 2026-11-06)

WP4 made the host gated delta rule ~2.4x faster, so the GDN host/array threshold (`gdn_npu_min_tokens()`) was
measured again. **New default: 5120 tokens** (was 1536). `FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS` still overrides it.
The attention cost model (`_attn_on_npu`) was checked the same way and left as it is.

Method: one `.so` (the installed one), GDN forced to the host (threshold 1000000) or to the array (threshold 0),
median of 5 processes, the two arms alternated, `--Length 1`, `FLM_MINICPM_OCR_PREFILL_PROFILE=1` (the per-layer-kind
split works in the default build). Prompts: the `table/p*.txt` files plus more lengths cut the same way from the
same story.

Fresh prompts, ms (linear layers = the 18 GDN layers with their projections and MLP):

| prompt tokens | linear layers, host GDN | linear layers, array GDN | prefill, host GDN | prefill, array GDN | array - host |
|---|---|---|---|---|---|
| 508 | 82 | 195 | 129 | 246 | +117 |
| 1,019 | 147 | 245 | 236 | 330 | +94 |
| 1,532 | 218 | 305 | 327 | 407 | +80 |
| 2,044 | 288 | 360 | 416 | 481 | +65 |
| 2,556 | 348 | 412 | 492 | 544 | +52 |
| 3,068 | 433 | 468 | 610 | 627 | +17 |
| 4,092 | 538 | 573 | 745 | 766 | +21 |
| 4,604 | 603 | 625 | 836 | 858 | +22 |
| 5,113 | 690 | 718 | 956 | 980 | +24 |
| 5,628 | 747 | 726 | 1,032 | 989 | -43 |
| 6,140 | 808 | 796 | 1,113 | 1,090 | -23 |
| 8,188 | 1,081 | 997 | 1,512 | 1,404 | -108 |
| 12,283 | 1,589 | 1,431 | 2,310 | 2,113 | -197 |
| 24,041 | 2,983 | 2,673 | 4,838 | 4,536 | -302 |

Fit up to 8K, linear layers: host 19 ms + 0.130 ms/token, array 144 ms + 0.105 ms/token, crossing at ~5.2K. The
array's fixed cost (~125 ms per call: two switches in each of the 18 layers, plus setup) did not change; the host's
per-token cost fell from 0.285 to 0.130 ms (over the whole linear layers now, not the GDN alone).

Cached follow-ups (3,120-token history, `--resume`; the prefix cache splits the turn into a 7-token call and the
new tokens; the array arm uses threshold 1024 so the 7-token call stays on the host):

| new tokens | prefill host GDN | prefill array GDN | array - host |
|---|---|---|---|
| 2,044 | 548 | 605 | +57 |
| 4,092 | 940 | 921 | -19 |
| 5,113 | 1,104 | 1,090 | -14 |
| 6,140 | 1,330 | 1,259 | -71 |
| 8,188 | 1,762 | 1,626 | -136 |

The linear layers alone cross near 5.1K here too; the whole call crosses earlier (~3.6K) because the attention
layers run 15-40 ms slower in every host-GDN run (fresh prompts too, smaller); WP4d found why (the array runs
right after the host GDN's all-core burst are slower: shared power budget, section "Attention-layer host steps"). Between 4K and 5.6K either side is within ~25 ms (2-3%). 5120 follows the fresh prompts
(first turns, image turns); a follow-up of 3.6K-5.1K new tokens loses up to ~20 ms.

Golden cases that change GDN side: `text_long_2k` (2,009 fresh), `resume_after_3k_array` turn 1 (3,120, so turn 2
starts from a host-made state) and `handoff_host_to_array` turn 2's tail (2,043 new on 988 cached) now run the GDN on
the host. Replies: 12/12 unchanged. Their last-row logits move (cos >= 0.99994, same top-1); against a CPU reference
(`PREFILL_MM=cpu`, GDN on the host or on the array) the moved calls are slightly closer than before (6 calls: mean
rel L2 8.3e-2 -> 7.8e-2). The GDN side alone moves the reference by rel L2 ~1e-2, well below the array path's own
error from the bfp16 mm.

End to end, prefill ms, installed engine (A, 1536) vs this one (B, 5120), installed xclbins, 128 generated tokens,
median of 5 processes, alternated:

| prompt tokens | A | B | change | decode tok/s A / B |
|---|---|---|---|---|
| 508 | 132 | 134 | same side | 65.3 / 65.4 |
| 1,019 | 240 | 239 | same side | 64.5 / 64.4 |
| 1,532 | 339 | 332 | same side (host) | 63.3 / 63.3 |
| 2,044 | 475 | **420** | -12% | 62.2 / 62.4 |
| 3,068 | 622 | **596** | -4% | 60.3 / 60.3 |
| 4,092 | 769 | 765 | -1% | 58.5 / 58.6 |
| 8,188 | 1,408 | 1,398 | same side | 52.4 / 52.4 |

Replies identical between A and B except the 3,068-token prompt, which changes wording at about token 100 of 128.

Attention placement (`FLM_MINICPM_OCR_ATTN_BACKEND=host|npu`, same method): the model picks the faster side at
379 / 508 / 636 (host) and 764 / 1,019 (array) fresh tokens, and on follow-ups after 3K and 8K histories. The two
closest follow-up cases (60 new on 3K cached, 28 new on 8K cached) are ties: their attention layers differ by 1 ms
either way. Constants unchanged.

## Attention-layer host steps (WP4d, 2026-11-06)

After WP4 the 6 attention layers still did most of their host work while the array waited. A stage profile of the
installed engine (WP4 + WP4c) with a measurement build that skips those host steps gives the exposed time, ms
summed over the 6 layers:

| host step | 1,019 tok | 4,092 | 8,188 | 24,041 |
|---|---|---|---|---|
| q pull + q norm/RoPE + q tiles beyond the g run | 0.5 | 1.7 | 7.4 | 24.7 |
| g pull + sigmoid beyond the k run | 0.3 | 1.8 | 5.6 | 15.9 |
| k / v pulls + k norm/RoPE beyond the v run | 0.3 | 0.9 | 1.8 | 5.3 |
| `fill_kv_cache` (K + V rows, flush) | 2.0 | 8.0 | 15.3 | 37.2 |
| bfp16 K/V copy: conversion | 0.4 | 2.1 | 4.5 | 11.8 |
| bfp16 K/V copy: BO allocation (first call at a new length) | 4.7 | 4.8 | 7.4 | 21.8 |
| **sum** | **8.1** | **19.4** | **42.1** | **116.8** |
| GDN layers: gate (z) read-back (array GDN only, >= 5,120 tok) | - | - | 5.4 | 16.2 |

Three changes, each bit-identical (same functions on the same rows, only the order, the threads and the time of a
sync change):

**(a) The attention stage as one pipeline** (`_qkvg_attend_bfp16_npu`, used for array attention on
`attn_bfp16.xclbin` with the mm on the array; host attention, `attn.xclbin` and host-mm builds keep the old code):

    array:  k     v                     q                          g                  attention
    host:         k: pull, norm+RoPE,   K tiles; v: pull, V rows,  q: pull, norm+RoPE, g: pull, sigmoid;
                  K rows                V^T tiles                  q tiles            flushes

The projections run k, v, q, g (was q, g, k, v), so the K and V work starts as soon as its projection is done. The
bfp16 KV copy update is split into `kv_store::prepare` (sizes, under the k run), `convert(K)`, `convert(V)` and
`finish`; `update()` is now these steps in a row (still used by host-mm builds). The new cache rows are copied by
all host threads (one contiguous block per half). The gate sigmoid and the flushes of q, k, g and the new cache rows
run under the attention run: nothing reads those device copies before the next layer or decode.

**(b) The gate (z) read-back under the array GDN.** With the GDN on the array, the g run's output is pulled while
the GDN run goes (the call is split into start / wait), instead of between the two runs with the array idle.
`_pull_gate()` does it once per call before the first host read (the gated norm, or the debug cross-check).

**(a2) The bfp16 KV copies sized on the second reserve thread.** The copies of all 6 attention layers grow on the
thread that already makes layer 0's new-length scratch (`_reserve_ahead`, WP4 step 7), with the same
`rows_for` / `allocate` code `prepare()` runs. The engine passes the attention KV cache size in (`kv_cache_elems`).

Results. Every step: golden text 12/12 (incl. `--resume`, clear, images), the 10 prefill logits and fresh prompts of
512 .. 24K tokens bit-identical to the installed build, `--multi-slot` passes, every A/B reply identical; also a 2K
turn + 4K cached follow-up that grows the KV copies, and the golden set under `FLM_MINICPM_OCR_PREFILL_CHUNKS=1`,
`GLU_NPU=0`, `ATTN_BFP16=0`, `MM_BFP16=0`, `ATTN_BACKEND=npu`, `GDN_NPU_MIN_TOKENS=0` identical to the installed engine
with the same switch. Prefill ms, installed engine (A) vs WP4d (B), installed xclbins, 128 generated tokens, median
of 5 processes, alternated:

| prompt tokens | A | B | change | decode tok/s A / B |
|---|---|---|---|---|
| 1,019 | 235 | **230** | -2% | 64.4 / 64.5 |
| 2,044 | 417 | 415 | 0% | 62.3 / 62.3 |
| 4,092 | 764 | **752** | -2% | 58.5 / 58.6 |
| 8,188 | 1,398 | **1,371** | -2% | 52.5 / 52.4 |
| 24,041 | 4,490 | **4,396** | -2% | 37.0 / 36.9 |

Per step (stage profile, ms summed over layers): (a) attention blocks 307 -> 278 at 8K, 1,597 -> 1,525 at 24K;
(b) the exposed z read-back 5.5 / 16.1 -> 0.1 at 8K / 24K; (a2) the KV-copy allocation left in the attention layers
5.1 / 9.3 / 25.2 -> 0.3 / 0.9 / 2.2 at 1K / 8K / 24K (at 24K the longer second thread costs layer 0 about as much).
Still exposed: the q norm + RoPE + q tiles run ~1-2 ms per layer longer than the g run at 8K / 24K.

### Why the attention layers are slower when the GDN runs on the host (WP4c's open question)

WP4c saw the attention layers run 15-40 ms slower in every prefill whose GDN ran on the host. The profile shows it
is not the attention's host steps (same times in both cases) but the **array runs right after the host GDN**: the
o_proj and MLP runs of that GDN layer, and in the next attention layer the first projection and the attention run.
At 8K, per layer: MLP 13.7 -> 18.2 ms, attention run 29.3 -> 34.1 ms. Tests (8K, GDN on the array, a 21 ms gap
added after the GDN run; or the host GDN with a pause after it; median of 5):

| what runs in the gap after the GDN | MLP after it | attention run | attention layers (6) |
|---|---|---|---|
| nothing (GDN on the array) | 13.7 | 29.3 | 384 |
| CPU sleeps 21 ms (array idle too) | 13.8 | 29.3 | 386 |
| all host threads write 64 MB (dirty cache lines left) | 13.7 | 29.3 | 386 |
| all host threads spin on AVX-512 FMAs for 21 ms | 14.6 | 30.3 | 392 |
| the same GDN math on copies (16 threads, ~27 ms) | 15.7 | 30.7 | 391 |
| host GDN (the real one) | 18.2 | 34.1 | 424 |
| host GDN, then the conv output flushed | 18.5 | 36.6 | 434 |
| host GDN, then the CPU sleeps 21 ms | 14.0 | 29.3 | 386 |

So an idle array, dirty cache lines and flushes are not the cause; heavy all-core CPU work is, and its effect on the
array fades within ~20 ms of CPU rest. This box is a laptop running at its package power limit during prefill (PPT
40-54 W, CPU 91-98 °C, `balanced` platform profile, NPU power mode `Performance`): CPU and NPU share one power budget,
and the array runs slower for a short while after a burst of the CPU's AVX-512 work. There is no bit-identical host
fix that pays (a pause costs more than it saves); the effect is already inside WP4c's measured threshold. To check
with root on another profile (`performance`), or a desktop part, before relying on it elsewhere.

## Decode attention on bfp16 K and V (WP6, 2026-11-06)

At long context decode is bound by reading the KV cache (12 KB per cached token per token). `layer_with_lm_head_kvb.xclbin`
(`FLM_Xclbin/minicpm_ocr/decoding`, `make bitstream KV_BFP16=1`; README there, "bfp16 K and V in decode attention")
is `layer_with_lm_head.xclbin` whose attention reads the finished 16-key chunks of K and V as bfp16: 44% fewer bytes,
the blocks its emulated bf16 mmul made anyway, so the result is the same bit for bit.

* **Choice at load time:** `layer_with_lm_head_kvb.xclbin` present -> this path (it includes the lm_head);
  `FLM_MINICPM_OCR_DECODE_KV_BFP16=0` or no such file -> `layer_with_lm_head.xclbin` as before;
  `FLM_MINICPM_OCR_LM_HEAD_IN_LAYER=0` -> the old two-xclbin path (it wins over the kvb file).
* **The copy** (`decode_kv_bfp16.hpp`): per attention layer one BO, passed as **arg 3** of its decode run (the 5 BO
  arguments are all taken; prefill reads `rope_rms_weights` on the host only, decode reads it from here):
  `[4 KB head area = rope_rms_weights][K unit x cap][V unit x cap]`, unit = one chunk of both KV heads, 9,216 B
  (`bfp16_pack.hpp`, the packers WP2's prefill attention uses too). `cap` grows in 4,096-token steps (BOs made again,
  runs re-created); at 32K it is 36 MB per layer, 216 MB for the 6 layers (the bf16 caches are 384 MB).
* **When chunks are converted:** the chunk a token's row finished, under that token's lm_head run (one chunk x 6
  layers, a few us of host time, hidden); anything missing before the next layer run list (after a prefill, a restore,
  a state write: e.g. ~1,500 chunks x 6 layers after a 24K prefill, multi-threaded). Rows the array wrote are pulled
  from the device (range sync) first. The newest, unfinished chunk is read from the bf16 cache by the array itself.
* **Invalidation:** `clear_context`, `restore`, `write_state`, `write_state_kv`, `update_max_length` mark rows from
  their start row on to be converted again (as WP2's `kv_cache_changed`). Prefill only appends rows the copy never
  holds.
* Why not one copy for prefill and decode: decode needs both KV heads of a chunk next to each other in one stream;
  WP2's copy keeps each head's rows contiguous, and a shim BD cannot gather it (strides are 20-bit words = 4 MB,
  sizes 10-bit; the head stride is past 4 MB above ~14K rows). The packing code is shared instead.
* **Rescale skip:** `FLM_MINICPM_OCR_DECODE_RESCALE_SKIP=0` turns it off (default on). The sequence writes RTP `L[2]`
  of the P.V core: 1 = when no row max moved, y is left as it is instead of the emulated multiply by exactly 1.0
  (decoding README, "Rescale skip"). **Not bit-identical**: tiny y values (after big rescales) can come out different
  in the last bits.

Correctness:

* bfp16 K/V, skip off: **bit-identical**. Per-step decode logits byte-identical to `layer_with_lm_head.xclbin`
  (316 / 316 steps: short, 1K, 8K, 24K prompts and a resumed 3K follow-up); golden replies 12/12, prefill logits 10/10,
  `--multi-slot` PASS; kernel-folder `--check` 45/45 byte-identical at 0..32K cached tokens.
* With the rescale skip (default): short / 1K / 8K identical; 24K differs from decode step 46, the resumed 3K
  follow-up from step 6; min cos 0.999978, top-1 the same at every step, replies unchanged (golden: see the WP6
  report).

Decode speed, 128 generated tokens, median of 5, A/B/C alternated (A installed `layer_with_lm_head.xclbin`, B bfp16
K/V with the skip off, C bfp16 K/V + rescale skip = default), decode tok/s; per-token layer time from the profile build
(median of 3):

| prompt tokens | A | B | C | ms/token in the 24 layer runs A -> B -> C |
|---|---|---|---|---|
| 1,019 | 64.5 | 64.9 | **65.4** | 10.22 -> 10.09 -> 9.98 |
| 8,188 | 52.3 | 53.9 | **57.8** | 13.84 -> 13.05 -> 11.86 |
| 24,041 | 37.1 | 40.0 | **47.1** | 21.62 -> 19.57 -> 15.92 |
| 31,001 | 32.5 | 35.4 | **43.3** | 25.38 -> 22.62 -> 17.87 |

Prefill unchanged. The first decode token after a prefill also converts the whole history (~45 ms after 24K, once;
included above). With the skip the attention layers run at the speed of their two KV streams (one shim channel each,
~13.7 GB/s); the math is hidden.

## End-to-end latency, as shipped

Conv on the host, GDN on the array for prefill calls of >= 1536 tokens (rewritten kernel staged),
attention placed per call, 128 generated tokens, median of 5 processes each. Model load is not included. TTFT = tokenize + prefill
+ one sampling step. Measured 2026-10-29 after the host-side speed-ups; "before" is the same table
measured earlier that day.

| prompt tokens | end-to-end | prefill | prefill before | TTFT | decode (128 tok) | decode tok/s |
|---|---|---|---|---|---|---|
| 128 | 2,545 ms* | 123 ms | 185 ms | 125 ms | 2,302 ms* | 47.4 |
| 512 | 3,086 ms | 190 ms | 237 ms | 192 ms | 2,775 ms | 46.9 |
| 1,023 | 3,351 ms | 376 ms | 395 ms | 379 ms | 2,843 ms | 45.7 |
| 2,048 | 3,723 ms | 629 ms | 707 ms | 634 ms | 2,931 ms | 44.4 |
| 4,096 | 4,382 ms | 1,112 ms | 1,260 ms | 1,119 ms | 3,111 ms | 41.8 |
| 8,192 | 5,877 ms | 2,241 ms | 2,516 ms | 2,251 ms | 3,475 ms | 37.4 |

\* At 128 tokens the reply now stops by itself after 107 tokens (host attention changed the text),
so that row's end-to-end and decode times cover 107 tokens, not 128.

Prefill spread across the 5 runs stayed within ~2% of the median at every length except 1K
(352-378 ms). A median of 3 landed in this box's slow mode once in the earlier table (1K read
475 ms instead of 395), which is why it is 5 here.

*(The paragraph below is from 2026-10-28, before the kernel rewrite.)* **So the default stays on the host**, and the kernels shipping in the xclbin folder
(`GateDeltaNet_prefill.xclbin`, `conv.xclbin`) are unused by it -- though at long context the
choice costs little either way. Two cases still worth measuring before trusting any of this:
`flm serve` under *concurrent* load (these stages use OpenMP while decode polls, so a prefill in
one request can slow a decode in another), and whether host prefill scales as well once the box is
not idle.

## Performance

Context-length sweep, medians of 3, since this box is bimodal
per process at about +/-15%. Measured 2026-10-28 on an idle box, shipping default build.
`prefill_ms` is the harness's prefill time, i.e. TTFT minus one decode step (~21 ms short-context).

| prompt tokens | prefill ms | prefill tok/s | decode tok/s | ms/token | peak RSS MB |
|---|---|---|---|---|---|
| 12 | 176 | 68 | 47.6 | 21.0 | 3801 |
| 123 | 186 | 663 | 47.7 | 21.0 | 3801 |
| 508 | 243 | 2089 | 47.2 | 21.2 | 3802 |
| 1,020 | 407 | 2508 | 46.4 | 21.6 | 3802 |
| 4,092 | 1,501 | 2726 | 41.8 | 23.9 | 3811 |
| 8,015 | 3,094 | 2590 | 38.0 | 26.3 | 4102 |
| 16,023 | 6,813 | 2352 | 31.2 | 32.0 | 4686 |
| 24,031 | 10,900 | 2205 | 26.3 | 38.0 | 5222 |

Cold start, process launch to `Model loaded`: **3.45 s**, and remarkably stable (3.44/3.45/3.45).

Three things the shape says:

* **Prefill peaks at 4k and falls after.** 2726 tok/s at 4,092 tokens against 2205 at 24k. This is
  attention, whose cost grows faster than linearly with length: the 6 attention layers take 757 ms
  at 8k but 4,029 ms at 24k (5.3x for 3x the tokens), while the 18 GatedDeltaNet layers grow
  linearly. *(Corrected 2026-10-29: this line first blamed chunking at `max_prefill_len`. There is
  none -- MiniCPM_OCR never reads `max_prefill_len`, and the profile shows a 24k prompt as one
  prefill call.)*
* **Short prompts are overhead-bound.** ~176 ms of fixed cost means 12 tokens prefill at 68 tok/s
  and 123 tokens at 663. Nothing is wrong; there is simply nothing to amortise over.
* **Decode degrades 45% from short context to 24k** (47.6 -> 26.3 tok/s, 21.0 -> 38.0 ms/token).
  Expected: only the 6 full-attention layers scan a growing KV cache, the 18 GatedDeltaNet layers
  carry fixed-size state. RSS grows ~60 KB per token of context.

Against the sibling's committed CI baseline (`.github/scripts/baseline/xrt/perf/minicpm-v-4.7_1b/`,
2026-10-08) **decode agrees closely** -- 46.4 vs 47.7 at 1k, 31.2 vs 31.7 at 16k -- which is the
useful comparison, since the two share a text tower. Prefill reads 40-65% higher here, but that is
**not** a like-for-like number: different box (CI uses a pinned perf-bench machine), different
harness (`flm bench` rather than this test harness), and a different context-length convention. Do
not quote it as a speedup.

**No perf target was ever set for this model** (Step 0 asks for one). These are the numbers; the
target is still the user's to choose.

## Verifying

| What | How |
|---|---|
| bf16 reference, per layer | `models_simple/verify_minicpm_ocr.py` — 24/24 layers at cos 1.0000000 |
| `model.q4nx` vs its GGUF | `FLM_Q4NX_Converter/verify_minicpm_v_q4nx.py --tied-lm-head --config configs/minicpm_ocr_1b.json --model-def .../minicpm_ocr_model_def.hpp` |
| what quantization cost | `FLM_Q4NX_Converter/verify_minicpm_ocr_q4nx_vs_bf16.py` — Q4_K body at cos ~0.9973, rest free |

`host_ref/` was carried over from the sibling and its scripts now point at `dumps/minicpm_ocr/`,
but they have **not** been re-run for this model yet.

## Known gaps

- `tools/verify_minicpm_ocr_vision.cpp` still builds positions with `canvas_mrope()`. For this
  model they should be sequential, so that tool tests a position rule the engine no longer uses.
- The engine has not been run on the NPU yet: no decode, no prefill, no end-to-end output.
- Not yet wired into FastFlowLM (`all_models.hpp`, `model_list.json`, `src/test/`).
