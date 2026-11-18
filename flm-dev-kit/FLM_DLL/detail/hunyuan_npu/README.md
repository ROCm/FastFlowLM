<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# hunyuan_npu

Engine for the `hunyuan-dense` GGUF architecture (Hy-MT2-1.8B).

**Status: decode, lm_head, and the prefill matmul + attention all run on the array.**
Two bitstreams: `layer.xclbin` (from `FLM_Xclbin/Hy_MT2_1.8B/decoding/`) carries decode
*and* lm_head via `gen_layer_seq` / `gen_set_context` / `gen_lm_head_seq`;
`prefill_overlay.xclbin` (from `FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay/`) carries the
prefill matmul and prefill attention, taking turns on the array, via
`hunyuan_overlay::gen_dq_run` / `gen_attn_run` (see "Prefill overlay"). Weights: Q4_K body,
Q8_0 lm_head (since 2026-11-12, see "Q4_K weights"). RoPE + QK norm, the GLU's multiply, both
RMS norms and `fill_kv_cache` are still host. Sections below that name `fused_prefill`,
`gen_dequant_mm_512`, `gen_mha_engine_seq` or q4_0 describe the history (all deleted).

## Architecture

A Llama-shaped GQA decoder (see the HunYuan V1 dense model card / HF modeling code) with two
deviations, both of which this engine implements and both of which are easy to
get wrong by pattern-matching another model:

* **QK norm runs after RoPE**, not before. `cpu_func::_rope_then_qk_rms_batch`
  rotates the full head first and normalizes second. Every other QK-norm model
  in the tree does the opposite; on the array side this is why
  `FLM_Xclbin/generic_decoding_layer` links `rope_hunyuan.o` for this model.
* **RMSNorm rounds to bf16 before the weight multiply**:
  `out = w * (bf16)(x * rsqrt(mean(x²) + eps))`, matching `HunYuanRMSNorm`.

RoPE is full-head (`RoPE_DIM == head_dim == 128`), NEOX/`rotate_half` ordering,
positions are plain 1-D absolute. The DynamicNTKAlpha base rescale
(`base = rope_theta * alpha**(DH/(DH-2))`) is static and is folded into
`rope_base` in `hunyuan_desc::build`; the GGUF already carries the rescaled
`rope.freq_base`, so a checkpoint that sets `rope_scaling.rope_type: "none"`
lands on the same number either way.

`head_dim` is read from the config and is **not** assumed to equal
`hidden_size / num_attention_heads`.

## Config → backend mapping

| config.json | `hunyuan_desc` | Hy-MT2-1.8B |
| --- | --- | --- |
| `hidden_size` | `D` | 2048 |
| `num_attention_heads` | `num_heads` | 16 |
| `num_key_value_heads` | `num_kv_heads` | 4 |
| `head_dim` | `DH`, `RoPE_DIM` | 128 |
| `intermediate_size` | `intermediate_size` | 6144 |
| `num_hidden_layers` | `num_hidden_layers` | 32 |
| `vocab_size` | `vocab_size` / `vocab_size_padded` | 120818 / 121856 |
| `rms_norm_eps` | `rms_norm_eps` | 1e-5 |
| `rope_theta` + `rope_parameters`/`rope_scaling` | `rope_base`, `inv_freq` | 11158840.0 |

Derived: `DQ = 2048`, `DK = DV = 512`, `attn_scale = DH^-0.5`.
Run with `DEBUG_LEVEL=1` to dump every parsed field.

## Weight layout

`hunyuan_desc::load_layer_weights` decides device memory layout by the order it
calls `add_weight`: **up, gate, down, q, k, v, o**. Up and gate are interleaved
in `UP_GATE_OUTDIM_REORDEER = 512`-column slabs; everything else is one
`reorder_cpy(V=2)` slab, the lm_head included: it runs on the decode layer's own
bitstream and its weights stream through the same `_move_weights`, so it needs no
lm_head-specific reordering.

The host matmul reads those reordered bytes directly rather than keeping a
second, host-friendly copy — `cpu_func::_block_index` inverts the
reordering and `_q4k_dequant_chunk` unpacks a 32×256 q4_k chunk (4736 B: uint8
group scales and mins, column-major nibbles, bf16 S and negated M; see
`cpu_functions.hpp`). The lm_head is Q8_0 (`_q80_pair_dequant_chunk`). So host and
array consume identical memory, and moving a stage to the NPU changes nothing about
weight loading.

The checkpoint's `lm_head.weight` is padded to a whole 32-row block (120832
rows) but not to `vocab_size_padded` (121856), so the buffer is zeroed at
allocation and the tail bands dequantize to zero.

`UP_GATE_OUTDIM_REORDEER` (512) is the row-interleave slab and is a different
quantity from `GLU_SLICE` (1024) in
`FLM_Xclbin/generic_decoding_layer/models/hunyuan-dense-1.8b.h`, the decode-side
chunk `kernels/glu.cc` consumes. The relation is `slab = GLU_SLICE / 2` — one
`GLU_SLICE` of glu input is one up slab followed by one gate slab — and every
engine on this overlay uses the same pair (`nanbeige_npu.cpp:189` computes its
chunk as `1024/2` rows). So the two numbers agree as they stand.

## Backends

`hunyuan_prefill.hpp` defines four switches, all now defaulting to 1.

| macro | covers | default |
| --- | --- | --- |
| `HUNYUAN_MM_ON_NPU` | q/k/v/o, up+gate, down, via `hunyuan_overlay::gen_dq_run` | **1** |
| `HUNYUAN_ATTN_ON_NPU` | prefill attention, via `hunyuan_overlay::gen_attn_run` | **1** |
| `HUNYUAN_LM_HEAD_ON_NPU` | final norm + lm_head, via `gen_lm_head_seq` | **1** |
| `HUNYUAN_DECODE_ON_NPU` | the fused decode layer, `gen_layer_seq` | **1** |

Consequences of the default configuration, stated plainly:

* Exactly two xclbins are registered: `xclbins/<model_name>/layer.xclbin`
  (decode *and* lm_head) and `xclbins/<model_name>/prefill_overlay.xclbin` (was `fused_prefill.xclbin`), the latter
  registered **once** and handed to both `gemm_app_manager` and
  `mha_app_manager`. Separate mm/attn bitstreams would force a reload 32× per
  prefill; the fused overlay exists to avoid that.
* Every buffer is a real device BO (`npu_xclbin_manager::create_bo_buffer` needs
  no registered xclbin), so every `sync_to_device`/`sync_from_device` is legal
  and stays where it belongs regardless of which stage runs where.
* Building with `-DHUNYUAN_DECODE_ON_NPU=0` restores the host decode — the
  prefill graph run at `L = 1`. That is the golden reference to diff the array
  path against; the kv cache layout is the array's in both cases, so the two
  are directly comparable.
* All configurations are kept compiling, and each prefill switch can be flipped
  independently: `attention_on_host` stays compiled in as the fallback for
  `HUNYUAN_ATTN_ON_NPU=0` *and* for the runtime `mha_engine_fits` case below,
  and the `after_mix.sync_from_device()` inside the NPU attention branch is
  guarded on `!HUNYUAN_MM_ON_NPU` so a host `o_proj` still sees valid data.

### Prefill on the array

`fused_prefill.xclbin` is one bitstream holding two designs. `gen_dequant_mm_512`
drives the mm half on physical columns 2-5 (shims `IT2..IT5`), tiled
`desired_m=32, desired_k=512, desired_n=128, half_lanes=4` → `unit_m=128`,
`unit_n=512`. `gen_mha_engine_seq` drives the attention half on columns 0,1,6,7
(`IT0, IT1, IT6, IT7`), 2 CUs × 8 cores, `lc=16` rows per core per round.

Both halves need **L padded to 128** — `4*32` for the mm, `16*8` for attention —
which is why `hunyuan_npu_sequence::pad_prefill_L` exists and why all three
padding sites in `hunyuan_prefill.hpp` go through it rather than the old
`(L+511)/512*512`. Padded rows are safe: `hunyuan_prefill_context::forward`
memsets the whole padded `x`, rows past `L` are row-independent through the
layer, cache rows past `L_begin + L` are zero, and only `L*D` is ever read back.

#### The up/gate matmul, and where the silu runs

`glu_slab = 0` leaves y in physical interleaved-slab order — up and gate
alternating in `UP_GATE_OUTDIM_REORDEER = 512`-column slabs, which is both how
`load_layer_weights` packed the matrix and how the host GLU walks it.

The silu **is** on the array, but getting it there needs a trick.
`dequant_mm_e.cc` / `dequant_mm_o.cc` read `rtp_activate[0]` once, after the k
accumulation, and apply silu (1) or gelu (2) to the core's own output tile. That
RTP is a single value for a whole sequence, so a run cannot silu the gate slabs
and leave the up slabs alone. Rewriting the RTP mid-sequence does not work
either: the host stream runs ahead of the cores by design and has no way to know
a core has already read it.

The way out is that **an up/gate slab is exactly one mm unit** (`unit_n =
desired_n * half_lanes = 512 = UP_GATE_OUTDIM_REORDEER`), so up is the even Dout
tiles and gate the odd ones. `gen_dequant_mm_512` therefore takes
`(n_stride, n_first)` and emits only every `n_stride`-th tile, and the MLP issues
the matmul as two runs over the same `upgate` buffer:

| run | tiles | `rtp_act` | writes |
| --- | --- | --- | --- |
| `up_proj` | `t = 0, 2, 4, …` | 0 | the up slabs, linear |
| `gate_proj` | `t = 1, 3, 5, …` | 1 | the gate slabs, already silu'd |

The two cover disjoint columns and together are exactly the matrix a single
unstrided run would have produced. The host is then left with an elementwise
product, `cpu_func::_glu_mul`, instead of `_exp_glu` — one AVX-512 `exp` per
element per layer removed. `_exp_glu` stays as the `HUNYUAN_MM_ON_NPU=0` path.

Cost: one extra kernel launch per layer and a pipeline drain at the seam. Note
the array computes silu in bf16 and rounds before the multiply while `_exp_glu`
keeps the expression in fp32, so the two paths are close but not bit-identical.

The mm/attn RTP addresses in `hunyuan_npu_sequence.cpp` are allocator-assigned
and were re-read from the built overlay, not copied — the `aie-opt
--aie-assign-buffer-addresses` command that recovers them is in a comment next
to the constants. They are **different** from the decode-side `l_qk_address` /
`l_kv_address`.

`gen_mha_engine_seq`'s k/v descriptors round the newest query row of each round
up to a whole 128-row block. When `L_begin` is 128-aligned that round-up is
exact — the last round's `kv_end` is precisely `L_begin + L_wanted`, so a
prefill ending flush with the cache still fits on the array. Only an unaligned
`L_begin` overshoots, and then by at most 127 rows. Those rows are causally
masked and zero, so reading them is harmless — but only while they are inside
the buffer. `mha_engine_fits()` recomputes the generator's own `kv_end` for the
final round and checks `kv_end <= MAX_L`; the call site falls back to
`attention_on_host` rather than throwing.

### Decode on the array

`gen_layer_seq` targets the model-independent `generic_decoding_layer`
floorplan; every tile id and RTP address in `hunyuan_npu_sequence.hpp` is read
off the built overlay's `xclbins/rtp_addresses.txt` and is **not** inherited
from another model (the earlier stub's values came from `qwen3_5vl`, which has
2 kv heads, and were wrong). The proj and rms cores hold an `rtp_available`
lock (id 6, initialised to 0) that the host must release with a write to
`0x1F000 + 16*6` after setting their RTPs; the attention cores have none.

`_move_weights` takes the chunk size from the weight type: a 32×256 q4_k chunk is
4736 B (it was 4608 B with q4_0 before 2026-11-12); the lm_head moves as pairs of
padded Q8_0 halves (9472 B).

### lm_head on the array

There is **no `lm_head.xclbin`**. The overlay's 16 proj cores plus the rms core
do double duty: writing `0` instead of `1` to their `is_attn` RTP turns them
into a final-RMSNorm + `[vocab_size_padded × D]` matvec. `gen_lm_head_seq`
therefore rebinds the same bitstream's arguments — `arg0 = logits`,
`arg1 = lm_head weights`, `arg2 = x`, `arg3 = model.norm weight` — where the
layer sequence uses `arg0 = x`, `arg1 = proj weights`, `arg2 = rms weights`,
`arg3 = rope/qk-norm weights`, `arg4 = kv cache`. `llama_npu` and
`nanbeige_npu` do exactly the same; the model's xclbin folder needs no second
kernel.

Two consequences for the host:

* The **array** performs the final norm, so `get_logits` must hand it the raw
  hidden state. The `#else` (host) branch is the one that calls
  `cpu_func::_rms_norm` first. Getting this backwards double-norms.
* `model.norm.weight` lives in its own `final_rms_weight` BO rather than in the
  upper half of `x`, because it is a separate kernel argument.

With decode and lm_head both on the array, `forward(int)` never moves the hidden
state back to the host: the layer runlist leaves it in `x` on the device and the
lm_head run reads it from there.

### KV cache layout

The array reads and writes the cache in a **four-way split**, and the host now
writes the same thing:

| quadrant | base | contents |
| --- | --- | --- |
| K03 | 0 | K, kv heads 0..1, `[MAX_L × DK/2]` |
| K47 | `size/4` | K, kv heads 2..3 |
| V03 | `size/2` | V, kv heads 0..1 |
| V47 | `3*size/4` | V, kv heads 2..3 |

The contiguous `[L × DK]` layout cannot be fed to the array instead: a strided
BD would need a wrap of `L` on the sequence dimension, and
`npu_instr_utils.hpp` caps wrap fields at `wrap_bits = 10` (1024) while `L`
reaches 4096. So the split layout is canonical. `fill_kv_cache` writes the four
quadrants, host prefill attention runs as two half-head calls (which is why
`cpu_func::_attention` takes an optional `qo_stride`), and
`get_k_cache`/`get_v_cache` reassemble a logical row from two quadrants.

#### Why the ported prefill attention reads this layout unchanged

This is the least obvious fact in the prefill port, so it is worth stating
outright. `gen_mha_engine_seq` is `gen_seq_1_4` from
`FLM_Xclbin/Hy_MT2_1.8B/fused_prefill/host/seq_gen.hpp`, whose cache addressing
is

```
K at  chunk*MAX_L*DH*Nd + head_in_chunk*DH + row*DH*Nd
V at  the same + KV_CACHE_SIZE/2
```

where `Nd` is the generator's "kv heads per chunk". The reference `host.cpp`
passes `Nd = 4` — one chunk, a contiguous `[MAX_L × DK] | [MAX_L × DV]`. The
DLL instead passes **`Nd = desc->num_kv_heads / 2` = 2**, and that single
substitution turns the same arithmetic into the four-way split above:

* chunk stride `MAX_L*DH*2 = MAX_L*DK/2 = kv_cache_size/4` = `K47_offset`
* kv heads 0,1 → chunk 0 (K03); kv heads 2,3 → chunk 1 (K47)
* `KV_CACHE_SIZE/2 = V03_offset`, hence `V47 = V03 + stride = 3*size/4`
* q head `h` → kv head `h/4`, matching `num_kv_groups = 4` and the host
  `cpu_func::_attention` half-split

So there is no cache reshuffle, no staging copy, and `fill_kv_cache` is
untouched — the DLL's layout stayed fixed and the ported generator was adapted
to it. `MAX_L` and `KV_CACHE_SIZE` must come from the engine
(`this->MAX_L`, `this->kv_cache_size`), not from the window being computed, or
the chunk stride stops being the real quadrant stride.

The weight side needed no adaptation at all: `hunyuan_desc::reorder_cpy(V=2)`
and the reference's free `reorder_cpy(dst, src, Din, 2)` are the same
permutation over the same 32×256 q4_0 blocks, and `UP_GATE_OUTDIM_REORDEER`
equals the reference's `GLU_SLAB` (both 512). The DLL applies `reorder_cpy` per
512-row up/gate slice rather than once globally, but the destination formula
`(b/V)*V*blocks_per_row + c*V + (b%V)` only mixes within a band pair and a
512-row slice is 16 whole bands, so the concatenation is byte-identical and
`gen_dequant_mm_512`'s absolute `w_byte_off` still lands correctly.

There is no `Dequant` module and no `mha` module in this engine: it uses the
fused dequant+matmul, so `build_hunyuan.sh` builds only
`q4nx gemm lm_head hunyuan_npu`.

## What is verified

* The reorder inversion and the q4_0 chunk decode are checked by a round-trip
  test (reorder → invert → compare raw blocks) plus one-hot matmuls covering
  both plain and sliced (up/gate) addressing.
* All macro configurations compile clean — `{MM, ATTN} × {0,1}` plus
  `-DHUNYUAN_DECODE_ON_NPU=0`.
* The prefill overlay itself is checked by its own test
  (`FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay`, `run.exe`), against this engine's
  `cpu_func::_dequant_mm` and `_attention`.
* The prefill RTP addresses were re-read from the built MLIR with `aie-opt
  --aie-assign-buffer-addresses`, not inherited.
* **Not** verified end to end from the DLL side: no logits have been compared
  against a reference yet. The model at `/path/to/models/Hy-MT2-1.8B-NPU2`
  is intact (the `config.json` corruption noted here previously is gone), so the
  remaining work is a golden diff of `-DHUNYUAN_MM_ON_NPU=0
  -DHUNYUAN_ATTN_ON_NPU=0` against both-1, plus the same for
  `-DHUNYUAN_DECODE_ON_NPU=0/1` — the latter also proving the kv cache the NPU
  prefill wrote is in the layout the decode overlay reads.

## Known limitations

Recorded deliberately, not bugs to chase here:

* (Solved by the prefill overlay, 2026-11-03: dequant_mm now gets all 8 columns.)
* The attention kernel runs at ~7.5% of MAC peak — per
  `FLM_Xclbin/Hy_MT2_1.8B/README.md` §2, 68% of core time is softmax
  bookkeeping. Overlay-side problem.
* RoPE + QK norm, the GLU's final multiply, both RMS norms and `fill_kv_cache`
  remain on the host during prefill. (The GLU's silu does run on the array — see
  *The up/gate matmul, and where the silu runs*.)

## Long context (2026-11-03)

The engine used to pin the context at 1024 tokens (`FORCED_MAX_L`). Now:

* **Context = what FastFlowLM asks for** (`--ctx-len`, or the catalog default). The KV cache
  gets `kv_cache_rows(MAX_L)` rows: whole 128-row prefill grains plus two spare grains. The
  prefill attention reads one 128-row block past the newest query row (`mha_engine_fits`);
  the spare rows keep that read inside the buffer, so the host-attention fallback never
  triggers. Cost: 256 rows x 2 KB x 32 layers = 16 MB.
* **`prefill()` splits by length:** `L <= 512` -> `_prefill_short` (the original code: one
  token at a time up to 8 tokens, else one batched call). `L > 512` -> `_prefill_long`:
  512-token pieces through `_prefill_batched`, lm_head on the last piece only. 512 rows
  already reach the batched path's top rate (1233 tok/s at 512 vs 1219 at 1004 in one call),
  and the prefill scratch never grows past 512 rows -- the scratch size was the reason for the
  old 1K cap. Long-prompt optimizations go into `_prefill_long` without touching short prompts.
* **`update_max_length()`** grows the cache (copying the four quadrants) instead of refusing;
  it also calls `invalidate_mha()` because the cached prefill attention sequence holds the
  old quadrant offsets.
* **`fill_kv_cache` flushes only the rows it wrote** (`hunyuan_flush_range`, same pattern as
  `gemma4e_dflash_bo.hpp`). It used to `sync_from_device()` + `sync_to_device()` the whole
  per-layer cache for every prefill call: with a 64K context that is 128 MB x 2 per layer per
  512-token piece, which cost ~17% of prefill time at every length. The host only writes those
  rows and never reads the cache here, so no invalidate is needed.
* **Reference build:** `make HOST_ATTN=1 BUILD_DIR=../../build/hunyuan_npu_ref` puts prefill
  and decode attention on the host in fp32 (everything else unchanged), for
  `FastFlowLM/src/test/hunyuan_npu/test_long` + `compare_logits.py`.
* Still open: `clear_context()` zeroes the whole cache (2 GB at 32K) on every clear; the
  catalog entry is labelled `single-turn`, which makes FastFlowLM force the 1K default.

## Decode: lm_head in the token's runlist (2026-11-03)

Each decode slot has two runlists: `run` (RTP run + 32 layers) and `run_logits` (the same +
the lm_head). `forward()` uses `run_logits` when the token's logits are read, so a decode
token is one submission instead of two; prompt tokens pushed through the decode path without
logits still use `run`. Preemption mode keeps the separate `lm_head_run`. Measured: decode
49.85 -> 50.18 tok/s, logits bit-identical. Small because decode is memory bound -- see
`FLM_Xclbin/Hy_MT2_1.8B/decoding/README.md` (all bytes move at ~51 GB/s; the lm_head is 139 MB
= 2.55 ms of every token).

## Prefill overlay (2026-11-03)

Prefill now runs on `prefill_overlay.xclbin` (`FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay`, Peano)
instead of `fused_prefill.xclbin`: dequant_mm and attention take turns on the whole array, so
dequant_mm gets all 8 columns instead of 4. `fused_prefill.xclbin` is no longer registered.

* `hunyuan_overlay.hpp` -- the switch and the two run generators (`gen_dq_run`, `gen_attn_run`).
  The overlay's own test includes this file too, so there is one copy.
* `hunyuan_overlay_configs.hpp` -- generated by the overlay's `make regs` (each op's
  configuration as register writes), inside `namespace hunyuan_overlay`: flm loads every
  engine `.so` into one process and `-Bsymbolic-functions` covers functions, not data, so
  same-named tables in another engine (Hy-Memory has one) could otherwise interpose.
* `hunyuan_prefill.hpp` -- per layer: q, kv (k and v weights are adjacent: one run, rows are
  k | v), attention, o, up|gate (one run: the left half computes the up slabs, the right half
  the gate slabs with the silu on chip), down. `overlay_role` keeps one app per starting
  state of the array (unknown / dq / attention), so each run's switch is built in once.
  `hunyuan_overlay_state::configured` tracks the array; `overlay_lost()` (decode `forward()`,
  `get_logits()`) resets it, because the NPU reloads the overlay after `layer.xclbin` ran.
* **Trap found while wiring it: flush host writes into a BO the NPU writes later.** Rope runs
  in place on the k half of `kv`. The old code flushed `k` afterwards though the NPU never
  reads it; leaving that out let dirty cache lines be written back over the next layer's
  NPU kv output, at random -- kv was fine in some layers and cos 0.46 in others, and the
  translations came out empty. `kv.sync_to_device()` after the rope. (Since WP1 the rope runs on
  the cache rows instead, so the host never writes `kv`; see "Prefill syncs".)
* `make HOST_MM=1` (with `HOST_ATTN=1`): the prefill matmuls on the host too -- the all-host
  golden for bisecting.

Measured (Strix Halo, `test_long` / `bench_prefill`): prefill 12 tokens 121 -> 84 ms, 512
415 -> 265 ms, 4K 4.4 -> 2.6 s, 32K 72.7 -> 58.7 s; translations 0 empty, 21/130 small wording
changes (the attention rounding fix). Against the host-attention reference: 16K cos 0.966 ->
0.995, 32K 0.631 -> 0.984 (the conv_even fix in the overlay's attention).

## bfp16 KV copy (2026-11-04, design B steps 2-4)

Decode attention multiplies with `aie::mmul<8,8,8,bf16>`, which under
`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` converts both operands to bfp16 (one exponent per 8
values along the summed dimension, truncated) on every call. Storing K and V already in that
format is bit-identical (checked: `FLM_Xclbin/Hy_MT2_1.8B/decoding/README.md`) and 44% smaller
(9 bytes per 8 values). Decode attention reads **K and V** from this copy (steps 3 and 4).

* Where: in each layer's kv-cache BO, after the four bf16 quadrants (the layer kernel has only 5
  BO arguments): `hunyuan_npu_sequence::Kb03/Kb47/Vb03/Vb47_offset` (bf16 elements), one block per
  attention column (kv heads 0,1 and 2,3), each `[chunk][head (2)][2304 B]`. K: rows of a key,
  blocks along dh (`_bfp16_tile_group`). V: V^T tiles, blocks along 8 keys (`_bfp16_vt_tile_group`).
  Packers copied from `qwen3_5vl_flash` into `hunyuan_bfp16.hpp`. `kv_buffer_size` is the whole BO.
* Only **complete 16-key chunks** are in it. A decode step at context length L (L - 1 cached rows +
  the new one) reads `v_bfp16_chunks(L) = (L - 1) / 16` chunks of K and V as bfp16 through shims 3/4
  (K channel 0, V channel 1), and the chunk holding the new row as bf16 through shim 5 (channel 0 ->
  column 3, channel 1 -> column 4; its K rows, then its V rows). Both attention cores get the count
  in their `RTP_L[1]` (`l_qk_address + 4`, `l_kv_address + 4`); the RTP buffers are at 3328 / 50176.
* `_update_bfp16(len)` converts the chunks completed since the last call: after each decode
  token, and at the start of `forward()` (normally a no-op; it is what guarantees the chunks the
  sequence asks for exist). A batched prefill call does it per layer with `_update_bfp16_layer`,
  while that layer's attention runs (WP1, "Prefill syncs"). It invalidates exactly the K and V rows
  it reads (the array wrote them) and flushes exactly the bytes it writes.
* `clear_context` -> 0 chunks (the memset clears the copy too); `restore` -> at most the restored
  length; `update_max_length` -> copy the bf16 quadrants (sized from the old `kv_cache_size`, not
  the BO), then rebuild the copy.
* `HY_BFP16_CHECK=1` re-packs every converted chunk after each update and compares byte for byte.
* Step 2 alone (copy kept, not read, separate BOs): decode +0.1%; a 1K prefill +20-25 ms. Steps 3-4
  numbers: `FLM_Xclbin/Hy_MT2_1.8B/decoding/README.md`.

## Prefill profile (2026-11-05, WP0)

`make PREFILL_PROFILE=1 BUILD_DIR=../../build/hunyuan_npu_prof` prints `[prefill-prof] <step>: <time>` per layer.
WP0 split the old coarse steps so each NPU run, each sync and each CPU step has its own line: `attn:q_run`,
`attn:kv_start` / `attn:kv_wait` (the kv run overlaps the CPU's q rope), `attn:{q,kv}_sync_{from,to}`,
`attn:{q,k}_rope_norm`, `attn:o_run` / `attn:o_sync_from`, `mlp:upgate_run`, `mlp:upgate_sync_from`, `mlp:glu_mul`,
`mlp:hid_sync_to`, `mlp:down_run`, `mlp:down_sync_from`, `layer:x_memset`, `layer:x_sync_to`, `layer:residual_add`
and `prefill:update_bfp16`. A normal build compiles them to nothing. An NPU run's line includes the overlay switch in
front of it. `FastFlowLM/src/test/hunyuan_npu/prof_summary.py` sums a `test_long` log; the Krackan tables are in
`FLM_Xclbin/Hy_MT2_1.8B/README.md`, "Krackan baseline (WP0)".

## Prefill syncs (2026-11-05, WP1)

Host-side clean-up of the batched prefill (`hunyuan_prefill.hpp`, `_prefill_batched`). Logits, decode
steps and the bfp16 copy are **byte-identical** to before at 9..24K tokens (`test_long --dump`, two
length orders, `HY_BFP16_CHECK=1`).

* **Ranged syncs.** `q`, `kv`, `x`, `upgate`, `hid` are flushed/invalidated only for the L real rows
  (`hunyuan_flush_range` / `hunyuan_invalidate_range`), not the padded BO. Every row is a multiple of
  64 B, so a range is always whole cache lines.
* **Fewer syncs.** Per layer 11 whole-BO syncs became 9 ranged ones (+ the 4 ranged cache flushes):
  * `hid` is never written by the NPU, so its invalidate is gone. It is flushed whole once, right
    after its memset at allocation (else those dirty lines would be written back some time later).
  * The q and kv runs read their own input `x_attn_in` instead of `x`. Its padding rows must be
    zero, so the per-layer memset + whole flush of `x` is gone: `x_attn_in` is zeroed once at
    allocation, and when a call has fewer rows than the last one, only rows `[L, old L)` are zeroed
    and flushed (once per call). `x` stays the output of o/down and the input of up|gate, as before.
  * K's rope + key norm run on the K rows in the cache (after `fill_kv_cache` copies them raw),
    not in `kv`. The host then never writes `kv`, so `kv` needs no flush before the next layer's
    kv run writes it (the dirty-line trap above).
* **Trap: the q/kv padding rows must stay zero.** A first try just dropped the memset (padding rows
  then held old o/down output). Every prompt that starts at 0 stayed byte-identical, but the third
  512-row piece of a 1500-token prompt (476 real rows, L_begin 1024, padding rows holding piece 2's
  values) changed: prefill cos 0.99956 vs before. So with L_begin > 0 the padding rows of the q/kv
  input reach real rows somewhere. Keeping them zero fixes it. Cause (found in WP12): the prefill
  attention's `calculate_y` (overlay `kernel/attn/attn_kernel.h`) skips the rescale of y only when
  all 8 rows of a group have c == 1, and its rescale is not exact for c == 1. A padding row whose
  running max moves (non-zero q) makes the real rows of its 8-row group get rescaled. This needs
  several KV blocks (L_begin > 0), which is why prompts from 0 never showed it. See "Rows that
  change other rows" in the overlay README.
* **bfp16 copy behind the attention.** `_update_bfp16_layer(i, first, target)` runs between the start
  and the wait of layer i's attention run. The attention never touches the bfp16 part of the cache BO,
  and the rows it reads are already flushed, so the invalidate/flush there do not race with it.
* **mv/batched crossover** (`HUNYUAN_MV_PREFILL_MAX_TOKENS`): mv costs ~15.3 ms per token + 5 ms,
  batched ~92 ms flat (from an empty context; ~100 ms on top of a 32-token pinned prefix). The
  batched path is also closer to the all-CPU reference at 6..8 tokens (cos 0.998 vs 0.988-0.991).
  `HY_MV_PREFILL_MAX=<n>` overrides it at run time. **It stayed 8 in WP1** (user's decision; 3 since WP11): in the
  translations test 6 gives e2e 245.7 -> 240.8 ms per turn, but 30/130 wording diffs vs the golden
  (gate 21/130), partly because a turn's output depends on the previous turn (next bullet).
* **Open: a turn depends on the previous turn.** With crossover 6, turns that take the mv path in
  both builds (4-6 new tokens) changed too, right after a turn that changed path, although the
  pinned prefix is restored in between. Something the previous turn left (likely KV rows past the
  restored length) is read. Every build is deterministic run to run, so it is not noise. Not
  investigated.
* Host helpers (`cpu_functions.hpp`) on 6 threads instead of 4 (Krackan: 6 cores / 12 threads).

Krackan, `bench_prefill` median of 3 alternated runs, ms (baseline -> WP1):
12 tok 96.7 -> 91.5, 36: 100.7 -> 93.5, 132: 167.1 -> 157.5, 504: 315.9 -> 297.9, 1004: 648.2 -> 612.6.

## Small-row plans (2026-11-05, WP11)

Short batched prompts used to pad every matmul to 128 rows (4 core rows x 32 tokens), ~69 of
~92 ms of a 12-token prompt. Now the five matmuls of a layer (q, kv, o, up|gate, down) run on
**32 rows for <= 32 tokens and 64 rows for 33..64 tokens**, on two more overlay configurations
(`OP_DQ32`, `OP_DQ64`; how they work: `FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay/README.md`,
"Small-row plans"). The results are **byte-identical** to the 128-row plan.

* `hunyuan_npu_sequence::pad_prefill_gemm_L(L, small_rows)` picks the matmul rows `Lg`; only
  `_prefill_short` passes `small_rows` (via `_prefill_batched`). `_prefill_long` is unchanged:
  all its 512-token pieces, also a short last one, keep the 128-row plan. `HY_SMALL_ROWS=0` in
  the environment turns the small plans off (pads to 128 again), for A/B and debugging.
* The attention keeps its own padding: 128 rows (16 rows x 8 cores per round), and every
  prefill buffer keeps `pad_prefill_L` rows as before. The attention reads all 128 q rows, so
  **q rows `[Lg, 128)` must be zero**, as the 128-row plan leaves them (it computes them from
  `x_attn_in`'s zero rows; see the padding-row trap in "Prefill syncs"). `q_rows` tracks which
  rows may hold values; a call that needs it zeroes and flushes `[Lg, q_rows)` once.
* The kv matmul (1024 columns, not a multiple of the 2048 a split round covers) also writes
  scratch rows `[Lg, 2Lg)` of `kv`; nothing reads them.
* A switch now depends on whose mem tile BDs are loaded (the three dq plans write the same BD
  registers with different words), so `hunyuan_overlay_state` keeps `bd_owner` next to
  `configured`, `ran(op)` updates both, and `bd_owner` is part of every cached sequence's key.
  A call that stays on one plan switches dq <-> attention as cheaply as before.

Checked (Krackan): `test_long --dump` at 9..128 and 1044 tokens (long-then-short orders) and the
G4 lengths 512..24K are byte-identical to the WP1 build; translations byte-identical (0 empty,
21/130). `bench_prefill`, median of 3 alternated runs, ms (WP1 -> WP11):

| tokens | 12 | 20 | 36 | 68 | 132 | 260 | 504 | 1004 |
|---|---|---|---|---|---|---|---|---|
| WP1 | 91.2 | 92.7 | 94.3 | 96.7 | 158.9 | 223.2 | 297.6 | 610.7 |
| WP11 | 55.4 | 55.0 | 68.8 | 97.0 | 156.6 | 222.8 | 294.9 | 605.4 |

Profile (12 tokens): GEMMs 72.5 -> 37.0 ms; what is left is mostly a fixed ~115 us per matmul
run (5 runs per layer), attention 11 ms (128 rows), lm_head 5 ms.

**mv/batched crossover, re-measured** (default **3** since 2026-11-05, was 8; `HY_MV_PREFILL_MAX`): from an empty
context the batched path is ~54 ms flat and mv costs 35.6 / 50.4 / 64.7 ms at 2 / 3 / 4 tokens,
so batched wins from 4 tokens. In the translations test (32-token pinned prefix) a batched turn
costs ~62-64 ms, mv ~68 ms at 4 new tokens and ~83 at 5. Batched is also closer to the all-CPU
reference at 2..8 tokens (prefill cos 0.990-0.999 vs mv 0.970-0.993). Translations test per
crossover (prefill / e2e ms per turn, wording diffs vs golden): 8: 97.8 / 242.2, 21/130; 6: 80.5 /
226.3, 30/130; 5: 68.7 / 216.3, 31/130; 4: 64.2 / 212.2, 41/130; 3: 62.8 / 211.1, 44/130. The
diff count partly measures turn-to-turn history (open, WP12). The user chose 3: fastest, and the
changed lines are rewordings of similar quality. The translations test is now judged against a
new golden made from this build.

## Stale KV rows after restore (2026-11-05, WP12)

* **What:** `restore()` lowers the context length to the checkpoint but leaves the K/V rows the
  later turns wrote. The attention kernels read past the last real row: the prefill attention's
  P.V converts V into bfp16 blocks of 8 keys (one shared exponent), and decode reads a 16-key bf16
  tail chunk and packs it the same way. P is 0 for those keys, but the shared exponent still
  changes the real rows. So a translation turn depended on the turn before it. Why exactly: see
  "Rows that change other rows" in `FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay/README.md`.
  `clear_context()` already zeroes the whole cache for this reason.
* **Fix, on by default since 2026-11-06 (user's decision; `HY_ZERO_STALE_KV=0` turns it off):**
  `restore()` zeroes rows `[checkpoint, kv_rows_used)` of
  the four bf16 quadrants of every layer and flush them (`_zero_stale_kv`). `kv_rows_used` is the
  highest context length since the last clear (`set_context_length` keeps it). The bfp16 copy needs
  nothing: `restore()` already lowers `bfp16_chunks`, and chunks past it are rebuilt before use.
  It was opt-in in WP12 because it changes the translations text (12/130 vs the x3 golden); the
  user turned it on and the translations test now uses a golden made with it.
* **Numbers (Krackan, 130 translations, crossover 3, `--order` of the test harness):** fix off, the
  same lines in reverse order give 14/130 different texts (shuffled orders: 15 lines vary); fix on,
  0 lines differ over forward, reverse and 2 shuffled orders (the only exception was a turn hit by
  a run failure before the run-again fix, below). Fix on vs the x3 golden: 12/130 rewordings,
  the same 12 in every run. Cost: prefill per turn 63.23 -> 63.53 ms, e2e 212.27 -> 212.50 ms
  (median of 3 alternated runs, turns with a run failure left out), i.e. within noise.
* `test_long` clears the context before every prompt, so its dumps are byte-identical with the fix
  on or off.

## Run trace (2026-11-05, WP12)

`make RUN_TRACE=1 BUILD_DIR=../../build/hunyuan_npu_trace_wp12` builds an engine that records the
submit (`S`) and the end of the wait (`D`, with the returned `ert_cmd_state`, 4 = completed) of every
NPU run: q, kv, mha, o, upgate, down per layer, the prefill pieces, the lm_head and the decode
runlist. The events go into a memory-mapped ring file (`HY_RUN_TRACE=<path>`, default
`/tmp/hy_run_trace.<pid>`, 8192 lines of 80 bytes), so the run a stuck process waits in can be read
from outside while it hangs: `sort -n <file> | tail`. In a normal build the macros
(`hunyuan_run_trace.hpp`) compile to the plain calls; dumps of the trace build are byte-identical.

## Run failures (2026-11-05, WP12)

* **What happens:** now and then a prefill overlay run never finishes (WP12, Krackan test machine: 9 in ~1.9M
  overlay runs; about one per 6 runs of the 130-turn translation test, one per ~540K runs in long
  prompts). The driver ends it after ~6-8 s with state 8 (`ERT_CMD_STATE_TIMEOUT`) and resets the
  array. All failing runs seen were dequant_mm runs (`down`, `upgate`, `q`, `o`); the ones with a
  trace had no switch in front. Short translation turns fail as well as long prompts.
* **Why it looked like a hang:** the engine ignored the returned state and kept `configured`, but
  after the reset the array no longer holds the overlay configuration the host wrote. So every
  later run (partial or no switch) hung too, each until its own 6.1 s timeout: ~20 min per
  512-token piece, garbage out, until another xclbin ran (the lm_head) and the next run did a
  full switch again. That is the "hang" of WP0/WP11 and the 628 s translation turn of WP12.
* **Fix (on by default; normal runs take the same path, results byte-identical):**
  `hunyuan_overlay_state::check()` looks at the state of every overlay run (q, kv, mha, o, up|gate,
  down). If it is not "completed", it prints a warning, forgets the configuration
  (`configured = bd_owner = -1`) and runs the op again from a full switch; a second failure throws.
  A failure then costs ~6-8 s (the timeout; the run again takes ~5 ms) and the output is still
  right. The decode runlist and the lm_head runs are not run again: a state other than
  "completed" throws (`flm::npu::wait_for_completion`), since their output is consumed at once.
* **Test:** the trace build has `HY_FAKE_FAIL_AT=<n,...>` (treat the n-th checked run as failed):
  9 forced failures over all six run kinds, dumps byte-identical to the baseline. Natural failures
  in the translation test: text identical to the runs without a failure.
* **Not found:** why the first run fails. See the WP12 report for the samples and the open questions.

## x as bfp16 tiles (2026-11-06, WP2)

The prefill matmuls on the overlay now read their input x as **bfp16 tiles made by the host**
(`hunyuan_bfp16::_bfp16_rne_tile_group`, layout `[L / 16][Din / 8][2][72 B]`) instead of bf16. The
kernel used to convert every bf16 x block to bfp16 inside the core before each MMUL; the host
blocks are exactly those blocks, so **every result is byte-identical** to before. Why and how the
blocks were matched: `FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay/README.md`, "x as bfp16 tiles".

Who writes what (NPU matmul build; the host matmul build keeps the bf16 buffers as before):

| matmul | its x | written by |
|---|---|---|
| q, kv | `xa_tiles` | the input RMSNorm, straight as tiles |
| o | `am_tiles` | after the attention: invalidate rows [0, L) of `after_mix`, pack (`attn:o_x_tiles`) |
| up+gate | `xm_tiles` | the post-attention RMSNorm, straight as tiles |
| down | `hid_tiles` | the GLU multiply, straight as tiles |

* **Same bf16 math.** The norm and the GLU were split into per-column helpers
  (`_rms_inv` + `_rms_norm_cols`, `_glu_mul_cols` in `cpu_functions.hpp`) that `_rms_norm` /
  `_glu_mul` use too, so the bf16 values packed are the ones the bf16 path stored.
* **`hunyuan_write_x_tiles`** cuts the work into pieces: one 16-row group, cut into the fewest
  column pieces (a multiple of 256 columns) that give each of the 6 threads about 4 pieces. A thread
  makes its piece's rows in a small scratch and packs them while they are in cache. So a 12-token
  prompt still uses every thread, and 8 groups (128 tokens) do not leave threads idle in a second
  round. A norm piece computes the 1 / rms of its rows itself (again for each column piece of a
  row: ~5 us per layer at 128 tokens, cheaper than one more parallel pass). Rows of the last group
  past L are packed as zero.
* **The packer** (`_bfp16_rne_tile_group`) works in 16-bit lanes, 32 values (4 blocks of one row)
  per step. Trick 1: a block of 8 values is exactly one 128-bit lane, so the block's largest exponent
  needs only in-lane shuffles (swap 64-bit halves, swap 32-bit pairs, rotate by 16 bits). Trick 2: a
  bf16 mantissa has only 8 bits (with the implicit one), so "24-bit mantissa >> (exponent gap + 17)"
  is "8-bit mantissa >> (gap + 1)" and fits 16-bit lanes; round to nearest even is
  `(m + 2^gap - 1 + odd bit) >> (gap + 1)`. Byte-identical to the first (32-bit lane) version on 25 M
  mixed values; 2.96 -> 3.82 G values/s on one thread (16.2 on 6).
* **Padding rows.** `xa_tiles` rows past the prompt must stay zero (the q padding rows rule, "Prefill
  syncs" and "Small-row plans"): groups are written whole, and when a call has fewer groups than the
  last one, the extra groups are zeroed and flushed once (`x_attn_in_rows` now counts rows of whole
  groups). The other tile buffers' padding rows only reach padding rows of the output.
* **Syncs.** One ranged flush per tile buffer per matmul, over whole groups (Din * 18 bytes, a
  multiple of 64: whole cache lines). The host no longer writes `x`, `x_attn_in` or `hid` in the NPU
  build, so `x` is only written by the NPU and read back by the host (no dirty-line risk), and the
  `layer:x_sync_to` / `mlp:hid_sync_to` flushes are gone. New: the invalidate of `after_mix`.
* Profile ticks: the two norms and `mlp:glu_mul` now include packing + flushing;
  `attn:o_x_tiles` is the read-back + packing of the attention output.

Checked (Krackan, A = the build before): `test_long --dump` byte-identical at 512..24K (30 dumps),
9..128 + 1044 tokens (65), and the WP12 mixed-length list incl. multi-piece prompts with a short last
piece (L_begin > 0, 105 dumps); translations 0 empty, 0/130 vs the x3z golden, text identical to A.

Krackan, A -> B, medians of alternated runs (`bench_prefill` 5 pairs, `test_long` 3 pairs), ms:

| tokens | 12 | 20 | 36 | 68 | 132 | 260 | 504 | 1004 | 4096 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 54.7 | 55.3 | 68.7 | 96.2 | 156.8 | 222.8 | 295.1 | 607.0 | 2919 | 20075 |
| B | 54.8 | 55.5 | 63.9 | 85.5 | 131.8 | 184.4 | 247.2 | 510.0 | 2532 | 18664 |

12 and 20 tokens are equal within run-to-run spread: the 32-row matmuls are mostly fixed cost per
run (~1.7 ms less per prompt) and the host packing costs about as much there. Profile (ms per prompt,
A -> B): matmuls 37.3 -> 35.7 at 12 tokens, 73.7 -> 59.5 at 128, 224 -> 168 at 512, 1781 -> 1332
at 4K; the host side (norms, o input, glu, syncs) 1.7 -> 2.2, 8.4 -> 10.3, 28.9 -> 35.0, 232 -> 285
(4K: the norms and the glu now pack, +61 ms; the attention output is read back and packed, +32 ms;
syncs -41 ms). Translations: prefill per turn 64.0 -> 63.1-63.8 ms.

## CMake MVPREFILL default (2026-11-10)

The Makefile builds with `-DMVPREFILL=0`, but `CMakeLists.txt` had `option(MVPREFILL ... ON)`, and CI
builds the shipped Windows/HRX libraries with CMake. With MVPREFILL on, `_prefill_short` sends every
prompt of up to 512 tokens through the decode overlay one token at a time (~15-20 ms per token), while
`_prefill_long` (> 512) ignores the flag. A tester saw 512 tokens -> 10.5 s, 513 -> 0.47 s.

Reproduced on Krackan without a rebuild via `HY_MV_PREFILL_MAX=512` (same path): `bench_prefill`
504 tokens 7,789 ms vs 362 ms normal; 1004 tokens 525 ms either way. The CMake default is now OFF, so
CMake and Makefile builds take the same path. Short prompts in the earlier CI builds also took the
token-by-token path, so their text can differ slightly from the batched path's.

## Q4_K weights (2026-11-12)

The body weights are Q4_K (32 x 256 chunks of 4736 B) and the lm_head Q8_0, read by every
backend from the same reordered buffer: the converter (FLM_Q4NX_Converter `hunyuan_dense`),
the host readers (`_q4k_dequant_chunk`, `_q80_pair_dequant_chunk`), the decode layer
(`FLM_Xclbin/Hy_MT2_1.8B/decoding`) and the prefill overlay (`gen_dq_run` streams 4736 B
chunks as two 2368 B halves; the kernel side is in the overlay README, "Q4_K weights").

Removed with the switch (all Q4_0, no callers left): `gen_dequant_mm_512` and
`gen_mha_engine_seq` (the `fused_prefill.xclbin` generators; `mha_engine_fits` stays, it checks
`gen_attn_run`'s kv read), and the folders `FLM_Xclbin/Hy_MT2_1.8B/fused_prefill` and
`FLM_Xclbin/Hy_MT2_1.8B/dequant_mm`.

Checked on Krackan (`test_long --decode 3`, prefill logits vs the all-host engine
`HOST_ATTN=1 HOST_MM=1`, host lm_head):

| prompt | prefill cos | top-1 | decode cos (3 steps, min) |
|---|---|---|---|
| 100 tokens | 0.99770 | same | 0.99338 |
| 400 tokens | 0.99949 | same | 0.99499 |
| 2000 tokens (wiki_states_4k) | 0.99958 | same | 0.99496 |
| 4000 tokens | 0.99924 | same | 0.99589 |

At 100 tokens most of the gap is the prefill matmuls (NPU vs `HOST_MM=1`: 0.99786; `HOST_MM=1`
vs all-host: 0.99901), i.e. the bfp16 x and w of the array's mmul, as with Q4_0 (earlier notes:
batched prefill vs all-host 0.990-0.999 at short prompts).

Translations test (130 turns): 0 empty outputs (its golden file is gone, so no
wording diff). **Prefill speed end to end is not measured yet**: during this work two other jobs
held all 12 CPU cores, and `bench_prefill` gave ~1.2 s at 12 tokens (~55 ms on a quiet machine),
i.e. it measured the busy host, not the array. From the overlay unit test, dequant_mm is ~10-15%
slower on the big shapes, so expect prefill ~8-12% slower from ~500 tokens and about the same for
short prompts. Re-run `bench_prefill` on a quiet machine.

## Short tails after a long prompt (2026-11-13)

**Bug.** FastFlowLM cuts a prompt into chunks of `max_prefill_len` (4096 for Hy-MT2) and calls
`prefill()` once per chunk. A 4129-token page with 32 cached tokens became 4096 + 1. The 1-token
chunk is short, so `_prefill_short` sent it through the mv path: the decode kernel ran the last
prompt token, and the first reply token's logits came from decode instead of the batched prefill.
On that page the reply then began with a made-up sentence (由于中文文本由系统控制…) instead of the
translation.

First-token probabilities on that page (Krackan, Q4_K weights):

| path for the 1-token tail | 不同 (right) | 由于 (wrong) |
|---|---|---|
| bf16 on CPU (transformers) | 0.65 | 0.05 |
| batched prefill (`HY_MV_PREFILL_MAX=0`) | 0.40 | 0.04 |
| mv path (old default) | 0.15 | 0.21 |

Tails of 2-3 tokens shifted the same way; tails of 4+ tokens and single-chunk prompts were never
affected. The mv path was already known to be less exact than batched at short context (see
"mv/batched crossover, re-measured"); after ~4K cached tokens the gap is large enough to flip
greedy decoding.

**Fix.** The mv path is only taken when the whole context after the call stays within
`HUNYUAN_SHORT_PREFILL_MAX_TOKENS` (512) -- the short prompts it was tuned for. A 1-3 token tail
on top of a longer context goes batched: one ~50-60 ms call instead of 15-45 ms, next to a
multi-second prefill. Chunked and unchunked prompts now take the same path for their last token.
`HY_MV_PREFILL_MAX` still sets the token crossover; `MVPREFILL` builds are unchanged.

## Files

| file | what |
| --- | --- |
| `hunyuan_model_def.hpp` | dims, weight descriptors, device layout, loaders |
| `cpu_functions.hpp` | host kernels + the q4_k / Q8_0 readers |
| `hunyuan_bfp16.hpp` | bfp16 packers: the decode KV copy, the prefill matmul input x (WP2) |
| `hunyuan_prefill.hpp` | per-block prefill contexts, backend switches |
| `hunyuan_npu_sequence.{hpp,cpp}` | instruction generation for the decode overlay (decode layer, lm_head) |
| `hunyuan_overlay.hpp`, `hunyuan_overlay_configs.hpp` | the prefill overlay: switch + run generators, generated op configurations |
| `hunyuan_npu_detail.hpp`, `hunyuan_npu.cpp` | `Impl` and the public forwarders |
| `hunyuan_run_trace.hpp` | run trace for hang hunting (`make RUN_TRACE=1`), compiles to nothing otherwise |
