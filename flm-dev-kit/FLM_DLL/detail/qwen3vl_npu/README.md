<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# qwen3vl_npu

Engine for the Qwen3-VL family — a Qwen3 decoder plus a vision tower, selected
by `intermediate_size`.

**Status: decode, lm_head and prefill all run on the array.**
Decode uses one fused `layer.xclbin`; prefill uses the three generic overlays
(`mm.xclbin`, `dequant.xclbin`, `attn.xclbin`). The vision tower has its own
apps and lives in `qwen3vl_image.*` / `rot_pos_emb.*` / `norm.*`.

## Layout

| File | Contents |
| --- | --- |
| `qwen3vl_model_def.hpp` | `qwen3vl_desc` — dimensions, weight descriptors, buffer sizes, weight loading |
| `qwen3vl_prefill.hpp` | `qwen3vl_prefill_context` and its three block contexts |
| `qwen3vl_cpu_functions.hpp` | host AVX-512 kernels (`_rms_norm`, `_rope_rms`, `_update_rope_values`, `_residual_add`, `_elementwise_mul`) |
| `qwen3vl_npu_detail.hpp` / `qwen3vl_npu.cpp` | `qwen3vl_npu::Impl` — decode path, lm_head, kv cache, mrope bookkeeping, checkpointing |
| `qwen3vl_npu_sequence.{hpp,cpp}` | decode / lm_head sequence generation and the kv cache layout |
| `qwen3vl_image.*`, `qwen3_vision_prefill_helper.cpp`, `rot_pos_emb.*`, `norm.*`, `seq_gen.hpp` | the vision tower |

## Config → `qwen3vl_desc`

| config.json | `qwen3vl_desc` |
| --- | --- |
| `hidden_size` | `D` |
| `head_dim` | `DH` |
| `num_attention_heads * head_dim` | `DQ` |
| `num_key_value_heads * head_dim` | `DK`, `DV` |
| `intermediate_size` | `intermediate_size` |

`intermediate_size` is what distinguishes the checkpoints, so it also picks the
attention kernel and the context floor:

| `intermediate_size` | `mha_type` | `min_max_length` |
| --- | --- | --- |
| 3072, 6144 | `mha_d128_q2` | 8192 |
| 9728, 12288 | `mha_d128_q4` | 4096 |

Anything else throws from `qwen3vl_desc::build`. `MAX_L` is then raised to
`min_max_length` and rounded up to a power of two by `Impl`.

## Weight layout

All projections are `flm_q41` (4-bit + zero-point = exactly 5/8 byte per
element), so the `weight_container` offsets are bit-identical to the hardcoded
`* 5 / 8` arithmetic they replaced. One `proj_weights` buffer per layer holds,
in this order:

```
[ q | k | v | o | up/gate interleaved | down ]
```

`qwen3vl_layer_weight_def::all()` returns the descriptors in exactly that order —
**do not reorder it**, the dequant sequences index the buffer by absolute byte
offset.

> Note: `qwen3_npu`, the text-only sibling, packs the same band **K | V | Q** so
> its decode layer can stream the KV half to the RoPE tile first. This engine's
> decode sequence and prefill gemm offsets both assume **Q | K | V**, so the two
> layouts are deliberately different. `get_qkv_offset()` returns the q offset
> here and the k offset there.

Up and gate are interleaved one 512-output-column slice at a time
(`UP_GATE_OUTDIM_REORDER`), so their individual `weight_container` offsets are
*not* their physical positions. `get_upgate_offset()` (= `ffn_up.offset`) is the
start of the combined band and is what the dequant kernel wants; modes 1 and 2
tell it which half to pull out. `get_down_offset()` is a real offset.

`rope_rms_weights` is one buffer of `3 * DH` bf16 per layer: the rope cos/sin
table, then `q_norm`, then `k_norm`. The table is rewritten by
`_set_rope_rms_weights` on every `set_context_length`; the two norms are loaded
once. Prefill reads rows `[1]` and `[2]` of that buffer and computes its own
cos/sin on the host.

Each quantized weight is copied through `qwen3vl_desc::reorder_cpy`, which
interleaves rows in groups of `vertical_blocks` (2) so each array column reads a
contiguous descriptor. It is dtype-generic: the block size comes from
`get_quantization_byte_size(32 * 256, dtype)`.

## Prefill

`qwen3vl_prefill_context` owns the three xclbins, the `Gemm`/`Dequant`/`MHA`
sequence generators, and every scratch buffer. `Impl` only hands it weights, a
kv cache and the mrope indices.

```
setup(L_in, L_begin) -> qwen3vl_prefill_shape
  |
  +-- forward(layer_idx, shape, input_idx, proj_w, rms_w, rope_rms_w, kv_cache)  x N layers
        |
        +-- dequant_block.start()  ||  input rms norm on host
        +-- dequant_block.wait()
        +-- attn_block.forward()   -> residual add
        +-- post-attention rms norm
        +-- mlp.forward()          -> residual add
        |
        +-- (in Impl, layers 0-2 only) deepstack vision residual add
```

**Row geometry.** The attention kernel consumes whole chunks, so the batch is
padded at both ends and every gemm runs over `L_padded` rows:

```
L_chunk         = mha->get_chunk_size()
L_begin_chunked = (L_begin / L_chunk) * L_chunk
L_offset        = L_begin - L_begin_chunked          // padding rows in front
L_end_chunked   = L_begin_chunked + ceil_to(L_in + L_offset, L_MIN)   // L_MIN = 256
L_padded        = L_end_chunked - L_begin_chunked
```

Every `tensor_2d` view over an activation buffer carries `L_offset`, so `[0]` is
the first *new* token. The host `residual` buffer is reused across calls and
zeroed in `setup`, since the padding rows still take part in every gemm.

**Blocks.**

* `qwen3vl_dequant_prefill_context` — five apps and five bf16 output buffers.
  With a runlist all five are enqueued and `wait()` blocks on the batch; under
  preemption the runlist is unavailable, so the first four are serialized and
  only the `down` run overlaps the caller's rms norm.
* `qwen3vl_attn_block_prefill_context` — q/k/v/o gemms plus the mha app, the
  four kv-cache band offsets, and the `rope_values` mrope scratch. The k gemm
  overlaps the host rotation of q; the v gemm overlaps the rotation of k.
  `fill_kv_cache` splits each row in half (`03` / `47`) because the cache is
  striped across two core pairs.
* `qwen3vl_mlp_prefill_context` — SiLU-gate, up, host glu, down.

**Caching.** Sequences are regenerated only when the thing they encode changes:
`L_padded_old` for the gemms, `L_begin_chunked_old` / `L_end_chunked_old` for
the mha. Buffers grow but never shrink (`L_max`). `set_kv_layout` and
`set_max_length` both invalidate the mha guard, since the cache layout and
`MAX_L` are baked into that sequence — `update_max_length` calls both.

## What makes this engine a VLM

* **mRoPE.** Each token carries three positional clocks — temporal, row,
  column — packed as `input_idx[i*3 + 0..2]`. `_update_rope_values` *interleaves*
  them across the first 60 frequency bins, three bins at a time; bins 60 and up
  all use the temporal clock. (This is a different scheme from `qwen2vl_npu`,
  which splits the bins into contiguous `[16, 24, 24]` sections.)
* **`general_token_idx`.** The temporal clock, advanced by `get_input_idx` as it
  walks the prompt: a text token costs 1, and an image as a whole costs one grid
  width (`grid_w / 2`). It is part of the model state — `checkpoint` /`restore`
  save and restore it alongside the context length, and the public
  `set_context_length(int)` is a warning no-op because the clock cannot be
  recovered from `L` alone.
* **Embedding splice.** Rows whose token is `image_soft_token_id` (151655) take
  their embedding from the vision encoder instead of the token table.
* **Deepstack.** `Qwen3VLImageEncoder::encode` returns four embeddings per
  patch: `[3]` is the input embedding, and `[0..2]` are added into the vision
  rows of the residual stream after decoder layers 0, 1 and 2 respectively.
  This add happens in `Impl::_prefill_with_mm`, not in the prefill context,
  since it is model state the blocks know nothing about.

## Gotchas

* The `sync_from_device(); sync_to_device();` pair on `common_bufs->o` before the
  down gemm is load-bearing. Dropping it corrupts the weights the gemm reads.
  It is carried over verbatim from the pre-refactor code and the underlying sync
  is not understood. Note this engine does *not* have `qwen3_npu`'s extra
  `gate_weights` / `up_weights` / `down_weights` syncs — the difference is
  preserved deliberately.
* The extra `this->q.sync_to_device()` after the checked V-projection wait is
  likewise carried over verbatim; q was already synced before the checked
  K-projection wait.
* `tensor_2d` caches a `buffer<T>*` and its row views, so it must be re-`assign`ed
  after any reallocation or any change to `L_offset`. `setup` does this
  unconditionally.
* `image_embedding[3]` is dereferenced whenever `payload != nullptr`, even if
  `is_vlm` is false — a non-null payload on a text-only config is undefined.
  Pre-existing; unchanged by the refactor.
* The `_prefill_with_mv` fallback (`-DMVPREFILL`) still walks the decode path one
  token at a time and does not touch `qwen3vl_prefill_context`, so it ignores the
  payload entirely.

## Refactor record (2026-09-02)

Brought this engine onto the `model_def` + modular-prefill pattern already used
by `qwen3_5vl_npu`, `llama_npu` and `qwen3_npu`. Behaviour-preserving;
`qwen3vl_npu.cpp` went 997 → 524 lines.

Extracted from `qwen3vl_npu.cpp` / `qwen3vl_npu_detail.hpp`:

1. **`qwen3vl_desc`** — dimension derivation, the intermediate-size dispatch,
   per-layer `weight_desc_t` descriptors with `weight_container`-assigned
   offsets, the aggregated buffer sizes, `reorder_cpy`, and both weight loaders.
   Every hardcoded `* 5 / 8` and every hand-computed dequant offset is now
   derived.
2. **`qwen3vl_prefill_context` + blocks** — replaces the monolithic
   `_prefill_with_mm`, which allocated ten BOs and regenerated eight sequences
   on *every* call. Those are now cached behind shape guards. `fill_kv_cache`
   moved into the attention block.
3. **`qwen3vl_cpu_func`** — the five host kernels, formerly `Impl` members;
   `_rope_rms` and `_update_rope_values` take `DH`, `inv_freq` and the
   `rope_values` scratch as arguments instead of reading `this`.

Preserved deliberately: the Q|K|V band order, NPU call order (including the
q-before-k rope overlap that differs from `qwen3_npu`), the dequant / rms-norm
overlap and its separate preemption path, the sync round trips above, the
`L_MIN = 256` chunk math, the up/gate 512-column interleave, and the deepstack
and `general_token_idx` bookkeeping. One intentional change: `residual` is
reused across calls and explicitly zeroed, where the old code relied on a fresh
(uninitialized) allocation each time.

Verified by rebuilding `libqwen3vl_npu.so` (`make` in this directory) and by
diffing the `Impl::` symbol list against `HEAD`: only `_rms_norm`, `_rope_rms`,
`_update_rope_values`, `_residual_add`, `_elementwise_mul` and `fill_kv_cache`
disappear — exactly the helpers relocated into the two new headers.
