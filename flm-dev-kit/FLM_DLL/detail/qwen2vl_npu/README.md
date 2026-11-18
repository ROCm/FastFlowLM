<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# qwen2vl_npu

Engine for the Qwen2-VL / Qwen2.5-VL family (text decoder + vision tower).

**Status: decode, lm_head and prefill all run on the array.**
Decode uses one fused `layer.xclbin`; prefill uses the three generic overlays
(`mm.xclbin`, `dequant.xclbin`, `attn.xclbin`).

## Layout

| File | Contents |
| --- | --- |
| `qwen2vl_model_def.hpp` | `qwen2vl_desc` — dimensions, weight descriptors, buffer sizes, weight loading |
| `qwen2vl_prefill.hpp` | `qwen2vl_prefill_context` and its three block contexts |
| `qwen2vl_cpu_functions.hpp` | host AVX-512 kernels (`_rms_norm`, `_rope`, `_residual_add`, `_elementwise_mul`) |
| `qwen2vl_npu_detail.hpp` / `qwen2vl_npu.cpp` | `qwen2vl_npu::Impl` — decode path, lm_head, kv cache, checkpointing |
| `qwen2vl_npu_sequence.{hpp,cpp}` | decode / lm_head / mha sequence generation and the kv cache layout |
| `qwen2vl_image.{hpp,cpp}` | `Qwen2VLImageEncoder` — the vision tower, untouched by the refactor |

## Config → `qwen2vl_desc`

| config.json | `qwen2vl_desc` |
| --- | --- |
| `hidden_size` | `D` |
| `head_dim` | `DH` |
| `num_attention_heads * head_dim` | `DQ` |
| `num_key_value_heads * head_dim` | `DK`, `DV` |
| `intermediate_size` (rounded up to 512) | `intermediate_size` |

`inv_freq` is a single 64-entry table (θ = 1e6), shared by every checkpoint in
the family. `MAX_L` is rounded up to a power of two by `Impl`. The text half of
`qwen2vl_desc` is identical to `qwen2_desc`; only the position encoding and the
embedding step differ.

## Weight layout

Qwen2 is the odd one out in this family: attention carries **q/k/v biases**.
Those live in the per-layer `rms_weights` bf16 buffer, not in `proj_weights`:

```
proj_weights: [ q | k | v | o | up/gate interleaved | down ]          (flm_q41)
rms_weights:  [ input_norm | post_attn_norm | q_bias | k_bias | v_bias | scratch ]
```

`qwen2vl_layer_weight_def::all()` returns the descriptors in exactly that order —
**do not reorder it**, the dequant sequences index `proj_weights` by absolute
byte offset and the gemms index the biases by
`get_{q,k,v}_bias_elem_offset()`. `RMS_SCRATCH_ELEMS` tail padding is what the
array kernel scribbles into; it must stay at the end.

All projections are `flm_q41` (4-bit + zero-point = exactly 5/8 byte per
element), so the `weight_container` offsets are bit-identical to the hardcoded
`* 5 / 8` arithmetic they replaced.

Up and gate are interleaved one `UP_GATE_OUTDIM_REORDER = 512`-column slice at
a time, so their individual `weight_container` offsets are *not* their physical
positions. `get_upgate_offset()` (= `ffn_up.offset`) is the start of the
combined band; dequant modes 1 and 2 pull out the two halves.
`get_down_offset()` is a real offset.

Each weight is copied through `qwen2vl_desc::reorder_cpy`, which interleaves rows
in groups of `vertical_blocks` (2) so each array column reads a contiguous
descriptor.

## Prefill

`qwen2vl_prefill_context` owns the three xclbins, the `Gemm`/`Dequant` sequence
generators, and every scratch buffer. `Impl` only hands it weights and a kv
cache.

```
setup(L_in, L_begin) -> qwen2vl_prefill_shape
  |
  +-- forward(layer_idx, shape, proj_weights, rms_weights, kv_cache)   x N layers
        |
        +-- dequant_block.start()  ||  input rms norm on host
        +-- dequant_block.wait()
        +-- attn_block.forward()   -> residual add
        +-- post-attention rms norm
        +-- mlp.forward()          -> residual add
```

**Row geometry.** The attention kernel consumes whole chunks, so the batch is
padded at both ends and every gemm runs over `L_padded` rows:

```
L_chunk         = L_CHUNK (128)
L_begin_chunked = (L_begin / L_chunk) * L_chunk
L_offset        = L_begin - L_begin_chunked          // padding rows in front
L_end_chunked   = L_begin_chunked + ceil_to(L_in + L_offset, L_MIN)   // L_MIN = 256
L_padded        = L_end_chunked - L_begin_chunked
```

Every `tensor_2d` view over an activation buffer carries `L_offset`, so `[0]` is
the first *new* token. The host `residual` buffer is reused across calls and
zeroed in `setup`, since the padding rows still take part in every gemm.

**Blocks.**

* `qwen2vl_dequant_prefill_context` — five apps (qkv is one fused dequant here,
  unlike llama) and five bf16 output buffers. With a runlist all five are
  enqueued and `wait()` blocks on the batch; under preemption the runlist is
  unavailable, so 0–3 are serialized and only the `down` run overlaps the
  caller's rms norm.
* `qwen2vl_attn_block_prefill_context` — q/k/v/o gemms plus the mha app. The
  gemms take the bias views as an extra argument. mRoPE on q overlaps the k
  gemm; mRoPE on k overlaps the v gemm. `fill_kv_cache` writes K at
  `L_begin * DK` and V at `V_offset + L_begin * DV` — one contiguous band each,
  not split like llama's.
* `qwen2vl_mlp_prefill_context` — SiLU-gate, up, host glu, down. The apps are
  *created* in the order gate, up, down; that order is what `npu_app_manager`
  uses to assign array resources, so it must match the pre-refactor code.

**Caching.** Sequences are regenerated only when the thing they encode changes:
`L_padded_old` for the gemms, `L_begin_chunked_old` / `L_end_chunked_old` for
the mha. Buffers grow but never shrink (`L_max`). `set_kv_layout` invalidates
the mha guard, since the cache layout is baked into that sequence —
`update_max_length` calls it after `V_offset` moves.

## Gotchas

* The `sync_from_device(); sync_to_device();` pairs around the dequantized
  weight buffers, before each gemm, are load-bearing. Dropping them corrupts
  the weights the gemm reads. Carried over verbatim from the pre-refactor code;
  the underlying sync is not understood.
* `k_proj` writes to a device BO, not a host buffer.
* `_rms_norm` here uses **eps 1e-6** (llama uses 1e-5). Do not unify them.
* `up_weights` is allocated from the `dequant_gate` app rather than
  `dequant_up`. That is a typo in the original, preserved because the app a
  buffer is allocated from determines which memory bank it lands in.
* `tensor_2d` caches a `buffer<T>*` and its row views, so it must be re-`assign`ed
  after any reallocation or any change to `L_offset`. `setup` does this
  unconditionally.
* The `_prefill_with_mv` fallback (`-DMVPREFILL`) still walks the decode path one
  token at a time and does not touch `qwen2vl_prefill_context`.

## Refactor record (2026-09-02)

Brought this engine onto the `model_def` + modular-prefill pattern used by
`llama_npu` and `qwen3_5vl_npu`. Behaviour-preserving; `qwen2vl_npu.cpp` went
930 → 375 lines.

Extracted from `qwen2vl_npu.cpp` / `qwen2vl_npu_detail.hpp`:

1. **`qwen2vl_desc`** — dimension derivation, per-layer `weight_desc_t`
   descriptors with `weight_container`-assigned offsets, the aggregated buffer
   sizes, `reorder_cpy`, and both weight loaders. Every hardcoded `* 5 / 8` and
   every hand-computed dequant / bias offset is now derived.
2. **`qwen2vl_prefill_context` + blocks** — replaces the monolithic
   `_prefill_with_mm`, which allocated ten BOs and regenerated eight sequences
   on *every* call. Those are now cached behind shape guards.
3. **`qwen2vl_cpu_func`** — the four host kernels, formerly `Impl` members;
   `_rope` takes `DH` and `inv_freq` as arguments instead of reading `this`,
   and drops the unused `rms_weight` parameter of the old `_rope_rms`.

One intentional fix: the old code sized `total_proj_weights` from the
*unpadded* `intermediate_size` while writing the *padded* amount — a latent
overflow. `qwen2vl_desc` uses the padded value everywhere.

Verified by rebuilding `libqwen2vl_npu.so` (`make` in this directory).

## What makes this engine a VLM

Everything below is the *only* difference from `qwen2_npu`; the weight layout,
the dequant band and the gemm shapes are identical.

**3-axis mRoPE.** Each token carries three position indices `(t, h, w)` instead
of one. `qwen2vl_cpu_func::_update_rope_values` splits the 64 frequency pairs as
`mrope_section = [16, 24, 24]` — the first 16 follow `t`, the next 24 `h`, the
last 24 `w` — and writes a `cos | sin` table that `_rope` then applies. The
split is a property of the checkpoint, not a tunable.

**Position bookkeeping.** `general_token_idx` is the mRoPE clock and runs
*ahead* of `current_context_length`: a run of image tokens occupying `grid_h/2 *
grid_w/2` rows advances it by only `grid_w/2`. `Impl::get_input_idx` walks the
token stream, marks the `image_soft_token_id` (151655) rows in `vision_mask`,
emits the three indices per token and advances the clock. Because of this,
`set_context_length` takes the new clock value as a second argument, every
caller must supply it, and `checkpoint`/`restore` save and restore it alongside
the context length. The public `qwen2vl_npu::set_context_length` is a no-op that
prints a warning — the clock cannot be reconstructed from `L` alone.

**Embedding.** Rows flagged in `vision_mask` are memcpy'd from
`Qwen2VLImageEncoder::encode(payload)` instead of going through the embedding
table. The encoder runs once per prefill call, before the layer loop.

### Sync ordering

The VL prefill has three device round trips the text engine does not, all
carried over verbatim because they are load-bearing (see Gotchas):

* an extra `q.sync_to_device()` after the v gemm completes;
* `up_weights.sync_to_device()` between the gate and up gemms;
* `o.sync_from_device(); o.sync_to_device();` before the down gemm, in place of
  the text engine's single `o.sync_to_device()` after the attention residual
  add.

## Refactor record (2026-09-02) — VL notes

Same three extractions as `qwen2_npu`; `qwen2vl_npu.cpp` went 990 → 449 lines.
`get_input_idx` moved out of the header into the `.cpp`, and the
`L_padded > MAX_L` guard moved into `qwen2vl_prefill_context::setup`, which is
why the context now carries `MAX_L` (`set_max_length`).
