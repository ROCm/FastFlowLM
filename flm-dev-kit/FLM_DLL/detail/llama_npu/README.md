<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# llama_npu

Engine for the Llama 3.x family (1B / 3B / 8B), selected by `hidden_size`.

**Status: decode, lm_head and prefill all run on the array.**
Decode uses one fused `layer.xclbin`; prefill uses the three generic overlays
(`mm.xclbin`, `dequant.xclbin`, `attn.xclbin`).

## Layout

| File | Contents |
| --- | --- |
| `llama_model_def.hpp` | `llama_desc` — dimensions, weight descriptors, buffer sizes, weight loading |
| `llama_prefill.hpp` | `llama_prefill_context` and its three block contexts |
| `llama_cpu_functions.hpp` | host AVX-512 kernels (`_rms_norm`, `_rope`, `_residual_add`, `_elementwise_mul`) |
| `llama_npu_detail.hpp` / `llama_npu.cpp` | `llama_npu::Impl` — decode path, lm_head, kv cache, checkpointing |
| `llama_npu_sequence*.{hpp,cpp}` | decode / lm_head sequence generation and the kv cache layout |

## Config → `llama_desc`

| config.json | `llama_desc` | 1B | 3B | 8B |
| --- | --- | --- | --- | --- |
| `hidden_size` | `D` | 2048 | 3072 | 4096 |
| `head_dim` | `DH` | 64 | 128 | 128 |
| `num_attention_heads * head_dim` | `DQ` | 2048 | 3072 | 4096 |
| `num_key_value_heads * head_dim` | `DK`, `DV` | 512 | 1024 | 1024 |
| `intermediate_size` | `intermediate_size` | 8192 | 8192 | 14336 |

`hidden_size` alone picks three things, and they are not independent knobs —
each is a property of the checkpoint family:

| `D` | `inv_freq` table | `mha_type` | `min_max_length` |
| --- | --- | --- | --- |
| 2048 | `inv_freq_1b` (32 entries) | `mha_d64_q4` | 8192 |
| 3072 | `inv_freq_3b_8b` (64 entries) | `mha_d128_q3` | 4096 |
| 4096 | `inv_freq_3b_8b` (64 entries) | `mha_d128_q4` | 4096 |

Anything else throws from `llama_desc::build`. `MAX_L` is then raised to
`min_max_length` and rounded up to a power of two by `Impl`.

## Weight layout

All projections are `flm_q41` (4-bit + zero-point = exactly 5/8 byte per
element), so the `weight_container` offsets are bit-identical to the hardcoded
`* 5 / 8` arithmetic they replaced. One `proj_weights` buffer per layer holds,
in this order:

```
[ q | k | v | o | up/gate interleaved | down ]
```

`llama_layer_weight_def::all()` returns the descriptors in exactly that order —
**do not reorder it**, the dequant sequences index the buffer by absolute byte
offset.

Up and gate are interleaved one `UP_GATE_OUTDIM_REORDER = 512`-column slice at
a time, so their individual `weight_container` offsets are *not* their physical
positions. `get_upgate_offset()` (= `ffn_up.offset`) is the start of the
combined band and is what the dequant kernel wants; modes 1 and 2 tell it which
half to pull out. `get_down_offset()` is a real offset.

Each weight is copied through `llama_desc::reorder_cpy`, which interleaves rows
in groups of `vertical_blocks` (2) so each array column reads a contiguous
descriptor. It is dtype-generic: the block size comes from
`get_quantization_byte_size(32 * 256, dtype)`.

## Prefill

`llama_prefill_context` owns the three xclbins, the `Gemm`/`Dequant`/`MHA`
sequence generators, and every scratch buffer. `Impl` only hands it weights and
a kv cache.

```
setup(L_in, L_begin) -> llama_prefill_shape
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

* `llama_dequant_prefill_context` — four apps and four bf16 output buffers. With
  a runlist all four are enqueued and `wait()` blocks on the batch; under
  preemption the runlist is unavailable, so 0–2 are serialized and only the
  `down` run overlaps the caller's rms norm.
* `llama_attn_block_prefill_context` — q/k/v/o gemms plus the mha app, and the
  four kv-cache band offsets. RoPE on q overlaps the k gemm; RoPE on k overlaps
  the v gemm. `fill_kv_cache` splits each row in half (`03` / `47`) because the
  cache is striped across two core pairs.
* `llama_mlp_prefill_context` — up, SiLU-gate, host glu, down.

**Caching.** Sequences are regenerated only when the thing they encode changes:
`L_padded_old` for the gemms, `L_begin_chunked_old` / `L_end_chunked_old` for
the mha. Buffers grow but never shrink (`L_max`). `set_kv_layout` and
`set_max_length` both invalidate the mha guard, since the cache layout and
`MAX_L` are baked into that sequence — `update_max_length` calls both.

## Gotchas

* The `sync_from_device(); sync_to_device();` pairs around the dequantized
  weight buffers, before each gemm, are load-bearing. Dropping them corrupts
  the weights the gemm reads. They are carried over verbatim from the
  pre-refactor code and the underlying sync is not understood.
* `k_proj` writes to a device BO, not a host buffer — a host buffer there
  misbehaves (same note as before the refactor).
* `tensor_2d` caches a `buffer<T>*` and its row views, so it must be re-`assign`ed
  after any reallocation or any change to `L_offset`. `setup` does this
  unconditionally.
* The `_prefill_with_mv` fallback (`-DMVPREFILL`) still walks the decode path one
  token at a time and does not touch `llama_prefill_context`.

## Refactor record (2026-09-02)

Brought this engine onto the `model_def` + modular-prefill pattern already used
by `qwen3_5vl_npu`. Behaviour-preserving; `llama_npu.cpp` went 895 → 340 lines.

Extracted from `llama_npu.cpp` / `llama_npu_detail.hpp`:

1. **`llama_desc`** — dimension derivation, the hidden-size dispatch, per-layer
   `weight_desc_t` descriptors with `weight_container`-assigned offsets, the
   aggregated buffer sizes, `reorder_cpy`, and both weight loaders. Every
   hardcoded `* 5 / 8` and every hand-computed dequant offset is now derived.
2. **`llama_prefill_context` + blocks** — replaces the monolithic
   `_prefill_with_mm`, which allocated ten BOs and regenerated eight sequences
   on *every* call. Those are now cached behind shape guards.
3. **`llama_cpu_func`** — the four host kernels, formerly `Impl` members;
   `_rope` takes `DH` and `inv_freq` as arguments instead of reading `this`.

Preserved deliberately: NPU call order, the dequant / rms-norm overlap and its
separate preemption path, the sync round trips above, the `L_MIN = 256` chunk
math, and the up/gate 512-column interleave. One intentional change: `residual`
is reused across calls and explicitly zeroed, where the old code relied on a
fresh (uninitialized) allocation each time.

Verified by rebuilding `libllama_npu.so` (`make` in this directory).
