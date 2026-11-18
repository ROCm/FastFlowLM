<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# gemma4-12b NPU backend — notes on the prefill path (attention focused)

Reading notes taken from `gemma4_12b_npu.cpp`, `gemma4_12b_prefill.hpp`,
`gemma4_12b_prefill_xref.hpp`, `gemma4_12b_npu_sequence.cpp` and
`gemma4_12b_npu_def.hpp`. Everything below describes what the code does today; where a
hook exists but has no implementation behind it, that is called out.

---

## 1. The model, in the shape this backend sees it

`gemma4_12b_desc` (in `gemma4_12b_npu_def.hpp`) parses `config.json` and owns the whole
device layout. The attention relevant facts:

* **Hybrid layer stack.** Layer `i` is a *global* (full attention) layer when
  `i % sliding_window_pattern == sliding_window_pattern - 1`, otherwise a *sliding
  window* (SWA) layer. `gemma4_12b_desc::layer_type()` is the only place this is decided;
  `Impl` caches the result in `layer_types[]`.
* **The two kinds disagree on almost everything.** Separate head dim (`DH` vs `SWA_DH`),
  separate kv head count (`num_global_key_value_heads` vs `num_key_value_heads`), separate
  RoPE theta and partial rotary factor, separate weight layouts. Hence `DQ/DK/DV` and
  `SWA_DQ/SWA_DK/SWA_DV` everywhere, and the `LAYER_SPECIFIC_DIM(type)` macro in the
  sequence generator.
* **`attention_k_eq_v`.** Global layers store **no `v_proj`** — V is the *raw* `k_proj`
  output, taken **before** `k_norm` and **before** RoPE. `DV == DK` by construction.
  `gemma4_12b_layer_weight_def::has_v_proj` is false for those layers and their packed
  weight buffer simply omits the V region.
* **No `1/sqrt(head_dim)` anywhere in the attention.** The scale is folded into
  `q_norm.weight` in the checkpoint (same as `SimpleGemma4Attention` upstream). Both the
  host attention kernel and the decode kernel therefore compute a bare `q·k`.
* **Norm flavours.** `q_norm` / `k_norm` are per-head RMS norms *with* weight; `value_norm`
  is a per-head RMS norm *without* a weight (`SimpleRMSNorm(with_scale=False)` →
  `_rms_norm_batch(..., w_ptr=nullptr, ...)`). Gemma-4 uses plain `x * rsqrt(...) * w`,
  **not** the `(1 + w)` of gemma-2/3.
* **Partial rotary.** `inv_freq[j] = theta^(-2j/head_dim)` for `j < RoPE_DIM/2`, and
  **zero** past that. A zero frequency is an identity rotation, so the full head can be
  roped blindly with no branch — this reproduces the checkpoint's `rope_freqs.weight`
  (1.0 on the kept lanes, 1e30 → 0 on the rest).
* **Per-layer output scale.** `layer_output_scale.weight` multiplies the whole layer
  output (residual included). On decode it is an RTP write; on prefill it is
  `cpu_func::_elementwise_scale_batch` at the end of `forward()`.

---

## 2. Two prefill paths

`Impl::prefill()` dispatches on a compile-time switch:

| build | path | what it does |
|---|---|---|
| `-DMVPREFILL` | `_prefill_with_mv()` | replays the **decode** graph one token at a time; correct but slow, and ignores the multimodal payload entirely |
| default | `_prefill_with_mm()` | the **batched** path: the whole prompt goes through the stack in **one pass** |

Everything below is the batched path.

### Backend selection (top of `gemma4_12b_prefill.hpp`)

Two stages can independently live on host or array:

```
make PREFILL_MM=cpu     -> -DFLM_PREFILL_MM_ON_CPU   -> GEMMA4_12B_MM_ON_NPU = 0
make PREFILL_ATTN=cpu   -> -DFLM_PREFILL_ATTN_ON_CPU -> GEMMA4_12B_ATTN_{GLOBAL,SWA}_ON_NPU = 0
```

Both **default to the NPU**. Note the polarity: the flags are opt-*out*, so a plain `make`
gives matmuls on `dequant_mm.xclbin` and **all 48 layers'** attention on the array.
With both set to `cpu` the prefill touches no xclbin at all and the scratch is plain host
memory, which is what makes a device-less CPU build runnable.

The attention switch is still expressed **per layer kind**, because the two kinds are two
separate overlays — separate bitstreams, separate RTP maps, separate floorplans — even
though `PREFILL_ATTN` currently moves both together:

| macro | layers | overlay | xclbin |
|---|---|---|---|
| `GEMMA4_12B_ATTN_GLOBAL_ON_NPU` | the 24 global (full attention) layers, `DH = 512`, 1 kv head | `attention_DH_512_prefill` | `attn_global.xclbin` |
| `GEMMA4_12B_ATTN_SWA_ON_NPU` | the 24 sliding-window layers, `SWA_DH = 256`, 8 kv heads | `attention_DH_256_prefill` | `attn_sliding.xclbin` |
| `GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU` | the §4.4 block rows of the global layers only | `attention_global_image` | `attn_global_image.xclbin` |
| `GEMMA4_12B_ATTN_ON_NPU` | either of the first two | — | — |

All three are host-fallback-clean: `_attend_global_npu()` / `_attend_sliding_npu()` return
`false` and the caller runs the host kernel instead, so setting one macro to 0 by hand is a
valid configuration and needs no other change.

Neither of the first two overlays knows what a vision block is, so the §4.4 fix-up is a
separate pass in every configuration. The third overlay is that pass, for the global layers:
`PREFILL_IMAGE_ATTN=npu` (the default, and only meaningful when `GEMMA4_12B_ATTN_GLOBAL_ON_NPU`
is 1) sends the block rows to `attn_global_image.xclbin`. It is registered **lazily**, from
`set_vision_blocks()` on the first prompt that carries an image, so a text-only session never
pays for a fourth hw context, and a missing bitstream is a one-line warning rather than a load
failure. The sliding layers keep the host kernel unconditionally — their per-row window makes
the block's key range row-dependent, which this overlay's single "keys stop at `L`" mask
cannot express. See §4.4.

`GEMMA4_12B_SCRATCH_ON_DEVICE = MM_ON_NPU || ATTN_ON_NPU` decides whether the activation
scratch is a device BO. `gemma4_12b_scratch_alloc/push/pull()` are no-ops on a pure host
build (`buffer<T>::sync_*` asserts on a non-BO buffer, so they cannot be called blindly).

Because the two stages toggle independently, the scratch allocator can no longer be
`mm_app_manager` — `PREFILL_MM=cpu PREFILL_ATTN=npu` leaves that null while the attention
still needs real BOs. `gemma4_12b_prefill_mm_context::bo_manager` is whichever manager
exists, and everything allocates from it. That is sound because **a BO belongs to the
`xrt::device`, not to a `hw_context`** (`create_bo_buffer` → `buffer<T>(*device, size)`), so
one allocation can be bound as an argument to runs on either bitstream.

---

## 3. Prefill structure

```
_prefill_with_mm(ids, payload)
  ├─ shape.update(L_begin, L_in)        # L_padded = round_up(L_in, 512)
  ├─ prefill_mm->reserve(shape)         # scratch + regenerate every mm control sequence
  ├─ prefill_mm->set_vision_blocks(...) # image/video runs, once per prompt
  ├─ embeddings (host)  + _embed_media() # projected media rows overwrite soft-token rows
  ├─ for l in layers: prefill_mm->forward(l, residual, ...)
  ├─ x <- residual[L_in - 1]  ; lm_head_run
  └─ set_context_length(L_begin + L_in) # hand the position over to the decode graph
```

`residual` is the only stage-to-stage tensor that stays a **plain host buffer** — no
projection ever touches it. Everything else lives in `prefill_mm`'s scratch.

One layer (`gemma4_12b_prefill_mm_context::forward`) is:

```
input_layernorm      (host)  -> common.x
attn_block.forward()                        # section 4
post_attention_norm  (host, in place on x)
residual += x
pre_feedforward_norm (host)  -> common.x
mlp_block.forward()                         # up/gate (2 NPU runs) -> glu_mul -> down
post_feedforward_norm(host, in place on x)
residual += x
residual *= layer_scale
```

Note the **four** norms per layer and the fact that the residual add happens *after* the
post-norm — the gemma sandwich-norm arrangement, not the llama one.

### Scratch ownership

* `gemma4_12b_mm_common_buffers` — `x [L,D]`, `upgate [L,2*IS]`, `hid [L,IS]`. Kind
  independent, so the biggest tensors exist **once** for the whole stack.
* `gemma4_12b_mm_attn_buffers` — `q/k/v/attn`, plus `swa_kv` (the unrolled sliding window,
  K then V, §4.2.2; allocated only for the sliding kind). **One instance per layer
  kind**, sized for that kind's `DQ/DK/DV`, so the global kind does not pay for the
  sliding kind being wider (or vice versa).

Both only ever grow (`reserve()` returns whether it reallocated).

---

## 4. The attention block — `gemma4_12b_mm_attn_block::forward()`

This is the interesting part.

### 4.1 q / k / v, and the overlap pattern

```
q = x·Wq                       (NPU, waited immediately)
k_run = start(k = x·Wk)        (NPU, in flight)
  rope_rms(q)                  (host, overlaps k_run)
  push(q)
k_run.wait()

if has_v_proj:                       # SWA layers
    v_run = start(v = x·Wv)
      rope_rms(k)                    (host, overlaps v_run)
      push(k); v_run.wait(); pull(v)
else:                                # global layers, attention_k_eq_v
    memcpy(v, k)                     # RAW k_proj output, pre-norm, pre-rope
    rope_rms(k); push(k)

rms_norm(v, w=nullptr)               # value_norm, per head, unweighted
push(v)
```

Two things worth remembering:

1. **Each host norm is issued against the projection that follows it**, not after all
   three matmuls. That is deliberate: the host stage runs while the array works on a
   tensor the host does not need yet.
2. **The `attention_k_eq_v` copy must happen before `k_norm`/rope.** The order in the code
   is the semantics, not a convenience — copy first, then rope `k` in place.

`cpu_func::_rope_rms_batch()` does the per-head RMS norm and the rotation in one pass, with
the rotation being the half-split form `(x1,x2) -> (x1 c - x2 s, x1 s + x2 c)` where `half
= DH/2`. Positions are absolute (`L_begin + row`), computed per row — the decode path's
precomputed cos/sin table in `rope_rms_weights` is **not** used by prefill.

### 4.2 The two attention flavours

`_attend_global()` — the cache is `[MAX_L * DK | MAX_L * DV]`, indexed by **absolute
position**, not a ring:

```
sync_from_device(cache)
k_cache[L_begin ..] = k ; v_cache[L_begin ..] = v      # append
sync_to_device(cache)
attention(attn, q, k_cache, v_cache,
          L_q = L, L_kv = L_end,
          q_pos_begin = L_begin, kv_pos_begin = 0, window = 0)
```

The V base is `cache.data() + cache.size()/2`, computed from what the layer *currently*
owns — never from a `MAX_L` captured at build time, which `update_max_length()` would
leave stale.

This is the flavour that runs on the array by default; see §4.2.1. The append and the two
cache syncs above are unchanged by that — only the `attention()` call is replaced.

`_attend_sliding()` — the cache is a ring of `SLIDING_LENGTH` rows indexed by
`pos % SLIDING_LENGTH`. Prefill cannot attend a ring directly, so it:

1. **Unrolls** the still-visible history `[max(0, L_begin - sliding + 1), L_begin)` from the
   ring into the linear `swa_kv` staging buffer;
2. appends the prompt's own rows to the unrolled window **and** writes them back into the
   ring, so the decode graph picks up where prefill left off;
3. attends over `[win_begin, L_end)` with `window = sliding`.

`swa_kv` is allocated from the prefill scratch — it must come from the same allocator as the
rest, because the host writes it and then pushes it; a plain host buffer would trip the BO
assert in `sync_to_device()` on an array build. See §4.2.2 for its shape and for the array
path that reads it.

### 4.2.1 The global attention on the array — `_attend_global_npu()`

Three buffers, in this run-argument order, and nothing else:

```
arg 0  attn  [L_padded][DQ] bf16   written by the array
arg 1  q     [L_padded][DQ] bf16   RoPE'd, normed, head-major within a row
arg 2  cache [MAX_L][DK] then [MAX_L][DV]   the layer's *decode* kv cache, verbatim
```

The last one is the point: **the decode cache is already in exactly the layout the kernel
wants** — one contiguous K half followed by one contiguous V half, indexed by absolute
position, V starting at `cache.size()/2`. No repacking, no staging buffer, no copy. That is
only true for the global kind — the sliding kind's ring is not, and §4.2.2 is mostly about
what it takes to make it look like one.

Everything else rides the control sequence, generated by
`gemma4_12b_npu_sequence::gen_attn_prefill_global()` (ported from
`attention_DH_512_prefill/host/seq_gen.hpp::gen_seq_1_4`). Four things it bakes in:

| | value | why it changes |
|---|---|---|
| `L_begin` | RTP, absolute position of query row 0 | new prompt position |
| `Q_chunks` | RTP, `L_attn / 128` | new prompt length |
| `window` | RTP, set to `MAX_L` | new cache capacity |
| the DMA offsets | derived from all of the above plus `cache.size()/2` | any of the above |

so the sequence is regenerated whenever any of the four changes, guarded by
`L_begin_seq / L_attn_seq / kv_rows_seq / kv_elems_seq`. That is **once per prefill per
layer kind**, shared by that kind's 24 layers — not once per layer. It deliberately does
*not* live in `reserve()`: `reserve()` never sees a kv cache, and the geometry of the cache
is half of what the sequence encodes.

Four things that are easy to get wrong here:

* **`MAX_L` is not `L_padded`.** The standalone testbench passes its own `L_padded` as the
  `MAX_L` argument, because its kv buffer is exactly that tall. In the DLL the cache is the
  real per-layer cache, so the value is `cache.size() / 2 / DK` — the row *capacity*. Get
  this wrong and the "no window" window clips real keys: the kernel computes
  `kv_begin = max(0, q_pos - window)`, so understating `window` silently drops the earliest
  part of the prefix from every late query row.
* **`L_attn`, not `L_padded`, is the query row count.** The matmuls need `L % 512 == 0`
  (`gen_dequant_mm_512`); the attention needs `L % 128 == 0` — one query round. So
  `gemma4_12b_prefill_mm_shape` carries both, and `L_attn = round_up(L, 128) <= L_padded`
  always, because 512 is a multiple of 128. Running attention over fewer rows than the
  matmuls is free and cannot overrun the scratch.
* **The kv stream over-reads past `L_end`.** The last query round streams keys to the end of
  the 128-row block containing its last query row, i.e. `round_up(L_begin + L_attn, 128)`.
  Those extra rows are all masked off by the causal term, so they never reach the output —
  but the DMA still reads them, so they must be *inside the BO*. `_attend_global_npu()`
  checks `kv_end <= kv_rows` and falls back to the host kernel if not, which is only
  reachable within 128 rows of the context cap.
* **Do not push `attn` in `forward()`.** The old code did `gemma4_12b_scratch_push(bufs.attn)`
  unconditionally before `o_proj`, which was right when the host wrote it. On the array path
  it is a flush of host lines over what the array just produced — and since one `bufs.attn`
  is shared by all 24 layers of a kind, the dirty lines in question are the *previous*
  layer's vision patch. The push now belongs to whoever wrote the buffer: the two host
  attention paths push, the array path does not, and `_patch_vision_blocks()` keeps its own
  pull-patch-push pair.

`ENABLE_DEBUG` builds score `attention_out` against `gemma4_12b_cpu::attention()` through
the usual `gemma4_12b_prefill_xref`, but only on the layers `xref->enabled()` selects and
only over `xref->rows()` rows — `attn` is `L × DQ`, megabytes per layer, and pulling it
unconditionally on all 24 would cost more than the attention.

### 4.2.2 The sliding attention on the array — `_attend_sliding_npu()`

Same three run arguments in the same order (`attn`, `q`, `kv`), same RTP triple, same
regeneration guard, same host fallback. Everything interesting is in the third argument.

**The overlay cannot read the ring.** It masks with the same counter it indexes kv rows
with: `pointer_block_k = max(0, Lq - window)`, then it walks 128-row blocks forward from
there and derives each key's position from its offset in that walk. `pos % sliding` is not
monotonic in `pos`, so there is no value of `L_begin` that makes the ring line up. It gets
the **unrolled window** instead — which `_attend_sliding()` was already building for the
host kernel, so the array path costs no extra host work at all.

**The relative frame.** Row 0 of the unrolled window is absolute position `win_begin`, so
the array is told `L_begin - win_begin` rather than `L_begin`:

```
absolute:  q row i at L_begin + i     kv row j at win_begin + j
relative:  q row i at L_begin - win_begin + i     kv row j at j
```

Both mask terms — `k <= q` (causal) and `k > q - window` (the window) — are *differences* of
positions, so the shift cancels and the mask is unchanged. Two consequences worth naming:

* The relative `L_begin` is `min(L_begin, sliding - 1)`, i.e. it stops growing at 1023 once
  the window has filled. **This is what bounds the work**: a global layer's kv stream grows
  with the prompt, a sliding layer's does not. Prefilling 8k tokens streams the same
  `window + 256` rows per round at position 8000 as at position 2000.
* `max(0, relative Lq - window)` clamps at relative 0, which *is* the oldest visible key. The
  clamp is correct rather than merely safe.

**One buffer, not two.** The overlay takes a single `kv` argument and finds V at half of it,
so `swa_k` / `swa_v` became one `swa_kv` allocation with two accessors:

```
swa_kv : [swa_rows][SWA_DK]  then  [swa_rows][SWA_DV]     (SWA_DV == SWA_DK)
swa_rows = sliding + L + 128
```

This is **prefill staging only**. The decoding ring in `gemma4_12b_npu_detail::kv_caches` is
untouched, and nothing outside `_attend_sliding()` reads these rows — that is what makes the
merge safe, and it is the reason the merge is here rather than in the cache itself.

The `+ 128` is the same over-read as in §4.2.1, and it is why the allocation grew: the last
round streams to `relative L_begin + L_attn + 127`, which with a filled window is
`L_attn + 1150` — past the old `sliding + L`. Rows beyond `win_len` hold whatever the
*previous layer* left there; the causal term masks every one of them, and a masked score is
replaced by `-inf` with a `select` before it reaches the running max, so even an `inf` out of
the mmul is harmless.

**Two things the DH_256 sequence does differently from DH_512** (see
`gen_attn_prefill_swa()`):

* **Floorplan.** Four compute units of two shim columns, not two of four. `q` goes out as two
  halves over the two channels of `IT[cu][0]`; `o` comes back on channel 0 of both columns;
  and **K/V are broadcast from `IT3` and `IT7` only** — one kv head per *pair* of units,
  which is the GQA-2 mapping (q head `4h + cu` reads kv head `2h + cu/2`). BD ids are
  `bd+0` for o, `bd+1..2` for q, `bd+3..4` for kv, with `bd = (round % 2) * 8`.
* **The kv block count is copied from the kernel**, not recomputed:
  `kv_iters = (Lq + 255 - kv_begin) / 128`. The testbench's
  `round_up(Lq + 256, 128) - kv_begin`, divided by 128, agrees with it *only while the window
  is a multiple of 128* — take the window off a 128 boundary and the DMA sends one block
  more or less than the kernel consumes, which deadlocks rather than producing a wrong
  answer. Using the kernel's own expression removes the invariant; an assert on
  `window % 128 == 0` documents the one that remains.

The earlier note about a `win_begin` off-by-one is resolved: reading the mask
(`apply_mask_and_get_max`, `mask_l = current_idx > q_pos - window`) confirms the visible range
is exactly `[p - window + 1, p]`, so the host's `win_begin = max(0, L_begin - sliding + 1)`
matches the kernel's mask. The kernel's `kv_begin = max(0, Lq - window)` is one row lower
because it is a *DMA extent*, deliberately a superset of what the mask keeps.

### 4.3 The host attention kernel — `gemma4_12b_cpu::attention()`

Straightforward per `(row, head)` grouped-query attention, OpenMP over
`collapse(2)`, one score vector and one fp32 accumulator per thread:

* `groups = num_heads / num_kv_heads`, `hk = h / groups` picks the kv head.
* The visible range for query at absolute `p` is `[j0, j1]`:
  * `j1 = p - kv_pos_begin` — causal.
  * `j0 = max(0, p - window + 1 - kv_pos_begin)` when `window > 0`.
  * **vision block override**: if `blk_begin <= p < blk_end`, `j1` is opened to
    `blk_end - 1` and `j0` lowered to `blk_begin` (the latter only matters for a block
    wider than the window, which cannot happen: 280 < 1024).
* Then the usual max-subtract / `expf` / normalize / `axpy` over V.
* **No scale factor** — see §1.
* Output column order is the `q_proj` head order. The upstream reference rearranges
  `B (h R) L d -> B h L R d` before the einsum and back afterwards; the two cancel, so no
  permutation is needed here.

`dot_bf16` and `axpy_bf16` are AVX-512, bf16 loaded and accumulated in fp32.

### 4.4 Vision blocks — the one place the mask is not causal

This is the subtlest piece of the prefill attention.

**The rule.** Image and video soft tokens attend to each other **bidirectionally** inside
one item. A "block" is a maximal run of `image_token_id` or `video_token_id` in the prompt.
It can never span two items because the processor always emits `boi`/`eoi` markers around
each one, and those are plain text ids, which breaks the run.

**Audio is NOT a block.** Audio soft tokens are masked exactly like text. This is the most
surprising rule of the whole scheme and the code calls it out explicitly in
`gemma4_12b_prefill_blocks`'s doc comment.

**How it is implemented.** Rather than teach a fixed-function attention an arbitrary mask,
the standard causal/sliding attention runs unchanged and the wrong rows are **recomputed
and overwritten**:

```
_stage_vision_blocks()   # BEFORE the standard attention is launched
    pull(q)
    for each block clipped to this chunk:
        q rows  = [max(blk.begin, L_begin), min(blk.end, L_end))
        kv rows = [window ? max(kv_begin, q_lo - window + 1) : kv_begin,
                   min(blk.end, kv_end))
    image_npu = _image_route(cache_rows)      # decided here, and final
    if !image_npu: memcpy q/k/v of those rows into plain host vectors
standard attention (array on the global layers, host on the sliding ones)
_patch_vision_blocks()   # AFTER
    _recompute_vision_blocks()      -> gemma4_12b_cpu::attention(..., blk_begin, blk_end)
      or
    _recompute_vision_blocks_npu()  -> attn_global_image, one run per block
    _commit_vision_blocks(): pull(attn); memcpy the recomputed rows over attn; push(attn)
```

Design points, all documented in the source:

* The host staging buffers are **plain host memory, never device BOs**, even when the
  standard attention runs on the array. Pointing the fix-up at the BO mappings would
  interleave host and accelerator traffic on the same memory banks for the whole recompute;
  one bounded copy in and one out is cheaper, and it makes the fix-up independent of where
  the standard attention ran.
* Staging happens *before* the array run so the recompute never shares a buffer with the
  accelerator and is free to overlap it. The sliding path uses that: it issues the array run,
  runs `_recompute_vision_blocks()` while it is in flight, then joins.
* `_commit_vision_blocks()` is the **only** writer of `bufs.attn` in the fix-up and is shared
  by both paths. Whichever path computed the rows leaves them in `stage_out`; one pull, one
  patch of every block, one push.

The `kv_hi = min(blk.end, ...)` bound is the reason this is cheap: everything above the
block end is masked for every query in the block anyway, so the recompute needs at most
`blk_begin_window .. blk_end` keys. At ≤280 rows per image against a whole prompt, the
recompute is a rounding error next to the pass it corrects.

#### 4.4.1 The global block rows on the array — `attention_global_image`

`FLM_Xclbin/Gemma4_12B_QAT/attention_global_image` is `attention_DH_512_prefill` with the
causal term deleted. Its kernel takes `(KV_CHUNKS, Q_CHUNKS, L)` where the causal build takes
`(L_begin, L_chunks, window_size)`, and — the tell — its `attn_core` signature **drops the
`row` argument** entirely: the mask does not depend on which query row is being computed.
Every row it is handed attends kv rows `[0, L)`, masked at the tail of the last 128-row block
by `apply_mask_and_get_max(pS, m, hi = L-1, k_base)`.

For a **global** layer that is exactly the §4.4 rule. Set `window = 0` and `kv_pos_begin = 0`
in the staging above and every block row gets the same `[0, min(blk.end, kv_end))` — one
range, no row dependence. Which is also why the sliding layers cannot use it: with a window
the range starts at `q_lo - window + 1` and moves with the row.

Two buffers move; the kv cache does not:

* **kv is the layer's decoding cache, bound verbatim** as run argument 2 — the same
  `buffer<bf16>&` `_attend_global_npu()` binds, same K half, V found at `cache.size()/2`. No
  staging buffer, no repack, no second layout to keep in step. `kv_rows` and the V base are
  derived from what the layer owns *at call time*, never from an `L_padded` and never from a
  `MAX_L` captured at build time, which `update_max_length()` would leave stale.
* **q and o go through `img_q` / `img_out`**, two small BOs sized to the tallest block rounded
  up to 128. They exist because the overlay works in whole 128-row chunks: a 260-row block
  runs as 384, and those 124 tail rows come back computed against the *block* mask, which is
  the wrong answer for whatever follows the block. Landing them in `bufs.attn` would corrupt
  rows the array had already got right. The padded q rows are **zeroed**, not left dirty — a
  zero query scores 0 against every key, so its softmax is uniform and finite, whereas stale
  bf16 can carry an `inf` into the running max and poison the tile.

`_image_route()` decides the path once per prefill, from `_stage_vision_blocks()`, **before
anything is copied** — because on the device path `stage_k`/`stage_v` are never filled, so a
late failure would have no way back. Everything that can disqualify the array is checked
there: no bitstream, `kv_lo != 0`, a rounded-up kv extent past the cache, or more than 1024
BD iterations. Any one of them leaves `image_npu` false and the host path fully intact.

The sequence guard is `(q_pad, kv_valid, kv_rows, kv_elems)` — no absolute query position,
because the mask has none. One image is **one** sequence regeneration shared by all 24 global
layers.

The honest cost: the host recompute this replaces is bounded by `blk.end`, not by prompt
length, so for an image early in a long prompt the win is modest, and it is paid for with 24
extra `hw_context` switches per prefill plus a fourth live context. Measure with
`PREFILL_PROF_TICK` before assuming; `PREFILL_IMAGE_ATTN=cpu` is the A/B.

A block flush against row 0 of a non-first chunk is **reported, not silently truncated** —
the ids of this chunk alone cannot recover a block that started in an earlier one. The
processor never splits an item across calls, so it should not happen in practice.

### 4.5 o_proj

`o_proj` writes straight over `common->x`, which q/k/v are long done reading. Note there is
**no `push(attn)` here any more** — `attn` is pushed by whichever stage wrote it, inside the
`_attend_*` functions; see the last bullet of §4.2.1 for why the unconditional push became
a correctness bug. On entry to `attn_block::forward()` `common->x` holds
`input_layernorm(residual)` and is on the device; on return it holds the attention output
and is on the host.

---

## 5. The NPU matmul — `gen_dequant_mm_512()`

The only array kernel the prefill uses (attention aside). `y[L, D_out] = dequant(W) · x[L, D_in]`.

* Tiling: `desired_m = 128` rows per core row, `desired_k = 256` (one q4_0 block wide),
  `desired_n = 64` columns per core column. Four shims per half ⇒ `unit_m = 512` rows,
  `unit_n = 256` columns per unit.
* Hard constraints (asserted): `D_in % 256 == 0`, `D_out % 256 == 0`, **`L % 512 == 0`** —
  which is exactly why `gemma4_12b_prefill_mm_shape::L_ALIGN = 512` and why every run sees
  `L_padded`, not `L`. Also `D_in / 256 < 64`, because that count rides a 6-bit BD
  iteration dimension.
* Two units per round, left half on shims 0–3 and right half on 4–7, double-buffered over
  BD ids with `(r % 2) * 8`; the drain of round `r-1` is issued after the descriptors of
  round `r`, so the array never idles between rounds.
* `rtp_act` is broadcast to **every** core — which is why the fused up/gate region is run
  **twice**, once per slab parity (`UP_MATRIX` / `GATE_MATRIX`), so that only the gate run
  carries `DEQUANT_MM_ACT_GELU`. `y_dout_base()` undoes the slab interleave so both runs
  scatter into one `[L, 2*IS]` row de-interleaved as `[ up | gate ]`, leaving
  `cpu_func::_glu_mul` with nothing but the multiply.
* The control sequences bake in `L`, `D_in`, `D_out` and the weight offset, so a projection
  app's only run argument is the packed weight buffer — and the sequences are regenerated
  whenever `L_padded` changes (`attn_block::reserve()` / `mlp_block::reserve()`).

The q4_0 block format (2304 B per 16×256 block: 128 bf16 scales indexed `g*16 + r`, then
2048 B of signed nibbles, low nibble first) is documented at the top of
`gemma4_12b_prefill.hpp` and mirrored by `gemma4_12b_cpu::dequant_row_block()`.
`gemma4_12b_desc::reorder_cpy()` interleaves pairs of 16-row bands per column block, which
is what `gemma4_12b_cpu::block_offset()` reproduces.

---

## 6. The BO sync discipline

`buffer<T>::data()` is the BO's **own** host mapping, so `sync_to/from_device()` are a cache
flush / invalidate of one allocation, not a copy between two. XRT gives no ordering beyond
that, which makes the rule strict and symmetric:

* **`pull()` before a host stage reads anything an NPU run wrote** — rope, the rms norms,
  the glu multiply, the kv-cache unroll.
* **`push()` immediately after *any* host write to a BO**, whether or not a run is about to
  read it. An unflushed host write leaves dirty lines that evict at an arbitrary later
  time, landing on top of whatever the array has written into that allocation since. *A
  push that is correct but late still leaves a window in which an intervening run can be
  clobbered.*

This is also why `reserve()` pushes buffers it only memset — including ones only the array
ever writes.

---

## 7. Cross reference (`gemma4_12b_prefill_xref.hpp`)

Because both a host and an array implementation of the dequantizing matmul exist,
`ENABLE_DEBUG` builds replay each NPU matmul on the host and score cosine / rel-L2 /
max-abs. Covered: `q/k/v/o_proj`, the fused up/gate pair (scored as `up_proj` and `gelu`
separately), `down_proj`, and `attention_out` on **both** layer kinds. **Not**
covered: the rms norms, rope, the glu multiply — they are the same `cpu_func` call in both
builds, so there is no second implementation to compare against.

`attention_out` differs from the matmuls in two ways. The replay is **inside**
`_attend_global_npu()` / `_attend_sliding_npu()` rather than deferred to
`gemma4_12b_mm_run::wait()`, because there is no run handle to hang it off; and it is gated
on `xref->enabled(layer_idx)` *before* the pull, not after, because `attn` is `L × DQ` and
pulling it on all 48 layers would cost more than the attention it is checking. Expect a
looser cosine than the projections get: the array path is on the bfp16 floor of
`aie::mmul<8,8,8,bf16>` for both `Q·Kᵀ` and `S·V`, while the host reference is fp32
throughout.

The sliding replay passes `swa_k_ptr()` / `swa_v_ptr()` with `kv_pos_begin = win_begin`, so
host and array read the *same* unrolled window. A disagreement there is the attention, not
the unroll — which is worth knowing, because the unroll is the one part of the sliding path
no reference implementation covers.

* Default layer set: layer 0 (first SWA), `sliding_window_pattern - 1` (first global), and
  the last layer (which carries the accumulated drift). Override with `FLM_XREF_LAYERS`
  (`all` or a comma list) and `FLM_XREF_ROWS` (default 256, `0` = every row).
* `xref::guard` fingerprints the scored tensor before/after and asserts it was not
  modified — the cross reference must stay an observer.
* One trap the header documents at length: `check()` indexes `got` with a stride but `want`
  **packed**, so a strided reference has to be compacted first. Getting this wrong scores as
  a plausible-looking low cosine rather than an obvious failure.

---

## 8. Decode-side attention, for contrast

Worth knowing because prefill has to hand off to it cleanly.

* `set_context_length(L)` regenerates `gen_layer_seq()` for both layer kinds with `L + 1`,
  regenerates layer 0's scale sequence, rebuilds the runlist (when preemption is off) and
  recomputes the cos/sin tables for position `L`.
* **Only layer 0 publishes the context length.** `gen_layer_scale_seq()` writes the `L` RTPs
  to the attention cores once per token; the other layers reuse what the cores already
  hold. The `L` written is clamped two different ways: `min(L, MAX_L)` for global (the
  cache is not a ring), `min(L, SLIDING_LENGTH)` for SWA (because `_move_kv_cache()` only
  ever streams that many rows into the engine).
* `_receive_kv_cache()` writes the new K/V row at `(L-1)` for global, `(L-1) %
  SLIDING_LENGTH` for SWA — the exact indexing `Impl::_kv_cache_row()` mirrors on the host
  and `_attend_sliding()` unrolls.
* `_move_kv_cache()` replays the cache into the engine that owns the layer. For SWA there
  are three cases: aligned (`L % SLIDING == 0`, one full-ring move), wrapped (`L > SLIDING`,
  two phases split at `L % SLIDING`), and the short fallback (`0..L`, rounded up to
  `L_CHUNK = 16` and **clamped to the physical row count** so V's descriptor cannot read
  off the end).

---

## 9. Gotchas worth remembering

* `L_padded = round_up(L, 512)` is a hard requirement of `gen_dequant_mm_512`, and
  `residual` is padded and zeroed to match even though the host stages only ever write the
  first `L` rows. `L_attn = round_up(L, 128)` is the attention's own, looser alignment; the
  two coexist because 512 is a multiple of 128 (§4.2.1).
* A prefill now **alternates between bitstreams on every layer**: `dequant_mm` for the
  projections, then `attn_global` or `attn_sliding` for the attention, then back. That is up
  to 96 `hw_context` switches per prefill (2 per layer × 48) across *three* live contexts, so
  if the array-side speedup does not show up end to end, measure the switch cost before
  touching the kernels. `npu_xclbin_manager` caps at `max_xclbins = 16` and dedupes by
  filename, so registering a third bitstream is free; keeping three contexts *live* is not.
* The two attention overlays are **not interchangeable** despite identical run signatures.
  Their RTP addresses differ (`59520/11520/59552` vs `61568/11904/61600`), their floorplans
  differ (2 units × 4 shims vs 4 × 2), and their kv addressing differs (1 head, absolute
  position vs 8 heads, window-relative). Feeding one's sequence to the other's context
  writes RTPs into whatever buffer happens to live at that address.
* Media rows deliberately **bypass** the `sqrt(hidden_size)` embedding scale:
  `Embedding::forward()` applies it to text rows, then `_embed_media()` overwrites the soft
  token rows with unscaled projected features.
* `_init_prefill_mm()` and `_init_mm_projector()` are lazy on purpose — registering
  `dequant_mm.xclbin` or reading the ~105 MB `image_audio_weights.q4nx` in the constructor
  would tax every decode-only / text-only session and would break models shipped without
  those files.
* `update_max_length()` migrates the global caches into the new `[MAX_L*DK | MAX_L*DV]`
  layout rather than dropping the conversation, leaves sliding rings alone (they are already
  sized for their window), and **resets `prefill_mm`** because its apps are bound to the old
  cache buffers.
* `Impl::_drain_preload()` must run before either prefill path, because both reassign
  `pre_load_run` on their way out and would otherwise destroy a run mid-flight.
* Both prefill paths, and `forward()`, leave a preload run in flight so the layer xclbin
  stays resident while the caller samples.

---

## 10. Where the kernels live — `FLM_Xclbin/Gemma4_12B_QAT/`

Everything above is the **host** side. The bitstreams it loads are built out of
`FLM_Xclbin/Gemma4_12B_QAT/`, one directory per xclbin. They all have the same shape:

```
<design>/
  iron/      IRON / MLIR-AIE python — the array floorplan: which tile does what,
             which DMA flow connects to which, what the buffer types are.
             `iron/<name>_aie.py` is the top level, and it names the bitstream:
             layer_aie.py -> layer.xclbin, lm_head_aie.py -> lm_head.xclbin, ...
  kernel(s)/ the AIE C++ that runs *on* a compute tile, compiled per-tile to a .o
             that `iron/` links into the named core.
  host/      a standalone testbench: builds the same instruction sequence the DLL
             builds, runs it on real weights, and scores it against a reference
             safetensors. This is where a kernel is debugged before the DLL ever
             sees it.
  makefiles/, Makefile, build/, visualize.py    build glue and a layout dump.
```

### Which directory is which part

| Directory | Builds | Used by |
| --- | --- | --- |
| **`decoding/`** | `layer.xclbin` — **one whole transformer layer, token-at-a-time (decode)**: rms norms, all four projections, RoPE, both attention flavours, GLU. | `gen_layer_seq()` / `gen_layer_scale_seq()`; the decode path (§8) |
| **`lm_head_npu_bin/`** | `lm_head.xclbin` — **the final vocab projection**, `[3840] -> [262144]`. Q8_0 weights, 16×256 blocks (E2B/E4B use Q4_K and live in a different tree; the python asserts on this). | `gen_lm_head_seq()` |
| **`dequant_mm/`** | `dequant_mm.xclbin` — **the batched dequantizing matmul used by prefill**. All 32 compute tiles are matmul cores; tiles are `m=128, k=256, n=64`, even and odd rows run different objects (`dequant_mm_e.cc` / `dequant_mm_o.cc`) so the two halves of a column can share the weight stream. | `gen_dequant_mm_512()` (§5) — q/k/v/o_proj, up/gate, down_proj in prefill |
| **`audio_image_mm/`** | the **multimodal projector** matmuls, plain bf16 × bf16 (not quantized): audio `K=640 -> 3840`, image `patch_dense K=6912 -> 3840` **with bias**, image `embedding_projection K=3840 -> 3840`. Two kernels, `mm.cc` and `mm_bias.cc`. *(untracked / WIP)* | would replace the host side of `_init_mm_projector()` / `_embed_media()` |
| **`attention_DH_512_prefill/`** | `attn_global.xclbin` — **prefill attention for the global layers** (`DH = 512`). | `gen_attn_prefill_global()` → `_attend_global_npu()` (§4.2.1); on by default |
| **`attention_DH_256_prefill/`** | `attn_sliding.xclbin` — **prefill attention for the sliding-window layers** (`SWA_DH = 256`). Same `attn.cc`, different tiling. | `gen_attn_prefill_swa()` → `_attend_sliding_npu()` (§4.2.2); on by default |
| **`attention_global_image/`** | `attn_global_image.xclbin` — **the §4.4 vision-block rows of the global layers**. `attention_DH_512_prefill`'s floorplan with the causal term deleted: RTPs are `(KV_CHUNKS, Q_CHUNKS, L)`, and `attn_core` has no `row` argument at all. | `gen_attn_prefill_global_image()` → `_recompute_vision_blocks_npu()` (§4.4.1); on by default, registered lazily on the first image |
| **`causal_attention_decoding/`, `sliding_attention_decoding/`, `dequant/`** | nothing the DLL loads — **standalone testbenches** for one piece in isolation (global decode attention, SWA decode attention, the q4nx dequantizer). | development only |

Note the naming: **`DH_512` is the global flavour and `DH_256` is the sliding-window one**,
which is the opposite of what "bigger window, bigger number" would suggest — gemma4-12b gives
the global layers the *wider* head and only one KV head (§1).

### `decoding/` in a bit more detail

`decoding/iron/layer_aie.py` is the floorplan, and it is the same map as
`gemma4_12b_npu_sequence.hpp`. Reading them side by side is the fastest way to see what a
tile constant in the host code actually is:

```
                col 0,1        col 2            col 3         col 4          col 6,7
 row 0..3   16 proj cores   global attn   rms / glu / rope   SWA attn    16 proj cores
   shim       IT0, IT1         IT2           IT3 (x in/out)    IT4         IT6, IT7
```

| host constant (`..._sequence.hpp`) | IRON object (`layer_aie.py`) | kernel source |
| --- | --- | --- |
| `rms_tile = CT03` | `rms_residual_tile` | `rms_residual.cc` |
| `glu_tile = CT13` | `glu_tile` | `glu.cc` |
| `rope_ct = CT23` / `swa_rope_ct = CT33` | `RoPE` / `SWA_RoPE` | `rope.cc` / `swa_rope.cc` |
| `attn_qk_tile = {CT02, CT22}`, `attn_kv_tile = {CT12, CT32}`, `attn_tile = IT2` | `attn_column` | `attn_qk.cc` / `attn_kv.cc` |
| `swa_attn_qk_tile = {CT04, CT24}`, `swa_attn_kv_tile = {CT14, CT34}`, `swa_attn_tile = IT4` | `SWA_attn_column` | same two sources, `-DATTN_ROLE=ATTN_ROLE_SWA` |
| `proj_tiles[16]` = cols 0,1,6,7 × rows 0..3 | `q4_proj_engine` | `proj_main.cc`, `q4_core.h` |
| `xr_tile = out_tile = IT3` | x in / x out shim | — |
| `rtp_addresses` | the `%RTP` buffers | dumped to `build/<MODEL>/xclbins/rtp_addresses.txt` by the bitstream rule |

The **one-source-two-roles** trick is worth knowing: `kernels/attn_qk.cc` and
`kernels/attn_kv.cc` are each compiled twice, plain and with `-DATTN_ROLE=ATTN_ROLE_SWA`
(`makefiles/kernel.mk`), producing `attn_qk.o` / `swa_attn_qk.o` etc. `kernels/attn_role.h`
is the switch: it picks `HEAD_DIM` (512 vs 256), `ROLE_KV_HEADS` (1 vs 8), `Q_PASSES_PER_CU`
(2 vs 1), and even the entry-point name, so each object only ever sees one geometry.
`kernels/model_spec.h` then turns that into the GQA tiling —
`ATTN_IMPL_1x8x1` for global (1 KV head × 8 Q heads per CU) and `ATTN_IMPL_4x2x1` for SWA
(4 KV heads × 2 Q heads), both filling exactly the 8 output lanes of `aie::mmul<8,8,8>`.
The numeric constants (`MODEL_DIM = 3840`, `NUM_ATTN_HEADS = 16`, `DH = 512`,
`SWA_DH = 256`, `INTERMEDIATE_SIZE = 15360`, the two attention scales, `QUANT_TYPE`) come
from `decoding/models/gemma4-12b.h` on the C side and `decoding/iron/models/gemma4.py` on
the python side; the round arithmetic (`GQA_LANES = 8`, `ATTN_ROUNDS`, `Q_PASSES`) is
derived in `decoding/iron/model_config.py`. `decoding/kernels/attn_*_e2b.cc` /
`_e4b.cc` are the other Gemma4 sizes sharing this tree — not built for 12B.

### The prefill attention designs

`attention_DH_512_prefill/` is what §4.2.1 loads and `attention_DH_256_prefill/` is what
§4.2.2 loads. Both are flash-attention:
all 32 compute tiles run the same `attn()` from `kernel/attn.cc`, each owning a slice of the
query rows chosen from its own `(row, col)`, and each streaming K/V through a ping-pong pair
of memtile buffers while carrying the running max / running sum (`prev_m`, `l`, rescale `c`)
in fp32. The runtime shape comes in as RTP buffers — `L_begin_buffer`, `L_chunks_buffer`,
`window_size_buffer` — so **one bitstream covers both the causal and the sliding mask**; the
window size is a parameter, not a build option. The two directories differ only in tiling:
`d=512, lq=lk=8, lq_ct=16` for global vs `d=256, lq=lk=16, lq_ct=32` for SWA.

Note what these designs do *not* have: any notion of a vision block. The mask they can
express is "causal, optionally windowed", which is exactly the §4.2 pass. The §4.4 block
patch stays on the host either way.

The DH_512 floorplan, as the host sequence sees it: **two compute units of four shim
columns**, `IT[0] = {IT0, IT1, IT2, IT3}` and `IT[1] = {IT4, IT5, IT6, IT7}`. Per round of
128 query rows and per unit, one attention head is processed: `q` goes out as four quarters
of 32 rows over the two channels of the unit's *outer* shims (`IT[cu][0]` and `IT[cu][3]`),
`o` comes back on channel 0 of all four columns, 32 rows each. K and V are the single kv
head of a global layer, so they are broadcast to all 32 cores from **one** shim, `IT2`, on
channels 0 and 1 — which is why the V offset is the only place `cache.size()/2` appears in
the sequence. BD ids are double-buffered by round (`bd = (round % 2) * 8`, slots +0..+6).

The DH_256 floorplan is the **transpose** of that: **four compute units of two shim
columns**, `IT[4][2] = {{IT0,IT1}, {IT2,IT3}, {IT4,IT5}, {IT6,IT7}}`. Four units × four
passes covers the 16 heads, one head per unit per round. `q` goes out as two halves of 64
rows over the two channels of the unit's *first* shim only (`IT[cu][0]`, so `IT0/IT2/IT4/IT6`);
`o` comes back on channel 0 of both columns, 64 rows each. The kv side is where the GQA
ratio shows: SWA has 8 kv heads for 16 q heads, so a kv head is shared by exactly **two**
adjacent units and is streamed from that pair's *second* shim — `IT3` feeds units 0–1 and
`IT7` feeds units 2–3, K on channel 0 and V on channel 1 at `+kv_elems/2`. BD ids are
double-buffered the same way (`bd = (round % 2) * 8`, slots +0..+4). Unlike the global
sequence the kv DMA is not a single flat extent: it is `kv_iters` blocks of 128 rows
starting at `max(0, Lq - window)`, and `kv_iters` is copied verbatim from the kernel's own
`kv_blocks` expression (§4.2.2) so the two sides cannot disagree on how many blocks are in
flight.

**Installing it.** All three designs' build rules already emit the exact filenames the DLL
looks up — `build/xclbins/attn_global.xclbin`, `build/xclbins/attn_sliding.xclbin` and
`build/xclbins/attn_global_image.xclbin` — so installing is a plain copy into
`<exec_path>/xclbins/<model_name>/`, next to `dequant_mm.xclbin`. No rename. The name is the
only link between the two trees, and `npu_xclbin_manager::register_xclbin()` dedupes by
filename, so a mismatch is a load failure at the first prefill rather than at startup —
except for `attn_global_image.xclbin`, which is registered inside a `try`/`catch` and whose
absence only downgrades the §4.4 fix-up to the host kernel.

Three deliberate divergences from the testbenches' `gen_seq_1_4()`, all noted in the ports:

1. the `MAX_L` argument, per §4.2.1;
2. the `kv_length > 128 * 1024` branch is **not** ported (any design). It reuses the same
   two BD ids twice inside one round, which is a hazard, and 128 K rows is far past any
   context this model runs; all three generators assert instead.
3. in the SWA port only, the testbench's `Num_of_d_in_KV_cache` chunking generality is
   dropped — `swa_kv` is one flat `[rows][SWA_DK]` staging buffer with V at half, so the
   chunk loop collapses to a single extent per kv head.
4. in the image port only, `kv_length` is **not** taken from a padded prompt length. The
   testbench conflates "rows the kv buffer has" with "rows the block may see" because for it
   the two are the same number; in the DLL the buffer is the layer's `MAX_L` cache and the
   visible range ends at the block. The extent is rounded up from `kv_valid_rows` and checked
   against the capacity separately.

Also worth carrying over from the decode side: `aie::mmul<8,8,8,bf16>` on npu2 truncates
both operands onto a shared-exponent-per-8 (bfp16) grid, so both `Q·Kᵀ` and `S·V` here sit
on that accuracy floor. Do not expect the cross-reference cosine against
`gemma4_12b_cpu::attention()` to be as tight as the projections are.
