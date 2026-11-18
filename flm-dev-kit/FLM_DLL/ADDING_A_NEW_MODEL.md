<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Adding a new model to FLM_DLL

Written for other sessions picking up a "support model X" task. It is a distillation of
three engines that were built at different times and represent two generations of the
pattern:

| engine | shape | prefill generation |
|---|---|---|
| `detail/qwen3_5vl_npu/` | hybrid attention + linear (SSM/gated-delta) layers, VL | 1st gen: `mm.xclbin` (bf16 gemm) contexts |
| `detail/qwen3_6_moe_npu/` | hybrid attention + linear, MoE FFN, VL | 1st gen + `dequant_mm` for the experts |
| `detail/gemma4_12b_npu/` | hybrid global/sliding attention, omni (image+audio) | **2nd gen**: `dequant_mm.xclbin` + a full host reference + cross reference |

**Copy `gemma4_12b_npu/` unless you have a reason not to.** It is the only one with a
complete CPU implementation of every array stage and a scoring harness that compares them,
and §5 below is the reason that matters. `detail/gemma4_12b_npu/README.md` is a much deeper
dive into that one engine (its prefill attention, kv-cache discipline, xclbin trees); this
document is the cross-cutting "how do I start a new one".

Read that README once before you start. Then come back here.

---

## 0. The map

An engine is a standalone `.so` (`libgemma4_12b_npu.so`) with exactly one exported class,
plus a public header the application compiles against. Nothing else crosses the boundary.

```
FLM_DLL/
  include/models/<model>/<model>_npu.hpp   the ONLY public header: class <model>_npu : causal_lm
  include/weight_desc.hpp                  weight_desc_t / weight_container, shared
  include/causal_lm.hpp                    the 11 virtuals every engine implements
  include/modules/                          Embedding, lm_head, gemm, dequant helpers
  include/npu_utils/                        npu_xclbin_manager / npu_app_manager / npu_app / npu_sequence
  detail/<model>_npu/
    <model>_npu_def.hpp        §1  config.json -> dimensions + every weight_desc_t + loaders
    <model>_npu_detail.hpp     §4  struct <model>_npu::Impl — every buffer and every method
    <model>_npu.cpp                Impl ctor / forward / prefill / load_weights / checkpoint
    <model>_npu_sequence.{hpp,cpp} instruction-sequence generators (gen_layer_seq, gen_*_prefill)
    <model>_prefill.hpp        §2  the batched prefill graph + the host reference kernels
    <model>_prefill_xref.hpp   §5  cpu/npu scoring, ENABLE_DEBUG only
    cpu_functions.hpp              namespace cpu_func: norms, rope, glu, residual (host, always)
    Makefile / CMakeLists.txt      linux (g++-13) / windows (MSVC) builds of the .so
../FastFlowLM/src/
  include/AutoModel/modeling_<model>.hpp   §4  the chat-level wrapper
  common/AutoModel/modeling_<model>.cpp
  include/AutoModel/all_models.hpp         §4  family enum + tag -> class dispatch
```

One repo. `flm-dev-kit/FLM_DLL` is the engine; `src/` is the application that links it.
The xclbins are built from `flm-dev-kit/FLM_Xclbin/<MODEL>/`.

---

## 1. `<model>_npu_def.hpp` — the model description

This file is the contract between the checkpoint on disk and the buffers on the device.
Get it right and everything downstream is mechanical; get an offset wrong and you debug a
plausible-looking wrong answer for a day.

It holds two things.

### 1.1 A per-layer-kind weight descriptor struct

```cpp
struct gemma4_12b_layer_weight_def {
    weight_desc_t attn_q, attn_k, attn_v, attn_output;
    weight_desc_t ffn_up, ffn_gate, ffn_down;
    weight_desc_t input_layer_norm, post_attention_norm,
                  pre_feedforward_norm, post_feedforward_norm;
    weight_desc_t rope_cos_sin, attn_q_norm, attn_k_norm, layer_output_scale;
    bool has_v_proj = true;

    std::vector<weight_desc_t*> proj_all();   // device-buffer order, quantized
    std::vector<weight_desc_t*> rms_all();    // device-buffer order, bf16
    std::vector<weight_desc_t*> rope_all();   // device-buffer order, bf16
};
```

A `weight_desc_t` (`include/weight_desc.hpp`) is `{shape, dtype, name, offset}` where
**`name` is a printf format string** — `"model.layers.%d.self_attn.q_proj.weight"` — resolved
per layer by `format_name(layer_idx)`. That name must match the tensor name in `model.q4nx`
exactly.

**The order of the `*_all()` vectors IS the device memory layout.** They are fed to a
`weight_container` in `_build_layer()`, which hands out offsets by simple accumulation, and
the sequence generator reads those offsets back through `get_qkv_offset()` /
`get_o_offset()` / `get_upgate_offset()` / `get_down_offset()`. The decode kernel assumes
q/k/(v) are contiguous, then o, then the interleaved up|gate block, then down. Reordering
the vector silently reorders device memory.

One descriptor struct **per layer kind**, not per layer — all layers of a kind share a
layout. gemma4-12b has two (`swa_weight_desc`, `global_weight_desc`); qwen3.5-VL has two
of a different sort (`qwen3_5vl_attn_layer`, `qwen3_5vl_linear_layer`, both deriving from a
common base and stored as `unique_ptr` in a `weight_desc[]` indexed by kind). If your model
is homogeneous you have one, and everything below collapses.

### 1.2 The `<model>_desc` struct

```cpp
struct gemma4_12b_desc {
    static constexpr flm_dtype_t PROJ_DTYPE   = flm_q40;   // body quantization
    static constexpr flm_dtype_t LM_HEAD_DTYPE = flm_q80;
    static constexpr int UP_GATE_OUTDIM_REORDER = 512;     // up/gate interleave slab

    uint32_t num_hidden_layers, D, intermediate_size, ...;  // parsed dims
    std::vector<f32> inv_freq_global, inv_freq_swa;         // derived rope tables

    void build(LM_Config& config);                          // parse + lay out
    bf16 load_layer_weights(int layer_idx, type, Q4NX&, proj, rms, rope);
    void load_final_norm(Q4NX&, bf16* dst);
    void load_lm_head_weights(Q4NX&, buffer<u8>&);
    void reorder_cpy(...);  void reorder_cpy_lm_head(...);
};
```

`build(config)` is the whole config.json parse. Use the `JSON_GET(field, json, "key",
default, type)` macro — it gives you a default rather than throwing, which is what lets one
engine serve several checkpoint revisions. Nested blocks come through
`cfg_sub(jc, "rope_parameters")` and `cfg_get<T>(sub, "key", fallback)`.

What `build()` must produce:

* **every dimension the rest of the engine reads.** Do the derivations here, once:
  `DQ = DH * num_attention_heads`, `DK = DH * num_kv_heads`, `RoPE_DIM = DH *
  partial_rotary_factor`, `vocab_size_padded = round_up(vocab_size, 1024)`.
* **the rope inverse-frequency tables** (`_build_inv_freq`). Note gemma4's trick: lay the
  frequencies over the *full* head dim and zero everything past the rotated part, so a
  partial rotary can be applied to the whole head with no branch — a zero frequency is an
  identity rotation.
* **the standalone descriptors** — `embed_tokens`, `final_norm`, `lm_head`. These do not go
  through a `weight_container`, so call `.indp()` on them by hand or `ready()` stays false.
* **a `DEBUG_BLOCK(1, ...)` dump of every parsed field.** Every engine has one. It is the
  first thing you look at when a new checkpoint misbehaves, and it costs nothing.
* **`_build_layer()` per kind**, ending in three `weight_container`s (proj / rms / rope) whose
  `get_size()` become the per-layer buffer byte sizes.

Then a family of `inline` accessors that are the *only* interface the rest of the engine
uses: `get_proj_weights_byte_size(type)`, `get_rms_elems()`, `get_rope_rms_elems(type)`,
`get_kv_cache_size(type, MAX_L)`, `get_lm_head_w_size()`, and the `get_*_offset(type)` family
which convert byte offsets to **bf16 element** offsets (that is what the sequence generator
wants — do not hand it bytes).

Also here: `layer_type(int idx)` — the single place the layer-kind rule is expressed.
`Impl` caches its result into `layer_types[]` at construction and never recomputes it.

### 1.3 `reorder_cpy` — the on-device block order

A quantized weight in `model.q4nx` is a flat run of `16 x 256` blocks (2304 B for q4_0,
4352 B for q8_0). The array reads them **interleaved by row band** because two rows of cores
share a weight stream. `reorder_cpy(dst, src, col, dtype, vertical_blocks)` does that
interleave, and `vertical_blocks` is the number of 16-row bands the overlay interleaves —
2 for the body, 4 for the lm_head (which additionally splits into two contiguous halves,
hence a separate `reorder_cpy_lm_head`).

**This function must agree with the IRON floorplan.** The comment on
`reorder_cpy_lm_head` names its counterpart:
`FLM_Xclbin/Gemma4_12B_QAT/lm_head_npu_bin/iron/q4k_mm.py::lm_head_memtile` and
`gen_lm_head_seq`'s `W_PER_STREAM`. When you port to a new overlay, re-derive this from that
overlay's memtile layout — do not assume 2.

### 1.4 `load_layer_weights` — the loader lives with the layout

The description owns the copies, because the description owns the offsets. Per layer:

1. q / k / (v) — one `reorder_cpy` each into `proj_weights.data() + desc->offset`.
2. o — `reorder_cpy` with `col = DQ` (not `D`).
3. up/gate — loaded separately, then **interleaved in slabs of `UP_GATE_OUTDIM_REORDER`
   output rows**: `up[0:512], gate[0:512], up[512:1024], ...`. This is what lets one array
   run produce the fused pair.
4. down — `reorder_cpy` with `col = intermediate_size`.
5. the rms norms — plain `memcpy` into the rms buffer at their offsets.
6. q_norm / k_norm — `memcpy` into the *rope* buffer, behind the cos/sin table.
7. any per-layer scalar (`layer_output_scale`) — into the rope buffer's tail; **zero the
   `MIN_BF16_PAD` padding behind it** so the kernel never reads garbage past the end.
8. `sync_to_device()` on all three buffers before returning.

Call `desc->load()` on each descriptor as you load it. `ready() == added & loaded` is your
cheap "did I forget a tensor" check.

---

## 2. The prefill graph — `<model>_prefill.hpp`

Decode is one token through `layer.xclbin`, driven by `gen_layer_seq()`. Prefill is the
whole prompt in one pass, and it is a **separate graph on separate bitstreams**. It is where
most of the work in a new model goes.

### 2.1 Backend switches at the top of the file

```
make PREFILL_MM=cpu     -> -DFLM_PREFILL_MM_ON_CPU   -> <MODEL>_MM_ON_NPU = 0
make PREFILL_ATTN=cpu   -> -DFLM_PREFILL_ATTN_ON_CPU -> <MODEL>_ATTN_*_ON_NPU = 0
```

**Note the polarity: these are opt-*out*.** A plain `make` is all-NPU; the flags move a
stage back to the host. Define one macro per *overlay*, not per stage — gemma4 has
`ATTN_GLOBAL_ON_NPU` and `ATTN_SWA_ON_NPU` separately because they are two different
bitstreams with different RTP maps and floorplans, even though one make flag currently
moves both.

Derive `<MODEL>_SCRATCH_ON_DEVICE = MM_ON_NPU || ATTN_ON_NPU`. When it is 0 the scratch is
plain host memory and `sync_to/from_device()` must never be called on it
(`buffer<T>::sync_*` asserts on a non-BO). Wrap them in `scratch_push/pull()` no-ops.

Do not tie the scratch allocator to one app manager. `PREFILL_MM=cpu PREFILL_ATTN=npu`
leaves the mm manager null while attention still needs real BOs, so keep a
`bo_manager = whichever manager exists`. This is sound because a BO belongs to the
`xrt::device`, not to a `hw_context` — one allocation can be an argument to runs on either
bitstream.

### 2.2 Structure

```
_prefill_with_mm(ids, payload)
  shape.update(L_begin, L_in)          // L_padded = round_up(L_in, 512)
  prefill_mm->reserve(shape)           // grow scratch, regenerate every control sequence
  prefill_mm->set_vision_blocks(...)   // multimodal: which rows are soft tokens
  embeddings (host) + _embed_media()
  for l in layers: prefill_mm->forward(l, residual, ...)
  x <- residual[L_in - 1]; lm_head_run
  set_context_length(L_begin + L_in)   // hand the position to the decode graph
```

Split the per-layer body into small structs, each owning its own `npu_app`s and its own
`reserve()`:

* `<model>_mm_common_buffers` — the layer-kind-independent tensors (`x`, `upgate`, `hid`).
  These are the biggest, so they exist **once** for the whole stack.
* `<model>_mm_attn_buffers` — `q/k/v/attn` and any staging. **One instance per layer kind**,
  sized for that kind, so a narrow kind does not pay for a wide one.
* `<model>_mm_attn_block` / `<model>_mm_mlp_block` — `build()`, `reserve()`, `forward()`.
* `<model>_prefill_mm_context` — owns all of the above plus the xclbin registration.

`reserve()` returns whether it reallocated; buffers only ever grow.

### 2.3 The run handle — how one call site serves both backends

```cpp
struct gemma4_12b_mm_run {
#if MM_ON_NPU
    flm_rt::run run; buffer<bf16>* dst;
    void wait() {
        if (!dst) return;
        flm::npu::wait_for_completion(run, "matrix multiply");
        dst->sync_from_device();
        _cross_reference();
        dst = nullptr;
    }
#else
    void wait() {}   // the host matmul already ran at issue time
#endif
};
```

This is the pattern that makes §5 possible: the same call sequence compiles on both
backends, and on the NPU side `wait()` is the natural hook for completion validation
and the cross reference. Every raw runtime wait must use
`flm::npu::wait_for_completion()` before consuming its output.

### 2.4 Overlap the host stages against the array

Issue each host stage against the projection that *follows* it, not after all three matmuls:

```
q = x·Wq          (wait immediately)
k_run = start(k = x·Wk)
    rope_rms(q)   (host, overlaps k_run)
    push(q)
k_run.wait()
v_run = start(v = x·Wv)
    rope_rms(k)   (host, overlaps v_run)
    push(k); v_run.wait(); pull(v)
```

The host works on a tensor the array is not touching. This is worth real time on a long
prompt.

### 2.5 BO sync discipline — the single most common source of "random" prefill bugs

`buffer<T>::data()` is the BO's own host mapping, so `sync_to/from_device()` are a cache
flush / invalidate of one allocation, not a copy between two. XRT gives no ordering beyond
that. So:

* **`pull()` before any host stage reads something an NPU run wrote.**
* **`push()` immediately after *any* host write to a BO**, whether or not a run is about to
  read it. An unflushed host write leaves dirty lines that evict at an arbitrary later time,
  landing on top of whatever the array wrote there since. A push that is correct but late
  still leaves a clobber window.
* This includes buffers you only `memset` in `reserve()`, including ones only the array ever
  writes.

### 2.6 Alignment

`L_padded = round_up(L, 512)` is a hard requirement of `gen_dequant_mm_512`. The attention
overlays have their own, looser alignment (`round_up(L, 128)`); the two coexist because 512
is a multiple of 128. `residual` is padded and zeroed to the *matmul* alignment even though
the host stages only write the first `L` rows.

### 2.7 Lazy init

`_init_prefill_mm()` and any multimodal projector init are called from the **first prefill
that needs them**, not from the constructor. Registering a second xclbin or reading a
~100 MB projector weight file taxes every decode-only or text-only session, and breaks
models shipped without those files.

### 2.8 What varies

* **MoE** (`qwen3_6_moe_expert_prefill_context`) — the router (`_cpu_mm_col` → `_moe_gate` →
  `_pack_moe_tasks`) stays on the host, the shared expert runs as ordinary matmuls, and the
  routed experts are packed into batched runs. Note it overlaps `sh_down_run.start()` with
  the host gating.
* **Linear / SSM layers** (`qwen3_5vl_linear_block_prefill_context`) — a conv state plus a
  recurrent state, both carried in the "kv cache" slot.
* **Sliding window** — the cache is a ring indexed `pos % SLIDING_LENGTH` for decode, but
  prefill wants a flat window, so it unrolls the ring into a staging buffer
  (`swa_kv`, K then V) before the attention run. That unroll is the one part of the sliding
  path no reference covers — see §5.

---

## 3. Weight loading, end to end

```
Gemma4_12B::load_model()                        // app side
  q4nx = make_unique<Q4NX>(model_path)
  lm_engine = make_unique<gemma4_12b_npu>(*lm_config, npu.get(), MAX_L)
      -> Impl::Impl()   : desc.build(config); allocate every BO; register layer.xclbin,
                          lm_head.xclbin; create npu_apps
  lm_engine->load_weights(*q4nx)
      -> Impl::load_weights()
           embedding->init_weights(&q4nx, "model.embed_tokens")
           for layer: layer_scale[l] = desc.load_layer_weights(l, type, q4nx, proj, rms, rope)
           for layer: sequence->gen_layer_scale_seq(scale_apps[l].seq(), layer_scale[l])
           desc.load_final_norm(q4nx, x.data() + D)     // final norm lives behind x
           desc.load_lm_head_weights(q4nx, lm_head_weights)
           sync_to_device() everything
           sequence->update_buffer_offsets(proj_offsets)
           clear_context()
  q4nx.reset()                                  // free the mmap immediately
```

Points that are easy to get wrong:

* **The constructor allocates, `load_weights` fills.** Sizes come from `desc` only; never
  hardcode a byte count in `Impl`.
* `MAX_L` is rounded up to a power of two and floored at 4096 in the constructor.
* Buffers are created from an **app manager** (`layer_app_manager->create_bo_buffer<T>(n)`),
  which binds them to the device. Plain `buffer<T>(n)` is host memory and cannot be synced.
* `x` is oversized on purpose: hidden states (D) + final norm weight (D) + initial embedding
  (D), rounded to 8192.
* Anything derived from a weight value rather than its layout — the per-layer output scale
  baked into `layer_scale_apps[l]` — has to be regenerated *after* the load, in a second pass.
* Free the `Q4NX` as soon as loading is done.

### 3.1 Lookup-only tables: map them, do not copy them

A table that is read one row per token (the token embedding, per-layer embeddings, a tied
`lm_head` used as the embedding) never needs to be resident as a whole. Copying it into a
`buffer<>` costs host RAM equal to the table (vocab x dim) for the life of the model, although
a token touches a few KB of it.

Map the tensor read-only from `model.q4nx` with `mapped_file` (`include/tensor_utils/mapped_file.hpp`:
`mmap` on Linux, `MapViewOfFile` on Windows) and dequantize the single row on demand. Only the
pages of tokens actually seen stay resident, and the OS can drop and re-read them under
pressure. `qwen3_5vl_npu/embedding_q8_0.hpp` is the reference: it reads the row straight out of the
packed q8_0 `lm_head` of a tied checkpoint, so there is no bf16 vocab x dim table at all
(Qwen3.5-2B: -970 MiB of anonymous memory, decode and TTFT unchanged, outputs bit-identical).

* `q4nx.get_tensor_metadata(name).offsets` are absolute file offsets, so the mapping needs only
  the file path (`model_path/model.q4nx`) and does not depend on the `Q4NX` object staying alive.
* `mapped_file::open` returns false when the range cannot be mapped; fall back to
  `q4nx.load_weights` into an owned buffer, as `embedding_q8_0` does.
* A row lookup is a page fault only when its pages are not in the page cache (after load they
  are: the same tensor was just read for the upload). If the OS dropped them, a lone row costs
  about 30 ms (a block row spans ~17 pages read one at a time). Prefill therefore hints the whole
  prompt first (`embedding_q8_0::prefetch` -> `mapped_file::will_need`: `madvise(MADV_WILLNEED)` /
  `PrefetchVirtualMemory`, one range per 32-row block). Qwen3.5-2B, page cache evicted, 1.4k-token
  prompt of new tokens: TTFT 1.16 s warm, 1.8-2.0 s with the hint, 3.4-4.9 s without it.
* The mapping pins the file: do not rewrite `model.q4nx` in place while a model is loaded
  (a Linux in-place truncate faults the process; Windows refuses to replace or delete it).
  Replace by writing a new file and renaming.
* Row-wise tables are simpler than the packed lm_head: the Gemma4 E-series token and per-layer
  tables (`gemma4e_npu`, `gemma4e_flash`, `gemma4e_dflash`, `gemma4e_dflash_flash`
  `embedding_q8_0.hpp`) are plain int8 `[vocab, dim]` plus F32 scales `[vocab, dim/32]`, two
  mappings, and the unchanged AVX512 row dequant reads straight from them. Gemma4-12B's bf16
  token table is one mapping and a row `memcpy` (`gemma4_12b_npu/embedding_bf16_mapped.hpp`).
  Gemma4 E4B / Nemotron-Bolt: -3.7 GiB anonymous memory (the per-layer table is 2.9 GiB of it),
  12B: -1.9 GiB (E2B: about -2.9 GiB by the numbers in FLM#807, not measured here); outputs,
  decode speed and TTFT unchanged.
* Nothing reads these tables at load any more, so after a cold boot (page cache empty) the
  first lookups hit the disk. The prefill hint covers it: a 4k-token prompt of new tokens on
  Gemma4 E4B costs +0.06 s with the hint, +0.2 s without it (NVMe).
* `mapped_file::open` refuses a range that ends past the end of the file (truncated download):
  the copy fallback then behaves as before, instead of a SIGBUS on the first lookup.
* Ids outside the vocabulary give a zero row (the old copy read past the buffer).
* Do this only for tables that are really read row-wise. Weights the array reads (`proj_weights`,
  the `lm_head` that feeds the NPU) are synced to device buffers and are not candidates.

---

## 4. The interface upward

### 4.1 The engine's public header

`include/models/<model>/<model>_npu.hpp` declares `class <model>_npu : public causal_lm`
with a pimpl (`struct Impl; Impl* _impl;`). The 11 virtuals from `include/causal_lm.hpp`:

```
forward(int ids) -> buffer<bf16>              decode one token, returns logits
prefill(vector<int>& ids, void* payload)      whole prompt; payload is the multimodal blob
set_context_length(int L)                     regenerate sequences for the new L
load_weights(Q4NX&)
update_max_length(uint32_t)                   grow the caches without dropping the conversation
clear_context()
get_k_cache(layer, idx) / get_v_cache(...)
get_current_context_length()
checkpoint() / restore()
```

`payload` is `void*` in the base class and a concrete `<model>_multi_modal_payload_t*` in
your engine. Define that struct, and any preprocessing parameter accessors, in the **public**
header — the application fills them.

Preprocessing parameters (patch size, sampling rate, soft-token budgets) come from
`processor_config.json`, not `config.json`; gemma4 reads them with a
processor→config→literal fallback chain (`_preprocess_value`). Do the same, so a checkpoint
packaged before `processor_config.json` existed still preprocesses correctly.

### 4.2 Registering the engine

1. `detail/<model>_npu/Makefile` (linux) and `CMakeLists.txt` (windows). Copy gemma4's.
   Two things in there are not boilerplate:
   * `-Wl,-Bsymbolic-functions`. Every engine `.so` compiles its own copy of the shared
     runtime and exports it with default visibility, and `flm` links every engine at once —
     without this, ELF interposition binds your internal calls to whichever engine comes
     first in `DT_NEEDED`, silently running another model's implementation.
   * per-directory object trees with `-MMD -MP`. A single compile-and-link step overwrites
     one `.d` per TU and header edits stop triggering rebuilds.
2. `build_<model>.sh` at the `FLM_DLL` root — a target list ending in your directory.
3. In `/path/to/FastFlowLM/src`:
   * `include/AutoModel/modeling_<model>.hpp` + `common/AutoModel/modeling_<model>.cpp` —
     `class <Model> : public AutoModel`, implementing `load_model`, `insert`, `generate`,
     `apply_chat_template`, `parse_*_content`, `get_chat_template_type`,
     `configure_parameter`. This is where tokenization, chat templating, image/audio
     decoding and sampling live.
   * `include/AutoModel/automodel.hpp` — `#include "models/<model>/<model>_npu.hpp"`.
   * `include/AutoModel/all_models.hpp` — add to `SupportedModelFamily`, add
     `{"<tag>", SupportedModelFamily::<model>}`, add the `make_unique<Model>` case.
   * `src/CMakeLists.txt` and `src/test/<model>/{Makefile,CMakeLists.txt,test.cpp}`.
4. `FLM_DLL/model_list.json` / `model_list_linux.json` — the tag, its files and sizes.

### 4.3 Where the xclbins go

`<exec_path>/xclbins/<model_name>/{layer,lm_head,dequant_mm,attn_global,attn_sliding}.xclbin`,
looked up by `npu_xclbin_manager::register_xclbin()`, which **dedupes by filename**. The
filename is the only link between the DLL and the `FLM_Xclbin/` tree, so make the overlay's
build rule emit the exact name the engine asks for. A mismatch fails at the first prefill,
not at startup.

`max_xclbins = 16`, so registering a third bitstream is free — but keeping three
`hw_context`s live is not. gemma4's prefill alternates bitstreams *every layer* (dequant_mm →
attention → dequant_mm), up to 96 context switches per prefill. If an array-side speedup
does not show up end to end, **measure the switch cost before touching the kernels.**

---

## 5. Build the CPU reference first. This is the method, not a nicety.

The single most valuable thing in the gemma4 engine is not any kernel — it is that
**every array stage has a host implementation with the same signature, selected at compile
time, and a harness that runs both and scores them.**

Do this in order. Do not skip ahead.

### Step 1 — a complete host model, before any xclbin exists

Write `namespace <model>_cpu` in `<model>_prefill.hpp` (and the elementwise ops in
`cpu_functions.hpp` under `namespace cpu_func`). It needs:

* `dequant_row_block()` / `block_offset()` — dequantize the on-device q4_0 layout back to
  fp32. **This is also your executable specification of the block format**, and writing it
  is how you find out you misread the memtile layout.
* `dequant_mm(y, w, x, L, K, N)` — the plain host matmul over that layout, OpenMP over row
  bands.
* `dequant_mm_upgate()` — the host equivalent of the fused pair, producing *exactly* what the
  two array runs leave behind (de-interleaved `[up | gate]`, GeLU already applied to gate) so
  everything downstream is shared between backends.
* `attention()` — grouped-query attention with explicit `q_pos_begin` / `kv_pos_begin` /
  `window` / `k_stride` / `v_stride`, so the same function serves causal, sliding, and the
  bidirectional vision-block case.
* `cpu_func::_rms_norm_batch`, `_rope_rms_batch`, `_residual_add_batch`,
  `_elementwise_scale_batch`, `_gelu_glu`, `_glu_mul`.

Get the numerics right against the reference implementation (HF / a torch dump) *here*,
with `PREFILL_MM=cpu PREFILL_ATTN=cpu`, which touches no xclbin at all and runs on a
device-less machine. Every semantic question — is the norm `x*rsqrt*w` or `(1+w)`? is
`1/sqrt(head_dim)` folded into `q_norm`? does V come before or after `k_norm`? — is answered
here, in code you can `printf` into, before any hardware is involved.

Give the host kernels the whole machine: `prefill_threads()` reads `omp_get_max_threads()`,
overridable with `FLM_PREFILL_THREADS`. This is deliberately *not* the decode thread count.

### Step 2 — bring up one array stage at a time

Add the overlay, add the `<MODEL>_*_ON_NPU` macro, and write the NPU path so it returns
`false` and falls through to the host kernel when it declines. Every array path in gemma4 is
host-fallback-clean: setting one macro to 0 by hand is a valid configuration needing no other
change. That gives you a bisection tool — flip stages until the wrong answer moves.

### Step 3 — turn on the cross reference

`<model>_prefill_xref.hpp`, `ENABLE_DEBUG` only. At `DEBUG_LEVEL=0` the header expands to
nothing and the prefill path is byte-for-byte the release one.

For a matmul, the replay hangs off `mm_run::wait()`: the run handle already captured
`{w, x, L, K, N}` at issue time, so `_cross_reference()` recomputes on the host and calls
`xref->check(name, layer, dst->data(), want.data(), N, N, rows)`. For attention there is no
run handle, so the replay goes *inside* `_attend_*_npu()` — and is gated on
`xref->enabled(layer)` **before** the pull, because pulling an `L × DQ` tensor on every layer
costs more than the attention it is checking.

Three things that make it usable rather than theoretical:

* **Bounds, both on by default, both widened from the environment.** `FLM_XREF_LAYERS`
  (`all` or a comma list; default = layer 0, the first layer of the other kind, and the last
  layer) and `FLM_XREF_ROWS` (default 256, `0` = all). Matmul rows are independent, so a
  prefix is a partial check rather than a weakened one. The default layer set is chosen:
  the first layers see an input both backends agree on by construction, so they score the
  matmuls in isolation; the last one is fed by 47 layers of accumulated divergence and is
  where a difference small enough to pass at layer 0 shows up.
* **`xref::guard`** — fingerprints the scored tensor before and after the block and asserts
  it was not modified. The cross reference must stay an observer: the *array* result has to
  be what reaches the rest of the model. Nothing in the type system enforces that, so this
  checks it instead of trusting it.
* **The `check()` stride trap**, documented at length in the header for good reason:
  `get_cosine_similarity` indexes `got` with a stride and `want` **packed**. A strided
  reference reads every row after the first from the wrong offset and scores as a stable,
  plausible-looking low cosine rather than an obvious failure. Compact your reference to
  `[rows][D]` first.

Report cosine / rel-L2 / max-abs per tensor per layer, green above 0.999. Scoring is off at
`DEBUG_LEVEL=1` (a plain `make`) because the replay is seconds per prefill; it turns on at
`DEBUG_LEVEL > 1` or at any level via `FLM_XREF_LAYERS`.

### What it does and does not prove

It proves the two backends agree. It does **not** prove either is right — that is Step 1's
job, against the upstream implementation. And it only covers stages with two
implementations: the norms, rope and the glu multiply are the same `cpu_func` call in both
builds, so a bug there is invisible. Know which of your stages are uncovered and say so in
your engine's README. For a sliding-window model the ring→window unroll is typically the
biggest uncovered surface; gemma4 mitigates it by having the sliding replay read the *same*
unrolled staging buffer, so a disagreement there is the attention and not the unroll.

Expect a looser cosine on attention than on the projections: `aie::mmul<8,8,8,bf16>` on npu2
truncates both operands onto a shared-exponent-per-8 (bfp16) grid for both `Q·Kᵀ` and `S·V`,
while the host reference is fp32 throughout.

### Why this beats an activation dump

The dump that used to fill this role could only score layer 0 of one fixed 1989-token
prompt, and had to replace the prompt and `exit(0)` to do it. The host kernel is a reference
that is always available, on the real prompt, at every layer, with no extra file next to the
model.

---

## 6. Recommended order of work

1. Read `detail/gemma4_12b_npu/README.md` end to end.
2. `<model>_npu_def.hpp`: parse config.json, lay out the weights, write the loaders. Dump
   the parsed fields at `DEBUG_LEVEL=1` and eyeball them against the checkpoint.
3. Public header + `Impl` skeleton + `load_weights`. Verify every descriptor `ready()`.
4. **The full CPU path** (§5 step 1). `PREFILL_MM=cpu PREFILL_ATTN=cpu`, no xclbin. Match
   the upstream implementation's logits on a short prompt.
5. Decode (`gen_layer_seq`) against `layer.xclbin`. It shares the weight layout you just
   validated.
6. Prefill matmuls on `dequant_mm.xclbin`. Turn on the xref (§5 step 3).
7. Prefill attention on its own overlay, per layer kind. xref again.
8. Multimodal projector, lazily initialized.
9. `AutoModel` wrapper, `all_models.hpp`, model_list entries, test target.
10. Write your engine's own `README.md` in the gemma4 style: what the config means in this
    backend's terms, the prefill structure, the gotchas you hit, and a table mapping each
    host constant to the IRON object and kernel source it corresponds to. That table is what
    the next session will actually use.

## 7. Gotchas that bit every engine

* Reordering a `*_all()` vector silently reorders device memory.
* Byte offsets vs bf16-element offsets. The `get_*_offset()` accessors return elements.
* `sync_to_device()` on a non-BO `buffer<T>` asserts. Guard every sync when the scratch may
  be host memory.
* A preload run left in flight keeps an xclbin resident; **drain it** before reassigning the
  `xrt::run` handle or the destructor tears down a command buffer the device is reading.
* `update_max_length()` must migrate existing caches into the new layout rather than dropping
  the conversation, and must reset the prefill context, whose apps are bound to the old
  buffers.
* Two overlays with identical run signatures are not interchangeable — different RTP
  addresses, floorplans and kv addressing. Feeding one's sequence to the other's context
  writes RTPs into whatever happens to live at that address.
* Multimodal rows must bypass the `sqrt(hidden_size)` embedding scale: apply it to text rows
  in `Embedding::forward()`, then overwrite the soft-token rows with unscaled projected
  features.
* Rebuild the engine `.so` and re-copy it into FastFlowLM's `src/lib/xrt/` before believing an experiment. A
  stale `.so` makes experiments lie.
