<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Qwen3.8-27B (MTP) engine

A CPU-first implementation of the Qwen3.8-27B hybrid stack with MTP speculative
decoding, structured so each of the three things it does can be replaced by
NPU kernels independently.

Model geometry lives in `qwen3_8mtp_desc` (`qwen3_8mtp_model_def.hpp`) and is
read from `config.json`; `cpu_ref::q38` holds the same numbers as compile-time
constants. 64 layers: layer *i* is gated full attention iff `(i+1) % 4 == 0`
(16 of them), gated delta-net otherwise (48). D = 5120, 24 q heads / 4 kv heads
of 256, intermediate 17408.

---

## The three modules

| # | module | what it does | replaced by |
|---|---|---|---|
| 1 | `prefill/` | batched sweep of a long prompt over the 64 layers, one token out | `mm.xclbin` + `attn.xclbin` + `gated_deltanet.xclbin` |
| 2 | `speculate/` | 1-layer MTP head drafts `MTP_STEPS = 7` tokens | `MTP_layer.xclbin` for steps 1–6; step 0 splits across `attn.xclbin` (core) + `dequant_mm.xclbin` (projections) |
| 3 | `verify/` | pushes the 8 verify rows through the stack, accepts a prefix | two fused `layer.xclbin` (attention, delta-net) |

Each is an abstract interface in `*_module.hpp` with one implementation in
`*_cpu.hpp`. The `_cpu` name is historical and now names only where the
*driver* runs: `prefill_cpu` and `verify_cpu` sweep the 64 layers and
`speculate_npu` walks the draft steps, but every matmul, every attention core
and both projections are device ops, and each of those files throws rather
than degrade when a bitstream is missing. All of it is in `namespace qwen38`;
`cpu_ref` is the ported fp32 reference math the device ops are scored
against, `q38npu` the NPU operators.

N2-N16 in the notes below name the step-verification drivers used during the
port (`tools/verify_nN_*.cpp`); they are not shipped, and the results quoted from
them are kept as the record.

```cpp
// prefill/prefill_module.hpp
void   begin_window(int L, const int32_t* pos_thw);  // scratch + m-rope
float* hidden();                                     // [L][D] in and out
void   run(int L);                                   // sweep 64 layers in place

// speculate/speculate_module.hpp
std::vector<int> draft(const std::vector<int>& ids, const float* prev_hidden,
                       int T, int pos0, int k);

// verify/verify_module.hpp
void             snapshot();   // before the batch
void             rollback();
void             discard();
std::vector<int> run(const std::vector<int>& ids, const int32_t* pos_thw);
const float*     hidden() const;
accept_result    accept_prefix(const std::vector<int>& argmax,
                               const std::vector<int>& drafts);   // free function
```

Modules 1 and 3 are two executors over one `qwen38::layer_stack`, which the
engine owns; module 2 owns its own private KV cache. Orchestration policy — the
catch-up window, the `mtp_pos_base` slide, headroom checks, the rollback
decision — stays in `Impl::speculate_inner` in `qwen3_8mtp_npu.cpp`.

## The layer seam

`common/layer.hpp` defines `decoder_layer`: `forward / reset / tokens_seen /
is_full_attention / set_max_length / save_state / restore_state / state_bytes`.
Two implementations exist, both on hardware —

- `q38npu::npu_attn_layer` (`verify/npu/`) — one gated full-attention block
- `q38npu::npu_delta_layer` (`verify/npu/`) — one gated delta-net block

The host adapters that used to sit beside them (`attn_layer_adapter`,
`delta_layer_adapter`) are gone, and so is `layer_backend`: the engine throws
without a device, and `layer_stack::build()` takes every bitstream path as a
required argument.

Module 2 has its own NPU driver, `q38npu::npu_mtp_layer` (`speculate/npu/`),
over `MTP_layer.xclbin`. The driver is matched to the scaffold under
`FLM_Xclbin/Qwen3_8/MTP_decoding` and proven so — the two hosts' live sequence
dumps are byte-identical at the same cursor, 41 896 words, and all five BO
sizes agree. Scored against the CPU head on hardware, it passed 29 of 29, from a cold window through a seeded context to the
16384-row cache ceiling.

It did not always. The design's first dispatch completed and every one after it
timed out; the cause was `enable_tb` in `iron/gate.py` building an MM2S chain
that `MTP_layer_aie.py` never routes, so the gate tile's producer lock ran pass
1 on its initial credit and was never recharged. Fixed in the IRON design (`iron/gate.py`), with the
history — including two confident wrong conclusions — in
`speculate/npu/SECOND_DISPATCH_TIMEOUT.md`.

`FLM_Q38_MTP_CPU=1` used to route the draft head to `speculate_cpu` while the
MTP bitstream's second-dispatch bug was open. It is gone. The CPU head was the
*faster* of the two, ~30 ms/step against ~34, but drafting is only 7–8% of
cycle time, and keeping a second full implementation of the head alive to buy
that back was not worth what it cost to maintain. `speculate_cpu.hpp` itself
stays — it is the reference to score `speculate_npu::draft()` against.

`common/layer_stack.hpp::build()` builds every layer on the device; there is
no backend to choose and no environment switch. The 48 delta-net layers reach
it by a different route — module 1 drives their bitstreams stage by stage and
module 3 swaps the layer type — but the outcome is the same, and a missing
`deltanet_layer.xclbin` now throws rather than falling back. The bitstream is
`<exec_path>/xclbins/<model_name>/attention_layer.xclbin`; every design in this
model builds to `layer.xclbin` and they share one flat directory, so each is
deployed under a distinct name (the runtime's
`test/qwen3_8mtp_npu/activate.sh` is the staging list). Modules 1 and 3
additionally keep small per-block structs (`prefill_attn_block`,
`prefill_linear_block`, `verify_linear_block`, …); module 1's two now drive the
device directly rather than calling `layer.forward()`, and they are where an
`npu_app` and per-block BOs go, exactly as
`qwen3_5vl_npu/qwen3_5vl_prefill.hpp` does.

### Prefill attention: a second bitstream over the same cache

`attention_layer.xclbin` is the *verify* design, `L_TOKENS = 8`, so
`npu_attn_layer::forward()` tiles a prefill window into `ceil(L/8)` dispatches
— 28 per layer at L = 221, 448 across the stack. Module 1 instead has a second
bitstream, `attention_prefill.xclbin` (`FLM_Xclbin/Qwen3_8/
gated_attention_prefill`), which computes **only** the causal GQA core,
`o = softmax(qKᵀ/√d + causal)V`, for a whole 128-aligned window in **one**
dispatch.

`prefill_attn_block::forward()` therefore runs the layer across two bitstreams,
with the host keeping only the elementwise stages:

```
in_norm()                             CPU  cpu_ref::attention_layer's own helper
prefill_proj::qkv_z()                 NPU  dequant_mm, q|k|v|gate woven
qkv_post_rope()                       CPU  same helper
npu_attn_layer::append_rows(k, v, L)  writes the shared cache BO, moves the cursor
prefill_attn_op::run(qh, L, base, …)  NPU  one dispatch
apply_gate()                          CPU  same helper
prefill_proj::o()                     NPU  dequant_mm
post_norm()                           CPU  same helper
prefill_proj::up_gate_swiglu()        NPU  dequant_mm, silu and the product fused
prefill_proj::down()                  NPU  dequant_mm
```

Four dispatches per 512-row tile plus one for the core. **There is no host
alternative.** Every one of these pieces is required: a missing bitstream
throws in `layer_stack::build()`, and `prefill_cpu`'s constructor re-checks all
64 layers and throws naming whatever is absent. That is deliberate — both a
device path and a host path produce fluent text, so a silent degrade used to
present as nothing but a prefill several times slower than it should be, which
nobody traces back to a file that was not staged.

Three things make the split safe to say plainly:

- **One cache, one cursor.** `prefill_attn_op` owns no cache. It is handed the
  BO `npu_attn_layer` allocated from the *other* xclbin's app, which is legal
  because BOs are device-scoped, not app-scoped. Verify and decode see rows
  prefill wrote and never learn which bitstream wrote them.
- **No duplicated reference math.** `cpu_ref::attention_layer::forward()` was
  split into named stages — `in_norm` / `qkv_post_rope` / `attn_core` /
  `apply_gate` / `post_norm` / `swiglu` — and the path above calls the ones it
  keeps by name. So the host stages *are* the scored reference rather than a
  copy of it, and `project_qkv_rope()`, `attn_core()`, `gate_out_mlp()` and
  `swiglu()` are now purely the reference the device stages are scored against.
- **The top of the buffer.** There is no apron: `plane_rows()` *is* `MAX_L`,
  the cache is two planes and nothing else, exactly as in `qwen3_5vl_npu`. A
  prefill pass streams kv through `round_up(L_begin + round_up(L, 128), 128)`
  rows — up to 128 past the cursor — and a DDR read past a BO does not fault,
  it reads the next layer's cache. So `prefill_attn_op::run` guards the
  **padded** window rather than the requested one, and `_kv_buffer_rows` makes
  `MAX_L` a whole `PREFILL_CHUNK`, which leaves that round-up nowhere to reach.
  Both halves are load-bearing, and the pitch they define is `v_at()` for
  *both* designs, which is why a change there needs the attention layer re-checked.
- **The context limit: 16377 tokens** (FLM#1001). The replay counts its
  16-row chunks in a 10-bit BD field, so the buffer tops out at 16384 rows
  (`attn_cache::max_context()`). `_kv_buffer_rows` keeps the old sizes up to
  8192 (a power of two plus 128 rows); above that it adds the 128 rows, rounds
  up to a 4096-row page and clamps to 16384, so 8193..12160 get 12288 rows and
  12161..16377 get 16384. Near 16384 there is less than a `PREFILL_CHUNK` of
  headroom, so `_prefill_with_mm` runs the rows a padded prefill window cannot
  take through the verify pass (`_verify_window`, 8 rows at a time), and
  `speculate()` declines once a whole 8-row verify pass no longer fits. The
  deepest verify pass starts at 16376, which makes `max_context_length()`
  16377; a larger request is lowered to it with a warning, and the runtime
  reads it back to cap its own context length.

Every L is a legal L. The core pads the window to its own `CHUNK` and
`dequant_mm` tiles at 512 rows and zero-pads the tail, so there is no short or
long window that goes anywhere else — the length gates that used to divert one
are gone.

### The draft head has two shapes too

The MTP head's seven steps are not one shape. **Step 0 is a window of `T`
tokens** — the catch-up that brings the head's private cache level with the base
stack — and **steps 1–6 are single tokens**. `MTP_layer.xclbin` moves exactly one
row per dispatch and streams ~250 MB of head weights to do it, so a `T`-row step
0 on that bitstream pays the weight stream `T` times, at ~34 ms each.

So `speculate_npu` splits step 0 off and gives it the prefill shape, over the
same cache BO and the same cursor as steps 1–6. Both halves are now on the
device — the attention core on `attn.xclbin` in one dispatch, the seven
projections on `dequant_mm.xclbin` as five (§N-MTP-PROJ):

```
fuse_pre()                            CPU  cpu_ref::mtp_head's own helper
mtp_proj::fc()                        NPU  dequant_mm       5120 x 10240
in_norm()                             CPU  same helper
mtp_proj::qkv_z()                     NPU  dequant_mm      14336 x  5120  [k|v|q|z]
qkv_post_rope()                       CPU  same helper
npu_mtp_layer::append_rows(k, v, T)   writes the cache BO, moves the cursor
mtp_prefill_attn_op::run(qh,T,base,…) NPU  ONE attn.xclbin dispatch
apply_gate()                          CPU  same helper
mtp_proj::o()                         NPU  dequant_mm       5120 x  6144
post_norm()                           CPU  same helper
mtp_proj::up_gate_swiglu()            NPU  dequant_mm      34816 x  5120  woven, silu on the gate bands
mtp_proj::down()                      NPU  dequant_mm       5120 x 17408
mtp.norm                              CPU
```

`up_gate_swiglu` is one dispatch and no host activation: the sequence writes
the kernel's `rtp_activate` **twice**, `none` before the even output-column
blocks and `silu` before the odd ones, so the device silu's the gate half of
the weave and leaves the up half alone (§N-MTP-SWIGLU). The product folds into
the un-weave the host was already doing. `cpu_ref::mtp_head::swiglu` stays
where it is, as the reference to score this against.

`base` is still read **before** `append_rows` — unchanged and load-bearing.
The attention core can still be withheld (`FLM_Q38_MTP_PREFILL=0`, below); the
projections cannot — `FLM_Q38_MTP_PROJ` is gone and `dequant_mm.xclbin` is now
a required argument to `speculate_npu`. Step 0's five projections are still
scored against `cpu_ref::mtp_head`'s helpers, which is what that switch
was actually for.

It is the same operator module 1 uses — `prefill_attn_op_t<Layout>` and
`gen_prefill_attn_seq<Layout>` are templated on the layout, with
`prefill_attn_op = ...<attn_layout>` for module 1 and `mtp_prefill_attn_op =
...<mtp_layout>` for module 2. The two layouts agree field-for-field on
everything the generator reads; keep it that way, nothing asserts it at build
time.

Three things to know before touching it:

- **`mtp_layout::MAX_L` is the whole layout.** `plane_rows()` returns `MAX_L`,
  `v_at(0)` is `MAX_L * DK` and `kv_cache_size` is `2 * MAX_L * DK` — the same
  two-plane shape `attn_layout` and `qwen3_5vl_npu` carry, with no apron inside
  the pitch. `build()` throws unless `MAX_L % PREFILL_CHUNK == 0`; that is what
  makes both round-ups (the replay's to 16, step 0's to 128) land *on* the top
  of the buffer rather than past it. Changing either moves `v_at()` and every
  descriptor `gen_mtp_layer_seq` emits, so re-checking the draft head on
  hardware is required, not optional.
- **`speculate_npu::cpu_half_` is a `cpu_ref::mtp_head` built with `max_len =
  0`.** Its cache is never allocated and never read: only `fuse`,
  `project_qkv_rope`, `gate_out_mlp` and `out_norm()` are called on it, and
  `attn_core()` — the one helper that touches the cache — is the one the
  bitstream replaces. Do not "fix" the 0; it is what turns an accidental route
  back through `step()` into a throw instead of a corrupted second cursor.
- **`base` is the cursor *before* the append.** `attn.cc` masks on absolute
  position, so a ragged `L_begin` is fine, but the post-append value would let
  every query attend to its own future.

**`FLM_Q38_MTP_PREFILL=0`** withholds the bitstream and forces the per-row path;
`FLM_Q38_MTP_PREFILL_MIN` overrides the width-3 floor. The floor is drawn so
that only cycle 1's step 0 — the prompt-wide catch-up — takes the batched shape.
From cycle 2 the head is cropped to `C - 1`, so step 0 is width 1 on a partial
accept and width 2 on a full one, and both stay on the decode bitstream at one
`MTP_layer.xclbin` dispatch per row. `=2` puts the width-2 case back on the
hybrid arm, which is the A/B worth running. Both compute the same thing — the
head's hit rate is identical across the switch — so this is a throughput switch,
not a correctness fallback.

For most of this engine's life it bought almost nothing on the live path, and
the reason was not this code. `Impl::speculate_inner` capped the catch-up at
`last_window_len`, and the app runs a `forward()` between `prefill()` and the
first `speculate()` — decode being a one-row prefill, that left the cap at 1. So
step 0 ran 1–4 rows wide against a head 315+ rows behind, below both floors,
while every in-tree driver saw the wide path because the drivers call
`prefill()` then `speculate()` with nothing in between.

The rows were never wrong; they were never kept. `_prefill_with_mm` now captures
the window's normed hidden states into `Impl::mtp_hist`, which holds context
positions `[mtp_hist_base, current_context_length)` — every row the head has not
folded — and `speculate_inner` primes from that instead. Cycle 1 is therefore
the full prompt width; `crop(C-1)` then does what its comment always claimed and
holds the steady state at width 1–2. Build with `DEBUG_LEVEL=1` to see
`step-0 width N (unclamped)` per cycle; a `CLAMPED` line outside a `restore()`
means a commit path appended ids without appending hidden rows.

## The DFlash2 drafter (optional, replaces the MTP head)

`speculate/dflash2_draft.hpp` runs z-lab's **Qwen3.8-27B-DFlash2** block-diffusion drafter in place of the MTP head, over
the same kernels. It is switched on by a draft package: the catalog's `draft_model_path` (FLM tag `qwen3.8-dflash2:27b`,
whose `base_model` is `qwen3.8-mtp:27b`) or, for a dev run, `FLM_DFLASH_PATH=<package dir>`. The package holds
`dflash_weight.q4nx` and the draft's `config.json`; `FLM_Q4NX_Converter/convert_dflash2.py` makes it from the HF checkpoint.
When the drafter is loaded the engine does not build the MTP head (`FLM_DFLASH=0` ignores the package and builds it
instead); verify, commit, rollback, the prefill and the vision tower are unchanged, and **no bitstream is new**.

What the drafter is (reference: `z-lab/dflash` `DFlash2DraftModel`):

- 5 decoder layers (32 q heads / 8 kv heads x 128, MLP 17408, sliding window 2048, no causal mask inside the block).
- It conditions on the **target**: the residual stream after target layers 5, 19, 33, 47, 61 for every committed token
  (`common/tap_buffer.hpp`, filled by the prefill and verify sweeps right after those layers). The 5 x 5120 features go through
  `fc` and `hidden_norm`; every draft layer takes K/V of them directly (no input norm) and caches those rows across cycles.
- One pass drafts the block: the input is the target's embedding of `[anchor, mask x 7]`, the 8 rows attend to the context rows
  and to each other, and the outputs go through the **target's lm_head** (its own `norm.weight` replaces `model.norm`).
- Two-tap dynamic convolutions around attention and around the MLP of every layer: `out[t] = sum_{o<2} (base[o] + dyn[t][o][g])
  * x[t-o]` over the 8 block rows, `dyn` = a 5120 -> 1280 projection of the layer's normed input (groups of 16 channels).
- A candidate selector: top-16 logits per position, `score = logit + <pred_cb[prev] * W h, succ_cb[cand]>`, greedy argmax,
  `prev = chosen` (two 248320 x 256 codebooks, a few rows gathered per step).

Where it runs:

| part | where |
|---|---|
| `fc`, K/V of the context rows (all 5 layers, one dispatch), per layer q\|k\|v, o, up\|gate (silu on the gate bands), down | `dequant_mm.xclbin`, Q4_K, a fixed 512-row tile each: 22 dispatches per cycle, about 8 ms each, whatever the 8 live rows |
| the draft's lm_head | `lm_head_8.xclbin`: the bitstream norms its input itself, so it gets the pre-norm hidden and `norm.weight` of the draft |
| attention (8 queries over the window), the convolutions, norms, rope, the 1280-wide dynamic-kernel projections (not a multiple of the 512-wide array tile), the selector | host |

Ledger: the drafter keeps its own rows (`append` / `truncate` / `reset`), tied to the committed token log. `_prefill_with_mm`
and `prefill_verify` hand it the taps of the rows they commit (`_df_append`), `_trim_committed` truncates it, and a saved
state (prefix cache) or `clear_context()` resets it: a state carries no taps, so the drafter starts cold and refills from the
rows that follow. A row that does not follow the last one also resets it (a commit path forgot to report rows): that costs
acceptance and never correctness, because verify overrides every draft. Rows older than the window are never read, so a long
prompt keeps only its last 2112 rows of taps (`tap_buffer::keep_rows`).

Accuracy against the reference, on engine dumps (`FLM_DFLASH_DUMP=<dir>`, replayed by `tools/dflash2_ref.py`): the selected
path agrees with the reference token for token, and the per-layer residual stream stays within 1 to 4% relative L2 of an fp32
run of the same Q4_K-dequantized weights through layers 0 to 3. Layer 4's MLP cancels the model's massive residual
activations (|h| about 1.4e5 at one dim, its down-projection output about 1.4e4, scaled by about -8.6 by the post-MLP
convolution), so a few percent of error in that one output shows up as 5 to 60% in the final hidden state. Sending the NPU the
input as a bf16 hi + lo pair did not change it (the error is not the input rounding; probably the array's shared-exponent bfp16
matmul), and the drafted tokens are not affected beyond what the acceptance numbers show.

| switch | what it does |
|---|---|
| `FLM_DFLASH_PATH` | the draft package for a dev run (the catalog sets `draft_model_path` for the tag) |
| `FLM_DFLASH=0` | do not load the drafter: the MTP head the base checkpoint carries is built instead |
| `FLM_DFLASH_PROF=1` | per-cycle draft breakdown on stderr (context fold, NPU layers, host layers, lm_head, selector), every 20 cycles |
| `FLM_DFLASH_DUMP=<dir>` | write the per-cycle inputs, per-layer hiddens, candidates and drafts the reference script replays |

## Layout

```
qwen3_8mtp_npu.cpp            engine: wires the modules, owns the policy
qwen3_8mtp_npu_detail.hpp     struct Impl
qwen3_8mtp_model_def.hpp      qwen3_8mtp_desc
common/  layer.hpp  layer_stack.hpp  window.hpp  tap_buffer.hpp
         embedding.hpp  lm_head.hpp  rope.hpp
         cpu_ref_common.hpp  cpu_ref_bridge.hpp  cpu_simd.hpp
         cpu_ref_attention_layer.hpp  cpu_ref_delta_net_layer.hpp/.cpp
         npu/ lm_head_op.hpp  lm_head_sequence.hpp/.cpp  lm_head_weights.hpp
              lm_head8_op.hpp  lm_head8_sequence.hpp/.cpp
prefill/ prefill_module.hpp  prefill_cpu.hpp
         npu/ prefill_attn_op.hpp  prefill_attn_sequence.hpp/.cpp
              prefill_proj.hpp  conv1d_sequence.hpp
              delta_net_op.hpp  delta_net_sequence.hpp/.cpp
              delta_proj.hpp  delta_weights.hpp
speculate/ speculate_module.hpp  speculate_cpu.hpp  speculate_npu.hpp
           mtp_head.hpp/.cpp
           dflash2_draft.hpp            (the DFlash2 drafter, see its section)
           npu/ mtp_layer.hpp  mtp_sequence.hpp/.cpp  mtp_weights.hpp
                mtp_proj.hpp  proj_op.hpp  proj_sequence.hpp/.cpp
                SECOND_DISPATCH_TIMEOUT.md   <-- solved; kept for the two false trails
verify/  verify_module.hpp  verify_cpu.hpp
         npu/ attn_layer.hpp  attn_sequence.hpp/.cpp  attn_weights.hpp  q4nx.hpp
              delta_verify_layer.hpp  delta_sequence.hpp/.cpp
image/   image_module.hpp  image_cpu.hpp/.cpp  vision_model_def.hpp
         vision_glue.hpp/.cpp  vision_ops.hpp/.cpp  vision_rope.hpp/.cpp
         vision_weights.hpp  mrope.hpp/.cpp  README.md
         npu/ vision_attn_op.hpp  vision_attn_sequence.hpp/.cpp
              vision_mm_op.hpp  vision_mm_sequence.hpp/.cpp
              vision_op_factory.hpp  README.md
tools/   verify_qwen3_8mtp.cpp  profile_prefill.cpp  profile_vision.cpp
         bench_lm_head_rows.cpp  bench_verify_lm_head.cpp
         verify_cpu_simd.cpp            (have a main(); not globbed)
         dflash2_ref.py                 (fp32 reference of the DFlash2 drafter, replays FLM_DFLASH_DUMP dumps)
```

`speculate_cpu.hpp` and the `cpu_ref_*` files are references, not backends:
nothing in the engine reaches them as an alternative to a device op. What the
engine still calls on the host is the *stage* helpers inside them — `in_norm`,
`qkv_post_rope`, `apply_gate`, `post_norm`, `fuse`, `project_qkv_rope` — which
is the point of having split `forward()` into named pieces rather than copying
it.

Internal includes are directory-rooted (`"common/layer.hpp"`); both build files
add this directory as an include root.

## Invariants a replacement must preserve

- **The RMSNorm `+1` is already folded into the q4nx norm weights.** Adding it
  again is a silent ~2x scale error.
- **The MTP head's cache trails the base cache by exactly one** (`mtp_pos_base
  = 1`). `pos0` passed to `draft()` is the head's absolute position, not the
  engine's context length.
- **`prev_hidden` is post-`model.norm`**, not the layer-63 tap. The wrong tap
  costs acceptance rate (cos 0.94 vs 1.0), not correctness.
- **`snapshot()` must precede the verify batch.** The delta-net fold
  `S_t = e^{g_t} S_{t-1} + k_t(...)^T` has no inverse; the copy is the only way
  back. Attention rolls back by moving a cursor (4 B), delta-net by a full
  state copy (~3.3 MB/layer).
- **`accepted + 1` tokens commit** — the accepted drafts plus the free base
  token at the first mismatch.
- **RTP addresses are per-build.** If an xclbin is rebuilt, diff its
  `rtp_addresses.txt` against `q38npu::attn_rtp_addresses`; a stale address
  hangs in `wait()` rather than faulting.

## Environment switches

Almost every switch that chose between a device op and a host copy of the same
op is gone, along with the host copy. What is left either picks between two
device paths, turns a whole feature off, gates threading, or — in the one case
below — picks the host arm because at the width that actually occurs it is
five times faster:

| switch | what it selects | why it survived |
|---|---|---|
| `FLM_Q38_MTP_PREFILL=0` | draft step 0 on the decode bitstream, one row per dispatch, instead of `attn.xclbin` | both arms are hardware; a throughput choice |
| `FLM_Q38_MTP_PREFILL_MIN` | the width floor below which step 0 is not worth batching (default 3) | drawn so only cycle 1's prompt-wide catch-up batches; cycle 2 onward is width 1 or 2 and stays per-row. `=2` restores the old floor and is the A/B |
| `FLM_Q38_MTP_PROJ_MIN` | the width floor below which step 0's five projections run on the host instead of `dequant_mm` (default 1, i.e. off) | was 25, because `dequant_mm` is flat at ~52 ms below one 512-row tile and live widths were 1–4. With the step-0 clamp lifted, cycle 1 is prompt-wide (11–19× to the device) and every later cycle is width 1–2 and never reaches the prefill shape at all, so the floor has nothing left to protect. Set it high to restore the split arm (the host reference for step 0's projections) |
| `FLM_Q38_DELTA_SYNC` | how module 1 syncs the delta-net state BOs | both arms are hardware |
| `FLM_Q38_FAST_ROLLBACK` | `commit_prefix()` vs re-prefilling the accepted prefix | an algorithm choice; both arms already drive the device |
| `FLM_Q38_VISION=0` | the vision tower off entirely, for a text-only deployment | not a backend choice — the feature, not its implementation |
| `FLM_Q38_VISION_FP32`, `FLM_Q38_VISION_MM_PREROUND` | numerics inside the tower | not backend choices |
| `FLM_Q38_HOST_PAR=1`/`=0` | overrides `host_par_min()`'s OpenMP wait-policy detection | threading, not placement |
| `FLM_Q38_PIPELINE=0` | serialises `dequant_mm_op`'s host work against its dispatches instead of overlapping them | scheduling, not placement: the device order is identical either way, so the two arms are bit-identical. A kill switch for the second x/y slot, which costs up to 41 MB per op |

Removed, with the host arm they selected: `FLM_Q38_MTP_CPU`,
`FLM_Q38_LM_HEAD`, `FLM_Q38_LM_HEAD_8`, `FLM_Q38_VISION_NPU`,
`FLM_Q38_VISION_MM`, `FLM_Q38_DELTA_VERIFY`, `layer_backend::cpu`, the
`MVPREFILL` build flag, and `Impl::project()`. `FLM_Q38_MTP_PROJ` is removed
too, but its host arm is **not**: `speculate_npu::_prefill_step_cpu_proj` is
still built and still reached, by the width floor in the table above rather
than by a switch. A missing bitstream now throws
naming the file, rather than degrading to a host path that produces equally
fluent text several times slower — which is the failure nobody traces back to a
file that was not staged.

The reference implementations those switches used to reach are still here and
still built: `speculate_cpu.hpp`, `cpu_ref::lm_head`, `vision::cpu_mm_op` /
`cpu_attn_op`, and all of `cpu_ref::*`. They are the reference the device was
scored against. The engine does not reach them.

## Build and run

```bash
cd FLM_DLL/detail/qwen3_8mtp_npu && make -j8     # -> ../../build/lib/libqwen3_8mtp_npu.so
```

The milestone driver has no Makefile rule (`tools/` has a `main()`). It takes a
model directory and an `exec_path` — the root holding
`xclbins/<model_name>/` — and needs a device, because the attention layers are
not optional. `-lq4_npu_eXpress` is required:

```bash
g++-13 -O2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -std=c++20 \
    -I. -I../../include -I../q4nx -I$XRT_ROOT/include \
    tools/verify_qwen3_8mtp.cpp \
    -L../../build/lib -lqwen3_8mtp_npu -lq4_npu_eXpress \
    -L$XRT_ROOT/lib -lxrt_coreutil \
    -Wl,-rpath,$PWD/../../build/lib \
    -o /tmp/verify_qwen3_8mtp

/tmp/verify_qwen3_8mtp /path/to/models/Qwen3.8-27B-NPU2 \
                       /path/to/FastFlowLM/src      # ~10 min
```

Gates, in order: **M9** argmax matches `verify.argmax` exactly (8/8); **M15**
≥6/7 drafts match `mtp.draft_tokens`; **M17** greedy output is identical with
and without `speculate()` — with one exception it states in full, an **exact
bf16 tie** in the logits, where "the argmax" is a fact about which reduction
ran first rather than about the model; **M19** cache bookkeeping holds over ≥4
cycles.

### Profiling module 1

`make PREFILL_PROFILE=1` turns on a per-stage tick in both prefill blocks and a
set of counters in `dequant_mm_op`. Every stage line reports its wall clock, how
much of it a bitstream was running, and the remainder as host time; the window
line rolls the counters up. `-DQWEN3_8MTP_PREFILL_PROFILE` is a `-D` flag with
no make dependency, so **`make clean` first** — the same trap as `DEBUG_LEVEL`.

`tools/profile_prefill.cpp` drives it. `verify_qwen3_8mtp` cannot: its window is
pinned to the reference dump at L = 221, which is below `dequant_mm_op::ROWS`
(512), so every projection runs as one partly-padded tile and the profile has
the wrong shape. This one takes the window length as an argument and runs as
many rounds as asked, so a resumed window gets measured too.

```bash
g++-13 -O2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -std=c++20 \
    -I. -I../../include -I../q4nx -I$XRT_ROOT/include \
    tools/profile_prefill.cpp \
    -L../../build/lib -lqwen3_8mtp_npu -lq4_npu_eXpress \
    -L$XRT_ROOT/lib -lxrt_coreutil \
    -Wl,-rpath,$PWD/../../build/lib -o /tmp/profile_prefill

/tmp/profile_prefill [model_dir] [exec_path] [L] [rounds]
```

Its ids are synthetic, and that is not a shortcut: token identity does not
change which matmuls run over which shapes. It scores nothing — correctness is
`verify_qwen3_8mtp`. The one thing that has to be real
is `L`, and that is the argument.

**Where the time goes**, L = 1024, per 64-layer window. There are two numbers,
because the host half still depends on the OpenMP wait policy the *process* was
started under — **8.67 s, 8.46 ms/token** with libgomp's barriers asleep and
**9.75 s, 9.52 ms/token** on the default spinning ones. See "The wait policy
decides 8%" below.

The per-line table below is the 8.98 s breakdown, i.e. everything up to and
including the fifth bullet. The sixth took the passive column to 8.67 s and the
seventh to 8.07 s; the lines they moved were not re-measured individually, so
the rows still sum to 8.98 s. Treat the shares as the shape of the cost, not as
the current total:

| | ms | share |
|---|---:|---:|
| device, `dequant_mm` (512 tiles) | 5362 | 59.3% |
| device, conv + fold + attention core | 1771 | 19.6% |
| host, y widen + y read | 262 | 2.9% |
| host, fp32 → bf16 x staging | 219 | 2.4% |
| host, scatter / un-weave of a fused output | 202 | 2.2% |
| host, `alpha_beta` | 166 | 1.8% |
| host, everything else (norms, RoPE, gates, the fold's host half) | 1053 | 11.7% |

**Device 78.9%, host 21.1%.** These rows were measured serially, and with
`FLM_Q38_PIPELINE` on (the default) **they no longer sum to wall clock**: the
staging, widen and scatter lines run *under* a dispatch, so their ticks are
double-counted against the device line rather than added to it. Read them as
what each stage costs, not as what it contributes. `FLM_Q38_PIPELINE=0` restores
a profile whose rows sum.

Cost per token is flat from L = 512 up and
substantially worse at L = 256, because `dequant_mm` tiles at 512 rows and
zero-pads: L = 256 and L = 512 issue the same 256 tiles and take the same
dispatch time. A window that is not a multiple of 512 buys padding.

This replaces an earlier 14.2 s / 13.9 ms/token profile that was 50% device /
50% host. Seven changes closed that gap, all output-neutral — `verify_qwen3_8mtp`
is byte-identical across every one of them apart from its timing lines, checked
against a worktree build of HEAD:

- **The zero-fill line is gone.** Every `assign(n, 0.0f)` on this path sized a
  buffer that the very next dispatch or scatter loop then wrote in full, so it
  is a `resize(n)`. 995 ms → 0 (the first window still pays ~48 ms to grow).
  See `speculate/npu/proj_op.hpp` for the per-call-site write-coverage argument
  a future partial writer would have to re-establish.
- **`alpha_beta` is threaded**, 1072 → 173 ms. It is two `dense_matmul` calls
  and it was the only unthreaded hot loop left on the host path. Blocking is
  *not* the lever here and was tried and removed; see the comment on
  `dense_matmul` in `common/cpu_ref_common.hpp` before re-deriving it.
- **The fp32 intermediates between two dispatches are gone** — the activation
  port below. 1447 + 1310 ms of staging and scatter → 614.
- **The marshalling that is left is vectorised** — `common/cpu_simd.hpp`, below.
  One vectorised core beats all 32 scalar ones by 4–8x, and it is the change
  that took the serial column from 11.50 s to 9.75 s.
- **…and then threaded**, conditionally, on top of that. That is the 8.98 s
  column against the 9.75 s one. It used to be worth 2.5 s and it is worth
  0.77 s now, because vectorising removed most of what the threads were for.
- **The four marshalling sites the first pass missed**, 8.98 → 8.67 s. The
  `assign`/vectorise/thread treatment above had only reached the `dequant_mm`
  op; `prefill_attn_op`'s q narrow and o widen, `delta_net_op`'s o widen (the
  widest one left — 48 layers x L x 6144) and its two conv-window shifts, and
  `verify/npu/attn_layer`'s k|v append were all still the scalar
  `bfloat16_t(float)` constructor one element at a time. The conv shifts also
  collapse to single contiguous `q38simd` calls, because `conv_state_` is
  `[conv_k][conv_dim]` k-major and the nested loops were only expressing the
  one-row shift.
- **The host work now runs under the dispatch**, 9.57 → 9.23 s spinning and
  8.23 → 8.07 s passive at L = 1024. Everything above made the host half
  *cheaper*; this one stops paying for it at all. `dequant_mm_op` keeps two x/y
  slots and `run_tiles()` stages tile *t+1* and consumes tile *t−1* while tile
  *t* is on the device, so staging and the widen/scatter disappear into the
  dispatch they sit next to. The device issue order is unchanged — `hrx::run`
  waits on one global stream, so there is never more than one dispatch in
  flight, and the whole point is that the *host* moves, not the device. That is
  why it is bit-identical rather than merely close. Gains more on the spinning
  policy, because that is the column with more host time to hide.

  The second slot is allocated lazily and only on a path with more than one
  tile, so decode and a step-0 speculate never pay it: +140 MB RSS on a 36 GB
  window. `FLM_Q38_PIPELINE=0` turns it off. The un-weave in `mlp` stays
  exposed and is meant to: `down(t)` reads what it writes, so covering it would
  mean running `upgate(t+1)` ahead of `down(t)`, a three-deep pipeline for the
  smallest of the three host terms.

Per-layer inter-stage buffers are also block-struct members now rather than
locals, which drops a malloc/free of up to 34 MB per layer per window. And the
per-layer hidden/output carry is a `std::swap` of the two `window_buffers`
vectors rather than a 21 MB `memcpy`, 64 times a window: the reference layers
cannot write in place, but they do write all `L x D` of `y`, so the two buffers
can simply trade names. Swapping rather than alternating on layer parity keeps
`hidden()` unconditionally `bufs_.x` for an odd layer count too.

#### The wait policy decides 8%

Same binary, same window, four runs (bold is what the detection picks):

| | default (spin) | `OMP_WAIT_POLICY=passive` |
|---|---:|---:|
| marshalling serial   | **9.75 s** | 9.51 s |
| marshalling threaded |   11.51 s  | **8.98 s** |

(Measured before the sixth and seventh bullets, which move the bottom-right
cell to 8.67 and then 8.07 s, and the top-left one to 9.23 s. The gate's *sign*
is what this table is for and neither changes it — the sixth widens the same
conditional threading to four more sites, and the seventh moves host work under
a dispatch on both policies, by more on the spinning column because that is
where there is more of it to hide.)

Vectorising fixed the serial row and left the threaded/spinning cell where it
was: 11.50 → 9.75 s serial, 11.67 → 11.51 s threaded. That is the whole reason
the gate still exists — threading costs 1.76 s on the default policy and saves
0.53 s on the passive one, so the right answer genuinely depends on the
environment.

The host loops themselves are ~3x slower in the spinning column (scatter 582
vs 202 ms, x stage 299 vs 219, widen 205 vs 153) while the device barely notices
(dispatch 5375 vs 5362). The shape of this path is what does it: a marshalling
loop, then a **blocking** NPU dispatch of ~10 ms, ~1500 times per window. On the
default policy libgomp leaves the team spinning after every parallel region, so
31 cores burn through every dispatch, and what they burn is the power and
frequency budget the next loop needs. A `#pragma omp parallel for` microbenchmark
with no dispatch in it has the opposite sign — 124 ms spinning against 709 ms
passive over 20000 tiny regions — so this is not a general claim about wait
policies, it is a claim about interleaving OpenMP with a blocking accelerator.

**The library cannot set the policy.** libgomp reads `GOMP_SPINCOUNT` and
`OMP_WAIT_POLICY` in its own ELF constructor, before any constructor of ours and
long before `main`; `setenv` from either has no effect, which is measured rather
than assumed. So `q38npu::host_par_min()` *detects* it instead and threads the
marshalling only when it will pay, and `prefill_cpu`'s constructor prints which
way it went (`[prefill] host marshalling threaded` / `... serial`).
`FLM_Q38_HOST_PAR=1` / `=0` overrides the detection.

Whoever owns the process launch gets the other 8% by exporting
`OMP_WAIT_POLICY=passive`.

### Porting 5vl's activation dataflow (intra-layer done; the residual is fp32)

**What 5vl does.** Every intermediate is a bf16 BO, allocated once per
`L_padded` and handed layer to layer: `common_bufs.x`, `qkv`, `conv_qkv`,
`gate`, `after_mix`, `alpha_beta` (`qwen3_5vl_prefill.hpp:111-224`). Host
stages mutate the BO's bf16 **in place** — `_gated_norm` writes into
`after_mix.data()`, then `sync_to_device()`, and the next dispatch reads the
same buffer. The residual is carried in bf16 too (`:499`). Nothing is ever
converted, because nothing is ever fp32.

**What this module used to do.** `dequant_mm_op` owns private `x_`/`y_` BOs and
its `run()` is fp32, so every dispatch paid a scalar `xd[i] = (bf16)x[i]` going
in and a widen coming out, and every `prefill_cpu.hpp` stage handed it a fresh
`std::vector<float>`. That was the x staging line and the scatter line. They
existed because the block structs were written against
`cpu_ref::attention_layer`'s fp32 signatures, not because anything requires
fp32.

**The three things that looked like blockers, and were not.**

1. *A BO must come from the app that dispatches it.* False, and the comment in
   `proj_op.hpp` that used to say so has been corrected.
   `npu_app::create_bo_buffer` is `buffer<T>(*this->device, size)` — a BO binds
   to the **device**, not to the `hw_context`. 5vl allocates `qkv` from an
   `mm.xclbin` app and hands it to `conv.xclbin`'s
   (`qwen3_5vl_prefill.hpp:122`, `:161`) in production.
2. *The sequence is generated at a fixed 512 rows.* `gen_dequant_mm_seq`
   already takes `L` as a parameter; this module just always passes `ROWS`.
   5vl regenerates only when `L_padded` changes.
3. *No precedent for the hand-off here.* `delta_proj::qkv_z_to_bo` already
   implemented exactly this idiom for one hand-off.

**What is done.** `dequant_mm_op` has the bf16 entry point —
`stage()` hands out its x BO, `run_staged()` dispatches what the caller wrote
there, `run_tile()` returns the y BO as bf16 — and `run()` is the fp32 wrapper,
kept because the host half of a layer genuinely is fp32.

Every intra-layer hand-off that was provably a round trip now uses it:

- **The SwiGLU.** `up|gate` came off the device as bf16, was widened into an
  `[L][17408]` fp32 vector, multiplied, and immediately re-narrowed by `down`'s
  staging. At L = 1024 that is 71 MB written and 71 MB read, 64 times a window,
  to get back the value it started from. `prefill_proj::mlp` / `delta_proj::mlp`
  now un-weave straight from the y BO into `down->stage()`. Bit-identical: the
  product is computed in fp32 from the same two bf16 operands and rounded once
  either way.
- **The `q|k|v|z` scatter**, driven a tile at a time off the bf16 y BO instead
  of through an `[L][14336]` fp32 staging vector.
- **The residual adds.** `o()` and `mlp()` take a `resid` pointer and fold it
  into the widen, so `y[i] = (float)out[i] + resid[i]` replaces a widen plus a
  separate `y[i] += h1[i]` pass over the same 5120-wide rows.

Worth 1447 + 1310 → 614 ms serial, and the loops that are left both vectorise
and thread — see "The conversion itself" below.

**What is not done, deliberately.** The inter-layer residual is still fp32.
`common/window.hpp` states the policy: "Everything is fp32; rounding between
layers would add error the NPU port has not committed to yet." Putting bf16 on
that residual is a real change of it, and it is not free — the 64-layer chain
already ends at cos 0.945 (`layer_63_out`), and the gate is M9 argmax, not
cosine. 5vl ships this way, so it is known to be survivable for *a* model; it is
not known for this one. Decide it on M9 over the full `verify_qwen3_8mtp` set
and a long-window run, not on M7.

#### The conversion itself: `common/cpu_simd.hpp`

**The lever that was left** was the conversion. `biovault::bfloat16_t(float)` is
a `std::fpclassify` switch (`biovault_bfloat16.h:124`): branchy, so gcc could not
vectorise a single one of these loops. The un-weave moves ~7 GB a window and ran
at 4 GB/s serial, an order of magnitude under DDR — this path was ALU- and
branch-bound, not bandwidth bound.

`common/cpu_simd.hpp` is the branch-free version, as five AVX-512 wrappers over
the marshalling shapes this module actually has (`f32_to_bf16`, `bf16_to_f32`,
`bf16_to_f32_add`, `bf16_mul_to_bf16`, `bf16_mul_to_f32`). One core against all
32 scalar ones, on a `[512][17408]` tile:

| | scalar, 32 threads | vectorised, 1 thread |
|---|---:|---:|
| x stage (fp32 → bf16) | 6.39 ms | 1.47 ms |
| y widen (bf16 → fp32) | 6.18 ms | 1.99 ms |
| un-weave (bf16 × bf16 → fp32) | 6.20 ms | 1.06 ms |

**It does not change a type the whole tree shares**, which is what made this its
own change before. Nothing in `biovault_bfloat16.h` moves; these are loop bodies,
and the scalar tail of each wrapper is still the constructor.

**It is bit-identical, and that is checked rather than asserted.**
`tools/verify_cpu_simd.cpp` pushes all 2^32 float bit patterns through
`f32_to_bf16` against `(bf16)f` — 0 differ — then `memcmp`s every wrapper against
its longhand scalar body at lengths 0..129 and at the real widths, which is what
covers the 16-lane tail. 0.45 s, no device, no model:

```bash
g++-13 -std=c++20 -O3 -fopenmp \
  -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -ffast-math \
  -I. -I../../include -I../q4nx \
  tools/verify_cpu_simd.cpp -o /tmp/verify_cpu_simd
/tmp/verify_cpu_simd          # ends "== bit-identical =="
```

Two things that fell out of writing it, and that a future edit has to respect:

- **5vl's `f32o_bf16_512` (`typedef.hpp:123`) truncates; this module rounds to
  nearest even.** So the obvious move — reuse 5vl's helper — silently changes
  numerics on every activation. `cpu_simd.hpp` re-derives the narrowing instead:
  add `0x7FFF + lsb`, then shift.
- **`-ffast-math` decides what "identical" means, so build the checker at the
  module's flags.** `-ffinite-math-only` folds the constructor's `FP_NAN` and
  `FP_INFINITE` arms away, so a masked NaN fixup is *more* correct than the code
  it replaces and therefore wrong. It does **not** fold `FP_SUBNORMAL`, so the
  zero/denormal arm is real and needs its `vfpclassps` mask. Exhaustive runs
  found both, at 12.6M and 16.7M mismatches; nothing short of exhaustive would
  have.

Runtime end-to-end:

```bash
cp ../../build/lib/libqwen3_8mtp_npu.so /path/to/FastFlowLM/src/lib/xrt/
cd /path/to/FastFlowLM/src/build && ninja flm
```

`REPLACEMENTWORK.md` has the porting notes for the NPU work.
