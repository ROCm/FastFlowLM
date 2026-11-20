<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Qwen3.8-27B — MTP draft head on NPU2

Orientation document. Read this instead of re-scanning the tree at the start of a
session. Everything here was verified against the working copy on 2026-09-14
(branch `qwen3.8_27B`, HEAD `e7256674`).

**Goal:** run the MTP (multi-token-prediction) draft head on the NPU, decode shape
only — **T = 1**. Step 0 (the T=221 cache-priming pass) is explicitly out of scope;
it is a prefill-fused problem, not this xclbin's.

---

## 0 · State of the tree — read this first

> **This section is superseded (checked 2026-09-18).** The rebase it recommends
> was done. `make bitstream MODEL_TYPE=QWEN3_8_27B`, `make host` and `./run.exe`
> all succeed, and the array computes the whole decoder layer at cos 0.999675
> (§7.4). The breakage inventory below is kept as the record of what was wrong
> and why each file moved, not as a description of today. The second-dispatch
> timeout that blocked the DLL driver is **fixed** — it was `enable_tb` in
> `iron/gate.py` building an MM2S chain that `MTP_layer_aie.py` never routes;
> see §7.5 and
> `FLM_DLL/detail/qwen3_8mtp_npu/speculate/npu/SECOND_DISPATCH_TIMEOUT.md`.

The directory was created by `git cp` of two different projects and **did not
build or import**. `dd3a50f8 "checkpoint: TODO MTP"` dropped in 8195 lines
wholesale; only `host/cpu_ref_mtp_head.cpp` (commit `e7256674`) is real new work.

Provenance of every file, measured by content hash:

| Source | Files |
|---|---|
| **Unique to MTP (new work)** | `host/cpu_ref_mtp_head.cpp` ✅ done and building |
| **Qwen3.5 template — STALE, must be ported** | `iron/{proj_tile,attn,aux_memtile,rms_residual,RoPE,glu,gate,constants,model_config,models/__init__}.py`, `kernels/{aie_array_layout.h,glu.cc,rope.cc,q4_core.h}`, `host/{decoding_layer.cpp,decoding_layer.hpp,models.h}`, `models/all_models.h`, `Makefile` |
| **Qwen3.8 `gated_attention_verify` (GAV) — current, correct** | `host/{cpu_ref_common.hpp,cpu_ref_bridge.hpp,quant_size.hpp}`, `iron/models/qwen38.py`, `models/qwen3.8-27b.h`, `kernels/q8_core.h` |
| **Hand-edited, uncommitted** | `iron/layer_aie.py`, `host/host.cpp` (see §5) |

### The sibling project is the reference implementation

`../gated_attention_verify/` (**GAV**) is the *same decoder layer*, for the same
model, already working. The MTP head's decoder layer is **line-for-line identical
math** to GAV's full-attention layer — same gated attention, same GQA 24/4, same
qk-norm-before-RoPE, same SwiGLU. `cpu_ref_mtp_head.cpp:26` says so explicitly.

**Therefore: the default move for any stale file is to copy GAV's version, not to
port the Qwen3.5 one.** GAV is ahead on every shared file. Diff sizes confirm it —
e.g. `kernels/attn_kv.cc` is 16 diff-lines from GAV but 449 from Qwen3.5.

### Concrete breakage inventory

1. **`iron/` does not import.** `iron/models/__init__.py` imports `.qwen35`,
   which does not exist here. Fix: use GAV's 5-line version
   (`from .qwen38 import qwen38_27b_config`).
   ```
   ModuleNotFoundError: No module named 'iron.models.qwen35'
   ```
2. **`iron/model_config.py` is missing the attention-partitioning block** that
   `layer_aie.py` already calls: `cus_per_column`, `q_heads_per_group`,
   `q_heads_per_group_padded`, `kv_heads_per_cu`, `num_attn_columns`,
   `kv_heads_per_column`. It also lacks `QuantType.Q4_K` and `_q4_k_size()`,
   while `iron/models/qwen38.py` passes `PROJ_QUANT_TYPE = QuantType.Q4_K`.
   It still *requires* dead Qwen3.5 params (`num_delta_net_cts`,
   `AB_PROJ_QUANT_TYPE`, the `linear_*` gated-delta-net geometry) that
   `qwen38.py` does not pass. Fix: take GAV's `model_config.py` wholesale.
3. **`iron/constants.py` is Qwen3.5's** — wrong packet ids
   (`to_rope=1`/`to_glu=15` vs GAV's `to_rms=2`/`to_rope=4`/`to_aux=8`/`to_glu=16`),
   `ModelArch` enumerates only QWEN3_5_*, no `Q4_K`, no `L_TOKENS`.
4. **`iron/attn.py` is Qwen3.5's** and uses the old names
   (`Q_HEAD_PER_ROUND_PADDED`, `kv_head_per_round`); GAV's uses the new ones and
   carries the real-vs-padded head distinction (`NQ_PER_CU` = 6 real,
   `NQ_PER_CU_PADDED` = 8 written) that the 24/4 geometry needs.
5. **`models/all_models.h` is Qwen3.5's**: `#define`s QWEN3_5_* only, defaults
   `MODEL_TYPE` to `QWEN3_5_OMNI`, and `#include`s five `qwen3.5-*.h` files that
   **do not exist in `models/`** (only `qwen3.8-27b.h` does). Kernel compile
   cannot succeed. GAV's version includes just `../models/qwen3.8-27b.h` and
   defaults to `QWEN3_8_27B`.
6. **`host/models.h` is Qwen3.5's**: paths point at `/path/to/models/`,
   no `QWEN3_8_27B` enumerator. GAV's points at
   `/path/to/models/Qwen3.8-27B-NPU2` + `ref.safetensors`.
7. **`Makefile` defaults `MODEL_TYPE ?= QWEN3_5_9B`** and has no `L_TOKENS`
   export. GAV's has `MODEL_TYPE ?= QWEN3_8_27B`, `L_TOKENS ?= 8`, and passes
   `-DL_TOKENS` into `KERNEL_ARGS`.
8. **`host/host.cpp`** is a gutted Qwen3.5 decoding-layer test (embeddings + 3
   decode steps + KV-cache compare). It references `linear_decoding_layer.hpp`,
   `metrices.hpp`, `lm_config.hpp` and tests nothing MTP-specific. Needs a
   rewrite against `cpu_ref_mtp_head.cpp`'s taps.
9. **`layer_aie.py` is mid-edit** (uncommitted). It was switched to
   `QWEN3_8_27B`, given two attention columns at cols 4 and 5, and a
   `cus_per_column` fan-out loop — but that attribute doesn't exist yet (#2),
   and the `gate_tile` is imported but never instantiated or wired, so the
   `attn_out * sigmoid(gate)` step is currently missing from the graph.

**Recommended first move:** rebase the whole `iron/` + `kernels/` + `models/` +
`Makefile` on GAV, then re-apply the MTP-specific deltas. Do *not* try to forward-port
the Qwen3.5 files.

---

## 1 · What the MTP head computes

Authoritative sources:
- an HF transformers dump of the model (not shipped) — what `ref.safetensors` holds
- `host/cpu_ref_mtp_head.cpp` — **the CPU golden reference, already working**

One speculative cycle is 7 draft steps through a single decoder layer that owns
its own KV cache. Step 0 primes that cache with T=221; steps 1..6 are chained
T=1 decodes. **This xclbin implements steps 1..6.**

```
e        = embed_tokens(draft_token)                 # Q8_0 row lookup, [5120]
fc_in    = cat[ rms(e, pre_fc_norm_embedding)        # cols    0:5120  <- EMBEDDING FIRST
              , rms(h, pre_fc_norm_hidden)   ]       # cols 5120:10240
fc_out   = W_fc @ fc_in                              # 10240 -> 5120
--- one Qwen3_5DecoderLayer, layer_idx 0, full attention ---
xn       = rms(fc_out, input_layernorm)
q|gate   = W_q  @ xn        # 12288 rows = 24 heads x 256 x 2
k, v     = W_kv @ xn        # 1024 each = 4 kv heads x 256
q, k     = q_norm(q), k_norm(k)      # per 256-dim head, BEFORE rope
q, k     = rope(q, k)                # dims [0:64] only, rotate_half; [64:256] pass through
cache   += k, v
o        = softmax(q K^T / 16) V     # GQA, kv_head = q_head / 6; T=1 => mask is all-ones
o       *= sigmoid(gate)             # <-- the output gate
h        = fc_out + W_o @ o          # o_proj takes 6144 (gate already consumed)
y        = h + down(silu(gate(rms(h, post_ln))) * up(rms(h, post_ln)))
norm_out = rms(y, mtp.norm)
logits   = lm_head(norm_out) ; draft = argmax
```

`h` (the fused hidden) for step *s* is the head's **own** `norm_out` from step
*s-1* — not the main model's. That self-chaining is what makes all *k-1* cached
slots poisoned at a cycle boundary (doc §10.3).

### Geometry (all verified in `models/qwen3.8-27b.h` and `cpu_ref_common.hpp`)

```
MODEL_DIM  5120     NUM_ATTN_HEADS 24      HEAD_DIM 256    ROPE_DIM 64 (partial 0.25)
DQ         6144     NUM_KV_HEADS    4      GQA_GROUP 6     ATTN_SCALE 1/16 = 0.0625
DKV        1024     INTERMEDIATE_SIZE 17408                rope_theta 1e7
VOCAB      248320   FC_IN 10240 = 2*D      GLU_SLICE 1024  rms_eps 1e-6
```

### Numerics traps (from doc §8 — each costs a day if missed)

- **RMSNorm is zero-centred**: `out = x_norm * (1.0 + w)`. All five norms in the
  head store weights around 0. Forget the `+1` and you get zeros.
- **q_proj is head-interleaved upstream** (`row h*512+0..255` = head *h* query,
  `+256..511` = head *h* gate) — **but the q4nx converter de-interleaves it** to
  `[all 24 q | all 24 gate]`, same as the main model's layers. The CPU ref
  defaults to `q_interleaved = false` and that is the verified-correct setting.
  `--q-interleaved` exists only
  as an escape hatch.
- **q/k norm runs BEFORE RoPE.** Same as the other Qwen ports, unlike HunYuan.
- **Attention scale 1/16 is applied by the attention op**, not folded into q_norm.
- **cos/sin are bf16.** transformers computes the table in fp32 then casts to
  bf16; the reference consumed the bf16 one. `mrope_table()` rounds by default.
- For T=1 drafts all three M-RoPE channels carry the same value, so interleaved
  M-RoPE **degenerates to plain 1-D RoPE**. Only step 0 touches the image plateau.
  This is a real simplification for the decode kernel.
- **Concat order is load-bearing**: `cat([norm_embed, norm_hidden])`. Swapping
  the halves yields plausible-looking garbage, not obvious garbage.

---

## 2 · Weights

15 tensors under `model.mtp.*` in `/path/to/models/Qwen3.8-27B-NPU2/model.q4nx`
(verified present). Projections are Q4_K in 32x256 blocks, 4736 B/block, stored
`[nrb][ncb][4736] I8`. The five norms are BF16 `[5120]` (q/k_norm are `[256]`).

| tensor | q4nx shape | logical |
|---|---|---|
| `fc.weight` | `[160, 40, 4736]` | `[5120][10240]` |
| `layers.0.self_attn.q_proj.weight` | `[384, 20, 4736]` | `[12288][5120]` q then gate |
| `layers.0.self_attn.{k,v}_proj.weight` | `[32, 20, 4736]` | `[1024][5120]` |
| `layers.0.self_attn.o_proj.weight` | `[160, 24, 4736]` | `[5120][6144]` |
| `layers.0.mlp.{gate,up}_proj.weight` | `[544, 20, 4736]` | `[17408][5120]` |
| `layers.0.mlp.down_proj.weight` | `[160, 68, 4736]` | `[5120][17408]` |
| `{pre_fc_norm_embedding, pre_fc_norm_hidden, norm}.weight` | `[5120]` BF16 | |
| `layers.0.{input_layernorm, post_attention_layernorm}.weight` | `[5120]` BF16 | |
| `layers.0.self_attn.{q,k}_norm.weight` | `[256]` BF16 | |

Plus shared `model.embed_tokens.weight` and `lm_head.weight`, Q8_0, 2.37 GiB each
(`tie_word_embeddings: false` — they are two separate tensors).

Head total 424.7 M params = 810 MiB. `lm_head` is 75 % of a draft step's traffic.

**Known error floor: cos ≈ 0.994 is Q4_K's floor for this head, not a bug**.
Don't chase it.

---

## 3 · The CPU golden reference — use it

`host/cpu_ref_mtp_head.cpp` is **done, building, and the thing to score against.**
Without `-DCPU_REF_STANDALONE` it defines no `main` and is harmless in the
`run.exe` source glob.

```bash
g++ -O3 -march=native -fopenmp -std=c++17 -DCPU_REF_STANDALONE -Ihost \
    host/cpu_ref_mtp_head.cpp -o cpu_ref_mtp_head
./cpu_ref_mtp_head            # defaults to the two paths below
```
Defaults: `/path/to/models/Qwen3.8-27B-NPU2/{model.q4nx,ref.safetensors}`.

Flags, and when to reach for each:

| flag | effect | use when |
|---|---|---|
| *(none)* | chained — `prev_hidden`/ids from our own previous step | **the number that matters**; errors compound as they will on the NPU |
| `--teacher-forced` | `prev_hidden`/ids from the dump each step | a chained run diverges and you need to find *which step* broke |
| `--fc-from-ref` | feed the dump's `fc_out` into the decoder layer | isolate the decoder layer from the fusion in front of it |
| `--rope-from-ref` | use the dumped cos/sin tables | separate a table mismatch from a math one |
| `--no-logits` | skip `lm_head` | fast iteration on head math (2.37 GiB saved per step) |
| `--per-token` | per-row error | a mask/position bug lands on specific rows long before it moves the aggregate |
| `--steps N` | truncate | |

It scores every tap the dump carries: `inputs_embeds`, `norm_embed`,
`norm_hidden`, `fc_out`, `layer_out`, `norm_out`, `logits_last`, `draft_token`.

**Score `layer_out - residual`, not `layer_out`.** `layer_out` carries `fc_out`
straight through the residual path, so its raw cosine is flattered by an identity
term. The driver already subtracts the layer's input — that residual-free number
is what actually moves when a head order or a mask is wrong.

Current status: **6/7 draft tokens match**.

The reference dump comes from an HF transformers run of the model (not shipped with the dev kit):
generate `ref.safetensors` with such a reference and pass its path.

---

## 4 · NPU2 (Strix / XDNA2) hardware facts

Constraints that shape every design decision here. Violating one usually
produces a **hang**, not an error.

### Hard limits
- **Lock maximum value is 63.** An acquire/release count above this silently
  wraps. Watch this when a shared buffer's credits are counted per core.
- **CT has 16 BDs. MT has 48 BDs.** A tile's whole DMA program must fit.
- **CT DMA has 3 dimensions** for data reorder. **MT and IT DMAs have 4.**
  A reorder needing 4 dims cannot live on a compute tile — push it to the memtile.
- A CT has **2 S2MM and 2 MM2S** channels; MT/IT have **6 and 6**.
- Shim BD: the **iteration dim is not part of the length**. Splitting an axis
  across `iter x dim2` silently truncates the transfer — keep the pattern in
  dims 1..3.
- **`npu_dma_wait` is per-shim.** A DDR round trip must enter and leave on the
  same IT. A BD repeat expressed as a zero-stride axis hangs the design; put
  repeats in `size[0]`.

### Dataflow rules learned the hard way
- **One S2MM channel, one producer.** Two flows into a DMA channel is a
  masterset error, and a merge has no ordering guarantee. Rejoin split operands
  in neighbour memory instead.
- **Split the consumer, not the producer.** A CT has only 2 S2MM channels, so
  fan an elementwise op across N tiles rather than merging N operand pieces.
  And *test the mapping* — a symmetric swap can't hang, so it will pass a smoke
  test while being wrong.
- **A shared buffer's credits are counted per core.** The outbound BD must
  acquire N. The signature of getting this wrong is a hang whose **first few
  tokens are correct**.
- **MT[3]'s o/down DDR tap can't be removed** — the array config differs by
  exactly one flow yet it hangs, and the stall lands upstream at k/v, not at o.


### Arithmetic
- **`aie::mmul<8,8,8,bf16>` truncates to bfp16 on npu2.** It floors both
  operands onto a shared-exponent-per-8 grid. Pre-round weights on the host to
  recover ~20x accuracy.
  The build already passes `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`.
- **`aie::accum::to_vector<bf16>()` floors** (truncates toward -inf), as does
  every fp32->bf16 in an AIE kernel. Model it, or your CPU reference is 1 ulp
  off half the time.
- **An mmul C-operand is dim-block major.** Attention output needs a transpose
  out with the *real* head count (6), not the padded one (8). Period-8
  permutations tell you nothing, since every tile is 8x8x8.


### Host-side traps
- **`_move_weights`' offset is in elements, not bytes.** A byte offset reads
  past the buffer end and the cores dequantise garbage into NaN. **All-NaN
  output means addressing, not arithmetic.**
- **RTP addresses are per-build.** They come from that build's
  `input_with_addresses.mlir` (the bitstream rule already dumps them to
  `build/$(MODEL_TYPE)/xclbins/rtp_addresses.txt`). A stale address hangs in
  `qds_device::wait()`.
  Note `host/models.h::get_rms_addr()` for QWEN3_8_27B is a `TODO: take from
  rtp_addresses.txt` returning 0 in GAV.
- **`#if` on a `constexpr` silently deletes the block.** `model_spec.h`
  constants are `constexpr`, not macros, so `#if X > 0` evaluates as `0 > 0`.
  Use `if constexpr`.

---

## 5 · Array layout (as `iron/layer_aie.py` currently stands)

npu2, 8 columns x 4 compute rows. `generate_tile_map(6, 8)` returns
`IT` (row 0, shim), `MT` (row 1, mem), `CT[row][col]` (rows 2..5, indexed 0..3).

```
col:      0        1        2         3       4        5        6        7
        ┌────────┬────────┬─────────┬───────┬────────┬────────┬────────┬────────┐
IT r0   │ proj   │ proj   │ x/rms   │ kv out│        │ TB out │ proj   │ proj   │
MT r1   │ assemb │ assemb │ aux_mt  │       │ attn   │ attn   │ assemb │ assemb │
CT r2   │ proj   │ proj   │ rms_res │       │ attn qk│ attn qk│ proj   │ proj   │
CT r3   │ proj   │ proj   │ RoPE    │       │ attn kv│ attn kv│ proj   │ proj   │
CT r4   │ proj   │ proj   │ gate    │       │ attn qk│ attn qk│ proj   │ proj   │
CT r5   │ proj   │ proj   │ glu     │       │ attn kv│ attn kv│ proj   │ proj   │
        └────────┴────────┴─────────┴───────┴────────┴────────┴────────┴────────┘
```
- **Projection engine**: 16 CTs across cols {0,1,6,7}, `q4_proj_engine`, with
  `MT[col]` assembling and `IT[col]` feeding weights. This is the weight-streaming
  workhorse — the MTP head is ~1 FLOP/byte, so *this* is the critical path.
- **Attention**: 2 columns (4, 5), each `attn_column` with 2 CUs
  (qk tile + kv tile sharing an `s_buffer`). 4 kv heads / 1 per CU / 2 columns
  = `cus_per_column` 2. Q is padded 6 -> 8 heads per CU for the 8x8x8 mmul
  (`q_heads_padding = 8`, `ATTN_IMPL_1x8x1`).
- **Serial ops** on col 2, top to bottom: `rms_residual` CT[0][2], `RoPE`
  CT[1][2], `gate` CT[2][2], `glu` CT[3][2], with `aux_memtile` at MT[2].
  That fills the column — there is no free compute tile left in it for
  `down_proj`'s consumer, which is why the plan is to reuse the rms tile's
  third phase rather than add a tile.
- Routing is **packet-switched** via `connect_port(..., pkt_id=, keep_pkt_header=)`.
  One physical MM2S can carry several logical streams demuxed by id.
  **Connection order matters** — long connections get routing priority.

### Resolved: the gate is wired
An earlier draft of this document recorded `gate_tile` as imported but never
instantiated. It is now a single `gate_core` at CT[2][2], fed by `attn_hub.O_out`
(the two columns' halves rejoined in MT[3]) and `aux_mt.Gate_out` (z, parked in
MT[2] for the whole attention latency). At T = 1 the whole row fits one tile, so
this tree needs neither GAV's helper/emitter split nor its `qkv_stage.py` and
`x_replay_mt` — do not port them.

---

## 6 · Build

```bash
source FLM_Xclbin/setup.sh # set MLIR_AIE_DIR, PEANO_INSTALL_DIR first
cd FLM_Xclbin/Qwen3_8/MTP_decoding
make bitstream MODEL_TYPE=QWEN3_8_27B   # NOT `make all`
make host
make test MODEL_TYPE=QWEN3_8_27B
```
- `make bitstream` -> `build/$(MODEL_TYPE)/xclbins/layer.xclbin` **and**
  `rtp_addresses.txt` (read it after every rebuild, see §4).
- `make route` renders the routing graph via `visualize.py`.
- Kernels: `xchesscc_wrapper aie2p ... -DNPU2 -DMODEL_TYPE=... -O3`.
- Host: `g++-13 -std=c++20`, links `xrt_coreutil`, `aiebu`, boost; pulls
  `../../host_common`. **Every `host/*.cpp` is globbed into `run.exe`**, which
  is why `cpu_ref_mtp_head.cpp` guards its `main` behind `CPU_REF_STANDALONE`.

Reminder: `Makefile` currently defaults `MODEL_TYPE ?= QWEN3_5_9B` and
`models/all_models.h` includes five nonexistent `qwen3.5-*.h` headers, so a bare
`make` fails at kernel compile until §0.5 and §0.7 are fixed.

---

## 7 · Suggested order of work

1. **Make it import and build.** Rebase `iron/`, `kernels/`, `models/`,
   `Makefile`, `host/models.h` on GAV (§0). Confirm
   `python3 iron/layer_aie.py QWEN3_8_27B` emits MLIR and `make bitstream` runs.
   Nothing below is testable until this is done.
2. **Wire the gate** (§5) so the graph computes the full layer.
3. **Rewrite `host/host.cpp`** against `cpu_ref_mtp_head.cpp`'s taps rather than
   the Qwen3.5 KV-cache test. Score `fc_out`, `layer_out - residual`, `norm_out`.
4. **Stage the bring-up**, one tap at a time:
   `fc_out` -> `q/k/v + norms + RoPE` -> `attn` -> `gated attn` -> `o + residual`
   -> `MLP` -> `norm_out`.

   **The decoder layer is complete as of 2026-09-15.** The array computes the
   whole thing, `h2 = h + W_down @ (silu(gate(yn)) * up(yn))` with
   `yn = rms(h, post_ln)` and `h = fc_out + W_o @ (o * sigmoid(z))`, and taps
   the D-wide `h2` on IT[2]. **cos 0.999675 end to end**, bit-identical across
   three runs. Two numbers accompany it and both matter:

   - `h2 - h`, the residual removed: **0.999327**. This is the one to read for
     the MLP — `h2`'s raw cosine is flattered by the identity term, which is
     the larger of the two.
   - the control, `h2` scored against `down` alone: **0.743**. If the rms tile
     had dropped the add and forwarded down unchanged, that number and the
     end-to-end one would swap. A residual is exactly the kind of op that
     passes a smoke test while doing nothing, so it needs a control saying
     the add happened.

   Earlier phases, for comparison: `up|gate + SwiGLU` read 0.999482 with its
   up/gate-swap control at 0.293; the post-attention-norm tap read 0.999802
   with controls 0.456 (no residual) and 0.992 (wrong norm).

   **No intermediate stage is separately observable.** The post-attention norm
   became up|gate's x, the SwiGLU row became down's x, and down became the rms
   tile's third-phase x — each time the producer's single MM2S had no room for
   a second destination, so the tap was *moved*, not joined. Both norms leave
   west under `rms_pkt_id.to_proj` and the engine's x FIFO tells them apart by
   arrival order; `h2` covers all of them.

   The DDR tap on `IT[2].S2MM[1]` has now outlived three producers — the
   post-attention norm, then down_proj, now the rms tile — with the same
   descriptor each time. Worth knowing when reading `git log`: a diff that
   looks like the tap moved is usually the stage in front of it changing.

   Two things about this phase worth keeping:
   - **up|gate is a `mvm_resident`, not a streaming `mvm`.** Streaming it would
     need the norm replayed `UP_GATE_REPEATS = 68` times and a lock saturates at
     63. Latching the row on the core sidesteps the ceiling rather than relaying
     credits through a fanout chain.
   - **The engine feeds the GLU tile directly, not through `aux_mt`.** Its GLU
     buffers are `2 * INTERMEDIATE_SIZE / 4 = 8704`, not a multiple of the y
     tree's 512-element packets, so a packet would straddle the ping/pong
     boundary. `glu_slice = 1024` is exactly two rounds, so the engine's output
     lands on the tile's own buffer.

   One trap the final phase turned up, worth keeping: **`down` could not keep
   its own packet id.** As a DDR tap it travelled under `proj_pkt_id.to_out = 1`
   heading south; as the residual's operand its route ends at CT[0][2], where
   `it1_pkt_id.to_rope` — also 1 — is already passing through northbound. One
   id, one slave port, two destinations is what a switchbox cannot express. It
   now shares `to_rms` with `fc_out`, which costs no new rule anywhere because
   `fc_out` already walks that exact route, and drops one match rule from every
   hop of the y tree. The rms tile's ping/pong chain orders the three arrivals
   (fc_out, o, down) with nothing in the kernel telling them apart.

   Next: `rms(h2, mtp.norm)` and `lm_head`, then chaining `norm_out` back as the
   next step's `prev_hidden` (§7.6). Neither norm nor lm_head is in this xclbin.
5. **The fusion in front is the genuinely new part.** The decoder layer is GAV's;
   what MTP adds is: Q8_0 embedding row lookup, two RMSNorms, the 10240-wide
   concat, and the 10240->5120 `fc`. `fc` is just another projection for
   `q4_proj_engine` — but its input is 2*D, which no existing path produces.
6. **Chaining.** `norm_out` must feed back as the next step's `prev_hidden`.
   Keep it resident; a DDR round trip per step costs more than the math.

### The KV cache: two planes, and no context ceiling (2026-09-18)

The cache is `[MAX_L][DK]` then `[MAX_L][DV]` — two contiguous planes, matching
`../gated_attention_verify`. `decoding_layer.hpp`'s `k_at` / `v_at` /
`k_row_stride` / `v_row_stride` are the only place that layout is written down;
every descriptor, the seeder and the read-back go through them.

**k and v are appended on separate shim channels.** The RoPE tile stamps them
with different packet ids — `kv_out_pkt_id.k_to_col_*` = 12/13,
`v_to_col_*` = 14/15 — and `MTP_layer_aie.py` routes k to `IT[4/5].S2MM[0]` and
v to `S2MM[1]`. Each plane then gets its own BD in `_receive_kv_cache`, whose
only stride is one row.

That split is what removed the cache's depth limit. Before it, one BD caught the
interleaved `k v` stream and had to cross between the planes with a stride of
`MAX_L * DK`. A shim BD's stride field is 20 bits and bf16 strides are halved on
lowering, so the cache capped at `MAX_L = 2048` — and above it `npu_instr_utils`
*printed and masked* rather than faulting. At 16384 the stride masks to exactly
**0**, dropping every v row on top of k, and only the next pass would have seen
it.

`FLM_MAX_L` overrides the cache depth. The bench still drives one row at
`current_context_length`, so the scores do not depend on it — what it exercises
is the addressing. Since v is read back from `v_at(slot)` on a zeroed BO,
scoring it at 16384 is direct evidence the plane landed 33.5 MB in rather than
wrapping onto k.

| depth | k | v | h2 | stderr |
|---|---|---|---|---|
| 1136 (default) | 0.999958 | 0.999979 | 0.999675 | clean |
| `FLM_MAX_L=4096` | 0.999958 | 0.999979 | 0.999675 | clean |
| `FLM_MAX_L=16384` | 0.999958 | 0.999979 | 0.999675 | clean |

Identical to every digit, and no `stride_2 out of range`. The bitstream routed
first try and the RTP addresses were unchanged (qk 9216, kv 36864).

### 7.5 · Re-dispatch: it does now, and the two revisions that said otherwise

This section has been wrong in both directions. The history is the useful part.

**Revision 1 claimed it re-dispatched.** `FLM_MTP_REPEAT=32 FLM_MTP_BASE=1088
FLM_MTP_ROT=1` was reported as 33 completing dispatches at rotating cursors with
the metrics unchanged. It was one completing dispatch and 32 timeouts.
`decoding_layer::operator()` discarded the `ert_cmd_state`, and a timed-out
dispatch never writes the output BO — so `sync_from_device()` returned the
*previous* pass's row and the bench scored it as a pass. "Metrics unchanged" was
the symptom, not the proof. The wall clock said so all along: baseline 1.61 s,
`FLM_MTP_REPEAT=1` 62.98 s, `FLM_MTP_REPEAT=4` 126.97 s — one XRT timeout per
extra dispatch. `operator()` now checks the state and exits on anything but
`ERT_CMD_STATE_COMPLETED`.

**Revision 2 claimed it could not re-dispatch, as a property of the design.**
True as a measurement, wrong as a conclusion — it was one line of IRON.

`iron/gate.py` had `enable_tb = True`, which builds a second MM2S chain on the
gate tile for a DDR tap. With it on, `block[5]` releases `tb_lock` rather than
`gated_attn_out_prod_lock`, and the only thing that recharges the producer lock
is `block[7]`, the tap itself. `MTP_layer_aie.py` never calls `connect_port` on
`gate_core.TB_Out` — but **an unrouted MM2S chain still fires**: `block[7]`
took all 10 `tb_lock` credits and stalled forever on a channel that reaches
nothing. Pass 1 ran entirely on the producer lock's initial credit of
`repeats` = 10 and looked perfect; pass 2 blocked on the first acquire.

The stall surfaced nowhere near the cause. `FLM_MTP_STALL=1` showed the hung
pass writing k 1024/1024 and v 1024/1024 but `layer_out` 0/5120, and
`FLM_MTP_TRUNC2` bisection put the first hanging wait at phase 5, o_proj's
weight round — because o's `x` is what never arrived.

With `enable_tb = False` and blocks 6/7 suppressed,
`FLM_MTP_REPEAT=5 FLM_MTP_ROT=1 FLM_MTP_BASE=1121` completes all six dispatches
in **1.70 s** total, k 0.999958 / v 0.999979 / h2 0.999675 — unchanged from the
single-dispatch numbers above. The DLL port's full verifier now passes 28/28.

To restore the tap, wire `TB_Out` to a shim in `MTP_layer_aie.py` **and** post
a matching receive in `_gen_sequence` and in the DLL's `gen_mtp_layer_seq()`.
Routing it without receiving it hangs the same way.

`gated_attention_verify/iron/gate.py` has no tb chain at all, which is the whole
reason that design always survived repeated dispatch. The contrast was real; the
inference drawn from it — that the fusion path or the k|v append was at fault —
was not.

Both S2MM channels are still waited on each pass (`_gen_sequence`), which is
necessary regardless: waiting only on channel 0 would leave v in flight.

The limit that remains is the replay's: `dim1` counts whole `L_CHUNK`s on a
10-bit field, so `MAX_L <= 1024 * 16 = 16384` (`decoding_layer.hpp::MAX_L_LIMIT`),
and `MAX_L` must be a whole number of `L_CHUNK`s because the replay rounds the
key count up to a chunk boundary and reads the rows it lands on. Both are
asserted in the constructor rather than left to a descriptor that would mask
them — `FLM_MAX_L=16400` and `FLM_MAX_L=1000` each abort with the matching
message.

### Scope reminders
- T=1 only. No mask needed — a T=1 step attends over the whole cache (all-ones).
- The head's KV cache is its own: 4 kv heads x 256, **4 KiB/token** — ~4.4 MiB at
  the 1136-row default and 64 MiB at the 16384 cap, which is a descriptor limit
  rather than a memory recommendation. Do **not** alias it with the main model's
  cache — MTP cache slot *j* ≡ main cache slot *j+1* (doc §6.1).
- `d_7` is never written to the cache. Step 6 emits it; no step 7 consumes it.
- At a cycle boundary the head's cache is cropped by **exactly k-1 = 6, always**,
  regardless of how many drafts were accepted, because every chained slot was
  fused against the head's own output (doc §10.3). This is the single most
  likely source of a slowly-degrading acceptance rate — it produces *plausible*
  drafts, not obviously wrong ones.
