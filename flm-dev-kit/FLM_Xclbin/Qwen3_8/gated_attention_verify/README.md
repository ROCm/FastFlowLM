<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# `gated_attention_verify` — Qwen3.8-27B full-attention layer

Qwen3.8-27B alternates two layer types: layer *i* is full attention iff
`(i + 1) % 4 == 0` (so 3, 7, … 63), and the other three in four are gated delta
net. The two used to share one xclbin with a runtime phase select; they are now
two projects.

| | |
|---|---|
| this one | the every-fourth **full-attention** layer |
| [`../gate_delta_net_verify`](../gate_delta_net_verify) | the other three, **verified through `post_attention_layernorm` at L = 8** |

Same target as the delta-net project: the MTP verify step, `L_TOKENS = 8` query
rows. The MLP is out of the design in both.

## Status

**`input_layernorm`, the fused q | k | v projection, q/k norm + partial RoPE,
the causal GQA softmax attention and the output gate are verified on hardware at
L = 8, against a seeded KV context.** That is the whole of what this xclbin
contains.

| | |
|---|---|
| `make kernel` | ✅ 8/8 |
| `make bitstream` | ✅ `layer.xclbin`, 20 core ELFs |
| `make host` | ✅ |
| `./run.exe` | ✅ gated 0.9992 · attention 0.9997 · k 0.999964 · v 0.999988, 8 ms |
| `FLM_MAX_L=16384 ./run.exe` | ✅ same four scores to every digit — the cache's depth limit is 16384 now, not 2048 |

Scored against the CPU golden in `host/host.cpp`:

* **gated** — `L × DQ`, `attn * sigmoid(z)`. The end of everything the array
  computes, and the stage under test.
* **attention** — `L × DQ`, ungated. The direct control for the gate: if this
  still scores 0.9997 and the gated row does not, the fault is in the gate or in
  z, not upstream of them.
* **k** — `L × DK`, k_norm then rotated, read back out of the KV cache.
* **v** — `L × DV`, untouched. Also scored against the *raw* projection, which
  is what separates a rotation bug from a projection or transpose bug.

`FLM_MAX_L` overrides the cache depth. The bench only ever drives 8 rows at a
small offset, so the score does not depend on it — what it exercises is the
cache's *addressing*. k and v land on two separate S2MM channels of the
column's shim (`iron/RoPE.py` stamps them with different packet ids), so each
plane is appended by its own BD whose only stride is one row. Before that split
one BD caught the interleaved `k(t) v(t)` stream and crossed between the planes
with a stride of `MAX_L * DK` — a 20-bit field, halved for bf16, so the cache
capped at 2048 and, above it, `npu_instr_utils` masked the stride instead of
faulting (at 16384 it masks to **0**, dropping every v row on top of k). Since
`v` is read back from `v_at(22)` on a zeroed BO, scoring it at 16384 is direct
evidence the plane landed 33.6 MB in rather than wrapping. The limit that
remains is the replay's 10-bit chunk count, `1024 * L_CHUNK = 16384`, which is
asserted rather than masked.

Everything but the first is an upstream diagnostic: they localise a failure the
gated output alone would only report. q is not read back — it goes straight from
the RoPE tile to the four attention CUs and its correctness is implied by the
output. The last build to score it directly got 0.999966 with the GQA padding
verified zero, which is the layout the CUs index against.

The run also **recovers the gate the NPU actually applied**, by dividing its
gated row by its own attention row. That is what separates a wrong `z` from a
misapplied one: a valid sigmoid in the wrong place still lands in (0, 1), so the
range alone is not evidence. Clean build: range [0, 0.763], mean
|g − sigmoid(z)| = 0.0014.

### The context sweep

The golden runs `LC` positions and hands the NPU the last `L_TOKENS` of them,
with the first `LC - L` rows of k and v seeded into its cache. `ATTN_LC`
overrides `LC`, so one binary covers every shape:

| `ATTN_LC` | context | chunks | gated | attention |
|---|---|---|---|---|
| 8 | 0 | 1 | 0.9994 | 0.99968 |
| 14 | 6 (ragged) | 1 | 0.999093 | 0.999651 |
| 30 | 22 (ragged) | 2 | 0.999246 | 0.999698 |
| 40 | 32 (aligned) | 3 | 0.999201 | 0.99973 |
| 56 | 48 (aligned) | 4 | 0.999501 | 0.999712 |

Two negative controls, because the two stages fail in different ways:

* **`ATTN_POISON=<row>`** corrupts one seeded row on the way in. Poisoning a row
  in any chunk moves the score — at `ATTN_LC=40`, 0.9992 → 0.989 for rows in
  chunk 0 and chunk 1 alike — which proves the later chunks are actually read. A
  replay that ignored them would still score 0.9997 and look fine. (How far it
  drops depends on how much context dilutes the one bad row, so compare against
  the clean run at the *same* `ATTN_LC`, not against a fixed number.)
* **`GATE_SWAP_HALVES=1`**, a build-time flag, feeds each gate tile the *other*
  half of z. Nothing hangs, because the two halves have identical shapes — it
  just gates each half by the wrong gate, which is the specific way this
  two-tile design can be wrong. It drops gated to **0.217** (halves 0.33 and
  0.19) while attention holds at 0.999698. The recovered gate stays inside
  (0, 1) throughout, which is exactly why the range check alone would not have
  caught it.

## What runs

```
IT[2] MM2S0 ─ x, 8 x 5120 ──────┐
IT[2] MM2S1 ─ input_layernorm ──┴─> rms CT[0][3] ─> MT[2]  (x replay, 96 KiB)
                                      │                │ 16 replays, col-block major
                            h (80 KiB)│                v
                    CT[0][2], CT[1][3]│   MT[1] 4 KiB FIFO ─circuit─> 16 proj cores
                                                                        │ q4_k mvm
                          MT[3] <── y tree ─────────────────────────────┘
                            │ de-permute + rotate to | k | v | q |
                            v
IT[3] MM2S0 ─ cos|sin, q/k norm ──> RoPE CT[2][3] ─┬─> IT[4],IT[5] S2MM0  new k | v
                                                   └─> 4 packets, q, straight to the CUs

IT[4],IT[5] MM2S0/1 ─ KV cache, L_CHUNK at a time ─> MT[4],MT[5] ─> 4 attention CUs
                                                     MT[4],MT[5] <─ o
IT[4],IT[5] S2MM1 <───────── o, the upstream diagnostic ─────────── o -> DDR
                    MT[4],MT[5] ─ o, half a row ─┐
                                                 v
      MT[3] ─ z, de-permuted, halved ─> gate helper  CT[2][2]
                                        gate emitter CT[3][2]
                                                 │ helper writes its half north
IT[2] S2MM0 <──── the assembled DQ row ──────────┘  gated -> DDR
```

The new k | v go out on the **attention columns' own shims**, not the RoPE
tile's. `npu_dma_wait` only orders DMA within one shim, so appending on IT[3]
and replaying from IT[4]/IT[5] orders nothing — the replay fires against rows
the append has not written yet. Splitting the append into two packets, one per
attention column, is what makes the wait mean something.

`kernels/proj_main.cc` runs one mvm, `(DQ + DK + DV) / 16` rows over
`MODEL_DIM`, out on `pkt_id_to_rope`.

## Array layout

```
col       0       1        2          3          4      5     6       7
CT[0]   proj    proj     h 0..3      rms         qk     qk   proj    proj
CT[1]   proj    proj       -         h 4..7      kv     kv   proj    proj
CT[2]   proj    proj   gate helper   RoPE        qk     qk   proj    proj
CT[3]   proj    proj   gate emitter    -         kv     kv   proj    proj
MT     y asm   y asm +  x replay    qkv stage   kv $   kv $  y asm   y asm
               x FIFO               + z park
IT     w in    w in    gated out    rope w in   kv in  kv in w in    w in
                                                o out  o out
```

The two gate tiles are **vertically adjacent in column 2**, which is
load-bearing: the emitter at `CT[3][2]` owns the assembled `DQ` row and the
helper at `CT[2][2]` writes its half into it through the north window. Same
reason the attention CUs are stacked. `CT[3][3]` is free.

**Columns 4 and 5 are attention**, two CUs each. A CU is a qk core with its kv
core directly north, because the two share the score buffers through neighbour
memory — `attn_kv.cc` acquires them with `_down_lock_acquire`, so the pairing is
fixed by the lock protocol, not just by convention.

Four CUs, one kv head each: 24 q heads over 4 kv heads is 6 per group, padded to
8 for the `1x8x1` mmul. q never goes to DDR — it leaves the RoPE tile as four
packets, one per CU.

**`CT[0][2]` and `CT[1][3]` are memory, not compute.** An 8-row bf16 activation
is 80 KiB against a 64 KiB tile, so the rms tile's residual branch cannot live
on the tile that computes it. A core can address its **west, north and south**
neighbours' memory modules directly (`_reachable_from` in
`iron/rms_residual.py`), so `h` is split four D-wide rows to a tile, one per
16 KiB bank, and the core loads and stores it with ordinary pointers — no DMA.
Those two tiles get no `aie.core` and no `aie.mem`, so they reserve no stack and
all four banks are free. This is a direct port of the delta-net project's
verified arrangement.

Keeping `h` resident from phase 1 is what will let the attention residual take
`L_TOKENS` rows instead of `2 * L_TOKENS`: the layer input never has to come
back from DDR.

## The gate is split, not the attention merged

`attn * sigmoid(z)` needs a full `DQ`-wide attention row and a full `DQ`-wide
`z` row — but attention leaves the array as **two half-rows**, one per attention
column, and a compute tile has **two S2MM channels**. Merging the halves first
would need three inbound streams on a two-channel tile, so it would cost an
extra memtile hop to rebuild a row that the arithmetic then splits again.

So the gate is split instead. Column 4 computes q heads 0..11, which is `z` dims
`[0, DQ/2)`, and column 5 computes heads 12..23; the operation is elementwise, so
a half-row of attention only ever meets the matching half of `z`. Each gate tile
takes one attention half and one `z` half — exactly two inbound streams — and the
two halves run concurrently.

The split is free on the `z` side too: `DQ/2 = 3072` is a whole number of the
engine's 32-row blocks, so "which half" is only where the staging memtile's
outbound BD starts.

**The row is rejoined on the way out, in neighbour memory.** o_proj reduces over
the whole `DQ` row, so the halves have to come back together. Two flows into one
port on the replay tile cannot express that — two producers on one S2MM channel
must be a single merged packet flow, and which half lands first would be a timing
race. So the tiles are stacked instead: the output buffer is one whole `DQ` row
on the emitter, and the helper writes its half into it through the north window,
no DMA. One MM2S sends the assembled row, so the consumer sees one producer.

The shared row's credits are counted **per core, not per row** — each core takes
one and releases one per token, and the outbound BD acquires both. Acquiring one
sends a half-written row *and* drifts: the first few tokens come out correct and
then the run hangs, which reads like a routing fault and is not.

**What this design can get wrong is the mapping**, and it cannot hang, because
the two halves have identical shapes — crossing them just gates each half by the
other's gate. `GATE_SWAP_HALVES=1` is the control that proves the test would
catch it. Delete `build/QWEN3_8_27B/mlir/layer.mlir` before rebuilding with it:
`make` does not treat the env var as a dependency, so a stale MLIR makes the
control silently pass.

`z` itself is the projection engine's second mvm over the *same* resident x, so
it needs no new replay phase — only a longer one, `QKV_REPEATS` 16 → 28. That is
what makes the gate the cheapest stage to add: every stage after it reduces over
something other than the input norm and needs a phase of its own. It is produced
immediately after qkv and not consumed until the whole softmax column has run,
so its 96 KiB parks in `MT[3]` alongside the qkv staging buffer (224 KiB of 512).

## The transpose moved onto the array

The engine emits y row-block permuted, `raw[(rb * L + t) * 32 + r]`, and the
RoPE tile consumes whole tokens — so the de-permute has to happen *before* RoPE,
which rules out the shim BD that used to do it on the way to DDR. `MT[3]`
(`iron/qkv_stage.py`) holds the whole `L × (DQ + DK + DV)` activation and
un-permutes on the way in, in three BD dimensions.

It also **rotates each token to `| k | v | q |`**, which is the order the RoPE
tile consumes: k and v are on their way to the cache while q is still being
normed. The rotation is free — it is expressed as *where the two inbound BDs
start*, not as extra work — and it makes the outbound side one linear BD per
token.

Consequently both shim descriptors are now linear. `_receive_q()` and
`_receive_kv_cache()` in `host/qkv_decoding_layer.cpp` carry no permutation at
all; the KV one is 4-dimensional only because it scatters the interleaved
`k v k v …` stream into two cache planes.

## Bugs found

* **`#if` on a `constexpr`.** `kernels/rope.cc` guarded its GQA padding
  zero-fill with `#if ATTN_GROUPS_PADDING > 0`, but that constant is a
  `constexpr int` in `model_spec.h`, not a macro — the preprocessor substituted
  `0`, and the block was dropped from every build that has ever been made. A
  second bug inside it (advancing `DH_2 = ROPE_DIM/2 = 32` per padding head
  rather than `DH = 256`) was invisible underneath. Symptom: q cosine 0.386 with
  k and v at 0.99997, and `|q padding|` up to 9.94. Now `if constexpr`.
* Earlier, in the qkv-only build: the un-permute shim BD split the row-block
  axis across the iteration dimension, which is not part of `buffer_length`, and
  silently transferred an eighth of the data. That descriptor is gone entirely
  now that the memtile does the transpose.

### From the attention port

* **The `o` output order, twice.** `attn_fv` accumulates y as mmul C tiles, so
  lane `8r + c` of a block is head `r`, dim `8d + c`. `scale_div_aie` drops the
  GQA padding but stores the blocks in place, so `o` is dense and *still*
  `[dim_block][head][8]`. It needs the transpose `[(NQ, 8), (DH/8, NQ*8),
  (8, 1)]` on the way out, with `NQ` the **real** head count (6), because 6 is
  the block `scale_div_aie` actually emits. Using the padded 8 indexes past the
  end of the real data and left 1570 of 6144 elements zero — which is exactly
  2/8 of the row and reads like a padding leak, so the obvious fix is to drop
  the transpose. That is wrong in the other direction: every value is then
  correct but dim-block ordered, cosine 0.024, a permutation with period 8. The
  period is just the mmul tile size and points at no particular stage.
* **The replay's chunk axis.** The descriptor has to walk all `chunks` on dim1
  with stride `L_CHUNK * DK`; the repeat in `size[0]` covers the *tokens*, not
  the chunks. Folding the chunk count into the repeat re-sends chunk 0 every
  firing and never reaches row 16, because a repeat replays a BD from the same
  address. Invisible at `context_offset = 0`, where 8 rows round up to one chunk
  and the two forms coincide — hence the sweep and the poison control.
* **`npu_dma_wait` is per-shim**, which is why the append is split across the two
  attention shims. See the dataflow section.
* **Packet id collision.** Switchboxes route on id alone, so q's four CU packets
  and the two k | v packets cannot both start at 0; `kv_pkt_base = 4`.

### From the gate port

* **`_move_weights` takes a bf16 element offset, not a byte offset.** Its
  descriptor is built with an element size of `sizeof(bf16)` and its own
  `a_block_size` is divided down to match, so `w_z_offset` — computed in bytes,
  like every other offset into `all_proj_weight_buffer` — has to be halved on the
  way in. Passing the byte offset addressed twice as far into the buffer and
  streamed whatever was past its end. The cores dequantised that into **NaN**
  rather than rejecting it, so the failure surfaced as `-nan` in every gated
  lane while attention still scored 0.9997. A gated output that is NaN rather
  than merely wrong is the signature of a weight-stream addressing bug, not an
  arithmetic one.
* **Two tiles running the same kernel need `ctx.func`, not
  `aie.external_func`.** The declaration is module-scope, so the second gate tile
  is a redefinition and `aiecc` rejects the parse. `iron/attn.py` uses `ctx.func`
  for the same reason across its four CUs.
* **A buffer written by two cores needs its credits counted in cores.** The
  outbound BD acquired one credit while both cores release one per token, so it
  sent half-written rows and the counts drifted: tokens 0–4 exact, token 5
  partial, 6–7 never arrived, and the run sat until XRT gave up at 62 s. A hang
  whose first few tokens are *correct* is credit arithmetic, not routing.
* **Two producers cannot share an S2MM channel** unless they are one merged
  packet flow — and a circuit flow and a packet flow into one channel is rejected
  outright (`masterset op targets same destination DMA`). Every landing channel
  on the replay tile is packet-switched for that reason.
* **`make` does not treat env vars as dependencies**, so `GATE_SWAP_HALVES=1 make
  bitstream` reuses a stale MLIR and re-runs the identical bitstream — the
  control passes and proves nothing. Delete the MLIR first and check with
  `grep -c GATE_SWAP build/QWEN3_8_27B/mlir/layer.mlir`.

## Adding the next stage

Each of these routes and builds on its own; do not batch them. Three things move
together for every projection: the `mvm()` call in `kernels/proj_main.cc`, its id
in `q4_proj_engine.pkt_ids`, and its `_move_weights()` call in the host sequence
— the weight stream is consumed positionally, so a mismatch hangs rather than
mis-compares.

1. ~~**Attention.**~~ Done — columns 4 and 5, the KV cache replay and the causal
   mask, verified across the context sweep above.
2. ~~**z, the output gate.**~~ Done — two half-width gate tiles on `CT[2][2]` and
   `CT[3][2]`, `z` parked in `MT[3]`, `QKV_REPEATS` at 28. The two are
   vertically adjacent and the emitter owns the assembled row; see below.
3. **o_proj, the residual, `post_attention_layernorm`.** The first stage that
   needs a **second `x_replay_phase`**, over `DQ` rather than `D`. Its x is
   already produced: the gate emitter's MM2S ch0 is left free to feed it.
4. **The MLP.** Only possible after the replay tile carries four phases, which is
   what `fanout` and `dedicated` exist for.

**[`PLAN_O_PROJ_AND_GLU.md`](PLAN_O_PROJ_AND_GLU.md) is the step-by-step for 3
and 4**, with the constraints that are not obvious from the code.

`aux_memtile` is now unused — `MT[3]` parks `z` instead, next to the qkv staging
buffer it shares a de-permute shape with. It is kept because the delta-net
project's copy stages two operands and is the closer starting point if the o
replay ever needs its own tile.

## Host

`host/qkv_decoding_layer.{hpp,cpp}` drives exactly the stages above. It replaces
`decoding_layer`, kept as `host/decoding_layer.*.l1ref`, which was the L = 1
whole-layer driver this project was copied with: it streamed one token, drove the
MLP this xclbin does not contain, wrote RTPs this design does not declare, and
sized q4_k weights with Q4_1's `5/8` bytes per weight. Driving a stage the array
does not contain is not a wasted transfer but a hang.

The rotary table is computed on the host, `cos(inv_freq * pos)` with theta 1e7
and no rope_scaling. It used to come from the dump's `verify.cos` / `verify.sin`,
but those are 8 rows at one absolute position and the context sweep needs `LC` of
them on a common origin. The formula was checked against the dumped table by
solving for its start position: it reproduces it at position 165 to 2e-3, which
is the dump's own bf16 rounding.

The table's layout is the ABI between three files that nothing cross-checks —
`set_rope_table()`, `iron/RoPE.py`'s buffer type, and `kernels/rope.cc`'s
pointer arithmetic:

```
[t * ROPE_DIM              .. + ROPE_DIM/2)   cos for token t
[t * ROPE_DIM + ROPE_DIM/2 .. + ROPE_DIM/2)   sin for token t
[L_TOKENS * ROPE_DIM       .. + DH)           q_norm.weight
[L_TOKENS * ROPE_DIM + DH  .. + DH)           k_norm.weight
```

Weight order is positional against `kernels/proj_main.cc`'s mvms: the fused
q | k | v matrix, then `z` at `w_z_offset`. `self_attn.q_proj.weight` is
`2 * DQ` rows — q in `[0, DQ)` and `z` in `[DQ, 2 * DQ)`, de-interleaved by the
q4nx converter — and because q4nx blocks are row-block major, splitting it is a
plain byte offset. The two halves are laid down as two separate rounds, so `z` is
*not* contiguous with q in the buffer even though it is in the checkpoint.

`host/host.cpp` computes the whole layer on the CPU — the golden is the
specification of where this is going — but scores only what the array produces.
`host/cpu_ref_attention_layer.cpp` is the standalone fp32 reference for the whole
layer; build it with `-DCPU_REF_STANDALONE`.

## Build

```
source ../../setup.sh      # FLM_Xclbin/setup.sh
make kernel                # compile kernels/*.cc
make bitstream             # not `make all`
make host
./run.exe -m QWEN3_8_27B
```
