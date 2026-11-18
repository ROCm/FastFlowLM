<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# o_proj, the residual, post_attention_layernorm and the GLU -- as built

**All three steps are done and on hardware. This is now a record, not a plan.**
The whole Qwen3.8-27B full-attention layer runs on the array, scored at
`h2 = h1 + down_proj(SwiGLU)` = **0.999989** (`ATTN_LC=30`), 26 ms for 8 rows.
Each stage went in as written below; the deviations are noted inline per step.

| stage | end-to-end | vs the NPU's own input |
|---|---|---|
| o_proj    | 0.999845 | 0.999995 |
| down_proj | 0.999622 | — |
| h2        | 0.999989 | 0.99999  |

What is left is the other three layers in four (`../gate_delta_net_verify`) and
stitching the two xclbins into a model.

An earlier attempt at o_proj is in that session's history and was **reverted** --
it was built on a wrong assumption (two gate tiles feeding one replay port). The
design below is the corrected one. Do not resurrect the old `parts` machinery.

---

## Where the build is now

```
IT[2] MM2S0 ─ x ────────────┐
IT[2] MM2S1 ─ ln.weight ────┴─> rms CT[0][3] ─> MT[2] x replay ─> MT[1] FIFO
                                                                    │ broadcast
                              MT[3] <── y tree ──── 16 proj cores <──┘
                                │ de-permute, rotate to | k | v | q |
                                v
IT[3] MM2S0 ─ cos|sin, q/k norm ──> RoPE CT[2][3] ─┬─> IT[4],IT[5] S2MM0  k | v
                                                   └─> 4 packets, q, to the CUs
IT[4],IT[5] MM2S0/1 ─ KV cache ─> MT[4],MT[5] ─> 4 attention CUs ─> o
IT[4],IT[5] S2MM1 <── attention halves (diagnostic)
                      MT[3] ─ z, halved ─┐   MT[4],MT[5] ─ o, halved ─┐
                                         v                            v
                     gate helper  CT[2][2]  <──────────────────────────┘
                     gate emitter CT[3][2]  (helper writes its half north)
                            │ ch0 (first)        │ ch1
                            v                    v
              MT[2] x replay, phase "o"      IT[2] S2MM0, the scoring tap
                            │ 10 replays
                            v
              MT[1] FIFO ─> 16 proj cores ─> MT[3] o de-permute ─┬─> IT[3] S2MM1
                                                        ch5 north │   (tap)
                                              rms CT[0][3] phase 2 <┘
                                                        │ h += o, then post-norm
                                                        v  (to_st, no DDR tap)
                              MT[2] "up_gate" ─ 68 replays ─> 16 proj cores
                                                        │ to_glu, a round at a time
                                                        v
                                              GLU CT[1][2]  silu(gate) * up
                                                        │ to_down_proj, in_block 32
                                                        v
                              MT[2] "down" ─ 10 replays ─> 16 proj cores
                                                        │ to_rms (o's flow, reused)
                                                        v
                                              MT[3] ─> rms phase 3: h += down
                                                        v
                                              IT[2] S2MM1  h2, the layer output
```

Array map:

```
col       0       1        2         3          4      5     6       7
CT[0]   proj    proj     h 0..3     rms         qk     qk   proj    proj
CT[1]   proj    proj      GLU       h 4..7      kv     kv   proj    proj
CT[2]   proj    proj   gate helper  RoPE        qk     qk   proj    proj
CT[3]   proj    proj   gate emitter   -         kv     kv   proj    proj
MT     y asm   y asm +  x replay   qkv stage   kv $   kv $  y asm   y asm
               x FIFO   4 phases   + z park
                                   + o/down
IT     w in    w in    gated out   rope w in   kv in  kv in w in    w in
                       (tap)       o + down    attn   attn
                       h2          out (tap)   out    out
```

**CT[0][2] and CT[1][3] are memory only and must stay that way** -- they are the
rms core's `h` lenders, and a core on either would take the banks. That leaves
CT[1][2] (the GLU) and CT[3][3] as the array's only free compute tiles.

MT[3] is at 304 KiB of 512: 128 qkv + 96 z + 80 o, with down reusing o's buffer.
MT[2] is 368 KiB: 96 shared across qkv_z / o / up_gate, plus down's dedicated
272. MT[3]'s BD budget is the tight one -- 19 of 24 even, 18 of 24 odd -- and the
parity split is what picks channels there, not preference.

---

## Step 1 -- o_proj (**done**, commit `38b5e04d`)

Two things came out differently from the text below, both in 1d:

- **The staging buffer went on `qkv_stage_memtile`, not `aux_memtile`.** MT[3] in
  this project is `iron/qkv_stage.py`; `aux_memtile` is dead code here and was
  left alone. Channels are S2MM ch2 in and MM2S ch3 out, lock ids 6/7.
- **The DDR tap is the only consumer, so the chain is `ready -> free`**, not
  `ready -> ddr -> free`. The third link appears when step 2 adds the rms leg.

The context sweep and both negative controls are in the "How to verify" section,
updated with o_proj's numbers.


`o = W_o * gated`, `[MODEL_DIM][DQ]`. **K is DQ, not MODEL_DIM** -- the first mvm
in the layer for which that is true.

Three things move together or the run hangs rather than mis-compares: the `mvm()`
in `kernels/proj_main.cc`, its id in `q4_proj_engine.pkt_ids`, and its
`_move_weights()` in the host sequence. The weight stream is positional.

**1a. Kernel.** `kernels/proj_main.cc`, after the z mvm:

```c
mvm(MODEL_DIM / MVM_CORES, DQ, ..., pkt_id_to_rms);
```

`kernels/aie_array_layout.h` already has `O_DOWN_REPEATS = 10`. Add totals and
assert on them -- **it is the sum over all mvms that must stay even, not each
term**, because `is_y_ping` / `is_x_ping` are static across invocations:

```c
constexpr int TOTAL_ROW_BLOCKS = QKV_REPEATS + O_DOWN_REPEATS;              // 38
constexpr int TOTAL_X_SLICES = QKV_REPEATS * (MODEL_DIM / QXNX_COL_BLOCK_SIZE)
                             + O_DOWN_REPEATS * (DQ / QXNX_COL_BLOCK_SIZE); // 800
```

`MAX_MVM_K` in proj_main.cc is already `max(MODEL_DIM, DQ)`, so the scratch fits.

**1b. The replay phase.** Add a second `x_replay_phase` in `iron/layer_aie.py`:

```python
x_replay_phase(name = "o", x_dim = model_config.DQ,
               replays = model_config.D // 16 // model_config.q4nx_row_block_size)
```

It is an ordinary **single-producer** phase -- the gate emitter sends the whole
assembled DQ row. `iron/proj_tile.py` is the verified delta-net copy and needs
**no change**; keep it that way.

The two phases time-multiplex one 96 KiB buffer, ordered by the prod_lock credit
chain. That works because o's x cannot exist until the whole attention column has
run, long after qkv|z's last replay.

**1c. Feed it.** The gate emitter's MM2S **ch0 is deliberately left free** for
this; ch1 is the DDR tap (`gate_tile.Attn_Out`). In `iron/gate.py` add a second
outbound BD on ch0, chained *ahead* of the DDR one -- it is the critical path:

```
ch0  acquire gated_attn_out_cons_lock (halves)  -> release a hand-over lock
ch1  acquire that hand-over lock (halves)       -> release gated_attn_out_prod_lock
```

Both counts are `self.halves` = 2, not 1 -- see Gotchas. Expose ch0 as a new
property and wire it:

```python
connect_port(gate_emitter.Attn_Out_replay, x_replay_mt.x_in(1),
             pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj, keep_pkt_header = False)
```

**Must be a packet flow.** The rms tile already reaches that tile's other landing
channel by packet, and a switchbox will not drive one DMA channel from both a
circuit and a packet flow (`masterset op targets same destination DMA`).

**1d. Stage o for the rms tile.** o comes back row-block permuted and the rms tile
wants whole D-wide tokens, so de-permute in MT[3] beside qkv and z: a third
buffer (`L * D` = 80 KiB, taking MT[3] to 304 KiB of 512), S2MM ch2 in with
`[(o_blocks, m), (L, D), (m, 1)]`, MM2S ch3 out to the rms tile, MM2S ch4 to DDR
for scoring. Lock ids 6/7/8, chain `ready -> ddr -> free`.

**Use one BD per outbound chain, not L.** o is contiguous token-major, so the
whole buffer goes in one BD. MT[3] runs out of BD ids otherwise -- even channels
may only use bd_id < 24 and odd channels the rest.

**1e. Host.** Third weight round, `Dout = D`, `Din = DQ`:

```cpp
w_o_offset = w_z_offset + q4k_bytes((size_t)DQ * D);
reorder_cpy(dst + w_o_offset, w_o, DQ);           // DQ, not D -- K is DQ
_move_weights(seq, D, DQ, w_o_offset / sizeof(bf16));   // ELEMENTS, not bytes
```

**Issue that `_move_weights` after `_replay_kv_cache(seq)`, not with the other two
rounds.** It waits on its own transfers between rounds, and the cores cannot
touch o's weights until the gate has filled o's replay phase -- which needs the
attention output, which needs the replay. Issuing it earlier is a cycle: the host
blocks before the replay the array is waiting for.

Add an `o` region to `hidden_states_buffer` and a `_receive_o()` on IT[3] S2MM
ch1. D = 5120 needs two axes of one descriptor (`{1, L, 4, D/4}` /
`{0, D, D/4, 1}`) -- a 10-bit dim halves to 2048 for bf16. Never two BDs: BDs on a
channel consume the stream in sequence and the first would swallow whole tokens.

Score `o` in host.cpp against the golden's `o_proj`, which is already computed.

**Verify o_proj alone before going on.** Expect attention and gated unchanged.

---

## Step 2 -- the residual and post_attention_layernorm (**done**, commit `c93df858`)

Two things came out differently from the text below:

- **The residual leg is MM2S ch5, not ch3**, and the channel is forced. Memtile
  BD ids split by parity -- even channels take `bd_id < 24`, odd ones the rest --
  and MT[3]'s even half was already at 19 of 24 after step 1, so another 8-BD
  chain there overflows. On the odd side it is 10 + 8 = 18. The o DDR tap keeps
  ch3 and stays, because it is the only thing that still makes o separately
  observable; the credit chain is `ready -> ddr -> free`.
- **`rms_pkt_id.to_st` is now `to_out`.** With no MLP the post-attention norm is
  the layer's output and goes to DDR (IT[2] S2MM ch1), not west to the replay
  tile. The value is unchanged; step 3 is what moves the destination back.

**The scored cosine goes *up* at this stage** -- 0.99984 against gated's 0.99925
-- because the norm divides out a common scale factor. A high number is therefore
weaker evidence here than upstream, and a residual that silently dropped `h`
would still look fine. The run prints a CPU-side negative control for exactly
this: the same NPU rows scored against a golden built without the residual,
which comes out at **0.282**. Read the two together.


`kernels/rms_residual.cc` phase 2 is already written and commented out, and `h`
is already resident on CT[0][2] / CT[1][3] from phase 1. Uncomment it.

- x arrives on the **same** S2MM channel as phase 1, as a packet flow for the
  reason in 1c. The tile does not distinguish the phases; the layer's data
  dependencies order them.
- **MT[3] ch3 is taken by o's DDR tap**, which step 1 added. The residual leg is
  `L` chained BDs walking the offset, one row per packet, so it cannot share
  ch3's single whole-buffer BD -- give it its own channel and extend the credit
  chain to `ready -> rms -> ddr -> free` (add a third lock, id 8). Or drop the
  tap and reuse ch3, which costs the ability to score o directly; keep it until
  the residual itself scores, then decide.
- The **output** is distinguished, by the packet id the kernel writes per row:
  `pkt_id_rms_to_proj` (phase 1) to the x replay tile, `pkt_id_rms_to_it`
  (phase 2) to DDR. Wrong id here does not mis-compare -- it sends the layer
  output into the projection engine's x buffer and hangs.
- Host sends `2 * D` of norm weight (input then post-attention); the tile's
  self-chained w BD reloads its single D-sized buffer between phases. The length
  must track the live phase count exactly: too few and phase 2 parks on the w
  acquire, too many and the extra desynchronises the next invocation.
- Ping/pong parity: `is_x_ping` is static, so rows per invocation must stay even.
  Two phases = 16. Fine.
- `_receive_post_norm()` on IT[2] S2MM ch1, same two-axis shape as `_receive_o`.

---

## Step 3 -- the MLP / GLU (**done**, commit `9e7d55b5`)

It went in as written. `iron/glu.py` and `kernels/glu.cc` were already present,
unused, and byte-identical to the verified delta-net copies but for two missing
`dma_bd_packet` stamps on `Hid_out` -- so the tile itself needed only those.

Three things worth knowing that the text below does not say:

- **The GLU tile went to CT[1][2].** Only CT[1][2] and CT[3][3] are genuinely
  free: CT[0][2] and CT[1][3] look idle in the array map but are the rms core's
  `h` lenders and have to stay coreless. CT[1][2] is in the replay tile's own
  column, so hid's hop back is one switchbox.
- **`rms_pkt_id.to_st` stops going to DDR, and that is forced.** `rms_core.Y_out`
  carries at most two destination sets; both norms share the westward one, so
  the post-attention norm's tap had to be *removed* to make room for up|gate's
  leg, not joined by it. The norm is consequently no longer separately
  observable -- `down_proj` covers it, reducing over it through two projections
  and a SwiGLU, which 0.9996 is not reachable without.
- **down's y needs no wire of its own.** It rejoins o's flow, id, buffer and BD
  on MT[3], so the MLP costs that tile nothing -- which matters, because its BD
  budget was already at 19/24 even and 18/24 odd after step 2.

The original plan follows. Only after the replay tile carries four phases, which
is what `fanout`, `dedicated` and `in_block` in `iron/proj_tile.py` exist for:

- **up|gate**, `2 * INTERMEDIATE_SIZE` over MODEL_DIM, 68 replays. A memtile lock
  saturates at 63, so it needs `fanout = 2` (a cycle of 2 BDs relaying one
  credit, 34 each).
- **down**, MODEL_DIM over INTERMEDIATE_SIZE, 10 replays, and the one
  `dedicated` phase -- the SwiGLU result is produced by the up|gate mvms
  themselves, so its buffer is written while an earlier phase is still replaying.
  It also needs `in_block = m`: the GLU output inherits the engine's row-block
  permute, so the landing BD un-permutes.
- `iron/glu.py` and `kernels/glu.cc` exist and are unused; `aux_memtile` is also
  unused and is the closer starting point if a phase needs its own tile.
- down's K = INTERMEDIATE_SIZE does not fit one scratch hole -- see the two-half
  split in `../gate_delta_net_verify/kernels/proj_main.cc`.
- rms phase 3 (`h += down`) then closes the layer; that makes 24 rows per
  invocation, still even.

Three things step 2 leaves for it, none of them hard but all easy to miss:

- **`rms_pkt_id.to_out` has to become a west destination again.** The
  post-attention norm currently goes to DDR because nothing on-chip wants it;
  with the MLP it is up|gate's x and goes to MT[2] instead. That is a change of
  destination, not of id -- and `rms_core.Y_out` is already at its two-set
  ceiling, so the DDR tap has to *go*, not be joined by a third.
- **The host's norm weight length becomes 3 * D only if phase 3 took a weight.**
  It does not -- phase 3 is a bare residual add -- so the length stays at 2 * D
  while the row count goes to 24. The w BD still fires exactly twice.
- **MT[3]'s BD budget is nearly spent** (19 even, 18 odd of 24 each). down_proj's
  y lands here too, under the same `to_rms` id and reusing o's buffer and BD, so
  it costs nothing -- but a phase that needs its own channel will not fit, and
  `aux_memtile` is the unused tile to reach for.

---

## Gotchas that cost time in this session

- **A shared buffer written by N cores needs its credits counted in cores.** The
  gate's outbound BD acquired 1 while two cores each release 1 per token: it sent
  half-written rows *and* drifted. Tokens 0-4 were exact, token 5 partial, 6-7
  never arrived, and the run sat until XRT's 62 s timeout. `halves = 2` fixed it.
  A hang whose first N tokens are correct is a credit-arithmetic bug.
- **`make` does not treat env vars as dependencies.** `GATE_SWAP_HALVES=1 make
  bitstream` silently reuses a stale MLIR and re-runs the identical bitstream, so
  the negative control *passes* and means nothing. Always
  `rm build/QWEN3_8_27B/mlir/layer.mlir` first. Verify with
  `grep -c GATE_SWAP build/QWEN3_8_27B/mlir/layer.mlir`.
- **Two producers on one S2MM channel must be one merged packet flow**, and even
  then arrival order is a timing race. Rejoin in neighbour memory instead.
- **Mixing a circuit flow and a packet flow into one DMA channel is rejected**
  (`masterset op targets same destination DMA`).
- **`_move_weights`'s offset is in bf16 elements**; every other offset into that
  buffer is bytes. The byte offset reads twice as far and streams past the end,
  which the cores dequantise into NaN. All-NaN output with upstream still at
  0.9997 is a weight-stream *addressing* bug.
- **Memtile BD dimension sizes cap at 512 for bf16** (10 bits, doubled on
  lowering); shim dims cap at 2048. Split across axes of one descriptor, never
  into two BDs.
- **MT[3] BD ids**: even channels get bd_id < 24, odd channels the rest. 8-BD
  chains exhaust it; use one BD where the data is contiguous.
- **Two tiles running one kernel need `ctx.func`**, not `aie.external_func`.
- **`#if` on a `constexpr` is silently false** -- use `if constexpr`.
- **`npu_dma_wait` is per shim *and* per channel.**
- y-tree packet ids are one-hot and load-bearing; an id nothing emits still costs
  a mask/match rule on every port. Add ids with the mvms that use them.

---

## How to verify

```bash
source ~/setup.sh                 # not the in-repo FLM_Xclbin/setup.sh
make kernel && make bitstream && make host    # not `make all`
./run.exe -m QWEN3_8_27B
```

Context sweep, one binary: `ATTN_LC=8 14 30 40 56` (context 0, 6, 22, 32, 48).
Current clean numbers, measured with o_proj on the array -- the gated and
attention columns are unchanged from before it landed, which is the check that
matters when adding a stage:

| ATTN_LC | context | gated | attention | o_proj | down_proj | h2 |
|---|---|---|---|---|---|---|
| 8  | 0  | 0.9994   | 0.99968  | 0.999892 | 0.999818 | 0.999986 |
| 14 | 6  | 0.999093 | 0.999651 | 0.99991  | 0.999604 | 0.999984 |
| 30 | 22 | 0.999246 | 0.999698 | 0.999845 | 0.999622 | 0.999989 |
| 40 | 32 | 0.999201 | 0.99973  | 0.999929 | 0.999521 | 0.99999  |
| 56 | 48 | 0.999501 | 0.999712 | 0.999956 | 0.999905 | 0.999995 |

Two of the printed lines are *isolated* scores -- the stage re-run against the
NPU's own input rather than the golden's, both ~0.99999 and flat across the
sweep:

- `o_proj vs W_o * (the NPU's own gated rows)`
- `h2 vs h1 + the NPU's own down`

The columns above charge each stage for everything upstream of it; these do not,
so a split between the two localises a fault without another build.

Negative controls (the first two need no rebuild):

- **the residual**, printed every run: the same NPU rows scored against a golden
  with no residual term. **0.37** against h2's 0.99999. This is the control that
  matters most at the end of the layer, because h2 scores *higher* than the rows
  feeding it -- adding a large h1 to a small down raises the cosine -- so a
  dropped `h` would otherwise pass. It is also the only check on this path that
  sees the resident row at all.


- `GATE_SWAP_HALVES=1` (build-time, delete the MLIR first) crosses the z halves:
  gated 0.9992 -> **0.217** (halves 0.33 / 0.19), attention unchanged. Note the
  recovered gate stays a valid sigmoid in (0,1) even when swapped, which is why
  the run divides the gated row by its own attention row and compares against
  `sigmoid(z)` rather than just checking the range.
- `ATTN_POISON=<row>` corrupts one seeded KV row; at `ATTN_LC=40` it moves gated
  to ~0.989 for rows in any chunk, proving the replay really reads them.
  **It barely moves `o` (0.99993 -> 0.99963), and that is not o_proj ignoring
  its input.** The poison perturbs a DC component within each head, and `W_o`'s
  gain along that direction is far below its average, so a large change in the
  gated row is a small one in o. The isolated score is what settles it: it stays
  at 0.999995 with the poison on, meaning the poisoned rows went in and the
  matching o came out. Read the two scores together or this control misleads.

RTP addresses are per-build and read from
`build/QWEN3_8_27B/xclbins/rtp_addresses.txt`. Never hardcode them: a stale one
does not fail loudly, it hangs in `qds_device::wait()`.
