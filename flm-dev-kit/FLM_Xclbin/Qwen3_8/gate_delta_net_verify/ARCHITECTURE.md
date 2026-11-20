<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# The gated delta-net layer on npu2

Design notes for [`iron/layer_aie.py`](iron/layer_aie.py). The Python file is the
wiring list; everything that explains *why* a wire is where it is lives here.
Comments in the source point at the section headings below.

- [1. What this xclbin is](#1-what-this-xclbin-is)
- [2. Tile map](#2-tile-map)
- [3. The activation schedule](#3-the-activation-schedule)
- [4. DDR sinks, and what is free](#4-ddr-sinks-and-what-is-free)
- [5. Stage wiring](#5-stage-wiring)
  - [5.1 Stage 1 — input_layernorm](#51-stage-1--input_layernorm)
  - [5.2 Stages 2/3 — the qkv projection](#52-stages-23--the-qkv-projection)
  - [5.3 Stage 4 — the a / b projections](#53-stage-4--the-a--b-projections)
  - [5.4 Stage 5 — the causal conv1d](#54-stage-5--the-causal-conv1d)
  - [5.5 Stage 6 — the z gate projection](#55-stage-6--the-z-gate-projection)
  - [5.6 Stage 7a — the delta rule and the triplet arena](#56-stage-7a--the-delta-rule-and-the-triplet-arena)
  - [5.7 Stage 7b — the gated norm](#57-stage-7b--the-gated-norm)
  - [5.8 Stage 7c — ssm_out_proj, residual, post-norm](#58-stage-7c--ssm_out_proj-residual-post-norm)
  - [5.9 Stages 7d/7e — up|gate and the SwiGLU](#59-stages-7d7e--upgate-and-the-swiglu)
  - [5.10 Stage 7f — down_proj and h2](#510-stage-7f--down_proj-and-h2)
- [6. Routing constraints](#6-routing-constraints)
- [7. Round trips that were removed](#7-round-trips-that-were-removed)
- [8. Tapping a stage for a bisect](#8-tapping-a-stage-for-a-bisect)

---

## 1. What this xclbin is

One gated delta-net layer of Qwen3.8-27B — the linear-attention layer type only.
The every-fourth full-attention layer is built by the sibling project
`../../gated_attention_verify`, so there is no phase select anywhere in this
design.

The whole layer is on the array: `input_layernorm`, the qkv projection, the
causal conv1d, the z gate projection, the a / b projections, the gated delta
rule, the gated RMSNorm, `ssm_out_proj`, the residual add,
`post_attention_layernorm`, the fused up|gate projection, the SwiGLU,
`down_proj` and the second residual.

`L_TOKENS` rows of `MODEL_DIM` stream in on `IT[3].MM2S[0]` and the
`input_layernorm` weight on `IT[3].MM2S[1]`.

## 2. Tile map

| tile | IP | notes |
| --- | --- | --- |
| `CT[0][3]` | `rms_residual_tile` | three phases; keeps h resident in `CT[1][3]` and `CT[0][2]` |
| `CT[1][5]` | `AB_tile` | alpha / beta, q8_0, one resident x |
| `CT[0][5]` | `Conv` | causal conv1d, packs (k\|v\|q) triplets |
| `CT[3][3]` | `DeltaNet_Norm` | gated RMSNorm's norm half |
| `CT[2][3]` | `gate_tile` | its silu(z) half |
| `CT[3][5]` | `glu_tile` | SwiGLU, one round at a time |
| `CT[*][0,1,6,7]` | `q4_proj_engine` | 16 mvm cores, q4_k weights off those four columns' shims |
| `MT[0,1,6,7]` | proj assemble memtiles | `MT[1]` holds the single broadcast x FIFO |
| `MT[2]` | `proj_x_replay_mt` | every x the engine reduces over |
| `MT[3]` | `aux_memtile` | de-permutes z, o and `ssm_out_proj`/`down_proj` y |
| `MT[4]` | delta-net column memtile | S in only — nothing comes back |
| `MT[5]` | `linear_aux_memtile` | conv staging, conv weights, `ssm_norm` |
| `CT[*][4]` | `delta_net_column` | 4 cores, S resident per head |

The rms tile keeps the layer's hidden state resident in the memory modules of
two neighbours it can address directly — `CT[1][3]` to the north, which carries
no IP, and `CT[0][2]` to the west, in the column that has no compute tiles.
Four D-sized rows each. Neither tile gets a core or a mem, so nothing else in
`layer_aie.py` has to know they are in use. See `rms_residual.py`.

The rms tile hands each normalised row to two consumers off the *same* y
buffer, on two MM2S channels:

- **ch0** (offset 14, len D+2) — the packet-framed row to the proj engine.
- **ch1** (offset 16, len D) — the bare payload to the a/b tile over a circuit
  flow.

Because both channels read one buffer, the rms kernel releases both consumer
locks and then waits on both producer locks before writing the next row — and
both producer locks are init 0. A stalled a/b path therefore stalls the whole
layer at row 0. See `rms_residual.py`.

## 3. The activation schedule

`MT[2]` (`proj_x_replay_mt`) holds every x the projection engine reduces over.
The engine holds no x of its own. Column 2 is otherwise idle in this design.
The four phases, in the order the layer's data dependencies impose:

| phase | x width | replays | note |
| --- | --- | --- | --- |
| `qkvz` | D | 32 | qkv (DQ+DK+DV) then z (DV) — the C++ `L_QKV_REPEATS` |
| `o` | `linear_DV` | 10 | the gated delta-net rows; `O_DOWN_REPEATS = D / (32*16)` |
| `up_gate` | D | 68 | fused up\|gate, `2 * intermediate_size` rows; `UP_GATE_REPEATS` |
| `down` | `intermediate_size` | 10 | the SwiGLU result |

Replay counts must match the total of the M / m loops in `proj_main()`.

The first three share one 96 KiB buffer and take it over in order: o cannot
start landing until qkv|z's 32nd replay is out, and the post-attention norm does
not exist until o has been through `MT[3]` and the rms tile's second phase.
`up_gate` and `qkvz` are both D wide, so they also share the inbound channel; it
runs a 2-BD relay cycle so that 68 fits a 63-count lock (see
`proj_x_replay_mt`), and qkv|z's 32 come along as 16 trips of 2.

`down` is the one that cannot share. It is produced *by* the up|gate mvms, a GLU
round at a time, so it lands while up|gate is still replaying. It gets a
dedicated 272 KiB buffer (368 of the tile's 512) and an inbound channel to
itself, and it is the only phase with a landing un-permute.

### `down`'s `in_block`

The GLU output is permuted, but the contiguous run is a *row block*, not a
round. A round is `M_PER_ROUND = 512` columns laid out `[core][token][row]`
(`kernels/glu.cc:17-18`), so core `c` owns columns `[32c, 32c+32)` and the token
index turns over every `q4nx_row_block_size = 32` words, 16 times per round.

That is the same 32 the hid tap un-permuted with — `{17,32,L,32}` /
`{1024,32,cols,1}`, which scored 0.999616 on hardware — so it is the order the
GLU really emits. `in_block = 512` would land the round transposed against the
token axis.

## 4. DDR sinks, and what is free

Two sinks, both real outputs rather than diagnostics:

- **h2** on `IT[3].S2MM[1]`
- **the conv window** on `IT[5].S2MM[0]`

There used to be a third, the recurrent state on `IT[4].S2MM[0]`. It is gone.
The cores keep S resident for all `L` steps by design, so the only state they
could ever write back is S after *all* `L` tokens — and speculative decode needs
S after whichever prefix verification accepts, which is exactly the case that
write-back cannot serve. The host replays the accepted prefix from the triplet
arena instead (`get_all_S_at()`), which already holds the identical bf16
`k`/`v`/`alpha`/`beta` the cores consumed. Removing it drops a 3 MB DDR
write-back per pass, 128 KiB of `MT[4]` (`MT_S_OUT_buffer` ×2), four memtile
lock ids, a per-core inbound BD and a whole shim channel.

S is still sent **in** every pass — the scan has to start somewhere. The state
path is one-way now, not absent.

`kv_cache` is therefore `| conv in | S in | conv out |`. `advance_state(j)`
rolls both carried states to the accepted prefix: the conv half slices the
emitted window block, the S half replays and writes over the `S in` region.
It used to copy the device's write-back and silently ignore `j` for S, so a
caller that accepted fewer than `L` tokens had to know to rebuild the state
first; it no longer can get that wrong.

Two device-side couplings had to be unpicked, neither visible from the host:

1. In `Gated_Delta.py` the core's `MM2S[0]` carried **S and `o` on one chain**,
   sequenced by a third lock (`CT_S_Finsh_O_start_lock`): the S BD released what
   the `o` BD acquired. Dropping the S BD naively leaves `o` waiting on a release
   that never comes — the column stalls and the layer emits nothing. `o` now
   acquires `CT_S_O_out_con_lock`, the lock the *kernel* releases; the credit
   balance is unchanged (kernel takes `prod` once per head and returns `con`,
   the BD takes `con` and returns `prod`).
2. The host sequence **waited** on `IT[4].S2MM[0]`. Nothing writes that channel
   now, so the wait had to be deleted rather than moved — it would block forever.
   It cost no ordering: the conv-window wait just below it is the same barrier,
   since `MT[5]` does not release the window until all `L` sliding windows have
   been taken, and h2 is downstream of the scan through `o` regardless.

Removing it did not change the numbers (h2 0.999963, S 0.999874, identical to
before) and did not change the measured 29 ms pass latency — the write-back was
not on the critical path, it was overlapped with compute. The win is MT space,
a shim channel and DDR traffic, not time.

A conv window is `CONV_SIZE - 1` *pre-conv* qkv rows — what the next pass
convolves its first token against, and the same layout it went in as, so one
descriptor shape serves both directions. The pass reads **one** window in and
writes **every accept length's** window out, because speculative decode does not
know at dispatch time how many of the `L` drafted tokens verification will keep:
accepting `j` of them means the next pass starts at token `j` and wants the
window as of `j`, not as of `L`. Those windows overlap, so their union is one
contiguous `L + CONV_SIZE - 2` row run and `get_conv_out(j)` indexes into it.
See [5.4](#54-stage-5--the-causal-conv1d).

The recurrent state is the exception: only the accept-all state comes back.
Emitting all `L` does not fit — eight 64 KB state buffers is the whole 512 KB of
MT[4], and its S-out channel's BD bank has three ids free — and it does not need
to. The arena below already holds the exact bf16 `k`, `v`, `alpha` and `beta` the
cores consumed, and `kv_cache` holds `S_0`, so `get_S_at(head, j)` replays

```
u_t = beta_t * (v_t - alpha_t * (S_{t-1} @ k_t))
S_t = alpha_t * S_{t-1} + k_t (x) u_t
```

on the host. (`beta` is folded into `u_t`, not applied again to the outer
product — doing both would square it.) Replaying all `L` steps reproduces the
device's own state at cos 1.0 / NMSE 6e-11, which is the self-test that licenses
trusting the shorter prefixes; each `j` then scores ≥ 0.9998 against the CPU
golden, against a previous-prefix control of 0.38–0.93.

The only other traffic to DDR is the triplet arena (`IT[6].S2MM[0]` +
`IT[5].S2MM[1]`), a round trip the array itself depends on — it is the
token-major → head-major transpose, and no tile on that path can do it. See
[5.6](#56-stage-7a--the-delta-rule-and-the-triplet-arena).

Every intermediate tap is gone, on both sides — the shim channels in
`layer_aie.py` and the tile-side halves in `Conv.py`, `gate.py` and
`aux_memtile.py`: z, the gated rows, the delta rule's o, `ssm_out_proj`, the
post-attention norm (0.999772) and the SwiGLU hid (0.999616). The norm's tap was
not merely tidying — see [6](#6-routing-constraints).

Free for the next bisect: `IT[1]`, `IT[2]`, `IT[3].S2MM[0]`, and now
`IT[4].S2MM[0]` — the state write-back released it. `IT[5].S2MM[0]` is not
among them; the conv writeback took it.

## 5. Stage wiring

Connection order matters — long connections get higher routing priority.

### 5.1 Stage 1 — `input_layernorm`

```python
connect_port(IT[3].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, ...)
connect_port(IT[3].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, ...)
```

Both inbound flows keep `it1_pkt_id.to_rms`: the two shim channels are
distinguished by the CT-side S2MM channel they land on, not by packet id.

### 5.2 Stages 2/3 — the qkv projection

The norm feeds the replay tile; the assembled qkv then z rows leave on the TB
port. Weights come in on `IT[0]`, `IT[1]`, `IT[6]`, `IT[7]` inside
`proj_engine.route()`.

The kernel runs two mvms per invocation off the same resident x, so the
assembled `L_TOKENS x (Linear_DQ + Linear_DK + Linear_DV)` qkv rows and then the
`L_TOKENS x Linear_DV` z rows leave back to back out of `MT[1]`. They carry
different packet ids (`to_conv`, `to_aux`) and part company at the memtile's
egress port: qkv into the conv staging memtile `MT[5]`, z into `MT[3]` to wait
out the delta scan as the gated norm's silu gate.

### 5.3 Stage 4 — the a / b projections

These do not go through the 16-core engine — 128 q8_0 rows is half of one core's
32-row granularity times the engine's 16-way split — so they run on a single
tile that keeps one x resident and streams the weights past it.

All three flows are circuit-switched: each is a single producer to a single
consumer, so there is nothing for a packet id to disambiguate.

```
rms -> ab   the same normalised row the proj engine gets
IT5 ch0     the a/b bias once, then the q8_0 weight stream
ab -> IT5   [alpha[64] | beta[64]] as f32, per token
```

`IT[5].S2MM[1]` is not a dump: since stage 7a the same BD scatters alpha and
beta into the arena's holes. The a/b tile also has to keep draining
`rms_core.AB_Yout` — see the two-producer-lock note in [2](#2-tile-map).

### 5.4 Stage 5 — the causal conv1d

```
proj -> MT5 QKV_in     L_TOKENS x CONV_WIDTH, de-permuted in the BD
IT5 ch1 -> MT5 W_in    the 4 conv weight taps, then the conv cache
MT5 -> conv X_in       the sliding 4-tap x window, per 128 channels
MT5 -> conv W_in       the fixed 4 weight taps, replayed per token
MT5 ch4 -> IT5 S2MM0   the outgoing window, the next pass's cache
```

x and w take separate channels because an L=8 sliding window over an interleaved
x/w layout is not an affine address pattern; see `linear_aux_tile.py`.

The writeback needs no buffer of its own: `conv_x_buffer` is
`(CONV_SIZE - 1) + L` rows, so *every* window the next pass might want is
already sitting in it, contiguous and in natural `[tap][channel]` order
(block[4] un-permuted it on the way in). The accept-`j` window is buffer rows
`j .. j + CONV_SIZE - 2`, and over `j = 1..L` that is rows
`1 .. L + CONV_SIZE - 2` — row 0 is the only row no accept length can want, as
it falls out of the window at `j = 1` already. So one BD reads the whole tail
straight out, 102400 words here, inside the memtile BD's 17-bit length field. A
per-`j` BD chain would cost `L` BDs and buy nothing, since the runs overlap and
the host can index.

What it does need is a lock of its own — `conv_cache_out_ready`, released by the
*last* sliding-window BD and acquired only by the writeback. Token `L-1`'s window
is the last read that touches the tail rows, so once it has been taken the
outgoing cache is final.

It is worth recording why the obvious alternative does not work, because it
builds, routes and looks balanced. Giving the writeback a share of the existing
`conv_x_ready` — release `L + 1`, have the writeback acquire one — deadlocks. A
lock is a *counting semaphore, not an assignment*: nothing binds the extra count
to the writeback. The ch5 window chain is circular, so after taking its `L`
counts it loops straight back to `block[10]` and races ch4 for the `(L + 1)`th.
ch5 wins, ch4 blocks forever, and the host's `IT5 S2MM ch0` wait never returns —
a 60 s XRT timeout with `-nan` on every output, not a wrong number. Chaining
fixes it by ordering the two instead of letting them compete. Qwen3.5's
`conv_new_cache_out_cons_lock` is the same pattern.

`conv_x_free` is seeded at `L` and acquired at `L` by the cache-fill BD:
`L - 1` counts return from the sliding windows and the last from the writeback,
so the *next* invocation's cache write cannot land on the tail rows before the
writeback has read them. That failure is the silent one — the second pass would
convolve against a half-overwritten window. Both counts stay well inside the
6-bit memtile lock counter.

The count the writeback hands back is the *last window's*, not one of its own,
and that is not a stylistic choice. It used to be `L + 1` seeded and `L + 1`
acquired, with all `L` windows releasing `conv_x_free` and the writeback adding
the extra — which put two releases on window `L - 1`'s BD, `conv_x_free` and
`conv_cache_out_ready`. An AIE2 BD descriptor has one release field, so one of
the two is dropped, silently, with no diagnostic anywhere in the lowering. See
[§6, one release per BD](#one-acquire-and-one-release-per-bd).

The writeback is scored per accept length: `host.cpp` compares each window
against the `CONV_SIZE - 1` pre-conv rows of the CPU golden ending at token `j`
(cos 0.99999 for all eight). h2 and S cannot stand in for it — both are computed
before the writeback, so a writeback that wrote nothing would leave them
unchanged. Each window is also scored against its *neighbour's* reference:
`CONV_SIZE - 1 = 3` means consecutive windows share two of three rows, so a
mis-indexed read would still look plausible. That control lands at 0.46–0.79,
which is what says the check has teeth.

The conv result leaves on `Y_out` packed head-major for the delta cores; the
flat per-token dump this stage was scored on is gone with the bench.

### 5.5 Stage 6 — the z gate projection

The proj engine's second mvm, off the same resident x, so nothing new is needed
on the way in — only somewhere for it to land. Stage 7b took that route away: z
is the gated norm's silu gate, so it goes to `MT[3]` and never reaches DDR.

### 5.6 Stage 7a — the delta rule and the triplet arena

The cores keep S resident for all `L_TOKENS` steps of a head, so they need the
`(k, v, q, alpha, beta)` stream head-major. Nothing on the array can transpose
it: `MT[4]`'s six channels each way are already spoken for by S and o, `MT[5]`
has neither the space nor a BD shape that can walk five differently-based chunks
per `(h, t)`, and the conv tile cannot hold 8 tokens x 10240 channels in 64 KB.
So DDR is the reorder buffer:

```
conv Y_out -> IT6 ch0   48 (k|v|q) triplets per token, scattered into the
                        arena's head-major slots
ab   Y_out -> IT5 ch1   alpha and beta into the two-word holes between k and v,
                        and between v and q
IT4 ch1    -> 4 cores   the arena read back; heads are contiguous, so linear
```

That is what buys the L=8 form: 3 MB of S in and 3 MB out per layer instead of
the ~49 MB a per-token S round trip would cost. The arena costs one extra DDR
round trip of 390 KB to save it.

The alpha/beta leg is load-bearing, not diagnostic — the delta kernel reads them
straight out of the holes (`gate_delta_net_decode_128_128_128_32.cc:103-104`,
`alpha_ptr = (float*)(cur_K_buffer + dK)`), and they drive
`u = beta*(v - alpha*S@K)`. Cutting it leaves the ctor's zeroed arena, hence
alpha = beta = 0 and S never updating.

`IT[6]`'s S2MM channels are otherwise free — it only sends weights.

### 5.7 Stage 7b — the gated norm

`o * rsqrt(mean(o^2)) * ssm_norm * silu(z)`.

Its two operands arrive in different permutations and at very different times —
o head-major and last, z row-block permuted and early — so `MT[3]` stages both,
de-permutes both to token-major in the inbound BD and hands them out one token
at a time. That keeps the norm and gate kernels at one row per invocation and it
means the gated result reaches DDR token-major. See `aux_memtile.py`.

```
proj  -> MT3 Z_in      L x Linear_DV, de-permuted in the BD
delta -> MT3 O_in      L x Linear_DV, de-permuted in the BD
MT3   -> norm DH_in    one o row per BD
MT3   -> gate Gate_In  one z row per BD
MT5   -> norm W_in     ssm_norm, replayed once per token
norm  -> gate Attn_In  the normalised row
```

Everything but the `ssm_norm` hop stays inside column 3. The stage-6 (z) and
stage-7a (o) regression dumps used to leave `MT[3]` as well, on a third pair of
MM2S channels chained behind the two above; both are gone, and with them
`MT[3]`'s `done` locks.

### 5.8 Stage 7c — `ssm_out_proj`, residual, post-norm

```
gate  -> MT2           the gated rows, one per token, as produced
MT2   -> 16 proj cores 10 replays of the whole L x Linear_DV activation,
                       column-block major
proj  -> MT3 OProj_in  o, de-permuted out of row-block order by the memtile BD
MT3   -> rms X_in      the o rows, one per BD, one hop north
rms   -> (phase 2)     rmsnorm(x + o) * post_attention_layernorm
```

Neither operand of the residual touches DDR. x is still on the array because the
rms tile saved each row into its neighbours' memory on the way past in phase 1
(see `rms_residual.py`), and o never leaves column 3.

Both x's come from `MT[2]` now and arrive on the same inbound port — the qkv|z
and o phases share a staging buffer, so nothing distinguishes them but arrival
order, and the lock chain enforces that. `Attn_Out` is still a packet flow
because the id is what the rms leg into the same port is told apart by; the two
producers cannot collide in any case, since `MT[2]`'s last qkv|z replay is
consumed to produce z and z has to clear the gated norm before `gate_core` has a
row to send.

o's y is `proj_pkt_id.to_rms`: east out of `MT[1]` alongside qkv (`to_conv`) and
z (`to_aux`), off at `MT[3]` — so all three ids part company at that tile's west
port — then one hop north into the rms tile's x port. That port is a merge: the
shim's phase-1 rows and these phase-2 rows land on the same S2MM channel under
different ids and are told apart by arrival order, which the layer's
dependencies fix (o cannot exist until phase 1 has normed every row).

### 5.9 Stages 7d/7e — up|gate and the SwiGLU

The same norm a second time, west into `MT[2]` as up|gate's x. It needs its own
`connect_port` even though `x_in(2)` is a port the `to_proj` leg already
reaches: the packet id is written by the *kernel*, at `y + 14`, not by a BD
attribute, and `rms_residual.cc` stamps `to_proj` in phase 1 and `to_st` in
phase 2. up|gate's x is the phase-2 row, so it travels under `to_st` and that
flow needs the destination.

The `up_gate` phase is the rest of it — it shares the main buffer and the D-wide
replay channel with qkv|z, so it costs no memory, no extra channel and no extra
id on the x side.

**The old wiring hung here.** `MT[2]` carried one broadcast port per activation
and the second never fired, so the proj cores parked on the x acquire at the top
of their 4th mvm. Everything else was ruled out by direct evidence — the norm
reaching `MT[2]`, the weight stream, the lock magnitude, the routing, the chunk
alternation, the DMA channel — leaving the second broadcast port itself. The
FIFO rewrite removed it: there is one circuit-switched x port now and every
phase looks the same on it. The same rewrite closed the routing failure that
came with it — see [6](#6-routing-constraints).

The GLU tile is one row-block-round wide rather than a whole hidden state: 8
tokens x `intermediate_size` is 278 KB, four times the tile, so the L = 1
design's replay-into-down-proj buffer cannot exist here. hid leaves a round at a
time and is caught on `MT[2]` in the `down` phase's own buffer. Both operands
carry the identical permutation, so the kernel never reshuffles; see `glu.py`.

### 5.10 Stage 7f — `down_proj` and h2

hid goes west into `MT[2]` as down's x under
`aux_to_proj_x_in_pkt_id.to_down_proj` — **not** `to_out_proj`. The gate tile
already sends `x_in(1)` under `to_out_proj`, and both flows enter `MT[2]` on the
same east slave port. Two packets with one id arriving on one slave and wanting
two different local DMAs is exactly what the arbiter allocator cannot express;
it hands the shared West master port two amsels on two arbiters and
`AIEDialect.cpp:2047` rejects it ("a master port can only be tied to one
arbiter", `dest_bundle` 4).

**down's y needs no wire of its own.** It carries `proj_pkt_id.to_rms`, the id
o_proj already uses, so it follows o's flow exactly: east out of `MT[1]`, off at
`MT[3]`, un-permuted by the same 3-dimensional BD into the same `op_buffer`,
then one row at a time north into the rms tile's x port. The buffer is free by
then — o's eight rows drained before the post-attention norm existed, and the
norm is four dependency steps upstream of down — and both of `MT[3]`'s op BD
chains are self-chained, so the second activation costs no channel, no BD and no
lock.

What tells the two apart is the rms tile's third phase, and it does so the way
the first two are told apart: by arrival order.

```
phase 3   h += down      -- the MLP residual, h1 -> h2
          y = h          -> DDR, the layer's output
```

No norm, so no third weight: the self-chained w BD still fires twice.
`3 * L_TOKENS = 24` x rows keeps the ping/pong parity even.

h2 leaves under a third id (`rms_pkt_id.to_out`) on the same port, straight
south into `IT[3]`, and it is the *only* thing that port sends south. See
[6](#6-routing-constraints) for why that matters.

## 6. Routing constraints

### One master port, at most two destination sets

`rms_core.Y_out` (`tile_3_2 DMA:0`) has three flows but only **two destination
sets**: both norms go west to the same `MT[2]` port, so
`AIECreatePacketFlows` merges `to_proj` and `to_st` into one group, and h2 goes
south to `IT[3]`.

A *third* set off this one master port does not route, whatever the id or the
destination. Measured with the pathfinder while the post-attention norm still
had a DDR tap of its own: `to_proj {west}`, `to_st {west, south}` and
`to_out {south}` made the allocator tie one master port to two arbiters, and
`AIEDialect.cpp:2047` rejects that. Retried with `to_out` on the same shim
channel as `to_st`, on `IT[1]`, `IT[2]`, `IT[5]`, `IT[6]` and under ids 3, 5, 8
and 16 — every one failed the same way. Only `IT[7]` (a route with no port in
common with either norm's) and dropping the norm's tap passed. The tap is what
went, because the layer wants its own output south, not seven columns east.

This is why the post-attention norm no longer reaches DDR. See `rms_pkt_id` in
`iron/constants.py`.

### One-hot packet ids in the y tree

`proj_pkt_id` values are one-hot on purpose. The router's rule generator widens
a mask to cover the ids that share an outgoing direction *without checking the
other ids on the same slave port*.

The original regression: a `to_glu = 15` flow heading east out of column 1 made
it emit `rule(16, 0) -> East` **ahead of** `rule(31, 5) -> South` on
`shim(1,0) North:3` — `to_rms` was 5 then, and o_proj's y still went to DDR.
First match wins in hardware, so o's y went east instead and never arrived.

The fix is the id's *value*, not the count: `to_glu = 16` is the only id in the
tree with bit 4 set, so a widened mask that admits it cannot admit anything
else. Stage 7c generalised that into the rule the whole tree follows — every id
one-hot — which is why `to_rms` reads 2 and not 5. Checked against the generated
code rather than assumed: adding `to_glu` widens exactly two masks in the design,
21 → 5 at `MT[1]`'s `DMA:5` and again one hop up at `tile(1,2)`, both of which
only drop bit 4 from the check, and every rule that carries `to_glu` itself is an
exact `(31, 16)`. See the `pkt_ids` comment in `proj_tile.py` for the rules in
full, and diff the physical MLIR against the previous build if these ids move
again.

### One acquire and one release per BD

An AIE2 BD descriptor carries a single lock-acquire field and a single
lock-release field. IRON will happily take two `lock_release` calls on one
`aie.dma_bd`; they survive `aie-translate` into `input_with_addresses.mlir` as
two `aie.use_lock(..., Release, 1)` ops, the xclbin builds and routes, and then
only one of them fires on hardware. There is no diagnostic at any stage.

This is what made the design single-dispatch. MT[5]'s last sliding-window BD
released `conv_x_free` and `conv_cache_out_ready` together; the writeback ran, so
`conv_cache_out_ready` was the survivor and `conv_x_free` came back one short
every pass. Pass one was perfect — the cache-fill BD had already armed off the
lock's init value — and pass two parked in the host's `IT5 S2MM ch1` wait for
XRT's 60 s timeout, which is indistinguishable at the host from a routing fault.

The bisect is worth skipping next time, because a hand-written lock ledger
closes on paper and every static check clears the design. Scan the lowered MLIR
for the shape instead:

```python
for b in re.split(r'\n(?=\s*\^bb)', open('input_with_addresses.mlir').read()):
    if 'dma_bd' not in b: continue
    if len(re.findall(r'use_lock\(.*Release', b)) > 1 or \
       len(re.findall(r'use_lock\(.*Acquire', b)) > 1: print(b)
```

The fix is to hand the count along the chain rather than double-release: drop
the second release and lower the consumer's acquire by one, so the next BD in
sequence supplies it. Here that also made the ordering explicit — the cache
write now waits on the writeback by construction instead of by counting.

`host.cpp`'s `FLM_PROBE_N` is the regression test. It re-dispatches the same
inputs `N` times on one hardware context and scores pass `n + 1` against pass
`n` word for word; the engine issues 48 per verify pass, so `FLM_PROBE_N=48` is
the number that matters. Note that the triplet arena is structurally sparse —
`arena_heads` is the *padded* v-head count and each 130-word triplet slot has an
unwritten tail, so 74.87% written is a healthy pass and a coverage gate on it
would fail every time. Coverage is checked on the conv writeback and h2, which
are dense; the arena is checked by comparison only.

### One multicast tree per hop on the x side

A distinct packet id per activation on the broadcast path does not route: each
id is its own packet flow, so the pathfinder grows each its own multicast tree
and the trees diverge. At `tile_6_3` that tied the core's `DMA:0` to arbiters 0
and 2 at once. Collapsing every replay onto one id, and then onto one
circuit-switched port, leaves a single tree per hop. See `proj_x_pkt_id` in
`iron/constants.py`.

## 7. Round trips that were removed

- **o_proj's un-permute (stage 7c).** o used to go out on `IT[1].S2MM[1]` and
  come straight back in on `IT[3].MM2S[0]`, un-permuted by the shim BD, because
  the *rms* tile cannot un-permute: the engine emits o in row-block order (32
  rows of one block for all L tokens, then the next block) and a compute tile's
  BD walks one buffer with no bank to spare. `MT[3]` is directly under the rms
  tile and un-permuting is what it already does for o and z, so the round trip
  is gone — 160 KB per layer and `IT[1].S2MM[1]` with it. The engine's y port
  loses its only *shim* destination in the bargain; `to_conv` still leaves
  `MT[1]` south, but into the conv staging memtile, not DDR.

  The cost is that o is no longer separately observable.

- **The SwiGLU hid tap.** `connect_port(glu.Hid_out, IT[5].S2MM[0])` scored
  0.999616. It costs a *second* direction off `Hid_out`, so it went with the
  rest once stage 7f closed.

- **The post-attention norm tap.** Removed under duress; see
  [6](#6-routing-constraints).

## 8. Tapping a stage for a bisect

`IT[1]`, `IT[2]`, `IT[3].S2MM[0]` and `IT[4].S2MM[0]` are free; `IT[5].S2MM[0]`
is the conv writeback now. `IT[4].S2MM[0]` is the newest of them and the most
useful for this column: it was the state write-back, so git history has a
working S-out path — buffers, locks, BDs and route — to recover if the delta
scan ever needs bisecting directly rather than through `o`.
Nothing is staged for a tap in-tree any more — the shim BDs and their un-permute
arithmetic, the `connect_port`s and the tile-side dump paths in `Conv.py` and
`gate.py` were all deleted once the layer closed end to end. Recover the one
you need from git history; each is self-contained, and a tap costs a DDR region
in `hidden_states_buffer`, one `connect_port` and one `npu_dma_wait`.

Two things to check before adding one: a *second* destination set off a master
port that already has two will not route ([6](#6-routing-constraints)), and a
new packet id in the y tree has to stay one-hot.
