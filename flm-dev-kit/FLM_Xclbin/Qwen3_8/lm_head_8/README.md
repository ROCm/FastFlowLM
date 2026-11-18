<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# lm_head_8 — verify-phase lm_head, L = 8

Final vocabulary projection for Qwen3.8-27B over **8 token rows per dispatch**,
for the verify phase of speculative decoding. The L = 1 decode variant is
`../lm_head_npu_bin`; the two share the Q8_0 kernel and the column topology.

## Result

Driven by `verify.layer_63_out`, scored against `verify.logits`:

```
Row 0: argmax npu=1156  ref=1156  OK;        cos=0.999958
Row 1: argmax npu=6587  ref=682   OK (tie);  cos=0.999966
Row 2: argmax npu=728   ref=728   OK;        cos=0.999981
Row 3: argmax npu=310   ref=310   OK;        cos=0.999981
Row 4: argmax npu=11290 ref=11290 OK;        cos=0.999971
Row 5: argmax npu=264   ref=264   OK;        cos=0.999983
Row 6: argmax npu=16547 ref=16547 OK;        cos=0.999975
Row 7: argmax npu=3377  ref=3377  OK;        cos=0.999978
Argmax match: 8/8
Time spent: 49 ms for 8 rows (6.1 ms/row)
```

**49 ms against 204 ms** for eight L = 1 dispatches — 4.2x — at the same
accuracy. The win is not arithmetic: it is reading the 169 MB of Q8_0 weights
**once** for all 8 tokens instead of once per token. Row 1 is an exact tie in the
bf16-rounded reference (two tokens at 24.75), scored by value rather than index.

**Repeat dispatches work.** 20 back-to-back dispatches, token slots rotated on
each, all score 8/8; steady state is ~31 ms per dispatch.

## What differs from L = 1, and why

npu2 core data memory is **64 KB** (`AIETargetModel.h:453`). At L = 8 the
activation is 8 x 5120 x 2 = **80 KB**, so the L = 1 trick — hold all of x on the
core and replay it across all 243 rounds — is impossible. x streams per 256-wide
column block, and the rest follows from that:

| | L = 1 | L = 8 |
|---|---|---|
| x on core | whole row, resident, 20 KB | one column block, ping/pong, 9 KB |
| x path | shim broadcast, packet-switched | per-column memtile slot, refilled per round |
| final rms_norm | on the core, in place, before the round loop | on the core, split across round 0's k loop |
| y per core per round | 32 | 8 x 32 |
| core memory | ~45 KB | ~37 KB |

**The final rms_norm is still on the core, but it cannot be done in place.** The
L = 1 kernel normalises the resident row before the round loop. Here no core ever
holds a whole row, so the reciprocal RMS is not available when x is first
consumed. RMSNorm splits to make that not matter:

```
logits[t][m] = SUM_i (x[t][i] * s[t] * nw[i]) * W[m][i]
             = s[t] * SUM_i (x[t][i] * nw[i]) * W[m][i]      s[t] = 1/sqrt(mean(x[t]^2)+eps)
```

The per-token scalar `s` commutes through the matmul, so it is applied to the
*finished* accumulator in `copy_y_scaled`. The per-column `nw` does not commute,
so it is folded into x inside `_Q80_x_bf16_L`, where x is loaded anyway. A sum of
squares is a reduction and so needs no residency: the k loop walks all
`K_BLOCKS` every round, so round 0 accumulates `ss` as it goes and `s` is ready
at exactly the moment round 0's `y_acc` is. Rounds 1.. reuse it.

`model.norm.weight` therefore rides in as a **ninth lane** of the per-round x
stream (`X_LANES = L + 1`) rather than living in a resident buffer. That is not
frugality, it is the post-mortem below: a one-shot fill BD is never revisited and
breaks the second dispatch. As a lane it is an ordinary per-round operand and
every round still looks identical to every other.

Cost, measured as a three-way A/B (norm on core / lane kept but norm stripped /
neither): **30.7 ms against 28.6 ms steady state, +7.3%**. The extra 10 KB of x
per round per column is free — 28.6 to 28.7 ms, inside the noise — so all of it
is core compute, in `_Q80_x_bf16_L`'s extra multiply and bf16 round per
(group, token). This xclbin's input is *raw* x, same as the L = 1 one.

**x left the shared shim port.** At L = 1, `IT[0].MM2S[0]` carries both the x
broadcast and column 0's group-b weights — safe only because x goes once, before
any weight is queued. Per-k-block x on that port deadlocks: a broadcast advances
at its slowest consumer, cores 2/3 of column 0 wait on weights queued *behind* x
on that same port, so they stop draining x and the port never advances. Each
column now takes its own copy of x into its memtile, which sinks all 90 KB and
frees the port.

## Data movement

```
DDR x [k_block][lane][256]        lanes 0..7 = tokens, lane 8 = model.norm.weight
  -> shim MM2S 1, ONCE PER ROUND, ahead of that round's group-a weights
  -> memtile x buffer, 90 KB
  -> memtile MM2S 0, one BD streaming the whole buffer to the column
  -> all 4 cores of the column, cut into 2304-element ping/pong by their own BDs
```

x is re-sent every round rather than filled once per dispatch, and is an ordinary
producer/consumer slot: the shim fills it, the replay channel drains it to the
cores, the slot comes back. Nothing distinguishes the first round of a dispatch
from any other, which is what makes the design re-dispatchable — see the
post-mortem below.

The replay is a single BD, not one per k-block. x is already contiguous in the
order the k-loop wants, so the *consumer's* BDs do the splitting.

y returns as `[round][col][core][token][32]`; the host un-interleaves it.

### Two ratios that have to hold

`x` shares shim channel 1 with the group-a weights, so the memtile's S2MM ring
must tile **exactly one round** of that channel or `x` drifts into a weight BD
and is dequantised as Q8_0 — NaN logits, ERT state 8, a 60 s dispatch.

1. **One x BD per round's worth of weight BDs.** The host sends one `x` and one
   weight transfer per round, but a round's weight transfer is `K_BLOCKS = 20`
   memtile slots wide against x's 1. A 1:1 ring drifts by 19 slots per round.
2. **A round must be a whole number of laps of the buffer cycle**, i.e.
   `K_BLOCKS / UNITS_PER_SLOT_A` divisible by `W_SLOTS_A`. Otherwise the MM2S
   chains begin round *r+1* mid-cycle and read from the wrong buffer. Both are
   asserted in `lm_head_memtile`.

To keep that ring short enough to fit, a group-a slot stages
`UNITS_PER_SLOT_A = 2` k-block pairs, so the ring is 11 BDs rather than 21.

### The BD budget is per parity, not per tile

A memtile has 48 BDs but they are **two fixed banks of 24**: even channels may
only use ids 0–23, odd channels 24–47 (`AIETargetModel.h:469`,
`isBdChannelAccessible`). A design at 42/48 total can still be rejected — this
one was, at 31 in the odd bank. The big x+weight ring is therefore on the *even*
S2MM 0 and group b on the odd S2MM 1, which is the opposite of the shim-side
naming. Current occupancy: **even 22/24, odd 14/24**.

Also note every memtile BD must carry **both** an acquire and a release
(`AIERT.cpp:304` asserts otherwise), so "lock on the first BD, release on the
last" is not expressible. A wide slot is instead handed out as
`UNITS_PER_SLOT_A` credits, one taken per BD.

### Shim BDs cap the lookahead at 2

A round issues four shim BDs (x, y, wa, wb) where the L = 1 design issued three.
`LOOKAHEAD = 3` would need all 16 shim BDs and leaves the sequencer no free id to
stage into — it hangs. `LOOKAHEAD = 2` uses 12 and costs ~3%.

## Where the speedup actually comes from

In `_Q80_x_bf16_L` the token loop sits *inside* the column loop. Converting a
weight column from int8 to bf16 is the expensive part of this kernel — it is
conversion-bound, not MAC-bound — and the
converted column does not depend on the token. One conversion now feeds 8 MACs.

## Test

```bash
source ../../setup.sh    # FLM_Xclbin/setup.sh
make bitstream           # NOT make all, which only builds the host
make host
./run.exe -m QWEN3_8_27B
```

Gate on **argmax 8/8**. Cosine is secondary, and being per-row it largely reports
the row norm.

**Neither of them can see the rms_norm.** A build with the norm stripped out
entirely, fed raw x, still scores 8/8 here at cos 0.9984 — argmax is
scale-invariant and `nw` barely rotates the vector.
The checks that do separate them (in the engine's port-time
driver) were: `||raw|| / ||normed||` must clear 1.5, and the device's output must
match a normed reference by *relative L2*, not cosine.

**Always run the repeat test before believing a change:**

```bash
FLM_LM8_REPEAT=19 ./run.exe -m QWEN3_8_27B   # 20 dispatches, slots rotated
FLM_LM8_REPEAT=same ./run.exe -m QWEN3_8_27B # repeat with identical x
```

A single dispatch cannot see per-dispatch state, and this design had a bug that
only appeared on dispatch 2 while still reporting `COMPLETED`. The rotation
matters as much as the repetition: a design that replayed the first dispatch's x
would return the first dispatch's argmaxes and score 8/8 against an unpermuted
expectation. The harness scores each slot against the row actually fed into it.

`FLM_LM8_LOOKAHEAD=n` overrides the shim BD depth, which is the first thing to
bisect if a change starts hanging.

The 8 argmaxes are all distinct, which is what rules out the failure mode where a
design silently computes row 0 eight times and still scores 8/8. Verified
directly: feeding row 0 into all 8 slots makes every output become 1156, so the
token lanes are independent and carry what is put into them.

## Post-mortem — the second dispatch used to return NaN

Worth keeping, because the failure was invisible to the obvious test.

The first version filled `x` from a **one-shot prologue BD**: the memtile ring
closed onto BD `n_prologue`, so — as `_build_dma_chains`' own docstring says —
the prologue was never revisited. `x_buffer` was written exactly once for the
life of the loaded xclbin. Dispatch 2's 80 KB of `x` fell through into a *weight*
BD, was dequantised as Q8_0, and produced NaN on every row.

There is no host-side reset available: `get_ready()`/`ready()` start and wait an
empty run, and the BD chains are static array configuration applied at xclbin
load, not instructions replayed per dispatch. So the prologue had to go.

Three things about how this presented:

- **It passed every single-dispatch test.** Verify dispatches once per
  speculation cycle, so dispatch 2 is the normal case, not an edge case.
- **Both dispatches reported `ERT_CMD_STATE_COMPLETED`.** Nothing hung and
  nothing errored; the DMA faithfully delivered x into the wrong buffer.
- **The decisive test was suppressing *only* the x transfer** on dispatch 2.
  That made it pass 8/8 and reproduce dispatch 1 bit-for-bit, which located the
  fault in the x path rather than in the weight or y streams — a same-x repeat
  fails too, so "stale x" was not the explanation.

The fix makes x a per-round streamed operand with the same lifetime as a weight
slot, and costs the ~24% extra bytes on shim channel 1 noted above.

## What the L = 8 win is actually worth

The 4.5x above is against **eight L = 1 NPU dispatches**, which is not the
alternative the engine faces — the alternative is the host, and the host does
not scale per row either. `q4nx_matmul` keeps its row loop *inside* the weight
tile loop, so the 1.35 GB table streams once regardless of `L`. Measured at the
engine's shipping flags (`tools/bench_lm_head_rows.cpp`): 24.9 ms for 1 row,
**75.6 ms for 8**.

So the honest figure is 31 ms steady-state against 75.6 — about **45 ms per
speculation cycle**, against a measured cycle total of 3545–3877 ms on a real
64-token generation. That is ~1%, on a harness whose run-to-run spread on
unchanged code paths is 9%. Worth having, not worth rushing.

(The 49 ms headline includes first-dispatch warmup; back-to-back dispatches
settle at ~31 ms, which is the number the engine would actually see. Both the 31
and the 75.6 now include the rms_norm; before the norm moved on-core the device
side was 28.6 ms and the host owed itself an extra pass over x.)
