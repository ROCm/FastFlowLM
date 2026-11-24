<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# Solved: the MTP design's *second* dispatch timed out

Written for: the next engineer picking this up cold.

**Status: fixed.** The design now sustains repeated dispatch, and
the draft head passed 28 of 28 against the CPU reference on hardware.
This note is kept because the two false trails on the way to the fix are each
worth more than the fix itself.

## The symptom

`MTP_layer.xclbin` completed its **first** dispatch in a process and returned
good output. **Every subsequent dispatch returned `ERT_CMD_STATE_TIMEOUT` (8),**
whatever the host driving it — this port, the standalone scaffold, and the
scaffold's independent `xrt::runlist` bench alike.

```
[mtp] dispatch ctx=0 pos=1 keys=1
  ok, out[0] = -3.5
[mtp] dispatch ctx=1 pos=2 keys=2
terminate called after throwing an instance of 'std::runtime_error'
  what():  npu_mtp_layer: dispatch state 8 at cache row 1, position 2
           -- a timeout is a sequence bug or a stale RTP address
```

## The cause: an unrouted MM2S chain still fires

`iron/gate.py` had `enable_tb = True`, which builds a second MM2S chain on the
gate tile — a DDR tap that sends the gated attention row out once per pass for
scoring. With the tap on, the credit accounting runs like this:

| block | what it does | per pass |
| --- | --- | --- |
| core | `gate()` releases `gated_attn_out_cons_lock` | +10 |
| `block[5]` | replays the row into o_proj's `x` FIFO | takes 10, releases **`tb_lock`** ×10 |
| `block[7]` | the DDR tap, on MM2S ch1 | takes 10 `tb_lock`, releases `gated_attn_out_prod_lock` ×10 |

`gated_attn_out_prod_lock` starts at `repeats` = 10, so **pass 1 runs entirely
on the lock's initial credit** and looks perfect. The only thing that ever puts
that credit back is `block[7]`.

And `MTP_layer_aie.py` never calls `connect_port` on `gate_core.TB_Out`. The
chain is still emitted and still reachable, so it still fires: `block[7]` takes
all 10 `tb_lock` credits, stalls forever on a channel that reaches nothing, and
never releases the producer lock. Pass 2 blocks on the first acquire and the
whole array parks behind it.

**An unrouted MM2S chain is not a no-op.** Block reachability in the MLIR
decides whether the locks move, not whether anything called `connect_port`.

The fix is `enable_tb = False` plus a guard so blocks 6 and 7 are never emitted,
and `block[5]` releasing `gated_attn_out_prod_lock` directly. To bring the tap
back, wire `TB_Out` to a shim in `MTP_layer_aie.py` **and** post a matching
receive in both `decoding_layer::_gen_sequence` and this port's
`gen_mtp_layer_seq()`. Routing it without receiving it hangs the same way.

`gated_attention_verify/iron/gate.py` has no tb chain at all. That — not
anything about the fusion path or the k|v append — is why that design
re-dispatched and this one did not.

## False trail 1: an unchecked `ert_cmd_state` looks like a pass

An earlier revision of this note said the standalone scaffold "dispatches
repeatedly without trouble", and concluded from that contrast that the bug had
to be in this port's host code. Both halves were false.

`decoding_layer::operator()` **discarded the `ert_cmd_state`**. On a timeout the
output BO is never written, so `sync_from_device()` hands back the *previous*
pass's row — a plausible number, in the right range, that the host then reports
as a success. Every scaffold-side experiment that "proved" repeated dispatch
worked was reading dispatch 1's output over and over.

The tell was in the wall clock all along: the scaffold's baseline run was 1.61 s,
`FLM_MTP_REPEAT=1` was 62.98 s and `FLM_MTP_REPEAT=4` was 126.97 s. Each extra
"dispatch" was an XRT timeout being waited out, not a pass being run.

Both scaffolds now check the state and `exit(1)` on anything but
`ERT_CMD_STATE_COMPLETED`.

**Lesson:** an unchecked `ert_cmd_state` makes a timeout indistinguishable from
a pass, because a timeout leaves the output buffer holding the last good answer.
Check it on every dispatch, in probes especially — and check the wall clock,
which is the one signal a stale buffer cannot fake.

## False trail 2: a correct change credited with the wrong fix

Dropping the packet id on `IT[3].MM2S[0]` — a point-to-point link with nothing
to demux — is correct and is still in the IRON design. But it was signed off against
the unchecked return above, and it did **not** fix the second-dispatch timeout.
A correct change and a fix for the symptom in front of you are different
claims; the second one needs its own evidence.

## How it was localised

Three probes, in order, each narrowing the last:

1. `FLM_MTP_BENCH=1` drives the scaffold's `xrt::runlist` path — no sequence
   regeneration, no `_setup_kernel()`, no host-side state between passes. It
   timed out too, which moved the bug from the host to the array.
2. `FLM_MTP_STALL=1` zeroes the k row, the v row and `layer_out` before
   dispatch, then counts non-zero words after a failed one. The hung pass had
   written **k 1024/1024, v 1024/1024, `layer_out` 0/5120** — it completes the
   whole k|v append and never produces output.
3. `FLM_MTP_TRUNC=n` / `FLM_MTP_TRUNC2=n` cut the sequence off after phase `n`.
   Phases 1-4 completed; phases 5, 6 and 7 timed out. Phase 5 is o_proj's
   weight round — **the first round whose `x` comes from attention**, which is
   to say the first consumer of the gate tile's replay.

That pointed straight at `gate.py`, where the lock ledger balances on paper and
still deadlocks, because one of its two chains can never complete.

Note how far the stall landed from the cause: the hang surfaced on the o weight
round because o's `x` never arrived, not anywhere near the gate tile itself.

## Design context worth carrying in

Two notes from the author of the design, recorded verbatim because they are not
derivable from the code:

- *"The KV cache offset is related to the MAX_L, and the real `l` sent to the
  kernel uses the RTP."* So the cache addressing in the descriptors is built
  from the **allocation geometry**, not the live cursor; the number of keys the
  kernels actually walk arrives separately, as the RTP word.
- *"The code under `FLM_Xclbin/Qwen3_8/MTP_decoding` is used for debugging, so in
  some cases it didn't do the consideration."* Treat the scaffold as a probe,
  not as the specification. It is still the right hardware control — but only
  once its return value is checked.

## Where the port stands

Matched to the scaffold and proven so. Live sequence dumps taken from both sides
at the same cursor are byte-identical — 41 896 words, `diff` empty at both
L=1121 and L=1122. All five BO sizes agree element for element (`x` 15 360
bf16, `proj` 245 514 240 B, `rms` 20 480 bf16, `rope` 576 bf16, `cache`
2 326 528 bf16 at `MAX_L` 1136). The RTP addresses survived both rebuilds
(`qk` 9216, `kv` 36864). The two `npu_utils` trees are functionally identical.

The port's KV layout is the scaffold's **two planes**, not the interleaved
layout an earlier revision of this file defended. That change is what makes the
dumps byte-identical, and it is also what lifts the length cap: see
`planar-kv-cache-masks-stride-past-2048`.

After the fix, `FLM_MTP_REPEAT=5 FLM_MTP_ROT=1 FLM_MTP_BASE=1121` runs all six
dispatches in **1.70 s** total with the accuracy unchanged (k 0.999958,
v 0.999979, h2 0.999675), against 63.9 s for a single repeat before it.

## Standing instrumentation

Kept rather than stripped: these are what localised the bug, they are all off by
default, and each returns before anything graded.

**Engine-side, nothing is left.** `npu_mtp_layer::forward()` carried an
`FLM_MTP_TRACE=1` per-dispatch trace and `gen_mtp_seq` an `FLM_MTP_SEQDUMP`
sequence dump; both were removed once the design was repaired and the port
matched word for word. Neither belonged in a shipping `.so` — the trace wrote
to `stderr` from the draft loop's inner step, and the dump opened a file per
sequence build from a path taken out of the environment. What remains is the
descriptive throw on a non-`COMPLETED` dispatch state, which is the line that
actually names the failure:

```
npu_mtp_layer: dispatch state 8 at cache row 1, position 2
  -- a timeout is a sequence bug or a stale RTP address
```

The scaffold below keeps its copies, and that is where to re-open this: it is a
bring-up harness, not a shipping path, and the trace above is two lines to put
back if a dispatch ever times out again.

Filter the aiebu banner out of any scaffold run with
`grep -v "^Header\|TransactionSize\|UID:"`; each banner is one
`_setup_kernel()`, which counts ELF re-assemblies.

Scaffold-side, in `MTP_decoding/host`:

| probe | what it does |
| --- | --- |
| `FLM_MTP_REPEAT=n` | `n` extra dispatches, with `FLM_MTP_BASE=b` and `FLM_MTP_ROT=1` |
| `FLM_MTP_BENCH=1` | drives the `xrt::runlist` path instead — no host state between passes |
| `FLM_MTP_STALL=1` | zeroes k, v and `layer_out` first; on a failed pass, counts what got written |
| `FLM_MTP_TRUNC=n` | ends the sequence after phase `n` (1 fc, 2 qkv, 3 z, 4 append+replay, 5 o, 6 up\|gate, 7 down) |
| `FLM_MTP_TRUNC2=n` | the same, but only for the repeat dispatches, so the scored first pass stays whole |
| `ATTN_REPEAT=n` | the same idea in `gated_attention_verify/host/host.cpp` |

Port-side, the (since removed) N3 driver had `FLM_N3_PROBE=n` with `FLM_N3_BASE`,
`FLM_N3_HOLD`, `FLM_N3_SAME`, `FLM_N3_NOISE`; and `FLM_N3_ROWS=1` for the
per-row error distribution. `FLM_N3_DUMPCTX=n` went with `FLM_MTP_SEQDUMP` —
it only ever emitted the port half of the diff above, so on its own it produced
nothing. To reproduce that diff, restore both halves together.

## Debug artefacts left on disk

- `FLM_Xclbin/Qwen3_8/MTP_decoding/build/QWEN3_8_27B/xclbins/layer.xclbin` is a
  symlink to `MTP_layer.xclbin`. The build rule copies `from_iron/layer.xclbin`
  under the `MTP_layer.xclbin` name, but `get_layer_xclbin_path()` in `models.h`
  still asks for `layer.xclbin`, so the scaffold cannot find its own bitstream
  without it. Either keep the symlink or fix `models.h`; do not just delete it.

## Scope

This blocked the module-2 NPU backend, and with it the end-to-end
`verify_qwen3_8mtp` driver (M13 / M19), since `load_weights()` selects
`speculate_npu` whenever a device is present. Both are unblocked.
