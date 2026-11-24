<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# fused8 verify xclbin: dispatch-count hang, root cause and fix (stream H, 2026-10-29)

## Symptom
A hardware context running the fused 8-row verify layer stopped answering after a fixed number of dispatches
(engine: verify run #300-305; standalone harness: `--repeat 400` aborts). After the hang the driver refuses new
hardware contexts for ~60 s, then recovers by itself. The engine worked around it by rebuilding the verify context
every 200 dispatches.

## Root cause: one task-completion token per dispatch was never consumed
The host instruction stream of one fused8 dispatch submits shim BDs with `issue_token = true` (every S2MM BD gets a
token automatically, MM2S BDs when requested) and then `npu_dma_wait`s for them. A wait consumes ONE token of its
(shim tile, direction, channel). Shim tile IT3, MM2S carries THREE token-issuing BDs but the stream waited only twice:

| IT3 MM2S channel | BDs with a token | waits (old stream) |
|---|---|---|
| ch0 | `bd_0` 8 input rows; global layers also `bd_2` (global RoPE table) | 1 |
| ch1 | `bd_1` 4 norm vectors; SWA layers also `bd_3` (SWA RoPE table) | 1 |

So every dispatch left exactly one token in the shim's task-completion queue (global layers on ch0, SWA layers on
ch1). Tokens are not cleared at the end of a dispatch; once ~255 are queued the queue is full and the array stops
(the shim can no longer post completions, the command never completes). The decode xclbin's stream waits once after
every token-issuing BD, which is why decode runs thousands of dispatches per context.

Evidence
- `hammer8` (new test: consecutive dispatches in one hardware context), layer 0, cursor 64, old stream: fails at
  dispatch **#254** (0-based) = the 255th unconsumed token, deterministically.
- Same build, balanced stream (one wait per issued token): 700 dispatches of layer 0 / cursor 64, then the acceptance
  run below, no failure.
- Count-based, independent of layer type, ctx, wall time, BO chaining and kernel module reuse (all observed in
  phase8_results.md) -- exactly what a per-dispatch leaked token predicts. The engine saw it at verify run ~300
  because its 42-layer cycles mix in runs of other contexts and the count is per hardware context.
- lm_head_8 (`gen_seq_l8`) was checked by inspection: its only token BDs are the per-round S2MM y BDs, each waited.

## Fix
- `FLM_Xclbin/Gemma4/verify_decoding/host/test_fused8.hpp` (`f8::run`) and the engine generator
  `FLM_DLL/detail/gemma4e_dflash/gemma4e_dflash_verify.cpp` (`verify_gen_seq`) keep a **token
  ledger**: every token-issuing memcpy increments a counter per (tile, direction, channel), every wait decrements it;
  the final waits are emitted from the ledger (one per outstanding token) and the generator throws if any counter is
  nonzero when the stream is closed. So an unbalanced stream cannot be generated any more.
- The engine's context-rebuild workaround stays as an optional switch, default OFF: `FLM_DFLASH_VERIFY_RELOAD=N`.
- Repro switch: `FUSED8_OLD_WAITS=1` regenerates the old (unbalanced) stream in the standalone test.

## Rule for every new xclbin stream (the P/A streams' designs too)
One `npu_dma_wait` per token-issuing shim BD on the same (tile, direction, channel), inside the same dispatch.
Two BDs queued on one channel need two waits. Use the ledger pattern from test_fused8.hpp.

## Separate finding (not a hang): first dispatch after the xclbin load differs at rounding level
SWA non-skip layers with more than 32 KV chunks (cursor >= 505): the FIRST dispatch after loading the xclbin differs
from every later one (60% of output elements, cosine 0.99999, max abs diff 0.125 = 1 bf16 ulp of large values);
dispatch 2..N are bit-identical to each other whatever ran in between. Cursor 300 (20 chunks) and global layers: no
difference. So some attention-path state is initialised at program load and differs after the first run (a value
that should be reset per dispatch, e.g. a running max / correction or a ping-pong parity in the SWA attention
kernels or the MT2 staging chain). Owner: the attention stream (A), which rewrites these kernels. Harmless for the
engine (rounding level), but it breaks strict reproducibility of the first verify cycle.

## Acceptance results (2026-10-29)
| check | result |
|---|---|
| A/B, same binary (`hammer8`, layer 5, cursor 64): old stream (`FUSED8_OLD_WAITS=1`) | **fails at dispatch #254** (earlier also #254 on layer 0 / cursor 64 with the pre-fix build): deterministic |
| fused8, balanced stream: `hammer8` 21,000 consecutive dispatches in ONE hardware context, round robin over layers 0/5/40/35 x cursors 0/505/1000/4000 (16 pairs), KV state restored before every dispatch | **21000/21000 complete, 0 diffs** vs each pair's reference (2nd dispatch); 3 first-after-load diffs (SWA L0 at 505/1000/4000, see above); 252 s |
| lm_head_8: 5000 dispatches alternating two inputs (`lm_head_8/run.exe --repeat 5000`) | 5000/5000 bit-exact vs the first results, PASS |
| engine (`test_spec_engine`, clean build with the fix in a separate worktree), `FLM_DFLASH_VERIFY_RELOAD=0`, 15 prompts (12 chat + 3 long, up to 1024 tokens each) in ONE process | **7783 tokens generated speculatively, 1982 cycles = 83,244 verify layer dispatches** (+ ~4000 lm_head_8 dispatches) in one verify context, no timeout; mean AL 3.26, 0 stream mismatches |
| whole layer vs the oracle (8 sequential decode dispatches), layers 0/5/40/35 x cursors 0,1,9,16,505,511,512,520,1000,4000, repeat 3, MAX_L 4608 | all 40 cases within tolerance (min cos 0.99916, typ 0.9995-0.9998; K/V min cos 0.99934); the only "FAIL" flags are the repeat-identity check of SWA L0 at cursor >= 505 = the first-after-load anomaly |

Two earlier attempts of the engine run died with `ERT_CMD_STATE_TIMEOUT` / context-creation errors that were NOT this design:
`xrt-smi` logging (work/scripts/npu_watch.sh -> work/npu_watch.log) shows stream A's `attn8.exe --stage attnx` submitting one command
that never completed (10:04:34 and 10:10:21); every other context on the device (my engine, P's toy) stopped at that instant and was
killed by the driver reset 60 s later. A hung design on the NPU stalls every hardware context. A third failure was the driver's
16-context limit (another fork's engine held 12 contexts, mine needs 12).

## Other findings
- **Harness poison bug class (pre-existing, also before the fix):** `verify8 --poison` with several cursors in one process fails
  (e.g. layer 5: cursor 0 then 9 -> cos 0.83; cursor 9 alone with poison passes, cursor 0 then 9 without poison passes and is
  bit-identical to cursor 9 alone). So a dispatch can be influenced by ON-CHIP data left by the previous dispatch when that data is
  extreme (the poisoned V rows of the previous dispatch); with sane data the effect is invisible. Same family as the first-after-load
  anomaly: some attention-path buffer is read before it is rewritten in the current dispatch. Owner: stream A (attention rewrite);
  repro: `run.exe -t verify8 --mode verify --fused-xclbin ... --layers 5 --cursors 0,9 --repeat 1 --poison`.
- **My own harness bug, fixed:** the pli memo was keyed by object address; the harness re-creates decoding_layer objects at the same
  address, so layer 40 got layer 5's pli rows (cos 0.89). Keyed by layer index now.
- **Partial-stage debug xclbins (`fused8_attn_L8` etc.) hang by design** on repeated dispatch (partial phase sets desync the cores):
  never run them in a loop.
