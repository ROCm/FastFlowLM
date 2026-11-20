<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# Hy-Memory-30B-A3B MoE decoding layer (NPU2)

One attention + MoE decoder layer for one token, as a single xclbin. Started as a
copy of `Qwen3_6/qwen3_6_moe_decoding` with the Qwen3.6-only parts removed
(gated DeltaNet layers, attention output gate, shared-expert gate).

Attention uses Hy-Memory sizes (32 q heads, 4 KV heads, head dim 128, full RoPE).
The MoE is Hy's: 128 experts, top 8, plus one shared expert, all 768 wide
(`expert_hidden_dim`), Q4_K.

## MoE

- **Router** (row 4 col 5, with the helper below it): fp32 logits from the bf16
  router weight, then `kernels/router_math.hpp`: fp32 sigmoid of all 128 logits,
  top 8 of score + bias (the bias only picks experts), weight = score / sum of the
  8 * 2.826. The fp32 bias (`moe_expert_bias`) is an RTP buffer the host writes
  each call (`RTP_router_expert_bias` in `rtp_addresses.txt`).
- **GLU slices**: one GLU slice is one proj round (16 cores x 32 rows = 512
  values). The host orders the up/gate weight rows so every round holds matching
  rows: shared expert `[up 256 | gate 256]`, routed expert (one per column)
  `[up 64 | gate 64]` per column. See `models/hy-memory-30b-a3b.h`.
- **Router weight**: the host stores it in the order it is read and sends it first
  in the call; MT5 holds all 512 KB and feeds the router (experts 0-63) and the
  helper (64-127) in parallel.
- **Routed experts**: 2 rounds of 4 experts, one expert per proj column. The host
  stores each expert as [channel 0 stream | channel 1 stream] (each its gate/up
  then down blocks, in read order), so the router points one linear shim BD per
  channel (4 / 5 round 0, 12 / 13 round 1) at the chosen expert with control
  packets. Both rounds are pushed at once, so round 1's weights follow round 0's
  with no gap.
- **Weight base address**: the router keeps one weight-buffer device address per
  layer (48), read from a shim BD on that layer's first call only
  (`kernels/router.cc`). The host writes the layer number each call
  (`RTP_router_layer`). `run.exe -L 1 --alt_layer 2` runs two layers, each with
  its own weight buffer, taking turns on one NPU context; this checks the table.
  shared-expert weight tasks, so the host sets the router's "BDs ready" lock only
  after it has pushed the last shared-expert task (otherwise a fast router can
  queue expert weights in front of them: wrong output, no hang).

## Tile map

|        | col 0 | col 1 | col 2        | col 3          | col 4          | col 5         | col 6 | col 7 |
|--------|-------|-------|--------------|----------------|----------------|---------------|-------|-------|
| row 5  | proj  | proj  | -            | attn kv (CU1)  | attn kv (CU3)  | GLU           | proj  | proj  |
| row 4  | proj  | proj  | -            | attn qk (CU1)  | attn qk (CU3)  | router        | proj  | proj  |
| row 3  | proj  | proj  | RoPE+qk-norm | attn kv (CU0)  | attn kv (CU2)  | router helper | proj  | proj  |
| row 2  | proj  | proj  | rms+residual | attn qk (CU0)  | attn qk (CU2)  | -             | proj  | proj  |
| MT     | gather| proj x/y | aux       | KV, O join     | KV, O half     | router w      | gather| proj w|
| IT     | w     | w     | x/norms/rope | KV (heads 0,1), O tap | KV (heads 2,3) | router w | w | w |

CU i holds KV head i and its 8 query heads.

## Q4_K math (proj cores)

`kernels/q4_core.h`, `_Q4K_x_int8` (`Q4K_INT8_MMUL 1`; must match `host/models.h`):
x is rounded to int8 per 32-value group with a power-of-two scale, once per x
slice; the group sums run on the native `mac_4x8_8x16` instruction against the
4-bit weights (unpacked by hardware); scales and mins stay fp32. The host
reorders every weight block at load time (`q4k_relayout_for_mmul`). About 810
cycles per block against 1780 for the old bf16 kernel (`_Q4K_x_bf16`, still there
with `Q4K_INT8_MMUL 0` on both sides). `../q4k_unit_test` (`./build.sh`) runs one
block on one core and checks every intermediate against the CPU.

Each proj core keeps a 4-block weight ring (two 2-block buffers,
`W_RING_BLOCKS` in `kernels/proj_main.cc`), so the weight stream can run further
ahead; the int8 x cache is four 4 KB arrays and the proj stack is 4 KB (every
proj function has a 0-byte frame) so that everything fits the 16 KB data banks.

## x into the proj engine

rms, the attention output (MT3) and GLU send packets into MT1's x input; MT1
broadcasts x to all 16 proj cores over one circuit (as in Gemma4 and Qwen3.5
decoding). In the routed-expert down step the 4 columns run 4 different experts:
GLU sends the 4 experts' h round robin and each core keeps its own column's
slices (`col_idx`, see `kernels/proj_main.cc`).

## Routing check

`check_routes.py` runs after every xclbin build (`makefiles/bitstream.mk`). It
fails the build if a packet flow is delivered somewhere it was not asked to go,
or if flows that wait on each other share a switch arbiter (a packet stalled
halfway holds the arbiter; this deadlocked earlier versions on hardware). Packets
into MT1's x input may share an arbiter only with each other, with a short list
of flows that cannot wait on them, or with flows from their own DMA channel.
`CHECK_ROUTES=0` skips it, for experiments only.

aiecc's router gives a different routing from run to run (its choices depend
on memory addresses, which Linux randomizes; with `setarch -R` and the same
directory and arguments it repeats). So the build reruns aiecc until the check
passes, up to `ROUTE_TRIES` (default 5) tries; a try takes about a minute.
`quick_route.py <layer.mlir>` routes like aiecc in about 2 seconds; run it a few
times to see how often a design change passes.

## Build and run

    make kernel      # AIE kernels (.cc -> .o)
    make link        # IRON python -> MLIR
    make bitstream   # MLIR -> build/<MODEL_TYPE>/xclbins/layer.xclbin (+ routing check)
    make host        # run.exe
    make test        # run the layer test on the NPU

Options: `ATTN_TAP=1` also sends the attention output to the host
(`run.exe --attn_tap`); `DEBUG_TAP=x` (debug only) copies rms's x to IT4.

Host test: `run.exe -L <layer> -l <tokens> [--attn_tap] [--ctx N] [--xclbin path] [--bench N] [--per_step] [--alt_layer N] [--kv_fill N --dump <prefix>]`.

KV cache (per layer, in DDR): four blocks of `MAX_L × 256` bf16 (a column's 2 KV
heads × DH 128 per token), `[K col 0][K col 1][V col 0][V col 1]` (all K, then all V at `kv_cache_size / 2`, as FLM_DLL's hunyuan_npu)
(`decoding_layer::kv_block(col, kv)`), the same as `../dense_decoding`. `MAX_L`
(`--ctx`, default 131072 = 128K) is a multiple of 16; 128K is 256 MB per layer,
one buffer per layer (DMA offsets are 32-bit byte offsets). Zero the cache at
allocation: the 16-token reads go past `L` (masked, but must be finite). At
`L == MAX_L` the cache is full and the host refuses another call. NPU time per
call: 0.76 / 2.55 / 4.35 / 7.94 ms at context 8 / 32K / 64K / 128K (2026-10-26).
`--kv_fill N --dump <prefix>` is the long-context accuracy check: it fills KV
rows 0..N-1 with the Q4_K reference's K/V rows (repeated), decodes at positions
N.., and writes the outputs and new K/V rows for
an offline check, which recomputes the layer from the same
cache. Layer 1 update cos 0.9988 at fill 0, 0.9994 at 131064: no drift.

It ends with one RESULT block: calls completed, time per token, and each check's
mean / min cos (with the worst token) against the Q4_K and bf16 references. It
exits 1 if any call did not complete. `--per_step` adds every token's numbers;
`make host VERBOSE=1` (or 2) brings back the load / buffer / call-state logs.
`HY_TIME_HOST=1 ./run.exe ...` prints each call's host time by step. Per call
the test copies only the layer input and output (the weights once at load, the
KV cache never; ~2 ms per call instead of ~6 ms); the rest is rebuilding and
reloading the instruction sequence for the new context length.
The per-token latency it prints includes the host syncing every buffer each call;
`--bench N` times the NPU alone (N rounds of 480 back-to-back calls, context
length 8). Use it to measure speed changes.
References (`--ref_dir`, or `$FLM_REFERENCE_DIR`; not shipped): `hy_mem_30a3b_reference.safetensors`
(bf16) and `hy_mem_layer_q4k_layer_<N>.safetensors` (the whole layer recomputed with the Q4_K
weights). Debug env vars: `HY_STOP_AFTER=<stage>` ends the run
sequence early, `HY_DEBUG_TAP=x` reads the x tap.

## Peano

Peano is the default (`ENABLE_CHESSCC ?= 0`); `make ENABLE_CHESSCC=1` still builds
with chess. Run `make clean_all` (or use another `BUILD_DIR=`) when switching, since
both write the same build folder. Use Peano
`llvm-aie 22.0.0.2026090201+a36c62b9`.

Result (2026-11-04, Strix Halo test machine): the Peano layer is **bit-identical** to chess
(outputs and new K/V rows, 0 values differ, also with 32K of KV filled), and the
chess build from these same sources is bit-identical to the old chess build.
NPU time per call: 0.762 / 0.967 / 2.398 ms at context 8 / 4K / 32K
(chess 0.756 / 0.979 / 2.550). RTP addresses are unchanged.

What had to change, and why (each is a no-op under chess):

1. **`.bss` is not zeroed.** chess's loader zeroes `.bss`; Peano's does not. So any
   variable that must start at 0 and keep its value between calls goes to `.data`
   with `AIE_PERSIST_ZERO`: the ping/pong flags (proj, glu, rms, attention) and the
   router's `base_known` table (garbage there = the router trusts garbage expert
   addresses). The int8 A-matrix caches in `proj_main.cc` were the costly one:
   `q4k_prepare_x` writes only 1 of every 4 A rows and needs the others to be 0.
   Under Peano `proj_main` now zeroes the 16 KB once, on its first call. Without
   it the layer output was cos 0.03.
2. **`restrict` + locks.** `rms_residual`, `rope`, `router` and `router_helper`
   took their DMA buffers as `restrict` and also take/give locks. A lock call takes
   no pointer, so Peano decides it cannot touch a `restrict` buffer and drops or
   moves writes across it: it deleted 3 of the 5 packet-id writes in
   `rms_residual` (packets would go to the wrong core). These parameters now use
   `AIE_DMA_RESTRICT`: `restrict` under chess, nothing under Peano. Helpers with
   no lock inside keep their `restrict`.
3. **Lock loops.** Peano can move a loop's last lock release out of the loop. All
   12 loops that contain lock calls count with `AIE_LOCK_LOOP_CTR` (a volatile int
   under Peano), as `../dense_decoding` does.
4. **Scalar float.** Peano runs scalar `+ -` on the vector unit and fuses `a*b+c`,
   which rounds differently from chess's IEEE softfloat. The rms norm scale, rope's
   Q/K norm and the router's weight normalization use `AIE_FADD / AIE_FMUL`
   (compiler-rt softfloat under Peano) for their sums and means, so both compilers
   give the same bits; the 1/sqrt is `flm_rsqrt` (one hardware `invsqrt`). Expert choice itself uses only vector ops and compares.
5. **Attention** (`attn_qk.cc`, `attn_kv.cc`) is now a copy of
   `../dense_decoding`'s, which only adds Peano fixes (runtime-built constants
   instead of `static const`, a workaround for Peano miscompiling `extract<8>` at a
   runtime index, `AIE_*` loop macros).
6. **proj core memory** (`iron/proj_tile.py`). chess puts static data and large
   locals into any free gap between buffers; Peano's linker only uses the space
   after the highest buffer. The int8 caches (16 KB + 4 KB) did not fit there.
   Under Peano the six proj buffers are pinned by bank: bank 0 = stack, w_0, x_0,
   y_0; bank 1 = w_1, an unused pad, y_1, x_1 (x_1 ends at the bank's end). So
   banks 2 and 3 (32 KB) are free for the caches, and the caches no longer share
   a bank with w (chess had one cache in w_1's bank). Only the bank is pinned, not
   the address: this mlir-aie's bank-aware allocator counts a buffer pinned by
   `address=` twice (`checkAndAddBufferWithAddress` sets `mem_bank`, then
   `checkAndAddBufferWithMemBank` places it again) and then fails. Pinned buffers
   must also be declared in address order.
7. **Stacks.** chess gave every proj function a 0-byte frame (it allocates locals
   statically); Peano's frames are real: proj's deepest call chain is 4480 B and
   attn_kv's 2944 B. Under Peano the proj stack is 6 KB (bank 0 still fits:
   6144 + 9472 + 512 + 160 B) and attn_kv's 4 KB; the RTP addresses did not move.
   `makefiles/check_peano_stack.sh` fails the build if a stack can overflow.
8. `rope.cc`: the `chess_*` loop keywords became `AIE_*` macros (under Peano the
   raw keywords expand to nothing). A clang loop pragma rejects a count of 0 even
   in a dead `if constexpr` branch, so the remainder loop's count is at least 1.

Program memory is no longer tight under Peano: router 13,024 B (chess 16,282 B of
16 KB), proj 7,696 B (chess 13,072 B).

Debugging tools that found these: `--dump` (K/V rows are the proj output, V passes
rope unchanged), a `DEBUG_TAP=x` build with `CHECK_ROUTES=0` (rms output was
bit-identical, so the bug was after rms), and `clang++ -S -emit-llvm` on one kernel
(counting the stores between lock calls).
