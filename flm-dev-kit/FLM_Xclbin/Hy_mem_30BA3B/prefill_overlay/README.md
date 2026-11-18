<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# prefill_overlay — dequant_mm and attention in one xclbin

Hy-Memory prefill switches between dequant_mm and attention twice per layer.
Between two separate xclbins that is a hardware context switch, **~2.1 ms** each.
This overlay keeps both ops in one xclbin; switching between them costs **~0.12 ms**.

| (Strix Halo, NPU2) | standalone xclbins | overlay |
|---|---|---|
| dequant_mm, L 512, 2048 -> 1024 | 362 us (512 x 512 x 512) | 347 us |
| attention, L 512, 32 q / 4 kv heads | 1469 us | 1415 us |
| switch dq <-> attention | 2141 us (context switch) | 118 us |
| first switch after another xclbin ran | — | ~360 us |

dequant_mm here works in rounds of 256 tokens and steps through K one Q4_K
block (256) at a time, so it takes every Hy-Memory projection: L % 256,
Din % 256, Dout % 512 (768 -> 2048, 6912 -> 2048, 2048 -> 1536, 2048 -> 5120
checked against the CPU). It adds over K in the same order as
`dequant_mm_512x512x512`, so on the shapes both take its output is
bit-identical to it, through every switch direction. Attention is checked
against the CPU (cos 0.9995 at L 512).

## How it works

An xclbin fixes the stream routes, DMA BDs and lock values at load time, and the
two ops need different ones (they feed the same core ports from different mem
tiles). So the xclbin only holds what both ops share, and the host writes the
rest into the array when the op changes.

* **base xclbin** (`iron/overlay_aie.py base`): per core an *arena* (one buffer
  both ops lay their buffers into, `iron/layout.py`), an RTP block and an op
  lock; per mem tile an arena; the core programs. No routes, no DMA chains.
* **op variants** (`overlay_aie.py dq | attn`): base + that op's original locks,
  chains and routes (`iron/dq_tiles.py`, `iron/attn_tiles.py`). Never run:
  `aiecc --aie-generate-txn` dumps their load-time configuration as register
  writes, `tools/extract_txn2.py` turns it into `host/op_configs.hpp`.
* **switch** (`host/overlay_host.hpp`, `gen_switch`): reset the old op's DMA
  channels (they loop forever on a lock), write the routing registers that
  differ, the shared BDs, the lock values, then start the new op's chains.
  Attention uses fixed BD ids away from dequant_mm's, so no BD is shared: all
  of them are written once by a full switch, and a switch rewrites none.
* **core program** (`kernel/fused_e.cc`, `fused_o.cc`): both kernels in one
  object, each `#include`d inside its own namespace (their helpers have the same
  names). `kernel/dispatch.h` waits on the op lock, reads the RTP block
  (op, number of calls, "just switched"), resets the op's ping/pong flags after a
  switch (the chains restart at ping), and runs the op's kernel calls.
* **rounding mode**: `aie::set_rounding` is per-core state that lasts across
  calls. Attention sets round-to-nearest-even (`conv_even`: with the default
  truncation its error grew with the KV length, cos 0.92 at 8K keys; now 0.999
  through 8K), so dequant_mm sets `floor` (the hardware default it was written
  for) at its start. A kernel added here must set its own mode too.

A run of an op = [switch if needed] + RTP blocks + op lock = 1 on the op's cores
+ the op's usual shim DMAs (`gen_dq_run`, `gen_attn_run`: copies of
`gen_dequant_mm_512` and `gen_seq_1_4` with only the RTP part changed).

## Rules for the caller

* Keep `configured` (the op the array is set up for) across runs.
* **After any other xclbin ran** (decode layers, lm_head), pass `configured = -1`:
  the NPU reloads the overlay and everything the host wrote is gone. A partial
  switch then hangs (`qds_device::wait() unexpected command state`).

## Build and test

```
export AIETOOLS_ROOT=/scratch/ryzen_ai-1.3.0/vitis_aie_essentials LM_LICENSE_FILE=/opt/Xilinx.lic
source /path/to/mlir-aie/ironenv/bin/activate
source /path/to/mlir-aie/utils/env_setup.sh /path/to/mlir-aie/install
export PEANO_INSTALL_DIR=/path/to/flm-toolchains/llvm-aie-22.0.0.2026090201/llvm-aie   # the CI Peano
make            # ~1 min: kernels, base xclbin, both txn dumps, op table, run.exe
./run.exe -i 20
```

`run.exe` also runs `../../dequant_mm_512x512x512` (it must be built) as the
exact dequant_mm reference. `./run.exe --shape L,Din,Dout` runs one dequant_mm
shape on a freshly loaded overlay against the CPU.

## Peano

Peano is the default (`ENABLE_CHESSCC ?= 0`); `make ENABLE_CHESSCC=1` builds the
same design with chess. Run `make clean_all` when switching: the two use different
dq layouts, so `host/op_configs.hpp` (and the engine's copy,
`FLM_DLL/detail/hy_mem_npu/hy_mem_overlay_configs.hpp`) differ too. Use the CI
Peano, llvm-aie `22.0.0.2026090201`.

Result (2026-11-04, Strix Halo test machine): every check gives the same cos as chess.
dq alone at the engine's shapes, median of 3 alternating runs (`./run.exe --shape
L,Din,Dout` prints the best of 10 runs):

| shape (L, Din -> Dout) | chess | Peano |
|---|---|---|
| 2048, 2048 -> 5120 (k\|v\|q) | 4755 us | 4781 us (+0.6%) |
| 2048, 4096 -> 2048 (o) | 3747 us | 3722 us (-0.7%) |
| 256, 2048 -> 1536 (expert up\|gate) | 582 us | 583 us |
| 256, 768 -> 2048 (expert down) | 518 us | 518 us |

Attention L = 512: ~1310 us (chess ~1415). The L = 512, 2048 -> 1024 dq test was equal
from the start; the bigger shapes were up to 7% slower until the last two items below.

Changes (each a no-op under chess):

* `kernel/attn/attn.cc`: the two function-local `static const` vectors became
  plain locals. Peano's loader does not zero `.bss`, so a static's "already
  built" guard can read as true and the vector is never built (same fix as
  `Hy_MT2_1.8B/prefill_overlay`).
* `kernel/dq/`: the ping/pong flags use `AIE_PING_FLAG` (in `.data`, and volatile:
  Peano does not treat a lock acquire as a barrier for a flag it can prove), and
  the bf16 mmul converts B with the accumulator constructor (`aie_mmul_bf16_888`),
  both from `dequant_mm_512x512x512` (commit 3e3c0daf).
* `iron/layout.py`: under Peano the odd dq core puts x first (`x0 0, x1 4096,
  y_acc 8192, w 40960`), so x and the dequantized w never share a memory bank.
  The mmul loop reads both in the same cycle, and Peano's schedule paired the
  clashing loads often enough to cost ~10% (331 -> 325 us).
* `kernel/dq/aie_kernel_utils.h` `zero_vectorized`: Peano left the 256-bit store
  loop unpipelined (one store per 3 cycles), and every 64 x 64 output tile zeroes
  its 16 KB `y_acc` first. Under Peano it uses 512-bit stores unrolled by 4.
* `kernel/dq/dequant_mm_{e,o}.cc` output loop: 64 lock handshakes per output tile,
  and each one read and wrote the volatile `is_y_ping` with wait cycles. Under
  Peano the flag stays in a register for the loop and is stored once at the end
  (it only picks the buffer to write, so no stale read can come from it).
  These two closed the gap: k|v|q was +7.2%, then +3.7%, then +0.6%.
  How it was found: builds that skip the dequant math, the mm math, or both
  (control + DMA only) were all ~5% slower under Peano, so the loss was in the
  code between the math, and the k|v|q shape (most output tiles) lost most.
  `-O3` (+10-17%) and `-mllvm -enable-misched=false` (+60%) were worse.
* `tools/check_stack.py` (from Hy-MT2's overlay) checks the deepest call chain
  of both kernels against the 4 KB stack (2176 B used).

## Not done yet

* Reads weights in the "pairs" layout (dense layer, attention, shared expert);
  MoE-layer blocks are in the decode's int8-mmul order (`relaid`) and need a
  decoder for it.
* Attention at very long KV: cos 0.989 but NMSE 1.8 at 128K keys (the online
  softmax rescale is kept in bf16); fine to ~8K.
* `FLM_DLL/detail/hy_mem_npu` runs only the attention on it so far (re-copy
  `host/op_configs.hpp` to its `hy_mem_overlay_configs.hpp` after a rebuild).
* Fewer writes per switch: lock values only where they change, a block-write
  instruction in `npu_sequence`.
