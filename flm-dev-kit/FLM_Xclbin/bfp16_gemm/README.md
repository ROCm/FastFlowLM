<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# bfp16_gemm

A 32-core GEMM for bf16 weights: `Y[M, N] = act(X[M, K] W[N, K]^T + b)`, bf16 out, with both operands multiplied as native
block float (`bfp16ebs8`, 8 values sharing one exponent, 9 bytes) and fp32 accumulation. The host packs the weights once at
load time (1.125 bytes per weight instead of 2) and packs X before each call. Bias and an optional GELU (tanh form) are
applied on the core.

* `iron/gemm_aie.py`, `iron/gemm_tile.py`: the dataflow (8 columns x 4 core rows; a round is 256 rows x 1024 columns).
* `kernel/gemm.h`: the core loop (`gemm_e.cc` / `gemm_o.cc` are the two cores of a vertical pair).
* `host/seq_gen.hpp`: the run-time sequence, the DDR layouts and the weight modes below. An engine includes it.
* `host/pack_bfp16.hpp`: the bf16 -> bfp16 block encoder (round to nearest even on the mantissas).
* `host/host.cpp`: the test (`run.exe`), random operands against an fp32 product of the packed operands.
* `check_memtile_bds.py`: build-time check that the design leaves the memtile descriptors, locks and start queues the host
  programs free.

## Constraints

* `K % 128 == 0`, `N % 1024 == 0` (pad), `M` any (padded to 256-row rounds, at most 31 rounds per call).
* `K > 3072` (or `--stream 1`): `K % 512 == 0` and `K / 128 x rounds <= 256` per call; an engine splits larger `M` over calls.
* `act` 0 = none, 1 = GELU. Peano only.

## Weight modes

The memtile weight path is not in the design: every call writes its descriptors, locks and start queues.

| mode | when | weights from DDR | starts computing |
|---|---|---|---|
| hybrid | `K <= 1536` | once per call | on the first 128-k chunk; rounds 2.. send the resident panel |
| panel | `1536 < K <= 3072` | once per call | after the whole panel (224 KB per column) landed |
| stream | `K > 3072`, or forced | once per round | on the first chunk (a ring of 4 or 8 chunks) |

## Speed

Ryzen AI 9 HX 370, median of 20 warm calls, padded rows, bit-exact against the reference in every case:

| M x K x N | time | TFLOPS |
|---|---|---|
| 256 x 1024 x 1024 | 157 us | 3.4 |
| 256 x 1024 x 4096 | 243 us | 8.8 |
| 1024 x 1024 x 4096 | 553-600 us | 14.3-15.5 |
| 4096 x 1024 x 4096 | 1750 us | 19.6 |
| 1024 x 3072 x 3072 | 1110 us | 17.4 |
| 2048 x 2048 x 6144 | 2659 us | 19.4 |
| 4096 x 2048 x 8192 | 6318 us | 21.8 |
| 884 x 4096 x 1024 (stream) | 528 us | 16.3 |
| 884 x 7680 x 1024 (stream) | 919 us | 17.5 |

The cores alone run at about 40 TFLOPS; the calls above are bound by DDR traffic (about 25 GB/s on the test machine). Per
flop a call moves about `1.125 / 2048` bytes of X (re-read for every 1024 columns), `1.125 / (2 M)` of W and `1 / K` of
bf16 Y. That model gives 21.7 TFLOPS at 4096 x 2048 x 8192, as measured; at `K = 1024` the Y writes alone cap a call at
about 25 TFLOPS. Small calls pay about 100 us of weight latency before the first multiply.

## Test

```
make kernel && make bitstream && make host PEANO_INSTALL_DIR=<pinned Peano>
./run.exe --M 1024 --K 1024 --N 4096 --iters 20 [--act 1] [--stream 1] [--hybrid 0]
```

A call that does not finish in 10 s writes `~/.npu_suspect` and holds the hardware context 90 s so the driver can recover.
