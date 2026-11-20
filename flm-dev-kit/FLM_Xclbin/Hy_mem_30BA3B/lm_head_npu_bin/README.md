<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# NPU_FlashAttention

## Peano (Hy-Memory LM head)

Peano is the default (`ENABLE_CHESSCC ?= 0` in `makefiles/`), as in `../../Qwen3_5/lm_head_npu_bin`;
`ENABLE_CHESSCC=1` builds with chess. The Peano parts are the parent's: the mean for the
RMS scale is computed with `AIE_FADD / AIE_FMUL` (compiler-rt softfloat) so it rounds like
chess, the scale itself is `flm_rsqrt` (one hardware `invsqrt` on both compilers), and the ping/pong flags live in `.data` (Peano's loader does not zero `.bss`).
Build with `make bitstream`: plain `make` builds only `run.exe` (its `all` target names
variables that do not exist, as in the parent).

Result (2026-11-04, CI Peano llvm-aie `22.0.0.2026090201`): logits vs the CPU Q8_0 reference
mean cos 0.999988 (same as chess), 5.15-5.17 ms per call host + NPU (chess 5.19-5.68).
