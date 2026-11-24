<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
  Portions of this file consist of AI-generated content.
-->

# LFM3 decode layer (`lfm3_npu_bin`)

One overlay (`layer.xclbin`) that runs **one decoder layer of LFM3-1.7B for one token** (gated short-conv block, sliding-window
attention block or full-attention block, each followed by an FFN, all wrapped in sandwich norms) and, in a second mode, the
**lm_head**. It is a fork of `../lfm_npu_bin` (LFM2 / LFM2.5) so that the shipped LFM2 xclbins stay untouched; the floorplan,
the matvec engine and the DMA dataflow are the same, what changed is listed below.

```
source <env with Peano llvm-aie + mlir-aie + Vitis headers>
make bitstream instructions MODEL_TYPE=LFM3_1_7B      # build/LFM3_1_7B/xclbins/layer.xclbin
```

`MODEL_TYPE`: `LFM3_1_7B` (default), plus the two LFM2 types the sources still compile for (`LFM_1_2B`, `LFM_2_6B`, used as a
regression check of the shared code: built as `LFM_1_2B` the RTP addresses and the behaviour match `../lfm_npu_bin`).

## What differs from `lfm_npu_bin`

| | LFM2 | LFM3-1.7B |
|---|---|---|
| norms per layer | 2 | 4 (operator, post_operator, ffn, post_ffn). The rms tile holds all four in one `4 x D` buffer filled by one DMA per layer; the kernel indexes `w[k * D]` and frees the buffer after the last use. A one-weight buffer refilled between the norms **deadlocks** (weights 3 and 4 can only land after weight 2 is consumed, the packet waiting for the tile blocks the stream) |
| conv width | 3 (history ring of 2 rows) | 7 (`conv.cc` takes `CONV_K` taps, ring of 6 rows, `conv taps` buffer `K x D`) |
| attention | 2 KV heads of 64 per core, 8 real Q rows | **1 KV head of 128 per core**, 4 real Q rows padded to 8 (same bytes per core tile, so the memtile/shim DMA layouts are unchanged, only parametrized: `KC`, `KR`, `O_ROWS`) |
| attention gate | none | per query head `sigmoid(h . Wg)` multiplies the head's output before o_proj. The host computes the 16 gates (it holds the layer input) and writes them (bf16) into the rope buffer; the rope kernel puts them in row 4 of each KV head's Q tile, the QK core forwards them through the S buffer, the KV core applies them in its final scale (`attn_kv.cc: scale_div_gate_aie`). Nothing per layer is in the instruction stream |
| sliding window | none | host only: the host streams the last `min(L, 256)` tokens and sets the cores' length RTP |
| RoPE | one table | per layer kind (full: theta 1e6 with partial rotary, sliding: theta 1e4); the tables are host data (zero frequency = identity), `qk_norm` eps from the model (1e-5) |
| lm_head | 65536 vocab, Q4_0 | 128000 vocab padded to a multiple of D (129024), tied embedding as a true **Q4_1**: `proj.cc` has a `HAS_MIN` path (`+ min' * sum(x)` per 32-input group, nibbles stored signed, `min' = min + 8 * scale`) used by the lm_head phase only |

## Host contract

The host side is `FLM_DLL/detail/lfm3_npu` (`lfm3_npu_sequence.*`, `lfm3_decode_npu.cpp`). Things it relies on:

- **RTP addresses are absolute L1 addresses** and are hard-coded in `lfm3_npu_sequence.hpp`. Any change that moves buffers
  (types, stack sizes) moves them: rebuild, read `build/<type>/bitstream/from_iron/flowArch.mlir.prj/input_with_addresses.mlir`
  and update the constants. Current: QK length 5120, KV length 34816, proj 33280, rms 5120, rope 46080.
- Argument order: `x/y`, projection weights (one buffer per layer: `q|k|v` or `in_proj` | `o_proj`/`out_proj` | up|gate band | down),
  norm weights (`4 x D`), rope buffer (`K x D` elements: attention layers use `cos[DH/2] sin[DH/2] q_norm[DH] k_norm[DH] gates[16]`,
  conv layers the taps), KV cache / conv ring. `IS_ATTN` RTP: 0 conv, 1 attention, 2 lm_head (final norm padded to `4 x D`).
- KV cache layout is the LFM2 four-band layout (K heads 0..h/2-1, K second half, V first half, V second half, each `MAX_L` rows of
  `DK/2` elements).

## Verification (see `lfm3-1.7b` workspace log)

Per layer against the host engine on the same input: cos >= 0.9994 (conv, full, sliding; positions 16, 1018, 3208); new K/V rows
>= 0.99995; lm_head >= 0.9999; goldens greedy-identical; next-token NLL on natural text on par with the PyTorch reference.

`host/host.cpp` is the unchanged LFM2 test harness and does not know LFM3.
