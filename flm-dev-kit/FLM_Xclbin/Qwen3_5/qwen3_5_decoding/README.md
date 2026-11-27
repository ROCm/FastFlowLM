<!--
  Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
-->

# FLM Decoding Layer generator

Only for generateing decoding layers on AMD Ryzen AI Strix+ chips.


## Constraints

- Must be traditional layer, sliding window is allowed

## Models

| Model Series | Head Dim | Num Heads | Num KV Heads | Num Heads per group | Num heads per CU | KV Bias | QK Norm | intrinsic mm | Pad Q | Note                       |
|--------------|----------|-----------|--------------|---------------------|------------------|---------|---------|--------------|-------|----------------------------|
| Llama        | 64, 128  | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | No      | 8x8x8        | No    | single 8x8 q, different k  |
| Llama-3B     | 256      | 24        | 8            | 8                   | 2 x (3 -> 1)     | No      | No      | 8x8x8        | Yes   | row 3, 7 are removed       |
| Qwen-small   | 128      | 16        | 8            | 2                   | 2 x (2 -> 1)     | No      | Yes     | 8x8x8        | Yes   | row 2,3,6,7 are removed    |
| Qwen-large   | 128      | 32        | 8            | 8                   | 2 x (4 -> 1)     | No      | Yes     | 8x8x8        | No    | single 8x8 q, different k  |
| *Gemma-small | 256      | 4         | 1            | 4                   | 4 -> 1           | No      | Yes     | 4x8x8        | No    | Only one CU is used        |
| Gemma-large  | 256      | 8         | 4            | 2                   | 2 -> 1           | No      | Yes     | 4x8x8        | Yes   | zero pad to 4x8 in kernel  |


## Attention implements

For kernels

| Type | Kernel spec  | intrinsic mm |
|------|--------------|--------------| 
| 1    | 2 x (4 -> 1) | 8x8x8        |
| 2    | 4 -> 1       | 4x8x8        |

Dataflow of Q:

- Always from (NH, DH) -> 4 x (NH // 4, DH)
- NH always pads to multiple of 4
- DIM: ((DH // 8, 8), (NH // 4, DH),(8, 1))

Inside Kernel:

- Requires process padding and remove useless data according to parameters

## Gated DeltaNet decode kernel

`kernels/gate_delta_net_decode_128_128_128_32.cc` writes the new state into a single `S_out` buffer that the core DMA
sends out together with `o`. It must acquire `CT_S_O_out_prod_lock` before it writes the state, not after: until the DMA
releases that lock it may still be sending the previous head's state from the same buffer. With the lock taken late,
host sequences that move the next head's inputs earlier corrupt the state now and then (non-deterministic decode).
