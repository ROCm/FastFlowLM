// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/delta_net_sequence.hpp
/// \brief The gated-delta-net prefill sequence, declared only.
///
/// Unlike every other sequence this engine drives, this one is **not** ported
/// into the engine: the shared runtime already carries it at
/// `FLM_DLL/include/npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp`,
/// which is on the `-I../../include` path and is the same generator
/// `qwen3_5vl_npu` uses. Porting it would be a second copy of a file that is
/// already shared, and the two would drift.
///
/// It is, however, not declared `inline`, so including that header in more than
/// one translation unit is a duplicate-symbol link error. This header declares
/// the prototype; `delta_net_sequence.cpp` is the one TU that defines it.
///
/// ## What the arguments mean, since the header has no comments
///
///   NUM_OF_COLUMN, NUM_CT_PER_COLUMN  the array shape, 8 x 4
///   QK_HEAD_GROUP                     v_heads / qk_heads, the repeat_interleave
///   NUM_HEAD                          v_heads -- o and v heads, not q/k
///   MT_L_size                         tokens per memtile chunk, 128
///   L                                 real tokens this dispatch folds
///   L_PADDED                          ceil(L / MT_L_size) * MT_L_size
///   DQ, DK, DV                        *per head* widths, 128 / 128 / 128
///   L_start                           row this dispatch starts at, in the qkv,
///                                     alpha/beta and O buffers. This is what
///                                     makes a long window chunkable: the
///                                     generator derives the qkv and alpha/beta
///                                     offsets from it.
///   S_in_out_offset_fp32              fp32 element offset of S inside arg 0
///   beta_start_offset_fp32            fp32 element distance from alpha's block
///                                     to beta's. **Not re-based by L_start** --
///                                     alpha and beta ride one BD whose inner
///                                     axis of 2 strides by exactly this. Pass
///                                     the *whole* buffer's L_PAD * NUM_HEAD and
///                                     it is right for every chunk.
///
/// Arguments to the app itself are `(S_in_out, QKV_in, alpha_beta_in, O_out)`.
#ifndef __QWEN3_8MTP_NPU_DELTA_NET_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_DELTA_NET_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"

#include <cstdint>

void generate_gate_delta_net_prefill_sequence(
    npu_sequence& seq,
    uint32_t NUM_OF_COLUMN, uint32_t NUM_CT_PER_COLUMN,
    uint32_t QK_HEAD_GROUP,
    uint32_t NUM_HEAD,
    uint32_t MT_L_size,
    uint32_t L, uint32_t L_PADDED,
    uint32_t DQ, uint32_t DK, uint32_t DV,
    uint32_t L_start,
    uint32_t S_in_out_offset_fp32,
    uint32_t beta_start_offset_fp32);

#endif  // __QWEN3_8MTP_NPU_DELTA_NET_SEQUENCE_HPP__
