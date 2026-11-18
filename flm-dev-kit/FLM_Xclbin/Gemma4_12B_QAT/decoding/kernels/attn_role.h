// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

///@file attn_role.h
///@brief Selects which attention flavor a translation unit is compiled for.
///
/// A model may need two different attention implementations at once: the
/// sliding-window layers and the global (full-causal) layers can have a
/// different head dim, a different number of KV heads, and therefore a
/// different GQA decomposition. Gemma4-12B is the first such model:
///   global -> ATTN_IMPL_1x8x1, DH = 512, 1 KV head
///   SWA    -> ATTN_IMPL_4x2x1, DH = 256, 8 KV heads
///
/// Both flavors live in the same sources (attn_qk.cc / attn_kv.cc). Each one is
/// compiled twice, once per role, so a role only ever sees one set of geometry
/// constants and the two cores get their own object:
///   attn_qk.cc -> attn_qk.o      (ATTN_ROLE_GLOBAL, default)
///   attn_qk.cc -> swa_attn_qk.o  (-DATTN_ROLE=ATTN_ROLE_SWA)
///
/// The role also picks the entry point name, since IRON binds `attn_qk` on the
/// global QK core and `swa_attn_qk` on the SWA QK core.
#ifndef __ATTN_ROLE_H__
#define __ATTN_ROLE_H__

#include "../models/all_models.h"

#define ATTN_ROLE_GLOBAL 0
#define ATTN_ROLE_SWA    1

#ifndef ATTN_ROLE
#define ATTN_ROLE ATTN_ROLE_GLOBAL
#endif

#if ATTN_ROLE == ATTN_ROLE_SWA

#define ROLE_IMPL SWA_ATTN_IMPL
#define QK_ENTRY  swa_attn_qk
#define KV_ENTRY  swa_attn_kv

constexpr int HEAD_DIM      = SWA_DH;
constexpr int ROLE_KV_HEADS = SWA_NUM_KV_HEADS;
/// @brief One pass covers every Q head of a group.
constexpr int Q_PASSES_PER_CU = 1;

#else

#define ROLE_IMPL ATTN_IMPL
#define QK_ENTRY  attn_qk
#define KV_ENTRY  attn_kv

constexpr int HEAD_DIM      = DH;
constexpr int ROLE_KV_HEADS = NUM_KV_HEADS;
/// @brief A group is split over two Q passes that share the same K/V tile.
constexpr int Q_PASSES_PER_CU = 2;

#endif

#endif // __ATTN_ROLE_H__
