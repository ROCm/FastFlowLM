// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SMOLVLA__EXPERT__MODE__HPP__
#define __SMOLVLA__EXPERT__MODE__HPP__
#pragma once
#include <cstdlib>
#include <cstring>

namespace smolvla_expert_mode {

/// The expert matmuls read their weights as bfp16ebs8 (9 B per 8 values, expert.xclbin). Kill switch:
/// SMOLVLA_EXPERT_BF16=1 loads the bf16 weights and runs expert_bf16.xclbin, the previous design.
inline bool bf16_weights() {
    static const bool v = [] {
        const char* e = std::getenv("SMOLVLA_EXPERT_BF16");
        return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
    }();
    return v;
}

/// RoPE runs inside the attention kernel (expert.xclbin), which also lets qkv -> mha -> o_proj be one runlist.
/// The bf16 kill switch (expert_bf16.xclbin) keeps the previous design: RoPE on the CPU.
inline bool rope_in_kernel() { return !bf16_weights(); }

} // namespace smolvla_expert_mode

#endif
