// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision_npu.hpp
/// \brief the matrix products and the attention of the Inkling-2 image tower on the NPU
///
/// Runs on the two vision xclbins of the Qwen3-VL-4B kernel set (same widths: 1024 hidden, 16 x 64 heads, 4096 MLP):
///   MM        (m128 k512 n64, bias fused, M a multiple of 512, K of 512, N of 64), bf16 in / out
///   attention (64 wide heads, L and S padded to 512, the real S masked)
/// The matrix products are bfp16 MMUL: the kernel floors both operands onto a shared-exponent grid per 8 values, so
/// the weights (once, at bind) and the activations (on every call) are first rounded onto that grid, which removes
/// the bias of the floor.
#pragma once

#include <memory>
#include <string>

#include "inkling2_vision.hpp"
#include "npu_utils/npu_utils.hpp"

namespace inkling2_vision {

struct npu_options {
    bool preround = true;       ///< round weights and activations onto the bfp16 grid before the kernel
    bool fused_gelu = false;    ///< fc1 and proj_in apply the kernel's tanh gelu (the tower is erf; default: erf on the host)
    bool profile = false;       ///< print the time per stage after every image
};

/// \brief the accel on the two xclbins (paths to vision_mm.xclbin and vision_attn.xclbin); npu must outlive it
std::unique_ptr<accel> make_npu_accel(npu_xclbin_manager* npu, const std::string& mm_xclbin, const std::string& attn_xclbin,
                                      const npu_options& opt = {});

}  // namespace inkling2_vision
