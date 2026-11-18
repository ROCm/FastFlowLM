// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_bridge.hpp
/// \brief Glue between the host's buffer<bf16> / LM_Config world and the
///        self-contained fp32 CPU reference in cpu_ref_*.
///
/// cpu_ref_common.hpp deliberately depends on nothing but libstdc++ and mmap so
/// the reference can be built and debugged without the XRT tree. Everything
/// that needs to know about both sides lives here.
#pragma once

#include "cpu_ref_delta_net_layer.hpp"

#include "typedef.hpp"
#include "buffer.hpp"
#include "lm_config.hpp"

#include <string>
#include <vector>

namespace cpu_ref {

/// Delta-net geometry as the loaded config describes it.
inline delta_net_geometry geometry_from(const LM_Config& c) {
    delta_net_geometry g;
    g.model_dim         = static_cast<int>(c.hidden_size);
    g.intermediate_size = static_cast<int>(c.intermediate_size);
    g.qk_heads          = static_cast<int>(c.linear_num_key_heads);
    g.v_heads           = static_cast<int>(c.linear_num_value_heads);
    g.k_head_dim        = static_cast<int>(c.linear_key_head_dim);
    g.v_head_dim        = static_cast<int>(c.linear_value_head_dim);
    g.conv_k            = static_cast<int>(c.linear_conv_kernel_dim);
    g.rms_eps           = c.rms_norm_eps;
    g.l2_eps            = q38::L2N_EPS;   // not in config.json; fixed in modeling_qwen3_5.py
    return g;
}

/// Path of the quantised checkpoint a Q4NX would open for the same model.
inline std::string q4nx_path_of(const std::string& model_path) {
    return model_path + "/model.q4nx";
}

/// First `count` elements of a bf16 buffer as fp32 (0 = the whole buffer).
inline std::vector<float> to_f32(const buffer<bf16>& src, size_t count = 0) {
    const size_t n = count ? count : src.size();
    std::vector<float> out(n);
    const bf16* p = src.data();
    for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(p[i]);
    return out;
}

/// fp32 back to a bf16 buffer, so the existing get_error_metrics() overloads
/// and utils::print_matrix() apply to CPU results unchanged. Rounding to bf16
/// is also what the NPU output already went through, so the comparison is not
/// biased by the reference carrying extra precision.
inline buffer<bf16> to_bf16(const float* src, size_t n) {
    buffer<bf16> out(n);
    bf16* p = out.data();
    for (size_t i = 0; i < n; i++) p[i] = static_cast<bf16>(src[i]);
    return out;
}

inline buffer<bf16> to_bf16(const std::vector<float>& src) {
    return to_bf16(src.data(), src.size());
}

} // namespace cpu_ref
