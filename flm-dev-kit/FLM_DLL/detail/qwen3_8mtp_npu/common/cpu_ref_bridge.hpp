// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/cpu_ref_bridge.hpp
/// \brief Glue between the host's LM_Config world and the self-contained fp32
///        reference, which knows nothing about it.
///
/// The bf16 <-> fp32 marshalling helpers that used to live here are gone: the
/// conversions on the live path are `q38simd`'s vectorised ones, and nothing
/// compared a `buffer<bf16>` against an fp32 reference through this header.
#pragma once

#include "common/cpu_ref_delta_net_layer.hpp"

#include "typedef.hpp"
#include "buffer.hpp"
#include "lm_config.hpp"

#include <string>
#include <vector>

namespace cpu_ref {

/// Delta-net geometry as the loaded config describes it.
/// \note LM_Config caches no members -- every parameter is read out of
///       _json_config through get<T>(). The defaults below are the
///       Qwen3.8-27B values, so a config missing a key still yields a
///       geometry that validate() accepts rather than a zero-sized one.
inline delta_net_geometry geometry_from(const LM_Config& c) {
    delta_net_geometry g;
    g.model_dim         = static_cast<int>(c.get("hidden_size",             u32(q38::MODEL_DIM)));
    g.intermediate_size = static_cast<int>(c.get("intermediate_size",       u32(q38::INTERMEDIATE_SIZE)));
    g.qk_heads          = static_cast<int>(c.get("linear_num_key_heads",    u32(q38::L_QK_HEADS)));
    g.v_heads           = static_cast<int>(c.get("linear_num_value_heads",  u32(q38::L_V_HEADS)));
    g.k_head_dim        = static_cast<int>(c.get("linear_key_head_dim",     u32(q38::L_K_HEAD_DIM)));
    g.v_head_dim        = static_cast<int>(c.get("linear_value_head_dim",   u32(q38::L_V_HEAD_DIM)));
    g.conv_k            = static_cast<int>(c.get("linear_conv_kernel_dim",  u32(q38::CONV_K)));
    g.rms_eps           = c.get<f32>("rms_norm_eps", q38::RMS_EPS);
    g.l2_eps            = q38::L2N_EPS;   // not in config.json; fixed in modeling_qwen3_5.py
    return g;
}

/// Path of the quantised checkpoint a Q4NX would open for the same model.
inline std::string q4nx_path_of(const std::string& model_path) {
    return model_path + "/model.q4nx";
}

} // namespace cpu_ref
