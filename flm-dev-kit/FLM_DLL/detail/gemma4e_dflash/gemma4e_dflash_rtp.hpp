// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file gemma4e_dflash_rtp.hpp
/// \brief RTP addresses of the shipped DFlash xclbins (verify.xclbin = Gemma4/verify_decoding fused8 with FUSED8_ATTN=x,
///        lm_head_8.xclbin = Gemma4/lm_head_8), both for GEMMA4_E4B.
/// \note CI ships only the .xclbin files, so the engine carries the addresses. The two builds compare their addresses
///       with FLM_Xclbin/Gemma4/verify_decoding/rtp_expected/fused8x_GEMMA4_E4B.txt and
///       FLM_Xclbin/Gemma4/lm_head_8/rtp_expected/GEMMA4_E4B.txt and fail when they move: change this file together with them.
///       A `<name>_rtp_addresses.txt` next to the xclbin (a local build) overrides these tables.
#pragma once
#include <cstdint>
#include <utility>

namespace gemma4e_dflash_rtp {

inline constexpr std::pair<const char*, uint32_t> verify_attnx[] = {
    {"RTP_PROJ_IS_SWA_BUFFER_2_0", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_2_0", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_3_0", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_3_0", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_4_0", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_4_0", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_5_0", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_5_0", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_2_1", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_2_1", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_3_1", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_3_1", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_4_1", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_4_1", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_5_1", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_5_1", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_2_6", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_2_6", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_3_6", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_3_6", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_4_6", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_4_6", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_5_6", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_5_6", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_2_7", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_2_7", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_3_7", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_3_7", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_4_7", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_4_7", 15360},
    {"RTP_PROJ_IS_SWA_BUFFER_5_7", 53248},
    {"RTP_PROJ_SKIP_KV_BUFFER_5_7", 15360},
    {"RTP_RoPE_SKIP_KV_4_3", 40960},
    {"RTP_SWA_RoPE_SKIP_KV_5_3", 20480},
    {"RTP_attnx_score_3_4", 59904},
    {"RTP_attnx_score_5_4", 59904},
    {"RTP_attnx_pv_2_2", 54528},
    {"RTP_attnx_pv_3_2", 54528},
    {"RTP_attnx_pv_4_2", 54528},
    {"RTP_attnx_pv_5_2", 54528},
};

inline constexpr std::pair<const char*, uint32_t> lm_head_8[] = {
    {"RTP_LM8_FINAL_SCALE_2_0", 61440},
    {"RTP_LM8_FINAL_SCALE_3_0", 61440},
    {"RTP_LM8_FINAL_SCALE_4_0", 61440},
    {"RTP_LM8_FINAL_SCALE_5_0", 61440},
    {"RTP_LM8_FINAL_SCALE_2_1", 61440},
    {"RTP_LM8_FINAL_SCALE_3_1", 61440},
    {"RTP_LM8_FINAL_SCALE_4_1", 61440},
    {"RTP_LM8_FINAL_SCALE_5_1", 61440},
    {"RTP_LM8_FINAL_SCALE_2_2", 61440},
    {"RTP_LM8_FINAL_SCALE_3_2", 61440},
    {"RTP_LM8_FINAL_SCALE_4_2", 61440},
    {"RTP_LM8_FINAL_SCALE_5_2", 61440},
    {"RTP_LM8_FINAL_SCALE_2_3", 61440},
    {"RTP_LM8_FINAL_SCALE_3_3", 61440},
    {"RTP_LM8_FINAL_SCALE_4_3", 61440},
    {"RTP_LM8_FINAL_SCALE_5_3", 61440},
    {"RTP_LM8_FINAL_SCALE_2_4", 61440},
    {"RTP_LM8_FINAL_SCALE_3_4", 61440},
    {"RTP_LM8_FINAL_SCALE_4_4", 61440},
    {"RTP_LM8_FINAL_SCALE_5_4", 61440},
    {"RTP_LM8_FINAL_SCALE_2_5", 61440},
    {"RTP_LM8_FINAL_SCALE_3_5", 61440},
    {"RTP_LM8_FINAL_SCALE_4_5", 61440},
    {"RTP_LM8_FINAL_SCALE_5_5", 61440},
    {"RTP_LM8_FINAL_SCALE_2_6", 61440},
    {"RTP_LM8_FINAL_SCALE_3_6", 61440},
    {"RTP_LM8_FINAL_SCALE_4_6", 61440},
    {"RTP_LM8_FINAL_SCALE_5_6", 61440},
    {"RTP_LM8_FINAL_SCALE_2_7", 61440},
    {"RTP_LM8_FINAL_SCALE_3_7", 61440},
    {"RTP_LM8_FINAL_SCALE_4_7", 61440},
    {"RTP_LM8_FINAL_SCALE_5_7", 61440},
};

} // namespace gemma4e_dflash_rtp
