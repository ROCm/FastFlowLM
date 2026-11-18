// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __gemm_detail__
#define __gemm_detail__
#include "npu_utils/npu_utils.hpp"
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "models/gemma_text/gemma_text_gemm.hpp"

struct GemmaTextGemm::Impl
{
private:
    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t total_cols = 8;
    static constexpr uint32_t total_rows = 4;
    static constexpr uint32_t m = 64; //TODO:
    static constexpr uint32_t k = 128;
    static constexpr uint32_t n = 32;
    static constexpr uint32_t CT_rtp_address = 13312; // for 128

public:
    Impl(){}
    Impl(LM_Config& config);

    ~Impl();

    void generate_seq(npu_sequence *seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset);

    LM_Config config;
};

#endif
