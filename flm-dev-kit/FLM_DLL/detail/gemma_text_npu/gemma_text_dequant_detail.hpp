// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "models/gemma_text/gemma_text_dequant.hpp"

struct GemmaTextDequant::Impl{
private:
    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr u32 total_cols = 8;
    static constexpr u32 total_rows = 4;
public:
    Impl(){}
    Impl(LM_Config& config);
    ~Impl();

    void generate_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset);

    LM_Config config;
};
