// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "packet_control_gen.hpp"
#include "gptossParam.hpp"
#include "common_external.hpp"
#include "passThroughCommon.hpp"


extern "C" {

void remove_hidden_state_padding(
    bfloat16* hidden_state_out,
    const bfloat16 *hidden_state_padded_in
){
    copy_bf16_to_bf16<D>(hidden_state_out, hidden_state_padded_in+16); // 16 is the offset from the packet header
}
}




