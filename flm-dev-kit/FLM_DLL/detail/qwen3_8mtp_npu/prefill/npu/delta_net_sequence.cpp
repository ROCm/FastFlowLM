// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/delta_net_sequence.cpp
/// \brief The one translation unit that defines the delta-net prefill sequence.
///
/// `GateDeltaNetPrefillRuntimeSequence.hpp` in the shared include path defines
/// `generate_gate_delta_net_prefill_sequence` without `inline`, so exactly one
/// TU in this .so may include it. That is this file, and it does nothing else:
/// callers include `delta_net_sequence.hpp` for the declaration.
/// The generator names `bf16` in a sizeof without including anything that
/// defines it -- it is written to be included after the engine's own headers,
/// as qwen3_5vl_npu.cpp does. typedef.hpp first, then.
#include "typedef.hpp"

#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"

#include "prefill/npu/delta_net_sequence.hpp"
