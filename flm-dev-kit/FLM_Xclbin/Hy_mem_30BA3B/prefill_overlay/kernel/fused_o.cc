// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Overlay core program for the odd core rows (3, 5): dequant_mm_o + attention.
// See fused_e.cc for why the kernels sit in namespaces.
#include <aie_api/aie.hpp>
#include <type_traits>
#include <math.h>
#include "overlay_layout.h"

namespace dq {
#include "dequant_mm_o.cc"
}

namespace at {
#include "attn.cc"
}

#define DQ_EVEN 0
#include "dispatch.h"
