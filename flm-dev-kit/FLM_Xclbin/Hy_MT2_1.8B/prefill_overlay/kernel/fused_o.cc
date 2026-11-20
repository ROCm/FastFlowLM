// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Overlay core program for the odd core rows (3, 5): dequant_mm_o + attention.
//
// Both kernel sources go into one object, each inside its own namespace, so
// their same-named helpers (_lock_acquire, passThrough_aie, zero_vectorized,
// the ping / pong globals, ...) don't clash. System headers come first: their
// include guards then keep them out of the namespaces. See README.md.
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
