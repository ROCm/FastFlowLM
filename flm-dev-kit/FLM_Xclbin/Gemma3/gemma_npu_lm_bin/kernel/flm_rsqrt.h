// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

#include <aie_api/aie.hpp>

// 1 / sqrt(x) for a positive float: one hardware invsqrt instruction. Keep x a float.
static inline float flm_rsqrt(float x) { return aie::invsqrt(x); }
