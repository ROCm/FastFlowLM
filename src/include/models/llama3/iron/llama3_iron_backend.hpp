// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "AutoModel/model_backend.hpp"
#include "iron/causal_lm.hpp"

namespace flm::llama3 {

flm::backend::BackendFactory iron_factory();

inline flm::backend::BackendTraits iron_traits() {
    return flm::iron::causal_traits();
}

}  // namespace flm::llama3