// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "models/llama3/iron/llama3_iron_backend.hpp"

#include "iron/causal_lm.hpp"

namespace flm::llama3 {

flm::backend::BackendFactory iron_factory() {
    return flm::iron::causal_factory("llama");
}

}  // namespace flm::llama3