// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "AutoModel/model_backend.hpp"

namespace flm::llama3 {

flm::backend::BackendFactory iron_factory();

inline flm::backend::BackendTraits iron_traits() {
    return flm::backend::BackendTraits{false, false, 32768};
}

}  // namespace flm::llama3