// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/embedding.hpp"

#include <stdexcept>

namespace flm::iron {

void Embedding::load_model(std::string model_path, json,
                           bool enable_preemption) {
    if (enable_preemption) {
        throw std::invalid_argument(
            "The IRON embedding backend does not support preemption");
    }
    worker_ = std::make_unique<Worker>(model_path, worker_model_);
    is_model_loaded = true;
    this->model_path = std::move(model_path);
}

std::vector<float> Embedding::embed(std::string& text,
                                    embedding_task_type_t task_type) {
    if (!worker_) {
        throw std::runtime_error("The IRON embedding model is not loaded");
    }
    return worker_->embed(text, task_type == task_document ? "document" : "query",
                          dimensions_);
}

}  // namespace flm::iron