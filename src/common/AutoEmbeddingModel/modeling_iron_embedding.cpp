// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "AutoEmbeddingModel/modeling_iron_embedding.hpp"

#include <cstdlib>
#include <stdexcept>

void IronEmbedding::load_model(std::string model_path, json, bool enable_preemption) {
    if (enable_preemption) {
        throw std::invalid_argument("The IRON embedding backend does not support preemption");
    }
    const char* override_path = std::getenv("FLM_IRON_EMBEDDING_PATH");
    if (override_path && *override_path) model_path = override_path;
    worker_ = std::make_unique<flm::iron::Worker>(model_path, 512, "embedding");
    is_model_loaded = true;
    this->model_path = std::move(model_path);
}

std::vector<float> IronEmbedding::embed(std::string& text,
                                        embedding_task_type_t task_type) {
    if (!worker_) throw std::runtime_error("The IRON embedding model is not loaded");
    return worker_->embed(text, task_type == task_document ? "document" : "query");
}