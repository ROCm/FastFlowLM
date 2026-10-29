// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "AutoEmbeddingModel/auto_embedding_model.hpp"
#include "iron/worker.hpp"

namespace flm::iron {

class Embedding final : public AutoEmbeddingModel {
public:
    Embedding(flm_rt::device* device, std::string model_tag,
              std::string worker_model, std::uint32_t dimensions)
        : AutoEmbeddingModel(device, std::move(model_tag)),
          worker_model_(std::move(worker_model)), dimensions_(dimensions) {}

    void load_model(std::string model_path, json, bool enable_preemption) override;
    std::vector<float> embed(std::string& text,
                             embedding_task_type_t task_type) override;

private:
    std::string worker_model_;
    std::uint32_t dimensions_;
    std::unique_ptr<Worker> worker_;
};

}  // namespace flm::iron