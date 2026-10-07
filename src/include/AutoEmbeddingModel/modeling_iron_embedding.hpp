// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "AutoEmbeddingModel/auto_embedding_model.hpp"
#include "iron/worker.hpp"

class IronEmbedding final : public AutoEmbeddingModel {
public:
    explicit IronEmbedding(flm_rt::device* device)
        : AutoEmbeddingModel(device, "embeddinggemma-2:740m") {}

    void load_model(std::string model_path, json, bool enable_preemption) override;
    std::vector<float> embed(std::string& text,
                             embedding_task_type_t task_type) override;

private:
    std::unique_ptr<flm::iron::Worker> worker_;
};