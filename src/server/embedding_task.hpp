/*!
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file embedding_task.hpp
 * \brief Resolve the task prompt of an embeddings request.
 */
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace flm {

/// The REST names for an embedding task, and the engine `task` name each maps
/// to. NOT the model's own vocabulary: several names are aliases for one task,
/// and an error message must quote these REST names because those are the
/// values the validator accepts. Names with no engine prefix behind them
/// (for example `MultilabelClassification` or `BitextMining`) are deliberately
/// absent: accepting one would silently embed under a different task.
inline const std::vector<std::pair<std::string_view, std::string_view>>& embedding_task_names() {
    static const std::vector<std::pair<std::string_view, std::string_view>> kTasks = {
        {"query", "query"},
        {"search_query", "query"},
        {"Retrieval-query", "query"},
        {"document", "document"},
        {"search_document", "document"},
        {"Retrieval-document", "document"},
        {"clustering", "clustering"},
        {"Clustering", "clustering"},
        {"classification", "classification"},
        {"Classification", "classification"},
        {"STS", "sentence_similarity"},
        {"sentence_similarity", "sentence_similarity"},
        {"summarization", "summarization"},
        {"Summarization", "summarization"},
        {"code_retrieval", "code_retrieval"},
    };
    return kTasks;
}

/// The canonical engine `task` name for a REST task name, or the name itself
/// when it is not a known alias (a provider's own `task` vocabulary passes
/// through untouched for the engine to judge).
inline std::string canonical_embedding_task(const std::string& name) {
    for (const auto& entry : embedding_task_names()) {
        if (name == entry.first) return std::string(entry.second);
    }
    return name;
}

/// Comma-separated REST task names, for error messages.
inline std::string embedding_task_names_csv() {
    std::string out;
    for (const auto& entry : embedding_task_names())
        out += (out.empty() ? "" : ", ") + std::string(entry.first);
    return out;
}

/// The outcome of reading a request's task prompt.
struct EmbeddingTaskResolution {
    enum class Status { Ok, Absent, NotAString, Unknown, Conflict };

    Status status = Status::Absent;
    std::string task;   ///< canonical engine task name, valid only when Ok
    std::string field;  ///< which key this is about
    std::string value;  ///< the offending value, when Unknown or Conflict
};

/// Read `prompt_name`, or its accepted alias `task_type`. BOTH are checked: an
/// invalid `task_type` beside a valid `prompt_name` is refused rather than
/// ignored, and sending both is fine only when they name the same task.
/// The handler calls this only for models that use the built-in task prompts;
/// extension/provider models never reach it (their requests are untouched).
inline EmbeddingTaskResolution resolve_embedding_task(const nlohmann::ordered_json& request) {
    EmbeddingTaskResolution out;
    bool have = false;
    for (const char* field : {"prompt_name", "task_type"}) {
        if (!request.contains(field)) continue;
        const nlohmann::ordered_json& value = request.at(field);
        if (!value.is_string())
            return {EmbeddingTaskResolution::Status::NotAString, std::string(), field, std::string()};
        const std::string want = value.get<std::string>();
        const auto& table = embedding_task_names();
        auto hit = table.end();
        for (auto it = table.begin(); it != table.end(); ++it) {
            if (want == it->first) {
                hit = it;
                break;
            }
        }
        if (hit == table.end())
            return {EmbeddingTaskResolution::Status::Unknown, std::string(), field, want};
        if (have && hit->second != out.task)
            return {EmbeddingTaskResolution::Status::Conflict, std::string(), field, want};
        out.task = std::string(hit->second);
        out.field = field;
        have = true;
    }
    out.status = have ? EmbeddingTaskResolution::Status::Ok : EmbeddingTaskResolution::Status::Absent;
    return out;
}

}  // namespace flm
