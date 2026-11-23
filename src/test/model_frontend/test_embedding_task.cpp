/// \file test_embedding_task.cpp
/// \brief Exercise the embeddings task-prompt resolution without a device or network.
#include "embedding_task.hpp"
#include "../phi4_core/test_support.hpp"

namespace {
using ojson = nlohmann::ordered_json;
using Status = flm::EmbeddingTaskResolution::Status;

void check_ok(const ojson& request, const std::string& task) {
    const auto resolved = flm::resolve_embedding_task(request);
    TEST_REQUIRE(resolved.status == Status::Ok);
    TEST_REQUIRE(resolved.task == task);
}

void check_bad(const ojson& request, Status status, const std::string& field) {
    const auto resolved = flm::resolve_embedding_task(request);
    TEST_REQUIRE(resolved.status == status);
    TEST_REQUIRE(resolved.field == field);
}
}

void test_embedding_task() {
    // Every REST name maps to its canonical engine task.
    for (const auto& [name, task] : flm::embedding_task_names()) {
        check_ok(ojson{{"prompt_name", std::string(name)}}, std::string(task));
        check_ok(ojson{{"task_type", std::string(name)}}, std::string(task));
    }
    check_ok(ojson{{"prompt_name", "query"}}, "query");
    check_ok(ojson{{"prompt_name", "search_query"}}, "query");
    check_ok(ojson{{"prompt_name", "Retrieval-query"}}, "query");
    check_ok(ojson{{"prompt_name", "document"}}, "document");
    check_ok(ojson{{"prompt_name", "search_document"}}, "document");
    check_ok(ojson{{"prompt_name", "Retrieval-document"}}, "document");
    check_ok(ojson{{"prompt_name", "clustering"}}, "clustering");
    check_ok(ojson{{"prompt_name", "Clustering"}}, "clustering");
    check_ok(ojson{{"prompt_name", "classification"}}, "classification");
    check_ok(ojson{{"prompt_name", "Classification"}}, "classification");
    check_ok(ojson{{"prompt_name", "STS"}}, "sentence_similarity");
    check_ok(ojson{{"prompt_name", "sentence_similarity"}}, "sentence_similarity");
    check_ok(ojson{{"prompt_name", "summarization"}}, "summarization");
    check_ok(ojson{{"prompt_name", "Summarization"}}, "summarization");
    check_ok(ojson{{"prompt_name", "code_retrieval"}}, "code_retrieval");

    // No task at all keeps today's default.
    TEST_REQUIRE(flm::resolve_embedding_task(ojson{{"model", "m"}}).status == Status::Absent);

    // Unknown names are refused, including names the engine has no prefix for
    // and an empty string, which names no task.
    for (const char* name : {"not_a_task", "", "MultilabelClassification", "BitextMining",
                             "search_result", "Retrieval"}) {
        check_bad(ojson{{"prompt_name", name}}, Status::Unknown, "prompt_name");
    }
    const auto unknown = flm::resolve_embedding_task(ojson{{"task_type", "not_a_task"}});
    TEST_REQUIRE(unknown.status == Status::Unknown);
    TEST_REQUIRE(unknown.field == "task_type");
    TEST_REQUIRE(unknown.value == "not_a_task");

    // Non-string values are refused, even beside a valid alias.
    check_bad(ojson{{"prompt_name", 7}}, Status::NotAString, "prompt_name");
    check_bad(ojson{{"prompt_name", "query"}, {"task_type", 7}}, Status::NotAString, "task_type");

    // Both fields are fine only when they name the same task.
    check_ok(ojson{{"prompt_name", "document"}, {"task_type", "search_document"}}, "document");
    check_ok(ojson{{"prompt_name", "query"}, {"task_type", "search_query"}}, "query");
    check_bad(ojson{{"prompt_name", "query"}, {"task_type", "document"}}, Status::Conflict, "task_type");
    const auto conflict =
        flm::resolve_embedding_task(ojson{{"prompt_name", "document"}, {"task_type", "query"}});
    TEST_REQUIRE(conflict.status == Status::Conflict);
    TEST_REQUIRE(conflict.value == "query");

    // Canonicalization passes a provider's own `task` vocabulary through.
    TEST_REQUIRE(flm::canonical_embedding_task("search_query") == "query");
    TEST_REQUIRE(flm::canonical_embedding_task("STS") == "sentence_similarity");
    TEST_REQUIRE(flm::canonical_embedding_task("provider-task") == "provider-task");
    TEST_REQUIRE(flm::canonical_embedding_task("query") == "query");

    // Error messages quote the REST names.
    RequireContains(flm::embedding_task_names_csv(), "search_query");
    RequireContains(flm::embedding_task_names_csv(), "search_document");
    RequireContains(flm::embedding_task_names_csv(), "STS");
}
