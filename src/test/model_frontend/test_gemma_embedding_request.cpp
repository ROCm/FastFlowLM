/// \file test_gemma_embedding_request.cpp
/// \brief Exercise the built-in request formatting with stand-in tokenization and inference.
#include "AutoEmbeddingModel/modeling_gemma_embedding.hpp"
#include "../phi4_core/test_support.hpp"

namespace {
std::string encoded_text;
class FakeEmbedding : public embedding_model {
public:
    void load_weights(Q4NX&) override {}
    buffer<bf16> embed(std::vector<int>&) override {
        return buffer<bf16>(std::vector<bf16>(768));
    }
};
class TestGemma : public Gemma_Embedding {
public:
    TestGemma() : Gemma_Embedding(nullptr) {
        tokenizer = std::make_unique<Tokenizer>("");
        embedding_model_impl = std::make_unique<FakeEmbedding>();
    }
};
}
Tokenizer::Tokenizer(const std::string&) {}
std::vector<int> Tokenizer::encode(const std::string& text) {
    encoded_text = text;
    return {2, 3, 1};
}
void AutoEmbeddingModel::_shared_load_model(std::string, json, bool) {
    throw std::logic_error("The request test must not load weights");
}

void test_gemma_embedding_request() {
    TestGemma model;
    const std::pair<const char*, const char*> tasks[] = {
        {"query", "task: search result | query: "},
        {"document", "title: none | text: "},
        {"clustering", "task: clustering | query: "},
        {"classification", "task: classification | query: "},
        {"code_retrieval", "task: code retrieval | query: "},
        {"sentence_similarity", "task: sentence similarity | query: "},
        {"summarization", "task: summarization | query: "},
    };
    for (const auto& [task, prefix] : tasks) {
        const auto result = model.embed({{"input", "hello"}, {"task", task}, {"dimensions", 768}});
        TEST_REQUIRE(encoded_text == std::string("<bos>") + prefix + "hello<eos>");
        TEST_REQUIRE(result.values.size() == 768 && result.prompt_tokens == 3);
    }
    for (const json& request : {json{{"input", "hello"}}, json{{"input", "hello"}, {"task", ""}}}) {
        model.embed(request);
        TEST_REQUIRE(encoded_text == "<bos>task: search result | query: hello<eos>");
    }
    model.embed({{"input", "hello"}, {"task", "document"}, {"title", "Example"}});
    TEST_REQUIRE(encoded_text == "<bos>title: Example | text: hello<eos>");
    for (const char* task : {"BitextMining", "Retrieval", "Clustering", "Classification",
                            "MultilabelClassification", "CodeRetrieval", "SentenceSimilarity",
                            "Summarization", "STS", "unknown"}) {
        RequireThrows<std::invalid_argument>([&] { model.embed({{"input", "hello"}, {"task", task}}); });
    }
    RequireThrows<std::invalid_argument>([&] { model.embed({{"input", "hello"}, {"title", "Example"}}); });
    RequireThrows<std::invalid_argument>([&] { model.embed({{"input", "hello"}, {"dimensions", 128}}); });
}
