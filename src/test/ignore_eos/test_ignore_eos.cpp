/// \file test_ignore_eos.cpp
/// \brief Grammar that allows only EOS, with ignore_eos set, must sample a
///        non-EOS token and drop the constraint. No NPU.
#include "AutoModel/automodel.hpp"
#include "chat/decode_constraint.hpp"

#include <cstdio>
#include <memory>

// The real tokenizer links the Rust tokenizers library, and the real builtin
// registry pulls in every NPU engine. This test only calls _sample, so both
// stay empty, the same way src/test/model_backend stubs the registry.
Tokenizer::Tokenizer(const std::string&) : is_doubled_encoded(false) {}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string&) { return {}; }
std::string Tokenizer::decode(const std::vector<int>&) { return {}; }
std::string Tokenizer::run_time_decoder(int) { return {}; }

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}

namespace {

int failures = 0;

/// \brief A grammar whose only legal token is EOS.
class eos_only_constraint : public decode_constraint {
public:
    eos_only_constraint(size_t vocab, int eos) : vocab_(vocab), eos_(eos) {}

    bool fill_mask(std::vector<uint32_t>& mask) override {
        mask.assign((vocab_ + 31) / 32, 0u);
        if (eos_ >= 0 && static_cast<size_t>(eos_) < vocab_) {
            mask[static_cast<size_t>(eos_) / 32] |= 1u << (eos_ % 32);
        }
        return true;
    }

    void accept(int) override {}

private:
    size_t vocab_;
    int eos_;
};

/// \brief Exposes _sample without loading a model.
class sample_probe : public AutoModel {
public:
    sample_probe() : AutoModel(nullptr, "ignore-eos-test") {}

    std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>) override {
        return {};
    }
    bool insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) override {
        return false;
    }
    std::string generate_with_prompt(chat_meta_info_t&, lm_uniform_input_t&, int, std::ostream&) override {
        return {};
    }
    std::string apply_chat_template(nlohmann::ordered_json&, nlohmann::ordered_json, bool) override {
        return {};
    }

    int sample(buffer<bf16>& logits, chat_meta_info_t& meta, int vocab, int eos) {
        sampler_config config;
        config.top_k = 1;
        config.top_p = 1.0f;
        config.min_p = 0.0f;
        config.temperature = 0.0f;
        config.rep_penalty = 1.0f;
        config.freq_penalty = 0.0f;
        config.pre_penalty = 0.0f;
        config.repeat_last_n = 0;
        this->eos_token_ids = {eos};
        this->sampler = std::make_unique<Sampler>(vocab, config);
        return this->_sample(logits, meta);
    }
};

}  // namespace

int main() {
    constexpr int kVocab = 8;
    constexpr int kEos = 3;
    constexpr int kPreferred = 5;

    sample_probe model;
    buffer<bf16> logits(static_cast<size_t>(kVocab));
    for (int i = 0; i < kVocab; ++i) {
        logits[static_cast<size_t>(i)] = 0.0f;
    }
    // EOS is the raw argmax. Once it is masked, kPreferred is the winner.
    logits[static_cast<size_t>(kEos)] = 5.0f;
    logits[static_cast<size_t>(kPreferred)] = 2.0f;

    chat_meta_info_t meta;
    meta.ignore_eos = true;
    meta.min_tokens = 0;
    meta.generated_tokens = 0;
    meta.logprobs_top_n = -1;
    meta.constraint = std::make_shared<eos_only_constraint>(kVocab, kEos);

    const int token = model.sample(logits, meta, kVocab, kEos);
    if (token == kEos) {
        std::printf("FAIL: ignore_eos sampled EOS %d\n", token);
        failures++;
    }
    if (token != kPreferred) {
        std::printf("FAIL: sampled %d, expected non-EOS token %d\n", token, kPreferred);
        failures++;
    }
    if (meta.constraint != nullptr) {
        std::printf("FAIL: constraint was kept after an EOS-only grammar\n");
        failures++;
    }

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
