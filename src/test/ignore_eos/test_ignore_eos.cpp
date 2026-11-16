/// \file test_ignore_eos.cpp
/// \brief Grammar that allows only EOS, with ignore_eos set, must sample a
///        non-EOS token and drop the constraint; a speculative verify row
///        (_sample_row) masks EOS the same way, counting min_tokens from the
///        rows committed before it. No NPU.
#include "AutoModel/automodel.hpp"
#include "chat/decode_constraint.hpp"

#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

// The real tokenizer links the Rust tokenizers library, and the real builtin
// registry pulls in every NPU engine. This test only calls _sample and _sample_row, so both
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

    /// \brief a sampling sampler (top_k 20, temperature 1, seeded) for row()
    void use_sampling(int vocab, int eos) {
        sampler_config config;
        config.top_k = 20;
        config.top_p = 1.0f;
        config.min_p = 0.0f;
        config.temperature = 1.0f;
        config.has_rng_seed = true;
        config.rng_seed = 99;
        config.repeat_last_n = 0;
        this->eos_token_ids = {eos};
        this->sampler = std::make_unique<Sampler>(vocab, config);
    }

    int row(const std::vector<float>& logits, chat_meta_info_t& meta, int ahead, bool greedy) {
        return this->_sample_row(logits.data(), static_cast<int>(logits.size()), meta, ahead, greedy);
    }
};

/// \brief Allows a fixed set of tokens and records what it is told.
class set_constraint : public decode_constraint {
public:
    set_constraint(size_t vocab, std::vector<int> ids) : vocab_(vocab), ids_(std::move(ids)) {}

    bool fill_mask(std::vector<uint32_t>& mask) override {
        mask.assign((vocab_ + 31) / 32, 0u);
        for (int id : ids_) mask[static_cast<size_t>(id) / 32] |= 1u << (id % 32);
        return true;
    }

    void accept(int token) override { accepted.push_back(token); }

    std::vector<int> accepted;

private:
    size_t vocab_;
    std::vector<int> ids_;
};

/// \brief _sample_row(): the EOS mask of a speculative verify row
void check_rows() {
    constexpr int kVocab = 64;
    constexpr int kEos = 5;
    // EOS is by far the most likely token; with top_k 20 the others still get drawn
    std::vector<float> logits(kVocab, -1.0f);
    logits[kEos] = 4.0f;
    for (int i = 0; i < 8; ++i) logits[static_cast<size_t>(20 + i)] = 0.5f;

    sample_probe model;
    model.use_sampling(kVocab, kEos);

    chat_meta_info_t meta;
    meta.logprobs_top_n = -1;
    meta.ignore_eos = true;
    for (int n = 0; n < 2000; ++n) {
        if (model.row(logits, meta, n % 8, false) == kEos) {
            std::printf("FAIL: ignore_eos: a sampled verify row gave EOS\n");
            failures++;
            break;
        }
    }

    // min_tokens 6 with 3 tokens generated: rows 0..2 of the batch are reply tokens 3..5, still masked; row 3 is
    // reply token 6, where EOS is allowed again
    meta.ignore_eos = false;
    meta.min_tokens = 6;
    meta.generated_tokens = 3;
    int eos_unmasked = 0;
    for (int n = 0; n < 2000; ++n) {
        const int ahead = n % 4;
        const int token = model.row(logits, meta, ahead, false);
        if (ahead < 3 && token == kEos) {
            std::printf("FAIL: min_tokens: row %d (reply token %d) gave EOS\n", ahead, 3 + ahead);
            failures++;
            break;
        }
        if (ahead == 3 && token == kEos) eos_unmasked++;
    }
    if (eos_unmasked < 250) {
        std::printf("FAIL: min_tokens: EOS taken %d of 500 times once allowed\n", eos_unmasked);
        failures++;
    }
    if (model.row(logits, meta, 2, true) == kEos || model.row(logits, meta, 3, true) != kEos) {
        std::printf("FAIL: min_tokens: greedy rows do not count the batch's earlier rows\n");
        failures++;
    }

    // ignore_eos and a grammar that only allows EOS: the row drops the grammar, as _sample() does
    meta.min_tokens = 0;
    meta.generated_tokens = 0;
    meta.ignore_eos = true;
    meta.constraint = std::make_shared<eos_only_constraint>(kVocab, kEos);
    if (model.row(logits, meta, 0, false) == kEos || meta.constraint != nullptr) {
        std::printf("FAIL: ignore_eos: an EOS-only grammar was not dropped on a verify row\n");
        failures++;
    }

    // NaN logits on a greedy row: the first allowed token, reported to the constraint, as in _sample()
    meta.ignore_eos = false;
    auto allowed = std::make_shared<set_constraint>(kVocab, std::vector<int>{9, 12});
    meta.constraint = allowed;
    std::vector<float> nan_row(kVocab, std::numeric_limits<float>::quiet_NaN());
    const int token = model.row(nan_row, meta, 0, true);
    if (token != 9 || allowed->accepted != std::vector<int>{9}) {
        std::printf("FAIL: NaN greedy row gave %d, expected the first allowed token 9\n", token);
        failures++;
    }
}

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

    check_rows();

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
