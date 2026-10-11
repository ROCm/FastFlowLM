#include "models/qwen36/rai/aie_next/qwen36_rai_backend.hpp"

#include "models/qwen36/rai/aie_next/qwen36_rai.hpp"
#include "rai/corelib_runtime.hpp"
#include "utils/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace flm::qwen36 {
namespace {

std::string GgufName(const nlohmann::ordered_json& model_info) {
    std::string found;
    if (model_info.contains("files") && model_info["files"].is_array()) {
        for (const auto& file : model_info["files"]) {
            if (!file.is_string()) continue;
            const auto name = file.get<std::string>();
            if (name.size() > 5 && name.ends_with(".gguf")) {
                if (!found.empty())
                    throw std::runtime_error("Qwen3.6 rai entry lists more than one .gguf file");
                found = name;
            }
        }
    }
    if (found.empty()) throw std::runtime_error("Qwen3.6 rai entry lists no .gguf file");
    return found;
}

// The published tokenizer_config names eos_token as a string and does not
// carry eos_token_id. The id is the added token of that same string.
std::vector<int> EosIds(const std::filesystem::path& model) {
    std::ifstream config_file(model / "tokenizer_config.json");
    if (!config_file) throw std::runtime_error("Qwen3.6 tokenizer_config.json is missing");
    const auto config = nlohmann::json::parse(config_file);
    if (!config.contains("eos_token") || !config["eos_token"].is_string())
        throw std::runtime_error("Qwen3.6 tokenizer_config.json has no eos_token string");
    const auto eos = config["eos_token"].get<std::string>();

    std::ifstream tokenizer_file(model / "tokenizer.json");
    if (!tokenizer_file) throw std::runtime_error("Qwen3.6 tokenizer.json is missing");
    const auto tokenizer = nlohmann::json::parse(tokenizer_file);
    std::vector<int> ids;
    if (tokenizer.contains("added_tokens") && tokenizer["added_tokens"].is_array()) {
        for (const auto& token : tokenizer["added_tokens"]) {
            if (token.value("content", std::string()) == eos && token.contains("id") &&
                token["id"].is_number_integer())
                ids.push_back(token["id"].get<int>());
        }
    }
    if (ids.empty())
        throw std::runtime_error("Qwen3.6 tokenizer.json has no id for eos token " + eos);
    return ids;
}

class RaiBackend final : public flm::backend::ModelBackend {
public:
    explicit RaiBackend(const flm::backend::BackendContext& context) {
        if (context.enable_preemption)
            throw std::invalid_argument("Qwen3.6 rai does not support preemption");
        if (context.context_length < 1 || context.context_length > kRaiContextLimit)
            throw std::out_of_range("Qwen3.6 rai context length must be in 1.." +
                                    std::to_string(kRaiContextLimit));
        context_length_ = context.context_length;
        eos_ids_ = EosIds(context.model_path);

        runtime_ = corelib::CorelibRuntime::GetOrCreate(utils::get_executable_directory());
        auto engine = std::make_unique<qwen36_rai>(
            std::filesystem::path(context.model_path) / GgufName(context.model_info),
            runtime_, context.context_length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~RaiBackend() override { engine_.reset(); }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kRaiBackendId; }
    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string() : std::string();
    }
    std::uint32_t max_decode_length() const override { return context_length_ - 1; }
    bool supports_preemption() const override { return false; }
    bool forwards_past_eos() const override { return false; }
    bool poisoned() const noexcept override { return engine_ && engine_->poisoned(); }
    std::optional<std::vector<int>> forced_eos_ids() const override { return eos_ids_; }

private:
    std::uint32_t context_length_{};
    std::vector<int> eos_ids_;
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<qwen36_rai> engine_;
};

}  // namespace

flm::backend::BackendFactory rai_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<RaiBackend>(context);
    };
}

}  // namespace flm::qwen36
