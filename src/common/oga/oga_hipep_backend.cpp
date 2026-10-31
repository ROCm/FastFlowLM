/// \file oga_hipep_backend.cpp
#include "oga/oga_hipep_backend.hpp"

#include "oga/oga_api.hpp"
#include "oga/oga_engine.hpp"
#include "utils/debug_utils.hpp"
#include "utils/file_access.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::oga {
namespace {

/// \brief every session on the AMDGPU EP's HIP profile, and no CPU fallback for the decoder
constexpr const char* kSessionOverlay =
    R"({"model":{)"
    R"("decoder":{"session_options":{"provider_options":[{"AMDGPU":{"profile":"hip"}}],"session.disable_cpu_ep_fallback":"1"}},)"
    R"("embedding":{"session_options":{"provider_options":[{"AMDGPU":{"profile":"hip"}}]}},)"
    R"("vision":{"session_options":{"provider_options":[{"AMDGPU":{"profile":"hip"}}]}}}})";

nlohmann::json ReadJson(const std::filesystem::path& path) {
    flm::file_access::ObserveOpen(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    try {
        return nlohmann::json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot parse " + path.string() + ": " + error.what());
    }
}

/// \brief a special token's text, whether stated bare or as {"content": ...}
std::optional<std::string> TokenText(const nlohmann::json& config, const char* key) {
    const auto it = config.find(key);
    if (it == config.end()) return std::nullopt;
    if (it->is_string()) return it->get<std::string>();
    if (it->is_object() && it->contains("content") && (*it)["content"].is_string())
        return (*it)["content"].get<std::string>();
    return std::nullopt;
}

/// \brief the id tokenizer.json gives `text`, from its added tokens or its vocabulary
std::optional<int> TokenId(const nlohmann::json& tokenizer, const std::string& text) {
    if (const auto added = tokenizer.find("added_tokens");
        added != tokenizer.end() && added->is_array()) {
        for (const auto& token : *added)
            if (token.value("content", std::string()) == text && token.contains("id") &&
                token["id"].is_number_integer())
                return token["id"].get<int>();
    }
    if (const auto model = tokenizer.find("model");
        model != tokenizer.end() && model->is_object() && model->contains("vocab")) {
        const auto& vocab = (*model)["vocab"];
        if (vocab.is_object() && vocab.contains(text) && vocab[text].is_number_integer())
            return vocab[text].get<int>();
    }
    return std::nullopt;
}

/// \brief what genai_config.json states about the model
struct GenaiConfig {
    std::string type;
    std::uint32_t vocab_size = 0;
    std::uint32_t context_length = 0;
    std::vector<int> eos_ids;
};

GenaiConfig ReadGenaiConfig(const std::filesystem::path& root) {
    const auto file = root / "genai_config.json";
    const auto document = ReadJson(file);
    const auto model = document.find("model");
    if (model == document.end() || !model->is_object())
        throw std::runtime_error(file.string() + " has no model object");
    GenaiConfig out;
    out.type = model->value("type", std::string());
    out.vocab_size = model->value("vocab_size", 0u);
    out.context_length = model->value("context_length", 0u);
    const auto eos = model->value("eos_token_id", nlohmann::json());
    if (eos.is_number_integer()) out.eos_ids.push_back(eos.get<int>());
    if (eos.is_array())
        for (const auto& id : eos)
            if (id.is_number_integer()) out.eos_ids.push_back(id.get<int>());
    if (out.type.empty() || !out.vocab_size || !out.context_length)
        throw std::runtime_error(file.string() +
                                 " must state model.type, vocab_size and context_length");
    return out;
}

}  // namespace

OgaApi::Owned<OgaModel> OpenModel(const OgaApi& api, const std::filesystem::path& root) {
    const auto& f = api.functions();
    const auto directory = root.u8string();
    OgaConfig* raw_config = nullptr;
    api.Check(f.create_config(reinterpret_cast<const char*>(directory.c_str()), &raw_config),
              "OgaCreateConfig");
    OgaApi::Owned<OgaConfig> config(raw_config, {f.destroy_config});
    api.Check(f.config_overlay(config.get(), kSessionOverlay), "OgaConfigOverlay");
    api.Check(f.config_set_provider_option(config.get(), "AMDGPU", "profile", "hip"),
              "OgaConfigSetProviderOption");
    OgaModel* raw_model = nullptr;
    api.Check(f.create_model_from_config(config.get(), &raw_model), "OgaCreateModelFromConfig");
    OgaApi::Owned<OgaModel> model(raw_model, {f.destroy_model});
    const char* raw_device = nullptr;
    api.Check(f.model_get_device_type(model.get(), &raw_device), "OgaModelGetDeviceType");
    const std::string device = raw_device ? raw_device : "";
    if (raw_device) f.destroy_string(raw_device);
    if (device != "AMDGPU")
        throw std::runtime_error("OGA created " + root.string() + " on '" + device +
                                 "', not on AMDGPU");
    return model;
}

namespace {

class OgaHipepBackend final : public flm::backend::ModelBackend {
public:
    explicit OgaHipepBackend(const flm::backend::BackendContext& context) {
        const std::filesystem::path root(context.model_path);
        const GenaiConfig genai = ReadGenaiConfig(root);
        const std::string name = "OGA " + genai.type;
        if (context.enable_preemption)
            throw std::invalid_argument(name + " does not support preemption");
        const auto length = context.context_length;
        if (length < 1 || length > genai.context_length)
            throw std::out_of_range(name + " context length must be in 1.." +
                                    std::to_string(genai.context_length));
        if (!context.config) throw std::invalid_argument(name + " needs the model's config.json");
        const auto vocab = context.config->get("vocab_size");
        if (vocab != genai.vocab_size)
            throw std::runtime_error(name + " config.json vocab_size " + std::to_string(vocab) +
                                     " differs from genai_config.json vocab_size " +
                                     std::to_string(genai.vocab_size));

        const auto tokenizer_config = context.tokenizer_config
                                          ? *context.tokenizer_config
                                          : ReadJson(root / "tokenizer_config.json");
        const auto tokenizer = ReadJson(root / "tokenizer.json");
        const auto id_of = [&](const char* key) -> std::optional<int> {
            const auto text = TokenText(tokenizer_config, key);
            if (!text) return std::nullopt;
            const auto id = TokenId(tokenizer, *text);
            if (!id)
                throw std::runtime_error(name + " tokenizer.json has no id for " + key + " " +
                                         *text);
            return id;
        };
        if (const auto eos = id_of("eos_token")) eos_ids_.push_back(*eos);
        for (const int id : genai.eos_ids)
            if (std::find(eos_ids_.begin(), eos_ids_.end(), id) == eos_ids_.end())
                eos_ids_.push_back(id);
        if (eos_ids_.empty())
            throw std::runtime_error(name + " package states no EOS token");
        bos_id_ = id_of("bos_token");

        api_ = OgaApi::GetOrCreate();
        auto model = OpenModel(*api_, root);
        header_print("FLM", "OGA device: AMDGPU (" << api_->package().bin.string() << ")");
        engine_ = std::make_unique<OgaEngine>(
            api_, std::move(model), name, genai.vocab_size, genai.context_length, length);
    }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kOgaHipepBackendId; }
    std::string detail() const override {
        const auto& package = api_->package();
        return package.bin.string() + " (onnxruntime-genai " + VersionText(package.oga_version) +
               ", from " + package.source + ")";
    }
    std::uint32_t max_decode_length() const override { return engine_->max_length() - 1; }
    bool supports_preemption() const override { return false; }
    bool forwards_past_eos() const override { return false; }
    std::optional<std::vector<int>> forced_eos_ids() const override { return eos_ids_; }
    std::optional<int> forced_bos_id() const override { return bos_id_; }

private:
    std::vector<int> eos_ids_;
    std::optional<int> bos_id_;
    std::shared_ptr<const OgaApi> api_;
    std::unique_ptr<OgaEngine> engine_;
};

}  // namespace

flm::backend::BackendFactory hipep_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<OgaHipepBackend>(context);
    };
}

}  // namespace flm::oga
