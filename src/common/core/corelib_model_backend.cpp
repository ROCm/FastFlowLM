/// \file corelib_model_backend.cpp
#include "core/corelib_model_backend.hpp"

#include "core/corelib_runtime.hpp"
#include "utils/file_access.hpp"
#include "utils/utils.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace flm::core {
namespace {

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

bool NamesAGguf(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".gguf";
}

/// \brief the catalog's one GGUF or, when it names none, the directory's one
std::filesystem::path RequireGguf(const nlohmann::ordered_json& model_info,
                                  const std::filesystem::path& root, const std::string& name) {
    std::vector<std::string> named;
    if (const auto files = model_info.find("files");
        files != model_info.end() && files->is_array()) {
        for (const auto& file : *files)
            if (file.is_string() && NamesAGguf(file.get<std::string>()))
                named.push_back(file.get<std::string>());
    }
    if (named.size() == 1) {
        const auto path = root / named.front();
        if (!std::filesystem::exists(path))
            throw std::runtime_error(name + " core: the catalog entry names " + named.front() +
                                     " but " + path.string() + " does not exist");
        return path;
    }
    if (named.size() > 1)
        throw std::runtime_error(name + " core: the catalog entry names " +
                                 std::to_string(named.size()) +
                                 " GGUFs; it must name exactly one");
    std::vector<std::filesystem::path> found;
    std::error_code failed;
    for (const auto& entry : std::filesystem::directory_iterator(root, failed))
        if (entry.is_regular_file(failed) && NamesAGguf(entry.path())) found.push_back(entry.path());
    if (found.size() == 1) return found.front();
    throw std::runtime_error(name + " core: the catalog entry names no GGUF and " + root.string() +
                             " holds " + std::to_string(found.size()) +
                             " of them; it must hold exactly one");
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

struct PackageTokens {
    std::vector<int> eos;
    std::optional<int> bos;
};

PackageTokens ReadPackageTokens(const BackendSpec& spec,
                                const flm::backend::BackendContext& context) {
    PackageTokens out;
    out.eos = spec.eos_ids;
    if (!spec.eos_ids.empty() && !spec.read_bos) return out;

    const std::filesystem::path root(context.model_path);
    const auto& name = spec.model.display_name;
    const auto config = context.tokenizer_config ? *context.tokenizer_config
                                                 : ReadJson(root / "tokenizer_config.json");
    const auto tokenizer = ReadJson(root / "tokenizer.json");
    const auto require = [&](const char* key) {
        const auto text = TokenText(config, key);
        if (!text) throw std::runtime_error(name + " tokenizer_config.json has no " + key);
        const auto id = TokenId(tokenizer, *text);
        if (!id)
            throw std::runtime_error(name + " tokenizer.json has no id for " + key + " " + *text);
        return *id;
    };

    if (spec.eos_ids.empty()) {
        out.eos.push_back(require("eos_token"));
        const auto model_config = root / "config.json";
        if (std::filesystem::exists(model_config)) {
            const auto document = ReadJson(model_config);
            if (!document.is_object())
                throw std::runtime_error(name + " " + model_config.string() +
                                         " is not a JSON object");
            const auto listed = document.value("eos_token_id", nlohmann::json());
            std::vector<int> ids;
            if (listed.is_number_integer()) ids.push_back(listed.get<int>());
            if (listed.is_array())
                for (const auto& id : listed)
                    if (id.is_number_integer()) ids.push_back(id.get<int>());
            for (const int id : ids)
                if (std::find(out.eos.begin(), out.eos.end(), id) == out.eos.end())
                    out.eos.push_back(id);
        }
    }
    if (spec.read_bos) out.bos = require("bos_token");
    return out;
}

class CorelibBackend final : public flm::backend::ModelBackend {
public:
    CorelibBackend(const BackendSpec& spec, const flm::backend::BackendContext& context) {
        const auto& name = spec.model.display_name;
        if (context.enable_preemption)
            throw std::invalid_argument(name + " core does not support preemption");
        const auto length = context.context_length;
        const bool fits = spec.context_fits ? spec.context_fits(length)
                                            : length >= 1 && length <= spec.model.context_limit;
        if (!fits)
            throw std::out_of_range(name + " core context length must be in " + spec.context_rule);
        if (spec.reliable_context && length > spec.reliable_context)
            header_print("WARNING", name << " core context " << length << " is past "
                                         << spec.reliable_context << ": "
                                         << spec.unreliable_reason);
        context_length_ = length;

        auto tokens = ReadPackageTokens(spec, context);
        eos_ids_ = std::move(tokens.eos);
        bos_id_ = tokens.bos;
        const auto gguf = RequireGguf(context.model_info, context.model_path, name);

        runtime_ = corelib::CorelibRuntime::GetOrCreate(utils::get_executable_directory());
        auto engine = std::make_unique<CorelibModel>(spec.model, gguf, runtime_, length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~CorelibBackend() override { engine_.reset(); }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kCoreBackendId; }
    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string() : std::string();
    }
    std::uint32_t max_decode_length() const override { return context_length_ - 1; }
    bool supports_preemption() const override { return false; }
    /// \note corelib refuses a forward past its context, so the extra one the
    ///       FastFlowLM engines take after an EOS token would fail here.
    bool forwards_past_eos() const override { return false; }
    bool aligns_prefill_chunks() const override { return true; }
    bool poisoned() const noexcept override { return engine_ && engine_->poisoned(); }
    std::optional<std::vector<int>> forced_eos_ids() const override { return eos_ids_; }
    std::optional<int> forced_bos_id() const override { return bos_id_; }

private:
    std::uint32_t context_length_{};
    std::vector<int> eos_ids_;
    std::optional<int> bos_id_;
    // Declared before the engine so it is destroyed after it: corelib objects
    // must not outlive the API they came from.
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<CorelibModel> engine_;
};

}  // namespace

flm::backend::BackendFactory Factory(BackendSpec spec) {
    return [spec = std::move(spec)](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<CorelibBackend>(spec, context);
    };
}

}  // namespace flm::core
