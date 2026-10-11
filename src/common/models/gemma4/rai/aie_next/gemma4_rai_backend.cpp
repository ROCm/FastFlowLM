/// \file gemma4_rai_backend.cpp
/// \brief The ryzenai-corelib backend for Gemma 4 (E2B and E4B)
#include "models/gemma4/rai/aie_next/gemma4_rai_backend.hpp"

#include "models/gemma4/rai/aie_next/gemma4_rai.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"
#include "utils/file_access.hpp"
#include "utils/utils.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace flm::gemma4 {
namespace {

/// \brief read a JSON file, recording the open for the file-access audit
/// \param path the file
/// \return the parsed document
/// \throws std::runtime_error naming the path, when it cannot be opened or
///         parsed -- a package problem has to name the file it is about
nlohmann::json ReadJson(const std::filesystem::path& path) {
    flm::file_access::ObserveOpen(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    try {
        return nlohmann::json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot parse " + path.string() + ": " +
                                 error.what());
    }
}

/// \brief case-insensitive ".gguf" test
bool NamesAGguf(std::string_view name) {
    constexpr std::string_view suffix = ".gguf";
    if (name.size() <= suffix.size()) return false;
    auto tail = name.substr(name.size() - suffix.size());
    return std::equal(tail.begin(), tail.end(), suffix.begin(),
                      [](unsigned char a, unsigned char b) {
                          return std::tolower(a) == std::tolower(b);
                      });
}

/// \brief the GGUF this package holds
///
/// NOT A CONSTANT, and that is the one structural difference from Phi-4's
/// backend. Phi-4 can name `Phi-4-mini-instruct.Q8_0.gguf` outright because
/// one file serves the whole family; Gemma 4's two shipped rows are
/// `gemma-4-E2B-it-Q8_0.gguf` and `gemma-4-E4B-it-Q8_0.gguf` under the SAME
/// family `gemma4e`, so a literal here would load exactly one of them and
/// report "file not found" for the other -- an error naming a file the user
/// never asked for.
///
/// The catalog entry is the authority: `files` is written by the same
/// model_list.json entry the downloader fetched the package with, so it names
/// the file that is actually on disk. A hand-assembled directory has no
/// catalog entry to consult, so a directory scan is the fallback -- one GGUF
/// in a directory is unambiguous, and anything else is named rather than
/// guessed at.
///
/// \param model_info the catalog entry, as AutoModel passes it down
/// \param root the model directory
/// \return the path to open
/// \throws std::runtime_error naming what it found, when the answer is not
///         exactly one file
std::filesystem::path RequireGgufPath(const nlohmann::ordered_json& model_info,
                                      const std::filesystem::path& root) {
    std::vector<std::string> named;
    if (const auto files = model_info.find("files");
        files != model_info.end() && files->is_array()) {
        for (const auto& entry : *files) {
            if (entry.is_string() && NamesAGguf(entry.get<std::string>())) {
                named.push_back(entry.get<std::string>());
            }
        }
    }
    if (named.size() == 1) {
        auto path = root / named.front();
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(
                "Gemma 4 rai: the catalog entry names " + named.front() +
                " but " + path.string() + " does not exist");
        }
        return path;
    }
    if (named.size() > 1) {
        std::string list;
        for (const auto& name : named) list += (list.empty() ? "" : ", ") + name;
        throw std::runtime_error(
            "Gemma 4 rai: the catalog entry names " +
            std::to_string(named.size()) + " GGUFs (" + list +
            "); it must name exactly one");
    }

    // No catalog entry to go on: scan.
    std::vector<std::filesystem::path> found;
    std::error_code failed;
    for (const auto& entry : std::filesystem::directory_iterator(root, failed)) {
        if (entry.is_regular_file(failed) &&
            NamesAGguf(entry.path().filename().string())) {
            found.push_back(entry.path());
        }
    }
    if (found.size() == 1) return found.front();
    throw std::runtime_error(
        "Gemma 4 rai: the catalog entry names no GGUF and " + root.string() +
        " holds " + std::to_string(found.size()) +
        " of them; it must hold exactly one");
}

/// \brief Gemma 4 on ryzenai-corelib, GGUF weights
class RaiBackend final : public flm::backend::ModelBackend {
public:
    explicit RaiBackend(const flm::backend::BackendContext& context) {
        // 1. Refuse what this backend cannot do, FIRST -- before a file is
        //    opened or a runtime acquired. Both are also BackendTraits, so
        //    AutoModel rejects them earlier still; these are the second line,
        //    for a caller that reaches the factory directly.
        if (!context.config) {
            throw std::runtime_error("Gemma 4 rai backend needs an LM_Config");
        }
        if (context.enable_preemption) {
            throw std::invalid_argument(
                "Gemma 4 rai does not support preemption");
        }
        if (!ContextFits(context.context_length)) {
            throw std::out_of_range(
                "Gemma 4 rai context length must be in 1..4096, or a multiple of 4096 up to " +
                std::to_string(kMaxPagedContext));
        }
        context_length_ = context.context_length;

        // 2-3. Read and cross-validate EVERY source of truth before the
        //      runtime is acquired or the engine built, so a mismatched
        //      package fails at load with an exact diagnostic and nothing
        //      allocated -- never mid-generation.
        const std::filesystem::path root(context.model_path);
        const auto config = ReadJson(root / "config.json");
        const auto tokenizer_json = ReadJson(root / "tokenizer.json");
        // Phi-4's backend takes tokenizer_config.json from the frontend,
        // which has already parsed it. Gemma4e cannot do that: its
        // load_model calls _shared_load_backend BEFORE setup_tokenizer, so
        // there is nothing parsed to hand down at the moment the factory
        // runs. Read it here when it is not supplied, and use the frontend's
        // copy when it is, so the file is opened once per load either way.
        const auto tokenizer_config =
            context.tokenizer_config ? *context.tokenizer_config
                                     : ReadJson(root / "tokenizer_config.json");
        auto package =
            Gemma4GgufPackage::Open(RequireGgufPath(context.model_info, root));
        package->ValidateGemma4Contract(config, tokenizer_json,
                                        tokenizer_config);

        // Both ids come from the GGUF, cross-checked by the contract above.
        // Generation stops on the GGUF's eos (106, <turn|>) and on every other
        // id config.json lists (1, <eos>).
        forced_eos_ids_.clear();
        for (const auto id : ConfigEosIds(config, package->Config().eos_token_id))
            forced_eos_ids_.push_back(static_cast<int>(id));
        forced_bos_id_ = package->Config().bos_token_id;

        // 4-6. Only now: the runtime, the engine, and a cleared context.
        runtime_ = corelib::CorelibRuntime::GetOrCreate(
            utils::get_executable_directory());
        auto engine = std::make_unique<gemma4_rai>(
            std::move(package), runtime_, *context.config,
            context.context_length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~RaiBackend() override {
        // The engine holds its own reference to the runtime, but destroy it
        // here anyway: corelib objects must not outlive the API they came
        // from, and an engine released after its runtime is the case that
        // access-violates.
        //
        // NOT AN ENUMERATED TEARDOWN, and the difference matters. A
        // destructor BODY runs before member destruction, so a list like
        // "engine_.reset(); runtime_.reset();" would release the runtime
        // ahead of any member the list forgot. What actually holds the order
        // is the DECLARATION ORDER below -- runtime_ first, so it is
        // destroyed last. See task C9.
        engine_.reset();
    }

    causal_lm& engine() override { return *engine_; }

    std::string id() const override { return flm::backend::kRaiBackendId; }

    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string()
                        : std::string();
    }

    std::uint32_t max_decode_length() const override {
        return context_length_ > 0 ? context_length_ - 1 : 0;
    }

    bool supports_preemption() const override { return false; }

    /// \note corelib rejects a decode past its own limit, so the extra
    ///       forward() the FastFlowLM engines want after an EOS token would
    ///       fail here.
    bool forwards_past_eos() const override { return false; }

    bool poisoned() const noexcept override {
        return engine_ && engine_->poisoned();
    }

    std::optional<std::vector<int>> forced_eos_ids() const override {
        return forced_eos_ids_;
    }

    std::optional<int> forced_bos_id() const override { return forced_bos_id_; }

private:
    /// \brief read from the package at construction; see the constructor
    std::uint32_t context_length_{};
    std::vector<int> forced_eos_ids_;
    /// \brief read from the package at construction; see the constructor
    int forced_bos_id_{};
    // Declared before the engine so it is destroyed AFTER it.
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<gemma4_rai> engine_;
};

}  // namespace

flm::backend::BackendFactory rai_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<RaiBackend>(context);
    };
}

}  // namespace flm::gemma4
