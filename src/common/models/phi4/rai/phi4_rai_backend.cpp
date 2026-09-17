/// \file phi4_rai_backend.cpp
/// \brief The ryzenai-corelib backend for Phi-4
#include "models/phi4/rai/phi4_rai_backend.hpp"

#include "rai/corelib_runtime.hpp"
#include "models/phi4/rai/phi4_rai.hpp"
#include "models/phi4/rai/phi4_rai_gguf.hpp"
#include "utils/file_access.hpp"
#include "utils/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::phi4 {
namespace {

/// \brief the only GGUF this backend accepts; there is deliberately no alias
constexpr const char* kRaiGguf = "Phi-4-mini-instruct.Q8_0.gguf";

/// \brief the EOS ids ValidatePhi4Contract proves against the GGUF
/// \note These come from three independent sources agreeing, which is a
///       stronger guarantee than tokenizer_config.json alone.
const std::vector<int> kRaiEosIds = {200020, 199999};

/// \brief read a JSON file, recording the open for the file-access audit
/// \param path the file
/// \return the parsed document
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

/// \brief Phi-4 on ryzenai-corelib, GGUF weights
class RaiBackend final : public flm::backend::ModelBackend {
public:
    explicit RaiBackend(const flm::backend::BackendContext& context) {
        if (!context.config) {
            throw std::runtime_error("Phi-4 rai backend needs an LM_Config");
        }
        if (context.enable_preemption) {
            throw std::invalid_argument("Phi-4 rai does not support preemption");
        }
        if (context.context_length < 1 ||
            context.context_length > kRaiContextLimit) {
            throw std::out_of_range("Phi-4 rai context length must be in 1..4096");
        }

        // Read and validate every source of truth before acquiring the runtime
        // or creating the engine, so a mismatched package fails while nothing
        // has been allocated on the device.
        const std::filesystem::path root(context.model_path);
        const auto config = ReadJson(root / "config.json");
        const auto tokenizer_json = ReadJson(root / "tokenizer.json");
        // The frontend already parsed this one and passes it down, so the file
        // is opened once per load and the directory layout stays its knowledge.
        if (context.tokenizer_config == nullptr) {
            throw std::runtime_error(
                "Phi-4 rai backend needs the frontend's tokenizer_config.json");
        }
        auto package = Phi4GgufPackage::Open(root / kRaiGguf);
        package->ValidatePhi4Contract(config, tokenizer_json,
                                      *context.tokenizer_config);

        runtime_ = corelib::CorelibRuntime::GetOrCreate(
            utils::get_executable_directory());
        auto engine = std::make_unique<phi4_rai>(
            *context.config, std::move(package), runtime_,
            context.context_length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~RaiBackend() override {
        // The engine holds its own reference to the runtime, but destroy it
        // first anyway: corelib objects must not outlive the API they came from.
        engine_.reset();
    }

    causal_lm& engine() override { return *engine_; }

    std::string id() const override { return flm::backend::kRaiBackendId; }

    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string()
                        : std::string();
    }

    std::uint32_t max_decode_length() const override { return kRaiDecodeLimit; }

    bool supports_preemption() const override { return false; }

    /// \note corelib rejects a decode past its own limit, so the extra forward()
    ///       the FastFlowLM engines want after an EOS token would fail here.
    bool forwards_past_eos() const override { return false; }

    bool poisoned() const noexcept override {
        return engine_ && engine_->poisoned();
    }

    std::optional<std::vector<int>> forced_eos_ids() const override {
        return kRaiEosIds;
    }

private:
    // Declared first so it outlives the engine.
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<phi4_rai> engine_;
};

}  // namespace

flm::backend::BackendFactory rai_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<RaiBackend>(context);
    };
}

}  // namespace flm::phi4
