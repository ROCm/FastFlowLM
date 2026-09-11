/// \file modeling_phi4.hpp
/// \brief Phi-4 frontend and backend routing
#pragma once
#include "AutoModel/automodel.hpp"

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include "corelib/corelib_runtime.hpp"
#endif

#if defined(FLM_CORELIB_TESTING)
#include <filesystem>
#include <functional>
namespace flm::phi4::testing { class Phi4FrontendTestAccess; }
#endif

class Phi4 : public AutoModel {
private:
    void setup_tokenizer(const std::string& model_path,
                         const nlohmann::json* verified_tokenizer_config = nullptr);
#if defined(FLM_ENABLE_CORELIB_AIE4)
    void validate_aie4_capacity(std::size_t rendered_tokens,
                               std::optional<int> requested) const;
    bool engine_is_poisoned() const noexcept;
    void clear_after_inference_failure(bool poisoned);
    std::string generate_aie4(chat_meta_info_t& meta_info,
                              std::ostream& os,
                              std::function<bool()> is_cancelled);

    bool uses_corelib_aie4_ = false;
    bool aie4_poisoned_ = false;
    int aie4_generation_budget_ = 0;
    std::shared_ptr<flm::corelib::CorelibRuntime> corelib_runtime_;
#endif

#if defined(FLM_CORELIB_TESTING)
    using EngineFactoryForTesting = std::function<std::unique_ptr<causal_lm>(
        bool, const LM_Config&, npu_xclbin_manager*,
        const std::filesystem::path&, std::uint32_t)>;
    static EngineFactoryForTesting engine_factory_for_testing_;
    static std::function<bool(const causal_lm*)> engine_poisoned_for_testing_;
    friend class flm::phi4::testing::Phi4FrontendTestAccess;
#endif

public:
    explicit Phi4(flm_rt::device* npu_device_inst);
    void load_model(std::string model_path, json model_info,
                    int default_context_length = -1,
                    bool enable_preemption = false) override;
    bool uses_corelib_aie4() const noexcept override {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        return uses_corelib_aie4_;
#else
        return false;
#endif
    }
    void clear_context() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit,
                         std::ostream& os,
                         std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info,
                                     lm_uniform_input_t& input,
                                     int length_limit,
                                     std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages,
                                    nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
};
