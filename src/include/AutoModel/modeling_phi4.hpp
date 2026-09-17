/// \file modeling_phi4.hpp
/// \brief Phi-4 frontend
/// \note Phi-4 runs on either of two engines -- FastFlowLM's own NPU kernels or
///       ryzenai-corelib -- but this class knows nothing about either.
///       Which engine to build, and every rule for driving it, lives behind
///       flm::backend::ModelBackend; see AutoModel/model_backend.hpp.
#pragma once
#include "AutoModel/automodel.hpp"

#if defined(FLM_CORELIB_TESTING)
namespace flm::phi4::testing { class Phi4FrontendTestAccess; }
#endif

class Phi4 : public AutoModel {
private:
#if defined(FLM_CORELIB_TESTING)
    /// \note The tokenizer contract is only observable from inside the class,
    ///       and the suite that checks it is not allowed to change it.
    friend class flm::phi4::testing::Phi4FrontendTestAccess;
#endif

    /// \brief Build the tokenizer, chat template and stop ids
    /// \param model_path the model directory
    /// \note Phi-4's contract differs from the shared one: minja receives no
    ///       textual BOS/EOS, and there is no automatic BOS token.
    /// \brief build the chat template and stop ids from an already-parsed
    ///        tokenizer_config.json
    /// \param config the parse load_model handed to the backend as well
    void setup_tokenizer(const nlohmann::json& config);

    /// \brief Turn a failed inference into a request error, clearing the session
    /// \throws ModelRequestError 500, always
    [[noreturn]] void fail_inference();

public:
    explicit Phi4(flm_rt::device* npu_device_inst);
    void load_model(std::string model_path, json model_info,
                    int default_context_length = -1,
                    bool enable_preemption = false, const std::string& backend = "") override;
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

    /// \note Phi-4 does not reuse a checkpointed prefix, so a matched prefix
    ///       still costs one round. Overrides the base default of true.
    bool check_using_checkpint() override { return false; }
};
