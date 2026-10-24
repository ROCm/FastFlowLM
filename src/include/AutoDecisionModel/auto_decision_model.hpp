/// \file auto_decision_model.hpp
/// \brief AutoDecisionModel: System One decisions (`POST /v1/systemone`) from a d3 decision model.
/// \note  The engine (d3_npu, IRON) returns one final-norm hidden row per question; everything else
///        -- request checks, the prompt, tokenization, the 255-way readout and the answer objects --
///        is the d3 contract in src/common/decision, which reproduces the upstream package
///        (vllm-sr/d3-lite v3.1.3) byte for byte (src/test/d3_contract).
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "decision/d3_contract.hpp"
#include "decision_model.hpp"
#include "image/image_reader.hpp"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tokenizer/tokenizer.hpp"

extern std::unordered_set<std::string> decisionModelTags;

/// The reply to one HTTP request: status and JSON body.
struct DecisionReply {
    int status = 200;
    nlohmann::ordered_json body;
};

class AutoDecisionModel {
public:
    /// Prompts longer than this get `max_length_exceeded`, as upstream does; nothing is truncated.
    /// The upstream d3 models set no limit of their own (max_length null); time per question grows with
    /// the square of the length, so FLM stops at 32,768 tokens.
    static constexpr int MAX_INPUT_TOKENS = 32768;

    AutoDecisionModel(flm_rt::device* npu_device_inst, std::string model_tag);
    ~AutoDecisionModel();

    /// Load a package directory: config, tokenizer, decision_config.json, readout.safetensors,
    /// model.q4nx. Refuses (throws) a package whose answer codes, chat template or readout do not
    /// match the contract, like the upstream runtime.
    void load_model(const std::string& model_path, const nlohmann::json& model_info, bool enable_preemption);

    /// One `POST /v1/systemone` body -> reply (200, or 422 `{"detail": ...}`, as d3_server).
    DecisionReply system_one(const std::string& raw_body);

    /// The in-process path without HTTP checks of media: probabilities per runnable question.
    /// `payload` is the engine's image payload (d3_image_payload*) or nullptr for text.
    std::map<std::string, std::vector<double>> run(const flm::d3::Prepared& prepared, void* payload = nullptr);

    /// Encode a few prompts so the first request in each common row bucket runs at steady-state speed.
    void warmup();

    const std::string& model_name() const { return name_; }
    const std::string& model_tag() const { return tag_; }
    int max_input_tokens() const { return MAX_INPUT_TOKENS; }

private:
    flm_rt::device* npu_device_inst_ = nullptr;
    std::string tag_;
    std::string name_;             //!< the upstream model name in replies ("d3-lite", "d3-nano", "d3-mini"), set on load
    std::unique_ptr<npu_xclbin_manager> npu_;
    std::unique_ptr<LM_Config> config_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<decision_model> engine_;
    std::vector<std::string> codes_;
    std::vector<float> readout_;   //!< [255, hidden] fp32
    int hidden_ = 0;
    float temperature_ = 1.0f;
    std::mutex turn_;              //!< one request on the device at a time (d3_server.Decider)
    ImageReader image_reader_;     //!< decodes request images (PNG, JPEG, WebP)
    int dump_count_ = 0;           //!< D3_DUMP_PIXELS: requests dumped so far

    std::vector<int> encode(const std::string& text);
};
