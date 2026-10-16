/// \file modeling_minicpm_v.hpp
/// \brief MiniCPM_V class
/// \author FastFlowLM Team
/// \date 2026-10-02
/// \version 0.9.28
/// \note This is a header file for the MiniCPM_V class
/// \note MiniCPM-V-4.7-1B's text tower is config-identical to Qwen3.5-0.8B and
///       shares its tokenizer ids, so the chat flow follows Qwen3_5VL.
/// \note Images (S7b): each picture is sliced the way MiniCPMV4ImageProcessorPil
///       slices it -- a source view plus up to max_slice_nums slices -- and each
///       VIEW becomes one minicpm_v_image_t in the payload. The prompt's single
///       <|image_pad|> per picture is expanded to the processor's placeholder
///       (<image_id>N</image_id><image>pad..</image><slice>pad..</slice>..). See
///       modeling_minicpm_v_image.cpp. Audio is dropped with a warning; video is
///       not supported.

#pragma once
#include "AutoModel/automodel.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"

#include <iostream>

/************              MiniCPM_V            **************/
class MiniCPM_V : public AutoModel {
private:

    bool enable_think = false;
    bool enable_tool = false;
    static constexpr int think_start_id = 248068;
    static constexpr int think_end_id = 248069;

    void setup_tokenizer(std::string model_path);

    // ---- images --------------------------------------------------------------
    /// \brief How one picture was sliced: what the prompt placeholder needs.
    struct image_layout {
        int views = 0;         ///< 1 + rows*cols
        int rows = 0, cols = 0;///< slice grid, 0 x 0 when unsliced
        int src_tokens = 0;    ///< LLM rows of the source view
        int slice_tokens = 0;  ///< LLM rows of each slice
        int tokens = 0;        ///< all image_token_id rows of the picture
        size_t values = 0;     ///< bf16 values it adds to _data__processed
    };
    ImageReader image_reader_;
    /// \brief Decode to uint8 CHW; false on failure.
    bool load_image(const std::string& source, bool base64, image_data_t& chw);
    /// \brief Slice, resize, normalise and patchify one picture into `payload`.
    image_layout preprocess_image(const image_data_t& chw, minicpm_v_image_payload_t& payload);
    /// \brief The token sequence `<|image_pad|>` expands to for picture `index`.
    std::vector<int> image_placeholder(const image_layout& l, int index);

    /// \brief Clear the conversation and throw after a failed inference
    [[noreturn]] void fail_inference();

    std::string decode(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled);

public:
    MiniCPM_V(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false, const std::string& backend = "") override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;

    bool thinking_enabled() const override {
        return this->enable_think;
    }

    /// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	bool configure_parameter(std::string parameter_name, const std::any& value) override{
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "reasoning_effort") {
            std::string reasoning_effort;
            try {
                reasoning_effort = std::any_cast<std::string>(value);
                if (reasoning_effort == "high" || reasoning_effort == "medium" || reasoning_effort == "low")
                    this->enable_think = true;
                else if (reasoning_effort == "none")
                    this->enable_think = false;
                else
                    header_print("WARNING", "Reasoning effort must be 'none', 'low', 'medium' or 'high'!");
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
		return AutoModel::configure_parameter(parameter_name, value);
	}
};
