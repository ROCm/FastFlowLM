/// \file Qwen3VL.hpp
/// \brief Qwen3VL class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the Qwen3VL class

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"


#include "typedef.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

/************              Qwen3VL_4b            **************/
class Qwen3VL : public AutoModel {
private:

    void setup_tokenizer(std::string model_path);

    // Image processing functionality
    ImageReader image_reader_;

    int image_pre_resize = 0;

    int debug_count= 0;
    void smart_resize(
    int height, int width,
    int& h_bar,int& w_bar,
    int factor,
    int min_pixels,
    int max_pixels);

protected:
    static constexpr int IMAGE_SOFT_TOKEN_ID = 151655;

    qwen3vl_image_t load_image(const std::string& filename);
    qwen3vl_image_t load_image_base64(const std::string& base64_string);
    void preprocess_image(qwen3vl_image_t& image,  std::vector<bf16> &pixel_values);

    /// \brief Decode the images, apply the chat template, and expand every
    ///        image placeholder into its block of image soft tokens.
    /// \param input the uniform input, either `messages` (REST) or `prompt` (CLI)
    /// \param image_payload receives the preprocessed pixels for every image that
    ///        survived decoding, in prompt order
    /// \param tokens receives the prompt token ids
    /// \return false if there was nothing to prefill
    /// \note  Images that fail to decode or preprocess are dropped without
    ///        leaving a placeholder behind, so `image_payload` always stays
    ///        aligned with the image soft tokens in `tokens`.
    bool _build_vl_tokens(lm_uniform_input_t& input,
                          qwen3vl_image_payload_t& image_payload,
                          std::vector<int>& tokens);

    /// \brief Build the engine that backs this wrapper.
    /// \note  The Qwen3-VL checkpoint is served by two engines that share this
    ///        whole wrapper -- the tokenizer, chat template, sampler and image
    ///        preprocessing are identical -- and differ only in how they run
    ///        prefill on the NPU. This is the single seam between them.
    virtual void create_engine();

public:
    Qwen3VL(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);
    bool check_using_checkpint() {
		return false;
	}

    /// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	bool configure_parameter(std::string parameter_name, const std::any& value) override{
		if (parameter_name == "system_prompt") {
			try {
				this->user_system_prompt = std::any_cast<std::string>(value);
				this->extra_context["user_system_prompt"] = this->user_system_prompt;
				return true;
			} catch (const std::bad_any_cast&) {
				return false;
			}
		}
        else if (parameter_name == "img_pre_resize") {
            try {
                this->image_pre_resize = std::any_cast<int>(value);
                int target_size;
                if (this->image_pre_resize <= 0) {
                    target_size = 0;
                } else if (this->image_pre_resize == 1) {
                    target_size = 480;
                } else if (this->image_pre_resize <= 2) {
                    target_size = 720;
                } else if (this->image_pre_resize <= 3) {
                    target_size = 1080;
                } else if (this->image_pre_resize <= 4) {
                    target_size = 1440;
                } else if (this->image_pre_resize <= 5) {
                    target_size = 2160;
                } else if (this->image_pre_resize <= 6) {
                    target_size = 2880;
                } else if (this->image_pre_resize <= 7) {
                    target_size = 3240;
                } else if (this->image_pre_resize <= 8) {
                    target_size = 4320;
                } else {
                    this->image_pre_resize = 0;
                    target_size = 0;
                }
                if (this->image_pre_resize > 0) {
                    header_print_r("FLM", "Qwen3VL pre-resize image height to " + std::to_string(target_size) + " pixels if larger than that");
                }
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
		return false;
	}
};


/************              Qwen3VL_Flash            **************/
#if FLM_HAS_QWEN3VL_FLASH
/// Same checkpoint and same wrapper as Qwen3VL, backed by the qwen3vl_flash
/// engine: prefill runs on one fused overlay (6 dequant+mm columns + 1
/// attention CU) instead of swapping the array between mm.xclbin and
/// attn.xclbin every layer, and the vision encoder overlaps the text prefill
/// setup. Tuned for short contexts; Qwen3VL stays the general-purpose path.
///
/// Logically single-turn: every insert() starts from a clean kv state. The one
/// exception is the system prompt — if the request carries one, insert() pins
/// its shared prefix with a KV-cache checkpoint on the first occurrence, then
/// rewinds to that checkpoint on subsequent requests that carry the same system
/// text. This avoids re-prefilling the instruction on every call while keeping
/// the per-turn isolation that makes the flash engine fast.
class Qwen3VL_Flash : public Qwen3VL {
private:
    /// \brief system text the current checkpoint was built from, empty when none
    std::string pinned_system_text;

    /// \brief token history at the time of the checkpoint
    std::vector<int> system_his;

    /// \brief number of tokens covered by the checkpoint, 0 when nothing is pinned
    int system_tokens = 0;

    /// \brief prefill the system prefix of `system_text` and checkpoint it
    /// \return number of tokens pinned, 0 if the prefix could not be isolated
    int _pin_system_prefix(const std::string& system_text);

protected:
    void create_engine() override;

    /// \brief Drop everything the previous turn left behind, back to the pin.
    /// \note  Deliberately not AutoModel::clear_context(): that also resets the
    ///        TTFT and TOTAL profilers, which the caller starts around the whole
    ///        turn and would lose by calling insert().
    void _reset_turn();

public:
    Qwen3VL_Flash(flm_rt::device* npu_device_inst) : Qwen3VL(npu_device_inst) { single_turn = true; }

    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;

    /// \brief Apply the chat template without tools — qwen3vl_flash does not
    ///        support tool calling, so the tools argument is always dropped.
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override {
        nlohmann::ordered_json no_tools = nlohmann::ordered_json::object();
        return Qwen3VL::apply_chat_template(messages, no_tools);
    }
};
#endif


/************              Qwen3VL_Thinking            **************/
class Qwen3VL_Thinking : public Qwen3VL {
    private:
        int think_marker_id;
    
    public:
        Qwen3VL_Thinking(flm_rt::device* npu_device_inst) : Qwen3VL(npu_device_inst) {
    
        }
        std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
        std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    };
